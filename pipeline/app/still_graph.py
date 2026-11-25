"""A second, small DeepStream graph for still images, running next to the live one.

    appsrc (JPEG) -> jpegparse -> nvjpegdec -> nvvideoconvert (crop + scale) --+
                                                                              +-> nvstreammux -> nvinfer SCRFD
    appsrc (raw RGB 640x640) -> nvvideoconvert -------------------------------+
        -> [probe: faces] -> nvdspreprocess (face alignment) -> nvinfer ArcFace -> fakesink

It is the live chain without the tracker: the same SCRFD and ArcFace engines, the same alignment
library, the same configs. A face registered from a photo is therefore detected, aligned and
embedded exactly like the same face seen on camera. Uploaded JPEGs are decoded, cropped and scaled
on the GPU; only pictures that are not plain JPEGs come in as raw pixels. Two kinds of request:

  detect  SCRFD finds the faces in the view (each one is also aligned and embedded).
  embed   the faces are given (boxes in view pixels). SCRFD still runs on the view, and the
          alignment library takes each face's landmarks from it, exactly as in the live graph.
          With prealigned=True the faces are already aligned 112x112 images (explanation maps):
          they are added as the "aligned face" class and only scaled, never re-aligned.
Every face comes back with its embedding and the aligned 112x112 face the library produced.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import gi
import numpy as np
import pyds

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from .config import Config  # noqa: E402
from .dsmeta import (  # noqa: E402
    ALIGNED_FACE_CLASS, FACE_CLASS, add_object, iter_list, read_aligned_face, read_embedding, read_tensors,
)
from .picture import VIEW, Picture, View  # noqa: E402
from .runner import _caps, _make  # noqa: E402
from .scrfd import Face, decode  # noqa: E402

log = logging.getLogger("still-graph")
PROBE_OK = 1  # Gst.PadProbeReturn.OK


class GraphNotReady(RuntimeError):
    pass


@dataclass
class StillFace:
    face: Face  # view pixels
    embedding: Optional[np.ndarray]
    aligned: Optional[np.ndarray]  # 112x112 RGB, what ArcFace saw


@dataclass
class _Request:
    mode: str  # "detect" | "embed"
    faces: tuple = ()  # embed: Face objects in view pixels
    min_px: int = 0
    prealigned: bool = False  # embed: the faces are already aligned 112x112 images


class StillGraph:
    def __init__(self, cfg: Config, engines_ready: threading.Event) -> None:
        self.cfg = cfg
        self._engines_ready = engines_ready  # set once the live graph has built/loaded the engines
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._built = threading.Event()
        self._failed = threading.Event()
        self._pipe: Optional[Gst.Pipeline] = None
        self._jpeg_src: Optional[Gst.Element] = None
        self._raw_src: Optional[Gst.Element] = None
        self._crop: Optional[Gst.Element] = None
        self._req = _Request("detect")
        self._faces: list[Face] = []
        self._results: dict[int, tuple] = {}
        self._error: Optional[str] = None
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        threading.Thread(target=self._supervise, name="still-graph", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if self._pipe is not None:
            self._pipe.set_state(Gst.State.NULL)

    @property
    def ready(self) -> bool:
        return self._built.is_set() and not self._failed.is_set()

    def _supervise(self) -> None:
        # The first start builds the TensorRT engines. Wait until the live graph has done that so
        # the two graphs never write the same engine file at once.
        self._engines_ready.wait(timeout=1800)
        while not self._stop.is_set():
            try:
                self._build()
            except Exception:
                log.exception("could not build the still-image graph")
                self._stop.wait(5)
                continue
            self._built.set()
            self._failed.clear()
            log.info("still-image graph ready")
            bus = self._pipe.get_bus()
            while not self._stop.is_set() and not self._failed.is_set():
                msg = bus.timed_pop_filtered(500 * Gst.MSECOND, Gst.MessageType.ERROR)
                if msg is not None:
                    err, dbg = msg.parse_error()
                    log.error("still-image graph error: %s (%s)", err.message, dbg)
                    self._failed.set()
            self._built.clear()
            self._pipe.set_state(Gst.State.NULL)

    def _build(self) -> None:
        cfg = self.cfg
        pipe = Gst.Pipeline.new("still-graph")
        rgba = f"video/x-raw(memory:NVMM),format=RGBA,width={VIEW},height={VIEW}"  # the alignment kernel reads RGBA
        # input 1: JPEG files, decoded, cropped and scaled on the GPU
        jpeg_src = _make("appsrc", "still_jpeg", is_live=True, format=Gst.Format.TIME, do_timestamp=True,
                         block=False, max_bytes=64_000_000, caps=Gst.Caps.from_string("image/jpeg"))
        jparse = _make("jpegparse", "still_jparse")
        jdec = _make("nvjpegdec", "still_jdec")
        crop = _make("nvvideoconvert", "still_crop")
        jcaps = _caps(rgba)
        # input 2: raw RGB views (pictures that are not plain JPEGs, explanation mosaics)
        raw_src = _make("appsrc", "still_raw", is_live=True, format=Gst.Format.TIME, do_timestamp=True,
                        block=False, max_bytes=8_000_000,
                        caps=Gst.Caps.from_string(f"video/x-raw,format=RGB,width={VIEW},height={VIEW},framerate=30/1"))
        rconv = _make("nvvideoconvert", "still_rconv")
        rcaps = _caps(rgba)

        mux = _make("nvstreammux", "still_mux", batch_size=1, width=VIEW, height=VIEW,
                    batched_push_timeout=40000, live_source=True)
        pgie = _make("nvinfer", "still_pgie", config_file_path=str(cfg.config_dir / "pgie_scrfd.txt"), unique_id=1)
        align = _make("nvdspreprocess", "still_align", config_file=str(cfg.config_dir / "preprocess_arcface.txt"))
        sgie = _make("nvinfer", "still_sgie", config_file_path=str(cfg.config_dir / "sgie_arcface.txt"),
                     unique_id=2, input_tensor_meta=True)
        sink = _make("fakesink", "still_sink", sync=False)
        for el in (jpeg_src, jparse, jdec, crop, jcaps, raw_src, rconv, rcaps, mux, pgie, align, sgie, sink):
            pipe.add(el)
        jpeg_src.link(jparse), jparse.link(jdec), jdec.link(crop), crop.link(jcaps)
        jcaps.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
        raw_src.link(rconv), rconv.link(rcaps)
        rcaps.get_static_pad("src").link(mux.request_pad_simple("sink_1"))
        mux.link(pgie), pgie.link(align), align.link(sgie), sgie.link(sink)
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_faces, None)
        sgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_result, None)
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("still-image graph refused to start")
        self._pipe, self._jpeg_src, self._raw_src, self._crop = pipe, jpeg_src, raw_src, crop

    # ------------------------------------------------------------------ probes
    def _on_faces(self, pad, info, _user=None):
        """After SCRFD: the faces of this request become objects."""
        try:
            buf = info.get_buffer()
            bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            for fm in iter_list(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
                req = self._req
                if req.mode == "detect":
                    faces = decode(read_tensors(fm), VIEW, VIEW, self.cfg.det_thresh)
                    faces = [f for f in faces if f.size >= req.min_px]
                else:
                    faces = list(req.faces)
                self._faces = faces
                cls = ALIGNED_FACE_CLASS if req.prealigned else FACE_CLASS
                for i, f in enumerate(faces):
                    add_object(bm, fm, f.box, f.score, label=str(i), frame_size=(VIEW, VIEW), class_id=cls)
        except Exception as exc:  # a probe must never raise into GStreamer
            log.exception("face probe failed")
            self._error = str(exc)
        return PROBE_OK

    def _on_result(self, pad, info, _user=None):
        try:
            buf = info.get_buffer()
            bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            results: dict[int, tuple] = {}
            for fm in iter_list(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
                for om in iter_list(fm.obj_meta_list, pyds.NvDsObjectMeta.cast):
                    results[int(om.obj_label)] = (read_embedding(om), read_aligned_face(om))
            self._results = results
        except Exception as exc:
            log.exception("result probe failed")
            self._error = str(exc)
        self._done.set()
        return PROBE_OK

    # ------------------------------------------------------------------ requests
    def _run(self, pic: Picture, view: View, req: _Request) -> list[StillFace]:
        if not self.ready:
            raise GraphNotReady("The still-image graph is still starting (the first start builds TensorRT engines)")
        self._req, self._error = req, None
        self._faces, self._results = [], {}
        self._done.clear()
        if pic.jpeg is not None:
            self._crop.set_property("src-crop", ":".join(map(str, view.src)))
            self._crop.set_property("dest-crop", ":".join(map(str, view.dst)))
            self._jpeg_src.emit("push-buffer", Gst.Buffer.new_wrapped(pic.jpeg))
        else:
            self._raw_src.emit("push-buffer", Gst.Buffer.new_wrapped(view.raw(pic).tobytes()))
        if not self._done.wait(20):
            self._failed.set()  # the supervisor rebuilds the graph
            raise GraphNotReady("The still-image graph did not answer")
        if self._error:
            raise RuntimeError(self._error)
        return [StillFace(f, *self._results.get(i, (None, None))) for i, f in enumerate(self._faces)]

    def detect(self, pic: Picture, view: View, min_px: int = 12) -> list[StillFace]:
        """Faces in a view of the picture (view pixels), each aligned and embedded."""
        with self._lock:
            return self._run(pic, view, _Request("detect", min_px=min_px))

    def embed(self, pic: Picture, view: View, faces: list[Face], prealigned: bool = False) -> list[StillFace]:
        """Embeddings and aligned faces of given faces of a view (boxes in view pixels)."""
        with self._lock:
            return self._run(pic, view, _Request("embed", faces=tuple(faces), prealigned=prealigned))
