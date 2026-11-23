"""A second, small DeepStream graph for still images, running next to the live one.

    appsrc (one 640x640 view) -> nvvideoconvert -> nvstreammux -> nvinfer SCRFD -> nvinfer ArcFace -> fakesink

Registration photos, group photos, camera captures and explanations go through the same two
models and the same nvinfer elements as the live video, so a face registered from a photo and the
same face seen on camera are embedded by identical code. Two kinds of request:

  detect      SCRFD finds faces in the view; ArcFace is skipped.
  embed_rois  SCRFD output is ignored; ArcFace embeds the given rectangles of the view.
              Used for one face (a crop around it) and for batches (a mosaic of 112x112 faces).
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
from .dsmeta import SKIP_CLASS, add_object, iter_list, read_embedding, read_tensors  # noqa: E402
from .runner import _caps, _make  # noqa: E402
from .scrfd import Face, decode  # noqa: E402

log = logging.getLogger("still-graph")

VIEW = 640  # SCRFD's input size: a view is fed to the network without any rescaling
PROBE_OK = 1  # Gst.PadProbeReturn.OK


class GraphNotReady(RuntimeError):
    pass


@dataclass
class _Request:
    mode: str  # "detect" | "rois"
    rects: tuple = ()
    min_px: int = 0


class StillGraph:
    def __init__(self, cfg: Config, engines_ready: threading.Event) -> None:
        self.cfg = cfg
        self._engines_ready = engines_ready  # set once the live graph has built/loaded the engines
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._built = threading.Event()
        self._failed = threading.Event()
        self._pipe: Optional[Gst.Pipeline] = None
        self._src: Optional[Gst.Element] = None
        self._req = _Request("detect")
        self._faces: list[Face] = []
        self._embs: list[Optional[np.ndarray]] = []
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
        src = _make(
            "appsrc", "still_src", is_live=True, format=Gst.Format.TIME, do_timestamp=True, block=False,
            max_bytes=4_000_000, caps=Gst.Caps.from_string(f"video/x-raw,format=BGR,width={VIEW},height={VIEW},framerate=30/1"),
        )
        conv = _make("nvvideoconvert", "still_conv")
        rgba = _caps("video/x-raw(memory:NVMM),format=RGBA")  # no chroma subsampling for stills
        mux = _make("nvstreammux", "still_mux", batch_size=1, width=VIEW, height=VIEW,
                    batched_push_timeout=40000, live_source=True)
        pgie = _make("nvinfer", "still_pgie", config_file_path=str(cfg.config_dir / "pgie_scrfd.txt"), unique_id=1)
        sgie = _make("nvinfer", "still_sgie", config_file_path=str(cfg.config_dir / "sgie_arcface.txt"), unique_id=2)
        sink = _make("fakesink", "still_sink", sync=False)
        for el in (src, conv, rgba, mux, pgie, sgie, sink):
            pipe.add(el)
        src.link(conv)
        conv.link(rgba)
        rgba.get_static_pad("src").link(mux.request_pad_simple("sink_0"))
        mux.link(pgie)
        pgie.link(sgie)
        sgie.link(sink)
        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_detect, None)
        sgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._on_result, None)
        if pipe.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("still-image graph refused to start")
        self._pipe, self._src = pipe, src

    # ------------------------------------------------------------------ probes
    def _on_detect(self, pad, info, _user=None):
        try:
            buf = info.get_buffer()
            bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            for fm in iter_list(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
                req = self._req
                fm.bInferDone = True
                if req.mode == "detect":
                    faces = decode(read_tensors(fm), VIEW, VIEW, self.cfg.det_thresh)
                    self._faces = [f for f in faces if f.size >= req.min_px]
                    for i, f in enumerate(self._faces):
                        om = add_object(bm, fm, f.box, f.score, label=str(i))
                        om.class_id = SKIP_CLASS  # ArcFace is not wanted here
                else:
                    for i, box in enumerate(req.rects):
                        add_object(bm, fm, box, 1.0, label=str(i))
        except Exception as exc:  # a probe must never raise into GStreamer
            log.exception("detect probe failed")
            self._error = str(exc)
        return PROBE_OK

    def _on_result(self, pad, info, _user=None):
        try:
            if self._req.mode == "rois":
                buf = info.get_buffer()
                bm = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
                embs: list[Optional[np.ndarray]] = [None] * len(self._req.rects)
                for fm in iter_list(bm.frame_meta_list, pyds.NvDsFrameMeta.cast):
                    for om in iter_list(fm.obj_meta_list, pyds.NvDsObjectMeta.cast):
                        i = int(om.obj_label)
                        if 0 <= i < len(embs):
                            embs[i] = read_embedding(om)
                self._embs = embs
        except Exception as exc:
            log.exception("result probe failed")
            self._error = str(exc)
        self._done.set()
        return PROBE_OK

    # ------------------------------------------------------------------ requests
    def _run(self, view: np.ndarray, req: _Request) -> None:
        if view.shape != (VIEW, VIEW, 3) or view.dtype != np.uint8:
            raise ValueError("a view must be a 640x640 BGR image")
        if not self.ready:
            raise GraphNotReady("The still-image graph is still starting (the first start builds TensorRT engines)")
        self._req, self._error = req, None
        self._faces, self._embs = [], []
        self._done.clear()
        self._src.emit("push-buffer", Gst.Buffer.new_wrapped(np.ascontiguousarray(view).tobytes()))
        if not self._done.wait(20):
            self._failed.set()  # the supervisor rebuilds the graph
            raise GraphNotReady("The still-image graph did not answer")
        if self._error:
            raise RuntimeError(self._error)

    def detect(self, view: np.ndarray, min_px: int = 12) -> list[Face]:
        """Faces in a 640x640 view, in view pixels."""
        with self._lock:
            self._run(view, _Request("detect", min_px=min_px))
            return list(self._faces)

    def embed_rois(self, view: np.ndarray, rects: list[tuple[float, float, float, float]]) -> list[Optional[np.ndarray]]:
        """ArcFace embeddings (L2-normalised) of rectangles x1,y1,x2,y2 of a view."""
        with self._lock:
            self._run(view, _Request("rois", rects=tuple(rects)))
            return list(self._embs)
