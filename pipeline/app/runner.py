"""Builds and supervises the DeepStream graph.

    source -> nvstreammux -> nvinfer SCRFD -> nvtracker -> nvvideoconvert (RGBA)
           -> nvdspreprocess (face alignment) -> nvinfer ArcFace -> nvdsosd
           -> tee -+-> nvjpegenc -> appsink                (MJPEG for the browser)
                   +-> nvv4l2h264enc -> rtspclientsink     (optional RTSP out)

Detection, tracking, face alignment, embedding, drawing and encoding all run inside the graph on
the GPU. Python wires the elements and handles the metadata in three pad probes (analytics.py).
"""

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

from .analytics import Analytics  # noqa: E402
from .config import Config  # noqa: E402

log = logging.getLogger("runner")
# rtspsrc asks GIO for a proxy; libproxy throws an uncaught C++ exception inside the container
# ("Unable to read configuration") and aborts the process. Cameras are on the LAN anyway.
os.environ.setdefault("GIO_USE_PROXY_RESOLVER", "dummy")
Gst.init(None)

DS_LIB = "/opt/nvidia/deepstream/deepstream/lib"


@dataclass(frozen=True)
class SourceSpec:
    kind: str  # "uri": rtsp or file read by the graph | "browser": JPEG frames pushed over WebSocket
    uri: str = ""


def placeholder_jpeg(text: str) -> bytes:
    img = np.full((540, 960, 3), (32, 26, 24), np.uint8)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.putText(img, text, ((960 - w) // 2, (540 + h) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (170, 156, 150), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


def _make(factory: str, name: Optional[str] = None, **props) -> Gst.Element:
    el = Gst.ElementFactory.make(factory, name)
    if el is None:
        raise RuntimeError(f"GStreamer element '{factory}' is not available")
    for k, v in props.items():
        el.set_property(k.replace("_", "-"), v)
    return el


def _caps(s: str) -> Gst.Element:
    return _make("capsfilter", caps=Gst.Caps.from_string(s))


class Runner(threading.Thread):
    def __init__(self, cfg: Config) -> None:
        super().__init__(name="runner", daemon=True)
        self.cfg = cfg
        self.analytics = Analytics(cfg)
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._spec = SourceSpec("uri", cfg.source_uri)
        self._gen = 0
        self._switch = threading.Event()
        self._appsrc: Optional[Gst.Element] = None
        self._jpeg = placeholder_jpeg("Waiting for video stream")
        self._jpeg_seq = 0
        self._times: list[float] = []
        self.connected = False
        self.engines_ready = threading.Event()
        self.last_error: Optional[str] = None
        self.browser_clients = 0

    # ---------- API-thread accessors ----------
    @property
    def source(self) -> SourceSpec:
        return self._spec

    @property
    def explain_overlay(self) -> bool:
        return self.analytics.explain_overlay

    @explain_overlay.setter
    def explain_overlay(self, value: bool) -> None:
        self.analytics.explain_overlay = value

    def set_source(self, spec: SourceSpec) -> None:
        with self._lock:
            self._spec = spec
            self._gen += 1
            self.last_error = None
        self._switch.set()
        log.info("source switched to %s", spec)

    def push_jpeg(self, data: bytes) -> None:
        """Called by the /ingest WebSocket; ignored unless the browser source is running."""
        src = self._appsrc
        if src is not None and self._spec.kind == "browser":
            src.emit("push-buffer", Gst.Buffer.new_wrapped(data))

    def latest_jpeg(self) -> tuple[bytes, int]:
        with self._lock:
            return self._jpeg, self._jpeg_seq

    def latest_raw_jpeg(self) -> Optional[bytes]:
        raw = self.analytics.last_raw
        if raw is None:
            return None
        ok, buf = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        return buf.tobytes() if ok else None

    @property
    def fps(self) -> float:
        t = list(self._times)
        if len(t) < 2 or t[-1] <= t[0] or time.monotonic() - t[-1] > 3:
            return 0.0
        return round((len(t) - 1) / (t[-1] - t[0]), 1)

    def stop(self) -> None:
        self._stop_evt.set()
        self._switch.set()

    # ---------- supervisor ----------
    def run(self) -> None:
        self.analytics.publisher.start()
        while not self._stop_evt.is_set():
            self._switch.clear()
            with self._lock:
                spec, gen = self._spec, self._gen
            self.analytics.reset()
            try:
                self._play(spec, gen)
            except Exception as exc:  # keep the supervisor alive no matter what
                self.last_error = str(exc)
                log.exception("pipeline crashed")
            self.connected = False
            self._set_jpeg(placeholder_jpeg("Waiting for browser camera" if spec.kind == "browser" else "Reconnecting to video source"))
            self._switch.wait(3)  # backoff; also what loops a file at end of stream

    def _play(self, spec: SourceSpec, gen: int) -> None:
        pipe = self._build(spec)
        bus = pipe.get_bus()
        pipe.set_state(Gst.State.PLAYING)
        self.engines_ready.set()  # nvinfer has loaded (or built) the TensorRT engines by now
        started = time.monotonic()
        try:
            while not self._stop_evt.is_set() and self._gen == gen:
                msg = bus.timed_pop_filtered(500 * Gst.MSECOND, Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg is not None:
                    if msg.type == Gst.MessageType.ERROR:
                        err, dbg = msg.parse_error()
                        self.last_error = err.message
                        log.error("gst error: %s (%s)", err.message, dbg)
                    else:
                        log.info("end of stream")
                    return
                if not self.connected and time.monotonic() - started > 60 and spec.kind != "browser":
                    self.last_error = "no frames for 60s"
                    return
        finally:
            self._appsrc = None
            pipe.set_state(Gst.State.NULL)

    # ---------- graph ----------
    def _build(self, spec: SourceSpec) -> Gst.Pipeline:
        cfg, an = self.cfg, self.analytics
        pipe = Gst.Pipeline.new("face-pipeline")
        live = spec.kind == "browser" or spec.uri.startswith(("rtsp://", "rtsps://"))

        mux = _make("nvstreammux", "mux", batch_size=1, width=cfg.frame_width, height=cfg.frame_height,
                    batched_push_timeout=40000, live_source=live, enable_padding=True)
        pgie = _make("nvinfer", "pgie", config_file_path=str(cfg.config_dir / "pgie_scrfd.txt"), unique_id=1)
        tracker = _make("nvtracker", "tracker", ll_lib_file=f"{DS_LIB}/libnvds_nvmultiobjecttracker.so",
                        ll_config_file=str(cfg.config_dir / "tracker_iou.yml"), tracker_width=640, tracker_height=384)
        conv = _make("nvvideoconvert", "conv")
        rgba = _caps("video/x-raw(memory:NVMM),format=RGBA")  # the alignment kernel reads RGBA
        align = _make("nvdspreprocess", "align", config_file=str(cfg.config_dir / "preprocess_arcface.txt"))
        sgie = _make("nvinfer", "sgie", config_file_path=str(cfg.config_dir / "sgie_arcface.txt"), unique_id=2,
                     input_tensor_meta=True)
        osd = _make("nvdsosd", "osd", display_text=True)
        tee = _make("tee", "tee")

        # branch 1: JPEG frames for the browser
        q1 = _make("queue", "q_jpeg", leaky=2, max_size_buffers=2)
        c1 = _make("nvvideoconvert", "conv_jpeg")
        f1 = _caps("video/x-raw(memory:NVMM),format=I420")
        jpg = _make("nvjpegenc", "jpeg", quality=cfg.jpeg_quality)
        sink = _make("appsink", "sink", emit_signals=True, max_buffers=1, drop=True, sync=False)
        sink.connect("new-sample", self._on_sample)

        chain = [mux, _make("queue", "q_mux"), pgie, tracker, conv, rgba, align, sgie, osd, tee]
        for el in chain + [q1, c1, f1, jpg, sink]:
            pipe.add(el)
        for a, b in zip(chain, chain[1:]):
            a.link(b)
        tee.link(q1)
        q1.link(c1), c1.link(f1), f1.link(jpg), jpg.link(sink)

        if cfg.rtsp_out:
            self._add_rtsp_out(pipe, tee, cfg.rtsp_out)

        self._add_source(pipe, spec, mux)

        pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, an.on_detect, None)
        tracker.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, an.on_tracked, None)
        sgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, an.on_identify, None)
        return pipe

    def _add_rtsp_out(self, pipe: Gst.Pipeline, tee: Gst.Element, url: str) -> None:
        els = [
            _make("queue", "q_rtsp", leaky=2, max_size_buffers=2),
            _make("nvvideoconvert", "conv_rtsp"),
            _caps("video/x-raw(memory:NVMM),format=NV12"),
            _make("nvv4l2h264enc", "h264", bitrate=3_000_000, insert_sps_pps=True, idrinterval=30),
            _make("h264parse", "h264parse", config_interval=-1),
            _make("rtspclientsink", "rtsp_out", location=url, protocols=4),  # 4 = TCP
        ]
        for el in els:
            pipe.add(el)
        tee.link(els[0])
        for a, b in zip(els, els[1:]):
            a.link(b)

    def _add_source(self, pipe: Gst.Pipeline, spec: SourceSpec, mux: Gst.Element) -> None:
        pad = mux.request_pad_simple("sink_0")
        if spec.kind == "browser":
            src = _make("appsrc", "appsrc", is_live=True, format=Gst.Format.TIME, do_timestamp=True,
                        max_bytes=4_000_000, block=False, caps=Gst.Caps.from_string("image/jpeg,framerate=15/1"))
            chain = [src, _make("jpegparse"), _make("nvjpegdec"), _make("nvvideoconvert"),
                     _caps("video/x-raw(memory:NVMM),format=NV12")]
            for el in chain:
                pipe.add(el)
            for a, b in zip(chain, chain[1:]):
                a.link(b)
            chain[-1].get_static_pad("src").link(pad)
            self._appsrc = src
            return

        if spec.uri.startswith(("rtsp://", "rtsps://")):
            self._add_rtsp_source(pipe, spec.uri, pad)
            return

        uri = spec.uri if "://" in spec.uri else f"file://{spec.uri}"
        dec = _make("uridecodebin", "src", uri=uri)
        pipe.add(dec)
        target = pad
        if not uri.startswith(("rtsp://", "rtsps://")):
            # A file would otherwise be decoded as fast as the GPU allows; pace it to real time.
            pace = _make("identity", "pace", sync=True)
            pipe.add(pace)
            pace.get_static_pad("src").link(pad)
            target = pace.get_static_pad("sink")
        dec.connect("pad-added", self._on_pad_added, target)

    @staticmethod
    def _add_rtsp_source(pipe: Gst.Pipeline, url: str, mux_pad: Gst.Pad) -> None:
        """rtspsrc -> depay -> parse -> hardware decoder, wired explicitly (H.264 cameras)."""
        src = _make("rtspsrc", "src", location=url, latency=200, protocols=4)  # 4 = TCP
        depay = _make("rtph264depay")
        parse = _make("h264parse", config_interval=-1)  # re-send SPS/PPS so the decoder can start mid-stream
        dec = _make("nvv4l2decoder")
        for el in (src, depay, parse, dec):
            pipe.add(el)
        depay.link(parse)
        parse.link(dec)
        dec.get_static_pad("src").link(mux_pad)

        def on_pad(_src, pad):
            caps = pad.get_current_caps()
            if caps is not None and caps.get_structure(0).get_value("media") != "video":
                return
            sink = depay.get_static_pad("sink")
            if not sink.is_linked():
                pad.link(sink)

        src.connect("pad-added", on_pad)

    @staticmethod
    def _on_pad_added(_dec, pad, sink_pad) -> None:
        caps = pad.get_current_caps() or pad.query_caps(None)
        s = caps.get_structure(0)
        if not s.get_name().startswith("video"):
            return
        if not caps.get_features(0).contains("memory:NVMM"):
            log.error("decoder produced system-memory frames; the NVIDIA hardware decoder was not used")
            return
        if not sink_pad.is_linked():
            pad.link(sink_pad)

    # ---------- output ----------
    def _on_sample(self, sink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                self._set_jpeg(bytes(info.data))
            finally:
                buf.unmap(info)
            self.connected, self.last_error = True, None
            self._times.append(time.monotonic())
            del self._times[:-30]
        return Gst.FlowReturn.OK

    def _set_jpeg(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg, self._jpeg_seq = jpeg, self._jpeg_seq + 1
