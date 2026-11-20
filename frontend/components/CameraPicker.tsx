"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { CAMERA_WS_URL, PipelineStatus, SourcePreset, SourcesResponse, api } from "@/lib/api";

const SEND_INTERVAL_MS = 100; // ~10 fps is plenty for face recognition
const MAX_WIDTH = 960;
const MAX_BUFFERED = 512 * 1024; // skip frames instead of building latency when the link is slow
const JSON_HEADERS = { "Content-Type": "application/json" };

/**
 * Chooses what the pipeline analyses:
 *  - server sources (RTSP presets, video files, custom RTSP URL): the pipeline pulls them itself
 *  - cameras of *this* device: the browser lists them (getUserMedia) and streams JPEG frames
 *    to the pipeline over a WebSocket. Needed because the Docker containers cannot see the
 *    host's webcams.
 */
export default function CameraPicker({
  status,
  onChanged,
}: {
  status: PipelineStatus | null;
  onChanged: () => void;
}) {
  const [presets, setPresets] = useState<SourcePreset[]>([]);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [choice, setChoice] = useState("");
  const [custom, setCustom] = useState("rtsp://");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [streaming, setStreaming] = useState<string | null>(null);
  const [note, setNote] = useState("");

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const initialised = useRef(false);

  const stopBrowser = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    wsRef.current?.close();
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStreaming(null);
  }, []);

  useEffect(() => stopBrowser, [stopBrowser]); // stop sending when leaving the page

  const loadSources = useCallback(async () => {
    try {
      const res = await api<SourcesResponse>("/api/sources");
      setPresets(res.presets);
      return res;
    } catch {
      return null;
    }
  }, []);

  const loadDevices = useCallback(async (askPermission: boolean) => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setNote("Camera access needs HTTPS or localhost");
      return;
    }
    try {
      if (askPermission) {
        // Labels are blank until the user grants permission once.
        const s = await navigator.mediaDevices.getUserMedia({ video: true });
        s.getTracks().forEach((t) => t.stop());
      }
      const all = await navigator.mediaDevices.enumerateDevices();
      setDevices(all.filter((d) => d.kind === "videoinput" && d.deviceId));
      setNote("");
    } catch (e) {
      setNote(`Camera access denied ${(e as Error).message}`);
    }
  }, []);

  useEffect(() => {
    loadSources();
    loadDevices(false);
    const onChange = () => loadDevices(false);
    navigator.mediaDevices?.addEventListener?.("devicechange", onChange);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", onChange);
  }, [loadSources, loadDevices]);

  // Preselect whatever the pipeline is currently using (once).
  useEffect(() => {
    if (initialised.current || !status || presets.length === 0) return;
    initialised.current = true;
    const match = presets.find((p) => p.uri.split("@").pop() === status.source);
    if (match) setChoice(`src:${match.id}`);
    else if (status.source_kind === "browser") setNote("A browser camera is active in another tab");
    else setChoice("custom");
  }, [status, presets]);

  const startBrowserCamera = async (deviceId: string, label: string) => {
    stopBrowser();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    streamRef.current = stream;
    // Tell the pipeline to expect browser frames *before* sending any (it ignores them otherwise).
    await api("/api/source", {
      method: "PUT",
      headers: JSON_HEADERS,
      body: JSON.stringify({ kind: "browser" }),
    });

    const ws = new WebSocket(CAMERA_WS_URL);
    await new Promise<void>((resolve, reject) => {
      ws.onopen = () => resolve();
      ws.onerror = () => reject(new Error("Cannot open the camera connection"));
    });
    ws.onclose = () => {
      if (wsRef.current === ws) stopBrowser();
    };
    wsRef.current = ws;

    const video = videoRef.current!;
    video.srcObject = stream;
    await video.play();
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d")!;
    timerRef.current = setInterval(() => {
      if (ws.readyState !== WebSocket.OPEN || ws.bufferedAmount > MAX_BUFFERED || !video.videoWidth) return;
      const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((b) => b && ws.readyState === WebSocket.OPEN && ws.send(b), "image/jpeg", 0.8);
    }, SEND_INTERVAL_MS);
    setStreaming(label);
  };

  const apply = async () => {
    setBusy(true);
    setError("");
    try {
      if (choice.startsWith("cam:")) {
        const id = choice.slice(4);
        const dev = devices.find((d) => d.deviceId === id);
        await startBrowserCamera(id, dev?.label || "camera");
      } else {
        const uri = choice === "custom" ? custom.trim() : presets.find((p) => `src:${p.id}` === choice)?.uri;
        if (!uri) throw new Error("Pick a camera first");
        stopBrowser();
        await api("/api/source", {
          method: "PUT",
          headers: JSON_HEADERS,
          body: JSON.stringify({ kind: "uri", uri }),
        });
      }
      onChanged();
    } catch (e) {
      stopBrowser();
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const noLabels = devices.length > 0 && devices.every((d) => !d.label);

  return (
    <div className="card picker">
      <div className="picker-row">
        <label htmlFor="camera-select">Camera</label>
        <select id="camera-select" value={choice} onChange={(e) => setChoice(e.target.value)}>
          <option value="" disabled>
            Select a camera
          </option>
          <optgroup label="This device">
            {devices.map((d, i) => (
              <option key={d.deviceId} value={`cam:${d.deviceId}`}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
            {devices.length === 0 && (
              <option disabled value="">
                No cameras found
              </option>
            )}
          </optgroup>
          <optgroup label="Server sources">
            {presets.map((p) => (
              <option key={p.id} value={`src:${p.id}`}>
                {p.label}
              </option>
            ))}
            <option value="custom">Custom RTSP URL</option>
          </optgroup>
        </select>
        {choice === "custom" && (
          <input
            className="picker-url"
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
            placeholder="rtsp://user:pass@host:554/stream"
          />
        )}
        <button onClick={apply} disabled={busy || !choice}>
          {busy ? "Switching" : "Use this camera"}
        </button>
        <button
          className="ghost"
          onClick={() => loadDevices(true)}
          title="Ask the browser for camera permission"
        >
          Detect cameras
        </button>
      </div>

      {streaming && (
        <p className="muted picker-msg">
          <span className="dot on" />
          Streaming <strong>{streaming}</strong> from this browser. Keep this tab open
        </p>
      )}
      {noLabels && (
        <p className="muted picker-msg">Camera names appear after you allow access with Detect cameras</p>
      )}
      {note && <p className="muted picker-msg">{note}</p>}
      {error && <p className="error picker-msg">{error}</p>}
      {/* Off-screen sink for the camera stream; frames are copied from it to a canvas. */}
      <video ref={videoRef} muted playsInline style={{ display: "none" }} />
    </div>
  );
}
