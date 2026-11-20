"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import CameraPicker from "@/components/CameraPicker";
import EvidencePanel from "@/components/EvidencePanel";
import { API, EventPage, PipelineStatus, RecEvent, WS_URL, api, fileUrl, fmtTime } from "@/lib/api";

const MAX_FEED = 50;

export default function LivePage() {
  const [events, setEvents] = useState<RecEvent[]>([]);
  const [freshId, setFreshId] = useState<number | null>(null);
  const [inspectId, setInspectId] = useState<number | null>(null); // event shown in the inspector
  const [overlayLocal, setOverlayLocal] = useState<boolean | null>(null); // optimistic toggle state
  const [overlayError, setOverlayError] = useState("");
  const [wsUp, setWsUp] = useState(false);
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [streamKey, setStreamKey] = useState(0);

  // Backfill the feed with recent events, then stream new ones over WebSocket.
  useEffect(() => {
    api<EventPage>("/api/events?page_size=20")
      .then((p) => setEvents(p.items))
      .catch(() => {});
  }, []);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout>;
    let closed = false;

    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => setWsUp(true);
      ws.onmessage = (m) => {
        const ev: RecEvent = JSON.parse(m.data);
        setEvents((prev) => [ev, ...prev.filter((e) => e.id !== ev.id)].slice(0, MAX_FEED));
        setFreshId(ev.id);
      };
      ws.onclose = () => {
        setWsUp(false);
        if (!closed) timer = setTimeout(connect, 2000);
      };
    };
    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      ws?.close();
    };
  }, []);

  const poll = useCallback(
    () =>
      api<PipelineStatus>("/api/status")
        .then(setStatus)
        .catch(() => setStatus(null)),
    [],
  );
  useEffect(() => {
    poll();
    const t = setInterval(poll, 3000);
    return () => clearInterval(t);
  }, [poll]);

  // After switching camera, reopen the MJPEG <img> and refresh the status promptly.
  const onSourceChanged = useCallback(() => {
    setStreamKey((k) => k + 1);
    setTimeout(poll, 800);
  }, [poll]);

  const overlayOn = overlayLocal ?? status?.explain_overlay ?? false;
  const toggleOverlay = async (next: boolean) => {
    setOverlayLocal(next);
    setOverlayError("");
    try {
      await api("/api/overlay", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ explain: next }),
      });
    } catch (e) {
      setOverlayLocal(!next);
      setOverlayError((e as Error).message);
    }
  };

  const wasConnected = useRef(false);
  useEffect(() => {
    // An <img> MJPEG stream dies silently if the pipeline restarts; re-open it on reconnect.
    if (status?.connected && !wasConnected.current) setStreamKey((k) => k + 1);
    wasConnected.current = !!status?.connected;
  }, [status?.connected]);

  return (
    <>
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 16 }}>
        <h1 style={{ margin: 0 }}>Live recognition</h1>
        <div className="muted">
          <span className={`dot ${status?.connected ? "on" : ""}`} />
          {status?.connected
            ? `${status.camera_id} ${status.fps} fps ${status.model}`
            : status?.pipeline_up === false || !status
              ? "pipeline offline"
              : (status.error ?? "waiting for camera")}
          <span style={{ marginLeft: 16 }} />
          <span className={`dot ${wsUp ? "on" : ""}`} />
          Events {wsUp ? "live" : "reconnecting"}
        </div>
      </div>

      <CameraPicker status={status} onChanged={onSourceChanged} />

      <div
        className="grid2"
        style={{
          marginTop: 16,
          ...(inspectId !== null ? { gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1fr)" } : {}),
        }}
      >
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 10 }}>
            <label className="switch" title="Draw landmarks and scores on the video">
              <input type="checkbox" checked={overlayOn} onChange={(e) => toggleOverlay(e.target.checked)} />
              <span className="slider" />
              Explain overlay
            </label>
          </div>
          {overlayError && <p className="error">{overlayError}</p>}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img key={streamKey} className="video" src={`${API}/api/stream.mjpg`} alt="Live stream" />
        </div>

        <div className="side">
          {inspectId !== null && (
            <div className="card inspector">
              <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
                <h2 style={{ margin: 0 }}>Event {inspectId}</h2>
                <button className="ghost" onClick={() => setInspectId(null)}>
                  Close
                </button>
              </div>
              <EvidencePanel eventId={inspectId} />
            </div>
          )}
          <div className="card">
            <h2>Recognition events</h2>
            <div className="feed">
              {events.length === 0 && <span className="muted">No events yet</span>}
              {events.map((e) => (
                <div
                  key={e.id}
                  role="button"
                  tabIndex={0}
                  onClick={() => setInspectId(e.id)}
                  onKeyDown={(k) => (k.key === "Enter" || k.key === " ") && setInspectId(e.id)}
                  className={`feed-item clickable ${e.id === freshId ? "fresh" : ""} ${e.id === inspectId ? "active" : ""}`}
                >
                  {e.snapshot_path ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img className="thumb thumb-lg" src={fileUrl(e.snapshot_path)} alt="" />
                  ) : (
                    <div className="thumb thumb-lg" />
                  )}
                  <div style={{ minWidth: 0 }}>
                    <div>
                      <strong>{e.person_name ?? "Unknown"}</strong>{" "}
                      <span className={`badge ${e.status}`}>{e.status}</span>
                    </div>
                    <div className="meta muted">
                      {e.person_code && <span>{e.person_code}</span>}
                      <span>{(e.confidence * 100).toFixed(0)}%</span>
                      <span>{fmtTime(e.ts)}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
