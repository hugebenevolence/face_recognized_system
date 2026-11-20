"use client";

import { useEffect, useState } from "react";
import { EventDetail, ExplainResult, RegionShare, api, dataUrl, fileUrl, fmtTime } from "@/lib/api";
import ImageWithBoxes from "./ImageWithBoxes";
import QualityChecks from "./QualityChecks";

function RegionBars({ regions }: { regions?: RegionShare[] }) {
  if (!regions || regions.every((r) => r.share === 0)) return <p className="muted">No standout region</p>;
  return (
    <div className="bars">
      {regions.map((r) => (
        <div className="bar-row" key={r.name}>
          <span className="bar-name">{r.name}</span>
          <div className="bar">
            <div style={{ width: `${Math.round(r.share * 100)}%` }} />
          </div>
          <span className="bar-val">{Math.round(r.share * 100)}%</span>
        </div>
      ))}
    </div>
  );
}

/** Base image with a transparent heat layer on top and an adjustable opacity. */
function HeatStack({ base, overlayB64, opacity }: { base: string; overlayB64?: string; opacity: number }) {
  return (
    <div className="heat">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={base} alt="" />
      {overlayB64 && (
        // eslint-disable-next-line @next/next/no-img-element
        <img className="heat-layer" src={dataUrl(overlayB64)} alt="" style={{ opacity }} />
      )}
    </div>
  );
}

/**
 * Why the system decided this for one recognition event:
 *  what it saw (box and landmarks and the aligned crop and quality)
 *  the decision (similarity of the closest people against the threshold)
 *  on demand: occlusion heatmaps showing which regions the scores depend on
 */
export default function EvidencePanel({ eventId }: { eventId: number }) {
  const [ev, setEv] = useState<EventDetail | null>(null);
  const [error, setError] = useState("");
  const [xai, setXai] = useState<ExplainResult | null>(null);
  const [xaiBusy, setXaiBusy] = useState(false);
  const [xaiError, setXaiError] = useState("");
  const [opacity, setOpacity] = useState(0.75);

  useEffect(() => {
    setEv(null);
    setXai(null);
    setError("");
    setXaiError("");
    api<EventDetail>(`/api/events/${eventId}`)
      .then(setEv)
      .catch((e) => setError((e as Error).message));
  }, [eventId]);

  const runExplain = async () => {
    setXaiBusy(true);
    setXaiError("");
    try {
      setXai(await api<ExplainResult>(`/api/events/${eventId}/explain`));
    } catch (e) {
      setXaiError((e as Error).message);
    } finally {
      setXaiBusy(false);
    }
  };

  if (error) return <p className="error">{error}</p>;
  if (!ev) return <p className="muted">Loading</p>;
  const evd = ev.evidence;
  if (!evd) return <p className="muted">No evidence stored for this event</p>;

  const known = ev.status === "known";
  const alignedUrl = fileUrl(evd.aligned_path);

  return (
    <div className="evidence">
      <div className="decision">
        <div>
          <strong>{ev.person_name ?? "Unknown"}</strong>{" "}
          <span className={`badge ${ev.status}`}>{ev.status}</span>
          <div className="meta muted">
            <span>{fmtTime(ev.ts)}</span>
            <span>{ev.camera_id}</span>
            <span>{evd.model}</span>
          </div>
        </div>
        <div className="verdict">
          <span className="verdict-score">{ev.confidence.toFixed(2)}</span>
          <span className="muted">Threshold {evd.threshold.toFixed(2)}</span>
        </div>
      </div>

      <h3>What the system saw</h3>
      <div className="seen">
        <div className="panel">
          <div className="cap">
            <strong>Detected face</strong>
            <div className="muted">Confidence {evd.det_score.toFixed(2)}</div>
          </div>
          <ImageWithBoxes
            frame
            src={fileUrl(ev.snapshot_path)}
            width={evd.crop_size[0]}
            height={evd.crop_size[1]}
            items={[{ id: 1, box: evd.bbox, kps: evd.kps, tone: known ? "good" : "bad" }]}
          />
          <div className="cap muted">Cyan dots mark the 5 landmarks</div>
        </div>
        <div className="panel">
          <div className="cap">
            <strong>Model input</strong>
            <div className="muted">Aligned 112 by 112</div>
          </div>
          <div className="frame">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={alignedUrl} alt="aligned face" />
          </div>
        </div>
      </div>
      <QualityChecks quality={evd.quality} />

      <h3>Decision</h3>
      {evd.top.length === 0 ? (
        <p className="muted">Nobody is registered for this model</p>
      ) : (
        <div className="bars">
          {evd.top.map((c) => (
            <div className="bar-row" key={c.person_id}>
              <span className="bar-name">
                {c.name} <span className="muted">{c.code}</span>
              </span>
              <div className="bar thr">
                <div
                  className={c.score >= evd.threshold ? "pass" : ""}
                  style={{ width: `${Math.max(0, Math.min(1, c.score)) * 100}%` }}
                />
                <i style={{ left: `${evd.threshold * 100}%` }} title={`Threshold ${evd.threshold}`} />
              </div>
              <span className="bar-val">{c.score.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
      {evd.margin !== null && <p className="muted small">Margin to runner up {evd.margin.toFixed(2)}</p>}

      <h3>Which regions matter</h3>
      {!xai && (
        <div>
          <button onClick={runExplain} disabled={xaiBusy || evd.top.length === 0}>
            {xaiBusy ? "Computing" : "Explain regions"}
          </button>
          <p className="muted small">Hides small patches of the face and measures how much the score drops</p>
          {xaiError && <p className="error">{xaiError}</p>}
        </div>
      )}
      {xai && (
        <div>
          <div className="row" style={{ marginBottom: 8 }}>
            <label style={{ margin: 0 }}>Overlay strength</label>
            <input
              type="range"
              min={0.2}
              max={1}
              step={0.05}
              value={opacity}
              onChange={(e) => setOpacity(+e.target.value)}
              style={{ width: 140 }}
            />
            <span className="legend">
              <i className="red" /> Hiding it lowers the score
            </span>
            <span className="legend">
              <i className="blue" /> Hiding it raises the score
            </span>
          </div>
          <div className="seen">
            <div className="panel">
              <div className="cap">
                <strong>Recognition</strong>
                {xai.recognition ? (
                  <div className="muted">
                    Against {xai.target.name ?? "unknown person"}{" "}
                    {xai.target.matched ? "" : "closest below threshold"}
                  </div>
                ) : (
                  <div className="muted">Not available for this model</div>
                )}
              </div>
              {xai.recognition && (
                <>
                  <HeatStack
                    base={alignedUrl}
                    overlayB64={xai.recognition.overlay_png_b64}
                    opacity={opacity}
                  />
                  <RegionBars regions={xai.recognition.regions} />
                  <div className="cap muted note">
                    Base score {xai.recognition.base_similarity.toFixed(2)}. Largest drop from one patch{" "}
                    {xai.recognition.max_drop.toFixed(2)}
                  </div>
                </>
              )}
            </div>
            <div className="panel">
              <div className="cap">
                <strong>Face detection</strong>
                <div className="muted">
                  {xai.detection ? `Base score ${xai.detection.base_score.toFixed(2)}` : "Not available"}
                </div>
              </div>
              {xai.detection?.overlay_png_b64 ? (
                <>
                  <HeatStack
                    base={fileUrl(ev.snapshot_path)}
                    overlayB64={xai.detection.overlay_png_b64}
                    opacity={opacity}
                  />
                  <RegionBars regions={xai.detection.regions} />
                  <div className="cap muted note">
                    {xai.detection.note ||
                      `Largest drop from one patch ${(xai.detection.max_drop ?? 0).toFixed(2)}`}
                  </div>
                </>
              ) : (
                <p className="muted">{xai.detection?.note || "No data"}</p>
              )}
            </div>
          </div>
          <p className="muted small">
            Shows model sensitivity and not human reasoning. Computed in {(xai.elapsed_ms / 1000).toFixed(1)}{" "}
            s
          </p>
        </div>
      )}
    </div>
  );
}
