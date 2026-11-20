"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Quality, api } from "@/lib/api";
import Modal from "./Modal";
import QualityChecks from "./QualityChecks";

type Probe = { faces: number; box?: number[]; score?: number; quality?: Quality };
type Compare = {
  model: string;
  threshold: number;
  items: { ok: boolean; yaw_ratio?: number; quality?: string }[];
  matrix: (number | null)[][];
};
type Shot = { id: number; file: File; url: string; label: string };
type Step = { key: string; label: string; hint: string; gate?: (yaw: number) => boolean };

// yaw_ratio > 0: nose is right of the eyes' midpoint in the camera image = the person turned to
// *their* left (the preview is mirrored, so it also looks like turning left on screen).
const STEPS: Step[] = [
  {
    key: "front",
    label: "Look straight",
    hint: "Look straight at the camera",
    gate: (y) => Math.abs(y) < 0.15,
  },
  { key: "left", label: "Turn left", hint: "Slowly turn your head to your left", gate: (y) => y > 0.18 },
  { key: "right", label: "Turn right", hint: "Slowly turn your head to your right", gate: (y) => y < -0.18 },
  { key: "up", label: "Tilt up", hint: "Raise your chin slightly and keep looking at the camera" },
  { key: "down", label: "Tilt down", hint: "Lower your chin slightly and keep looking at the camera" },
];

const PROBE_MS = 700;
const AUTO_FRAMES = 3; // consecutive good probes before auto-capture

/**
 * Take registration photos with a camera of this device, in a dialog.
 *  - free mode: take as many shots as you like
 *  - guided mode: 5 head angles, with live feedback on face / pose / light, optional
 *    auto-capture, and at the end a measurement of how much the face changes with angle
 *    (so you can see whether extra angles are really needed).
 */
export default function CameraCaptureModal({
  onDone,
  onClose,
}: {
  onDone: (files: File[]) => void;
  onClose: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [ready, setReady] = useState(false);
  const [error, setError] = useState("");

  const [mode, setMode] = useState<"free" | "guided">("guided");
  const [shots, setShots] = useState<Shot[]>([]);
  const nextId = useRef(1);
  const [probe, setProbe] = useState<Probe | null>(null);
  const [auto, setAuto] = useState(true);
  const [flash, setFlash] = useState(false);
  const [cmp, setCmp] = useState<Compare | null>(null);
  const [cmpBusy, setCmpBusy] = useState(false);
  const [cmpError, setCmpError] = useState("");

  const stepIdx = Math.min(shots.length, STEPS.length);
  const guidedDone = mode === "guided" && shots.length >= STEPS.length;
  const step = mode === "guided" && !guidedDone ? STEPS[stepIdx] : null;

  // ---------------------------------------------------------------- camera lifecycle
  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setReady(false);
  }, []);

  const startStream = useCallback(
    async (id: string) => {
      stopStream();
      setError("");
      try {
        if (!navigator.mediaDevices?.getUserMedia) throw new Error("Camera access needs HTTPS or localhost");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            ...(id ? { deviceId: { exact: id } } : {}),
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
        streamRef.current = stream;
        const v = videoRef.current!;
        v.srcObject = stream;
        await v.play();
        setReady(true);
        const all = await navigator.mediaDevices.enumerateDevices(); // labels exist once permission is granted
        setDevices(all.filter((d) => d.kind === "videoinput" && d.deviceId));
        if (!id) setDeviceId(stream.getVideoTracks()[0]?.getSettings().deviceId ?? "");
      } catch (e) {
        setError(`Cannot open the camera ${(e as Error).message}`);
      }
    },
    [stopStream],
  );

  useEffect(() => {
    startStream("");
    return stopStream;
  }, [startStream, stopStream]);

  // ---------------------------------------------------------------- frame helpers
  const grab = useCallback((maxWidth: number, quality: number): Promise<Blob | null> => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return Promise.resolve(null);
    const scale = Math.min(1, maxWidth / v.videoWidth);
    const c = document.createElement("canvas");
    c.width = Math.round(v.videoWidth * scale);
    c.height = Math.round(v.videoHeight * scale);
    c.getContext("2d")!.drawImage(v, 0, 0, c.width, c.height); // un-mirrored, as the model should see it
    return new Promise((res) => c.toBlob(res, "image/jpeg", quality));
  }, []);

  const capture = useCallback(
    async (label: string) => {
      const blob = await grab(1280, 0.92);
      if (!blob) return;
      const file = new File([blob], `camera-${Date.now()}.jpg`, { type: "image/jpeg" });
      setShots((s) => [...s, { id: nextId.current++, file, url: URL.createObjectURL(file), label }]);
      setCmp(null);
      setFlash(true);
      setTimeout(() => setFlash(false), 150);
    },
    [grab],
  );

  // ---------------------------------------------------------------- live feedback (probe)
  const goodStreak = useRef(0);
  const probing = useRef(false);
  const captureRef = useRef(capture);
  captureRef.current = capture;

  useEffect(() => {
    if (!ready) return;
    const timer = setInterval(async () => {
      if (probing.current) return;
      probing.current = true;
      try {
        const blob = await grab(640, 0.7);
        if (!blob) return;
        const body = new FormData();
        body.set("file", new File([blob], "frame.jpg", { type: "image/jpeg" }));
        const p = await api<Probe>("/api/faces/probe", { method: "POST", body });
        setProbe(p);

        const yaw = p.quality?.metrics.yaw_ratio ?? 0;
        const usable = p.faces === 1 && p.quality && p.quality.overall !== "fail";
        const posed = !step?.gate || step.gate(yaw);
        if (auto && step?.gate && usable && posed) {
          goodStreak.current += 1;
          if (goodStreak.current >= AUTO_FRAMES) {
            goodStreak.current = 0;
            await captureRef.current(step.label);
          }
        } else goodStreak.current = 0;
      } catch {
        setProbe(null);
      } finally {
        probing.current = false;
      }
    }, PROBE_MS);
    return () => clearInterval(timer);
  }, [ready, grab, auto, step]);

  useEffect(() => {
    goodStreak.current = 0; // a new step starts from scratch
  }, [stepIdx, mode]);

  // free the preview URLs when the dialog goes away
  const shotsRef = useRef<Shot[]>([]);
  shotsRef.current = shots;
  useEffect(() => () => shotsRef.current.forEach((s) => URL.revokeObjectURL(s.url)), []);

  // ---------------------------------------------------------------- derived state
  const yaw = probe?.quality?.metrics.yaw_ratio;
  const usable = probe?.faces === 1 && !!probe.quality && probe.quality.overall !== "fail";
  const posed = !step?.gate || (yaw !== undefined && step.gate(yaw));
  const tone: "ok" | "warn" | "bad" = !usable ? "bad" : posed ? "ok" : "warn";

  const message = !ready
    ? "Opening camera"
    : !probe
      ? "Checking"
      : probe.faces === 0
        ? "No face found. Move to the centre"
        : probe.faces > 1
          ? "More than one face. Keep one person in view"
          : probe.quality?.overall === "fail"
            ? "Face too small. Move closer"
            : step && !posed
              ? step.hint
              : step
                ? auto && step.gate
                  ? "Pose is right. Hold still"
                  : "Pose is right. Press Capture"
                : "Ready. Press Capture";

  const canShoot = ready && usable && !guidedDone;

  const runCompare = async () => {
    setCmpBusy(true);
    setCmpError("");
    try {
      const body = new FormData();
      shots.forEach((s) => body.append("files", s.file));
      setCmp(await api<Compare>("/api/faces/compare", { method: "POST", body }));
    } catch (e) {
      setCmpError((e as Error).message);
    } finally {
      setCmpBusy(false);
    }
  };

  useEffect(() => {
    if (guidedDone && !cmp && !cmpBusy && !cmpError) runCompare();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [guidedDone]);

  const reset = () => {
    shots.forEach((s) => URL.revokeObjectURL(s.url));
    setShots([]);
    setCmp(null);
    setCmpError("");
  };

  const finish = () => {
    onDone(shots.map((s) => s.file));
    onClose();
  };

  // ---------------------------------------------------------------- verdict from the measurement
  const verdict = (() => {
    if (!cmp || !guidedDone) return null;
    const rows = STEPS.slice(1).map((s, i) => ({ label: s.label, sim: cmp.matrix[0]?.[i + 1] ?? null }));
    const known = rows.filter((r) => r.sim !== null) as { label: string; sim: number }[];
    if (known.length === 0) return { rows, text: "No angle could be measured", needed: true };
    const miss = known.filter((r) => r.sim < cmp.threshold);
    const min = Math.min(...known.map((r) => r.sim));
    if (miss.length > 0)
      return {
        rows,
        needed: true,
        text: `With only the straight photo ${miss.length} of ${known.length} angles fall below the threshold ${cmp.threshold.toFixed(2)} and would be treated as unknown. Keep all angles`,
      };
    return {
      rows,
      needed: min < cmp.threshold + 0.1,
      text:
        min < cmp.threshold + 0.1
          ? `All ${known.length} angles pass but only just. The lowest is ${min.toFixed(2)}. More angles make recognition steadier`
          : `The straight photo already recognizes all ${known.length} angles. The lowest is ${min.toFixed(2)} against threshold ${cmp.threshold.toFixed(2)}. Extra angles are optional`,
    };
  })();

  return (
    <Modal title="Capture face" onClose={onClose}>
      <div className="capture">
        <div className="panel cam-col">
          <div className={`cam ${tone}`}>
            <video ref={videoRef} muted playsInline />
            <svg viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet" className="guide">
              <ellipse cx="200" cy="140" rx="82" ry="108" />
            </svg>
            <div className={`cam-badge ${tone}`}>{message}</div>
            {flash && <div className="cam-flash" />}
          </div>
          {error && <p className="error">{error}</p>}

          <div className="row">
            <select
              value={deviceId}
              onChange={(e) => (setDeviceId(e.target.value), startStream(e.target.value))}
              style={{ flex: 1, minWidth: 0 }}
            >
              {devices.length === 0 && <option value="">Default camera</option>}
              {devices.map((d, i) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label || `Camera ${i + 1}`}
                </option>
              ))}
            </select>
            <button onClick={() => capture(step?.label ?? `Photo ${shots.length + 1}`)} disabled={!canShoot}>
              Capture
            </button>
          </div>
        </div>

        <div className="panel side-col">
          <div className="tabs" style={{ margin: 0 }}>
            <button
              type="button"
              className={mode === "guided" ? "active" : ""}
              onClick={() => (reset(), setMode("guided"))}
            >
              Guided 5 angles
            </button>
            <button
              type="button"
              className={mode === "free" ? "active" : ""}
              onClick={() => (reset(), setMode("free"))}
            >
              Free capture
            </button>
          </div>

          {mode === "guided" ? (
            <>
              <ol className="steps">
                {STEPS.map((s, i) => (
                  <li key={s.key} className={i < shots.length ? "done" : i === stepIdx ? "current" : ""}>
                    <span className="step-dot">{i + 1}</span>
                    <span>{s.label}</span>
                    {shots[i] && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img className="step-thumb" src={shots[i].url} alt="" />
                    )}
                  </li>
                ))}
              </ol>
              <label className="check" style={{ marginTop: 0 }}>
                <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} />
                Auto capture when the pose is right
              </label>
            </>
          ) : null}

          {probe?.quality && probe.faces === 1 && <QualityChecks quality={probe.quality} />}
        </div>
      </div>

      {mode === "free" && shots.length > 0 && (
        <div className="previews" style={{ marginTop: 12 }}>
          {shots.map((s) => (
            <div className="preview" key={s.id}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={s.url} alt="" />
              <button type="button" onClick={() => setShots((p) => p.filter((x) => x.id !== s.id))}>
                x
              </button>
            </div>
          ))}
        </div>
      )}

      {guidedDone && (
        <div className="experiment">
          <h3>Do we need several angles</h3>
          {cmpBusy && <p className="muted">Comparing angles</p>}
          {cmpError && <p className="error">{cmpError}</p>}
          {cmp && verdict && (
            <>
              <div className="bars">
                {verdict.rows.map((r) => (
                  <div className="bar-row" key={r.label}>
                    <span className="bar-name">{r.label}</span>
                    <div className="bar thr">
                      <div
                        className={r.sim !== null && r.sim >= cmp.threshold ? "pass" : ""}
                        style={{ width: `${Math.max(0, Math.min(1, r.sim ?? 0)) * 100}%` }}
                      />
                      <i style={{ left: `${cmp.threshold * 100}%` }} />
                    </div>
                    <span className="bar-val">{r.sim === null ? "None" : r.sim.toFixed(2)}</span>
                  </div>
                ))}
              </div>
              <p className="small muted">
                Similarity of each angle to the straight photo. Model {cmp.model}. The white line is the
                threshold {cmp.threshold.toFixed(2)}
              </p>
              <div className={verdict.needed ? "notice warn" : "notice"}>{verdict.text}</div>
            </>
          )}
        </div>
      )}

      <div className="row" style={{ justifyContent: "flex-end", marginTop: 16 }}>
        {shots.length > 0 && (
          <button className="ghost" onClick={reset}>
            Start over
          </button>
        )}
        <button className="ghost" onClick={onClose}>
          Cancel
        </button>
        <button onClick={finish} disabled={shots.length === 0}>
          Use {shots.length} {shots.length === 1 ? "photo" : "photos"}
        </button>
      </div>
    </Modal>
  );
}
