"use client";

import { FormEvent, useEffect, useState } from "react";
import { Analysis, ApiError, FaceInfo, Person, api, dataUrl } from "@/lib/api";
import ImageWithBoxes, { BoxItem } from "./ImageWithBoxes";
import QualityChecks from "./QualityChecks";

const DUP_WARN = 0.5; // keep in step with the backend's DUP_THRESHOLD

/**
 * Register one person from a photo that may contain several people:
 * upload -> the system boxes every face -> click one (or drag your own box) -> review
 * what it detected -> fill in who it is -> register.
 */
export default function GroupPhotoRegister({ onRegistered }: { onRegistered: (p: Person) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [selected, setSelected] = useState<FaceInfo | null>(null);
  const [selectedId, setSelectedId] = useState<number | "custom" | null>(null);
  const [draft, setDraft] = useState<number[] | null>(null);
  const [drawMode, setDrawMode] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const [fullName, setFullName] = useState("");
  const [code, setCode] = useState("");
  const [notes, setNotes] = useState("");
  const [force, setForce] = useState(false);
  const [conflict, setConflict] = useState("");

  useEffect(
    () => () => {
      if (url) URL.revokeObjectURL(url);
    },
    [url],
  );

  const analyze = (f: File, box?: number[]) => {
    const body = new FormData();
    body.set("file", f);
    if (box) body.set("box", box.map((v) => v.toFixed(1)).join(","));
    return api<Analysis>("/api/faces/analyze", { method: "POST", body });
  };

  const pick = async (f: File, box: number[], id: number | "custom") => {
    setBusy(true);
    setError("");
    setForce(false);
    setConflict("");
    try {
      const a = await analyze(f, box); // asks again for just this face -> also tells who it resembles
      setSelected(a.faces[0]);
      setSelectedId(id);
    } catch (e) {
      setSelected(null);
      setSelectedId(null);
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const onFile = async (f: File | null) => {
    setFile(f);
    setAnalysis(null);
    setSelected(null);
    setSelectedId(null);
    setDraft(null);
    setError("");
    setUrl(f ? URL.createObjectURL(f) : null);
    if (!f) return;
    setBusy(true);
    try {
      const a = await analyze(f);
      setAnalysis(a);
      if (a.faces.length === 1) await pick(f, a.faces[0].box, a.faces[0].index);
      else if (a.faces.length === 0)
        setError("No faces found automatically. Use Draw box and outline the face");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const dup = selected?.similar?.[0];
  const dupWarn = !!dup && dup.score >= DUP_WARN;
  const needsForce = dupWarn || !!conflict;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!file || !selected) return;
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.set("full_name", fullName);
      body.set("person_code", code);
      if (notes) body.set("notes", notes);
      body.append("images", file);
      body.set("boxes", JSON.stringify([selected.box]));
      if (force) body.set("force", "true");
      const person = await api<Person>("/api/persons", { method: "POST", body });
      setFullName("");
      setCode("");
      setNotes("");
      await onFile(null);
      onRegistered(person);
    } catch (err) {
      if (err instanceof ApiError && err.status === 409 && err.code) setConflict(err.message);
      else setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const items: BoxItem[] = [
    ...(analysis?.faces ?? []).map((f) => ({
      id: f.index,
      box: f.box,
      label: String(f.index + 1),
      tone: (selectedId === f.index ? "selected" : "default") as BoxItem["tone"],
    })),
    ...(selectedId === "custom" && selected
      ? [{ id: "custom", box: selected.box, kps: selected.kps, label: "Selected", tone: "selected" as const }]
      : []),
  ];

  return (
    <div className="group-reg">
      <label>Photo with one or more people</label>
      <input
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={(e) => {
          const f = e.target.files?.[0] ?? null;
          e.target.value = ""; // so choosing the same file again still fires onChange
          onFile(f);
        }}
      />

      {url && analysis && (
        <>
          <div className="row" style={{ margin: "10px 0 6px" }}>
            <span className="muted">
              {analysis.faces.length > 0
                ? `${analysis.faces.length} ${analysis.faces.length === 1 ? "face" : "faces"} found. Click a numbered box`
                : "No faces found"}
            </span>
            <button type="button" className={drawMode ? "" : "ghost"} onClick={() => setDrawMode((d) => !d)}>
              {drawMode ? "Drag on the photo" : "Draw box"}
            </button>
          </div>
          <ImageWithBoxes
            src={url}
            width={analysis.image.width}
            height={analysis.image.height}
            items={items}
            draft={draft}
            drawing={drawMode}
            onSelect={(id) => file && !drawMode && pick(file, analysis.faces[id as number].box, id as number)}
            onDraw={(box) => {
              if (!file) return;
              setDraft(box);
              setDrawMode(false);
              pick(file, box, "custom");
            }}
          />
        </>
      )}

      {busy && <p className="muted">Working</p>}
      {error && <p className="error">{error}</p>}

      {selected && (
        <form onSubmit={submit} className="selected-face">
          <h2>Selected face</h2>
          <div className="panel" style={{ width: "100%" }}>
            <div className="cap">
              <strong>Model input</strong>
              <div className="muted">Aligned 112 by 112</div>
            </div>
            <div className="frame">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={dataUrl(selected.aligned_png_b64)} alt="aligned face" />
            </div>
            <QualityChecks quality={selected.quality} />
          </div>

          {dup && (
            <div className={dupWarn ? "notice warn" : "notice"}>
              {dupWarn ? (
                <>
                  This face is very similar to <strong>{dup.name}</strong> with ID {dup.code} at{" "}
                  <strong>{dup.score.toFixed(2)}</strong>. It may be the same person
                </>
              ) : (
                <>
                  Closest registered person is {dup.name} with ID {dup.code} at {dup.score.toFixed(2)}.
                  Clearly different
                </>
              )}
            </div>
          )}
          {conflict && <div className="notice warn">{conflict}</div>}

          <label>Full name</label>
          <input value={fullName} onChange={(e) => setFullName(e.target.value)} required maxLength={200} />
          <label>Person ID</label>
          <input value={code} onChange={(e) => setCode(e.target.value)} required maxLength={64} />
          <label>Notes</label>
          <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} />

          {needsForce && (
            <label className="check">
              <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
              Register anyway
            </label>
          )}
          <div style={{ marginTop: 12 }}>
            <button disabled={busy || (needsForce && !force)}>
              {busy ? "Registering" : "Register this face"}
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
