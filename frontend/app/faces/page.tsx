"use client";

import { FormEvent, useCallback, useEffect, useRef, useState } from "react";
import CameraCaptureModal from "@/components/CameraCaptureModal";
import GroupPhotoRegister from "@/components/GroupPhotoRegister";
import Modal from "@/components/Modal";
import RegistrationReport from "@/components/RegistrationReport";
import { API, ApiError, ModelStatus, Person, api, fileUrl, fmtTime } from "@/lib/api";

type Pending = { file: File; url: string };

export default function FacesPage() {
  const [people, setPeople] = useState<Person[]>([]);
  const [q, setQ] = useState("");
  const [loadError, setLoadError] = useState("");
  const [loaded, setLoaded] = useState(false); // avoid flashing "Nobody registered yet" before the first fetch

  const [tab, setTab] = useState<"files" | "group">("files");
  const [fullName, setFullName] = useState("");
  const [code, setCode] = useState("");
  const [notes, setNotes] = useState("");
  const [pending, setPending] = useState<Pending[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState("");
  const [force, setForce] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const [reportFor, setReportFor] = useState<Person | null>(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [updating, setUpdating] = useState(false);

  const load = useCallback(async () => {
    try {
      setPeople(await api<Person[]>(`/api/persons${q ? `?q=${encodeURIComponent(q)}` : ""}`));
      setLoadError("");
    } catch (e) {
      setLoadError((e as Error).message);
    } finally {
      setLoaded(true);
    }
  }, [q]);

  const loadModelStatus = useCallback(() => {
    api<ModelStatus>("/api/admin/model-status")
      .then(setModelStatus)
      .catch(() => setModelStatus(null));
  }, []);

  useEffect(() => {
    const t = setTimeout(load, q ? 250 : 0); // debounce typing only; load immediately otherwise
    return () => clearTimeout(t);
  }, [load, q]);

  useEffect(loadModelStatus, [loadModelStatus]);

  // Free object URLs when previews are dropped.
  useEffect(() => () => pending.forEach((p) => URL.revokeObjectURL(p.url)), [pending]);

  const addFiles = (files: File[]) =>
    setPending((prev) => [...prev, ...files.map((file) => ({ file, url: URL.createObjectURL(file) }))]);

  const capture = async () => {
    setError("");
    try {
      const res = await fetch(`${API}/api/snapshot.jpg`, { cache: "no-store" });
      if (!res.ok) throw new Error("No frame available. Check the live stream");
      const blob = await res.blob();
      addFiles([new File([blob], `capture-${Date.now()}.jpg`, { type: "image/jpeg" })]);
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const registered = async (p: Person) => {
    await load();
    loadModelStatus();
    setReportFor(p); // show how it was registered right away
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (pending.length === 0) return setError("Add at least one photo");
    setBusy(true);
    setError("");
    try {
      const body = new FormData();
      body.set("full_name", fullName);
      body.set("person_code", code);
      if (notes) body.set("notes", notes);
      pending.forEach((p) => body.append("images", p.file));
      if (force) body.set("force", "true");
      const person = await api<Person>("/api/persons", { method: "POST", body });
      setFullName("");
      setCode("");
      setNotes("");
      setPending([]);
      setForce(false);
      setConflict("");
      if (fileInput.current) fileInput.current.value = "";
      await registered(person);
    } catch (err) {
      // 409 with a code = "looks like a mistake, confirm to continue" (not a plain failure)
      if (err instanceof ApiError && err.status === 409 && err.code) setConflict(err.message);
      else setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (p: Person) => {
    if (!confirm(`Delete ${p.full_name}? History is kept`)) return;
    try {
      await api(`/api/persons/${p.id}`, { method: "DELETE" });
      await load();
      loadModelStatus();
    } catch (err) {
      setLoadError((err as Error).message);
    }
  };

  const updateModel = async () => {
    setUpdating(true);
    try {
      await api("/api/admin/reanalyze", { method: "POST" });
      await load();
      loadModelStatus();
    } catch (err) {
      setLoadError((err as Error).message);
    } finally {
      setUpdating(false);
    }
  };

  return (
    <>
      <h1>Registered faces</h1>

      {modelStatus && modelStatus.needs_update > 0 && (
        <div className="notice warn" style={{ marginBottom: 16 }}>
          {modelStatus.needs_update} {modelStatus.needs_update === 1 ? "photo needs" : "photos need"} an
          update for the current model {modelStatus.current_model ?? ""}. Those people are not recognized
          until you update{" "}
          <button onClick={updateModel} disabled={updating}>
            {updating ? "Updating" : "Update"}
          </button>
        </div>
      )}

      <div className="grid2" style={{ gridTemplateColumns: "minmax(300px, 1fr) minmax(0, 2fr)" }}>
        <div className="card" style={{ alignSelf: "start" }}>
          <h2>Register a person</h2>
          <div className="tabs">
            <button type="button" className={tab === "files" ? "active" : ""} onClick={() => setTab("files")}>
              Photos
            </button>
            <button type="button" className={tab === "group" ? "active" : ""} onClick={() => setTab("group")}>
              Group photo
            </button>
          </div>

          {tab === "group" ? (
            <GroupPhotoRegister onRegistered={registered} />
          ) : (
            <form onSubmit={submit}>
              <label>Full name</label>
              <input
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
                maxLength={200}
              />
              <label>Person ID</label>
              <input value={code} onChange={(e) => setCode(e.target.value)} required maxLength={64} />
              <label>Notes</label>
              <textarea value={notes} onChange={(e) => setNotes(e.target.value)} rows={2} />

              <label>Face photos</label>
              <div className="row">
                <input
                  ref={fileInput}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  multiple
                  onChange={(e) => {
                    addFiles(Array.from(e.target.files ?? []));
                    e.target.value = "";
                  }}
                />
              </div>
              <div className="row" style={{ marginTop: 8 }}>
                <button type="button" onClick={() => setCameraOpen(true)}>
                  Open camera
                </button>
                <button type="button" className="ghost" onClick={capture}>
                  Capture from live stream
                </button>
              </div>
              {pending.length > 0 && (
                <div className="previews">
                  {pending.map((p, i) => (
                    <div className="preview" key={p.url}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={p.url} alt="" />
                      <button
                        type="button"
                        onClick={() => setPending((prev) => prev.filter((_, j) => j !== i))}
                      >
                        x
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {conflict && (
                <div className="notice warn" style={{ marginTop: 10 }}>
                  {conflict}
                </div>
              )}
              {conflict && (
                <label className="check">
                  <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
                  Register anyway
                </label>
              )}
              {error && <p className="error">{error}</p>}
              <div style={{ marginTop: 14 }}>
                <button disabled={busy || (!!conflict && !force)}>{busy ? "Registering" : "Register"}</button>
              </div>
            </form>
          )}
        </div>

        <div className="card">
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 8 }}>
            <h2 style={{ margin: 0 }}>
              {loaded ? `${people.length} ${people.length === 1 ? "person" : "people"}` : "Loading"}
            </h2>
            <input
              style={{ maxWidth: 240 }}
              placeholder="Search name or ID"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          {loadError && <p className="error">{loadError}</p>}
          <table>
            <thead>
              <tr>
                <th>Faces</th>
                <th>Name</th>
                <th>ID</th>
                <th>Notes</th>
                <th>Registered</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {people.map((p) => (
                <tr key={p.id}>
                  <td>
                    <div className="faces">
                      {p.faces.slice(0, 3).map((f) => (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img key={f.id} className="thumb" src={fileUrl(f.image_path)} alt="" />
                      ))}
                    </div>
                  </td>
                  <td>{p.full_name}</td>
                  <td>{p.person_code}</td>
                  <td className="muted">{p.notes}</td>
                  <td className="muted">{fmtTime(p.created_at)}</td>
                  <td>
                    <div className="row" style={{ flexWrap: "nowrap" }}>
                      <button className="ghost" onClick={() => setReportFor(p)}>
                        Report
                      </button>
                      <button className="danger" onClick={() => remove(p)}>
                        Delete
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {people.length === 0 && (
                <tr>
                  <td colSpan={6} className="muted">
                    {loaded ? "No people yet" : "Loading"}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {cameraOpen && <CameraCaptureModal onDone={addFiles} onClose={() => setCameraOpen(false)} />}

      {reportFor && (
        <Modal title={`Registration report ${reportFor.full_name}`} onClose={() => setReportFor(null)}>
          <RegistrationReport personId={reportFor.id} />
        </Modal>
      )}
    </>
  );
}
