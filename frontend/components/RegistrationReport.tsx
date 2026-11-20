"use client";

import { useCallback, useEffect, useState } from "react";
import { Report, api, fileUrl, fmtTime } from "@/lib/api";
import ImageWithBoxes from "./ImageWithBoxes";
import QualityChecks from "./QualityChecks";

/** What the system stored for a person and how reliable that registration is. */
export default function RegistrationReport({ personId }: { personId: number }) {
  const [rep, setRep] = useState<Report | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    try {
      setRep(await api<Report>(`/api/persons/${personId}/report`));
      setError("");
    } catch (e) {
      setError((e as Error).message);
    }
  }, [personId]);

  useEffect(() => {
    load();
  }, [load]);

  const update = async () => {
    setBusy(true);
    try {
      await api("/api/admin/reanalyze", { method: "POST" });
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  if (error) return <p className="error">{error}</p>;
  if (!rep) return <p className="muted">Loading</p>;

  const { person, pairs, consistency, nearest_other: other, thresholds } = rep;
  const missing = person.faces.some((f) => !f.meta);
  const nPhotos = person.faces.length;
  const models = Array.from(new Set(person.faces.map((f) => f.model ?? "buffalo_l")));
  const coverage = nPhotos >= 3 ? "Good" : nPhotos === 2 ? "Fair" : "Low";

  return (
    <div className="evidence">
      <div className="decision">
        <div>
          <strong>{person.full_name}</strong> <span className="muted">ID {person.person_code}</span>
          <div className="meta muted">
            <span>Registered {fmtTime(person.created_at)}</span>
            {person.notes && <span>{person.notes}</span>}
          </div>
        </div>
        <div className="verdict">
          <span className="verdict-score">{nPhotos}</span>
          <span className="muted">
            {nPhotos === 1 ? "photo" : "photos"} {models.join(" ")}
          </span>
        </div>
      </div>

      {missing && (
        <div className="notice">
          Some photos have no quality data yet{" "}
          <button className="ghost" onClick={update} disabled={busy}>
            {busy ? "Updating" : "Update report"}
          </button>
        </div>
      )}

      <h3>Registered photos</h3>
      {person.faces.map((f, i) => {
        const size = f.meta?.image_size;
        return (
          <div className="face-report" key={f.id}>
            <div className="cap">
              <strong>Photo {i + 1}</strong>
              <div className="meta muted">
                <span>{f.model ?? "buffalo_l"}</span>
                <span>Detection {f.det_score.toFixed(2)}</span>
                {f.meta?.faces_in_photo && f.meta.faces_in_photo > 1 && (
                  <span>{f.meta.faces_in_photo} faces in photo</span>
                )}
              </div>
            </div>
            <div className="seen">
              <div className="panel">
                <div className="cap">
                  <strong>Original photo</strong>
                </div>
                {size && f.meta?.bbox ? (
                  <ImageWithBoxes
                    frame
                    src={fileUrl(f.image_path)}
                    width={size[0]}
                    height={size[1]}
                    items={[{ id: f.id, box: f.meta.bbox, kps: f.meta.kps ?? undefined, tone: "good" }]}
                  />
                ) : (
                  <div className="frame">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={fileUrl(f.image_path)} alt="" />
                  </div>
                )}
              </div>
              <div className="panel">
                <div className="cap">
                  <strong>Model input</strong>
                </div>
                <div className="frame">
                  {f.crop_path && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={fileUrl(f.crop_path)} alt="aligned" />
                  )}
                </div>
              </div>
            </div>
            <QualityChecks quality={f.meta?.quality} />
          </div>
        );
      })}

      <h3>Reliability</h3>
      <ul className="kv">
        <li>
          <span className="k">Photos</span>
          <span className="v">{nPhotos}</span>
          <span className={coverage === "Low" ? "s warn-text" : "s ok-text"}>
            {coverage === "Low" ? "Add 2 or 3 more at other angles" : `${coverage} coverage`}
          </span>
        </li>
        <li>
          <span className="k">Consistency</span>
          {consistency ? (
            <>
              <span className="v">
                Lowest {consistency.min.toFixed(2)} Average {consistency.mean.toFixed(2)}
              </span>
              <span className={consistency.min < thresholds.consistency ? "s warn-text" : "s ok-text"}>
                {consistency.min < thresholds.consistency ? "Photos may be mixed" : "Photos agree"}
              </span>
            </>
          ) : (
            <>
              <span className="v">None</span>
              <span className="s muted">Needs 2 photos</span>
            </>
          )}
        </li>
        <li>
          <span className="k">Most similar person</span>
          {other ? (
            <>
              <span className="v">
                {other.full_name} {other.score.toFixed(2)}
              </span>
              <span className={other.score >= thresholds.duplicate ? "s warn-text" : "s ok-text"}>
                {other.score >= thresholds.duplicate ? "Easy to confuse" : "Clearly different"}
              </span>
            </>
          ) : (
            <>
              <span className="v">None</span>
              <span className="s muted">Nobody else registered</span>
            </>
          )}
        </li>
      </ul>
      {pairs.length > 0 && (
        <p className="muted small">
          Photo pair similarity {pairs.map((p) => p.similarity.toFixed(2)).join(" ")}
        </p>
      )}

      <h3>Stored data</h3>
      <ul className="facts">
        <li>Original photos and aligned faces</li>
        <li>A 512 number vector per photo from {models.join(" ")}</li>
        <li>Face box and landmarks and quality checks</li>
        <li>The vector cannot rebuild the photo</li>
        <li>Deleting the person removes all of this and history keeps the name</li>
      </ul>
    </div>
  );
}
