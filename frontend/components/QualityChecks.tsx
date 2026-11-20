import { Check, Quality } from "@/lib/api";

// Labels and hints are built here from the check key and status, not from the stored text,
// so photos and events saved earlier read the same as new ones.
const LABEL: Record<string, string> = {
  size: "Face size",
  det: "Detection confidence",
  blur: "Sharpness",
  light: "Brightness",
  pose: "Head turn estimate",
};

function hint(c: Check): string {
  switch (c.key) {
    case "size":
      return c.status === "fail"
        ? "Below the 60 px minimum"
        : c.status === "warn"
          ? "Needs at least 100 px"
          : "Large enough";
    case "det":
      return c.status === "ok" ? "Confident" : "Low confidence";
    case "blur":
      return c.status === "ok" ? "Sharp enough" : "Blurry";
    case "light":
      return c.status === "ok" ? "Lighting is fine" : parseFloat(c.value) < 128 ? "Too dark" : "Too bright";
    case "pose":
      return c.status === "ok" ? "Facing the camera" : "Face is turned";
    default:
      return "";
  }
}

export default function QualityChecks({ quality }: { quality?: Quality | null }) {
  if (!quality) return <p className="muted">No quality data</p>;
  return (
    <ul className="checks">
      {quality.checks.map((c) => (
        <li key={c.key} className={c.status}>
          <span className="check-icon" title={c.status} />
          <span className="check-label">{LABEL[c.key] ?? c.label}</span>
          <span className="check-value">{c.value.replace(/\/255$/, "")}</span>
          <span className="check-hint muted">{hint(c)}</span>
        </li>
      ))}
    </ul>
  );
}
