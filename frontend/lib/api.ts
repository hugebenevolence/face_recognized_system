export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
export const WS_URL = API.replace(/^http/, "ws") + "/ws/events";
export const CAMERA_WS_URL = API.replace(/^http/, "ws") + "/api/ws/camera";

export type SourcePreset = { id: string; label: string; uri: string };
export type SourcesResponse = {
  presets: SourcePreset[];
  current: { kind: "uri" | "browser"; uri: string | null };
};

// ---- quality / evidence ----
export type Check = {
  key: string;
  label: string;
  value: string;
  status: "ok" | "warn" | "fail";
  hint: string;
};
export type Quality = {
  overall: "ok" | "warn" | "fail";
  checks: Check[];
  metrics: Record<string, number>;
};
export type Candidate = { person_id: number; name: string; code: string; score: number };

export type FaceMeta = {
  bbox?: number[] | null;
  kps?: number[][] | null;
  image_size?: number[] | null;
  faces_in_photo?: number | null;
  quality?: Quality | null;
};
export type Face = {
  id: number;
  image_path: string;
  det_score: number;
  model?: string;
  crop_path?: string | null;
  meta?: FaceMeta | null;
  created_at?: string | null;
};
export type Person = {
  id: number;
  full_name: string;
  person_code: string;
  notes: string | null;
  created_at: string;
  faces: Face[];
};

export type RecEvent = {
  id: number;
  ts: string;
  status: "known" | "unknown";
  person_id: number | null;
  person_name: string | null;
  person_code: string | null;
  confidence: number;
  camera_id: string;
  track_id: number | null;
  snapshot_path: string | null;
  has_evidence?: boolean;
};
export type Evidence = {
  version: number;
  model: string;
  threshold: number;
  det_score: number;
  crop_size: [number, number];
  bbox: number[];
  kps: number[][];
  quality: Quality | null;
  top: Candidate[];
  margin: number | null;
  aligned_path?: string | null;
};
export type EventDetail = RecEvent & { evidence: Evidence | null };
export type EventPage = { items: RecEvent[]; total: number; page: number; page_size: number };

export type PipelineStatus = {
  pipeline_up?: boolean;
  connected: boolean;
  camera_id?: string;
  source?: string;
  source_kind?: "uri" | "browser";
  fps?: number;
  error?: string | null;
  model?: string;
  explain_overlay?: boolean;
};

// ---- group-photo analysis / reports / explanations ----
export type FaceInfo = {
  index: number;
  box: number[];
  kps: number[][];
  score: number;
  quality: Quality;
  aligned_png_b64: string;
  similar?: Candidate[];
};
export type Analysis = {
  image: { width: number; height: number };
  model: string;
  faces: FaceInfo[];
  truncated: boolean;
};
export type Report = {
  person: Person;
  pairs: { a: number; b: number; similarity: number }[];
  consistency: { min: number; mean: number } | null;
  nearest_other: { person_id: number; full_name: string; person_code: string; score: number } | null;
  thresholds: { duplicate: number; consistency: number };
};
export type ModelStatus = {
  current_model: string | null;
  embeddings_by_model: Record<string, number>;
  images_total: number;
  needs_update: number;
};
export type RegionShare = { name: string; share: number };
export type ExplainResult = {
  event_id: number;
  target: {
    person_id: number | null;
    name: string | null;
    code: string | null;
    matched: boolean;
    available: boolean;
  };
  method: string;
  model: string;
  elapsed_ms: number;
  recognition: null | {
    base_similarity: number;
    max_drop: number;
    regions: RegionShare[];
    overlay_png_b64: string;
  };
  detection: null | {
    base_score: number;
    max_drop?: number;
    regions?: RegionShare[];
    overlay_png_b64?: string;
    note?: string;
  };
};

export const fileUrl = (path?: string | null) => (path ? `${API}/files/${path}` : "");
export const dataUrl = (b64?: string | null) => (b64 ? `data:image/png;base64,${b64}` : "");

/** FastAPI errors are `{detail: string | object | [{msg, loc}]}`; flatten to one readable line. */
function describe(detail: unknown): { message: string; code?: string } {
  if (typeof detail === "string") return { message: detail };
  if (Array.isArray(detail))
    return { message: detail.map((d) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg}`).join("; ") };
  if (detail && typeof detail === "object") {
    const d = detail as { message?: string; code?: string };
    return { message: d.message ?? "Request failed", code: d.code };
  }
  return { message: "Request failed" };
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code?: string,
    public detail?: unknown,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`, { cache: "no-store", ...init });
  } catch {
    // fetch only throws when it never got a response (server down / restarting / network)
    throw new ApiError(
      "Cannot reach the server. It may be restarting. Your input is kept so try again in a few seconds",
      0,
      "network",
    );
  }
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`;
    let code: string | undefined;
    let detail: unknown;
    try {
      detail = (await res.json()).detail;
      ({ message, code } = describe(detail));
    } catch {}
    throw new ApiError(message, res.status, code, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const p2 = (n: number) => String(n).padStart(2, "0");
/** 20 Nov 2026 14:35:34 (no commas unlike toLocaleString) */
export const fmtTime = (iso: string) => {
  const d = new Date(iso);
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()} ${p2(d.getHours())}:${p2(d.getMinutes())}:${p2(d.getSeconds())}`;
};
