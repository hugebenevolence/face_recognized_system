# Real-time Face Recognition (GStreamer + FastAPI + Next.js + pgvector)

Register faces in a web UI, watch a live video stream with recognised names drawn on it, and browse a searchable history of every recognition event.

<p align="center"><img src="docs/figures/architecture.png" alt="System architecture" width="900"></p>

**User guide:** [`docs/USER_GUIDE.pdf`](docs/USER_GUIDE.pdf) (two pages with screenshots; LaTeX source in [`docs/`](docs/)).


| Service    | Tech                                              | Port |
|------------|---------------------------------------------------|------|
| `db`       | `pgvector/pgvector:pg16`                          | -    |
| `backend`  | FastAPI, SQLAlchemy 2, pgvector                   | 8000 |
| `pipeline` | GStreamer (PyGObject), insightface (SCRFD+ArcFace) on ONNX Runtime | 8090 |
| `frontend` | Next.js 14 (App Router)                           | 3000 |
| `mediamtx` | optional simulated RTSP camera (`--profile sim`)  | 8554 |

## Quick start

Requirements: Docker with Compose v2. For GPU: an NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (on Windows: Docker Desktop with the WSL2 backend).

### 1. Provide a video source

Easiest is a simulated RTSP camera: put any MP4 containing faces at `samples/demo.mp4`.

### 2. Start

```bash
# CPU
docker compose --profile sim up --build

# NVIDIA GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile sim up --build
```

The first build downloads the model weights and takes several minutes. Then open <http://localhost:3000>.

### 3. Try it

1. **Faces** → fill in name + ID, upload one or more clear face photos (or use *Capture from live stream*) → Register.
2. **Live** → the stream shows boxes with name and similarity; events appear on the right in real time.
3. **History** → search by name/ID, filter by status and time range, paginate.

## Choosing a camera in the UI

The **Live** page has a camera picker. Switching is instant (no restart):

| Option | How it works |
|---|---|
| **This device (browser)** | Lists the cameras your browser can see (webcam, USB, phone-as-webcam such as Camo). Click *Detect cameras* once to grant permission so names show up, pick one, then *Use this camera*. The page captures frames (~10 fps, max 960 px wide) and streams them as JPEG over a WebSocket (`/api/ws/camera`) to the pipeline. **Keep the tab open** while it streams. Needs `http://localhost` or HTTPS. |
| **Server sources** | RTSP presets, any video in `./samples/` (looped), or a custom `rtsp://` URL. The pipeline pulls these itself through GStreamer. |

Browser frames are decoded with OpenCV rather than GStreamer (they are already individual JPEGs); detection, tracking, recognition and event logging are identical for every source. The choice lasts until the pipeline container restarts, then it falls back to `SOURCE_URI`.

For safety the source endpoint only accepts `rtsp(s)://` URLs and video files under `/samples`.

## Other video sources

Set `SOURCE_URI` in `.env` (copy `.env.example`) for the default source at start-up:

| Source               | `SOURCE_URI`                                       |
|----------------------|----------------------------------------------------|
| Real RTSP camera     | `rtsp://user:pass@192.168.1.10:554/stream`         |
| Video file (looped)  | `/samples/demo.mp4` (and drop `--profile sim`)     |
| Linux webcam         | `v4l2:/dev/video0` + pass the device to `pipeline` |

**Windows/macOS webcam**: Docker can't see the host webcam. The simplest fix is the *This device (browser)* option above. Alternatively, run `scripts\publish-webcam.bat ["Camera name"]` on the host (needs ffmpeg) to publish it to the bundled RTSP server (`--profile sim`), then choose *Host webcam via RTSP* in the picker or set `SOURCE_URI=rtsp://mediamtx:8554/webcam`.

## How recognition works

- **Detection**: SCRFD. **Embedding**: ArcFace (512-d, L2-normalised). Both come from insightface's `buffalo_l` pack (`MODEL_PACK=buffalo_s` for a lighter one).
- **Matching**: cosine similarity via pgvector (`<=>` operator, HNSW index). A face is *known* if similarity ≥ `MATCH_THRESHOLD` (default 0.40), otherwise *Unknown*.
- **Tracking**: an IoU tracker gives each face a stable ID, so the embedding is computed only every few frames per face, not every frame, and each appearance produces **one** event (re-logged after `EVENT_COOLDOWN` seconds if the person stays in view, or immediately if their status changes).
- **Events** are posted by the pipeline to the backend, which stores them in PostgreSQL (timestamp, person or Unknown, confidence, camera, snapshot path) and pushes them to browsers over `WS /ws/events`.
- Snapshots and registered images are stored on the `appdata` volume and served at `/files/...`.

## API

Interactive docs: <http://localhost:8000/docs>

| Method & path                              | Purpose                                           |
|--------------------------------------------|---------------------------------------------------|
| `GET /api/persons?q=`                      | List / search registered people                   |
| `POST /api/persons` (multipart)            | Register: `full_name`, `person_code`, `notes`, `images[]` |
| `POST /api/persons/{id}/faces`             | Add more face images to a person                  |
| `DELETE /api/persons/{id}`                 | Delete person + embeddings + images (history kept)|
| `GET /api/events?q=&status=&from=&to=&page=&page_size=` | Paginated history                    |
| `WS /ws/events`                            | Live event push                                   |
| `GET /api/stream.mjpg`, `/api/snapshot.jpg`, `/api/status` | Video, frame capture, pipeline health |
| `POST /api/faces/analyze` (+ optional `box`) | Faces in a photo, or the one in a box, with quality and who it resembles |
| `POST /api/faces/probe`, `POST /api/faces/compare` | Live camera feedback; similarity between several shots |
| `GET /api/persons/{id}/report` | Registration report |
| `GET /api/events/{id}`, `GET /api/events/{id}/explain` | Evidence for an event; occlusion heatmaps (cached) |
| `PUT /api/overlay` | Toggle the explain overlay on the live video |
| `GET /api/admin/model-status`, `POST /api/admin/reanalyze` | Model/embedding status; re-embed stored photos |

## Registering faces (and trusting the result)

On **Faces** there are three ways to add a person and all of them produce the same *registration report*. The interface text is plain English with no special characters by design:

| Way | Notes |
|---|---|
| **Photos** tab: upload or capture from the live stream | One or more photos. The largest face in each is used. |
| **Open camera** | A dialog opens the camera of *this device*. *Guided 5 angles* guides straight / left / right / up / down with live feedback (face found? one person? size, light, pose) and optional auto-capture; *Free capture* takes any number of shots. |
| **Group photo** | Upload a photo with several people. Every face is boxed and numbered: click one, or drag your own box if the detector missed it. One person per registration. |

Before anything is saved the system checks for mistakes (overridable with a confirmation):
the face is almost identical to a *different* registered person (`possible_duplicate`), photos uploaded
together look like different people (`images_disagree`), or a photo added to a person does not look like
them (`inconsistent_with_person`). Thresholds: `DUP_THRESHOLD`, `CONSISTENCY_THRESHOLD`.

The **registration report** (button *Report* on each person, also shown right after registering) shows the
original photo with the chosen face and its 5 landmarks, the aligned 112×112 crop the model really uses,
quality checks (size, detection confidence, sharpness, light, head turn: heuristics, pose is an estimate),
consistency between the person's photos, the most similar other person, and what is stored.

**Do we need several angles?** It depends on the face and the model, so the guided capture *measures* it:
after the 5 shots it shows how similar each angle is to the straight-on shot and says whether a
straight-on photo alone would have been enough. Matching uses the best of all stored photos per person, so extra
angles can only help recall; poor frames can hurt, which is why every shot is quality-gated.

## Explaining decisions (XAI)

* **What the system saw** (always available): on **Live**, click an event; in **History**, *Why*. Shows
  the detected face + landmarks, the aligned crop, quality, the closest people with their scores against the
  threshold and the margin to the runner-up.
* **Explain on video**: the *Explain overlay* switch above the live video draws the 5 landmarks, the detection score and
  `similarity / threshold` on every face.
* **Which regions matter** (on demand): *Explain regions* hides one patch at a time (occlusion
  sensitivity) and measures how far the score falls: for recognition (similarity to the matched person)
  and for detection (detector confidence). It runs on the stored crops, never on the live stream, and is
  cached. It shows *model sensitivity*, not human-style reasons, and depends on patch size.

## CPU or GPU

`docker compose up` (CPU) defaults to `MODEL_PACK=buffalo_s`; the GPU override defaults to `buffalo_l`.
Measured on a Ryzen 5 5600H (12 threads): `buffalo_l` ≈ 190 ms detect + 100 ms/face embed, `buffalo_s`
≈ 35 ms + 15 ms/face. **Embeddings of different model packs are not comparable.** After switching, open
**Faces**: a banner offers *Update* (or `POST /api/admin/reanalyze`), which embeds every stored photo again
with the current model. Photos are kept, so nothing is lost.

## Configuration

All optional; see `.env.example`. Tuning knobs for the pipeline (`MATCH_THRESHOLD`, `DET_SIZE`, `ID_INTERVAL`, `EVENT_COOLDOWN`, ...) are read in `pipeline/app/config.py`.

## Design notes & limitations

- **GStreamer rather than DeepStream.** The task allows either. DeepStream needs a large NGC image and a custom face-recognition plugin chain; the GStreamer + ONNX Runtime design runs on CPU or any NVIDIA GPU and is easy to reproduce. Models run on ONNX Runtime (CUDA provider when the GPU override is used).
- Tables are created at backend start-up (`create_all`) plus a few idempotent `ALTER TABLE ... IF NOT EXISTS` for columns added later; there is no migration tool yet.
- No authentication on the API; it is meant for a trusted network. The pipeline → backend channel uses a shared secret (`INTERNAL_TOKEN`).
- Single camera per pipeline container.
- Detection runs on every frame; if the machine is too slow, frames are dropped (appsink `drop=true`) rather than queued, so latency stays low.

## Project layout

```
backend/    FastAPI app (models, routers, WebSocket, pipeline client)
pipeline/   GStreamer loop, face engine, tracker, matcher, event publisher
frontend/   Next.js UI (live, faces, history)
samples/    put demo.mp4 here
```
