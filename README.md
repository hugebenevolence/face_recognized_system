# Real-time Face Recognition (DeepStream + FastAPI + Next.js + pgvector)

Register faces in a web UI, watch a live video stream with recognised names drawn on it, and browse a searchable history of every recognition event.

<p align="center"><img src="docs/figures/architecture.png" alt="System architecture" width="900"></p>

**User guide:** [`docs/USER_GUIDE.pdf`](docs/USER_GUIDE.pdf) (two pages with screenshots; LaTeX source in [`docs/`](docs/)).

| Service    | Tech                                                                    | Port |
|------------|-------------------------------------------------------------------------|------|
| `db`       | `pgvector/pgvector:pg16`                                                | -    |
| `backend`  | FastAPI, SQLAlchemy 2, pgvector                                         | 8000 |
| `pipeline` | **NVIDIA DeepStream 7.1** (GStreamer, nvinfer / TensorRT, nvtracker), pyds. Live video and still images | 8090 |
| `frontend` | Next.js 14 (App Router)                                                 | 3000 |
| `mediamtx` | optional simulated RTSP camera (`--profile sim`)                        | 8554 |

## The video pipeline

Detection, tracking, face alignment, recognition, drawing and encoding all run **inside one DeepStream (GStreamer) graph**, on the GPU:

```
source ──► nvstreammux ──► nvinfer SCRFD ──► nvtracker ──► nvdspreprocess ──► nvinfer ArcFace ──► nvdsosd ──► nvjpegenc ──► MJPEG
(RTSP,      batches        face boxes +      stable ids    face alignment     512-d embedding     draws boxes     (optional: nvv4l2h264enc
 file,      frames         5 landmarks       per face      (CUDA warp)        per face            and labels       ──► rtspclientsink)
 browser)
                       probe 1: decode    probe 2: drop tracker-only               probe 3: match in pgvector,
                       SCRFD tensors      boxes (no landmarks)                     label, post events
```

* **SCRFD** (face detector, box + 5 landmarks) and **ArcFace** (`w600k_r50`, 512-d embedding) run as `nvinfer` elements on TensorRT (FP16). The engines are built on first start and cached in the `models` volume.
* **`nvtracker`** gives each face a stable id, so each appearance produces one event and the database is not queried on every frame.
* **Face alignment** happens in the graph, the way ArcFace was trained, in `nvdspreprocess` with a custom library (`pipeline/face_align`, C++ and CUDA). It only uses standard DeepStream metadata: the face objects, and SCRFD's raw output that `nvinfer` attaches to each frame (`output-tensor-meta=1`). For each face it decodes the five landmarks, fits the similarity transform (rotation, scale, translation) onto the ArcFace template, warps the face out of the GPU frame and writes ArcFace's input tensor. Faces with a tilted head stay recognisable (see *How recognition works*).
* Three small **pad probes** (`pipeline/app/analytics.py`) handle the metadata: decode SCRFD's raw tensors into boxes and landmarks, pair each tracked box with its fresh detection, then read the embedding, match it with pgvector and label the box. Matching and event delivery stay in Python because they talk to PostgreSQL and the backend.
* Video reaches the browser as MJPEG through the backend (`/api/stream.mjpg`). Set `RTSP_OUT_URL` to also publish the annotated stream as H.264 over RTSP.

### Registration uses the same pipeline

Photos (registration, group photos, camera captures) and the explanation maps go through a second, small DeepStream graph that lives in the same container and shares the two models:

```
photo ──► view (640x640) ──► nvstreammux ──► nvinfer SCRFD ──► nvdspreprocess ──► nvinfer ArcFace ──► embedding
          tiles for big      batch of 1       finds faces        face alignment     same engine        stored in pgvector
          group photos                        (or given faces)   same library       as live
```

It is the live chain without the tracker (`pipeline/app/still_graph.py`, `engine.py`), so a face registered from a photo is detected, aligned and embedded by exactly the same code as the same face on camera. Large photos are cut into overlapping tiles so small faces keep their pixels. For the explanation maps, many masked, already aligned 112x112 faces are laid out as a mosaic in one view. They are added as a second object class, *aligned face* (class 1), which the alignment library scales from its box instead of aligning it again, and ArcFace embeds all of them in one pass.

## Quick start

Requirements: Docker with Compose v2 and an **NVIDIA GPU** (DeepStream has no CPU mode) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (on Windows: Docker Desktop with the WSL2 backend). Disk: the DeepStream base image is about 20 GB.

### 1. Provide a video source

Easiest is a simulated RTSP camera: put any MP4 containing faces at `samples/demo.mp4`.

### 2. Start

```bash
docker compose --profile sim up --build
```

The first start builds the TensorRT engines, which takes a few minutes (cached afterwards). Registration waits for that to finish. Then open <http://localhost:3000>.

### 3. Try it

1. **Faces** → fill in name + ID, upload one or more clear face photos (or use *Capture from live stream*) → Register.
2. **Live** → the stream shows boxes with name and similarity; events appear on the right in real time.
3. **History** → search by name/ID, filter by status and time range, paginate.

## Choosing a camera in the UI

The **Live** page has a camera picker. Switching is instant (no restart):

| Option | How it works |
|---|---|
| **This device (browser)** | Lists the cameras your browser can see (webcam, USB, phone-as-webcam such as Camo). Click *Detect cameras* once to grant permission so names show up, pick one, then *Use this camera*. The page captures frames (~10 fps, max 960 px wide) and streams them as JPEG over a WebSocket (`/api/ws/camera`) into the graph (`appsrc ! jpegparse ! nvjpegdec`). **Keep the tab open** while it streams. Needs `http://localhost` or HTTPS. |
| **Server sources** | RTSP presets, any video in `./samples/` (looped, paced to real time), or a custom `rtsp://` URL. The graph reads these itself (`rtspsrc` or `uridecodebin`, hardware decode with `nvv4l2decoder`). |

Detection, tracking, recognition and event logging are identical for every source. The choice lasts until the pipeline container restarts, then it falls back to `SOURCE_URI`. For safety the source endpoint only accepts `rtsp(s)://` URLs and video files under `/samples`.

## Other video sources

Set `SOURCE_URI` in `.env` (copy `.env.example`) for the default source at start-up:

| Source               | `SOURCE_URI`                                       |
|----------------------|----------------------------------------------------|
| Real RTSP camera     | `rtsp://user:pass@192.168.1.10:554/stream` (H.264) |
| Video file (looped)  | `/samples/demo.mp4` (and drop `--profile sim`)     |

**Windows/macOS webcam**: Docker can't see the host webcam. The simplest fix is the *This device (browser)* option above. Alternatively, run `scripts\publish-webcam.bat ["Camera name"]` on the host (needs ffmpeg) to publish it to the bundled RTSP server (`--profile sim`), then choose *Host webcam via RTSP* in the picker or set `SOURCE_URI=rtsp://mediamtx:8554/webcam`.

## How recognition works

- **Detection**: SCRFD. **Embedding**: ArcFace (512-d, L2-normalised). Both come from insightface's `buffalo_l` pack, converted to TensorRT by `nvinfer`.
- **Matching**: cosine similarity via pgvector (`<=>` operator, HNSW index). A face is *known* if similarity ≥ `MATCH_THRESHOLD` (default 0.40), otherwise *Unknown*.
- **Alignment**: a least-squares similarity transform (Umeyama, as in insightface's `norm_crop`) from the 5 landmarks onto the 112x112 ArcFace template, applied by the CUDA kernel in `nvdspreprocess`. The kernel matches an OpenCV `warpAffine` of the same face to a cosine similarity of 0.9998.
- **Tracking**: `nvtracker` (IOU) gives each face a stable ID. ArcFace embeds every tracked face on every frame; the database is queried for a face every `ID_INTERVAL` frames (5) while it is unresolved and every `REID_INTERVAL` frames (45) once known. Each appearance produces **one** event (re-logged after `EVENT_COOLDOWN` seconds if the person stays in view, or immediately if their status changes).
- **Events** are posted by the pipeline to the backend, which stores them in PostgreSQL (timestamp, person or Unknown, confidence, camera, snapshot path, evidence) and pushes them to browsers over `WS /ws/events`.
- Snapshots and registered images are stored on the `appdata` volume and served at `/files/...`.
- **Registration and live video use the same pipeline.** Photos are embedded by the still-image graph, live faces by the live graph, with the same alignment library and the same ArcFace engine, so the vectors are directly comparable.
- **Why alignment matters** (24 faces from a test video, each rotated in the image plane and compared with the same face upright, through the same TensorRT engine; recognition threshold 0.40):

  | Head tilt | Crop without rotation (previous version) | Full alignment (now) |
  |---|---|---|
  | 10° | 0.82 | 0.99 |
  | 20° | 0.56 | 0.98 |
  | 30° | 0.38 (below the threshold) | 0.97 |
  | 45° | 0.17 | 0.81 |

- **Embedding spaces.** Every stored vector carries the id of what produced it (`MODEL_ID`, now `buffalo_l-aligned`) and is only compared with vectors of the same id. After a change of model or alignment, `POST /api/admin/reanalyze` re-embeds every stored registration photo under the new id and keeps the old vectors.

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
  `similarity / threshold` on every face (drawn by `nvdsosd` from display metadata).
* **Which regions matter** (on demand): *Explain regions* hides one patch at a time (occlusion
  sensitivity) and measures how far the score falls: for recognition (similarity to the matched person)
  and for detection (detector confidence). It runs in the still-image graph on the stored crops, never on the live stream, and is
  cached. It shows *model sensitivity*, not human-style reasons, and depends on patch size.

## Performance

Measured on a laptop with an RTX 3050 (4 GB), 1280x720 input, six faces in view, Docker Desktop on WSL2: the whole graph
(decode, SCRFD, tracker, alignment and ArcFace for every face on every frame, drawing, JPEG encode) ran at about 50 frames per second
with an unpaced file source, about twice the rate of a real camera (25 to 30 fps). The first start builds the TensorRT engines (about 4 minutes for both models);
later starts load them from the cache in about 2 seconds.

## Configuration

All optional; see `.env.example`. Tuning knobs for the pipeline (`MATCH_THRESHOLD`, `DET_THRESH`, `MIN_FACE_PX`, `REID_INTERVAL`, `EVENT_COOLDOWN`, ...) are read in `pipeline/app/config.py`.

## Design notes & limitations

- **Why DeepStream.** Inference, tracking and drawing are elements of the GStreamer graph and run on the GPU without leaving it, instead of pulling frames into application code. The cost is an NVIDIA-only, roughly 20 GB base image.
- **SCRFD output is decoded in a pad probe.** `nvinfer` runs the network, but its stock parser interface cannot return the five landmarks, so the raw tensors are decoded in Python (`app/scrfd.py`). A C++ parser would be faster but is not needed at this scale.
- **Alignment in `nvdspreprocess`.** `nvinfer` can only crop axis-aligned rectangles, which cannot undo a tilted head. The alignment therefore runs in `nvdspreprocess`, DeepStream's element for custom input preparation, with a custom library. Pixels outside the frame are filled with black, as in insightface's `norm_crop`, so a face at the image border is still embedded.
- **One fix to NVIDIA's `gst-nvinfer` (DeepStream 7.1).** A secondary `nvinfer` on objects whose input comes from `nvdspreprocess` adds one ROI user meta to both the frame and the object, and that meta's release function also deletes the object, which belongs to the metadata pool. The buffer is therefore freed twice and the process crashes with heap corruption. The same double attachment exists for classifiers, which matches an NVIDIA forum report that was closed without a fix ([double free with nvdspreprocess and SGIE](https://forums.developer.nvidia.com/t/double-free-or-corruption-crash-with-nvdspreprocess-and-sgie-nvinfer/342784)). NVIDIA ships this plugin as source in the DeepStream image, and the Dockerfile rebuilds it with a small patch (`pipeline/patches/`, the problem is described at the top of the file). The patch attaches ArcFace's output to the object only, which is where a regular secondary `nvinfer` puts it, so the probe reads it from `obj_user_meta_list` as usual. A newer DeepStream would be the alternative; it was not an option here (a second 20 GB base image).
- Turning the head sideways (out of the image plane) is not corrected by any 2D alignment; register a few angles (the guided camera capture does this).
- RTSP cameras must send H.264. Single camera per pipeline container.
- Tables are created at backend start-up (`create_all`) plus a few idempotent `ALTER TABLE ... IF NOT EXISTS` for columns added later; there is no migration tool yet.
- No authentication on the API; it is meant for a trusted network. The pipeline to backend channel uses a shared secret (`INTERNAL_TOKEN`).

## Project layout

```
backend/    FastAPI app (models, routers, WebSocket, client for the pipeline service)
pipeline/   DeepStream: live graph (runner.py, analytics.py), still-image graph (still_graph.py, engine.py, stills.py, explain.py),
            SCRFD decoder, configs for nvinfer / nvtracker / nvdspreprocess, face_align/ (CUDA alignment library),
            patches/ (gst-nvinfer fix)
docs/graphs/ the real GStreamer graphs of both pipelines, exported from the running service
frontend/   Next.js UI (live, faces, history)
samples/    put demo.mp4 here
```
