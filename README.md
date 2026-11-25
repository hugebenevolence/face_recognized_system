# Real-time Face Recognition (DeepStream + FastAPI + Next.js + pgvector)

Register faces in a web UI, watch a live video stream with recognised names drawn on it, and browse a searchable history of every recognition event.

<p align="center"><img src="docs/figures/architecture.png" alt="System architecture" width="900"></p>

**User guide:** [`docs/USER_GUIDE.pdf`](docs/USER_GUIDE.pdf) (two pages with screenshots; LaTeX source in [`docs/`](docs/)).

| Service    | Tech                                                                           | Port |
|------------|--------------------------------------------------------------------------------|------|
| `db`       | `pgvector/pgvector:pg16`                                                       | -    |
| `backend`  | FastAPI, SQLAlchemy 2, pgvector                                                | 8000 |
| `pipeline` | **NVIDIA DeepStream 7.1** (GStreamer, nvinfer / TensorRT, nvtracker, nvdspreprocess), pyds | 8090 |
| `frontend` | Next.js 14 (App Router)                                                        | 3000 |
| `mediamtx` | optional simulated RTSP camera (`--profile sim`)                               | 8554 |

## Quick start

Requirements: Docker with Compose v2 and an **NVIDIA GPU** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) (on Windows: Docker Desktop with the WSL2 backend). DeepStream runs on the GPU only. Disk: the DeepStream base image is about 20 GB.

1. **Video source.** The easiest is the simulated RTSP camera: put any MP4 with faces at `samples/demo.mp4`.
2. **Start.**

   ```bash
   docker compose --profile sim up --build
   ```

   The first start builds the TensorRT engines, which takes a few minutes (cached afterwards). Then open <http://localhost:3000>.
3. **Try it.**
   1. **Faces**: enter a name and an ID, add one or more face photos (upload, camera or group photo), then *Register*.
   2. **Live**: the stream shows each face with a box, a name and a similarity score; events appear on the right in real time.
   3. **History**: search by name or ID, filter by status and time range, page through the results.

## The video pipeline

Detection, tracking, face alignment, recognition, drawing and encoding run **inside one DeepStream (GStreamer) graph** on the GPU:

```
source ──► nvstreammux ──► nvinfer SCRFD ──► nvtracker ──► nvdspreprocess ──► nvinfer ArcFace ──► tee ─┬─► nvdsosd ──► nvjpegenc ──► MJPEG
(RTSP,      batches        face boxes +      stable ids    face alignment     512-d embedding           │   boxes and      (optional: H.264
 file,      frames         5 landmarks       per face                         per face                  │   labels          over RTSP)
 browser)                                                                                               └─► nvjpegenc ──► clean frames for
                                                                                                            snapshots and event images
```

* **SCRFD** (face detector: box and five landmarks) and **ArcFace** (`w600k_r50`, 512-d embedding) run as `nvinfer` elements on TensorRT (FP16). The engines are built on first start and cached in the `models` volume.
* **`nvtracker`** gives each face a stable id, so each appearance produces one event and the database is not queried on every frame.
* **Face alignment** runs in `nvdspreprocess` with a custom library (`pipeline/face_align`, C++ and CUDA). For each face it reads SCRFD's landmarks from the frame's metadata, fits the similarity transform (rotation, scale, translation) onto the ArcFace template, warps the face out of the GPU frame and writes ArcFace's input tensor. It also attaches the aligned 112x112 face to the object, so the evidence kept for an event is exactly the image ArcFace saw.
* **Pad probes** (`pipeline/app/analytics.py`) handle the application logic on the metadata: decode SCRFD's output into boxes and landmarks, keep each tracked face with its detection, match the embedding in pgvector, label the box and post events.
* **Images are encoded by `nvjpegenc`**: the annotated stream for the browser, and on a second branch, before drawing, the clean frames that are needed (a periodic snapshot and the frames of new events).
* Video reaches the browser as MJPEG through the backend (`/api/stream.mjpg`). Set `RTSP_OUT_URL` to also publish the annotated stream as H.264 over RTSP.

### Registration uses the same pipeline

Photos (registration, group photos, camera captures) and the explanation maps go through a second, small DeepStream graph in the same container, with the same models and the same alignment library:

```
JPEG ──► nvjpegdec ──► nvvideoconvert ─┐
         GPU decode    crop + scale    ├─► nvstreammux ──► nvinfer SCRFD ──► nvdspreprocess ──► nvinfer ArcFace ──► embedding
other formats ──► raw view ────────────┘                   finds faces       face alignment     same engine        + aligned face
```

A face registered from a photo is therefore detected, aligned and embedded by the same code as the same face on camera. Uploaded JPEGs are decoded on the GPU, and each view (the whole photo, overlapping tiles of a large group photo so small faces keep their pixels, a view centred on one face) is cut and scaled by `nvvideoconvert`. Other image formats are decoded in Python and sent as raw pixels.

## Choosing a camera in the UI

The **Live** page has a camera picker. Switching takes effect immediately:

| Option | How it works |
|---|---|
| **This device (browser)** | Lists the cameras your browser can see (webcam, USB, phone as webcam such as Camo). Click *Detect cameras* once to grant permission, pick one, then *Use this camera*. The page sends JPEG frames (about 10 fps) over a WebSocket (`/api/ws/camera`) into the graph. **Keep the tab open** while it streams. Needs `http://localhost` or HTTPS. |
| **Server sources** | RTSP presets, any video in `./samples/` (looped, paced to real time), or a custom `rtsp://` URL, decoded by the graph with the hardware decoder. |

Detection, tracking, recognition and event logging are identical for every source. The choice lasts until the pipeline container restarts, then it falls back to `SOURCE_URI`. The source endpoint only accepts `rtsp(s)://` URLs and video files under `/samples`.

### Default source

Set `SOURCE_URI` in `.env` (copy `.env.example`):

| Source               | `SOURCE_URI`                                       |
|----------------------|----------------------------------------------------|
| Real RTSP camera     | `rtsp://user:pass@192.168.1.10:554/stream` (H.264) |
| Video file (looped)  | `/samples/demo.mp4` (and drop `--profile sim`)     |

**Windows or macOS webcam**: Docker cannot see the host webcam. Use the *This device (browser)* option, or run `scripts\publish-webcam.bat ["Camera name"]` on the host (needs ffmpeg) to publish it to the bundled RTSP server (`--profile sim`) and choose *Host webcam via RTSP*.

## How recognition works

- **Detection**: SCRFD. **Embedding**: ArcFace (512-d, L2-normalised). Both come from insightface's `buffalo_l` pack and run on TensorRT.
- **Alignment**: a least-squares similarity transform from the five landmarks onto the 112x112 ArcFace template (the method insightface uses), applied on the GPU. Aligning on the landmarks rather than cropping the box keeps a tilted head recognisable: on test faces rotated by 30 degrees the same-person similarity stays around 0.97.
- **Matching**: cosine similarity in pgvector (`<=>`, HNSW index). A face is *known* if the similarity is at least `MATCH_THRESHOLD` (default 0.40), otherwise *Unknown*. Each person is matched on their best stored photo.
- **Tracking**: `nvtracker` keeps one id per face. The database is queried every `ID_INTERVAL` frames (5) while a face is unresolved and every `REID_INTERVAL` frames (45) once known. Each appearance produces **one** event, logged again after `EVENT_COOLDOWN` seconds if the person stays in view, or at once if their status changes.
- **Events** are posted by the pipeline to the backend, stored in PostgreSQL (timestamp, person or Unknown, confidence, camera, snapshot path, evidence) and pushed to browsers over `WS /ws/events`.
- **Files**: snapshots and registered images are stored on the `appdata` volume and served at `/files/...`.
- **Embedding versions**: every stored vector carries the id of the model that produced it (`MODEL_ID`, currently `buffalo_l-aligned`) and is only compared with vectors of the same id. After a model change, `POST /api/admin/reanalyze` re-embeds all stored photos and keeps the old vectors.

## Registering faces

On **Faces** there are three ways to add a person, and all of them produce the same *registration report*:

| Way | Notes |
|---|---|
| **Photos** | Upload one or more photos, or capture a frame from the live stream. The largest face in each is used. |
| **Open camera** | Uses the camera of this device. *Guided 5 angles* asks for straight, left, right, up and down with live feedback (one face, size, light, pose) and optional auto-capture; *Free capture* takes any number of shots. |
| **Group photo** | Every face is boxed and numbered: click one, or drag a box around a face the detector missed. One person per registration. |

Before saving, the system flags likely mistakes, which can be confirmed and overridden: the face is almost identical to a *different* registered person, photos uploaded together look like different people, or a photo added to a person does not look like them (`DUP_THRESHOLD`, `CONSISTENCY_THRESHOLD`).

The **registration report** shows the chosen face and its landmarks, the aligned 112x112 face the model uses, quality checks (size, detection confidence, sharpness, light, head turn), consistency between the person's photos, the most similar other person, and what is stored. After a guided capture it also shows how similar each angle is to the straight photo, so you can see whether extra angles help.

## Explaining decisions

* **What the system saw**: click an event on **Live**, or *Why* in **History**. It shows the detected face and landmarks, the aligned face, quality, the closest people with their scores against the threshold and the margin to the runner-up.
* **Explain overlay**: a switch above the live video draws the five landmarks, the detection score and `similarity / threshold` on every face.
* **Which regions matter**: *Explain regions* hides one patch of the face at a time and measures how much the score drops, for recognition and for detection. It runs on demand on the stored images and is cached. It describes the model's sensitivity, not human reasoning.

## API

Interactive docs: <http://localhost:8000/docs>

| Method and path | Purpose |
|---|---|
| `GET /api/persons?q=` | List or search registered people |
| `POST /api/persons` (multipart) | Register: `full_name`, `person_code`, `notes`, `images[]` |
| `POST /api/persons/{id}/faces` | Add face images to a person |
| `DELETE /api/persons/{id}` | Delete a person, their embeddings and images (history is kept) |
| `GET /api/persons/{id}/report` | Registration report |
| `GET /api/events?q=&status=&from=&to=&page=&page_size=` | Paginated history |
| `GET /api/events/{id}`, `GET /api/events/{id}/explain` | Evidence for an event; region heatmaps |
| `WS /ws/events` | Live event push |
| `GET /api/stream.mjpg`, `/api/snapshot.jpg`, `/api/status` | Video, clean frame, pipeline status |
| `POST /api/faces/analyze` (optional `box`) | Faces in a photo, or the one in a box, with quality and closest people |
| `POST /api/faces/probe`, `POST /api/faces/compare` | Live camera feedback; similarity between several shots |
| `PUT /api/overlay` | Toggle the explain overlay on the live video |
| `GET /api/admin/model-status`, `POST /api/admin/reanalyze` | Embedding status; re-embed stored photos |

## Performance

On a laptop with an RTX 3050 (4 GB), 1280x720 input and six faces in view, the whole graph (decode, detection, tracking, alignment and embedding of every face on every frame, drawing, JPEG encoding) runs at about 50 frames per second from a file read as fast as possible, about twice the rate of a real camera. The first start builds the TensorRT engines (about 4 minutes); later starts load them in about 2 seconds.

## Configuration

All settings are optional; see `.env.example`. Pipeline tuning (`MATCH_THRESHOLD`, `DET_THRESH`, `MIN_FACE_PX`, `ID_INTERVAL`, `REID_INTERVAL`, `EVENT_COOLDOWN`, ...) is read in `pipeline/app/config.py`; the DeepStream element settings are in `pipeline/configs/`.

## Notes and limitations

- An NVIDIA GPU is required, and the DeepStream base image is large (about 20 GB).
- Alignment corrects head tilt within the image plane; a head turned to the side is helped by registering a few angles, which the guided camera capture does.
- RTSP cameras must send H.264. One camera per pipeline container.
- Tables are created when the backend starts, and there is no migration tool yet.
- The API has no authentication and is meant for a trusted network. The pipeline reports events to the backend with a shared secret (`INTERNAL_TOKEN`).

## Project layout

```
backend/      FastAPI app: models, routers, WebSocket, client for the pipeline service
pipeline/     DeepStream service
  app/          live graph (runner.py, analytics.py), still-image graph (still_graph.py, engine.py,
                picture.py, stills.py, explain.py), SCRFD decoder, matching, events
  configs/      nvinfer, nvtracker and nvdspreprocess settings
  face_align/   face alignment library for nvdspreprocess (C++ and CUDA)
  patches/      small change applied to NVIDIA's gst-nvinfer source at build time
frontend/     Next.js UI: live, faces, history
docs/         user guide, architecture figure, exported GStreamer graphs of both pipelines
samples/      put demo.mp4 here
```
