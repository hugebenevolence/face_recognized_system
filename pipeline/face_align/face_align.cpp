/*
 * Face alignment for ArcFace, as an nvdspreprocess custom library.
 *
 * Everything it needs is standard DeepStream metadata: the face objects (unit.roi_meta) and the
 * raw output of the SCRFD nvinfer, which nvinfer attaches to each frame because the detector
 * runs with output-tensor-meta=1 (NVDSINFER_TENSOR_OUTPUT_META in frame_user_meta_list).
 * For every face object the library
 *   1. decodes SCRFD's five landmarks for that frame and takes the detection that matches the
 *      object's box,
 *   2. fits the least-squares similarity transform (rotation, uniform scale, translation) from
 *      the landmarks onto the ArcFace 112x112 template, the transform insightface's norm_crop uses,
 *   3. warps the face out of the RGBA frame on the GPU with it (face_align_kernel.cu) and writes
 *      the normalised tensor that the ArcFace nvinfer (input-tensor-meta=1) reads,
 *   4. attaches the aligned face itself (112x112 RGB, the image ArcFace sees) to the object as a
 *      user meta of type "TNI.FACE_ALIGN.CHIP" ({int32 width, int32 height, RGB bytes}), so the
 *      application can keep it as evidence without touching the frame.
 * Objects of the "aligned face" class (already aligned 112x112 faces given to the still-image
 * graph) and faces without a matching detection are scaled from their box instead, like a
 * regular secondary nvinfer would do.
 *
 * user-configs: pixel-normalization-factor, offsets (ArcFace normalisation), detector-unique-id,
 * detector-threshold, detector-input-size (SCRFD), aligned-class-id.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "nvbufsurftransform.h"
#include "nvdspreprocess_interface.h"  // brings in GStreamer and the DeepStream meta headers
#include "gstnvdsinfer.h"              // NvDsInferTensorMeta (needs the headers above)
#include "face_align_kernel.h"

namespace {

/* ArcFace landmark template inside the 112x112 input: eyes, nose tip, mouth corners. */
const float kTemplate[5][2] = {{38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
                               {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};
const int kStrides[3] = {8, 16, 32};
const int kAnchorsPerCell = 2;
const float kMatchIou = 0.4f;

struct Detection {
  float box[4];  // x1 y1 x2 y2, frame pixels
  float kps[5][2];
  float score;
};

float iou(const float a[4], const float b[4]) {
  float ix = std::max(0.f, std::min(a[2], b[2]) - std::max(a[0], b[0]));
  float iy = std::max(0.f, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  float inter = ix * iy;
  float uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return uni > 0 ? inter / uni : 0.f;
}

}  // namespace

struct CustomCtx {
  CustomInitParams init;
  cudaStream_t stream = nullptr;
  int out_w = 112, out_h = 112;
  float scale = 0.0078125f;
  float offset[3] = {127.5f, 127.5f, 127.5f};
  unsigned char *chips_dev = nullptr;   // aligned faces of one batch, 8-bit RGB (device)
  unsigned char *chips_host = nullptr;  // same, pinned host copy
  int max_units = 0;
  NvDsMetaType chip_type = NVDS_INVALID_META;
  unsigned detector_id = 1;
  float detector_threshold = 0.5f;
  int detector_size = 640;
  int aligned_class = 1;  // class id of objects that are already aligned faces
  bool debug = std::getenv("FACE_ALIGN_DEBUG") != nullptr;  // per-face log for troubleshooting
};

/* SCRFD's raw output for one frame -> detections above the threshold (before NMS; the object
 * boxes already went through it, here we only need the landmarks that belong to each box).
 * Output layers are told apart by their shape: (anchors, 1) scores, (anchors, 4) box distances,
 * (anchors, 10) landmark offsets, for each stride. */
static std::vector<Detection> decode_scrfd(const CustomCtx *ctx, const NvDsFrameMeta *fm, int frame_w,
                                           int frame_h) {
  std::vector<Detection> out;
  for (NvDsMetaList *l = fm->frame_user_meta_list; l != nullptr; l = l->next) {
    auto *um = static_cast<NvDsUserMeta *>(l->data);
    if (um->base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;
    auto *tm = static_cast<NvDsInferTensorMeta *>(um->user_meta_data);
    if (tm->unique_id != ctx->detector_id) continue;

    // (rows, cols) -> host data
    std::unordered_map<long long, const float *> layers;
    for (unsigned i = 0; i < tm->num_output_layers; ++i) {
      const NvDsInferLayerInfo &info = tm->output_layers_info[i];
      long long rows = info.inferDims.d[0];
      if (rows <= 0) continue;
      long long cols = info.inferDims.numElements / rows;
      layers[rows * 100 + cols] = static_cast<const float *>(tm->out_buf_ptrs_host[i]);
    }
    const int net = ctx->detector_size;
    const float s = std::min(float(net) / frame_w, float(net) / frame_h);  // letterbox at top-left
    for (int stride : kStrides) {
      const int cells = net / stride;
      const long long n = 1ll * cells * cells * kAnchorsPerCell;
      auto sc = layers.find(n * 100 + 1), bb = layers.find(n * 100 + 4), kp = layers.find(n * 100 + 10);
      if (sc == layers.end() || bb == layers.end() || kp == layers.end()) continue;
      for (long long a = 0; a < n; ++a) {
        float score = sc->second[a];
        if (score < ctx->detector_threshold) continue;
        const long long cell = a / kAnchorsPerCell;
        const float cx = float(cell % cells) * stride, cy = float(cell / cells) * stride;
        const float *d = bb->second + a * 4, *k = kp->second + a * 10;
        Detection det;
        det.score = score;
        det.box[0] = (cx - d[0] * stride) / s;
        det.box[1] = (cy - d[1] * stride) / s;
        det.box[2] = (cx + d[2] * stride) / s;
        det.box[3] = (cy + d[3] * stride) / s;
        for (int p = 0; p < 5; ++p) {
          det.kps[p][0] = (cx + k[2 * p] * stride) / s;
          det.kps[p][1] = (cy + k[2 * p + 1] * stride) / s;
        }
        out.push_back(det);
      }
    }
    break;
  }
  return out;
}

/* Least-squares similarity transform src -> template, as [a -b tx; b a ty] (the closed form of
 * Umeyama's method in 2D, without reflection). Returns the inverse map (output -> frame). */
static bool alignment_inverse(const float kps[5][2], float inv[6]) {
  float ms[2] = {0, 0}, md[2] = {0, 0};
  for (int p = 0; p < 5; ++p) {
    ms[0] += kps[p][0] / 5;
    ms[1] += kps[p][1] / 5;
    md[0] += kTemplate[p][0] / 5;
    md[1] += kTemplate[p][1] / 5;
  }
  float num_a = 0, num_b = 0, den = 0;
  for (int p = 0; p < 5; ++p) {
    float sx = kps[p][0] - ms[0], sy = kps[p][1] - ms[1];
    float dx = kTemplate[p][0] - md[0], dy = kTemplate[p][1] - md[1];
    num_a += sx * dx + sy * dy;
    num_b += sx * dy - sy * dx;
    den += sx * sx + sy * sy;
  }
  if (den < 1e-6f) return false;
  const float a = num_a / den, b = num_b / den;
  const float tx = md[0] - (a * ms[0] - b * ms[1]), ty = md[1] - (b * ms[0] + a * ms[1]);
  const float det = a * a + b * b;  // forward [a -b; b a] -> inverse [a b; -b a] / det
  if (det < 1e-12f) return false;
  inv[0] = a / det;
  inv[1] = b / det;
  inv[3] = -b / det;
  inv[4] = a / det;
  inv[2] = -(inv[0] * tx + inv[1] * ty);
  inv[5] = -(inv[3] * tx + inv[4] * ty);
  return true;
}

/* Fallback: the object's box scaled to the output (what nvinfer itself would crop). */
static void box_inverse(const NvOSD_RectParams &r, int out_w, int out_h, float inv[6]) {
  inv[0] = r.width / out_w;
  inv[1] = 0;
  inv[2] = r.left;
  inv[3] = 0;
  inv[4] = r.height / out_h;
  inv[5] = r.top;
}

/* User meta holding one aligned face: {int32 width, int32 height, width*height*3 RGB bytes}. */
static gpointer chip_copy(gpointer data, gpointer) {
  auto *um = static_cast<NvDsUserMeta *>(data);
  const int *hdr = static_cast<const int *>(um->user_meta_data);
  size_t size = 2 * sizeof(int) + (size_t)hdr[0] * hdr[1] * 3;
  void *copy = g_malloc(size);
  std::memcpy(copy, um->user_meta_data, size);
  return copy;
}

static void chip_release(gpointer data, gpointer) {
  auto *um = static_cast<NvDsUserMeta *>(data);
  g_free(um->user_meta_data);
  um->user_meta_data = nullptr;
}

static void attach_chip(CustomCtx *ctx, NvDsObjectMeta *obj, const unsigned char *rgb) {
  NvDsBatchMeta *bm = obj->base_meta.batch_meta;
  if (bm == nullptr) return;
  const size_t pixels = (size_t)ctx->out_w * ctx->out_h * 3;
  int *hdr = static_cast<int *>(g_malloc(2 * sizeof(int) + pixels));
  hdr[0] = ctx->out_w;
  hdr[1] = ctx->out_h;
  std::memcpy(hdr + 2, rgb, pixels);
  nvds_acquire_meta_lock(bm);
  NvDsUserMeta *um = nvds_acquire_user_meta_from_pool(bm);
  um->user_meta_data = hdr;
  um->base_meta.meta_type = ctx->chip_type;
  um->base_meta.copy_func = chip_copy;
  um->base_meta.release_func = chip_release;
  um->base_meta.batch_meta = bm;
  nvds_add_user_meta_to_obj(obj, um);
  nvds_release_meta_lock(bm);
}

extern "C" {

NvDsPreProcessStatus CustomTensorPreparation(CustomCtx *ctx, NvDsPreProcessBatch *batch,
                                             NvDsPreProcessCustomBuf *&buf,
                                             CustomTensorParams &tensorParam,
                                             NvDsPreProcessAcquirer *acquirer) {
  buf = acquirer->acquire();
  float *out = static_cast<float *>(buf->memory_ptr);
  const size_t per_face = 3ul * ctx->out_w * ctx->out_h;
  std::unordered_map<const NvDsFrameMeta *, std::vector<Detection>> frames;  // decoded once per frame
  const size_t chip_bytes = (size_t)ctx->out_w * ctx->out_h * 3;
  if ((int)batch->units.size() > ctx->max_units) {  // grow the chip buffers when needed
    cudaFree(ctx->chips_dev);
    cudaFreeHost(ctx->chips_host);
    ctx->max_units = (int)batch->units.size();
    cudaMalloc(&ctx->chips_dev, chip_bytes * ctx->max_units);
    cudaMallocHost(&ctx->chips_host, chip_bytes * ctx->max_units);
  }
  std::vector<NvDsObjectMeta *> chip_objs(batch->units.size(), nullptr);

  for (size_t i = 0; i < batch->units.size(); ++i) {
    const NvDsPreProcessUnit &u = batch->units[i];
    // For objects nvdspreprocess leaves unit.obj_meta empty and keeps the object in roi_meta.
    const NvDsObjectMeta *obj = u.obj_meta ? u.obj_meta : u.roi_meta.object_meta;
    const NvDsFrameMeta *fm = u.frame_meta ? u.frame_meta : u.roi_meta.frame_meta;
    const NvBufSurfaceParams *s = u.input_surf_params;
    float *dst = out + i * per_face;
    if (s == nullptr || obj == nullptr || fm == nullptr || s->colorFormat != NVBUF_COLOR_FORMAT_RGBA) {
      cudaMemsetAsync(dst, 0, per_face * sizeof(float), ctx->stream);
      continue;
    }
    const NvOSD_RectParams &r = obj->rect_params;
    const Detection *best = nullptr;
    if (obj->class_id != ctx->aligned_class) {
      auto it = frames.find(fm);
      if (it == frames.end()) it = frames.emplace(fm, decode_scrfd(ctx, fm, s->width, s->height)).first;
      const float box[4] = {r.left, r.top, r.left + r.width, r.top + r.height};
      for (const Detection &d : it->second)
        if (iou(box, d.box) >= kMatchIou && (best == nullptr || d.score > best->score)) best = &d;
    }
    float inv[6];
    if (best == nullptr || !alignment_inverse(best->kps, inv)) box_inverse(r, ctx->out_w, ctx->out_h, inv);
    if (ctx->debug)
      std::printf("face_align: frame %d object %lu class %d %s\n", fm->frame_num,
                  (unsigned long)obj->object_id, obj->class_id, best ? "aligned on landmarks" : "scaled from box");
    fa_launch(static_cast<const unsigned char *>(s->dataPtr), (int)s->pitch, (int)s->width, (int)s->height,
              inv, dst, ctx->chips_dev + i * chip_bytes, ctx->out_w, ctx->out_h, ctx->scale, ctx->offset,
              ctx->stream);
    chip_objs[i] = const_cast<NvDsObjectMeta *>(obj);
  }
  if (!batch->units.empty())
    cudaMemcpyAsync(ctx->chips_host, ctx->chips_dev, chip_bytes * batch->units.size(), cudaMemcpyDeviceToHost,
                    ctx->stream);
  cudaError_t err = cudaStreamSynchronize(ctx->stream);
  if (err == cudaSuccess) err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::printf("face_align: CUDA error %s\n", cudaGetErrorString(err));
    acquirer->release(buf);
    return NVDSPREPROCESS_CUSTOM_TENSOR_FAILED;
  }
  for (size_t i = 0; i < chip_objs.size(); ++i)
    if (chip_objs[i] != nullptr) attach_chip(ctx, chip_objs[i], ctx->chips_host + i * chip_bytes);
  tensorParam.params.network_input_shape[0] = (int)batch->units.size();
  return NVDSPREPROCESS_SUCCESS;
}

/* The plugin always scales each object into its scaling pool first. The tensor above is built
 * from the full frame instead, so this only keeps the plugin's bookkeeping intact (same as the
 * NVIDIA sample). */
NvDsPreProcessStatus CustomAsyncTransformation(NvBufSurface *in_surf, NvBufSurface *out_surf,
                                               CustomTransformParams &params) {
  NvBufSurfTransform_Error err = NvBufSurfTransformSetSessionParams(&params.transform_config_params);
  if (err != NvBufSurfTransformError_Success) return NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
  err = NvBufSurfTransformAsync(in_surf, out_surf, &params.transform_params, &params.sync_obj);
  return err == NvBufSurfTransformError_Success ? NVDSPREPROCESS_SUCCESS
                                                 : NVDSPREPROCESS_CUSTOM_TRANSFORMATION_FAILED;
}

CustomCtx *initLib(CustomInitParams initparams) {
  auto ctx = std::make_unique<CustomCtx>();
  ctx->init = initparams;
  const auto &shape = initparams.tensor_params.network_input_shape;  // N;C;H;W
  if (shape.size() == 4) {
    ctx->out_h = shape[2];
    ctx->out_w = shape[3];
  }
  auto &uc = initparams.user_configs;
  if (!uc["pixel-normalization-factor"].empty()) ctx->scale = std::stof(uc["pixel-normalization-factor"]);
  std::string offs = uc["offsets"];
  for (int k = 0; k < 3 && !offs.empty(); ++k) {
    size_t p = offs.find(';');
    ctx->offset[k] = std::stof(offs.substr(0, p));
    offs = p == std::string::npos ? "" : offs.substr(p + 1);
  }
  if (!uc["detector-unique-id"].empty()) ctx->detector_id = std::stoul(uc["detector-unique-id"]);
  if (!uc["detector-threshold"].empty()) ctx->detector_threshold = std::stof(uc["detector-threshold"]);
  if (!uc["detector-input-size"].empty()) ctx->detector_size = std::stoi(uc["detector-input-size"]);
  if (!uc["aligned-class-id"].empty()) ctx->aligned_class = std::stoi(uc["aligned-class-id"]);
  if (cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking) != cudaSuccess) return nullptr;
  ctx->chip_type = nvds_get_user_meta_type(const_cast<gchar *>("TNI.FACE_ALIGN.CHIP"));
  std::printf("face_align: ready, output %dx%d, landmarks from nvinfer unique-id %u\n", ctx->out_w,
              ctx->out_h, ctx->detector_id);
  return ctx.release();
}

void deInitLib(CustomCtx *ctx) {
  if (ctx == nullptr) return;
  if (ctx->stream) cudaStreamDestroy(ctx->stream);
  cudaFree(ctx->chips_dev);
  cudaFreeHost(ctx->chips_host);
  delete ctx;
}

}  // extern "C"
