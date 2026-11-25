/*
 * Face alignment for ArcFace as an nvdspreprocess custom library.
 *
 * For every face object, the pipeline stores the 2x3 similarity transform (rotation, scale,
 * translation) that maps the frame onto the ArcFace landmark template in the object's
 * user-defined fields (NvDsObjectMeta.misc_obj_info). This library warps the face out of the
 * RGBA frame on the GPU with that transform and writes the normalised 3x112x112 tensor that
 * nvinfer (input-tensor-meta=1) feeds to ArcFace. It does what insightface's norm_crop does,
 * but inside the DeepStream graph and without leaving GPU memory.
 *
 * misc_obj_info layout: [0..2] six float32 (forward matrix a b c / d e f), [3] = FA_MAGIC.
 * An object without the marker gets an all-zero tensor (the probe ignores its embedding).
 */

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "nvbufsurftransform.h"
#include "nvdspreprocess_interface.h"
#include "face_align_kernel.h"

#define FA_MAGIC 0x46414C4EL /* "FALN" */

struct CustomCtx {
  CustomInitParams init;
  cudaStream_t stream = nullptr;
  int out_w = 112, out_h = 112;
  float scale = 0.0078125f;
  float offset[3] = {127.5f, 127.5f, 127.5f};
  int warned = 0;  // diagnostics printed so far
};

static bool read_matrix(const NvDsObjectMeta *om, float m[6]) {
  if (om == nullptr || om->misc_obj_info[3] != FA_MAGIC) return false;
  std::memcpy(m, om->misc_obj_info, 6 * sizeof(float));
  return std::isfinite(m[0]) && std::isfinite(m[4]);
}

/* Inverse of [a b c; d e f] (the output pixel -> frame pixel mapping). */
static bool invert(const float m[6], float inv[6]) {
  float det = m[0] * m[4] - m[1] * m[3];
  if (std::fabs(det) < 1e-12f) return false;
  inv[0] = m[4] / det;
  inv[1] = -m[1] / det;
  inv[3] = -m[3] / det;
  inv[4] = m[0] / det;
  inv[2] = -(inv[0] * m[2] + inv[1] * m[5]);
  inv[5] = -(inv[3] * m[2] + inv[4] * m[5]);
  return true;
}

extern "C" {

NvDsPreProcessStatus CustomTensorPreparation(CustomCtx *ctx, NvDsPreProcessBatch *batch,
                                             NvDsPreProcessCustomBuf *&buf,
                                             CustomTensorParams &tensorParam,
                                             NvDsPreProcessAcquirer *acquirer) {
  buf = acquirer->acquire();
  float *out = static_cast<float *>(buf->memory_ptr);
  const size_t per_face = 3ul * ctx->out_w * ctx->out_h;

  for (size_t i = 0; i < batch->units.size(); ++i) {
    const NvDsPreProcessUnit &u = batch->units[i];
    /* for objects the plugin leaves unit.obj_meta empty and keeps the object in roi_meta */
    const NvDsObjectMeta *obj = u.obj_meta ? u.obj_meta : u.roi_meta.object_meta;
    float *dst = out + i * per_face;
    const NvBufSurfaceParams *s = u.input_surf_params;
    float m[6], inv[6];
    if (s == nullptr || s->colorFormat != NVBUF_COLOR_FORMAT_RGBA || !read_matrix(obj, m) ||
        !invert(m, inv)) {
      if (ctx->warned < 5) {
        ctx->warned++;
        std::printf("face_align: face %zu not aligned (surface %s, format %d, marker %lx)\n", i,
                    s ? "ok" : "missing", s ? (int)s->colorFormat : -1,
                    obj ? (unsigned long)obj->misc_obj_info[3] : 0ul);
      }
      cudaMemsetAsync(dst, 0, per_face * sizeof(float), ctx->stream);
      continue;
    }
    fa_launch(static_cast<const unsigned char *>(s->dataPtr), (int)s->pitch, (int)s->width, (int)s->height, inv,
              dst, ctx->out_w, ctx->out_h, ctx->scale, ctx->offset, ctx->stream);
  }
  cudaError_t err = cudaStreamSynchronize(ctx->stream);
  if (err == cudaSuccess) err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::printf("face_align: CUDA error %s\n", cudaGetErrorString(err));
    acquirer->release(buf);
    return NVDSPREPROCESS_CUSTOM_TENSOR_FAILED;
  }
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
  if (cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking) != cudaSuccess) return nullptr;
  std::printf("face_align: ready, output %dx%d, scale %g, offsets %g %g %g\n", ctx->out_w, ctx->out_h,
              ctx->scale, ctx->offset[0], ctx->offset[1], ctx->offset[2]);
  return ctx.release();
}

void deInitLib(CustomCtx *ctx) {
  if (ctx == nullptr) return;
  if (ctx->stream) cudaStreamDestroy(ctx->stream);
  delete ctx;
}

}  // extern "C"
