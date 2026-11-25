/* CUDA part of face_align: the warp kernel (kept apart from the DeepStream and glib headers,
 * which nvcc cannot parse). */
#include <cuda_runtime.h>

#include "face_align_kernel.h"

/* One thread per output pixel: inverse-map into the frame, bilinear sample (zero outside, like
 * insightface's norm_crop), then write
 *   - the normalised planar RGB tensor (value - offset) * scale that ArcFace reads, and
 *   - optionally the same aligned face as 8-bit interleaved RGB (the image ArcFace sees, kept as
 *     evidence and shown in the interface). */
__global__ void warp_normalise(const unsigned char *src, int pitch, int src_w, int src_h,
                               float a, float b, float c, float d, float e, float f,
                               float *out, unsigned char *rgb, int out_w, int out_h, float scale,
                               float o0, float o1, float o2) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= out_w || y >= out_h) return;

  float sx = a * x + b * y + c;
  float sy = d * x + e * y + f;
  int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
  float fx = sx - x0, fy = sy - y0;

  float acc[3] = {0.f, 0.f, 0.f};
  const float w[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
  const int px[4] = {x0, x0 + 1, x0, x0 + 1};
  const int py[4] = {y0, y0, y0 + 1, y0 + 1};
  for (int k = 0; k < 4; ++k) {
    if (px[k] < 0 || py[k] < 0 || px[k] >= src_w || py[k] >= src_h) continue;
    const unsigned char *p = src + (size_t)py[k] * pitch + (size_t)px[k] * 4;  // RGBA
    acc[0] += w[k] * p[0];
    acc[1] += w[k] * p[1];
    acc[2] += w[k] * p[2];
  }
  size_t plane = (size_t)out_w * out_h, i = (size_t)y * out_w + x;
  out[i] = (acc[0] - o0) * scale;
  out[plane + i] = (acc[1] - o1) * scale;
  out[2 * plane + i] = (acc[2] - o2) * scale;
  if (rgb != nullptr) {
    for (int ch = 0; ch < 3; ++ch) rgb[i * 3 + ch] = (unsigned char)fminf(fmaxf(acc[ch] + 0.5f, 0.f), 255.f);
  }
}

extern "C" void fa_launch(const unsigned char *src, int pitch, int src_w, int src_h, const float inv[6],
                          float *out, unsigned char *rgb, int out_w, int out_h, float scale,
                          const float offset[3], cudaStream_t stream) {
  const dim3 block(16, 16), grid((out_w + 15) / 16, (out_h + 15) / 16);
  warp_normalise<<<grid, block, 0, stream>>>(src, pitch, src_w, src_h, inv[0], inv[1], inv[2], inv[3],
                                             inv[4], inv[5], out, rgb, out_w, out_h, scale, offset[0],
                                             offset[1], offset[2]);
}
