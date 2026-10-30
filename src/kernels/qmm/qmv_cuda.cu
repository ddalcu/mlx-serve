__shared__ float red[SK][32][4];
const int lane = threadIdx.x, warp = threadIdx.y;
const int gid = lane >> 2, tig = lane & 3;
const int m = x_shape[0], k = x_shape[1], n = w_shape[0];
const int r0 = blockIdx.z * 16 + gid, r1 = r0 + 8;
const int groups = k / GROUP, nblk = k / 128;
const int b0 = warp * nblk / SK, b1 = (warp + 1) * nblk / SK;
T* tag = nullptr;
float c[4] = {0.f, 0.f, 0.f, 0.f};
const bool xrow = gid < m;
const T* xr = x + (size_t)(xrow ? gid : 0) * k;
const uint4 zero = make_uint4(0, 0, 0, 0);
const bool al = ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) & 15) == 0;
for (int blk = b0; blk < b1; ++blk) {
  const int kb = blk * 128 + tig * 32;
  const int g = kb / GROUP;
  uint4 xv[4];
  for (int j = 0; j < 4; ++j) xv[j] = xrow ? msv_ld16(xr + kb + j * 8, al) : zero;
  const uint4 q0 = r0 < n ? msv_ld16(w + ((size_t)r0 * k + kb) / 8, al) : zero;
  const uint4 q1 = r1 < n ? msv_ld16(w + ((size_t)r1 * k + kb) / 8, al) : zero;
  const float s0 = r0 < n ? static_cast<float>(scales[(size_t)r0 * groups + g]) : 0.f;
  const float c0 = r0 < n ? static_cast<float>(biases[(size_t)r0 * groups + g]) : 0.f;
  const float s1 = r1 < n ? static_cast<float>(scales[(size_t)r1 * groups + g]) : 0.f;
  const float c1 = r1 < n ? static_cast<float>(biases[(size_t)r1 * groups + g]) : 0.f;
  const uint32_t* xp = reinterpret_cast<const uint32_t*>(xv);
  const uint32_t* p0 = reinterpret_cast<const uint32_t*>(&q0);
  const uint32_t* p1 = reinterpret_cast<const uint32_t*>(&q1);
#pragma unroll
  for (int s = 0; s < 8; ++s) {
    const uint32_t w0 = p0[s / 2] >> ((s & 1) * 16), w1 = p1[s / 2] >> ((s & 1) * 16);
    float f0[4], f1[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f0[i] = float((w0 >> (4 * i)) & 0xF) * s0 + c0;
      f1[i] = float((w1 >> (4 * i)) & 0xF) * s1 + c1;
    }
    const uint32_t a[4] = {msv_pack2(f0[0], f0[1], tag), msv_pack2(f1[0], f1[1], tag),
                           msv_pack2(f0[2], f0[3], tag), msv_pack2(f1[2], f1[3], tag)};
    const uint32_t b[2] = {xp[s * 2], xp[s * 2 + 1]};
    msv_mma(c, a, b, tag);
  }
}
for (int i = 0; i < 4; ++i) red[warp][lane][i] = c[i];
__syncthreads();
if (warp != 0) return;
float v[4] = {0.f, 0.f, 0.f, 0.f};
for (int sw = 0; sw < SK; ++sw)
  for (int i = 0; i < 4; ++i) v[i] += red[sw][lane][i];
const int col = tig * 2;
for (int e = 0; e < 2; ++e) {
  if (col + e >= m) continue;
  if (r0 < n) out[(size_t)(col + e) * n + r0] = static_cast<T>(v[e]);
  if (r1 < n) out[(size_t)(col + e) * n + r1] = static_cast<T>(v[2 + e]);
}