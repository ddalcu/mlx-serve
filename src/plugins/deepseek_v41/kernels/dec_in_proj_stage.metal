// dec_in_proj_stage: h = rms(collapse(x, cpre), w) of one row a threadgroup of D / 8 threads, rounded to bf16 where hc_stage rounds it.
constexpr int HC = 4, NT = D / 8, NSG = NT / 32;
const uint t = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int r = threadgroup_position_in_grid.x;
threadgroup float ps[NSG];
float v[HC][8], cp[HC];
#pragma unroll
for (int j = 0; j < HC; ++j) {
  cp[j] = cpre[r * HC + j];
  const uint4 xv = *(const device uint4*)(x + (size_t(r) * HC + j) * D + 8 * t);
  const uint xw[4] = {xv.x, xv.y, xv.z, xv.w};
  #pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float2 u = dsv41_bf2(xw[q]);
    v[j][2 * q] = u.x;
    v[j][2 * q + 1] = u.y;
  }
}
const uint4 nv = *(const device uint4*)(w + 8 * t);
float hb[8], s = 0.0f;
#pragma unroll
for (int k = 0; k < 8; ++k) {
  float h8 = 0.0f;
  #pragma unroll
  for (int j = 0; j < HC; ++j) h8 += cp[j] * v[j][k];
  hb[k] = float(bfloat16_t(h8));
  s += hb[k] * hb[k];
}
s = simd_sum(s);
if (lane == 0) ps[sg] = s;
threadgroup_barrier(mem_flags::mem_threadgroup);
float tot = 0.0f;
for (int g = 0; g < NSG; ++g) tot += ps[g];
const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
const uint nw[4] = {nv.x, nv.y, nv.z, nv.w};
uint o[4];
#pragma unroll
for (int q = 0; q < 4; ++q) {
  const float2 w2 = dsv41_bf2(nw[q]);
  const float a = float(bfloat16_t(bfloat16_t(w2.x) * bfloat16_t(hb[2 * q] * inv)));
  const float b = float(bfloat16_t(bfloat16_t(w2.y) * bfloat16_t(hb[2 * q + 1] * inv)));
  o[q] = (as_type<uint>(a) >> 16) | (as_type<uint>(b) & 0xFFFF0000u);
}
*(device uint4*)(h + size_t(r) * D + 8 * t) = uint4(o[0], o[1], o[2], o[3]);
