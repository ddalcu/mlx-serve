threadgroup float vals[uint(TOPK) * 128u];
uint block = uint(threadgroup_position_in_grid.x);
uint row = uint(threadgroup_position_in_grid.y);
uint sg = uint(simdgroup_index_in_threadgroup);
ushort lane = thread_index_in_simdgroup;
const uint base = block * 128u;
const uint slot = row * uint(TOPK) + sg;
const uint eid = uint(slots[slot]);
const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
const float scv = 0.08838834764831845f;
float4 v = float4(float(inner[xb + lane]), float(inner[xb + lane + 32u]), float(inner[xb + lane + 64u]), float(inner[xb + lane + 96u]));
for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
  const float p0 = simd_shuffle_xor(v.x, bit);
  const float p1 = simd_shuffle_xor(v.y, bit);
  const float p2 = simd_shuffle_xor(v.z, bit);
  const float p3 = simd_shuffle_xor(v.w, bit);
  const bool lower = (lane & bit) == 0u;
  v.x = lower ? v.x + p0 : p0 - v.x;
  v.y = lower ? v.y + p1 : p1 - v.y;
  v.z = lower ? v.z + p2 : p2 - v.z;
  v.w = lower ? v.w + p3 : p3 - v.w;
}
const float s0 = v.x + v.y;
const float s1 = v.x - v.y;
const float s2 = v.z + v.w;
const float s3 = v.z - v.w;
vals[sg * 128u + lane] = (s0 + s2) * scv * float(svh[sb + lane]);
vals[sg * 128u + lane + 32u] = (s1 + s3) * scv * float(svh[sb + lane + 32u]);
vals[sg * 128u + lane + 64u] = (s0 - s2) * scv * float(svh[sb + lane + 64u]);
vals[sg * 128u + lane + 96u] = (s1 - s3) * scv * float(svh[sb + lane + 96u]);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0u) {
  float a0 = 0.0f;
  float a1 = 0.0f;
  float a2 = 0.0f;
  float a3 = 0.0f;
  for (uint k = 0u; k < uint(TOPK); k++) {
    const float w = float(sc[row * uint(TOPK) + k]);
    a0 += vals[k * 128u + lane] * w;
    a1 += vals[k * 128u + lane + 32u] * w;
    a2 += vals[k * 128u + lane + 64u] * w;
    a3 += vals[k * 128u + lane + 96u] * w;
  }
  const size_t yb = (size_t)row * (size_t)(ODIM) + base;
  y[yb + lane] = T(a0);
  y[yb + lane + 32u] = T(a1);
  y[yb + lane + 64u] = T(a2);
  y[yb + lane + 96u] = T(a3);
}