uint block = uint(threadgroup_position_in_grid.x);
uint slot = uint(threadgroup_position_in_grid.y);
ushort lane = thread_index_in_simdgroup;
const uint eid = uint(slots[slot]);
const uint orig = uint(order[slot]);
const uint row = orig / uint(TOPK);
const uint base = block * 128u;
const size_t xb = (size_t)row * (size_t)(IDIM) + base;
const size_t yb = (size_t)slot * (size_t)(IDIM) + base;
const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
const float x0 = float(x[xb + lane]);
const float x1 = float(x[xb + lane + 32u]);
const float x2 = float(x[xb + lane + 64u]);
const float x3 = float(x[xb + lane + 96u]);
float4 v = float4(
  x0 * float(suhg[sb + lane]),
  x1 * float(suhg[sb + lane + 32u]),
  x2 * float(suhg[sb + lane + 64u]),
  x3 * float(suhg[sb + lane + 96u]));
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
const float sc = 0.08838834764831845f;
float s0 = v.x + v.y;
float s1 = v.x - v.y;
float s2 = v.z + v.w;
float s3 = v.z - v.w;
yg[yb + lane] = half((s0 + s2) * sc);
yg[yb + lane + 32u] = half((s1 + s3) * sc);
yg[yb + lane + 64u] = half((s0 - s2) * sc);
yg[yb + lane + 96u] = half((s1 - s3) * sc);
v = float4(
  x0 * float(suhu[sb + lane]),
  x1 * float(suhu[sb + lane + 32u]),
  x2 * float(suhu[sb + lane + 64u]),
  x3 * float(suhu[sb + lane + 96u]));
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
s0 = v.x + v.y;
s1 = v.x - v.y;
s2 = v.z + v.w;
s3 = v.z - v.w;
yu[yb + lane] = half((s0 + s2) * sc);
yu[yb + lane + 32u] = half((s1 + s3) * sc);
yu[yb + lane + 64u] = half((s0 - s2) * sc);
yu[yb + lane + 96u] = half((s1 - s3) * sc);