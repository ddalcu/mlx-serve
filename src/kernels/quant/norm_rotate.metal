constexpr int BLK = 1024;
constexpr int H = NB * BLK;
threadgroup float tg[BLK];
threadgroup float red[4];
uint t = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint sgb = t / 32;
uint blk = threadgroup_position_in_grid.x;
uint row = threadgroup_position_in_grid.y;
const device uint4* a4 = (const device uint4*)(a + row * H);
const device uint4* b4 = (const device uint4*)(b + row * H);
float own[8];
float ss = 0.f;
for (int k = 0; k < NB; ++k) {
  uint4 av = a4[k * 128 + t];
  uint4 bv = HAS_ADD ? b4[k * 128 + t] : uint4(0);
  thread const T* ae = (thread const T*)&av;
  thread const T* be = (thread const T*)&bv;
  for (int q = 0; q < 8; ++q) {
    float e = float(ae[q]);
    if (HAS_ADD) e = float(static_cast<T>(e + float(be[q])));
    ss += e * e;
    if (k == int(blk)) own[q] = e;
  }
}
ss = simd_sum(ss);
if (lane == 0) red[sgb] = ss;
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = metal::precise::rsqrt((red[0] + red[1] + red[2] + red[3]) / H + eps);
uint cb = blk * BLK + 8 * t;
uint rb = row * H + cb;
if (HAS_ADD) { T sv[8]; for (int q = 0; q < 8; ++q) sv[q] = static_cast<T>(own[q]); *(device uint4*)(sum + rb) = *(thread uint4*)sv; }
uint4 wv = *(const device uint4*)(w + cb);
thread const T* we = (thread const T*)&wv;
float4 s0 = *(const device float4*)(signs + cb), s1 = *(const device float4*)(signs + cb + 4);
float sgn8[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
float v[8];
T nv[8];
for (int q = 0; q < 8; ++q) { nv[q] = we[q] * static_cast<T>(own[q] * inv); v[q] = float(nv[q]) * sgn8[q]; }
if (WRITE_NORMED) *(device uint4*)(normed + rb) = *(thread uint4*)nv;
for (int h = 1; h < 8; h <<= 1)
  for (int q = 0; q < 8; ++q) if ((q & h) == 0) { float x0 = v[q], x1 = v[q + h]; v[q] = x0 + x1; v[q + h] = x0 - x1; }
for (uint m = 1; m < 32; m <<= 1) {
  float sgn = (lane & m) ? -1.0f : 1.0f;
  for (int q = 0; q < 8; ++q) v[q] = fma(sgn, v[q], simd_shuffle_xor(v[q], m));
}
for (int q = 0; q < 8; ++q) tg[8 * t + q] = v[q];
threadgroup_barrier(mem_flags::mem_threadgroup);
const float scale = 1.0f / 32.0f;
for (uint base = t; base < 256; base += 128) {
  float q0 = tg[base], q1 = tg[base + 256], q2 = tg[base + 512], q3 = tg[base + 768];
  float a0 = q0 + q1, a1 = q0 - q1, a2 = q2 + q3, a3 = q2 - q3;
  uint ob = row * H + blk * BLK + base;
  rot[ob] = static_cast<T>((a0 + a2) * scale);
  rot[ob + 256] = static_cast<T>((a1 + a3) * scale);
  rot[ob + 512] = static_cast<T>((a0 - a2) * scale);
  rot[ob + 768] = static_cast<T>((a1 - a3) * scale);
}