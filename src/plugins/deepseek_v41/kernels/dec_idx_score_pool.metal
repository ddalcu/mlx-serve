// The compressor's pool step of one KV source and the index key of its owner, one threadgroup (128 threads, 4
// channels a thread) a group the step completes. POOL (ratio 2): the group is rows 2g, 2g+1 of the open group's
// pending rows then the step's, softmax-pooled over the pair in f32; else (ratio 1) it is row g. Then MLX's rms_norm
// (its reduction order), the comp row (rope + fp4 / 16 / e4m3) and the index key (wk, rms_norm, rope + fp4 / 32 /
// ue8m0). An extra threadgroup past the groups keeps the step's last row as the new pending row.
const uint t = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int gi = threadgroup_position_in_grid.x;
const int np = meta[0], g0 = meta[1], R = meta[2];
threadgroup vec<bfloat, 8> lat_v[64];
threadgroup bfloat* lat_s = (threadgroup bfloat*)lat_v;
threadgroup bfloat key_s[128];
threadgroup float sums[32];
threadgroup float inv_s;

#if POOL
if (gi * 2 >= np + R - (np + R) % 2) {
  // The step's last row stays open.
  const device float* last = rows + size_t(R - 1) * 1024;
  *(device float4*)(pend_kv_o + 4 * t) = *(const device float4*)(last + 4 * t);
  *(device float4*)(pend_gate_o + 4 * t) = *(const device float4*)(last + 512 + 4 * t);
  return;
}
float pooled[4];
{
  #pragma clang fp contract(off)
  const int c0 = 2 * gi, c1 = c0 + 1;
  const device float* kv0 = c0 < np ? pend_kv + size_t(c0) * 512 : rows + size_t(c0 - np) * 1024;
  const device float* ga0 = c0 < np ? pend_gate + size_t(c0) * 512 : rows + size_t(c0 - np) * 1024 + 512;
  const device float* kv1 = c1 < np ? pend_kv + size_t(c1) * 512 : rows + size_t(c1 - np) * 1024;
  const device float* ga1 = c1 < np ? pend_gate + size_t(c1) * 512 : rows + size_t(c1 - np) * 1024 + 512;
  const float4 vk0 = *(const device float4*)(kv0 + 4 * t), vk1 = *(const device float4*)(kv1 + 4 * t);
  const float4 vg0 = *(const device float4*)(ga0 + 4 * t), vg1 = *(const device float4*)(ga1 + 4 * t);
  for (ushort i = 0; i < 4; ++i) {
    const float m = metal::max(vg0[i], vg1[i]);
    const float e0 = metal::precise::exp(vg0[i] - m), e1 = metal::precise::exp(vg1[i] - m);
    const float s = e0 + e1;
    const float w0 = e0 / s, w1 = e1 / s;
    const float p0_ = vk0[i] * w0, p1_ = vk1[i] * w1;
    pooled[i] = p0_ + p1_;
  }
}
float x[4];
for (ushort i = 0; i < 4; ++i) x[i] = float(bfloat(pooled[i]));
#else
float x[4];
{
  const device bfloat* row = rows + size_t(gi) * 512 + 4 * t;
  for (ushort i = 0; i < 4; ++i) x[i] = float(row[i]);
}
#endif

// rms_norm over 512, as MLX's rms_single_row sums it: 4 values a thread in order, simd sums, the 4 simdgroups' sums.
float acc = 0.0f;
for (ushort i = 0; i < 4; ++i) acc = fma(x[i], x[i], acc);
acc = simd_sum(acc);
if (sg == 0) sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) sums[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  acc = simd_sum(sums[lane]);
  if (lane == 0) inv_s = metal::precise::rsqrt(acc / 512.0f + eps[0]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inv = inv_s;
bfloat lat[4];
for (ushort i = 0; i < 4; ++i) {
  lat[i] = norm[4 * t + i] * bfloat(x[i] * inv);
  lat_s[4 * t + i] = lat[i];
}

// The comp row: rope on the last 64 channels at the group's first position, fp4 / 16 / e4m3 round trip.
const float pos = float(g0 + gi);
{
  float r[4];
  for (ushort i = 0; i < 4; ++i) r[i] = float(lat[i]);
  if (t >= 112) {
    for (ushort pp = 0; pp < 2; ++pp) {
      const int p = (4 * int(t) - 448) / 2 + pp;
      const float x1 = float(lat[2 * pp]), x2 = float(lat[2 * pp + 1]);
      const float theta = float(RATIO) * pos * (1.0f / periods[p]);
      const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
      r[2 * pp] = float(bfloat(x1 * cs - x2 * sn));
      r[2 * pp + 1] = float(bfloat(x1 * sn + x2 * cs));
    }
  }
  float amax = metal::max(metal::max(metal::abs(r[0]), metal::abs(r[1])), metal::max(metal::abs(r[2]), metal::abs(r[3])));
  amax = metal::max(amax, metal::simd_shuffle_xor(amax, 1));
  amax = metal::max(amax, metal::simd_shuffle_xor(amax, 2));
  const float scale = dsv41_fp8_roundtrip(metal::max(amax, 6.0f * 0.001953125f) / 6.0f);
  for (ushort i = 0; i < 4; ++i) comp[size_t(gi) * 512 + 4 * t + i] = bfloat(dsv41_e2m1(metal::clamp(r[i] / scale, -6.0f, 6.0f)) * scale);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// The index key: wk . latent in f32, rounded to bf16 where the linear leaves it. A simdgroup takes 32 rows of wk, a
// lane 16 channels of each (two 16-byte loads that the simdgroup reads whole), 8 rows at a time.
{
  const device bfloat* wr = wk + size_t(sg * 32) * 512 + lane * 8;
  const vec<bfloat, 8> l0 = *(const threadgroup vec<bfloat, 8>*)(lat_s + lane * 8);
  const vec<bfloat, 8> l1 = *(const threadgroup vec<bfloat, 8>*)(lat_s + 256 + lane * 8);
  for (ushort r0 = 0; r0 < 32; r0 += 8) {
    float part[8];
    #pragma unroll
    for (ushort r = 0; r < 8; ++r) {
      const vec<bfloat, 8> w0 = *(const device vec<bfloat, 8>*)(wr + size_t(r0 + r) * 512);
      const vec<bfloat, 8> w1 = *(const device vec<bfloat, 8>*)(wr + size_t(r0 + r) * 512 + 256);
      float a = 0.0f, b = 0.0f;
      #pragma unroll
      for (ushort i = 0; i < 8; ++i) {
        a = fma(float(w0[i]), float(l0[i]), a);
        b = fma(float(w1[i]), float(l1[i]), b);
      }
      part[r] = a + b;
    }
    #pragma unroll
    for (ushort r = 0; r < 8; ++r) {
      const float total = simd_sum(part[r]);
      if (lane == r) key_s[sg * 32 + r0 + r] = bfloat(total);
    }
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  float y[4];
  float acc2 = 0.0f;
  for (ushort i = 0; i < 4; ++i) {
    y[i] = float(key_s[4 * lane + i]);
    acc2 = fma(y[i], y[i], acc2);
  }
  acc2 = simd_sum(acc2);
  const float inv2 = metal::precise::rsqrt(acc2 / 128.0f + eps[0]);
  bfloat kn[4];
  for (ushort i = 0; i < 4; ++i) kn[i] = knorm[4 * lane + i] * bfloat(y[i] * inv2);
  float r[4];
  for (ushort i = 0; i < 4; ++i) r[i] = float(kn[i]);
  if (lane >= 16) {
    for (ushort pp = 0; pp < 2; ++pp) {
      const int p = (4 * int(lane) - 64) / 2 + pp;
      const float x1 = float(kn[2 * pp]), x2 = float(kn[2 * pp + 1]);
      const float theta = float(RATIO) * pos * (1.0f / periods[p]);
      const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
      r[2 * pp] = float(bfloat(x1 * cs - x2 * sn));
      r[2 * pp + 1] = float(bfloat(x1 * sn + x2 * cs));
    }
  }
  float amax = metal::max(metal::max(metal::abs(r[0]), metal::abs(r[1])), metal::max(metal::abs(r[2]), metal::abs(r[3])));
  amax = metal::max(amax, metal::simd_shuffle_xor(amax, 1));
  amax = metal::max(amax, metal::simd_shuffle_xor(amax, 2));
  amax = metal::max(amax, metal::simd_shuffle_xor(amax, 4));
  const float scale = dsv41_pow2ceil(metal::max(amax, 6.0f * 1.17549435e-38f) * (1.0f / 6.0f));
  for (ushort i = 0; i < 4; ++i) idx[size_t(gi) * 128 + 4 * lane + i] = bfloat(dsv41_e2m1(metal::clamp(r[i] / scale, -6.0f, 6.0f)) * scale);
}
