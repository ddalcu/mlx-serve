const int l = threadgroup_position_in_grid.y;
const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
const int D = x_shape[2];
constexpr int MIX = (2 + HC) * HC;
const float e = eps[1];
if (sg == 0) {
  // Lane a * HC + b holds comb[a][b]; each sum gathers its HC terms in the serial order.
  const device float* zr = z + l * MIX;
  if (lane < HC) {
    pre[l * HC + lane] = 1.0f / (1.0f + metal::exp(-zr[lane])) + e;
    post[l * HC + lane] = 2.0f / (1.0f + metal::exp(-zr[HC + lane]));
  }
  const bool act = lane < HC * HC;
  const ushort a = act ? lane / HC : 0, b = act ? lane % HC : 0;
  float v = zr[2 * HC + a * HC + b];
  float mx = metal::simd_shuffle(v, a * HC);
  for (ushort k = 1; k < HC; ++k) mx = metal::max(mx, metal::simd_shuffle(v, a * HC + k));
  v = metal::exp(v - mx);
  float sm = 0.0f;
  for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, a * HC + k);
  v = v / sm + e;
  for (int t = 0; t < 2 * ITERS - 1; ++t) {
    const bool rows = t % 2 == 1;
    sm = 0.0f;
    for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, rows ? a * HC + k : k * HC + b);
    v = v / (sm + e);
  }
  if (act) comb[l * HC * HC + lane] = v;
}
float cp[HC];
for (int j = 0; j < HC; ++j) cp[j] = cpre[l * HC + j];
float ss = 0.0f;
for (int d = tid; d < D; d += TG) {
  float h = 0.0f;
  for (int j = 0; j < HC; ++j) h += cp[j] * float(x[(size_t(l) * HC + j) * D + d]);
  const T hb = T(h);
  h_out[size_t(l) * D + d] = hb;
  ss += float(hb) * float(hb);
}
threadgroup float t_s[TG / 32];
ss = metal::simd_sum(ss);
if (lane == 0) t_s[sg] = ss;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
float tot = 0.0f;
for (int g = 0; g < TG / 32; ++g) tot += t_s[g];
const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
for (int d = tid; d < D; d += TG) {
  const size_t o = size_t(l) * D + d;
  h_out[o] = w[d] * T(float(h_out[o]) * inv);
}