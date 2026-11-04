const uint lane = thread_index_in_simdgroup;
const int h = thread_position_in_grid.x / 32, l = thread_position_in_grid.y;
const int H = sink_shape[1];
const int wp = meta[0], window = meta[1], Wk = meta[2], K = meta[3];
const int N = Wk + K;
const float scale = scal[0];
const size_t rw = size_t(l) * H + h;
float s[NJ];
float mx = -INFINITY;
for (int j = 0; j < NJ; ++j) {
  const int c = lane + 32 * j;
  float v = -INFINITY;
  if (c < Wk) { if (c <= wp + l && c + window > wp + l) v = scale * float(lw[rw * Wk + c]); }
  else if (c < N) { if (topk[size_t(l) * K + c - Wk] >= 0) v = scale * float(lc[rw * K + c - Wk]); }
  s[j] = v;
  mx = metal::max(mx, v);
}
mx = metal::simd_max(mx);
const float sk = sink[h];
const float M = metal::max(mx, sk);
float den = 0.0f;
for (int j = 0; j < NJ; ++j) { const float p = metal::exp(s[j] - M); s[j] = p; den += p; }
den = metal::simd_sum(den) + metal::exp(sk - M);
const float inv = 1.0f / den;
for (int j = 0; j < NJ; ++j) {
  const int c = lane + 32 * j;
  if (c < Wk) pw[rw * Wk + c] = OT(s[j] * inv);
  else if (c < N) pc[rw * K + c - Wk] = OT(s[j] * inv);
}