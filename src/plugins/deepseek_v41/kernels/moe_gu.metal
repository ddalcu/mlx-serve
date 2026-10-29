const uint lane = thread_index_in_simdgroup;
const uint n0 = thread_position_in_grid.y * RPS, e = thread_position_in_grid.z;
float a[2 * RPS];
size_t rows[2 * RPS];
const size_t eb = e < TOPK ? size_t(inds[e]) * 2 * N + n0 : n0;
for (int r = 0; r < RPS; ++r) { rows[r] = eb + r; rows[RPS + r] = eb + N + r; }
if (e < TOPK) dsv41_dots<BITS, GS, K, 2 * RPS, false>(w, scales, biases, rows, x, lane, a);
else dsv41_dots<SBITS, SGS, K, 2 * RPS, SMX>(sw, sscales, sbiases, rows, x, lane, a);
const float wt = e < TOPK ? wts[e] : 1.0f;
if (lane == 0) for (int r = 0; r < RPS; ++r) {
  float g = float(T(a[r])), u = float(T(a[RPS + r]));
  if (lim[0] > 0.0f) { g = metal::min(g, lim[0]); u = metal::clamp(u, -lim[0], lim[0]); }
  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
  out[size_t(e) * N + n0 + r] = T((g * (g < 0.0f ? y : 1.0f - y)) * u * wt);
}