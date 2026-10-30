const uint lane = thread_index_in_simdgroup;
const uint n0 = thread_position_in_grid.y * RPS;
float tot[RPS], a[RPS];
size_t rows[RPS];
for (int r = 0; r < RPS; ++r) tot[r] = 0.0f;
for (int e = 0; e < TOPK; ++e) {
  for (int r = 0; r < RPS; ++r) rows[r] = size_t(inds[e]) * N + n0 + r;
  dsv41_dots<BITS, GS, K, RPS, false>(w, scales, biases, rows, x + size_t(e) * K, lane, a);
  for (int r = 0; r < RPS; ++r) tot[r] += float(T(a[r]));
}
for (int r = 0; r < RPS; ++r) rows[r] = n0 + r;
dsv41_dots<SBITS, SGS, K, RPS, SMX>(sw, sscales, sbiases, rows, x + size_t(TOPK) * K, lane, a);
if (lane == 0) for (int r = 0; r < RPS; ++r) {
  const uint d = n0 + r;
  const float y = float(T(tot[r] + float(T(a[r]))));
  for (int i = 0; i < HC; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < HC; ++j) acc = metal::fma(comb[j * HC + i], float(res[j * N + d]), acc);
    out[i * N + d] = T(metal::fma(post[i], y, acc));
  }
}