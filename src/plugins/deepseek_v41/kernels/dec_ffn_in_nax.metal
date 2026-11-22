// dec_ffn_in, launch 1 on the matrix units (`MLX_SERVE_DSV41_GEMV_R26_FFN_IN`): wo_b for R rows, then each output
// column's hc expand and collapse, as dec_ffn_in.metal's epilogue. A threadgroup owns 32 output columns; its NSG
// simdgroups split K, each contracting its share of the 32 weight rows against every row of oa at once
// (`dsv41_nax_mxfp8`), so the rows cost the vector kernel's per-row unpack and FMAs no more. The simdgroups' sums meet
// in threadgroup memory.
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int tg = int(threadgroup_position_in_grid.x);
const int tid = int(thread_index_in_threadgroup);
constexpr int KW = K / 4, KG = K / 32, KS = K / NSG, NTH = NSG * 32;
const short qid = short(lane >> 2);
const short fm = (qid & 4) | short((lane >> 1) & 3);
const short f4 = (qid & 2) | short(lane & 1);
const int n0 = tg * 32;
dsv41_nax_rows rows;
for (int j = 0; j < 4; ++j) {
  const uint row = uint(n0 + fm + 8 * j);
  rows.w[j] = (const device uint4*)w_q + row * (KW / 4) + int(sg) * (KS / 16) + f4;
  rows.s[j] = scales + row * KG + int(sg) * (KS / 32) + (f4 >> 1);
}
const dsv41_nax_first first = dsv41_nax_head(rows);
const bool live = int(fm) < R;
float acc[16];
for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
const device uint4* xp = (const device uint4*)(oa + (size_t)(live ? int(fm) : 0) * K) + int(sg) * (KS / 8) + f4 * 2;
dsv41_nax_mxfp8<KS / 64, 4, 2, 8, FOLD>(acc, rows, first, xp, live);

threadgroup float red[NSG * 8 * 32];
if (live) for (int i = 0; i < 4; ++i) {
  red[(int(sg) * 8 + fm) * 32 + f4 * 4 + i] = acc[i];
  red[(int(sg) * 8 + fm) * 32 + 16 + f4 * 4 + i] = acc[8 + i];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// The arrival counters of launch 2.
if (tg == 0 && tid < 16) ctr[tid] = 0u;
for (int it = tid; it < R * 32; it += NTH) {
  const int r = it / 32, c = it % 32;
  const int n = n0 + c;
  float y = 0.0f;
  for (int s = 0; s < NSG; ++s) y += red[(s * 8 + r) * 32 + c];
  const float yv = float(T(y));
  float sv[HC];
  for (int j = 0; j < HC; ++j) sv[j] = float(x[((size_t)r * HC + j) * D + n]);
  const auto po = spost + r * HC;
  const auto cb = scomb + r * HC * HC;
  const auto pr = spre + r * HC;
  float h = 0.0f;
  for (int i = 0; i < HC; ++i) {
    float a = 0.0f;
    for (int j = 0; j < HC; ++j) a = metal::fma(cb[j * HC + i], sv[j], a);
    const T o = T(metal::fma(po[i], yv, a));
    x1[((size_t)r * HC + i) * D + n] = o;
    h = metal::fma(pr[i], float(o), h);
  }
  hv[(size_t)r * D + n] = T(h);
}
