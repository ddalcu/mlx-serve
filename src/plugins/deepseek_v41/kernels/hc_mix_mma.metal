// hc_mix.metal's mixes on the simdgroup matrix units: z[l, m] = (x[l] . fn[m]) * rsqrt(mean(x[l]^2) + eps[0]) *
// scale[m] + base[m], every product and sum in f32. A threadgroup takes RT tiles of 8 rows and every mix (MT tiles of
// 8), simdgroup s the row tile s. Each 64-column step stages the rows (as f32) and fn's columns (transposed) in
// threadgroup memory with whole-row loads, so each row is read once for all the mixes and fn once per 8 RT rows.
const uint tg = threadgroup_position_in_grid.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg * 32 + lane;
constexpr int KC = 64;
constexpr int NT = RT * 32;
constexpr int ROWS = RT * 8;
constexpr int MIX = MT * 8;
const int K = fn_shape[1];
const int L = x_shape[0];
const int r0 = int(tg) * ROWS;
threadgroup float xs[ROWS * KC];
threadgroup float ft[KC * MIX];
simdgroup_matrix<float, 8, 8> a;
simdgroup_matrix<float, 8, 8> b;
simdgroup_matrix<float, 8, 8> c[MT];
for (int t = 0; t < MT; ++t) c[t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
float ss = 0.0f;
for (int k0 = 0; k0 < K; k0 += KC) {
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  // The rows' columns k0 .. k0 + 63: each thread a run of ROWS * KC / NT = 16 consecutive values of one row.
  {
    const int i = int(tid) * 16;
    const int r = i / KC, col = i % KC;
    const bool rv = r0 + r < L;
    const device auto* src = x + size_t(rv ? r0 + r : 0) * K + k0 + col;
    for (int q = 0; q < 16; ++q) xs[r * KC + col + q] = rv ? float(src[q]) : 0.0f;
  }
  // fn's mixes at those columns, transposed to (k, mix): consecutive threads read consecutive columns.
  for (int i = int(tid); i < KC * MIX; i += NT) {
    const int m = i / KC, col = i % KC;
    ft[col * MIX + m] = fn[size_t(m) * K + k0 + col];
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int kk = 0; kk < KC; kk += 8) {
    simdgroup_load(a, xs + sg * 8 * KC + kk, KC);
    const float2 av = reinterpret_cast<thread float2&>(a.thread_elements());
    ss = fma(av.x, av.x, fma(av.y, av.y, ss));
    for (int t = 0; t < MT; ++t) {
      simdgroup_load(b, ft + kk * MIX + t * 8, MIX);
      simdgroup_multiply_accumulate(c[t], a, b, c[t]);
    }
  }
}
// A fragment's lane holds row mr's columns mc and mc + 1 (glm5_next/kernels/mixes.metal's layout): a row's squares
// are the sum over the 4 lanes that share its mr.
const short qid = short(lane / 4);
const short mr = (qid & 4) + short((lane / 2) % 4);
const short mc = (qid & 2) * 2 + short(lane % 2) * 2;
threadgroup float zs[ROWS * MIX];
threadgroup float rs[ROWS * 4];
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (int t = 0; t < MT; ++t) {
  const float2 cv = reinterpret_cast<thread float2&>(c[t].thread_elements());
  zs[(sg * 8 + mr) * MIX + t * 8 + mc] = cv.x;
  zs[(sg * 8 + mr) * MIX + t * 8 + mc + 1] = cv.y;
}
rs[(sg * 8 + mr) * 4 + mc / 2] = ss;
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (int i = int(tid); i < ROWS * MIX; i += NT) {
  const int row = i / MIX, m = i % MIX;
  const int l = r0 + row;
  if (l < L) {
    const float q = rs[row * 4] + rs[row * 4 + 1] + rs[row * 4 + 2] + rs[row * 4 + 3];
    z[size_t(l) * MIX + m] = zs[i] * metal::precise::rsqrt(q / float(K) + eps[0]) * scale[m] + base[m];
  }
}
