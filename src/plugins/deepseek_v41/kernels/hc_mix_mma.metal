// hc_mix.metal's mixes on the simdgroup matrix units: z[l, m] = (x[l] . fn[m]) * rsqrt(mean(x[l]^2) + eps[0]) *
// scale[m] + base[m], every product and sum in f32. A threadgroup takes RT tiles of 8 rows and every mix (MT tiles of
// 8); its SG simdgroups split K, and the threadgroup sums their partial dots and squares. Each row is read once for all
// the mixes (hc_mix reads it once per group of M mixes), and fn once per 8 RT rows (hc_mix: once per R rows). An 8x8
// fragment's lane holds (mr, mc) and (mr, mc + 1), as glm5_next/kernels/mixes.metal's.
const uint tg = threadgroup_position_in_grid.x;
const uint sg = simdgroup_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint tid = sg * 32 + lane;
const int K = fn_shape[1];
const int L = x_shape[0];
constexpr int MIX = MT * 8;
const int KS = K / SG;
const short qid = short(lane / 4);
const short mr = (qid & 4) + short((lane / 2) % 4);
const short mc = (qid & 2) * 2 + short(lane % 2) * 2;
const int r0 = int(tg) * RT * 8;
simdgroup_matrix<float, 8, 8> a;
simdgroup_matrix<float, 8, 8> b[MT];
simdgroup_matrix<float, 8, 8> c[RT][MT];
float ss[RT];
for (int rt = 0; rt < RT; ++rt) {
  ss[rt] = 0.0f;
  for (int t = 0; t < MT; ++t) c[rt][t] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
}
for (int k0 = int(sg) * KS; k0 < int(sg + 1) * KS; k0 += 8) {
  // b[t]: fn's mixes t * 8 .. t * 8 + 7 at columns k0 .. k0 + 7, as (k, mix).
  const device float* wcol = fn + size_t(mc) * K + k0 + mr;
  for (int t = 0; t < MT; ++t) reinterpret_cast<thread float2&>(b[t].thread_elements()) = float2(wcol[size_t(t * 8) * K], wcol[size_t(t * 8 + 1) * K]);
  for (int rt = 0; rt < RT; ++rt) {
    const int r = r0 + rt * 8 + mr;
    const float2 av = r < L ? float2(float(x[size_t(r) * K + k0 + mc]), float(x[size_t(r) * K + k0 + mc + 1])) : float2(0.0f);
    ss[rt] = fma(av.x, av.x, fma(av.y, av.y, ss[rt]));
    reinterpret_cast<thread float2&>(a.thread_elements()) = av;
    for (int t = 0; t < MT; ++t) simdgroup_multiply_accumulate(c[rt][t], a, b[t], c[rt][t]);
  }
}
threadgroup float part[SG][RT * 8][MIX];
threadgroup float ssp[SG][RT][32];
threadgroup float rowss[RT * 8];
for (int rt = 0; rt < RT; ++rt) {
  for (int t = 0; t < MT; ++t) {
    const float2 cv = reinterpret_cast<thread float2&>(c[rt][t].thread_elements());
    part[sg][rt * 8 + mr][t * 8 + mc] = cv.x;
    part[sg][rt * 8 + mr][t * 8 + mc + 1] = cv.y;
  }
  ssp[sg][rt][lane] = ss[rt];
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
// A row's squares: the lanes whose fragment row is the row's, in every simdgroup.
if (tid < uint(RT * 8)) {
  const int rt = int(tid) / 8, q = int(tid) % 8;
  float acc = 0.0f;
  for (int g = 0; g < SG; ++g)
    for (int l = 0; l < 32; ++l) {
      const short lq = short(l / 4);
      if ((lq & 4) + short((l / 2) % 4) == q) acc += ssp[g][rt][l];
    }
  rowss[tid] = acc;
}
threadgroup_barrier(metal::mem_flags::mem_threadgroup);
for (int i = int(tid); i < RT * 8 * MIX; i += SG * 32) {
  const int row = i / MIX, m = i % MIX;
  const int l = r0 + row;
  if (l < L) {
    float acc = 0.0f;
    for (int g = 0; g < SG; ++g) acc += part[g][row][m];
    z[size_t(l) * MIX + m] = acc * metal::precise::rsqrt(rowss[row] / float(K) + eps[0]) * scale[m] + base[m];
  }
}
