const int t = threadIdx.x;
const int lane = t & 31;
const int sg = t >> 5;
const int j = blockIdx.y * COLS + sg;
constexpr bool DENSE = BITS == 16;
constexpr int VPW = 32 / BITS;
constexpr int R_by_p = R / VPW;
constexpr int R_by_gs = R / GS;
constexpr int RIT = (R_by_p + 31) / 32;
constexpr int PR = R + HC;
constexpr unsigned MASK = (1u << BITS) - 1u;
__shared__ T acts[R];
// This column's up rows (one per stream) do not wait for the activation, loaded once for all NR rows.
unsigned pw[HC][RIT];
float ps[HC][RIT];
float pb[HC][RIT];
for (int h = 0; h < HC; ++h) {
  const size_t hrow = (size_t)h * H + (size_t)j;
  for (int i = 0; i < RIT; ++i) {
    const int pack = lane + 32 * i;
    if (pack < R_by_p) {
      pw[h][i] = uw_q[hrow * R_by_p + (size_t)pack];
      const size_t gi = hrow * R_by_gs + (size_t)((pack * VPW) / GS);
      ps[h][i] = DENSE ? 1.0f : static_cast<float>(uw_s[DENSE ? 0 : gi]);
      pb[h][i] = DENSE ? 0.0f : static_cast<float>(uw_b[DENSE ? 0 : gi]);
    }
  }
}
const int NR = xn_in_shape[0] / (HC * H);
for (int row = 0; row < NR; ++row) {
float xnv[HC];
for (int h = 0; h < HC; ++h) xnv[h] = static_cast<float>(xn_in[(size_t)row * HC * H + (size_t)h * H + (size_t)j]);
// act = silu(T(sum of the stream partials)), as the composed down projection ends.
for (int i = t; i < R; i += COLS * 32) {
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(row * HC + s) * PR + i];
  const T v = T(tsum);
  const T sig = T(hc_sig(static_cast<float>(v)));
  acts[i] = T(static_cast<float>(v) * static_cast<float>(sig));
}
if (INJ != 0 && blockIdx.y == 0 && t < HC) {
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(row * HC + s) * PR + R + t];
  const T v = T(tsum);
  const T sig = T(hc_sig(static_cast<float>(v)));
  inj_out[row * HC + t] = T(static_cast<float>(sig) * 2.0f);
}
__syncthreads();
float sum = 0.0f;
for (int h = 0; h < HC; ++h) {
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int i = 0; i < RIT; ++i) {
    const int pack = lane + 32 * i;
    if (pack >= R_by_p) continue;
    const int k = pack * VPW;
    if constexpr (DENSE) {
      a0 += static_cast<float>(acts[k]) * hc_val<true>(pw[h][i] & 0xFFFFu);
      a1 += static_cast<float>(acts[k + 1]) * hc_val<true>(pw[h][i] >> 16);
    } else {
      for (int ki = 0; ki < VPW; ki += 4) {
        const unsigned q = pw[h][i] >> (ki * BITS);
        a0 += static_cast<float>(acts[k + ki + 0]) * (float((q >> (0 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
        a1 += static_cast<float>(acts[k + ki + 1]) * (float((q >> (1 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
        a2 += static_cast<float>(acts[k + ki + 2]) * (float((q >> (2 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
        a3 += static_cast<float>(acts[k + ki + 3]) * (float((q >> (3 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
      }
    }
  }
  const float acc = hc_warp_sum((a0 + a1) + (a2 + a3));
  const T u = T(acc);
  const T sgv = T(hc_sig(static_cast<float>(u)));
  sum += static_cast<float>(T(static_cast<float>(sgv) * xnv[h]));
}
if (lane == 0) mixed_out[(size_t)row * H + j] = T(static_cast<float>(T(sum)) * static_cast<float>(T(1.0f / float(HC))));
__syncthreads();
}