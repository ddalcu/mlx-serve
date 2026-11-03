const uint t = thread_index_in_threadgroup;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int row = int(threadgroup_position_in_grid.z);
const int j = int(threadgroup_position_in_grid.y) * COLS + int(sg);
constexpr bool WORD = (32 % BITS) == 0;
constexpr bool DENSE = BITS == 16;
constexpr int VPW = WORD ? 32 / BITS : 8;
constexpr int R_by_p = R / VPW;
constexpr int R_by_gs = R / GS;
constexpr int RIT = (R_by_p + 31) / 32;
constexpr int PR = R + HC;
constexpr uint MASK = (1u << BITS) - 1u;
threadgroup T acts[R];

// This column's up rows (one per stream) do not wait for the activation.
uint pw[HC][WORD ? RIT : 1];
ulong pr[HC][WORD ? 1 : RIT];
float ps[HC][RIT];
float pb[HC][RIT];
float xnv[HC];
for (int h = 0; h < HC; ++h) {
  const size_t hrow = (size_t)h * H + (size_t)j;
  xnv[h] = float(xn_in[(size_t)row * HC * H + hrow]);
  for (int i = 0; i < RIT; ++i) {
    const int pack = int(lane) + 32 * i;
    if (pack < R_by_p) {
      if constexpr (WORD) pw[h][i] = uw_q[hrow * R_by_p + (size_t)pack];
      else pr[h][i] = hc_pack8<BITS>((const device uchar*)uw_q + hrow * (size_t)(R * BITS / 8) + (size_t)pack * BITS);
      const size_t gi = hrow * R_by_gs + (size_t)((pack * VPW) / GS);
      ps[h][i] = DENSE ? 1.0f : float(uw_s[DENSE ? 0 : gi]);
      pb[h][i] = DENSE ? 0.0f : float(uw_b[DENSE ? 0 : gi]);
    }
  }
}
// act = silu(T(sum of the stream partials)), as the three-kernel down launch ends.
for (int i = int(t); i < R; i += COLS * 32) {
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(row * HC + s) * PR + i];
  const T v = T(tsum);
  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  acts[i] = v * sig;
}
if (INJ != 0 && threadgroup_position_in_grid.y == 0 && t < uint(HC)) {
  float tsum = 0.0f;
  for (int s = 0; s < HC; ++s) tsum += parts_in[(row * HC + s) * PR + R + int(t)];
  const T v = T(tsum);
  const T sig = T(1.0f / (1.0f + metal::exp(-float(v))));
  inj_out[row * HC + t] = sig * T(2.0f);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float sum = 0.0f;
for (int h = 0; h < HC; ++h) {
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int i = 0; i < RIT; ++i) {
    const int pack = int(lane) + 32 * i;
    if (pack < R_by_p && DENSE) {
      // Two bf16 values a word.
      const int k_base = pack * VPW;
      a0 += float(acts[k_base]) * hc_val<T, true>(pw[h][i] & 0xFFFFu);
      a1 += float(acts[k_base + 1]) * hc_val<T, true>(pw[h][i] >> 16);
    } else if (pack < R_by_p) {
      const int k_base = pack * VPW;
      for (int ki = 0; ki < VPW; ki += 4) {
        const int k = k_base + ki;
        // Values ki..ki+3 of this pack; a 3/5/6-bit pack is 8 values in a ulong.
        const uint q = WORD ? (pw[h][i] >> (ki * BITS)) : uint(pr[h][i] >> (ki * BITS));
        a0 += float(acts[k + 0]) * (float((q >> (0 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
        a1 += float(acts[k + 1]) * (float((q >> (1 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
        a2 += float(acts[k + 2]) * (float((q >> (2 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
        a3 += float(acts[k + 3]) * (float((q >> (3 * BITS)) & MASK) * ps[h][i] + pb[h][i]);
      }
    }
  }
  const float acc = simd_sum((a0 + a1) + (a2 + a3));
  const T u = T(acc);
  const T sgv = T(1.0f / (1.0f + metal::exp(-float(u))));
  sum += float(T(float(sgv) * xnv[h]));
}
if (lane == 0) mixed_out[(size_t)row * H + j] = T(float(T(sum)) * float(T(1.0f / float(HC))));