const uint t = thread_index_in_threadgroup;
const int row = int(threadgroup_position_in_grid.z);
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const int s = int(threadgroup_position_in_grid.y) % HC;
const int g = int(threadgroup_position_in_grid.y) / HC;
constexpr int PER = H / 256;
constexpr int K = HC * H;
// A slot's 16 values are WPL words at 2/4/8 bits, else two 8-value packs of BITS bytes.
constexpr bool WORD = (32 % BITS) == 0;
constexpr bool DENSE = BITS == 16;
constexpr int VPW = WORD ? 32 / BITS : 8;
constexpr int WPL = 16 / VPW;
constexpr int KW = K / VPW;
constexpr int KG = K / GS;
constexpr int NSLOT = H / 16;
constexpr int SITER = (NSLOT + 31) / 32;
constexpr int PR = R + HC;
constexpr int NDG = R / ROWS_TG;
constexpr uint MASK = (1u << BITS) - 1u;
threadgroup T xs[H];
threadgroup float npart[8];
threadgroup float tgi[8 * HC];
const int base = s * H;
const size_t xrow = (size_t)row * K;
const bool down = g < NDG;

// The down weights do not wait for the norm: every load of this simdgroup's rows goes out first.
uint pw[RPS][SITER][WORD ? WPL : 1];
ulong pr[RPS][SITER][WORD ? 1 : WPL];
float ps[RPS][SITER];
float pb[RPS][SITER];
const int n0 = g * ROWS_TG + int(sg) * RPS;
if (down) {
  for (int r = 0; r < RPS; ++r) {
    for (int i = 0; i < SITER; ++i) {
      const int slot = i * 32 + int(lane);
      if (slot < NSLOT) {
        if constexpr (WORD) {
          const device uint32_t* wp = dw_q + (size_t)(n0 + r) * KW + (size_t)(s * (H / VPW) + slot * WPL);
          for (int e = 0; e < WPL; ++e) pw[r][i][e] = wp[e];
        } else {
          const device uchar* wb = (const device uchar*)dw_q + (size_t)(n0 + r) * (size_t)(K * BITS / 8) + (size_t)(base + slot * 16) * BITS / 8;
          for (int e = 0; e < WPL; ++e) pr[r][i][e] = hc_pack8<BITS>(wb + e * BITS);
        }
        const size_t gi = (size_t)(n0 + r) * KG + (size_t)((base + slot * 16) / GS);
        ps[r][i] = DENSE ? 1.0f : float(dw_s[DENSE ? 0 : gi]);
        pb[r][i] = DENSE ? 0.0f : float(dw_b[DENSE ? 0 : gi]);
      }
    }
  }
}

float wv[PER];
for (int i = 0; i < PER; ++i) wv[i] = float(nw[base + int(t) + 256 * i]);
float xv[PER];
T written[PER];
if (WR) {
  // Pending hcWrite: stream' = T(stream + T(out * inj)), the chain's two roundings.
  const float gate = float(wi_in[row * HC + s]);
  for (int i = 0; i < PER; ++i) {
    const int k = int(t) + 256 * i;
    const T v = T(float(x_in[xrow + base + k]) + float(T(float(wo_in[(size_t)row * H + k]) * gate)));
    written[i] = v;
    xv[i] = float(v);
  }
} else {
  for (int i = 0; i < PER; ++i) xv[i] = float(x_in[xrow + base + int(t) + 256 * i]);
}
float a = 0.0f;
for (int i = 0; i < PER; ++i) a += xv[i] * xv[i];
a = simd_sum(a);
if (lane == 0) npart[sg] = a;
threadgroup_barrier(mem_flags::mem_threadgroup);
float tot = 0.0f;
for (int q = 0; q < 8; ++q) tot += npart[q];
const float rsh = rsqrt(tot / float(H) + eps[0]);
T xnv[PER];
for (int i = 0; i < PER; ++i) {
  xnv[i] = T(float(T(xv[i] * rsh)) * wv[i]);
  xs[int(t) + 256 * i] = xnv[i];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

device float* pp = parts_out + (size_t)(row * HC + s) * PR;
if (down) {
  float acc[RPS];
  for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
  for (int i = 0; i < SITER; ++i) {
    const int slot = i * 32 + int(lane);
    if (slot < NSLOT) {
      float xr[16];
      float xsum = 0.0f;
      for (int e = 0; e < 16; ++e) { xr[e] = float(xs[slot * 16 + e]); xsum += xr[e]; }
      for (int r = 0; r < RPS; ++r) {
        float d = 0.0f;
        for (int w = 0; w < WPL; ++w) {
          if constexpr (WORD) {
            for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * hc_val<T, DENSE>((pw[r][i][w] >> (j * BITS)) & MASK);
          } else {
            for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * float(uint(pr[r][i][w] >> (j * BITS)) & MASK);
          }
        }
        acc[r] += ps[r][i] * d + pb[r][i] * xsum;
      }
    }
  }
  for (int r = 0; r < RPS; ++r) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) pp[n0 + r] = v;
  }
} else if (INJ != 0) {
  // This stream's partial of the inject matvec; the up launch adds the streams.
  float ip[HC];
  for (int c = 0; c < HC; ++c) ip[c] = 0.0f;
  for (int i = 0; i < PER; ++i) {
    const size_t k = (size_t)(base + int(t) + 256 * i);
    for (int c = 0; c < HC; ++c) ip[c] += float(xnv[i]) * float(iw[k * HC + c]);
  }
  for (int c = 0; c < HC; ++c) {
    const float pa = simd_sum(ip[c]);
    if (lane == 0) tgi[sg * HC + c] = pa;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < uint(HC)) {
    float tt = 0.0f;
    for (int q = 0; q < 8; ++q) tt += tgi[q * HC + t];
    pp[R + t] = tt;
  }
}
// Device stores come last: a later device load waits on every earlier store it cannot prove disjoint.
if (g == 0) {
  for (int i = 0; i < PER; ++i) {
    xn_out[xrow + base + int(t) + 256 * i] = xnv[i];
    if (WR) xs_out[xrow + base + int(t) + 256 * i] = written[i];
  }
}