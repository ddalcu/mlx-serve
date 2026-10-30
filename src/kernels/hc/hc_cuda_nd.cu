const int t = threadIdx.x;
const int lane = t & 31;
const int sg = t >> 5;
const int s = blockIdx.y % HC;
const int g = blockIdx.y / HC;
constexpr int PER = H / 256;
constexpr int K = HC * H;
constexpr bool DENSE = BITS == 16;
constexpr int VPW = 32 / BITS;
constexpr int WPL = 16 / VPW;
constexpr int KW = K / VPW;
constexpr int KG = K / GS;
constexpr int NSLOT = H / 16;
constexpr int SITER = (NSLOT + 31) / 32;
constexpr int PR = R + HC;
constexpr int NDG = R / 8;
constexpr unsigned MASK = (1u << BITS) - 1u;
__shared__ __align__(16) T xs[H];
__shared__ float npart[8];
__shared__ float tgi[8 * HC];
const int base = s * H;
const bool down = g < NDG;
// The down weights do not wait for the norm: every load of this warp's row goes out first, once
// for all NR rows.
unsigned pw[SITER][WPL];
float ps[SITER];
float pb[SITER];
const int n0 = g * 8 + sg;
if (down) {
  for (int i = 0; i < SITER; ++i) {
    const int slot = i * 32 + lane;
    if (slot < NSLOT) {
      const uint32_t* wp = dw_q + (size_t)n0 * KW + (size_t)(s * (H / VPW) + slot * WPL);
      for (int e = 0; e < WPL; ++e) pw[i][e] = wp[e];
      const size_t gi = (size_t)n0 * KG + (size_t)((base + slot * 16) / GS);
      ps[i] = DENSE ? 1.0f : static_cast<float>(dw_s[DENSE ? 0 : gi]);
      pb[i] = DENSE ? 0.0f : static_cast<float>(dw_b[DENSE ? 0 : gi]);
    }
  }
}
float wv[PER];
for (int i = 0; i < PER; ++i) wv[i] = static_cast<float>(nw[base + t + 256 * i]);
// The row count is read off the input, not templated: one compiled kernel serves every width.
int NR = 1;
for (int d = 0; d < x_in_ndim; ++d) NR *= x_in_shape[d];
NR /= K;
for (int row = 0; row < NR; ++row) {
const size_t xrow = (size_t)row * K;
float xv[PER];
T written[PER];
if (WR) {
  // Pending hcWrite: stream' = T(stream + T(out * inj)), the chain's two roundings.
  const float gate = static_cast<float>(wi_in[row * HC + s]);
  for (int i = 0; i < PER; ++i) {
    const int k = t + 256 * i;
    const T prod = T(static_cast<float>(wo_in[(size_t)row * H + k]) * gate);
    const T v = T(static_cast<float>(x_in[xrow + base + k]) + static_cast<float>(prod));
    written[i] = v;
    xv[i] = static_cast<float>(v);
  }
} else {
  for (int i = 0; i < PER; ++i) xv[i] = static_cast<float>(x_in[xrow + base + t + 256 * i]);
}
float a = 0.0f;
for (int i = 0; i < PER; ++i) a += xv[i] * xv[i];
a = hc_warp_sum(a);
if (lane == 0) npart[sg] = a;
__syncthreads();
float tot = 0.0f;
for (int q = 0; q < 8; ++q) tot += npart[q];
const float rsh = rsqrtf(tot / float(H) + eps[0]);
T xnv[PER];
for (int i = 0; i < PER; ++i) {
  xnv[i] = T(static_cast<float>(T(xv[i] * rsh)) * wv[i]);
  xs[t + 256 * i] = xnv[i];
}
__syncthreads();
float* pp = parts_out + (size_t)(row * HC + s) * PR;
if (down) {
  float acc = 0.0f;
  for (int i = 0; i < SITER; ++i) {
    const int slot = i * 32 + lane;
    if (slot < NSLOT) {
      float xr[16];
      float xsum = 0.0f;
      // Two 16-byte loads: one bf16 at a time, lanes 32 bytes apart conflict 8 ways.
      uint4 xq[2] = {reinterpret_cast<const uint4*>(xs + slot * 16)[0], reinterpret_cast<const uint4*>(xs + slot * 16)[1]};
      const T* xe = reinterpret_cast<const T*>(xq);
      for (int e = 0; e < 16; ++e) { xr[e] = static_cast<float>(xe[e]); xsum += xr[e]; }
      float d = 0.0f;
      for (int w = 0; w < WPL; ++w) {
        for (int j = 0; j < VPW; ++j) d += xr[w * VPW + j] * hc_val<DENSE>((pw[i][w] >> (j * BITS)) & MASK);
      }
      acc += ps[i] * d + pb[i] * xsum;
    }
  }
  const float v = hc_warp_sum(acc);
  if (lane == 0) pp[n0] = v;
} else if (INJ != 0) {
  // This stream's partial of the inject matvec; the up launch adds the streams.
  float ip[HC];
  for (int c = 0; c < HC; ++c) ip[c] = 0.0f;
  for (int i = 0; i < PER; ++i) {
    const size_t k = (size_t)(base + t + 256 * i);
    for (int c = 0; c < HC; ++c) ip[c] += static_cast<float>(xnv[i]) * static_cast<float>(iw[k * HC + c]);
  }
  for (int c = 0; c < HC; ++c) {
    const float pa = hc_warp_sum(ip[c]);
    if (lane == 0) tgi[sg * HC + c] = pa;
  }
  __syncthreads();
  if (t < HC) {
    float tt = 0.0f;
    for (int q = 0; q < 8; ++q) tt += tgi[q * HC + t];
    pp[R + t] = tt;
  }
}
if (g == 0) {
  for (int i = 0; i < PER; ++i) {
    xn_out[xrow + base + t + 256 * i] = xnv[i];
    if (WR) xs_out[xrow + base + t + 256 * i] = written[i];
  }
}
__syncthreads();
}