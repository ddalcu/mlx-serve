// dec_moe_up: a simdgroup computes one output row (its gate and its up dot) of one expert, or of the shared expert.
// A lane owns the UPL = K / 1024 units of 32 inputs u = 32 k + lane, so one load covers 32 adjacent units: an affine
// unit is BITS words (a quant group is two units), an mxfp8 unit 32 bytes (one e8m0 scale).

inline float dec_up_bf(uint w, int i) {
  return as_type<float>(i & 1 ? (w & 0xFFFF0000u) : (w << 16));
}

// Element i (0..31) of a unit of BITS-bit fields spread over BITS words, as mx.quantize packs them (dense, straddling).
template <int BITS>
inline uint dec_up_field(thread const uint* q, int i) {
  constexpr uint MASK = (1u << BITS) - 1u;
  const int bit = BITS * i, wi = bit >> 5, s = bit & 31;
  if (s + BITS > 32) return ((q[wi] >> s) | (q[wi + 1] << (32 - s))) & MASK;
  return (q[wi] >> s) & MASK;
}

// Four e4m3 bytes of `w` as floats, bytes 0..3 in order, each 2^-8 of its value (the half exponent bias is 8 above e4m3's).
inline float4 dec_up_e4m3x4(uint w) {
  const uint lo = ((w << 7) & 0x3F803F80u) | ((w << 8) & 0x80008000u);
  const uint hi = ((w >> 1) & 0x3F803F80u) | (w & 0x80008000u);
  const half2 a = as_type<half2>(lo), c = as_type<half2>(hi);
  return float4(a.x, c.x, a.y, c.y);
}

// Entry i of the decode table: the EPL fields packed in i (one byte of 2-bit fields, six bits of 3-bit ones), as floats.
template <int BITS>
inline void dec_up_table(threadgroup vec<float, BITS == 2 ? 4 : 2>* lut, uint tid, uint threads) {
  constexpr int EPL = BITS == 2 ? 4 : 2;
  constexpr uint MASK = (1u << BITS) - 1u;
  for (uint i = tid; i < (1u << (EPL * BITS)); i += threads)
#pragma unroll
    for (int e = 0; e < EPL; ++e) lut[i][e] = float((i >> (BITS * e)) & MASK);
}

// The gate (acc[rr][0]) and up (acc[rr][1]) sums of one expert's output row against RG x rows, over the simdgroup; the
// weights are decoded once whatever RG is. `xrow[rr]` is the row of `x` rr reads; `lut` decodes EPL fields at a time.
template <int BITS, int K, int N, int RG, typename T>
inline void dec_up_routed(const device uint32_t* w, const device T* sc, const device T* bi, uint row, const device T* x, thread const uint* xrow, uint lane, thread float (*acc)[2], threadgroup const vec<float, BITS == 2 ? 4 : 2>* lut) {
  constexpr uint KW = K * BITS / 32, KG = K / 64, UPL = K / 1024;
  constexpr int EPL = BITS == 2 ? 4 : 2, NIDX = 8 / EPL;
  const uint wo[2] = {row * KW + lane * BITS, (row + N) * KW + lane * BITS};
  const uint so[2] = {row * KG + lane / 2, (row + N) * KG + lane / 2};
#pragma unroll
  for (int rr = 0; rr < RG; ++rr) acc[rr][0] = acc[rr][1] = 0.0f;
#pragma unroll
  for (int k = 0; k < UPL; ++k) {
    uint q[2][BITS];
    float s[2], b[2];
#pragma unroll
    for (int wr = 0; wr < 2; ++wr) {
      if constexpr (BITS == 3) {
        const uint3 v = uint3(*(const device packed_uint3*)(w + wo[wr] + k * 32 * BITS));
        q[wr][0] = v.x;
        q[wr][1] = v.y;
        q[wr][2] = v.z;
      } else {
#pragma unroll
        for (int j = 0; j < BITS; ++j) q[wr][j] = w[wo[wr] + k * 32 * BITS + j];
      }
      s[wr] = float(sc[so[wr] + 16 * k]);
      b[wr] = float(bi[so[wr] + 16 * k]);
    }
    float xs[RG], dq[RG][2];
#pragma unroll
    for (int rr = 0; rr < RG; ++rr) xs[rr] = dq[rr][0] = dq[rr][1] = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {  // eight inputs
      vec<float, EPL> wf[2][NIDX];
#pragma unroll
      for (int wr = 0; wr < 2; ++wr)
#pragma unroll
        for (int h = 0; h < NIDX; ++h) wf[wr][h] = lut[dec_up_field<EPL * BITS>(q[wr], NIDX * c + h)];
#pragma unroll
      for (int rr = 0; rr < RG; ++rr) {
        const uint4 xq = ((const device uint4*)(x + xrow[rr] * K))[(32 * k + lane) * 4 + c];
        const uint xw[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const float xf = dec_up_bf(xw[e >> 1], e);
          xs[rr] += xf;
          dq[rr][0] = fma(xf, wf[0][e / EPL][e % EPL], dq[rr][0]);
          dq[rr][1] = fma(xf, wf[1][e / EPL][e % EPL], dq[rr][1]);
        }
      }
    }
#pragma unroll
    for (int rr = 0; rr < RG; ++rr) {
      acc[rr][0] = fma(s[0], dq[rr][0], fma(b[0], xs[rr], acc[rr][0]));
      acc[rr][1] = fma(s[1], dq[rr][1], fma(b[1], xs[rr], acc[rr][1]));
    }
  }
#pragma unroll
  for (int rr = 0; rr < RG; ++rr) {
    acc[rr][0] = simd_sum(acc[rr][0]);
    acc[rr][1] = simd_sum(acc[rr][1]);
  }
}

// The shared expert's gate (acc[r][0]) and up (acc[r][1]) of output row n against each of R x rows; its weights are
// decoded once whatever R is.
template <int R, int K, int N, typename T>
inline void dec_up_shared(const device uint32_t* sw, const device uint8_t* ss, uint n, const device T* x, uint lane, thread float (*acc)[2]) {
  constexpr uint KWS = K / 4, KGS = K / 32, UPL = K / 1024;
  const uint wo[2] = {n * KWS + lane * 8, (n + N) * KWS + lane * 8};
  const uint so[2] = {n * KGS + lane, (n + N) * KGS + lane};
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r][0] = acc[r][1] = 0.0f;
#pragma unroll
  for (int k = 0; k < UPL; ++k) {
    uint q[2][8];
    float f[2];
#pragma unroll
    for (int wr = 0; wr < 2; ++wr) {
#pragma unroll
      for (int j = 0; j < 8; ++j) q[wr][j] = sw[wo[wr] + k * 256 + j];
      const uint e = uint(ss[so[wr] + 32 * k]);
      f[wr] = as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f;
    }
    float dq[R][2];
#pragma unroll
    for (int r = 0; r < R; ++r) dq[r][0] = dq[r][1] = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {  // eight inputs: words 2c and 2c + 1
      float4 wv[2][2];
#pragma unroll
      for (int wr = 0; wr < 2; ++wr) {
        wv[wr][0] = dec_up_e4m3x4(q[wr][2 * c]);
        wv[wr][1] = dec_up_e4m3x4(q[wr][2 * c + 1]);
      }
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4 xq = ((const device uint4*)(x + r * K))[(32 * k + lane) * 4 + c];
        const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
#pragma unroll
        for (int wr = 0; wr < 2; ++wr) {
          float d = dq[r][wr];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            d = fma(xf[i], wv[wr][0][i], d);
            d = fma(xf[4 + i], wv[wr][1][i], d);
          }
          dq[r][wr] = d;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
      acc[r][0] = fma(f[0], dq[r][0], acc[r][0]);
      acc[r][1] = fma(f[1], dq[r][1], acc[r][1]);
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r) {
    acc[r][0] = simd_sum(acc[r][0]);
    acc[r][1] = simd_sum(acc[r][1]);
  }
}

// clampedSwiglu(gate, up) * wt with the two sums rounded to T where gather_qmm writes them; a NaN stays NaN
// (MLX's minimum and clip propagate it, Metal's drop it).
template <typename T>
inline T dec_up_act(float gate, float up, float lim, float wt) {
  float g = float(T(gate)), u = float(T(up));
  if (lim > 0.0f) {
    g = metal::isnan(g) ? g : metal::min(g, lim);
    u = metal::isnan(u) ? u : metal::clamp(u, -lim, lim);
  }
  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
  return T((g * (g < 0.0f ? y : 1.0f - y)) * u * wt);
}
