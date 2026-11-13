// dec_moe_fp4: the decode MoE over mxfp4 routed experts (and the mxfp8 shared expert, whose gate/up reader is
// `dec_up_shared` of dec_moe_up_header.metal). A lane owns the units of 32 inputs u = 32 k + lane, so one load covers 32
// adjacent units: 16 bytes of codes and one e8m0 scale byte a unit.

// e8m0 `e` as 2^(e - 127), times the 2^14 `dec_fp4_word` leaves out.
inline float dec_fp4_scale(uint e) {
  return as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 16384.0f;
}

// The eight e2m1 codes of word `a` (code i in bits 4i..4i+3) as floats, each 2^-14 of its value: sign, then the other
// three bits placed as an fp16's top exponent and mantissa bits (a zero exponent field is the 0.5 subnormal); codes i
// and i + 4 land in one half2.
inline void dec_fp4_word(uint a, thread float* w) {
  const half2 h0 = as_type<half2>(((a << 9) & 0x0E000E00u) | ((a << 12) & 0x80008000u));
  const half2 h1 = as_type<half2>(((a << 5) & 0x0E000E00u) | ((a << 8) & 0x80008000u));
  const half2 h2 = as_type<half2>(((a << 1) & 0x0E000E00u) | ((a << 4) & 0x80008000u));
  const half2 h3 = as_type<half2>(((a >> 3) & 0x0E000E00u) | (a & 0x80008000u));
  w[0] = h0.x;
  w[1] = h1.x;
  w[2] = h2.x;
  w[3] = h3.x;
  w[4] = h0.y;
  w[5] = h1.y;
  w[6] = h2.y;
  w[7] = h3.y;
}

// The dots of NR rows' unit codes `q[i]` (32 codes each) against one unit of activations at `x` (bf16) into d[i].
template <int NR>
inline void dec_fp4_dots(thread const uint4* q, const device uint4* x, thread float* d) {
#pragma unroll
  for (int i = 0; i < NR; ++i) d[i] = 0.0f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 xq = x[c];
    const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      float wv[8];
      dec_fp4_word(q[i][c], wv);
#pragma unroll
      for (int e = 0; e < 8; ++e) d[i] = fma(xf[e], wv[e], d[i]);
    }
  }
}

// A lane's part of NR expert rows `row`, `row + rs`, ... of the fused-down bank (mxfp4, K inputs) against `x` into
// acc[i]: units first, first + step...
template <int K, int NR, typename T>
inline void dec_fp4_down_rows(const device uint32_t* w, const device uint8_t* sc, size_t row, size_t rs, const device T* x, uint first, uint step, thread float* acc) {
  constexpr uint NU = K / 32;
#pragma unroll
  for (int i = 0; i < NR; ++i) acc[i] = 0.0f;
  for (uint u = first; u < NU; u += step) {
    uint4 q[NR];
    float f[NR], d[NR];
#pragma unroll
    for (int i = 0; i < NR; ++i) {
      q[i] = ((const device uint4*)(w + (row + i * rs) * (K / 8)))[u];
      f[i] = dec_fp4_scale(uint(sc[(row + i * rs) * (K / 32) + u]));
    }
    dec_fp4_dots<NR>(q, (const device uint4*)x + u * 4, d);
#pragma unroll
    for (int i = 0; i < NR; ++i) acc[i] = fma(f[i], d[i], acc[i]);
  }
}

// The gate (acc[0]) and up (acc[1]) sums of the expert row `row` and `row + N` of the fused gate+up bank against `x`,
// a unit's gate and up codes read against one decode of its activations.
template <int K, int N, typename T>
inline void dec_fp4_up_pair(const device uint32_t* w, const device uint8_t* sc, size_t row, const device T* x, uint lane, thread float (&acc)[2]) {
  constexpr uint KW = K / 8, KG = K / 32, UPL = K / 1024;
  const device uint4* w0 = (const device uint4*)(w + row * KW) + lane;
  const device uint4* w1 = (const device uint4*)(w + (row + N) * KW) + lane;
  const device uint8_t* s0 = sc + row * KG + lane;
  const device uint8_t* s1 = sc + (row + N) * KG + lane;
  acc[0] = acc[1] = 0.0f;
#pragma unroll
  for (uint k = 0; k < UPL; ++k) {
    const uint4 q[2] = {w0[32 * k], w1[32 * k]};
    const float f0 = dec_fp4_scale(uint(s0[32 * k])), f1 = dec_fp4_scale(uint(s1[32 * k]));
    float d[2];
    dec_fp4_dots<2>(q, (const device uint4*)x + (32 * k + lane) * 4, d);
    acc[0] = fma(f0, d[0], acc[0]);
    acc[1] = fma(f1, d[1], acc[1]);
  }
  acc[0] = simd_sum(acc[0]);
  acc[1] = simd_sum(acc[1]);
}

// A lane's part of the shared expert's down row `row` (mxfp8, K inputs: a unit is 32 bytes) against `x`: units first,
// first + step...
template <int K, typename T>
inline float dec_fp4_down_shared(const device uint32_t* sw, const device uint8_t* ss, size_t row, const device T* x, uint first, uint step) {
  constexpr uint NU = K / 32;
  const device uint4* wp = (const device uint4*)(sw + row * (K / 4));
  const device uint8_t* sp = ss + row * (K / 32);
  float acc = 0.0f;
  for (uint u = first; u < NU; u += step) {
    const uint4 q0 = wp[2 * u], q1 = wp[2 * u + 1];
    const uint qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    float d = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const uint4 xq = ((const device uint4*)x)[u * 4 + c];
      const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
      const float4 a = dec_up_e4m3x4(qs[2 * c]), b = dec_up_e4m3x4(qs[2 * c + 1]);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        d = fma(xf[i], a[i], d);
        d = fma(xf[4 + i], b[i], d);
      }
    }
    const uint e = uint(sp[u]);
    acc = fma(as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f, d, acc);
  }
  return acc;
}
