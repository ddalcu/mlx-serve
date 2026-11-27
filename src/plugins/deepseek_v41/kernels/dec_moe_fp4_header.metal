// dec_moe_fp4: the decode MoE over fp4 routed experts, MXFP4 (GS 32) or NVFP4 (GS 16), and the mxfp8 shared expert
// (whose gate/up reader is `dec_up_shared` of dec_moe_up_header.metal). A lane owns the units of 32 inputs
// u = 32 k + lane, so one load covers 32 adjacent units: 16 bytes of codes and 32 / GS group-scale bytes a unit, each
// decoded exactly by `mlxserve_fp4_scale_of<GS>` (kernels/common/fp4_scale.metal).

// The two e2m1 codes of byte `b` (low nibble first) as floats, each 2^-14 of its value: sign, then the other three bits
// placed as an fp16's top exponent and mantissa bits (a zero exponent field is the 0.5 subnormal). -0 stays -0.
inline float2 dec_fp4_pair(uint b) {
  const uint v = ((b & 0x07u) << 9) | ((b & 0x08u) << 12) | ((b & 0x70u) << 21) | ((b & 0x80u) << 24);
  return float2(as_type<half2>(v));
}

// The eight e2m1 codes of word `a` (code i in bits 4i..4i+3) as floats in code order, each the bits `dec_fp4_pair` gives
// it: four masked moves place two codes at once, codes i and i + 4 in one half2 (DW, `MLX_SERVE_DSV41_FP4_DEC_WORD`).
inline void dec_fp4_word(uint a, thread float (&wv)[8]) {
  const float2 c04 = float2(as_type<half2>(((a & 0x00070007u) << 9) | ((a & 0x00080008u) << 12)));
  const float2 c15 = float2(as_type<half2>(((a & 0x00700070u) << 5) | ((a & 0x00800080u) << 8)));
  const float2 c26 = float2(as_type<half2>(((a & 0x07000700u) << 1) | ((a & 0x08000800u) << 4)));
  const float2 c37 = float2(as_type<half2>(((a & 0x70007000u) >> 3) | (a & 0x80008000u)));
  wv[0] = c04.x;
  wv[1] = c15.x;
  wv[2] = c26.x;
  wv[3] = c37.x;
  wv[4] = c04.y;
  wv[5] = c15.y;
  wv[6] = c26.y;
  wv[7] = c37.y;
}

// `d` plus the dot of the codes in words [C0, C1) of `q` (eight codes a word) against their activations at `x` (bf16,
// a uint4 a word), one fma chain in code order. DW decodes a word at a time (`dec_fp4_word`): the same values, the same
// chain.
template <int C0, int C1, int DW = 0>
inline float dec_fp4_dot(uint4 q, const device uint4* x, float d) {
#pragma unroll
  for (int c = C0; c < C1; ++c) {
    const uint4 xq = x[c];
    const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
    const uint a = c == 0 ? q.x : c == 1 ? q.y : c == 2 ? q.z : q.w;
    if constexpr (DW != 0) {
      float wv[8];
      dec_fp4_word(a, wv);
#pragma unroll
      for (int i = 0; i < 8; ++i) d = fma(xf[i], wv[i], d);
    } else {
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 wv = dec_fp4_pair((a >> (8 * p)) & 0xFFu);
        d = fma(xf[2 * p], wv.x, d);
        d = fma(xf[2 * p + 1], wv.y, d);
      }
    }
  }
  return d;
}

// `acc` plus one unit (the 32 codes in `q` against the 32 activations at `x`): each group's dot, its 2^14 restored
// (exact), times its scale as MLX reads it (`s0`, `s1`: the unit's 32 / GS scale bytes; GS 32 reads only `s0`). FR 0 (MLX_SERVE_DSV41_FP4_READS
// off): two equal NVFP4 scale bytes take MXFP4's order, one chain over the unit and one add, so a bank whose
// 16-halves share a scale (NVIDIA's DeepSeek pack) gives its MXFP4 conversion's bytes; FR 1: each 16-group apart.
template <int GS, int FR, int DW = 0>
inline float dec_fp4_unit_s(uint4 q, const device uint4* x, uint s0, uint s1, float acc) {
  if constexpr (GS == 32) {
    return fma(mlxserve_fp4_scale_of<32>(s0), dec_fp4_dot<0, 4, DW>(q, x, 0.0f) * 16384.0f, acc);
  } else if constexpr (FR == 0) {
    const bool one = s0 == s1;
    const float d0 = dec_fp4_dot<0, 2, DW>(q, x, 0.0f);
    const float d1 = dec_fp4_dot<2, 4, DW>(q, x, one ? d0 : 0.0f);
    acc = one ? acc : fma(mlxserve_fp4_scale_of<16>(s0), d0 * 16384.0f, acc);
    return fma(mlxserve_fp4_scale_of<16>(s1), d1 * 16384.0f, acc);
  } else {
    acc = fma(mlxserve_fp4_scale_of<16>(s0), dec_fp4_dot<0, 2, DW>(q, x, 0.0f) * 16384.0f, acc);
    return fma(mlxserve_fp4_scale_of<16>(s1), dec_fp4_dot<2, 4, DW>(q, x, 0.0f) * 16384.0f, acc);
  }
}

// `dec_fp4_unit_s` with the unit's scale bytes read at `s`.
template <int GS, int FR, int DW = 0>
inline float dec_fp4_unit(uint4 q, const device uint4* x, const device uint8_t* s, float acc) {
  return dec_fp4_unit_s<GS, FR, DW>(q, x, uint(s[0]), GS == 32 ? 0u : uint(s[1]), acc);
}

// The gate (acc[0]) and up (acc[1]) sums of the expert row `row` and `row + N` of the fused gate+up bank against `x`.
template <int K, int N, int GS, int FR, typename T, int DW = 0>
inline void dec_fp4_up_routed(const device uint32_t* w, const device uint8_t* sc, size_t row, const device T* x, uint lane, thread float (&acc)[2]) {
  constexpr uint KW = K / 8, KG = K / GS, SPU = 32 / GS, UPL = K / 1024;
  const device uint4* w0 = (const device uint4*)(w + row * KW) + lane;
  const device uint4* w1 = (const device uint4*)(w + (row + N) * KW) + lane;
  const device uint8_t* s0 = sc + row * KG + lane * SPU;
  const device uint8_t* s1 = sc + (row + N) * KG + lane * SPU;
  acc[0] = acc[1] = 0.0f;
#pragma unroll
  for (uint k = 0; k < UPL; ++k) {
    const uint4 q0 = w0[32 * k], q1 = w1[32 * k];
    const device uint4* xu = (const device uint4*)x + (32 * k + lane) * 4;
    acc[0] = dec_fp4_unit<GS, FR, DW>(q0, xu, s0 + 32 * SPU * k, acc[0]);
    acc[1] = dec_fp4_unit<GS, FR, DW>(q1, xu, s1 + 32 * SPU * k, acc[1]);
  }
  acc[0] = simd_sum(acc[0]);
  acc[1] = simd_sum(acc[1]);
}

// One row `row` of the shared expert's fused gate+up bank (mxfp8: the gate's n or the up's n + N) against each of R x
// rows, summed over the simdgroup into `acc[r]`: `dec_up_shared`'s chain for that row, step for step (its gate and up
// chains never meet), so US (`MLX_SERVE_DSV41_FP4_UP_SPLIT`) gives the gate and the up a simdgroup each.
template <int R, int K, typename T>
inline void dec_fp4_up_shared_row(const device uint32_t* sw, const device uint8_t* ss, uint row, const device T* x, uint lane, thread float (&acc)[R]) {
  constexpr uint KWS = K / 4, KGS = K / 32, UPL = K / 1024;
  const uint wo = row * KWS + lane * 8;
  const uint so = row * KGS + lane;
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
#pragma unroll
  for (int k = 0; k < UPL; ++k) {
    uint q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) q[j] = sw[wo + k * 256 + j];
    const float2 f = dsv41_e8m0<0>(uint(ss[so + 32 * k]));
    float dq[R];
#pragma unroll
    for (int r = 0; r < R; ++r) dq[r] = 0.0f;
#pragma unroll
    for (int c = 0; c < 4; ++c) {  // eight inputs: words 2c and 2c + 1
      const float4 w0 = dsv41_e4m3x4<0>(q[2 * c]), w1 = dsv41_e4m3x4<0>(q[2 * c + 1]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4 xq = ((const device uint4*)(x + r * K))[(32 * k + lane) * 4 + c];
        const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
        float d = dq[r];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          d = fma(xf[i], w0[i], d);
          d = fma(xf[4 + i], w1[i], d);
        }
        dq[r] = d;
      }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = fma(f.x, dq[r] * f.y, acc[r]);
  }
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = simd_sum(acc[r]);
}

// A lane's part of the expert row `row` of the fused-down bank (fp4, K inputs) against `x`: units first, first + STEP...
template <int K, int GS, int FR, int STEP, typename T, int DW = 0>
inline float dec_fp4_down_routed(const device uint32_t* w, const device uint8_t* sc, size_t row, const device T* x, uint first) {
  constexpr uint NU = K / 32, SPU = 32 / GS;
  const device uint4* wp = (const device uint4*)(w + row * (K / 8));
  const device uint8_t* sp = sc + row * (K / GS);
  float acc = 0.0f;
  for (uint u = first; u < NU; u += STEP) acc = dec_fp4_unit<GS, FR, DW>(wp[u], (const device uint4*)x + u * 4, sp + u * SPU, acc);
  return acc;
}

// One unit of the shared expert's down row: its 32 e4m3 bytes `q0`, `q1` against the 32 activations at `x` (bf16), times
// its e8m0 scale `e`, added to `acc`.
template <int FR>
inline float dec_fp4_down_shared_unit(uint4 q0, uint4 q1, const device uint4* x, uint e, float acc) {
  const uint qs[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
  float d = 0.0f;
#pragma unroll
  for (int c = 0; c < 4; ++c) {
    const uint4 xq = x[c];
    const float xf[8] = {dec_up_bf(xq.x, 0), dec_up_bf(xq.x, 1), dec_up_bf(xq.y, 0), dec_up_bf(xq.y, 1), dec_up_bf(xq.z, 0), dec_up_bf(xq.z, 1), dec_up_bf(xq.w, 0), dec_up_bf(xq.w, 1)};
    const float4 a = dsv41_e4m3x4<FR>(qs[2 * c]), b = dsv41_e4m3x4<FR>(qs[2 * c + 1]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      d = fma(xf[i], a[i], d);
      d = fma(xf[4 + i], b[i], d);
    }
  }
  if constexpr (FR == 0) return fma(mlxserve_fp4_e8m0(e), d * 256.0f, acc);
  else return fma(as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f, d, acc);
}

// A lane's part of the shared expert's down row `row` (mxfp8, K inputs: a unit is 32 bytes) against `x`: units first,
// first + STEP... FR 0: e4m3 and e8m0 as MLX reads them (S.1111.111 and e8m0 0xFF NaN, codes 247..254 finite); FR 1:
// e4m3 by bit placement and e8m0 as a float exponent times 2^8 (the same bytes on every finite code under 247).
template <int K, int FR, int STEP, typename T>
inline float dec_fp4_down_shared(const device uint32_t* sw, const device uint8_t* ss, size_t row, const device T* x, uint first) {
  constexpr uint NU = K / 32;
  const device uint4* wp = (const device uint4*)(sw + row * (K / 4));
  const device uint8_t* sp = ss + row * (K / 32);
  const device uint4* xp = (const device uint4*)x;
  float acc = 0.0f;
  for (uint u = first; u < NU; u += STEP) acc = dec_fp4_down_shared_unit<FR>(wp[2 * u], wp[2 * u + 1], xp + u * 4, uint(sp[u]), acc);
  return acc;
}

// ModelOpt's per-expert scales (T [E]) on expert e's gate and up sums, each rounded to T where `gather_qmm` writes the
// sum and again after the product, as the composed path multiplies them; a flag off leaves its sum as it is. `P` is
// the scales' pointer: MLX binds a short input in the constant address space.
template <int GSCALE, int USCALE, typename T, typename P>
inline void dec_fp4_expert_scales(thread float (&acc)[2], P gsc, P usc, uint e) {
  if constexpr (GSCALE != 0) acc[0] = float(T(float(T(acc[0])) * float(gsc[e])));
  if constexpr (USCALE != 0) acc[1] = float(T(float(T(acc[1])) * float(usc[e])));
}
