// GLM-5.3 EXL3 decode helpers (mlx-stream native): the GEMV lanes' reads of a trellis tile and their accumulators
// (header_glm53_common's codebook and rotation before this).

// The natural-order 128-point Walsh-Hadamard butterflies (unscaled) of a block held as v = (e[l], e[l + 32], e[l + 64],
// e[l + 96]) by lane l, the result in the same layout: exl3/header_dig2_x.metal's dig2_widen / dig2_round butterflies, on
// registers (theirs read and write device memory).
static inline float4 g53_fwht128(float4 v, ushort lane) {
  for (ushort bit = 1; bit <= 16; bit <<= 1) {
    const float4 p = simd_shuffle_xor(v, bit);
    v = ((lane & bit) == 0) ? v + p : p - v;
  }
  const float s0 = v.x + v.y;
  const float s1 = v.x - v.y;
  const float s2 = v.z + v.w;
  const float s3 = v.z - v.w;
  return float4(s0 + s2, s1 + s3, s0 - s2, s1 - s3);
}

// Where element e = l + 32 c of a 128-block lands in lane order: each 16-row tile keeps its rows (2q, 2q + 1, 2q + 8,
// 2q + 9) at 4q .. 4q + 3, the four a GEMV lane with l & 3 == q reads as one vector.
static inline uint g53_lane_slot(uint l) {
  return (l & 16u) + ((l & 7u) >> 1u) * 4u + ((l & 15u) >> 3u) * 2u + (l & 1u);
}

// A lane's reads of a tile, fixed per lane: the two words that hold its eight codewords (`a` the one its last
// codeword ends in, `b` the one before, circular) and the shift that ends the pair (b:a) at that bit.
struct g53_lane_plan {
  uint a;
  uint b;
  uint s;
};

template <int K>
static inline g53_lane_plan g53_plan(uint lane);

// K3: the lane's last codeword ends at bit 24 (l + 1).
template <>
inline g53_lane_plan g53_plan<3>(uint lane) {
  const uint a = (3u * lane + 2u) >> 2u;
  return {a, a == 0u ? 23u : a - 1u, 8u * ((lane + 1u) & 3u)};
}

// K4: the lane's codewords end at bits 32 l + 4 (j + 1): its own word and the 12 last bits of the one before.
template <>
inline g53_lane_plan g53_plan<4>(uint lane) {
  return {lane, (lane + 31u) & 31u, 0u};
}

// The eight codewords of the lane in the tile at `w`: cw[m] is weight 7 - m's, the 16 bits ending K m bits before the
// lane's last bit of v = (w[b] : w[a]) >> s.
template <int K>
static inline void g53_lane_codewords(const device uint* w, g53_lane_plan pl, thread uint* cw) {
  const uint lo = w[pl.a];
  const uint hi = w[pl.b];
  const uint v0 = (K == 4) ? lo : ((lo >> pl.s) | ((hi << 1u) << (31u - pl.s)));
  const uint vh = (K == 4) ? hi : (hi >> pl.s);
  for (uint m = 0u; m < 8u; ++m) {
    const uint sh = uint(K) * m;
    cw[m] = (sh <= 16u ? (v0 >> sh) : ((v0 >> sh) | (vh << (32u - sh)))) & 0xffffu;
  }
}

// The eight weights of the lane (w8[m] for codeword m).
template <int K>
static inline void g53_lane_weights(const device uint* w, g53_lane_plan pl, thread half* w8) {
  uint cw[8];
  g53_lane_codewords<K>(w, pl, cw);
  for (uint p = 0u; p < 4u; ++p) {
    const half2 h = g53_mcg2(uint2(cw[2u * p], cw[2u * p + 1u]));
    w8[2u * p] = h.x;
    w8[2u * p + 1u] = h.y;
  }
}

// One lane's eight products with its four inputs x4 = rows (r0, r0 + 1, r0 + 8, r0 + 9), r0 = 2 (l & 3).
static inline void g53_fma8(thread float* acc, float4 x4, thread const half* w8) {
  acc[0] = fma(x4.w, float(w8[0]), acc[0]);
  acc[1] = fma(x4.z, float(w8[1]), acc[1]);
  acc[2] = fma(x4.y, float(w8[2]), acc[2]);
  acc[3] = fma(x4.x, float(w8[3]), acc[3]);
  acc[4] = fma(x4.w, float(w8[4]), acc[4]);
  acc[5] = fma(x4.z, float(w8[5]), acc[5]);
  acc[6] = fma(x4.y, float(w8[6]), acc[6]);
  acc[7] = fma(x4.x, float(w8[7]), acc[7]);
}

// A lane's two column sums of a tile (column l >> 2 and l >> 2 + 8), over its rows and the four lanes of the column.
static inline float2 g53_cols(thread const float* acc) {
  float lo = (acc[4] + acc[5]) + (acc[6] + acc[7]);
  float hi = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  lo += simd_shuffle_xor(lo, ushort(1));
  hi += simd_shuffle_xor(hi, ushort(1));
  lo += simd_shuffle_xor(lo, ushort(2));
  hi += simd_shuffle_xor(hi, ushort(2));
  return float2(lo, hi);
}
