// GLM-5.3 EXL3 GEMM helpers (mlx-stream native), K 3 and 4: after the MLX steel NAX tiles (exl3/header_dig_mul1h_k3.metal,
// DeepSeek-V4.1's text, embedded as it is) and header_glm53_common's codebook.
using namespace mlx::steel;

// The 16 bits ending (exclusive) at bit E of the words P (bit 0 = P[0]'s MSB), E a constant after unrolling.
METAL_FUNC uint g53_bits16(thread const uint* P, uint E) {
  const uint last = E - 1u;
  const uint w = last >> 5u;
  const uint sh = 31u - (last & 31u);
  return (sh <= 16u ? (P[w] >> sh) : ((P[w] >> sh) | (P[w - 1u] << (32u - sh)))) & 0xffffu;
}

// One column job of a 16 x 16 trellis tile of 8 K u32 words: column c = 8 H + c7 holds the weights 4 H .. 4 H + 3 of
// lanes 4 c7 .. 4 c7 + 3 (tensor-core order: lane 4 c7 + i's weight 4 H + jj sits at k-row 2 i + {0, 1, 8, 9}[jj]).
// The column's bits start in word K c7 + D (D below), so a thread's word offsets are fixed: o[q] = (K c7 + D + q) mod 8 K.
template <int K, int H>
struct g53_col {
  static constant constexpr int D = (K * (4 * H + 1) >= 16) ? (K * (4 * H + 1) - 16) / 32 : -1;
  static constant constexpr uint NQ = (K == 4 && H == 0) ? 5u : 4u;

  METAL_FUNC static void offsets(uint c7, thread uint* o) {
    constexpr uint NW = 8u * uint(K);
    const uint w0 = (uint(K) * c7 + uint(int(NW) + D)) % NW;
    STEEL_PRAGMA_UNROLL
    for (uint q = 0; q < NQ; ++q) o[q] = (w0 + q) % NW;
  }

  // The tile's words this thread's column reads (a load the decode can trail).
  METAL_FUNC static void words(const device uint* tile, thread const uint* o, thread uint* P) {
    STEEL_PRAGMA_UNROLL
    for (uint q = 0; q < NQ; ++q) P[q] = tile[o[q]];
  }

  // The column's 16 weights at k-rows 0 .. 15 from its words (`words`) into v.
  METAL_FUNC static void decode_p(thread const uint* P, thread float* v) {
    STEEL_PRAGMA_UNROLL
    for (uint i = 0; i < 4u; ++i) {
      STEEL_PRAGMA_UNROLL
      for (uint pr = 0; pr < 2u; ++pr) {
        const uint jj = 2u * pr;
        const uint e0 = uint(K) * (8u * i + 4u * uint(H) + jj + 1u) - uint(32 * D);
        const half2 w = g53_mcg2(uint2(g53_bits16(P, e0), g53_bits16(P, e0 + uint(K))));
        v[2u * i + 8u * pr] = float(w.x);
        v[2u * i + 8u * pr + 1u] = float(w.y);
      }
    }
  }

  // The column's 16 weights at k-rows 0 .. 15 into v.
  METAL_FUNC static void decode_v(const device uint* tile, thread const uint* o, thread float* v) {
    uint P[NQ];
    words(tile, o, P);
    decode_p(P, v);
  }

  // 16 weights v into dst (two 16-byte stores), as T.
  template <typename T>
  METAL_FUNC static void store(thread const float* v, threadgroup T* dst) {
    threadgroup uint4* d4 = reinterpret_cast<threadgroup uint4*>(dst);
    uint pk[8];
    STEEL_PRAGMA_UNROLL
    for (uint q = 0; q < 8u; ++q) {
      pk[q] = uint(as_type<ushort>(static_cast<T>(v[2u * q]))) | (uint(as_type<ushort>(static_cast<T>(v[2u * q + 1u]))) << 16u);
    }
    d4[0] = uint4(pk[0], pk[1], pk[2], pk[3]);
    d4[1] = uint4(pk[4], pk[5], pk[6], pk[7]);
  }

  // The column's 16 weights at k-rows 0 .. 15 into dst (two 16-byte stores), as T.
  template <typename T>
  METAL_FUNC static void decode(const device uint* tile, thread const uint* o, threadgroup T* dst) {
    float v[16];
    decode_v(tile, o, v);
    store<T>(v, dst);
  }
};

// A 128-block's t128 (natural order, unscaled) along k of one column held by 4 lanes: lane bits 0-1 the quarter q, v the
// 32 values k = 32 q .. 32 q + 31 (bits 0-4 in the thread, bits 5-6 across lanes xor 1 and 2).
METAL_FUNC void g53_t128_quarters(thread float* v, uint lane) {
  STEEL_PRAGMA_UNROLL
  for (uint st = 1u; st < 32u; st <<= 1u) {
    STEEL_PRAGMA_UNROLL
    for (uint i = 0u; i < 32u; ++i) {
      if ((i & st) == 0u) {
        const float a = v[i];
        const float b = v[i + st];
        v[i] = a + b;
        v[i + st] = a - b;
      }
    }
  }
  STEEL_PRAGMA_UNROLL
  for (uint lb = 1u; lb <= 2u; lb <<= 1u) {
    const bool upper = (lane & lb) != 0u;
    STEEL_PRAGMA_UNROLL
    for (uint i = 0u; i < 32u; ++i) {
      const float o = simd_shuffle_xor(v[i], ushort(lb));
      v[i] = upper ? (o - v[i]) : (v[i] + o);
    }
  }
}
