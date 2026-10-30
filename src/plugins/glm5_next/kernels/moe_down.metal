// Ported from oMLX (jundot/omlx) omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py,
// oMLX 0.7.0 (Apache-2.0, see NOTICE): MLX's qmv_fast arithmetic for one-token MoE decode.
#define HAS_SHARED 1
#define ADD_SHARED_Y 0
#define SHARED_WIDE_DOWN 0

  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  // SM (slot-major): rows vary fastest, so experts they share are read back to back.
  const int tile = SM ? int(threadgroup_position_in_grid.z) : int(threadgroup_position_in_grid.y);
  const int token = SM ? int(threadgroup_position_in_grid.y) : int(threadgroup_position_in_grid.z);
  // Activation slots per token: the routed ones, then the shared expert's.
  constexpr int RT = TOPK + HAS_SHARED;
  const int out_row = (tile * NSG + int(simd_gid)) * RPS;

  float acc[RPS] = {0};
  constexpr int WB = K * RBITS / 8;
  constexpr int G = K / RGS;
  for (int r = 0; r < TOPK; r++) {
    const int expert = int(indices[token * TOPK + r]);
    const size_t row0 = size_t(expert) * N + out_row;
    float res[RPS] = {0};
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)down_w + row0 * WB, down_s + row0 * G,
        down_b + row0 * G, act + (size_t(token) * RT + r) * K, simd_lid, res);
    const float score = scores[token * TOPK + r];
    for (int row = 0; row < RPS; row++) {
      float v = simd_sum(res[row]);
      // The reference rounds the fp32 product in its own Multiply kernel
      // before the Sum; keep the compiler from contracting it into an FMA.
      volatile float weighted = static_cast<float>(static_cast<T>(v)) * score;
      acc[row] += weighted;
    }
  }
#if HAS_SHARED
  float sres[RPS] = {0};
  {
    constexpr int SWB = K * SBITS / 8;
    constexpr int SG = K / SGS;
    const size_t row0 = size_t(out_row);
    const device T* sh_x = act + (size_t(token) * RT + TOPK) * K;
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_down_w + row0 * SWB, sh_down_s + row0 * SG,
        sh_down_b + row0 * SG, sh_x, simd_lid, sres);
  }
#endif
#if SHARED_WIDE_DOWN
  // Shared expert down projection of this token with the multi-row qmv_wide
  // arithmetic its own [T, K] matmul uses (8 lanes per row; each token's
  // accumulation is independent of the others).
  {
    static_assert(RPS == 4, "qmv_wide rows per simdgroup");
    constexpr int SWB = K * SBITS / 8;
    constexpr int SG = K / SGS;
    const int k_lane = int(simd_lid) % 8;
    const int srow = out_row + int(simd_lid) / 8;
    float sv[1] = {0.0f};
    glm_qmv_wide_row<T, K, SGS, SBITS, 1>(
        (const device uint8_t*)sh_down_w + size_t(srow) * SWB, sh_down_s + srow * SG,
        sh_down_b + srow * SG, sh_act + size_t(token) * K, 1, k_lane, sv);
    sv[0] += simd_shuffle_down(sv[0], 4);
    sv[0] += simd_shuffle_down(sv[0], 2);
    sv[0] += simd_shuffle_down(sv[0], 1);
    if (k_lane == 0) {
      const int r = int(simd_lid) / 8;
      float a = acc[0];
      for (int row = 1; row < RPS; row++) {
        a = row == r ? acc[row] : a;
      }
      out[size_t(token) * N + srow] = static_cast<T>(a) + static_cast<T>(sv[0]);
    }
  }
  return;
#endif
  device T* o = out + size_t(token) * N + out_row;
  for (int row = 0; row < RPS; row++) {
#if HAS_SHARED
    float sv = simd_sum(sres[row]);
#endif
    if (simd_lid == 0) {
#if HAS_SHARED
      o[row] = static_cast<T>(acc[row]) + static_cast<T>(sv);
#elif ADD_SHARED_Y
      o[row] = static_cast<T>(acc[row]) + shared_y[size_t(token) * N + out_row + row];
#else
      o[row] = static_cast<T>(acc[row]);
#endif
    }
  }
