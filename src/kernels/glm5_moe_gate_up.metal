// Ported from oMLX (jundot/omlx) omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py,
// oMLX 0.7.0 (Apache-2.0, see NOTICE): MLX's qmv_fast arithmetic for one-token MoE decode.
#define HAS_SHARED 1
#define SHARED_WIDE 0
#define SELECT 0

  const uint simd_lid = thread_index_in_simdgroup;
  const uint simd_gid = simdgroup_index_in_threadgroup;
  // SM (slot-major): route slots vary fastest, and `order` lists them expert-sorted, so the
  // slots of an expert that several rows share read its row block back to back (cache hits).
  const int tile = SM ? int(threadgroup_position_in_grid.z) : int(threadgroup_position_in_grid.y);
  const int z = SM ? int(threadgroup_position_in_grid.y) : int(threadgroup_position_in_grid.z);
  const T lim = T(limit[0]);
  const T neg_lim = T(-limit[0]);
#if SHARED_WIDE
  // Last z slice: the shared expert for all NTOK tokens with MLX's
  // multi-row qmv_wide arithmetic (8 lanes per row, 4 rows per simdgroup).
  if (z == NTOK * TOPK) {
    const int k_lane = int(simd_lid) % 8;
    const int row = (tile * NSG + int(simd_gid)) * 4 + int(simd_lid) / 8;
    constexpr int WB = K * SBITS / 8;
    constexpr int G = K / SGS;
    float g_res[NTOK];
    float u_res[NTOK];
    for (int v = 0; v < NTOK; v++) {
      g_res[v] = 0.0f;
      u_res[v] = 0.0f;
    }
    glm_qmv_wide_row<T, K, SGS, SBITS, NTOK>(
        (const device uint8_t*)sh_gate_w + size_t(row) * WB, sh_gate_s + row * G,
        sh_gate_b + row * G, x, NTOK, k_lane, g_res);
    glm_qmv_wide_row<T, K, SGS, SBITS, NTOK>(
        (const device uint8_t*)sh_up_w + size_t(row) * WB, sh_up_s + row * G,
        sh_up_b + row * G, x, NTOK, k_lane, u_res);
    for (int v = 0; v < NTOK; v++) {
      g_res[v] += simd_shuffle_down(g_res[v], 4);
      g_res[v] += simd_shuffle_down(g_res[v], 2);
      g_res[v] += simd_shuffle_down(g_res[v], 1);
      u_res[v] += simd_shuffle_down(u_res[v], 4);
      u_res[v] += simd_shuffle_down(u_res[v], 2);
      u_res[v] += simd_shuffle_down(u_res[v], 1);
    }
    if (k_lane == 0) {
      for (int v = 0; v < NTOK; v++) {
        shared_out[size_t(v) * N + row] = glm_clamped_swiglu<T>(
            static_cast<T>(g_res[v]), static_cast<T>(u_res[v]), lim, neg_lim);
      }
    }
    return;
  }
  constexpr int RT = TOPK;
#else
  constexpr int RT = TOPK + HAS_SHARED;
#endif
  // Routed slots first (in `order` under SM), then one shared-expert slot per row.
  int token, r;
  if (z < NTOK * TOPK) {
    const int slot = SM ? int(order[z]) : z;
    token = slot / TOPK;
    r = slot - token * TOPK;
  } else {
    token = z - NTOK * TOPK;
    r = TOPK;
  }
  const int out_row = (tile * NSG + int(simd_gid)) * RPS;
  const device T* xr = x + token * K;

  float g_res[RPS] = {0};
  float u_res[RPS] = {0};
  if (r < TOPK) {
#if SELECT
    // One token: this simdgroup replays the router's selection on the
    // biased sigmoid scores (no separate select dispatch); slot 0 / tile 0
    // publishes the routes and routing weights for the down kernel.
    int picked[TOPK];
    glm_router_topk<NE, TOPK>(sel_biased, simd_lid, picked);
    if (z == 0 && tile == 0 && simd_gid == 0 && simd_lid == 0) {
      float total = 0.0f;
      float gathered[TOPK];
      for (int q = 0; q < TOPK; q++) {
        gathered[q] = sel_sig[picked[q]];
        total = gathered[q] + total;
      }
      for (int q = 0; q < TOPK; q++) {
        float qv = SEL_NORM ? gathered[q] / total : gathered[q];
        float sv = qv * sel_scaling[0];
        sel_indices[q] = uint(picked[q]);
        sel_scores[q] = sv;
      }
    }
    const int expert = picked[r];
#else
    const int expert = int(indices[token * TOPK + r]);
#endif
    constexpr int WB = K * RBITS / 8;   // bytes per weight row
    constexpr int G = K / RGS;          // groups per row
    // ESTRIDE rows per expert; a fused [gate; up] tensor (ESTRIDE = 2N) is
    // passed as both gate and up with the up rows UP_OFF = N further on.
    const size_t row0 = size_t(expert) * ESTRIDE + out_row;
    const size_t urow0 = row0 + UP_OFF;
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)gate_w + row0 * WB, gate_s + row0 * G,
        gate_b + row0 * G, xr, simd_lid, g_res);
    glm_qmv_rows<T, K, RGS, RBITS, RPS>(
        (const device uint8_t*)up_w + urow0 * WB, up_s + urow0 * G,
        up_b + urow0 * G, xr, simd_lid, u_res);
  } else {
#if HAS_SHARED
    constexpr int WB = K * SBITS / 8;
    constexpr int G = K / SGS;
    const size_t row0 = size_t(out_row);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_gate_w + row0 * WB, sh_gate_s + row0 * G,
        sh_gate_b + row0 * G, xr, simd_lid, g_res);
    glm_qmv_rows<T, K, SGS, SBITS, RPS>(
        (const device uint8_t*)sh_up_w + row0 * WB, sh_up_s + row0 * G,
        sh_up_b + row0 * G, xr, simd_lid, u_res);
#endif
  }
  device T* o = out + (size_t(token) * RT + r) * N + out_row;
  for (int row = 0; row < RPS; row++) {
    float gv = simd_sum(g_res[row]);
    float uv = simd_sum(u_res[row]);
    if (simd_lid == 0) {
      o[row] = glm_clamped_swiglu<T>(static_cast<T>(gv), static_cast<T>(uv), lim, neg_lim);
    }
  }
