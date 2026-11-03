// Ported from oMLX (jundot/omlx) omlx/patches/mlx_vlm_glm5_next_compat/decode_kernels.py,
// oMLX 0.7.0 (Apache-2.0, see NOTICE): the KDA layer body for one sequence and T <= 8 tokens.
// GATE5 is prepended by the caller.
#define HAS_CONV_STATE 1
#define HAS_STATE 1
#define PRE_AG 0
#define SIG_B_PRECISE 1
#define SIG_G_PRECISE 1

  constexpr int CK = 4;
  constexpr int NROW = CK - 1 + TOK;
  constexpr int NP = 3 * QKV;
  const uint tid = thread_position_in_threadgroup.x;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const int h = int(threadgroup_position_in_grid.x);
  const float q_scale = consts[0];
  const float l2_eps = consts[1];
  const float norm_eps = consts[2];
  const float lower = consts[3];
  const float inv_n = consts[4];

  threadgroup T qs[TOK][DK];
  threadgroup T ks[TOK][DK];
  threadgroup T vs[TOK][DK];
  threadgroup T as_[TOK][DK];
  threadgroup T gates[TOK][DK];
  threadgroup T ys[TOK][DK];
  threadgroup float gs[TOK][DK];
  threadgroup T betas[TOK];

  // ---- 1. short conv + SiLU -------------------------------------------------
  if (tid < uint(3 * DK)) {
    const int part = int(tid) / DK;
    const int i = int(tid) % DK;
    const int gc = part * QKV + h * DK + i;
    T win[NROW];
    for (int r = 0; r < CK - 1; r++) {
#if HAS_CONV_STATE
      win[r] = conv_state[r * NP + gc];
#else
      win[r] = static_cast<T>(0);
#endif
    }
    for (int t = 0; t < TOK; t++) {
      win[CK - 1 + t] = proj[t * PROJ_W + gc];
    }
    const device T* w = conv_w + gc * CK;
    for (int t = 0; t < TOK; t++) {
      float acc = 0.0;
      for (int j = 0; j < CK; ++j) {
        acc += static_cast<float>(win[t + j]) * w[j];
      }
      T co = static_cast<T>(acc);
      T sgm = glm_sigmoid<T>(co);
      T sv = co * sgm;
      if (part == 0) {
        qs[t][i] = sv;
      } else if (part == 1) {
        ks[t][i] = sv;
      } else {
        vs[t][i] = sv;
      }
    }
    for (int r = 0; r < CK - 1; r++) {
      conv_state_out[r * NP + gc] = win[TOK + r];
    }
  }

  // ---- 3a. low-rank gate projections (qmv_quad rows of this head) ----------
#if PRE_AG
  for (int e = int(tid); e < TOK * DK; e += 1024) {
    const int t = e / DK;
    const int i = e % DK;
    as_[t][i] = a_pre[t * QKV + h * DK + i];
    gates[t][i] = gate_pre[t * QKV + h * DK + i];
  }
#elif GATE5
  // One token, 5-bit K = 128 rows: MLX's qmv (qmv_impl): lanes 0..15 load
  // 8 values each (load_vector_safe / qdot_safe with N = 8, i.e. load_vector
  // / qdot), lanes 16..31 add nothing, one simd_sum per row.
  {
    static_assert(TOK == 1 && DK == 128, "5-bit gate rows: one token");
    constexpr int WBYTES = 128 * 5 / 8;           // 80 bytes per weight row
    constexpr int G = 128 / GS;                   // groups per row
    for (int rr = 0; rr < (2 * DK) / 32; rr++) {
      const int q = int(sg) * ((2 * DK) / 32) + rr;
      const int which = q / DK;
      const int i = q % DK;
      const int row = h * DK + i;
      float result = 0;
      if (lane < 16u) {
        const device T* xin = proj + (which == 0 ? OFF_FA : OFF_GA) + int(lane) * 8;
        float x_thread[8];
        float sum = glm_load_vector<T, 8, 5>(xin, x_thread);
        const device uint8_t* wl = (const device uint8_t*)(which == 0 ? fb_w : gb_w)
            + size_t(row) * WBYTES + int(lane) * 5;
        const device T* sl = (which == 0 ? fb_s : gb_s) + row * G + int(lane) / (GS / 8);
        const device T* bl = (which == 0 ? fb_b : gb_b) + row * G + int(lane) / (GS / 8);
        const float s = sl[0];
        const float b = bl[0];
        result += glm_qdot<8, 5>(wl, x_thread, s, b, sum);
      }
      result = simd_sum(result);
      if (lane == 0) {
        if (which == 0) {
          as_[0][i] = static_cast<T>(result);
        } else {
          gates[0][i] = static_cast<T>(result);
        }
      }
    }
  }
#else
  {
    constexpr int VPT = 32;                       // values per thread (K = 128)
    constexpr int WBYTES = 128 * BITS / 8;        // bytes per weight row
    constexpr int G = 128 / GS;                   // groups per row
    const int quad = int(tid) / 4;
    const int ql = int(tid) % 4;
    const int which = quad / DK;
    const int i = quad % DK;
    const int row = h * DK + i;
    const device uint8_t* wl = (const device uint8_t*)(which == 0 ? fb_w : gb_w)
        + size_t(row) * WBYTES + ql * (VPT * BITS / 8);
    const device T* sl = (which == 0 ? fb_s : gb_s) + row * G + ql / (GS / VPT);
    const device T* bl = (which == 0 ? fb_b : gb_b) + row * G + ql / (GS / VPT);
    const float s = sl[0];
    const float b = bl[0];
    for (int t = 0; t < TOK; t++) {
      const device T* xin = proj + t * PROJ_W + (which == 0 ? OFF_FA : OFF_GA) + ql * VPT;
      float x_thread[VPT];
      float sum = glm_load_vector<T, VPT, BITS>(xin, x_thread);
      float result = 0;
      result += glm_qdot<VPT, BITS>(wl, x_thread, s, b, sum);
      result = quad_sum(result);
      if (ql == 0) {
        if (which == 0) {
          as_[t][i] = static_cast<T>(result);
        } else {
          gates[t][i] = static_cast<T>(result);
        }
      }
    }
  }
#endif
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 2. l2norm(q) * scale, l2norm(k) --------------------------------------
  if (sg < uint(2 * TOK)) {
    const int t = int(sg) / 2;
    const bool is_q = (sg % 2) == 0;
    threadgroup T* row = is_q ? qs[t] : ks[t];
    float x[4];
    float tot = 0.0f;
    for (int e = 0; e < 4; e++) {
      x[e] = static_cast<float>(row[4 * lane + e]);
      float sq = x[e] * x[e];
      tot = sq + tot;
    }
    tot = simd_sum(tot);
    float u = tot + l2_eps;
    float r = metal::precise::rsqrt(u);
    for (int e = 0; e < 4; e++) {
      float xn = x[e] * r;
      if (is_q) {
        float xs = xn * q_scale;
        row[4 * lane + e] = static_cast<T>(xs);
      } else {
        row[4 * lane + e] = static_cast<T>(xn);
      }
    }
  }

  // ---- 3b. g = exp(lower * sigmoid(exp(A_log) * (a + dt_bias))), beta -------
  if (tid < uint(TOK * DK)) {
    const int t = int(tid) / DK;
    const int i = int(tid) % DK;
    float ea = metal::precise::exp(a_log[h]);
    float af = static_cast<float>(as_[t][i]);
    float s1 = af + dt_bias[h * DK + i];
    float s2 = ea * s1;
    float s3 = glm_sigmoid<float>(s2);
    float s4 = lower * s3;
    gs[t][i] = metal::precise::exp(s4);
  }
  if (tid < uint(TOK)) {
#if SIG_B_PRECISE
    betas[tid] = glm_sigmoid_precise<T>(proj[tid * PROJ_W + OFF_B + h]);
#else
    betas[tid] = glm_sigmoid<T>(proj[tid * PROJ_W + OFF_B + h]);
#endif
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 4. vector-gated delta rule --------------------------------------------
  for (int j = 0; j < DK / 32; j++) {
    const int dv_idx = int(sg) + 32 * j;
    constexpr int n_per_t = DK / 32;
    const int dk_idx = int(lane);
    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
#if HAS_STATE
      state[i] = static_cast<float>(state_in[(size_t(h) * DK + dv_idx) * DK + s_idx]);
#else
      state[i] = 0.0f;
#endif
    }
    for (int t = 0; t < TOK; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * gs[t][s_idx];
        kv_mem += state[i] * ks[t][s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (vs[t][dv_idx] - kv_mem) * betas[t];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + ks[t][s_idx] * delta;
        out += state[i] * qs[t][s_idx];
      }
      if (CAP) {
        // A spec verify keeps every row's state for the partial-accept rollback.
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state_seq[((size_t(t) * (QKV / DK) + h) * DK + dv_idx) * DK + s_idx] = state[i];
        }
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        ys[t][dv_idx] = static_cast<T>(out);
      }
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state_out[(size_t(h) * DK + dv_idx) * DK + s_idx] = static_cast<float>(state[i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // ---- 5. RMSNormGated ---------------------------------------------------------
  if (sg < uint(TOK)) {
    const int t = int(sg);
    float x[4];
    float tot = 0.0f;
    for (int e = 0; e < 4; e++) {
      x[e] = static_cast<float>(ys[t][4 * lane + e]);
      float sq = x[e] * x[e];
      tot = sq + tot;
    }
    tot = simd_sum(tot);
    float var = tot * inv_n;
    float u = var + norm_eps;
    float r = metal::precise::rsqrt(u);
    for (int e = 0; e < 4; e++) {
      const int c = 4 * lane + e;
      float xn = x[e] * r;
      float wf = static_cast<float>(norm_w[c]);
      float wx = wf * xn;
      float gf = static_cast<float>(gates[t][c]);
#if SIG_G_PRECISE
      float gsg = glm_sigmoid_precise<float>(gf);
#else
      float gsg = glm_sigmoid<float>(gf);
#endif
      float o = wx * gsg;
      y[t * QKV + h * DK + c] = static_cast<T>(o);
    }
  }
