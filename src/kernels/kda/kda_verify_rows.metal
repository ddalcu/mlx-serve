// A KDA layer's exact verify, TOK <= 8 rows of one sequence, in one launch per layer, one threadgroup (1024
// threads) a head (sdk/blocks/kda/verify_rows.zig). Row t is serial decode's one-row composed step on row t: each
// piece is the arithmetic of the op or kernel it replaces, in its lane order and with its roundings:
// 1. kernels/gdn/gdn_prework.metal's (its table arm): the conv over the window and the rows (f32 taps), the T-rounded
//    conv, its sigmoid read from `sigtab` (MLX's own sigmoid of every 16-bit pattern), the SiLU; q and k rms-normed over
//    their head (4 channels a lane, simd_sum, precise rsqrt, the ones-weight rounding, then the scale's); v as
//    activated; beta = sigtab[b].
// 2. kdaGateChain's ops: the per-channel bounded gate exp(lb * sigmoid(exp(A_log) * (a + dt_bias))), f32.
// 3. kernels/gdn/gated_delta.fmt.metal's (vector gate): the delta rule per value row, its dk lanes 4 channels each,
//    the two simd_sum's; simdgroup sg runs value rows sg, sg + 32, sg + 64 and sg + 96. After each row the state takes
//    the StT store and float load serial decode puts between two forwards; state_seq keeps it ([TOK, HV, DV, DK]).
// 4. kernels/gdn/gdn_normgate.metal's (the sigmoid gate, its table arm): rms over the head's values, precise rsqrt,
//    norm_w * T(x * inv), times sigtab[z].
// Every sigmoid of a T value is the table's: the formula's fast exp rounds a bf16 sigmoid apart from MLX's op on rare
// inputs, the table never. The next conv window is rows [TOK, TOK + NKEEP) of concat(conv_state, qkv), copied.
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint h = threadgroup_position_in_grid.x;

threadgroup T qs[TOK][DK];
threadgroup T ks[TOK][DK];
threadgroup T vs[TOK][DV];
threadgroup float gs[TOK][DK];
threadgroup T betas[TOK];
threadgroup T ys[TOK][DV];

// ---- 1. the prework (gdn_prework.metal): simdgroup 3 t + part, part 0 q, 1 k, 2 v ----
if (sg < uint(3 * TOK)) {
  const uint part = sg % 3;
  const uint t = sg / 3;
  const uint channel_base = part == 0 ? h * uint(DK)
                          : (part == 1 ? uint(HK) * uint(DK) + h * uint(DK)
                                       : 2 * uint(HK) * uint(DK) + h * uint(DV));
  T activated[4];
  float sumsq = 0.0f;
  for (uint i = 0; i < 4; ++i) {
    uint channel = channel_base + lane * 4 + i;
    float acc = 0.0f;
    for (uint tap = 0; tap < 4; ++tap) {
      uint input_row = t + tap;
      const T xv = input_row < uint(NKEEP)
          ? conv_state[input_row * uint(C) + channel]
          : qkv[(t + tap - uint(NKEEP)) * uint(C) + channel];
      acc += float(xv) * float(conv_w[channel * 4 + tap]);
    }
    const T conv = T(acc);
    const T sig = sigtab[as_type<ushort>(conv)];
    const T act = conv * sig;
    activated[i] = act;
    float value = float(act);
    sumsq += value * value;
  }
  if (part < 2) {
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DK) + qk_eps);
    const T scale = part == 0 ? q_scale : k_scale;
    for (uint i = 0; i < 4; ++i) {
      const T rms = T(1) * T(float(activated[i]) * inv);
      const T value = scale * rms;
      if (part == 0) {
        qs[t][lane * 4 + i] = value;
      } else {
        ks[t][lane * 4 + i] = value;
      }
    }
  } else {
    for (uint i = 0; i < 4; ++i) {
      vs[t][lane * 4 + i] = activated[i];
    }
    if (lane == 0) {
      const T bv = b[t * uint(HV) + h];
      betas[t] = sigtab[as_type<ushort>(bv)];
    }
  }
}

// ---- 2. the per-channel gate (kdaGateChain's ops) ----
for (uint e = tid; e < uint(TOK * DK); e += 1024) {
  const uint t = e / uint(DK);
  const uint i = e % uint(DK);
  const uint c = h * uint(DK) + i;
  const float ea = metal::precise::exp(static_cast<float>(A_log[c]));
  const float apd = static_cast<float>(a[t * uint(HV * DK) + c]) + static_cast<float>(dt_bias[c]);
  const float sc = ea * apd;
  const float sy = 1 / (1 + metal::precise::exp(metal::abs(sc)));
  const float gsg = (sc < 0) ? sy : 1 - sy;
  const float gl = gsg * lb;
  gs[t][i] = metal::precise::exp(gl);
}

// ---- the next conv window: rows [TOK, TOK + NKEEP) of concat(conv_state, qkv), this head's q, k and v channels ----
for (uint e = tid; e < uint(NKEEP * 3 * DK); e += 1024) {
  const uint srow = e / uint(3 * DK);
  const uint rem = e % uint(3 * DK);
  const uint part = rem / uint(DK);
  const uint i = rem % uint(DK);
  const uint channel = (part == 0 ? h * uint(DK)
                     : (part == 1 ? uint(HK) * uint(DK) + h * uint(DK)
                                  : 2 * uint(HK) * uint(DK) + h * uint(DV))) + i;
  const uint src = uint(TOK) + srow;
  conv_out[srow * uint(C) + channel] = src < uint(NKEEP) ? conv_state[src * uint(C) + channel]
                                                         : qkv[(src - uint(NKEEP)) * uint(C) + channel];
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- 3. the delta rule (gated_delta.fmt.metal, the vector gate), the state stored and loaded between rows ----
constexpr int n_per_t = DK / 32;
for (uint j = 0; j < uint(DV) / 32; ++j) {
  const uint dv_idx = sg + 32 * j;
  const uint dk_idx = lane;
  float state[n_per_t];
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = static_cast<float>(state_in[(h * uint(DV) + dv_idx) * uint(DK) + s_idx]);
  }
  for (uint t = 0; t < uint(TOK); ++t) {
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
    out = simd_sum(out);
    if (thread_index_in_simdgroup == 0) {
      ys[t][dv_idx] = static_cast<T>(out);
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      const StT st = static_cast<StT>(state[i]);
      state_seq[((t * uint(HV) + h) * uint(DV) + dv_idx) * uint(DK) + s_idx] = st;
      state[i] = static_cast<float>(st);
    }
  }
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state_out[(h * uint(DV) + dv_idx) * uint(DK) + s_idx] = static_cast<StT>(state[i]);
  }
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// ---- 4. the gated rms norm (gdn_normgate.metal, the sigmoid gate) ----
if (sg < uint(TOK)) {
  const uint t = sg;
  float xs[4];
  float sumsq = 0.0f;
  for (uint i = 0; i < 4; ++i) {
    xs[i] = float(ys[t][lane * 4 + i]);
    sumsq += xs[i] * xs[i];
  }
  sumsq = simd_sum(sumsq);
  float inv = metal::precise::rsqrt(sumsq / float(DV) + norm_eps);
  const uint base = t * uint(HV) * uint(DV) + h * uint(DV) + lane * 4;
  for (uint i = 0; i < 4; ++i) {
    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    const T zv = z[base + i];
    const T sig = sigtab[as_type<ushort>(zv)];
    y[base + i] = normed * sig;
  }
}
