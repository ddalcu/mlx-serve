// sushi_attn_pd on the matrix units: the same inputs, outputs and fp32 carries as the SIMD kernel
// (kr = {begin, end, koff, kL_abs}, phase bit 0 = carry-in, bit 1 = final; causal and band in cache
// coordinates; optional per-head sink), structured after MLX's steel attention_nax: each simdgroup
// owns 16 query rows, K/V fragments read straight from device memory (no threadgroup staging).
constexpr int NSG = 4;
constexpr int BQ = 16 * NSG;
constexpr int BK = 32;
constexpr int TDK = BDK / 16;
constexpr int TDV = BDV / 16;
constexpr int TK = BK / 16;
typedef metal::vec<float, 8> ffrag;
typedef metal::vec<T, 8> tfrag;

const int qL = q_shape[2];
const int kL = k_shape[2];
const int Hq = q_shape[1];
const int Hk = k_shape[1];
const int gqa = Hq / Hk;
const int tqx = int(threadgroup_position_in_grid.x);
const int hq = int(threadgroup_position_in_grid.y);
const int bb = int(threadgroup_position_in_grid.z);
const ushort warp = ushort(simdgroup_index_in_threadgroup);
const float scale_log2e = scl[0] * 1.44269504088896340736f;
const int SW = win[0];
const int k_begin = kr[0];
const int k_end = metal::min(kr[1], kL);
const int koff = kr[2];
const int kL_abs = kr[3];
const bool has_carry = (phase[0] & 1) != 0;
const bool is_final = (phase[0] & 2) != 0;
const int q_off = kL_abs - qL;

const int row0 = tqx * BQ + int(warp) * 16;
const int q_rows = metal::min(16, qL - row0);
// Causal: the simdgroups walk their threadgroup's key range in step (the barriers below keep each
// K/V block in cache for all four). Band: each walks its own rows' band, no barriers.
const bool lockstep = SW == 0;
if (!lockstep && q_rows <= 0) return;
const device T* Qh = q + bb * q_strides[0] + hq * q_strides[1];
const device T* Kp = k + bb * k_strides[0] + (hq / gqa) * k_strides[1];
const device T* Vp = v + bb * v_strides[0] + (hq / gqa) * v_strides[1];
const int ldq = int(q_strides[2]), ldk = int(k_strides[2]), ldv = int(v_strides[2]);

// This lane holds rows cc.y and cc.y + 8 of the simdgroup's 16, at absolute cache positions:
const short2 cc = SushiNax::coord();
const int r_abs0 = q_off + row0 + cc.y;
const int r_abs1 = r_abs0 + 8;
// Loads past the last query or key row clamp to it, so no load branches: those rows' scores are
// never stored (queries) or are masked to -inf (keys), and a zero P times a finite V adds zero.
const device T* Q0 = Qh + (long)metal::min(row0 + cc.y, qL - 1) * ldq + cc.x;
const device T* Q1 = Qh + (long)metal::min(row0 + cc.y + 8, qL - 1) * ldq + cc.x;

ffrag O[TDV];
SUSHI_UNROLL for (short i = 0; i < TDV; ++i) O[i] = ffrag(0.0f);
// Finite initial max (see the SIMD kernel): exp2(old - new) never sees inf - inf.
float2 mx_s = float2(-3.0e38f);
float2 sm_s = float2(0.0f);
// Carries are indexed directly, never through a typed pointer: the no-carry dummies bind in
// `constant`, real carries in `device`.
if (has_carry) {
  SUSHI_UNROLL for (short r = 0; r < 2; ++r) {
    const int qr = cc.y + r * 8;
    if (qr < q_rows) {
      const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(row0 + qr));
      mx_s[r] = m_in[crow];
      sm_s[r] = l_in[crow];
      SUSHI_UNROLL for (short d = 0; d < TDV; ++d) SUSHI_UNROLL for (short j = 0; j < 4; ++j)
        O[d][r * 4 + j] = o_in[crow * BDV + d * 16 + cc.x + j];
    }
  }
}
if (SINK && !has_carry) {
  mx_s = float2(sinks[hq] * 1.44269504088896340736f);
  sm_s = float2(1.0f);
}

const int NK = (k_end + BK - 1) / BK;
const int q_lo = (lockstep ? tqx * BQ : row0) + q_off;
const int q_hi = (lockstep ? tqx * BQ + BQ - 1 : row0 + 15) + q_off;
const int kb_lim = metal::min(NK, (q_hi - koff + BK) / BK);
int kb = k_begin / BK;
if (SW > 0) kb = metal::max(kb, metal::max(0, q_lo - SW + 1 - koff) / BK);
const int kb_min_causal = metal::max(0, q_lo - koff) / BK;

for (; kb < kb_lim; kb++) {
  const int c0 = kb * BK;
  const int rows_k = metal::min(BK, kL - c0);
  const bool full_k = rows_k == BK;
  ffrag S[TK];
  SUSHI_UNROLL for (short i = 0; i < TK; ++i) S[i] = ffrag(0.0f);
  SUSHI_UNROLL for (short ik = 0; ik < TK; ik += 2) {
    const int kr = c0 + ik * 16 + cc.y;
    const device T* K0 = Kp + (long)metal::min(kr, kL - 1) * ldk + cc.x;
    const device T* K1 = Kp + (long)metal::min(kr + 8, kL - 1) * ldk + cc.x;
    const device T* K2 = Kp + (long)metal::min(kr + 16, kL - 1) * ldk + cc.x;
    const device T* K3 = Kp + (long)metal::min(kr + 24, kL - 1) * ldk + cc.x;
#pragma clang loop unroll_count(4)
    for (short d = 0; d < TDK; ++d) {
      tfrag qf, k0f, k1f;
      SushiNax::load2(qf, Q0 + d * 16, Q1 + d * 16);
      SushiNax::load2(k0f, K0 + d * 16, K1 + d * 16);
      SushiNax::load2(k1f, K2 + d * 16, K3 + d * 16);
      SushiNax::mma<float, T, T, false, true>(S[ik], S[ik + 1], qf, k0f, k1f);
    }
  }
  SUSHI_UNROLL for (short i = 0; i < TK; ++i) S[i] *= scale_log2e;

  // kL remainder + causal + sliding band, element-wise in cache coordinates.
  const bool need_causal = kb >= kb_min_causal;
  const bool need_band = (SW > 0) && (koff + c0 <= q_hi - SW);
  if (!full_k || need_causal || need_band) {
    SUSHI_UNROLL for (short ik = 0; ik < TK; ++ik) SUSHI_UNROLL for (short j = 0; j < 8; ++j) {
      const int col = c0 + ik * 16 + cc.x + (j % 4);
      const int col_abs = koff + col;
      const int ra = (j < 4) ? r_abs0 : r_abs1;
      if (col >= kL || ra < col_abs || (SW > 0 && (ra - col_abs) >= SW)) S[ik][j] = -INFINITY;
    }
  }

  float2 nm = mx_s;
  SUSHI_UNROLL for (short ik = 0; ik < TK; ++ik) SUSHI_UNROLL for (short r = 0; r < 2; ++r) {
    float t = metal::max(metal::max(S[ik][r * 4], S[ik][r * 4 + 1]), metal::max(S[ik][r * 4 + 2], S[ik][r * 4 + 3]));
    t = metal::max(t, metal::simd_shuffle_xor(t, ushort(1)));
    t = metal::max(t, metal::simd_shuffle_xor(t, ushort(8)));
    nm[r] = metal::max(nm[r], t);
  }
  SUSHI_UNROLL for (short ik = 0; ik < TK; ++ik) SUSHI_UNROLL for (short j = 0; j < 8; ++j)
    S[ik][j] = metal::exp2(S[ik][j] - nm[j / 4]);
  float2 rs = float2(0.0f);
  SUSHI_UNROLL for (short ik = 0; ik < TK; ++ik) SUSHI_UNROLL for (short r = 0; r < 2; ++r) {
    float t = (S[ik][r * 4] + S[ik][r * 4 + 1]) + (S[ik][r * 4 + 2] + S[ik][r * 4 + 3]);
    t += metal::simd_shuffle_xor(t, ushort(1));
    t += metal::simd_shuffle_xor(t, ushort(8));
    rs[r] += t;
  }
  const float2 fac = metal::exp2(mx_s - nm);
  mx_s = nm;
  sm_s = sm_s * fac + rs;
  SUSHI_UNROLL for (short d = 0; d < TDV; ++d) SUSHI_UNROLL for (short j = 0; j < 8; ++j) O[d][j] *= fac[j / 4];

  // O += P @ V with P in f16: P is in [0, 1], where f16 keeps 11 bits to bf16's 8 (a float P operand
  // is truncated by the relaxed matmul).
  metal::vec<half, 8> ph[TK];
  SUSHI_UNROLL for (short ik = 0; ik < TK; ++ik) SUSHI_UNROLL for (short j = 0; j < 8; ++j) ph[ik][j] = half(S[ik][j]);
  metal::simdgroup_barrier(metal::mem_flags::mem_none);
  SUSHI_UNROLL for (short d = 0; d < TDV; d += 2) {
    if (lockstep && d == TDV / 2) metal::threadgroup_barrier(metal::mem_flags::mem_none);
    SUSHI_UNROLL for (short ik = 0; ik < TK; ++ik) {
      const int vr = c0 + ik * 16 + cc.y;
      const device T* V0 = Vp + (long)metal::min(vr, kL - 1) * ldv + cc.x + d * 16;
      const device T* V1 = Vp + (long)metal::min(vr + 8, kL - 1) * ldv + cc.x + d * 16;
      tfrag v0f, v1f;
      SushiNax::load2(v0f, V0, V1);
      SushiNax::load2(v1f, V0 + 16, V1 + 16);
      SushiNax::mma<float, half, T, false, false>(O[d], O[d + 1], ph[ik], v0f, v1f);
    }
  }
}

// Final chunk: normalize + store bf16. Mid chunk: the raw fp32 state for the next dispatch.
SUSHI_UNROLL for (short r = 0; r < 2; ++r) {
  const int qr = cc.y + r * 8;
  if (qr < q_rows) {
    const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(row0 + qr));
    if (is_final) {
      const float inv = 1.0f / sm_s[r];
      device T* Orow = out + crow * BDV;
      SUSHI_UNROLL for (short d = 0; d < TDV; ++d) SUSHI_UNROLL for (short j = 0; j < 4; ++j)
        Orow[d * 16 + cc.x + j] = T(O[d][r * 4 + j] * inv);
    } else {
      m_out[crow] = mx_s[r];
      l_out[crow] = sm_s[r];
      SUSHI_UNROLL for (short d = 0; d < TDV; ++d) SUSHI_UNROLL for (short j = 0; j < 4; ++j)
        o_out[crow * BDV + d * 16 + cc.x + j] = O[d][r * 4 + j];
    }
  }
}
