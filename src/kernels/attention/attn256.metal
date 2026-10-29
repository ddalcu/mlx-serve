constexpr int BQ = 64;
constexpr int BK = 32;
// BDK = q/k head width, BDV = v head width (256/256, or MiMo's 192/128).
constexpr int LDK = BK + 8;
constexpr int LDV = BDV + 8;
constexpr int NT = 256;

const int qL = q_shape[2];
const int kL = k_shape[2];
const int Hq = q_shape[1];
const int Hk = k_shape[1];
const int gqa = Hq / Hk;

const int tqx = int(threadgroup_position_in_grid.x);
const int hq = int(threadgroup_position_in_grid.y);
const int bb = int(threadgroup_position_in_grid.z);
const ushort lane = ushort(thread_index_in_simdgroup);
const ushort warp = ushort(simdgroup_index_in_threadgroup);
const int tix = int(thread_index_in_threadgroup);

// exp2-based softmax (steel convention): fold log2(e) into the scale.
const float scale_log2e = scl[0] * 1.44269504088896340736f;
const int SW = win[0];

// Budgeted kv-chunk dispatch (causal arm): this dispatch covers keys
// [k_begin, k_end); online-softmax state (m/l/unnormalized O) rides fp32
// carry buffers between dispatches — exact register precision, so the
// chunked chain is bit-identical to one long dispatch. phase bit 0 =
// carry-in present, bit 1 = final chunk (normalize + bf16 store).
const int k_begin = kr[0];
const int k_end = metal::min(kr[1], kL);
const bool has_carry = (phase[0] & 1) != 0;
const bool is_final = (phase[0] & 2) != 0;

// Bottom-right aligned causal: query row r of this chunk sits at
// absolute KV position q_off + r.
const int q_off = kL - qL;

const device T* Qp = q + bb * q_strides[0] + hq * q_strides[1]
    + (long)(tqx * BQ) * q_strides[2];
const device T* Kp = k + bb * k_strides[0] + (hq / gqa) * k_strides[1];
const device T* Vp = v + bb * v_strides[0] + (hq / gqa) * v_strides[1];
device T* Op = out + (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ)) * BDV;

// The K^T tile [BDK][LDK] and the V tile [BK][LDV] share one buffer,
// steel-style (barriers separate the K reads from the V staging).
threadgroup T KVs[LDK * BDK > BK * LDV ? LDK * BDK : BK * LDV];
threadgroup T* Ks = KVs;
threadgroup T* Vs = KVs;

const int q_rows = metal::min(BQ, qL - tqx * BQ);

const short2 sc = msv_coord(lane);
const short sn = sc.x;
const short sm = sc.y;
const short tm = 8 * short(warp);
const int Ks_off = sm * LDK + sn;
const int Vs_off = sm * LDV + sn;

// Q fragment straight from global into registers, once per threadgroup:
// row (tm+sm), element pairs (dd*8+sn, +1). sn is always even, so the
// vec2 load is 4B-aligned for any row whose stride is even (structural:
// q rows stride 256). Rows past the ragged tail read zero — their
// outputs are discarded by the store guard.
float2 Qfrag[BDK / 8];
{
  const int qr = tm + sm;
  if (qr < q_rows) {
    const device T* Qrow = Qp + (long)qr * q_strides[2];
    for (int dd = 0; dd < BDK / 8; ++dd) {
      const vec<T, 2> p = *((const device vec<T, 2>*)(Qrow + dd * 8 + sn));
      Qfrag[dd] = float2(float(p.x), float(p.y));
    }
  } else {
    for (int dd = 0; dd < BDK / 8; ++dd) Qfrag[dd] = float2(0.0f);
  }
}

float2 Ofrag[BDV / 8];
for (int i = 0; i < BDV / 8; ++i) Ofrag[i] = float2(0.0f);
// Init max FINITE (not -inf) so the rescale factor exp2(old-new) can
// never be exp2(nan); masked scores use true -INFINITY so a row whose
// first blocks are fully banded out contributes exp2(-inf)=0, not 1.
float max_score = -3.0e38f;
float sum_score = 0.0f;
// A learned per-head sink is one more softmax column with no value row:
// it seeds the running max and sum (exp2(0) = 1), never O.
if (SINK) {
  max_score = float(sinks[hq]) * 1.44269504088896340736f;
  sum_score = 1.0f;
}

// Resume carried softmax state from the previous kv chunk. Rows past
// the ragged q tail never carry (their outputs are discarded anyway).
// (m_in/l_in/o_in are indexed directly, never through a typed pointer:
// MLX binds sub-4KB inputs — the no-carry dummies — in the `constant`
// address space and real carries in `device`, and the source must
// compile under both.)
if (has_carry) {
  const int qr = tm + sm;
  if (qr < q_rows) {
    const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ + qr));
    max_score = m_in[crow];
    sum_score = l_in[crow];
    const long obase = crow * (long)BDV;
    for (int dd = 0; dd < BDV / 8; ++dd) {
      Ofrag[dd] = float2(o_in[obase + dd * 8 + sn], o_in[obase + dd * 8 + sn + 1]);
    }
  }
}

const int NK = (k_end + BK - 1) / BK;
const int q_lo = tqx * BQ + q_off;
const int q_hi = q_lo + BQ - 1;
const int kb_lim = metal::min(NK, (q_hi + BK) / BK);
int kb = k_begin / BK;
if (SW > 0) kb = metal::max(kb, metal::max(0, q_lo - SW + 1) / BK);
const int kb_min_causal = metal::max(0, q_lo) / BK;
const int row_pos = q_lo + tm + sm;

// QSA arm (template QSA=1): `mask` is the [B,1,qL,kL] bool visibility
// (causal folded in), `skip` the [B, nq_tiles, NK] per-tile any-visible
// table — a key block no row of this tile sees is never staged.
const long skip_base = ((long)bb * ((qL + BQ - 1) / BQ) + tqx) * (long)((kL + BK - 1) / BK);
const long mask_row0 = QSA ? ((long)bb * mask_strides[0] + (long)(tqx * BQ + tm + sm) * mask_strides[2]) : 0;
const bool mask_row_ok = (tqx * BQ + tm + sm) < qL;

for (; kb < kb_lim; kb++) {
  if (QSA && !skip[skip_base + kb]) continue;
  const int c0 = kb * BK;
  const int rows_k = metal::min(BK, kL - c0);

  // Stage K transposed (Ks[d][kk]): uint4 global loads (8 bf16 each),
  // scalar transposed scatter into smem.
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BDK / 8); i += NT) {
    const int r = i / (BDK / 8);
    const int c8 = i % (BDK / 8);
    uint4 w = uint4(0);
    if (r < rows_k) {
      w = *((const device uint4*)(Kp + (long)(c0 + r) * k_strides[2]) + c8);
    }
    thread T* e = (thread T*)&w;
    const int cb = c8 * 8;
    Ks[(cb + 0) * LDK + r] = e[0];
    Ks[(cb + 1) * LDK + r] = e[1];
    Ks[(cb + 2) * LDK + r] = e[2];
    Ks[(cb + 3) * LDK + r] = e[3];
    Ks[(cb + 4) * LDK + r] = e[4];
    Ks[(cb + 5) * LDK + r] = e[5];
    Ks[(cb + 6) * LDK + r] = e[6];
    Ks[(cb + 7) * LDK + r] = e[7];
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // S = Q @ K^T for this simdgroup's 8 query rows.
  float2 Sfrag[BK / 8];
  for (int i = 0; i < BK / 8; ++i) Sfrag[i] = float2(0.0f);
  for (int dd = 0; dd < BDK / 8; ++dd) {
    const float2 qf = Qfrag[dd];
    const int kbase = Ks_off + dd * 8 * LDK;
    const float2 kf0 = float2(float(Ks[kbase]), float(Ks[kbase + 1]));
    const float2 kf1 = float2(float(Ks[kbase + 8]), float(Ks[kbase + 9]));
    const float2 kf2 = float2(float(Ks[kbase + 16]), float(Ks[kbase + 17]));
    const float2 kf3 = float2(float(Ks[kbase + 24]), float(Ks[kbase + 25]));
    msv_mma(Sfrag[0], qf, kf0);
    msv_mma(Sfrag[1], qf, kf1);
    msv_mma(Sfrag[2], qf, kf2);
    msv_mma(Sfrag[3], qf, kf3);
  }
  Sfrag[0] *= scale_log2e;
  Sfrag[1] *= scale_log2e;
  Sfrag[2] *= scale_log2e;
  Sfrag[3] *= scale_log2e;

  // Masking: kL remainder + causal + sliding band, all element-wise.
  const bool tail_k = (rows_k < BK);
  const bool need_causal = (kb >= kb_min_causal);
  const bool need_band = (SW > 0) && (c0 <= q_hi - SW);
  if (QSA || tail_k || need_causal || need_band) {
    for (int kt = 0; kt < BK / 8; ++kt) {
      for (int jj = 0; jj < 2; ++jj) {
        const int col = c0 + kt * 8 + sn + jj;
        bool masked = false;
        if (tail_k && col >= kL) masked = true;
        if (need_causal && row_pos < col) masked = true;
        if (need_band && (row_pos - col) >= SW) masked = true;
        if (QSA && !masked && (!mask_row_ok || !mask[mask_row0 + (long)col * mask_strides[3]])) masked = true;
        if (masked) Sfrag[kt][jj] = -INFINITY;
      }
    }
  }

  // Stage V (same smem as K — K reads are done): uint4 on both sides.
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (BDV / 8); i += NT) {
    const int r = i / (BDV / 8);
    const int c8 = i % (BDV / 8);
    uint4 w = uint4(0);
    if (r < rows_k) {
      w = *((const device uint4*)(Vp + (long)(c0 + r) * v_strides[2]) + c8);
    }
    *((threadgroup uint4*)(Vs + r * LDV) + c8) = w;
  }

  // Online softmax (registers only, overlaps the V staging above).
  float new_max = max_score;
  new_max = metal::max(new_max, msv_row_max(Sfrag[0]));
  new_max = metal::max(new_max, msv_row_max(Sfrag[1]));
  new_max = metal::max(new_max, msv_row_max(Sfrag[2]));
  new_max = metal::max(new_max, msv_row_max(Sfrag[3]));
  Sfrag[0] = metal::exp2(Sfrag[0] - new_max);
  Sfrag[1] = metal::exp2(Sfrag[1] - new_max);
  Sfrag[2] = metal::exp2(Sfrag[2] - new_max);
  Sfrag[3] = metal::exp2(Sfrag[3] - new_max);
  const float factor = metal::exp2(max_score - new_max);
  max_score = new_max;
  const float rowsum = msv_row_sum(Sfrag[0]) + msv_row_sum(Sfrag[1])
      + msv_row_sum(Sfrag[2]) + msv_row_sum(Sfrag[3]);
  sum_score = sum_score * factor + rowsum;
  for (int i = 0; i < BDV / 8; ++i) Ofrag[i] *= factor;

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  // O += P @ V.
  for (int id = 0; id < BDV / 8; ++id) {
    const int vbase = Vs_off + id * 8;
    const float2 vf0 = float2(float(Vs[vbase]), float(Vs[vbase + 1]));
    const float2 vf1 = float2(float(Vs[vbase + 8 * LDV]),
                              float(Vs[vbase + 8 * LDV + 1]));
    const float2 vf2 = float2(float(Vs[vbase + 16 * LDV]),
                              float(Vs[vbase + 16 * LDV + 1]));
    const float2 vf3 = float2(float(Vs[vbase + 24 * LDV]),
                              float(Vs[vbase + 24 * LDV + 1]));
    msv_mma(Ofrag[id], Sfrag[0], vf0);
    msv_mma(Ofrag[id], Sfrag[1], vf1);
    msv_mma(Ofrag[id], Sfrag[2], vf2);
    msv_mma(Ofrag[id], Sfrag[3], vf3);
  }
}

// Final chunk: normalize + store bf16 (output freshly allocated,
// contiguous). Mid chunk: store the raw fp32 state for the next
// dispatch (m/l written by all 4 lane-threads of a row — same value).
const int local_row = tm + sm;
if (local_row < q_rows) {
  if (is_final) {
    const float inv = 1.0f / sum_score;
    device T* Optr = Op + (long)local_row * BDV + sn;
    for (int id = 0; id < BDV / 8; ++id) {
      Optr[id * 8] = T(Ofrag[id].x * inv);
      Optr[id * 8 + 1] = T(Ofrag[id].y * inv);
    }
  } else {
    const long crow = (((long)bb * Hq + hq) * (long)qL + (long)(tqx * BQ + local_row));
    m_out[crow] = max_score;
    l_out[crow] = sum_score;
    device float* Orow = o_out + crow * (long)BDV;
    for (int id = 0; id < BDV / 8; ++id) {
      Orow[id * 8 + sn] = Ofrag[id].x;
      Orow[id * 8 + sn + 1] = Ofrag[id].y;
    }
  }
}