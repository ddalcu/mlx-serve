// SPDX-License-Identifier: Apache-2.0
// relative_attention on the NAX tiles: MLX's attention_nax loop (Copyright (c) 2024-25 Apple Inc., MIT, via oMLX's
// port in kernels/attention/nax_attention.metal) with relative_attention's score: the relative bias by distance, a
// sink logit per query, the band and key_start. Header: nax_tiles.metal + nax_attention.metal. D is 128.
static_assert(D == 128, "the P x V barrier and the pair MMAs assume head dim 128");
constexpr short kU = 16;
constexpr int BQ = 64;
constexpr int BK = 32;
constexpr int TD = D / kU;
constexpr short TK = BK / kU;
constexpr float LOG2E = 1.44269504089f;
using otile_t = omlx_nax::NAXTile<float, 1, TD>;
using stile_t = omlx_nax::NAXTile<float, 1, TK>;
constexpr short kRowsPT = otile_t::kRowsPerThread;

const int L = sizes[0];
const int S = sizes[1];
const int start = key_start[0];
const uint3 tid = threadgroup_position_in_grid;
const int head = tid.y;
const int kv_head = head / (HQ / HKV);
const int q0 = tid.x * BQ;
const short tm = kU * simdgroup_index_in_threadgroup;
const int qld = int(q_strides[2]);
const int kld = int(k_strides[2]);
const int vld = int(v_strides[2]);
const device T* Q = q + head * q_strides[1] + int64_t(q0 + tm) * qld;
const device T* K = k + kv_head * k_strides[1];
const device T* V = v + kv_head * v_strides[1];

otile_t Otile;
Otile.clear();
const short2 coord = otile_t::NAXFrag_t::get_coord();
const short sm = coord.y;
const short sn = coord.x;

// This thread's two rows: each online softmax starts from its sink logit (rows past L from 0).
int rows[kRowsPT];
const device T* brow[kRowsPT];
metal::vec<float, kRowsPT> max_score;
metal::vec<float, kRowsPT> sum_score;
OMLX_NAX_UNROLL
for (short i = 0; i < kRowsPT; ++i) {
  rows[i] = q0 + tm + sm + i * otile_t::kFragRowsJump;
  const bool ok = rows[i] < L;
  max_score[i] = ok ? sinks[rows[i] * HQ + head] * LOG2E : 0.0f;
  sum_score[i] = 1.0f;
  brow[i] = rel + (int64_t(ok ? rows[i] : 0) * HQ + head) * R;
}

const int off = S - L;
const int first_diag = off + q0;
const int last_diag = off + min(q0 + BQ, L) - 1;
int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
first_key = max(first_key, start);
const int NK = (S + BK - 1) / BK;
const int NK_aligned = S / BK;
const int kb_lo = first_key / BK;
const int kb_hi = min(NK, last_diag / BK + 1);
const int lim_rows_q = min(L - (q0 + tm), int(kU));
const int lim_rows_k = S - NK_aligned * BK;
const int srow_lo = q0 + tm;
const int srow_hi = min(q0 + tm + kU, L) - 1;
const float scale2 = scale * LOG2E;
K += int64_t(kb_lo) * BK * kld;
V += int64_t(kb_lo) * BK * vld;

// BIA (BIAS_IN_ATTN): `rel` is the rows' clipped relative states [S, HQ, 16] (T) and `relp` the bias projection
// [16, RP] (T) with PADL zero columns before distance 0 and zeros past R. For each 16-row, 32-key block the bias at
// the block's 47 distances is one product on the tensor units, states [16 x 16] times relp's 64 columns from the
// block's least distance, in f32 (no bf16 rounding of the bias), staged in threadgroup memory for the scores.
constexpr int MLD = 49;
constexpr short kFR = stile_t::kFragThrRows;
threadgroup float Mtg[BIA ? 4 * kU * MLD : 1];
threadgroup float* msg = Mtg + (BIA ? int(simdgroup_index_in_threadgroup) * kU * MLD : 0);
omlx_nax::NAXTile<T, 1, 1> Rtile;
if (BIA) {
  const device T* rs = rel + (int64_t(q0 + tm) * HQ + head) * 16;
  if (!ALIGN_Q && lim_rows_q < kU) {
    Rtile.load_rows(rs, HQ * 16, short(max(lim_rows_q, 0)));
  } else {
    Rtile.load(rs, HQ * 16);
  }
}

for (int kb = kb_lo; kb < kb_hi; kb++) {
  const bool last_k = !ALIGN_K && kb == NK_aligned;
  stile_t Stile;
  Stile.clear();

  // S = Q K^T, the query fragments re-read for every block (held in registers they cost occupancy).
  OMLX_NAX_UNROLL
  for (short ik = 0; ik < TK; ik += 2) {
#pragma clang loop unroll_count(4)
    for (short id = 0; id < TD; id++) {
      omlx_nax::NAXTile<T, 1, 1> Qtile;
      omlx_nax::NAXTile<T, 2, 1> Ktile;
      if (!ALIGN_Q && lim_rows_q < kU) {
        Qtile.load_rows(Q + id * kU, qld, lim_rows_q);
      } else {
        Qtile.load(Q + id * kU, qld);
      }
      const int off_k = ik * kU * kld + id * kU;
      if (last_k) {
        Ktile.load_rows(K + off_k, kld, lim_rows_k - ik * kU);
      } else {
        Ktile.load(K + off_k, kld);
      }
      stile_t::NAXFrag_t::mma(
          Stile.frag_at(0, ik), Stile.frag_at(0, ik + 1), Qtile.frag_at(0, 0), metal::false_type{},
          Ktile.frag_at(0, 0), Ktile.frag_at(1, 0), metal::true_type{});
    }
  }

  // Scores in log2 units, plus the bias at distance d = diag - key below R; -inf outside the band. A block every row
  // of this simdgroup sees whole and at distances of R or more needs only the scale.
  const int c_lo = kb * BK;
  const int c_hi = c_lo + BK - 1;
  const bool plain = off + srow_lo - c_hi >= R && (WINDOW == 0 || off + srow_hi - c_lo < WINDOW) && c_lo >= start
      && c_hi < S && srow_lo + kU - 1 < L;
  if (plain) {
    OMLX_NAX_UNROLL
    for (short i = 0; i < stile_t::kElemsPerTile; i++) {
      Stile.elems()[i] *= scale2;
    }
  } else {
    // BIA: the block's least distance; past R every bias is zero (a last block's ragged keys or rows).
    const int dbase = off + srow_lo - c_lo - (BK - 1);
    const bool near = BIA && dbase < R;
    if (near) {
      omlx_nax::NAXTile<float, 1, 2> M0, M1;
      M0.clear();
      M1.clear();
      omlx_nax::NAXTile<T, 1, 2> P0, P1;
      P0.load(relp + PADL + dbase, RP);
      P1.load(relp + PADL + dbase + 32, RP);
      stile_t::NAXFrag_t::mma(M0.frag_at(0, 0), M0.frag_at(0, 1), Rtile.frag_at(0, 0), metal::false_type{},
          P0.frag_at(0, 0), P0.frag_at(0, 1), metal::false_type{});
      stile_t::NAXFrag_t::mma(M1.frag_at(0, 0), M1.frag_at(0, 1), Rtile.frag_at(0, 0), metal::false_type{},
          P1.frag_at(0, 0), P1.frag_at(0, 1), metal::false_type{});
      simdgroup_barrier(mem_flags::mem_threadgroup);
      OMLX_NAX_UNROLL
      for (short f = 0; f < 4; f++) {
        thread auto& mf = f < 2 ? M0.frag_at(0, f) : M1.frag_at(0, f - 2);
        OMLX_NAX_UNROLL
        for (short ii = 0; ii < kFR; ii++) {
          OMLX_NAX_UNROLL
          for (short jj = 0; jj < 4; jj++) {
            const int col = f * kU + sn + jj;
            if (col < MLD - 1) msg[(sm + ii * 8) * MLD + col] = mf[ii * 4 + jj];
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    OMLX_NAX_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      thread auto& fg = Stile.frag_at(0, ik);
      OMLX_NAX_UNROLL
      for (short ii = 0; ii < stile_t::kFragThrRows; ii++) {
        const int r = rows[ii];
        const int c0 = c_lo + ik * kU + sn;
        // The 4 columns' distances d0 - jj are 4 consecutive bias entries: one packed load of their bf16 bits when all
        // are below R.
        const int d0 = off + r - c0;
        float b[4];
        if (BIA) {
          OMLX_NAX_UNROLL
          for (short jj = 0; jj < 4; jj++) {
            b[jj] = near ? msg[(sm + ii * 8) * MLD + (sm + ii * 8) - (ik * kU + sn + jj) + (BK - 1)] : 0.0f;
          }
        } else if (d0 >= 3 && d0 < R) {
          const packed_ushort4 u = *(const device packed_ushort4*)((const device ushort*)brow[ii] + d0 - 3);
          OMLX_NAX_UNROLL
          for (short jj = 0; jj < 4; jj++) {
            b[jj] = as_type<float>(uint(u[3 - jj]) << 16);
          }
        } else {
          OMLX_NAX_UNROLL
          for (short jj = 0; jj < 4; jj++) {
            const int d = d0 - jj;
            b[jj] = d >= 0 && d < R ? float(brow[ii][d]) : 0.0f;
          }
        }
        OMLX_NAX_UNROLL
        for (short jj = 0; jj < stile_t::kFragThrCols; jj++) {
          const int c = c0 + jj;
          const int d = d0 - jj;
          const bool visible = r < L && c < S && d >= 0 && c >= start && (WINDOW == 0 || d < WINDOW);
          const short at = ii * stile_t::kFragThrCols + jj;
          fg[at] = visible ? fma(b[jj], LOG2E, fg[at] * scale2) : -INFINITY;
        }
      }
    }
  }

  metal::vec<float, kRowsPT> new_max;
  metal::vec<float, kRowsPT> factor;
  OMLX_NAX_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    new_max[i] = max_score[i];
  }
  Stile.template row_reduce<omlx_nax::MaxOp>(new_max);
  Stile.template row_bin_op<omlx_nax::ExpSubOp>(new_max);
  OMLX_NAX_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    factor[i] = fast::exp2(max_score[i] - new_max[i]);
    max_score[i] = new_max[i];
    sum_score[i] = sum_score[i] * factor[i];
  }
  Stile.template row_reduce<omlx_nax::SumOp>(sum_score);
  Otile.template row_bin_op<omlx_nax::MulOp>(factor);

  simdgroup_barrier(mem_flags::mem_none);

  // O += P V
  OMLX_NAX_UNROLL
  for (short id = 0; id < TD; id += 2) {
    if (id == 4) {
      threadgroup_barrier(mem_flags::mem_none);
    }
    OMLX_NAX_UNROLL
    for (short ik = 0; ik < TK; ik++) {
      omlx_nax::NAXTile<T, 1, 2> Vtile;
      const int off_v = ik * kU * vld + id * kU;
      if (last_k) {
        Vtile.load_rows(V + off_v, vld, lim_rows_k - ik * kU);
      } else {
        Vtile.load(V + off_v, vld);
      }
      otile_t::NAXFrag_t::mma(
          Otile.frag_at(0, id), Otile.frag_at(0, id + 1), Stile.frag_at(0, ik), metal::false_type{},
          Vtile.frag_at(0, 0), Vtile.frag_at(0, 1), metal::false_type{});
    }
  }
  K += BK * kld;
  V += BK * vld;
}

threadgroup_barrier(mem_flags::mem_none);
metal::vec<float, kRowsPT> rcp;
OMLX_NAX_UNROLL
for (short i = 0; i < kRowsPT; ++i) {
  rcp[i] = 1.0f / sum_score[i];
}
Otile.template row_bin_op<omlx_nax::MulOp>(rcp);
device T* O = out + (int64_t(q0 + tm) * HQ + head) * D;
if (!ALIGN_Q && lim_rows_q < kU) {
  if (lim_rows_q > 0) {
    Otile.store_rows(O, HQ * D, lim_rows_q);
  }
} else {
  Otile.store(O, HQ * D);
}
