//! Row-exact 4/6/8-bit matmul for 1..MAX_ROWS rows, ported from TensorFold's
//! `simd_qmm.py` (MIT, see NOTICE). Every row runs one fixed FMA chain per
//! 64-input group, the groups split over S chunks combined by a fixed tree; S
//! depends on the shape only. `mma` (2+ rows) runs the chain on 8x8 fp32
//! simdgroup MMAs, `scalar` (one row) with scalar FMAs; where a chip's MMA is
//! not that chain, `prepare` sends one-row calls of the shape through `mma`.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const MAX_ROWS = 16;
const RT_MAX = 2;
const SGS = 2;
const NR = 2;
const XB = 32;

const HEADER =
    \\#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
    \\inline float bf8(uint4 v, int e) {
    \\  const uint w = v[e / 2];
    \\  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
    \\}
    \\inline float sum8(uint4 v, float one) {
    \\  float t = bf8(v, 0);
    \\  for (int e = 1; e < 8; e++) t = fma(bf8(v, e), one, t);
    \\  return t;
    \\}
    \\// Value 8j + p of a 64-group's words (j = k-slot, p = pass): 4 and 8 bits
    \\// masked in place, the input pre-scaled by wpre(p) to cancel the shift;
    \\// 6 bits straddle words, so they are shifted down and wpre is 1.
    \\template <int BITS> inline float wpre(int p) {
    \\  return BITS == 6 ? 1.0f : as_type<float>(uint(127 - (BITS == 4 ? 4 * p : 8 * (p % 4))) << 23);
    \\}
    \\template <int BITS> inline float wfield(const thread uint* lw, int j, int p) {
    \\  const int bit = (8 * j + p) * BITS, off = bit % 32;
    \\  const uint w0 = lw[bit / 32];
    \\  if (BITS == 6) return float((off <= 26 ? w0 >> off : (w0 >> off) | (lw[bit / 32 + 1] << (32 - off))) & 63u);
    \\  return float(w0 & (((1u << BITS) - 1u) << off));
    \\}
;

const LOAD8 = "  #define LOAD8(r, j) ((((const device uint4*)X)[size_t(r) * (K / 8) + (j)]))\n";

const SCALAR =
    \\  constexpr int XP = 76;
    \\  threadgroup float xs[XB * XP];
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
    \\  const int c = int(lane) % S;
    \\  constexpr int SLOTS = 32 / S;
    \\  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
    \\                 + int(lane) / S;
    \\  constexpr int G = K / 64;
    \\  const float one = ONE[0];
    \\  constexpr int WQ = BITS / 2;
    \\  const device uint4* wr[NR];
    \\  const device bfloat* sr[NR];
    \\  const device bfloat* br[NR];
    \\  float acc[NR];
    \\  PRAGMA_UNROLL
    \\  for (int u = 0; u < NR; u++) {
    \\    const int nn = min(n0 + SLOTS * u, N - 1);
    \\    wr[u] = (const device uint4*)(W + size_t(nn) * (K * BITS / 32));
    \\    sr[u] = SC + size_t(nn) * G;
    \\    br[u] = BI + size_t(nn) * G;
    \\    acc[u] = 0.0f;
    \\  }
    \\  uint4 nw[NR][WQ];
    \\  PRAGMA_UNROLL
    \\  for (int u = 0; u < NR; u++)
    \\    PRAGMA_UNROLL
    \\    for (int q = 0; q < WQ; q++) nw[u][q] = c < G ? wr[u][WQ * c + q] : uint4(0);
    \\  for (int b0 = 0; b0 < G; b0 += XB) {
    \\    const int nbk = min(XB, G - b0);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (int idx = tid; idx < nbk * 8; idx += SGS * 32) {
    \\      const int gl = idx / 8, i = idx % 8;
    \\      const uint4 v = LOAD8(0, 8 * (b0 + gl) + i);
    \\      PRAGMA_UNROLL
    \\      for (int s = 0; s < 8; s++) xs[gl * XP + 8 * s + i] = bf8(v, s) * wpre<BITS>(s);
    \\      xs[gl * XP + 64 + i] = sum8(v, one);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (int g = b0 + c; g < b0 + nbk; g += S) {
    \\      uint wd[NR][2 * BITS];
    \\      PRAGMA_UNROLL
    \\      for (int u = 0; u < NR; u++)
    \\        PRAGMA_UNROLL
    \\        for (int q = 0; q < 2 * BITS; q++) wd[u][q] = nw[u][q / 4][q % 4];
    \\      if (g + S < G) {
    \\        PRAGMA_UNROLL
    \\        for (int u = 0; u < NR; u++)
    \\          PRAGMA_UNROLL
    \\          for (int q = 0; q < WQ; q++) nw[u][q] = wr[u][WQ * (g + S) + q];
    \\      }
    \\      const threadgroup float* xg = xs + (g - b0) * XP;
    \\      const float4 p0 = *(const threadgroup float4*)(xg + 64), p1 = *(const threadgroup float4*)(xg + 68);
    \\      const float xsum = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one,
    \\                             fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x)));
    \\      float P[NR];
    \\      PRAGMA_UNROLL
    \\      for (int u = 0; u < NR; u++) P[u] = 0.0f;
    \\      PRAGMA_UNROLL
    \\      for (int s = 0; s < 8; s++) {
    \\        const float4 lo = *(const threadgroup float4*)(xg + 8 * s), hi = *(const threadgroup float4*)(xg + 8 * s + 4);
    \\        const float xq[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    \\        PRAGMA_UNROLL
    \\        for (int u = 0; u < NR; u++)
    \\          PRAGMA_UNROLL
    \\          for (int i = 0; i < 8; i++) P[u] = fma(xq[i], wfield<BITS>(wd[u], i, s), P[u]);
    \\      }
    \\      PRAGMA_UNROLL
    \\      for (int u = 0; u < NR; u++) {
    \\        acc[u] = fma(float(sr[u][g]), P[u], acc[u]);
    \\        acc[u] = fma(float(br[u][g]), xsum, acc[u]);
    \\      }
    \\    }
    \\  }
    \\  PRAGMA_UNROLL
    \\  for (int u = 0; u < NR; u++) {
    \\    float v = acc[u];
    \\    PRAGMA_UNROLL
    \\    for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
    \\    const int n = n0 + SLOTS * u;
    \\    if (n < N && c == 0) OUT[n] = bfloat(v);
    \\  }
    \\
;

const SCALAR_ROWS =
    \\  constexpr int XP = 76;
    \\  threadgroup float xs[RR][XB * XP];
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
    \\  const int c = int(lane) % S;
    \\  constexpr int SLOTS = 32 / S;
    \\  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
    \\                 + int(lane) / S;
    \\  constexpr int G = K / 64;
    \\  const float one = ONE[0];
    \\  constexpr int WQ = BITS / 2;
    \\  const device uint4* wr[NR];
    \\  const device bfloat* sr[NR];
    \\  const device bfloat* br[NR];
    \\  float acc[NR][RR];
    \\  PRAGMA_UNROLL
    \\  for (int u = 0; u < NR; u++) {
    \\    const int nn = min(n0 + SLOTS * u, N - 1);
    \\    wr[u] = (const device uint4*)(W + size_t(nn) * (K * BITS / 32));
    \\    sr[u] = SC + size_t(nn) * G;
    \\    br[u] = BI + size_t(nn) * G;
    \\    for (int r = 0; r < RR; r++) acc[u][r] = 0.0f;
    \\  }
    \\  uint4 nw[NR][WQ];
    \\  PRAGMA_UNROLL
    \\  for (int u = 0; u < NR; u++)
    \\    PRAGMA_UNROLL
    \\    for (int q = 0; q < WQ; q++) nw[u][q] = c < G ? wr[u][WQ * c + q] : uint4(0);
    \\  for (int b0 = 0; b0 < G; b0 += XB) {
    \\    const int nbk = min(XB, G - b0);
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (int idx = tid; idx < RR * nbk * 8; idx += SGS * 32) {
    \\      const int r = idx / (nbk * 8), gl = (idx / 8) % nbk, i = idx % 8;
    \\      const uint4 v = LOAD8(r, 8 * (b0 + gl) + i);
    \\      PRAGMA_UNROLL
    \\      for (int s = 0; s < 8; s++) xs[r][gl * XP + 8 * s + i] = bf8(v, s) * wpre<BITS>(s);
    \\      xs[r][gl * XP + 64 + i] = sum8(v, one);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (int g = b0 + c; g < b0 + nbk; g += S) {
    \\      uint wd[NR][2 * BITS];
    \\      PRAGMA_UNROLL
    \\      for (int u = 0; u < NR; u++)
    \\        PRAGMA_UNROLL
    \\        for (int q = 0; q < 2 * BITS; q++) wd[u][q] = nw[u][q / 4][q % 4];
    \\      if (g + S < G) {
    \\        PRAGMA_UNROLL
    \\        for (int u = 0; u < NR; u++)
    \\          PRAGMA_UNROLL
    \\          for (int q = 0; q < WQ; q++) nw[u][q] = wr[u][WQ * (g + S) + q];
    \\      }
    \\      // The chain per row is the one-row kernel's (s, then i); the masked
    \\      // weights are built once for every row.
    \\      float P[NR][RR];
    \\      PRAGMA_UNROLL
    \\      for (int u = 0; u < NR; u++)
    \\        for (int r = 0; r < RR; r++) P[u][r] = 0.0f;
    \\      PRAGMA_UNROLL
    \\      for (int s = 0; s < 8; s++) {
    \\        float wf[NR][8];
    \\        PRAGMA_UNROLL
    \\        for (int u = 0; u < NR; u++)
    \\          PRAGMA_UNROLL
    \\          for (int i = 0; i < 8; i++) wf[u][i] = wfield<BITS>(wd[u], i, s);
    \\        PRAGMA_UNROLL
    \\        for (int r = 0; r < RR; r++) {
    \\          const threadgroup float* xg = xs[r] + (g - b0) * XP;
    \\          const float4 lo = *(const threadgroup float4*)(xg + 8 * s), hi = *(const threadgroup float4*)(xg + 8 * s + 4);
    \\          const float xq[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    \\          PRAGMA_UNROLL
    \\          for (int u = 0; u < NR; u++)
    \\            PRAGMA_UNROLL
    \\            for (int i = 0; i < 8; i++) P[u][r] = fma(xq[i], wf[u][i], P[u][r]);
    \\        }
    \\      }
    \\      PRAGMA_UNROLL
    \\      for (int r = 0; r < RR; r++) {
    \\        const threadgroup float* xg = xs[r] + (g - b0) * XP;
    \\        const float4 p0 = *(const threadgroup float4*)(xg + 64), p1 = *(const threadgroup float4*)(xg + 68);
    \\        const float xsum = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one,
    \\                               fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x)));
    \\        PRAGMA_UNROLL
    \\        for (int u = 0; u < NR; u++) {
    \\          acc[u][r] = fma(float(sr[u][g]), P[u][r], acc[u][r]);
    \\          acc[u][r] = fma(float(br[u][g]), xsum, acc[u][r]);
    \\        }
    \\      }
    \\    }
    \\  }
    \\  PRAGMA_UNROLL
    \\  for (int u = 0; u < NR; u++)
    \\    for (int r = 0; r < RR; r++) {
    \\      float v = acc[u][r];
    \\      PRAGMA_UNROLL
    \\      for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
    \\      const int n = n0 + SLOTS * u;
    \\      if (n < N && c == 0) OUT[size_t(r) * N + n] = bfloat(v);
    \\    }
    \\
;

// `mma` is split where the fragment variant differs: how R is read, and
// where each group's input values and sums come from.
const MMA_HEAD =
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const int sg = int(simdgroup_index_in_threadgroup);
    \\  const int qid = int(lane) / 4;
    \\  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
    \\  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
;
const MMA_MID =
    \\  constexpr int G = K / 64;
    \\  const float one = ONE[0];
    \\  const int nb = int(threadgroup_position_in_grid.x) * (8 * NT);
    \\  const int rb = int(threadgroup_position_in_grid.y) * (8 * RT);
    \\  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
    \\  int wrow[NT];
    \\  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
    \\  int xr0[RT], xr1[RT];
    \\  for (int rt = 0; rt < RT; rt++) { xr0[rt] = min(rb + 8 * rt + fn, R - 1); xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1); }
    \\  float acc[RT][NT][2];
    \\  for (int c = sg; c < S; c += SG) {
    \\  for (int rt = 0; rt < RT; rt++)
    \\    for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
    \\  for (int g = c; g < G; g += S) {
    \\    uint wv[NT][BITS / 2];
    \\    PRAGMA_UNROLL
    \\    for (int t = 0; t < NT; t++) {
    \\      const device uint* wp = (const device uint*)W + size_t(wrow[t]) * (K * BITS / 32) + 2 * BITS * g + (BITS / 2) * (fn / 2);
    \\      if (BITS == 4) { const uint2 v = *(const device uint2*)wp; wv[t][0] = v.x; wv[t][1] = v.y; }
    \\      else if (BITS == 8) { const uint4 v = *(const device uint4*)wp; wv[t][0] = v.x; wv[t][1] = v.y; wv[t][2] = v.z; wv[t][3] = v.w; }
    \\      else { wv[t][0] = wp[0]; wv[t][1] = wp[1]; wv[t][2] = wp[2]; }
    \\    }
;
const MMA_BM =
    \\    simdgroup_matrix<float, 8, 8> P[RT][NT];
    \\    PRAGMA_UNROLL
    \\    for (int rt = 0; rt < RT; rt++)
    \\      for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
    \\    PRAGMA_UNROLL
    \\    for (int s = 0; s < 8; s++) {
    \\      const float ps = wpre<BITS>(s);
    \\      simdgroup_matrix<float, 8, 8> bm[RT];
    \\      PRAGMA_UNROLL
    \\      for (int rt = 0; rt < RT; rt++) {
;
const MMA_TAIL =
    \\      }
    \\      PRAGMA_UNROLL
    \\      for (int t = 0; t < NT; t++) {
    \\        simdgroup_matrix<float, 8, 8> am;
    \\        am.thread_elements()[0] = wfield<BITS>(wv[t], 0, s);
    \\        am.thread_elements()[1] = wfield<BITS>(wv[t], 1, s);
    \\        PRAGMA_UNROLL
    \\        for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
    \\      }
    \\    }
    \\    PRAGMA_UNROLL
    \\    for (int t = 0; t < NT; t++) {
    \\      const float sc = float(SC[size_t(wrow[t]) * G + g]);
    \\      const float bi = float(BI[size_t(wrow[t]) * G + g]);
    \\      PRAGMA_UNROLL
    \\      for (int rt = 0; rt < RT; rt++) {
    \\        acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
    \\        acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
    \\      }
    \\    }
    \\  }
    \\  if (S == 1) {
    \\    for (int rt = 0; rt < RT; rt++)
    \\      for (int t = 0; t < NT; t++)
    \\        for (int e = 0; e < 2; e++) {
    \\          const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
    \\          if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[rt][t][e]);
    \\        }
    \\    return;
    \\  }
    \\  for (int rt = 0; rt < RT; rt++)
    \\    for (int t = 0; t < NT; t++)
    \\      for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SG * 32) {
    \\    float v[S];
    \\    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    \\    for (int w = 1; w < S; w *= 2)
    \\      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    \\    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    \\    const int lq = l / 4;
    \\    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    \\    if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(v[0]);
    \\  }
    \\
;
const MMA = MMA_HEAD ++ "\n" ++
    \\  const int R = X_shape[0];
++ "\n" ++ MMA_MID ++ "\n" ++
    \\    uint4 xa[RT], xb[RT];
    \\    float xs0[RT], xs1[RT];
    \\    PRAGMA_UNROLL
    \\    for (int rt = 0; rt < RT; rt++) {
    \\      xa[rt] = LOAD8(xr0[rt], 8 * g + fm);
    \\      xb[rt] = LOAD8(xr1[rt], 8 * g + fm);
    \\      float v = sum8(xa[rt], one), u = sum8(xb[rt], one);
    \\      v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
    \\      v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
    \\      v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
    \\      xs0[rt] = v; xs1[rt] = u;
    \\    }
++ "\n" ++ MMA_BM ++ "\n" ++
    \\        bm[rt].thread_elements()[0] = bf8(xa[rt], s) * ps;
    \\        bm[rt].thread_elements()[1] = bf8(xb[rt], s) * ps;
++ "\n" ++ MMA_TAIL;

/// `mma` over pre-scaled input fragments (`PREP`): the same values from the
/// same float ops, read instead of rebuilt by every threadgroup.
const MMA_FRAG = MMA_HEAD ++ "\n" ++
    \\  const int R = XS_shape[0];
    \\  const int T8 = (R + 7) / 8;
    \\  const device float2* XF2 = (const device float2*)XF;
++ "\n" ++ MMA_MID ++ "\n" ++
    \\    float xs0[RT], xs1[RT];
    \\    PRAGMA_UNROLL
    \\    for (int rt = 0; rt < RT; rt++) { xs0[rt] = XS[size_t(xr0[rt]) * G + g]; xs1[rt] = XS[size_t(xr1[rt]) * G + g]; }
++ "\n" ++ MMA_BM ++ "\n" ++
    \\        const float2 f = XF2[(size_t(min(rb / 8 + rt, T8 - 1)) * G + g) * 256 + 32 * s + lane];
    \\        bm[rt].thread_elements()[0] = f.x;
    \\        bm[rt].thread_elements()[1] = f.y;
++ "\n" ++ MMA_TAIL;

/// One simdgroup a (row tile, group): the `mma` fragments of x pre-scaled by
/// wpre (XF[tile][g][s][lane]; rows past R copy row R - 1) and each row's
/// group sum by the kernels' tree (XS[r][g]).
const PREP =
    \\  const uint lane = thread_index_in_simdgroup;
    \\  const int unit = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
    \\  const int R = X_shape[0];
    \\  constexpr int G = K / 64;
    \\  const int T8 = (R + 7) / 8;
    \\  if (unit >= T8 * G) return;
    \\  const int tile = unit / G, g = unit % G;
    \\  const int qid = int(lane) / 4;
    \\  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
    \\  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
    \\  const float one = ONE[0];
    \\  const int r0 = min(8 * tile + fn, R - 1), r1 = min(8 * tile + fn + 1, R - 1);
    \\  const uint4 xa = LOAD8(r0, 8 * g + fm), xb = LOAD8(r1, 8 * g + fm);
    \\  device float2* xf = (device float2*)XF + (size_t(tile) * G + g) * 256 + lane;
    \\  PRAGMA_UNROLL
    \\  for (int s = 0; s < 8; s++) xf[32 * s] = float2(bf8(xa, s) * wpre<BITS>(s), bf8(xb, s) * wpre<BITS>(s));
    \\  float v = sum8(xa, one), u = sum8(xb, one);
    \\  v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
    \\  v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
    \\  v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
    \\  if (fm == 0) {
    \\    if (8 * tile + fn < R) XS[size_t(8 * tile + fn) * G + g] = v;
    \\    if (8 * tile + fn + 1 < R) XS[size_t(8 * tile + fn + 1) * G + g] = u;
    \\  }
    \\
;

/// `frag` is `mma` over inputs `prep` wrote once; only 2..FRAGMENT_ROWS rows take it.
const Kind = enum { scalar, rows, mma, frag, prep };
pub const FRAGMENT_ROWS = 8;
/// Widest window `rows` serves: past it `mma` is cheaper. At 6 and 8 bits
/// `mma` is cheaper from two rows.
pub const SCALAR_ROWS_MAX = 3;

/// Chunks the K groups split into: a function of the shape only (it sets the bits).
/// Capped at 16: `mma` runs S simdgroups per threadgroup, and a 1024-thread
/// group exceeds what older GPUs grant this register-heavy kernel.
fn splits(n: c_int) c_int {
    return if (n <= 2048) 16 else 8;
}

fn rowTiles(rows: c_int) c_int {
    return @min(RT_MAX, @divTrunc(rows + 7, 8));
}

/// Tiles of 8 outputs a simdgroup in `mma` (speed only), at most 16 KB of
/// threadgroup memory for the split reduction.
fn tiles(n: c_int, rows: c_int, s: c_int) c_int {
    var nt: c_int = if (@rem(n, 32) == 0) 4 else if (@rem(n, 16) == 0) 2 else 1;
    while (nt > 1 and s * @divTrunc(rows + 7, 8) * nt * 64 * 4 > 16384) nt = @divTrunc(nt, 2);
    return nt;
}

/// Staging for `rows` (every row, at least S groups a block) fits 32 KB.
fn rowsFit(rows: c_int, n: c_int) bool {
    return rows * splits(n) * 76 * 4 <= 32 * 1024;
}

const PlanKey = struct { kind: Kind, rows: c_int, n: c_int, k: c_int, bits: c_int };
const Plan = struct { kernel: mlx.mlx_fast_metal_kernel, config: mlx.mlx_fast_metal_kernel_config };
var plans: std.AutoHashMapUnmanaged(PlanKey, Plan) = .{};
/// Kernels by (kind, baked constants): a new row count reuses the compiled one.
const KernelKey = struct { kind: Kind, k: c_int, n: c_int, bits: c_int, s: c_int, a: c_int, b: c_int, rr: c_int = 1, xb: c_int = XB, sg: c_int = 1 };
var kernels: std.AutoHashMapUnmanaged(KernelKey, mlx.mlx_fast_metal_kernel) = .{};
/// Simdgroups per `mma` threadgroup by (n, k, bits, rt) where fewer than S fit:
/// Metal caps a register-heavy kernel's threadgroup (M1/M2: 448 for this one),
/// and the S chunks walked by fewer simdgroups sum in the same order.
var mma_sg: std.AutoHashMapUnmanaged([4]c_int, c_int) = .{};
/// Shapes (n, k, bits) whose one-row calls take `mma`: there the scalar chain's bits differ.
var mma_one_row: std.AutoHashMapUnmanaged([3]c_int, void) = .{};
var prepared: std.AutoHashMapUnmanaged([3]c_int, void) = .{};
/// Shapes whose probe failed on this GPU, with MLX's reason: `qmm` declines them.
var declined: std.AutoHashMapUnmanaged([3]c_int, []const u8) = .{};
var one_arr: mlx.mlx_array = .{ .ctx = null };

fn kernelFor(key: KernelKey) !mlx.mlx_fast_metal_kernel {
    if (kernels.get(key)) |k| return k;
    const a = std.heap.c_allocator;
    const consts = if (key.kind == .prep)
        try std.fmt.allocPrint(a, "  constexpr int K = {d};\n  constexpr int BITS = {d};\n", .{ key.k, key.bits })
    else if (key.kind != .mma and key.kind != .frag)
        try std.fmt.allocPrint(a, "  constexpr int K = {d};\n  constexpr int N = {d};\n  constexpr int BITS = {d};\n  constexpr int S = {d};\n  constexpr int SGS = {d};\n  constexpr int NR = {d};\n  constexpr int XB = {d};\n  constexpr int RR = {d};\n", .{ key.k, key.n, key.bits, key.s, key.a, key.b, key.xb, key.rr })
    else
        try std.fmt.allocPrint(a, "  constexpr int K = {d};\n  constexpr int N = {d};\n  constexpr int BITS = {d};\n  constexpr int S = {d};\n  constexpr int NT = {d};\n  constexpr int RT = {d};\n  constexpr int SG = {d};\n", .{ key.k, key.n, key.bits, key.s, key.a, key.b, key.sg });
    defer a.free(consts);
    const body = switch (key.kind) {
        .scalar => SCALAR,
        .rows => SCALAR_ROWS,
        .mma => MMA,
        .frag => MMA_FRAG,
        .prep => PREP,
    };
    const source = try std.mem.concatWithSentinel(a, u8, &.{ consts, LOAD8, body, "  #undef LOAD8\n" }, 0);
    defer a.free(source);
    const name = try std.fmt.allocPrintSentinel(a, "msv_simd_qmm_{s}_k{d}_n{d}_b{d}_s{d}_{d}_{d}_r{d}_x{d}_g{d}", .{ @tagName(key.kind), key.k, key.n, key.bits, key.s, key.a, key.b, key.rr, key.xb, key.sg }, 0);
    defer a.free(name);
    const in_names: []const [*:0]const u8 = switch (key.kind) {
        .frag => &.{ "XF", "XS", "W", "SC", "BI", "ONE" },
        .prep => &.{ "X", "ONE" },
        else => &.{ "X", "W", "SC", "BI", "ONE" },
    };
    const out_names: []const [*:0]const u8 = if (key.kind == .prep) &.{ "XF", "XS" } else &.{"OUT"};
    const in_vec = mlx.mlx_vector_string_new_data(in_names.ptr, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(out_names.ptr, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name.ptr, in_vec, out_vec, source.ptr, HEADER, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    try kernels.put(a, key, k);
    return k;
}

fn planFor(kind: Kind, rows: c_int, n: c_int, k: c_int, bits: c_int) !Plan {
    const pk = PlanKey{ .kind = kind, .rows = rows, .n = n, .k = k, .bits = bits };
    if (plans.get(pk)) |p| return p;
    const s = splits(n);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    var kkey: KernelKey = undefined;
    if (kind == .prep) {
        const g = @divExact(k, 64);
        const units = @divTrunc(rows + 7, 8) * g;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{units * 512}, 1, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ rows, g }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(units + 3, 4) * 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
        kkey = .{ .kind = kind, .k = k, .n = 0, .bits = bits, .s = 0, .a = 0, .b = 0 };
        const p = Plan{ .kernel = try kernelFor(kkey), .config = config };
        try plans.put(std.heap.c_allocator, pk, p);
        return p;
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ rows, n }, 2, .bfloat16));
    if (kind != .mma and kind != .frag) {
        const nr: c_int = if (n > 2048) NR else 1;
        const sgs: c_int = if (n > 2048) SGS else 8;
        const per = sgs * @divExact(32, s) * nr;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(n + per - 1, per) * sgs * 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, sgs * 32, 1, 1));
        // `rows` stages every row: the staging block shrinks to fit 32 KB but
        // stays a multiple of S, so each chunk still visits its groups in order.
        const rr: c_int = if (kind == .rows) rows else 1;
        const xb: c_int = if (kind == .rows) @max(s, @divTrunc(XB, rr) & ~(s - 1)) else XB;
        kkey = .{ .kind = kind, .k = k, .n = n, .bits = bits, .s = s, .a = sgs, .b = nr, .rr = rr, .xb = xb };
    } else {
        const rt = rowTiles(rows);
        const nt = tiles(n, rt * 8, s);
        const sg = mma_sg.get(.{ n, k, bits, rt }) orelse s;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divTrunc(n + 8 * nt - 1, 8 * nt) * sg * 32, @divTrunc(rows + 8 * rt - 1, 8 * rt), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, sg * 32, 1, 1));
        kkey = .{ .kind = kind, .k = k, .n = n, .bits = bits, .s = s, .a = nt, .b = rt, .sg = sg };
    }
    const p = Plan{ .kernel = try kernelFor(kkey), .config = config };
    try plans.put(std.heap.c_allocator, pk, p);
    return p;
}

/// A 4, 6 or 8-bit affine bf16 matrix [N, K * bits / 32] in groups of 64
/// with K % 64 == 0 and N % 8 == 0.
pub fn fits(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32) bool {
    if ((bits != 4 and bits != 6 and bits != 8) or group_size != 64 or bi.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    const ws = mlx.getShape(w);
    return ws.len == 2 and @rem(ws[1] * 32, 64 * @as(c_int, @intCast(bits))) == 0 and @rem(ws[0], 8) == 0;
}

fn launch(kind: Kind, x2: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, rows: c_int, n: c_int, k: c_int, bits: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (one_arr.ctx == null) {
        const one: f32 = 1.0;
        one_arr = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .float32);
    }
    const pk = PlanKey{ .kind = kind, .rows = rows, .n = n, .k = k, .bits = bits };
    var frags: [2]mlx.mlx_array = .{ .{ .ctx = null }, .{ .ctx = null } };
    defer for (frags) |f| if (f.ctx != null) {
        _ = mlx.mlx_array_free(f);
    };
    if (kind == .frag) {
        const pp = try planFor(.prep, rows, 0, k, bits);
        const pin = [_]mlx.mlx_array{ x2, one_arr };
        const pin_vec = mlx.mlx_vector_array_new_data(&pin, pin.len);
        defer _ = mlx.mlx_vector_array_free(pin_vec);
        var pouts = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(pouts);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&pouts, pp.kernel, pin_vec, pp.config, s));
        for (&frags, 0..) |*f, i| {
            f.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(f, pouts, i));
        }
    }
    while (true) {
        const fresh = (kind == .mma or kind == .frag) and !plans.contains(pk);
        const p = try planFor(kind, rows, n, k, bits);
        const direct = [_]mlx.mlx_array{ x2, w, sc, bi, one_arr };
        const fragged = [_]mlx.mlx_array{ frags[0], frags[1], w, sc, bi, one_arr };
        const inputs: []const mlx.mlx_array = if (kind == .frag) &fragged else &direct;
        const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var outs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, p.kernel, in_vec, p.config, s));
        var y = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
        if (!fresh) return y;
        // A new mma plan runs once here: where Metal caps this kernel's threadgroup
        // below S simdgroups, halve them and rebuild.
        mlx.check(mlx.mlx_array_eval(y)) catch |e| {
            _ = mlx.mlx_array_free(y);
            const sgk = [4]c_int{ n, k, bits, rowTiles(rows) };
            const cur = mma_sg.get(sgk) orelse splits(n);
            if (e != error.MlxError or cur <= 1 or !mlx.takeErrorIf("Thread group size")) return e;
            try mma_sg.put(std.heap.c_allocator, sgk, @divTrunc(cur, 2));
            forgetPlan(pk);
            log.info("[simd_qmm] mma n={d} k={d} bits={d}: {d} simdgroups per threadgroup on this GPU\n", .{ n, k, bits, @divTrunc(cur, 2) });
            continue;
        };
        return y;
    }
}

fn forgetPlan(pk: PlanKey) void {
    if (plans.fetchRemove(pk)) |kv| _ = mlx.mlx_fast_metal_kernel_config_free(kv.value.config);
}

/// `x [..., K] @ w.T` for 1..MAX_ROWS rows, or null outside the kernels.
/// A row's bits do not depend on how many rows ride with it.
pub fn qmm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    return qmmKind(null, x, w, sc, bi, bits, group_size, s);
}

fn qmmKind(force: ?Kind, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) anyerror!?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !fits(w, sc, bi, bits, group_size) or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    if (xs.len == 0 or xs.len > 8) return null;
    const ws = mlx.getShape(w);
    const k = xs[xs.len - 1];
    const b: c_int = @intCast(bits);
    if (k * b != ws[1] * 32) return null;
    var rows: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= d;
    if (rows < 1 or rows > MAX_ROWS) return null;
    const n = ws[0];
    const shape = [3]c_int{ n, k, b };
    if (declined.contains(shape)) return null;
    if (force == null and !prepared.contains(shape)) prepare(w, sc, bi, bits, s) catch |e| {
        if (e != error.MlxError) return e;
        var buf: [256]u8 = undefined;
        const msg = mlx.takeError(&buf) orelse "";
        const a = std.heap.c_allocator;
        try declined.put(a, shape, try a.dupe(u8, msg));
        log.warn("[simd_qmm] n={d} k={d} bits={d} declined on this GPU: {s}\n", .{ n, k, b, msg });
        return null;
    };
    const rows_max: c_int = if (bits == 4) SCALAR_ROWS_MAX else 1;
    const kind: Kind = force orelse if (mma_one_row.contains(shape) or rows > rows_max or !rowsFit(rows, n))
        (if (rows >= 2 and rows <= FRAGMENT_ROWS) Kind.frag else Kind.mma)
    else if (rows == 1) .scalar else .rows;
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, k }, 2, s));
    const y = try launch(kind, x2, w, sc, bi, rows, n, k, b, s);
    defer _ = mlx.mlx_array_free(y);
    var out_shape: [8]c_int = undefined;
    @memcpy(out_shape[0..xs.len], xs);
    out_shape[xs.len - 1] = n;
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r, y, &out_shape, xs.len, s));
    return r;
}

/// Once per shape: whether one-row `scalar` calls give `mma`'s rows bit for
/// bit on this GPU; where they do not, one-row calls of the shape take `mma`.
fn prepare(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, s: mlx.mlx_stream) anyerror!void {
    const ws = mlx.getShape(w);
    const n = ws[0];
    const k = @divExact(ws[1] * 32, @as(c_int, @intCast(bits)));
    const shape = [3]c_int{ n, k, @intCast(bits) };
    try prepared.put(std.heap.c_allocator, shape, {});
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 0x5eed));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_random_normal(&xf, &[_]c_int{ 8, k }, 2, .float32, 0.0, 0.5, key, s));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
    const full = (try qmmKind(.mma, x, w, sc, bi, bits, 64, s)).?;
    defer _ = mlx.mlx_array_free(full);
    var same_all = true;
    var r: c_int = 0;
    while (r < 8) : (r += 1) {
        var xr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xr);
        try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, k }, 2, &[_]c_int{ 1, 1 }, 2, s));
        const one = (try qmmKind(.scalar, xr, w, sc, bi, bits, 64, s)).?;
        defer _ = mlx.mlx_array_free(one);
        var fr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(fr);
        try mlx.check(mlx.mlx_slice(&fr, full, &[_]c_int{ r, 0 }, 2, &[_]c_int{ r + 1, n }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, one, fr, false, s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        if (!same) same_all = false;
    }
    if (!same_all) try mma_one_row.put(std.heap.c_allocator, shape, {});
}

// ── tests ──

const testing = std.testing;

fn randBf16(shape: []const c_int, scale: f32, seed: u64, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0.0, scale, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

fn rowsOf(x: mlx.mlx_array, lo: c_int, hi: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const k = mlx.getShape(x)[1];
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &[_]c_int{ lo, 0 }, 2, &[_]c_int{ hi, k }, 2, &[_]c_int{ 1, 1 }, 2, s));
    return out;
}

fn expectSame(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(same);
}

test "simd_qmm: mma over half the simdgroups per threadgroup gives the same bits" {
    const s = mlx.gpuStream();
    const wf = try randBf16(&.{ 136, 256 }, 0.02, 11, s);
    defer _ = mlx.mlx_array_free(wf);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
    const x = try randBf16(&.{ 12, 256 }, 1.0, 12, s);
    defer _ = mlx.mlx_array_free(x);
    const full = (try qmmKind(.mma, x, w, sc, bi, 4, 64, s)).?;
    defer _ = mlx.mlx_array_free(full);
    try mlx.check(mlx.mlx_array_eval(full));
    const rt = rowTiles(12);
    try mma_sg.put(std.heap.c_allocator, .{ 136, 256, 4, rt }, @divTrunc(splits(136), 2));
    defer _ = mma_sg.remove(.{ 136, 256, 4, rt });
    forgetPlan(.{ .kind = .mma, .rows = 12, .n = 136, .k = 256, .bits = 4 });
    const half = (try qmmKind(.mma, x, w, sc, bi, 4, 64, s)).?;
    defer _ = mlx.mlx_array_free(half);
    try expectSame(half, full, s);
    forgetPlan(.{ .kind = .mma, .rows = 12, .n = 136, .k = 256, .bits = 4 });
}

test "simd_qmm: a shape whose probe fails on this GPU declines by name and leaves no latch" {
    const s = mlx.gpuStream();
    const wf = try randBf16(&.{ 72, 128 }, 0.02, 3, s);
    defer _ = mlx.mlx_array_free(wf);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
    const x = try randBf16(&.{ 2, 128 }, 1.0, 4, s);
    defer _ = mlx.mlx_array_free(x);
    // The 4th checked op after this line sits inside the probe.
    mlx.armLatchingFaultForTest(4);
    const got = try qmm(x, w, sc, bi, 4, 64, s);
    const fired = mlx.latchingFaultFiredForTest();
    mlx.armLatchingFaultForTest(0);
    try testing.expect(fired);
    try testing.expect(got == null);
    try testing.expect(!mlx.errorPending());
    try testing.expect(declined.get(.{ 72, 128, 4 }) != null);
    // Declined shapes never re-probe.
    const ops = mlx.op_count.load(.monotonic);
    try testing.expect((try qmm(x, w, sc, bi, 4, 64, s)) == null);
    try testing.expectEqual(ops, mlx.op_count.load(.monotonic));
}

test "simd_qmm: every row of an R-row call equals its one-row call bit for bit, and matches f32 truth at 4, 6 and 8 bits" {
    const s = mlx.gpuStream();
    // A kernel failure is this test's: name it and drop its latch before the next test.
    errdefer {
        var buf: [512]u8 = undefined;
        if (mlx.takeError(&buf)) |msg| std.debug.print("[simd_qmm] mlx: {s}\n", .{msg});
    }
    // Qwen3.8-27B's shapes, cut down: both split counts (16 / 8) and odd tile counts.
    const Shape = struct { n: c_int, k: c_int };
    for ([_]u32{ 4, 6, 8 }) |bits| for ([_]Shape{ .{ .n = 64, .k = 512 }, .{ .n = 1024, .k = 5120 }, .{ .n = 4104, .k = 1024 }, .{ .n = 2056, .k = 640 } }, 0..) |sh, si| {
        const wf = try randBf16(&.{ sh.n, sh.k }, 0.02, 100 + si, s);
        defer _ = mlx.mlx_array_free(wf);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        const x = try randBf16(&.{ MAX_ROWS, sh.k }, 1.0, 7 + si, s);
        defer _ = mlx.mlx_array_free(x);
        const all = (try qmm(x, w, sc, bi, bits, 64, s)) orelse {
            std.debug.print("[simd_qmm] n={d} k={d} bits={d} declined: {s}\n", .{ sh.n, sh.k, bits, declined.get(.{ sh.n, sh.k, @intCast(bits) }) orelse "not a fit" });
            return error.SkipZigTest;
        };
        defer _ = mlx.mlx_array_free(all);
        var r: c_int = 0;
        while (r < MAX_ROWS) : (r += 1) {
            const xr = try rowsOf(x, r, r + 1, s);
            defer _ = mlx.mlx_array_free(xr);
            const one = (try qmm(xr, w, sc, bi, bits, 64, s)).?;
            defer _ = mlx.mlx_array_free(one);
            const want = try rowsOf(all, r, r + 1, s);
            defer _ = mlx.mlx_array_free(want);
            try expectSame(one, want, s);
        }
        for ([_][2]c_int{ .{ 2, 5 }, .{ 0, 8 }, .{ 3, 12 }, .{ 6, 8 }, .{ 9, 13 } }) |win| {
            const xw = try rowsOf(x, win[0], win[1], s);
            defer _ = mlx.mlx_array_free(xw);
            const got = (try qmm(xw, w, sc, bi, bits, 64, s)).?;
            defer _ = mlx.mlx_array_free(got);
            const want = try rowsOf(all, win[0], win[1], s);
            defer _ = mlx.mlx_array_free(want);
            try expectSame(got, want, s);
        }
        // f32 truth: x @ dequant(w).T
        var wd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wd);
        try mlx.check(mlx.mlx_dequantize(&wd, w, sc, bi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, wd, s));
        var xf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xf);
        try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_matmul(&truth, xf, wt, s));
        var gf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gf);
        try mlx.check(mlx.mlx_astype(&gf, all, .float32, s));
        var d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d);
        try mlx.check(mlx.mlx_subtract(&d, gf, truth, s));
        var d2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d2);
        try mlx.check(mlx.mlx_multiply(&d2, d, d, s));
        var t2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t2);
        try mlx.check(mlx.mlx_multiply(&t2, truth, truth, s));
        var sd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sd);
        try mlx.check(mlx.mlx_sum(&sd, d2, false, s));
        var st = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(st);
        try mlx.check(mlx.mlx_sum(&st, t2, false, s));
        var a: f32 = 0;
        var b: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&a, sd));
        try mlx.check(mlx.mlx_array_item_float32(&b, st));
        try testing.expect(@sqrt(a / b) < 1e-2);
    };
}

test "simd_qmm: reading pre-scaled input fragments gives the direct mma's bits at 2..8 rows and 4, 6 and 8 bits" {
    const s = mlx.gpuStream();
    errdefer {
        var buf: [512]u8 = undefined;
        if (mlx.takeError(&buf)) |msg| std.debug.print("[simd_qmm] mlx: {s}\n", .{msg});
    }
    for ([_]u32{ 4, 6, 8 }) |bits| for ([_][2]c_int{ .{ 1024, 5120 }, .{ 4104, 1024 } }, 0..) |sh, si| {
        const wf = try randBf16(&.{ sh[0], sh[1] }, 0.02, 200 + si, s);
        defer _ = mlx.mlx_array_free(wf);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        var rows: c_int = 2;
        while (rows <= FRAGMENT_ROWS) : (rows += 1) {
            const x = try randBf16(&.{ rows, sh[1] }, 1.0, 30 + si, s);
            defer _ = mlx.mlx_array_free(x);
            const direct = (try qmmKind(.mma, x, w, sc, bi, bits, 64, s)).?;
            defer _ = mlx.mlx_array_free(direct);
            const frag = (try qmmKind(.frag, x, w, sc, bi, bits, 64, s)).?;
            defer _ = mlx.mlx_array_free(frag);
            try expectSame(frag, direct, s);
        }
    };
}
