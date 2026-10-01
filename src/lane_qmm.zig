//! Row-exact 4-, 5-, 6- and 8-bit matmul on the M5 tensor units for
//! 1..MAX_ROWS rows, ported from TensorFold's `lane_qmm.py` and `lane_widen.py`
//! (MIT, see NOTICE). Every 64- (or 32-) input group runs one fixed 16-row
//! `matmul2d` (32 rows past 16, in 32-row blocks across threadgroups) over the
//! packed codes (8-bit as bytes, 5- and 6-bit widened to bytes per group), then
//! `C = s * P + b * XS` in group order in fp32, the K slices summed in slice
//! order; the slice count follows the weight's shape only. A row's bits never
//! depend on how many rows ride with it, so a drafted window verifies with the
//! one-row step's bits. 4-bit weights whose N is a multiple of 64 take the
//! 64-column tile two simdgroups run together (`COOP`), the rest 32 columns (`NARROW`),
//! both over MLX's packed layout or, once `tileInPlace` re-ordered a weight in
//! its own buffer, the tiled one (each column tile's group one contiguous block,
//! scales and biases group-major); all four give the same bits. A tiled weight
//! is read by nothing else: past MAX_ROWS rows (a prompt) it takes `PREFILL`,
//! MLX's NAX qmm with the tiled addressing.
const std = @import("std");
const mlx = @import("mlx.zig");
const steel = @import("mlx_steel_sources");

pub const MAX_ROWS = 128;

const HEADER =
    \\#include <metal_tensor>
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace mpp::tensor_ops;
    \\
;

// XS (a row's sum over a group) is summed in the kernel in a fixed order per
// row: lane 2r + h adds half h of row r, and the two halves join low + high.
const ROW_SUMS =
    \\  const int xr = lane >> 1, xh = lane & 1;
    \\  bool xlive[TMR];
    \\  const device bfloat* xp[TMR];
    \\  for (int t = 0; t < TMR; t++) {
    \\    xlive[t] = rb + t * 16 + xr < M;
    \\    xp[t] = X + size_t(xlive[t] ? rb + t * 16 + xr : 0) * K + xh * (GS / 2);
    \\  }
    \\
;
const ROW_SUM_G =
    \\    float row_sum[TMR];
    \\    for (int t = 0; t < TMR; t++) {
    \\      float half_sum = 0.0f;
    \\      if (xlive[t]) for (int i = 0; i < GS / 2; i++) half_sum += float(xp[t][g * GS + i]);
    \\      const float other = simd_shuffle_xor(half_sum, ushort(1));
    \\      row_sum[t] = xh ? other + half_sum : half_sum + other;
    \\    }
    \\
;

// 32 output columns a simdgroup; SBt holds (s, b) bf16 pairs group-major [K/GS][N][2].
// A threadgroup covers 16 x TMR rows from `rb` with one op per group; every row
// gets the 16-row op's bits.
const NARROW =
    \\  const ushort lane = thread_index_in_simdgroup;
    \\  const ushort sg = simdgroup_index_in_threadgroup;     // K slice
    \\  const short qid = lane >> 2;
    \\  const short fm = (qid & 4) | ((lane >> 1) & 3);       // fragment row of this lane (and fm + 8)
    \\  const short fn = ((qid & 2) | (lane & 1)) * 4;        // first of its four fragment columns
    \\  const int M = mdims[0];
    \\  constexpr int KG = K / GS;
    \\  constexpr int NF = 2;
    \\  const int n0 = threadgroup_position_in_grid.x * 32;
    \\  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
    \\  const int g_begin = (sg * KG) / SK;
    \\  const int g_end = ((sg + 1) * KG) / SK;
    \\  constexpr auto desc = matmul2d_descriptor(16 * TMR, 32, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  matmul2d<desc, execution_simdgroup> op;
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
    \\#if BITS == 4
    \\  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)W, dextents<int32_t, 2>(K, N));
    \\#elif BITS == 8 && !TILED
    \\  tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> tB((device uint8_t*)W, dextents<int32_t, 2>(K, N));
    \\#elif BITS == 8
    \\#else
    \\  // 5- and 6-bit: lane l widens column n0 + l's group into bytes for the bf16 x uint8 op.
    \\  constexpr int WPG = GS * BITS / 32;
    \\  threadgroup uint stage_all[SK * 32 * (GS / 4)];
    \\  threadgroup uint* stage = stage_all + sg * 32 * (GS / 4);
    \\  tensor<threadgroup uint8_t, dextents<int32_t, 2>, tensor_inline> b((threadgroup uint8_t*)stage, dextents<int32_t, 2>(GS, 32));
    \\  const device uint* wcol = TILED ? (const device uint*)W + (int64_t)(R0 / 32 + threadgroup_position_in_grid.x) * KG * 32 * WPG + lane * WPG
    \\                                  : (const device uint*)W + (int64_t)(n0 + lane) * (K * BITS / 32);
    \\#endif
    \\  float C[TMR][NF * 8];
    \\  for (int t = 0; t < TMR; t++) for (int i = 0; i < NF * 8; i++) C[t][i] = 0.0f;
    \\#if !TILED
    \\  const device uint4* sbv = (const device uint4*)SBt;
    \\#endif
    \\  bool colok[NF];
    \\  for (int f = 0; f < NF; f++) colok[f] = n0 + f * 16 + fn < N;
++ "\n" ++ ROW_SUMS ++
    \\  for (int g = g_begin; g < g_end; g++) {
++ "\n" ++ ROW_SUM_G ++
    \\    float s[NF][4], bb[NF][4];
    \\    for (int f = 0; f < NF; f++) {
    \\#if TILED
    \\      const device uint4* sbv = (const device uint4*)(g < KG / 2 ? ST : BT);
    \\      const uint4 q = colok[f] ? sbv[(size_t(g % (KG / 2)) * NTOT + R0 + n0 + f * 16 + fn) / 4] : uint4(0);
    \\#else
    \\      const uint4 q = colok[f] ? sbv[(size_t(g) * N + n0 + f * 16 + fn) / 4] : uint4(0);
    \\#endif
    \\      const vec<bfloat, 8> v = as_type<vec<bfloat, 8>>(q);
    \\      for (int j = 0; j < 4; j++) { s[f][j] = float(v[2 * j]); bb[f][j] = float(v[2 * j + 1]); }
    \\    }
    \\    auto a = tA.slice(g * GS, 0);
    \\#if BITS != 4 && BITS != 8
    \\    {
    \\      // A group is one little-endian bit stream; each word takes 4 values.
    \\      uint w[WPG + 1];
    \\      for (int i = 0; i <= WPG; i++) w[i] = 0;
    \\      if (n0 + lane < N) for (int i = 0; i < WPG; i++) w[i] = wcol[g * (TILED ? 32 * WPG : WPG) + i];
    \\      for (int c = 0; c < GS / 4; c++) {
    \\        const int bit = 4 * BITS * c, i = bit >> 5, sh = bit & 31;
    \\        uint word = w[i] >> sh;
    \\        if (sh + 4 * BITS > 32) word |= w[i + 1] << (32 - sh);
    \\        word = (word & ((1u << (2 * BITS)) - 1u)) | (((word >> (2 * BITS)) & ((1u << (2 * BITS)) - 1u)) << 16);
    \\        word = (word & ((0x10001u << BITS) - 0x10001u)) | (((word >> BITS) & ((0x10001u << BITS) - 0x10001u)) << 8);
    \\        stage[lane * (GS / 4) + c] = word;
    \\      }
    \\    }
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\#elif TILED && BITS == 4
    \\    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
    \\        (device uchar*)W + (int64_t)((R0 / 32 + threadgroup_position_in_grid.x) * KG + g) * (32 * GS / 2), dextents<int32_t, 2>(GS, 32));
    \\#elif TILED
    \\    tensor<device uint8_t, dextents<int32_t, 2>, tensor_inline> b(
    \\        (device uint8_t*)W + (int64_t)((R0 / 32 + threadgroup_position_in_grid.x) * KG + g) * (32 * GS), dextents<int32_t, 2>(GS, 32));
    \\#else
    \\    auto b = tB.slice(g * GS, n0);
    \\#endif
    \\    auto P = op.template get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    \\    op.run(a, b, P);
    \\#if BITS != 4 && BITS != 8
    \\    simdgroup_barrier(mem_flags::mem_threadgroup); // the op has read the stage
    \\#endif
    \\    for (int t = 0; t < TMR; t++) {
    \\      const float xs0 = simd_shuffle(row_sum[t], ushort(2 * fm));
    \\      const float xs1 = simd_shuffle(row_sum[t], ushort(2 * (fm + 8)));
    \\      for (int f = 0; f < NF; f++)
    \\        for (int r = 0; r < 2; r++)
    \\          for (int j = 0; j < 4; j++) {
    \\            const int i = f * 8 + r * 4 + j;
    \\            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));
    \\          }
    \\    }
    \\  }
    \\  // K slices are added in slice order, one 16-row block at a time
    \\  threadgroup float part[(SK > 1 ? SK - 1 : 1) * NF * 8 * 32];
    \\  for (int t = 0; t < TMR; t++) {
    \\    if (SK > 1) {
    \\      if (sg > 0) for (int i = 0; i < NF * 8; i++) part[((sg - 1) * NF * 8 + i) * 32 + lane] = C[t][i];
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      if (sg == 0)
    \\        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < NF * 8; i++) C[t][i] += part[((s2 - 1) * NF * 8 + i) * 32 + lane];
    \\      if (TMR > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (sg == 0)
    \\      for (int f = 0; f < NF; f++)
    \\        for (int r = 0; r < 2; r++) {
    \\          const int m = rb + t * 16 + fm + 8 * r;
    \\          const int n = n0 + f * 16 + fn;
    \\          if (m < M && n < N)
    \\            for (int j = 0; j < 4; j++) Y[m * N + n + j] = static_cast<bfloat>(C[t][f * 8 + r * 4 + j]);
    \\        }
    \\  }
    \\
;

// 64 output columns run by two simdgroups together (N % 64 == 0); a pair per K slice.
const COOP =
    \\  const ushort sg = simdgroup_index_in_threadgroup;
    \\  const ushort lane = thread_index_in_simdgroup;
    \\  const ushort slice = sg >> 1;
    \\  const int M = mdims[0];
    \\  constexpr int KG = K / GS;
    \\  const int n0 = threadgroup_position_in_grid.x * 64;
    \\  const int rb = threadgroup_position_in_grid.y * 16 * TMR;
    \\  const int g_begin = (slice * KG) / SK;
    \\  const int g_end = ((slice + 1) * KG) / SK;
    \\  constexpr auto desc = matmul2d_descriptor(16 * TMR, 64, GS, false, true, false, matmul2d_descriptor::mode::multiply);
    \\  matmul2d<desc, execution_simdgroups<2>> op;
    \\  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));
    \\  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)W, dextents<int32_t, 2>(K, N));
    \\  auto a0 = tA.slice(0, 0);
    \\  auto b0 = tB.slice(0, 0);
    \\  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
    \\  constexpr int CAP = 16 * TMR;                             // 16 TMR x 64 outputs over 64 threads
    \\  short ecol[CAP], erow[CAP];
    \\  for (int i = 0; i < CAP; i++) { auto ids = P.get_multidimensional_index(i); ecol[i] = ids[0]; erow[i] = ids[1]; }
    \\  float C[CAP];
    \\  for (int i = 0; i < CAP; i++) C[i] = 0.0f;
    \\#if !TILED
    \\  const device uint* sbw = (const device uint*)SBt;
    \\#endif
++ "\n" ++ ROW_SUMS ++
    \\  for (int g = g_begin; g < g_end; g++) {
++ "\n" ++ ROW_SUM_G ++
    \\    auto a = tA.slice(g * GS, 0);
    \\#if TILED
    \\    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(
    \\        (device uchar*)W + (int64_t)((R0 / 64 + threadgroup_position_in_grid.x) * KG + g) * (64 * GS / 2), dextents<int32_t, 2>(GS, 64));
    \\#else
    \\    auto b = tB.slice(g * GS, n0);
    \\#endif
    \\    op.run(a, b, P);
    \\    for (int i = 0; i < CAP; i++) {
    \\#if TILED
    \\      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(((const device uint*)(g < KG / 2 ? ST : BT))[size_t(g % (KG / 2)) * NTOT + R0 + n0 + ecol[i]]);
    \\#else
    \\      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(sbw[size_t(g) * N + n0 + ecol[i]]);
    \\#endif
    \\      const float sv = float(sb[0]), bv = float(sb[1]);
    \\      const float xs = simd_shuffle(row_sum[erow[i] >> 4], ushort(2 * (erow[i] & 15)));
    \\      C[i] = fma(sv, P[i], fma(bv, xs, C[i]));
    \\    }
    \\  }
    \\  // K slices are added in slice order, 16 outputs a thread at a time
    \\  threadgroup float part[(SK > 1 ? SK - 1 : 1) * 16 * 64];
    \\  const ushort tip = ushort(thread_position_in_threadgroup.x) - slice * 64;
    \\  if (SK > 1)
    \\    for (int c0 = 0; c0 < CAP; c0 += 16) {
    \\      if (slice > 0) for (int i = 0; i < 16; i++) part[((slice - 1) * 16 + i) * 64 + tip] = C[c0 + i];
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      if (slice == 0)
    \\        for (int s2 = 1; s2 < SK; s2++) for (int i = 0; i < 16; i++) C[c0 + i] += part[((s2 - 1) * 16 + i) * 64 + tip];
    \\      if (TMR > 1) threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\  if (slice == 0)
    \\    for (int i = 0; i < CAP; i++) {
    \\      const int m = rb + erow[i], n = n0 + ecol[i];
    \\      if (m < M) Y[m * N + n] = static_cast<bfloat>(C[i]);
    \\    }
    \\
;

// MLX's affine `qmm_t_nax` (lib/mlx-src/.../quantized_nax.h, MIT) for a tiled
// weight: the same BM x 64 x 64 blocks over 2 x 2 simdgroups, the same
// dequantization into threadgroup memory and the same NAX tile matmuls; only
// the weight and scale addresses differ (and rows past N load zeros).
const PREFILL =
    \\  const ushort simd_gid = simdgroup_index_in_threadgroup;
    \\  const ushort simd_lid = thread_index_in_simdgroup;
    \\  const int M = mdims[0];
    \\  constexpr int BK = 64, BN = 64, WM = 2, WN = 2, KG = K / BK;
    \\  constexpr int BK_padded = BK + 8;
    \\  threadgroup bfloat Ws[BN * BK_padded];
    \\  const int y_row = threadgroup_position_in_grid.y * BM;
    \\  const int y_col = threadgroup_position_in_grid.x * BN;
    \\  // thread t dequantizes 32 codes (4 * BITS bytes) of weight row t / 2 per group
    \\  constexpr int GB = 8 * BITS; // bytes per column per group
    \\  const short tidx = simd_gid * 32 + simd_lid;
    \\  const short bi = tidx / 2, bj = (tidx % 2) * (GB / 2);
    \\  const bool row_ok = y_col + bi < N;
    \\  const int nrow = R0 + y_col + bi;
    \\  const device uint8_t* wsrc = (const device uint8_t*)W + ((int64_t)(nrow / NT) * KG * (NT * GB) + (nrow % NT) * GB + bj);
    \\  threadgroup bfloat* wdst = Ws + bi * BK_padded + (tidx % 2) * 32;
    \\  constexpr short SM = BM / WM, SN = BN / WN, SKK = 32;
    \\  constexpr short TM = SM / 16, TN = SN / 16, TK = SKK / 16;
    \\  const short tm = SM * (simd_gid / WN);
    \\  const short tn = SN * (simd_gid % WN);
    \\  const short sgp_sm = min(int(SM), M - (y_row + tm));
    \\  const bool is_unaligned_sm = (sgp_sm != SM);
    \\  const short sgp_sn = ALIGNED_N ? SN : min(int(SN), N - (y_col + tn));
    \\  const short tgp_bn = ALIGNED_N ? BN : min(BN, int(N - y_col));
    \\  const bool is_unaligned_bn = ALIGNED_N ? false : (tgp_bn != BN);
    \\  device bfloat* y = Y + (int64_t)y_row * N + y_col;
    \\  const device bfloat* x = X + (int64_t)(y_row + tm) * K;
    \\  NAXTile<float, TM, TN> Dtile;
    \\  Dtile.clear();
    \\  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    \\    dispatch_bool(ALIGNED_N || !is_unaligned_bn, [&](auto kAlignedN) {
    \\      for (int g = 0; g < KG; g++) {
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        if (row_ok) {
    \\          const device bfloat* sb = (g < KG / 2 ? ST : BT) + (size_t(g % (KG / 2)) * NTOT + nrow) * 2;
    \\          const float s = float(sb[0]), b = float(sb[1]);
    \\          const device uint8_t* w = wsrc + (int64_t)g * (NT * GB);
    \\#if BITS == 4
    \\          const float sc[2] = {s, s / 16.0f};
    \\          for (int i = 0; i < 16; i++) {
    \\            wdst[2 * i] = static_cast<bfloat>(sc[0] * (w[i] & 0x0f) + b);
    \\            wdst[2 * i + 1] = static_cast<bfloat>(sc[1] * (w[i] & 0xf0) + b);
    \\          }
    \\#else
    \\          // code i sits at bit i * BITS of the little-endian stream (MLX's `code * s + b`)
    \\          for (int i = 0; i < 32; i++) {
    \\            const int bit = i * BITS, by = bit >> 3, sh = bit & 7;
    \\            const uint hi = sh + BITS > 8 ? uint(w[by + 1]) << 8 : 0u;
    \\            const uint code = ((uint(w[by]) | hi) >> sh) & ((1u << BITS) - 1u);
    \\            wdst[i] = static_cast<bfloat>(code * s + b);
    \\          }
    \\#endif
    \\        } else {
    \\          for (int i = 0; i < 32; i++) wdst[i] = bfloat(0);
    \\        }
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\        STEEL_PRAGMA_NO_UNROLL
    \\        for (int kk1 = 0; kk1 < BK; kk1 += SKK) {
    \\          NAXTile<bfloat, TM, TK> Atile;
    \\          NAXTile<bfloat, TN, TK> Btile;
    \\          volatile int compiler_barrier;
    \\          if constexpr (kAlignedM.value) {
    \\            Atile.load(x + kk1, K);
    \\          } else {
    \\            Atile.load_safe(x + kk1, K, short2(SKK, sgp_sm));
    \\          }
    \\          Btile.template load<bfloat, BK_padded, 1>(Ws + tn * BK_padded + kk1);
    \\          tile_matmad_nax(Dtile, Atile, metal::bool_constant<false>{}, Btile, metal::bool_constant<true>{});
    \\          (void)compiler_barrier;
    \\        }
    \\        x += BK;
    \\      }
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      if constexpr (kAlignedM.value && kAlignedN.value) {
    \\        Dtile.store(y + tm * N + tn, N);
    \\      } else if (kAlignedM.value && sgp_sn == SN) {
    \\        Dtile.store(y + tm * N + tn, N);
    \\      } else {
    \\        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
    \\      }
    \\    });
    \\  });
    \\
;

/// MLX's steel NAX headers in include order, their own includes and
/// include guards dropped: they sit inline in one `metal_kernel` header.
fn prefillHeader() ![:0]const u8 {
    if (prefill_header) |h| return h;
    const a = std.heap.c_allocator;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\n");
    for ([_][]const u8{ steel.defines, steel.type_traits, steel.integral_constant, steel.nax }) |src| {
        var lines = std.mem.splitScalar(u8, src, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "#include \"mlx/") or std.mem.startsWith(u8, line, "#pragma once")) continue;
            try out.appendSlice(a, line);
            try out.append(a, '\n');
        }
    }
    try out.appendSlice(a, "using namespace metal;\nusing namespace mlx::steel;\n");
    const h = try out.toOwnedSliceSentinel(a, 0);
    prefill_header = h;
    return h;
}
var prefill_header: ?[:0]const u8 = null;

/// K slices for an (n, k) weight: fixed by the shape, never by the row count.
fn splitK(n: c_int, k: c_int) c_int {
    const tiles = @divTrunc(n + 31, 32);
    var sk: c_int = 1;
    while (sk < 8 and tiles * sk < 1024 and @divTrunc(@divTrunc(k, 64), sk * 2) >= 8) sk *= 2;
    return sk;
}

/// The pick follows the weight's shape only: 64-column tiles where N allows.
fn coopFor(n: c_int) bool {
    return @rem(n, 64) == 0;
}

const Kind = enum { narrow, coop, prefill };
/// `tmr` is the prefill block's BM there; `ntot`/`r0` place a tiled view in its buffer.
const KernelKey = struct { kind: Kind, tiled: bool, tmr: c_int, k: c_int, n: c_int, gs: c_int, sk: c_int, ntot: c_int = 0, r0: c_int = 0, nt: c_int = 0, bits: c_int = 4 };

/// 16-row blocks a threadgroup's op covers: one up to 16 rows, two past it.
fn tmrFor(rows: c_int) c_int {
    return if (rows <= 16) 1 else 2;
}
var kernels: std.AutoHashMapUnmanaged(KernelKey, mlx.mlx_fast_metal_kernel) = .{};
const PlanKey = struct { kind: Kind, rows: c_int, n: c_int, k: c_int };
/// Launch plans and row counts at lane widths (rows <= MAX_ROWS). A
/// prompt-width call builds its own and frees it: a long session's chunk
/// tails take every row count there is.
var plans: std.AutoHashMapUnmanaged(PlanKey, mlx.mlx_fast_metal_kernel_config) = .{};
var mdims_cache: std.AutoHashMapUnmanaged(c_int, mlx.mlx_array) = .{};

/// (s, b) pairs group-major per (scales, biases) of a weight read in MLX's
/// layout, built on first use (an eighth of the weight's bytes: the lm_head
/// and the ragged GDN gates on a tiled trunk). Keyed by the data the handles
/// point at (a handle's address is reused once its caller frees it) plus the
/// shape (a joined weight and its first part start at the same address); an
/// entry holds its sources alive, so no key is reused.
const DerivedKey = struct { a: usize, b: usize, n: c_int, w: c_int };
const Derived = struct { src_a: mlx.mlx_array, src_b: mlx.mlx_array, out: mlx.mlx_array };
var packed_scales: std.AutoHashMapUnmanaged(DerivedKey, Derived) = .{};

/// A weight (or a row view of one) whose buffer `tileInPlace` re-ordered: the
/// base handles it lives in, its first row there, and the column tile width.
/// The entry holds the base alive, so its addresses are never reused.
const Tiled = struct { owner: usize, w: mlx.mlx_array, st: mlx.mlx_array, bt: mlx.mlx_array, ntot: c_int, r0: c_int, nt: c_int, bits: c_int };
const TiledKey = struct { ptr: usize, n: c_int, kw: c_int };
var tiled: std.AutoHashMapUnmanaged(TiledKey, Tiled) = .{};
/// Registered weights per shape: a weight of any other shape is answered
/// without a data read (a lazy weight's eval mid-graph).
const TiledShape = struct { n: c_int, kw: c_int };
var tiled_shapes: std.AutoHashMapUnmanaged(TiledShape, u32) = .{};

fn shapeRegistered(key: TiledKey) !void {
    const gop = try tiled_shapes.getOrPut(std.heap.c_allocator, .{ .n = key.n, .kw = key.kw });
    if (!gop.found_existing) gop.value_ptr.* = 0;
    gop.value_ptr.* += 1;
}

fn shapeReleased(key: TiledKey) void {
    const count = tiled_shapes.getPtr(.{ .n = key.n, .kw = key.kw }) orelse return;
    count.* -= 1;
    if (count.* == 0) _ = tiled_shapes.remove(.{ .n = key.n, .kw = key.kw });
}

/// Frees `owner`'s tiled entries and EVERY model's derived (s, b) copies (a
/// model unload): a copy holds its sources alive and nothing says whose they
/// are, and another model rebuilds its own on the next read. That model's
/// tiled weights stay registered.
pub fn release(owner: usize) void {
    var it = packed_scales.valueIterator();
    while (it.next()) |e| {
        _ = mlx.mlx_array_free(e.src_a);
        if (e.src_b.ctx != null) _ = mlx.mlx_array_free(e.src_b);
        _ = mlx.mlx_array_free(e.out);
    }
    packed_scales.clearAndFree(std.heap.c_allocator);
    var keys: std.ArrayList(TiledKey) = .empty;
    defer keys.deinit(std.heap.c_allocator);
    var ti = tiled.iterator();
    while (ti.next()) |e| if (e.value_ptr.owner == owner) keys.append(std.heap.c_allocator, e.key_ptr.*) catch {};
    for (keys.items) |k| if (tiled.fetchRemove(k)) |kv| {
        for ([_]mlx.mlx_array{ kv.value.w, kv.value.st, kv.value.bt }) |h| _ = mlx.mlx_array_free(h);
        shapeReleased(k);
    };
}

fn dataKey(a: mlx.mlx_array, b: ?mlx.mlx_array) !DerivedKey {
    // Loaded weights are evaluated already (a no-op); a data read needs it.
    try mlx.check(mlx.mlx_array_eval(a));
    const shape = mlx.getShape(a);
    const ap: usize = switch (mlx.mlx_array_dtype(a)) {
        .uint32 => @intFromPtr(mlx.mlx_array_data_uint32(a) orelse return error.UnreadableWeight),
        else => @intFromPtr(mlx.mlx_array_data_bfloat16(a) orelse return error.UnreadableWeight),
    };
    var bp: usize = 0;
    if (b) |bb| {
        try mlx.check(mlx.mlx_array_eval(bb));
        bp = @intFromPtr(mlx.mlx_array_data_bfloat16(bb) orelse return error.UnreadableWeight);
    }
    return .{ .a = ap, .b = bp, .n = shape[0], .w = shape[1] };
}

fn hold(a: mlx.mlx_array) !mlx.mlx_array {
    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_array_set(&h, a));
    return h;
}

fn packedFor(sc: mlx.mlx_array, bi: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const key = try dataKey(sc, bi);
    if (packed_scales.get(key)) |e| return e.out;
    var st = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(st);
    try mlx.check(mlx.mlx_transpose(&st, sc, s));
    var bt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bt);
    try mlx.check(mlx.mlx_transpose(&bt, bi, s));
    const pair = [_]mlx.mlx_array{ st, bt };
    const vec = mlx.mlx_vector_array_new_data(&pair, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stacked);
    try mlx.check(mlx.mlx_stack_axis(&stacked, vec, -1, s));
    var sbt = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sbt);
    try mlx.check(mlx.mlx_contiguous(&sbt, stacked, false, s));
    const held_a = try hold(sc);
    errdefer _ = mlx.mlx_array_free(held_a);
    const held_b = try hold(bi);
    errdefer _ = mlx.mlx_array_free(held_b);
    try packed_scales.put(std.heap.c_allocator, key, .{ .src_a = held_a, .src_b = held_b, .out = sbt });
    return sbt;
}

fn dropPacked(sc: mlx.mlx_array, bi: mlx.mlx_array) void {
    const key = dataKey(sc, bi) catch return;
    if (packed_scales.fetchRemove(key)) |kv| {
        for ([_]mlx.mlx_array{ kv.value.src_a, kv.value.src_b, kv.value.out }) |h| _ = mlx.mlx_array_free(h);
    }
}

fn tiledEntry(w: mlx.mlx_array) !?Tiled {
    if (tiled.count() == 0 or mlx.mlx_array_dtype(w) != .uint32) return null;
    const sh = mlx.getShape(w);
    if (sh.len != 2 or !tiled_shapes.contains(.{ .n = sh[0], .kw = sh[1] })) return null;
    try mlx.check(mlx.mlx_array_eval(w));
    const p = @intFromPtr(mlx.mlx_array_data_uint32(w) orelse return null);
    return tiled.get(.{ .ptr = p, .n = sh[0], .kw = sh[1] });
}

/// Writes `bytes` of `a`'s evaluated contents from `offset` over `dst`'s buffer.
fn overwrite(dst: mlx.mlx_array, a: mlx.mlx_array, offset: usize, bytes: usize) !void {
    try mlx.check(mlx.mlx_array_eval(a));
    const d: [*]u8 = switch (mlx.mlx_array_dtype(dst)) {
        .uint32 => @ptrCast(@constCast(mlx.mlx_array_data_uint32(dst) orelse return error.UnreadableWeight)),
        else => @ptrCast(@constCast(mlx.mlx_array_data_bfloat16(dst) orelse return error.UnreadableWeight)),
    };
    const src: [*]const u8 = switch (mlx.mlx_array_dtype(a)) {
        .uint32 => @ptrCast(mlx.mlx_array_data_uint32(a) orelse return error.UnreadableWeight),
        else => @ptrCast(mlx.mlx_array_data_bfloat16(a) orelse return error.UnreadableWeight),
    };
    @memcpy(d[0..bytes], src[offset .. offset + bytes]);
}

fn rowContiguous(a: mlx.mlx_array) bool {
    const sh = mlx.getShape(a);
    const st = mlx.mlx_array_strides(a);
    return sh.len == 2 and st[1] == 1 and st[0] == @as(usize, @intCast(sh[1]));
}

/// Re-orders a 4-bit, group-64 weight in its own buffer into column tiles
/// (64 wide where N allows, else 32) and its (scale, bias) pairs group-major
/// over the two buffers (the first half of the groups in the scales'), then
/// registers it and the row views `widths` splits it into (a joined
/// weight's parts). From then on only this module reads it; the caller has
/// synchronized the GPU. Returns the bytes re-ordered, 0 when it does not fit.
pub fn tileInPlace(owner: usize, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, widths: []const c_int, s: mlx.mlx_stream) !u64 {
    // Shapes alone cannot tell 4-bit g64 from 8-bit g32: the caller says which.
    if (group_size != 64 or !fits(w, sc, bi, bits, 64)) return 0;
    const ws = mlx.getShape(w);
    const n = ws[0];
    const kw = ws[1];
    const wpg: c_int = @intCast(2 * bits); // words per column per group
    const kg = mlx.getShape(sc)[1];
    if (@rem(n, 32) != 0 or kg * wpg != kw or @rem(kg, 2) != 0) return 0;
    if (!rowContiguous(w) or !rowContiguous(sc) or !rowContiguous(bi)) return 0;
    if (try tiledEntry(w) != null) return 0;
    const nt: c_int = if (bits == 4 and coopFor(n)) 64 else 32;
    // [N, KW] -> [N/nt][K/64][nt columns x a group's words]
    var r4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r4);
    try mlx.check(mlx.mlx_reshape(&r4, w, &[_]c_int{ @divExact(n, nt), nt, kg, wpg }, 4, s));
    var tr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tr);
    try mlx.check(mlx.mlx_transpose_axes(&tr, r4, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var tw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tw);
    try mlx.check(mlx.mlx_contiguous(&tw, tr, false, s));
    const sbt = try packedFor(sc, bi, s);
    const w_bytes: usize = @intCast(@as(i64, n) * kw * 4);
    const s_bytes: usize = @intCast(@as(i64, n) * kg * 2);
    try overwrite(w, tw, 0, w_bytes);
    try overwrite(sc, sbt, 0, s_bytes);
    try overwrite(bi, sbt, s_bytes, s_bytes);
    dropPacked(sc, bi);
    const base = @intFromPtr(mlx.mlx_array_data_uint32(w).?);
    var r0: c_int = 0;
    const views = [_]c_int{n};
    for ([_][]const c_int{ &views, widths }, 0..) |list, li| for (list) |rows| {
        const key = TiledKey{ .ptr = base + @as(usize, @intCast(r0)) * @as(usize, @intCast(kw)) * 4, .n = rows, .kw = kw };
        const e = Tiled{ .owner = owner, .w = try hold(w), .st = try hold(sc), .bt = try hold(bi), .ntot = n, .r0 = r0, .nt = nt, .bits = @intCast(bits) };
        try tiled.put(std.heap.c_allocator, key, e);
        try shapeRegistered(key);
        if (li == 1) r0 += rows;
    };
    return w_bytes + 2 * s_bytes;
}

fn kernelFor(key: KernelKey) !mlx.mlx_fast_metal_kernel {
    if (kernels.get(key)) |k| return k;
    const a = std.heap.c_allocator;
    const consts = try std.fmt.allocPrint(a, "#define BITS {d}\n#define TILED {d}\n  constexpr int TMR = {d};\n  constexpr int BM = {d};\n  constexpr int K = {d};\n  constexpr int N = {d};\n  constexpr int GS = {d};\n  constexpr int SK = {d};\n  constexpr int NTOT = {d};\n  constexpr int R0 = {d};\n  constexpr int NT = {d};\n  constexpr bool ALIGNED_N = {};\n", .{ key.bits, @intFromBool(key.tiled), key.tmr, key.tmr, key.k, key.n, key.gs, key.sk, key.ntot, key.r0, key.nt, @rem(key.n, 64) == 0 });
    defer a.free(consts);
    const body = switch (key.kind) {
        .narrow => NARROW,
        .coop => COOP,
        .prefill => PREFILL,
    };
    const source = try std.mem.concatWithSentinel(a, u8, &.{ consts, body }, 0);
    defer a.free(source);
    const name = try std.fmt.allocPrintSentinel(a, "msv_lane_qmm_{t}{s}_b{d}_t{d}_k{d}_n{d}_g{d}_s{d}_o{d}_{d}", .{ key.kind, if (key.tiled) "_tiled" else "", key.bits, key.tmr, key.k, key.n, key.gs, key.sk, key.ntot, key.r0 }, 0);
    defer a.free(name);
    const in_plain = [_][*:0]const u8{ "X", "W", "SBt", "mdims" };
    const in_tiled = [_][*:0]const u8{ "X", "W", "ST", "BT", "mdims" };
    const in_names: []const [*:0]const u8 = if (key.tiled) &in_tiled else &in_plain;
    const out_names = [_][*:0]const u8{"Y"};
    const in_vec = mlx.mlx_vector_string_new_data(in_names.ptr, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const header: [*:0]const u8 = if (key.kind == .prefill) (try prefillHeader()).ptr else HEADER;
    const k = mlx.mlx_fast_metal_kernel_new(name.ptr, in_vec, out_vec, source.ptr, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    try kernels.put(a, key, k);
    return k;
}

fn planFor(key: PlanKey) !mlx.mlx_fast_metal_kernel_config {
    if (plans.get(key)) |p| return p;
    const cfg = try buildPlan(key);
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try plans.put(std.heap.c_allocator, key, cfg);
    return cfg;
}

fn buildPlan(key: PlanKey) !mlx.mlx_fast_metal_kernel_config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.rows, key.n }, 2, .bfloat16));
    if (key.kind == .prefill) {
        const bm = prefillBm(key.rows);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(key.n + 63, 64) * 128, @divTrunc(key.rows + bm - 1, bm), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    } else {
        const sk = splitK(key.n, key.k);
        const width: c_int = if (key.kind == .coop) 64 else 32;
        const block = 16 * tmrFor(key.rows);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @divTrunc(key.n + width - 1, width) * width * sk, @divTrunc(key.rows + block - 1, block), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, width * sk, 1, 1));
    }
    return cfg;
}

/// MLX's block height: 32 rows when one block covers them all.
fn prefillBm(rows: c_int) c_int {
    return if (rows <= 32) 32 else 64;
}

/// A 4-, 5-, 6- or 8-bit affine bf16 matrix [N, K * bits / 32] in groups of
/// 64 or 32, K % 64 == 0, N % 4 == 0.
pub fn fits(w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32) bool {
    if ((bits != 4 and bits != 5 and bits != 6 and bits != 8) or (group_size != 64 and group_size != 32) or bi.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16) return false;
    const ws = mlx.getShape(w);
    const b: c_int = @intCast(bits);
    return ws.len == 2 and @rem(ws[1] * 32, b * 64) == 0 and @rem(ws[0], 4) == 0;
}

fn mdimsFor(rows: c_int) !mlx.mlx_array {
    if (mdims_cache.get(rows)) |m| return m;
    const m = newMdims(rows);
    errdefer _ = mlx.mlx_array_free(m);
    try mdims_cache.put(std.heap.c_allocator, rows, m);
    return m;
}

fn newMdims(rows: c_int) mlx.mlx_array {
    const d = [_]i32{rows};
    return mlx.mlx_array_new_data(&d, &[_]c_int{1}, 1, .int32);
}

fn run(key: KernelKey, rows: c_int, x: mlx.mlx_array, xs: []const c_int, weights: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, key.k }, 2, s));
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_contiguous(&xc, x2, false, s));
    const mk = try kernelFor(key);
    const cached = rows <= MAX_ROWS;
    const pkey: PlanKey = .{ .kind = key.kind, .rows = rows, .n = key.n, .k = key.k };
    const mcfg = if (cached) try planFor(pkey) else try buildPlan(pkey);
    defer if (!cached) {
        _ = mlx.mlx_fast_metal_kernel_config_free(mcfg);
    };
    const mdims = if (cached) try mdimsFor(rows) else newMdims(rows);
    defer if (!cached) {
        _ = mlx.mlx_array_free(mdims);
    };
    var ins: [5]mlx.mlx_array = undefined;
    ins[0] = xc;
    @memcpy(ins[1 .. 1 + weights.len], weights);
    ins[1 + weights.len] = mdims;
    const in_vec = mlx.mlx_vector_array_new_data(&ins, weights.len + 2);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, mk, in_vec, mcfg, s));
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outs, 0));
    var shape: [8]c_int = undefined;
    @memcpy(shape[0 .. xs.len - 1], xs[0 .. xs.len - 1]);
    shape[xs.len - 1] = key.n;
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&y, y2, &shape, xs.len, s));
    return y;
}

/// `x [..., K] @ w.T` for a weight `tileInPlace` re-ordered, at any row count;
/// null for every other weight. Up to MAX_ROWS rows a tile-aligned view takes
/// the lane kernel's bits, past it (or unaligned) MLX's prompt-width qmm.
pub fn tiledQmm(x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    const t = (try tiledEntry(w)) orelse return null;
    if (!mlx.streamIsGpu(s)) return error.TiledWeightOffGpu;
    const xs = mlx.getShape(x);
    const n = mlx.getShape(w)[0];
    const k = @divExact(mlx.getShape(w)[1] * 32, t.bits);
    if (xs.len == 0 or xs.len > 8 or xs[xs.len - 1] != k or mlx.mlx_array_dtype(x) != .bfloat16) return error.TiledWeightShape;
    var rows: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= d;
    const coop = t.nt == 64;
    const lane = rows <= MAX_ROWS and @rem(t.r0, t.nt) == 0 and (!coop or coopFor(n));
    const key: KernelKey = if (lane)
        .{ .kind = if (coop) .coop else .narrow, .tiled = true, .tmr = tmrFor(rows), .k = k, .n = n, .gs = 64, .sk = splitK(n, k), .ntot = t.ntot, .r0 = t.r0, .bits = t.bits }
    else
        .{ .kind = .prefill, .tiled = true, .tmr = prefillBm(rows), .k = k, .n = n, .gs = 64, .sk = 1, .ntot = t.ntot, .r0 = t.r0, .nt = t.nt, .bits = t.bits };
    return try run(key, rows, x, xs, &.{ t.w, t.st, t.bt }, s);
}

/// `x [..., K] @ w.T` for 1..MAX_ROWS rows on the tensor units (any row count
/// for a tiled weight), or null outside the kernel (the caller's other
/// row-exact kernels take it).
pub fn qmm(x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, bits: u32, group_size: u32, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (try tiledQmm(x, w, s)) |y| return y;
    if (!mlx.streamIsGpu(s) or !fits(w, sc, bi, bits, group_size) or mlx.mlx_array_dtype(x) != .bfloat16) return null;
    const xs = mlx.getShape(x);
    if (xs.len == 0 or xs.len > 8) return null;
    const ws = mlx.getShape(w);
    const k = xs[xs.len - 1];
    if (k * @as(c_int, @intCast(bits)) != ws[1] * 32) return null;
    var rows: c_int = 1;
    for (xs[0 .. xs.len - 1]) |d| rows *= d;
    if (rows < 1 or rows > MAX_ROWS) return null;
    const n = ws[0];
    const gs: c_int = @intCast(group_size);
    // Wider codes are widened per group into bytes, 32 columns at a time.
    const coop = bits == 4 and coopFor(n);
    const sbt = try packedFor(sc, bi, s);
    return try run(.{ .kind = if (coop) .coop else .narrow, .tiled = false, .tmr = tmrFor(rows), .k = k, .n = n, .gs = gs, .sk = splitK(n, k), .bits = @intCast(bits) }, rows, x, xs, &.{ w, sbt }, s);
}

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

test "lane_qmm: prompt-width calls on a tiled weight leave the plan caches at their lane-width size" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    defer release(0);
    const wf = try randBf16(&.{ 256, 1024 }, 0.02, 300, s);
    defer _ = mlx.mlx_array_free(wf);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var parts: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (parts) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&parts, 0..) |*a, i| try mlx.check(mlx.mlx_vector_array_get(a, triple, i));
    try mlx.check(mlx.mlx_array_eval(parts[0]));
    try testing.expect(try tileInPlace(0, parts[0], parts[1], parts[2], 4, 64, &.{}, s) > 0);
    const x = try randBf16(&.{ MAX_ROWS + 3, 1024 }, 1.0, 301, s);
    defer _ = mlx.mlx_array_free(x);
    var sizes: [2]usize = undefined;
    for ([_]c_int{ 8, MAX_ROWS + 1, MAX_ROWS + 2, MAX_ROWS + 3 }, 0..) |rows, i| {
        var xr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xr);
        try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ rows, 1024 }, 2, &[_]c_int{ 1, 1 }, 2, s));
        const y = (try tiledQmm(xr, parts[0], s)).?;
        defer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_array_eval(y));
        // The lane-width call fills the caches; prompt-width ones add nothing.
        if (i == 0) sizes = .{ plans.count(), mdims_cache.count() };
    }
    try testing.expectEqual(sizes[0], plans.count());
    try testing.expectEqual(sizes[1], mdims_cache.count());
}

test "lane_qmm: every row of an R-row call equals its one-row call bit for bit up to 128 rows, at 4, 5, 6 and 8 bits, tiled in place or not, and the product is the fp32 one" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    errdefer {
        var buf: [512]u8 = undefined;
        if (mlx.takeError(&buf)) |msg| std.debug.print("[lane_qmm] mlx: {s}\n", .{msg});
    }
    defer release(0);
    // 64-column tiles (MLP, down projection), 32-column ones (a joined GDN
    // projection: N % 64 == 32), and a ragged untiled one (GDN a/b: 48 columns).
    for ([_]u32{ 4, 5, 6, 8 }) |bits| for ([_][2]c_int{ .{ 1024, 5120 }, .{ 5120, 1024 }, .{ 16480, 1024 }, .{ 48, 5120 } }, 0..) |sh, si| for ([_]u32{ 64, 32 }) |gs| {
        const wf = try randBf16(&.{ sh[0], sh[1] }, 0.02, 100 + si, s);
        defer _ = mlx.mlx_array_free(wf);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&w, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));
        const x = try randBf16(&.{ MAX_ROWS, sh[1] }, 1.0, 7 + si, s);
        defer _ = mlx.mlx_array_free(x);
        // fp32 truth: the dequantized weight times x in fp32.
        var wd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wd);
        try mlx.check(mlx.mlx_dequantize(&wd, w, sc, bi, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        var xf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xf);
        try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, wd, s));
        var truth = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(truth);
        try mlx.check(mlx.mlx_matmul(&truth, xf, wt, s));
        const plain = (try qmm(x, w, sc, bi, bits, gs, s)).?;
        defer _ = mlx.mlx_array_free(plain);
        // Everything read in MLX's layout is evaluated before the buffers change.
        try mlx.check(mlx.mlx_array_eval(truth));
        try mlx.check(mlx.mlx_array_eval(plain));
        _ = try tileInPlace(0, w, sc, bi, bits, gs, &.{}, s);
        const all = (try qmm(x, w, sc, bi, bits, gs, s)).?;
        defer _ = mlx.mlx_array_free(all);
        try testing.expect(try bitEqual(all, plain, s));
        // Parity: relative RMS error vs fp32 truth at bf16 output rounding.
        {
            var af = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(af);
            try mlx.check(mlx.mlx_astype(&af, all, .float32, s));
            var d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(d);
            try mlx.check(mlx.mlx_subtract(&d, af, truth, s));
            const err = try meanSquare(d, s);
            const ref = try meanSquare(truth, s);
            try testing.expect(@sqrt(err / ref) < 1e-2);
        }
        for ([_]c_int{ 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 18, 31, 32, 33, 47, 48, 49, 64, 65, 100, 127, 128 }) |r| {
            var xr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xr);
            try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ r, sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
            const part = (try qmm(xr, w, sc, bi, bits, gs, s)).?;
            defer _ = mlx.mlx_array_free(part);
            // Row r-1 of the r-row call == row r-1 of the one-row call == row r-1 of the full call.
            var last = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(last);
            try mlx.check(mlx.mlx_slice(&last, xr, &[_]c_int{ r - 1, 0 }, 2, &[_]c_int{ r, sh[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
            const one = (try qmm(last, w, sc, bi, bits, gs, s)).?;
            defer _ = mlx.mlx_array_free(one);
            var pr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pr);
            try mlx.check(mlx.mlx_slice(&pr, part, &[_]c_int{ r - 1, 0 }, 2, &[_]c_int{ r, sh[0] }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var fr = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(fr);
            try mlx.check(mlx.mlx_slice(&fr, all, &[_]c_int{ r - 1, 0 }, 2, &[_]c_int{ r, sh[0] }, 2, &[_]c_int{ 1, 1 }, 2, s));
            try testing.expect(try bitEqual(pr, one, s));
            try testing.expect(try bitEqual(fr, one, s));
        }
    };
}

fn quantized(n: c_int, k: c_int, seed: u64, s: mlx.mlx_stream) ![3]mlx.mlx_array {
    return quantizedBits(n, k, 4, seed, s);
}

fn quantizedBits(n: c_int, k: c_int, bits: u32, seed: u64, s: mlx.mlx_stream) ![3]mlx.mlx_array {
    const wf = try randBf16(&.{ n, k }, 0.02, seed, s);
    defer _ = mlx.mlx_array_free(wf);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
    var out: [3]mlx.mlx_array = undefined;
    for (&out, 0..) |*a, j| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, triple, j));
        try mlx.check(mlx.mlx_array_eval(a.*));
    }
    return out;
}

/// MLX's own pick in `QuantizedMatmul::eval_gpu` / `qmm_splitk` (transposed,
/// group 64): NAX qmm unless under 512 threadgroups of 32 x 32 leave a K split.
fn mlxTakesNax(m: c_int, n: c_int, k: c_int) bool {
    var split: c_int = @max(1, @divTrunc(512, @divTrunc(n + 31, 32) * @divTrunc(m + 31, 32)));
    split = @min(split, @divTrunc(k, 64));
    while (split > 1 and @rem(k, split * 64) != 0) split -= 1;
    return split <= 1;
}

/// x times the dequantized weight in fp32, evaluated.
fn fp32Truth(x: mlx.mlx_array, t: [3]mlx.mlx_array, bits: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    var wd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wd);
    try mlx.check(mlx.mlx_dequantize(&wd, t[0], t[1], t[2], mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wd, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&out, xf, wt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

fn relErr(y: mlx.mlx_array, truth: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var yf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(yf);
    try mlx.check(mlx.mlx_astype(&yf, y, .float32, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, yf, truth, s));
    return @sqrt((try meanSquare(d, s)) / (try meanSquare(truth, s)));
}

fn rowsOf(a: mlx.mlx_array, r0: c_int, rows: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const cols = mlx.getShape(a)[1];
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, a, &[_]c_int{ r0, 0 }, 2, &[_]c_int{ r0 + rows, cols }, 2, &[_]c_int{ 1, 1 }, 2, s));
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

test "lane_qmm: past 128 rows a weight tiled in place gives MLX's qmm bits at 4, 5, 6 and 8 bits, whole and for every row view" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    defer release(0);
    // A joined GDN in-projection in miniature (32-column tiles, views of 48 rows
    // off the tile grid) and a joined gate|up (64-column tiles).
    for ([_]u32{ 4, 5, 6, 8 }) |bits| for ([_][]const c_int{ &.{ 1024, 512, 48, 48 }, &.{ 512, 512 } }, 0..) |widths, ci| {
        var n: c_int = 0;
        for (widths) |wd| n += wd;
        const k: c_int = 1024;
        const t = try quantizedBits(n, k, bits, 400 + ci, s);
        defer for (t) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var views: [5][3]mlx.mlx_array = undefined;
        var starts: [5]c_int = undefined;
        var rows_of: [5]c_int = undefined;
        rows_of[0] = n;
        starts[0] = 0;
        var r0: c_int = 0;
        for (widths, 1..) |wd, i| {
            starts[i] = r0;
            rows_of[i] = wd;
            r0 += wd;
        }
        const nv = widths.len + 1;
        for (0..nv) |i| for (0..3) |j| {
            views[i][j] = try rowsOf(t[j], starts[i], rows_of[i], s);
        };
        defer for (views[0..nv]) |v| for (v) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const row_counts = [_]c_int{ 129, 300, 8200 };
        var refs: [3][5]mlx.mlx_array = undefined;
        var truths: [3][5]mlx.mlx_array = undefined;
        var xs: [3]mlx.mlx_array = undefined;
        for (row_counts, 0..) |rows, ri| {
            xs[ri] = try randBf16(&.{ rows, k }, 1.0, 30 + ri, s);
            for (0..nv) |i| {
                refs[ri][i] = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(&refs[ri][i], xs[ri], views[i][0], views[i][1], views[i][2], true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
                try mlx.check(mlx.mlx_array_eval(refs[ri][i]));
                truths[ri][i] = try fp32Truth(xs[ri], views[i], bits, s);
            }
        }
        defer for (0..row_counts.len) |ri| {
            _ = mlx.mlx_array_free(xs[ri]);
            for (refs[ri][0..nv]) |a| _ = mlx.mlx_array_free(a);
            for (truths[ri][0..nv]) |a| _ = mlx.mlx_array_free(a);
        };
        try testing.expect(try tileInPlace(0, t[0], t[1], t[2], bits, 64, widths, s) > 0);
        for (0..row_counts.len) |ri| for (0..nv) |i| {
            const y = (try tiledQmm(xs[ri], views[i][0], s)).?;
            defer _ = mlx.mlx_array_free(y);
            if (mlxTakesNax(row_counts[ri], rows_of[i], k)) {
                try testing.expect(try bitEqual(y, refs[ri][i], s));
            } else {
                // MLX splits K there (bf16 partials): no worse than it against fp32 truth.
                try testing.expect(try relErr(y, truths[ri][i], s) <= try relErr(refs[ri][i], truths[ri][i], s));
            }
        };
    };
}

test "lane_qmm: releasing one model's tiled weights keeps another's" {
    if (!@import("transformer.zig").naxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const a = try quantized(256, 512, 500, s);
    defer for (a) |h| {
        _ = mlx.mlx_array_free(h);
    };
    const b = try quantized(256, 512, 501, s);
    defer for (b) |h| {
        _ = mlx.mlx_array_free(h);
    };
    const x = try randBf16(&.{ 3, 512 }, 1.0, 9, s);
    defer _ = mlx.mlx_array_free(x);
    const want = (try qmm(x, b[0], b[1], b[2], 4, 64, s)).?;
    defer _ = mlx.mlx_array_free(want);
    try mlx.check(mlx.mlx_array_eval(want));
    _ = try tileInPlace(1, a[0], a[1], a[2], 4, 64, &.{}, s);
    _ = try tileInPlace(2, b[0], b[1], b[2], 4, 64, &.{}, s);
    release(1);
    defer release(2);
    try testing.expect(try tiledEntry(a[0]) == null);
    const got = (try tiledQmm(x, b[0], s)).?;
    defer _ = mlx.mlx_array_free(got);
    try testing.expect(try bitEqual(got, want, s));
}

fn meanSquare(a: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_square(&sq, a, s));
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_mean(&m, sq, false, s));
    try mlx.check(mlx.mlx_array_eval(m));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, m));
    return v;
}

fn bitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var e = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e);
    try mlx.check(mlx.mlx_array_equal(&e, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(e));
    var v: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, e));
    return v;
}
