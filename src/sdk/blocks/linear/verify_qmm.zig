//! sdk/blocks/linear/verify_qmm.
//! Owns vqmmAccLines, vqmmColumnLoad, vqmmColumnChain, verifyQmmSource, verifyQmmMsgSource, vqmmMsgBn, and 63 more.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const macosVersionAtLeast = @import("../../../mlx/device.zig").macosVersionAtLeast;
const naxArchGeneration = @import("../../../mlx/device.zig").naxArchGeneration;
const KernelCfgCache = @import("../../../mlx/kernel_cache.zig").KernelCfgCache;
const naxMinMForBits = @import("verify_qmm_nax.zig").naxMinMForBits;
const runVerifyQmmNax = @import("verify_qmm_nax.zig").runVerifyQmmNax;
const verifyQmmTile = @import("verify_qmm_nax.zig").verifyQmmTile;

// ── Verify-width split-K quantized matmul (spec-decode fast path) ──
//
// Stock MLX qmm is tuned for M=1 decode (qmv) and large-M prefill (steel);
// the M=2..8-row shapes of speculative VERIFY forwards (and small decode
// batches) fall in a dead zone that underuses memory bandwidth, and at huge
// N (lm_head) the tiny-tile grid thrashes the scheduler. This kernel is a
// port of MTPLX's split-K verify family (mtplx/verify_kernels.py,
// Apache-2.0; design + measurement ledger in that file's header — their
// in-context winner over the wide multi-simdgroup tile): each threadgroup
// owns FOUR output columns, the K reduction splits across K_PARTS
// simdgroups (2 for big N, 4 for small N — deep occupancy queues + latency
// hiding under mixed scheduling with attention/GDN kernels), per-column
// sequential dequant+FMA chains (the interleaved construction measurably
// loses), fp32 accumulation, one threadgroup-barrier partial reduction.
// MROWS/GS/K_PARTS ride as template ints so MLX caches one specialization
// per shape class. 4-bit affine only (the shipping trunk class); everything
// else falls through to stock qmm. Numerics: fp32 accumulate in a different
// order than stock → bf16 tail-ULP class differences (same accepted class
// as every fused kernel here); parity pinned by the verifyQmm test.
//
// NAX lane (M5-class machines): on "applegpu_g17" GPUs under macOS >= 26.2
// a fixed 16x32x16 tensor-ops tile (MetalPerformancePrimitives matmul2d on
// the per-core matrix units) extends the family to M 8..16 — past the
// plain-SIMD register cliff at M=8 (T(7) rounds measured 636 ms vs 115
// stock on the M4) — covering deep MTP verify widths (seq 8..9), the merged
// MTP-head forward at high accepted counts, lm_head at M 8..16, and small
// decode batches. Third-generation port; keep the attribution chain:
// DFlash (arXiv:2602.06036) → bstnxbt/dflash-mlx verify_qmm.py (Apache-2.0)
// → MTPLX nax_verify.py (Apache-2.0: the M-padding dispatch, env gating,
// availability probes) → us (plan: todo-m5-nax.md). The kernel object is
// NEVER built — not just never dispatched — where the probe is false:
// matmul2d<.., execution_simdgroup> pipeline creation can fail on non-G17
// hardware. diag force-gpu-family-fallback pretends the units are
// absent (QA rehearsal of the exact M1-M4 path on an M5). The takeover width
// is 8 (7 at 8-bit); their dispatcher keeps plain SIMD through M=6, but our
// SIMD lanes differ — measure, don't inherit.
/// Comptime codegen of the per-M split-K kernel body. The row loads, weight
/// loads, and dequant+FMA chains are emitted as NAMED SCALARS with LITERAL
/// accumulator indices — the ledger's load-bearing constraint: array-indexed
/// rows behind a runtime-ish loop stack-spill (measured 10x at M=6 with the
/// `Vec8 v[MROWS]` form of this same kernel). GS and K_PARTS stay template
/// ints; M is baked into the source (one cached kernel per M, distinct Metal
/// host_names — two specializations sharing a name bind the wrong binary).
/// The activation-FMA lines one output column contributes for a single K
/// value: `wv` is the dequantized weight, `idx` the element inside the Vec8
/// activation group. Shared by both plain-SIMD verify tiles so the two cannot
/// drift.
fn vqmmAccLines(comptime m: usize, comptime j: usize, comptime idx: []const u8) []const u8 {
    comptime {
        @setEvalBranchQuota(200_000);
        var out: []const u8 = "";
        for (0..m) |r| {
            out = out ++ std.fmt.comptimePrint("        acc[{d}] += float(v{d}[{s}]) * wv;\n", .{ j * m + r, r, idx });
        }
        return out;
    }
}

/// Hoisted weight WORDS for one column, decoded per width.
///
/// One `pack` iteration always covers 8 K values (that is the Vec8 activation
/// load), so its weights are exactly BITS bytes at
/// `w_q + n * K_bytes + pack * BITS`. The loads stay HOISTED above the FMA
/// chains — the tile's whole shape is "issue every column's weight word, then
/// run one sequential chain per column", and leaving the load inside the chain
/// serializes it behind the previous column's arithmetic. Hoisting only the
/// POINTER is not enough: the widened arms must hoist the BYTES.
///
/// Widest load each width's alignment allows. 4-bit keeps its original aligned
/// uint32 verbatim — that width must stay bit-identical, and it is this
/// refactor's tripwire. 8-bit sits on `pack * 8` and takes two uints (not one
/// ulong: 4-byte alignment is what the shipped 4-bit path already assumes of
/// this buffer, and widening that assumption for one fused load buys nothing).
/// 6-bit lands on `pack * 6`, only 2-aligned, so three ushorts re-assembled
/// into two 3-byte words. 5-bit on `pack * 5` is 1-aligned and takes bytes.
fn vqmmColumnLoad(comptime j: usize) []const u8 {
    comptime {
        @setEvalBranchQuota(200_000);
        return std.fmt.comptimePrint(kernelSource("kernels/linear/vqmm_column_load.fmt.metal"), .{ j, j, j, j, j, j, j, j, j, j, j });
    }
}

/// Per-column dequant + FMA chain, one arm per supported affine width. The
/// byte-addressed arms are lifted from the NAX m16 tile's validated unpack
/// (4 values per 3 bytes at 6 bits, 8 per 5 bytes at 5, one byte per value at
/// 8) — this is not a fresh reading of MLX's packing.
fn vqmmColumnChain(comptime m: usize, comptime j: usize) []const u8 {
    comptime {
        @setEvalBranchQuota(200_000);
        const acc_ki = vqmmAccLines(m, j, "ki");
        const acc_hi = vqmmAccLines(m, j, "4 + ki");
        return std.fmt.comptimePrint(kernelSource("kernels/linear/vqmm_column_chain.fmt.metal"), .{ j, j, j, acc_ki, j, acc_ki, j, acc_ki, j, acc_ki, j, acc_hi });
    }
}

fn verifyQmmSource(comptime m: usize, comptime bn: usize) [:0]const u8 {
    comptime {
        // The per-width unpack ladder multiplies the comptime string work.
        @setEvalBranchQuota(200_000);
        const nacc = bn * m;
        var body: []const u8 = "";
        // Row activation loads, all up front.
        for (0..m) |r| {
            body = body ++ std.fmt.comptimePrint("  Vec8 v{d} = xv[({d} * K + k_base) / 8];\n", .{ r, r });
        }
        // Weight handles for the owned columns, then scales/biases.
        for (0..bn) |j| {
            body = body ++ vqmmColumnLoad(j);
        }
        for (0..bn) |j| {
            body = body ++ std.fmt.comptimePrint("  float s{d} = float(scales[(n0 + {d}) * K_by_gs + gi]); float b{d} = float(biases[(n0 + {d}) * K_by_gs + gi]);\n", .{ j, j, j, j });
        }
        // One sequential dequant+FMA chain per output column (column-major
        // construction; the interleaved form measurably loses).
        for (0..bn) |j| {
            body = body ++ vqmmColumnChain(m, j);
        }
        var acc_init: []const u8 = "";
        for (0..nacc) |i| {
            acc_init = acc_init ++ std.fmt.comptimePrint("acc[{d}] = 0.0f; ", .{i});
        }
        var acc_sum: []const u8 = "";
        for (0..nacc) |i| {
            acc_sum = acc_sum ++ std.fmt.comptimePrint("acc[{d}] = simd_sum(acc[{d}]); ", .{ i, i });
        }
        return std.fmt.comptimePrint(kernelSource("kernels/linear/vqmm.fmt.metal"), .{ bn, nacc, acc_init, body, acc_sum, nacc, nacc, nacc, nacc, nacc, m, m });
    }
}

/// Wide multi-simdgroup tile for HUGE N (the lm_head class): NSG independent
/// simdgroups ride one threadgroup purely to give the scheduler fewer,
/// heavier units — the split-K tiny-tile grid measurably thrashes there
/// (2.1x stock at M=4), while this tile holds 1.18x off the weight-stream
/// floor per the source ledger. Full-K per simdgroup, no barrier.
fn verifyQmmMsgSource(comptime m: usize, comptime bn: usize) [:0]const u8 {
    comptime {
        // The per-width unpack ladder multiplies the comptime string work.
        @setEvalBranchQuota(200_000);
        const nacc = bn * m;
        var body: []const u8 = "";
        for (0..m) |r| {
            body = body ++ std.fmt.comptimePrint("  Vec8 v{d} = xv[({d} * K + k_base) / 8];\n", .{ r, r });
        }
        for (0..bn) |j| {
            body = body ++ vqmmColumnLoad(j);
        }
        for (0..bn) |j| {
            body = body ++ std.fmt.comptimePrint("  float s{d} = float(scales[(n0 + {d}) * K_by_gs + gi]); float b{d} = float(biases[(n0 + {d}) * K_by_gs + gi]);\n", .{ j, j, j, j });
        }
        for (0..bn) |j| {
            body = body ++ vqmmColumnChain(m, j);
        }
        var acc_init: []const u8 = "";
        for (0..nacc) |i| {
            acc_init = acc_init ++ std.fmt.comptimePrint("acc[{d}] = 0.0f; ", .{i});
        }
        var acc_sum: []const u8 = "";
        for (0..nacc) |i| {
            acc_sum = acc_sum ++ std.fmt.comptimePrint("acc[{d}] = simd_sum(acc[{d}]); ", .{ i, i });
        }
        return std.fmt.comptimePrint(kernelSource("kernels/linear/vqmm_msg.fmt.metal"), .{ bn, bn - 1, nacc, acc_init, body, acc_sum, nacc, m, m });
    }
}

/// msg column-tile width per M: 4 columns through M=6, 2 for M=7 (14
/// accumulators — under the 24 ceiling; covers the depth-6 verify lm_head).
fn vqmmMsgBn(m: c_int) c_int {
    return if (m <= 6) 4 else 2;
}

pub const VQMM_MSG_SOURCES = blk: {
    @setEvalBranchQuota(20_000_000);
    break :blk [6][:0]const u8{
        verifyQmmMsgSource(2, 4), verifyQmmMsgSource(3, 4), verifyQmmMsgSource(4, 4),
        verifyQmmMsgSource(5, 4), verifyQmmMsgSource(6, 4), verifyQmmMsgSource(7, 2),
    };
};
const VQMM_MSG_NAMES = [6][*:0]const u8{
    "mlxserve_vqmm_msg_m2", "mlxserve_vqmm_msg_m3", "mlxserve_vqmm_msg_m4",
    "mlxserve_vqmm_msg_m5", "mlxserve_vqmm_msg_m6", "mlxserve_vqmm_msg_m7",
};

var vqmm_msg_kernels: [6]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getVerifyQmmMsgKernel(m: c_int) !mlx.mlx_fast_metal_kernel {
    if (m < 2 or m > 7) return error.UnsupportedShape;
    const idx: usize = @intCast(m - 2);
    if (vqmm_msg_kernels[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        VQMM_MSG_NAMES[idx],
        in_vec,
        out_vec,
        VQMM_MSG_SOURCES[idx],
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_msg_kernels[idx] = kernel;
    return kernel;
}

/// Column-tile width per M. A threadgroup owns `bn` output columns and
/// streams ALL M activation rows over its K partition, so the x traffic is
/// `N/bn * K * M * 2` bytes against `N * K * bits/8` of weight — at M=5,
/// K=6656, N=19968, bn=4 that is 332 MB of activation reads for 66 MB of
/// weights. Widening the tile is the direct lever on that ratio; the ceiling
/// is the accumulator budget (`bn * m` live floats per thread) and N's
/// divisibility by bn. Measured per (M, bn) by the width-sweep µbench.
pub const VQMM_BN_CHOICES = [_]c_int{ 2, 4, 8 };

/// Test seam: the parity test drives every column tile (0 = per-M default).
pub var vqmm_bn_test_override: ?c_int = null;

/// Column tile per M: 4 columns through M=6 (24 accumulators, the measured
/// register ceiling), 2 for M=7 (14) — one deep-verify width past MTPLX's
/// family. M=8 measured a CLIFF at bn=4 (T(7) round 636 ms vs 115 stock —
/// spill/occupancy past 7 live Vec8 row vectors), so 8+ falls back to stock.
///
/// WIDENING THE TILE IS DISPROVEN. The activation-traffic arithmetic above
/// predicts bn=8 should nearly halve the dominant read; measured on the Muse
/// 4-bit trunk shapes it is 5-8% SLOWER at every width (M=5: 45.9 ms vs 43.0
/// whole-forward, same direction at M 2/3/4). The x rows are 66 KB and stay
/// cache-resident, so the "traffic" was never leaving the cache — the extra
/// accumulators just cost occupancy. The ceiling below admits the 8-column tile
/// only under the test seam.
pub const VQMM_MAX_ACC: c_int = 40;

/// Simdgroups per threadgroup, each owning a K partition. Wide-N shapes
/// already emit N/bn threadgroups, so the grid does not need more
/// parallelism there — 2 keeps the reduction cheap, and the sweep agrees:
/// on the Muse trunk shapes at M=5 the whole-forward estimate is 42.4 ms at
/// 2, 42.5 at 1, 44.5 at 4, 49.0 at 8.
fn vqmmKParts(N: c_int) c_int {
    return if (N >= 4096) 2 else 4;
}

pub fn vqmmBnFor(m: c_int, N: c_int) c_int {
    const forced = vqmm_bn_test_override orelse 0;
    if (forced != 0 and @mod(N, forced) == 0 and forced * m <= VQMM_MAX_ACC) return forced;
    const want: c_int = if (m <= 6) 4 else 2;
    return if (@mod(N, want) == 0) want else 0;
}

pub fn vqmmSourceIdx(m: c_int, bn: c_int) usize {
    const mi: usize = @intCast(m - 2);
    const bi: usize = switch (bn) {
        2 => 0,
        4 => 1,
        else => 2,
    };
    return mi * VQMM_BN_CHOICES.len + bi;
}

const VQMM_N_KERNELS = 6 * VQMM_BN_CHOICES.len;

pub const VQMM_SOURCES = blk: {
    // 18 tiles x a four-width unpack ladder is a lot of comptime string work.
    @setEvalBranchQuota(20_000_000);
    var out: [VQMM_N_KERNELS][:0]const u8 = undefined;
    for (2..8) |m| {
        for (VQMM_BN_CHOICES, 0..) |bn, bi| {
            out[(m - 2) * VQMM_BN_CHOICES.len + bi] = verifyQmmSource(m, @intCast(bn));
        }
    }
    break :blk out;
};
const VQMM_NAMES = blk: {
    var out: [VQMM_N_KERNELS][*:0]const u8 = undefined;
    for (2..8) |m| {
        for (VQMM_BN_CHOICES, 0..) |bn, bi| {
            out[(m - 2) * VQMM_BN_CHOICES.len + bi] =
                std.fmt.comptimePrint("mlxserve_vqmm_ks_m{d}_n{d}", .{ m, bn });
        }
    }
    break :blk out;
};

var vqmm_kernels: [VQMM_N_KERNELS]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getVerifyQmmKernel(m: c_int, bn: c_int) !mlx.mlx_fast_metal_kernel {
    if (m < 2 or m > 7) return error.UnsupportedShape;
    const idx = vqmmSourceIdx(m, bn);
    if (vqmm_kernels[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const source: [*:0]const u8 = VQMM_SOURCES[idx];
    const name: [*:0]const u8 = VQMM_NAMES[idx];
    const kernel = mlx.mlx_fast_metal_kernel_new(
        name,
        in_vec,
        out_vec,
        source,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_kernels[idx] = kernel;
    return kernel;
}

/// Process-lifetime cache of 0-d int scalars for the kernel's K/N inputs —
/// only a handful of distinct layer geometries exist per model, and the
/// verify path builds ~300 kernel nodes per round (inference thread only,
/// same single-caller discipline as the kernel caches above).
pub const VqmmScalar = struct { v: c_int, arr: mlx.mlx_array, tick: u64 };
var vqmm_scalar_cache: [16]VqmmScalar = undefined;
var vqmm_scalar_count: usize = 0;
var vqmm_scalar_tick: u64 = 0;

/// Which slot may be evicted: the LEAST RECENTLY USED one, never a fixed slot.
///
/// The old policy always evicted slot 0, and that is a use-after-free. Every
/// caller takes TWO scalars back to back (`K_arr` then `N_arr`) and holds BOTH
/// in one `inputs_arr`. With the cache full, the K call evicts slot 0 and lands
/// there; the N call then frees slot 0 again — which is now K's array — so the
/// caller carried a dangling handle into `mlx_vector_array_new_data`. Observed
/// live on a 40-layer qwen3_5_moe (GatedDeltaNet + MoE + a 248k
/// vocab: enough distinct (K, N) pairs to overflow 16 slots) as SIGBUS building
/// the vector and SIGSEGV freeing it — one request killed the server, and the
/// surfaces probed after it all read as 0%.
///
/// The old comment claimed eviction was safe because "in-flight lazy graph
/// nodes hold their own refs". True, and irrelevant: the array has not reached
/// a graph node yet, it is still sitting in the caller's local array.
///
/// LRU is correct by construction here — the scalars a call site is still
/// holding are by definition the most recently ticked, so they can never be
/// the minimum while the cache (16) is far larger than the most any one site
/// takes (2).
pub fn vqmmScalarEvictIndex(entries: []const VqmmScalar) usize {
    var best: usize = 0;
    for (entries, 0..) |e, i| {
        if (e.tick < entries[best].tick) best = i;
    }
    return best;
}

pub fn cachedScalarInt(v: c_int) mlx.mlx_array {
    vqmm_scalar_tick += 1;
    for (vqmm_scalar_cache[0..vqmm_scalar_count]) |*e| {
        if (e.v == v) {
            e.tick = vqmm_scalar_tick;
            return e.arr;
        }
    }
    const arr = mlx.mlx_array_new_int(v);
    if (vqmm_scalar_count < vqmm_scalar_cache.len) {
        vqmm_scalar_cache[vqmm_scalar_count] = .{ .v = v, .arr = arr, .tick = vqmm_scalar_tick };
        vqmm_scalar_count += 1;
    } else {
        const victim = vqmmScalarEvictIndex(vqmm_scalar_cache[0..vqmm_scalar_count]);
        _ = mlx.mlx_array_free(vqmm_scalar_cache[victim].arr);
        vqmm_scalar_cache[victim] = .{ .v = v, .arr = arr, .tick = vqmm_scalar_tick };
    }
    return arr;
}

pub var verify_qmm_override: ?bool = null;
pub fn verifyQmmEnabled() bool {
    return verify_qmm_override orelse true;
}

pub const VqmmLane = enum { none, splitk, msg, nax };

pub var vqmm_nax_engaged = false;

/// The byte-unpack NAX tile is profitable for wide q5/q6/q8 projections, but
/// loses to stock MLX on the 1024-wide K/V projections. This threshold is the
/// measured split on M5 Max at M=8; q4 keeps its existing full geometry.
/// Qwen3.8-27B 8-bit/g64, forced depth 8, six paired settled boots: 89.81
/// tok/s NAX vs 79.51 stock (+12.96%); every arm asserted its own engagement.
/// Flash Next oQ4e's 5/6-bit 2560-wide and g128 projections lose 2% of the forward
/// at 8 rows, tie at 10 and win 5% at 12 (M5 Ultra), so they ride the tile past 8 rows;
/// 8-bit at N 2560 loses 1-4% at M 7-8.
pub fn mixedNaxShapeEnabled(bits: u32, group_size: u32, n: c_int, m: c_int) bool {
    if (bits == 4) return true;
    const wide = group_size == 64 and n >= 5120;
    if (bits == 8) return wide;
    if (bits != 5 and bits != 6) return false;
    return wide or ((group_size == 64 or group_size == 128) and n >= 2560 and m > 8);
}

/// Whether the PLAIN-SIMD tiles (split-K / msg) adopt a non-4-bit affine
/// weight. Profitability is a SEPARATE question from "the kernel is correct":
/// the byte-addressed unpack does strictly more ALU per value than the 4-bit
/// nibble path, so the shapes where it beats stock differ from the NAX tile's
/// (`mixedNaxShapeEnabled` exists because that tile LOSES on the 1024-wide K/V
/// projections).
///
/// MEASURED, Qwen3.8-27B on an M4 Max (no NAX lane, so the plain-SIMD tiles are
/// all this machine has), MTP on, decode tok/s from the server's own timings,
/// per-prompt medians over two passes with the CELL order reversed on the
/// second (the first cell of a pass reads ~2% fast, which a control pack that
/// ignores the lever entirely made visible):
///
/// | pack | code | prose | agentish |
/// |---|---|---|---|
/// | 6-bit gs64 | **1.10x** | **1.22x** | **1.17x** |
/// | 8-bit gs64 | 1.03x | 1.04x | 1.01x |
///
/// For scale: killing the 4-bit lane outright on the same box costs
/// 1.09-1.12x, so the 6-bit lane is worth MORE than the 4-bit one it was
/// modelled on — stock's 6-bit qmm has more to give back at these widths.
///
/// So 6-bit is default-on and 8-bit is NOT: 0.7-3.7% is inside the noise this
/// harness can resolve, and a lane that only adds dispatch surface for that is
/// not worth defaulting. 5-bit has the kernel arm and the parity test but no
/// pack on this box to measure, so it stays off for the same reason rather
/// than riding 6-bit's number. Both flip with one env var.
///
/// The N floor is where the whole result lives: forcing every shape on
/// (`MIXED_PLAIN=1`, which adopts the 1024-wide K/V projections too) measured
/// only 1.01-1.05x on the same pack. Excluding them is worth ~10 points.
///
/// MEASURED, Flash Next oQ4e on an M5 Ultra (diag `vqmm-fn-ubench`, two
/// runs): 5-bit g128 z/out and 5/6-bit g64 o (N 2560/6144) at 0.79-0.86x stock
/// at M=7, 0.95-0.98x at M 3-5. Shared-expert shapes (N/K 640) lose and stay below the floor.
pub fn mixedPlainShapeEnabled(bits: u32, group_size: u32, n: c_int) bool {
    if (bits == 4) return true;
    if (bits != 5 and bits != 6) return false;
    return (group_size == 64 or group_size == 128) and n >= MIXED_PLAIN_MIN_N;
}

/// The N below which the byte-addressed unpack loses to stock (27B's 1024-wide
/// K/V projections), and the reason force-all-shapes gives back most of the win.
pub const MIXED_PLAIN_MIN_N: c_int = 2560;

var vqmm_mixed_plain_engaged: bool = false;

/// Test seam: the parity test drives the plain-SIMD mixed-width arms without
/// touching the process environment (correctness is not a perf question).
pub var vqmm_mixed_plain_test_force: ?bool = null;
/// Same seam for the NAX tile's mixed-width shape gate.
pub var vqmm_mixed_nax_test_force: ?bool = null;

pub fn mixedPlainEnabled(bits: u32, group_size: u32, n: c_int) bool {
    if (vqmm_mixed_plain_test_force) |v| return v;
    return mixedPlainShapeEnabled(bits, group_size, n);
}

/// Pure lane selection for the verify-qmm family (hermetically pinned by
/// the NAX dispatch-table test). Shared floors: M 2..16, N >= 512 (tiny-N
/// projections — GDN ba proj, MoE routers — gain nothing over stock qmv
/// while the per-call kernel-node build ~10us is real; ~50 such calls ride
/// every verify round). Lanes:
/// - nax: M >= nax_min_m (8; 7 at 8-bit) when
///   the M5-class probe is live AND the m16 tile's stricter geometry holds
///   (K % 256 == 0, N % 32 == 0) — the lm_head N=151936 qualifies natively,
///   so no msg variant is needed past M=7.
/// - splitk/msg: the plain-SIMD M 2..7 lanes (K % 64 == 0, N % 4 == 0),
///   byte-identical to the pre-NAX dispatch; msg takes N >= 100000.
/// - none: stock mlx_quantized_matmul.
pub fn vqmmLaneFor(m: c_int, K: c_int, N: c_int, nax_on: bool, nax_min_m: c_int) VqmmLane {
    return vqmmLaneForTile(m, K, N, if (nax_on) .nax else .off, nax_min_m);
}

/// Which matmul2d tile this machine runs: the M5 matrix units (`nax`, one 16-row tile,
/// M up to 16) or the same kernel as plain shader code on the M4 family (`shader`,
/// 8/16/24-row tiles, M 8..24; a 32-row tile exceeds threadgroup memory and stock
/// qmm is at its own roofline there).
pub const VqmmTile = enum { off, nax, shader };

pub const SHADER_TILE_MAX_M: c_int = 24;

pub fn vqmmTileFrom(nax_available: bool, arch: []const u8, os_ver: []const u8) VqmmTile {
    if (nax_available) return .nax;
    const gpu = naxArchGeneration(arch);
    if (gpu.gen == 16 and !gpu.phone and macosVersionAtLeast(os_ver, 26, 2)) return .shader;
    return .off;
}

/// Tile row height for an M-row call.
pub fn vqmmTileRows(tile: VqmmTile, m: c_int) c_int {
    return if (tile == .shader) @divTrunc(m + 7, 8) * 8 else 16;
}

pub fn vqmmLaneForTile(m: c_int, K: c_int, N: c_int, tile: VqmmTile, nax_min_m: c_int) VqmmLane {
    const max_m: c_int = if (tile == .shader) SHADER_TILE_MAX_M else 16;
    if (m < 2 or m > max_m) return .none;
    if (N < 512) return .none;
    const tile_min_m: c_int = if (tile == .shader) 8 else nax_min_m;
    if (tile != .off and m >= tile_min_m and @mod(K, 256) == 0 and @mod(N, 32) == 0) return .nax;
    if (m > 7) return .none;
    if (@mod(K, 64) != 0 or @mod(N, 4) != 0) return .none;
    return if (N >= 100000) .msg else .splitk;
}

/// Verify-width qmm dispatch (three lanes; see vqmmLaneFor). Returns the
/// [.., M, N] product when the shape is eligible — split-K (M 2..7),
/// msg wide tile (M 2..7 at huge N), or the NAX m16 tile (M 8..16,
/// M5-class machines only) — null to fall through to stock. 4-bit affine,
/// gs in {32,64,128}, bf16/fp16 activations.
pub fn verifyQmm(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
) !?mlx.mlx_array {
    if (!verifyQmmEnabled() or !mlx.streamIsGpu(s)) return null;
    // The plain-SIMD split-K/msg kernels carry byte-addressed 5/6/8-bit unpack,
    // but mixed-width adoption remains restricted to measured width/shape
    // combinations. The M5 NAX tile additionally handles eligible mixed-bit
    // projections at and above its takeover row.
    if (bits != 4 and bits != 5 and bits != 6 and bits != 8) return null;
    if (group_size != 32 and group_size != 64 and group_size != 128) return null;
    if (sc.ctx == null or bi.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return null;
    const K: c_int = xsh[xsh.len - 1];
    var m: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| m *= d;
    const wsh = mlx.getShape(w);
    if (wsh.len != 2) return null;
    const N: c_int = wsh[0];
    // Affine packed geometry sanity: one uint32 column carries 32/bits source
    // values (5/6-bit rows are byte-packed but still exposed as uint32 arrays).
    if (@as(i64, wsh[1]) * 32 != @as(i64, K) * @as(i64, bits)) return null;
    const tile = verifyQmmTile();
    const lane = vqmmLaneForTile(m, K, N, tile, naxMinMForBits(bits));
    // The shader tile is measured at 4-bit only; other widths stay on stock.
    if (lane == .nax and tile == .shader and bits != 4) return null;
    if (bits != 4) switch (lane) {
        .nax => if (!(vqmm_mixed_nax_test_force orelse mixedNaxShapeEnabled(bits, group_size, N, m))) return null,
        // The plain-SIMD tiles now carry the same byte-addressed unpack the
        // NAX tile validated, so 5/6/8-bit no longer fall through to stock on
        // every M1-M4 machine — but adoption is per-shape and measured.
        .splitk, .msg => {
            if (!mixedPlainEnabled(bits, group_size, N)) return null;
            // A lane that silently declines is indistinguishable from one that
            // is merely slow, so every A/B arm has to be readable from its own
            // log rather than from the env it was launched with.
            if (!vqmm_mixed_plain_engaged) {
                vqmm_mixed_plain_engaged = true;
                log.info("[vqmm] plain-SIMD verify lane engaged at {d}-bit: lane={s} M={d} K={d} N={d} gs={d}\n", .{ bits, @tagName(lane), m, K, N, group_size });
            }
        },
        .none => return null,
    };
    switch (lane) {
        .none => return null,
        // NAX m16 tile (M5-class): M 8..16 by default — past the plain-SIMD
        // register cliff. Construction stays strictly behind the probe.
        .nax => return try runVerifyQmmNax(s, x, w, sc, bi, bits, group_size, m, vqmmTileRows(tile, m), K, N, xd, xsh),
        // Huge-N (lm_head class): the tiny-tile split-K grid thrashes the
        // scheduler there (measured 2.1x stock at M=4) — route through the
        // wide multi-simdgroup tile instead (2-column tile at M=7).
        .msg => return try runVerifyQmmMsg(s, x, w, sc, bi, bits, group_size, m, K, N, xd, xsh),
        .splitk => {},
    }

    // x rides in with its original shape — the kernel indexes the buffer
    // linearly, and a contiguous [.., M, K] has the same layout as [M, K]
    // (ensure_row_contiguous copies the rare non-contiguous case).
    const k_parts: c_int = vqmmKParts(N);
    const bn = vqmmBnFor(m, N);
    if (bn == 0) return null;
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);

    const key = VqmmSplitkKey{ .m = m, .n = N, .bn = bn, .k_parts = k_parts, .gs = group_size, .bits = bits, .dtype = xd };
    const config = if (vqmm_splitk_cfgs.get(key)) |hit| hit[0] else blk: {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const y_shape = [_]c_int{ m, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * k_parts, @divExact(N, bn), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * k_parts, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K_PARTS", k_parts));
        vqmm_splitk_cfgs.put(key, .{cfg});
        break :blk cfg;
    };

    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmKernel(m, bn);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));

    // Restore the caller's leading shape with N as the last dim.
    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

/// Every verify width x projection shape x quant width keys its own launch config (~10 FFI calls each to build).
const VqmmSplitkKey = struct { m: c_int, n: c_int, bn: c_int, k_parts: c_int, gs: u32, bits: u32, dtype: mlx.mlx_dtype };
var vqmm_splitk_cfgs = KernelCfgCache(VqmmSplitkKey, 1, 64){};

const VQMM_MSG_NSG: c_int = 8; // simdgroups per threadgroup (their sweep winner)

pub fn runVerifyQmmMsg(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    m: c_int,
    K: c_int,
    N: c_int,
    xd: mlx.mlx_dtype,
    xsh: []const c_int,
) !?mlx.mlx_array {
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ m, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    const cols: c_int = vqmmMsgBn(m) * VQMM_MSG_NSG;
    const tg_count: c_int = @divTrunc(N + cols - 1, cols); // in-kernel n0 guard covers the tail
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * VQMM_MSG_NSG, tg_count, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * VQMM_MSG_NSG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", VQMM_MSG_NSG));

    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmMsgKernel(m);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));

    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}
