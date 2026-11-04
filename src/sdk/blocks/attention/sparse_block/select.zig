//! Exact top-k block selection (`msv_qsa_select`: radix select, single, sliced and merged setups) and its 16-way split
//! at decode.

const std = @import("std");
const kernelSource = @import("../../../../mlx/kernels.zig").source;
const mlx = @import("../../../../mlx/ffi.zig");
const log = @import("../../../util/log.zig");
const QsaCfgCache = @import("../../../../mlx/kernel_cache.zig").QsaCfgCache;
const attn256MaxDiff = @import("../attn256.zig").attn256MaxDiff;
const OneShotBits = @import("one_shot.zig").OneShotBits;
const qsaWidthBucket = @import("one_shot.zig").widthBucket;
const Blocks = @import("params.zig").Blocks;
const ops = @import("../../../../mlx/ops.zig");
const Graph = ops.Graph;
const testing = std.testing;

/// The block geometry the select tests run at (Qwen3.8-Flash-Next's).
const blocks4: Blocks = .{ .size = 4, .top_k = 512 };

// ── Fused QSA top-k block select (msv_qsa_select) ──
//
// The composed arm is ~10 dependent dispatches per attention layer (~0.31 ms/layer at kv
// 62.7k); this kernel is one dispatch: one threadgroup per query row, an MSD radix select
// over the f32 score's monotone ordinal, then an index-ordered compaction. Semantics are
// `qsaChunkSelect`'s: top-K visible blocks per row, ties to the LOWER index (torch.topk),
// emitted ascending, INT_MAX past the row's count. The tie rule is exact where the composed
// arm's `b * 1e-7` bias only approximates it.
pub const QSA_SELECT_KERNEL_HEADER = kernelSource("kernels/attention/qsa_select_header.metal");

/// One threadgroup per query row. 1. MSD radix select over the 32-bit ordinal (the DIGITS
/// template picks 11/11/10-bit digits at prefill widths or four 8-bit digits at decode widths,
/// where each level's fixed cost, not the walk, is the bill) counting only visible blocks,
/// stopping early when a bin's count is exactly what is still needed. 2. Compaction: selected
/// iff `ord > T`, or `ord == T` with rank below `need_eq`; the output slot is the count of
/// selected elements with a smaller index (a simd prefix scan per chunk), so the output is
/// born sorted. 3. Rows with `bounds[row] <= K` are `0 .. bounds-1` then INT_MAX.
/// Threadgroup memory ~8.3 KiB at 2048 bins, ~1.3 KiB at 256.
const QSA_SELECT_SHARED = kernelSource("kernels/attention/qsa_select_shared.metal");

// One radix-select body composed into three kernels (single row, row slice, merge) so a
// fix to the digit walk or the tie rule lands once; the kernels differ only in setup.
const QSA_SELECT_RADIX_BODY = kernelSource("kernels/attention/qsa_select_radix_body.metal");

const QSA_SELECT_SINGLE_SETUP = kernelSource("kernels/attention/qsa_select_single_setup.metal");

const QSA_SELECT_SLICE_SETUP = kernelSource("kernels/attention/qsa_select_slice_setup.metal");

// Candidates arrive ascending by ORIGINAL index with SENTINEL slots skipped, never sorted
// in; that is what resolves a cross-slice tie at the K-th ordinal to the lowest index. Every
// simd reduction below runs in a uniform base-stepped loop: a divergent simd_sum is undefined.
const QSA_SELECT_MERGE_SETUP = kernelSource("kernels/attention/qsa_select_merge_setup.metal");

pub const QSA_SELECT_KERNEL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_SINGLE_SETUP, QSA_SELECT_RADIX_BODY });
pub const QSA_SELECT_SPLIT_LOCAL_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_SLICE_SETUP, QSA_SELECT_RADIX_BODY });
pub const QSA_SELECT_SPLIT_MERGE_SOURCE = std.fmt.comptimePrint("{s}{s}{s}", .{ QSA_SELECT_SHARED, QSA_SELECT_MERGE_SETUP, QSA_SELECT_RADIX_BODY });

/// Threads per row-threadgroup.
pub const QSA_SELECT_TG_DEFAULT: c_int = 1024;
pub var qsa_select_tg_override: ?c_int = null;

// A row's threadgroup is a per-shape choice: the wide passes of a prefill chunk run more
// rows concurrently at 256 threads, a narrow pass wants 512, and the decode widths keep
// 1024 (the split kernels size their own slices). Outputs are identical at every width.
pub fn qsaSelectTgFor(rows: c_int) c_int {
    if (qsa_select_tg_override) |forced| return forced;
    if (rows >= 256) return 256;
    if (rows >= 64) return 512;
    return QSA_SELECT_TG_DEFAULT;
}

var qsa_select_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaSelectKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_select_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "bounds" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_select",
        in_vec,
        out_vec,
        QSA_SELECT_KERNEL_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true, // the row walk indexes linearly; let mlx materialize a view
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_select_kernel_cached = kernel;
    return kernel;
}

/// Keyed on what the config reads: output `[1, rows, kb]`, grid `(tg, rows, 1)`, templates
/// TGS and K. `nb` is a runtime shape (it advances every `ratio` decode tokens).
const QsaSelectCfgKey = struct { rows: c_int, k: c_int, tg: c_int, digits: c_int };
var qsa_select_cfgs = QsaCfgCache(QsaSelectCfgKey, 1){};

var qsa_select_engaged_bits: OneShotBits = .{};

pub var qsa_select_kernel_override: ?bool = null;

/// Fused QSA block select; the test seam's false runs the argpartition chain.
pub fn qsaSelectKernelEnabled() bool {
    return qsa_select_kernel_override orelse true;
}

/// Fused replacement for the composed selection tail: f32 `[1, rows, nb]` scores + int32
/// `[rows]` bounds -> sorted int32 `[1, rows, kb]` selection. Null = declined.
pub fn qsaSelectTopBlocks(
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    bounds: mlx.mlx_array,
    kb: c_int,
) !?mlx.mlx_array {
    if (!qsaSelectKernelEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    if (scores.ctx == null or bounds.ctx == null) return null;
    if (mlx.mlx_array_dtype(scores) != .float32 or mlx.mlx_array_dtype(bounds) != .int32) return null;
    if (mlx.mlx_array_ndim(scores) != 3 or mlx.mlx_array_ndim(bounds) != 1) return null;
    const ssh = mlx.getShape(scores);
    const bsh = mlx.getShape(bounds);
    if (ssh[0] != 1) return null;
    const rows = ssh[1];
    const nb = ssh[2];
    if (rows <= 0 or nb <= 0 or bsh[0] != rows) return null;
    if (kb <= 0 or kb > nb) return null;

    qsa_select_used_split = false;
    if (qsaSelectSplitEnabled() and qsaSelectSplitServes(rows, nb, kb)) {
        if (try qsaSelectTopBlocksSplit(s, scores, bounds, kb, rows, nb)) |picks| {
            qsa_select_used_split = true;
            return picks;
        }
    }

    const kernel = getQsaSelectKernel() catch return null;
    const tg = qsaSelectTgFor(rows);
    const digits = qsaSelectDigitBits(rows);
    const key = QsaSelectCfgKey{ .rows = rows, .k = kb, .tg = tg, .digits = digits };
    const sel_cfgs: [1]mlx.mlx_fast_metal_kernel_config = qsa_select_cfgs.get(key) orelse blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const o_shape = [_]c_int{ 1, rows, kb };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TGS", tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "DIGITS", digits));
        qsa_select_cfgs.put(key, .{config});
        break :blk .{config};
    };

    const inputs_arr = [_]mlx.mlx_array{ scores, bounds };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, sel_cfgs[0], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (qsa_select_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-select] engaged (S={d} nb={d} k={d} tg={d})\n",
            .{ rows, nb, kb, tg },
        );
    }
    return out;
}

// ── The composed selection arm (the kernel's reference) ──
// Free functions so the parity test can run both arms on the same scores.

/// argpartition of the visibility-masked scores; a tiny index-ascending bias reproduces
/// torch.topk's lower-index-wins. A null-ctx `vis3` skips the mask op (all visible).
pub fn qsaTopBlocksOps(s: mlx.mlx_stream, scores: mlx.mlx_array, vis3: mlx.mlx_array, tie_bias: mlx.mlx_array, nb: c_int, block_topk: c_int) !mlx.mlx_array {
    var biased = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(biased);
    try mlx.check(mlx.mlx_subtract(&biased, scores, tie_bias, s));
    if (vis3.ctx == null) {
        var part_all = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_argpartition_axis(&part_all, biased, nb - block_topk, -1, s));
        return part_all;
    }
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try mlx.check(mlx.mlx_where(&masked, vis3, biased, neg_inf, s));
    var part = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_argpartition_axis(&part, masked, nb - block_topk, -1, s));
    return part;
}

/// Sort a `[1, rows, kb]` pick list ascending, mapping rejected picks to INT_MAX first.
pub fn qsaSortWithInvisible(s: mlx.mlx_stream, top_idx: mlx.mlx_array, vis3: mlx.mlx_array, all_vis: bool) !mlx.mlx_array {
    var sorted = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sorted);
    if (all_vis) {
        try mlx.check(mlx.mlx_sort_axis(&sorted, top_idx, -1, s));
        return sorted;
    }
    const int_max = mlx.mlx_array_new_int(std.math.maxInt(i32));
    defer _ = mlx.mlx_array_free(int_max);
    var picked_vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked_vis);
    try mlx.check(mlx.mlx_take_along_axis(&picked_vis, vis3, top_idx, -1, s));
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try mlx.check(mlx.mlx_where(&sel, picked_vis, top_idx, int_max, s));
    try mlx.check(mlx.mlx_sort_axis(&sorted, sel, -1, s));
    return sorted;
}

/// The composed arm the fused kernel replaces, end to end.
pub fn qsaSelectComposedOps(s: mlx.mlx_stream, scores: mlx.mlx_array, vis3: mlx.mlx_array, tie_bias: mlx.mlx_array, rows: c_int, nb: c_int, kb: c_int, block_topk: c_int, all_vis: bool) !mlx.mlx_array {
    const strides3 = [_]c_int{ 1, 1, 1 };
    const part = try qsaTopBlocksOps(s, scores, vis3, tie_bias, nb, block_topk);
    defer _ = mlx.mlx_array_free(part);
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_slice(&tail, part, &[_]c_int{ 0, 0, nb - kb }, 3, &[_]c_int{ 1, rows, nb }, 3, &strides3, 3, s));
    var top_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_idx);
    try mlx.check(mlx.mlx_astype(&top_idx, tail, .int32, s));
    return qsaSortWithInvisible(s, top_idx, vis3, all_vis);
}

// ── Picks under any `Blocks` ──

/// f32 `[nb]` `b * 1e-7`: the index-ascending bias the composed chain breaks ties with (torch.topk's lower index).
pub fn tieBias(s: mlx.mlx_stream, nb: c_int) !mlx.mlx_array {
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_arange(&idx, 0, @floatFromInt(nb), 1.0, .float32, s));
    const eps = mlx.mlx_array_new_float(1e-7);
    defer _ = mlx.mlx_array_free(eps);
    var bias = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&bias, idx, eps, s));
    return bias;
}

/// Picks `[1, lists * rows, min(top_k, nb)]` int32 of score rows `[1, lists * rows, nb]` f32 (list-major; row r of a
/// list is the query at `row0 + r`): forced blocks win, ties go to the lower index, each row ascending with INT_MAX
/// past its count, and a pick of the query's incomplete block dropped (its keys are the tail).
pub fn selectPicks(a: std.mem.Allocator, s: mlx.mlx_stream, b: Blocks, scores: mlx.mlx_array, row0: c_int, rows: c_int) !mlx.mlx_array {
    var g = Graph.init(a, s);
    defer g.deinit();
    const sh = ops.shape(scores);
    const n = sh[1];
    const nb = sh[2];
    const lists = @divExact(n, rows);
    const kb = @min(@as(c_int, @intCast(b.top_k)), nb);
    const per: usize = @intCast(rows);
    const bounds_host = try a.alloc(i32, @intCast(n));
    defer a.free(bounds_host);
    b.selectBounds(bounds_host[0..per], row0, nb, false);
    for (1..@intCast(lists)) |l| @memcpy(bounds_host[l * per ..][0..per], bounds_host[0..per]);
    const bounds = try g.fromSlice(i32, bounds_host, &.{n});
    const pos = try g.reshape(try g.arange(@floatFromInt(row0), @floatFromInt(row0 + rows), 1, .int32), &.{ 1, rows, 1 });
    const col = try g.reshape(try g.arange(0, @floatFromInt(nb), 1, .int32), &.{ 1, 1, nb });
    const size = try g.adopt(mlx.mlx_array_new_int(@intCast(b.size)));
    var sc = scores;
    if (b.init > 0 or b.local > 0) {
        const own = try g.call(mlx.mlx_floor_divide, .{ pos, size });
        const after = try g.greater(col, try g.sub(own, try g.adopt(mlx.mlx_array_new_int(@intCast(b.local)))));
        const local = try g.call(mlx.mlx_logical_and, .{ try g.call(mlx.mlx_less_equal, .{ col, own }), after });
        const forced = try g.call(mlx.mlx_logical_or, .{ local, try g.less(col, try g.adopt(mlx.mlx_array_new_int(@intCast(b.init)))) });
        const by_list = try g.reshape(scores, &.{ lists, rows, nb });
        sc = try g.reshape(try g.where(forced, try g.scalar(std.math.inf(f32), .float32), by_list), &.{ 1, n, nb });
    }
    var picks = if (try qsaSelectTopBlocks(s, sc, bounds, kb)) |k| try g.adopt(k) else blk: {
        const vis = try g.less(col, try g.reshape(bounds, &.{ 1, n, 1 }));
        const tie = try g.adopt(try tieBias(s, nb));
        break :blk try g.adopt(try qsaSelectComposedOps(s, sc, vis, tie, n, nb, kb, kb, false));
    };
    if (b.local > 0) {
        const complete = try g.call(mlx.mlx_floor_divide, .{ try g.add(pos, try g.adopt(mlx.mlx_array_new_int(1))), size });
        const by_list = try g.reshape(picks, &.{ lists, rows, kb });
        const tail = try g.call(mlx.mlx_greater_equal, .{ by_list, complete });
        picks = try g.reshape(try g.where(tail, try g.adopt(mlx.mlx_array_new_int(std.math.maxInt(i32))), by_list), &.{ 1, n, kb });
    }
    return g.escape(picks);
}

// One threadgroup per row is latency-bound at decode widths (0.72 ms at 215k blocks; ~20x
// under the bandwidth floor), so rows split over G slices, each running the exact select,
// and a second dispatch selects over the G*K candidates. Exact because the order is total
// (ordinal, then lowest original index): every global top-K element is in its slice's
// top-K. The floor is the crossover of the marginal in-graph cost (probe: the split pair
// beats the single kernel from ~24k blocks once the compaction scan and 8-bit digits landed;
// live, +4% at 40k blocks and +18% serial at 200k); 2*G*K <= nb (16k blocks at G=16) keeps
// the candidate list smaller than the row it replaces and is the design's hard minimum.
pub const QSA_SELECT_SPLIT_MIN_NB: c_int = 24576;
pub const QSA_SELECT_SPLIT_MAX_ROWS: c_int = 15;
pub const QSA_SELECT_SPLIT_G: c_int = 16;

pub var qsa_select_digits_override: ?c_int = null;

pub fn qsaSelectDigitBits(rows: c_int) c_int {
    if (qsa_select_digits_override) |v| return v;
    if (rows <= QSA_SELECT_SPLIT_MAX_ROWS) return 8;
    return 11;
}

pub var qsa_select_split_override: ?bool = null;
pub var qsa_select_used_split: bool = false;

pub fn qsaSelectSplitEnabled() bool {
    return qsa_select_split_override orelse true;
}

pub fn qsaSelectSplitServes(rows: c_int, nb: c_int, kb: c_int) bool {
    if (rows < 1 or rows > QSA_SELECT_SPLIT_MAX_ROWS) return false;
    if (nb < QSA_SELECT_SPLIT_MIN_NB) return false;
    if (kb <= 0 or kb > nb) return false;
    const span = std.math.mul(c_int, kb, QSA_SELECT_SPLIT_G) catch return false;
    const twice = std.math.mul(c_int, span, 2) catch return false;
    return twice <= nb;
}

var qsa_select_split_local_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
var qsa_select_split_merge_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getQsaSelectSplitLocalKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_select_split_local_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "bounds" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_select_split_local",
        in_vec,
        out_vec,
        QSA_SELECT_SPLIT_LOCAL_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_select_split_local_kernel_cached = kernel;
    return kernel;
}

fn getQsaSelectSplitMergeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_select_split_merge_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "scores", "local_ids" };
    const output_names = [_][*:0]const u8{"ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_select_split_merge",
        in_vec,
        out_vec,
        QSA_SELECT_SPLIT_MERGE_SOURCE,
        QSA_SELECT_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_select_split_merge_kernel_cached = kernel;
    return kernel;
}

const QsaSelectSplitCfgKey = struct { rows: c_int, k: c_int, slice_tg: c_int, merge_tg: c_int, digits: c_int };
var qsa_select_split_cfgs = QsaCfgCache(QsaSelectSplitCfgKey, 2){};

var qsa_select_split_engaged_bits: OneShotBits = .{};

fn qsaSelectTopBlocksSplit(
    s: mlx.mlx_stream,
    scores: mlx.mlx_array,
    bounds: mlx.mlx_array,
    kb: c_int,
    rows: c_int,
    nb: c_int,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const g = QSA_SELECT_SPLIT_G;
    const loc_count = std.math.mul(c_int, kb, g) catch return null;
    const local_kernel = getQsaSelectSplitLocalKernel() catch return null;
    const merge_kernel = getQsaSelectSplitMergeKernel() catch return null;
    const slice_tg: c_int = qsa_select_tg_override orelse 512;
    const merge_tg: c_int = qsa_select_tg_override orelse QSA_SELECT_TG_DEFAULT;
    const digits = qsaSelectDigitBits(rows);
    const key = QsaSelectSplitCfgKey{ .rows = rows, .k = kb, .slice_tg = slice_tg, .merge_tg = merge_tg, .digits = digits };
    const cfgs: [2]mlx.mlx_fast_metal_kernel_config = qsa_select_split_cfgs.get(key) orelse blk: {
        const local_cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(local_cfg);
        const local_shape = [_]c_int{ 1, rows, loc_count };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(local_cfg, &local_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(local_cfg, slice_tg * g, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(local_cfg, slice_tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "TGS", slice_tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(local_cfg, "DIGITS", digits));
        const merge_cfg = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(merge_cfg);
        const o_shape = [_]c_int{ 1, rows, kb };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(merge_cfg, &o_shape, 3, .int32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(merge_cfg, merge_tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(merge_cfg, merge_tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "TGS", merge_tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "K", kb));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(merge_cfg, "DIGITS", digits));
        qsa_select_split_cfgs.put(key, .{ local_cfg, merge_cfg });
        break :blk .{ local_cfg, merge_cfg };
    };

    const local_in = [_]mlx.mlx_array{ scores, bounds };
    const local_in_vec = mlx.mlx_vector_array_new_data(&local_in, local_in.len);
    defer _ = mlx.mlx_vector_array_free(local_in_vec);
    var local_out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(local_out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&local_out_vec, local_kernel, local_in_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(local_out_vec) != 1) return error.MetalKernelBadOutputCount;
    var local_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(local_ids);
    try mlx.check(mlx.mlx_vector_array_get(&local_ids, local_out_vec, 0));

    const merge_in = [_]mlx.mlx_array{ scores, local_ids };
    const merge_in_vec = mlx.mlx_vector_array_new_data(&merge_in, merge_in.len);
    defer _ = mlx.mlx_vector_array_free(merge_in_vec);
    var merge_out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(merge_out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&merge_out_vec, merge_kernel, merge_in_vec, cfgs[1], s));
    if (mlx.mlx_vector_array_size(merge_out_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, merge_out_vec, 0));
    if (qsa_select_split_engaged_bits.take(qsaWidthBucket(rows))) {
        log.info(
            "[qsa-select] engaged (split G={d} S={d} nb={d} k={d} tg={d})\n",
            .{ g, rows, nb, kb, slice_tg },
        );
    }
    return out;
}

// ── tests ──

pub fn qsaSelectCfgBuilds() usize {
    return qsa_select_cfgs.builds;
}

pub fn qsaSelectSplitCfgBuilds() usize {
    return qsa_select_split_cfgs.builds;
}

// ── Fused QSA block select (msv_qsa_select) ──

/// The `[1, rows, nb]` bool visibility sheet the composed arm masks with, free of a Transformer.
pub fn qsaVisSheetForTest(s: mlx.mlx_stream, row0: c_int, rows: c_int, nb: c_int, ratio: c_int) !mlx.mlx_array {
    var blk_end = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_end);
    try mlx.check(mlx.mlx_arange(&blk_end, @floatFromInt(ratio - 1), @floatFromInt(nb * ratio), @floatFromInt(ratio), .int32, s));
    var blk_end2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk_end2);
    try mlx.check(mlx.mlx_reshape(&blk_end2, blk_end, &[_]c_int{ 1, nb }, 2, s));
    var pos = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos);
    try mlx.check(mlx.mlx_arange(&pos, @floatFromInt(row0), @floatFromInt(row0 + rows), 1.0, .int32, s));
    var pos2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pos2);
    try mlx.check(mlx.mlx_reshape(&pos2, pos, &[_]c_int{ rows, 1 }, 2, s));
    var vis = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vis);
    try mlx.check(mlx.mlx_less_equal(&vis, blk_end2, pos2, s));
    var vis3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_expand_dims(&vis3, vis, 0, s));
    return vis3;
}

/// The reference both arms are measured against: per row, the top-`k` visible blocks by
/// (score descending, index ascending), emitted ascending with INT_MAX past the row's count.
fn qsaExactTopHost(alloc: std.mem.Allocator, scores: []const f32, rows: c_int, nb: c_int, bounds: []const i32, k: c_int) ![]i32 {
    const nbu: usize = @intCast(nb);
    const ku: usize = @intCast(k);
    const out = try alloc.alloc(i32, @as(usize, @intCast(rows)) * ku);
    errdefer alloc.free(out);
    const Cand = struct { v: f32, i: i32 };
    const cand = try alloc.alloc(Cand, nbu);
    defer alloc.free(cand);
    const pick = try alloc.alloc(i32, ku);
    defer alloc.free(pick);
    const Lt = struct {
        fn lt(_: void, a: Cand, b: Cand) bool {
            if (a.v != b.v) return a.v > b.v;
            return a.i < b.i;
        }
    };
    for (0..@intCast(rows)) |r| {
        const vb: usize = @intCast(@max(bounds[r], 0));
        for (0..vb) |i| cand[i] = .{ .v = scores[r * nbu + i], .i = @intCast(i) };
        std.sort.pdq(Cand, cand[0..vb], {}, Lt.lt);
        const take = @min(vb, ku);
        for (0..take) |i| pick[i] = cand[i].i;
        std.sort.pdq(i32, pick[0..take], {}, std.sort.asc(i32));
        for (0..take) |i| out[r * ku + i] = pick[i];
        for (take..ku) |i| out[r * ku + i] = std.math.maxInt(i32);
    }
    return out;
}

fn qsaExpectMatchesHost(got: mlx.mlx_array, want: []const i32, rows: c_int, k: c_int, who: []const u8) !void {
    const gs = mlx.getShape(got);
    try testing.expectEqual(@as(usize, 3), gs.len);
    try testing.expectEqual(@as(c_int, 1), gs[0]);
    try testing.expectEqual(rows, gs[1]);
    try testing.expectEqual(k, gs[2]);
    try testing.expectEqual(mlx.mlx_dtype.int32, mlx.mlx_array_dtype(got));
    try mlx.check(mlx.mlx_array_eval(got));
    const b = mlx.mlx_array_data_int32(got).?;
    const ku: usize = @intCast(k);
    for (want, 0..) |w, i| {
        if (w != b[i]) {
            std.debug.print(
                "[qsa-select] {s} differs from the exact reference at row {d} slot {d}: reference {d} got {d}\n",
                .{ who, i / ku, i % ku, w, b[i] },
            );
            return error.QsaSelectMismatch;
        }
    }
}

test "qsa select kernel: the fused radix-select IS the exact top-k, and so is the chain where its bias can resolve" {
    // Bit equality against `qsaExactTopHost` across the nb the arch reaches (700..262144), the
    // query widths it runs at and three visibility regimes. Scores are exact binary fractions
    // k/16 in [0, 0.5] plus forced zeros: the regime where the chain's bias is also exact, so
    // both arms can be held to the same bar.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const prev_split = qsa_select_split_override;
    defer qsa_select_split_override = prev_split;
    qsa_select_split_override = false;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const ratio: c_int = 4;
    const k: c_int = 512;

    const cases = [_]struct { nb: c_int, rows: c_int }{
        .{ .nb = 700, .rows = 1 },
        .{ .nb = 700, .rows = 6 },
        .{ .nb = 700, .rows = 15 },
        .{ .nb = 700, .rows = 256 },
        .{ .nb = 2048, .rows = 1 },
        .{ .nb = 2048, .rows = 15 },
        .{ .nb = 2048, .rows = 256 },
        .{ .nb = 15675, .rows = 1 },
        .{ .nb = 15675, .rows = 6 },
        .{ .nb = 15675, .rows = 15 },
        .{ .nb = 15675, .rows = 256 },
        .{ .nb = 65536, .rows = 1 },
        .{ .nb = 65536, .rows = 15 },
        .{ .nb = 65536, .rows = 64 },
        .{ .nb = 262144, .rows = 1 },
        .{ .nb = 262144, .rows = 6 },
    };
    var prng = std.Random.DefaultPrng.init(0x9A50_C711);
    const rnd = prng.random();

    for (cases) |c| {
        const nbu: usize = @intCast(c.nb);
        const rowsu: usize = @intCast(c.rows);
        const buf = try ta.alloc(f32, rowsu * nbu);
        defer ta.free(buf);
        for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0)
            0.0
        else
            @as(f32, @floatFromInt(rnd.uintLessThan(u32, 9))) / 16.0;
        const sc_shape = [_]c_int{ 1, c.rows, c.nb };
        const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(scores);

        const row0s = [_]c_int{ 0, 2044, 2000, c.nb * ratio };
        for (row0s) |row0| {
            const all_vis = blocks4.allSee(row0, c.nb);
            const bounds_host = try ta.alloc(i32, rowsu);
            defer ta.free(bounds_host);
            blocks4.selectBounds(bounds_host, row0, c.nb, all_vis);
            const want = try qsaExactTopHost(ta, buf, c.rows, c.nb, bounds_host, k);
            defer ta.free(want);

            const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{c.rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bounds);
            const got = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(got);

            var vis3 = mlx.mlx_array{ .ctx = null };
            defer if (vis3.ctx != null) {
                _ = mlx.mlx_array_free(vis3);
            };
            if (!all_vis) vis3 = try qsaVisSheetForTest(s, row0, c.rows, c.nb, ratio);
            const bias = try tieBias(s, c.nb);
            defer _ = mlx.mlx_array_free(bias);
            const chain = try qsaSelectComposedOps(s, scores, vis3, bias, c.rows, c.nb, k, k, all_vis);
            defer _ = mlx.mlx_array_free(chain);

            qsaExpectMatchesHost(got, want, c.rows, k, "the fused kernel") catch |e| {
                std.debug.print("[qsa-select] case nb={d} rows={d} row0={d} all_vis={}\n", .{ c.nb, c.rows, row0, all_vis });
                return e;
            };
            qsaExpectMatchesHost(chain, want, c.rows, k, "the composed chain") catch |e| {
                std.debug.print("[qsa-select] case nb={d} rows={d} row0={d} all_vis={}\n", .{ c.nb, c.rows, row0, all_vis });
                return e;
            };
        }
    }
}

test "qsa select kernel: past the bias's f32 resolution the chain's tie rule is UNSPECIFIED and the kernel's is not" {
    // The chain's `b * 1e-7` bias only survives rounding below |v| ~= 0.84 (one f32 ulp is
    // v * 2^-23); above it ties fall back to argpartition's arbitrary order. A relu sum over
    // four heads sits far above that, so this is the production regime.
    const eps: f32 = 1e-7;
    // Distinct biased values the chain can see across a 2048-block row: 4 at production
    // magnitude, every block at |v| <= 0.5. Pure f32 arithmetic.
    const Distinct = struct {
        fn count(v: f32, n: u32) usize {
            var seen: usize = 0;
            var prev: f32 = 0;
            var b: u32 = 0;
            while (b < n) : (b += 1) {
                const cur = v - @as(f32, @floatFromInt(b)) * eps;
                if (b == 0 or cur != prev) seen += 1;
                prev = cur;
            }
            return seen;
        }
    };
    try testing.expectEqual(@as(usize, 4), Distinct.count(625.0, 2048));
    try testing.expectEqual(@as(usize, 2048), Distinct.count(0.5, 2048));
    const big: f32 = 625.0;
    try testing.expectEqual(big - 1369 * eps, big - 1370 * eps);

    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const ratio: c_int = 4;
    const nb: c_int = 2048;
    const k: c_int = 512;
    const rows: c_int = 16;

    const nbu: usize = @intCast(nb);
    const buf = try ta.alloc(f32, @as(usize, @intCast(rows)) * nbu);
    defer ta.free(buf);
    var prng = std.Random.DefaultPrng.init(0x9A50_C711);
    const rnd = prng.random();
    for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0)
        0.0
    else
        @floatFromInt(rnd.uintLessThan(u32, 1000));

    const sc_shape = [_]c_int{ 1, rows, nb };
    const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const row0: c_int = nb * ratio;
    try testing.expect(blocks4.allSee(row0, nb));

    const bounds_host = try ta.alloc(i32, @intCast(rows));
    defer ta.free(bounds_host);
    @memset(bounds_host, nb);
    const want = try qsaExactTopHost(ta, buf, rows, nb, bounds_host, k);
    defer ta.free(want);
    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);

    // The kernel is the reference at production magnitude.
    const got = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(got);
    try qsaExpectMatchesHost(got, want, rows, k, "the fused kernel");

    // The chain may only ever be wrong about a tie: the bar is the selected score multiset per row.
    const none = mlx.mlx_array{ .ctx = null };
    const bias = try tieBias(s, nb);
    defer _ = mlx.mlx_array_free(bias);
    const chain = try qsaSelectComposedOps(s, scores, none, bias, rows, nb, k, k, true);
    defer _ = mlx.mlx_array_free(chain);
    try mlx.check(mlx.mlx_array_eval(chain));
    const ch = mlx.mlx_array_data_int32(chain).?;
    const ku: usize = @intCast(k);
    const sw = try ta.alloc(f32, ku);
    defer ta.free(sw);
    const sc = try ta.alloc(f32, ku);
    defer ta.free(sc);
    var diffs: usize = 0;
    for (0..@intCast(rows)) |r| {
        for (0..ku) |i| {
            sw[i] = buf[r * nbu + @as(usize, @intCast(want[r * ku + i]))];
            sc[i] = buf[r * nbu + @as(usize, @intCast(ch[r * ku + i]))];
            if (want[r * ku + i] != ch[r * ku + i]) diffs += 1;
        }
        std.sort.pdq(f32, sw, {}, std.sort.asc(f32));
        std.sort.pdq(f32, sc, {}, std.sort.asc(f32));
        try testing.expectEqualSlices(f32, sw, sc);
    }
    if (diffs > 0) {
        std.debug.print("[qsa-select] the chain differs from the exact rule in {d}/{d} slots at production magnitude (tied blocks only)\n", .{ diffs, want.len });
    }
}

test "qsa select kernel: on a near-tie the exact rule and the bias trick DISAGREE, and the kernel takes the exact one" {
    // The bias is exact only on exact ties: at nb = 262144 it spans 0.026, so a genuinely higher
    // score loses to a lower-indexed rival. torch.topk and the kernel take the higher score.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const nb: c_int = 262144;
    const k: c_int = 512;
    const ratio: c_int = 4;
    const winner: usize = 200000;

    const buf = try ta.alloc(f32, @intCast(nb));
    defer ta.free(buf);
    @memset(buf, 0.0);
    for (1..512) |i| buf[i] = 1000.0;
    buf[0] = 10.0; //          biased: 10.000  (bias 0)
    buf[winner] = 10.005; //   biased:  9.985  (bias 0.02) — the bias flips it
    const sc_shape = [_]c_int{ 1, 1, nb };
    const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(scores);

    const row0: c_int = nb * ratio; // all-visible
    try testing.expect(blocks4.allSee(row0, nb));
    const none = mlx.mlx_array{ .ctx = null };
    const bias = try tieBias(s, nb);
    defer _ = mlx.mlx_array_free(bias);
    const chain = try qsaSelectComposedOps(s, scores, none, bias, 1, nb, k, k, true);
    defer _ = mlx.mlx_array_free(chain);

    var bounds_host = [_]i32{nb};
    const bounds = mlx.mlx_array_new_data(&bounds_host, &[_]c_int{1}, 1, .int32);
    defer _ = mlx.mlx_array_free(bounds);
    const fused = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
    defer _ = mlx.mlx_array_free(fused);

    try mlx.check(mlx.mlx_array_eval(chain));
    try mlx.check(mlx.mlx_array_eval(fused));
    const ch = mlx.mlx_array_data_int32(chain).?;
    const fu = mlx.mlx_array_data_int32(fused).?;
    const H = struct {
        fn has(row: [*]const i32, n: usize, v: i32) bool {
            for (0..n) |i| if (row[i] == v) return true;
            return false;
        }
    };
    const ku: usize = @intCast(k);
    try testing.expect(H.has(fu, ku, @intCast(winner)));
    try testing.expect(!H.has(fu, ku, 0));
    try testing.expect(H.has(ch, ku, 0));
    try testing.expect(!H.has(ch, ku, @intCast(winner)));
    for (1..ku) |i| try testing.expect(fu[i] > fu[i - 1]);
}

test "qsa select kernel: the config is cached per FULL shape and rebuilt when rows or nb change" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const k: c_int = 8;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int, kk: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, kk)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    // Normalize first: one global cache shared with every test in the file.
    try Run.once(s, ta, 4, 64, k);
    const base = qsaSelectCfgBuilds();
    for (0..3) |_| try Run.once(s, ta, 4, 64, k);
    try testing.expectEqual(base, qsaSelectCfgBuilds());
    // `nb` is a runtime shape, not a config input.
    try Run.once(s, ta, 4, 128, k);
    try Run.once(s, ta, 4, 517, k);
    try Run.once(s, ta, 4, 1027, k);
    try testing.expectEqual(base, qsaSelectCfgBuilds());
    try Run.once(s, ta, 9, 64, k);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
    try Run.once(s, ta, 4, 64, k);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
    try testing.expect(@TypeOf(qsa_select_cfgs).SLOTS == 4);
}

fn qsaIdsBitEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !bool {
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);
    try mlx.check(mlx.mlx_array_equal(&res, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(res));
    var v: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, res));
    return v;
}

fn qsaSelectMakeSheet(kind: u8, buf: []f32, rnd: std.Random) void {
    switch (kind) {
        0 => {
            for (buf) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
        },
        1 => {
            for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 5) == 0) rnd.float(f32) else 0.0;
        },
        2 => {
            @memset(buf, 0.0);
            const n = buf.len;
            const span = @min(n, 1024);
            for (0..span) |i| buf[i] = 5.0;
        },
        3 => {
            for (buf) |*v| v.* = rnd.float(f32);
            buf[0] = std.math.nan(f32);
            if (buf.len > 1) buf[1] = -0.0;
            if (buf.len > 2) buf[2] = 0.0;
        },
        else => {
            for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0) 0.0 else @as(f32, @floatFromInt(rnd.uintLessThan(u32, 9))) / 16.0;
        },
    }
}

test "qsa select split: ids are identical to the single-threadgroup kernel" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const k: c_int = 512;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = false;

    const nbs = [_]c_int{ 2048, 15675, 65536, 215000, 262144 };
    const row_counts = [_]c_int{ 1, 2, 4, 7, 15, 16 };
    var prng = std.Random.DefaultPrng.init(0x51E7_0A11);
    const rnd = prng.random();

    for (nbs) |nb| {
        for (row_counts) |rows| {
            if (nb >= 215000 and rows >= 16) continue;
            const nbu: usize = @intCast(nb);
            const rowsu: usize = @intCast(rows);
            const buf = try ta.alloc(f32, rowsu * nbu);
            defer ta.free(buf);
            var kind: u8 = 0;
            while (kind < 5) : (kind += 1) {
                qsaSelectMakeSheet(kind, buf, rnd);
                const sc_shape = [_]c_int{ 1, rows, nb };
                const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
                defer _ = mlx.mlx_array_free(scores);
                const bound_kinds = [_]i32{ nb, @divTrunc(nb, 2), 10, 0 };
                for (bound_kinds) |bv| {
                    const bounds_host = try ta.alloc(i32, rowsu);
                    defer ta.free(bounds_host);
                    var r: usize = 0;
                    while (r < rowsu) : (r += 1) {
                        const adj = bv + @as(i32, @intCast(r % 3)) - 1;
                        bounds_host[r] = @max(0, @min(nb, adj));
                    }
                    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
                    defer _ = mlx.mlx_array_free(bounds);
                    const single = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(single);
                    const split = (try qsaSelectTopBlocksSplit(s, scores, bounds, k, rows, nb)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(split);
                    if (!try qsaIdsBitEqual(s, single, split)) {
                        std.debug.print("[qsa-select-split] mismatch nb={d} rows={d} kind={d} bound={d}\n", .{ nb, rows, kind, bv });
                        return error.QsaSelectSplitMismatch;
                    }
                }
            }
        }
    }

    {
        const nb: c_int = 65536;
        const rows: c_int = 1;
        const kk: c_int = 511;
        const buf = try ta.alloc(f32, @intCast(nb));
        defer ta.free(buf);
        qsaSelectMakeSheet(1, buf, rnd);
        const scores = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, rows, nb }, 3, .float32);
        defer _ = mlx.mlx_array_free(scores);
        var bh = [_]i32{nb};
        const bounds = mlx.mlx_array_new_data(&bh, &[_]c_int{rows}, 1, .int32);
        defer _ = mlx.mlx_array_free(bounds);
        const single = (try qsaSelectTopBlocks(s, scores, bounds, kk)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(single);
        const split = (try qsaSelectTopBlocksSplit(s, scores, bounds, kk, rows, nb)) orelse return error.SelectKernelDeclined;
        defer _ = mlx.mlx_array_free(split);
        if (!try qsaIdsBitEqual(s, single, split)) return error.QsaSelectSplitMismatch;
    }
}

test "qsa select split: declines above the width floor and below the nb floor" {
    const k: c_int = 512;
    try testing.expect(!qsaSelectSplitServes(16, 215000, k));
    try testing.expect(!qsaSelectSplitServes(32, 262144, k));
    try testing.expect(!qsaSelectSplitServes(1, 24575, k));
    try testing.expect(qsaSelectSplitServes(1, 24576, k));
    try testing.expect(qsaSelectSplitServes(1, 32768, k));
    try testing.expect(!qsaSelectSplitServes(15, 2048, k));
    try testing.expect(!qsaSelectSplitServes(0, 65536, k));
    try testing.expect(!qsaSelectSplitServes(-1, 65536, k));
    try testing.expect(!qsaSelectSplitServes(1, 65536, 0));
    try testing.expect(!qsaSelectSplitServes(1, 65536, 4096));
    try testing.expect(qsaSelectSplitServes(1, 65536, k));
    try testing.expect(qsaSelectSplitServes(15, 65536, k));
    try testing.expect(qsaSelectSplitServes(7, 215000, k));

    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = true;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, 8)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    try Run.once(s, ta, 16, 65536);
    try testing.expect(!qsa_select_used_split);
    try Run.once(s, ta, 1, 2048);
    try testing.expect(!qsa_select_used_split);
    try Run.once(s, ta, 1, 24576);
    try testing.expect(qsa_select_used_split);
}

test "qsa select split: the fused radix-select IS the exact top-k" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const ratio: c_int = 4;
    const k: c_int = 512;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = true;

    const cases = [_]struct { nb: c_int, rows: c_int }{
        .{ .nb = 700, .rows = 1 },
        .{ .nb = 700, .rows = 6 },
        .{ .nb = 2048, .rows = 1 },
        .{ .nb = 2048, .rows = 15 },
        .{ .nb = 15675, .rows = 1 },
        .{ .nb = 15675, .rows = 6 },
        .{ .nb = 65536, .rows = 1 },
        .{ .nb = 65536, .rows = 15 },
        .{ .nb = 262144, .rows = 1 },
        .{ .nb = 262144, .rows = 6 },
    };
    var prng = std.Random.DefaultPrng.init(0x9A50_C711);
    const rnd = prng.random();

    for (cases) |c| {
        const nbu: usize = @intCast(c.nb);
        const rowsu: usize = @intCast(c.rows);
        const buf = try ta.alloc(f32, rowsu * nbu);
        defer ta.free(buf);
        for (buf) |*v| v.* = if (rnd.uintLessThan(u8, 3) == 0)
            0.0
        else
            @as(f32, @floatFromInt(rnd.uintLessThan(u32, 9))) / 16.0;
        const sc_shape = [_]c_int{ 1, c.rows, c.nb };
        const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
        defer _ = mlx.mlx_array_free(scores);

        const row0s = [_]c_int{ 0, 2044, 2000, c.nb * ratio };
        for (row0s) |row0| {
            const all_vis = blocks4.allSee(row0, c.nb);
            const bounds_host = try ta.alloc(i32, rowsu);
            defer ta.free(bounds_host);
            blocks4.selectBounds(bounds_host, row0, c.nb, all_vis);
            const want = try qsaExactTopHost(ta, buf, c.rows, c.nb, bounds_host, k);
            defer ta.free(want);

            const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{c.rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bounds);
            const got = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(got);
            qsaExpectMatchesHost(got, want, c.rows, k, "split-forced fused kernel") catch |e| {
                std.debug.print("[qsa-select] split-forced case nb={d} rows={d} row0={d} all_vis={}\n", .{ c.nb, c.rows, row0, all_vis });
                return e;
            };
        }
    }
}

test "qsa select split: kill switch off uses the single-threadgroup kernel" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, 512)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    qsa_select_split_override = false;
    try Run.once(s, ta, 1, 65536);
    try testing.expect(!qsa_select_used_split);

    qsa_select_split_override = true;
    try Run.once(s, ta, 1, 65536);
    try testing.expect(qsa_select_used_split);
}

test "qsa select split: the config is cached per FULL shape and rebuilt when rows change" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev = qsa_select_split_override;
    defer qsa_select_split_override = prev;
    qsa_select_split_override = true;
    const k: c_int = 512;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, rows: c_int, nb: c_int, kk: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(rows * nb));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, rows, nb };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(rows));
            defer alloc.free(bh);
            @memset(bh, nb);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{rows}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, kk)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    try Run.once(s, ta, 1, 65536, k);
    try testing.expect(qsa_select_used_split);
    const base = qsaSelectSplitCfgBuilds();
    for (0..3) |_| try Run.once(s, ta, 1, 65536, k);
    try testing.expectEqual(base, qsaSelectSplitCfgBuilds());
    try Run.once(s, ta, 1, 131072, k);
    try testing.expectEqual(base, qsaSelectSplitCfgBuilds());
    try Run.once(s, ta, 2, 65536, k);
    try testing.expectEqual(base + 1, qsaSelectSplitCfgBuilds());
    try Run.once(s, ta, 1, 65536, k);
    try testing.expectEqual(base + 1, qsaSelectSplitCfgBuilds());
}

test "qsa select: the digit scheme is a pure function of rows" {
    const saved_digits = qsa_select_digits_override;
    defer qsa_select_digits_override = saved_digits;
    qsa_select_digits_override = null;
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(1));
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(7));
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(15));
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(16));
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(256));
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(4096));
    qsa_select_digits_override = 8;
    try testing.expectEqual(@as(c_int, 8), qsaSelectDigitBits(256));
    qsa_select_digits_override = 11;
    try testing.expectEqual(@as(c_int, 11), qsaSelectDigitBits(1));
    qsa_select_digits_override = null;

    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const prev_split = qsa_select_split_override;
    defer qsa_select_split_override = prev_split;
    qsa_select_split_override = false;
    const kk: c_int = 13;
    const rows: c_int = 5;
    const nb: c_int = 96;

    const Run = struct {
        fn once(st: mlx.mlx_stream, alloc: std.mem.Allocator, r: c_int, n: c_int, k: c_int) !void {
            const buf = try alloc.alloc(f32, @intCast(r * n));
            defer alloc.free(buf);
            for (buf, 0..) |*v, i| v.* = @floatFromInt(i % 97);
            const sh = [_]c_int{ 1, r, n };
            const sc = mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
            defer _ = mlx.mlx_array_free(sc);
            const bh = try alloc.alloc(i32, @intCast(r));
            defer alloc.free(bh);
            @memset(bh, n);
            const bd = mlx.mlx_array_new_data(bh.ptr, &[_]c_int{r}, 1, .int32);
            defer _ = mlx.mlx_array_free(bd);
            const out = (try qsaSelectTopBlocks(st, sc, bd, k)) orelse return error.SelectKernelDeclined;
            defer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_array_eval(out));
        }
    };

    qsa_select_digits_override = 11;
    try Run.once(s, ta, rows, nb, kk);
    const base = qsaSelectCfgBuilds();
    try Run.once(s, ta, rows, nb, kk);
    try testing.expectEqual(base, qsaSelectCfgBuilds());
    qsa_select_digits_override = 8;
    try Run.once(s, ta, rows, nb, kk);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
    try Run.once(s, ta, rows, nb, kk);
    try testing.expectEqual(base + 1, qsaSelectCfgBuilds());
}

test "qsa select: 8-bit digits are the exact top-k" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const ta = testing.allocator;
    const saved_digits = qsa_select_digits_override;
    const saved_split = qsa_select_split_override;
    defer {
        qsa_select_digits_override = saved_digits;
        qsa_select_split_override = saved_split;
    }
    const k: c_int = 512;
    const nbs = [_]c_int{ 2048, 15675, 24576, 65536, 215000 };
    const row_counts = [_]c_int{ 1, 7, 15, 16, 256 };
    var prng = std.Random.DefaultPrng.init(0x8B17_D161);
    const rnd = prng.random();

    for (nbs) |nb| {
        for (row_counts) |rows| {
            if (nb >= 215000 and rows >= 16) continue;
            const nbu: usize = @intCast(nb);
            const rowsu: usize = @intCast(rows);
            const ku: usize = @intCast(k);
            const buf = try ta.alloc(f32, rowsu * nbu);
            defer ta.free(buf);
            var kind: u8 = 0;
            while (kind < 5) : (kind += 1) {
                switch (kind) {
                    0 => {
                        for (buf) |*v| v.* = rnd.float(f32) * 2.0 - 1.0;
                    },
                    1 => {
                        for (buf) |*v| v.* = if (rnd.float(f32) < 0.60) 0.0 else rnd.float(f32) * 2.0 - 1.0;
                    },
                    2 => {
                        for (buf) |*v| v.* = if (rnd.float(f32) < 0.95) 0.0 else rnd.float(f32) * 2.0 - 1.0;
                    },
                    3 => {
                        for (buf) |*v| v.* = rnd.float(f32) * 0.5;
                        var r: usize = 0;
                        while (r < rowsu) : (r += 1) {
                            const row = buf[r * nbu ..][0..nbu];
                            var t: usize = 0;
                            while (t < ku + 400) : (t += 1) {
                                row[rnd.uintLessThan(usize, nbu)] = 3.0;
                            }
                            var h: usize = 0;
                            while (h < 50) : (h += 1) {
                                row[rnd.uintLessThan(usize, nbu)] = 9.0;
                            }
                        }
                    },
                    else => {
                        for (buf) |*v| v.* = rnd.float(f32);
                        var i: usize = 0;
                        while (i < buf.len) : (i += 1) {
                            const u = rnd.uintLessThan(u8, 100);
                            if (u < 2) buf[i] = std.math.nan(f32) else if (u < 32) buf[i] = -0.0 else if (u < 52) buf[i] = 0.0;
                        }
                    },
                }
                const sc_shape = [_]c_int{ 1, rows, nb };
                const scores = mlx.mlx_array_new_data(buf.ptr, &sc_shape, 3, .float32);
                defer _ = mlx.mlx_array_free(scores);
                const bound_kinds = [_]i32{ nb, @divTrunc(k, 2), @min(nb, k + 17) };
                for (bound_kinds) |bv| {
                    const bounds_host = try ta.alloc(i32, rowsu);
                    defer ta.free(bounds_host);
                    var r: usize = 0;
                    while (r < rowsu) : (r += 1) {
                        const adj = bv + @as(i32, @intCast(r % 3)) - 1;
                        bounds_host[r] = @max(0, @min(nb, adj));
                    }
                    const bounds = mlx.mlx_array_new_data(bounds_host.ptr, &[_]c_int{rows}, 1, .int32);
                    defer _ = mlx.mlx_array_free(bounds);

                    qsa_select_split_override = false;
                    qsa_select_digits_override = 11;
                    const ref11 = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(ref11);

                    qsa_select_digits_override = 8;
                    const single8 = (try qsaSelectTopBlocks(s, scores, bounds, k)) orelse return error.SelectKernelDeclined;
                    defer _ = mlx.mlx_array_free(single8);
                    if (!try qsaIdsBitEqual(s, ref11, single8)) {
                        std.debug.print("[qsa-select-8bit] single mismatch nb={d} rows={d} kind={d} bound={d}\n", .{ nb, rows, kind, bv });
                        return error.QsaSelect8BitMismatch;
                    }

                    if (rows <= QSA_SELECT_SPLIT_MAX_ROWS) {
                        const split8 = (try qsaSelectTopBlocksSplit(s, scores, bounds, k, rows, nb)) orelse return error.SelectKernelDeclined;
                        defer _ = mlx.mlx_array_free(split8);
                        if (!try qsaIdsBitEqual(s, ref11, split8)) {
                            std.debug.print("[qsa-select-8bit] split mismatch nb={d} rows={d} kind={d} bound={d}\n", .{ nb, rows, kind, bv });
                            return error.QsaSelect8BitMismatch;
                        }
                    }
                }
            }
        }
    }
}

test "qsa select tg policy: a pure function of rows, forced width wins" {
    const saved = qsa_select_tg_override;
    defer qsa_select_tg_override = saved;
    qsa_select_tg_override = null;
    try testing.expectEqual(@as(c_int, 256), qsaSelectTgFor(4096));
    try testing.expectEqual(@as(c_int, 256), qsaSelectTgFor(1024));
    try testing.expectEqual(@as(c_int, 256), qsaSelectTgFor(256));
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(255));
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(64));
    try testing.expectEqual(@as(c_int, 1024), qsaSelectTgFor(63));
    try testing.expectEqual(@as(c_int, 1024), qsaSelectTgFor(16));
    try testing.expectEqual(@as(c_int, 1024), qsaSelectTgFor(1));
    qsa_select_tg_override = 512;
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(4096));
    try testing.expectEqual(@as(c_int, 512), qsaSelectTgFor(1));
}
