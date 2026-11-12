//! MLX_SERVE_GLM_DSA_PROMPT_INDEXER: a GLM-5.3-Flash DSA layer's prompt pass, each piece the stock bytes.
//! - The indexer (`glm5_next.selectRows`' prompt branch, 2 rows or more, on NAX): the stock path writes the head-summed
//!   scores f32 [S, P] (`dsa_nax.indexerScores`), negates them, argpartitions them (MLX's GPU merge sort and its
//!   temporaries), slices, casts and sorts the 512 best pools and expands them to tokens. Here one launch
//!   (kernels/dsa_prompt_index.metal) scores each query row with the NAX text's MMAs in its order, keeps them in a
//!   scratch row of its threadgroup's and takes the 512 best by radix select in MLX's sort order (ties to the lower
//!   pool): the same tokens. A threadgroup takes rows g, g + G, ..., so the scratch is G rows (`groupsFor`).
//! - The output (`dsaAttn`'s dense and sparse parts, [1, 64, rows, 256] each): the stock path concatenates them along
//!   the rows, then its transpose's reshape copies again. Here each part's transpose is concatenated along the rows:
//!   one copy into the output projection's layout, the same values.
//! Each piece declines (null or false: the stock path runs) outside GLM-5.3-Flash's shapes.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");
const naxAvailable = @import("../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const Indexer = @import("glm5_next.zig").Indexer;

const Array = mlx.mlx_array;

/// The indexer: 32 heads of 128, a cache row its key and gate halves, pools of 4 tokens, 2,048 token slots (512 pools).
pub const ix_heads: c_int = 32;
pub const ix_dim: c_int = 128;
pub const ix_kpool: c_int = 4;
pub const ix_topk: c_int = 2048;
pub const sel_k: c_int = @divExact(ix_topk, ix_kpool);
/// The token slots a row: the selection's, then the row's unfinished pool's.
pub const out_w: c_int = ix_topk + ix_kpool - 1;
/// The attention's output: 64 heads, a value head of 256.
pub const attn_heads: c_int = 64;
pub const attn_v: c_int = 256;
/// The indexer's threadgroup: 8 simdgroups (the kernel's NSG).
pub const threads: c_int = 256;
/// The scratch rows' budget, and the fewest rows (groups) a launch keeps in flight.
pub const scratch_bytes: usize = 32 << 20;
pub const min_groups: c_int = 256;

pub const INDEX_HEADER = kernelSource("plugins/glm5_next/kernels/dsa_sparse_nax_header.metal") ++ "\n" ++
    kernelSource("plugins/glm5_next/kernels/dsa_prompt_index_header.metal");
pub const INDEX_SOURCE = kernelSource("plugins/glm5_next/kernels/dsa_prompt_index.metal");
var index_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// MLX_SERVE_GLM_DSA_PROMPT_INDEXER, read once (tests set it).
pub var switch_on: ?bool = null;
/// Indexer launches and one-copy outputs since the process started (tests read them).
pub var index_runs: u64 = 0;
pub var assemble_runs: u64 = 0;
var said = false;

pub fn on() bool {
    if (switch_on == null) switch_on = if (env.get(.MLX_SERVE_GLM_DSA_PROMPT_INDEXER)) |v| v[0] != '0' else true;
    return switch_on.?;
}

fn engaged(what: []const u8) void {
    if (said) return;
    said = true;
    log.info("[glm5_next] DSA prompt indexer engaged: first piece {s}\n", .{what});
}

/// The groups a launch over `rows` query rows of `pools` pools takes: the scratch budget's rows (at least
/// `min_groups`), at most one a row.
pub fn groupsFor(rows: c_int, pools: c_int) c_int {
    const per: usize = @as(usize, @intCast(pools)) * @sizeOf(f32);
    const budget: c_int = @intCast(@min(@as(usize, std.math.maxInt(c_int)), scratch_bytes / per));
    return @max(1, @min(rows, @max(min_groups, budget)));
}

pub const Plan = struct { rows: c_int, pools: c_int, groups: c_int };

/// Whether the indexer launch serves these inputs: `q` [1, S, 32, 128] bf16 (S >= 2), `w` f32 [1, S, 32], the indexer
/// cache [1, T, 256] bf16 with more pools than the selection takes. MLX-free.
pub fn indexPlan(q: []const c_int, q_bf16: bool, w: []const c_int, w_f32: bool, cache: []const c_int, cache_bf16: bool) ?Plan {
    if (!q_bf16 or !w_f32 or !cache_bf16) return null;
    if (q.len != 4 or q[0] != 1 or q[1] < 2 or q[2] != ix_heads or q[3] != ix_dim) return null;
    if (w.len != 3 or w[0] != 1 or w[1] != q[1] or w[2] != ix_heads) return null;
    if (cache.len != 3 or cache[0] != 1 or cache[2] != 2 * ix_dim) return null;
    const pools = @divTrunc(cache[1], ix_kpool);
    if (pools <= sel_k) return null;
    return .{ .rows = q[1], .pools = pools, .groups = groupsFor(q[1], pools) };
}

/// Whether `ix` is GLM-5.3-Flash's indexer, the one the kernel's constants hold. MLX-free.
pub fn indexerServes(ix: Indexer) bool {
    return ix.heads == ix_heads and ix.head_dim == ix_dim and ix.kpool == ix_kpool and ix.topk == ix_topk;
}

/// The launch's outputs: the tokens int32 [1, S, 2051] and the scratch rows f32 [G, P] (with one row a group, G = S,
/// each row's scores of the pools it can see; the rest unwritten).
pub const Index = struct { tokens: Array, scores: Array };

/// The indexer over queries at positions `q_start ..`: `q_idx` [1, S, 32, 128], `ws` f32 [1, S, 32] (the head weights
/// with both scales, as `selectRows` makes them), `idx_cache` [1, T, 256]; `groups` the launch's (null: `groupsFor`).
/// Null outside `indexPlan`.
pub fn index(s: mlx.mlx_stream, q_idx: Array, ws: Array, idx_cache: Array, q_start: c_int, groups: ?c_int) !?Index {
    if (!mlx.streamIsGpu(s)) return null;
    const plan = indexPlan(mlx.getShape(q_idx), mlx.mlx_array_dtype(q_idx) == .bfloat16, mlx.getShape(ws), mlx.mlx_array_dtype(ws) == .float32, mlx.getShape(idx_cache), mlx.mlx_array_dtype(idx_cache) == .bfloat16) orelse return null;
    const g = groups orelse plan.groups;
    if (g < 1 or g > plan.rows) return null;
    const kern = index_kernel orelse blk: {
        const ins = [_][*:0]const u8{ "q", "w", "cache", "dims" };
        const outs = [_][*:0]const u8{ "out", "scr" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const k = mlx.mlx_fast_metal_kernel_new("glm5_dsa_prompt_index", iv, ov, INDEX_SOURCE, INDEX_HEADER, true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        index_kernel = k;
        break :blk k;
    };
    const dims = mlx.mlx_array_new_data(&[_]i32{ plan.rows, q_start, plan.pools, g }, &[_]c_int{4}, 1, .int32);
    defer _ = mlx.mlx_array_free(dims);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, plan.rows, out_w }, 3, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ g, plan.pools }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, threads * g, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    const inputs = [_]Array{ q_idx, ws, idx_cache, dims };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var tokens = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(tokens);
    try mlx.check(mlx.mlx_vector_array_get(&tokens, ov, 0));
    var scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_vector_array_get(&scores, ov, 1));
    return .{ .tokens = tokens, .scores = scores };
}

/// `glm5_next.selectRows`' prompt branch for queries at positions `q_start ..` (`w` [1, S, 32] f32, scaled by the
/// heads' count, as `dsaAttn` makes it) over `p` pools of which `select_k` are taken: the tokens int32 [1, S, 2051], a
/// new array; null with the switch off, off NAX, at one row, or outside GLM-5.3-Flash's indexer.
pub fn selectRows(s: mlx.mlx_stream, q_idx: Array, w: Array, idx_cache: Array, q_start: c_int, p: c_int, select_k: c_int, ix: Indexer) !?Array {
    if (!mlx.streamIsGpu(s) or !indexerServes(ix) or select_k != sel_k or p <= select_k) return null;
    const qs = mlx.getShape(q_idx);
    if (qs.len != 4 or qs[1] < 2 or !on() or !naxAvailable()) return null;
    // the head weights as `selectRows` scales them (by head_dim^-0.5, the same multiply)
    const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
    defer _ = mlx.mlx_array_free(scale);
    var ws = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ws);
    try mlx.check(mlx.mlx_multiply(&ws, w, scale, s));
    const cs = mlx.getShape(idx_cache);
    if (cs.len != 3 or @divTrunc(cs[1], ix_kpool) != p) return null;
    const r = (try index(s, q_idx, ws, idx_cache, q_start, null)) orelse return null;
    _ = mlx.mlx_array_free(r.scores);
    index_runs += 1;
    engaged("the indexer launch");
    return r.tokens;
}

/// `dsaAttn`'s output from its dense rows' `dense` and sparse rows' `sparse` (each [1, 64, rows, 256], one dtype) in
/// the output projection's row order, [1, S, 64, 256] in one copy, into `o`; false (nothing set) with the switch off or
/// outside GLM-5.3-Flash's attention (the stock concatenate and transpose run).
pub fn assemble(o: *Array, s: mlx.mlx_stream, dense: Array, sparse: Array) !bool {
    const ds = mlx.getShape(dense);
    const ss = mlx.getShape(sparse);
    if (ds.len != 4 or ss.len != 4 or ds[0] != 1 or ss[0] != 1) return false;
    if (ds[1] != attn_heads or ss[1] != attn_heads or ds[3] != attn_v or ss[3] != attn_v) return false;
    if (mlx.mlx_array_dtype(dense) != mlx.mlx_array_dtype(sparse) or !on()) return false;
    var td = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(td);
    try mlx.check(mlx.mlx_transpose_axes(&td, dense, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var ts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ts);
    try mlx.check(mlx.mlx_transpose_axes(&ts, sparse, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    const two = [_]Array{ td, ts };
    const vec = mlx.mlx_vector_array_new_data(&two, two.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_concatenate_axis(o, vec, 1, s));
    assemble_runs += 1;
    engaged("the one-copy output");
    return true;
}

// ── tests ──

const testing = std.testing;
const ops = @import("../../mlx/ops.zig");
const glm5 = @import("glm5_next.zig");
const dsa_nax = @import("dsa_nax.zig");

test "glm5_next dsa prompt indexer: plans GLM-5.3-Flash's indexer shapes from 2 rows and more pools than it selects, its scratch 32 MiB of rows at most and at least 256 rows" {
    // a 16K prompt's main call and tail, a 64K and a 128K context
    try testing.expectEqual(Plan{ .rows = 14305, .pools = 4089, .groups = 2051 }, indexPlan(&.{ 1, 14305, 32, 128 }, true, &.{ 1, 14305, 32 }, true, &.{ 1, 16356, 256 }, true).?);
    try testing.expectEqual(Plan{ .rows = 31, .pools = 4096, .groups = 31 }, indexPlan(&.{ 1, 31, 32, 128 }, true, &.{ 1, 31, 32 }, true, &.{ 1, 16387, 256 }, true).?);
    try testing.expectEqual(Plan{ .rows = 4096, .pools = 16384, .groups = 512 }, indexPlan(&.{ 1, 4096, 32, 128 }, true, &.{ 1, 4096, 32 }, true, &.{ 1, 65536, 256 }, true).?);
    try testing.expectEqual(Plan{ .rows = 4096, .pools = 65536, .groups = 256 }, indexPlan(&.{ 1, 4096, 32, 128 }, true, &.{ 1, 4096, 32 }, true, &.{ 1, 262144, 256 }, true).?);
    // one row, another indexer, other dtypes, a batch, a cache within the selection, mismatched weights
    try testing.expect(indexPlan(&.{ 1, 1, 32, 128 }, true, &.{ 1, 1, 32 }, true, &.{ 1, 16356, 256 }, true) == null);
    try testing.expect(indexPlan(&.{ 1, 64, 64, 128 }, true, &.{ 1, 64, 64 }, true, &.{ 1, 16356, 256 }, true) == null);
    try testing.expect(indexPlan(&.{ 1, 64, 32, 64 }, true, &.{ 1, 64, 32 }, true, &.{ 1, 16356, 128 }, true) == null);
    try testing.expect(indexPlan(&.{ 1, 64, 32, 128 }, false, &.{ 1, 64, 32 }, true, &.{ 1, 16356, 256 }, true) == null);
    try testing.expect(indexPlan(&.{ 1, 64, 32, 128 }, true, &.{ 1, 64, 32 }, false, &.{ 1, 16356, 256 }, true) == null);
    try testing.expect(indexPlan(&.{ 2, 64, 32, 128 }, true, &.{ 2, 64, 32 }, true, &.{ 2, 16356, 256 }, true) == null);
    try testing.expect(indexPlan(&.{ 1, 64, 32, 128 }, true, &.{ 1, 64, 32 }, true, &.{ 1, 2051, 256 }, true) == null);
    try testing.expect(indexPlan(&.{ 1, 64, 32, 128 }, true, &.{ 1, 63, 32 }, true, &.{ 1, 16356, 256 }, true) == null);
    // the scratch: budget rows, at least min_groups, at most one a query row
    try testing.expectEqual(@as(c_int, 2051), groupsFor(100000, 4089));
    try testing.expectEqual(@as(c_int, 17), groupsFor(17, 4089));
    try testing.expectEqual(@as(c_int, 256), groupsFor(100000, 1 << 20));
    try testing.expect(@as(usize, @intCast(groupsFor(100000, 4089))) * 4089 * 4 <= scratch_bytes);
    // only GLM-5.3-Flash's indexer
    try testing.expect(indexerServes(test_ix));
    try testing.expect(!indexerServes(.{ .topk = 1024, .kpool = 4, .heads = 32, .head_dim = 128 }));
    try testing.expect(!indexerServes(.{ .topk = 2048, .kpool = 8, .heads = 32, .head_dim = 128 }));
}

/// GLM-5.3-Flash's indexer (the DSA levers' tests share it).
pub const test_ix: Indexer = .{ .topk = 2048, .kpool = 4, .heads = 32, .head_dim = 128 };

/// What a selection case's inputs hold.
pub const Fill = enum { random, zero_weights, repeating_keys };
/// A selection case: the keys in the cache and the query rows (its last ones).
pub const SelCase = struct { name: []const u8, t: c_int, rows: c_int, fill: Fill = .random };

/// Random normal values of `shape`, evaluated; `seed` advances.
pub fn testNormal(s: mlx.mlx_stream, seed: *u64, shape: []const c_int, dtype: mlx.mlx_dtype, scale: f32) !Array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, seed.*));
    seed.* += 1;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_random_normal(&out, shape.ptr, shape.len, dtype, 0, scale, key, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// A selection's inputs: q_idx bf16 [1, rows, 32, 128], the head weights f32 [1, rows, 32], the indexer cache bf16
/// [1, t, 256] (a pool's key its completing row's gate half).
pub const SelInputs = struct {
    q: Array,
    w: Array,
    cache: Array,
    pub fn deinit(x: *SelInputs) void {
        for ([_]Array{ x.q, x.w, x.cache }) |a| _ = mlx.mlx_array_free(a);
    }
};

pub fn testSelInputs(s: mlx.mlx_stream, seed: *u64, c: SelCase) !SelInputs {
    const ix = test_ix;
    const q = try testNormal(s, seed, &.{ 1, c.rows, ix.heads, ix.head_dim }, .bfloat16, 1.0);
    errdefer _ = mlx.mlx_array_free(q);
    var w = try testNormal(s, seed, &.{ 1, c.rows, ix.heads }, .float32, 1.0);
    errdefer _ = mlx.mlx_array_free(w);
    if (c.fill == .zero_weights) {
        _ = mlx.mlx_array_free(w);
        w = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&w, &[_]c_int{ 1, c.rows, ix.heads }, 3, .float32, s));
        try mlx.check(mlx.mlx_array_eval(w));
    }
    const width = 2 * ix.head_dim;
    const cache = if (c.fill == .repeating_keys) blk: {
        // 7 pools' rows tiled over the cache: pool p's key is pool (p mod 7)'s
        const period: c_int = 7 * ix.kpool;
        const base = try testNormal(s, seed, &.{ 1, period, width }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(base);
        var tiled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tiled);
        try mlx.check(mlx.mlx_tile(&tiled, base, &[_]c_int{ 1, @divTrunc(c.t + period - 1, period), 1 }, 3, s));
        var cut = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cut);
        try mlx.check(mlx.mlx_slice(&cut, tiled, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, c.t, width }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var cont = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cont);
        try mlx.check(mlx.mlx_contiguous(&cont, cut, false, s));
        try mlx.check(mlx.mlx_array_eval(cont));
        break :blk cont;
    } else try testNormal(s, seed, &.{ 1, c.t, width }, .bfloat16, 1.0);
    return .{ .q = q, .w = w, .cache = cache };
}

/// `got` bit-equal to `want`, naming the case when not.
pub fn expectCase(what: []const u8, name: []const u8, got: Array, want: Array) !void {
    ops.expectBitEqual(got, want) catch |e| {
        std.debug.print("glm5_next {s}: case {s} differs from the stock path\n", .{ what, name });
        return e;
    };
}

fn selectWith(s: mlx.mlx_stream, x: SelInputs, lever: bool) !Array {
    const saved = switch_on;
    switch_on = lever;
    defer switch_on = saved;
    const out = try glm5.selectTokens(s, x.q, x.w, x.cache, test_ix);
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The launch's scores at one row a threadgroup against `dsa_nax.indexerScores`, every score a row can see, as bits.
fn expectScores(s: mlx.mlx_stream, x: SelInputs, c: SelCase) !void {
    const ix = test_ix;
    const q_start = c.t - c.rows;
    const p = @divTrunc(c.t, ix.kpool);
    const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(ix.head_dim))));
    defer _ = mlx.mlx_array_free(scale);
    var ws = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ws);
    try mlx.check(mlx.mlx_multiply(&ws, x.w, scale, s));
    // the pools' keys as `storedPoolKeys` slices them: each pool's last row's gate half
    var pk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pk);
    try mlx.check(mlx.mlx_slice(&pk, x.cache, &[_]c_int{ 0, ix.kpool - 1, ix.head_dim }, 3, &[_]c_int{ 1, p * ix.kpool, 2 * ix.head_dim }, 3, &[_]c_int{ 1, ix.kpool, 1 }, 3, s));
    const want = (try dsa_nax.indexerScores(s, x.q, ws, pk, q_start, ix.kpool)) orelse return error.TestUnexpectedResult;
    defer _ = mlx.mlx_array_free(want);
    const got = (try index(s, x.q, ws, x.cache, q_start, c.rows)) orelse return error.TestUnexpectedResult;
    defer {
        _ = mlx.mlx_array_free(got.tokens);
        _ = mlx.mlx_array_free(got.scores);
    }
    var wc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wc);
    try mlx.check(mlx.mlx_contiguous(&wc, want, false, s));
    var gc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gc);
    try mlx.check(mlx.mlx_contiguous(&gc, got.scores, false, s));
    try mlx.check(mlx.mlx_array_eval(wc));
    try mlx.check(mlx.mlx_array_eval(gc));
    const pu: usize = @intCast(p);
    const n: usize = @as(usize, @intCast(c.rows)) * pu;
    const wf = (mlx.mlx_array_data_float32(wc) orelse return error.TestUnexpectedResult)[0..n];
    const gf = (mlx.mlx_array_data_float32(gc) orelse return error.TestUnexpectedResult)[0..n];
    for (0..@as(usize, @intCast(c.rows))) |r| {
        const q_pos = q_start + @as(c_int, @intCast(r));
        const pv: usize = @intCast(@min(p, @divTrunc(q_pos + 1, ix.kpool)));
        testing.expectEqualSlices(u8, std.mem.sliceAsBytes(wf[r * pu ..][0..pv]), std.mem.sliceAsBytes(gf[r * pu ..][0..pv])) catch |e| {
            std.debug.print("glm5_next dsa prompt indexer scores: case {s} row {d} differs\n", .{ c.name, r });
            return e;
        };
    }
}

test "glm5_next dsa prompt indexer: the one launch writes the stock selection's tokens and the NAX indexer's scores bit for bit, and declines rows that select every pool" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    if (!naxAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const Case = struct { sel: SelCase, runs: bool };
    const cases = [_]Case{
        .{ .sel = .{ .name = "main16k", .t = 16356, .rows = 14305 }, .runs = true },
        .{ .sel = .{ .name = "tail16k", .t = 16387, .rows = 31 }, .runs = true },
        .{ .sel = .{ .name = "warm995", .t = 16356 + 995, .rows = 995 }, .runs = true },
        .{ .sel = .{ .name = "warm17", .t = 16356 + 17, .rows = 17 }, .runs = true },
        .{ .sel = .{ .name = "last4k_of_32k", .t = 32768, .rows = 4096 }, .runs = true },
        .{ .sel = .{ .name = "last256_of_64k", .t = 65536, .rows = 256 }, .runs = true },
        .{ .sel = .{ .name = "last64_of_128k", .t = 131072, .rows = 64 }, .runs = true },
        .{ .sel = .{ .name = "main16k_zero_weights", .t = 16356, .rows = 14305, .fill = .zero_weights }, .runs = true },
        .{ .sel = .{ .name = "main16k_repeating_keys", .t = 16356, .rows = 14305, .fill = .repeating_keys }, .runs = true },
        .{ .sel = .{ .name = "cold995", .t = 995, .rows = 995 }, .runs = false },
        .{ .sel = .{ .name = "cold17", .t = 17, .rows = 17 }, .runs = false },
    };
    var seed: u64 = 20261109;
    for (cases) |c| {
        var x = try testSelInputs(s, &seed, c.sel);
        defer x.deinit();
        const want = try selectWith(s, x, false);
        defer _ = mlx.mlx_array_free(want);
        const before = index_runs;
        const got = try selectWith(s, x, true);
        defer _ = mlx.mlx_array_free(got);
        try expectCase("dsa prompt indexer", c.sel.name, got, want);
        try testing.expectEqual(@as(u64, @intFromBool(c.runs)), index_runs - before);
        if (c.runs) try expectScores(s, x, c.sel);
    }
}

/// `dsaAttn`'s stock output: the parts concatenated along the rows, transposed and reshaped.
fn stockOut(s: mlx.mlx_stream, dn: Array, sp: Array, rows: c_int) !Array {
    const two = [_]Array{ dn, sp };
    const vec = mlx.mlx_vector_array_new_data(&two, two.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 2, s));
    var ot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ot);
    try mlx.check(mlx.mlx_transpose_axes(&ot, o, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, ot, &[_]c_int{ 1, rows, attn_heads * attn_v }, 3, s));
    return flat;
}

test "glm5_next dsa prompt indexer: the one-copy output equals the stock concatenate, transpose and reshape bit for bit, and another attention's parts keep the stock path" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const saved = switch_on;
    switch_on = true;
    defer switch_on = saved;
    const OutCase = struct { name: []const u8, dense: c_int, sparse: c_int, strided: bool = false };
    // a 16K prompt's main call (again with the dense part a transposed view), a 4,096-token prompt, 2,068 tokens
    const cases = [_]OutCase{
        .{ .name = "main16k", .dense = 2051, .sparse = 14305 },
        .{ .name = "main16k_strided", .dense = 2051, .sparse = 14305, .strided = true },
        .{ .name = "cold4k", .dense = 2051, .sparse = 2045 },
        .{ .name = "cold2068", .dense = 2051, .sparse = 17 },
    };
    var seed: u64 = 20261110;
    const h = attn_heads;
    const v = attn_v;
    for (cases) |c| {
        const rows = c.dense + c.sparse;
        const dn_shape: [4]c_int = if (c.strided) .{ 1, c.dense, h, v } else .{ 1, h, c.dense, v };
        const dn_src = try testNormal(s, &seed, &dn_shape, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(dn_src);
        var dn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dn);
        if (c.strided) {
            try mlx.check(mlx.mlx_transpose_axes(&dn, dn_src, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        } else _ = mlx.mlx_array_set(&dn, dn_src);
        const sp = try testNormal(s, &seed, &.{ 1, h, c.sparse, v }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(sp);
        const want = try stockOut(s, dn, sp, rows);
        defer _ = mlx.mlx_array_free(want);
        var o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o);
        try testing.expect(try assemble(&o, s, dn, sp));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, o, &[_]c_int{ 1, rows, h * v }, 3, s));
        try expectCase("dsa prompt output", c.name, flat, want);
    }
    // another model's attention (32 heads) is left to the stock path
    const dn = try testNormal(s, &seed, &.{ 1, 32, 64, v }, .bfloat16, 1.0);
    defer _ = mlx.mlx_array_free(dn);
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try testing.expect(!try assemble(&o, s, dn, dn));
}
