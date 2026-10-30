//! sdk/blocks/moe/verify_kernels.
//! Owns VERIFY_EXPERT_REUSE_SOURCE, verify_expert_reuse_kernel, and the other verify-row MoE kernels.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const QsaCfgCache = @import("../../../mlx/kernel_cache.zig").QsaCfgCache;
const verifySharedCapable = @import("../linear/verify_projection.zig").verifySharedCapable;
const verifySharedHardware = @import("../linear/verify_projection.zig").verifySharedHardware;
const fusedSwiGLU = @import("../mlp/fused_activations.zig").fusedSwiGLU;
const swigluFusedEnabled = @import("../mlp/fused_activations.zig").swigluFusedEnabled;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;

// Preserve MLX's affine qmv K reduction while sharing each packed word across a pair.
pub const VERIFY_EXPERT_REUSE_SOURCE = kernelSource("kernels/moe/verify_expert_reuse.metal");

var verify_expert_reuse_kernel: ?mlx.mlx_fast_metal_kernel = null;


const ExpertReuseKey = struct { rows: c_int, n: c_int, k: c_int };
var verify_expert_reuse_cfgs = QsaCfgCache(ExpertReuseKey, 1){};

pub fn verifyExpertReuse(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, ids: mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware()) return null;
    if (x.ctx == null or w.ctx == null or sc.ctx == null or bi.ctx == null or ids.ctx == null) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(sc) != .bfloat16 or mlx.mlx_array_dtype(bi) != .bfloat16 or mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(sc);
    const bs = mlx.getShape(bi);
    const is = mlx.getShape(ids);
    if (xs.len != 3 or xs[1] != 1 or ws.len != 3 or ss.len != 3 or !std.mem.eql(c_int, ss, bs) or is.len != 1 or is[0] != xs[0]) return null;
    const count = xs[0];
    const k = xs[2];
    const n = ws[1];
    if (count < 2 or count > 320 or ws[0] < 1 or @divTrunc(count, ws[0]) >= 4) return null;
    if (k != 640 or n != 2560) return null;
    if (ws[2] != @divExact(k, 8) or ss[0] != ws[0] or ss[1] != n or ss[2] != @divExact(k, 64)) return null;
    const kernel = blk: {
        if (verify_expert_reuse_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "x", "w", "sc", "bi", "ids" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_expert_reuse", iv, ov, VERIFY_EXPERT_REUSE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_expert_reuse_kernel = value;
        break :blk value;
    };
    const key = ExpertReuseKey{ .rows = count, .n = n, .k = k };
    const cfg = blk: {
        if (verify_expert_reuse_cfgs.get(key)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ count, 1, n }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, 32, @divExact(n, 4), count));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 32, 2, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "K", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "N", n));
        verify_expert_reuse_cfgs.put(key, .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ x, w, sc, bi, ids };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// Match the eight bf16 partial sums used by MLX col_reduce_small (top-k rows, hidden H).
pub const VERIFY_EXPERT_REDUCE_SOURCE = kernelSource("kernels/moe/verify_expert_reduce.metal");
var verify_expert_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
const ExpertReduceCfgs = QsaCfgCache([3]c_int, 1);
var verify_expert_reduce_cfgs = ExpertReduceCfgs{};
var prefill_expert_reduce_cfgs = ExpertReduceCfgs{};

pub fn verifyExpertReduce(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !?mlx.mlx_array {
    if (!verifySharedHardware() or scores.ctx == null) return null;
    const ss = mlx.getShape(scores);
    if (ss.len != 3 or ss[0] != 1 or ss[1] > 32) return null;
    return expertReduceApply(s, down, inverse, scores, &verify_expert_reduce_cfgs);
}

/// The verify reduction at prefill widths, on every GPU: it only removes the
/// [B*S*K, hidden] unsorted and weighted intermediates.
pub fn moePrefillExpertReduce(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array) !?mlx.mlx_array {
    if (scores.ctx == null) return null;
    const ss = mlx.getShape(scores);
    // Verify widths (<= 32 tokens) keep verifyExpertReduce's own gates.
    if (ss.len != 3 or ss[0] * ss[1] <= 32) return null;
    if (mlx.cudaAvailable()) return @import("moe_prefill_cuda.zig").weightedSum(s, down, inverse, scores);
    return expertReduceApply(s, down, inverse, scores, &prefill_expert_reduce_cfgs);
}

fn expertReduceApply(s: mlx.mlx_stream, down: mlx.mlx_array, inverse: mlx.mlx_array, scores: mlx.mlx_array, cfgs: *ExpertReduceCfgs) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    if (down.ctx == null or inverse.ctx == null or scores.ctx == null) return null;
    if (mlx.mlx_array_dtype(down) != .bfloat16 or mlx.mlx_array_dtype(scores) != .bfloat16 or mlx.mlx_array_dtype(inverse) != .uint32) return null;
    const ds = mlx.getShape(down);
    const ss = mlx.getShape(scores);
    const ix = mlx.getShape(inverse);
    // The partial-sum layout reproduces MLX's tree only at these (top_k, hidden) pairs.
    if (ds.len != 2 or ss.len != 3 or ix.len != 1) return null;
    const topk = ss[2];
    const hidden = ds[1];
    if (!((topk == 10 and hidden == 2560) or (topk == 8 and hidden == 4096))) return null;
    const batch = ss[0];
    const tokens = ss[1];
    if (batch < 1 or tokens < 1 or batch * tokens < 2 or ds[0] != batch * tokens * topk or ix[0] != ds[0]) return null;
    const kernel = blk: {
        if (verify_expert_reduce_kernel) |value| break :blk value;
        const ins = [_][*:0]const u8{ "down", "inverse", "scores" };
        const outs = [_][*:0]const u8{"y"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const value = mlx.mlx_fast_metal_kernel_new("verify_expert_reduce", iv, ov, VERIFY_EXPERT_REDUCE_SOURCE, "", true, false);
        if (value.ctx == null) return error.MetalKernelCompileFailed;
        verify_expert_reduce_kernel = value;
        break :blk value;
    };
    const key = [3]c_int{ batch, tokens, topk };
    const cfg = blk: {
        if (cfgs.get(key)) |hit| break :blk hit[0];
        const value = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(value);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(value, &.{ batch, tokens, hidden }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(value, batch * tokens * hidden, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(value, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(value, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(value, "H", hidden));
        cfgs.put(key, .{value});
        break :blk value;
    };
    const values = [_]mlx.mlx_array{ down, inverse, scores };
    const iv = mlx.mlx_vector_array_new_data(&values, values.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, ov, 0));
    return result;
}

// Preserve input order on ties so routing and the inverse permutation stay stable.
pub const VERIFY_ROUTE_PACK_SOURCE = kernelSource("kernels/moe/verify_route_pack.metal");
var verify_route_pack_kernel: ?mlx.mlx_fast_metal_kernel = null;
var verify_route_pack_configs = QsaCfgCache(c_int, 1){};
const VerifyRoutePack = struct {
    inverse: mlx.mlx_array = .{},
    sorted: mlx.mlx_array = .{},
    lhs: mlx.mlx_array = .{},
    pub fn deinit(self: *VerifyRoutePack) void {
        inline for (.{ "inverse", "sorted", "lhs" }) |name| {
            if (@field(self, name).ctx != null) _ = mlx.mlx_array_free(@field(self, name));
        }
    }
};

pub fn verifyRoutePack(s: mlx.mlx_stream, ids: mlx.mlx_array, top_k: c_int) !?VerifyRoutePack {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or top_k != 10 or ids.ctx == null or mlx.mlx_array_dtype(ids) != .uint32) return null;
    const shape = mlx.getShape(ids);
    if (shape.len != 1 or shape[0] < 20 or shape[0] > 320 or @mod(shape[0], 10) != 0) return null;
    const kernel = blk: {
        if (verify_route_pack_kernel) |kernel| break :blk kernel;
        const ins = [_][*:0]const u8{"ids"};
        const outs = [_][*:0]const u8{ "inverse", "sorted", "lhs" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const kernel = mlx.mlx_fast_metal_kernel_new("verify_route_pack", iv, ov, VERIFY_ROUTE_PACK_SOURCE, "", true, false);
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        verify_route_pack_kernel = kernel;
        break :blk kernel;
    };
    const config = blk: {
        if (verify_route_pack_configs.get(shape[0])) |hit| break :blk hit[0];
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, shape.ptr, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
        verify_route_pack_configs.put(shape[0], .{config});
        break :blk config;
    };
    const iv = mlx.mlx_vector_array_new_data(&.{ids}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, config, s));
    var result: VerifyRoutePack = .{};
    errdefer result.deinit();
    inline for (.{ "inverse", "sorted", "lhs" }, 0..) |name, i| {
        @field(result, name) = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&@field(result, name), ov, i));
    }
    return result;
}

// A stable counting sort of routed expert ids for prefill widths, in three
// dispatches over 256-id blocks: per-block histograms, one scan to (expert,
// block) offsets, and a scatter that ranks ties by position within the block.
const PREFILL_ROUTE_BLOCK: c_int = 256;
pub const PREFILL_ROUTE_HIST_SOURCE = kernelSource("kernels/moe/prefill_route_hist.metal");
pub const PREFILL_ROUTE_SCAN_SOURCE = kernelSource("kernels/moe/prefill_route_scan.metal");
pub const PREFILL_ROUTE_SCATTER_SOURCE = kernelSource("kernels/moe/prefill_route_scatter.metal");
var prefill_route_kernels: [3]?mlx.mlx_fast_metal_kernel = .{ null, null, null };
var prefill_route_configs = QsaCfgCache([3]c_int, 3){};

fn prefillRouteKernel(slot: usize, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (prefill_route_kernels[slot]) |kernel| return kernel;
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, iv, ov, source, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    prefill_route_kernels[slot] = kernel;
    return kernel;
}

fn prefillRouteApply(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, cfg: mlx.mlx_fast_metal_kernel_config, outs: []const *mlx.mlx_array) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kernel, iv, cfg, s));
    for (outs, 0..) |out, i| try mlx.check(mlx.mlx_vector_array_get(out, ov, i));
}

/// `verifyRoutePack`'s outputs at prefill widths, where its all-pairs rank is
/// quadratic. Ties keep input order, which is what MLX's argsort does.
pub fn moePrefillRoutePack(s: mlx.mlx_stream, ids: mlx.mlx_array, top_k: c_int, experts: c_int) !?VerifyRoutePack {
    if (!mlx.streamIsGpu(s) or ids.ctx == null or mlx.mlx_array_dtype(ids) != .uint32) return null;
    if (top_k < 1 or experts < 1 or experts > 1024) return null;
    const shape = mlx.getShape(ids);
    // Verify widths (<= 32 tokens) keep verifyRoutePack's own gates.
    if (shape.len != 1 or @mod(shape[0], top_k) != 0 or @divTrunc(shape[0], top_k) <= 32) return null;
    const n = shape[0];
    const blocks = @divTrunc(n + PREFILL_ROUTE_BLOCK - 1, PREFILL_ROUTE_BLOCK);
    const hist = try prefillRouteKernel(0, "prefill_route_hist", &.{"ids"}, &.{"counts"}, PREFILL_ROUTE_HIST_SOURCE);
    const scan = try prefillRouteKernel(1, "prefill_route_scan", &.{"counts"}, &.{"offsets"}, PREFILL_ROUTE_SCAN_SOURCE);
    const scatter = try prefillRouteKernel(2, "prefill_route_scatter", &.{ "ids", "offsets" }, &.{ "inverse", "sorted", "lhs" }, PREFILL_ROUTE_SCATTER_SOURCE);
    const cfgs = blk: {
        const key = [3]c_int{ n, top_k, experts };
        if (prefill_route_configs.get(key)) |hit| break :blk hit;
        var made: [3]mlx.mlx_fast_metal_kernel_config = undefined;
        var built: usize = 0;
        errdefer for (made[0..built]) |cfg| {
            _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        };
        const table = [_]c_int{blocks * experts};
        for (&made, 0..) |*cfg, i| {
            cfg.* = mlx.mlx_fast_metal_kernel_config_new();
            built += 1;
            if (i < 2) {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg.*, &table, 1, .uint32));
            } else for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg.*, shape.ptr, 1, .uint32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg.*, if (i == 1) 256 else blocks * PREFILL_ROUTE_BLOCK, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg.*, 256, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg.*, "E", experts));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg.*, "TOPK", top_k));
        }
        prefill_route_configs.put(key, made);
        break :blk made;
    };
    var counts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(counts);
    try prefillRouteApply(s, hist, &.{ids}, cfgs[0], &.{&counts});
    var offsets = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(offsets);
    try prefillRouteApply(s, scan, &.{counts}, cfgs[1], &.{&offsets});
    var result: VerifyRoutePack = .{ .inverse = mlx.mlx_array_new(), .sorted = mlx.mlx_array_new(), .lhs = mlx.mlx_array_new() };
    errdefer result.deinit();
    try prefillRouteApply(s, scatter, &.{ ids, offsets }, cfgs[2], &.{ &result.inverse, &result.sorted, &result.lhs });
    return result;
}

pub fn verifyIndexedExpertInput(s: mlx.mlx_stream, x: mlx.mlx_array, lhs: mlx.mlx_array, experts: c_int) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !verifySharedHardware() or experts <= 0) return null;
    if (x.ctx == null or lhs.ctx == null or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(lhs) != .uint32) return null;
    const xs = mlx.getShape(x);
    const ix = mlx.getShape(lhs);
    if (xs.len != 2 or xs[1] != 2560 or xs[0] < 2 or xs[0] > 32 or ix.len != 1 or ix[0] != xs[0] * 10) return null;
    // Explicit LHS indices disable MLX's sorted matrix path; keep both arms below it.
    if (@divTrunc(ix[0], experts) >= 4) return null;
    var view = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(view);
    try mlx.check(mlx.mlx_reshape(&view, x, &.{ xs[0], 1, xs[1] }, 3, s));
    return view;
}
