//! Test fixtures for the base parity proofs: a tiny random-weight pack written to disk and loaded through the legacy
//! path (`parseConfigFromJson` → `loadModelWeights` → `Transformer.init`), and the helpers that hand the legacy
//! model's loaded arrays to a base's blocks.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const model = @import("../../config/model.zig");
const Transformer = @import("../../../engines/mlx/transformer.zig").Transformer;
const computeQuantParams = @import("../../quant/params.zig").computeQuantParams;
const FusedRows = @import("../../blocks/linear/fused_rows.zig").FusedRows;
const linear = @import("../blocks/linear.zig");
const abi = @import("../../../abi/plugin.zig");
const block = @import("../block.zig");
const cache_ops = @import("../../../engines/mlx/kv/cache_ops.zig");
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;

const gguf_kernels = @import("../../quant/gguf/kernels.zig");
const gguf_quants = @import("../../quant/gguf/quants.zig");
const Weights = @import("../../weights/resident.zig").Weights;

const testing = std.testing;
const Array = mlx.mlx_array;

pub fn stream() mlx.mlx_stream {
    return if (mlx.noGpuBackend()) mlx.mlx_default_cpu_stream_new() else mlx.gpuStream();
}

/// A pack being written: random tensors into a safetensors map.
pub const Pack = struct {
    map: mlx.mlx_map_string_to_array,
    rnd: std.Random.DefaultPrng,
    s: mlx.mlx_stream,

    pub fn init(seed: u64) Pack {
        return .{ .map = mlx.mlx_map_string_to_array_new(), .rnd = .init(seed), .s = stream() };
    }

    pub fn deinit(p: *Pack) void {
        _ = mlx.mlx_map_string_to_array_free(p.map);
    }

    /// `arr` under `name` (the map takes its own reference).
    pub fn put(p: *Pack, name: []const u8, arr: Array) !void {
        const z = try std.fmt.allocPrintSentinel(testing.allocator, "{s}", .{name}, 0);
        defer testing.allocator.free(z);
        try mlx.check(mlx.mlx_map_string_to_array_insert(p.map, z.ptr, arr));
    }

    fn random(p: *Pack, shape: []const c_int, scale: f32, offset: f32, dt: mlx.mlx_dtype) !Array {
        var n: usize = 1;
        for (shape) |d| n *= @intCast(d);
        const host = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(host);
        const r = p.rnd.random();
        for (host) |*v| v.* = offset + scale * (r.float(f32) * 2 - 1);
        const f = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
        defer _ = mlx.mlx_array_free(f);
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, f, dt, p.s));
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    /// A bf16 tensor of uniform values in `offset ± scale`.
    pub fn dense(p: *Pack, name: []const u8, shape: []const c_int, scale: f32, offset: f32) !void {
        return p.denseAs(name, shape, scale, offset, .bfloat16);
    }

    pub fn denseAs(p: *Pack, name: []const u8, shape: []const c_int, scale: f32, offset: f32, dt: mlx.mlx_dtype) !void {
        const a = try p.random(shape, scale, offset, dt);
        defer _ = mlx.mlx_array_free(a);
        try p.put(name, a);
    }

    /// A norm weight near one.
    pub fn norm(p: *Pack, name: []const u8, n: c_int) !void {
        try p.dense(name, &.{n}, 0.2, 1.0);
    }

    /// `<base>.weight` `[rows, cols]`: bf16, or 4-bit affine (group 64) with its scales and biases.
    pub fn weight(p: *Pack, base: []const u8, rows: c_int, cols: c_int, quantized: bool) !void {
        return p.tensor(base, &.{ rows, cols }, quantized);
    }

    /// As `weight`, any rank (an expert bank is `[experts, rows, cols]`); quantized along the last axis.
    pub fn tensor(p: *Pack, base: []const u8, shape: []const c_int, quantized: bool) !void {
        var buf: [192]u8 = undefined;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(shape[shape.len - 1])));
        if (!quantized) return p.dense(try std.fmt.bufPrint(&buf, "{s}.weight", .{base}), shape, scale, 0);
        return p.affine(base, shape, 4, 64);
    }

    /// As `tensor`, 2-bit at `group` with every bias exactly -scale (a ternary pack's layout).
    pub fn ternary(p: *Pack, base: []const u8, shape: []const c_int, group: c_int) !void {
        try p.affine(base, shape, 2, group);
        var nb: [192]u8 = undefined;
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        try mlx.check(mlx.mlx_map_string_to_array_get(&sc, p.map, (try std.fmt.bufPrintSentinel(&nb, "{s}.scales", .{base}, 0)).ptr));
        var neg = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(neg);
        try mlx.check(mlx.mlx_negative(&neg, sc, p.s));
        try mlx.check(mlx.mlx_array_eval(neg));
        try p.put(try std.fmt.bufPrint(&nb, "{s}.biases", .{base}), neg);
    }

    /// As `tensor`, affine at `bits` / `group`.
    pub fn affine(p: *Pack, base: []const u8, shape: []const c_int, bits: c_int, group: c_int) !void {
        return p.quantize(base, shape, "affine", bits, group);
    }

    /// As `tensor`, in MLX's `mode` ("affine", "mxfp4", "nvfp4", "mxfp8") at `bits` / `group`.
    pub fn quantize(p: *Pack, base: []const u8, shape: []const c_int, mode: [*:0]const u8, bits: c_int, group: c_int) !void {
        var buf: [192]u8 = undefined;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(shape[shape.len - 1])));
        const d = try p.random(shape, scale, 0, .bfloat16);
        defer _ = mlx.mlx_array_free(d);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, d, mlx.mlx_optional_int.some(group), mlx.mlx_optional_int.some(bits), mode, .{}, p.s));
        const suffixes = [_][]const u8{ ".weight", ".scales", ".biases" };
        for (suffixes[0..mlx.mlx_vector_array_size(triple)], 0..) |suffix, i| {
            var a = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(a);
            try mlx.check(mlx.mlx_vector_array_get(&a, triple, i));
            try mlx.check(mlx.mlx_array_eval(a));
            try p.put(try std.fmt.bufPrint(&buf, "{s}{s}", .{ base, suffix }), a);
        }
    }

    /// `<base>.weight` and its sentinel as the GGUF loader leaves a block-quantized tensor (`sdk/quant/gguf/weights.zig`):
    /// `ggufBlocks`, split rows packed as the kernels read them. `shape` is `[..., rows, in]`.
    pub fn gguf(p: *Pack, base: []const u8, shape: []const c_int, comptime ty: gguf_quants.GgmlType) !void {
        const a = testing.allocator;
        const in: usize = @intCast(shape[shape.len - 1]);
        var rows: usize = 1;
        for (shape[0 .. shape.len - 1]) |d| rows *= @intCast(d);
        const raw = try ggufBlocks(p.rnd.random(), ty, rows * (in / ty.blockElems()));
        defer a.free(raw);
        const rb = gguf_kernels.rowBytes(ty, in);
        const packed_rows = try a.alloc(u8, rows * rb);
        defer a.free(packed_rows);
        if (gguf_kernels.isSplit(ty, in)) gguf_kernels.packRows(ty, in, packed_rows, raw) else @memcpy(packed_rows, raw);
        var sh: [4]c_int = undefined;
        @memcpy(sh[0..shape.len], shape);
        sh[shape.len - 1] = @intCast(rb);
        var nb: [192]u8 = undefined;
        const w = mlx.mlx_array_new_data(packed_rows.ptr, &sh, @intCast(shape.len), .uint8);
        defer _ = mlx.mlx_array_free(w);
        try p.put(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}), w);
        const sc = try gguf_kernels.sentinel(.{ .ty = ty }, p.s);
        defer _ = mlx.mlx_array_free(sc);
        try p.put(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}), sc);
    }

    /// Writes the pack as `<dir>/model.safetensors`.
    pub fn save(p: *const Pack, dir: []const u8) !void {
        const file = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model.safetensors", .{dir}, 0);
        defer testing.allocator.free(file);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_save_safetensors(file.ptr, p.map, meta));
    }

};

/// `ids` `[1, n]` int32.
/// `n` ggml blocks of `ty` as a GGUF file stores them: drawn at random from the type's reference blocks, their f16
/// scales redrawn so every weight stays under ~0.1. Caller frees (testing allocator).
pub fn ggufBlocks(r: std.Random, comptime ty: gguf_quants.GgmlType, n: usize) ![]u8 {
    const ref = @embedFile("../../quant/gguf/fixtures/" ++ @tagName(ty) ++ ".bin")[0 .. 4 * ty.blockBytes()];
    // Byte offsets of the block's f16 scales, and the largest |scale| keeping a weight under ~0.1.
    const scales: struct { at: []const usize, max: f32 } = comptime switch (ty) {
        .q8_0 => .{ .at = &.{0}, .max = 0.0008 },
        .q5_0 => .{ .at = &.{0}, .max = 0.006 },
        .q4_k, .q5_k => .{ .at = &.{ 0, 2 }, .max = 0.0001 },
        .q6_k => .{ .at = &.{208}, .max = 0.00002 },
        .iq4_xs => .{ .at = &.{0}, .max = 0.00002 },
        else => @compileError("fixture.ggufBlocks: no scale layout for " ++ @tagName(ty)),
    };
    const raw = try testing.allocator.alloc(u8, n * ty.blockBytes());
    for (0..n) |i| {
        const blk = raw[i * ty.blockBytes() ..][0..ty.blockBytes()];
        @memcpy(blk, ref[r.uintLessThan(usize, 4) * ty.blockBytes() ..][0..ty.blockBytes()]);
        for (scales.at) |o| std.mem.writeInt(u16, blk[o..][0..2], @bitCast(@as(f16, @floatCast(scales.max * (r.float(f32) * 2 - 1)))), .little);
    }
    return raw;
}

pub fn tokens(g: *ops.Graph, ids: []const i32) !Array {
    return g.fromSlice(i32, ids, &.{ 1, @intCast(ids.len) });
}

/// A legacy `[1, S, V]` result as `[S, V]`, owned by `g`.
pub fn rows2(g: *ops.Graph, x: Array) !Array {
    const sh = ops.shape(x);
    return g.reshape(x, &.{ sh[0] * sh[1], sh[2] });
}

pub const Step = struct { mode: abi.Mode, ids: []const i32 };

/// Prefill, then decode and verify windows of several widths interleaved.
pub const steps = [_]Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 99, 7, 250, 1 } },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .decode, .ids = &.{64} },
};

/// Fresh recurrent entries, one per layer.
pub fn newEntries(n: usize) ![]SSMCacheEntry {
    const es = try testing.allocator.alloc(SSMCacheEntry, n);
    for (es) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
    return es;
}

pub fn freeEntries(es: []SSMCacheEntry) void {
    for (es) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
        ssmFreeSpecCapture(e);
    }
    testing.allocator.free(es);
}

/// Every recurrent state and spec capture of `got` bit-equal to `want`'s.
pub fn expectEntries(want: []const SSMCacheEntry, got: []const SSMCacheEntry) !void {
    for (want, got) |*w, *g| {
        try testing.expectEqual(w.initialized, g.initialized);
        inline for (.{ "conv_state", "ssm_state", "spec_conv_input", "spec_state_seq" }) |f| {
            const wa = @field(w, f);
            const ga = @field(g, f);
            try testing.expectEqual(wa.ctx == null, ga.ctx == null);
            if (wa.ctx != null) try expectStateEqual(ga, wa);
        }
    }
}

/// Bit-equal, except a state sequence's last row: the GDN verify kernels never write it (a rollback never reads it).
fn expectStateEqual(got: Array, want: Array) !void {
    var g = ops.Graph.init(testing.allocator, stream());
    defer g.deinit();
    const sh = ops.shape(want);
    if (sh.len == 5 and sh[0] > 1 and ops.shape(got).len == 5)
        return ops.expectBitEqual(try g.sliceAxis(got, 0, 0, sh[0] - 1), try g.sliceAxis(want, 0, 0, sh[0] - 1));
    return ops.expectBitEqual(got, want);
}

/// The base's test cache slot behind a block's `Ctx`.
pub fn slotOf(c: *const block.Ctx) *cache_ops.Slot {
    return @ptrCast(@alignCast(c.cache().?));
}
