//! GGUF tensors -> the name -> array map mlx-serve's Transformer binds from.
//! Quantized tensors keep their bytes (see kernels.Info for how they travel),
//! float tensors become bf16 like an mlx-community checkpoint.
const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const gguf = @import("gguf.zig");
const kernels = @import("kernels.zig");
const arch_mod = @import("arch.zig");
const qwen35 = arch_mod.qwen35;
const Arch = arch_mod.Arch;

pub const WeightMap = std.StringHashMap(mlx.mlx_array);

/// Fills `out` (keys owned by `allocator`, the contract of mlx-serve's Weights).
pub fn load(allocator: std.mem.Allocator, f: *const gguf.File, out: *WeightMap, s: mlx.mlx_stream) !void {
    const arch = try Arch.read(f);
    // Only the qwen35 map yields un-tiling transforms, the rest never look at this.
    const hp = if (arch == .qwen35) arch.qwen35 else std.mem.zeroes(qwen35.Hparams);
    var name_buf: [160]u8 = undefined;
    var it = f.tensors.iterator();
    while (it.next()) |e| {
        const mapped = arch.mapTensor(&name_buf, e.key_ptr.*) orelse return error.UnsupportedTensor;
        if (mapped.name.len == 0) continue;
        const t = e.value_ptr.*;
        if (t.ty.isQuantized()) {
            try loadQuantized(allocator, t, mapped, hp, out, s);
        } else {
            try put(allocator, out, mapped.name, try loadFloat(allocator, t, mapped.transform, hp, s));
        }
    }
}

/// Runs the decode and the prefill kernel of every distinct (type, shape) in
/// `map` once, so Metal compiles them during load and not inside the first
/// request (a few hundred ms there).
pub fn warmKernels(allocator: std.mem.Allocator, map: *const WeightMap, s: mlx.mlx_stream) !void {
    var seen = std.AutoHashMap([4]c_int, void).init(allocator);
    defer seen.deinit();
    var it = map.iterator();
    while (it.next()) |e| {
        if (!std.mem.endsWith(u8, e.key_ptr.*, ".weight")) continue;
        var sc_name: [160]u8 = undefined;
        const base = e.key_ptr.*[0 .. e.key_ptr.len - "weight".len];
        const sc = map.get(std.fmt.bufPrint(&sc_name, "{s}scales", .{base}) catch continue) orelse continue;
        const w = e.value_ptr.*;
        const info = kernels.infoOf(w, sc) orelse continue;
        const ws = mlx.getShape(w);
        const bank = ws.len == 3;
        const key = [4]c_int{ @intCast(@backingInt(info.ty)), @intFromBool(bank), ws[ws.len - 2], ws[ws.len - 1] };
        if ((try seen.getOrPut(key)).found_existing) continue;

        const in: c_int = @intCast(@as(usize, @intCast(ws[ws.len - 1])) / info.ty.blockBytes() * info.ty.blockElems());
        // One row = the decode matvec, 16 = the prefill tile kernel.
        for ([_]c_int{ 1, 16 }) |m| {
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            const x_shape: []const c_int = if (bank) &.{ m, 1, in } else &.{ m, in };
            try mlx.check(mlx.mlx_zeros(&x, x_shape.ptr, x_shape.len, .bfloat16, s));
            const y = if (bank) blk: {
                var experts = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(experts);
                try mlx.check(mlx.mlx_zeros(&experts, &[_]c_int{m}, 1, .uint32, s));
                break :blk try kernels.gatherLinear(.{ .ty = info.ty }, x, w, .{ .ctx = null }, experts, s);
            } else try kernels.linear(.{ .ty = info.ty }, x, w, s);
            defer _ = mlx.mlx_array_free(y);
            try mlx.check(mlx.mlx_array_eval(y));
        }
    }
}

fn put(allocator: std.mem.Allocator, out: *WeightMap, name: []const u8, arr: mlx.mlx_array) !void {
    errdefer _ = mlx.mlx_array_free(arr);
    const key = try allocator.dupe(u8, name);
    errdefer allocator.free(key);
    try out.putNoClobber(key, arr);
}

fn loadQuantized(allocator: std.mem.Allocator, t: gguf.Tensor, mapped: arch_mod.Mapped, hp: qwen35.Hparams, out: *WeightMap, s: mlx.mlx_stream) !void {
    if (!std.mem.endsWith(u8, mapped.name, ".weight")) return error.UnsupportedTensor;
    const rows: usize = @intCast(t.dims[1]);
    if (t.n_dims == 3) {
        // An expert bank, experts outermost: [experts, rows, row bytes].
        if (mapped.transform != .none) return error.UnsupportedTensor;
        const experts: usize = @intCast(t.dims[2]);
        const bank = [_]c_int{ @intCast(experts), @intCast(rows), @intCast(t.data.len / experts / rows) };
        try put(allocator, out, mapped.name, try packedRows(allocator, t.ty, t.data, &bank, s));
        return putSentinel(allocator, out, mapped.name, .{ .ty = t.ty }, s);
    }
    if (t.n_dims != 2) return error.UnsupportedTensor;
    const row_bytes = t.data.len / rows;
    const shape = [_]c_int{ @intCast(rows), @intCast(row_bytes) };

    if (mapped.transform == .split_ba) {
        const g = mapped.transform.split_ba;
        const nv = g.nk * g.r;
        if (rows != 2 * nv) return error.BadShape;
        const marker = ".in_proj_ba.";
        const at = std.mem.indexOf(u8, mapped.name, marker) orelse return error.UnsupportedTensor;
        const tmp = try allocator.alloc(u8, nv * row_bytes);
        defer allocator.free(tmp);
        for ([_][]const u8{ ".in_proj_b.", ".in_proj_a." }, 0..) |half_name, half| {
            for (0..g.nk) |k| for (0..g.r) |j| {
                const src = (k * 2 * g.r + half * g.r + j) * row_bytes;
                @memcpy(tmp[(k * g.r + j) * row_bytes ..][0..row_bytes], t.data[src..][0..row_bytes]);
            };
            var name_buf: [160]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "{s}{s}{s}", .{ mapped.name[0..at], half_name, mapped.name[at + marker.len ..] });
            try put(allocator, out, name, mlx.mlx_array_new_data(tmp.ptr, &[_]c_int{ @intCast(nv), @intCast(row_bytes) }, 2, .uint8));
            try putSentinel(allocator, out, name, .{ .ty = t.ty }, s);
        }
        return;
    }

    var info = kernels.Info{ .ty = t.ty };
    const w = switch (mapped.transform) {
        .none => try packedRows(allocator, t.ty, t.data, &shape, s),
        .tiled_input => blk: {
            info.tiled = .{ .nk = @intCast(hp.nk), .r = @intCast(hp.r()) };
            break :blk mlx.mlx_array_new_data(t.data.ptr, &shape, 2, .uint8);
        },
        .untile_rows => |u| blk: {
            // Rows are whole blocks, so moving a row moves its bytes and nothing else.
            if (u.start + hp.nv * u.unit != rows) return error.BadShape;
            const tmp = try allocator.alloc(u8, t.data.len);
            defer allocator.free(tmp);
            untile(tmp, t.data, u.start * row_bytes, u.unit * row_bytes, hp);
            break :blk try packedRows(allocator, t.ty, tmp, &shape, s);
        },
        else => return error.UnsupportedTensor,
    };
    try put(allocator, out, mapped.name, w);
    try putSentinel(allocator, out, mapped.name, info, s);
}

/// The weight as an MLX array, rows in the split layout when the kernels want it (see kernels.isSplit).
fn packedRows(allocator: std.mem.Allocator, ty: @import("quants.zig").GgmlType, bytes: []const u8, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    _ = s;
    const row_bytes: usize = @intCast(shape[shape.len - 1]);
    const in = row_bytes / ty.blockBytes() * ty.blockElems();
    if (!kernels.isSplit(ty, in)) return mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), .uint8);
    const rows = bytes.len / row_bytes;
    const tmp = try allocator.alloc(u8, rows * kernels.rowBytes(ty, in));
    defer allocator.free(tmp);
    kernels.packRows(ty, in, tmp, bytes);
    var padded: [3]c_int = undefined;
    @memcpy(padded[0..shape.len], shape);
    padded[shape.len - 1] = @intCast(kernels.rowBytes(ty, in));
    return mlx.mlx_array_new_data(tmp.ptr, &padded, @intCast(shape.len), .uint8);
}

/// `<base>.scales` next to `<base>.weight`, see kernels.Info.
fn putSentinel(allocator: std.mem.Allocator, out: *WeightMap, weight_name: []const u8, info: kernels.Info, s: mlx.mlx_stream) !void {
    var sc_name: [160]u8 = undefined;
    const base = weight_name[0 .. weight_name.len - "weight".len];
    try put(allocator, out, try std.fmt.bufPrint(&sc_name, "{s}scales", .{base}), try kernels.sentinel(info, s));
}

/// Copy `src` to `dst` moving each tiled value-head chunk (`unit` bytes, the
/// region starting at `start`) to its grouped position.
fn untile(dst: []u8, src: []const u8, start: usize, unit: usize, hp: qwen35.Hparams) void {
    @memcpy(dst[0..start], src[0..start]);
    for (0..hp.nv) |tiled| {
        const grouped = qwen35.groupedIndex(@intCast(tiled), hp.nk, hp.r());
        @memcpy(dst[start + grouped * unit ..][0..unit], src[start + tiled * unit ..][0..unit]);
    }
}

fn loadFloat(allocator: std.mem.Allocator, t: gguf.Tensor, transform: arch_mod.Transform, hp: qwen35.Hparams, s: mlx.mlx_stream) !mlx.mlx_array {
    // ggml dims are innermost first.
    var shape_buf: [5]c_int = undefined;
    for (0..t.n_dims) |i| shape_buf[i] = @intCast(t.dims[t.n_dims - 1 - i]);
    var shape: []const c_int = shape_buf[0..t.n_dims];
    const dtype: mlx.mlx_dtype = switch (t.ty) {
        .f32 => .float32,
        .f16 => .float16,
        .bf16 => .bfloat16,
        else => unreachable,
    };

    const elem = t.ty.blockBytes();
    const raw = switch (transform) {
        .none => mlx.mlx_array_new_data(t.data.ptr, shape.ptr, @intCast(shape.len), dtype),
        .shared_gate => blk: {
            if (t.n_dims != 1) return error.BadShape;
            break :blk mlx.mlx_array_new_data(t.data.ptr, &[_]c_int{ 1, shape[0] }, 2, dtype);
        },
        .trailing_axis => blk: {
            if (t.n_dims + 1 > shape_buf.len) return error.BadShape;
            shape_buf[t.n_dims] = 1;
            shape = shape_buf[0 .. t.n_dims + 1];
            break :blk mlx.mlx_array_new_data(t.data.ptr, shape.ptr, @intCast(shape.len), dtype);
        },
        .minus_one, .neg_log => blk: {
            if (t.ty != .f32) return error.UnsupportedTensor;
            const tmp = try allocator.alloc(f32, t.elems());
            defer allocator.free(tmp);
            for (tmp, std.mem.bytesAsSlice(f32, t.data)) |*o, v| o.* = if (transform == .neg_log) @log(-v) else v - 1;
            const flat = [_]c_int{@intCast(t.elems())};
            break :blk mlx.mlx_array_new_data(tmp.ptr, if (transform == .neg_log) &flat else shape.ptr, if (transform == .neg_log) 1 else @intCast(shape.len), dtype);
        },
        .flatten => mlx.mlx_array_new_data(t.data.ptr, &[_]c_int{@intCast(t.elems())}, 1, dtype),
        .split_ba => return error.UnsupportedTensor,
        // Un-tiling only moves bytes, so it works the same on any float type.
        .untile_rows, .untile_vec, .a_log, .conv1d, .tiled_input => blk: {
            const tmp = try allocator.alignedAlloc(u8, .@"4", t.data.len);
            defer allocator.free(tmp);
            const row: usize = if (t.n_dims == 2) @intCast(t.dims[0]) else 1;
            switch (transform) {
                .untile_rows => |u| {
                    if (t.n_dims != 2 or u.start + hp.nv * u.unit != t.dims[1]) return error.BadShape;
                    untile(tmp, t.data, u.start * row * elem, u.unit * row * elem, hp);
                },
                // [channels, kernel]: q and k channels first, then the value channels.
                .conv1d => {
                    if (t.n_dims != 2 or 2 * hp.nk * hp.dk + hp.nv * hp.dv != t.dims[1]) return error.BadShape;
                    untile(tmp, t.data, 2 * hp.nk * hp.dk * row * elem, hp.dv * row * elem, hp);
                    shape_buf[t.n_dims] = 1;
                    shape = shape_buf[0 .. t.n_dims + 1];
                },
                // A float weight can simply have its columns moved, row by row.
                .tiled_input => {
                    if (t.n_dims != 2 or hp.nv * hp.dv != row) return error.BadShape;
                    for (0..@intCast(t.dims[1])) |r| untile(tmp[r * row * elem ..][0 .. row * elem], t.data[r * row * elem ..][0 .. row * elem], 0, hp.dv * elem, hp);
                },
                else => {
                    if (t.elems() != hp.nv) return error.BadShape;
                    untile(tmp, t.data, 0, elem, hp);
                },
            }
            if (transform == .a_log) {
                if (t.ty != .f32) return error.UnsupportedTensor;
                for (std.mem.bytesAsSlice(f32, tmp)) |*v| v.* = @log(-v.*);
            }
            break :blk mlx.mlx_array_new_data(tmp.ptr, shape.ptr, @intCast(shape.len), dtype);
        },
    };
    // A_log stays f32 like the reference checkpoints, the gate math runs in f32.
    if (transform == .a_log or transform == .neg_log or dtype == .bfloat16) return raw;
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

test "untile moves tiled value-head chunks to grouped order and keeps the prefix" {
    var hp = std.mem.zeroes(qwen35.Hparams);
    hp.nk = 2;
    hp.nv = 6;
    // prefix "PP", then heads tiled as g0j0 g1j0 g0j1 g1j1 g0j2 g1j2, two bytes each.
    const src = "PP" ++ "a0" ++ "b0" ++ "a1" ++ "b1" ++ "a2" ++ "b2";
    var dst: [src.len]u8 = undefined;
    untile(&dst, src, 2, 2, hp);
    try std.testing.expectEqualStrings("PPa0a1a2b0b1b2", &dst);
}

test "real model loads into a weight map (a real GGUF, env-gated)" {
    const path = std.mem.span(env.get(.MLX_SERVE_GGUF_TEST_MODEL) orelse return error.SkipZigTest);
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var f = try gguf.File.open(a, path);
    defer f.deinit();
    var map = WeightMap.init(a);
    defer {
        var it = map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            a.free(e.key_ptr.*);
        }
        map.deinit();
    }
    try load(a, &f, &map, s);
    try warmKernels(a, &map, s);

    const arch = try Arch.read(&f);
    const p = "language_model.model.";
    const prefix: []const u8 = switch (arch) {
        .gemma3, .lfm2, .gpt_oss, .qwen3next => "model.",
        .nemotron_h => "backbone.",
        else => p,
    };
    const norm_name = try std.fmt.allocPrint(a, "{s}{s}norm{s}.weight", .{ prefix, if (arch == .lfm2) "embedding_" else "", if (arch == .nemotron_h) "_f" else "" });
    defer a.free(norm_name);
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(map.get(norm_name).?));
    switch (arch) {
        .qwen35 => |hp| {
            const qkv = map.get(p ++ "layers.0.linear_attn.in_proj_qkv.weight").?;
            try std.testing.expectEqual(@as(c_int, @intCast(2 * hp.nk * hp.dk + hp.nv * hp.dv)), mlx.getShape(qkv)[0]);
            const out_sc = map.get(p ++ "layers.0.linear_attn.out_proj.scales").?;
            const out_w = map.get(p ++ "layers.0.linear_attn.out_proj.weight").?;
            try std.testing.expectEqual(hp.r() > 1, kernels.infoOf(out_w, out_sc).?.tiled != null);
            try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(map.get(p ++ "layers.0.linear_attn.A_log").?));
            try std.testing.expectEqualSlices(c_int, &.{ @intCast(2 * hp.nk * hp.dk + hp.nv * hp.dv), @intCast(hp.conv_kernel), 1 }, mlx.getShape(map.get(p ++ "layers.0.linear_attn.conv1d.weight").?));
        },
        .gemma4 => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{1}, mlx.getShape(map.get(p ++ "layers.0.layer_scalar").?));
            try std.testing.expect(map.get("rope_freqs.weight") == null);
            try std.testing.expectEqual(hp.ple_dim > 0, map.get(p ++ "embed_tokens_per_layer.weight") != null);
            try std.testing.expectEqual(hp.ple_dim > 0, map.get(p ++ "layers.0.per_layer_input_gate.weight") != null);
            const q = map.get(p ++ "layers.0.self_attn.q_proj.weight").?;
            try std.testing.expectEqual(@as(c_int, @intCast(hp.n_heads * hp.head_dim)), mlx.getShape(q)[0]);
        },
        .lfm2 => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{ @intCast(hp.hidden), @intCast(hp.conv_cache), 1 }, mlx.getShape(map.get("model.layers.0.conv.conv.weight").?));
            try std.testing.expectEqual(hp.moe, map.get("model.layers.2.feed_forward.switch_mlp.up_proj.weight") != null);
        },
        .qwen3next => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.nv)}, mlx.getShape(map.get("model.layers.0.linear_attn.A_log").?));
            try std.testing.expectEqual(@as(c_int, @intCast(hp.nv)), mlx.getShape(map.get("model.layers.0.linear_attn.in_proj_b.weight").?)[0]);
            try std.testing.expect(map.get("model.layers.0.linear_attn.in_proj_ba.weight") == null);
        },
        .nemotron_h => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.mamba_heads)}, mlx.getShape(map.get("backbone.layers.0.mixer.A_log").?));
            try std.testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(map.get("backbone.layers.0.mixer.A_log").?));
        },
        .gpt_oss => |hp| {
            try std.testing.expectEqualSlices(c_int, &.{@intCast(hp.n_heads)}, mlx.getShape(map.get("model.layers.0.self_attn.sinks").?));
            try std.testing.expect(map.get("model.layers.0.mlp.experts.gate_proj.bias") != null);
        },
        .gemma3 => |hp| {
            const q = map.get("model.layers.0.self_attn.q_proj.weight").?;
            try std.testing.expectEqual(@as(c_int, @intCast(hp.n_heads * hp.head_dim)), mlx.getShape(q)[0]);
            try std.testing.expect(map.get("model.layers.0.input_layernorm.weight") != null);
        },
    }
}
