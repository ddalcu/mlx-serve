//! sdk/quant/ternary.
//! Owns biasNegScalePeak, firstRowExactMiss, ternaryKernelCandidate, firstNonTernaryBias.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const qmv2 = @import("../blocks/qmm/qmv2.zig");
const rowqmv = @import("../blocks/qmm/rowqmv.zig");
const Weights = @import("../weights/resident.zig").Weights;
const computeQuantParams = @import("params.zig").computeQuantParams;
const QuantParams = @import("params.zig").QuantParams;

/// Binds every `<base>.signs` to `<base>.weight`. A sign vector without its
/// weight fails the load by name.
/// Prism's codec writes affine biases as exactly -scales (ternary codes
/// 0..2 -> -1..1); the verify-width kernel's fast path depends on it.
/// Lazy `max |scales + biases|` as f32, or null when either is absent.
pub fn biasNegScalePeak(weights: *const Weights, base: []const u8, s: mlx.mlx_stream) !?mlx.mlx_array {
    var nb: [256]u8 = undefined;
    const sc = weights.get(std.fmt.bufPrint(&nb, "{s}.scales", .{base}) catch return null) orelse return null;
    const bi = weights.get(std.fmt.bufPrint(&nb, "{s}.biases", .{base}) catch return null) orelse return null;
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, sc, bi, s));
    var mx_abs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx_abs);
    try mlx.check(mlx.mlx_abs(&mx_abs, sum, s));
    var peak = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(peak);
    try mlx.check(mlx.mlx_max(&peak, mx_abs, false, s));
    var peak32 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&peak32, peak, .float32, s));
    return peak32;
}

/// A pack qmv2's ternary kernel may serve once its biases check out: affine
/// 2-bit group-128 without rotations (Hadamard packs bind per weight).
/// The first quantized trunk projection the row-exact kernels cannot take
/// (not 4-bit affine in their geometry), or null when they cover the pack.
/// Vision towers, the MTP head and a gather-only embedding never reach a verify row.
pub fn firstRowExactMiss(model: QuantParams, weights: *const Weights, gather_only: mlx.mlx_array) ?[]const u8 {
    var it = weights.map.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        if (!std.mem.endsWith(u8, key, ".scales")) continue;
        if (std.mem.indexOf(u8, key, "vision") != null or std.mem.indexOf(u8, key, "visual") != null or
            std.mem.indexOf(u8, key, "mtp") != null) continue;
        const base = key[0 .. key.len - ".scales".len];
        var buf: [256]u8 = undefined;
        const w = weights.map.get(std.fmt.bufPrint(&buf, "{s}.weight", .{base}) catch return key) orelse continue;
        if (gather_only.ctx != null and w.ctx == gather_only.ctx) continue;
        const bi = weights.map.get(std.fmt.bufPrint(&buf, "{s}.biases", .{base}) catch return key) orelse return key;
        const qp = computeQuantParams(model, w, kv.value_ptr.*, null);
        if (qp.mode != .affine or !rowqmv.fits(w, kv.value_ptr.*, bi, qp.bits, qp.group_size)) return key;
    }
    return null;
}

pub fn ternaryKernelCandidate(hadamard_block: u32, model: QuantParams) bool {
    return hadamard_block == 0 and model.mode == .affine and model.bits == 2 and model.group_size == 128;
}

/// The first `<base>.biases` key whose biases are not exactly -scales, or null
/// when every one is (ONE eval). `gather_only` (an embedding table no matmul
/// reads) is skipped; a pack with no biases at all is not ternary.
pub fn firstNonTernaryBias(weights: *const Weights, gather_only: mlx.mlx_array, s: mlx.mlx_stream) !?[]const u8 {
    const peaks = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(peaks);
    var keys = std.ArrayList([]const u8).empty;
    defer keys.deinit(weights.allocator);
    var it = weights.map.iterator();
    while (it.next()) |kv| {
        const key = kv.key_ptr.*;
        if (!std.mem.endsWith(u8, key, ".biases")) continue;
        if (gather_only.ctx != null and kv.value_ptr.ctx == gather_only.ctx) continue;
        const peak = (try biasNegScalePeak(weights, key[0 .. key.len - ".biases".len], s)) orelse return key;
        defer _ = mlx.mlx_array_free(peak);
        try mlx.check(mlx.mlx_vector_array_append_value(peaks, peak));
        try keys.append(weights.allocator, key);
    }
    if (keys.items.len == 0) return "no quantized biases";
    try mlx.check(mlx.mlx_eval(peaks));
    for (keys.items, 0..) |key, i| {
        var a = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a);
        try mlx.check(mlx.mlx_vector_array_get(&a, peaks, i));
        var v: f32 = 1;
        try mlx.check(mlx.mlx_array_item_float32(&v, a));
        if (v != 0) return key;
    }
    return null;
}

// ── tests ──

test "firstNonTernaryBias: every matmul bias must be -scale; a gather-only embedding is exempt" {
    const s = mlx.gpuStream();
    const Put = struct {
        fn add(w: *Weights, name: []const u8, vals: []const f32) !mlx.mlx_array {
            const a = mlx.mlx_array_new_data(vals.ptr, &[_]c_int{@intCast(vals.len)}, 1, .float32);
            try w.map.put(try std.testing.allocator.dupe(u8, name), a);
            return a;
        }
    };
    const sc = [_]f32{ 0.5, 0.25 };
    const neg = [_]f32{ -0.5, -0.25 };
    const generic = [_]f32{ -0.5, 0.1 };
    {
        var w = Weights.init(std.testing.allocator);
        defer w.deinit();
        try std.testing.expectEqualStrings("no quantized biases", (try firstNonTernaryBias(&w, .{ .ctx = null }, s)).?);
        _ = try Put.add(&w, "m.q.scales", &sc);
        _ = try Put.add(&w, "m.q.biases", &neg);
        _ = try Put.add(&w, "m.embed_tokens.scales", &sc);
        const emb_b = try Put.add(&w, "m.embed_tokens.biases", &generic);
        // Untied: the table is only gathered. Tied (or the lm_head's fallback): it is a matmul weight.
        try std.testing.expectEqual(@as(?[]const u8, null), try firstNonTernaryBias(&w, emb_b, s));
        try std.testing.expectEqualStrings("m.embed_tokens.biases", (try firstNonTernaryBias(&w, .{ .ctx = null }, s)).?);
    }
    {
        var w = Weights.init(std.testing.allocator);
        defer w.deinit();
        _ = try Put.add(&w, "m.q.scales", &sc);
        _ = try Put.add(&w, "m.q.biases", &neg);
        _ = try Put.add(&w, "m.k.scales", &sc);
        _ = try Put.add(&w, "m.k.biases", &generic);
        _ = try Put.add(&w, "m.v.biases", &neg); // biases without scales
        const bad = (try firstNonTernaryBias(&w, .{ .ctx = null }, s)).?;
        try std.testing.expect(std.mem.eql(u8, bad, "m.k.biases") or std.mem.eql(u8, bad, "m.v.biases"));
    }
}

test "ternaryKernelCandidate: affine 2-bit group-128 packs without rotations" {
    try std.testing.expect(ternaryKernelCandidate(0, .{ .bits = 2, .group_size = 128, .mode = .affine }));
    try std.testing.expect(!ternaryKernelCandidate(1024, .{ .bits = 2, .group_size = 128, .mode = .affine }));
    try std.testing.expect(!ternaryKernelCandidate(0, .{ .bits = 2, .group_size = 64, .mode = .affine }));
    try std.testing.expect(!ternaryKernelCandidate(0, .{ .bits = 4, .group_size = 128, .mode = .affine }));
}
