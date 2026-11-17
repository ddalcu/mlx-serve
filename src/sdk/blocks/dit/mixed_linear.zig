//! A media loader's OWNED projection: dense bf16 or affine-quantized as the checkpoint stores it, with an optional
//! additive bias, stacked runtime LoRAs and the int8 copy `--w8a8` swaps in. One type loads a pack's bf16 build and
//! its quantized ones. (`linear.zig`'s `Linear` is the borrowed, adapter-less one.)

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const log = @import("../../util/log.zig");
const lora_mod = @import("../../weights/lora.zig");
const w8a8 = @import("w8a8.zig");
const Weights = @import("../../weights/load.zig").Weights;

const Graph = ops.Graph;
const Array = ops.Array;
const S = mlx.mlx_stream;

pub const QuantGeometry = struct { bits: u32, group_size: u32 };

/// (bits, group_size) of an affine weight from its packed geometry alone. `w` packs `in` columns into u32s
/// (w_cols = in·bits/32) beside s_cols = in/gs scales, so w_cols·32/s_cols = bits·gs; the product is ambiguous
/// ((3,64) vs (6,32)), so group sizes are tried in convention order: 64 (mflux, mlx-community), then 32 and 128.
pub fn inferQuantGeometry(w_cols: usize, s_cols: usize) QuantGeometry {
    const fallback: QuantGeometry = .{ .bits = 4, .group_size = 64 };
    if (s_cols == 0 or (w_cols * 32) % s_cols != 0) return fallback;
    const product = w_cols * 32 / s_cols;
    for ([_]u32{ 64, 32, 128 }) |gs| {
        if (product % gs != 0) continue;
        const bits: u32 = @intCast(product / gs);
        for ([_]u32{ 2, 3, 4, 5, 6, 8 }) |vb| {
            if (bits == vb) return .{ .bits = bits, .group_size = gs };
        }
    }
    return fallback;
}

fn own(w: *const Weights, key: []const u8) ?mlx.mlx_array {
    const a = w.get(key) orelse return null;
    var o = mlx.mlx_array_new();
    mlx.check(mlx.mlx_array_set(&o, a)) catch return null;
    return o;
}

/// `bf16_input`: cast the input to bf16 first, for a reference that computes there; otherwise the input's dtype
/// carries through.
pub fn MixedLinear(comptime bf16_input: bool) type {
    return struct {
        quantized: bool,
        /// Quantized: packed u32 `[out, in*bits/32]`. Dense: bf16, pre-transposed `[in, out]`.
        w: mlx.mlx_array,
        scales: mlx.mlx_array = .{ .ctx = null },
        /// The quant zero points (NOT the additive bias).
        biases: mlx.mlx_array = .{ .ctx = null },
        add_bias: ?mlx.mlx_array = null,
        bits: u32 = 0,
        group_size: u32 = 0,
        /// Runtime adapters, non-owning (the plugin's `lora.Stack` owns the arrays); fixed capacity, so attach and
        /// forward never allocate.
        lora_refs: [lora_mod.MAX_LORAS]lora_mod.Ref = undefined,
        lora_count: u8 = 0,
        /// The int8 copy that REPLACES `w`/`scales`/`biases` (`--w8a8`), built by `toW8a8`.
        w8: ?w8a8.Weight = null,

        const Self = @This();

        /// `<prefix>.weight` with `.scales` + `.biases` (affine) or alone (dense), and an optional `.bias`.
        /// `in_features` solves (bits, group_size) exactly; 0 infers them from the geometry (`inferQuantGeometry`).
        pub fn load(w: *const Weights, a: std.mem.Allocator, prefix: []const u8, in_features: u32, s: S) !Self {
            var kb: [4][]u8 = undefined;
            inline for (.{ "weight", "scales", "biases", "bias" }, &kb) |leaf, *k| k.* = try std.fmt.allocPrint(a, "{s}." ++ leaf, .{prefix});
            defer for (kb) |k| a.free(k);
            const weight = own(w, kb[0]) orelse {
                log.err("[weights] MISSING WEIGHT: {s}\n", .{kb[0]});
                return error.MissingWeight;
            };
            errdefer _ = mlx.mlx_array_free(weight);
            if (own(w, kb[1])) |scales| {
                errdefer _ = mlx.mlx_array_free(scales);
                const biases = own(w, kb[2]) orelse {
                    log.err("[weights] MISSING WEIGHT: {s}\n", .{kb[2]});
                    return error.MissingWeight;
                };
                const w_cols: u32 = @intCast(mlx.getShape(weight)[1]);
                const s_cols: u32 = @intCast(mlx.getShape(scales)[1]);
                const geo: QuantGeometry = if (in_features == 0)
                    inferQuantGeometry(w_cols, s_cols)
                else
                    .{ .bits = @intCast(@divExact(32 * w_cols, in_features)), .group_size = @intCast(@divExact(in_features, s_cols)) };
                return .{ .quantized = true, .w = weight, .scales = scales, .biases = biases, .add_bias = own(w, kb[3]), .bits = geo.bits, .group_size = geo.group_size };
            }
            // Dense: [out, in] → [in, out], materialized (a transposed view copies on every matmul), bf16.
            defer _ = mlx.mlx_array_free(weight);
            var t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(t);
            try mlx.check(mlx.mlx_transpose_axes(&t, weight, &[_]c_int{ 1, 0 }, 2, s));
            var tc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tc);
            try mlx.check(mlx.mlx_contiguous(&tc, t, false, s));
            var wt = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&wt, tc, .bfloat16, s));
            return .{ .quantized = false, .w = wt, .add_bias = own(w, kb[3]) };
        }

        pub fn deinit(self: *Self) void {
            if (self.w8) |*q| q.deinit();
            _ = mlx.mlx_array_free(self.w);
            if (self.quantized) {
                _ = mlx.mlx_array_free(self.scales);
                _ = mlx.mlx_array_free(self.biases);
            }
            if (self.add_bias) |b| _ = mlx.mlx_array_free(b);
        }

        pub fn forward(self: *const Self, parent: *Graph, x: Array) !Array {
            var g = parent.scope();
            defer g.deinit();
            const xin = if (bf16_input) try g.astype(x, .bfloat16) else x;
            var o = if (self.w8) |*q| try g.adopt(try w8a8.forward(try g.astype(xin, .bfloat16), q, g.s)) else try self.matmul(&g, xin);
            if (self.add_bias) |b| o = try g.add(o, b);
            return parent.keep(try self.addLora(&g, o, xin));
        }

        /// `o` plus every attached adapter's delta on `x`.
        pub fn addLora(self: *const Self, parent: *Graph, o: Array, x: Array) !Array {
            if (self.lora_count == 0) return o;
            var g = parent.scope();
            defer g.deinit();
            return parent.keep(try g.add(o, try g.adopt(try lora_mod.deltaSum(x, self.lora_refs[0..self.lora_count], g.s))));
        }

        pub fn matmul(self: *const Self, g: *Graph, x: Array) !Array {
            if (self.w.ctx == null) return error.WeightReleased;
            if (self.quantized) return g.qmm(x, self.w, self.scales, self.biases, @intCast(self.group_size), @intCast(self.bits));
            return g.matmul(x, self.w);
        }

        pub fn outFeatures(self: *const Self) u32 {
            if (self.w8) |q| return @intCast(q.n);
            return @intCast(if (self.quantized) mlx.getShape(self.w)[0] else mlx.getShape(self.w)[1]);
        }

        /// Replace the weights with their int8 copy when the route takes this shape.
        pub fn toW8a8(self: *Self, s: S) !void {
            self.w8 = try w8a8.takeLinear(&self.w, &self.scales, &self.biases, self.bits, self.group_size, s);
        }

        /// Install the stacked adapter Refs for this linear (from `Stack.findAll`).
        pub fn setLoraRefs(self: *Self, refs: []const lora_mod.Ref) void {
            self.lora_count = @intCast(refs.len);
            @memcpy(self.lora_refs[0..refs.len], refs);
        }

        pub fn clearLoraRefs(self: *Self) void {
            self.lora_count = 0;
        }
    };
}

// ── tests ──

const testing = std.testing;

test "inferQuantGeometry resolves the mflux and mlx-community bit widths" {
    // in = 3072 logical columns at gs = 64 → s_cols = 48; w_cols = in·bits/32.
    try testing.expectEqual(@as(u32, 3), inferQuantGeometry(288, 48).bits); // 3-bit (8 GB iPhones)
    try testing.expectEqual(@as(u32, 64), inferQuantGeometry(288, 48).group_size);
    try testing.expectEqual(@as(u32, 4), inferQuantGeometry(384, 48).bits);
    try testing.expectEqual(@as(u32, 5), inferQuantGeometry(480, 48).bits);
    try testing.expectEqual(@as(u32, 6), inferQuantGeometry(576, 48).bits);
    try testing.expectEqual(@as(u32, 8), inferQuantGeometry(768, 48).bits);
    // gs-128 conversion of the same 4-bit width: s_cols = 24 → product 512;
    // 512/64 = 8 valid → convention prefers 64 (correct for every catalog
    // repo; a real gs-128 build would need explicit metadata to win here).
    try testing.expectEqual(@as(u32, 8), inferQuantGeometry(384, 24).bits);
    // Degenerate/odd geometry falls back to the historical 4/64.
    try testing.expectEqual(@as(u32, 4), inferQuantGeometry(0, 0).bits);
    try testing.expectEqual(@as(u32, 4), inferQuantGeometry(383, 48).bits);
}

test "MixedLinear: a dense weight and its affine twin load from one key shape and agree" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const kit = @import("testkit.zig");
    const x = try kit.randn(&g, &.{ 1, 5, 64 }, 1, .float32, 1);
    const w = try kit.randn(&g, &.{ 32, 64 }, 2, .bfloat16, 0.1);
    const b = try kit.randn(&g, &.{32}, 3, .bfloat16, 0.1);

    // The affine twin, and the dense weight it decodes to.
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, w, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));
    var parts: [3]Array = undefined;
    for (&parts, 0..) |*p, i| {
        p.* = mlx.mlx_array_new();
        _ = mlx.mlx_vector_array_get(p, qv, i);
        _ = try g.adopt(p.*);
    }
    var dq = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_dequantize(&dq, parts[0], parts[1], parts[2], mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, .{ .value = .bfloat16, .has_value = true }, s));
    _ = try g.adopt(dq);

    // A map owns what it is handed: give each its own reference.
    const ref = struct {
        fn f(src: Array) !Array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&o, src));
            return o;
        }
    }.f;
    var dense = Weights.init(testing.allocator);
    defer dense.deinit();
    try dense.put("lin.weight", try ref(dq));
    try dense.put("lin.bias", try ref(b));
    var affine = Weights.init(testing.allocator);
    defer affine.deinit();
    try affine.put("lin.weight", try ref(parts[0]));
    try affine.put("lin.scales", try ref(parts[1]));
    try affine.put("lin.biases", try ref(parts[2]));
    try affine.put("lin.bias", try ref(b));

    const L = MixedLinear(false);
    var d = try L.load(&dense, testing.allocator, "lin", 0, s);
    defer d.deinit();
    var q = try L.load(&affine, testing.allocator, "lin", 0, s);
    defer q.deinit();
    try testing.expect(!d.quantized and q.quantized and q.bits == 8 and q.group_size == 64);
    try testing.expectEqual(@as(u32, 32), d.outFeatures());
    try testing.expectEqual(@as(u32, 32), q.outFeatures());
    const yd = try d.forward(&g, x);
    const yq = try q.forward(&g, x);
    // The input's dtype carries through both arms; the cast variant answers in bf16.
    try testing.expect(mlx.mlx_array_dtype(yd) == .float32 and mlx.mlx_array_dtype(yq) == .float32);
    // The dense arm is the plain projection; the affine arm is apart only by the dense copy's bf16 rounding.
    const maxAbsDiff = @import("../../testing/parity.zig").maxAbsDiffF32;
    try testing.expect(try maxAbsDiff(yd, try g.add(try g.matmul(x, try g.transpose(dq, &.{ 1, 0 })), b), s) < 1e-5);
    try testing.expect(try maxAbsDiff(yd, yq, s) < 5e-2);

    var cast = try MixedLinear(true).load(&dense, testing.allocator, "lin", 0, s);
    defer cast.deinit();
    try testing.expect(mlx.mlx_array_dtype(try cast.forward(&g, x)) == .bfloat16);
    try testing.expectError(error.MissingWeight, L.load(&dense, testing.allocator, "absent", 0, s));
}
