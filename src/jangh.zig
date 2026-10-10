// SPDX-License-Identifier: Apache-2.0
// Ported from vMLX (jjang-ai/vmlx) vmlx_engine/jangh/contract.py, payload.py and
// vmlx_engine/models/qwen4_exp/loader.py @ 5f007e6d.
//! JANGH bundles on qwen4_exp: routed experts in the `jangtq2` codebook format beside an affine
//! trunk, under the converter's own tensor names. The format contract, the map from those names
//! onto the layout our qwen4_exp loader reads, and the tensor moves that follow the load.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const qwen4_ple = @import("qwen4_ple.zig");

const Weights = model_mod.Weights;
const ModelConfig = model_mod.ModelConfig;

/// The odd-cubic codebook: level(q) = u * (alpha + beta * u^2), u = q - (2^bits - 1) / 2. The
/// kernels evaluate it arithmetically, so a bundle must declare exactly these coefficients.
const Cubic = struct { bits: u8, name: []const u8, alpha: f64, beta: f64 };
const CUBIC_PARAMS = [_]Cubic{
    .{ .bits = 2, .name = "2", .alpha = 0.8929999999999999, .beta = 0.05065 },
    .{ .bits = 3, .name = "3", .alpha = 0.47124999999999995, .beta = 0.011599999999999997 },
    .{ .bits = 4, .name = "4", .alpha = 0.2405, .beta = 0.002100000000000002 },
    .{ .bits = 6, .name = "6", .alpha = 0.10413533834586466, .beta = 0.0 },
    .{ .bits = 8, .name = "8", .alpha = 0.030780075187969925, .beta = 0.0 },
};

const MAX_LAYERS = 128;

/// One trunk layer's routed bank. Gate and up share a width: one fused kernel reads both.
pub const Layer = struct { gate_up_bits: u8, down_bits: u8, rotated: bool };

/// A JANGH bundle's routed-expert contract, per trunk layer.
pub const Spec = struct {
    layers: [MAX_LAYERS]Layer = @splat(.{ .gate_up_bits = 0, .down_bits = 0, .rotated = false }),
    /// The text config's `swiglu_limit`; 0 is the plain SwiGLU.
    swiglu_limit: f32 = 0,
};

/// The routed-expert contract a config declares: the `jangtq` block (version 2, LSB bitstream, f16 row
/// scales, odd-cubic codebooks at the kernels' coefficients) and `{gate,up,down}_proj` entries for every
/// trunk layer (vMLX `validate_format`). Null when it declares neither; what the kernels cannot serve is refused.
pub fn parseSpec(root: std.json.ObjectMap, cfg_obj: std.json.ObjectMap, num_layers: u32) !?Spec {
    const quant = objectField(root, "quantization");
    const decl = field(root, "jangtq") orelse {
        if (quant) |q| if (anyCodebookEntry(q)) return error.Jangtq2Undeclared;
        return null;
    };
    if (decl != .object) return error.Jangtq2Declaration;
    const version = decl.object.get("version") orelse return error.Jangtq2Version;
    if (version != .integer or version.integer != 2) return error.Jangtq2Version;
    inline for (.{ .{ "packing", "lsb-bitstream" }, .{ "scale_dtype", "float16" }, .{ "codebook_family", "odd-cubic" } }) |want| {
        const v = decl.object.get(want[0]) orelse return error.Jangtq2Format;
        if (v != .string or !std.mem.eql(u8, v.string, want[1])) return error.Jangtq2Format;
    }
    const default_rotated = try rotatedOf(decl.object.get("rotation"), false);
    const books = try declaredCodebooks(decl.object.get("codebooks") orelse return error.Jangtq2Codebook);
    const q = quant orelse return error.Jangtq2Quantization;
    if (num_layers == 0 or num_layers > MAX_LAYERS) return error.Jangtq2Layers;

    const Proj2 = struct { bits: u8, rotated: bool };
    var seen: [MAX_LAYERS][3]?Proj2 = @splat(@splat(null));
    var it = q.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .object) continue; // the global `bits` / `group_size`
        const entry = e.value_ptr.object;
        if (!isCodebookEntry(entry)) {
            try checkAffineEntry(entry);
            continue;
        }
        const at = try routedSlot(e.key_ptr.*, num_layers);
        const p: Proj2 = .{ .bits = try bitsOf(entry.get("bits"), books), .rotated = try rotatedOf(entry.get("rotation"), default_rotated) };
        if (seen[at.layer][at.proj]) |prior| if (!std.meta.eql(prior, p)) return error.Jangtq2Alias;
        seen[at.layer][at.proj] = p;
    }
    var spec: Spec = .{ .swiglu_limit = try swigluLimit(cfg_obj) };
    for (seen[0..num_layers], spec.layers[0..num_layers]) |slots, *layer| {
        const gate = slots[0] orelse return error.Jangtq2IncompleteStack;
        const up = slots[1] orelse return error.Jangtq2IncompleteStack;
        const down = slots[2] orelse return error.Jangtq2IncompleteStack;
        if (!std.meta.eql(gate, up)) return error.Jangtq2GateUpMismatch;
        // The kernels take one rotation per bank.
        if (down.rotated != gate.rotated) return error.Jangtq2MixedRotation;
        layer.* = .{ .gate_up_bits = gate.bits, .down_bits = down.bits, .rotated = gate.rotated };
    }
    return spec;
}

/// Whether a config declares JANGTQ banks in any form.
pub fn declared(root: std.json.ObjectMap) bool {
    if (field(root, "jangtq") != null) return true;
    const q = objectField(root, "quantization") orelse return false;
    return anyCodebookEntry(q);
}

/// A JANG bundle's norm storage: the converter applied the runtime's `+1` to every shifted norm, the only
/// convention vMLX's qwen4_exp loader accepts.
pub fn checkNormConvention(root: std.json.ObjectMap) !void {
    const jc = field(root, "jang_config") orelse field(root, "jang") orelse return error.JangNormConventionMissing;
    if (jc != .object) return error.JangNormConventionMissing;
    const nc = field(jc.object, "norm_convention") orelse return error.JangNormConventionMissing;
    if (nc != .string or !std.mem.eql(u8, std.mem.trim(u8, nc.string, " \t\r\n"), "runtime_plus1_applied")) return error.UnsupportedJangNormConvention;
}

/// A JSON null reads as an absent key, as Python's `dict.get` sees it.
fn field(obj: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = obj.get(key) orelse return null;
    return if (v == .null) null else v;
}

fn objectField(obj: std.json.ObjectMap, key: []const u8) ?std.json.ObjectMap {
    const v = field(obj, key) orelse return null;
    return if (v == .object) v.object else null;
}

fn isCodebookEntry(entry: std.json.ObjectMap) bool {
    const mode = entry.get("mode") orelse return false;
    return mode == .string and std.mem.eql(u8, mode.string, "jangtq2");
}

fn anyCodebookEntry(q: std.json.ObjectMap) bool {
    var it = q.iterator();
    while (it.next()) |e| if (e.value_ptr.* == .object and isCodebookEntry(e.value_ptr.object)) return true;
    return false;
}

/// A trunk module beside the codebook banks: affine, at a width and group size MLX runs.
fn checkAffineEntry(entry: std.json.ObjectMap) !void {
    if (entry.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return error.Jangtq2TrunkEntry;
    const bits = entry.get("bits") orelse return error.Jangtq2TrunkEntry;
    const gs = entry.get("group_size") orelse return error.Jangtq2TrunkEntry;
    if (bits != .integer or gs != .integer) return error.Jangtq2TrunkEntry;
    switch (bits.integer) {
        2, 3, 4, 5, 6, 8 => {},
        else => return error.Jangtq2TrunkEntry,
    }
    switch (gs.integer) {
        32, 64, 128 => {},
        else => return error.Jangtq2TrunkEntry,
    }
}

const Slot = struct { layer: u32, proj: u2 };

/// `model.layers.N.mlp.switch_mlp.<proj>` under any of the converters' roots.
fn routedSlot(path: []const u8, num_layers: u32) !Slot {
    if (std.mem.indexOf(u8, path, ".switch_mlp.") == null) return error.Jangtq2DenseProjection;
    const rest = for ([_][]const u8{ "language_model.model.", "model.language_model.", "language_model.", "model." }) |p| {
        if (std.mem.startsWith(u8, path, p)) break path[p.len..];
    } else return error.Jangtq2Stack;
    if (!std.mem.startsWith(u8, rest, "layers.")) return error.Jangtq2Stack;
    const after = rest["layers.".len..];
    const dot = std.mem.indexOfScalar(u8, after, '.') orelse return error.Jangtq2Stack;
    const layer = parseIndex(after[0..dot]) orelse return error.Jangtq2Stack;
    if (layer >= num_layers) return error.Jangtq2Stack;
    const container = ".mlp.switch_mlp.";
    if (!std.mem.startsWith(u8, after[dot..], container)) return error.Jangtq2Stack;
    const name = after[dot + container.len ..];
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, i| {
        if (std.mem.eql(u8, name, p)) return .{ .layer = layer, .proj = @intCast(i) };
    }
    return error.Jangtq2Projection;
}

fn parseIndex(digits: []const u8) ?u32 {
    if (digits.len == 0 or (digits.len > 1 and digits[0] == '0')) return null;
    for (digits) |c| if (c < '0' or c > '9') return null;
    return std.fmt.parseInt(u32, digits, 10) catch null;
}

fn bitsOf(v: ?std.json.Value, books: u16) !u8 {
    const b = v orelse return error.Jangtq2Bits;
    if (b != .integer) return error.Jangtq2Bits;
    for (CUBIC_PARAMS) |c| {
        if (b.integer == c.bits and books & (@as(u16, 1) << @intCast(c.bits)) != 0) return c.bits;
    }
    return error.Jangtq2Bits;
}

fn rotatedOf(v: ?std.json.Value, default: bool) !bool {
    const r = v orelse return default;
    if (r != .string) return error.Jangtq2Rotation;
    if (std.mem.eql(u8, r.string, "hadamard32")) return true;
    if (std.mem.eql(u8, r.string, "none")) return false;
    return error.Jangtq2Rotation;
}

/// Bit `b` set for each declared width, every level checked against the runtime's codebook.
/// Widths 2, 3 and 4 are always declared; 6 and 8 only when a projection uses them.
fn declaredCodebooks(v: std.json.Value) !u16 {
    if (v != .object) return error.Jangtq2Codebook;
    var mask: u16 = 0;
    var it = v.object.iterator();
    while (it.next()) |e| {
        const cubic = for (CUBIC_PARAMS) |c| {
            if (std.mem.eql(u8, e.key_ptr.*, c.name)) break c;
        } else return error.Jangtq2Codebook;
        try checkCodebook(e.value_ptr.*, cubic);
        mask |= @as(u16, 1) << @intCast(cubic.bits);
    }
    for ([_]u4{ 2, 3, 4 }) |b| if (mask & (@as(u16, 1) << b) == 0) return error.Jangtq2Codebook;
    return mask;
}

fn checkCodebook(v: std.json.Value, c: Cubic) !void {
    if (v != .object) return error.Jangtq2Codebook;
    if (try number(v.object.get("alpha")) != c.alpha or try number(v.object.get("beta")) != c.beta) return error.Jangtq2Codebook;
    const levels = v.object.get("levels") orelse return error.Jangtq2Codebook;
    const n = @as(usize, 1) << @intCast(c.bits);
    if (levels != .array or levels.array.items.len != n) return error.Jangtq2Codebook;
    const mid = @as(f64, @floatFromInt(n - 1)) / 2.0;
    for (levels.array.items, 0..) |level, i| {
        const u = @as(f64, @floatFromInt(i)) - mid;
        if (try toF32(try number(level)) != try toF32(u * (c.alpha + c.beta * u * u))) return error.Jangtq2Codebook;
    }
}

fn number(v: ?std.json.Value) !f64 {
    const x: f64 = switch (v orelse return error.Jangtq2Codebook) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.Jangtq2Codebook,
    };
    if (!std.math.isFinite(x)) return error.Jangtq2Codebook;
    return x;
}

fn toF32(x: f64) !f32 {
    if (@abs(x) > std.math.floatMax(f32)) return error.Jangtq2Codebook;
    return @floatCast(x);
}

fn swigluLimit(cfg_obj: std.json.ObjectMap) !f32 {
    const v = cfg_obj.get("swiglu_limit") orelse return 0;
    const x: f64 = switch (v) {
        .null => return 0,
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return error.Jangtq2SwigluLimit,
    };
    if (!std.math.isFinite(x) or x < 0) return error.Jangtq2SwigluLimit;
    return @floatCast(x);
}

/// The bundle's n-gram table (`ple.ngram_embedding.shards.N.{weight,scales,biases}`) and the hash
/// constants beside it (`qwen4_ple.HashBuffer`), read from the shards in place: never part of the weight map.
fn isNgramTable(key: []const u8) bool {
    if (std.mem.indexOf(u8, key, ".ple.ngram_embedding.") != null or
        std.mem.indexOf(u8, key, ".ple.ple_embedding.ngram_embedding.") != null) return true;
    inline for (comptime std.enums.values(qwen4_ple.HashBuffer)) |which| {
        if (std.mem.endsWith(u8, key, ".ple." ++ @tagName(which))) return true;
    }
    return false;
}

/// The name our qwen4_exp loader reads a bundle tensor under, written into `buf`; null for the n-gram
/// table and its hash constants (read in place) and the quantized vision tower (`qwen_vision.zig` runs
/// dense towers only). An unknown root is refused: nothing would read it.
pub fn runtimeName(buf: []u8, key: []const u8) !?[]const u8 {
    if (isNgramTable(key) or std.mem.startsWith(u8, key, "visual.")) return null;
    const lm = "language_model.";
    var head: []const u8 = "";
    var body: []const u8 = key;
    const nested = for ([_][]const u8{ "language_model.model.", "language_model.mtp.", "language_model.lm_head." }) |p| {
        if (std.mem.startsWith(u8, key, p)) break true;
    } else false;
    if (nested) {} else if (std.mem.startsWith(u8, key, "mtp.") or std.mem.startsWith(u8, key, "lm_head.") or
        (std.mem.startsWith(u8, key, "model.layers.") and std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null))
    {
        head = lm;
    } else if (std.mem.startsWith(u8, key, lm)) {
        head = "language_model.model.";
        body = key[lm.len..];
    } else {
        log.err("[jangh] no binding reads tensor {s}\n", .{key});
        return error.JanghTensorName;
    }
    const conv = ".ple.conv1d_weight";
    if (std.mem.endsWith(u8, body, conv)) {
        return std.fmt.bufPrint(buf, "{s}{s}.ple.conv1d.weight", .{ head, body[0 .. body.len - conv.len] }) catch return error.NameTooLong;
    }
    return std.fmt.bufPrint(buf, "{s}{s}", .{ head, body }) catch return error.NameTooLong;
}

/// The loaded tensors where the binders read them: the PLE conv as `[C, K, 1]` and each MTP layer's
/// fused `experts.gate_up_proj` split along its output rows (gate first) into `switch_mlp` gate and up banks.
pub fn adapt(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    try reshapePleConv(config, weights, s);
    try splitMtpExperts(config, weights, s);
}

fn reshapePleConv(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    var nb: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.conv1d.weight", .{ config.weight_prefix, config.ple_layer_idx });
    const w = weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
    const c: c_int = @intCast(config.hc_count * config.hidden_size);
    const k: c_int = @intCast(config.ple_conv_kernel);
    const shape = mlx.getShape(w);
    if (std.mem.eql(c_int, shape, &.{ c, k, 1 })) return;
    if (!std.mem.eql(c_int, shape, &.{ c, k })) return error.JanghPleConvShape;
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, w, &.{ c, k, 1 }, 3, s));
    weights.replace(name, r);
}

fn splitMtpExperts(config: *const ModelConfig, weights: *Weights, s: mlx.mlx_stream) !void {
    var layer: u32 = 0;
    while (true) : (layer += 1) {
        var probe: [256]u8 = undefined;
        if (weights.get(try std.fmt.bufPrint(&probe, "language_model.mtp.layers.{d}.mlp.experts.gate_up_proj.weight", .{layer})) == null) break;
        for ([_][]const u8{ "weight", "scales", "biases" }) |leaf| {
            var fb: [256]u8 = undefined;
            var db: [256]u8 = undefined;
            const fused_name = try std.fmt.bufPrint(&fb, "language_model.mtp.layers.{d}.mlp.experts.gate_up_proj.{s}", .{ layer, leaf });
            const down_name = try std.fmt.bufPrint(&db, "language_model.mtp.layers.{d}.mlp.experts.down_proj.{s}", .{ layer, leaf });
            const fused = weights.get(fused_name) orelse return error.JanghMtpExperts;
            if (weights.get(down_name) == null) return error.JanghMtpExperts;
            const shape = mlx.getShape(fused);
            if (shape.len != 3 or shape[0] != @as(c_int, @intCast(config.num_experts)) or shape[1] != 2 * @as(c_int, @intCast(config.moe_intermediate_size))) return error.JanghMtpExperts;
            const rows = @divExact(shape[1], 2);
            var halves: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
            errdefer for (halves) |h| {
                if (h.ctx != null) _ = mlx.mlx_array_free(h);
            };
            for (&halves, 0..) |*h, i| {
                const r0 = rows * @as(c_int, @intCast(i));
                var view = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(view);
                try mlx.check(mlx.mlx_slice(&view, fused, &.{ 0, r0, 0 }, 3, &.{ shape[0], r0 + rows, shape[2] }, 3, &.{ 1, 1, 1 }, 3, s));
                // The banks are read as row-contiguous buffers; the fused parent is dropped below.
                try mlx.check(mlx.mlx_contiguous(h, view, false, s));
            }
            const vec = mlx.mlx_vector_array_new_data(&halves, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_eval(vec));
            for (&halves, [_][]const u8{ "gate_proj", "up_proj" }) |*h, proj| {
                var nb: [256]u8 = undefined;
                try putNew(weights, try std.fmt.bufPrint(&nb, "language_model.mtp.layers.{d}.mlp.switch_mlp.{s}.{s}", .{ layer, proj, leaf }), h.*);
                h.* = .{ .ctx = null };
            }
            weights.remove(fused_name);
            var nb: [256]u8 = undefined;
            try rename(weights, down_name, try std.fmt.bufPrint(&nb, "language_model.mtp.layers.{d}.mlp.switch_mlp.down_proj.{s}", .{ layer, leaf }));
        }
    }
}

/// Hand the map `arr` under a name it does not hold yet (the map frees both from then on).
fn putNew(weights: *Weights, name: []const u8, arr: mlx.mlx_array) !void {
    if (weights.map.contains(name)) return error.JanghDuplicateTensor;
    const key = try weights.allocator.dupe(u8, name);
    errdefer weights.allocator.free(key);
    try weights.map.put(key, arr);
}

fn rename(weights: *Weights, from: []const u8, to: []const u8) !void {
    if (weights.map.contains(to)) return error.JanghDuplicateTensor;
    const kv = weights.map.fetchRemove(from) orelse return error.MissingWeight;
    weights.allocator.free(kv.key);
    putNew(weights, to, kv.value) catch |err| {
        _ = mlx.mlx_array_free(kv.value);
        return err;
    };
}

/// A bank as stored: uint32 `[E, N, K*bits/32]` packed rows beside f16 `[E, N]` row scales. K is a
/// multiple of 32: every 32 values fill `bits` whole words, and the rotation runs over 32-wide blocks.
pub fn bankAdmitted(packed_w: mlx.mlx_array, scales: mlx.mlx_array, experts: u32, out: u32, in: u32, bits: u8) bool {
    if (packed_w.ctx == null or scales.ctx == null or in % 32 != 0) return false;
    if (mlx.mlx_array_dtype(packed_w) != .uint32 or mlx.mlx_array_dtype(scales) != .float16) return false;
    const e: c_int = @intCast(experts);
    const n: c_int = @intCast(out);
    const words: c_int = @intCast(in / 32 * bits);
    return std.mem.eql(c_int, mlx.getShape(packed_w), &.{ e, n, words }) and std.mem.eql(c_int, mlx.getShape(scales), &.{ e, n });
}

// ── Tests ──

const testing = std.testing;

/// A two-layer qwen4_exp JANGH config: gate/up 4-bit and down 6-bit Hadamard-32 banks beside one
/// affine entry, with a vision tower and the JANG norm stamp.
fn testConfigText(a: std.mem.Allocator) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    const w = &out.writer;
    try w.writeAll("{\"model_type\":\"qwen4_exp\",\"vision_config\":{\"depth\":27,\"hidden_size\":1152,\"num_heads\":16,\"out_hidden_size\":2560}," ++
        "\"jang_config\":{\"format\":\"jangtq2\",\"norm_convention\":\"runtime_plus1_applied\"}," ++
        "\"text_config\":{\"model_type\":\"qwen4_exp_text\",\"hidden_size\":2560,\"num_hidden_layers\":2,\"ple_layer_ids\":[2]," ++
        "\"num_attention_heads\":24,\"num_key_value_heads\":2,\"head_dim\":256,\"full_attention_interval\":2,\"num_experts\":512," ++
        "\"num_experts_per_tok\":8,\"moe_intermediate_size\":640,\"vocab_size\":248320,\"eos_token_id\":248044}," ++
        "\"jangtq\":{\"version\":2,\"packing\":\"lsb-bitstream\",\"scale_dtype\":\"float16\"," ++
        "\"rotation\":\"hadamard32\",\"codebook_family\":\"odd-cubic\",\"codebooks\":{");
    for (CUBIC_PARAMS[0..4], 0..) |c, i| {
        if (i > 0) try w.writeAll(",");
        try w.print("\"{s}\":{{\"alpha\":{d},\"beta\":{d},\"levels\":[", .{ c.name, c.alpha, c.beta });
        const n = @as(usize, 1) << @intCast(c.bits);
        for (0..n) |q| {
            const u = @as(f64, @floatFromInt(q)) - @as(f64, @floatFromInt(n - 1)) / 2.0;
            try w.print("{s}{d}", .{ if (q > 0) "," else "", @as(f32, @floatCast(u * (c.alpha + c.beta * u * u))) });
        }
        try w.writeAll("]}");
    }
    try w.writeAll("}},\"quantization\":{\"group_size\":64,\"bits\":8,\"language_model.layers.0.self_attn.q_proj\":{\"group_size\":64,\"bits\":8}");
    for (0..2) |li| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |p| {
        const bits: u8 = if (std.mem.eql(u8, p, "down_proj")) 6 else 4;
        try w.print(",\"model.layers.{d}.mlp.switch_mlp.{s}\":{{\"bits\":{d},\"mode\":\"jangtq2\",\"rotation\":\"hadamard32\"}}", .{ li, p, bits });
    };
    try w.writeAll("}}");
    return out.toOwnedSlice();
}

fn testRoot(a: std.mem.Allocator) !std.json.ObjectMap {
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, try testConfigText(a), .{})).object;
}

fn testSpec(root: std.json.ObjectMap) !?Spec {
    return parseSpec(root, root.get("text_config").?.object, 2);
}

fn entryOf(root: std.json.ObjectMap, name: []const u8) *std.json.ObjectMap {
    return &root.get("quantization").?.object.getPtr(name).?.object;
}

test "jangh spec: a complete declaration reads one bank per layer" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const spec = (try testSpec(try testRoot(arena.allocator()))).?;
    for (spec.layers[0..2]) |l| try testing.expectEqual(Layer{ .gate_up_bits = 4, .down_bits = 6, .rotated = true }, l);
    try testing.expectEqual(@as(f32, 0), spec.swiglu_limit);
}

test "jangh spec: a config declaring no codebook banks is not a JANGH bundle" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"text_config\":{},\"quantization\":{\"group_size\":64,\"bits\":4}}", .{})).object;
    try testing.expect(try testSpec(root) == null);
    try testing.expect(!declared(root));
    const nulls = (try std.json.parseFromSliceLeaky(std.json.Value, a, "{\"text_config\":{},\"jangtq\":null,\"quantization\":null}", .{})).object;
    try testing.expect(try testSpec(nulls) == null);
    try testing.expect(!declared(nulls));
}

test "jangh spec: each corruption of the declaration is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    { // a codebook coefficient that differs from the kernels'
        const root = try testRoot(a);
        root.get("jangtq").?.object.get("codebooks").?.object.getPtr("4").?.object.getPtr("alpha").?.* = .{ .float = 0.2406 };
        try testing.expectError(error.Jangtq2Codebook, testSpec(root));
    }
    { // one level off by one f32 step
        const root = try testRoot(a);
        const levels = root.get("jangtq").?.object.get("codebooks").?.object.get("3").?.object.get("levels").?.array.items;
        levels[5] = .{ .float = @as(f64, std.math.nextAfter(f32, @floatCast(levels[5].float), 10)) };
        try testing.expectError(error.Jangtq2Codebook, testSpec(root));
    }
    { // a required width missing
        const root = try testRoot(a);
        _ = root.get("jangtq").?.object.getPtr("codebooks").?.object.orderedRemove("3");
        try testing.expectError(error.Jangtq2Codebook, testSpec(root));
    }
    { // a missing up_proj entry
        const root = try testRoot(a);
        _ = root.getPtr("quantization").?.object.orderedRemove("model.layers.1.mlp.switch_mlp.up_proj");
        try testing.expectError(error.Jangtq2IncompleteStack, testSpec(root));
    }
    { // gate and up at different widths
        const root = try testRoot(a);
        entryOf(root, "model.layers.0.mlp.switch_mlp.up_proj").getPtr("bits").?.* = .{ .integer = 3 };
        try testing.expectError(error.Jangtq2GateUpMismatch, testSpec(root));
    }
    { // a width with no declared codebook
        const root = try testRoot(a);
        entryOf(root, "model.layers.1.mlp.switch_mlp.down_proj").getPtr("bits").?.* = .{ .integer = 8 };
        try testing.expectError(error.Jangtq2Bits, testSpec(root));
    }
    { // a width the format has no codebook for
        const root = try testRoot(a);
        entryOf(root, "model.layers.1.mlp.switch_mlp.down_proj").getPtr("bits").?.* = .{ .integer = 5 };
        try testing.expectError(error.Jangtq2Bits, testSpec(root));
    }
    { // an unknown rotation, per entry and as the default
        const root = try testRoot(a);
        entryOf(root, "model.layers.0.mlp.switch_mlp.gate_proj").getPtr("rotation").?.* = .{ .string = "hadamard64" };
        try testing.expectError(error.Jangtq2Rotation, testSpec(root));
        const root2 = try testRoot(a);
        root2.getPtr("jangtq").?.object.getPtr("rotation").?.* = .{ .string = "random" };
        try testing.expectError(error.Jangtq2Rotation, testSpec(root2));
    }
    { // one bank rotated per projection
        const root = try testRoot(a);
        entryOf(root, "model.layers.1.mlp.switch_mlp.down_proj").getPtr("rotation").?.* = .{ .string = "none" };
        try testing.expectError(error.Jangtq2MixedRotation, testSpec(root));
    }
    { // a dense codebook projection
        const root = try testRoot(a);
        var dense: std.json.ObjectMap = .empty;
        try dense.put(a, "mode", .{ .string = "jangtq2" });
        try dense.put(a, "bits", .{ .integer = 4 });
        try root.getPtr("quantization").?.object.put(a, "model.layers.0.mlp.shared_expert.down_proj", .{ .object = dense });
        try testing.expectError(error.Jangtq2DenseProjection, testSpec(root));
    }
    { // a codebook bank on the MTP head, and one past the trunk
        for ([_][]const u8{ "mtp.layers.0.mlp.switch_mlp.gate_proj", "model.layers.2.mlp.switch_mlp.gate_proj", "model.layers.01.mlp.switch_mlp.gate_proj" }) |name| {
            const root = try testRoot(a);
            try root.getPtr("quantization").?.object.put(a, name, root.get("quantization").?.object.get("model.layers.0.mlp.switch_mlp.gate_proj").?);
            try testing.expectError(error.Jangtq2Stack, testSpec(root));
        }
    }
    { // two spellings of one projection that disagree; agreeing ones are one entry
        const root = try testRoot(a);
        var alias: std.json.ObjectMap = .empty;
        try alias.put(a, "mode", .{ .string = "jangtq2" });
        try alias.put(a, "bits", .{ .integer = 4 });
        try alias.put(a, "rotation", .{ .string = "hadamard32" });
        try root.getPtr("quantization").?.object.put(a, "language_model.layers.0.mlp.switch_mlp.gate_proj", .{ .object = alias });
        try testing.expect(try testSpec(root) != null);
        try alias.put(a, "bits", .{ .integer = 3 });
        try root.getPtr("quantization").?.object.put(a, "language_model.layers.0.mlp.switch_mlp.gate_proj", .{ .object = alias });
        try testing.expectError(error.Jangtq2Alias, testSpec(root));
    }
    { // version 1 banks (`tq_packed`), another packing, and entries with no declaration
        const root = try testRoot(a);
        root.getPtr("jangtq").?.object.getPtr("version").?.* = .{ .integer = 1 };
        try testing.expectError(error.Jangtq2Version, testSpec(root));
        const root2 = try testRoot(a);
        root2.getPtr("jangtq").?.object.getPtr("packing").?.* = .{ .string = "msb-bitstream" };
        try testing.expectError(error.Jangtq2Format, testSpec(root2));
        var root3 = try testRoot(a);
        _ = root3.orderedRemove("jangtq");
        try testing.expectError(error.Jangtq2Undeclared, testSpec(root3));
        try testing.expect(declared(root3));
    }
    { // a trunk entry at a width MLX does not run
        const root = try testRoot(a);
        entryOf(root, "language_model.layers.0.self_attn.q_proj").getPtr("bits").?.* = .{ .integer = 7 };
        try testing.expectError(error.Jangtq2TrunkEntry, testSpec(root));
    }
}

test "jangh config: a JANGH bundle parses as a text-only qwen4_exp under the JANG norm convention" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const c = try model_mod.parseConfigFromJson(a, try testConfigText(a));
    try testing.expect(c.isQwen4() and c.jangtq2 != null);
    try testing.expectEqual(model_mod.Qwen4NormConvention.runtime_plus1_applied, c.qwen4_norm_convention.?);
    try testing.expect(!c.has_vision and !c.qwen_vision);
    const Edit = enum { marker, convention, no_stamp, exl3, family };
    for ([_]Edit{ .marker, .convention, .no_stamp, .exl3, .family }) |edit| {
        var root = try testRoot(a);
        switch (edit) {
            .marker => try root.put(a, "qwen4_norm_convention", .{ .string = "folded" }),
            .convention => root.get("jang_config").?.object.getPtr("norm_convention").?.* = .{ .string = "delta" },
            .no_stamp => _ = root.orderedRemove("jang_config"),
            .exl3 => {
                var eq: std.json.ObjectMap = .empty;
                try eq.put(a, "format", .{ .string = "exl3" });
                try eq.put(a, "k", .{ .float = 2.5 });
                try eq.put(a, "codebook", .{ .string = "mcg" });
                try root.put(a, "expert_quant", .{ .object = eq });
            },
            .family => root.getPtr("model_type").?.* = .{ .string = "llama" },
        }
        const want: anyerror = switch (edit) {
            .marker => error.ConflictingQwen4NormConvention,
            .convention => error.UnsupportedJangNormConvention,
            .no_stamp => error.JangNormConventionMissing,
            .exl3 => error.ConflictingRoutedExpertFormats,
            .family => error.Jangtq2UnsupportedFamily,
        };
        try testing.expectError(want, model_mod.parseConfigFromJson(a, try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = root }, .{})));
    }
}

test "jangh norm convention: only runtime_plus1_applied is a JANG bundle's" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]struct { json: []const u8, err: ?anyerror }{
        .{ .json = "{\"jang_config\":{\"norm_convention\":\"runtime_plus1_applied\"}}", .err = null },
        .{ .json = "{\"jang\":{\"norm_convention\":\" runtime_plus1_applied\\n\"}}", .err = null },
        .{ .json = "{\"jang_config\":{\"norm_convention\":\"delta\"}}", .err = error.UnsupportedJangNormConvention },
        .{ .json = "{\"jang_config\":{}}", .err = error.JangNormConventionMissing },
        .{ .json = "{}", .err = error.JangNormConventionMissing },
    }) |case| {
        const root = (try std.json.parseFromSliceLeaky(std.json.Value, a, case.json, .{})).object;
        if (case.err) |e| try testing.expectError(e, checkNormConvention(root)) else try checkNormConvention(root);
    }
}

test "jangh names: the bundle's roots map onto the layout the qwen4_exp loader reads" {
    var buf: [256]u8 = undefined;
    for ([_][2][]const u8{
        .{ "language_model.layers.3.self_attn.q_proj.weight", "language_model.model.layers.3.self_attn.q_proj.weight" },
        .{ "language_model.embed_tokens.scales", "language_model.model.embed_tokens.scales" },
        .{ "language_model.hyper_connection_mixer.hc_norm.weight", "language_model.model.hyper_connection_mixer.hc_norm.weight" },
        .{ "model.layers.7.mlp.switch_mlp.down_proj.tq2_packed", "language_model.model.layers.7.mlp.switch_mlp.down_proj.tq2_packed" },
        .{ "mtp.layers.0.mlp.experts.gate_up_proj.weight", "language_model.mtp.layers.0.mlp.experts.gate_up_proj.weight" },
        .{ "lm_head.biases", "language_model.lm_head.biases" },
        .{ "language_model.layers.1.ple.conv1d_weight", "language_model.model.layers.1.ple.conv1d.weight" },
        .{ "language_model.layers.1.ple.key_proj.weight", "language_model.model.layers.1.ple.key_proj.weight" },
        .{ "language_model.model.layers.0.mlp.gate.weight", "language_model.model.layers.0.mlp.gate.weight" },
    }) |case| try testing.expectEqualStrings(case[1], (try runtimeName(&buf, case[0])).?);
    for ([_][]const u8{
        "language_model.layers.1.ple.ngram_embedding.shards.7.scales",
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight",
        "language_model.layers.1.ple.layer_multipliers",
        "language_model.layers.1.ple.ngram_heads_vocab_sizes",
        "language_model.layers.1.ple.ngram_heads_offsets",
        "visual.blocks.0.attn.qkv.weight",
    }) |dropped| try testing.expect(try runtimeName(&buf, dropped) == null);
    for ([_][]const u8{ "model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight", "vision_tower.x" }) |unknown| {
        try testing.expectError(error.JanghTensorName, runtimeName(&buf, unknown));
    }
    var tiny: [8]u8 = undefined;
    try testing.expectError(error.NameTooLong, runtimeName(&tiny, "lm_head.weight"));
}

/// One tensor as a shard header describes it.
const Info = struct { dtype: []const u8, shape: []const i64 };

/// Every tensor's dtype and shape from an indexed checkpoint's shard headers; no tensor data is read.
fn readHeaders(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.StringHashMapUnmanaged(Info) {
    const index = try dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 << 20));
    const wm = (try std.json.parseFromSliceLeaky(std.json.Value, a, index, .{})).object.get("weight_map").?.object;
    var shards: std.StringHashMapUnmanaged(void) = .empty;
    var it = wm.iterator();
    while (it.next()) |e| try shards.put(a, e.value_ptr.string, {});
    var out: std.StringHashMapUnmanaged(Info) = .empty;
    var sit = shards.keyIterator();
    while (sit.next()) |shard| {
        var file = try dir.openFile(io, shard.*, .{});
        defer file.close(io);
        var rbuf: [8192]u8 = undefined;
        var rs = file.reader(io, &rbuf);
        const len = try rs.interface.takeInt(u64, .little);
        const raw = try rs.interface.readAlloc(a, @intCast(len));
        var hit = (try std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{})).object.iterator();
        while (hit.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const dims = e.value_ptr.object.get("shape").?.array.items;
            const shape = try a.alloc(i64, dims.len);
            for (dims, shape) |d, *o| o.* = d.integer;
            try out.put(a, e.key_ptr.*, .{ .dtype = e.value_ptr.object.get("dtype").?.string, .shape = shape });
        }
    }
    try testing.expectEqual(wm.count(), out.count());
    return out;
}

/// A checkpoint's shard headers through the loader's own naming (`model.loadedName` under `loadOptsFor`).
/// A JANGH bundle: every kept tensor renamed into our pack's roots, only the n-gram table with its hash
/// constants and the vision tower skipped, each layer's banks at its declared widths. Any other pack: unchanged.
fn checkHeaders(dir_path: []const u8, expect_jangh: bool) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{});
    defer dir.close(io);
    const config = try model_mod.parseConfigFromJson(a, try dir.readFileAlloc(io, "config.json", a, .limited(16 << 20)));
    try testing.expectEqual(expect_jangh, config.jangtq2 != null);
    const headers = try readHeaders(a, io, dir);
    const opts = model_mod.loadOptsFor(&config, config.has_vision);
    var loaded: std.StringHashMapUnmanaged(Info) = .empty;
    var renamed: usize = 0;
    var hit = headers.iterator();
    while (hit.next()) |e| {
        var buf: [256]u8 = undefined;
        const key = e.key_ptr.*;
        const name = (try model_mod.loadedName(opts, key, &buf)) orelse {
            try testing.expect(expect_jangh and (isNgramTable(key) or std.mem.startsWith(u8, key, "visual.")));
            continue;
        };
        renamed += @intFromBool(!std.mem.eql(u8, name, key));
        const gop = try loaded.getOrPut(a, try a.dupe(u8, name));
        try testing.expect(!gop.found_existing);
        gop.value_ptr.* = e.value_ptr.*;
    }
    if (!expect_jangh) return testing.expectEqual(@as(usize, 0), renamed);
    var lit = loaded.keyIterator();
    while (lit.next()) |k| {
        const rooted = for ([_][]const u8{ "language_model.model.", "language_model.mtp.", "language_model.lm_head." }) |root| {
            if (std.mem.startsWith(u8, k.*, root)) break true;
        } else false;
        try testing.expect(rooted);
    }
    const spec = config.jangtq2.?;
    const e: i64 = config.num_experts;
    const h: i64 = config.hidden_size;
    const inter: i64 = config.moe_intermediate_size;
    for (spec.layers[0..config.num_hidden_layers], 0..) |layer, li| {
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
            const down = std.mem.eql(u8, proj, "down_proj");
            const bits: i64 = if (down) layer.down_bits else layer.gate_up_bits;
            const out: i64 = if (down) h else inter;
            const in: i64 = if (down) inter else h;
            const base = try std.fmt.allocPrint(a, "language_model.model.layers.{d}.mlp.switch_mlp.{s}", .{ li, proj });
            const pk = loaded.get(try std.fmt.allocPrint(a, "{s}.tq2_packed", .{base})).?;
            const sc = loaded.get(try std.fmt.allocPrint(a, "{s}.tq2_scales", .{base})).?;
            try testing.expectEqualStrings("U32", pk.dtype);
            try testing.expectEqualSlices(i64, &.{ e, out, @divExact(in * bits, 32) }, pk.shape);
            try testing.expectEqualStrings("F16", sc.dtype);
            try testing.expectEqualSlices(i64, &.{ e, out }, sc.shape);
            try testing.expect(loaded.get(try std.fmt.allocPrint(a, "{s}.weight", .{base})) == null);
        }
    }
    try testing.expectEqual(loaded.count(), renamed);
}

test "jangh headers: a JANGH bundle's tensors load under our pack's names, banks at their declared widths (QWEN4_JANGH_TEST_MODEL)" {
    const path = std.c.getenv("QWEN4_JANGH_TEST_MODEL") orelse return error.SkipZigTest;
    try checkHeaders(std.mem.span(path), true);
}

test "jangh headers: an existing qwen4_exp pack loads under its stored names (QWEN4_TEST_MODEL)" {
    const path = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try checkHeaders(std.mem.span(path), false);
}
