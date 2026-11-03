//! A config.json read the way the engine reads it (JSON null is absent, numbers checked and narrowed), for the
//! plugins' MLX-free parses: fields, the `quantization` block and the eos ids. Shared by lfm2, nemotron_h, hy_v3,
//! mimo_v2 and bert.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const generation = @import("generation.zig");
const fp8_grid = @import("../quant/fp8_block128_gemv.zig");

pub const Object = std.json.ObjectMap;
pub const Value = std.json.Value;

pub fn field(o: Object, key: []const u8) ?Value {
    const v = o.get(key) orelse return null;
    return if (v == .null) null else v;
}

/// `layer_types` naming only full attention (transformers v5 writes it for every model) is the every-layer full
/// attention a dense decoder serves; absent too.
pub fn allFullAttention(cfg: Object) bool {
    const lt = field(cfg, "layer_types") orelse return true;
    if (lt != .array) return false;
    for (lt.array.items) |item| if (item != .string or !std.mem.eql(u8, item.string, "full_attention")) return false;
    return true;
}

pub fn int(v: ?Value) ?u32 {
    const x = v orelse return null;
    if (x != .integer) return null;
    return std.math.cast(u32, x.integer);
}

pub fn float(v: ?Value) ?f64 {
    const x = v orelse return null;
    const f: f64 = switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return null,
    };
    return if (std.math.isFinite(f)) f else null;
}

pub fn boolean(v: ?Value) ?bool {
    const x = v orelse return null;
    return if (x == .bool) x.bool else null;
}

pub fn string(v: ?Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

pub fn object(v: ?Value) ?Object {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

/// `key` as an integer the engine accepts; `error.Invalid` for a present value it would refuse.
pub fn intField(o: Object, key: []const u8) error{Invalid}!?u32 {
    const v = field(o, key) orelse return null;
    return int(v) orelse error.Invalid;
}

pub fn floatField(o: Object, key: []const u8) error{Invalid}!?f64 {
    const v = field(o, key) orelse return null;
    return float(v) orelse error.Invalid;
}

/// An integer-valued key read only when it holds an integer (the engine skips other values).
pub fn intIfInt(o: Object, key: []const u8) error{Invalid}!?u32 {
    const v = field(o, key) orelse return null;
    if (v != .integer) return null;
    return int(v) orelse error.Invalid;
}

fn quantMode(name: []const u8) ?abi.QuantMode {
    inline for (.{ .affine, .nvfp4, .mxfp4, .mxfp8 }) |m| if (std.mem.eql(u8, name, @tagName(m))) return m;
    return null;
}

/// The root's `quantization` block as the engine reads it (`group_size` 64 when absent); `error.Invalid` for a mode
/// or width the engine refuses, `error.Unsupported` for one these plugins do not serve.
pub fn quant(root: Object) error{ Invalid, Unsupported }!abi.Quant {
    return quantOf(root, "quantization", false);
}

/// `quant` over the block under `key`; `ternary_ok` = the plugin runs a ternary 2-bit group-128 pack as the engine
/// does (`Linear.ternary`).
pub fn quantOf(root: Object, key: []const u8, ternary_ok: bool) error{ Invalid, Unsupported }!abi.Quant {
    const q = object(field(root, key)) orelse {
        if (field(root, key) != null) return error.Invalid;
        return .{};
    };
    var out: abi.Quant = .{ .bits = std.math.cast(u8, (try intField(q, "bits")) orelse 0) orelse return error.Invalid, .group_size = (try intField(q, "group_size")) orelse 64, .mode = .affine };
    // A mode MLX has no kernels for (a GGUF-style `iq2_xxs`) is a pack this build does not serve, not a broken config.
    if (string(field(q, "mode"))) |m| out.mode = quantMode(m) orelse return error.Unsupported;
    var it = q.iterator();
    while (it.next()) |kv| if (kv.value_ptr.* == .object) if (kv.value_ptr.object.get("mode")) |m| {
        if (m == .string and quantMode(m.string) == null and !fp8_grid.isEntry(kv.value_ptr.object)) return error.Unsupported;
    };
    if (out.mode == .affine and out.bits != 0) switch (out.bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return error.Invalid,
    };
    // 2-bit at group 128 may be a ternary pack the engine's own kernels serve.
    if (!ternary_ok and out.mode == .affine and out.bits == 2 and out.group_size == 128) return error.Unsupported;
    if (out.bits == 0) return .{};
    return out;
}

/// The block of a `--mlx-gguf` file's rebuilt config (`quantization.mode: gguf`; each tensor's block type rides its
/// sentinel, `sdk/quant/gguf`), as the engine reads its bits and group; null for any other block.
pub fn ggufQuant(root: Object) error{Invalid}!?abi.Quant {
    const q = object(field(root, "quantization")) orelse return null;
    const m = string(field(q, "mode")) orelse return null;
    if (!std.mem.eql(u8, m, "gguf")) return null;
    return .{ .bits = std.math.cast(u8, (try intField(q, "bits")) orelse 0) orelse return error.Invalid, .group_size = (try intField(q, "group_size")) orelse 64, .mode = .gguf };
}

/// The root's `eos_token_id` (one id or a list) into the shell.
pub fn eos(root: Object, s: *abi.Shell) error{Invalid}!void {
    const v = field(root, "eos_token_id") orelse return;
    switch (v) {
        .integer => generation.addEos(s, int(v) orelse return error.Invalid),
        .array => |arr| for (arr.items) |item| if (item == .integer) generation.addEos(s, int(item) orelse return error.Invalid),
        else => {},
    }
}

/// The config's `rope_theta` as the engine resolves it (`rope_theta`, then `rope_parameters`' own and its
/// `full_attention` entry, later wins), `fallback` when none.
pub fn ropeTheta(cfg: Object, fallback: f64) error{Invalid}!f64 {
    var theta = (try floatField(cfg, "rope_theta")) orelse fallback;
    if (object(field(cfg, "rope_parameters"))) |rp| {
        if (object(field(rp, "full_attention"))) |fa| if (try floatField(fa, "rope_theta")) |t| {
            theta = t;
        };
        if (try floatField(rp, "rope_theta")) |t| theta = t;
    }
    return theta;
}

/// `max_position_embeddings` from the text config, else the root.
pub fn maxPositions(cfg: Object, root: Object) error{Invalid}!u32 {
    const m = (try intField(cfg, "max_position_embeddings")) orelse 0;
    if (m != 0) return m;
    return (try intIfInt(root, "max_position_embeddings")) orelse 0;
}

/// The `{"bits":..,"group_size":..,"mode":".."}` block a quant format opens (`{}` = dense), allocated.
pub fn blockJson(q: abi.Quant) ![]u8 {
    const gpa = std.heap.c_allocator;
    if (q.bits == 0) return gpa.dupe(u8, "{}");
    return std.fmt.allocPrint(gpa, "{{\"bits\":{d},\"group_size\":{d},\"mode\":\"{t}\"}}", .{ q.bits, q.group_size, q.mode });
}

/// Parses `config_json` into `arena`; the root object.
pub fn parseRoot(arena: std.mem.Allocator, config_json: []const u8) error{ Malformed, NotObject }!Object {
    const doc = std.json.parseFromSliceLeaky(Value, arena, config_json, .{}) catch return error.Malformed;
    if (doc != .object) return error.NotObject;
    return doc.object;
}

// ── tests ──

const testing = std.testing;

test "plugin json: the quantization block reads as the engine reads it" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const q = try quant(try parseRoot(a, "{\"quantization\":{\"bits\":4,\"group_size\":32}}"));
    try testing.expect(q.bits == 4 and q.group_size == 32 and q.mode == .affine);
    try testing.expectEqual(@as(u8, 0), (try quant(try parseRoot(a, "{}"))).bits);
    try testing.expectError(error.Invalid, quant(try parseRoot(a, "{\"quantization\":{\"bits\":7}}")));
    try testing.expectError(error.Unsupported, quant(try parseRoot(a, "{\"quantization\":{\"bits\":4,\"mode\":\"iq2\"}}")));
    const fp8 = try quant(try parseRoot(a, "{\"quantization\":{\"bits\":4,\"group_size\":32,\"mode\":\"mxfp4\",\"l.q\":{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]}}}"));
    try testing.expect(fp8.bits == 4 and fp8.mode == .mxfp4);
    try testing.expectError(error.Unsupported, quant(try parseRoot(a, "{\"quantization\":{\"bits\":4,\"l.q\":{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[128,128]}}}")));
    try testing.expectError(error.Unsupported, quant(try parseRoot(a, "{\"quantization\":{\"bits\":2,\"group_size\":128}}")));
}

test "plugin json: rope_theta resolves through rope_parameters, later wins" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqual(@as(f64, 5), try ropeTheta(try parseRoot(a, "{\"rope_theta\":1,\"rope_parameters\":{\"rope_theta\":5}}"), 9));
    try testing.expectEqual(@as(f64, 9), try ropeTheta(try parseRoot(a, "{}"), 9));
}
