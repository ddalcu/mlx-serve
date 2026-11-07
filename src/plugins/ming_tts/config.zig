//! Ming-3.0-tts-0.5b's claims, caps and bills, and its geometry from `config.json`, MLX-free. The pack's `model_type` is
//! `dense`, a generic word, so the claim also asks for the flow head and aggregator blocks only this model has.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const bill = @import("../../sdk/media/bill.zig");
const pack_claims = @import("../../sdk/media/pack_claims.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "ming_tts",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/Ming-3.0-tts-0.5B-8bit"},
};

pub const model_types = [_][]const u8{"dense"};

pub const required_file = "model.safetensors";

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    if (!std.mem.eql(u8, model_type.slice(), model_types[0])) return .no;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = pack_claims.readJson(arena.allocator(), dir, "config.json") orelse return .no;
    if (pack_claims.path(v, &.{"ditar_config"}) == null or pack_claims.path(v, &.{"aggregator_config"}) == null) return .no;
    return .by_marker;
}

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{ .routes = .{ .speech = true } };
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    return bill.safetensorsIn(dir, "") + bill.files(dir, &.{"campplus.onnx"});
}

/// MLX's working memory for one fragment (KV caches, VAE decode), above the resident weights.
const fragment_peak: u64 = 3 << 30;
/// The output buffer's bound: f32 at 44.1 kHz, one second per input byte (a CJK character is 3 bytes, about 0.3 s).
const output_per_input_byte: u64 = 4 * 44100;

/// The bytes of `input` (or `text`). `ref_audio` is a clip of at most 20 s and does not grow the output.
fn inputBytes(body: []const u8) u64 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), body, .{}) catch return 0;
    const t = pack_claims.path(v, &.{"input"}) orelse pack_claims.path(v, &.{"text"}) orelse return 0;
    return if (t == .string) t.string.len else 0;
}

pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = m;
    return billLoad(dir, &.{}) + fragment_peak + inputBytes(body.slice()) * output_per_input_byte;
}

/// One Qwen2 stack (the LM, the VAE encoder, the VAE decoder).
pub const Qwen = struct {
    layers: usize,
    hidden: c_int,
    heads: c_int,
    kv_heads: c_int,
    theta: f32,
    window: ?c_int,

    pub fn headDim(q: Qwen) c_int {
        return @divExact(q.hidden, q.heads);
    }
};

/// One DiT stack (the flow head, the aggregator).
pub const Dit = struct {
    layers: c_int,
    hidden: c_int,
    heads: c_int,
};

/// The pack-wide affine scheme from `config.json`.
pub const Quant = struct {
    bits: u32,
    group_size: u32,
};

pub const Dims = struct {
    lm: Qwen,
    enc: Qwen,
    dec: Qwen,
    dit: Dit,
    agg: Dit,
    /// Null on a bf16 pack.
    quant: ?Quant = null,
};

pub const ParseError = error{ InvalidConfig, UnservedQuantMode };

/// A positive integer at `keys`, at most `max`.
fn intAt(v: std.json.Value, keys: []const []const u8, max: i64) ParseError!c_int {
    const x = pack_claims.path(v, keys) orelse return error.InvalidConfig;
    if (x != .integer or x.integer < 1 or x.integer > max) return error.InvalidConfig;
    return @intCast(x.integer);
}

fn qwenAt(v: std.json.Value, comptime keys: []const []const u8) ParseError!Qwen {
    const q: Qwen = .{
        .layers = @intCast(try intAt(v, keys ++ .{"num_hidden_layers"}, 64)),
        .hidden = try intAt(v, keys ++ .{"hidden_size"}, 8192),
        .heads = try intAt(v, keys ++ .{"num_attention_heads"}, 64),
        .kv_heads = try intAt(v, keys ++ .{"num_key_value_heads"}, 64),
        .theta = blk: {
            const t = pack_claims.path(v, keys ++ .{"rope_theta"}) orelse return error.InvalidConfig;
            break :blk switch (t) {
                .integer => |i| @floatFromInt(i),
                .float => |f| @floatCast(f),
                else => return error.InvalidConfig,
            };
        },
        .window = blk: {
            const on = pack_claims.path(v, keys ++ .{"use_sliding_window"}) orelse break :blk null;
            if (on != .bool or !on.bool) break :blk null;
            break :blk try intAt(v, keys ++ .{"sliding_window"}, 1 << 20);
        },
    };
    if (@mod(q.hidden, q.heads) != 0 or @mod(q.heads, q.kv_heads) != 0 or @mod(q.headDim(), 2) != 0 or !(q.theta > 0)) return error.InvalidConfig;
    return q;
}

fn ditAt(v: std.json.Value, comptime keys: []const []const u8) ParseError!Dit {
    const d: Dit = .{
        .layers = try intAt(v, keys ++ .{"depth"}, 64),
        .hidden = try intAt(v, keys ++ .{"hidden_size"}, 8192),
        .heads = try intAt(v, keys ++ .{"num_heads"}, 64),
    };
    if (@mod(d.hidden, d.heads) != 0) return error.InvalidConfig;
    return d;
}

fn quantAt(v: std.json.Value) ParseError!?Quant {
    const q = pack_claims.path(v, &.{"quantization"}) orelse pack_claims.path(v, &.{"quantization_config"}) orelse return null;
    if (pack_claims.path(q, &.{"mode"})) |m| {
        if (m != .string or !std.mem.eql(u8, m.string, "affine")) return error.UnservedQuantMode;
    }
    const bits = try intAt(q, &.{"bits"}, 8);
    const group_size = try intAt(q, &.{"group_size"}, 128);
    switch (bits) {
        2, 3, 4, 5, 6, 8 => {},
        else => return error.InvalidConfig,
    }
    switch (group_size) {
        32, 64, 128 => {},
        else => return error.InvalidConfig,
    }
    return .{ .bits = @intCast(bits), .group_size = @intCast(group_size) };
}

/// The geometry a `config.json` declares. Config JSON is untrusted: every field is bounds-checked.
pub fn parseDims(a: std.mem.Allocator, config_json: []const u8) ParseError!Dims {
    const v = std.json.parseFromSliceLeaky(std.json.Value, a, config_json, .{}) catch return error.InvalidConfig;
    return .{
        .lm = try qwenAt(v, &.{"llm_config"}),
        .enc = try qwenAt(v, &.{ "audio_tokenizer_config", "enc_kwargs", "backbone" }),
        .dec = try qwenAt(v, &.{ "audio_tokenizer_config", "dec_kwargs", "backbone" }),
        .dit = try ditAt(v, &.{"ditar_config"}),
        .agg = try ditAt(v, &.{"aggregator_config"}),
        .quant = try quantAt(v),
    };
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

const released = @embedFile("../../fixtures/model-configs/ming_tts-0.5b.json");

test "ming_tts config: the release is claimed by marker; another `dense` or a stranger is not" {
    const rel: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = released }} };
    try testing.expectEqual(abi.Claim.by_marker, claims(.of("dense"), &rel.packDir()));
    try testing.expectEqual(abi.Claim.no, claims(.of("qwen3"), &rel.packDir()));
    const other: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = "{\"model_type\":\"dense\"}" }} };
    try testing.expectEqual(abi.Claim.no, claims(.of("dense"), &other.packDir()));
    const none: fixture.MemDir = .{ .files = &.{} };
    try testing.expectEqual(abi.Claim.no, claims(.of("dense"), &none.packDir()));
}

test "ming_tts config: a request's bill grows with its text, not with its reference clip" {
    const none: fixture.MemDir = .{ .files = &.{} };
    const dir = none.packDir();
    const m: abi.Machine = .{};
    const plain = billRequest(&dir, &m, .of("{\"input\":\"hi\"}"));
    try testing.expectEqual(plain, billRequest(&dir, &m, .of("{\"input\":\"hi\",\"ref_audio\":\"UklGUklGUklGUklGUklG\"}")));
    try testing.expect(billRequest(&dir, &m, .of("{\"input\":\"hello there\"}")) > plain);
    try testing.expectEqual(fragment_peak, billRequest(&dir, &m, .of("not json")));
}

test "ming_tts config: the release's geometry" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const d = try parseDims(arena.allocator(), released);
    try testing.expectEqual(@as(usize, 24), d.lm.layers);
    try testing.expectEqual(@as(c_int, 64), d.lm.headDim());
    try testing.expectEqual(@as(c_int, 2), d.lm.kv_heads);
    try testing.expectEqual(@as(?c_int, null), d.lm.window);
    try testing.expectEqual(@as(?c_int, 64), d.dec.window);
    try testing.expectEqual(@as(?c_int, 64), d.enc.window);
    try testing.expectEqual(@as(c_int, 1024), d.dit.hidden);
    try testing.expectEqual(@as(c_int, 8), d.agg.layers);
}

test "ming_tts config: a quantized pack declares its scheme; another mode and bad widths are refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqual(@as(?Quant, null), (try parseDims(a, released)).quant);
    const head = "{\"quantization\":";
    const q4 = try parseDims(a, head ++ "{\"bits\":4,\"group_size\":64,\"mode\":\"affine\"}," ++ released[1..]);
    try testing.expectEqual(Quant{ .bits = 4, .group_size = 64 }, q4.quant.?);
    try testing.expectError(error.UnservedQuantMode, parseDims(a, head ++ "{\"bits\":4,\"group_size\":32,\"mode\":\"mxfp4\"}," ++ released[1..]));
    try testing.expectError(error.InvalidConfig, parseDims(a, head ++ "{\"bits\":7,\"group_size\":64}," ++ released[1..]));
    try testing.expectError(error.InvalidConfig, parseDims(a, head ++ "{\"bits\":8,\"group_size\":48}," ++ released[1..]));
}

test "ming_tts config: malformed geometry is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectError(error.InvalidConfig, parseDims(a, "not json"));
    try testing.expectError(error.InvalidConfig, parseDims(a, "{\"llm_config\":{\"num_hidden_layers\":-1}}"));
    try testing.expectError(error.InvalidConfig, parseDims(a, "{\"llm_config\":{\"num_hidden_layers\":1,\"hidden_size\":7,\"num_attention_heads\":2,\"num_key_value_heads\":1,\"rope_theta\":1e6}}"));
}
