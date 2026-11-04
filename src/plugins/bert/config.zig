//! BERT packs (`bert`, the mlx-community embedding conversions): the claim and the parse into `Parsed` plus this
//! plugin's own fields. MLX-free. An encoder: no chat, no KV cache, prefill only (`caps.embed`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const json = @import("../../sdk/config/json.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "bert",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/all-MiniLM-L6-v2-4bit"},
};

pub const model_types = [_][]const u8{"bert"};

pub const claims = sdk.claimsByType(&model_types);

/// The fields the forward and the load read.
pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    head_dim: u32,
    layer_norm_eps: f32 = 1e-12,
    max_position_embeddings: u32,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "bert: " ++ why);
}

fn invalid(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.failed, "bert: " ++ why);
}

const foreign_keys = [_][]const u8{ "expert_quant", "modules", "text_config" };

fn pooling(name: []const u8) ?abi.Pooling {
    inline for (.{ .mean, .cls, .last_token }) |p| if (std.mem.eql(u8, name, @tagName(p))) return p;
    return null;
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = json.parseRoot(arena.allocator(), config_json.slice()) catch return invalid("config.json is not a JSON object");
    const type_name = json.string(json.field(root, "model_type")) orelse return unsupported("no model_type");
    if (!std.mem.eql(u8, type_name, "bert")) return unsupported("another model_type");
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
    for (foreign_keys) |k| if (json.field(root, k) != null) return unsupported("a key this plugin does not model");
    var c = readConfig(root) catch |e| return switch (e) {
        error.Unsupported => unsupported("a shape this plugin does not serve"),
        error.Invalid => invalid("a config value out of range"),
    };

    var p: abi.Parsed = .{ .caps = .{ .embed = true, .prefix_cache = false } };
    // The pooling contract, explicit first, then the sentence-transformers sidecar; one we do not implement is
    // refused, never a silent mean-pool. Unset = the host's name fallback, else mean.
    if (json.string(json.field(root, "pooling_mode"))) |name| p.embed.pooling = pooling(name) orelse return invalid("an unknown pooling_mode");
    if (p.embed.pooling == .unset and !stub and dir.has(sidecar)) {
        var buf: [16 * 1024]u8 = undefined;
        var why: abi.Verdict = .{};
        if (dir.readFile(sidecar, &buf, &why)) |bytes| {
            p.embed.pooling = sidecarPooling(arena.allocator(), bytes) catch return invalid("1_Pooling declares only pooling modes this plugin does not implement");
        }
    }
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_attention_heads,
        .head_dim = c.head_dim,
        .max_position_embeddings = c.max_position_embeddings,
        .context_cap = c.max_position_embeddings,
        .quant = c.quant,
        .act_dtype = .bf16,
        .model_type = abi.Str.of("bert"),
    };
    if (json.field(root, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        s.bos = json.int(v) orelse return invalid("bos_token_id");
    };
    json.eos(root, s) catch return invalid("eos_token_id");

    const cfg_out = gpa.create(Config) catch return invalid("out of memory");
    c.block_json = json.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return invalid("out of memory");
    };
    cfg_out.* = c;
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

const sidecar = "1_Pooling/config.json";

/// The mode a sentence-transformers `1_Pooling/config.json` declares (`.unset` when it is no pooling config);
/// `error.Unsupported` when it declares only modes we do not implement.
fn sidecarPooling(a: std.mem.Allocator, bytes: []const u8) error{Unsupported}!abi.Pooling {
    const obj = json.parseRoot(a, bytes) catch return .unset;
    for ([_]struct { []const u8, abi.Pooling }{ .{ "pooling_mode_lasttoken", .last_token }, .{ "pooling_mode_cls_token", .cls }, .{ "pooling_mode_mean_tokens", .mean } }) |e| {
        if (json.boolean(json.field(obj, e[0])) orelse false) return e[1];
    }
    // ST 6 names the mode in one string.
    if (json.string(json.field(obj, "pooling_mode"))) |s| {
        if (std.mem.eql(u8, s, "lasttoken")) return .last_token;
        return pooling(s) orelse error.Unsupported;
    }
    var it = obj.iterator();
    while (it.next()) |e| if (std.mem.startsWith(u8, e.key_ptr.*, "pooling_mode_")) return error.Unsupported;
    return .unset;
}

/// The engine's reading of a BERT config (its common block, then the `bert` arm).
fn readConfig(root: json.Object) error{ Unsupported, Invalid }!Config {
    var c: Config = .{
        .vocab_size = (try json.intField(root, "vocab_size")) orelse 262208,
        .hidden_size = (try json.intField(root, "hidden_size")) orelse 3840,
        .intermediate_size = (try json.intField(root, "intermediate_size")) orelse 15360,
        .num_hidden_layers = (try json.intField(root, "num_hidden_layers")) orelse 48,
        .num_attention_heads = (try json.intField(root, "num_attention_heads")) orelse 16,
        .head_dim = 0,
        .max_position_embeddings = (try json.intField(root, "max_position_embeddings")) orelse 0,
    };
    if (c.num_attention_heads == 0) return error.Invalid;
    c.head_dim = c.hidden_size / c.num_attention_heads;
    if (try json.floatField(root, "layer_norm_eps")) |e| c.layer_norm_eps = @floatCast(e);
    // A 2-bit group-128 pack runs stock kernels: the engine's BERT load never arms the ternary ones.
    c.quant = json.quantOf(root, "quantization", true) catch |e| return if (e == error.Unsupported) error.Unsupported else error.Invalid;
    // The engine's BERT load reads every table and projection quantized: a dense pack has nothing to match.
    if (c.quant.bits == 0) return error.Unsupported;
    if (c.vocab_size == 0 or c.hidden_size == 0 or c.num_hidden_layers == 0 or c.head_dim == 0 or
        c.hidden_size % c.num_attention_heads != 0 or c.intermediate_size == 0)
        return error.Unsupported;
    return c;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| {
        const c: *Config = @ptrCast(@alignCast(ptr));
        gpa.free(c.block_json);
        gpa.destroy(c);
    }
    p.private = null;
}

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

/// No layer keeps a cache: an encoder reads its whole sequence each forward.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const n = @min(of(p).num_hidden_layers, cap);
    for (out[0..n]) |*o| o.* = .{ .kind = .none };
    return n;
}

// ── tests ──

const testing = std.testing;
const fixture = sdk.fixture;

pub fn parseJson(js: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = js }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(js), .{}, null, true, out);
}

test "bert config: a MiniLM pack parses into an encoder shell with its pooling contract" {
    const js =
        \\{"model_type":"bert","vocab_size":30522,"hidden_size":384,"intermediate_size":1536,"num_hidden_layers":6,
        \\ "num_attention_heads":12,"max_position_embeddings":512,"type_vocab_size":2,"layer_norm_eps":1e-12,
        \\ "pooling_mode":"cls","quantization":{"group_size":64,"bits":4}}
    ;
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(js, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 32), p.shell.head_dim);
    try testing.expectEqual(@as(u32, 12), p.shell.num_key_value_heads);
    try testing.expectEqual(abi.Pooling.cls, p.embed.pooling);
    try testing.expect(p.caps.embed and !p.caps.prefix_cache);
}

test "bert config: an unknown pooling mode fails by name; a dense pack and another type are unsupported" {
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("{\"model_type\":\"bert\",\"num_attention_heads\":2,\"pooling_mode\":\"max\",\"quantization\":{\"bits\":4}}", &p).code);
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson("{\"model_type\":\"bert\"}", &p).code);
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson("{\"model_type\":\"roberta\"}", &p).code);
    try testing.expect(p.private == null);
}

test "bert config: the sentence-transformers sidecar names the pooling (booleans, ST 6's string, unsupported)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqual(abi.Pooling.cls, try sidecarPooling(a, "{\"pooling_mode_cls_token\":true,\"pooling_mode_mean_tokens\":false}"));
    try testing.expectEqual(abi.Pooling.last_token, try sidecarPooling(a, "{\"pooling_mode\":\"lasttoken\"}"));
    try testing.expectEqual(abi.Pooling.unset, try sidecarPooling(a, "{\"word_embedding_dimension\":384}"));
    try testing.expectError(error.Unsupported, sidecarPooling(a, "{\"pooling_mode_weightedmean_tokens\":true}"));
}
