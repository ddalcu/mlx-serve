//! EmbeddingGemma 2 (`embedding_gemma2`, the release's layout): the claim and the parse. The Gemma 4 text trunk as a
//! bidirectional encoder under `language_model`: plain norms, q/k/v norms, unit attention scale, wider full layers
//! (`per_layer_config`), per-layer inputs from the embeddings alone. Image and video rows splice at their placeholder
//! ids; the audio tower is not served. MLX-free.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const gemma = @import("../gemma3/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "embedding_gemma2",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"google/embeddinggemma-2"},
};

pub const model_types = [_][]const u8{"embedding_gemma2"};

pub const claims = sdk.claimsByType(&model_types);

pub const Config = gemma.Config;
pub const freeParsed = gemma.freeParsed;
pub const of = gemma.of;
pub const layout = gemma.encoderLayout;
const field = gemma.field;
const unsupported = gemma.unsupported;

/// The model card's window, not the trunk's 262144-position rope table.
const window = 8192;

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    var d: gemma.Doc = undefined;
    const opened = gemma.Doc.open(config_json, &d);
    if (!opened.isOk()) return opened;
    defer d.close();
    if (span != null) return unsupported("a layer span (the cluster split)");
    if (!std.mem.eql(u8, d.model_type, "embedding_gemma2")) return unsupported("another model_type");
    if (gemma.foreignQuant(d.root)) return unsupported("a quantization_config naming a quant_method");

    var c: Config = .{};
    gemma.readCommon(d.root, d.cfg, &c) catch return unsupported("a field out of range");
    gemma.readQuant(d.root, &c) catch return unsupported("quantization");
    c.model_type = "embedding_gemma2";
    c.weight_prefix = "language_model";
    c.norm_has_offset = false;
    c.has_v_norm = true;
    c.tie_word_embeddings = true;
    c.max_position_embeddings = if (c.max_position_embeddings == 0) window else @min(c.max_position_embeddings, window);
    readGeometry(d.cfg, &c) catch return unsupported("a per-layer geometry this plugin does not model");
    if (c.rope_proportional or c.partial_rotary_factor_global != 1.0) return unsupported("a partial or proportional rope");
    if (c.num_experts > 0 or c.num_kv_shared_layers > 0) return unsupported("experts or KV sharing");
    if (c.vocab_size == 0 or c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
        c.num_hidden_layers == 0 or c.num_attention_heads == 0 or c.num_key_value_heads == 0 or
        c.intermediate_size == 0 or c.head_dim == 0 or c.head_dim % 2 != 0 or c.head_dim > 512 or
        c.global_head_dim % 2 != 0 or c.global_head_dim > 512 or c.hidden_size_per_layer_input == 0)
        return unsupported("dimensions out of range");

    const pooling = gemma.readPooling(dir, d.root, stub) catch return abi.Verdict.refuse(.failed, "gemma: the pooling contract names a mode this plugin does not implement");
    const vision = field(d.root, "vision_config") != null;
    const fin = gemma.finish(dir, d.root, c, stub, .{ .embed = true, .prefix_cache = false, .images = vision, .video = vision }, out);
    if (!fin.isOk()) return fin;
    out.embed = .{ .pooling = pooling, .has_projection = true };
    // The stop set is the text config's own, without the chat terminators: an encoder ends no turn.
    const s = &out.shell;
    s.eos_count = 0;
    out.tokenizer.extra_eos_count = 0;
    gemma.readEos(d.root, s) catch return unsupported("eos_token_id");
    if (field(d.cfg, "eos_token_id")) |e| switch (e) {
        .integer => generation.mergeEos(s, gemma.int(e) catch return unsupported("eos_token_id")),
        .array => |arr| for (arr.items) |item| if (item == .integer) generation.mergeEos(s, gemma.int(item) catch return unsupported("eos_token_id")),
        else => {},
    };
    if (!stub) generation.apply(dir, s);
    if (vision) {
        inline for (.{ .{ "image_token_id", "image" }, .{ "video_token_id", "video" } }) |k| if (field(d.root, k[0])) |v| if (v == .integer) {
            @field(out.placeholder_ids, k[1]) = gemma.int(v) catch return unsupported("a placeholder id");
        };
    }
    return .ok;
}

/// The full layers' head width and KV heads (512 and 1 unless `per_layer_config` names them): one geometry for every
/// full layer, the base one for the rest.
fn readGeometry(cfg: std.json.ObjectMap, c: *Config) error{Unsupported}!void {
    if (c.num_hidden_layers > gemma.max_layer_types) return error.Unsupported;
    c.global_head_dim = 512;
    c.num_global_key_value_heads = 1;
    const block = field(cfg, "per_layer_config") orelse return;
    if (block != .object) return error.Unsupported;
    var listed: [gemma.max_layer_types]bool = @splat(false);
    var first = true;
    var it = block.object.iterator();
    while (it.next()) |e| {
        const layer = std.fmt.parseInt(u32, e.key_ptr.*, 10) catch return error.Unsupported;
        if (layer >= c.num_hidden_layers or !c.isGlobalLayer(layer)) return error.Unsupported;
        if (e.value_ptr.* != .object) return error.Unsupported;
        const entry = e.value_ptr.object;
        const hd = try gemma.int(field(entry, "head_dim") orelse return error.Unsupported);
        const kv = try gemma.int(field(entry, "num_key_value_heads") orelse return error.Unsupported);
        if (!first and (hd != c.global_head_dim or kv != c.num_global_key_value_heads)) return error.Unsupported;
        c.global_head_dim = hd;
        c.num_global_key_value_heads = kv;
        listed[layer] = true;
        first = false;
    }
    if (!first) for (0..c.num_hidden_layers) |l| {
        if (c.isGlobalLayer(@intCast(l)) and !listed[l]) return error.Unsupported;
    };
}

// ── tests ──

const testing = std.testing;
const fixture = sdk.fixture;

pub fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

test "embedding_gemma2 config: the tiny release config parses into an encoder with images, video and its geometry" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(@embedFile("../../fixtures/embedding_gemma2_tiny_config.json"), &p).isOk());
    defer freeParsed(&p);
    try testing.expect(p.caps.embed and p.caps.images and p.caps.video and !p.caps.prefix_cache);
    try testing.expect(p.embed.has_projection);
    try testing.expectEqual(@as(u32, 40), p.placeholder_ids.image);
    try testing.expectEqual(@as(u32, 41), p.placeholder_ids.video);
    try testing.expectEqualSlices(u32, &.{1}, p.shell.eos[0..p.shell.eos_count]);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 16), c.global_head_dim);
    try testing.expectEqual(@as(u32, 1), c.num_global_key_value_heads);
    try testing.expect(c.isGlobalLayer(2) and !c.isGlobalLayer(3) and c.has_v_norm and !c.norm_has_offset);
}

test "embedding_gemma2 config: a per-layer geometry on a sliding layer or two disagreeing full layers is declined" {
    const base = @embedFile("../../fixtures/embedding_gemma2_tiny_config.json");
    for ([_][2][]const u8{
        .{ "\"02\": {", "\"01\": {\"head_dim\": 16, \"num_key_value_heads\": 1}, \"02\": {" },
        .{ "\"05\": {\n        \"head_dim\": 16", "\"05\": {\n        \"head_dim\": 32" },
    }) |r| {
        const json = try std.mem.replaceOwned(u8, testing.allocator, base, r[0], r[1]);
        defer testing.allocator.free(json);
        try testing.expect(!std.mem.eql(u8, json, base));
        var p: abi.Parsed = .{};
        try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(json, &p).code);
    }
}
