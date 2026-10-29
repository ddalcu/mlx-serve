//! LFM2 packs (`lfm2`, `lfm2_moe`, any `lfm2*` type, the engine's prefix rule): the claim and the parse into
//! `Parsed` plus this plugin's own fields. MLX-free. A shape the plugin does not serve bit-equal yet (the LFM2-VL
//! tower, an embedding pack, a layer span) is refused `unsupported` and keeps the legacy path.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const json = @import("../../sdk/config/json.zig");

/// `lfm2-vl` is mlx-community's spelling (LFM2-VL-1.6B-4bit): the engine reads it as text, never as `lfm2_vl`.
pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "lfm2",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{ "mlx-community/LFM2.5-1.2B-Instruct-4bit", "mlx-community/LFM2.5-8B-A1B-4bit" },
};

pub const model_types = [_][]const u8{ "lfm2", "lfm2_moe", "lfm2_vl", "lfm2-vl" };

/// Every `lfm2*` model_type, as the engine's parse dispatches them.
pub fn claims(model_type: abi.Str, _: abi.Str, _: *const abi.PackDir) callconv(.c) abi.Claim {
    return if (std.mem.startsWith(u8, model_type.slice(), "lfm2")) .by_type else .no;
}

/// No block of the hybrid stack serves batched rows (the legacy decodes LFM2 serially).
pub const batches_decode = false;

pub const max_layers = 128;

pub const LayerKind = enum(u8) { conv, attention };

/// lfm2_moe's sparse blocks past `num_dense_layers`: sigmoid routing with a selection-only `expert_bias`.
pub const Moe = struct {
    experts: u32,
    top_k: u32,
    intermediate: u32,
    dense_layers: u32,
    norm: bool = true,
    scale: f32 = 1.0,
};

/// The fields the forward and the load read.
pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: u32,
    conv_kernel: u32 = 3,
    tie_word_embeddings: bool = true,
    layers: [max_layers]LayerKind = @splat(.attention),
    moe: ?Moe = null,
    /// `language_model.model` when the decoder nests under a `text_config`, else `model`.
    prefix: []const u8 = "model",
    quant: abi.Quant = .{},
    /// The quant format's `open` block.
    block_json: []const u8 = "{}",

    pub fn moeAt(c: *const Config, layer: u32) ?Moe {
        const m = c.moe orelse return null;
        return if (layer >= m.dense_layers) m else null;
    }
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "lfm2: " ++ why);
}

fn invalid(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.failed, "lfm2: " ++ why);
}

/// Keys whose presence means a shape this plugin does not serve here (an embedding pack).
const foreign_keys = [_][]const u8{ "pooling_mode", "use_bidirectional_attention", "expert_quant", "modules" };

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = json.parseRoot(arena.allocator(), config_json.slice()) catch return invalid("config.json is not a JSON object");
    const type_name = json.string(json.field(root, "model_type")) orelse return unsupported("no model_type");
    if (!std.mem.startsWith(u8, type_name, "lfm2")) return unsupported("another model_type");
    const cfg = if (json.field(root, "text_config")) |tc| (json.object(tc) orelse return invalid("text_config")) else root;
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
    for (foreign_keys) |k| if (json.field(cfg, k) != null or json.field(root, k) != null) return unsupported("a key this plugin does not model");
    if (dir.has("1_Pooling/config.json")) return unsupported("an embedding pack (1_Pooling)");
    // LFM2-VL: the engine runs the tower and hands its rows in `Forward.media`; the trunk splices them at `<image>`.
    const vision = std.mem.eql(u8, type_name, "lfm2_vl") and json.object(json.field(root, "vision_config")) != null;
    // Without one the engine reads the tokenizer's `<image>`, which the driver hands the table (`placeholder_ids`).
    const image_id: u32 = if (!vision) 0 else json.int(json.field(root, "image_token_id")) orelse json.int(json.field(root, "image_token_index")) orelse 0;
    var c = readConfig(root, cfg, type_name) catch |e| return switch (e) {
        error.Unsupported => unsupported("a shape this plugin does not serve"),
        error.Incomplete => invalid("lfm2_moe without experts, top-k or expert width"),
        error.Invalid => invalid("a config value out of range"),
    };

    var p: abi.Parsed = .{ .caps = .{ .batches_decode = batches_decode, .moe = c.moe != null, .exact_resumed = true, .dflash_capture = true, .images = vision } };
    p.placeholder_ids.image = image_id;
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .max_position_embeddings = c.max_position_embeddings,
        .context_cap = c.max_position_embeddings,
        .quant = c.quant,
        .act_dtype = .bf16,
        .model_type = abi.Str.of("lfm2"),
    };
    if (c.moe) |m| {
        s.num_experts_per_tok = m.top_k;
        s.moe_intermediate_size = m.intermediate;
    }
    if (json.field(cfg, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        s.bos = json.int(v) orelse return invalid("bos_token_id");
    };
    json.eos(root, s) catch return invalid("eos_token_id");
    if (s.eos_count == 0) if (json.field(cfg, "eos_token_id")) |v| if (v == .integer) {
        generation.addEos(s, json.int(v) orelse return invalid("eos_token_id"));
    };
    p.cache.ssm_state = .{ .entries = c.num_hidden_layers, .dtype = .bf16 };
    p.weights.prefix = abi.Str.of(if (std.mem.eql(u8, c.prefix, "model")) "model" else "language_model.model");
    if (!stub) generation.apply(dir, s);

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

const ReadError = error{ Unsupported, Incomplete, Invalid };

/// The engine's reading of an LFM2 config (its common block, then the `lfm2` arm).
fn readConfig(root: json.Object, cfg: json.Object, type_name: []const u8) ReadError!Config {
    var c: Config = .{
        .vocab_size = (try json.intField(cfg, "vocab_size")) orelse 262208,
        .hidden_size = (try json.intField(cfg, "hidden_size")) orelse 3840,
        .intermediate_size = (try json.intField(cfg, "intermediate_size")) orelse 15360,
        .num_hidden_layers = (try json.intField(cfg, "num_hidden_layers")) orelse 48,
        .num_attention_heads = (try json.intField(cfg, "num_attention_heads")) orelse 16,
        .num_key_value_heads = (try json.intField(cfg, "num_key_value_heads")) orelse 8,
        .head_dim = (try json.intField(cfg, "head_dim")) orelse 256,
        .rms_norm_eps = @floatCast((try json.floatField(cfg, "rms_norm_eps")) orelse 1e-6),
        .rope_theta = @floatCast(try json.ropeTheta(cfg, 1000000.0)),
        .max_position_embeddings = try json.maxPositions(cfg, root),
        .prefix = if (json.field(root, "text_config") != null) "language_model.model" else "model",
    };
    // The engine's 256 is Gemma's default: an LFM2 pack's head is hidden / heads.
    if (c.head_dim == 256) {
        if (c.num_attention_heads == 0) return error.Invalid;
        c.head_dim = c.hidden_size / c.num_attention_heads;
    }
    if (json.boolean(json.field(cfg, "tie_embedding"))) |t| c.tie_word_embeddings = t;
    if (try json.floatField(cfg, "norm_eps")) |e| c.rms_norm_eps = @floatCast(e);
    if (try json.intIfInt(cfg, "conv_L_cache")) |k| c.conv_kernel = k;
    if (std.mem.eql(u8, type_name, "lfm2_moe")) {
        var m: Moe = .{
            .experts = (try json.intIfInt(cfg, "num_experts")) orelse 0,
            .top_k = (try json.intIfInt(cfg, "num_experts_per_tok")) orelse (try json.intIfInt(cfg, "top_k_experts")) orelse 0,
            .intermediate = (try json.intIfInt(cfg, "moe_intermediate_size")) orelse 0,
            .dense_layers = (try json.intIfInt(cfg, "num_dense_layers")) orelse 0,
        };
        if (json.boolean(json.field(cfg, "norm_topk_prob"))) |n| m.norm = n;
        if (try json.floatField(cfg, "routed_scaling_factor")) |f| m.scale = @floatCast(f);
        if (m.experts == 0 or m.top_k == 0 or m.intermediate == 0) return error.Incomplete;
        c.moe = m;
    }
    if (json.field(cfg, "layer_types")) |lt| if (lt == .array) for (lt.array.items, 0..) |item, i| {
        if (i >= max_layers) break;
        if (item != .string) continue;
        c.layers[i] = if (std.mem.eql(u8, item.string, "conv")) .conv else .attention;
    };
    c.quant = (try json.ggufQuant(root)) orelse json.quantOf(root, "quantization", true) catch |e| return if (e == error.Unsupported) error.Unsupported else error.Invalid;
    // The engine binds a dense pack's expert banks untransposed and its gather_mm refuses them: nothing to match.
    if (c.moe != null and c.quant.bits == 0) return error.Unsupported;
    if (c.vocab_size == 0 or c.hidden_size == 0 or c.num_hidden_layers == 0 or c.num_hidden_layers > max_layers or
        c.num_attention_heads == 0 or c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or
        c.head_dim == 0 or c.head_dim % 2 != 0 or c.conv_kernel < 2 or c.intermediate_size == 0)
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

/// One entry per layer: the conv layers keep a window, the attention layers full KV at the shell's geometry.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = @min(c.num_hidden_layers, cap);
    for (out[0..n], c.layers[0..n]) |*o, k| o.* = switch (k) {
        .conv => .{ .kind = .conv, .window = c.conv_kernel },
        .attention => .{ .kind = .full, .kv_heads = c.num_key_value_heads, .k_dim = c.head_dim, .v_dim = c.head_dim },
    };
    return n;
}

// ── tests ──

const testing = std.testing;
const fixture = sdk.fixture;

fn parseJson(js: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = js }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(js), .{}, null, true, out);
}

pub const lfm25_text =
    \\{"model_type":"lfm2","vocab_size":65536,"hidden_size":2048,"intermediate_size":8192,"num_hidden_layers":16,
    \\ "num_attention_heads":32,"num_key_value_heads":8,"max_position_embeddings":128000,"norm_eps":1e-5,
    \\ "rope_theta":1000000.0,"conv_L_cache":3,"tie_embedding":true,"bos_token_id":1,"eos_token_id":7,
    \\ "layer_types":["conv","conv","full_attention","conv","conv","full_attention","conv","conv","full_attention",
    \\   "conv","full_attention","conv","full_attention","conv","full_attention","conv"],
    \\ "quantization":{"group_size":64,"bits":4}}
;

test "lfm2 config: an LFM2.5 pack parses into the shell, its conv layers and its head width" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(lfm25_text, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 64), p.shell.head_dim);
    try testing.expectEqual(@as(f32, 1e-5), c.rms_norm_eps);
    try testing.expectEqual(LayerKind.attention, c.layers[2]);
    try testing.expectEqual(LayerKind.conv, c.layers[15]);
    try testing.expectEqualSlices(u32, &.{7}, p.shell.eos[0..p.shell.eos_count]);
    try testing.expect(p.caps.exact_resumed and !p.caps.batches_decode and !p.caps.moe);
    var lay: [17]abi.LayerCache = undefined;
    try testing.expectEqual(@as(u32, 16), layout(&p, null, &lay, lay.len));
    try testing.expectEqual(abi.LayerKind.conv, lay[0].kind);
    try testing.expectEqual(@as(u32, 8), lay[2].kv_heads);
}

test "lfm2 config: an embedding pack and a moe pack without experts are refused by name" {
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson("{\"model_type\":\"lfm2\",\"pooling_mode\":\"mean\"}", &p).code);
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("{\"model_type\":\"lfm2_moe\",\"hidden_size\":64}", &p).code);
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson("{\"model_type\":\"lfm2_moe\",\"num_experts\":4,\"num_experts_per_tok\":2,\"moe_intermediate_size\":8}", &p).code);
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("[1]", &p).code);
    try testing.expect(p.private == null);
}
