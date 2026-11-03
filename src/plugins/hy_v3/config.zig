//! Tencent Hunyuan 3 packs (`hy_v3`, mlx-lm's layout): the claim and the parse into `Parsed` plus this plugin's own
//! fields. MLX-free. A shape the plugin does not serve bit-equal yet (no q/k norm, a layer span) is refused
//! `unsupported` and keeps the legacy path.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const json = @import("../../sdk/config/json.zig");

pub const plugin = sdk.Plugin{
    .name = "hy_v3",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/Hy3-oQ2e"},
};

pub const model_types = [_][]const u8{"hy_v3"};

pub const claims = sdk.claimsByType(&model_types);

/// `<｜hy_eos:opensource｜>`: the family's terminator, merged into whatever the config declares.
pub const hy_eos: u32 = 120025;

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
    partial_rotary_factor: f32 = 1.0,
    query_pre_attn_scalar: u32,
    max_position_embeddings: u32,
    tie_word_embeddings: bool = false,
    experts: u32 = 0,
    top_k: u32 = 0,
    moe_intermediate: u32 = 0,
    /// `num_shared_experts` x the expert width; 0 = no shared expert.
    shared_intermediate: u32 = 0,
    route_norm: bool = true,
    route_scale: f32 = 1.0,
    /// Layers below this one are dense.
    first_k_dense: u32 = 0,
    /// Per-head q/k RMS norms (`qk_norm`, default on).
    qk_norm: bool = true,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",

    pub fn moeAt(c: *const Config, layer: u32) bool {
        return c.experts > 0 and layer >= c.first_k_dense;
    }
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "hy_v3: " ++ why);
}

fn invalid(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.failed, "hy_v3: " ++ why);
}

const foreign_keys = [_][]const u8{ "pooling_mode", "use_bidirectional_attention", "expert_quant", "modules" };

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = json.parseRoot(arena.allocator(), config_json.slice()) catch return invalid("config.json is not a JSON object");
    const type_name = json.string(json.field(root, "model_type")) orelse return unsupported("no model_type");
    if (!std.mem.eql(u8, type_name, "hy_v3")) return unsupported("another model_type");
    const cfg = if (json.field(root, "text_config")) |tc| (json.object(tc) orelse return invalid("text_config")) else root;
    for (foreign_keys) |k| if (json.field(cfg, k) != null or json.field(root, k) != null) return unsupported("a key this plugin does not model");
    var c = readConfig(root, cfg) catch |e| return switch (e) {
        error.Unsupported => unsupported("a shape this plugin does not serve"),
        error.Invalid => invalid("a config value out of range"),
    };

    // A tower the engine reads as its generic (SigLIP) one: its rows splice at the image token.
    const vision = json.field(root, "vision_config") != null;
    var p: abi.Parsed = .{ .caps = .{ .moe = c.experts > 0, .dflash_capture = true, .images = vision } };
    // The `mtp.*` head in `model-mtp.safetensors` (or the trunk's shards).
    p.spec = .{ .mtp_layout = .sidecar };
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .num_experts_per_tok = c.top_k,
        .moe_intermediate_size = c.moe_intermediate,
        .max_position_embeddings = c.max_position_embeddings,
        .context_cap = c.max_position_embeddings,
        .quant = c.quant,
        .act_dtype = .bf16,
        .model_type = abi.Str.of("hy_v3"),
    };
    if (json.field(cfg, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        s.bos = json.int(v) orelse return invalid("bos_token_id");
    };
    json.eos(root, s) catch return invalid("eos_token_id");
    generation.mergeEos(s, hy_eos);
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

/// The engine's reading of a Hunyuan 3 config (its common block, then the `hy_v3` arm).
fn readConfig(root: json.Object, cfg: json.Object) error{ Unsupported, Invalid }!Config {
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
        .query_pre_attn_scalar = 0,
        .max_position_embeddings = try json.maxPositions(cfg, root),
    };
    c.query_pre_attn_scalar = (try json.intField(cfg, "query_pre_attn_scalar")) orelse c.head_dim;
    if (json.object(json.field(cfg, "rope_parameters"))) |rp| if (try json.floatField(rp, "partial_rotary_factor")) |f| {
        c.partial_rotary_factor = @floatCast(f);
    };
    if (json.boolean(json.field(root, "tie_word_embeddings"))) |t| c.tie_word_embeddings = t;
    if (json.boolean(json.field(cfg, "tie_word_embeddings"))) |t| c.tie_word_embeddings = t;
    c.experts = (try json.intIfInt(cfg, "num_experts")) orelse 0;
    c.top_k = (try json.intIfInt(cfg, "num_experts_per_tok")) orelse 0;
    if (try json.intIfInt(cfg, "top_k_experts")) |k| c.top_k = k;
    c.moe_intermediate = (try json.intIfInt(cfg, "moe_intermediate_size")) orelse 0;
    // The expert width may ride as `expert_hidden_dim` when `moe_intermediate_size` is absent.
    if (c.moe_intermediate == 0) if (try json.intIfInt(cfg, "expert_hidden_dim")) |n| {
        c.moe_intermediate = n;
    };
    c.shared_intermediate = (try json.intIfInt(cfg, "shared_expert_intermediate_size")) orelse 0;
    if (try json.intIfInt(cfg, "num_shared_experts")) |n| c.shared_intermediate = std.math.mul(u32, n, c.moe_intermediate) catch return error.Invalid;
    if (json.boolean(json.field(cfg, "route_norm"))) |n| c.route_norm = n;
    if (try json.floatField(cfg, "router_scaling_factor")) |f| c.route_scale = @floatCast(f);
    if (try json.intIfInt(cfg, "first_k_dense_replace")) |n| c.first_k_dense = n;
    // Off: the engine norms q and k without a weight (null norms in `GatedAttention`), where mlx-lm skips the norm.
    if (json.boolean(json.field(cfg, "qk_norm"))) |q| c.qk_norm = q;
    // mlx-lm copies `quantization` to `quantization_config`, which the engine never reads; a block with a
    // `quant_method` is a format neither reads (the engine took its tensors as plain ones).
    if (json.object(json.field(root, "quantization_config"))) |o| if (json.field(o, "quant_method") != null) return error.Unsupported;
    c.quant = json.quantOf(root, "quantization", true) catch |e| return if (e == error.Unsupported) error.Unsupported else error.Invalid;
    if (c.vocab_size == 0 or c.hidden_size == 0 or c.num_hidden_layers == 0 or c.num_attention_heads == 0 or
        c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or c.head_dim == 0 or
        c.head_dim % 2 != 0 or c.query_pre_attn_scalar == 0 or c.intermediate_size == 0 or
        (c.experts > 0 and (c.top_k == 0 or c.moe_intermediate == 0)))
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

// ── tests ──

const testing = std.testing;
const fixture = sdk.fixture;

pub fn parseJson(js: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = js }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(js), .{}, null, true, out);
}

pub const hy3_295b =
    \\{"model_type":"hy_v3","vocab_size":120832,"hidden_size":4096,"intermediate_size":13312,"num_hidden_layers":80,
    \\ "num_attention_heads":64,"num_key_value_heads":8,"head_dim":128,"rms_norm_eps":1e-5,"rope_theta":11158840.0,
    \\ "max_position_embeddings":262144,"num_experts":192,"num_experts_per_tok":8,"expert_hidden_dim":1536,
    \\ "num_shared_experts":1,"router_scaling_factor":2.826,"first_k_dense_replace":1,"eos_token_id":120020,
    \\ "quantization":{"group_size":64,"bits":4}}
;

test "hy_v3 config: the 295B config parses its experts, shared width, dense prefix and terminator" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(hy3_295b, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 1536), c.moe_intermediate);
    try testing.expectEqual(@as(u32, 1536), c.shared_intermediate);
    try testing.expect(!c.moeAt(0) and c.moeAt(1) and c.route_norm);
    try testing.expectEqualSlices(u32, &.{ 120020, hy_eos }, p.shell.eos[0..p.shell.eos_count]);
    try testing.expect(p.caps.moe and !p.caps.batches_decode);
}

test "hy_v3 config: an HF quant_method is refused unsupported, broken JSON failed" {
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson("{\"model_type\":\"hy_v3\",\"quantization_config\":{\"quant_method\":\"fp8\"}}", &p).code);
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("{", &p).code);
}
