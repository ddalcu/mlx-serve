//! Nemotron-H packs (`nemotron_h`, the Puzzle latent-MoE trunk `nemotron_h_puzzle`, the omni wrapper
//! `nemotron_h_omni` whose `llm_config` holds the trunk): the claim and the parse into `Parsed` plus this plugin's own
//! fields. MLX-free. A compressed-tensors block and a layer span are refused `unsupported`; the RADIO tower is the engine's.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const json = @import("../../sdk/config/json.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "nemotron_h",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"},
};

pub const model_types = [_][]const u8{ "nemotron_h", "nemotron_h_puzzle", "nemotron_h_omni" };

pub const claims = sdk.claimsByType(&model_types);

/// No block of the hybrid stack serves batched rows (the legacy decodes Nemotron-H serially).
pub const batches_decode = false;

pub const max_layers = 128;

/// One layer's operation: `M` Mamba2, `*` attention (NoPE), `E` the routed MoE, `-` the ReLU² MLP.
pub const LayerKind = enum(u8) { mamba2, attention, moe, mlp };

pub const Mamba = struct {
    heads: u32 = 0,
    head_dim: u32 = 0,
    groups: u32 = 8,
    state: u32 = 128,
    kernel: u32 = 4,
    dt_min: f32 = 0.0,
    dt_max: f32 = std.math.inf(f32),
};

pub const Moe = struct {
    experts: u32 = 0,
    top_k: u32 = 0,
    intermediate: u32 = 0,
    shared_intermediate: u32 = 0,
    n_group: u32 = 1,
    topk_group: u32 = 1,
    norm: bool = true,
    scale: f32 = 1.0,
    /// The Puzzle trunk's expert width; 0 = experts read the full hidden.
    latent: u32 = 0,
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
    tie_word_embeddings: bool = false,
    layers: [max_layers]LayerKind = @splat(.attention),
    mamba: Mamba = .{},
    moe: Moe = .{},
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "nemotron_h: " ++ why);
}

fn invalid(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.failed, "nemotron_h: " ++ why);
}

const foreign_keys = [_][]const u8{ "pooling_mode", "use_bidirectional_attention", "expert_quant", "modules" };

/// The RADIO tower (`vision_config.model_type == "radio"`), as discovery reads it.
fn hasRadioVision(root: json.Object) bool {
    const vc = json.object(json.field(root, "vision_config")) orelse return false;
    const mt = json.string(json.field(vc, "model_type")) orelse return false;
    return std.mem.eql(u8, mt, "radio");
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = json.parseRoot(arena.allocator(), config_json.slice()) catch return invalid("config.json is not a JSON object");
    var type_name = json.string(json.field(root, "model_type")) orelse return unsupported("no model_type");
    var cfg = if (json.field(root, "text_config")) |tc| (json.object(tc) orelse return invalid("text_config")) else root;
    if (std.mem.eql(u8, type_name, "nemotron_h_omni")) if (json.object(json.field(root, "llm_config"))) |lc| {
        cfg = lc;
        if (json.string(json.field(lc, "model_type"))) |mt| type_name = mt;
    };
    if (!std.mem.eql(u8, type_name, "nemotron_h") and !std.mem.eql(u8, type_name, "nemotron_h_puzzle")) return unsupported("another model_type");
    for (foreign_keys) |k| if (json.field(cfg, k) != null or json.field(root, k) != null) return unsupported("a key this plugin does not model");
    // A compressed-tensors (or other non-MLX) block: no built-in quant format claims it.
    if (json.object(json.field(root, "quantization_config"))) |qc| if (json.field(qc, "quant_method") != null)
        return unsupported("a quantization_config block");
    var c = readConfig(root, cfg) catch |e| return switch (e) {
        error.Unsupported => unsupported("a shape or quantization mode this plugin does not serve"),
        error.Invalid => invalid("a config value out of range"),
    };

    // The RADIO tower is the engine's; its image and video rows splice at `<image>`, a tokenizer id the engine reads.
    const radio = hasRadioVision(root);
    var p: abi.Parsed = .{ .caps = .{ .batches_decode = batches_decode, .moe = c.moe.experts > 0, .exact_resumed = true, .dflash_capture = true, .exact_drafted = true, .images = radio, .video = radio } };
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .num_experts_per_tok = c.moe.top_k,
        .moe_intermediate_size = c.moe.intermediate,
        .max_position_embeddings = c.max_position_embeddings,
        .context_cap = c.max_position_embeddings,
        .quant = c.quant,
        .act_dtype = .bf16,
        .model_type = abi.Str.of("nemotron_h"),
    };
    if (json.field(cfg, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        s.bos = json.int(v) orelse return invalid("bos_token_id");
    };
    json.eos(root, s) catch return invalid("eos_token_id");
    if (s.eos_count == 0) if (json.field(cfg, "eos_token_id")) |v| if (v == .integer) {
        generation.addEos(s, json.int(v) orelse return invalid("eos_token_id"));
    };
    // NoPE keys: a fresh SSD root, so a pack's earlier roped entries never restore.
    p.cache.layout_namespace = abi.Str.of("nemotron-h-nope-v1");
    p.cache.ssm_state = .{ .entries = c.num_hidden_layers, .dtype = .bf16 };
    p.weights.prefix = abi.Str.of("backbone");
    // The bare-key `mtp_head.safetensors` head reads the final-normed hidden; depth 2 when auto.
    p.spec = .{ .mtp_layout = .nemotron, .head_reads_final_norm = true, .mtp_default_depth = 2 };
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

/// `time_step_limit`'s bound: a number, or positive infinity as the string or the `__float__` object.
fn ssmBound(v: json.Value, allow_infinity: bool) error{Invalid}!f32 {
    switch (v) {
        .integer, .float => return @floatCast(json.float(v) orelse return error.Invalid),
        .string => |name| if (allow_infinity and std.mem.eql(u8, name, "Infinity")) return std.math.inf(f32),
        .object => |obj| if (allow_infinity and obj.count() == 1) if (obj.get("__float__")) |x| {
            if (x == .string and std.mem.eql(u8, x.string, "Infinity")) return std.math.inf(f32);
        },
        else => {},
    }
    return error.Invalid;
}

/// An integer key the engine reads as `default` when it holds any other value.
fn intOr(o: json.Object, key: []const u8, default: u32) error{Invalid}!?u32 {
    const v = json.field(o, key) orelse return null;
    if (v != .integer) return default;
    return json.int(v) orelse error.Invalid;
}

/// The engine's reading of a Nemotron-H config (its common block, then the `nemotron_h` arm).
fn readConfig(root: json.Object, cfg: json.Object) error{ Unsupported, Invalid }!Config {
    var c: Config = .{
        .vocab_size = (try json.intField(cfg, "vocab_size")) orelse 262208,
        .hidden_size = (try json.intField(cfg, "hidden_size")) orelse 3840,
        .intermediate_size = (try json.intField(cfg, "intermediate_size")) orelse 15360,
        .num_hidden_layers = (try json.intField(cfg, "num_hidden_layers")) orelse 48,
        .num_attention_heads = (try json.intField(cfg, "num_attention_heads")) orelse 16,
        .num_key_value_heads = (try json.intField(cfg, "num_key_value_heads")) orelse 8,
        .head_dim = (try json.intField(cfg, "head_dim")) orelse 256,
        .rms_norm_eps = 1e-6,
        .rope_theta = @floatCast(try json.ropeTheta(cfg, 1000000.0)),
        .max_position_embeddings = try json.maxPositions(cfg, root),
    };
    if (json.boolean(json.field(root, "tie_word_embeddings"))) |t| c.tie_word_embeddings = t;
    if (json.boolean(json.field(cfg, "tie_word_embeddings"))) |t| c.tie_word_embeddings = t;
    // Bolt's llm_config omits num_hidden_layers: the block list is the depth.
    if (json.field(cfg, "num_hidden_layers") == null) {
        const listed: usize = if (json.field(cfg, "layers_block_type")) |v|
            (if (v == .array) v.array.items.len else 0)
        else if (json.field(cfg, "block_configs")) |v|
            (if (v == .array) v.array.items.len else 0)
        else if (json.string(json.field(cfg, "hybrid_override_pattern"))) |pat| pat.len else 0;
        if (listed > max_layers) return error.Unsupported;
        if (listed > 0) c.num_hidden_layers = @intCast(listed);
    }
    if (try json.floatField(cfg, "rms_norm_eps")) |e| {
        c.rms_norm_eps = @floatCast(e);
    } else if (try json.floatField(cfg, "layer_norm_epsilon")) |e| c.rms_norm_eps = @floatCast(e);

    const m = &c.mamba;
    if (json.field(cfg, "mamba_num_heads")) |v| m.heads = if (v == .integer) json.int(v) orelse return error.Invalid else 0;
    if (json.field(cfg, "mamba_head_dim")) |v| m.head_dim = if (v == .integer) json.int(v) orelse return error.Invalid else 0;
    if (try intOr(cfg, "n_groups", 8)) |v| m.groups = v;
    if (try intOr(cfg, "ssm_state_size", 128)) |v| m.state = v;
    if (try intOr(cfg, "conv_kernel", 4)) |v| m.kernel = v;
    if (json.field(cfg, "time_step_limit")) |v| {
        if (v != .array or v.array.items.len != 2) return error.Invalid;
        m.dt_min = try ssmBound(v.array.items[0], false);
        m.dt_max = try ssmBound(v.array.items[1], true);
        if (m.dt_min < 0 or m.dt_max < m.dt_min) return error.Invalid;
    }

    if (json.string(json.field(cfg, "hybrid_override_pattern"))) |pat| {
        for (pat, 0..) |ch, i| {
            if (i >= max_layers) break;
            c.layers[i] = switch (ch) {
                'M' => .mamba2,
                '-' => .mlp,
                'E' => .moe,
                else => .attention,
            };
        }
    } else if (json.field(cfg, "layers_block_type")) |v| if (v == .array and json.field(cfg, "hybrid_override_pattern") == null) {
        for (v.array.items, 0..) |item, i| {
            if (i >= max_layers) break;
            if (item != .string) continue;
            const name = item.string;
            c.layers[i] = if (std.mem.eql(u8, name, "mamba")) .mamba2 else if (std.mem.eql(u8, name, "mlp")) .mlp else if (std.mem.eql(u8, name, "moe")) .moe else if (std.mem.eql(u8, name, "attention") or std.mem.eql(u8, name, "full_attention")) .attention else
                // The engine reads any other name as attention (`linear_attention`, the Mamba layers of transformers
                // 5.15's configs, among them): declined, so that pack keeps the legacy reading.
                return error.Unsupported;
        }
    };

    const e = &c.moe;
    e.top_k = (try json.intIfInt(cfg, "num_experts_per_tok")) orelse (try json.intIfInt(cfg, "top_k_experts")) orelse 0;
    if (try json.intIfInt(cfg, "top_k_experts")) |k| e.top_k = k;
    e.intermediate = (try json.intIfInt(cfg, "moe_intermediate_size")) orelse 0;
    e.experts = (try json.intIfInt(cfg, "num_experts")) orelse 0;
    if (try json.intIfInt(cfg, "n_routed_experts")) |n| e.experts = n;
    e.shared_intermediate = (try json.intIfInt(cfg, "shared_expert_intermediate_size")) orelse 0;
    if (try json.intIfInt(cfg, "moe_shared_expert_intermediate_size")) |n| e.shared_intermediate = n;
    if (try json.intIfInt(cfg, "n_group")) |n| e.n_group = n;
    if (try json.intIfInt(cfg, "topk_group")) |n| e.topk_group = n;
    if (json.boolean(json.field(cfg, "norm_topk_prob"))) |n| e.norm = n;
    if (try json.floatField(cfg, "routed_scaling_factor")) |f| e.scale = @floatCast(f);
    if (try json.intIfInt(cfg, "moe_latent_size")) |n| e.latent = n;
    // Per-layer expert banks: the forward reads each tensor's own shape; these are the maxima, for the bill.
    if (json.field(cfg, "block_configs")) |v| if (v == .array) {
        var max_experts: u32 = 0;
        var max_inter: u32 = 0;
        for (v.array.items) |item| {
            if (item != .object) continue;
            if (try json.intIfInt(item.object, "n_routed_experts")) |n| max_experts = @max(max_experts, n);
            if (try json.intIfInt(item.object, "moe_intermediate_size")) |n| max_inter = @max(max_inter, n);
            if (try json.intIfInt(item.object, "num_experts_per_tok")) |n| e.top_k = n;
        }
        if (max_experts > 0) e.experts = max_experts;
        if (max_inter > 0) e.intermediate = max_inter;
    };

    c.quant = (try json.ggufQuant(root)) orelse json.quantOf(root, "quantization", true) catch |err| return if (err == error.Unsupported) error.Unsupported else error.Invalid;
    if (c.vocab_size == 0 or c.hidden_size == 0 or c.num_hidden_layers == 0 or c.num_hidden_layers > max_layers or
        c.num_attention_heads == 0 or c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or
        c.head_dim == 0 or c.head_dim % 2 != 0)
        return error.Unsupported;
    for (c.layers[0..c.num_hidden_layers]) |k| switch (k) {
        .mamba2 => if (m.heads == 0 or m.head_dim == 0 or m.groups == 0 or m.state == 0 or m.kernel < 2) return error.Unsupported,
        .moe => if (e.experts == 0 or e.top_k == 0) return error.Unsupported,
        else => {},
    };
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

/// One entry per layer: Mamba2 keeps a conv window and a scan state, attention full NoPE KV, the MLPs nothing.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = @min(c.num_hidden_layers, cap);
    for (out[0..n], c.layers[0..n]) |*o, k| o.* = switch (k) {
        .mamba2 => .{ .kind = .ssm, .window = c.mamba.kernel },
        .attention => .{ .kind = .full, .nope = true, .kv_heads = c.num_key_value_heads, .k_dim = c.head_dim, .v_dim = c.head_dim },
        .moe, .mlp => .{ .kind = .none },
    };
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

test "nemotron_h config: a pattern list, the Mamba2 geometry and the infinite dt bound parse" {
    const js =
        \\{"model_type":"nemotron_h","hidden_size":2688,"num_hidden_layers":4,"num_attention_heads":32,
        \\ "num_key_value_heads":2,"head_dim":128,"vocab_size":131072,"mamba_num_heads":64,"mamba_head_dim":64,
        \\ "layers_block_type":["mamba","moe","full_attention","mlp"],"n_routed_experts":128,"num_experts_per_tok":6,
        \\ "time_step_limit":[0.0,{"__float__":"Infinity"}]}
    ;
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(js, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqualSlices(LayerKind, &.{ .mamba2, .moe, .attention, .mlp }, c.layers[0..4]);
    try testing.expectEqual(std.math.inf(f32), c.mamba.dt_max);
    try testing.expect(p.caps.moe and p.caps.exact_resumed and !p.caps.batches_decode);
    try testing.expectEqualStrings("nemotron-h-nope-v1", p.cache.layout_namespace.slice());
}

test "nemotron_h config: the Nano fixture (compressed-tensors, linear_attention names) keeps the legacy path" {
    const nano = @embedFile("../../fixtures/model-configs/nemotron-h.json");
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(nano, &p).code);
    const plain = try std.mem.replaceOwned(u8, testing.allocator, nano, "\"quantization_config\"", "\"quantization_config_off\"");
    defer testing.allocator.free(plain);
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(plain, &p).code);
}

test "nemotron_h config: the omni wrapper's RADIO tower declares images and video for the engine's tower; a bad dt bound fails" {
    var omni: abi.Parsed = .{};
    try testing.expect(parseJson(@embedFile("../../fixtures/model-configs/nemotron_h_super_omni.json"), &omni).isOk());
    try testing.expect(omni.caps.images and omni.caps.video and omni.placeholder_ids.image == 0);
    freeParsed(&omni);
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("{\"model_type\":\"nemotron_h\",\"time_step_limit\":[0.5,0.25]}", &p).code);
    try testing.expect(p.private == null);
}
