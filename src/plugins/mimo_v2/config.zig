//! Xiaomi MiMo-V2 packs (`mimo_v2`, `mimo_v2_flash`; mlx-lm's layout, `tests/convert_mimo_v2.py`): the claim and the
//! parse into `Parsed` plus this plugin's own fields. MLX-free. A Sushi pack (EXL3 experts under `expert_quant`,
//! an e4m3 trunk) is served, and so is a streaming pack (its experts in an MXFP4 `experts.bin`, `expert_streaming`); the
//! original release is refused by name.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const json = @import("../../sdk/config/json.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "mimo_v2",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/MiMo-V2.6-Flash-mxfp4-q8"},
};

pub const model_types = [_][]const u8{ "mimo_v2", "mimo_v2_flash" };

pub const claims = sdk.claimsByType(&model_types);

pub const max_layers = 128;

/// The rows a sliding layer's ring keeps beyond its window (the engine's `SLIDING_RING_SLACK`).
pub const ring_slack: u32 = 2048;

/// The fields the forward and the load read.
/// MiMo-V2's MTP heads the plugin serves.
pub const mtp_max_heads: u32 = 3;

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    /// Sliding layers' KV heads (`swa_num_key_value_heads`); global layers read `global_kv_heads`.
    num_key_value_heads: u32,
    global_kv_heads: u32,
    head_dim: u32,
    v_head_dim: u32,
    rms_norm_eps: f32,
    /// Global layers' theta; sliding layers' is `swa_rope_theta`.
    rope_theta: f32,
    swa_rope_theta: f32 = 10000.0,
    partial_rotary_factor: f32 = 1.0,
    /// 0 = every layer global (`sliding_window: null`).
    sliding_window: u32 = 1024,
    global: [max_layers]bool = @splat(true),
    max_position_embeddings: u32,
    tie_word_embeddings: bool = false,
    value_scale: f32 = 1.0,
    experts: u32 = 0,
    top_k: u32 = 0,
    moe_intermediate: u32 = 0,
    route_norm: bool = true,
    route_scale: f32 = 1.0,
    /// The dense MLP layers lead the stack.
    first_k_dense: u32 = 0,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",
    /// A Sushi pack's `expert_quant` block (its routed experts are EXL3 banks); empty = none.
    expert_block_json: []const u8 = "",
    /// The routed experts are read from the pack's MXFP4 bank (`experts.bin`), the MTP heads from its sidecar.
    streamed: bool = false,

    pub fn moeAt(c: *const Config, layer: u32) bool {
        return c.experts > 0 and layer >= c.first_k_dense;
    }

    pub fn isGlobal(c: *const Config, layer: u32) bool {
        return c.sliding_window == 0 or c.global[layer];
    }

    pub fn kvHeads(c: *const Config, layer: u32) u32 {
        return if (c.isGlobal(layer)) c.global_kv_heads else c.num_key_value_heads;
    }

    pub fn ropeDims(c: *const Config) u32 {
        return @intFromFloat(@as(f32, @floatFromInt(c.head_dim)) * c.partial_rotary_factor);
    }

    pub fn ring(c: *const Config) u32 {
        return if (c.sliding_window > 0) c.sliding_window + ring_slack else 0;
    }
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "mimo_v2: " ++ why);
}

fn invalid(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.failed, "mimo_v2: " ++ why);
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = json.parseRoot(arena.allocator(), config_json.slice()) catch return invalid("config.json is not a JSON object");
    const type_name = json.string(json.field(root, "model_type")) orelse return unsupported("no model_type");
    if (!std.mem.eql(u8, type_name, "mimo_v2") and !std.mem.eql(u8, type_name, "mimo_v2_flash")) return unsupported("another model_type");
    const streamed = json.field(root, "expert_streaming") != null;
    if (streamed and !isBank(json.object(json.field(root, "expert_streaming"))))
        return unsupported("an expert_streaming block other than mlx-stream's MXFP4 expert bank");
    if (json.field(root, "residents_format")) |v| if (!std.mem.eql(u8, json.string(v) orelse "", "fp8_block128"))
        return unsupported("a residents_format other than fp8_block128");
    for ([_][]const u8{ "text_config", "modules" }) |k| if (json.field(root, k) != null) return unsupported("a key this plugin does not model");
    // A Sushi pack's `quant_method: fp8` describes its trunk; its experts are EXL3.
    const sushi = json.object(json.field(root, "expert_quant"));
    if (sushi == null) if (json.object(json.field(root, "quantization_config"))) |qc| if (json.field(qc, "quant_method") != null)
        return invalid("this is the original release; convert it first with tests/convert_mimo_v2.py");
    var c = readConfig(root) catch |e| return switch (e) {
        error.Unsupported => unsupported("a shape this plugin does not serve"),
        error.Invalid => invalid("a config value out of range"),
    };

    var p: abi.Parsed = .{ .caps = .{ .moe = c.experts > 0, .batches_decode = true, .dflash_capture = true, .verify_truncates = true } };
    p.cache.sliding_ring = c.ring();
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
        .model_type = abi.Str.of("mimo_v2"),
    };
    if (json.field(root, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        s.bos = json.int(v) orelse return invalid("bos_token_id");
    };
    json.eos(root, s) catch return invalid("eos_token_id");
    if (!stub) generation.apply(dir, s);
    // The template thinks unless told `enable_thinking: false`.
    p.chat.thinking_default = .on;
    // One MTP head per draft depth (`model.mtp.layers.{k}`), each reading the trunk's final-normed hidden.
    const heads = @min(json.int(json.field(root, "num_nextn_predict_layers")) orelse 0, mtp_max_heads);
    if (heads > 0) p.spec = .{ .mtp_layout = .in_checkpoint, .head_reads_final_norm = true, .mtp_heads = heads };

    const cfg_out = gpa.create(Config) catch return invalid("out of memory");
    c.block_json = json.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return invalid("out of memory");
    };
    if (sushi) |eq| c.expert_block_json = std.json.Stringify.valueAlloc(gpa, json.Value{ .object = eq }, .{}) catch {
        gpa.free(c.block_json);
        gpa.destroy(cfg_out);
        return invalid("out of memory");
    };
    c.streamed = streamed;
    cfg_out.* = c;
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

/// mlx-stream's MXFP4 expert bank (`experts.bin` beside `expert-manifest.json`).
fn isBank(es: ?json.Object) bool {
    const o = es orelse return false;
    const want = [_][2][]const u8{ .{ "family", "mlx-stream" }, .{ "format", "mlx-stream-mxfp4-expert-bank-v1" }, .{ "manifest", "expert-manifest.json" } };
    for (want) |kv| if (!std.mem.eql(u8, json.string(json.field(o, kv[0])) orelse "", kv[1])) return false;
    return true;
}

/// The engine's reading of a MiMo config (its common block, then the `mimo_v2` arm).
fn readConfig(root: json.Object) error{ Unsupported, Invalid }!Config {
    var c: Config = .{
        .vocab_size = (try json.intField(root, "vocab_size")) orelse 262208,
        .hidden_size = (try json.intField(root, "hidden_size")) orelse 3840,
        .intermediate_size = (try json.intField(root, "intermediate_size")) orelse 15360,
        .num_hidden_layers = (try json.intField(root, "num_hidden_layers")) orelse 48,
        .num_attention_heads = (try json.intField(root, "num_attention_heads")) orelse 16,
        .num_key_value_heads = (try json.intField(root, "num_key_value_heads")) orelse 8,
        .global_kv_heads = 0,
        .head_dim = (try json.intField(root, "head_dim")) orelse 256,
        .v_head_dim = 0,
        .rms_norm_eps = @floatCast((try json.floatField(root, "rms_norm_eps")) orelse 1e-6),
        .rope_theta = @floatCast(try json.ropeTheta(root, 1000000.0)),
        .max_position_embeddings = try json.maxPositions(root, root),
    };
    if (try json.floatField(root, "layernorm_epsilon")) |e| c.rms_norm_eps = @floatCast(e);
    if (json.object(json.field(root, "rope_parameters"))) |rp| if (try json.floatField(rp, "partial_rotary_factor")) |f| {
        c.partial_rotary_factor = @floatCast(f);
    };
    if (root.get("sliding_window")) |v| c.sliding_window = if (v == .null) 0 else (json.int(v) orelse return error.Invalid);
    if (try json.floatField(root, "rope_local_base_freq")) |f| c.swa_rope_theta = @floatCast(f);
    if (json.boolean(json.field(root, "tie_word_embeddings"))) |t| c.tie_word_embeddings = t;
    if (try json.floatField(root, "attention_value_scale")) |f| c.value_scale = @floatCast(f);

    // Experts: laguna's sigmoid routing with a selection-only bias and no shared expert.
    c.experts = (try json.intIfInt(root, "num_experts")) orelse 0;
    if (try json.intField(root, "n_routed_experts")) |n| c.experts = n;
    c.top_k = (try json.intIfInt(root, "num_experts_per_tok")) orelse 0;
    if (try json.intIfInt(root, "top_k_experts")) |k| c.top_k = k;
    c.moe_intermediate = (try json.intIfInt(root, "moe_intermediate_size")) orelse 0;
    if (json.boolean(json.field(root, "norm_topk_prob"))) |n| c.route_norm = n;
    if (try json.floatField(root, "routed_scaling_factor")) |f| c.route_scale = @floatCast(f);
    if (root.get("n_shared_experts")) |v| if (v == .integer and v.integer != 0) return error.Invalid;
    // Group-limited routing is the engine's own arm; every shipped pack routes in one group.
    if (((try json.intField(root, "n_group")) orelse 1) > 1) return error.Unsupported;

    // Pattern 0 = global, 1 = sliding; the dense MLP layers lead the stack.
    const pattern = arr(root, "hybrid_layer_pattern") orelse return error.Invalid;
    const freq = arr(root, "moe_layer_freq") orelse return error.Invalid;
    if (pattern.len != c.num_hidden_layers or freq.len != c.num_hidden_layers or c.num_hidden_layers > max_layers) return error.Invalid;
    for (pattern, freq, 0..) |pv, fv, i| {
        c.global[i] = (json.int(pv) orelse return error.Invalid) == 0;
        const moe = (json.int(fv) orelse return error.Invalid) != 0;
        if (!moe and c.first_k_dense != i) return error.Invalid;
        if (!moe) c.first_k_dense += 1;
    }

    // Root KV heads / theta are the GLOBAL layers'; `swa_*` the sliding ones.
    c.global_kv_heads = c.num_key_value_heads;
    if (try json.intField(root, "swa_num_key_value_heads")) |n| c.num_key_value_heads = n;
    if (try json.floatField(root, "swa_rope_theta")) |t| c.swa_rope_theta = @floatCast(t);
    if (try json.intField(root, "v_head_dim")) |v| c.v_head_dim = v;
    if (c.v_head_dim == 0) c.v_head_dim = c.head_dim;
    for ([_]struct { []const u8, u32 }{ .{ "swa_head_dim", c.head_dim }, .{ "swa_v_head_dim", c.v_head_dim }, .{ "swa_num_attention_heads", c.num_attention_heads } }) |kv| {
        if (try json.intField(root, kv[0])) |v| if (v != kv[1]) return error.Invalid;
    }
    c.quant = json.quant(root) catch |e| return if (e == error.Unsupported) error.Unsupported else error.Invalid;
    if (c.vocab_size == 0 or c.hidden_size == 0 or c.num_attention_heads == 0 or c.num_key_value_heads == 0 or
        c.global_kv_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or
        c.num_attention_heads % c.global_kv_heads != 0 or c.head_dim == 0 or c.ropeDims() % 2 != 0 or
        (c.experts > 0 and (c.top_k == 0 or c.moe_intermediate == 0)))
        return error.Unsupported;
    return c;
}

fn arr(o: json.Object, key: []const u8) ?[]const json.Value {
    const v = json.field(o, key) orelse return null;
    return if (v == .array) v.array.items else null;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| {
        const c: *Config = @ptrCast(@alignCast(ptr));
        gpa.free(c.block_json);
        if (c.expert_block_json.len > 0) gpa.free(c.expert_block_json);
        gpa.destroy(c);
    }
    p.private = null;
}

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

/// Global layers keep every row at their own KV heads; sliding layers a ring of window + slack rows.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = @min(c.num_hidden_layers, cap);
    for (out[0..n], 0..) |*o, i| {
        const l: u32 = @intCast(i);
        o.* = .{ .kind = if (c.isGlobal(l)) .full else .sliding, .window = if (c.isGlobal(l)) 0 else c.sliding_window, .kv_heads = c.kvHeads(l), .k_dim = c.head_dim, .v_dim = c.v_head_dim };
    }
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

/// MiMo-V2.6-Flash as mlx-community packs it (mxfp4-q8), layers trimmed to 6.
pub const pack_json =
    \\{"model_type":"mimo_v2","add_full_attention_sink_bias":false,"add_swa_attention_sink_bias":true,
    \\ "attention_projection_layout":"fused_qkv","attention_value_scale":0.707,"eos_token_id":151645,"head_dim":192,
    \\ "hidden_size":4096,"hybrid_layer_pattern":[0,1,1,1,1,0],"intermediate_size":16384,"layernorm_epsilon":1e-06,
    \\ "max_position_embeddings":1048576,"moe_intermediate_size":2048,"moe_layer_freq":[0,1,1,1,1,1],"n_group":1,
    \\ "n_routed_experts":256,"n_shared_experts":null,"norm_topk_prob":true,"num_attention_heads":64,
    \\ "num_experts_per_tok":8,"num_hidden_layers":6,"num_key_value_heads":4,"num_nextn_predict_layers":3,
    \\ "partial_rotary_factor":0.334,
    \\ "rope_parameters":{"partial_rotary_factor":0.334,"rope_theta":10000000.0,"rope_type":"default"},
    \\ "rope_theta":10000000.0,"routed_scaling_factor":null,"scoring_func":"sigmoid","sliding_window":128,
    \\ "swa_head_dim":192,"swa_num_attention_heads":64,"swa_num_key_value_heads":8,"swa_rope_theta":10000.0,
    \\ "swa_v_head_dim":128,"tie_word_embeddings":false,"topk_group":1,"topk_method":"noaux_tc","v_head_dim":128,
    \\ "vocab_size":152576,"quantization":{"group_size":32,"bits":4,"mode":"mxfp4"},
    \\ "quantization_config":{"group_size":32,"bits":4,"mode":"mxfp4"}}
;

test "mimo_v2 config: the pack parses its per-layer geometry, dense prefix and ring" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(pack_json, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expect(c.isGlobal(0) and !c.isGlobal(1) and c.isGlobal(5));
    try testing.expectEqual(@as(u32, 4), c.kvHeads(0));
    try testing.expectEqual(@as(u32, 8), c.kvHeads(1));
    try testing.expectEqual(@as(u32, 64), c.ropeDims());
    try testing.expectEqual(@as(u32, 1), c.first_k_dense);
    try testing.expectEqual(@as(u32, 128 + ring_slack), p.cache.sliding_ring);
    var lay: [8]abi.LayerCache = undefined;
    try testing.expectEqual(@as(u32, 6), layout(&p, null, &lay, 8));
    try testing.expect(lay[1].kind == .sliding and lay[1].window == 128 and lay[1].k_dim == 192 and lay[1].v_dim == 128);
}

test "mimo_v2 config: the original release fails by name (a Sushi pack's fp8 trunk does not)" {
    var p: abi.Parsed = .{};
    const release = "{\"model_type\":\"mimo_v2\",\"quantization_config\":{\"quant_method\":\"fp8\"}}";
    const v = parseJson(release, &p);
    try testing.expectEqual(abi.Verdict.Code.failed, v.code);
    try testing.expect(std.mem.indexOf(u8, v.why.slice(), "convert_mimo_v2.py") != null);
    const sushi = parseJson("{\"model_type\":\"mimo_v2\",\"quantization_config\":{\"quant_method\":\"fp8\"},\"expert_quant\":{}}", &p);
    try testing.expect(std.mem.indexOf(u8, sushi.why.slice(), "convert_mimo_v2.py") == null);
}

test "mimo_v2 config: a streaming pack with the MXFP4 bank and fp8 residents parses; another bank or residents format is unsupported" {
    const bank = "\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-mxfp4-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"}";
    const fp8 = "\"model.layers.0.self_attn.q_proj\":{\"mode\":\"fp8_e4m3\",\"bits\":8,\"block\":[64,128]},\"lm_head\":false,";
    const stream_json = try std.mem.concat(testing.allocator, u8, &.{ pack_json[0 .. pack_json.len - 1], ",", bank, ",\"residents_format\":\"fp8_block128\"}" });
    defer testing.allocator.free(stream_json);
    const with_fp8 = try std.mem.replaceOwned(u8, testing.allocator, stream_json, "\"quantization\":{", "\"quantization\":{" ++ fp8);
    defer testing.allocator.free(with_fp8);
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(with_fp8, &p).isOk());
    try testing.expect(of(&p).streamed and of(&p).quant.mode == .mxfp4);
    freeParsed(&p);
    for ([_][]const u8{ "\"expert_streaming\":{}", "\"expert_streaming\":{\"family\":\"mlx-stream\",\"format\":\"mlx-stream-exl3-expert-bank-v1\",\"manifest\":\"expert-manifest.json\"}" }) |other| {
        const js = try std.mem.replaceOwned(u8, testing.allocator, stream_json, bank, other);
        defer testing.allocator.free(js);
        try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(js, &p).code);
    }
    const res = try std.mem.replaceOwned(u8, testing.allocator, stream_json, "fp8_block128", "int8");
    defer testing.allocator.free(res);
    try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(res, &p).code);
}
