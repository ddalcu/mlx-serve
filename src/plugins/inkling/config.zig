//! Inkling's claim and parse: `text_config` as the engine reads it (the checkpoint labels the MoE width
//! `intermediate_size` and the dense width `dense_intermediate_size`; `local_layer_ids` names the sliding layers), plus
//! the per-layer cache layout and one recurrent entry per layer (its four short-conv windows). MLX-free. A config the
//! forward does not model (other sliding geometry or gate, convs or the shared-expert sink turned off) is refused
//! `unsupported`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const llama = @import("../llama/config.zig");

const field = llama.field;
const int = llama.int;
const float = llama.float;

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "inkling",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{},
};

pub const model_types = [_][]const u8{"inkling_mm_model"};
pub const claims = sdk.claimsByType(&model_types);

pub const max_layers = 128;

pub const Config = struct {
    vocab_size: u32,
    /// The logits the model emits: the padded vocab's real rows (0 = all).
    unpadded_vocab_size: u32 = 0,
    hidden_size: u32,
    /// The dense bottom layers' width.
    intermediate_size: u32 = 0,
    moe_intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32 = 1e-6,
    /// Layers below this one are dense (`dense_mlp_idx`).
    first_moe: u32 = 0,
    num_experts: u32,
    top_k: u32,
    n_shared: u32,
    route_scale: f32 = 1.0,
    d_rel: u32,
    log_n_floor: u32 = 0,
    log_alpha: f32 = 0.1,
    sconv_kernel: u32 = 4,
    embed_norm: bool = true,
    sliding_window: u32 = 0,
    global: [max_layers]bool = @splat(false),
    mup: f32 = 1.0,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",

    /// Columns of a layer's conv state before the MLP's window: k, v, then the attention output.
    pub fn mlpStateOffset(c: *const Config) u32 {
        return 2 * c.num_key_value_heads * c.head_dim + c.hidden_size;
    }
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "inkling_mm_model: " ++ why);
}

/// Root keys of shapes this plugin does not serve.
const foreign_keys = [_][]const u8{"expert_quant"};

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "inkling_mm_model: config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, "inkling_mm_model: config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return unsupported("no model_type");
    if (mt != .string or !std.mem.eql(u8, mt.string, model_types[0])) return unsupported("another model_type");
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
    for (foreign_keys) |k| if (field(root, k) != null) return unsupported("a key this plugin does not model");
    const cfg = if (field(root, "text_config")) |tc| (if (tc == .object) tc.object else return unsupported("text_config")) else root;

    var c: Config = .{
        .vocab_size = if (field(cfg, "vocab_size")) |v| int(v) orelse return unsupported("vocab_size") else 0,
        .hidden_size = int(field(cfg, "hidden_size")) orelse return unsupported("hidden_size"),
        .moe_intermediate_size = int(field(cfg, "intermediate_size")) orelse 0,
        .num_hidden_layers = int(field(cfg, "num_hidden_layers")) orelse return unsupported("num_hidden_layers"),
        .num_attention_heads = int(field(cfg, "num_attention_heads")) orelse return unsupported("num_attention_heads"),
        .num_key_value_heads = int(field(cfg, "num_key_value_heads")) orelse return unsupported("num_key_value_heads"),
        .head_dim = int(field(cfg, "head_dim")) orelse return unsupported("head_dim"),
        .num_experts = int(field(cfg, "n_routed_experts")) orelse return unsupported("n_routed_experts"),
        .top_k = int(field(cfg, "num_experts_per_tok")) orelse return unsupported("num_experts_per_tok"),
        .n_shared = int(field(cfg, "n_shared_experts")) orelse return unsupported("n_shared_experts"),
        .d_rel = int(field(cfg, "d_rel")) orelse return unsupported("d_rel"),
        .sliding_window = int(field(cfg, "sliding_window_size")) orelse return unsupported("sliding_window_size"),
    };
    c.intermediate_size = int(field(cfg, "dense_intermediate_size")) orelse c.moe_intermediate_size;
    if (field(cfg, "dense_mlp_idx")) |v| c.first_moe = int(v) orelse return unsupported("dense_mlp_idx");
    if (field(cfg, "unpadded_vocab_size")) |v| c.unpadded_vocab_size = int(v) orelse return unsupported("unpadded_vocab_size");
    if (field(cfg, "log_scaling_n_floor")) |v| c.log_n_floor = int(v) orelse return unsupported("log_scaling_n_floor");
    if (field(cfg, "sconv_kernel_size")) |v| c.sconv_kernel = int(v) orelse return unsupported("sconv_kernel_size");
    inline for (.{ .{ "rms_norm_eps", &c.rms_norm_eps }, .{ "route_scale", &c.route_scale }, .{ "log_scaling_alpha", &c.log_alpha }, .{ "logits_mup_width_multiplier", &c.mup } }) |f| {
        if (field(cfg, f[0])) |v| f[1].* = @floatCast(float(v) orelse return unsupported(f[0]));
    }
    if (field(cfg, "use_embed_norm")) |v| if (v == .bool) {
        c.embed_norm = v.bool;
    };
    // The forward runs the convs and the shared-expert sink unconditionally, the gate as a sigmoid.
    if (field(cfg, "use_sconv")) |v| if (v != .bool or !v.bool) return unsupported("use_sconv off");
    if (field(cfg, "shared_expert_sink")) |v| if (v != .bool or !v.bool) return unsupported("shared_expert_sink off");
    if (field(cfg, "gate_activation")) |v| if (v != .string or !std.mem.eql(u8, v.string, "sigmoid")) return unsupported("gate_activation other than sigmoid");
    inline for (.{ .{ "swa_num_attention_heads", c.num_attention_heads }, .{ "swa_num_key_value_heads", c.num_key_value_heads }, .{ "swa_head_dim", c.head_dim } }) |f| {
        if (field(cfg, f[0])) |v| if ((int(v) orelse return unsupported(f[0])) != f[1]) return unsupported("sliding-attention geometry other than the global one");
    }
    if (c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
        c.num_hidden_layers == 0 or c.num_hidden_layers > max_layers or c.num_attention_heads == 0 or
        c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or c.head_dim == 0 or
        c.head_dim > 512 or c.num_experts == 0 or c.num_experts > 4096 or c.top_k == 0 or c.top_k > c.num_experts or
        c.n_shared == 0 or c.d_rel == 0 or
        c.sconv_kernel < 2 or c.sconv_kernel > 64 or c.sliding_window == 0 or (c.vocab_size > 0 and c.unpadded_vocab_size > c.vocab_size) or
        c.mup <= 0)
        return unsupported("dimensions out of range");

    const local = field(cfg, "local_layer_ids") orelse return unsupported("no local_layer_ids");
    if (local != .array) return unsupported("local_layer_ids");
    @memset(c.global[0..c.num_hidden_layers], true);
    for (local.array.items) |item| {
        const i = int(item) orelse return unsupported("local_layer_ids");
        if (i < c.num_hidden_layers) c.global[i] = false;
    }

    var p: abi.Parsed = .{};
    const s = &p.shell;
    s.* = .{
        .vocab_size = if (c.unpadded_vocab_size > 0) c.unpadded_vocab_size else c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .num_experts_per_tok = c.top_k,
        .moe_intermediate_size = c.moe_intermediate_size,
        .act_dtype = .bf16,
        .model_type = abi.Str.of(model_types[0]),
    };
    llama.readShell(cfg, root, s) catch return unsupported("a token id or the context out of range");
    if (int(field(cfg, "model_max_length"))) |n| {
        s.max_position_embeddings = n;
        s.context_cap = n;
    }
    c.quant = llama.readQuant(root) catch return unsupported("the quantization block");
    s.quant = c.quant;
    p.cache.ssm_state = .{ .entries = c.num_hidden_layers, .dtype = .f32 };
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "inkling_mm_model: out of memory");
    cfg_out.* = c;
    cfg_out.block_json = llama.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, "inkling_mm_model: out of memory");
    };
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

/// One entry per layer: a global layer keeps full KV, a local one its window; no layer ropes.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = @min(c.num_hidden_layers, cap);
    for (out[0..n], c.global[0..n]) |*o, g| o.* = .{
        .kind = if (g) .full else .sliding,
        .nope = true,
        .window = if (g) 0 else c.sliding_window,
        .kv_heads = c.num_key_value_heads,
        .k_dim = c.head_dim,
        .v_dim = c.head_dim,
    };
    return n;
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

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

/// pipenetwork/Inkling-Small-MLX-REAP25-4bit's config.json, trimmed to the keys the engine reads.
const reap25 =
    \\{"model_type":"inkling_mm_model","eos_token_id":200006,"text_config":{"model_max_length":1048576,
    \\"hidden_size":4096,"num_hidden_layers":42,"vocab_size":201024,"num_attention_heads":32,"num_key_value_heads":8,
    \\"head_dim":128,"d_rel":16,"rel_extent":1024,"log_scaling_n_floor":128000,"log_scaling_alpha":0.1,
    \\"rms_norm_eps":1e-06,"use_embed_norm":true,"local_layer_ids":[0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,
    \\22,24,25,26,27,28,30,31,32,33,34,36,37,38,39,40],"dense_mlp_idx":2,"use_sconv":true,"sconv_kernel_size":4,
    \\"unpadded_vocab_size":200058,"logits_mup_width_multiplier":16.0,"swa_head_dim":128,"swa_num_attention_heads":32,
    \\"swa_num_key_value_heads":8,"sliding_window_size":512,"n_routed_experts":192,"num_experts_per_tok":6,
    \\"n_shared_experts":2,"shared_expert_sink":true,"dense_intermediate_size":16384,"intermediate_size":2048,
    \\"route_scale":8.0,"use_gate_bias":true,"gate_activation":"sigmoid"},
    \\"vision_config":{"vision_encoder_type":"hmlp"},"mtp_config":{"num_nextn_predict_layers":8},
    \\"quantization":{"group_size":64,"bits":4}}
;

test "inkling config: the REAP25 config's widths, global layers, padded vocab and recurrent entries" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(reap25, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 16384), c.intermediate_size);
    try testing.expectEqual(@as(u32, 2048), c.moe_intermediate_size);
    try testing.expectEqual(@as(u32, 2), c.first_moe);
    try testing.expect(c.global[5] and c.global[41] and !c.global[0] and !c.global[40]);
    try testing.expectEqual(@as(f32, 16.0), c.mup);
    try testing.expectEqual(@as(u32, 200058), p.shell.vocab_size);
    try testing.expectEqual(@as(u64, 1048576), p.shell.context_cap);
    try testing.expectEqual(@as(u32, 42), p.cache.ssm_state.entries);
    try testing.expectEqual(@as(u32, 2 * 8 * 128 + 4096), c.mlpStateOffset());
    var lay: [42]abi.LayerCache = undefined;
    try testing.expectEqual(@as(u32, 42), layout(&p, null, &lay, lay.len));
    try testing.expect(lay[5].kind == .full and lay[0].kind == .sliding and lay[0].window == 512 and lay[0].nope);
}

test "inkling config: a sliding geometry, gate or sink the forward does not model is refused unsupported" {
    inline for (.{ .{ "\"swa_head_dim\":128", "\"swa_head_dim\":64" }, .{ "\"gate_activation\":\"sigmoid\"", "\"gate_activation\":\"softmax\"" }, .{ "\"shared_expert_sink\":true", "\"shared_expert_sink\":false" }, .{ "\"use_sconv\":true", "\"use_sconv\":false" } }) |swap| {
        const json = comptime blk: {
            @setEvalBranchQuota(100_000);
            const at = std.mem.indexOf(u8, reap25, swap[0]).?;
            break :blk reap25[0..at] ++ swap[1] ++ reap25[at + swap[0].len ..];
        };
        var p: abi.Parsed = .{};
        try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(json, &p).code);
    }
}
