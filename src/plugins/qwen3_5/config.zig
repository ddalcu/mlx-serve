//! Qwen3.5 / 3.6 / 3.8 packs (mlx-lm's `qwen3_5` and `qwen3_5_moe` layouts, their `_text` siblings) and MiniCPM-V 4.7's
//! Qwen3.5 trunk: the claim, and the parse into `Parsed` plus the trunk's own fields. MLX-free. GatedDeltaNet layers
//! with gated full attention every `full_attention_interval`; dense or routed MLPs. Prism's Hadamard packs
//! (`prism_hadamard_qwen35`) parse here too. A pack this plugin does not serve (a foreign quant block) is refused
//! `unsupported`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const q3 = @import("../qwen3/config.zig");
const mrope = @import("../../sdk/vision/mrope.zig");
const rope_scaling = @import("../../sdk/config/rope_scaling.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "qwen3_5",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{ "mlx-community/Qwen3.5-4B-4bit", "mlx-community/Qwen3.6-35B-A3B-4bit" },
};

pub const model_types = [_][]const u8{ "qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "minicpmv4_7" };

pub const claims = sdk.claimsByType(&model_types);

/// The layouts `parseAs` reads: qwen3_5's, Qwen3-Next's, Prism's (a dense qwen3_5 trunk behind Hadamard rotations).
pub const Variant = enum { qwen3_5, next, prism };

const field = q3.field;
const int = q3.int;
const float = q3.float;
const gpa = q3.gpa;
const P = "qwen3_5";

fn unsupported(comptime why: []const u8) abi.Verdict {
    return q3.unsupported(P, why);
}

/// The fields the forward and the load read.
pub const Config = struct {
    /// Qwen3-Next (`qwen3_next`): flat `model.*` weights, q|k|v|z and b|a projections combined per key head, its own
    /// pipelined-prefill table, serial decode.
    next: bool = false,
    /// A Prism pack's Hadamard block (`modules[].block`): f16 activations, an f32 GDN state; 0 = not rotated.
    hadamard_block: u32 = 0,
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32 = 0,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    partial_rotary_factor: f32 = 1.0,
    /// The `rope_scaling` / `rope_parameters` block (mlx-lm's `initialize_rope` over the rotated dims).
    rope: rope_scaling.Scaling = .none,
    max_position_embeddings: u32 = 0,
    query_pre_attn_scalar: u32,
    tie_word_embeddings: bool = false,
    full_attention_interval: u32,
    linear_num_key_heads: u32,
    linear_num_value_heads: u32,
    linear_key_head_dim: u32,
    linear_value_head_dim: u32,
    linear_conv_kernel_dim: u32,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    shared_expert_intermediate_size: u32 = 0,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",
    /// The M-RoPE spectrum (`Parsed.mrope.inv_freq`) of a pack with an image tower; empty without one.
    inv_freq: []f64 = &.{},

    pub fn moe(c: *const Config) bool {
        return c.num_experts > 0;
    }

    /// Layer `i` is a GatedDeltaNet layer (every `full_attention_interval`-th is full attention).
    pub fn linear(c: *const Config, i: u32) bool {
        return (i + 1) % c.full_attention_interval != 0;
    }

    /// The rotated width of a full-attention head.
    pub fn ropeDims(c: *const Config) u32 {
        return @intFromFloat(@as(f32, @floatFromInt(c.head_dim)) * c.partial_rotary_factor);
    }
};

/// The M-RoPE table's inverse frequencies under the pack's scaling (a dynamic NTK base stays the static one there).
fn mropeSpectrum(out: []f64, rd: u32, c: *const Config) void {
    if (rope_scaling.freqs(c.rope, out, rd, c.rope_theta)) {
        for (out) |*v| v.* = 1.0 / v.*;
        return;
    }
    mrope.computeInvFreq(out, rd, c.rope_theta);
    for (out) |*v| v.* *= rope_scaling.positionScale(c.rope);
}

/// Keys whose presence means a shape this plugin does not serve bit-equal yet.
const foreign_keys = [_][]const u8{
    "pooling_mode",                "expert_quant",    "use_bidirectional_attention", "modules",
    "hidden_size_per_layer_input", "qk_scale_factor", "linear_attn_tail_from",       "kda_lower_bound",
    "hc_count",
};

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    return parseAs(.qwen3_5, dir, config_json, span, stub, out);
}

/// The parse of a `variant` layout.
pub fn parseAs(comptime variant: Variant, dir: *const abi.PackDir, config_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) abi.Verdict {
    const next = variant == .next;
    const prism = variant == .prism;
    const types: []const []const u8 = switch (variant) {
        .qwen3_5 => &model_types,
        .next => &.{"qwen3_next"},
        .prism => &.{"prism_hadamard_qwen35"},
    };
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const d = q3.doc(arena.allocator(), config_json.slice()) catch |e| return switch (e) {
        error.Broken => abi.Verdict.refuse(.failed, "qwen3_5: config.json does not parse"),
        error.NotAnObject => abi.Verdict.refuse(.failed, "qwen3_5: config.json is not an object"),
        else => unsupported("no model_type or text_config"),
    };
    for (types) |t| {
        if (std.mem.eql(u8, t, d.model_type)) break;
    } else return unsupported("another model_type");
    const root = d.root;
    const cfg = d.cfg;
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
    for (foreign_keys) |k| {
        if (prism and std.mem.eql(u8, k, "modules")) continue;
        if (field(cfg, k) != null or field(root, k) != null) return unsupported("a key this plugin does not model");
    }
    if (dir.has("1_Pooling/config.json")) return unsupported("an embedding pack (1_Pooling)");
    if (field(cfg, "hidden_act")) |act| if (act != .string or !std.mem.eql(u8, act.string, "silu")) return unsupported("hidden_act other than silu");

    var c: Config = .{
        .next = next,
        .vocab_size = int(field(cfg, "vocab_size")) orelse return unsupported("vocab_size"),
        .hidden_size = int(field(cfg, "hidden_size")) orelse return unsupported("hidden_size"),
        .num_hidden_layers = int(field(cfg, "num_hidden_layers")) orelse return unsupported("num_hidden_layers"),
        .num_attention_heads = int(field(cfg, "num_attention_heads")) orelse return unsupported("num_attention_heads"),
        .num_key_value_heads = int(field(cfg, "num_key_value_heads")) orelse return unsupported("num_key_value_heads"),
        .head_dim = int(field(cfg, "head_dim")) orelse return unsupported("head_dim"),
        .rms_norm_eps = @floatCast(float(field(cfg, "rms_norm_eps")) orelse return unsupported("rms_norm_eps")),
        .rope_theta = @floatCast(q3.ropeTheta(cfg) orelse return unsupported("no rope_theta")),
        .query_pre_attn_scalar = 0,
        .full_attention_interval = int(field(cfg, "full_attention_interval")) orelse return unsupported("full_attention_interval"),
        .linear_num_key_heads = int(field(cfg, "linear_num_key_heads")) orelse return unsupported("linear_num_key_heads"),
        .linear_num_value_heads = int(field(cfg, "linear_num_value_heads")) orelse return unsupported("linear_num_value_heads"),
        .linear_key_head_dim = int(field(cfg, "linear_key_head_dim")) orelse return unsupported("linear_key_head_dim"),
        .linear_value_head_dim = int(field(cfg, "linear_value_head_dim")) orelse return unsupported("linear_value_head_dim"),
        .linear_conv_kernel_dim = int(field(cfg, "linear_conv_kernel_dim")) orelse return unsupported("linear_conv_kernel_dim"),
    };
    if (c.vocab_size == 0 or c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
        c.num_hidden_layers == 0 or c.num_hidden_layers > 1024 or c.num_attention_heads == 0 or
        c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or
        c.head_dim == 0 or c.head_dim % 2 != 0 or c.head_dim > 512 or c.full_attention_interval == 0 or
        c.linear_num_key_heads == 0 or c.linear_num_value_heads % c.linear_num_key_heads != 0 or
        c.linear_key_head_dim == 0 or c.linear_value_head_dim == 0 or c.linear_conv_kernel_dim < 2)
        return unsupported("dimensions out of range");
    c.query_pre_attn_scalar = if (field(cfg, "query_pre_attn_scalar")) |v| int(v) orelse return unsupported("query_pre_attn_scalar") else c.head_dim;
    if (c.query_pre_attn_scalar == 0) return unsupported("query_pre_attn_scalar");
    c.partial_rotary_factor = @floatCast(q3.partialRotary(cfg, next) catch return unsupported("partial_rotary_factor"));
    if (c.partial_rotary_factor <= 0 or c.partial_rotary_factor > 1) return unsupported("partial_rotary_factor");
    c.rope = rope_scaling.parse(cfg);
    c.max_position_embeddings = int(field(cfg, "max_position_embeddings")) orelse int(field(root, "max_position_embeddings")) orelse 0;
    if (c.rope == .dynamic and c.max_position_embeddings == 0) c.rope = .unmodeled;
    if (q3.boolean(field(cfg, "attn_output_gate"))) |g| if (!g) return unsupported("no attention output gate");
    c.tie_word_embeddings = q3.tied(root, cfg);
    if (int(field(cfg, "num_experts"))) |e| if (e > 0) {
        c.num_experts = e;
        c.num_experts_per_tok = int(field(cfg, "num_experts_per_tok")) orelse return unsupported("num_experts_per_tok");
        c.moe_intermediate_size = int(field(cfg, "moe_intermediate_size")) orelse return unsupported("moe_intermediate_size");
        c.shared_expert_intermediate_size = int(field(cfg, "shared_expert_intermediate_size")) orelse 0;
        if (e < 2 or e > 1024 or c.num_experts_per_tok == 0 or c.num_experts_per_tok > e or c.moe_intermediate_size == 0)
            return unsupported("expert dimensions out of range");
        if (field(cfg, "mlp_only_layers")) |v| if (v != .array or v.array.items.len != 0) return unsupported("mlp_only_layers");
    };
    c.intermediate_size = int(field(cfg, "intermediate_size")) orelse 0;
    if (!c.moe() and c.intermediate_size == 0) return unsupported("intermediate_size");
    c.quant = q3.quantizationOf(root, prism) catch return unsupported("quantization");
    if (prism) {
        c.hadamard_block = hadamardBlock(root) orelse return unsupported("modules without one shared rotation block");
        // The format refuses rotated expert banks.
        if (c.moe()) return unsupported("a routed-expert Hadamard pack");
    }

    // Qwen3-Next decodes serial: the engine batches only the qwen3_5 MoE trunk's GDN state. An affine dense trunk's
    // verify rows are row-exact (the `.mma` lane, set at load where every projection fits); a bf16 matmul's bits
    // follow its row count. A dense trunk verifies draft trees (the engine's row-exact reach, bf16 included). The ANE
    // prefill programs read unrotated separate GDN projections.
    const dense = !next and !prism and !c.moe();
    const exact = dense and c.quant.bits > 0 and c.quant.mode == .affine;
    var p: abi.Parsed = .{ .caps = .{ .batches_decode = !next and c.rope != .dynamic, .moe = c.moe(), .exact_resumed = true, .dflash_capture = true, .exact_drafted = exact, .spec_tree = dense and c.rope != .dynamic, .ane_prefill = !next and !prism, .merged_state = true } };
    if (rope_scaling.layoutNamespace(c.rope)) |ns| p.cache.layout_namespace = abi.Str.of(ns);
    // The grafted head over-speculates in auto mode (`ModelConfig.mtpDepth`).
    if (prism) p.spec.mtp_default_depth = 2;
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .num_experts_per_tok = c.num_experts_per_tok,
        .moe_intermediate_size = c.moe_intermediate_size,
        .act_dtype = if (prism) .f16 else .bf16,
        .state_dtype = if (prism) .f32 else .bf16,
        .quant = c.quant,
        // The engine serves every Qwen3.5 layout under one name.
        .model_type = abi.Str.of(if (next) "qwen3_next" else "qwen3_5_moe"),
    };
    q3.shellCommon(root, cfg, s) catch return unsupported("bos or eos ids");
    p.cache.ssm_state = .{ .entries = c.num_hidden_layers, .dtype = if (prism) .f32 else .bf16, .checkpoint_bytes = checkpointBytes(&c) };
    p.chat.tool_dialect = .qwen_xml;
    p.chat.sampling_defaults = .{ .top_k = 20, .top_p = 0.95 };
    if (!stub) generation.apply(dir, s);
    // Thinking stays off unless the pack's generation_config asks (the host reads that): the family documents no default.
    p.chat.thinking_default = .off;
    p.chat.reasoning_dialect = .think_tags;

    // An image tower's rows splice at the placeholders, positioned on the prompt's M-RoPE table (the engine runs the
    // tower and builds the table; the interleaved t/h/w split is the only one the engine serves).
    var sections: [3]u32 = .{ 0, 0, 0 };
    const vision = field(root, "vision_config") != null;
    if (vision) {
        if (field(cfg, "rope_parameters")) |rp| if (rp == .object) if (field(rp.object, "mrope_section")) |v| if (v == .array) {
            for (v.array.items, 0..) |item, i| {
                if (i >= 3) break;
                sections[i] = int(item) orelse return unsupported("mrope_section");
            }
        };
        if (sections[0] + sections[1] + sections[2] == 0) return unsupported("an image tower without M-RoPE sections");
        const minicpm = std.mem.eql(u8, d.model_type, "minicpmv4_7");
        p.placeholder_ids = .{
            .image = int(field(root, "image_token_id")) orelse int(field(root, "image_token_index")) orelse return unsupported("image_token_id"),
            // MiniCPM-V serves no video frames.
            .video = if (minicpm) 0 else int(field(root, "video_token_id")) orelse 0,
            .vision_start = int(field(root, "vision_start_token_id")) orelse 0,
            .vision_end = int(field(root, "vision_end_token_id")) orelse 0,
        };
    }

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "qwen3_5: out of memory");
    cfg_out.* = c;
    cfg_out.block_json = q3.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, "qwen3_5: out of memory");
    };
    if (vision) {
        const rd = c.ropeDims();
        cfg_out.inv_freq = gpa.alloc(f64, rd / 2) catch {
            gpa.free(cfg_out.block_json);
            gpa.destroy(cfg_out);
            return abi.Verdict.refuse(.failed, "qwen3_5: out of memory");
        };
        mropeSpectrum(cfg_out.inv_freq, rd, &c);
        p.mrope = .{ .sections = sections, .rope_dims = rd, .inv_freq = .of(cfg_out.inv_freq), .mscale = rope_scaling.mscale(c.rope) };
    }
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

/// The rotation block every listed module shares (Prism's `modules[].block`), as the engine's parse reads it; null
/// when there is no list or the modules disagree.
fn hadamardBlock(root: std.json.ObjectMap) ?u32 {
    const mods = field(root, "modules") orelse return null;
    if (mods != .array) return null;
    var blk: u32 = 0;
    for (mods.array.items) |m| {
        if (m != .object) return null;
        const v = int(field(m.object, "block")) orelse return null;
        if (v == 0 or (blk != 0 and v != blk)) return null;
        blk = v;
    }
    return if (blk == 0) null else blk;
}

/// One snapshot of every linear layer's recurrent state and conv window (`ModelConfig.ssmCheckpointBytes`).
fn checkpointBytes(c: *const Config) u64 {
    var linear: u64 = 0;
    for (0..c.num_hidden_layers) |i| {
        if (c.linear(@intCast(i))) linear += 1;
    }
    const elem: u64 = if (c.hadamard_block > 0) 4 else 2;
    const state: u64 = @as(u64, c.linear_num_value_heads) * c.linear_value_head_dim * c.linear_key_head_dim * elem;
    const conv_dim: u64 = 2 * @as(u64, c.linear_num_key_heads) * c.linear_key_head_dim + @as(u64, c.linear_num_value_heads) * c.linear_value_head_dim;
    return linear * (state + (c.linear_conv_kernel_dim - 1) * conv_dim * 2);
}

/// Linear layers keep a recurrent state, every `full_attention_interval`-th layer full attention KV.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = @min(c.num_hidden_layers, cap);
    for (out[0..n], 0..) |*l, i| l.* = if (c.linear(@intCast(i)))
        .{ .kind = .linear }
    else
        .{ .kind = .full, .kv_heads = c.num_key_value_heads, .k_dim = c.head_dim, .v_dim = c.head_dim };
    return n;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| {
        const c: *Config = @ptrCast(@alignCast(ptr));
        gpa.free(c.block_json);
        gpa.free(c.inv_freq);
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

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

const qwen36_35b =
    \\{"model_type":"qwen3_5_moe","tie_word_embeddings":false,"text_config":{"model_type":"qwen3_5_moe_text",
    \\ "hidden_size":2048,"num_hidden_layers":40,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,
    \\ "vocab_size":248320,"rms_norm_eps":1e-6,"full_attention_interval":4,"linear_num_key_heads":16,
    \\ "linear_num_value_heads":32,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,
    \\ "max_position_embeddings":262144,"hidden_act":"silu","attn_output_gate":true,"eos_token_id":248044,
    \\ "rope_parameters":{"rope_type":"default","rope_theta":10000000,"partial_rotary_factor":0.25,
    \\   "mrope_section":[11,11,10],"mrope_interleaved":true}},
    \\ "eos_token_id":248046,"quantization":{"group_size":64,"bits":4}}
;

test "qwen3_5 config: a Qwen3.6-35B-A3B text pack parses into the shell, its layout and the GDN state facts" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(qwen36_35b, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(p.caps.batches_decode and p.caps.moe and p.caps.exact_resumed);
    try testing.expectEqualStrings("qwen3_5_moe", p.shell.model_type.slice());
    try testing.expectEqual(@as(u32, 64), of(&p).ropeDims());
    var lay: [41]abi.LayerCache = undefined;
    try testing.expectEqual(@as(u32, 40), layout(&p, null, &lay, 41));
    try testing.expect(lay[0].kind == .linear and lay[3].kind == .full and lay[3].k_dim == 256);
    // 30 linear layers x (32 x 128 x 128 bf16 state + 3 x 8192 bf16 conv rows).
    try testing.expectEqual(@as(u64, 30 * (32 * 128 * 128 * 2 + 3 * 8192 * 2)), p.cache.ssm_state.checkpoint_bytes);
    try testing.expectEqual(abi.ToolDialect.qwen_xml, p.chat.tool_dialect);
}

test "qwen3_5 config: a --mlx-gguf file's rebuilt config parses with its gguf quant mode" {
    const gguf_json = try std.mem.replaceOwned(u8, testing.allocator, qwen36_35b, "\"quantization\":{\"group_size\":64,\"bits\":4}", "\"quantization\":{\"group_size\":32,\"bits\":4,\"mode\":\"gguf\"}");
    defer testing.allocator.free(gguf_json);
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(gguf_json, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(abi.QuantMode.gguf, p.shell.quant.mode);
}

test "qwen3_5 config: a Qwen3-VL pack and MiniCPM-V 4.7 declare their placeholders and the M-RoPE spectrum" {
    const vl = try std.mem.replaceOwned(u8, testing.allocator, qwen36_35b, "\"eos_token_id\":248046,", "\"eos_token_id\":248046,\"vision_config\":{\"depth\":27},\"image_token_id\":248056,\"video_token_id\":248057,");
    defer testing.allocator.free(vl);
    for ([_][]const u8{ vl, @embedFile("../../fixtures/model-configs/minicpmv4_7.json") }, [_]u32{ 248057, 0 }) |json, video| {
        var p: abi.Parsed = .{};
        try testing.expect(parseJson(json, &p).isOk());
        defer freeParsed(&p);
        try testing.expectEqual(@as(u32, 248056), p.placeholder_ids.image);
        try testing.expectEqual(video, p.placeholder_ids.video);
        try testing.expectEqual([3]u32{ 11, 11, 10 }, p.mrope.sections);
        try testing.expectEqual(@as(u32, 64), p.mrope.rope_dims);
        var want: [32]f64 = undefined;
        mrope.computeInvFreq(&want, 64, @as(f32, 10000000));
        try testing.expectEqualSlices(f64, &want, p.mrope.inv_freq.slice());
        try testing.expect(p.chat.thinking_default == .off and p.chat.reasoning_dialect == .think_tags);
    }
}

test "qwen3_5 config: shapes this plugin does not serve yet are refused unsupported" {
    const vl = try std.mem.replaceOwned(u8, testing.allocator, qwen36_35b, "\"mrope_section\":[11,11,10],", "");
    defer testing.allocator.free(vl);
    const vl2 = try std.mem.replaceOwned(u8, testing.allocator, vl, "\"eos_token_id\":248046,", "\"eos_token_id\":248046,\"vision_config\":{\"depth\":27},\"image_token_id\":5,");
    defer testing.allocator.free(vl2);
    for ([_][]const u8{ vl2, "{\"model_type\":\"qwen3_next\"}" }) |json| {
        var p: abi.Parsed = .{};
        try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(json, &p).code);
    }
}

test "qwen3_5 config: a YaRN rope_parameters block is served over the rotated dims, its M-RoPE table scaled too" {
    const yarn = try std.mem.replaceOwned(u8, testing.allocator, qwen36_35b, "\"rope_type\":\"default\"", "\"rope_type\":\"yarn\",\"factor\":4,\"original_max_position_embeddings\":262144");
    defer testing.allocator.free(yarn);
    const vl = try std.mem.replaceOwned(u8, testing.allocator, yarn, "\"eos_token_id\":248046,", "\"eos_token_id\":248046,\"vision_config\":{\"depth\":27},\"image_token_id\":248056,");
    defer testing.allocator.free(vl);
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(vl, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(of(&p).rope == .yarn);
    try testing.expectEqualStrings("scaled-rope-v1", p.cache.layout_namespace.slice());
    var want: [32]f64 = undefined;
    try testing.expect(rope_scaling.freqs(of(&p).rope, &want, 64, 10000000));
    for (&want) |*v| v.* = 1.0 / v.*;
    try testing.expectEqualSlices(f64, &want, p.mrope.inv_freq.slice());
    try testing.expectApproxEqRel(0.1 * @log(4.0) + 1.0, p.mrope.mscale, 1e-12);
}

test "qwen3_5 config: a routed-expert pack carrying an MTP head is served (its head runs through SpecHooks.head)" {
    const mem: fixture.MemDir = .{ .files = &.{
        .{ .name = "config.json", .bytes = qwen36_35b },
        .{ .name = "model.safetensors.index.json", .bytes = "{\"weight_map\":{\"language_model.mtp.fc.weight\":\"model-00002.safetensors\"}}" },
    } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(qwen36_35b), .{}, null, true, &p).isOk());
    freeParsed(&p);
}
