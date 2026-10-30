//! Qwen3 packs (mlx-lm's `qwen3` and `qwen3_moe` layouts): the claim, and the parse into `Parsed` plus this plugin's
//! own fields. MLX-free. The JSON readers here are the Qwen family's (qwen3_5 and qwen3_asr parse through them). A
//! config this plugin does not model the way the engine always served it is refused `unsupported`, so the legacy path
//! keeps it.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const rope_scaling = @import("../../sdk/config/rope_scaling.zig");
const json_read = @import("../../sdk/config/json.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "qwen3",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{ "mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3-30B-A3B-4bit" },
};

pub const model_types = [_][]const u8{ "qwen3", "qwen3_moe", "qwen3_moe_text" };

pub const claims = sdk.claimsByType(&model_types);

/// The fields the forward and the load read.
pub const Config = struct {
    moe: bool = false,
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32 = 0,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    /// `1 / sqrt(query_pre_attn_scalar)` is the attention scale; the head width when the config omits it.
    query_pre_attn_scalar: u32,
    /// The `rope_scaling` block (mlx-lm's `initialize_rope`).
    rope: rope_scaling.Scaling = .none,
    max_position_embeddings: u32 = 0,
    tie_word_embeddings: bool = false,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    /// The `quantization` block as the quant formats read it (bits 0 = a dense pack).
    quant: abi.Quant = .{},
    /// `{"bits":..,"group_size":..,"mode":".."}` for the quant format's `open`.
    block_json: []const u8 = "{}",
};

pub const gpa = std.heap.c_allocator;

pub fn unsupported(comptime name: []const u8, comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, name ++ ": " ++ why);
}

/// A config value as the engine reads it: JSON null is absent.
pub fn field(o: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = o.get(key) orelse return null;
    return if (v == .null) null else v;
}

pub fn int(v: ?std.json.Value) ?u32 {
    const x = v orelse return null;
    if (x != .integer) return null;
    return std.math.cast(u32, x.integer);
}

pub fn float(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    const f: f64 = switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => return null,
    };
    return if (std.math.isFinite(f)) f else null;
}

pub fn boolean(v: ?std.json.Value) ?bool {
    const x = v orelse return null;
    return if (x == .bool) x.bool else null;
}

/// The root object and the text config the engine reads (`text_config` when nested, else the root).
pub const Doc = struct {
    root: std.json.ObjectMap,
    cfg: std.json.ObjectMap,
    model_type: []const u8,
};

pub const DocError = error{ Broken, NotAnObject, NoModelType, BadTextConfig };

pub fn doc(arena: std.mem.Allocator, json: []const u8) DocError!Doc {
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{}) catch return error.Broken;
    if (v != .object) return error.NotAnObject;
    const root = v.object;
    const mt = field(root, "model_type") orelse return error.NoModelType;
    if (mt != .string) return error.NoModelType;
    const cfg = if (field(root, "text_config")) |tc| (if (tc == .object) tc.object else return error.BadTextConfig) else root;
    return .{ .root = root, .cfg = cfg, .model_type = mt.string };
}

/// `rope_theta` as the engine reads it: the config's, a `rope_parameters` block's winning.
pub fn ropeTheta(cfg: std.json.ObjectMap) ?f64 {
    var theta = float(field(cfg, "rope_theta"));
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) if (float(field(rp.object, "rope_theta"))) |t| {
        theta = t;
    };
    return theta;
}

/// The `partial_rotary_factor` the engine reads (in `rope_parameters`, or at the root of the text config); 1 when absent.
pub fn partialRotary(cfg: std.json.ObjectMap, root_too: bool) error{Unsupported}!f64 {
    var p: f64 = 1.0;
    if (root_too) if (field(cfg, "partial_rotary_factor")) |v| {
        p = float(v) orelse return error.Unsupported;
    };
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) if (field(rp.object, "partial_rotary_factor")) |v| {
        p = float(v) orelse return error.Unsupported;
    };
    return p;
}

fn quantMode(name: []const u8) ?abi.QuantMode {
    inline for (.{ .affine, .nvfp4, .mxfp4, .mxfp8 }) |m| if (std.mem.eql(u8, name, @tagName(m))) return m;
    return null;
}

/// The outer `quantization` block as the engine reads it; `error.Unsupported` for what the native formats cannot
/// serve here (an unknown mode, an affine width MLX has no kernels for, a 2-bit group-128 pack that may be ternary).
pub fn quantization(root: std.json.ObjectMap) error{Unsupported}!abi.Quant {
    return quantizationOf(root, false);
}

/// `quantization`; `rotated` (a Hadamard pack) also takes 2-bit group 128, whose rotated arm the plugin carries.
pub fn quantizationOf(root: std.json.ObjectMap, rotated: bool) error{Unsupported}!abi.Quant {
    // A `--mlx-gguf` file's rebuilt block: the GGUF format reads each tensor's own block type.
    if (json_read.ggufQuant(root) catch return error.Unsupported) |g| return g;
    const q = field(root, "quantization") orelse return .{};
    if (q != .object) return error.Unsupported;
    var out: abi.Quant = .{ .bits = std.math.cast(u8, int(field(q.object, "bits")) orelse 0) orelse return error.Unsupported, .group_size = 64, .mode = .affine };
    if (field(q.object, "group_size")) |g| out.group_size = int(g) orelse return error.Unsupported;
    if (field(q.object, "mode")) |m| if (m == .string) {
        out.mode = quantMode(m.string) orelse return error.Unsupported;
    };
    var it = q.object.iterator();
    while (it.next()) |kv| if (kv.value_ptr.* == .object) if (kv.value_ptr.object.get("mode")) |m| {
        if (m != .string or quantMode(m.string) == null) return error.Unsupported;
    };
    if (out.mode == .affine and out.bits != 0) switch (out.bits) {
        3, 4, 5, 6, 8 => {},
        2 => if (out.group_size == 128 and !rotated) return error.Unsupported,
        else => return error.Unsupported,
    };
    return if (out.bits == 0) .{} else out;
}

/// The quant block as the format's `open` reads it (owned by `gpa`).
pub fn blockJson(q: abi.Quant) ![]const u8 {
    if (q.bits == 0) return gpa.dupe(u8, "{}");
    return std.fmt.allocPrint(gpa, "{{\"bits\":{d},\"group_size\":{d},\"mode\":\"{t}\"}}", .{ q.bits, q.group_size, q.mode });
}

/// Shell facts every Qwen trunk reads the same way: the context, bos, config.json's eos list, the quant block.
pub fn shellCommon(root: std.json.ObjectMap, cfg: std.json.ObjectMap, s: *abi.Shell) error{Unsupported}!void {
    var max_pos = int(field(cfg, "max_position_embeddings")) orelse 0;
    if (max_pos == 0) max_pos = int(field(root, "max_position_embeddings")) orelse 0;
    s.max_position_embeddings = max_pos;
    s.context_cap = max_pos;
    if (field(cfg, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        s.bos = int(v) orelse return error.Unsupported;
    };
    if (field(root, "eos_token_id")) |v| switch (v) {
        .integer => generation.addEos(s, int(v) orelse return error.Unsupported),
        .array => |arr| for (arr.items) |item| if (item == .integer) generation.addEos(s, int(item) orelse return error.Unsupported),
        else => {},
    };
}

pub fn tied(root: std.json.ObjectMap, cfg: std.json.ObjectMap) bool {
    var t = false;
    if (boolean(field(root, "tie_word_embeddings"))) |v| t = v;
    if (boolean(field(cfg, "tie_word_embeddings"))) |v| t = v;
    return t;
}


/// Keys whose presence means a shape this plugin does not serve (another family's, or a feature the base lacks).
const foreign_keys = [_][]const u8{
    "vision_config",               "audio_config", "linear_num_key_heads",        "full_attention_interval", "expert_quant",
    "use_bidirectional_attention", "modules",      "hidden_size_per_layer_input", "qk_scale_factor",         "query_pre_attn_scalar_global",
};

const P = "qwen3";

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    return parseAs(false, dir, config_json, span, stub, out);
}

/// The parse of a `qwen3` / `qwen3_moe` pack, or (`asr`) of Qwen3-ASR's: a dense qwen3 trunk nested under
/// `thinker_config` (its `text_config`), the pack's quantization at the outer root, audio rows spliced at its
/// `audio_token_id` (the engine runs the audio tower).
pub fn parseAs(comptime asr: bool, dir: *const abi.PackDir, config_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) abi.Verdict {
    if (span != null) return unsupported(P, "a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const d = doc(arena.allocator(), config_json.slice()) catch |e| return switch (e) {
        error.Broken => abi.Verdict.refuse(.failed, "qwen3: config.json does not parse"),
        error.NotAnObject => abi.Verdict.refuse(.failed, "qwen3: config.json is not an object"),
        else => unsupported(P, "no model_type or text_config"),
    };
    const types: []const []const u8 = if (asr) &.{"qwen3_asr"} else &model_types;
    const type_name = for (types) |t| {
        if (std.mem.eql(u8, t, d.model_type)) break t;
    } else return unsupported(P, "another model_type");
    const moe = !asr and !std.mem.eql(u8, type_name, "qwen3");
    const outer = d.root;
    const root = if (asr) thinker: {
        const t = field(outer, "thinker_config") orelse return unsupported(P, "no thinker_config");
        break :thinker if (t == .object) t.object else return unsupported(P, "thinker_config");
    } else outer;
    const cfg = if (asr) text: {
        const t = field(root, "text_config") orelse return unsupported(P, "no thinker text_config");
        if (t != .object) return unsupported(P, "thinker text_config");
        const mt = field(t.object, "model_type") orelse return unsupported(P, "a thinker trunk other than qwen3");
        if (mt != .string or !std.mem.eql(u8, mt.string, "qwen3")) return unsupported(P, "a thinker trunk other than qwen3");
        break :text t.object;
    } else d.cfg;
    if (quant_config.foreign(outer)) return unsupported(P, "a quantization_config naming a quant_method");
    for (foreign_keys) |k| {
        // Qwen3-ASR's audio tower is the engine's; its rows splice at the placeholder.
        if (asr and std.mem.eql(u8, k, "audio_config")) continue;
        if (field(cfg, k) != null or field(root, k) != null) return unsupported(P, "a key this plugin does not model");
    }
    if (!json_read.allFullAttention(cfg)) return unsupported(P, "layer_types other than full attention");
    // An embedding pack (Qwen3-Embedding: `pooling_mode` or a sentence-transformers sidecar) is the dense trunk; the
    // engine reads its pooling, the dense attention serves the padded batch.
    if ((moe or asr) and (field(root, "pooling_mode") != null or dir.has("1_Pooling/config.json"))) return unsupported(P, "an embedding pack on the MoE layout");
    const act = field(cfg, "hidden_act") orelse return unsupported(P, "no hidden_act");
    if (act != .string or !std.mem.eql(u8, act.string, "silu")) return unsupported(P, "hidden_act other than silu");

    var c: Config = .{
        .moe = moe,
        .vocab_size = int(field(cfg, "vocab_size")) orelse return unsupported(P, "vocab_size"),
        .hidden_size = int(field(cfg, "hidden_size")) orelse return unsupported(P, "hidden_size"),
        .num_hidden_layers = int(field(cfg, "num_hidden_layers")) orelse return unsupported(P, "num_hidden_layers"),
        .num_attention_heads = int(field(cfg, "num_attention_heads")) orelse return unsupported(P, "num_attention_heads"),
        .num_key_value_heads = int(field(cfg, "num_key_value_heads")) orelse return unsupported(P, "num_key_value_heads"),
        .head_dim = 0,
        .rms_norm_eps = @floatCast(float(field(cfg, "rms_norm_eps")) orelse return unsupported(P, "rms_norm_eps")),
        .rope_theta = @floatCast(ropeTheta(cfg) orelse return unsupported(P, "no rope_theta")),
        .query_pre_attn_scalar = 0,
    };
    if (c.vocab_size == 0 or c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
        c.num_hidden_layers == 0 or c.num_hidden_layers > 1024 or c.num_attention_heads == 0 or
        c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0)
        return unsupported(P, "dimensions out of range");
    // The MoE parse keeps the engine's stale 256 when the config omits head_dim; refuse that instead.
    c.head_dim = if (field(cfg, "head_dim")) |v| int(v) orelse return unsupported(P, "head_dim") else if (moe) return unsupported(P, "no head_dim") else c.hidden_size / c.num_attention_heads;
    if (c.head_dim == 0 or c.head_dim % 2 != 0 or c.head_dim > 512) return unsupported(P, "head_dim");
    c.query_pre_attn_scalar = if (field(cfg, "query_pre_attn_scalar")) |v| int(v) orelse return unsupported(P, "query_pre_attn_scalar") else c.head_dim;
    if (c.query_pre_attn_scalar == 0) return unsupported(P, "query_pre_attn_scalar");
    if ((partialRotary(cfg, false) catch return unsupported(P, "partial_rotary_factor")) != 1.0) return unsupported(P, "a partial rotary factor");
    c.rope = rope_scaling.parse(cfg);
    c.max_position_embeddings = int(field(cfg, "max_position_embeddings")) orelse int(field(root, "max_position_embeddings")) orelse 0;
    if (c.rope == .dynamic and c.max_position_embeddings == 0) c.rope = .unmodeled;
    c.tie_word_embeddings = tied(root, cfg);
    if (moe) {
        c.num_experts = int(field(cfg, "num_experts")) orelse return unsupported(P, "num_experts");
        c.num_experts_per_tok = int(field(cfg, "num_experts_per_tok")) orelse return unsupported(P, "num_experts_per_tok");
        c.moe_intermediate_size = int(field(cfg, "moe_intermediate_size")) orelse return unsupported(P, "moe_intermediate_size");
        if (c.num_experts < 2 or c.num_experts > 1024 or c.num_experts_per_tok == 0 or c.num_experts_per_tok > c.num_experts or c.moe_intermediate_size == 0)
            return unsupported(P, "expert dimensions out of range");
        // Every layer routed, no shared expert: the layout the engine's MoE parse serves.
        if (field(cfg, "mlp_only_layers")) |v| if (v != .array or v.array.items.len != 0) return unsupported(P, "mlp_only_layers");
        if (field(cfg, "decoder_sparse_step")) |v| if ((int(v) orelse 0) != 1) return unsupported(P, "decoder_sparse_step");
        if ((int(field(cfg, "shared_expert_intermediate_size")) orelse 0) != 0) return unsupported(P, "a shared expert");
        c.intermediate_size = int(field(cfg, "intermediate_size")) orelse 0;
    } else {
        c.intermediate_size = int(field(cfg, "intermediate_size")) orelse return unsupported(P, "intermediate_size");
        if (c.intermediate_size == 0) return unsupported(P, "intermediate_size");
        if (field(cfg, "num_experts")) |v| if ((int(v) orelse 0) > 0) return unsupported(P, "experts on the dense layout");
    }
    c.quant = quantization(outer) catch return unsupported(P, "quantization");

    var p: abi.Parsed = .{ .caps = .{ .batches_decode = !moe and c.rope != .dynamic, .moe = moe, .dflash_capture = true } };
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
        .act_dtype = .bf16,
        .quant = c.quant,
        .model_type = abi.Str.of(if (moe) "qwen3_moe" else "qwen3"),
    };
    shellCommon(root, cfg, s) catch return unsupported(P, "bos or eos ids");
    if (rope_scaling.layoutNamespace(c.rope)) |ns| p.cache.layout_namespace = abi.Str.of(ns);
    p.chat.sampling_defaults = .{ .top_k = 20, .top_p = 0.95 };
    // Thinking stays off unless the pack's generation_config asks (the host reads that).
    p.chat.thinking_default = .off;
    p.chat.reasoning_dialect = .think_tags;
    // Discovery's stub parse stops at config.json; the load's parse reads the sidecar too.
    if (!stub) generation.apply(dir, s);
    if (asr) p.placeholder_ids.audio = int(field(root, "audio_token_id")) orelse return unsupported(P, "audio_token_id");

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "qwen3: out of memory");
    cfg_out.* = c;
    cfg_out.block_json = blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, "qwen3: out of memory");
    };
    p.private = cfg_out;
    out.* = p;
    return .ok;
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

fn parseJson(json: []const u8, stub: bool, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, stub, out);
}

const qwen3_8b =
    \\{"architectures":["Qwen3ForCausalLM"],"model_type":"qwen3","hidden_size":4096,"intermediate_size":12288,
    \\ "num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"vocab_size":151936,
    \\ "rms_norm_eps":1e-6,"rope_theta":1000000,"rope_scaling":null,"max_position_embeddings":40960,"hidden_act":"silu",
    \\ "sliding_window":null,"use_sliding_window":false,"max_window_layers":36,"tie_word_embeddings":false,
    \\ "bos_token_id":151643,"eos_token_id":151645,"quantization":{"group_size":64,"bits":4}}
;

const qwen3_30b_a3b =
    \\{"model_type":"qwen3_moe","hidden_size":2048,"intermediate_size":6144,"num_hidden_layers":48,
    \\ "num_attention_heads":32,"num_key_value_heads":4,"head_dim":128,"vocab_size":151936,"rms_norm_eps":1e-6,
    \\ "rope_theta":1000000.0,"max_position_embeddings":40960,"hidden_act":"silu","num_experts":128,
    \\ "num_experts_per_tok":8,"moe_intermediate_size":768,"norm_topk_prob":true,"decoder_sparse_step":1,
    \\ "mlp_only_layers":[],"tie_word_embeddings":false,"eos_token_id":151645,"quantization":{"group_size":64,"bits":4}}
;

test "qwen3 config: a Qwen3-8B pack parses into the shell, batching, the family sampling defaults" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(qwen3_8b, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(p.caps.batches_decode and !p.caps.moe);
    try testing.expectEqual(@as(u32, 128), of(&p).query_pre_attn_scalar);
    try testing.expectEqualSlices(u32, &.{151645}, p.shell.eos[0..p.shell.eos_count]);
    try testing.expectEqualStrings("qwen3", p.shell.model_type.slice());
    try testing.expectEqual(@as(i32, 20), p.chat.sampling_defaults.top_k);
    try testing.expect(p.chat.thinking_default == .off and p.chat.reasoning_dialect == .think_tags);
}

test "qwen3 config: a Qwen3-30B-A3B pack is the MoE layout, serial, its expert facts in the shell" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(qwen3_30b_a3b, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(!p.caps.batches_decode and p.caps.moe);
    try testing.expectEqual(@as(u32, 8), p.shell.num_experts_per_tok);
    try testing.expectEqual(@as(u32, 128), of(&p).num_experts);
    try testing.expectEqualStrings("qwen3_moe", p.shell.model_type.slice());
}

test "qwen3 config: shapes this plugin does not serve are refused unsupported, broken JSON failed" {
    inline for (.{
        "{\"model_type\":\"llama\"}",
        qwen3_8b[0 .. qwen3_8b.len - 1] ++ ",\"layer_types\":[\"sliding_attention\"]}",
        qwen3_30b_a3b[0 .. qwen3_30b_a3b.len - 1] ++ ",\"pooling_mode\":\"last_token\"}",
        qwen3_30b_a3b[0 .. qwen3_30b_a3b.len - 1] ++ ",\"shared_expert_intermediate_size\":512}",
    }) |json| {
        var p: abi.Parsed = .{};
        const v = parseJson(json, true, &p);
        try testing.expect(v.code == .unsupported and p.private == null);
    }
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("{", true, &p).code);
}

test "qwen3 config: the defaults released MoE configs carry are served (empty mlp_only_layers, sparse step 1, all-full layer_types, a null rope_scaling)" {
    const v5 = qwen3_30b_a3b[0 .. qwen3_30b_a3b.len - 1] ++ ",\"layer_types\":[\"full_attention\",\"full_attention\"],\"rope_scaling\":null}";
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(v5, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(of(&p).rope == .none and p.caps.moe);
}

test "qwen3 config: the rope scalings mlx-lm builds are served, their K/V under a fresh SSD namespace" {
    inline for (.{
        .{ ",\"rope_parameters\":{\"rope_type\":\"yarn\",\"factor\":4.0,\"original_max_position_embeddings\":32768}}", true },
        .{ ",\"rope_parameters\":{\"type\":\"linear\",\"factor\":2.0}}", true },
        .{ ",\"rope_parameters\":{\"type\":\"dynamic\",\"factor\":2.0}}", false },
    }) |case| {
        inline for (.{ qwen3_8b, qwen3_30b_a3b }) |base| {
            var p: abi.Parsed = .{};
            const cfg = base[0 .. base.len - 1] ++ case[0];
            try testing.expect(parseJson(cfg, true, &p).isOk());
            defer freeParsed(&p);
            try testing.expect(of(&p).rope != .none);
            try testing.expectEqualStrings("scaled-rope-v1", p.cache.layout_namespace.slice());
            // Dynamic NTK ropes each forward at the base its positions reach: no batched rows.
            if (!case[1]) try testing.expect(!p.caps.batches_decode);
        }
    }
}

test "qwen3 config: transformers v5's all-full layer_types and rope_parameters read as the plain rope" {
    const v5 = qwen3_8b[0 .. qwen3_8b.len - 1] ++ ",\"layer_types\":[\"full_attention\"],\"rope_parameters\":{\"rope_type\":\"default\",\"rope_theta\":5000000}}";
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(v5, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(f32, 5000000), of(&p).rope_theta);
}
