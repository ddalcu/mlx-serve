//! Gemma 3 packs (`gemma3_text`, and `gemma3` whose SigLIP rows the engine splices at `image_token_index`): the claim,
//! and the parse into `Parsed` plus the Gemma decoder `Config` that Gemma 4 shares (`readCommon`). MLX-free. A config
//! this plugin does not serve bit-equal is refused `unsupported` (the legacy path keeps it).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const json_read = @import("../../sdk/config/json.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "gemma3",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/gemma-3-1b-it-4bit"},
};

pub const model_types = [_][]const u8{ "gemma3", "gemma3_text" };

pub const claims = sdk.claimsByType(&model_types);

/// Every block of the Gemma 3 decoder serves batched rows (`plugin.zig` checks it against the base).
pub const batches_decode = true;

/// Layers past this index follow the pattern rule (the engine's per-layer table length).
pub const max_layer_types = 128;

/// The Gemma decoder's fields, named as the engine's `ModelConfig` names them, with its defaults: a field the pack
/// omits reads as the engine read it.
pub const Config = struct {
    model_type: []const u8 = "gemma3",
    weight_prefix: []const u8 = "model",
    vocab_size: u32 = 262208,
    hidden_size: u32 = 3840,
    intermediate_size: u32 = 15360,
    num_hidden_layers: u32 = 48,
    num_attention_heads: u32 = 16,
    num_key_value_heads: u32 = 8,
    head_dim: u32 = 256,
    max_position_embeddings: u32 = 0,
    rms_norm_eps: f32 = 1e-6,
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    rope_scaling_factor: f32 = 1.0,
    rope_proportional: bool = false,
    rope_proportional_factor: f32 = 1.0,
    partial_rotary_factor_global: f32 = 1.0,
    has_sliding_window: bool = true,
    sliding_window: u32 = 1024,
    sliding_window_pattern: u32 = 6,
    has_explicit_layer_types: bool = false,
    layer_is_global: [max_layer_types]bool = @splat(false),
    query_pre_attn_scalar: u32 = 256,
    final_logit_softcapping: f32 = 0.0,
    tie_word_embeddings: bool = false,
    /// (1 + w) norms (Gemma 3); Gemma 4's are plain.
    norm_has_offset: bool = true,
    global_head_dim: u32 = 0,
    num_global_key_value_heads: u32 = 0,
    num_kv_shared_layers: u32 = 0,
    attention_k_eq_v: bool = false,
    has_v_norm: bool = false,
    hidden_size_per_layer_input: u32 = 0,
    num_experts: u32 = 0,
    num_experts_per_tok: u32 = 0,
    moe_intermediate_size: u32 = 0,
    bos_token_id: ?u32 = null,
    /// The `quantization` block as the quant formats read it (bits 0 = a dense pack).
    quant: abi.Quant = .{},
    /// `{"bits":..,"group_size":..,"mode":".."}` for the quant format's `open`.
    block_json: []const u8 = "{}",

    /// Full attention on `layer`: every layer without a window, the declared type map, else the pattern (the global
    /// layer closes each group).
    pub fn isGlobalLayer(c: *const Config, layer: u32) bool {
        if (!c.has_sliding_window) return true;
        if (c.has_explicit_layer_types and layer < max_layer_types) return c.layer_is_global[layer];
        return layer % c.sliding_window_pattern == c.sliding_window_pattern - 1;
    }

    /// The earlier layer whose KV `layer` reads (Gemma 4's shared tail: the last layer of its own type before the
    /// tail), null when it projects its own.
    pub fn kvSource(c: *const Config, layer: u32) ?u32 {
        if (c.num_kv_shared_layers == 0) return null;
        const first_shared = c.num_hidden_layers - c.num_kv_shared_layers;
        if (layer < first_shared) return null;
        const global = c.isGlobalLayer(layer);
        var j: u32 = first_shared;
        while (j > 0) {
            j -= 1;
            if (c.isGlobalLayer(j) == global) return j;
        }
        return null;
    }

    /// Global layers project wider heads than sliding ones.
    pub fn dualHeadDim(c: *const Config) bool {
        return c.global_head_dim > 0 and c.global_head_dim != c.head_dim;
    }
};

const gpa = std.heap.c_allocator;

pub fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "gemma: " ++ why);
}

/// A config value as the engine reads it: JSON null is absent.
pub fn field(o: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = o.get(key) orelse return null;
    return if (v == .null) null else v;
}

/// An integer that fits u32 (the engine refuses any other value).
pub fn int(v: std.json.Value) error{Unsupported}!u32 {
    if (v != .integer) return error.Unsupported;
    return std.math.cast(u32, v.integer) orelse error.Unsupported;
}

/// A number as the engine reads it: an integer converts, any other kind reads 0, a non-finite value is refused.
pub fn float(v: std.json.Value) error{Unsupported}!f32 {
    const f: f32 = switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| @floatCast(f),
        else => 0.0,
    };
    if (!std.math.isFinite(f)) return error.Unsupported;
    return f;
}

fn quantMode(name: []const u8) ?abi.QuantMode {
    inline for (.{ .affine, .nvfp4, .mxfp4, .mxfp8 }) |m| if (std.mem.eql(u8, name, @tagName(m))) return m;
    return null;
}

/// The fields every Gemma decoder reads, in the engine's order and precedence (`text_config` first where nested).
pub fn readCommon(root: std.json.ObjectMap, cfg: std.json.ObjectMap, c: *Config) error{Unsupported}!void {
    inline for (.{ "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings", "query_pre_attn_scalar" }) |k| {
        if (field(cfg, k)) |v| @field(c, k) = try int(v);
    }
    inline for (.{ "rms_norm_eps", "rope_theta" }) |k| if (field(cfg, k)) |v| {
        @field(c, k) = try float(v);
    };
    inline for (.{ "num_experts", "num_experts_per_tok", "moe_intermediate_size" }) |k| if (field(cfg, k)) |v| if (v == .integer) {
        @field(c, k) = try int(v);
    };
    if (field(cfg, "top_k_experts")) |v| if (v == .integer) {
        c.num_experts_per_tok = try int(v);
    };
    if (field(cfg, "bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        c.bos_token_id = try int(v);
    };
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) {
        if (field(rp.object, "rope_theta")) |v| c.rope_theta = try float(v);
    };
    if (cfg.get("sliding_window")) |v| {
        if (v == .null) c.has_sliding_window = false else {
            c.sliding_window = try int(v);
            c.has_sliding_window = true;
        }
    }
    if (field(cfg, "sliding_window_pattern")) |v| c.sliding_window_pattern = try int(v);
    if (field(cfg, "rope_local_base_freq")) |v| c.rope_local_base_freq = try float(v);
    if (field(cfg, "rope_scaling")) |rs| if (rs == .object) if (field(rs.object, "factor")) |v| {
        c.rope_scaling_factor = try float(v);
    };
    if (field(cfg, "layer_types")) |lt| if (lt == .array) {
        c.has_explicit_layer_types = true;
        for (lt.array.items, 0..) |item, i| {
            if (i >= max_layer_types) break;
            if (item == .string) c.layer_is_global[i] = std.mem.eql(u8, item.string, "full_attention");
        }
    };
    inline for (.{ "global_head_dim", "num_global_key_value_heads", "num_kv_shared_layers", "hidden_size_per_layer_input" }) |k| if (field(cfg, k)) |v| if (v == .integer) {
        @field(c, k) = try int(v);
    };
    if (field(cfg, "attention_k_eq_v")) |v| if (v == .bool) {
        c.attention_k_eq_v = v.bool;
    };
    if (field(cfg, "final_logit_softcapping")) |v| c.final_logit_softcapping = try float(v);
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) {
        if (field(rp.object, "full_attention")) |fa| if (fa == .object) {
            if (field(fa.object, "rope_theta")) |v| c.rope_theta = try float(v);
            if (field(fa.object, "partial_rotary_factor")) |v| c.partial_rotary_factor_global = try float(v);
            if (field(fa.object, "rope_type")) |v| if (v == .string and std.mem.eql(u8, v.string, "proportional")) {
                c.rope_proportional = true;
                if (field(fa.object, "factor")) |f| c.rope_proportional_factor = try float(f);
            };
        };
        if (field(rp.object, "sliding_attention")) |sa| if (sa == .object) {
            if (field(sa.object, "rope_theta")) |v| c.rope_local_base_freq = try float(v);
        };
        if (field(rp.object, "rope_theta")) |v| c.rope_theta = try float(v);
    };
    inline for (.{ root, cfg }) |o| if (field(o, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };
    if (c.max_position_embeddings == 0) if (field(root, "max_position_embeddings")) |v| if (v == .integer) {
        c.max_position_embeddings = try int(v);
    };
}

/// The `quantization` block (affine bits the MLX kernels serve, modes MLX knows) into `c.quant`.
pub fn readQuant(root: std.json.ObjectMap, c: *Config) error{Unsupported}!void {
    // A `--mlx-gguf` file's rebuilt block: the GGUF format reads each tensor's own block type.
    if (json_read.ggufQuant(root) catch return error.Unsupported) |g| {
        c.quant = g;
        return;
    }
    const q = field(root, "quantization") orelse return;
    if (q != .object) return error.Unsupported;
    c.quant = .{ .group_size = 64, .mode = .affine };
    if (field(q.object, "bits")) |v| c.quant.bits = std.math.cast(u8, try int(v)) orelse return error.Unsupported;
    if (field(q.object, "group_size")) |v| c.quant.group_size = try int(v);
    if (field(q.object, "mode")) |m| if (m == .string) {
        c.quant.mode = quantMode(m.string) orelse return error.Unsupported;
    };
    // Per-tensor overrides resolve from each tensor's geometry at load; their modes must be ones MLX serves.
    var it = q.object.iterator();
    while (it.next()) |kv| if (kv.value_ptr.* == .object) if (kv.value_ptr.object.get("mode")) |m| {
        if (m == .string and quantMode(m.string) == null) return error.Unsupported;
    };
    if (c.quant.mode == .affine and c.quant.bits != 0) switch (c.quant.bits) {
        // 2-bit at group 128 may be ternary: the load arms qmv2's kernels when it is (`armTernary`).
        2, 3, 4, 5, 6, 8 => {},
        else => return error.Unsupported,
    };
    if (c.quant.bits == 0) c.quant = .{};
}

/// config.json's `eos_token_id` (scalar or list) into the stop set.
pub fn readEos(root: std.json.ObjectMap, s: *abi.Shell) error{Unsupported}!void {
    const v = field(root, "eos_token_id") orelse return;
    switch (v) {
        .integer => generation.addEos(s, try int(v)),
        .array => |arr| for (arr.items) |item| if (item == .integer) generation.addEos(s, try int(item)),
        else => {},
    }
}

/// `<eos>` and `<end_of_turn>`, merged after the config's own ids: every Gemma template ends a turn with 106.
pub const terminators = [_]u32{ 1, 106 };

/// Keys a pack of this family carries only for a shape this plugin does not serve bit-equal yet.
const gemma4_keys = [_][]const u8{ "global_head_dim", "num_kv_shared_layers", "hidden_size_per_layer_input", "attention_k_eq_v" };

/// The shell, caps and facts every Gemma decoder declares from its `Config`, then its private `Config` (owned by the
/// result: `freeParsed`).
pub fn finish(dir: *const abi.PackDir, root: std.json.ObjectMap, c: Config, stub: bool, caps: abi.Caps, out: *abi.Parsed) abi.Verdict {
    var p: abi.Parsed = .{ .caps = caps };
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
        .max_position_embeddings = c.max_position_embeddings,
        .context_cap = c.max_position_embeddings,
        .quant = c.quant,
        .act_dtype = .bf16,
        .model_type = abi.Str.of(c.model_type),
    };
    if (c.bos_token_id) |b| s.bos = b;
    readEos(root, s) catch return unsupported("eos_token_id");
    for (terminators) |t| generation.mergeEos(s, t);
    for (terminators, 0..) |t, i| p.tokenizer.extra_eos[i] = t;
    p.tokenizer.extra_eos_count = terminators.len;
    p.chat.sampling_defaults = .{ .top_k = 64, .top_p = 0.95 };
    p.chat.thinking_default = .off;
    p.spec.embed_scale = @sqrt(@as(f32, @floatFromInt(c.hidden_size)));
    p.weights.prefix = abi.Str.of(c.weight_prefix);
    // Discovery's stub parse stops at config.json; the load's parse reads the sidecar too.
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "gemma: out of memory");
    cfg_out.* = c;
    cfg_out.block_json = (if (c.quant.bits == 0)
        gpa.dupe(u8, "{}")
    else
        std.fmt.allocPrint(gpa, "{{\"bits\":{d},\"group_size\":{d},\"mode\":\"{t}\"}}", .{ c.quant.bits, c.quant.group_size, c.quant.mode })) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, "gemma: out of memory");
    };
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

/// The parsed JSON of a Gemma pack: the root and the object the text fields live in.
pub const Doc = struct {
    arena: std.heap.ArenaAllocator,
    root: std.json.ObjectMap,
    cfg: std.json.ObjectMap,
    model_type: []const u8,

    pub fn open(config_json: abi.Str, out: *Doc) abi.Verdict {
        out.arena = std.heap.ArenaAllocator.init(gpa);
        const doc = std.json.parseFromSliceLeaky(std.json.Value, out.arena.allocator(), config_json.slice(), .{}) catch {
            out.arena.deinit();
            return abi.Verdict.refuse(.failed, "gemma: config.json does not parse");
        };
        if (doc != .object) {
            out.arena.deinit();
            return abi.Verdict.refuse(.failed, "gemma: config.json is not an object");
        }
        out.root = doc.object;
        out.cfg = out.root;
        out.model_type = "";
        if (field(out.root, "model_type")) |mt| if (mt == .string) {
            out.model_type = mt.string;
        };
        if (field(out.root, "text_config")) |tc| {
            if (tc != .object) {
                out.arena.deinit();
                return unsupported("text_config is not an object");
            }
            out.cfg = tc.object;
        }
        return .ok;
    }

    pub fn close(d: *Doc) void {
        d.arena.deinit();
    }
};

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    var d: Doc = undefined;
    const opened = Doc.open(config_json, &d);
    if (!opened.isOk()) return opened;
    defer d.close();
    if (span != null) return unsupported("a layer span (the cluster split)");
    const known = for (model_types) |t| {
        if (std.mem.eql(u8, t, d.model_type)) break true;
    } else false;
    if (!known) return unsupported("another model_type");
    // EmbeddingGemma is an encoder (pooled rows, no decode): the embedding_gemma plugin's.
    if (field(d.cfg, "use_bidirectional_attention")) |v| if (v == .bool and v.bool) return unsupported("EmbeddingGemma (a bidirectional encoder)");
    if (field(d.root, "pooling_mode") != null or dir.has("1_Pooling/config.json")) return unsupported("an embedding pack");
    if (foreignQuant(d.root)) return unsupported("a quantization_config naming a quant_method");
    for (gemma4_keys) |k| if (field(d.cfg, k)) |v| switch (v) {
        .integer => if (v.integer != 0) return unsupported("a Gemma 4 trait"),
        .bool => if (v.bool) return unsupported("a Gemma 4 trait"),
        else => {},
    };

    var c: Config = .{};
    const v = readGemma3(&d, &c);
    if (!v.isOk()) return v;
    const fin = finish(dir, d.root, c, stub, .{ .batches_decode = batches_decode, .prefill_mask = true }, out);
    if (!fin.isOk()) return fin;
    readMedia(d.root, out) catch {
        freeParsed(out);
        return unsupported("a placeholder id");
    };
    return .ok;
}

/// The engine's Gemma 3 arm over `d` into `c`: Gemma3TextConfig's head defaults, tied head, (1 + w) norms, the linear
/// rope scale of 8 unless the pack names a factor.
pub fn readGemma3(d: *const Doc, c: *Config) abi.Verdict {
    readCommon(d.root, d.cfg, c) catch return unsupported("a field out of range");
    readQuant(d.root, c) catch return unsupported("quantization");
    // The engine's Gemma 3 arm: Gemma3TextConfig's head defaults, tied head, (1 + w) norms, the linear rope scale
    // of 8 unless the pack names a factor.
    if (field(d.cfg, "num_attention_heads") == null) c.num_attention_heads = 8;
    if (field(d.cfg, "num_key_value_heads") == null) c.num_key_value_heads = 4;
    if (field(d.cfg, "head_dim") == null) c.head_dim = 256;
    c.model_type = "gemma3";
    c.weight_prefix = if (field(d.root, "text_config") != null) "language_model.model" else "model";
    c.tie_word_embeddings = true;
    c.norm_has_offset = true;
    if (c.rope_scaling_factor == 1.0) {
        const named = if (field(d.cfg, "rope_scaling")) |rs| rs == .object and field(rs.object, "factor") != null else false;
        if (!named) c.rope_scaling_factor = 8.0;
    }
    if (c.rope_proportional or c.partial_rotary_factor_global != 1.0) return unsupported("a partial or proportional rope");
    if (c.num_experts > 0) return unsupported("experts");
    if (c.vocab_size == 0 or c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
        c.num_hidden_layers == 0 or c.num_hidden_layers > 1024 or c.num_attention_heads == 0 or c.num_key_value_heads == 0 or
        c.num_attention_heads % c.num_key_value_heads != 0 or c.intermediate_size == 0 or c.head_dim == 0 or
        c.head_dim % 2 != 0 or c.head_dim > 512 or c.query_pre_attn_scalar == 0 or
        (c.has_sliding_window and c.sliding_window_pattern == 0))
        return unsupported("dimensions out of range");
    return .ok;
}

pub const foreignQuant = quant_config.foreign;

/// A 2-bit group-128 affine pack, which may be ternary (every bias exactly -scale).
pub fn ternaryCandidate(c: *const Config) bool {
    return c.quant.mode == .affine and c.quant.bits == 2 and c.quant.group_size == 128;
}

/// The towers a pack carries (`vision_config`, `audio_config`) and the root placeholder ids their rows splice at, as
/// the engine reads them (`image_token_id`, else `image_token_index`).
pub fn readMedia(root: std.json.ObjectMap, out: *abi.Parsed) error{Unsupported}!void {
    out.caps.images = field(root, "vision_config") != null;
    out.caps.audio = field(root, "audio_config") != null;
    if (!out.caps.images and !out.caps.audio) return;
    const ids = &out.placeholder_ids;
    inline for (.{ .{ "image_token_id", "image" }, .{ "audio_token_id", "audio" }, .{ "video_token_id", "video" } }) |k| if (field(root, k[0])) |v| if (v == .integer) {
        @field(ids, k[1]) = try int(v);
    };
    if (ids.image == 0) if (field(root, "image_token_index")) |v| if (v == .integer) {
        ids.image = try int(v);
    };
}

/// An embedding pack's pooling contract: config.json's `pooling_mode`, then the sentence-transformers sidecar (the
/// load's parse only); `.unset` = the host's fallback. A mode this family does not implement fails the parse, never a
/// silent mean-pool. copied_from: plugins/bert/config.zig (`sidecarPooling`).
pub fn readPooling(dir: *const abi.PackDir, root: std.json.ObjectMap, stub: bool) error{Invalid}!abi.Pooling {
    if (field(root, "pooling_mode")) |v| if (v == .string) return poolingNamed(v.string) orelse error.Invalid;
    const sidecar = "1_Pooling/config.json";
    if (stub or !dir.has(sidecar)) return .unset;
    var buf: [16 * 1024]u8 = undefined;
    var why: abi.Verdict = .{};
    const bytes = dir.readFile(sidecar, &buf, &why) orelse return .unset;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), bytes, .{}) catch return .unset;
    if (doc != .object) return .unset;
    const obj = doc.object;
    for ([_]struct { []const u8, abi.Pooling }{ .{ "pooling_mode_lasttoken", .last_token }, .{ "pooling_mode_cls_token", .cls }, .{ "pooling_mode_mean_tokens", .mean } }) |e| {
        if (obj.get(e[0])) |v| if (v == .bool and v.bool) return e[1];
    }
    // ST 6 names the mode in one string.
    if (obj.get("pooling_mode")) |v| if (v == .string) {
        if (std.mem.eql(u8, v.string, "lasttoken")) return .last_token;
        return poolingNamed(v.string) orelse error.Invalid;
    };
    var it = obj.iterator();
    while (it.next()) |e| if (std.mem.startsWith(u8, e.key_ptr.*, "pooling_mode_")) return error.Invalid;
    return .unset;
}

fn poolingNamed(name: []const u8) ?abi.Pooling {
    inline for (.{ .mean, .cls, .last_token }) |p| if (std.mem.eql(u8, name, @tagName(p))) return p;
    return null;
}

/// An encoder's layers keep no cache.
pub fn encoderLayout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const n = @min(of(p).num_hidden_layers, cap);
    for (out[0..n]) |*o| o.* = .{ .kind = .none };
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

/// Every layer's cache as the forward fills it: sliding layers keep a window, the rest every row.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = c.num_hidden_layers;
    if (cap < n) return 0;
    const ghd = if (c.dualHeadDim()) c.global_head_dim else c.head_dim;
    const gkv = if (c.num_global_key_value_heads > 0) c.num_global_key_value_heads else c.num_key_value_heads;
    for (0..n) |i| {
        const li: u32 = @intCast(i);
        const global = c.isGlobalLayer(li);
        const dual = c.dualHeadDim() and global;
        const hd = if (dual) ghd else c.head_dim;
        out[i] = .{
            .kind = if (global) .full else .sliding,
            .window = if (global) 0 else c.sliding_window,
            .kv_heads = if (dual) gkv else c.num_key_value_heads,
            .k_dim = hd,
            .v_dim = hd,
            .kv_source = if (c.kvSource(li)) |s| @intCast(s) else -1,
        };
    }
    return n;
}

// ── tests ──

const testing = std.testing;
const fixture = sdk.fixture;

fn parseJson(json: []const u8, stub: bool, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, stub, out);
}

test "gemma3 config: the harrier fixture parses with the engine's Gemma 3 traits and stop set" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(@embedFile("../../fixtures/model-configs/gemma3-text.json"), true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 256), c.head_dim);
    try testing.expectEqual(@as(f32, 8.0), c.rope_scaling_factor);
    try testing.expectEqual(@as(f32, 1000000.0), c.rope_theta);
    try testing.expect(c.tie_word_embeddings and c.norm_has_offset and c.isGlobalLayer(3));
    try testing.expectEqualSlices(u32, &.{ 1, 106 }, p.shell.eos[0..p.shell.eos_count]);
    try testing.expectEqualStrings("gemma3", p.shell.model_type.slice());
}

test "gemma3 config: a vision pack declares images at its placeholder; EmbeddingGemma and Gemma 4 traits are declined" {
    {
        var p: abi.Parsed = .{};
        try testing.expect(parseJson("{\"model_type\":\"gemma3\",\"text_config\":{},\"vision_config\":{},\"image_token_index\":262144}", true, &p).isOk());
        defer freeParsed(&p);
        try testing.expect(p.caps.images and !p.caps.audio);
        try testing.expectEqual(@as(u32, 262144), p.placeholder_ids.image);
    }
    inline for (.{
        "{\"model_type\":\"gemma3_text\",\"use_bidirectional_attention\":true}",
        "{\"model_type\":\"gemma3_text\",\"num_kv_shared_layers\":4}",
        "{\"model_type\":\"llama\"}",
    }) |json| {
        var p: abi.Parsed = .{};
        try testing.expect(parseJson(json, true, &p).code == .unsupported and p.private == null);
    }
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("{", true, &p).code);
}
