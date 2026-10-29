//! Llama, Mistral and Qwen2 packs (mlx-lm's `llama`/`mistral`/`qwen2` layouts): the claim, and the parse into
//! `Parsed` plus this plugin's own fields. MLX-free. A config this plugin does not model the way the engine always
//! served it is refused `unsupported` (the legacy path keeps it until the plugin learns it).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const rope_scaling = @import("../../sdk/config/rope_scaling.zig");
const json_read = @import("../../sdk/config/json.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "llama",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/Llama-3.2-3B-Instruct-4bit"},
};

pub const model_types = [_][]const u8{ "llama", "mistral", "qwen2" };

pub const claims = sdk.claimsByType(&model_types);

/// A plugin that parses this family's layout: its name (in refusals), its types, and whether its RMS norms are
/// grouped (`layernorm_num_groups`, K2-Horizon).
pub const Family = struct {
    name: []const u8,
    model_types: []const []const u8,
    grouped_norms: bool = false,
};

pub const family: Family = .{ .name = "llama", .model_types = &model_types };

/// Every layer's blocks serve batched rows (`plugin.zig` checks it against the base).
pub const batches_decode = true;

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
    /// The `rope_scaling` block (mlx-lm's `initialize_rope`).
    rope: rope_scaling.Scaling = .none,
    max_position_embeddings: u32 = 0,
    tie_word_embeddings: bool = false,
    /// Channel groups each residual-width RMS norm normalizes apart; 1 = a plain norm.
    norm_groups: u32 = 1,
    /// The `quantization` block as the quant formats read it (bits 0 = a dense pack).
    quant: abi.Quant = .{},
    /// `{"bits":..,"group_size":..,"mode":".."}` for the quant format's `open`.
    block_json: []const u8 = "{}",
};

const gpa = std.heap.c_allocator;

/// A config value as the engine reads it: JSON null is absent.
pub fn field(o: std.json.ObjectMap, key: []const u8) ?std.json.Value {
    const v = o.get(key) orelse return null;
    return if (v == .null) null else v;
}

/// A key that turns a feature on: present with a value other than false or 0.
pub fn set(o: std.json.ObjectMap, key: []const u8) bool {
    return switch (field(o, key) orelse return false) {
        .bool => |b| b,
        .integer => |i| i != 0,
        else => true,
    };
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

fn quantMode(name: []const u8) ?abi.QuantMode {
    inline for (.{ .affine, .nvfp4, .mxfp4, .mxfp8 }) |m| if (std.mem.eql(u8, name, @tagName(m))) return m;
    return null;
}

/// Keys whose presence means a shape this plugin does not serve (another family's, or a feature the base lacks).
const foreign_keys = [_][]const u8{
    "query_pre_attn_scalar", "vision_config",               "num_experts",
    "num_local_experts",     "pooling_mode",                "expert_quant",
    "modules",               "use_bidirectional_attention", "linear_num_key_heads",
    "hidden_size_per_layer_input",
};

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    return parseFamily(family, dir, config_json, span, stub, out);
}

/// `fam`'s parse: the shell and this layout's `Config` into `out`.
pub fn parseFamily(comptime fam: Family, dir: *const abi.PackDir, config_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) abi.Verdict {
    const unsupported = struct {
        fn v(comptime why: []const u8) abi.Verdict {
            return abi.Verdict.refuse(.unsupported, fam.name ++ ": " ++ why);
        }
    }.v;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, fam.name ++ ": config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, fam.name ++ ": config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return unsupported("no model_type");
    if (mt != .string) return unsupported("no model_type");
    const type_name = for (fam.model_types) |t| {
        if (std.mem.eql(u8, t, mt.string)) break t;
    } else return unsupported("another model_type");
    const cfg = if (field(root, "text_config")) |tc| (if (tc == .object) tc.object else return unsupported("text_config")) else root;
    if (fam.grouped_norms) {
        // K2-Horizon's MoVA sizes run another attention; its dense configs ship these keys idle.
        inline for (.{ "mova_num_experts", "num_experts" }) |k| if (field(cfg, k)) |v| if (v != .integer or v.integer != 0) return unsupported("a MoVA / MoE size");
        if (field(cfg, "query_key_norm")) |v| if (v != .bool or v.bool) return unsupported("query_key_norm");
        if (field(cfg, "attention_gate_func") != null) return unsupported("attention_gate_func");
    }
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
    for (foreign_keys) |k| {
        if (fam.grouped_norms and std.mem.eql(u8, k, "num_experts")) continue;
        if (field(cfg, k) != null or field(root, k) != null) return unsupported("a key this plugin does not model");
    }
    if (!json_read.allFullAttention(cfg)) return unsupported("layer_types other than full attention");
    // A sentence-transformers pooling sidecar makes the pack an embedding model.
    if (dir.has("1_Pooling/config.json")) return unsupported("an embedding pack (1_Pooling)");

    // mlx-lm's llama MLP is SwiGLU whatever the config says; an absent `hidden_act` is HF's default, silu.
    if (field(cfg, "hidden_act")) |act| if (act != .string or !std.mem.eql(u8, act.string, "silu")) return unsupported("hidden_act other than silu");

    var c: Config = .{
        .vocab_size = int(field(cfg, "vocab_size")) orelse return unsupported("vocab_size"),
        .hidden_size = int(field(cfg, "hidden_size")) orelse return unsupported("hidden_size"),
        .intermediate_size = int(field(cfg, "intermediate_size")) orelse return unsupported("intermediate_size"),
        .num_hidden_layers = int(field(cfg, "num_hidden_layers")) orelse return unsupported("num_hidden_layers"),
        .num_attention_heads = int(field(cfg, "num_attention_heads")) orelse return unsupported("num_attention_heads"),
        .num_key_value_heads = int(field(cfg, "num_key_value_heads")) orelse return unsupported("num_key_value_heads"),
        .head_dim = 0,
        .rms_norm_eps = @floatCast(float(field(cfg, "rms_norm_eps")) orelse return unsupported("rms_norm_eps")),
        .rope_theta = 0,
    };
    if (c.vocab_size == 0 or c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
        c.num_hidden_layers == 0 or c.num_hidden_layers > 1024 or c.num_attention_heads == 0 or
        c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or c.intermediate_size == 0)
        return unsupported("dimensions out of range");
    c.head_dim = if (field(cfg, "head_dim")) |v| int(v) orelse return unsupported("head_dim") else c.hidden_size / c.num_attention_heads;
    // The legacy decode step at hd 256 runs a fused norm+RoPE kernel the dense base does not carry yet.
    if (c.head_dim == 0 or c.head_dim % 2 != 0 or c.head_dim == 256 or c.head_dim > 512) return unsupported("head_dim");
    const theta = float(field(cfg, "rope_theta")) orelse
        (if (field(cfg, "rope_parameters")) |rp| (if (rp == .object) float(field(rp.object, "rope_theta")) else null) else null) orelse
        return unsupported("no rope_theta");
    c.rope_theta = @floatCast(theta);
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) if (float(field(rp.object, "rope_theta"))) |t| {
        c.rope_theta = @floatCast(t);
    };
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) if (field(rp.object, "partial_rotary_factor")) |v| {
        if ((float(v) orelse return unsupported("a partial rotary factor")) != 1.0) return unsupported("a partial rotary factor");
    };
    c.rope = rope_scaling.parse(cfg);
    if (fam.grouped_norms) if (field(cfg, "layernorm_num_groups")) |v| if (v == .integer and v.integer > 1) {
        c.norm_groups = int(v) orelse return unsupported("layernorm_num_groups");
        if (c.hidden_size % c.norm_groups != 0) return unsupported("layernorm_num_groups");
    };
    if (field(root, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };
    if (field(cfg, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };

    var p: abi.Parsed = .{ .caps = .{ .batches_decode = batches_decode and c.rope != .dynamic, .dflash_capture = true } };
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.intermediate_size,
        .num_hidden_layers = c.num_hidden_layers,
        .num_attention_heads = c.num_attention_heads,
        .num_key_value_heads = c.num_key_value_heads,
        .head_dim = c.head_dim,
        .act_dtype = .bf16,
        .model_type = abi.Str.of(type_name),
    };
    readShell(cfg, root, s) catch return unsupported("a token id or the context out of range");
    c.max_position_embeddings = @intCast(s.max_position_embeddings);
    if (c.rope == .dynamic and c.max_position_embeddings == 0) c.rope = .unmodeled;
    c.quant = readQuant(root) catch return unsupported("the quantization block");
    s.quant = c.quant;
    if (rope_scaling.layoutNamespace(c.rope)) |ns| p.cache.layout_namespace = abi.Str.of(ns);
    // Discovery's stub parse stops at config.json; the load's parse reads the sidecar too.
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, fam.name ++ ": out of memory");
    cfg_out.* = c;
    cfg_out.block_json = blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, fam.name ++ ": out of memory");
    };
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

/// The context, bos and eos ids every dense config carries, into the shell (`text_config` first, then the root).
pub fn readShell(cfg: std.json.ObjectMap, root: std.json.ObjectMap, s: *abi.Shell) error{Unsupported}!void {
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

/// The root `quantization` block as the engine reads it (absent or bits 0 = a dense pack). Refused: modes MLX does
/// not serve, affine widths without kernels, and 2-bit group 128 (possibly a ternary pack the engine's own kernels
/// serve).
pub fn readQuant(root: std.json.ObjectMap) error{Unsupported}!abi.Quant {
    const q = field(root, "quantization") orelse return .{};
    if (q != .object) return error.Unsupported;
    var out: abi.Quant = .{ .bits = std.math.cast(u8, int(field(q.object, "bits")) orelse 0) orelse return error.Unsupported, .group_size = 64, .mode = .affine };
    if (field(q.object, "group_size")) |g| out.group_size = int(g) orelse return error.Unsupported;
    if (field(q.object, "mode")) |m| if (m == .string) {
        out.mode = quantMode(m.string) orelse return error.Unsupported;
    };
    // Per-tensor overrides resolve from each tensor's geometry at load; their modes must be ones MLX serves.
    var it = q.object.iterator();
    while (it.next()) |kv| if (kv.value_ptr.* == .object) if (kv.value_ptr.object.get("mode")) |m| {
        if (m != .string or quantMode(m.string) == null) return error.Unsupported;
    };
    if (out.mode == .affine and out.bits != 0) switch (out.bits) {
        3, 4, 5, 6, 8 => {},
        2 => if (out.group_size == 128) return error.Unsupported,
        else => return error.Unsupported,
    };
    return if (out.bits == 0) .{} else out;
}

/// `{"bits":..,"group_size":..,"mode":".."}` for the quant format's `open` ("{}" = dense); owned by `gpa`.
pub fn blockJson(q: abi.Quant) ![]u8 {
    if (q.bits == 0) return gpa.dupe(u8, "{}");
    return std.fmt.allocPrint(gpa, "{{\"bits\":{d},\"group_size\":{d},\"mode\":\"{t}\"}}", .{ q.bits, q.group_size, q.mode });
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

const llama32 =
    \\{"model_type":"llama","hidden_size":3072,"intermediate_size":8192,"num_hidden_layers":28,
    \\ "num_attention_heads":24,"num_key_value_heads":8,"head_dim":128,"vocab_size":128256,"rms_norm_eps":1e-5,
    \\ "rope_theta":500000.0,"max_position_embeddings":131072,"hidden_act":"silu","tie_word_embeddings":true,
    \\ "bos_token_id":128000,"eos_token_id":[128001,128008,128009],
    \\ "rope_scaling":{"factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,
    \\   "original_max_position_embeddings":8192,"rope_type":"llama3"},
    \\ "quantization":{"group_size":64,"bits":4}}
;

test "llama config: a Llama 3.2 pack parses into the shell, its llama3 rope and its SSD namespace" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(llama32, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 128), p.shell.head_dim);
    try testing.expectEqualSlices(u32, &.{ 128001, 128008, 128009 }, p.shell.eos[0..p.shell.eos_count]);
    try testing.expectEqual(@as(i64, 128000), p.shell.bos);
    try testing.expectEqual(@as(u8, 4), p.shell.quant.bits);
    try testing.expectEqualStrings("llama3-rope-v1", p.cache.layout_namespace.slice());
    try testing.expectEqual(@as(f64, 32), of(&p).rope.llama3.factor);
    try testing.expectEqualStrings("{\"bits\":4,\"group_size\":64,\"mode\":\"affine\"}", of(&p).block_json);
}

test "llama config: shapes this plugin does not serve are refused unsupported, broken JSON failed" {
    inline for (.{
        "{\"model_type\":\"gemma3\"}",
        "{\"model_type\":\"llama\",\"hidden_act\":\"gelu\"}",
        llama32[0 .. llama32.len - 1] ++ ",\"layer_types\":[\"full_attention\",\"sliding_attention\"]}",
    }) |json| {
        var p: abi.Parsed = .{};
        const v = parseJson(json, true, &p);
        try testing.expect(v.code == .unsupported and p.private == null);
    }
    var p: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.failed, parseJson("{", true, &p).code);
}

test "llama config: transformers v5's all-full layer_types is the layout served" {
    const v5 = llama32[0 .. llama32.len - 1] ++ ",\"layer_types\":[\"full_attention\",\"full_attention\"]}";
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(v5, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(p.caps.batches_decode);
}

test "llama config: the rope scalings mlx-lm builds are served; any other block runs unscaled, as the engine served it" {
    inline for (.{
        .{ "\"rope_type\":\"linear\"", "scaled-rope-v1", true },
        .{ "\"rope_type\":\"dynamic\"", "scaled-rope-v1", false },
        .{ "\"rope_type\":\"yarn\"", "scaled-rope-v1", true },
        .{ "\"rope_type\":\"longrope\"", "", true },
        .{ "\"rope_type\":\"su\"", "", true },
    }) |case| {
        const json = try std.mem.replaceOwned(u8, testing.allocator, llama32, "\"rope_type\":\"llama3\"", case[0]);
        defer testing.allocator.free(json);
        var p: abi.Parsed = .{};
        try testing.expect(parseJson(json, true, &p).isOk());
        defer freeParsed(&p);
        try testing.expectEqualStrings(case[1], p.cache.layout_namespace.slice());
        try testing.expectEqual(case[2], p.caps.batches_decode);
    }
}
