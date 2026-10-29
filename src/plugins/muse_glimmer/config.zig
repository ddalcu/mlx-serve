//! Muse-Glimmer-30B text trunks (`muse_glimmer` with a `text_config`, or the flat `muse_glimmer_text`): the claim and
//! the parse into `Parsed` plus Muse's own fields. MLX-free. A pack with a vision tower parses with `Caps.images` (the
//! engine loads the tower beside the plugin); a config whose geometry the legacy reading filled from defaults declines.

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
    .name = "muse_glimmer",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{},
};

pub const model_types = [_][]const u8{ "muse_glimmer", "muse_glimmer_text" };
pub const claims = sdk.claimsByType(&model_types);

pub const max_layers = 128;

/// `<|end_of_text|>` and `<|eot|>` (the template's turn terminator; `<|eom|>` is not one).
pub const terminators = [_]u32{ 200001, 200008 };

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32 = 1e-6,
    post_norm_eps: f32 = 1e-8,
    qk_scale_factor: f32 = 3.87,
    output_multiplier: f32 = 0.19611613513818404,
    /// 0 = none.
    softcap: f32 = 0,
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 0,
    sliding_window: u32 = 0,
    global: [max_layers]bool = @splat(false),
    no_rope: [max_layers]bool = @splat(false),
    tie_word_embeddings: bool = false,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "muse_glimmer: " ++ why);
}

/// Keys that, set, mean a shape this plugin does not serve.
const foreign_keys = [_][]const u8{ "rope_scaling", "num_experts", "global_head_dim", "num_kv_shared_layers", "use_bidirectional_attention", "attention_k_eq_v" };

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "muse_glimmer: config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, "muse_glimmer: config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return unsupported("no model_type");
    if (mt != .string or !(std.mem.eql(u8, mt.string, "muse_glimmer") or std.mem.eql(u8, mt.string, "muse_glimmer_text"))) return unsupported("another model_type");
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
    for (foreign_keys) |k| if (llama.set(root, k)) return unsupported("a key this plugin does not model");
    const tc = field(root, "text_config");
    if (tc != null and tc.? != .object) return unsupported("text_config");
    const cfg = if (tc) |t| t.object else root;
    for (foreign_keys) |k| if (llama.set(cfg, k)) return unsupported("a key this plugin does not model");

    var c: Config = .{
        .vocab_size = if (field(cfg, "vocab_size")) |v| int(v) orelse return unsupported("vocab_size") else 0,
        .hidden_size = int(field(cfg, "hidden_size")) orelse return unsupported("hidden_size"),
        .intermediate_size = int(field(cfg, "intermediate_size")) orelse 0,
        .num_hidden_layers = int(field(cfg, "num_hidden_layers")) orelse return unsupported("num_hidden_layers"),
        .num_attention_heads = int(field(cfg, "num_attention_heads")) orelse return unsupported("num_attention_heads"),
        .num_key_value_heads = int(field(cfg, "num_key_value_heads")) orelse return unsupported("num_key_value_heads"),
        .head_dim = int(field(cfg, "head_dim")) orelse return unsupported("head_dim"),
        .sliding_window = int(field(cfg, "sliding_window")) orelse return unsupported("sliding_window"),
    };
    if (c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
        c.num_hidden_layers == 0 or c.num_hidden_layers > max_layers or c.num_attention_heads == 0 or
        c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or
        c.head_dim == 0 or c.head_dim % 2 != 0 or c.head_dim > 512 or c.sliding_window == 0)
        return unsupported("dimensions out of range");
    // The legacy decode at hd 256 runs a fused norm+RoPE kernel the shared attention block does not carry.
    if (c.head_dim == 256) return unsupported("head_dim 256");
    inline for (.{ .{ "rms_norm_eps", &c.rms_norm_eps }, .{ "post_norm_eps", &c.post_norm_eps }, .{ "qk_scale_factor", &c.qk_scale_factor }, .{ "output_multiplier", &c.output_multiplier }, .{ "final_logit_softcapping", &c.softcap }, .{ "rope_theta", &c.rope_theta }, .{ "rope_local_base_freq", &c.rope_local_base_freq } }) |kv| {
        if (field(cfg, kv[0])) |v| kv[1].* = @floatCast(float(v) orelse return unsupported("a float field"));
    }
    if (c.qk_scale_factor <= 0 or c.output_multiplier <= 0 or c.post_norm_eps <= 0) return unsupported("a scale switched off");
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) {
        if (field(rp.object, "full_attention") != null or field(rp.object, "sliding_attention") != null) return unsupported("per-type rope parameters");
        if (field(rp.object, "rope_type")) |v| if (v != .string or !std.mem.eql(u8, v.string, "default")) return unsupported("a rope schedule");
        if (field(rp.object, "rope_theta")) |v| c.rope_theta = @floatCast(float(v) orelse return unsupported("rope_theta"));
    };
    // One theta for every roped layer unless the config names a local one.
    if (field(cfg, "rope_local_base_freq") == null) c.rope_local_base_freq = c.rope_theta;

    const lt = field(cfg, "layer_types") orelse return unsupported("no layer_types");
    if (lt != .array or lt.array.items.len < c.num_hidden_layers) return unsupported("layer_types");
    for (lt.array.items[0..c.num_hidden_layers], 0..) |item, i| {
        if (item != .string) return unsupported("layer_types");
        c.global[i] = std.mem.eql(u8, item.string, "full_attention");
    }
    if (field(cfg, "layer_rope_theta")) |v| {
        if (v != .array) return unsupported("layer_rope_theta");
        for (v.array.items, 0..) |item, i| {
            if (i >= max_layers) break;
            c.no_rope[i] = (float(item) orelse return unsupported("layer_rope_theta")) == 0;
        }
    }
    if (field(root, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };
    if (field(cfg, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };

    // A vision tower loads into the engine's map beside the plugin (`driver.loadWith`); its rows splice at `image`.
    var p: abi.Parsed = .{ .caps = .{ .batches_decode = true, .dflash_capture = true, .images = field(root, "vision_config") != null } };
    // Media rows splice at the root's image id, as the legacy read it.
    for ([_][]const u8{ "image_token_id", "image_token_index" }) |k| if (p.placeholder_ids.image == 0) if (int(field(root, k))) |id| {
        p.placeholder_ids.image = @intCast(id);
    };
    p.chat = .{ .thinking_default = .on_with_tools, .reasoning_dialect = .muse, .reasoning_round_trip = .drop_prior_turns };
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
        .model_type = abi.Str.of("muse_glimmer"),
    };
    llama.readShell(cfg, root, s) catch return unsupported("a token id or the context out of range");
    for (terminators) |t| if (std.mem.indexOfScalar(u32, s.eos[0..s.eos_count], t) == null) generation.addEos(s, t);
    c.quant = llama.readQuant(root) catch return unsupported("the quantization block");
    s.quant = c.quant;
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "muse_glimmer: out of memory");
    cfg_out.* = c;
    cfg_out.block_json = llama.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, "muse_glimmer: out of memory");
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

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

/// A trimmed copy of meta-models/Muse-Glimmer-30B's text_config.
const muse =
    \\{"model_type":"muse_glimmer","text_config":{"model_type":"muse_glimmer_text","hidden_size":6656,
    \\"intermediate_size":19968,"num_hidden_layers":8,"num_attention_heads":32,"num_key_value_heads":2,"head_dim":128,
    \\"hidden_activation":"silu","rms_norm_eps":1e-05,"post_norm_eps":1e-08,"qk_scale_factor":3.87,
    \\"output_multiplier":0.19611613513818404,"final_logit_softcapping":20.0,"vocab_size":202048,
    \\"max_position_embeddings":131072,"tie_word_embeddings":false,"bos_token_id":200000,"eos_token_id":200001,
    \\"sliding_window":2048,"layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention",
    \\"sliding_attention","sliding_attention","sliding_attention","full_attention"],
    \\"layer_rope_theta":[500000.0,500000.0,500000.0,0,500000.0,500000.0,500000.0,0],
    \\"rope_parameters":{"rope_theta":500000.0,"rope_type":"default"}},"quantization":{"group_size":64,"bits":8}}
;

test "muse_glimmer config: NoPE full layers, one theta for every roped layer, the template's terminators" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(muse, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expect(c.global[3] and c.no_rope[3] and !c.no_rope[0]);
    try testing.expectEqual(@as(f32, 500000.0), c.rope_local_base_freq);
    try testing.expectEqual(@as(f32, 20.0), c.softcap);
    try testing.expectEqualSlices(u32, &.{ 200001, 200008 }, p.shell.eos[0..p.shell.eos_count]);
    try testing.expectEqual(abi.ThinkingDefault.on_with_tools, p.chat.thinking_default);
}

test "muse_glimmer config: a pack with its vision tower parses with images in its caps" {
    const json = "{\"vision_config\":{\"model_type\":\"muse_glimmer_vision\"}," ++ muse[1..];
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(json, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(p.caps.images);
}
