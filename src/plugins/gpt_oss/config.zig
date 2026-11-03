//! OpenAI gpt-oss (`gpt_oss`): the claim and the parse into `Parsed` plus gpt-oss's own fields (layer kinds, the flat
//! YaRN block with its computed mscale, the clamped SwiGLU limit). MLX-free.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const quant_config = @import("../../sdk/config/quantization_config.zig");
const llama = @import("../llama/config.zig");
const fam = @import("../laguna/config.zig");

const field = llama.field;
const int = llama.int;
const float = llama.float;

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "gpt_oss",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/gpt-oss-20b-MXFP4-Q8"},
};

pub const model_types = [_][]const u8{"gpt_oss"};
pub const claims = sdk.claimsByType(&model_types);

pub const max_layers = 128;

/// `<|return|>` (the declared eos) and `<|call|>` (ends a tool call). Never `<|end|>`: it closes the analysis channel
/// mid-generation.
pub const terminators = [_]u32{ 200002, 200012 };

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    /// The expert width (gpt-oss has no `moe_intermediate_size`).
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32 = 1e-6,
    scale: f32 = 0,
    rope_theta: f32 = 1000000.0,
    yarn: ?fam.Yarn = null,
    sliding_window: u32 = 0,
    global: [max_layers]bool = @splat(false),
    num_experts: u32,
    top_k: u32,
    swiglu_limit: f32,
    tie_word_embeddings: bool = false,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "gpt_oss: " ++ why);
}

const foreign_keys = [_][]const u8{ "rope_parameters", "swiglu_clamp" };

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "gpt_oss: config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, "gpt_oss: config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return unsupported("no model_type");
    if (mt != .string or !std.mem.eql(u8, mt.string, "gpt_oss")) return unsupported("another model_type");
    const cfg = if (field(root, "text_config")) |tc| (if (tc == .object) tc.object else return unsupported("text_config")) else root;
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
    for (foreign_keys) |k| if (field(cfg, k) != null) return unsupported("a key this plugin does not model");

    var c: Config = .{
        .vocab_size = if (field(cfg, "vocab_size")) |v| int(v) orelse return unsupported("vocab_size") else 0,
        .hidden_size = int(field(cfg, "hidden_size")) orelse return unsupported("hidden_size"),
        .intermediate_size = int(field(cfg, "intermediate_size")) orelse 0,
        .num_hidden_layers = int(field(cfg, "num_hidden_layers")) orelse return unsupported("num_hidden_layers"),
        .num_attention_heads = int(field(cfg, "num_attention_heads")) orelse return unsupported("num_attention_heads"),
        .num_key_value_heads = int(field(cfg, "num_key_value_heads")) orelse return unsupported("num_key_value_heads"),
        .head_dim = int(field(cfg, "head_dim")) orelse return unsupported("head_dim"),
        .sliding_window = int(field(cfg, "sliding_window")) orelse return unsupported("sliding_window"),
        .num_experts = int(field(cfg, "num_local_experts")) orelse int(field(cfg, "num_experts")) orelse return unsupported("num_local_experts"),
        .top_k = int(field(cfg, "experts_per_token")) orelse int(field(cfg, "num_experts_per_tok")) orelse return unsupported("experts_per_token"),
        .swiglu_limit = @floatCast(float(field(cfg, "swiglu_limit")) orelse return unsupported("swiglu_limit")),
    };
    // The engine reads a separate expert width when one is declared.
    if (field(cfg, "moe_intermediate_size") != null) return unsupported("moe_intermediate_size");
    if (c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
        c.num_hidden_layers == 0 or c.num_hidden_layers > max_layers or c.num_attention_heads == 0 or
        c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or
        c.head_dim == 0 or c.head_dim % 2 != 0 or c.head_dim > 256 or c.sliding_window == 0 or c.num_experts == 0 or
        c.num_experts > 4096 or c.top_k == 0 or c.top_k > c.num_experts or !(c.swiglu_limit > 0))
        return unsupported("dimensions out of range");
    if (field(cfg, "rms_norm_eps")) |v| c.rms_norm_eps = @floatCast(float(v) orelse return unsupported("rms_norm_eps"));
    if (field(cfg, "rope_theta")) |v| c.rope_theta = @floatCast(float(v) orelse return unsupported("rope_theta"));
    const qpas = if (field(cfg, "query_pre_attn_scalar")) |v| int(v) orelse return unsupported("query_pre_attn_scalar") else c.head_dim;
    if (qpas == 0) return unsupported("query_pre_attn_scalar");
    c.scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qpas)));

    const lt = field(cfg, "layer_types") orelse return unsupported("no layer_types");
    if (lt != .array or lt.array.items.len < c.num_hidden_layers) return unsupported("layer_types");
    for (lt.array.items[0..c.num_hidden_layers], 0..) |item, i| {
        if (item != .string) return unsupported("layer_types");
        c.global[i] = std.mem.eql(u8, item.string, "full_attention");
    }
    // A flat YaRN block, its mscale computed (the config ships none); any other scaling is not gpt-oss's.
    if (field(cfg, "rope_scaling")) |rs| {
        if (rs != .object) return unsupported("rope_scaling");
        const rt = field(rs.object, "rope_type") orelse return unsupported("rope_scaling");
        if (rt != .string or !std.mem.eql(u8, rt.string, "yarn")) return unsupported("a rope scaling other than YaRN");
        var y: fam.Yarn = .{ .factor = 1.0, .orig_max = int(field(rs.object, "original_max_position_embeddings")) orelse return unsupported("YaRN without its window") };
        if (field(rs.object, "factor")) |v| y.factor = @floatCast(float(v) orelse return unsupported("YaRN factor"));
        if (field(rs.object, "beta_fast")) |v| y.beta_fast = @floatCast(float(v) orelse return unsupported("YaRN beta_fast"));
        if (field(rs.object, "beta_slow")) |v| y.beta_slow = @floatCast(float(v) orelse return unsupported("YaRN beta_slow"));
        if (y.orig_max == 0 or y.factor <= 0) return unsupported("the YaRN block");
        if (y.factor > 1.0) y.attention_factor = 0.1 * @log(y.factor) + 1.0;
        c.yarn = y;
    }
    if (field(cfg, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };

    var p: abi.Parsed = .{ .caps = .{ .dflash_capture = true } };
    p.chat = .{ .thinking_default = .on, .reasoning_dialect = .harmony };
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
        .model_type = abi.Str.of("gpt_oss"),
    };
    llama.readShell(cfg, root, s) catch return unsupported("a token id or the context out of range");
    if (c.yarn) |y| {
        const window: u64 = @intFromFloat(@floor(@as(f64, @floatFromInt(y.orig_max)) * @as(f64, y.factor)));
        s.context_cap = if (s.max_position_embeddings == 0) window else @min(window, s.max_position_embeddings);
    }
    for (terminators) |t| if (std.mem.indexOfScalar(u32, s.eos[0..s.eos_count], t) == null) generation.addEos(s, t);
    c.quant = readQuant(root) catch return unsupported("the quantization block");
    s.quant = c.quant;
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "gpt_oss: out of memory");
    cfg_out.* = c;
    cfg_out.block_json = llama.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, "gpt_oss: out of memory");
    };
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

/// The quantization block; a `--mlx-gguf` file's rebuilt config says `mode: gguf` (each tensor's block type rides its
/// sentinel, `sdk/quant/gguf`).
fn readQuant(cfg: std.json.ObjectMap) error{Unsupported}!abi.Quant {
    if (field(cfg, "quantization")) |q| if (q == .object) if (field(q.object, "mode")) |m| if (m == .string and std.mem.eql(u8, m.string, "gguf")) {
        return .{ .bits = 4, .group_size = 32, .mode = .gguf };
    };
    return llama.readQuant(cfg);
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

/// A trimmed copy of openai/gpt-oss-20b's config.json (MLX mxfp4 pack).
const gpt_oss_20b =
    \\{"model_type":"gpt_oss","hidden_size":2880,"intermediate_size":2880,"num_hidden_layers":4,"num_attention_heads":64,
    \\"num_key_value_heads":8,"head_dim":64,"num_local_experts":32,"experts_per_token":4,"rms_norm_eps":1e-05,
    \\"layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],
    \\"max_position_embeddings":131072,"rope_scaling":{"beta_fast":32.0,"beta_slow":1.0,"factor":32.0,
    \\"original_max_position_embeddings":4096,"rope_type":"yarn","truncate":false},"rope_theta":150000,
    \\"sliding_window":128,"swiglu_limit":7.0,"tie_word_embeddings":false,"vocab_size":201088,"eos_token_id":200002,
    \\"quantization":{"group_size":32,"bits":4,"mode":"mxfp4"}}
;

test "gpt_oss config: YaRN with the computed mscale, the experts, Harmony's terminators" {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = gpt_oss_20b }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(gpt_oss_20b), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectApproxEqAbs(@as(f32, 1.3465735902799727), c.yarn.?.attention_factor, 1e-6);
    try testing.expectEqual(@as(u32, 32), c.num_experts);
    try testing.expectEqual(@as(u32, 4), c.top_k);
    try testing.expect(!c.global[0] and c.global[1]);
    try testing.expectEqualSlices(u32, &.{ 200002, 200012 }, p.shell.eos[0..p.shell.eos_count]);
    try testing.expectEqual(@as(u64, 4096 * 32), p.shell.context_cap);
    try testing.expect(!p.caps.batches_decode);
}
