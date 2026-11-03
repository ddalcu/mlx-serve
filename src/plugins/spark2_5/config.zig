//! XHToken Spark-X2.5 packs (`spark2_5`): the claim and the parse into `Parsed` plus Spark's own fields (per-layer
//! sliding/full kinds, per-type RoPE, the window). MLX-free. A config whose geometry the legacy reading fills from
//! defaults (no `head_dim`, no `sliding_window`, no `layer_types`) is refused `unsupported`.

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
    .name = "spark2_5",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"abenzerps/Spark-X2.5-4B-MLX-4bit"},
};

pub const model_types = [_][]const u8{"spark2_5"};

pub const claims = sdk.claimsByType(&model_types);

/// Every layer's blocks serve batched rows (`plugin.zig` checks it against the base).
pub const batches_decode = true;

pub const max_layers = 128;

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32 = 1e-6,
    /// `1/sqrt(query_pre_attn_scalar)`, the scalar defaulting to `head_dim`.
    scale: f32 = 0,
    /// Full layers' theta and rotary fraction; sliding layers rotate the whole head at `rope_local_base_freq`.
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    partial_rotary_factor_global: f32 = 1.0,
    sliding_window: u32 = 0,
    /// Per layer: full attention (true) or sliding.
    global: [max_layers]bool = @splat(false),
    tie_word_embeddings: bool = false,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",

    pub fn ropeDims(c: *const Config, global: bool) u32 {
        if (!global or c.partial_rotary_factor_global >= 1.0) return c.head_dim;
        return @intFromFloat(@as(f32, @floatFromInt(c.head_dim)) * c.partial_rotary_factor_global);
    }
};

const gpa = std.heap.c_allocator;

const unsupported = struct {
    fn v(comptime why: []const u8) abi.Verdict {
        return abi.Verdict.refuse(.unsupported, "spark2_5: " ++ why);
    }
}.v;

/// Keys that, set, mean a shape this plugin does not serve (the legacy reading dropped them, or failed on them).
const foreign_keys = [_][]const u8{ "num_experts", "num_local_experts", "expert_quant", "rope_scaling", "num_kv_shared_layers", "global_head_dim" };

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "spark2_5: config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, "spark2_5: config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return unsupported("no model_type");
    if (mt != .string or !std.mem.eql(u8, mt.string, "spark2_5")) return unsupported("another model_type");
    const cfg = if (field(root, "text_config")) |tc| (if (tc == .object) tc.object else return unsupported("text_config")) else root;
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
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
    if (field(cfg, "rms_norm_eps")) |v| c.rms_norm_eps = @floatCast(float(v) orelse return unsupported("rms_norm_eps"));
    const qpas = if (field(cfg, "query_pre_attn_scalar")) |v| int(v) orelse return unsupported("query_pre_attn_scalar") else c.head_dim;
    if (qpas == 0) return unsupported("query_pre_attn_scalar");
    c.scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qpas)));

    const lt = field(cfg, "layer_types") orelse return unsupported("no layer_types");
    if (lt != .array or lt.array.items.len < c.num_hidden_layers) return unsupported("layer_types");
    for (lt.array.items[0..c.num_hidden_layers], 0..) |item, i| {
        if (item != .string) return unsupported("layer_types");
        c.global[i] = std.mem.eql(u8, item.string, "full_attention");
    }

    // The engine's reading order: the flat theta, a full-layer theta, then a flat theta in `rope_parameters` last.
    if (field(cfg, "rope_theta")) |v| c.rope_theta = @floatCast(float(v) orelse return unsupported("rope_theta"));
    if (field(cfg, "rope_local_base_freq")) |v| c.rope_local_base_freq = @floatCast(float(v) orelse return unsupported("rope_local_base_freq"));
    if (field(cfg, "rope_parameters")) |rp| if (rp == .object) {
        // A flat rotary fraction or schedule would reach the sliding layers too (the engine's generic reading).
        if (field(rp.object, "partial_rotary_factor") != null) return unsupported("a flat rotary fraction");
        if (field(rp.object, "rope_type")) |rt| if (rt != .string or !std.mem.eql(u8, rt.string, "default")) return unsupported("a flat rope schedule");
        if (field(rp.object, "full_attention")) |fa| if (fa == .object) {
            if (field(fa.object, "rope_theta")) |v| c.rope_theta = @floatCast(float(v) orelse return unsupported("rope_theta"));
            if (field(fa.object, "partial_rotary_factor")) |v| c.partial_rotary_factor_global = @floatCast(float(v) orelse return unsupported("partial_rotary_factor"));
            if (field(fa.object, "rope_type")) |v| if (v != .string or !std.mem.eql(u8, v.string, "default")) return unsupported("a full-layer rope type");
        };
        if (field(rp.object, "sliding_attention")) |sa| if (sa == .object) {
            if (field(sa.object, "rope_theta")) |v| c.rope_local_base_freq = @floatCast(float(v) orelse return unsupported("rope_theta"));
        };
        if (field(rp.object, "rope_theta")) |v| c.rope_theta = @floatCast(float(v) orelse return unsupported("rope_theta"));
    };
    if (field(cfg, "tie_word_embeddings") orelse field(root, "tie_word_embeddings")) |v| if (v == .bool) {
        c.tie_word_embeddings = v.bool;
    };

    var p: abi.Parsed = .{ .caps = .{ .batches_decode = batches_decode, .dflash_capture = true } };
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
        .model_type = abi.Str.of("spark2_5"),
    };
    llama.readShell(cfg, root, s) catch return unsupported("a token id or the context out of range");
    c.quant = llama.readQuant(root) catch return unsupported("the quantization block");
    s.quant = c.quant;
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "spark2_5: out of memory");
    cfg_out.* = c;
    cfg_out.block_json = llama.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, "spark2_5: out of memory");
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

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

pub const spark4b =
    \\{"model_type":"spark2_5","hidden_size":2560,"intermediate_size":9728,"num_hidden_layers":4,
    \\ "num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,"vocab_size":151936,"rms_norm_eps":1e-6,
    \\ "sliding_window":512,"layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],
    \\ "rope_parameters":{"full_attention":{"rope_theta":5000000.0,"partial_rotary_factor":0.25},
    \\ "sliding_attention":{"rope_theta":10000.0}},"tie_word_embeddings":true,"max_position_embeddings":131072,
    \\ "quantization":{"group_size":64,"bits":4}}
;

test "spark2_5 config: per-type rope, the window and the layer kinds" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(spark4b, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(f32, 5000000.0), c.rope_theta);
    try testing.expectEqual(@as(f32, 10000.0), c.rope_local_base_freq);
    try testing.expectEqual(@as(u32, 64), c.ropeDims(true));
    try testing.expectEqual(@as(u32, 256), c.ropeDims(false));
    try testing.expect(!c.global[0] and c.global[3]);
    try testing.expectEqual(@as(u32, 512), c.sliding_window);
}

test "spark2_5 config: a config missing the window or the layer kinds is refused unsupported" {
    inline for (.{ "\"sliding_window\":512,", "\"head_dim\":256," }) |cut| {
        const json = comptime blk: {
            @setEvalBranchQuota(100_000);
            const at = std.mem.indexOf(u8, spark4b, cut).?;
            break :blk spark4b[0..at] ++ spark4b[at + cut.len ..];
        };
        var p: abi.Parsed = .{};
        try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(json, &p).code);
    }
}
