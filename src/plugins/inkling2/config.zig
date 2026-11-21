//! Inkling 2's claim and parse: `text_config` as mlx-lm's `TextArgs` reads it, the visit table of the looped core
//! (prelude, core x passes, tail), and the shell, caches and layout the engine reads. Everything that counts layers
//! counts visits: one KV slot and one recurrent entry (the four short-conv windows) per visit. MLX-free. A config the
//! forward does not model is refused `unsupported` by name.

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
    .name = "inkling2",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{},
};

pub const model_types = [_][]const u8{"inkling2_loop"};
pub const claims = sdk.claimsByType(&model_types);

pub const max_layers = 64;
/// The engine reads at most 128 layout entries.
pub const max_visits = 128;

pub const Config = struct {
    hidden: u32,
    /// Physical layers.
    layers: u32,
    /// One geometry: the sliding layers' `swa_*` must equal the global ones.
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    d_rel: u32,
    window: u32,
    /// The global layers' relative extent; a sliding layer's is `window`.
    rel_extent: u32,
    log_floor: u32,
    log_alpha: f32,
    qk_clip: f32,
    rel_clip: f32,
    eps: f32,
    kernel: u32,
    /// `n_layers_per_attn_res_block`.
    block: u32,
    experts: u32,
    top_k: u32,
    inter: u32,
    gate_cap: f32,
    up_cap: f32,
    route_scale: f32,
    final_softcap: f32,
    vocab: u32,
    unpadded_vocab: u32,
    global: [max_layers]bool = @splat(false),
    /// The looped core `[core_start, core_end)`, run `passes` times; empty with `passes` 1 without a `loop_config`.
    core_start: u32 = 0,
    core_end: u32 = 0,
    passes: u32 = 1,
    gate_rank: u32 = 0,
    n_visits: u32 = 0,
    /// Visit -> physical layer.
    visits: [max_visits]u8 = @splat(0),
    quant: abi.Quant = .{},
    block_json: []u8 = &.{},

    fn pushVisit(c: *Config, i: usize) void {
        c.visits[c.n_visits] = @intCast(i);
        c.n_visits += 1;
    }

    pub fn visitLayer(c: *const Config, v: u32) u32 {
        return c.visits[v];
    }

    /// The core pass visit `v` runs in; 0 outside the core.
    pub fn passOf(c: *const Config, v: u32) u32 {
        const len = c.core_end - c.core_start;
        if (len == 0 or v < c.core_start or v >= c.core_start + c.passes * len) return 0;
        return (v - c.core_start) / len;
    }

    pub fn inCore(c: *const Config, i: u32) bool {
        return c.core_start <= i and i < c.core_end;
    }

    /// Layer `i` opens an attention-residual block.
    pub fn boundary(c: *const Config, i: u32) bool {
        return i % c.block == 0;
    }

    /// The blocks the prelude completes before the core starts.
    pub fn nPreludeBlocks(c: *const Config) u32 {
        return std.math.divCeil(u32, c.core_start, c.block) catch unreachable;
    }

    /// Columns of a visit's conv state: k, v, then the attention and MLP outputs.
    pub fn convWidth(c: *const Config) u32 {
        return 2 * c.kv_heads * c.head_dim + 2 * c.hidden;
    }
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "inkling2_loop: " ++ why);
}

fn missing(comptime key: []const u8) abi.Verdict {
    return unsupported("missing or invalid " ++ key);
}

fn real(v: ?std.json.Value) ?f32 {
    return @floatCast(float(v) orelse return null);
}

/// Flags mlx-lm hard-wires: present at another value, the forward would compute a different model.
const fixed_flags = .{
    .{ "use_gate_bias", true },
    .{ "attn_res_normalize_values", true },
    .{ "use_value_norm_no_gamma", true },
    .{ "q_bias", false },
    .{ "o_bias", false },
};

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (a pipeline split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "inkling2_loop: config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, "inkling2_loop: config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return unsupported("a model_type other than inkling2_loop");
    if (mt != .string or !std.mem.eql(u8, mt.string, model_types[0])) return unsupported("a model_type other than inkling2_loop");
    if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method (the ModelOpt source)");
    const tc = field(root, "text_config") orelse return unsupported("no text_config");
    if (tc != .object) return unsupported("no text_config");
    const cfg = tc.object;

    var c: Config = .{
        .hidden = int(field(cfg, "hidden_size")) orelse return missing("hidden_size"),
        .layers = int(field(cfg, "num_hidden_layers")) orelse return missing("num_hidden_layers"),
        .heads = int(field(cfg, "num_attention_heads")) orelse return missing("num_attention_heads"),
        .kv_heads = int(field(cfg, "num_key_value_heads")) orelse return missing("num_key_value_heads"),
        .head_dim = int(field(cfg, "head_dim")) orelse return missing("head_dim"),
        .d_rel = int(field(cfg, "d_rel")) orelse return missing("d_rel"),
        .window = int(field(cfg, "sliding_window_size")) orelse return missing("sliding_window_size"),
        .rel_extent = int(field(cfg, "rel_extent")) orelse return missing("rel_extent"),
        .log_floor = int(field(cfg, "log_scaling_n_floor")) orelse return missing("log_scaling_n_floor"),
        .log_alpha = real(field(cfg, "log_scaling_alpha")) orelse return missing("log_scaling_alpha"),
        .qk_clip = real(field(cfg, "qk_l2_clip_threshold")) orelse return missing("qk_l2_clip_threshold"),
        .rel_clip = real(field(cfg, "rel_l2_clip_threshold")) orelse return missing("rel_l2_clip_threshold"),
        .eps = real(field(cfg, "rms_norm_eps")) orelse return missing("rms_norm_eps"),
        .kernel = int(field(cfg, "sconv_kernel_size")) orelse return missing("sconv_kernel_size"),
        .block = int(field(cfg, "n_layers_per_attn_res_block")) orelse return missing("n_layers_per_attn_res_block"),
        .experts = int(field(cfg, "n_routed_experts")) orelse return missing("n_routed_experts"),
        .top_k = int(field(cfg, "num_experts_per_tok")) orelse return missing("num_experts_per_tok"),
        .inter = int(field(cfg, "intermediate_size")) orelse return missing("intermediate_size"),
        .gate_cap = real(field(cfg, "mlp_gate_softcap")) orelse return missing("mlp_gate_softcap"),
        .up_cap = real(field(cfg, "mlp_up_softcap")) orelse return missing("mlp_up_softcap"),
        .route_scale = real(field(cfg, "route_scale")) orelse return missing("route_scale"),
        .final_softcap = real(field(cfg, "final_logit_softcapping")) orelse return missing("final_logit_softcapping"),
        .vocab = int(field(cfg, "vocab_size")) orelse return missing("vocab_size"),
        .unpadded_vocab = int(field(cfg, "unpadded_vocab_size")) orelse return missing("unpadded_vocab_size"),
    };
    const local = field(cfg, "local_layer_ids") orelse return missing("local_layer_ids");
    if (local != .array) return missing("local_layer_ids");
    inline for (.{ .{ "swa_num_attention_heads", c.heads }, .{ "swa_num_key_value_heads", c.kv_heads }, .{ "swa_head_dim", c.head_dim } }) |f| {
        if ((int(field(cfg, f[0])) orelse return missing(f[0])) != f[1]) return unsupported("a sliding geometry other than the global one: " ++ f[0]);
    }
    inline for (fixed_flags) |f| if (field(cfg, f[0])) |v| {
        if (v != .bool or v.bool != f[1]) return unsupported(f[0] ++ (if (f[1]) " false" else " true"));
    };
    if (field(cfg, "n_shared_experts")) |v| if ((int(v) orelse 1) != 0) return unsupported("n_shared_experts above 0");
    // The SDK router renormalizes the top-k weights only when k > 1; mlx-lm always does.
    if (c.top_k < 2) return unsupported("num_experts_per_tok below 2");
    if (c.log_floor == 0) return unsupported("log_scaling_n_floor 0");
    if (c.layers > max_layers) return unsupported("more than 64 layers");

    if (field(cfg, "loop_config")) |lv| {
        if (lv != .object) return missing("loop_config");
        const lc = lv.object;
        if (field(lc, "is_varloop")) |v| if (v != .bool or v.bool) return unsupported("loop_config.is_varloop true");
        if (field(lc, "core_start")) |v| c.core_start = int(v) orelse return missing("loop_config.core_start");
        if (field(lc, "core_end")) |v| c.core_end = int(v) orelse return missing("loop_config.core_end");
        c.passes = int(field(lc, "num_passes")) orelse return missing("loop_config.num_passes");
        if (c.passes == 0) return unsupported("loop_config.num_passes 0");
        if (c.core_start > c.core_end or c.core_end > c.layers) return unsupported("a loop core outside the layers");
        if (c.core_end > c.core_start) c.gate_rank = int(field(lc, "gate_rank")) orelse return missing("loop_config.gate_rank");
    }
    const core_len = c.core_end - c.core_start;
    if (@as(u64, c.layers) + @as(u64, c.passes - 1) * core_len > max_visits) return unsupported("more than 128 visits");

    if (c.hidden == 0 or c.hidden > 1 << 16 or c.layers == 0 or c.heads == 0 or c.kv_heads == 0 or
        c.heads % c.kv_heads != 0 or c.head_dim == 0 or c.head_dim > 512 or c.d_rel == 0 or c.window == 0 or
        c.rel_extent == 0 or c.kernel < 2 or c.kernel > 64 or c.block == 0 or c.experts == 0 or c.experts > 4096 or
        c.top_k > c.experts or c.inter == 0 or c.vocab == 0 or c.vocab > 1 << 22 or c.unpadded_vocab == 0 or
        c.unpadded_vocab > c.vocab or c.eps <= 0 or c.gate_cap <= 0 or c.up_cap <= 0 or c.final_softcap <= 0 or
        (core_len > 0 and c.gate_rank == 0))
        return unsupported("dimensions out of range");

    @memset(c.global[0..c.layers], true);
    for (local.array.items) |item| {
        const i = int(item) orelse return missing("local_layer_ids");
        if (i < c.layers) c.global[i] = false;
    }
    for (0..c.core_start) |i| c.pushVisit(i);
    for (0..c.passes) |_| for (c.core_start..c.core_end) |i| c.pushVisit(i);
    for (c.core_end..c.layers) |i| c.pushVisit(i);

    var p: abi.Parsed = .{ .caps = .{ .moe = true } };
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.unpadded_vocab,
        .hidden_size = c.hidden,
        .intermediate_size = c.inter,
        .num_hidden_layers = c.n_visits,
        .num_attention_heads = c.heads,
        .num_key_value_heads = c.kv_heads,
        .head_dim = c.head_dim,
        .num_experts_per_tok = c.top_k,
        .moe_intermediate_size = c.inter,
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
    p.cache.ssm_state = .{ .entries = c.n_visits, .dtype = .f32 };
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "inkling2_loop: out of memory");
    cfg_out.* = c;
    cfg_out.block_json = llama.blockJson(c.quant) catch {
        gpa.destroy(cfg_out);
        return abi.Verdict.refuse(.failed, "inkling2_loop: out of memory");
    };
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

/// One entry per visit: a global layer's visit keeps full KV, a sliding one its window; no layer ropes.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = @min(c.n_visits, cap);
    for (out[0..n], c.visits[0..n]) |*o, i| o.* = .{
        .kind = if (c.global[i]) .full else .sliding,
        .nope = true,
        .window = if (c.global[i]) 0 else c.window,
        .kv_heads = c.kv_heads,
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

fn parseJson(json: []const u8, span: ?*const abi.LayerSpan, out: *abi.Parsed) abi.Verdict {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, span, true, out);
}

/// The plugin fixture's config: 8 layers, a core of 4 run 3 times (16 visits), globals 1, 3 and 7.
const tiny =
    \\{"model_type":"inkling2_loop","eos_token_id":2,"text_config":{"hidden_size":32,"num_hidden_layers":8,
    \\"vocab_size":128,"unpadded_vocab_size":120,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
    \\"swa_num_attention_heads":4,"swa_num_key_value_heads":2,"swa_head_dim":8,"sliding_window_size":8,
    \\"local_layer_ids":[0,2,4,5,6],"d_rel":4,"rel_extent":6,"log_scaling_n_floor":4,"log_scaling_alpha":0.1,
    \\"rms_norm_eps":1e-06,"qk_l2_clip_threshold":2.0,"rel_l2_clip_threshold":1.0,"sconv_kernel_size":4,
    \\"n_layers_per_attn_res_block":2,"n_routed_experts":4,"num_experts_per_tok":2,"intermediate_size":32,
    \\"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":8.0,"final_logit_softcapping":30.0,
    \\"n_shared_experts":0,"use_gate_bias":true,"attn_res_normalize_values":true,"use_value_norm_no_gamma":true,
    \\"q_bias":false,"o_bias":false,
    \\"loop_config":{"core_start":2,"core_end":6,"num_passes":3,"gate_rank":4,"is_varloop":false}}}
;

test "inkling2 config: the tiny and released configs parse to their visit tables, layouts and shells" {
    {
        var p: abi.Parsed = .{};
        const v = parseJson(tiny, null, &p);
        try testing.expectEqualStrings("", v.why.slice());
        defer freeParsed(&p);
        const c = of(&p);
        try testing.expectEqual(@as(u32, 16), c.n_visits);
        try testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7 }, c.visits[0..16]);
        try testing.expectEqual(@as(u32, 2), c.passOf(14 - 4));
        try testing.expectEqual(@as(u32, 0), c.passOf(14));
        try testing.expectEqual(@as(u32, 1), c.nPreludeBlocks());
        try testing.expectEqual(@as(u32, 6), c.rel_extent);
        try testing.expectEqual(@as(u32, 16), p.shell.num_hidden_layers);
        try testing.expectEqual(@as(u32, 120), p.shell.vocab_size);
        try testing.expectEqual(@as(u32, 16), p.cache.ssm_state.entries);
        try testing.expectEqual(abi.DType.f32, p.cache.ssm_state.dtype);
        try testing.expectEqual(@as(u32, 0), p.cache.sliding_ring);
        try testing.expect(p.caps.moe and p.caps.prefix_cache and !p.caps.exact_resumed and !p.caps.batches_decode);
        var lay: [max_visits]abi.LayerCache = undefined;
        try testing.expectEqual(@as(u32, 16), layout(&p, null, &lay, lay.len));
        for (lay[0..16], "SFSFSSSFSSSFSSSF") |l, k| {
            try testing.expectEqual(if (k == 'F') abi.LayerKind.full else abi.LayerKind.sliding, l.kind);
            try testing.expectEqual(@as(u32, if (k == 'F') 0 else 8), l.window);
            try testing.expect(l.nope and l.kv_heads == 2 and l.k_dim == 8 and l.v_dim == 8);
        }
    }
    const released = .{
        .{ @embedFile("../../fixtures/model-configs/inkling2_loop-mlx-community-4bit.json"), abi.Quant{ .bits = 4, .mode = .affine, .group_size = 64 } },
        .{ @embedFile("fixtures/mlx-8bit-config.json"), abi.Quant{ .bits = 4, .mode = .nvfp4, .group_size = 16 } },
    };
    inline for (released) |r| {
        var p: abi.Parsed = .{};
        const v = parseJson(r[0], null, &p);
        try testing.expectEqualStrings("", v.why.slice());
        defer freeParsed(&p);
        const c = of(&p);
        try testing.expectEqual(@as(u32, 48), c.n_visits);
        const core: [12]u8 = .{ 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 };
        for (0..3) |pass| try testing.expectEqualSlices(u8, &core, c.visits[6 + 12 * pass ..][0..12]);
        try testing.expectEqual(@as(u32, 2), c.passOf(41));
        try testing.expectEqual(@as(u32, 0), c.passOf(42));
        try testing.expectEqual(@as(u32, 6656), c.convWidth());
        try testing.expectEqual(@as(u32, 48), p.shell.num_hidden_layers);
        try testing.expectEqual(@as(u32, 200058), p.shell.vocab_size);
        try testing.expectEqual(@as(u64, 1048576), p.shell.context_cap);
        try testing.expectEqual(@as(u32, 48), p.cache.ssm_state.entries);
        try testing.expectEqual(r[1].bits, p.shell.quant.bits);
        try testing.expectEqual(r[1].mode, p.shell.quant.mode);
        try testing.expectEqual(r[1].group_size, p.shell.quant.group_size);
        var lay: [max_visits]abi.LayerCache = undefined;
        try testing.expectEqual(@as(u32, 48), layout(&p, null, &lay, lay.len));
        for (lay[0..48], 0..) |l, vi| {
            const global = vi % 6 == 5;
            try testing.expectEqual(if (global) abi.LayerKind.full else abi.LayerKind.sliding, l.kind);
            try testing.expectEqual(@as(u32, if (global) 0 else 1024), l.window);
        }
    }
}

test "inkling2 config: every unsupported shape is refused by name" {
    @setEvalBranchQuota(1_000_000);
    const rows = .{
        .{ "\"is_varloop\":false", "\"is_varloop\":true", "loop_config.is_varloop true" },
        .{ "\"use_gate_bias\":true", "\"use_gate_bias\":false", "use_gate_bias false" },
        .{ "\"n_shared_experts\":0", "\"n_shared_experts\":1", "n_shared_experts above 0" },
        .{ "\"attn_res_normalize_values\":true", "\"attn_res_normalize_values\":false", "attn_res_normalize_values false" },
        .{ "\"use_value_norm_no_gamma\":true", "\"use_value_norm_no_gamma\":false", "use_value_norm_no_gamma false" },
        .{ "\"q_bias\":false", "\"q_bias\":true", "q_bias true" },
        .{ "\"o_bias\":false", "\"o_bias\":true", "o_bias true" },
        .{ "\"swa_num_key_value_heads\":2", "\"swa_num_key_value_heads\":1", "a sliding geometry other than the global one: swa_num_key_value_heads" },
        .{ "\"swa_head_dim\":8", "\"swa_head_dim\":16", "a sliding geometry other than the global one: swa_head_dim" },
        .{ "\"swa_num_attention_heads\":4", "\"swa_num_attention_heads\":8", "a sliding geometry other than the global one: swa_num_attention_heads" },
        .{ "\"num_experts_per_tok\":2", "\"num_experts_per_tok\":1", "num_experts_per_tok below 2" },
        .{ "\"log_scaling_n_floor\":4", "\"log_scaling_n_floor\":0", "log_scaling_n_floor 0" },
        .{ "\"num_hidden_layers\":8", "\"num_hidden_layers\":65", "more than 64 layers" },
        .{ "\"num_passes\":3", "\"num_passes\":32", "more than 128 visits" },
        .{ "\"core_end\":6", "\"core_end\":9", "a loop core outside the layers" },
        .{ "\"num_passes\":3", "\"num_passes\":0", "loop_config.num_passes 0" },
        .{ "\"text_config\"", "\"text_cfg\"", "no text_config" },
        .{ "\"rel_extent\"", "\"rel_extent_\"", "missing or invalid rel_extent" },
        .{ "\"model_type\":\"inkling2_loop\"", "\"model_type\":\"inkling2\"", "a model_type other than inkling2_loop" },
        .{ "\"eos_token_id\":2", "\"eos_token_id\":2,\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"NVFP4\"}", "a quantization_config naming a quant_method (the ModelOpt source)" },
    };
    inline for (rows) |row| {
        const json = comptime blk: {
            const at = std.mem.indexOf(u8, tiny, row[0]).?;
            break :blk tiny[0..at] ++ row[1] ++ tiny[at + row[0].len ..];
        };
        var p: abi.Parsed = .{};
        const v = parseJson(json, null, &p);
        defer freeParsed(&p);
        try testing.expectEqualStrings("inkling2_loop: " ++ row[2], v.why.slice());
        try testing.expectEqual(abi.Verdict.Code.unsupported, v.code);
    }
    var p: abi.Parsed = .{};
    const span: abi.LayerSpan = .{ .first = 0, .count = 4 };
    const v = parseJson(tiny, &span, &p);
    try testing.expectEqualStrings("inkling2_loop: a layer span (a pipeline split)", v.why.slice());
    try testing.expectEqual(abi.Verdict.Code.unsupported, v.code);
}
