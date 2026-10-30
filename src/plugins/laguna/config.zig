//! Laguna's attention-MoE family (`laguna`, `flint`, `kolibri1`): the claim and the parse into `Parsed` plus the
//! family's own fields (per-layer kinds, heads and RoPE, YaRN on Laguna's full layers, the router). MLX-free. A config
//! the legacy reading filled from another family's defaults (no `head_dim`, `sliding_window` or `layer_types`), a
//! malformed value, or a feature key it dropped silently, is refused `unsupported`.

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
    .name = "laguna",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{},
};

pub const Kind = enum { laguna, flint, kolibri1 };

/// Laguna's own claim and parse (Flint's and Kolibri's live in their folders' config.zig).
pub const model_types = [_][]const u8{"laguna"};
pub const claims = sdk.claimsByType(&model_types);
pub const parse = Parser(.laguna).parse;

pub const max_layers = 128;

pub const Yarn = struct {
    factor: f32,
    beta_fast: f32 = 32.0,
    beta_slow: f32 = 1.0,
    orig_max: u32,
    /// 0.1·ln(factor)+1, computed as the checkpoints were trained (never the config's `attention_factor`).
    attention_factor: f32 = 1.0,
};

pub const Config = struct {
    kind: Kind,
    vocab_size: u32,
    hidden_size: u32,
    /// The dense MLP's width (`mlp_only_layers`).
    intermediate_size: u32 = 0,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32 = 1e-6,
    /// `1/sqrt(query_pre_attn_scalar)`.
    scale: f32 = 0,
    rope_theta: f32 = 1000000.0,
    rope_local_base_freq: f32 = 10000.0,
    partial_rotary_factor: f32 = 1.0,
    partial_rotary_factor_global: f32 = 1.0,
    yarn: ?Yarn = null,
    sliding_window: u32 = 0,
    global: [max_layers]bool = @splat(false),
    no_rope: [max_layers]bool = @splat(false),
    heads: [max_layers]u32 = @splat(0),
    num_experts: u32 = 0,
    top_k: u32 = 0,
    moe_intermediate_size: u32 = 0,
    route_norm: bool = true,
    route_scale: f32 = 1.0,
    tie_word_embeddings: bool = false,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",

    /// A full layer with a YaRN table rotates `head_dim·partial_global`; every other roped layer `head_dim·partial`.
    pub fn ropeDims(c: *const Config, layer: usize) u32 {
        const yarn = c.global[layer] and c.yarn != null;
        const pf = if (yarn) c.partial_rotary_factor_global else c.partial_rotary_factor;
        return @intFromFloat(@as(f32, @floatFromInt(c.head_dim)) * pf);
    }

    /// The YaRN table's rotating slice: `head_dim·partial_global`.
    pub fn yarnRotaryDim(c: *const Config) u32 {
        return @intFromFloat(@as(f32, @floatFromInt(c.head_dim)) * c.partial_rotary_factor_global);
    }
};

const gpa = std.heap.c_allocator;

/// Keys that, set, mean a shape this family does not serve (the legacy reading dropped them, or failed on them).
const foreign_keys = [_][]const u8{ "expert_quant", "rope_scaling", "num_kv_shared_layers", "global_head_dim", "use_bidirectional_attention", "linear_num_key_heads", "full_attention_interval", "hidden_size_per_layer_input", "swiglu_limit", "attention_k_eq_v" };

pub fn Parser(comptime kind: Kind) type {
    return struct {
        const name = @tagName(kind);

        fn unsupported(comptime why: []const u8) abi.Verdict {
            return abi.Verdict.refuse(.unsupported, name ++ ": " ++ why);
        }

        pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
            _ = settings_json;
            if (span != null) return unsupported("a layer span (the cluster split)");
            var arena = std.heap.ArenaAllocator.init(gpa);
            defer arena.deinit();
            const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
                return abi.Verdict.refuse(.failed, name ++ ": config.json does not parse");
            if (doc != .object) return abi.Verdict.refuse(.failed, name ++ ": config.json is not an object");
            const root = doc.object;
            const mt = field(root, "model_type") orelse return unsupported("no model_type");
            if (mt != .string or !std.mem.eql(u8, mt.string, name)) return unsupported("another model_type");
            const cfg = if (field(root, "text_config")) |tc| (if (tc == .object) tc.object else return unsupported("text_config")) else root;
            if (quant_config.foreign(root)) return unsupported("a quantization_config naming a quant_method");
            for (foreign_keys) |k| if (llama.set(cfg, k)) return unsupported("a key this plugin does not model");

            var c: Config = .{
                .kind = kind,
                .vocab_size = if (field(cfg, "vocab_size")) |v| int(v) orelse return unsupported("vocab_size") else 0,
                .hidden_size = int(field(cfg, "hidden_size")) orelse return unsupported("hidden_size"),
                .num_hidden_layers = int(field(cfg, "num_hidden_layers")) orelse return unsupported("num_hidden_layers"),
                .num_attention_heads = int(field(cfg, "num_attention_heads")) orelse return unsupported("num_attention_heads"),
                .num_key_value_heads = int(field(cfg, "num_key_value_heads")) orelse return unsupported("num_key_value_heads"),
                .head_dim = int(field(cfg, "head_dim")) orelse return unsupported("head_dim"),
                .sliding_window = int(field(cfg, "sliding_window")) orelse return unsupported("sliding_window"),
                .num_experts = int(field(cfg, "num_experts")) orelse 0,
                .moe_intermediate_size = int(field(cfg, "moe_intermediate_size")) orelse 0,
            };
            if (field(cfg, "intermediate_size")) |v| c.intermediate_size = int(v) orelse return unsupported("intermediate_size");
            c.top_k = int(field(cfg, "top_k_experts")) orelse int(field(cfg, "num_experts_per_tok")) orelse 0;
            if (c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
                c.num_hidden_layers == 0 or c.num_hidden_layers > max_layers or c.num_attention_heads == 0 or
                c.num_key_value_heads == 0 or c.head_dim == 0 or c.head_dim % 2 != 0 or c.head_dim > 512 or
                c.sliding_window == 0 or c.num_experts == 0 or c.num_experts > 4096 or c.top_k == 0 or
                c.top_k > c.num_experts)
                return unsupported("dimensions out of range");
            if (field(cfg, "rms_norm_eps")) |v| c.rms_norm_eps = @floatCast(float(v) orelse return unsupported("rms_norm_eps"));

            const lt = field(cfg, "layer_types") orelse return unsupported("no layer_types");
            if (lt != .array or lt.array.items.len < c.num_hidden_layers) return unsupported("layer_types");
            for (lt.array.items[0..c.num_hidden_layers], 0..) |item, i| {
                if (item != .string) return unsupported("layer_types");
                c.global[i] = std.mem.eql(u8, item.string, "full_attention");
            }

            // The engine's reading order: the flat theta, then a full-layer theta, then a flat one in
            // `rope_parameters` last; the flat fraction rotates every layer a YaRN table does not.
            if (field(cfg, "rope_theta")) |v| c.rope_theta = @floatCast(float(v) orelse return unsupported("rope_theta"));
            if (field(cfg, "rope_local_base_freq")) |v| c.rope_local_base_freq = @floatCast(float(v) orelse return unsupported("rope_local_base_freq"));
            if (field(cfg, "rope_parameters")) |rp| if (rp == .object) {
                if (field(rp.object, "rope_type")) |rt| if (rt != .string or !std.mem.eql(u8, rt.string, "default")) return unsupported("a flat rope schedule");
                if (field(rp.object, "partial_rotary_factor")) |v| c.partial_rotary_factor = @floatCast(float(v) orelse return unsupported("partial_rotary_factor"));
                if (field(rp.object, "full_attention")) |fa| if (fa == .object) {
                    if (field(fa.object, "rope_theta")) |v| c.rope_theta = @floatCast(float(v) orelse return unsupported("rope_theta"));
                    if (field(fa.object, "partial_rotary_factor")) |v| c.partial_rotary_factor_global = @floatCast(float(v) orelse return unsupported("partial_rotary_factor"));
                    if (field(fa.object, "rope_type")) |rt| {
                        if (rt != .string) return unsupported("a full-layer rope type");
                        if (std.mem.eql(u8, rt.string, "yarn")) {
                            if (kind != .laguna) return unsupported("YaRN");
                            c.yarn = readYarn(fa.object) orelse return unsupported("the full layers' YaRN block");
                        } else if (!std.mem.eql(u8, rt.string, "default")) return unsupported("a full-layer rope type");
                    }
                };
                if (field(rp.object, "sliding_attention")) |sa| if (sa == .object) {
                    if (field(sa.object, "rope_theta")) |v| c.rope_local_base_freq = @floatCast(float(v) orelse return unsupported("rope_theta"));
                };
                if (field(rp.object, "rope_theta")) |v| c.rope_theta = @floatCast(float(v) orelse return unsupported("rope_theta"));
            };

            var qpas: u32 = c.head_dim;
            switch (kind) {
                .laguna => {
                    // Without it the legacy reading rotated with another family's theta and no YaRN.
                    if (field(cfg, "rope_parameters") == null) return unsupported("no rope_parameters");
                    if (field(cfg, "query_pre_attn_scalar")) |v| qpas = int(v) orelse return unsupported("query_pre_attn_scalar");
                    if (field(cfg, "norm_topk_prob")) |v| if (v == .bool) {
                        c.route_norm = v.bool;
                    };
                    if (field(cfg, "moe_routed_scaling_factor")) |v| c.route_scale = @floatCast(float(v) orelse return unsupported("moe_routed_scaling_factor"));
                    if (field(cfg, "moe_router_logit_softcapping")) |v| if ((float(v) orelse return unsupported("router softcapping")) > 0) return unsupported("router logit softcapping");
                    // `gating` may be a bool (the published config's `true`); a string names the granularity.
                    if (field(cfg, "gating")) |v| if (v == .string and !(std.mem.eql(u8, v.string, "per-head") or std.mem.eql(u8, v.string, "per_head"))) return unsupported("gating other than per-head");
                    if (field(cfg, "num_attention_heads_per_layer")) |v| {
                        if (v != .array) return unsupported("num_attention_heads_per_layer");
                        for (v.array.items, 0..) |item, i| {
                            if (i >= max_layers) break;
                            if (item == .integer) c.heads[i] = int(item) orelse return unsupported("num_attention_heads_per_layer");
                        }
                    }
                },
                .flint => {
                    c.rope_local_base_freq = c.rope_theta;
                    if (field(cfg, "no_rope_layer_indices")) |v| {
                        if (v != .array) return unsupported("no_rope_layer_indices");
                        for (v.array.items) |item| {
                            const i = int(item) orelse return unsupported("no_rope_layer_indices");
                            if (i < max_layers) c.no_rope[i] = true;
                        }
                    }
                },
                .kolibri1 => {
                    c.rope_local_base_freq = c.rope_theta;
                    c.route_norm = false;
                    if (field(cfg, "norm_topk_prob")) |v| if (v == .bool and v.bool) return unsupported("norm_topk_prob");
                    for (0..c.num_hidden_layers) |i| c.no_rope[i] = c.global[i];
                },
            }
            if (qpas == 0) return unsupported("query_pre_attn_scalar");
            c.scale = 1.0 / @sqrt(@as(f32, @floatFromInt(qpas)));
            for (0..c.num_hidden_layers) |i| {
                if (c.heads[i] == 0) c.heads[i] = c.num_attention_heads;
                if (c.heads[i] % c.num_key_value_heads != 0) return unsupported("heads per layer");
            }
            if (field(cfg, "tie_word_embeddings") orelse field(root, "tie_word_embeddings")) |v| if (v == .bool) {
                c.tie_word_embeddings = v.bool;
            };

            var p: abi.Parsed = .{ .caps = .{ .batches_decode = kind == .flint, .dflash_capture = true } };
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
                .model_type = abi.Str.of(name),
            };
            llama.readShell(cfg, root, s) catch return unsupported("a token id or the context out of range");
            // YaRN resolves only `original_max_position_embeddings × factor` positions (the engine's `contextCap`).
            if (c.yarn) |y| {
                const window: u64 = @intFromFloat(@floor(@as(f64, @floatFromInt(y.orig_max)) * @as(f64, y.factor)));
                s.context_cap = if (s.max_position_embeddings == 0) window else @min(window, s.max_position_embeddings);
            }
            c.quant = llama.readQuant(root) catch return unsupported("the quantization block");
            s.quant = c.quant;
            if (!stub) generation.apply(dir, s);

            const cfg_out = gpa.create(Config) catch return abi.Verdict.refuse(.failed, name ++ ": out of memory");
            cfg_out.* = c;
            cfg_out.block_json = llama.blockJson(c.quant) catch {
                gpa.destroy(cfg_out);
                return abi.Verdict.refuse(.failed, name ++ ": out of memory");
            };
            p.private = cfg_out;
            out.* = p;
            return .ok;
        }
    };
}

/// The full layers' YaRN block as the engine reads it for Laguna; null where it cannot be reproduced.
fn readYarn(fa: std.json.ObjectMap) ?Yarn {
    var y: Yarn = .{ .factor = 1.0, .orig_max = int(field(fa, "original_max_position_embeddings")) orelse return null };
    if (field(fa, "factor")) |v| y.factor = @floatCast(float(v) orelse return null);
    if (field(fa, "beta_fast")) |v| y.beta_fast = @floatCast(float(v) orelse return null);
    if (field(fa, "beta_slow")) |v| y.beta_slow = @floatCast(float(v) orelse return null);
    if (field(fa, "truncate")) |v| if (v != .bool or !v.bool) return null;
    if (y.orig_max == 0 or y.factor <= 0) return null;
    if (y.factor > 1.0) y.attention_factor = 0.1 * @log(y.factor) + 1.0;
    return y;
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

fn parseJson(comptime kind: Kind, json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return Parser(kind).parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

const laguna_json =
    \\{"model_type":"laguna","hidden_size":128,"intermediate_size":256,"num_hidden_layers":4,"num_attention_heads":4,
    \\"num_key_value_heads":1,"head_dim":128,"rms_norm_eps":1e-6,"vocab_size":256,"max_position_embeddings":1048576,
    \\"gating":"per-head","sliding_window":4,"num_experts":8,"num_experts_per_tok":2,"moe_intermediate_size":64,
    \\"num_attention_heads_per_layer":[2,4,4,2],"moe_routed_scaling_factor":2.5,
    \\"layer_types":["full_attention","sliding_attention","sliding_attention","full_attention"],
    \\"rope_parameters":{"full_attention":{"rope_theta":500000.0,"rope_type":"yarn","factor":32.0,
    \\"original_max_position_embeddings":8192,"beta_slow":1.0,"beta_fast":32.0,"partial_rotary_factor":0.5},
    \\"sliding_attention":{"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":1.0}}}
;

test "laguna config: per-layer heads, YaRN on full layers with the computed mscale, and the context it resolves" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(.laguna, laguna_json, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 2), c.heads[0]);
    try testing.expectEqual(@as(u32, 4), c.heads[1]);
    try testing.expectEqual(@as(f32, 0.1 * @log(@as(f32, 32.0)) + 1.0), c.yarn.?.attention_factor);
    try testing.expectEqual(@as(u32, 64), c.ropeDims(0));
    try testing.expectEqual(@as(u32, 128), c.ropeDims(1));
    try testing.expectEqual(@as(f32, 2.5), c.route_scale);
    try testing.expectEqual(@as(u64, 8192 * 32), p.shell.context_cap);
    try testing.expect(!p.caps.batches_decode);
}

test "laguna config: router softcapping and per-channel gating are refused unsupported" {
    inline for (.{ .{ "\"gating\":\"per-head\"", "\"gating\":\"per-channel\"" }, .{ "\"gating\":\"per-head\"", "\"gating\":\"per-head\",\"moe_router_logit_softcapping\":30.0" } }) |swap| {
        const json = comptime blk: {
            @setEvalBranchQuota(100_000);
            const at = std.mem.indexOf(u8, laguna_json, swap[0]).?;
            break :blk laguna_json[0..at] ++ swap[1] ++ laguna_json[at + swap[0].len ..];
        };
        var p: abi.Parsed = .{};
        try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(.laguna, json, &p).code);
    }
}

test "laguna config: the published Laguna config parses (gating as a bool, YaRN window as its context)" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(.laguna, @embedFile("../../fixtures/model-configs/laguna.json"), &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 512), c.sliding_window);
    try testing.expect(c.global[0] and !c.global[1]);
    try testing.expectEqual(@as(u64, 4096 * 64), p.shell.context_cap);
}
