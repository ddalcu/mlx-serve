//! Inkling 2's claim and parse (`inkling2_loop`, the looped Inkling 2 Mini): `text_config` as mlx-lm's `TextArgs`
//! reads it, the loop's span and pass count, and the cache layout: one entry per layer VISIT (a looped layer keeps a
//! KV cache and four short-conv windows per pass). MLX-free. A config the forward does not model is refused
//! `unsupported` by name.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const sliding_ring = @import("../../sdk/config/sliding_ring.zig");
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

pub const max_layers = 128;
pub const max_visits = 512;

/// The image tower (`vision_config`, mlx-lm's `VisionArgs`): a ViT over native-resolution patches under a 2-D rotary,
/// then a merger folding each `merge` x `merge` block of patches into one row of the text width.
pub const Vision = struct {
    hidden: u32 = 1024,
    layers: u32 = 23,
    heads: u32 = 16,
    channels: u32 = 3,
    patch: u32 = 20,
    merge: u32 = 2,
    /// The learned position table's side; a patch reads it bilinearly at its place in the image.
    grid: u32 = 32,
    proj_hidden: u32 = 4096,
    proj_bias: bool = false,
    mlp_ratio: f32 = 4.0,
    eps: f32 = 1e-5,
    theta: f32 = 10000.0,

    pub fn intermediate(v: *const Vision) u32 {
        return @intFromFloat(@as(f32, @floatFromInt(v.hidden)) * v.mlp_ratio);
    }

    /// The side in pixels of the block one row stands for.
    pub fn block(v: *const Vision) u32 {
        return v.patch * v.merge;
    }
};

/// The bundled block drafter (`mtp_config`, weights in `mtp.safetensors`): a few layers over a block of an anchor
/// token and masks, attending the trunk's taps at the newest `window - 1` positions, with an RNN Markov head.
pub const Draft = struct {
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    eps: f32 = 1e-6,
    theta: f32 = 1e6,
    window: u32,
    /// The anchor and `block - 1` masks; every mask proposes a token.
    block: u32,
    rank: u32,
    /// The trunk's hiddens the drafter reads: the stream entering these visits, the last one (`visits`) the
    /// final-normed output.
    taps: [max_taps]u32 = @splat(0),
    tap_count: u32,
    /// `mtp_config.local_layer_ids` as a mask: the sliding layers; the others attend every committed position. Null =
    /// the config names none.
    local: ?u32 = null,

    /// Whether drafter layer `i` attends every committed position (not only the newest `window - 1`).
    pub fn full(d: *const Draft, i: usize) bool {
        const m = d.local orelse return false;
        return i < 32 and (m >> @intCast(i)) & 1 == 0;
    }

    /// Bytes a slot's full layers keep per committed position: bf16 keys and values (`ContextKv.fk`, `fv`).
    pub fn fullBytesPerToken(d: *const Draft) u64 {
        var n: u64 = 0;
        for (0..d.layers) |i| n += @intFromBool(d.full(i));
        return n * d.kv_heads * d.head_dim * 2 * 2;
    }
};
pub const max_taps = 8;
/// The most layers a drafter's per-slot context cache holds.
pub const max_draft_layers = 8;

/// The DFlash2 block drafter beside the pack (`drafter/`, `incoai/Inkling-2-Mini-DFlash2`): sliding layers with grouped
/// dynamic convolutions around attention and the MLP, over a block of an anchor and masks, attending the taps through
/// `fc`; a selector then chains one path through the top `top_k` candidates of each block row.
pub const Dflash2 = struct {
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    intermediate: u32,
    eps: f32 = 1e-6,
    theta: f32 = 1e4,
    window: u32,
    /// The anchor and `block - 1` masks; every mask proposes a token.
    block: u32,
    mask_id: u32,
    /// The trunk's hiddens it reads: the stream entering these visits, the last one (`visits`) the final-normed output.
    taps: [max_taps]u32 = @splat(0),
    tap_count: u32,
    conv_kernel: u32,
    conv_group: u32,
    rank: u32,
    top_k: u32,
    /// The trunk head's soft cap, applied to the selector's scores after the edges are added.
    softcap: f32,
};
pub const dflash2_dir = "drafter";
pub const dflash2_config = dflash2_dir ++ "/config.json";

pub const Config = struct {
    vocab_size: u32,
    /// The logits the model emits: the padded vocab's real rows.
    unpadded_vocab_size: u32,
    hidden_size: u32,
    /// Layers of weights; `visits` of them run per token.
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    d_rel: u32,
    /// The relative bias's reach on a global layer; a sliding layer's is its window.
    rel_extent: u32,
    sliding_window: u32,
    /// Rows a sliding visit's cache keeps (`sliding_ring.rows`, declared as `CacheFacts.sliding_ring`); 0 = every row.
    ring: u32 = 0,
    global: [max_layers]bool = @splat(false),
    sconv_kernel: u32,
    log_n_floor: u32,
    log_alpha: f32,
    rms_norm_eps: f32 = 1e-6,
    qk_clip: f32,
    rel_clip: f32,
    /// A layer whose index divides by this opens a depth block (`n_layers_per_attn_res_block`).
    block_every: u32,
    num_experts: u32,
    top_k: u32,
    moe_intermediate_size: u32,
    gate_softcap: f32,
    up_softcap: f32,
    route_scale: f32,
    logit_softcap: f32,
    /// Layers `[core_start, core_end)` run `passes` times; one pass = no loop.
    core_start: u32 = 0,
    core_end: u32 = 0,
    passes: u32 = 1,
    gate_rank: u32 = 0,
    quant: abi.Quant = .{},
    block_json: []const u8 = "{}",
    vision: ?Vision = null,
    draft: ?Draft = null,
    /// The DFlash2 drafter beside the pack; it serves in place of the bundled one.
    dflash2: ?Dflash2 = null,

    pub fn core(c: *const Config) u32 {
        return c.core_end - c.core_start;
    }

    /// Layer applications per token.
    pub fn visits(c: *const Config) u32 {
        return c.num_hidden_layers + (c.passes - 1) * c.core();
    }

    /// The layer visit `v` runs and its pass through the loop (0 outside it).
    pub fn visit(c: *const Config, v: u32) struct { layer: u32, pass: u32 } {
        if (v < c.core_start or c.core() == 0) return .{ .layer = v, .pass = 0 };
        const looped = c.passes * c.core();
        if (v < c.core_start + looped) return .{ .layer = c.core_start + (v - c.core_start) % c.core(), .pass = (v - c.core_start) / c.core() };
        return .{ .layer = v - looped + c.core(), .pass = 0 };
    }

    /// Depth blocks opened by layers below `layer`.
    pub fn blocksBefore(c: *const Config, layer: u32) u32 {
        return (layer + c.block_every - 1) / c.block_every;
    }

    pub fn kvDim(c: *const Config) u32 {
        return c.num_key_value_heads * c.head_dim;
    }
};

/// The one-row step kernels: Metal only (as `mlx.metalKernelsAvailable`), on unless `MLX_SERVE_INKLING2_FUSED=0`. Only
/// they serve a verified or a batched visit, so without them (CUDA, the switch off) the pack loads no drafter and
/// batches no slots.
pub fn fusedOn() bool {
    return kernelsServe(@import("builtin").os.tag.isDarwin(), @import("../../env.zig").span(.MLX_SERVE_INKLING2_FUSED));
}

fn kernelsServe(metal: bool, raw: ?[]const u8) bool {
    const r = raw orelse return metal;
    return metal and !(r.len > 0 and r[0] == '0');
}

const gpa = std.heap.c_allocator;

fn modelopt(root: std.json.ObjectMap) bool {
    const qc = field(root, "quantization_config") orelse return false;
    const m = if (qc == .object) field(qc.object, "quant_method") orelse return false else return false;
    return m == .string and std.mem.eql(u8, m.string, "modelopt");
}

pub const draft_file = "mtp.safetensors";

fn readDraft(o: std.json.ObjectMap, c: *const Config) ?Draft {
    const kind = field(o, "markov_head_type") orelse return null;
    if (kind != .string or !std.mem.eql(u8, kind.string, "rnn_hidden")) return null;
    inline for (.{ "q_bias", "o_bias" }) |k| if (field(o, k)) |v| if (v != .bool or v.bool) return null;
    if ((int(field(o, "hidden_size")) orelse return null) != c.hidden_size) return null;
    var d: Draft = .{
        .layers = int(field(o, "num_hidden_layers")) orelse return null,
        .heads = int(field(o, "num_attention_heads")) orelse return null,
        .kv_heads = int(field(o, "num_key_value_heads")) orelse return null,
        .head_dim = int(field(o, "head_dim")) orelse return null,
        .intermediate = int(field(o, "intermediate_size")) orelse return null,
        .window = int(field(o, "sliding_window_size")) orelse return null,
        .block = int(field(o, "block_size")) orelse return null,
        .rank = int(field(o, "markov_rank")) orelse return null,
        .tap_count = 0,
    };
    if (field(o, "rms_norm_eps")) |v| d.eps = @floatCast(float(v) orelse return null);
    if (field(o, "rope_theta")) |v| d.theta = @floatCast(float(v) orelse return null);
    const ids = field(o, "target_layer_ids") orelse return null;
    if (ids != .array or ids.array.items.len < 2 or ids.array.items.len > max_taps) return null;
    for (ids.array.items, 0..) |item, i| {
        d.taps[i] = int(item) orelse return null;
        // Every tap but the last is a visit's input; the last is the final-normed output.
        const last = i + 1 == ids.array.items.len;
        if (if (last) d.taps[i] != c.visits() else d.taps[i] >= c.visits()) return null;
    }
    d.tap_count = @intCast(ids.array.items.len);
    if (field(o, "local_layer_ids")) |v| {
        if (v != .array) return null;
        var m: u32 = 0;
        for (v.array.items) |item| {
            const i = int(item) orelse return null;
            if (i >= 32) return null;
            m |= @as(u32, 1) << @intCast(i);
        }
        d.local = m;
    }
    if (d.layers == 0 or d.layers > 16 or d.heads == 0 or d.kv_heads == 0 or d.heads % d.kv_heads != 0 or d.head_dim == 0 or d.head_dim % 2 != 0 or
        d.intermediate == 0 or d.window < 2 or d.block < 2 or d.block > abi.max_round_tokens or d.rank == 0 or !(d.eps > 0) or !(d.theta > 1)) return null;
    return d;
}

/// `drafter/config.json` as the DFlash2 port runs it: every layer sliding, blocks attending themselves in both
/// directions, the selector's soft cap after the correction (the one placement ported), the trunk's width and taps.
pub fn readDflash2(json: []const u8, hidden: u32, visits_n: u32) ?Dflash2 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), json, .{}) catch return null;
    if (doc != .object) return null;
    const o = doc.object;
    const df = field(o, "dflash_config") orelse return null;
    if (df != .object) return null;
    const dc = df.object;
    if ((int(field(o, "hidden_size")) orelse return null) != hidden) return null;
    var d: Dflash2 = .{
        .layers = int(field(o, "num_hidden_layers")) orelse return null,
        .heads = int(field(o, "num_attention_heads")) orelse return null,
        .kv_heads = int(field(o, "num_key_value_heads")) orelse return null,
        .head_dim = int(field(o, "head_dim")) orelse return null,
        .intermediate = int(field(o, "intermediate_size")) orelse return null,
        .window = int(field(o, "sliding_window")) orelse return null,
        .block = int(field(dc, "block_size")) orelse return null,
        .mask_id = int(field(dc, "mask_token_id")) orelse return null,
        .conv_kernel = int(field(dc, "conv_kernel_size")) orelse return null,
        .conv_group = int(field(dc, "conv_group_size")) orelse return null,
        .rank = int(field(dc, "selector_rank")) orelse return null,
        .top_k = int(field(dc, "selector_top_k")) orelse return null,
        .softcap = @floatCast(float(field(dc, "final_logit_softcapping")) orelse return null),
        .tap_count = 0,
    };
    if (field(o, "rms_norm_eps")) |v| d.eps = @floatCast(float(v) orelse return null);
    if (field(o, "rope_parameters")) |rp| {
        if (rp != .object) return null;
        if (field(rp.object, "rope_theta")) |v| d.theta = @floatCast(float(v) orelse return null);
    } else if (field(o, "rope_theta")) |v| d.theta = @floatCast(float(v) orelse return null);
    // Without `is_causal` a sliding layer's block reads causally; the port reads it in both directions.
    const causal = field(o, "is_causal") orelse return null;
    if (causal != .bool or causal.bool) return null;
    const types = field(o, "layer_types") orelse return null;
    if (types != .array or types.array.items.len != d.layers) return null;
    for (types.array.items) |t| if (t != .string or !std.mem.eql(u8, t.string, "sliding_attention")) return null;
    const mode = field(dc, "selector_softcap_mode") orelse return null;
    if (mode != .string or !std.mem.eql(u8, mode.string, "after_correction")) return null;
    if (field(dc, "output_multiplier")) |v| if ((float(v) orelse return null) != 1.0) return null;
    const ids = field(dc, "target_layer_ids") orelse return null;
    if (ids != .array or ids.array.items.len < 2 or ids.array.items.len > max_taps) return null;
    for (ids.array.items, 0..) |item, i| {
        d.taps[i] = int(item) orelse return null;
        const last = i + 1 == ids.array.items.len;
        if (if (last) d.taps[i] != visits_n else d.taps[i] >= visits_n) return null;
    }
    d.tap_count = @intCast(ids.array.items.len);
    if (d.layers == 0 or d.layers > max_draft_layers or d.heads == 0 or d.kv_heads == 0 or d.heads % d.kv_heads != 0 or d.head_dim == 0 or d.head_dim % 2 != 0 or
        d.intermediate == 0 or d.window < d.block + 1 or d.block < 2 or d.block > abi.max_round_tokens or d.rank == 0 or d.top_k == 0 or d.top_k > 64 or
        d.conv_kernel == 0 or d.conv_kernel > 4 or d.conv_group == 0 or hidden % d.conv_group != 0 or
        !(d.eps > 0) or !(d.theta > 1) or !(d.softcap > 0)) return null;
    return d;
}

/// `drafter/config.json` of the pack, read through `dir`; null when it is missing or the port does not run it.
fn readDflash2File(dir: *const abi.PackDir, hidden: u32, visits_n: u32) ?Dflash2 {
    const name = abi.Str.of(dflash2_config);
    if (!dir.exists(dir.ctx, name)) return null;
    const cap: usize = @intCast(dir.file_bytes(dir.ctx, name));
    if (cap == 0 or cap > 1 << 20) return null;
    const buf = gpa.alloc(u8, cap) catch return null;
    defer gpa.free(buf);
    var len: usize = 0;
    if (!dir.read(dir.ctx, name, buf.ptr, cap, &len).isOk()) return null;
    return readDflash2(buf[0..len], hidden, visits_n);
}

fn readVision(o: std.json.ObjectMap) ?Vision {
    var v: Vision = .{};
    inline for (.{
        .{ "hidden_size", &v.hidden },            .{ "num_hidden_layers", &v.layers },   .{ "num_attention_heads", &v.heads },
        .{ "num_channels", &v.channels },         .{ "native_patch_size", &v.patch },    .{ "adapter_merge_size", &v.merge },
        .{ "posemb_grid_size", &v.grid },         .{ "projection_hidden_dim", &v.proj_hidden },
    }) |f| if (field(o, f[0])) |x| {
        f[1].* = int(x) orelse return null;
    };
    inline for (.{ .{ "mlp_ratio", &v.mlp_ratio }, .{ "layer_norm_eps", &v.eps }, .{ "rope_theta", &v.theta } }) |f| if (field(o, f[0])) |x| {
        f[1].* = @floatCast(float(x) orelse return null);
    };
    if (field(o, "projector_bias")) |x| v.proj_bias = x == .bool and x.bool;
    if (v.hidden == 0 or v.hidden > 1 << 14 or v.layers == 0 or v.layers > 128 or v.heads == 0 or v.hidden % v.heads != 0 or
        (v.hidden / v.heads) % 4 != 0 or v.channels != 3 or v.patch == 0 or v.patch > 64 or v.merge == 0 or v.merge > 8 or
        v.grid == 0 or v.grid > 1024 or v.proj_hidden == 0 or v.proj_hidden > 1 << 16 or !(v.mlp_ratio > 0) or !(v.eps > 0) or !(v.theta > 1))
        return null;
    return v;
}

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "inkling2_loop: " ++ why);
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const doc = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json.slice(), .{}) catch
        return abi.Verdict.refuse(.failed, "inkling2_loop: config.json does not parse");
    if (doc != .object) return abi.Verdict.refuse(.failed, "inkling2_loop: config.json is not an object");
    const root = doc.object;
    const mt = field(root, "model_type") orelse return unsupported("no model_type");
    if (mt != .string or !std.mem.eql(u8, mt.string, model_types[0])) return unsupported("another model_type");
    // The release names ModelOpt for its experts' NVFP4 and MXFP8, which the load reads as they are.
    if (quant_config.foreign(root) and !modelopt(root)) return unsupported("a quantization_config naming another quant_method");
    const tc = field(root, "text_config") orelse return unsupported("no text_config");
    if (tc != .object) return unsupported("text_config");
    const cfg = tc.object;

    // Undefined, not defaulted: a field a parse leaves out (`draft`, `dflash2`) is reset below with the others.
    var c: Config = undefined;
    inline for (.{
        .{ "vocab_size", &c.vocab_size },                         .{ "unpadded_vocab_size", &c.unpadded_vocab_size },
        .{ "hidden_size", &c.hidden_size },                       .{ "num_hidden_layers", &c.num_hidden_layers },
        .{ "num_attention_heads", &c.num_attention_heads },       .{ "num_key_value_heads", &c.num_key_value_heads },
        .{ "head_dim", &c.head_dim },                             .{ "d_rel", &c.d_rel },
        .{ "rel_extent", &c.rel_extent },                         .{ "sliding_window_size", &c.sliding_window },
        .{ "sconv_kernel_size", &c.sconv_kernel },                .{ "log_scaling_n_floor", &c.log_n_floor },
        .{ "n_layers_per_attn_res_block", &c.block_every },       .{ "n_routed_experts", &c.num_experts },
        .{ "num_experts_per_tok", &c.top_k },                     .{ "intermediate_size", &c.moe_intermediate_size },
    }) |f| f[1].* = int(field(cfg, f[0])) orelse return unsupported(f[0]);
    inline for (.{
        .{ "log_scaling_alpha", &c.log_alpha },       .{ "qk_l2_clip_threshold", &c.qk_clip },
        .{ "rel_l2_clip_threshold", &c.rel_clip },    .{ "mlp_gate_softcap", &c.gate_softcap },
        .{ "mlp_up_softcap", &c.up_softcap },         .{ "route_scale", &c.route_scale },
        .{ "final_logit_softcapping", &c.logit_softcap },
    }) |f| f[1].* = @floatCast(float(field(cfg, f[0])) orelse return unsupported(f[0]));
    c.rms_norm_eps = if (field(cfg, "rms_norm_eps")) |v| @floatCast(float(v) orelse return unsupported("rms_norm_eps")) else 1e-6;
    c.global = @splat(false);
    c.core_start = 0;
    c.core_end = 0;
    c.passes = 1;
    c.gate_rank = 0;
    c.quant = .{};
    c.block_json = "{}";
    c.vision = null;
    c.draft = null;
    c.dflash2 = null;

    // What the forward runs unconditionally, as mlx-lm's.
    if (field(cfg, "n_shared_experts")) |v| if ((int(v) orelse return unsupported("n_shared_experts")) != 0) return unsupported("shared experts");
    inline for (.{ "use_value_norm_no_gamma", "attn_res_normalize_values", "use_gate_bias" }) |k| {
        if (field(cfg, k)) |v| if (v != .bool or !v.bool) return unsupported(k ++ " off");
    }
    inline for (.{ "q_bias", "o_bias" }) |k| {
        if (field(cfg, k)) |v| if (v != .bool or v.bool) return unsupported(k);
    }
    inline for (.{ .{ "swa_num_attention_heads", c.num_attention_heads }, .{ "swa_num_key_value_heads", c.num_key_value_heads }, .{ "swa_head_dim", c.head_dim } }) |f| {
        if (field(cfg, f[0])) |v| if ((int(v) orelse return unsupported(f[0])) != f[1]) return unsupported("sliding-attention geometry other than the global one");
    }
    if (c.vocab_size == 0 or c.vocab_size > 1 << 22 or c.unpadded_vocab_size == 0 or c.unpadded_vocab_size > c.vocab_size or
        c.hidden_size == 0 or c.hidden_size > 1 << 16 or c.num_hidden_layers == 0 or c.num_hidden_layers > max_layers or
        c.num_attention_heads == 0 or c.num_key_value_heads == 0 or c.num_attention_heads % c.num_key_value_heads != 0 or
        c.head_dim == 0 or c.head_dim > 512 or c.d_rel == 0 or c.d_rel > 1024 or c.rel_extent == 0 or c.rel_extent > 1 << 20 or
        c.sliding_window == 0 or c.sliding_window > 1 << 20 or c.sconv_kernel < 2 or c.sconv_kernel > 64 or
        c.block_every == 0 or c.num_experts == 0 or c.num_experts > 4096 or c.top_k == 0 or c.top_k > c.num_experts or
        c.moe_intermediate_size == 0 or c.moe_intermediate_size > 1 << 16 or c.log_n_floor == 0 or
        !(c.rms_norm_eps > 0) or !(c.qk_clip > 0) or !(c.rel_clip > 0) or !(c.gate_softcap > 0) or !(c.up_softcap > 0) or
        !(c.logit_softcap > 0) or !(c.route_scale > 0))
        return unsupported("dimensions out of range");

    const local = field(cfg, "local_layer_ids") orelse return unsupported("no local_layer_ids");
    if (local != .array) return unsupported("local_layer_ids");
    @memset(c.global[0..c.num_hidden_layers], true);
    for (local.array.items) |item| {
        const i = int(item) orelse return unsupported("local_layer_ids");
        if (i < c.num_hidden_layers) c.global[i] = false;
    }

    if (field(cfg, "loop_config")) |lc| {
        if (lc != .object) return unsupported("loop_config");
        const l = lc.object;
        c.core_start = int(field(l, "core_start")) orelse return unsupported("loop_config.core_start");
        c.core_end = int(field(l, "core_end")) orelse return unsupported("loop_config.core_end");
        c.passes = int(field(l, "num_passes")) orelse return unsupported("loop_config.num_passes");
        c.gate_rank = int(field(l, "gate_rank")) orelse return unsupported("loop_config.gate_rank");
        if (field(l, "is_varloop")) |v| if (v != .bool or v.bool) return unsupported("a variable loop");
        // The pass-to-pass message reads the blocks a pass opens after its first, so the loop must open two.
        if (c.core_start == 0 or c.core_start >= c.core_end or c.core_end > c.num_hidden_layers or c.passes == 0 or c.passes > 16 or
            c.gate_rank == 0 or c.gate_rank > 4096 or c.blocksBefore(c.core_end) - c.blocksBefore(c.core_start) < 2)
            return unsupported("loop_config out of range");
    }
    if (c.visits() > max_visits) return unsupported("too many layer visits");

    if (field(root, "vision_config")) |vc| if (vc == .object) {
        c.vision = readVision(vc.object) orelse return unsupported("vision_config out of range");
    };

    // A drafter this plugin does not model leaves the pack serving serially.
    if (field(root, "mtp_config")) |mc| if (mc == .object) {
        c.draft = readDraft(mc.object, &c);
    };

    // A batched tick reads each slot's own cache (`attendSlots`): no slot pads KV to its neighbours'.
    var p: abi.Parsed = .{ .caps = .{ .moe = true, .images = c.vision != null, .batches_decode = fusedOn(), .batch_pads_kv = false, .prompt_last_row = true } };
    // A DFlash2 draft in `drafter/` serves in place of the bundled head, when the port runs it.
    if (!stub and dir.exists(dir.ctx, abi.Str.of(dflash2_dir ++ "/model.safetensors"))) c.dflash2 = readDflash2File(dir, c.hidden_size, c.visits());
    if (c.dflash2) |d| {
        p.spec.native_stages = d.layers;
        p.spec.native_block = d.block;
    } else if (c.draft) |d| if (!stub and dir.exists(dir.ctx, abi.Str.of(draft_file))) {
        p.spec.native_stages = d.layers;
        p.spec.native_block = d.block;
    };
    if (c.vision != null) p.placeholder_ids.image = int(field(root, "image_token_id")) orelse 200054;
    // The release thinks unless asked not to: its template's effort defaults to high (0.9).
    p.chat.thinking_default = .on;
    const s = &p.shell;
    s.* = .{
        .vocab_size = c.unpadded_vocab_size,
        .hidden_size = c.hidden_size,
        .intermediate_size = c.moe_intermediate_size,
        // The engine sizes caches and bills per layer applied.
        .num_hidden_layers = c.visits(),
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
    p.cache.ssm_state = .{ .entries = c.visits(), .dtype = .f32 };
    c.ring = sliding_ring.rows(c.sliding_window);
    p.cache.sliding_ring = c.ring;
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

/// One entry per layer visit: a global layer keeps full KV, a local one its window; no layer ropes.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    if (span != null) return 0;
    const c = of(p);
    const n = @min(c.visits(), cap);
    for (out[0..n], 0..) |*o, v| {
        const g = c.global[c.visit(@intCast(v)).layer];
        o.* = .{
            .kind = if (g) .full else .sliding,
            .nope = true,
            .window = if (g) 0 else c.sliding_window,
            .kv_heads = c.num_key_value_heads,
            .k_dim = c.head_dim,
            .v_dim = c.head_dim,
        };
    }
    return n;
}

/// The bytes per token a slot's drafter keeps beside the KV (`Bill.state_per_token`): its full layers' context, billed
/// wherever the head ships and the kernels are on, Apple9+ (`draftsOn`) or not.
pub fn draftStatePerToken(p: *const abi.Parsed) u64 {
    const d = of(p).draft orelse return 0;
    return if (of(p).dflash2 == null and p.spec.native_stages > 0 and fusedOn()) d.fullBytesPerToken() else 0;
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

/// mlx-community/Inkling-2-Mini-4bit's text_config at fixture width: 8 layers, the loop over layers 2..6, 3 passes.
pub const fixture_text =
    \\"text_config":{"model_max_length":4096,"hidden_size":64,"num_hidden_layers":8,"vocab_size":64,
    \\"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"d_rel":4,"rel_extent":8,"q_bias":false,
    \\"o_bias":false,"log_scaling_n_floor":4,"log_scaling_alpha":0.1,"rms_norm_eps":1e-06,
    \\"use_value_norm_no_gamma":true,"qk_l2_clip_threshold":4.0,"rel_l2_clip_threshold":2.0,
    \\"local_layer_ids":[0,2,3,4,6],"sconv_kernel_size":4,"unpadded_vocab_size":60,"final_logit_softcapping":30.0,
    \\"swa_head_dim":16,"swa_num_attention_heads":4,"swa_num_key_value_heads":2,"sliding_window_size":6,
    \\"n_layers_per_attn_res_block":2,"attn_res_normalize_values":true,"n_routed_experts":4,"num_experts_per_tok":2,
    \\"n_shared_experts":0,"intermediate_size":24,"mlp_gate_softcap":4.0,"mlp_up_softcap":25.0,"route_scale":8.0,
    \\"use_gate_bias":true,"loop_config":{"core_start":2,"core_end":6,"num_passes":3,"gate_rank":8,"is_varloop":false}}
;
pub const fixture_config = "{\"model_type\":\"inkling2_loop\",\"eos_token_id\":2," ++ fixture_text ++ "}";

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

test "inkling2 config: a looped pack parses to one cache entry per layer visit, each looped layer once per pass" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(fixture_config, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expectEqual(@as(u32, 16), c.visits());
    try testing.expectEqual(@as(u32, 16), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u32, 16), p.cache.ssm_state.entries);
    try testing.expectEqual(@as(u32, 60), p.shell.vocab_size);
    try testing.expectEqual(@as(u64, 4096), p.shell.context_cap);
    const want = [_][2]u32{ .{ 0, 0 }, .{ 1, 0 }, .{ 2, 0 }, .{ 3, 0 }, .{ 4, 0 }, .{ 5, 0 }, .{ 2, 1 }, .{ 5, 1 }, .{ 2, 2 }, .{ 5, 2 }, .{ 6, 0 }, .{ 7, 0 } };
    for ([_]u32{ 0, 1, 2, 3, 4, 5, 6, 9, 10, 13, 14, 15 }, want) |v, w| {
        const got = c.visit(v);
        try testing.expectEqual(w[0], got.layer);
        try testing.expectEqual(w[1], got.pass);
    }
    var lay: [max_visits]abi.LayerCache = undefined;
    try testing.expectEqual(@as(u32, 16), layout(&p, null, &lay, max_visits));
    // Layer 5 is global: its three visits keep full KV; layer 2's keep the window.
    for ([_]u32{ 5, 9, 13 }) |v| try testing.expect(lay[v].kind == .full and lay[v].window == 0 and lay[v].nope);
    for ([_]u32{ 2, 6, 10 }) |v| try testing.expect(lay[v].kind == .sliding and lay[v].window == 6);
}

test "inkling2 config: a vision_config declares images at the image placeholder; one out of range is refused" {
    const with = "{\"model_type\":\"inkling2_loop\",\"eos_token_id\":2,\"vision_config\":{\"native_patch_size\":20,\"adapter_merge_size\":2,\"hidden_size\":1024,\"num_attention_heads\":16}," ++ fixture_text ++ "}";
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(with, &p).isOk());
    defer freeParsed(&p);
    const v = of(&p).vision.?;
    try testing.expect(p.caps.images and p.placeholder_ids.image == 200054);
    try testing.expectEqual(@as(u32, 40), v.block());
    try testing.expectEqual(@as(u32, 4096), v.intermediate());

    var plain: abi.Parsed = .{};
    try testing.expect(parseJson(fixture_config, &plain).isOk());
    defer freeParsed(&plain);
    try testing.expect(!plain.caps.images and plain.placeholder_ids.image == 0 and of(&plain).vision == null);

    const bad = try std.mem.replaceOwned(u8, testing.allocator, with, "\"num_attention_heads\":16}", "\"num_attention_heads\":7}");
    defer testing.allocator.free(bad);
    var q: abi.Parsed = .{};
    try testing.expect(!parseJson(bad, &q).isOk() and q.private == null);
}

test "inkling2 config: a draft head beside the pack declares its stages; one this plugin does not model serves serially" {
    const mtp = "\"mtp_config\":{\"hidden_size\":64,\"num_hidden_layers\":3,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":16,\"intermediate_size\":128," ++
        "\"q_bias\":false,\"o_bias\":false,\"rms_norm_eps\":1e-06,\"rope_theta\":1000000.0,\"sliding_window_size\":6,\"block_size\":8,\"markov_rank\":8," ++
        "\"markov_head_type\":\"rnn_hidden\",\"target_layer_ids\":[2,5,10,16]},";
    const with = "{\"model_type\":\"inkling2_loop\",\"eos_token_id\":2," ++ mtp ++ fixture_text ++ "}";
    const Case = struct { json: []const u8, sidecar: bool, stages: u32, modelled: bool };
    const other = try std.mem.replaceOwned(u8, testing.allocator, with, "rnn_hidden", "low_rank");
    defer testing.allocator.free(other);
    const late = try std.mem.replaceOwned(u8, testing.allocator, with, "[2,5,10,16]", "[2,5,10,15]");
    defer testing.allocator.free(late);
    {
        const locals = try std.mem.replaceOwned(u8, testing.allocator, with, "\"block_size\"", "\"local_layer_ids\":[0,2],\"block_size\"");
        defer testing.allocator.free(locals);
        for ([_][]const u8{ with, locals }, [_]?u32{ null, 0b101 }) |json, mask| {
            const files = [_]fixture.File{ .{ .name = "config.json", .bytes = json }, .{ .name = draft_file, .bytes = "x" } };
            const mem: fixture.MemDir = .{ .files = &files };
            const dir = mem.packDir();
            var p: abi.Parsed = .{};
            try testing.expect(parse(&dir, abi.Str.of(json), .{}, null, false, &p).isOk());
            defer freeParsed(&p);
            const dc = of(&p).draft.?;
            try testing.expectEqual(mask, dc.local);
            try testing.expectEqual([3]bool{ false, mask != null, false }, [3]bool{ dc.full(0), dc.full(1), dc.full(2) });
            // One full layer: 2 KV heads x 16 x keys and values x bf16 a token.
            const per_token: u64 = if (mask != null and fusedOn()) 2 * 16 * 2 * 2 else 0;
            try testing.expectEqual(per_token, draftStatePerToken(&p));
        }
    }
    for ([_]Case{
        .{ .json = with, .sidecar = true, .stages = 3, .modelled = true },
        .{ .json = with, .sidecar = false, .stages = 0, .modelled = true },
        .{ .json = other, .sidecar = true, .stages = 0, .modelled = false },
        .{ .json = late, .sidecar = true, .stages = 0, .modelled = false },
    }) |k| {
        const files = [_]fixture.File{ .{ .name = "config.json", .bytes = k.json }, .{ .name = draft_file, .bytes = "x" } };
        const mem: fixture.MemDir = .{ .files = files[0..if (k.sidecar) 2 else 1] };
        const dir = mem.packDir();
        var p: abi.Parsed = .{};
        try testing.expect(parse(&dir, abi.Str.of(k.json), .{}, null, false, &p).isOk());
        defer freeParsed(&p);
        try testing.expectEqual(k.stages, p.spec.native_stages);
        try testing.expectEqual(@as(u32, if (k.stages > 0) 8 else 0), p.spec.native_block);
        try testing.expectEqual(k.modelled, of(&p).draft != null);
        if (of(&p).draft) |d| try testing.expect(d.tap_count == 4 and d.taps[3] == 16 and d.window == 6);
    }
}

test "inkling2 config: an unmodelled switch, another quant method and a malformed loop are refused by name; the release's block parses" {
    const Case = struct { from: []const u8, to: []const u8, why: []const u8 };
    const cases = [_]Case{
        .{ .from = "\"eos_token_id\":2,", .to = "\"eos_token_id\":2,\"quantization_config\":{\"quant_method\":\"fp8\"},", .why = "another quant_method" },
        .{ .from = "\"n_shared_experts\":0", .to = "\"n_shared_experts\":1", .why = "shared experts" },
        .{ .from = "\"is_varloop\":false", .to = "\"is_varloop\":true", .why = "a variable loop" },
        .{ .from = "\"use_value_norm_no_gamma\":true", .to = "\"use_value_norm_no_gamma\":false", .why = "use_value_norm_no_gamma off" },
        .{ .from = "\"swa_head_dim\":16", .to = "\"swa_head_dim\":8", .why = "sliding-attention geometry" },
        .{ .from = "\"core_end\":6", .to = "\"core_end\":3", .why = "loop_config out of range" },
        .{ .from = "\"core_start\":2", .to = "\"core_start\":0", .why = "loop_config out of range" },
        .{ .from = "\"num_passes\":3", .to = "\"num_passes\":3000", .why = "loop_config out of range" },
        .{ .from = "\"hidden_size\":64", .to = "\"hidden_size\":0", .why = "dimensions out of range" },
        .{ .from = "\"d_rel\":4,", .to = "", .why = "d_rel" },
        .{ .from = "inkling2_loop", .to = "inkling2", .why = "another model_type" },
    };
    for (cases) |k| {
        const json = try std.mem.replaceOwned(u8, testing.allocator, fixture_config, k.from, k.to);
        defer testing.allocator.free(json);
        try testing.expect(!std.mem.eql(u8, json, fixture_config));
        var p: abi.Parsed = .{};
        const v = parseJson(json, &p);
        try testing.expect(!v.isOk() and p.private == null);
        try testing.expect(std.mem.indexOf(u8, v.why.slice(), k.why) != null);
    }
    var p: abi.Parsed = .{};
    try testing.expect(!parseJson("{\"model_type\":\"inkling2_loop\"", &p).isOk());
    // The release's own block parses: its experts keep ModelOpt's NVFP4 and MXFP8.
    const release = try std.mem.replaceOwned(u8, testing.allocator, fixture_config, "\"eos_token_id\":2,", "\"eos_token_id\":2,\"quantization_config\":{\"quant_method\":\"modelopt\",\"quant_algo\":\"NVFP4\"},");
    defer testing.allocator.free(release);
    try testing.expect(parseJson(release, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u8, 0), p.shell.quant.bits);
}

// Bar: with the kernels off the pack batches no slots: only the kernels serve a batched or a verified visit, and they
// run on a Metal build only.
test "inkling2 config: the kernels off batch no slots" {
    var on: abi.Parsed = .{};
    try testing.expect(parseJson(fixture_config, &on).isOk());
    defer freeParsed(&on);
    try testing.expectEqual(@import("builtin").os.tag.isDarwin(), on.caps.batches_decode);
    const saved = @import("../../env.zig").override(.MLX_SERVE_INKLING2_FUSED, "0");
    defer saved.restore();
    var off: abi.Parsed = .{};
    try testing.expect(parseJson(fixture_config, &off).isOk());
    defer freeParsed(&off);
    try testing.expect(!off.caps.batches_decode);
}

// Bar: a batched tick reads each slot's own cache (`attendSlots`), so the engine's pad-waste cap never splits a group.
test "inkling2 config: a batched tick pads no KV" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(fixture_config, &p).isOk());
    defer freeParsed(&p);
    try testing.expect(!p.caps.batch_pads_kv);
}

// Bar: the step kernels serve only a Metal build, and there only while the kill switch is not 0.
test "inkling2 config: the step kernels serve a Metal build unless switched off" {
    try testing.expect(kernelsServe(true, null) and kernelsServe(true, "1"));
    try testing.expect(!kernelsServe(true, "0"));
    try testing.expect(!kernelsServe(false, null) and !kernelsServe(false, "1"));
}

/// `incoai/Inkling-2-Mini-DFlash2`'s config.json, as published.
const dflash2_release =
    \\{"architectures":["DFlash2DraftModel"],"head_dim":128,"hidden_size":2304,"intermediate_size":6912,"is_causal":false,
    \\"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
    \\"num_attention_heads":32,"num_hidden_layers":5,"num_key_value_heads":8,"rms_norm_eps":1e-06,
    \\"rope_parameters":{"rope_theta":10000.0,"rope_type":"default"},"sliding_window":2048,"vocab_size":201024,
    \\"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"final_logit_softcapping":30.0,
    \\"mask_token_id":200056,"selector_rank":256,"selector_softcap_mode":"after_correction","selector_top_k":16,
    \\"target_layer_ids":[5,23,41,48]}}
;

// Bar: the published DFlash2 draft's config reads to the geometry its weights have, its taps ending at the final norm;
// a draft this port does not run (a full-attention layer, causal blocks, another selector softcap, another width, taps
// past the visits) is refused rather than drafted wrongly.
test "inkling2 config: a DFlash2 draft reads to its geometry; one the port does not run is refused" {
    const d = readDflash2(dflash2_release, 2304, 48) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 5), d.layers);
    try testing.expect(d.heads == 32 and d.kv_heads == 8 and d.head_dim == 128 and d.intermediate == 6912);
    try testing.expect(d.window == 2048 and d.block == 8 and d.mask_id == 200056);
    try testing.expect(d.conv_kernel == 2 and d.conv_group == 16 and d.rank == 256 and d.top_k == 16);
    try testing.expectEqual(@as(f32, 30), d.softcap);
    try testing.expectEqual(@as(f32, 10000), d.theta);
    try testing.expectEqualSlices(u32, &.{ 5, 23, 41, 48 }, d.taps[0..d.tap_count]);
    const refused = [_][2][]const u8{
        .{ "\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\"", "\"sliding_attention\",\"sliding_attention\",\"full_attention\",\"sliding_attention\",\"sliding_attention\"" },
        .{ "\"is_causal\":false", "\"is_causal\":true" },
        .{ "after_correction", "before_correction" },
        .{ "\"hidden_size\":2304", "\"hidden_size\":2048" },
        .{ "[5,23,41,48]", "[5,23,41,47]" },
        .{ "[5,23,41,48]", "[5,23,49,48]" },
        .{ "\"conv_group_size\":16", "\"conv_group_size\":17" },
    };
    for (refused) |r| {
        const bad = try std.mem.replaceOwned(u8, testing.allocator, dflash2_release, r[0], r[1]);
        defer testing.allocator.free(bad);
        try testing.expect(readDflash2(bad, 2304, 48) == null);
    }
    try testing.expect(readDflash2("{}", 2304, 48) == null);
}

// Bar: a `drafter/` folder holding a DFlash2 draft this port runs declares its stages and block and serves in place of
// the bundled head; one it does not run leaves the bundled head (or serial decode) as it was, and it bills no state per
// token (every layer slides).
test "inkling2 config: a DFlash2 drafter beside the pack serves in place of the bundled head" {
    const mtp = "\"mtp_config\":{\"hidden_size\":64,\"num_hidden_layers\":3,\"num_attention_heads\":4,\"num_key_value_heads\":2,\"head_dim\":16,\"intermediate_size\":128," ++
        "\"q_bias\":false,\"o_bias\":false,\"rms_norm_eps\":1e-06,\"rope_theta\":1000000.0,\"sliding_window_size\":6,\"block_size\":8,\"markov_rank\":8," ++
        "\"markov_head_type\":\"rnn_hidden\",\"local_layer_ids\":[0,2],\"target_layer_ids\":[2,5,10,16]},";
    const with = "{\"model_type\":\"inkling2_loop\",\"eos_token_id\":2," ++ mtp ++ fixture_text ++ "}";
    const df2 =
        \\{"head_dim":16,"hidden_size":64,"intermediate_size":128,"is_causal":false,"layer_types":["sliding_attention","sliding_attention"],
        \\"num_attention_heads":4,"num_hidden_layers":2,"num_key_value_heads":2,"rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":10000.0},
        \\"sliding_window":32,"dflash_config":{"block_size":8,"conv_group_size":16,"conv_kernel_size":2,"final_logit_softcapping":30.0,
        \\"mask_token_id":50,"selector_rank":8,"selector_softcap_mode":"after_correction","selector_top_k":4,"target_layer_ids":[2,5,10,16]}}
    ;
    const unmodelled = try std.mem.replaceOwned(u8, testing.allocator, df2, "after_correction", "before_correction");
    defer testing.allocator.free(unmodelled);
    const Case = struct { draft: ?[]const u8, weights: bool, bundled: bool, stages: u32, df2: bool };
    for ([_]Case{
        .{ .draft = df2, .weights = true, .bundled = true, .stages = 2, .df2 = true },
        .{ .draft = df2, .weights = true, .bundled = false, .stages = 2, .df2 = true },
        .{ .draft = df2, .weights = false, .bundled = true, .stages = 3, .df2 = false },
        .{ .draft = unmodelled, .weights = true, .bundled = true, .stages = 3, .df2 = false },
        .{ .draft = unmodelled, .weights = true, .bundled = false, .stages = 0, .df2 = false },
        .{ .draft = null, .weights = false, .bundled = true, .stages = 3, .df2 = false },
    }) |k| {
        var files: [4]fixture.File = undefined;
        var n: usize = 0;
        files[n] = .{ .name = "config.json", .bytes = with };
        n += 1;
        if (k.bundled) {
            files[n] = .{ .name = draft_file, .bytes = "x" };
            n += 1;
        }
        if (k.draft) |j| {
            files[n] = .{ .name = dflash2_config, .bytes = j };
            n += 1;
        }
        if (k.weights) {
            files[n] = .{ .name = dflash2_dir ++ "/model.safetensors", .bytes = "x" };
            n += 1;
        }
        const mem: fixture.MemDir = .{ .files = files[0..n] };
        const dir = mem.packDir();
        var p: abi.Parsed = .{};
        try testing.expect(parse(&dir, abi.Str.of(with), .{}, null, false, &p).isOk());
        defer freeParsed(&p);
        try testing.expectEqual(k.stages, p.spec.native_stages);
        try testing.expectEqual(@as(u32, if (k.stages > 0) 8 else 0), p.spec.native_block);
        try testing.expectEqual(k.df2, of(&p).dflash2 != null);
        if (k.df2) try testing.expectEqual(@as(u64, 0), draftStatePerToken(&p));
    }
}
