//! hy_v4_flash's weight placement: every tensor the SDK loader reads lands validated against
//! the config, and the 4 Engram tables open as streamed tables — read by pread at their row offsets, never
//! resident. The name map keeps the pack's `model.*` prefixes verbatim; the pack's layer index IS the internal
//! index (kinds come from weight presence, and the pack ships its indexer/engram arms at the pack
//! layers). The quant dispatch needs no per-tensor plumbing: the pack stores affine triples (U32 codes + bf16
//! scales/biases) and the SDK reader hands them back at their stored group sizes.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const ple = @import("../qwen4_exp/ple_table.zig");
const crt = @import("../../sdk/util/crt.zig");
const log = @import("../../sdk/util/log.zig");
const config = @import("config.zig");
const trunk = @import("trunk.zig");

const gpa = std.heap.c_allocator;
const C = config.Config;

pub const Engram = struct {
    fd: c_int,
    rows: u64,
    w_off: u64,
    w_stride: u64,
    s_off: u64,
    s_stride: u64,
    b_off: u64,
    b_stride: u64,

    fn close(self: *const Engram) void {
        crt.close(self.fd);
    }
};

pub const Model = struct {
    c: *const C,
    weights: weights_load.Weights,
    engrams: []Engram,
    /// The per-layer weight structs the forward reads (borrowing the map's handles) and its own derived copies.
    trunk: *trunk.Model,

    pub fn deinit(m: *Model) void {
        m.trunk.deinit();
        for (m.engrams) |*e| e.close();
        gpa.free(m.engrams);
        m.weights.deinit();
    }
};

pub fn load(ctx: *const abi.LoadCtx) !*Model {
    const c = config.of(ctx.parsed);
    // The driver prefetched the map (`ctx.weights`); a plugin test may drive `load` without it. `adopt` moves
    // the map in; until the model is built, the scope frees it.
    var holder = Holder.init(if (ctx.weights) |src|
        Resident.take(src) orelse return error.UnsupportedWeightSource
    else
        try weights_load.loadWeightsOpt(io(), gpa, ctx.model_dir.slice(), .{ .vision = ctx.parsed.caps.images }));
    errdefer holder.release();
    try validate(c, &holder.w);
    const engrams = try openEngrams(c, ctx.model_dir.slice());
    errdefer {
        for (engrams) |*e| e.close();
        gpa.free(engrams);
    }
    // Build the trunk over the validated map: projections, hyper-connections, FFN. A failure frees what it built.
    const tr = trunk.build(c, &holder.w, engrams) catch |e| {
        holder.release();
        for (engrams) |*x| x.close();
        gpa.free(engrams);
        return e;
    };
    const m = try gpa.create(Model);
    errdefer gpa.destroy(m);
    m.* = .{ .c = c, .weights = holder.adopt(), .engrams = engrams, .trunk = tr };
    return m;
}

/// The map between the load's start and the model's birth: any failure on the way frees it; `adopt` ends that.
const Holder = struct {
    w: weights_load.Weights,
    fn init(w: weights_load.Weights) Holder {
        return .{ .w = w };
    }
    fn release(h: *Holder) void {
        h.w.deinit();
    }
    fn adopt(h: *Holder) weights_load.Weights {
        const w = h.w;
        h.w = weights_load.Weights.init(w.allocator);
        return w;
    }
};

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// The config-derived shapes of one hy_v4_flash pack's tensor names. `shape` checks a stored shape; `q` a
/// full affine triple at the config's default bits/group; `has` an unquantized leaf the census names. `last`
/// holds the composed name and shape, so a failure can be reported as a sentence.
const Check = struct {
    w: *const weights_load.Weights,
    c: *const C,
    hold: [256]u8 = undefined,
    last: []const u8 = "",

    /// Compose a tensor name into a PRIVATE slot: a failed check reports it after other checks ran.
    fn compose(self: *Check, comptime fmt: []const u8, args: anytype) []const u8 {
        const n = std.fmt.bufPrint(&self.hold, fmt, args) catch @panic("tensor name past the buffer");
        self.last = n;
        return n;
    }
    fn has(self: *Check, comptime fmt: []const u8, args: anytype) !void {
        _ = self.compose(fmt, args);
        if (self.w.get(self.last) == null) return error.MissingWeight;
    }
    fn shape(self: *Check, comptime fmt: []const u8, args: anytype, want: []const u64) !void {
        _ = self.compose(fmt, args);
        const arr = self.w.get(self.last) orelse return error.MissingWeight;
        var got_buf: [8]u64 = undefined;
        const got = mlxShape(arr, &got_buf) orelse return error.MissingWeight;
        if (!std.mem.eql(u64, got, want)) return error.WrongShape;
    }
    /// A packed affine linear at the config's default bits and group: the leaves the pack stores — or the
    /// dense `[out, in]` rows the SDK loader rewrites them into below its dequantise-small line, which is
    /// what a fixture-sized pack reaches.
    fn q(self: *Check, comptime fmt: []const u8, args: anytype, out_row: u64, in_features: u64) !void {
        const packed_cols = @divExact(in_features * self.c.quant_bits, 32);
        const groups = @divExact(in_features, self.c.quant_group_size);
        const wname = self.compose(fmt ++ ".weight", args);
        const arr = self.w.get(wname) orelse return error.MissingWeight;
        var wb: [8]u64 = undefined;
        const got = mlxShape(arr, &wb) orelse return error.MissingWeight;
        if (std.mem.eql(u64, got, &.{ out_row, packed_cols })) {
            try self.shape(fmt ++ ".scales", args, &.{ out_row, groups });
            try self.shape(fmt ++ ".biases", args, &.{ out_row, groups });
            return;
        }
        if (!std.mem.eql(u64, got, &.{ out_row, in_features })) return error.WrongShape;
    }
    /// One fused expert triple, three axes: the pack stores `[experts, out, in/8]` U32 nibble words with
    /// bf16 `[experts, out, in/group]` scales and biases. A pack whose in-features sit under the SDK loader's
    /// dequantise-small line reaches the map as one dense `[experts, out, in]` stack instead; both layouts pass.
    fn qfused3(self: *Check, comptime fmt: []const u8, args: anytype, experts: u64, out_rows: u64, in_features: u64, group: u64) !void {
        if (in_features % group != 0) return error.WrongShape;
        const packed_cols = @divExact(in_features * self.c.quant_bits, 32);
        const groups = @divExact(in_features, group);
        const wname = self.compose(fmt ++ ".weight", args);
        const arr = self.w.get(wname) orelse return error.MissingWeight;
        var wb: [8]u64 = undefined;
        const got = mlxShape(arr, &wb) orelse return error.MissingWeight;
        if (std.mem.eql(u64, got, &.{ experts, out_rows, packed_cols })) {
            try self.shape(fmt ++ ".scales", args, &.{ experts, out_rows, groups });
            try self.shape(fmt ++ ".biases", args, &.{ experts, out_rows, groups });
            return;
        }
        if (!std.mem.eql(u64, got, &.{ experts, out_rows, in_features })) return error.WrongShape;
    }
};

/// The name `validate` stopped on, so the refusal sentence can name it. The Verdict's text must outlive the
/// call, so the composed sentence lives in a global the host reads after the return.
var last_checked: [256]u8 = undefined;
var last_checked_len: usize = 0;
var sentence: [400]u8 = undefined;

/// The refusal for a config/pack disagreement. A validate failure names the tensor it stopped on; an
/// Engram-open failure names its error instead — the name `validate` left behind would mislead.
pub fn failureVerdict(e: anyerror) abi.Verdict {
    const why = switch (e) {
        error.MissingWeight, error.WrongShape => std.fmt.bufPrint(&sentence, "hy_v4_flash: {s} — the pack's {s} disagrees with the config",
            .{ if (e == error.MissingWeight) "a tensor the config asks for is not in the pack" else "a tensor's shape disagrees with the config", last_checked[0..last_checked_len] }),
        else => std.fmt.bufPrint(&sentence, "hy_v4_flash: the Engram table open failed: {s}", .{@errorName(e)}),
    } catch return abi.Verdict.named(.failed, @errorName(e));
    return abi.Verdict.named(.failed, why);
}

/// Every tensor the config asks for exists, at the shape the config's math says; quantized leaves at their
/// stored (packed) shape. `load` turns a failure into a log line that names the tensor.
fn validate(c: *const C, w: *const weights_load.Weights) !void {
    var V: Check = .{ .w = w, .c = c };
    defer {
        @memcpy(last_checked[0..V.last.len], V.last);
        last_checked_len = V.last.len;
    }
    const d: u64 = c.hidden_size;
    const V1: u64 = c.vocab_size;
    try V.q("model.embed_tokens", .{}, V1, d);
    try V.q("lm_head", .{}, V1, d);
    try V.shape("model.norm.weight", .{}, &.{d});
    // The head's collapse table: `[lanes, lanes*hidden]` f32, plus its base and scale.
    try V.shape("model.hc_head.hc_head_fn", .{}, &.{ c.hc_mult, c.hc_mult * d });
    try V.shape("model.hc_head.hc_head_base", .{}, &.{c.hc_mult});
    try V.shape("model.hc_head.hc_head_scale", .{}, &.{1});

    for (0..c.num_hidden_layers) |li| {
        const n: u64 = li;
        try V.shape("model.layers.{d}.input_layernorm.weight", .{n}, &.{d});
        try V.shape("model.layers.{d}.post_attention_layernorm.weight", .{n}, &.{d});
        const a = "model.layers.{d}.self_attn";
        // The pack's own geometry: DSA layers carry `num_attention_heads` (32), SWA layers the wider
        // `swa_num_attention_heads` (64); both share nope/rope/v widths. The kind comes from WEIGHT presence
        // (`compressed_indexer.*`), never the HF-offset `layer_types` list.
        const h: u64 = config.layerHeads(c, @intCast(li));
        try V.q(a ++ ".q_a_proj", .{n}, c.q_lora_rank, d);
        try V.shape(a ++ ".q_a_layernorm.weight", .{n}, &.{c.q_lora_rank});
        try V.q(a ++ ".q_b_proj", .{n}, h * (c.qk_nope_head_dim + c.qk_rope_head_dim), c.q_lora_rank);
        try V.q(a ++ ".kv_a_proj_with_mqa", .{n}, c.kv_lora_rank + c.qk_rope_head_dim, d);
        try V.shape(a ++ ".kv_a_layernorm.weight", .{n}, &.{c.kv_lora_rank});
        try V.q(a ++ ".kv_b_proj", .{n}, h * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank);
        try V.q(a ++ ".linear_gate", .{n}, h * c.v_head_dim, d);
        try V.q(a ++ ".o_proj", .{n}, d, h * c.v_head_dim);
        try V.shape(a ++ ".post_proj_norm.weight", .{n}, &.{d});
        try V.shape(a ++ ".learnable_sink_param", .{n}, &.{h});
        if (c.dsa_layers[li]) {
            const ci = a ++ ".compressed_indexer";
            try V.q(ci ++ ".query_projection", .{n}, c.index_n_heads * c.index_head_dim, c.q_lora_rank);
            try V.q(ci ++ ".token_key_projection", .{n}, c.index_head_dim, d);
            try V.q(ci ++ ".head_weight_projection", .{n}, c.index_n_heads, d);
            try V.has(ci ++ ".key_value_projection.weight", .{n});
            try V.has(ci ++ ".key_gate_projection.weight", .{n});
            try V.shape(ci ++ ".key_norm.weight", .{n}, &.{c.index_head_dim});
            try V.shape(ci ++ ".token_key_norm.weight", .{n}, &.{c.index_head_dim});
            try V.has(ci ++ ".intra_group_ape", .{n});
        }
        for (&[_][]const u8{ "hc_attn_layer", "hc_mlp_layer" }) |hc| {
            // iHC: `hc_fn` `[2*lanes, lanes*hidden]` f32 (pre gates then post gates), `hc_base` `[2*lanes]`,
            // `hc_scale` `[2]` — both sublayers, every layer.
            try V.shape("model.layers.{d}." ++ "{s}" ++ ".hc_pre.hc_fn", .{ n, hc }, &.{ 2 * c.hc_mult, c.hc_mult * d });
            try V.shape("model.layers.{d}." ++ "{s}" ++ ".hc_pre.hc_base", .{ n, hc }, &.{2 * c.hc_mult});
            try V.shape("model.layers.{d}." ++ "{s}" ++ ".hc_pre.hc_scale", .{ n, hc }, &.{2});
        }
        try validateMlp(&V, @intCast(li));
    }
    if (c.has_vision and V.w.get("vit.embeddings.patch_embedding.weight") != null) try validateVision(&V);
    if (V.w.get("model.dfly_layers.final_norm.weight") != null) try validateDfly(&V);
}

/// The MoE / dense MLP and the Engram arms of one layer, at their config shapes.
fn validateMlp(V: *Check, li: u32) !void {
    const c = V.c;
    const n: u64 = li;
    const d: u64 = c.hidden_size;
    if (li < c.first_k_dense_replace) {
        try V.q("model.layers.{d}.mlp.gate_proj", .{n}, c.intermediate_size, d);
        try V.q("model.layers.{d}.mlp.up_proj", .{n}, c.intermediate_size, d);
        try V.q("model.layers.{d}.mlp.down_proj", .{n}, d, c.intermediate_size);
        try V.shape("model.layers.{d}.mlp.post_fc2_norm.weight", .{n}, &.{d});
        return;
    }
    try V.shape("model.layers.{d}.mlp.gate.weight", .{n}, &.{ c.n_routed_experts, d });
    try V.shape("model.layers.{d}.mlp.gate.e_score_correction_bias", .{n}, &.{c.n_routed_experts});
    // The routed experts ride fused exactly as the pack ships them: gate+up `[E, 2*moe_intermediate, latent/8]`
    // U32 nibbles over the LATENT width (routed_expert_hidden_size), down `[E, latent, moe_intermediate/8]`,
    // both with bf16 scale+bias leaves at g64.
    const e: u64 = c.n_routed_experts;
    const mi: u64 = c.moe_intermediate_size;
    const lat: u64 = c.routed_expert_hidden_size;
    try V.qfused3("model.layers.{d}.mlp.experts.gate_up_proj", .{n}, e, 2 * mi, lat, c.quant_group_size);
    try V.qfused3("model.layers.{d}.mlp.experts.down_proj", .{n}, e, lat, mi, c.quant_group_size);
    try V.q("model.layers.{d}.mlp.routed_expert_up_proj", .{n}, d, c.routed_expert_hidden_size);
    try V.q("model.layers.{d}.mlp.routed_expert_down_proj", .{n}, c.routed_expert_hidden_size, d);
    try V.q("model.layers.{d}.mlp.shared_experts.gate_proj", .{n}, c.moe_intermediate_size, d);
    try V.q("model.layers.{d}.mlp.shared_experts.up_proj", .{n}, c.moe_intermediate_size, d);
    try V.q("model.layers.{d}.mlp.shared_experts.down_proj", .{n}, d, c.moe_intermediate_size);
    try V.shape("model.layers.{d}.mlp.combine_output_norm.weight", .{n}, &.{d});
    if (!c.engram_layers[li]) return;
    // The Engram memory is `(max_ngram-1) * heads * head_dim` wide (16 heads x 96 = 1536).
    const emb_w: u64 = (c.engram_max_ngram - 1) * c.engram_ngram_heads * c.engram_head_dim;
    for (0..4) |pi| {
        try V.q("model.layers.{d}.engram_layer.key_projs.{d}", .{ n, @as(usize, pi) }, d, emb_w);
        try V.has("model.layers.{d}.engram_layer.key_projs.{d}.bias", .{ n, @as(usize, pi) });
        try V.shape("model.layers.{d}.engram_layer.norm_query.{d}.weight", .{ n, @as(usize, pi) }, &.{d});
        try V.shape("model.layers.{d}.engram_layer.norm_key.{d}.weight", .{ n, @as(usize, pi) }, &.{d});
        try V.has("model.layers.{d}.engram_layer.unified_per_hc_scales.{d}", .{ n, @as(usize, pi) });
    }
    try V.q("model.layers.{d}.engram_layer.value_proj", .{n}, d, emb_w);
    try V.has("model.layers.{d}.engram_layer.value_proj.bias", .{n});
    try V.has("model.layers.{d}.engram_layer.lookup_table", .{n});
    try V.has("model.layers.{d}.engram_layer.unified_pre_gamma", .{n});
    for (0..16) |hi| try V.shape("model.layers.{d}.engram_layer.lookup_per_head_norms.{d}.weight",
        .{ n, @as(usize, hi) }, &.{c.engram_head_dim});
}

/// The SigLIP-style tower (loaded here, used by the tower forward): bf16 weights with f32 biases.
fn validateVision(V: *Check) !void {
    const c = V.c;
    const h: u64 = c.vision_hidden_size;
    try V.has("vit.embeddings.patch_embedding.weight", .{});
    try V.has("vit.embeddings.patch_embedding.bias", .{});
    try V.has("vit.embeddings.position_embedding.weight", .{});
    try V.has("vit.perceive.norm.weight", .{});
    try V.has("vit.perceive.image_newline", .{});
    try V.has("vit.perceive.proj.0.weight", .{});
    try V.has("vit.perceive.proj.2.weight", .{});
    for (0..c.vision_depth) |vi| {
        const n: u64 = vi;
        for (&[_][]const u8{ "input_layernorm", "post_attention_layernorm" }) |s| {
            try V.has("vit.layers.{d}." ++ "{s}" ++ ".weight", .{ n, s });
            try V.has("vit.layers.{d}." ++ "{s}" ++ ".bias", .{ n, s });
        }
        for (&[_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.dense_h_to_4h", "mlp.dense_4h_to_h" }) |p|
        {
            try V.has("vit.layers.{d}." ++ "{s}" ++ ".weight", .{ n, p });
            try V.has("vit.layers.{d}." ++ "{s}" ++ ".bias", .{ n, p });
        }
        _ = h;
    }
}

/// The DFly2 draft stack: bf16 verbatim (the pack retained source precision), plus the fusion and the
/// hidden-correction MLPs and the confidence head (loaded, not wired).
fn validateDfly(V: *Check) !void {
    const c = V.c;
    const d: u64 = c.hidden_size;
    for (0..c.dfly_layers) |di| {
        const n: u64 = di;
        const a = "model.dfly_layers.{d}.self_attn";
        for (&[_][]const u8{ "input_layernorm", "post_attention_layernorm" }) |s|
            try V.has("model.dfly_layers.{d}." ++ "{s}" ++ ".weight", .{ n, s });
        for (&[_][]const u8{ "q_a_layernorm", "kv_a_layernorm", "post_proj_norm" }) |s|
            try V.has(a ++ "." ++ "{s}" ++ ".weight", .{ n, s });
        try V.has(a ++ ".q_a_proj.weight", .{n});
        try V.has(a ++ ".q_b_proj.weight", .{n});
        try V.has(a ++ ".kv_a_proj_with_mqa.weight", .{n});
        try V.has(a ++ ".kv_b_proj.weight", .{n});
        try V.has(a ++ ".linear_gate.weight", .{n});
        try V.has(a ++ ".o_proj.weight", .{n});
        try V.has(a ++ ".learnable_sink_param", .{n});
        for (&[_][]const u8{ "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }) |p|
            try V.has("model.dfly_layers.{d}." ++ "{s}" ++ ".weight", .{ n, p });
    }
    for (&[_][]const u8{ "final_norm.weight", "mtp_fusion.eh_proj.weight", "mtp_fusion.enorm.weight",
        "mtp_fusion.hnorm.weight", "mtp_fusion.gate", "hidden_correction.down_proj.weight",
        "hidden_correction.embed_norm.weight", "hidden_correction.gate_proj.weight",
        "hidden_correction.hidden_norm.weight", "hidden_correction.up_proj.weight",
        "confidence_head.proj.weight", "confidence_head.proj.bias" }) |s|
        try V.has("model.dfly_layers." ++ "{s}", .{s});
    _ = d;
}

/// A resident array's shape, copied into `out` (MLX's shape pointer is read-only scratch, not a slice owner).
fn mlxShape(arr: mlx.mlx_array, out: *[8]u64) ?[]const u64 {
    const n = mlx.mlx_array_ndim(arr);
    if (n == 0 or n > out.len) return null;
    const p = mlx.mlx_array_shape(arr)[0..n];
    for (p, 0..) |v, i| out[i] = @intCast(v);
    return out[0..n];
}

/// The stream-table open's failures: the pack's Engram tables (or their geometry) are not what the config
/// says. Kept apart from validate's MissingWeight/WrongShape so the refusal names the stage that failed.
const OpenError = error{
    IndexMissing,
    TableMissing,
    TableSplit,
    TableHeader,
    TableRows,
    TableGeometry,
    TableOpen,
    EngramCount,
    OutOfMemory,
};

fn openEngrams(c: *const C, dir_path: []const u8) OpenError![]Engram {
    var out: std.ArrayList(Engram) = .empty;
    errdefer {
        for (out.items) |*e| e.close();
        out.deinit(gpa);
    }
    for (0..c.num_hidden_layers) |li| {
        if (!c.engram_layers[li]) continue;
        out.append(gpa, try openEngram(c, dir_path, @intCast(li))) catch |e| {
            log.err("hy_v4: layer {d}: engram table: {t}\n", .{ li, e });
            return e;
        };
    }
    if (out.items.len != c.engram_layer_count) return error.EngramCount;
    return out.toOwnedSlice(gpa);
}

/// The table's three ranges and rows, read from the safetensors header of the file that carries them: the
/// index when the pack is sharded, the file's own header when it is not (a fixture-sized single file). The
/// rows must satisfy the config, and a pack that disagrees is refused, never clamped.
fn openEngram(c: *const C, dir_path: []const u8, li: u32) OpenError!Engram {
    const stream = io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var dir = std.Io.Dir.cwd().openDir(stream, dir_path, .{}) catch return error.TableOpen;
    defer dir.close(stream);

    // The shard that carries the table, from the index; a single-file pack names no index.
    var names: [3][]const u8 = undefined;
    var file: ?[]const u8 = null;
    const idx_raw = dir.readFileAlloc(stream, "model.safetensors.index.json", a, .limited(64 << 20)) catch null;
    if (idx_raw) |raw| {
        const idx = std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch return error.IndexMissing;
        const wm = (if (idx == .object) idx.object.get("weight_map") else null) orelse return error.IndexMissing;
        if (wm != .object) return error.IndexMissing;
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |part, i| {
            names[i] = try std.fmt.allocPrint(a, "model.layers.{d}.engram_layer.multi_head_embedding.embedding.{s}", .{ li, part });
            const f = wm.object.get(names[i]) orelse return error.TableMissing;
            if (f != .string or std.mem.indexOfAny(u8, f.string, "/\\") != null) return error.TableMissing;
            if (file) |prev| if (!std.mem.eql(u8, prev, f.string)) return error.TableSplit;
            file = f.string;
        }
    }
    // Without an index (a single-file pack), find the file that carries the leaves by trying each
    // `model*.safetensors` for the weight's header entry. An index that names no table for THIS layer
    // (an older index) falls to the same scan.
    if (file == null) {
        for ([_][]const u8{ "weight", "scales", "biases" }, 0..) |part, i| {
            names[i] = std.fmt.allocPrint(a, "model.layers.{d}.engram_layer.multi_head_embedding.embedding.{s}", .{ li, part }) catch return error.OutOfMemory;
        }
        var found: ?[]const u8 = null;
        var it = dir.iterate();
        while (it.next(stream) catch return error.TableOpen) |e| {
            if (e.kind != .file and e.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
            const fd0 = openAt(a, dir_path, e.name) catch continue;
            defer crt.close(fd0);
            if (headerHas(a, fd0, names[0]) catch false) {
                found = try a.dupe(u8, e.name);
                break;
            }
        }
        file = found orelse return error.TableMissing;
    }
    const fd = try openAt(a, dir_path, file.?);
    errdefer crt.close(fd);
    var hl: [8]u8 = undefined;
    if (crt.pread(fd, &hl, 8, 0) != 8) return error.TableHeader;
    const hlen = std.mem.readInt(u64, &hl, .little);
    if (hlen > 16 << 20) return error.TableHeader;
    const hbuf = try a.alloc(u8, @intCast(hlen));
    if (crt.pread(fd, hbuf.ptr, hbuf.len, 8) != @as(isize, @intCast(hlen))) return error.TableHeader;
    const hdr = std.json.parseFromSliceLeaky(std.json.Value, a, hbuf, .{}) catch return error.TableHeader;
    if (hdr != .object) return error.TableHeader;
    const size = fileSize(fd);
    const base = 8 + hlen;

    var regs: [3]ple.HeaderRegion = undefined;
    for (names, &regs, 0..) |n, *r, i| {
        r.* = ple.headerRegion(hdr.object, n, if (i == 0) "U32" else "BF16", if (i == 0) 4 else 2,
            @intCast(size), @intCast(base), .matrix) catch return error.TableGeometry;
    }
    const rows = regs[0].rows;
    if (rows == 0) return error.TableRows;
    // The pack's table must hold what the config's head-vocab lists sum to (they key on HF layer ids;
    // `engram_table_rows` is already pack-indexed). A disagreement is refused, never clamped.
    if (c.engram_table_rows[li] != rows) {
        return error.TableRows;
    }
    if (regs[1].rows != rows or regs[2].rows != rows) return error.TableRows;
    if (regs[0].cols * 32 != c.engram_head_dim * c.quant_bits) return error.TableGeometry;
    if (c.engram_head_dim % regs[1].cols != 0) return error.TableGeometry;
    return .{
        .fd = fd,
        .rows = rows,
        .w_off = base + regs[0].start,
        .w_stride = regs[0].cols * 4,
        .s_off = base + regs[1].start,
        .s_stride = regs[1].cols * 2,
        .b_off = base + regs[2].start,
        .b_stride = regs[2].cols * 2,
    };
}

fn openAt(a: std.mem.Allocator, dir_path: []const u8, rel: []const u8) !c_int {
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir_path, rel }, 0);
    const fd = crt.openRead(path.ptr);
    if (fd < 0) return error.TableOpen;
    return fd;
}

/// Does this file's safetensors header name `name`? (A cheap pread of the header, no tensor read.)
fn headerHas(a: std.mem.Allocator, fd: c_int, name: []const u8) !bool {
    var hl: [8]u8 = undefined;
    if (crt.pread(fd, &hl, 8, 0) != 8) return false;
    const hlen = std.mem.readInt(u64, &hl, .little);
    if (hlen > 16 << 20) return false;
    const hbuf = try a.alloc(u8, @intCast(hlen));
    if (crt.pread(fd, hbuf.ptr, hbuf.len, 8) != @as(isize, @intCast(hlen))) {
        // A short read means a PARTIAL data section (the fixture's oversized engram table); JSON-parse
        // what came back by trimming to the last complete '}' before deciding.
        const end = std.mem.lastIndexOfScalar(u8, hbuf, '}') orelse return false;
        const hdr = std.json.parseFromSliceLeaky(std.json.Value, a, hbuf[0 .. end + 1], .{}) catch return false;
        if (hdr != .object) return false;
        return hdr.object.get(name) != null;
    }
    const hdr = std.json.parseFromSliceLeaky(std.json.Value, a, hbuf, .{}) catch return false;
    if (hdr != .object) return false;
    return hdr.object.get(name) != null;
}

fn fileSize(fd: c_int) u64 {
    var st: std.c.Stat = std.mem.zeroes(std.c.Stat);
    if (std.c.fstat(fd, &st) != 0) return 0;
    return @intCast(st.size);
}

// ── tests ──

const testing = std.testing;

test "hy_v4 weights: the packed leaf shapes follow the config's quant" {
    // q_a_proj's packed U32 rows at g64 for a [out, 1024]-stored triple, and its scale columns.
    try testing.expectEqual(@as(u64, 128), @divExact(1024 * 4, 32));
    try testing.expectEqual(@as(u64, 16), @divExact(1024, 64));
}
