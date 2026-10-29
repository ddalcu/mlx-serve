//! GLM-5.3 (`glm_moe_dsa`, GlmMoeDsaForCausalLM) native arch: the typed config, the per-layer indexer and MLP tables,
//! the per-tensor affine bits, and the resident tensor spec checked against the pack's shard headers. Host-side: a pack
//! this build does not implement is refused by name before any MLX array (and so the Metal device) exists. Oracle: the
//! checkpoint's bundled `glm_moe_dsa.py` (mlx-lm's `deepseek_v32` with GLM's indexer schedule, fp32 indexer scores and
//! router logits); the routed experts stream from the affine bank (`bank.zig`), never from these shards.

const std = @import("std");
const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const checkpoint = @import("../../sdk/weights/streamed/checkpoint.zig");

/// The shard headers of a pack (`sdk/weights/streamed/checkpoint.zig`: 8-byte length + JSON, never tensor data).
pub const Checkpoint = checkpoint.Checkpoint;
pub const StDtype = checkpoint.StDtype;
pub const Diag = @import("sdk").Diag;

pub const model_type = "glm_moe_dsa";
pub const max_layers = 128;
/// Experts routed per token: the bank module's lookahead selector is compiled for it (`glm_moe_dsa_bank.routed_top_k`).
pub const routed_top_k = 8;
/// The affine group every quantized tensor of a pack uses.
pub const group_size = 64;
/// The q / kv latents' RMSNorm epsilon and the indexer key LayerNorm's (the reference hard-codes both).
pub const latent_norm_eps: f32 = 1e-6;
pub const index_norm_eps: f32 = 1e-6;

/// The prompt pass's shape (the graph runs it, the bill charges it): layer-major calls of at most `prompt_call_rows`
/// rows (`promptCallRows`); on the DSA kernels (`dsaKernelsFit`) each layer's attention in spans of query rows whose
/// own arrays stay under `span_target_bytes` (`promptSpanRows`), else in query blocks whose score arrays (the
/// indexer's fp32 heads, the latent attention's heads, over every key) stay under `score_budget_bytes`; a routed
/// group's rows through the gather math in slices of at most `group_slice_rows`; the weighted combine in slices of
/// `combine_slice_tokens` tokens.
pub const prompt_call_rows: u64 = 16384;
/// A last call shorter than this joins the call before it: a call re-reads every expert it routes that the call before
/// it held in a transient row.
pub const prompt_tail_rows: u64 = 2048;
pub const span_target_bytes: u64 = 2 << 30;
pub const score_budget_bytes: u64 = 256 << 20;
pub const group_slice_rows: u64 = 16384;
pub const combine_slice_tokens: u64 = 2048;

/// The prompt pass's work in flight, bounded by construction (the bill charges exactly these, `glm_moe_dsa_bill`): a
/// routed call's GEMM slices are committed one by one, and the slice `prompt_gemm_slices_in_flight` back is waited for
/// before the next is built (`glm_moe_dsa_experts`, `SliceWindow`); a layer's attention spans the same way,
/// `attentionSpansInFlight` of them (`glm_moe_dsa_graph.SpanWindow`). Before the windows these counts were margins
/// read off the M5 Ultra's prompts (how far the host's encode ran ahead of the GPU); now they hold on any box.
pub const prompt_gemm_slices_in_flight: u64 = 2;
/// The routed rows a call's GEMMs hold in flight at most: the slices in flight, each at most `group_slice_rows`.
pub const prompt_gemm_rows_in_flight: u64 = prompt_gemm_slices_in_flight * group_slice_rows;
/// The attention spans in flight per layer kind: the routed full layers', the routed shared layers' and the dense
/// layers' (the counts the measured bill charged before the window held them: its probe read at most 1.62, 2.43 and
/// 2.22 spans).
pub const attention_spans_in_flight: struct { full: u32, shared: u32, dense: u32 } = .{ .full = 2, .shared = 3, .dense = 3 };
pub const max_attention_spans_in_flight: u32 = @max(attention_spans_in_flight.full, @max(attention_spans_in_flight.shared, attention_spans_in_flight.dense));

/// The attention spans in flight of a layer of kind (`full`, `sparse`): `attention_spans_in_flight`'s.
pub fn attentionSpansInFlight(full: bool, sparse: bool) u32 {
    return if (!sparse) attention_spans_in_flight.dense else if (full) attention_spans_in_flight.full else attention_spans_in_flight.shared;
}

/// The rows of the prompt pass's call at row `at` of a `total`-row prompt: `prompt_call_rows`, or every row left when
/// fewer than `prompt_call_rows + prompt_tail_rows` are.
pub fn promptCallRows(total: u64, at: u64) u64 {
    const left = total - at;
    return if (left < prompt_call_rows + prompt_tail_rows) left else prompt_call_rows;
}

/// The most rows a prompt call of a `tokens`-token prompt runs.
pub fn maxPromptCallRows(tokens: u64) u64 {
    return @min(tokens, prompt_call_rows + prompt_tail_rows - 1);
}

/// One query row's part of a prompt span's own arrays over `positions` keys (the DSA kernels): the indexer's f32 score
/// and its partition over every position read, the absorbed query and its roped part, the latent output, the output
/// per head and its transpose for the output projection.
pub fn promptSpanRowBytes(c: *const Config, positions: u64) u64 {
    return 8 * positions + @as(u64, c.n_heads) * (2 * @as(u64, c.kv_lora_rank) + c.qk_rope_head_dim + 2 * @as(u64, c.v_head_dim)) * 2;
}

/// The query rows of one prompt span of a `rows`-row call over at most `positions` keys: its arrays under
/// `span_target_bytes`, at least one row.
pub fn promptSpanRows(c: *const Config, rows: u64, positions: u64) u64 {
    return @max(1, @min(rows, span_target_bytes / promptSpanRowBytes(c, positions)));
}

/// The DSA kernels (`mlx_host.dsa_nax`) serve these dims: the 512-wide latent with a 64-wide roped key, heads in
/// halves of 32, the indexer's 128-wide keys.
pub fn dsaKernelsFit(c: *const Config) bool {
    return c.kv_lora_rank == 512 and c.qk_rope_head_dim == 64 and c.n_heads % 32 == 0 and c.index_head_dim == 128;
}

/// The query rows of one attention block over `keys` keys: the widest score array (`heads` x rows x keys, fp32) under
/// `score_budget_bytes`, at least one row.
pub fn queryBlockRows(heads: u64, keys: u64) u64 {
    return @max(1, score_budget_bytes / (heads * @max(keys, 1) * 4));
}

pub const Refusal = error{
    ConfigSyntax,
    ConfigField,
    ModelType,
    NotImplemented,
    LayerTable,
    QuantOverride,
    DimsNotImplemented,
    TensorMissing,
    TensorDtype,
    TensorShape,
    TensorUnexpected,
    ExpertsInResidents,
};
pub const Error = Refusal || checkpoint.Error;

fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

pub const IndexerType = enum { full, shared };
pub const MlpType = enum { dense, sparse };

/// The dims of a release this build serves (`Config.parse`'s `want`): a pack with any other is refused by name.
pub const Dims = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    moe_intermediate_size: u32,
    n_layers: u32,
    n_heads: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    index_topk: u32,
    n_routed_experts: u32,
    n_shared_experts: u32,
};

/// GLM-5.3 (zai-org/GLM-5.3 and its MLX conversions, the MTP layer dropped).
pub const glm53: Dims = .{
    .vocab_size = 154880,
    .hidden_size = 6144,
    .intermediate_size = 12288,
    .moe_intermediate_size = 2048,
    .n_layers = 78,
    .n_heads = 64,
    .q_lora_rank = 2048,
    .kv_lora_rank = 512,
    .qk_nope_head_dim = 192,
    .qk_rope_head_dim = 64,
    .v_head_dim = 256,
    .index_n_heads = 32,
    .index_head_dim = 128,
    .index_topk = 2048,
    .n_routed_experts = 256,
    .n_shared_experts = 1,
};

/// The resident tensors' affine quantization: the default bits and the per-module overrides of a mixed build (each
/// `{bits, group_size}` object keyed by the module's path), sorted by path.
pub const Quant = struct {
    bits: u32,
    overrides: []const Override = &.{},

    pub const Override = struct { path: []const u8, bits: u32 };

    /// The bits of the quantized module at `path` (its tensors are `path.weight` / `.scales` / `.biases`).
    pub fn bitsOf(q: *const Quant, path: []const u8) u32 {
        const i = std.sort.binarySearch(Override, q.overrides, path, struct {
            fn cmp(p: []const u8, o: Override) std.math.Order {
                return std.mem.order(u8, p, o.path);
            }
        }.cmp) orelse return q.bits;
        return q.overrides[i].bits;
    }
};

/// The affine widths MLX quantizes at group 64 (`sdk_ext.quant.GatherQmm.valid`).
pub fn validBits(bits: i64) bool {
    return bits >= 2 and bits <= 8 and bits != 7;
}

/// One stream layer's slots as the bill and the stream size them (a bank module's `geometryOf`).
pub const LayerSlots = struct {
    /// One slot row's bytes (the layer's record; the stream allocates its rows of every component).
    record_bytes: u64,
    /// The layer's records: the most rows it takes.
    n_records: u32,
    /// Slot rows one expert slot takes (an EXL3 expert's 4 minis in adjacent rows; 1 for one record per expert).
    rows_per_unit: u32 = 1,
    /// The layer's expert slots per fill unit (the bank module's split; `rowsAt` rounds it): 1 for an even fill.
    share: f64 = 1,
    /// Each slot array's bytes per slot row: the stream allocates one MLX array per record segment, a row's minis times
    /// the segment's length a row, each array's buffer page-rounded (`glm_moe_dsa_bill.slotArrays`). All zero: one
    /// array of `record_bytes` a row.
    array_bytes: [9]u64 = @splat(0),

    /// An empty stream layer: an EXL3 pack's routed layer of one K keeps its second bank layer empty
    /// (`glm_moe_dsa_exl3_bank.Bank.closeLayer`). It holds no expert: it takes no slot at any units and no row, nothing
    /// routes on it (so no read, no gate, no wave), and the MTP MoE has no group for it. The one rule every consumer
    /// asks (the bill, the wide lane, the MTP lane).
    pub fn empty(l: LayerSlots) bool {
        return l.n_records == 0;
    }
};

/// A bank's slot geometry: the bill fills in units (`glm_moe_dsa_bill.unitBytes`, `maxUnits`), the module gives each
/// stream layer its rows at them (`rowsAt`).
pub const Geometry = struct {
    /// The stream's layers, in stream order (borrowed from the bank).
    layers: []const LayerSlots,
    /// The widest record (the transient rows' and decode's window) and the widest gate/up or down span (the read pool's
    /// staging).
    widest_record: u64,
    widest_span: u64,
    /// Slot rows one routed id takes.
    rows_per_id: u32 = 1,
    /// A routed id's records (an EXL3 expert's tensor-parallel minis; 1 for one record per expert): the routed call's
    /// GEMM runs per mini (the bill's `prefillRowBytes`).
    minis_per_id: u32 = 1,
    /// A routed id's span in the file at the stream's widest layer (`StreamOf.spanOf`): the lookahead's speculative slot
    /// holds one (the read pool's staging, the bill's `poolStagingBytes`).
    spec_record: u64 = 0,
    /// The transient scratches' slot arrays per row (`LayerSlots.array_bytes` of the widest layer, and with a scratch
    /// per geometry the other's; all zero: `widest_record` a row).
    scratch: [2][9]u64 = @splat(@splat(0)),
    /// An EXL3 bank's K values (bit k for each K of its manifest's `quantization.k_values`): the decode split's key
    /// (`glm_moe_dsa_split.packOf`). 0: another bank.
    k_mask: u16 = 0,
};

pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    moe_intermediate_size: u32,
    n_layers: u32,
    n_heads: u32,
    q_lora_rank: u32,
    kv_lora_rank: u32,
    qk_nope_head_dim: u32,
    qk_rope_head_dim: u32,
    v_head_dim: u32,
    index_n_heads: u32,
    index_head_dim: u32,
    index_topk: u32,
    n_routed_experts: u32,
    n_shared_experts: u32,
    n_experts_per_tok: u32,
    norm_topk_prob: bool,
    routed_scaling_factor: f32,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: u64,
    eos_ids: [8]u32 = @splat(0),
    n_eos: u8 = 0,
    indexer_types: [max_layers]IndexerType = @splat(.shared),
    mlp_types: [max_layers]MlpType = @splat(.dense),
    /// The MTP layers the release ships after the trunk (`num_nextn_predict_layers`; the MLX builds drop them, their
    /// config keeps the count), and whether a draft step past the first reuses the first step's indexer selection
    /// (`index_share_for_mtp_iteration`, absent = false).
    n_nextn: u32 = 0,
    index_share_mtp: bool = false,
    quant: Quant,
    /// The overrides' storage (`quant.overrides` and their paths), owned by `deinit`.
    owned_overrides: []Quant.Override = &.{},
    owned_paths: []u8 = &.{},

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, want: ?*const Dims, diag: ?*Diag) Error!Config {
        const path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
        defer gpa.free(path);
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(16 << 20)) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ConfigSyntax, "{s}: {s}", .{ path, @errorName(e) }),
        };
        defer gpa.free(text);
        return parse(gpa, text, want, diag);
    }

    /// config.json as this build implements it; anything else is refused by name, never approximated. `want` pins a
    /// release's dims (the served arch passes `glm53`; a harness's tiny model passes null).
    pub fn parse(gpa: std.mem.Allocator, text: []const u8, want: ?*const Dims, diag: ?*Diag) Error!Config {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), text, .{}) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(diag, error.ConfigSyntax, "config.json: {s}", .{@errorName(e)}),
        };
        if (v != .object) return refuse(diag, error.ConfigSyntax, "config.json: not an object", .{});
        const src: Src = .{ .root = v.object, .diag = diag };
        const mt = src.get("model_type") orelse return refuse(diag, error.ModelType, "config: model_type missing", .{});
        if (mt != .string or !std.mem.eql(u8, mt.string, model_type)) return refuse(diag, error.ModelType, "config: model_type is not " ++ model_type, .{});

        const u32max = std.math.maxInt(u32);
        var c: Config = .{
            .vocab_size = try src.uint("vocab_size", 1, u32max),
            .hidden_size = try src.uint("hidden_size", 64, 1 << 20),
            .intermediate_size = try src.uint("intermediate_size", 64, 1 << 20),
            .moe_intermediate_size = try src.uint("moe_intermediate_size", 64, 1 << 20),
            .n_layers = try src.uint("num_hidden_layers", 1, max_layers),
            .n_heads = try src.uint("num_attention_heads", 1, 1024),
            .q_lora_rank = try src.uint("q_lora_rank", 64, 1 << 20),
            .kv_lora_rank = try src.uint("kv_lora_rank", 64, 1 << 20),
            .qk_nope_head_dim = try src.uint("qk_nope_head_dim", 64, 4096),
            .qk_rope_head_dim = try src.uint("qk_rope_head_dim", 2, 4096),
            .v_head_dim = try src.uint("v_head_dim", 1, 4096),
            .index_n_heads = try src.uint("index_n_heads", 1, 1024),
            .index_head_dim = try src.uint("index_head_dim", 2, 4096),
            .index_topk = try src.uint("index_topk", 1, u32max),
            .n_routed_experts = try src.uint("n_routed_experts", routed_top_k, 1 << 16),
            .n_shared_experts = try src.uint("n_shared_experts", 1, 64),
            .n_experts_per_tok = try src.uint("num_experts_per_tok", 1, 1 << 16),
            .norm_topk_prob = try src.boolean("norm_topk_prob"),
            .routed_scaling_factor = @floatCast(try src.float("routed_scaling_factor")),
            .rms_norm_eps = @floatCast(try src.float("rms_norm_eps")),
            .rope_theta = 0,
            .max_position_embeddings = try src.uint64("max_position_embeddings"),
            .quant = undefined,
        };

        // What this build implements; anything else is refused, never approximated.
        if (c.n_experts_per_tok != routed_top_k) return refuse(diag, error.NotImplemented, "config: num_experts_per_tok = {d} (this build routes {d})", .{ c.n_experts_per_tok, routed_top_k });
        try src.expectString("topk_method", "noaux_tc");
        try src.expectString("scoring_func", "sigmoid");
        try src.expectString("hidden_act", "silu");
        if (try src.uint("n_group", 0, 1 << 16) != 1 or try src.uint("topk_group", 0, 1 << 16) != 1)
            return refuse(diag, error.NotImplemented, "config: n_group / topk_group must be 1 (no group selection)", .{});
        if (try src.boolean("attention_bias")) return refuse(diag, error.NotImplemented, "config: attention_bias must be false", .{});
        if (src.get("tie_word_embeddings")) |t| if (t != .bool or t.bool) return refuse(diag, error.NotImplemented, "config: tie_word_embeddings must be false", .{});
        if (try src.optBoolean("rope_interleave", true) != true or try src.optBoolean("indexer_rope_interleave", true) != true)
            return refuse(diag, error.NotImplemented, "config: rope_interleave / indexer_rope_interleave must be true (the interleaved rope)", .{});
        if (c.rms_norm_eps <= 0 or c.routed_scaling_factor <= 0) return refuse(diag, error.ConfigField, "config: rms_norm_eps / routed_scaling_factor out of range", .{});
        try c.parseRope(&src);

        // The dims every affine group-64 projection contracts over, and the rope inside each head.
        if (c.qk_rope_head_dim % 2 != 0 or c.qk_rope_head_dim > c.index_head_dim)
            return refuse(diag, error.ConfigField, "config: qk_rope_head_dim {d} is odd or wider than index_head_dim {d}", .{ c.qk_rope_head_dim, c.index_head_dim });
        for ([_]u32{ c.hidden_size, c.q_lora_rank, c.kv_lora_rank, c.qk_nope_head_dim, c.n_heads * c.v_head_dim, c.intermediate_size, c.moe_intermediate_size * c.n_shared_experts, c.moe_intermediate_size }) |d| {
            if (d % group_size != 0) return refuse(diag, error.NotImplemented, "config: contraction dim {d} is not a multiple of the affine group {d}", .{ d, group_size });
        }

        var ids: [8]i64 = undefined;
        const eos = try src.intsOrInt("eos_token_id", &ids);
        for (eos, 0..) |id, i| {
            if (id < 0 or id >= c.vocab_size) return refuse(diag, error.ConfigField, "config: eos_token_id {d} outside the vocabulary", .{id});
            c.eos_ids[i] = @intCast(id);
        }
        c.n_eos = @intCast(eos.len);

        try c.parseTables(&src);
        if (src.get("num_nextn_predict_layers") != null) c.n_nextn = try src.uint("num_nextn_predict_layers", 0, 64);
        c.index_share_mtp = try src.optBoolean("index_share_for_mtp_iteration", false);
        if (want) |w| try c.checkDims(w, diag);
        try c.parseQuant(gpa, &src);
        return c;
    }

    pub fn deinit(c: *Config, gpa: std.mem.Allocator) void {
        gpa.free(c.owned_overrides);
        gpa.free(c.owned_paths);
        c.owned_overrides = &.{};
        c.owned_paths = &.{};
        c.quant.overrides = &.{};
    }

    /// `rope_parameters` (the reference's `rope_scaling`): the default rope at `rope_theta`; a scaled rope is refused.
    fn parseRope(c: *Config, src: *const Src) Refusal!void {
        const diag = src.diag;
        const rp = src.get("rope_parameters") orelse src.get("rope_scaling") orelse {
            c.rope_theta = @floatCast(try src.float("rope_theta"));
            return;
        };
        if (rp != .object) return refuse(diag, error.ConfigField, "config: rope_parameters is not an object", .{});
        const r: Src = .{ .root = rp.object, .diag = diag };
        const kind = r.get("rope_type") orelse r.get("type");
        if (kind) |k| if (k != .string or !std.mem.eql(u8, k.string, "default"))
            return refuse(diag, error.NotImplemented, "config: rope_parameters.rope_type is not \"default\" (this build implements the plain rope)", .{});
        if (r.get("mscale_all_dim")) |m| if (m != .integer or m.integer != 0)
            return refuse(diag, error.NotImplemented, "config: rope_parameters.mscale_all_dim is set (an attention scale this build does not implement)", .{});
        c.rope_theta = @floatCast(try r.float("rope_theta"));
        if (c.rope_theta <= 0) return refuse(diag, error.ConfigField, "config: rope_theta {d} out of range", .{c.rope_theta});
    }

    /// `indexer_types` (else the reference's schedule from `index_topk_freq` / `index_skip_topk_offset`) and
    /// `mlp_layer_types`, which must equal the reference's own MoE rule (`first_k_dense_replace`, `moe_layer_freq`).
    fn parseTables(c: *Config, src: *const Src) Refusal!void {
        const diag = src.diag;
        const n = c.n_layers;
        if (src.get("indexer_types")) |it| {
            if (it != .array or it.array.items.len != n) return refuse(diag, error.LayerTable, "config: indexer_types is not a list of {d} entries", .{n});
            for (it.array.items, 0..) |e, l| {
                if (e != .string) return refuse(diag, error.LayerTable, "config: indexer_types[{d}] is not a string", .{l});
                c.indexer_types[l] = std.meta.stringToEnum(IndexerType, e.string) orelse return refuse(diag, error.LayerTable, "config: indexer_types[{d}] = \"{s}\" (full or shared)", .{ l, e.string });
            }
        } else {
            const f = try src.uint("index_topk_freq", 1, 1 << 16);
            const o: i64 = try src.int("index_skip_topk_offset", 0, 1 << 16);
            for (0..n) |l| c.indexer_types[l] = if (@mod(@max(@as(i64, @intCast(l)) - o + 1, 0), @as(i64, f)) == 0) .full else .shared;
        }
        const k_dense = try src.uint("first_k_dense_replace", 0, max_layers);
        const freq = try src.uint("moe_layer_freq", 1, max_layers);
        const mt = src.get("mlp_layer_types") orelse return refuse(diag, error.LayerTable, "config: mlp_layer_types missing", .{});
        if (mt != .array or mt.array.items.len != n) return refuse(diag, error.LayerTable, "config: mlp_layer_types is not a list of {d} entries", .{n});
        var sparse: u32 = 0;
        for (mt.array.items, 0..) |e, l| {
            if (e != .string) return refuse(diag, error.LayerTable, "config: mlp_layer_types[{d}] is not a string", .{l});
            c.mlp_types[l] = std.meta.stringToEnum(MlpType, e.string) orelse return refuse(diag, error.LayerTable, "config: mlp_layer_types[{d}] = \"{s}\" (dense or sparse)", .{ l, e.string });
            const moe = l >= k_dense and l % freq == 0;
            if (moe != (c.mlp_types[l] == .sparse)) return refuse(diag, error.LayerTable, "config: mlp_layer_types[{d}] = {t} where first_k_dense_replace {d} / moe_layer_freq {d} build {s}", .{ l, c.mlp_types[l], k_dense, freq, if (moe) "an MoE layer" else "a dense MLP" });
            sparse += @intFromBool(moe);
        }
        if (sparse == 0) return refuse(diag, error.LayerTable, "config: mlp_layer_types has no sparse layer (nothing streams)", .{});
    }

    fn checkDims(c: *const Config, w: *const Dims, diag: ?*Diag) Refusal!void {
        inline for (comptime std.meta.fieldNames(Dims)) |f| {
            if (@field(c, f) != @field(w, f)) return refuse(diag, error.DimsNotImplemented, "config: " ++ f ++ " = {d} (this build implements {d})", .{ @field(c, f), @field(w, f) });
        }
    }

    /// The root `quantization`: affine (mode absent or "affine"), group 64, the default bits, and per-module
    /// `{bits, group_size}` overrides (any other entry refused).
    fn parseQuant(c: *Config, gpa: std.mem.Allocator, src: *const Src) Error!void {
        const diag = src.diag;
        const q = src.get("quantization") orelse return refuse(diag, error.QuantOverride, "config: quantization missing (an affine pack)", .{});
        if (q != .object) return refuse(diag, error.QuantOverride, "config: quantization is not an object", .{});
        const qs: Src = .{ .root = q.object, .diag = diag };
        if (qs.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return refuse(diag, error.NotImplemented, "config: quantization.mode is not affine", .{});
        if (try qs.int("group_size", 1, 1024) != group_size) return refuse(diag, error.NotImplemented, "config: quantization.group_size is not {d}", .{group_size});
        const bits = try qs.int("bits", 1, 16);
        if (!validBits(bits)) return refuse(diag, error.NotImplemented, "config: quantization.bits {d} is not an affine width", .{bits});
        var n: usize = 0;
        var path_bytes: usize = 0;
        for (q.object.keys(), q.object.values()) |k, val| {
            if (isQuantKey(k)) continue;
            if (val != .object) return refuse(diag, error.QuantOverride, "config: quantization.{s} is not a {{bits, group_size}} object", .{k});
            n += 1;
            path_bytes += k.len;
        }
        const ov = try gpa.alloc(Quant.Override, n);
        errdefer gpa.free(ov);
        const buf = try gpa.alloc(u8, path_bytes);
        errdefer gpa.free(buf);
        var i: usize = 0;
        var at: usize = 0;
        for (q.object.keys(), q.object.values()) |k, val| {
            if (isQuantKey(k)) continue;
            const o: Src = .{ .root = val.object, .diag = diag };
            if (try o.int("group_size", 1, 1024) != group_size) return refuse(diag, error.QuantOverride, "config: quantization.{s}.group_size is not {d}", .{ k, group_size });
            const b = try o.int("bits", 1, 16);
            if (!validBits(b)) return refuse(diag, error.QuantOverride, "config: quantization.{s}.bits {d} is not an affine width", .{ k, b });
            @memcpy(buf[at..][0..k.len], k);
            ov[i] = .{ .path = buf[at..][0..k.len], .bits = @intCast(b) };
            at += k.len;
            i += 1;
        }
        std.sort.pdq(Quant.Override, ov, {}, struct {
            fn lt(_: void, x: Quant.Override, y: Quant.Override) bool {
                return std.mem.lessThan(u8, x.path, y.path);
            }
        }.lt);
        c.quant = .{ .bits = @intCast(bits), .overrides = ov };
        c.owned_overrides = ov;
        c.owned_paths = buf;
    }

    fn isQuantKey(k: []const u8) bool {
        return std.mem.eql(u8, k, "bits") or std.mem.eql(u8, k, "group_size") or std.mem.eql(u8, k, "mode");
    }

    pub fn isFull(c: *const Config, l: u32) bool {
        return c.indexer_types[l] == .full;
    }

    pub fn isSparse(c: *const Config, l: u32) bool {
        return c.mlp_types[l] == .sparse;
    }

    /// The routed layers in model order (the bank's layer index = position here).
    pub fn sparseLayers(c: *const Config, out: *[max_layers]u32) []const u32 {
        var n: usize = 0;
        for (0..c.n_layers) |l| if (c.mlp_types[l] == .sparse) {
            out[n] = @intCast(l);
            n += 1;
        };
        return out[0..n];
    }

    pub fn nSparse(c: *const Config) u32 {
        var n: u32 = 0;
        for (c.mlp_types[0..c.n_layers]) |t| n += @intFromBool(t == .sparse);
        return n;
    }

    pub fn nFull(c: *const Config) u32 {
        var n: u32 = 0;
        for (c.indexer_types[0..c.n_layers]) |t| n += @intFromBool(t == .full);
        return n;
    }

    /// The bank index of model layer `l` (its rank among the sparse layers), null for a dense layer.
    pub fn bankLayer(c: *const Config, l: u32) ?u32 {
        if (c.mlp_types[l] != .sparse) return null;
        var n: u32 = 0;
        for (c.mlp_types[0..l]) |t| n += @intFromBool(t == .sparse);
        return n;
    }

    pub fn qHeadDim(c: *const Config) u32 {
        return c.qk_nope_head_dim + c.qk_rope_head_dim;
    }

};

/// A config.json object's fields, refused by name when missing or mistyped.
const Src = struct {
    root: std.json.ObjectMap,
    diag: ?*Diag,

    fn get(s: Src, name: []const u8) ?std.json.Value {
        return s.root.get(name);
    }

    fn int(s: Src, name: []const u8, lo: i64, hi: i64) Refusal!i64 {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        if (v != .integer) return refuse(s.diag, error.ConfigField, "config: {s} is not an integer", .{name});
        if (v.integer < lo or v.integer > hi) return refuse(s.diag, error.ConfigField, "config: {s} = {d} outside [{d}, {d}]", .{ name, v.integer, lo, hi });
        return v.integer;
    }

    fn uint(s: Src, name: []const u8, lo: i64, hi: i64) Refusal!u32 {
        return @intCast(try s.int(name, lo, hi));
    }

    fn uint64(s: Src, name: []const u8) Refusal!u64 {
        return @intCast(try s.int(name, 1, std.math.maxInt(i64)));
    }

    fn float(s: Src, name: []const u8) Refusal!f64 {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        return switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => refuse(s.diag, error.ConfigField, "config: {s} is not a number", .{name}),
        };
    }

    fn boolean(s: Src, name: []const u8) Refusal!bool {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        if (v != .bool) return refuse(s.diag, error.ConfigField, "config: {s} is not a bool", .{name});
        return v.bool;
    }

    /// A bool the reference defaults when absent.
    fn optBoolean(s: Src, name: []const u8, default: bool) Refusal!bool {
        if (s.get(name) == null) return default;
        return s.boolean(name);
    }

    fn expectString(s: Src, name: []const u8, want: []const u8) Refusal!void {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        if (v != .string) return refuse(s.diag, error.ConfigField, "config: {s} is not a string", .{name});
        if (!std.mem.eql(u8, v.string, want)) return refuse(s.diag, error.NotImplemented, "config: {s} = \"{s}\" (this build implements \"{s}\")", .{ name, v.string, want });
    }

    /// An integer or a list of them (`eos_token_id`), at most `out.len`.
    fn intsOrInt(s: Src, name: []const u8, out: []i64) Refusal![]i64 {
        const v = s.get(name) orelse return refuse(s.diag, error.ConfigField, "config: {s} missing", .{name});
        if (v == .integer) {
            out[0] = v.integer;
            return out[0..1];
        }
        if (v != .array or v.array.items.len == 0 or v.array.items.len > out.len) return refuse(s.diag, error.ConfigField, "config: {s} is not an integer or a list of 1..{d}", .{ name, out.len });
        for (v.array.items, 0..) |item, i| {
            if (item != .integer) return refuse(s.diag, error.ConfigField, "config: {s}[{d}] is not an integer", .{ name, i });
            out[i] = item.integer;
        }
        return out[0..v.array.items.len];
    }
};

// ── the resident tensor spec ──

/// A float tensor the reference keeps as stored (the indexer, the router and its bias): bf16 or f32.
pub const float_dtypes = [_]StDtype{ .BF16, .F32 };
const bf16_only = [_]StDtype{.BF16};
const u32_only = [_]StDtype{.U32};

/// One resident tensor: its name, the dtypes it may be stored in, its shape.
pub const Param = struct {
    name: []const u8,
    dtypes: []const StDtype,
    shape: [3]u64 = @splat(0),
    rank: u8,

    pub fn shapeOf(p: *const Param) []const u64 {
        return p.shape[0..p.rank];
    }
};

/// Every resident tensor the pack must carry, names in `a` (an arena): the trunk's affine projections (`.weight` U32
/// `[out, in * bits / 32]`, `.scales` / `.biases` BF16 `[out, in / 64]`, the bits from `quantization`), the latent
/// projections `embed_q` / `unembed_out` per head (`[heads, out, ...]`), the norms in bf16, and the indexer (full
/// layers only), the router and its correction bias as stored. The routed experts are the bank's.
pub fn residentSpec(a: std.mem.Allocator, c: *const Config) ![]Param {
    var out: std.ArrayList(Param) = .empty;
    const S = struct {
        fn dense(o: *std.ArrayList(Param), al: std.mem.Allocator, name: []const u8, dtypes: []const StDtype, shape: []const u64) !void {
            var p: Param = .{ .name = name, .dtypes = dtypes, .rank = @intCast(shape.len) };
            @memcpy(p.shape[0..shape.len], shape);
            try o.append(al, p);
        }
        /// An affine projection (`heads` > 0: one per head, mlx-lm's `QuantizedMultiLinear`).
        fn quant(o: *std.ArrayList(Param), al: std.mem.Allocator, cfg: *const Config, path: []const u8, heads: u64, out_dim: u64, in_dim: u64) !void {
            const bits = cfg.quant.bitsOf(path);
            const packed_in = in_dim * bits / 32;
            const groups = in_dim / group_size;
            const lead: []const u64 = if (heads > 0) &.{heads} else &.{};
            var w: [3]u64 = undefined;
            var s: [3]u64 = undefined;
            const r = lead.len;
            @memcpy(w[0..r], lead);
            @memcpy(s[0..r], lead);
            w[r] = out_dim;
            w[r + 1] = packed_in;
            s[r] = out_dim;
            s[r + 1] = groups;
            try dense(o, al, try std.fmt.allocPrint(al, "{s}.weight", .{path}), &u32_only, w[0 .. r + 2]);
            try dense(o, al, try std.fmt.allocPrint(al, "{s}.scales", .{path}), &bf16_only, s[0 .. r + 2]);
            try dense(o, al, try std.fmt.allocPrint(al, "{s}.biases", .{path}), &bf16_only, s[0 .. r + 2]);
        }
    };
    const h: u64 = c.hidden_size;
    try S.quant(&out, a, c, "model.embed_tokens", 0, c.vocab_size, h);
    try S.dense(&out, a, "model.norm.weight", &bf16_only, &.{h});
    try S.quant(&out, a, c, "lm_head", 0, c.vocab_size, h);
    for (0..c.n_layers) |li| {
        const l: u32 = @intCast(li);
        const p = try std.fmt.allocPrint(a, "model.layers.{d}", .{l});
        const at = struct {
            fn f(al: std.mem.Allocator, prefix: []const u8, rest: []const u8) ![]const u8 {
                return std.fmt.allocPrint(al, "{s}.{s}", .{ prefix, rest });
            }
        }.f;
        try S.dense(&out, a, try at(a, p, "input_layernorm.weight"), &bf16_only, &.{h});
        try S.dense(&out, a, try at(a, p, "post_attention_layernorm.weight"), &bf16_only, &.{h});
        try S.quant(&out, a, c, try at(a, p, "self_attn.q_a_proj"), 0, c.q_lora_rank, h);
        try S.dense(&out, a, try at(a, p, "self_attn.q_a_layernorm.weight"), &bf16_only, &.{c.q_lora_rank});
        try S.quant(&out, a, c, try at(a, p, "self_attn.q_b_proj"), 0, @as(u64, c.n_heads) * c.qHeadDim(), c.q_lora_rank);
        try S.quant(&out, a, c, try at(a, p, "self_attn.kv_a_proj_with_mqa"), 0, c.kv_lora_rank + c.qk_rope_head_dim, h);
        try S.dense(&out, a, try at(a, p, "self_attn.kv_a_layernorm.weight"), &bf16_only, &.{c.kv_lora_rank});
        try S.quant(&out, a, c, try at(a, p, "self_attn.embed_q"), c.n_heads, c.kv_lora_rank, c.qk_nope_head_dim);
        try S.quant(&out, a, c, try at(a, p, "self_attn.unembed_out"), c.n_heads, c.v_head_dim, c.kv_lora_rank);
        try S.quant(&out, a, c, try at(a, p, "self_attn.o_proj"), 0, h, @as(u64, c.n_heads) * c.v_head_dim);
        if (c.isFull(l)) {
            try S.dense(&out, a, try at(a, p, "self_attn.indexer.wq_b.weight"), &float_dtypes, &.{ @as(u64, c.index_n_heads) * c.index_head_dim, c.q_lora_rank });
            try S.dense(&out, a, try at(a, p, "self_attn.indexer.wk.weight"), &float_dtypes, &.{ c.index_head_dim, h });
            try S.dense(&out, a, try at(a, p, "self_attn.indexer.k_norm.weight"), &float_dtypes, &.{c.index_head_dim});
            try S.dense(&out, a, try at(a, p, "self_attn.indexer.k_norm.bias"), &float_dtypes, &.{c.index_head_dim});
            try S.dense(&out, a, try at(a, p, "self_attn.indexer.weights_proj.weight"), &float_dtypes, &.{ c.index_n_heads, h });
        }
        if (c.isSparse(l)) {
            const si: u64 = @as(u64, c.moe_intermediate_size) * c.n_shared_experts;
            try S.dense(&out, a, try at(a, p, "mlp.gate.weight"), &float_dtypes, &.{ c.n_routed_experts, h });
            try S.dense(&out, a, try at(a, p, "mlp.gate.e_score_correction_bias"), &float_dtypes, &.{c.n_routed_experts});
            try S.quant(&out, a, c, try at(a, p, "mlp.shared_experts.gate_proj"), 0, si, h);
            try S.quant(&out, a, c, try at(a, p, "mlp.shared_experts.up_proj"), 0, si, h);
            try S.quant(&out, a, c, try at(a, p, "mlp.shared_experts.down_proj"), 0, h, si);
        } else {
            try S.quant(&out, a, c, try at(a, p, "mlp.gate_proj"), 0, c.intermediate_size, h);
            try S.quant(&out, a, c, try at(a, p, "mlp.up_proj"), 0, c.intermediate_size, h);
            try S.quant(&out, a, c, try at(a, p, "mlp.down_proj"), 0, h, c.intermediate_size);
        }
    }
    return out.toOwnedSlice(a);
}

/// The pack's residents against the spec, from the shard headers alone: every spec tensor present in one of its
/// dtypes at its exact shape, and nothing else (a routed expert left in the shards is refused by name: it belongs
/// in the bank). Returns the residents' bytes.
pub fn checkResidents(spec: []const Param, ck: *const Checkpoint, diag: ?*Diag) Refusal!u64 {
    for (spec) |p| {
        const t = ck.tensors.get(p.name) orelse return refuse(diag, error.TensorMissing, "{s}: not in the pack's shards", .{p.name});
        if (std.mem.indexOfScalar(StDtype, p.dtypes, t.dtype) == null) return refuse(diag, error.TensorDtype, "{s}: dtype {t}, want {any}", .{ p.name, t.dtype, p.dtypes });
        if (t.rank != p.rank or !std.mem.eql(u64, t.shape[0..t.rank], p.shapeOf())) return refuse(diag, error.TensorShape, "{s}: shape {any}, want {any}", .{ p.name, t.shape[0..t.rank], p.shapeOf() });
    }
    var bytes: u64 = 0;
    for (ck.tensors.keys(), ck.tensors.values()) |name, t| {
        if (std.mem.indexOf(u8, name, ".mlp.switch_mlp.") != null) return refuse(diag, error.ExpertsInResidents, "{s}: a routed expert tensor in the resident shards (the bank serves them)", .{name});
        if (!inSpec(spec, name)) return refuse(diag, error.TensorUnexpected, "{s}: in the pack's shards, not a tensor of this arch", .{name});
        bytes += t.end - t.begin;
    }
    return bytes;
}

fn inSpec(spec: []const Param, name: []const u8) bool {
    for (spec) |p| if (std.mem.eql(u8, p.name, name)) return true;
    return false;
}

/// The pack's resident bytes after the spec check (`checkResidents` over the indexed shards).
pub fn residentBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const Config, diag: ?*Diag) Error!u64 {
    var ck = try Checkpoint.openIndexed(gpa, io, dir, diag);
    defer ck.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const spec = try residentSpec(arena.allocator(), c);
    return checkResidents(spec, &ck, diag);
}

/// The pack's residents as MLX allocates them once they load (`bufferBytes` of every shard tensor), after the spec check.
/// `classes`: they load with the allocator's size classes on (GLM-5.3's module: the first prompt's forward layers read
/// them when the process's MLX has the switch, `bill.sizeClassSwitch`), each at its class (`glm_moe_dsa_cache.sizeClassBytes`).
pub fn residentBufferBytes(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const Config, page: u64, classes: bool, diag: ?*Diag) Error!u64 {
    var ck = try Checkpoint.openIndexed(gpa, io, dir, diag);
    defer ck.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    _ = try checkResidents(try residentSpec(arena.allocator(), c), &ck, diag);
    if (!classes) return bufferBytes(&ck, page);
    var n: u64 = 0;
    for (ck.tensors.values()) |t| n += @import("cache.zig").sizeClassBytes(t.end - t.begin);
    return n;
}

/// Every tensor of `ck` in its own MLX buffer: past one page, rounded up to whole pages (`glm_moe_dsa_cache.bufferBytes`).
pub fn bufferBytes(ck: *const Checkpoint, page: u64) u64 {
    var n: u64 = 0;
    for (ck.tensors.values()) |t| n += @import("cache.zig").bufferBytes(t.end - t.begin, page);
    return n;
}

/// A pack's resident shard for tests: every spec tensor of `c` zero-filled in one `model.safetensors`, and its index.
pub fn writeResidents(a: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, c: *const Config) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const spec = try residentSpec(arena.allocator(), c);
    var idx: std.ArrayList(u8) = .empty;
    defer idx.deinit(a);
    try idx.appendSlice(a, "{\"weight_map\":{");
    const st = @import("sdk").testing;
    const ts = try arena.allocator().alloc(st.StTensor, spec.len);
    var off: u64 = 0;
    for (spec, ts, 0..) |*p, *t, i| {
        var n: u64 = p.dtypes[0].size();
        for (p.shapeOf()) |d| n *= d;
        t.* = .{ .name = p.name, .dtype = @tagName(p.dtypes[0]), .shape = p.shapeOf(), .len = n };
        try idx.print(a, "{s}\"{s}\":\"model.safetensors\"", .{ if (i == 0) "" else ",", p.name });
        off += n;
    }
    try idx.print(a, "}},\"metadata\":{{\"total_size\":{d}}}}}", .{off});
    const file = try st.safetensors(a, ts, .{ .metadata = true, .pad = true });
    defer a.free(file);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = file });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = idx.items });
}

// ── tests (host-only: no MLX array is created anywhere below) ──

const testing = std.testing;

/// The release's config.json (the pack's), as a test fixture: GLM-5.3's dims and tables, the uniform 4-bit build's
/// quantization unless `overrides` adds a mixed build's entries.
pub fn testConfigJson(a: std.mem.Allocator, overrides: []const u8) ![]u8 {
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    try j.appendSlice(a, "{\"architectures\":[\"GlmMoeDsaForCausalLM\"],\"model_type\":\"glm_moe_dsa\",\"attention_bias\":false,\"eos_token_id\":[154820,154827,154829],");
    try j.appendSlice(a, "\"first_k_dense_replace\":3,\"hidden_act\":\"silu\",\"hidden_size\":6144,\"index_head_dim\":128,\"index_n_heads\":32,\"index_topk\":2048,\"index_topk_freq\":4,\"index_skip_topk_offset\":3,");
    try j.appendSlice(a, "\"indexer_rope_interleave\":true,\"intermediate_size\":12288,\"kv_lora_rank\":512,\"max_position_embeddings\":1048576,\"moe_intermediate_size\":2048,\"moe_layer_freq\":1,");
    try j.appendSlice(a, "\"n_group\":1,\"n_routed_experts\":256,\"n_shared_experts\":1,\"norm_topk_prob\":true,\"num_attention_heads\":64,\"num_experts_per_tok\":8,\"num_hidden_layers\":78,\"num_key_value_heads\":64,");
    try j.appendSlice(a, "\"q_lora_rank\":2048,\"qk_nope_head_dim\":192,\"qk_rope_head_dim\":64,\"rms_norm_eps\":1e-05,\"rope_interleave\":true,\"rope_parameters\":{\"rope_theta\":8000000,\"rope_type\":\"default\"},");
    try j.appendSlice(a, "\"routed_scaling_factor\":2.5,\"scoring_func\":\"sigmoid\",\"tie_word_embeddings\":false,\"topk_group\":1,\"topk_method\":\"noaux_tc\",\"v_head_dim\":256,\"vocab_size\":154880,");
    try j.appendSlice(a, "\"num_nextn_predict_layers\":1,\"index_share_for_mtp_iteration\":true,");
    try j.appendSlice(a, "\"indexer_types\":[");
    for (0..78) |l| try j.print(a, "{s}\"{s}\"", .{ if (l == 0) "" else ",", if (@mod(@max(@as(i64, @intCast(l)) - 3 + 1, 0), 4) == 0) "full" else "shared" });
    try j.appendSlice(a, "],\"mlp_layer_types\":[");
    for (0..78) |l| try j.print(a, "{s}\"{s}\"", .{ if (l == 0) "" else ",", if (l < 3) "dense" else "sparse" });
    try j.print(a, "],\"quantization\":{{\"group_size\":64,\"bits\":4{s}}}}}", .{overrides});
    return j.toOwnedSlice(a);
}

/// A tiny model of the same arch (the synthetic packs' and the parity harness's): 5 layers (1 dense, 4 routed), the
/// indexer full on layers 0, 1 and 3, 16 experts top-8, every contraction one or two affine groups. `quant` is the
/// `quantization` object's body.
pub fn tinyConfigJson(a: std.mem.Allocator, quant: []const u8) ![]u8 {
    return std.fmt.allocPrint(a, "{{\"model_type\":\"glm_moe_dsa\",\"attention_bias\":false,\"eos_token_id\":[1],\"first_k_dense_replace\":1,\"hidden_act\":\"silu\"," ++
        "\"hidden_size\":128,\"index_head_dim\":32,\"index_n_heads\":16,\"index_topk\":8,\"indexer_rope_interleave\":true,\"intermediate_size\":128,\"kv_lora_rank\":64," ++
        "\"max_position_embeddings\":4096,\"moe_intermediate_size\":64,\"moe_layer_freq\":1,\"n_group\":1,\"n_routed_experts\":16,\"n_shared_experts\":1,\"norm_topk_prob\":true," ++
        "\"num_attention_heads\":2,\"num_experts_per_tok\":8,\"num_hidden_layers\":5,\"num_key_value_heads\":2,\"q_lora_rank\":64,\"qk_nope_head_dim\":64,\"qk_rope_head_dim\":16," ++
        "\"rms_norm_eps\":1e-05,\"rope_interleave\":true,\"rope_parameters\":{{\"rope_theta\":10000,\"rope_type\":\"default\"}},\"routed_scaling_factor\":2.5,\"scoring_func\":\"sigmoid\"," ++
        "\"tie_word_embeddings\":false,\"topk_group\":1,\"topk_method\":\"noaux_tc\",\"v_head_dim\":64,\"vocab_size\":256," ++
        "\"indexer_types\":[\"full\",\"full\",\"shared\",\"full\",\"shared\"],\"mlp_layer_types\":[\"dense\",\"sparse\",\"sparse\",\"sparse\",\"sparse\"]," ++
        "\"quantization\":{{{s}}}}}", .{quant});
}

pub const tiny_quant = "\"group_size\":64,\"bits\":4,\"model.embed_tokens\":{\"group_size\":64,\"bits\":8},\"model.layers.1.self_attn.o_proj\":{\"group_size\":64,\"bits\":8}";

test "glm config: the tiny model parses with no release pinned and is refused as GLM-5.3" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const text = try tinyConfigJson(testing.allocator, tiny_quant);
    defer testing.allocator.free(text);
    var diag: Diag = .{};
    var c = Config.parse(testing.allocator, text, null, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer c.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 4), c.nSparse());
    try testing.expectEqual(@as(u32, 3), c.nFull());
    try testing.expectEqual(@as(u32, 8), c.quant.bitsOf("model.embed_tokens"));
    try testing.expectError(error.DimsNotImplemented, Config.parse(testing.allocator, text, &glm53, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "vocab_size") != null);
}

test "glm config: the release's config parses to GLM-5.3's dims, 21 full indexer layers and 75 routed layers" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const text = try testConfigJson(testing.allocator, "");
    defer testing.allocator.free(text);
    var diag: Diag = .{};
    var c = Config.parse(testing.allocator, text, &glm53, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer c.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 78), c.n_layers);
    try testing.expectEqual(@as(u32, 21), c.nFull());
    try testing.expectEqual(@as(u32, 75), c.nSparse());
    try testing.expect(c.isFull(0) and c.isFull(1) and c.isFull(2) and !c.isFull(3) and c.isFull(6) and !c.isFull(77));
    try testing.expectEqual(@as(?u32, null), c.bankLayer(2));
    try testing.expectEqual(@as(?u32, 0), c.bankLayer(3));
    try testing.expectEqual(@as(?u32, 74), c.bankLayer(77));
    try testing.expectEqual(@as(f32, 8_000_000), c.rope_theta);
    try testing.expectEqual(@as(f32, 2.5), c.routed_scaling_factor);
    try testing.expectEqualSlices(u32, &.{ 154820, 154827, 154829 }, c.eos_ids[0..c.n_eos]);
    try testing.expectEqual(@as(u32, 4), c.quant.bitsOf("model.layers.0.self_attn.q_a_proj"));
    try testing.expectEqual(@as(u32, 1), c.n_nextn);
    try testing.expect(c.index_share_mtp);
}

test "glm config: a mixed build's per-module bits override the default, every other module keeps it" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const text = try testConfigJson(testing.allocator, ",\"model.embed_tokens\":{\"bits\":8,\"group_size\":64},\"model.layers.3.self_attn.o_proj\":{\"group_size\":64,\"bits\":8}");
    defer testing.allocator.free(text);
    var c = try Config.parse(testing.allocator, text, &glm53, null);
    defer c.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 8), c.quant.bitsOf("model.embed_tokens"));
    try testing.expectEqual(@as(u32, 8), c.quant.bitsOf("model.layers.3.self_attn.o_proj"));
    try testing.expectEqual(@as(u32, 4), c.quant.bitsOf("model.layers.4.self_attn.o_proj"));
    try testing.expectEqual(@as(u32, 4), c.quant.bitsOf("lm_head"));
}

test "glm config: every refusal names its field" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const Case = struct { from: []const u8, to: []const u8, err: anyerror, why: []const u8 };
    const cases = [_]Case{
        .{ .from = "\"model_type\":\"glm_moe_dsa\"", .to = "\"model_type\":\"glm4_moe\"", .err = error.ModelType, .why = "model_type" },
        .{ .from = "\"topk_method\":\"noaux_tc\"", .to = "\"topk_method\":\"greedy\"", .err = error.NotImplemented, .why = "topk_method" },
        .{ .from = "\"scoring_func\":\"sigmoid\"", .to = "\"scoring_func\":\"softmax\"", .err = error.NotImplemented, .why = "scoring_func" },
        .{ .from = "\"n_group\":1", .to = "\"n_group\":8", .err = error.NotImplemented, .why = "n_group" },
        .{ .from = "\"rope_type\":\"default\"", .to = "\"rope_type\":\"yarn\"", .err = error.NotImplemented, .why = "rope_type" },
        .{ .from = "\"num_hidden_layers\":78", .to = "\"num_hidden_layers\":79", .err = error.LayerTable, .why = "indexer_types" },
        .{ .from = "\"num_experts_per_tok\":8", .to = "\"num_experts_per_tok\":6", .err = error.NotImplemented, .why = "num_experts_per_tok" },
        .{ .from = "\"first_k_dense_replace\":3", .to = "\"first_k_dense_replace\":4", .err = error.LayerTable, .why = "mlp_layer_types[3]" },
        .{ .from = "\"hidden_size\":6144", .to = "\"hidden_size\":6208", .err = error.DimsNotImplemented, .why = "hidden_size" },
        .{ .from = "\"index_topk\":2048", .to = "\"index_topk\":1024", .err = error.DimsNotImplemented, .why = "index_topk" },
        .{ .from = "\"group_size\":64,\"bits\":4", .to = "\"group_size\":32,\"bits\":4", .err = error.NotImplemented, .why = "group_size" },
        .{ .from = "\"group_size\":64,\"bits\":4", .to = "\"group_size\":64,\"bits\":7", .err = error.NotImplemented, .why = "bits" },
        .{ .from = "\"group_size\":64,\"bits\":4", .to = "\"group_size\":64,\"bits\":4,\"lm_head\":false", .err = error.QuantOverride, .why = "lm_head" },
        .{ .from = "\"rope_interleave\":true", .to = "\"rope_interleave\":false", .err = error.NotImplemented, .why = "rope_interleave" },
        .{ .from = "\"attention_bias\":false", .to = "\"attention_bias\":true", .err = error.NotImplemented, .why = "attention_bias" },
    };
    const base = try testConfigJson(testing.allocator, "");
    defer testing.allocator.free(base);
    for (cases) |cs| {
        const text = try std.mem.replaceOwned(u8, testing.allocator, base, cs.from, cs.to);
        defer testing.allocator.free(text);
        try testing.expect(!std.mem.eql(u8, text, base));
        var diag: Diag = .{};
        testing.expectError(cs.err, Config.parse(testing.allocator, text, &glm53, &diag)) catch |e| {
            std.debug.print("case {s}: {s}\n", .{ cs.to, diag.message() });
            return e;
        };
        testing.expect(std.mem.indexOf(u8, diag.message(), cs.why) != null) catch |e| {
            std.debug.print("want \"{s}\" in \"{s}\"\n", .{ cs.why, diag.message() });
            return e;
        };
    }
}

test "glm config: without indexer_types the reference's schedule builds the same table" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const base = try testConfigJson(testing.allocator, "");
    defer testing.allocator.free(base);
    const start = std.mem.indexOf(u8, base, "\"indexer_types\":[").?;
    const end = std.mem.indexOfPos(u8, base, start, "],").? + 2;
    const text = try std.mem.concat(testing.allocator, u8, &.{ base[0..start], base[end..] });
    defer testing.allocator.free(text);
    var with = try Config.parse(testing.allocator, base, &glm53, null);
    defer with.deinit(testing.allocator);
    var without = try Config.parse(testing.allocator, text, &glm53, null);
    defer without.deinit(testing.allocator);
    try testing.expectEqualSlices(IndexerType, with.indexer_types[0..78], without.indexer_types[0..78]);
}

test "glm residents: the spec names the release's tensors at their shapes, the latent projections per head, the indexer on full layers only" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const text = try testConfigJson(testing.allocator, ",\"model.layers.0.self_attn.embed_q\":{\"bits\":8,\"group_size\":64}");
    defer testing.allocator.free(text);
    var c = try Config.parse(testing.allocator, text, &glm53, null);
    defer c.deinit(testing.allocator);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const spec = try residentSpec(arena.allocator(), &c);
    const find = struct {
        fn f(s: []const Param, name: []const u8) ?Param {
            for (s) |p| if (std.mem.eql(u8, p.name, name)) return p;
            return null;
        }
    }.f;
    try testing.expectEqualSlices(u64, &.{ 64, 512, 24 }, find(spec, "model.layers.5.self_attn.embed_q.weight").?.shapeOf());
    try testing.expectEqualSlices(u64, &.{ 64, 512, 3 }, find(spec, "model.layers.5.self_attn.embed_q.scales").?.shapeOf());
    try testing.expectEqualSlices(u64, &.{ 64, 512, 48 }, find(spec, "model.layers.0.self_attn.embed_q.weight").?.shapeOf());
    try testing.expectEqualSlices(u64, &.{ 64, 256, 64 }, find(spec, "model.layers.5.self_attn.unembed_out.weight").?.shapeOf());
    try testing.expectEqualSlices(u64, &.{ 64, 256, 8 }, find(spec, "model.layers.5.self_attn.unembed_out.biases").?.shapeOf());
    try testing.expectEqualSlices(u64, &.{ 256, 6144 }, find(spec, "model.layers.3.mlp.gate.weight").?.shapeOf());
    try testing.expectEqualSlices(u64, &.{ 64 * 256, 2048 / 8 }, find(spec, "model.layers.3.self_attn.q_b_proj.weight").?.shapeOf());
    try testing.expect(find(spec, "model.layers.6.self_attn.indexer.wq_b.weight") != null);
    try testing.expect(find(spec, "model.layers.7.self_attn.indexer.wq_b.weight") == null);
    try testing.expect(find(spec, "model.layers.2.mlp.gate_proj.weight") != null and find(spec, "model.layers.3.mlp.gate_proj.weight") == null);
    try testing.expect(find(spec, "model.layers.3.mlp.switch_mlp.gate_proj.weight") == null);
    // 3 top-level modules (7 tensors) + per layer: 2 norms + 6 projections x 3 + 2 latent norms, + 5 indexer on 21 full
    // layers, + 9 dense MLP on 3 layers or 2 router + 9 shared on 75.
    try testing.expectEqual(@as(usize, 7 + 78 * 22 + 21 * 5 + 3 * 9 + 75 * 11), spec.len);
}

test "glm residents: a pack's shards pass the spec; a missing, mistyped or extra tensor is refused by name" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    var c = try Config.parse(testing.allocator, try tinyText(), null, null);
    defer c.deinit(testing.allocator);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeResidents(testing.allocator, testing.io, tmp.dir, &c);
    var rbuf: [512]u8 = undefined;
    const dir = rbuf[0..try tmp.dir.realPath(testing.io, &rbuf)];
    var diag: Diag = .{};
    const bytes = residentBytes(testing.allocator, testing.io, dir, &c, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    try testing.expect(bytes > 0);
    // The checkpoint's view, perturbed: a spec tensor gone, one at another dtype, an extra one, a routed expert.
    var ck = try Checkpoint.openIndexed(testing.allocator, testing.io, dir, &diag);
    defer ck.deinit();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const spec = try residentSpec(arena.allocator(), &c);
    const t0 = ck.tensors.get("lm_head.scales").?;
    _ = ck.tensors.orderedRemove("lm_head.scales");
    try testing.expectError(error.TensorMissing, checkResidents(spec, &ck, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "lm_head.scales") != null);
    var t1 = t0;
    t1.dtype = .F16;
    try ck.tensors.put(ck.arena.allocator(), "lm_head.scales", t1);
    try testing.expectError(error.TensorDtype, checkResidents(spec, &ck, &diag));
    try ck.tensors.put(ck.arena.allocator(), "lm_head.scales", t0);
    try ck.tensors.put(ck.arena.allocator(), "model.layers.1.mlp.switch_mlp.gate_proj.weight", t0);
    try testing.expectError(error.ExpertsInResidents, checkResidents(spec, &ck, &diag));
    _ = ck.tensors.orderedRemove("model.layers.1.mlp.switch_mlp.gate_proj.weight");
    try ck.tensors.put(ck.arena.allocator(), "model.layers.5.self_attn.o_proj.weight", t0);
    try testing.expectError(error.TensorUnexpected, checkResidents(spec, &ck, &diag));
}

var tiny_text_buf: [8192]u8 = undefined;

fn tinyText() ![]const u8 {
    var fba = std.heap.FixedBufferAllocator.init(&tiny_text_buf);
    return tinyConfigJson(fba.allocator(), tiny_quant);
}

test "glm prompt shape: calls of 16,384 rows with a short tail joined; spans under their target; the kernels fit GLM-5.3" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = std.testing.allocator;
    var calls: std.ArrayList(u64) = .empty;
    defer calls.deinit(a);
    for ([_]struct { total: u64, want: []const u64 }{
        .{ .total = 16387, .want = &.{16387} },
        .{ .total = 18431, .want = &.{18431} },
        .{ .total = 18432, .want = &.{ 16384, 2048 } },
        .{ .total = 40000, .want = &.{ 16384, 16384, 7232 } },
    }) |cs| {
        calls.clearRetainingCapacity();
        var at: u64 = 0;
        while (at < cs.total) : (at += calls.items[calls.items.len - 1]) try calls.append(a, promptCallRows(cs.total, at));
        try std.testing.expectEqualSlices(u64, cs.want, calls.items);
        try std.testing.expect(maxPromptCallRows(cs.total) >= std.mem.max(u64, calls.items));
    }
    const text = try testConfigJson(a, "");
    defer a.free(text);
    var c = try Config.parse(a, text, &glm53, null);
    defer c.deinit(a);
    try std.testing.expect(dsaKernelsFit(&c));
    // 16K positions: 8 B a position and 204,800 B of the attention's own arrays a row.
    try std.testing.expectEqual(@as(u64, 8 * 16387 + 204800), promptSpanRowBytes(&c, 16387));
    for ([_]u64{ 16387, 262144, 1048576 }) |p| {
        const rows = promptSpanRows(&c, 16384, p);
        try std.testing.expect(rows * promptSpanRowBytes(&c, p) <= span_target_bytes and (rows + 1) * promptSpanRowBytes(&c, p) > span_target_bytes);
    }
    try std.testing.expectEqual(@as(u64, 100), promptSpanRows(&c, 100, 16387));
}
