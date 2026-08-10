//! DFlash block-drafter — generic block-parallel speculative decoding.
//!
//! A small assistant transformer drafts `block_size - 1` tokens in ONE
//! forward, conditioned on the trunk's intermediate hidden states. Mirrors
//! transformers' `DFlashTokenCandidateGenerator` + `DFlashCache` and the
//! `muse_glimmer_assistant` reference model. DFlash is a METHOD, not a
//! model-family feature: detection keys on the CONFIG CONTRACT
//! (`block_size` + `mask_token_id` + `target_layer_ids`), never on the
//! `model_type` string.
//!
//! Protocol per round (assistant borrows the trunk's embed table + lm_head):
//!   1. Context delta = trunk hiddens at the OUTPUT of layers
//!      `target_layer_ids` for the tokens the trunk just accepted,
//!      concatenated on features → `encoder.fc` → RMS norm → per-layer K/V
//!      appended to the assistant cache at those tokens' absolute positions.
//!   2. `noise_embeds` = RAW trunk embedding lookup (no norm, no scale) of
//!      `[anchor(t1), mask_token × (block_size-1)]`.
//!   3. Assistant forward: Q from the block only; K/V = cached context +
//!      fresh block K/V. Block attends bidirectionally among itself,
//!      sliding-windowed/full to context.
//!   4. Draft logits = trunk `lm_head(hidden)[:, 1:]` — the anchor position
//!      is DROPPED → block_size-1 drafts.
//!   5. Trunk verify over `[t1, drafts...]` (standard spec verify invariant);
//!      the verify forward doubles as the next round's context producer.
//!   6. Block K/V are NEVER cached (reference evicts them via
//!      `crop(-block_size)`; we append them into spare capacity and
//!      truncate back — same math, no copy).

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
// The chunked row requantizer + its packed-triple handle are shared with the
// MTP head — both sidecars shrink the SAME trunk lm_head for drafts only, and
// one requantizer with one chunking discipline is the point.
const mtp_mod = @import("mtp.zig");

const Weights = model_mod.Weights;
const ModelConfig = model_mod.ModelConfig;
const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;

pub const LayerType = enum {
    sliding_attention,
    full_attention,

    pub fn fromString(s: []const u8) !LayerType {
        if (std.mem.eql(u8, s, "sliding_attention")) return .sliding_attention;
        if (std.mem.eql(u8, s, "full_attention")) return .full_attention;
        return error.UnknownLayerType;
    }
};

pub const DflashConfig = struct {
    hidden_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    intermediate_size: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
    sliding_window: u32,
    layer_types: []LayerType, // owned
    // ── The DFlash contract fields ──
    block_size: u32,
    mask_token_id: u32,
    target_layer_ids: []u32, // owned, ascending

    pub fn deinit(self: *DflashConfig, allocator: std.mem.Allocator) void {
        allocator.free(self.layer_types);
        allocator.free(self.target_layer_ids);
    }
};

/// Does this config.json declare the DFlash contract? ALL THREE fields must
/// be present — `model_type` (`*_assistant`) is only the discovery-level
/// "this is a drafter" signal, never the DFlash detection.
pub fn isDflashConfigJson(root: std.json.ObjectMap) bool {
    return root.get("block_size") != null and
        root.get("mask_token_id") != null and
        root.get("target_layer_ids") != null;
}

/// Read `<dir>/config.json` and answer whether it declares DFlash. Any
/// read/parse failure is a quiet false — the caller falls through to the
/// gemma drafter loader, whose own errors are the user-facing ones.
pub fn probeIsDflash(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    const content = readConfigFile(io, allocator, model_dir) catch return false;
    defer allocator.free(content);
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, content, .{}) catch return false;
    defer parsed.deinit();
    if (parsed.value != .object) return false;
    return isDflashConfigJson(parsed.value.object);
}

fn readConfigFile(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);
    const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer file.close(io);
    var read_buf: [4096]u8 = undefined;
    var reader_state = file.reader(io, &read_buf);
    return try reader_state.interface.allocRemaining(allocator, .limited(1 << 20));
}

/// Conventional in-checkpoint home for a DFlash assistant: a subdirectory of
/// the model's own directory declaring the config contract. Mirrors
/// `mtp.resolveMtpSource`'s sidecar-or-in-checkpoint shape, and it is what
/// turns the drafter from a LAUNCH-flag dependency into a LOAD-time one — a
/// hot model switch brings its own drafter, no pairing table decides which
/// sidecar goes with which checkpoint, and a mismatched pair is unbuildable.
pub const IN_DIR_SUBDIR = "drafter";

/// `<model_dir>/drafter` when it declares the DFlash contract, else null.
/// Caller owns the returned path. Only ever consulted when no explicit
/// `--drafter` was given — an explicit flag always wins, so an external
/// sidecar can still be pointed at a merged checkpoint.
pub fn resolveInDirDrafter(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) ?[]u8 {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return null;
    const path = std.fs.path.join(allocator, &.{ model_dir, IN_DIR_SUBDIR }) catch return null;
    if (!probeIsDflash(io, allocator, path)) {
        allocator.free(path);
        return null;
    }
    return path;
}

pub fn parseConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !DflashConfig {
    const content = try readConfigFile(io, allocator, model_dir);
    defer allocator.free(content);
    return parseConfigFromJson(allocator, content);
}

pub fn parseConfigFromJson(allocator: std.mem.Allocator, content: []const u8) !DflashConfig {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.NotDflashConfig;
    const root = parsed.value.object;

    if (!isDflashConfigJson(root)) return error.NotDflashConfig;

    const block_size: u32 = try jsonU32(root.get("block_size").?);
    if (block_size < 2) return error.InvalidDflashBlockSize;
    const mask_token_id: u32 = try jsonU32(root.get("mask_token_id").?);

    const tl_val = root.get("target_layer_ids").?;
    if (tl_val != .array or tl_val.array.items.len == 0) return error.InvalidDflashTargetLayers;
    const target_layer_ids = try allocator.alloc(u32, tl_val.array.items.len);
    errdefer allocator.free(target_layer_ids);
    for (tl_val.array.items, 0..) |elem, i| {
        target_layer_ids[i] = try jsonU32(elem);
        // The encoder concatenates in list order and the fc weight is trained
        // against it; a non-ascending list is a converter bug worth naming.
        if (i > 0 and target_layer_ids[i] <= target_layer_ids[i - 1]) return error.InvalidDflashTargetLayers;
    }

    const hidden_size: u32 = try jsonU32(root.get("hidden_size") orelse return error.IncompleteDflashConfig);
    const num_layers: u32 = try jsonU32(root.get("num_hidden_layers") orelse return error.IncompleteDflashConfig);
    const n_heads: u32 = try jsonU32(root.get("num_attention_heads") orelse return error.IncompleteDflashConfig);
    const kv_heads: u32 = if (root.get("num_key_value_heads")) |v| try jsonU32(v) else n_heads;
    const head_dim: u32 = try jsonU32(root.get("head_dim") orelse return error.IncompleteDflashConfig);
    const intermediate: u32 = try jsonU32(root.get("intermediate_size") orelse return error.IncompleteDflashConfig);
    const eps: f32 = jsonFloat(root.get("rms_norm_eps") orelse return error.IncompleteDflashConfig);
    const sliding: u32 = if (root.get("sliding_window")) |v| try jsonU32(v) else 0;

    var rope_theta: f32 = 10000.0;
    if (root.get("rope_parameters")) |rp| if (rp == .object) {
        if (rp.object.get("rope_theta")) |t| rope_theta = jsonFloat(t);
    };

    const lt_val = root.get("layer_types") orelse return error.IncompleteDflashConfig;
    if (lt_val != .array or lt_val.array.items.len != num_layers) return error.LayerTypesLengthMismatch;
    const layer_types = try allocator.alloc(LayerType, num_layers);
    errdefer allocator.free(layer_types);
    for (lt_val.array.items, 0..) |elem, i| {
        if (elem != .string) return error.InvalidLayerTypeEntry;
        layer_types[i] = try LayerType.fromString(elem.string);
        if (layer_types[i] == .sliding_attention and sliding == 0) return error.IncompleteDflashConfig;
    }

    return DflashConfig{
        .hidden_size = hidden_size,
        .num_hidden_layers = num_layers,
        .num_attention_heads = n_heads,
        .num_key_value_heads = kv_heads,
        .head_dim = head_dim,
        .intermediate_size = intermediate,
        .rms_norm_eps = eps,
        .rope_theta = rope_theta,
        .sliding_window = sliding,
        .layer_types = layer_types,
        .block_size = block_size,
        .mask_token_id = mask_token_id,
        .target_layer_ids = target_layer_ids,
    };
}

fn jsonU32(v: std.json.Value) !u32 {
    return switch (v) {
        .integer => |i| if (i >= 0 and i <= std.math.maxInt(u32)) @intCast(i) else error.InvalidDflashConfigValue,
        else => error.InvalidDflashConfigValue,
    };
}

fn jsonFloat(v: std.json.Value) f32 {
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => 0.0,
    };
}

/// Every `target_layer_ids` entry must name an existing trunk layer — the
/// capture seam retains `hidden_states[i+1]`, so `i` must be < trunk depth.
/// Named error at load, not warmup death.
pub fn validateTargetLayers(ids: []const u32, trunk_num_layers: u32) !void {
    for (ids) |id| {
        if (id >= trunk_num_layers) return error.DflashTargetLayerOutOfRange;
    }
}

/// Widest verify a machine without the NAX m16 tile serves well: the block
/// IS the verify width, and `transformer.vqmmLaneFor`'s split-K lane stops at
/// M=7. Past it MLX's own 32x32-tiled `qmm_splitk` takes over and a trunk
/// forward jumps from ~1.3x a serial step to ~4.2x, for barely more accepted
/// tokens. Measured on Muse 4-bit / applegpu_g16s, 160-token generations,
/// serial reference in the same boot (28.2 tok/s):
///   block 16 → 0.92x   8 → 1.16x   7 → 1.43x   6 → 1.79x   5 → 1.97x
///   4 → 1.81x   3 → 1.77x
/// The block is a REQUEST cost, not a training constant — the assistant
/// drafts against the same mask token at any width. NAX-capable machines
/// (M5-class) have a real M 8..16 lane and keep the checkpoint's block.
pub const NO_WIDE_LANE_BLOCK_CAP: u32 = 5;

/// Resolve the effective block size: the config's value, capped by what the
/// machine's verify lanes serve, then clamped DOWNWARD by an explicit
/// `--draft-block-size` (never raised past the config — the assistant was
/// trained at its config block). Floor 2 (1 draft + t1).
pub fn resolveBlockSize(config_block: u32, cli_block: u32, cli_explicit: bool, wide_verify_lane: bool) u32 {
    const base = if (cli_explicit)
        @min(config_block, cli_block)
    else if (wide_verify_lane)
        config_block
    else
        @min(config_block, NO_WIDE_LANE_BLOCK_CAP);
    return @max(base, 2);
}

/// True when this machine has a verify lane for widths past the split-K
/// ceiling (the NAX m16 tile, M5-class + macOS >= 26.2).
pub fn wideVerifyLaneAvailable() bool {
    return transformer_mod.naxLaneEnvEnabled() and transformer_mod.verifyQmmNaxAvailable();
}

// ── Weight precision ──

/// Default affine width the dense bf16 checkpoint is quantized to at load.
/// The assistant weight read is the per-round cost the block forward is
/// bound by (5.11 GB bf16 on the Muse sidecar); 8-bit halves it. Only DRAFT
/// quality rides on it — the trunk verify is exact either way, so the worst
/// a bad width can do is cost acceptance.
pub const DEFAULT_QUANT_BITS: u32 = 8;

/// Draft-only lm_head width — DEFAULT OFF (drafts use the trunk's own head).
/// A narrower head shrinks a read the round barely notices and pays for it in
/// ACCEPTANCE, which is the scarce resource: measured on Muse 4-bit at the
/// resolved block (5), 160-token generations, same serial reference —
///   3-bit draft head: 55.4 tok/s, 2.19 accepted/round
///   trunk head:       57.8 tok/s, 2.37 accepted/round
/// The lever was defaulted on at block 16, where the head was 10 ms of a
/// 175 ms round; at block 5 it is under 2 ms of ~60 and the acceptance it
/// costs is worth more than the bytes it saves. `MLX_SERVE_DFLASH_DRAFT_HEAD_BITS`
/// still selects a width for the A/B.
pub const DEFAULT_DRAFT_HEAD_BITS: u32 = 0;
/// Preferred affine group size, narrowed to 32 when 64 does not divide the
/// contraction dim, dense when neither does.
pub const QUANT_GROUP: u32 = 64;

/// `MLX_SERVE_DFLASH_QUANT_BITS`: absent → `DEFAULT_QUANT_BITS`, a supported
/// affine width → that, anything else ("0", "off") → dense bf16.
pub fn quantBitsFromEnv() u32 {
    const p = std.c.getenv("MLX_SERVE_DFLASH_QUANT_BITS") orelse return DEFAULT_QUANT_BITS;
    const v = std.fmt.parseInt(u32, std.mem.span(p), 10) catch return 0;
    return switch (v) {
        2, 3, 4, 5, 6, 8 => v,
        else => 0,
    };
}

/// Widest supported group that divides the contraction dim, or null when the
/// weight cannot be affine-quantized at all (that weight stays dense).
pub fn quantGroupFor(in_features: u32) ?u32 {
    if (in_features == 0) return null;
    if (in_features % QUANT_GROUP == 0) return QUANT_GROUP;
    if (in_features % 32 == 0) return 32;
    return null;
}

/// One assistant linear. Dense weights are pre-transposed to `[in, out]` and
/// contracted with a plain matmul (drafter.zig's convention); quantized ones
/// keep the checkpoint's `[out, in]` packing and ride `mlx_quantized_matmul`
/// with transpose=true, exactly like the trunk. `bits == 0` IS the dense
/// discriminator — never probe the scales handle.
pub const DflashLinear = struct {
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    bits: u32 = 0,
    group_size: u32 = 0,

    pub fn isQuantized(self: *const DflashLinear) bool {
        return self.bits != 0;
    }

    pub fn deinit(self: *DflashLinear) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
    }

    pub fn apply(self: *const DflashLinear, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var out = mlx.mlx_array_new();
        if (!self.isQuantized()) {
            try mlx.check(mlx.mlx_matmul(&out, x, self.w, s));
            return out;
        }
        try mlx.check(mlx.mlx_quantized_matmul(
            &out,
            x,
            self.w,
            self.scales,
            self.biases,
            true,
            mlx.mlx_optional_int.some(@intCast(self.group_size)),
            mlx.mlx_optional_int.some(@intCast(self.bits)),
            "affine",
            s,
        ));
        return out;
    }

    fn appendEval(self: *const DflashLinear, vec: mlx.mlx_vector_array) void {
        _ = mlx.mlx_vector_array_append_value(vec, self.w);
        if (!self.isQuantized()) return;
        _ = mlx.mlx_vector_array_append_value(vec, self.scales);
        _ = mlx.mlx_vector_array_append_value(vec, self.biases);
    }
};

// ── Model ──

/// Per-layer assistant weights.
pub const DflashLayer = struct {
    layer_type: LayerType,

    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,

    q: DflashLinear, // [hidden → n_heads*head_dim]
    q_norm: mlx.mlx_array, // [head_dim]
    k: DflashLinear, // [hidden → kv_heads*head_dim]
    k_norm: mlx.mlx_array, // [head_dim]
    v: DflashLinear, // [hidden → kv_heads*head_dim]
    o: DflashLinear, // [n_heads*head_dim → hidden]

    gate: DflashLinear, // [hidden → intermediate]
    up: DflashLinear, // [hidden → intermediate]
    down: DflashLinear, // [intermediate → hidden]

    fn deinit(self: *DflashLayer) void {
        _ = mlx.mlx_array_free(self.input_norm);
        _ = mlx.mlx_array_free(self.post_attn_norm);
        _ = mlx.mlx_array_free(self.q_norm);
        _ = mlx.mlx_array_free(self.k_norm);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.gate.deinit();
        self.up.deinit();
        self.down.deinit();
    }

    fn appendEval(self: *const DflashLayer, vec: mlx.mlx_vector_array) void {
        _ = mlx.mlx_vector_array_append_value(vec, self.input_norm);
        _ = mlx.mlx_vector_array_append_value(vec, self.post_attn_norm);
        _ = mlx.mlx_vector_array_append_value(vec, self.q_norm);
        _ = mlx.mlx_vector_array_append_value(vec, self.k_norm);
        for ([_]*const DflashLinear{ &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down }) |lin| {
            lin.appendEval(vec);
        }
    }
};

pub const DflashModel = struct {
    config: DflashConfig,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,

    fc: DflashLinear, // encoder.fc [n_targets*hidden → hidden]
    enc_norm: mlx.mlx_array, // encoder.output_norm_enc [hidden]
    final_norm: mlx.mlx_array, // norm [hidden]
    layers: []DflashLayer,

    /// Optional DRAFT-ONLY low-bit lm_head, requantized from the trunk's at
    /// bind time (`MLX_SERVE_DFLASH_DRAFT_HEAD_BITS`, default 3, 0 disables).
    /// Only the block's draft argmax projects through it — VERIFICATION is a
    /// trunk forward and never touches it, so the emitted distribution is
    /// untouched; drafts just read ~⅓ of the bytes of a full-vocab head.
    draft_head: ?mtp_mod.QLinear = null,
    draft_head_bits: u32 = 0,
    draft_head_group: u32 = 0,

    pub fn deinit(self: *DflashModel) void {
        const allocator = self.allocator;
        if (self.draft_head) |*dh| dh.deinit();
        self.fc.deinit();
        _ = mlx.mlx_array_free(self.enc_norm);
        _ = mlx.mlx_array_free(self.final_norm);
        for (self.layers) |*lw| lw.deinit();
        allocator.free(self.layers);
        self.config.deinit(allocator);
    }

    /// Validate compatibility with the target trunk. The assistant borrows
    /// the trunk's embedding table and lm_head, so hidden size and token-id
    /// space must line up; the capture seam lives in the STANDARD forward
    /// path only (v1), so module-owned / MoE / hybrid trunks are refused by
    /// name at load rather than silently drafting from empty captures.
    pub fn bind(self: *DflashModel, target: *Transformer) !void {
        if (self.config.hidden_size != target.config.hidden_size) {
            log.err("[dflash] hidden_size mismatch: assistant={d}, target={d}\n", .{
                self.config.hidden_size, target.config.hidden_size,
            });
            return error.DflashTargetMismatch;
        }
        if (self.config.mask_token_id >= target.config.vocab_size) {
            log.err("[dflash] mask_token_id {d} outside target vocab {d}\n", .{
                self.config.mask_token_id, target.config.vocab_size,
            });
            return error.DflashTargetMismatch;
        }
        validateTargetLayers(self.config.target_layer_ids, target.config.num_hidden_layers) catch |err| {
            log.err("[dflash] target_layer_ids {any} out of range for {d}-layer target\n", .{
                self.config.target_layer_ids, target.config.num_hidden_layers,
            });
            return err;
        };
        if (!target.usesStandardForward()) {
            log.err("[dflash] target arch '{s}' does not run the standard forward path (capture seam v1)\n", .{
                target.config.model_type,
            });
            return error.DflashTargetMismatch;
        }
        self.buildDraftHead(target, draftHeadBitsFromEnv()) catch |err| {
            log.warn("[dflash] draft lm_head build failed ({s}) — drafts use the trunk head\n", .{@errorName(err)});
        };
    }

    /// `bind` with the draft-head width passed explicitly (0 = no draft head).
    pub fn bindWithDraftBits(self: *DflashModel, target: *Transformer, bits: u32) !void {
        try self.bind(target);
        if (self.draft_head) |*dh| {
            dh.deinit();
            self.draft_head = null;
            self.draft_head_bits = 0;
            self.draft_head_group = 0;
        }
        try self.buildDraftHead(target, bits);
    }

    /// `MLX_SERVE_DFLASH_DRAFT_HEAD_BITS`: absent → DEFAULT_DRAFT_HEAD_BITS,
    /// a supported affine width → that, anything else ("0", "off") → disabled.
    pub fn draftHeadBitsFromEnv() u32 {
        const p = std.c.getenv("MLX_SERVE_DFLASH_DRAFT_HEAD_BITS") orelse return DEFAULT_DRAFT_HEAD_BITS;
        const v = std.fmt.parseInt(u32, std.mem.span(p), 10) catch return 0;
        return switch (v) {
            2, 3, 4, 6, 8 => v,
            else => 0,
        };
    }

    /// Requantize the trunk lm_head down to `draftHeadBitsFromEnv()` for the
    /// draft projection only. A quantized trunk head is re-encoded from its
    /// TRUE per-weight params (a mixed checkpoint's head routinely differs
    /// from the trunk global); a dense bf16 head is quantized outright. Never
    /// built when it would not shrink the read.
    fn buildDraftHead(self: *DflashModel, target: *Transformer, bits: u32) !void {
        if (bits == 0) {
            log.info("[dflash] draft lm_head: trunk head (no requantization)\n", .{});
            return;
        }
        // A dense head has a NULL scales handle — every quant-param resolver
        // dereferences it, so the dense arm must be decided first.
        const dense_head = target.lm_head_s.ctx == null;
        const head_qp: transformer_mod.QuantParams = if (dense_head)
            .{ .bits = 0, .group_size = 0, .mode = .affine }
        else
            transformer_mod.computeQuantParams(
                &target.config,
                target.lm_head_w,
                target.lm_head_s,
                target.config.hidden_size,
            );
        // 16 bits is what a dense bf16 head costs per weight.
        const src_bits: u32 = if (dense_head) 16 else head_qp.bits;
        if (bits >= src_bits) return; // no byte saving over the trunk head
        const group = quantGroupFor(target.config.hidden_size) orelse return;

        var dh = try mtp_mod.requantizeRows(
            self.s,
            target.lm_head_w,
            target.lm_head_s,
            target.lm_head_b,
            head_qp.group_size,
            head_qp.bits,
            head_qp.mode.cstr(),
            group,
            bits,
            32768,
        );
        errdefer dh.deinit();
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, dh.w);
            _ = mlx.mlx_vector_array_append_value(eval_vec, dh.s);
            _ = mlx.mlx_vector_array_append_value(eval_vec, dh.b);
            try mlx.check(mlx.mlx_eval(eval_vec));
        }
        self.draft_head = dh;
        self.draft_head_bits = bits;
        self.draft_head_group = group;
        log.info("[dflash] draft-only lm_head requantized to {d}-bit/gs{d}\n", .{ bits, group });
    }

    /// Draft logits for the block hidden: the low-bit draft head when one was
    /// built, else the trunk's own head. Both are the bare Linear — softcap /
    /// output_multiplier are monotone, so draft argmax is unaffected.
    pub fn draftLogits(self: *const DflashModel, target: *const Transformer, x: mlx.mlx_array) !mlx.mlx_array {
        if (self.draft_head) |*dh| {
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_quantized_matmul(
                &out,
                x,
                dh.w,
                dh.s,
                dh.b,
                true,
                mlx.mlx_optional_int.some(@intCast(self.draft_head_group)),
                mlx.mlx_optional_int.some(@intCast(self.draft_head_bits)),
                "affine",
                self.s,
            ));
            return out;
        }
        return target.lmHeadForDraft(x);
    }
};

// ── Weight loading ──

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[dflash] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingDflashWeight;
    };
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, arr));
    return owned;
}

/// Load `<prefix>.weight` as an assistant linear. A checkpoint that already
/// ships `<prefix>.scales` is served packed as-is (affine only — its true
/// params are solved from the packed geometry, never assumed); a dense bf16
/// weight is quantized to `bits` when the contraction dim allows it and
/// pre-transposed for a plain matmul otherwise.
fn loadLinear(
    w: *const Weights,
    prefix: []const u8,
    in_features: u32,
    bits: u32,
    s: mlx.mlx_stream,
) !DflashLinear {
    var key_buf: [256]u8 = undefined;
    const scales_key = try std.fmt.bufPrint(&key_buf, "{s}.scales", .{prefix});
    if (w.get(scales_key)) |_| {
        var kb: [256]u8 = undefined;
        var out = DflashLinear{
            .w = try ownWeight(w, try std.fmt.bufPrint(&kb, "{s}.weight", .{prefix})),
            .scales = try ownWeight(w, try std.fmt.bufPrint(&kb, "{s}.scales", .{prefix})),
            .biases = try ownWeight(w, try std.fmt.bufPrint(&kb, "{s}.biases", .{prefix})),
        };
        errdefer out.deinit();
        const qp = transformer_mod.affineParamsFromGeometry(out.w, out.scales, in_features) orelse {
            log.err("[dflash] {s}: packed geometry is not affine-servable (in={d})\n", .{ prefix, in_features });
            return error.UnsupportedDflashQuant;
        };
        out.bits = qp.bits;
        out.group_size = qp.group_size;
        return out;
    }

    var kb: [256]u8 = undefined;
    const raw = try ownWeight(w, try std.fmt.bufPrint(&kb, "{s}.weight", .{prefix}));
    defer _ = mlx.mlx_array_free(raw);

    if (bits != 0) {
        if (quantGroupFor(in_features)) |group| return quantizeDense(raw, bits, group, s);
    }
    var transposed = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(transposed);
    const perm = [_]c_int{ 1, 0 };
    try mlx.check(mlx.mlx_transpose_axes(&transposed, raw, &perm, 2, s));
    return .{ .w = transposed, .scales = mlx.mlx_array_new(), .biases = mlx.mlx_array_new() };
}

/// Affine-quantize a dense `[out, in]` weight in place of a transpose — the
/// packed layout is exactly what `mlx_quantized_matmul(transpose=true)` reads.
fn quantizeDense(raw: mlx.mlx_array, bits: u32, group: u32, s: mlx.mlx_stream) !DflashLinear {
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(
        &triple,
        raw,
        mlx.mlx_optional_int.some(@intCast(group)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale
        s,
    ));
    if (mlx.mlx_vector_array_size(triple) != 3) return error.UnexpectedQuantizeOutput;
    var out = DflashLinear{
        .w = mlx.mlx_array_new(),
        .scales = mlx.mlx_array_new(),
        .biases = mlx.mlx_array_new(),
        .bits = bits,
        .group_size = group,
    };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&out.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.scales, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.biases, triple, 2));
    return out;
}

/// Load a DFlash assistant from `model_dir`. After loading, call
/// `bind(target)` before serving.
pub fn loadDflash(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []const u8,
) !DflashModel {
    return loadDflashQuant(io, allocator, s, model_dir, quantBitsFromEnv());
}

/// `loadDflash` with the load-time quantization width passed explicitly
/// (0 = dense bf16). Tests drive both arms from here rather than through the
/// environment — a skipped arm reads as a pass.
pub fn loadDflashQuant(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []const u8,
    bits: u32,
) !DflashModel {
    var cfg = try parseConfig(io, allocator, model_dir);
    errdefer cfg.deinit(allocator);

    var weights = try model_mod.loadWeights(io, allocator, model_dir);
    defer weights.deinit();

    const hidden = cfg.hidden_size;
    const q_out = cfg.num_attention_heads * cfg.head_dim;
    const fc_in: u32 = @intCast(cfg.target_layer_ids.len * hidden);

    var fc = try loadLinear(&weights, "encoder.fc", fc_in, bits, s);
    errdefer fc.deinit();
    const enc_norm = try ownWeight(&weights, "encoder.output_norm_enc.weight");
    errdefer _ = mlx.mlx_array_free(enc_norm);
    const final_norm = try ownWeight(&weights, "norm.weight");
    errdefer _ = mlx.mlx_array_free(final_norm);

    const layers = try allocator.alloc(DflashLayer, cfg.num_hidden_layers);
    errdefer allocator.free(layers);
    var layers_inited: u32 = 0;
    errdefer {
        var i: u32 = 0;
        while (i < layers_inited) : (i += 1) layers[i].deinit();
    }

    var key_buf: [256]u8 = undefined;
    var li: u32 = 0;
    while (li < cfg.num_hidden_layers) : (li += 1) {
        layers[li] = .{
            .layer_type = cfg.layer_types[li],
            .input_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.input_layernorm.weight", .{li})),
            .post_attn_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.post_attention_layernorm.weight", .{li})),
            .q = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.q_proj", .{li}), hidden, bits, s),
            .q_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.q_norm.weight", .{li})),
            .k = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.k_proj", .{li}), hidden, bits, s),
            .k_norm = try ownWeight(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.k_norm.weight", .{li})),
            .v = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.v_proj", .{li}), hidden, bits, s),
            .o = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.o_proj", .{li}), q_out, bits, s),
            .gate = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.gate_proj", .{li}), hidden, bits, s),
            .up = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.up_proj", .{li}), hidden, bits, s),
            .down = try loadLinear(&weights, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.down_proj", .{li}), cfg.intermediate_size, bits, s),
        };
        layers_inited += 1;
    }

    // Force-eval all weights so serve time never faults a lazy transpose or
    // an un-materialized load-time quantization.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        fc.appendEval(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, enc_norm);
        _ = mlx.mlx_vector_array_append_value(eval_vec, final_norm);
        for (layers) |*lw| lw.appendEval(eval_vec);
        _ = mlx.mlx_eval(eval_vec);
    }

    if (fc.isQuantized()) {
        log.info("[dflash] loaded {d} layers, hidden={d}, block_size={d}, targets={any}, weights={d}-bit/gs{d}\n", .{
            cfg.num_hidden_layers, cfg.hidden_size, cfg.block_size, cfg.target_layer_ids, fc.bits, fc.group_size,
        });
    } else {
        log.info("[dflash] loaded {d} layers, hidden={d}, block_size={d}, targets={any}, weights=dense bf16\n", .{
            cfg.num_hidden_layers, cfg.hidden_size, cfg.block_size, cfg.target_layer_ids,
        });
    }

    return DflashModel{
        .config = cfg,
        .allocator = allocator,
        .s = s,
        .fc = fc,
        .enc_norm = enc_norm,
        .final_norm = final_norm,
        .layers = layers,
    };
}

// ── Per-request context cache ──

/// The assistant's context K/V, one entry per ASSISTANT layer, dense.
/// Grows with committed trunk tokens; block K/V transit through spare
/// capacity and are truncated straight back out (never cached). Invariant
/// at round start: `base_pos + cache.step == trunk cache.step`.
pub const DflashCtx = struct {
    cache: KVCache,
    /// Absolute trunk position of cache index 0. Nonzero on hot-prefix-cache
    /// reuse, where only the freshly forwarded tail produced captures — a
    /// late-starting context is self-consistent (sliding-window semantics,
    /// same rule as the MTP history).
    base_pos: usize,

    pub fn init(allocator: std.mem.Allocator, model: *const DflashModel, base_pos: usize) !DflashCtx {
        return .{
            .cache = try KVCache.init(allocator, model.config.num_hidden_layers),
            .base_pos = base_pos,
        };
    }

    pub fn deinit(self: *DflashCtx) void {
        self.cache.deinit();
    }

    /// Absolute trunk position one past the last cached context token.
    pub fn absLen(self: *const DflashCtx) usize {
        return self.base_pos + self.cache.step;
    }

    /// Collect cache buffers for a batched eval (prefill-chunk discipline:
    /// materialize appended context so the chunk's activation graph frees).
    pub fn appendEvalArrays(self: *DflashCtx, vec: mlx.mlx_vector_array) void {
        for (self.cache.entries) |*entry| {
            if (!entry.initialized) continue;
            _ = mlx.mlx_vector_array_append_value(vec, entry.keys);
            _ = mlx.mlx_vector_array_append_value(vec, entry.values);
        }
    }
};

// ── Forward helpers ──

inline fn rmsNormFn(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

/// silu(gate) * up.
fn swiglu(gate: mlx.mlx_array, up: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
    var silu_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu_out);
    try mlx.check(mlx.mlx_multiply(&silu_out, gate, sig, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, silu_out, up, s));
    return out;
}

/// Project x through `w_t`, reshape to heads, apply optional per-head RMS
/// norm, transpose to `[B, heads, L, hd]`, RoPE at `offset`.
fn projectHeads(
    x: mlx.mlx_array,
    lin: *const DflashLinear,
    norm_w: mlx.mlx_array,
    n_heads: u32,
    head_dim: u32,
    eps: f32,
    theta: f32,
    rope_offset: usize,
    apply_rope: bool,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const proj = try lin.apply(x, s);
    defer _ = mlx.mlx_array_free(proj);
    const xsh = mlx.getShape(x);
    const hs = [_]c_int{ xsh[0], xsh[1], @intCast(n_heads), @intCast(head_dim) };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, proj, &hs, 4, s));

    const normed = try rmsNormFn(reshaped, norm_w, eps, s);
    defer _ = mlx.mlx_array_free(normed);

    const perm = [_]c_int{ 0, 2, 1, 3 };
    var transposed = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&transposed, normed, &perm, 4, s));
    if (!apply_rope) return transposed;
    defer _ = mlx.mlx_array_free(transposed);

    var roped = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rope(
        &roped,
        transposed,
        @intCast(head_dim),
        false,
        .{ .value = theta, .has_value = true },
        1.0,
        @intCast(rope_offset),
        .{ .ctx = null },
        s,
    ));
    return roped;
}

/// Additive attention bias `[1, 1, q_len, kv_len]` for one assistant layer,
/// or null when nothing is masked. KV layout: context at absolute positions
/// `[base_pos, base_pos + ctx_len)` followed by the block at
/// `[anchor_pos, anchor_pos + q_len)`. Sliding layers: query q sees key k iff
/// `|q - k| < window` (block queries sit at/after every context position, so
/// "bidirectional sliding" reduces to a back-window over context plus full
/// visibility inside the block whenever `q_len <= window`). Full layers see
/// everything → null.
fn buildBlockMask(
    layer_type: LayerType,
    base_pos: usize,
    ctx_len: usize,
    anchor_pos: usize,
    q_len: u32,
    window: u32,
    s: mlx.mlx_stream,
) !?mlx.mlx_array {
    if (layer_type == .full_attention) return null;
    std.debug.assert(q_len <= window);
    // No context row falls outside the LAST query's back-window → nothing masked.
    const max_dist = anchor_pos + q_len - 1 - base_pos;
    if (max_dist < window) return null;

    // Absolute key positions: [ctx…, block…].
    const kv_len = ctx_len + q_len;
    var k_abs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_abs);
    {
        var ctx_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ctx_pos);
        try mlx.check(mlx.mlx_arange(&ctx_pos, @floatFromInt(base_pos), @floatFromInt(base_pos + ctx_len), 1, .float32, s));
        var blk_pos = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(blk_pos);
        try mlx.check(mlx.mlx_arange(&blk_pos, @floatFromInt(anchor_pos), @floatFromInt(anchor_pos + q_len), 1, .float32, s));
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, ctx_pos);
        _ = mlx.mlx_vector_array_append_value(vec, blk_pos);
        try mlx.check(mlx.mlx_concatenate_axis(&k_abs, vec, 0, s));
    }
    var q_abs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_abs);
    try mlx.check(mlx.mlx_arange(&q_abs, @floatFromInt(anchor_pos), @floatFromInt(anchor_pos + q_len), 1, .float32, s));
    var q_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_col);
    const q_shape = [_]c_int{ @intCast(q_len), 1 };
    try mlx.check(mlx.mlx_reshape(&q_col, q_abs, &q_shape, 2, s));

    // diff = q - k; allowed iff |diff| < window.
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, q_col, k_abs, s));
    var abs_diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(abs_diff);
    try mlx.check(mlx.mlx_abs(&abs_diff, diff, s));
    const win_arr = mlx.mlx_array_new_float(@floatFromInt(window));
    defer _ = mlx.mlx_array_free(win_arr);
    var allowed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(allowed);
    try mlx.check(mlx.mlx_less(&allowed, abs_diff, win_arr, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    const ninf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(ninf);
    var bias_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias_f32);
    try mlx.check(mlx.mlx_where(&bias_f32, allowed, zero, ninf, s));

    var bias_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bias_bf16);
    try mlx.check(mlx.mlx_astype(&bias_bf16, bias_f32, .bfloat16, s));

    var bias_4d = mlx.mlx_array_new();
    const out_shape = [_]c_int{ 1, 1, @intCast(q_len), @intCast(kv_len) };
    try mlx.check(mlx.mlx_reshape(&bias_4d, bias_bf16, &out_shape, 4, s));
    return bias_4d;
}

// ── Context append + block forward ──

/// Project trunk captures through the encoder and append per-layer K/V for
/// `n` new context tokens at absolute positions `[first_pos, first_pos+n)`.
/// `captures` are the trunk layer OUTPUTS at `target_layer_ids`, in config
/// order, each `[1, n, hidden]`. Purely lazy — caller decides when to eval
/// (`ctx.appendEvalArrays`).
pub fn appendContext(
    model: *const DflashModel,
    ctx: *DflashCtx,
    captures: []const mlx.mlx_array,
    first_pos: usize,
) !void {
    const s = model.s;
    const cfg = &model.config;
    std.debug.assert(first_pos == ctx.absLen()); // contiguity — no holes

    const enc = try encodeContext(model, captures);
    defer _ = mlx.mlx_array_free(enc);

    for (model.layers, 0..) |*lw, li| {
        const k = try projectHeads(enc, &lw.k, lw.k_norm, cfg.num_key_value_heads, cfg.head_dim, cfg.rms_norm_eps, cfg.rope_theta, first_pos, true, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try projectHeadsNoNorm(enc, &lw.v, cfg.num_key_value_heads, cfg.head_dim, s);
        defer _ = mlx.mlx_array_free(v);
        _ = try ctx.cache.update(@intCast(li), k, v, s, 0);
    }
}

/// Encoder projection: concatenate the trunk captures on features →
/// `encoder.fc` → RMS norm. Returns `[1, n, hidden]`, caller frees.
pub fn encodeContext(model: *const DflashModel, captures: []const mlx.mlx_array) !mlx.mlx_array {
    const s = model.s;
    std.debug.assert(captures.len == model.config.target_layer_ids.len);
    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (captures) |c| _ = mlx.mlx_vector_array_append_value(vec, c);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 2, s));
    }
    const projected = try model.fc.apply(cat, s);
    defer _ = mlx.mlx_array_free(projected);
    return rmsNormFn(projected, model.enc_norm, model.config.rms_norm_eps, s);
}

/// V path: project + reshape + transpose, no norm, no RoPE.
fn projectHeadsNoNorm(
    x: mlx.mlx_array,
    lin: *const DflashLinear,
    n_heads: u32,
    head_dim: u32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const proj = try lin.apply(x, s);
    defer _ = mlx.mlx_array_free(proj);
    const xsh = mlx.getShape(x);
    const hs = [_]c_int{ xsh[0], xsh[1], @intCast(n_heads), @intCast(head_dim) };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, proj, &hs, 4, s));
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var transposed = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&transposed, reshaped, &perm, 4, s));
    return transposed;
}

/// One assistant forward over the noise block. `noise_embeds` is the RAW
/// trunk embedding lookup `[1, q_len, hidden]` of `[anchor, mask…]`;
/// `anchor_pos` is the anchor's absolute position (== trunk `cache.step`).
/// Returns the post-final-norm hidden `[1, q_len, hidden]` (caller frees;
/// the caller projects through the trunk lm_head and DROPS row 0). Block
/// K/V transit through the cache's spare capacity and are truncated back
/// out before returning — the context cache is unchanged.
pub fn forwardBlock(
    model: *const DflashModel,
    ctx: *DflashCtx,
    noise_embeds: mlx.mlx_array,
    anchor_pos: usize,
) !mlx.mlx_array {
    const s = model.s;
    const cfg = &model.config;
    const q_len_c = mlx.getShape(noise_embeds)[1];
    const q_len: u32 = @intCast(q_len_c);
    std.debug.assert(anchor_pos == ctx.absLen()); // block starts one past context
    const ctx_len = ctx.cache.step;
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.head_dim)));
    const perm_back = [_]c_int{ 0, 2, 1, 3 };
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);

    var x = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&x, noise_embeds));
    errdefer _ = mlx.mlx_array_free(x);

    for (model.layers, 0..) |*lw, li| {
        const normed = try rmsNormFn(x, lw.input_norm, cfg.rms_norm_eps, s);
        defer _ = mlx.mlx_array_free(normed);

        const q = try projectHeads(normed, &lw.q, lw.q_norm, cfg.num_attention_heads, cfg.head_dim, cfg.rms_norm_eps, cfg.rope_theta, anchor_pos, true, s);
        defer _ = mlx.mlx_array_free(q);
        const bk = try projectHeads(normed, &lw.k, lw.k_norm, cfg.num_key_value_heads, cfg.head_dim, cfg.rms_norm_eps, cfg.rope_theta, anchor_pos, true, s);
        defer _ = mlx.mlx_array_free(bk);
        const bv = try projectHeadsNoNorm(normed, &lw.v, cfg.num_key_value_heads, cfg.head_dim, s);
        defer _ = mlx.mlx_array_free(bv);

        // Append block K/V into spare capacity; the view spans ctx + block.
        const view = try ctx.cache.update(@intCast(li), bk, bv, s, 0);

        const mask = try buildBlockMask(lw.layer_type, ctx.base_pos, ctx_len, anchor_pos, q_len, cfg.sliding_window, s);
        defer if (mask) |m| {
            _ = mlx.mlx_array_free(m);
        };

        var attn_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_out);
        if (mask) |m| {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q, view.k, view.v, attn_scale, "array", m, .{ .ctx = null }, s));
        } else {
            try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q, view.k, view.v, attn_scale, "", none_mask, .{ .ctx = null }, s));
        }

        var attn_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_t);
        try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm_back, 4, s));
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        const flat_shape = [_]c_int{ 1, q_len_c, @intCast(cfg.num_attention_heads * cfg.head_dim) };
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, s));
        const o_out = try lw.o.apply(attn_flat, s);
        defer _ = mlx.mlx_array_free(o_out);

        var h_new = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&h_new, x, o_out, s));
        _ = mlx.mlx_array_free(x);
        x = h_new;

        const ff_normed = try rmsNormFn(x, lw.post_attn_norm, cfg.rms_norm_eps, s);
        defer _ = mlx.mlx_array_free(ff_normed);
        const gate = try lw.gate.apply(ff_normed, s);
        defer _ = mlx.mlx_array_free(gate);
        const up = try lw.up.apply(ff_normed, s);
        defer _ = mlx.mlx_array_free(up);
        const act = try swiglu(gate, up, s);
        defer _ = mlx.mlx_array_free(act);
        const down = try lw.down.apply(act, s);
        defer _ = mlx.mlx_array_free(down);

        var h_next = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&h_next, x, down, s));
        _ = mlx.mlx_array_free(x);
        x = h_next;
    }

    // Evict the block K/V — the context cache must be exactly as it was.
    try ctx.cache.truncate(ctx_len, s);

    const out = try rmsNormFn(x, model.final_norm, cfg.rms_norm_eps, s);
    _ = mlx.mlx_array_free(x);
    return out;
}

// ── Tests ──

const testing = std.testing;

const MUSE_ASSISTANT_CONFIG_JSON =
    \\{
    \\  "architectures": ["MuseGlimmerAssistantModel"],
    \\  "block_size": 16,
    \\  "head_dim": 128,
    \\  "hidden_act": "silu",
    \\  "hidden_size": 6656,
    \\  "intermediate_size": 19968,
    \\  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    \\  "mask_token_id": 201818,
    \\  "max_position_embeddings": 131072,
    \\  "model_type": "muse_glimmer_assistant",
    \\  "num_attention_heads": 32,
    \\  "num_hidden_layers": 5,
    \\  "num_key_value_heads": 8,
    \\  "rms_norm_eps": 1e-05,
    \\  "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    \\  "sliding_window": 2048,
    \\  "target_layer_ids": [1, 13, 25, 37, 49]
    \\}
;

// Gemma 4 assistant drafter shape (cross-attention drafter, NOT DFlash):
// no block_size / mask_token_id / target_layer_ids.
const GEMMA_ASSISTANT_CONFIG_JSON =
    \\{
    \\  "model_type": "gemma4_assistant",
    \\  "backbone_hidden_size": 2560,
    \\  "num_centroids": 512,
    \\  "centroid_intermediate_top_k": 16,
    \\  "use_ordered_embeddings": true,
    \\  "text_config": {
    \\    "hidden_size": 256,
    \\    "num_hidden_layers": 4,
    \\    "num_attention_heads": 4,
    \\    "head_dim": 256,
    \\    "intermediate_size": 1024,
    \\    "sliding_window": 512,
    \\    "vocab_size": 262144,
    \\    "rms_norm_eps": 1e-06,
    \\    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
    \\  }
    \\}
;

test "dflash: muse assistant config parses with the full DFlash contract" {
    const allocator = testing.allocator;
    var cfg = try parseConfigFromJson(allocator, MUSE_ASSISTANT_CONFIG_JSON);
    defer cfg.deinit(allocator);

    try testing.expectEqual(@as(u32, 16), cfg.block_size);
    try testing.expectEqual(@as(u32, 201818), cfg.mask_token_id);
    try testing.expectEqualSlices(u32, &[_]u32{ 1, 13, 25, 37, 49 }, cfg.target_layer_ids);
    try testing.expectEqual(@as(u32, 6656), cfg.hidden_size);
    try testing.expectEqual(@as(u32, 5), cfg.num_hidden_layers);
    try testing.expectEqual(@as(u32, 32), cfg.num_attention_heads);
    try testing.expectEqual(@as(u32, 8), cfg.num_key_value_heads);
    try testing.expectEqual(@as(u32, 128), cfg.head_dim);
    try testing.expectEqual(@as(u32, 19968), cfg.intermediate_size);
    try testing.expectEqual(@as(u32, 2048), cfg.sliding_window);
    try testing.expectApproxEqAbs(@as(f32, 1e-5), cfg.rms_norm_eps, 1e-9);
    try testing.expectApproxEqAbs(@as(f32, 500000.0), cfg.rope_theta, 1.0);
    try testing.expectEqual(@as(usize, 5), cfg.layer_types.len);
    for (cfg.layer_types) |lt| try testing.expectEqual(LayerType.sliding_attention, lt);
}

test "dflash: gemma assistant config is NOT detected as DFlash" {
    const allocator = testing.allocator;
    // Detection helper says no…
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, GEMMA_ASSISTANT_CONFIG_JSON, .{});
    defer parsed.deinit();
    try testing.expect(!isDflashConfigJson(parsed.value.object));
    // …and the parser rejects it by name.
    try testing.expectError(error.NotDflashConfig, parseConfigFromJson(allocator, GEMMA_ASSISTANT_CONFIG_JSON));
}

test "dflash: muse assistant config IS detected" {
    const allocator = testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, MUSE_ASSISTANT_CONFIG_JSON, .{});
    defer parsed.deinit();
    try testing.expect(isDflashConfigJson(parsed.value.object));
}

test "dflash: out-of-range target_layer_ids rejected by name" {
    // Muse trunk has 52 layers; id 49 is fine, id 52 is not.
    try validateTargetLayers(&[_]u32{ 1, 13, 25, 37, 49 }, 52);
    try testing.expectError(error.DflashTargetLayerOutOfRange, validateTargetLayers(&[_]u32{ 1, 52 }, 52));
    try testing.expectError(error.DflashTargetLayerOutOfRange, validateTargetLayers(&[_]u32{60}, 52));
}

test "dflash: block size resolves from config, clamped downward by explicit CLI" {
    // A machine WITH a wide verify lane keeps the checkpoint's own block.
    try testing.expectEqual(@as(u32, 16), resolveBlockSize(16, 4, false, true));
    // Explicit smaller CLI clamps down.
    try testing.expectEqual(@as(u32, 8), resolveBlockSize(16, 8, true, true));
    // Explicit LARGER CLI never raises past the config (training contract).
    try testing.expectEqual(@as(u32, 16), resolveBlockSize(16, 32, true, true));
    // Floor 2.
    try testing.expectEqual(@as(u32, 2), resolveBlockSize(16, 1, true, true));
}

test "dflash: an assistant merged into the checkpoint is found without a flag" {
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const model_dir = path_buf[0..root_len];

    // A model dir with nothing in it resolves to nothing.
    try testing.expectEqual(@as(?[]u8, null), resolveInDirDrafter(io, allocator, model_dir));

    // A `drafter/` subdir declaring the CONTRACT is the sidecar — the same
    // three fields the flag path probes, so detection cannot drift between
    // "pointed at" and "shipped with".
    try tmp.dir.createDirPath(io, IN_DIR_SUBDIR);
    var sub = try tmp.dir.openDir(io, IN_DIR_SUBDIR, .{});
    defer sub.close(io);
    {
        var f = try sub.createFile(io, "config.json", .{});
        defer f.close(io);
        var wbuf: [512]u8 = undefined;
        var w = f.writer(io, &wbuf);
        try w.interface.writeAll(
            \\{"model_type":"muse_glimmer_assistant","block_size":16,"mask_token_id":7,
            \\ "target_layer_ids":[1,3],"hidden_size":64,"num_hidden_layers":2,
            \\ "num_attention_heads":4,"head_dim":16,"intermediate_size":128}
        );
        try w.interface.flush();
    }
    const found = resolveInDirDrafter(io, allocator, model_dir) orelse
        return error.TestExpectedInDirDrafter;
    defer allocator.free(found);
    try testing.expect(std.mem.endsWith(u8, found, IN_DIR_SUBDIR));
    try testing.expect(probeIsDflash(io, allocator, found));

    // A relative or empty model dir is refused rather than joined blindly —
    // `openDirAbsolute` on a non-absolute path is ReleaseFast UB.
    try testing.expectEqual(@as(?[]u8, null), resolveInDirDrafter(io, allocator, ""));
    try testing.expectEqual(@as(?[]u8, null), resolveInDirDrafter(io, allocator, "relative/dir"));
}

test "dflash: no wide verify lane caps the block at the split-K width" {
    // Without an M 8..16 verify lane the trunk forward falls off a cliff at
    // width 8, so the block is capped even though the checkpoint asks for 16.
    try testing.expectEqual(NO_WIDE_LANE_BLOCK_CAP, resolveBlockSize(16, 4, false, false));
    // The cap is a CEILING, never a floor: an explicit smaller CLI still wins,
    // and a checkpoint whose own block is already narrow is left alone.
    try testing.expectEqual(@as(u32, 3), resolveBlockSize(16, 3, true, false));
    try testing.expectEqual(@as(u32, 4), resolveBlockSize(4, 8, true, false));
    // Explicit CLI is the escape hatch for a wider block on capped hardware:
    // it clamps against the CONFIG block, not the cap.
    try testing.expectEqual(@as(u32, 8), resolveBlockSize(16, 8, true, false));
}

// ── Hermetic fixtures: tiny llama trunk + tiny DFlash assistant ──
// pub: generate.zig's nextDflash equivalence test builds the same pair.

pub const TinyFix = struct {
    // Dims are small but affine-quantizable: every contraction dim is a
    // multiple of 32, so the load-time quantization and the draft-only head
    // are exercisable hermetically (MLX affine groups are 32/64/128).
    pub const HIDDEN: usize = 64;
    pub const VOCAB: usize = 128;
    pub const HEAD_DIM: usize = 16;
    pub const N_HEADS: usize = 4;
    pub const N_KV: usize = 2;
    pub const INTER: usize = 128;
    pub const TRUNK_LAYERS: usize = 4;
    pub const ASSISTANT_LAYERS: usize = 2;
    pub const N_TARGETS: usize = 2;
    pub const BLOCK: usize = 4;

    /// Deterministic small weight values; `seed` de-correlates tensors.
    pub fn val(i: usize, seed: usize) f32 {
        return @as(f32, @floatFromInt((i * 7 + seed * 13) % 23)) * 0.02 - 0.2;
    }

    pub fn bf16Arr(rows: usize, cols: usize, seed: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const total = rows * @max(cols, 1);
        const data = try testing.allocator.alloc(f32, total);
        defer testing.allocator.free(data);
        for (data, 0..) |*x, i| x.* = val(i, seed);
        const shape2 = [_]c_int{ @intCast(rows), @intCast(cols) };
        const shape1 = [_]c_int{@intCast(rows)};
        const f32_arr = if (cols > 0)
            mlx.mlx_array_new_data(data.ptr, &shape2, 2, .float32)
        else
            mlx.mlx_array_new_data(data.ptr, &shape1, 1, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var bf = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(bf);
        try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(bf));
        return bf;
    }

    /// Norm weights near 1.0 so norms stay neutral-ish.
    pub fn normArr(n: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const data = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(data);
        for (data, 0..) |*x, i| x.* = 1.0 + val(i, 5) * 0.1;
        const shape = [_]c_int{@intCast(n)};
        const f32_arr = mlx.mlx_array_new_data(data.ptr, &shape, 1, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var bf = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(bf);
        try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(bf));
        return bf;
    }

    pub fn put(map: mlx.mlx_map_string_to_array, key: []const u8, arr: mlx.mlx_array) !void {
        const key_z = try std.fmt.allocPrintSentinel(testing.allocator, "{s}", .{key}, 0);
        defer testing.allocator.free(key_z);
        _ = mlx.mlx_map_string_to_array_insert(map, key_z.ptr, arr);
    }

    pub fn putW(map: mlx.mlx_map_string_to_array, key: []const u8, rows: usize, cols: usize, seed: usize, s: mlx.mlx_stream) !void {
        const arr = try bf16Arr(rows, cols, seed, s);
        defer _ = mlx.mlx_array_free(arr);
        try put(map, key, arr);
    }

    pub fn putNorm(map: mlx.mlx_map_string_to_array, key: []const u8, n: usize, s: mlx.mlx_stream) !void {
        const arr = try normArr(n, s);
        defer _ = mlx.mlx_array_free(arr);
        try put(map, key, arr);
    }

    // Tiny llama trunk: vocab 128, hidden 64, 4 layers, 4 q / 2 kv heads,
    // hd 16, inter 128, tied embeddings.
    pub const TRUNK_CONFIG =
        \\{
        \\  "model_type": "llama",
        \\  "hidden_size": 64,
        \\  "intermediate_size": 128,
        \\  "num_hidden_layers": 4,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 16,
        \\  "vocab_size": 128,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_theta": 10000.0,
        \\  "tie_word_embeddings": true,
        \\  "max_position_embeddings": 2048,
        \\  "torch_dtype": "bfloat16"
        \\}
    ;

    pub fn writeTrunk(io: std.Io, dir: std.Io.Dir, dir_path: []const u8, s: mlx.mlx_stream) !void {
        try dir.writeFile(io, .{ .sub_path = "config.json", .data = TRUNK_CONFIG });
        const st_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model.safetensors", .{dir_path}, 0);
        defer testing.allocator.free(st_path);
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);

        const q_out = N_HEADS * HEAD_DIM;
        const kv_out = N_KV * HEAD_DIM;
        try putW(map, "model.embed_tokens.weight", VOCAB, HIDDEN, 1, s);
        try putNorm(map, "model.norm.weight", HIDDEN, s);
        var key_buf: [128]u8 = undefined;
        var li: usize = 0;
        while (li < TRUNK_LAYERS) : (li += 1) {
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.input_layernorm.weight", .{li}), HIDDEN, s);
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.post_attention_layernorm.weight", .{li}), HIDDEN, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.q_proj.weight", .{li}), q_out, HIDDEN, 10 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.k_proj.weight", .{li}), kv_out, HIDDEN, 20 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.v_proj.weight", .{li}), kv_out, HIDDEN, 30 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.self_attn.o_proj.weight", .{li}), HIDDEN, q_out, 40 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.gate_proj.weight", .{li}), INTER, HIDDEN, 50 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.up_proj.weight", .{li}), INTER, HIDDEN, 60 + li, s);
            try putW(map, try std.fmt.bufPrint(&key_buf, "model.layers.{d}.mlp.down_proj.weight", .{li}), HIDDEN, INTER, 70 + li, s);
        }
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    // Tiny assistant: hidden 64 (== trunk), 2 layers (sliding + full),
    // 4 q / 2 kv heads, hd 16, inter 128, window 8, block 4, mask id 127,
    // targets [0, 2].
    pub const ASSISTANT_CONFIG =
        \\{
        \\  "model_type": "tiny_assistant",
        \\  "block_size": 4,
        \\  "mask_token_id": 127,
        \\  "target_layer_ids": [0, 2],
        \\  "hidden_size": 64,
        \\  "intermediate_size": 128,
        \\  "num_hidden_layers": 2,
        \\  "num_attention_heads": 4,
        \\  "num_key_value_heads": 2,
        \\  "head_dim": 16,
        \\  "rms_norm_eps": 1e-5,
        \\  "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        \\  "sliding_window": 8,
        \\  "layer_types": ["sliding_attention", "full_attention"]
        \\}
    ;

    /// `quantize`: also emit `.scales`/`.biases` for every matmul weight, the
    /// shape a sidecar published pre-quantized would ship.
    pub fn writeAssistant(io: std.Io, dir: std.Io.Dir, dir_path: []const u8, s: mlx.mlx_stream) !void {
        return writeAssistantOpts(io, dir, dir_path, s, false);
    }

    pub fn writeAssistantOpts(io: std.Io, dir: std.Io.Dir, dir_path: []const u8, s: mlx.mlx_stream, quantize: bool) !void {
        try dir.writeFile(io, .{ .sub_path = "config.json", .data = ASSISTANT_CONFIG });
        const st_path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/model.safetensors", .{dir_path}, 0);
        defer testing.allocator.free(st_path);
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);

        const q_out = N_HEADS * HEAD_DIM;
        const kv_out = N_KV * HEAD_DIM;
        const putLin = struct {
            fn f(m: mlx.mlx_map_string_to_array, key: []const u8, rows: usize, cols: usize, seed: usize, st: mlx.mlx_stream, q: bool) !void {
                if (q) return putQuantW(m, key, rows, cols, seed, st);
                return putW(m, key, rows, cols, seed, st);
            }
        }.f;

        try putLin(map, "encoder.fc.weight", HIDDEN, N_TARGETS * HIDDEN, 100, s, quantize);
        try putNorm(map, "encoder.output_norm_enc.weight", HIDDEN, s);
        try putNorm(map, "norm.weight", HIDDEN, s);
        var key_buf: [128]u8 = undefined;
        var li: usize = 0;
        while (li < ASSISTANT_LAYERS) : (li += 1) {
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.input_layernorm.weight", .{li}), HIDDEN, s);
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.post_attention_layernorm.weight", .{li}), HIDDEN, s);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.q_proj.weight", .{li}), q_out, HIDDEN, 110 + li, s, quantize);
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.q_norm.weight", .{li}), HEAD_DIM, s);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.k_proj.weight", .{li}), kv_out, HIDDEN, 120 + li, s, quantize);
            try putNorm(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.k_norm.weight", .{li}), HEAD_DIM, s);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.v_proj.weight", .{li}), kv_out, HIDDEN, 130 + li, s, quantize);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.self_attn.o_proj.weight", .{li}), HIDDEN, q_out, 140 + li, s, quantize);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.gate_proj.weight", .{li}), INTER, HIDDEN, 150 + li, s, quantize);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.up_proj.weight", .{li}), INTER, HIDDEN, 160 + li, s, quantize);
            try putLin(map, try std.fmt.bufPrint(&key_buf, "layers.{d}.mlp.down_proj.weight", .{li}), HIDDEN, INTER, 170 + li, s, quantize);
        }
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    /// A `<prefix>.weight` written PACKED, with its `.scales`/`.biases`
    /// siblings — the pre-quantized-sidecar shape.
    pub fn putQuantW(map: mlx.mlx_map_string_to_array, key: []const u8, rows: usize, cols: usize, seed: usize, s: mlx.mlx_stream) !void {
        const dense = try bf16Arr(rows, cols, seed, s);
        defer _ = mlx.mlx_array_free(dense);
        var lin = try quantizeDense(dense, 8, @intCast(@min(cols, 64)), s);
        defer lin.deinit();
        try put(map, key, lin.w);
        const base = key[0 .. key.len - ".weight".len];
        var buf: [160]u8 = undefined;
        try put(map, try std.fmt.bufPrint(&buf, "{s}.scales", .{base}), lin.scales);
        try put(map, try std.fmt.bufPrint(&buf, "{s}.biases", .{base}), lin.biases);
    }

    pub fn readF32(arr: mlx.mlx_array, allocator: std.mem.Allocator, s: mlx.mlx_stream) ![]f32 {
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const sh = mlx.getShape(f);
        var total: c_int = 1;
        for (sh) |d| total *= d;
        const flat_shape = [_]c_int{total};
        try mlx.check(mlx.mlx_reshape(&flat, f, &flat_shape, 1, s));
        try mlx.check(mlx.mlx_array_eval(flat));
        const ptr = mlx.mlx_array_data_float32(flat) orelse return error.MlxArrayDataNull;
        const out = try allocator.alloc(f32, @intCast(total));
        @memcpy(out, ptr[0..@intCast(total)]);
        return out;
    }

    /// Synthetic trunk-capture array `[1, n, HIDDEN]`.
    pub fn capArr(n: usize, seed: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        const data = try testing.allocator.alloc(f32, n * HIDDEN);
        defer testing.allocator.free(data);
        for (data, 0..) |*x, i| x.* = val(i, seed);
        const shape = [_]c_int{ 1, @intCast(n), HIDDEN };
        const f32_arr = mlx.mlx_array_new_data(data.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(f32_arr);
        var bf = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(bf);
        try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(bf));
        return bf;
    }
};

test "dflash: loadDflash keeps a dense bf16 assistant dense + pre-transposed when quantization is off" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeAssistant(io, tmp_dir.dir, dir_path, s);

    var m = try loadDflashQuant(io, allocator, s, dir_path, 0);
    defer m.deinit();

    try testing.expectEqual(@as(u32, 4), m.config.block_size);
    try testing.expectEqual(TinyFix.ASSISTANT_LAYERS, m.layers.len);
    // fc pre-transposed: checkpoint [H, NT*H] → [n_targets*hidden, hidden].
    try testing.expect(!m.fc.isQuantized());
    const fc_shape = mlx.getShape(m.fc.w);
    try testing.expectEqual(@as(c_int, TinyFix.N_TARGETS * TinyFix.HIDDEN), fc_shape[0]);
    try testing.expectEqual(@as(c_int, TinyFix.HIDDEN), fc_shape[1]);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(m.fc.w));
    // k pre-transposed: [kv_out, H] → [H, kv_out].
    const k_shape = mlx.getShape(m.layers[0].k.w);
    try testing.expectEqual(@as(c_int, TinyFix.HIDDEN), k_shape[0]);
    try testing.expectEqual(@as(c_int, TinyFix.N_KV * TinyFix.HEAD_DIM), k_shape[1]);
    try testing.expectEqual(LayerType.sliding_attention, m.layers[0].layer_type);
    try testing.expectEqual(LayerType.full_attention, m.layers[1].layer_type);
}

test "dflash: buildBlockMask sliding/full arms and window edges" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // Full layer: never a mask.
    try testing.expectEqual(@as(?mlx.mlx_array, null), try buildBlockMask(.full_attention, 0, 100, 100, 4, 8, s));
    // Sliding, everything inside the window: no mask needed.
    try testing.expectEqual(@as(?mlx.mlx_array, null), try buildBlockMask(.sliding_attention, 0, 4, 4, 4, 8, s));

    // Sliding, ctx_len 10, anchor 10, block 4, window 8: rows are queries at
    // abs 10..13, cols are ctx abs 0..9 then block abs 10..13.
    const mask = (try buildBlockMask(.sliding_attention, 0, 10, 10, 4, 8, s)).?;
    defer _ = mlx.mlx_array_free(mask);
    const msh = mlx.getShape(mask);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 1, 1, 4, 14 }, msh);
    const vals = try TinyFix.readF32(mask, allocator, s);
    defer allocator.free(vals);
    const at = struct {
        fn f(v: []const f32, row: usize, col: usize) f32 {
            return v[row * 14 + col];
        }
    }.f;
    // q=10: k=0,1,2 masked (dist ≥ 8), k=3.. allowed.
    try testing.expect(std.math.isNegativeInf(at(vals, 0, 0)));
    try testing.expect(std.math.isNegativeInf(at(vals, 0, 2)));
    try testing.expectEqual(@as(f32, 0.0), at(vals, 0, 3));
    // q=13: k=5 masked (dist 8), k=6 allowed (dist 7).
    try testing.expect(std.math.isNegativeInf(at(vals, 3, 5)));
    try testing.expectEqual(@as(f32, 0.0), at(vals, 3, 6));
    // Block ↔ block always visible (|q-k| ≤ 3 < 8).
    try testing.expectEqual(@as(f32, 0.0), at(vals, 0, 13));
    try testing.expectEqual(@as(f32, 0.0), at(vals, 3, 10));
}

test "dflash: appendContext grows the cache; forwardBlock evicts its block K/V" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeAssistant(io, tmp_dir.dir, dir_path, s);
    var m = try loadDflash(io, allocator, s, dir_path);
    defer m.deinit();

    var ctx = try DflashCtx.init(allocator, &m, 0);
    defer ctx.deinit();

    // Append 6 context tokens (2 capture streams, one per target layer id).
    {
        const c0 = try TinyFix.capArr(6, 200, s);
        defer _ = mlx.mlx_array_free(c0);
        const c1 = try TinyFix.capArr(6, 201, s);
        defer _ = mlx.mlx_array_free(c1);
        try appendContext(&m, &ctx, &[_]mlx.mlx_array{ c0, c1 }, 0);
    }
    try testing.expectEqual(@as(usize, 6), ctx.cache.step);
    try testing.expectEqual(@as(usize, 6), ctx.absLen());
    // Context K/V shape per layer: [1, N_KV, 6, HEAD_DIM].
    const ksh = mlx.getShape(ctx.cache.entries[0].key_view);
    try testing.expectEqual(@as(c_int, TinyFix.N_KV), ksh[1]);
    try testing.expectEqual(@as(c_int, TinyFix.HEAD_DIM), ksh[3]);

    // Block forward at anchor 6 → [1, BLOCK, HIDDEN]; cache unchanged after.
    const noise = try TinyFix.capArr(4, 300, s);
    defer _ = mlx.mlx_array_free(noise);
    const hidden = try forwardBlock(&m, &ctx, noise, 6);
    defer _ = mlx.mlx_array_free(hidden);
    try mlx.check(mlx.mlx_array_eval(hidden));
    const hsh = mlx.getShape(hidden);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 1, TinyFix.BLOCK, TinyFix.HIDDEN }, hsh);
    try testing.expectEqual(@as(usize, 6), ctx.cache.step); // block evicted

    // Continue appending (next round's accepted tokens).
    {
        const c0 = try TinyFix.capArr(3, 210, s);
        defer _ = mlx.mlx_array_free(c0);
        const c1 = try TinyFix.capArr(3, 211, s);
        defer _ = mlx.mlx_array_free(c1);
        try appendContext(&m, &ctx, &[_]mlx.mlx_array{ c0, c1 }, 6);
    }
    try testing.expectEqual(@as(usize, 9), ctx.cache.step);
}

/// Run the full context-append + block forward on `m` and return the block
/// hidden as f32 — the comparison surface for weight-precision arms.
fn tinyBlockHidden(m: *DflashModel, allocator: std.mem.Allocator, s: mlx.mlx_stream) ![]f32 {
    var ctx = try DflashCtx.init(allocator, m, 0);
    defer ctx.deinit();
    const c0 = try TinyFix.capArr(10, 600, s);
    defer _ = mlx.mlx_array_free(c0);
    const c1 = try TinyFix.capArr(10, 601, s);
    defer _ = mlx.mlx_array_free(c1);
    try appendContext(m, &ctx, &[_]mlx.mlx_array{ c0, c1 }, 0);
    const noise = try TinyFix.capArr(TinyFix.BLOCK, 700, s);
    defer _ = mlx.mlx_array_free(noise);
    const hidden = try forwardBlock(m, &ctx, noise, 10);
    defer _ = mlx.mlx_array_free(hidden);
    return TinyFix.readF32(hidden, allocator, s);
}

test "dflash: load-time quantization packs every matmul weight and tracks the dense forward" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeAssistant(io, tmp_dir.dir, dir_path, s);

    var dense = try loadDflashQuant(io, allocator, s, dir_path, 0);
    defer dense.deinit();
    var quant = try loadDflashQuant(io, allocator, s, dir_path, 8);
    defer quant.deinit();

    // Every contracted weight is packed at the requested width, and the
    // packing is the CHECKPOINT's [out, in] layout (qmm transpose=true) —
    // a pre-transposed packed weight would contract the wrong axis.
    try testing.expect(quant.fc.isQuantized());
    try testing.expectEqual(@as(u32, 8), quant.fc.bits);
    try testing.expectEqual(QUANT_GROUP, quant.fc.group_size);
    try testing.expectEqual(@as(c_int, TinyFix.HIDDEN), mlx.getShape(quant.fc.w)[0]);
    for (quant.layers) |*lw| {
        for ([_]*const DflashLinear{ &lw.q, &lw.k, &lw.v, &lw.o, &lw.gate, &lw.up, &lw.down }) |lin| {
            try testing.expect(lin.isQuantized());
            try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(lin.w));
        }
    }

    const a = try tinyBlockHidden(&dense, allocator, s);
    defer allocator.free(a);
    const b = try tinyBlockHidden(&quant, allocator, s);
    defer allocator.free(b);
    const p = try paritySlices(a, b);
    try testing.expect(p.cos > 0.99);
    try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
}

test "dflash: a sidecar shipping packed weights loads at its own declared width" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;

    var tmp_dense = std.testing.tmpDir(.{});
    defer tmp_dense.cleanup();
    var dense_buf: [512]u8 = undefined;
    const dense_path = dense_buf[0..try tmp_dense.dir.realPath(io, &dense_buf)];
    try TinyFix.writeAssistant(io, tmp_dense.dir, dense_path, s);

    var tmp_packed = std.testing.tmpDir(.{});
    defer tmp_packed.cleanup();
    var packed_buf: [512]u8 = undefined;
    const packed_path = packed_buf[0..try tmp_packed.dir.realPath(io, &packed_buf)];
    try TinyFix.writeAssistantOpts(io, tmp_packed.dir, packed_path, s, true);

    var dense = try loadDflashQuant(io, allocator, s, dense_path, 0);
    defer dense.deinit();
    // The request for dense bf16 does NOT unpack a packed checkpoint — the
    // shipped weights are served as they are, at the width the geometry says.
    var shipped = try loadDflashQuant(io, allocator, s, packed_path, 0);
    defer shipped.deinit();

    try testing.expect(shipped.fc.isQuantized());
    try testing.expectEqual(@as(u32, 8), shipped.fc.bits);
    try testing.expectEqual(@as(u32, 64), shipped.layers[0].down.group_size);

    const a = try tinyBlockHidden(&dense, allocator, s);
    defer allocator.free(a);
    const b = try tinyBlockHidden(&shipped, allocator, s);
    defer allocator.free(b);
    const p = try paritySlices(a, b);
    try testing.expect(p.cos > 0.99);
    try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
}

test "dflash: the draft-only lm_head shrinks the draft read and leaves verify alone" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;

    var tmp_trunk = std.testing.tmpDir(.{});
    defer tmp_trunk.cleanup();
    var trunk_buf: [512]u8 = undefined;
    const trunk_path = trunk_buf[0..try tmp_trunk.dir.realPath(io, &trunk_buf)];
    try TinyFix.writeTrunk(io, tmp_trunk.dir, trunk_path, s);

    var tmp_asst = std.testing.tmpDir(.{});
    defer tmp_asst.cleanup();
    var asst_buf: [512]u8 = undefined;
    const asst_path = asst_buf[0..try tmp_asst.dir.realPath(io, &asst_buf)];
    try TinyFix.writeAssistant(io, tmp_asst.dir, asst_path, s);

    var config = try model_mod.parseConfig(io, allocator, trunk_path);
    var weights = try model_mod.loadWeights(io, allocator, trunk_path);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();

    const bits: u32 = 3;
    var m = try loadDflashQuant(io, allocator, s, asst_path, 0);
    defer m.deinit();
    try m.bindWithDraftBits(&xfm, bits);

    try testing.expect(m.draft_head != null);
    try testing.expectEqual(bits, m.draft_head_bits);
    try testing.expectEqual(QUANT_GROUP, m.draft_head_group);
    // Packed [vocab, hidden*bits/32] — the head's own rows, not a transpose.
    const dh_shape = mlx.getShape(m.draft_head.?.w);
    try testing.expectEqual(@as(c_int, TinyFix.VOCAB), dh_shape[0]);
    try testing.expectEqual(@as(c_int, (TinyFix.HIDDEN * bits) / 32), dh_shape[1]);

    // The draft head is a DRAFT surface: same shape as the trunk projection,
    // correlated with it, and the trunk head itself is untouched (verify
    // reads it through the ordinary forward).
    const x = try TinyFix.capArr(TinyFix.BLOCK, 800, s);
    defer _ = mlx.mlx_array_free(x);
    const via_draft = try m.draftLogits(&xfm, x);
    defer _ = mlx.mlx_array_free(via_draft);
    const via_trunk = try xfm.lmHeadForDraft(x);
    defer _ = mlx.mlx_array_free(via_trunk);
    try testing.expectEqualSlices(c_int, mlx.getShape(via_trunk), mlx.getShape(via_draft));
    const a = try TinyFix.readF32(via_draft, allocator, s);
    defer allocator.free(a);
    const b = try TinyFix.readF32(via_trunk, allocator, s);
    defer allocator.free(b);
    const p = try paritySlices(a, b);
    try testing.expect(p.cos > 0.9);

    // A width that would not shrink the read never builds one (a bf16 head
    // costs 16 bits/weight — an 8-bit re-encode saves, a 16-bit one does not).
    try m.bindWithDraftBits(&xfm, 16);
    try testing.expect(m.draft_head == null);
    try m.bindWithDraftBits(&xfm, 8);
    try testing.expect(m.draft_head != null);
    try m.bindWithDraftBits(&xfm, 0);
    try testing.expect(m.draft_head == null);
}

test "dflash: quantGroupFor picks the widest divisor, declines what affine cannot pack" {
    try testing.expectEqual(@as(?u32, 64), quantGroupFor(6656));
    try testing.expectEqual(@as(?u32, 64), quantGroupFor(19968));
    try testing.expectEqual(@as(?u32, 64), quantGroupFor(33280));
    try testing.expectEqual(@as(?u32, 32), quantGroupFor(96));
    try testing.expectEqual(@as(?u32, null), quantGroupFor(48));
    try testing.expectEqual(@as(?u32, null), quantGroupFor(0));
}

test "dflash: trunk capture_layers = hidden_states[i+1], consistent across chunked prefill" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeTrunk(io, tmp_dir.dir, dir_path, s);

    var config = try model_mod.parseConfig(io, allocator, dir_path);
    var weights = try model_mod.loadWeights(io, allocator, dir_path);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();

    const prompt = [_]i32{ 3, 7, 1, 12, 30, 5, 9, 22 };

    // target_layer_ids semantics use TRUNK indices; here just pick 1 and 3.
    const cap_ids = [_]u32{ 1, 3 };

    // ── Full forward over all 8 tokens, capturing layers + final hidden ──
    var full_out = [_]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (&full_out) |*a| {
        _ = mlx.mlx_array_free(a.*);
    };
    var all_hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all_hidden);
    {
        var cl = transformer_mod.CaptureLayers{ .ids = &cap_ids, .out = &full_out };
        var ctx = xfm.defaultCtx();
        ctx.capture_layers = &cl;
        ctx.capture_hidden_all = &all_hidden;
        const shape = [_]c_int{ 1, 8 };
        const input = mlx.mlx_array_new_data(&prompt, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);
        const logits = try xfm.forwardWith(&ctx, input);
        _ = mlx.mlx_array_free(logits);
    }

    // hidden_states[i+1] pin: final-norm(capture at LAST layer) must equal
    // the post-final-norm capture_hidden_all — the capture sits exactly one
    // final_norm before it.
    {
        var manual = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(manual);
        try mlx.check(mlx.mlx_fast_rms_norm(&manual, full_out[1], xfm.final_norm, xfm.config.rms_norm_eps, s));
        const a = try TinyFix.readF32(manual, allocator, s);
        defer allocator.free(a);
        const b = try TinyFix.readF32(all_hidden, allocator, s);
        defer allocator.free(b);
        for (a, b) |x, y| try testing.expect(@abs(x - y) < 1e-5);
    }

    // ── Chunked: [0..5) then [5..8) — captures must concatenate to the full ──
    try xfm.resetCache();
    var chunk_vals = std.ArrayList(f32).empty;
    defer chunk_vals.deinit(allocator);
    var chunk_vals2 = std.ArrayList(f32).empty;
    defer chunk_vals2.deinit(allocator);
    const splits = [_][2]usize{ .{ 0, 5 }, .{ 5, 8 } };
    for (splits) |sp| {
        var out = [_]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (&out) |*a| {
            _ = mlx.mlx_array_free(a.*);
        };
        var cl = transformer_mod.CaptureLayers{ .ids = &cap_ids, .out = &out };
        var ctx = xfm.defaultCtx();
        ctx.capture_layers = &cl;
        const shape = [_]c_int{ 1, @intCast(sp[1] - sp[0]) };
        const input = mlx.mlx_array_new_data(@ptrCast(&prompt[sp[0]]), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);
        const logits = try xfm.forwardWith(&ctx, input);
        _ = mlx.mlx_array_free(logits);
        const v0 = try TinyFix.readF32(out[0], allocator, s);
        defer allocator.free(v0);
        try chunk_vals.appendSlice(allocator, v0);
        const v1 = try TinyFix.readF32(out[1], allocator, s);
        defer allocator.free(v1);
        try chunk_vals2.appendSlice(allocator, v1);
    }
    const full0 = try TinyFix.readF32(full_out[0], allocator, s);
    defer allocator.free(full0);
    const full1 = try TinyFix.readF32(full_out[1], allocator, s);
    defer allocator.free(full1);
    try testing.expectEqual(full0.len, chunk_vals.items.len);
    // bf16 + different GEMM widths: tolerance, not bit equality.
    for (full0, chunk_vals.items) |x, y| try testing.expect(@abs(x - y) < 0.05);
    for (full1, chunk_vals2.items) |x, y| try testing.expect(@abs(x - y) < 0.05);
}

test "dflash: trunk rawEmbedding is the bare table row; embedding layers norm on top" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeTrunk(io, tmp_dir.dir, dir_path, s);

    var config = try model_mod.parseConfig(io, allocator, dir_path);
    var weights = try model_mod.loadWeights(io, allocator, dir_path);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();

    const ids = [_]i32{ 4, 31 };
    const shape = [_]c_int{ 1, 2 };
    const input = mlx.mlx_array_new_data(&ids, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(input);

    const raw = try xfm.rawEmbedding(input);
    defer _ = mlx.mlx_array_free(raw);
    const raw_vals = try TinyFix.readF32(raw, allocator, s);
    defer allocator.free(raw_vals);
    // Row 4 of the table was written as val(4*16 + c, seed 1).
    for (0..TinyFix.HIDDEN) |c| {
        try testing.expect(@abs(raw_vals[c] - TinyFix.val(4 * TinyFix.HIDDEN + c, 1)) < 0.01);
    }

    // Simulate a normed-embeddings trunk (the muse shape): embedding() must
    // now be rms_norm(raw) while rawEmbedding stays the bare row — this is
    // the DFlash noise-embed contract ("embed without the norm").
    var ones = mlx.mlx_array_new();
    const one_scalar = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one_scalar);
    const ones_shape = [_]c_int{TinyFix.HIDDEN};
    try mlx.check(mlx.mlx_full(&ones, &ones_shape, 1, one_scalar, .bfloat16, s));
    xfm.config.normed_embeddings = true;
    std.debug.assert(xfm.ones_hidden == null);
    xfm.ones_hidden = ones; // freed by xfm.deinit
    const normed = try xfm.rawEmbedding(input); // still raw
    defer _ = mlx.mlx_array_free(normed);
    const still_raw = try TinyFix.readF32(normed, allocator, s);
    defer allocator.free(still_raw);
    for (raw_vals, still_raw) |x, y| try testing.expectEqual(x, y);
}

test "dflash: sliding window hides out-of-window context from the block, in-window reaches it" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    try TinyFix.writeAssistant(io, tmp_dir.dir, dir_path, s);
    var m = try loadDflash(io, allocator, s, dir_path);
    defer m.deinit();
    // Both layers sliding for this test — layer 1's full attention would
    // legitimately see the far row and mask nothing.
    m.layers[1].layer_type = .sliding_attention;

    // ctx_len 12, window 8, block 4 at anchor 12: queries at 12..15.
    // ctx row 0 (abs 0) is ≥ 8 away from every query → invisible.
    // ctx row 11 (abs 11) is ≤ 4 away → visible to all queries.
    const Run = struct {
        fn f(mm: *DflashModel, alloc: std.mem.Allocator, st: mlx.mlx_stream, perturb_row: ?usize) ![]f32 {
            var ctx = try DflashCtx.init(alloc, mm, 0);
            defer ctx.deinit();
            const c0 = try TinyFix.capArr(12, 400, st);
            defer _ = mlx.mlx_array_free(c0);
            var c1 = try TinyFix.capArr(12, 401, st);
            defer _ = mlx.mlx_array_free(c1);
            if (perturb_row) |row| {
                // Overwrite one context row with a big value.
                const big: [TinyFix.HIDDEN]f32 = @splat(100.0);
                const bshape = [_]c_int{ 1, 1, TinyFix.HIDDEN };
                const big_arr = mlx.mlx_array_new_data(&big, &bshape, 3, .float32);
                defer _ = mlx.mlx_array_free(big_arr);
                var big_bf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(big_bf);
                try mlx.check(mlx.mlx_astype(&big_bf, big_arr, .bfloat16, st));
                const start = [_]c_int{ 0, @intCast(row), 0 };
                const stop = [_]c_int{ 1, @as(c_int, @intCast(row)) + 1, TinyFix.HIDDEN };
                const strides = [_]c_int{ 1, 1, 1 };
                var updated = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice_update(&updated, c1, big_bf, &start, 3, &stop, 3, &strides, 3, st));
                _ = mlx.mlx_array_free(c1);
                c1 = updated;
            }
            try appendContext(mm, &ctx, &[_]mlx.mlx_array{ c0, c1 }, 0);
            const noise = try TinyFix.capArr(4, 500, st);
            defer _ = mlx.mlx_array_free(noise);
            const hidden = try forwardBlock(mm, &ctx, noise, 12);
            defer _ = mlx.mlx_array_free(hidden);
            return TinyFix.readF32(hidden, alloc, st);
        }
    }.f;

    const base = try Run(&m, allocator, s, null);
    defer allocator.free(base);
    const far = try Run(&m, allocator, s, 0); // out-of-window row
    defer allocator.free(far);
    const near = try Run(&m, allocator, s, 11); // in-window row
    defer allocator.free(near);

    var far_diff: f32 = 0;
    var near_diff: f32 = 0;
    for (base, far) |b, f| far_diff = @max(far_diff, @abs(b - f));
    for (base, near) |b, n| near_diff = @max(near_diff, @abs(b - n));
    // The masked-out row must not leak into the block at all…
    try testing.expect(far_diff == 0.0);
    // …while an in-window row must move it.
    try testing.expect(near_diff > 0.001);
}

// ── Env-gated real-checkpoint parity vs the executable reference ──
// Fixtures: tests/dump_dflash_fixtures.py (transformers 5.15 CPU fp32).

fn fixtureF32Slice(allocator: std.mem.Allocator, v: std.json.Value) ![]f32 {
    const arr = v.array;
    const out = try allocator.alloc(f32, arr.items.len);
    for (arr.items, 0..) |elem, i| {
        out[i] = switch (elem) {
            .float => |f| @floatCast(f),
            .integer => |n| @floatFromInt(n),
            else => return error.BadFixture,
        };
    }
    return out;
}

/// [1, n, H] bf16 mlx array from a flat f32 slice.
fn fixtureArr(vals: []const f32, n: usize, h: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = [_]c_int{ 1, @intCast(n), @intCast(h) };
    const f32_arr = mlx.mlx_array_new_data(vals.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, s));
    return bf;
}

/// Split a flat [1, n, NT*H] context stream into NT per-target [1, n, H]
/// capture arrays (feature-dim chunks, target order — the encoder's concat
/// reassembles them).
fn fixtureCaptures(allocator: std.mem.Allocator, flat: []const f32, n: usize, nt: usize, h: usize, s: mlx.mlx_stream) ![]mlx.mlx_array {
    const caps = try allocator.alloc(mlx.mlx_array, nt);
    const buf = try allocator.alloc(f32, n * h);
    defer allocator.free(buf);
    for (0..nt) |t| {
        for (0..n) |row| {
            @memcpy(buf[row * h .. (row + 1) * h], flat[row * nt * h + t * h .. row * nt * h + (t + 1) * h]);
        }
        caps[t] = try fixtureArr(buf, n, h, s);
    }
    return caps;
}

const ParityStats = struct { cos: f64, rms_ratio: f64 };

/// Cosine AND rms ratio — a cosine alone cannot see a scale error.
fn paritySlices(got: []const f32, want: []const f32) !ParityStats {
    try testing.expectEqual(want.len, got.len);
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (got, want) |a, b| {
        dot += @as(f64, a) * @as(f64, b);
        na += @as(f64, a) * @as(f64, a);
        nb += @as(f64, b) * @as(f64, b);
    }
    // Finiteness before the diff — an all-NaN stream scores cos 0/0.
    try testing.expect(std.math.isFinite(dot) and na > 0 and nb > 0);
    return .{ .cos = dot / (@sqrt(na) * @sqrt(nb)), .rms_ratio = @sqrt(na / nb) };
}

fn parityVs(got: mlx.mlx_array, want: []const f32, allocator: std.mem.Allocator, s: mlx.mlx_stream) !ParityStats {
    const g = try TinyFix.readF32(got, allocator, s);
    defer allocator.free(g);
    return paritySlices(g, want);
}

test "dflash fixture parity: encoder + two draft rounds vs transformers reference" {
    const fixtures_path = std.c.getenv("DFLASH_FIXTURES") orelse return;
    const assistant_dir = std.c.getenv("DFLASH_ASSISTANT_DIR") orelse return;
    if (mlx.noGpuBackend()) return;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    const file = try std.Io.Dir.openFileAbsolute(io, std.mem.span(fixtures_path), .{});
    var rb: [65536]u8 = undefined;
    var rs = file.reader(io, &rb);
    const content = try rs.interface.allocRemaining(allocator, .limited(1 << 30));
    file.close(io);
    defer allocator.free(content);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, content, .{});
    defer parsed.deinit();
    const root = parsed.value.object;

    const h: usize = @intCast(root.get("hidden_size").?.integer);
    const nt: usize = @intCast(root.get("n_targets").?.integer);
    const bs: usize = @intCast(root.get("block_size").?.integer);
    const n_ctx: usize = @intCast(root.get("n_ctx").?.integer);
    const n_delta: usize = @intCast(root.get("n_delta").?.integer);

    const ctx_stream = try fixtureF32Slice(allocator, root.get("ctx_stream").?);
    defer allocator.free(ctx_stream);
    const ctx_delta = try fixtureF32Slice(allocator, root.get("ctx_delta").?);
    defer allocator.free(ctx_delta);
    const noise1 = try fixtureF32Slice(allocator, root.get("noise1").?);
    defer allocator.free(noise1);
    const noise2 = try fixtureF32Slice(allocator, root.get("noise2").?);
    defer allocator.free(noise2);
    const encoder_out = try fixtureF32Slice(allocator, root.get("encoder_out").?);
    defer allocator.free(encoder_out);
    const round1_hidden = try fixtureF32Slice(allocator, root.get("round1_hidden").?);
    defer allocator.free(round1_hidden);
    const round2_hidden = try fixtureF32Slice(allocator, root.get("round2_hidden").?);
    defer allocator.free(round2_hidden);

    var m = try loadDflash(io, allocator, s, std.mem.span(assistant_dir));
    defer m.deinit();
    try testing.expectEqual(bs, m.config.block_size);
    try testing.expectEqual(nt, m.config.target_layer_ids.len);

    const caps1 = try fixtureCaptures(allocator, ctx_stream, n_ctx, nt, h, s);
    defer {
        for (caps1) |a| _ = mlx.mlx_array_free(a);
        allocator.free(caps1);
    }

    // Encoder projection parity (bf16 engine vs fp32 oracle — cos AND rms,
    // the concat-stream rule: a cosine alone cannot see a scale error).
    {
        const enc = try encodeContext(&m, caps1);
        defer _ = mlx.mlx_array_free(enc);
        const p = try parityVs(enc, encoder_out, allocator, s);
        std.debug.print("[dflash-fixture] encoder cos={d:.6} rms_ratio={d:.4}\n", .{ p.cos, p.rms_ratio });
        try testing.expect(p.cos > 0.99);
        try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
    }

    // Round 1: context [0, n_ctx), block anchored at n_ctx.
    var dctx = try DflashCtx.init(allocator, &m, 0);
    defer dctx.deinit();
    try appendContext(&m, &dctx, caps1, 0);
    {
        const noise = try fixtureArr(noise1, bs, h, s);
        defer _ = mlx.mlx_array_free(noise);
        const hidden = try forwardBlock(&m, &dctx, noise, n_ctx);
        defer _ = mlx.mlx_array_free(hidden);
        const p = try parityVs(hidden, round1_hidden, allocator, s);
        std.debug.print("[dflash-fixture] round1 cos={d:.6} rms_ratio={d:.4}\n", .{ p.cos, p.rms_ratio });
        try testing.expect(p.cos > 0.99);
        try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
    }

    // Round 2: n_delta more context, new anchor — pins absolute-position
    // RoPE and the block-K/V eviction across rounds.
    const caps2 = try fixtureCaptures(allocator, ctx_delta, n_delta, nt, h, s);
    defer {
        for (caps2) |a| _ = mlx.mlx_array_free(a);
        allocator.free(caps2);
    }
    try appendContext(&m, &dctx, caps2, n_ctx);
    {
        const noise = try fixtureArr(noise2, bs, h, s);
        defer _ = mlx.mlx_array_free(noise);
        const hidden = try forwardBlock(&m, &dctx, noise, n_ctx + n_delta);
        defer _ = mlx.mlx_array_free(hidden);
        const p = try parityVs(hidden, round2_hidden, allocator, s);
        std.debug.print("[dflash-fixture] round2 cos={d:.6} rms_ratio={d:.4}\n", .{ p.cos, p.rms_ratio });
        try testing.expect(p.cos > 0.99);
        try testing.expect(@abs(p.rms_ratio - 1.0) < 0.05);
    }
}

test "dflash: every server-side drafter-loaded gate also consults lm.dflash (per-site wiring class)" {
    // The live 2026-08-10 miss: the request PARSE said drafter=enabled, but
    // four per-surface `use_drafter = ... lm.drafter != null ...` re-derivations
    // silently dropped the dflash handle and every request decoded serial.
    // A drafter-loaded conjunct written per-site is a list of ONE — pin that
    // any non-comment line reading `lm.drafter != null` (or the entry./
    // scheduler. spellings) also reads the dflash sibling.
    const src = @embedFile("server.zig");
    var it = std.mem.splitScalar(u8, src, '\n');
    var lineno: usize = 0;
    var checked: usize = 0;
    while (it.next()) |raw| {
        lineno += 1;
        const trimmed = std.mem.trimStart(u8, raw, " ");
        if (std.mem.startsWith(u8, trimmed, "//")) continue;
        const has_drafter_gate = std.mem.indexOf(u8, raw, "lm.drafter != null") != null or
            std.mem.indexOf(u8, raw, "entry.drafter != null") != null or
            std.mem.indexOf(u8, raw, "scheduler.drafter != null") != null;
        if (!has_drafter_gate) continue;
        checked += 1;
        if (std.mem.indexOf(u8, raw, "dflash") == null) {
            std.debug.print("server.zig:{d}: drafter-loaded gate without a dflash sibling: {s}\n", .{ lineno, std.mem.trim(u8, raw, " ") });
            return error.DrafterGateMissesDflash;
        }
    }
    // Zero means the gates were renamed and this guard went vacuous.
    try testing.expect(checked >= 10);
}
