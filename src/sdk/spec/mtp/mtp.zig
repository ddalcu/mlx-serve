//! Qwen 3.5/3.6 native MTP (multi-token prediction) head.
//!
//! Some Qwen 3.6 checkpoints ship a trained one-layer "MTP" sidecar
//! (`mtp/weights.safetensors`, ~15 tensors) that predicts the token AFTER the
//! next one from `(trunk_hidden, next_token)`. Chaining it K times drafts K
//! tokens which the trunk verifies in one batched forward — same
//! draft/verify contract as the Gemma 4 assistant drafter, but the drafter
//! is the model's own head, so acceptance stays high even on novel content.
//!
//! Architecture (matches mlx-lm `qwen3_5` MTP contract):
//!   x      = fc(concat([rmsnorm_e(embed(token)), rmsnorm_h(hidden)]))   [bf16 fc]
//!   x      = full-attention decoder layer(x)    — own 1-layer KV cache,
//!            q/gate split + sigmoid output gate, q/k per-head RMS norm,
//!            partial RoPE (rotary_factor * head_dim dims) at explicit offset
//!   post   = rmsnorm(x, mtp.norm)
//!   logits = trunk lm_head(post);  next-depth hidden = post
//!
//! The MTP layer keeps a COMMITTED-HISTORY KV cache: entry j pairs the trunk
//! hidden at position p_j with the token at p_j+1, built over the prompt at
//! prefill and maintained over committed tokens each decode round (drafts
//! append temporary entries; the round's commit restores the snapshot and
//! re-appends from true verify hiddens). Text-only RoPE offsets are
//! cache-relative ("cache" position mode). Multimodal requests additionally
//! map those cache positions to the trunk's absolute M-RoPE positions so the
//! sidecar and trunk agree on image-token geometry.
//!
//! Everything MTP-specific lives in this file plus `Generator.nextMtp`
//! (src/engines/mlx/generate.zig); deleting the feature is removing those two.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../vision/mrope.zig");
const weights_load = @import("../../weights/load.zig");
const axisView = @import("../../../mlx/array.zig").axisView;
const kv_quant = @import("../../blocks/attention/kv_quant.zig");
const cache_mod = @import("../../kv/cache.zig");
const attn256 = @import("../../blocks/attention/attn256.zig");
const sdpa_mod = @import("../../blocks/attention/sdpa.zig");
const verify_projection = @import("../../blocks/linear/verify_projection.zig");
const verify_qmm = @import("../../blocks/linear/verify_qmm.zig");
const verify_qmm_nax = @import("../../blocks/linear/verify_qmm_nax.zig");
const weights_mod = @import("../../blocks/moe/weights.zig");
const params_mod = @import("../../quant/params.zig");
const log = @import("../../util/log.zig");
const io_util_mod = @import("../../util/io_util.zig");
const ane_mod = @import("../../ane/offload.zig");
const sushi_pack = @import("../../quant/exl3/sushi_pack.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;

const KVCache = cache_mod.KVCache;
const Weights = @import("../../weights/resident.zig").Weights;
const mtp_source = @import("source.zig");

/// Default draft depth (tokens drafted per round). Flipped 1 -> 3 after the
/// round-v2 rebuild made rejected drafts ~free (scalar-anchor rollback + the
/// 3-bit draft-only lm_head replaced the old full trunk re-forward): the old
/// cost model was what made depth 1 optimal. validation matrix on
/// Qwen3.6-27B 4-bit (M4 Max, adaptive controller active, decode tok/s
/// depth-3 vs depth-1): code 54.3 vs 43.3 (+25%), coding-agent ladder 2K
/// 52.1 vs 41.4 (+26%) and 16K 43.9 vs 38.1 (+15%), 2-turn pi agentic
/// (weighted) 48.6 vs 40.9 (+19%), creative temp-0.8 39.1 vs 37.9 (+3% —
/// the class that REGRESSED under the old cost model now holds even at ~30%
/// per-draft acceptance because the controller demotes without churn).
/// Users can cap rounds with `--mtp-max-depth`; the Generator's adaptive
/// controller demotes/promotes within [1, configured].
pub const DEFAULT_DEPTH: u32 = 3;
pub const MAX_DEPTH: u32 = 8;

/// `--max-mtp-ctx`: MTP stays off past this many prompt tokens (inclusive; 0 = unlimited).
/// A machine limit: `enable_mtp:true` in the body does not lift it. MTP only.
pub fn mtpCtxWithinLimit(max: u32, ctx_tokens: usize) bool {
    if (max == 0) return true;
    return ctx_tokens <= @as(usize, max);
}

/// Per-silicon adaptive depth cap for machines on the `.generic` cost
/// surface. The cap is a MACHINE measurement, so each row is one, never
/// interpolated between chips; an unmeasured chip keeps the default row.
///   M1 Pro: 4 (Qwen3.8-27B iQ-3.8bpw, forced-depth sweep:
///     13.01 tok/s at depth 4 vs 10.78/9.63 at 5/6 — the verify width 6
///     cliff; auto at cap 6 measured 10.64, barely over --no-mtp's 10.57).
/// `chip` is sysctl machdep.cpu.brand_string via `ane_mod.chipBrand`
/// (the GPU arch string cannot tell Ultra from Max); "" lands on default.
/// The row carries its own LABEL so the resolve site can say which one it
/// applied: a bare depth=4 in the spec-stats line is indistinguishable from
/// the EV controller having picked 4 on its own, or from `--mtp-max-depth 4`.
/// `measured` marks a row a HUMAN swept as realized throughput. Those beat the
/// boot probe's cost ladder, which cannot see acceptance or the extension sync
/// — see `generate.mtpDepthCapResolved`.
pub const DepthCap = struct { cap: u32, label: []const u8, measured: bool = false };

pub fn adaptiveDepthCapForMachine(chip: []const u8, default_cap: u32) DepthCap {
    if (std.mem.indexOf(u8, chip, "M1 Pro") != null) return .{ .cap = 4, .label = "m1-pro", .measured = true };
    // Base M4 (tester report, Qwen3.5-9B-MTPLX 6-bit, 16 GB):
    // saturated echo, median of 5, decode tok/s by forced depth —
    //   3: 47.44   4: 55.50   5: 47.50   6: 47.28
    // Depth 4 is the only width where the cap BINDS (m_lo == cap), so the
    // plan collapses to one chunk and pays no extension sync; from 5 on every
    // round pays it to buy ~1 more accepted token (4.00 -> 4.96/round) and
    // that trade is a 17% net LOSS there. Novel content sits at depth ~2
    // whatever the cap is, so capping at 4 costs nothing outside the echo
    // regime. The boot probe picks 6 on this chip — a cost ladder cannot see
    // either half of that sentence.
    if (std.mem.indexOf(u8, chip, "M4") != null and
        std.mem.indexOf(u8, chip, "M4 Pro") == null and
        std.mem.indexOf(u8, chip, "M4 Max") == null and
        std.mem.indexOf(u8, chip, "M4 Ultra") == null) return .{ .cap = 4, .label = "m4-base", .measured = true };
    if (std.mem.indexOf(u8, chip, "M4 Max") != null) return .{ .cap = default_cap, .label = "m4-max", .measured = true };
    // Base M5 only — the Pro/Max/Ultra dies are their own (unmeasured) rows.
    if (std.mem.indexOf(u8, chip, "M5") != null and
        std.mem.indexOf(u8, chip, "M5 Pro") == null and
        std.mem.indexOf(u8, chip, "M5 Max") == null and
        std.mem.indexOf(u8, chip, "M5 Ultra") == null) return .{ .cap = 4, .label = "m5", .measured = true };
    return .{ .cap = default_cap, .label = "default" };
}

/// Exact full-round cost surfaces known to the adaptive MTP controller.
/// Selection is based on runtime tensor geometry, never a model/repository
/// name. `generic` retains the conservative M1-M4 surface and auto cap.
pub const MtpCostProfile = enum {
    generic,
    g17_nax_q8_gs32,
    g17_nax_q4_gs32,
    g17_nax_q4_gs64,
    g17_nax_q6_gs64,
    g17_nax_q8_gs64,
    g17_nax_oq4e_q4_gs64,
    g17_nax_qwen4_q4_gs64,
    g17_nax_qwen4_mixed_4_8_gs64,
};

/// Target-side tensors that contribute materially to a complete MTP round.
/// This is deliberately a single classification rather than independent
/// booleans: one bound target can match at most one measured cost surface.
pub const MtpNaxTargetSurface = enum {
    none,
    uniform_quantized_embedding,
    uniform_bf16_embedding,
    uniform_q6_quantized_embedding,
    uniform_q8_bf16_embedding,
    oqe_quantized_embedding,
};

/// Pure first-stage classifier for a complete sidecar/target fingerprint.
/// Runtime tensor-shape validation below must still pass before the returned
/// calibrated profile is used.
pub fn m5NaxCostProfileForFingerprint(
    bits: u32,
    group_size: u32,
    target_surface: MtpNaxTargetSurface,
) MtpCostProfile {
    return switch (target_surface) {
        .uniform_quantized_embedding => if (group_size == 32) switch (bits) {
            8 => .g17_nax_q8_gs32,
            4 => .g17_nax_q4_gs32,
            else => .generic,
        } else .generic,
        .uniform_bf16_embedding => if (bits == 4 and group_size == 64)
            .g17_nax_q4_gs64
        else
            .generic,
        .uniform_q6_quantized_embedding => if (bits == 6 and group_size == 64)
            .g17_nax_q6_gs64
        else
            .generic,
        .uniform_q8_bf16_embedding => if (bits == 8 and group_size == 64)
            .g17_nax_q8_gs64
        else
            .generic,
        .oqe_quantized_embedding => if (bits == 4 and group_size == 64)
            .g17_nax_oq4e_q4_gs64
        else
            .generic,
        .none => .generic,
    };
}

/// The trunk geometry a head's fingerprint is read against.
pub const HeadDims = struct { hidden: u32, heads: u32, kv_heads: u32, head_dim: u32, intermediate: u32 };

/// A head that is the measured dense layout, at its projections' width.
pub const HeadFingerprint = struct { bits: u32, group_size: u32 };

/// Select the exact full-round cost surface for a head fingerprint (`MtpModel.m5NaxHeadFingerprint`) on `target`.
/// Every G17 profile requires the successfully built 3-bit/gs-64 draft-only lm_head and an exact dense 27B head.
/// Uniform q4/q6/q8 gs64 trunks and mixed-q4/q5/q6 oQ4e trunks remain separate profiles; target embedding storage
/// is part of the fingerprint too. Compatible but off-profile geometry remains correct under `generic`.
pub fn m5NaxProfileFor(fp: ?HeadFingerprint, target: anytype, draft: ?*const QLinear, draft_bits: u32, draft_group: u32) MtpCostProfile {
    const head = fp orelse return .generic;
    const target_surface: MtpNaxTargetSurface = if (target.mtpNaxQ4Gs64ProfileEnabled())
        .uniform_bf16_embedding
    else if (target.mtpNaxQ6Gs64ProfileEnabled())
        .uniform_q6_quantized_embedding
    else if (target.mtpNaxQ8Gs64ProfileEnabled())
        .uniform_q8_bf16_embedding
    else if (target.mtpNaxProfileEnabled())
        .uniform_quantized_embedding
    else if (target.mtpOqeNaxProfileEnabled())
        .oqe_quantized_embedding
    else
        .none;
    const profile = m5NaxCostProfileForFingerprint(head.bits, head.group_size, target_surface);
    if (profile == .generic) return .generic;
    const cfg = &target.config;
    return if (m5NaxDraftHeadMatches(draft, draft_bits, draft_group, cfg.hidden_size, cfg.vocab_size)) profile else .generic;
}

/// Prefill history windowing (OPT-IN via `--mtp-history-window <n>`; mirrors
/// others `last_window 8192` above a 16384-token threshold): prompts whose
/// forwarded tail exceeds the threshold only build MTP history for the LAST
/// n positions — earlier chunks skip the full-hidden capture AND the head
/// forward entirely (and become eligible for the compiled trunk forward).
/// A history that starts mid-sequence is already a supported state: warm
/// hot-cache hits produce exactly that (RoPE offsets are cache-relative).
/// DEFAULT IS FULL HISTORY (0): the A/B failed for windowing on the stock
/// Qwen head — 64K ctx measured 68.2% -> 54.0% per-draft acceptance and
/// -4.2 decode tok/s for zero prefill benefit.
/// `SUGGESTED_HISTORY_WINDOW` is what to pass when
/// experimenting with window-trained sidecars.
pub const SUGGESTED_HISTORY_WINDOW: usize = 8192;
pub const HISTORY_WINDOW_THRESHOLD: usize = 16384;

/// One linear: quantized (w packed u32, s/b bf16) when `s.ctx != null`,
/// otherwise a pre-transposed bf16 weight `[in, out]` for plain matmul.
pub const QLinear = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,

    pub fn deinit(self: *QLinear) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.s);
        _ = mlx.mlx_array_free(self.b);
    }
};

/// Does the Qwen head's concat projection map `[2H] -> [H]` for this trunk?
/// Dense arm: the pre-transposed weight is literally `[2H, H]`. Quantized arm:
/// the packed weight is `[H, in_packed]`, so the logical input width is solved
/// from geometry (`affineParamsFromGeometry` succeeds only when the packed
/// columns and the scales groups both agree with `2H`).
fn fcMatchesHidden(fc: *const QLinear, hidden_size: u32) bool {
    if (fc.w.ctx == null or hidden_size == 0 or hidden_size > std.math.maxInt(c_int) / 2) return false;
    const shape = mlx.getShape(fc.w);
    if (shape.len != 2) return false;
    const h: c_int = @intCast(hidden_size);
    if (fc.s.ctx == null) return shape[0] == h * 2 and shape[1] == h;
    if (shape[0] != h) return false;
    return params_mod.quantParamsFromGeometry(fc.w, fc.s, fc.b.ctx != null, hidden_size * 2) != null;
}

fn m5NaxQLinearMatches(q: *const QLinear, in_dim: u32, out_dim: u32, bits: u32, group_size: u32) bool {
    if (q.w.ctx == null or q.s.ctx == null or q.b.ctx == null) return false;
    if (mlx.mlx_array_dtype(q.w) != .uint32 or
        mlx.mlx_array_dtype(q.s) != .bfloat16 or
        mlx.mlx_array_dtype(q.b) != .bfloat16) return false;
    if (in_dim == 0 or out_dim == 0 or out_dim > std.math.maxInt(c_int)) return false;
    const out: c_int = @intCast(out_dim);
    const w_shape = mlx.getShape(q.w);
    const s_shape = mlx.getShape(q.s);
    const b_shape = mlx.getShape(q.b);
    if (w_shape.len != 2 or s_shape.len != 2 or b_shape.len != 2) return false;
    if (w_shape[0] != out or s_shape[0] != out or b_shape[0] != out) return false;
    if (s_shape[1] != b_shape[1]) return false;
    const qp = params_mod.affineParamsFromGeometry(q.w, q.s, in_dim) orelse return false;
    return qp.bits == bits and qp.group_size == group_size and qp.mode == .affine;
}

fn m5NaxNormMatches(norm: mlx.mlx_array, len: u32) bool {
    if (norm.ctx == null or mlx.mlx_array_dtype(norm) != .bfloat16) return false;
    if (len == 0 or len > std.math.maxInt(c_int)) return false;
    const shape = mlx.getShape(norm);
    return shape.len == 1 and shape[0] == @as(c_int, @intCast(len));
}

const M5NaxDenseSidecarLinears = struct {
    q: *const QLinear,
    k: *const QLinear,
    v: *const QLinear,
    o: *const QLinear,
    gate: *const QLinear,
    up: *const QLinear,
    down: *const QLinear,
};

const M5NaxDenseSidecarGeometry = struct {
    hidden: u32,
    q_out: u32,
    kv_out: u32,
    full_out: u32,
    intermediate: u32,
    bits: u32,
    group_size: u32,
};

fn m5NaxDenseSidecarMatches(linears: M5NaxDenseSidecarLinears, geom: M5NaxDenseSidecarGeometry) bool {
    return m5NaxQLinearMatches(linears.q, geom.hidden, geom.q_out, geom.bits, geom.group_size) and
        m5NaxQLinearMatches(linears.k, geom.hidden, geom.kv_out, geom.bits, geom.group_size) and
        m5NaxQLinearMatches(linears.v, geom.hidden, geom.kv_out, geom.bits, geom.group_size) and
        m5NaxQLinearMatches(linears.o, geom.full_out, geom.hidden, geom.bits, geom.group_size) and
        m5NaxQLinearMatches(linears.gate, geom.hidden, geom.intermediate, geom.bits, geom.group_size) and
        m5NaxQLinearMatches(linears.up, geom.hidden, geom.intermediate, geom.bits, geom.group_size) and
        m5NaxQLinearMatches(linears.down, geom.intermediate, geom.hidden, geom.bits, geom.group_size);
}

fn m5NaxDraftHeadMatches(
    draft: ?*const QLinear,
    bits: u32,
    group_size: u32,
    hidden_size: u32,
    vocab_size: u32,
) bool {
    const q = draft orelse return false;
    return bits == 3 and
        group_size == 64 and
        m5NaxQLinearMatches(q, hidden_size, vocab_size, 3, 64);
}

/// MTP MLP: dense SwiGLU (0.8B/27B-class sidecars) or the sparse MoE of a
/// qwen3_5_moe trunk (35B-A3B-class sidecars: router `mlp.gate` + packed
/// `switch_mlp` experts + shared expert + shared-expert gate). The MoE arm
/// stores the trunk's own `MoeMlpWeights` shape and forwards through
/// `Transformer.moeMLP` — same math, same gather-sort path, same per-weight
/// quant resolution (the sidecar mixes bits AND group sizes, e.g. 8-bit/gs-128
/// shared expert over a 4-bit/gs-64 trunk — `affineParamsFromGeometry`).
const MtpMlp = union(enum) {
    dense: struct {
        gate: QLinear,
        up: QLinear,
        down: QLinear,
    },
    moe: weights_mod.MoeMlpWeights,

    fn deinit(self: *MtpMlp) void {
        switch (self.*) {
            .dense => |*d| {
                d.gate.deinit();
                d.up.deinit();
                d.down.deinit();
            },
            .moe => |*m| {
                const arrs = [_]mlx.mlx_array{
                    m.router_w,      m.router_s,      m.router_b,
                    m.switch_gate_w, m.switch_gate_s, m.switch_gate_b,
                    m.switch_up_w,   m.switch_up_s,   m.switch_up_b,
                    m.switch_down_w, m.switch_down_s, m.switch_down_b,
                    m.shared_gate_w, m.shared_gate_s, m.shared_gate_b,
                    m.shared_up_w,   m.shared_up_s,   m.shared_up_b,
                    m.shared_down_w, m.shared_down_s, m.shared_down_b,
                };
                for (arrs) |a| _ = mlx.mlx_array_free(a);
                if (m.shared_expert_gate_w) |a| _ = mlx.mlx_array_free(a);
                if (m.shared_expert_gate_s) |a| _ = mlx.mlx_array_free(a);
                if (m.shared_expert_gate_b) |a| _ = mlx.mlx_array_free(a);
                if (m.expert_bias) |a| _ = mlx.mlx_array_free(a);
            },
        }
    }
};

/// Which head shape the sidecar loaded as. Everything the shapes share reads
/// the same fields; the few forks (attention gate, QK norms, MLP kind) key here
/// or on `eh_proj`.
pub const Layout = enum { qwen, hy3 };

pub const MtpModel = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,

    /// Quant params for the MTP layer's own linears — inferred from tensor
    /// geometry at load (sidecars are often quantized differently from the
    /// trunk, e.g. group 32 over a group-64 trunk, or 8-bit over 4-bit).
    quant_bits: u32,
    quant_group_size: u32,
    /// Quant MODE the sidecar's packed weights are in. A sidecar file carries
    /// no config, so this is solved from geometry at load
    /// (`params_mod.quantParamsFromGeometry`) and handed to every
    /// `mlx_quantized_matmul` — passing "affine" over fp8 scales throws.
    quant_mode: QuantMode = .affine,

    /// Qwen heads' concat projection (empty for Hy3). Dense bf16 ships
    /// pre-transposed `[2H, H]` for plain matmul; a checkpoint that ships it
    /// QUANTIZED (avlp12 Alis) keeps `(w, scales, biases)` verbatim — packed
    /// `[out=H, in=2H]` is already what quantized_matmul(transpose) wants.
    fc: QLinear,
    /// Hy3 (hy_v3) heads: the concat projection ships QUANTIZED as
    /// `mtp.eh_proj` instead of Qwen's bf16 `mtp.fc`. Non-null selects the
    /// Hy3 layer shape everywhere it differs: eh_proj replaces the fc matmul
    /// and the attention has NO output gate (FrontOut.gate stays a null-ctx
    /// handle; backChain skips the sigmoid multiply).
    eh_proj: ?QLinear = null,
    pre_fc_norm_emb: mlx.mlx_array, // Qwen pre_fc_norm_embedding / Hy3 enorm
    pre_fc_norm_hidden: mlx.mlx_array, // Qwen pre_fc_norm_hidden / Hy3 hnorm
    final_norm: mlx.mlx_array, // mtp.norm
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    q: QLinear,
    k: QLinear,
    v: QLinear,
    o: QLinear,
    mlp: MtpMlp,
    layout: Layout = .qwen,

    /// Optional DRAFT-ONLY low-bit lm_head, requantized from the trunk's at
    /// bind time (`DRAFT_HEAD_BITS`).
    /// Only draft steps project through it — VERIFICATION always uses the
    /// trunk head, so the output distribution is untouched; drafts just read
    /// ~40% fewer bytes per full-vocab projection (the dominant draft cost).
    draft_head: ?QLinear = null,
    draft_head_bits: u32 = 0,
    draft_head_group: u32 = 0,

    /// Draft-rerank scheme: a coarse 2-bit/gs64
    /// requant of the trunk lm_head; greedy drafts shortlist its top-32 and
    /// re-score them through the trunk head's own rows (`draftSelect`). When
    /// built, the 3-bit draft head is dropped — non-greedy draft paths fall
    /// back to the trunk head.
    rerank_coarse: ?RerankCoarse = null,
    rerank_logged: bool = false,

    /// Cross-request EV controller seed (inference thread only, like every
    /// mutable field here): the last HEALTHY request's per-index acceptance
    /// EMAs + base depth, written by `Generator.deinit`, consumed by the
    /// first `nextMtp` round of the next request. A fresh controller burns
    /// ~10 legacy-warmup rounds plus a +1/round base climb per request —
    /// a third of a short protocol-style generation; seeding restores the
    /// learned surface from round 1. Never written by disabled/short runs.
    ev_seed_accept: ?[MAX_DEPTH]f32 = null,
    ev_seed_m_lo: u32 = 1,

    pub fn deinit(self: *MtpModel) void {
        if (self.rerank_coarse) |*rc| rc.deinit();
        if (self.draft_head) |*dh| dh.deinit();
        if (self.eh_proj) |*ep| ep.deinit();
        self.fc.deinit();
        _ = mlx.mlx_array_free(self.pre_fc_norm_emb);
        _ = mlx.mlx_array_free(self.pre_fc_norm_hidden);
        _ = mlx.mlx_array_free(self.final_norm);
        _ = mlx.mlx_array_free(self.input_norm);
        _ = mlx.mlx_array_free(self.post_attn_norm);
        _ = mlx.mlx_array_free(self.q_norm);
        _ = mlx.mlx_array_free(self.k_norm);
        self.q.deinit();
        self.k.deinit();
        self.v.deinit();
        self.o.deinit();
        self.mlp.deinit();
    }

    /// A fresh single-layer KV cache for the MTP attention layer. Always
    /// dense — the head's history is small and rollback must be exact.
    pub fn makeCache(self: *const MtpModel, allocator: std.mem.Allocator) !KVCache {
        _ = self;
        return KVCache.init(allocator, 1);
    }

    /// The measured cost surface for this bound sidecar and target (`m5NaxProfileFor`).
    pub fn m5NaxCostProfile(self: *const MtpModel, target: anytype) MtpCostProfile {
        const cfg = &target.config;
        const fp = self.m5NaxHeadFingerprint(.{
            .hidden = cfg.hidden_size,
            .heads = cfg.num_attention_heads,
            .kv_heads = cfg.num_key_value_heads,
            .head_dim = cfg.head_dim,
            .intermediate = cfg.intermediate_size,
        });
        const draft: ?*const QLinear = if (self.draft_head) |*q| q else null;
        return m5NaxProfileFor(fp, target, draft, self.draft_head_bits, self.draft_head_group);
    }

    /// The head's half of the fingerprint: its projections' (bits, group) when every head tensor is the measured
    /// dense layout at that width for `d`; null otherwise.
    pub fn m5NaxHeadFingerprint(self: *const MtpModel, d: HeadDims) ?HeadFingerprint {
        if (self.eh_proj != null) return null;
        const bits = self.quant_bits;
        const group_size = self.quant_group_size;
        const full_out_wide = @as(u64, d.heads) * d.head_dim;
        const q_out_wide = full_out_wide * 2;
        const kv_out_wide = @as(u64, d.kv_heads) * d.head_dim;
        if (full_out_wide == 0 or full_out_wide > std.math.maxInt(u32) or
            q_out_wide > std.math.maxInt(u32) or
            kv_out_wide == 0 or kv_out_wide > std.math.maxInt(u32)) return null;
        if (d.hidden == 0 or d.hidden > std.math.maxInt(c_int) / 2) return null;

        const fc_shape = mlx.getShape(self.fc.w);
        if (mlx.mlx_array_dtype(self.fc.w) != .bfloat16 or
            fc_shape.len != 2 or
            fc_shape[0] != @as(c_int, @intCast(d.hidden * 2)) or
            fc_shape[1] != @as(c_int, @intCast(d.hidden))) return null;
        if (!m5NaxNormMatches(self.pre_fc_norm_emb, d.hidden) or
            !m5NaxNormMatches(self.pre_fc_norm_hidden, d.hidden) or
            !m5NaxNormMatches(self.final_norm, d.hidden) or
            !m5NaxNormMatches(self.input_norm, d.hidden) or
            !m5NaxNormMatches(self.post_attn_norm, d.hidden) or
            !m5NaxNormMatches(self.q_norm, d.head_dim) or
            !m5NaxNormMatches(self.k_norm, d.head_dim)) return null;

        switch (self.mlp) {
            .dense => |*mlp| {
                if (!m5NaxDenseSidecarMatches(
                    .{ .q = &self.q, .k = &self.k, .v = &self.v, .o = &self.o, .gate = &mlp.gate, .up = &mlp.up, .down = &mlp.down },
                    .{
                        .hidden = d.hidden,
                        .q_out = @intCast(q_out_wide),
                        .kv_out = @intCast(kv_out_wide),
                        .full_out = @intCast(full_out_wide),
                        .intermediate = d.intermediate,
                        .bits = bits,
                        .group_size = group_size,
                    },
                )) return null;
            },
            .moe => return null,
        }
        return .{ .bits = bits, .group_size = group_size };
    }

    /// Validate the head against the target trunk: dims must line up and the
    /// trunk must be a Qwen 3.5/3.6-family hybrid (full-attention MTP layer
    /// cross-checks `attn_output_gate`). On success, optionally builds the
    /// draft-only low-bit lm_head (a failed build only logs — drafts fall
    /// back to the trunk head).
    pub fn bind(self: *MtpModel, target: anytype) !void {
        const cfg = &target.config;
        if (self.eh_proj != null) {
            // Hy3 heads: no attention output gate, sigmoid-router MoE.
            if (!std.mem.eql(u8, cfg.model_type, "hy_v3")) return error.UnsupportedMtpArch;
            const en_shape = mlx.getShape(self.pre_fc_norm_emb);
            if (en_shape.len != 1 or en_shape[0] != @as(c_int, @intCast(cfg.hidden_size)))
                return error.MtpTargetMismatch;
            // The route params live on the weights struct so moeMLP2 needs no
            // config re-derivation; the loader has no config, so fill here.
            if (self.mlp == .moe and self.mlp.moe.expert_bias != null) {
                self.mlp.moe.route_norm = cfg.moe_route_norm;
                self.mlp.moe.route_scale = cfg.router_scaling_factor;
            }
            const had_err = mlx.errorPending();
            self.buildDraftHead(target) catch |err| {
                log.warn("[mtp] draft lm_head build failed ({s}) — drafts use the trunk head\n", .{@errorName(err)});
            };
            self.maybeBuildDraftRerank(target);
            mlx.dropLatchedErrorUnless(had_err); // best effort: never the first request's MlxFailure
            return;
        }
        if (!cfg.attn_output_gate) return error.UnsupportedMtpArch;
        if (!fcMatchesHidden(&self.fc, cfg.hidden_size)) return error.MtpTargetMismatch;

        const had_err = mlx.errorPending();
        self.buildDraftHead(target) catch |err| {
            log.warn("[mtp] draft lm_head build failed ({s}) — drafts use the trunk head\n", .{@errorName(err)});
        };
        self.maybeBuildDraftRerank(target);
        mlx.dropLatchedErrorUnless(had_err); // best effort: never the first request's MlxFailure
        // NOTE: mlx_compile'ing the offset-free front/back
        // halves of the seq-1 draft step (the compileMoeRouting pattern,
        // weights captured via payload) was built, verified equivalent at
        // toy scale, and A/B'd live on the 27B — DEAD EVEN at depths 3 and 6
        // (interleaved traced boots, <0.2 ms/step delta). The draft step is
        // qmm-weight-read-bound and MLX's lazy pipeline already batches
        // dispatch, so there is no launch overhead for compile to remove.
        // Removed rather than shipped dark; frontChain/backChain keep the
        // step's halves factored if a future backend changes the calculus.
    }

    pub fn buildDraftHead(self: anytype, target: anytype) !void {
        const bits = DRAFT_HEAD_BITS;
        if (bits == 0) return;
        if (target.lm_head_s.ctx == null) return; // dense bf16 head — nothing to shrink
        if (target.rht != null) return; // a requantized copy would need the rotated input at every draft site
        if (sushi_pack.isRowScaledFp8(target.lm_head_w, target.lm_head_s)) return; // e4m3 head: mlx dequantize cannot read it
        // The head's TRUE params, not the trunk global — mixed checkpoints
        // (hy_v3: 8-bit head over a 2-bit trunk) diverge, and requantizing
        // with the wrong source bits reads garbage.
        const head_qp = headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
        if (bits >= head_qp.bits) return; // no byte saving over the trunk head
        const group: u32 = 64;

        var dh = try requantizeRows(
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

        // Materialize now so the first draft doesn't pay for it.
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
        log.info("[mtp] draft-only lm_head requantized to {d}-bit/gs{d}\n", .{ bits, group });
    }

    /// Build the coarse 2-bit rerank head (see the draft-rerank section
    /// below). Infallible by design: any refusal keeps the draft-head path.
    pub fn maybeBuildDraftRerank(self: anytype, target: anytype) void {
        if (sushi_pack.isRowScaledFp8(target.lm_head_w, target.lm_head_s)) return; // e4m3 head: no affine rows to gather
        if (self.m5NaxCostProfile(target) != .generic) return;
        const w_shape = mlx.getShape(target.lm_head_w);
        if (w_shape.len != 2 or w_shape[0] < TOP32_MIN_ROWS) return;
        if (target.lm_head_s.ctx != null and headQuantParams(&target.config, target.lm_head_w, target.lm_head_s).bits <= 2) return; // nothing coarser to gain

        // The built head carries its width from here on.
        const rc = buildRerankCoarse(self.s, target, DRAFT_HEAD_BITS) orelse return;

        if (self.draft_head) |*dh| {
            dh.deinit();
            self.draft_head = null;
            self.draft_head_bits = 0;
            self.draft_head_group = 0;
        }
        self.rerank_coarse = rc;
        log.info("[mtp] draft rerank: coarse {d}-bit/gs{d} head built ({d} rows); greedy drafts re-rank through the trunk head\n", .{ rc.bits, rc.group_size, rc.rows });
    }

    pub fn canRerankDrafts(self: *const MtpModel) bool {
        return self.rerank_coarse != null;
    }

    /// One greedy draft proposal via the rerank scheme: coarse 2-bit
    /// full-vocab readout → exact top-32 shortlist (two dispatches) → the
    /// trunk head's own 32 rows re-score → argmax. Returns a lazy [1,1]
    /// int32 token id. PROPOSAL SIDE ONLY — verification still reads the
    /// trunk forward's logits, so a coarse miss costs acceptance, never
    /// output. A shortlist-kernel failure permanently drops back to the
    /// trunk-head readout (this call included).
    pub fn draftSelect(
        self: anytype,
        target: anytype,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        // The scheme itself is head-agnostic (it only ever touches the
        // TARGET's lm_head), so it lives in `rerankSelect` and the qwen4_exp
        // in-checkpoint head calls exactly the same code. Null = the coarse
        // head is gone (never built, or a shortlist failure just dropped it):
        // this arm's own full readout answers, and it can still use the
        // low-bit draft head when one is resident.
        if (try rerankSelect(self.s, target, &self.rerank_coarse, &self.rerank_logged, x, suppress_mask)) |tok| return tok;
        return MtpModel.draftFallbackArgmax(self, target, x, suppress_mask);
    }

    /// The exact re-scored top-32 shortlist for a SAMPLED draft, or null when
    /// the coarse head is gone (the caller proposes greedily instead). Same
    /// coarse readout as `draftSelect`; only the tail differs.
    pub fn draftShortlist(
        self: anytype,
        target: anytype,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !?Shortlist {
        return rerankShortlist(self.s, target, &self.rerank_coarse, &self.rerank_logged, x, suppress_mask);
    }

    /// Greedy drafts for lane rows `x` `[N,1,H]`: `[N,1]` int32 ids off the trunk head's
    /// exact readout. One N-row matmul on the head costs less than N coarse shortlists.
    pub fn draftSelectBatched(self: *MtpModel, target: anytype, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
        return fullReadoutArgmax(self.s, target, x, suppress_mask);
    }

    /// Per-lane shortlists for sampled drafts; false = the caller drafts greedily.
    pub fn draftShortlistsBatched(self: *MtpModel, target: anytype, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array, out: []Shortlist) !bool {
        return exactShortlistsBatched(self.s, target, x, suppress_mask, out);
    }

    /// Rerank's failure fallback: full trunk-head readout + argmax, same
    /// [1,1] int32 shape as `draftSelect`.
    fn draftFallbackArgmax(
        self: anytype,
        target: anytype,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        const s = self.s;
        const logits = try targetLmHead(self, target, x, s);
        defer _ = mlx.mlx_array_free(logits);
        return maskAndArgmax(s, logits, suppress_mask);
    }
};

// ── Draft rerank: the head-agnostic half ──
//
// Nothing in this scheme is MTP-head-shaped. It reads the TARGET's lm_head
// and a vector `x` in that head's input space, and answers with a token id —
// so the qwen3.5/3.6 sidecar (`MtpModel`) and the qwen4_exp in-checkpoint
// head (`Transformer.qwen4_mtp`) share it verbatim instead of the qwen4 arm
// growing a second copy that could drift from the measured one.
//
// What `x` IS, is per-arm and load-bearing: whatever the lm_head consumes.
// On the sidecar that is the head's own post-norm hidden; on qwen4_exp it is
// the MIXER output, NOT the pre-mixer stream the head hands back as
// `hidden_next`.

/// Coarse-head width AT BUILD TIME (0 disables the build). A coarser head is fewer bytes and a longer tail of
/// shortlist misses, and a miss costs acceptance, never output. The width a head is READ at is a property of its
/// packed weights (`RerankCoarse.bits`): packed geometry implies K, so a read at another width reads a different
/// vocab entirely.
pub const DRAFT_HEAD_BITS: u32 = 3;

/// Group size every coarse head is packed at.
pub const RERANK_COARSE_GROUP: u32 = 64;

/// A BUILT coarse readout head: the packed weights plus the geometry they
/// were packed with. Both live together because `mlx_quantized_matmul` needs
/// all three and only the build knows them.
pub const RerankCoarse = struct {
    q: QLinear,
    bits: u32,
    group_size: u32,
    rows: c_int,

    pub fn deinit(self: *RerankCoarse) void {
        self.q.deinit();
    }
};

/// The coarse readout head: a low-bit/gs64 requantized copy of the target's
/// lm_head, materialized so the first draft does not pay for it. Infallible —
/// every refusal returns null and the caller keeps its full-readout path.
pub fn buildRerankCoarse(s: mlx.mlx_stream, target: anytype, bits: u32) ?RerankCoarse {
    if (bits == 0) return null;
    if (comptime @hasField(@TypeOf(target.*), "rht")) {
        if (target.rht != null) return null;
    }
    if (sushi_pack.isRowScaledFp8(target.lm_head_w, target.lm_head_s)) return null; // e4m3 head: no affine rows to gather
    const w_shape = mlx.getShape(target.lm_head_w);
    if (w_shape.len != 2 or w_shape[0] < TOP32_MIN_ROWS) return null;
    // A dense bf16 head ([V, H], 16 bits) always reads more than the coarse copy.
    const dense = target.lm_head_s.ctx == null;
    const head_qp: params_mod.QuantParams = if (dense) .{ .bits = 16, .group_size = 0, .mode = .affine } else headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
    if (bits >= head_qp.bits) return null; // no byte saving over the head we already read

    var rc = requantizeRows(
        s,
        target.lm_head_w,
        target.lm_head_s,
        target.lm_head_b,
        head_qp.group_size,
        head_qp.bits,
        head_qp.mode.cstr(),
        RERANK_COARSE_GROUP,
        bits,
        32768,
    ) catch |err| {
        log.warn("[mtp] draft rerank coarse build failed ({s}) - keeping the full-readout path\n", .{@errorName(err)});
        return null;
    };
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    _ = mlx.mlx_vector_array_append_value(eval_vec, rc.w);
    _ = mlx.mlx_vector_array_append_value(eval_vec, rc.s);
    _ = mlx.mlx_vector_array_append_value(eval_vec, rc.b);
    mlx.check(mlx.mlx_eval(eval_vec)) catch {
        rc.deinit();
        log.warn("[mtp] draft rerank coarse eval failed - keeping the full-readout path\n", .{});
        return null;
    };
    return .{ .q = rc, .bits = bits, .group_size = RERANK_COARSE_GROUP, .rows = w_shape[0] };
}

/// Resident bytes of a coarse head over `rows` x `hidden` at `bits`/gs64:
/// packed weights plus bf16 scales AND biases. Reported at build so the log
/// line carries the number the change is claiming.
pub fn rerankCoarseBytes(rows: c_int, hidden: c_int, bits: u32) u64 {
    const r: u64 = @intCast(rows);
    const h: u64 = @intCast(hidden);
    const packed_bytes = r * h * @as(u64, bits) / 8;
    const groups = h / 64;
    return packed_bytes + 2 * (r * groups * 2);
}

/// `argmax(where(mask, -inf, logits))` as a lazy [.., 1] int32 id - the tail
/// every full-readout draft path shares.
pub fn maskAndArgmax(s: mlx.mlx_stream, logits_in: mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !mlx.mlx_array {
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_set(&logits, logits_in));
    if (suppress_mask) |m| {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        var masked = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
        _ = mlx.mlx_array_free(logits);
        logits = masked;
    }
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_argmax_axis(&amax, logits, -1, false, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, amax, .int32, s));
    return out;
}

/// Full trunk-lm_head readout + argmax. The qwen4 arm's fallback: unlike the
/// sidecar's, there is no low-bit draft head to prefer.
pub fn fullReadoutArgmax(
    s: mlx.mlx_stream,
    target: anytype,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !mlx.mlx_array {
    const logits = try target.lmHeadForeign(x);
    defer _ = mlx.mlx_array_free(logits);
    return maskAndArgmax(s, logits, suppress_mask);
}

/// Row i of `out` = the top-32 of the trunk head's exact readout of `x[i]` (`[N,1,H]`), so
/// nothing is re-scored. False = the top-32 kernel declined.
pub fn exactShortlistsBatched(s: mlx.mlx_stream, target: anytype, x: mlx.mlx_array, suppress_mask: ?mlx.mlx_array, out: []Shortlist) !bool {
    var logits = try target.lmHeadForeign(x);
    defer _ = mlx.mlx_array_free(logits);
    try rerankMaskCoarse(s, &logits, suppress_mask);
    const vocab = mlx.getShape(logits)[2];
    var built: usize = 0;
    errdefer for (out[0..built]) |*sl| sl.deinit();
    for (out, 0..) |*slot, i| {
        const row = try axisView(s, logits, 0, i);
        defer _ = mlx.mlx_array_free(row);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, row, &[_]c_int{vocab}, 1, s));
        const cands = draftTop32(s, flat, vocab) catch {
            for (out[0..built]) |*sl| sl.deinit();
            built = 0;
            return false;
        };
        errdefer _ = mlx.mlx_array_free(cands);
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, flat, cands, 0, s));
        var exact = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(exact);
        try mlx.check(mlx.mlx_reshape(&exact, picked, &[_]c_int{ 1, 1, TOP32_K }, 3, s));
        slot.* = .{ .cands = cands, .exact = exact, .rows = vocab };
        built = i + 1;
    }
    return true;
}

/// `-inf` on every suppressed id of a coarse readout, in place.
fn rerankMaskCoarse(s: mlx.mlx_stream, logits: *mlx.mlx_array, suppress_mask: ?mlx.mlx_array) !void {
    // The mask goes on the COARSE row: a reserved id that never enters the
    // shortlist cannot be re-scored back into the draft.
    const m = suppress_mask orelse return;
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    var masked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits.*, s));
    _ = mlx.mlx_array_free(logits.*);
    logits.* = masked;
}

/// The exact re-scored shortlist behind every rerank draft: the coarse head's
/// top-32 candidate ids plus their trunk-head logits. A greedy draft is the
/// argmax over `exact`; a sampled draft is drawn from the filtered softmax over
/// it, so q's support is exactly these ids.
pub const Shortlist = struct {
    /// `[32]` uint32 vocab ids.
    cands: mlx.mlx_array,
    /// `[.., 32]` trunk-head logits for those ids.
    exact: mlx.mlx_array,
    /// The head's vocab rows — the width a dense proposal row must have.
    rows: c_int,

    pub fn deinit(self: *Shortlist) void {
        _ = mlx.mlx_array_free(self.cands);
        _ = mlx.mlx_array_free(self.exact);
    }
};

/// Shortlist one coarse row `flat` `[rows]` and re-score it through the trunk head.
/// Null retires the coarse head, so every later step lands on the caller's fallback.
fn rerankShortlistRow(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    flat: mlx.mlx_array,
    rows: c_int,
    x_row: mlx.mlx_array,
) !?Shortlist {
    const cands = draftTop32(s, flat, rows) catch |err| {
        log.warn("[mtp] draft rerank shortlist failed ({s}) - dropping to the full readout\n", .{@errorName(err)});
        var dead = coarse.*.?;
        dead.deinit();
        coarse.* = null;
        return null;
    };
    errdefer _ = mlx.mlx_array_free(cands);

    // Re-score the shortlist through the trunk head's own rows: gathered
    // packed rows are self-contained (w [V, K*bits/32], scales/biases
    // [V, K/gs]), so a 32-row quantized matmul is exact; dense rows are a plain matmul.
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    if (target.lm_head_s.ctx == null) {
        try mlx.check(mlx.mlx_take_axis(&w32, target.lm_head_w, cands, 0, s));
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, w32, s));
        var dense = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_matmul(&dense, x_row, wt, s));
        return .{ .cands = cands, .exact = dense, .rows = rows };
    }
    const qp = headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
    var s32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_take_axis(&w32, target.lm_head_w, cands, 0, s));
    try mlx.check(mlx.mlx_take_axis(&s32, target.lm_head_s, cands, 0, s));
    if (target.lm_head_b.ctx != null)
        try mlx.check(mlx.mlx_take_axis(&b32, target.lm_head_b, cands, 0, s));

    var exact = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(exact);
    try mlx.check(mlx.mlx_quantized_matmul(
        &exact,
        x_row,
        w32,
        s32,
        b32,
        true,
        mlx.mlx_optional_int.some(@intCast(qp.group_size)),
        mlx.mlx_optional_int.some(@intCast(qp.bits)),
        qp.mode.cstr(),
        s,
    ));
    return .{ .cands = cands, .exact = exact, .rows = rows };
}

/// The greedy draft's tail: argmax over the re-scored shortlist, mapped back to
/// a vocab id as a lazy int32 array.
pub fn shortlistArgmax(s: mlx.mlx_stream, sl: Shortlist) !mlx.mlx_array {
    var amax = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(amax);
    try mlx.check(mlx.mlx_argmax_axis(&amax, sl.exact, -1, false, s));
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try mlx.check(mlx.mlx_take_axis(&picked, sl.cands, amax, 0, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, picked, .int32, s));
    return out;
}

/// The coarse full-vocab readout of one row as a flat `[rows]` array, reserved
/// ids already at `-inf` — the front half every rerank draft shares.
fn rerankCoarseFlat(
    s: mlx.mlx_stream,
    rc: *const RerankCoarse,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !mlx.mlx_array {
    var coarse_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(coarse_logits);
    try mlx.check(mlx.mlx_quantized_matmul(
        &coarse_logits,
        x,
        rc.q.w,
        rc.q.s,
        rc.q.b,
        true,
        mlx.mlx_optional_int.some(@intCast(rc.group_size)),
        // The BUILT width and group, never `DRAFT_HEAD_BITS`: the packed
        // geometry implies K, so a per-step env re-read that disagrees with the
        // build does not read a coarser vocab - it reads a different one.
        mlx.mlx_optional_int.some(@intCast(rc.bits)),
        "affine",
        s,
    ));
    try rerankMaskCoarse(s, &coarse_logits, suppress_mask);
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    const flat_shape = [_]c_int{rc.rows};
    try mlx.check(mlx.mlx_reshape(&flat, coarse_logits, &flat_shape, 1, s));
    return flat;
}

/// `rerankSelect` without its argmax tail: the caller owns the shortlist and
/// decides the proposal over it (argmax for a greedy request, a draw from the
/// draft sampler's filtered softmax for a sampled one). Null means the coarse
/// head is unavailable, exactly as in `rerankSelect`.
pub fn rerankShortlist(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !?Shortlist {
    const rc = if (coarse.*) |*p| p else return null;
    const coarse_bits = rc.bits;
    const rows = rc.rows;

    const flat = try rerankCoarseFlat(s, rc, x, suppress_mask);
    defer _ = mlx.mlx_array_free(flat);

    const sl = (try rerankShortlistRow(s, target, coarse, flat, rows, x)) orelse return null;
    if (!logged.*) {
        log.info("[mtp] draft rerank engaged ({d}-bit coarse -> top-32 -> trunk re-score)\n", .{coarse_bits});
        logged.* = true;
    }
    return sl;
}

/// One greedy draft proposal via the rerank scheme: coarse full-vocab readout
/// -> exact top-32 shortlist (two dispatches) -> the trunk head's own 32 rows
/// re-score -> argmax. Returns a lazy [1,1] int32 token id, or NULL when the
/// coarse head is unavailable - never built, or a shortlist-kernel failure
/// just dropped it (`coarse` is freed and nulled in that case, so every later
/// step of the same chain lands on the caller's fallback too).
///
/// PROPOSAL SIDE ONLY: verification still reads the trunk forward's logits, so
/// a coarse miss costs acceptance, never output.
pub fn rerankSelect(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !?mlx.mlx_array {
    var sl = (try rerankShortlist(s, target, coarse, logged, x, suppress_mask)) orelse return null;
    defer sl.deinit();
    return try shortlistArgmax(s, sl);
}

pub fn rerankCoarseRows(s: mlx.mlx_stream, x: mlx.mlx_array, rc: *const RerankCoarse) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    if (shape.len < 2 or shape.len > 3 or shape[0] < 1 or shape[0] > 32) return error.MtpRerankBatchTooWide;
    if (shape.len == 3 and shape[1] != 1) return error.MtpRerankInputShape;
    if (try verify_projection.mtpCoarsePairs(s, x, rc.q.w, rc.q.s, rc.q.b, rc.bits, rc.group_size)) |out| return out;
    if (try @import("../../blocks/linear/qmv_rows.zig").msvQmvRows(s, x, rc.q.w, rc.q.s, rc.q.b, rc.bits, rc.group_size)) |out| return out;
    const n: usize = @intCast(shape[0]);
    var parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (parts[0..n]) |p| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
    };
    for (0..n) |i| {
        var start: [3]c_int = @splat(0);
        var stop: [3]c_int = @splat(0);
        const strides: [3]c_int = @splat(1);
        @memcpy(stop[0..shape.len], shape);
        start[0] = @intCast(i);
        stop[0] = @intCast(i + 1);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, x, &start, @intCast(shape.len), &stop, @intCast(shape.len), &strides, @intCast(shape.len), s));
        parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_quantized_matmul(&parts[i], row, rc.q.w, rc.q.s, rc.q.b, true, mlx.mlx_optional_int.some(@intCast(rc.group_size)), mlx.mlx_optional_int.some(@intCast(rc.bits)), "affine", s));
    }
    const vec = mlx.mlx_vector_array_new_data(parts[0..n].ptr, n);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}
/// Per-row exact re-scored shortlists from mixer rows `x` `[N,1,H]`: ONE coarse
/// readout over N, then the top-32 select and the 32-row re-score per row.
/// False = the coarse head is unavailable (nothing is written to `out`).
pub fn rerankShortlistsBatched(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
    out: []Shortlist,
) !bool {
    const rc = if (coarse.*) |*p| p else return false;
    const xsh = mlx.getShape(x);
    const N: usize = @intCast(xsh[0]);
    if (N != out.len) return error.MtpRerankBatchLengthMismatch;
    if (N > 32) return error.MtpRerankBatchTooWide;

    var coarse_logits = try rerankCoarseRows(s, x, rc);
    defer _ = mlx.mlx_array_free(coarse_logits);
    try rerankMaskCoarse(s, &coarse_logits, suppress_mask);
    if (!logged.*) {
        log.info("[mtp] draft rerank engaged ({d}-bit coarse -> top-32 -> trunk re-score)\n", .{rc.bits});
        logged.* = true;
    }

    const rows = rc.rows;
    var built: usize = 0;
    errdefer for (out[0..built]) |*sl| sl.deinit();
    const csh = mlx.getShape(coarse_logits);
    const last = csh[csh.len - 1];
    for (0..N) |i| {
        const i_c: c_int = @intCast(i);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        if (csh.len == 3) {
            try mlx.check(mlx.mlx_slice(&row, coarse_logits, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, last }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        } else {
            try mlx.check(mlx.mlx_slice(&row, coarse_logits, &[_]c_int{ i_c, 0 }, 2, &[_]c_int{ i_c + 1, last }, 2, &[_]c_int{ 1, 1 }, 2, s));
        }
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, row, &[_]c_int{rows}, 1, s));
        var xi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xi);
        try mlx.check(mlx.mlx_slice(&xi, x, &[_]c_int{ i_c, 0, 0 }, 3, &[_]c_int{ i_c + 1, 1, xsh[xsh.len - 1] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        // A shortlist failure retires the coarse head mid-loop; the caller's
        // fallback answers for the whole group, so the rows already built go.
        out[i] = (try rerankShortlistRow(s, target, coarse, flat, rows, xi)) orelse {
            for (out[0..built]) |*sl| sl.deinit();
            return false;
        };
        built = i + 1;
    }
    return true;
}

/// Batched greedy draft ids from mixer rows `x` `[N,1,H]`. Same scheme as
/// `rerankSelect` per row; one coarse readout over N.
pub fn rerankSelectBatched(
    s: mlx.mlx_stream,
    target: anytype,
    coarse: *?RerankCoarse,
    logged: *bool,
    x: mlx.mlx_array,
    suppress_mask: ?mlx.mlx_array,
) !?mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const N: usize = @intCast(xsh[0]);
    if (N <= 1) return rerankSelect(s, target, coarse, logged, x, suppress_mask);
    if (N > 32) return error.MtpRerankBatchTooWide;

    var shortlists: [32]Shortlist = undefined;
    if (!try rerankShortlistsBatched(s, target, coarse, logged, x, suppress_mask, shortlists[0..N])) return null;
    defer for (shortlists[0..N]) |*sl| sl.deinit();

    var id_parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (id_parts[0..N]) |p| {
        if (p.ctx != null) _ = mlx.mlx_array_free(p);
    };
    for (0..N) |i| {
        const id = try shortlistArgmax(s, shortlists[i]);
        defer _ = mlx.mlx_array_free(id);
        id_parts[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&id_parts[i], id, &[_]c_int{ 1, 1 }, 2, s));
    }
    const vec = mlx.mlx_vector_array_new_data(id_parts[0..N].ptr, N);
    defer _ = mlx.mlx_vector_array_free(vec);
    var stacked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&stacked, vec, 0, s));
    return stacked;
}

pub const sidecar_rel_paths = mtp_source.sidecar_rel_paths;
pub const MtpSource = mtp_source.MtpSource;
pub const dirAdvertisesMtp = mtp_source.dirAdvertisesMtp;
pub const resolveMtpSidecarInDir = mtp_source.resolveMtpSidecarInDir;
const checkpoint_header_limit = mtp_source.checkpoint_header_limit;
const mtpShardsFromIndexJson = mtp_source.mtpShardsFromIndexJson;
const readDirFileAlloc = mtp_source.readDirFileAlloc;
pub const resolveMtpSource = mtp_source.resolveMtpSource;
pub const hasMtpHead = mtp_source.hasMtpHead;

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[mtp] missing tensor: {s}\n", .{key});
        return error.MissingMtpWeight;
    };
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, arr));
    return owned;
}

/// Fraction of `arr`'s entries that are strictly negative (0..1). Used to tell
/// a delta-encoded RMSNorm weight (many negatives) from a pre-folded one.
fn negFraction(arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var lt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lt);
    try mlx.check(mlx.mlx_less(&lt, arr, zero, s));
    var ltf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ltf);
    try mlx.check(mlx.mlx_astype(&ltf, lt, .float32, s));
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_mean(&m, ltf, false, s));
    try mlx.check(mlx.mlx_array_eval(m));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, m));
    return out;
}

/// Fold `+1` into a delta-encoded RMSNorm weight, preserving its dtype. Mirrors
/// tests/build_mtp_sidecar.py (upcast f32 → add 1 → cast back), so a folded
/// bf16 head is byte-identical to a natively-folded mlx-serve sidecar.
fn foldNormPlusOne(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const dt = mlx.mlx_array_dtype(arr);
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, f, one, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, sum, dt, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Whether the head's RMSNorm weights are stored DELTA-encoded (the layer
/// computes `1 + w`, so `w` clusters near 0 with a large NEGATIVE fraction) vs
/// pre-folded (`1 + w` baked in → strictly positive weights, which is what
/// mlx-serve's runtime `rmsnorm(x) * w` and build_mtp_sidecar.py expect). The
/// Qwen original checkpoints and oMLX's OptiQ export ship delta norms; a naive
/// copy of such a head loads but accepts ~0% (see the CLAUDE.md gotcha), so we
/// detect and fold at load. Folded RMSNorm scales are positive by construction;
/// delta ones are ~30-50% negative (every channel that downscales), and the
/// threshold sits far below that — a miss can only make the runtime acceptance
/// gate turn MTP off, never corrupt output. All norms in a head share one
/// convention, so probing a few always-present ones decides for the whole head.
fn mtpNormsAreDeltaEncoded(w: *const Weights, p: []const u8, s: mlx.mlx_stream) bool {
    const probes = [_][]const u8{
        "layers.0.input_layernorm.weight",
        "layers.0.self_attn.q_norm.weight",
        "norm.weight",
    };
    var kb: [256]u8 = undefined;
    var max_neg: f32 = 0;
    for (probes) |rest| {
        const key = std.fmt.bufPrint(&kb, "{s}mtp.{s}", .{ p, rest }) catch continue;
        const arr = w.get(key) orelse continue;
        const nf = negFraction(arr, s) catch continue;
        if (nf > max_neg) max_neg = nf;
    }
    return max_neg > NORM_DELTA_NEG_FRACTION;
}

/// Negative-fraction bar that separates a delta-encoded gamma (~30-50%
/// negative) from a folded one (positive by construction).
pub const NORM_DELTA_NEG_FRACTION: f32 = 0.05;

/// Own an RMSNorm weight, folding `+1` when the head stores delta-encoded norms.
fn ownNorm(w: *const Weights, key: []const u8, s: mlx.mlx_stream, fold: bool) !mlx.mlx_array {
    const owned = try ownWeight(w, key);
    if (!fold) return owned;
    defer _ = mlx.mlx_array_free(owned);
    return foldNormPlusOne(owned, s);
}

/// oMLX's `norm_repair` margin: a head RMSNorm whose mean sits more than this
/// below its backbone anchor is missing the `+1` zero-centered-gamma shift.
/// Mirrors `_REPAIR_MARGIN` in oMLX `patches/mlx_lm_mtp/norm_repair.py`.
pub const MTP_NORM_REPAIR_MARGIN: f32 = 0.4;

/// Mean of `arr` cast to f32 over all axes (RMSNorm gammas are 1-D). Same
/// eval-then-item pattern as `negFraction`.
fn arrayMeanF32(arr: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    var m = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(m);
    try mlx.check(mlx.mlx_mean(&m, f, false, s));
    try mlx.check(mlx.mlx_array_eval(m));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, m));
    return out;
}

/// oMLX `norm_repair` rule (pure): repair (fold `+1`) when the head norm's mean
/// sits more than the margin below its backbone anchor. A correctly-stored head
/// norm sits at/above its anchor (gap ≤ 0 → false); idempotent — after the `+1`
/// the mean lands above the anchor, so a second pass is a no-op.
/// `neg_frac` is the norm's OWN negative fraction. A FOLDED gamma is strictly
/// positive by construction (the same evidence `mtpNormsAreDeltaEncoded`
/// reads, one tensor at a time); a delta one always carries some negatives —
/// only ~0.2-0.8% on the vulnerable norms (q/k_norm, post_attn, final), which
/// is why the whole-head detector needs its 5% bar and this one must not use
/// it. The anchor gap alone cannot tell the two apart: a backbone
/// mean-of-means spans layers from 0.02 to 2.24, so a correctly folded head
/// norm can sit a long way under it.
fn mtpNormNeedsRepair(head_mean: f32, anchor: f32, neg_frac: f32) bool {
    if (neg_frac <= 0) return false;
    return anchor - head_mean > MTP_NORM_REPAIR_MARGIN;
}

/// Anchor for a vulnerable head norm = mean-of-means of the BACKBONE
/// counterpart norms carried in the same payload (non-`mtp.`, 1-D, ending in
/// `suffix`). `null` when none are present: a sidecar head ships mtp-only
/// weights, so there is no anchor and the reference repair is skipped (those
/// heads never had the oQ bug — their delta norms ride the global fold).
/// Mirrors oMLX's reference-mean pass in `norm_repair.py`.
fn mtpBackboneAnchorMean(w: *const Weights, suffix: []const u8, s: mlx.mlx_stream) ?f32 {
    var it = w.map.iterator();
    var sum: f32 = 0;
    var n: u32 = 0;
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        if (std.mem.indexOf(u8, key, "mtp.") != null) continue;
        if (!std.mem.endsWith(u8, key, suffix)) continue;
        if (mlx.mlx_array_ndim(entry.value_ptr.*) != 1) continue;
        const mean = arrayMeanF32(entry.value_ptr.*, s) catch continue;
        sum += mean;
        n += 1;
    }
    if (n == 0) return null;
    return sum / @as(f32, @floatFromInt(n));
}

/// Own a vulnerable head RMSNorm with oMLX-style reference-based repair. When
/// the global delta-fold already handled this head (`fold`), return it verbatim
/// (both would double-shift). Otherwise, if a backbone anchor exists and the
/// head norm sits a full `+1` below it (`mtpNormNeedsRepair`), fold `+1`; else
/// leave it untouched. `backbone_suffix` is the head norm's `_REPAIR_GROUPS`
/// counterpart (e.g. `mtp.norm.weight` → `model.norm.weight`).
fn ownHeadNormWithRepair(
    w: *const Weights,
    head_key: []const u8,
    backbone_suffix: []const u8,
    s: mlx.mlx_stream,
    fold: bool,
) !mlx.mlx_array {
    const owned = try ownNorm(w, head_key, s, fold);
    if (fold) return owned;
    const anchor = mtpBackboneAnchorMean(w, backbone_suffix, s) orelse return owned;
    const head_mean = arrayMeanF32(owned, s) catch return owned;
    const neg_frac = negFraction(owned, s) catch return owned;
    if (!mtpNormNeedsRepair(head_mean, anchor, neg_frac)) return owned;
    defer _ = mlx.mlx_array_free(owned);
    log.info("[mtp] repairing head norm {s}: mean {d:.3} < backbone anchor {d:.3} (+1)\n", .{ head_key, head_mean, anchor });
    return foldNormPlusOne(owned, s);
}

fn ownAndTranspose2D(w: *const Weights, key: []const u8, s: mlx.mlx_stream) !mlx.mlx_array {
    const arr = w.get(key) orelse {
        log.err("[mtp] missing tensor: {s}\n", .{key});
        return error.MissingMtpWeight;
    };
    const axes = [_]c_int{ 1, 0 };
    var t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&t, arr, &axes, 2, s));
    return t;
}

/// Head-trunk requantization width: dense
/// bf16 trunk weights (q/k/v/o, gate/up/down) are affine-quantized to this
/// width at group 64 during load — the head only PROPOSES tokens (verify
/// corrects), so the cost is acceptance, never output. 0 disables. Idea from
/// Layr-Labs/qwen-3.8-mtp-challenge @ deb63ad (see NOTICE). Test seam:
/// `head_quant_override`.
pub const DEFAULT_HEAD_QUANT_BITS: u32 = 4;
pub const HEAD_QUANT_GROUP: u32 = 64;
pub var head_quant_override: ?u32 = null;
fn headQuantBits() u32 {
    return head_quant_override orelse DEFAULT_HEAD_QUANT_BITS;
}

/// Per-load accounting for the head-trunk requant log line.
const HeadQuantStats = struct {
    n: u32 = 0,
    before_bytes: u64 = 0,
    after_bytes: u64 = 0,
};

/// Load a (possibly quantized) linear `<prefix>.{weight,scales,biases?}`.
/// bf16 weights (no scales) are pre-transposed for plain matmul.
///
/// `.biases` is optional PER TENSOR: affine is the only mode whose
/// checkpoints carry per-group biases, so an nvfp4/mxfp4/mxfp8 sidecar ships
/// `(weight, scales)` pairs only. Demanding the triple unconditionally made
/// every fp-quantized head fail at load with `missing tensor: mtp.fc.biases`
/// and silently disable MTP (ollama's `*-mlx` packs and mlx-community's
/// `*-nvfp4` sidecars are all nvfp4). A null-ctx `b` is the trunk's own
/// convention for an absent bias and rides straight into `qLinearFwd`.
fn loadLinear(w: *const Weights, allocator: std.mem.Allocator, prefix: []const u8, s: mlx.mlx_stream) !QLinear {
    var key_buf: [256]u8 = undefined;
    const scales_key = try std.fmt.bufPrint(&key_buf, "{s}.scales", .{prefix});
    if (w.get(scales_key) != null) {
        var key_buf2: [256]u8 = undefined;
        return .{
            .w = try ownWeight(w, try std.fmt.bufPrint(&key_buf2, "{s}.weight", .{prefix})),
            .s = try ownWeight(w, try std.fmt.bufPrint(&key_buf2, "{s}.scales", .{prefix})),
            .b = ownWeightOpt(w, try std.fmt.bufPrint(&key_buf2, "{s}.biases", .{prefix})),
        };
    }
    _ = allocator;
    var key_buf3: [256]u8 = undefined;
    return .{
        .w = try ownAndTranspose2D(w, try std.fmt.bufPrint(&key_buf3, "{s}.weight", .{prefix}), s),
        .s = mlx.mlx_array_new(),
        .b = mlx.mlx_array_new(),
    };
}

/// Trunk flavor of `loadLinear` (q/k/v/o, gate/up/down): a dense bf16 weight
/// whose contraction dim divides HEAD_QUANT_GROUP is requantized to
/// `headQuantBits()`/g64 packed at load (`stats` accounts it for the one log
/// line). Quantized checkpoints, indivisible widths and lever-off load
/// exactly as `loadLinear` does. fc, norms, routers and embeddings never
/// come through here (NEVER_QUANTIZE class / m5Nax fc contract).
fn loadTrunkLinear(w: *const Weights, allocator: std.mem.Allocator, prefix: []const u8, s: mlx.mlx_stream, stats: *HeadQuantStats) !QLinear {
    const bits = headQuantBits();
    if (bits != 0) {
        var key_buf: [256]u8 = undefined;
        const scales_key = try std.fmt.bufPrint(&key_buf, "{s}.scales", .{prefix});
        if (w.get(scales_key) == null) {
            const weight_key = try std.fmt.bufPrint(&key_buf, "{s}.weight", .{prefix});
            if (w.get(weight_key)) |raw| {
                const shape = mlx.getShape(raw);
                if (shape.len == 2 and shape[1] > 0 and @rem(shape[1], @as(c_int, @intCast(HEAD_QUANT_GROUP))) == 0) {
                    const none = mlx.mlx_array{ .ctx = null };
                    const lin = try requantizeRows(s, raw, none, none, 0, 0, "affine", HEAD_QUANT_GROUP, bits, 4096);
                    const rows: u64 = @intCast(shape[0]);
                    const cols: u64 = @intCast(shape[1]);
                    stats.n += 1;
                    stats.before_bytes += rows * cols * 2;
                    stats.after_bytes += rows * cols * bits / 8 + 2 * rows * (cols / HEAD_QUANT_GROUP) * 2;
                    return lin;
                }
            }
        }
    }
    return loadLinear(w, allocator, prefix, s);
}

/// Requantize a row-quantized affine weight `(w, scales, biases)` from
/// `(from_gs, from_bits)` to `(to_gs, to_bits)`, chunk-wise over rows so the
/// dequantized bf16 transient stays bounded (~chunk_rows × in_features × 2 B
/// instead of the whole matrix — a 248K×5120 lm_head would otherwise
/// materialize 2.5 GB). Rows quantize independently in MLX's affine packing,
/// so per-chunk triples concatenate along axis 0 into a valid whole.
///
/// A DENSE source (`scales.ctx == null`, e.g. a bf16 lm_head) skips the
/// dequantize and quantizes the row chunk directly — same chunking, one
/// requantizer.
pub fn requantizeRows(
    s: mlx.mlx_stream,
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    from_gs: u32,
    from_bits: u32,
    from_mode: [*:0]const u8,
    to_gs: u32,
    to_bits: u32,
    chunk_rows: c_int,
) !QLinear {
    const w_shape = mlx.getShape(w);
    if (w_shape.len != 2) return error.UnsupportedDraftHeadShape;
    const rows: c_int = w_shape[0];

    const wv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(wv);
    const sv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(sv);
    const bv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(bv);

    var r0: c_int = 0;
    while (r0 < rows) : (r0 += chunk_rows) {
        const r1: c_int = @min(rows, r0 + chunk_rows);

        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        if (scales.ctx == null) {
            var raw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(raw);
            try sliceRows(&raw, w, r0, r1, s);
            try mlx.check(mlx.mlx_astype(&dense, raw, .bfloat16, s));
        } else {
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var sq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sq);
            var bq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bq);
            try sliceRows(&wq, w, r0, r1, s);
            try sliceRows(&sq, scales, r0, r1, s);
            if (biases.ctx != null) try sliceRows(&bq, biases, r0, r1, s);
            try mlx.check(mlx.mlx_dequantize(
                &dense,
                wq,
                sq,
                bq,
                mlx.mlx_optional_int.some(@intCast(from_gs)),
                mlx.mlx_optional_int.some(@intCast(from_bits)),
                from_mode,
                .{}, // global_scale
                .{ .value = .bfloat16, .has_value = true },
                s,
            ));
        }

        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(
            &triple,
            dense,
            mlx.mlx_optional_int.some(@intCast(to_gs)),
            mlx.mlx_optional_int.some(@intCast(to_bits)),
            "affine",
            .{}, // global_scale
            s,
        ));
        if (mlx.mlx_vector_array_size(triple) != 3) return error.UnexpectedQuantizeOutput;
        var part = [3]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        try mlx.check(mlx.mlx_vector_array_get(&part[0], triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&part[1], triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&part[2], triple, 2));
        // Realize the chunk so its dense transient can be reclaimed before
        // the next chunk builds (lazy eval would otherwise stack them all).
        {
            const ev = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(ev);
            for (part) |p| _ = mlx.mlx_vector_array_append_value(ev, p);
            try mlx.check(mlx.mlx_eval(ev));
        }
        _ = mlx.mlx_vector_array_append_value(wv, part[0]);
        _ = mlx.mlx_vector_array_append_value(sv, part[1]);
        _ = mlx.mlx_vector_array_append_value(bv, part[2]);
        for (part) |p| _ = mlx.mlx_array_free(p);
    }

    var out = QLinear{
        .w = mlx.mlx_array_new(),
        .s = mlx.mlx_array_new(),
        .b = mlx.mlx_array_new(),
    };
    errdefer out.deinit();
    try mlx.check(mlx.mlx_concatenate_axis(&out.w, wv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.s, sv, 0, s));
    try mlx.check(mlx.mlx_concatenate_axis(&out.b, bv, 0, s));
    return out;
}

fn sliceRows(out: *mlx.mlx_array, src: mlx.mlx_array, r0: c_int, r1: c_int, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(src);
    const start = [_]c_int{ r0, 0 };
    const stop = [_]c_int{ r1, shape[1] };
    const strides = [_]c_int{ 1, 1 };
    try mlx.check(mlx.mlx_slice(out, src, &start, 2, &stop, 2, &strides, 2, s));
}

/// Infer the quant group size from packed-weight vs scales geometry:
/// expanded_cols = packed_cols * (32/bits); group = expanded_cols / scale_cols.
fn inferGroupSize(q: *const QLinear, bits: u32) ?u32 {
    if (q.s.ctx == null or bits == 0) return null;
    const w_shape = mlx.getShape(q.w);
    const s_shape = mlx.getShape(q.s);
    if (w_shape.len < 2 or s_shape.len < 2) return null;
    const packed_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const scale_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (scale_cols == 0) return null;
    const packed_bits = @as(u64, packed_cols) * 32;
    if (packed_bits % bits != 0) return null;
    const expanded = packed_bits / bits;
    if (expanded % scale_cols != 0) return null;
    const group_size = expanded / scale_cols;
    if (group_size > std.math.maxInt(u32)) return null;
    return @intCast(group_size);
}

/// Infer the quant BIT WIDTH from packed-weight geometry. The MTP layer's
/// linears all have `in_features == hidden` (known exactly from the bf16 fc
/// weight, `[2*hidden, hidden]`), and MLX packs along the input dim:
/// packed_cols = in_features * bits / 32  →  bits = 32 * packed_cols / hidden.
fn inferBits(q: *const QLinear, hidden: u32) ?u32 {
    if (q.s.ctx == null or hidden == 0) return null;
    const w_shape = mlx.getShape(q.w);
    if (w_shape.len < 2) return null;
    const packed_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const packed_bits = 32 * packed_cols;
    if (packed_bits % hidden != 0) return null;
    const bits = packed_bits / hidden;
    return switch (bits) {
        2, 3, 4, 5, 6, 8 => bits,
        else => null,
    };
}

/// Model-wide quant MODE of a sidecar, read off its `q_proj` — the one
/// contracted weight whose input width is known exactly (`hidden`). Per-weight
/// modes still resolve individually in `qLinearFwd`; this is only the fallback
/// for geometry that will not solve.
fn sidecarQuantMode(q: *const QLinear, hidden: u32) QuantMode {
    const qp = params_mod.quantParamsFromGeometry(q.w, q.s, q.b.ctx != null, hidden) orelse return .affine;
    return qp.mode;
}

/// Root prefix the sidecar's keys carry: mlx-serve-native sidecars use bare
/// `mtp.*`, mlx-lm-exported ones (the 35B-A3B artifacts) `language_model.mtp.*`.
fn mtpKeyPrefix(weights: *const Weights) []const u8 {
    if (weights.get("language_model.mtp.fc.weight") != null) return "language_model.";
    return "";
}

/// Own an optional tensor — absent keys become a null-ctx handle (the trunk's
/// `orelse mlx.mlx_array_new()` convention for optional scales/biases).
fn ownWeightOpt(w: *const Weights, key: []const u8) mlx.mlx_array {
    const arr = w.get(key) orelse return mlx.mlx_array_new();
    var owned = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&owned, arr);
    return owned;
}

/// Load a `<prefix>.{weight,scales?,biases?}` triple raw (no transpose) —
/// the shape the trunk's gather/qmatmul paths expect for MoE tensors.
const MoeTriple = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,

    fn deinit(self: *MoeTriple) void {
        for ([_]mlx.mlx_array{ self.w, self.s, self.b }) |a| _ = mlx.mlx_array_free(a);
    }
};

fn loadMoeTriple(w: *const Weights, prefix: []const u8) !MoeTriple {
    var key_buf: [256]u8 = undefined;
    return .{
        .w = try ownWeight(w, try std.fmt.bufPrint(&key_buf, "{s}.weight", .{prefix})),
        .s = ownWeightOpt(w, try std.fmt.bufPrint(&key_buf, "{s}.scales", .{prefix})),
        .b = ownWeightOpt(w, try std.fmt.bufPrint(&key_buf, "{s}.biases", .{prefix})),
    };
}

/// An expert bank in `loadMoeTriple`'s shape from mlx-lm's stacked `<mlp>.switch_mlp.<proj>`,
/// or from one `<mlp>.experts.N.<proj>` tensor per expert (oMLX OptiQ), stacked on axis 0. A
/// per-expert bank holds exactly the router's `experts`, its scales/biases for all or none.
fn loadMoeBank(w: *const Weights, allocator: std.mem.Allocator, mlp: []const u8, proj: []const u8, experts: usize, s: mlx.mlx_stream) !MoeTriple {
    var key_buf: [256]u8 = undefined;
    const stacked = try std.fmt.bufPrint(&key_buf, "{s}.switch_mlp.{s}", .{ mlp, proj });
    var probe_buf: [256]u8 = undefined;
    if (w.get(try std.fmt.bufPrint(&probe_buf, "{s}.weight", .{stacked})) != null) return loadMoeTriple(w, stacked);

    var parts: std.ArrayList(mlx.mlx_array) = .empty;
    defer parts.deinit(allocator);
    var t: MoeTriple = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    errdefer t.deinit();
    inline for (.{ .{ &t.w, "weight", true }, .{ &t.s, "scales", false }, .{ &t.b, "biases", false } }) |field| {
        const fmt = "{s}.experts.{d}.{s}." ++ field[1];
        parts.clearRetainingCapacity();
        var missing: ?usize = null;
        for (0..experts) |e| {
            if (w.get(try std.fmt.bufPrint(&probe_buf, fmt, .{ mlp, e, proj }))) |arr| {
                try parts.append(allocator, arr);
            } else if (missing == null) missing = e;
        }
        const past = try std.fmt.bufPrint(&probe_buf, fmt, .{ mlp, experts, proj });
        if (w.get(past) != null) {
            log.err("[mtp] {s} is past the router's {d} experts\n", .{ past, experts });
            return error.MissingMtpWeight;
        }
        if (missing) |e| if (field[2] or parts.items.len > 0) {
            log.err("[mtp] missing tensor: " ++ fmt ++ " (the router scores {d} experts)\n", .{ mlp, e, proj, experts });
            return error.MissingMtpWeight;
        };
        if (parts.items.len > 0) field[0].* = try mlx.stackEvaluated(parts.items, s);
    }
    return t;
}

/// Load the head's weights from the MAIN checkpoint: the shards the index
/// names for `mtp.*` keys (typically one), or the single
/// `model.safetensors` when there is no index. Safetensors loads are
/// lazy/mmapped — pulling a multi-GB shard in costs its header parse; only
/// the head's tensors ever materialize, and `weights.deinit()` after the
/// head build releases the rest untouched.
fn loadMtpWeightsFromCheckpoint(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !Weights {
    var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
    defer dir.close(io);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;

    if (readDirFileAlloc(io, allocator, dir, "model.safetensors.index.json", checkpoint_header_limit)) |bytes| {
        defer allocator.free(bytes);
        const shards = try mtpShardsFromIndexJson(allocator, bytes);
        defer {
            for (shards) |sh| allocator.free(sh);
            allocator.free(shards);
        }
        if (shards.len == 0) return error.MissingMtpWeight;
        var weights = Weights.init(allocator);
        errdefer weights.deinit();
        const s = mlx.weightLoadStream();
        defer _ = mlx.mlx_stream_free(s);
        for (shards) |sh| {
            const p = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ model_dir, sh });
            const pz = try allocator.dupeSentinel(u8, p, 0);
            defer allocator.free(pz);
            try weights_load.loadSafetensorsFile(allocator, &weights, pz, s, .{});
        }
        return weights;
    }
    const single = try std.fmt.bufPrint(&path_buf, "{s}/model.safetensors", .{model_dir});
    return weights_load.loadWeightsSingleFile(allocator, single);
}

/// Load the MTP head: from the model's sidecar file (any `sidecar_rel_paths`
/// layout — native `mtp/weights.safetensors` and other compatible ones) or
/// straight from the main checkpoint when the head rides the trunk shards.
pub fn loadMtp(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []const u8,
) !MtpModel {
    const source = blk: {
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        break :blk resolveMtpSource(io, allocator, dir) orelse return error.MissingMtpWeight;
    };
    var weights = switch (source) {
        .sidecar_file => |rel| blk: {
            var path_buf: [std.fs.max_path_bytes]u8 = undefined;
            const sidecar_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ model_dir, rel });
            break :blk try weights_load.loadWeightsSingleFile(allocator, sidecar_path);
        },
        .in_checkpoint => blk: {
            log.info("[mtp] loading in-checkpoint head from the trunk shards\n", .{});
            break :blk try loadMtpWeightsFromCheckpoint(io, allocator, model_dir);
        },
    };
    defer weights.deinit();

    const p = mtpKeyPrefix(&weights);
    var kb: [256]u8 = undefined;
    const K = struct {
        fn k(buf: []u8, pref: []const u8, rest: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}mtp.{s}", .{ pref, rest }) catch unreachable;
        }
    };

    // A Nemotron-H head (sevren-ai's bare keys, TensorFold's `layers.0/1`) is the nemotron_h plugin's.
    if (weights.get("attn.mixer.q_proj.weight") != null or weights.get("layers.0.eh_proj.weight") != null) return error.UnsupportedMtpArch;
    // Hy3 (hy_v3) layout: `mtp.eh_proj` + `mtp.layer.*` (full decoder layer,
    // sigmoid-router MoE). Detected by its distinctive projection name.
    if (weights.get(K.k(&kb, p, "eh_proj.weight")) != null) {
        return loadHy3Mtp(allocator, s, &weights, p);
    }

    return loadQwenMtp(allocator, s, &weights, p);
}

/// Qwen 3.5/3.6/3.8 native head (sidecar or in-checkpoint `mtp.*`): fc over
/// [pre_fc_norm(emb), pre_fc_norm(h)], one decoder layer, the trunk's lm_head.
fn loadQwenMtp(allocator: std.mem.Allocator, s: mlx.mlx_stream, weights: *const Weights, p: []const u8) !MtpModel {
    var kb: [256]u8 = undefined;
    const K = struct {
        fn k(buf: []u8, pref: []const u8, rest: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}mtp.{s}", .{ pref, rest }) catch unreachable;
        }
    };

    // MLP flavor: a router/expert pack (stacked `switch_mlp` or per-expert `experts.N`) marks
    // a MoE-trunk sidecar (35B-A3B); plain gate/up/down is the dense one-layer head.
    const is_moe = weights.get(K.k(&kb, p, "layers.0.mlp.switch_mlp.gate_proj.weight")) != null or
        weights.get(K.k(&kb, p, "layers.0.mlp.experts.0.gate_proj.weight")) != null;

    // Delta-encoded norms (Qwen original layout, oMLX OptiQ) need `+1` folded
    // in at load so the runtime `rmsnorm(x) * w` matches; a natively-folded
    // mlx-serve sidecar has strictly-positive norms and is left untouched.
    const fold_norms = mtpNormsAreDeltaEncoded(weights, p, s);
    if (fold_norms) log.info("[mtp] delta-encoded norms detected; folding +1 at load\n", .{});

    var hq_stats: HeadQuantStats = .{};
    // Every tensor lands in a local that frees itself on error; `m` below is
    // built with no fallible call and takes them all over.
    // fc via loadLinear (never loadTrunkLinear — the m5Nax profile contract
    // wants a bf16 fc): dense gets the pre-transpose, a quantized one (Alis)
    // loads verbatim.
    var fc = try loadLinear(weights, allocator, K.k(&kb, p, "fc"), s);
    errdefer fc.deinit();
    const pre_fc_norm_emb = try ownNorm(weights, K.k(&kb, p, "pre_fc_norm_embedding.weight"), s, fold_norms);
    errdefer _ = mlx.mlx_array_free(pre_fc_norm_emb);
    const pre_fc_norm_hidden = try ownNorm(weights, K.k(&kb, p, "pre_fc_norm_hidden.weight"), s, fold_norms);
    errdefer _ = mlx.mlx_array_free(pre_fc_norm_hidden);
    // The 4 norms an oQ `mean<0.5 → +1` conversion can leave a full +1 too
    // low (their raw HF means sit above 0.5): fold or repair per oMLX's
    // norm_repair — the global delta-fold when it fired, else a reference
    // anchor from the backbone counterpart. pre_fc_norm_* + input_norm are
    // always converted correctly, so they stay on plain ownNorm.
    const final_norm = try ownHeadNormWithRepair(weights, K.k(&kb, p, "norm.weight"), "model.norm.weight", s, fold_norms);
    errdefer _ = mlx.mlx_array_free(final_norm);
    const input_norm = try ownNorm(weights, K.k(&kb, p, "layers.0.input_layernorm.weight"), s, fold_norms);
    errdefer _ = mlx.mlx_array_free(input_norm);
    const post_attn_norm = try ownHeadNormWithRepair(weights, K.k(&kb, p, "layers.0.post_attention_layernorm.weight"), ".post_attention_layernorm.weight", s, fold_norms);
    errdefer _ = mlx.mlx_array_free(post_attn_norm);
    const q_norm = try ownHeadNormWithRepair(weights, K.k(&kb, p, "layers.0.self_attn.q_norm.weight"), ".self_attn.q_norm.weight", s, fold_norms);
    errdefer _ = mlx.mlx_array_free(q_norm);
    const k_norm = try ownHeadNormWithRepair(weights, K.k(&kb, p, "layers.0.self_attn.k_norm.weight"), ".self_attn.k_norm.weight", s, fold_norms);
    errdefer _ = mlx.mlx_array_free(k_norm);
    var q = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.self_attn.q_proj"), s, &hq_stats);
    errdefer q.deinit();
    var k = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.self_attn.k_proj"), s, &hq_stats);
    errdefer k.deinit();
    var v = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.self_attn.v_proj"), s, &hq_stats);
    errdefer v.deinit();
    var o = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.self_attn.o_proj"), s, &hq_stats);
    errdefer o.deinit();
    const mlp = try loadQwenMtpMlp(weights, allocator, s, p, is_moe, &hq_stats);

    var m = MtpModel{
        .allocator = allocator,
        .s = s,
        .quant_bits = 0, // inferred from tensor geometry below
        .quant_group_size = 0,
        .fc = fc,
        .pre_fc_norm_emb = pre_fc_norm_emb,
        .pre_fc_norm_hidden = pre_fc_norm_hidden,
        .final_norm = final_norm,
        .input_norm = input_norm,
        .post_attn_norm = post_attn_norm,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .q = q,
        .k = k,
        .v = v,
        .o = o,
        .mlp = mlp,
    };

    if (hq_stats.n > 0) {
        log.info("[mtp] head trunk quantized: {d} weights bf16→{d}b/g{d} ({d}→{d} MB)\n", .{
            hq_stats.n,                           headQuantBits(),
            HEAD_QUANT_GROUP,                     hq_stats.before_bytes / (1024 * 1024),
            hq_stats.after_bytes / (1024 * 1024),
        });
    } else if (headQuantBits() != 0) {
        log.debug("[mtp] head trunk already quantized — requant skipped\n", .{});
    }

    // Sidecars carry no quant metadata — infer bits from packed-column
    // geometry against the hidden size (exact: the bf16 fc weight pins
    // hidden), then group size from the scales shape. These are FALLBACK
    // globals: qLinearFwd re-solves per weight/call via
    // affineParamsFromGeometry, since sidecars mix bits AND group sizes
    // (the 35B-A3B head: q/k 5-bit gs-128, v 6-bit gs-128, o 4-bit gs-64).
    {
        // Dense fc is pre-transposed [2H, H]; a quantized one keeps its
        // [out=H, in_packed] orientation — read hidden off the axis that is
        // H in both spellings, never the packed column count.
        const fc_shape = mlx.getShape(m.fc.w);
        const hidden: u32 = if (fc_shape.len != 2)
            0
        else if (m.fc.s.ctx == null)
            @intCast(fc_shape[1])
        else
            @intCast(fc_shape[0]);
        m.quant_bits = inferBits(&m.q, hidden) orelse 4;
        m.quant_group_size = inferGroupSize(&m.q, m.quant_bits) orelse 64;
        m.quant_mode = sidecarQuantMode(&m.q, hidden);
    }

    // Materialize all weights now so first-token latency doesn't pay for it.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        const base = [_]mlx.mlx_array{
            m.fc.w,               m.fc.s,       m.fc.b,       m.pre_fc_norm_emb,
            m.pre_fc_norm_hidden, m.final_norm, m.input_norm, m.post_attn_norm,
            m.q_norm,             m.k_norm,     m.q.w,        m.k.w,
            m.v.w,                m.o.w,
        };
        for (base) |a| if (a.ctx != null) {
            _ = mlx.mlx_vector_array_append_value(eval_vec, a);
        };
        switch (m.mlp) {
            .dense => |*d| {
                _ = mlx.mlx_vector_array_append_value(eval_vec, d.gate.w);
                _ = mlx.mlx_vector_array_append_value(eval_vec, d.up.w);
                _ = mlx.mlx_vector_array_append_value(eval_vec, d.down.w);
            },
            .moe => |*mw| {
                const moe_ws = [_]mlx.mlx_array{
                    mw.router_w,      mw.switch_gate_w, mw.switch_up_w,   mw.switch_down_w,
                    mw.shared_gate_w, mw.shared_up_w,   mw.shared_down_w,
                };
                for (moe_ws) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
                if (mw.shared_expert_gate_w) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
            },
        }
        _ = mlx.mlx_eval(eval_vec);
    }

    // Bits/group here are only the degenerate-geometry FALLBACK — every
    // quantized matmul re-solves per weight (affineParamsFromGeometry), since
    // sidecars mix widths (the 35B-A3B head: 5/6-bit gs-128 q/k/v beside
    // 4-bit gs-64 o and experts).
    log.info("[mtp] loaded native MTP head ({s}; per-weight quant, fallback bits={d}/gs={d})\n", .{
        if (is_moe) "moe-mlp" else "dense-mlp",
        m.quant_bits,
        m.quant_group_size,
    });
    return m;
}

/// The Qwen head's MLP: dense gate/up/down, or the MoE-trunk sidecar's router,
/// stacked experts and gated shared expert. Frees what it loaded on error.
fn loadQwenMtpMlp(
    weights: *const Weights,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    p: []const u8,
    is_moe: bool,
    hq_stats: *HeadQuantStats,
) !MtpMlp {
    var kb: [256]u8 = undefined;
    const K = struct {
        fn k(buf: []u8, pref: []const u8, rest: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}mtp.{s}", .{ pref, rest }) catch unreachable;
        }
    };
    if (!is_moe) {
        var gate = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.mlp.gate_proj"), s, hq_stats);
        errdefer gate.deinit();
        var up = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.mlp.up_proj"), s, hq_stats);
        errdefer up.deinit();
        const down = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.mlp.down_proj"), s, hq_stats);
        return .{ .dense = .{ .gate = gate, .up = up, .down = down } };
    }
    // Router (`mlp.gate`) via loadLinear: a bf16 router gets pre-transposed
    // for the trunk's dense-matmul fallback, a quantized one loads verbatim.
    var router = try loadLinear(weights, allocator, K.k(&kb, p, "layers.0.mlp.gate"), s);
    errdefer router.deinit();
    // Packed 3D expert tensors load raw (the trunk's gather paths own the
    // orientation); 2D shared/seg linears ride loadLinear so a bf16 build
    // gets the dense pre-transpose, exactly like the trunk.
    const experts: usize = @intCast(mlx.getShape(weights.get(K.k(&kb, p, "layers.0.mlp.gate.weight")) orelse return error.MissingMtpWeight)[0]);
    var sg = try loadMoeBank(weights, allocator, K.k(&kb, p, "layers.0.mlp"), "gate_proj", experts, s);
    errdefer sg.deinit();
    var su = try loadMoeBank(weights, allocator, K.k(&kb, p, "layers.0.mlp"), "up_proj", experts, s);
    errdefer su.deinit();
    var sd = try loadMoeBank(weights, allocator, K.k(&kb, p, "layers.0.mlp"), "down_proj", experts, s);
    errdefer sd.deinit();
    var shg = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert.gate_proj"), s, hq_stats);
    errdefer shg.deinit();
    var shu = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert.up_proj"), s, hq_stats);
    errdefer shu.deinit();
    var shd = try loadTrunkLinear(weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert.down_proj"), s, hq_stats);
    errdefer shd.deinit();
    const seg = try loadLinear(weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert_gate"), s);
    return .{ .moe = .{
        .router_w = router.w,
        .router_s = router.s,
        .router_b = router.b,
        .switch_gate_w = sg.w,
        .switch_gate_s = sg.s,
        .switch_gate_b = sg.b,
        .switch_up_w = su.w,
        .switch_up_s = su.s,
        .switch_up_b = su.b,
        .switch_down_w = sd.w,
        .switch_down_s = sd.s,
        .switch_down_b = sd.b,
        .shared_gate_w = shg.w,
        .shared_gate_s = shg.s,
        .shared_gate_b = shg.b,
        .shared_up_w = shu.w,
        .shared_up_s = shu.s,
        .shared_up_b = shu.b,
        .shared_down_w = shd.w,
        .shared_down_s = shd.s,
        .shared_down_b = shd.b,
        .shared_expert_gate_w = seg.w,
        .shared_expert_gate_s = seg.s,
        .shared_expert_gate_b = seg.b,
    } };
}

/// Hy3 (hy_v3) MTP block loader — `model-mtp.safetensors` with post-sanitize
/// names: mtp.{enorm,hnorm,eh_proj,final_layernorm} + mtp.layer.* holding a
/// FULL hy3 decoder layer (8-bit attention with per-head QK norms, 2/3-bit
/// stacked experts, 8-bit sigmoid router + f32 expert_bias, 8-bit UNGATED
/// shared expert). Norms load VERBATIM — unlike the original Qwen repo,
/// nothing here is delta-encoded. route_norm/route_scale are filled at
/// bind() (the loader has no config).
fn loadHy3Mtp(
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    weights: *const Weights,
    p: []const u8,
) !MtpModel {
    var kb: [256]u8 = undefined;
    const K = struct {
        fn k(buf: []u8, pref: []const u8, rest: []const u8) []const u8 {
            return std.fmt.bufPrint(buf, "{s}mtp.{s}", .{ pref, rest }) catch unreachable;
        }
    };

    // Every tensor lands in a local that frees itself on error; `m` below is
    // built with no fallible call and takes them all over.
    var router = try loadLinear(weights, allocator, K.k(&kb, p, "layer.mlp.router.gate"), s);
    errdefer router.deinit();
    var sg = try loadMoeTriple(weights, K.k(&kb, p, "layer.mlp.experts.gate_proj"));
    errdefer sg.deinit();
    var su = try loadMoeTriple(weights, K.k(&kb, p, "layer.mlp.experts.up_proj"));
    errdefer su.deinit();
    var sd = try loadMoeTriple(weights, K.k(&kb, p, "layer.mlp.experts.down_proj"));
    errdefer sd.deinit();
    var shg = try loadLinear(weights, allocator, K.k(&kb, p, "layer.mlp.shared_mlp.gate_proj"), s);
    errdefer shg.deinit();
    var shu = try loadLinear(weights, allocator, K.k(&kb, p, "layer.mlp.shared_mlp.up_proj"), s);
    errdefer shu.deinit();
    var shd = try loadLinear(weights, allocator, K.k(&kb, p, "layer.mlp.shared_mlp.down_proj"), s);
    errdefer shd.deinit();
    var eh_proj = try loadLinear(weights, allocator, K.k(&kb, p, "eh_proj"), s);
    errdefer eh_proj.deinit();
    const enorm = try ownWeight(weights, K.k(&kb, p, "enorm.weight"));
    errdefer _ = mlx.mlx_array_free(enorm);
    const hnorm = try ownWeight(weights, K.k(&kb, p, "hnorm.weight"));
    errdefer _ = mlx.mlx_array_free(hnorm);
    const final_norm = try ownWeight(weights, K.k(&kb, p, "final_layernorm.weight"));
    errdefer _ = mlx.mlx_array_free(final_norm);
    const input_norm = try ownWeight(weights, K.k(&kb, p, "layer.input_layernorm.weight"));
    errdefer _ = mlx.mlx_array_free(input_norm);
    const post_attn_norm = try ownWeight(weights, K.k(&kb, p, "layer.post_attention_layernorm.weight"));
    errdefer _ = mlx.mlx_array_free(post_attn_norm);
    const q_norm = try ownWeight(weights, K.k(&kb, p, "layer.self_attn.q_norm.weight"));
    errdefer _ = mlx.mlx_array_free(q_norm);
    const k_norm = try ownWeight(weights, K.k(&kb, p, "layer.self_attn.k_norm.weight"));
    errdefer _ = mlx.mlx_array_free(k_norm);
    var q = try loadLinear(weights, allocator, K.k(&kb, p, "layer.self_attn.q_proj"), s);
    errdefer q.deinit();
    var k = try loadLinear(weights, allocator, K.k(&kb, p, "layer.self_attn.k_proj"), s);
    errdefer k.deinit();
    var v = try loadLinear(weights, allocator, K.k(&kb, p, "layer.self_attn.v_proj"), s);
    errdefer v.deinit();
    var o = try loadLinear(weights, allocator, K.k(&kb, p, "layer.self_attn.o_proj"), s);
    errdefer o.deinit();
    const expert_bias = try ownWeight(weights, K.k(&kb, p, "layer.mlp.expert_bias"));
    errdefer _ = mlx.mlx_array_free(expert_bias);

    var m = MtpModel{
        .allocator = allocator,
        .s = s,
        .quant_bits = 0,
        .quant_group_size = 0,
        .fc = .{ .w = .{ .ctx = null }, .s = .{ .ctx = null }, .b = .{ .ctx = null } },
        .eh_proj = eh_proj,
        .pre_fc_norm_emb = enorm,
        .pre_fc_norm_hidden = hnorm,
        .final_norm = final_norm,
        .input_norm = input_norm,
        .post_attn_norm = post_attn_norm,
        .q_norm = q_norm,
        .k_norm = k_norm,
        .q = q,
        .k = k,
        .v = v,
        .o = o,
        .layout = .hy3,
        .mlp = .{ .moe = .{
            .router_w = router.w,
            .router_s = router.s,
            .router_b = router.b,
            .switch_gate_w = sg.w,
            .switch_gate_s = sg.s,
            .switch_gate_b = sg.b,
            .switch_up_w = su.w,
            .switch_up_s = su.s,
            .switch_up_b = su.b,
            .switch_down_w = sd.w,
            .switch_down_s = sd.s,
            .switch_down_b = sd.b,
            .shared_gate_w = shg.w,
            .shared_gate_s = shg.s,
            .shared_gate_b = shg.b,
            .shared_up_w = shu.w,
            .shared_up_s = shu.s,
            .shared_up_b = shu.b,
            .shared_down_w = shd.w,
            .shared_down_s = shd.s,
            .shared_down_b = shd.b,
            .expert_bias = expert_bias,
            .shared_ungated = true,
        } },
    };

    // Fallback quant globals from the q projection geometry (hidden pinned by
    // the enorm length); every matmul re-solves per weight anyway.
    {
        const en_shape = mlx.getShape(m.pre_fc_norm_emb);
        const hidden: u32 = if (en_shape.len == 1) @intCast(en_shape[0]) else 0;
        m.quant_bits = inferBits(&m.q, hidden) orelse 8;
        m.quant_group_size = inferGroupSize(&m.q, m.quant_bits) orelse 64;
        m.quant_mode = sidecarQuantMode(&m.q, hidden);
    }

    // Materialize now so the first draft doesn't pay for it.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        const base = [_]mlx.mlx_array{
            m.eh_proj.?.w, m.pre_fc_norm_emb, m.pre_fc_norm_hidden, m.final_norm,
            m.input_norm,  m.post_attn_norm,  m.q_norm,             m.k_norm,
            m.q.w,         m.k.w,             m.v.w,                m.o.w,
        };
        for (base) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
        const mw = &m.mlp.moe;
        const moe_ws = [_]mlx.mlx_array{
            mw.router_w,      mw.switch_gate_w, mw.switch_up_w,   mw.switch_down_w,
            mw.shared_gate_w, mw.shared_up_w,   mw.shared_down_w,
        };
        for (moe_ws) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
        if (mw.expert_bias) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
        _ = mlx.mlx_eval(eval_vec);
    }

    log.info("[mtp] loaded Hy3 MTP head (sigmoid-MoE layer; per-weight quant, fallback bits={d}/gs={d})\n", .{
        m.quant_bits,
        m.quant_group_size,
    });
    return m;
}

// ── Forward ──

/// The head is the layer after the trunk's last; a NoPE arch (Nemotron-H)
/// marks it like the trunk layers.
inline fn headSkipsRope(cfg: anytype) bool {
    return cfg.layerSkipsRope(cfg.num_hidden_layers);
}

inline fn rmsNormFn(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&out, x, w, eps, s));
    return out;
}

/// Quantized (or pre-transposed bf16) linear projection. Quant params are
/// solved PER WEIGHT from packed-column geometry against the activation's
/// inner dim (sidecars mix bits and group sizes across tensors — the
/// 35B-A3B head runs 5-bit/gs-128 q/k beside 4-bit/gs-64 o); the load-time
/// globals are only the fallback for degenerate geometry.
fn qLinearFwd(self: *const MtpModel, x: mlx.mlx_array, lin: *const QLinear) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    if (lin.s.ctx == null) {
        try mlx.check(mlx.mlx_matmul(&out, x, lin.w, self.s));
        return out;
    }
    var bits = self.quant_bits;
    var group = self.quant_group_size;
    var mode = self.quant_mode;
    const x_shape = mlx.getShape(x);
    if (x_shape.len > 0 and x_shape[x_shape.len - 1] > 0) {
        const in_dim: u32 = @intCast(x_shape[x_shape.len - 1]);
        if (params_mod.quantParamsFromGeometry(lin.w, lin.s, lin.b.ctx != null, in_dim)) |qp| {
            bits = qp.bits;
            group = qp.group_size;
            mode = qp.mode;
        }
    }
    // Multi-row head forwards (the merged history+draft consume, prefill
    // history rebuilds) ride the same verify-width split-K kernel as the
    // trunk; ineligible shapes (seq 1 drafts, 5/6-bit MoE sidecars) fall
    // through to stock.
    // The verify lanes are affine-only kernels (they unpack affine
    // scales/biases); fp modes fall straight through to stock.
    if (mode == .affine) {
        if (try verify_qmm.verifyQmm(self.s, x, lin.w, lin.s, lin.b, bits, group)) |vy| {
            _ = mlx.mlx_array_free(out);
            return vy;
        }
    }
    try mlx.check(mlx.mlx_quantized_matmul(
        &out,
        x,
        lin.w,
        lin.s,
        lin.b,
        true,
        mlx.mlx_optional_int.some(@intCast(group)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        mode.cstr(),
        self.s,
    ));
    return out;
}

/// Embed `[n]`-shaped int32 token ids through the TARGET's embedding table
/// → `[1, n, H]` bf16. Mirrors `Transformer.embedding` (quantized) with a
/// dense-bf16 fallback. No embed scaling — Qwen does not scale embeddings.
fn embedTargetTokens(
    target: anytype,
    id_arr: mlx.mlx_array,
    n: c_int,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const hidden: c_int = @intCast(target.config.hidden_size);
    const out_shape = [_]c_int{ 1, n, hidden };

    var tw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tw);
    try mlx.check(mlx.mlx_take_axis(&tw, target.emb_w, id_arr, 0, s));

    if (target.emb_s.ctx == null) {
        var emb_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(emb_b);
        try mlx.check(mlx.mlx_astype(&emb_b, tw, .bfloat16, s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&out, emb_b, &out_shape, 3, s));
        return out;
    }

    var ts = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ts);
    try mlx.check(mlx.mlx_take_axis(&ts, target.emb_s, id_arr, 0, s));
    // Bias-less trunk quant modes (nvfp4 etc.) have a null-ctx emb_b.
    var tb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tb);
    if (target.emb_b.ctx != null) {
        try mlx.check(mlx.mlx_take_axis(&tb, target.emb_b, id_arr, 0, s));
    }

    // Resolve the embed table's OWN quant params from geometry, not the trunk's
    // global `config.quant_bits` — a mixed-precision checkpoint (oMLX OptiQ)
    // quantizes embed_tokens to 8-bit while the base is 4-bit, and dequantizing
    // an 8-bit table as 4-bit crashes (`scales/biases shape mismatch`). Mirrors
    // `Transformer.embedding` → `quantParamsHinted`. Uniform-4-bit checkpoints
    // (ddalcu, MTPLX) resolve to the same 4/gs64/affine, so they're unchanged.
    const emb_qp = params_mod.computeQuantParams(.of(&target.config), target.emb_w, target.emb_s, target.config.hidden_size);
    var dequant = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dequant);
    try mlx.check(mlx.mlx_dequantize(
        &dequant,
        tw,
        ts,
        tb,
        mlx.mlx_optional_int.some(@intCast(emb_qp.group_size)),
        mlx.mlx_optional_int.some(@intCast(emb_qp.bits)),
        emb_qp.mode.cstr(),
        .{}, // global_scale
        .{ .value = .bfloat16, .has_value = true },
        s,
    ));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, dequant, &out_shape, 3, s));
    return target.unrotateEmbedding(out);
}

/// Project the MTP post-norm hidden through the lm_head. Draft steps go
/// through the low-bit draft-only head when one was built (verification
/// never routes here — trunk logits come from the trunk forward).
pub fn targetLmHead(self: anytype, target: anytype, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
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
            s,
        ));
        return out;
    }
    return target.lmHeadForeign(x);
}

/// The trunk lm_head's TRUE quant params, via the same dtype-gated resolver
/// the trunk itself uses: uint8 scales → the config's non-affine mode (a raw
/// geometry solve mis-reads mxfp8 8-bit/gs32 as AFFINE 8-bit/gs32 — issue
/// #81, "Biases must be provided" crash on biasless heads); float scales →
/// exact per-geometry affine solve (the head's bits routinely differ from
/// the trunk global on mixed checkpoints — hy_v3 ships an 8-bit head over a
/// 2-bit trunk).
pub fn headQuantParams(config: anytype, w: mlx.mlx_array, sc: mlx.mlx_array) params_mod.QuantParams {
    return params_mod.computeQuantParams(.of(config), w, sc, config.hidden_size);
}

/// Outputs of the pre-rope half of the MTP layer. All arrays owned.
const FrontOut = struct {
    q_t: mlx.mlx_array, // [1, H, L, D] normed, pre-rope
    k_t: mlx.mlx_array, // [1, Hkv, L, D] normed, pre-rope
    v_t: mlx.mlx_array, // [1, Hkv, L, D]
    gate: mlx.mlx_array, // [1, L, H, D] raw output gate (pre-sigmoid, strided split view)
    x: mlx.mlx_array, // [1, L, H] fc output — the residual input

    fn deinit(self: *FrontOut) void {
        _ = mlx.mlx_array_free(self.q_t);
        _ = mlx.mlx_array_free(self.k_t);
        _ = mlx.mlx_array_free(self.v_t);
        _ = mlx.mlx_array_free(self.gate);
        _ = mlx.mlx_array_free(self.x);
    }
};

/// Pre-rope half of the MTP layer: fc(concat([norm(embed), norm(hidden)])),
/// input_norm, q/k/v projections, q/gate split, per-head norms, transposes.
/// Offset-free — the body the compiled front closure traces AND the
/// uncompiled fallback.
/// The fusion stub shared by the full layer forward and the KV-only history
/// append: x = fc(concat([norm(embed ids), norm(hidden)])). Returns owned x.
/// `x` in `dt` as an owned handle (a retained reference when it already is).
fn castOwned(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    if (mlx.mlx_array_dtype(x) == dt) {
        try mlx.check(mlx.mlx_array_set(&out, x));
    } else {
        try mlx.check(mlx.mlx_astype(&out, x, dt, s));
    }
    return out;
}

fn fcConcat(self: *const MtpModel, target: anytype, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, seq_len: c_int) !mlx.mlx_array {
    const s = self.s;
    const eps = target.config.rms_norm_eps;
    // A grafted head can run in another dtype than the trunk (a bf16 head on
    // an f16 Hadamard pack): its inputs enter in the head's own dtype.
    const head_dt = mlx.mlx_array_dtype(self.pre_fc_norm_hidden);
    const emb_raw = try embedTargetTokens(target, id_arr, seq_len, s);
    defer _ = mlx.mlx_array_free(emb_raw);
    const emb = try castOwned(emb_raw, head_dt, s);
    defer _ = mlx.mlx_array_free(emb);
    const hid = try castOwned(hidden, head_dt, s);
    defer _ = mlx.mlx_array_free(hid);
    const e_normed = try rmsNormFn(emb, self.pre_fc_norm_emb, eps, s);
    defer _ = mlx.mlx_array_free(e_normed);
    const h_normed = try rmsNormFn(hid, self.pre_fc_norm_hidden, eps, s);
    defer _ = mlx.mlx_array_free(h_normed);

    var cat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cat);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, e_normed);
        _ = mlx.mlx_vector_array_append_value(vec, h_normed);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 2, s));
    }
    if (self.eh_proj) |*ep| {
        // Hy3: the concat projection is quantized (mtp.eh_proj, 8-bit).
        return qLinearFwd(self, cat, ep);
    }
    // Dense fc keeps the plain matmul (qLinearFwd's own dense arm); a
    // quantized fc (Alis packs) takes the quantized path unchanged.
    return qLinearFwd(self, cat, &self.fc);
}

fn frontChain(self: *const MtpModel, target: anytype, id_arr: mlx.mlx_array, hidden: mlx.mlx_array) !FrontOut {
    const s = self.s;
    const cfg = &target.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const kv_h: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const eps = cfg.rms_norm_eps;
    const h_shape = mlx.getShape(hidden);
    const seq_len: c_int = h_shape[1];
    const x = try fcConcat(self, target, id_arr, hidden, seq_len);
    errdefer _ = mlx.mlx_array_free(x);

    // ── Decoder layer: full attention (Qwen: gated q; Hy3: plain q) ──
    const normed = try rmsNormFn(x, self.input_norm, eps, s);
    defer _ = mlx.mlx_array_free(normed);

    const q_proj = try qLinearFwd(self, normed, &self.q);
    defer _ = mlx.mlx_array_free(q_proj);

    var queries = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(queries);
    var gate = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(gate);
    if (self.eh_proj != null) {
        // Hy3: no attention output gate — q_proj IS the queries. `gate`
        // stays a null-ctx handle; backChain skips the sigmoid multiply.
        const q_shape = [_]c_int{ 1, seq_len, h_count, hd };
        try mlx.check(mlx.mlx_reshape(&queries, q_proj, &q_shape, 4, s));
    } else {
        // q_proj is [1, L, 2*H*D]: reshape to [1, L, H, 2D], split → (queries, gate)
        const q_gate_shape = [_]c_int{ 1, seq_len, h_count, hd * 2 };
        var q_gate_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_gate_r);
        try mlx.check(mlx.mlx_reshape(&q_gate_r, q_proj, &q_gate_shape, 4, s));

        var split_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(split_vec);
        try mlx.check(mlx.mlx_split(&split_vec, q_gate_r, 2, -1, s));
        if (mlx.mlx_vector_array_size(split_vec) != 2) return error.UnexpectedSplitCount;
        try mlx.check(mlx.mlx_vector_array_get(&queries, split_vec, 0));

        // The gate STAYS 4-D — flattening the split view is a REAL Copy
        // kernel per draft step (backChain multiplies it strided for free).
        try mlx.check(mlx.mlx_vector_array_get(&gate, split_vec, 1));
    }

    const k_proj = try qLinearFwd(self, normed, &self.k);
    defer _ = mlx.mlx_array_free(k_proj);
    const v_proj = try qLinearFwd(self, normed, &self.v);
    defer _ = mlx.mlx_array_free(v_proj);

    const kv_shape = [_]c_int{ 1, seq_len, kv_h, hd };
    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, s));
    try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, s));

    const q_normed = try rmsNormFn(queries, self.q_norm, eps, s);
    defer _ = mlx.mlx_array_free(q_normed);
    const k_normed = try rmsNormFn(k_r, self.k_norm, eps, s);
    defer _ = mlx.mlx_array_free(k_normed);

    const perm = [_]c_int{ 0, 2, 1, 3 };
    var q_t = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q_t);
    var k_t = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(k_t);
    var v_t = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_transpose_axes(&q_t, q_normed, &perm, 4, s));
    try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, s));
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, s));

    return .{ .q_t = q_t, .k_t = k_t, .v_t = v_t, .gate = gate, .x = x };
}

/// Post-sdpa half of the MTP layer: output gate, o_proj, residual,
/// post_attn_norm, MLP, residual, final_norm. Offset-free — the body the
/// compiled back closure traces AND the uncompiled fallback. Returns the
/// post-final-norm hidden (owned).
fn backChain(self: *const MtpModel, target: anytype, attn_out: mlx.mlx_array, gate: mlx.mlx_array, x: mlx.mlx_array, seq_len: c_int) !mlx.mlx_array {
    const s = self.s;
    const cfg = &target.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const eps = cfg.rms_norm_eps;
    const flat_shape = [_]c_int{ 1, seq_len, h_count * hd };
    const perm = [_]c_int{ 0, 2, 1, 3 };

    var attn_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_t);
    try mlx.check(mlx.mlx_transpose_axes(&attn_t, attn_out, &perm, 4, s));

    // Output gate: o_proj(attn * sigmoid(gate)); ungated archs (Hy3) pass a
    // null-ctx gate → straight o_proj(attn). The gated arm multiplies 4-D
    // (strided views are copy-free in the elementwise kernel) and flattens
    // the CONTIGUOUS product — flattening attn_t/gate first paid two REAL
    // Copy kernels per draft step.
    const o_out = if (gate.ctx == null) blk: {
        var attn_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(attn_flat);
        try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, s));
        break :blk try qLinearFwd(self, attn_flat, &self.o);
    } else blk: {
        var gate_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_sig);
        try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate, s));
        var gated_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated_4d);
        try mlx.check(mlx.mlx_multiply(&gated_4d, attn_t, gate_sig, s));
        var gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated);
        try mlx.check(mlx.mlx_reshape(&gated, gated_4d, &flat_shape, 3, s));
        break :blk try qLinearFwd(self, gated, &self.o);
    };
    defer _ = mlx.mlx_array_free(o_out);

    var h1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h1);
    try mlx.check(mlx.mlx_add(&h1, x, o_out, s));

    // MLP: dense SwiGLU, or the trunk's own sparse-MoE forward (router +
    // switch experts + shared expert — same math/quant resolution as a
    // trunk qwen3_5_moe layer).
    const ff_normed = try rmsNormFn(h1, self.post_attn_norm, eps, s);
    defer _ = mlx.mlx_array_free(ff_normed);
    const mlp_out = switch (self.mlp) {
        .dense => |*d| blk: {
            const g = try qLinearFwd(self, ff_normed, &d.gate);
            defer _ = mlx.mlx_array_free(g);
            const up = try qLinearFwd(self, ff_normed, &d.up);
            defer _ = mlx.mlx_array_free(up);
            var g_sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g_sig);
            try mlx.check(mlx.mlx_sigmoid(&g_sig, g, s));
            var g_silu = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(g_silu);
            try mlx.check(mlx.mlx_multiply(&g_silu, g, g_sig, s));
            var act = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(act);
            try mlx.check(mlx.mlx_multiply(&act, g_silu, up, s));
            break :blk try qLinearFwd(self, act, &d.down);
        },
        .moe => |*mw| try target.moeMLP(ff_normed, mw),
    };
    defer _ = mlx.mlx_array_free(mlp_out);

    var x_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_out);
    try mlx.check(mlx.mlx_add(&x_out, h1, mlp_out, s));

    return rmsNormFn(x_out, self.final_norm, eps, s);
}

pub const StepOut = struct {
    /// `[1, 1, vocab]` LAST-row logits, or `.ctx == null` when the caller did
    /// not ask for `.logits` (multi-row calls never project the history rows —
    /// only the last row feeds the draft chain).
    logits: mlx.mlx_array,
    /// MTP post-norm hidden — the next depth's `hidden` input. `[1, 1, H]`
    /// (last row) under `.logits`, the full `[1, L, H]` otherwise.
    hidden_next: mlx.mlx_array,
    /// The vector the TARGET's lm_head consumes, for `.mixed` steps. Null on
    /// every arm where that is just `hidden_next` (the sidecar); non-null on
    /// qwen4_exp, whose `hidden_next` is the PRE-mixer stream and whose
    /// lm_head input is the mixer output. Owned by the caller when non-null.
    rerank_x: mlx.mlx_array = .{ .ctx = null },
};

/// What a head step owes its caller. `want_logits: bool` could not tell the
/// two zero-logits cases apart: a history append wants nothing past the
/// stream, while a rerank draft still needs the vector the lm_head consumes.
pub const StepWant = enum {
    /// Last-row logits (and the last-row hidden).
    logits,
    /// No logits, but the lm_head's INPUT vector for the last row — the
    /// rerank draft path.
    mixed,
    /// Nothing past the stream: committed-history appends.
    none,
};

pub const MropeContext = mrope.PositionContext;

/// Core MTP forward over `L` positions.
///
/// `id_arr`     — `[L]` int32 token ids (may be a lazy array mid-chain)
/// `hidden`     — `[1, L, H]` trunk (depth 1) or MTP (depth >1) hidden states
/// `cache`      — the head's own single-layer KV cache; entries appended here
/// `rope_offset`— RoPE position of the FIRST of the L tokens (cache-relative)
///
/// Appends L entries to `cache`. Multi-token calls use a causal mask
/// (bottom-right aligned, matching trunk chunked prefill).
pub fn forward(
    self: *const MtpModel,
    target: anytype,
    cache: *KVCache,
    id_arr: mlx.mlx_array,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
    want_logits: bool,
) !StepOut {
    return forwardWithMrope(self, target, cache, id_arr, hidden, rope_offset, want_logits, null);
}

/// MTP forward with an optional mapping from the head's cache-relative offsets
/// to the target's M-RoPE positions. Image-prompt positions use the explicit
/// three-axis table. Once decoding moves past that table, generated text uses
/// the scalar `absolute + delta` fast-RoPE path.
pub var mtp_kv_only_override: ?bool = null; // test seam

/// KV-only history rows ship ON (mlxfast-challenge lastHiddenWithKVOnlyHistory
/// class): a committed-history row's layer OUTPUT has no consumer — only its
/// K/V entries matter — so the q(+gate) projection, the attention, and the
/// whole post-attention half are dead work for every row but the last.
fn mtpKvOnlyEnabled() bool {
    return mtp_kv_only_override orelse true;
}

/// Append `hidden`/`id_arr` rows to the head's cache through the K/V-only
/// path: fc fusion + input_norm + K/V projections + k_norm + rope + cache
/// update. No query, no attention, no MLP — nothing downstream reads those
/// rows' outputs. Byte parity with the full path is NOT the bar (a different
/// GEMM M reorders reductions); the head only proposes, so the cost of any
/// near-tie flip is acceptance, which the live equivalence script gates.
pub fn appendKvOnly(
    self: *const MtpModel,
    target: anytype,
    cache: *KVCache,
    id_arr: mlx.mlx_array,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
    mrope_ctx: ?MropeContext,
) !void {
    const s = self.s;
    const cfg = &target.config;
    const kv_h: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const eps = cfg.rms_norm_eps;
    const h_shape = mlx.getShape(hidden);
    const seq_len: c_int = h_shape[1];
    const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);

    const x = try fcConcat(self, target, id_arr, hidden, seq_len);
    defer _ = mlx.mlx_array_free(x);
    const normed = try rmsNormFn(x, self.input_norm, eps, s);
    defer _ = mlx.mlx_array_free(normed);

    const k_proj = try qLinearFwd(self, normed, &self.k);
    defer _ = mlx.mlx_array_free(k_proj);
    const v_proj = try qLinearFwd(self, normed, &self.v);
    defer _ = mlx.mlx_array_free(v_proj);

    const kv_shape = [_]c_int{ 1, seq_len, kv_h, hd };
    var k_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_r);
    var v_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_r);
    try mlx.check(mlx.mlx_reshape(&k_r, k_proj, &kv_shape, 4, s));
    try mlx.check(mlx.mlx_reshape(&v_r, v_proj, &kv_shape, 4, s));

    const k_normed = try rmsNormFn(k_r, self.k_norm, eps, s);
    defer _ = mlx.mlx_array_free(k_normed);
    const perm = [_]c_int{ 0, 2, 1, 3 };
    var k_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_t);
    var v_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_t);
    try mlx.check(mlx.mlx_transpose_axes(&k_t, k_normed, &perm, 4, s));
    try mlx.check(mlx.mlx_transpose_axes(&v_t, v_r, &perm, 4, s));

    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    const relative_offset: usize = @intCast(rope_offset);
    const needs_explicit_mrope = if (mrope_ctx) |positions|
        positions.absolutePosition(relative_offset) < positions.total
    else
        false;
    if (headSkipsRope(cfg)) {
        try mlx.check(mlx.mlx_array_set(&k_rope, k_t));
    } else if (needs_explicit_mrope) {
        const positions = mrope_ctx.?;
        const cs = try target.buildMropeCosSin(positions, relative_offset, @intCast(seq_len));
        defer _ = mlx.mlx_array_free(cs.cos);
        defer _ = mlx.mlx_array_free(cs.sin);
        _ = mlx.mlx_array_free(k_rope);
        k_rope = try target.applyMrope(k_t, cs.cos, cs.sin, rope_dims);
    } else {
        const effective_offset: c_int = if (mrope_ctx) |positions|
            @intCast(@as(i64, @intCast(positions.absolutePosition(relative_offset))) + positions.delta)
        else
            rope_offset;
        try mlx.check(mlx.mlx_fast_rope(&k_rope, k_t, rope_dims, false, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, effective_offset, .{ .ctx = null }, s));
    }

    var kv_view = try cache.update(0, k_rope, v_t, s, 0);
    kv_view.deinit();
}

pub fn forwardWithMrope(
    self: *const MtpModel,
    target: anytype,
    cache: *KVCache,
    id_arr: mlx.mlx_array,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
    want_logits: bool,
    mrope_ctx: ?MropeContext,
) !StepOut {
    const s = self.s;
    const cfg = &target.config;
    const hidden_size: c_int = @intCast(cfg.hidden_size);
    const h_shape = mlx.getShape(hidden);
    const seq_len: c_int = h_shape[1];

    // Merged history+draft forward: only the LAST row's output has a consumer
    // (the logits slice below already said so). Flush the leading rows
    // through the K/V-only path, then run the full layer on the last row
    // alone — it attends the whole cache, so the math it feeds the draft
    // chain sees the same history.
    if (want_logits and seq_len > 1 and mtpKvOnlyEnabled()) {
        const strides1 = [_]c_int{1};
        var ids_head = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_head);
        var ids_last = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_last);
        {
            const start0 = [_]c_int{0};
            const stop0 = [_]c_int{seq_len - 1};
            try mlx.check(mlx.mlx_slice(&ids_head, id_arr, &start0, 1, &stop0, 1, &strides1, 1, s));
            const start1 = [_]c_int{seq_len - 1};
            const stop1 = [_]c_int{seq_len};
            try mlx.check(mlx.mlx_slice(&ids_last, id_arr, &start1, 1, &stop1, 1, &strides1, 1, s));
        }
        const strides3 = [_]c_int{ 1, 1, 1 };
        var hid_head = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hid_head);
        var hid_last = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hid_last);
        {
            const start0 = [_]c_int{ 0, 0, 0 };
            const stop0 = [_]c_int{ 1, seq_len - 1, hidden_size };
            try mlx.check(mlx.mlx_slice(&hid_head, hidden, &start0, 3, &stop0, 3, &strides3, 3, s));
            const start1 = [_]c_int{ 0, seq_len - 1, 0 };
            const stop1 = [_]c_int{ 1, seq_len, hidden_size };
            try mlx.check(mlx.mlx_slice(&hid_last, hidden, &start1, 3, &stop1, 3, &strides3, 3, s));
        }
        try appendKvOnly(self, target, cache, ids_head, hid_head, rope_offset, mrope_ctx);
        return forwardWithMrope(self, target, cache, ids_last, hid_last, rope_offset + seq_len - 1, true, mrope_ctx);
    }
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);

    var front = try frontChain(self, target, id_arr, hidden);
    defer front.deinit();

    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    const relative_offset: usize = @intCast(rope_offset);
    const needs_explicit_mrope = if (mrope_ctx) |positions|
        positions.absolutePosition(relative_offset) < positions.total
    else
        false;
    if (headSkipsRope(cfg)) {
        try mlx.check(mlx.mlx_array_set(&q_rope, front.q_t));
        try mlx.check(mlx.mlx_array_set(&k_rope, front.k_t));
    } else if (needs_explicit_mrope) {
        const positions = mrope_ctx.?;
        const cs = try target.buildMropeCosSin(positions, relative_offset, @intCast(seq_len));
        defer _ = mlx.mlx_array_free(cs.cos);
        defer _ = mlx.mlx_array_free(cs.sin);
        q_rope = try target.applyMrope(front.q_t, cs.cos, cs.sin, rope_dims);
        k_rope = try target.applyMrope(front.k_t, cs.cos, cs.sin, rope_dims);
    } else {
        const effective_offset: c_int = if (mrope_ctx) |positions|
            @intCast(@as(i64, @intCast(positions.absolutePosition(relative_offset))) + positions.delta)
        else
            rope_offset;
        try mlx.check(mlx.mlx_fast_rope(&q_rope, front.q_t, rope_dims, false, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, effective_offset, .{ .ctx = null }, s));
        try mlx.check(mlx.mlx_fast_rope(&k_rope, front.k_t, rope_dims, false, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, effective_offset, .{ .ctx = null }, s));
    }

    var kv_view = try cache.update(0, k_rope, front.v_t, s, 0);
    defer kv_view.deinit();

    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    // Multi-token (history rebuild / draft batch): try the fused hd-256
    // flash kernel first — same dispatch the trunk's prefill uses.
    var fused_done = false;
    if (seq_len > 1) {
        if (try attn256.fusedSdpa256Prefill(s, q_rope, kv_view.k, kv_view.v, attn_scale, 0)) |fused| {
            _ = mlx.mlx_array_free(attn_out);
            attn_out = fused;
            fused_done = true;
        }
    }
    if (!fused_done) {
        const mask_mode: [*:0]const u8 = if (seq_len > 1) "causal" else "";
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, kv_view.k, kv_view.v, attn_scale, mask_mode, none_mask, .{ .ctx = null }, seq_len > 1 and sdpa_mod.sdpaForceFused(q_rope, kv_view.k), s));
    }

    const post = try backChain(self, target, attn_out, front.gate, front.x, seq_len);

    // Logits (and the chained hidden) are only ever consumed for the LAST
    // row — the draft chain's next token / confidence. Multi-row calls (the
    // merged history+draft forward) slice to the last row BEFORE the vocab
    // head projection: projecting the history rows through a 248k-vocab head
    // is pure waste, and a [1,L,H] hidden_next would break the L=1 chain.
    // The slice applies on the want_logits=false arm too: the rerank draft
    // path chains hidden_next without ever asking for logits, and
    // appendHistory (the other no-logits caller) frees hidden_next unused.
    var post_last = post;
    if (seq_len > 1) {
        var sliced = mlx.mlx_array_new();
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, hidden_size };
        const strides = [_]c_int{ 1, 1, 1 };
        mlx.check(mlx.mlx_slice(&sliced, post, &start, 3, &stop, 3, &strides, 3, s)) catch |err| {
            _ = mlx.mlx_array_free(sliced);
            _ = mlx.mlx_array_free(post);
            return err;
        };
        _ = mlx.mlx_array_free(post);
        post_last = sliced;
    }
    if (!want_logits) {
        return .{ .logits = .{ .ctx = null }, .hidden_next = post_last };
    }
    const logits = targetLmHead(self, target, post_last, s) catch |err| {
        _ = mlx.mlx_array_free(post_last);
        return err;
    };
    return .{ .logits = logits, .hidden_next = post_last };
}

pub const Lane = struct {
    cache: *KVCache,
    /// One int32 id, may be lazy.
    id: mlx.mlx_array,
    /// `[1, 1, H]`.
    hidden: mlx.mlx_array,
    rope_offset: c_int,
};

pub const MAX_LANES = 32;

/// One draft step for N requests as the N rows of ONE head forward: the projections and
/// the MLP run once, only the cache append and the attention read are per lane (each
/// lane owns its cache and its length). Returns the post-norm hidden `[1, N, H]`.
pub fn forwardLanes(self: *const MtpModel, target: anytype, lanes: []const Lane) !mlx.mlx_array {
    const s = self.s;
    const cfg = &target.config;
    const n = lanes.len;
    if (n == 0 or n > MAX_LANES) return error.MtpLaneCount;
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);

    var parts: [MAX_LANES]mlx.mlx_array = undefined;
    var offsets: [MAX_LANES]i32 = undefined;
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    {
        var flat: [MAX_LANES]mlx.mlx_array = undefined;
        var flat_n: usize = 0;
        defer for (flat[0..flat_n]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (lanes, 0..) |lane, k| {
            flat[k] = mlx.mlx_array_new();
            flat_n = k + 1;
            try mlx.check(mlx.mlx_reshape(&flat[k], lane.id, &[_]c_int{1}, 1, s));
            offsets[k] = lane.rope_offset;
        }
        const idv = mlx.mlx_vector_array_new_data(&flat, n);
        defer _ = mlx.mlx_vector_array_free(idv);
        try mlx.check(mlx.mlx_concatenate_axis(&ids, idv, 0, s));
        for (lanes, 0..) |lane, k| parts[k] = lane.hidden;
        const hv = mlx.mlx_vector_array_new_data(&parts, n);
        defer _ = mlx.mlx_vector_array_free(hv);
        try mlx.check(mlx.mlx_concatenate_axis(&hidden, hv, 1, s));
    }

    var front = try frontChain(self, target, ids, hidden);
    defer front.deinit();

    // Lanes become the batch axis for the rope: one offset per lane.
    const off_arr = mlx.mlx_array_new_data(&offsets, &[_]c_int{@intCast(n)}, 1, .int32);
    defer _ = mlx.mlx_array_free(off_arr);
    const lanes_first = [_]c_int{ 2, 1, 0, 3 };
    var roped: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (roped) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]mlx.mlx_array{ front.q_t, front.k_t }, &roped) |x, *out| {
        var by_lane = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(by_lane);
        try mlx.check(mlx.mlx_transpose_axes(&by_lane, x, &lanes_first, 4, s));
        if (headSkipsRope(cfg)) {
            try mlx.check(mlx.mlx_array_set(out, by_lane));
            continue;
        }
        try mlx.check(mlx.mlx_fast_rope_dynamic(out, by_lane, rope_dims, false, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, off_arr, .{ .ctx = null }, s));
    }

    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var outs: [MAX_LANES]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (outs[0..built]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (lanes, 0..) |lane, k| {
        const q = try axisView(s, roped[0], 0, k);
        defer _ = mlx.mlx_array_free(q);
        const key = try axisView(s, roped[1], 0, k);
        defer _ = mlx.mlx_array_free(key);
        const val = try axisView(s, front.v_t, 2, k);
        defer _ = mlx.mlx_array_free(val);
        var kv_view = try lane.cache.update(0, key, val, s, 0);
        defer kv_view.deinit();
        outs[built] = mlx.mlx_array_new();
        built += 1;
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&outs[k], q, kv_view.k, kv_view.v, attn_scale, "", none_mask, .{ .ctx = null }, false, s));
    }
    var attn_out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_out);
    {
        const vec = mlx.mlx_vector_array_new_data(&outs, n);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&attn_out, vec, 2, s));
    }
    return backChain(self, target, attn_out, front.gate, front.x, @intCast(n));
}

/// Append committed-history entries: pair `hidden[:, i, :]` with
/// `token_ids[i]` for each i. One batched MTP-layer forward, no logits.
pub fn appendHistory(
    self: *const MtpModel,
    target: anytype,
    cache: *KVCache,
    token_ids: []const u32,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
) !void {
    return appendHistoryWithMrope(self, target, cache, token_ids, hidden, rope_offset, null);
}

pub fn appendHistoryWithMrope(
    self: *const MtpModel,
    target: anytype,
    cache: *KVCache,
    token_ids: []const u32,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
    mrope_ctx: ?MropeContext,
) !void {
    if (token_ids.len == 0) return;
    const ids_i32 = try self.allocator.alloc(i32, token_ids.len);
    defer self.allocator.free(ids_i32);
    for (token_ids, 0..) |t, i| ids_i32[i] = @intCast(t);
    const id_shape = [_]c_int{@intCast(token_ids.len)};
    const id_arr = mlx.mlx_array_new_data(ids_i32.ptr, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(id_arr);

    // Pure history: EVERY row's output is dead — K/V-only for all of them.
    if (mtpKvOnlyEnabled()) {
        return appendKvOnly(self, target, cache, id_arr, hidden, rope_offset, mrope_ctx);
    }
    // KVCache.update advances `cache.step` (layer 0) by the batch length.
    var out = try forwardWithMrope(self, target, cache, id_arr, hidden, rope_offset, false, mrope_ctx);
    _ = mlx.mlx_array_free(out.hidden_next);
    out.hidden_next = .{ .ctx = null };
}

pub fn stepArr(
    self: *const MtpModel,
    target: anytype,
    cache: *KVCache,
    prev_token_arr: mlx.mlx_array,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
) !StepOut {
    return stepArrWithMrope(self, target, cache, prev_token_arr, hidden, rope_offset, null);
}

pub fn stepArrWithMrope(
    self: *const MtpModel,
    target: anytype,
    cache: *KVCache,
    prev_token_arr: mlx.mlx_array,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
    mrope_ctx: ?MropeContext,
) !StepOut {
    // KVCache.update advances `cache.step` (layer 0) by 1.
    return forwardWithMrope(self, target, cache, prev_token_arr, hidden, rope_offset, true, mrope_ctx);
}

// ── Draft-rerank shortlist (port of the mlx.fast challenge draft-rerank
// scheme, submission 942e5ab2 in mlx.fast-qwen-3.8-mtp-challenge) ──
//
// PROPOSAL SIDE ONLY: a coarse 2-bit requant of the trunk lm_head scores the
// full vocab, an exact top-32 shortlist is taken in TWO dispatches (MLX's
// GPU argpartition is a full multi-block argsort — 14 dependent dispatches
// for 32 read values), and the trunk head's own 32 rows re-score the
// shortlist. The draft is the trunk's argmax over those rows, so whenever
// the coarse top-32 contains the trunk argmax the draft IS the trunk's
// choice — acceptance rides the trunk's order at ~2-bit readout cost.
// Verification is untouched: a coarse miss costs acceptance, never output.
//
// Ordinal: a monotone map from float into uint32 inducing (value asc, NaN
// above every number, -0.0 == +0.0). Real values never map to 0, so the
// zero-initialized empty slots below can never be selected while a
// simdgroup still has real candidates (guaranteed by rows >= TILES*TG).
const TOP32_ORDINAL_HEADER = kernelSource("kernels/spec/mtp_top32_ordinal_header.metal");

// Stage 1: TILES threadgroups partition [0, RC); each emits its local top 32
// as (ordinal, index) pairs — TILES * 32 candidates. Selection is 32 rounds
// of simd_max over per-thread slots with a `taken` bitmask (hence the
// PER_THREAD <= 32 static_assert), then one simdgroup reduces the
// threadgroup's per-simd lists the same way.
const TOP32_PARTIAL_SRC = kernelSource("kernels/spec/mtp_top32_partial.metal");

// Stage 2: one threadgroup reduces the TILES*32 candidates to the final 32
// ids (written ascending by (value, index) — consumers are order-blind).
const TOP32_FINAL_SRC = kernelSource("kernels/spec/mtp_top32_final.metal");

const TOP32_K: c_int = 32;
const TOP32_TG: c_int = 256;
const TOP32_TILES: c_int = 64;
const TOP32_CANDS: c_int = TOP32_TILES * TOP32_K;
/// Minimum row count: below TILES*TG some simdgroups hold fewer than 32 real
/// candidates and the zero-initialized empty slots become reachable.
pub const TOP32_MIN_ROWS: c_int = TOP32_TILES * TOP32_TG;

var top32_partial_cached: ?mlx.mlx_fast_metal_kernel = null;
var top32_final_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getTop32Partial() !mlx.mlx_fast_metal_kernel {
    if (top32_partial_cached) |k| return k;
    const input_names = [_][*:0]const u8{"logits"};
    const output_names = [_][*:0]const u8{ "cand_ord", "cand_idx" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_mtp_top32_partial",
        in_vec,
        out_vec,
        TOP32_PARTIAL_SRC,
        TOP32_ORDINAL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    top32_partial_cached = kernel;
    return kernel;
}

fn getTop32Final() !mlx.mlx_fast_metal_kernel {
    if (top32_final_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "cand_ord", "cand_idx" };
    const output_names = [_][*:0]const u8{"token_ids"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_mtp_top32_finalize",
        in_vec,
        out_vec,
        TOP32_FINAL_SRC,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    top32_final_cached = kernel;
    return kernel;
}

/// `draftTop32` through MLX's argpartition (CUDA, where the Metal kernels do not run): the same
/// set of ids, in no promised order or tie break. Proposal side only, so neither changes output.
fn draftTop32Partition(s: mlx.mlx_stream, row: mlx.mlx_array) !mlx.mlx_array {
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, row, s));
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_argpartition_axis(&part, neg, TOP32_K - 1, 0, s));
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, part, &[_]c_int{0}, 1, &[_]c_int{TOP32_K}, 1, &[_]c_int{1}, 1, s));
    var ids = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, head, .uint32, s));
    return ids;
}

/// Exact top-32 ids of `row` ([rows], any float dtype) as a lazy [32] uint32
/// array, ascending by (value, index) — ties break toward the higher index,
/// NaN above every number. `rows` rides as a template int (stable per model,
/// so MLX caches one specialization per vocab width).
pub fn draftTop32(s: mlx.mlx_stream, row: mlx.mlx_array, rows: c_int) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) {
        if (mlx.cudaAvailable()) return draftTop32Partition(s, row);
        return error.MetalKernelNeedsGpuStream;
    }
    if (rows < TOP32_MIN_ROWS) return error.UnsupportedTop32Shape;
    const per_thread = @divTrunc(rows + TOP32_TILES * TOP32_TG - 1, TOP32_TILES * TOP32_TG);
    if (per_thread > 32) return error.UnsupportedTop32Shape;

    const pk = try getTop32Partial();
    var cand_ord = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ord);
    var cand_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_idx);
    {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const cand_shape = [_]c_int{TOP32_CANDS};
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &cand_shape, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TILES * TOP32_TG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RC", rows));
        const inputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(inputs);
        _ = mlx.mlx_vector_array_append_value(inputs, row);
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, pk, inputs, cfg, s));
        try mlx.check(mlx.mlx_vector_array_get(&cand_ord, outputs, 0));
        try mlx.check(mlx.mlx_vector_array_get(&cand_idx, outputs, 1));
    }

    const fk = try getTop32Final();
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{TOP32_K};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, TOP32_TG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, TOP32_TG, 1, 1));
    const inputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(inputs);
    _ = mlx.mlx_vector_array_append_value(inputs, cand_ord);
    _ = mlx.mlx_vector_array_append_value(inputs, cand_idx);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, fk, inputs, cfg, s));
    var token_ids = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&token_ids, outputs, 0));
    return token_ids;
}

// ── Tests ──

const testing = std.testing;

test "draftTop32 on CUDA: the 32 largest ids of a vocab row, as a set" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const rows: usize = 248320;
    const data = try testing.allocator.alloc(f32, rows);
    defer testing.allocator.free(data);
    var prng = std.Random.DefaultPrng.init(32);
    for (data) |*x| x.* = prng.random().float(f32) * 10.0;
    var want: [TOP32_K]u32 = undefined;
    for (&want, 0..) |*w, i| {
        w.* = @intCast((i * 7919 + 13) % rows);
        data[w.*] = 20.0 + @as(f32, @floatFromInt(i));
    }
    const host = mlx.mlx_array_new_data(data.ptr, &[_]c_int{@intCast(rows)}, 1, .float32);
    defer _ = mlx.mlx_array_free(host);
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_astype(&row, host, .bfloat16, s));
    const ids = try draftTop32(s, row, @intCast(rows));
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_array_eval(ids));
    try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(ids));
    var got: [TOP32_K]u32 = undefined;
    @memcpy(&got, mlx.mlx_array_data_uint32(ids).?[0..TOP32_K]);
    std.mem.sort(u32, &got, {}, std.sort.asc(u32));
    std.mem.sort(u32, &want, {}, std.sort.asc(u32));
    try testing.expectEqualSlices(u32, &want, &got);
}

test "mtp: loadMtp detects the Hy3 layout (eh_proj + full decoder layer + sigmoid MoE)" {
    // Hy3 (hy_v3) checkpoints ship the MTP block in `model-mtp.safetensors`
    // under post-sanitize names: mtp.{enorm,hnorm,eh_proj,final_layernorm} +
    // mtp.layer.* (a FULL hy3 decoder layer: attention + 192-expert sigmoid
    // MoE + UNGATED shared expert + expert_bias). Toy bf16 geometry — this
    // pins the LAYOUT detection and struct shape; the head's math is pinned
    // live by tests/test_mtp_equivalence.sh (acceptance floor + temp-0
    // equivalence).
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model-mtp.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);

    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);

        const H = struct {
            fn put(m: mlx.mlx_map_string_to_array, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
                var total: usize = 1;
                for (shape) |d| total *= @intCast(d);
                const data = try std.testing.allocator.alloc(f32, total);
                defer std.testing.allocator.free(data);
                for (data, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.1;
                const f32_arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var bf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(bf);
                try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(bf));
                _ = mlx.mlx_map_string_to_array_insert(m, key, bf);
            }
            fn putF32(m: mlx.mlx_map_string_to_array, key: [*:0]const u8, shape: []const c_int) void {
                var total: usize = 1;
                for (shape) |d| total *= @intCast(d);
                var data: [16]f32 = @splat(0.0);
                const f32_arr = mlx.mlx_array_new_data(&data, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                _ = mlx.mlx_map_string_to_array_insert(m, key, f32_arr);
            }
        };
        // hidden 8, heads 2 × hd 4, kv 1, experts 4, expert inter 6.
        try H.put(map, "mtp.enorm.weight", &.{8}, s);
        try H.put(map, "mtp.hnorm.weight", &.{8}, s);
        try H.put(map, "mtp.final_layernorm.weight", &.{8}, s);
        try H.put(map, "mtp.eh_proj.weight", &.{ 8, 16 }, s);
        try H.put(map, "mtp.layer.input_layernorm.weight", &.{8}, s);
        try H.put(map, "mtp.layer.post_attention_layernorm.weight", &.{8}, s);
        try H.put(map, "mtp.layer.self_attn.q_norm.weight", &.{4}, s);
        try H.put(map, "mtp.layer.self_attn.k_norm.weight", &.{4}, s);
        try H.put(map, "mtp.layer.self_attn.q_proj.weight", &.{ 8, 8 }, s);
        try H.put(map, "mtp.layer.self_attn.k_proj.weight", &.{ 4, 8 }, s);
        try H.put(map, "mtp.layer.self_attn.v_proj.weight", &.{ 4, 8 }, s);
        try H.put(map, "mtp.layer.self_attn.o_proj.weight", &.{ 8, 8 }, s);
        try H.put(map, "mtp.layer.mlp.router.gate.weight", &.{ 4, 8 }, s);
        try H.put(map, "mtp.layer.mlp.experts.gate_proj.weight", &.{ 4, 6, 8 }, s);
        try H.put(map, "mtp.layer.mlp.experts.up_proj.weight", &.{ 4, 6, 8 }, s);
        try H.put(map, "mtp.layer.mlp.experts.down_proj.weight", &.{ 4, 8, 6 }, s);
        try H.put(map, "mtp.layer.mlp.shared_mlp.gate_proj.weight", &.{ 6, 8 }, s);
        try H.put(map, "mtp.layer.mlp.shared_mlp.up_proj.weight", &.{ 6, 8 }, s);
        try H.put(map, "mtp.layer.mlp.shared_mlp.down_proj.weight", &.{ 8, 6 }, s);
        H.putF32(map, "mtp.layer.mlp.expert_bias", &.{4});
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    var m = try loadMtp(io, allocator, s, dir_path);
    defer m.deinit();

    // Hy3 shape: quantizable eh_proj bound, no bf16 fc, MoE mlp with the
    // sigmoid-router extras, UNGATED shared expert.
    try testing.expect(m.eh_proj != null);
    try testing.expect(m.fc.w.ctx == null);
    try testing.expect(m.mlp == .moe);
    try testing.expect(m.mlp.moe.expert_bias != null);
    try testing.expect(m.mlp.moe.shared_ungated);
    try testing.expect(m.mlp.moe.shared_expert_gate_w == null);
    // enorm/hnorm ride the pre_fc_norm slots (same role, no +1 folding).
    const en_shape = mlx.getShape(m.pre_fc_norm_emb);
    try testing.expectEqual(@as(c_int, 8), en_shape[0]);
}

/// Builds a head's weights in memory (evaluated, so a leaked handle pins a
/// live buffer) with tensor `skip` left out; `skip == cases.len` keeps all.
const HeadLoadLeakProbe = struct {
    const Case = struct { key: []const u8, shape: []const c_int, dt: mlx.mlx_dtype = .bfloat16 };

    fn build(cases: []const Case, skip: usize, s: mlx.mlx_stream) !Weights {
        const a = testing.allocator;
        var w = Weights.init(a);
        errdefer w.deinit();
        for (cases, 0..) |c, i| {
            if (i == skip) continue;
            var total: usize = 1;
            for (c.shape) |d| total *= @intCast(d);
            const data = try a.alloc(f32, total);
            defer a.free(data);
            for (data, 0..) |*x, j| x.* = @as(f32, @floatFromInt(j % 7)) * 0.1;
            const f32_arr = mlx.mlx_array_new_data(data.ptr, c.shape.ptr, @intCast(c.shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var arr = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(arr);
            try mlx.check(mlx.mlx_astype(&arr, f32_arr, c.dt, s));
            try mlx.check(mlx.mlx_array_eval(arr));
            const key = try a.dupe(u8, c.key);
            errdefer a.free(key);
            try w.map.put(key, arr);
        }
        return w;
    }

    /// Every drop either fails with MissingMtpWeight or (only for `optional`)
    /// loads; either way active memory returns to where it started.
    fn expectNoLeak(
        cases: []const Case,
        comptime load: fn (std.mem.Allocator, mlx.mlx_stream, *const Weights) anyerror!MtpModel,
        optional: ?[]const u8,
    ) !void {
        const s = mlx.gpuStream();
        for (0..cases.len + 1) |skip| {
            try mlx.check(mlx.mlx_synchronize(s));
            var before: usize = 0;
            try mlx.check(mlx.mlx_get_active_memory(&before));
            {
                var w = try build(cases, skip, s);
                defer w.deinit();
                const may_load = skip == cases.len or
                    (optional != null and std.mem.eql(u8, cases[skip].key, optional.?));
                if (load(testing.allocator, s, &w)) |loaded| {
                    var m = loaded;
                    m.deinit();
                    try testing.expect(may_load);
                } else |err| {
                    try testing.expect(!may_load);
                    try testing.expectEqual(error.MissingMtpWeight, err);
                }
            }
            try mlx.check(mlx.mlx_synchronize(s));
            var after: usize = 0;
            try mlx.check(mlx.mlx_get_active_memory(&after));
            if (after != before) std.debug.print("active memory {d} -> {d} with '{s}' missing\n", .{ before, after, if (skip < cases.len) cases[skip].key else "(none)" });
            try testing.expectEqual(before, after);
        }
    }
};

test "mtp: a Hy3 head missing any one tensor fails to load and frees what it loaded" {
    const C = HeadLoadLeakProbe.Case;
    const cases = [_]C{
        .{ .key = "mtp.enorm.weight", .shape = &.{8} },
        .{ .key = "mtp.hnorm.weight", .shape = &.{8} },
        .{ .key = "mtp.final_layernorm.weight", .shape = &.{8} },
        .{ .key = "mtp.eh_proj.weight", .shape = &.{ 8, 16 } },
        .{ .key = "mtp.layer.input_layernorm.weight", .shape = &.{8} },
        .{ .key = "mtp.layer.post_attention_layernorm.weight", .shape = &.{8} },
        .{ .key = "mtp.layer.self_attn.q_norm.weight", .shape = &.{4} },
        .{ .key = "mtp.layer.self_attn.k_norm.weight", .shape = &.{4} },
        .{ .key = "mtp.layer.self_attn.q_proj.weight", .shape = &.{ 8, 8 } },
        .{ .key = "mtp.layer.self_attn.k_proj.weight", .shape = &.{ 4, 8 } },
        .{ .key = "mtp.layer.self_attn.v_proj.weight", .shape = &.{ 4, 8 } },
        .{ .key = "mtp.layer.self_attn.o_proj.weight", .shape = &.{ 8, 8 } },
        .{ .key = "mtp.layer.mlp.router.gate.weight", .shape = &.{ 4, 8 } },
        .{ .key = "mtp.layer.mlp.experts.gate_proj.weight", .shape = &.{ 4, 6, 8 } },
        .{ .key = "mtp.layer.mlp.experts.up_proj.weight", .shape = &.{ 4, 6, 8 } },
        .{ .key = "mtp.layer.mlp.experts.down_proj.weight", .shape = &.{ 4, 8, 6 } },
        .{ .key = "mtp.layer.mlp.shared_mlp.gate_proj.weight", .shape = &.{ 6, 8 } },
        .{ .key = "mtp.layer.mlp.shared_mlp.up_proj.weight", .shape = &.{ 6, 8 } },
        .{ .key = "mtp.layer.mlp.shared_mlp.down_proj.weight", .shape = &.{ 8, 6 } },
        .{ .key = "mtp.layer.mlp.expert_bias", .shape = &.{4}, .dt = .float32 },
    };
    const L = struct {
        fn load(a: std.mem.Allocator, s: mlx.mlx_stream, w: *const Weights) anyerror!MtpModel {
            return loadHy3Mtp(a, s, w, "");
        }
    };
    try HeadLoadLeakProbe.expectNoLeak(&cases, L.load, null);
}

test "mtp: a Qwen head missing any one tensor fails to load and frees what it loaded" {
    const C = HeadLoadLeakProbe.Case;
    // Hidden 64 so the dense trunk linears take the head-requant path.
    const common = [_]C{
        .{ .key = "mtp.fc.weight", .shape = &.{ 64, 128 } },
        .{ .key = "mtp.pre_fc_norm_embedding.weight", .shape = &.{64} },
        .{ .key = "mtp.pre_fc_norm_hidden.weight", .shape = &.{64} },
        .{ .key = "mtp.norm.weight", .shape = &.{64} },
        .{ .key = "mtp.layers.0.input_layernorm.weight", .shape = &.{64} },
        .{ .key = "mtp.layers.0.post_attention_layernorm.weight", .shape = &.{64} },
        .{ .key = "mtp.layers.0.self_attn.q_norm.weight", .shape = &.{16} },
        .{ .key = "mtp.layers.0.self_attn.k_norm.weight", .shape = &.{16} },
        .{ .key = "mtp.layers.0.self_attn.q_proj.weight", .shape = &.{ 128, 64 } },
        .{ .key = "mtp.layers.0.self_attn.k_proj.weight", .shape = &.{ 32, 64 } },
        .{ .key = "mtp.layers.0.self_attn.v_proj.weight", .shape = &.{ 32, 64 } },
        .{ .key = "mtp.layers.0.self_attn.o_proj.weight", .shape = &.{ 64, 64 } },
    };
    const dense = common ++ [_]C{
        .{ .key = "mtp.layers.0.mlp.gate_proj.weight", .shape = &.{ 32, 64 } },
        .{ .key = "mtp.layers.0.mlp.up_proj.weight", .shape = &.{ 32, 64 } },
        .{ .key = "mtp.layers.0.mlp.down_proj.weight", .shape = &.{ 64, 32 } },
    };
    const moe = common ++ [_]C{
        .{ .key = "mtp.layers.0.mlp.gate.weight", .shape = &.{ 4, 64 } },
        .{ .key = "mtp.layers.0.mlp.switch_mlp.gate_proj.weight", .shape = &.{ 4, 32, 64 } },
        .{ .key = "mtp.layers.0.mlp.switch_mlp.up_proj.weight", .shape = &.{ 4, 32, 64 } },
        .{ .key = "mtp.layers.0.mlp.switch_mlp.down_proj.weight", .shape = &.{ 4, 64, 32 } },
        .{ .key = "mtp.layers.0.mlp.shared_expert.gate_proj.weight", .shape = &.{ 32, 64 } },
        .{ .key = "mtp.layers.0.mlp.shared_expert.up_proj.weight", .shape = &.{ 32, 64 } },
        .{ .key = "mtp.layers.0.mlp.shared_expert.down_proj.weight", .shape = &.{ 64, 32 } },
        .{ .key = "mtp.layers.0.mlp.shared_expert_gate.weight", .shape = &.{ 1, 64 } },
    };
    const L = struct {
        fn load(a: std.mem.Allocator, s: mlx.mlx_stream, w: *const Weights) anyerror!MtpModel {
            return loadQwenMtp(a, s, w, "");
        }
    };
    try HeadLoadLeakProbe.expectNoLeak(&dense, L.load, null);
    try HeadLoadLeakProbe.expectNoLeak(&moe, L.load, null);
}

test "mtp: requantizeRows round-trips through a finer re-encode (chunked)" {
    const s = mlx.gpuStream();
    const rows: usize = 64;
    const cols: usize = 256;

    var prng = std.Random.DefaultPrng.init(42);
    const buf = try testing.allocator.alloc(f32, rows * cols);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);
    const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
    const dense_f32 = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(dense_f32);
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_astype(&dense, dense_f32, .bfloat16, s));

    // "Trunk" 4-bit/gs64 triple.
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var q4 = QLinear{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    defer q4.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&q4.w, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q4.s, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q4.b, triple, 2));

    // Requantize to 8-bit/gs64, chunked at 16 rows (4 chunks → exercises concat).
    var q8 = try requantizeRows(s, q4.w, q4.s, q4.b, 64, 4, "affine", 64, 8, 16);
    defer q8.deinit();

    // Dequantize both and compare — an 8-bit re-encode of 4-bit-quantized
    // values is near-lossless, so cosine must be ~1.
    var deq = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (deq) |d| {
        _ = mlx.mlx_array_free(d);
    };
    try mlx.check(mlx.mlx_dequantize(&deq[0], q4.w, q4.s, q4.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_dequantize(&deq[1], q8.w, q8.s, q8.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    for (deq) |d| try mlx.check(mlx.mlx_array_eval(d));

    const a = mlx.mlx_array_data_float32(deq[0]).?;
    const b = mlx.mlx_array_data_float32(deq[1]).?;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..rows * cols) |i| {
        dot += @as(f64, a[i]) * b[i];
        na += @as(f64, a[i]) * a[i];
        nb += @as(f64, b[i]) * b[i];
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    try testing.expect(cos > 0.999);

    // Shape sanity: 8-bit packs 4 in-features per u32 → cols/4 packed cols.
    const w8_shape = mlx.getShape(q8.w);
    try testing.expectEqual(@as(c_int, @intCast(rows)), w8_shape[0]);
    try testing.expectEqual(@as(c_int, @intCast(cols / 4)), w8_shape[1]);
}

test "mtp: requantizeRows accepts a non-affine (mxfp8) source — the issue-#81 draft-head path" {
    // With headQuantParams fixed, buildDraftHead on an mxfp8 trunk passes
    // mode="mxfp8" and the load path's null-ctx biases into requantizeRows.
    // The source dequantize must ride that without demanding affine biases
    // (else the crash just moves from the first forward to bind time), and
    // the 3-bit affine draft re-encode must stay correlated.
    const s = mlx.gpuStream();
    const rows: usize = 64;
    const cols: usize = 256;

    var prng = std.Random.DefaultPrng.init(7);
    const buf = try testing.allocator.alloc(f32, rows * cols);
    defer testing.allocator.free(buf);
    for (buf) |*x| x.* = prng.random().floatNorm(f32);
    const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
    const dense_f32 = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(dense_f32);
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_astype(&dense, dense_f32, .bfloat16, s));

    // mxfp8 "trunk head": a (w, scales) pair — no biases tensor, by design.
    var pair = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(pair);
    try mlx.check(mlx.mlx_quantize(&pair, dense, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, s));
    var qmx = QLinear{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
    defer qmx.deinit();
    try mlx.check(mlx.mlx_vector_array_get(&qmx.w, pair, 0));
    try mlx.check(mlx.mlx_vector_array_get(&qmx.s, pair, 1));
    // qmx.b stays null-ctx — exactly what the load path hands buildDraftHead.

    // The live buildDraftHead call shape: 3-bit/gs64 draft re-encode, chunked.
    var q3 = try requantizeRows(s, qmx.w, qmx.s, qmx.b, 32, 8, "mxfp8", 64, 3, 16);
    defer q3.deinit();

    var deq = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (deq) |d| {
        _ = mlx.mlx_array_free(d);
    };
    try mlx.check(mlx.mlx_dequantize(&deq[0], qmx.w, qmx.s, qmx.b, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", .{}, .{ .value = .float32, .has_value = true }, s));
    try mlx.check(mlx.mlx_dequantize(&deq[1], q3.w, q3.s, q3.b, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(3), "affine", .{}, .{ .value = .float32, .has_value = true }, s));
    for (deq) |d| try mlx.check(mlx.mlx_array_eval(d));

    const a = mlx.mlx_array_data_float32(deq[0]).?;
    const b = mlx.mlx_array_data_float32(deq[1]).?;
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (0..rows * cols) |i| {
        dot += @as(f64, a[i]) * b[i];
        na += @as(f64, a[i]) * a[i];
        nb += @as(f64, b[i]) * b[i];
    }
    const cos = dot / (@sqrt(na) * @sqrt(nb));
    try testing.expect(cos > 0.95);
}

/// Minimal safetensors bytes whose HEADER names `key` — enough for the
/// marker peek (the resolver never reads tensor data).
fn writeFakeSidecar(io: std.Io, dir: std.Io.Dir, sub_path: []const u8, key: []const u8) !void {
    var buf: [512]u8 = undefined;
    const header = try std.fmt.bufPrint(&buf, "{{\"{s}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}}}", .{key});
    var file_buf: [512 + 12]u8 = undefined;
    std.mem.writeInt(u64, file_buf[0..8], header.len, .little);
    @memcpy(file_buf[8..][0..header.len], header);
    @memcpy(file_buf[8 + header.len ..][0..4], &[_]u8{ 0, 0, 0, 0 });
    try dir.writeFile(io, .{ .sub_path = sub_path, .data = file_buf[0 .. 8 + header.len + 4] });
}

test "mtp: sidecar resolution accepts native and Forge layouts in priority order" {
    const io = testing.io;
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // Nothing present → null.
    try testing.expectEqual(@as(?[]const u8, null), resolveMtpSidecarInDir(io, allocator, tmp.dir));

    // oMLX OptiQ layout is discovered when it's the only head present.
    try tmp.dir.createDirPath(io, "optiq");
    try writeFakeSidecar(io, tmp.dir, "optiq/mtp.safetensors", "mtp.fc.weight");
    try testing.expectEqualStrings("optiq/mtp.safetensors", resolveMtpSidecarInDir(io, allocator, tmp.dir).?);

    try writeFakeSidecar(io, tmp.dir, "model-mtp.safetensors", "language_model.mtp.fc.weight");
    try testing.expectEqualStrings("model-mtp.safetensors", resolveMtpSidecarInDir(io, allocator, tmp.dir).?);

    // Forge current name outranks legacy (hy3's eh_proj marker also claims).
    try writeFakeSidecar(io, tmp.dir, "mtp.safetensors", "mtp.eh_proj.weight");
    try testing.expectEqualStrings("mtp.safetensors", resolveMtpSidecarInDir(io, allocator, tmp.dir).?);

    // Native mlx-serve layout outranks both Forge names.
    try tmp.dir.createDirPath(io, "mtp");
    try writeFakeSidecar(io, tmp.dir, "mtp/weights.safetensors", "mtp.fc.weight");
    try testing.expectEqualStrings("mtp/weights.safetensors", resolveMtpSidecarInDir(io, allocator, tmp.dir).?);
}

test "mtp: a sidecar-NAMED file without a marker key is not a sidecar (dsv4 module class)" {
    // DeepSeek-V4 mirrors ship their OWN (dsv4-shaped) MTP module at
    // `model-mtp.safetensors` — one of the house sidecar names. Claiming it
    // by NAME alone sent the qwen-shaped loader into MissingMtpWeight on
    // every `--mtp` boot. The sidecar claim rides the same
    // marker gate as discovery + the in-checkpoint shard sweep: the header
    // must PROVE a loadable head.
    const io = testing.io;
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try writeFakeSidecar(io, tmp.dir, "model-mtp.safetensors", "mtp.0.attn.wq_a.weight");
    try testing.expectEqual(@as(?[]const u8, null), resolveMtpSidecarInDir(io, allocator, tmp.dir));
    // …and hasMtpHead (the scheduler's attempt gate) says no as well, so a
    // dsv4 boot with --mtp never logs a scary MissingMtpWeight warning.
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    try testing.expect(!hasMtpHead(io, allocator, path_buf[0..root_len]));
}

test "mtp: empty sidecar file is not a sidecar" {
    const io = testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp.safetensors", .data = "" });
    try testing.expectEqual(@as(?[]const u8, null), resolveMtpSidecarInDir(io, testing.allocator, tmp.dir));
}

test "mtp: delta-encoded norms are detected and folded +1; folded norms untouched" {
    const s = mlx.gpuStream();
    const shape = [_]c_int{6};

    // A delta-encoded RMSNorm weight (the layer computes `1 + w`) clusters at 0
    // with a large negative fraction; the pre-folded form is delta + 1 and is
    // strictly positive. This is exactly the mlx-serve-vs-OptiQ +1.0 offset.
    var delta_buf = [_]f32{ -0.5, -0.2, 0.0, 0.3, 0.8, -0.1 };
    const delta = mlx.mlx_array_new_data(&delta_buf, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(delta);
    var folded_buf = [_]f32{ 0.5, 0.8, 1.0, 1.3, 1.8, 0.9 };
    const folded = mlx.mlx_array_new_data(&folded_buf, &shape, 1, .float32);
    defer _ = mlx.mlx_array_free(folded);

    // Detection: the delta head is ~50% negative, the folded head 0% — the
    // 0.05 threshold sits far from both.
    try testing.expect((try negFraction(delta, s)) > 0.05);
    try testing.expect((try negFraction(folded, s)) < 0.01);

    // Folding the delta head recovers the folded weights byte-for-byte, and no
    // longer trips detection (so a second load can't double-fold).
    const rec = try foldNormPlusOne(delta, s);
    defer _ = mlx.mlx_array_free(rec);
    try testing.expect((try maxAbsDiff(rec, folded, s)) < 1e-6);
    try testing.expect((try negFraction(rec, s)) < 0.01);
    try testing.expect(!mtpNormsAreDeltaEnc1D(rec, s));

    // A natively-folded head must be left untouched (dtype + values preserved).
    const keep = try foldNormPlusOne_ifDelta(folded, s);
    defer _ = mlx.mlx_array_free(keep);
    try testing.expect((try maxAbsDiff(keep, folded, s)) < 1e-6);
}

// Test-only: run the negFraction threshold on a single 1-D norm array.
fn mtpNormsAreDeltaEnc1D(arr: mlx.mlx_array, s: mlx.mlx_stream) bool {
    const nf = negFraction(arr, s) catch return false;
    return nf > 0.05;
}

// Test-only: fold only when the single array reads as delta-encoded.
fn foldNormPlusOne_ifDelta(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (mtpNormsAreDeltaEnc1D(arr, s)) return foldNormPlusOne(arr, s);
    var owned = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&owned, arr));
    return owned;
}

// Test-only: max |a-b| over all elements.
fn maxAbsDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, a, b, s));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, s));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ad, false, s));
    try mlx.check(mlx.mlx_array_eval(mx));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, mx));
    return out;
}

test "mtp: reference-based head-norm repair rule (oMLX norm_repair)" {
    // Gap beyond the margin (an oQ-broken head, ~1 below its backbone anchor)
    // → repair; a head at/above its anchor → no-op; idempotent after the +1.
    // Delta-encoded (some negatives) + gap beyond the margin → repair.
    try testing.expect(mtpNormNeedsRepair(0.75, 1.45, 0.008)); // gap 0.70 → repair
    try testing.expect(!mtpNormNeedsRepair(1.30, 1.45, 0.008)); // gap 0.15 → no-op
    try testing.expect(!mtpNormNeedsRepair(1.50, 1.45, 0.008)); // above anchor → no-op
    try testing.expect(!mtpNormNeedsRepair(0.75 + 1.0, 1.45, 0.0)); // post-shift → no-op

    // A STRICTLY POSITIVE norm is folded by construction, whatever the anchor
    // says: avlp12's Alis head ships post_attention_layernorm at 1.206 — the
    // exact value the delta fold produces — against a 1.93 mean-of-means
    // anchor whose own layers span 0.02..2.24. The gap rule alone convicted
    // it, double-shifted it to 2.21 and dropped per-draft acceptance from
    // ~86% to 33%.
    try testing.expect(!mtpNormNeedsRepair(1.206, 1.930, 0.0));
}

test "mtp: inferGroupSize geometry" {
    // 4-bit packed: weight [out, in*4/32] u32, scales [out, in/group].
    // Synthetic pair: packed_cols=4 → expanded in=32; scale_cols=2 → group 16.
    var q = QLinear{
        .w = mlx.mlx_array_new_data(&@as([8]i32, @splat(0)), &[_]c_int{ 2, 4 }, 2, .int32),
        .s = mlx.mlx_array_new_data(&@as([4]f32, @splat(0)), &[_]c_int{ 2, 2 }, 2, .float32),
        .b = mlx.mlx_array_new(),
    };
    defer q.deinit();
    try testing.expectEqual(@as(?u32, 16), inferGroupSize(&q, 4));
    try testing.expectEqual(@as(?u32, null), inferGroupSize(&q, 0));
    // Bits inference: packed_cols=4 with hidden=32 -> 4-bit; hidden=16 -> 8-bit.
    try testing.expectEqual(@as(?u32, 4), inferBits(&q, 32));
    try testing.expectEqual(@as(?u32, 8), inferBits(&q, 16));
    try testing.expectEqual(@as(?u32, null), inferBits(&q, 0));
    try testing.expectEqual(@as(?u32, null), inferBits(&q, 100));
    // Every affine width accepted by the shared geometry solver must also be
    // a valid sidecar fallback; q3/q5/q6 used to silently fall back to q4.
    var q_mixed = QLinear{
        .w = mlx.mlx_array_new_data(&@as([12]i32, @splat(0)), &[_]c_int{ 2, 6 }, 2, .int32),
        .s = mlx.mlx_array_new_data(&@as([4]f32, @splat(0)), &[_]c_int{ 2, 2 }, 2, .float32),
        .b = mlx.mlx_array_new(),
    };
    defer q_mixed.deinit();
    try testing.expectEqual(@as(?u32, 3), inferBits(&q_mixed, 64));
    try testing.expectEqual(@as(?u32, 6), inferBits(&q_mixed, 32));
    try testing.expectEqual(@as(?u32, 32), inferGroupSize(&q_mixed, 3));
    try testing.expectEqual(@as(?u32, 16), inferGroupSize(&q_mixed, 6));
    const q5_w = mlx.mlx_array_new_data(&@as([10]i32, @splat(0)), &[_]c_int{ 2, 5 }, 2, .int32);
    _ = mlx.mlx_array_free(q_mixed.w);
    q_mixed.w = q5_w;
    try testing.expectEqual(@as(?u32, 5), inferBits(&q_mixed, 32));
    try testing.expectEqual(@as(?u32, 16), inferGroupSize(&q_mixed, 5));
    // The real sidecar geometry: in=5120 packed to 640 u32 cols at 4 bits,
    // scales 160 cols → group 32.
    try testing.expectEqual(@as(u32, 32), (5120 / 160));
}

test "mtp: M5 NAX cost profiles require exact sidecar and draft-head quant geometry" {
    const s = mlx.gpuStream();
    const IN: u32 = 128;
    const OUT: u32 = 64;
    const mk = struct {
        fn qlinear(in_dim: u32, out_dim: u32, bits: u32, group: u32, stream: mlx.mlx_stream) !QLinear {
            const w_shape = [_]c_int{ @intCast(out_dim), @intCast(in_dim * bits / 32) };
            const sb_shape = [_]c_int{ @intCast(out_dim), @intCast(in_dim / group) };
            var q: QLinear = .{
                .w = mlx.mlx_array_new(),
                .s = mlx.mlx_array_new(),
                .b = mlx.mlx_array_new(),
            };
            errdefer q.deinit();
            try mlx.check(mlx.mlx_zeros(&q.w, &w_shape, 2, .uint32, stream));
            try mlx.check(mlx.mlx_zeros(&q.s, &sb_shape, 2, .bfloat16, stream));
            try mlx.check(mlx.mlx_zeros(&q.b, &sb_shape, 2, .bfloat16, stream));
            return q;
        }
    };

    try testing.expectEqual(MtpCostProfile.g17_nax_q8_gs32, m5NaxCostProfileForFingerprint(8, 32, .uniform_quantized_embedding));
    try testing.expectEqual(MtpCostProfile.g17_nax_q4_gs32, m5NaxCostProfileForFingerprint(4, 32, .uniform_quantized_embedding));
    try testing.expectEqual(MtpCostProfile.generic, m5NaxCostProfileForFingerprint(8, 64, .uniform_quantized_embedding));
    // Qwen3.8's resident bf16 embedding is part of its measured q4/gs64
    // round surface. A uniformly-quantized embedding is a different, still
    // unmeasured surface and must remain generic.
    try testing.expectEqual(MtpCostProfile.generic, m5NaxCostProfileForFingerprint(4, 64, .uniform_quantized_embedding));
    try testing.expectEqual(MtpCostProfile.g17_nax_q4_gs64, m5NaxCostProfileForFingerprint(4, 64, .uniform_bf16_embedding));
    try testing.expectEqual(MtpCostProfile.g17_nax_q6_gs64, m5NaxCostProfileForFingerprint(6, 64, .uniform_q6_quantized_embedding));
    try testing.expectEqual(MtpCostProfile.g17_nax_q8_gs64, m5NaxCostProfileForFingerprint(8, 64, .uniform_q8_bf16_embedding));
    try testing.expectEqual(MtpCostProfile.generic, m5NaxCostProfileForFingerprint(8, 64, .uniform_q6_quantized_embedding));
    try testing.expectEqual(MtpCostProfile.generic, m5NaxCostProfileForFingerprint(6, 64, .uniform_q8_bf16_embedding));
    // It must not be mistaken for the older mixed-q4/q5/q6 oQ4e trunk merely
    // because both native sidecars are q4/gs64.
    try testing.expectEqual(MtpCostProfile.g17_nax_oq4e_q4_gs64, m5NaxCostProfileForFingerprint(4, 64, .oqe_quantized_embedding));
    try testing.expectEqual(MtpCostProfile.generic, m5NaxCostProfileForFingerprint(4, 64, .none));
    try testing.expectEqual(MtpCostProfile.generic, m5NaxCostProfileForFingerprint(3, 32, .uniform_quantized_embedding));


    var sidecar = try mk.qlinear(IN, OUT, 8, 32, s);
    defer sidecar.deinit();
    try testing.expect(m5NaxQLinearMatches(&sidecar, IN, OUT, 8, 32));
    try testing.expect(!m5NaxQLinearMatches(&sidecar, IN, OUT + 1, 8, 32));
    try testing.expect(!m5NaxQLinearMatches(&sidecar, IN, OUT, 4, 32));
    try testing.expect(!m5NaxQLinearMatches(&sidecar, IN, OUT, 8, 64));

    var sidecar_q4 = try mk.qlinear(IN, OUT, 4, 32, s);
    defer sidecar_q4.deinit();
    try testing.expect(m5NaxQLinearMatches(&sidecar_q4, IN, OUT, 4, 32));
    try testing.expect(!m5NaxQLinearMatches(&sidecar_q4, IN, OUT, 8, 32));
    try testing.expect(!m5NaxQLinearMatches(&sidecar_q4, IN, OUT, 4, 64));
    var off_group = try mk.qlinear(IN, OUT, 8, 64, s);
    defer off_group.deinit();
    try testing.expect(!m5NaxQLinearMatches(&off_group, IN, OUT, 8, 32));
    var sidecar_q4_gs64 = try mk.qlinear(IN, OUT, 4, 64, s);
    defer sidecar_q4_gs64.deinit();
    try testing.expect(m5NaxQLinearMatches(&sidecar_q4_gs64, IN, OUT, 4, 64));
    try testing.expect(!m5NaxQLinearMatches(&sidecar_q4_gs64, IN, OUT, 4, 32));

    var q4_set = [_]QLinear{
        try mk.qlinear(IN, OUT, 4, 32, s),
        try mk.qlinear(IN, OUT, 4, 32, s),
        try mk.qlinear(IN, OUT, 4, 32, s),
        try mk.qlinear(OUT, IN, 4, 32, s),
        try mk.qlinear(IN, OUT, 4, 32, s),
        try mk.qlinear(IN, OUT, 4, 32, s),
        try mk.qlinear(OUT, IN, 4, 32, s),
    };
    defer for (&q4_set) |*q| q.deinit();
    var q8_set = [_]QLinear{
        try mk.qlinear(IN, OUT, 8, 32, s),
        try mk.qlinear(IN, OUT, 8, 32, s),
        try mk.qlinear(IN, OUT, 8, 32, s),
        try mk.qlinear(OUT, IN, 8, 32, s),
        try mk.qlinear(IN, OUT, 8, 32, s),
        try mk.qlinear(IN, OUT, 8, 32, s),
        try mk.qlinear(OUT, IN, 8, 32, s),
    };
    defer for (&q8_set) |*q| q.deinit();
    const q4_linears: M5NaxDenseSidecarLinears = .{
        .q = &q4_set[0],
        .k = &q4_set[1],
        .v = &q4_set[2],
        .o = &q4_set[3],
        .gate = &q4_set[4],
        .up = &q4_set[5],
        .down = &q4_set[6],
    };
    const q8_linears: M5NaxDenseSidecarLinears = .{
        .q = &q8_set[0],
        .k = &q8_set[1],
        .v = &q8_set[2],
        .o = &q8_set[3],
        .gate = &q8_set[4],
        .up = &q8_set[5],
        .down = &q8_set[6],
    };
    const q4_geom: M5NaxDenseSidecarGeometry = .{
        .hidden = IN,
        .q_out = OUT,
        .kv_out = OUT,
        .full_out = OUT,
        .intermediate = OUT,
        .bits = 4,
        .group_size = 32,
    };
    const q8_geom: M5NaxDenseSidecarGeometry = .{
        .hidden = IN,
        .q_out = OUT,
        .kv_out = OUT,
        .full_out = OUT,
        .intermediate = OUT,
        .bits = 8,
        .group_size = 32,
    };
    try testing.expect(m5NaxDenseSidecarMatches(q4_linears, q4_geom));
    try testing.expect(m5NaxDenseSidecarMatches(q8_linears, q8_geom));
    var mixed = q4_linears;
    mixed.up = &q8_set[5];
    try testing.expect(!m5NaxDenseSidecarMatches(mixed, q4_geom));

    var draft = try mk.qlinear(IN, OUT, 3, 64, s);
    defer draft.deinit();
    try testing.expect(m5NaxQLinearMatches(&draft, IN, OUT, 3, 64));
    try testing.expect(m5NaxDraftHeadMatches(&draft, 3, 64, IN, OUT));
    try testing.expect(!m5NaxDraftHeadMatches(null, 3, 64, IN, OUT));
    try testing.expect(!m5NaxDraftHeadMatches(&draft, 4, 64, IN, OUT));
    try testing.expect(!m5NaxDraftHeadMatches(&draft, 3, 32, IN, OUT));
    try testing.expect(!m5NaxDraftHeadMatches(&sidecar, 3, 64, IN, OUT));

    var dense: QLinear = .{
        .w = mlx.mlx_array_new(),
        .s = mlx.mlx_array_new(),
        .b = mlx.mlx_array_new(),
    };
    defer dense.deinit();
    const dense_shape = [_]c_int{ @intCast(OUT), @intCast(IN) };
    try mlx.check(mlx.mlx_zeros(&dense.w, &dense_shape, 2, .bfloat16, s));
    try testing.expect(!m5NaxQLinearMatches(&dense, IN, OUT, 8, 32));
}

test "loadMtp: MoE sidecar layout (language_model. prefix, switch_mlp experts)" {
    // Synthetic 35B-A3B-shaped sidecar: `language_model.mtp.*` keys, MoE MLP
    // (router `mlp.gate` + 3D switch_mlp experts + shared expert + SEG), all
    // bf16 (quantized loading shares the same key paths). Red-on-revert: the
    // pre-MoE loader misses `mtp.fc.weight` (prefix) and `mlp.gate_proj`
    // (dense-only MLP) and returns error.MissingMtpWeight.
    const io = testing.io;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    const save_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(save_map);
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(allocator);
    }
    const put = struct {
        fn f(map: mlx.mlx_map_string_to_array, list: *std.ArrayList(mlx.mlx_array), alloc: std.mem.Allocator, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, .bfloat16, st));
            try mlx.check(mlx.mlx_array_eval(a));
            _ = mlx.mlx_map_string_to_array_insert(map, key, a);
            try list.append(alloc, a);
        }
    }.f;

    // hidden 8, head_dim 4, 2 q heads (x2 for the q/gate split), 4 experts,
    // expert inter 16, shared inter 16.
    try put(save_map, &owned, allocator, "language_model.mtp.fc.weight", &.{ 16, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.pre_fc_norm_embedding.weight", &.{8}, s);
    try put(save_map, &owned, allocator, "language_model.mtp.pre_fc_norm_hidden.weight", &.{8}, s);
    try put(save_map, &owned, allocator, "language_model.mtp.norm.weight", &.{8}, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.input_layernorm.weight", &.{8}, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.post_attention_layernorm.weight", &.{8}, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.self_attn.q_norm.weight", &.{4}, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.self_attn.k_norm.weight", &.{4}, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.self_attn.q_proj.weight", &.{ 16, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.self_attn.k_proj.weight", &.{ 8, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.self_attn.v_proj.weight", &.{ 8, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.self_attn.o_proj.weight", &.{ 8, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.mlp.gate.weight", &.{ 4, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight", &.{ 4, 16, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.mlp.switch_mlp.up_proj.weight", &.{ 4, 16, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight", &.{ 4, 8, 16 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.mlp.shared_expert.gate_proj.weight", &.{ 16, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.mlp.shared_expert.up_proj.weight", &.{ 16, 8 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.mlp.shared_expert.down_proj.weight", &.{ 8, 16 }, s);
    try put(save_map, &owned, allocator, "language_model.mtp.layers.0.mlp.shared_expert_gate.weight", &.{ 1, 8 }, s);

    var dir_buf: [512]u8 = undefined;
    const dir_n = try tmp.dir.realPath(io, &dir_buf);
    const dir_abs = dir_buf[0..dir_n];
    const file_path = try std.fs.path.joinZ(allocator, &.{ dir_abs, "model-mtp.safetensors" });
    defer allocator.free(file_path);
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    try mlx.check(mlx.mlx_save_safetensors(file_path.ptr, save_map, meta));

    var m = try loadMtp(io, allocator, s, dir_abs);
    defer m.deinit();

    // MoE arm selected; router pre-transposed for the trunk's dense fallback
    // ([hidden, experts]); packed switch experts kept raw 3D.
    switch (m.mlp) {
        .dense => return error.TestUnexpectedResult,
        .moe => |*mw| {
            const rs = mlx.getShape(mw.router_w);
            try testing.expectEqual(@as(c_int, 8), rs[0]);
            try testing.expectEqual(@as(c_int, 4), rs[1]);
            const sgs = mlx.getShape(mw.switch_gate_w);
            try testing.expectEqual(@as(usize, 3), sgs.len);
            try testing.expectEqual(@as(c_int, 4), sgs[0]);
            // Shared expert + SEG present (Qwen3.5-style gated combination).
            try testing.expect(mw.shared_expert_gate_w != null);
            // bf16 shared linears pre-transposed: [in, out] = [8, 16].
            const shs = mlx.getShape(mw.shared_gate_w);
            try testing.expectEqual(@as(c_int, 8), shs[0]);
            try testing.expectEqual(@as(c_int, 16), shs[1]);
        },
    }
    // fc transposed to [H, 2H].
    const fcs = mlx.getShape(m.fc.w);
    try testing.expectEqual(@as(c_int, 8), fcs[0]);
    try testing.expectEqual(@as(c_int, 16), fcs[1]);
}

/// A tiny OptiQ-shaped MoE head (`mlp.experts.{N}.{gate,up,down}_proj.*`, no `switch_mlp`) in a
/// temp dir: hidden 8, head_dim 4, a router over `EXPERTS` (two-digit indices catch a lexical order).
const PerExpertHead = struct {
    const EXPERTS = 12;
    const P = "mtp.layers.0.";
    tmp: std.testing.TmpDir,
    map: mlx.mlx_map_string_to_array,
    owned: std.ArrayList(mlx.mlx_array) = .empty,
    dir_buf: [512]u8 = undefined,
    dir_len: usize = 0,

    /// Leaves out every tensor whose name contains `drop`; writes `extra` experts past the router's.
    fn init(self: *PerExpertHead, drop: ?[]const u8, extra: usize) !void {
        const allocator = testing.allocator;
        const s = mlx.gpuStream();
        self.* = .{ .tmp = std.testing.tmpDir(.{ .iterate = true }), .map = mlx.mlx_map_string_to_array_new() };
        errdefer self.deinit();
        try self.put("mtp.fc.weight", &.{ 8, 16 }, drop, s);
        inline for (.{ "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight", P ++ "input_layernorm.weight", P ++ "post_attention_layernorm.weight" }) |n|
            try self.put(n, &.{8}, drop, s);
        try self.put(P ++ "self_attn.q_norm.weight", &.{4}, drop, s);
        try self.put(P ++ "self_attn.k_norm.weight", &.{4}, drop, s);
        try self.put(P ++ "self_attn.q_proj.weight", &.{ 16, 8 }, drop, s);
        inline for (.{ "k_proj", "v_proj", "o_proj" }) |n| try self.put(P ++ "self_attn." ++ n ++ ".weight", &.{ 8, 8 }, drop, s);
        try self.put(P ++ "mlp.gate.weight", &.{ EXPERTS, 8 }, drop, s);
        try self.put(P ++ "mlp.shared_expert.gate_proj.weight", &.{ 16, 8 }, drop, s);
        try self.put(P ++ "mlp.shared_expert.up_proj.weight", &.{ 16, 8 }, drop, s);
        try self.put(P ++ "mlp.shared_expert.down_proj.weight", &.{ 8, 16 }, drop, s);
        try self.put(P ++ "mlp.shared_expert_gate.weight", &.{ 1, 8 }, drop, s);
        var name_buf: [128]u8 = undefined;
        for (0..EXPERTS + extra) |e| {
            inline for (.{ .{ "gate_proj", 16, 8 }, .{ "up_proj", 16, 8 }, .{ "down_proj", 8, 16 } }) |proj| {
                inline for (.{ "weight", "scales", "biases" }) |suffix| {
                    const name = try std.fmt.bufPrint(&name_buf, P ++ "mlp.experts.{d}." ++ proj[0] ++ "." ++ suffix, .{e});
                    try self.put(name, &.{ proj[1], proj[2] }, drop, s);
                }
            }
        }
        self.dir_len = try self.tmp.dir.realPath(testing.io, &self.dir_buf);
        const file_path = try std.fs.path.joinZ(allocator, &.{ self.dir(), "mtp.safetensors" });
        defer allocator.free(file_path);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_save_safetensors(file_path.ptr, self.map, meta));
    }

    fn put(self: *PerExpertHead, name: []const u8, shape: []const c_int, drop: ?[]const u8, s: mlx.mlx_stream) !void {
        if (drop) |d| if (std.mem.indexOf(u8, name, d) != null) return;
        var a = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&a, shape.ptr, shape.len, .bfloat16, 1, 0.5, .{ .ctx = null }, s));
        self.owned.append(testing.allocator, a) catch |e| {
            _ = mlx.mlx_array_free(a);
            return e;
        };
        try mlx.check(mlx.mlx_array_eval(a));
        const name_z = try testing.allocator.dupeSentinel(u8, name, 0);
        defer testing.allocator.free(name_z);
        _ = mlx.mlx_map_string_to_array_insert(self.map, name_z.ptr, a);
    }

    fn dir(self: *const PerExpertHead) []const u8 {
        return self.dir_buf[0..self.dir_len];
    }

    fn deinit(self: *PerExpertHead) void {
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(testing.allocator);
        _ = mlx.mlx_map_string_to_array_free(self.map);
        self.tmp.cleanup();
    }
};

test "loadMtp: a MoE head with one tensor per expert loads as stacked switch banks, in expert order" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const P = PerExpertHead.P;
    const E = PerExpertHead.EXPERTS;
    var head: PerExpertHead = undefined;
    try head.init(null, 0);
    defer head.deinit();

    var m = try loadMtp(testing.io, allocator, s, head.dir());
    defer m.deinit();
    const mw = switch (m.mlp) {
        .moe => |*mw| mw,
        .dense => return error.TestUnexpectedResult,
    };
    const banks = [_]struct { proj: []const u8, suffix: []const u8, got: mlx.mlx_array }{
        .{ .proj = "gate_proj", .suffix = "weight", .got = mw.switch_gate_w },
        .{ .proj = "gate_proj", .suffix = "scales", .got = mw.switch_gate_s },
        .{ .proj = "up_proj", .suffix = "biases", .got = mw.switch_up_b },
        .{ .proj = "down_proj", .suffix = "weight", .got = mw.switch_down_w },
        .{ .proj = "down_proj", .suffix = "scales", .got = mw.switch_down_s },
    };
    var name_buf: [128]u8 = undefined;
    for (banks) |bank| {
        try testing.expectEqual(@as(c_int, E), mlx.getShape(bank.got)[0]);
        for (0..E) |e| {
            const name = try std.fmt.bufPrint(&name_buf, P ++ "mlp.experts.{d}.{s}.{s}", .{ e, bank.proj, bank.suffix });
            const name_z = try allocator.dupeSentinel(u8, name, 0);
            defer allocator.free(name_z);
            var want = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(want);
            try mlx.check(mlx.mlx_map_string_to_array_get(&want, head.map, name_z.ptr));
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            const ei: c_int = @intCast(e);
            const sh = mlx.getShape(bank.got);
            try mlx.check(mlx.mlx_slice(&row, bank.got, &[_]c_int{ ei, 0, 0 }, 3, &[_]c_int{ ei + 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&flat, row, sh[1..].ptr, 2, s));
            var same = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(same);
            try mlx.check(mlx.mlx_array_equal(&same, flat, want, false, s));
            try mlx.check(mlx.mlx_array_eval(same));
            var equal = false;
            try mlx.check(mlx.mlx_array_item_bool(&equal, same));
            try testing.expect(equal);
        }
    }
}

test "loadMtp: a per-expert MoE head that does not cover the router's experts is a named load error" {
    // A short or ragged bank behind the router lets gather_qmm read past it; it must fail at load.
    const cases = [_]struct { drop: ?[]const u8, extra: usize }{
        .{ .drop = "experts.7.", .extra = 0 },
        .{ .drop = "experts.7.up_proj.scales", .extra = 0 },
        .{ .drop = "experts.11.down_proj.biases", .extra = 0 },
        .{ .drop = null, .extra = 1 },
    };
    for (cases) |case| {
        var head: PerExpertHead = undefined;
        try head.init(case.drop, case.extra);
        defer head.deinit();
        const got = loadMtp(testing.io, testing.allocator, mlx.gpuStream(), head.dir());
        if (got) |m| {
            var mm = m;
            mm.deinit();
            std.debug.print("loaded a head with drop={?s} extra={d}\n", .{ case.drop, case.extra });
            return error.TestUnexpectedResult;
        } else |err| try testing.expectEqual(error.MissingMtpWeight, err);
    }
}

test "mtp: index.json shard sweep is marker-gated (in-checkpoint heads)" {
    const allocator = testing.allocator;

    // Jundot/oQ4e shape: the head rides the LAST shard of the trunk under
    // `language_model.mtp.*`; the sweep must name exactly that shard.
    const jundot =
        \\{"metadata":{"total_size":1},"weight_map":{
        \\ "language_model.model.layers.0.mlp.down_proj.weight":"model-00001-of-00004.safetensors",
        \\ "language_model.mtp.fc.weight":"model-00004-of-00004.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.q_proj.weight":"model-00004-of-00004.safetensors",
        \\ "language_model.mtp.norm.weight":"model-00004-of-00004.safetensors"}}
    ;
    const shards = try mtpShardsFromIndexJson(allocator, jundot);
    defer {
        for (shards) |sh| allocator.free(sh);
        allocator.free(shards);
    }
    try testing.expectEqual(@as(usize, 1), shards.len);
    try testing.expectEqualStrings("model-00004-of-00004.safetensors", shards[0]);

    // Auxiliary mtp.* keys WITHOUT a marker projection (fc / hy3 eh_proj)
    // never claim a loadable head — empty sweep, not a partial head that
    // dies later at ownWeight.
    const no_marker =
        \\{"weight_map":{"language_model.mtp.norm.weight":"a.safetensors",
        \\ "model.layers.0.mlp.up_proj.weight":"b.safetensors"}}
    ;
    const none = try mtpShardsFromIndexJson(allocator, no_marker);
    defer allocator.free(none);
    try testing.expectEqual(@as(usize, 0), none.len);

    // Bare-prefix (mtp.*) layout, head spanning TWO shards: both, deduped,
    // first-seen order.
    const two =
        \\{"weight_map":{"mtp.fc.weight":"s2.safetensors",
        \\ "mtp.norm.weight":"s2.safetensors",
        \\ "mtp.layers.0.self_attn.q_proj.weight":"s3.safetensors"}}
    ;
    const both = try mtpShardsFromIndexJson(allocator, two);
    defer {
        for (both) |sh| allocator.free(sh);
        allocator.free(both);
    }
    try testing.expectEqual(@as(usize, 2), both.len);
    try testing.expectEqualStrings("s2.safetensors", both[0]);
    try testing.expectEqualStrings("s3.safetensors", both[1]);
}

test "mtp: resolveMtpSource — sidecar file outranks in-checkpoint; markerless index is null" {
    const io = testing.io;
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try testing.expect(resolveMtpSource(io, allocator, tmp.dir) == null);

    // A trunk-only index is not a head.
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"model.layers.0.mlp.up_proj.weight":"model-00001-of-00002.safetensors"}}
    });
    try testing.expect(resolveMtpSource(io, allocator, tmp.dir) == null);

    // Index carrying the head → in-checkpoint.
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{"language_model.mtp.fc.weight":"model-00002-of-00002.safetensors"}}
    });
    try testing.expect(resolveMtpSource(io, allocator, tmp.dir).? == .in_checkpoint);

    // A sidecar FILE (with a marker-bearing header — name alone no longer
    // claims, see the dsv4-module test) always outranks the in-checkpoint
    // head — repos shipping both keep loading exactly what they loaded before.
    try writeFakeSidecar(io, tmp.dir, "mtp.safetensors", "mtp.fc.weight");
    const src = resolveMtpSource(io, allocator, tmp.dir).?;
    try testing.expect(src == .sidecar_file);
    try testing.expectEqualStrings("mtp.safetensors", src.sidecar_file);
}

test "mtp: single-file model.safetensors header probe (no index.json)" {
    const io = testing.io;
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // Minimal valid safetensors: 8-byte LE header length + header JSON + data.
    const W = struct {
        fn write(io_: std.Io, dir: std.Io.Dir, header: []const u8) !void {
            var buf: [512]u8 = undefined;
            std.mem.writeInt(u64, buf[0..8], @intCast(header.len), .little);
            @memcpy(buf[8..][0..header.len], header);
            buf[8 + header.len] = 0;
            buf[8 + header.len + 1] = 0;
            try dir.writeFile(io_, .{ .sub_path = "model.safetensors", .data = buf[0 .. 8 + header.len + 2] });
        }
    };

    try W.write(io, tmp.dir,
        \\{"language_model.mtp.fc.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}
    );
    try testing.expect(resolveMtpSource(io, allocator, tmp.dir).? == .in_checkpoint);

    // Head-less single-file checkpoint → null.
    try W.write(io, tmp.dir,
        \\{"model.embed_tokens.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}
    );
    try testing.expect(resolveMtpSource(io, allocator, tmp.dir) == null);

    // Garbage length prefix → null, never a huge allocation or a crash.
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "\xff\xff\xff\xff\xff\xff\xff\xff!!" });
    try testing.expect(resolveMtpSource(io, allocator, tmp.dir) == null);
}

test "mtp: loadMtp loads a dense head straight from checkpoint shards (oQ4e in-checkpoint layout)" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];

    // The dense one-layer head (toy bf16 geometry: hidden 8, 2 heads × hd 4,
    // 1 kv head, mlp inter 6) written into "shard 2". Shard 1 — the trunk —
    // deliberately does NOT exist on disk: the loader must open only the
    // shards the index names for mtp keys, never sweep the directory.
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model-00002-of-00002.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        const H = struct {
            fn put(m: mlx.mlx_map_string_to_array, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
                var total: usize = 1;
                for (shape) |d| total *= @intCast(d);
                const data = try std.testing.allocator.alloc(f32, total);
                defer std.testing.allocator.free(data);
                for (data, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.1 + 0.1;
                const f32_arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var bf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(bf);
                try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(bf));
                _ = mlx.mlx_map_string_to_array_insert(m, key, bf);
            }
            // Fill a tensor with a constant value — lets the test pin a norm's
            // mean exactly, to drive (or suppress) reference-based repair.
            fn putConst(m: mlx.mlx_map_string_to_array, key: [*:0]const u8, shape: []const c_int, value: f32, st: mlx.mlx_stream) !void {
                var total: usize = 1;
                for (shape) |d| total *= @intCast(d);
                const data = try std.testing.allocator.alloc(f32, total);
                defer std.testing.allocator.free(data);
                for (data) |*x| x.* = value;
                const f32_arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var bf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(bf);
                try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(bf));
                _ = mlx.mlx_map_string_to_array_insert(m, key, bf);
            }
            // A delta-encoded gamma: mostly `value`, with the single negative
            // entry that is all a real one carries on these norms (~0.2-0.8%).
            fn putDelta(m: mlx.mlx_map_string_to_array, key: [*:0]const u8, len: usize, value: f32, neg: f32, st: mlx.mlx_stream) !void {
                const data = try std.testing.allocator.alloc(f32, len);
                defer std.testing.allocator.free(data);
                for (data) |*x| x.* = value;
                data[len - 1] = neg;
                const shape = [_]c_int{@intCast(len)};
                const f32_arr = mlx.mlx_array_new_data(data.ptr, &shape, 1, .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var bf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(bf);
                try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(bf));
                _ = mlx.mlx_map_string_to_array_insert(m, key, bf);
            }
        };
        try H.put(map, "language_model.mtp.fc.weight", &.{ 8, 16 }, s);
        try H.put(map, "language_model.mtp.pre_fc_norm_embedding.weight", &.{8}, s);
        try H.put(map, "language_model.mtp.pre_fc_norm_hidden.weight", &.{8}, s);
        try H.put(map, "language_model.mtp.norm.weight", &.{8}, s);
        try H.put(map, "language_model.mtp.layers.0.input_layernorm.weight", &.{8}, s);
        try H.put(map, "language_model.mtp.layers.0.post_attention_layernorm.weight", &.{8}, s);
        // Head q_norm sits a full +1 below its backbone anchor (mean 0.7 vs
        // 1.4) and is DELTA-encoded (one negative entry, like every real
        // unfolded gamma) — the oQ conversion bug; k_norm sits at/above
        // (1.5 vs 1.4) — correct.
        try H.putDelta(map, "language_model.mtp.layers.0.self_attn.q_norm.weight", 32, 0.745, -0.7, s);
        try H.putConst(map, "language_model.mtp.layers.0.self_attn.k_norm.weight", &.{4}, 1.5, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.q_proj.weight", &.{ 8, 8 }, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.k_proj.weight", &.{ 4, 8 }, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.v_proj.weight", &.{ 4, 8 }, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.o_proj.weight", &.{ 8, 8 }, s);
        try H.put(map, "language_model.mtp.layers.0.mlp.gate_proj.weight", &.{ 6, 8 }, s);
        try H.put(map, "language_model.mtp.layers.0.mlp.up_proj.weight", &.{ 6, 8 }, s);
        try H.put(map, "language_model.mtp.layers.0.mlp.down_proj.weight", &.{ 8, 6 }, s);
        // Backbone counterpart norms ride the LAST trunk shard (this same file),
        // so the head's reference anchors are already loaded — no extra I/O.
        // Two q_norm layers exercise the mean-of-means anchor.
        try H.putConst(map, "language_model.model.layers.0.self_attn.q_norm.weight", &.{4}, 1.4, s);
        try H.putConst(map, "language_model.model.layers.1.self_attn.q_norm.weight", &.{4}, 1.4, s);
        try H.putConst(map, "language_model.model.layers.0.self_attn.k_norm.weight", &.{4}, 1.4, s);
        try H.putConst(map, "language_model.model.layers.0.post_attention_layernorm.weight", &.{8}, 1.4, s);
        try H.putConst(map, "language_model.model.norm.weight", &.{8}, 1.9, s);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    try tmp_dir.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{
        \\ "language_model.model.embed_tokens.weight":"model-00001-of-00002.safetensors",
        \\ "language_model.mtp.fc.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.pre_fc_norm_embedding.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.pre_fc_norm_hidden.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.norm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.input_layernorm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.post_attention_layernorm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.q_norm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.k_norm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.q_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.k_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.v_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.o_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.mlp.gate_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.mlp.up_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.mlp.down_proj.weight":"model-00002-of-00002.safetensors"}}
    });

    var m = try loadMtp(io, allocator, s, dir_path);
    defer m.deinit();

    // Dense flavor, bf16 fc bound and pre-transposed to [2H, H].
    try testing.expect(m.mlp == .dense);
    try testing.expect(m.fc.w.ctx != null);
    const fc_shape = mlx.getShape(m.fc.w);
    try testing.expectEqual(@as(c_int, 16), fc_shape[0]);
    try testing.expectEqual(@as(c_int, 8), fc_shape[1]);

    // oMLX head-norm repair: q_norm sat +1 below its backbone anchor (0.7 vs
    // 1.4) → repaired to ~1.7; k_norm sat at/above (1.5 vs 1.4) → untouched.
    const q_mean = try arrayMeanF32(m.q_norm, s);
    try testing.expect(q_mean > 1.6 and q_mean < 1.8);
    const k_mean = try arrayMeanF32(m.k_norm, s);
    try testing.expect(k_mean > 1.45 and k_mean < 1.55);
    // post_attention_layernorm is strictly positive (already folded) and sits
    // far below the 1.4 anchor — the gap alone must NOT convict it (the Alis
    // false repair: a correct 1.206 double-shifted to 2.206).
    const pa_mean = try arrayMeanF32(m.post_attn_norm, s);
    try testing.expect(pa_mean > 0.2 and pa_mean < 0.4);

    // …and binds against a hidden-8 trunk.
    try testing.expect(fcMatchesHidden(&m.fc, 8));
    try testing.expect(!fcMatchesHidden(&m.fc, 16));
}

test "mtp: a QUANTIZED fc loads verbatim and binds (avlp12 Alis layout)" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];

    // Toy dense head at hidden 16 (2 heads x hd 8, 1 kv head, inter 6) whose
    // `fc` ships QUANTIZED 4-bit/gs-32: w u32 [16, 4] + scales/biases [16, 1]
    // (4 packed cols x 8 values = 32 logical = 2H). Everything else is the
    // ordinary dense in-checkpoint layout.
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/model-00002-of-00002.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        const H = struct {
            fn bf16(shape: []const c_int, st: mlx.mlx_stream) !mlx.mlx_array {
                var total: usize = 1;
                for (shape) |d| total *= @intCast(d);
                const data = try std.testing.allocator.alloc(f32, total);
                defer std.testing.allocator.free(data);
                for (data, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 5)) * 0.1 + 0.1;
                const f32_arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var bf = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(bf);
                try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(bf));
                return bf;
            }
            fn put(m: mlx.mlx_map_string_to_array, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
                const bf = try bf16(shape, st);
                defer _ = mlx.mlx_array_free(bf);
                _ = mlx.mlx_map_string_to_array_insert(m, key, bf);
            }
        };
        {
            const dense_fc = try H.bf16(&.{ 16, 32 }, s);
            defer _ = mlx.mlx_array_free(dense_fc);
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, dense_fc, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(4), "affine", .{}, s));
            const names = [_][*:0]const u8{
                "language_model.mtp.fc.weight",
                "language_model.mtp.fc.scales",
                "language_model.mtp.fc.biases",
            };
            for (names, 0..) |name, i| {
                var a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a);
                try mlx.check(mlx.mlx_vector_array_get(&a, triple, i));
                try mlx.check(mlx.mlx_array_eval(a));
                _ = mlx.mlx_map_string_to_array_insert(map, name, a);
            }
        }
        try H.put(map, "language_model.mtp.pre_fc_norm_embedding.weight", &.{16}, s);
        try H.put(map, "language_model.mtp.pre_fc_norm_hidden.weight", &.{16}, s);
        try H.put(map, "language_model.mtp.norm.weight", &.{16}, s);
        try H.put(map, "language_model.mtp.layers.0.input_layernorm.weight", &.{16}, s);
        try H.put(map, "language_model.mtp.layers.0.post_attention_layernorm.weight", &.{16}, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.q_norm.weight", &.{8}, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.k_norm.weight", &.{8}, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.q_proj.weight", &.{ 16, 16 }, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.k_proj.weight", &.{ 8, 16 }, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.v_proj.weight", &.{ 8, 16 }, s);
        try H.put(map, "language_model.mtp.layers.0.self_attn.o_proj.weight", &.{ 16, 16 }, s);
        try H.put(map, "language_model.mtp.layers.0.mlp.gate_proj.weight", &.{ 6, 16 }, s);
        try H.put(map, "language_model.mtp.layers.0.mlp.up_proj.weight", &.{ 6, 16 }, s);
        try H.put(map, "language_model.mtp.layers.0.mlp.down_proj.weight", &.{ 16, 6 }, s);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    try tmp_dir.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data =
        \\{"weight_map":{
        \\ "language_model.model.embed_tokens.weight":"model-00001-of-00002.safetensors",
        \\ "language_model.mtp.fc.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.fc.scales":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.fc.biases":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.pre_fc_norm_embedding.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.pre_fc_norm_hidden.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.norm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.input_layernorm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.post_attention_layernorm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.q_norm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.k_norm.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.q_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.k_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.v_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.self_attn.o_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.mlp.gate_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.mlp.up_proj.weight":"model-00002-of-00002.safetensors",
        \\ "language_model.mtp.layers.0.mlp.down_proj.weight":"model-00002-of-00002.safetensors"}}
    });

    var m = try loadMtp(io, allocator, s, dir_path);
    defer m.deinit();

    // fc kept verbatim: packed [out=H, in_packed], NOT transposed, NOT dequantized.
    try testing.expect(m.fc.s.ctx != null);
    try testing.expect(m.fc.b.ctx != null);
    try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(m.fc.w));
    const fc_shape = mlx.getShape(m.fc.w);
    try testing.expectEqual(@as(c_int, 16), fc_shape[0]);
    try testing.expectEqual(@as(c_int, 4), fc_shape[1]);

    // …and binds against a hidden-16 trunk (this is what used to 400 the head
    // with MtpTargetMismatch: a packed shape compared against hidden_size * 2).
    try testing.expect(fcMatchesHidden(&m.fc, 16));
    try testing.expect(!fcMatchesHidden(&m.fc, 8));
}

fn fpSidecarCase(mode: [*:0]const u8, bits: c_int, group: c_int, want: QuantMode) !void {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];

    // Every fp quant mode ships a `(weight, scales)` PAIR: affine is the only
    // mode whose checkpoints carry per-group biases. `loadLinear` demanded
    // `.biases` whenever `.scales` was present, so the whole head failed with
    // `missing tensor: mtp.fc.biases` and MTP silently disabled — and
    // `qLinearFwd` then passed a literal "affine" to mlx_quantized_matmul,
    // which throws over fp8 scales. nvfp4 is the mode that shipped broken
    // (ollama's `*-mlx` tags, mlx-community's `*-nvfp4`); mxfp4 and mxfp8
    // are the same bug waiting, so the case runs for all three.
    // Hidden 128 so every group size under test divides both H and fc's 2H.
    const H: c_int = 128;
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/mtp.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        const Hlp = struct {
            fn bf16(shape: []const c_int, st: mlx.mlx_stream) !mlx.mlx_array {
                var total: usize = 1;
                for (shape) |d| total *= @intCast(d);
                const data = try std.testing.allocator.alloc(f32, total);
                defer std.testing.allocator.free(data);
                for (data, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.1 + 0.1;
                const f32_arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var bf = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(bf);
                try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(bf));
                return bf;
            }
            fn put(m: mlx.mlx_map_string_to_array, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
                const bf = try bf16(shape, st);
                defer _ = mlx.mlx_array_free(bf);
                _ = mlx.mlx_map_string_to_array_insert(m, key, bf);
            }
            /// fp-quantize a dense weight and insert ONLY `(weight, scales)`.
            fn putFp(m: mlx.mlx_map_string_to_array, prefix: []const u8, shape: []const c_int, st: mlx.mlx_stream, md: [*:0]const u8, bt: c_int, gp: c_int) !void {
                const dense = try bf16(shape, st);
                defer _ = mlx.mlx_array_free(dense);
                var pair = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(pair);
                try mlx.check(mlx.mlx_quantize(&pair, dense, mlx.mlx_optional_int.some(gp), mlx.mlx_optional_int.some(bt), md, .{}, st));
                const suffixes = [_][]const u8{ "weight", "scales" };
                for (suffixes, 0..) |suf, i| {
                    var a = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(a);
                    try mlx.check(mlx.mlx_vector_array_get(&a, pair, i));
                    try mlx.check(mlx.mlx_array_eval(a));
                    var key_buf: [128]u8 = undefined;
                    const key = try std.fmt.bufPrintSentinel(&key_buf, "{s}.{s}", .{ prefix, suf }, 0);
                    _ = mlx.mlx_map_string_to_array_insert(m, key.ptr, a);
                }
            }
        };
        // fc: [out=H, in=2H] nvfp4, no biases — the tensor that used to fail.
        try Hlp.putFp(map, "mtp.fc", &.{ H, H * 2 }, s, mode, bits, group);
        try Hlp.put(map, "mtp.pre_fc_norm_embedding.weight", &.{H}, s);
        try Hlp.put(map, "mtp.pre_fc_norm_hidden.weight", &.{H}, s);
        try Hlp.put(map, "mtp.norm.weight", &.{H}, s);
        try Hlp.put(map, "mtp.layers.0.input_layernorm.weight", &.{H}, s);
        try Hlp.put(map, "mtp.layers.0.post_attention_layernorm.weight", &.{H}, s);
        try Hlp.put(map, "mtp.layers.0.self_attn.q_norm.weight", &.{64}, s);
        try Hlp.put(map, "mtp.layers.0.self_attn.k_norm.weight", &.{64}, s);
        try Hlp.putFp(map, "mtp.layers.0.self_attn.q_proj", &.{ H, H }, s, mode, bits, group);
        try Hlp.putFp(map, "mtp.layers.0.self_attn.k_proj", &.{ 64, H }, s, mode, bits, group);
        try Hlp.putFp(map, "mtp.layers.0.self_attn.v_proj", &.{ 64, H }, s, mode, bits, group);
        try Hlp.putFp(map, "mtp.layers.0.self_attn.o_proj", &.{ H, H }, s, mode, bits, group);
        try Hlp.putFp(map, "mtp.layers.0.mlp.gate_proj", &.{ H, H }, s, mode, bits, group);
        try Hlp.putFp(map, "mtp.layers.0.mlp.up_proj", &.{ H, H }, s, mode, bits, group);
        try Hlp.putFp(map, "mtp.layers.0.mlp.down_proj", &.{ H, H }, s, mode, bits, group);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    var m = try loadMtp(io, allocator, s, dir_path);
    defer m.deinit();

    // The head loaded at all — biases are optional PER TENSOR, not mandatory.
    try testing.expect(m.fc.s.ctx != null);
    try testing.expect(m.fc.b.ctx == null);
    try testing.expect(m.q.s.ctx != null);
    try testing.expect(m.q.b.ctx == null);

    // Geometry resolves to nvfp4's 4 bits / group 16, and the mode is carried
    // so `qLinearFwd` does not hand fp8 scales to an "affine" matmul.
    try testing.expectEqual(@as(u32, @intCast(bits)), m.quant_bits);
    try testing.expectEqual(@as(u32, @intCast(group)), m.quant_group_size);
    try testing.expectEqual(want, m.quant_mode);

    // A packed fp fc still binds against its trunk — nvfp4's group of 16 is
    // outside `affineParamsFromGeometry`'s 32/64/128 set, so the affine solve
    // alone would reject it outright.
    try testing.expect(fcMatchesHidden(&m.fc, @intCast(H)));
    try testing.expect(!fcMatchesHidden(&m.fc, @intCast(H * 2)));

    // And the forward actually runs in the checkpoint's own mode — an
    // "affine" matmul over bias-less fp8 scales throws inside MLX.
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    const x_shape = [_]c_int{ 1, H * 2 };
    try mlx.check(mlx.mlx_zeros(&x, &x_shape, 2, .bfloat16, s));
    const y = try qLinearFwd(&m, x, &m.fc);
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_array_eval(y));
    const y_shape = mlx.getShape(y);
    try testing.expectEqual(@as(c_int, 1), y_shape[0]);
    try testing.expectEqual(H, y_shape[1]);
}

test "mtp: an nvfp4 sidecar loads bias-less and forwards in its own mode (ollama -mlx packs)" {
    try fpSidecarCase("nvfp4", 4, 16, .nvfp4);
}

test "mtp: an mxfp4 sidecar loads bias-less and forwards in its own mode" {
    try fpSidecarCase("mxfp4", 4, 32, .mxfp4);
}

test "mtp: an mxfp8 sidecar loads bias-less and forwards in its own mode" {
    try fpSidecarCase("mxfp8", 8, 32, .mxfp8);
}

test "mtp: rerankCoarseBytes prices the packed weights AND both bf16 group tables" {
    // The build log claims a size; a bill that forgot scales+biases would
    // under-report the qwen4 coarse head by 40 MB.
    // 248320 x 2560 at 3 bits = 238.4 MiB packed, + 2 x (248320 x 40 x 2 B).
    const packed_mib: u64 = 248320 * 2560 * 3 / 8;
    const tables: u64 = 2 * (248320 * (2560 / 64) * 2);
    try testing.expectEqual(packed_mib + tables, rerankCoarseBytes(248320, 2560, 3));
    // 2-bit is exactly two thirds of the packed half, tables unchanged.
    try testing.expectEqual(248320 * 2560 * 2 / 8 + tables, rerankCoarseBytes(248320, 2560, 2));
}

test "mtp: draftTop32 matches a host top-32 reference (bf16, ties, -inf)" {
    const allocator = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();

    // Two widths: the guard edge (just past TILES*TG with a ragged tail) and
    // the live Qwen3.5-family vocab.
    const counts = [_]c_int{ TOP32_MIN_ROWS + 7, 248320 };
    var prng = std.Random.DefaultPrng.init(0x7031);
    const rand = prng.random();

    for (counts) |n| {
        const un: usize = @intCast(n);
        // bf16-exact values with deliberate duplicates (ties) and a few -inf
        // (the suppress-mask spelling).
        const vals = try allocator.alloc(f32, un);
        defer allocator.free(vals);
        for (vals) |*v| {
            const level: f32 = @floatFromInt(rand.intRangeAtMost(i32, -512, 511));
            v.* = level * 0.125; // exact in bf16
        }
        vals[0] = -std.math.inf(f32);
        vals[un / 2] = -std.math.inf(f32);

        const shape = [_]c_int{n};
        const row_f32 = mlx.mlx_array_new_data(vals.ptr, &shape, 1, .float32);
        defer _ = mlx.mlx_array_free(row_f32);
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_astype(&row, row_f32, .bfloat16, s));

        const ids = try draftTop32(s, row, n);
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_array_eval(ids));
        const ids_ptr = mlx.mlx_array_data_uint32(ids) orelse return error.NoData;

        // Host reference: (value asc, index asc), take the tail 32.
        const Entry = struct { v: f32, i: u32 };
        const entries = try allocator.alloc(Entry, un);
        defer allocator.free(entries);
        for (vals, 0..) |v, i| entries[i] = .{ .v = v, .i = @intCast(i) };
        std.mem.sort(Entry, entries, {}, struct {
            fn lt(_: void, a: Entry, b: Entry) bool {
                if (a.v != b.v) return a.v < b.v;
                return a.i < b.i;
            }
        }.lt);

        var expect_set = std.AutoHashMap(u32, void).init(allocator);
        defer expect_set.deinit();
        for (entries[un - 32 ..]) |e| try expect_set.put(e.i, {});

        for (0..32) |k| {
            const id = ids_ptr[k];
            try testing.expect(expect_set.contains(id));
            _ = expect_set.remove(id); // no duplicates either
        }
        try testing.expectEqual(@as(u32, 0), expect_set.count());
    }
}

test "adaptiveDepthCapForMachine names the row it applied" {
    // The cap must be nameable at runtime or an M1 Pro's depth=4 is
    // indistinguishable from the EV controller having chosen 4 by itself —
    // same reason `dflash.blockCapForMachine` carries a label.
    try testing.expectEqualStrings("m1-pro", adaptiveDepthCapForMachine("Apple M1 Pro", 6).label);
    try testing.expectEqualStrings("m4-max", adaptiveDepthCapForMachine("Apple M4 Max", 6).label);
    try testing.expectEqualStrings("default", adaptiveDepthCapForMachine("", 6).label);
}

test "base M4 caps at 4 where M4 Pro/Max keep the default (2026-08-22 sweep)" {
    // Depth 4 is the only width where the cap BINDS, collapsing the plan to
    // one chunk; from 5 on every round pays an extension sync for ~1 more
    // accepted token and loses 17%. The probe's cost cliff says 6 here.
    try testing.expectEqual(@as(u32, 4), adaptiveDepthCapForMachine("Apple M4", 6).cap);
    try testing.expectEqualStrings("m4-base", adaptiveDepthCapForMachine("Apple M4", 6).label);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M4 Pro", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M4 Max", 6).cap);
    // Every row above is a HUMAN sweep and outranks the boot probe.
    for ([_][]const u8{ "Apple M4", "Apple M4 Max", "Apple M1 Pro", "Apple M5" }) |c| {
        try testing.expect(adaptiveDepthCapForMachine(c, 6).measured);
    }
    try testing.expect(!adaptiveDepthCapForMachine("Apple M2 Pro", 6).measured);
}

test "adaptiveDepthCapForMachine: measured rows only, unmeasured chips keep the default" {
    try testing.expectEqual(@as(u32, 4), adaptiveDepthCapForMachine("Apple M1 Pro", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M1 Max", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("", 6).cap);
}

test "adaptiveDepthCapForMachine: base M5 caps at 4, Pro/Max/Ultra keep the default" {
    try testing.expectEqual(@as(u32, 4), adaptiveDepthCapForMachine("Apple M5", 6).cap);
    try testing.expectEqualStrings("m5", adaptiveDepthCapForMachine("Apple M5", 6).label);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M5 Pro", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M5 Max", 6).cap);
    try testing.expectEqual(@as(u32, 6), adaptiveDepthCapForMachine("Apple M5 Ultra", 6).cap);
}

test "mtpCtxWithinLimit: 0 is unlimited and the ceiling is inclusive" {
    try testing.expect(mtpCtxWithinLimit(0, 0));
    try testing.expect(mtpCtxWithinLimit(0, 1_000_000));

    try testing.expect(mtpCtxWithinLimit(4096, 4095));
    try testing.expect(mtpCtxWithinLimit(4096, 4096));
    try testing.expect(!mtpCtxWithinLimit(4096, 4097));
    try testing.expect(!mtpCtxWithinLimit(4096, 1_000_000));

    try testing.expect(mtpCtxWithinLimit(1, 1));
    try testing.expect(!mtpCtxWithinLimit(1, 2));
}
