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
//! (src/generate.zig); deleting the feature is removing those two.

const std = @import("std");
const mlx = @import("mlx.zig");
const mrope = @import("mrope.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const log = @import("log.zig");
const io_util_mod = @import("io_util.zig");
const ane_mod = @import("ane.zig");

const Transformer = transformer_mod.Transformer;
const KVCache = transformer_mod.KVCache;
const Weights = model_mod.Weights;

/// Default draft depth (tokens drafted per round). Flipped 1 -> 3 after the
/// round-v2 rebuild made rejected drafts ~free (scalar-anchor rollback + the
/// 3-bit draft-only lm_head replaced the old full trunk re-forward): the old
/// cost model was what made depth 1 optimal. 2026-07-12 validation matrix on
/// Qwen3.6-27B 4-bit (M4 Max, adaptive controller active, decode tok/s
/// depth-3 vs depth-1): code 54.3 vs 43.3 (+25%), coding-agent ladder 2K
/// 52.1 vs 41.4 (+26%) and 16K 43.9 vs 38.1 (+15%), 2-turn pi agentic
/// (weighted) 48.6 vs 40.9 (+19%), creative temp-0.8 39.1 vs 37.9 (+3% —
/// the class that REGRESSED under the old cost model now holds even at ~30%
/// per-draft acceptance because the controller demotes without churn).
/// Users can cap rounds with `--mtp-depth`; the Generator's adaptive
/// controller demotes/promotes within [1, configured].
pub const DEFAULT_DEPTH: u32 = 3;
pub const MAX_DEPTH: u32 = 8;

/// Per-silicon adaptive depth cap for machines on the `.generic` cost
/// surface. The cap is a MACHINE measurement, so each row is one, never
/// interpolated between chips; an unmeasured chip keeps the default row.
///   M1 Pro: 4 (2026-08-20, Qwen3.8-27B iQ-3.8bpw, forced-depth sweep:
///     13.01 tok/s at depth 4 vs 10.78/9.63 at 5/6 — the verify width 6
///     cliff; auto at cap 6 measured 10.64, barely over --no-mtp's 10.57).
/// `chip` is sysctl machdep.cpu.brand_string via `ane_mod.chipBrand`
/// (the GPU arch string cannot tell Ultra from Max); "" lands on default.
/// The row carries its own LABEL so the resolve site can say which one it
/// applied: a bare depth=4 in the spec-stats line is indistinguishable from
/// the EV controller having picked 4 on its own, or from `--mtp-depth 4`.
/// `measured` marks a row a HUMAN swept as realized throughput. Those beat the
/// boot probe's cost ladder, which cannot see acceptance or the extension sync
/// — see `generate.mtpDepthCapResolved`.
pub const DepthCap = struct { cap: u32, label: []const u8, measured: bool = false };

pub fn adaptiveDepthCapForMachine(chip: []const u8, default_cap: u32) DepthCap {
    if (std.mem.indexOf(u8, chip, "M1 Pro") != null) return .{ .cap = 4, .label = "m1-pro", .measured = true };
    // Base M4 (2026-08-22, tester report, Qwen3.5-9B-MTPLX 6-bit, 16 GB):
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
    return transformer_mod.affineParamsFromGeometry(fc.w, fc.s, hidden_size * 2) != null;
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
    const qp = transformer_mod.affineParamsFromGeometry(q.w, q.s, in_dim) orelse return false;
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
    moe: transformer_mod.MoeMlpWeights,

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

pub const MtpModel = struct {
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,

    /// Quant params for the MTP layer's own linears — inferred from tensor
    /// geometry at load (sidecars are often quantized differently from the
    /// trunk, e.g. group 32 over a group-64 trunk, or 8-bit over 4-bit).
    quant_bits: u32,
    quant_group_size: u32,

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

    /// Optional DRAFT-ONLY low-bit lm_head, requantized from the trunk's at
    /// bind time (MLX_SERVE_MTP_DRAFT_HEAD_BITS, default 3, 0 disables).
    /// Only draft steps project through it — VERIFICATION always uses the
    /// trunk head, so the output distribution is untouched; drafts just read
    /// ~40% fewer bytes per full-vocab projection (the dominant draft cost).
    draft_head: ?QLinear = null,
    draft_head_bits: u32 = 0,
    draft_head_group: u32 = 0,

    /// Draft-rerank scheme (MLX_SERVE_MTP_DRAFT_RERANK): a coarse 2-bit/gs64
    /// requant of the trunk lm_head; greedy drafts shortlist its top-32 and
    /// re-score them through the trunk head's own rows (`draftSelect`). When
    /// built, the 3-bit draft head is dropped — non-greedy draft paths fall
    /// back to the trunk head.
    rerank_coarse: ?QLinear = null,
    rerank_rows: c_int = 0,
    rerank_logged: bool = false,

    /// Cross-request EV controller seed (inference thread only, like every
    /// mutable field here): the last HEALTHY request's per-index acceptance
    /// EMAs + base depth, written by `Generator.deinit`, consumed by the
    /// first `nextMtp` round of the next request. A fresh controller burns
    /// ~10 legacy-warmup rounds plus a +1/round base climb per request —
    /// a third of a short protocol-style generation; seeding restores the
    /// learned surface from round 1. Never written by disabled/short runs;
    /// set MLX_SERVE_MTP_EV_SEED=0 to opt into request isolation.
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

    /// Select the exact full-round cost surface for this bound sidecar and
    /// target. Every G17 profile requires the successfully built 3-bit/gs-64
    /// draft-only lm_head and an exact dense 27B sidecar. Uniform q4/q6/q8
    /// gs64 trunks and mixed-q4/q5/q6 oQ4e trunks remain separate profiles;
    /// target embedding storage is part of the fingerprint too.
    /// Compatible but off-profile geometry remains correct under `generic`.
    pub fn m5NaxCostProfile(self: *const MtpModel, target: *const Transformer) MtpCostProfile {
        if (self.eh_proj != null) return .generic;
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
        const profile = m5NaxCostProfileForFingerprint(
            self.quant_bits,
            self.quant_group_size,
            target_surface,
        );
        if (profile == .generic) return .generic;
        const sidecar_bits: u32 = switch (profile) {
            .g17_nax_q8_gs32, .g17_nax_q8_gs64 => 8,
            .g17_nax_q6_gs64 => 6,
            .g17_nax_q4_gs32, .g17_nax_q4_gs64, .g17_nax_oq4e_q4_gs64 => 4,
            .generic => return .generic,
        };
        const sidecar_group_size: u32 = switch (profile) {
            .g17_nax_q8_gs32, .g17_nax_q4_gs32 => 32,
            .g17_nax_q4_gs64, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64 => 64,
            .generic => return .generic,
        };

        const cfg = &target.config;
        const full_out_wide = @as(u64, cfg.num_attention_heads) * cfg.head_dim;
        const q_out_wide = full_out_wide * 2;
        const kv_out_wide = @as(u64, cfg.num_key_value_heads) * cfg.head_dim;
        if (full_out_wide == 0 or full_out_wide > std.math.maxInt(u32) or
            q_out_wide > std.math.maxInt(u32) or
            kv_out_wide == 0 or kv_out_wide > std.math.maxInt(u32)) return .generic;
        const full_out: u32 = @intCast(full_out_wide);
        const q_out: u32 = @intCast(q_out_wide);
        const kv_out: u32 = @intCast(kv_out_wide);

        const fc_shape = mlx.getShape(self.fc.w);
        if (mlx.mlx_array_dtype(self.fc.w) != .bfloat16 or
            fc_shape.len != 2 or
            fc_shape[0] != @as(c_int, @intCast(cfg.hidden_size * 2)) or
            fc_shape[1] != @as(c_int, @intCast(cfg.hidden_size))) return .generic;
        if (!m5NaxNormMatches(self.pre_fc_norm_emb, cfg.hidden_size) or
            !m5NaxNormMatches(self.pre_fc_norm_hidden, cfg.hidden_size) or
            !m5NaxNormMatches(self.final_norm, cfg.hidden_size) or
            !m5NaxNormMatches(self.input_norm, cfg.hidden_size) or
            !m5NaxNormMatches(self.post_attn_norm, cfg.hidden_size) or
            !m5NaxNormMatches(self.q_norm, cfg.head_dim) or
            !m5NaxNormMatches(self.k_norm, cfg.head_dim)) return .generic;

        switch (self.mlp) {
            .dense => |*mlp| {
                if (!m5NaxDenseSidecarMatches(
                    .{
                        .q = &self.q,
                        .k = &self.k,
                        .v = &self.v,
                        .o = &self.o,
                        .gate = &mlp.gate,
                        .up = &mlp.up,
                        .down = &mlp.down,
                    },
                    .{
                        .hidden = cfg.hidden_size,
                        .q_out = q_out,
                        .kv_out = kv_out,
                        .full_out = full_out,
                        .intermediate = cfg.intermediate_size,
                        .bits = sidecar_bits,
                        .group_size = sidecar_group_size,
                    },
                )) return .generic;
            },
            .moe => return .generic,
        }

        const draft: ?*const QLinear = if (self.draft_head) |*q| q else null;
        return if (m5NaxDraftHeadMatches(
            draft,
            self.draft_head_bits,
            self.draft_head_group,
            cfg.hidden_size,
            cfg.vocab_size,
        )) profile else .generic;
    }

    /// Legacy q8 boolean view retained for source compatibility. New callers
    /// should use `m5NaxCostProfile` to distinguish q8 and q4 surfaces.
    pub fn m5NaxCostProfileEnabled(self: *const MtpModel, target: *const Transformer) bool {
        return self.m5NaxCostProfile(target) == .g17_nax_q8_gs32;
    }

    /// Validate the head against the target trunk: dims must line up and the
    /// trunk must be a Qwen 3.5/3.6-family hybrid (full-attention MTP layer
    /// cross-checks `attn_output_gate`). On success, optionally builds the
    /// draft-only low-bit lm_head (a failed build only logs — drafts fall
    /// back to the trunk head).
    pub fn bind(self: *MtpModel, target: *Transformer) !void {
        const cfg = &target.config;
        if (self.eh_proj != null) {
            // Hy3 head: no attention output gate, sigmoid-router MoE.
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
            self.buildDraftHead(target) catch |err| {
                log.warn("[mtp] draft lm_head build failed ({s}) — drafts use the trunk head\n", .{@errorName(err)});
            };
            self.maybeBuildDraftRerank(target);
            return;
        }
        if (!cfg.attn_output_gate) return error.UnsupportedMtpArch;
        if (!fcMatchesHidden(&self.fc, cfg.hidden_size)) return error.MtpTargetMismatch;

        self.buildDraftHead(target) catch |err| {
            log.warn("[mtp] draft lm_head build failed ({s}) — drafts use the trunk head\n", .{@errorName(err)});
        };
        self.maybeBuildDraftRerank(target);
        // NOTE (2026-07-13): mlx_compile'ing the offset-free front/back
        // halves of the seq-1 draft step (the compileMoeRouting pattern,
        // weights captured via payload) was built, verified equivalent at
        // toy scale, and A/B'd live on the 27B — DEAD EVEN at depths 3 and 6
        // (interleaved traced boots, <0.2 ms/step delta). The draft step is
        // qmm-weight-read-bound and MLX's lazy pipeline already batches
        // dispatch, so there is no launch overhead for compile to remove.
        // Removed rather than shipped dark; frontChain/backChain keep the
        // step's halves factored if a future backend changes the calculus.
    }

    /// MLX_SERVE_MTP_DRAFT_HEAD_BITS: absent → 3 (default on)
    /// a supported bit width → that; anything else ("0", "off") → disabled.
    fn draftHeadBitsFromEnv() u32 {
        const p = std.c.getenv("MLX_SERVE_MTP_DRAFT_HEAD_BITS") orelse return 3;
        const raw = std.mem.span(p);
        const v = std.fmt.parseInt(u32, raw, 10) catch return 0;
        return switch (v) {
            2, 3, 4, 6, 8 => v,
            else => 0,
        };
    }

    fn buildDraftHead(self: *MtpModel, target: *Transformer) !void {
        const bits = draftHeadBitsFromEnv();
        if (bits == 0) return;
        if (target.lm_head_s.ctx == null) return; // dense bf16 head — nothing to shrink
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

    /// MLX_SERVE_MTP_DRAFT_RERANK: "0" off, "1" force-on, absent/other →
    /// auto (on where the cost surface is generic; the calibrated G17 NAX
    /// surfaces keep the 3-bit draft head they were measured with).
    fn draftRerankMode() enum { auto, on, off } {
        const p = std.c.getenv("MLX_SERVE_MTP_DRAFT_RERANK") orelse return .auto;
        const raw = std.mem.span(p);
        if (std.mem.eql(u8, raw, "0")) return .off;
        if (std.mem.eql(u8, raw, "1")) return .on;
        return .auto;
    }

    /// Build the coarse 2-bit rerank head (see the draft-rerank section
    /// below). Infallible by design: any refusal keeps the draft-head path.
    fn maybeBuildDraftRerank(self: *MtpModel, target: *Transformer) void {
        const mode = draftRerankMode();
        if (mode == .off) return;
        if (target.lm_head_s.ctx == null) return; // dense bf16 head — row-gather rerank unbuilt/unmeasured
        if (mode == .auto and self.m5NaxCostProfile(target) != .generic) return;
        const w_shape = mlx.getShape(target.lm_head_w);
        if (w_shape.len != 2 or w_shape[0] < TOP32_MIN_ROWS) return;
        const head_qp = headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
        if (head_qp.bits <= 2) return; // nothing coarser to gain

        var rc = requantizeRows(
            self.s,
            target.lm_head_w,
            target.lm_head_s,
            target.lm_head_b,
            head_qp.group_size,
            head_qp.bits,
            head_qp.mode.cstr(),
            64,
            2,
            32768,
        ) catch |err| {
            log.warn("[mtp] draft rerank coarse build failed ({s}) — keeping the draft-head path\n", .{@errorName(err)});
            return;
        };

        // Materialize now so the first draft doesn't pay for it.
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, rc.w);
            _ = mlx.mlx_vector_array_append_value(eval_vec, rc.s);
            _ = mlx.mlx_vector_array_append_value(eval_vec, rc.b);
            mlx.check(mlx.mlx_eval(eval_vec)) catch {
                rc.deinit();
                log.warn("[mtp] draft rerank coarse eval failed — keeping the draft-head path\n", .{});
                return;
            };
        }

        if (self.draft_head) |*dh| {
            dh.deinit();
            self.draft_head = null;
            self.draft_head_bits = 0;
            self.draft_head_group = 0;
        }
        self.rerank_coarse = rc;
        self.rerank_rows = w_shape[0];
        log.info("[mtp] draft rerank: coarse 2-bit/gs64 head built ({d} rows); greedy drafts re-rank through the trunk head\n", .{w_shape[0]});
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
        self: *MtpModel,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        const s = self.s;
        // A mid-chain kernel failure nulls rerank_coarse; later steps of the
        // same chain land here and must still answer.
        const rc = if (self.rerank_coarse) |*p| p else return self.draftFallbackArgmax(target, x, suppress_mask);

        var coarse = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(coarse);
        try mlx.check(mlx.mlx_quantized_matmul(
            &coarse,
            x,
            rc.w,
            rc.s,
            rc.b,
            true,
            mlx.mlx_optional_int.some(64),
            mlx.mlx_optional_int.some(2),
            "affine",
            s,
        ));
        if (suppress_mask) |m| {
            const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
            defer _ = mlx.mlx_array_free(neg_inf);
            var masked = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_where(&masked, m, neg_inf, coarse, s));
            _ = mlx.mlx_array_free(coarse);
            coarse = masked;
        }
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const flat_shape = [_]c_int{self.rerank_rows};
        try mlx.check(mlx.mlx_reshape(&flat, coarse, &flat_shape, 1, s));

        const cands = draftTop32(s, flat, self.rerank_rows) catch |err| {
            log.warn("[mtp] draft rerank shortlist failed ({s}) — dropping to the trunk-head readout\n", .{@errorName(err)});
            var dead = self.rerank_coarse.?;
            dead.deinit();
            self.rerank_coarse = null;
            return self.draftFallbackArgmax(target, x, suppress_mask);
        };
        defer _ = mlx.mlx_array_free(cands);

        if (!self.rerank_logged) {
            log.info("[mtp] draft rerank engaged (2-bit coarse → top-32 → trunk re-score)\n", .{});
            self.rerank_logged = true;
        }

        // Re-score the shortlist through the trunk head's own rows: gathered
        // packed rows are self-contained (w [V, K·bits/32], scales/biases
        // [V, K/gs]), so a 32-row quantized matmul is exact.
        const qp = headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
        var w32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w32);
        var s32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(s32);
        var b32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b32);
        try mlx.check(mlx.mlx_take_axis(&w32, target.lm_head_w, cands, 0, s));
        try mlx.check(mlx.mlx_take_axis(&s32, target.lm_head_s, cands, 0, s));
        if (target.lm_head_b.ctx != null)
            try mlx.check(mlx.mlx_take_axis(&b32, target.lm_head_b, cands, 0, s));

        var exact = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(exact);
        try mlx.check(mlx.mlx_quantized_matmul(
            &exact,
            x,
            w32,
            s32,
            b32,
            true,
            mlx.mlx_optional_int.some(@intCast(qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(qp.bits)),
            qp.mode.cstr(),
            s,
        ));

        var amax = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(amax);
        try mlx.check(mlx.mlx_argmax_axis(&amax, exact, -1, false, s));
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, cands, amax, 0, s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&out, picked, .int32, s));
        return out;
    }

    /// Rerank's failure fallback: full trunk-head readout + argmax, same
    /// [1,1] int32 shape as `draftSelect`.
    fn draftFallbackArgmax(
        self: *const MtpModel,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        const s = self.s;
        var logits = try targetLmHead(self, target, x, s);
        defer _ = mlx.mlx_array_free(logits);
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
};

/// Sidecar file layouts we accept, in priority order. The native layout wins
/// so a repo shipping several keeps loading exactly what it loaded before.
/// Root-level names are what others publish (mutual compat: their
/// loader accepts our `mtp/weights.safetensors` too).
pub const sidecar_rel_paths = [_][]const u8{
    "mtp/weights.safetensors", // mlx-serve native (ddalcu repos, build_mtp_sidecar.py)
    "mtp.safetensors", // others
    "model-mtp.safetensors", // others
    "optiq/mtp.safetensors", // oMLX OptiQ (delta-encoded norms — folded at load)
};

/// Relative path (one of `sidecar_rel_paths`) of the first sidecar file under
/// `dir` whose HEADER carries a marker key, or null when the model ships no
/// loadable MTP head. The marker gate is the same one discovery and the
/// in-checkpoint shard sweep apply — a file at a sidecar PATH is only a
/// sidecar when it provably contains a head this loader can bind (DeepSeek-V4
/// mirrors ship their own dsv4-shaped MTP module at `model-mtp.safetensors`;
/// claiming it by NAME alone sent the qwen-shaped loader into
/// MissingMtpWeight on every `--mtp` boot, live 2026-07-31).
pub fn resolveMtpSidecarInDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) ?[]const u8 {
    for (&sidecar_rel_paths) |rel| {
        const st = dir.statFile(io, rel, .{}) catch continue;
        if (st.size == 0) continue;
        if (!safetensorsHeaderHasMtpHead(io, allocator, dir, rel)) continue;
        return rel;
    }
    return null;
}

/// Where a model's MTP head lives.
pub const MtpSource = union(enum) {
    /// A separate sidecar file — rel path, one of `sidecar_rel_paths`.
    sidecar_file: []const u8,
    /// Inside the main checkpoint safetensors (Qwen HF releases and oMLX
    /// oQ4e-class conversions ship `[language_model.]mtp.*` in the trunk
    /// shards). Loading reads ONLY the shards the index names for mtp keys.
    in_checkpoint,
};

/// Marker projections that prove a LOADABLE head: `fc` (Qwen dense/MoE
/// layouts) or `eh_proj` (hy3). Discovery and the shard sweep both gate on
/// this same set, so a checkpoint with stray `mtp.*` auxiliaries but no
/// marker never claims a head it can't bind.
const mtp_marker_keys = [_][]const u8{
    "mtp.fc.weight",
    "language_model.mtp.fc.weight",
    "mtp.eh_proj.weight",
    "language_model.mtp.eh_proj.weight",
};

/// Any tensor belonging to the head (either root prefix).
fn isMtpHeadKey(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "mtp.") or
        std.mem.indexOf(u8, key, ".mtp.") != null;
}

/// Sanity bound for index.json / safetensors headers (the Jundot 27B index
/// is ~212 KB; headers of the largest checkpoints stay well under this).
const checkpoint_header_limit: usize = 64 * 1024 * 1024;

/// Parse a sharded checkpoint's `model.safetensors.index.json` and return
/// the unique shard basenames holding MTP-head tensors, in first-seen order
/// (caller frees each name + the slice). Empty when the checkpoint carries
/// no loadable head — the sweep is gated on `mtp_marker_keys`, so partial
/// auxiliaries never produce a doomed load.
fn mtpShardsFromIndexJson(allocator: std.mem.Allocator, bytes: []const u8) ![][]u8 {
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |sh| allocator.free(sh);
        out.deinit(allocator);
    }
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch
        return out.toOwnedSlice(allocator);
    defer parsed.deinit();
    if (parsed.value != .object) return out.toOwnedSlice(allocator);
    const weight_map = parsed.value.object.get("weight_map") orelse
        return out.toOwnedSlice(allocator);
    if (weight_map != .object) return out.toOwnedSlice(allocator);

    var has_marker = false;
    for (&mtp_marker_keys) |marker| {
        if (weight_map.object.get(marker) != null) {
            has_marker = true;
            break;
        }
    }
    if (!has_marker) return out.toOwnedSlice(allocator);

    var it = weight_map.object.iterator();
    outer: while (it.next()) |entry| {
        if (!isMtpHeadKey(entry.key_ptr.*)) continue;
        if (entry.value_ptr.* != .string) continue;
        const shard = entry.value_ptr.string;
        for (out.items) |seen| {
            if (std.mem.eql(u8, seen, shard)) continue :outer;
        }
        try out.append(allocator, try allocator.dupe(u8, shard));
    }
    return out.toOwnedSlice(allocator);
}

/// Read a file under `dir` fully (bounded); null on absence or overflow.
fn readDirFileAlloc(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, sub_path: []const u8, limit: usize) ?[]u8 {
    const f = dir.openFile(io, sub_path, .{}) catch return null;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(limit)) catch null;
}

/// True when the sharded index names an in-checkpoint head.
fn indexJsonHasMtpHead(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) bool {
    const bytes = readDirFileAlloc(io, allocator, dir, "model.safetensors.index.json", checkpoint_header_limit) orelse return false;
    defer allocator.free(bytes);
    const shards = mtpShardsFromIndexJson(allocator, bytes) catch return false;
    defer {
        for (shards) |sh| allocator.free(sh);
        allocator.free(shards);
    }
    return shards.len > 0;
}

/// Single-file checkpoints have no index — peek the safetensors JSON header
/// (8-byte LE length prefix) for a marker key, without touching tensor data.
/// Marker names are plain ASCII (dots/letters), so a quoted substring scan
/// is exact — no JSON-escape variants exist for them.
fn safetensorsHeaderHasMtpHead(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, sub_path: []const u8) bool {
    const f = dir.openFile(io, sub_path, .{}) catch return false;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    const header_len = rs.interface.takeInt(u64, .little) catch return false;
    if (header_len == 0 or header_len > checkpoint_header_limit) return false;
    const header = allocator.alloc(u8, @intCast(header_len)) catch return false;
    defer allocator.free(header);
    rs.interface.readSliceAll(header) catch return false;
    for (&mtp_marker_keys) |marker| {
        var quoted_buf: [64]u8 = undefined;
        const quoted = std.fmt.bufPrint(&quoted_buf, "\"{s}\"", .{marker}) catch continue;
        if (std.mem.indexOf(u8, header, quoted) != null) return true;
    }
    return false;
}

/// Resolve where (if anywhere) this model's MTP head lives. A sidecar file
/// always outranks an in-checkpoint head so repos shipping both keep
/// loading exactly what they loaded before.
pub fn resolveMtpSource(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) ?MtpSource {
    if (resolveMtpSidecarInDir(io, allocator, dir)) |rel| return .{ .sidecar_file = rel };
    if (indexJsonHasMtpHead(io, allocator, dir)) return .in_checkpoint;
    if (safetensorsHeaderHasMtpHead(io, allocator, dir, "model.safetensors")) return .in_checkpoint;
    return null;
}

/// True when `model_dir` carries an MTP head we know how to load — a
/// sidecar file OR in-checkpoint tensors. `model_dir` is absolute (same
/// contract as `model.parseConfig`).
pub fn hasMtpHead(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return false;
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    return resolveMtpSource(io, allocator, dir) != null;
}

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

/// Head-trunk requantization width (`MLX_SERVE_MTP_HEAD_QUANT_BITS`): dense
/// bf16 trunk weights (q/k/v/o, gate/up/down) are affine-quantized to this
/// width at group 64 during load — the head only PROPOSES tokens (verify
/// corrects), so the cost is acceptance, never output. 0 disables. Idea from
/// Layr-Labs/qwen-3.8-mtp-challenge @ deb63ad (see NOTICE). Test seam:
/// `head_quant_override`.
pub const DEFAULT_HEAD_QUANT_BITS: u32 = 4;
pub const HEAD_QUANT_GROUP: u32 = 64;
pub var head_quant_override: ?u32 = null;
fn headQuantBits() u32 {
    if (head_quant_override) |v| return v;
    const p = std.c.getenv("MLX_SERVE_MTP_HEAD_QUANT_BITS") orelse return DEFAULT_HEAD_QUANT_BITS;
    const v = std.fmt.parseInt(u32, std.mem.sliceTo(p, 0), 10) catch return DEFAULT_HEAD_QUANT_BITS;
    return switch (v) {
        0, 2, 3, 4, 5, 6, 8 => v,
        else => DEFAULT_HEAD_QUANT_BITS,
    };
}

/// Per-load accounting for the head-trunk requant log line.
const HeadQuantStats = struct {
    n: u32 = 0,
    before_bytes: u64 = 0,
    after_bytes: u64 = 0,
};

/// Load a (possibly quantized) linear `<prefix>.{weight,scales,biases}`.
/// bf16 weights (no scales) are pre-transposed for plain matmul.
fn loadLinear(w: *const Weights, allocator: std.mem.Allocator, prefix: []const u8, s: mlx.mlx_stream) !QLinear {
    var key_buf: [256]u8 = undefined;
    const scales_key = try std.fmt.bufPrint(&key_buf, "{s}.scales", .{prefix});
    if (w.get(scales_key) != null) {
        var key_buf2: [256]u8 = undefined;
        return .{
            .w = try ownWeight(w, try std.fmt.bufPrint(&key_buf2, "{s}.weight", .{prefix})),
            .s = try ownWeight(w, try std.fmt.bufPrint(&key_buf2, "{s}.scales", .{prefix})),
            .b = try ownWeight(w, try std.fmt.bufPrint(&key_buf2, "{s}.biases", .{prefix})),
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
fn loadMoeTriple(w: *const Weights, prefix: []const u8) !struct { w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array } {
    var key_buf: [256]u8 = undefined;
    return .{
        .w = try ownWeight(w, try std.fmt.bufPrint(&key_buf, "{s}.weight", .{prefix})),
        .s = ownWeightOpt(w, try std.fmt.bufPrint(&key_buf, "{s}.scales", .{prefix})),
        .b = ownWeightOpt(w, try std.fmt.bufPrint(&key_buf, "{s}.biases", .{prefix})),
    };
}

/// Load the head's weights from the MAIN checkpoint: the shards the index
/// names for `mtp.*` keys (typically one), or the single
/// `model.safetensors` when there is no index. Safetensors loads are
/// lazy/mmapped — pulling a multi-GB shard in costs its header parse; only
/// the head's tensors ever materialize, and `weights.deinit()` after the
/// head build releases the rest untouched.
fn loadMtpWeightsFromCheckpoint(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !model_mod.Weights {
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
        var weights = model_mod.Weights.init(allocator);
        errdefer weights.deinit();
        const s = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(s);
        for (shards) |sh| {
            const p = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ model_dir, sh });
            const pz = try allocator.dupeSentinel(u8, p, 0);
            defer allocator.free(pz);
            try model_mod.loadSafetensorsFile(allocator, &weights, pz, s, false);
        }
        return weights;
    }
    const single = try std.fmt.bufPrint(&path_buf, "{s}/model.safetensors", .{model_dir});
    return model_mod.loadWeightsSingleFile(allocator, single);
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
            break :blk try model_mod.loadWeightsSingleFile(allocator, sidecar_path);
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

    // Hy3 (hy_v3) layout: `mtp.eh_proj` + `mtp.layer.*` (full decoder layer,
    // sigmoid-router MoE). Detected by its distinctive projection name.
    if (weights.get(K.k(&kb, p, "eh_proj.weight")) != null) {
        return loadHy3Mtp(allocator, s, &weights, p);
    }

    // MLP flavor: a `switch_mlp` router/expert pack marks a MoE-trunk sidecar
    // (35B-A3B); plain gate/up/down is the dense one-layer head.
    const is_moe = weights.get(K.k(&kb, p, "layers.0.mlp.switch_mlp.gate_proj.weight")) != null;

    // Delta-encoded norms (Qwen original layout, oMLX OptiQ) need `+1` folded
    // in at load so the runtime `rmsnorm(x) * w` matches; a natively-folded
    // mlx-serve sidecar has strictly-positive norms and is left untouched.
    const fold_norms = mtpNormsAreDeltaEncoded(&weights, p, s);
    if (fold_norms) log.info("[mtp] delta-encoded norms detected; folding +1 at load\n", .{});

    var hq_stats: HeadQuantStats = .{};
    var m = MtpModel{
        .allocator = allocator,
        .s = s,
        .quant_bits = 0, // inferred from tensor geometry below
        .quant_group_size = 0,
        // fc via loadLinear (never loadTrunkLinear — the m5Nax profile
        // contract wants a bf16 fc): dense gets the pre-transpose, a
        // quantized one (Alis) loads verbatim.
        .fc = try loadLinear(&weights, allocator, K.k(&kb, p, "fc"), s),
        .pre_fc_norm_emb = try ownNorm(&weights, K.k(&kb, p, "pre_fc_norm_embedding.weight"), s, fold_norms),
        .pre_fc_norm_hidden = try ownNorm(&weights, K.k(&kb, p, "pre_fc_norm_hidden.weight"), s, fold_norms),
        // The 4 norms an oQ `mean<0.5 → +1` conversion can leave a full +1 too
        // low (their raw HF means sit above 0.5): fold or repair per oMLX's
        // norm_repair — the global delta-fold when it fired, else a reference
        // anchor from the backbone counterpart. pre_fc_norm_* + input_norm are
        // always converted correctly, so they stay on plain ownNorm.
        .final_norm = try ownHeadNormWithRepair(&weights, K.k(&kb, p, "norm.weight"), "model.norm.weight", s, fold_norms),
        .input_norm = try ownNorm(&weights, K.k(&kb, p, "layers.0.input_layernorm.weight"), s, fold_norms),
        .post_attn_norm = try ownHeadNormWithRepair(&weights, K.k(&kb, p, "layers.0.post_attention_layernorm.weight"), ".post_attention_layernorm.weight", s, fold_norms),
        .q_norm = try ownHeadNormWithRepair(&weights, K.k(&kb, p, "layers.0.self_attn.q_norm.weight"), ".self_attn.q_norm.weight", s, fold_norms),
        .k_norm = try ownHeadNormWithRepair(&weights, K.k(&kb, p, "layers.0.self_attn.k_norm.weight"), ".self_attn.k_norm.weight", s, fold_norms),
        .q = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.self_attn.q_proj"), s, &hq_stats),
        .k = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.self_attn.k_proj"), s, &hq_stats),
        .v = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.self_attn.v_proj"), s, &hq_stats),
        .o = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.self_attn.o_proj"), s, &hq_stats),
        .mlp = if (is_moe) blk: {
            // Router (`mlp.gate`) via loadLinear: a bf16 router gets
            // pre-transposed for the trunk's dense-matmul fallback, a
            // quantized one loads verbatim.
            const router = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.gate"), s);
            // Packed 3D expert tensors load raw (the trunk's gather paths own
            // the orientation); 2D shared/seg linears ride loadLinear so a
            // bf16 build gets the dense pre-transpose, exactly like the trunk.
            const sg = try loadMoeTriple(&weights, K.k(&kb, p, "layers.0.mlp.switch_mlp.gate_proj"));
            const su = try loadMoeTriple(&weights, K.k(&kb, p, "layers.0.mlp.switch_mlp.up_proj"));
            const sd = try loadMoeTriple(&weights, K.k(&kb, p, "layers.0.mlp.switch_mlp.down_proj"));
            const shg = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert.gate_proj"), s, &hq_stats);
            const shu = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert.up_proj"), s, &hq_stats);
            const shd = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert.down_proj"), s, &hq_stats);
            const seg = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert_gate"), s);
            break :blk .{ .moe = .{
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
        } else .{ .dense = .{
            .gate = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.gate_proj"), s, &hq_stats),
            .up = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.up_proj"), s, &hq_stats),
            .down = try loadTrunkLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.down_proj"), s, &hq_stats),
        } },
    };
    errdefer m.deinit();

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

    const router = try loadLinear(weights, allocator, K.k(&kb, p, "layer.mlp.router.gate"), s);
    const sg = try loadMoeTriple(weights, K.k(&kb, p, "layer.mlp.experts.gate_proj"));
    const su = try loadMoeTriple(weights, K.k(&kb, p, "layer.mlp.experts.up_proj"));
    const sd = try loadMoeTriple(weights, K.k(&kb, p, "layer.mlp.experts.down_proj"));
    const shg = try loadLinear(weights, allocator, K.k(&kb, p, "layer.mlp.shared_mlp.gate_proj"), s);
    const shu = try loadLinear(weights, allocator, K.k(&kb, p, "layer.mlp.shared_mlp.up_proj"), s);
    const shd = try loadLinear(weights, allocator, K.k(&kb, p, "layer.mlp.shared_mlp.down_proj"), s);

    var m = MtpModel{
        .allocator = allocator,
        .s = s,
        .quant_bits = 0,
        .quant_group_size = 0,
        .fc = .{ .w = .{ .ctx = null }, .s = .{ .ctx = null }, .b = .{ .ctx = null } },
        .eh_proj = try loadLinear(weights, allocator, K.k(&kb, p, "eh_proj"), s),
        .pre_fc_norm_emb = try ownWeight(weights, K.k(&kb, p, "enorm.weight")),
        .pre_fc_norm_hidden = try ownWeight(weights, K.k(&kb, p, "hnorm.weight")),
        .final_norm = try ownWeight(weights, K.k(&kb, p, "final_layernorm.weight")),
        .input_norm = try ownWeight(weights, K.k(&kb, p, "layer.input_layernorm.weight")),
        .post_attn_norm = try ownWeight(weights, K.k(&kb, p, "layer.post_attention_layernorm.weight")),
        .q_norm = try ownWeight(weights, K.k(&kb, p, "layer.self_attn.q_norm.weight")),
        .k_norm = try ownWeight(weights, K.k(&kb, p, "layer.self_attn.k_norm.weight")),
        .q = try loadLinear(weights, allocator, K.k(&kb, p, "layer.self_attn.q_proj"), s),
        .k = try loadLinear(weights, allocator, K.k(&kb, p, "layer.self_attn.k_proj"), s),
        .v = try loadLinear(weights, allocator, K.k(&kb, p, "layer.self_attn.v_proj"), s),
        .o = try loadLinear(weights, allocator, K.k(&kb, p, "layer.self_attn.o_proj"), s),
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
            .expert_bias = try ownWeight(weights, K.k(&kb, p, "layer.mlp.expert_bias")),
            .shared_ungated = true,
        } },
    };
    errdefer m.deinit();

    // Fallback quant globals from the q projection geometry (hidden pinned by
    // the enorm length); every matmul re-solves per weight anyway.
    {
        const en_shape = mlx.getShape(m.pre_fc_norm_emb);
        const hidden: u32 = if (en_shape.len == 1) @intCast(en_shape[0]) else 0;
        m.quant_bits = inferBits(&m.q, hidden) orelse 8;
        m.quant_group_size = inferGroupSize(&m.q, m.quant_bits) orelse 64;
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
    const x_shape = mlx.getShape(x);
    if (x_shape.len > 0 and x_shape[x_shape.len - 1] > 0) {
        const in_dim: u32 = @intCast(x_shape[x_shape.len - 1]);
        if (transformer_mod.affineParamsFromGeometry(lin.w, lin.s, in_dim)) |qp| {
            bits = qp.bits;
            group = qp.group_size;
        }
    }
    // Multi-row head forwards (the merged history+draft consume, prefill
    // history rebuilds) ride the same verify-width split-K kernel as the
    // trunk; ineligible shapes (seq 1 drafts, 5/6-bit MoE sidecars) fall
    // through to stock.
    if (try transformer_mod.verifyQmm(self.s, x, lin.w, lin.s, lin.b, bits, group)) |vy| {
        _ = mlx.mlx_array_free(out);
        return vy;
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
        "affine",
        self.s,
    ));
    return out;
}

/// Embed `[n]`-shaped int32 token ids through the TARGET's embedding table
/// → `[1, n, H]` bf16. Mirrors `Transformer.embedding` (quantized) with a
/// dense-bf16 fallback. No embed scaling — Qwen does not scale embeddings.
fn embedTargetTokens(
    target: *Transformer,
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
    const emb_qp = transformer_mod.computeQuantParams(&target.config, target.emb_w, target.emb_s, target.config.hidden_size);
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
    return out;
}

/// Project the MTP post-norm hidden through the lm_head. Draft steps go
/// through the low-bit draft-only head when one was built (verification
/// never routes here — trunk logits come from the trunk forward).
fn targetLmHead(self: *const MtpModel, target: *Transformer, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
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
    var out = mlx.mlx_array_new();
    if (target.lm_head_s.ctx == null) {
        // Dense bf16 lm_head is stored [vocab, hidden]; contract via lazy transpose.
        const axes = [_]c_int{ 1, 0 };
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose_axes(&wt, target.lm_head_w, &axes, 2, s));
        try mlx.check(mlx.mlx_matmul(&out, x, wt, s));
        return out;
    }
    // Per-WEIGHT quant params: mixed checkpoints override the head's width
    // (hy_v3 2-bit trunk ships an 8-bit lm_head — the global bits crashed the
    // whole process in mlx's shape check, live 2026-07-14). Non-affine trunks
    // keep the config fallback.
    const qp = headQuantParams(&target.config, target.lm_head_w, target.lm_head_s);
    try mlx.check(mlx.mlx_quantized_matmul(
        &out,
        x,
        target.lm_head_w,
        target.lm_head_s,
        target.lm_head_b,
        true,
        mlx.mlx_optional_int.some(@intCast(qp.group_size)),
        mlx.mlx_optional_int.some(@intCast(qp.bits)),
        qp.mode.cstr(),
        s,
    ));
    return out;
}

/// The trunk lm_head's TRUE quant params, via the same dtype-gated resolver
/// the trunk itself uses: uint8 scales → the config's non-affine mode (a raw
/// geometry solve mis-reads mxfp8 8-bit/gs32 as AFFINE 8-bit/gs32 — issue
/// #81, "Biases must be provided" crash on biasless heads); float scales →
/// exact per-geometry affine solve (the head's bits routinely differ from
/// the trunk global on mixed checkpoints — hy_v3 ships an 8-bit head over a
/// 2-bit trunk).
fn headQuantParams(config: *const model_mod.ModelConfig, w: mlx.mlx_array, sc: mlx.mlx_array) transformer_mod.QuantParams {
    return transformer_mod.computeQuantParams(config, w, sc, config.hidden_size);
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
fn fcConcat(self: *const MtpModel, target: *Transformer, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, seq_len: c_int) !mlx.mlx_array {
    const s = self.s;
    const eps = target.config.rms_norm_eps;
    const emb = try embedTargetTokens(target, id_arr, seq_len, s);
    defer _ = mlx.mlx_array_free(emb);
    const e_normed = try rmsNormFn(emb, self.pre_fc_norm_emb, eps, s);
    defer _ = mlx.mlx_array_free(e_normed);
    const h_normed = try rmsNormFn(hidden, self.pre_fc_norm_hidden, eps, s);
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

fn frontChain(self: *const MtpModel, target: *Transformer, id_arr: mlx.mlx_array, hidden: mlx.mlx_array) !FrontOut {
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
fn backChain(self: *const MtpModel, target: *Transformer, attn_out: mlx.mlx_array, gate: mlx.mlx_array, x: mlx.mlx_array, seq_len: c_int) !mlx.mlx_array {
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
    /// `[1, 1, vocab]` LAST-row logits, or `.ctx == null` when `want_logits`
    /// was false (multi-row calls never project the history rows — only the
    /// last row feeds the draft chain).
    logits: mlx.mlx_array,
    /// MTP post-norm hidden — the next depth's `hidden` input. `[1, 1, H]`
    /// (last row) when `want_logits`, the full `[1, L, H]` otherwise.
    hidden_next: mlx.mlx_array,
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
    target: *Transformer,
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
var mtp_kv_only_env: ?bool = null;

/// KV-only history rows ship ON (mlxfast-challenge lastHiddenWithKVOnlyHistory
/// class): a committed-history row's layer OUTPUT has no consumer — only its
/// K/V entries matter — so the q(+gate) projection, the attention, and the
/// whole post-attention half are dead work for every row but the last.
/// MLX_SERVE_MTP_KV_ONLY=0 restores the full-forward history path.
fn mtpKvOnlyEnabled() bool {
    if (mtp_kv_only_override) |v| return v;
    if (mtp_kv_only_env) |v| return v;
    const raw = std.c.getenv("MLX_SERVE_MTP_KV_ONLY");
    const enabled = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    mtp_kv_only_env = enabled;
    return enabled;
}

/// Append `hidden`/`id_arr` rows to the head's cache through the K/V-only
/// path: fc fusion + input_norm + K/V projections + k_norm + rope + cache
/// update. No query, no attention, no MLP — nothing downstream reads those
/// rows' outputs. Byte parity with the full path is NOT the bar (a different
/// GEMM M reorders reductions); the head only proposes, so the cost of any
/// near-tie flip is acceptance, which the live equivalence script gates.
fn appendKvOnly(
    self: *const MtpModel,
    target: *Transformer,
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
    if (needs_explicit_mrope) {
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
    target: *Transformer,
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
    if (needs_explicit_mrope) {
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
        if (try transformer_mod.fusedSdpa256Prefill(s, q_rope, kv_view.k, kv_view.v, attn_scale, 0)) |fused| {
            _ = mlx.mlx_array_free(attn_out);
            attn_out = fused;
            fused_done = true;
        }
    }
    if (!fused_done) {
        const mask_mode: [*:0]const u8 = if (seq_len > 1) "causal" else "";
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&attn_out, q_rope, kv_view.k, kv_view.v, attn_scale, mask_mode, none_mask, .{ .ctx = null }, s));
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

/// Append committed-history entries: pair `hidden[:, i, :]` with
/// `token_ids[i]` for each i. One batched MTP-layer forward, no logits.
pub fn appendHistory(
    self: *const MtpModel,
    target: *Transformer,
    cache: *KVCache,
    token_ids: []const u32,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
) !void {
    return appendHistoryWithMrope(self, target, cache, token_ids, hidden, rope_offset, null);
}

pub fn appendHistoryWithMrope(
    self: *const MtpModel,
    target: *Transformer,
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
    target: *Transformer,
    cache: *KVCache,
    prev_token_arr: mlx.mlx_array,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
) !StepOut {
    return stepArrWithMrope(self, target, cache, prev_token_arr, hidden, rope_offset, null);
}

pub fn stepArrWithMrope(
    self: *const MtpModel,
    target: *Transformer,
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
const TOP32_ORDINAL_HEADER =
    \\inline uint msv_top32_ordinal(float v) {
    \\    if (isnan(v))  { return 0xFFFFFFFFu; }
    \\    if (v == 0.0f) { return 0x80000000u; }
    \\    uint u = as_type<uint>(v);
    \\    return (u & 0x80000000u) ? (~u) : (u | 0x80000000u);
    \\}
;

// Stage 1: TILES threadgroups partition [0, RC); each emits its local top 32
// as (ordinal, index) pairs — TILES * 32 candidates. Selection is 32 rounds
// of simd_max over per-thread slots with a `taken` bitmask (hence the
// PER_THREAD <= 32 static_assert), then one simdgroup reduces the
// threadgroup's per-simd lists the same way.
const TOP32_PARTIAL_SRC =
    \\constexpr uint REAL_COUNT = (uint)RC;
    \\constexpr uint TG_SIZE    = 256;
    \\constexpr uint STRIDE     = 64u * 256u;
    \\constexpr uint PER_THREAD = (REAL_COUNT + STRIDE - 1u) / STRIDE;
    \\constexpr uint TOPK       = 32;
    \\constexpr uint SIMD_SIZE  = 32;
    \\constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
    \\constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;
    \\static_assert(PER_THREAD <= 32, "PER_THREAD exceeds taken-bitmask width");
    \\static_assert(PB <= 32, "PB exceeds tk2-bitmask width");
    \\
    \\uint tile = threadgroup_position_in_grid.x;
    \\uint tid  = thread_position_in_threadgroup.x;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg   = simdgroup_index_in_threadgroup;
    \\
    \\uint ord[PER_THREAD];
    \\uint idx[PER_THREAD];
    \\for (uint t = 0; t < PER_THREAD; ++t) { ord[t] = 0u; idx[t] = 0u; }
    \\uint n = 0;
    \\for (uint i = tile * TG_SIZE + tid; i < REAL_COUNT; i += STRIDE) {
    \\    ord[n] = msv_top32_ordinal(float(logits[i]));
    \\    idx[n] = i;
    \\    n++;
    \\}
    \\
    \\threadgroup uint sc_ord[NSIMD * TOPK];
    \\threadgroup uint sc_idx[NSIMD * TOPK];
    \\
    \\uint taken = 0u;
    \\for (uint r = 0; r < TOPK; ++r) {
    \\    uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
    \\    for (uint t = 0; t < PER_THREAD; ++t) {
    \\        if ((taken & (1u << t)) != 0u) { continue; }
    \\        if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {
    \\            bo = ord[t]; bi = idx[t]; bs = t;
    \\        }
    \\    }
    \\    uint mo = simd_max(bo);
    \\    uint mi = simd_max((bo == mo) ? bi : 0u);
    \\    if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
    \\        taken |= (1u << bs);
    \\    }
    \\    if (lane == 0) {
    \\        sc_ord[sg * TOPK + r] = mo;
    \\        sc_idx[sg * TOPK + r] = mi;
    \\    }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\if (sg == 0) {
    \\    uint o2[PB];
    \\    uint i2[PB];
    \\    for (uint t = 0; t < PB; ++t) {
    \\        uint p = t * SIMD_SIZE + lane;
    \\        o2[t] = sc_ord[p];
    \\        i2[t] = sc_idx[p];
    \\    }
    \\    uint tk2 = 0u;
    \\    for (uint r = 0; r < TOPK; ++r) {
    \\        uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
    \\        for (uint t = 0; t < PB; ++t) {
    \\            if ((tk2 & (1u << t)) != 0u) { continue; }
    \\            if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {
    \\                bo = o2[t]; bi = i2[t]; bs = t;
    \\            }
    \\        }
    \\        uint mo = simd_max(bo);
    \\        uint mi = simd_max((bo == mo) ? bi : 0u);
    \\        if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
    \\            tk2 |= (1u << bs);
    \\        }
    \\        if (lane == 0) {
    \\            cand_ord[tile * TOPK + r] = mo;
    \\            cand_idx[tile * TOPK + r] = mi;
    \\        }
    \\    }
    \\}
;

// Stage 2: one threadgroup reduces the TILES*32 candidates to the final 32
// ids (written ascending by (value, index) — consumers are order-blind).
const TOP32_FINAL_SRC =
    \\constexpr uint TG_SIZE    = 256;
    \\constexpr uint PER_THREAD = 8;
    \\constexpr uint TOPK       = 32;
    \\constexpr uint SIMD_SIZE  = 32;
    \\constexpr uint NSIMD      = TG_SIZE / SIMD_SIZE;
    \\constexpr uint PB         = (NSIMD * TOPK) / SIMD_SIZE;
    \\
    \\uint tid  = thread_position_in_threadgroup.x;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg   = simdgroup_index_in_threadgroup;
    \\
    \\uint ord[PER_THREAD];
    \\uint idx[PER_THREAD];
    \\for (uint t = 0; t < PER_THREAD; ++t) {
    \\    uint p = t * TG_SIZE + tid;
    \\    ord[t] = cand_ord[p];
    \\    idx[t] = cand_idx[p];
    \\}
    \\
    \\threadgroup uint sc_ord[NSIMD * TOPK];
    \\threadgroup uint sc_idx[NSIMD * TOPK];
    \\
    \\uint taken = 0u;
    \\for (uint r = 0; r < TOPK; ++r) {
    \\    uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
    \\    for (uint t = 0; t < PER_THREAD; ++t) {
    \\        if ((taken & (1u << t)) != 0u) { continue; }
    \\        if (ord[t] > bo || (ord[t] == bo && idx[t] > bi)) {
    \\            bo = ord[t]; bi = idx[t]; bs = t;
    \\        }
    \\    }
    \\    uint mo = simd_max(bo);
    \\    uint mi = simd_max((bo == mo) ? bi : 0u);
    \\    if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
    \\        taken |= (1u << bs);
    \\    }
    \\    if (lane == 0) {
    \\        sc_ord[sg * TOPK + r] = mo;
    \\        sc_idx[sg * TOPK + r] = mi;
    \\    }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\if (sg == 0) {
    \\    uint o2[PB];
    \\    uint i2[PB];
    \\    for (uint t = 0; t < PB; ++t) {
    \\        uint p = t * SIMD_SIZE + lane;
    \\        o2[t] = sc_ord[p];
    \\        i2[t] = sc_idx[p];
    \\    }
    \\    uint tk2 = 0u;
    \\    for (uint r = 0; r < TOPK; ++r) {
    \\        uint bo = 0u, bi = 0u, bs = 0xFFFFFFFFu;
    \\        for (uint t = 0; t < PB; ++t) {
    \\            if ((tk2 & (1u << t)) != 0u) { continue; }
    \\            if (o2[t] > bo || (o2[t] == bo && i2[t] > bi)) {
    \\                bo = o2[t]; bi = i2[t]; bs = t;
    \\            }
    \\        }
    \\        uint mo = simd_max(bo);
    \\        uint mi = simd_max((bo == mo) ? bi : 0u);
    \\        if (bs != 0xFFFFFFFFu && bo == mo && bi == mi) {
    \\            tk2 |= (1u << bs);
    \\        }
    \\        if (lane == 0) { token_ids[TOPK - 1u - r] = mi; }
    \\    }
    \\}
;

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

/// Exact top-32 ids of `row` ([rows], any float dtype) as a lazy [32] uint32
/// array, ascending by (value, index) — ties break toward the higher index,
/// NaN above every number. `rows` rides as a template int (stable per model,
/// so MLX caches one specialization per vocab width).
pub fn draftTop32(s: mlx.mlx_stream, row: mlx.mlx_array, rows: c_int) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return error.MetalKernelNeedsGpuStream;
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
    // every `--mtp` boot (live 2026-07-31). The sidecar claim rides the same
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
    // ~86% to 33% (live 2026-08-19).
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

test "mtp: headQuantParams never mis-resolves a non-affine lm_head as affine (issue #81)" {
    // An mxfp8 8-bit/gs32 lm_head's GEOMETRY coincidentally solves as a valid
    // affine 8-bit/gs32 (w [V, H/4] u32, scales [V, H/32]) — but the scales
    // are fp8-encoded uint8 and no biases tensor exists, so a mode="affine"
    // matmul/dequantize throws "Biases must be provided for affine
    // quantization" and kills the first MTP forward. The scales-dtype gate
    // (uint8 → the config's non-affine mode) must win over the geometry
    // shortcut, exactly as computeQuantParams resolves the trunk.
    const s = mlx.gpuStream();
    const H = 512;
    const V = 8;

    const mk = struct {
        fn arr(shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, st));
            return a;
        }
    };

    // mxfp8 head: w [V, H*8/32] u32, scales [V, H/32] u8.
    const w8 = try mk.arr(&.{ V, H * 8 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w8);
    const s8 = try mk.arr(&.{ V, H / 32 }, .uint8, s);
    defer _ = mlx.mlx_array_free(s8);

    var cfg = model_mod.ModelConfig{};
    cfg.hidden_size = H;
    cfg.quant_bits = 8;
    cfg.quant_group_size = 32;
    cfg.quant_mode = .mxfp8;

    const qp8 = headQuantParams(&cfg, w8, s8);
    try testing.expectEqual(model_mod.QuantMode.mxfp8, qp8.mode);
    try testing.expectEqual(@as(u32, 8), qp8.bits);
    try testing.expectEqual(@as(u32, 32), qp8.group_size);

    // mxfp4 (same class, 4-bit/gs32 — also a false-positive affine geometry).
    const w4 = try mk.arr(&.{ V, H * 4 / 32 }, .uint32, s);
    defer _ = mlx.mlx_array_free(w4);
    cfg.quant_bits = 4;
    cfg.quant_mode = .mxfp4;
    const qp4 = headQuantParams(&cfg, w4, s8);
    try testing.expectEqual(model_mod.QuantMode.mxfp4, qp4.mode);

    // Characterization (green before AND after): the mixed-AFFINE shape this
    // function exists for — hy_v3's 8-bit/gs32 head (bf16 scales) over a
    // 2-bit/gs64 trunk still resolves per-geometry, never per-config.
    const sb = try mk.arr(&.{ V, H / 32 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sb);
    cfg.quant_bits = 2;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    const qpa = headQuantParams(&cfg, w8, sb);
    try testing.expectEqual(model_mod.QuantMode.affine, qpa.mode);
    try testing.expectEqual(@as(u32, 8), qpa.bits);
    try testing.expectEqual(@as(u32, 32), qpa.group_size);
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

test "mtp: multi-row forward projects the LAST row only and equals appendHistory + stepArr" {
    // The deferred-history round shape (Generator.nextMtp) folds the old
    // appendHistory head forward into the next round's first draft step: ONE
    // (n+1)-row forward over [committed..., t1] must append the same cache
    // entries AND produce the same last-row logits/hidden as the two-call
    // sequence appendHistory([committed], hist_hidden) + stepArr(t1, h_prev).
    // Logits must be [1, 1, V]: projecting every row through the vocab head
    // is pure waste, and the caller (draft chain) only consumes the last row.
    const io = testing.io;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // Pin the KV-only history path ON regardless of the env, so the merged
    // and appendHistory arms below exercise it; the last arm flips it OFF
    // for the full-path cross-check.
    mtp_kv_only_override = true;
    defer mtp_kv_only_override = null;

    // ── synthetic DENSE sidecar (random bf16; zeros would make every rms-norm
    // output zero and the equivalence trivially true) ──
    var prng = std.Random.DefaultPrng.init(7);
    const save_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(save_map);
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(allocator);
    }
    const putRand = struct {
        fn f(map: mlx.mlx_map_string_to_array, list: *std.ArrayList(mlx.mlx_array), alloc: std.mem.Allocator, rng: *std.Random.DefaultPrng, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            var n: usize = 1;
            for (shape) |d| n *= @intCast(d);
            const buf = try alloc.alloc(f32, n);
            defer alloc.free(buf);
            for (buf) |*x| x.* = rng.random().floatNorm(f32) * 0.5;
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&a, f32_arr, .bfloat16, st));
            try mlx.check(mlx.mlx_array_eval(a));
            _ = mlx.mlx_map_string_to_array_insert(map, key, a);
            try list.append(alloc, a);
            return a;
        }
    }.f;

    // hidden 8, head_dim 4, 2 q heads (x2 for the q/gate split), 2 kv heads,
    // mlp inter 16, vocab 16.
    // Disk orientation is torch [out, in]: fc maps concat(2H) -> H.
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.fc.weight", &.{ 8, 16 }, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.pre_fc_norm_embedding.weight", &.{8}, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.pre_fc_norm_hidden.weight", &.{8}, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.norm.weight", &.{8}, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.input_layernorm.weight", &.{8}, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.post_attention_layernorm.weight", &.{8}, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.q_norm.weight", &.{4}, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.k_norm.weight", &.{4}, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.q_proj.weight", &.{ 16, 8 }, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.k_proj.weight", &.{ 8, 8 }, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.v_proj.weight", &.{ 8, 8 }, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.self_attn.o_proj.weight", &.{ 8, 8 }, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.mlp.gate_proj.weight", &.{ 16, 8 }, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.mlp.up_proj.weight", &.{ 16, 8 }, s);
    _ = try putRand(save_map, &owned, allocator, &prng, "mtp.layers.0.mlp.down_proj.weight", &.{ 8, 16 }, s);

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

    // ── toy target: only the fields forward() reads (config scalars, dense
    // bf16 embed table, dense bf16 lm_head) ──
    var emb_prng = std.Random.DefaultPrng.init(11);
    const mk2d = struct {
        fn f(alloc: std.mem.Allocator, rng: *std.Random.DefaultPrng, rows: usize, cols: usize, st: mlx.mlx_stream) !mlx.mlx_array {
            const buf = try alloc.alloc(f32, rows * cols);
            defer alloc.free(buf);
            for (buf) |*x| x.* = rng.random().floatNorm(f32) * 0.5;
            const shape = [_]c_int{ @intCast(rows), @intCast(cols) };
            const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 2, .float32);
            defer _ = mlx.mlx_array_free(f32_arr);
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&a, f32_arr, .bfloat16, st));
            try mlx.check(mlx.mlx_array_eval(a));
            return a;
        }
    }.f;
    const emb_w = try mk2d(allocator, &emb_prng, 16, 8, s);
    defer _ = mlx.mlx_array_free(emb_w);
    const lm_w = try mk2d(allocator, &emb_prng, 16, 8, s);
    defer _ = mlx.mlx_array_free(lm_w);

    var xfm: Transformer = undefined;
    xfm.allocator = allocator;
    xfm.s = s;
    xfm.config = .{};
    xfm.config.hidden_size = 8;
    xfm.config.num_attention_heads = 2;
    xfm.config.num_key_value_heads = 2;
    xfm.config.head_dim = 4;
    xfm.config.query_pre_attn_scalar = 4;
    xfm.config.partial_rotary_factor = 0.5;
    xfm.config.attn_output_gate = true;
    xfm.emb_w = emb_w;
    xfm.emb_s = .{ .ctx = null };
    xfm.emb_b = .{ .ctx = null };
    xfm.lm_head_w = lm_w;
    xfm.lm_head_s = .{ .ctx = null };
    xfm.lm_head_b = .{ .ctx = null };

    // ── shared inputs: 3 hidden rows, tokens [5, 7] committed + t1 = 9 ──
    var hid_prng = std.Random.DefaultPrng.init(23);
    const hid_buf = try allocator.alloc(f32, 3 * 8);
    defer allocator.free(hid_buf);
    for (hid_buf) |*x| x.* = hid_prng.random().floatNorm(f32) * 0.5;
    const hid_shape = [_]c_int{ 1, 3, 8 };
    const hid_f32 = mlx.mlx_array_new_data(hid_buf.ptr, &hid_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hid_f32);
    var hidden3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden3);
    try mlx.check(mlx.mlx_astype(&hidden3, hid_f32, .bfloat16, s));

    const strides = [_]c_int{ 1, 1, 1 };
    var hid01 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hid01);
    try mlx.check(mlx.mlx_slice(&hid01, hidden3, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, 2, 8 }, 3, &strides, 3, s));
    var hid2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hid2);
    try mlx.check(mlx.mlx_slice(&hid2, hidden3, &[_]c_int{ 0, 2, 0 }, 3, &[_]c_int{ 1, 3, 8 }, 3, &strides, 3, s));

    // ── reference: appendHistory([5,7]) then stepArr(9) ──
    var cache_a = try m.makeCache(allocator);
    defer cache_a.deinit();
    try appendHistory(&m, &xfm, &cache_a, &[_]u32{ 5, 7 }, hid01, 0);
    const t9 = [_]i32{9};
    const t9_shape = [_]c_int{1};
    const t9_arr = mlx.mlx_array_new_data(&t9, &t9_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(t9_arr);
    const ref = try stepArr(&m, &xfm, &cache_a, t9_arr, hid2, 2);
    defer {
        _ = mlx.mlx_array_free(ref.logits);
        _ = mlx.mlx_array_free(ref.hidden_next);
    }

    // ── merged: one 3-row forward over [5, 7, 9] ──
    var cache_b = try m.makeCache(allocator);
    defer cache_b.deinit();
    const ids3 = [_]i32{ 5, 7, 9 };
    const ids3_shape = [_]c_int{3};
    const ids3_arr = mlx.mlx_array_new_data(&ids3, &ids3_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(ids3_arr);
    const merged = try forward(&m, &xfm, &cache_b, ids3_arr, hidden3, 0, true);
    defer {
        _ = mlx.mlx_array_free(merged.logits);
        _ = mlx.mlx_array_free(merged.hidden_next);
    }

    // Same cache length; logits/hidden are LAST-row-only.
    try testing.expectEqual(cache_a.step, cache_b.step);
    const ml_shape = mlx.getShape(merged.logits);
    try testing.expectEqual(@as(c_int, 1), ml_shape[1]);
    try testing.expectEqual(@as(c_int, 16), ml_shape[2]);
    const mh_shape = mlx.getShape(merged.hidden_next);
    try testing.expectEqual(@as(c_int, 1), mh_shape[1]);

    // Value equivalence vs the two-call reference (bf16 reduction-order
    // tolerance at toy scale).
    const close = struct {
        fn f(a: mlx.mlx_array, b: mlx.mlx_array, n: usize, st: mlx.mlx_stream) !void {
            var af = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(af);
            var bf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bf);
            try mlx.check(mlx.mlx_astype(&af, a, .float32, st));
            try mlx.check(mlx.mlx_astype(&bf, b, .float32, st));
            try mlx.check(mlx.mlx_array_eval(af));
            try mlx.check(mlx.mlx_array_eval(bf));
            const ad = mlx.mlx_array_data_float32(af).?;
            const bd = mlx.mlx_array_data_float32(bf).?;
            for (0..n) |i| {
                const denom = @max(1.0, @max(@abs(ad[i]), @abs(bd[i])));
                if (@abs(ad[i] - bd[i]) / denom > 0.05) {
                    std.debug.print("mismatch at {d}: {d} vs {d}\n", .{ i, ad[i], bd[i] });
                    return error.TestExpectedApproxEq;
                }
            }
        }
    }.f;
    try close(merged.logits, ref.logits, 16, s);
    try close(merged.hidden_next, ref.hidden_next, 8, s);

    // A sequential three-axis table must be equivalent to ordinary RoPE.
    // Keep only two explicit positions so the final row also exercises the
    // prompt-table → generated-text (`absolute + delta`) boundary.
    var cache_c = try m.makeCache(allocator);
    defer cache_c.deinit();
    const sequential_pos = [_]i32{
        0, 1,
        0, 1,
        0, 1,
    };
    const positioned = try forwardWithMrope(
        &m,
        &xfm,
        &cache_c,
        ids3_arr,
        hidden3,
        0,
        true,
        .{
            .pos = &sequential_pos,
            .total = 2,
            .delta = 0,
        },
    );
    defer {
        _ = mlx.mlx_array_free(positioned.logits);
        _ = mlx.mlx_array_free(positioned.hidden_next);
    }
    try testing.expectEqual(cache_b.step, cache_c.step);
    try close(positioned.logits, merged.logits, 16, s);
    try close(positioned.hidden_next, merged.hidden_next, 8, s);

    // ── KV-only vs FULL history path: the committed rows' layer outputs are
    // dead, so the KV-only append must leave the same cache length and a
    // last-row result within reduction-order tolerance of the full forward
    // (a different GEMM M reorders reductions — byte parity is not the bar).
    mtp_kv_only_override = false;
    var cache_d = try m.makeCache(allocator);
    defer cache_d.deinit();
    const full = try forward(&m, &xfm, &cache_d, ids3_arr, hidden3, 0, true);
    defer {
        _ = mlx.mlx_array_free(full.logits);
        _ = mlx.mlx_array_free(full.hidden_next);
    }
    try testing.expectEqual(cache_b.step, cache_d.step);
    try close(full.logits, merged.logits, 16, s);
    try close(full.hidden_next, merged.hidden_next, 8, s);

    // Same cross-check through the mrope table (appendKvOnly's explicit-table
    // branch vs the full path's).
    var cache_e = try m.makeCache(allocator);
    defer cache_e.deinit();
    const full_pos = try forwardWithMrope(&m, &xfm, &cache_e, ids3_arr, hidden3, 0, true, .{
        .pos = &sequential_pos,
        .total = 2,
        .delta = 0,
    });
    defer {
        _ = mlx.mlx_array_free(full_pos.logits);
        _ = mlx.mlx_array_free(full_pos.hidden_next);
    }
    try testing.expectEqual(cache_c.step, cache_e.step);
    try close(full_pos.logits, positioned.logits, 16, s);
    try close(full_pos.hidden_next, positioned.hidden_next, 8, s);
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

test "mtp: dense bf16 head trunk is requantized at load (4b/g64); indivisible widths stay dense" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var tmp_dir = std.testing.tmpDir(.{});
    defer tmp_dir.cleanup();
    var path_buf: [512]u8 = undefined;
    const root_len = try tmp_dir.dir.realPath(io, &path_buf);
    const dir_path = path_buf[0..root_len];

    // Dense bf16 sidecar with a 64-divisible hidden (64): q/k/v/o + gate/up
    // quantize; down's contraction dim is 96 (not 64-divisible) — per-weight
    // skip, stays dense pre-transposed. fc stays bf16 by contract (the m5Nax
    // cost profile demands it).
    const st_path = try std.fmt.allocPrintSentinel(allocator, "{s}/mtp.safetensors", .{dir_path}, 0);
    defer allocator.free(st_path);
    {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        const H = struct {
            fn put(m2: mlx.mlx_map_string_to_array, key: [*:0]const u8, shape: []const c_int, st: mlx.mlx_stream) !void {
                var total: usize = 1;
                for (shape) |d| total *= @intCast(d);
                const data = try std.testing.allocator.alloc(f32, total);
                defer std.testing.allocator.free(data);
                for (data, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.1 - 0.3;
                const f32_arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
                defer _ = mlx.mlx_array_free(f32_arr);
                var bf = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(bf);
                try mlx.check(mlx.mlx_astype(&bf, f32_arr, .bfloat16, st));
                try mlx.check(mlx.mlx_array_eval(bf));
                _ = mlx.mlx_map_string_to_array_insert(m2, key, bf);
            }
        };
        try H.put(map, "mtp.fc.weight", &.{ 64, 128 }, s);
        try H.put(map, "mtp.pre_fc_norm_embedding.weight", &.{64}, s);
        try H.put(map, "mtp.pre_fc_norm_hidden.weight", &.{64}, s);
        try H.put(map, "mtp.norm.weight", &.{64}, s);
        try H.put(map, "mtp.layers.0.input_layernorm.weight", &.{64}, s);
        try H.put(map, "mtp.layers.0.post_attention_layernorm.weight", &.{64}, s);
        try H.put(map, "mtp.layers.0.self_attn.q_norm.weight", &.{32}, s);
        try H.put(map, "mtp.layers.0.self_attn.k_norm.weight", &.{32}, s);
        try H.put(map, "mtp.layers.0.self_attn.q_proj.weight", &.{ 128, 64 }, s);
        try H.put(map, "mtp.layers.0.self_attn.k_proj.weight", &.{ 32, 64 }, s);
        try H.put(map, "mtp.layers.0.self_attn.v_proj.weight", &.{ 32, 64 }, s);
        try H.put(map, "mtp.layers.0.self_attn.o_proj.weight", &.{ 64, 64 }, s);
        try H.put(map, "mtp.layers.0.mlp.gate_proj.weight", &.{ 96, 64 }, s);
        try H.put(map, "mtp.layers.0.mlp.up_proj.weight", &.{ 96, 64 }, s);
        try H.put(map, "mtp.layers.0.mlp.down_proj.weight", &.{ 64, 96 }, s);
        try mlx.check(mlx.mlx_save_safetensors(st_path.ptr, map, meta));
    }

    {
        var m = try loadMtp(io, allocator, s, dir_path);
        defer m.deinit();

        // q packed to 4-bit g64: uint32 codes, geometry re-solves.
        try testing.expect(m.q.s.ctx != null);
        try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(m.q.w));
        const qp = transformer_mod.affineParamsFromGeometry(m.q.w, m.q.s, 64) orelse return error.NoGeometry;
        try testing.expectEqual(@as(u32, 4), qp.bits);
        try testing.expectEqual(@as(u32, 64), qp.group_size);
        try testing.expect(m.mlp == .dense);
        try testing.expect(m.mlp.dense.gate.s.ctx != null);
        // down: contraction 96 not 64-divisible — stays dense pre-transposed.
        try testing.expect(m.mlp.dense.down.s.ctx == null);
        const down_shape = mlx.getShape(m.mlp.dense.down.w);
        try testing.expectEqual(@as(c_int, 96), down_shape[0]);
        // fc bf16 by contract.
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(m.fc.w));
    }

    // Lever off (0): everything stays dense.
    head_quant_override = 0;
    defer head_quant_override = null;
    {
        var m = try loadMtp(io, allocator, s, dir_path);
        defer m.deinit();
        try testing.expect(m.q.s.ctx == null);
        try testing.expect(m.mlp.dense.gate.s.ctx == null);
    }
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
