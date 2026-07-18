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
//! re-appends from true verify hiddens). RoPE offsets are cache-relative
//! ("cache" position mode), so a history that starts mid-conversation (KV
//! prefix reuse) is still self-consistent.
//!
//! Everything MTP-specific lives in this file plus `Generator.nextMtp`
//! (src/generate.zig); deleting the feature is removing those two.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const transformer_mod = @import("transformer.zig");
const log = @import("log.zig");

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

/// Exact full-round cost surfaces known to the adaptive MTP controller.
/// Selection is based on runtime tensor geometry, never a model/repository
/// name. `generic` retains the conservative M1-M4 surface and auto cap.
pub const MtpCostProfile = enum {
    generic,
    g17_nax_q8_gs32,
    g17_nax_q4_gs32,
};

fn m5NaxCostProfileForQuant(bits: u32, group_size: u32) MtpCostProfile {
    if (group_size != 32) return .generic;
    return switch (bits) {
        8 => .g17_nax_q8_gs32,
        4 => .g17_nax_q4_gs32,
        else => .generic,
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
const QLinear = struct {
    w: mlx.mlx_array,
    s: mlx.mlx_array,
    b: mlx.mlx_array,

    fn deinit(self: *QLinear) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.s);
        _ = mlx.mlx_array_free(self.b);
    }
};

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

    fc_w_t: mlx.mlx_array, // [2H, H] bf16, pre-transposed (Qwen heads; empty for Hy3)
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
        if (self.draft_head) |*dh| dh.deinit();
        if (self.eh_proj) |*ep| ep.deinit();
        _ = mlx.mlx_array_free(self.fc_w_t);
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
    /// target. Both G17 profiles require the successfully built 3-bit/gs-64
    /// draft-only lm_head and a homogeneous dense Qwen3.6-27B sidecar; their
    /// q8/gs-32 and q4/gs-32 draft costs are calibrated independently.
    /// Compatible but off-profile geometry remains correct under `generic`.
    pub fn m5NaxCostProfile(self: *const MtpModel, target: *const Transformer) MtpCostProfile {
        if (!target.mtpNaxProfileEnabled()) return .generic;
        if (self.eh_proj != null) return .generic;
        const profile = m5NaxCostProfileForQuant(self.quant_bits, self.quant_group_size);
        const sidecar_bits: u32 = switch (profile) {
            .g17_nax_q8_gs32 => 8,
            .g17_nax_q4_gs32 => 4,
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

        const fc_shape = mlx.getShape(self.fc_w_t);
        if (mlx.mlx_array_dtype(self.fc_w_t) != .bfloat16 or
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
                        .group_size = 32,
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
            return;
        }
        if (!cfg.attn_output_gate) return error.UnsupportedMtpArch;
        const fc_shape = mlx.getShape(self.fc_w_t);
        if (fc_shape.len != 2 or
            fc_shape[0] != @as(c_int, @intCast(cfg.hidden_size * 2)) or
            fc_shape[1] != @as(c_int, @intCast(cfg.hidden_size)))
            return error.MtpTargetMismatch;

        self.buildDraftHead(target) catch |err| {
            log.warn("[mtp] draft lm_head build failed ({s}) — drafts use the trunk head\n", .{@errorName(err)});
        };
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
        const head_qp = headQuantParams(target);
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

/// Relative path (one of `sidecar_rel_paths`) of the first present, non-empty
/// sidecar file under `dir`, or null when the model ships no MTP head.
pub fn resolveMtpSidecarInDir(io: std.Io, dir: std.Io.Dir) ?[]const u8 {
    for (&sidecar_rel_paths) |rel| {
        const st = dir.statFile(io, rel, .{}) catch continue;
        if (st.size > 0) return rel;
    }
    return null;
}

/// True when `model_dir` carries an MTP sidecar file we know how to load.
/// `model_dir` is absolute (same contract as `model.parseConfig`).
pub fn hasMtpSidecar(io: std.Io, model_dir: []const u8) bool {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return false;
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    return resolveMtpSidecarInDir(io, dir) != null;
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
    return max_neg > 0.05;
}

/// Own an RMSNorm weight, folding `+1` when the head stores delta-encoded norms.
fn ownNorm(w: *const Weights, key: []const u8, s: mlx.mlx_stream, fold: bool) !mlx.mlx_array {
    const owned = try ownWeight(w, key);
    if (!fold) return owned;
    defer _ = mlx.mlx_array_free(owned);
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

/// Requantize a row-quantized affine weight `(w, scales, biases)` from
/// `(from_gs, from_bits)` to `(to_gs, to_bits)`, chunk-wise over rows so the
/// dequantized bf16 transient stays bounded (~chunk_rows × in_features × 2 B
/// instead of the whole matrix — a 248K×5120 lm_head would otherwise
/// materialize 2.5 GB). Rows quantize independently in MLX's affine packing,
/// so per-chunk triples concatenate along axis 0 into a valid whole.
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
        {
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
    const expanded = packed_cols * (32 / bits);
    if (expanded % scale_cols != 0) return null;
    return expanded / scale_cols;
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
    const bits = (32 * packed_cols) / hidden;
    return switch (bits) {
        2, 4, 8 => bits,
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

/// Load the MTP head from the model's sidecar file (any `sidecar_rel_paths`
/// layout — native `mtp/weights.safetensors` and other compatible ones).
pub fn loadMtp(
    io: std.Io,
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
    model_dir: []const u8,
) !MtpModel {
    const rel = blk: {
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        break :blk resolveMtpSidecarInDir(io, dir) orelse return error.MissingMtpWeight;
    };
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sidecar_path = try std.fmt.bufPrint(&path_buf, "{s}/{s}", .{ model_dir, rel });
    var weights = try model_mod.loadWeightsSingleFile(allocator, sidecar_path);
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

    var m = MtpModel{
        .allocator = allocator,
        .s = s,
        .quant_bits = 0, // inferred from tensor geometry below
        .quant_group_size = 0,
        .fc_w_t = try ownAndTranspose2D(&weights, K.k(&kb, p, "fc.weight"), s),
        .pre_fc_norm_emb = try ownNorm(&weights, K.k(&kb, p, "pre_fc_norm_embedding.weight"), s, fold_norms),
        .pre_fc_norm_hidden = try ownNorm(&weights, K.k(&kb, p, "pre_fc_norm_hidden.weight"), s, fold_norms),
        .final_norm = try ownNorm(&weights, K.k(&kb, p, "norm.weight"), s, fold_norms),
        .input_norm = try ownNorm(&weights, K.k(&kb, p, "layers.0.input_layernorm.weight"), s, fold_norms),
        .post_attn_norm = try ownNorm(&weights, K.k(&kb, p, "layers.0.post_attention_layernorm.weight"), s, fold_norms),
        .q_norm = try ownNorm(&weights, K.k(&kb, p, "layers.0.self_attn.q_norm.weight"), s, fold_norms),
        .k_norm = try ownNorm(&weights, K.k(&kb, p, "layers.0.self_attn.k_norm.weight"), s, fold_norms),
        .q = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.self_attn.q_proj"), s),
        .k = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.self_attn.k_proj"), s),
        .v = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.self_attn.v_proj"), s),
        .o = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.self_attn.o_proj"), s),
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
            const shg = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert.gate_proj"), s);
            const shu = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert.up_proj"), s);
            const shd = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.shared_expert.down_proj"), s);
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
            .gate = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.gate_proj"), s),
            .up = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.up_proj"), s),
            .down = try loadLinear(&weights, allocator, K.k(&kb, p, "layers.0.mlp.down_proj"), s),
        } },
    };
    errdefer m.deinit();

    // Sidecars carry no quant metadata — infer bits from packed-column
    // geometry against the hidden size (exact: the bf16 fc weight pins
    // hidden), then group size from the scales shape. These are FALLBACK
    // globals: qLinearFwd re-solves per weight/call via
    // affineParamsFromGeometry, since sidecars mix bits AND group sizes
    // (the 35B-A3B head: q/k 5-bit gs-128, v 6-bit gs-128, o 4-bit gs-64).
    {
        const fc_shape = mlx.getShape(m.fc_w_t); // [2H, H] (pre-transposed)
        const hidden: u32 = if (fc_shape.len == 2) @intCast(fc_shape[1]) else 0;
        m.quant_bits = inferBits(&m.q, hidden) orelse 4;
        m.quant_group_size = inferGroupSize(&m.q, m.quant_bits) orelse 64;
    }

    // Materialize all weights now so first-token latency doesn't pay for it.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        const base = [_]mlx.mlx_array{
            m.fc_w_t,     m.pre_fc_norm_emb, m.pre_fc_norm_hidden, m.final_norm,
            m.input_norm, m.post_attn_norm,  m.q_norm,             m.k_norm,
            m.q.w,        m.k.w,             m.v.w,                m.o.w,
        };
        for (base) |a| _ = mlx.mlx_vector_array_append_value(eval_vec, a);
        switch (m.mlp) {
            .dense => |*d| {
                _ = mlx.mlx_vector_array_append_value(eval_vec, d.gate.w);
                _ = mlx.mlx_vector_array_append_value(eval_vec, d.up.w);
                _ = mlx.mlx_vector_array_append_value(eval_vec, d.down.w);
            },
            .moe => |*mw| {
                const moe_ws = [_]mlx.mlx_array{
                    mw.router_w, mw.switch_gate_w, mw.switch_up_w, mw.switch_down_w,
                    mw.shared_gate_w, mw.shared_up_w, mw.shared_down_w,
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
        .fc_w_t = .{ .ctx = null },
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
            mw.router_w, mw.switch_gate_w, mw.switch_up_w, mw.switch_down_w,
            mw.shared_gate_w, mw.shared_up_w, mw.shared_down_w,
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
    const qp = headQuantParams(target);
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

/// The trunk lm_head's TRUE quant params — solved from tensor geometry when
/// affine (the head's bits routinely differ from the trunk global on mixed
/// checkpoints), config fallback otherwise.
fn headQuantParams(target: *Transformer) transformer_mod.QuantParams {
    if (transformer_mod.affineParamsFromGeometry(
        target.lm_head_w,
        target.lm_head_s,
        @intCast(target.config.hidden_size),
    )) |qp| return qp;
    return .{
        .bits = target.config.quant_bits,
        .group_size = target.config.quant_group_size,
        .mode = target.config.quant_mode,
    };
}

/// Outputs of the pre-rope half of the MTP layer. All arrays owned.
const FrontOut = struct {
    q_t: mlx.mlx_array, // [1, H, L, D] normed, pre-rope
    k_t: mlx.mlx_array, // [1, Hkv, L, D] normed, pre-rope
    v_t: mlx.mlx_array, // [1, Hkv, L, D]
    gate: mlx.mlx_array, // [1, L, H*D] raw output gate (pre-sigmoid)
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
fn frontChain(self: *const MtpModel, target: *Transformer, id_arr: mlx.mlx_array, hidden: mlx.mlx_array) !FrontOut {
    const s = self.s;
    const cfg = &target.config;
    const h_count: c_int = @intCast(cfg.num_attention_heads);
    const kv_h: c_int = @intCast(cfg.num_key_value_heads);
    const hd: c_int = @intCast(cfg.head_dim);
    const eps = cfg.rms_norm_eps;
    const h_shape = mlx.getShape(hidden);
    const seq_len: c_int = h_shape[1];
    const flat_shape = [_]c_int{ 1, seq_len, h_count * hd };

    // fc(concat([norm(embed), norm(hidden)]))
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
    var x = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(x);
    if (self.eh_proj) |*ep| {
        // Hy3: the concat projection is quantized (mtp.eh_proj, 8-bit).
        _ = mlx.mlx_array_free(x);
        x = try qLinearFwd(self, cat, ep);
    } else {
        try mlx.check(mlx.mlx_matmul(&x, cat, self.fc_w_t, s));
    }

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

        var gate_4d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_4d);
        try mlx.check(mlx.mlx_vector_array_get(&gate_4d, split_vec, 1));
        try mlx.check(mlx.mlx_reshape(&gate, gate_4d, &flat_shape, 3, s));
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
    var attn_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn_flat);
    try mlx.check(mlx.mlx_reshape(&attn_flat, attn_t, &flat_shape, 3, s));

    // Output gate: o_proj(attn * sigmoid(gate)); ungated archs (Hy3) pass a
    // null-ctx gate → straight o_proj(attn).
    const o_out = if (gate.ctx == null)
        try qLinearFwd(self, attn_flat, &self.o)
    else blk: {
        var gate_sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_sig);
        try mlx.check(mlx.mlx_sigmoid(&gate_sig, gate, s));
        var gated = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gated);
        try mlx.check(mlx.mlx_multiply(&gated, attn_flat, gate_sig, s));
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
    const s = self.s;
    const cfg = &target.config;
    const hidden_size: c_int = @intCast(cfg.hidden_size);
    const h_shape = mlx.getShape(hidden);
    const seq_len: c_int = h_shape[1];
    const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(cfg.query_pre_attn_scalar)));
    const rope_dims: c_int = @intFromFloat(@as(f32, @floatFromInt(cfg.head_dim)) * cfg.partial_rotary_factor);

    var front = try frontChain(self, target, id_arr, hidden);
    defer front.deinit();

    var q_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_rope);
    var k_rope = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_rope);
    try mlx.check(mlx.mlx_fast_rope(&q_rope, front.q_t, rope_dims, false, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, rope_offset, .{ .ctx = null }, s));
    try mlx.check(mlx.mlx_fast_rope(&k_rope, front.k_t, rope_dims, false, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, rope_offset, .{ .ctx = null }, s));

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

    if (!want_logits) {
        return .{ .logits = .{ .ctx = null }, .hidden_next = post };
    }
    // Logits (and the chained hidden) are only ever consumed for the LAST
    // row — the draft chain's next token / confidence. Multi-row calls (the
    // merged history+draft forward) slice to the last row BEFORE the vocab
    // head projection: projecting the history rows through a 248k-vocab head
    // is pure waste, and a [1,L,H] hidden_next would break the L=1 chain.
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
    if (token_ids.len == 0) return;
    const ids_i32 = try self.allocator.alloc(i32, token_ids.len);
    defer self.allocator.free(ids_i32);
    for (token_ids, 0..) |t, i| ids_i32[i] = @intCast(t);
    const id_shape = [_]c_int{@intCast(token_ids.len)};
    const id_arr = mlx.mlx_array_new_data(ids_i32.ptr, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(id_arr);

    // KVCache.update advances `cache.step` (layer 0) by the batch length.
    var out = try forward(self, target, cache, id_arr, hidden, rope_offset, false);
    _ = mlx.mlx_array_free(out.hidden_next);
    out.hidden_next = .{ .ctx = null };
}

/// One lazy draft step: `[1]`-shaped (possibly lazy) token id + `[1,1,H]`
/// hidden → logits + next hidden. Appends one entry to `cache`.
pub fn stepArr(
    self: *const MtpModel,
    target: *Transformer,
    cache: *KVCache,
    prev_token_arr: mlx.mlx_array,
    hidden: mlx.mlx_array,
    rope_offset: c_int,
) !StepOut {
    // KVCache.update advances `cache.step` (layer 0) by 1.
    return forward(self, target, cache, prev_token_arr, hidden, rope_offset, true);
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
                var data: [16]f32 = .{0.0} ** 16;
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
    try testing.expect(m.fc_w_t.ctx == null);
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

test "mtp: sidecar resolution accepts native and Forge layouts in priority order" {
    const io = testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // Nothing present → null.
    try testing.expectEqual(@as(?[]const u8, null), resolveMtpSidecarInDir(io, tmp.dir));

    // oMLX OptiQ layout is discovered when it's the only head present.
    try tmp.dir.createDirPath(io, "optiq");
    try tmp.dir.writeFile(io, .{ .sub_path = "optiq/mtp.safetensors", .data = "x" });
    try testing.expectEqualStrings("optiq/mtp.safetensors", resolveMtpSidecarInDir(io, tmp.dir).?);

    try tmp.dir.writeFile(io, .{ .sub_path = "model-mtp.safetensors", .data = "x" });
    try testing.expectEqualStrings("model-mtp.safetensors", resolveMtpSidecarInDir(io, tmp.dir).?);

    // Forge current name outranks legacy.
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp.safetensors", .data = "x" });
    try testing.expectEqualStrings("mtp.safetensors", resolveMtpSidecarInDir(io, tmp.dir).?);

    // Native mlx-serve layout outranks both Forge names.
    try tmp.dir.createDirPath(io, "mtp");
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp/weights.safetensors", .data = "x" });
    try testing.expectEqualStrings("mtp/weights.safetensors", resolveMtpSidecarInDir(io, tmp.dir).?);
}

test "mtp: empty sidecar file is not a sidecar" {
    const io = testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp.safetensors", .data = "" });
    try testing.expectEqual(@as(?[]const u8, null), resolveMtpSidecarInDir(io, tmp.dir));
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

test "mtp: inferGroupSize geometry" {
    // 4-bit packed: weight [out, in*4/32] u32, scales [out, in/group].
    // Synthetic pair: packed_cols=4 → expanded in=32; scale_cols=2 → group 16.
    var q = QLinear{
        .w = mlx.mlx_array_new_data(&[_]i32{0} ** 8, &[_]c_int{ 2, 4 }, 2, .int32),
        .s = mlx.mlx_array_new_data(&[_]f32{0} ** 4, &[_]c_int{ 2, 2 }, 2, .float32),
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

    try testing.expectEqual(MtpCostProfile.g17_nax_q8_gs32, m5NaxCostProfileForQuant(8, 32));
    try testing.expectEqual(MtpCostProfile.g17_nax_q4_gs32, m5NaxCostProfileForQuant(4, 32));
    try testing.expectEqual(MtpCostProfile.generic, m5NaxCostProfileForQuant(8, 64));
    try testing.expectEqual(MtpCostProfile.generic, m5NaxCostProfileForQuant(4, 64));
    try testing.expectEqual(MtpCostProfile.generic, m5NaxCostProfileForQuant(3, 32));

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
        .q = &q4_set[0], .k = &q4_set[1], .v = &q4_set[2], .o = &q4_set[3],
        .gate = &q4_set[4], .up = &q4_set[5], .down = &q4_set[6],
    };
    const q8_linears: M5NaxDenseSidecarLinears = .{
        .q = &q8_set[0], .k = &q8_set[1], .v = &q8_set[2], .o = &q8_set[3],
        .gate = &q8_set[4], .up = &q8_set[5], .down = &q8_set[6],
    };
    const q4_geom: M5NaxDenseSidecarGeometry = .{
        .hidden = IN, .q_out = OUT, .kv_out = OUT, .full_out = OUT,
        .intermediate = OUT, .bits = 4, .group_size = 32,
    };
    const q8_geom: M5NaxDenseSidecarGeometry = .{
        .hidden = IN, .q_out = OUT, .kv_out = OUT, .full_out = OUT,
        .intermediate = OUT, .bits = 8, .group_size = 32,
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
    const fcs = mlx.getShape(m.fc_w_t);
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

}
