//! The qwen4_exp MTP head's measured cost surfaces (M5 NAX): the all-affine-4/gs-64 trunk and the mixed 4/8-bit
//! pack, each matched against the live checkpoint's geometry, packs, head KV and rerank before it prices a round.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const model_mod = @import("../../sdk/config/model.zig");
const kv_quant = @import("../../engines/mlx/cache/kv_quant.zig");
const verify_qmm_nax = @import("../../sdk/blocks/linear/verify_qmm_nax.zig");
const MtpCostProfile = @import("../../sdk/spec/mtp/mtp.zig").MtpCostProfile;
const Trunk = @import("trunk.zig").Trunk;
const testing = std.testing;

/// Pure first-stage classifier for the qwen4_exp in-checkpoint head. The
/// trunk is MoE, so the sidecar fingerprint path above can never match it;
/// this is its own measured surface, not a widening of the sidecar ones.
pub fn qwen4G17CostProfileForFingerprint(
    trunk_affine: bool,
    trunk_bits: u32,
    trunk_group_size: u32,
    head_packs_match: bool,
    nax_lane_live: bool,
) MtpCostProfile {
    if (!trunk_affine or trunk_bits != 4 or trunk_group_size != 64) return .generic;
    if (!head_packs_match) return .generic;
    if (!nax_lane_live) return .generic;
    return .g17_nax_qwen4_q4_gs64;
}

const Qwen4MixedGeometry = struct {
    hidden: u32 = 2560,
    layers: u32 = 48,
    hc: u32 = 4,
    experts: u32 = 512,
    top_k: u32 = 10,
    expert_width: u32 = 640,
    vocab: u32 = 248320,
    ple_layer: i32 = 1,
    kv_bits: u32 = 8,
    kv_group: u32 = 64,
    head_kv_bits: u32 = 8,
    head_kv_group: u32 = 64,
};

fn qwen4MixedCostProfileForFingerprint(geometry: Qwen4MixedGeometry, packs_match: bool, rerank_ready: bool, nax_live: bool) MtpCostProfile {
    if (!std.meta.eql(geometry, Qwen4MixedGeometry{}) or !packs_match or !rerank_ready or !nax_live) return .generic;
    return .g17_nax_qwen4_mixed_4_8_gs64;
}

fn qwen4MixedPackMatches(w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array, logical: []const c_int, bits: u32) bool {
    if (logical.len < 2 or logical.len > 3 or (bits != 4 and bits != 8)) return false;
    if (w.ctx == null or s.ctx == null or b.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32 or mlx.mlx_array_dtype(s) != .bfloat16 or mlx.mlx_array_dtype(b) != .bfloat16) return false;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(s);
    const bs = mlx.getShape(b);
    if (ws.len != logical.len or ss.len != logical.len or !std.mem.eql(c_int, ss, bs)) return false;
    const last = logical.len - 1;
    for (logical[0..last], ws[0..last], ss[0..last]) |want, wc, sc| {
        if (want <= 0 or wc != want or sc != want) return false;
    }
    const k = logical[last];
    if (k <= 0 or @mod(k, 64) != 0) return false;
    return @as(i64, ws[last]) * 32 == @as(i64, k) * bits and ss[last] == @divExact(k, 64);
}

test "Qwen4 mixed pack matcher pins actual quantization geometry" {
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    for ([_]u32{ 4, 8 }) |bits| for ([_]bool{ false, true }) |stacked| {
        const words: c_int = @intCast(bits * 2);
        const logical: []const c_int = if (stacked) &.{ 2, 16, 64 } else &.{ 16, 64 };
        const ws: []const c_int = if (stacked) &.{ 2, 16, words } else &.{ 16, words };
        const ss: []const c_int = if (stacked) &.{ 2, 16, 1 } else &.{ 16, 1 };
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        var scales = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scales);
        var biases = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(biases);
        try mlx.check(mlx.mlx_zeros(&w, ws.ptr, @intCast(ws.len), .uint32, stream));
        try mlx.check(mlx.mlx_zeros(&scales, ss.ptr, @intCast(ss.len), .bfloat16, stream));
        try mlx.check(mlx.mlx_zeros(&biases, ss.ptr, @intCast(ss.len), .bfloat16, stream));
        try testing.expect(qwen4MixedPackMatches(w, scales, biases, logical, bits));
        try testing.expect(!qwen4MixedPackMatches(w, scales, biases, logical, if (bits == 4) 8 else 4));
        try testing.expect(!qwen4MixedPackMatches(w, scales, .{}, logical, bits));
        try testing.expect(!qwen4MixedPackMatches(w, scales, biases, &.{ 16, 128 }, bits));
        var wrong_dtype = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wrong_dtype);
        try mlx.check(mlx.mlx_zeros(&wrong_dtype, ss.ptr, @intCast(ss.len), .float32, stream));
        try testing.expect(!qwen4MixedPackMatches(w, wrong_dtype, biases, logical, bits));
        try testing.expect(!qwen4MixedPackMatches(w, scales, wrong_dtype, logical, bits));
    };
}

test "Qwen4 mixed MTP profile requires its measured geometry and packs" {
    try testing.expectEqual(MtpCostProfile.g17_nax_qwen4_mixed_4_8_gs64, qwen4MixedCostProfileForFingerprint(.{}, true, true, true));
    inline for (.{ "hidden", "layers", "hc", "experts", "top_k", "expert_width", "vocab", "ple_layer", "kv_bits", "kv_group", "head_kv_bits", "head_kv_group" }) |field| {
        var geometry: Qwen4MixedGeometry = .{};
        @field(geometry, field) += 1;
        try testing.expectEqual(MtpCostProfile.generic, qwen4MixedCostProfileForFingerprint(geometry, true, true, true));
    }
    for ([_]bool{ false, true }) |packs| for ([_]bool{ false, true }) |rerank| for ([_]bool{ false, true }) |nax| {
        if (packs and rerank and nax) continue;
        try testing.expectEqual(MtpCostProfile.generic, qwen4MixedCostProfileForFingerprint(.{}, packs, rerank, nax));
    };
}

test "Qwen4 mixed MTP profile validates the live checkpoint (QWEN4_TEST_MODEL)" {
    const path = env.get(.QWEN4_TEST_MODEL) orelse return error.SkipZigTest;
    if (mlx.noGpuBackend() or !verify_qmm_nax.verifyQmmNaxAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer if (config.ngram_table_path) |name| a.free(name);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const kv8 = kv_quant.KVQuantConfig.affine(8);
    try xfm.qwen4MtpApplyKvQuant(kv8);
    try testing.expect(xfm.qwen4BuildDraftRerank());
    try testing.expectEqual(MtpCostProfile.g17_nax_qwen4_mixed_4_8_gs64, qwen4G17CostProfile(&xfm, kv8));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForKv(&xfm, kv_quant.KVQuantConfig.dense));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForKv(&xfm, kv_quant.KVQuantConfig.affine(4)));
    const old_bias = xfm.qwen4_mtp.?.fc_emb_b;
    defer xfm.qwen4_mtp.?.fc_emb_b = old_bias;
    xfm.qwen4_mtp.?.fc_emb_b = .{};
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfile(&xfm, kv8));
}

var qwen4_g17_profile_logged: bool = false;
var qwen4_mixed_profile_logged: bool = false;

fn qwen4MixedHcMatches(hc: anytype) bool {
    return qwen4MixedPackMatches(hc.down_w, hc.down_s, hc.down_b, &.{ 320, 10240 }, 8) and
        qwen4MixedPackMatches(hc.up_w, hc.up_s, hc.up_b, &.{ 10240, 320 }, 8);
}

fn qwen4MixedLayerMatches(layer: anytype) bool {
    const mw = switch (layer.mlp) {
        .moe => |value| value,
        else => return false,
    };
    if (!qwen4MixedPackMatches(mw.switch_gate_w, mw.switch_gate_s, mw.switch_gate_b, &.{ 512, 640, 2560 }, 4) or
        !qwen4MixedPackMatches(mw.switch_up_w, mw.switch_up_s, mw.switch_up_b, &.{ 512, 640, 2560 }, 4) or
        !qwen4MixedPackMatches(mw.switch_down_w, mw.switch_down_s, mw.switch_down_b, &.{ 512, 2560, 640 }, 4) or
        !qwen4MixedPackMatches(mw.shared_gate_w, mw.shared_gate_s, mw.shared_gate_b, &.{ 640, 2560 }, 8) or
        !qwen4MixedPackMatches(mw.shared_up_w, mw.shared_up_s, mw.shared_up_b, &.{ 640, 2560 }, 8) or
        !qwen4MixedPackMatches(mw.shared_down_w, mw.shared_down_s, mw.shared_down_b, &.{ 2560, 640 }, 8) or
        !qwen4MixedPackMatches(mw.router_w, mw.router_s, mw.router_b, &.{ 512, 2560 }, 8)) return false;
    if (!qwen4MixedHcMatches(layer.hc_attn orelse return false) or !qwen4MixedHcMatches(layer.hc_mlp orelse return false)) return false;
    if (layer.ple) |ple| {
        if (!qwen4MixedPackMatches(ple.key_w, ple.key_s, ple.key_b, &.{ 10240, 2560 }, 8) or
            !qwen4MixedPackMatches(ple.value_w, ple.value_s, ple.value_b, &.{ 2560, 2560 }, 8)) return false;
    }
    return switch (layer.attn) {
        .full => |f| qwen4MixedPackMatches(f.q_w, f.q_s, f.q_b, &.{ 12288, 2560 }, 8) and
            qwen4MixedPackMatches(f.k_w, f.k_s, f.k_b, &.{ 512, 2560 }, 8) and
            qwen4MixedPackMatches(f.v_w, f.v_s, f.v_b, &.{ 512, 2560 }, 8) and
            qwen4MixedPackMatches(f.o_w, f.o_s, f.o_b, &.{ 2560, 6144 }, 8) and
            qwen4MixedPackMatches(f.idx_qk_w, f.idx_qk_s, f.idx_qk_b, &.{ 640, 2560 }, 8),
        .linear => |l| qwen4MixedPackMatches(l.qkv_w, l.qkv_s, l.qkv_b, &.{ 10240, 2560 }, 8) and
            qwen4MixedPackMatches(l.z_w, l.z_s, l.z_b, &.{ 6144, 2560 }, 8) and
            qwen4MixedPackMatches(l.a_w, l.a_s, l.a_b, &.{ 48, 2560 }, 8) and
            qwen4MixedPackMatches(l.b_w, l.b_s, l.b_b, &.{ 48, 2560 }, 8) and
            qwen4MixedPackMatches(l.out_w, l.out_s, l.out_b, &.{ 2560, 6144 }, 8),
    };
}

fn qwen4MixedModelPacksMatch(target: *const Trunk) bool {
    if (target.qwen4 == null or target.qwen4_stream_f32 or target.config.quant_mode != .affine or target.config.quant_group_size != 64) return false;
    const cfg = &target.config;
    if (cfg.linear_key_head_dim != 128 or cfg.linear_value_head_dim != 128 or cfg.linear_num_key_heads != 16 or
        cfg.linear_num_value_heads != 48 or cfg.linear_conv_kernel_dim != 4 or cfg.ple_embed_dim != 2560) return false;
    const head = target.qwen4_mtp orelse return false;
    if (!qwen4MixedPackMatches(target.emb_w, target.emb_s, target.emb_b, &.{ 248320, 2560 }, 4) or
        !qwen4MixedPackMatches(target.lm_head_w, target.lm_head_s, target.lm_head_b, &.{ 248320, 2560 }, 8) or
        !qwen4MixedPackMatches(head.fc_emb_w, head.fc_emb_s, head.fc_emb_b, &.{ 2560, 2560 }, 8) or
        !qwen4MixedPackMatches(head.fc_hid_w, head.fc_hid_s, head.fc_hid_b, &.{ 2560, 2560 }, 8)) return false;
    if (!qwen4MixedHcMatches(target.qwen4_mixer) or !qwen4MixedHcMatches(head.mixer) or
        head.layer.is_linear or !qwen4MixedLayerMatches(&head.layer)) return false;
    const layers = target.moe_layers;
    if (layers.len != 48) return false;
    for (layers, 0..) |*layer, i| {
        if (layer.is_linear != (i % 4 != 3) or (layer.ple != null) != (i == 1) or !qwen4MixedLayerMatches(layer)) return false;
    }
    return true;
}

/// Live resolver for the qwen4_exp head: validates the runtime fingerprint
/// the G17 surface was measured on — uniform affine-4/gs-64 config and both
/// head fc packs at that width (K derived from the scales, so no input-dim
/// formula is assumed).
pub fn qwen4G17CostProfile(target: *const Trunk, trunk_kv: kv_quant.KVQuantConfig) MtpCostProfile {
    return qwen4G17CostProfileForKv(target, trunk_kv);
}

/// Mixed-pack prices use the request's actual KV format, including a slot
/// override. The original all-4-bit classifier retains its existing scope.
pub fn qwen4G17CostProfileForKv(target: *const Trunk, kv: kv_quant.KVQuantConfig) MtpCostProfile {
    const head = if (target.qwen4_mtp) |*h| h else return .generic;
    const cfg = &target.config;
    const geometry = Qwen4MixedGeometry{
        .hidden = cfg.hidden_size,
        .layers = cfg.num_hidden_layers,
        .hc = cfg.hc_count,
        .experts = cfg.num_experts,
        .top_k = cfg.num_experts_per_tok,
        .expert_width = cfg.moe_intermediate_size,
        .vocab = cfg.vocab_size,
        .ple_layer = cfg.ple_layer_idx,
        .kv_bits = if (kv.scheme == .affine) kv.bits else 0,
        .kv_group = kv.group_size,
        .head_kv_bits = if (head.cache.config.scheme == .affine) head.cache.config.bits else 0,
        .head_kv_group = head.cache.config.group_size,
    };
    const rerank_ready = if (head.rerank) |r| r.bits == 3 and r.group_size == 64 and r.rows == 248320 else false;
    const mixed = qwen4MixedCostProfileForFingerprint(geometry, std.meta.eql(geometry, Qwen4MixedGeometry{}) and qwen4MixedModelPacksMatch(target), rerank_ready, mlx.streamIsGpu(target.s) and verify_qmm_nax.verifyQmmNaxAvailable());
    if (mixed != .generic) {
        if (!qwen4_mixed_profile_logged) {
            qwen4_mixed_profile_logged = true;
            log.info("[mtp] cost profile g17-nax-qwen4-mixed-4-8-gs64 engaged (affine8 target/head KV)\n", .{});
        }
        return mixed;
    }
    if (cfg.hidden_size == 0 or cfg.hidden_size > std.math.maxInt(c_int)) return .generic;
    const hidden: c_int = @intCast(cfg.hidden_size);
    const packs = qwen4PackTripleMatches(head.fc_emb_w, head.fc_emb_s, head.fc_emb_b, hidden) and
        qwen4PackTripleMatches(head.fc_hid_w, head.fc_hid_s, head.fc_hid_b, hidden);
    const profile = qwen4G17CostProfileForFingerprint(
        cfg.quant_mode == .affine,
        cfg.quant_bits,
        cfg.quant_group_size,
        packs,
        verify_qmm_nax.verifyQmmNaxAvailable(),
    );
    if (profile != .generic and !qwen4_g17_profile_logged) {
        qwen4_g17_profile_logged = true;
        log.info("[mtp] cost profile g17-nax-qwen4-q4-gs64 engaged\n", .{});
    }
    return profile;
}

/// One packed projection of the qwen4 head: uint32 codes [N, K*4/32] with
/// bf16 scales and biases [N, K/64], N == hidden. K comes from the scales.
fn qwen4PackTripleMatches(w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array, hidden: c_int) bool {
    if (w.ctx == null or s.ctx == null or b.ctx == null) return false;
    if (mlx.mlx_array_dtype(w) != .uint32) return false;
    if (mlx.mlx_array_dtype(s) != .bfloat16 or mlx.mlx_array_dtype(b) != .bfloat16) return false;
    const ws = mlx.getShape(w);
    const ss = mlx.getShape(s);
    const bs = mlx.getShape(b);
    if (ws.len != 2 or ss.len != 2 or bs.len != 2) return false;
    if (ws[0] != hidden or ss[0] != hidden or bs[0] != hidden) return false;
    if (ss[1] <= 0 or ss[1] != bs[1]) return false;
    const k = @as(u64, @intCast(ss[1])) * 64;
    return @as(u64, @intCast(ws[1])) * 32 == k * 4;
}

test "qwen4 MTP profile: only the affine-4/gs-64 trunk with matching head packs and a live NAX lane is calibrated" {
    try testing.expectEqual(MtpCostProfile.g17_nax_qwen4_q4_gs64, qwen4G17CostProfileForFingerprint(true, 4, 64, true, true));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForFingerprint(false, 4, 64, true, true));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForFingerprint(true, 8, 64, true, true));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForFingerprint(true, 4, 32, true, true));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForFingerprint(true, 4, 64, false, true));
    try testing.expectEqual(MtpCostProfile.generic, qwen4G17CostProfileForFingerprint(true, 4, 64, true, false));
}
