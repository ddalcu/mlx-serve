//! sdk/blocks/attention/sdpa.
//! Owns nax_sdpa_avail_cache, nax_sdpa_override, nax_sdpa_env, naxSdpaPreferredFrom, naxSdpaPreferred, sdpaForceFusedFor, and 15 more.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const computeNaxAvailable = @import("../../../mlx/device.zig").computeNaxAvailable;
const FUSED256_MIN_Q_LEN = @import("attn256.zig").FUSED256_MIN_Q_LEN;

var nax_sdpa_avail_cache: ?bool = null;
pub var nax_sdpa_override: ?bool = null; // test seam


/// mlx >= 0.32.2 serves hd-256 attention on NAX with a fused full-attention kernel (`sdpa_full_self_attention_nax`),
/// auto only at >= 1024 causal rows and reachable below that through `force_fused`: preferred wherever NAX is.
pub fn naxSdpaPreferred() bool {
    if (nax_sdpa_override) |v| return v;
    if (nax_sdpa_avail_cache) |v| return v;
    const on = computeNaxAvailable();
    nax_sdpa_avail_cache = on;
    if (on) log.info("[nax-sdpa] engaged: stock fused sdpa (force_fused) serves hd-256 causal prefill and verify blocks > 8 rows; msv_attn_p256 causal arm declined\n", .{});
    return on;
}

/// force_fused is legal only where MLX has a fused kernel for the shape:
/// hd 256 (<= 128 fuses on its own, 192 is not ours), more than 8 query
/// rows (<= 8 is the vector kernel and its q*gqa <= 32 wall makes
/// force_fused THROW), q <= k (the full kernel's causal precondition).
pub fn sdpaForceFusedFor(preferred: bool, q_len: c_int, k_len: c_int, head_dim: c_int) bool {
    return preferred and head_dim == 256 and q_len > 8 and q_len <= k_len;
}

pub fn sdpaForceFused(q: mlx.mlx_array, k: mlx.mlx_array) bool {
    if (!naxSdpaPreferred()) return false;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4) return false;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const on = sdpaForceFusedFor(true, qs[2], ks[2], qs[3]);
    if (on and !nax_sdpa_dispatch_logged) {
        nax_sdpa_dispatch_logged = true;
        log.info("[nax-sdpa] force_fused dispatch engaged: qL={d} kL={d} Hq={d} Hkv={d} hd={d}\n", .{ qs[2], ks[2], qs[1], ks[1], qs[3] });
    }
    return on;
}
var nax_sdpa_dispatch_logged = false;

/// Test seam: `sdpa_split_override` forces the arm on/off.
pub var sdpa_split_override: ?bool = null;
fn sdpaSplitEnabled() bool {
    return sdpa_split_override orelse true;
}

/// One engagement log per width (2..15): the first engagement is usually the
/// warmup's own short prefill — a per-width line is what witnesses the split
/// at REAL verify shapes in an A/B arm's log.
var sdpa_split_logged: [16]bool = @splat(false);

/// Rows per group that MLX's vector sdpa kernel serves for this geometry, or
/// null when one call already takes the fast path (or no split can help).
/// The vector kernel serves q_len <= 8 with q_len * gqa <= 32 at hd
/// {64, 96, 128, 256}; q_len > 8 runs MLX's fused full kernel except at hd
/// 256, where MLX takes its unfused fallback unless NAX force-fuses it.
/// Engages at hd 256 only until other head dims are measured.
fn causalSplitGroupRows(q_len: c_int, gqa: c_int, head_dim: c_int) ?c_int {
    const vector_hd = head_dim == 64 or head_dim == 96 or head_dim == 128 or head_dim == 256;
    if (!vector_hd or gqa < 1) return null;
    if (q_len < 2 or q_len >= FUSED256_MIN_Q_LEN) return null;
    const fits_vector = q_len <= 8 and q_len * gqa <= 32;
    if (fits_vector) return null;
    if (q_len > 8 and head_dim != 256) return null;
    // Measured only at hd 256 (Qwen3.8-Flash-Next, gqa 12). Other head dims
    // need their own per-arch A/B before this default-on path reaches them.
    if (head_dim != 256) return null;
    return @min(8, @max(1, @divTrunc(32, gqa)));
}

/// Width-wall query split for dense causal spec-verify blocks (B==1). MLX's
/// vector sdpa kernel serves q_len * gqa <= 32 (q_len <= 8), so a wider
/// verify block otherwise runs the unfused fallback (~8 dispatches, scores
/// materialized). Rows split into groups of min(8, 32/gqa): 5 at gqa 6, 2 at
/// gqa 12 (Qwen3.8-Flash-Next). Group [r0, r1) runs "causal" over keys[0 ..
/// kL-(qL-r1)]; bottom-right alignment then gives row r the window
/// kL-qL+r, the same one a width-(r1-r0) round at that offset gets. K/V are
/// views, never recomputed. Returns null outside the envelope — the caller
/// falls through to the single sdpa call. Generalizes the Layr-Labs
/// qwen-3.8-mtp-challenge width-wall split (see NOTICE).
pub fn splitCausalSdpa(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
) !?mlx.mlx_array {
    if (!sdpaSplitEnabled()) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const hd = qs[3];
    if (ks[3] != hd or vs[3] != hd) return null;
    if (qs[0] != 1 or ks[0] != 1 or vs[0] != 1) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[1] < 1) return null;
    if (@rem(qs[1], ks[1]) != 0) return null;
    const qL = qs[2];
    const kL = ks[2];
    const gqa = @divTrunc(qs[1], ks[1]);
    const group = causalSplitGroupRows(qL, gqa, hd) orelse return null;
    // Fallback-only: where the caller's single call force-fuses onto MLX's
    // NAX kernel (hd 256, qL > 8), that one dispatch beats any split.
    if (sdpaForceFusedFor(naxSdpaPreferred(), qL, kL, hd)) return null;

    const strides = [_]c_int{ 1, 1, 1, 1 };
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    var parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.heap.c_allocator);
    }
    var r0: c_int = 0;
    while (r0 < qL) : (r0 += group) {
        const r1: c_int = @min(r0 + group, qL);
        const k_end: c_int = kL - (qL - r1);
        var q_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_g);
        var k_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_g);
        var v_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_g);
        try mlx.check(mlx.mlx_slice(&q_g, q, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, qs[1], r1, hd }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&k_g, k, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, ks[1], k_end, hd }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&v_g, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ 1, vs[1], k_end, hd }, 4, &strides, 4, s));
        var o_g = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o_g);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_g, q_g, k_g, v_g, scale, "causal", none_mask, .{ .ctx = null }, false, s));
        try parts.append(std.heap.c_allocator, o_g);
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));

    const log_idx: usize = @intCast(qL);
    if (!sdpa_split_logged[log_idx]) {
        sdpa_split_logged[log_idx] = true;
        log.info("[sdpa-split] engaged: qL={d} kL={d} Hq={d} Hkv={d} rows/group={d}\n", .{ qL, kL, qs[1], ks[1], group });
    }
    return out;
}

var sdpa_mask_split_logged = false;

/// Array-mask verify widths on hd 256 (qwen4_exp QSA past the budget): MLX's
/// vector kernel serves `qL * gqa <= 32` rows, anything wider (S >= 3 at
/// gqa 12) and every array-masked qL 9..15 is the unfused fallback. Rows are
/// independent under a per-row mask, so slice the block into vector-sized row
/// groups. The test seam is shared with the causal split.
pub fn splitMaskedSdpa256(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    mask: mlx.mlx_array,
) !?mlx.mlx_array {
    if (!sdpaSplitEnabled()) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(mask) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const ms = mlx.getShape(mask);
    if (qs[3] != 256 or ks[3] != 256) return null;
    if (qs[0] != 1 or ks[0] != 1 or ms[0] != 1 or ms[1] != 1 or ms[2] != qs[2] or ms[3] != ks[2]) return null;
    const qL = qs[2];
    if (qL < 2 or qL >= FUSED256_MIN_Q_LEN) return null;
    const gqa: c_int = @divTrunc(qs[1], ks[1]);
    if (qL * gqa <= 32) return null;
    const group: c_int = @max(1, @divTrunc(32, gqa));

    var parts = std.ArrayList(mlx.mlx_array).empty;
    defer {
        for (parts.items) |a| _ = mlx.mlx_array_free(a);
        parts.deinit(std.heap.c_allocator);
    }
    const strides = [_]c_int{ 1, 1, 1, 1 };
    var r0: c_int = 0;
    while (r0 < qL) : (r0 += group) {
        const r1: c_int = @min(r0 + group, qL);
        var q_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_g);
        var m_g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(m_g);
        try mlx.check(mlx.mlx_slice(&q_g, q, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, qs[1], r1, 256 }, 4, &strides, 4, s));
        try mlx.check(mlx.mlx_slice(&m_g, mask, &[_]c_int{ 0, 0, r0, 0 }, 4, &[_]c_int{ 1, 1, r1, ms[3] }, 4, &strides, 4, s));
        var o_g = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(o_g);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o_g, q_g, k, v, scale, "array", m_g, .{ .ctx = null }, false, s));
        try parts.append(std.heap.c_allocator, o_g);
    }
    const vec = mlx.mlx_vector_array_new_data(parts.items.ptr, parts.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 2, s));
    if (!sdpa_mask_split_logged) {
        sdpa_mask_split_logged = true;
        log.info("[sdpa-split] masked arm engaged: qL={d} kL={d} gqa={d} rows/group={d}\n", .{ qL, ks[2], gqa, group });
    }
    return out;
}

/// q [1, H, S, hd] over k/v [1, KVH, L, hd] whose last S rows are the
/// window's: row r takes the one-query sdpa over keys [0, L - S + r].
pub fn rowByRowSdpa(out: *mlx.mlx_array, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, seq: c_int, s: mlx.mlx_stream) !void {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const none_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none_mask);
    const rows = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(rows);
    const st4 = [_]c_int{ 1, 1, 1, 1 };
    var r: c_int = 0;
    while (r < seq) : (r += 1) {
        const keys = ks[2] - seq + r + 1;
        var qr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qr);
        try mlx.check(mlx.mlx_slice(&qr, q, &[_]c_int{ 0, 0, r, 0 }, 4, &[_]c_int{ qs[0], qs[1], r + 1, qs[3] }, 4, &st4, 4, s));
        var kr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kr);
        try mlx.check(mlx.mlx_slice(&kr, k, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ ks[0], ks[1], keys, ks[3] }, 4, &st4, 4, s));
        var vr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(vr);
        try mlx.check(mlx.mlx_slice(&vr, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ vs[0], vs[1], keys, vs[3] }, 4, &st4, 4, s));
        var o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&o, qr, kr, vr, scale, "", none_mask, .{ .ctx = null }, false, s));
        _ = mlx.mlx_vector_array_append_value(rows, o);
    }
    try mlx.check(mlx.mlx_concatenate_axis(out, rows, 2, s));
}
