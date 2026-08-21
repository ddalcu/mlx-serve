//! SeedVR2 video-VAE ENCODER — SD3's 2D VAE inflated to causal 3D.
//!
//! Transcribed from ByteDance-Seed/SeedVR `models/video_vae_v3/modules/
//! attn_video_vae.py` + `causal_inflation_lib.py` (Apache-2.0). Geometry and
//! the trap list live in `docs/seedvr2-arch.md` §3; the pure shape/ladder math
//! is `seedvr2_vae_shape.zig` and is oracle-pinned separately.
//!
//! LAYOUT: everything here is NTHWC — `[1, T, H, W, C]` — because that is what
//! `mlx_conv3d` consumes. The reference is NCTHW. Weights are transposed ONCE
//! at load (`[O, I, kt, kh, kw] -> [O, kt, kh, kw, I]`); the hot path never
//! permutes. Fixtures dumped from torch are NCTHW and are permuted by the test
//! reader, not by the encoder.
//!
//! THREE THINGS THAT ARE NOT THE OBVIOUS CHOICE, each of which produces
//! correct SHAPES and wrong numbers:
//!
//!   1. Causality is head REPLICATION, not masking. `InflatedCausalConv3d`
//!      zeroes its own temporal padding and instead prepends
//!      `2 * temporal_padding` copies of FRAME 0, then convolves. There is no
//!      mask anywhere in this VAE.
//!   2. GroupNorm statistics are PER FRAME. The reference folds time into the
//!      batch (`b c t h w -> (b t) c h w`) before every norm, so a frame's
//!      statistics never see another frame. Normalising over the 3D volume is
//!      the natural reading and it couples frames.
//!   3. The encoder's downsamplers pad ASYMMETRICALLY — `(left, right, top,
//!      bottom) = (0, 1, 0, 1)` applied manually, with the conv itself at
//!      padding 0. Symmetric `p=1` shifts the feature map half a pixel per
//!      block and compounds down the ladder.
//!
//! `time_receptive_field` is **full** for this checkpoint (resnet convs are
//! `(3,3,3)`), even though every inner module in the reference defaults to
//! "half" — `VideoAutoencoderKL` overrides it. See `seedvr2_vae_shape.zig`.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const shape = @import("seedvr2_vae_shape.zig");
const log = @import("log.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;

/// GroupNorm/attention epsilon. `resnet_eps=1e-6` everywhere in this VAE,
/// including `conv_norm_out` and the mid-block attention's own group norm.
const EPS: f32 = 1e-6;
const GROUPS: c_int = 32;

// ════════════════════════════════════════════════════════════════════════
// Small array helpers. Deliberate near-duplicates of minimax_h3_vae.zig's —
// these two VAEs share a lineage but not a checkpoint, and coupling them so
// one can be "fixed" for the other is how a shared helper becomes a bug in
// two places at once.
// ════════════════════════════════════════════════════════════════════════

fn contig(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}

fn transpose(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}

fn reshape(x: mlx.mlx_array, shp: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shp.ptr, shp.len, s));
    return o;
}

fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}

/// Evaluate `x` and hand it straight back, so a stage boundary reads as one
/// expression.
///
/// WHY THE VAE IS FULL OF THESE. MLX is lazy and batches an unevaluated graph
/// into ONE command buffer, and it does not release a buffer's arrays until
/// that buffer completes — so the peak is set by how much of the frame is in
/// flight at once, NOT by how promptly this file drops its handles (measured:
/// cutting `groupNormPerFrame` from five frame-sized temporaries to two moved
/// peak by exactly zero bytes). Every full-frame intermediate here is
/// hundreds of megabytes, so cutting the graph at each one is what bounds
/// residency: at 1024x1024 the encoder block's peak went 5.02 -> 4.02 -> 3.02
/// GB as these were added, with no measurable time cost — these are big ops
/// and the sync disappears into them.
///
/// Semantically a no-op: evaluation order never changes a result.
fn evalA(x: mlx.mlx_array) !mlx.mlx_array {
    try mlx.check(mlx.mlx_array_eval(x));
    return x;
}

fn addA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, a, b, s));
    return o;
}

fn subA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, a, b, s));
    return o;
}

fn mulA(a: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, a, b, s));
    return o;
}

fn concat(arrs: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (arrs) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, s));
    return o;
}

fn sliceAxis(x: mlx.mlx_array, axis: usize, lo: c_int, hi: c_int, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    var start: [8]c_int = undefined;
    var stop: [8]c_int = undefined;
    var step: [8]c_int = undefined;
    const nd = shp.len;
    for (0..nd) |i| {
        start[i] = 0;
        stop[i] = @intCast(shp[i]);
        step[i] = 1;
    }
    start[axis] = lo;
    stop[axis] = hi;
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_slice(&o, x, &start, nd, &stop, nd, &step, nd, s));
    return contig(o, s);
}

fn siluA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&o, x, s));
    defer _ = mlx.mlx_array_free(o);
    return mulA(x, o, s);
}

fn ownWeight(w: *const Weights, key: []const u8) !mlx.mlx_array {
    const a = w.get(key) orelse {
        log.err("[seedvr2-vae] MISSING WEIGHT: {s}\n", .{key});
        return error.MissingSeedVr2VaeWeight;
    };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, a));
    return o;
}

/// Which axis order a checkpoint's conv weights are STORED in.
///
/// `mlx_conv3d` wants `[O, kt, kh, kw, I]`. Our own converter ships PyTorch's
/// `[O, I, kt, kh, kw]` and is permuted once at load; the mlx-community
/// mirror ships the destination layout already, and permuting that again is
/// an uncatchable MLX abort rather than a wrong number.
///
/// Decided ONCE per checkpoint rather than per tensor, because per tensor it
/// is not decidable: `encoder.conv_in` is `[C, 3, 3, 3, 3]` under BOTH orders
/// (in_channels 3 is also the kernel extent), so any shape heuristic has to
/// guess on exactly the tensor that starts the encoder. The probe is a
/// mid-block resnet conv, where in_channels is 512 and the two orders differ
/// unmistakably.
const ConvLayout = enum {
    pytorch,
    mlx,

    /// Absent probe ⇒ `.pytorch`, the layout every load before the mirror
    /// existed used. A missing tensor is not evidence of a new layout.
    fn detect(w: *const Weights) ConvLayout {
        for ([_][]const u8{
            "decoder.mid_block.resnets.0.conv1.weight",
            "encoder.mid_block.resnets.0.conv1.weight",
            "decoder.conv_in.weight",
        }) |probe| {
            const t = w.get(probe) orelse continue;
            const shp = mlx.getShape(t);
            if (shp.len != 5) continue;
            // Axis 1 is in_channels under PyTorch and the temporal kernel
            // extent under MLX. The probe's in_channels is 16 or 512, so a
            // small axis 1 beside a large last axis can only be MLX order.
            return if (shp[1] <= 7 and shp[4] > 7) .mlx else .pytorch;
        }
        return .pytorch;
    }
};

/// A conv weight in the layout `mlx_conv3d` wants: `[O, kt, kh, kw, I]`,
/// permuted from PyTorch's `[O, I, kt, kh, kw]` only when that is what the
/// checkpoint actually stores (`ConvLayout`).
fn loadConvW(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, s: S) !mlx.mlx_array {
    const name = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(name);
    const raw = try ownWeight(w, name);
    defer _ = mlx.mlx_array_free(raw);
    if (ConvLayout.detect(w) == .mlx) return astype(raw, mlx.mlx_dtype.float32, s);
    const tr = try transpose(raw, &[_]c_int{ 0, 2, 3, 4, 1 }, s);
    defer _ = mlx.mlx_array_free(tr);
    const trc = try contig(tr, s);
    defer _ = mlx.mlx_array_free(trc);
    return astype(trc, mlx.mlx_dtype.float32, s);
}

/// A 1-D tensor (bias / norm weight) as f32.
fn loadVec(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, s: S) !mlx.mlx_array {
    const name = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(name);
    const raw = try ownWeight(w, name);
    defer _ = mlx.mlx_array_free(raw);
    return astype(raw, mlx.mlx_dtype.float32, s);
}

/// A Linear stored `[out, in]`, pre-transposed to `[in, out]` for matmul.
fn loadLinT(w: *const Weights, a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, s: S) !mlx.mlx_array {
    const name = try std.fmt.allocPrint(a, fmt, args);
    defer a.free(name);
    const raw = try ownWeight(w, name);
    defer _ = mlx.mlx_array_free(raw);
    const f = try astype(raw, mlx.mlx_dtype.float32, s);
    defer _ = mlx.mlx_array_free(f);
    const t = try transpose(f, &[_]c_int{ 1, 0 }, s);
    defer _ = mlx.mlx_array_free(t);
    return contig(t, s);
}

// ════════════════════════════════════════════════════════════════════════
// The three non-obvious primitives
// ════════════════════════════════════════════════════════════════════════

/// `extend_head`: prepend `times` copies of FRAME 0 along T (axis 1 in NTHWC).
///
/// This IS the causality mechanism. `InflatedCausalConv3d.__init__` sets its
/// own temporal padding to zero and replaces it with this replication, so a
/// stride-1 conv stays length-preserving while never reading a future frame.
/// `times == 0` is a no-op — a conv that does not mix time must not extend.
fn extendHead(x: mlx.mlx_array, times: u32, s: S) !mlx.mlx_array {
    if (times == 0) {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&o, x));
        return o;
    }
    const first = try sliceAxis(x, 1, 0, 1, s);
    defer _ = mlx.mlx_array_free(first);
    var parts: [9]mlx.mlx_array = undefined;
    std.debug.assert(times < parts.len);
    var i: u32 = 0;
    while (i < times) : (i += 1) parts[i] = first;
    parts[times] = x;
    return concat(parts[0 .. times + 1], 1, s);
}

/// Env-gated per-stage memory probe. `MLX_SERVE_SEEDVR2_MEM=1` prints active
/// and peak bytes at each VAE stage boundary — the only way to attribute a
/// peak that only appears at sizes too big to hold twice.
var probe_last_ms: i64 = 0;

pub fn memProbeEnabled() bool {
    const v = std.c.getenv("MLX_SERVE_SEEDVR2_MEM") orelse return false;
    return std.mem.span(v).len > 0;
}

/// Env-gated per-stage probe: active/peak bytes AND the wall time since the
/// previous probe. The two belong together — this restoration's cost splits
/// roughly encode/DiT/decode, and which stage owns a number is the whole
/// question every time one of them moves.
pub fn memProbe(tag: []const u8) void {
    if (!memProbeEnabled()) return;
    var active: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_peak_memory(&peak);
    const gb = 1024.0 * 1024.0 * 1024.0;
    const now = std.Io.Timestamp.now(
        std.Io.Threaded.global_single_threaded.io(), .boot).toMilliseconds();
    const dt: f64 = if (probe_last_ms == 0) 0 else @as(f64, @floatFromInt(now - probe_last_ms)) / 1e3;
    probe_last_ms = now;
    std.debug.print("[seedvr2-mem] {s}: active={d:.2} GB peak={d:.2} GB (+{d:.2}s)\n", .{
        tag, @as(f64, @floatFromInt(active)) / gb, @as(f64, @floatFromInt(peak)) / gb, dt,
    });
}

/// Causal 3D conv. `temporal_padding` is the padding the PyTorch module was
/// DECLARED with — it is converted to a head extension of `2 * temporal_padding`
/// frames and the conv itself runs with zero temporal padding.
fn causalConv3d(
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    b: mlx.mlx_array,
    stride: [3]c_int,
    temporal_padding: u32,
    spatial_padding: c_int,
    s: S,
) !mlx.mlx_array {
    // A SINGLE-FRAME input makes the whole temporal machinery a 3x memory
    // waste. `extendHead` prepends `times` copies of frame 0, so at T == 1
    // every temporal tap of the kernel reads the SAME frame and the output is
    // `sum_k w[:,k] conv x0` — identical to a one-tap conv with the taps
    // summed. Taking that shortcut skips the 3x concatenation AND the
    // contiguous copy it forces: measured at 1024x1024x128, that pair is
    // 3.2 GB of the encoder block's 5.0 GB peak, and it scales with pixel
    // area, which is what put a 1536x1536 restore past physical memory.
    //
    // EXACT, not an approximation — but only because the frames are literally
    // replicas. A real video (T > 1) reads different frames per tap and must
    // take the general path below.
    const times = shape.extendHeadTimes(temporal_padding);
    const in_t = mlx.getShape(x)[1];
    const kt = mlx.getShape(w)[1];
    if (in_t == 1 and times > 0 and kt == 1 + @as(c_int, @intCast(times))) {
        var wsum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsum);
        try mlx.check(mlx.mlx_sum_axis(&wsum, w, 1, true, s));
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_conv3d(
            &o, x, wsum,
            stride[0], stride[1], stride[2],
            0, spatial_padding, spatial_padding,
            1, 1, 1, 1, s,
        ));
        defer _ = mlx.mlx_array_free(o);
        return addA(o, b, s);
    }

    const ext = try extendHead(x, times, s);
    defer _ = mlx.mlx_array_free(ext);
    const xc = try contig(ext, s);
    defer _ = mlx.mlx_array_free(xc);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv3d(
        &o,
        xc,
        w,
        stride[0],
        stride[1],
        stride[2],
        0, // temporal padding is ALWAYS zero — extendHead did it
        spatial_padding,
        spatial_padding,
        1,
        1,
        1,
        1,
        s,
    ));
    defer _ = mlx.mlx_array_free(o);
    return addA(o, b, s);
}

/// GroupNorm with PER-FRAME statistics, on `[1, T, H, W, C]`.
///
/// The reference reshapes `b c t h w -> (b t) c h w` before every norm, so the
/// reduction is over `(H, W, C/32)` for each `(batch, frame, group)`. Time is
/// NOT a reduction axis. Channels split as `(group, channel_in_group)` with the
/// group as the slower axis, matching PyTorch's contiguous-within-group layout.
fn groupNormPerFrame(x: mlx.mlx_array, w: mlx.mlx_array, b: mlx.mlx_array, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    const t = shp[1];
    const h = shp[2];
    const wd = shp[3];
    const c = shp[4];
    const g = try reshape(x, &[_]c_int{ 1, t, h, wd, GROUPS, @divExact(c, GROUPS) }, s);
    defer _ = mlx.mlx_array_free(g);
    // Axes 2,3,5 = H, W, channel-within-group. NOT axis 1 (T) — that is the
    // whole point.
    const axes = [_]c_int{ 2, 3, 5 };
    var mean = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mean);
    try mlx.check(mlx.mlx_mean_axes(&mean, g, &axes, axes.len, true, s));
    const diff = try subA(g, mean, s);
    defer _ = mlx.mlx_array_free(diff);
    const sq = try mulA(diff, diff, s);
    defer _ = mlx.mlx_array_free(sq);
    var vr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vr);
    try mlx.check(mlx.mlx_mean_axes(&vr, sq, &axes, axes.len, true, s));
    const eps = mlx.mlx_array_new_float(EPS);
    defer _ = mlx.mlx_array_free(eps);
    const ve = try addA(vr, eps, s);
    defer _ = mlx.mlx_array_free(ve);
    var rs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rs);
    try mlx.check(mlx.mlx_rsqrt(&rs, ve, s));
    const nrm = try mulA(diff, rs, s);
    defer _ = mlx.mlx_array_free(nrm);
    const back = try reshape(nrm, &[_]c_int{ 1, t, h, wd, c }, s);
    defer _ = mlx.mlx_array_free(back);
    const sc = try mulA(back, w, s);
    defer _ = mlx.mlx_array_free(sc);
    return addA(sc, b, s);
}

/// diffusers' `Downsample2D` asymmetric pad: `(0,1)` on H and on W — i.e. one
/// row at the BOTTOM and one column at the RIGHT, nothing at top/left. Applied
/// only when the conv's own spatial padding is 0, which is the encoder's case
/// (`downsample_padding=0`).
fn padBottomRight(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const axes = [_]c_int{ 2, 3 }; // H, W in NTHWC
    const lo = [_]c_int{ 0, 0 };
    const hi = [_]c_int{ 1, 1 };
    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_pad(
        &o,
        x,
        &axes,
        axes.len,
        &lo,
        lo.len,
        &hi,
        hi.len,
        zero,
        "constant",
        s,
    ));
    return o;
}

// ════════════════════════════════════════════════════════════════════════
// Modules
// ════════════════════════════════════════════════════════════════════════

const Resnet = struct {
    norm1_w: mlx.mlx_array,
    norm1_b: mlx.mlx_array,
    conv1_w: mlx.mlx_array,
    conv1_b: mlx.mlx_array,
    norm2_w: mlx.mlx_array,
    norm2_b: mlx.mlx_array,
    conv2_w: mlx.mlx_array,
    conv2_b: mlx.mlx_array,
    /// `conv_shortcut`, present only where the block changes channel count.
    /// `.ctx == null` means the identity path.
    short_w: mlx.mlx_array = .{ .ctx = null },
    short_b: mlx.mlx_array = .{ .ctx = null },

    fn load(a: std.mem.Allocator, w: *const Weights, prefix: []const u8, with_shortcut: bool, s: S) !Resnet {
        var r: Resnet = .{
            .norm1_w = try loadVec(w, a, "{s}.norm1.weight", .{prefix}, s),
            .norm1_b = try loadVec(w, a, "{s}.norm1.bias", .{prefix}, s),
            .conv1_w = try loadConvW(w, a, "{s}.conv1.weight", .{prefix}, s),
            .conv1_b = try loadVec(w, a, "{s}.conv1.bias", .{prefix}, s),
            .norm2_w = try loadVec(w, a, "{s}.norm2.weight", .{prefix}, s),
            .norm2_b = try loadVec(w, a, "{s}.norm2.bias", .{prefix}, s),
            .conv2_w = try loadConvW(w, a, "{s}.conv2.weight", .{prefix}, s),
            .conv2_b = try loadVec(w, a, "{s}.conv2.bias", .{prefix}, s),
        };
        if (with_shortcut) {
            r.short_w = try loadConvW(w, a, "{s}.conv_shortcut.weight", .{prefix}, s);
            r.short_b = try loadVec(w, a, "{s}.conv_shortcut.bias", .{prefix}, s);
        }
        return r;
    }

    fn deinit(r: *Resnet) void {
        for ([_]*mlx.mlx_array{
            &r.norm1_w, &r.norm1_b, &r.conv1_w, &r.conv1_b,
            &r.norm2_w, &r.norm2_b, &r.conv2_w, &r.conv2_b,
            &r.short_w, &r.short_b,
        }) |p| if (p.ctx != null) {
            _ = mlx.mlx_array_free(p.*);
        };
    }

    /// norm1 -> silu -> conv1 -> norm2 -> silu -> conv2, plus the (optionally
    /// projected) residual. `output_scale_factor` is 1 throughout this VAE, so
    /// the reference's divide is omitted rather than multiplied by one.
    fn forward(r: *const Resnet, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const n1 = try evalA(try groupNormPerFrame(x, r.norm1_w, r.norm1_b, s));
        const a1 = try evalA(try siluA(n1, s));
        _ = mlx.mlx_array_free(n1);
        const c1 = try evalA(try causalConv3d(a1, r.conv1_w, r.conv1_b, .{ 1, 1, 1 }, 1, 1, s));
        _ = mlx.mlx_array_free(a1);

        const n2 = try evalA(try groupNormPerFrame(c1, r.norm2_w, r.norm2_b, s));
        const a2 = try evalA(try siluA(n2, s));
        _ = mlx.mlx_array_free(n2);
        const c2 = try evalA(try causalConv3d(a2, r.conv2_w, r.conv2_b, .{ 1, 1, 1 }, 1, 1, s));
        _ = mlx.mlx_array_free(a2);
        _ = mlx.mlx_array_free(c1);
        defer _ = mlx.mlx_array_free(c2);

        if (r.short_w.ctx == null) return evalA(try addA(x, c2, s));
        // 1x1x1 conv: no time mixing, so no head extension and no padding.
        const sh = try evalA(try causalConv3d(x, r.short_w, r.short_b, .{ 1, 1, 1 }, 0, 0, s));
        defer _ = mlx.mlx_array_free(sh);
        return evalA(try addA(sh, c2, s));
    }
};

/// How many bytes ONE query tile of the mid-block score matrix may cost.
///
/// The scores are `[T, q_chunk, HW]` f32 and the `precise=true` softmax keeps
/// its own f32 copy, so a tile costs this twice over. 256 MiB therefore peaks
/// around half a gigabyte of transient whatever the image size, against the
/// 17 GB a 2048x2048 restore asked for unchunked.
///
/// Not a tuning knob so much as a ceiling: the chunk loop's only cost is one
/// extra matmul dispatch per tile, and at the sizes that matter the tile count
/// is in the dozens.
pub const ATTN_SCORE_TILE_BYTES: u64 = 256 * 1024 * 1024;

/// Query rows per attention tile for a key axis of `key_len`.
///
/// Returns `key_len` itself whenever the whole matrix already fits, so the
/// geometries that worked before this existed take the identical single-tile
/// path. Floored at 1: a key axis long enough that ONE row busts the budget
/// must still make progress rather than divide to zero.
pub fn attnQueryChunk(key_len: u32) u32 {
    const row_bytes: u64 = @as(u64, key_len) * 4;
    if (row_bytes == 0) return 1;
    const rows = ATTN_SCORE_TILE_BYTES / row_bytes;
    if (rows == 0) return 1;
    return @intCast(@min(rows, @as(u64, key_len)));
}

/// One `[T, q_rows, C] x [T, C, HW] -> [T, q_rows, C]` attention tile:
/// scaled scores, `precise=true` softmax (the reference's
/// `upcast_softmax=True`), and the value gather. Whole-matrix and tiled paths
/// share it so the two cannot drift in scale or softmax precision.
fn attendTile(q: mlx.mlx_array, kt: mlx.mlx_array, v: mlx.mlx_array, scale: mlx.mlx_array, s: S) !mlx.mlx_array {
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_matmul(&scores, q, kt, s));
    const sc = try mulA(scores, scale, s);
    defer _ = mlx.mlx_array_free(sc);
    var pr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pr);
    try mlx.check(mlx.mlx_softmax_axis(&pr, sc, -1, true, s));
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_matmul(&o, pr, v, s));
    return o;
}

/// Single-head spatial self-attention, applied PER FRAME. `heads =
/// in_channels // attention_head_dim = 512 // 512 = 1`, `residual_connection`,
/// `rescale_output_factor = 1`, and its own GroupNorm(32).
const Attention = struct {
    /// Only for the query-tile list in `forward` — the weights themselves are
    /// mlx-owned handles.
    alloc: std.mem.Allocator,
    gn_w: mlx.mlx_array,
    gn_b: mlx.mlx_array,
    q_wt: mlx.mlx_array,
    q_b: mlx.mlx_array,
    k_wt: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_wt: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_wt: mlx.mlx_array,
    o_b: mlx.mlx_array,

    fn load(a: std.mem.Allocator, w: *const Weights, prefix: []const u8, s: S) !Attention {
        return .{
            .alloc = a,
            .gn_w = try loadVec(w, a, "{s}.group_norm.weight", .{prefix}, s),
            .gn_b = try loadVec(w, a, "{s}.group_norm.bias", .{prefix}, s),
            .q_wt = try loadLinT(w, a, "{s}.to_q.weight", .{prefix}, s),
            .q_b = try loadVec(w, a, "{s}.to_q.bias", .{prefix}, s),
            .k_wt = try loadLinT(w, a, "{s}.to_k.weight", .{prefix}, s),
            .k_b = try loadVec(w, a, "{s}.to_k.bias", .{prefix}, s),
            .v_wt = try loadLinT(w, a, "{s}.to_v.weight", .{prefix}, s),
            .v_b = try loadVec(w, a, "{s}.to_v.bias", .{prefix}, s),
            .o_wt = try loadLinT(w, a, "{s}.to_out.0.weight", .{prefix}, s),
            .o_b = try loadVec(w, a, "{s}.to_out.0.bias", .{prefix}, s),
        };
    }

    fn deinit(at: *Attention) void {
        for ([_]*mlx.mlx_array{
            &at.gn_w, &at.gn_b, &at.q_wt, &at.q_b, &at.k_wt,
            &at.k_b,  &at.v_wt, &at.v_b,  &at.o_wt, &at.o_b,
        }) |p| _ = mlx.mlx_array_free(p.*);
    }

    fn forward(at: *const Attention, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        const shp = mlx.getShape(x);
        const t = shp[1];
        const h = shp[2];
        const wd = shp[3];
        const c = shp[4];

        // The attention's own GroupNorm is per-frame like every other norm here.
        const gn = try groupNormPerFrame(x, at.gn_w, at.gn_b, s);
        defer _ = mlx.mlx_array_free(gn);
        // Fold every frame into its own sequence of H*W tokens. Frames never
        // attend to each other — this block is spatial only.
        const seq = try reshape(gn, &[_]c_int{ t, h * wd, c }, s);
        defer _ = mlx.mlx_array_free(seq);

        const q = try evalA(try linT(seq, at.q_wt, at.q_b, s));
        defer _ = mlx.mlx_array_free(q);
        const k = try evalA(try linT(seq, at.k_wt, at.k_b, s));
        defer _ = mlx.mlx_array_free(k);
        const v = try evalA(try linT(seq, at.v_wt, at.v_b, s));
        defer _ = mlx.mlx_array_free(v);

        // Single head: [T, HW, C] is already [batch, seq, head_dim].
        const kt = try transpose(k, &[_]c_int{ 0, 2, 1 }, s);
        defer _ = mlx.mlx_array_free(kt);
        const scale = mlx.mlx_array_new_float(1.0 / @sqrt(@as(f32, @floatFromInt(c))));
        defer _ = mlx.mlx_array_free(scale);

        // QUERY-TILED. The score matrix is `HW x HW` and HW is quadratic in
        // pixel area, so an unchunked mid block asks for 17 GB at 2048x2048
        // and lands on the Metal working-set edge (where MLX returns zeros
        // rather than failing) well before that. Softmax runs along the KEY
        // axis, so tiling the QUERY axis is exact — see the chunk test.
        const seq_len: u32 = @intCast(h * wd);
        const chunk = attnQueryChunk(seq_len);
        const ctx = if (chunk >= seq_len)
            try attendTile(q, kt, v, scale, s)
        else blk: {
            var parts: std.ArrayList(mlx.mlx_array) = .empty;
            defer {
                for (parts.items) |pt| _ = mlx.mlx_array_free(pt);
                parts.deinit(at.alloc);
            }
            var lo: u32 = 0;
            while (lo < seq_len) : (lo += chunk) {
                const hi = @min(lo + chunk, seq_len);
                const qs = try sliceAxis(q, 1, @intCast(lo), @intCast(hi), s);
                defer _ = mlx.mlx_array_free(qs);
                const part = try attendTile(qs, kt, v, scale, s);
                errdefer _ = mlx.mlx_array_free(part);
                try parts.append(at.alloc, part);
                // Every tile is its own shape at the ragged tail, and the
                // pool otherwise holds one 256 MiB score tile per iteration
                // for the whole block.
                try mlx.check(mlx.mlx_array_eval(part));
                _ = mlx.mlx_clear_cache();
            }
            break :blk try concat(parts.items, 1, s);
        };
        defer _ = mlx.mlx_array_free(ctx);

        const out = try evalA(try linT(ctx, at.o_wt, at.o_b, s));
        defer _ = mlx.mlx_array_free(out);
        const back = try reshape(out, &[_]c_int{ 1, t, h, wd, c }, s);
        defer _ = mlx.mlx_array_free(back);
        // residual_connection=True, rescale_output_factor=1.
        return evalA(try addA(back, x, s));
    }
};

fn linT(x: mlx.mlx_array, wt: mlx.mlx_array, bias: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, x, wt, s));
    defer _ = mlx.mlx_array_free(o);
    return addA(o, bias, s);
}

const DownBlock = struct {
    resnets: [2]Resnet,
    down_w: mlx.mlx_array = .{ .ctx = null },
    down_b: mlx.mlx_array = .{ .ctx = null },
    temporal_down: bool = false,

    fn deinit(d: *DownBlock) void {
        for (&d.resnets) |*r| r.deinit();
        if (d.down_w.ctx != null) _ = mlx.mlx_array_free(d.down_w);
        if (d.down_b.ctx != null) _ = mlx.mlx_array_free(d.down_b);
    }

    fn forward(d: *const DownBlock, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var cur = try contig(x, s);
        for (&d.resnets) |*r| {
            const nxt = try r.forward(cur, s);
            _ = mlx.mlx_array_free(cur);
            cur = nxt;
            // `temporal_modules` in the reference is Identity for this
            // checkpoint (it ships no such tensors), so nothing runs here.
        }
        if (d.down_w.ctx == null) return cur;
        defer _ = mlx.mlx_array_free(cur);
        // Spatial padding lives OUTSIDE the conv and is asymmetric.
        const padded = try evalA(try padBottomRight(cur, s));
        defer _ = mlx.mlx_array_free(padded);
        const tp: u32 = if (d.temporal_down) 1 else 0;
        const st: c_int = if (d.temporal_down) 2 else 1;
        return evalA(try causalConv3d(padded, d.down_w, d.down_b, .{ st, 2, 2 }, tp, 0, s));
    }
};

pub const Encoder = struct {
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    blocks: [4]DownBlock,
    mid_r0: Resnet,
    mid_attn: Attention,
    mid_r1: Resnet,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(e: *Encoder) void {
        _ = mlx.mlx_array_free(e.conv_in_w);
        _ = mlx.mlx_array_free(e.conv_in_b);
        for (&e.blocks) |*b| b.deinit();
        e.mid_r0.deinit();
        e.mid_attn.deinit();
        e.mid_r1.deinit();
        _ = mlx.mlx_array_free(e.norm_out_w);
        _ = mlx.mlx_array_free(e.norm_out_b);
        _ = mlx.mlx_array_free(e.conv_out_w);
        _ = mlx.mlx_array_free(e.conv_out_b);
    }
};

pub fn loadEncoder(a: std.mem.Allocator, w: *const Weights, s: S) !Encoder {
    const cfg: shape.Config = .{};
    var e: Encoder = undefined;
    e.conv_in_w = try loadConvW(w, a, "encoder.conv_in.weight", .{}, s);
    e.conv_in_b = try loadVec(w, a, "encoder.conv_in.bias", .{}, s);

    var bi: u32 = 0;
    while (bi < 4) : (bi += 1) {
        var blk: DownBlock = .{ .resnets = undefined };
        var ri: u32 = 0;
        while (ri < 2) : (ri += 1) {
            const prefix = try std.fmt.allocPrint(a, "encoder.down_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer a.free(prefix);
            // The shortcut exists only on the FIRST resnet of a block that
            // changes width — resnet 1 always runs at the block's output width.
            const with_short = ri == 0 and shape.hasConvShortcut(cfg, bi);
            blk.resnets[ri] = try Resnet.load(a, w, prefix, with_short, s);
        }
        if (shape.addDownsample(cfg, bi)) {
            blk.down_w = try loadConvW(w, a, "encoder.down_blocks.{d}.downsamplers.0.conv.weight", .{bi}, s);
            blk.down_b = try loadVec(w, a, "encoder.down_blocks.{d}.downsamplers.0.conv.bias", .{bi}, s);
            blk.temporal_down = shape.temporalDownEffective(cfg, bi);
        }
        e.blocks[bi] = blk;
    }

    e.mid_r0 = try Resnet.load(a, w, "encoder.mid_block.resnets.0", false, s);
    e.mid_attn = try Attention.load(a, w, "encoder.mid_block.attentions.0", s);
    e.mid_r1 = try Resnet.load(a, w, "encoder.mid_block.resnets.1", false, s);

    e.norm_out_w = try loadVec(w, a, "encoder.conv_norm_out.weight", .{}, s);
    e.norm_out_b = try loadVec(w, a, "encoder.conv_norm_out.bias", .{}, s);
    e.conv_out_w = try loadConvW(w, a, "encoder.conv_out.weight", .{}, s);
    e.conv_out_b = try loadVec(w, a, "encoder.conv_out.bias", .{}, s);
    return e;
}

/// `remove_head`: keep frame 0 and drop the next `times` frames —
/// `cat(x[:, :1], x[:, times+1:])`. The decoder's mirror of `extendHead`.
///
/// After a temporal upsample the head carries `times` frames that only exist
/// because the encoder replicated frame 0 on the way in. Dropping them is what
/// makes the round trip land on the original length: a latent of T frames
/// becomes `2T` and then `2T - 1`, which is why the pixel timeline is 4k+1 and
/// not 4k. A decoder that skips this is one frame too long at every temporal
/// stage and the error compounds.
fn removeHead(x: mlx.mlx_array, times: u32, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x);
    const t = shp[1];
    const first = try sliceAxis(x, 1, 0, 1, s);
    // A single-frame input has nothing after the head to keep.
    if (t <= @as(c_int, @intCast(times + 1))) return first;
    defer _ = mlx.mlx_array_free(first);
    const rest = try sliceAxis(x, 1, @intCast(times + 1), t, s);
    defer _ = mlx.mlx_array_free(rest);
    return concat(&[_]mlx.mlx_array{ first, rest }, 1, s);
}

/// MAGViT-v2 style upsample: a 1x1x1 conv fans the channels out by
/// `spatial^2 * temporal`, then a pixel shuffle moves them into space/time.
///
/// The channel decomposition is `(x y z c)` — x is the H offset and the
/// SLOWEST axis, then y (W offset), then z (T offset), and c is fastest. The
/// reference's einops is
///   `b (x y z c) f h w -> b c (f z) (h x) (w y)`
/// so a port that reads the channel block as `(c z y x)` — the more natural
/// C-fastest-last ordering — shuffles the right number of elements into the
/// wrong places and still produces an image.
fn pixelShuffleUp(x: mlx.mlx_array, spatial: c_int, temporal: c_int, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(x); // [1, T, H, W, x*y*z*C]
    const t = shp[1];
    const h = shp[2];
    const w = shp[3];
    const packed_c = shp[4];
    const c = @divExact(packed_c, spatial * spatial * temporal);

    // [1,T,H,W,(x y z c)] -> [1,T,H,W,x,y,z,c]
    const split = try reshape(x, &[_]c_int{ 1, t, h, w, spatial, spatial, temporal, c }, s);
    defer _ = mlx.mlx_array_free(split);
    // -> [1, T, z, H, x, W, y, c] so each output axis is (coarse, fine).
    //      0  1  6  2  4  3  5  7
    const perm = try transpose(split, &[_]c_int{ 0, 1, 6, 2, 4, 3, 5, 7 }, s);
    defer _ = mlx.mlx_array_free(perm);
    const permc = try contig(perm, s);
    defer _ = mlx.mlx_array_free(permc);
    return reshape(permc, &[_]c_int{ 1, t * temporal, h * spatial, w * spatial, c }, s);
}

const Upsampler = struct {
    up_w: mlx.mlx_array,
    up_b: mlx.mlx_array,
    conv_w: mlx.mlx_array,
    conv_b: mlx.mlx_array,
    temporal_up: bool,

    fn deinit(u: *Upsampler) void {
        for ([_]*mlx.mlx_array{ &u.up_w, &u.up_b, &u.conv_w, &u.conv_b }) |p|
            _ = mlx.mlx_array_free(p.*);
    }

    fn forward(u: *const Upsampler, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        // 1x1x1: mixes no time, so no head extension and no padding.
        const fan = try evalA(try causalConv3d(x, u.up_w, u.up_b, .{ 1, 1, 1 }, 0, 0, s));
        defer _ = mlx.mlx_array_free(fan);
        const tr: c_int = if (u.temporal_up) 2 else 1;
        // The shuffle is where the frame QUADRUPLES, so it is the single
        // biggest boundary in the decoder to cut the graph at.
        const sh = try evalA(try pixelShuffleUp(fan, 2, tr, s));
        defer _ = mlx.mlx_array_free(sh);
        if (!u.temporal_up) return evalA(try causalConv3d(sh, u.conv_w, u.conv_b, .{ 1, 1, 1 }, 1, 1, s));
        const cut = try evalA(try removeHead(sh, 1, s));
        defer _ = mlx.mlx_array_free(cut);
        return evalA(try causalConv3d(cut, u.conv_w, u.conv_b, .{ 1, 1, 1 }, 1, 1, s));
    }
};

const UpBlock = struct {
    /// `layers_per_block + 1` — the decoder runs THREE resnets per block where
    /// the encoder runs two.
    resnets: [3]Resnet,
    up: ?Upsampler = null,

    fn deinit(u: *UpBlock) void {
        for (&u.resnets) |*r| r.deinit();
        if (u.up) |*p| p.deinit();
    }

    fn forward(u: *const UpBlock, x: mlx.mlx_array, s: S) !mlx.mlx_array {
        var cur = try contig(x, s);
        for (&u.resnets) |*r| {
            const nxt = try r.forward(cur, s);
            _ = mlx.mlx_array_free(cur);
            cur = nxt;
        }
        if (u.up) |*p| {
            defer _ = mlx.mlx_array_free(cur);
            return p.forward(cur, s);
        }
        return cur;
    }
};

pub const Decoder = struct {
    conv_in_w: mlx.mlx_array,
    conv_in_b: mlx.mlx_array,
    mid_r0: Resnet,
    mid_attn: Attention,
    mid_r1: Resnet,
    blocks: [4]UpBlock,
    norm_out_w: mlx.mlx_array,
    norm_out_b: mlx.mlx_array,
    conv_out_w: mlx.mlx_array,
    conv_out_b: mlx.mlx_array,

    pub fn deinit(d: *Decoder) void {
        _ = mlx.mlx_array_free(d.conv_in_w);
        _ = mlx.mlx_array_free(d.conv_in_b);
        d.mid_r0.deinit();
        d.mid_attn.deinit();
        d.mid_r1.deinit();
        for (&d.blocks) |*b| b.deinit();
        _ = mlx.mlx_array_free(d.norm_out_w);
        _ = mlx.mlx_array_free(d.norm_out_b);
        _ = mlx.mlx_array_free(d.conv_out_w);
        _ = mlx.mlx_array_free(d.conv_out_b);
    }
};

/// Decoder channel ladder: `reversed(block_out_channels)` = [512,512,256,128].
/// Block `i` takes the previous block's output and emits `reversed[i]`, so the
/// shortcut appears wherever those differ (blocks 2 and 3).
fn decoderBlockIn(cfg: shape.Config, i: u32) u32 {
    const n = cfg.numBlocks();
    if (i == 0) return cfg.block_out_channels[n - 1];
    return cfg.block_out_channels[n - 1 - (i - 1)];
}

fn decoderBlockOut(cfg: shape.Config, i: u32) u32 {
    return cfg.block_out_channels[cfg.numBlocks() - 1 - i];
}

/// `add_upsample = not is_final_block` — blocks 0,1,2.
fn decoderHasUpsample(cfg: shape.Config, i: u32) bool {
    return i != cfg.numBlocks() - 1;
}

/// `is_temporal_up = i < temporal_up_num` — blocks 0,1. Note this is a PLAIN
/// index test, unlike the encoder's trailing-window predicate, and it is NOT
/// the mirror image of `temporalDownEffective`. Deriving one from the other
/// puts the temporal upsample on the wrong blocks.
fn decoderTemporalUp(cfg: shape.Config, i: u32) bool {
    return i < cfg.temporal_down_num;
}

pub fn loadDecoder(a: std.mem.Allocator, w: *const Weights, s: S) !Decoder {
    const cfg: shape.Config = .{};
    var d: Decoder = undefined;
    d.conv_in_w = try loadConvW(w, a, "decoder.conv_in.weight", .{}, s);
    d.conv_in_b = try loadVec(w, a, "decoder.conv_in.bias", .{}, s);

    d.mid_r0 = try Resnet.load(a, w, "decoder.mid_block.resnets.0", false, s);
    d.mid_attn = try Attention.load(a, w, "decoder.mid_block.attentions.0", s);
    d.mid_r1 = try Resnet.load(a, w, "decoder.mid_block.resnets.1", false, s);

    var bi: u32 = 0;
    while (bi < 4) : (bi += 1) {
        var blk: UpBlock = .{ .resnets = undefined };
        var ri: u32 = 0;
        while (ri < 3) : (ri += 1) {
            const prefix = try std.fmt.allocPrint(a, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri });
            defer a.free(prefix);
            const with_short = ri == 0 and decoderBlockIn(cfg, bi) != decoderBlockOut(cfg, bi);
            blk.resnets[ri] = try Resnet.load(a, w, prefix, with_short, s);
        }
        if (decoderHasUpsample(cfg, bi)) {
            blk.up = .{
                .up_w = try loadConvW(w, a, "decoder.up_blocks.{d}.upsamplers.0.upscale_conv.weight", .{bi}, s),
                .up_b = try loadVec(w, a, "decoder.up_blocks.{d}.upsamplers.0.upscale_conv.bias", .{bi}, s),
                .conv_w = try loadConvW(w, a, "decoder.up_blocks.{d}.upsamplers.0.conv.weight", .{bi}, s),
                .conv_b = try loadVec(w, a, "decoder.up_blocks.{d}.upsamplers.0.conv.bias", .{bi}, s),
                .temporal_up = decoderTemporalUp(cfg, bi),
            };
        }
        d.blocks[bi] = blk;
    }

    d.norm_out_w = try loadVec(w, a, "decoder.conv_norm_out.weight", .{}, s);
    d.norm_out_b = try loadVec(w, a, "decoder.conv_norm_out.bias", .{}, s);
    d.conv_out_w = try loadConvW(w, a, "decoder.conv_out.weight", .{}, s);
    d.conv_out_b = try loadVec(w, a, "decoder.conv_out.bias", .{}, s);
    return d;
}

/// Decode a latent `[1, Tz, Hz, Wz, 16]` back to pixels `[1, T, H, W, 3]`.
/// The `scaling_factor` is NOT undone here — pass the same representation
/// `encode` produced.
pub fn decode(d: *const Decoder, z: mlx.mlx_array, s: S) !mlx.mlx_array {
    memProbe("decode enter");
    var cur = try evalA(try causalConv3d(z, d.conv_in_w, d.conv_in_b, .{ 1, 1, 1 }, 1, 1, s));
    {
        const r0 = try d.mid_r0.forward(cur, s);
        _ = mlx.mlx_array_free(cur);
        const at = try evalA(try d.mid_attn.forward(r0, s));
        _ = mlx.mlx_array_free(r0);
        const r1 = try d.mid_r1.forward(at, s);
        _ = mlx.mlx_array_free(at);
        cur = r1;
        try mlx.check(mlx.mlx_array_eval(cur));
        _ = mlx.mlx_clear_cache();
        memProbe("decode mid");
    }
    for (&d.blocks) |*b| {
        const nxt = try b.forward(cur, s);
        _ = mlx.mlx_array_free(cur);
        cur = nxt;
        try mlx.check(mlx.mlx_array_eval(cur));
        _ = mlx.mlx_clear_cache();
        memProbe("decode block");
    }
    const n = try evalA(try groupNormPerFrame(cur, d.norm_out_w, d.norm_out_b, s));
    _ = mlx.mlx_array_free(cur);
    defer _ = mlx.mlx_array_free(n);
    const act = try evalA(try siluA(n, s));
    defer _ = mlx.mlx_array_free(act);
    return evalA(try causalConv3d(act, d.conv_out_w, d.conv_out_b, .{ 1, 1, 1 }, 1, 1, s));
}

/// Encode pixels to the RAW latent parameters.
///
/// Input `[1, T, H, W, 3]` in the reference's own range; output
/// `[1, Tz, Hz, Wz, 32]` — 16 mean channels followed by 16 logvar channels
/// (`double_z=True`, and there is no quant_conv in this VAE). The
/// `scaling_factor` is deliberately NOT applied here: the caller decides
/// whether it wants the distribution or the scaled sample.
pub fn encode(e: *const Encoder, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    memProbe("encode enter");
    var cur = try evalA(try causalConv3d(x, e.conv_in_w, e.conv_in_b, .{ 1, 1, 1 }, 1, 1, s));

    for (&e.blocks) |*b| {
        const nxt = try b.forward(cur, s);
        _ = mlx.mlx_array_free(cur);
        cur = nxt;
        try mlx.check(mlx.mlx_array_eval(cur));
        _ = mlx.mlx_clear_cache();
        memProbe("encode block");
    }

    {
        const r0 = try e.mid_r0.forward(cur, s);
        _ = mlx.mlx_array_free(cur);
        const at = try evalA(try e.mid_attn.forward(r0, s));
        _ = mlx.mlx_array_free(r0);
        const r1 = try e.mid_r1.forward(at, s);
        _ = mlx.mlx_array_free(at);
        cur = r1;
    }

    const n = try evalA(try groupNormPerFrame(cur, e.norm_out_w, e.norm_out_b, s));
    _ = mlx.mlx_array_free(cur);
    defer _ = mlx.mlx_array_free(n);
    const act = try evalA(try siluA(n, s));
    defer _ = mlx.mlx_array_free(act);
    return evalA(try causalConv3d(act, e.conv_out_w, e.conv_out_b, .{ 1, 1, 1 }, 1, 1, s));
}

/// The mean half of `encode`'s output — the first `latent_channels` channels.
pub fn latentMean(z: mlx.mlx_array, latent_channels: c_int, s: S) !mlx.mlx_array {
    return sliceAxis(z, 4, 0, latent_channels, s);
}

// ════════════════════════════════════════════════════════════════════════
// Tests. The numeric ones are gated on SEEDVR2_FIXTURES + SEEDVR2_VAE, since
// they need the real 478 MB checkpoint and reference activations produced by
//   tests/dump_seedvr2_fixtures.py vae
// A missing fixture SKIPS loudly rather than passing quietly.
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "extendHeadTimes agrees with the shape module" {
    // The head extension is the causality mechanism and its size is derived in
    // exactly one place. This test exists so the two files cannot drift.
    try testing.expectEqual(@as(u32, 2), shape.extendHeadTimes(1));
    try testing.expectEqual(@as(u32, 0), shape.extendHeadTimes(0));
}

/// Pull a fixture tensor (written by `tests/dump_seedvr2_fixtures.py vae`) and
/// return it as NTHWC `[1, T, H, W, C]`.
///
/// The reference is channels-first and this encoder is channels-last; the
/// permutation lives HERE, in the test reader, so the hot path never permutes
/// and a layout bug cannot hide behind a compensating transpose.
///
/// Single-frame fixtures come back 4-D — the reference squeezes T when it is 1
/// — so the rank is normalised before permuting.
fn fixtureNTHWC(fx: *const Weights, name: []const u8, s: S) !mlx.mlx_array {
    const raw = try ownWeight(fx, name);
    defer _ = mlx.mlx_array_free(raw);
    const shp = mlx.getShape(raw);
    var five = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(five);
    if (shp.len == 5) {
        try mlx.check(mlx.mlx_array_set(&five, raw));
    } else if (shp.len == 4) {
        // [B, C, H, W] -> [B, C, 1, H, W]
        const r = try reshape(raw, &[_]c_int{ shp[0], shp[1], 1, shp[2], shp[3] }, s);
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_array_set(&five, r));
    } else {
        log.err("[seedvr2-vae] fixture {s} has rank {d}\n", .{ name, shp.len });
        return error.FixtureShapeMismatch;
    }
    const tr = try transpose(five, &[_]c_int{ 0, 2, 3, 4, 1 }, s);
    defer _ = mlx.mlx_array_free(tr);
    return contig(tr, s);
}

fn cosineSim(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: S) !f32 {
    const dot = try mulA(a_arr, b_arr, s);
    defer _ = mlx.mlx_array_free(dot);
    const aa = try mulA(a_arr, a_arr, s);
    defer _ = mlx.mlx_array_free(aa);
    const bb = try mulA(b_arr, b_arr, s);
    defer _ = mlx.mlx_array_free(bb);
    var sd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sd);
    var sa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sa);
    var sb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sb);
    try mlx.check(mlx.mlx_sum(&sd, dot, false, s));
    try mlx.check(mlx.mlx_sum(&sa, aa, false, s));
    try mlx.check(mlx.mlx_sum(&sb, bb, false, s));
    try mlx.check(mlx.mlx_array_eval(sd));
    try mlx.check(mlx.mlx_array_eval(sa));
    try mlx.check(mlx.mlx_array_eval(sb));
    var d: f32 = 0;
    var na: f32 = 0;
    var nb: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&d, sd));
    try mlx.check(mlx.mlx_array_item_float32(&na, sa));
    try mlx.check(mlx.mlx_array_item_float32(&nb, sb));
    return d / (@sqrt(na) * @sqrt(nb) + 1e-12);
}

/// RMS ratio. A cosine test is BLIND to a uniform scale error — the project
/// has been bitten by exactly that (`docs/gotchas/models-media.md`), so any
/// activation compared by cosine reports its magnitude beside it.
fn rmsRatio(a_arr: mlx.mlx_array, b_arr: mlx.mlx_array, s: S) !f32 {
    const aa = try mulA(a_arr, a_arr, s);
    defer _ = mlx.mlx_array_free(aa);
    const bb = try mulA(b_arr, b_arr, s);
    defer _ = mlx.mlx_array_free(bb);
    var sa = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sa);
    var sb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sb);
    try mlx.check(mlx.mlx_mean(&sa, aa, false, s));
    try mlx.check(mlx.mlx_mean(&sb, bb, false, s));
    try mlx.check(mlx.mlx_array_eval(sa));
    try mlx.check(mlx.mlx_array_eval(sb));
    var na: f32 = 0;
    var nb: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&na, sa));
    try mlx.check(mlx.mlx_array_item_float32(&nb, sb));
    return @sqrt(na) / (@sqrt(nb) + 1e-12);
}

test "seedvr2 vae live: encoder matches the reference moments" {
    // PARITY. Needs the real checkpoint and the reference activations:
    //   SEEDVR2_VAE=~/.mlx-serve/downloads/seedvr2-3b/ema_vae_fp16.safetensors \
    //   SEEDVR2_FIXTURES=tests/fixtures/seedvr2 \
    //   zig build test -Dtest-filter="seedvr2 vae live"
    //
    // The target is `latent_dist.parameters` — the raw 32-channel conv_out
    // result — NOT `encode().latent`, which is a stochastic sample and cannot
    // be matched by any correct port.
    const a = testing.allocator;
    const vae_path = std.mem.span(std.c.getenv("SEEDVR2_VAE") orelse return error.SkipZigTest);
    const fx = std.mem.span(std.c.getenv("SEEDVR2_FIXTURES") orelse return error.SkipZigTest);

    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var w = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer w.deinit();
    var enc = try loadEncoder(a, &w, s);
    defer enc.deinit();

    const fx_path = try std.fmt.allocPrint(a, "{s}/vae_fixture.safetensors", .{fx});
    defer a.free(fx_path);
    var fxw = try model_mod.loadWeightsSingleFile(a, fx_path);
    defer fxw.deinit();

    // WHY THESE FOUR, measured rather than guessed — each was verified by
    // breaking the encoder and watching which cases went red:
    //
    //   * MULTI-FRAME is mandatory. Making GroupNorm reduce over time as well
    //     (the natural 3D reading) leaves `1x64x64` at cos=1.000000 — with
    //     T=1 there is nothing to reduce over — and only `5x*` catches it at
    //     0.9986. A still-image-only fixture set cannot see that trap at all.
    //   * NON-SQUARE is mandatory: an H/W axis swap is invisible on 64x64.
    //   * Two spatial sizes exercise a different number of ladder steps.
    //
    // For reference, symmetric downsample padding instead of (0,1,0,1) shows
    // up on every case (cos 0.9914, rms 1.0342) — that one is cheap to catch.
    const cases = [_][]const u8{
        "1x64x64",
        "5x64x64",
        "1x128x128",
        "5x128x96",
    };

    for (cases) |tag| {
        const in_name = try std.fmt.allocPrint(a, "in_{s}", .{tag});
        defer a.free(in_name);
        const ref_name = try std.fmt.allocPrint(a, "moments_{s}", .{tag});
        defer a.free(ref_name);

        const x = try fixtureNTHWC(&fxw, in_name, s);
        defer _ = mlx.mlx_array_free(x);
        const ref = try fixtureNTHWC(&fxw, ref_name, s);
        defer _ = mlx.mlx_array_free(ref);

        const got = try encode(&enc, x, s);
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(got));

        // Shape first: a cosine over mismatched shapes is meaningless, and the
        // shape is what `seedvr2_vae_shape.zig` predicts independently.
        const gs = mlx.getShape(got);
        const rs = mlx.getShape(ref);
        testing.expectEqualSlices(c_int, rs, gs) catch |err| {
            std.debug.print("{s}: shape {any} != reference {any}\n", .{ tag, gs, rs });
            return err;
        };

        const cos = try cosineSim(got, ref, s);
        const rms = try rmsRatio(got, ref, s);
        std.debug.print("[seedvr2-vae] {s}: cos={d:.6} rms_ratio={d:.4}\n", .{ tag, cos, rms });

        // FINITENESS IS CHECKED BY THE COSINE ITSELF: a NaN anywhere makes
        // `cos` NaN, and `NaN > 0.999` is false, so the bar below fails rather
        // than passing vacuously. That is the only reason a separate isnan
        // sweep is not here.
        testing.expect(cos > 0.999) catch |err| {
            std.debug.print("{s}: cosine {d:.6} below 0.999\n", .{ tag, cos });
            return err;
        };
        // The scale check a cosine CANNOT make — a uniform gain error scores a
        // perfect cosine.
        testing.expect(rms > 0.98 and rms < 1.02) catch |err| {
            std.debug.print("{s}: rms_ratio {d:.4} outside [0.98, 1.02]\n", .{ tag, rms });
            return err;
        };
    }
}

test "seedvr2 vae live: decoder matches the reference reconstruction" {
    // PARITY, driven from the SAME fixture file:
    //   -Dseedvr2-fixtures=tests/fixtures/seedvr2
    //   -Dseedvr2-vae=<ema_vae_fp16.safetensors>
    //
    // Input is the posterior MEAN and the target is the reference's decode of
    // that same mean, so the comparison is deterministic end to end — the
    // dumper deliberately does not decode a sample.
    const a = testing.allocator;
    const vae_path = std.mem.span(std.c.getenv("SEEDVR2_VAE") orelse return error.SkipZigTest);
    const fx = std.mem.span(std.c.getenv("SEEDVR2_FIXTURES") orelse return error.SkipZigTest);
    if (vae_path.len == 0 or fx.len == 0) return error.SkipZigTest;

    const s = mlx.gpuStream();

    var w = try model_mod.loadWeightsSingleFile(a, vae_path);
    defer w.deinit();
    var dec = try loadDecoder(a, &w, s);
    defer dec.deinit();

    const fx_path = try std.fmt.allocPrint(a, "{s}/vae_fixture.safetensors", .{fx});
    defer a.free(fx_path);
    var fxw = try model_mod.loadWeightsSingleFile(a, fx_path);
    defer fxw.deinit();

    const cases = [_][]const u8{ "1x64x64", "5x64x64", "1x128x128", "5x128x96" };

    for (cases) |tag| {
        const z_name = try std.fmt.allocPrint(a, "mean_{s}", .{tag});
        defer a.free(z_name);
        const ref_name = try std.fmt.allocPrint(a, "recon_{s}", .{tag});
        defer a.free(ref_name);

        const z = try fixtureNTHWC(&fxw, z_name, s);
        defer _ = mlx.mlx_array_free(z);
        const ref = try fixtureNTHWC(&fxw, ref_name, s);
        defer _ = mlx.mlx_array_free(ref);

        const got = try decode(&dec, z, s);
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(got));

        // The TEMPORAL length is the decoder's own claim and the thing
        // `removeHead` exists to get right: a latent of Tz frames must decode
        // to 4*(Tz-1)+1 pixels frames, not 4*Tz. Checking the shape before the
        // cosine is what turns "one frame too long" into a named failure
        // instead of a mysteriously low correlation.
        const gs = mlx.getShape(got);
        const rs = mlx.getShape(ref);
        testing.expectEqualSlices(c_int, rs, gs) catch |err| {
            std.debug.print("{s}: shape {any} != reference {any}\n", .{ tag, gs, rs });
            return err;
        };

        const cos = try cosineSim(got, ref, s);
        const rms = try rmsRatio(got, ref, s);
        std.debug.print("[seedvr2-vae-dec] {s}: cos={d:.6} rms_ratio={d:.4}\n", .{ tag, cos, rms });

        testing.expect(cos > 0.999) catch |err| {
            std.debug.print("{s}: decoder cosine {d:.6} below 0.999\n", .{ tag, cos });
            return err;
        };
        testing.expect(rms > 0.98 and rms < 1.02) catch |err| {
            std.debug.print("{s}: decoder rms_ratio {d:.4} outside [0.98, 1.02]\n", .{ tag, rms });
            return err;
        };
    }
}

test "seedvr2 vae: the conv layout is read from the checkpoint, not assumed" {
    const a = testing.allocator;
    const s = mlx.gpuStream();

    // The two SeedVR2 converters ship conv weights in DIFFERENT axis orders:
    // ours in PyTorch's `[O, I, kt, kh, kw]`, the mlx-community mirror already
    // in the `[O, kt, kh, kw, I]` that `mlx_conv3d` consumes. Permuting the
    // mirror's again is what produced
    //   [conv] input: (1,1,512,512,128) weight: (128,1,3,128,3)
    // — an uncatchable MLX abort, live 2026-08-21, immediately after the DiT
    // half started loading. 64 of the VAE's tensors are affected.
    const mk = struct {
        fn f(dims: []const c_int, st: S) mlx.mlx_array {
            var o = mlx.mlx_array_new();
            _ = mlx.mlx_zeros(&o, dims.ptr, dims.len, mlx.mlx_dtype.float32, st);
            return o;
        }
    }.f;

    // The probe tensor is unambiguous in a way `encoder.conv_in` is not: an
    // RGB input conv is `[C, 3, 3, 3, 3]` in BOTH layouts, so a per-tensor
    // size heuristic cannot decide it and the choice has to be made once, for
    // the whole checkpoint, from a tensor whose channel count is not 3.
    var ours = Weights.init(a);
    defer ours.deinit();
    try ours.map.put(try a.dupe(u8, "decoder.mid_block.resnets.0.conv1.weight"),
        mk(&[_]c_int{ 512, 512, 3, 3, 3 }, s));
    try testing.expectEqual(ConvLayout.pytorch, ConvLayout.detect(&ours));

    var mirror = Weights.init(a);
    defer mirror.deinit();
    try mirror.map.put(try a.dupe(u8, "decoder.mid_block.resnets.0.conv1.weight"),
        mk(&[_]c_int{ 512, 3, 3, 3, 512 }, s));
    try testing.expectEqual(ConvLayout.mlx, ConvLayout.detect(&mirror));

    // No probe tensor at all ⇒ the layout every load before the mirror
    // existed used, never a guess that silently reshapes someone's weights.
    var empty = Weights.init(a);
    defer empty.deinit();
    try testing.expectEqual(ConvLayout.pytorch, ConvLayout.detect(&empty));
}

test "the decoder ladder mirrors the encoder without being derived from it" {
    // The two temporal predicates are NOT mirror images and must not be
    // written as one. The encoder flags a TRAILING window
    // (`i >= n - temporal_down_num - 1`, true for blocks 1,2,3 with only 1,2
    // effective); the decoder uses a plain LEADING test (`i < temporal_up_num`,
    // blocks 0,1). Confirmed by the checkpoint: up_blocks 0 and 1 carry an
    // upscale_conv of ratio 8 (2^2 * 2) and block 2 carries ratio 4 (2^2 * 1).
    const cfg: shape.Config = .{};
    try testing.expect(decoderTemporalUp(cfg, 0));
    try testing.expect(decoderTemporalUp(cfg, 1));
    try testing.expect(!decoderTemporalUp(cfg, 2));
    try testing.expect(!decoderTemporalUp(cfg, 3));
    // Upsamplers on every block but the last.
    try testing.expect(decoderHasUpsample(cfg, 0));
    try testing.expect(decoderHasUpsample(cfg, 2));
    try testing.expect(!decoderHasUpsample(cfg, 3));
    // Channel ladder [512,512,256,128]: the shortcut appears where in != out.
    try testing.expectEqual(@as(u32, 512), decoderBlockIn(cfg, 0));
    try testing.expectEqual(@as(u32, 512), decoderBlockOut(cfg, 0));
    try testing.expectEqual(@as(u32, 512), decoderBlockOut(cfg, 1));
    try testing.expectEqual(@as(u32, 256), decoderBlockOut(cfg, 2));
    try testing.expectEqual(@as(u32, 128), decoderBlockOut(cfg, 3));
    try testing.expect(decoderBlockIn(cfg, 2) != decoderBlockOut(cfg, 2));
    try testing.expect(decoderBlockIn(cfg, 3) != decoderBlockOut(cfg, 3));
    try testing.expect(decoderBlockIn(cfg, 0) == decoderBlockOut(cfg, 0));
}

test "the mid-block attention chunks its queries so the score tile is bounded" {
    // THE SCORE MATRIX IS QUADRATIC IN PIXEL AREA, and nothing upstream caps
    // it: the mid block runs at latent resolution, so `HW = (px/8)^2` and the
    // `[T, HW, HW]` f32 scores are `HW^2 * 4` bytes. Measured live 2026-08-21
    // on a 24 GB Mac with the 3B resident (12.8 GB):
    //
    //   1024x1024 -> HW 16384 ->  1.07 GB  restored correctly
    //   1536x1536 -> HW 36864 ->  5.44 GB  flat brown rectangle
    //   2048x2048 -> HW 65536 -> 17.18 GB  server DIED on the allocation
    //
    // The 2048 case is the exact byte count in the crash log
    // (`Attempting to allocate 17179869184 bytes`, past Metal's 14.3 GB
    // max buffer). 1536 is the same term one rung down and allocates it
    // TWICE — `precise=true` softmax keeps its own f32 copy — which lands on
    // the Metal working-set edge, where MLX returns ZEROS instead of failing.
    // A degenerate flat image is what that looks like from the outside.
    //
    // Chunking the QUERY axis is EXACT, not an approximation: the softmax
    // runs along the KEY axis, so a query row's output does not depend on
    // which other rows share its tile.
    const budget: u64 = ATTN_SCORE_TILE_BYTES;
    for ([_]u32{ 1024, 4096, 16384, 36864, 65536, 262144 }) |kl| {
        const q = attnQueryChunk(kl);
        try testing.expect(q >= 1);
        try testing.expect(q <= kl);
        // One tile never exceeds the budget (unless a single row already does,
        // which no reachable geometry reaches but must still not divide to 0).
        try testing.expect(q == 1 or @as(u64, q) * kl * 4 <= budget);
        // ...and it is not needlessly small: one more row would exceed it.
        if (q < kl) try testing.expect(@as(u64, q + 1) * kl * 4 > budget);
    }
    // A grid small enough to fit stays in ONE tile, so the sizes that already
    // worked pay nothing for the bookkeeping. 512x512 pixels -> HW 4096.
    try testing.expectEqual(@as(u32, 4096), attnQueryChunk(4096));
    // And the two broken sizes are genuinely divided.
    try testing.expect(attnQueryChunk(36864) < 36864);
    try testing.expect(attnQueryChunk(65536) < 65536);
}

test "the encoder's shortcut plan matches the checkpoint's" {
    // `loadEncoder` asks `hasConvShortcut` which tensors to expect. If that
    // ever disagrees with the dump, load fails with a MISSING WEIGHT rather
    // than silently skipping a projection — but only if the plan is this one.
    const cfg: shape.Config = .{};
    try testing.expect(!shape.hasConvShortcut(cfg, 0));
    try testing.expect(shape.hasConvShortcut(cfg, 1));
    try testing.expect(shape.hasConvShortcut(cfg, 2));
    try testing.expect(!shape.hasConvShortcut(cfg, 3));
}
