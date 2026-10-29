//! The trunk's projections, norms, activations and embedding: per-weight quant params, the lm_head, the decode-only
//! attention requant, the dense MLP, the token embedding and the vision splice.
//! copied_from: sdk/blocks/linear/qmatmul.zig, sdk/blocks/linear/verify_projection.zig, sdk/blocks/attention/proj.zig,
//! sdk/blocks/norms/rms.zig, sdk/blocks/mlp/{activations,dense}.zig, sdk/blocks/embedding.zig, sdk/vision/splice.zig
//! (the qwen4_exp arms: no Hadamard, ternary, row-exact, GGUF, grouped-norm or Gemma activation paths).

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const sushi_pack = @import("../../../sdk/quant/exl3/sushi_pack.zig");
const decode_attn_quant = @import("../../../sdk/blocks/attention/decode_attn_quant.zig");
const verify_projection = @import("../../../sdk/blocks/linear/verify_projection.zig");
const Trunk = @import("../trunk.zig").Trunk;
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const DenseMlpWeights = @import("../../../sdk/blocks/mlp/dense.zig").DenseMlpWeights;
const MAX_BATCH_ROWS = @import("../../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const lastDim = @import("../../../engines/mlx/load/weights.zig").lastDim;
const transposeBf16Weight = @import("../../../engines/mlx/load/weights.zig").transposeBf16Weight;
const concatAxis0 = @import("../../../mlx/array.zig").concatAxis0;
const BITS_CACHE_CAP = @import("../../../sdk/quant/params.zig").BITS_CACHE_CAP;
const BitsCache = @import("../../../sdk/quant/params.zig").BitsCache;
const QuantParams = @import("../../../sdk/quant/params.zig").QuantParams;
const computeQuantParams = @import("../../../sdk/quant/params.zig").computeQuantParams;
const qmatmulBits = @import("../../../sdk/blocks/linear/qmatmul.zig").qmatmulBits;
const qmv2 = @import("../../../sdk/blocks/qmm/qmv2.zig");
const qmv_nax2 = @import("../../../sdk/blocks/qmm/qmv_nax2.zig");
const msvQmvRows = @import("../../../sdk/blocks/linear/qmv_rows.zig").msvQmvRows;
const FUSED_ROWS_MAX_M = @import("../../../sdk/blocks/linear/fused_rows.zig").FUSED_ROWS_MAX_M;
const AttnDqCopy = decode_attn_quant.AttnDqCopy;
const swigluJoined = @import("../../../sdk/blocks/mlp/fused_activations.zig").swigluJoined;
const fusedSwiGLU = @import("../../../sdk/blocks/mlp/fused_activations.zig").fusedSwiGLU;
const spliceVisionRows = @import("../../../sdk/vision/splice.zig").spliceVisionRows;

pub inline fn qmatmul(self: *const Trunk, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array) !mlx.mlx_array {
    if (sushi_pack.isRowScaledFp8(w, sc)) return sushi_pack.fp8Linear(x, w, sc, self.s);
    const qp = self.quantParamsHinted(w, sc, lastDim(x));
    // A ternary 2-bit pack (bias == -scale on every weight) reads the 2-bit kernels; they decline other shapes.
    if (self.ternary_2bit) {
        if (try qmv_nax2.qmm(x, w, sc, bi, qp.bits, qp.group_size, self.s)) |y| return y;
        if (try qmv2.qmm(x, w, sc, bi, qp.bits, qp.group_size, true, false, self.s)) |y| return y;
    }
    return qmatmulBits(x, w, sc, bi, qp.bits, qp.group_size, qp.mode, self.s);
}

/// The logits projection: a dense bf16 head is `[vocab, hidden]`, projected through a transposed view.
pub inline fn lmHeadProject(self: *const Trunk, x: mlx.mlx_array) !mlx.mlx_array {
    if (self.lm_head_s.ctx == null) {
        const wt = try transposeBf16Weight(self.lm_head_w, self.s);
        defer _ = mlx.mlx_array_free(wt);
        var result = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&result, x, wt, self.s));
        return result;
    }
    return self.qmatmul(x, self.lm_head_w, self.lm_head_s, self.lm_head_b);
}

pub inline fn quantParamsFor(self: *const Trunk, w: mlx.mlx_array, sc: mlx.mlx_array) QuantParams {
    return self.quantParamsHinted(w, sc, null);
}

/// Per-weight (bits, group_size, mode), cached by the scales array; `in_dim` pins an override's geometry.
pub inline fn quantParamsHinted(self: *const Trunk, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
    const key_raw = sc.ctx orelse return .{
        .bits = self.config.quant_bits,
        .group_size = self.config.quant_group_size,
        .mode = self.config.quant_mode,
    };
    const cache = @constCast(&self.bits_cache);
    const start = BitsCache.slot(key_raw);
    var i: usize = 0;
    while (i < 4) : (i += 1) {
        const idx = (start + i) & (BITS_CACHE_CAP - 1);
        const entry = cache.keys[idx];
        if (entry == key_raw) {
            return .{
                .bits = cache.vals_bits[idx],
                .group_size = @as(u32, cache.vals_gs_div8[idx]) * 8,
                .mode = @fromBackingInt(@intCast(cache.vals_mode[idx])),
            };
        }
        if (entry == null) {
            const detected = computeQuantParams(.of(&self.config), w, sc, in_dim);
            _ = cache.put(key_raw, detected);
            return detected;
        }
    }
    return computeQuantParams(.of(&self.config), w, sc, in_dim);
}

/// Every row of `x` `[N, …]` projected on its own (`lm_head`: through the trunk's head), the rows stacked again.
pub fn mtpProjectionRows(self: *Trunk, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, lm_head: bool) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    if (sh.len < 2 or sh.len > 8 or sh[0] < 1 or sh[0] > MAX_BATCH_ROWS) return error.MtpProjectionShape;
    const n: usize = @intCast(sh[0]);
    var virtual_rows: c_int = 1;
    for (sh[1 .. sh.len - 1]) |d| virtual_rows *= d;
    if (virtual_rows == 1) {
        const qp = self.quantParamsHinted(w, sc, lastDim(x));
        if (qp.mode == .affine) {
            if (try msvQmvRows(self.s, x, w, sc, bi, qp.bits, qp.group_size)) |y| return y;
        }
    }
    var parts: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (parts[0..n]) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    };
    for (0..n) |row| {
        var starts: [8]c_int = @splat(0);
        var stops: [8]c_int = @splat(0);
        const strides: [8]c_int = @splat(1);
        @memcpy(stops[0..sh.len], sh);
        starts[0] = @intCast(row);
        stops[0] = @intCast(row + 1);
        var xrow = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xrow);
        try mlx.check(mlx.mlx_slice(&xrow, x, &starts, @intCast(sh.len), &stops, @intCast(sh.len), &strides, @intCast(sh.len), self.s));
        parts[row] = if (lm_head) try self.lmHeadProject(xrow) else try self.qmatmul(xrow, w, sc, bi);
    }
    return concatAxis0(self.s, parts[0..n]);
}

/// A single verify's fused route packing and expert reduction: only while capturing for a spec verify, 2..6 rows.
pub fn verifyFeatureEnabled(self: *const Trunk, comptime feature: verify_projection.SingleVerifyFeature, grouped: bool, batch: c_int, width: c_int) bool {
    if (grouped) return true;
    if (@import("builtin").is_test and !verify_projection.single_verify_test_features[@backingInt(feature)]) return false;
    return self.spec_capture_ssm and batch == 1 and width >= 2 and width <= 6;
}

/// A joined row group serves one-row steps.
pub fn rowGroupServes(self: *const Trunk, rows: c_int) bool {
    _ = self;
    return rows <= FUSED_ROWS_MAX_M;
}

/// Attention projection with the decode-only requant (`batch == 1 and !is_prefill`: serial decode and verify).
pub inline fn attnProj(self: *Trunk, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, decode_shape: bool, layer: u32) !mlx.mlx_array {
    if (decode_shape and sc.ctx == null and decode_attn_quant.decodeAttnQuantEnabled()) {
        if (self.attnDqFor(w, layer) catch null) |dq| {
            return qmatmulBits(x, dq.w, dq.s, dq.b, dq.bits, dq.group_size, dq.mode, self.s);
        }
    }
    return self.qmatmul(x, w, sc, bi);
}

/// The decode side copy of a dense attention weight, cached by the weight's pointer with its
/// refusal; the tail layers take nvfp4-g16 (`attnDqUseNvfp4`).
pub fn attnDqFor(self: *Trunk, w: mlx.mlx_array, layer: u32) !?AttnDqCopy {
    const key = @intFromPtr(w.ctx orelse return null);
    if (self.attn_dq.get(key)) |slot| return slot;
    if (!decode_attn_quant.decodeAttnQuantEligible(mlx.getShape(w), mlx.mlx_array_dtype(w), false)) {
        try self.attn_dq.put(self.allocator, key, null);
        log.debug("[decode-attn-quant] declined a projection: shape/dtype ineligible\n", .{});
        return null;
    }
    const nvfp4 = decode_attn_quant.attnDqUseNvfp4(layer, @intCast(self.config.num_hidden_layers));
    const copy = (try decode_attn_quant.buildAttnDqCopy(self.s, w, nvfp4)) orelse {
        try self.attn_dq.put(self.allocator, key, null);
        return null;
    };
    try self.attn_dq.put(self.allocator, key, copy);
    if (!decode_attn_quant.attn_dq_engaged) {
        decode_attn_quant.attn_dq_engaged = true;
        log.info("[decode-attn-quant] engaged: dense attention projections -> int8 g32 side copies (decode-only, lossy; --no-decode-attn-quant restores dense)\n", .{});
    }
    if (nvfp4 and !decode_attn_quant.attn_dq_nvfp4_engaged) {
        decode_attn_quant.attn_dq_nvfp4_engaged = true;
        const nl: u32 = @intCast(self.config.num_hidden_layers);
        log.info("[decode-attn-quant] nvfp4-g16 tail engaged from layer {d}/{d}\n", .{ decode_attn_quant.attnDqNvfp4From(nl), nl });
    }
    return copy;
}

pub inline fn rmsNorm(self: *const Trunk, x: mlx.mlx_array, w: mlx.mlx_array) !mlx.mlx_array {
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&result, x, w, self.config.rms_norm_eps, self.s));
    return result;
}

pub fn silu(self: *const Trunk, x: mlx.mlx_array) !mlx.mlx_array {
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, x, self.s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, x, sig, self.s));
    return result;
}

/// silu(gate) * up: the one fused kernel, else the two ops.
pub fn computeGeglu(self: *const Trunk, gate: mlx.mlx_array, up: mlx.mlx_array) !mlx.mlx_array {
    if (try fusedSwiGLU(self.s, gate, up)) |fused| return fused;
    const activated = try self.silu(gate);
    defer _ = mlx.mlx_array_free(activated);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, activated, up, self.s));
    return result;
}

pub fn denseMLP(self: *Trunk, x: mlx.mlx_array, dw: *const DenseMlpWeights) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    var rows: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| rows *= d;
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    if (dw.gu.w.ctx != null and self.rowGroupServes(rows)) {
        const both = try self.qmatmul(x, dw.gu.w, dw.gu.s, dw.gu.b);
        defer _ = mlx.mlx_array_free(both);
        if (dw.gu.widths[0] == dw.gu.widths[1]) {
            if (try swigluJoined(self.s, both)) |act| {
                defer _ = mlx.mlx_array_free(act);
                return self.qmatmul(act, dw.down_w, dw.down_s, dw.down_b);
            }
        }
        try dw.gu.part(&gate, both, 0, self.s);
        try dw.gu.part(&up, both, 1, self.s);
    } else {
        gate = try self.qmatmul(x, dw.gate_w, dw.gate_s, dw.gate_b);
        up = try self.qmatmul(x, dw.up_w, dw.up_s, dw.up_b);
    }
    const activated = try self.computeGeglu(gate, up);
    defer _ = mlx.mlx_array_free(activated);
    return self.qmatmul(activated, dw.down_w, dw.down_s, dw.down_b);
}

/// The token rows `[B, L, hidden]` in the activation dtype (dense bf16 tables gathered as they are).
pub fn embedding(self: *const Trunk, token_ids: mlx.mlx_array) !mlx.mlx_array {
    if (self.emb_w.ctx == null) return error.MissingWeight;
    const id_shape = mlx.getShape(token_ids);
    const batch = id_shape[0];
    const seq_len = id_shape[1];
    var flat_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_ids);
    try mlx.check(mlx.mlx_reshape(&flat_ids, token_ids, &[_]c_int{batch * seq_len}, 1, self.s));
    var taken_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken_w);
    try mlx.check(mlx.mlx_take_axis(&taken_w, self.emb_w, flat_ids, 0, self.s));
    var emb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(emb);
    if (self.emb_s.ctx == null) {
        try mlx.check(mlx.mlx_astype(&emb, taken_w, .bfloat16, self.s));
    } else {
        var taken_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_s);
        try mlx.check(mlx.mlx_take_axis(&taken_s, self.emb_s, flat_ids, 0, self.s));
        // Bias-less modes (nvfp4/mxfp4/mxfp8) carry no biases to gather.
        var taken_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken_b);
        if (self.emb_b.ctx != null) try mlx.check(mlx.mlx_take_axis(&taken_b, self.emb_b, flat_ids, 0, self.s));
        const emb_qp = self.quantParamsHinted(self.emb_w, self.emb_s, self.config.hidden_size);
        try mlx.check(mlx.mlx_dequantize(
            &emb,
            taken_w,
            taken_s,
            taken_b,
            mlx.mlx_optional_int.some(@intCast(emb_qp.group_size)),
            mlx.mlx_optional_int.some(@intCast(emb_qp.bits)),
            emb_qp.mode.cstr(),
            .{},
            .{ .value = self.actDtype(), .has_value = true },
            self.s,
        ));
    }
    var reshaped = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&reshaped, emb, &[_]c_int{ batch, seq_len, @intCast(self.config.hidden_size) }, 3, self.s));
    return reshaped;
}

/// A prefill chunk's media rows spliced at the placeholder ids (as encoded: no rescale), from `vision_splice_offset`.
pub fn applyVisionEmbeddingsWith(self: *Trunk, ctx: *ForwardCtx, h: mlx.mlx_array, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const cfg = &self.config;
    if (mlx.getShape(h)[1] <= 1) return h;
    const ve = ctx.vision_embeddings orelse return h;
    if (cfg.image_token_id == 0 and cfg.audio_token_id == 0 and cfg.video_token_id == 0) return h;
    return spliceVisionRows(h, token_ids, ve, cfg.image_token_id, cfg.audio_token_id, cfg.video_token_id, ctx.vision_splice_offset, self.s);
}

/// The head over a hidden of another dtype (a draft's), cast to the trunk's activation dtype first.
pub fn lmHeadForeign(self: *const Trunk, x: mlx.mlx_array) !mlx.mlx_array {
    if (mlx.mlx_array_dtype(x) == self.actDtype()) return self.lmHeadProject(x);
    var xc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xc);
    try mlx.check(mlx.mlx_astype(&xc, x, self.actDtype(), self.s));
    return self.lmHeadProject(xc);
}
test "trunk qmatmul: a ternary 2-bit pack routes through qmv2; without the flag it stays stock" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const n: c_int = 2048;
    const k: c_int = 512;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const codes = try std.testing.allocator.alloc(u32, @divExact(2048 * 512, 16));
    defer std.testing.allocator.free(codes);
    for (codes) |*wd| wd.* = rnd.int(u32);
    const scv = try std.testing.allocator.alloc(f32, @divExact(2048 * 512, 128));
    defer std.testing.allocator.free(scv);
    for (scv) |*e| e.* = 0.01 + 0.01 * rnd.float(f32);
    var xv: [2 * 512]f32 = undefined;
    for (&xv) |*e| e.* = rnd.floatNorm(f32);
    const w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ n, @divExact(k, 16) }, 2, .uint32);
    defer _ = mlx.mlx_array_free(w);
    const sc32 = mlx.mlx_array_new_data(scv.ptr, &[_]c_int{ n, @divExact(k, 128) }, 2, .float32);
    defer _ = mlx.mlx_array_free(sc32);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, sc32, .bfloat16, s));
    // bias = +scale: the ternary kernel never reads biases, so it disagrees with stock by construction.
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_copy(&bi, sc, s));
    const x32 = mlx.mlx_array_new_data(&xv, &[_]c_int{ 1, 2, k }, 3, .float32);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

    @import("../../../mlx/device.zig").device_gen_override = .{ .gen = 13, .phone = false };
    defer @import("../../../mlx/device.zig").device_gen_override = null;
    var xfm: Trunk = undefined;
    xfm.s = s;
    xfm.bits_cache = .{};
    xfm.config = std.mem.zeroes(@TypeOf(xfm.config));
    xfm.config.quant_mode = .affine;
    xfm.config.quant_bits = 2;
    xfm.config.quant_group_size = 128;

    const kernel = (try qmv2.qmm(x, w, sc, bi, 2, 128, true, false, s)).?;
    defer _ = mlx.mlx_array_free(kernel);
    const stock = try qmatmulBits(x, w, sc, bi, 2, 128, .affine, s);
    defer _ = mlx.mlx_array_free(stock);
    {
        // The two paths must be tellable apart, or the routing check below is vacuous.
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, kernel, stock, false, s));
        var same = true;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try std.testing.expect(!same);
    }
    for ([_]bool{ true, false }) |flag| {
        xfm.ternary_2bit = flag;
        const got = try xfm.qmatmul(x, w, sc, bi);
        defer _ = mlx.mlx_array_free(got);
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, got, if (flag) kernel else stock, false, s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try std.testing.expect(same);
    }
}
