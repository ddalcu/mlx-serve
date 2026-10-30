//! engines/mlx/ane_prefill.
//! Owns AneChanRest, denseMLPMaybeAne, denseMLPAneSplit, gdnProjMaybeAne, gdnProjAneSplit, anePackPlane, and 10 more.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const ane_offload = @import("../../sdk/ane/offload.zig");
const materializedOwnedCopy = @import("../../mlx/array.zig").materializedOwnedCopy;
const totalMemBytes = @import("../../mlx/device.zig").totalMemBytes;
const attn256RandBf16 = @import("../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const GdnProj = @import("../../sdk/blocks/gdn/gated_delta_net.zig").GdnProj;
const LinearAttnWeights = @import("../../sdk/blocks/gdn/gated_delta_net.zig").LinearAttnWeights;
const DenseMlpWeights = @import("../../sdk/blocks/mlp/dense.zig").DenseMlpWeights;
const maxAbsDiffF32 = @import("../../sdk/testing/parity.zig").maxAbsDiffF32;
const Transformer = @import("transformer.zig").Transformer;

/// ANE channel-split GPU complement for one layer (A1): the output-channel
/// rest-slices the GPU computes while the ANE runs channels [0..k). The
/// mlp arm reuses DenseMlpWeights so denseMLP serves it unchanged (its
/// down slice yields the rest PARTIAL sum, added to the ANE partial at the
/// seam); the gdn arm carries only the projection slices gdnProjGpu reads.
pub const AneChanRest = struct {
    mlp: ?DenseMlpWeights = null,
    gdn_qkv_w: mlx.mlx_array = .{ .ctx = null },
    gdn_qkv_s: mlx.mlx_array = .{ .ctx = null },
    gdn_qkv_b: mlx.mlx_array = .{ .ctx = null },
    gdn_z_w: mlx.mlx_array = .{ .ctx = null },
    gdn_z_s: mlx.mlx_array = .{ .ctx = null },
    gdn_z_b: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *AneChanRest) void {
        self.deinitMlp();
        self.deinitGdn();
    }

    fn deinitMlp(self: *AneChanRest) void {
        if (self.mlp) |*m| {
            inline for (.{ m.gate_w, m.gate_s, m.gate_b, m.up_w, m.up_s, m.up_b, m.down_w, m.down_s, m.down_b }) |a| {
                if (a.ctx != null) _ = mlx.mlx_array_free(a);
            }
            self.mlp = null;
        }
    }

    fn deinitGdn(self: *AneChanRest) void {
        inline for (.{ self.gdn_qkv_w, self.gdn_qkv_s, self.gdn_qkv_b, self.gdn_z_w, self.gdn_z_s, self.gdn_z_b }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        }
        self.gdn_qkv_w = .{ .ctx = null };
        self.gdn_qkv_s = .{ .ctx = null };
        self.gdn_qkv_b = .{ .ctx = null };
        self.gdn_z_w = .{ .ctx = null };
        self.gdn_z_s = .{ .ctx = null };
        self.gdn_z_b = .{ .ctx = null };
    }
};

// ── ANE prefill-MLP offload (`--ane-prefill`, perf-plan-aug-17 P5) ──

/// Dense-MLP dispatch with the optional ANE row split: engages only on a
/// full-width prefill chunk of a covered layer; everything else (decode,
/// verify widths, tail chunks, uncovered layers, no engine) runs the
/// plain GPU chain. Fixed shapes are an ANE constraint — the compiled
/// programs only accept exactly `eng.rows` rows.
pub fn denseMLPMaybeAne(self: *Transformer, x: mlx.mlx_array, dw: *const DenseMlpWeights, layer_idx: usize, is_prefill: bool, seq_len: c_int) !mlx.mlx_array {
    if (is_prefill) if (self.aneMlp(x, layer_idx, seq_len)) |y| return y;
    return self.denseMLP(x, dw);
}

/// The split dense MLP of a covered layer's full-width prefill chunk `x`; null = the GPU computes the whole layer
/// (another width, an uncovered layer, no engine, or an ANE failure).
pub fn aneMlp(self: *Transformer, x: mlx.mlx_array, layer_idx: usize, seq_len: c_int) ?mlx.mlx_array {
    const eng = self.ane_prefill orelse return null;
    if (seq_len != @as(c_int, @intCast(eng.chunk_rows))) return null;
    if (!eng.mlpReady(layer_idx)) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 3 or xsh[0] != 1 or xsh[2] != @as(c_int, @intCast(eng.hidden))) return null;
    const rests = self.ane_chan orelse return null;
    const rest = if (rests[layer_idx].mlp) |*m| m else return null;
    // An ANE failure is never a request failure — whole chunk on GPU.
    return self.denseMLPAneChannel(eng, x, rest, layer_idx) catch |err| {
        log.warn("[ane] channel split failed on layer {d} ({s}) — GPU fallback\n", .{ layer_idx, @errorName(err) });
        return null;
    };
}

/// GDN separate-arm input-projection dispatch with the optional ANE row
/// split (v2 beside the MLP seam): same engagement conditions — a
/// full-width prefill chunk of a covered layer; everything else runs the
/// plain GPU projections.
pub fn gdnProjMaybeAne(self: *Transformer, x: mlx.mlx_array, la: *const LinearAttnWeights, layer_idx: usize, is_prefill: bool, seq_len: c_int) !GdnProj {
    if (is_prefill) if (self.aneGdnProj(x, layer_idx, seq_len)) |p| return p;
    return self.gdnProjGpu(x, la);
}

/// The split q|k|v and z projections of a covered layer's full-width prefill chunk `x`; null = the GPU projects them.
pub fn aneGdnProj(self: *Transformer, x: mlx.mlx_array, layer_idx: usize, seq_len: c_int) ?GdnProj {
    const eng = self.ane_prefill orelse return null;
    if (seq_len != @as(c_int, @intCast(eng.chunk_rows))) return null;
    if (!eng.gdnReady(layer_idx)) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len != 3 or xsh[0] != 1 or xsh[2] != @as(c_int, @intCast(eng.hidden))) return null;
    const rests = self.ane_chan orelse return null;
    const rest = &rests[layer_idx];
    if (rest.gdn_qkv_w.ctx == null or rest.gdn_z_w.ctx == null) return null;
    return self.gdnProjAneChannel(eng, x, rest, layer_idx) catch |err| {
        log.warn("[ane] gdn channel split failed on layer {d} ({s}) — GPU fallback\n", .{ layer_idx, @errorName(err) });
        return null;
    };
}

/// The plugin forward's seam (`Forward.offload`) over this shell's ANE programs, at the legacy forward's two sites.
pub fn aneOffload(self: *Transformer) abi.Offload {
    return .{ .ctx = self, .mlp = &offloadMlp, .gdn_proj = &offloadGdn };
}

fn chunkRows(x: mlx.mlx_array) c_int {
    const sh = mlx.getShape(x);
    return if (sh.len == 3) sh[1] else 0;
}

fn offloadMlp(ctx: *anyopaque, layer: u32, x: abi.Array, out: *abi.Array) callconv(.c) bool {
    const self: *Transformer = @ptrCast(@alignCast(ctx));
    const xa: mlx.mlx_array = .{ .ctx = x.ctx };
    const y = self.aneMlp(xa, layer, chunkRows(xa)) orelse return false;
    out.* = .{ .ctx = y.ctx };
    return true;
}

fn offloadGdn(ctx: *anyopaque, layer: u32, x: abi.Array, qkv: *abi.Array, z: *abi.Array) callconv(.c) bool {
    const self: *Transformer = @ptrCast(@alignCast(ctx));
    const xa: mlx.mlx_array = .{ .ctx = x.ctx };
    const p = self.aneGdnProj(xa, layer, chunkRows(xa)) orelse return false;
    qkv.* = .{ .ctx = p.qkv.ctx };
    z.* = .{ .ctx = p.z.ctx };
    return true;
}

/// Pack a [1, R, H] hidden tensor into an fp16 CHANNEL-major ANE input
/// plane ([H][R], element (r, c) at plane[c*R + r]): transpose + astype
/// flattened 1-D to force a row-major materialization (the raw-read
/// contiguity rule), then one contiguous memcpy.
pub fn anePackPlane(self: *Transformer, x_rows: mlx.mlx_array, plane: [*]f16) !void {
    return ane_offload.packPlane(self.s, x_rows, plane);
}

/// Read an ANE output plane ([width][R] fp16 channel-major) back as a
/// [1, R, width] tensor in `dtype`.
pub fn aneReadPlane(self: *Transformer, plane: [*]f16, width: c_int, rows: c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    return ane_offload.readPlane(self.s, plane, width, rows, dtype);
}

// ── ANE channel split ──

/// Channel-mode dense MLP: both units see ALL chunk rows — the ANE
/// computes output channels [0..k) of gate/up and the matching down
/// K-slabs (a PARTIAL down sum), the GPU the rest via the stored
/// complement slices; the two partials ADD at the seam. Same
/// pack/kick/async/join discipline as the row split. null = a unit
/// failed: the caller recomputes the whole layer on the GPU.
pub fn denseMLPAneChannel(self: *Transformer, eng: *ane_offload.AnePrefill, x: mlx.mlx_array, rest: *const DenseMlpWeights, layer_idx: usize) !?mlx.mlx_array {
    try self.anePackUnitPlanes(eng, x);
    eng.kickMlp(layer_idx);
    var waited = false;
    errdefer if (!waited) {
        _ = eng.waitAll();
    };

    // GPU rest-channel partial: build AND submit so it computes while
    // the ANE runs.
    const y_gpu = try self.denseMLP(x, rest);
    defer _ = mlx.mlx_array_free(y_gpu);
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, y_gpu);
        _ = mlx.mlx_async_eval(ev);
    }

    const ok = eng.waitAll();
    waited = true;
    // Failed eval on ANY unit: a subset of the partials is not an
    // answer — the whole layer recomputes on the GPU from the original
    // weights.
    if (!ok) return null;
    // Every unit contributes a PARTIAL down sum over its own channel
    // slice; they add with the GPU's rest partial.
    var acc = try self.aneReadPlane(eng.units[0].mlpOutputBase() orelse return error.AnePlaneMissing, @intCast(eng.hidden), @intCast(eng.rows), mlx.mlx_array_dtype(x));
    errdefer _ = mlx.mlx_array_free(acc);
    for (eng.units[1..]) |*u| {
        const y_ane = try self.aneReadPlane(u.mlpOutputBase() orelse return error.AnePlaneMissing, @intCast(eng.hidden), @intCast(eng.rows), mlx.mlx_array_dtype(x));
        defer _ = mlx.mlx_array_free(y_ane);
        var sum = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, acc, y_ane, self.s));
        _ = mlx.mlx_array_free(acc);
        acc = sum;
    }
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_add(&out, acc, y_gpu, self.s));
    _ = mlx.mlx_array_free(acc);
    eng.logEngagedOnce("prefill");
    return out;
}

pub fn anePackUnitPlanes(self: *Transformer, eng: *ane_offload.AnePrefill, x: mlx.mlx_array) !void {
    return ane_offload.packUnitPlanes(self.s, eng, x);
}

/// Channel-mode GDN projections: the ANE computes output channels
/// [0..kq) of qkv and [0..kz) of z (one fused program), the GPU the
/// rest; each tensor CONCATENATES on the channel axis. null = a unit
/// failed: the caller projects on the GPU.
pub fn gdnProjAneChannel(self: *Transformer, eng: *ane_offload.AnePrefill, x: mlx.mlx_array, rest: *const AneChanRest, layer_idx: usize) !?GdnProj {
    const rows_a: c_int = @intCast(eng.rows);
    const kq: c_int = @intCast(eng.gdn_qkv_out);
    const kz: c_int = @intCast(eng.gdn_z_out);
    const out_w: c_int = kq + kz;
    try self.anePackUnitPlanes(eng, x);
    eng.kickGdn(layer_idx);
    var waited = false;
    errdefer if (!waited) {
        _ = eng.waitAll();
    };

    const gpu = try self.gdnProjArrays(x, rest.gdn_qkv_w, rest.gdn_qkv_s, rest.gdn_qkv_b, rest.gdn_z_w, rest.gdn_z_s, rest.gdn_z_b);
    defer _ = mlx.mlx_array_free(gpu.qkv);
    defer _ = mlx.mlx_array_free(gpu.z);
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, gpu.qkv);
        _ = mlx.mlx_vector_array_append_value(ev, gpu.z);
        _ = mlx.mlx_async_eval(ev);
    }

    const ok = eng.waitAll();
    waited = true;
    if (!ok) return null;
    // Unit u holds channels [u*kq, (u+1)*kq) of qkv and [u*kz, (u+1)*kz)
    // of z, so the units' slices are contiguous and each tensor
    // CONCATENATES: unit 0 ++ unit 1 ++ ... ++ the GPU rest.
    const strides3 = [_]c_int{ 1, 1, 1 };
    var qkv_parts: [ane_offload.MAX_UNITS + 1]mlx.mlx_array = undefined;
    var z_parts: [ane_offload.MAX_UNITS + 1]mlx.mlx_array = undefined;
    var np: usize = 0;
    defer for (qkv_parts[0..np]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    defer for (z_parts[0..np]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (eng.units) |*u| {
        const out_plane = u.gdnOutputBase() orelse return error.AnePlaneMissing;
        const y_all = try self.aneReadPlane(out_plane, out_w, rows_a, mlx.mlx_array_dtype(x));
        defer _ = mlx.mlx_array_free(y_all);
        var qkv_ane = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&qkv_ane, y_all, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, rows_a, kq }, 3, &strides3, 3, self.s));
        qkv_parts[np] = qkv_ane;
        var z_ane = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&z_ane, y_all, &[_]c_int{ 0, 0, kq }, 3, &[_]c_int{ 1, rows_a, out_w }, 3, &strides3, 3, self.s));
        z_parts[np] = z_ane;
        np += 1;
    }
    qkv_parts[np] = gpu.qkv;
    z_parts[np] = gpu.z;

    var out_qkv = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out_qkv);
    {
        const vec = mlx.mlx_vector_array_new_data(&qkv_parts, np + 1);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&out_qkv, vec, 2, self.s));
    }
    var out_z = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out_z);
    {
        const vec = mlx.mlx_vector_array_new_data(&z_parts, np + 1);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&out_z, vec, 2, self.s));
    }
    eng.logGdnEngagedOnce();
    return .{ .qkv = out_qkv, .z = out_z };
}

/// Axis-0 (output-channel) slice [from..n) of a packed weight or its
/// scales/biases as a zero-copy VIEW — pinned bit-exact through qmm by
/// the "axis-0 slice VIEW" test. A null-sentinel input stays null.
pub fn aneSliceRowsView(self: *Transformer, a: mlx.mlx_array, from: c_int) !mlx.mlx_array {
    if (a.ctx == null) return a;
    const sh = mlx.getShape(a);
    if (sh.len != 2) return error.AneWeightShape;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const start = [_]c_int{ from, 0 };
    const stop = [_]c_int{ sh[0], sh[1] };
    const strides = [_]c_int{ 1, 1 };
    try mlx.check(mlx.mlx_slice(&out, a, &start, 2, &stop, 2, &strides, 2, self.s));
    return out;
}

/// Axis-1 (input-channel) slice [from..m) of a packed weight or its
/// scales/biases, MATERIALIZED (the splitPackedGateUp rule — an axis-1
/// slice is not contiguous). A null-sentinel input stays null.
pub fn aneSliceColsCopy(self: *Transformer, a: mlx.mlx_array, from: c_int) !mlx.mlx_array {
    if (a.ctx == null) return a;
    const sh = mlx.getShape(a);
    if (sh.len != 2) return error.AneWeightShape;
    var view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(view);
    const start = [_]c_int{ 0, from };
    const stop = [_]c_int{ sh[0], sh[1] };
    const strides = [_]c_int{ 1, 1 };
    try mlx.check(mlx.mlx_slice(&view, a, &start, 2, &stop, 2, &strides, 2, self.s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_contiguous(&out, view, false, self.s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Dequantize one [out, in] weight to a host f32 buffer (caller frees).
/// v1 serves quantized packs only — a dense bf16 checkpoint declines
/// (its ANE copy would want the transpose-back path; add when a real
/// pack needs it).
pub fn dequantToHostF32(self: *Transformer, allocator: std.mem.Allocator, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, in_dim: u32, out_dim: u32) ![]f32 {
    if (sc.ctx == null) return error.AneDenseBf16Unsupported;
    const qp = self.quantParamsHinted(w, sc, in_dim);
    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    try mlx.check(mlx.mlx_dequantize(
        &deq,
        w,
        sc,
        bi,
        mlx.mlx_optional_int.some(@intCast(qp.group_size)),
        mlx.mlx_optional_int.some(@intCast(qp.bits)),
        qp.mode.cstr(),
        .{}, // global_scale
        .{ .value = .float32, .has_value = true },
        self.s,
    ));
    const sh = mlx.getShape(deq);
    if (sh.len != 2 or sh[0] != @as(c_int, @intCast(out_dim)) or sh[1] != @as(c_int, @intCast(in_dim))) return error.AneWeightShape;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const n: c_int = @intCast(out_dim * in_dim);
    const fs = [_]c_int{n};
    try mlx.check(mlx.mlx_reshape(&flat, deq, &fs, 1, self.s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const ptr = mlx.mlx_array_data_float32(flat) orelse return error.AneDequantRead;
    const out = try allocator.alloc(f32, @intCast(n));
    @memcpy(out, ptr[0..@intCast(n)]);
    return out;
}

/// A GatedDeltaNet layer's separate q|k|v and z projections, borrowed.
pub const GdnIn = struct {
    qkv_w: mlx.mlx_array,
    qkv_s: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    z_w: mlx.mlx_array,
    z_s: mlx.mlx_array,
    z_b: mlx.mlx_array,
};

/// What one layer offers the ANE programs, borrowed: a dense MLP, separate GDN projections, both or neither.
pub const AneLayer = struct { mlp: ?DenseMlpWeights = null, gdn: ?GdnIn = null };

/// Every layer's `AneLayer` (caller frees the slice): the legacy layer list, or a plugin trunk that declares
/// `Caps.ane_prefill` through its `layer_weight` hook; null = neither serves one.
pub fn aneLayers(self: *const Transformer, a: std.mem.Allocator) !?[]AneLayer {
    if (self.moe_layers) |ml| {
        const out = try a.alloc(AneLayer, ml.len);
        for (ml, out) |*lw, *o| {
            o.* = .{};
            switch (lw.mlp) {
                .dense => |dw| o.mlp = dw,
                .moe => {},
            }
            switch (lw.attn) {
                .linear => |*la| if (la.aneGdnServes()) {
                    o.gdn = .{ .qkv_w = la.qkv_w, .qkv_s = la.qkv_s, .qkv_b = la.qkv_b, .z_w = la.z_w, .z_s = la.z_s, .z_b = la.z_b };
                },
                else => {},
            }
        }
        return out;
    }
    const b = self.plugin orelse return null;
    const caps = (self.config.plugin orelse return null).parsed.caps;
    const get = b.table.weights.layer_weight orelse return null;
    if (!caps.ane_prefill) return null;
    const n = self.config.num_hidden_layers;
    const out = try a.alloc(AneLayer, n);
    for (out, 0..) |*o, i| {
        o.* = .{};
        const li: u32 = @intCast(i);
        var w: [5]quant_abi.Weight = @splat(.{});
        var have: [5]bool = undefined;
        inline for (.{ .mlp_gate, .mlp_up, .mlp_down, .gdn_qkv, .gdn_z }, 0..) |which, k| have[k] = get(b.model, li, which, &w[k]).isOk();
        const arr = struct {
            fn of(x: abi.Array) mlx.mlx_array {
                return .{ .ctx = x.ctx };
            }
        }.of;
        if (have[0] and have[1] and have[2]) o.mlp = .{
            .gate_w = arr(w[0].w),
            .gate_s = arr(w[0].scales),
            .gate_b = arr(w[0].biases),
            .up_w = arr(w[1].w),
            .up_s = arr(w[1].scales),
            .up_b = arr(w[1].biases),
            .down_w = arr(w[2].w),
            .down_s = arr(w[2].scales),
            .down_b = arr(w[2].biases),
        };
        if (have[3] and have[4]) o.gdn = .{ .qkv_w = arr(w[3].w), .qkv_s = arr(w[3].scales), .qkv_b = arr(w[3].biases), .z_w = arr(w[4].w), .z_s = arr(w[4].scales), .z_b = arr(w[4].biases) };
    }
    return out;
}

/// The offload's shape over `layers` at `share` across `units`: coverage, the per-unit channel widths, and the
/// GPU-side down-rest bytes channel mode materializes.
pub const AnePlan = struct {
    dense_layers: u64 = 0,
    gdn_layers: u64 = 0,
    eng_ffn: u32 = 0,
    eng_qkv: u32 = 0,
    eng_z: u32 = 0,
    /// A GDN layer's full q|k|v and z widths.
    qkv_out: u32 = 0,
    z_out: u32 = 0,
    rest_bytes: u64 = 0,
};

pub fn anePlan(cfg: *const model_mod.ModelConfig, layers: []const AneLayer, share: f32, units: u32) AnePlan {
    const ffn = cfg.intermediate_size;
    // GDN input projections (v2): separate-proj GDN layers only (qwen3_next's combined arm stays GPU).
    const gdn_qkv_out: u32 = @intCast(2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + cfg.linear_num_value_heads * cfg.linear_value_head_dim);
    const gdn_z_out: u32 = @intCast(cfg.linear_num_value_heads * cfg.linear_value_head_dim);
    var p: AnePlan = .{
        .eng_ffn = ane_offload.channelSliceWidthUnits(ffn, share, units),
        .eng_qkv = ane_offload.channelSliceWidthUnits(gdn_qkv_out, share, units),
        .eng_z = ane_offload.channelSliceWidthUnits(gdn_z_out, share, units),
        .qkv_out = gdn_qkv_out,
        .z_out = gdn_z_out,
    };
    for (layers) |*l| {
        if (l.mlp != null) p.dense_layers += 1;
        if (l.gdn != null and gdn_qkv_out > 0 and gdn_z_out > 0) p.gdn_layers += 1;
    }
    if (p.eng_qkv == 0 or p.eng_z == 0) p.gdn_layers = 0;
    if (p.eng_ffn > 0) for (layers) |*l| if (l.mlp) |*dw| {
        const down_bytes = mlx.mlx_array_size(dw.down_w) * mlx.mlx_array_itemsize(dw.down_w) +
            mlx.mlx_array_size(dw.down_s) * mlx.mlx_array_itemsize(dw.down_s) +
            (if (dw.down_b.ctx != null) mlx.mlx_array_size(dw.down_b) * mlx.mlx_array_itemsize(dw.down_b) else 0);
        p.rest_bytes += down_bytes * (ffn - units * p.eng_ffn) / ffn;
    };
    return p;
}

/// Build the per-layer ANE MLP programs (`--ane-prefill`). Never fails
/// the load: every refusal is a NAMED `[ane]` line and the model serves
/// GPU-only. v1 scope: qwen3_5-family DENSE MLP (data-dependent MoE
/// routing is not expressible as a fixed-shape ANE graph).
pub fn buildAnePrefill(
    self: *Transformer,
    io: std.Io,
    chunk_rows: u32,
    share: f32,
    headroom_resolver: ?*const fn (*const model_mod.ModelConfig, u32) u64,
) void {
    if (self.ane_prefill != null) return;
    if (!ane_offload.available()) {
        log.warn("[ane] unavailable: AppleNeuralEngine framework not present — --ane-prefill disabled\n", .{});
        return;
    }
    const cfg = &self.config;
    const layers = (self.aneLayers(self.allocator) catch null) orelse {
        log.warn("[ane] --ane-prefill: {s} is outside the v1 scope (qwen3_5-family dense MLP) — disabled\n", .{cfg.model_type});
        return;
    };
    defer self.allocator.free(layers);
    if (!std.mem.eql(u8, cfg.model_type, "qwen3_5_moe")) {
        log.warn("[ane] --ane-prefill: {s} is outside the qwen3_5-family scope — disabled\n", .{cfg.model_type});
        return;
    }
    // A MoE checkpoint's routed experts are data-dependent and can never
    // ride a fixed-shape ANE graph, and its shared-expert MLP measured a
    // ~5% FLOPs ceiling (A7 spike, 35B-A3B) — but the GDN input
    // projections are per-token dense and a 32% ceiling there, so a MoE
    // target gets GDN-ONLY coverage (no layer offers a dense MLP).
    const hidden = cfg.hidden_size;
    const ffn = cfg.intermediate_size;
    // Channel split (A1): ALL chunk rows run through OUTPUT-channel slices (the engine widths become the sliced
    // widths and the GPU serves the complement).
    // Dual ANE (on the M3 Ultra, where it was measured):
    // two units pinned to the two dies, each computing HALF the total
    // channel share concurrently.
    const units = ane_offload.unitCount(ane_offload.chipBrand(), ane_offload.dualEnabled());
    if (chunk_rows % 32 != 0) {
        log.warn("[ane] --ane-prefill: channel mode needs a 32-multiple prefill chunk (got {d}) — disabled\n", .{chunk_rows});
        return;
    }
    const rows: u32 = chunk_rows;
    const plan = anePlan(cfg, layers, share, units);
    const dense_layer_count = plan.dense_layers;
    const eng_ffn = plan.eng_ffn;
    const eng_qkv = plan.eng_qkv;
    const eng_z = plan.eng_z;
    const gdn_qkv_out = plan.qkv_out;
    const gdn_z_out = plan.z_out;
    if (eng_ffn == 0 and dense_layer_count > 0) {
        log.warn("[ane] --ane-prefill: share {d:.2} of ffn {d} over {d} unit(s) leaves no usable channel slice — disabled\n", .{ share, ffn, units });
        return;
    }
    const gdn_target_layers = plan.gdn_layers;
    const gdn_on = gdn_target_layers > 0;
    // Per-model admission (replaces the v1 flat 96 GB total-RAM gate):
    // bill the int8 copies + fp16 planes THIS config will build — plus,
    // in channel mode, the materialized GPU-side down-rest slices —
    // against what is already resident. A 27B bills ~20 GB row / ~10 GB
    // channel and needs a big Mac; a 0.8B bills ~1 GB and runs on a
    // small one.
    {
        const chan_rest_bytes = plan.rest_bytes;
        const bill = ane_offload.engineBillBytes(dense_layer_count, gdn_target_layers, hidden, eng_ffn, eng_qkv, eng_z, rows, units) + chan_rest_bytes;
        const total_mem = totalMemBytes();
        var resident: usize = 0;
        _ = mlx.mlx_get_active_memory(&resident);
        // Model-derived: KV for a context worth serving + this chunk's
        // prefill envelope + baseline (`server.aneGateHeadroom`). Without
        // a resolver (tests, other load paths) only the baseline stands.
        const headroom: u64 = if (headroom_resolver) |hr| hr(cfg, chunk_rows) else ane_offload.GATE_BASELINE_BYTES;
        if (!ane_offload.gateAllows(total_mem, resident, bill, headroom)) {
            const gib = 1024 * 1024 * 1024;
            log.warn("[ane] --ane-prefill: the offload bills ~{d:.1} GB (int8 copies + I/O planes{s}) on top of {d:.1} GB resident + {d:.1} GB headroom (KV for {d} tokens + the chunk-{d} prefill envelope + baseline), exceeding {d} GB total RAM — disabled\n", .{
                                @as(f64, @floatFromInt(bill)) / gib,
                " + GPU rest slices",
                @as(f64, @floatFromInt(resident)) / gib,
                @as(f64, @floatFromInt(headroom)) / gib,
                ane_offload.MIN_CONTEXT_TOKENS,
                chunk_rows,
                total_mem / gib,
            });
            return;
        }
    }
    // Pre-build disk check (A8): under the hard floor even cache
    // restores and the framework's own saves fail bare, and the build
    // ships silent partial coverage — refuse by name and serve GPU-only
    // instead. Above the floor but under the full cold-build budget is
    // only a WARNING: warm cache entries need ~no scratch and a cold
    // build converges across boots.
    {
        const free_disk = ane_offload.internalFreeDiskBytes();
        const gib = 1024 * 1024 * 1024;
        if (free_disk > 0 and free_disk < ane_offload.BUILD_DISK_FLOOR_BYTES) {
            log.warn("[ane] --ane-prefill: only {d:.1} GB free on the internal disk — below the {d} GB build floor, where ANE compiles fail bare and coverage ships partial; build skipped (free disk, then reload)\n", .{
                @as(f64, @floatFromInt(free_disk)) / gib,
                ane_offload.BUILD_DISK_FLOOR_BYTES / gib,
            });
            return;
        }
        ane_offload.setCacheLineage("prefill", cfg.num_hidden_layers, hidden, ffn, share);
        const cold_budget: u64 = @as(u64, cfg.num_hidden_layers) * 3 * hidden * eng_ffn * units;
        if (free_disk > 0 and free_disk < cold_budget) {
            log.info("[ane] internal free disk {d:.1} GB is under a fully-cold build's ~{d:.1} GB compiler-scratch budget — a cold build will converge across boots (warm cache entries are unaffected)\n", .{
                @as(f64, @floatFromInt(free_disk)) / gib,
                @as(f64, @floatFromInt(cold_budget)) / gib,
            });
        }
    }
    const eng = ane_offload.AnePrefill.init(self.allocator, io, cfg.num_hidden_layers, hidden, eng_ffn, rows, chunk_rows, if (gdn_on) eng_qkv else 0, if (gdn_on) eng_z else 0, units) catch |err| {
        log.warn("[ane] engine init failed ({s}) — disabled\n", .{@errorName(err)});
        return;
    };
    const rests = self.allocator.alloc(AneChanRest, cfg.num_hidden_layers) catch {
        log.warn("[ane] channel rest table alloc failed — disabled\n", .{});
        eng.deinit();
        return;
    };
    @memset(rests, .{});
    self.ane_chan = rests;
    const build_start = std.Io.Timestamp.now(io, .awake);
    // A failed layer compile is worth ONE retry pass: the dominant
    // cause is the build's own staging burst running the disk out
    // (bounded now that compile inputs are deleted post-load, and a
    // partial build's cache entries make the NEXT boot converge), and
    // the rare aned-margin failure heals on a second attempt. A partial
    // build otherwise stays partial for the whole serve.
    var pass: usize = 0;
    while (pass < 2) : (pass += 1) {
        for (layers, 0..) |*al, i| {
            if (al.mlp) |*dw| {
                if (eng.mlpReady(i)) continue;
                const gate = self.dequantToHostF32(self.allocator, dw.gate_w, dw.gate_s, dw.gate_b, hidden, ffn) catch |err| {
                    log.warn("[ane] layer {d} gate dequant failed ({s}) — layer stays on GPU\n", .{ i, @errorName(err) });
                    continue;
                };
                defer self.allocator.free(gate);
                const up = self.dequantToHostF32(self.allocator, dw.up_w, dw.up_s, dw.up_b, hidden, ffn) catch |err| {
                    log.warn("[ane] layer {d} up dequant failed ({s}) — layer stays on GPU\n", .{ i, @errorName(err) });
                    continue;
                };
                defer self.allocator.free(up);
                const down = self.dequantToHostF32(self.allocator, dw.down_w, dw.down_s, dw.down_b, ffn, hidden) catch |err| {
                    log.warn("[ane] layer {d} down dequant failed ({s}) — layer stays on GPU\n", .{ i, @errorName(err) });
                    continue;
                };
                defer self.allocator.free(down);
                const k = eng_ffn;
                if (self.ane_chan.?[i].mlp == null) {
                    self.ane_chan.?[i].mlp = self.aneBuildMlpRest(dw, k * units, ffn) catch |err| {
                        log.warn("[ane] layer {d} channel rest slicing failed ({s}) — layer stays on GPU\n", .{ i, @errorName(err) });
                        continue;
                    };
                }
                // Per-unit slice of the dequant buffers: unit u
                // owns output channels [u*k, (u+1)*k), so gate/up
                // rows are a plain host slice and down repacks
                // that column window.
                const kh: usize = @as(usize, k) * @as(usize, hidden);
                const down_slice = self.allocator.alloc(f32, @as(usize, hidden) * k) catch continue;
                defer self.allocator.free(down_slice);
                for (0..units) |u| {
                    const c0: usize = u * @as(usize, k);
                    for (0..hidden) |r| @memcpy(down_slice[r * k ..][0..k], down[r * ffn + c0 ..][0..k]);
                    eng.addMlpLayer(u, i, gate[u * kh ..][0..kh], up[u * kh ..][0..kh], down_slice) catch |err| {
                        log.warn("[ane] layer {d} unit {d} mlp enqueue failed ({s})\n", .{ i, u, @errorName(err) });
                        break;
                    };
                }
            }
        }
        if (gdn_on) for (layers, 0..) |*al, i| {
            if (al.gdn) |*la| {
                if (eng.gdnReady(i)) continue;
                const qkv = self.dequantToHostF32(self.allocator, la.qkv_w, la.qkv_s, la.qkv_b, hidden, gdn_qkv_out) catch |err| {
                    log.warn("[ane] layer {d} gdn qkv dequant failed ({s}) — projections stay on GPU\n", .{ i, @errorName(err) });
                    continue;
                };
                defer self.allocator.free(qkv);
                const z = self.dequantToHostF32(self.allocator, la.z_w, la.z_s, la.z_b, hidden, gdn_z_out) catch |err| {
                    log.warn("[ane] layer {d} gdn z dequant failed ({s}) — projections stay on GPU\n", .{ i, @errorName(err) });
                    continue;
                };
                defer self.allocator.free(z);
                if (self.ane_chan.?[i].gdn_qkv_w.ctx == null) {
                    self.aneBuildGdnRest(&self.ane_chan.?[i], la, eng_qkv * units, eng_z * units) catch |err| {
                        log.warn("[ane] layer {d} gdn channel rest slicing failed ({s}) — projections stay on GPU\n", .{ i, @errorName(err) });
                        continue;
                    };
                }
                for (0..units) |u| {
                    const q0: usize = u * @as(usize, eng_qkv) * hidden;
                    const z0: usize = u * @as(usize, eng_z) * hidden;
                    eng.addGdnLayer(u, i, qkv[q0..][0 .. @as(usize, eng_qkv) * hidden], z[z0..][0 .. @as(usize, eng_z) * hidden]) catch |err| {
                        log.warn("[ane] layer {d} unit {d} gdn enqueue failed ({s})\n", .{ i, u, @errorName(err) });
                        break;
                    };
                }
            }
        };
        // Compile everything this pass queued. A bank that the runtime
        // refuses walks the split ladder inside finishPending; only a
        // bank of ONE that still fails drops its layer to the GPU, and
        // those are what the retry pass re-queues.
        eng.finishPending();
        const failed = eng.droppedPrograms();
        if (failed == 0) break;
        eng.resetDropped();
        if (pass == 0) {
            log.warn("[ane] {d} layer program builds failed — one retry pass\n", .{failed});
            std.Io.sleep(io, .fromMilliseconds(2000), .real) catch {};
        }
    }
    const built = eng.coveredLayers();
    const gdn_built = eng.coveredGdnLayers();
    if (built == 0 and gdn_built == 0) {
        log.warn("[ane] no layer produced an ANE program — disabled\n", .{});
        eng.deinit();
        for (rests) |*r| r.deinit();
        self.allocator.free(rests);
        self.ane_chan = null;
        return;
    }
    // A rest slice whose ANE half never compiled is dead weight AND a
    // wrong answer if the seam ever read it — drop it with the layer.
    for (rests, 0..) |*r, i| {
        if (r.mlp != null and !eng.mlpReady(i)) r.deinitMlp();
        if (r.gdn_qkv_w.ctx != null and !eng.gdnReady(i)) r.deinitGdn();
    }
    const int8_bytes: u64 = (@as(u64, built) * 3 * hidden * eng_ffn + @as(u64, gdn_built) * (@as(u64, eng_qkv) + eng_z) * hidden) * units;
    const secs: f64 = @as(f64, @floatFromInt(@as(u64, @intCast(build_start.untilNow(io, .awake).nanoseconds)))) / 1e9;
    eng.publishLive(share, int8_bytes);
    self.ane_prefill = eng;
    log.info("[ane] prefill offload ready: units={d} {d}/{d} mlp + {d} gdn layers in {d} banks, rows={d}/{d} (share {d:.2}), int8 ~{d} MB, built in {d:.1}s\n", .{ units, built, layers.len, gdn_built, eng.compiledBanks(), rows, chunk_rows, share, int8_bytes / (1024 * 1024), secs });
    eng.logDualReady();
}

/// Channel-mode GPU MLP complement for one layer: gate/up rest-channel
/// VIEWS (rows [k..ffn) of the packed weights — axis-0, pinned safe) +
/// the down rest-K slice MATERIALIZED (axis-1). The slice boundary must
/// sit on the down weight's quant-group and packed-word grid, else the
/// layer declines by name.
pub fn aneBuildMlpRest(self: *Transformer, dw: *const DenseMlpWeights, k: u32, ffn: u32) !DenseMlpWeights {
    const qp = self.quantParamsHinted(dw.down_w, dw.down_s, ffn);
    if (qp.group_size == 0 or k % @as(u32, @intCast(qp.group_size)) != 0 or (k * @as(u32, @intCast(qp.bits))) % 32 != 0)
        return error.AneChanAlignment;
    const kk: c_int = @intCast(k);
    var rest: DenseMlpWeights = undefined;
    rest.gate_w = try self.aneSliceRowsView(dw.gate_w, kk);
    errdefer _ = mlx.mlx_array_free(rest.gate_w);
    rest.gate_s = try self.aneSliceRowsView(dw.gate_s, kk);
    errdefer if (rest.gate_s.ctx != null) {
        _ = mlx.mlx_array_free(rest.gate_s);
    };
    rest.gate_b = try self.aneSliceRowsView(dw.gate_b, kk);
    errdefer if (rest.gate_b.ctx != null) {
        _ = mlx.mlx_array_free(rest.gate_b);
    };
    rest.up_w = try self.aneSliceRowsView(dw.up_w, kk);
    errdefer _ = mlx.mlx_array_free(rest.up_w);
    rest.up_s = try self.aneSliceRowsView(dw.up_s, kk);
    errdefer if (rest.up_s.ctx != null) {
        _ = mlx.mlx_array_free(rest.up_s);
    };
    rest.up_b = try self.aneSliceRowsView(dw.up_b, kk);
    errdefer if (rest.up_b.ctx != null) {
        _ = mlx.mlx_array_free(rest.up_b);
    };
    rest.down_w = try self.aneSliceColsCopy(dw.down_w, @intCast(k * @as(u32, @intCast(qp.bits)) / 32));
    errdefer _ = mlx.mlx_array_free(rest.down_w);
    const g_from: c_int = @intCast(k / @as(u32, @intCast(qp.group_size)));
    rest.down_s = try self.aneSliceColsCopy(dw.down_s, g_from);
    errdefer if (rest.down_s.ctx != null) {
        _ = mlx.mlx_array_free(rest.down_s);
    };
    rest.down_b = try self.aneSliceColsCopy(dw.down_b, g_from);
    return rest;
}

/// Channel-mode GPU GDN complement: qkv/z rest-channel views written
/// into the layer's AneChanRest entry (all axis-0, zero-copy).
pub fn aneBuildGdnRest(self: *Transformer, entry: *AneChanRest, la: *const GdnIn, kq: u32, kz: u32) !void {
    errdefer entry.deinitGdn();
    entry.gdn_qkv_w = try self.aneSliceRowsView(la.qkv_w, @intCast(kq));
    entry.gdn_qkv_s = try self.aneSliceRowsView(la.qkv_s, @intCast(kq));
    entry.gdn_qkv_b = try self.aneSliceRowsView(la.qkv_b, @intCast(kq));
    entry.gdn_z_w = try self.aneSliceRowsView(la.z_w, @intCast(kz));
    entry.gdn_z_s = try self.aneSliceRowsView(la.z_s, @intCast(kz));
    entry.gdn_z_b = try self.aneSliceRowsView(la.z_b, @intCast(kz));
}

// ── tests ──

test "ane channel-split GPU complement: axis-0 slice VIEW of a packed weight through qmm vs its materialized copy" {
    // The channel-split GPU complement wants the rest-channel gate/up/qkv/z
    // weights as zero-copy axis-0 slice views of the resident packed
    // buffers (the down slice is axis-1 and always materialized). The
    // splitPackedGateUp rule says slice-born weights are materialized at
    // load — this pins whether an OFFSET row-contiguous view is in that
    // class on the current mlx pin, and the channel-mode design follows
    // the verdict.
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xA1C4);
    const rnd = prng.random();
    const n_out: c_int = 256;
    const k_in: c_int = 512;
    const half: c_int = n_out / 2;

    const w_shape = [_]c_int{ n_out, k_in };
    const w = try attn256RandBf16(rnd, &w_shape, s);
    defer _ = mlx.mlx_array_free(w);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    var bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&bi, triple, 2));

    const x_shape = [_]c_int{ 1, 8, k_in };
    const x = try attn256RandBf16(rnd, &x_shape, s);
    defer _ = mlx.mlx_array_free(x);

    var full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(full);
    try mlx.check(mlx.mlx_quantized_matmul(&full, x, wq, sc, bi, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", s));
    // Reference tail: output channels [half..n_out) of the FULL qmm.
    var ref_tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_tail);
    {
        const start = [_]c_int{ 0, 0, half };
        const stop = [_]c_int{ 1, 8, n_out };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&ref_tail, full, &start, 3, &stop, 3, &strides, 3, s));
    }

    // Axis-0 slice VIEWS of the packed weight + scales + biases.
    var wq_v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq_v);
    var sc_v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_v);
    var bi_v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bi_v);
    inline for (.{ .{ &wq_v, wq }, .{ &sc_v, sc }, .{ &bi_v, bi } }) |pair| {
        const src_shape = mlx.getShape(pair[1]);
        const start = [_]c_int{ half, 0 };
        const stop = [_]c_int{ n_out, src_shape[1] };
        const strides = [_]c_int{ 1, 1 };
        try mlx.check(mlx.mlx_slice(pair[0], pair[1], &start, 2, &stop, 2, &strides, 2, s));
    }
    var y_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_view);
    try mlx.check(mlx.mlx_quantized_matmul(&y_view, x, wq_v, sc_v, bi_v, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", s));

    // Materialized copies of the same slices.
    var y_copy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_copy);
    {
        const wq_c = try materializedOwnedCopy(s, wq_v);
        defer _ = mlx.mlx_array_free(wq_c);
        const sc_c = try materializedOwnedCopy(s, sc_v);
        defer _ = mlx.mlx_array_free(sc_c);
        const bi_c = try materializedOwnedCopy(s, bi_v);
        defer _ = mlx.mlx_array_free(bi_c);
        try mlx.check(mlx.mlx_quantized_matmul(&y_copy, x, wq_c, sc_c, bi_c, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", s));
    }

    const copy_err = try maxAbsDiffF32(y_copy, ref_tail, s);
    const view_err = try maxAbsDiffF32(y_view, ref_tail, s);
    std.debug.print("[ane-chan] qmm slice: copy_err={d} view_err={d}\n", .{ copy_err, view_err });
    try std.testing.expectEqual(@as(f32, 0), copy_err);
    try std.testing.expectEqual(@as(f32, 0), view_err);
}
