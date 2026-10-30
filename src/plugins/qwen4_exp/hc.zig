//! Hyper-connection streams: the compiled read/write callbacks (`compileQwen4Hc`), the stand-ins, `HcRead`, and the
//! grouped RMS norm over the `hc` streams.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const ple = @import("ple.zig");
const verify_projection = @import("../../sdk/blocks/linear/verify_projection.zig");
const applyClosure = @import("../../engines/mlx/compiled.zig").applyClosure;
const closureIn = @import("../../engines/mlx/compiled.zig").closureIn;
const closureOut = @import("../../engines/mlx/compiled.zig").closureOut;
const scalarOf = @import("../../engines/mlx/load/weights.zig").scalarOf;
const Trunk = @import("trunk.zig").Trunk;
const standinOnes = @import("../../sdk/diag/standin.zig").standinOnes;
const standinRef = @import("../../sdk/diag/standin.zig").standinRef;
const QuantParams = @import("../../sdk/quant/params.zig").QuantParams;
const HC_FUSED_MAX_ROWS = @import("hc_fused.zig").HC_FUSED_MAX_ROWS;
const HcFusedOut = @import("hc_fused.zig").HcFusedOut;
const HcPending = @import("hc_fused.zig").HcPending;
const hcFusedEnabled = @import("hc_fused.zig").hcFusedEnabled;
const hcReadFused = @import("hc_fused.zig").hcReadFused;
const hcReadJoined = @import("hc_fused.zig").hcReadJoined;
const HcWeights = @import("weights.zig").HcWeights;
const PleWeights = @import("weights.zig").PleWeights;

/// qwen4_exp: compile the four hyper-connection elementwise tails.
pub fn compileQwen4Hc(self: *Trunk) void {
    const specs = .{
        .{ &self.compiled_hc_silu, &hcSiluCallback, true },
        .{ &self.compiled_hc_mix, &hcMixCallback, false },
        .{ &self.compiled_hc_inj, &hcInjCallback, true },
        .{ &self.compiled_hc_write, &hcWriteCallback, true },
    };
    var n: usize = 0;
    inline for (specs) |sp| {
        const raw = mlx.mlx_closure_new_func_payload(sp[1], @ptrCast(self), null);
        var compiled = mlx.mlx_closure{ .ctx = null };
        const rc = mlx.mlx_compile(&compiled, raw, sp[2]);
        _ = mlx.mlx_closure_free(raw);
        if (rc == 0 and compiled.ctx != null) {
            sp[0].* = compiled;
            n += 1;
        }
    }
    if (n > 0) log.info("[qwen4] hyper-connection tails compiled ({d}/4 kernels)\n", .{n});
}

pub fn hcSiluCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const x = closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(x);
    const y = self.silu(x) catch return -1;
    return closureOut(res, y);
}

pub fn hcMixCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const up4 = closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(up4);
    const n4 = closureIn(input, 1) orelse return -1;
    defer _ = mlx.mlx_array_free(n4);
    var mix = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mix);
    if (mlx.mlx_sigmoid(&mix, up4, self.s) != 0) return -1;
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    if (mlx.mlx_multiply(&prod, mix, n4, self.s) != 0) return -1;
    var mixed = mlx.mlx_array_new();
    if (mlx.mlx_mean_axis(&mixed, prod, 2, false, self.s) != 0) return -1;
    return closureOut(res, mixed);
}

pub fn hcInjCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const x = closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(x);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    if (mlx.mlx_sigmoid(&sig, x, self.s) != 0) return -1;
    const two = scalarOf(2.0, mlx.mlx_array_dtype(x), self.s) catch return -1;
    defer _ = mlx.mlx_array_free(two);
    var g = mlx.mlx_array_new();
    if (mlx.mlx_multiply(&g, sig, two, self.s) != 0) return -1;
    return closureOut(res, g);
}

pub fn hcWriteCallback(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const self: *Trunk = @ptrCast(@alignCast(payload.?));
    const stream4 = closureIn(input, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(stream4);
    const out4 = closureIn(input, 1) orelse return -1;
    defer _ = mlx.mlx_array_free(out4);
    const inj = closureIn(input, 2) orelse return -1;
    defer _ = mlx.mlx_array_free(inj);
    var add4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(add4);
    if (mlx.mlx_multiply(&add4, out4, inj, self.s) != 0) return -1;
    var next = mlx.mlx_array_new();
    if (mlx.mlx_add(&next, stream4, add4, self.s) != 0) return -1;
    return closureOut(res, next);
}

// ── MoE forward pass (Qwen3.5 + Gemma 4) ──

// ── qwen4_exp (Qwen3.8-Flash-Next) ──

pub const HcRead = struct {
    mixed: mlx.mlx_array, // [B, S, hidden]
    inj: mlx.mlx_array, // [B, S, hc, 1] (null-ctx on the mixer)
    pub fn deinit(self: *@This()) void {
        _ = mlx.mlx_array_free(self.mixed);
        if (self.inj.ctx != null) _ = mlx.mlx_array_free(self.inj);
    }
};

/// `hcRead` that first applies a deferred write (`*pending`, consumed):
/// fused when the N kernel takes it, else the chain's hcWrite. `h` is
/// replaced by the written stream either way.
/// Diagnostic (`QWEN4_STANDIN=hc`): stream 0 as the block input, unit
/// gates — the read kernels drop out, the writes stay live.
pub fn hcReadStandin(self: *Trunk, h: mlx.mlx_array, batch: c_int, seq_len: c_int) !HcRead {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const start = [_]c_int{ 0, 0, 0 };
    const stop = [_]c_int{ batch, seq_len, hidden };
    const strides = [_]c_int{ 1, 1, 1 };
    var mixed = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&mixed, h, &start, 3, &stop, 3, &strides, 3, self.s));
    errdefer _ = mlx.mlx_array_free(mixed);
    const inj_shape = [_]c_int{ batch, seq_len, hc, 1 };
    return .{ .mixed = mixed, .inj = try standinOnes(&inj_shape, self.s) };
}

pub fn hcReadPending(self: *Trunk, h: *mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int, pending: *?HcPending) !HcRead {
    if (pending.*) |*pd| {
        defer pending.* = null;
        defer pd.deinit();
        if (try self.hcReadFusedFor(h.*, w, batch, seq_len, pd.*)) |o| {
            _ = mlx.mlx_array_free(h.*);
            h.* = o.stream;
            return .{ .mixed = o.mixed, .inj = o.inj };
        }
        h.* = try self.hcWrite(h.*, pd.out, pd.inj, batch, seq_len);
    }
    return self.hcRead(h.*, w, batch, seq_len);
}

pub fn hcReadFusedFor(self: *Trunk, stream: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int, pend: ?HcPending) !?HcFusedOut {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const hp = @import("../../sdk/blocks/hc/hc_prefill.zig");
    if (hp.eligible(batch, seq_len, self.config.hc_count, self.config.hidden_size, w.inject_flat, mlx.mlx_array_dtype(stream))) {
        if (try hp.norm(self.s, stream, w.norm_w, w.inject_flat, self.rms_eps_arr, batch, seq_len, if (pend) |p| .{ .out = p.out, .inj = p.inj } else null)) |n| {
            defer n.deinit();
            var read = try self.hcReadNormed(n.normalized, w, batch, seq_len, n.raw_inject);
            errdefer read.deinit();
            return .{ .stream = try standinRef(n.stream), .mixed = read.mixed, .inj = read.inj };
        }
        if (try @import("../../sdk/blocks/hc/hc_prefill_cuda.zig").norm(self.s, stream, w.norm_w, self.rms_eps_arr, batch, seq_len, if (pend) |p| .{ .out = p.out, .inj = p.inj } else null)) |n| {
            defer n.deinit();
            var read = try self.hcReadNormed(n.normalized, w, batch, seq_len, null);
            errdefer read.deinit();
            return .{ .stream = try standinRef(n.stream), .mixed = read.mixed, .inj = read.inj };
        }
    }
    if (batch * seq_len > HC_FUSED_MAX_ROWS or (w.inject_w.ctx != null and w.inject_flat.ctx == null)) return null;
    if (w.down_rows.ctx != null) return hcReadFused(self.s, stream, batch, seq_len, w.norm_w, w.down_rows, .{ .ctx = null }, .{ .ctx = null }, w.up_rows, .{ .ctx = null }, .{ .ctx = null }, w.inject_flat, self.config.rms_norm_eps, hc, hidden, 16, 16, pend);
    if (w.down_s.ctx == null or w.up_s.ctx == null) return null;
    const dqp = self.quantParamsHinted(w.down_w, w.down_s, @intCast(hc * hidden));
    const uqp = self.quantParamsFor(w.up_w, w.up_s);
    if (dqp.bits != uqp.bits or dqp.group_size != uqp.group_size or dqp.mode != .affine or uqp.mode != .affine) return null;
    return hcReadFused(self.s, stream, batch, seq_len, w.norm_w, w.down_w, w.down_s, w.down_b, w.up_w, w.up_s, w.up_b, w.inject_flat, self.config.rms_norm_eps, hc, hidden, dqp.bits, dqp.group_size, pend);
}

pub fn addVerifyPle(self: *Trunk, row: anytype, pw: *const PleWeights, layer: usize, index: usize) !void {
    try self.hcFlush(&row.h, 1, row.seq, &row.pending);
    const add = try self.pleForward(row.ctx, row.h, row.tokens, pw, &row.entries[layer], layer, 1, row.seq);
    defer _ = mlx.mlx_array_free(add);
    var with_ple = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(with_ple);
    if (comptime @import("builtin").is_test) {
        if (ple.verify_fault_after_ple_row == index) mlx.fault.arm(1);
    }
    try mlx.check(mlx.mlx_add(&with_ple, row.h, add, self.s));
    _ = mlx.mlx_array_free(row.h);
    row.h = with_ple;
}

pub fn hcReadVerifyRows(self: *Trunk, rows: anytype, w: *const HcWeights) !?[]HcRead {
    if (self.qwen4 == null or self.config.hc_count != 4 or self.config.hidden_size != 2560 or rows.len < 2 or rows.len > 8) return null;
    const width = rows[0].seq;
    const has_pending = rows[0].pending != null;
    var inputs: [8]mlx.mlx_array = undefined;
    var pending: [8]HcPending = undefined;
    for (rows, 0..) |row, i| {
        if (row.seq != width or (row.pending != null) != has_pending) return null;
        inputs[i] = row.h;
        if (row.pending) |p| pending[i] = p;
    }
    const dense = w.down_rows.ctx != null;
    const dqp: QuantParams = if (dense) .{ .bits = 16, .group_size = 16 } else self.quantParamsHinted(w.down_w, w.down_s, 10240);
    const uqp: QuantParams = if (dense) dqp else self.quantParamsFor(w.up_w, w.up_s);
    if (dqp.bits != uqp.bits or dqp.group_size != uqp.group_size or dqp.mode != .affine or uqp.mode != .affine) return null;
    const joined = (try hcReadJoined(self.allocator, self.s, inputs[0..rows.len], w.*, width, self.config.rms_norm_eps, dqp.bits, dqp.group_size, if (has_pending) pending[0..rows.len] else null)) orelse return null;
    defer self.allocator.free(joined);
    errdefer for (joined) |value| inline for (.{ "mixed", "inj", "stream" }) |name| {
        if (@field(value, name).ctx != null) _ = mlx.mlx_array_free(@field(value, name));
    };
    const result = try self.allocator.alloc(HcRead, rows.len);
    for (rows, joined, result) |*row, value, *read| {
        read.* = .{ .mixed = value.mixed, .inj = value.inj };
        if (has_pending) {
            _ = mlx.mlx_array_free(row.h);
            row.h = value.stream;
            row.pending.?.deinit();
            row.pending = null;
        }
    }
    verify_projection.mtp_verify_kernel_calls[1] +%= 1;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[batched] joined hyper-connection verify reads engaged\n", .{});
    }
    return result;
}

/// Defer `stream += out * inj` to the next read when the fused read will
/// take it (decode/verify/batched widths); otherwise write now.
pub fn hcWriteOrDefer(self: *Trunk, h: *mlx.mlx_array, out: mlx.mlx_array, inj: mlx.mlx_array, batch: c_int, seq_len: c_int, inject_flat: mlx.mlx_array, pending: *?HcPending) !void {
    std.debug.assert(pending.* == null);
    const prefill = @import("../../sdk/blocks/hc/hc_prefill.zig").eligible(batch, seq_len, self.config.hc_count, self.config.hidden_size, inject_flat, mlx.mlx_array_dtype(h.*)) and mlx.mlx_array_dtype(inj) == .bfloat16;
    if ((prefill or (batch * seq_len <= HC_FUSED_MAX_ROWS and hcFusedEnabled())) and mlx.mlx_array_dtype(h.*) == mlx.mlx_array_dtype(out)) {
        var pd: HcPending = undefined;
        pd.out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(pd.out);
        try mlx.check(mlx.mlx_array_set(&pd.out, out));
        pd.inj = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(pd.inj);
        try mlx.check(mlx.mlx_array_set(&pd.inj, inj));
        pending.* = pd;
        return;
    }
    h.* = try self.hcWrite(h.*, out, inj, batch, seq_len);
}

pub fn hcFlush(self: *Trunk, h: *mlx.mlx_array, batch: c_int, seq_len: c_int, pending: *?HcPending) !void {
    if (pending.*) |*pd| {
        defer pending.* = null;
        defer pd.deinit();
        h.* = try self.hcWrite(h.*, pd.out, pd.inj, batch, seq_len);
    }
}

/// Grouped RMS norm over the last `hidden` of each of the `hc` streams:
/// `x [B,S,hc*hidden]` → `[B,S,hc,hidden]`, weight `[hc,hidden]` (already
/// carrying the reference's +1). Caller frees.
pub fn hcGroupNorm(self: *Trunk, x: mlx.mlx_array, w: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &shape4, 4, self.s));
    var n4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n4);
    try mlx.check(mlx.mlx_fast_rms_norm(&n4, x4, self.ones_hidden, self.config.rms_norm_eps, self.s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&out, n4, w, self.s));
    return out;
}

/// Qwen4ExpTextGatedResidual.forward: the block input is a sigmoid-mixed
/// mean of the normalized streams; the write gates are `2·σ(inject/hc)`.
pub fn hcRead(self: *Trunk, stream: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int) !HcRead {
    if (try self.hcReadFusedFor(stream, w, batch, seq_len, null)) |o| {
        if (o.stream.ctx != null) _ = mlx.mlx_array_free(o.stream);
        return .{ .mixed = o.mixed, .inj = o.inj };
    }
    const n4 = try self.hcGroupNorm(stream, w.norm_w, batch, seq_len);
    defer _ = mlx.mlx_array_free(n4);
    return self.hcReadNormed(n4, w, batch, seq_len, null);
}

pub fn hcReadNormed(self: *Trunk, n4: mlx.mlx_array, w: *const HcWeights, batch: c_int, seq_len: c_int, raw_inject: ?mlx.mlx_array) !HcRead {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const flat_shape = [_]c_int{ batch, seq_len, hc * hidden };
    var n_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n_flat);
    try mlx.check(mlx.mlx_reshape(&n_flat, n4, &flat_shape, 3, self.s));

    // down/inject carry the reference's `/hc` in their weights (load-time fold).
    const down = try self.qmatmul(n_flat, w.down_w, w.down_s, w.down_b);
    defer _ = mlx.mlx_array_free(down);
    const act = (try applyClosure(self.compiled_hc_silu, &.{down})) orelse try self.silu(down);
    defer _ = mlx.mlx_array_free(act);
    const up = try self.qmatmul(act, w.up_w, w.up_s, w.up_b);
    defer _ = mlx.mlx_array_free(up);
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var up4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up4);
    try mlx.check(mlx.mlx_reshape(&up4, up, &shape4, 4, self.s));
    var mixed: mlx.mlx_array = undefined;
    const prefill_mix = if (@import("../../sdk/blocks/hc/hc_prefill.zig").eligible(batch, seq_len, self.config.hc_count, self.config.hidden_size, w.inject_flat, mlx.mlx_array_dtype(n4)))
        (try @import("../../sdk/blocks/hc/hc_prefill.zig").mix(self.s, up4, n4, batch, seq_len)) orelse try @import("../../sdk/blocks/hc/hc_prefill_cuda.zig").mix(self.s, up4, n4)
    else
        null;
    if (prefill_mix) |m| {
        mixed = m;
    } else if (try applyClosure(self.compiled_hc_mix, &.{ up4, n4 })) |m| {
        mixed = m;
    } else {
        var mix = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(mix);
        try mlx.check(mlx.mlx_sigmoid(&mix, up4, self.s));
        var prod = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(prod);
        try mlx.check(mlx.mlx_multiply(&prod, mix, n4, self.s));
        mixed = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_mean_axis(&mixed, prod, 2, false, self.s));
    }
    errdefer _ = mlx.mlx_array_free(mixed);

    var inj = mlx.mlx_array_new();
    if (w.inject_w.ctx != null) {
        const raw = if (raw_inject) |r| try standinRef(r) else try self.qmatmul(n_flat, w.inject_w, w.inject_s, w.inject_b);
        defer _ = mlx.mlx_array_free(raw);
        const g = (try applyClosure(self.compiled_hc_inj, &.{raw})) orelse blk: {
            var sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sig);
            try mlx.check(mlx.mlx_sigmoid(&sig, raw, self.s));
            const two = try scalarOf(2.0, mlx.mlx_array_dtype(raw), self.s);
            defer _ = mlx.mlx_array_free(two);
            var g2 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&g2, sig, two, self.s));
            break :blk g2;
        };
        defer _ = mlx.mlx_array_free(g);
        const inj_shape = [_]c_int{ batch, seq_len, hc, 1 };
        try mlx.check(mlx.mlx_reshape(&inj, g, &inj_shape, 4, self.s));
    }
    return .{ .mixed = mixed, .inj = inj };
}

/// `stream + flatten(out[..., None, :] * inj)`. Consumes `stream`.
pub fn hcWrite(self: *Trunk, stream: mlx.mlx_array, out: mlx.mlx_array, inj: mlx.mlx_array, batch: c_int, seq_len: c_int) !mlx.mlx_array {
    const hc: c_int = @intCast(self.config.hc_count);
    const hidden: c_int = @intCast(self.config.hidden_size);
    const out_shape = [_]c_int{ batch, seq_len, 1, hidden };
    var out4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out4);
    try mlx.check(mlx.mlx_reshape(&out4, out, &out_shape, 4, self.s));
    const flat_shape = [_]c_int{ batch, seq_len, hc * hidden };
    const shape4 = [_]c_int{ batch, seq_len, hc, hidden };
    var stream4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(stream4);
    try mlx.check(mlx.mlx_reshape(&stream4, stream, &shape4, 4, self.s));
    var next4: mlx.mlx_array = undefined;
    if (try applyClosure(self.compiled_hc_write, &.{ stream4, out4, inj })) |n| {
        next4 = n;
    } else {
        var add4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(add4);
        try mlx.check(mlx.mlx_multiply(&add4, out4, inj, self.s));
        next4 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&next4, stream4, add4, self.s));
    }
    defer _ = mlx.mlx_array_free(next4);
    var next = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(next);
    try mlx.check(mlx.mlx_reshape(&next, next4, &flat_shape, 3, self.s));
    _ = mlx.mlx_array_free(stream);
    return next;
}
