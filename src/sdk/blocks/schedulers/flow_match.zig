//! Flow-matching sampler pieces shared by the media backends: the two time-shift transforms,
//! the resolution-dependent mu, the two sigma grids, the terminal stretch, and the Euler
//! step. Each backend's schedule is a composition of these in its own precision (the tests
//! rebuild FLUX.2, Krea, Qwen-Image, LTX, mage_flow, TRELLIS.2, MOSS-Image and ACE-Step
//! schedules from them bit for bit). Host math, except `eulerStep`.

const std = @import("std");
const ops = @import("../../../mlx/ops.zig");

const Graph = ops.Graph;
const Array = ops.Array;
const Dtype = ops.Dtype;

/// Linear-shift transform s·t / (1 + (s - 1)·t) (SD3 `shift`).
pub fn timeShift(comptime T: type, shift: T, t: T) T {
    return shift * t / (1.0 + (shift - 1.0) * t);
}

/// Exponential-shift transform eᵘ / (eᵘ + (1/t - 1)) (FLUX `time_shift`); t = 0 gives 0.
pub fn timeShiftExp(emu: f64, t: f64) f64 {
    return emu / (emu + (1.0 / t - 1.0));
}

/// mu on the line through (x1, y1), (x2, y2) at `seq` image tokens.
pub fn muLinear(seq: f64, x1: f64, y1: f64, x2: f64, y2: f64) f64 {
    const slope = (y2 - y1) / (x2 - x1);
    return slope * seq + (y1 - slope * x1);
}

/// linspace(1, 0, n + 1)[i] as 1 - i/n.
pub fn gridToZero(comptime T: type, i: usize, n: usize) T {
    return 1.0 - @as(T, @floatFromInt(i)) / @as(T, @floatFromInt(n));
}

/// linspace(1, 1/n, n)[i] (diffusers' FlowMatchEuler default grid); n = 1 gives 1.
pub fn gridToInvN(i: usize, n: usize) f64 {
    if (n == 1) return 1.0;
    const nf: f64 = @floatFromInt(n);
    return 1.0 + (1.0 / nf - 1.0) * @as(f64, @floatFromInt(i)) / (nf - 1.0);
}

/// Rescale so `last` (the final non-zero sigma) lands on `terminal`.
pub fn stretchToTerminal(sigma: f64, last: f64, terminal: f64) f64 {
    return 1.0 - (1.0 - sigma) / ((1.0 - last) / (1.0 - terminal));
}

/// x + v·dt, dt made in `sdt` (f32 in FLUX.2, Krea, MOSS-Image; the velocity's dtype in
/// Qwen-Image; mage_flow steps its f32 casts with f32).
pub fn eulerStep(parent: *Graph, x: Array, v: Array, dt: f32, sdt: Dtype) !Array {
    var g = parent.scope();
    defer g.deinit();
    return parent.keep(try g.add(x, try g.mul(v, try g.scalar(dt, sdt))));
}

// ── tests ──

const testing = std.testing;

// Oracles: each backend's schedule as written (module constants inlined).

/// flux.zig `computeSchedule` (sigmas only).
fn fluxSigmas(out: []f32, image_seq_len: u32, steps: u32) void {
    const isl: f64 = @floatFromInt(image_seq_len);
    const a1 = 8.73809524e-05;
    const b1 = 1.89833333;
    const a2 = 0.00016927;
    const b2 = 0.45666666;
    var mu: f64 = undefined;
    if (isl > 4300) {
        mu = a2 * isl + b2;
    } else {
        const m_200 = a2 * isl + b2;
        const m_10 = a1 * isl + b1;
        const a = (m_200 - m_10) / 190.0;
        const b = m_200 - 200.0 * a;
        mu = a * @as(f64, @floatFromInt(steps)) + b;
    }
    const emu = std.math.exp(mu);
    for (0..steps) |i| {
        const sl = 1.0 - (@as(f64, @floatFromInt(i)) * (1.0 - 1.0 / @as(f64, @floatFromInt(steps))) / @as(f64, @floatFromInt(steps - 1)));
        const shifted = emu / (emu + (1.0 / sl - 1.0));
        out[i] = @floatCast(shifted);
    }
    out[steps] = 0.0;
}

/// krea.zig `computeTimesteps`.
fn kreaTimesteps(ts: []f64, seq_len: u32, steps: u32) void {
    const y1: f64 = 0.5;
    const y2: f64 = 1.15;
    const align_ = 8 * 2;
    const x1: f64 = @floatFromInt((256 / align_) * (256 / align_));
    const x2: f64 = @floatFromInt((1280 / align_) * (1280 / align_));
    const slope = (y2 - y1) / (x2 - x1);
    const mu = slope * @as(f64, @floatFromInt(seq_len)) + (y1 - slope * x1);
    const emu = std.math.exp(mu);
    for (0..steps + 1) |i| {
        const lin = 1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps));
        if (lin <= 0.0) {
            ts[i] = 0.0;
        } else {
            const denom = emu + (1.0 / lin - 1.0);
            ts[i] = emu / denom;
        }
    }
}

/// qwen_image.zig `computeSigmas`.
fn qwenSigmas(out: []f32, steps: u32, image_seq_len: u32) void {
    const base_shift: f64 = 0.5;
    const max_shift: f64 = 0.9;
    const base_seq: f64 = 256;
    const max_seq: f64 = 8192;
    const terminal: f64 = 0.02;
    const m = (max_shift - base_shift) / (max_seq - base_seq);
    const mu = m * @as(f64, @floatFromInt(image_seq_len)) + (base_shift - m * base_seq);
    const emu = @exp(mu);
    const n: f64 = @floatFromInt(steps);
    const shifted = struct {
        fn at(e: f64, steps_f: f64, i: f64) f64 {
            const sigma = if (steps_f <= 1) 1.0 else 1.0 + i * (1.0 / steps_f - 1.0) / (steps_f - 1.0);
            return e / (e + (1.0 / sigma - 1.0));
        }
    }.at;
    const scale = (1.0 - shifted(emu, n, n - 1)) / (1.0 - terminal);
    for (0..steps) |i| out[i] = @floatCast(1.0 - (1.0 - shifted(emu, n, @floatFromInt(i))) / scale);
    out[steps] = 0;
}

/// ltx video.zig `dynamicShiftSchedule` (stretch on).
fn ltxSchedule(out: []f32, sig: []f64, num_steps: u32, num_tokens: u32, base_shift: f64, max_shift: f64, base_tokens: f64, max_tokens: f64, terminal: f64) void {
    const n = num_steps + 1;
    const slope = (max_shift - base_shift) / (max_tokens - base_tokens);
    const intercept = base_shift - slope * base_tokens;
    const sigma_shift = @as(f64, @floatFromInt(num_tokens)) * slope + intercept;
    const es = @exp(sigma_shift);
    for (0..n) |i| {
        const lin = 1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(num_steps));
        if (lin == 0.0) {
            sig[i] = 0.0;
        } else {
            sig[i] = es / (es + (1.0 / lin - 1.0));
        }
    }
    const last_nz = sig[num_steps - 1];
    const scale_factor = (1.0 - last_nz) / (1.0 - terminal);
    if (scale_factor != 0.0) {
        for (0..n) |i| {
            if (sig[i] != 0.0) sig[i] = 1.0 - (1.0 - sig[i]) / scale_factor;
        }
    }
    for (0..n) |i| out[i] = @floatCast(sig[i]);
}

/// mage_flow.zig `computeSigmas`.
fn mageSigmas(out: []f32, steps: u32, shift: f32) void {
    const n: usize = steps;
    const sh: f64 = shift;
    for (0..n) |i| {
        const base: f64 = if (n == 1)
            1.0
        else
            1.0 + (1.0 / @as(f64, @floatFromInt(n)) - 1.0) * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(n - 1));
        out[i] = @floatCast(sh * base / (1.0 + (sh - 1.0) * base));
    }
    out[n] = 0.0;
}

/// trellis2.zig `timeSchedule`.
fn trellisSchedule(out: []f64, rescale_t: f64) void {
    const steps = out.len - 1;
    for (out, 0..) |*t, i| {
        const lin = 1.0 - @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps));
        t.* = rescale_t * lin / (1 + (rescale_t - 1) * lin);
    }
}

/// acestep.zig `timestepSchedule`.
fn acestepSchedule(buf: []f32, shift: f32) void {
    const n = buf.len;
    for (0..n) |i| {
        var t: f32 = 1.0 - @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(n));
        if (shift != 1.0) t = shift * t / (1.0 + (shift - 1.0) * t);
        buf[i] = t;
    }
}

test "media blocks: the flow-match pieces rebuild every backend's sigma schedule bit for bit" {
    var a: [65]f32 = undefined;
    var b: [65]f32 = undefined;
    var a64: [65]f64 = undefined;
    var b64: [65]f64 = undefined;
    for ([_]u32{ 1, 2, 4, 8, 28, 50, 64 }) |steps| for ([_]u32{ 256, 1024, 4096, 4400, 9000 }) |seq| {
        const n: usize = steps;
        const seqf: f64 = @floatFromInt(seq);
        if (steps > 1) {
            fluxSigmas(&a, seq, steps);
            const mu = if (seqf > 4300) 0.00016927 * seqf + 0.45666666 else blk: {
                const m_200 = 0.00016927 * seqf + 0.45666666;
                const m_10 = 8.73809524e-05 * seqf + 1.89833333;
                // flux's empirical fit: mu at `steps` on the line through its 200- and 10-step values.
                break :blk muLinear(@floatFromInt(steps), 200, m_200, 10, m_10);
            };
            for (0..n) |i| b[i] = @floatCast(timeShiftExp(std.math.exp(mu), gridToInvN(i, n)));
            b[n] = 0;
            try testing.expectEqualSlices(f32, a[0 .. n + 1], b[0 .. n + 1]);

            qwenSigmas(&a, steps, seq);
            const e = @exp(muLinear(seqf, 256, 0.5, 8192, 0.9));
            const last = timeShiftExp(e, gridToInvN(n - 1, n));
            for (0..n) |i| b[i] = @floatCast(stretchToTerminal(timeShiftExp(e, gridToInvN(i, n)), last, 0.02));
            b[n] = 0;
            try testing.expectEqualSlices(f32, a[0 .. n + 1], b[0 .. n + 1]);

            ltxSchedule(&a, &a64, steps, seq, 0.95, 2.05, 1024, 4096, 0.1);
            const es = @exp(muLinear(seqf, 1024, 0.95, 4096, 2.05));
            const lz = timeShiftExp(es, gridToZero(f64, n - 1, n));
            for (0..n + 1) |i| {
                const sg = timeShiftExp(es, gridToZero(f64, i, n));
                b[i] = @floatCast(if (sg != 0) stretchToTerminal(sg, lz, 0.1) else sg);
            }
            try testing.expectEqualSlices(f32, a[0 .. n + 1], b[0 .. n + 1]);
        }
        kreaTimesteps(&a64, seq, steps);
        const ek = std.math.exp(muLinear(seqf, 256, 0.5, 6400, 1.15));
        for (0..n + 1) |i| b64[i] = timeShiftExp(ek, gridToZero(f64, i, n));
        try testing.expectEqualSlices(f64, a64[0 .. n + 1], b64[0 .. n + 1]);
    };
    for ([_]u32{ 1, 4, 8, 50 }) |steps| for ([_]f32{ 1.0, 3.0, 5.0, 0.7 }) |shift| {
        const n: usize = steps;
        mageSigmas(&a, steps, shift);
        for (0..n) |i| b[i] = @floatCast(timeShift(f64, shift, gridToInvN(i, n)));
        b[n] = 0;
        try testing.expectEqualSlices(f32, a[0 .. n + 1], b[0 .. n + 1]);

        trellisSchedule(a64[0 .. n + 1], shift);
        for (0..n + 1) |i| b64[i] = timeShift(f64, shift, gridToZero(f64, i, n));
        try testing.expectEqualSlices(f64, a64[0 .. n + 1], b64[0 .. n + 1]);

        acestepSchedule(a[0..n], shift);
        for (0..n) |i| b[i] = if (shift != 1.0) timeShift(f32, shift, gridToZero(f32, i, n)) else gridToZero(f32, i, n);
        try testing.expectEqualSlices(f32, a[0..n], b[0..n]);
    };
}

test "media blocks: eulerStep is bit-equal to FLUX.2's and Qwen-Image's step" {
    const mlx = @import("../../../mlx/ffi.zig");
    const kit = @import("../dit/testkit.zig");
    const s = kit.cpu();
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    for ([_]Dtype{ .bfloat16, .float32 }, 0..) |dt, i| {
        const x = try kit.randn(&g, &.{ 1, 16, 8 }, 700 + i, dt, 1);
        const v = try kit.randn(&g, &.{ 1, 16, 8 }, 710 + i, dt, 1);
        const dtv: f32 = -0.0371;
        // flux.zig: `mulA(nz, mlx_array_new_float(dt))` then `addA(latents, step)`.
        const dta = try g.adopt(mlx.mlx_array_new_float(dtv));
        try ops.expectBitEqual(try eulerStep(&g, x, v, dtv, .float32), try g.add(x, try g.mul(v, dta)));
        // qwen_image.zig: `mulScalar(v, dt)` (v's dtype) then `addA(img, dv)`.
        try ops.expectBitEqual(try eulerStep(&g, x, v, dtv, dt), try g.add(x, try g.mulScalar(v, dtv)));
    }
}
