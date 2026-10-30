//! X12 measurement (mimo/gpu-warm; csrc/mimo_gpu_power.inc): the GPU's energy, its DVFS-state residency (IOReport's
//! "GPU Performance States" GPUPH: OFF, P1..P15, 24 MHz ticks) mapped to MHz through pmgr's "voltage-states9" table,
//! and its closed-loop thermal-limit residency ("CLTM-induced GPU Performance States": any state but NO_CLTM), between
//! two samples. System-wide counters, read without root and without any GPU work. Used by the `gpuwarm` bench (per
//! train) and the MTP profile (per request: `MIMO_GPU_POWER {...}`); no served path calls it.
const std = @import("std");
const builtin = @import("builtin");

pub const max_states = 32;

/// csrc/mimo_gpu_power.inc `MimoPowerDelta`.
pub const Delta = extern struct {
    gpu_energy_j: f64 = -1,
    seconds: f64 = 0,
    n_states: u32 = 0,
    has_cltm: u32 = 0,
    state_s: [max_states]f64 = @splat(0),
    cltm_s: f64 = 0,
    state_names: [max_states][8]u8 = @splat(@splat(0)),
};

comptime {
    if (@sizeOf(Delta) != 8 + 8 + 4 + 4 + 8 * max_states + 8 + 8 * max_states) @compileError("gpu_power: Delta is not the shim's layout");
}

const c = if (builtin.os.tag == .macos) struct {
    extern fn mimo_power_open() c_int;
    extern fn mimo_power_sample() ?*anyopaque;
    extern fn mimo_power_free(s: ?*anyopaque) void;
    extern fn mimo_power_delta(a: ?*anyopaque, b: ?*anyopaque, out: *Delta) c_int;
    extern fn mimo_power_freqs(mhz: [*]u32, cap: u32) c_int;
    extern fn mimo_power_last_error() [*:0]const u8;
} else struct {
    fn mimo_power_open() c_int {
        return -1;
    }
    fn mimo_power_sample() ?*anyopaque {
        return null;
    }
    fn mimo_power_free(_: ?*anyopaque) void {}
    fn mimo_power_delta(_: ?*anyopaque, _: ?*anyopaque, _: *Delta) c_int {
        return -1;
    }
    fn mimo_power_freqs(_: [*]u32, _: u32) c_int {
        return -1;
    }
    fn mimo_power_last_error() [*:0]const u8 {
        return "no IOReport";
    }
};

pub fn lastError() []const u8 {
    return std.mem.span(c.mimo_power_last_error());
}

/// The process's subscription (opened once) and the DVFS table (MHz per GPUPH state index; 0 = OFF or unknown).
pub const Sampler = struct {
    mhz: [max_states]u32 = @splat(0),
    n_mhz: u32 = 0,

    pub fn open() !Sampler {
        if (c.mimo_power_open() != 0) return error.GpuPowerUnavailable;
        var s: Sampler = .{};
        const n = c.mimo_power_freqs(&s.mhz, max_states);
        s.n_mhz = if (n > 0) @intCast(n) else 0;
        return s;
    }

    /// A cumulative sample (free it), or null.
    pub fn sample(_: *const Sampler) ?*anyopaque {
        return c.mimo_power_sample();
    }

    pub fn free(_: *const Sampler, s: ?*anyopaque) void {
        c.mimo_power_free(s);
    }

    pub fn delta(_: *const Sampler, a: ?*anyopaque, b: ?*anyopaque) !Delta {
        var d: Delta = .{};
        if (c.mimo_power_delta(a, b, &d) != 0) return error.GpuPowerDelta;
        return d;
    }
};

/// One interval's reading: energy and mean power, GPU active share, the active-time mean frequency, thermal limiting.
pub const Summary = struct {
    seconds: f64 = 0,
    energy_j: f64 = -1,
    watts: f64 = -1,
    active_s: f64 = 0,
    active_frac: f64 = 0,
    /// Active-time mean over the states with a known frequency (0 when none).
    mean_active_mhz: f64 = 0,
    /// The active-time share of the states at or above `top_mhz_floor` of the table's top frequency.
    top_frac: f64 = 0,
    cltm_s: f64 = -1,
};

/// A state at or above this share of the table's top frequency counts as "top" (`Summary.top_frac`).
pub const top_mhz_floor: f64 = 0.95;

pub fn summarize(d: Delta, mhz: []const u32) Summary {
    var s: Summary = .{ .seconds = d.seconds, .energy_j = d.gpu_energy_j, .cltm_s = if (d.has_cltm != 0) d.cltm_s else -1 };
    if (d.seconds > 0 and d.gpu_energy_j >= 0) s.watts = d.gpu_energy_j / d.seconds;
    var top: u32 = 0;
    for (mhz) |f| top = @max(top, f);
    var known_s: f64 = 0;
    var mhz_s: f64 = 0;
    var top_s: f64 = 0;
    const n: usize = @min(d.n_states, max_states);
    for (1..n) |k| {
        const t = d.state_s[k];
        s.active_s += t;
        if (k < mhz.len and mhz[k] > 0) {
            known_s += t;
            mhz_s += t * @as(f64, @floatFromInt(mhz[k]));
            if (@as(f64, @floatFromInt(mhz[k])) >= top_mhz_floor * @as(f64, @floatFromInt(top))) top_s += t;
        }
    }
    if (d.seconds > 0) s.active_frac = s.active_s / d.seconds;
    if (known_s > 0) {
        s.mean_active_mhz = mhz_s / known_s;
        s.top_frac = top_s / known_s;
    }
    return s;
}

/// `{"seconds": .., "gpu_energy_j": .., "gpu_w": .., "active_frac": .., "mean_active_mhz": .., "top_frac": .., "cltm_s": ..,
/// "mhz_share": {"<MHz>": <share of active time>, ...}}` (states with >= 0.5 % of the active time).
pub fn json(w: *std.Io.Writer, d: Delta, mhz: []const u32) !void {
    const s = summarize(d, mhz);
    try w.print("{{\"seconds\": {d:.3}, \"gpu_energy_j\": {d:.3}, \"gpu_w\": {d:.3}, \"active_frac\": {d:.4}, \"mean_active_mhz\": {d:.1}, \"top_frac\": {d:.4}, \"cltm_s\": {d:.3}, \"mhz_share\": {{", .{ s.seconds, s.energy_j, s.watts, s.active_frac, s.mean_active_mhz, s.top_frac, s.cltm_s });
    var first = true;
    const n: usize = @min(d.n_states, max_states);
    for (1..n) |k| {
        if (s.active_s <= 0 or d.state_s[k] / s.active_s < 0.005) continue;
        const name = std.mem.sliceTo(&d.state_names[k], 0);
        if (k < mhz.len and mhz[k] > 0) {
            try w.print("{s}\"{d}\": {d:.4}", .{ if (first) "" else ", ", mhz[k], d.state_s[k] / s.active_s });
        } else try w.print("{s}\"{s}\": {d:.4}", .{ if (first) "" else ", ", name, d.state_s[k] / s.active_s });
        first = false;
    }
    try w.writeAll("}}");
}

const testing = std.testing;

fn fake() Delta {
    var d: Delta = .{ .gpu_energy_j = 30.0, .seconds = 2.0, .n_states = 16, .has_cltm = 1, .cltm_s = 0.25 };
    d.state_s[0] = 1.0; // OFF
    d.state_s[3] = 0.25; // 636 MHz
    d.state_s[13] = 0.75; // 1620 MHz
    for (0..16) |k| {
        var b: [8]u8 = @splat(0);
        const t = if (k == 0) "OFF" else std.fmt.bufPrint(&b, "P{d}", .{k}) catch unreachable;
        @memcpy(d.state_names[k][0..t.len], t);
    }
    return d;
}

const table = [_]u32{ 0, 338, 486, 636, 796, 888, 988, 1084, 1182, 1278, 1374, 1470, 1578, 1620 };

test "mimo gpu power summary: energy over the interval, the active share, the active-time mean MHz through the DVFS table, the top states' share, thermal limiting; the line" {
    const s = summarize(fake(), &table);
    try testing.expectApproxEqAbs(@as(f64, 15.0), s.watts, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 1.0), s.active_s, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.5), s.active_frac, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.25 * 636 + 0.75 * 1620), s.mean_active_mhz, 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.75), s.top_frac, 1e-12);
    try testing.expectApproxEqAbs(@as(f64, 0.25), s.cltm_s, 1e-12);
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try json(&w, fake(), &table);
    try testing.expectEqualStrings("{\"seconds\": 2.000, \"gpu_energy_j\": 30.000, \"gpu_w\": 15.000, \"active_frac\": 0.5000, \"mean_active_mhz\": 1374.0, \"top_frac\": 0.7500, \"cltm_s\": 0.250, \"mhz_share\": {\"636\": 0.2500, \"1620\": 0.7500}}", w.buffered());
    // without a table: no MHz, the states by name
    var w2: std.Io.Writer = .fixed(&buf);
    try json(&w2, fake(), &.{});
    try testing.expect(std.mem.indexOf(u8, w2.buffered(), "\"P13\": 0.7500") != null);
}

test "mimo gpu power IOReport: the subscription opens, two samples give a delta with the GPUPH states, the DVFS table reads (read-only kernel counters; no GPU work); skipped where IOReport is absent" {
    var s = Sampler.open() catch return error.SkipZigTest;
    const a = s.sample() orelse return error.GpuPowerSample;
    defer s.free(a);
    const ts = std.c.timespec{ .sec = 0, .nsec = 20 * std.time.ns_per_ms };
    _ = std.c.nanosleep(&ts, null);
    const b = s.sample() orelse return error.GpuPowerSample;
    defer s.free(b);
    const d = try s.delta(a, b);
    try testing.expect(d.n_states >= 2);
    try testing.expectEqualStrings("OFF", std.mem.sliceTo(&d.state_names[0], 0));
    try testing.expect(d.seconds > 0.005 and d.seconds < 5);
    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    try json(&w, d, s.mhz[0..s.n_mhz]);
    std.debug.print("MIMO_GPU_POWER_CPU states={d} table_entries={d} top_mhz={d} sample={s}\n", .{ d.n_states, s.n_mhz, if (s.n_mhz > 0) s.mhz[s.n_mhz - 1] else 0, w.buffered() });
}
