//! engines/mlx/diag/profile.
//! Owns debugTraceHead, DecodeProf, layer_cap_env, layerCap, decode_prof, decode_prof_enabled, and 5 more.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");

pub fn debugTraceHead(label: []const u8, arr: mlx.mlx_array, s: mlx.mlx_stream) void {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    mlx.check(mlx.mlx_astype(&f, arr, .float32, s)) catch return;
    mlx.check(mlx.mlx_array_eval(f)) catch return;
    const d = mlx.mlx_array_data_float32(f) orelse return;
    std.debug.print("{s}: {any}\n", .{ label, d[0..8] });
}

// ── Decode sub-block profiler (diag decode-profile) ──
// Serializes the S=1 decode forward with per-sub-block evals to attribute the
// per-token GPU cost across embed / attn / mlp / lm_head. DIAGNOSTIC ONLY: the
// evals defeat the async pipeline (absolute tok/s drops while active), so the
// signal is the RELATIVE split plus the serialized total vs the pipelined
// per-token time — a serialized total near the pipelined per-token means we are
// compute-bound; far below it means the gap is dispatch/pipeline overhead.
const DecodeProf = struct {
    calls: u64 = 0,
    embed_ns: u64 = 0,
    attn_ns: u64 = 0,
    gdn_ns: u64 = 0,
    mlp_ns: u64 = 0,
    lmhead_ns: u64 = 0,
    // MoE internals (summed across all MoE layers per token)
    moe_router_ns: u64 = 0,
    moe_experts_ns: u64 = 0,
    moe_shared_ns: u64 = 0,
};

/// DIAGNOSTIC (diag layer-cap): clamp the layer loop so a ms-vs-layers
/// sweep separates per-layer slope from fixed cost. Unset — the only shipped
/// state — returns the model's real depth.
var layer_cap_env: ?usize = null;

pub fn layerCap(n: usize) usize {
    if (layer_cap_env == null) {
        layer_cap_env = blk: {
            const raw = env.get(.LAYER_CAP) orelse break :blk 0;
            break :blk std.fmt.parseInt(usize, std.mem.sliceTo(raw, 0), 10) catch 0;
        };
    }
    const cap = layer_cap_env.?;
    return if (cap == 0) n else @min(cap, n);
}

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const ProfileState = struct {
    decode_prof: DecodeProf = .{},
    decode_prof_enabled: ?bool = null,
};
pub const profileState_shared = @import("../../../sdk/util/shared.zig").Shared(ProfileState, "decode_profile", .{});
pub inline fn profileState() *ProfileState {
    return profileState_shared.get();
}

pub const ProfClock = struct {
    io: std.Io,
    start: std.Io.Timestamp,
    mark_ns: u64 = 0,
    pub fn init() ProfClock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .boot) };
    }
    pub fn lap(self: *ProfClock) u64 {
        const cum: u64 = @intCast(self.start.untilNow(self.io, .boot).nanoseconds);
        const d = cum - self.mark_ns;
        self.mark_ns = cum;
        return d;
    }
};

pub fn decodeProfileEnabled() bool {
    if (profileState().decode_prof_enabled) |v| return v;
    const v = env.get(.DECODE_PROFILE) != null;
    profileState().decode_prof_enabled = v;
    return v;
}

/// Extra per-MoE-layer elementwise dispatches injected by the sizing probe
/// (diag `dispatch-probe`). 0 = off, which is every non-diagnostic run.
var dispatch_probe_cached: ?usize = null;
pub fn dispatchProbeCount() usize {
    if (dispatch_probe_cached) |v| return v;
    const raw = env.get(.DISPATCH_PROBE);
    const n: usize = if (raw) |r| std.fmt.parseInt(usize, std.mem.sliceTo(r, 0), 10) catch 0 else 0;
    dispatch_probe_cached = n;
    return n;
}

pub fn decodeProfReport() void {
    const n = profileState().decode_prof.calls;
    if (n == 0) return;
    const per = struct {
        fn ms(ns: u64, calls: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(calls)) / 1.0e6;
        }
    };
    const e = per.ms(profileState().decode_prof.embed_ns, n);
    const a = per.ms(profileState().decode_prof.attn_ns, n);
    const g = per.ms(profileState().decode_prof.gdn_ns, n);
    const m = per.ms(profileState().decode_prof.mlp_ns, n);
    const l = per.ms(profileState().decode_prof.lmhead_ns, n);
    const mr = per.ms(profileState().decode_prof.moe_router_ns, n);
    const mx_ = per.ms(profileState().decode_prof.moe_experts_ns, n);
    const ms_ = per.ms(profileState().decode_prof.moe_shared_ns, n);
    log.info("[decode-prof] n={d} serial/tok={d:.2}ms  embed={d:.2} attn={d:.2} gdn={d:.2} mlp={d:.2} lmhead={d:.2} ms | moe: router={d:.2} experts={d:.2} shared={d:.2} ms\n", .{ n, e + a + g + m + l, e, a, g, m, l, mr, mx_, ms_ });
    profileState().decode_prof = .{};
}
