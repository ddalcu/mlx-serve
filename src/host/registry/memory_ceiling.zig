//! GPU working set, wired limit floor; static/current GPU ceiling

const std = @import("std");
const env = @import("../../env.zig");
const builtin = @import("builtin");
const mlx = @import("../../mlx/ffi.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const metrics = @import("../status.zig");
const ane_mod = @import("../../sdk/ane/offload.zig");
const testing = std.testing;
const http_server = @import("../http/server.zig");

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// Get the max buffer allocation limit (~75% of system unified memory).
/// `hw.memsize` on Darwin; on Linux there is no unified-memory sysctl, so the
/// same 75%-of-physical-RAM heuristic runs against /proc/meminfo.
fn getMetalBufferLimit() u64 {
    var mem: u64 = 0;
    if (comptime builtin.os.tag.isDarwin()) {
        var len: usize = @sizeOf(u64);
        _ = sysctlbyname("hw.memsize", @ptrCast(&mem), &len, null, 0);
    } else {
        mem = metrics.getTotalMemBytes();
    }
    if (mem == 0) return 8 * 1024 * 1024 * 1024; // fallback 8GB
    return mem * 75 / 100;
}

/// Metal's recommended max working-set size for the default device — the real
/// ceiling whose breach throws `[METAL] … Insufficient Memory` from the
/// command-buffer completion handler (which terminates the process: ggml's
/// global std::terminate handler prints the backtrace, but the throw is MLX's).
/// `getMetalBufferLimit()` (75% of physical RAM) over-estimates this on
/// small-RAM Macs — a 16 GB Mac reports ~11.9 GB recommended vs the 12 GB that
/// hw.memsize×0.75 yields — so budgeting against hw.memsize lets auto-context
/// oversubscribe. Falls back to `getMetalBufferLimit()` when the device query
/// is unavailable (CI / non-Metal hosts).
fn getGpuWorkingSetLimit() u64 {
    if (overrides().static_ceiling) |v| return v;
    const max_rec = mlx.maxRecommendedWorkingSet();
    return if (max_rec > 0) max_rec else getMetalBufferLimit();
}

/// PURE (unit-testable): the real ceiling a NEW MLX allocation must fit under.
///
/// `working_set_limit` (Metal's `max_recommended_working_set_size`) is a STATIC
/// device maximum — it assumes the whole GPU working set is MLX's to claim and
/// is BLIND to memory held by anything else on the machine. `mlx_footprint`
/// (MLX active + reclaimable cache) + `free_system` (physically free RAM) is
/// what MLX can actually reach RIGHT NOW; when another process holds a big
/// chunk of unified memory the second term binds and the ceiling collapses.
///
/// #64 (2026-07): a Claude Code session on a 128 GB Mac ran a docker-compose
/// stack (firecrawl/rabbitmq/postgres/playwright) holding tens of GB. The guard
/// budgeted against the static 115 GB max, admitted a 90 K-token MoE prefill,
/// and Metal threw `Insufficient Memory` from the command-buffer completion
/// handler — an UNCATCHABLE async C++ throw that terminates the process (the
/// libllama frames in the backtrace are just its global std::terminate handler;
/// the throw is MLX's). Capping by real free RAM makes the two prefill guards
/// reject/clamp before that allocation is ever submitted.
pub fn physicalMemoryCeiling(working_set_limit: u64, mlx_footprint: u64, free_system: u64) u64 {
    return @min(working_set_limit, mlx_footprint +| free_system);
}

/// How far under the enforced wired limit a plan may reach: past the limit Metal returns
/// zeros before an uncatchable abort, so a real transient's worth of margin stays unplanned.
pub const WIRED_LIMIT_MARGIN_BYTES: u64 = 8 << 30;

/// PURE: the floor an explicitly raised `iogpu.wired_limit_mb` puts under the ceiling; 0 when
/// the sysctl is absent or at the macOS default (75% of RAM), which leaves the ceiling as is.
/// A floor, not a cap: the working-set term already tracks the sysctl. The term it lifts is
/// the free-RAM one, which counts other processes' pages as gone; the operator who raised the
/// enforced limit has said the GPU may claim that much. `margin` (`--wired-margin-gib`)
/// is what stays unplanned under the limit: other processes' wired GPU memory, the
/// allocator cache and the estimator's slack; past the limit Metal returns zeros.
pub fn wiredLimitFloor(wired_limit: u64, total_ram: u64, margin: u64) u64 {
    if (wired_limit == 0 or total_ram == 0) return 0;
    if (wired_limit <= total_ram * 75 / 100) return 0; // the macOS default: no declaration
    return @min(wired_limit -| margin, total_ram -| margin);
}

pub fn parseWiredMarginGib(raw: []const u8) error{InvalidWiredMargin}!u64 {
    const n = std.fmt.parseInt(u32, raw, 10) catch return error.InvalidWiredMargin;
    if (n < 2 or n > 32) return error.InvalidWiredMargin;
    return @as(u64, n) << 30;
}

/// PURE: the ceiling with the wired-limit floor applied. `wired_floor == 0` returns
/// `physicalMemoryCeiling` byte for byte.
pub fn gpuCeilingWithWiredFloor(
    working_set_limit: u64,
    mlx_footprint: u64,
    free_system: u64,
    wired_floor: u64,
) u64 {
    const physical = physicalMemoryCeiling(working_set_limit, mlx_footprint, free_system);
    return @max(physical, wired_floor);
}

/// Test seam: `iogpu.wired_limit_mb` in MB; `null` reads the machine.
/// The operator's overrides of the machine's memory limits, one copy for every compile unit (`Shared`).
pub const Overrides = struct { wired_limit_mb: ?u64 = null, static_ceiling: ?u64 = null };
pub const overrides_shared = @import("../../sdk/util/shared.zig").Shared(Overrides, "memory_overrides", .{});
pub inline fn overrides() *Overrides {
    return overrides_shared.get();
}

var wired_limit_bytes_cached: u64 = 0;

var wired_limit_read: bool = false;

/// `iogpu.wired_limit_mb` in bytes, read ONCE per process: re-reading it per admission would
/// let the ceiling move under a live request. 0 when the OID is absent.
pub fn wiredLimitBytes() u64 {
    if (overrides().wired_limit_mb) |mb| return mb *| (1024 * 1024);
    // iogpu.wired_limit_mb is an Apple GPU sysctl; Linux has no wired-limit
    // concept, so the query reads as absent (0), same as a non-Apple-Silicon Mac.
    if (comptime !builtin.os.tag.isDarwin()) return 0;
    if (wired_limit_read) return wired_limit_bytes_cached;
    var v: u32 = 0;
    var len: usize = @sizeOf(u32);
    wired_limit_bytes_cached = if (sysctlbyname("iogpu.wired_limit_mb", @ptrCast(&v), &len, null, 0) == 0)
        @as(u64, v) * 1024 * 1024
    else
        0;
    wired_limit_read = true;
    return wired_limit_bytes_cached;
}

var wired_floor_logged = std.atomic.Value(bool).init(false);

/// The floor for THIS model, gated on `ModelConfig.longCtxGated()`. `null` config (no model
/// resolved yet) takes no floor.
fn wiredCeilingFloorFor(config: ?*const model_mod.ModelConfig) u64 {
    return wiredCeilingFloorForRam(config, metrics.getTotalMemBytes());
}

/// PURE: the floor for a machine with `total_ram` bytes; the wrapper above reads the machine.
pub fn wiredCeilingFloorForRam(config: ?*const model_mod.ModelConfig, total_ram: u64) u64 {
    const c = config orelse return 0;
    if (!c.longCtxGated()) return 0;
    const floor = wiredLimitFloor(wiredLimitBytes(), total_ram, http_server.serverState().wired_limit_margin_bytes);
    if (floor > 0 and wired_floor_logged.cmpxchgStrong(false, true, .monotonic, .monotonic) == null) {
        log.info("[mem] ceiling {d} MB from iogpu.wired_limit_mb={d} (working set {d} MB, margin {d} MB)\n", .{
            floor >> 20,
            wiredLimitBytes() >> 20,
            getGpuWorkingSetLimit() >> 20,
            http_server.serverState().wired_limit_margin_bytes >> 20,
        });
    }
    return floor;
}

/// Stands in for the machine's working-set limit (Metal's recommended working set
/// or the wired limit): tests (CI runners have 7 GB) and `applyGpuCeilingEnv`.
/// Load-time bills price this static term (stable across boots); request-time admission reads live RAM.

pub fn staticGpuMemoryCeiling() u64 {
    return getGpuWorkingSetLimit();
}

/// Diagnostic: `MLX_SERVE_GPU_CEILING_MB=N` stands in for the working-set limit in every bill
/// (absent or 0 = the machine's). It bounds the guards' arithmetic, not what Metal will grant.
pub fn applyGpuCeilingEnv() void {
    const raw = env.get(.MLX_SERVE_GPU_CEILING_MB) orelse return;
    const mb = std.fmt.parseInt(u64, std.mem.sliceTo(raw, 0), 10) catch return;
    if (mb == 0) return;
    overrides().static_ceiling = mb << 20;
    log.warn("[diag] GPU ceiling forced to {d} MB (MLX_SERVE_GPU_CEILING_MB)\n", .{mb});
}

/// THE ceiling helper: `available`, `/props`, the hot-cache clamp, the auto-context pin and the
/// admission bill all read it, so the wired-limit floor cannot reach some of them and not others.
pub fn currentGpuMemoryCeiling(config: ?*const model_mod.ModelConfig, active_mem: u64) u64 {
    // The ANE's int8 copies are wired host buffers: invisible to MLX's own
    // accounting, but genuinely gone from free RAM. Left to leak in through
    // the noisy free-RAM term they made auto-context swing across boots of the
    // SAME build (measured 5,120 / 10,240 / 55,296 tokens, 27B iQ on a 32 GB
    // M1 Pro). Add them back and subtract the known figure once. Zero — and so
    // exactly the old expression — whenever no offload is resident.
    const ane_bytes = ane_mod.aneState().live_int8_bytes.load(.monotonic);
    var cache_mem: usize = 0;
    _ = mlx.mlx_get_cache_memory(&cache_mem);
    return gpuCeilingWithWiredFloor(
        getGpuWorkingSetLimit(),
        active_mem +| @as(u64, cache_mem),
        (metrics.getAvailableMemBytes() -| metrics.osReserveBytes(metrics.getTotalMemBytes(), mlx.weightsInDeviceMemory())) +| ane_bytes,
        wiredCeilingFloorFor(config),
    ) -| ane_bytes;
}

// ── tests ──

test "an explicitly raised iogpu.wired_limit_mb is a FLOOR under the ceiling" {
    // Bar: a raised sysctl lifts the free-RAM term and nothing else; every other arch and the
    // default sysctl keep the old expression byte for byte.
    const t = std.testing;
    const mb: u64 = 1024 * 1024;
    const total_ram: u64 = 131_072 * mb;
    const working_set: u64 = 120_000 * mb;
    const saved = overrides().wired_limit_mb;
    defer overrides().wired_limit_mb = saved;
    overrides().wired_limit_mb = 120_000;

    try t.expectEqual(@as(u64, 111_808), wiredLimitFloor(120_000 * mb, total_ram, WIRED_LIMIT_MARGIN_BYTES) / mb);
    // The macOS default (75% of RAM) and anything under it declares nothing; and however
    // absurd the sysctl, never plan within the margin of physical RAM.
    try t.expectEqual(@as(u64, 0), wiredLimitFloor(98_304 * mb, total_ram, WIRED_LIMIT_MARGIN_BYTES));
    try t.expectEqual(@as(u64, 0), wiredLimitFloor(0, total_ram, WIRED_LIMIT_MARGIN_BYTES));
    try t.expectEqual(@as(u64, 0), wiredLimitFloor(120_000 * mb, 0, WIRED_LIMIT_MARGIN_BYTES));
    try t.expectEqual(total_ram - WIRED_LIMIT_MARGIN_BYTES, wiredLimitFloor(400 * (1 << 30), total_ram, WIRED_LIMIT_MARGIN_BYTES));

    const box96: u64 = 96_000 * mb;
    const six: u64 = 6 << 30;
    try t.expectEqual(@as(u64, 90_000) * mb - six, wiredLimitFloor(90_000 * mb, box96, six));
    try t.expectEqual(box96 - six, wiredLimitFloor(96_000 * mb, box96, six));
    try t.expectEqual(@as(u64, 90_000) * mb - WIRED_LIMIT_MARGIN_BYTES, wiredLimitFloor(90_000 * mb, box96, WIRED_LIMIT_MARGIN_BYTES));

    // Gated: every other arch takes floor 0, which is `physicalMemoryCeiling` exactly.
    var cfg: model_mod.ModelConfig = .{ .model_type = "qwen4_exp" };
    cfg.legacy.caps.long_ctx = true;
    try t.expect(wiredCeilingFloorForRam(&cfg, total_ram) > 0);
    try t.expectEqual(@as(u64, 0), wiredCeilingFloorForRam(null, total_ram));
    for ([_][]const u8{ "qwen3_5_moe", "gemma3", "llama", "deepseek_v4", "bailing_hybrid" }) |mt| {
        var other = cfg;
        other.model_type = mt;
        other.legacy.caps.long_ctx = std.mem.eql(u8, other.model_type, "qwen4_exp");
        try t.expect(!other.longCtxGated());
        try t.expectEqual(@as(u64, 0), wiredCeilingFloorForRam(&other, total_ram));
    }
    const footprint: u64 = 87_722 * mb;
    for ([_]u64{ 0, 12_934 * mb, 400 * (1 << 30) }) |free| {
        try t.expectEqual(
            physicalMemoryCeiling(working_set, footprint, free),
            gpuCeilingWithWiredFloor(working_set, footprint, free, 0),
        );
    }

}

test "parseWiredMarginGib accepts 2..32 and names a bad value" {
    const t = std.testing;
    try t.expectEqual(@as(u64, 8) << 30, try parseWiredMarginGib("8"));
    try t.expectEqual(@as(u64, 6) << 30, try parseWiredMarginGib("6"));
    try t.expectEqual(@as(u64, 2) << 30, try parseWiredMarginGib("2"));
    try t.expectEqual(@as(u64, 32) << 30, try parseWiredMarginGib("32"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("1"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("33"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib("nope"));
    try t.expectError(error.InvalidWiredMargin, parseWiredMarginGib(""));
}

test "physicalMemoryCeiling caps the static GPU max by real free RAM (#64 docker OOM)" {
    const GB: u64 = 1 << 30;
    const static_max: u64 = 115 * GB; // Metal max_recommended_working_set_size, 128 GB Mac

    // Idle: MLX holds 26 GB (active) + 5 GB (reclaimable cache), ~84 GB free.
    // 31 + 84 = 115 ⇒ the static device max still binds — NO regression when the
    // machine is uncontended (auto-context stays generous).
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, 31 * GB, 84 * GB));

    // #64: a docker-compose stack (firecrawl/rabbitmq/postgres/playwright) holds
    // ~45 GB. MLX holds 26 GB, only ~35 GB is physically free. The static max is
    // BLIND to docker, but the ceiling must collapse to 26 + 35 = 61 GB — otherwise
    // a large MoE prefill is admitted and Metal throws an uncatchable OOM.
    try testing.expectEqual(@as(u64, 61 * GB), physicalMemoryCeiling(static_max, 26 * GB, 35 * GB));

    // Saturating add: a bogus/huge footprint reading can never wrap to a tiny ceiling.
    try testing.expectEqual(static_max, physicalMemoryCeiling(static_max, std.math.maxInt(u64), GB));
}

test "the MLX buffer pool is inside `available` by construction, so a refusal cannot buy it back" {
    // `currentGpuMemoryCeiling` adds the pool to the footprint before the physical minimum, so on
    // the physical arm `available` is `cache_bytes + free_system` and a clear moves it by zero.
    const t = std.testing;
    const wired: u64 = 120 << 30;
    const active: u64 = 71_310 << 20;
    const pool: u64 = 8_268 << 20;
    const free_system: u64 = 12_546 << 20;
    const ceiling = physicalMemoryCeiling(wired, active + pool, free_system);
    try t.expect(ceiling < wired); // the physical arm, as on the box
    try t.expectEqual(pool + free_system, ceiling - active);
    const cleared = physicalMemoryCeiling(wired, active, free_system + pool);
    try t.expectEqual(ceiling, cleared);
}
