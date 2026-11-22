//! `MLX_SERVE_DSV41_PF_KEEP_WIDE` (default off): a prompt pass's large intermediates come from a reserve of Metal
//! buffers MLX keeps outside its cache (`patches/mlx-metal-wide-reserve.patch`), allocated and written once at load at
//! the pinned pass width, not fresh in every request's first layer.
//!
//! Without it the engine clears MLX's cache at each request's end, so the first wide layer of every request allocates
//! its large arrays anew (the stream copy, `wq_b`'s heads, `wo_a`, `wo_b`, the shared and routed experts' outputs):
//! the host stops encoding ~15 ms a GB while Metal makes each buffer (16.9 + 28.5 ms before layer 0's stream copy and
//! `wq_b` at 17,638 rows), and the GPU runs that layer's kernels slower while they first write the fresh pages (layer
//! 0 busy 189-193 ms against 153 for layer 3). MLX's cache could not serve them across requests anyway: it hands out a
//! buffer only within two pages of the size asked, so a buffer kept at one prompt's width never serves another's.
//!
//! Armed (the plugin's first forward once the engine pinned the pass width, `pinned`): one buffer per entry of
//! `sizes` at the pinned width, at least `min_bytes` each, the floor MLX's reserve serves from. An allocation of at least
//! the floor that MLX's cache cannot serve takes the smallest idle reserve buffer at least its size (so a narrower
//! prompt's arrays take the pinned width's buffers), and its free puts it back. MLX's `clear_cache` (the engine's, at
//! each request's end) and its cache limit never release the reserve; `disarm` (the model's unload) does.
//!
//! Exact: the same kernels over the same inputs; only which buffer an output lands in moves (MLX's cache hands out used
//! buffers already). A reserve buffer larger than an array is never donated to an op's output (MLX donates only a
//! buffer of the output's size), so such an op takes another buffer: the same words.
//!
//! Memory: the reserve is a subset of the pinned pass's own working set, which the engine bills on every request
//! (`prefillTransientReserve` at the pinned width) and left free at the load; armed only when it fits that bill. Between
//! requests it is held (the admission's live reading counts it as in use). The load's line names its bytes.
//!
//! Without the patched libmlx (no `mlx_metal_wide_reserve` symbol) the switch is declined by name.

const std = @import("std");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const crt = @import("../../../sdk/util/crt.zig");
const mlx = @import("../../../mlx/ffi.zig");

/// The smallest allocation the reserve serves (and its smallest buffer at the pinned width): a fresh allocation of it
/// costs the host ~1 ms (MLX's new Metal buffer, ~15 ms a GB).
pub const min_bytes: usize = 64 << 20;
/// The reserve's buffers at most.
pub const max_buffers = 32;

const ReserveFn = *const fn (min: usize, sizes: ?[*]const usize, n: usize) callconv(.c) c_int;
const StatsFn = *const fn (out: [*]usize, n: usize) callconv(.c) void;
const PlantFn = *const fn (alias: c_int) callconv(.c) void;

/// MLX's reserve counters (`mlx_metal_wide_reserve_stats`).
pub const Stats = struct {
    floor: usize = 0,
    buffers: usize = 0,
    bytes: usize = 0,
    idle: usize = 0,
    hits: usize = 0,
    misses: usize = 0,
    missed_bytes: usize = 0,
};

/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    looked: bool = false,
    reserve_fn: ?ReserveFn = null,
    stats_fn: ?StatsFn = null,
    plant_fn: ?PlantFn = null,
    /// The engine's pinned pass width and its billed working set (`pinned`); 0 = none yet.
    rows: usize = 0,
    bill: u64 = 0,
    /// Decided for this load (armed or declined); `disarm` clears it.
    decided: bool = false,
    armed: bool = false,
    /// The counters at the last pass's end (`passEnd`).
    last: Stats = .{},
};
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_keep_wide", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

fn lookUp() void {
    if (ps().looked) return;
    ps().looked = true;
    const self = std.c.dlopen(null, .{ .LAZY = true }) orelse return;
    if (std.c.dlsym(self, "mlx_metal_wide_reserve")) |f| ps().reserve_fn = @ptrCast(@alignCast(f));
    if (std.c.dlsym(self, "mlx_metal_wide_reserve_stats")) |f| ps().stats_fn = @ptrCast(@alignCast(f));
    if (std.c.dlsym(self, "mlx_metal_wide_reserve_plant")) |f| ps().plant_fn = @ptrCast(@alignCast(f));
}

/// Whether the loaded libmlx carries the reserve (`patches/mlx-metal-wide-reserve.patch`).
pub fn available() bool {
    lookUp();
    return ps().reserve_fn != null and ps().stats_fn != null;
}

/// The engine, once it pinned the load's widest prompt pass (`dsv41PassRows`): its rows and the working set it bills
/// at them (`prefillTransientReserve`).
pub fn pinned(rows: usize, bill: u64) void {
    ps().rows = rows;
    ps().bill = bill;
}

/// The reserve's buffers at `rows` rows (bytes each, into `out`; the count): a wide layer's arrays of at least
/// `min_bytes` that it allocates while the ones before it are still live, at most `max_buffers`.
pub fn sizes(cfg: anytype, rows: usize, out: *[max_buffers]usize) usize {
    const r: usize = rows;
    const h: usize = cfg.hidden_size;
    const heads: usize = cfg.num_attention_heads;
    const hd: usize = cfg.head_dim;
    const bf16: usize = 2;
    const per_row = [_]usize{
        // `wq_b`'s heads [rows, heads, head_dim].
        heads * hd * bf16,
        // The layer's stream [rows, 4, hidden].
        4 * h * bf16,
    };
    var n: usize = 0;
    for (per_row) |b| {
        if (n == max_buffers) break;
        const bytes = r * b;
        if (bytes < min_bytes) continue;
        out[n] = bytes;
        n += 1;
    }
    return n;
}

/// The plugin's forward: arms the reserve once a load, at the first forward after the engine pinned the pass width
/// (the load's own forwards, before any request). One line names the outcome.
pub fn armOnce(cfg: anytype) void {
    if (ps().decided or ps().rows == 0) return;
    if (!env.on(.MLX_SERVE_DSV41_PF_KEEP_WIDE)) return;
    ps().decided = true;
    if (!available()) return log.info("deepseek_v41: MLX_SERVE_DSV41_PF_KEEP_WIDE declined: the loaded libmlx has no mlx_metal_wide_reserve (patches/mlx-metal-wide-reserve.patch)\n", .{});
    var buf: [max_buffers]usize = undefined;
    const n = sizes(cfg, ps().rows, &buf);
    var total: u64 = 0;
    for (buf[0..n]) |b| total += b;
    if (n == 0) return log.info("deepseek_v41: MLX_SERVE_DSV41_PF_KEEP_WIDE declined: no array of {d} rows reaches {d} MB\n", .{ ps().rows, min_bytes >> 20 });
    if (total > ps().bill) return log.info("deepseek_v41: MLX_SERVE_DSV41_PF_KEEP_WIDE declined: its {d} MB pass {d} MB of the {d}-row pass's working set\n", .{ total >> 20, ps().bill >> 20, ps().rows });
    const t0 = crt.monotonicMs();
    const rc = arm(buf[0..n]);
    const st = ps().last;
    log.info("deepseek_v41: MLX_SERVE_DSV41_PF_KEEP_WIDE: a reserve of {d} buffers, {d} MB ({d:.2} GB) at {d} rows, inside the pass's {d} MB working set, made in {d} ms{s}\n", .{ st.buffers, st.bytes >> 20, @as(f64, @floatFromInt(st.bytes)) / 1e9, ps().rows, ps().bill >> 20, crt.monotonicMs() - t0, if (rc != 0) " (a buffer failed to allocate)" else "" });
}

/// MLX's reserve made anew: one buffer of each of `bytes` (`available` first).
pub fn arm(bytes: []const usize) c_int {
    const rc = ps().reserve_fn.?(min_bytes, bytes.ptr, bytes.len);
    ps().last = stats();
    ps().armed = ps().last.buffers > 0;
    return rc;
}

/// The tests' planted control: MLX's reserve hands out a buffer it keeps idle, so two live arrays may share it.
pub fn plant(alias: bool) void {
    if (ps().plant_fn) |f| f(@intFromBool(alias));
}

/// MLX's reserve counters (all 0 without the patch).
pub fn stats() Stats {
    lookUp();
    const f = ps().stats_fn orelse return .{};
    var v: [7]usize = @splat(0);
    f(&v, v.len);
    return .{ .floor = v[0], .buffers = v[1], .bytes = v[2], .idle = v[3], .hits = v[4], .misses = v[5], .missed_bytes = v[6] };
}

/// A wide prompt pass's end: one line with the pass's hits and misses (armed only).
pub fn passEnd() void {
    if (!ps().armed) return;
    const st = stats();
    const was = ps().last;
    ps().last = st;
    log.info("deepseek_v41: MLX_SERVE_DSV41_PF_KEEP_WIDE: the pass took {d} reserve buffers, {d} allocations missed ({d} MB); reserve {d} MB, {d} MB idle\n", .{ st.hits -| was.hits, st.misses -| was.misses, (st.missed_bytes -| was.missed_bytes) >> 20, st.bytes >> 20, st.idle >> 20 });
}

/// The model's unload: the reserve's buffers released (MLX disarmed), the next load decides again.
pub fn disarm() void {
    if (ps().armed) if (ps().reserve_fn) |f| {
        _ = f(0, null, 0);
    };
    ps().armed = false;
    ps().decided = false;
    ps().rows = 0;
    ps().bill = 0;
    ps().last = .{};
}

const testing = std.testing;

fn evalZeros(halves: usize) !mlx.mlx_array {
    var z = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(z);
    try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{@intCast(halves)}, 1, .float16, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(z));
    return z;
}

test "dsv41 keep wide: MLX's reserve serves a narrower array from its wider buffer, keeps it through clear_cache, misses a wider one, and releases it on disarm" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    if (!available()) {
        std.debug.print("keep wide: the loaded libmlx has no mlx_metal_wide_reserve: skipped\n", .{});
        return error.SkipZigTest;
    }
    disarm();
    defer disarm();
    const s = mlx.gpuStream();
    var active0: usize = 0;
    _ = mlx.mlx_get_active_memory(&active0);
    const wide: usize = 128 << 20;
    try testing.expectEqual(@as(c_int, 0), arm(&.{wide}));
    var st = stats();
    try testing.expectEqual(@as(usize, 1), st.buffers);
    try testing.expectEqual(wide, st.bytes);
    try testing.expectEqual(wide, st.idle);
    // A 96 MB array (past the floor, under the buffer) takes the reserve's buffer.
    const narrow = try evalZeros(48 << 20);
    st = stats();
    try testing.expectEqual(@as(usize, 1), st.hits);
    try testing.expectEqual(@as(usize, 0), st.idle);
    _ = mlx.mlx_array_free(narrow);
    _ = mlx.mlx_synchronize(s);
    try testing.expectEqual(wide, stats().idle);
    // The engine's per-request clear, and a cache limit of 0, leave the reserve whole.
    try mlx.check(mlx.mlx_clear_cache());
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, 0);
    const small = try evalZeros(1 << 20);
    _ = mlx.mlx_array_free(small);
    _ = mlx.mlx_synchronize(s);
    var tmp: usize = 0;
    _ = mlx.mlx_set_cache_limit(&tmp, prev);
    st = stats();
    try testing.expectEqual(@as(usize, 1), st.buffers);
    try testing.expectEqual(wide, st.idle);
    // A 192 MB array is wider than every reserve buffer: a fresh allocation.
    const big = try evalZeros(96 << 20);
    st = stats();
    try testing.expectEqual(@as(usize, 1), st.misses);
    try testing.expectEqual(wide, st.idle);
    _ = mlx.mlx_array_free(big);
    _ = mlx.mlx_synchronize(s);
    try mlx.check(mlx.mlx_clear_cache());
    // The unload's disarm releases it; MLX's active memory is where it started.
    disarm();
    st = stats();
    try testing.expectEqual(@as(usize, 0), st.buffers);
    try testing.expectEqual(@as(usize, 0), st.bytes);
    var active1: usize = 0;
    _ = mlx.mlx_get_active_memory(&active1);
    try testing.expectEqual(active0, active1);
}
