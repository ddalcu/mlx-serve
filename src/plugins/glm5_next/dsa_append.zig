//! A GLM-5.3-Flash DSA layer's decode, verify and draft window rows' cache append, the stock bytes:
//! - MLX_SERVE_GLM_DSA_TAILS: the pooled-keys launch reads the up to kpool - 1 indexer rows before the forward, which
//!   `dsaAttn` slices from the layer's cache view just before it appends. MLX's eval holds every input's buffer until
//!   its command buffer completes, so the append's slice_update finds the cache buffer held twice and copies it whole
//!   (its capacity times 512 bytes, every DSA layer, every window). Here `begin` (the trunk forward's start and the MTP
//!   head step's) copies every DSA layer's tail rows into a buffer of their own in one launch and commits it at once,
//!   and each layer reads its rows there (`take`): the same rows, and the cache buffer is free to be donated by the
//!   time the layer appends. `end` drops them.
//! It declines (null: the stock path runs) outside GLM-5.3-Flash's dims or past the window's rows.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");
const trunk = @import("trunk.zig");

const Array = mlx.mlx_array;

/// GLM-5.3-Flash's indexer: key and gate width, a cache row's (key | gate), pool size.
pub const head_dim: c_int = 128;
pub const row_w: c_int = 2 * head_dim;
pub const kpool: c_int = 4;
/// The rows a decode, verify or draft window has at most (`glm5_next.HC_PRE_MAX_ROWS`): a prompt keeps the stock path.
pub const max_rows: c_int = 16;
/// The trunk's DSA layers (layers 3, 7, ..., 43) and the MTP head's.
pub const trunk_layers: usize = 11;
pub const head_layers: usize = 1;

pub const TAILS_SOURCE = kernelSource("plugins/glm5_next/kernels/dsa_tails.metal");

/// MLX_SERVE_GLM_DSA_TAILS, read once (tests set it).
pub var tails_switch: ?bool = null;
/// DSA layers given their tail rows by `begin` since the process started (tests read them).
pub var tails_served: u64 = 0;
var tails_said = false;

pub fn tailsOn() bool {
    if (tails_switch == null) tails_switch = if (env.get(.MLX_SERVE_GLM_DSA_TAILS)) |v| v[0] != '0' else true;
    return tails_switch.?;
}

fn kernelOf(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

fn apply(s: mlx.mlx_stream, kern: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, inputs: []const Array) !Array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// ── MLX_SERVE_GLM_DSA_TAILS ──

/// The tail rows `begin` reads before a forward of `rows` rows at `offset` over `n_layers` DSA layers (B `batch`,
/// `slots` slots): the stock `n_prev`, 0 where the switch does not serve. MLX-free.
pub fn tailRows(batch: c_int, rows: c_int, slots: usize, offset: u32, n_layers: usize) c_int {
    if (batch != 1 or rows < 1 or rows > max_rows or slots != 1) return 0;
    if (n_layers != trunk_layers and n_layers != head_layers) return 0;
    return @intCast(@min(offset, @as(u32, @intCast(kpool - 1))));
}

/// The trunk's DSA layers in order, in `buf` (none past its length).
pub fn trunkDsaLayers(m: *const trunk.Model, buf: *[trunk_layers]u32) []const u32 {
    var n: usize = 0;
    for (m.layers[0..@min(m.layers.len, @as(usize, m.c.num_hidden_layers))], 0..) |*lw, li| if (lw.mixer == .dsa) {
        if (n == buf.len) return buf[0..0];
        buf[n] = @intCast(li);
        n += 1;
    };
    return buf[0..n];
}

fn Gather(comptime n: usize) type {
    return struct {
        const name = std.fmt.comptimePrint("glm5_dsa_tails{d}", .{n});
        const names = blk: {
            var a: [n][*:0]const u8 = undefined;
            for (0..n) |k| a[k] = std.fmt.comptimePrint("t{d}", .{k});
            break :blk a;
        };
        const source = std.fmt.comptimePrint("#define GLM5_TAILS_N {d}\n", .{n}) ++ TAILS_SOURCE;
        var kernel: ?mlx.mlx_fast_metal_kernel = null;

        /// The tails [1, 1, R, 256] each, in one launch: [n, R, 256], a new array.
        fn run(s: mlx.mlx_stream, tails: []const Array, rows: c_int, dt: mlx.mlx_dtype) !Array {
            const k = try kernelOf(&kernel, name, &names, source);
            const cfg = mlx.mlx_fast_metal_kernel_config_new();
            defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ @intCast(n), rows, row_w }, 3, dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, row_w, rows, @intCast(n)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, row_w, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROW_W", row_w));
            return apply(s, k, cfg, tails);
        }
    };
}
const trunk_gather = Gather(trunk_layers);
const head_gather = Gather(head_layers);

/// One forward's rows: `rows[i]` [1, n_prev, 256] is layer `layers[i]`'s tail, a view of `gathered`.
const Table = struct {
    active: bool = false,
    cache: ?*anyopaque = null,
    offset: u32 = 0,
    n_prev: c_int = 0,
    n: usize = 0,
    layers: [trunk_layers]u32 = undefined,
    rows: [trunk_layers]Array = undefined,
    gathered: Array = .{ .ctx = null },
};
var table: Table = .{};

/// Reads the tail rows of `layers` (DSA layers of slot 0's cache in `sl`, a `Slots`) before a forward of the
/// `ids_shape` [B, S] ids (`batched`: a batched tick, declined) and commits them; `take` then hands them to each layer.
/// A shape it does not serve leaves no rows (each layer reads its cache view as before).
pub fn begin(s: mlx.mlx_stream, sl: anytype, batched: bool, ids_shape: []const c_int, layers: []const u32) !void {
    end();
    if (batched or ids_shape.len != 2 or !mlx.streamIsGpu(s)) return;
    if (sl.caches.len != 1 or sl.positions.len != 1) return;
    const offset = sl.positions[0];
    const n_prev = tailRows(ids_shape[0], ids_shape[1], sl.caches.len, offset, layers.len);
    if (n_prev == 0 or !tailsOn()) return;
    // each layer's tail, as `dsaAttn` slices it: the view's last n_prev rows
    var tails: [trunk_layers]Array = undefined;
    var made: usize = 0;
    defer for (tails[0..made]) |t| {
        _ = mlx.mlx_array_free(t);
    };
    var dt: ?mlx.mlx_dtype = null;
    for (layers) |layer| {
        var v = try sl.view(0, layer, s);
        defer v.deinit();
        if (v.v.ctx == null) return;
        const sh = mlx.getShape(v.v);
        if (sh.len != 4 or sh[0] != 1 or sh[1] != 1 or sh[2] < n_prev or sh[3] != row_w) return;
        const d = mlx.mlx_array_dtype(v.v);
        if (d != .bfloat16 and d != .float16) return;
        if (dt) |x| if (x != d) return;
        dt = d;
        var t = mlx.mlx_array_new();
        mlx.check(mlx.mlx_slice(&t, v.v, &[_]c_int{ 0, 0, sh[2] - n_prev, 0 }, 4, &[_]c_int{ 1, 1, sh[2], row_w }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s)) catch |e| {
            _ = mlx.mlx_array_free(t);
            return e;
        };
        tails[made] = t;
        made += 1;
    }
    // the copy, in a buffer of its own (not MLX's copy, which shares its input's), committed now: its command buffer
    // runs while the forward is built
    const g = if (layers.len == head_layers) try head_gather.run(s, tails[0..made], n_prev, dt.?) else try trunk_gather.run(s, tails[0..made], n_prev, dt.?);
    var tb: Table = .{ .active = true, .cache = sl.caches[0], .offset = offset, .n_prev = n_prev, .gathered = g };
    errdefer {
        for (tb.rows[0..tb.n]) |r| _ = mlx.mlx_array_free(r);
        _ = mlx.mlx_array_free(g);
    }
    {
        const vec = mlx.mlx_vector_array_new_data(&[_]Array{g}, 1);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
    }
    for (layers, 0..) |layer, i| {
        var one = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one);
        const li: c_int = @intCast(i);
        try mlx.check(mlx.mlx_slice(&one, g, &[_]c_int{ li, 0, 0 }, 3, &[_]c_int{ li + 1, n_prev, row_w }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var r = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_reshape(&r, one, &[_]c_int{ 1, n_prev, row_w }, 3, s));
        tb.layers[i] = layer;
        tb.rows[i] = r;
        tb.n += 1;
    }
    table = tb;
    if (!tails_said) {
        tails_said = true;
        log.info("[glm5_next] DSA tail rows engaged: a {d}-row forward's {d} DSA layer(s), {d} tail rows each, read first in one launch\n", .{ ids_shape[1], layers.len, n_prev });
    }
}

/// Layer `layer`'s tail rows [1, n_prev, 256] read by this forward's `begin` (borrowed: valid until `end`), or null
/// (the layer reads its cache view as before): another slot, cache, position or row count than `begin` read for.
pub fn take(sl: anytype, slot: usize, layer: u32, n_prev: c_int) ?Array {
    if (!table.active or slot != 0 or n_prev != table.n_prev) return null;
    if (sl.caches.len <= slot or sl.positions.len <= slot) return null;
    if (sl.caches[slot] != table.cache or sl.positions[slot] != table.offset) return null;
    for (table.layers[0..table.n], table.rows[0..table.n]) |l, r| if (l == layer) {
        tails_served += 1;
        return r;
    };
    return null;
}

/// Drops what `begin` read (the forward's end; also the start of the next `begin`).
pub fn end() void {
    if (!table.active) return;
    for (table.rows[0..table.n]) |r| _ = mlx.mlx_array_free(r);
    _ = mlx.mlx_array_free(table.gathered);
    table = .{};
}

// ── tests ──

const testing = std.testing;
const glm5 = @import("glm5_next.zig");
const dsa_prompt = @import("dsa_prompt.zig");
const test_ix = dsa_prompt.test_ix;

test "glm5_next dsa window rows: TAILS reads the stock tail's 3 rows (fewer at the cache's start) for a one-slot window over the trunk's 11 DSA layers or the head's one, and nothing for a prompt, a batch or other layer counts" {
    for (1..17) |r| try testing.expectEqual(@as(c_int, 3), tailRows(1, @intCast(r), 1, 16390, trunk_layers));
    try testing.expectEqual(@as(c_int, 3), tailRows(1, 1, 1, 16390, head_layers));
    try testing.expectEqual(@as(c_int, 2), tailRows(1, 3, 1, 2, trunk_layers));
    try testing.expectEqual(@as(c_int, 1), tailRows(1, 1, 1, 1, head_layers));
    try testing.expectEqual(@as(c_int, 0), tailRows(1, 3, 1, 0, trunk_layers));
    try testing.expectEqual(@as(c_int, 0), tailRows(1, 17, 1, 16390, trunk_layers));
    try testing.expectEqual(@as(c_int, 0), tailRows(2, 3, 1, 16390, trunk_layers));
    try testing.expectEqual(@as(c_int, 0), tailRows(1, 3, 2, 16390, trunk_layers));
    try testing.expectEqual(@as(c_int, 0), tailRows(1, 3, 1, 16390, 10));
    try testing.expectEqual(@as(c_int, 0), tailRows(1, 3, 1, 16390, 0));
}

fn sliceRows(s: mlx.mlx_stream, a: Array, axis: usize, lo: c_int, hi: c_int) !Array {
    const sh = mlx.getShape(a);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = undefined;
    for (sh, 0..) |v, i| stop[i] = v;
    start[axis] = lo;
    stop[axis] = hi;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, a, &start, sh.len, &stop, sh.len, &[_]c_int{ 1, 1, 1, 1 }, sh.len, s));
    return out;
}

/// The stock tail `dsaAttn` slices from a cache view [1, 1, T, 256]: its last n rows as [1, n, 256].
fn stockTail(s: mlx.mlx_stream, view: Array, n: c_int) !Array {
    const t = mlx.getShape(view)[2];
    const tail = try sliceRows(s, view, 2, t - n, t);
    defer _ = mlx.mlx_array_free(tail);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, tail, &[_]c_int{ 1, n, row_w }, 3, s));
    return out;
}

/// `Slots` as `begin` and `take` read it: one slot's cache handle and position, a view per layer.
var mock_cache: u8 = 0;
const MockView = struct {
    v: Array,
    fn deinit(self: *MockView) void {
        self.* = .{ .v = .{ .ctx = null } };
    }
};
const MockSlots = struct {
    caches: []const ?*anyopaque,
    positions: []const u32,
    layers: []const u32,
    views: []const Array,
    fn view(self: MockSlots, i: usize, layer: u32, s: mlx.mlx_stream) !MockView {
        _ = s;
        if (i != 0) return error.TestUnexpectedResult;
        for (self.layers, self.views) |l, v| if (l == layer) return .{ .v = v };
        return error.TestUnexpectedResult;
    }
};

/// A cache buffer [1, 1, t + 61, 256] and its view of the first `t` rows, both evaluated.
const MockCache = struct {
    buf: Array,
    view: Array,
    fn init(s: mlx.mlx_stream, seed: *u64, t: c_int) !MockCache {
        const buf = try dsa_prompt.testNormal(s, seed, &.{ 1, 1, t + 61, row_w }, .bfloat16, 1.0);
        errdefer _ = mlx.mlx_array_free(buf);
        const view = try sliceRows(s, buf, 2, 0, t);
        errdefer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_array_eval(view));
        return .{ .buf = buf, .view = view };
    }
    fn deinit(c: *MockCache) void {
        _ = mlx.mlx_array_free(c.view);
        _ = mlx.mlx_array_free(c.buf);
    }
};

test "glm5_next dsa window rows: TAILS hands each DSA layer the stock tail's rows bit for bit, and the pooled-keys kernel writes the same rows from them" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const saved = tails_switch;
    tails_switch = true;
    defer {
        tails_switch = saved;
        end();
    }
    var seed: u64 = 20261109;
    const trunk_ids = [_]u32{ 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43 };
    const Case = struct { name: []const u8, layers: []const u32, t: c_int, rows: c_int };
    // deep in a 16K cache (3 rows) and at its first rows (1 and 2), the trunk's layers and the head's
    const cases = [_]Case{
        .{ .name = "trunk16k", .layers = &trunk_ids, .t = 16390, .rows = 3 },
        .{ .name = "trunk_at1", .layers = &trunk_ids, .t = 1, .rows = 3 },
        .{ .name = "trunk_at2", .layers = &trunk_ids, .t = 2, .rows = 1 },
        .{ .name = "head16k", .layers = &.{0}, .t = 16390, .rows = 1 },
        .{ .name = "head_at1", .layers = &.{0}, .t = 1, .rows = 2 },
    };
    const handles = [_]?*anyopaque{@ptrCast(&mock_cache)};
    for (cases) |c| {
        var caches: [trunk_layers]MockCache = undefined;
        var views: [trunk_layers]Array = undefined;
        var made: usize = 0;
        defer for (caches[0..made]) |*x| x.deinit();
        for (c.layers, 0..) |_, i| {
            caches[i] = try MockCache.init(s, &seed, c.t);
            views[i] = caches[i].view;
            made += 1;
        }
        const pos = [_]u32{@intCast(c.t)};
        const sl: MockSlots = .{ .caches = &handles, .positions = &pos, .layers = c.layers, .views = views[0..made] };
        const n_prev: c_int = @min(c.t, 3);
        const before = tails_served;
        try begin(s, sl, false, &.{ 1, c.rows }, c.layers);
        defer end();
        for (c.layers, 0..) |layer, i| {
            const want = try stockTail(s, views[i], n_prev);
            defer _ = mlx.mlx_array_free(want);
            const got = take(sl, 0, layer, n_prev) orelse return error.TestUnexpectedResult;
            try dsa_prompt.expectCase("dsa tail rows", c.name, got, want);
        }
        try testing.expectEqual(@as(u64, c.layers.len), tails_served - before);
        // another position or row count: the layer reads its view
        try testing.expect(take(sl, 0, c.layers[0], n_prev + 1) == null);
    }
    // the pooled-keys kernel's rows from the tail against its rows from the stock tail, a window of 1 to 4 rows whose
    // first pool began 1, 2 or 3 rows before it
    for ([_]c_int{ 16389, 16390, 16391 }) |t| for ([_]c_int{ 1, 2, 3, 4 }) |rows| {
        var c = try MockCache.init(s, &seed, t);
        defer c.deinit();
        const chunk = try dsa_prompt.testNormal(s, &seed, &.{ 1, rows, row_w }, .bfloat16, 1.0);
        defer _ = mlx.mlx_array_free(chunk);
        const ape = try dsa_prompt.testNormal(s, &seed, &.{ kpool, head_dim }, .float32, 1.0);
        defer _ = mlx.mlx_array_free(ape);
        const pos = [_]u32{@intCast(t)};
        const layers = [_]u32{0};
        const views = [_]Array{c.view};
        const sl: MockSlots = .{ .caches = &handles, .positions = &pos, .layers = &layers, .views = &views };
        const want_prev = try stockTail(s, c.view, 3);
        defer _ = mlx.mlx_array_free(want_prev);
        const want = try glm5.withPooledKeys(s, want_prev, chunk, ape, t, test_ix);
        defer _ = mlx.mlx_array_free(want);
        try begin(s, sl, false, &.{ 1, rows }, &layers);
        defer end();
        const got_prev = take(sl, 0, 0, 3) orelse return error.TestUnexpectedResult;
        const got = try glm5.withPooledKeys(s, got_prev, chunk, ape, t, test_ix);
        defer _ = mlx.mlx_array_free(got);
        try dsa_prompt.expectCase("dsa tail rows (pooled)", "window", got, want);
    };
    // a prompt's rows and a batched tick: no rows read
    var c = try MockCache.init(s, &seed, 16390);
    defer c.deinit();
    const pos = [_]u32{16390};
    const views = [_]Array{c.view};
    const sl: MockSlots = .{ .caches = &handles, .positions = &pos, .layers = &.{0}, .views = &views };
    try begin(s, sl, false, &.{ 1, 17 }, &.{0});
    try testing.expect(take(sl, 0, 0, 3) == null);
    try begin(s, sl, true, &.{ 1, 1 }, &.{0});
    try testing.expect(take(sl, 0, 0, 3) == null);
}
