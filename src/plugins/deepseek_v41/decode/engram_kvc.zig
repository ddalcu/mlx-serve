//! `MLX_SERVE_DSV41_ENGRAM_KV_CACHE`: a decode row's Engram wkv is the sum over its slots (each Engram layer's 24
//! table rows of 256 inputs) of W_slot . row, and a slot's partial depends only on the layer's weights, the slot and
//! the table row id. The slot-blocked kernel (`engram_kv.launchSlots`) sums the partials in slot order, so a partial
//! kept from an earlier step gives the same bits as computing it again, and its weights (6.8 MB a slot on the
//! release) are never read. The cache keeps them on the GPU: per table (layer x slot) `sets` x `ways` entries of an
//! f32 [N] partial, keyed by row id, oldest out first; at most `cap_bytes`, billed at load.
//!
//! The look-up runs on the GPU (`dec_engram_kvc_plan.metal`) once a step, on the step's GPU row ids: its new keys and
//! ages are fresh arrays (each step's look-up reads the last one's), the partials are written in place by the slot
//! kernel. Each look-up binds the last cached step's wkv (`after`), so evaluating the keys an entry was taken in also
//! runs the launch that writes it (a step built and dropped unevaluated included), and every later read is behind it.
//!
//! The partials are identical across requests by construction (weights and table rows never change after load), so
//! sharing them would stay exact; the cache is still scoped to one decoding state and starts empty when another
//! state (a new request's, after `reset`) steps (`Cache.plan`'s `gen`), so a request never reads another's rows.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const dsv41 = @import("../deepseek_v41.zig");
const engram_kv = @import("engram_kv.zig");
const engram_ids = @import("engram_ids.zig");
const engram = @import("../engram.zig");

const A = mlx.mlx_array;
const Scope = dsv41.Scope;

pub const PLAN_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_engram_kvc_plan.metal");
pub const PLAN_INS = [_][*:0]const u8{ "ids", "keys", "ages", "clk", "after" };
pub const PLAN_OUTS = [_][*:0]const u8{ "plan", "nkeys", "nages" };
pub const ways = 8;
pub const plan_threads = 256;
/// The cache's bound: partials, keys and ages.
pub const cap_bytes: u64 = 4_000_000_000;

var plan_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub fn on() bool {
    return lev.onUnlessZero(.ENGRAM_KV_CACHE);
}

/// What decides the cache, known alike before the load (the bill, `residentDiskBytes`) and at it (`engram_ids.settle`).
pub const Facts = struct {
    layers: usize,
    /// Every Engram table held in memory: as the bill bills them, as the load holds them.
    held: bool,
    /// The routed experts stream (`Config.dsv41_streamed`).
    streamed: bool,
    /// Every table `tableReads`.
    reads: bool,
};

/// Whether a load allocates the cache: this switch and `MLX_SERVE_DSV41_ENGRAM_GPU_IDS` on, every Engram table held
/// in the form the slot kernel reads, resident experts. The load may still decline on what only it sees (a pair, a
/// Metal buffer refused, the wkv's layout); the bill then counts a cap that is not used.
pub fn allocates(f: Facts) bool {
    return on() and engram_ids.on() and f.layers > 0 and f.held and f.reads and !f.streamed;
}

/// The load's bill for the cache: its bound where the load allocates it.
pub fn billBytes(f: Facts) u64 {
    return if (allocates(f)) cap_bytes else 0;
}

/// Whether the GPU ids path and the slot kernel read table `t`: separate mxfp4 or mxfp8 code and scale regions,
/// `slot_hd` wide, the codes aligned to the kernels' loads (`fuse.wrapHeld`, `fuse.kvServes`, `engram_kv.launchSlots`).
pub fn tableReads(t: *const engram.Table) bool {
    if (!((t.mode == .mxfp4 and t.bits == 4) or (t.mode == .mxfp8 and t.bits == 8)) or t.group_size != 32) return false;
    if (t.records() or t.head_dim != engram_kv.slot_hd or t.wlen != t.head_dim * t.bits / 8 or t.slen != t.head_dim / 32) return false;
    const al: u64 = if (t.bits == 4) 8 else 16;
    return (t.w_off % std.heap.pageSize()) % al == 0 and t.w_stride % al == 0;
}

/// Sets a table for `nt` tables of `n`-wide partials within `cap_bytes`, at most `max_sets` (each step's look-up
/// copies every key and age: a narrow pack's cache stays small).
pub fn setsFor(nt: usize, n: usize) usize {
    return @min(max_sets, @as(usize, @intCast(cap_bytes / (nt * ways * (4 * n + 8)))));
}

pub const max_sets = 1024;

pub const Cache = struct {
    nt: usize,
    sets: usize,
    n: usize,
    /// u32 [nt * sets * ways] each: a way's row id + 1 (0: empty), the step it was last used.
    keys: A,
    ages: A,
    /// f32 [nt * sets * ways, n]: each way's partial.
    vals: A,
    clock: u32 = 0,
    owner: u64 = 0,
    owners: u64 = 0,
    /// The last step's plan (`plan`), for the tests' hit counts.
    last: A = .{ .ctx = null },
    /// The last cached launch's wkv (`after`): the next look-up's order behind it.
    tick: A = .{ .ctx = null },

    /// A cache for `nt` tables of `n`-wide partials, allocated on `s` (null when not one set fits the bound).
    pub fn init(s: mlx.mlx_stream, nt: usize, n: usize) !?Cache {
        const sets = setsFor(nt, n);
        if (sets == 0 or nt * sets * ways >= 1 << 31) return null;
        var c: Cache = .{ .nt = nt, .sets = sets, .n = n, .keys = mlx.mlx_array_new(), .ages = mlx.mlx_array_new(), .vals = mlx.mlx_array_new() };
        errdefer c.deinit();
        try mlx.check(mlx.mlx_zeros(&c.vals, &[_]c_int{ @intCast(nt * sets * ways), @intCast(n) }, 2, .float32, s));
        try mlx.check(mlx.mlx_array_eval(c.vals));
        try c.clear(s);
        return c;
    }

    pub fn deinit(c: *Cache) void {
        for ([_]A{ c.keys, c.ages, c.vals, c.last, c.tick }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
    }

    pub fn bytes(c: *const Cache) u64 {
        return @as(u64, c.nt * c.sets * ways) * (4 * c.n + 8);
    }

    fn clear(c: *Cache, s: mlx.mlx_stream) !void {
        const sh = [_]c_int{@intCast(c.nt * c.sets * ways)};
        try mlx.check(mlx.mlx_zeros(&c.keys, &sh, 1, .uint32, s));
        try mlx.check(mlx.mlx_zeros(&c.ages, &sh, 1, .uint32, s));
        c.clock = 0;
        if (c.tick.ctx != null) _ = mlx.mlx_array_free(c.tick);
        c.tick = .{ .ctx = null };
    }

    /// A new decoding state's id (`State.kvc_owner`).
    pub fn newOwner(c: *Cache) u64 {
        c.owners += 1;
        return c.owners;
    }

    /// A cached launch's wkv `kv`: the next look-up waits for it.
    pub fn after(c: *Cache, kv: A) void {
        if (c.tick.ctx == null) c.tick = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&c.tick, kv);
    }

    /// The decode row's plan (u32 [nt], `engram_kv.Kvc.plan` sliced per layer) for the row ids `ids` (u32 [nt]) of a
    /// step on state `gen`: emptied first when the state is not the one it holds.
    pub fn plan(c: *Cache, sc: *Scope, gen: u64, ids: A) !A {
        if (gen != c.owner) {
            try c.clear(sc.s);
            c.owner = gen;
        }
        c.clock += 1;
        const clk = try sc.keep(mlx.mlx_array_new_data(&c.clock, &[_]c_int{1}, 1, .uint32));
        if (plan_kernel == null) plan_kernel = try dsv41.buildKernel("dsv41_dec_engram_kvc_plan", &PLAN_INS, &PLAN_OUTS, PLAN_SOURCE, "");
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const e: c_int = @intCast(c.nt * c.sets * ways);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{@intCast(c.nt)}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{e}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{e}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(c.nt * c.sets), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, plan_threads, 1, 1));
        inline for (.{ .{ "NT", c.nt }, .{ "S", c.sets }, .{ "W", ways } }) |a| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, a[0], @intCast(a[1])));
        const ins = [_]A{ ids, c.keys, c.ages, clk, if (c.tick.ctx != null) c.tick else clk };
        const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, plan_kernel.?, iv, cfg, sc.s));
        var p = mlx.mlx_array_new();
        const out = try sc.res(mlx.mlx_vector_array_get(&p, ov, 0), &p);
        try mlx.check(mlx.mlx_vector_array_get(&c.keys, ov, 1));
        try mlx.check(mlx.mlx_vector_array_get(&c.ages, ov, 2));
        if (c.last.ctx == null) c.last = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&c.last, out);
        return out;
    }

    /// Layer `ek`'s part of a step's `plan` (each layer `nc` slots).
    pub fn of(c: *const Cache, sc: *Scope, p: A, ek: usize, nc: usize) !engram_kv.Kvc {
        return .{ .plan = try sc.slice(p, 0, ek * nc, (ek + 1) * nc), .vals = c.vals };
    }
};

// ── the kernel contract ──

const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const cv = @import("../contract.zig");

const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;
const spec = kit.spec;

/// The look-up: `rows` tables of `n` sets of `k` ways (`k` = `ways`) holding some of the row ids and missing others.
const Plan = struct {
    pub const Params = struct {};

    pub fn geometry(_: Params, s: Shape) ?metal.Geometry {
        if (s.rows == 0 or s.n == 0 or s.k != ways) return null;
        var g = metal.Geometry{ .grid = .{ ci(s.rows * s.n), 1, 1 }, .tg = .{ plan_threads, 1, 1 } };
        g.out(&.{ci(s.rows)}, .uint32);
        g.out(&.{ci(s.rows * s.n * ways)}, .uint32);
        g.out(&.{ci(s.rows * s.n * ways)}, .uint32);
        g.int("NT", s.rows);
        g.int("S", s.n);
        g.int("W", ways);
        return g;
    }

    /// Ids over a few multiples of the sets; every way of a set holds a distinct id of that set or nothing, and half
    /// the tables hold their row's id.
    pub fn fill(k: anytype, _: Fill, rng: std.Random) !void {
        const nt = k.shape.rows;
        const S = k.shape.n;
        const ids = try k.add(.uint32, &.{nt});
        const keys = try k.add(.uint32, &.{nt * S * ways});
        const ages = try k.add(.uint32, &.{nt * S * ways});
        const clk = try k.add(.uint32, &.{1});
        clk[0] = 5000;
        _ = try k.add(.uint32, &.{1});
        for (0..nt) |t| {
            const id = rng.uintLessThan(u32, @intCast(8 * S));
            ids[t] = @floatFromInt(id);
            for (0..S) |s| {
                const at = (t * S + s) * ways;
                for (0..ways) |w| {
                    // Way w of set s may hold row id s + (w + 9) * S: distinct, never the row's own.
                    const full = rng.boolean();
                    keys[at + w] = if (full) @floatFromInt(s + (w + 9) * S + 1) else 0;
                    ages[at + w] = if (full) @floatFromInt(1 + rng.uintLessThan(u32, 4000)) else 0;
                }
            }
            if (t % 2 == 0) {
                const at = (t * S + id % S) * ways + rng.uintLessThan(usize, ways);
                keys[at] = @floatFromInt(id + 1);
                ages[at] = @floatFromInt(1 + rng.uintLessThan(u32, 4000));
            }
        }
    }

    pub fn count(_: Params, s: Shape) usize {
        return s.rows + 2 * s.rows * s.n * ways;
    }

    fn expected(k: anytype, want: []u32) void {
        const nt = k.shape.rows;
        const S = k.shape.n;
        const e = nt * S * ways;
        const now: u32 = @intFromFloat(k.in(3)[0]);
        for (0..e) |i| {
            want[nt + i] = @intFromFloat(k.in(1)[i]);
            want[nt + e + i] = @intFromFloat(k.in(2)[i]);
        }
        for (0..nt) |t| {
            const id: u32 = @intFromFloat(k.in(0)[t]);
            const base = (t * S + id % S) * ways;
            const keys = want[nt + base ..][0..ways];
            const ages = want[nt + e + base ..][0..ways];
            var hit: u32 = 0x80000000;
            const way = std.mem.indexOfScalar(u32, keys, id + 1) orelse blk: {
                hit = 0;
                const v = std.mem.indexOfMin(u32, ages);
                keys[v] = id + 1;
                break :blk v;
            };
            ages[way] = now;
            want[t] = hit | @as(u32, @intCast(base + way));
        }
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        const want = try k.scratch().alloc(u32, out.len);
        defer k.scratch().free(want);
        expected(k, want);
        for (out, want) |*o, w| o.* = @floatFromInt(w);
        @memset(mag, 0);
    }

    /// Exact: every plan, key and age is the reference's.
    pub fn check(k: anytype, _: spec.Gate) !?[]const u8 {
        const want = try k.scratch().alloc(u32, count(k.p, k.shape));
        defer k.scratch().free(want);
        expected(k, want);
        const got = k.rig.outputBytes();
        for (want, 0..) |w, i| {
            const v = std.mem.readInt(u32, got[4 * i ..][0..4], .little);
            if (v != w) return std.fmt.bufPrint(&k.msg, "output {d} is {d} where the reference gives {d}", .{ i, v, w }) catch "output";
        }
        return null;
    }
};

fn ci(v: usize) c_int {
    return @intCast(v);
}

fn depthOne(_: Shape) usize {
    return 1;
}

const plan_l = cv.launch("dsv41_dec_engram_kvc_plan", &PLAN_INS, &PLAN_OUTS, PLAN_SOURCE, "", 0, plan_threads, false);

/// The release's 48 tables (2 layers x 24 slots) at its sets, and a few odd ones.
pub fn register(set: *kc.Set) !void {
    try kit.add(set, Plan, &plan_l, .{}, .{
        .name = "dsv41 dec engram kv cache look-up (hits, misses into the oldest way)",
        .accum = .none,
        .depth = depthOne,
        .out = .u32,
        .nan = .{ .stated = "integer ids, keys and ages: no NaN" },
        .served = &.{ .{ .rows = 48, .n = comptime setsFor(48, 25600), .k = ways }, .{ .rows = 3, .n = 7, .k = ways }, .{ .rows = 1, .n = 1, .k = ways } },
    });
}

// ── tests ──

const testing = std.testing;

fn gpuOk() bool {
    return mlx.metalKernelsAvailable() and !mlx.noGpuBackend() and !mlx.virtualGpu();
}

test "dsv41 dec engram kv cache: the look-up's launch is the contract's, byte for byte" {
    if (!gpuOk()) return error.SkipZigTest;
    const a = testing.allocator;
    var c = kit.Case(Plan).init(a, &plan_l, .{});
    defer c.deinit();
    const S = 7;
    try testing.expect(try c.prepare(.{ .rows = 3, .n = S, .k = ways }, .random, 5));
    try c.run();
    const mine = try a.dupe(u8, c.outBytes());
    defer a.free(mine);
    var sc = Scope.init(a, c.rig.stream());
    defer sc.deinit();
    var cache: Cache = .{ .nt = 3, .sets = S, .n = 32, .keys = mlx.mlx_array_new(), .ages = mlx.mlx_array_new(), .vals = mlx.mlx_array_new(), .owner = 1 };
    defer cache.deinit();
    _ = mlx.mlx_array_set(&cache.keys, c.args[1]);
    _ = mlx.mlx_array_set(&cache.ages, c.args[2]);
    cache.clock = 4999;
    const p = try cache.plan(&sc, 1, c.args[0]);
    const theirs = try c.rig.bytesOf(&.{ p, cache.keys, cache.ages });
    try testing.expectEqualSlices(u8, mine, theirs);
}

/// A bf16 array's bytes.
fn bytesOf(a: std.mem.Allocator, x: A) ![]u8 {
    try mlx.check(mlx.mlx_array_eval(x));
    const p = mlx.mlx_array_data_uint8(x) orelse return error.NoData;
    return a.dupe(u8, p[0 .. 2 * mlx.mlx_array_size(x)]);
}

/// The hits in a plan (bit 31 of each slot's entry).
pub fn hits(p: A) !usize {
    try mlx.check(mlx.mlx_array_eval(p));
    const v = (mlx.mlx_array_data_uint32(p) orelse return error.NoData)[0..mlx.mlx_array_size(p)];
    var n: usize = 0;
    for (v) |x| n += x >> 31;
    return n;
}

test "dsv41 dec engram kv cache: each step's wkv through the cache is the slot kernel's bit for bit (all misses, all hits on zeroed weights, a mix, steps that evict, a state change), two layers, mxfp4 and mxfp8 table rows; a planted hit on another slot's entry differs" {
    if (!gpuOk()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xCAC4E);
    const rng = prng.random();
    const nc = 24;
    const hd = engram_kv.slot_hd;
    const K = nc * hd;
    const N = 256;
    const layers = 2;
    const rows_n = 300;
    for ([_]usize{ 4, 8 }) |tb| {
        // A table of `rows_n` random rows, each layer's weights random mxfp8.
        const tw = hd * tb / 32;
        const codes_h = try gpa.alloc(u32, rows_n * tw);
        defer gpa.free(codes_h);
        const tsc_h = try gpa.alloc(u8, rows_n * hd / 32);
        defer gpa.free(tsc_h);
        for (codes_h) |*v| v.* = if (tb == 4) rng.int(u32) else (rng.int(u32) & 0x77777777) | 0x30303030;
        for (tsc_h) |*v| v.* = 120 + rng.uintLessThan(u8, 7);
        const codes = mlx.mlx_array_new_data(codes_h.ptr, &[_]c_int{@intCast(codes_h.len)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(codes);
        const tsc = mlx.mlx_array_new_data(tsc_h.ptr, &[_]c_int{@intCast(tsc_h.len)}, 1, .uint8);
        defer _ = mlx.mlx_array_free(tsc);
        const gw = (engram_kv.Geo{ .w_base = 0, .w_stride = tw * 4, .s_base = 0, .s_stride = hd / 32 }).words();
        const geo = mlx.mlx_array_new_data(&gw, &[_]c_int{6}, 1, .uint32);
        defer _ = mlx.mlx_array_free(geo);
        var w: [layers]A = undefined;
        var wsc: [layers]A = undefined;
        const wh = try gpa.alloc(u32, N * K / 4);
        defer gpa.free(wh);
        const sh = try gpa.alloc(u8, N * K / 32);
        defer gpa.free(sh);
        for (0..layers) |l| {
            for (wh) |*v| v.* = (rng.int(u32) & 0x77777777) | 0x30303030;
            for (sh) |*v| v.* = 120 + rng.uintLessThan(u8, 7);
            w[l] = mlx.mlx_array_new_data(wh.ptr, &[_]c_int{ N, K / 4 }, 2, .uint32);
            wsc[l] = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{ N, K / 32 }, 2, .uint8);
        }
        defer for (0..layers) |l| {
            _ = mlx.mlx_array_free(w[l]);
            _ = mlx.mlx_array_free(wsc[l]);
        };
        @memset(wh, 0);
        @memset(sh, 127);
        const w0 = mlx.mlx_array_new_data(wh.ptr, &[_]c_int{ N, K / 4 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(w0);
        const s0 = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{ N, K / 32 }, 2, .uint8);
        defer _ = mlx.mlx_array_free(s0);
        // One set of 8 ways a table: the churn steps (12 rows a slot) evict.
        var cache: Cache = .{ .nt = layers * nc, .sets = 1, .n = N, .keys = mlx.mlx_array_new(), .ages = mlx.mlx_array_new(), .vals = mlx.mlx_array_new() };
        defer cache.deinit();
        try mlx.check(mlx.mlx_zeros(&cache.vals, &[_]c_int{ @intCast(cache.nt * cache.sets * ways), N }, 2, .float32, s));
        try cache.clear(s);
        const t = engram_kv.Table{ .hd = hd, .tb = tb };
        var ids_h: [layers * nc]u32 = undefined;
        for (&ids_h) |*v| v.* = rng.uintLessThan(u32, rows_n);
        const Step = enum { miss, hit, mix, again, churn, other_state, planted };
        const steps = [_]Step{ .miss, .hit, .mix, .again } ++ @as([24]Step, @splat(.churn)) ++ [_]Step{ .other_state, .planted };
        var churn_hits: usize = 0;
        for (steps) |step| {
            if (step == .mix) for (ids_h[0 .. nc / 2]) |*v| {
                v.* = rng.uintLessThan(u32, rows_n);
            };
            if (step == .churn) for (&ids_h) |*v| {
                v.* = rng.uintLessThan(u32, 12);
            };
            var sc = Scope.init(gpa, s);
            defer sc.deinit();
            const ids = try sc.keep(mlx.mlx_array_new_data(&ids_h, &[_]c_int{layers * nc}, 1, .uint32));
            const gen: u64 = if (step == .other_state or step == .planted) 2 else 1;
            var p = try cache.plan(&sc, gen, ids);
            const nh = try hits(p);
            if (step == .churn) churn_hits += nh;
            if (step == .planted) {
                // Every slot of a layer reads slot 0's entry: a hit on another row's partial.
                const ph = try gpa.dupe(u32, (mlx.mlx_array_data_uint32(p) orelse return error.NoData)[0 .. layers * nc]);
                defer gpa.free(ph);
                for (0..layers) |l| for (1..nc) |j| {
                    ph[l * nc + j] = ph[l * nc];
                };
                p = try sc.keep(mlx.mlx_array_new_data(ph.ptr, &[_]c_int{@intCast(ph.len)}, 1, .uint32));
            }
            var same = true;
            for (0..layers) |l| {
                const mine = try sc.slice(ids, 0, l * nc, (l + 1) * nc);
                const want = (try engram_kv.launchSlots(&sc, t, codes, tsc, mine, geo, w[l], wsc[l], null)) orelse return error.Declined;
                // On the all-hit steps the weights are zeros: every partial must come from the cache.
                const all_hit = step == .hit or step == .planted;
                const got = (try engram_kv.launchSlots(&sc, t, codes, tsc, mine, geo, if (all_hit) w0 else w[l], if (all_hit) s0 else wsc[l], try cache.of(&sc, p, l, nc))) orelse return error.Declined;
                cache.after(got);
                const a = try bytesOf(gpa, want);
                defer gpa.free(a);
                const b = try bytesOf(gpa, got);
                defer gpa.free(b);
                same = same and std.mem.eql(u8, a, b);
            }
            if (step != .churn) std.debug.print("[engram-kvc] tb {d} {t}: {d} of {d} slots hit; the cached wkv {s} the slot kernel's\n", .{ tb, step, nh, layers * nc, if (same) "equals" else "DIFFERS FROM" });
            switch (step) {
                .miss, .other_state => try testing.expectEqual(@as(usize, 0), nh),
                .hit, .again, .planted => try testing.expectEqual(@as(usize, layers * nc), nh),
                .mix => try testing.expect(nh >= nc and nh < layers * nc),
                .churn => {},
            }
            try testing.expectEqual(step != .planted, same);
        }
        std.debug.print("[engram-kvc] tb {d}: 24 churn steps (12 rows a slot, 8 ways) equal the slot kernel's, {d} of {d} slots hit\n", .{ tb, churn_hits, 24 * layers * nc });
        // 12 rows over 8 ways: some reuse hits, some evict (a ninth row in a full set).
        try testing.expect(churn_hits > 0 and churn_hits < 24 * layers * nc * 8 / 12);
    }
}
