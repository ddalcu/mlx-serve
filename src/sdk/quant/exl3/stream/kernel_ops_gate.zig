//! The kernel-ops GPU gate: the Python lanes' own device outputs (a fixture written in a
//! GPU run by the reference runtime's dump_kernel_ops_fixture.py) replayed through the ported
//! routes on the registry's kernels, every output word compared. Inputs are regenerated from
//! the fixture's seeded generators (splitmix64, the dump's twins) and checked by sha256 first.
//! GPU only: DSV41_KERNELS_GPU=1 and DSV41_KERNEL_OPS_FIXTURE=<dir> in a lock-holding window.

const std = @import("std");
const env = @import("mlx_host").env;
const kset = @import("../../../weights/streamed/sdk_ext/kernel_set.zig");
const mlx = @import("sdk").mlx;
const xk = @import("kernels.zig");
const sdk = @import("sdk");
const sdk_ext = @import("../../../weights/streamed/sdk_ext.zig");
const kr = sdk_ext.kernels.Routes(xk);
const tr = @import("kernel_routes.zig");
const xq = @import("quant.zig");

const Allocator = std.mem.Allocator;
const Dtype = mlx.mlx_dtype;

/// The routes' backend on real MLX arrays (the model's own backend plays this role there):
/// arrays created here are freed by `reset`; kept arrays by `release`.
pub const MlxG = struct {
    pub const T = mlx.mlx_array;

    a: Allocator,
    s: mlx.mlx_stream,
    bound: *const xk.Bound,
    live: std.ArrayList(mlx.mlx_array) = .empty,

    pub fn deinit(g: *MlxG) void {
        g.reset();
        g.live.deinit(g.a);
    }

    pub fn reset(g: *MlxG) void {
        g.resetTo(0);
    }

    /// The live list's length (a wave's start).
    pub fn mark(g: *const MlxG) usize {
        return g.live.items.len;
    }

    /// Frees every array tracked since `m`; kept handles survive.
    pub fn resetTo(g: *MlxG, m: usize) void {
        for (g.live.items[m..]) |x| _ = mlx.mlx_array_free(x);
        g.live.shrinkRetainingCapacity(m);
    }

    fn track(g: *MlxG, x: T) !T {
        g.live.append(g.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    pub fn shapeOf(_: *MlxG, x: T) kr.Shape {
        return .of(mlx.getShape(x));
    }

    pub fn dtypeOf(_: *MlxG, x: T) Dtype {
        return mlx.mlx_array_dtype(x);
    }

    pub fn hostArray(g: *MlxG, bytes: []const u8, shape: []const c_int, dt: Dtype) !T {
        const x = mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dt);
        if (x.ctx == null) return error.MlxError;
        return g.track(x);
    }

    pub fn keep(_: *MlxG, x: T) T {
        var k = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&k, x);
        return k;
    }

    pub fn release(_: *MlxG, x: T) void {
        _ = mlx.mlx_array_free(x);
    }

    pub fn reshape(g: *MlxG, x: T, shape: []const c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_reshape(&r, x, shape.ptr, shape.len, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn astype(g: *MlxG, x: T, dt: Dtype) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_astype(&r, x, dt, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn launch(g: *MlxG, k: xk.Kernel, inputs: []const T, cfg: *const xk.LaunchConfig, out: []T) !void {
        try g.bound.apply(k, inputs, cfg, out);
        for (out) |o| _ = try g.track(o);
    }

    pub const Prepared = xk.Prepared;

    pub fn prepareLaunch(g: *MlxG, k: xk.Kernel, cfg: *const xk.LaunchConfig) !Prepared {
        return g.bound.prepare(k, cfg);
    }

    pub fn launchPrepared(g: *MlxG, p: *const Prepared, inputs: []const T, out: []T) !void {
        try g.bound.applyPrepared(p, inputs, out);
        for (out) |o| _ = try g.track(o);
    }

    pub fn releasePrepared(_: *MlxG, p: *Prepared) void {
        p.deinit();
    }

    pub fn evalAll(_: *MlxG, xs: []const T) !void {
        const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_eval(v));
    }

    pub fn asyncEval(_: *MlxG, xs: []const T) !void {
        const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_async_eval(v));
    }

    pub fn concat(g: *MlxG, xs: []const T, axis: c_int) !T {
        const v = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(v);
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_concatenate_axis(&r, v, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    pub fn take(g: *MlxG, x: T, idx: T, axis: c_int) !T {
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_take_axis(&r, x, idx, axis, g.s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        return g.track(r);
    }

    /// Row-contiguous host bytes of `x` (caller frees).
    fn hostBytes(g: *MlxG, x: T) ![]u8 {
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, x, false, g.s));
        try mlx.check(mlx.mlx_array_eval(c));
        if (mlx.errorPending()) return error.MlxError;
        const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
        const p = mlx.mlx_array_data_uint8(c) orelse return error.MlxArrayDataNull;
        return g.a.dupe(u8, p[0..n]);
    }
};

// ── The fixture (dump_kernel_ops_fixture.py spec.json) ──

const JGen = struct {
    kind: []const u8,
    seed: u64 = 0,
    lo: f64 = 0,
    hi: f64 = 0,
    rows: []const u32 = &.{},
    slots: []const u32 = &.{},
    tiles: u32 = 0,
    values: []const f64 = &.{},
};
const JArray = struct { name: []const u8, dtype: []const u8, shape: []const i64, gen: ?JGen = null, sha256: []const u8, file: ?[]const u8 = null };
const JVars = struct { rows: u64 = 0, cap: u64 = 0, experts: u64 = 0 };
const JCase = struct { family: []const u8, case: []const u8, vars: JVars = .{}, site: ?[]const u8 = null, proj: ?[]const u8 = null, eps: ?f64 = null, inputs: []const JArray, outputs: []const JArray };
const JSpec = struct { format: []const u8, manifest_sha256: []const u8, cases: []const JCase };

const fixture_format = "mlx-serve-exl3-kernel-ops-fixture-v1";
const draft_fixture_format = "mlx-serve-exl3-kernel-draft-fixture-v1";
const decode2_fixture_format = "mlx-serve-exl3-kernel-decode2-fixture-v1";
const prefill2_fixture_format = "mlx-serve-exl3-kernel-prefill2-fixture-v1";
const golden: u64 = 0x9E3779B97F4A7C15;

fn sm(seed: u64, i: u64) u64 {
    var z = seed +% (i + 1) *% golden;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn dtypeOf(name: []const u8) Dtype {
    const map = [_]struct { []const u8, Dtype }{
        .{ "float32", .float32 }, .{ "float16", .float16 }, .{ "bfloat16", .bfloat16 }, .{ "int16", .int16 },
        .{ "int32", .int32 },     .{ "uint32", .uint32 },   .{ "uint8", .uint8 },       .{ "bool", .bool_ },
    };
    for (map) |m| if (std.mem.eql(u8, m[0], name)) return m[1];
    unreachable;
}

fn size(dt: Dtype) usize {
    return switch (dt) {
        .uint8, .bool_ => 1,
        .int16, .float16, .bfloat16 => 2,
        .int32, .uint32, .float32 => 4,
        else => unreachable,
    };
}

fn putInt(b: []u8, i: usize, dt: Dtype, v: u64) void {
    switch (dt) {
        .uint8, .bool_ => b[i] = @truncate(v),
        .int32, .uint32 => std.mem.writeInt(u32, b[i * 4 ..][0..4], @truncate(v), .little),
        else => unreachable,
    }
}

/// The bytes of one input, as the dump's `generate` builds them.
fn generate(a: Allocator, gen: JGen, dt: Dtype, n: usize) ![]u8 {
    const b = try a.alloc(u8, n * size(dt));
    errdefer a.free(b);
    const kind = std.meta.stringToEnum(enum { bits, bf16bits, uniform, index, range, srange, wave_rhs, wave_table, slots16, values }, gen.kind) orelse return error.FixtureGenerator;
    switch (kind) {
        .bits, .bf16bits => {
            var w: usize = 0;
            while (w * 8 < b.len) : (w += 1) {
                var le: [8]u8 = undefined;
                std.mem.writeInt(u64, &le, sm(gen.seed, w), .little);
                const end = @min(b.len, w * 8 + 8);
                @memcpy(b[w * 8 .. end], le[0 .. end - w * 8]);
            }
            // bf16bits: the stream's u16 words as bf16, exponent pinned to 0x78..0x7B
            if (kind == .bf16bits) for (0..n) |i| {
                const v = std.mem.readInt(u16, b[i * 2 ..][0..2], .little);
                std.mem.writeInt(u16, b[i * 2 ..][0..2], (v & 0x807F) | ((0x78 + ((v >> 7) & 3)) << 7), .little);
            };
        },
        .uniform => for (0..n) |i| {
            const t = @as(f64, @floatFromInt(sm(gen.seed, i) >> 11)) * 0x1p-53;
            const v: f32 = @floatCast(gen.lo + (gen.hi - gen.lo) * t);
            switch (dt) {
                .float32 => std.mem.writeInt(u32, b[i * 4 ..][0..4], @bitCast(v), .little),
                .float16 => std.mem.writeInt(u16, b[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
                .bfloat16 => {
                    const u: u64 = @as(u32, @bitCast(v));
                    std.mem.writeInt(u16, b[i * 2 ..][0..2], @truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16), .little);
                },
                else => unreachable,
            }
        },
        .index => for (0..n) |i| putInt(b, i, dt, sm(gen.seed, i) % @as(u64, @intFromFloat(gen.hi))),
        .range => for (0..n) |i| putInt(b, i, dt, @as(u64, @intFromFloat(gen.lo)) + sm(gen.seed, i) % @as(u64, @intFromFloat(gen.hi - gen.lo))),
        // prefill batch 2: a signed range (lo may be negative: the compressed selection's -1 = no key)
        .srange => for (0..n) |i| putInt(b, i, dt, @bitCast(@as(i64, @intFromFloat(gen.lo)) + @as(i64, @intCast(sm(gen.seed, i) % @as(u64, @intFromFloat(gen.hi - gen.lo)))))),
        .wave_rhs => {
            var r: usize = 0;
            for (gen.rows, 0..) |rows, j| for (0..rows) |_| {
                putInt(b, r, dt, j);
                r += 1;
            };
        },
        .wave_table => {
            var ex: [xq.wave_max]xq.WaveExpert = undefined;
            for (gen.slots, gen.rows, 0..) |s, r, j| ex[j] = .{ .slot = s, .rows = r };
            const t = xq.digTableBm(ex[0..gen.slots.len], gen.tiles, 64);
            @memcpy(b, std.mem.sliceAsBytes(&t.table));
        },
        .slots16 => {
            const t = xq.rebuildSlots(gen.slots);
            @memcpy(b, std.mem.sliceAsBytes(&t));
        },
        .values => for (0..n) |i| std.mem.writeInt(u32, b[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(gen.values[i]))), .little),
    }
    return b;
}

fn hexEql(want: []const u8, bytes: []const u8) bool {
    return std.mem.eql(u8, want, &kset.shaHex(bytes));
}

pub const Line = struct {
    family: []const u8,
    case: []const u8,
    output: []const u8,
    words: u64 = 0,
    bad: u64 = 0,
    ok: bool = false,
    err: []const u8 = "",
};

/// One case: its inputs regenerated and checked, the routes run, every output compared.
fn replayCase(a: Allocator, g: *MlxG, reg: *const xk.Registry, dir: []const u8, c: *const JCase, lines: *std.ArrayList(Line)) !void {
    var ins: std.StringHashMapUnmanaged(mlx.mlx_array) = .empty;
    defer ins.deinit(a);
    for (c.inputs) |*i| {
        const dt = dtypeOf(i.dtype);
        var shape: [8]c_int = undefined;
        var n: usize = 1;
        for (i.shape, 0..) |d, k| {
            shape[k] = @intCast(d);
            n *= @intCast(d);
        }
        const bytes = try generate(a, i.gen.?, dt, n);
        defer a.free(bytes);
        if (!hexEql(i.sha256, bytes)) {
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = i.name, .err = "input generator differs from the dump's" });
            return;
        }
        try ins.put(a, i.name, try g.hostArray(bytes, shape[0..i.shape.len], dt));
    }
    var outs: [16]mlx.mlx_array = undefined;
    const n_out = try runFamily(g, reg, c, &ins, &outs);
    if (n_out != c.outputs.len) return error.FixtureOutputs;
    const v = mlx.mlx_vector_array_new_data(&outs, n_out);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
    for (c.outputs, outs[0..n_out]) |*o, got_arr| {
        var line: Line = .{ .family = c.family, .case = c.case, .output = o.name };
        const got = try g.hostBytes(got_arr);
        defer a.free(got);
        const path = try std.fs.path.join(a, &.{ dir, o.file.? });
        defer a.free(path);
        const want = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 30));
        defer a.free(want);
        const w = size(dtypeOf(o.dtype));
        if (g.dtypeOf(got_arr) != dtypeOf(o.dtype) or got.len != want.len) {
            line.err = "dtype or size differs";
        } else {
            line.words = got.len / w;
            var k: usize = 0;
            while (k < got.len) : (k += w) line.bad += @intFromBool(!std.mem.eql(u8, got[k..][0..w], want[k..][0..w]));
            line.ok = line.bad == 0 and hexEql(o.sha256, want);
        }
        try lines.append(a, line);
    }
}

fn in(ins: *std.StringHashMapUnmanaged(mlx.mlx_array), name: []const u8) mlx.mlx_array {
    return ins.get(name).?;
}

/// The ported calls of one family, outputs in the dump's order.
fn runFamily(g: *MlxG, reg: *const xk.Registry, c: *const JCase, ins: *std.StringHashMapUnmanaged(mlx.mlx_array), outs: *[16]mlx.mlx_array) !usize {
    const f = c.family;
    const eq = std.mem.eql;
    // decode batch 2 (dump_kernel_decode2_fixture.py): the members the RC tiers still run,
    // ATTN_FUSE softmax, INDEX_TOPK and the wo_a ring transpose
    if (eq(u8, f, "woa_transpose")) {
        var r = try tr.WoaRingTranspose(MlxG).init(g, reg);
        defer r.deinit(g);
        try r.checkLayer(g, in(ins, "packed"), in(ins, "scales"), null);
        outs[0] = try r.call(g, in(ins, "packed"), in(ins, "scales"));
        return 1;
    }
    if (eq(u8, f, "index_topk")) {
        var r = try tr.IndexTopk(MlxG).init(g, reg, &.derived, null);
        defer r.deinit(g);
        outs[0..2].* = try r.select(g, in(ins, "score"), in(ins, "clen"));
        return 2;
    }
    if (eq(u8, f, "attn_fuse")) {
        var r = try tr.AttnSoftmax(MlxG).init(g, reg, null);
        defer r.deinit(g);
        outs[0..2].* = try r.call(g, in(ins, "qk"), in(ins, "valid"), in(ins, "sink"));
        return 2;
    }
    // x1..x8: one call per M on the case's one weight
    const x_names = [_][]const u8{ "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8" };
    if (eq(u8, f, "mxfp8_rows")) {
        const site = std.meta.stringToEnum(tr.M1Site, c.site.?) orelse return error.FixtureSite;
        var r = try tr.Mxfp8Rows(MlxG).init(g, reg, site, in(ins, "w"), in(ins, "scales"), null);
        defer r.deinit(g);
        for (0..8) |i| outs[i] = try r.call(g, in(ins, x_names[i]));
        return 8;
    }
    if (eq(u8, f, "head_rows")) {
        var r = try tr.HeadRows(MlxG).init(g, reg, in(ins, "w"), null);
        defer r.deinit(g);
        for (0..8) |i| outs[i] = try r.call(g, in(ins, x_names[i]));
        return 8;
    }
    if (eq(u8, f, "smallm")) {
        const site = std.meta.stringToEnum(tr.SmallMSite, c.site.?) orelse return error.FixtureSite;
        var r = try tr.SmallM(MlxG).init(g, reg, site, in(ins, "w"), null);
        defer r.deinit(g);
        for (0..8) |i| outs[i] = try r.call(g, in(ins, x_names[i]));
        return 8;
    }
    // prefill batch 2 (dump_kernel_prefill2_fixture.py): the P line's prefill-rows texts; the
    // prefill-rows index top-k replays through the index_topk branch above
    if (eq(u8, f, "idxscore")) {
        const r = try tr.IdxScore(MlxG).init(reg, &.derived, null);
        outs[0] = try r.call(g, in(ins, "q"), in(ins, "k"), in(ins, "w"), in(ins, "clen"));
        return 1;
    }
    if (eq(u8, f, "core_vec") or eq(u8, f, "core_rope")) {
        const kind: tr.CoreKind = if (eq(u8, f, "core_vec")) .vec else .rope;
        const ckv = ins.get("ckv");
        var r = try tr.PrefillAttn(MlxG).init(g, reg, &.derived, kind, g.dtypeOf(in(ins, "q")), g.dtypeOf(in(ins, "win")), ckv != null, null);
        defer r.deinit(g);
        const cmp: ?[2]mlx.mlx_array = if (ckv) |store| .{ store, in(ins, "cidx") } else null;
        const rope: ?[2]mlx.mlx_array = if (kind == .rope) .{ in(ins, "qcos"), in(ins, "qsin") } else null;
        outs[0] = try r.attend(g, in(ins, "q"), in(ins, "win"), in(ins, "widx"), in(ins, "wval"), cmp, in(ins, "sink"), rope);
        return 1;
    }
    if (eq(u8, f, "hcnorm")) {
        const x = in(ins, "x");
        var r = try tr.HcNorm(MlxG).init(g, reg, &.derived, g.dtypeOf(x), @floatCast(c.eps orelse return error.FixtureEps), null);
        defer r.deinit(g);
        outs[0] = try r.rsqrt(g, x);
        outs[1] = try r.preNorm(g, x, in(ins, "pre"), in(ins, "w"));
        return 2;
    }
    if (eq(u8, f, "smallk")) {
        const r = try tr.SmallKCombine(MlxG).init(reg, &.derived, null);
        outs[0] = try r.call(g, in(ins, "routed"), in(ins, "weights"), in(ins, "shared"));
        return 1;
    }
    // DRAFTRC (dump_draftrc_fixture.py): the draft routes, outputs in the dump's order
    if (eq(u8, f, "draft_proj")) {
        const site = std.meta.stringToEnum(tr.DraftSite, c.site.?) orelse return error.FixtureSite;
        const x1 = in(ins, "x1");
        var r = try tr.DraftProj(MlxG).init(g, reg, site, g.dtypeOf(x1), in(ins, "w"), in(ins, "scales"), null);
        defer r.deinit(g);
        outs[0] = try r.call(g, x1);
        outs[1] = try r.call(g, in(ins, "x6"));
        outs[2] = try r.call(g, in(ins, "x8"));
        return 3;
    }
    if (eq(u8, f, "draft_router")) {
        var r = try tr.Router(MlxG).init(g, reg, in(ins, "w"), in(ins, "bias"), null);
        defer r.deinit(g);
        outs[0..2].* = try r.call(g, in(ins, "x1"));
        outs[2..4].* = try r.call(g, in(ins, "x6"));
        outs[4..6].* = try r.call(g, in(ins, "x8"));
        return 6;
    }
    if (eq(u8, f, "draft_tape")) {
        var r = try tr.HcTape(MlxG).init(g, reg, .float32, null);
        defer r.deinit(g);
        var mixed = try tr.HcTapeMixed(MlxG).init(g, reg, null);
        defer mixed.deinit(g);
        const x, const rr, const rb, const post, const comb, const pre, const w = .{ in(ins, "x"), in(ins, "r"), in(ins, "rb"), in(ins, "post"), in(ins, "comb"), in(ins, "pre"), in(ins, "w") };
        outs[0] = try r.combine(g, x, rr, post, comb);
        outs[1..4].* = try r.collapseNorm(g, rr, pre, w);
        outs[4..8].* = try r.combineCollapseNorm(g, x, rr, post, comb, pre, w);
        outs[8..12].* = try mixed.call(g, x, rb, post, comb, pre, w);
        return 12;
    }
    if (eq(u8, f, "router")) {
        var r = try tr.Router(MlxG).init(g, reg, in(ins, "w"), in(ins, "bias"), null);
        defer r.deinit(g);
        outs[0..2].* = try r.call(g, in(ins, "x"));
        return 2;
    }
    if (eq(u8, f, "premix")) {
        var r = try tr.Premix(MlxG).init(g, reg, in(ins, "w"), null);
        defer r.deinit(g);
        outs[0] = try r.call(g, in(ins, "x"));
        return 1;
    }
    if (eq(u8, f, "sinkhorn")) {
        var r = try tr.Sinkhorn(MlxG).init(g, reg);
        defer r.deinit(g);
        outs[0] = try r.call(g, in(ins, "comb"));
        return 1;
    }
    if (eq(u8, f, "rcproj")) {
        const site = std.meta.stringToEnum(tr.RcSite, c.site.?).?;
        var r = try tr.RcProj(MlxG).init(g, reg, site, in(ins, "w"), in(ins, "scales"), null);
        defer r.deinit(g);
        outs[0] = try r.call(g, in(ins, "x"));
        return 1;
    }
    if (eq(u8, f, "hctape")) {
        var r = try tr.HcTape(MlxG).init(g, reg, .bfloat16, null);
        defer r.deinit(g);
        const x, const rr, const post, const comb, const pre, const w = .{ in(ins, "x"), in(ins, "r"), in(ins, "post"), in(ins, "comb"), in(ins, "pre"), in(ins, "w") };
        outs[0] = try r.combine(g, x, rr, post, comb);
        outs[1..4].* = try r.collapseNorm(g, rr, pre, w);
        outs[4..8].* = try r.combineCollapseNorm(g, x, rr, post, comb, pre, w);
        outs[8..11].* = try r.mixfin(g, in(ins, "mm"), in(ins, "ssq"), in(ins, "scale"), in(ins, "base"));
        return 11;
    }
    if (eq(u8, f, "fused_proj")) {
        // the fixture's eps: dump_kernel_ops_fixture.py reads the bank config's rms_norm_eps (1e-20)
        var r = try tr.FusedProj(MlxG).init(g, reg, in(ins, "q_norm"), in(ins, "kv_norm"), 1e-20, null);
        defer r.deinit(g);
        const cos, const sin = .{ in(ins, "cos"), in(ins, "sin") };
        outs[0] = try r.qNorm(g, in(ins, "x_q"));
        outs[1] = try r.kvNormRope(g, in(ins, "x_kv"), cos, sin);
        outs[2] = try r.ropeHeads(g, in(ins, "x_rope"), cos, sin, .fwd);
        outs[3] = try r.ropeHeads(g, in(ins, "x_o"), cos, sin, .inv);
        return 4;
    }
    if (eq(u8, f, "gemv")) {
        var r = try xq.Gemv(MlxG).init(g, reg);
        defer r.deinit(g);
        const proj: xq.Proj = if (eq(u8, c.proj.?, "down")) .down else .gate;
        outs[0] = try r.project(g, proj, in(ins, "xh"), in(ins, "ids"), in(ins, "code"));
        return 1;
    }
    if (eq(u8, f, "prep")) {
        var r = try xq.RinPrep(MlxG).init(g, reg);
        defer r.deinit(g);
        const ids = in(ins, "ids");
        outs[0..2].* = try r.inRin(g, in(ins, "x"), in(ins, "tok"), in(ins, "rin_g"), in(ins, "rin_u"), ids);
        outs[2] = try r.guEpi(g, in(ins, "zg"), in(ins, "zu"), in(ins, "rout_g"), in(ins, "rout_u"), ids);
        outs[3] = try r.dinRin(g, in(ins, "hid"), in(ins, "rin_d"), ids);
        outs[4] = try r.dpost(g, in(ins, "zd"), in(ins, "rout_d"), ids);
        return 5;
    }
    const gate: xq.ProjArrays(mlx.mlx_array) = .{ .code = in(ins, "code_g"), .rout = in(ins, "rout_g"), .rin = in(ins, "rin_g") };
    const up: xq.ProjArrays(mlx.mlx_array) = .{ .code = in(ins, "code_u"), .rout = in(ins, "rout_u"), .rin = in(ins, "rin_u") };
    const down: xq.ProjArrays(mlx.mlx_array) = .{ .code = in(ins, "code_d"), .rout = in(ins, "rout_d"), .rin = in(ins, "rin_d") };
    if (eq(u8, f, "rebuild")) {
        const r = xq.Rebuild(MlxG).init(reg);
        outs[0..3].* = try r.call(g, gate, up, down, in(ins, "slots"), @intCast(c.vars.experts));
        return 3;
    }
    if (eq(u8, f, "digx")) {
        const r = xq.DigX(MlxG).init(reg);
        const rhs = in(ins, "rhs");
        const tbl_gu = in(ins, "tbl_gu");
        const tbl_dn = in(ins, "tbl_dn");
        // the GEMMs over the fixture's tables at the routed M tile; the rot and onepass stages read their slot
        // column only
        const gu = try routedTable(g, tbl_gu);
        const dn = try routedTable(g, tbl_dn);
        const x = try r.take2(g, in(ins, "act"), in(ins, "ridx"), rhs, tbl_gu, gate.rin, up.rin);
        const z = try r.gemmGateUp(g, x[0], x[1], gate.code, up.code, gu.tbl, gu.tgs);
        const hd = try r.onePass(g, z[0], z[1], rhs, tbl_gu, gate.rout, up.rout, down.rin);
        const zd = try r.gemmDown(g, hd, down.code, dn.tbl, dn.tgs);
        const o = try r.widen1(g, zd, rhs, tbl_dn, down.rout);
        const rx = try r.roundx(g, in(ins, "act_r"), rhs, tbl_gu, down.rin);
        const w2 = try r.widen2(g, in(ins, "act_g"), in(ins, "act_u"), rhs, tbl_gu, gate.rout, up.rout);
        outs[0..10].* = .{ x[0], x[1], z[0], z[1], hd, zd, o, rx, w2[0], w2[1] };
        return 10;
    }
    return error.FixtureFamily;
}

/// A fixture's 64-row wave table at the routed GEMMs' M tile (`DigX.m_tile`): the same slot, first-row and row
/// columns, the first-threadgroup column counting ceil(rows / m_tile) tiles; the tiles per M tile from its own
/// threadgroup count (entry 65).
fn routedTable(g: *MlxG, tbl: mlx.mlx_array) !xq.DigTableArray(MlxG) {
    const b = try g.hostBytes(tbl);
    defer g.a.free(b);
    var t: [80]i32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&t), b[0 .. 80 * 4]);
    const n: usize = @intCast(t[64]);
    var ex: [xq.wave_max]xq.WaveExpert = undefined;
    var t64: u32 = 0;
    for (0..n) |j| {
        ex[j] = .{ .slot = @intCast(t[j]), .rows = @intCast(t[32 + j]) };
        t64 += (ex[j].rows + 63) / 64;
    }
    const tiles = @divExact(@as(u32, @intCast(t[65])), t64);
    const r = xq.digTableBm(ex[0..n], tiles, xq.DigX(MlxG).m_tile);
    return .{ .tbl = try g.hostArray(std.mem.sliceAsBytes(&r.table), &.{80}, .int32), .tgs = r.tgs };
}

fn wanted(filter: ?[]const u8, family: []const u8) bool {
    const f = filter orelse return true;
    var it = std.mem.splitScalar(u8, f, ',');
    while (it.next()) |name| if (std.mem.eql(u8, name, family) or std.mem.eql(u8, name, "all")) return true;
    return false;
}

// ── The prefill wave fixture (dump_prefill_waves.py spec.json, window PF) ──

const PShape = struct { wave: u32, inflight: u32, row_budget: u32, carry_rows: u32 };
const PCall = struct { name: []const u8, a_rows: u32, slots: []const u32, act: JArray, output: JArray };
const PCase = struct { family: []const u8, case: []const u8, shape: PShape, cap: u32, inputs: []const JArray, calls: []const PCall };
const PSpec = struct { format: []const u8, manifest_sha256: []const u8, cases: []const PCase };
const prefill_format = "mlx-serve-exl3-prefill-waves-fixture-v1";

/// An input regenerated from its generator; null when its bytes are not the dump's (sha256).
fn regen(a: Allocator, g: *MlxG, i: *const JArray) !?mlx.mlx_array {
    const dt = dtypeOf(i.dtype);
    var shape: [8]c_int = undefined;
    var n: usize = 1;
    for (i.shape, 0..) |d, k| {
        shape[k] = @intCast(d);
        n *= @intCast(d);
    }
    const bytes = try generate(a, i.gen.?, dt, n);
    defer a.free(bytes);
    if (!hexEql(i.sha256, bytes)) return null;
    return try g.hostArray(bytes, shape[0..i.shape.len], dt);
}

/// One case: the bank and each call's act regenerated and checked, the route's calls in order on
/// one route (then its prefill boundary), every call's result compared with the lane's, word for word.
fn replayPrefill(a: Allocator, g: *MlxG, reg: *const xk.Registry, dir: []const u8, c: *const PCase, lines: *std.ArrayList(Line), diag: *xk.Diag) !void {
    var ins: std.StringHashMapUnmanaged(mlx.mlx_array) = .empty;
    defer ins.deinit(a);
    for (c.inputs) |*i| {
        const x = try regen(a, g, i) orelse {
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = i.name, .err = "input generator differs from the dump's" });
            return;
        };
        try ins.put(a, i.name, x);
    }
    const P = xq.ProjArrays(mlx.mlx_array);
    const bank: xq.BankArrays(mlx.mlx_array) = .{
        .gate = P{ .code = in(&ins, "gate_proj.code"), .rout = in(&ins, "gate_proj.rout"), .rin = in(&ins, "gate_proj.rin") },
        .up = P{ .code = in(&ins, "up_proj.code"), .rout = in(&ins, "up_proj.rout"), .rin = in(&ins, "up_proj.rin") },
        .down = P{ .code = in(&ins, "down_proj.code"), .rout = in(&ins, "down_proj.rout"), .rin = in(&ins, "down_proj.rin") },
    };
    try xq.checkBank(MlxG, g, reg, .gate, bank.gate, diag);
    try xq.checkBank(MlxG, g, reg, .up, bank.up, diag);
    try xq.checkBank(MlxG, g, reg, .down, bank.down, diag);
    const shape: xq.PrefillShape = .{ .wave = c.shape.wave, .inflight = c.shape.inflight, .row_budget = c.shape.row_budget, .carry_rows = c.shape.carry_rows };
    var r = try xq.DigXPrefill(MlxG).init(a, reg, shape, diag);
    defer r.deinit(g);
    var results: std.ArrayList(mlx.mlx_array) = .empty;
    defer results.deinit(a);
    for (c.calls) |*cl| {
        const act = try regen(a, g, &cl.act) orelse {
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = cl.name, .err = "act generator differs from the dump's" });
            return;
        };
        try results.append(a, try r.call(g, act, .{ .slot = cl.slots }, bank));
    }
    defer for (results.items) |x| g.release(x);
    try r.finish(g);
    try g.evalAll(results.items);
    for (c.calls, results.items) |*cl, got_arr| {
        const o = &cl.output;
        var line: Line = .{ .family = c.family, .case = c.case, .output = cl.name };
        const got = try g.hostBytes(got_arr);
        defer a.free(got);
        const path = try std.fs.path.join(a, &.{ dir, o.file.? });
        defer a.free(path);
        const want = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, a, .limited(1 << 31));
        defer a.free(want);
        if (g.dtypeOf(got_arr) != dtypeOf(o.dtype) or got.len != want.len) {
            line.err = "dtype or size differs";
        } else {
            line.words = got.len / 4;
            var k: usize = 0;
            while (k < got.len) : (k += 4) line.bad += @intFromBool(!std.mem.eql(u8, got[k..][0..4], want[k..][0..4]));
            line.ok = line.bad == 0 and hexEql(o.sha256, want);
        }
        try lines.append(a, line);
    }
}

const testing = std.testing;

/// One fixture replayed through the routes (the ops gate and the draft gate): the spec's format and
/// manifest pin checked, every case's inputs regenerated and checked, every output word compared;
/// one JSON line per output (the receipt), then `[<label>] <cases> cases, <outputs> outputs, <failed> failed`.
fn replayFixture(dir: []const u8, format: []const u8, filter: ?[]const u8, receipt: ?[*:0]const u8, comptime label: []const u8) !void {
    const a = testing.allocator;
    mlx.installErrorHandler();
    const spec_path = try std.fs.path.join(a, &.{ dir, "spec.json" });
    defer a.free(spec_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, spec_path, a, .limited(16 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(JSpec, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const spec = parsed.value;
    try testing.expectEqualStrings(format, spec.format);
    var diag: xk.Diag = .{};
    var reg = xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("exl3 kernels refused: {s}\n", .{diag.message()});
        return e;
    };
    defer reg.deinit();
    // this manifest or a predecessor whose kernels are unchanged here (the exporter's check)
    if (!reg.acceptsManifest(spec.manifest_sha256)) std.debug.print("fixture manifest {s} is neither {s} nor a predecessor\n", .{ spec.manifest_sha256, xk.manifest_sha256 });
    try testing.expect(reg.acceptsManifest(spec.manifest_sha256));
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    var lines: std.ArrayList(Line) = .empty;
    defer lines.deinit(a);
    var cases: usize = 0;
    for (spec.cases) |*c| {
        if (!wanted(filter, c.family)) continue;
        cases += 1;
        replayCase(a, &g, &reg, dir, c, &lines) catch |e| {
            var buf: [256]u8 = undefined;
            const msg = mlx.takeError(&buf) orelse "";
            std.debug.print("[" ++ label ++ "] {s} {s}: {t} {s}\n", .{ c.family, c.case, e, msg });
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = "", .err = @errorName(e) });
        };
        g.reset();
    }
    var failed: usize = 0;
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    for (lines.items) |l| {
        failed += @intFromBool(!l.ok);
        try j.print(a, "{{\"family\":\"{s}\",\"case\":\"{s}\",\"output\":\"{s}\",\"words\":{d},\"bad\":{d},\"ok\":{},\"err\":\"{s}\"}}\n", .{ l.family, l.case, l.output, l.words, l.bad, l.ok, l.err });
    }
    std.debug.print("{s}", .{j.items});
    if (receipt) |path| try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = std.mem.span(path), .data = j.items });
    std.debug.print("[" ++ label ++ "] {d} cases, {d} outputs, {d} failed\n", .{ cases, lines.items.len, failed });
    try testing.expect(cases > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

// GPU only: DSV41_KERNELS_GPU=1,
// DSV41_KERNEL_OPS_FIXTURE=<fixture dir>; DSV41_KERNEL_OPS_FAMILIES=<a,b|all> narrows it,
// DSV41_KERNEL_OPS_RECEIPT=<path> keeps the per-output JSON lines.
test "dsv41 kernels ops gpu: every route reproduces its lane's own device output (fixture), bitwise" {
    _ = env.get(.DSV41_KERNELS_GPU) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_KERNEL_OPS_FIXTURE) orelse return error.SkipZigTest);
    const filter: ?[]const u8 = if (env.get(.DSV41_KERNEL_OPS_FAMILIES)) |f| std.mem.span(f) else null;
    try replayFixture(dir, fixture_format, filter, env.get(.DSV41_KERNEL_OPS_RECEIPT), "kernel ops gate");
}

// GPU only (block (t)): DSV41_KERNELS_GPU=1, DSV41_KERNEL_DRAFT_FIXTURE=<dir> (the
// dump_draftrc_fixture.py fixture); DSV41_KERNEL_DRAFT_RECEIPT=<path> keeps the per-output JSON lines.
test "dsv41 kernels ops gpu: the DRAFTRC routes reproduce the lane's own draft kernels (fixture), bitwise" {
    _ = env.get(.DSV41_KERNELS_GPU) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_KERNEL_DRAFT_FIXTURE) orelse return error.SkipZigTest);
    try replayFixture(dir, draft_fixture_format, null, env.get(.DSV41_KERNEL_DRAFT_RECEIPT), "kernel draft gate");
}

// GPU only (decode batch 2): DSV41_KERNELS_GPU=1, DSV41_KERNEL_DECODE2_FIXTURE=<dir>
// (the dump_kernel_decode2_fixture.py fixture); DSV41_KERNEL_DECODE2_RECEIPT=<path> keeps the lines.
test "dsv41 kernels ops gpu: the decode batch 2 routes reproduce their lanes' own device outputs (fixture), bitwise" {
    _ = env.get(.DSV41_KERNELS_GPU) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_KERNEL_DECODE2_FIXTURE) orelse return error.SkipZigTest);
    try replayFixture(dir, decode2_fixture_format, null, env.get(.DSV41_KERNEL_DECODE2_RECEIPT), "kernel decode2 gate");
}

// GPU only (prefill batch 2): DSV41_KERNELS_GPU=1, DSV41_KERNEL_PREFILL2_FIXTURE=<dir>
// (the dump_kernel_prefill2_fixture.py fixture); DSV41_KERNEL_PREFILL2_RECEIPT=<path> keeps the lines.
test "dsv41 kernels ops gpu: the prefill batch 2 routes reproduce their lanes' own device outputs (fixture), bitwise" {
    _ = env.get(.DSV41_KERNELS_GPU) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_KERNEL_PREFILL2_FIXTURE) orelse return error.SkipZigTest);
    try replayFixture(dir, prefill2_fixture_format, null, env.get(.DSV41_KERNEL_PREFILL2_RECEIPT), "kernel prefill2 gate");
}

// GPU only (window PG): DSV41_KERNELS_GPU=1, DSV41_KERNEL_PREFILL_FIXTURE=<dir>;
// DSV41_KERNEL_PREFILL_RECEIPT=<path> keeps the per-call JSON lines.
test "dsv41 kernels ops gpu: the prefill wave route reproduces the lane's own dispatch output (fixture), bitwise" {
    _ = env.get(.DSV41_KERNELS_GPU) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_KERNEL_PREFILL_FIXTURE) orelse return error.SkipZigTest);
    const a = testing.allocator;
    mlx.installErrorHandler();
    const spec_path = try std.fs.path.join(a, &.{ dir, "spec.json" });
    defer a.free(spec_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, spec_path, a, .limited(16 << 20));
    defer a.free(text);
    const parsed = try std.json.parseFromSlice(PSpec, a, text, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const spec = parsed.value;
    try testing.expectEqualStrings(prefill_format, spec.format);
    var diag: xk.Diag = .{};
    var reg = xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("exl3 kernels refused: {s}\n", .{diag.message()});
        return e;
    };
    defer reg.deinit();
    try testing.expect(reg.acceptsManifest(spec.manifest_sha256));
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = try reg.bind(stream, &diag);
    defer bound.deinit();
    var g: MlxG = .{ .a = a, .s = stream, .bound = &bound };
    defer g.deinit();
    var lines: std.ArrayList(Line) = .empty;
    defer lines.deinit(a);
    for (spec.cases) |*c| {
        replayPrefill(a, &g, &reg, dir, c, &lines, &diag) catch |e| {
            var buf: [256]u8 = undefined;
            const msg = mlx.takeError(&buf) orelse "";
            std.debug.print("[kernel prefill gate] {s}: {t} {s} {s}\n", .{ c.case, e, msg, diag.message() });
            try lines.append(a, .{ .family = c.family, .case = c.case, .output = "", .err = @errorName(e) });
        };
        g.reset();
    }
    var failed: usize = 0;
    var j: std.ArrayList(u8) = .empty;
    defer j.deinit(a);
    for (lines.items) |l| {
        failed += @intFromBool(!l.ok);
        try j.print(a, "{{\"family\":\"{s}\",\"case\":\"{s}\",\"call\":\"{s}\",\"words\":{d},\"bad\":{d},\"ok\":{},\"err\":\"{s}\"}}\n", .{ l.family, l.case, l.output, l.words, l.bad, l.ok, l.err });
    }
    std.debug.print("{s}", .{j.items});
    if (env.get(.DSV41_KERNEL_PREFILL_RECEIPT)) |path| try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = std.mem.span(path), .data = j.items });
    std.debug.print("[kernel prefill gate] {d} cases, {d} calls, {d} failed\n", .{ spec.cases.len, lines.items.len, failed });
    try testing.expect(spec.cases.len > 0);
    try testing.expectEqual(@as(usize, 0), failed);
}

test "dsv41 kernels ops: the fixture generators are the dump's (sha256 of its --golden cases)" {
    const a = testing.allocator;
    const cases = [_]struct { JGen, Dtype, usize, []const u8 }{
        .{ .{ .kind = "bits", .seed = 11 }, .uint8, 37, "83e82e581b5636875ba0a57d454acc2de295c61eb916838277834128e144a6dd" },
        .{ .{ .kind = "bits", .seed = 12 }, .int16, 15, "a6e7271b92ce8247a1e92589fe17653b23a7a8bb7ea59ab63b86facf00561c15" },
        .{ .{ .kind = "uniform", .seed = 13, .lo = -1, .hi = 1 }, .float32, 9, "c0cf7502358cba5ade2fa5177a4208979444cc8d6ea99c8d65acbecae8321582" },
        .{ .{ .kind = "uniform", .seed = 14, .lo = -0.1, .hi = 0.1 }, .float16, 9, "9e7c8144bd6900bfdc81408800a4c77b9361857ec3976ae04210637dd1f77cc7" },
        .{ .{ .kind = "uniform", .seed = 15, .lo = -2, .hi = 2 }, .bfloat16, 9, "6b080ede7166f2356114342d02f7f4dc2748a99f638ce8e42edc1970ee6906c7" },
        .{ .{ .kind = "uniform", .seed = 16, .lo = 1e3, .hi = 1e5 }, .float32, 4, "8d14c13205546afac9e4fe76dd954a3eb0aea86a5b49fdba335ee50c9785a48e" },
        .{ .{ .kind = "index", .seed = 17, .hi = 4 }, .uint32, 7, "99b96e18a705b3317b58dd3060a7b821e2ccf09e2e0b20a115c9b892a7ef46eb" },
        .{ .{ .kind = "index", .seed = 18, .hi = 160 }, .int32, 7, "9e5c1c1d0e3e5585bc0d757bac84819debfa61f1169c2d3b9d8dfc0eea209113" },
        .{ .{ .kind = "range", .seed = 19, .lo = 112, .hi = 125 }, .uint8, 11, "c262ec63d66f8749a41f817211f3d5e28baaf7e71bd12b9b9cc27aa6d9fdc887" },
        .{ .{ .kind = "wave_rhs", .rows = &.{ 70, 37, 20 } }, .uint32, 127, "72ae2751e9e0066acd9e415b3ac7fa451dbf145597204f4f95a52ed16b0d40c3" },
        .{ .{ .kind = "wave_table", .slots = &.{ 3, 0, 2 }, .rows = &.{ 70, 37, 20 }, .tiles = 72 }, .int32, 80, "e7c9e15972414d0b85f223fc695913b94650858ba19f877f499ff124961b2968" },
        .{ .{ .kind = "slots16", .slots = &.{ 3, 1 } }, .int32, 16, "631b74d731b6e3635ad3bf3bb463aba6f2c678e26f9aa0fb5f5ca4b6d1d29f2f" },
        .{ .{ .kind = "values", .values = &.{ 0.3, 0.7, 1.1 } }, .float32, 3, "bcadc79e95123af764e6813ecf10b40cab57dafc8ef17f8f60118dfcf23ccbfd" },
        // decode batch 2 (dump_kernel_decode2_fixture.py --golden): bool bytes and the bf16 head weight words
        .{ .{ .kind = "range", .seed = 21, .lo = 0, .hi = 2 }, .bool_, 13, "61e3fd288705a89f08816c151a557cdd31d7a9ef95ba607c437d6bd4f66095e3" },
        .{ .{ .kind = "bf16bits", .seed = 22 }, .bfloat16, 11, "7b6c6bc9a8d38b9b98954f6abd0aee09c1693bbbe62c9bac446bde806cf9d0f1" },
        // prefill batch 2 (dump_kernel_prefill2_fixture.py --golden): the signed range (-1 .. -64 = no key)
        .{ .{ .kind = "srange", .seed = 23, .lo = -64, .hi = 2048 }, .int32, 13, "967d23bfcc0bc9205fd577bcd469258765f0c31131017d7fb326be0c5a4a325e" },
    };
    for (cases) |c| {
        const b = try generate(a, c[0], c[1], c[2]);
        defer a.free(b);
        if (!hexEql(c[3], b)) std.debug.print("generator {s} {t} differs from the dump's\n", .{ c[0].kind, c[1] });
        try testing.expect(hexEql(c[3], b));
    }
    // the registry's golden-plane stream is the same splitmix64 (state += golden before each output)
    var st: u64 = 0x5EED0001;
    for (0..4) |i| try testing.expectEqual(xk.splitmix64(&st), sm(0x5EED0001, i));
}
