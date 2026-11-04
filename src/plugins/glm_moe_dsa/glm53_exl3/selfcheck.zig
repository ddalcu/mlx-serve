//! The self-checks of GLM-5.3's EXL3 kernel registry (`glm53_exl3_kernels`), as `sdk_ext.kernels.KernelSet` runs them.
//!   - `compile` (the quant's accept): every kernel of the subset builds, binds and completes one launch of its stage
//!     chain at each registered instantiation (one pair, one prompt row, zero records); no numeric probe.
//!   - `full` (the device test): also `chain`, each stage chain's words against sushi's host decode of the same records
//!     (seeded random trellis, rin and rout), one bf16 ulp of the output's scale: the decode chain at 1 and 2 rows, the
//!     prompt chain over experts of 1 to 129 rows.

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const log = @import("sdk").log;
const xk = @import("kernels.zig");
const kset = @import("../../../sdk/weights/streamed/sdk_ext/kernel_set.zig");
const routes = @import("routes.zig");
const host_caps = @import("../../../sdk/weights/streamed/host_caps.zig");
const sushi = if (host_caps.exl3) @import("mlx_host").exl3.sushi else @compileError("glm_moe_dsa: this host exports no exl3 (host_caps): the reference must sit behind host_caps.glm");

const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Check = xk.Check;

const scaffold = kset.Checks(Kernel, Check);
pub const Result = scaffold.Result;
pub const Report = scaffold.Report;
pub const Depth = kset.CheckDepth;

/// The checks of `want`'s entries at `depth`, per kernel.
pub fn plan(reg: *const xk.Registry, want: std.EnumSet(Kernel), depth: Depth) std.EnumArray(Kernel, std.EnumSet(Check)) {
    return scaffold.plan(&reg.entries, want, depth);
}

/// The plan over `subset` at `depth` on the bound kernels, at every dims the manifest registers.
pub fn runSubset(a: Allocator, reg: *const xk.Registry, bound: *const xk.Bound, subset: []const Kernel, depth: Depth, report: *Report) !void {
    return runSubsetAt(a, reg, bound, subset, depth, reg.dims, report);
}

/// The plan over `subset` at `depth` at `dims` (both Ks): each family's chain per instantiation; each kernel of the
/// subset gets one result per check (its worst instantiation's).
pub fn runSubsetAt(a: Allocator, reg: *const xk.Registry, bound: *const xk.Bound, subset: []const Kernel, depth: Depth, dims: []const xk.Instance, report: *Report) !void {
    var want: std.EnumSet(Kernel) = .empty;
    for (subset) |k| want.insert(k);
    const checks = plan(reg, want, depth);
    var res: std.EnumArray(Kernel, std.EnumArray(Check, ?Result)) = .initFill(.initFill(null));
    for ([_]xk.Family{ .decode, .prompt }) |fam| {
        var any = false;
        for (&reg.entries) |*e| any = any or (e.family == fam and want.contains(e.kernel));
        if (!any) continue;
        for (dims) |d| for (xk.bank_ks) |k| {
            const inst: xk.Instance = .{ .k = k, .hidden = d.hidden, .mini = d.mini };
            for ([_]Check{ .compile, .chain }) |c| {
                var needed = false;
                for (&reg.entries) |*e| needed = needed or (e.family == fam and checks.get(e.kernel).contains(c));
                if (!needed) continue;
                const r = runChain(a, bound, fam, inst, c);
                for (&reg.entries) |*e| if (e.family == fam and checks.get(e.kernel).contains(c)) {
                    const prev = res.get(e.kernel).get(c);
                    if (prev == null or (prev.?.ok and !r.ok) or (prev.?.ok == r.ok and r.metric > prev.?.metric)) {
                        var x = r;
                        x.kernel = e.kernel;
                        var row = res.get(e.kernel);
                        row.set(c, x);
                        res.set(e.kernel, row);
                    }
                };
            }
        };
    }
    for (&reg.entries) |*e| {
        if (!want.contains(e.kernel)) continue;
        var it = checks.get(e.kernel).iterator();
        while (it.next()) |c| try report.results.append(a, res.get(e.kernel).get(c) orelse .{ .kernel = e.kernel, .check = c, .err = "no result" });
    }
}

/// The verdict over a finished plan: refused (error.SelfCheckFailed, `diag` naming the first failure) unless every
/// result passed and the plan ran.
pub fn judge(report: *const Report, diag: ?*xk.Diag) error{SelfCheckFailed}!void {
    for (report.results.items) |r| if (!r.ok) {
        log.err("[glm53 selfcheck] FAIL {t} {t}: {d} of {d} words, metric {e} limit {e}, {s}\n", .{ r.kernel, r.check, r.bad, r.words, r.metric, r.limit, r.err });
        break;
    };
    return scaffold.judge(report, "glm53 exl3 kernels", diag);
}

/// A defect injected between a chain and its reference (the tests' proof that a check fails on a wrong chain): `k`
/// decodes records of the other K, `rin_rout` binds each projection's rin as its rout and back (fixture dims: their
/// shapes agree), `row_order` reads the rows' x rows rotated by one.
pub const Mutation = enum { none, k, rin_rout, row_order };

fn runChain(a: Allocator, bound: *const xk.Bound, fam: xk.Family, inst: xk.Instance, c: Check) Result {
    return runChainMutated(a, bound, fam, inst, c, .none);
}

pub fn runChainMutated(a: Allocator, bound: *const xk.Bound, fam: xk.Family, inst: xk.Instance, c: Check, mutation: Mutation) Result {
    var r: Result = .{ .kernel = undefined, .check = c };
    const run = switch (fam) {
        .decode => decodeChain(a, bound, inst, c == .chain, mutation),
        .prompt => promptChain(a, bound, inst, c == .chain, mutation),
    };
    const v = run catch |e| {
        var buf: [256]u8 = undefined;
        if (mlx.takeError(&buf)) |m| log.err("[glm53 selfcheck] {t} at K {d}, {d} / {d}: {s}\n", .{ fam, inst.k, inst.hidden, inst.mini, m });
        r.err = @errorName(e);
        return r;
    };
    r.words = v.words;
    r.bad = v.bad;
    r.metric = v.metric;
    r.limit = v.limit;
    r.ok = v.bad == 0;
    return r;
}

const Verdict = struct { words: u64, bad: u64, metric: f64, limit: f64 };

/// Arrays freed together.
pub const Scope = kset.Scope;

/// A bank kind's arrays for the three projections at (K, dims), `slots` slot rows of four minis each: seeded bytes
/// (with `seed`, else zeros), kept on the host for the reference.
pub const SynthBank = struct {
    slots: usize,
    code: [3][]u16,
    rout: [3][]u16,
    rin: [3][]u16,
    bank: routes.Bank = undefined,

    /// `slots` slots of random records (`seed`) or zeros; `nan_slot`'s rin and rout NaN (a launch that reads it gives NaN).
    pub fn init(a: Allocator, sc: *Scope, inst: xk.Instance, slots: usize, seed: ?u64, nan_slot: ?usize) !SynthBank {
        var b: SynthBank = .{ .slots = slots, .code = undefined, .rout = undefined, .rin = undefined };
        var prng = std.Random.DefaultPrng.init(seed orelse 0);
        const rng = prng.random();
        const minis = slots * 4;
        var n_done: usize = 0;
        errdefer for (0..n_done) |p| {
            a.free(b.code[p]);
            a.free(b.rout[p]);
            a.free(b.rin[p]);
        };
        var projs: [3]routes.Proj = undefined;
        for (0..3) |p| {
            const in_d: usize = if (p == 2) inst.mini else inst.hidden;
            const out_d: usize = if (p == 2) inst.hidden else inst.mini;
            b.code[p] = try a.alloc(u16, minis * (in_d / 16) * (out_d / 16) * 16 * inst.k);
            b.rout[p] = try a.alloc(u16, minis * out_d);
            b.rin[p] = try a.alloc(u16, minis * in_d);
            n_done = p + 1;
            if (seed != null) {
                for (b.code[p]) |*w| w.* = rng.int(u16);
                for (b.rout[p]) |*w| w.* = @bitCast(@as(f16, @floatCast(0.5 + rng.float(f32))));
                for (b.rin[p]) |*w| w.* = @bitCast(@as(f16, @floatCast((rng.float(f32) * 2 - 1) * 0.15)));
            } else {
                @memset(b.code[p], 0);
                @memset(b.rout[p], 0);
                @memset(b.rin[p], 0);
            }
            if (nan_slot) |ns| {
                @memset(b.rout[p][ns * 4 * out_d ..][0 .. 4 * out_d], 0x7e00);
                @memset(b.rin[p][ns * 4 * in_d ..][0 .. 4 * in_d], 0x7e00);
            }
            const ci: c_int = @intCast(minis);
            projs[p] = .{
                .trellis = try sc.host(std.mem.sliceAsBytes(b.code[p]), &.{ ci, @intCast(in_d / 16), @intCast(out_d / 16), @intCast(16 * inst.k) }, .uint16),
                .svh = try sc.host(std.mem.sliceAsBytes(b.rout[p]), &.{ ci, @intCast(out_d) }, .float16),
                .suh = try sc.host(std.mem.sliceAsBytes(b.rin[p]), &.{ ci, @intCast(in_d) }, .float16),
            };
        }
        b.bank = .{ .gate = projs[0], .up = projs[1], .down = projs[2] };
        return b;
    }

    pub fn deinit(b: *SynthBank, a: Allocator) void {
        for (0..3) |p| {
            a.free(b.code[p]);
            a.free(b.rout[p]);
            a.free(b.rin[p]);
        }
    }

    /// The host reference of one routed pair: slot `slot`'s four minis through sushi's own projections (`format.project`),
    /// SwiGLU between, summed, times `score`, into `out` [hidden].
    pub fn hostPair(b: *const SynthBank, a: Allocator, inst: xk.Instance, slot: usize, x: []const f32, score: f32, out: []f32) !void {
        const hid: usize = inst.hidden;
        const mini: usize = inst.mini;
        const rate = sushi.format.Rate.fromK(inst.k);
        const buf = try a.alloc(f32, 3 * hid + 3 * mini);
        defer a.free(buf);
        const t_in = buf[0..hid];
        const inner = buf[hid .. 2 * hid];
        const down = buf[2 * hid .. 3 * hid];
        const gate = buf[3 * hid ..][0..mini];
        const up = buf[3 * hid + mini ..][0..mini];
        const act = buf[3 * hid + 2 * mini ..][0..mini];
        const ng = (hid / 16) * (mini / 16) * 16 * inst.k;
        @memset(out, 0);
        for (0..4) |r| {
            const mi = slot * 4 + r;
            sushi.format.project(x, b.code[0][mi * ng ..][0..ng], b.rin[0][mi * hid ..][0..hid], b.rout[0][mi * mini ..][0..mini], hid, mini, rate, .mcg, t_in, inner[0..mini], gate);
            sushi.format.project(x, b.code[1][mi * ng ..][0..ng], b.rin[1][mi * hid ..][0..hid], b.rout[1][mi * mini ..][0..mini], hid, mini, rate, .mcg, t_in, inner[0..mini], up);
            for (act, gate, up) |*o, gv, uv| o.* = gv / (1 + @exp(-gv)) * uv;
            sushi.format.project(act, b.code[2][mi * ng ..][0..ng], b.rin[2][mi * mini ..][0..mini], b.rout[2][mi * hid ..][0..hid], mini, hid, rate, .mcg, t_in[0..mini], inner, down);
            for (out, down) |*o, d| o.* += d;
        }
        for (out) |*o| o.* *= score;
    }
};

fn swapRinRout(b: *routes.Bank) void {
    inline for (.{ &b.gate, &b.up, &b.down }) |p| std.mem.swap(mlx.mlx_array, &p.suh, &p.svh);
}

/// One bf16 ulp at `v`'s scale.
pub fn bf16Ulp(v: f32) f32 {
    if (v == 0) return std.math.floatMin(f32);
    const e = std.math.frexp(v).exponent;
    return std.math.ldexp(@as(f32, 1.0), e - 8);
}

/// got against want: words, words off by more than one bf16 ulp of the output's scale, the worst error over that ulp.
pub fn verdictOf(got: []const f32, want: []const f32) Verdict {
    var scale: f32 = 0;
    for (want) |w| scale = @max(scale, @abs(w));
    const ulp = bf16Ulp(scale);
    var bad: u64 = 0;
    var worst: f32 = 0;
    for (got, want) |g, w| {
        const d = @abs(g - w);
        if (!(d <= ulp)) bad += 1;
        worst = @max(worst, if (std.math.isFinite(d)) d else std.math.inf(f32));
    }
    if (scale == 0) bad += 1;
    return .{ .words = got.len, .bad = bad, .metric = worst / ulp, .limit = 1 };
}

fn f32Of(sc: *Scope, s: mlx.mlx_stream, x: mlx.mlx_array, n: usize) ![]const f32 {
    var f = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&f, x, .float32, s));
    _ = try sc.keep(f);
    try mlx.check(mlx.mlx_array_eval(f));
    return (mlx.mlx_array_data_float32(f) orelse return error.MlxNoData)[0..n];
}

fn bf16Rows(sc: *Scope, s: mlx.mlx_stream, xs: []const f32, rows: usize, hid: usize) !mlx.mlx_array {
    const xf = try sc.host(std.mem.sliceAsBytes(xs), &.{ @intCast(rows), @intCast(hid) }, .float32);
    var xb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&xb, xf, .bfloat16, s));
    return sc.keep(xb);
}

/// The decode chain at `inst`: `numeric` 2 rows x top-8 over 3 bank kinds of 6 slots (rows sharing slots) against the host
/// decode, else one pair on zero records; then the device mode over the same positions, some skipped.
fn decodeChain(a: Allocator, bound: *const xk.Bound, inst: xk.Instance, numeric: bool, mutation: Mutation) !Verdict {
    const s = bound.stream;
    var sc: Scope = .{ .a = a };
    defer sc.deinit();
    const n_rows: usize = if (numeric) 2 else 1;
    const n_slots: usize = if (numeric) 6 else 1;
    var banks: [3]SynthBank = undefined;
    var nb: usize = 0;
    defer for (banks[0..nb]) |*b| b.deinit(a);
    const rec: xk.Instance = if (mutation == .k) .{ .k = 7 - inst.k, .hidden = inst.hidden, .mini = inst.mini } else inst;
    // a seventh slot past the pairs' of NaN rin and rout: the device mode's pairs of score 0 name it
    for (&banks, 0..) |*b, i| {
        b.* = try SynthBank.init(a, &sc, rec, n_slots + 1, if (numeric) 0x6c3 + i else null, if (numeric) n_slots else null);
        nb += 1;
    }
    var prng = std.Random.DefaultPrng.init(0x53d);
    const rng = prng.random();
    const n_pairs = if (numeric) n_rows * routes.top_k else 1;
    var pairs: [routes.max_pairs][3]u32 = undefined;
    if (numeric) {
        for (0..n_rows) |row| {
            var j: usize = 0;
            while (j < routes.top_k) {
                const pk: u32 = (rng.uintLessThan(u32, 3) << 24) | rng.uintLessThan(u32, @intCast(n_slots));
                var dup = false;
                for (pairs[row * routes.top_k ..][0..j]) |pr| dup = dup or pr[1] == pk;
                if (dup) continue;
                pairs[row * routes.top_k + j] = .{ @intCast(row), pk, @intCast(row * routes.top_k + j) };
                j += 1;
            }
        }
    } else pairs[0] = .{ 0, 0, 0 };
    const tabs = try decodeTables(a, pairs[0..n_pairs]);
    defer a.free(tabs.pt);
    defer a.free(tabs.gt);
    const xs = try a.alloc(f32, n_rows * inst.hidden);
    defer a.free(xs);
    for (xs) |*v| v.* = if (numeric) rng.floatNorm(f32) * 0.5 else 0;
    var scores: [routes.max_rows * routes.top_k]f32 = undefined;
    for (&scores) |*v| v.* = 0.05 + rng.float(f32) * 0.2;
    const x = try bf16Rows(&sc, s, xs, n_rows, inst.hidden);
    if (mutation == .row_order) for (0..n_pairs) |i| {
        tabs.pt[4 * i] = (tabs.pt[4 * i] + 1) % @as(u32, @intCast(n_rows));
    };
    const pt = try sc.host(std.mem.sliceAsBytes(tabs.pt), &.{@intCast(tabs.pt.len)}, .uint32);
    const gt = try sc.host(std.mem.sliceAsBytes(tabs.gt), &.{@intCast(tabs.gt.len)}, .uint32);
    const scd = try sc.host(std.mem.sliceAsBytes(scores[0 .. n_rows * routes.top_k]), &.{@intCast(n_rows * routes.top_k)}, .float32);
    var dec = routes.Decode.init(bound, inst);
    defer dec.deinit();
    var cmb = routes.Combine.init(bound, inst);
    defer cmb.deinit();
    var bk = [3]routes.Bank{ banks[0].bank, banks[1].bank, banks[2].bank };
    if (mutation == .rin_rout) for (&bk) |*b| swapRinRout(b);
    const y = try sc.keep(try dec.wave(x, pt, gt, n_pairs, tabs.n_slots, &bk, scd));
    var ct: [routes.max_rows * routes.top_k]u32 = @splat(0xffffffff);
    for (pairs[0..n_pairs], 0..) |pr, i| ct[pr[2]] = @intCast(i);
    const ctd = try sc.host(std.mem.sliceAsBytes(ct[0..@max(8, n_rows * routes.top_k)]), &.{@intCast(@max(8, n_rows * routes.top_k))}, .uint32);
    const out = try sc.keep(try cmb.launch(&.{y}, ctd, null, n_rows, true));
    // The device mode over every routed position: every 5th position skipped by its slot, every 7th by its score (its
    // slot the NaN one: a pair that read it would give NaN). The combine sums every position (the host does not know the
    // skips): a skipped pair's row must be zero.
    const n_ids = n_rows * routes.top_k;
    var dslots: [routes.max_rows * routes.top_k]u32 = @splat(0xffffffff);
    var dscores = scores;
    var served: [routes.max_rows * routes.top_k]bool = @splat(false);
    for (pairs[0..n_pairs]) |pr| dslots[pr[2]] = pr[1];
    for (0..n_ids) |q| {
        if (q % 7 == 3) {
            dscores[q] = 0;
            dslots[q] = (@as(u32, @intCast(q % 3)) << 24) | @as(u32, @intCast(n_slots));
        }
        if (q % 5 == 4) dslots[q] = 0xffffffff;
        served[q] = dslots[q] != 0xffffffff and dscores[q] != 0;
    }
    const dsd = try sc.host(std.mem.sliceAsBytes(dslots[0..n_ids]), &.{@intCast(n_ids)}, .uint32);
    const dscd = try sc.host(std.mem.sliceAsBytes(dscores[0..n_ids]), &.{@intCast(n_ids)}, .float32);
    const yd = try sc.keep(try dec.waveDevice(x, dsd, n_ids, &bk, dscd));
    var every: [routes.max_rows * routes.top_k]u32 = undefined;
    for (&every, 0..) |*c, q| c.* = @intCast(q);
    const dctd = try sc.host(std.mem.sliceAsBytes(every[0..@max(8, n_ids)]), &.{@intCast(@max(8, n_ids))}, .uint32);
    const dout = try sc.keep(try cmb.launch(&.{yd}, dctd, null, n_rows, true));
    const got = try f32Of(&sc, s, out, n_rows * inst.hidden);
    const dgot = try f32Of(&sc, s, dout, n_rows * inst.hidden);
    const yrows = try f32Of(&sc, s, yd, n_ids * inst.hidden);
    if (!numeric) return .{ .words = got.len + dgot.len, .bad = 0, .metric = 0, .limit = 0 };
    var nonzero: u64 = 0;
    for (0..n_ids) |q| if (!served[q]) {
        for (yrows[q * inst.hidden ..][0..inst.hidden]) |v| nonzero += @intFromBool(v != 0);
    };
    const xin = try f32Of(&sc, s, x, n_rows * inst.hidden);
    const want = try a.alloc(f32, n_rows * inst.hidden);
    defer a.free(want);
    @memset(want, 0);
    const tmp = try a.alloc(f32, inst.hidden);
    defer a.free(tmp);
    const dwant = try a.alloc(f32, n_rows * inst.hidden);
    defer a.free(dwant);
    @memset(dwant, 0);
    for (pairs[0..n_pairs]) |pr| {
        try banks[pr[1] >> 24].hostPair(a, rec, pr[1] & 0xffffff, xin[pr[0] * inst.hidden ..][0..inst.hidden], scores[pr[2]], tmp);
        for (want[pr[0] * inst.hidden ..][0..inst.hidden], tmp) |*w, t| w.* += t;
        if (served[pr[2]]) for (dwant[pr[0] * inst.hidden ..][0..inst.hidden], tmp) |*w, t| {
            w.* += t;
        };
    }
    const v = verdictOf(got, want);
    const dv = verdictOf(dgot, dwant);
    return .{ .words = v.words + dv.words, .bad = v.bad + dv.bad + nonzero, .metric = @max(v.metric, dv.metric), .limit = 1 };
}

/// A decode wave's tables: pt (per pair: x row, packed slot, routed position, 0) and gt (per distinct slot: packed slot,
/// members, the member pairs; a slot of more pairs than `routes.max_members` takes several entries), each at least 8
/// words (an MLX input below 8 elements binds as `constant`).
pub fn decodeTables(a: Allocator, pairs: []const [3]u32) !struct { pt: []u32, gt: []u32, n_slots: usize } {
    const mm = routes.max_members;
    const pt = try a.alloc(u32, @max(8, 4 * pairs.len));
    errdefer a.free(pt);
    @memset(pt, 0);
    for (pairs, 0..) |pr, i| {
        pt[4 * i] = pr[0];
        pt[4 * i + 1] = pr[1];
        pt[4 * i + 2] = pr[2];
    }
    const gt = try a.alloc(u32, @max(8, (2 + mm) * pairs.len));
    errdefer a.free(gt);
    @memset(gt, 0);
    var used: [routes.max_pairs]bool = @splat(false);
    var n: usize = 0;
    for (pairs, 0..) |pr, i| {
        if (used[i]) continue;
        const g = gt[(2 + mm) * n ..][0 .. 2 + mm];
        g[0] = pr[1];
        var cnt: usize = 0;
        for (pairs[i..], i..) |q, j| {
            if (used[j] or q[1] != pr[1]) continue;
            used[j] = true;
            g[2 + cnt] = @intCast(j);
            cnt += 1;
            if (cnt == mm) break;
        }
        g[1] = @intCast(cnt);
        n += 1;
    }
    return .{ .pt = pt, .gt = gt, .n_slots = n };
}

/// The prompt chain at `inst` (x as f16, then one wave): `numeric` 5 experts of 1, 17, 128, 129 and 40 rows against the
/// host decode on each expert's first and last row, else one expert of one row on zero records.
fn promptChain(a: Allocator, bound: *const xk.Bound, inst: xk.Instance, numeric: bool, mutation: Mutation) !Verdict {
    const s = bound.stream;
    var sc: Scope = .{ .a = a };
    defer sc.deinit();
    const counts_n = [_]u32{ 1, 17, 128, 129, 40 };
    const counts_c = [_]u32{1};
    const counts: []const u32 = if (numeric) &counts_n else &counts_c;
    const n_slots = counts.len;
    const rec: xk.Instance = if (mutation == .k) .{ .k = 7 - inst.k, .hidden = inst.hidden, .mini = inst.mini } else inst;
    var bank = try SynthBank.init(a, &sc, rec, n_slots, if (numeric) 0x9f1 else null, null);
    defer bank.deinit(a);
    var rows: usize = 0;
    for (counts) |c| rows += c;
    var prng = std.Random.DefaultPrng.init(0xa11);
    const rng = prng.random();
    const n_tok = rows / 3 + 1;
    const rt = try a.alloc(u32, @max(8, 2 * rows));
    defer a.free(rt);
    @memset(rt, 0);
    var slots: [routes.wave_experts]u32 = undefined;
    var i: usize = 0;
    for (counts, 0..) |c, j| {
        slots[j] = @intCast((j * 3 + 1) % n_slots);
        for (0..c) |_| {
            rt[2 * i] = rng.uintLessThan(u32, @intCast(n_tok));
            rt[2 * i + 1] = slots[j];
            i += 1;
        }
    }
    const xs = try a.alloc(f32, n_tok * inst.hidden);
    defer a.free(xs);
    for (xs) |*v| v.* = if (numeric) rng.floatNorm(f32) * 0.5 else 0;
    const x = try bf16Rows(&sc, s, xs, n_tok, inst.hidden);
    const rt_dev = try a.dupe(u32, rt);
    defer a.free(rt_dev);
    if (mutation == .row_order) for (0..rows) |q| {
        rt_dev[2 * q] = (rt[2 * q] + 1) % @as(u32, @intCast(n_tok));
    };
    const rtd = try sc.host(std.mem.sliceAsBytes(rt_dev), &.{@intCast(rt_dev.len)}, .uint32);
    const tgu = routes.waveTable(slots[0..n_slots], counts, routes.tilesGateUp(inst.mini), routes.bm_gate_up);
    const tdn = routes.waveTable(slots[0..n_slots], counts, routes.tilesDown(inst.hidden), routes.bm_down);
    const tgud = try sc.host(std.mem.sliceAsBytes(&tgu.table), &.{80}, .int32);
    const tdnd = try sc.host(std.mem.sliceAsBytes(&tdn.table), &.{80}, .int32);
    const pr = routes.Prompt.init(bound, inst);
    var bk = bank.bank;
    if (mutation == .rin_rout) swapRinRout(&bk);
    // x as the served pass gives it (`beginWide`: MLX's astype)
    var xh = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&xh, x, .float16, s));
    _ = try sc.keep(xh);
    const y = try sc.keep(try pr.wave(xh, rtd, rows, tgud, tgu.tgs, tdnd, tdn.tgs, bk));
    const got = try f32Of(&sc, s, y, rows * inst.hidden);
    if (!numeric) return .{ .words = got.len, .bad = 0, .metric = 0, .limit = 0 };
    const xin = try f32Of(&sc, s, x, n_tok * inst.hidden);
    var want: std.ArrayList(f32) = .empty;
    defer want.deinit(a);
    var gsel: std.ArrayList(f32) = .empty;
    defer gsel.deinit(a);
    const tmp = try a.alloc(f32, inst.hidden);
    defer a.free(tmp);
    var row0: usize = 0;
    for (counts, 0..) |c, j| {
        for ([_]usize{ row0, row0 + c - 1 }, 0..) |row, q| {
            if (q == 1 and c == 1) break;
            try bank.hostPair(a, rec, slots[j], xin[rt[2 * row] * inst.hidden ..][0..inst.hidden], 1.0, tmp);
            try want.appendSlice(a, tmp);
            try gsel.appendSlice(a, got[row * inst.hidden ..][0..inst.hidden]);
        }
        row0 += c;
    }
    return verdictOf(gsel.items, want.items);
}

// ── Tests (GPU: DSV41_PHASE0B_MLX) ──

const testing = std.testing;
const ks = @import("../../../sdk/weights/streamed/sdk_ext.zig").kernels.KernelSet(xk);

fn gpuSet(a: Allocator, s: mlx.mlx_stream) !*ks.Set {
    var diag: xk.Diag = .{};
    return ks.Set.init(a, .{ .device = .{ .stream = s } }, &diag) catch |e| {
        std.debug.print("kernel set refused: {s}\n", .{diag.message()});
        return e;
    };
}

test "glm53 selfcheck: the full plan on the device, every kernel at every registered dims and K: each chain within one bf16 ulp of sushi's host decode" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const set = try gpuSet(a, s);
    defer set.deinit();
    var report: Report = .{};
    defer report.deinit(a);
    var diag: xk.Diag = .{};
    set.selfCheck(a, std.enums.values(Kernel), .full, &report, &diag) catch |e| {
        std.debug.print("self-check refused: {s}\n", .{diag.message()});
        return e;
    };
    try testing.expectEqual(@as(usize, 2 * xk.n_kernels), report.results.items.len);
    for (report.results.items) |r| {
        if (r.check == .chain) std.debug.print("glm53 selfcheck {t} chain: {d} words, worst {d:.3} bf16 ulp of the output's scale\n", .{ r.kernel, r.words, r.metric });
        try testing.expect(r.ok and r.words > 0);
    }
}

/// The chain's verdict over a registry whose text `which` has `needle` replaced (re-pinned).
fn mutatedText(a: Allocator, s: mlx.mlx_stream, which: xk.TextRef, needle: []const u8, repl: []const u8, fam: xk.Family, inst: xk.Instance) !Result {
    const base = switch (which) {
        .kernel => |k| xk.embedded.sources[@backingInt(k)],
        .header => |h| xk.embedded.headers[@backingInt(h)],
    };
    if (std.mem.count(u8, base, needle) == 0) return error.NeedleMissing;
    const t = try std.mem.replaceOwned(u8, a, base, needle, repl);
    defer a.free(t);
    const z = try a.dupeSentinel(u8, t, 0);
    defer a.free(z);
    var texts: xk.Texts = undefined;
    const m = try xk.retexted(a, &xk.embedded, which, z, &texts);
    defer a.free(m);
    texts.manifest = m;
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(m, &d, .{});
    var reg = try xk.Registry.init(a, &texts, &std.fmt.bytesToHex(d, .lower), null);
    defer reg.deinit();
    var b = try reg.bind(s, null);
    defer b.deinit();
    return runChainMutated(a, &b, fam, inst, .chain, .none);
}

test "glm53 selfcheck: each chain check fails on a wrong codebook, a wrong K, a missing rank, swapped rin and rout, a wrong row order, and a device pair of score 0 computed" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const set = try gpuSet(a, s);
    defer set.deinit();
    for ([_]xk.Family{ .decode, .prompt }) |fam| for (xk.bank_ks) |k| {
        const inst: xk.Instance = .{ .k = k, .hidden = 128, .mini = 128 };
        const clean = runChainMutated(a, &set.bound, fam, inst, .chain, .none);
        try testing.expect(clean.ok and clean.words > 0);
        for ([_]Mutation{ .k, .rin_rout, .row_order }) |m| {
            // a K4 kernel over K3 records reads past them: the wrong K is a K3 kernel over K4 records
            if (m == .k and k == 4) continue;
            const r = runChainMutated(a, &set.bound, fam, inst, .chain, m);
            testing.expect(!r.ok and r.bad > 0) catch |e| {
                std.debug.print("{t} K{d} mutation {t} passed ({d} bad of {d})\n", .{ fam, k, m, r.bad, r.words });
                return e;
            };
        }
        // the codebook: mul1's multiplier in both families' common header
        const cb = try mutatedText(a, s, .{ .header = .glm53_common }, "0xCBAC1FEDu", "0x83DCD12Du", fam, inst);
        try testing.expect(!cb.ok and cb.bad > 0);
        // the rank sum: three of the four minis summed
        const rank = if (fam == .decode)
            try mutatedText(a, s, .{ .kernel = .glm53_dec_down }, "for (uint r = 0u; r < 4u; ++r) {\n            const threadgroup float* zr", "for (uint r = 0u; r < 3u; ++r) {\n            const threadgroup float* zr", fam, inst)
        else
            try mutatedText(a, s, .{ .kernel = .glm53_pf_down }, "for (int r = 0; r < 4; ++r) {", "for (int r = 0; r < 3; ++r) {", fam, inst);
        try testing.expect(!rank.ok and rank.bad > 0);
        std.debug.print("glm53 selfcheck mutations {t} K{d}: clean {d:.3} ulp; codebook {d} bad, rank sum {d} bad of {d}\n", .{ fam, k, clean.metric, cb.bad, rank.bad, rank.words });
        if (fam == .decode) {
            // the device mode's pair of score 0 through the down (its slot's NaN rin and rout read)
            const skip = try mutatedText(a, s, .{ .kernel = .glm53_dec_down }, "if (packed == 0xffffffffu || sc[pos] == 0.0f) {", "if (packed == 0xffffffffu) {", fam, inst);
            try testing.expect(!skip.ok and skip.bad > 0);
            std.debug.print("glm53 selfcheck mutations decode K{d}: score-0 skip dropped {d} bad of {d}\n", .{ k, skip.bad, skip.words });
        }
    };
}
