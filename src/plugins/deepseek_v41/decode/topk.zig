//! k4 `topk` (`~/plans/dsv41-decode-kernels-spec.md`): the index scores' top-k picks and the candidate source's blocks.
//!
//! A row of up to `route_ept` scores a thread is one threadgroup's registers. A longer one is first reduced to its
//! candidate blocks (a block's value is its largest score): a pick lies among the K + 1 best blocks, so the picks come
//! from the keys of the KBLK best blocks. The candidate source keeps KB of them for the later index layers, whose
//! scores arrive gathered to them (`cand`); a row beyond one threadgroup's blocks is cut in chunks and merged.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const kc = @import("../../../sdk/testing/kernel_contract/root.zig");
const kit = @import("../../deepseek_v4/contract_kit.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const dsv41 = @import("../deepseek_v41.zig");
const decode = @import("decode.zig");

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };
const Scope = dsv41.Scope;
const testing = std.testing;

pub const HEADER = kernelSource("plugins/deepseek_v41/kernels/dec_topk_header.metal");
pub const PICK_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_topk.metal");
pub const BLOCKS_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_topk_blocks.metal");
pub const CHUNK_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_topk_chunk.metal");
pub const MERGE_SOURCE = kernelSource("plugins/deepseek_v41/kernels/dec_topk_merge.metal");

const pick_ins = [_][*:0]const u8{ "scores", "meta", "cand" };
const scores_ins = [_][*:0]const u8{ "scores", "meta" };
const merge_ins = [_][*:0]const u8{ "scores", "meta", "ck", "cid" };
const pick_outs = [_][*:0]const u8{"picks"};
const blocks_outs = [_][*:0]const u8{ "picks", "cand" };
const chunk_outs = [_][*:0]const u8{ "ck", "cid" };

/// The most elements a threadgroup selects among: `dec_scan` counts ties and keys above in 16 bits each.
const max_elements: usize = 65535;
/// Registers per thread an instantiation holds: scores in a pick, blocks in a whole-row or chunk launch, pairs in a merge.
const max_ept: usize = 64;
/// Scores a thread holds before a plain row goes the way of its blocks.
const route_ept: usize = 16;
/// Blocks a thread holds in a whole-row launch: more when it makes the candidate blocks, whose chunks cost as much as
/// the row; the picks' alone are cheaper cut across threadgroups.
const whole_bpt_cand: usize = 16;
const whole_bpt_picks: usize = 4;
/// Blocks a chunk covers when it selects the KB best; the picks' route cuts a row in `max_chunks` chunks at most.
const chunk_blocks_cand: usize = 16384;
/// Chunks a row may have: the longest row is `max_chunks * chunk_blocks_cand * bs` scores, beyond it the caller's chain.
const max_chunks: usize = 8;

/// The model's shapes: picks a row, keys a block, blocks the candidate source keeps.
pub const Geom = struct {
    k: usize = 512,
    bs: usize = 8,
    kb: usize = 2048,

    pub fn of(m: *const dsv41.Dsv41Model) Geom {
        return .{ .k = m.cfg.dsv4_index_topk, .bs = m.cfg.dsv41_candidate_block_size, .kb = m.cfg.dsv41_candidate_topk_blocks };
    }
};

pub const Kernels = struct {
    pick: mlx.mlx_fast_metal_kernel,
    blocks: mlx.mlx_fast_metal_kernel,
    chunk: mlx.mlx_fast_metal_kernel,
    merge: mlx.mlx_fast_metal_kernel,
    /// Threads a threadgroup: what the pipelines were granted (`probe`).
    tg: c_int = 1024,
    /// Stands in for the `cand` a direct pick never reads.
    absent: A,

    pub fn build() !Kernels {
        const b = dsv41.buildKernel;
        const single = try b("dsv41_dec_topk", &pick_ins, &pick_outs, PICK_SOURCE, HEADER);
        errdefer _ = mlx.mlx_fast_metal_kernel_free(single);
        const whole = try b("dsv41_dec_topk_blocks", &scores_ins, &blocks_outs, BLOCKS_SOURCE, HEADER);
        errdefer _ = mlx.mlx_fast_metal_kernel_free(whole);
        const chunk = try b("dsv41_dec_topk_chunk", &scores_ins, &chunk_outs, CHUNK_SOURCE, HEADER);
        errdefer _ = mlx.mlx_fast_metal_kernel_free(chunk);
        const merge = try b("dsv41_dec_topk_merge", &merge_ins, &blocks_outs, MERGE_SOURCE, HEADER);
        errdefer _ = mlx.mlx_fast_metal_kernel_free(merge);
        return .{ .pick = single, .blocks = whole, .chunk = chunk, .merge = merge, .absent = mlx.mlx_array_new_data(&[_]i32{0}, &[_]c_int{1}, 1, .int32) };
    }

    pub fn deinit(k: *Kernels) void {
        for ([_]mlx.mlx_fast_metal_kernel{ k.pick, k.blocks, k.chunk, k.merge }) |x| _ = mlx.mlx_fast_metal_kernel_free(x);
        _ = mlx.mlx_array_free(k.absent);
    }
};

/// One call: the scores `[R, n]` f32 of rows at positions p0.. (a row r sees the ids below (p0 + r + 1) / ratio), or with
/// `cand` (the candidate source's blocks `[R, kb]` i32) the `[R, kb * bs]` scores gathered to those blocks.
/// `made`: this is the candidate source, which also returns its blocks when it has more than `kb`.
pub const Args = struct { scores: A, cand: A = none, p0: usize, ratio: usize, made: bool = false };

/// `picks` `[R, k]` i32: ids ascending (ties to the lower id), -1 past a row's reach and as padding. `cand` `[R, kb]`
/// i32: the candidate blocks ascending, -1 padded; `none` unless made.
pub const Out = struct { picks: A, cand: A = none };

fn pow2Ceil(x: usize) usize {
    return std.math.ceilPowerOfTwo(usize, @max(x, 1)) catch unreachable;
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

const OutSpec = struct { shape: [3]c_int = .{ 0, 0, 0 }, ndim: usize = 0, dt: mlx.mlx_dtype = .int32 };
const Targ = struct { name: [*:0]const u8, v: union(enum) { int: c_int, flag: bool } };

/// One launch as the wrapper and the kernel contract both read it.
const Plan = struct {
    /// Threads in x and y; the threadgroup is `tg` wide.
    threads: [2]usize,
    tg: usize,
    outs: [2]OutSpec,
    n_outs: usize,
    targs: [6]Targ,
    n_targs: usize,
};

fn int(name: [*:0]const u8, v: usize) Targ {
    return .{ .name = name, .v = .{ .int = ci(v) } };
}

fn shape2(a: usize, b: usize, dt: mlx.mlx_dtype) OutSpec {
    return .{ .shape = .{ ci(a), ci(b), 0 }, .ndim = 2, .dt = dt };
}

fn shape3(a: usize, b: usize, c: usize, dt: mlx.mlx_dtype) OutSpec {
    return .{ .shape = .{ ci(a), ci(b), ci(c) }, .ndim = 3, .dt = dt };
}

/// A row in registers: `n` scores, or with `gather` the slots of the candidate blocks.
fn pickPlan(g: Geom, tg: usize, rows: usize, n: usize, gather: bool) ?Plan {
    const ept = pow2Ceil(ceilDiv(n, tg));
    if (ept > max_ept or n > max_elements) return null;
    return .{
        .threads = .{ tg, rows },
        .tg = tg,
        .outs = .{ shape2(rows, g.k, .int32), .{} },
        .n_outs = 1,
        .targs = .{ int("TG", tg), int("EPT", ept), int("K", g.k), int("BS", g.bs), int("KB", g.kb), .{ .name = "GATHER", .v = .{ .flag = gather } } },
        .n_targs = 6,
    };
}

/// The blocks a whole-row launch of `kbl` best blocks covers.
fn wholeBlocks(kbl: usize, tg: usize) usize {
    return (if (kbl >= 1024) whole_bpt_cand else whole_bpt_picks) * tg;
}

/// A row of up to `wholeBlocks`: the `kbl` best blocks and the picks in one threadgroup.
fn wholePlan(g: Geom, tg: usize, rows: usize, n: usize, kbl: usize) ?Plan {
    const nblk = ceilDiv(n, g.bs);
    if (nblk > wholeBlocks(kbl, tg)) return null;
    return .{
        .threads = .{ tg, rows },
        .tg = tg,
        .outs = .{ shape2(rows, g.k, .int32), shape2(rows, kbl, .int32) },
        .n_outs = 2,
        .targs = .{ int("TG", tg), int("K", g.k), int("KBLK", kbl), int("BS", g.bs), int("BPT", pow2Ceil(ceilDiv(nblk, tg))), .{ .name = "", .v = .{ .int = 0 } } },
        .n_targs = 5,
    };
}

const Long = struct { chunk: Plan, merge: Plan, chunks: usize, per_chunk: usize };

/// A longer row: a threadgroup a chunk of blocks selects its `kbl` best, one a row merges the chunks.
fn longPlans(g: Geom, tg: usize, rows: usize, n: usize, kbl: usize) ?Long {
    const nblk = ceilDiv(n, g.bs);
    const per = if (kbl >= 1024) chunk_blocks_cand else std.math.clamp(pow2Ceil(ceilDiv(nblk, max_chunks)), tg, chunk_blocks_cand);
    const chunks = ceilDiv(nblk, per);
    // The merge holds the chunks' pairs; 16 a thread compiles to a pipeline the offline limits measure at 64 threads (17: 512).
    const held = ceilDiv(pow2Ceil(chunks) * kbl, tg);
    const ept = if (held == 16 and kbl >= 1024) 17 else held;
    if (chunks > max_chunks or ept > max_ept or per % tg != 0 or chunks * kbl > max_elements) return null;
    const none_t: Targ = .{ .name = "", .v = .{ .int = 0 } };
    return .{
        .chunks = chunks,
        .per_chunk = per,
        .chunk = .{
            .threads = .{ chunks * tg, rows },
            .tg = tg,
            .outs = .{ shape3(rows, chunks, kbl, .uint32), shape3(rows, chunks, kbl, .int32) },
            .n_outs = 2,
            .targs = .{ int("TG", tg), int("K", g.k), int("KBLK", kbl), int("BS", g.bs), int("BPT", per / tg), none_t },
            .n_targs = 5,
        },
        .merge = .{
            .threads = .{ tg, rows },
            .tg = tg,
            .outs = .{ shape2(rows, g.k, .int32), shape2(rows, kbl, .int32) },
            .n_outs = 2,
            .targs = .{ int("TG", tg), int("K", g.k), int("KBLK", kbl), int("BS", g.bs), int("EPT", ept), none_t },
            .n_targs = 5,
        },
    };
}

/// The picks (and candidates) of a decode call; null where no instantiation covers the shape.
pub fn select(sc: *Scope, m: *const dsv41.Dsv41Model, k: *const Kernels, a: Args) !?Out {
    return selectG(sc, Geom.of(m), k, a);
}

pub fn selectG(sc: *Scope, g: Geom, k: *const Kernels, a: Args) !?Out {
    if (!mlx.streamIsGpu(sc.s) or mlx.mlx_array_dtype(a.scores) != .float32) return null;
    const sh = mlx.getShape(a.scores);
    if (sh.len != 2 or sh[0] < 1 or sh[0] > decode.max_rows or sh[1] < 1 or g.k == 0 or g.k > 1024 or g.bs == 0 or g.kb == 0) return null;
    const rows: usize = @intCast(sh[0]);
    const n: usize = @intCast(sh[1]);
    const tg: usize = @intCast(k.tg);
    const meta = try keep(sc, mlx.mlx_array_new_data(&[_]i32{ @intCast(a.p0), @intCast(a.ratio) }, &[_]c_int{2}, 1, .int32));
    if (a.cand.ctx != null) {
        const csh = mlx.getShape(a.cand);
        if (csh.len != 2 or csh[0] != sh[0] or csh[1] != g.kb or n != g.kb * g.bs) return null;
        const plan = pickPlan(g, tg, rows, n, true) orelse return null;
        return .{ .picks = (try run(sc, k.pick, &.{ a.scores, meta, a.cand }, plan))[0] };
    }
    const made = a.made and ceilDiv(n, g.bs) > g.kb;
    if (!made and n <= route_ept * tg) {
        const plan = pickPlan(g, tg, rows, n, false) orelse return null;
        return .{ .picks = (try run(sc, k.pick, &.{ a.scores, meta, k.absent }, plan))[0] };
    }
    const kbl = if (made) g.kb else g.k + 1;
    var out: [2]A = undefined;
    if (wholePlan(g, tg, rows, n, kbl)) |plan| {
        out = try run(sc, k.blocks, &.{ a.scores, meta }, plan);
    } else {
        const long = longPlans(g, tg, rows, n, kbl) orelse return null;
        const pairs = try run(sc, k.chunk, &.{ a.scores, meta }, long.chunk);
        out = try run(sc, k.merge, &.{ a.scores, meta, pairs[0], pairs[1] }, long.merge);
    }
    return .{ .picks = out[0], .cand = if (made) out[1] else none };
}

/// One launch of `kern` on `ins`, the outputs' arrays kept by the scope.
fn run(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, ins: []const A, plan: Plan) ![2]A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for (plan.outs[0..plan.n_outs]) |o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &o.shape, o.ndim, o.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(plan.threads[0]), ci(plan.threads[1]), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(plan.tg), 1, 1));
    for (plan.targs[0..plan.n_targs]) |t| try mlx.check(switch (t.v) {
        .int => |v| mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, v),
        .flag => |v| mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, t.name, v),
    });
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var res: [2]A = .{ none, none };
    for (res[0..plan.n_outs], 0..) |*r, i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, ov, i));
        r.* = try keep(sc, o);
    }
    return res;
}

fn keep(sc: *Scope, x: A) !A {
    sc.items.append(sc.a, x) catch |e| {
        _ = mlx.mlx_array_free(x);
        return e;
    };
    return x;
}

/// Halves `k.tg` until the widest instantiation of every pipeline launches on this GPU (M1/M2 grant fewer than 1024).
pub fn probe(gpa: std.mem.Allocator, k: *Kernels, s: mlx.mlx_stream) !void {
    if (!mlx.streamIsGpu(s)) return;
    while (true) {
        probeOnce(gpa, k, s) catch |e| {
            if (e != error.MlxError or k.tg <= 64 or !mlx.takeErrorIf("Thread group size")) return e;
            k.tg = @divExact(k.tg, 2);
            continue;
        };
        return;
    }
}

fn probeOnce(gpa: std.mem.Allocator, k: *const Kernels, s: mlx.mlx_stream) !void {
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const g: Geom = .{};
    const tg: usize = @intCast(k.tg);
    // The widest row each pipeline serves: a direct pick, the whole-row launches, the chunks and the widest merge.
    for ([_]struct { n: usize, made: bool }{
        .{ .n = route_ept * tg, .made = false },
        .{ .n = wholeBlocks(g.k + 1, tg) * g.bs, .made = false },
        .{ .n = wholeBlocks(g.kb, tg) * g.bs, .made = true },
        .{ .n = max_chunks * chunk_blocks_cand * g.bs, .made = true },
        .{ .n = max_chunks * chunk_blocks_cand * g.bs, .made = false },
    }) |c| {
        const x = try zeros(&sc, &.{ 1, ci(c.n) }, .float32);
        const out = (try selectG(&sc, g, k, .{ .scores = x, .p0 = c.n - 1, .ratio = 1, .made = c.made })) orelse continue;
        try dsv41.evalAll(&.{out.picks});
    }
    const x = try zeros(&sc, &.{ 1, ci(g.kb * g.bs) }, .float32);
    const cand = try zeros(&sc, &.{ 1, ci(g.kb) }, .int32);
    if (try selectG(&sc, g, k, .{ .scores = x, .cand = cand, .p0 = g.kb * g.bs, .ratio = 1 })) |out| try dsv41.evalAll(&.{out.picks});
}

fn zeros(sc: *Scope, shape: []const c_int, dt: mlx.mlx_dtype) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, sc.s));
    return keep(sc, o);
}

// ── the kernel contract ──

const spec = kit.spec;
const metal = kit.metal;
const Shape = kit.Shape;
const Fill = kit.Fill;
const Sum = kit.Sum;

/// The width the launches are declared at; the wrapper probes the GPU's.
const contract_tg = 1024;
/// Keys the later index layers' call sees in the gathered case (more blocks than the candidate source keeps).
const gathered_keys = 20000;

fn geometryOf(p: Plan) metal.Geometry {
    var g = metal.Geometry{ .grid = .{ ci(p.threads[0]), ci(p.threads[1]), 1 }, .tg = .{ ci(p.tg), 1, 1 } };
    for (p.outs[0..p.n_outs]) |o| g.out(o.shape[0..o.ndim], o.dt);
    for (p.targs[0..p.n_targs]) |t| switch (t.v) {
        .int => |v| g.int(t.name, v),
        .flag => |v| g.flag(t.name, v),
    };
    // No output value an unwritten element could pass for.
    g.poison = 1.0e9;
    return g;
}

/// `limit`: the least thread count the offline tool measures for M1 and M2 over the launch's registered shapes.
fn launch(comptime name: [:0]const u8, comptime ins: []const [*:0]const u8, comptime outs: []const [*:0]const u8, comptime src: [:0]const u8, comptime tg_bytes: u32, comptime limit: u32) metal.Launch {
    return .{ .name = name, .inputs = ins, .outputs = outs, .source = src, .header = HEADER, .tg_bytes = tg_bytes, .m1 = limit, .m2 = limit, .adapts = true };
}

const hist_bytes = 4 * (2 * 256 + 2 + contract_tg / 32);
const pick_l = launch("dsv41_dec_topk", &pick_ins, &pick_outs, PICK_SOURCE, hist_bytes, 640);
const whole_l = launch("dsv41_dec_topk_blocks", &scores_ins, &blocks_outs, BLOCKS_SOURCE, hist_bytes + 4 * 2048, 512);
const chunk_l = launch("dsv41_dec_topk_chunk", &scores_ins, &chunk_outs, CHUNK_SOURCE, hist_bytes, 640);
const merge_l = launch("dsv41_dec_topk_merge", &merge_ins, &blocks_outs, MERGE_SOURCE, hist_bytes + 4 * 2048, 512);

const Mode = enum { pick, gathered, whole, chunk, merge };

/// One test type for the five launches: random rows (ties, -inf, the planted non-finite), the host reference of every
/// output, an exact check.
const Sel = struct {
    pub const Params = struct { mode: Mode, kbl: usize = 2048 };
    const g: Geom = .{};

    fn plan(p: Params, s: Shape) ?Plan {
        if (s.rows < 1 or s.rows > decode.max_rows) return null;
        const tg = contract_tg;
        return switch (p.mode) {
            .pick => if (s.n < 1 or s.n > route_ept * tg) null else pickPlan(g, tg, s.rows, s.n, false),
            .gathered => if (s.n != g.kb * g.bs) null else pickPlan(g, tg, s.rows, s.n, true),
            .whole => if (s.n <= route_ept * tg) null else wholePlan(g, tg, s.rows, s.n, p.kbl),
            .chunk, .merge => blk: {
                if (wholePlan(g, tg, s.rows, s.n, p.kbl) != null) break :blk null;
                const l = longPlans(g, tg, s.rows, s.n, p.kbl) orelse break :blk null;
                break :blk if (p.mode == .chunk) l.chunk else l.merge;
            },
        };
    }

    pub fn geometry(p: Params, s: Shape) ?metal.Geometry {
        return geometryOf(plan(p, s) orelse return null);
    }

    /// The call's key count, scores a row, and the first position: the call sees `keys` keys at p0 + rows.
    fn call(p: Params, s: Shape) struct { keys: usize, width: usize, ratio: usize, p0: usize } {
        const gather = p.mode == .gathered;
        const keys = if (gather) gathered_keys else s.n;
        const ratio: usize = if (s.rows % 2 == 0) 2 else 1;
        return .{ .keys = keys, .width = if (gather) g.kb * g.bs else s.n, .ratio = ratio, .p0 = ratio * keys -| s.rows };
    }

    pub fn fill(k: anytype, f: Fill, rng: std.Random) !void {
        const gpa = k.scratch();
        const rows = k.shape.rows;
        const c = call(k.p, k.shape);
        const x = try k.add(.float32, &.{ rows, c.width });
        const meta = try k.add(.int32, &.{2});
        meta[0] = @floatFromInt(c.p0);
        meta[1] = @floatFromInt(c.ratio);
        const row = try gpa.alloc(f32, c.keys);
        defer gpa.free(row);
        // The launch's third input: the candidate blocks, or one unread element.
        const cand: []f64 = if (k.p.mode == .gathered) try k.add(.int32, &.{ rows, g.kb }) else if (k.p.mode == .pick) try k.add(.int32, &.{1}) else undefined;
        for (0..rows) |r| {
            const lens = (c.p0 + r + 1) / c.ratio;
            fillRow(rng, r, row, lens);
            if (k.p.mode != .gathered) {
                for (x[r * c.width ..][0..c.width], row) |*d, v| d.* = v;
                continue;
            }
            // The slots of the row's candidate blocks; an absent block's slots hold a score no pick may take.
            const ids = try gpa.alloc(i32, g.kb);
            defer gpa.free(ids);
            try refCand(gpa, row, g.bs, lens, ids);
            for (ids, 0..) |b, bi| for (0..g.bs) |j| {
                const key: usize = if (b >= 0) @as(usize, @intCast(b)) * g.bs + j else row.len;
                x[(r * g.kb + bi) * g.bs + j] = if (b < 0) 1e9 else if (key < row.len) row[key] else ninf;
            };
            for (cand[r * g.kb ..][0..g.kb], ids) |*d, b| d.* = @floatFromInt(b);
        }
        if (f == .nonfinite) kit.plantRows(x, rows, c.width);
        if (k.p.mode != .merge) return;
        // The chunks' pairs the merge reads: the chunk reference of the (planted) rows.
        const l = longPlans(g, contract_tg, rows, c.width, k.p.kbl).?;
        const ck = try k.add(.uint32, &.{ rows, l.chunks, k.p.kbl });
        const cid = try k.add(.int32, &.{ rows, l.chunks, k.p.kbl });
        const ks = try gpa.alloc(u32, k.p.kbl);
        defer gpa.free(ks);
        const ids = try gpa.alloc(i32, k.p.kbl);
        defer gpa.free(ids);
        for (0..rows) |r| {
            for (row, x[r * c.width ..][0..c.width]) |*d, v| d.* = @floatCast(v);
            for (0..l.chunks) |ch| {
                try refChunk(gpa, row, g.bs, (c.p0 + r + 1) / c.ratio, l.per_chunk, ch, ks, ids);
                for (ks, ids, 0..) |q, id, i| {
                    ck[(r * l.chunks + ch) * k.p.kbl + i] = @floatFromInt(q);
                    cid[(r * l.chunks + ch) * k.p.kbl + i] = @floatFromInt(id);
                }
            }
        }
    }

    pub fn count(p: Params, s: Shape) usize {
        return switch (p.mode) {
            .pick, .gathered => s.rows * g.k,
            .whole, .merge => s.rows * (g.k + p.kbl),
            .chunk => 2 * s.rows * longPlans(g, contract_tg, s.rows, s.n, p.kbl).?.chunks * p.kbl,
        };
    }

    /// Every output in order as i32 (the chunks' keys bit-cast), from the host reference.
    fn expected(k: anytype, want: []i32) !void {
        const gpa = k.scratch();
        const rows = k.shape.rows;
        const c = call(k.p, k.shape);
        const meta = k.in(1);
        const p0: usize = @intFromFloat(meta[0]);
        const ratio: usize = @intFromFloat(meta[1]);
        const row = try gpa.alloc(f32, c.width);
        defer gpa.free(row);
        for (0..rows) |r| {
            const lens = (p0 + r + 1) / ratio;
            for (row, k.in(0)[r * c.width ..][0..c.width]) |*d, v| d.* = @floatCast(v);
            switch (k.p.mode) {
                .pick => try refPicks(gpa, row, g.k, lens, want[r * g.k ..][0..g.k]),
                .gathered => try refGathered(gpa, row, k.in(2)[r * g.kb ..][0..g.kb], g.bs, g.k, lens, want[r * g.k ..][0..g.k]),
                .whole, .merge => {
                    try refPicks(gpa, row, g.k, lens, want[r * g.k ..][0..g.k]);
                    try refCand(gpa, row, g.bs, lens, want[rows * g.k + r * k.p.kbl ..][0..k.p.kbl]);
                },
                .chunk => {
                    const l = longPlans(g, contract_tg, rows, c.width, k.p.kbl).?;
                    const ks = try gpa.alloc(u32, k.p.kbl);
                    defer gpa.free(ks);
                    const half = rows * l.chunks * k.p.kbl;
                    for (0..l.chunks) |ch| {
                        const at = (r * l.chunks + ch) * k.p.kbl;
                        try refChunk(gpa, row, g.bs, lens, l.per_chunk, ch, ks, want[half + at ..][0..k.p.kbl]);
                        for (ks, 0..) |q, i| want[at + i] = @bitCast(q);
                    }
                },
            }
        }
    }

    pub fn compute(k: anytype, out: []f64, mag: []f64, _: Sum) !void {
        const want = try k.scratch().alloc(i32, out.len);
        defer k.scratch().free(want);
        try expected(k, want);
        for (out, want) |*o, w| o.* = @floatFromInt(w);
        @memset(mag, 0);
    }

    /// Exact: every pick, candidate and pair is the reference's, at every gate.
    pub fn check(k: anytype, _: spec.Gate) !?[]const u8 {
        const want = try k.scratch().alloc(i32, count(k.p, k.shape));
        defer k.scratch().free(want);
        try expected(k, want);
        const got = k.rig.outputBytes();
        for (want, 0..) |w, i| {
            const v = std.mem.readInt(i32, got[4 * i ..][0..4], .little);
            if (v != w) return std.fmt.bufPrint(&k.msg, "output {d} is {d} where the reference gives {d}", .{ i, v, w }) catch "output";
        }
        return null;
    }
};

fn depthOne(_: Shape) usize {
    return 1;
}

pub fn register(set: *kc.Set) !void {
    const nan_rule: spec.Nan = .{ .stated = "a NaN takes its order key's place (positive NaN above +inf, negative below -inf), as the reference's selection does on the keys" };
    const rows_edge: spec.Tile = .{ .size = decode.max_rows };
    try kit.add(set, Sel, &pick_l, .{ .mode = .pick }, .{
        .name = "dsv41 dec topk (picks, row in registers)",
        .accum = .none,
        .depth = depthOne,
        .out = .u32,
        .nan = nan_rule,
        .tiles = .{ rows_edge, .{ .size = route_ept * contract_tg }, .{} },
        .served = &.{ .{ .rows = 6, .n = 4096, .k = 1 }, .{ .rows = 1, .n = 3, .k = 1 }, .{ .rows = 2, .n = route_ept * contract_tg, .k = 1 } },
    });
    try kit.add(set, Sel, &pick_l, .{ .mode = .gathered }, .{
        .name = "dsv41 dec topk (picks, gathered to the candidate blocks)",
        .accum = .none,
        .depth = depthOne,
        .out = .u32,
        .nan = nan_rule,
        .tiles = .{ rows_edge, .{}, .{} },
        .served = &.{ .{ .rows = 6, .n = 16384, .k = 1 }, .{ .rows = 1, .n = 16384, .k = 1 } },
    });
    inline for (.{ 2048, 513 }) |kbl| {
        const whole_n = comptime wholeBlocks(kbl, contract_tg) * 8;
        try kit.add(set, Sel, &whole_l, .{ .mode = .whole, .kbl = kbl }, .{
            .name = std.fmt.comptimePrint("dsv41 dec topk (blocks and picks of a row, {d} blocks)", .{kbl}),
            .accum = .none,
            .depth = depthOne,
            .out = .u32,
            .nan = nan_rule,
            .tiles = .{ rows_edge, .{ .size = whole_n, .min = route_ept * contract_tg + 1 }, .{} },
            .served = &.{ .{ .rows = 2, .n = 20000, .k = 1 }, .{ .rows = 6, .n = whole_n, .k = 1 } },
        });
        const long_max = max_chunks * chunk_blocks_cand * 8;
        const long_tiles: [3]spec.Tile = .{ rows_edge, .{ .size = long_max, .min = whole_n + 1 }, .{} };
        // One width a chunk size (the picks' chunks double with the row), the merge's counts in between.
        const ladder = [_]Shape{ .{ .rows = 1, .n = 262144, .k = 1 }, .{ .rows = 2, .n = 524288, .k = 1 }, .{ .rows = 3, .n = long_max, .k = 1 } };
        const long_served: []const Shape = &(if (kbl >= 1024)
            [_]Shape{.{ .rows = 1, .n = whole_n + 1, .k = 1 }} ++ ladder
        else
            [_]Shape{ .{ .rows = 1, .n = whole_n + 1, .k = 1 }, .{ .rows = 1, .n = 65537, .k = 1 }, .{ .rows = 1, .n = 131073, .k = 1 } } ++ ladder);
        try kit.add(set, Sel, &chunk_l, .{ .mode = .chunk, .kbl = kbl }, .{
            .name = std.fmt.comptimePrint("dsv41 dec topk (a chunk's best blocks, {d} blocks)", .{kbl}),
            .accum = .none,
            .depth = depthOne,
            .out = .u32,
            .nan = nan_rule,
            .tiles = long_tiles,
            .served = long_served,
        });
        try kit.add(set, Sel, &merge_l, .{ .mode = .merge, .kbl = kbl }, .{
            .name = std.fmt.comptimePrint("dsv41 dec topk (the chunks' merge, {d} blocks)", .{kbl}),
            .accum = .none,
            .depth = depthOne,
            .out = .u32,
            .nan = nan_rule,
            .tiles = long_tiles,
            .served = long_served,
        });
    }
}

// ── tests ──

fn orderKey(f: f32) u32 {
    const b: u32 = @bitCast(f);
    return if (b & 0x80000000 != 0) ~b else b | 0x80000000;
}

const ninf = -std.math.inf(f32);

/// The `k` largest of `row` by order key (ties to the lower index) as ascending ids, -1 from `lens` on and as padding.
fn refPicks(gpa: std.mem.Allocator, row: []const f32, k: usize, lens: usize, out: []i32) !void {
    const keys = try gpa.alloc(u32, row.len);
    defer gpa.free(keys);
    for (keys, row) |*q, v| q.* = orderKey(v);
    const take = try gpa.alloc(u32, k);
    defer gpa.free(take);
    const got = try bestOf(gpa, keys, 0, keys.len, take);
    @memset(out, -1);
    for (take[0..got], 0..) |id, i| out[i] = if (id < lens) @intCast(id) else -1;
}

/// The picks among the slots of the candidate blocks `cand` (-1: absent, its slots never picked): slot s holds key
/// cand[s / bs] * bs + s % bs, and slot order is key order.
fn refGathered(gpa: std.mem.Allocator, slots: []const f32, cand: []const f64, bs: usize, k: usize, lens: usize, out: []i32) !void {
    const keys = try gpa.alloc(u32, slots.len);
    defer gpa.free(keys);
    for (keys, slots, 0..) |*q, v, s| q.* = if (cand[s / bs] >= 0) orderKey(v) else 0;
    const take = try gpa.alloc(u32, k);
    defer gpa.free(take);
    const got = try bestOf(gpa, keys, 0, keys.len, take);
    @memset(out, -1);
    for (take[0..got], 0..) |s, i| {
        const b: i64 = @intFromFloat(cand[s / bs]);
        const id = b * @as(i64, @intCast(bs)) + @as(i64, @intCast(s % bs));
        out[i] = if (b >= 0 and id < lens) @intCast(id) else -1;
    }
}

/// The order key of each block of `row`: its largest key (-inf past the row), the newest group's block pinned to +inf.
fn blockKeys(gpa: std.mem.Allocator, row: []const f32, bs: usize, lens: usize) ![]u32 {
    const keys = try gpa.alloc(u32, ceilDiv(row.len, bs));
    for (keys, 0..) |*q, b| {
        var m: u32 = 0;
        for (0..bs) |j| m = @max(m, if (b * bs + j < row.len) orderKey(row[b * bs + j]) else orderKey(ninf));
        q.* = m;
    }
    if (lens > 0) keys[(lens - 1) / bs] = orderKey(std.math.inf(f32));
    return keys;
}

const By = struct {
    keys: []const u32,
    fn before(c: @This(), a: u32, b: u32) bool {
        return c.keys[a] > c.keys[b] or (c.keys[a] == c.keys[b] and a < b);
    }
};

/// The `out.len` best of `keys[lo..hi]` by key (ties to the lower index) as ascending indices.
fn bestOf(gpa: std.mem.Allocator, keys: []const u32, lo: usize, hi: usize, out: []u32) !usize {
    const ids = try gpa.alloc(u32, hi - lo);
    defer gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast(lo + i);
    std.mem.sort(u32, ids, By{ .keys = keys }, By.before);
    const take = ids[0..@min(out.len, ids.len)];
    std.mem.sort(u32, take, {}, std.sort.asc(u32));
    @memcpy(out[0..take.len], take);
    return take.len;
}

/// The candidate source's blocks of one row: ascending ids above -inf, -1 padded to `out.len` (= kbl).
fn refCand(gpa: std.mem.Allocator, row: []const f32, bs: usize, lens: usize, out: []i32) !void {
    const keys = try blockKeys(gpa, row, bs, lens);
    defer gpa.free(keys);
    const take = try gpa.alloc(u32, out.len);
    defer gpa.free(take);
    const got = try bestOf(gpa, keys, 0, keys.len, take);
    @memset(out, -1);
    var o: usize = 0;
    for (take[0..got]) |b| if (keys[b] > orderKey(ninf)) {
        out[o] = @intCast(b);
        o += 1;
    };
}

/// Chunk `c` (`per` blocks) of a row: its `kbl` best block keys with their ids in id order, (0, -1) padded.
fn refChunk(gpa: std.mem.Allocator, row: []const f32, bs: usize, lens: usize, per: usize, c: usize, ks: []u32, ids: []i32) !void {
    const keys = try blockKeys(gpa, row, bs, lens);
    defer gpa.free(keys);
    const take = try gpa.alloc(u32, ks.len);
    defer gpa.free(take);
    const lo = @min(c * per, keys.len);
    const got = try bestOf(gpa, keys, lo, @min(lo + per, keys.len), take);
    @memset(ks, 0);
    @memset(ids, -1);
    for (take[0..got], 0..) |b, i| {
        ks[i] = keys[b];
        ids[i] = @intCast(b);
    }
}

/// A row of scores of `kind`, -inf from `lens` on.
fn fillRow(rng: std.Random, kind: usize, row: []f32, lens: usize) void {
    for (row, 0..) |*v, i| v.* = switch (kind % 6) {
        // Few distinct values (ties at every rank) and scattered -inf, as the indexer's scores carry.
        0 => if (rng.uintLessThan(u32, 8) == 0) ninf else @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -20, 20))) * 0.25,
        1 => if (rng.uintLessThan(u32, 16) == 0) ninf else rng.floatNorm(f32),
        2 => 1.0,
        3 => @as(f32, @floatFromInt(i)) * 1e-3,
        4 => -@as(f32, @floatFromInt(i)) * 1e-3,
        else => if (i / 997 % 3 == 0) ninf else @as(f32, @floatFromInt(rng.intRangeAtMost(i32, 0, 3))),
    };
    for (row[@min(lens, row.len)..]) |*v| v.* = ninf;
}

fn upload(sc: *Scope, data: anytype, shape: []const c_int) !A {
    const dt: mlx.mlx_dtype = if (@typeInfo(@TypeOf(data)).pointer.child == f32) .float32 else .int32;
    return keep(sc, mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), dt));
}

const Case = struct { rows: usize, n: usize, ratio: usize };

/// p0 so that the call sees `n` keys: (p0 + rows) / ratio.
fn p0Of(c: Case) usize {
    return c.ratio * c.n - c.rows;
}

fn lensOf(c: Case, r: usize) usize {
    return (p0Of(c) + r + 1) / c.ratio;
}

fn expectRows(label: []const u8, got: []const i32, want: []const i32, width: usize) !void {
    try testing.expectEqual(want.len, got.len);
    for (0..want.len / width) |r| {
        const g = got[r * width ..][0..width];
        const w = want[r * width ..][0..width];
        if (!std.mem.eql(i32, g, w)) {
            const at = std.mem.indexOfDiff(i32, g, w).?;
            std.debug.print("{s}: row {d} differs from index {d}: got {d}, want {d}\n", .{ label, r, at, g[at], w[at] });
            return error.PicksDiffer;
        }
    }
}

const widths = [_]c_int{ 1024, 512, 256 };

test "dsv41 dec topk: picks and candidate blocks equal the reference on scores with ties and -inf, at every width" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var ks = try Kernels.build();
    defer ks.deinit();
    const g: Geom = .{};
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    const cases = [_]Case{
        .{ .rows = 1, .n = 3, .ratio = 1 },      .{ .rows = 2, .n = 100, .ratio = 2 },    .{ .rows = 3, .n = 511, .ratio = 1 },
        .{ .rows = 6, .n = 512, .ratio = 2 },    .{ .rows = 4, .n = 513, .ratio = 1 },    .{ .rows = 6, .n = 4100, .ratio = 1 },
        .{ .rows = 6, .n = 16384, .ratio = 2 },  .{ .rows = 1, .n = 16385, .ratio = 1 },  .{ .rows = 6, .n = 20000, .ratio = 2 },
        .{ .rows = 3, .n = 70001, .ratio = 1 },  .{ .rows = 1, .n = 131072, .ratio = 1 }, .{ .rows = 6, .n = 131073, .ratio = 2 },
        .{ .rows = 2, .n = 262151, .ratio = 1 }, .{ .rows = 6, .n = 300001, .ratio = 1 }, .{ .rows = 1, .n = 1048576, .ratio = 1 },
    };
    for (cases) |c| {
        const scores = try gpa.alloc(f32, c.rows * c.n);
        defer gpa.free(scores);
        const want = try gpa.alloc(i32, c.rows * g.k);
        defer gpa.free(want);
        const want_cand = try gpa.alloc(i32, c.rows * g.kb);
        defer gpa.free(want_cand);
        for (0..c.rows) |r| {
            fillRow(rng, r, scores[r * c.n ..][0..c.n], lensOf(c, r));
            try refPicks(gpa, scores[r * c.n ..][0..c.n], g.k, lensOf(c, r), want[r * g.k ..][0..g.k]);
            if (ceilDiv(c.n, g.bs) > g.kb) try refCand(gpa, scores[r * c.n ..][0..c.n], g.bs, lensOf(c, r), want_cand[r * g.kb ..][0..g.kb]);
        }
        for (widths) |tg| {
            ks.tg = tg;
            for ([_]bool{ false, true }) |made| {
                var sc = Scope.init(gpa, mlx.gpuStream());
                defer sc.deinit();
                const x = try upload(&sc, scores, &.{ ci(c.rows), ci(c.n) });
                const out = (try selectG(&sc, g, &ks, .{ .scores = x, .p0 = p0Of(c), .ratio = c.ratio, .made = made })) orelse {
                    std.debug.print("declined: {d} rows, n {d}, width {d}\n", .{ c.rows, c.n, tg });
                    try testing.expect(tg < 512 and c.n > 1 << 17);
                    continue;
                };
                const label = try std.fmt.allocPrint(gpa, "{d} rows, n {d}, width {d}, made {}", .{ c.rows, c.n, tg, made });
                defer gpa.free(label);
                const picks = try dsv41.hostI32(out.picks);
                defer gpa.free(picks);
                try expectRows(label, picks, want, g.k);
                if (made and ceilDiv(c.n, g.bs) > g.kb) {
                    const cand = try dsv41.hostI32(out.cand);
                    defer gpa.free(cand);
                    try expectRows(label, cand, want_cand, g.kb);
                } else try testing.expect(out.cand.ctx == null);
            }
        }
    }
}

test "dsv41 dec topk: gathered picks equal the picks of the full row masked to the candidate blocks" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var ks = try Kernels.build();
    defer ks.deinit();
    const g: Geom = .{};
    var prng = std.Random.DefaultPrng.init(12);
    const rng = prng.random();
    for ([_]Case{ .{ .rows = 3, .n = 20000, .ratio = 1 }, .{ .rows = 6, .n = 70001, .ratio = 2 }, .{ .rows = 1, .n = 300001, .ratio = 1 } }) |c| {
        const full = try gpa.alloc(f32, c.rows * c.n);
        defer gpa.free(full);
        const cand = try gpa.alloc(i32, c.rows * g.kb);
        defer gpa.free(cand);
        const slots = try gpa.alloc(f32, c.rows * g.kb * g.bs);
        defer gpa.free(slots);
        const want = try gpa.alloc(i32, c.rows * g.k);
        defer gpa.free(want);
        for (0..c.rows) |r| {
            const row = full[r * c.n ..][0..c.n];
            fillRow(rng, r + 1, row, lensOf(c, r));
            try refCand(gpa, row, g.bs, lensOf(c, r), cand[r * g.kb ..][0..g.kb]);
            // A slot of an absent block holds a score no pick may take.
            const masked = try gpa.alloc(f32, c.n);
            defer gpa.free(masked);
            @memset(masked, ninf);
            for (cand[r * g.kb ..][0..g.kb], 0..) |b, bi| for (0..g.bs) |j| {
                const key: usize = if (b >= 0) @as(usize, @intCast(b)) * g.bs + j else c.n;
                slots[(r * g.kb + bi) * g.bs + j] = if (b < 0) 1e9 else if (key < c.n) row[key] else ninf;
                if (b >= 0 and key < c.n) masked[key] = row[key];
            };
            try refPicks(gpa, masked, g.k, lensOf(c, r), want[r * g.k ..][0..g.k]);
        }
        for (widths) |tg| {
            ks.tg = tg;
            var sc = Scope.init(gpa, mlx.gpuStream());
            defer sc.deinit();
            const x = try upload(&sc, slots, &.{ ci(c.rows), ci(g.kb * g.bs) });
            const cd = try upload(&sc, cand, &.{ ci(c.rows), ci(g.kb) });
            const out = (try selectG(&sc, g, &ks, .{ .scores = x, .cand = cd, .p0 = p0Of(c), .ratio = c.ratio })) orelse return error.Declined;
            const picks = try dsv41.hostI32(out.picks);
            defer gpa.free(picks);
            try expectRows("gathered", picks, want, g.k);
        }
    }
}

// ── today's chain, the replaced one (`dsv41.indexer`'s steps after the scores), and the ubench ──

fn mop(sc: *Scope, comptime f: anytype, a: A, b: A) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(f(&o, a, b, sc.s));
    return keep(sc, o);
}

fn mop1(sc: *Scope, comptime f: anytype, a: A) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(f(&o, a, sc.s));
    return keep(sc, o);
}

fn lensArr(sc: *Scope, rows: usize, p0: usize, ratio: usize) !struct { arr: A, first: usize } {
    const h = try sc.a.alloc(i32, rows);
    defer sc.a.free(h);
    for (h, 0..) |*v, l| v.* = @intCast((p0 + l + 1) / ratio);
    return .{ .arr = try keep(sc, mlx.mlx_array_new_data(h.ptr, &[_]c_int{ ci(rows), 1 }, 2, .int32)), .first = @intCast(h[0]) };
}

/// The picks as `indexer` makes them: `topkRows` of the scores, then -1 past each row's reach.
fn todayPicks(sc: *Scope, m: *const dsv41.Dsv41Model, scores: A, p0: usize, ratio: usize) !A {
    const sh = mlx.getShape(scores);
    const lens = try lensArr(sc, @intCast(sh[0]), p0, ratio);
    const idx = try dsv41.topkRows(sc, m, scores, @min(512, @as(usize, @intCast(sh[1]))));
    if (lens.first == sh[1]) return idx;
    return mop3(sc, try mop(sc, mlx.mlx_less, idx, lens.arr), idx, try keep(sc, mlx.mlx_array_new_int(-1)));
}

fn mop3(sc: *Scope, c: A, x: A, y: A) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_where(&o, c, x, y, sc.s));
    return keep(sc, o);
}

/// The candidate source's mask `[R, n_blocks]` bool, `indexer`'s block steps.
fn todayCand(sc: *Scope, m: *const dsv41.Dsv41Model, scores: A, p0: usize, ratio: usize, bs: usize, kb: usize) !A {
    const sh = mlx.getShape(scores);
    const rows: usize = @intCast(sh[0]);
    const n: usize = @intCast(sh[1]);
    const nblk = ceilDiv(n, bs);
    const lens = try lensArr(sc, rows, p0, ratio);
    const ninf_a = try keep(sc, mlx.mlx_array_new_float(ninf));
    var padded = scores;
    if (nblk * bs > n) {
        const pad = try zeros(sc, &.{ ci(rows), ci(nblk * bs - n) }, .float32);
        padded = try concat2(sc, scores, try mop(sc, mlx.mlx_add, pad, ninf_a));
    }
    var by_block = [_]c_int{ ci(rows), ci(nblk), ci(bs) };
    var r3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r3, padded, &by_block, 3, sc.s));
    _ = try keep(sc, r3);
    var mx = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_max_axis(&mx, r3, -1, false, sc.s));
    var blocks_a = try keep(sc, mx);
    const lf = try mop1(sc, mlx.mlx_floor, try mop(sc, mlx.mlx_divide, try astype(sc, try mop(sc, mlx.mlx_subtract, lens.arr, try keep(sc, mlx.mlx_array_new_int(1))), .float32), try keep(sc, mlx.mlx_array_new_float(@floatFromInt(bs)))));
    var ar = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(nblk), 1, .int32, sc.s));
    _ = try keep(sc, ar);
    var bidx = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&bidx, ar, &[_]c_int{ 1, ci(nblk) }, 2, sc.s));
    _ = try keep(sc, bidx);
    blocks_a = try mop3(sc, try mop(sc, mlx.mlx_equal, try astype(sc, bidx, .float32), lf), try keep(sc, mlx.mlx_array_new_float(std.math.inf(f32))), blocks_a);
    const chosen = try dsv41.topkRows(sc, m, blocks_a, kb);
    var tk = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_along_axis(&tk, blocks_a, chosen, -1, sc.s));
    _ = try keep(sc, tk);
    const reach = try mop(sc, mlx.mlx_greater, tk, ninf_a);
    var out = mlx.mlx_array_new();
    const z = try zeros(sc, &.{ ci(rows), ci(nblk) }, .bool_);
    try mlx.check(mlx.mlx_put_along_axis(&out, z, chosen, reach, -1, sc.s));
    return keep(sc, out);
}

/// The picks of a later index layer: scores outside the candidate blocks' mask `[R, n_blocks]` are -inf.
fn todayMasked(sc: *Scope, m: *const dsv41.Dsv41Model, scores: A, mask: A, p0: usize, ratio: usize, bs: usize) !A {
    const n: usize = @intCast(mlx.getShape(scores)[1]);
    var rep = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_repeat_axis(&rep, mask, ci(bs), -1, sc.s));
    _ = try keep(sc, rep);
    const rows: usize = @intCast(mlx.getShape(scores)[0]);
    var cut = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&cut, rep, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(rows), ci(n) }, 2, &[_]c_int{ 1, 1 }, 2, sc.s));
    _ = try keep(sc, cut);
    const masked = try mop3(sc, cut, scores, try keep(sc, mlx.mlx_array_new_float(ninf)));
    return todayPicks(sc, m, masked, p0, ratio);
}

fn concat2(sc: *Scope, a: A, b: A) !A {
    const v = mlx.mlx_vector_array_new_data(&[_]A{ a, b }, 2);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, v, -1, sc.s));
    return keep(sc, o);
}

fn astype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, sc.s));
    return keep(sc, o);
}

/// A model with only today's top-k kernel set, enough for `topkRows`.
fn todayModel() !dsv41.Dsv41Model {
    var m: dsv41.Dsv41Model = undefined;
    m.topk_kernel = try dsv41.buildKernel("dsv41_topk", &.{"x"}, &.{"out"}, dsv41.TOPK_SOURCE, dsv41.TOPK_HEADER);
    m.topk_tg = 1024;
    return m;
}

test "dsv41 dec topk: picks and candidate blocks equal today's topkRows chain" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var ks = try Kernels.build();
    defer ks.deinit();
    var m = try todayModel();
    defer _ = mlx.mlx_fast_metal_kernel_free(m.topk_kernel);
    const g: Geom = .{};
    var prng = std.Random.DefaultPrng.init(21);
    const rng = prng.random();
    for ([_]Case{ .{ .rows = 6, .n = 1000, .ratio = 2 }, .{ .rows = 6, .n = 20000, .ratio = 1 }, .{ .rows = 1, .n = 70001, .ratio = 2 }, .{ .rows = 6, .n = 300001, .ratio = 1 } }) |c| {
        const scores = try gpa.alloc(f32, c.rows * c.n);
        defer gpa.free(scores);
        for (0..c.rows) |r| fillRow(rng, r, scores[r * c.n ..][0..c.n], lensOf(c, r));
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const x = try upload(&sc, scores, &.{ ci(c.rows), ci(c.n) });
        const out = (try selectG(&sc, g, &ks, .{ .scores = x, .p0 = p0Of(c), .ratio = c.ratio, .made = true })) orelse return error.Declined;
        // Today's picks are k = min(512, n) wide; ours are 512 wide, -1 padded.
        const want = try dsv41.hostI32(try todayPicks(&sc, &m, x, p0Of(c), c.ratio));
        defer gpa.free(want);
        const got = try dsv41.hostI32(out.picks);
        defer gpa.free(got);
        const w = want.len / c.rows;
        for (0..c.rows) |r| {
            try testing.expectEqualSlices(i32, want[r * w ..][0..w], got[r * g.k ..][0..w]);
            for (got[r * g.k + w ..][0 .. g.k - w]) |v| try testing.expectEqual(@as(i32, -1), v);
        }
        if (ceilDiv(c.n, g.bs) <= g.kb) continue;
        const mask = try dsv41.hostI32(try astype(&sc, try todayCand(&sc, &m, x, p0Of(c), c.ratio, g.bs, g.kb), .int32));
        defer gpa.free(mask);
        const cand = try dsv41.hostI32(out.cand);
        defer gpa.free(cand);
        const nblk = ceilDiv(c.n, g.bs);
        for (0..c.rows) |r| {
            const row = mask[r * nblk ..][0..nblk];
            var at: usize = 0;
            for (cand[r * g.kb ..][0..g.kb]) |b| {
                if (b < 0) break;
                try testing.expectEqual(@as(i32, 1), row[@intCast(b)]);
                at += 1;
            }
            var ones: usize = 0;
            for (row) |v| ones += @intCast(v);
            try testing.expectEqual(ones, at);
        }
    }
}

const touch_src = "const uint i = thread_position_in_grid.x; out[i] = x[i] + float(as_type<uint>(prev[0]) & 0u);";

/// One chain of the ubench: every step makes the scores of a copy from the previous step's output (a streaming kernel,
/// the same for every arm, so each step depends on the last), then runs the arm.
const Chain = struct {
    sc: *Scope,
    ks: *const Kernels,
    mdl: *const dsv41.Dsv41Model,
    touch: mlx.mlx_fast_metal_kernel,
    g: Geom,
    scores: []const A,
    cands: []const A,
    masks: []const A,
    rows: usize,
    p0: usize,
    ratio: usize,
    arm: enum { head, mine, today },
    kind: enum { plain, made, gathered },
    prev: A,

    pub fn step(self: *Chain, i: usize) !A {
        const sc = self.sc;
        if (i == 0) {
            // Each chain starts clean: the last one's arrays are done.
            sc.deinit();
            sc.* = Scope.init(sc.a, sc.s);
            self.prev = self.scores[0];
        }
        const x = self.scores[i % self.scores.len];
        const sh = mlx.getShape(x);
        const s2 = (try run(sc, self.touch, &.{ x, self.prev }, .{ .threads = .{ mlx.mlx_array_size(x), 1 }, .tg = 256, .outs = .{ .{ .shape = .{ sh[0], sh[1], 0 }, .ndim = 2, .dt = .float32 }, .{} }, .n_outs = 1, .targs = undefined, .n_targs = 0 }))[0];
        const out: A = switch (self.arm) {
            .head => s2,
            .mine => (try selectG(sc, self.g, self.ks, .{ .scores = s2, .cand = if (self.kind == .gathered) self.cands[i % self.cands.len] else none, .p0 = self.p0, .ratio = self.ratio, .made = self.kind == .made })).?.picks,
            .today => switch (self.kind) {
                .plain => try todayPicks(sc, self.mdl, s2, self.p0, self.ratio),
                .made => blk: {
                    const mask = try todayCand(sc, self.mdl, s2, self.p0, self.ratio, self.g.bs, self.g.kb);
                    const picks = try todayPicks(sc, self.mdl, s2, self.p0, self.ratio);
                    // The layer needs both: evaluating the picks evaluates the mask.
                    const ins = mlx.mlx_vector_array_new_data(&[_]A{picks}, 1);
                    defer _ = mlx.mlx_vector_array_free(ins);
                    const deps = mlx.mlx_vector_array_new_data(&[_]A{mask}, 1);
                    defer _ = mlx.mlx_vector_array_free(deps);
                    var res = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(res);
                    try mlx.check(mlx.mlx_depends(&res, ins, deps));
                    var o = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_vector_array_get(&o, res, 0));
                    break :blk try keep(sc, o);
                },
                .gathered => try todayMasked(sc, self.mdl, s2, self.masks[i % self.masks.len], self.p0, self.ratio, self.g.bs),
            },
        };
        self.prev = out;
        if (i % 4 == 3) {
            const v = mlx.mlx_vector_array_new_data(&[_]A{out}, 1);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_async_eval(v));
        }
        return out;
    }
};

/// `DSV41_DEC_UBENCH` is "1" for every scene, else substrings of the scenes' "kind n" labels (`|` apart) to run.
fn sceneOn(label: []const u8) bool {
    const only = env.span(.DSV41_DEC_UBENCH) orelse return false;
    if (std.mem.eql(u8, only, "1")) return true;
    var it = std.mem.tokenizeScalar(u8, only, '|');
    while (it.next()) |part| if (std.mem.indexOf(u8, label, part) != null) return true;
    return false;
}

test "dsv41 dec topk: ubench" {
    if (!decode.ubenchOn() or !mlx.metalKernelsAvailable() or mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var ks = try Kernels.build();
    defer ks.deinit();
    try probe(gpa, &ks, mlx.gpuStream());
    var m = try todayModel();
    defer _ = mlx.mlx_fast_metal_kernel_free(m.topk_kernel);
    const touch = try dsv41.buildKernel("dec_topk_touch", &.{ "x", "prev" }, &.{"out"}, touch_src, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(touch);
    const g: Geom = .{};
    var data = Scope.init(gpa, mlx.gpuStream());
    defer data.deinit();
    var chain_sc = Scope.init(gpa, mlx.gpuStream());
    defer chain_sc.deinit();
    var prng = std.Random.DefaultPrng.init(31);
    const rng = prng.random();
    const Scene = struct { kind: @TypeOf(@as(Chain, undefined).kind), n: usize };
    const scenes = [_]Scene{
        .{ .kind = .plain, .n = 4096 },      .{ .kind = .plain, .n = 32768 },      .{ .kind = .plain, .n = 131072 }, .{ .kind = .plain, .n = 262144 }, .{ .kind = .plain, .n = 1048576 },
        .{ .kind = .made, .n = 32768 },      .{ .kind = .made, .n = 131072 },      .{ .kind = .made, .n = 262144 },  .{ .kind = .made, .n = 1048576 }, .{ .kind = .gathered, .n = 32768 },
        .{ .kind = .gathered, .n = 131072 }, .{ .kind = .gathered, .n = 1048576 },
    };
    const copies = 4;
    for (scenes) |sn| for ([_]usize{ 1, 6 }) |rows| {
        var tag: [48]u8 = undefined;
        if (!sceneOn(try std.fmt.bufPrint(&tag, "{t} {d}", .{ sn.kind, sn.n }))) continue;
        const ratio: usize = if (sn.kind == .plain) 2 else 1;
        const c: Case = .{ .rows = rows, .n = sn.n, .ratio = ratio };
        var full: [copies]A = undefined;
        var slots: [copies]A = undefined;
        var cands: [copies]A = undefined;
        var masks: [copies]A = undefined;
        const nblk = ceilDiv(sn.n, g.bs);
        for (0..copies) |ci_| {
            const host = try gpa.alloc(f32, rows * sn.n);
            defer gpa.free(host);
            for (0..rows) |r| fillRow(rng, 1, host[r * sn.n ..][0..sn.n], lensOf(c, r));
            full[ci_] = try upload(&data, host, &.{ ci(rows), ci(sn.n) });
            if (sn.kind != .gathered) continue;
            const gs = try gpa.alloc(f32, rows * g.kb * g.bs);
            defer gpa.free(gs);
            for (gs) |*v| v.* = rng.floatNorm(f32);
            slots[ci_] = try upload(&data, gs, &.{ ci(rows), ci(g.kb * g.bs) });
            const cd = try gpa.alloc(i32, rows * g.kb);
            defer gpa.free(cd);
            const mk = try gpa.alloc(u8, rows * nblk);
            defer gpa.free(mk);
            @memset(mk, 0);
            for (0..rows) |r| {
                // kb distinct blocks, ascending: a stride walk with a random start.
                const step = nblk / g.kb;
                for (0..g.kb) |j| {
                    const b = j * step + rng.uintLessThan(usize, step);
                    cd[r * g.kb + j] = @intCast(b);
                    mk[r * nblk + b] = 1;
                }
            }
            cands[ci_] = try upload(&data, cd, &.{ ci(rows), ci(g.kb) });
            masks[ci_] = try keep(&data, mlx.mlx_array_new_data(mk.ptr, &[_]c_int{ ci(rows), ci(nblk) }, 2, .bool_));
        }
        var mine_total: f64 = 0;
        var head_mine: f64 = 0;
        var today_total: f64 = 0;
        var head_today: f64 = 0;
        for ([_]@TypeOf(@as(Chain, undefined).arm){ .head, .mine, .today }) |arm| for ([_]bool{ false, true }) |on_slots| {
            // The head arm runs once over each arm's own inputs, so its streaming copy is subtracted like for like.
            if (on_slots and sn.kind != .gathered) continue;
            var ch: Chain = .{
                .sc = &chain_sc,
                .ks = &ks,
                .mdl = &m,
                .touch = touch,
                .g = g,
                .scores = if (on_slots) &slots else &full,
                .cands = &cands,
                .masks = &masks,
                .rows = rows,
                .p0 = p0Of(c),
                .ratio = ratio,
                .arm = arm,
                .kind = sn.kind,
                .prev = full[0],
            };
            if (on_slots and arm == .today) continue;
            if (!on_slots and sn.kind == .gathered and arm == .mine) continue;
            const us = try decode.chainUs(&ch, 24);
            switch (arm) {
                .head => if (on_slots) {
                    head_mine = us;
                } else {
                    head_today = us;
                    if (sn.kind != .gathered) head_mine = us;
                },
                .mine => mine_total = us,
                .today => today_total = us,
            }
        };
        var name: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&name, "topk {t} n={d}", .{ sn.kind, sn.n });
        const bytes: f64 = if (sn.kind == .gathered) @floatFromInt(rows * (g.kb * g.bs * 4 + g.kb * 4)) else @floatFromInt(rows * sn.n * 4);
        decode.report(label, rows, mine_total - head_mine, bytes, today_total - head_today);
        std.debug.print("[dec-ubench] topk {t} n={d} R={d} raw: mine {d:.1} us (head {d:.1}), today {d:.1} us (head {d:.1})\n", .{ sn.kind, sn.n, rows, mine_total, head_mine, today_total, head_today });
    };
}

test "dsv41 dec topk: each declared launch is the wrapper's, byte for byte" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const a = testing.allocator;
    var ks = try Kernels.build();
    defer ks.deinit();
    const g: Geom = .{};
    const Served = struct { l: *const metal.Launch, p: Sel.Params, shape: Shape, made: bool };
    for ([_]Served{
        .{ .l = &pick_l, .p = .{ .mode = .pick }, .shape = .{ .rows = 3, .n = 5000, .k = 1 }, .made = false },
        .{ .l = &pick_l, .p = .{ .mode = .gathered }, .shape = .{ .rows = 3, .n = 16384, .k = 1 }, .made = false },
        .{ .l = &whole_l, .p = .{ .mode = .whole, .kbl = 2048 }, .shape = .{ .rows = 2, .n = 30000, .k = 1 }, .made = true },
        .{ .l = &whole_l, .p = .{ .mode = .whole, .kbl = 513 }, .shape = .{ .rows = 2, .n = 30000, .k = 1 }, .made = false },
        .{ .l = &merge_l, .p = .{ .mode = .merge, .kbl = 2048 }, .shape = .{ .rows = 2, .n = 140001, .k = 1 }, .made = true },
        .{ .l = &merge_l, .p = .{ .mode = .merge, .kbl = 513 }, .shape = .{ .rows = 2, .n = 140001, .k = 1 }, .made = false },
    }) |sv| {
        const c = try a.create(kit.Case(Sel));
        c.* = .init(a, sv.l, sv.p);
        defer {
            c.deinit();
            a.destroy(c);
        }
        try testing.expect(try c.prepare(sv.shape, .random, 7));
        try c.run();
        const want = try a.dupe(u8, c.rig.outputBytes());
        defer a.free(want);
        var sc = Scope.init(a, mlx.gpuStream());
        defer sc.deinit();
        const meta = c.in(1);
        const out = (try selectG(&sc, g, &ks, .{ .scores = c.args[0], .cand = if (sv.p.mode == .gathered) c.args[2] else none, .p0 = @intFromFloat(meta[0]), .ratio = @intFromFloat(meta[1]), .made = sv.made })) orelse return error.Declined;
        const picks = try dsv41.hostI32(out.picks);
        defer a.free(picks);
        try testing.expectEqualSlices(u8, want[0 .. picks.len * 4], std.mem.sliceAsBytes(picks));
        if (!sv.made) continue;
        const cand = try dsv41.hostI32(out.cand);
        defer a.free(cand);
        try testing.expectEqualSlices(u8, want[picks.len * 4 ..][0 .. cand.len * 4], std.mem.sliceAsBytes(cand));
    }
}

test "dsv41 dec topk: the kernel contract's gates 1-4 hold for every registered launch" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const a = testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try register(&set);
    var t: kc.gates.Tally = .{};
    try kc.gates.runAll(a, set.kernels.items, 0xc0de, &t);
    std.debug.print("[dec] topk contract: {d}/{d} launches pass gates 1-4 (virtual GPU {})\n", .{ t.passed, t.kernels, mlx.virtualGpu() });
    try testing.expectEqual(@as(usize, 0), t.problems());
    try testing.expectEqual(set.kernels.items.len, t.passed);
}

const fixture = @import("fixture.zig");

/// The scores of the indexer of layer `li` for rows at p0.. of the fixture's state, as `dsv41.indexer` makes them.
fn fixtureScores(sc: *Scope, fx: *fixture.Fixture, li: usize, x: A, qr: A, p0: usize) !A {
    const m = fx.m;
    const c = &m.cfg;
    const iw = &fx.layer(li).idx.?;
    const rows: usize = @intCast(mlx.getShape(x)[0]);
    const owner = &fx.state().layers[20].idx_k.?;
    const nb = (p0 + rows) / m.cfg.dsv4_compress_ratios[li];
    var keys = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&keys, owner.buf, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(nb), ci(owner.width) }, 2, &[_]c_int{ 1, 1 }, 2, sc.s));
    _ = try keep(sc, keys);
    const qm = try iw.wq_b.apply(sc, qr);
    var q3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&q3, qm, &[_]c_int{ ci(rows), ci(c.dsv4_index_n_heads), ci(c.dsv4_index_head_dim) }, 3, sc.s));
    _ = try keep(sc, q3);
    const q = try dsv41.ropeQat(sc, m, q3, m.rope_yarn.fwd, try keep(sc, mlx.mlx_array_new_int(ci(p0))), 1.0, .fp4_ue8m0, .float32);
    const wscale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.dsv4_index_head_dim))) / @sqrt(@as(f32, @floatFromInt(c.dsv4_index_n_heads)));
    return dsv41.indexScores(sc, m, q, keys, try iw.weights_proj.apply(sc, x), wscale, p0, m.cfg.dsv4_compress_ratios[li]);
}

test "dsv41 dec topk: the fixture's real scores at layers 20 and 24 give today's picks and candidate blocks" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var ks = try Kernels.build();
    defer ks.deinit();
    const g: Geom = .{};
    // More keys than the candidate source keeps blocks for: layer 20 has 20 000 keys, 2 500 blocks.
    const ctx = 20000;
    const fx = try fixture.Fixture.init(gpa, .{ .ctx = ctx });
    defer fx.deinit();
    for ([_]usize{ 1, 2, 6 }) |rows| {
        var sc = Scope.init(gpa, mlx.gpuStream());
        defer sc.deinit();
        const p0 = ctx - rows;
        const x = try keep(&sc, try fx.normal(&.{ ci(rows), ci(fx.m.cfg.hidden_size) }, .bfloat16, 0, 1));
        const qr = try keep(&sc, try fx.normal(&.{ ci(rows), ci(fx.m.cfg.dsv4_q_lora_rank) }, .bfloat16, 0, 1));
        // Today's: the indexer itself, layer 20 (makes the mask) then 24 (reads it).
        var sh: dsv41.Shared = .{};
        const want20 = try dsv41.hostI32(try dsv41.indexer(&sc, fx.m, fx.state(), 20, &fx.layer(20).idx.?, x, qr, p0, &sh));
        defer gpa.free(want20);
        const mask = try dsv41.hostI32(try astype(&sc, sh.cand, .int32));
        defer gpa.free(mask);
        const want24 = try dsv41.hostI32(try dsv41.indexer(&sc, fx.m, fx.state(), 24, &fx.layer(24).idx.?, x, qr, p0, &sh));
        defer gpa.free(want24);
        // Ours, from the same scores.
        const s20 = try fixtureScores(&sc, fx, 20, x, qr, p0);
        const out = (try selectG(&sc, g, &ks, .{ .scores = s20, .p0 = p0, .ratio = 1, .made = true })) orelse return error.Declined;
        const got20 = try dsv41.hostI32(out.picks);
        defer gpa.free(got20);
        try expectRows("layer 20 picks", got20, want20, g.k);
        const cand = try dsv41.hostI32(out.cand);
        defer gpa.free(cand);
        const nblk = ceilDiv(ctx, g.bs);
        for (0..rows) |r| {
            const row = mask[r * nblk ..][0..nblk];
            var kept: usize = 0;
            for (cand[r * g.kb ..][0..g.kb]) |b| {
                if (b < 0) break;
                try testing.expectEqual(@as(i32, 1), row[@intCast(b)]);
                kept += 1;
            }
            var ones: usize = 0;
            for (row) |v| ones += @intCast(v);
            try testing.expectEqual(ones, kept);
        }
        // Layer 24 scores only the candidate blocks: their slots of its full scores.
        const s24 = try dsv41.hostF32(try fixtureScores(&sc, fx, 24, x, qr, p0));
        defer gpa.free(s24);
        const slots = try gpa.alloc(f32, rows * g.kb * g.bs);
        defer gpa.free(slots);
        for (0..rows) |r| for (cand[r * g.kb ..][0..g.kb], 0..) |b, bi| for (0..g.bs) |j| {
            const key: usize = if (b >= 0) @as(usize, @intCast(b)) * g.bs + j else ctx;
            slots[(r * g.kb + bi) * g.bs + j] = if (key < ctx) s24[r * ctx + key] else ninf;
        };
        const got24 = (try selectG(&sc, g, &ks, .{ .scores = try upload(&sc, slots, &.{ ci(rows), ci(g.kb * g.bs) }), .cand = try upload(&sc, cand, &.{ ci(rows), ci(g.kb) }), .p0 = p0, .ratio = 1 })) orelse return error.Declined;
        const picks24 = try dsv41.hostI32(got24.picks);
        defer gpa.free(picks24);
        try expectRows("layer 24 picks", picks24, want24, g.k);
    }
}
