//! `MLX_SERVE_DSV41_ENGRAM_GPU_IDS`: a decode step whose token is still on the GPU hashes its Engram row ids on the
//! GPU, so the step is committed whole before the host reads any token. Off, `extendIds` commits layer 0, waits for
//! the token, hashes on the host and only then builds the Engram layers: the GPU idled ~0.7 ms of an 18 ms step there.
//!
//! `Hash.rows` is `(t0 * m0) ^ P_k` per n-gram size k, then `% prime + offset`, where `P_k` XORs the earlier tokens'
//! terms. The host computes `P` from the history (already complete: it holds every token before this one), the GPU
//! the rest from the token, in MLX ops (wrapping u64 multiply, xor, remainder). The rows are gathered by id from the
//! held tables on the GPU (`fuse.wrapHeld`, `dec_engram_kv`): resident packs whose every table is wrapped. The token
//! joins the history after the step is committed (`extendIds`), before anything reads it.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");
const log = @import("../../../sdk/util/log.zig");
const dsv41 = @import("../deepseek_v41.zig");
const engram = @import("../engram.zig");
const fusion = @import("../fuse.zig");
const decode = @import("decode.zig");
const engram_kv = @import("engram_kv.zig");
const engram_kvc = @import("engram_kvc.zig");

const A = mlx.mlx_array;
const Scope = dsv41.Scope;

/// The hash's per-column constants and the token map on the GPU, built at load.
pub const Ctx = struct {
    /// `[n_layers * cols]` u64: each column's layer's first multiplier, its prime, its first row.
    m0: A,
    primes: A,
    offsets: A,
    /// The compressed id of every token, u32 `[vocab]`.
    map: A,
    n: usize,

    pub fn deinit(c: *Ctx) void {
        for ([_]A{ c.m0, c.primes, c.offsets, c.map }) |a| _ = mlx.mlx_array_free(a);
    }
};

pub fn on() bool {
    return lev.onUnlessZero(.ENGRAM_GPU_IDS);
}

/// The constants of `hs` over `map` (`Dsv41Model.token_map`).
pub fn ctxOf(hs: *const engram.Hash, map: []const u32) !Ctx {
    const nc = hs.cols();
    const n = hs.n_layers * nc;
    var m0: [engram.MAX_LAYERS * engram.MAX_COLS]u64 = undefined;
    var pr: [engram.MAX_LAYERS * engram.MAX_COLS]u64 = undefined;
    var off: [engram.MAX_LAYERS * engram.MAX_COLS]u64 = undefined;
    for (0..hs.n_layers) |l| for (0..nc) |c| {
        m0[l * nc + c] = hs.mult[l][0];
        pr[l * nc + c] = hs.primes[l][c];
        off[l * nc + c] = hs.offsets[l][c];
    };
    const sh = [_]c_int{@intCast(n)};
    return .{
        .m0 = mlx.mlx_array_new_data(&m0, &sh, 1, .uint64),
        .primes = mlx.mlx_array_new_data(&pr, &sh, 1, .uint64),
        .offsets = mlx.mlx_array_new_data(&off, &sh, 1, .uint64),
        .map = mlx.mlx_array_new_data(map.ptr, &[_]c_int{@intCast(map.len)}, 1, .uint32),
        .n = n,
    };
}

/// Each column's XOR of the earlier tokens' terms, for the token after `hist` (`Hash.rows` with its first term out):
/// the slots before the sequence start read `pad`, and every slot at or past a `DEAD` one reads `pad`. `plant` 1
/// shifts each column to the next n-gram size's terms (the tests' planted bug).
pub fn partials(hs: *const engram.Hash, hist: []const u32, out: []u64, plant: u8) void {
    const nc = hs.cols();
    std.debug.assert(out.len == hs.n_layers * nc);
    const pos = hist.len;
    var toks: [engram.MAX_NGRAM]u64 = @splat(0);
    var blocked = false;
    for (1..hs.max_ngram) |s| {
        const t: u32 = if (pos < s) hs.pad else hist[pos - s];
        blocked = blocked or t == engram.Hash.DEAD;
        toks[s] = if (blocked) hs.pad else t;
    }
    for (0..hs.n_layers) |l| {
        var p: u64 = 0;
        for (1..hs.max_ngram) |k| {
            p ^= toks[k] *% hs.mult[l][k];
            const kk = if (plant == 1 and k + 1 < hs.max_ngram) k + 1 else k;
            var q: u64 = p;
            if (kk != k) q ^= toks[kk] *% hs.mult[l][kk];
            for (0..hs.heads) |hd| out[l * nc + (k - 1) * hs.heads + hd] = q;
        }
    }
}

/// The row ids `[n_layers * cols]` u32 (`Hash.rows`' layout for one row) of the token `tok` (any integer dtype, one
/// element, on the GPU) after `hist`, as MLX ops on `sc`'s stream: the compressed id (0 past the map), its first term,
/// the host's `partials`, the prime and the first row.
pub fn rows(sc: *Scope, hs: *const engram.Hash, ctx: *const Ctx, hist: []const u32, tok: A, plant: u8) !A {
    var p: [engram.MAX_LAYERS * engram.MAX_COLS]u64 = undefined;
    partials(hs, hist, p[0..ctx.n], plant);
    const pa = try sc.keep(mlx.mlx_array_new_data(&p, &[_]c_int{@intCast(ctx.n)}, 1, .uint64));
    const len: i32 = @intCast(mlx.mlx_array_size(ctx.map));
    const t = try sc.reshape(try sc.astype(tok, .int32), &.{1});
    const inside = try sc.op2(mlx.mlx_less, t, try sc.ints(&.{len}, &.{1}));
    const at = try sc.op2(mlx.mlx_minimum, t, try sc.ints(&.{len - 1}, &.{1}));
    const zero = try sc.keep(mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{1}, 1, .uint32));
    const comp = try sc.where(inside, try sc.take(ctx.map, at, 0), zero);
    const first = try sc.mul(try sc.astype(comp, .uint64), ctx.m0);
    const rolling = try sc.op2(mlx.mlx_bitwise_xor, first, pa);
    const row = try sc.add(try sc.op2(mlx.mlx_remainder, rolling, ctx.primes), ctx.offsets);
    return sc.astype(row, .uint32);
}

/// `MLX_SERVE_DSV41_SPEC_ENGRAM_AHEAD`: `rows` of each token of `toks` (`[R]`, any integer dtype, on the GPU) after
/// `hist`, as `Hash.rows` reads a run of R tokens: row j's earlier tokens are toks[j-1], toks[j-2], ... then the
/// history's (`[R, n_layers * cols]` u32). The history's slots are the host's (pad before the start and at or past a
/// DEAD one, as `partials`), the tokens' the GPU's; `plant` 1 reads each row's tokens one slot late (the tests').
pub fn rowsMany(sc: *Scope, hs: *const engram.Hash, ctx: *const Ctx, hist: []const u32, toks: A, plant: u8) !A {
    const R = mlx.mlx_array_size(toks);
    const M: usize = hs.max_ngram;
    if (R == 0 or R > 16) return error.Dsv41EngramRows;
    // cv: the history's last M - 1 slots (earliest first), then the R tokens' compressed ids.
    var cv: [engram.MAX_NGRAM]u64 = @splat(0);
    var blocked = false;
    for (1..M) |b| {
        const t: u32 = if (hist.len < b) hs.pad else hist[hist.len - b];
        blocked = blocked or t == engram.Hash.DEAD;
        cv[M - 1 - b] = if (blocked) hs.pad else t;
    }
    const len: i32 = @intCast(mlx.mlx_array_size(ctx.map));
    const t = try sc.reshape(try sc.astype(toks, .int32), &.{@intCast(R)});
    const inside = try sc.op2(mlx.mlx_less, t, try sc.ints(&.{len}, &.{1}));
    const at = try sc.op2(mlx.mlx_minimum, t, try sc.ints(&.{len - 1}, &.{1}));
    const zero = try sc.keep(mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{1}, 1, .uint32));
    const comp = try sc.astype(try sc.where(inside, try sc.take(ctx.map, at, 0), zero), .uint64);
    const all = try sc.concat(&.{ try sc.keep(mlx.mlx_array_new_data(&cv, &[_]c_int{@intCast(M - 1)}, 1, .uint64)), comp }, 0);
    // Slot s of row j (s = 0: the row's own token) at cv index (M - 1) + j - s.
    var idx: [16 * engram.MAX_NGRAM]i32 = undefined;
    for (0..R) |j| for (0..M) |sl| {
        const s_ = if (plant == 1 and sl > 0) @min(sl + 1, M - 1 + j) else sl;
        idx[j * M + sl] = @intCast(M - 1 + j - s_);
    };
    const tk = try sc.take(all, try sc.keep(mlx.mlx_array_new_data(&idx, &[_]c_int{ @intCast(R), @intCast(M) }, 2, .int32)), 0);
    // Column c (layer l, n-gram size k + 1, head) XORs slots 0..k, each times its layer's multiplier.
    const nc = hs.cols();
    var ms: [engram.MAX_NGRAM * engram.MAX_LAYERS * engram.MAX_COLS]u64 = @splat(0);
    for (0..M) |sl| for (0..hs.n_layers) |l| for (1..M) |k| for (0..hs.heads) |hd| {
        if (sl <= k) ms[sl * ctx.n + l * nc + (k - 1) * hs.heads + hd] = hs.mult[l][sl];
    };
    const mult = try sc.keep(mlx.mlx_array_new_data(&ms, &[_]c_int{ @intCast(M), @intCast(ctx.n) }, 2, .uint64));
    var rolling: A = undefined;
    for (0..M) |sl| {
        const term = try sc.mul(try sc.slice(tk, 1, sl, sl + 1), try sc.slice(mult, 0, sl, sl + 1));
        rolling = if (sl == 0) term else try sc.op2(mlx.mlx_bitwise_xor, rolling, term);
    }
    const row = try sc.add(try sc.op2(mlx.mlx_remainder, rolling, ctx.primes), ctx.offsets);
    return sc.astype(row, .uint32);
}

/// The Engram write of one decode row `x` `[1, hc, dim]` at table `ek` (`e`, wrapped) from the GPU row ids `ids`
/// (`rows`): the held-table gather and wkv in one launch (`dec_engram_kv`), then the gate. With the step's cache
/// `plan` (`engram_kvc.Cache.plan`), the slot kernel through the cache. Null where the kernel declines the table.
pub fn write(sc: *Scope, m: *dsv41.Dsv41Model, e: *const dsv41.EngramLayer, ids: A, plan: A, ek: usize, nc: usize, x: A) !?A {
    const g = e.gpu orelse return null;
    if (mlx.getShape(x)[0] != 1 or !fusion.kvServes(m, e, 1)) return null;
    const t = &e.table;
    const page = std.heap.pageSize();
    const geo: engram_kv.Geo = .{ .w_base = t.w_off % page, .w_stride = t.w_stride, .s_base = t.s_off % page, .s_stride = t.s_stride };
    const gw = geo.words();
    const ga = try sc.keep(mlx.mlx_array_new_data(&gw, &[_]c_int{6}, 1, .uint32));
    const mine = try sc.slice(ids, 0, ek * nc, (ek + 1) * nc);
    const tab: engram_kv.Table = .{ .hd = t.head_dim, .tb = t.bits };
    const kv = if (plan.ctx != null) blk: {
        const c = &m.kvc.?;
        const kv = try engram_kv.launchSlots(sc, tab, g[0], g[1], mine, ga, e.wkv.w, e.wkv.s, try c.of(sc, plan, ek, nc)) orelse return null;
        c.after(kv);
        break :blk kv;
    } else try engram_kv.launch(sc, tab, 1, g[0], g[1], mine, ga, e.wkv.w, e.wkv.s) orelse return null;
    return try dsv41.engramGate(sc, m, e.qk, x, kv);
}

/// `write` of each row of an exact verify's span `x` `[hi - lo, hc, dim]` alone, from its host-hashed ids (`ids` in
/// `Hash.rows`' layout over `tables` tables: chunk row t's at `(t * tables + ek) * nc`), without the cache (a hit
/// gives the slot kernel's bits, `engram_kvc.zig`): every row the served plain step's write at its position (the
/// owner's `engram_slots`, `plan.zig`). Null where `write` declines a row.
pub fn writeRows(sc: *Scope, m: *dsv41.Dsv41Model, e: *const dsv41.EngramLayer, ids: []const u32, tables: usize, lo: usize, hi: usize, ek: usize, nc: usize, x: A) !?A {
    var ys: [decode.max_rows]A = undefined;
    const L = hi - lo;
    if (L == 0 or L > ys.len or ids.len < hi * tables * nc) return null;
    for (0..L) |t| {
        const row = ids[((lo + t) * tables + ek) * nc ..][0..nc];
        const a = try sc.keep(mlx.mlx_array_new_data(row.ptr, &[_]c_int{@intCast(nc)}, 1, .uint32));
        ys[t] = (try write(sc, m, e, a, .{ .ctx = null }, 0, nc, try sc.slice(x, 0, t, t + 1))) orelse return null;
    }
    return try sc.concat(ys[0..L], 0);
}

/// At load, after `fuse.settle`: the switch serves only where every Engram table is wrapped as GPU buffers (wrapped
/// here) and the fused wkv reads it; one line says which.
pub fn settle(m: *dsv41.Dsv41Model) !void {
    if (!on() or m.engrams.len == 0) return;
    const hs = &(m.hash orelse return);
    const why: ?[]const u8 = blk: {
        if (m.streamed != null or m.streamed_experts) break :blk "the experts stream";
        if (m.pair != null or m.tp != null) break :blk "a pair or tensor split";
        if (m.embed_tab != null) break :blk "the embedding is read from the pack";
        for (m.engrams) |*e| {
            if (e.gpu == null and !fusion.wrapHeld(e)) break :blk "a table not held as GPU buffers";
            if (!fusion.kvServes(m, e, 1)) break :blk "a table or wkv the fused kernel does not read";
        }
        break :blk null;
    };
    if (why) |w| {
        log.info("deepseek_v41: MLX_SERVE_DSV41_ENGRAM_GPU_IDS off: {s}\n", .{w});
        return;
    }
    m.engram_ids = try ctxOf(hs, m.token_map);
    log.info("deepseek_v41: MLX_SERVE_DSV41_ENGRAM_GPU_IDS: a decode step's Engram ids hashed on the GPU ({d} tables)\n", .{m.engrams.len});
    if (!engram_kvc.on()) return;
    const e0 = &m.engrams[0];
    const n: usize = @intCast(mlx.getShape(e0.wkv.w)[0]);
    const k: usize = @as(usize, @intCast(mlx.getShape(e0.wkv.w)[1])) * 4;
    const nc = hs.cols();
    var facts: engram_kvc.Facts = .{ .layers = m.engrams.len, .held = true, .streamed = m.cfg.dsv41_streamed, .reads = true };
    for (m.engrams) |*e| {
        facts.held = facts.held and e.table.isHeld();
        facts.reads = facts.reads and engram_kvc.tableReads(&e.table);
    }
    const serves = engram_kvc.allocates(facts) and for (m.engrams) |*e| {
        if (@as(usize, @intCast(mlx.getShape(e.wkv.w)[0])) != n or k != nc * engram_kv.slot_hd or k % 512 != 0 or n % engram_kv.slot_cols != 0) break false;
    } else true;
    m.kvc = if (serves) try engram_kvc.Cache.init(m.s, m.engrams.len * nc, n) else null;
    if (m.kvc) |*c| {
        log.info("deepseek_v41: MLX_SERVE_DSV41_ENGRAM_KV_CACHE: {d} tables x {d} sets x {d} ways of {d}-wide partials, {d} MB\n", .{ c.nt, c.sets, engram_kvc.ways, n, c.bytes() >> 20 });
    } else log.info("deepseek_v41: MLX_SERVE_DSV41_ENGRAM_KV_CACHE off: the slot kernel does not read these tables\n", .{});
}

// ── tests ──

const testing = std.testing;

/// A hash with the release's geometry (2 tables, 3-grams, 8 heads) on random odd multipliers and buckets from `base`.
fn testHash(rng: std.Random, base: u64) engram.Hash {
    return testHashOf(rng, base, 3);
}

/// `testHash` over 2- to `ngram`-grams.
fn testHashOf(rng: std.Random, base: u64, ngram: u32) engram.Hash {
    var h: engram.Hash = .{ .n_layers = 2, .max_ngram = ngram, .heads = 8, .pad = 2 };
    var p: u64 = base + rng.uintLessThan(u64, base);
    for (0..h.n_layers) |l| {
        for (0..h.max_ngram) |k| h.mult[l][k] = rng.uintLessThan(u64, 1 << 47) * 2 + 1;
        var off: u64 = 0;
        for (0..h.cols()) |c| {
            p += 1 + rng.uintLessThan(u64, @max(base / 64, 2));
            h.primes[l][c] = p;
            h.offsets[l][c] = off;
            off += p;
        }
    }
    return h;
}

fn gpuOk() bool {
    return mlx.metalKernelsAvailable() and !mlx.noGpuBackend() and !mlx.virtualGpu();
}

test "dsv41 engram gpu ids: the row ids hashed on the GPU from a GPU token are Hash.rows' ids, every history length from 0 (the pad slots), a DEAD slot, a token past the map; a planted n-gram shift differs" {
    if (!gpuOk()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xE1D5);
    const rng = prng.random();
    const s = mlx.gpuStream();
    const vocab = 4096;
    const map = try gpa.alloc(u32, vocab);
    defer gpa.free(map);
    for (map) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
    for (0..4) |round| {
        const hs = testHash(rng, 1 << 20);
        var ctx = try ctxOf(&hs, map);
        defer ctx.deinit();
        var differ_plant: usize = 0;
        for (0..12) |len| {
            const hist = try gpa.alloc(u32, len);
            defer gpa.free(hist);
            for (hist) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
            if (len > 3 and round % 2 == 1) hist[len - 2] = engram.Hash.DEAD;
            // A token inside the map, then one past it (compressed id 0).
            for ([_]u32{ rng.uintLessThan(u32, vocab), vocab + 7 }) |tok| {
                const comp: u32 = if (tok < vocab) map[tok] else 0;
                var want: [engram.MAX_LAYERS * engram.MAX_COLS]u32 = undefined;
                hs.rows(hist, &.{comp}, want[0..ctx.n]);
                var sc = Scope.init(gpa, s);
                defer sc.deinit();
                const t = try sc.keep(mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .uint32));
                for ([_]u8{ 0, 1 }) |plant| {
                    const got_a = try rows(&sc, &hs, &ctx, hist, t, plant);
                    try mlx.check(mlx.mlx_array_eval(got_a));
                    const got = (mlx.mlx_array_data_uint32(got_a) orelse return error.NoData)[0..ctx.n];
                    if (plant == 0) {
                        try testing.expectEqualSlices(u32, want[0..ctx.n], got);
                    } else if (!std.mem.eql(u32, want[0..ctx.n], got)) differ_plant += 1;
                }
            }
        }
        std.debug.print("[engram-ids] round {d}: 24 cases equal Hash.rows; the planted shift differs in {d} of 24\n", .{ round, differ_plant });
        // The 2-gram columns read no 3-gram term, so the shift moves every case with a token in its 3-gram slot.
        try testing.expect(differ_plant >= 20);
    }
}

test "dsv41 engram gpu ids (resident): the write from the GPU-hashed ids is the held-table write from the host ids bit for bit, mxfp4 and mxfp8 tables; the ids of another token differ" {
    if (!gpuOk()) return error.SkipZigTest;
    const fixture = @import("fixture.zig");
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 0, .hi = 1, .ctx = 8, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    const s = mlx.gpuStream();
    const d = m.cfg.hidden_size;
    const hc = m.cfg.dsv4_hc_mult;
    var prng = std.Random.DefaultPrng.init(0xE1D6);
    const rng = prng.random();
    const hs = testHash(rng, 300);
    const nc = hs.cols();
    const hd: usize = 256;
    const K = nc * hd;
    const N = (hc + 1) * d;
    const ew = try fx.normal(&.{ @intCast(N), @intCast(K) }, .bfloat16, 0, 0.02);
    defer _ = mlx.mlx_array_free(ew);
    const eq = try fixture.quantize(ew, 32, 8, "mxfp8");
    defer for (eq) |q| if (q.ctx != null) {
        _ = mlx.mlx_array_free(q);
    };
    const qk = try fx.normal(&.{ @intCast(hc), @intCast(d) }, .float32, 0.5, 0.2);
    defer _ = mlx.mlx_array_free(qk);
    const map = try gpa.alloc(u32, 512);
    defer gpa.free(map);
    for (map) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
    var ctx = try ctxOf(&hs, map);
    defer ctx.deinit();
    // Every id of the hash lands in a table of its rows (the second table's offsets past the first's).
    const n_rows: usize = @intCast(hs.offsets[1][nc - 1] + hs.primes[1][nc - 1]);
    for ([_]@import("../../../sdk/quant/params.zig").QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        var e: dsv41.EngramLayer = .{ .layer = 0, .wkv = .{ .w = eq[0], .s = eq[1], .b = eq[2], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } }, .qk = qk, .table = .{ .fd = -1, .rows = n_rows, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen } };
        try fusion.heldTable(&e.table, n_rows, if (bits == 4) 4096 + 952 else 4096 + 48, rng);
        defer for (e.table.held[0..2]) |mem| std.posix.munmap(mem);
        try testing.expect(fusion.wrapHeld(&e));
        defer {
            _ = mlx.mlx_synchronize(s);
            fusion.unwrap(&e);
        }
        for (0..4) |case| {
            var sc = Scope.init(gpa, s);
            defer sc.deinit();
            const hist = try gpa.alloc(u32, 2 + case);
            defer gpa.free(hist);
            for (hist) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
            const tok = rng.uintLessThan(u32, 512);
            var host: [engram.MAX_LAYERS * engram.MAX_COLS]u32 = undefined;
            hs.rows(hist, &.{map[tok]}, host[0..ctx.n]);
            const x = try sc.keep(try fx.normal(&.{ 1, @intCast(hc), @intCast(d) }, .bfloat16, 0, 1));
            const t = try sc.keep(mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .uint32));
            const ids = try rows(&sc, &hs, &ctx, hist, t, 0);
            const got = (try write(&sc, m, &e, ids, .{ .ctx = null }, 1, nc, x)) orelse return error.Declined;
            const want = (try fusion.heldWrite(&sc, m, &e, host[0..ctx.n], 2, 1, nc, 0, 1, x)) orelse return error.Declined;
            const other: u32 = (tok + 1) % 512;
            const ot = try sc.keep(mlx.mlx_array_new_data(&other, &[_]c_int{1}, 1, .uint32));
            const plant = (try write(&sc, m, &e, try rows(&sc, &hs, &ctx, hist, ot, 0), .{ .ctx = null }, 1, nc, x)) orelse return error.Declined;
            const a = try dsv41.hostF32(got);
            defer gpa.free(a);
            const b = try dsv41.hostF32(want);
            defer gpa.free(b);
            const c = try dsv41.hostF32(plant);
            defer gpa.free(c);
            var same = true;
            var moved: usize = 0;
            for (a, b, c) |va, vb, vc| {
                same = same and @as(u32, @bitCast(va)) == @as(u32, @bitCast(vb));
                moved += @intFromBool(va != vc);
            }
            std.debug.print("[engram-ids] {t} case {d}: write from the GPU ids {s} the host ids' write; another token's ids move {d} of {d} values\n", .{ mode, case, if (same) "equals" else "DIFFERS FROM", moved, a.len });
            try testing.expect(same);
            try testing.expect(moved > a.len / 2);
        }
    }
}

test "dsv41 engram gpu ids rows: a run of R = 1..6 GPU tokens hashes to Hash.rows' ids of the run, every history length from 0 (the pad slots), a DEAD slot, a token past the map; a planted slot shift differs (SPEC_ENGRAM_AHEAD)" {
    if (!gpuOk()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xE1D7);
    const rng = prng.random();
    const s = mlx.gpuStream();
    const vocab = 4096;
    const map = try gpa.alloc(u32, vocab);
    defer gpa.free(map);
    for (map) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
    var cases: usize = 0;
    var differ_plant: usize = 0;
    for (0..4) |round| {
        var hs = testHash(rng, 1 << 20);
        // The release hashes 2- to 4-grams: rounds 2 and 3 widen the test hash to it.
        if (round >= 2) {
            hs.max_ngram = 4;
            for (0..hs.n_layers) |l| for (0..hs.max_ngram) |k| {
                hs.mult[l][k] = rng.uintLessThan(u64, 1 << 47) * 2 + 1;
            };
            var p: u64 = (1 << 20) + rng.uintLessThan(u64, 1 << 20);
            for (0..hs.n_layers) |l| {
                var off: u64 = 0;
                for (0..hs.cols()) |c| {
                    p += 1 + rng.uintLessThan(u64, 1 << 14);
                    hs.primes[l][c] = p;
                    hs.offsets[l][c] = off;
                    off += p;
                }
            }
        }
        var ctx = try ctxOf(&hs, map);
        defer ctx.deinit();
        for (0..10) |len| for (1..7) |R| {
            const hist = try gpa.alloc(u32, len);
            defer gpa.free(hist);
            for (hist) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
            if (len > 2 and round % 2 == 1) hist[len - 1 - rng.uintLessThan(usize, 2)] = engram.Hash.DEAD;
            var toks: [6]u32 = undefined;
            var comp: [6]u32 = undefined;
            for (toks[0..R], comp[0..R], 0..) |*t, *c, j| {
                t.* = if (j == 1) vocab + 7 else rng.uintLessThan(u32, vocab);
                c.* = if (t.* < vocab) map[t.*] else 0;
            }
            const want = try gpa.alloc(u32, R * ctx.n);
            defer gpa.free(want);
            hs.rows(hist, comp[0..R], want);
            var sc = Scope.init(gpa, s);
            defer sc.deinit();
            const t = try sc.keep(mlx.mlx_array_new_data(&toks, &[_]c_int{@intCast(R)}, 1, .uint32));
            for ([_]u8{ 0, 1 }) |plant| {
                const got_a = try rowsMany(&sc, &hs, &ctx, hist, t, plant);
                try mlx.check(mlx.mlx_array_eval(got_a));
                try testing.expectEqualSlices(c_int, &.{ @intCast(R), @intCast(ctx.n) }, mlx.getShape(got_a));
                const got = (mlx.mlx_array_data_uint32(got_a) orelse return error.NoData)[0 .. R * ctx.n];
                if (plant == 0) {
                    try testing.expectEqualSlices(u32, want, got);
                    cases += 1;
                } else if (!std.mem.eql(u32, want, got)) differ_plant += 1;
            }
        };
    }
    std.debug.print("[engram-ids] rows: {d} runs equal Hash.rows; the planted shift differs in {d}\n", .{ cases, differ_plant });
    try testing.expect(differ_plant >= cases * 9 / 10);
}

test "dsv41 engram gpu ids (resident): a verify's Engram write from the kv made ahead (the GPU-hashed run of t1 and 4 drafts, the wrapped table's rows, wkv over those 5 rows, the first R taken) is the served verify write (host ids, the host-copied rows, wkv over R rows) bit for bit, R = 2..5, mxfp4 and mxfp8 tables; the kv rotated a row differs (SPEC_ENGRAM_AHEAD)" {
    if (!gpuOk()) return error.SkipZigTest;
    const fixture = @import("fixture.zig");
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 0, .hi = 1, .ctx = 8, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    const s = mlx.gpuStream();
    const d = m.cfg.hidden_size;
    const hc = m.cfg.dsv4_hc_mult;
    var prng = std.Random.DefaultPrng.init(0xE1D8);
    const rng = prng.random();
    const hs = testHash(rng, 300);
    const nc = hs.cols();
    const hd: usize = 256;
    const K = nc * hd;
    const N = (hc + 1) * d;
    const ew = try fx.normal(&.{ @intCast(N), @intCast(K) }, .bfloat16, 0, 0.02);
    defer _ = mlx.mlx_array_free(ew);
    const eq = try fixture.quantize(ew, 32, 8, "mxfp8");
    defer for (eq) |q| if (q.ctx != null) {
        _ = mlx.mlx_array_free(q);
    };
    const qk = try fx.normal(&.{ @intCast(hc), @intCast(d) }, .float32, 0.5, 0.2);
    defer _ = mlx.mlx_array_free(qk);
    const map = try gpa.alloc(u32, 512);
    defer gpa.free(map);
    for (map) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
    var ctx = try ctxOf(&hs, map);
    defer ctx.deinit();
    const n_rows: usize = @intCast(hs.offsets[1][nc - 1] + hs.primes[1][nc - 1]);
    const B = dsv41.eng_ahead_rows;
    var served: usize = 0;
    for ([_]@import("../../../sdk/quant/params.zig").QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        var e: dsv41.EngramLayer = .{ .layer = 0, .wkv = .{ .w = eq[0], .s = eq[1], .b = eq[2], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } }, .qk = qk, .table = .{ .fd = -1, .rows = n_rows, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen } };
        try fusion.heldTable(&e.table, n_rows, if (bits == 4) 4096 + 952 else 4096 + 48, rng);
        defer for (e.table.held[0..2]) |mem| std.posix.munmap(mem);
        try testing.expect(fusion.wrapHeld(&e));
        defer {
            _ = mlx.mlx_synchronize(s);
            fusion.unwrap(&e);
        }
        const g = e.gpu.?;
        const page = std.heap.pageSize();
        const geo: engram_kv.Geo = .{ .w_base = e.table.w_off % page, .w_stride = e.table.w_stride, .s_base = e.table.s_off % page, .s_stride = e.table.s_stride };
        const gw = geo.words();
        for (2..B + 1) |R| for (0..2) |ek| {
            var sc = Scope.init(gpa, s);
            defer sc.deinit();
            const hist = try gpa.alloc(u32, 3 + R);
            defer gpa.free(hist);
            for (hist) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
            var toks: [B]u32 = undefined;
            var comp: [B]u32 = undefined;
            for (&toks, &comp) |*t, *c| {
                t.* = rng.uintLessThan(u32, 512);
                c.* = map[t.*];
            }
            // Ahead: the whole block's ids hashed on the GPU, its rows gathered from the wrapped table, wkv over B rows.
            const ga = try sc.keep(mlx.mlx_array_new_data(&gw, &[_]c_int{6}, 1, .uint32));
            const t = try sc.keep(mlx.mlx_array_new_data(&toks, &[_]c_int{B}, 1, .uint32));
            const ids = try rowsMany(&sc, &hs, &ctx, hist, t, 0);
            const mine = try sc.reshape(try sc.slice(ids, 1, ek * nc, (ek + 1) * nc), &.{@intCast(B * nc)});
            const block = (try engram_kv.gatherRows(&sc, .{ .hd = hd, .tb = bits }, g[0], g[1], mine, ga)) orelse return error.Declined;
            const kv = try dsv41.engramWkv(&sc, &e, try sc.reshape(block, &.{ B, -1 }));
            const x = try sc.keep(try fx.normal(&.{ @intCast(R), @intCast(hc), @intCast(d) }, .bfloat16, 0, 1));
            const got = try dsv41.engramGate(&sc, m, qk, x, try sc.slice(kv, 0, 0, R));
            // Planted: the rows rotated one place (each row's kv another token's).
            const plant = try dsv41.engramGate(&sc, m, qk, x, try sc.concat(&.{ try sc.slice(kv, 0, 1, R), try sc.slice(kv, 0, 0, 1) }, 0));
            // Served: the verify's R tokens hashed on the host, their rows copied out of the table, dequantized to f32,
            // engramApply over R rows.
            const host = try gpa.alloc(u32, R * ctx.n);
            defer gpa.free(host);
            hs.rows(hist, comp[0..R], host);
            const raw: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", R * nc * wlen), .s = try gpa.alignedAlloc(u8, .@"4", R * nc * slen), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
            defer raw.deinit(gpa);
            for (0..R) |r| for (0..nc) |j| {
                const id = host[r * ctx.n + ek * nc + j];
                const i = r * nc + j;
                @memcpy(raw.w[i * wlen ..][0..wlen], e.table.held[0][e.table.w_off + id * wlen ..][0..wlen]);
                @memcpy(raw.s[i * slen ..][0..slen], e.table.held[1][e.table.s_off + id * slen ..][0..slen]);
            };
            const hrows = try sc.keep(try e.table.dequantTo(raw, R * nc, .float32, s));
            const want = try dsv41.engramApply(&sc, m, &e, x, try sc.reshape(hrows, &.{ @intCast(R), -1 }));
            const a = try dsv41.hostF32(got);
            defer gpa.free(a);
            const b = try dsv41.hostF32(want);
            defer gpa.free(b);
            const p = try dsv41.hostF32(plant);
            defer gpa.free(p);
            var differ: usize = 0;
            var planted: usize = 0;
            for (a, b, p) |u, v, w| {
                if (@as(u32, @bitCast(u)) != @as(u32, @bitCast(v))) differ += 1;
                if (@as(u32, @bitCast(w)) != @as(u32, @bitCast(v))) planted += 1;
            }
            std.debug.print("[engram-ahead] {t} R {d} table {d}: {d} of {d} words differ from the served write; the kv rotated a row, {d}\n", .{ mode, R, ek, differ, b.len, planted });
            try testing.expectEqual(@as(usize, 0), differ);
            try testing.expect(planted > b.len / 2);
            served += 1;
        };
    }
    try testing.expectEqual(@as(usize, 2 * (B - 1) * 2), served);
}

test "dsv41 engram gpu ids (MLX_SERVE_DSV41_ENGRAM_GPU_IDS): the served fused wkv over the GPU-hashed ids' rows of the held table, against MLX's chain over the host ids' rows (dequantized, cast to bf16, the mxfp8 wkv), has a worst element no farther from float64 and differs in at most 1e-4 of the words, at the release's Engram row (24 rows of 256 into (hc + 1) x hidden), both tables, mxfp4 and mxfp8 rows; one slot's row one off is farther" {
    if (!gpuOk()) return error.SkipZigTest;
    const fixture = @import("fixture.zig");
    const gpa = testing.allocator;
    const none: A = .{ .ctx = null };
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 0, .hi = 1, .ctx = 8, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    const s = mlx.gpuStream();
    const d = m.cfg.hidden_size;
    const hc = m.cfg.dsv4_hc_mult;
    var prng = std.Random.DefaultPrng.init(0xE1F6);
    const rng = prng.random();
    const hs = testHashOf(rng, 300, 4);
    const nc = hs.cols();
    const hd: usize = 256;
    const K = nc * hd;
    const N = (hc + 1) * d;
    const ew = try fx.normal(&.{ @intCast(N), @intCast(K) }, .bfloat16, 0, 0.02);
    defer _ = mlx.mlx_array_free(ew);
    const eq = try fixture.quantize(ew, 32, 8, "mxfp8");
    defer for (eq) |q| if (q.ctx != null) {
        _ = mlx.mlx_array_free(q);
    };
    var deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(deq);
    try mlx.check(mlx.mlx_dequantize(&deq, eq[0], eq[1], none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .float32, .has_value = true }, s));
    const wd = try dsv41.hostF32(deq);
    defer gpa.free(wd);
    const map = try gpa.alloc(u32, 512);
    defer gpa.free(map);
    for (map) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
    var ctx = try ctxOf(&hs, map);
    defer ctx.deinit();
    const n_rows: usize = @intCast(hs.offsets[1][nc - 1] + hs.primes[1][nc - 1]);
    const truth = try gpa.alloc(f64, N);
    defer gpa.free(truth);
    var cells: usize = 0;
    var caught: usize = 0;
    var agg: [4]decode.Dist = @splat(.{});
    var awf: decode.WordDiff = .{};
    for ([_]@import("../../../sdk/quant/params.zig").QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        var e: dsv41.EngramLayer = .{ .layer = 0, .wkv = .{ .w = eq[0], .s = eq[1], .b = eq[2], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } }, .qk = none, .table = .{ .fd = -1, .rows = n_rows, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen } };
        try fusion.heldTable(&e.table, n_rows, if (bits == 4) 4096 + 952 else 4096 + 48, rng);
        defer for (e.table.held[0..2]) |mem| std.posix.munmap(mem);
        try testing.expect(fusion.wrapHeld(&e));
        defer {
            _ = mlx.mlx_synchronize(s);
            fusion.unwrap(&e);
        }
        const g = e.gpu.?;
        const page = std.heap.pageSize();
        const gw = (engram_kv.Geo{ .w_base = e.table.w_off % page, .w_stride = e.table.w_stride, .s_base = e.table.s_off % page, .s_stride = e.table.s_stride }).words();
        const tab: engram_kv.Table = .{ .hd = hd, .tb = bits };
        for (0..4) |case| for (0..2) |ek| {
            var sc = Scope.init(gpa, s);
            defer sc.deinit();
            const ga = try sc.keep(mlx.mlx_array_new_data(&gw, &[_]c_int{6}, 1, .uint32));
            const hist = try gpa.alloc(u32, 3 + case);
            defer gpa.free(hist);
            for (hist) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
            const tok = rng.uintLessThan(u32, 512);
            var host: [engram.MAX_LAYERS * engram.MAX_COLS]u32 = undefined;
            hs.rows(hist, &.{map[tok]}, host[0..ctx.n]);
            const mine_host = host[ek * nc ..][0..nc];
            const t = try sc.keep(mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .uint32));
            const mine = try sc.slice(try rows(&sc, &hs, &ctx, hist, t, 0), 0, ek * nc, (ek + 1) * nc);
            // [0] MLX's chain (=0), [1] the served fused wkv, [2] the plant, [3] the rows kernel (ENGRAM_KV_SLOTS=0).
            var kv: [4]A = undefined;
            kv[1] = (try engram_kv.launch(&sc, tab, 1, g[0], g[1], mine, ga, e.wkv.w, e.wkv.s)) orelse return error.Declined;
            {
                const off = env.override(.MLX_SERVE_DSV41_ENGRAM_KV_SLOTS, "0");
                defer off.restore();
                kv[3] = (try engram_kv.launch(&sc, tab, 1, g[0], g[1], mine, ga, e.wkv.w, e.wkv.s)) orelse return error.Declined;
            }
            var moved: [engram.MAX_COLS]u32 = undefined;
            @memcpy(moved[0..nc], mine_host);
            moved[nc - 1] = @intCast((moved[nc - 1] + 1) % n_rows);
            const pa = try sc.keep(mlx.mlx_array_new_data(&moved, &[_]c_int{@intCast(nc)}, 1, .uint32));
            kv[2] = (try engram_kv.launch(&sc, tab, 1, g[0], g[1], pa, ga, e.wkv.w, e.wkv.s)) orelse return error.Declined;
            const raw: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", nc * wlen), .s = try gpa.alignedAlloc(u8, .@"4", nc * slen), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
            defer raw.deinit(gpa);
            for (mine_host, 0..) |id, j| {
                @memcpy(raw.w[j * wlen ..][0..wlen], e.table.held[0][e.table.w_off + id * wlen ..][0..wlen]);
                @memcpy(raw.s[j * slen ..][0..slen], e.table.held[1][e.table.s_off + id * slen ..][0..slen]);
            }
            const xr = try sc.keep(try e.table.dequantTo(raw, nc, .float32, s));
            kv[0] = try dsv41.engramWkv(&sc, &e, try sc.astype(try sc.reshape(xr, &.{ 1, @intCast(K) }), m.act));
            const x = try dsv41.hostF32(xr);
            defer gpa.free(x);
            for (truth, 0..) |*o, n| {
                var acc: f64 = 0;
                for (x, wd[n * K ..][0..K]) |a, b| acc += @as(f64, a) * @as(f64, b);
                o.* = acc;
            }
            var dist: [4]decode.Dist = @splat(.{});
            var hk: [4][]f32 = undefined;
            for (kv, &dist, &hk) |a, *dd, *h| {
                h.* = try dsv41.hostF32(a);
                dd.add(h.*, truth);
            }
            defer for (hk) |h| gpa.free(h);
            var wf: decode.WordDiff = .{};
            wf.add(hk[1], hk[0], truth);
            var wr: decode.WordDiff = .{};
            wr.add(hk[3], hk[0], truth);
            for (&agg, dist) |*a, dd| a.join(dd);
            awf.differ += wf.differ;
            awf.nearer += wf.nearer;
            awf.farther += wf.farther;
            awf.words += wf.words;
            const near = dist[1].within(dist[0]);
            cells += 1;
            caught += @intFromBool(!dist[2].within(dist[0]));
            std.debug.print("[engram-gpu-ids f64] {t} case {d} table {d}: MLX chain l2 {e:.6} max {e:.6}; fused (slots) l2 {e:.6} max {e:.6}, {d} of {d} words differ from the chain's ({d} nearer, {d} farther); rows kernel l2 {e:.6} max {e:.6} ({d} differ, {d} nearer, {d} farther); plant l2 {e:.4} max {e:.4}{s}\n", .{ mode, case, ek, dist[0].l2(), dist[0].max, dist[1].l2(), dist[1].max, wf.differ, wf.words, wf.nearer, wf.farther, dist[3].l2(), dist[3].max, wr.differ, wr.nearer, wr.farther, dist[2].l2(), dist[2].max, if (near) "" else "  (this cell a rounding step farther)" });
        };
    }
    std.debug.print("[engram-gpu-ids f64] all {d} cells: MLX chain l2 {e:.6} max {e:.6}; fused l2 {e:.6} max {e:.6}, {d} of {d} words differ ({d} nearer, {d} farther); rows kernel l2 {e:.6} max {e:.6}; squared error fused / chain - 1 = {e:.3}; the fused wkv {s} in all\n", .{ cells, agg[0].l2(), agg[0].max, agg[1].l2(), agg[1].max, awf.differ, awf.words, awf.nearer, awf.farther, agg[3].l2(), agg[3].max, agg[1].se / agg[0].se - 1, if (agg[1].within(agg[0])) "no farther" else "FARTHER" });
    std.debug.print("[engram-gpu-ids f64] {d} cells: the plant farther in {d}\n", .{ cells, caught });
    // A rounding-class reorder (the release ruling): the worst element held, a word a rounding step either side.
    try testing.expect(agg[1].bad == 0 and agg[1].max <= agg[0].max);
    try testing.expect(@as(f64, @floatFromInt(awf.differ)) <= 1e-4 * @as(f64, @floatFromInt(awf.words)));
    try testing.expectEqual(cells, caught);
}

test "dsv41 exact rows: engram: an exact verify's rows each write the served plain step's Engram (the slot kernel from the host-hashed ids, writeRows) bit for bit, R = 1..6, both tables, mxfp4 and mxfp8 rows; another token's ids differ" {
    if (!gpuOk()) return error.SkipZigTest;
    const fixture = @import("fixture.zig");
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 0, .hi = 1, .ctx = 8, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    const s = mlx.gpuStream();
    const d = m.cfg.hidden_size;
    const hc = m.cfg.dsv4_hc_mult;
    var prng = std.Random.DefaultPrng.init(0xE1DA);
    const rng = prng.random();
    const hs = testHash(rng, 300);
    const nc = hs.cols();
    const hd: usize = 256;
    const ew = try fx.normal(&.{ @intCast((hc + 1) * d), @intCast(nc * hd) }, .bfloat16, 0, 0.02);
    defer _ = mlx.mlx_array_free(ew);
    const eq = try fixture.quantize(ew, 32, 8, "mxfp8");
    defer for (eq) |q| if (q.ctx != null) {
        _ = mlx.mlx_array_free(q);
    };
    const qk = try fx.normal(&.{ @intCast(hc), @intCast(d) }, .float32, 0.5, 0.2);
    defer _ = mlx.mlx_array_free(qk);
    const map = try gpa.alloc(u32, 512);
    defer gpa.free(map);
    for (map) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
    var ctx = try ctxOf(&hs, map);
    defer ctx.deinit();
    const n_rows: usize = @intCast(hs.offsets[1][nc - 1] + hs.primes[1][nc - 1]);
    const MR = decode.max_rows;
    var bad: usize = 0;
    var cases: usize = 0;
    for ([_]@import("../../../sdk/quant/params.zig").QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        var e: dsv41.EngramLayer = .{ .layer = 0, .wkv = .{ .w = eq[0], .s = eq[1], .b = eq[2], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } }, .qk = qk, .table = .{ .fd = -1, .rows = n_rows, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen } };
        try fusion.heldTable(&e.table, n_rows, if (bits == 4) 4096 + 952 else 4096 + 48, rng);
        defer for (e.table.held[0..2]) |mem| std.posix.munmap(mem);
        try testing.expect(fusion.wrapHeld(&e));
        defer {
            _ = mlx.mlx_synchronize(s);
            fusion.unwrap(&e);
        }
        for (1..MR + 1) |R| for (0..2) |ek| {
            var sc = Scope.init(gpa, s);
            defer sc.deinit();
            var hist: [3 + MR]u32 = undefined;
            for (hist[0..3]) |*v| v.* = rng.uintLessThan(u32, 1 << 17);
            var toks: [MR]u32 = undefined;
            for (toks[0..R], hist[3..][0..R]) |*t, *c| {
                t.* = rng.uintLessThan(u32, 512);
                c.* = map[t.*];
            }
            const x = try sc.keep(try fx.normal(&.{ @intCast(R), @intCast(hc), @intCast(d) }, .bfloat16, 0, 1));
            // The verify: the R tokens hashed on the host after the history, each row's slot write from its ids.
            var host: [MR * engram.MAX_LAYERS * engram.MAX_COLS]u32 = undefined;
            hs.rows(hist[0..3], hist[3..][0..R], host[0 .. R * ctx.n]);
            const got = (try writeRows(&sc, m, &e, host[0 .. R * ctx.n], hs.n_layers, 0, R, ek, nc, x)) orelse return error.Declined;
            // The plain steps: each token on the GPU after the history and the run's earlier tokens.
            for (0..R) |r| {
                const xr = try sc.slice(x, 0, r, r + 1);
                const t = try sc.keep(mlx.mlx_array_new_data(&toks[r], &[_]c_int{1}, 1, .uint32));
                const want = (try write(&sc, m, &e, try rows(&sc, &hs, &ctx, hist[0 .. 3 + r], t, 0), .{ .ctx = null }, ek, nc, xr)) orelse return error.Declined;
                const differ = try decode.rowWordsDiffer(got, r, want);
                const other: u32 = (toks[r] + 1) % 512;
                const ot = try sc.keep(mlx.mlx_array_new_data(&other, &[_]c_int{1}, 1, .uint32));
                const plant = (try write(&sc, m, &e, try rows(&sc, &hs, &ctx, hist[0 .. 3 + r], ot, 0), .{ .ctx = null }, ek, nc, xr)) orelse return error.Declined;
                const moved = try decode.rowWordsDiffer(got, r, plant);
                if (differ > 0) std.debug.print("[exact-rows] engram {t} R={d} table {d} row {d}: {d} words differ from the plain step's write\n", .{ mode, R, ek, r, differ });
                bad += @intFromBool(differ > 0);
                try testing.expect(moved > 0);
                cases += 1;
            }
        };
    }
    std.debug.print("[exact-rows] engram: {d} of {d} verify rows differ from their plain step's write\n", .{ bad, cases });
    try testing.expectEqual(@as(usize, 0), bad);
}
