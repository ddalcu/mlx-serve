//! Launch fusions for a resident model, each behind its own switch (`down_scale` and `mxfp8_qmm` on by default, `engram`
//! and `pf_bf16_trunk` off), all of them behind the weight path (`MLX_SERVE_DSV41_WEIGHT_PATH`, `settle`, read at
//! load): `resident` serves them, `ssd` the streamed and staged kernels as before (no fusion engages), `auto` resident
//! when no bank streams and every Engram table is held. A forced `resident` on a streamed or partly held pack is
//! refused by name.
//! - `down_scale`: a decode step or verify puts an NVFP4 bank's per-expert down scale on the routing weights inside the
//!   fused fp4 gate/up launch (`DSCALE`: `wts * dsc[e]`, the f32 product `routeScaled`'s take, cast and multiply make).
//! - `engram`: a decode step's or verify's Engram write takes its wkv projection straight from the table's codes in one
//!   launch (`decode/engram_kv.zig`), then the gate: the held table is one GPU buffer (`wrapHeld`) the kernel gathers
//!   its rows from by id (no host copy, dequant, cast or rows in memory; the rows copied out of the table where the
//!   wrap declines); a prompt chunk's rows dequantize straight to the activation dtype (`Table.dequantTo`), not to f32
//!   and a cast. Not bit-equal: the projection sums in another order.
//! - `mxfp8_qmm`: an mxfp8 linear over a prompt chunk's rows runs MLX's quantized matmul (its NAX qmm dequantizes the
//!   codes in the GEMM's tile loads) instead of dequantizing the whole weight to bf16 and the dense GEMM
//!   (`Lin.denseAt`, through `ps()`: `Lin` has no model).
//! - `pf_bf16_trunk`: the trunk's mxfp8 linears dequantized to bf16 once at load (`bf16Trunk`, `Lin.pf`), the copy
//!   every prompt chunk's dense GEMM reads instead of a dequantize per chunk; decode keeps the mxfp8 weights. The
//!   same bf16 values the per-chunk dequantize makes (e4m3 times a power of two is exact in bf16): the same bytes.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const dsv41 = @import("deepseek_v41.zig");
const dkern = @import("decode/decode.zig");
const engram = @import("engram.zig");
const QuantMode = @import("../../sdk/quant/params.zig").QuantMode;

const A = mlx.mlx_array;
const none: A = .{ .ctx = null };

pub const Fuse = struct {
    down_scale: bool = false,
    engram: bool = false,
    mxfp8_qmm: bool = false,
    pf_bf16_trunk: bool = false,

    pub fn wanted() Fuse {
        return .{
            .down_scale = env.onUnlessZero(.MLX_SERVE_DSV41_FUSE_DOWN_SCALE),
            .engram = env.on(.MLX_SERVE_DSV41_FUSE_ENGRAM),
            .mxfp8_qmm = env.onUnlessZero(.MLX_SERVE_DSV41_FUSE_MXFP8_QMM),
            .pf_bf16_trunk = env.on(.MLX_SERVE_DSV41_PF_BF16_TRUNK),
        };
    }
};

/// What `Lin.denseAt` reads (it has no model), one copy across `-Dunits` units.
const State = struct { mxfp8_qmm: bool = false };
pub const shared_state = @import("../../sdk/util/shared.zig").Shared(State, "dsv41_fuse", .{});
pub inline fn ps() *State {
    return shared_state.get();
}

/// `MLX_SERVE_DSV41_WEIGHT_PATH`.
pub const Path = enum { auto, resident, ssd };

/// The setting: unset or empty `auto`; anything but the three names refused by name.
pub fn pathOf(v: ?[]const u8) error{Dsv41WeightPathSetting}!Path {
    const t = std.mem.trim(u8, v orelse return .auto, " ");
    if (t.len == 0) return .auto;
    return std.meta.stringToEnum(Path, t) orelse error.Dsv41WeightPathSetting;
}

/// The kernels a load serves and why.
pub const Mode = struct { resident: bool, why: []const u8 };

/// `p` on a model whose experts stream from disk (`streamed`) or whose Engram tables are not all held (`held`).
pub fn choose(p: Path, streamed: bool, held: bool) error{ Dsv41WeightPathResidentStreamed, Dsv41WeightPathResidentEngram }!Mode {
    switch (p) {
        .ssd => return .{ .resident = false, .why = "MLX_SERVE_DSV41_WEIGHT_PATH=ssd" },
        .resident => {
            if (streamed) return error.Dsv41WeightPathResidentStreamed;
            if (!held) return error.Dsv41WeightPathResidentEngram;
            return .{ .resident = true, .why = "MLX_SERVE_DSV41_WEIGHT_PATH=resident" };
        },
        .auto => {
            if (streamed) return .{ .resident = false, .why = "auto: the routed experts stream from disk" };
            if (!held) return .{ .resident = false, .why = "auto: the Engram tables are read from disk per lookup" };
            return .{ .resident = true, .why = "auto: every bank and Engram table is resident" };
        },
    }
}

/// At load, once the Engram tables are held or not (`holdEngram`): the weight path, the fusions it lets serve, and
/// under `engram` each held table wrapped as a GPU buffer; one line names them.
pub fn settle(m: *dsv41.Dsv41Model) !void {
    const p = try pathOf(env.span(.MLX_SERVE_DSV41_WEIGHT_PATH));
    var held = true;
    for (m.engrams) |*e| held = held and e.table.isHeld();
    const mode = try choose(p, m.cfg.dsv41_streamed, held);
    m.fuse = if (mode.resident) Fuse.wanted() else .{};
    ps().mxfp8_qmm = m.fuse.mxfp8_qmm;
    var wrapped: usize = 0;
    var bytes: u64 = 0;
    if (m.fuse.engram) for (m.engrams) |*e| {
        if (!wrapHeld(e)) continue;
        wrapped += 1;
        bytes += e.table.held[0].len + e.table.held[1].len;
    };
    log.info("deepseek_v41: weight path {s} ({s}); fusions down_scale={} engram={} ({d} of {d} tables gathered on the GPU) mxfp8_qmm={}\n", .{ if (mode.resident) "resident" else "ssd", mode.why, m.fuse.down_scale, m.fuse.engram, wrapped, m.engrams.len, m.fuse.mxfp8_qmm });
    if (m.fuse.pf_bf16_trunk) {
        const n = try bf16Trunk(m);
        log.info("deepseek_v41: pf bf16 trunk: {d:.1} GB, {d} mxfp8 linears dequantized once for the prompt's GEMMs (billed {d:.1} GB)\n", .{ gbOf(n.bytes), n.linears, gbOf(trunkBytes(&m.cfg)) });
    }
    if (wrapped > 0) {
        var active: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        const ws = mlx.maxRecommendedWorkingSet();
        const room = evictRoom(active, ws);
        log.info("deepseek_v41: engram: {d:.1} GB wrapped no-copy (pages already held, no new memory) now count in MLX's active memory, {d:.1} GB with them; MLX evicts its buffer cache on each miss past 0.95 x the working set, {d:.1} GB: {d:.1} GB of room{s}\n", .{ gbOf(bytes), gbOf(active), gbOf(evictLine(ws)), gbOf(room), if (room == 0) " (none: expect slower allocations)" else "" });
    }
}

/// The trunk's mxfp8 linears a prompt chunk's dense GEMMs read, of layer `ly` (unset ones skipped by the caller).
fn trunkLins(ly: *dsv41.Layer) [10]?*dsv41.Lin {
    var out: [10]?*dsv41.Lin = @splat(null);
    const all = [_]*dsv41.Lin{ &ly.wq_a, &ly.wq_b, &ly.wkv, &ly.wo_a, &ly.wo_b, &ly.shared[0], &ly.shared[1], &ly.shared[2] };
    for (all, 0..) |l, i| out[i] = l;
    if (ly.q_kv) |*q| out[8] = q;
    if (ly.shared_gu) |*g| out[9] = g;
    return out;
}

fn trunkLin(l: *const dsv41.Lin) bool {
    return l.ext == null and l.w.ctx != null and l.s.ctx != null and l.qp.mode == .mxfp8 and l.qp.bits == 8 and l.qp.group_size == 32;
}

/// Each held layer's mxfp8 trunk linears (attention, the indexer's wq_b, the shared expert) and each Engram wkv
/// dequantized to bf16 once (`Lin.pf`), evaluated a layer at a time.
pub fn bf16Trunk(m: *dsv41.Dsv41Model) !struct { bytes: u64, linears: usize } {
    var bytes: u64 = 0;
    var n: usize = 0;
    for (0..m.n_layers) |li| {
        if (!m.held(li)) continue;
        const ly = &m.layers[li];
        var lins: [11]?*dsv41.Lin = undefined;
        lins[0..10].* = trunkLins(ly);
        lins[10] = if (ly.idx) |*ix| &ix.wq_b else null;
        for (lins) |maybe| {
            const l = maybe orelse continue;
            if (!trunkLin(l) or l.pf.ctx != null) continue;
            l.pf = try m.own(try plainDense(l, m.s));
            bytes += 2 * mlx.mlx_array_size(l.pf);
            n += 1;
        }
        try evalOwned(m);
    }
    for (m.engrams) |*e| if (trunkLin(&e.wkv) and e.wkv.pf.ctx == null) {
        e.wkv.pf = try m.own(try plainDense(&e.wkv, m.s));
        bytes += 2 * mlx.mlx_array_size(e.wkv.pf);
        n += 1;
    };
    try evalOwned(m);
    return .{ .bytes = bytes, .linears = n };
}

/// `l` dequantized to bf16, a fresh array the caller owns.
fn plainDense(l: *const dsv41.Lin, s: mlx.mlx_stream) !A {
    var d = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_dequantize(&d, l.w, l.s, l.b, mlx.mlx_optional_int.some(@intCast(l.qp.group_size)), mlx.mlx_optional_int.some(@intCast(l.qp.bits)), l.qp.mode.cstr(), none, .{ .value = .bfloat16, .has_value = true }, s));
    return d;
}

/// The model's owned arrays made since the last call evaluated (the bf16 copies, so none waits on the first prompt).
fn evalOwned(m: *dsv41.Dsv41Model) !void {
    const items = m.owned.items;
    const vec = mlx.mlx_vector_array_new_data(items.ptr, items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// What `bf16Trunk` holds for config `c`, in bytes: two per weight of each layer's mxfp8 trunk linears (wq_a, wkv,
/// wq_b, wo_a, wo_b, the shared expert's three, an index source layer's wq_b) and each Engram wkv. Billed at load
/// under the switch (`bill.zig`).
pub fn trunkBytes(c: anytype) u64 {
    var w: u64 = 0;
    for (0..c.num_hidden_layers) |li| w += layerWeights(c, li);
    if (c.dsv41_n_engram_layers > 0) {
        const cols: u64 = (@as(u64, c.dsv41_engram_max_ngram) - 1) * c.dsv41_engram_n_heads;
        w += @as(u64, c.dsv41_n_engram_layers) * (c.dsv4_hc_mult + 1) * c.hidden_size * cols * c.dsv41_engram_head_dim;
    }
    return 2 * w;
}

/// Layer `li`'s mxfp8 trunk weights (`trunkBytes`).
pub fn layerWeights(c: anytype, li: usize) u64 {
    const d: u64 = c.hidden_size;
    const hd: u64 = c.head_dim;
    const ql: u64 = c.dsv4_q_lora_rank;
    const H: u64 = c.num_attention_heads;
    const og: u64 = c.dsv4_o_groups;
    const ol: u64 = c.dsv4_o_lora_rank;
    var w = ql * d + hd * d + H * hd * ql + og * ol * (H * hd / og) + d * og * ol + 3 * @as(u64, c.moe_intermediate_size) * d;
    if ((c.dsv41_index_sources >> @intCast(li)) & 1 != 0) w += @as(u64, c.dsv4_index_n_heads) * c.dsv4_index_head_dim * ql;
    return w;
}

/// MLX's line (`gc_limit_`, min(1.5 x, 0.95 x) the recommended working set `ws`): past it every new buffer first evicts
/// cached ones. Its memory limit cannot move it higher, so wrapped bytes are only room taken.
pub fn evictLine(ws: u64) u64 {
    return ws / 20 * 19;
}

/// What is left under `evictLine` with `active` bytes counted.
pub fn evictRoom(active: u64, ws: u64) u64 {
    return evictLine(ws) -| active;
}

fn gbOf(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / 1e9;
}

/// MLX's deleter for a wrapped region: the table owns the memory. `armed` stays false until the wrap proved the
/// buffer is the region itself (MLX copies, and calls the deleter at once, where Metal refuses the no-copy buffer).
const Wrap = struct { armed: bool = false, fired: bool = false };

fn onRelease(payload: ?*anyopaque) callconv(.c) void {
    const w: *Wrap = @ptrCast(@alignCast(payload.?));
    if (!w.armed) {
        w.fired = true;
        return;
    }
    std.heap.page_allocator.destroy(w);
}

/// `mem` (page-aligned, a page multiple) as one no-copy uint8 `[pages, 4096]` array; null where MLX would copy it.
fn wrapRegion(mem: []align(std.heap.page_size_min) u8) !?A {
    const row: usize = 4096;
    if (mem.len == 0 or mem.len % row != 0 or mem.len / row > std.math.maxInt(c_int) or mem.len > mlx.maxBufferLength()) return null;
    const w = try std.heap.page_allocator.create(Wrap);
    w.* = .{};
    const arr = mlx.mlx_array_new_data_managed_payload(@ptrCast(mem.ptr), &[_]c_int{ @intCast(mem.len / row), @intCast(row) }, 2, .uint8, w, onRelease);
    if (arr.ctx == null or w.fired) {
        if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
        std.heap.page_allocator.destroy(w);
        return null;
    }
    w.armed = true;
    return arr;
}

/// `e`'s held codes and scales as GPU buffers (`EngramLayer.gpu`), where the kernel can gather from them: a held
/// mxfp4 or mxfp8 table of separate code and scale regions, rows aligned to the kernels' code loads (8 bytes for e2m1,
/// 16 for e4m3), each region within Metal's largest
/// buffer. Logs why it declines.
pub fn wrapHeld(e: *dsv41.EngramLayer) bool {
    const t = &e.table;
    const why: []const u8 = blk: {
        if (!t.isHeld()) break :blk "not held";
        if (t.records() or t.held[1].len == 0 or !t.dequantsTo()) break :blk "not separate mxfp4/mxfp8 code and scale regions";
        const page = std.heap.pageSize();
        // A lane loads 16 codes at once: a uint2 of e2m1, a uint4 of e4m3.
        const al: u64 = if (t.bits == 4) 8 else 16;
        if ((t.w_off % page) % al != 0 or t.w_stride % al != 0) break :blk "code rows not aligned to the kernels' loads";
        var arrs: [2]A = undefined;
        for (&arrs, t.held[0..2]) |*a, mem| a.* = (wrapRegion(mem) catch null) orelse {
            if (a != &arrs[0]) _ = mlx.mlx_array_free(arrs[0]);
            break :blk "Metal refused a no-copy buffer (or it passes the largest buffer)";
        };
        e.gpu = arrs;
        return true;
    };
    log.info("deepseek_v41: Engram layer {d}'s table stays on the host copy ({s})\n", .{ e.layer, why });
    return false;
}

/// `e`'s GPU buffers let go (before the table unmaps what they wrap).
pub fn unwrap(e: *dsv41.EngramLayer) void {
    const g = e.gpu orelse return;
    for (g) |a| _ = mlx.mlx_array_free(a);
    e.gpu = null;
}

/// The fusions that have served, each logged the first time (an A/B arm's engagement lines).
pub var engaged: struct { down_scale: bool = false, engram: bool = false, engram_held: bool = false, engram_rows: bool = false, engram_rows_held: bool = false, mxfp8_qmm: bool = false } = .{};

fn engage(comptime which: []const u8, comptime what: []const u8) void {
    if (@field(engaged, which)) return;
    @field(engaged, which) = true;
    log.info("deepseek_v41: fused " ++ what ++ " engaged\n", .{});
}

/// `ffnDecode` over routes whose weights `wts` do not hold the layer's down scale: the fused fp4 launches with the
/// scale applied inside the gate/up launch. Null where the layer has no down scale or a launch declines; the caller
/// then scales the route.
pub fn ffnDownScaled(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, ly: *const dsv41.Layer, h: A, idx: A, wts: A, x1: A, sp: *const dsv41.Split) !?A {
    const ds = ly.experts.scale[2] orelse return null;
    const kf = dsv41.fp4Decode(m) orelse return null;
    const act = try dkern.moe_fp4.moeUpScaled(sc, m, kf, ly, h, idx, wts, .{ .gate = ly.experts.scale[0], .up = ly.experts.scale[1], .down = ds }) orelse return null;
    const y = try dkern.moe_fp4.moeDown(sc, kf, ly, act, idx, x1, sp) orelse return null;
    engage("down_scale", "down scale (an NVFP4 bank's per-expert down scale inside the fp4 gate/up launch)");
    return y;
}

/// An Engram layer's kv `[rows, (hc + 1) * dim]` from the table's raw rows `raw` (`rows * K / head_dim` of them, a
/// row's n-gram rows in order) on the one-launch kernel. Null where it does not serve: the table not mxfp4 or mxfp8 at
/// group 32, wkv not mxfp8 group 32 in MLX arrays, activations not bf16, more rows than a decode step.
pub fn engramKv(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.EngramLayer, raw: engram.Table.Raw, rows: usize) !?A {
    if (!kvServes(m, e, rows)) return null;
    const t = &e.table;
    const K: usize = @intCast(mlx.getShape(e.wkv.w)[1] * 4);
    if (K % t.head_dim != 0) return null;
    const n = rows * K / t.head_dim;
    if (raw.w.len < n * t.wlen or raw.s.len < n * t.slen) return null;
    const codes = try sc.keep(mlx.mlx_array_new_data(raw.w.ptr, &[_]c_int{ ci(n), ci(t.wlen / 4) }, 2, .uint32));
    const tsc = try sc.keep(mlx.mlx_array_new_data(raw.s.ptr, &[_]c_int{ ci(n), ci(t.slen) }, 2, .uint8));
    var seq: [dkern.max_rows * engram.MAX_COLS]u32 = undefined;
    if (n > seq.len) return null;
    for (seq[0..n], 0..) |*v, i| v.* = @intCast(i);
    const geo: dkern.engram_kv.Geo = .{ .w_base = 0, .w_stride = t.wlen, .s_base = 0, .s_stride = t.slen };
    return kvLaunch(sc, e, rows, codes, tsc, seq[0..n], geo);
}

/// The kernel over `ids` (`rows * K / head_dim` table rows) in the byte buffers `codes` and `tsc` at `geo`.
fn kvLaunch(sc: *dsv41.Scope, e: *const dsv41.EngramLayer, rows: usize, codes: A, tsc: A, ids: []const u32, geo: dkern.engram_kv.Geo) !?A {
    const t = &e.table;
    const ia = try sc.keep(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ci(ids.len)}, 1, .uint32));
    const gw = geo.words();
    const ga = try sc.keep(mlx.mlx_array_new_data(&gw, &[_]c_int{6}, 1, .uint32));
    return dkern.engram_kv.launch(sc, .{ .hd = t.head_dim, .tb = t.bits }, rows, codes, tsc, ia, ga, e.wkv.w, e.wkv.s);
}

/// Whether `engramKv`'s kernel reads `e`'s table and projection.
pub fn kvServes(m: *const dsv41.Dsv41Model, e: *const dsv41.EngramLayer, rows: usize) bool {
    const t = &e.table;
    const l = &e.wkv;
    if (m.act != .bfloat16 or rows < 1 or rows > dkern.max_rows) return false;
    if (!((t.mode == .mxfp4 and t.bits == 4) or (t.mode == .mxfp8 and t.bits == 8)) or t.group_size != 32) return false;
    if (l.ext != null or l.s.ctx == null or l.b.ctx != null or l.qp.mode != .mxfp8 or l.qp.bits != 8 or l.qp.group_size != 32) return false;
    return mlx.getShape(l.w).len == 2 and t.head_dim > 0 and t.wlen == t.head_dim * t.bits / 8 and t.slen == t.head_dim / 32;
}

/// Under `engram` with the table on the GPU (`wrapHeld`): the Engram write of span rows [lo, hi) (`x`
/// `[hi - lo, hc, dim]`) gathered by id straight out of the resident table, `ids` the chunk's row ids
/// (`[rows][tables][nc]`, `Hash.rows`), this table the `ek`-th. Null where it does not serve (an id past the table:
/// an image row's), and the caller reads the rows as before.
pub fn engramHeld(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.EngramLayer, ids: []const u32, tables: usize, ek: usize, nc: usize, lo: usize, hi: usize, x: A) !?A {
    if (!m.fuse.engram) return null;
    const g = e.gpu orelse return null;
    const rows = hi - lo;
    if (!kvServes(m, e, rows)) return null;
    const t = &e.table;
    const K: usize = @intCast(mlx.getShape(e.wkv.w)[1] * 4);
    if (K != nc * t.head_dim or hi * tables * nc > ids.len) return null;
    var mine: [dkern.max_rows * engram.MAX_COLS]u32 = undefined;
    if (rows * nc > mine.len) return null;
    for (lo..hi) |r| for (0..nc) |j| {
        const id = ids[(r * tables + ek) * nc + j];
        if (id >= t.rows) return null;
        mine[(r - lo) * nc + j] = id;
    };
    const page = std.heap.pageSize();
    const geo: dkern.engram_kv.Geo = .{ .w_base = t.w_off % page, .w_stride = t.w_stride, .s_base = t.s_off % page, .s_stride = t.s_stride };
    const kv = try kvLaunch(sc, e, rows, g[0], g[1], mine[0 .. rows * nc], geo) orelse return null;
    engage("engram_held", "Engram write (the wkv projection gathering its rows from the resident table in one launch)");
    return try dsv41.engramGate(sc, m, e.qk, x, kv);
}

/// Under `engram`: the Engram write of the stream `x` `[rows, hc, dim]` from the table's raw rows (`engramKv`, then the
/// gate). Null where the switch is off or the kernel declines; the caller then dequantizes the rows.
pub fn engramWrite(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.EngramLayer, raw: engram.Table.Raw, x: A) !?A {
    if (!m.fuse.engram) return null;
    const kv = try engramKv(sc, m, e, raw, @intCast(mlx.getShape(x)[0])) orelse return null;
    engage("engram", "Engram write (the wkv projection from the table's codes in one launch)");
    return try dsv41.engramGate(sc, m, e.qk, x, kv);
}

/// Under `engram` with the table on the GPU (`wrapHeld`): a prompt span's Engram rows [lo, hi) `[(hi - lo) * nc, hd]`
/// bf16, gathered by id out of the resident table and dequantized in one launch (`engram_kv.gatherRows`), the wkv
/// GEMM's input with no host read, copy or upload. Null where it does not serve (an id past the table).
pub fn engramRowsHeld(sc: *dsv41.Scope, m: *const dsv41.Dsv41Model, e: *const dsv41.EngramLayer, ids: []const u32, tables: usize, ek: usize, nc: usize, lo: usize, hi: usize) !?A {
    if (!m.fuse.engram or m.act != .bfloat16) return null;
    const g = e.gpu orelse return null;
    const t = &e.table;
    if (hi <= lo or hi * tables * nc > ids.len) return null;
    const n = (hi - lo) * nc;
    const mine = try sc.a.alloc(u32, n);
    defer sc.a.free(mine);
    for (lo..hi) |r| for (0..nc) |j| {
        const id = ids[(r * tables + ek) * nc + j];
        if (id >= t.rows) return null;
        mine[(r - lo) * nc + j] = id;
    };
    const page = std.heap.pageSize();
    const geo: dkern.engram_kv.Geo = .{ .w_base = t.w_off % page, .w_stride = t.w_stride, .s_base = t.s_off % page, .s_stride = t.s_stride };
    const ia = try sc.keep(mlx.mlx_array_new_data(mine.ptr, &[_]c_int{ci(n)}, 1, .uint32));
    const gw = geo.words();
    const ga = try sc.keep(mlx.mlx_array_new_data(&gw, &[_]c_int{6}, 1, .uint32));
    const rows = try dkern.engram_kv.gatherRows(sc, .{ .hd = t.head_dim, .tb = t.bits }, g[0], g[1], ia, ga) orelse return null;
    engage("engram_rows_held", "Engram rows (a prompt span's rows gathered from the resident table on the GPU)");
    return rows;
}

/// The dtype the table `t`'s rows dequantize into: the activations' under `engram` where the codes allow it.
pub fn engramRowsDtype(m: *const dsv41.Dsv41Model, t: *const engram.Table) mlx.mlx_dtype {
    if (!m.fuse.engram or !t.dequantsTo()) return .float32;
    engage("engram_rows", "Engram rows (a prompt chunk's rows straight to the activation dtype)");
    return m.act;
}

/// Whether an `mode` linear over a prompt chunk's rows stays on MLX's quantized matmul (`Lin.denseAt`): mxfp8 under
/// `mxfp8_qmm` (`settle`).
pub fn mxfp8Qmm(mode: QuantMode) bool {
    if (mode != .mxfp8 or !ps().mxfp8_qmm) return false;
    engage("mxfp8_qmm", "mxfp8 prompt linears (MLX's quantized matmul, no bf16 weight)");
    return true;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("decode/fixture.zig");

fn free(a: A) void {
    _ = mlx.mlx_array_free(a);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

/// Elements whose bits differ (f32 of bf16 or f32 values, exact either way). The reference must be finite: two equal
/// NaN rows would pass vacuously.
fn differing(got: A, want: A) !usize {
    const g = try dsv41.hostF32(got);
    defer testing.allocator.free(g);
    const w = try dsv41.hostF32(want);
    defer testing.allocator.free(w);
    if (g.len != w.len) return error.ShapeMismatch;
    var n: usize = 0;
    for (g, w) |x, y| {
        if (!std.math.isFinite(y)) return error.NonFinite;
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) n += 1;
    }
    return n;
}

test "dsv41 fuse down_scale: the fp4 launches over unscaled routes, the down scale inside, give ffnDecode's bytes over the scaled routes, R = 1..6 (NVFP4 with per-expert scales); the gate's scales in its place differ" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const sw = env.override(.MLX_SERVE_DSV41_MOE_FP4_NAX, "1");
    defer sw.restore();
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .experts = 16, .ctx = 0, .experts_nvfp4 = true, .expert_scales = .all, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(20);
    var bare = ly.*;
    bare.experts.scale = .{ null, null, null };
    var wrong = ly.*;
    wrong.experts.scale[2] = ly.experts.scale[0];
    var sc = dsv41.Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const MR = dkern.max_rows;
    const hc = m.cfg.dsv4_hc_mult;
    const h = try sc.keep(try fx.normal(&.{ ci(MR), ci(m.cfg.hidden_size) }, .bfloat16, 0, 1));
    const raw = try dsv41.route(&sc, m, &bare, h, none);
    const scaled = try dsv41.route(&sc, m, ly, h, none);
    const x1 = try sc.keep(try fx.stream(MR));
    const post = try sc.keep(try fx.uniform(&.{ ci(MR), ci(hc) }, 0.05, 1.5));
    const comb = try sc.keep(try fx.uniform(&.{ ci(MR), ci(hc), ci(hc) }, -0.5, 0.5));
    var differ: usize = 0;
    var plants_equal: usize = 0;
    for (1..MR + 1) |R| {
        const sp: dsv41.Split = .{ .pre = none, .post = try sc.slice(post, 0, 0, R), .comb = try sc.slice(comb, 0, 0, R) };
        const hr = try sc.slice(h, 0, 0, R);
        const xr = try sc.slice(x1, 0, 0, R);
        const idx = try sc.slice(scaled.idx, 0, 0, R);
        const want = (try dsv41.ffnDecode(&sc, m, ly, hr, idx, try sc.slice(scaled.wts, 0, 0, R), xr, &sp)) orelse return error.Declined;
        const got = (try ffnDownScaled(&sc, m, ly, hr, idx, try sc.slice(raw.wts, 0, 0, R), xr, &sp)) orelse return error.Declined;
        const plant = (try ffnDownScaled(&sc, m, &wrong, hr, idx, try sc.slice(raw.wts, 0, 0, R), xr, &sp)) orelse return error.Declined;
        const d = try differing(got, want);
        const p = try differing(plant, want);
        std.debug.print("[fuse] down_scale R={d}: {d} elements differ from the scaled route's; with the gate's scales as the down's, {d}\n", .{ R, d, p });
        differ += d;
        if (p == 0) plants_equal += 1;
    }
    try testing.expectEqual(@as(usize, 0), differ);
    try testing.expectEqual(@as(usize, 0), plants_equal);
}

test "dsv41 fuse down_scale: decode steps of 1..6 rows through two layers give the control's streams and mixes bit for bit with the fusion on (NVFP4 with per-expert scales; MXFP4, where it stays inert)" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const sw = env.override(.MLX_SERVE_DSV41_MOE_FP4_NAX, "1");
    defer sw.restore();
    var bad: usize = 0;
    for ([_]bool{ true, false }) |nv| {
        const opts: fixture.Opts = .{ .lo = 20, .hi = 22, .experts = 16, .ctx = 300, .seed = 5, .experts_fp4 = !nv, .experts_nvfp4 = nv, .expert_scales = if (nv) .all else .none, .fused = true };
        const a = try fixture.Fixture.init(gpa, opts);
        defer a.deinit();
        const b = try fixture.Fixture.init(gpa, opts);
        defer b.deinit();
        b.m.fuse.down_scale = true;
        engaged.down_scale = false;
        var p0: usize = opts.ctx;
        for ([_]usize{ 1, 2, 3, 4, 5, 6, 1 }) |R| {
            const x = try a.stream(R);
            defer free(x);
            const pre = try a.cpre(R);
            defer free(pre);
            const oa = try dsv41.workerStep(a.m, gpa, x, pre, p0, false, false);
            defer for (oa) |o| free(o);
            const ob = try dsv41.workerStep(b.m, gpa, x, pre, p0, false, false);
            defer for (ob) |o| free(o);
            const ds = try differing(ob[0], oa[0]);
            const dp = try differing(ob[1], oa[1]);
            std.debug.print("[fuse] down_scale {s} R={d} at {d}: stream {d}, mix {d} elements differ\n", .{ if (nv) "nvfp4+scales" else "mxfp4", R, p0, ds, dp });
            bad += ds + dp;
            p0 += R;
        }
        try testing.expectEqual(nv, engaged.down_scale);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}

/// Random table rows: e2m1 codes (any nibble) or e4m3 bytes (never the NaN code), e8m0 scales 2^-7 .. 2^-1.
fn rawRows(gpa: std.mem.Allocator, t: *const engram.Table, n: usize, rng: std.Random) !engram.Table.Raw {
    const raw: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", n * t.wlen), .s = try gpa.alignedAlloc(u8, .@"4", n * t.slen), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
    for (raw.w) |*b| b.* = if (t.bits == 4) rng.int(u8) else @as(u8, rng.uintLessThan(u8, 2)) << 7 | (2 + rng.uintLessThan(u8, 12)) << 3 | rng.uintLessThan(u8, 8);
    for (raw.s) |*b| b.* = 120 + rng.uintLessThan(u8, 7);
    return raw;
}

fn dupRaw(gpa: std.mem.Allocator, r: engram.Table.Raw) !engram.Table.Raw {
    const out: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", r.w.len), .s = try gpa.alignedAlloc(u8, .@"4", r.s.len), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
    @memcpy(out.w, r.w);
    @memcpy(out.s, r.s);
    return out;
}

fn hostOf(a: A) ![]f32 {
    return dsv41.hostF32(a);
}

test "dsv41 fuse engram: the one-launch wkv from the table's codes is no further from f32 truth than the dequant, cast and quantized_matmul chain, and the gated write within 0.1 rel rms of the chain's, R = 1..6, mxfp4 and mxfp8 tables at the release's shape; a scale a code up, two rows swapped differ" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 0, .hi = 1, .ctx = 8, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    m.fuse.engram = true;
    defer m.fuse.engram = false;
    const s = mlx.gpuStream();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    const d = m.cfg.hidden_size;
    const hc = m.cfg.dsv4_hc_mult;
    const hd: usize = 256;
    const K: usize = 24 * hd;
    const N = (hc + 1) * d;
    const ew = try fx.normal(&.{ ci(N), ci(K) }, .bfloat16, 0, 0.02);
    defer free(ew);
    const eq = try fixture.quantize(ew, 32, 8, "mxfp8");
    defer for (eq) |q| if (q.ctx != null) free(q);
    var wd = mlx.mlx_array_new();
    defer free(wd);
    try mlx.check(mlx.mlx_dequantize(&wd, eq[0], eq[1], none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .float32, .has_value = true }, s));
    var wdt = mlx.mlx_array_new();
    defer free(wdt);
    try mlx.check(mlx.mlx_transpose(&wdt, wd, s));
    const qk = try fx.normal(&.{ ci(hc), ci(d) }, .float32, 0.5, 0.2);
    defer free(qk);
    var prng = std.Random.DefaultPrng.init(0xE6);
    const rng = prng.random();
    var sc = dsv41.Scope.init(gpa, s);
    defer sc.deinit();
    for ([_]QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        const t: engram.Table = .{ .fd = -1, .rows = 1 << 20, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen };
        const e: dsv41.EngramLayer = .{ .layer = 0, .wkv = .{ .w = eq[0], .s = eq[1], .b = eq[2], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } }, .qk = qk, .table = t };
        for (1..dkern.max_rows + 1) |R| {
            const n = R * K / hd;
            const raw = try rawRows(gpa, &t, n, rng);
            defer raw.deinit(gpa);
            const rows = try sc.keep(try t.dequant(raw, n, s));
            const r2 = try sc.reshape(rows, &.{ ci(R), ci(K) });
            const got = (try engramKv(&sc, m, &e, raw, R)) orelse return error.Declined;
            const chain = try e.wkv.apply(&sc, try sc.astype(r2, .bfloat16));
            var truth = mlx.mlx_array_new();
            defer free(truth);
            try mlx.check(mlx.mlx_matmul(&truth, r2, wdt, cpu));
            const g = try hostOf(got);
            defer gpa.free(g);
            const c = try hostOf(chain);
            defer gpa.free(c);
            const tr = try hostOf(truth);
            defer gpa.free(tr);
            var label_buf: [64]u8 = undefined;
            try dkern.expectNoWorse(try std.fmt.bufPrint(&label_buf, "fuse engram kv {t} R={d}", .{ mode, R }), g, c, tr);
            // Plants: every scale a code up (the rows doubled); rows 0 and 1 swapped.
            const up = try dupRaw(gpa, raw);
            defer up.deinit(gpa);
            for (up.s) |*b| b.* += 1;
            const pu = try hostOf((try engramKv(&sc, m, &e, up, R)).?);
            defer gpa.free(pu);
            const eu = dkern.relRms(pu, c);
            var es: f64 = 1;
            if (R >= 2) {
                const sw = try dupRaw(gpa, raw);
                defer sw.deinit(gpa);
                const per = K / hd;
                @memcpy(sw.w[0 .. per * wlen], raw.w[per * wlen ..][0 .. per * wlen]);
                @memcpy(sw.w[per * wlen ..][0 .. per * wlen], raw.w[0 .. per * wlen]);
                @memcpy(sw.s[0 .. per * slen], raw.s[per * slen ..][0 .. per * slen]);
                @memcpy(sw.s[per * slen ..][0 .. per * slen], raw.s[0 .. per * slen]);
                const pw = try hostOf((try engramKv(&sc, m, &e, sw, R)).?);
                defer gpa.free(pw);
                es = dkern.relRms(pw, c);
            }
            // The gated write against the chain's (`engramApply` over the dequantized rows), on the increment.
            const x = try sc.keep(try fx.normal(&.{ ci(R), ci(hc), ci(d) }, .bfloat16, 0, 1));
            engaged.engram = false;
            const yg = (try engramWrite(&sc, m, &e, raw, x)) orelse return error.Declined;
            try testing.expect(engaged.engram);
            const yw = try dsv41.engramApply(&sc, m, &e, x, r2);
            const xh = try hostOf(x);
            defer gpa.free(xh);
            const ygh = try hostOf(yg);
            defer gpa.free(ygh);
            const ywh = try hostOf(yw);
            defer gpa.free(ywh);
            for (ygh, ywh, xh) |*a, *b, xv| {
                a.* -= xv;
                b.* -= xv;
            }
            const ew_ = dkern.relRms(ygh, ywh);
            std.debug.print("[fuse] engram {t} R={d}: write increment {e:.2} from the chain's (bar 0.1); plants: scales up {e:.2}, rows swapped {e:.2} (bar > 0.5)\n", .{ mode, R, ew_, eu, es });
            try testing.expect(ew_ <= 0.1);
            try testing.expect(eu > 0.5 and es > 0.5);
        }
    }
}

test "dsv41 fuse engram: a prompt chunk's mxfp4 and mxfp8 rows dequantize straight to bf16 as the f32 rows cast to bf16, every code under every e8m0 scale; the scales a code up differ" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    const hd: usize = 256;
    const n: usize = 256;
    for ([_]QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        const t: engram.Table = .{ .fd = -1, .rows = n, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen };
        try testing.expect(t.dequantsTo());
        // Row r: e8m0 code r on each of its groups; element j code j mod 2^bits, so each group runs through every code.
        const raw: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", n * wlen), .s = try gpa.alignedAlloc(u8, .@"4", n * slen), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
        defer raw.deinit(gpa);
        @memset(raw.w, 0);
        for (0..n) |r| {
            @memset(raw.s[r * slen ..][0..slen], @intCast(r));
            for (0..hd) |j| {
                const c: u8 = @intCast(j % (@as(usize, 1) << @intCast(bits)));
                if (bits == 8) raw.w[r * wlen + j] = c else raw.w[r * wlen + j / 2] |= c << @intCast(4 * (j % 2));
            }
        }
        const rows = try t.dequant(raw, n, s);
        defer free(rows);
        var want = mlx.mlx_array_new();
        defer free(want);
        try mlx.check(mlx.mlx_astype(&want, rows, .bfloat16, s));
        const got = try t.dequantTo(raw, n, .bfloat16, s);
        defer free(got);
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(got));
        for (raw.s) |*c| c.* +%= 1;
        const plant = try t.dequantTo(raw, n, .bfloat16, s);
        defer free(plant);
        const wb = try dkern.moe_up.bitsOf(want);
        defer gpa.free(wb);
        const gb = try dkern.moe_up.bitsOf(got);
        defer gpa.free(gb);
        const pb = try dkern.moe_up.bitsOf(plant);
        defer gpa.free(pb);
        var differ: usize = 0;
        var planted: usize = 0;
        for (wb, gb, pb) |w, g, p| {
            const w_nan = (w & 0x7f80) == 0x7f80 and (w & 0x7f) != 0;
            const g_nan = (g & 0x7f80) == 0x7f80 and (g & 0x7f) != 0;
            if (w != g and !(w_nan and g_nan)) differ += 1;
            if (p != w) planted += 1;
        }
        std.debug.print("[fuse] engram rows {t}: {d} of {d} bf16 values differ from the f32 rows cast; with the scales a code up, {d}\n", .{ mode, differ, wb.len, planted });
        try testing.expectEqual(@as(usize, 0), differ);
        try testing.expect(planted > 0);
    }
}

/// `x` [.., K] f32 against `l`'s weight dequantized to f32 on the CPU stream (no TF32): `[.., N]`, the grouped
/// `[G, N, K]` weight per group.
fn truthOf(l: *const dsv41.Lin, x: A, cpu: mlx.mlx_stream) !A {
    var wd = mlx.mlx_array_new();
    defer free(wd);
    try mlx.check(mlx.mlx_dequantize(&wd, l.w, l.s, l.b, mlx.mlx_optional_int.some(@intCast(l.qp.group_size)), mlx.mlx_optional_int.some(@intCast(l.qp.bits)), l.qp.mode.cstr(), none, .{ .value = .float32, .has_value = true }, cpu));
    const nd = mlx.mlx_array_ndim(wd);
    var wt = mlx.mlx_array_new();
    defer free(wt);
    try mlx.check(mlx.mlx_swapaxes(&wt, wd, @intCast(nd - 2), @intCast(nd - 1), cpu));
    var x32 = mlx.mlx_array_new();
    defer free(x32);
    try mlx.check(mlx.mlx_astype(&x32, x, .float32, cpu));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&y, x32, wt, cpu));
    return y;
}

test "dsv41 fuse mxfp8_qmm: a prompt chunk's mxfp8 linears (q_kv, wo_b, the grouped wo_a) on MLX's quantized matmul are no further from f32 truth than the dequantized bf16 GEMM, 2048 rows; the scales a code up differ" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .ctx = 0, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(20);
    const s = mlx.gpuStream();
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    var sc = dsv41.Scope.init(gpa, s);
    defer sc.deinit();
    const L: usize = 2048;
    const og: usize = @intCast(mlx.getShape(ly.wo_a.w)[0]);
    const cases = [_]struct { []const u8, *const dsv41.Lin, bool }{ .{ "q_kv", &ly.q_kv.?, false }, .{ "wo_b", &ly.wo_b, false }, .{ "wo_a", &ly.wo_a, true } };
    for (cases) |c| {
        const l = c[1];
        try testing.expectEqual(QuantMode.mxfp8, l.qp.mode);
        const K: usize = @intCast(mlx.getShape(l.w)[mlx.getShape(l.w).len - 1] * 4);
        const grouped = [_]c_int{ ci(og), ci(L), ci(K) };
        const x = try sc.keep(try fx.normal(if (c[2]) grouped[0..] else grouped[1..], .bfloat16, 0, 1));
        var ys: [3]A = undefined;
        for ([_]?[*:0]const u8{ null, "1" }, 0..) |v, i| {
            const was = ps().mxfp8_qmm;
            ps().mxfp8_qmm = v != null;
            defer ps().mxfp8_qmm = was;
            engaged.mxfp8_qmm = false;
            ys[i] = if (c[2]) try dsv41.attnOutA(&sc, m, ly, x, null) else try l.apply(&sc, x);
            try testing.expectEqual(i == 1, engaged.mxfp8_qmm);
        }
        // The plant: the same codes under scales a code up (every weight doubled).
        var up = mlx.mlx_array_new();
        defer free(up);
        try mlx.check(mlx.mlx_add(&up, l.s, try sc.keep(mlx.mlx_array_new_data(&[_]u8{1}, &[_]c_int{1}, 1, .uint8)), s));
        var bad = l.*;
        bad.s = up;
        {
            const was = ps().mxfp8_qmm;
            ps().mxfp8_qmm = true;
            defer ps().mxfp8_qmm = was;
            var lb = ly.*;
            lb.wo_a = bad;
            ys[2] = if (c[2]) try dsv41.attnOutA(&sc, m, &lb, x, null) else try bad.apply(&sc, x);
        }
        var truth = try truthOf(l, x, cpu);
        defer free(truth);
        if (c[2]) {
            // [og, L, n] to the wrapper's [L, og * n].
            var tt = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_transpose_axes(&tt, truth, &[_]c_int{ 1, 0, 2 }, 3, cpu));
            free(truth);
            var r = mlx.mlx_array_new();
            defer free(tt);
            try mlx.check(mlx.mlx_reshape(&r, tt, &[_]c_int{ ci(L), -1 }, 2, cpu));
            truth = r;
        }
        const off = try dsv41.hostF32(ys[0]);
        defer gpa.free(off);
        const on = try dsv41.hostF32(ys[1]);
        defer gpa.free(on);
        const pl = try dsv41.hostF32(ys[2]);
        defer gpa.free(pl);
        const tr = try dsv41.hostF32(truth);
        defer gpa.free(tr);
        var label: [48]u8 = undefined;
        try dkern.expectNoWorse(try std.fmt.bufPrint(&label, "fuse mxfp8_qmm {s}", .{c[0]}), on, off, tr);
        const ep = dkern.relRms(pl, off);
        std.debug.print("[fuse] mxfp8_qmm {s}: scales a code up {e:.2} from the dense path (bar > 0.5)\n", .{ c[0], ep });
        try testing.expect(ep > 0.5);
    }
}

test "dsv41 fuse engram: the wrapped tables' bytes against MLX's eviction line: the release's 104.5 GB over a 325 GB decode leaves room under a 500 GiB working set; past it, none" {
    const ws: u64 = 500 << 30;
    try testing.expectEqual(@as(u64, ws / 20 * 19), evictLine(ws));
    try testing.expect(evictRoom(325_000_000_000 + 104_500_000_000, ws) > 70_000_000_000);
    try testing.expectEqual(@as(u64, 0), evictRoom(evictLine(ws) + 1, ws));
}

test "dsv41 fuse weight path: auto is resident only when nothing streams and every Engram table is held; resident is refused by name on a streamed or partly held pack; ssd never fuses" {
    try testing.expectEqual(Path.auto, try pathOf(null));
    try testing.expectEqual(Path.auto, try pathOf(""));
    try testing.expectEqual(Path.resident, try pathOf("resident"));
    try testing.expectEqual(Path.ssd, try pathOf(" ssd "));
    try testing.expectError(error.Dsv41WeightPathSetting, pathOf("memory"));
    try testing.expect((try choose(.auto, false, true)).resident);
    try testing.expect(!(try choose(.auto, true, true)).resident);
    try testing.expect(!(try choose(.auto, false, false)).resident);
    try testing.expect((try choose(.resident, false, true)).resident);
    try testing.expectError(error.Dsv41WeightPathResidentStreamed, choose(.resident, true, true));
    try testing.expectError(error.Dsv41WeightPathResidentEngram, choose(.resident, false, false));
    for ([_]bool{ false, true }) |st| for ([_]bool{ false, true }) |h| try testing.expect(!(try choose(.ssd, st, h)).resident);
}

test "dsv41 fuse weight path: a fixture model on the ssd path serves no fusion with every switch on; on resident (auto, no Engram) it serves them" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .ctx = 0, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    defer m.fuse = .{};
    defer ps().mxfp8_qmm = false;
    const ds = env.override(.MLX_SERVE_DSV41_FUSE_DOWN_SCALE, "1");
    defer ds.restore();
    const eg = env.override(.MLX_SERVE_DSV41_FUSE_ENGRAM, "1");
    defer eg.restore();
    const mq = env.override(.MLX_SERVE_DSV41_FUSE_MXFP8_QMM, "1");
    defer mq.restore();
    {
        const p = env.override(.MLX_SERVE_DSV41_WEIGHT_PATH, "ssd");
        defer p.restore();
        try settle(m);
        try testing.expect(!m.fuse.down_scale and !m.fuse.engram and !m.fuse.mxfp8_qmm and !ps().mxfp8_qmm);
    }
    try settle(m);
    try testing.expect(m.fuse.down_scale and m.fuse.engram and m.fuse.mxfp8_qmm and ps().mxfp8_qmm);
}

/// A held table for the gather tests: `n` random rows (`rawRows`' codes and scales) in two page-aligned anonymous
/// regions, as `Table.hold` lays them out, row 0 at `base` bytes into each (`base` under a page, as the file offset's
/// remainder is).
pub fn heldTable(t: *engram.Table, n: usize, base: usize, rng: std.Random) !void {
    const page = std.heap.pageSize();
    const raw = try rawRows(testing.allocator, t, n, rng);
    defer raw.deinit(testing.allocator);
    for ([_][]const u8{ raw.w, raw.s }, [_]usize{ t.wlen, t.slen }, 0..) |src, len, i| {
        const mem = try std.posix.mmap(null, std.mem.alignForward(usize, base + n * len, page), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        @memcpy(mem[base..][0 .. n * len], src);
        t.held[i] = mem;
    }
    t.w_off = base;
    t.s_off = base;
}

test "dsv41 fuse engram (resident): the wkv gathered by id out of the held table wrapped as GPU buffers equals the kernel over the rows copied out of it, and the gated write is the chain's within 0.1 rel rms, R = 1..6, mxfp4 and mxfp8 tables; ids one row off differ" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 0, .hi = 1, .ctx = 8, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    m.fuse.engram = true;
    defer m.fuse.engram = false;
    const s = mlx.gpuStream();
    const d = m.cfg.hidden_size;
    const hc = m.cfg.dsv4_hc_mult;
    const hd: usize = 256;
    const nc: usize = 24;
    const K = nc * hd;
    const N = (hc + 1) * d;
    const ew = try fx.normal(&.{ ci(N), ci(K) }, .bfloat16, 0, 0.02);
    defer free(ew);
    const eq = try fixture.quantize(ew, 32, 8, "mxfp8");
    defer for (eq) |q| if (q.ctx != null) free(q);
    const qk = try fx.normal(&.{ ci(hc), ci(d) }, .float32, 0.5, 0.2);
    defer free(qk);
    var prng = std.Random.DefaultPrng.init(0xE7);
    const rng = prng.random();
    const n_rows: usize = 5000;
    for ([_]QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        var e: dsv41.EngramLayer = .{ .layer = 0, .wkv = .{ .w = eq[0], .s = eq[1], .b = eq[2], .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } }, .qk = qk, .table = .{ .fd = -1, .rows = n_rows, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen } };
        // Row 0 at 8 bytes past a 16-byte boundary for e2m1 (the release's layer-1 table), 16-aligned for e4m3.
        try heldTable(&e.table, n_rows, if (bits == 4) 4096 + 952 else 4096 + 48, rng);
        defer for (e.table.held[0..2]) |mem| std.posix.munmap(mem);
        try testing.expect(wrapHeld(&e));
        defer {
            _ = mlx.mlx_synchronize(s);
            unwrap(&e);
        }
        var sc = dsv41.Scope.init(gpa, s);
        defer sc.deinit();
        for (1..dkern.max_rows + 1) |R| {
            // Two tables' ids interleaved as `Hash.rows` lays them out, this one the second.
            const ids = try gpa.alloc(u32, R * 2 * nc);
            defer gpa.free(ids);
            for (ids) |*v| v.* = rng.uintLessThan(u32, @intCast(n_rows));
            const x = try sc.keep(try fx.normal(&.{ ci(R), ci(hc), ci(d) }, .bfloat16, 0, 1));
            // The same rows copied out of the held regions, as the host read hands them over.
            const raw: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", R * nc * wlen), .s = try gpa.alignedAlloc(u8, .@"4", R * nc * slen), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
            defer raw.deinit(gpa);
            for (0..R) |r| for (0..nc) |j| {
                const id = ids[(r * 2 + 1) * nc + j];
                @memcpy(raw.w[(r * nc + j) * wlen ..][0..wlen], e.table.held[0][e.table.w_off + id * wlen ..][0..wlen]);
                @memcpy(raw.s[(r * nc + j) * slen ..][0..slen], e.table.held[1][e.table.s_off + id * slen ..][0..slen]);
            };
            engaged.engram_held = false;
            const got = (try engramHeld(&sc, m, &e, ids, 2, 1, nc, 0, R, x)) orelse return error.Declined;
            try testing.expect(engaged.engram_held);
            const copied = (try engramWrite(&sc, m, &e, raw, x)) orelse return error.Declined;
            const rows = try sc.reshape(try sc.keep(try e.table.dequant(raw, R * nc, s)), &.{ ci(R), ci(K) });
            const chain = try dsv41.engramApply(&sc, m, &e, x, rows);
            // The plant: every id one row on.
            const off = try gpa.dupe(u32, ids);
            defer gpa.free(off);
            for (off) |*v| v.* = (v.* + 1) % @as(u32, @intCast(n_rows));
            const plant = (try engramHeld(&sc, m, &e, off, 2, 1, nc, 0, R, x)) orelse return error.Declined;
            const xh = try dsv41.hostF32(x);
            defer gpa.free(xh);
            var inc: [4][]f32 = undefined;
            for (&inc, [_]A{ got, copied, chain, plant }) |*o, a| {
                o.* = try dsv41.hostF32(a);
                for (o.*, xh) |*v, xv| v.* -= xv;
            }
            defer for (inc) |o| gpa.free(o);
            const same = try differing(got, copied);
            const ec = dkern.relRms(inc[0], inc[2]);
            const ep = dkern.relRms(inc[3], inc[2]);
            std.debug.print("[fuse] engram held {t} R={d}: {d} elements differ from the copied rows' launch; increment {e:.2} from the chain's (bar 0.1); ids one off {e:.2} (bar > 0.5)\n", .{ mode, R, same, ec, ep });
            try testing.expectEqual(@as(usize, 0), same);
            try testing.expect(ec <= 0.1 and ep > 0.5);
        }
    }
}

test "dsv41 fuse engram (resident): a prompt span's rows gathered by id on the GPU from the held table are the host-copied rows dequantized to bf16, bit for bit, 64 rows of 24 x 256, mxfp4 and mxfp8; ids one row off differ" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 0, .hi = 1, .ctx = 8, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    m.fuse.engram = true;
    defer m.fuse.engram = false;
    const s = mlx.gpuStream();
    const hd: usize = 256;
    const nc: usize = 24;
    const L: usize = 64;
    var prng = std.Random.DefaultPrng.init(0xE8);
    const rng = prng.random();
    const n_rows: usize = 5000;
    for ([_]QuantMode{ .mxfp4, .mxfp8 }) |mode| {
        const bits: u32 = if (mode == .mxfp4) 4 else 8;
        const wlen = hd * bits / 8;
        const slen = hd / 32;
        var e: dsv41.EngramLayer = .{ .layer = 0, .wkv = undefined, .qk = none, .table = .{ .fd = -1, .rows = n_rows, .head_dim = @intCast(hd), .mode = mode, .bits = bits, .group_size = 32, .w_off = 0, .w_stride = wlen, .wlen = wlen, .s_off = 0, .s_stride = slen, .slen = slen } };
        try heldTable(&e.table, n_rows, 8192 + 16, rng);
        defer for (e.table.held[0..2]) |mem| std.posix.munmap(mem);
        try testing.expect(wrapHeld(&e));
        defer {
            _ = mlx.mlx_synchronize(s);
            unwrap(&e);
        }
        var sc = dsv41.Scope.init(gpa, s);
        defer sc.deinit();
        const ids = try gpa.alloc(u32, L * nc);
        defer gpa.free(ids);
        for (ids) |*v| v.* = rng.uintLessThan(u32, @intCast(n_rows));
        const raw: engram.Table.Raw = .{ .w = try gpa.alignedAlloc(u8, .@"4", L * nc * wlen), .s = try gpa.alignedAlloc(u8, .@"4", L * nc * slen), .b = try gpa.alignedAlloc(u8, .@"4", 0) };
        defer raw.deinit(gpa);
        for (ids, 0..) |id, i| {
            @memcpy(raw.w[i * wlen ..][0..wlen], e.table.held[0][e.table.w_off + id * wlen ..][0..wlen]);
            @memcpy(raw.s[i * slen ..][0..slen], e.table.held[1][e.table.s_off + id * slen ..][0..slen]);
        }
        engaged.engram_rows_held = false;
        const got = (try engramRowsHeld(&sc, m, &e, ids, 1, 0, nc, 0, L)) orelse return error.Declined;
        try testing.expect(engaged.engram_rows_held);
        const want = try sc.keep(try e.table.dequantTo(raw, L * nc, .bfloat16, s));
        for (ids) |*v| v.* = (v.* + 1) % @as(u32, @intCast(n_rows));
        const plant = (try engramRowsHeld(&sc, m, &e, ids, 1, 0, nc, 0, L)) orelse return error.Declined;
        const gb = try dkern.moe_up.bitsOf(got);
        defer gpa.free(gb);
        const wb = try dkern.moe_up.bitsOf(want);
        defer gpa.free(wb);
        const pb = try dkern.moe_up.bitsOf(plant);
        defer gpa.free(pb);
        var differ: usize = 0;
        var planted: usize = 0;
        for (gb, wb, pb) |g, w, p| {
            if (g != w) differ += 1;
            if (p != w) planted += 1;
        }
        std.debug.print("[fuse] engram rows held {t}: {d} of {d} bf16 values differ from the copied rows'; ids one off, {d}\n", .{ mode, differ, wb.len, planted });
        try testing.expectEqual(@as(usize, 0), differ);
        try testing.expect(planted > wb.len / 2);
    }
}

test "dsv41 fuse pf_bf16_trunk: a prompt chunk's mxfp8 linears (q_kv, wo_b, the shared gate/up and down, the grouped wo_a) on the bf16 copy made at load give the per-chunk dequantize's bytes, 2048 rows; a copy under scales a code up differs; the copy is the bill's bytes" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const gpa = testing.allocator;
    const fx = try fixture.Fixture.init(gpa, .{ .lo = 20, .hi = 21, .ctx = 0, .fused = true });
    defer fx.deinit();
    const m = fx.m;
    const ly = fx.layer(20);
    const s = mlx.gpuStream();
    var sc = dsv41.Scope.init(gpa, s);
    defer sc.deinit();
    const L: usize = 2048;
    const og: usize = @intCast(mlx.getShape(ly.wo_a.w)[0]);
    // The per-chunk dequantize is the path the copy replaces: mxfp8_qmm (default on) would take these calls off it.
    const qmm = ps().mxfp8_qmm;
    ps().mxfp8_qmm = false;
    defer ps().mxfp8_qmm = qmm;
    // Today's outputs, before any copy exists.
    const cases = [_]struct { []const u8, *dsv41.Lin, bool }{ .{ "q_kv", &ly.q_kv.?, false }, .{ "wo_b", &ly.wo_b, false }, .{ "shared_gu", &ly.shared_gu.?, false }, .{ "shared down", &ly.shared[2], false }, .{ "wo_a", &ly.wo_a, true } };
    var xs: [cases.len]A = undefined;
    var want: [cases.len]A = undefined;
    for (cases, &xs, &want) |c, *x, *w| {
        try testing.expect(trunkLin(c[1]));
        const K: usize = @intCast(mlx.getShape(c[1].w)[mlx.getShape(c[1].w).len - 1] * 4);
        const grouped = [_]c_int{ ci(og), ci(L), ci(K) };
        x.* = try sc.keep(try fx.normal(if (c[2]) grouped[0..] else grouped[1..], .bfloat16, 0, 1));
        w.* = if (c[2]) try dsv41.attnOutA(&sc, m, ly, x.*, null) else try c[1].apply(&sc, x.*);
    }
    const n = try bf16Trunk(m);
    try testing.expectEqual(2 * layerWeights(&m.cfg, 20), n.bytes);
    for (cases, xs, want) |c, x, w| {
        try testing.expect(c[1].pf.ctx != null);
        const got = if (c[2]) try dsv41.attnOutA(&sc, m, ly, x, null) else try c[1].apply(&sc, x);
        // The plant: a copy dequantized under scales a code up.
        var up = mlx.mlx_array_new();
        defer free(up);
        try mlx.check(mlx.mlx_add(&up, c[1].s, try sc.keep(mlx.mlx_array_new_data(&[_]u8{1}, &[_]c_int{1}, 1, .uint8)), s));
        var bad = c[1].*;
        bad.s = up;
        bad.pf = try sc.keep(try plainDense(&bad, s));
        bad.s = c[1].s;
        var lb = ly.*;
        lb.wo_a = bad;
        const plant = if (c[2]) try dsv41.attnOutA(&sc, m, &lb, x, null) else try bad.apply(&sc, x);
        const d = try differing(got, w);
        const p = try differing(plant, w);
        std.debug.print("[fuse] pf_bf16_trunk {s}: {d} elements differ from the per-chunk dequantize; the planted copy, {d}\n", .{ c[0], d, p });
        try testing.expectEqual(@as(usize, 0), d);
        try testing.expect(p > 0);
    }
}
