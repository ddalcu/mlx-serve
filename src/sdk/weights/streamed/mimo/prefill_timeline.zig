//! `MIMO_PREFILL_TIMELINE` (PROFILE builds only, `sdk.plugin_profile`): the prompt pass's per-layer, per-stage and
//! per-tile timeline of one request, on the host clock AND the GPU's. In every other build `enabled` is false: the
//! storage is never referenced, every hook below compiles to nothing and no shim is linked (the served prompt pass
//! carries no clock, tag, counter or branch for it).
//!
//! `MIMO_PREFILL_TIMELINE=1` (read once at construction, `want`) arms it for the first prompt pass of the process (the
//! cell's one request). The pass is cut into STAGES by host marks (`stage`, `tile`): each mark stamps the host clock
//! (CLOCK_UPTIME_RAW, the base of Metal's GPU timestamps), the read pool's busy gauge and the MLX allocator's fresh
//! Metal buffers (`newBufferWithLength`, the cache misses). Every command buffer MLX commits while a mark is current is
//! recorded with that mark's tag by DSV's command-buffer shim (`csrc/dsv41_cb_timeline.mm`: host commit time,
//! GPUStartTime, GPUEndTime), and the read pool's event-signal log gives the time each gated part's bytes landed. No
//! fence is added: the marks sit at evals the pass already makes.
//!
//! Stages (the host's position; a stage's GPU time is its command buffers', wherever they run):
//!   edge    before the first layer (embedding, rope tables) and after the last (final norm, head, logits eval)
//!   predict P1's read-ahead predictor (router tiles on the layer input, host ids, the seed's reads posted)
//!   attn    pre-norm + the prompt attention route (NAX spans or stock tiles, each tile evaluated) + residual/post-norm
//!           built lazily
//!   router  the routing barrier: the router graph, `hostIds` (its eval: residual, post-norm, F32 router, ids to the
//!           host), then the wide lane's call setup (await read-ahead, seed, hot order, first routes)
//!   dense   layer 0's dense MLP tiles
//!   moe     one mark per expert projection tile (`tile`, `Math.projectTiles`): take + gate/up/SiLU/down gathers,
//!           evaluated; the scheduler's work after a tile (group release, next route, next bank's row lists) is
//!           charged to the tile before it
//!   join    the joinless sources' eval (the shared merge's concatenate) + `flush`
//!   gather  one combine gather tile (1,024 tokens' [count, 8, H] rows taken from the sources, evaluated)
//!   reduce  one mark per 256-token weighted reduction tile (`tile`), evaluated
//!   output  the reduction tiles' concatenate [n, H], evaluated
//!   tail    the layer's residual add, evaluated
//!
//! GLM-5.3-Flash (glm5_next) arms the same timeline (`wantFor(io, .glm5_next)`: `GLM5_NEXT_PREFILL_TIMELINE`, its lines
//! tagged `GLM5_NEXT_`) with three more stages on its DSA layers, right after `attn` in its line and TSV:
//!   indexer the indexer's pool scores (NAX or composed), evaluated
//!   topk    the top-k pool selection and its token expansion, evaluated
//!   sparse  the sparse latent attention over the selected tokens, evaluated
//! (`attn` keeps the rest: the norms, the projections, the KDA route). The stages are appended to the enum, so MiMo's tags
//! and its lines (its eleven stages, `Arch.mimo`'s order) are unchanged.
//!
//! The run's line `MIMO_PREFILL_TIMELINE {...}`: the wall (first mark to the end), per stage the host ms, the GPU busy
//! ms (union of its buffers), the GPU idle ms while the host was in it, the drive's busy ms and the fresh Metal
//! buffers; the MoE tiles split by MLX's kernel path (sorted rhs NAX GEMM vs per-row gather_qmv, `quantized.cpp`
//! `GatherQMM::eval_gpu`: M = 1 per row, B = rows, rhs iff B >= 16 and B / E >= 4; bm 32 iff B / E < 64), their GPU ms,
//! the exposed waits on expert bytes (a tile's GPU start to its gate's signal, capped at its end) and achieved TFLOPS.
//! By construction the stages' host ms add up to the wall (residual 0 up to the marks' own ns); the GPU side reports
//! the untagged and dropped buffers. With the route trace armed the per-layer, per-tile and per-buffer rows land in
//! `<trace base>.prefill-{layers,tiles,cbs}.tsv`.
const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const profile = @import("profile.zig");
const gpu_power = @import("gpu_power.zig");
/// The command-buffer rows, the signal log's entries, the host clock and the span arithmetic are the verify timeline's
/// (the same shim and pool log).
const verify_timeline = @import("../profile/verify_timeline.zig");

pub const enabled = profile.enabled;

pub const Stage = enum(u8) { edge, predict, attn, router, dense, moe, join, gather, reduce, output, tail, indexer, topk, sparse };
pub const n_stages = @typeInfo(Stage).@"enum".field_names.len;
pub const TileKind = enum(u8) { moe, reduce };

/// The arch that armed the timeline: its lines' tag (`<TAG>_PREFILL_TIMELINE`, `<TAG>_GPU_POWER_PREFILL`, the switch's
/// env), its log scope and its stages in its line's and TSV's order.
pub const Arch = enum { mimo, glm5_next };

pub fn tagOf(comptime a: Arch) []const u8 {
    return switch (a) {
        .mimo => "MIMO",
        .glm5_next => "GLM5_NEXT",
    };
}

pub fn logOf(comptime a: Arch) type {
    return switch (a) {
        .mimo => std.log.scoped(.mimo),
        .glm5_next => std.log.scoped(.glm5_next),
    };
}

pub const mimo_order = [_]Stage{ .edge, .predict, .attn, .router, .dense, .moe, .join, .gather, .reduce, .output, .tail };
pub const glm5_next_order = [_]Stage{ .edge, .predict, .attn, .indexer, .topk, .sparse, .router, .dense, .moe, .join, .gather, .reduce, .output, .tail };

pub fn orderOf(a: Arch) []const Stage {
    return switch (a) {
        .mimo => &mimo_order,
        .glm5_next => &glm5_next_order,
    };
}

/// One committed command buffer (the shim's `Dsv41tlRow`).
pub const Row = verify_timeline.Row;
/// A host mark: from `t` to the next mark's `t` the host is in `stage` (of `layer`; `no_layer` on the edge). `io_ns`:
/// the drive's busy ns so far; `nb_calls` / `nb_bytes`: fresh Metal buffers so far; `tile`: the tile index + 1 (0: none).
pub const Mark = struct { t: u64 = 0, io_ns: u64 = 0, nb_calls: u64 = 0, nb_bytes: u64 = 0, tile: u32 = 0, layer: u16 = no_layer, stage: Stage = .edge };
/// One tile: its layer, kind, its index in the layer, rows (B), the bank rows handed to MLX (E: the whole bank, or the
/// tile's slot window under `prefill_bank_view: window`, math.zig `BankView`), the distinct slots it reads
/// (sorted rows: the experts the GEMM's row blocks walk), their span (last - first + 1: the E a slot-window view of
/// the bank would pass) and the event value its waves wait for (0: none).
pub const Tile = struct { layer: u16 = 0, kind: TileKind = .moe, idx: u16 = 0, rows: u32 = 0, bank_rows: u32 = 0, slots: u32 = 0, span: u32 = 0, gate: u64 = 0 };
/// The read pool's signal log entry (`q3ld_test_ev_log`): value, signal call time, prefix-advance time, call ns.
pub const Signal = verify_timeline.Signal;
pub const no_layer: u16 = 0xFFFF;

pub const max_rows = 1 << 17;
pub const max_marks = 1 << 15;
pub const max_tiles = 1 << 15;
pub const max_signals = 1 << 16;
pub const max_layers = 48;

const c = if (enabled) struct {
    pub extern fn dsv41tl_mlx_buffer(s: mlx.mlx_stream) ?*anyopaque;
    pub extern fn dsv41tl_install(buf: ?*anyopaque, rows: [*]Row, cap: u32) c_int;
    pub extern fn dsv41tl_tag(tag: u64) void;
    pub extern fn dsv41tl_counts(out: *[3]u32) void;
    pub extern fn dsv41tl_rehook(buf: ?*anyopaque) c_int;
    pub extern fn q3ld_test_ev_log(buf: ?[*]i64, cap: i64) i64;
    pub extern fn q3ld_gauge(out: *[6]i64) void;
    pub extern fn dsv41nb_install() c_int;
    pub extern fn dsv41nb_read(out: *[2]u64) void;
} else struct {};

const State = struct {
    arch: Arch = .mimo,
    /// X12: the GPU's energy / DVFS / thermal counters over the prompt pass (IOReport; `MIMO_GPU_POWER_PREFILL`).
    power: ?gpu_power.Sampler = null,
    power0: ?*anyopaque = null,
    wanted: bool = false,
    done: bool = false,
    active: bool = false,
    stream: ?mlx.mlx_stream = null,
    io: ?std.Io = null,
    t_end: u64 = 0,
    layer: u16 = no_layer,
    gate: u64 = 0,
    layer_tiles: [2]u16 = .{ 0, 0 },
    n_marks: u32 = 0,
    n_tiles: u32 = 0,
    marks_dropped: u32 = 0,
    tiles_dropped: u32 = 0,
    n_signals: u32 = 0,
    nb_ok: bool = false,
};
var state: if (enabled) State else void = if (enabled) .{} else {};
/// The storage, zero-fill (undefined): kept out of `State` so it lands in the profile binary's bss, not its data.
const Storage = struct {
    var rows: [max_rows]Row = undefined;
    var marks: [max_marks]Mark = undefined;
    var tiles: [max_tiles]Tile = undefined;
    var signals: [max_signals]Signal = undefined;
};

pub const now = verify_timeline.now;

/// Construction: `MIMO_PREFILL_TIMELINE` read once (`"1"` arms it; unset or `"0"` leaves it off; anything else refuses).
pub fn want(io: std.Io) !bool {
    return wantFor(io, .mimo) catch return error.MimoPrefillTimelineSetting;
}

/// Construction: arch `a`'s switch (`<TAG>_PREFILL_TIMELINE`) read once, as `want`.
pub fn wantFor(io: std.Io, comptime a: Arch) error{PrefillTimelineSetting}!bool {
    if (comptime !enabled) return false;
    const name = comptime tagOf(a) ++ "_PREFILL_TIMELINE";
    const raw = env.named(name) orelse return false;
    const v = std.mem.span(raw);
    if (std.mem.eql(u8, v, "0")) return false;
    if (!std.mem.eql(u8, v, "1")) {
        logOf(a).err(name ++ "={s} refused: 0 | 1", .{v});
        return error.PrefillTimelineSetting;
    }
    state.wanted = true;
    state.arch = a;
    state.io = io;
    return true;
}

/// The timeline is recording a prompt pass inside one of the model's layers (profile builds; false in every other
/// build): an arch's profile-only marks and evals at its stage boundaries (glm5_next's DSA split) run only then, never
/// for work outside the layers (the edge, glm5_next's MTP lane catching up between prompt calls).
pub fn recording() bool {
    if (comptime !enabled) return false;
    return state.active and state.layer != no_layer;
}

/// The prompt phase begins: the first one of the process installs the hook on `s`'s queue and arms the storage.
pub fn begin(s: mlx.mlx_stream) void {
    if (comptime !enabled) return;
    if (!state.wanted or state.done or state.active) return;
    const rc = c.dsv41tl_install(c.dsv41tl_mlx_buffer(s), &Storage.rows, max_rows);
    if (rc != 0) {
        state.done = true;
        switch (state.arch) {
            inline else => |a| logOf(a).info(comptime tagOf(a) ++ "_PREFILL_TIMELINE_UNAVAILABLE {{\"reason\": \"install rc {d}\"}}", .{rc}),
        }
        return;
    }
    state.nb_ok = c.dsv41nb_install() == 0;
    _ = c.q3ld_test_ev_log(@ptrCast(&Storage.signals), max_signals);
    state.stream = s;
    state.n_marks = 0;
    state.n_tiles = 0;
    state.marks_dropped = 0;
    state.tiles_dropped = 0;
    state.layer = no_layer;
    state.gate = 0;
    state.active = true;
    state.power = gpu_power.Sampler.open() catch null;
    if (state.power) |*p| state.power0 = p.sample();
    stage(.edge);
}

fn gauges(m: *Mark) void {
    var g: [6]i64 = undefined;
    c.q3ld_gauge(&g);
    // 4: wall ns with a read in flight (closed periods), 5: the open period's start (mach time, ns), 0: in flight.
    const open: i64 = if (g[0] > 0 and g[5] > 0) @as(i64, @intCast(m.t)) - g[5] else 0;
    m.io_ns = @intCast(@max(g[4] + @max(open, 0), 0));
    if (state.nb_ok) {
        var nb: [2]u64 = undefined;
        c.dsv41nb_read(&nb);
        m.nb_calls = nb[0];
        m.nb_bytes = nb[1];
    }
}

fn push(st: Stage, tile_plus_one: u32) void {
    if (state.n_marks == max_marks) {
        state.marks_dropped += 1;
        return;
    }
    const m = &Storage.marks[state.n_marks];
    m.* = .{ .t = now(), .tile = tile_plus_one, .layer = state.layer, .stage = st };
    gauges(m);
    state.n_marks += 1;
}

/// The command-buffer tag of a stage mark (`tagOf`'s inverse is `tagStage` / `tagTile`).
pub fn stageTag(st: Stage, l: u16) u64 {
    return @as(u64, @intFromEnum(st)) << 32 | (@as(u64, l) + 1);
}
pub fn tileTag(st: Stage, ti: u32) u64 {
    return @as(u64, @intFromEnum(st)) << 32 | 1 << 31 | @as(u64, ti);
}
pub fn tagStage(tag: u64) Stage {
    return @enumFromInt(@as(u8, @intCast(tag >> 32)));
}
pub fn tagTile(tag: u64) ?u32 {
    if (tag & (1 << 31) == 0) return null;
    return @intCast(tag & 0x7fff_ffff);
}

/// A routed or dense layer of the prompt pass begins (absolute index).
pub fn layer(l: u32) void {
    if (comptime !enabled) return;
    if (!state.active) return;
    state.layer = @intCast(l);
    state.layer_tiles = .{ 0, 0 };
    if (state.stream) |s| _ = c.dsv41tl_rehook(c.dsv41tl_mlx_buffer(s));
}

/// The pass leaves the layers (the edge stage: final norm, head, logits).
pub fn leaveLayers() void {
    if (comptime !enabled) return;
    if (!state.active) return;
    state.layer = no_layer;
    stage(.edge);
}

pub fn stage(st: Stage) void {
    if (comptime !enabled) return;
    if (!state.active) return;
    push(st, 0);
    c.dsv41tl_tag(stageTag(st, state.layer));
}

/// A wide group's event gates were registered (`StreamSource.gate`): its waves wait for `value` (0: no reads).
pub fn gate(value: u64) void {
    if (comptime !enabled) return;
    if (!state.active) return;
    state.gate = value;
}

/// A tile begins (before its graph is built): its mark, its record and its buffers' tag.
pub fn tile(kind: TileKind, rows: u32, bank_rows: u32, slots: []const u32) void {
    if (comptime !enabled) return;
    if (!state.active) return;
    const st: Stage = if (kind == .moe) .moe else .reduce;
    const k = @intFromEnum(kind);
    const idx = state.layer_tiles[k];
    state.layer_tiles[k] +%= 1;
    if (state.n_tiles == max_tiles) {
        state.tiles_dropped += 1;
        return stage(st);
    }
    const ti = state.n_tiles;
    Storage.tiles[ti] = .{
        .layer = state.layer,
        .kind = kind,
        .idx = idx,
        .rows = rows,
        .bank_rows = bank_rows,
        .slots = distinctSorted(slots),
        .span = if (slots.len == 0) 0 else slots[slots.len - 1] - slots[0] + 1,
        .gate = if (kind == .moe) state.gate else 0,
    };
    state.n_tiles += 1;
    push(st, ti + 1);
    c.dsv41tl_tag(tileTag(st, ti));
}

/// The distinct slots of sorted `slots` (the bank rows a sorted GEMM tile spans).
pub fn distinctSorted(slots: []const u32) u32 {
    if (slots.len == 0) return 0;
    var n: u32 = 1;
    for (slots[1..], slots[0 .. slots.len - 1]) |x, p| n += @intFromBool(x != p);
    return n;
}

/// The prompt phase ends (after the logits eval): every buffer settled, the line, the files; uninstalled.
pub fn finish(a: std.mem.Allocator, base: ?[]const u8, hidden: u32, inter: u32) void {
    if (comptime !enabled) return;
    if (!state.active) return;
    state.t_end = now();
    c.dsv41tl_tag(0);
    state.active = false;
    state.done = true;
    if (state.stream) |s| _ = mlx.mlx_synchronize(s);
    state.n_signals = @intCast(@min(c.q3ld_test_ev_log(null, 0), max_signals));
    var k: [3]u32 = undefined;
    const t_wait = now();
    while (true) {
        c.dsv41tl_counts(&k);
        if (k[1] >= @min(k[0], max_rows) or now() - t_wait > 2 * std.time.ns_per_s) break;
        std.Thread.yield() catch {};
    }
    _ = c.q3ld_test_ev_log(null, -1);
    const n_rows: u32 = @min(k[0], max_rows);
    const in: Input = .{
        .rows = Storage.rows[0..n_rows],
        .marks = Storage.marks[0..state.n_marks],
        .tiles = Storage.tiles[0..state.n_tiles],
        .signals = Storage.signals[0..state.n_signals],
        .t_end = state.t_end,
        .hidden = hidden,
        .inter = inter,
        .dropped_rows = k[2],
        .pending_rows = @min(k[0], max_rows) -| k[1],
        .dropped_marks = state.marks_dropped,
        .dropped_tiles = state.tiles_dropped,
        .newbuffer = state.nb_ok,
        .model_queue_only = true,
        .order = orderOf(state.arch),
    };
    switch (state.arch) {
        inline else => |arch| finishAs(arch, a, base, in),
    }
}

fn finishAs(comptime arch: Arch, a: std.mem.Allocator, base: ?[]const u8, in: Input) void {
    const alog = logOf(arch);
    const tag = comptime tagOf(arch);
    const sum = summarize(a, in) catch |e| return alog.err(tag ++ "_PREFILL_TIMELINE summary: {s}", .{@errorName(e)});
    defer sum.deinit(a);
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    writeLineOf(tag, &aw.writer, in, sum) catch return;
    alog.info("{s}", .{aw.written()});
    // X12: the prompt pass's GPU energy, active-time mean MHz and thermal limiting (the pass's window, system-wide).
    if (state.power) |*p| if (state.power0) |p0| {
        const p1 = p.sample();
        defer p.free(p1);
        defer p.free(p0);
        state.power0 = null;
        if (p.delta(p0, p1)) |d| {
            var pb: [1024]u8 = undefined;
            var w: std.Io.Writer = .fixed(&pb);
            w.writeAll(tag ++ "_GPU_POWER_PREFILL ") catch {};
            gpu_power.json(&w, d, p.mhz[0..p.n_mhz]) catch {};
            alog.info("{s}", .{w.buffered()});
        } else |_| {}
    };
    if (base) |b| if (state.io) |io| {
        writeFile(arch, a, io, b, ".prefill-layers.tsv", in, sum, writeLayers);
        writeFile(arch, a, io, b, ".prefill-tiles.tsv", in, sum, writeTiles);
        writeFile(arch, a, io, b, ".prefill-cbs.tsv", in, sum, writeRows);
    };
}

// ── the summary (pure over the stored rows; the tests feed it by hand) ──

pub const Input = struct {
    rows: []const Row,
    marks: []const Mark,
    tiles: []const Tile,
    signals: []const Signal,
    t_end: u64,
    hidden: u32,
    inter: u32,
    dropped_rows: u32 = 0,
    pending_rows: u32 = 0,
    dropped_marks: u32 = 0,
    dropped_tiles: u32 = 0,
    newbuffer: bool = true,
    /// The stages the line and the layers TSV print, in order (the arch's: `orderOf`).
    order: []const Stage = &mimo_order,
    /// X12 (mimo/gpu-warm): count only the MLX stream queue's buffers (`Row.queue` 1); the warm engine's chunks run on
    /// their own queue and would fill the gpu_idle they exist to cover. The real summary sets it; hand-made rows keep
    /// every row.
    model_queue_only: bool = false,
};

pub const StageSum = struct { host: u64 = 0, gpu: u64 = 0, idle: u64 = 0, io: u64 = 0, nb_calls: u64 = 0, nb_bytes: u64 = 0, marks: u32 = 0, cbs: u32 = 0 };
pub const TileSum = struct { host: u64 = 0, gpu: u64 = 0, start: u64 = 0, end: u64 = 0, wait: u64 = 0, cbs: u32 = 0 };
pub const LayerSum = struct { stages: [n_stages]StageSum = @splat(.{}), moe_tiles: u32 = 0, moe_rows: u64 = 0, moe_qmv_tiles: u32 = 0, moe_wait: u64 = 0, moe_gpu: u64 = 0 };
/// MoE tiles by MLX's path: `gpu` their buffers' GPU ms (incl. exposed waits), `wait` the exposed waits.
pub const PathSum = struct { tiles: u32 = 0, rows: u64 = 0, gpu: u64 = 0, wait: u64 = 0, waited: u32 = 0 };

pub const Sum = struct {
    wall: u64 = 0,
    busy: u64 = 0,
    stages: [n_stages]StageSum = @splat(.{}),
    layers: [max_layers]LayerSum = @splat(.{}),
    tiles: []TileSum = &.{},
    /// [0] sorted rhs bm 32, [1] sorted rhs bm 64, [2] per-row gather_qmv.
    paths: [3]PathSum = @splat(.{}),
    /// The MoE tiles' paths had each passed a slot-window view of its bank (E = its slot span): [bm 32, bm 64, qmv].
    span_paths: [3]u32 = @splat(0),
    reduce: PathSum = .{},
    untagged: u32 = 0,
    outside: u32 = 0,
    /// X12: other-queue buffers left out (`Input.model_queue_only`: the warm engine's chunks).
    other_queue: u32 = 0,
    /// Buffers Metal reported with GPUStartTime 0 (timed from their host commit: `startOf`).
    zero_start: u32 = 0,
    io: u64 = 0,

    pub fn deinit(s: Sum, a: std.mem.Allocator) void {
        a.free(s.tiles);
    }
};

/// MLX's kernel path for a sorted gather_qmm of `rows` (B, M = 1 each) over a bank of `bank_rows` slots (E):
/// 0 = rhs NAX bm 32, 1 = rhs NAX bm 64, 2 = gather_qmv (one GEMV per row).
pub fn pathOf(rows: u32, bank_rows: u32) u2 {
    if (bank_rows == 0 or rows < 16 or rows / bank_rows < 4) return 2;
    return if (rows / bank_rows < 64) 0 else 1;
}

const Span = verify_timeline.Span;
const busyIn = verify_timeline.busyIn;
const signalOf = verify_timeline.signalOf;

/// A buffer's GPU start. Metal reports GPUStartTime 0 for some committed buffers (cells/p1-20261009: the first buffer of
/// 293 of 546 prompt-pass dispatches, the one holding a group's event wait); such a buffer started no earlier than its
/// host commit, which is used instead (else its span reads as the whole uptime: gpu_ms 2.0e9).
pub fn startOf(r: Row) u64 {
    return if (r.gpu_start == 0) r.host_commit else r.gpu_start;
}

fn unionOf(a: std.mem.Allocator, rs: []const Row, lo: u64, hi: u64, filter: ?Stage, model_only: bool) ![]Span {
    var list: std.ArrayList(Span) = .empty;
    errdefer list.deinit(a);
    for (rs) |r| {
        if (r.tag == 0 or r.gpu_end <= startOf(r)) continue;
        if (model_only and r.queue != 1) continue;
        if (filter) |f| if (tagStage(r.tag) != f) continue;
        const x = @max(startOf(r), lo);
        const y = @min(r.gpu_end, hi);
        if (y > x) try list.append(a, .{ x, y });
    }
    std.mem.sort(Span, list.items, {}, struct {
        fn lt(_: void, p: Span, q: Span) bool {
            return p[0] < q[0];
        }
    }.lt);
    var m: usize = 0;
    for (list.items) |sp| {
        if (m > 0 and sp[0] <= list.items[m - 1][1]) {
            list.items[m - 1][1] = @max(list.items[m - 1][1], sp[1]);
        } else {
            list.items[m] = sp;
            m += 1;
        }
    }
    list.shrinkRetainingCapacity(m);
    return list.toOwnedSlice(a);
}

fn totalOf(u: []const Span) u64 {
    var s: u64 = 0;
    for (u) |sp| s += sp[1] - sp[0];
    return s;
}

pub fn summarize(a: std.mem.Allocator, in: Input) !Sum {
    var s: Sum = .{};
    if (in.marks.len == 0) return s;
    const t0 = in.marks[0].t;
    s.wall = in.t_end -| t0;
    const all = try unionOf(a, in.rows, t0, in.t_end, null, in.model_queue_only);
    defer a.free(all);
    s.busy = totalOf(all);
    s.io = (if (in.marks.len > 0) lastIo(in) else 0) -| in.marks[0].io_ns;
    // Host side: each mark to the next (the last to the end).
    for (in.marks, 0..) |m, i| {
        const t1 = if (i + 1 < in.marks.len) in.marks[i + 1].t else in.t_end;
        const next_io = if (i + 1 < in.marks.len) in.marks[i + 1].io_ns else m.io_ns;
        const next_nb = if (i + 1 < in.marks.len) in.marks[i + 1] else m;
        const d: StageSum = .{
            .host = t1 -| m.t,
            .idle = (t1 -| m.t) -| busyIn(all, m.t, t1),
            .io = next_io -| m.io_ns,
            .nb_calls = next_nb.nb_calls -| m.nb_calls,
            .nb_bytes = next_nb.nb_bytes -| m.nb_bytes,
            .marks = 1,
        };
        add(&s.stages[@intFromEnum(m.stage)], d);
        if (m.layer < max_layers) add(&s.layers[m.layer].stages[@intFromEnum(m.stage)], d);
    }
    // GPU side: per stage the union of its buffers; per layer the sum of its buffers' spans.
    inline for (0..n_stages) |si| {
        const u = try unionOf(a, in.rows, t0, in.t_end, @as(Stage, @enumFromInt(si)), in.model_queue_only);
        defer a.free(u);
        s.stages[si].gpu = totalOf(u);
    }
    s.tiles = try a.alloc(TileSum, in.tiles.len);
    errdefer a.free(s.tiles);
    @memset(s.tiles, .{});
    for (in.rows) |r| {
        if (in.model_queue_only and r.queue != 1) {
            s.other_queue += 1;
            continue;
        }
        if (r.tag == 0) {
            s.untagged += 1;
            continue;
        }
        const r0 = startOf(r);
        s.zero_start += @intFromBool(r.gpu_start == 0);
        if (r.gpu_end <= r0 or r.gpu_end <= t0 or r0 >= in.t_end) {
            s.outside += 1;
            continue;
        }
        const st = tagStage(r.tag);
        s.stages[@intFromEnum(st)].cbs += 1;
        const d = r.gpu_end - r0;
        if (tagTile(r.tag)) |ti| {
            if (ti >= in.tiles.len) continue;
            const t = &s.tiles[ti];
            t.start = if (t.cbs == 0) r0 else @min(t.start, r0);
            t.end = @max(t.end, r.gpu_end);
            t.gpu += d;
            t.cbs += 1;
            const l = in.tiles[ti].layer;
            if (l < max_layers) s.layers[l].stages[@intFromEnum(st)].gpu += d;
        } else {
            const l: u64 = (r.tag & 0x7fff_ffff) -| 1;
            if (l < max_layers) s.layers[@intCast(l)].stages[@intFromEnum(st)].gpu += d;
        }
    }
    // Tiles: host span (their mark to the next), exposed waits, path sums.
    for (in.marks, 0..) |m, i| if (m.tile > 0 and m.tile - 1 < s.tiles.len) {
        const t1 = if (i + 1 < in.marks.len) in.marks[i + 1].t else in.t_end;
        s.tiles[m.tile - 1].host += t1 -| m.t;
    };
    for (in.tiles, s.tiles) |t, *ts| {
        if (t.gate > 0 and ts.cbs > 0) if (signalOf(in.signals, t.gate)) |sv| {
            ts.wait = @min(sv, ts.end) -| ts.start;
        };
        const p: *PathSum = if (t.kind == .reduce) &s.reduce else &s.paths[pathOf(t.rows, t.bank_rows)];
        p.tiles += 1;
        p.rows += t.rows;
        p.gpu += ts.gpu;
        p.wait += ts.wait;
        p.waited += @intFromBool(ts.wait > 0);
        if (t.kind == .moe) s.span_paths[pathOf(t.rows, t.span)] += 1;
        if (t.kind == .moe and t.layer < max_layers) {
            const l = &s.layers[t.layer];
            l.moe_tiles += 1;
            l.moe_rows += t.rows;
            l.moe_qmv_tiles += @intFromBool(pathOf(t.rows, t.bank_rows) == 2);
            l.moe_wait += ts.wait;
            l.moe_gpu += ts.gpu;
        }
    }
    return s;
}

fn lastIo(in: Input) u64 {
    return in.marks[in.marks.len - 1].io_ns;
}

fn add(dst: *StageSum, d: StageSum) void {
    dst.host += d.host;
    dst.idle += d.idle;
    dst.io += d.io;
    dst.nb_calls += d.nb_calls;
    dst.nb_bytes += d.nb_bytes;
    dst.marks += d.marks;
}

fn ms(x: u64) f64 {
    return @as(f64, @floatFromInt(x)) / 1e6;
}

/// TFLOPS of `rows` routed rows through gate, up (H -> I) and down (I -> H) in `ns`.
pub fn tflops(rows: u64, hidden: u32, inter: u32, ns: u64) f64 {
    if (ns == 0) return 0;
    const flop = @as(f64, @floatFromInt(rows)) * 3.0 * 2.0 * @as(f64, @floatFromInt(hidden)) * @as(f64, @floatFromInt(inter));
    return flop / (@as(f64, @floatFromInt(ns)) * 1e3);
}

pub fn writeLine(w: *std.Io.Writer, in: Input, s: Sum) !void {
    return writeLineOf("MIMO", w, in, s);
}

/// The run's line under `tag` (`<tag>_PREFILL_TIMELINE {...}`), its stages `in.order`'s.
pub fn writeLineOf(comptime tag: []const u8, w: *std.Io.Writer, in: Input, s: Sum) !void {
    var host_sum: u64 = 0;
    for (s.stages) |st| host_sum += st.host;
    try w.print(tag ++ "_PREFILL_TIMELINE {{\"wall_ms\": {d:.3}, \"gpu_busy_ms\": {d:.3}, \"gpu_idle_ms\": {d:.3}, \"io_busy_ms\": {d:.3}, \"host_residual_ms\": {d:.3}, \"stages\": {{", .{
        ms(s.wall), ms(s.busy), ms(s.wall -| s.busy), ms(s.io), ms(s.wall) - ms(host_sum),
    });
    for (in.order, 0..) |stg, i| {
        const st = s.stages[@intFromEnum(stg)];
        try w.print("{s}\"{s}\": {{\"host_ms\": {d:.3}, \"gpu_ms\": {d:.3}, \"gpu_idle_ms\": {d:.3}, \"io_busy_ms\": {d:.3}, \"newbuf\": {d}, \"newbuf_mb\": {d:.1}, \"marks\": {d}, \"cbs\": {d}}}", .{
            if (i == 0) "" else ", ", @tagName(stg), ms(st.host), ms(st.gpu), ms(st.idle), ms(st.io), st.nb_calls, @as(f64, @floatFromInt(st.nb_bytes)) / 1e6, st.marks, st.cbs,
        });
    }
    try w.print("}}, \"moe\": {{", .{});
    const names = [_][]const u8{ "rhs_bm32", "rhs_bm64", "qmv" };
    for (s.paths, names, 0..) |p, n, i| {
        try w.print("{s}\"{s}\": {{\"tiles\": {d}, \"rows\": {d}, \"gpu_ms\": {d:.3}, \"wait_ms\": {d:.3}, \"waited_tiles\": {d}, \"work_tflops\": {d:.2}}}", .{
            if (i == 0) "" else ", ", n, p.tiles, p.rows, ms(p.gpu), ms(p.wait), p.waited, tflops(p.rows, in.hidden, in.inter, p.gpu -| p.wait),
        });
    }
    try w.print("}}, \"moe_span_view_tiles\": {{\"rhs_bm32\": {d}, \"rhs_bm64\": {d}, \"qmv\": {d}}}", .{ s.span_paths[0], s.span_paths[1], s.span_paths[2] });
    try w.print(", \"reduce\": {{\"tiles\": {d}, \"rows\": {d}, \"gpu_ms\": {d:.3}}}, \"cbs\": {d}, \"untagged_cbs\": {d}, \"outside_cbs\": {d}, \"other_queue_cbs\": {d}, \"zero_start_cbs\": {d}, \"dropped_cbs\": {d}, \"pending_cbs\": {d}, \"marks\": {d}, \"dropped_marks\": {d}, \"tiles\": {d}, \"dropped_tiles\": {d}, \"signals\": {d}, \"newbuffer_counted\": {}}}", .{
        s.reduce.tiles, s.reduce.rows, ms(s.reduce.gpu), in.rows.len, s.untagged, s.outside, s.other_queue, s.zero_start, in.dropped_rows, in.pending_rows, in.marks.len, in.dropped_marks, in.tiles.len, in.dropped_tiles, in.signals.len, in.newbuffer,
    });
}

pub fn writeLayers(w: *std.Io.Writer, in: Input, s: Sum) anyerror!void {
    try w.print("layer", .{});
    for (in.order) |stg| {
        const n = @tagName(stg);
        try w.print("\t{s}_host_ms\t{s}_gpu_ms\t{s}_idle_ms", .{ n, n, n });
    }
    try w.print("\tio_busy_ms\tnewbuf\tnewbuf_mb\tmoe_tiles\tmoe_rows\tmoe_qmv_tiles\tmoe_gpu_ms\tmoe_wait_ms\n", .{});
    for (s.layers, 0..) |l, li| {
        var any = false;
        for (l.stages) |st| any = any or st.marks > 0;
        if (!any) continue;
        try w.print("{d}", .{li});
        var io: u64 = 0;
        var nb: u64 = 0;
        var nbb: u64 = 0;
        for (in.order) |stg| {
            const st = l.stages[@intFromEnum(stg)];
            try w.print("\t{d:.3}\t{d:.3}\t{d:.3}", .{ ms(st.host), ms(st.gpu), ms(st.idle) });
        }
        for (l.stages) |st| {
            io += st.io;
            nb += st.nb_calls;
            nbb += st.nb_bytes;
        }
        try w.print("\t{d:.3}\t{d}\t{d:.1}\t{d}\t{d}\t{d}\t{d:.3}\t{d:.3}\n", .{ ms(io), nb, @as(f64, @floatFromInt(nbb)) / 1e6, l.moe_tiles, l.moe_rows, l.moe_qmv_tiles, ms(l.moe_gpu), ms(l.moe_wait) });
    }
}

pub fn writeTiles(w: *std.Io.Writer, in: Input, s: Sum) anyerror!void {
    try w.print("tile\tlayer\tkind\tidx\trows\tbank_rows\tslots\tspan\tpath\tspan_path\tgate\thost_ms\tgpu_ms\twait_ms\tcbs\tgpu_start_ms\n", .{});
    const t0 = if (in.marks.len > 0) in.marks[0].t else 0;
    for (in.tiles, s.tiles, 0..) |t, ts, i| {
        const path = if (t.kind == .reduce) "reduce" else switch (pathOf(t.rows, t.bank_rows)) {
            0 => "rhs_bm32",
            1 => "rhs_bm64",
            else => "qmv",
        };
        const span_path = if (t.kind == .reduce) "reduce" else switch (pathOf(t.rows, t.span)) {
            0 => "rhs_bm32",
            1 => "rhs_bm64",
            else => "qmv",
        };
        try w.print("{d}\t{d}\t{s}\t{d}\t{d}\t{d}\t{d}\t{d}\t{s}\t{s}\t{d}\t{d:.4}\t{d:.4}\t{d:.4}\t{d}\t{d:.4}\n", .{
            i, t.layer, @tagName(t.kind), t.idx, t.rows, t.bank_rows, t.slots, t.span, path, span_path, t.gate, ms(ts.host), ms(ts.gpu), ms(ts.wait), ts.cbs, ms(ts.start -| t0),
        });
    }
}

pub fn writeRows(w: *std.Io.Writer, in: Input, _: Sum) anyerror!void {
    try w.print("host_commit_ms\tgpu_start_ms\tgpu_end_ms\thost_done_ms\tstage\tlayer\ttile\tstatus\n", .{});
    const t0 = if (in.marks.len > 0) in.marks[0].t else 0;
    for (in.rows) |r| {
        if (r.tag == 0) continue;
        const tl = tagTile(r.tag);
        const l: i64 = if (tl) |ti| (if (ti < in.tiles.len) @as(i64, in.tiles[ti].layer) else -1) else @as(i64, @intCast(r.tag & 0x7fff_ffff)) - 1;
        try w.print("{d:.4}\t{d:.4}\t{d:.4}\t{d:.4}\t{s}\t{d}\t{d}\t{d}\n", .{
            ms(r.host_commit -| t0), ms(r.gpu_start -| t0), ms(r.gpu_end -| t0), ms(r.host_done -| t0), @tagName(tagStage(r.tag)), l, if (tl) |ti| @as(i64, ti) else -1, r.status,
        });
    }
}

fn writeFile(comptime arch: Arch, a: std.mem.Allocator, io: std.Io, base: []const u8, ext: []const u8, in: Input, s: Sum, comptime body: fn (*std.Io.Writer, Input, Sum) anyerror!void) void {
    const alog = logOf(arch);
    const tag = comptime tagOf(arch);
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    body(&aw.writer, in, s) catch |e| return alog.err(tag ++ "_PREFILL_TIMELINE {s}: {s}", .{ ext, @errorName(e) });
    const path = std.fmt.allocPrint(a, "{s}{s}", .{ base, ext }) catch return;
    defer a.free(path);
    std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = aw.written() }) catch |e| alog.err(tag ++ "_PREFILL_TIMELINE write {s}: {s}", .{ path, @errorName(e) });
}

// ── tests (any build: the summary is pure; the hooks are checked compiled out in a served build) ──

const testing = std.testing;

test "mimo prefill timeline: a served build compiles every hook to nothing" {
    if (enabled) return error.SkipZigTest;
    try testing.expect(@TypeOf(state) == void);
    layer(3);
    stage(.attn);
    gate(9);
    tile(.moe, 1024, 146, &.{ 1, 2, 5 });
    leaveLayers();
    begin(undefined);
    finish(testing.allocator, null, 4096, 2048);
    try testing.expect(!(try want(undefined)));
}

test "mimo prefill timeline: tags round-trip; the MLX path rule at the real shapes" {
    try testing.expectEqual(Stage.router, tagStage(stageTag(.router, 7)));
    try testing.expectEqual(@as(?u32, null), tagTile(stageTag(.router, 7)));
    try testing.expectEqual(@as(?u32, 4242), tagTile(tileTag(.moe, 4242)));
    try testing.expectEqual(Stage.reduce, tagStage(tileTag(.reduce, 0)));
    try testing.expect(stageTag(.edge, no_layer) != 0 and tileTag(.edge, 0) != 0);
    // quantized.cpp GatherQMM::eval_gpu + gather_qmm_rhs_nax: rhs iff B >= 16 and B / E >= 4; bm 32 iff B / E < 64.
    try testing.expectEqual(@as(u2, 0), pathOf(1024, 146)); // 7: bm 32
    try testing.expectEqual(@as(u2, 0), pathOf(584, 146)); // 4: rhs
    try testing.expectEqual(@as(u2, 2), pathOf(583, 146)); // 3: per-row qmv
    try testing.expectEqual(@as(u2, 2), pathOf(383, 96));
    try testing.expectEqual(@as(u2, 0), pathOf(384, 96));
    try testing.expectEqual(@as(u2, 1), pathOf(1024, 3)); // a 3-slot view: bm 64
    try testing.expectEqual(@as(u2, 2), pathOf(15, 1));
    try testing.expectEqual(@as(u32, 3), distinctSorted(&.{ 4, 4, 9, 9, 9, 11 }));
    try testing.expectEqual(@as(u32, 0), distinctSorted(&.{}));
}

test "mimo prefill timeline: stages partition the wall; GPU busy, idle, exposed waits and tiles reconcile" {
    const a = testing.allocator;
    // Host: edge [0,10) attn [10,40) router [40,50) moe tile 0 [50,70) moe tile 1 [70,90) reduce tile 2 [90,100), end 100.
    const marks = [_]Mark{
        .{ .t = 0, .stage = .edge, .io_ns = 0, .nb_calls = 0, .nb_bytes = 0 },
        .{ .t = 10, .stage = .attn, .layer = 1, .io_ns = 5, .nb_calls = 1, .nb_bytes = 100 },
        .{ .t = 40, .stage = .router, .layer = 1, .io_ns = 25, .nb_calls = 1, .nb_bytes = 100 },
        .{ .t = 50, .stage = .moe, .layer = 1, .tile = 1, .io_ns = 30, .nb_calls = 3, .nb_bytes = 300 },
        .{ .t = 70, .stage = .moe, .layer = 1, .tile = 2, .io_ns = 45, .nb_calls = 3, .nb_bytes = 300 },
        .{ .t = 90, .stage = .reduce, .layer = 1, .tile = 3, .io_ns = 45, .nb_calls = 4, .nb_bytes = 340 },
    };
    const tiles = [_]Tile{
        .{ .layer = 1, .kind = .moe, .idx = 0, .rows = 1024, .bank_rows = 146, .slots = 3, .span = 3, .gate = 7 },
        .{ .layer = 1, .kind = .moe, .idx = 1, .rows = 300, .bank_rows = 146, .slots = 1, .gate = 7 },
        .{ .layer = 1, .kind = .reduce, .idx = 0, .rows = 256 },
    };
    // GPU: attn [12,38), router [41,49), tile 0 [52,68) waiting to 60 (signal of 7 at 60), tile 1 [72,88), reduce [91,99),
    // one untagged buffer, one before the pass.
    const rows = [_]Row{
        .{ .gpu_start = 12, .gpu_end = 38, .tag = stageTag(.attn, 1) },
        .{ .gpu_start = 41, .gpu_end = 49, .tag = stageTag(.router, 1) },
        .{ .gpu_start = 52, .gpu_end = 68, .tag = tileTag(.moe, 0) },
        .{ .gpu_start = 72, .gpu_end = 88, .tag = tileTag(.moe, 1) },
        .{ .gpu_start = 91, .gpu_end = 99, .tag = tileTag(.reduce, 2) },
        .{ .gpu_start = 1, .gpu_end = 2, .tag = 0 },
    };
    const sig = [_]Signal{ .{ 5, 30, 30, 1 }, .{ 7, 60, 60, 1 } };
    const in: Input = .{ .rows = &rows, .marks = &marks, .tiles = &tiles, .signals = &sig, .t_end = 100, .hidden = 4096, .inter = 2048 };
    const s = try summarize(a, in);
    defer s.deinit(a);
    try testing.expectEqual(@as(u64, 100), s.wall);
    var host: u64 = 0;
    for (s.stages) |st| host += st.host;
    try testing.expectEqual(s.wall, host);
    try testing.expectEqual(@as(u64, 26 + 8 + 16 + 16 + 8), s.busy);
    try testing.expectEqual(@as(u64, 30), s.stages[@intFromEnum(Stage.attn)].host);
    try testing.expectEqual(@as(u64, 26), s.stages[@intFromEnum(Stage.attn)].gpu);
    try testing.expectEqual(@as(u64, 4), s.stages[@intFromEnum(Stage.attn)].idle);
    try testing.expectEqual(@as(u64, 20), s.stages[@intFromEnum(Stage.attn)].io);
    try testing.expectEqual(@as(u64, 32), s.stages[@intFromEnum(Stage.moe)].gpu);
    try testing.expectEqual(@as(u64, 8), s.stages[@intFromEnum(Stage.moe)].idle);
    try testing.expectEqual(@as(u64, 1), s.stages[@intFromEnum(Stage.moe)].nb_calls);
    try testing.expectEqual(@as(u64, 2), s.stages[@intFromEnum(Stage.router)].nb_calls);
    try testing.expectEqual(@as(u64, 200), s.stages[@intFromEnum(Stage.router)].nb_bytes);
    try testing.expectEqual(@as(u64, 8), s.tiles[0].wait);
    try testing.expectEqual(@as(u64, 0), s.tiles[1].wait);
    try testing.expectEqual(@as(u64, 20), s.tiles[0].host);
    try testing.expectEqual(@as(u32, 1), s.paths[0].tiles);
    try testing.expectEqual(@as(u32, 1), s.paths[2].tiles);
    try testing.expectEqual(@as(u64, 8), s.paths[0].wait);
    try testing.expectEqual(@as(u32, 1), s.reduce.tiles);
    try testing.expectEqual([3]u32{ 0, 1, 1 }, s.span_paths);
    try testing.expectEqual(@as(u32, 2), s.layers[1].moe_tiles);
    try testing.expectEqual(@as(u32, 1), s.layers[1].moe_qmv_tiles);
    try testing.expectEqual(@as(u64, 26), s.layers[1].stages[@intFromEnum(Stage.attn)].gpu);
    try testing.expectEqual(@as(u32, 1), s.untagged);
    try testing.expectEqual(@as(u64, 45), s.io);
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try writeLine(&aw.writer, in, s);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, aw.written()["MIMO_PREFILL_TIMELINE ".len..], .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(f64, 0), parsed.value.object.get("host_residual_ms").?.float);
    try testing.expect(parsed.value.object.get("stages").?.object.get("tail") != null);
    try testing.expectEqual(@as(i64, 1), parsed.value.object.get("moe").?.object.get("qmv").?.object.get("tiles").?.integer);
    var lw: std.Io.Writer.Allocating = .init(a);
    defer lw.deinit();
    try writeLayers(&lw.writer, in, s);
    try writeTiles(&lw.writer, in, s);
    try writeRows(&lw.writer, in, s);
    try testing.expect(std.mem.indexOf(u8, lw.written(), "rhs_bm32") != null);
}

test "glm5_next prefill timeline: the DSA split's stages sit after attn in its line and TSV; MiMo's order keeps its eleven" {
    const a = testing.allocator;
    // A DSA layer: attn [10,20) indexer [20,25) topk [25,28) sparse [28,40) attn [40,45) router [45,50), end 50.
    const marks = [_]Mark{
        .{ .t = 0, .stage = .edge },
        .{ .t = 10, .stage = .attn, .layer = 3 },
        .{ .t = 20, .stage = .indexer, .layer = 3 },
        .{ .t = 25, .stage = .topk, .layer = 3 },
        .{ .t = 28, .stage = .sparse, .layer = 3 },
        .{ .t = 40, .stage = .attn, .layer = 3 },
        .{ .t = 45, .stage = .router, .layer = 3 },
    };
    const rows = [_]Row{
        .{ .gpu_start = 11, .gpu_end = 19, .tag = stageTag(.attn, 3) },
        .{ .gpu_start = 21, .gpu_end = 24, .tag = stageTag(.indexer, 3) },
        .{ .gpu_start = 26, .gpu_end = 27, .tag = stageTag(.topk, 3) },
        .{ .gpu_start = 29, .gpu_end = 39, .tag = stageTag(.sparse, 3) },
    };
    const in: Input = .{ .rows = &rows, .marks = &marks, .tiles = &.{}, .signals = &.{}, .t_end = 50, .hidden = 4096, .inter = 2048, .order = &glm5_next_order };
    const s = try summarize(a, in);
    defer s.deinit(a);
    try testing.expectEqual(@as(u64, 3), s.layers[3].stages[@intFromEnum(Stage.indexer)].gpu);
    try testing.expectEqual(@as(u64, 10), s.layers[3].stages[@intFromEnum(Stage.sparse)].gpu);
    try testing.expectEqual(@as(u64, 15), s.stages[@intFromEnum(Stage.attn)].host);
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try writeLineOf("GLM5_NEXT", &aw.writer, in, s);
    const line = aw.written();
    try testing.expect(std.mem.startsWith(u8, line, "GLM5_NEXT_PREFILL_TIMELINE {\"wall_ms\": "));
    const parsed = try std.json.parseFromSlice(std.json.Value, a, line["GLM5_NEXT_PREFILL_TIMELINE ".len..], .{});
    defer parsed.deinit();
    const stages = parsed.value.object.get("stages").?.object;
    try testing.expectEqual(@as(usize, 14), stages.count());
    try testing.expectEqualStrings("indexer", stages.keys()[3]);
    try testing.expectEqualStrings("sparse", stages.keys()[5]);
    try testing.expectEqualStrings("router", stages.keys()[6]);
    try testing.expectEqual(@as(f64, 0), parsed.value.object.get("host_residual_ms").?.float);
    var lw: std.Io.Writer.Allocating = .init(a);
    defer lw.deinit();
    try writeLayers(&lw.writer, in, s);
    const tsv = lw.written();
    try testing.expect(std.mem.startsWith(u8, tsv, "layer\tedge_host_ms\tedge_gpu_ms\tedge_idle_ms\tpredict_host_ms"));
    try testing.expect(std.mem.indexOf(u8, tsv, "\tattn_idle_ms\tindexer_host_ms\tindexer_gpu_ms\tindexer_idle_ms\ttopk_host_ms") != null);
    try testing.expect(std.mem.indexOf(u8, tsv, "\tsparse_idle_ms\trouter_host_ms") != null);
    // MiMo's line: its eleven stages, none of the DSA split's.
    var mw: std.Io.Writer.Allocating = .init(a);
    defer mw.deinit();
    const mimo_in: Input = .{ .rows = &rows, .marks = &marks, .tiles = &.{}, .signals = &.{}, .t_end = 50, .hidden = 4096, .inter = 2048 };
    try writeLine(&mw.writer, mimo_in, s);
    try testing.expect(std.mem.startsWith(u8, mw.written(), "MIMO_PREFILL_TIMELINE {\"wall_ms\": "));
    try testing.expect(std.mem.indexOf(u8, mw.written(), "\"indexer\"") == null);
    const mp = try std.json.parseFromSlice(std.json.Value, a, mw.written()["MIMO_PREFILL_TIMELINE ".len..], .{});
    defer mp.deinit();
    try testing.expectEqual(@as(usize, 11), mp.value.object.get("stages").?.object.count());
}

test "mimo prefill timeline: a buffer Metal reports with GPUStartTime 0 is timed from its host commit, not from 0" {
    const a = testing.allocator;
    const marks = [_]Mark{
        .{ .t = 0, .stage = .edge },
        .{ .t = 10, .stage = .moe, .layer = 1, .tile = 1 },
    };
    const tiles = [_]Tile{.{ .layer = 1, .kind = .moe, .idx = 0, .rows = 1024, .bank_rows = 128, .slots = 2, .span = 2 }};
    // The tile's first buffer (its group's event wait) as Metal reported it in the p1 cells: start 0, committed at 12.
    const rows = [_]Row{
        .{ .host_commit = 12, .gpu_start = 0, .gpu_end = 20, .tag = tileTag(.moe, 0) },
        .{ .host_commit = 13, .gpu_start = 20, .gpu_end = 30, .tag = tileTag(.moe, 0) },
    };
    const in: Input = .{ .rows = &rows, .marks = &marks, .tiles = &tiles, .signals = &.{}, .t_end = 40, .hidden = 4096, .inter = 2048 };
    const s = try summarize(a, in);
    defer s.deinit(a);
    try testing.expectEqual(@as(u32, 1), s.zero_start);
    try testing.expectEqual(@as(u32, 0), s.outside);
    try testing.expectEqual(@as(u64, 12), s.tiles[0].start);
    try testing.expectEqual(@as(u64, 18), s.tiles[0].gpu);
    try testing.expectEqual(@as(u64, 18), s.paths[0].gpu);
    try testing.expectEqual(@as(u64, 18), s.busy);
    try testing.expectEqual(@as(u64, 18), s.stages[@intFromEnum(Stage.moe)].gpu);
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try writeLine(&aw.writer, in, s);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, aw.written()["MIMO_PREFILL_TIMELINE ".len..], .{});
    defer parsed.deinit();
    try testing.expectEqual(@as(i64, 1), parsed.value.object.get("zero_start_cbs").?.integer);
}

test "mimo prefill timeline: with model_queue_only the warm engine's other-queue buffers fill no idle and are counted apart (X12)" {
    const a = testing.allocator;
    const marks = [_]Mark{
        .{ .t = 0, .stage = .edge },
        .{ .t = 10, .stage = .moe, .layer = 1, .tile = 1 },
    };
    const tiles = [_]Tile{.{ .layer = 1, .kind = .moe, .idx = 0, .rows = 1024, .bank_rows = 128, .slots = 2, .span = 2 }};
    // the model's tile 10..20 and 30..35; a warm chunk 20..30 on the other queue (it would make the pass all busy)
    const rows = [_]Row{
        .{ .host_commit = 10, .gpu_start = 10, .gpu_end = 20, .tag = tileTag(.moe, 0), .queue = 1 },
        .{ .host_commit = 19, .gpu_start = 20, .gpu_end = 30, .tag = tileTag(.moe, 0), .queue = 0 },
        .{ .host_commit = 29, .gpu_start = 30, .gpu_end = 35, .tag = tileTag(.moe, 0), .queue = 1 },
    };
    const in: Input = .{ .rows = &rows, .marks = &marks, .tiles = &tiles, .signals = &.{}, .t_end = 40, .hidden = 4096, .inter = 2048, .model_queue_only = true };
    const s = try summarize(a, in);
    defer s.deinit(a);
    try testing.expectEqual(@as(u64, 15), s.busy);
    try testing.expectEqual(@as(u32, 1), s.other_queue);
    try testing.expectEqual(@as(u64, 15), s.stages[@intFromEnum(Stage.moe)].gpu);
    const all: Input = .{ .rows = &rows, .marks = &marks, .tiles = &tiles, .signals = &.{}, .t_end = 40, .hidden = 4096, .inter = 2048 };
    const s2 = try summarize(a, all);
    defer s2.deinit(a);
    try testing.expectEqual(@as(u64, 25), s2.busy);
}
