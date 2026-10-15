//! The qwen3_5 family (Gated DeltaNet + gated full attention, dense MLP) as Neural Engine programs,
//! built from an MLX pack. The graph follows ANEMLL Forge's ANE build (`qwen38_coreai_build.py` and
//! its MIL ancestor `qwen38_ane_chunk.py`, MIT), layer for layer:
//!
//! - A program is a CHUNK of consecutive layers. Its functions are the ENTRIES: `d<T>_<ctx>` runs P
//!   decode / verify rows, `p<TP>_<ctx>` TP prefill rows, each against a KV history of `ctx`
//!   positions. Every entry of a chunk reads the same weights (one copy on the ANE).
//! - DeltaNet states are host-owned inputs and outputs. Decode rows are committed LAZILY: a call
//!   first commits the first `commit` pending rows of the previous call into the state (exact for
//!   any count: the UT transform's leading rows do not depend on later ones), then leaves its own
//!   rows pending (`pend`). So a speculative verify needs no rollback. The prefill entry commits all
//!   its rows (8-row sub-chunks), padding rows masked out by `valid`.
//! - Attention reads the host's KV history (keys transposed: (KV head, head dim, position)) under an
//!   additive mask and returns this call's K / V rows for the host to commit.
//! - fp16 numerics, each from Forge: RMSNorm over channels in the scale-free form (x / max|x|, the
//!   squares stay in [0, 1]); silu as 0.5 x (1 + tanh(x / 2)) (the ANE's silu has ~1e-3 absolute
//!   error near 0, where nearly all DeltaNet conv outputs sit); softplus as relu(x) + log(1 + e^-|x|)
//!   (the ANE's returns 0 past ~11); q scaled by SQ and v by SV so q . S stays out of fp16 subnormals
//!   (the gated RMSNorm absorbs both through its epsilon).

const std = @import("std");
const mil = @import("ane_mil.zig");
const pack_mod = @import("ane_pack.zig");

const Builder = mil.Builder;
const Var = mil.Var;
const Shape = mil.Shape;
const Blob = mil.Blob;
const Weight = mil.Weight;
const Pack = pack_mod.Pack;
const Linear = pack_mod.Linear;

/// Rows of a decode / verify call, and the pending rows it leaves.
pub const P: u32 = 8;
/// Rows of a prefill call: the conv's weight fetch costs the same up to 64 rows on the M5 ANE.
pub const TP: u32 = 64;
/// DeltaNet sub-chunk of the prefill entry (fp16 error vs the per-token recurrence stays ~1e-3 at 8).
const C_SUB: u32 = 8;
const SQ: f32 = 16;
const SV: f32 = 64;

pub const Arch = struct {
    hidden: u32,
    ffn: u32,
    layers: u32,
    vocab: u32,
    nk: u32,
    nv: u32,
    dk: u32,
    dv: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
    rot: u32,
    rope_theta: f64,
    eps: f32,
    full_interval: u32,
    tie: bool,
    /// `q_proj` carries a sigmoid output gate per head (qwen3_5); Qwen3 dense has none.
    gate: bool = true,

    pub fn isLinear(self: Arch, l: u32) bool {
        return (l + 1) % self.full_interval != 0;
    }

    /// Whether any layer is DeltaNet (every one is attention when the interval is 1).
    pub fn hasLinear(self: Arch) bool {
        return self.full_interval > 1;
    }
    pub fn kd(self: Arch) u32 {
        return self.nk * self.dk;
    }
    pub fn vd(self: Arch) u32 {
        return self.nv * self.dv;
    }
    pub fn cdim(self: Arch) u32 {
        return 2 * self.kd() + self.vd();
    }
    pub fn grp(self: Arch) u32 {
        return self.nh / self.nkv;
    }

    /// One lane's share of a layer: half the heads (key, value, query, KV) and half the MLP. The
    /// two lanes run on the two ANE clusters at once; their outputs are partial sums.
    pub fn half(self: Arch) Arch {
        var a = self;
        a.nk /= 2;
        a.nv /= 2;
        a.nh /= 2;
        a.nkv /= 2;
        a.ffn /= 2;
        return a;
    }

    /// The shapes this graph handles, refused by name otherwise.
    pub fn check(self: Arch) error{UnsupportedArch}!void {
        if (self.full_interval == 0 or self.nh % self.nkv != 0 or self.rot % 2 != 0 or self.rot > self.hd or self.hidden % 64 != 0) return error.UnsupportedArch;
        if (self.hasLinear() and (self.nk == 0 or self.nv % self.nk != 0 or self.dv < P)) return error.UnsupportedArch;
    }

    /// Rows of `q_proj` per query head: the query, then its gate when there is one.
    pub fn qRows(self: Arch) u32 {
        return if (self.gate) 2 * self.hd else self.hd;
    }

    /// Whether two lanes can split every layer: whole heads and whole 64-wide quantization groups.
    pub fn splits(self: Arch) bool {
        return self.nk % 2 == 0 and self.nkv % 2 == 0 and self.ffn % 128 == 0 and (self.vd() / 2) % 64 == 0 and (self.nh * self.hd / 2) % 64 == 0;
    }

    /// One lane's share of a layer when `lanes` split it.
    pub fn share(self: Arch, lanes: u32) Arch {
        return if (lanes == 2) self.half() else self;
    }
};

/// One weight entry of a program: where its codes and scales go and what fills them.
const Pending = struct { lin: Linear, w: Weight };

const LayerW = struct {
    linear: bool,
    ln1: Blob.Entry,
    ln2: Blob.Entry,
    gate: Weight,
    up: Weight,
    down: Weight,
    // DeltaNet
    qkv: Weight = undefined,
    z: Weight = undefined,
    a: Weight = undefined,
    b: Weight = undefined,
    out: Weight = undefined,
    cw: [4]Blob.Entry = undefined,
    neg_a: Blob.Entry = undefined,
    dt: Blob.Entry = undefined,
    norm: Blob.Entry = undefined,
    // attention
    q: Weight = undefined,
    k: Weight = undefined,
    v: Weight = undefined,
    o: Weight = undefined,
    qn: Blob.Entry = undefined,
    kn: Blob.Entry = undefined,
};

pub const EntryKind = enum { decode, prefill };

/// A program under construction: its blob, its text, and the weights still to convert.
pub const Program = struct {
    allocator: std.mem.Allocator,
    arch: Arch,
    /// Lanes splitting each layer (1 or 2); a lane holds `arch.share(lanes)`.
    lanes: u32 = 1,
    pack: *const Pack,
    prefix: []const u8,
    blob: *Blob,
    g: Builder,
    pending: std.ArrayList(Pending) = .empty,

    pub fn init(allocator: std.mem.Allocator, arch: Arch, pack: *const Pack, prefix: []const u8, capacity: usize) !Program {
        const blob = try allocator.create(Blob);
        errdefer allocator.destroy(blob);
        blob.* = try Blob.init(std.heap.c_allocator, capacity);
        errdefer blob.deinit();
        return .{ .allocator = allocator, .arch = arch, .pack = pack, .prefix = prefix, .blob = blob, .g = try Builder.init(allocator, blob) };
    }

    pub fn deinit(self: *Program) void {
        self.g.deinit();
        self.blob.deinit();
        self.allocator.destroy(self.blob);
        self.pending.deinit(self.allocator);
    }

    /// Weight `base` (out x in) as the rows `runs` over input columns [c0, c0 + cn).
    fn linear(self: *Program, base: []const u8, out: u32, in: u32, runs: []const [2]u32, c0: u32, cn: u32) !Weight {
        const src = try (try Linear.find(self.pack, base, out, in)).view(runs, c0, cn);
        const w: Weight = .{ .out = src.out, .in = src.in, .codes = try self.blob.add(.int8, @as(usize, src.out) * src.in), .scales = try self.blob.add(.fp16, @as(usize, src.out) * 2) };
        try self.pending.append(self.allocator, .{ .lin = src, .w = w });
        return w;
    }

    /// A dense vector of the pack as an FP16 entry, `plus` added. MLX packs store the qwen3_5 norms
    /// with HF's +1 folded in (mlx-lm's sanitize), all but the DeltaNet gated norm's plain weight.
    fn vector(self: *Program, name: []const u8, len: u32, plus: f32) !Blob.Entry {
        var buf: [256]u8 = undefined;
        const n = std.fmt.bufPrint(&buf, "{s}.{s}", .{ self.prefix, name }) catch return error.MissingWeight;
        const t = self.pack.get(n) orelse return error.MissingWeight;
        if (t.count() != len) return error.BadWeightShape;
        const vals = try self.allocator.alloc(f32, len);
        defer self.allocator.free(vals);
        for (vals, 0..) |*v, i| v.* = t.at(i) + plus;
        return self.blob.addF16(vals);
    }

    /// Layer l's weights for `lane`: its share of the heads (DeltaNet key / value heads, attention
    /// query / KV heads) and of the MLP, the output projections over the matching input columns.
    fn layer(self: *Program, l: u32, lane: u1) !LayerW {
        const A = self.arch;
        const H = A.share(self.lanes);
        const ln_: u32 = lane;
        var nb: [64]u8 = undefined;
        const ln = std.fmt.bufPrint(&nb, "layers.{d}", .{l}) catch unreachable;
        var b: [256]u8 = undefined;
        const p = self.prefix;
        const all = [_][2]u32{};
        const ffn = [_][2]u32{.{ ln_ * H.ffn, H.ffn }};
        var w: LayerW = .{
            .linear = A.isLinear(l),
            .ln1 = try self.vector(try fmtb(&b, "{s}.input_layernorm.weight", .{ln}), A.hidden, 0),
            .ln2 = try self.vector(try fmtb(&b, "{s}.post_attention_layernorm.weight", .{ln}), A.hidden, 0),
            .gate = try self.linear(try fmtb(&b, "{s}.{s}.mlp.gate_proj", .{ p, ln }), A.ffn, A.hidden, &ffn, 0, A.hidden),
            .up = try self.linear(try fmtb(&b, "{s}.{s}.mlp.up_proj", .{ p, ln }), A.ffn, A.hidden, &ffn, 0, A.hidden),
            .down = try self.linear(try fmtb(&b, "{s}.{s}.mlp.down_proj", .{ p, ln }), A.hidden, A.ffn, &all, ln_ * H.ffn, H.ffn),
        };
        if (w.linear) {
            // The conv's channels and in_proj_qkv's rows: this lane's q, k and v heads.
            const ch = [_][2]u32{ .{ ln_ * H.kd(), H.kd() }, .{ A.kd() + ln_ * H.kd(), H.kd() }, .{ 2 * A.kd() + ln_ * H.vd(), H.vd() } };
            const vh = [_][2]u32{.{ ln_ * H.vd(), H.vd() }};
            const heads_ = [_][2]u32{.{ ln_ * H.nv, H.nv }};
            w.qkv = try self.linear(try fmtb(&b, "{s}.{s}.linear_attn.in_proj_qkv", .{ p, ln }), A.cdim(), A.hidden, &ch, 0, A.hidden);
            w.z = try self.linear(try fmtb(&b, "{s}.{s}.linear_attn.in_proj_z", .{ p, ln }), A.vd(), A.hidden, &vh, 0, A.hidden);
            w.a = try self.linear(try fmtb(&b, "{s}.{s}.linear_attn.in_proj_a", .{ p, ln }), A.nv, A.hidden, &heads_, 0, A.hidden);
            w.b = try self.linear(try fmtb(&b, "{s}.{s}.linear_attn.in_proj_b", .{ p, ln }), A.nv, A.hidden, &heads_, 0, A.hidden);
            w.out = try self.linear(try fmtb(&b, "{s}.{s}.linear_attn.out_proj", .{ p, ln }), A.hidden, A.vd(), &all, ln_ * H.vd(), H.vd());
            // conv1d.weight (cdim, 4, 1), oldest tap first: one (1, cdim) row per tap.
            const cw = self.pack.get(try fmtb(&b, "{s}.{s}.linear_attn.conv1d.weight", .{ p, ln })) orelse return error.MissingWeight;
            if (cw.count() != @as(u64, A.cdim()) * 4) return error.BadWeightShape;
            const taps = cw.dim(1);
            const row = try self.allocator.alloc(f32, H.cdim());
            defer self.allocator.free(row);
            for (0..4) |j| {
                var o: usize = 0;
                for (ch) |r| for (r[0]..r[0] + r[1]) |c| {
                    row[o] = if (taps == 4) cw.at(c * 4 + j) else cw.at(j * A.cdim() + c);
                    o += 1;
                };
                w.cw[j] = try self.blob.addF16(row);
            }
            const alog = self.pack.get(try fmtb(&b, "{s}.{s}.linear_attn.A_log", .{ p, ln })) orelse return error.MissingWeight;
            const dtb = self.pack.get(try fmtb(&b, "{s}.{s}.linear_attn.dt_bias", .{ p, ln })) orelse return error.MissingWeight;
            var na: [256]f32 = undefined;
            var dt: [256]f32 = undefined;
            if (H.nv > na.len or alog.count() != A.nv or dtb.count() != A.nv) return error.BadWeightShape;
            for (0..H.nv) |i| {
                na[i] = -@exp(alog.at(ln_ * H.nv + i));
                dt[i] = dtb.at(ln_ * H.nv + i);
            }
            w.neg_a = try self.blob.addF16(na[0..H.nv]);
            w.dt = try self.blob.addF16(dt[0..H.nv]);
            w.norm = try self.vector(try fmtb(&b, "{s}.linear_attn.norm.weight", .{ln}), A.dv, 0);
        } else {
            const qh = [_][2]u32{.{ ln_ * H.nh * A.qRows(), H.nh * A.qRows() }};
            const kvh = [_][2]u32{.{ ln_ * H.nkv * A.hd, H.nkv * A.hd }};
            w.q = try self.linear(try fmtb(&b, "{s}.{s}.self_attn.q_proj", .{ p, ln }), A.nh * A.qRows(), A.hidden, &qh, 0, A.hidden);
            w.k = try self.linear(try fmtb(&b, "{s}.{s}.self_attn.k_proj", .{ p, ln }), A.nkv * A.hd, A.hidden, &kvh, 0, A.hidden);
            w.v = try self.linear(try fmtb(&b, "{s}.{s}.self_attn.v_proj", .{ p, ln }), A.nkv * A.hd, A.hidden, &kvh, 0, A.hidden);
            w.o = try self.linear(try fmtb(&b, "{s}.{s}.self_attn.o_proj", .{ p, ln }), A.hidden, A.nh * A.hd, &all, ln_ * H.nh * A.hd, H.nh * A.hd);
            w.qn = try self.vector(try fmtb(&b, "{s}.self_attn.q_norm.weight", .{ln}), A.hd, 0);
            w.kn = try self.vector(try fmtb(&b, "{s}.self_attn.k_norm.weight", .{ln}), A.hd, 0);
        }
        return w;
    }

    /// Convert every pending weight into the blob (rows split over `threads`).
    pub fn fill(self: *Program, threads: usize) !void {
        const jobs = try self.allocator.alloc(pack_mod.Job, self.pending.items.len);
        defer self.allocator.free(jobs);
        for (self.pending.items, jobs) |pd, *j| j.* = .{
            .lin = pd.lin,
            .codes = std.mem.bytesAsSlice(i8, self.blob.data(pd.w.codes)),
            .scales = std.mem.bytesAsSlice(f16, self.blob.data(pd.w.scales.?)),
        };
        try pack_mod.convert(self.allocator, jobs, threads);
    }

    /// The finished text and the blob bytes (std.heap.c_allocator; ownership moves to the caller).
    pub fn take(self: *Program) !struct { text: [:0]const u8, blob: []u8 } {
        const text = try self.g.finish();
        const blob = try self.blob.bytes.toOwnedSlice(std.heap.c_allocator);
        return .{ .text = text, .blob = blob };
    }
};

fn fmtb(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch error.NameTooLong;
}

/// What the host needs to bind a unit (a lane of a chunk): which of its layers are DeltaNet
/// (`gdn_j`) and attention (`att_j`), by index within the chunk.
pub const ChunkInfo = struct {
    first: u32,
    count: u32,
    gdn_j: []const u32,
    att_j: []const u32,
};

/// The bytes one of `lanes` lanes' program over layers [first, first + count) holds (INT8 weights + scales).
pub fn unitBytes(A: Arch, first: u32, count: u32, lanes: u32) u64 {
    var n: u64 = 0;
    for (first..first + count) |l| {
        n += 3 * @as(u64, A.ffn) * A.hidden;
        n += if (A.isLinear(@intCast(l)))
            @as(u64, A.hidden) * (A.cdim() + A.vd() + 2 * A.nv) + @as(u64, A.vd()) * A.hidden
        else
            @as(u64, A.hidden) * (A.nh * A.qRows() + 2 * A.nkv * A.hd) + @as(u64, A.nh) * A.hd * A.hidden;
    }
    n /= lanes;
    return n + n / 512 + (8 << 20);
}

/// Bytes of a whole model's converted weights (every layer and the head).
pub fn modelBytes(A: Arch) u64 {
    return unitBytes(A, 0, A.layers, 1) + @as(u64, A.vocab) * A.hidden;
}

/// A model small enough to be bound by its calls' fixed cost, not by reading its weights: one lane
/// beats two.
pub fn smallModel(A: Arch) bool {
    return modelBytes(A) < 1 << 30;
}

/// A lane program's ceilings: the compiler refuses, with no reason given, a program whose functions
/// have more than ~256 inputs plus outputs in total; MAX_PROGRAM_BYTES is the largest known to compile.
pub const MAX_IO: u32 = 244;
pub const MAX_PROGRAM_BYTES: u64 = 1600 << 20;

/// Inputs plus outputs over the procedures of a lane program on layers [first, first + count)
/// (`buildUnit`; `taps` are the drafter's).
pub fn unitIo(A: Arch, first: u32, count: u32, contexts: usize, lanes: u32, taps: []const u32) u32 {
    // x and each lane's partial in, xo and po out: every round but `po` (its own input, po out);
    // `pm` returns its projections in place of po.
    const resid = 1 + lanes + 2;
    const gate: u32 = @intFromBool(A.gate);
    const c: u32 = @intCast(contexts);
    var n: u32 = 0;
    var linear: u32 = 0;
    // A fused decode's per-layer inputs and outputs (states or K / V), and its taps.
    var own: u32 = 0;
    for (first..first + count) |li| {
        const l: u32 = @intCast(li);
        if (A.isLinear(l)) {
            n += (resid - 1 + 4) + 2 + resid;
            if (lanes > 1) n += (resid + 6 + 3) + resid;
            linear += 1;
            own += 6;
        } else {
            n += (resid - 1 + 2 + 3 + gate) + (2 + gate) + resid;
            if (lanes > 1) n += c * (resid + 5 + 2) + resid;
            own += 4;
        }
        if (l > 0 and std.mem.indexOfScalar(u32, taps, l - 1) != null) own += 1;
    }
    if (lanes == 1) {
        const shared: u32 = 4 + @as(u32, if (linear > 0) 3 else 0) + @as(u32, if (linear < count) 3 else 0);
        n += (if (linear < count) c else 1) * (shared + own);
    }
    return n;
}

/// The first layer of each lane program: as many layers as fit both ceilings (fewer programs leave
/// more of the runtime's ~121 resident model handles to everything else).
pub fn chunkStarts(a: std.mem.Allocator, A: Arch, lanes: u32, contexts: usize, taps: []const u32) ![]u32 {
    var starts: std.ArrayList(u32) = .empty;
    errdefer starts.deinit(a);
    var first: u32 = 0;
    while (first < A.layers) {
        try starts.append(a, first);
        var end = first + 1;
        while (end < A.layers) : (end += 1) {
            const n = end + 1 - first;
            if (unitIo(A, first, n, contexts, lanes, taps) > MAX_IO or unitBytes(A, first, n, lanes) > MAX_PROGRAM_BYTES) break;
        }
        first = end;
    }
    return starts.toOwnedSlice(a);
}

/// The rounds of one layer on one lane. A round reads the residual `x` and every lane's partial sum
/// of the previous round (`p0`, `p1` with two lanes), adds them, and returns the new residual
/// (`xo`) and its own partial (`po`); the host runs every lane's round at once, then the next.
///
/// - `dm`: decode mixer (P rows): DeltaNet `dm<j>`, attention `dm<j>_<ctx>` per context; its
///   states / this call's K and V rows come back as with the full layer.
/// - `df`: decode MLP.
/// - `pm`: prefill projections (TP rows), for the shared mixer cores; `po`: prefill out projection
///   of the core's output; `pf`: prefill MLP.
pub const Round = enum { dm, df, pm, po, pf };

pub fn roundName(buf: []u8, r: Round, j: u32, ctx: u32) [:0]const u8 {
    return if (r == .dm and ctx > 0)
        std.fmt.bufPrintSentinel(buf, "dm{d}_{d}", .{ j, ctx }, 0) catch unreachable
    else
        std.fmt.bufPrintSentinel(buf, "{s}{d}", .{ @tagName(r), j }, 0) catch unreachable;
}

/// Emit lane `lane`'s program over layers [first, first + count) (`prog.lanes` lanes split each
/// layer): every prefill round of every layer, and the decode rounds, which one lane fuses into a
/// procedure per context (`emitDecode`; `taps` are the drafter's).
pub fn buildUnit(prog: *Program, first: u32, count: u32, lane: u1, ctxs: []const u32, taps: []const u32, arena: std.mem.Allocator) !ChunkInfo {
    var gdn: std.ArrayList(u32) = .empty;
    var att: std.ArrayList(u32) = .empty;
    const ws = try arena.alloc(LayerW, count);
    for (ws, 0..) |*w, jj| {
        const j: u32 = @intCast(jj);
        w.* = try prog.layer(first + j, lane);
        try (if (w.linear) &gdn else &att).append(arena, j);
        if (prog.lanes > 1) {
            if (w.linear) try emitRound(prog, w.*, j, .dm, 0) else for (ctxs) |c| try emitRound(prog, w.*, j, .dm, c);
            try emitRound(prog, w.*, j, .df, 0);
        }
        for ([_]Round{ .pm, .po, .pf }) |r| try emitRound(prog, w.*, j, r, 0);
    }
    if (prog.lanes == 1) {
        if (att.items.len == 0) try emitDecode(prog, ws, first, 0, taps) else for (ctxs) |c| try emitDecode(prog, ws, first, c, taps);
    }
    return .{ .first = first, .count = count, .gdn_j = gdn.items, .att_j = att.items };
}

fn emitRound(prog: *Program, w: LayerW, j: u32, r: Round, ctx: u32) !void {
    const A = prog.arch;
    const H = A.share(prog.lanes);
    const g = &prog.g;
    const T: u32 = if (r == .dm or r == .df) P else TP;
    var ins: [12]mil.Input = undefined;
    var n: usize = 0;
    var nb: [8][24]u8 = undefined;
    const S = struct {
        fn add(list: *[12]mil.Input, at: *usize, name: []const u8, shape: Shape) void {
            list[at.*] = .{ .name = name, .shape = shape };
            at.* += 1;
        }
    };
    const resid = Shape.of(&.{ 1, A.hidden, 1, T });
    const names = [_][]const u8{ "x", "p0", "p1" };
    if (r != .po) for (names[0 .. 1 + prog.lanes]) |nm| S.add(&ins, &n, nm, resid);
    const np: usize = if (r == .po) 0 else 1 + prog.lanes; // inputs before the round's own
    switch (r) {
        .dm => if (w.linear) {
            S.add(&ins, &n, "conv_sel", Shape.of(&.{ 3, P + 3 }));
            S.add(&ins, &n, "commit", Shape.of(&.{ 1, P, 1 }));
            S.add(&ins, &n, "commit_last", Shape.of(&.{ 1, P, 1 }));
            S.add(&ins, &n, try fmtb(&nb[0], "conv{d}", .{j}), Shape.of(&.{ P + 3, H.cdim() }));
            S.add(&ins, &n, try fmtb(&nb[1], "rec{d}", .{j}), Shape.of(&.{ H.nv, A.dk, A.dv }));
            S.add(&ins, &n, try fmtb(&nb[2], "pend{d}", .{j}), Shape.of(&.{ H.nv, 3 * P + 1, A.dv }));
        } else {
            S.add(&ins, &n, "cos", Shape.of(&.{ T, A.rot }));
            S.add(&ins, &n, "sin", Shape.of(&.{ T, A.rot }));
            S.add(&ins, &n, "mask", histShape(H, ctx, .mask));
            S.add(&ins, &n, try fmtb(&nb[0], "k{d}", .{j}), histShape(H, ctx, .keys));
            S.add(&ins, &n, try fmtb(&nb[1], "v{d}", .{j}), histShape(H, ctx, .values));
        },
        .pm => if (!w.linear) {
            S.add(&ins, &n, "cos", Shape.of(&.{ T, A.rot }));
            S.add(&ins, &n, "sin", Shape.of(&.{ T, A.rot }));
        },
        .po => if (w.linear) {
            S.add(&ins, &n, "og", Shape.of(&.{ 1, H.vd(), 1, T }));
        } else {
            S.add(&ins, &n, "oa", Shape.of(&.{ T, H.nh * A.hd }));
            if (A.gate) S.add(&ins, &n, "gate_in", Shape.of(&.{ T, H.nh * A.hd }));
        },
        .df, .pf => {},
    }
    var v: [12]Var = undefined;
    var fb: [32]u8 = undefined;
    try g.begin(roundName(&fb, r, j, ctx), ins[0..n], v[0..n]);
    var outs: [8]Var = undefined;
    var no: usize = 0;
    const O = struct {
        fn put(list: *[8]Var, at: *usize, gg: *Builder, x: Var, name: []const u8) !void {
            list[at.*] = try gg.named(x, name);
            at.* += 1;
        }
    };
    if (r == .po) {
        try O.put(&outs, &no, g, if (w.linear) try lin(g, v[0], w.out) else try attnOut(g, H, w, v[0], if (A.gate) v[1] else null, T), "po");
        return g.end(outs[0..no]);
    }
    var x = try g.add(v[0], v[1]);
    if (prog.lanes == 2) x = try g.add(x, v[2]);
    try O.put(&outs, &no, g, x, "xo");
    switch (r) {
        .df, .pf => try O.put(&outs, &no, g, try mlp(g, H, w, try rmsHidden(g, x, w.ln2, A)), "po"),
        .dm => {
            const mx = try decodeMixer(g, H, w, try rmsHidden(g, x, w.ln1, A), v[np..n], ctx);
            try O.put(&outs, &no, g, mx.po, "po");
            no += try putMixerState(g, w, j, mx, outs[no..]);
        },
        .pm => {
            const h = try rmsHidden(g, x, w.ln1, A);
            if (w.linear) {
                const pj = try gdnProj(g, w, h);
                for ([_]Var{ pj.qkv, pj.z, pj.a, pj.b }, [_][]const u8{ "qkv", "z", "a", "b" }) |pv, pn| try O.put(&outs, &no, g, pv, pn);
            } else {
                const pj = try attnProj(g, H, w, h, v[np], v[np + 1], T);
                try O.put(&outs, &no, g, pj.q, "q");
                try O.put(&outs, &no, g, pj.k, try fmtb(&nb[3], "k{d}_new", .{j}));
                try O.put(&outs, &no, g, pj.v, try fmtb(&nb[4], "v{d}_new", .{j}));
                if (pj.gate) |gt| try O.put(&outs, &no, g, gt, "gate");
            }
        },
        .po => unreachable,
    }
    try g.end(outs[0..no]);
}

/// A decode mixer over P rows of the normed residual `h`: its partial and the state it hands on.
/// `in` is the mixer's own inputs in `emitRound` order: DeltaNet conv_sel, commit, commit_last,
/// conv, rec, pend; attention cos, sin, mask, k, v.
const Mixed = struct { po: Var, st: [3]Var };

fn decodeMixer(g: *Builder, H: Arch, w: LayerW, h: Var, in: []const Var, ctx: u32) !Mixed {
    if (w.linear) {
        const c: Common = .{ .cos = undefined, .sin = undefined, .mask = undefined, .conv_sel = in[0], .commit = in[1], .commit_last = in[2], .conv_sel_out = null, .valid = null, .T = P, .ctx = 0 };
        const res = try gdnCoreDecode(g, H, try gdnParamsConst(g, H, w), try gdnProj(g, w, h), .{ in[3], in[4], in[5] }, c);
        return .{ .po = try lin(g, res.o, w.out), .st = .{ res.conv, res.rec, res.pend } };
    }
    const pj = try attnProj(g, H, w, h, in[0], in[1], P);
    const o = try attnCore(g, H, pj.q, pj.k, pj.v, in[3], in[4], in[2], P, ctx, ctx);
    return .{ .po = try attnOut(g, H, w, o, pj.gate, P), .st = .{ pj.k, pj.v, undefined } };
}

/// Name a decode mixer's state outputs into `out` (`conv<j>_out`, `rec<j>_out`, `pend<j>_out`;
/// `k<j>_new`, `v<j>_new`); returns how many.
fn putMixerState(g: *Builder, w: LayerW, j: u32, mx: Mixed, out: []Var) !usize {
    var nb: [24]u8 = undefined;
    if (w.linear) {
        out[0] = try g.named(mx.st[0], try fmtb(&nb, "conv{d}_out", .{j}));
        out[1] = try g.named(mx.st[1], try fmtb(&nb, "rec{d}_out", .{j}));
        out[2] = try g.named(mx.st[2], try fmtb(&nb, "pend{d}_out", .{j}));
        return 3;
    }
    out[0] = try g.named(mx.st[0], try fmtb(&nb, "k{d}_new", .{j}));
    out[1] = try g.named(mx.st[1], try fmtb(&nb, "v{d}_new", .{j}));
    return 2;
}

/// One lane's whole unit decoded in one procedure (`dc`, or `dc_<ctx>` with attention): the
/// layers' mixer and MLP back to back as their rounds would run, handing on the last residual
/// (`xo`) and partial (`po`). The residual after each drafter tap layer in [first - 1,
/// first + count - 1) comes out as `tap<l>`.
fn emitDecode(prog: *Program, ws: []const LayerW, first: u32, ctx: u32, taps: []const u32) !void {
    const A = prog.arch;
    const g = &prog.g;
    var ins: [64]mil.Input = undefined;
    var nb: [64][24]u8 = undefined;
    var n: usize = 0;
    const resid = Shape.of(&.{ 1, A.hidden, 1, P });
    var lin_n: usize = 0;
    for (ws) |w| lin_n += @intFromBool(w.linear);
    const S = struct {
        fn add(list: []mil.Input, at: *usize, name: []const u8, shape: Shape) void {
            list[at.*] = .{ .name = name, .shape = shape };
            at.* += 1;
        }
    };
    S.add(&ins, &n, "x", resid);
    S.add(&ins, &n, "p0", resid);
    if (lin_n > 0) {
        S.add(&ins, &n, "conv_sel", Shape.of(&.{ 3, P + 3 }));
        S.add(&ins, &n, "commit", Shape.of(&.{ 1, P, 1 }));
        S.add(&ins, &n, "commit_last", Shape.of(&.{ 1, P, 1 }));
    }
    if (lin_n < ws.len) {
        S.add(&ins, &n, "cos", Shape.of(&.{ P, A.rot }));
        S.add(&ins, &n, "sin", Shape.of(&.{ P, A.rot }));
        S.add(&ins, &n, "mask", histShape(A, ctx, .mask));
    }
    const shared_n = n;
    for (ws, 0..) |w, jj| {
        const j: u32 = @intCast(jj);
        const k = 3 * jj;
        if (w.linear) {
            S.add(&ins, &n, try fmtb(&nb[k], "conv{d}", .{j}), Shape.of(&.{ P + 3, A.cdim() }));
            S.add(&ins, &n, try fmtb(&nb[k + 1], "rec{d}", .{j}), Shape.of(&.{ A.nv, A.dk, A.dv }));
            S.add(&ins, &n, try fmtb(&nb[k + 2], "pend{d}", .{j}), Shape.of(&.{ A.nv, 3 * P + 1, A.dv }));
        } else {
            S.add(&ins, &n, try fmtb(&nb[k], "k{d}", .{j}), histShape(A, ctx, .keys));
            S.add(&ins, &n, try fmtb(&nb[k + 1], "v{d}", .{j}), histShape(A, ctx, .values));
        }
    }
    var v: [64]Var = undefined;
    var fb: [32]u8 = undefined;
    try g.begin(decodeName(&fb, ctx), ins[0..n], v[0..n]);
    var outs: [64]Var = undefined;
    var no: usize = 2; // xo, po go first
    var x = v[0];
    var p = v[1];
    var at = shared_n;
    for (ws, 0..) |w, jj| {
        const j: u32 = @intCast(jj);
        const layer = first + j;
        var h = try g.add(x, p);
        if (layer > 0 and std.mem.indexOfScalar(u32, taps, layer - 1) != null) {
            outs[no] = try g.named(h, try fmtb(&fb, "tap{d}", .{layer - 1}));
            no += 1;
        }
        var in: [6]Var = undefined;
        const ns: usize = if (w.linear) 3 else 2;
        if (w.linear) {
            @memcpy(in[0..3], v[2..5]);
        } else {
            const c0: usize = if (lin_n > 0) 5 else 2;
            @memcpy(in[0..3], v[c0 .. c0 + 3]);
        }
        @memcpy(in[3 .. 3 + ns], v[at .. at + ns]);
        at += ns;
        const mx = try decodeMixer(g, A, w, try rmsHidden(g, h, w.ln1, A), in[0 .. 3 + ns], ctx);
        no += try putMixerState(g, w, j, mx, outs[no..]);
        x = h;
        h = try g.add(x, mx.po);
        p = try mlp(g, A, w, try rmsHidden(g, h, w.ln2, A));
        x = h;
    }
    outs[0] = try g.named(x, "xo");
    outs[1] = try g.named(p, "po");
    try g.end(outs[0..no]);
}

pub fn decodeName(buf: []u8, ctx: u32) [:0]const u8 {
    return if (ctx > 0)
        std.fmt.bufPrintSentinel(buf, "dc_{d}", .{ctx}, 0) catch unreachable
    else
        std.fmt.bufPrintSentinel(buf, "dc", .{}, 0) catch unreachable;
}

const Common = struct {
    cos: Var,
    sin: Var,
    mask: Var,
    conv_sel: Var,
    commit: Var,
    commit_last: Var,
    conv_sel_out: ?Var,
    valid: ?Var,
    T: u32,
    ctx: u32,
};

// ── pieces ──

fn tsilu(g: *Builder, x: Var) !Var {
    const h = try g.mulS(x, 0.5);
    return g.mul(h, try g.addS(try g.tanh(h), 1));
}

fn softplus(g: *Builder, x: Var) !Var {
    return g.add(try g.relu(x), try g.log(try g.addS(try g.exp(try g.mulS(try g.abs(x), -1)), 1)));
}

/// RMSNorm over the channels of x (1, C, 1, T), scale-free: xs = x / max|x| keeps the squares in [0, 1].
fn rmsHidden(g: *Builder, x: Var, w: Blob.Entry, A: Arch) !Var {
    const m = try g.maxS(try g.max(try g.abs(x), &.{1}), 1e-3);
    const xs = try g.div(x, m);
    const eps = try g.constant(&.{ 1, 1, 1, 1 }, &.{A.eps});
    const ms = try g.add(try g.mean(try g.mul(xs, xs), &.{1}), try g.div(eps, try g.mul(m, m)));
    return g.mul(try g.mul(xs, try g.rsqrt(ms, 0)), try g.constBlob(&.{ 1, A.hidden, 1, 1 }, w));
}

/// RMSNorm over the last axis.
fn rmsLast(g: *Builder, x: Var, w: Blob.Entry, len: u32, eps: f32) !Var {
    const ms = try g.mean(try g.mul(x, x), &.{-1});
    return g.mul(try g.mul(x, try g.rsqrt(ms, eps)), try g.constBlob(&.{len}, w));
}

fn lin(g: *Builder, x: Var, w: Weight) !Var {
    return g.conv(x, try g.weight(w), 1);
}

fn mlp(g: *Builder, A: Arch, w: LayerW, h: Var) !Var {
    _ = A;
    const a = try g.mul(try tsilu(g, try lin(g, h, w.gate)), try lin(g, h, w.up));
    return lin(g, a, w.down);
}

/// Lower-triangular ones (i >= j, or i > j when strict), n x n.
fn tri(g: *Builder, n: u32, strict: bool, transposed: bool) !Var {
    var vals: [64 * 64]f32 = undefined;
    for (0..n) |i| for (0..n) |j| {
        const on = if (strict) i > j else i >= j;
        vals[if (transposed) j * n + i else i * n + j] = if (on) 1 else 0;
    };
    return g.constant(&.{ n, n }, vals[0 .. n * n]);
}

fn eye(g: *Builder, n: u32) !Var {
    var vals: [64 * 64]f32 = @splat(0);
    for (0..n) |i| vals[i * n + i] = 1;
    return g.constant(&.{ n, n }, vals[0 .. n * n]);
}

/// A DeltaNet layer's small parameters as values of the current function: constants in a fused
/// entry, inputs in the shared prefill core.
const GdnParams = struct { cw: [4]Var, neg_a: Var, dt: Var, norm: Var };

fn gdnParamsConst(g: *Builder, A: Arch, w: LayerW) !GdnParams {
    var p: GdnParams = undefined;
    for (&p.cw, w.cw) |*v, e| v.* = try g.constBlob(&.{ 1, A.cdim() }, e);
    p.neg_a = try g.constBlob(&.{ A.nv, 1, 1 }, w.neg_a);
    p.dt = try g.constBlob(&.{ A.nv, 1, 1 }, w.dt);
    p.norm = try g.constBlob(&.{A.dv}, w.norm);
    return p;
}

/// A DeltaNet layer's projections, each (1, C, 1, T).
const GdnProj = struct { qkv: Var, z: Var, a: Var, b: Var };

fn gdnProj(g: *Builder, w: LayerW, h: Var) !GdnProj {
    return .{ .qkv = try lin(g, h, w.qkv), .z = try lin(g, h, w.z), .a = try lin(g, h, w.a), .b = try lin(g, h, w.b) };
}

/// The mixer's output (normed and gated, (1, vd, 1, T), before out_proj) and its new states.
const GdnOut = struct { o: Var, conv: Var, rec: Var, pend: Var };

const Gdn = struct { qh: Var, kh: Var, vh: Var, beta: Var, gg: Var, z: Var, rows: Var, s1: Var };

/// The front of a DeltaNet core: the causal conv over [3 previous rows | this call's rows], q / k / v
/// heads, the gates, and the commit of the previous call's pending rows.
fn gdnFront(g: *Builder, A: Arch, prm: GdnParams, pj: GdnProj, st: [3]Var, c: Common) !Gdn {
    const T = c.T;
    const cdim = A.cdim();
    const qkv = try g.reshape(pj.qkv, &.{ cdim, T });
    const z = try g.transpose(try g.reshape(pj.z, &.{ A.nv, A.dv, T }), &.{ 0, 2, 1 });
    var bb = try g.reshape(pj.b, &.{ A.nv, T, 1 });
    const aa = try g.reshape(pj.a, &.{ A.nv, T, 1 });
    const prev = try g.matmul(c.conv_sel, st[0], false, false);
    const rows = try g.concat(&.{ prev, try g.transpose(qkv, &.{ 1, 0 }) }, 0);
    var conv: ?Var = null;
    for (0..4) |j| {
        const term = try g.mul(try g.sliceAxis(rows, 0, @intCast(j), @intCast(j + T)), prm.cw[j]);
        conv = if (conv) |cv| try g.add(cv, term) else term;
    }
    const cs = try g.transpose(try tsilu(g, conv.?), &.{ 1, 0 });
    const kd = A.kd();
    const qq = try g.sliceAxis(cs, 0, 0, kd);
    const kk = try g.sliceAxis(cs, 0, kd, 2 * kd);
    const vv = try g.sliceAxis(cs, 0, 2 * kd, cdim);
    const qh = try l2n(g, try heads(g, A, qq, T), 1.0 / @sqrt(@as(f32, @floatFromInt(A.dk))) * SQ);
    const kh = try l2n(g, try heads(g, A, kk, T), 1.0);
    const vh = try g.mulS(try g.transpose(try g.reshape(vv, &.{ A.nv, A.dv, T }), &.{ 0, 2, 1 }), SV);
    bb = try g.sigmoid(bb);
    var gg = try g.mul(try softplus(g, try g.add(aa, prm.dt)), prm.neg_a);
    if (c.valid) |va| {
        bb = try g.mul(bb, va);
        gg = try g.mul(gg, va);
    }
    // S' = S exp(cum_k-1) + (k_j c_j exp(min(cum_k-1 - cum_j, 0)))^T (u - wk S)
    const pend = st[2];
    const kp = try g.slice(pend, &.{ 0, 0, 0 }, &.{ A.nv, P, A.dk });
    const up = try g.slice(pend, &.{ 0, P, 0 }, &.{ A.nv, 2 * P, A.dv });
    const wkp = try g.slice(pend, &.{ 0, 2 * P, 0 }, &.{ A.nv, 3 * P, A.dk });
    const cum_p = try g.reshape(try g.slice(pend, &.{ 0, 3 * P, 0 }, &.{ A.nv, 3 * P + 1, P }), &.{ A.nv, P, 1 });
    const total = try g.sum(try g.mul(cum_p, c.commit_last), &.{1});
    const kd_ = try g.mul(try g.mul(kp, c.commit), try g.exp(try g.minS(try g.sub(total, cum_p), 0)));
    const s1 = try g.add(try g.mul(st[1], try g.exp(total)), try g.matmul(kd_, try g.sub(up, try g.matmul(wkp, st[1], false, false)), true, false));
    return .{ .qh = qh, .kh = kh, .vh = vh, .beta = bb, .gg = gg, .z = z, .rows = rows, .s1 = s1 };
}

/// (nk*dk, T) -> (nv, T, dk): value head h reads key head h / (nv / nk).
fn heads(g: *Builder, A: Arch, t: Var, T: u32) !Var {
    const x = try g.transpose(try g.reshape(t, &.{ A.nk, A.dk, T }), &.{ 0, 2, 1 });
    return g.reshape(try g.tile(try g.reshape(x, &.{ A.nk, 1, T, A.dk }), &.{ 1, A.nv / A.nk, 1, 1 }), &.{ A.nv, T, A.dk });
}

fn l2n(g: *Builder, t: Var, scale: f32) !Var {
    const ss = try g.sum(try g.mul(t, t), &.{-1});
    return g.mulS(try g.mul(t, try g.rsqrt(ss, 1e-6)), scale);
}

/// The UT transform of n rows: (I + N)^-1 as (I - N)(I + N^2)(I + N^4)... (N strictly lower, so
/// nilpotent), N^2 formed as I - (I - N)(I + N).
fn utInverse(g: *Builder, n: Var, rows: u32) !Var {
    const I = try eye(g, rows);
    var inv = try g.sub(I, n);
    var npow = n;
    var m: u32 = 1;
    while (2 * m < rows) : (m *= 2) {
        npow = try g.sub(I, try g.matmul(try g.sub(I, npow), try g.add(I, npow), false, false));
        inv = try g.matmul(inv, try g.add(I, npow), false, false);
    }
    return inv;
}

/// The gated RMSNorm of the core's output rows, as (1, vd, 1, T) for out_proj.
fn gdnFinish(g: *Builder, A: Arch, prm: GdnParams, o: Var, z: Var, T: u32) !Var {
    const eps = A.eps * (SQ * SV) * (SQ * SV);
    const ms = try g.mean(try g.mul(o, o), &.{-1});
    const on = try g.mul(try g.mul(try g.mul(o, try g.rsqrt(ms, eps)), prm.norm), try tsilu(g, z));
    return g.reshape(try g.transpose(on, &.{ 0, 2, 1 }), &.{ 1, A.vd(), 1, T });
}

/// Decode / verify core: T = P rows, left pending.
fn gdnCoreDecode(g: *Builder, A: Arch, prm: GdnParams, pj: GdnProj, st: [3]Var, c: Common) !GdnOut {
    const T = c.T;
    const f = try gdnFront(g, A, prm, pj, st, c);
    const cum = try g.reshape(try g.matmul(try g.reshape(f.gg, &.{ A.nv, 1, T }), try tri(g, T, false, true), false, false), &.{ A.nv, T, 1 });
    const pair = try g.mul(try g.exp(try g.minS(try g.sub(cum, try g.reshape(cum, &.{ A.nv, 1, T })), 0)), try tri(g, T, false, false));
    const kb = try g.mul(f.kh, f.beta);
    const vb = try g.mul(f.vh, f.beta);
    const n = try g.mul(try g.matmul(kb, f.kh, false, true), try g.mul(pair, try tri(g, T, true, false)));
    const inv = try utInverse(g, n, T);
    const u = try g.matmul(inv, vb, false, false);
    const wk = try g.matmul(inv, try g.mul(kb, try g.exp(cum)), false, false);
    const zeros = try g.zeros(&.{ A.nv, 1, A.dv - T });
    const crow = try g.concat(&.{ try g.reshape(cum, &.{ A.nv, 1, T }), zeros }, 2);
    const pend_out = try g.concat(&.{ f.kh, u, wk, crow }, 1);
    const vn = try g.sub(u, try g.matmul(wk, f.s1, false, false));
    const o = try g.add(try g.matmul(try g.mul(f.qh, try g.exp(cum)), f.s1, false, false), try g.matmul(try g.mul(try g.matmul(f.qh, f.kh, false, true), pair), vn, false, false));
    return .{ .o = try gdnFinish(g, A, prm, o, f.z, T), .conv = f.rows, .rec = f.s1, .pend = pend_out };
}

/// Prefill core: T = TP rows in C_SUB-row sub-chunks, all committed; the conv rows come back in the
/// P-row layout (the 3 ending at the last valid row, then zeros) and the pending rows as zeros.
fn gdnCorePrefill(g: *Builder, A: Arch, prm: GdnParams, pj: GdnProj, st: [3]Var, c: Common) !GdnOut {
    const T = c.T;
    const C = C_SUB;
    const NB = T / C;
    const f = try gdnFront(g, A, prm, pj, st, c);
    const q4 = try g.reshape(f.qh, &.{ A.nv, NB, C, A.dk });
    const k4 = try g.reshape(f.kh, &.{ A.nv, NB, C, A.dk });
    const v4 = try g.reshape(f.vh, &.{ A.nv, NB, C, A.dv });
    const b4 = try g.reshape(f.beta, &.{ A.nv, NB, C, 1 });
    const g4 = try g.reshape(f.gg, &.{ A.nv, NB, C, 1 });
    const cum = try g.reshape(try g.matmul(try g.reshape(g4, &.{ A.nv, NB, 1, C }), try tri(g, C, false, true), false, false), &.{ A.nv, NB, C, 1 });
    const pair = try g.mul(try g.exp(try g.minS(try g.sub(cum, try g.reshape(cum, &.{ A.nv, NB, 1, C })), 0)), try tri(g, C, false, false));
    const kb = try g.mul(k4, b4);
    const vb = try g.mul(v4, b4);
    const n = try g.mul(try g.matmul(kb, k4, false, true), try g.mul(pair, try tri(g, C, true, false)));
    const inv = try utInverse(g, n, C);
    const u = try g.matmul(inv, vb, false, false);
    const wk = try g.matmul(inv, try g.mul(kb, try g.exp(cum)), false, false);
    const total4 = try g.slice(cum, &.{ 0, 0, C - 1, 0 }, &.{ A.nv, NB, C, 1 });
    const qd = try g.mul(q4, try g.exp(cum));
    const kdc = try g.mul(k4, try g.exp(try g.sub(total4, cum)));
    const intra = try g.mul(try g.matmul(q4, k4, false, true), pair);
    var s = f.s1;
    var outs: [TP / C_SUB]Var = undefined;
    for (0..NB) |bi| {
        const b: u32 = @intCast(bi);
        const Sl = struct {
            fn of(gb: *Builder, t: Var, nv: u32, blk: u32, rows: u32, d: u32) !Var {
                return gb.reshape(try gb.slice(t, &.{ 0, blk, 0, 0 }, &.{ nv, blk + 1, rows, d }), &.{ nv, rows, d });
            }
        };
        // wk @ S and qd @ S as ONE matmul: two matmuls on the same state often fail to compile
        // ("Couldn't do topological sort").
        const ws = try g.matmul(try g.concat(&.{ try Sl.of(g, wk, A.nv, b, C, A.dk), try Sl.of(g, qd, A.nv, b, C, A.dk) }, 1), s, false, false);
        const vn = try g.sub(try Sl.of(g, u, A.nv, b, C, A.dv), try g.sliceAxis(ws, 1, 0, C));
        outs[bi] = try g.add(try g.sliceAxis(ws, 1, C, 2 * C), try g.matmul(try Sl.of(g, intra, A.nv, b, C, C), vn, false, false));
        const tb = try g.reshape(try g.slice(total4, &.{ 0, b, 0, 0 }, &.{ A.nv, b + 1, 1, 1 }), &.{ A.nv, 1, 1 });
        s = try g.add(try g.mul(s, try g.exp(tb)), try g.matmul(try Sl.of(g, kdc, A.nv, b, C, A.dk), vn, true, false));
    }
    const o = try g.concat(outs[0..NB], 1);
    const conv_out = try g.concat(&.{ try g.matmul(c.conv_sel_out.?, f.rows, false, false), try g.zeros(&.{ P, A.cdim() }) }, 0);
    return .{ .o = try gdnFinish(g, A, prm, o, f.z, T), .conv = conv_out, .rec = s, .pend = try g.mulS(st[2], 0) };
}

/// An attention layer's projections: queries (KV head, (group, row), head dim) and this call's K / V
/// rows (KV head, T, head dim), roped and normed, and the output gate (T, nh * hd) when there is one.
const AttnProj = struct { q: Var, k: Var, v: Var, gate: ?Var };

fn attnProj(g: *Builder, A: Arch, w: LayerW, h: Var, cos: Var, sin: Var, T: u32) !AttnProj {
    const hd = A.hd;
    const qg = try g.reshape(try tmajor(g, try lin(g, h, w.q), A.nh * A.qRows(), T), &.{ T, A.nh, A.qRows() });
    const qh0 = try rmsLast(g, if (A.gate) try g.slice(qg, &.{ 0, 0, 0 }, &.{ T, A.nh, hd }) else qg, w.qn, hd, A.eps);
    const gate = if (A.gate) try g.reshape(try g.slice(qg, &.{ 0, 0, hd }, &.{ T, A.nh, 2 * hd }), &.{ T, A.nh * hd }) else null;
    const kh0 = try rmsLast(g, try g.reshape(try tmajor(g, try lin(g, h, w.k), A.nkv * hd, T), &.{ T, A.nkv, hd }), w.kn, hd, A.eps);
    const vh = try g.reshape(try tmajor(g, try lin(g, h, w.v), A.nkv * hd, T), &.{ T, A.nkv, hd });
    const c3 = try g.reshape(cos, &.{ T, 1, A.rot });
    const s3 = try g.reshape(sin, &.{ T, 1, A.rot });
    const qh = try rope(g, qh0, A, T, A.nh, c3, s3);
    return .{
        .q = try g.reshape(try g.transpose(qh, &.{ 1, 0, 2 }), &.{ A.nkv, A.grp() * T, hd }),
        .k = try g.transpose(try rope(g, kh0, A, T, A.nkv, c3, s3), &.{ 1, 0, 2 }),
        .v = try g.transpose(vh, &.{ 1, 0, 2 }),
        .gate = gate,
    };
}

/// A KV history of `ctx` positions in tiles of at most KV_TILE (the compiler takes no axis past
/// 16384): keys (tile, head, dim, position), values (tile, head, position, dim), the mask (tile,
/// position). One tile drops the tile axis.
pub const KV_TILE: u32 = 8192;

pub const Hist = enum { keys, values, mask };

pub fn histShape(A: Arch, ctx: u32, what: Hist) Shape {
    const n = @max(ctx / KV_TILE, 1);
    const t = @min(ctx, KV_TILE);
    return switch (what) {
        .keys => if (n == 1) Shape.of(&.{ A.nkv, A.hd, t }) else Shape.of(&.{ n, A.nkv, A.hd, t }),
        .values => if (n == 1) Shape.of(&.{ A.nkv, t, A.hd }) else Shape.of(&.{ n, A.nkv, t, A.hd }),
        .mask => Shape.of(&.{ n, t }),
    };
}

/// Attention over [the first h positions of a `ctx` history (masked) | this call's T rows
/// (causal)]. Within one tile, one softmax; across tiles, each tile's own softmax output, merged
/// by its exponential sum against the overall max (scaled by 1/256 so the sums stay in fp16).
fn attnCore(g: *Builder, A: Arch, q: Var, k: Var, v: Var, kc: Var, vc: Var, mask: Var, T: u32, ctx: u32, h: u32) !Var {
    const hd = A.hd;
    const grp = A.grp();
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    // Causal block mask, rows (g, t) per KV head.
    const cm = try g.allocator.alloc(f32, grp * T * T);
    defer g.allocator.free(cm);
    for (0..grp * T) |r| for (0..T) |col| {
        cm[r * T + col] = if (col <= r % T) 0 else -1e4;
    };
    const sc_b = try g.add(try g.mulS(try g.matmul(q, k, false, true), scale), try g.constant(&.{ grp * T, T }, cm));
    const o = if (ctx <= KV_TILE) blk: {
        const kh = if (h < ctx) try g.sliceAxis(kc, 2, 0, h) else kc;
        const vh = if (h < ctx) try g.sliceAxis(vc, 1, 0, h) else vc;
        const mh = if (h < ctx) try g.sliceAxis(mask, 1, 0, h) else mask;
        const sc_h = try g.add(try g.mulS(try g.matmul(q, kh, false, false), scale), try g.reshape(mh, &.{ 1, 1, h }));
        const pr = try g.softmax(try g.concat(&.{ sc_h, sc_b }, 2), -1);
        break :blk try g.add(
            try g.matmul(try g.sliceAxis(pr, 2, 0, h), vh, false, false),
            try g.matmul(try g.sliceAxis(pr, 2, h, h + T), v, false, false),
        );
    } else blk: {
        const Part = struct { m: Var, pm: Var, o: Var };
        const S = struct {
            // A part's own softmax output, its max score, and 256 x its largest probability (= 256 /
            // its exponential sum, no second exp).
            fn part(gg: *Builder, sc: Var, vv: Var) !Part {
                const p = try gg.softmax(sc, -1);
                return .{ .m = try gg.max(sc, &.{-1}), .pm = try gg.mulS(try gg.max(p, &.{-1}), 256), .o = try gg.matmul(p, vv, false, false) };
            }
        };
        var parts: [64]Part = undefined;
        const n = (h + KV_TILE - 1) / KV_TILE;
        for (0..n) |i| {
            const w = @min(KV_TILE, h - @as(u32, @intCast(i)) * KV_TILE);
            const ii: u32 = @intCast(i);
            var kt = try g.reshape(try g.sliceAxis(kc, 0, ii, ii + 1), &.{ A.nkv, hd, KV_TILE });
            var vt = try g.reshape(try g.sliceAxis(vc, 0, ii, ii + 1), &.{ A.nkv, KV_TILE, hd });
            var mt = try g.sliceAxis(mask, 0, ii, ii + 1);
            if (w < KV_TILE) {
                kt = try g.sliceAxis(kt, 2, 0, w);
                vt = try g.sliceAxis(vt, 1, 0, w);
                mt = try g.sliceAxis(mt, 1, 0, w);
            }
            const sc = try g.add(try g.mulS(try g.matmul(q, kt, false, false), scale), try g.reshape(mt, &.{ 1, 1, w }));
            parts[i] = try S.part(g, sc, vt);
        }
        parts[n] = try S.part(g, sc_b, v);
        var m = parts[0].m;
        for (parts[1 .. n + 1]) |pt| m = try g.maximum(m, pt.m);
        var num: ?Var = null;
        var den: ?Var = null;
        for (parts[0 .. n + 1]) |pt| {
            const wt = try g.div(try g.exp(try g.sub(pt.m, m)), pt.pm);
            const on = try g.mul(pt.o, wt);
            num = if (num) |x| try g.add(x, on) else on;
            den = if (den) |x| try g.add(x, wt) else wt;
        }
        break :blk try g.div(num.?, den.?);
    };
    return g.reshape(try g.transpose(try g.reshape(o, &.{ A.nh, T, hd }), &.{ 1, 0, 2 }), &.{ T, A.nh * hd });
}

/// The (gated) output projection of an attention layer.
fn attnOut(g: *Builder, A: Arch, w: LayerW, o: Var, gate: ?Var, T: u32) !Var {
    const oo = if (gate) |gt| try g.mul(o, try g.sigmoid(gt)) else o;
    return lin(g, try g.reshape(try g.transpose(oo, &.{ 1, 0 }), &.{ 1, A.nh * A.hd, 1, T }), w.o);
}

/// (1, C, 1, T) -> (T, C).
fn tmajor(g: *Builder, x: Var, ch: u32, T: u32) !Var {
    return g.transpose(try g.reshape(x, &.{ ch, T }), &.{ 1, 0 });
}

/// RoPE over the first `rot` dims of each head (halves rotated, as HF's rotate_half).
fn rope(g: *Builder, t: Var, A: Arch, T: u32, n: u32, c3: Var, s3: Var) !Var {
    // Full rotary has no unrotated tail (the compiler refuses an empty slice).
    const partial = A.rot < A.hd;
    const r = if (partial) try g.slice(t, &.{ 0, 0, 0 }, &.{ T, n, A.rot }) else t;
    const rest = if (partial) try g.slice(t, &.{ 0, 0, A.rot }, &.{ T, n, A.hd }) else null;
    const r1 = try g.slice(r, &.{ 0, 0, 0 }, &.{ T, n, A.rot / 2 });
    const r2 = try g.slice(r, &.{ 0, 0, A.rot / 2 }, &.{ T, n, A.rot });
    const rh = try g.concat(&.{ try g.mulS(r2, -1), r1 }, 2);
    const roped = try g.add(try g.mul(r, c3), try g.mul(rh, s3));
    return if (rest) |tail| g.concat(&.{ roped, tail }, 2) else roped;
}

// ── the shared prefill cores ──

/// The inputs of the DeltaNet prefill core (`gdn_p64`): one layer's projections, states and
/// parameters; every DeltaNet layer of the model runs through it.
pub fn gdnCoreInputs(A: Arch) [16]mil.Input {
    const T = TP;
    return .{
        .{ .name = "qkv", .shape = Shape.of(&.{ 1, A.cdim(), 1, T }) },
        .{ .name = "z", .shape = Shape.of(&.{ 1, A.vd(), 1, T }) },
        .{ .name = "a", .shape = Shape.of(&.{ 1, A.nv, 1, T }) },
        .{ .name = "b", .shape = Shape.of(&.{ 1, A.nv, 1, T }) },
        .{ .name = "conv", .shape = Shape.of(&.{ P + 3, A.cdim() }) },
        .{ .name = "rec", .shape = Shape.of(&.{ A.nv, A.dk, A.dv }) },
        .{ .name = "pend", .shape = Shape.of(&.{ A.nv, 3 * P + 1, A.dv }) },
        .{ .name = "conv_sel", .shape = Shape.of(&.{ 3, P + 3 }) },
        .{ .name = "conv_sel_out", .shape = Shape.of(&.{ 3, T + 3 }) },
        .{ .name = "commit", .shape = Shape.of(&.{ 1, P, 1 }) },
        .{ .name = "commit_last", .shape = Shape.of(&.{ 1, P, 1 }) },
        .{ .name = "valid", .shape = Shape.of(&.{ 1, T, 1 }) },
        .{ .name = "cw", .shape = Shape.of(&.{ 4, A.cdim() }) },
        .{ .name = "neg_a", .shape = Shape.of(&.{ A.nv, 1, 1 }) },
        .{ .name = "dt", .shape = Shape.of(&.{ A.nv, 1, 1 }) },
        .{ .name = "normw", .shape = Shape.of(&.{ 1, A.dv }) },
    };
}

/// The weight-free program of the prefill mixers: `gdn_p64` and one `att_p64_<ctx>` per context.
pub fn buildCores(g: *Builder, A: Arch, ctxs: []const u32) !void {
    const T = TP;
    if (A.hasLinear()) {
        const ins = gdnCoreInputs(A);
        var v: [16]Var = undefined;
        try g.begin("gdn_p64", &ins, &v);
        var prm: GdnParams = undefined;
        for (&prm.cw, 0..) |*cwv, j| cwv.* = try g.sliceAxis(v[12], 0, @intCast(j), @intCast(j + 1));
        prm.neg_a = v[13];
        prm.dt = v[14];
        prm.norm = try g.reshape(v[15], &.{A.dv});
        const c: Common = .{ .cos = undefined, .sin = undefined, .mask = undefined, .conv_sel = v[7], .commit = v[9], .commit_last = v[10], .conv_sel_out = v[8], .valid = v[11], .T = T, .ctx = 0 };
        const r = try gdnCorePrefill(g, A, prm, .{ .qkv = v[0], .z = v[1], .a = v[2], .b = v[3] }, .{ v[4], v[5], v[6] }, c);
        try g.end(&.{ try g.named(r.o, "o"), try g.named(r.conv, "conv_out"), try g.named(r.rec, "rec_out"), try g.named(r.pend, "pend_out") });
    }
    // Per context, ATT_SLICES variants over the first h history positions of the same inputs: a
    // prompt's attention costs what its history holds, not what its context could.
    for (ctxs) |ctx| for (1..ATT_SLICES + 1) |i| {
        const h: u32 = @intCast(ctx * i / ATT_SLICES);
        var nb: [32]u8 = undefined;
        var v: [6]Var = undefined;
        try g.begin(try attCoreName(&nb, ctx, h), &.{
            .{ .name = "q", .shape = Shape.of(&.{ A.nkv, A.grp() * T, A.hd }) },
            .{ .name = "k_new", .shape = Shape.of(&.{ A.nkv, T, A.hd }) },
            .{ .name = "v_new", .shape = Shape.of(&.{ A.nkv, T, A.hd }) },
            .{ .name = "k", .shape = histShape(A, ctx, .keys) },
            .{ .name = "v", .shape = histShape(A, ctx, .values) },
            .{ .name = "mask", .shape = histShape(A, ctx, .mask) },
        }, &v);
        const o = try attnCore(g, A, v[0], v[1], v[2], v[3], v[4], v[5], T, ctx, h);
        try g.end(&.{try g.named(o, "o")});
    };
}

/// History slices of each context's prefill attention core.
pub const ATT_SLICES = 8;

pub fn attCoreName(buf: []u8, ctx: u32, h: u32) ![:0]const u8 {
    return std.fmt.bufPrintSentinel(buf, "att_p{d}_{d}_{d}", .{ TP, ctx, h }, 0);
}

// ── head ──

/// Vocabulary parts of the head program over `hidden` inputs: each part's logits are one output.
/// A part stays at or under 96 MiB of weights: the compiler cannot always split a larger kernel into
/// its 128 MB sections.
pub fn headParts(vocab: u32, hidden: u32) u32 {
    const rows: u32 = @intCast(@min(32768, (96 << 20) / hidden));
    return (vocab + rows - 1) / rows;
}

pub fn partRange(vocab: u32, parts: u32, i: u32) struct { lo: u32, hi: u32 } {
    const step = (vocab + parts - 1) / parts;
    return .{ .lo = i * step, .hi = @min(vocab, (i + 1) * step) };
}

/// The vocabulary rows of lane `lane`'s head program: the first or second half.
pub fn headHalf(vocab: u32, lane: u1, lanes: u32) struct { lo: u32, hi: u32 } {
    if (lanes == 1) return .{ .lo = 0, .hi = vocab };
    const mid = vocab / 2;
    return if (lane == 0) .{ .lo = 0, .hi = mid } else .{ .lo = mid, .hi = vocab };
}

/// Lane `lane`'s share of the vocabulary head (half of it with two lanes). `head` adds the last
/// round's residual and partials (`x`, `p0`, `p1`, (1, hidden, 1, P)), takes the final norm and
/// returns `logits<i>` (1, part, 1, P) per part of its share; `head_raw` takes hidden states
/// already normed (the drafter's, under its own norm) as `x` through the same weights.
pub fn buildHead(prog: *Program, lane: u1) !void {
    const A = prog.arch;
    var b: [256]u8 = undefined;
    const norm = try prog.vector("norm.weight", A.hidden, 0);
    const base = if (A.tie) try fmtb(&b, "{s}.embed_tokens", .{prog.prefix}) else try headBase(prog.pack, prog.prefix, &b);
    const full = try Linear.find(prog.pack, base, A.vocab, A.hidden);
    const half = headHalf(A.vocab, lane, prog.lanes);
    const parts = headParts(half.hi - half.lo, A.hidden);
    var ws: [32]Weight = undefined;
    for (0..parts) |i| {
        const r = partRange(half.hi - half.lo, parts, @intCast(i));
        const part = try full.view(&.{.{ half.lo + r.lo, r.hi - r.lo }}, 0, A.hidden);
        ws[i] = .{ .out = part.out, .in = A.hidden, .codes = try prog.blob.add(.int8, @as(usize, part.out) * A.hidden), .scales = try prog.blob.add(.fp16, @as(usize, part.out) * 2) };
        try prog.pending.append(prog.allocator, .{ .lin = part, .w = ws[i] });
    }
    const g = &prog.g;
    const resid = Shape.of(&.{ 1, A.hidden, 1, P });
    for ([_][]const u8{ "head", "head_raw" }) |fname| {
        const normed = std.mem.eql(u8, fname, "head");
        var v: [3]Var = undefined;
        const ins = [_]mil.Input{ .{ .name = "x", .shape = resid }, .{ .name = "p0", .shape = resid }, .{ .name = "p1", .shape = resid } };
        const n: usize = if (normed) 1 + prog.lanes else 1;
        try g.begin(fname, ins[0..n], v[0..n]);
        var x = v[0];
        if (normed) {
            x = try g.add(x, v[1]);
            if (prog.lanes == 2) x = try g.add(x, v[2]);
        }
        const h = if (normed) try rmsHidden(g, x, norm, A) else x;
        var outs: [32]Var = undefined;
        // Channel-major (1, part, 1, P): transposing to (P, part) on the ANE doubled the head's time.
        for (0..parts) |i| outs[i] = try g.named(try lin(g, h, ws[i]), try fmtb(&b, "logits{d}", .{i}));
        try g.end(outs[0..parts]);
    }
}

fn headBase(pack: *const Pack, prefix: []const u8, b: []u8) ![]const u8 {
    // `lm_head` sits beside the text model (`language_model.lm_head`, `lm_head`).
    const cut = std.mem.lastIndexOfScalar(u8, prefix, '.');
    const outer = if (cut) |c| prefix[0..c] else "";
    var probe: [256]u8 = undefined;
    const name = if (outer.len > 0) try fmtb(b, "{s}.lm_head", .{outer}) else try fmtb(b, "lm_head", .{});
    if (pack.getf(&probe, "{s}.weight", .{name}) != null) return name;
    return fmtb(b, "lm_head", .{});
}

// ── tests ──

const testing = std.testing;

test "head parts: every vocabulary row in exactly one part, at most 32768 rows and 96 MiB of weights" {
    const v: u32 = 248320;
    for ([_]u32{ 1024, 5120 }) |hidden| {
        const n = headParts(v, hidden);
        var covered: u32 = 0;
        for (0..n) |i| {
            const r = partRange(v, n, @intCast(i));
            try testing.expectEqual(covered, r.lo);
            try testing.expect(r.hi - r.lo <= 32768);
            try testing.expect(@as(u64, r.hi - r.lo) * hidden <= 96 << 20);
            covered = r.hi;
        }
        try testing.expectEqual(v, covered);
    }
    try testing.expectEqual(@as(u32, 8), headParts(v, 1024));
}

/// A pack of random dense bf16 tensors for a tiny `A` under the prefix "model" (tests).
pub const TestPack = struct {
    pack: Pack,
    bufs: std.ArrayList([]u16) = .empty,

    pub fn init(a: std.mem.Allocator, A: Arch, seed: u64) !*TestPack {
        const self = try a.create(TestPack);
        self.* = .{ .pack = .{ .allocator = a, .arena = .init(a) } };
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const S = struct {
            fn put(t: *TestPack, rr: std.Random, name: []const u8, shape: []const u64, scale: f32, base: f32) !void {
                var n: u64 = 1;
                for (shape) |d| n *= d;
                const buf = try t.pack.allocator.alloc(u16, n);
                for (buf) |*v| v.* = @truncate(@as(u32, @bitCast(base + scale * (rr.float(f32) * 2 - 1))) >> 16);
                try t.bufs.append(t.pack.allocator, buf);
                var ten: pack_mod.Tensor = .{ .dtype = .bf16, .rank = @intCast(shape.len), .shape = @splat(1), .data = std.mem.sliceAsBytes(buf) };
                for (shape, 0..) |d, i| ten.shape[i] = d;
                try t.pack.tensors.put(t.pack.allocator, try t.pack.arena.allocator().dupe(u8, name), ten);
            }
        };
        var b: [128]u8 = undefined;
        const H: u64 = A.hidden;
        const fan = struct {
            fn s(in: u64) f32 {
                return 1.0 / @sqrt(@as(f32, @floatFromInt(in)));
            }
        }.s;
        try S.put(self, r, "model.embed_tokens.weight", &.{ A.vocab, H }, 1, 0);
        try S.put(self, r, "model.norm.weight", &.{H}, 0.1, 0);
        if (!A.tie) try S.put(self, r, "lm_head.weight", &.{ A.vocab, H }, fan(H), 0);
        for (0..A.layers) |l| {
            try S.put(self, r, try fmtb(&b, "model.layers.{d}.input_layernorm.weight", .{l}), &.{H}, 0.1, 0);
            try S.put(self, r, try fmtb(&b, "model.layers.{d}.post_attention_layernorm.weight", .{l}), &.{H}, 0.1, 0);
            try S.put(self, r, try fmtb(&b, "model.layers.{d}.mlp.gate_proj.weight", .{l}), &.{ A.ffn, H }, fan(H), 0);
            try S.put(self, r, try fmtb(&b, "model.layers.{d}.mlp.up_proj.weight", .{l}), &.{ A.ffn, H }, fan(H), 0);
            try S.put(self, r, try fmtb(&b, "model.layers.{d}.mlp.down_proj.weight", .{l}), &.{ H, A.ffn }, fan(A.ffn), 0);
            if (A.isLinear(@intCast(l))) {
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.linear_attn.in_proj_qkv.weight", .{l}), &.{ A.cdim(), H }, fan(H), 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.linear_attn.in_proj_z.weight", .{l}), &.{ A.vd(), H }, fan(H), 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.linear_attn.in_proj_a.weight", .{l}), &.{ A.nv, H }, fan(H), 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.linear_attn.in_proj_b.weight", .{l}), &.{ A.nv, H }, fan(H), 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.linear_attn.out_proj.weight", .{l}), &.{ H, A.vd() }, fan(A.vd()), 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.linear_attn.conv1d.weight", .{l}), &.{ A.cdim(), 4, 1 }, 0.5, 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.linear_attn.A_log", .{l}), &.{A.nv}, 1.2, 0.2);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.linear_attn.dt_bias", .{l}), &.{A.nv}, 1, 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.linear_attn.norm.weight", .{l}), &.{A.dv}, 0.1, 1);
            } else {
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.self_attn.q_proj.weight", .{l}), &.{ A.nh * A.qRows(), H }, fan(H), 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.self_attn.k_proj.weight", .{l}), &.{ A.nkv * A.hd, H }, fan(H), 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.self_attn.v_proj.weight", .{l}), &.{ A.nkv * A.hd, H }, fan(H), 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.self_attn.o_proj.weight", .{l}), &.{ H, A.nh * A.hd }, fan(A.nh * A.hd), 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.self_attn.q_norm.weight", .{l}), &.{A.hd}, 0.1, 0);
                try S.put(self, r, try fmtb(&b, "model.layers.{d}.self_attn.k_norm.weight", .{l}), &.{A.hd}, 0.1, 0);
            }
        }
        return self;
    }

    pub fn deinit(self: *TestPack) void {
        const a = self.pack.allocator;
        for (self.bufs.items) |buf| a.free(buf);
        self.bufs.deinit(a);
        self.pack.tensors.deinit(a);
        self.pack.arena.deinit();
        a.destroy(self);
    }
};

pub const tiny: Arch = .{ .hidden = 256, .ffn = 512, .layers = 4, .vocab = 1000, .nk = 2, .nv = 4, .dk = 128, .dv = 128, .nh = 4, .nkv = 2, .hd = 256, .rot = 64, .rope_theta = 1e7, .eps = 1e-6, .full_interval = 4, .tie = false };

test "chunk starts: every lane program within the compiler's ceilings, as large as they allow" {
    const a = testing.allocator;
    const q27: Arch = .{ .hidden = 5120, .ffn = 17408, .layers = 64, .vocab = 248320, .nk = 16, .nv = 48, .dk = 128, .dv = 128, .nh = 24, .nkv = 4, .hd = 256, .rot = 64, .rope_theta = 1e7, .eps = 1e-6, .full_interval = 4, .tie = false };
    for ([_]Arch{ q27, tiny }) |A| for ([_]u32{ 1, 2 }) |lanes| {
        const starts = try chunkStarts(a, A, lanes, 4, &.{ 1, 2 });
        defer a.free(starts);
        try testing.expectEqual(@as(u32, 0), starts[0]);
        for (starts, 0..) |first, i| {
            const end = if (i + 1 < starts.len) starts[i + 1] else A.layers;
            try testing.expect(end > first);
            try testing.expect(unitIo(A, first, end - first, 4, lanes, &.{ 1, 2 }) <= MAX_IO);
            try testing.expect(unitBytes(A, first, end - first, lanes) <= MAX_PROGRAM_BYTES);
            // The next layer would break a ceiling.
            if (end < A.layers) try testing.expect(unitIo(A, first, end + 1 - first, 4, lanes, &.{ 1, 2 }) > MAX_IO or unitBytes(A, first, end + 1 - first, lanes) > MAX_PROGRAM_BYTES);
        }
    };
}

test "unitIo counts the inputs and outputs buildUnit emits" {
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var q3 = tiny;
    q3.full_interval = 1;
    q3.gate = false;
    q3.rot = q3.hd;
    for ([_]Arch{ tiny, q3 }) |A| for ([_]u32{ 1, 2 }) |lanes| {
        const tp = try TestPack.init(a, A, 1);
        defer tp.deinit();
        var prog = try Program.init(a, A, &tp.pack, "model", unitBytes(A, 0, 4, lanes));
        defer prog.deinit();
        prog.lanes = lanes;
        _ = try buildUnit(&prog, 0, 4, 0, &.{ 256, 512 }, &.{ 1, 2 }, arena.allocator());
        try testing.expectEqual(unitIo(A, 0, 4, 2, lanes, &.{ 1, 2 }), prog.g.io);
    };
}

test "lane program: every round takes the residual and both partials and returns its own" {
    const a = testing.allocator;
    const tp = try TestPack.init(a, tiny, 1);
    defer tp.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    for ([_]u1{ 0, 1 }) |lane| {
        var prog = try Program.init(a, tiny, &tp.pack, "model", unitBytes(tiny, 0, 4, 2));
        defer prog.deinit();
        prog.lanes = 2;
        const info = try buildUnit(&prog, 0, 4, lane, &.{256}, &.{}, arena.allocator());
        try testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, info.gdn_j);
        try testing.expectEqualSlices(u32, &.{3}, info.att_j);
        try prog.fill(2);
        const out = try prog.take();
        defer std.heap.c_allocator.free(out.blob);
        const t = out.text;
        // Half the heads: 1 key head, 2 value heads (conv channels 128 + 128 + 256), 1 KV head.
        for ([_][]const u8{
            "func dm0<ios18>(tensor<fp16, [1, 256, 1, 8]> x, tensor<fp16, [1, 256, 1, 8]> p0, tensor<fp16, [1, 256, 1, 8]> p1",
            "tensor<fp16, [11, 512]> conv0",
            "func dm3_256<ios18>(",
            "tensor<fp16, [1, 256, 256]> k3",
            "func pf3<ios18>(",
            "func po0<ios18>(tensor<fp16, [1, 256, 1, 64]> og)",
        }) |needle| try testing.expect(std.mem.indexOf(u8, t, needle) != null);
        try testing.expect(std.mem.indexOf(u8, t, "-> (xo, po, conv0_out, rec0_out, pend0_out);") != null);
        try testing.expect(std.mem.indexOf(u8, t, "-> (xo, po, k3_new, v3_new);") != null);
        try testing.expect(std.mem.indexOf(u8, t, "-> (xo, qkv, z, a, b);") != null);
        try testing.expect(std.mem.indexOfScalar(u8, out.blob, 127) != null);
    }
}

test "lane weights: a lane's rows are its heads' rows of the full layer" {
    const a = testing.allocator;
    const tp = try TestPack.init(a, tiny, 4);
    defer tp.deinit();
    var prog = try Program.init(a, tiny, &tp.pack, "model", unitBytes(tiny, 0, 4, 2));
    defer prog.deinit();
    prog.lanes = 2;
    const w = try prog.layer(0, 1);
    const H = tiny.half();
    try testing.expectEqual(H.cdim(), w.qkv.out);
    try testing.expectEqual(H.ffn, w.down.in);
    const full = try Linear.find(&tp.pack, "model.layers.0.linear_attn.in_proj_qkv", tiny.cdim(), tiny.hidden);
    const lane1 = prog.pending.items[3].lin; // gate, up, down, then qkv
    var want: [256]f32 = undefined;
    var got: [256]f32 = undefined;
    // Lane 1's first q row is the full layer's row kd / 2; its first v row is the full v row vd / 2.
    full.row(H.kd(), &want);
    lane1.row(0, &got);
    try testing.expectEqualSlices(f32, &want, &got);
    full.row(2 * tiny.kd() + H.vd(), &want);
    lane1.row(2 * H.kd(), &got);
    try testing.expectEqualSlices(f32, &want, &got);
}

test "head program: each lane's half of the vocabulary, the partials added before the final norm" {
    const a = testing.allocator;
    const tp = try TestPack.init(a, tiny, 2);
    defer tp.deinit();
    for ([_]u1{ 0, 1 }) |lane| {
        var prog = try Program.init(a, tiny, &tp.pack, "model", 4 << 20);
        defer prog.deinit();
        prog.lanes = 2;
        try buildHead(&prog, lane);
        try prog.fill(2);
        const out = try prog.take();
        defer std.heap.c_allocator.free(out.blob);
        try testing.expect(std.mem.indexOf(u8, out.text, "func head<ios18>(tensor<fp16, [1, 256, 1, 8]> x, tensor<fp16, [1, 256, 1, 8]> p0, tensor<fp16, [1, 256, 1, 8]> p1)") != null);
        try testing.expect(std.mem.indexOf(u8, out.text, "tensor<fp16, [1, 500, 1, 8]> logits0") != null);
        try testing.expect(std.mem.indexOf(u8, out.text, "func head_raw<ios18>(tensor<fp16, [1, 256, 1, 8]> x)") != null);
    }
}

test "dump: one lane program for the ANE runner (ANE_DUMP_DIR, ANE_DUMP_PACK, ANE_DUMP_FIRST/LAYERS, ANE_DUMP_CTXS, ANE_DUMP_LANE/LANES)" {
    // ANE_DUMP_CTXS: comma-separated contexts; ANE_DUMP_PACK: an MLX pack (else a tiny random one).
    const dir = std.c.getenv("ANE_DUMP_DIR") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    const layers = std.fmt.parseInt(u32, std.mem.span(std.c.getenv("ANE_DUMP_LAYERS") orelse "2"), 10) catch 2;
    const first = std.fmt.parseInt(u32, std.mem.span(std.c.getenv("ANE_DUMP_FIRST") orelse "0"), 10) catch 0;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var A = tiny;
    A.layers = 8;
    var tp: ?*TestPack = null;
    defer if (tp) |t| t.deinit();
    var real: ?*Pack = null;
    defer if (real) |r| r.close();
    var pk: *const Pack = undefined;
    var pfx: []const u8 = "model";
    if (std.c.getenv("ANE_DUMP_PACK")) |pd| {
        const dir_s = std.mem.span(pd);
        real = try Pack.open(io, a, dir_s);
        pk = real.?;
        pfx = real.?.layerPrefix().?;
        const cfg = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena.allocator(), "{s}/config.json", .{dir_s}), arena.allocator(), .limited(1 << 20));
        const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), cfg, .{});
        const t = (root.object.get("text_config") orelse root).object;
        const n = struct {
            fn f(o: std.json.ObjectMap, k: []const u8) u32 {
                return if (o.get(k)) |v| @intCast(v.integer) else 0;
            }
        }.f;
        // Qwen3 dense has no DeltaNet keys: every layer attention, full rotary, no output gate.
        const qwen3 = std.mem.eql(u8, t.get("model_type").?.string, "qwen3");
        A = .{ .hidden = n(t, "hidden_size"), .ffn = n(t, "intermediate_size"), .layers = n(t, "num_hidden_layers"), .vocab = n(t, "vocab_size"), .nk = n(t, "linear_num_key_heads"), .nv = n(t, "linear_num_value_heads"), .dk = n(t, "linear_key_head_dim"), .dv = n(t, "linear_value_head_dim"), .nh = n(t, "num_attention_heads"), .nkv = n(t, "num_key_value_heads"), .hd = n(t, "head_dim"), .rot = if (qwen3) n(t, "head_dim") else n(t, "head_dim") / 4, .rope_theta = 1e7, .eps = 1e-6, .full_interval = if (qwen3) 1 else n(t, "full_attention_interval"), .tie = false, .gate = !qwen3 };
    } else {
        tp = try TestPack.init(a, A, 1);
        pk = &tp.?.pack;
    }
    var prog = try Program.init(a, A, pk, pfx, unitBytes(A, first, layers, 2));
    defer prog.deinit();
    var ctxs: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(std.c.getenv("ANE_DUMP_CTXS") orelse "256"), ',');
    while (it.next()) |c| try ctxs.append(arena.allocator(), try std.fmt.parseInt(u32, c, 10));
    const lane: u1 = if (std.c.getenv("ANE_DUMP_LANE")) |l| @intCast(l[0] - '0') else 0;
    prog.lanes = if (std.c.getenv("ANE_DUMP_LANES")) |l| l[0] - '0' else 2;
    _ = try buildUnit(&prog, first, layers, lane, ctxs.items, &.{}, arena.allocator());
    try prog.fill(8);
    const out = try prog.take();
    defer std.heap.c_allocator.free(out.blob);
    var d = try std.Io.Dir.cwd().createDirPathOpen(io, std.mem.span(dir), .{});
    defer d.close(io);
    try d.createDirPath(io, "weights");
    try d.writeFile(io, .{ .sub_path = "model.mil", .data = out.text });
    try d.writeFile(io, .{ .sub_path = "weights/weight.bin", .data = out.blob });
}

test "dump: the shared prefill cores for the ANE runner (ANE_CORES_DIR, ANE_CORES_CTXS)" {
    const dir = std.c.getenv("ANE_CORES_DIR") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    var A = tiny;
    if (std.c.getenv("ANE_CORES_27B") != null) A = .{ .hidden = 5120, .ffn = 17408, .layers = 64, .vocab = 248320, .nk = 16, .nv = 48, .dk = 128, .dv = 128, .nh = 24, .nkv = 4, .hd = 256, .rot = 64, .rope_theta = 1e7, .eps = 1e-6, .full_interval = 4, .tie = false };
    var ctxs: std.ArrayList(u32) = .empty;
    defer ctxs.deinit(a);
    var it = std.mem.tokenizeScalar(u8, std.mem.span(std.c.getenv("ANE_CORES_CTXS") orelse ""), ',');
    while (it.next()) |c| try ctxs.append(a, try std.fmt.parseInt(u32, c, 10));
    var blob = try Blob.init(std.heap.c_allocator, 1 << 20);
    defer blob.deinit();
    var g = try Builder.init(a, &blob);
    defer g.deinit();
    if (std.c.getenv("ANE_CORES_NO_GDN") != null) {
        var x: [1]Var = undefined;
        try g.begin("nop", &.{.{ .name = "x", .shape = Shape.of(&.{ 1, 64 }) }}, &x);
        try g.end(&.{try g.named(try g.mulS(x[0], 2), "y")});
    }
    try buildCores(&g, A, ctxs.items);
    const text = try g.finish();
    var d = try std.Io.Dir.cwd().createDirPathOpen(io, std.mem.span(dir), .{});
    defer d.close(io);
    try d.createDirPath(io, "weights");
    try d.writeFile(io, .{ .sub_path = "model.mil", .data = text });
    try d.writeFile(io, .{ .sub_path = "weights/weight.bin", .data = blob.bytes.items });
}

test "dump: head variants for the ANE runner (ANE_HEAD_DIR, ANE_HEAD_PARTS, ANE_HEAD_NOT)" {
    const dir = std.c.getenv("ANE_HEAD_DIR") orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    const A: Arch = .{ .hidden = 5120, .ffn = 17408, .layers = 64, .vocab = 248320, .nk = 16, .nv = 48, .dk = 128, .dv = 128, .nh = 24, .nkv = 4, .hd = 256, .rot = 64, .rope_theta = 1e7, .eps = 1e-6, .full_interval = 4, .tie = false };
    const parts = std.fmt.parseInt(u32, std.mem.span(std.c.getenv("ANE_HEAD_PARTS") orelse "8"), 10) catch 8;
    const transpose = std.c.getenv("ANE_HEAD_NOT") == null;
    var blob = try Blob.init(std.heap.c_allocator, @as(usize, A.vocab) * A.hidden + (64 << 20));
    defer blob.deinit();
    var g = try Builder.init(a, &blob);
    defer g.deinit();
    var x: [1]Var = undefined;
    try g.begin("head", &.{.{ .name = "x", .shape = Shape.of(&.{ 1, A.hidden, 1, P }) }}, &x);
    var outs: [64]Var = undefined;
    var b: [16]u8 = undefined;
    for (0..parts) |i| {
        const r = partRange(A.vocab, parts, @intCast(i));
        const w: Weight = .{ .out = r.hi - r.lo, .in = A.hidden, .codes = try blob.add(.int8, @as(usize, r.hi - r.lo) * A.hidden), .scales = try blob.add(.fp16, @as(usize, r.hi - r.lo) * 2) };
        const y = try lin(&g, x[0], w);
        outs[i] = try g.named(if (transpose) try tmajor(&g, y, w.out, P) else y, try fmtb(&b, "logits{d}", .{i}));
    }
    try g.end(outs[0..parts]);
    const text = try g.finish();
    // Random codes: an all-zero weight streams faster than a real one.
    var prng = std.Random.DefaultPrng.init(3);
    var off: usize = 64;
    while (off + 64 <= blob.bytes.items.len) {
        const h = blob.bytes.items[off..][0..64];
        if (std.mem.readInt(u32, h[0..4], .little) != 0xDEADBEEF) break;
        const len: usize = @intCast(std.mem.readInt(u64, h[8..16], .little));
        const data = blob.bytes.items[off + 64 ..][0..len];
        if (std.mem.readInt(u32, h[4..8], .little) == 1) {
            for (std.mem.bytesAsSlice(f16, data)) |*v| v.* = 0.001;
        } else prng.random().bytes(data);
        off += 64 + len + ((64 - len % 64) % 64);
    }
    var d = try std.Io.Dir.cwd().createDirPathOpen(io, std.mem.span(dir), .{});
    defer d.close(io);
    try d.createDirPath(io, "weights");
    try d.writeFile(io, .{ .sub_path = "model.mil", .data = text });
    try d.writeFile(io, .{ .sub_path = "weights/weight.bin", .data = blob.bytes.items });
}
