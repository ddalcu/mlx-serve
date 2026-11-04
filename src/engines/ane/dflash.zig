//! The DFlash2 drafter of an MLX pack (`<pack>/drafter`) on the Neural Engine: one program, built at
//! load like the target's, after ANEMLL Forge's ANE drafter (`dflash2_coreai_build.py`, MIT).
//!
//! `draft` writes the context rows committed since the last call (the target's tap features -> per
//! layer K / V) and runs the T-row query block [anchor, mask x (T - 1)] over [ring (W) | new rows (R)
//! | block (T)]; the host keeps the K / V ring (slot = position % W: a W-token sliding window),
//! reads the block's final hidden states through the target head's `head_raw` function and walks
//! each draft row's top-16 candidates through the selector's predecessor / successor codebooks.
//! `ctx64` ingests RP prompt rows at a time.
//!
//! fp16 range, as Forge: the residual stream is carried / RESID_SCALE from layer 0's input norm on
//! (every reader is an RMSNorm, so the scale is exact), the host scales the features by FEAT_SCALE
//! (fc -> RMSNorm).

const std = @import("std");
const env = @import("../../env.zig");
const log = @import("../../sdk/util/log.zig");
const mil = @import("mil.zig");
const pack_mod = @import("pack.zig");
const rt = @import("rt.zig");

const Builder = mil.Builder;
const Var = mil.Var;
const Shape = mil.Shape;
const Weight = mil.Weight;
const Blob = mil.Blob;

pub const TOP_K = 16;
const RESID_SCALE: f32 = 256;
pub const FEAT_SCALE: f32 = 0.125;
/// Context rows a `draft` call ingests, and a `ctx64` call.
pub const R: u32 = 8;
pub const RP: u32 = 64;
const NEG: f16 = -1e4;
/// Tests: layer 0's intermediates as extra `draft` outputs (`d_*`).
var debug_outs: ?*std.ArrayList(Var) = null;
var debug_alloc: std.mem.Allocator = undefined;

fn dbg(g: *Builder, x: Var, name: []const u8) !void {
    if (debug_outs) |l| try l.append(debug_alloc, try g.named(x, name));
}

pub const Config = struct {
    hidden: u32,
    ffn: u32,
    layers: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
    window: u32,
    theta: f64,
    eps: f32,
    block: u32,
    mask_token: u32,
    taps: []const u32,
    group: u32,
    rank: u32,

    pub fn grp(self: Config) u32 {
        return self.nh / self.nkv;
    }
    pub fn groups(self: Config) u32 {
        return self.hidden / self.group;
    }
};

/// The drafter's config.json (a DFlash2 `dflash_config` with a 2-tap conv and a candidate selector).
pub fn parseConfig(a: std.mem.Allocator, json: []const u8) !Config {
    const root = std.json.parseFromSliceLeaky(std.json.Value, a, json, .{}) catch return error.BadDrafter;
    if (root != .object) return error.BadDrafter;
    const o = root.object;
    const dc = o.get("dflash_config") orelse return error.BadDrafter;
    if (dc != .object) return error.BadDrafter;
    const d = dc.object;
    const rp = o.get("rope_parameters") orelse return error.BadDrafter;
    const taps_v = d.get("target_layer_ids") orelse return error.BadDrafter;
    if (taps_v != .array) return error.BadDrafter;
    const taps = try a.alloc(u32, taps_v.array.items.len);
    for (taps_v.array.items, taps) |v, *t| t.* = if (v == .integer and v.integer >= 0) @intCast(v.integer) else return error.BadDrafter;
    const c: Config = .{
        .hidden = try int(o.get("hidden_size")),
        .ffn = try int(o.get("intermediate_size")),
        .layers = try int(o.get("num_hidden_layers")),
        .nh = try int(o.get("num_attention_heads")),
        .nkv = try int(o.get("num_key_value_heads")),
        .hd = try int(o.get("head_dim")),
        .window = try int(o.get("sliding_window")),
        .theta = num(if (rp == .object) rp.object.get("rope_theta") else null) orelse return error.BadDrafter,
        .eps = @floatCast(num(o.get("rms_norm_eps")) orelse 1e-6),
        .block = try int(d.get("block_size")),
        .mask_token = try int(d.get("mask_token_id")),
        .taps = taps,
        .group = try int(d.get("conv_group_size")),
        .rank = try int(d.get("selector_rank")),
    };
    if ((try int(d.get("conv_kernel_size"))) != 2 or (try int(d.get("selector_top_k"))) != TOP_K) return error.UnsupportedDrafter;
    if (c.hidden % c.group != 0 or c.nh % c.nkv != 0 or c.hd > 256 or c.rank > 512) return error.UnsupportedDrafter;
    return c;
}

fn int(v: ?std.json.Value) !u32 {
    const x = v orelse return error.BadDrafter;
    return if (x == .integer and x.integer > 0) @intCast(x.integer) else error.BadDrafter;
}

fn num(v: ?std.json.Value) ?f64 {
    const x = v orelse return null;
    return switch (x) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

/// The drafter's RoPE as Forge writes it: angles in f64, cos / sin rounded to f32, then stored as f16.
pub fn ropeRow(out_cos: []align(1) f16, out_sin: []align(1) f16, pos: u64, theta: f64) void {
    const half = out_cos.len / 2;
    for (0..out_cos.len) |c| {
        const inv = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(2 * (c % half))) / @as(f64, @floatFromInt(out_cos.len)));
        const ang = @as(f64, @floatFromInt(pos)) * inv;
        out_cos[c] = @floatCast(@as(f32, @floatCast(@cos(ang))));
        out_sin[c] = @floatCast(@as(f32, @floatCast(@sin(ang))));
    }
}

pub fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// The selector walk: from `anchor`, each row picks the candidate maximizing
/// unary + successor[c] . (predecessor[prev] * hp[row]).
pub fn walk(anchor: u32, rank: usize, cand: []const [TOP_K]u32, unary: []const [TOP_K]f32, hp: []const []const f32, pred_cb: []const u16, succ_cb: []const u16, out: []u32) void {
    var prev = anchor;
    var v: [512]f32 = undefined;
    for (cand, unary, hp, out) |cs, us, h, *dst| {
        const p = pred_cb[@as(usize, prev) * rank ..][0..rank];
        for (v[0..rank], p, h[0..rank]) |*x, pc, hv| x.* = bf16ToF32(pc) * hv;
        var best: usize = 0;
        var best_s: f32 = -std.math.inf(f32);
        for (cs, us, 0..) |c, u, j| {
            const sc = succ_cb[@as(usize, c) * rank ..][0..rank];
            var s: f32 = u;
            for (sc, v[0..rank]) |x, y| s += bf16ToF32(x) * y;
            if (s > best_s) {
                best_s = s;
                best = j;
            }
        }
        prev = cs[best];
        dst.* = prev;
    }
}

/// The additive attention mask of a draft call: columns [ring (W) | new rows (R) | block (T)]. A ring
/// slot or a new row is visible to query position q when it holds a position within `window` of q.
pub fn fillMask(m: rt.Tensor, slot_pos: []const i64, rows: []const i64, q0: i64, T: usize, window: i64) void {
    const W = slot_pos.len;
    for (0..T) |i| {
        const q = q0 + @as(i64, @intCast(i));
        const row = m.row(f16, &.{ @intCast(i), 0 });
        for (slot_pos, row[0..W]) |p, *x| x.* = if (p >= 0 and @abs(q - p) < window) 0 else NEG;
        for (row[W .. W + R], 0..) |*x, j| x.* = if (j < rows.len and @abs(q - rows[j]) < window) 0 else NEG;
        @memset(row[W + R ..], 0);
    }
}

// ── the program ──

const LayerW = struct {
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    akp: Weight,
    mkp: Weight,
    in_ln: Blob.Entry,
    post_ln: Blob.Entry,
    qn: Blob.Entry,
    kn: Blob.Entry,
    /// [half][tap] per-channel base kernels, each `hidden` long.
    abase: [2][2]Blob.Entry,
    mbase: [2][2]Blob.Entry,
};

const Pending = struct { lin: pack_mod.Linear, w: Weight };

pub const Program = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    pack: *const pack_mod.Pack,
    blob: *Blob,
    g: Builder,
    pending: std.ArrayList(Pending) = .empty,

    pub fn init(allocator: std.mem.Allocator, cfg: Config, pack: *const pack_mod.Pack) !Program {
        const blob = try allocator.create(Blob);
        errdefer allocator.destroy(blob);
        blob.* = try Blob.init(std.heap.c_allocator, programBytes(cfg));
        errdefer blob.deinit();
        return .{ .allocator = allocator, .cfg = cfg, .pack = pack, .blob = blob, .g = try Builder.init(allocator, blob) };
    }

    pub fn deinit(self: *Program) void {
        self.g.deinit();
        self.blob.deinit();
        self.allocator.destroy(self.blob);
        self.pending.deinit(self.allocator);
    }

    fn linear(self: *Program, name: []const u8, out: u32, in: u32) !Weight {
        const src = try pack_mod.Linear.find(self.pack, name, out, in);
        const w: Weight = .{ .out = out, .in = in, .codes = try self.blob.add(.int8, @as(usize, out) * in), .scales = try self.blob.add(.fp16, @as(usize, out) * 2) };
        try self.pending.append(self.allocator, .{ .lin = src, .w = w });
        return w;
    }

    fn vector(self: *Program, name: []const u8, offset: usize, len: usize) !Blob.Entry {
        const t = self.pack.get(name) orelse return error.MissingWeight;
        if (t.count() < offset + len) return error.BadWeightShape;
        const vals = try self.allocator.alloc(f32, len);
        defer self.allocator.free(vals);
        for (vals, 0..) |*v, i| v.* = t.at(offset + i);
        return self.blob.addF16(vals);
    }

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

    pub fn take(self: *Program) !struct { text: [:0]const u8, blob: []u8 } {
        return .{ .text = try self.g.finish(), .blob = try self.blob.bytes.toOwnedSlice(std.heap.c_allocator) };
    }
};

pub fn programBytes(c: Config) u64 {
    const per = @as(u64, c.hidden) * (c.nh * c.hd + 2 * c.nkv * c.hd + 3 * c.ffn + 4 * c.groups() * 2) + @as(u64, c.nh) * c.hd * c.hidden;
    const n = per * c.layers + @as(u64, c.hidden) * c.hidden * c.taps.len + @as(u64, c.rank) * c.hidden;
    return n + n / 256 + (32 << 20);
}

fn fmtb(buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
    return std.fmt.bufPrint(buf, fmt, args) catch error.NameTooLong;
}

/// Both functions of the drafter program over the tensors of the pack at `drafter/`.
pub fn build(prog: *Program) !void {
    const c = prog.cfg;
    var b: [128]u8 = undefined;
    const fc = try prog.linear("fc", c.hidden, c.hidden * @as(u32, @intCast(c.taps.len)));
    const hidden_norm = try prog.vector("hidden_norm.weight", 0, c.hidden);
    const norm = try prog.vector("norm.weight", 0, c.hidden);
    const sel = try prog.linear("candidate_selector.hidden_projection", c.rank, c.hidden);
    const lw = try prog.allocator.alloc(LayerW, c.layers);
    defer prog.allocator.free(lw);
    for (lw, 0..) |*w, i| {
        w.q = try prog.linear(try fmtb(&b, "layers.{d}.self_attn.q_proj", .{i}), c.nh * c.hd, c.hidden);
        w.k = try prog.linear(try fmtb(&b, "layers.{d}.self_attn.k_proj", .{i}), c.nkv * c.hd, c.hidden);
        w.v = try prog.linear(try fmtb(&b, "layers.{d}.self_attn.v_proj", .{i}), c.nkv * c.hd, c.hidden);
        w.o = try prog.linear(try fmtb(&b, "layers.{d}.self_attn.o_proj", .{i}), c.hidden, c.nh * c.hd);
        w.gate = try prog.linear(try fmtb(&b, "layers.{d}.mlp.gate_proj", .{i}), c.ffn, c.hidden);
        w.up = try prog.linear(try fmtb(&b, "layers.{d}.mlp.up_proj", .{i}), c.ffn, c.hidden);
        w.down = try prog.linear(try fmtb(&b, "layers.{d}.mlp.down_proj", .{i}), c.hidden, c.ffn);
        w.akp = try prog.linear(try fmtb(&b, "layers.{d}.attention_conv.kernel_projection", .{i}), 4 * c.groups(), c.hidden);
        w.mkp = try prog.linear(try fmtb(&b, "layers.{d}.mlp_conv.kernel_projection", .{i}), 4 * c.groups(), c.hidden);
        w.in_ln = try prog.vector(try fmtb(&b, "layers.{d}.input_layernorm.weight", .{i}), 0, c.hidden);
        w.post_ln = try prog.vector(try fmtb(&b, "layers.{d}.post_attention_layernorm.weight", .{i}), 0, c.hidden);
        w.qn = try prog.vector(try fmtb(&b, "layers.{d}.self_attn.q_norm.weight", .{i}), 0, c.hd);
        w.kn = try prog.vector(try fmtb(&b, "layers.{d}.self_attn.k_norm.weight", .{i}), 0, c.hd);
        for (0..2) |half| for (0..2) |tap| {
            w.abase[half][tap] = try prog.vector(try fmtb(&b, "layers.{d}.attention_conv.base_kernel", .{i}), (half * 2 + tap) * c.hidden, c.hidden);
            w.mbase[half][tap] = try prog.vector(try fmtb(&b, "layers.{d}.mlp_conv.base_kernel", .{i}), (half * 2 + tap) * c.hidden, c.hidden);
        };
    }
    const g = &prog.g;
    const T = c.block;
    const W = c.window;
    const feat_rows = c.hidden * @as(u32, @intCast(c.taps.len));

    // draft
    {
        var ins: std.ArrayList(mil.Input) = .empty;
        defer ins.deinit(prog.allocator);
        var names: std.ArrayList([]u8) = .empty;
        defer {
            for (names.items) |n| prog.allocator.free(n);
            names.deinit(prog.allocator);
        }
        try ins.appendSlice(prog.allocator, &.{
            .{ .name = "feat", .shape = Shape.of(&.{ 1, feat_rows, 1, R }) },
            .{ .name = "ctx_cos", .shape = Shape.of(&.{ R, c.hd }) },
            .{ .name = "ctx_sin", .shape = Shape.of(&.{ R, c.hd }) },
            .{ .name = "x", .shape = Shape.of(&.{ 1, c.hidden, 1, T }) },
            .{ .name = "q_cos", .shape = Shape.of(&.{ T, c.hd }) },
            .{ .name = "q_sin", .shape = Shape.of(&.{ T, c.hd }) },
            .{ .name = "mask", .shape = Shape.of(&.{ T, W + R + T }) },
        });
        for (0..c.layers) |i| for ([_][]const u8{ "kc", "vc" }) |p| {
            const n = try std.fmt.allocPrint(prog.allocator, "{s}{d}", .{ p, i });
            try names.append(prog.allocator, n);
            try ins.append(prog.allocator, .{ .name = n, .shape = Shape.of(&.{ c.nkv, W, c.hd }) });
        };
        const v = try prog.allocator.alloc(Var, ins.items.len);
        defer prog.allocator.free(v);
        try g.begin("draft", ins.items, v);
        const kvs = try context(g, c, lw, fc, hidden_norm, v[0], v[1], v[2], R);
        defer prog.allocator.free(kvs);
        var x = v[3];
        var dl: std.ArrayList(Var) = .empty;
        defer dl.deinit(prog.allocator);
        if (env.get(.ANE_DRAFT_DEBUG) != null) {
            debug_outs = &dl;
            debug_alloc = prog.allocator;
        }
        defer debug_outs = null;
        for (lw, 0..) |w, i| {
            x = try layer(g, c, w, x, v[7 + 2 * i], v[8 + 2 * i], kvs[2 * i], kvs[2 * i + 1], v[4], v[5], v[6], i == 0);
            if (i == 0) {
                try dbg(g, x, "d_x0");
                debug_outs = null;
            }
        }
        const h = try rmsHidden(g, x, norm, c.hidden, c.eps);
        var outs: std.ArrayList(Var) = .empty;
        defer outs.deinit(prog.allocator);
        for (0..c.layers) |i| {
            try outs.append(prog.allocator, try g.named(kvs[2 * i], try fmtb(&b, "k_new{d}", .{i})));
            try outs.append(prog.allocator, try g.named(kvs[2 * i + 1], try fmtb(&b, "v_new{d}", .{i})));
        }
        try outs.append(prog.allocator, try g.named(try g.conv(h, try g.weight(sel), 1), "hp"));
        try outs.append(prog.allocator, try g.named(h, "hidden"));
        try outs.appendSlice(prog.allocator, dl.items);
        try g.end(outs.items);
    }
    // ctx64
    {
        var v: [3]Var = undefined;
        try g.begin("ctx64", &.{
            .{ .name = "feat", .shape = Shape.of(&.{ 1, feat_rows, 1, RP }) },
            .{ .name = "ctx_cos", .shape = Shape.of(&.{ RP, c.hd }) },
            .{ .name = "ctx_sin", .shape = Shape.of(&.{ RP, c.hd }) },
        }, &v);
        const kvs = try context(g, c, lw, fc, hidden_norm, v[0], v[1], v[2], RP);
        defer prog.allocator.free(kvs);
        var outs: std.ArrayList(Var) = .empty;
        defer outs.deinit(prog.allocator);
        for (0..c.layers) |i| {
            try outs.append(prog.allocator, try g.named(kvs[2 * i], try fmtb(&b, "k_new{d}", .{i})));
            try outs.append(prog.allocator, try g.named(kvs[2 * i + 1], try fmtb(&b, "v_new{d}", .{i})));
        }
        try g.end(outs.items);
    }
}

fn rmsHidden(g: *Builder, x: Var, w: Blob.Entry, n: u32, eps: f32) !Var {
    const m = try g.maxS(try g.max(try g.abs(x), &.{1}), 1e-3);
    const xs = try g.div(x, m);
    const e = try g.constant(&.{ 1, 1, 1, 1 }, &.{eps});
    const ms = try g.add(try g.mean(try g.mul(xs, xs), &.{1}), try g.div(e, try g.mul(m, m)));
    return g.mul(try g.mul(xs, try g.rsqrt(ms, 0)), try g.constBlob(&.{ 1, n, 1, 1 }, w));
}

fn rmsLast(g: *Builder, x: Var, w: Blob.Entry, len: u32, eps: f32) !Var {
    const ms = try g.mean(try g.mul(x, x), &.{-1});
    return g.mul(try g.mul(x, try g.rsqrt(ms, eps)), try g.constBlob(&.{len}, w));
}

fn lin(g: *Builder, x: Var, w: Weight) !Var {
    return g.conv(x, try g.weight(w), 1);
}

/// (1, n * hd, 1, rows) -> (n, rows, hd).
fn heads(g: *Builder, x: Var, n: u32, hd: u32, rows: u32) !Var {
    return g.transpose(try g.reshape(x, &.{ n, hd, rows }), &.{ 0, 2, 1 });
}

/// NeoX halves over the whole head: t (heads, rows, hd), cos / sin (rows, hd).
fn rope(g: *Builder, t: Var, cos: Var, sin: Var, hd: u32) !Var {
    const s = g.shape(t);
    const lo = try g.sliceAxis(t, 2, 0, hd / 2);
    const hi = try g.sliceAxis(t, 2, hd / 2, hd);
    _ = s;
    const rh = try g.concat(&.{ try g.mulS(hi, -1), lo }, 2);
    return g.add(try g.mul(t, cos), try g.mul(rh, sin));
}

/// Per layer K / V of `rows` context rows: fc over the taps, RMSNorm, the layer's k / v projections.
fn context(g: *Builder, c: Config, lw: []const LayerW, fc: Weight, hidden_norm: Blob.Entry, feat: Var, cos: Var, sin: Var, rows: u32) ![]Var {
    const fused = try rmsHidden(g, try lin(g, feat, fc), hidden_norm, c.hidden, c.eps);
    const out = try g.allocator.alloc(Var, 2 * lw.len);
    for (lw, 0..) |w, i| {
        const k = try rmsLast(g, try heads(g, try lin(g, fused, w.k), c.nkv, c.hd, rows), w.kn, c.hd, c.eps);
        out[2 * i] = try rope(g, k, cos, sin, c.hd);
        out[2 * i + 1] = try heads(g, try lin(g, fused, w.v), c.nkv, c.hd, rows);
    }
    return out;
}

/// The dynamic 2-tap grouped conv within the block: out[t] = (base0 + dyn0[t]) u[t] +
/// (base1 + dyn1[t]) u[t - 1], u[-1] = 0, one dynamic coefficient per group of channels.
/// `dyn` (1, 4 * groups, 1, T) holds [half][tap][group].
fn gconv(g: *Builder, c: Config, u: Var, dyn: Var, half: u32, base: [2]Blob.Entry, T: u32) !Var {
    const G = c.groups();
    const ug = try g.reshape(u, &.{ G, c.group, T });
    const d4 = try g.reshape(dyn, &.{ 4, G, T });
    var shift: [64 * 64]f32 = @splat(0);
    for (1..T) |t| shift[(t - 1) * T + t] = 1;
    const prev = try g.matmul(ug, try g.constant(&.{ T, T }, shift[0 .. T * T]), false, false);
    var out: ?Var = null;
    for (0..2) |tap| {
        const coef = try g.reshape(try g.sliceAxis(d4, 0, half * 2 + @as(u32, @intCast(tap)), half * 2 + @as(u32, @intCast(tap)) + 1), &.{ G, 1, T });
        const k = try g.add(coef, try g.constBlob(&.{ G, c.group, 1 }, base[tap]));
        const term = try g.mul(k, if (tap == 0) ug else prev);
        out = if (out) |o| try g.add(o, term) else term;
    }
    return g.reshape(out.?, &.{ 1, c.hidden, 1, T });
}

fn tsilu(g: *Builder, x: Var) !Var {
    const h = try g.mulS(x, 0.5);
    return g.mul(h, try g.addS(try g.tanh(h), 1));
}

fn layer(g: *Builder, c: Config, w: LayerW, x_in: Var, kc: Var, vc: Var, kx: Var, vx: Var, q_cos: Var, q_sin: Var, mask: Var, first: bool) !Var {
    const T = c.block;
    const W = c.window;
    const grp = c.grp();
    var x = x_in;
    var n = try rmsHidden(g, x, w.in_ln, c.hidden, c.eps);
    // The residual is carried / RESID_SCALE from here (layer 0's down projection reaches ~1.4e5,
    // past fp16); the norm above reads the embeddings unscaled.
    if (first) x = try g.mulS(x, 1.0 / RESID_SCALE);
    try dbg(g, n, "d_n");
    var dyn = try lin(g, n, w.akp);
    try dbg(g, dyn, "d_dyn");
    const a = try gconv(g, c, n, dyn, 0, w.abase[0], T);
    try dbg(g, a, "d_a");
    const q = try rope(g, try rmsLast(g, try heads(g, try lin(g, a, w.q), c.nh, c.hd, T), w.qn, c.hd, c.eps), q_cos, q_sin, c.hd);
    const kb = try rope(g, try rmsLast(g, try heads(g, try lin(g, a, w.k), c.nkv, c.hd, T), w.kn, c.hd, c.eps), q_cos, q_sin, c.hd);
    const vb = try heads(g, try lin(g, a, w.v), c.nkv, c.hd, T);
    try dbg(g, q, "d_q");
    try dbg(g, kb, "d_kb");
    const qg = try g.reshape(q, &.{ c.nkv, grp * T, c.hd });
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(c.hd)));
    const sc = try g.mulS(try g.concat(&.{
        try g.matmul(qg, kc, false, true),
        try g.matmul(qg, kx, false, true),
        try g.matmul(qg, kb, false, true),
    }, 2), scale);
    const cols = W + R + T;
    const s4 = try g.add(try g.reshape(sc, &.{ c.nkv, grp, T, cols }), mask);
    const pr = try g.reshape(try g.softmax(s4, -1), &.{ c.nkv, grp * T, cols });
    var o = try g.add(try g.matmul(try g.sliceAxis(pr, 2, 0, W), vc, false, false), try g.matmul(try g.sliceAxis(pr, 2, W, W + R), vx, false, false));
    o = try g.add(o, try g.matmul(try g.sliceAxis(pr, 2, W + R, cols), vb, false, false));
    o = try g.reshape(try g.transpose(try g.reshape(o, &.{ c.nh, T, c.hd }), &.{ 0, 2, 1 }), &.{ 1, c.nh * c.hd, 1, T });
    try dbg(g, o, "d_o");
    // The residual scale goes on the projection's INPUT: its output would overflow before a later scale.
    o = try lin(g, try g.mulS(o, 1.0 / RESID_SCALE), w.o);
    x = try g.add(x, try gconv(g, c, o, dyn, 1, w.abase[1], T));
    try dbg(g, x, "d_x1");
    n = try rmsHidden(g, x, w.post_ln, c.hidden, c.eps);
    dyn = try lin(g, n, w.mkp);
    const m = try gconv(g, c, n, dyn, 0, w.mbase[0], T);
    const y = try lin(g, try g.mulS(try g.mul(try tsilu(g, try lin(g, m, w.gate)), try lin(g, m, w.up)), 1.0 / RESID_SCALE), w.down);
    return g.add(x, try gconv(g, c, y, dyn, 1, w.mbase[1], T));
}

// ── tests ──

const testing = std.testing;

test "drafter config: the DFlash2 contract with its conv and selector" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const c = try parseConfig(arena.allocator(),
        \\{"dflash_config": {"block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": 248070,
        \\ "selector_rank": 256, "selector_top_k": 16, "target_layer_ids": [5, 19, 33, 47, 61]},
        \\ "head_dim": 128, "hidden_size": 5120, "intermediate_size": 17408, "num_attention_heads": 32,
        \\ "num_hidden_layers": 5, "num_key_value_heads": 8, "rms_norm_eps": 1e-06,
        \\ "rope_parameters": {"rope_theta": 10000000}, "sliding_window": 2048}
    );
    try testing.expectEqual(@as(u32, 320), c.groups());
    try testing.expectEqualSlices(u32, &.{ 5, 19, 33, 47, 61 }, c.taps);
    try testing.expectEqual(@as(u32, 4), c.grp());
    try testing.expectEqual(@as(f64, 1e7), c.theta);
}

test "selector walk: each row picks by unary + successor . (predecessor[prev] * hp), chaining prev" {
    const rank = 2;
    var pred: [4 * rank]u16 = @splat(0);
    var succ: [4 * rank]u16 = @splat(0);
    pred[1 * rank] = 0x3F80; // pred[1] = e0
    pred[2 * rank] = 0xBF80; // pred[2] = -e0
    succ[2 * rank] = 0x4000; // succ[2] = 2 e0
    succ[3 * rank] = 0x3F80; // succ[3] = e0
    const h = [_]f32{ 1, 0 };
    const hp = [_][]const f32{ &h, &h };
    var cand = [2][TOP_K]u32{ @splat(3), @splat(3) };
    cand[0][0] = 2; // row 0 from anchor 1: 2 (unary 0 + 2) beats 3 (unary 0.5 + 1)
    cand[1][0] = 2; // row 1 from prev 2: 2 scores 0 - 2, 3 scores 0.5 - 1: 3 wins
    var unary: [2][TOP_K]f32 = .{ @splat(-100), @splat(-100) };
    unary[0][0] = 0;
    unary[0][1] = 0.5;
    unary[1][0] = 0;
    unary[1][1] = 0.5;
    var out: [2]u32 = undefined;
    walk(1, rank, &cand, &unary, &hp, &pred, &succ, &out);
    try testing.expectEqualSlices(u32, &.{ 2, 3 }, &out);
}

test "drafter rope rounds through f32 like Forge" {
    var c: [4]f16 = undefined;
    var s: [4]f16 = undefined;
    ropeRow(&c, &s, 3, 1e7);
    try testing.expectEqual(@as(f16, @floatCast(@as(f32, @floatCast(@cos(@as(f64, 3)))))), c[0]);
    try testing.expectEqual(c[0], c[2]);
    try testing.expectEqual(@as(f16, @floatCast(@as(f32, @floatCast(@sin(3.0 / std.math.pow(f64, 1e7, 0.5)))))), s[1]);
}

// ── host side ──

const Row = struct { feat: []f16, pos: i64 };

/// Selector codebooks, mmapped from the drafter's safetensors (bf16, [vocab, rank]).
pub const Selector = struct {
    pred: []const u16,
    succ: []const u16,

    pub fn of(pack: *const pack_mod.Pack, vocab: u32, rank: u32) !Selector {
        const p = pack.get("candidate_selector.predecessor_codebook") orelse pack.get("candidate_selector.predecessor_codebook.weight") orelse return error.MissingWeight;
        const s = pack.get("candidate_selector.successor_codebook") orelse pack.get("candidate_selector.successor_codebook.weight") orelse return error.MissingWeight;
        for ([_]pack_mod.Tensor{ p, s }) |t| if (t.dtype != .bf16 or t.dim(0) < vocab or t.dim(1) != rank or @intFromPtr(t.data.ptr) % 2 != 0) return error.BadDrafter;
        const pp: [*]const u16 = @ptrCast(@alignCast(p.data.ptr));
        const sp: [*]const u16 = @ptrCast(@alignCast(s.data.ptr));
        return .{ .pred = pp[0 .. @as(usize, vocab) * rank], .succ = sp[0 .. @as(usize, vocab) * rank] };
    }
};

/// The bound buffers of one drafter function on one program instance.
pub const Fn = struct {
    req: rt.Request,
    ins: []Buf,
    outs: []Buf,

    pub fn t(self: Fn, name: []const u8, out: bool) rt.Tensor {
        for (if (out) self.outs else self.ins) |b| if (std.mem.eql(u8, b.name, name)) return b.t;
        unreachable;
    }

    pub fn surf(self: Fn, name: []const u8, out: bool) rt.SurfaceRef {
        for (if (out) self.outs else self.ins) |b| if (std.mem.eql(u8, b.name, name)) return b.surf.ref;
        unreachable;
    }
};

pub const Buf = struct { name: []const u8, surf: rt.Surface, t: rt.Tensor };

fn rankOf(name: []const u8) u8 {
    if (std.mem.eql(u8, name, "feat") or std.mem.eql(u8, name, "x") or std.mem.eql(u8, name, "hp") or std.mem.eql(u8, name, "hidden")) return 4;
    if (std.mem.endsWith(u8, name, "cos") or std.mem.endsWith(u8, name, "sin") or std.mem.eql(u8, name, "mask")) return 2;
    return 3;
}

/// Buffers for every input and output of `proc` (each its own surface) and a request over them.
pub fn bind(a: std.mem.Allocator, prog: rt.Program, instance: u32, proc: u32) !Fn {
    var bufs: [2]std.ArrayList(Buf) = .{ .empty, .empty };
    errdefer for (bufs) |l| for (l.items) |b| b.surf.release();
    var refs: [2][64]rt.SurfaceRef = undefined;
    for (0..2) |o| {
        const n = prog.ioCount(proc, o == 1);
        if (n > 64) return error.BadDrafter;
        for (0..n) |i| {
            const io = try prog.io(proc, o == 1, @intCast(i));
            const surf = try rt.Surface.create(io.bytes());
            const lanes = [4]u32{ io.batches, io.channels, io.height, io.width };
            const name = try a.dupe(u8, io.nameSlice());
            const r = rankOf(name);
            try bufs[o].append(a, .{ .name = name, .surf = surf, .t = try rt.Tensor.of(surf.base, io, lanes[4 - r ..]) });
            refs[o][i] = surf.ref;
        }
    }
    var err: rt.ErrBuf = .{};
    const req = prog.request(instance, proc, refs[0][0..bufs[0].items.len], refs[1][0..bufs[1].items.len], &err) catch |e| {
        log.err("[ane] drafter bind: {s}\n", .{err.text()});
        return e;
    };
    return .{ .req = req, .ins = bufs[0].items, .outs = bufs[1].items };
}

pub fn release(f: Fn) void {
    f.req.deinit();
    for (f.ins) |b| b.surf.release();
    for (f.outs) |b| b.surf.release();
}

/// One slot's drafter: its two functions' buffers, the K / V ring, the queued context rows.
pub const Drafter = struct {
    allocator: std.mem.Allocator,
    cfg: Config,
    draft: Fn,
    ctx64: Fn,
    /// The `kc<i>` / `vc<i>` inputs of `draft`: the ring, written by the host.
    rings: [][2]rt.Tensor,
    slot_pos: []i64,
    pending: std.ArrayList(Row) = .empty,
    feat_dim: usize,

    pub fn init(a: std.mem.Allocator, cfg: Config, draft: Fn, ctx64: Fn) !Drafter {
        var d: Drafter = .{ .allocator = a, .cfg = cfg, .draft = draft, .ctx64 = ctx64, .rings = try a.alloc([2]rt.Tensor, cfg.layers), .slot_pos = try a.alloc(i64, cfg.window), .feat_dim = @as(usize, cfg.hidden) * cfg.taps.len };
        var name: [16]u8 = undefined;
        for (d.rings, 0..) |*r, i| {
            r[0] = draft.t(std.fmt.bufPrint(&name, "kc{d}", .{i}) catch unreachable, false);
            r[1] = draft.t(std.fmt.bufPrint(&name, "vc{d}", .{i}) catch unreachable, false);
        }
        d.reset();
        return d;
    }

    pub fn deinit(self: *Drafter) void {
        self.clearPending();
        self.pending.deinit(self.allocator);
        self.allocator.free(self.rings);
        self.allocator.free(self.slot_pos);
    }

    pub fn bufferBytes(self: *const Drafter) u64 {
        var n: u64 = 0;
        for ([_]Fn{ self.draft, self.ctx64 }) |f| {
            for (f.ins) |b| n += b.surf.bytes;
            for (f.outs) |b| n += b.surf.bytes;
        }
        return n;
    }

    pub fn reset(self: *Drafter) void {
        for (self.rings) |r| for (r) |t| t.zero();
        @memset(self.slot_pos, -1);
        self.clearPending();
    }

    /// Whether the ring or the queue holds any rows (an empty drafter snapshots as a reset).
    pub fn hasContext(self: *const Drafter) bool {
        if (self.pending.items.len > 0) return true;
        for (self.slot_pos) |p| if (p >= 0) return true;
        return false;
    }

    /// Bytes `save` writes: the rings, their slot positions, then the queued rows.
    pub fn snapLen(self: *const Drafter) usize {
        var n: usize = self.slot_pos.len * @sizeOf(i64) + @sizeOf(u32);
        for (self.rings) |r| for (r) |t| {
            n += t.bytes;
        };
        return n + self.pending.items.len * (@sizeOf(i64) + self.feat_dim * @sizeOf(f16));
    }

    pub fn save(self: *const Drafter, out: []u8) void {
        var o: usize = 0;
        for (self.rings) |r| for (r) |t| put(out, &o, t.base[0..t.bytes]);
        put(out, &o, std.mem.sliceAsBytes(self.slot_pos));
        put(out, &o, std.mem.asBytes(&@as(u32, @intCast(self.pending.items.len))));
        for (self.pending.items) |r| {
            put(out, &o, std.mem.asBytes(&r.pos));
            put(out, &o, std.mem.sliceAsBytes(r.feat));
        }
    }

    pub fn load(self: *Drafter, in: []const u8) !void {
        self.clearPending();
        var o: usize = 0;
        for (self.rings) |r| for (r) |t| get(in, &o, t.base[0..t.bytes]);
        get(in, &o, std.mem.sliceAsBytes(self.slot_pos));
        var n: u32 = undefined;
        get(in, &o, std.mem.asBytes(&n));
        for (0..n) |_| {
            var pos: i64 = undefined;
            get(in, &o, std.mem.asBytes(&pos));
            const feat = try self.allocator.alloc(f16, self.feat_dim);
            errdefer self.allocator.free(feat);
            get(in, &o, std.mem.sliceAsBytes(feat));
            try self.pending.append(self.allocator, .{ .feat = feat, .pos = pos });
        }
    }

    fn put(out: []u8, o: *usize, bytes: []const u8) void {
        @memcpy(out[o.*..][0..bytes.len], bytes);
        o.* += bytes.len;
    }

    fn get(in: []const u8, o: *usize, bytes: []u8) void {
        @memcpy(bytes, in[o.*..][0..bytes.len]);
        o.* += bytes.len;
    }

    fn clearPending(self: *Drafter) void {
        for (self.pending.items) |r| self.allocator.free(r.feat);
        self.pending.clearRetainingCapacity();
    }

    /// Queue committed target feature rows (n x feat_dim, contiguous); flushed in RP-row `ctx64`
    /// calls, the last <= R go with the next draft call.
    pub fn addContext(self: *Drafter, feats: []const f16, p0: i64, n: usize) !void {
        for (0..n) |j| try self.pending.append(self.allocator, .{
            .feat = try self.allocator.dupe(f16, feats[j * self.feat_dim ..][0..self.feat_dim]),
            .pos = p0 + @as(i64, @intCast(j)),
        });
        while (self.pending.items.len > R) {
            const n_rows = @min(RP, self.pending.items.len - R);
            const rows = self.pending.items[0..n_rows];
            self.fill(self.ctx64, rows);
            try run(self.ctx64);
            self.commit(self.ctx64, rows);
            for (rows) |r| self.allocator.free(r.feat);
            self.pending.replaceRangeAssumeCapacity(0, n_rows, &.{});
        }
    }

    fn fill(self: *Drafter, f: Fn, rows: []const Row) void {
        const feat = f.t("feat", false);
        const cos = f.t("ctx_cos", false);
        const sin = f.t("ctx_sin", false);
        const d = cos.dims[1];
        const n = feat.dims[3];
        for (0..n) |j| {
            const pos: i64 = if (j < rows.len) rows[j].pos else 0;
            ropeRow(cos.row(f16, &.{ @intCast(j), 0 }), sin.row(f16, &.{ @intCast(j), 0 }), @intCast(pos), self.cfg.theta);
            for (0..feat.dims[1]) |c| feat.f16At(&.{ 0, @intCast(c), 0, @intCast(j) }).* = if (j < rows.len) @floatCast(@as(f32, rows[j].feat[c]) * FEAT_SCALE) else 0;
        }
        _ = d;
    }

    /// The new rows' K / V into their ring slots, after the call that computed them.
    fn commit(self: *Drafter, f: Fn, rows: []const Row) void {
        var name: [16]u8 = undefined;
        for (self.rings, 0..) |r, i| for (r, [_][]const u8{ "k_new", "v_new" }) |ring, kind| {
            const src = f.t(std.fmt.bufPrint(&name, "{s}{d}", .{ kind, i }) catch unreachable, true);
            for (rows, 0..) |row, j| {
                const s: u32 = @intCast(@mod(row.pos, @as(i64, self.cfg.window)));
                for (0..self.cfg.nkv) |h| @memcpy(ring.row(f16, &.{ @intCast(h), s, 0 }), src.row(f16, &.{ @intCast(h), @intCast(j), 0 }));
            }
        };
        for (rows) |row| self.slot_pos[@intCast(@mod(row.pos, @as(i64, self.cfg.window)))] = row.pos;
    }

    fn run(f: Fn) !void {
        var err: rt.ErrBuf = .{};
        f.req.run(&err) catch {
            log.err("[ane] drafter run: {s}\n", .{err.text()});
            return error.RunFailed;
        };
    }

    /// Run the draft block after `anchor` (at position p0, not yet in the target): `emb_anchor` and
    /// `emb_mask` are the target's embedding rows. The block's normed hidden states are left in the
    /// `hidden` output for the head; the queued rows are committed to the ring.
    pub fn block(self: *Drafter, p0: i64, emb_anchor: []const f32, emb_mask: []const f32) !void {
        std.debug.assert(self.pending.items.len <= R);
        const rows = self.pending.items;
        const f = self.draft;
        const T = self.cfg.block;
        self.fill(f, rows);
        const qc = f.t("q_cos", false);
        const qs = f.t("q_sin", false);
        for (0..T) |i| ropeRow(qc.row(f16, &.{ @intCast(i), 0 }), qs.row(f16, &.{ @intCast(i), 0 }), @intCast(p0 + @as(i64, @intCast(i))), self.cfg.theta);
        var rp: [R]i64 = undefined;
        for (rows, 0..) |r, j| rp[j] = r.pos;
        fillMask(f.t("mask", false), self.slot_pos, rp[0..rows.len], p0, T, self.cfg.window);
        const x = f.t("x", false);
        for (0..self.cfg.hidden) |h| for (0..T) |r| {
            x.f16At(&.{ 0, @intCast(h), 0, @intCast(r) }).* = @floatCast(if (r == 0) emb_anchor[h] else emb_mask[h]);
        };
        try run(f);
        self.commit(f, rows);
        self.clearPending();
    }

    /// Row r of the selector projection (`hp`), into `out`.
    pub fn hp(self: *const Drafter, r: usize, out: []f32) void {
        const t = self.draft.t("hp", true);
        for (out, 0..) |*v, c| v.* = t.f16At(&.{ 0, @intCast(c), 0, @intCast(r) }).*;
    }
};

test "drafter program: draft and ctx64 over one weight blob, every ring and new-row output named" {
    const a = testing.allocator;
    const c: Config = .{ .hidden = 256, .ffn = 512, .layers = 2, .nh = 4, .nkv = 2, .hd = 64, .window = 64, .theta = 1e7, .eps = 1e-6, .block = 8, .mask_token = 3, .taps = &.{ 1, 3 }, .group = 16, .rank = 32 };
    var pk: pack_mod.Pack = .{ .allocator = a, .arena = .init(a) };
    defer {
        pk.tensors.deinit(a);
        pk.arena.deinit();
    }
    const aa = pk.arena.allocator();
    const S = struct {
        fn put(p: *pack_mod.Pack, al: std.mem.Allocator, name: []const u8, shape: []const u64) !void {
            var n: u64 = 1;
            for (shape) |d| n *= d;
            const buf = try al.alloc(u16, n);
            for (buf, 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 7)) * 0.01 + 0.5)) >> 16);
            var t: pack_mod.Tensor = .{ .dtype = .bf16, .rank = @intCast(shape.len), .shape = @splat(1), .data = std.mem.sliceAsBytes(buf) };
            for (shape, 0..) |d, i| t.shape[i] = d;
            try p.tensors.put(p.allocator, try al.dupe(u8, name), t);
        }
    };
    var b: [128]u8 = undefined;
    try S.put(&pk, aa, "fc.weight", &.{ 256, 512 });
    try S.put(&pk, aa, "hidden_norm.weight", &.{256});
    try S.put(&pk, aa, "norm.weight", &.{256});
    try S.put(&pk, aa, "candidate_selector.hidden_projection.weight", &.{ 32, 256 });
    for (0..2) |i| {
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.self_attn.q_proj.weight", .{i}), &.{ 256, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.self_attn.k_proj.weight", .{i}), &.{ 128, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.self_attn.v_proj.weight", .{i}), &.{ 128, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.self_attn.o_proj.weight", .{i}), &.{ 256, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.mlp.gate_proj.weight", .{i}), &.{ 512, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.mlp.up_proj.weight", .{i}), &.{ 512, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.mlp.down_proj.weight", .{i}), &.{ 256, 512 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.attention_conv.kernel_projection.weight", .{i}), &.{ 64, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.mlp_conv.kernel_projection.weight", .{i}), &.{ 64, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.attention_conv.base_kernel", .{i}), &.{ 2, 2, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.mlp_conv.base_kernel", .{i}), &.{ 2, 2, 256 });
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.input_layernorm.weight", .{i}), &.{256});
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.post_attention_layernorm.weight", .{i}), &.{256});
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.self_attn.q_norm.weight", .{i}), &.{64});
        try S.put(&pk, aa, try fmtb(&b, "layers.{d}.self_attn.k_norm.weight", .{i}), &.{64});
    }
    var prog = try Program.init(a, c, &pk);
    defer prog.deinit();
    try build(&prog);
    try prog.fill(2);
    const out = try prog.take();
    defer std.heap.c_allocator.free(out.blob);
    for ([_][]const u8{ "func draft<ios18>(", "func ctx64<ios18>(", "tensor<fp16, [8, 80]> mask", "tensor<fp16, [1, 512, 1, 64]> feat", "-> (k_new0, v_new0, k_new1, v_new1, hp, hidden);", "-> (k_new0, v_new0, k_new1, v_new1);" }) |needle|
        try testing.expect(std.mem.indexOf(u8, out.text, needle) != null);
}

test "dump: the drafter program for the ANE runner (ANE_DRAFT_DIR, ANE_DRAFT_PACK = a drafter dir)" {
    const dir = env.get(.ANE_DRAFT_DIR) orelse return error.SkipZigTest;
    const src = env.get(.ANE_DRAFT_PACK) orelse return error.SkipZigTest;
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const json = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena.allocator(), "{s}/config.json", .{std.mem.span(src)}), arena.allocator(), .limited(1 << 20));
    const c = try parseConfig(arena.allocator(), json);
    const pk = try pack_mod.Pack.open(io, a, std.mem.span(src));
    defer pk.close();
    var prog = try Program.init(a, c, pk);
    defer prog.deinit();
    try build(&prog);
    try prog.fill(8);
    const out = try prog.take();
    defer std.heap.c_allocator.free(out.blob);
    var d = try std.Io.Dir.cwd().createDirPathOpen(io, std.mem.span(dir), .{});
    defer d.close(io);
    try d.createDirPath(io, "weights");
    try d.writeFile(io, .{ .sub_path = "model.mil", .data = out.text });
    try d.writeFile(io, .{ .sub_path = "weights/weight.bin", .data = out.blob });
}
