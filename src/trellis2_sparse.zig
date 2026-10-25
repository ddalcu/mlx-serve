//! TRELLIS.2 (Microsoft, MIT) sparse voxels: the submanifold 3x3x3 convolution,
//! the channel-to-space upsample, and the SC-VAE decoder built from them.
//!
//! A feature row per active voxel; the voxel set is host data (`Voxels`), the
//! features are an `[n, C]` array. The convolution is FlexGEMM's EXPLICIT_GEMM:
//! `out[i] = bias + sum_k W[:, k, :] . x[nbr[i, k]]` over the 27 offsets in
//! row-major (dx, dy, dz) order, a missing neighbour contributing zero. The
//! weight is stored `[Co, 3, 3, 3, Ci]`, so `[Co, 27*Ci]` is a reshape.
//!
//! Decoder (`SparseUnetVaeDecoder`): from_latent -> per level N ConvNeXt blocks,
//! then a `C2S` up block (conv to 8x channels, channel-to-space into the voxels
//! the subdivision picks, conv) -> LayerNorm -> output. The shape decoder PREDICTS
//! the subdivision (`to_subdiv > 0`) and the texture decoder FOLLOWS it, so both
//! see the same voxel sets and share one `Plan`.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const scope_mod = @import("mlx_scope.zig");

const A = mlx.mlx_array;
const Scope = scope_mod.Scope;
const Weights = model_mod.Weights;
const Linear = scope_mod.Linear;

/// Kernel volume of the only convolution this model uses.
pub const KV: usize = 27;
pub const AXIS_BITS = 21;
const MAX_THREADS = 8;
const MIN_ROWS_PER_THREAD = 16384;
/// A convolution gathers `rows * 27 * Ci` values at a time; this bounds that block.
const GATHER_BYTES: usize = 1 << 30;
const LN_EPS: f32 = 1e-6;
const NO_ARRAY = A{ .ctx = null };

// ── Voxel sets (host) ─────────────────────────────────────────────────────────

pub fn voxelKey(x: i32, y: i32, z: i32) u64 {
    const lim: i32 = 1 << AXIS_BITS;
    std.debug.assert(x >= 0 and y >= 0 and z >= 0 and x < lim and y < lim and z < lim);
    return (@as(u64, @intCast(x)) << (2 * AXIS_BITS)) | (@as(u64, @intCast(y)) << AXIS_BITS) | @as(u64, @intCast(z));
}

fn hashSlot(key: u64, mask: usize) usize {
    var h = key *% 0x9E3779B97F4A7C15;
    h ^= h >> 29;
    return @intCast(h & mask);
}

/// Open-addressing voxel -> row table (coords are `[n, 4]` batch, x, y, z; batch is ignored).
pub const Table = struct {
    keys: []u64,
    vals: []u32,
    mask: usize,

    const EMPTY = std.math.maxInt(u64);

    pub fn init(gpa: std.mem.Allocator, coords: []const i32) !Table {
        const n = coords.len / 4;
        const cap = std.math.ceilPowerOfTwoAssert(usize, @max(16, 2 * n));
        const keys = try gpa.alloc(u64, cap);
        errdefer gpa.free(keys);
        const vals = try gpa.alloc(u32, cap);
        errdefer gpa.free(vals);
        @memset(keys, EMPTY);
        for (0..n) |i| {
            const c = coords[i * 4 ..][0..4];
            const key = voxelKey(c[1], c[2], c[3]);
            var s = hashSlot(key, cap - 1);
            while (keys[s] != EMPTY) : (s = (s + 1) & (cap - 1)) {
                if (keys[s] == key) return error.DuplicateVoxel;
            }
            keys[s] = key;
            vals[s] = @intCast(i);
        }
        return .{ .keys = keys, .vals = vals, .mask = cap - 1 };
    }

    pub fn deinit(self: Table, gpa: std.mem.Allocator) void {
        gpa.free(self.keys);
        gpa.free(self.vals);
    }

    pub fn find(self: *const Table, key: u64) ?u32 {
        var s = hashSlot(key, self.mask);
        while (self.keys[s] != EMPTY) : (s = (s + 1) & self.mask) {
            if (self.keys[s] == key) return self.vals[s];
        }
        return null;
    }
};

pub fn parallelRange(comptime Ctx: type, ctx: Ctx, n: usize, comptime body: fn (Ctx, usize, usize) void) void {
    const parts = std.math.clamp(n / MIN_ROWS_PER_THREAD, 1, MAX_THREADS);
    if (parts == 1) return body(ctx, 0, n);
    const Run = struct {
        fn run(c: Ctx, lo: usize, hi: usize) void {
            body(c, lo, hi);
        }
    };
    var threads: [MAX_THREADS]?std.Thread = @splat(null);
    for (1..parts) |k| {
        threads[k] = std.Thread.spawn(.{}, Run.run, .{ ctx, n * k / parts, n * (k + 1) / parts }) catch null;
    }
    body(ctx, 0, n / parts);
    for (1..parts) |k| {
        if (threads[k]) |t| t.join() else body(ctx, n * k / parts, n * (k + 1) / parts);
    }
}

/// `[n, 27]` neighbour rows: tap `(dx+1)*9 + (dy+1)*3 + (dz+1)` of voxel i is the row
/// of the voxel at `+ (dx, dy, dz)`, or `n` (the zero row) where none is active.
pub fn neighborMap(gpa: std.mem.Allocator, coords: []const i32) ![]i32 {
    const n = coords.len / 4;
    const table = try Table.init(gpa, coords);
    defer table.deinit(gpa);
    const out = try gpa.alloc(i32, n * KV);
    errdefer gpa.free(out);
    const Ctx = struct { t: *const Table, coords: []const i32, out: []i32, n: usize };
    const body = struct {
        fn run(c: Ctx, lo: usize, hi: usize) void {
            for (lo..hi) |i| {
                const v = c.coords[i * 4 ..][0..4];
                var k: usize = 0;
                var dx: i32 = -1;
                while (dx <= 1) : (dx += 1) {
                    var dy: i32 = -1;
                    while (dy <= 1) : (dy += 1) {
                        var dz: i32 = -1;
                        while (dz <= 1) : (dz += 1) {
                            const x = v[1] + dx;
                            const y = v[2] + dy;
                            const z = v[3] + dz;
                            const hit = if (x < 0 or y < 0 or z < 0) null else c.t.find(voxelKey(x, y, z));
                            c.out[i * KV + k] = if (hit) |h| @intCast(h) else @intCast(c.n);
                            k += 1;
                        }
                    }
                }
            }
        }
    }.run;
    parallelRange(Ctx, .{ .t = &table, .coords = coords, .out = out, .n = n }, n, body);
    return out;
}

/// The voxels a subdivision keeps: parent i's child `c` (bit 0 = x, 1 = y, 2 = z) sits at
/// `2 * parent + bits(c)`. Parents ascend, then children. `gather[j] = parent * 8 + c`.
pub const Children = struct {
    coords: []i32,
    gather: []i32,

    pub fn deinit(self: Children, gpa: std.mem.Allocator) void {
        gpa.free(self.coords);
        gpa.free(self.gather);
    }
};

pub fn subdivide(gpa: std.mem.Allocator, coords: []const i32, keep: []const bool) !Children {
    const n = coords.len / 4;
    std.debug.assert(keep.len == n * 8);
    var m: usize = 0;
    for (keep) |k| m += @intFromBool(k);
    const out = try gpa.alloc(i32, m * 4);
    errdefer gpa.free(out);
    const gather = try gpa.alloc(i32, m);
    errdefer gpa.free(gather);
    var j: usize = 0;
    for (0..n) |i| {
        const p = coords[i * 4 ..][0..4];
        for (0..8) |c| {
            if (!keep[i * 8 + c]) continue;
            out[j * 4] = p[0];
            out[j * 4 + 1] = p[1] * 2 + @as(i32, @intCast(c & 1));
            out[j * 4 + 2] = p[2] * 2 + @as(i32, @intCast((c >> 1) & 1));
            out[j * 4 + 3] = p[3] * 2 + @as(i32, @intCast((c >> 2) & 1));
            gather[j] = @intCast(i * 8 + c);
            j += 1;
        }
    }
    return .{ .coords = out, .gather = gather };
}

/// One resolution's active voxels; the device neighbour map is built on first use.
pub const Voxels = struct {
    coords: []i32,
    nbr: ?A = null,

    pub fn len(self: Voxels) usize {
        return self.coords.len / 4;
    }

    /// Device `[n, 27]` int32 (the zero row is index n). Owned here.
    pub fn neighbors(self: *Voxels, gpa: std.mem.Allocator) !A {
        if (self.nbr) |a| return a;
        const host = try neighborMap(gpa, self.coords);
        defer gpa.free(host);
        const shape = [_]c_int{ @intCast(self.len()), @intCast(KV) };
        self.nbr = mlx.mlx_array_new_data(host.ptr, &shape, 2, .int32);
        return self.nbr.?;
    }

    pub fn deinit(self: *Voxels, gpa: std.mem.Allocator) void {
        gpa.free(self.coords);
        if (self.nbr) |a| _ = mlx.mlx_array_free(a);
    }
};

/// The voxel sets of every resolution a decode visited and the subdivisions between
/// them: what a texture decode follows.
pub const Plan = struct {
    gpa: std.mem.Allocator,
    levels: std.ArrayList(Voxels) = .empty,
    /// `subs[i]`: `[levels[i].len() * 8]` keep flags of up block i.
    subs: std.ArrayList([]bool) = .empty,
    gathers: std.ArrayList([]i32) = .empty,

    /// Copies `coords`.
    pub fn init(gpa: std.mem.Allocator, coords: []const i32) !Plan {
        var p = Plan{ .gpa = gpa };
        errdefer p.deinit();
        try p.levels.append(gpa, .{ .coords = try gpa.dupe(i32, coords) });
        return p;
    }

    pub fn deinit(self: *Plan) void {
        for (self.levels.items) |*v| v.deinit(self.gpa);
        self.levels.deinit(self.gpa);
        for (self.subs.items) |s| self.gpa.free(s);
        self.subs.deinit(self.gpa);
        for (self.gathers.items) |g| self.gpa.free(g);
        self.gathers.deinit(self.gpa);
    }

    fn push(self: *Plan, keep: []bool, ch: Children) !void {
        errdefer {
            self.gpa.free(keep);
            ch.deinit(self.gpa);
        }
        try self.subs.ensureUnusedCapacity(self.gpa, 1);
        try self.gathers.ensureUnusedCapacity(self.gpa, 1);
        try self.levels.ensureUnusedCapacity(self.gpa, 1);
        self.subs.appendAssumeCapacity(keep);
        self.gathers.appendAssumeCapacity(ch.gather);
        self.levels.appendAssumeCapacity(.{ .coords = ch.coords });
    }
};

// ── Device ops ───────────────────────────────────────────────────────────────

/// `x` in `like`'s dtype (the weights are f16; the oracle runs the activations f32).
fn asLike(sc: *Scope, x: A, like: A) !A {
    const dt = mlx.mlx_array_dtype(like);
    return if (mlx.mlx_array_dtype(x) == dt) x else sc.astype(x, dt);
}

/// LayerNorm over the last axis; a null `w`/`b` is the non-affine form.
fn layerNorm(sc: *Scope, x: A, w: A, b: A, eps: f32) !A {
    if (w.ctx == null) return sc.layerNorm(x, w, b, eps);
    return sc.layerNorm(x, try asLike(sc, w, x), try asLike(sc, b, x), eps);
}

fn dim0(x: A) usize {
    return @intCast(mlx.getShape(x)[0]);
}

/// A submanifold 3x3x3 convolution; `w` `[Co, 3, 3, 3, Ci]` and `b` `[Co]` are borrowed.
pub const Conv3 = struct {
    w: A,
    b: A,
    ci: usize,
    co: usize,

    pub fn load(w: *const Weights, comptime fmt: []const u8, args: anytype) !Conv3 {
        const wt = try scope_mod.getW(w, fmt ++ ".weight", args);
        const sh = mlx.getShape(wt);
        if (sh.len != 5 or sh[1] != 3 or sh[2] != 3 or sh[3] != 3) return error.BadConvShape;
        return .{ .w = wt, .b = try scope_mod.getW(w, fmt ++ ".bias", args), .co = @intCast(sh[0]), .ci = @intCast(sh[4]) };
    }

    /// `x` `[n, Ci]`, `nbr` the voxels' `[n, 27]` map.
    pub fn forward(self: Conv3, sc: *Scope, x: A, nbr: A) !A {
        const elem: usize = mlx.mlx_array_itemsize(x);
        const rows = std.math.clamp(GATHER_BYTES / (KV * self.ci * elem), 256, @max(dim0(x), 256));
        return self.forwardRows(sc, x, nbr, rows);
    }

    /// The same in blocks of `rows` voxels so the gathered `[rows, 27*Ci]` stays bounded;
    /// each block is evaluated and released before the next is built.
    fn forwardRows(self: Conv3, sc: *Scope, x: A, nbr: A, rows: usize) !A {
        const n = dim0(x);
        const co: c_int = @intCast(self.co);
        const taps: c_int = @intCast(KV * self.ci);
        const wt = try asLike(sc, try sc.transpose(try sc.reshape(self.w, &.{ co, taps }), &.{ 1, 0 }), x);
        const bias = try asLike(sc, self.b, x);
        const xp = try sc.padAxis(x, 0, 0, 1);
        if (n <= rows) return self.block(sc, xp, nbr, wt, bias, 0, @intCast(n));
        var parts: std.ArrayList(A) = .empty;
        defer parts.deinit(sc.a);
        var lo: usize = 0;
        while (lo < n) : (lo += rows) {
            var csc = Scope.init(sc.a, sc.s);
            defer csc.deinit();
            const y = try self.block(&csc, xp, nbr, wt, bias, @intCast(lo), @intCast(@min(n, lo + rows)));
            try mlx.check(mlx.mlx_array_eval(y));
            try parts.append(sc.a, try sc.keep(csc.out(y)));
        }
        return sc.concat(parts.items, 0);
    }

    /// Voxels `lo..hi`: gather their 27 taps from `xp` (x plus the zero row), one GEMM against `wt`.
    fn block(self: Conv3, sc: *Scope, xp: A, nbr: A, wt: A, bias: A, lo: c_int, hi: c_int) !A {
        const taps: c_int = @intCast(KV * self.ci);
        const idx = try sc.slice(nbr, 0, lo, hi);
        const g = try sc.reshape(try sc.take(xp, idx), &.{ hi - lo, taps });
        return sc.add(try sc.matmul(g, wt), bias);
    }
};

/// Rows `gather` of `h` seen as `[n*8, C/8]`: the channel-to-space of a subdivision.
fn channelToSpace(sc: *Scope, h: A, gather: A) !A {
    const sh = mlx.getShape(h);
    const n = sh[0];
    const c = @divExact(sh[1], 8);
    return sc.take(try sc.reshape(h, &.{ n * 8, c }), gather);
}

// ── Decoder ───────────────────────────────────────────────────────────────────

const ConvNeXt = struct {
    conv: Conv3,
    norm_w: A,
    norm_b: A,
    fc1: Linear,
    fc2: Linear,

    fn load(w: *const Weights, stage: usize, idx: usize, ch: usize) !ConvNeXt {
        return .{
            .conv = try Conv3.load(w, "blocks.{d}.{d}.conv", .{ stage, idx }),
            .norm_w = try scope_mod.getW(w, "blocks.{d}.{d}.norm.weight", .{ stage, idx }),
            .norm_b = try scope_mod.getW(w, "blocks.{d}.{d}.norm.bias", .{ stage, idx }),
            .fc1 = try scope_mod.linear(w, "blocks.{d}.{d}.mlp.0", .{ stage, idx }, @intCast(ch)),
            .fc2 = try scope_mod.linear(w, "blocks.{d}.{d}.mlp.2", .{ stage, idx }, @intCast(ch * 4)),
        };
    }

    fn forward(self: *const ConvNeXt, sc: *Scope, x: A, nbr: A) !A {
        var h = try self.conv.forward(sc, x, nbr);
        h = try layerNorm(sc, h, self.norm_w, self.norm_b, LN_EPS);
        h = try self.fc2.forward(sc, try sc.silu(try self.fc1.forward(sc, h)));
        return sc.add(h, x);
    }
};

const UpBlock = struct {
    norm_w: A,
    norm_b: A,
    conv1: Conv3,
    conv2: Conv3,
    to_subdiv: ?Linear,
    cin: usize,
    cout: usize,

    fn load(w: *const Weights, stage: usize, idx: usize, cin: usize, cout: usize, pred_subdiv: bool) !UpBlock {
        var up = UpBlock{
            .norm_w = try scope_mod.getW(w, "blocks.{d}.{d}.norm1.weight", .{ stage, idx }),
            .norm_b = try scope_mod.getW(w, "blocks.{d}.{d}.norm1.bias", .{ stage, idx }),
            .conv1 = try Conv3.load(w, "blocks.{d}.{d}.conv1", .{ stage, idx }),
            .conv2 = try Conv3.load(w, "blocks.{d}.{d}.conv2", .{ stage, idx }),
            .to_subdiv = null,
            .cin = cin,
            .cout = cout,
        };
        if (pred_subdiv) up.to_subdiv = try scope_mod.linear(w, "blocks.{d}.{d}.to_subdiv", .{ stage, idx }, @intCast(cin));
        if (up.conv1.co != cout * 8 or up.conv1.ci != cin or up.conv2.ci != cout or up.conv2.co != cout) return error.BadUpBlock;
        return up;
    }

    /// The decisions of `to_subdiv(x)`: `[n*8]` keep flags (logit > 0).
    fn predict(self: *const UpBlock, sc: *Scope, gpa: std.mem.Allocator, x: A) ![]bool {
        const logits = try self.to_subdiv.?.forward(sc, x);
        var gt = mlx.mlx_array_new();
        const flag = try sc.res(mlx.mlx_greater(&gt, logits, try sc.scalarLike(logits, 0), sc.s), &gt);
        const u8s = try sc.astype(flag, .uint8);
        try mlx.check(mlx.mlx_array_eval(u8s));
        const data = mlx.mlx_array_data_uint8(u8s) orelse return error.MlxError;
        const out = try gpa.alloc(bool, mlx.mlx_array_size(u8s));
        for (out, data[0..out.len]) |*o, d| o.* = d != 0;
        return out;
    }

    fn forward(self: *const UpBlock, sc: *Scope, gpa: std.mem.Allocator, x: A, plan: *Plan, i: usize, guided: bool) !A {
        if (!guided) {
            const keep = try self.predict(sc, gpa, x);
            const ch = subdivide(gpa, plan.levels.items[i].coords, keep) catch |e| {
                gpa.free(keep);
                return e;
            };
            try plan.push(keep, ch);
        }
        const gather = try sc.fromI32(plan.gathers.items[i], &.{@intCast(plan.gathers.items[i].len)});
        var h = try sc.silu(try layerNorm(sc, x, self.norm_w, self.norm_b, LN_EPS));
        h = try self.conv1.forward(sc, h, try plan.levels.items[i].neighbors(gpa));
        h = try channelToSpace(sc, h, gather);
        const xs = try channelToSpace(sc, x, gather);
        h = try sc.silu(try layerNorm(sc, h, NO_ARRAY, NO_ARRAY, LN_EPS));
        h = try self.conv2.forward(sc, h, try plan.levels.items[i + 1].neighbors(gpa));
        // repeat_interleave(cout / (cin / 8), dim=1): each channel of the shuffled input, repeated in place.
        const rep: c_int = @intCast(self.cout / (self.cin / 8));
        const m: c_int = @intCast(dim0(xs));
        const cs: c_int = @intCast(self.cin / 8);
        const skip = try sc.reshape(try sc.broadcast(try sc.reshape(xs, &.{ m, cs, 1 }), &.{ m, cs, rep }), &.{ m, cs * rep });
        return sc.add(h, skip);
    }
};

const Stage = struct {
    blocks: []ConvNeXt,
    up: ?UpBlock,
};

/// `SparseUnetVaeDecoder` as `shape_dec_*` (predicts subdivisions, 7 outputs) and
/// `tex_dec_*` (follows them, 6 outputs). Weights are borrowed from `w`.
pub const Decoder = struct {
    gpa: std.mem.Allocator,
    stages: []Stage,
    from_latent: Linear,
    output: Linear,
    pred_subdiv: bool,
    act: mlx.mlx_dtype,
    owned: std.ArrayList(A) = .empty,

    pub const Cfg = struct {
        model_channels: []const usize,
        num_blocks: []const usize,
        pred_subdiv: bool,
    };

    /// `act`: the dtype of the block activations (f16 as published; f32 for the oracle bars).
    /// `from_latent` and `output_layer` always run f32, like the reference's unconverted layers.
    pub fn load(gpa: std.mem.Allocator, w: *const Weights, cfg: Cfg, act: mlx.mlx_dtype) !Decoder {
        const ns = cfg.num_blocks.len;
        if (cfg.model_channels.len != ns) return error.BadDecoderConfig;
        const stages = try gpa.alloc(Stage, ns);
        @memset(stages, .{ .blocks = &.{}, .up = null });
        var d = Decoder{ .gpa = gpa, .stages = stages, .from_latent = undefined, .output = undefined, .pred_subdiv = cfg.pred_subdiv, .act = act };
        errdefer d.deinit();
        d.from_latent = try d.f32Linear(w, "from_latent");
        d.output = try d.f32Linear(w, "output_layer");
        for (0..ns) |i| {
            const nb = cfg.num_blocks[i];
            d.stages[i].blocks = try gpa.alloc(ConvNeXt, nb);
            for (d.stages[i].blocks, 0..) |*b, j| b.* = try ConvNeXt.load(w, i, j, cfg.model_channels[i]);
            if (i + 1 < ns) d.stages[i].up = try UpBlock.load(w, i, nb, cfg.model_channels[i], cfg.model_channels[i + 1], cfg.pred_subdiv);
        }
        return d;
    }

    fn f32Linear(self: *Decoder, w: *const Weights, comptime name: []const u8) !Linear {
        return f32LinearOwned(self.gpa, &self.owned, w, name);
    }

    pub fn deinit(self: *Decoder) void {
        for (self.stages) |st| self.gpa.free(st.blocks);
        self.gpa.free(self.stages);
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.gpa);
    }

    /// Shape decode: `z` `[n, latent]` f32 over `plan.levels[0]`; the plan grows with each
    /// predicted subdivision. Returns `[M, out]` f32 over the last level.
    pub fn decode(self: *const Decoder, sc: *Scope, z: A, plan: *Plan) !A {
        return (try self.run(sc, z, plan, false, true)).?;
    }

    /// Texture decode over a finished plan.
    pub fn decodeGuided(self: *const Decoder, sc: *Scope, z: A, plan: *Plan) !A {
        return (try self.run(sc, z, plan, true, true)).?;
    }

    /// The decode without its head: only the plan grows. `plan.levels` ends at the finest voxel set
    /// (what the cascade quantizes onto the next resolution's grid).
    pub fn upsample(self: *const Decoder, sc: *Scope, z: A, plan: *Plan) !void {
        _ = try self.run(sc, z, plan, false, false);
    }

    /// One block in its own scope: only its output outlives it, so a million-voxel MLP's intermediates are
    /// freed before the next block.
    fn stepScoped(sc: *Scope, h: A, b: *const ConvNeXt, nbr: A) !A {
        var bsc = Scope.init(sc.a, sc.s);
        defer bsc.deinit();
        const y = try b.forward(&bsc, h, nbr);
        try mlx.check(mlx.mlx_array_eval(y));
        return sc.keep(bsc.out(y));
    }

    fn run(self: *const Decoder, sc: *Scope, z: A, plan: *Plan, guided: bool, head: bool) !?A {
        const gpa = plan.gpa;
        var h = try sc.astype(try self.from_latent.forward(sc, z), self.act);
        for (self.stages, 0..) |*st, i| {
            const nbr = try plan.levels.items[i].neighbors(gpa);
            for (st.blocks) |*b| h = try stepScoped(sc, h, b, nbr);
            if (st.up) |*up| {
                var bsc = Scope.init(gpa, sc.s);
                defer bsc.deinit();
                const y = try up.forward(&bsc, gpa, h, plan, i, guided);
                try mlx.check(mlx.mlx_array_eval(y));
                h = try sc.keep(bsc.out(y));
            }
        }
        if (!head) return null;
        h = try layerNorm(sc, try sc.astype(h, .float32), NO_ARRAY, NO_ARRAY, 1e-5);
        return try self.output.forward(sc, h);
    }
};

/// `<name>.weight` / `.bias` cast to f32 (kept alive in `owned`): the layers the reference never converts to f16/bf16.
pub fn f32LinearOwned(gpa: std.mem.Allocator, owned: *std.ArrayList(A), w: *const Weights, comptime name: []const u8) !Linear {
    // A packed weight's codes cast to f32 are numbers that mean nothing: only `blocks.*` projections ship quantized.
    if (w.get(name ++ ".scales") != null) return error.PackedWeight;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var l = Linear{ .w = undefined };
    inline for (.{ "weight", "bias" }) |part| {
        const src = try scope_mod.getW(w, name ++ "." ++ part, .{});
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&o, src, .float32, s));
        errdefer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_array_eval(o));
        try owned.append(gpa, o);
        if (comptime std.mem.eql(u8, part, "weight")) l.w = o else l.bias = o;
    }
    return l;
}

/// A decoder and the weights it borrows from.
pub const Opened = struct {
    w: Weights,
    dec: Decoder,

    pub fn deinit(self: *Opened) void {
        self.dec.deinit();
        self.w.deinit();
    }
};

fn readFile(io: std.Io, gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(gpa, .limited(1 << 20));
}

/// `<dir>/<stem>.json` (block counts and widths) + `<dir>/<stem>.safetensors`, as published in `ckpts/`.
/// Whether the decoder predicts subdivisions is read off the weights (`to_subdiv`).
pub fn openDecoder(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, stem: []const u8, act: mlx.mlx_dtype) !Opened {
    const json_path = try std.fmt.allocPrint(gpa, "{s}/{s}.json", .{ dir, stem });
    defer gpa.free(json_path);
    const bytes = try readFile(io, gpa, json_path);
    defer gpa.free(bytes);
    const Raw = struct { args: struct { model_channels: []usize, num_blocks: []usize } };
    var parsed = try std.json.parseFromSlice(Raw, gpa, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const file = try std.fmt.allocPrint(gpa, "{s}.safetensors", .{stem});
    defer gpa.free(file);
    var w = try model_mod.loadWeightsFile(gpa, dir, file);
    errdefer w.deinit();
    const a = parsed.value.args;
    if (a.num_blocks.len == 0) return error.BadDecoderConfig;
    var key: [64]u8 = undefined;
    const probe = try std.fmt.bufPrint(&key, "blocks.0.{d}.to_subdiv.weight", .{a.num_blocks[0]});
    const dec = try Decoder.load(gpa, &w, .{ .model_channels = a.model_channels, .num_blocks = a.num_blocks, .pred_subdiv = w.get(probe) != null }, act);
    return .{ .w = w, .dec = dec };
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic first, then the env-gated oracle (TRELLIS2_* ; fixtures from
// tests/dump_trellis2_fixtures.py).
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

fn randomVoxels(gpa: std.mem.Allocator, rng: std.Random, grid: i32, count: usize) ![]i32 {
    var seen = std.AutoHashMap(u64, void).init(gpa);
    defer seen.deinit();
    var out: std.ArrayList(i32) = .empty;
    errdefer out.deinit(gpa);
    while (seen.count() < count) {
        const x = rng.intRangeLessThan(i32, 0, grid);
        const y = rng.intRangeLessThan(i32, 0, grid);
        const z = rng.intRangeLessThan(i32, 0, grid);
        const k = voxelKey(x, y, z);
        if (seen.contains(k)) continue;
        try seen.put(k, {});
        try out.appendSlice(gpa, &.{ 0, x, y, z });
    }
    return out.toOwnedSlice(gpa);
}

test "trellis2 sparse: the neighbour map is every active voxel within one step on each axis, in (dx,dy,dz) row-major order" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const coords = try randomVoxels(gpa, prng.random(), 6, 60);
    defer gpa.free(coords);
    const n = coords.len / 4;
    const map = try neighborMap(gpa, coords);
    defer gpa.free(map);
    for (0..n) |i| {
        var k: usize = 0;
        for ([_]i32{ -1, 0, 1 }) |dx| for ([_]i32{ -1, 0, 1 }) |dy| for ([_]i32{ -1, 0, 1 }) |dz| {
            const want: i32 = for (0..n) |j| {
                const c = coords[j * 4 ..][0..4];
                if (c[1] == coords[i * 4 + 1] + dx and c[2] == coords[i * 4 + 2] + dy and c[3] == coords[i * 4 + 3] + dz) break @intCast(j);
            } else @intCast(n);
            try testing.expectEqual(want, map[i * KV + k]);
            k += 1;
        };
    }
}

test "trellis2 sparse: a duplicate voxel is an error, never a silent overwrite" {
    const gpa = testing.allocator;
    const coords = [_]i32{ 0, 1, 2, 3, 0, 1, 2, 3 };
    try testing.expectError(error.DuplicateVoxel, neighborMap(gpa, &coords));
}

test "trellis2 sparse: a large voxel set splits across threads and gives the one-thread map" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(11);
    const coords = try randomVoxels(gpa, prng.random(), 80, 3 * MIN_ROWS_PER_THREAD);
    defer gpa.free(coords);
    const n = coords.len / 4;
    const map = try neighborMap(gpa, coords);
    defer gpa.free(map);
    // Spot-check against the table directly.
    const table = try Table.init(gpa, coords);
    defer table.deinit(gpa);
    for ([_]usize{ 0, 1, n / 2, n - 1 }) |i| {
        const c = coords[i * 4 ..][0..4];
        try testing.expectEqual(@as(i32, @intCast(i)), map[i * KV + 13]); // the centre tap is the voxel itself
        const east = if (c[1] + 1 < 1 << AXIS_BITS) table.find(voxelKey(c[1] + 1, c[2], c[3])) else null;
        try testing.expectEqual(if (east) |e| @as(i32, @intCast(e)) else @as(i32, @intCast(n)), map[i * KV + 22]); // (dx, dy, dz) = (+1, 0, 0)
    }
}

test "trellis2 sparse: subdivide places kept children at 2*parent + bits (x lowest), parents then children" {
    const gpa = testing.allocator;
    const coords = [_]i32{ 0, 1, 2, 3, 0, 4, 5, 6 };
    var keep: [16]bool = @splat(false);
    keep[0] = true; // parent 0, child 0 -> (2,4,6)
    keep[5] = true; // parent 0, child 5 = x,z -> (3,4,7)
    keep[8 + 7] = true; // parent 1, child 7 -> (9,11,13)
    const ch = try subdivide(gpa, &coords, &keep);
    defer ch.deinit(gpa);
    try testing.expectEqualSlices(i32, &.{ 0, 2, 4, 6, 0, 3, 4, 7, 0, 9, 11, 13 }, ch.coords);
    try testing.expectEqualSlices(i32, &.{ 0, 5, 15 }, ch.gather);
}

/// Row-major f32 values of `a` (any layout, any float dtype), valid for `sc`'s lifetime.
fn arrayF32(sc: *Scope, a: A) ![]const f32 {
    var c = mlx.mlx_array_new();
    const flat = try sc.res(mlx.mlx_contiguous(&c, try sc.astype(a, .float32), false, sc.s), &c);
    try mlx.check(mlx.mlx_array_eval(flat));
    return (mlx.mlx_array_data_float32(flat) orelse return error.MlxError)[0..mlx.mlx_array_size(flat)];
}

test "trellis2 sparse: the gather-GEMM convolution is the sum over taps of W_k . x[nbr], missing taps zero" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const rng = prng.random();
    const ci = 5;
    const co = 7;
    const coords = try randomVoxels(gpa, rng, 6, 50);
    defer gpa.free(coords);
    const n = coords.len / 4;
    const x = try gpa.alloc(f32, n * ci);
    defer gpa.free(x);
    const wt = try gpa.alloc(f32, co * KV * ci);
    defer gpa.free(wt);
    var bias: [co]f32 = undefined;
    for (x) |*v| v.* = rng.floatNorm(f32);
    for (wt) |*v| v.* = rng.floatNorm(f32) * 0.3;
    for (&bias) |*v| v.* = rng.floatNorm(f32);
    const map = try neighborMap(gpa, coords);
    defer gpa.free(map);

    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const xa = try sc.fromF32(x, &.{ @intCast(n), ci });
    const wa = try sc.fromF32(wt, &.{ co, 3, 3, 3, ci });
    const ba = try sc.fromF32(&bias, &.{co});
    var vox = Voxels{ .coords = try gpa.dupe(i32, coords) };
    defer vox.deinit(gpa);
    const conv = Conv3{ .w = wa, .b = ba, .ci = ci, .co = co };
    const got = try arrayF32(&sc, try conv.forward(&sc, xa, try vox.neighbors(gpa)));
    try testing.expectEqual(n * co, got.len);
    for (0..n) |i| for (0..co) |o| {
        var want: f64 = bias[o];
        for (0..KV) |k| {
            const j: usize = @intCast(map[i * KV + k]);
            if (j == n) continue;
            for (0..ci) |c| want += @as(f64, wt[(o * KV + k) * ci + c]) * x[j * ci + c];
        }
        // MLX may run f32 GEMMs as TF32 on M5.
        try testing.expectApproxEqAbs(@as(f32, @floatCast(want)), got[i * co + o], 2e-2);
    };
}

test "trellis2 sparse: a convolution split into blocks equals the one-block result" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    const ci = 3;
    const co = 4;
    const coords = try randomVoxels(gpa, rng, 10, 700);
    defer gpa.free(coords);
    const n = coords.len / 4;
    const x = try gpa.alloc(f32, n * ci);
    defer gpa.free(x);
    const wt = try gpa.alloc(f32, co * KV * ci);
    defer gpa.free(wt);
    for (x) |*v| v.* = rng.floatNorm(f32);
    for (wt) |*v| v.* = rng.floatNorm(f32) * 0.3;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const xa = try sc.fromF32(x, &.{ @intCast(n), ci });
    const wa = try sc.fromF32(wt, &.{ co, 3, 3, 3, ci });
    const ba = try sc.zeros(&.{co}, .float32);
    var vox = Voxels{ .coords = try gpa.dupe(i32, coords) };
    defer vox.deinit(gpa);
    const nbr = try vox.neighbors(gpa);
    const conv = Conv3{ .w = wa, .b = ba, .ci = ci, .co = co };
    const want = try gpa.dupe(f32, try arrayF32(&sc, try conv.forwardRows(&sc, xa, nbr, n)));
    defer gpa.free(want);
    const got = try arrayF32(&sc, try conv.forwardRows(&sc, xa, nbr, 256));
    try testing.expectEqualSlices(f32, want, got);
}

test "trellis2 sparse: a null weight and bias is the non-affine LayerNorm" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const x = [_]f32{ 1, 2, 3, 4, -1, 0, 5, 9 };
    const got = try arrayF32(&sc, try layerNorm(&sc, try sc.fromF32(&x, &.{ 2, 4 }), NO_ARRAY, NO_ARRAY, 1e-5));
    for (0..2) |r| {
        var mean: f64 = 0;
        for (x[r * 4 ..][0..4]) |v| mean += v;
        mean /= 4;
        var variance: f64 = 0;
        for (x[r * 4 ..][0..4]) |v| variance += (v - mean) * (v - mean);
        variance /= 4;
        for (0..4) |c| try testing.expectApproxEqAbs(@as(f32, @floatCast((x[r * 4 + c] - mean) / @sqrt(variance + 1e-5))), got[r * 4 + c], 1e-5);
    }
}

const OracleEnv = struct { ckpts: []const u8, fixtures: []const u8 };

fn oracleEnv() !OracleEnv {
    const w = std.c.getenv("TRELLIS2_WEIGHTS") orelse return error.SkipZigTest;
    const f = std.c.getenv("TRELLIS2_FIXTURES") orelse return error.SkipZigTest;
    return .{ .ckpts = std.fmt.allocPrint(testing.allocator, "{s}/ckpts", .{std.mem.span(w)}) catch return error.OutOfMemory, .fixtures = std.mem.span(f) };
}

fn hostI32(gpa: std.mem.Allocator, a: A) ![]i32 {
    try mlx.check(mlx.mlx_array_eval(a));
    const d = mlx.mlx_array_data_int32(a) orelse return error.MlxError;
    return gpa.dupe(i32, d[0..mlx.mlx_array_size(a)]);
}

/// Cosine, rms ratio and the largest error against the peak of `want`; both read in f32.
fn expectClose(label: []const u8, sc: *Scope, got: A, want: A, min_cos: f64, tol_ratio: f64) !void {
    const g = try arrayF32(sc, got);
    const w = try arrayF32(sc, want);
    try testing.expectEqual(w.len, g.len);
    var dot: f64 = 0;
    var gg: f64 = 0;
    var ww: f64 = 0;
    var peak: f64 = 0;
    var worst: f64 = 0;
    for (g, w) |a, b| {
        try testing.expect(std.math.isFinite(a));
        dot += @as(f64, a) * b;
        gg += @as(f64, a) * a;
        ww += @as(f64, b) * b;
        peak = @max(peak, @abs(@as(f64, b)));
        worst = @max(worst, @abs(@as(f64, a) - b));
    }
    const cos = dot / @sqrt(gg * ww);
    const ratio = @sqrt(gg / ww);
    std.debug.print("{s}: cos {d:.7} rms ratio {d:.6} max err/peak {e:.2}\n", .{ label, cos, ratio, worst / peak });
    try testing.expect(cos > min_cos);
    try testing.expect(@abs(ratio - 1) < tol_ratio);
}

test "trellis2 oracle: the shape decoder predicts the reference's subdivisions and its per-voxel outputs (TRELLIS2_WEIGHTS, TRELLIS2_FIXTURES)" {
    const gpa = testing.allocator;
    const env = try oracleEnv();
    defer gpa.free(env.ckpts);
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try model_mod.loadWeightsFile(gpa, env.fixtures, "shape_dec.safetensors");
    defer fx.deinit();
    var dec = try openDecoder(io, gpa, env.ckpts, "shape_dec_next_dc_f16c32_fp16", .float32);
    defer dec.deinit();

    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const coords = try hostI32(gpa, fx.get("coords").?);
    defer gpa.free(coords);
    var plan = try Plan.init(gpa, coords);
    defer plan.deinit();
    const out = try dec.dec.decode(&sc, fx.get("slat").?, &plan);

    try testing.expectEqual(@as(usize, 5), plan.levels.items.len);
    for (0..4) |i| {
        var name: [24]u8 = undefined;
        const lc = try hostI32(gpa, fx.get(try std.fmt.bufPrint(&name, "sub{d}_coords", .{i})).?);
        defer gpa.free(lc);
        try testing.expectEqualSlices(i32, lc, plan.levels.items[i].coords);
        const logits = try arrayF32(&sc, fx.get(try std.fmt.bufPrint(&name, "sub{d}", .{i})).?);
        var flips: usize = 0;
        for (logits, plan.subs.items[i]) |l, k| flips += @intFromBool((l > 0) != k);
        std.debug.print("subdivision {d}: {d} voxels, {d} decisions differ\n", .{ i, logits.len / 8, flips });
        try testing.expectEqual(@as(usize, 0), flips);
    }
    const last = plan.levels.items[4].coords;
    const want_coords = try hostI32(gpa, fx.get("out_coords").?);
    defer gpa.free(want_coords);
    try testing.expectEqualSlices(i32, want_coords, last);
    try expectClose("shape decoder raw", &sc, out, fx.get("raw").?, 0.9999, 1e-3);

    // The texture decoder follows the same plan.
    var tfx = try model_mod.loadWeightsFile(gpa, env.fixtures, "tex_dec.safetensors");
    defer tfx.deinit();
    var tdec = try openDecoder(io, gpa, env.ckpts, "tex_dec_next_dc_f16c32_fp16", .float32);
    defer tdec.deinit();
    const tex = try tdec.dec.decodeGuided(&sc, tfx.get("slat").?, &plan);
    try expectClose("texture decoder raw", &sc, tex, tfx.get("raw").?, 0.9999, 1e-3);
}

test "trellis2 oracle: the decoders in their published f16 stay within half-precision of the reference (TRELLIS2_WEIGHTS, TRELLIS2_FIXTURES)" {
    const gpa = testing.allocator;
    const env = try oracleEnv();
    defer gpa.free(env.ckpts);
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try model_mod.loadWeightsFile(gpa, env.fixtures, "shape_dec.safetensors");
    defer fx.deinit();
    var dec = try openDecoder(io, gpa, env.ckpts, "shape_dec_next_dc_f16c32_fp16", .float16);
    defer dec.deinit();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const coords = try hostI32(gpa, fx.get("coords").?);
    defer gpa.free(coords);
    var plan = try Plan.init(gpa, coords);
    defer plan.deinit();
    const out = try dec.dec.decode(&sc, fx.get("slat").?, &plan);
    // Past the first flipped decision the voxel sets differ and rows no longer line up: compare only while they do.
    var flips_total: usize = 0;
    var decisions: usize = 0;
    for (0..4) |i| {
        var name: [24]u8 = undefined;
        const lc = try hostI32(gpa, fx.get(try std.fmt.bufPrint(&name, "sub{d}_coords", .{i})).?);
        defer gpa.free(lc);
        if (!std.mem.eql(i32, lc, plan.levels.items[i].coords)) break;
        const logits = try arrayF32(&sc, fx.get(try std.fmt.bufPrint(&name, "sub{d}", .{i})).?);
        for (logits, plan.subs.items[i]) |l, k| flips_total += @intFromBool((l > 0) != k);
        decisions += logits.len;
    }
    std.debug.print("f16 decoder: {d} of {d} subdivision decisions differ while the voxel sets agree\n", .{ flips_total, decisions });
    try testing.expect(decisions >= 960 + 6136);
    try testing.expect(flips_total * 200 <= decisions);
    const want_coords = try hostI32(gpa, fx.get("out_coords").?);
    defer gpa.free(want_coords);
    if (std.mem.eql(i32, want_coords, plan.levels.items[4].coords))
        try expectClose("f16 shape decoder raw", &sc, out, fx.get("raw").?, 0.999, 2e-2);
}

test "f32LinearOwned refuses a packed weight instead of reading its codes as numbers" {
    const io = std.testing.io;
    const gpa = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    // lin.weight [2,2], lin.bias [2], and for the packed file also lin.scales [2,1]
    const plain = "{\"lin.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"lin.bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]}}";
    const packed_hdr = "{\"lin.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"lin.bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]},\"lin.scales\":{\"dtype\":\"F32\",\"shape\":[2,1],\"data_offsets\":[24,32]}}";
    inline for (.{ .{ "plain.safetensors", plain, 24 }, .{ "packed.safetensors", packed_hdr, 32 } }) |f| {
        var buf: [8 + f[1].len + f[2]]u8 = undefined;
        std.mem.writeInt(u64, buf[0..8], f[1].len, .little);
        @memcpy(buf[8..][0..f[1].len], f[1]);
        @memset(buf[8 + f[1].len ..], 0);
        try tmp.dir.writeFile(io, .{ .sub_path = f[0], .data = &buf });
    }
    var owned: std.ArrayList(A) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(gpa);
    }
    var ok = try model_mod.loadWeightsFile(gpa, dir, "plain.safetensors");
    defer ok.deinit();
    _ = try f32LinearOwned(gpa, &owned, &ok, "lin");
    var bad = try model_mod.loadWeightsFile(gpa, dir, "packed.safetensors");
    defer bad.deinit();
    try std.testing.expectError(error.PackedWeight, f32LinearOwned(gpa, &owned, &bad, "lin"));
}
