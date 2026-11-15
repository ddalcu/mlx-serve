//! Directional ablation (Arditi et al. 2024) in the built-in DeepSeek-V4.1 module: a request's direction read from
//! `abi.Forward.ablation` and armed on its slot's state (`stateArm`), the ops that take it out of every write to the
//! residual stream (`deepseek_v41.zig` calls them at each write, under the ablation of the state its pass runs), and the
//! calibration tap that records the stream entering each layer of a prompt forward (`StreamTap`, `dumpTap`).
//!
//! The ablation lives on the state, not on the call: the engine runs a slot's DSpark rounds on the module between its
//! forwards, and they ablate too. A state's `variant` names the ablation its rows were computed under, so no forward
//! continues rows that another ablation (or none) wrote.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const dsv41 = @import("deepseek_v41.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const Scope = dsv41.Scope;

pub const Refusal = error{ AblationStrength, AblationSlots };

/// Whether this tree's ABI has `Forward.ablation`; without it the module serves none.
pub const supported = @hasField(abi.Forward, "ablation");

/// One slot's ablation as the forward hands it: the directions table `[slots, hidden]` f32, the slot's row of it, its
/// strength and the host's key (equal keys ablate identically; 0 = none given).
pub const Spec = struct { directions: abi.Array, row: usize = 0, strength: f32, key: u64 = 0 };

/// Slot `k`'s ablation, or null: a host whose `Forward` ends before `ablation`, a forward without one, or a strength of
/// 0 (that slot runs plain). A table without one strength per slot, or with a strength outside [0, 1], refuses whichever
/// slot asks, so a batch refuses whole.
pub fn ofSlot(f: *const abi.Forward, k: usize) Refusal!?Spec {
    if (comptime !supported) return null else {
        if (f.size < @offsetOf(abi.Forward, "ablation") + @sizeOf(?*const abi.Ablation)) return null;
        const a = f.ablation orelse return null;
        const strengths = a.strengths.slice();
        if (strengths.len == 0 or strengths.len != f.slots.len or k >= strengths.len) return error.AblationSlots;
        for (strengths) |v| if (!(v >= 0 and v <= 1)) return error.AblationStrength;
        const strength = strengths[k];
        if (strength == 0) return null;
        const keyed = a.size >= @offsetOf(abi.Ablation, "keys") + @sizeOf(abi.Span(u64)) and a.keys.len == strengths.len;
        return .{ .directions = a.directions, .row = k, .strength = strength, .key = if (keyed) a.keys.slice()[k] else 0 };
    }
}

/// `ofSlot` of a forward's one slot.
pub fn of(f: *const abi.Forward) Refusal!?Spec {
    return ofSlot(f, 0);
}

/// Whether `a`'s table is `[slots, dim]` f32 and holds the slot's row.
pub fn shapeOk(a: Spec, slots: usize, dim: usize) bool {
    const d: A = .{ .ctx = a.directions.ctx };
    if (d.ctx == null) return false;
    const sh = mlx.getShape(d);
    return sh.len == 2 and sh[0] == slots and sh[1] == dim and a.row < slots and mlx.mlx_array_dtype(d) == .float32;
}

/// An armed ablation: the unit direction `r` `[dim]`, the same as a column `col` `[dim, 1]` and the strength `s`, all
/// f32, with the key of the table it came from.
pub const Ablation = struct {
    r: A,
    col: A,
    s: A,
    key: u64 = 0,

    pub fn deinit(self: Ablation) void {
        for ([_]A{ self.r, self.col, self.s }) |x| _ = mlx.mlx_array_free(x);
    }
};

/// `a` armed: its row of the table as `r` and as `col`, and its strength, new arrays the caller owns. Nothing is left
/// allocated when it fails.
pub fn armed(a: Spec, dim: usize, s: S) !Ablation {
    const d: A = .{ .ctx = a.directions.ctx };
    if (d.ctx == null) return error.AblationShape;
    const sh = mlx.getShape(d);
    if (sh.len != 2 or a.row >= sh[0] or sh[1] != dim or mlx.mlx_array_dtype(d) != .float32) return error.AblationShape;
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    const lo = [_]c_int{ @intCast(a.row), 0 };
    const hi = [_]c_int{ @intCast(a.row + 1), @intCast(dim) };
    const step = [_]c_int{ 1, 1 };
    try mlx.check(mlx.mlx_slice(&row, d, &lo, 2, &hi, 2, &step, 2, s));
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, row, &[_]c_int{@intCast(dim)}, 1, s));
    var col = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, row, &[_]c_int{ @intCast(dim), 1 }, 2, s));
    return .{ .r = r, .col = col, .s = mlx.mlx_array_new_float(a.strength), .key = a.key };
}

/// New handles on `a`'s arrays: a snapshot's own, or the ablation of a state restored from one.
pub fn retain(a: ?Ablation) ?Ablation {
    const x = a orelse return null;
    return .{ .r = ref(x.r), .col = ref(x.col), .s = ref(x.s), .key = x.key };
}

fn ref(x: A) A {
    var o = mlx.mlx_array_new();
    _ = mlx.mlx_array_set(&o, x);
    return o;
}

/// Whether passes under `a` and under `b` ablate alike: both plain, or one key the host gave both (an unkeyed ablation
/// matches nothing, itself included).
pub fn same(a: ?Ablation, b: ?Ablation) bool {
    const x = a orelse return b == null;
    const y = b orelse return false;
    return x.key != 0 and x.key == y.key;
}

/// The variant of an ablated slot whose host gave no key: it matches only itself.
pub const unkeyed: u64 = std.math.maxInt(u64);

/// The variant a forward under `a` computes its state's rows in (`State.variant`): the host's key, `unkeyed` without
/// one, 0 for none.
pub fn variantOf(a: ?Spec) u64 {
    const x = a orelse return 0;
    return if (x.key != 0) x.key else unkeyed;
}

/// Whether a forward of variant `key` may run on a state of `n` rows computed under `variant`: a fresh state takes any,
/// a state read back from the disk tier (null) takes the host's word (it keys those entries by variant), any other only
/// its own.
pub fn fits(n: usize, variant: ?u64, key: u64) bool {
    if (n == 0) return true;
    const v = variant orelse return true;
    return v == key;
}

/// Arms slot ablation `a` (null: none) on `st` for every pass over it until the next `stateArm` or `stateDisarm`, the
/// DSpark rounds the engine runs between forwards included; `State.deinit` frees it. A state already under the same
/// keyed ablation keeps its arrays; a fresh state, or one read back from the disk tier, takes the forward's variant
/// (the caller checks `fits` first). Nothing changes when arming fails.
pub fn stateArm(st: *dsv41.State, a: ?Spec, dim: usize, s: S) !void {
    if (a) |spec| {
        const keep = if (st.ablation) |cur| cur.key != 0 and cur.key == spec.key else false;
        if (!keep) {
            const fresh = try armed(spec, dim, s);
            stateDisarm(st);
            st.ablation = fresh;
        }
    } else stateDisarm(st);
    if (st.n == 0 or st.variant == null) st.variant = variantOf(a);
}

/// Drops `st`'s ablation; graphs built under it keep their own references.
pub fn stateDisarm(st: *dsv41.State) void {
    const ab = st.ablation orelse return;
    ab.deinit();
    st.ablation = null;
}

/// `y - s (y . r) r` over the last axis, in f32, back in `y`'s dtype, under the ablation of the pass being built
/// (`Dsv41Model.ablation`); `y` itself under none.
pub fn ablate(sc: *Scope, m: *const dsv41.Dsv41Model, y: A) !A {
    return if (m.ablation) |ab| ablateWith(sc, ab, y) else y;
}

pub fn ablateWith(sc: *Scope, ab: Ablation, y: A) !A {
    const yf = try sc.astype(y, .float32);
    const proj = try sc.mul(try sc.matmul(yf, ab.col), ab.s);
    return sc.astype(try sc.sub(yf, try sc.mul(proj, ab.r)), mlx.mlx_array_dtype(y));
}

/// `ablate` of an FFN output an expand already wrote into the stream (`x2 = post y + comb^T x1`, `[L, hc, dim]`, its
/// mix `comb` `[L, hc, hc]`): copy k's component of the output along `r` is `x2_k . r - sum_j comb[j, k] (x1_j . r)`,
/// so the stream ends as the expand of the ablated output would.
pub fn ablateExpanded(sc: *Scope, m: *const dsv41.Dsv41Model, x1: A, x2: A, comb: A) !A {
    return if (m.ablation) |ab| ablateExpandedWith(sc, ab, x1, x2, comb) else x2;
}

pub fn ablateExpandedWith(sc: *Scope, ab: Ablation, x1: A, x2: A, comb: A) !A {
    const x2f = try sc.astype(x2, .float32);
    const d1 = try sc.matmul(try sc.astype(try sc.reshape(x1, mlx.getShape(x2)), .float32), ab.col); // [L, hc, 1]
    const mixed = try sc.matmul(try sc.transpose(try sc.astype(comb, .float32), &.{ 0, 2, 1 }), d1);
    const share = try sc.mul(try sc.sub(try sc.matmul(x2f, ab.col), mixed), ab.s);
    return sc.astype(try sc.sub(x2f, try sc.mul(share, ab.r)), mlx.mlx_array_dtype(x2));
}

/// `ablate` of the value slice of the Engram `wkv` rows `[L, (hc + 1) * dim]`: the per-copy keys only feed the gate.
pub fn ablateValue(sc: *Scope, m: *const dsv41.Dsv41Model, kv: A) !A {
    return if (m.ablation) |ab| ablateValueWith(sc, ab, kv, @as(usize, m.cfg.dsv4_hc_mult) * m.cfg.hidden_size) else kv;
}

/// `ablateWith` of the columns of `kv` past `split` (the value), the rest as they are.
pub fn ablateValueWith(sc: *Scope, ab: Ablation, kv: A, split: usize) !A {
    const sh = mlx.getShape(kv);
    const n: usize = @intCast(sh[sh.len - 1]);
    const value = try ablateWith(sc, ab, try sc.slice(kv, -1, split, n));
    return sc.concat(&.{ try sc.slice(kv, -1, 0, split), value }, -1);
}

/// Calibration: per layer, the mean of the hyper-connection copies of the stream entering it (and, last, leaving the
/// final layer), for a pass's last `keep` rows, f32 `[rows, dim]` apiece. The owner clears and frees it.
pub const StreamTap = struct {
    gpa: std.mem.Allocator,
    keep: usize,
    /// `layers + 1` lists, one entry per pass that ran the layer.
    sites: []std.ArrayList(A),
    /// Passes record only while on (the owner's prompt forwards, not the decode that follows).
    on: bool = false,

    pub fn init(gpa: std.mem.Allocator, layers: usize, keep: usize) !StreamTap {
        const sites = try gpa.alloc(std.ArrayList(A), layers + 1);
        for (sites) |*s| s.* = .empty;
        return .{ .gpa = gpa, .keep = keep, .sites = sites };
    }

    /// Records `site` from a pass's streams `xs` (its spans in row order, `[rows, hc, dim]` each): the last `keep`
    /// rows, handed to the GPU here, so the record holds its own rows rather than the pass's whole streams.
    pub fn record(self: *StreamTap, s: S, site: usize, xs: []const A) !void {
        if (!self.on or xs.len == 0) return;
        for (xs) |part| if (part.ctx == null) return;
        var sc = Scope.init(self.gpa, s);
        defer sc.deinit();
        // The last spans that hold the last `keep` rows.
        var first = xs.len;
        var rows: usize = 0;
        while (first > 0 and rows < self.keep) {
            first -= 1;
            rows += @as(usize, @intCast(mlx.getShape(xs[first])[0]));
        }
        const x = try sc.concat(xs[first..], 0);
        const L: usize = @intCast(mlx.getShape(x)[0]);
        const last = try sc.slice(x, 0, L - @min(L, self.keep), L);
        const mean = sc.out(try sc.mean(try sc.astype(last, .float32), 1, false));
        self.sites[site].append(self.gpa, mean) catch |e| {
            _ = mlx.mlx_array_free(mean);
            return e;
        };
        const v = mlx.mlx_vector_array_new_data(&[_]A{mean}, 1);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_async_eval(v));
    }

    pub fn clear(self: *StreamTap) void {
        for (self.sites) |*s| {
            for (s.items) |a| _ = mlx.mlx_array_free(a);
            s.clearRetainingCapacity();
        }
    }

    pub fn deinit(self: *StreamTap) void {
        self.clear();
        for (self.sites) |*s| s.deinit(self.gpa);
        self.gpa.free(self.sites);
    }
};

/// The tap's rows for one prompt forward as `<dir>/tap-<seq>.npy` (f32 `[layers + 1, rows, dim]`: per site the last
/// `keep` rows its passes recorded, in prompt order, the last one the prompt's last token) and a line in
/// `<dir>/index.jsonl` (`pos` the forward's first position, `ids` its tokens). The dir is the owner's only (0700), its
/// files 0600. Clears the tap; a site no pass recorded (a layer another Mac ran) writes nothing.
pub fn dumpTap(gpa: std.mem.Allocator, tap: *StreamTap, dir: []const u8, seq: u64, pos: usize, ids: []const u32, s: S) !void {
    defer tap.clear();
    const sites = tap.sites.len;
    if (sites == 0 or tap.sites[0].items.len == 0) return;
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const per_site = try gpa.alloc(A, sites);
    defer gpa.free(per_site);
    // Every site keeps the same last rows: the fewest any site recorded, at most `keep`.
    var rows = tap.keep;
    for (tap.sites, per_site) |list, *o| {
        if (list.items.len == 0) return error.TapIncomplete;
        o.* = try sc.concat(list.items, 0); // [rows, dim]
        rows = @min(rows, @as(usize, @intCast(mlx.getShape(o.*)[0])));
    }
    for (per_site) |*o| {
        const n: usize = @intCast(mlx.getShape(o.*)[0]);
        o.* = try sc.slice(o.*, 0, n - rows, n);
    }
    const all = try sc.stack(per_site, 0); // [sites, rows, dim]
    try mlx.check(mlx.mlx_array_eval(all));
    const shape = mlx.getShape(all);
    const data = (mlx.mlx_array_data_float32(all) orelse return error.TapNotF32)[0..mlx.mlx_array_size(all)];

    const io = std.Io.Threaded.global_single_threaded.io();
    var d = try std.Io.Dir.cwd().createDirPathOpen(io, dir, .{});
    defer d.close(io);
    // Prompt activations and ids: the owner's only.
    const dir_z = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(dir_z);
    _ = std.c.chmod(dir_z, 0o700);
    var name_buf: [64]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "tap-{d:0>6}.npy", .{seq});
    var header_buf: [256]u8 = undefined;
    const npy = try npyHeader(&header_buf, @intCast(shape[0]), @intCast(shape[1]), @intCast(shape[2]));
    const body = std.mem.sliceAsBytes(data);
    const file_bytes = try gpa.alloc(u8, npy.len + body.len);
    defer gpa.free(file_bytes);
    @memcpy(file_bytes[0..npy.len], npy);
    @memcpy(file_bytes[npy.len..], body);
    try d.writeFile(io, .{ .sub_path = name, .data = file_bytes, .flags = .{ .permissions = .fromMode(0o600) } });

    var line: std.Io.Writer.Allocating = .init(gpa);
    defer line.deinit();
    try line.writer.print("{{\"seq\":{d},\"file\":\"{s}\",\"position\":{d},\"prompt_tokens\":{d},\"rows\":{d},\"tail_ids\":[", .{ seq, name, pos, pos + ids.len, shape[1] });
    const tail = ids[ids.len - @min(ids.len, 8) ..];
    for (tail, 0..) |t, i| try line.writer.print("{s}{d}", .{ if (i == 0) "" else ",", t });
    try line.writer.writeAll("]}\n");
    var idx = d.openFile(io, "index.jsonl", .{ .mode = .write_only }) catch |e| switch (e) {
        error.FileNotFound => try d.createFile(io, "index.jsonl", .{ .permissions = .fromMode(0o600) }),
        else => return e,
    };
    defer idx.close(io);
    const end = (try idx.stat(io)).size;
    try idx.writePositionalAll(io, line.written(), end);
}

/// A version 1.0 `.npy` header for little-endian f32 `[a, b, c]`, padded to a multiple of 64 bytes.
pub fn npyHeader(buf: []u8, a: usize, b: usize, c: usize) ![]const u8 {
    const dict = try std.fmt.bufPrint(buf[10..], "{{'descr': '<f4', 'fortran_order': False, 'shape': ({d}, {d}, {d}), }}", .{ a, b, c });
    var total = 10 + dict.len + 1;
    const pad = (64 - total % 64) % 64;
    if (total + pad > buf.len) return error.NoSpaceLeft;
    @memset(buf[10 + dict.len ..][0..pad], ' ');
    total += pad;
    buf[total - 1] = '\n';
    @memcpy(buf[0..6], "\x93NUMPY");
    buf[6] = 1;
    buf[7] = 0;
    std.mem.writeInt(u16, buf[8..10], @intCast(total - 10), .little);
    return buf[0..total];
}
