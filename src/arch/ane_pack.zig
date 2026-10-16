//! An MLX pack's tensors read on the host (mmapped safetensors, no MLX) and converted into the
//! Neural Engine's weight forms.
//!
//! The ANE compiler takes per-output-row weight scales only (a block along the input axis, MLX's
//! group of 64, is rejected), and INT8, INT4 and LUT weights all stream at the same rate on the M5
//! (~150 G weights/s); FP16 at half. So every linear goes to INT8 with one FP16 scale per row: from
//! a 4-bit affine pack that is exact up to the INT8 step of the row's largest weight.

const std = @import("std");
const builtin = @import("builtin");

pub const DType = enum { u32, bf16, f16, f32, u8, i8, other };

pub const Tensor = struct {
    dtype: DType,
    rank: u8,
    shape: [4]u64,
    data: []const u8,

    pub fn dim(self: Tensor, i: usize) u64 {
        return self.shape[i];
    }

    pub fn count(self: Tensor) u64 {
        var c: u64 = 1;
        for (self.shape[0..self.rank]) |d| c *= d;
        return c;
    }

    /// Element i as f32 (bf16 / f16 / f32 tensors).
    pub fn at(self: Tensor, i: usize) f32 {
        return switch (self.dtype) {
            .bf16 => bf16(std.mem.readInt(u16, self.data[i * 2 ..][0..2], .little)),
            .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, self.data[i * 2 ..][0..2], .little)))),
            .f32 => @bitCast(std.mem.readInt(u32, self.data[i * 4 ..][0..4], .little)),
            else => 0,
        };
    }
};

pub fn bf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

const Map = struct { bytes: []align(std.heap.page_size_min) const u8 };

pub const Pack = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    maps: std.ArrayList(Map) = .empty,
    tensors: std.StringHashMapUnmanaged(Tensor) = .empty,
    /// Bytes of every shard (the pack's size on disk).
    bytes: u64 = 0,

    /// Every `*.safetensors` file in `dir`, mapped read-only.
    pub fn open(io: std.Io, allocator: std.mem.Allocator, dir: []const u8) !*Pack {
        const self = try allocator.create(Pack);
        errdefer allocator.destroy(self);
        self.* = .{ .allocator = allocator, .arena = .init(allocator) };
        errdefer self.deinit();
        var d = try std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true });
        defer d.close(io);
        var it = d.iterate();
        var names: std.ArrayList([]const u8) = .empty;
        const a = self.arena.allocator();
        while (try it.next(io)) |e| {
            if (e.kind != .file and e.kind != .sym_link) continue;
            if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
            try names.append(a, try a.dupe(u8, e.name));
        }
        if (names.items.len == 0) return error.NoSafetensors;
        for (names.items) |n| {
            const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ dir, n }, 0);
            const map = mapFile(path) orelse return error.MapFailed;
            try self.maps.append(allocator, .{ .bytes = map });
            self.bytes += map.len;
            try self.index(map);
        }
        return self;
    }

    pub fn deinit(self: *Pack) void {
        if (comptime builtin.os.tag != .windows) for (self.maps.items) |m| std.posix.munmap(m.bytes);
        self.maps.deinit(self.allocator);
        self.tensors.deinit(self.allocator);
        self.arena.deinit();
    }

    pub fn close(self: *Pack) void {
        const a = self.allocator;
        self.deinit();
        a.destroy(self);
    }

    fn index(self: *Pack, map: []const u8) !void {
        if (map.len < 8) return error.BadSafetensors;
        const hlen: usize = @intCast(std.mem.readInt(u64, map[0..8], .little));
        if (8 + hlen > map.len) return error.BadSafetensors;
        const a = self.arena.allocator();
        const root = std.json.parseFromSliceLeaky(std.json.Value, a, map[8 .. 8 + hlen], .{}) catch return error.BadSafetensors;
        if (root != .object) return error.BadSafetensors;
        var it = root.object.iterator();
        const base = 8 + hlen;
        while (it.next()) |e| {
            if (std.mem.eql(u8, e.key_ptr.*, "__metadata__")) continue;
            const o = if (e.value_ptr.* == .object) e.value_ptr.object else return error.BadSafetensors;
            const dt = o.get("dtype") orelse return error.BadSafetensors;
            const sh = o.get("shape") orelse return error.BadSafetensors;
            const off = o.get("data_offsets") orelse return error.BadSafetensors;
            if (dt != .string or sh != .array or off != .array or off.array.items.len != 2) return error.BadSafetensors;
            if (sh.array.items.len > 4) continue; // vision convs: never read here
            var t: Tensor = .{ .dtype = parseDType(dt.string), .rank = @intCast(sh.array.items.len), .shape = @splat(1), .data = &.{} };
            for (sh.array.items, 0..) |v, i| t.shape[i] = if (v == .integer and v.integer >= 0) @intCast(v.integer) else return error.BadSafetensors;
            const lo: usize = @intCast(off.array.items[0].integer);
            const hi: usize = @intCast(off.array.items[1].integer);
            if (base + hi > map.len or lo > hi) return error.BadSafetensors;
            t.data = map[base + lo .. base + hi];
            try self.tensors.put(self.allocator, e.key_ptr.*, t);
        }
    }

    pub fn get(self: *const Pack, name: []const u8) ?Tensor {
        return self.tensors.get(name);
    }

    /// `prefix` + `name`, or null.
    pub fn getf(self: *const Pack, buf: []u8, comptime fmt: []const u8, args: anytype) ?Tensor {
        const n = std.fmt.bufPrint(buf, fmt, args) catch return null;
        return self.get(n);
    }

    /// The tensor-name prefix of the text model's layers ("language_model.model", "model", ...).
    pub fn layerPrefix(self: *const Pack) ?[]const u8 {
        for ([_][]const u8{ "language_model.model", "model.language_model", "model" }) |p| {
            var buf: [128]u8 = undefined;
            const n = std.fmt.bufPrint(&buf, "{s}.layers.0.input_layernorm.weight", .{p}) catch continue;
            if (self.get(n) != null) return p;
        }
        return null;
    }
};

fn parseDType(s: []const u8) DType {
    const pairs = .{ .{ "U32", DType.u32 }, .{ "BF16", DType.bf16 }, .{ "F16", DType.f16 }, .{ "F32", DType.f32 }, .{ "U8", DType.u8 }, .{ "I8", DType.i8 } };
    inline for (pairs) |p| if (std.mem.eql(u8, s, p[0])) return p[1];
    return .other;
}

pub fn mapFile(path: [*:0]const u8) ?[]align(std.heap.page_size_min) const u8 {
    if (comptime builtin.os.tag == .windows) return null; // the Neural Engine is Apple-only
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const size: usize = @intCast(@max(std.c.lseek(fd, 0, std.c.SEEK.END), 0));
    if (size == 0) return null;
    return std.posix.mmap(null, size, .{ .READ = true }, .{ .TYPE = .SHARED }, fd, 0) catch null;
}

/// A linear weight of the pack, `out` x `in`: dense (bf16 / f16 / f32) or MLX affine (packed
/// `bits`-wide codes, element i of a row at bit i*bits, a scale and bias per `group` inputs).
pub const Linear = struct {
    out: u32,
    in: u32,
    w: Tensor,
    scales: ?Tensor = null,
    biases: ?Tensor = null,
    bits: u32 = 0,
    group: u32 = 0,
    /// A view (`view`): the source's input width, the first input column, and the source rows as
    /// (start, count) runs (none: every row in order).
    src_in: u32 = 0,
    col0: u32 = 0,
    runs: [3][2]u32 = undefined,
    nruns: u8 = 0,

    /// `<base>.weight` (+ `.scales` / `.biases`) for a weight of `out` x `in`; the bit width and group
    /// come from the tensors' geometry.
    pub fn find(pack: *const Pack, base: []const u8, out: u32, in: u32) !Linear {
        var buf: [256]u8 = undefined;
        const w = pack.getf(&buf, "{s}.weight", .{base}) orelse return error.MissingWeight;
        const s = pack.getf(&buf, "{s}.scales", .{base});
        if (s == null) {
            if (w.rank != 2 or w.dim(0) != out or w.dim(1) != in) return error.BadWeightShape;
            if (w.dtype != .bf16 and w.dtype != .f16 and w.dtype != .f32) return error.BadWeightShape;
            return .{ .out = out, .in = in, .w = w };
        }
        const scales = s.?;
        const biases = pack.getf(&buf, "{s}.biases", .{base}) orelse return error.UnsupportedQuant;
        if (w.dtype != .u32 or w.rank != 2 or w.dim(0) != out or scales.dim(0) != out) return error.BadWeightShape;
        const words = w.dim(1);
        if ((words * 32) % in != 0 or in % scales.dim(1) != 0) return error.BadWeightShape;
        const bits: u32 = @intCast(words * 32 / in);
        if (bits == 0 or bits > 8 or bits == 7) return error.UnsupportedQuant;
        return .{ .out = out, .in = in, .w = w, .scales = scales, .biases = biases, .bits = bits, .group = @intCast(in / scales.dim(1)) };
    }

    /// Rows `runs` ((source start, count), in order; none: every row) over input columns
    /// [c0, c0 + cn): one tensor-parallel lane's share. An affine window keeps whole groups.
    pub fn view(self: Linear, runs: []const [2]u32, c0: u32, cn: u32) !Linear {
        if (c0 + cn > self.in or runs.len > 3) return error.BadWeightShape;
        if (self.group > 0 and (c0 % self.group != 0 or cn % self.group != 0)) return error.BadWeightShape;
        var v = self;
        v.src_in = self.srcIn();
        v.col0 = self.col0 + c0;
        v.in = cn;
        if (runs.len > 0) {
            if (self.nruns > 0) return error.BadWeightShape;
            v.out = 0;
            for (runs, 0..) |run, i| {
                if (run[0] + run[1] > self.out) return error.BadWeightShape;
                v.runs[i] = run;
                v.out += run[1];
            }
            v.nruns = @intCast(runs.len);
        }
        return v;
    }

    fn srcIn(self: Linear) u32 {
        return if (self.src_in > 0) self.src_in else self.in;
    }

    fn srcRow(self: Linear, r: usize) usize {
        var at = r;
        for (self.runs[0..self.nruns]) |run| {
            if (at < run[1]) return run[0] + at;
            at -= run[1];
        }
        return r;
    }

    /// Row r dequantized into `dst` (len `in`).
    pub fn row(self: Linear, r: usize, dst: []f32) void {
        std.debug.assert(dst.len == self.in);
        const sr = self.srcRow(r);
        const width = self.srcIn();
        if (self.scales == null) {
            const base = sr * width + self.col0;
            for (dst, 0..) |*d, k| d.* = self.w.at(base + k);
            return;
        }
        const s = self.scales.?;
        const b = self.biases.?;
        const groups = width / self.group;
        const words: [*]align(1) const u32 = @ptrCast(self.w.data.ptr);
        const wrow = words + sr * (@as(usize, width) * self.bits / 32);
        const mask: u32 = (@as(u32, 1) << @intCast(self.bits)) - 1;
        for (self.col0 / self.group..(self.col0 + self.in) / self.group) |g| {
            const sc = s.at(sr * groups + g);
            const bi = b.at(sr * groups + g);
            const k0 = g * self.group;
            for (k0..k0 + self.group) |k| {
                const bit = k * self.bits;
                const word = bit / 32;
                const sh: u5 = @intCast(bit % 32);
                var q = wrow[word] >> sh;
                if (@as(u32, sh) + self.bits > 32) q |= wrow[word + 1] << @intCast(32 - @as(u32, sh));
                dst[k - self.col0] = sc * @as(f32, @floatFromInt(q & mask)) + bi;
            }
        }
    }

    /// A 1-D or 2-D dense tensor, as f32 (norms, small projections, conv kernels).
    pub fn values(t: Tensor, out: []f32) void {
        for (out, 0..) |*o, i| o.* = t.at(i);
    }
};

/// Rows [r0, r1) of `lin` as INT8 codes with one FP16 scale per row. The codes are rounded against the
/// FP16 scale the ANE will apply, half away from zero.
pub fn int8Rows(lin: Linear, r0: usize, r1: usize, codes: []i8, scales: []align(1) f16, scratch: []f32) void {
    for (r0..r1) |r| {
        lin.row(r, scratch);
        var amax: f32 = 0;
        for (scratch) |v| amax = @max(amax, @abs(v));
        // Floored in f16 itself: a zero row's scale must not round to 0 (and its codes to 0 * inf).
        const s16: f16 = @max(@as(f16, @floatCast(amax / 127.0)), std.math.floatTrueMin(f16));
        scales[r] = s16;
        const inv = 1.0 / @as(f32, s16);
        const dst = codes[r * lin.in ..][0..lin.in];
        for (scratch, dst) |v, *c| c.* = @intFromFloat(std.math.clamp(@round(v * inv), -127, 127));
    }
}

/// Rows [r0, r1) of `lin` as FP16.
pub fn f16Rows(lin: Linear, r0: usize, r1: usize, out: []align(1) f16, scratch: []f32) void {
    for (r0..r1) |r| {
        lin.row(r, scratch);
        for (scratch, out[r * lin.in ..][0..lin.in]) |v, *d| d.* = @floatCast(v);
    }
}

/// A conversion job: one weight into its blob entries.
pub const Job = struct {
    lin: Linear,
    codes: []i8 = &.{},
    scales: []align(1) f16 = &.{},
    /// FP16 output instead of INT8 (codes / scales unused).
    dense: []align(1) f16 = &.{},
};

/// Run every job, split into chunks of 16 rows over `threads` workers.
pub fn convert(allocator: std.mem.Allocator, jobs: []const Job, threads: usize) !void {
    const step = 16;
    var total: usize = 0;
    var widest: usize = 0;
    for (jobs) |j| {
        total += (j.lin.out + step - 1) / step;
        widest = @max(widest, j.lin.in);
    }
    if (total == 0) return;
    const n = @max(1, @min(threads, total));
    const Worker = struct {
        fn run(js: []const Job, next: *std.atomic.Value(usize), scratch: []f32) void {
            while (true) {
                // The next chunk; a chunk never spans two jobs.
                const at = next.fetchAdd(1, .monotonic);
                var base: usize = 0;
                const j, const r0 = for (js) |j| {
                    const chunks = (j.lin.out + step - 1) / step;
                    if (at < base + chunks) break .{ j, (at - base) * step };
                    base += chunks;
                } else return;
                const r1 = @min(r0 + step, j.lin.out);
                const s = scratch[0..j.lin.in];
                if (j.dense.len > 0) f16Rows(j.lin, r0, r1, j.dense, s) else int8Rows(j.lin, r0, r1, j.codes, j.scales, s);
            }
        }
    };
    var next = std.atomic.Value(usize).init(0);
    const scratch = try allocator.alloc(f32, widest * n);
    defer allocator.free(scratch);
    const ts = try allocator.alloc(std.Thread, n - 1);
    defer allocator.free(ts);
    var spawned: usize = 0;
    defer for (ts[0..spawned]) |t| t.join();
    for (ts, 1..) |*t, i| {
        t.* = std.Thread.spawn(.{}, Worker.run, .{ jobs, &next, scratch[i * widest ..][0..widest] }) catch break;
        spawned += 1;
    }
    Worker.run(jobs, &next, scratch[0..widest]);
}

// ── tests ──

const testing = std.testing;

fn testAffine(comptime bits: u32, out: u32, in: u32, group: u32, codes: []const u32, words: []u32, scales: []u16, biases: []u16) Linear {
    @memset(words, 0);
    for (0..out) |r| for (0..in) |k| {
        const q = codes[r * in + k];
        const bit = (r * in + k) * bits; // rows are whole words for these shapes
        words[bit / 32] |= q << @intCast(bit % 32);
        if (bit % 32 + bits > 32) words[bit / 32 + 1] |= q >> @intCast(32 - bit % 32);
    };
    const groups = in / group;
    for (0..out * groups) |i| {
        scales[i] = @truncate(@as(u32, @bitCast(@as(f32, 0.5) * @as(f32, @floatFromInt(i + 1)))) >> 16);
        biases[i] = @truncate(@as(u32, @bitCast(@as(f32, -1.0))) >> 16);
    }
    const t: Tensor = .{ .dtype = .u32, .rank = 2, .shape = .{ out, in * bits / 32, 1, 1 }, .data = std.mem.sliceAsBytes(words) };
    const s: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ out, groups, 1, 1 }, .data = std.mem.sliceAsBytes(scales) };
    const b: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ out, groups, 1, 1 }, .data = std.mem.sliceAsBytes(biases) };
    return .{ .out = out, .in = in, .w = t, .scales = s, .biases = b, .bits = bits, .group = group };
}

test "affine rows: every element unpacks at bit i*bits, straddling words at 3 / 5 / 6 bits" {
    inline for (.{ 3, 4, 5, 6, 8 }) |bits| {
        const out = 2;
        const in = 64;
        var codes: [out * in]u32 = undefined;
        for (&codes, 0..) |*c, i| c.* = @intCast((i * 7 + 3) % (1 << bits));
        var words: [out * in * bits / 32]u32 = undefined;
        var scales: [out * 2]u16 = undefined;
        var biases: [out * 2]u16 = undefined;
        const lin = testAffine(bits, out, in, 32, &codes, &words, &scales, &biases);
        var row: [in]f32 = undefined;
        lin.row(1, &row);
        for (row, 0..) |v, k| {
            const g = 2 + k / 32; // row 1's groups are 2 and 3
            const want = 0.5 * @as(f32, @floatFromInt(g + 1)) * @as(f32, @floatFromInt(codes[in + k])) - 1.0;
            try testing.expectApproxEqAbs(want, v, 1e-3);
        }
    }
}

test "a view's rows are the chosen source rows over the chosen input columns, dense and affine" {
    const out = 6;
    const in = 128;
    var codes: [out * in]u32 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 11 + 5) % 16);
    var words: [out * in * 4 / 32]u32 = undefined;
    var scales: [out * 4]u16 = undefined;
    var biases: [out * 4]u16 = undefined;
    const affine = testAffine(4, out, in, 32, &codes, &words, &scales, &biases);
    var dense_buf: [out * in]f32 = undefined;
    for (&dense_buf, 0..) |*v, i| v.* = @floatFromInt(i);
    const dense: Linear = .{ .out = out, .in = in, .w = .{ .dtype = .f32, .rank = 2, .shape = .{ out, in, 1, 1 }, .data = std.mem.sliceAsBytes(&dense_buf) } };
    for ([_]Linear{ affine, dense }) |lin| {
        // Rows 4, 1 and 2 (two runs), columns [64, 96).
        const v = try lin.view(&.{ .{ 4, 1 }, .{ 1, 2 } }, 64, 32);
        try testing.expectEqual(@as(u32, 3), v.out);
        try testing.expectEqual(@as(u32, 32), v.in);
        var full: [in]f32 = undefined;
        var part: [32]f32 = undefined;
        for ([_]usize{ 4, 1, 2 }, 0..) |src, r| {
            lin.row(src, &full);
            v.row(r, &part);
            try testing.expectEqualSlices(f32, full[64..96], &part);
        }
    }
    // A window that splits an affine group is refused.
    try testing.expectError(error.BadWeightShape, affine.view(&.{}, 16, 32));
}

test "int8 rows: the row's largest weight maps to 127 and codes reproduce the row within half a step" {
    const out = 3;
    const in = 64;
    var codes: [out * in]u32 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 5 + 1) % 16);
    var words: [out * in * 4 / 32]u32 = undefined;
    var scales: [out]u16 = undefined;
    var biases: [out]u16 = undefined;
    const lin = testAffine(4, out, in, 64, &codes, &words, &scales, &biases);
    var q: [out * in]i8 = undefined;
    var s: [out]f16 = undefined;
    try convert(testing.allocator, &.{.{ .lin = lin, .codes = &q, .scales = &s }}, 4);
    var row: [in]f32 = undefined;
    for (0..out) |r| {
        lin.row(r, &row);
        var amax: f32 = 0;
        for (row) |v| amax = @max(amax, @abs(v));
        var hit127 = false;
        for (row, q[r * in ..][0..in]) |v, c| {
            if (@abs(c) == 127) hit127 = true;
            try testing.expect(@abs(@as(f32, @floatFromInt(c)) * @as(f32, s[r]) - v) <= @as(f32, s[r]) * 0.5 + 1e-4);
        }
        try testing.expect(hit127);
    }
}

test "convert: every row of every job, whatever the jobs' row counts" {
    const a = testing.allocator;
    const in = 64;
    var codes: [40 * in]u32 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast((i * 5 + 1) % 16);
    var words: [40 * in * 4 / 32]u32 = undefined;
    var scales: [40 * 2]u16 = undefined;
    var biases: [40 * 2]u16 = undefined;
    const lin = testAffine(4, 40, in, 32, &codes, &words, &scales, &biases);
    // A lane's 24 and 8 rows, as two jobs back to back, against each row converted alone.
    const views = [_]Linear{ try lin.view(&.{.{ 0, 24 }}, 0, in), try lin.view(&.{.{ 24, 8 }}, 0, in), try lin.view(&.{.{ 32, 8 }}, 0, in) };
    var got: [40 * in]i8 = @splat(0);
    var got_s: [40]f16 = @splat(0);
    var jobs: [views.len]Job = undefined;
    var at: usize = 0;
    for (views, &jobs) |v, *j| {
        j.* = .{ .lin = v, .codes = got[at * in ..][0 .. v.out * in], .scales = got_s[at..][0..v.out] };
        at += v.out;
    }
    try convert(a, &jobs, 4);
    var want: [40 * in]i8 = undefined;
    var want_s: [40]f16 = undefined;
    var scratch: [in]f32 = undefined;
    int8Rows(lin, 0, 40, &want, &want_s, &scratch);
    try testing.expectEqualSlices(i8, &want, &got);
    try testing.expectEqualSlices(f16, &want_s, &got_s);
}

test "int8 rows: a row of zeros gets a positive scale and zero codes" {
    const in = 64;
    var codes: [2 * in]u32 = @splat(3);
    var words: [2 * in * 4 / 32]u32 = undefined;
    var scales: [2 * 2]u16 = undefined;
    var biases: [2 * 2]u16 = undefined;
    const lin = testAffine(4, 2, in, 32, &codes, &words, &scales, &biases);
    // Row 0 dequantizes to zeros.
    @memset(scales[0..2], 0);
    @memset(biases[0..2], 0);
    var q: [2 * in]i8 = undefined;
    var s: [2]f16 = undefined;
    var scratch: [in]f32 = undefined;
    int8Rows(lin, 0, 2, &q, &s, &scratch);
    try testing.expect(s[0] > 0 and std.math.isFinite(s[0]));
    for (q[0..in]) |c| try testing.expectEqual(@as(i8, 0), c);
}
