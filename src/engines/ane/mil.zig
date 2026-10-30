//! MIL text programs for the Neural Engine (`lib/ane`): one program is several functions over one
//! weight blob, so every function of a program reads the same weight copy. The builder infers each
//! value's shape, names values `v<n>`, and writes every constant as a named `const` the way
//! coremltools does (the ANE compiler's MIL parser takes nothing else).
//!
//! Blob layout (coremltools' `weight.bin`): a 64-byte file header, then per tensor a 64-byte entry
//! header (0xDEADBEEF, dtype, size, data offset) followed by its data padded to 64 bytes. A `BLOBFILE`
//! reference names the entry header's offset.

const std = @import("std");

pub const DType = enum {
    fp16,
    fp32,
    int8,
    uint8,
    int32,

    pub fn mil(self: DType) []const u8 {
        return @tagName(self);
    }

    /// The blob entry's dtype code (MIL `BlobDataType`).
    fn blobCode(self: DType) u32 {
        return switch (self) {
            .fp16 => 1,
            .fp32 => 2,
            .uint8 => 3,
            .int8 => 4,
            .int32 => 14,
        };
    }

    pub fn size(self: DType) usize {
        return switch (self) {
            .fp16 => 2,
            .fp32, .int32 => 4,
            .int8, .uint8 => 1,
        };
    }
};

pub const MAX_RANK = 6;

pub const Shape = struct {
    d: [MAX_RANK]u32 = @splat(0),
    n: u8 = 0,

    pub fn of(ds: []const u32) Shape {
        std.debug.assert(ds.len <= MAX_RANK);
        var s: Shape = .{ .n = @intCast(ds.len) };
        @memcpy(s.d[0..ds.len], ds);
        return s;
    }

    pub fn dims(self: *const Shape) []const u32 {
        return self.d[0..self.n];
    }

    pub fn count(self: Shape) usize {
        var c: usize = 1;
        for (self.dims()) |x| c *= x;
        return c;
    }

    /// Axis `a`, negative from the end.
    pub fn axis(self: Shape, a: i32) usize {
        return @intCast(if (a < 0) @as(i32, self.n) + a else a);
    }

    pub fn eql(a: Shape, b: Shape) bool {
        return std.mem.eql(u32, a.dims(), b.dims());
    }
};

/// The weight file of one program, written in place: `add` reserves an entry (zeroed) and returns
/// where its data goes; the caller fills it before the program is compiled.
pub const Blob = struct {
    bytes: std.ArrayList(u8) = .empty,
    entries: u32 = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, capacity: usize) !Blob {
        var b: Blob = .{ .allocator = allocator };
        try b.bytes.ensureTotalCapacityPrecise(allocator, @max(capacity, 128));
        b.bytes.appendNTimesAssumeCapacity(0, 64);
        b.bytes.items[4] = 2; // format version
        return b;
    }

    pub fn deinit(self: *Blob) void {
        self.bytes.deinit(self.allocator);
    }

    pub const Entry = struct { header: u64, data: u64, len: usize };

    pub fn add(self: *Blob, dtype: DType, len: usize) !Entry {
        const header = self.bytes.items.len;
        const padded = len + ((64 - len % 64) % 64);
        try self.bytes.appendNTimes(self.allocator, 0, 64 + padded);
        const h = self.bytes.items[header..][0..64];
        std.mem.writeInt(u32, h[0..4], 0xDEADBEEF, .little);
        std.mem.writeInt(u32, h[4..8], dtype.blobCode(), .little);
        std.mem.writeInt(u64, h[8..16], len, .little);
        std.mem.writeInt(u64, h[16..24], header + 64, .little);
        self.entries += 1;
        std.mem.writeInt(u32, self.bytes.items[0..4], self.entries, .little);
        return .{ .header = header, .data = header + 64, .len = len };
    }

    pub fn data(self: *Blob, e: Entry) []u8 {
        return self.bytes.items[@intCast(e.data)..][0..e.len];
    }

    /// An entry holding `values` as fp16.
    pub fn addF16(self: *Blob, values: []const f32) !Entry {
        const e = try self.add(.fp16, values.len * 2);
        const dst: []align(1) f16 = std.mem.bytesAsSlice(f16, self.data(e));
        for (values, dst) |v, *d| d.* = @floatCast(v);
        return e;
    }
};

/// A linear layer's weight in the blob: INT8 codes (out x in) and one FP16 scale per output row, or
/// dense FP16. Several functions of a program reference the same entries.
pub const Weight = struct {
    out: u32,
    in: u32,
    codes: Blob.Entry,
    /// INT8 per-row scales; null for an FP16 weight held in `codes`.
    scales: ?Blob.Entry,
    /// A depthwise / grouped conv: `in` is the per-group width and the kernel spans `kw` columns.
    groups: u32 = 1,
    kw: u32 = 1,
};

pub const Var = u32;

pub const Input = struct { name: []const u8, dtype: DType = .fp16, shape: Shape };

/// One program's text. Functions are written one after another; constants local to a function go
/// to the blob as they are made.
pub const Builder = struct {
    allocator: std.mem.Allocator,
    text: std.ArrayList(u8) = .empty,
    /// The current function's body; written into `text` when the function ends.
    body: std.ArrayList(u8) = .empty,
    blob: *Blob,
    shapes: std.ArrayList(Shape) = .empty,
    dtypes: std.ArrayList(DType) = .empty,
    names: std.ArrayList([]const u8) = .empty,
    names_arena: std.heap.ArenaAllocator,
    fn_header: std.ArrayList(u8) = .empty,
    consts: u32 = 0,
    /// Inputs plus outputs over every function so far (the compiler caps a program's total).
    io: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, blob: *Blob) !Builder {
        var b: Builder = .{ .allocator = allocator, .blob = blob, .names_arena = .init(allocator) };
        try b.text.appendSlice(allocator, "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3600.16.1\"}, {\"coremlc-version\", \"3600.25.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n");
        return b;
    }

    pub fn deinit(self: *Builder) void {
        self.text.deinit(self.allocator);
        self.body.deinit(self.allocator);
        self.shapes.deinit(self.allocator);
        self.dtypes.deinit(self.allocator);
        self.names.deinit(self.allocator);
        self.fn_header.deinit(self.allocator);
        self.names_arena.deinit();
    }

    /// The finished program text (NUL-terminated, owned by the builder).
    pub fn finish(self: *Builder) ![:0]const u8 {
        try self.text.appendSlice(self.allocator, "}\n");
        try self.text.append(self.allocator, 0);
        return self.text.items[0 .. self.text.items.len - 1 :0];
    }

    pub fn shape(self: *const Builder, v: Var) Shape {
        return self.shapes.items[v];
    }

    pub fn name(self: *const Builder, v: Var) []const u8 {
        return self.names.items[v];
    }

    /// Start a function `fname` with these inputs (their vars, in order).
    pub fn begin(self: *Builder, fname: []const u8, inputs: []const Input, out: []Var) !void {
        self.fn_header.clearRetainingCapacity();
        try self.fn_header.print(self.allocator, "    func {s}<ios18>(", .{fname});
        for (inputs, 0..) |in, i| {
            if (i > 0) try self.fn_header.appendSlice(self.allocator, ", ");
            try self.fn_header.print(self.allocator, "tensor<{s}, ", .{in.dtype.mil()});
            try writeDims(self.allocator, &self.fn_header, in.shape.dims());
            try self.fn_header.print(self.allocator, "> {s}", .{in.name});
            out[i] = try self.newVar(in.shape, in.dtype, try self.names_arena.allocator().dupe(u8, in.name));
        }
        try self.fn_header.appendSlice(self.allocator, ") {\n");
        self.body.clearRetainingCapacity();
        self.io += @intCast(inputs.len);
    }

    pub fn end(self: *Builder, outputs: []const Var) !void {
        try self.text.appendSlice(self.allocator, self.fn_header.items);
        try self.text.appendSlice(self.allocator, self.body.items);
        try self.text.appendSlice(self.allocator, "    } -> (");
        for (outputs, 0..) |o, i| {
            if (i > 0) try self.text.appendSlice(self.allocator, ", ");
            try self.text.appendSlice(self.allocator, self.name(o));
        }
        try self.text.appendSlice(self.allocator, ");\n");
        self.io += @intCast(outputs.len);
    }

    fn newVar(self: *Builder, s: Shape, dt: DType, nm: ?[]const u8) !Var {
        const id: Var = @intCast(self.shapes.items.len);
        try self.shapes.append(self.allocator, s);
        try self.dtypes.append(self.allocator, dt);
        try self.names.append(self.allocator, nm orelse try std.fmt.allocPrint(self.names_arena.allocator(), "v{d}", .{id}));
        return id;
    }

    // ── constants ──

    fn constName(self: *Builder) ![]const u8 {
        self.consts += 1;
        return std.fmt.allocPrint(self.names_arena.allocator(), "c{d}", .{self.consts});
    }

    fn line(self: *Builder, comptime fmt: []const u8, args: anytype) !void {
        try self.body.print(self.allocator, "        " ++ fmt ++ "\n", args);
    }

    fn constInts(self: *Builder, vals: []const i32) ![]const u8 {
        const n = try self.constName();
        try self.body.print(self.allocator, "        tensor<int32, [{d}]> {s} = const()[name = string(\"{s}\"), val = tensor<int32, [{d}]>([", .{ vals.len, n, n, vals.len });
        for (vals, 0..) |v, i| try self.body.print(self.allocator, "{s}{d}", .{ if (i > 0) ", " else "", v });
        try self.body.appendSlice(self.allocator, "])];\n");
        return n;
    }

    fn constDims(self: *Builder, dims: []const u32) ![]const u8 {
        var v: [MAX_RANK]i32 = undefined;
        for (dims, 0..) |d, i| v[i] = @intCast(d);
        return self.constInts(v[0..dims.len]);
    }

    fn constInt(self: *Builder, v: i32) ![]const u8 {
        const n = try self.constName();
        try self.line("int32 {s} = const()[name = string(\"{s}\"), val = int32({d})];", .{ n, n, v });
        return n;
    }

    fn constBool(self: *Builder, v: bool) ![]const u8 {
        const n = try self.constName();
        try self.line("bool {s} = const()[name = string(\"{s}\"), val = bool({s})];", .{ n, n, if (v) "true" else "false" });
        return n;
    }

    fn constScalar(self: *Builder, dt: []const u8, v: f32) ![]const u8 {
        const n = try self.constName();
        try self.line("{s} {s} = const()[name = string(\"{s}\"), val = {s}({e})];", .{ dt, n, n, dt, v });
        return n;
    }

    fn constString(self: *Builder, v: []const u8) ![]const u8 {
        const n = try self.constName();
        try self.line("string {s} = const()[name = string(\"{s}\"), val = string(\"{s}\")];", .{ n, n, v });
        return n;
    }

    fn blobRef(self: *Builder, dt: DType, dims: []const u32, e: Blob.Entry) !void {
        try self.body.print(self.allocator, "tensor<{s}, ", .{dt.mil()});
        try writeDims(self.allocator, &self.body, dims);
        try self.body.print(self.allocator, ">(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64({d})))", .{e.header});
    }

    /// An FP16 constant tensor (stored in the blob).
    pub fn constant(self: *Builder, dims: []const u32, values: []const f32) !Var {
        std.debug.assert(Shape.of(dims).count() == values.len);
        const e = try self.blob.addF16(values);
        const v = try self.newVar(Shape.of(dims), .fp16, null);
        try self.body.print(self.allocator, "        tensor<fp16, ", .{});
        try writeDims(self.allocator, &self.body, dims);
        try self.body.print(self.allocator, "> {s} = const()[name = string(\"{s}\"), val = ", .{ self.name(v), self.name(v) });
        try self.blobRef(.fp16, dims, e);
        try self.body.appendSlice(self.allocator, "];\n");
        return v;
    }

    /// An FP16 tensor already in the blob (shared by several functions) as a value of this function.
    pub fn constBlob(self: *Builder, dims: []const u32, e: Blob.Entry) !Var {
        const v = try self.newVar(Shape.of(dims), .fp16, null);
        try self.body.appendSlice(self.allocator, "        tensor<fp16, ");
        try writeDims(self.allocator, &self.body, dims);
        try self.body.print(self.allocator, "> {s} = const()[name = string(\"{s}\"), val = ", .{ self.name(v), self.name(v) });
        try self.blobRef(.fp16, dims, e);
        try self.body.appendSlice(self.allocator, "];\n");
        return v;
    }

    /// An FP16 constant of zeros.
    pub fn zeros(self: *Builder, dims: []const u32) !Var {
        const e = try self.blob.add(.fp16, Shape.of(dims).count() * 2);
        return self.constBlob(dims, e);
    }

    /// A weight (blob entries made by the caller) as a conv weight value of this function.
    pub fn weight(self: *Builder, w: Weight) !Var {
        const dims = [_]u32{ w.out, w.in, 1, w.kw };
        const v = try self.newVar(Shape.of(&dims), .fp16, null);
        try self.body.appendSlice(self.allocator, "        tensor<fp16, ");
        try writeDims(self.allocator, &self.body, &dims);
        if (w.scales) |s| {
            try self.body.print(self.allocator, "> {s} = constexpr_blockwise_shift_scale(data = ", .{self.name(v)});
            try self.blobRef(.int8, &dims, w.codes);
            try self.body.appendSlice(self.allocator, ", scale = ");
            try self.blobRef(.fp16, &.{ w.out, 1, 1, 1 }, s);
        } else {
            try self.body.print(self.allocator, "> {s} = const()[name = string(\"{s}\"), val = ", .{ self.name(v), self.name(v) });
            try self.blobRef(.fp16, &dims, w.codes);
            try self.body.appendSlice(self.allocator, "];\n");
            return v;
        }
        try self.body.print(self.allocator, ")[name = string(\"{s}\")];\n", .{self.name(v)});
        return v;
    }

    // ── ops ──

    const Arg = struct { k: []const u8, v: []const u8 };

    fn op(self: *Builder, opname: []const u8, s: Shape, dt: DType, args: []const Arg, nm: ?[]const u8) !Var {
        const v = try self.newVar(s, dt, nm);
        try self.body.print(self.allocator, "        tensor<{s}, ", .{dt.mil()});
        try writeDims(self.allocator, &self.body, s.dims());
        try self.body.print(self.allocator, "> {s} = {s}(", .{ self.name(v), opname });
        for (args, 0..) |a, i| try self.body.print(self.allocator, "{s}{s} = {s}", .{ if (i > 0) ", " else "", a.k, a.v });
        try self.body.print(self.allocator, ")[name = string(\"{s}\")];\n", .{self.name(v)});
        return v;
    }

    fn broadcast(a: Shape, b: Shape) Shape {
        const n = @max(a.n, b.n);
        var out: Shape = .{ .n = n };
        for (0..n) |i| {
            const da: u32 = if (i + a.n >= n) a.d[i + a.n - n] else 1;
            const db: u32 = if (i + b.n >= n) b.d[i + b.n - n] else 1;
            std.debug.assert(da == db or da == 1 or db == 1);
            out.d[i] = @max(da, db);
        }
        return out;
    }

    fn binary(self: *Builder, opname: []const u8, x: Var, y: Var) !Var {
        return self.op(opname, broadcast(self.shape(x), self.shape(y)), .fp16, &.{ .{ .k = "x", .v = self.name(x) }, .{ .k = "y", .v = self.name(y) } }, null);
    }

    fn binaryS(self: *Builder, opname: []const u8, x: Var, y: f32) !Var {
        const c = try self.constScalar("fp16", y);
        return self.op(opname, self.shape(x), .fp16, &.{ .{ .k = "x", .v = self.name(x) }, .{ .k = "y", .v = c } }, null);
    }

    pub fn add(self: *Builder, x: Var, y: Var) !Var {
        return self.binary("add", x, y);
    }
    pub fn sub(self: *Builder, x: Var, y: Var) !Var {
        return self.binary("sub", x, y);
    }
    pub fn mul(self: *Builder, x: Var, y: Var) !Var {
        return self.binary("mul", x, y);
    }
    pub fn div(self: *Builder, x: Var, y: Var) !Var {
        return self.binary("real_div", x, y);
    }
    pub fn minimum(self: *Builder, x: Var, y: Var) !Var {
        return self.binary("minimum", x, y);
    }
    pub fn maximum(self: *Builder, x: Var, y: Var) !Var {
        return self.binary("maximum", x, y);
    }
    pub fn addS(self: *Builder, x: Var, y: f32) !Var {
        return self.binaryS("add", x, y);
    }
    pub fn mulS(self: *Builder, x: Var, y: f32) !Var {
        return self.binaryS("mul", x, y);
    }
    pub fn minS(self: *Builder, x: Var, y: f32) !Var {
        return self.binaryS("minimum", x, y);
    }
    pub fn maxS(self: *Builder, x: Var, y: f32) !Var {
        return self.binaryS("maximum", x, y);
    }

    fn unary(self: *Builder, opname: []const u8, x: Var) !Var {
        return self.op(opname, self.shape(x), .fp16, &.{.{ .k = "x", .v = self.name(x) }}, null);
    }
    pub fn exp(self: *Builder, x: Var) !Var {
        return self.unary("exp", x);
    }
    pub fn tanh(self: *Builder, x: Var) !Var {
        return self.unary("tanh", x);
    }
    pub fn sigmoid(self: *Builder, x: Var) !Var {
        return self.unary("sigmoid", x);
    }
    pub fn relu(self: *Builder, x: Var) !Var {
        return self.unary("relu", x);
    }
    pub fn abs(self: *Builder, x: Var) !Var {
        return self.unary("abs", x);
    }

    pub fn log(self: *Builder, x: Var) !Var {
        const e = try self.constScalar("fp16", 0);
        return self.op("log", self.shape(x), .fp16, &.{ .{ .k = "epsilon", .v = e }, .{ .k = "x", .v = self.name(x) } }, null);
    }

    pub fn rsqrt(self: *Builder, x: Var, eps: f32) !Var {
        const e = try self.constScalar("fp32", eps);
        return self.op("rsqrt", self.shape(x), .fp16, &.{ .{ .k = "epsilon", .v = e }, .{ .k = "x", .v = self.name(x) } }, null);
    }

    pub fn reshape(self: *Builder, x: Var, dims: []const u32) !Var {
        std.debug.assert(Shape.of(dims).count() == self.shape(x).count());
        const c = try self.constDims(dims);
        return self.op("reshape", Shape.of(dims), self.dtypes.items[x], &.{ .{ .k = "shape", .v = c }, .{ .k = "x", .v = self.name(x) } }, null);
    }

    pub fn transpose(self: *Builder, x: Var, perm: []const u32) !Var {
        const s = self.shape(x);
        var o: Shape = .{ .n = s.n };
        for (perm, 0..) |p, i| o.d[i] = s.d[p];
        const c = try self.constDims(perm);
        return self.op("transpose", o, self.dtypes.items[x], &.{ .{ .k = "perm", .v = c }, .{ .k = "x", .v = self.name(x) } }, null);
    }

    pub fn matmul(self: *Builder, x: Var, y: Var, tx: bool, ty: bool) !Var {
        var a = self.shape(x);
        var b = self.shape(y);
        if (tx) std.mem.swap(u32, &a.d[a.n - 1], &a.d[a.n - 2]);
        if (ty) std.mem.swap(u32, &b.d[b.n - 1], &b.d[b.n - 2]);
        std.debug.assert(a.d[a.n - 1] == b.d[b.n - 2]);
        var out = broadcast(Shape.of(a.dims()[0 .. a.n - 2]), Shape.of(b.dims()[0 .. b.n - 2]));
        out.d[out.n] = a.d[a.n - 2];
        out.d[out.n + 1] = b.d[b.n - 1];
        out.n += 2;
        const ctx = try self.constBool(tx);
        const cty = try self.constBool(ty);
        return self.op("matmul", out, .fp16, &.{ .{ .k = "transpose_x", .v = ctx }, .{ .k = "transpose_y", .v = cty }, .{ .k = "x", .v = self.name(x) }, .{ .k = "y", .v = self.name(y) } }, null);
    }

    pub fn concat(self: *Builder, vals: []const Var, axis_: i32) !Var {
        var s = self.shape(vals[0]);
        const ax = s.axis(axis_);
        s.d[ax] = 0;
        var list: std.ArrayList(u8) = .empty;
        defer list.deinit(self.allocator);
        try list.append(self.allocator, '(');
        for (vals, 0..) |v, i| {
            s.d[ax] += self.shape(v).d[ax];
            if (i > 0) try list.appendSlice(self.allocator, ", ");
            try list.appendSlice(self.allocator, self.name(v));
        }
        try list.append(self.allocator, ')');
        const ca = try self.constInt(axis_);
        const ci = try self.constBool(false);
        return self.op("concat", s, .fp16, &.{ .{ .k = "axis", .v = ca }, .{ .k = "interleave", .v = ci }, .{ .k = "values", .v = list.items } }, null);
    }

    /// Elements [begin, end) along every axis.
    pub fn slice(self: *Builder, x: Var, begin_: []const u32, end_: []const u32) !Var {
        var s: Shape = .{ .n = @intCast(begin_.len) };
        for (begin_, end_, 0..) |b, e, i| s.d[i] = e - b;
        const cb = try self.constDims(begin_);
        const ce = try self.constDims(end_);
        return self.op("slice_by_index", s, .fp16, &.{ .{ .k = "begin", .v = cb }, .{ .k = "end", .v = ce }, .{ .k = "x", .v = self.name(x) } }, null);
    }

    /// Axis `axis_` cut to [b, e), every other axis whole.
    pub fn sliceAxis(self: *Builder, x: Var, axis_: i32, b: u32, e: u32) !Var {
        const s = self.shape(x);
        var lo: [MAX_RANK]u32 = @splat(0);
        var hi: [MAX_RANK]u32 = s.d;
        const ax = s.axis(axis_);
        lo[ax] = b;
        hi[ax] = e;
        return self.slice(x, lo[0..s.n], hi[0..s.n]);
    }

    fn reduce(self: *Builder, opname: []const u8, x: Var, axes: []const i32) !Var {
        var s = self.shape(x);
        for (axes) |a| s.d[s.axis(a)] = 1;
        const ca = try self.constInts(axes);
        const ck = try self.constBool(true);
        return self.op(opname, s, .fp16, &.{ .{ .k = "axes", .v = ca }, .{ .k = "keep_dims", .v = ck }, .{ .k = "x", .v = self.name(x) } }, null);
    }
    pub fn sum(self: *Builder, x: Var, axes: []const i32) !Var {
        return self.reduce("reduce_sum", x, axes);
    }
    pub fn max(self: *Builder, x: Var, axes: []const i32) !Var {
        return self.reduce("reduce_max", x, axes);
    }
    pub fn mean(self: *Builder, x: Var, axes: []const i32) !Var {
        return self.reduce("reduce_mean", x, axes);
    }

    pub fn tile(self: *Builder, x: Var, reps: []const u32) !Var {
        var s = self.shape(x);
        for (reps, 0..) |r, i| s.d[i] *= r;
        const c = try self.constDims(reps);
        return self.op("tile", s, .fp16, &.{ .{ .k = "reps", .v = c }, .{ .k = "x", .v = self.name(x) } }, null);
    }

    pub fn softmax(self: *Builder, x: Var, axis_: i32) !Var {
        const c = try self.constInt(axis_);
        return self.op("softmax", self.shape(x), .fp16, &.{ .{ .k = "axis", .v = c }, .{ .k = "x", .v = self.name(x) } }, null);
    }

    /// `x` under the name `nm` (a function output).
    pub fn named(self: *Builder, x: Var, nm: []const u8) !Var {
        return self.op("identity", self.shape(x), self.dtypes.items[x], &.{.{ .k = "x", .v = self.name(x) }}, try self.names_arena.allocator().dupe(u8, nm));
    }

    /// 1x`kw` conv of x [1, C, 1, W] with `w` (a `weight` value), valid padding.
    pub fn conv(self: *Builder, x: Var, w: Var, groups: u32) !Var {
        const xs = self.shape(x);
        const ws = self.shape(w);
        const out = Shape.of(&.{ xs.d[0], ws.d[0], xs.d[2], xs.d[3] - (ws.d[3] - 1) });
        const cd = try self.constInts(&.{ 1, 1 });
        const cg = try self.constInt(@intCast(groups));
        const cp = try self.constInts(&.{ 0, 0, 0, 0 });
        const ct = try self.constString("valid");
        const cs = try self.constInts(&.{ 1, 1 });
        return self.op("conv", out, .fp16, &.{
            .{ .k = "dilations", .v = cd },
            .{ .k = "groups", .v = cg },
            .{ .k = "pad", .v = cp },
            .{ .k = "pad_type", .v = ct },
            .{ .k = "strides", .v = cs },
            .{ .k = "weight", .v = self.name(w) },
            .{ .k = "x", .v = self.name(x) },
        }, null);
    }
};

fn writeDims(a: std.mem.Allocator, out: *std.ArrayList(u8), dims: []const u32) !void {
    try out.append(a, '[');
    for (dims, 0..) |d, i| try out.print(a, "{s}{d}", .{ if (i > 0) ", " else "", d });
    try out.append(a, ']');
}

// ── tests ──

const testing = std.testing;

test "blob: file header counts entries; each entry header points at its 64-byte aligned data" {
    var b = try Blob.init(testing.allocator, 0);
    defer b.deinit();
    const e1 = try b.add(.int8, 10);
    const e2 = try b.addF16(&.{ 1.0, -2.0 });
    try testing.expectEqual(@as(u64, 64), e1.header);
    try testing.expectEqual(@as(u64, 128), e1.data);
    try testing.expectEqual(@as(u64, 128 + 64), e2.header);
    try testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, b.bytes.items[0..4], .little));
    try testing.expectEqual(@as(u8, 2), b.bytes.items[4]);
    const h = b.bytes.items[@intCast(e2.header)..][0..64];
    try testing.expectEqual(@as(u32, 0xDEADBEEF), std.mem.readInt(u32, h[0..4], .little));
    try testing.expectEqual(@as(u32, 1), std.mem.readInt(u32, h[4..8], .little));
    try testing.expectEqual(@as(u64, 4), std.mem.readInt(u64, h[8..16], .little));
    try testing.expectEqual(e2.data, std.mem.readInt(u64, h[16..24], .little));
    const v: []align(1) const f16 = std.mem.bytesAsSlice(f16, b.data(e2));
    try testing.expectEqual(@as(f16, -2.0), v[1]);
    try testing.expectEqual(@as(usize, 0), b.bytes.items.len % 64);
}

test "builder: shapes follow the ops, and the text names every input, const and output" {
    var blob = try Blob.init(testing.allocator, 0);
    defer blob.deinit();
    var g = try Builder.init(testing.allocator, &blob);
    defer g.deinit();
    var in: [2]Var = undefined;
    try g.begin("main", &.{ .{ .name = "a", .shape = Shape.of(&.{ 4, 8, 32 }) }, .{ .name = "b", .shape = Shape.of(&.{ 4, 32, 16 }) } }, &in);
    const m = try g.matmul(in[0], in[1], false, false);
    try testing.expect(g.shape(m).eql(Shape.of(&.{ 4, 8, 16 })));
    const t = try g.transpose(m, &.{ 0, 2, 1 });
    try testing.expect(g.shape(t).eql(Shape.of(&.{ 4, 16, 8 })));
    const r = try g.sum(t, &.{-1});
    try testing.expect(g.shape(r).eql(Shape.of(&.{ 4, 16, 1 })));
    const c = try g.concat(&.{ t, t }, 1);
    try testing.expect(g.shape(c).eql(Shape.of(&.{ 4, 32, 8 })));
    const s = try g.sliceAxis(c, 1, 3, 7);
    try testing.expect(g.shape(s).eql(Shape.of(&.{ 4, 4, 8 })));
    const tt = try g.matmul(in[0], in[0], false, true);
    try testing.expect(g.shape(tt).eql(Shape.of(&.{ 4, 8, 8 })));
    const out = try g.named(try g.mulS(s, 0.5), "y");
    try g.end(&.{out});
    const text = try g.finish();
    try testing.expect(std.mem.indexOf(u8, text, "func main<ios18>(tensor<fp16, [4, 8, 32]> a, tensor<fp16, [4, 32, 16]> b) {") != null);
    try testing.expect(std.mem.indexOf(u8, text, "} -> (y);") != null);
    try testing.expect(std.mem.indexOf(u8, text, "tensor<fp16, [4, 8, 16]> v2 = matmul(") != null);
    try testing.expect(std.mem.endsWith(u8, text, "}\n"));
}

test "builder: an INT8 weight is a per-row shift-scale over blob entries; its conv keeps the rows" {
    var blob = try Blob.init(testing.allocator, 0);
    defer blob.deinit();
    var g = try Builder.init(testing.allocator, &blob);
    defer g.deinit();
    const w: Weight = .{ .out = 96, .in = 64, .codes = try blob.add(.int8, 96 * 64), .scales = try blob.add(.fp16, 96 * 2) };
    var in: [1]Var = undefined;
    try g.begin("main", &.{.{ .name = "x", .shape = Shape.of(&.{ 1, 64, 1, 8 }) }}, &in);
    const y = try g.conv(in[0], try g.weight(w), 1);
    try testing.expect(g.shape(y).eql(Shape.of(&.{ 1, 96, 1, 8 })));
    try g.end(&.{y});
    const text = try g.finish();
    try testing.expect(std.mem.indexOf(u8, text, "constexpr_blockwise_shift_scale(data = tensor<int8, [96, 64, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))") != null);
    try testing.expect(std.mem.indexOf(u8, text, "scale = tensor<fp16, [96, 1, 1, 1]>") != null);
}
