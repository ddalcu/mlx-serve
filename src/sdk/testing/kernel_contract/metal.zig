//! The Metal side of the kernel contract: a kernel's launch as it registers it (the MLX `metal_kernel` it builds,
//! its threadgroup bytes and recorded M1/M2 limits, and per shape its grid, groups, template args and outputs), and
//! the rig that runs one launch for gates 1-4: inputs inside guard bytes, outputs poisoned, bytes back on the host.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const limits = @import("limits.zig");
const spec = @import("spec.zig");
const crt = @import("../../util/crt.zig");

extern "c" fn mlx_fast_metal_kernel_config_set_init_value(cls: mlx.mlx_fast_metal_kernel_config, value: f32) c_int;

/// A template argument as `mlx_fast_metal_kernel_config_add_template_arg_*` takes it.
pub const TArg = struct {
    name: [*:0]const u8,
    v: union(enum) { int: c_int, flag: bool, dtype: mlx.mlx_dtype },
};

/// An output's shape and dtype.
pub const Out = struct { shape: [4]c_int = @splat(0), ndim: usize, dtype: mlx.mlx_dtype };

/// One dispatch at one shape: grid and threadgroup (MLX's thread counts), template args, outputs, the poison the
/// outputs start as (MLX's `init_value`), so an element the kernel never writes reads back as it.
pub const Geometry = struct {
    grid: [3]c_int,
    tg: [3]c_int,
    outs: [6]Out = undefined,
    n_outs: usize = 0,
    targs: [20]TArg = undefined,
    n_targs: usize = 0,
    poison: f32 = std.math.nan(f32),
    /// Outputs (bit i: output i) this dispatch never writes by design: left out of the output bytes.
    unread: u8 = 0,

    pub fn out(g: *Geometry, shape: []const c_int, dt: mlx.mlx_dtype) void {
        var o = Out{ .ndim = shape.len, .dtype = dt };
        @memcpy(o.shape[0..shape.len], shape);
        g.outs[g.n_outs] = o;
        g.n_outs += 1;
    }

    pub fn int(g: *Geometry, name: [*:0]const u8, v: anytype) void {
        g.targs[g.n_targs] = .{ .name = name, .v = .{ .int = @intCast(v) } };
        g.n_targs += 1;
    }

    pub fn flag(g: *Geometry, name: [*:0]const u8, v: bool) void {
        g.targs[g.n_targs] = .{ .name = name, .v = .{ .flag = v } };
        g.n_targs += 1;
    }

    pub fn dtype(g: *Geometry, name: [*:0]const u8, dt: mlx.mlx_dtype) void {
        g.targs[g.n_targs] = .{ .name = name, .v = .{ .dtype = dt } };
        g.n_targs += 1;
    }

    pub fn threads(g: Geometry) u32 {
        return @intCast(g.tg[0] * g.tg[1] * g.tg[2]);
    }
};

/// The M1/M2 thread limits one instantiation compiles for; a zero field of the shape matches any size.
pub const Cap = struct { rows: usize = 0, n: usize = 0, k: usize = 0, m1: u32, m2: u32 };

/// A kernel's launch: the `metal_kernel` it builds and what gate 5 checks of it.
pub const Launch = struct {
    name: [:0]const u8,
    inputs: []const [*:0]const u8,
    outputs: []const [*:0]const u8,
    source: [:0]const u8,
    header: [:0]const u8 = "",
    /// MLX's `ensure_row_contiguous`: false passes a strided input as it is (the kernel reads its strides).
    contiguous: bool = true,
    atomic_outputs: bool = false,
    /// Threadgroup bytes its source declares, at its largest instantiation.
    tg_bytes: u32,
    /// M1 / M2 compiled thread limits of its instantiations (`offline.zig` derives them): `caps` where one
    /// matches the shape, else these; 0: none recorded.
    m1: u32 = 0,
    m2: u32 = 0,
    caps: []const Cap = &.{},
    /// The launcher probes or reads the pipeline's thread limit and declines past it, at every instantiation.
    adapts: bool = false,
    /// False for kernels the toolchain cannot build for M1/M2 ahead of time (tensor ops).
    offline: bool = true,
    /// The least Apple family it is dispatched on (13 = M1, 17 = M5/NAX).
    min_gen: u32 = 13,

    /// The recorded M1 / M2 limits of the instantiation at `shape`.
    pub fn limitsAt(l: Launch, shape: spec.Shape) [2]u32 {
        for (l.caps) |c| {
            if ((c.rows == 0 or c.rows == shape.rows) and (c.n == 0 or c.n == shape.n) and (c.k == 0 or c.k == shape.k)) return .{ c.m1, c.m2 };
        }
        return .{ l.m1, l.m2 };
    }

    /// Gate 5's record of this launch's dispatch at `shape`.
    pub fn record(l: Launch, name: []const u8, shape: spec.Shape, threads: u32) limits.MetalLaunch {
        const m = l.limitsAt(shape);
        return .{ .name = name, .threads = threads, .tg_memory = l.tg_bytes, .m1 = m[0], .m2 = m[1], .adapts = l.adapts, .offline = l.offline, .min_gen = l.min_gen };
    }
};

/// What a bool output starts as: a byte that is neither false nor true.
const BOOL_POISON: f32 = 0xaa;

/// Guard bytes on each side of every input buffer, and their pattern.
const GUARD = 64;
const PATTERN: u8 = 0xa5;

fn itemSize(dt: mlx.mlx_dtype) usize {
    return switch (dt) {
        .bool_, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        else => 8,
    };
}

/// `v` stored as `dt` at `dst` (little-endian); float values are expected already rounded to `dt`.
fn store(dt: mlx.mlx_dtype, dst: []u8, v: f64) !void {
    switch (dt) {
        .float32 => std.mem.writeInt(u32, dst[0..4], @bitCast(@as(f32, @floatCast(v))), .little),
        .float16 => std.mem.writeInt(u16, dst[0..2], @bitCast(@as(f16, @floatCast(v))), .little),
        .bfloat16 => std.mem.writeInt(u16, dst[0..2], spec.toBf16(@floatCast(v)), .little),
        .int32 => std.mem.writeInt(i32, dst[0..4], @intFromFloat(v), .little),
        .uint32 => std.mem.writeInt(u32, dst[0..4], @intFromFloat(v), .little),
        .uint16 => std.mem.writeInt(u16, dst[0..2], @intFromFloat(v), .little),
        .uint8 => dst[0] = @intFromFloat(v),
        .int8 => dst[0] = @bitCast(@as(i8, @intFromFloat(v))),
        else => return error.UnsupportedInputDtype,
    }
}

fn bytes(x: mlx.mlx_array) ![]const u8 {
    try mlx.check(mlx.mlx_array_eval(x));
    const p = mlx.mlx_array_data_uint8(x) orelse return error.NoData;
    return p[0 .. mlx.mlx_array_size(x) * mlx.mlx_array_itemsize(x)];
}

const Input = struct { parent: mlx.mlx_array, view: mlx.mlx_array, len: usize };

/// One launch's buffers on the device. A test type holds one as `rig` and calls `reset`, `input`, then `launch`.
pub const Rig = struct {
    a: std.mem.Allocator,
    s: mlx.mlx_stream = .{ .ctx = null },
    ins: std.ArrayList(Input) = .empty,
    outs: std.ArrayList(mlx.mlx_array) = .empty,
    out_bytes: std.ArrayList(u8) = .empty,
    in_bytes: std.ArrayList(u8) = .empty,
    scratch: std.ArrayList(u8) = .empty,
    kernels: std.ArrayList(struct { l: *const Launch, k: mlx.mlx_fast_metal_kernel }) = .empty,
    /// Set: the next launch also writes MLX's generated source for it here (`offline.zig`).
    capture: ?*std.ArrayList(u8) = null,

    /// MLX-free until the first input, so a registration can be listed (gate 5) with no device.
    pub fn init(a: std.mem.Allocator) Rig {
        return .{ .a = a };
    }

    pub fn stream(r: *Rig) mlx.mlx_stream {
        if (r.s.ctx == null) {
            mlx.installErrorHandler();
            r.s = mlx.gpuStream();
        }
        return r.s;
    }

    pub fn deinit(r: *Rig) void {
        r.reset();
        for (r.kernels.items) |e| _ = mlx.mlx_fast_metal_kernel_free(e.k);
        r.kernels.deinit(r.a);
        r.ins.deinit(r.a);
        r.outs.deinit(r.a);
        r.out_bytes.deinit(r.a);
        r.in_bytes.deinit(r.a);
        r.scratch.deinit(r.a);
    }

    /// Frees the inputs and outputs of the last prepare.
    pub fn reset(r: *Rig) void {
        for (r.ins.items) |in| {
            _ = mlx.mlx_array_free(in.view);
            _ = mlx.mlx_array_free(in.parent);
        }
        r.ins.clearRetainingCapacity();
        r.freeOuts();
    }

    fn freeOuts(r: *Rig) void {
        for (r.outs.items) |o| _ = mlx.mlx_array_free(o);
        r.outs.clearRetainingCapacity();
    }

    /// An input of `shape` holding `values` as `dt`, inside guard bytes (a view into a larger buffer), owned by the
    /// rig until the next reset. An input under 8 elements MLX passes by value, so it gets no guards.
    pub fn input(r: *Rig, dt: mlx.mlx_dtype, shape: []const c_int, values: []const f64) !mlx.mlx_array {
        const item = itemSize(dt);
        const raw = try r.a.alloc(u8, values.len * item);
        defer r.a.free(raw);
        for (values, 0..) |v, i| try store(dt, raw[i * item ..][0..item], v);
        return r.inputRaw(dt, shape, raw);
    }

    /// `input` from the elements' little-endian bytes (packed tables, 64-bit params no f64 holds).
    pub fn inputRaw(r: *Rig, dt: mlx.mlx_dtype, shape: []const c_int, raw: []const u8) !mlx.mlx_array {
        const item = itemSize(dt);
        const len = raw.len / item;
        const g = if (len < 8) 0 else GUARD / item;
        const n = len + 2 * g;
        const host = try r.a.alloc(u8, n * item);
        defer r.a.free(host);
        @memset(host, PATTERN);
        @memcpy(host[g * item ..][0..raw.len], raw);
        const parent = mlx.mlx_array_new_data(host.ptr, &[_]c_int{@intCast(n)}, 1, dt);
        errdefer _ = mlx.mlx_array_free(parent);
        try mlx.check(mlx.mlx_array_eval(parent));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_slice(&flat, parent, &[_]c_int{@intCast(g)}, 1, &[_]c_int{@intCast(g + len)}, 1, &[_]c_int{1}, 1, r.stream()));
        var view = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_reshape(&view, flat, shape.ptr, shape.len, r.stream()));
        try mlx.check(mlx.mlx_array_eval(view));
        try r.ins.append(r.a, .{ .parent = parent, .view = view, .len = len });
        return view;
    }

    /// A 0-dim f32 input (an eps): MLX passes it by value.
    pub fn scalar(r: *Rig, v: f32) !mlx.mlx_array {
        const x = mlx.mlx_array_new_float(v);
        try r.ins.append(r.a, .{ .parent = x, .view = mlx.mlx_array_new(), .len = 1 });
        return x;
    }

    fn kernelFor(r: *Rig, l: *const Launch) !mlx.mlx_fast_metal_kernel {
        // Metal source only runs on Metal; launched on CUDA it latches an MLX error every later test inherits.
        if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
        for (r.kernels.items) |e| if (e.l == l) return e.k;
        // Its own name: MLX caches libraries by name, so the shipped wrapper's build never stands in for this one.
        const name = try std.fmt.allocPrintSentinel(r.a, "contract_{s}", .{l.name}, 0);
        defer r.a.free(name);
        const in_vec = mlx.mlx_vector_string_new_data(l.inputs.ptr, l.inputs.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(l.outputs.ptr, l.outputs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new(name.ptr, in_vec, out_vec, l.source.ptr, l.header.ptr, l.contiguous, l.atomic_outputs);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        try r.kernels.append(r.a, .{ .l = l, .k = k });
        return k;
    }

    /// One dispatch of `l` at `g` on `args` (the inputs, in the launch's order), waited for; the outputs' bytes
    /// land in `outputBytes`.
    pub fn launch(r: *Rig, l: *const Launch, g: Geometry, args: []const mlx.mlx_array) !void {
        const k = try r.kernelFor(l);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        // A bool output is bound as bytes so its poison can be neither 0 nor 1 (`spec.read` then says unwritten).
        var bools: usize = 0;
        for (g.outs[0..g.n_outs]) |o| {
            if (o.dtype == .bool_) bools += 1;
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &o.shape, o.ndim, if (o.dtype == .bool_) .uint8 else o.dtype));
        }
        if (bools != 0 and bools != g.n_outs) return error.MixedBoolOutputs;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, g.grid[0], g.grid[1], g.grid[2]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, g.tg[0], g.tg[1], g.tg[2]));
        for (g.targs[0..g.n_targs]) |t| try mlx.check(switch (t.v) {
            .int => |v| mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, v),
            .flag => |v| mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, t.name, v),
            .dtype => |v| mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t.name, v),
        });
        try mlx.check(mlx_fast_metal_kernel_config_set_init_value(cfg, if (bools > 0) BOOL_POISON else g.poison));
        if (r.capture != null) try mlx.check(mlx.mlx_fast_metal_kernel_config_set_verbose(cfg, true));
        const v = mlx.mlx_vector_array_new_data(args.ptr, args.len);
        defer _ = mlx.mlx_vector_array_free(v);
        var o = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(o);
        if (r.capture) |dst| {
            try captureStdout(r.a, dst, mlx.mlx_fast_metal_kernel_apply, .{ &o, k, v, cfg, r.stream() });
            r.capture = null;
        } else try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, k, v, cfg, r.stream()));
        r.freeOuts();
        for (0..mlx.mlx_vector_array_size(o)) |i| {
            var x = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&x, o, i));
            try r.outs.append(r.a, x);
        }
        _ = mlx.mlx_eval(o);
        try latched();
        r.out_bytes.clearRetainingCapacity();
        for (r.outs.items, 0..) |x, i| if (g.unread & (@as(u8, 1) << @intCast(i)) == 0) try r.out_bytes.appendSlice(r.a, try bytes(x));
    }

    /// The i-th output of the last launch (a rig-owned handle).
    pub fn output(r: *Rig, i: usize) mlx.mlx_array {
        return r.outs.items[i];
    }

    pub fn outputBytes(r: *Rig) []const u8 {
        return r.out_bytes.items;
    }

    /// Every input's whole buffer, guards included, as it is now.
    pub fn inputBytes(r: *Rig) ![]const u8 {
        r.in_bytes.clearRetainingCapacity();
        for (r.ins.items) |in| try r.in_bytes.appendSlice(r.a, try bytes(in.parent));
        return r.in_bytes.items;
    }

    /// The guard bytes around every input still hold their pattern.
    pub fn intact(r: *Rig) !bool {
        for (r.ins.items) |in| {
            const b = try bytes(in.parent);
            const g = (b.len - in.len * mlx.mlx_array_itemsize(in.parent)) / 2;
            for (b[0..g]) |x| if (x != PATTERN) return false;
            for (b[b.len - g ..]) |x| if (x != PATTERN) return false;
        }
        return true;
    }

    /// The bytes of `arrs` (evaluated, each contiguous), concatenated: a replaced chain's baseline.
    pub fn bytesOf(r: *Rig, arrs: []const mlx.mlx_array) ![]const u8 {
        r.scratch.clearRetainingCapacity();
        for (arrs) |x| {
            var c = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(c);
            try mlx.check(mlx.mlx_contiguous(&c, x, false, r.stream()));
            try mlx.check(mlx.mlx_array_eval(c));
            try latched();
            try r.scratch.appendSlice(r.a, try bytes(c));
        }
        return r.scratch.items;
    }
};

/// An MLX failure since the last check, printed and returned as an error (the latch is dropped). A pipeline the GPU
/// caps below the launch's threadgroup (M1/M2) is `error.ThreadgroupCap`: the engine's wrapper declines it there.
pub fn latched() !void {
    var buf: [512]u8 = undefined;
    if (mlx.takeError(&buf)) |msg| {
        std.debug.print("[kernel-contract] mlx: {s}\n", .{msg});
        if (std.mem.indexOf(u8, msg, "maximum allowed threads per threadgroup") != null) return error.ThreadgroupCap;
        return error.MlxError;
    }
}

/// Calls `f(args)` with fd 1 sent to a file, then appends what MLX's verbose printer wrote between its fences.
fn captureStdout(a: std.mem.Allocator, dst: *std.ArrayList(u8), comptime f: anytype, args: anytype) !void {
    crt.mkdir(".zig-cache/tmp");
    const tmp = ".zig-cache/tmp/kernel-contract-source.txt";
    const fd = crt.openWrite(tmp, true);
    if (fd < 0) return error.CaptureFailed;
    const saved = std.c.dup(1);
    if (saved < 0) {
        crt.close(fd);
        return error.CaptureFailed;
    }
    _ = std.c.dup2(fd, 1);
    crt.close(fd);
    const ret = @call(.auto, f, args);
    _ = std.c.dup2(saved, 1);
    crt.close(saved);
    try mlx.check(ret);
    const rd = crt.openRead(tmp);
    if (rd < 0) return error.CaptureFailed;
    defer crt.close(rd);
    const end = crt.seekEnd(rd);
    if (end <= 0) return error.CaptureFailed;
    const text = try a.alloc(u8, @intCast(end));
    defer a.free(text);
    if (crt.pread(rd, text.ptr, text.len, 0) != end) return error.CaptureFailed;
    const open = std.mem.indexOf(u8, text, "```\n") orelse return error.CaptureFailed;
    const close = std.mem.lastIndexOf(u8, text, "\n```") orelse return error.CaptureFailed;
    try dst.appendSlice(a, text[open + 4 .. close]);
}
