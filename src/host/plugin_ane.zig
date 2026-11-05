//! `Host.ane` (`abi/host.zig`'s `HostAne`): `lib/ane`'s programs and banks for plugins. One runtime serves the host's
//! own offloads and every plugin, so the compile cache, the resident-model count and the dies are shared. Null where the
//! build or the machine has no Neural Engine.

const std = @import("std");
const builtin = @import("builtin");
const abi = @import("../abi/host.zig");

const Program = abi.AneProgram;
const Request = abi.AneRequest;
const Surface = abi.AneSurface;
const Bank = abi.AneBank;
const BankProgram = abi.AneBankProgram;
const Plane = abi.AnePlane;

extern fn msv_ane_bridge_available() c_int;
extern fn msv_ane_resident() u32;
extern fn msv_ane_bridge_surface(bytes: usize) ?*Surface;
extern fn msv_ane_program_create(name: [*:0]const u8, mil: [*:0]const u8, weights: ?*anyopaque, weight_bytes: usize, instances: u32, hints: ?[*]const u32, err: [*]u8, err_len: usize) ?*Program;
extern fn msv_ane_program_free(p: ?*Program) void;
extern fn msv_ane_program_procedure(p: *Program, function: [*:0]const u8) c_int;
extern fn msv_ane_program_io_count(p: *Program, procedure: u32, output: c_int) u32;
extern fn msv_ane_program_io(p: *Program, procedure: u32, output: c_int, index: u32, out: *abi.AneIo) c_int;
extern fn msv_ane_program_compile_seconds(p: *const Program) f64;
extern fn msv_ane_program_cache_hit(p: *const Program) bool;
extern fn msv_ane_request_create(p: *Program, instance: u32, procedure: u32, inputs: [*]const ?*Surface, n_in: u32, outputs: [*]const ?*Surface, n_out: u32, err: [*]u8, err_len: usize) ?*Request;
extern fn msv_ane_request_eval(r: *Request, err: [*]u8, err_len: usize) c_int;
extern fn msv_ane_request_free(r: ?*Request) void;
extern fn msv_ane_plane_create(bytes: usize) ?*Plane;
extern fn msv_ane_plane_free(p: ?*Plane) void;
extern fn msv_ane_plane_base(p: *Plane) ?[*]f16;
extern fn msv_ane_bank_create() ?*Bank;
extern fn msv_ane_bank_free(b: ?*Bank) void;
extern fn msv_ane_bank_add_mlp(b: *Bank, hidden: u32, ffn: u32, rows: u32, gate_q: [*]const i8, gate_s: [*]const f32, up_q: [*]const i8, up_s: [*]const f32, down_q: [*]const i8, down_s: [*]const f32, limit: f32, err: [*]u8, err_len: usize) c_int;
extern fn msv_ane_bank_add_gdn(b: *Bank, hidden: u32, qkv_out: u32, z_out: u32, rows: u32, qkv_q: [*]const i8, qkv_s: [*]const f32, z_q: [*]const i8, z_s: [*]const f32, err: [*]u8, err_len: usize) c_int;
extern fn msv_ane_bank_finish(b: *Bank, name: [*:0]const u8, die: c_int, input: ?*Plane, output: ?*Plane, err: [*]u8, err_len: usize) ?*BankProgram;
extern fn msv_ane_mlp_free(m: ?*BankProgram) void;
extern fn msv_ane_mlp_input(m: *BankProgram) ?[*]f16;
extern fn msv_ane_mlp_output(m: *BankProgram) ?[*]f16;
extern fn msv_ane_mlp_eval(m: *BankProgram, procedure: u32, err: [*]u8, err_len: usize) c_int;
extern fn msv_ane_mlp_compile_seconds(m: *const BankProgram) f64;
extern fn msv_ane_mlp_cache_hit(m: *const BankProgram) c_int;
extern fn IOSurfaceGetBaseAddress(s: *Surface) ?*anyopaque;
extern fn IOSurfaceGetAllocSize(s: *Surface) usize;
extern fn CFRelease(cf: *anyopaque) void;

const C = ?*anyopaque;

fn programCreate(_: C, name: [*:0]const u8, text: [*:0]const u8, weights: ?*anyopaque, weight_bytes: usize, instances: u32, dies: ?[*]const u32, err: [*]u8, err_len: usize) callconv(.c) ?*Program {
    return msv_ane_program_create(name, text, weights, weight_bytes, instances, dies, err, err_len);
}
fn programFree(_: C, p: ?*Program) callconv(.c) void {
    msv_ane_program_free(p);
}
fn programProcedure(_: C, p: *Program, function: [*:0]const u8) callconv(.c) i32 {
    return msv_ane_program_procedure(p, function);
}
fn programIoCount(_: C, p: *Program, procedure: u32, output: bool) callconv(.c) u32 {
    return msv_ane_program_io_count(p, procedure, @intFromBool(output));
}
fn programIo(_: C, p: *Program, procedure: u32, output: bool, index: u32, out: *abi.AneIo) callconv(.c) bool {
    return msv_ane_program_io(p, procedure, @intFromBool(output), index, out) != 0;
}
fn programCompileSeconds(_: C, p: *const Program) callconv(.c) f64 {
    return msv_ane_program_compile_seconds(p);
}
fn programCacheHit(_: C, p: *const Program) callconv(.c) bool {
    return msv_ane_program_cache_hit(p);
}
fn surfaceCreate(_: C, bytes: usize, out: *abi.AneSurfaceView) callconv(.c) bool {
    const s = msv_ane_bridge_surface(bytes) orelse return false;
    const base: [*]u8 = @ptrCast(IOSurfaceGetBaseAddress(s) orelse {
        CFRelease(s);
        return false;
    });
    const n = IOSurfaceGetAllocSize(s);
    @memset(base[0..n], 0);
    out.* = .{ .surface = s, .base = base, .bytes = n };
    return true;
}
fn surfaceRelease(_: C, s: ?*Surface) callconv(.c) void {
    if (s) |x| CFRelease(x);
}
fn requestCreate(_: C, p: *Program, instance: u32, procedure: u32, inputs: [*]const ?*Surface, n_in: u32, outputs: [*]const ?*Surface, n_out: u32, err: [*]u8, err_len: usize) callconv(.c) ?*Request {
    return msv_ane_request_create(p, instance, procedure, inputs, n_in, outputs, n_out, err, err_len);
}
fn requestEval(_: C, r: *Request, err: [*]u8, err_len: usize) callconv(.c) bool {
    return msv_ane_request_eval(r, err, err_len) == 1;
}
fn requestFree(_: C, r: ?*Request) callconv(.c) void {
    msv_ane_request_free(r);
}
fn planeCreate(_: C, bytes: usize) callconv(.c) ?*Plane {
    return msv_ane_plane_create(bytes);
}
fn planeFree(_: C, p: ?*Plane) callconv(.c) void {
    msv_ane_plane_free(p);
}
fn planeBase(_: C, p: *Plane) callconv(.c) ?[*]f16 {
    return msv_ane_plane_base(p);
}
fn bankCreate(_: C) callconv(.c) ?*Bank {
    return msv_ane_bank_create();
}
fn bankFree(_: C, b: ?*Bank) callconv(.c) void {
    msv_ane_bank_free(b);
}
fn bankAddSwiglu(_: C, b: *Bank, hidden: u32, ffn: u32, rows: u32, gate_q: [*]const i8, gate_s: [*]const f32, up_q: [*]const i8, up_s: [*]const f32, down_q: [*]const i8, down_s: [*]const f32, limit: f32, err: [*]u8, err_len: usize) callconv(.c) i32 {
    return msv_ane_bank_add_mlp(b, hidden, ffn, rows, gate_q, gate_s, up_q, up_s, down_q, down_s, limit, err, err_len);
}
fn bankAddGdn(_: C, b: *Bank, hidden: u32, qkv_out: u32, z_out: u32, rows: u32, qkv_q: [*]const i8, qkv_s: [*]const f32, z_q: [*]const i8, z_s: [*]const f32, err: [*]u8, err_len: usize) callconv(.c) i32 {
    return msv_ane_bank_add_gdn(b, hidden, qkv_out, z_out, rows, qkv_q, qkv_s, z_q, z_s, err, err_len);
}
fn bankFinish(_: C, b: *Bank, name: [*:0]const u8, die: i32, input: ?*Plane, output: ?*Plane, err: [*]u8, err_len: usize) callconv(.c) ?*BankProgram {
    return msv_ane_bank_finish(b, name, die, input, output, err, err_len);
}
fn bankProgramFree(_: C, m: ?*BankProgram) callconv(.c) void {
    msv_ane_mlp_free(m);
}
fn bankProgramInput(_: C, m: *BankProgram) callconv(.c) ?[*]f16 {
    return msv_ane_mlp_input(m);
}
fn bankProgramOutput(_: C, m: *BankProgram) callconv(.c) ?[*]f16 {
    return msv_ane_mlp_output(m);
}
fn bankProgramEval(_: C, m: *BankProgram, procedure: u32, err: [*]u8, err_len: usize) callconv(.c) bool {
    return msv_ane_mlp_eval(m, procedure, err, err_len) == 1;
}
fn bankProgramCompileSeconds(_: C, m: *const BankProgram) callconv(.c) f64 {
    return msv_ane_mlp_compile_seconds(m);
}
fn bankProgramCacheHit(_: C, m: *const BankProgram) callconv(.c) bool {
    return msv_ane_mlp_cache_hit(m) != 0;
}
fn resident(_: C) callconv(.c) u32 {
    return msv_ane_resident();
}

const table_value: abi.HostAne = .{
    .ctx = null,
    .program_create = programCreate,
    .program_free = programFree,
    .program_procedure = programProcedure,
    .program_io_count = programIoCount,
    .program_io = programIo,
    .program_compile_seconds = programCompileSeconds,
    .program_cache_hit = programCacheHit,
    .surface_create = surfaceCreate,
    .surface_release = surfaceRelease,
    .request_create = requestCreate,
    .request_eval = requestEval,
    .request_free = requestFree,
    .plane_create = planeCreate,
    .plane_free = planeFree,
    .plane_base = planeBase,
    .bank_create = bankCreate,
    .bank_free = bankFree,
    .bank_add_swiglu = bankAddSwiglu,
    .bank_add_gdn = bankAddGdn,
    .bank_finish = bankFinish,
    .bank_program_free = bankProgramFree,
    .bank_program_input = bankProgramInput,
    .bank_program_output = bankProgramOutput,
    .bank_program_eval = bankProgramEval,
    .bank_program_compile_seconds = bankProgramCompileSeconds,
    .bank_program_cache_hit = bankProgramCacheHit,
    .resident = resident,
};

/// The table, or null where there is no Neural Engine to reach.
pub fn table() ?*const abi.HostAne {
    if (comptime !builtin.os.tag.isDarwin()) return null;
    return if (msv_ane_bridge_available() != 0) &table_value else null;
}

// ── Tests: the table end to end on this machine's ANE (skipped without one) ──

const testing = std.testing;
const mil = @import("../sdk/ane/mil.zig");

/// The table where a Neural Engine runs programs: a VM (GitHub's macOS runners) loads the bridge but has no ANE.
fn testTable() ?*const abi.HostAne {
    return if (@import("../mlx/ffi.zig").virtualGpu()) null else table();
}
const env = @import("../env.zig");

/// The compile cache under a fresh directory, so a test neither reads nor fills the user's.
fn testCache(buf: []u8) ![:0]const u8 {
    const root = try std.fmt.bufPrintSentinel(buf, "/tmp/msv-ane-plugin-{d}", .{std.c.getpid()}, 0);
    _ = std.c.mkdir(root, 0o755);
    env.set(.MLX_SERVE_ANE_CACHE_DIR, root);
    return root;
}

fn errText(err: []const u8) []const u8 {
    return std.mem.sliceTo(err, 0);
}

/// A Mac without a Neural Engine (CI's virtual one) loads the framework, so `table()` answers, then refuses every
/// compile with this.
fn noAneHardware(err: []const u8) bool {
    return std.mem.indexOf(u8, errText(err), "ANE hardware not available") != null;
}

test "plugin ane: a raw MIL program compiles, binds by name, evaluates on the ANE, and counts as resident until freed" {
    const ane = testTable() orelse return error.SkipZigTest;
    var cb: [128]u8 = undefined;
    const root = try testCache(&cb);
    defer {
        env.unset(.MLX_SERVE_ANE_CACHE_DIR);
        std.Io.Dir.cwd().deleteTree(testing.io, root) catch {};
    }
    var blob = try mil.Blob.init(testing.allocator, 0);
    defer blob.deinit();
    var g = try mil.Builder.init(testing.allocator, &blob);
    defer g.deinit();
    var in: [1]mil.Var = undefined;
    try g.begin("main", &.{.{ .name = "x", .shape = mil.Shape.of(&.{ 1, 64, 1, 32 }) }}, &in);
    const y = try g.named(try g.addS(try g.mulS(in[0], 2.0), 1.0), "y");
    try g.end(&.{y});
    const text = try g.finish();
    const bytes = blob.bytes.items;
    const weights = std.c.malloc(@max(bytes.len, 1)) orelse return error.OutOfMemory;
    @memcpy(@as([*]u8, @ptrCast(weights))[0..bytes.len], bytes);

    var err: [512]u8 = @splat(0);
    const before = ane.resident(ane.ctx);
    const p = ane.program_create(ane.ctx, "plugin-ane-test-affine", text.ptr, weights, bytes.len, 1, null, &err, err.len) orelse {
        if (noAneHardware(&err)) return error.SkipZigTest;
        std.debug.print("program_create: {s}\n", .{errText(&err)});
        return error.TestUnexpectedResult;
    };
    try testing.expectEqual(before + 1, ane.resident(ane.ctx));
    const proc: u32 = @intCast(ane.program_procedure(ane.ctx, p, "main"));
    try testing.expectEqual(@as(i32, -1), ane.program_procedure(ane.ctx, p, "absent"));
    try testing.expectEqual(@as(u32, 1), ane.program_io_count(ane.ctx, p, proc, false));
    var io_in: abi.AneIo = undefined;
    var io_out: abi.AneIo = undefined;
    try testing.expect(ane.program_io(ane.ctx, p, proc, false, 0, &io_in));
    try testing.expect(ane.program_io(ane.ctx, p, proc, true, 0, &io_out));
    try testing.expect(!ane.program_io(ane.ctx, p, proc, false, 1, &io_in));
    try testing.expectEqualStrings("y", std.mem.sliceTo(&io_out.name, 0));

    var sx: abi.AneSurfaceView = .{};
    var sy: abi.AneSurfaceView = .{};
    try testing.expect(ane.surface_create(ane.ctx, @intCast(io_in.batch_stride), &sx));
    defer ane.surface_release(ane.ctx, sx.surface);
    try testing.expect(ane.surface_create(ane.ctx, @intCast(io_out.batch_stride), &sy));
    defer ane.surface_release(ane.ctx, sy.surface);
    // x[c][w] = c + w / 32 at each channel's row (rows padded to `row_stride`).
    for (0..64) |c| {
        const row: [*]f16 = @ptrCast(@alignCast(sx.base.? + c * io_in.plane_stride));
        for (0..32) |w| row[w] = @floatCast(@as(f32, @floatFromInt(c)) + @as(f32, @floatFromInt(w)) / 32.0);
    }
    const ins = [_]?*abi.AneSurface{sx.surface};
    const outs = [_]?*abi.AneSurface{sy.surface};
    const r = ane.request_create(ane.ctx, p, 0, proc, &ins, 1, &outs, 1, &err, err.len) orelse {
        std.debug.print("request_create: {s}\n", .{errText(&err)});
        return error.TestUnexpectedResult;
    };
    defer ane.request_free(ane.ctx, r);
    try testing.expect(ane.request_eval(ane.ctx, r, &err, err.len));
    for (0..64) |c| {
        const row: [*]const f16 = @ptrCast(@alignCast(sy.base.? + c * io_out.plane_stride));
        for (0..32) |w| {
            const x = @as(f32, @floatFromInt(c)) + @as(f32, @floatFromInt(w)) / 32.0;
            try testing.expectApproxEqAbs(2 * x + 1, @as(f32, @floatCast(row[w])), 0.07);
        }
    }
    ane.program_free(ane.ctx, p);
    try testing.expectEqual(before, ane.resident(ane.ctx));
}

test "plugin ane: a bad program is refused with its reason, and nothing stays resident" {
    const ane = testTable() orelse return error.SkipZigTest;
    var cb: [128]u8 = undefined;
    const root = try testCache(&cb);
    defer {
        env.unset(.MLX_SERVE_ANE_CACHE_DIR);
        std.Io.Dir.cwd().deleteTree(testing.io, root) catch {};
    }
    var err: [512]u8 = @splat(0);
    const before = ane.resident(ane.ctx);
    try testing.expect(ane.program_create(ane.ctx, "plugin-ane-test-bad", "program(1.3) { not mil }", null, 0, 1, null, &err, err.len) == null);
    try testing.expect(errText(&err).len > 0);
    try testing.expectEqual(before, ane.resident(ane.ctx));
}

/// Row `r` of the int8 SwiGLU the bank builds, `x` `[H]` into `y` `[H]`: gate <= limit, |up| <= limit when limit > 0.
fn refSwiglu(h: usize, f: usize, gq: []const i8, gs: []const f32, uq: []const i8, us: []const f32, dq: []const i8, ds: []const f32, limit: f32, x: []const f32, y: []f32, act: []f32) void {
    for (0..f) |j| {
        var g: f32 = 0;
        var u: f32 = 0;
        for (0..h) |k| {
            g += @as(f32, @floatFromInt(gq[j * h + k])) * x[k];
            u += @as(f32, @floatFromInt(uq[j * h + k])) * x[k];
        }
        g *= gs[j];
        u *= us[j];
        if (limit > 0) {
            g = @min(g, limit);
            u = std.math.clamp(u, -limit, limit);
        }
        act[j] = g / (1 + @exp(-g)) * u;
    }
    for (0..h) |n| {
        var acc: f32 = 0;
        for (0..f) |j| acc += @as(f32, @floatFromInt(dq[n * f + j])) * act[j];
        y[n] = acc * ds[n];
    }
}

test "plugin ane: a SwiGLU bank gives the int8 reference, clamped by its limit and not without one" {
    const ane = testTable() orelse return error.SkipZigTest;
    var cb: [128]u8 = undefined;
    const root = try testCache(&cb);
    defer {
        env.unset(.MLX_SERVE_ANE_CACHE_DIR);
        std.Io.Dir.cwd().deleteTree(testing.io, root) catch {};
    }
    const H = 128;
    const F = 256;
    const R = 64;
    const a = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const gq = try a.alloc(i8, F * H);
    defer a.free(gq);
    const uq = try a.alloc(i8, F * H);
    defer a.free(uq);
    const dq = try a.alloc(i8, H * F);
    defer a.free(dq);
    for (gq) |*v| v.* = rnd.intRangeAtMost(i8, -127, 127);
    for (uq) |*v| v.* = rnd.intRangeAtMost(i8, -127, 127);
    for (dq) |*v| v.* = rnd.intRangeAtMost(i8, -127, 127);
    const gs: [F]f32 = @splat(0.004);
    const us: [F]f32 = @splat(0.004);
    const ds: [H]f32 = @splat(0.0005);
    // Two procedures over the same weights: limit 0 (unclamped) and a limit the pre-activations exceed.
    const limit: f32 = 1.5;
    var err: [512]u8 = @splat(0);
    const b = ane.bank_create(ane.ctx) orelse return error.TestUnexpectedResult;
    const p0 = ane.bank_add_swiglu(ane.ctx, b, H, F, R, gq.ptr, &gs, uq.ptr, &us, dq.ptr, &ds, 0, &err, err.len);
    const p1 = ane.bank_add_swiglu(ane.ctx, b, H, F, R, gq.ptr, &gs, uq.ptr, &us, dq.ptr, &ds, limit, &err, err.len);
    if (p0 < 0 or p1 < 0) {
        ane.bank_free(ane.ctx, b);
        std.debug.print("bank_add_swiglu: {s}\n", .{errText(&err)});
        return error.TestUnexpectedResult;
    }
    const m = ane.bank_finish(ane.ctx, b, "plugin-ane-test-swiglu", 0, null, null, &err, err.len) orelse {
        if (noAneHardware(&err)) return error.SkipZigTest;
        std.debug.print("bank_finish: {s}\n", .{errText(&err)});
        return error.TestUnexpectedResult;
    };
    defer ane.bank_program_free(ane.ctx, m);
    const in = ane.bank_program_input(ane.ctx, m).?;
    const out = ane.bank_program_output(ane.ctx, m).?;
    // The input plane is channel-major `[H][R]`.
    var x: [R][H]f32 = undefined;
    for (0..R) |r| for (0..H) |k| {
        x[r][k] = @floatCast(rnd.floatNorm(f32) * 2);
        in[k * R + r] = @floatCast(x[r][k]);
    };
    var y: [H]f32 = undefined;
    var act: [F]f32 = undefined;
    inline for (.{ .{ p0, @as(f32, 0) }, .{ p1, limit } }) |pl| {
        try testing.expect(ane.bank_program_eval(ane.ctx, m, @intCast(pl[0]), &err, err.len));
        var dot: f64 = 0;
        var na: f64 = 0;
        var nr: f64 = 0;
        var clamped_differs = false;
        for (0..R) |r| {
            refSwiglu(H, F, gq, &gs, uq, &us, dq, &ds, pl[1], &x[r], &y, &act);
            var y_un: [H]f32 = undefined;
            if (pl[1] > 0) {
                refSwiglu(H, F, gq, &gs, uq, &us, dq, &ds, 0, &x[r], &y_un, &act);
            }
            for (0..H) |n| {
                const v: f64 = @floatCast(out[n * R + r]);
                dot += v * y[n];
                na += v * v;
                nr += @as(f64, y[n]) * y[n];
                if (pl[1] > 0 and @abs(y_un[n] - y[n]) > 1e-3) clamped_differs = true;
            }
        }
        const cos = dot / @sqrt(na * nr);
        try testing.expect(cos > 0.999);
        // The limit binds on these inputs, so the clamped procedure is not the unclamped one.
        if (pl[1] > 0) try testing.expect(clamped_differs);
    }
}
