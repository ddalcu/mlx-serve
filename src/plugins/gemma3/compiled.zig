//! The Gemma decoders' compiled activations: the tanh GELU, GeGLU and the logit softcap as shapeless `mlx_compile`
//! closures over the same op chains the engine compiles, and the GELU lookup tables the fused gate kernels read (the
//! model's own, built from that GELU: the engine's process tables hold the same bits). Compiled only where a GPU backend exists, as the engine does.
//! copied_from: engines/mlx/compiled.zig (`compileGelu`, `compileGeglu`, `compileSoftcap` and their callbacks),
//! sdk/blocks/mlp/activations.zig (`geluUncompiled`, `geluTanhTable`).

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const fused_act = @import("../../sdk/blocks/mlp/fused_activations.zig");
const log = @import("../../sdk/util/log.zig");

const Array = mlx.mlx_array;
const Closure = mlx.mlx_closure;

fn bf16(v: f32, s: mlx.mlx_stream) Array {
    const f = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&out, f, .bfloat16, s);
    return out;
}

/// The constants the closures read and the stream they trace on; owned by the model (its address is the payload).
pub const Compiled = struct {
    s: mlx.mlx_stream,
    three: Array = .{},
    inner: Array = .{},
    coeff: Array = .{},
    one: Array = .{},
    half: Array = .{},
    /// bf16 0-d softcap; null = none.
    cap: Array = .{},
    gelu: Closure = .{ .ctx = null },
    geglu: Closure = .{ .ctx = null },
    softcap: Closure = .{ .ctx = null },
    /// bf16 and f16 GELU tables, built at first use; `declined`: that dtype's table would come out in another.
    tables: [2]Array = .{ .{}, .{} },
    declined: [2]bool = .{ false, false },

    /// The constants and, with a GPU backend, the closures (`softcap > 0` adds that one). `c` must not move after.
    pub fn init(c: *Compiled, s: mlx.mlx_stream, softcap: f32) void {
        c.* = .{
            .s = s,
            .three = bf16(3.0, s),
            .inner = bf16(0.044715, s),
            .coeff = bf16(0.7978845608028654, s),
            .one = bf16(1.0, s),
            .half = bf16(0.5, s),
            .cap = if (softcap > 0) bf16(softcap, s) else .{},
        };
        if (mlx.noGpuBackend()) return;
        c.gelu = compile(c, &geluCallback);
        c.geglu = compile(c, &gegluCallback);
        if (softcap > 0) c.softcap = compile(c, &softcapCallback);
        log.info("[gemma] GELU + GeGLU{s} compiled\n", .{if (softcap > 0) " + softcap" else ""});
    }

    pub fn deinit(c: *Compiled) void {
        for ([_]Closure{ c.gelu, c.geglu, c.softcap }) |f| if (f.ctx != null) {
            _ = mlx.mlx_closure_free(f);
        };
        for ([_]Array{ c.three, c.inner, c.coeff, c.one, c.half, c.cap, c.tables[0], c.tables[1] }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
    }

    /// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))` with bf16 constants, in the engine's op order (caller owns).
    pub fn geluUncompiled(c: *const Compiled, x: Array) !Array {
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_power(&x3, x, c.three, c.s));
        var in = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(in);
        try mlx.check(mlx.mlx_multiply(&in, c.inner, x3, c.s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, x, in, c.s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, c.coeff, sum, c.s));
        var t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t);
        try mlx.check(mlx.mlx_tanh(&t, scaled, c.s));
        var one_plus = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(one_plus);
        try mlx.check(mlx.mlx_add(&one_plus, c.one, t, c.s));
        var x_times = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_times);
        try mlx.check(mlx.mlx_multiply(&x_times, x, one_plus, c.s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&out, x_times, c.half, c.s));
        return out;
    }

    /// The GELU through its closure when compiled, else the op chain (caller owns).
    pub fn geluOf(c: *const Compiled, x: Array) !Array {
        if (c.gelu.ctx != null) return apply(c.gelu, &.{x});
        return c.geluUncompiled(x);
    }

    /// The GELU table for `dt` (bf16 or f16), built once from this GELU; null where the fused gates do not serve
    /// (they are off, or the table would come out in another dtype). Borrowed until `deinit`.
    pub fn geluTable(cc: *const Compiled, dt: mlx.mlx_dtype) !Array {
        if (!fused_act.swigluFusedEnabled()) return .{};
        const c = @constCast(cc);
        const slot: usize = if (dt == .bfloat16) 0 else 1;
        if (c.tables[slot].ctx != null) return c.tables[slot];
        if (c.declined[slot]) return .{};
        const vals = try fused_act.all16BitPatterns(c.s, dt, std.heap.c_allocator);
        defer _ = mlx.mlx_array_free(vals);
        const table = try c.geluOf(vals);
        if (mlx.mlx_array_dtype(table) != dt) {
            _ = mlx.mlx_array_free(table);
            c.declined[slot] = true;
            return .{};
        }
        c.tables[slot] = table;
        try mlx.check(mlx.mlx_array_eval(table));
        return table;
    }
};

fn compile(c: *Compiled, cb: *const fn (*mlx.mlx_vector_array, mlx.mlx_vector_array, ?*anyopaque) callconv(.c) c_int) Closure {
    const raw = mlx.mlx_closure_new_func_payload(cb, @ptrCast(c), null);
    defer _ = mlx.mlx_closure_free(raw);
    var out = Closure{ .ctx = null };
    if (mlx.mlx_compile(&out, raw, true) != 0) return .{ .ctx = null };
    return out;
}

/// A closure's only output (caller owns).
pub fn apply(f: Closure, args: []const Array) !Array {
    const in = mlx.mlx_vector_array_new_data(args.ptr, args.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var out = mlx.mlx_vector_array{ .ctx = null };
    defer _ = mlx.mlx_vector_array_free(out);
    try mlx.check(mlx.mlx_closure_apply(&out, f, in));
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_vector_array_get(&r, out, 0));
    return r;
}

fn payload(p: ?*anyopaque) *const Compiled {
    return @ptrCast(@alignCast(p.?));
}

fn input(v: mlx.mlx_vector_array, i: usize) ?Array {
    var a = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&a, v, i) != 0) {
        _ = mlx.mlx_array_free(a);
        return null;
    }
    return a;
}

fn output(res: *mlx.mlx_vector_array, a: Array) c_int {
    const arr = [_]Array{a};
    res.* = mlx.mlx_vector_array_new_data(&arr, 1);
    _ = mlx.mlx_array_free(a);
    return 0;
}

fn geluCallback(res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array, p: ?*anyopaque) callconv(.c) c_int {
    const x = input(in, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(x);
    return output(res, payload(p).geluUncompiled(x) catch return -1);
}

fn gegluCallback(res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array, p: ?*anyopaque) callconv(.c) c_int {
    const c = payload(p);
    const gate = input(in, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(gate);
    const up = input(in, 1) orelse return -1;
    defer _ = mlx.mlx_array_free(up);
    const act = c.geluUncompiled(gate) catch return -1;
    defer _ = mlx.mlx_array_free(act);
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_multiply(&out, act, up, c.s)) catch {
        _ = mlx.mlx_array_free(out);
        return -1;
    };
    return output(res, out);
}

fn softcapCallback(res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array, p: ?*anyopaque) callconv(.c) c_int {
    const c = payload(p);
    const x = input(in, 0) orelse return -1;
    defer _ = mlx.mlx_array_free(x);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    mlx.check(mlx.mlx_divide(&scaled, x, c.cap, c.s)) catch return -1;
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    mlx.check(mlx.mlx_tanh(&t, scaled, c.s)) catch return -1;
    var out = mlx.mlx_array_new();
    mlx.check(mlx.mlx_multiply(&out, t, c.cap, c.s)) catch {
        _ = mlx.mlx_array_free(out);
        return -1;
    };
    return output(res, out);
}
