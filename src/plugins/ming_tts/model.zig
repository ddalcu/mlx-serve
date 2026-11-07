//! Ming-3.0-tts-0.5b on mlx-c, after mlx-audio's `tts/models/dense` + `bailingmm` modules:
//!
//!   prompt ids -> Qwen2.5-0.5B (24 layers, KV cache, 512 positions)
//!     per step: last hidden z -+- stop head (softmax[1] > 0.5 after step 3)
//!                              +- CFM: 10 Euler steps of an 8-layer DiT with CFG 2
//!                                 over [t+c, history latent, x] -> one 64-d latent
//!     next input = aggregator(latent) (8-layer DiT stack, CLS row -> 896)
//!   latents (12.5 Hz) -> VAE decoder (x4 upsample, Qwen2 24 layers, sliding window 64, ISTFT head n_fft 3528 / hop 882)
//!     -> 44.1 kHz waveform
//!
//! Prompt audio for cloning goes through the VAE encoder (Qwen2 24 + 4 layer patch aggregator, posterior std =
//! softplus(scale) + 1e-4) and CAM++ (`campplus.zig`). Text frontend and prompt layout: `text.zig`.
//!
//! Every op mirrors the reference graph (same op order, same weak-scalar dtypes, `addmm` for biased linears). On the
//! GPU in bf16, the DiT stacks and the LM decode step run on fused kernels instead (`fused.zig`). An affine-quantized
//! pack has the same tensor names plus `.scales` and `.biases`, and always runs the op chain.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const config = @import("config.zig");
const mlx = @import("../../mlx/ffi.zig");
const fused = @import("fused.zig");
const embedding = @import("../../sdk/blocks/embedding.zig");
const istft_head = @import("../../sdk/media/istft.zig");

pub const Arr = mlx.mlx_array;
pub const S = mlx.mlx_stream;
pub const Dtype = mlx.mlx_dtype;

extern "c" fn mlx_addmm(res: *Arr, c: Arr, a: Arr, b: Arr, alpha: f32, beta: f32, s: S) c_int;
extern "c" fn mlx_power(res: *Arr, a: Arr, b: Arr, s: S) c_int;

pub const SAMPLE_RATE: u32 = 44100;
const LATENT: c_int = 64;
/// The geometry the fused LM decode kernels are written for.
const FUSED_LM_WIDTH: c_int = 896;
/// Layers of the VAE encoder's patch aggregator (the config does not carry them).
const ENC_AGG_LAYERS: usize = 4;
const HOP: usize = 882;
const N_FFT: usize = 3528;
const VAE_PATCH: usize = 4;
pub const SPK_DIM: usize = 192;
/// Vendor modeling_bailingmm.MAX_CACHE_LEN: prompt + generated latents.
pub const MAX_CACHE_LEN: usize = 512;
pub const FLOW_STEPS: usize = 10;
/// Runaway-decode cap: 2 s plus ~1 s per spoken unit.
pub const BASE_STEPS: usize = 25;
pub const STEPS_PER_UNIT: usize = 12;
/// Front-pad alignment for prompt audio: hop × VAE patch × DiT patch.
pub const PAD_ALIGN: usize = HOP * VAE_PATCH;

// ─────────────────────────────────────────────────────────────────────────
// Graph scope: every op result is parked here and freed together, so model
// code reads like the Python it mirrors. `own` copies a handle out of scope.
// ─────────────────────────────────────────────────────────────────────────

pub const G = struct {
    s: S,
    a: std.mem.Allocator,
    live: std.ArrayList(Arr) = .empty,
    /// Inside an mlx_compile trace: no nested compiled closures.
    in_trace: bool = false,

    pub fn init(a: std.mem.Allocator, s: S) G {
        return .{ .a = a, .s = s };
    }

    pub fn deinit(g: *G) void {
        for (g.live.items) |x| _ = mlx.mlx_array_free(x);
        g.live.deinit(g.a);
    }

    pub fn keep(g: *G, x: Arr) !Arr {
        g.live.append(g.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    fn ret(g: *G, rc: c_int, o: Arr) !Arr {
        if (rc != 0) {
            _ = mlx.mlx_array_free(o);
            try mlx.check(rc);
        }
        return g.keep(o);
    }

    pub fn add(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_add(&o, x, y, g.s), o);
    }
    pub fn sub(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_subtract(&o, x, y, g.s), o);
    }
    pub fn mul(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_multiply(&o, x, y, g.s), o);
    }
    pub fn div(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_divide(&o, x, y, g.s), o);
    }
    pub fn minimum(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_minimum(&o, x, y, g.s), o);
    }
    pub fn matmul(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_matmul(&o, x, y, g.s), o);
    }
    pub fn qmm(g: *G, x: Arr, w: Arr, q: Quant) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_quantized_matmul(&o, x, w, q.scales, q.biases, true, mlx.mlx_optional_int.some(@intCast(q.group_size)), mlx.mlx_optional_int.some(@intCast(q.bits)), "affine", g.s), o);
    }
    /// Rows `ids` of a quantized table, dequantized to `dt`.
    pub fn gatherRows(g: *G, w: Arr, q: Quant, ids: Arr, dt: Dtype) !Arr {
        return g.keep(try embedding.gatherQuantizedRows(g.s, w, q.scales, q.biases, ids, q.group_size, q.bits, .affine, .{ .ctx = null }, .{ .value = dt, .has_value = true }));
    }
    pub fn addmm(g: *G, c: Arr, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx_addmm(&o, c, x, y, 1.0, 1.0, g.s), o);
    }
    pub fn power(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx_power(&o, x, y, g.s), o);
    }
    pub fn exp(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_exp(&o, x, g.s), o);
    }
    pub fn cos(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_cos(&o, x, g.s), o);
    }
    pub fn sin(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_sin(&o, x, g.s), o);
    }
    pub fn tanh(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_tanh(&o, x, g.s), o);
    }
    pub fn abs(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_abs(&o, x, g.s), o);
    }
    pub fn sigmoid(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_sigmoid(&o, x, g.s), o);
    }
    pub fn logaddexp(g: *G, x: Arr, y: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_logaddexp(&o, x, y, g.s), o);
    }
    pub fn floor(g: *G, x: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_floor(&o, x, g.s), o);
    }
    pub fn softmax(g: *G, x: Arr, axis: c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_softmax_axis(&o, x, axis, false, g.s), o);
    }
    pub fn astype(g: *G, x: Arr, dt: Dtype) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_astype(&o, x, dt, g.s), o);
    }
    pub fn reshape(g: *G, x: Arr, shape: []const c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, g.s), o);
    }
    pub fn transpose(g: *G, x: Arr, axes: []const c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, g.s), o);
    }
    pub fn take(g: *G, x: Arr, idx: Arr, axis: c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_take_axis(&o, x, idx, axis, g.s), o);
    }
    pub fn rms(g: *G, x: Arr, w: Arr, eps: f32) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_fast_rms_norm(&o, x, w, eps, g.s), o);
    }
    pub fn clip(g: *G, x: Arr, lo: Arr, hi: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_clip(&o, x, lo, hi, g.s), o);
    }
    pub fn concat(g: *G, xs: []const Arr, axis: c_int) !Arr {
        const v = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(v);
        for (xs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_concatenate_axis(&o, v, axis, g.s), o);
    }
    pub fn stack(g: *G, xs: []const Arr, axis: c_int) !Arr {
        const v = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(v);
        for (xs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_stack_axis(&o, v, axis, g.s), o);
    }
    /// x[start:stop:step] along one axis, everything else whole.
    pub fn sliceAxis(g: *G, x: Arr, axis: usize, start: c_int, stop: c_int, step: c_int) !Arr {
        const sh = mlx.getShape(x);
        var st: [8]c_int = @splat(0);
        var sp: [8]c_int = undefined;
        var str: [8]c_int = @splat(1);
        for (sh, 0..) |d, i| sp[i] = d;
        st[axis] = start;
        sp[axis] = stop;
        str[axis] = step;
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_slice(&o, x, &st, sh.len, &sp, sh.len, &str, sh.len, g.s), o);
    }
    /// Python-style weak scalar: a constant in `dt`, so `x * 0.5` keeps x's dtype.
    pub fn scalar(g: *G, v: f32, dt: Dtype) !Arr {
        const f = mlx.mlx_array_new_float(v);
        defer _ = mlx.mlx_array_free(f);
        return g.astype(f, dt);
    }
    pub fn zeros(g: *G, shape: []const c_int, dt: Dtype) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, g.s), o);
    }
    pub fn fromF32(g: *G, data: []const f32, shape: []const c_int) !Arr {
        return g.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32));
    }
    pub fn fromI32(g: *G, data: []const i32, shape: []const c_int) !Arr {
        return g.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .int32));
    }
    pub fn normal(g: *G, shape: []const c_int, dt: Dtype) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_random_normal(&o, shape.ptr, shape.len, dt, 0, 1, .{ .ctx = null }, g.s), o);
    }
    pub fn split2(g: *G, x: Arr, axis: c_int) ![2]Arr {
        var v = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_split(&v, x, 2, axis, g.s));
        var out: [2]Arr = undefined;
        for (0..2) |i| {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&o, v, i));
            out[i] = try g.keep(o);
        }
        return out;
    }
    pub fn sdpa(g: *G, q: Arr, k: Arr, v: Arr, scale: f32, mode: [*:0]const u8, mask: Arr) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_fast_scaled_dot_product_attention(&o, q, k, v, scale, mode, mask, .{ .ctx = null }, false, g.s), o);
    }
    pub fn rope(g: *G, x: Arr, dims: c_int, traditional: bool, base: f32, offset: c_int) !Arr {
        var o = mlx.mlx_array_new();
        return g.ret(mlx.mlx_fast_rope(&o, x, dims, traditional, mlx.mlx_optional_float.some(base), 1.0, offset, .{ .ctx = null }, g.s), o);
    }
    pub fn silu(g: *G, x: Arr) !Arr {
        return g.mul(x, try g.sigmoid(x));
    }
};

pub fn own(x: Arr) !Arr {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&o, x));
    return o;
}

pub fn asyncEvalAll(xs: []const Arr) !void {
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    for (xs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
    try mlx.check(mlx.mlx_async_eval(v));
}

pub fn evalAll(xs: []const Arr) !void {
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    for (xs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
    try mlx.check(mlx.mlx_eval(v));
}

/// Evaluate `x` and copy it out as f32 (caller frees).
pub fn toF32(a: std.mem.Allocator, x: Arr, s: S) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_astype(&c, x, .float32, s));
    var cc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cc);
    try mlx.check(mlx.mlx_contiguous(&cc, c, false, s));
    try mlx.check(mlx.mlx_array_eval(cc));
    const n = mlx.mlx_array_size(cc);
    const p = mlx.mlx_array_data_float32(cc) orelse return error.NoData;
    return a.dupe(f32, p[0..n]);
}

fn itemF32(x: Arr, s: S) !f32 {
    if (mlx.mlx_array_dtype(x) == .float32) {
        try mlx.check(mlx.mlx_array_eval(x));
        var v: f32 = 0;
        try mlx.check(mlx.mlx_array_item_float32(&v, x));
        return v;
    }
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_astype(&c, x, .float32, s));
    try mlx.check(mlx.mlx_array_eval(c));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, c));
    return v;
}

// ─────────────────────────────────────────────────────────────────────────
// Weights
// ─────────────────────────────────────────────────────────────────────────

/// All tensors of one safetensors file, keyed by name. Owns the handles.
pub const Tensors = struct {
    map: std.StringHashMap(Arr),
    a: std.mem.Allocator,
    extra: std.ArrayList(Arr) = .empty, // derived views (transposed linears)
    /// The scheme `config.json` declares; null on a dense pack.
    scheme: ?config.Quant = null,

    pub fn load(a: std.mem.Allocator, path: []const u8, cast_to: ?Dtype, s: S) !Tensors {
        var self: Tensors = .{ .map = std.StringHashMap(Arr).init(a), .a = a };
        errdefer self.deinit();
        const pz = try a.dupeSentinel(u8, path, 0);
        defer a.free(pz);
        var tm = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(tm);
        var mm = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(mm);
        // Loading is a CPU-only primitive; casts and views then run on `s`.
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        try mlx.check(mlx.mlx_load_safetensors(&tm, &mm, pz.ptr, cpu));
        const it = mlx.mlx_map_string_to_array_iterator_new(tm);
        defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
        while (true) {
            var key: ?[*:0]const u8 = null;
            var v = mlx.mlx_array_new();
            if (mlx.mlx_map_string_to_array_iterator_next(&key, &v, it) != 0 or key == null) {
                _ = mlx.mlx_array_free(v);
                break;
            }
            if (cast_to) |dt| {
                if (mlx.mlx_array_dtype(v) != dt and mlx.mlx_array_dtype(v) != .int32) {
                    var c = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_astype(&c, v, dt, s));
                    _ = mlx.mlx_array_free(v);
                    v = c;
                }
            }
            const k = try a.dupe(u8, std.mem.span(key.?));
            try self.map.put(k, v);
        }
        return self;
    }

    /// The initializers of an ONNX model (`graph.initializer`), keyed by name,
    /// as f32/i64 arrays: what mlx-audio's ONNX-to-safetensors conversion
    /// writes, read directly so no converted file is needed. Handles raw_data
    /// and packed float_data/int64_data; external data is not supported.
    pub fn loadOnnx(a: std.mem.Allocator, io: std.Io, path: []const u8, cast_to: ?Dtype, s: S) !Tensors {
        const file = try std.Io.Dir.openFileAbsolute(io, path, .{});
        defer file.close(io);
        var rb: [64 * 1024]u8 = undefined;
        var rs = file.reader(io, &rb);
        const bytes = try rs.interface.allocRemaining(a, .limited(1 << 30));
        defer a.free(bytes);
        var self: Tensors = .{ .map = std.StringHashMap(Arr).init(a), .a = a };
        errdefer self.deinit();
        var model_pb = Pb{ .b = bytes };
        while (try model_pb.next()) |mf| {
            if (mf.num != 7 or mf.wire != 2) continue; // ModelProto.graph
            var graph = Pb{ .b = mf.bytes };
            while (try graph.next()) |gf| {
                if (gf.num != 5 or gf.wire != 2) continue; // GraphProto.initializer
                try self.addOnnxTensor(gf.bytes, cast_to, s);
            }
        }
        if (self.map.count() == 0) return error.NoOnnxInitializers;
        return self;
    }

    fn addOnnxTensor(self: *Tensors, tensor: []const u8, cast_to: ?Dtype, s: S) !void {
        var dims: [8]c_int = undefined;
        var nd: usize = 0;
        var dtype: u64 = 0;
        var name: []const u8 = "";
        var raw: ?[]const u8 = null;
        var packed_data: ?[]const u8 = null; // float_data (4) or int64_data (7), packed
        var pb = Pb{ .b = tensor };
        while (try pb.next()) |f| switch (f.num) {
            1 => if (f.wire == 0) {
                if (nd == dims.len) return error.OnnxRank;
                dims[nd] = @intCast(f.int);
                nd += 1;
            } else {
                var d = Pb{ .b = f.bytes };
                while (d.pos < d.b.len) {
                    if (nd == dims.len) return error.OnnxRank;
                    dims[nd] = @intCast(try d.varint());
                    nd += 1;
                }
            },
            2 => dtype = f.int,
            4, 7 => packed_data = f.bytes,
            8 => name = f.bytes,
            9 => raw = f.bytes,
            14 => return error.OnnxExternalData,
            else => {},
        };
        const dt: Dtype, const size: usize = switch (dtype) {
            1 => .{ .float32, 4 },
            7 => .{ .int64, 8 },
            else => return error.OnnxDtype,
        };
        var count: usize = 1;
        for (dims[0..nd]) |d| count *= @intCast(d);
        const data = raw orelse packed_data orelse return error.OnnxNoData;
        if (data.len != count * size) return error.OnnxDataSize;
        // Copy to aligned memory; mlx_array_new_data copies again into its own buffer.
        const buf = try self.a.alignedAlloc(u8, .@"8", data.len);
        defer self.a.free(buf);
        @memcpy(buf, data);
        var v = mlx.mlx_array_new_data(buf.ptr, &dims, @intCast(nd), dt);
        if (cast_to) |want| if (dt == .float32 and want != .float32) {
            var c = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&c, v, want, s));
            _ = mlx.mlx_array_free(v);
            v = c;
        };
        const k = try self.a.dupe(u8, name);
        errdefer self.a.free(k);
        try self.map.put(k, v);
    }

    pub fn deinit(self: *Tensors) void {
        var it = self.map.iterator();
        while (it.next()) |e| {
            _ = mlx.mlx_array_free(e.value_ptr.*);
            self.a.free(e.key_ptr.*);
        }
        self.map.deinit();
        for (self.extra.items) |x| _ = mlx.mlx_array_free(x);
        self.extra.deinit(self.a);
    }

    pub fn get(self: *const Tensors, comptime fmt: []const u8, args: anytype) !Arr {
        var buf: [256]u8 = undefined;
        const k = try std.fmt.bufPrint(&buf, fmt, args);
        return self.map.get(k) orelse {
            log.err("[ming_tts] missing weight {s}\n", .{k});
            return error.MissingWeight;
        };
    }

    pub fn has(self: *const Tensors, comptime fmt: []const u8, args: anytype) bool {
        var buf: [256]u8 = undefined;
        const k = std.fmt.bufPrint(&buf, fmt, args) catch return false;
        return self.map.contains(k);
    }

    /// The affine triple of `<prefix>`, or null for a dense weight (scales absence is per tensor). A tensor that
    /// disagrees with the declared scheme is an error here, not a wrong output later.
    pub fn quantOf(self: *const Tensors, comptime fmt: []const u8, args: anytype) !?Quant {
        if (!self.has(fmt ++ ".scales", args)) return null;
        const q = self.scheme orelse return error.UndeclaredQuantization;
        const w = mlx.getShape(try self.get(fmt ++ ".weight", args));
        const scales = try self.get(fmt ++ ".scales", args);
        const sc = mlx.getShape(scales);
        if (w.len != 2 or sc.len != 2 or w[0] != sc[0]) return error.QuantizedGeometry;
        if (@as(i64, w[1]) * 32 != @as(i64, sc[1]) * q.group_size * q.bits) return error.QuantizedGeometry;
        return .{ .scales = scales, .biases = try self.get(fmt ++ ".biases", args), .bits = q.bits, .group_size = q.group_size };
    }

    /// Linear `<prefix>.weight` [out, in] (+ optional `.bias`). A dense weight is
    /// kept as a transposed VIEW, like `x @ W.T` in mlx.nn.Linear.
    pub fn lin(self: *Tensors, s: S, comptime fmt: []const u8, args: anytype) !Lin {
        const w = try self.get(fmt ++ ".weight", args);
        const b: ?Arr = if (self.has(fmt ++ ".bias", args)) try self.get(fmt ++ ".bias", args) else null;
        if (try self.quantOf(fmt, args)) |q| return .{ .w = w, .wt = .{ .ctx = null }, .b = b, .q = q };
        var wt = mlx.mlx_array_new();
        const perm = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wt, w, &perm, 2, s));
        try self.extra.append(self.a, wt);
        return .{ .w = w, .wt = wt, .b = b };
    }
};

/// A quantized weight's affine extras. The weight itself holds the packed u32 codes.
pub const Quant = struct {
    scales: Arr,
    biases: Arr,
    bits: u32,
    group_size: u32,
};

pub const Lin = struct {
    w: Arr, // [out, in], for the fused kernels; packed codes when `q` is set
    wt: Arr, // null when `q` is set
    b: ?Arr,
    q: ?Quant = null,

    pub fn f(l: Lin, g: *G, x: Arr) !Arr {
        if (l.q) |q| {
            const y = try g.qmm(x, l.w, q);
            return if (l.b) |b| g.add(y, b) else y;
        }
        if (l.b) |b| return g.addmm(b, x, l.wt);
        return g.matmul(x, l.wt);
    }
};

// ─────────────────────────────────────────────────────────────────────────
// Qwen2 stack (LM backbone, VAE encoder/aggregator/decoder)
// ─────────────────────────────────────────────────────────────────────────

/// The fused kernels read dense weights and an additive bias on these linears.
fn denseBiased(ls: []const Lin) bool {
    for (ls) |l| if (l.q != null or l.b == null) return false;
    return true;
}

const QLayer = struct {
    ln1: Arr,
    ln2: Arr,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    gate: Lin,
    up: Lin,
    down: Lin,
};

const QStack = struct {
    layers: []QLayer,
    norm: Arr,
    heads: c_int,
    kv_heads: c_int,
    head_dim: c_int,
    eps: f32 = 1e-6,
    theta: f32,
    /// Sliding-window size (VAE backbones); null = plain causal (the LM).
    window: ?c_int = null,
    /// Fused decode kernels' view of the layers (the LM only).
    fl: ?[]fused.LmLayer = null,

    /// A quantized pack keeps `fl` null and decodes on the op chain.
    fn fuse(st: *QStack, a: std.mem.Allocator) !void {
        for (st.layers) |l| {
            if (!denseBiased(&.{ l.q, l.k, l.v })) return;
            for ([_]Lin{ l.o, l.gate, l.up, l.down }) |x| if (x.q != null) return;
        }
        const fl = try a.alloc(fused.LmLayer, st.layers.len);
        for (st.layers, fl) |l, *f| f.* = .{
            .ln1 = l.ln1,
            .ln2 = l.ln2,
            .wq = l.q.w,
            .wk = l.k.w,
            .wv = l.v.w,
            .bq = l.q.b.?,
            .bk = l.k.b.?,
            .bv = l.v.b.?,
            .wo = l.o.w,
            .wg = l.gate.w,
            .wu = l.up.w,
            .wd = l.down.w,
        };
        st.fl = fl;
    }

    /// One-token decode on the fused kernels (5 dispatches per layer).
    fn decodeFused(st: *const QStack, g: *G, x_in: Arr, c: *KvCache) !Arr {
        const pos = try g.keep(mlx.mlx_array_new_int(c.len));
        var x = x_in;
        for (st.fl.?, 0..) |*l, li| {
            const o = try fused.lmLayer(l, x, pos, c.k[li], c.v[li], @log2(st.theta), g.s);
            for ([_]Arr{ o.qkv, o.o, o.x1, o.hh, o.x2 }) |y| _ = try g.keep(y);
            _ = mlx.mlx_array_free(c.k[li]);
            _ = mlx.mlx_array_free(c.v[li]);
            c.k[li] = o.k;
            c.v[li] = o.v;
            x = o.x2;
        }
        c.len += 1;
        return g.rms(x, st.norm, st.eps);
    }

    fn load(t: *Tensors, a: std.mem.Allocator, s: S, comptime prefix: []const u8, d: config.Qwen, n: usize) !QStack {
        const layers = try a.alloc(QLayer, n);
        for (layers, 0..) |*l, i| {
            l.* = .{
                .ln1 = try t.get(prefix ++ ".layers.{d}.input_layernorm.weight", .{i}),
                .ln2 = try t.get(prefix ++ ".layers.{d}.post_attention_layernorm.weight", .{i}),
                .q = try t.lin(s, prefix ++ ".layers.{d}.self_attn.q_proj", .{i}),
                .k = try t.lin(s, prefix ++ ".layers.{d}.self_attn.k_proj", .{i}),
                .v = try t.lin(s, prefix ++ ".layers.{d}.self_attn.v_proj", .{i}),
                .o = try t.lin(s, prefix ++ ".layers.{d}.self_attn.o_proj", .{i}),
                .gate = try t.lin(s, prefix ++ ".layers.{d}.mlp.gate_proj", .{i}),
                .up = try t.lin(s, prefix ++ ".layers.{d}.mlp.up_proj", .{i}),
                .down = try t.lin(s, prefix ++ ".layers.{d}.mlp.down_proj", .{i}),
            };
        }
        return .{ .layers = layers, .norm = try t.get(prefix ++ ".norm.weight", .{}), .window = d.window, .heads = d.heads, .kv_heads = d.kv_heads, .head_dim = d.headDim(), .theta = d.theta };
    }

    /// `x` [B, L, 896] → normed hidden [B, L, 896]. With `cache`, keys/values
    /// are appended (the LM decode path).
    fn forward(st: *const QStack, g: *G, x_in: Arr, cache: ?*KvCache) !Arr {
        const sh = mlx.getShape(x_in);
        const B = sh[0];
        const L = sh[1];
        const offset: c_int = if (cache) |c| c.len else 0;
        if (cache) |c| if (st.fl != null and B == 1 and L == 1 and offset > 0 and offset < MAX_CACHE_LEN and
            fused.enabled() and mlx.streamIsGpu(g.s) and mlx.mlx_array_dtype(x_in) == .bfloat16 and
            mlx.mlx_array_dtype(st.layers[0].q.w) == .bfloat16 and st.heads == 14 and st.kv_heads == 2 and st.head_dim == 64 and
            mlx.getShape(x_in)[2] == FUSED_LM_WIDTH)
            return st.decodeFused(g, x_in, c);
        // Sliding mask (min(causal, window) additive, in the activation dtype),
        // or SDPA's causal mode for a multi-token LM prefill.
        var mask: Arr = .{ .ctx = null };
        var mode: [*:0]const u8 = "";
        if (st.window) |w| {
            mask = try slidingMask(g, L, offset + L, offset, w, mlx.mlx_array_dtype(x_in));
            mode = "array";
        } else if (L > 1) {
            mode = "causal";
        }
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(st.head_dim)));
        var x = x_in;
        for (st.layers, 0..) |*l, li| {
            const h = try g.rms(x, l.ln1, st.eps);
            var q = try g.transpose(try g.reshape(try l.q.f(g, h), &.{ B, L, st.heads, st.head_dim }), &.{ 0, 2, 1, 3 });
            var k = try g.transpose(try g.reshape(try l.k.f(g, h), &.{ B, L, st.kv_heads, st.head_dim }), &.{ 0, 2, 1, 3 });
            var v = try g.transpose(try g.reshape(try l.v.f(g, h), &.{ B, L, st.kv_heads, st.head_dim }), &.{ 0, 2, 1, 3 });
            q = try g.rope(q, st.head_dim, false, st.theta, offset);
            k = try g.rope(k, st.head_dim, false, st.theta, offset);
            if (cache) |c| {
                if (c.k[li].ctx != null) {
                    k = try g.concat(&.{ c.k[li], k }, 2);
                    v = try g.concat(&.{ c.v[li], v }, 2);
                    _ = mlx.mlx_array_free(c.k[li]);
                    _ = mlx.mlx_array_free(c.v[li]);
                }
                c.k[li] = try own(k);
                c.v[li] = try own(v);
            }
            var o = try g.sdpa(q, k, v, scale, mode, mask);
            o = try g.reshape(try g.transpose(o, &.{ 0, 2, 1, 3 }), &.{ B, L, st.heads * st.head_dim });
            const hres = try g.add(x, try l.o.f(g, o));
            const h2 = try g.rms(hres, l.ln2, st.eps);
            const m = try l.down.f(g, try compiledAct(g, .swiglu, &.{ try l.gate.f(g, h2), try l.up.f(g, h2) }));
            x = try g.add(hres, m);
        }
        if (cache) |c| c.len += L;
        return g.rms(x, st.norm, st.eps);
    }
};

/// mlx-audio `MingQwen2Attention._build_sliding_additive_mask` for mask="causal".
fn slidingMask(g: *G, q_len: c_int, kv_len: c_int, offset: c_int, window: c_int, dt: Dtype) !Arr {
    const n: usize = @intCast(q_len * kv_len);
    const buf = try g.a.alloc(f32, n);
    defer g.a.free(buf);
    const neg: f32 = -3.4028235e38;
    for (0..@intCast(q_len)) |i| {
        const qp: c_int = offset + @as(c_int, @intCast(i));
        for (0..@intCast(kv_len)) |j| {
            const kp: c_int = @intCast(j);
            const ok = kp > qp - window and qp >= kp;
            buf[i * @as(usize, @intCast(kv_len)) + j] = if (ok) 0 else neg;
        }
    }
    const m = try g.fromF32(buf, &.{ 1, 1, q_len, kv_len });
    // fromF32 does not copy until eval; materialize before `buf` is freed.
    const c = try g.astype(m, dt);
    try mlx.check(mlx.mlx_array_eval(c));
    return c;
}

pub const KvCache = struct {
    k: []Arr,
    v: []Arr,
    len: c_int = 0,
    a: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, n: usize) !KvCache {
        const k = try a.alloc(Arr, n);
        const v = try a.alloc(Arr, n);
        for (k, v) |*x, *y| {
            x.* = .{ .ctx = null };
            y.* = .{ .ctx = null };
        }
        return .{ .k = k, .v = v, .a = a };
    }

    pub fn deinit(c: *KvCache) void {
        for (c.k, c.v) |x, y| {
            if (x.ctx != null) _ = mlx.mlx_array_free(x);
            if (y.ctx != null) _ = mlx.mlx_array_free(y);
        }
        c.a.free(c.k);
        c.a.free(c.v);
    }
};

// ─────────────────────────────────────────────────────────────────────────
// DiT block stack (flow head + aggregator)
// ─────────────────────────────────────────────────────────────────────────

const DitBlock = struct {
    norm1: Arr,
    norm2: Arr,
    q: Lin,
    k: Lin,
    v: Lin,
    out: Lin,
    ff1: Lin,
    ff2: Lin,
};

/// x_transformers-style interleaved-pair rotary, tabulated per sequence
/// length from the checkpoint's `inv_freq` exactly as `RotaryEmbedding` does.
const Rotary = struct {
    inv_freq: Arr,
    /// cos/sin [1, 1, L, D/2] for the one length each stack runs at (DiT 3,
    /// aggregator 2), built once at load instead of in every compiled step.
    tab_len: c_int = 0,
    tab: [2]Arr = undefined,

    fn table(r: Rotary, g: *G, L: c_int) ![2]Arr {
        if (L == r.tab_len) return r.tab;
        var pa = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_arange(&pa, 0, @floatFromInt(L), 1, mlx.mlx_array_dtype(r.inv_freq), g.s));
        const pos = try g.keep(pa);
        // outer(pos, inv) = pos[:, None] * inv[None, :]
        const freqs = try g.mul(try g.reshape(pos, &.{ L, 1 }), try g.reshape(r.inv_freq, &.{ 1, -1 }));
        return .{ try g.reshape(try g.cos(freqs), &.{ 1, 1, L, -1 }), try g.reshape(try g.sin(freqs), &.{ 1, 1, L, -1 }) };
    }

    fn precompute(r: *Rotary, a: std.mem.Allocator, s: S, L: c_int) !void {
        var g = G.init(a, s);
        defer g.deinit();
        const t = try r.table(&g, L);
        try evalAll(&t);
        r.tab = .{ try own(t[0]), try own(t[1]) };
        r.tab_len = L;
    }

    fn apply(r: Rotary, g: *G, x: Arr) !Arr {
        // x: [B, H, L, D]
        const sh = mlx.getShape(x);
        const t = try r.table(g, sh[2]);
        const c = t[0];
        const s = t[1];
        const D: c_int = sh[3];
        const xe = try g.sliceAxis(x, 3, 0, D, 2);
        const xo = try g.sliceAxis(x, 3, 1, D, 2);
        const oe = try g.sub(try g.mul(xe, c), try g.mul(xo, s));
        const oo = try g.add(try g.mul(xe, s), try g.mul(xo, c));
        const st = try g.stack(&.{ oe, oo }, -1);
        return g.astype(try g.reshape(st, sh), mlx.mlx_array_dtype(x));
    }
};

const DitStack = struct {
    blocks: []DitBlock,
    rot: Rotary,
    hidden: c_int,
    heads: c_int,
    fb: []fused.Block,

    fn load(t: *Tensors, a: std.mem.Allocator, s: S, comptime prefix: []const u8, d: config.Dit, seq_len: c_int) !DitStack {
        const n: usize = @intCast(d.layers);
        const blocks = try a.alloc(DitBlock, n);
        for (blocks, 0..) |*b, i| {
            b.* = .{
                .norm1 = try t.get(prefix ++ ".blocks.{d}.norm1.weight", .{i}),
                .norm2 = try t.get(prefix ++ ".blocks.{d}.norm2.weight", .{i}),
                .q = try t.lin(s, prefix ++ ".blocks.{d}.attn.to_q", .{i}),
                .k = try t.lin(s, prefix ++ ".blocks.{d}.attn.to_k", .{i}),
                .v = try t.lin(s, prefix ++ ".blocks.{d}.attn.to_v", .{i}),
                .out = try t.lin(s, prefix ++ ".blocks.{d}.attn.to_out.0", .{i}),
                .ff1 = try t.lin(s, prefix ++ ".blocks.{d}.mlp.ff.0.0", .{i}),
                .ff2 = try t.lin(s, prefix ++ ".blocks.{d}.mlp.ff.2", .{i}),
            };
        }
        var rot: Rotary = .{ .inv_freq = try t.get(prefix ++ ".rotary_embed.inv_freq", .{}) };
        try rot.precompute(a, s, seq_len);
        try t.extra.appendSlice(t.a, &rot.tab);
        // A quantized pack leaves `fb` empty and runs the op chain.
        const dense = for (blocks) |b| {
            if (!denseBiased(&.{ b.q, b.k, b.v, b.out, b.ff1, b.ff2 })) break false;
        } else true;
        const fb = try a.alloc(fused.Block, if (dense) n else 0);
        for (blocks[0..fb.len], fb) |b, *f| f.* = .{
            .norm1 = b.norm1,
            .norm2 = b.norm2,
            .wq = b.q.w,
            .wk = b.k.w,
            .wv = b.v.w,
            .bq = b.q.b.?,
            .bk = b.k.b.?,
            .bv = b.v.b.?,
            .wo = b.out.w,
            .bo = b.out.b.?,
            .w1 = b.ff1.w,
            .b1 = b.ff1.b.?,
            .w2 = b.ff2.w,
            .b2 = b.ff2.b.?,
        };
        return .{ .blocks = blocks, .rot = rot, .fb = fb, .hidden = d.hidden, .heads = d.heads };
    }

    fn forward(st: *const DitStack, g: *G, x_in: Arr) !Arr {
        const sh = mlx.getShape(x_in);
        const B = sh[0];
        const L = sh[1];
        var x = x_in;
        if (st.fb.len > 0 and L == st.rot.tab_len and fused.eligible(x, g.s) and mlx.mlx_array_dtype(st.fb[0].wq) == .bfloat16) {
            for (st.fb) |*b| {
                const o = try fused.block(b, x, st.rot.tab[0], st.rot.tab[1], g.s);
                for ([_]Arr{ o.qkv, o.att, o.x1, o.hh, o.x2 }) |y| _ = try g.keep(y);
                x = o.x2;
            }
            return x;
        }
        for (st.blocks) |*b| {
            const h = try g.rms(x, b.norm1, 1e-6);
            var q = try g.transpose(try g.reshape(try b.q.f(g, h), &.{ B, L, st.heads, @divExact(st.hidden, st.heads) }), &.{ 0, 2, 1, 3 });
            var k = try g.transpose(try g.reshape(try b.k.f(g, h), &.{ B, L, st.heads, @divExact(st.hidden, st.heads) }), &.{ 0, 2, 1, 3 });
            const v = try g.transpose(try g.reshape(try b.v.f(g, h), &.{ B, L, st.heads, @divExact(st.hidden, st.heads) }), &.{ 0, 2, 1, 3 });
            q = try st.rot.apply(g, q);
            k = try st.rot.apply(g, k);
            var o = try g.sdpa(q, k, v, 1.0 / @sqrt(@as(f32, @floatFromInt(@divExact(st.hidden, st.heads)))), "", .{ .ctx = null });
            o = try g.reshape(try g.transpose(o, &.{ 0, 2, 1, 3 }), &.{ B, L, st.hidden });
            x = try g.add(x, try b.out.f(g, o));
            const h2 = try g.rms(x, b.norm2, 1e-6);
            x = try g.add(x, try b.ff2.f(g, try compiledAct(g, .gelu, &.{try b.ff1.f(g, h2)})));
        }
        return x;
    }
};

// ── mx.compile'd activations ──
// mlx.nn.gelu_approx / silu and mlx-audio's swiglu are `mx.compile(shapeless=True)`
// in Python: one fused kernel each. Same here, built once per process.

const Act = enum { gelu, silu, swiglu };
var act_cls: [3]?mlx.mlx_closure = .{ null, null, null };
/// Stream the traces run on: the loaded model's.
var act_stream: S = .{ .ctx = null };

/// A closure keeps the stream it was traced on, so a model loaded on another stream traces its own.
fn resetActs(s: S) void {
    for (&act_cls) |*c| if (c.*) |x| {
        _ = mlx.mlx_closure_free(x);
        c.* = null;
    };
    act_stream = s;
}

fn actTrace(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
    const which: Act = @fromBackingInt(@intCast(@intFromPtr(payload) - 1));
    var g = G.init(std.heap.c_allocator, act_stream);
    defer g.deinit();
    var x = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&x, input, 0) != 0) return -1;
    _ = g.keep(x) catch return -1;
    const out = switch (which) {
        .gelu => geluApprox(&g, x),
        .silu => g.silu(x),
        .swiglu => blk: {
            var up = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(&up, input, 1) != 0) return -1;
            _ = g.keep(up) catch return -1;
            break :blk g.mul(g.silu(x) catch return -1, up);
        },
    } catch return -1;
    res.* = mlx.mlx_vector_array_new_value(out);
    return 0;
}

fn compiledAct(g: *G, which: Act, inputs: []const Arr) !Arr {
    if (g.in_trace) return switch (which) {
        .gelu => geluApprox(g, inputs[0]),
        .silu => g.silu(inputs[0]),
        .swiglu => g.mul(try g.silu(inputs[0]), inputs[1]),
    };
    const i = @backingInt(which);
    if (act_cls[i] == null) {
        const raw = mlx.mlx_closure_new_func_payload(&actTrace, @ptrFromInt(i + 1), null);
        defer _ = mlx.mlx_closure_free(raw);
        var c = mlx.mlx_closure{ .ctx = null };
        try mlx.check(mlx.mlx_compile(&c, raw, true));
        act_cls[i] = c;
    }
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    for (inputs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
    var r = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(r);
    try mlx.check(mlx.mlx_closure_apply(&r, act_cls[i].?, v));
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&o, r, 0));
    return g.keep(o);
}

/// mlx.nn.gelu_approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3))).
fn geluApprox(g: *G, x: Arr) !Arr {
    const dt = mlx.mlx_array_dtype(x);
    const x3 = try g.power(x, try g.scalar(3, dt));
    const inner = try g.add(x, try g.mul(try g.scalar(0.044715, dt), x3));
    const t = try g.tanh(try g.mul(try g.scalar(@sqrt(2.0 / std.math.pi), dt), inner));
    return g.mul(try g.mul(try g.scalar(0.5, dt), x), try g.add(try g.scalar(1, dt), t));
}

// ─────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────

pub const Model = struct {
    a: std.mem.Allocator,
    s: S,
    t: Tensors,
    /// Activation dtype for prompt rounding and the Python `self.dtype`.
    dtype: Dtype,

    embed: Arr,
    embed_q: ?Quant,
    lm: QStack,
    stop_head: Lin,
    spk_head: Lin,
    // flow head
    dit: DitStack,
    t_mlp0: Lin,
    t_mlp2: Lin,
    x_emb: Lin,
    c_emb: Lin,
    dit_final_norm: Arr,
    dit_final: Lin,
    // aggregator
    agg: DitStack,
    agg_x_emb: Lin,
    agg_cls: Arr,
    agg_final_norm: Arr,
    agg_final: Lin,
    // VAE
    enc: QStack,
    enc_agg: QStack,
    enc_fc1: Lin,
    enc_fc2: Lin,
    enc_fc3: Lin,
    enc_cls: Arr,
    dec: QStack,
    dec_fc1: Lin,
    dec_head: Lin,
    window: Arr, // ISTFT window (checkpoint buffer), f32
    cfm_compiled: ?mlx.mlx_closure = null,
    cfm_params: [3]f32 = .{ 0, 0, 0 },
    agg_compiled: ?mlx.mlx_closure = null,
    /// SinusPositionEmbedding frequencies: exp(arange(128, f32) * -log(1e4)/127).
    t_freq: Arr,
    /// Timestep embeddings for the 10 Euler steps in `t_emb_dt`: they depend
    /// only on the fixed schedule, so they are built once, not every LM step.
    t_emb: [FLOW_STEPS]Arr,
    t_emb_dt: Dtype,
    /// Per Euler step in `t_emb_dt`: t[i+1] - t[i] and |dt|**0.5 (fused path).
    flow_dt: Arr,
    flow_sq: Arr,

    pub fn load(a: std.mem.Allocator, s: S, model_dir: []const u8, dims: config.Dims) !*Model {
        const path = try std.fmt.allocPrint(a, "{s}/model.safetensors", .{model_dir});
        defer a.free(path);
        const m = try a.create(Model);
        errdefer a.destroy(m);
        m.* = undefined;
        m.cfm_compiled = null;
        m.agg_compiled = null;
        m.a = a;
        m.s = s;
        resetActs(s);
        m.dtype = .bfloat16;
        m.t = try Tensors.load(a, path, null, s);
        errdefer m.t.deinit();
        m.t.scheme = dims.quant;
        const t = &m.t;
        m.embed = try t.get("model.model.embed_tokens.weight", .{});
        m.embed_q = try t.quantOf("model.model.embed_tokens", .{});
        m.lm = try QStack.load(t, a, s, "model.model", dims.lm, dims.lm.layers);
        try m.lm.fuse(a);
        m.stop_head = try t.lin(s, "stop_head", .{});
        m.spk_head = try t.lin(s, "spk_head", .{});
        const F = "flowloss.cfm.model";
        m.dit = try DitStack.load(t, a, s, F, dims.dit, 3); // [y, hist, x]
        m.t_mlp0 = try t.lin(s, F ++ ".t_embedder.time_mlp.0", .{});
        m.t_mlp2 = try t.lin(s, F ++ ".t_embedder.time_mlp.2", .{});
        m.x_emb = try t.lin(s, F ++ ".x_embedder", .{});
        m.c_emb = try t.lin(s, F ++ ".c_embedder.cond_embedder", .{});
        m.dit_final_norm = try t.get(F ++ ".final_layer.norm_final.weight", .{});
        m.dit_final = try t.lin(s, F ++ ".final_layer.linear", .{});
        const A = "linear_proj_audio";
        m.agg = try DitStack.load(t, a, s, A, dims.agg, 2); // [cls, x]
        m.agg_x_emb = try t.lin(s, A ++ ".x_embedder", .{});
        m.agg_cls = try t.get(A ++ ".word_embedder.weight", .{});
        if (try t.quantOf(A ++ ".word_embedder", .{})) |q| {
            var g = G.init(a, s);
            defer g.deinit();
            m.agg_cls = try own(try g.gatherRows(m.agg_cls, q, try g.fromI32(&.{0}, &.{1}), m.dtype));
            try t.extra.append(a, m.agg_cls);
        }
        m.agg_final_norm = try t.get(A ++ ".final_layer.norm_final.weight", .{});
        m.agg_final = try t.lin(s, A ++ ".final_layer.linear", .{});
        m.enc = try QStack.load(t, a, s, "audio.encoder.encoder", dims.enc, dims.enc.layers);
        m.enc_agg = try QStack.load(t, a, s, "audio.encoder.aggregator", dims.enc, ENC_AGG_LAYERS);
        m.enc_fc1 = try t.lin(s, "audio.encoder.fc1", .{});
        m.enc_fc2 = try t.lin(s, "audio.encoder.fc2", .{});
        m.enc_fc3 = try t.lin(s, "audio.encoder.fc3", .{});
        m.enc_cls = try t.get("audio.encoder.cls_embed", .{});
        m.dec = try QStack.load(t, a, s, "audio.decoder.decoder", dims.dec, dims.dec.layers);
        m.dec_fc1 = try t.lin(s, "audio.decoder.fc1", .{});
        m.dec_head = try t.lin(s, "audio.decoder.head.out", .{});
        {
            var g = G.init(a, s);
            defer g.deinit();
            var ar = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_arange(&ar, 0, 128, 1, .float32, s));
            const e: f32 = @floatCast(@log(@as(f64, 10000.0)) / 127.0);
            const f = try g.exp(try g.mul(try g.keep(ar), try g.scalar(-e, .float32)));
            m.t_freq = try own(f);
            try t.extra.append(a, m.t_freq);
        }
        {
            var g = G.init(a, s);
            defer g.deinit();
            m.t_emb_dt = m.dtype;
            const ts = try flowSchedule(&g, m.dtype);
            for (&m.t_emb, 1..) |*e, i| {
                e.* = try own(try m.timestepEmbed(&g, try g.reshape(try g.sliceAxis(ts, 0, @intCast(i - 1), @intCast(i), 1), &.{}), m.dtype));
                try t.extra.append(a, e.*);
            }
            try evalAll(&m.t_emb);
            const n: c_int = @intCast(FLOW_STEPS);
            m.flow_dt = try own(try g.sub(try g.sliceAxis(ts, 0, 1, n + 1, 1), try g.sliceAxis(ts, 0, 0, n, 1)));
            try t.extra.append(a, m.flow_dt);
            m.flow_sq = try own(try g.power(try g.abs(m.flow_dt), try g.scalar(0.5, m.dtype)));
            try t.extra.append(a, m.flow_sq);
            try evalAll(&.{ m.flow_dt, m.flow_sq });
        }
        var w32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&w32, try t.get("audio.decoder.head.istft.window", .{}), .float32, s));
        try t.extra.append(a, w32);
        m.window = w32;
        // Materialize everything now so the first request does not pay for it.
        var all: std.ArrayList(Arr) = .empty;
        defer all.deinit(a);
        var it = t.map.valueIterator();
        while (it.next()) |v| try all.append(a, v.*);
        try all.appendSlice(a, t.extra.items);
        try evalAll(all.items);
        return m;
    }

    pub fn deinit(m: *Model) void {
        const a = m.a;
        if (m.cfm_compiled) |c| _ = mlx.mlx_closure_free(c);
        if (m.agg_compiled) |c| _ = mlx.mlx_closure_free(c);
        for ([_][]QLayer{ m.lm.layers, m.enc.layers, m.enc_agg.layers, m.dec.layers }) |ls| a.free(ls);
        if (m.lm.fl) |fl| a.free(fl);
        a.free(m.dit.blocks);
        a.free(m.agg.blocks);
        a.free(m.dit.fb);
        a.free(m.agg.fb);
        m.t.deinit();
        a.destroy(m);
    }

    // ── LM ──

    pub fn embedIds(m: *Model, g: *G, ids: []const i32) !Arr {
        const idx = try g.fromI32(ids, &.{@intCast(ids.len)});
        const e = if (m.embed_q) |q| try g.gatherRows(m.embed, q, idx, m.dtype) else try g.take(m.embed, idx, 0);
        return g.reshape(e, &.{ 1, @intCast(ids.len), -1 });
    }

    pub fn lmForward(m: *Model, g: *G, embeds: Arr, cache: *KvCache) !Arr {
        return m.lm.forward(g, embeds, cache);
    }

    /// softmax(stop_head(z).astype(f32))[0, 0, 1]
    pub fn stopProb(m: *Model, g: *G, z: Arr) !Arr {
        const logits = try g.astype(try m.stop_head.f(g, z), .float32);
        const p = try g.softmax(logits, -1);
        return g.reshape(try g.sliceAxis(p, 2, 1, 2, 1), &.{});
    }

    // ── Aggregator: latent [B, 1, 64] → LM input rows [B, 1, 896] ──

    pub fn aggregateRaw(m: *Model, g: *G, latent: Arr) !Arr {
        const x = try m.agg_x_emb.f(g, latent);
        const B = mlx.getShape(x)[0];
        var cb = mlx.mlx_array_new();
        const csh = [_]c_int{ B, 1, m.agg.hidden };
        try mlx.check(mlx.mlx_broadcast_to(&cb, try g.reshape(m.agg_cls, &.{ 1, 1, m.agg.hidden }), &csh, 3, g.s));
        const cls_b = try g.keep(cb);
        var h = try g.concat(&.{ cls_b, x }, 1);
        h = try m.agg.forward(g, h);
        h = try m.agg_final.f(g, try g.rms(h, m.agg_final_norm, 1e-6));
        return g.sliceAxis(h, 1, 0, 1, 1);
    }

    // ── Flow head ──

    /// DiT.forward_with_cfg: x [1,1,64], t scalar (solver dtype), c [1,1,896],
    /// hist [1,1,64] → [2, 1, 64] (cond, uncond).
    pub fn ditCfg(m: *Model, g: *G, x1: Arr, t: Arr, c1: Arr, hist1: Arr) !Arr {
        return m.ditCfgEmb(g, x1, try m.timestepEmbed(g, t, mlx.mlx_array_dtype(x1)), c1, hist1);
    }

    /// TimestepEmbedder for scalar t (solver dtype) → [2, 1, 1024] in `xdt`.
    fn timestepEmbed(m: *Model, g: *G, t: Arr, xdt: Dtype) !Arr {
        var tt = mlx.mlx_array_new();
        const two = [_]c_int{2};
        try mlx.check(mlx.mlx_full(&tt, &two, 1, t, xdt, g.s));
        const tb = try g.keep(tt);
        // Sinusoid in f32 (x * 1000 rounds in t's dtype first).
        const ts = try g.mul(try g.scalar(1000.0, xdt), tb);
        const emb = try g.mul(try g.reshape(ts, &.{ 2, 1 }), try g.reshape(m.t_freq, &.{ 1, 128 }));
        const sincos = try g.astype(try g.concat(&.{ try g.sin(emb), try g.cos(emb) }, -1), xdt);
        var te = try m.t_mlp0.f(g, sincos);
        te = try m.t_mlp2.f(g, try compiledAct(g, .silu, &.{te}));
        return g.reshape(te, &.{ 2, 1, m.dit.hidden });
    }

    fn ditCfgEmb(m: *Model, g: *G, x1: Arr, t_embed: Arr, c1: Arr, hist1: Arr) !Arr {
        const x = try g.concat(&.{ x1, x1 }, 0);
        const hist = try g.concat(&.{ hist1, hist1 }, 0);
        const c = try g.concat(&.{ c1, try g.zeros(mlx.getShape(c1), mlx.mlx_array_dtype(c1)) }, 0);
        const x_now = try m.x_emb.f(g, x);
        const x_hist = try m.x_emb.f(g, hist);
        const c_embed = try m.c_emb.f(g, c);
        const y = try g.add(t_embed, c_embed);
        var h = try g.concat(&.{ y, x_hist, x_now }, 1);
        h = try m.dit.forward(g, h);
        h = try m.dit_final.f(g, try g.rms(h, m.dit_final_norm, 1e-6));
        const L = mlx.getShape(h)[1];
        return g.sliceAxis(h, 1, L - 1, L, 1);
    }

    /// Run `body(m, g, inputs)` through an mlx_compile'd closure cached in
    /// `slot` (shapes are fixed per call site, so each traces once).
    fn compiled(m: *Model, g: *G, slot: *?mlx.mlx_closure, comptime body: fn (*Model, *G, []const Arr) anyerror!Arr, inputs: []const Arr) !Arr {
        if (g.in_trace) return body(m, g, inputs);
        if (slot.* == null) {
            const Tr = struct {
                fn trace(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                    const mm: *Model = @ptrCast(@alignCast(payload.?));
                    var tg = G.init(std.heap.c_allocator, mm.s);
                    tg.in_trace = true;
                    defer tg.deinit();
                    var xs: [8]Arr = undefined;
                    const n = mlx.mlx_vector_array_size(input);
                    for (0..n) |i| {
                        xs[i] = mlx.mlx_array_new();
                        if (mlx.mlx_vector_array_get(&xs[i], input, i) != 0) return -1;
                        _ = tg.keep(xs[i]) catch return -1;
                    }
                    const out = body(mm, &tg, xs[0..n]) catch return -1;
                    res.* = mlx.mlx_vector_array_new_value(out);
                    return 0;
                }
            };
            const raw = mlx.mlx_closure_new_func_payload(&Tr.trace, m, null);
            defer _ = mlx.mlx_closure_free(raw);
            var c = mlx.mlx_closure{ .ctx = null };
            try mlx.check(mlx.mlx_compile(&c, raw, false));
            slot.* = c;
        }
        const v = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(v);
        for (inputs) |x| _ = mlx.mlx_vector_array_append_value(v, x);
        var r = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(r);
        try mlx.check(mlx.mlx_closure_apply(&r, slot.*.?, v));
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, r, 0));
        return g.keep(o);
    }

    fn cfmBody(m: *Model, g: *G, in: []const Arr) !Arr {
        return m.cfmSampleRaw(g, in[0], in[1], in[2], in[3], m.cfm_params[0], m.cfm_params[1], m.cfm_params[2]);
    }

    fn aggBody(m: *Model, g: *G, in: []const Arr) !Arr {
        return m.aggregateRaw(g, in[0]);
    }

    /// CFM.sample for one LM step, compiled whole (10 Euler steps × CFG DiT).
    pub fn cfmSample(m: *Model, g: *G, z: Arr, hist: Arr, noise: Arr, sde: Arr, cfg: f32, sigma: f32, temperature: f32) !Arr {
        const p = [3]f32{ cfg, sigma, temperature };
        if (m.cfm_compiled != null and !std.mem.eql(f32, &p, &m.cfm_params)) {
            _ = mlx.mlx_closure_free(m.cfm_compiled.?);
            m.cfm_compiled = null;
        }
        m.cfm_params = p;
        return m.compiled(g, &m.cfm_compiled, cfmBody, &.{ z, hist, noise, sde });
    }

    /// Aggregator for one latent [1, 1, 64] (compiled); batched prompt rows go uncompiled.
    pub fn aggregate(m: *Model, g: *G, latent: Arr) !Arr {
        if (mlx.getShape(latent)[0] != 1) return m.aggregateRaw(g, latent);
        return m.compiled(g, &m.agg_compiled, aggBody, &.{latent});
    }

    /// CFM.sample (ming3 sampling): Euler over the EPSS + sway schedule in the
    /// noise dtype. `noise` [1,1,64], `sde` [10,1,1,64].
    pub fn cfmSampleRaw(m: *Model, g: *G, z: Arr, hist: Arr, noise: Arr, sde: Arr, cfg: f32, sigma: f32, temperature: f32) !Arr {
        const dt = mlx.mlx_array_dtype(noise);
        const shift_k = sigma * std.math.sqrt(temperature);
        if (m.dit.fb.len > 0 and denseBiased(&.{ m.x_emb, m.dit_final }) and fused.enabled() and m.dit.hidden == 1024 and mlx.streamIsGpu(g.s) and dt == .bfloat16 and dt == m.t_emb_dt and mlx.mlx_array_dtype(z) == .bfloat16)
            return m.cfmSampleFused(g, z, hist, noise, sde, cfg, shift_k);
        const t = try flowSchedule(g, dt);
        var y = noise;
        for (1..FLOW_STEPS + 1) |i| {
            const t0 = try g.reshape(try g.sliceAxis(t, 0, @intCast(i - 1), @intCast(i), 1), &.{});
            const t1 = try g.reshape(try g.sliceAxis(t, 0, @intCast(i), @intCast(i + 1), 1), &.{});
            const dtt = try g.sub(t1, t0);
            const out = if (dt == m.t_emb_dt)
                try m.ditCfgEmb(g, y, m.t_emb[i - 1], z, hist)
            else
                try m.ditCfg(g, y, t0, z, hist);
            const pn = try g.split2(out, 0);
            const cfg_a = try g.scalar(cfg, mlx.mlx_array_dtype(out));
            var f0 = try g.add(pn[0], try g.mul(try g.sub(pn[0], pn[1]), cfg_a));
            f0 = try g.astype(f0, dt);
            y = try g.add(y, try g.mul(try g.astype(dtt, dt), f0));
            if (shift_k != 0) {
                const nz = try g.reshape(try g.sliceAxis(sde, 0, @intCast(i - 1), @intCast(i), 1), mlx.getShape(y));
                const sq = try g.power(try g.abs(dtt), try g.scalar(0.5, dt));
                y = try g.add(y, try g.mul(try g.mul(try g.scalar(shift_k, dt), sq), nz));
            }
        }
        return y;
    }

    /// cfmSampleRaw on the fused kernels: per Euler step one embed dispatch,
    /// 4 per DiT block, one for final layer + CFG + Euler. The cond embedding
    /// and x_embed(hist) are fixed across the 10 steps, so they run once.
    fn cfmSampleFused(m: *Model, g: *G, z: Arr, hist: Arr, noise: Arr, sde: Arr, cfg: f32, shift_k: f32) !Arr {
        const ce = try m.c_emb.f(g, try g.concat(&.{ z, try g.zeros(mlx.getShape(z), mlx.mlx_array_dtype(z)) }, 0));
        const xh = try m.x_emb.f(g, hist);
        var y = noise;
        for (0..FLOW_STEPS) |i| {
            var h = try g.keep(try fused.embed(y, m.x_emb.w, m.x_emb.b.?, m.t_emb[i], ce, xh, g.s));
            h = try m.dit.forward(g, h);
            y = try g.keep(try fused.final(h, m.dit_final_norm, m.dit_final.w, m.dit_final.b.?, y, m.flow_dt, m.flow_sq, sde, i, cfg, shift_k, g.s));
        }
        return y;
    }

    /// EPSS + sway schedule, [FLOW_STEPS + 1] in `dt`:
    /// t = (1/32) * array(sched, dt); t = t + (-1) * (cos(pi/2 * t) - 1 + t)
    fn flowSchedule(g: *G, dt: Dtype) !Arr {
        const sched = [_]f32{ 0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32 };
        comptime std.debug.assert(sched.len == FLOW_STEPS + 1);
        var t = try g.astype(try g.fromF32(&sched, &.{sched.len}), dt);
        t = try g.mul(try g.scalar(1.0 / 32.0, dt), t);
        const ct = try g.cos(try g.mul(try g.scalar(std.math.pi / 2.0, dt), t));
        const sw = try g.add(try g.sub(ct, try g.scalar(1, dt)), t);
        return g.add(t, try g.mul(try g.scalar(-1, dt), sw));
    }

    // ── VAE ──

    /// Encoder: waveform [1, N] f32 (N a multiple of 882) → h [1, T, 128].
    pub fn vaeEncodeH(m: *Model, g: *G, wav: Arr) !Arr {
        const HIDDEN = m.enc.heads * m.enc.head_dim;
        const N = mlx.getShape(wav)[1];
        const nf = @divFloor(N + @as(c_int, HOP) - 1, @as(c_int, HOP));
        var x = wav;
        const need = nf * @as(c_int, HOP);
        if (need > N) x = try g.concat(&.{ x, try g.zeros(&.{ 1, need - N }, .float32) }, 1);
        x = try g.reshape(x, &.{ 1, nf, @as(c_int, HOP) });
        x = try g.astype(x, mlx.mlx_array_dtype(if (m.enc_fc1.q) |q| q.scales else m.enc_fc1.w));
        x = try m.enc_fc2.f(g, try m.enc_fc1.f(g, x));
        x = try m.enc.forward(g, x, null);
        // pad_patch_insert_cls
        const T = mlx.getShape(x)[1];
        const r = @mod(T, @as(c_int, VAE_PATCH));
        if (r != 0) x = try g.concat(&.{ x, try g.zeros(&.{ 1, @as(c_int, VAE_PATCH) - r, HIDDEN }, mlx.mlx_array_dtype(x)) }, 1);
        x = try g.reshape(x, &.{ -1, @as(c_int, VAE_PATCH), HIDDEN });
        const np_ = mlx.getShape(x)[0];
        const cls = try g.astype(m.enc_cls, mlx.mlx_array_dtype(x));
        var cb = mlx.mlx_array_new();
        const csh = [_]c_int{ np_, 1, HIDDEN };
        try mlx.check(mlx.mlx_broadcast_to(&cb, cls, &csh, 3, g.s));
        x = try g.concat(&.{ x, try g.keep(cb) }, 1);
        x = try g.reshape(x, &.{ 1, -1, HIDDEN });
        x = try m.enc_agg.forward(g, x, null);
        x = try g.reshape(x, &.{ -1, @as(c_int, VAE_PATCH) + 1, HIDDEN });
        x = try g.reshape(try g.sliceAxis(x, 1, @as(c_int, VAE_PATCH), @as(c_int, VAE_PATCH) + 1, 1), &.{ 1, -1, HIDDEN });
        return m.enc_fc3.f(g, x);
    }

    /// Posterior sample: mean + (softplus(scale) + 1e-4) * noise.
    pub fn vaePosterior(g: *G, h: Arr, noise: Arr) !Arr {
        const ms = try g.split2(h, -1);
        const dt = mlx.mlx_array_dtype(ms[0]);
        const sp = try g.logaddexp(ms[1], try g.scalar(0, dt));
        const std_ = try g.add(sp, try g.scalar(1e-4, dt));
        return g.add(ms[0], try g.mul(std_, try g.astype(noise, dt)));
    }

    /// Decoder: latents [1, T, 64] → waveform samples (T * 4 * 882, f32, owned).
    pub fn vaeDecode(m: *Model, g: *G, latents: Arr) ![]f32 {
        var x = try m.dec_fc1.f(g, latents);
        x = try linearUpsample(g, x, VAE_PATCH);
        x = try m.dec.forward(g, x, null);
        // ISTFT head: [1, F, 3530] → log-magnitude and phase halves; "same" padding.
        const mp = try g.split2(try m.dec_head.f(g, x), 2);
        const mag = try g.minimum(try g.exp(mp[0]), try g.scalar(1e2, mlx.mlx_array_dtype(mp[0])));
        const ph = mp[1];
        const i_unit = try g.keep(mlx.mlx_array_new_complex(0, 1));
        const spec = try g.mul(mag, try g.add(try g.cos(ph), try g.mul(i_unit, try g.sin(ph))));
        const win = try toF32(g.a, m.window, g.s);
        defer g.a.free(win);
        return istft_head.istft(g.a, spec, N_FFT, HOP, win, (N_FFT - HOP) / 2, g.s);
    }
};

/// Decoder._linear_upsample_1d (align_corners=False), in f32 like the Python.
fn linearUpsample(g: *G, x_btc: Arr, scale: usize) !Arr {
    const x = try g.astype(try g.transpose(x_btc, &.{ 0, 2, 1 }), .float32);
    const in_t: usize = @intCast(mlx.getShape(x)[2]);
    const out_t = in_t * scale;
    const idx = try g.a.alloc(f32, out_t);
    defer g.a.free(idx);
    for (0..out_t) |i| idx[i] = @floatFromInt(i);
    const ar = try g.fromF32(idx, &.{@intCast(out_t)});
    const sc = try g.scalar(@floatFromInt(scale), .float32);
    const pos = try g.sub(try g.div(try g.add(ar, try g.scalar(0.5, .float32)), sc), try g.scalar(0.5, .float32));
    try mlx.check(mlx.mlx_array_eval(pos));
    const left_f = try g.floor(pos);
    const left = try g.astype(left_f, .int32);
    const right = try g.add(left, try g.astype(try g.scalar(1, .float32), .int32));
    const w_right = try g.sub(pos, try g.astype(left, .float32));
    const lo = try g.astype(try g.scalar(0, .float32), .int32);
    const hi = try g.astype(try g.scalar(@floatFromInt(in_t - 1), .float32), .int32);
    const lv = try g.take(x, try g.clip(left, lo, hi), -1);
    const rv = try g.take(x, try g.clip(right, lo, hi), -1);
    const one = try g.scalar(1, .float32);
    const wl = try g.reshape(try g.sub(one, w_right), &.{ 1, 1, -1 });
    const wr = try g.reshape(w_right, &.{ 1, 1, -1 });
    const y = try g.add(try g.mul(lv, wl), try g.mul(rv, wr));
    return g.transpose(y, &.{ 0, 2, 1 });
}

// ─────────────────────────────────────────────────────────────────────────
// AR + flow-matching loop (core.py `sample_latents`)
// ─────────────────────────────────────────────────────────────────────────

/// The flow head's Gaussian draws for one step: start noise and SDE noise, in the reference's call order and shapes.
fn drawNoise(g: *G, dt: Dtype) ![2]Arr {
    return .{ try g.normal(&.{ 1, 1, LATENT }, dt), try g.normal(&.{ @intCast(FLOW_STEPS), 1, 1, LATENT }, dt) };
}

pub const SampleParams = struct {
    max_steps: usize = 1000,
    cfg: f32 = 2.0,
    sigma: f32 = 0.25,
    temperature: f32 = 0.0,
};

/// embeds [1, L, 896] → latents [1, steps, 64] (owned).
pub fn sampleLatents(m: *Model, embeds_in: Arr, prompt_latent: ?Arr, p: SampleParams) !Arr {
    const a = m.a;
    const L: usize = @intCast(mlx.getShape(embeds_in)[1]);
    if (L >= MAX_CACHE_LEN) return error.PromptTooLong;
    const budget = MAX_CACHE_LEN - L; // patch_size 1
    const steps = @min(p.max_steps, budget);
    var cache = try KvCache.init(a, m.lm.layers.len);
    defer cache.deinit();
    var latents: std.ArrayList(Arr) = .empty;
    defer {
        for (latents.items) |x| _ = mlx.mlx_array_free(x);
        latents.deinit(a);
    }
    var history: Arr = .{ .ctx = null };
    defer if (history.ctx != null) {
        _ = mlx.mlx_array_free(history);
    };
    // Pipelined one step ahead: each iteration queues CFM for step k plus the
    // LM pass for k+1, then reads stop_k (queued one iteration earlier), so the
    // GPU never idles while the CPU encodes. The only waste is one LM pass after
    // the last step. Noise is drawn only for steps that run, so the RNG stream
    // and the output match the serial loop bit for bit.
    var z: Arr = undefined;
    var stop: Arr = undefined;
    var zdt: Dtype = undefined;
    {
        var g = G.init(a, m.s);
        defer g.deinit();
        const hidden = try m.lmForward(&g, embeds_in, &cache);
        const Lh = mlx.getShape(hidden)[1];
        const z0 = try g.sliceAxis(hidden, 1, Lh - 1, Lh, 1);
        zdt = mlx.mlx_array_dtype(z0);
        var h = try g.zeros(&.{ 1, 1, LATENT }, zdt);
        if (prompt_latent) |pl_in| {
            // history_patch_size 1: the prompt's last latent seeds the history.
            const pl = try g.astype(pl_in, zdt);
            const T = mlx.getShape(pl)[1];
            h = try g.sliceAxis(pl, 1, T - 1, T, 1);
        }
        history = try own(h);
        z = try own(z0);
        stop = try own(try m.stopProb(&g, z0));
        try asyncEvalAll(&.{ z, stop });
    }
    defer {
        _ = mlx.mlx_array_free(z);
        _ = mlx.mlx_array_free(stop);
    }
    for (0..steps) |step| {
        var g = G.init(a, m.s);
        defer g.deinit();
        const nz = try drawNoise(&g, zdt);
        const latent = try m.cfmSample(&g, z, history, nz[0], nz[1], p.cfg, p.sigma, p.temperature);
        var z_next: ?Arr = null;
        var stop_next: Arr = undefined;
        if (step + 1 < steps) {
            const hidden = try m.lmForward(&g, try m.aggregate(&g, latent), &cache);
            z_next = hidden; // [1, 1, 896]: one new position
            stop_next = try m.stopProb(&g, hidden);
            try asyncEvalAll(&.{ latent, hidden, stop_next });
        } else try asyncEvalAll(&.{latent});
        const sp = try itemF32(stop, m.s);
        try latents.append(a, try own(latent));
        if (sp > 0.5 and step > 3) break;
        const nh = try g.concat(&.{ try g.sliceAxis(history, 1, 1, 1, 1), latent }, 1);
        _ = mlx.mlx_array_free(history);
        history = try own(nh);
        const zn = z_next orelse break;
        _ = mlx.mlx_array_free(z);
        z = try own(zn);
        _ = mlx.mlx_array_free(stop);
        stop = try own(stop_next);
    }
    var g = G.init(a, m.s);
    defer g.deinit();
    return own(try g.concat(latents.items, 1));
}

/// Minimal protobuf wire-format reader for `Tensors.loadOnnx`.
const Pb = struct {
    b: []const u8,
    pos: usize = 0,

    const Field = struct { num: u64, wire: u3, int: u64 = 0, bytes: []const u8 = "" };

    fn varint(p: *Pb) !u64 {
        var v: u64 = 0;
        var shift: u6 = 0;
        while (true) {
            if (p.pos >= p.b.len) return error.OnnxTruncated;
            const c = p.b[p.pos];
            p.pos += 1;
            v |= @as(u64, c & 0x7f) << shift;
            if (c & 0x80 == 0) return v;
            if (shift >= 63) return error.OnnxVarint;
            shift += 7;
        }
    }

    fn next(p: *Pb) !?Field {
        if (p.pos >= p.b.len) return null;
        const key = try p.varint();
        var f: Field = .{ .num = key >> 3, .wire = @truncate(key & 7) };
        switch (f.wire) {
            0 => f.int = try p.varint(),
            1 => p.pos += 8,
            2 => {
                const n: usize = @intCast(try p.varint());
                if (p.pos + n > p.b.len) return error.OnnxTruncated;
                f.bytes = p.b[p.pos .. p.pos + n];
                p.pos += n;
            },
            5 => p.pos += 4,
            else => return error.OnnxWireType,
        }
        if (p.pos > p.b.len) return error.OnnxTruncated;
        return f;
    }
};

// ── tests ──

const testing = std.testing;

test "ming_tts model: a quantized linear tracks its dense weight, adds its bias, and is not fused" {
    const a = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = G.init(a, s);
    defer g.deinit();
    var data: [2 * 64]f32 = undefined;
    for (&data, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.37);
    const dense = try g.fromF32(&data, &.{ 2, 64 });
    const bias = try g.fromF32(&.{ 0.25, -0.25 }, &.{2});
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));

    var t: Tensors = .{ .map = std.StringHashMap(Arr).init(a), .a = a, .scheme = .{ .bits = 8, .group_size = 64 } };
    defer t.deinit();
    for ([_][]const u8{ "l.weight", "l.scales", "l.biases" }, 0..) |k, i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, qv, i));
        try t.map.put(try a.dupe(u8, k), o);
    }
    try t.map.put(try a.dupe(u8, "l.bias"), try own(bias));
    try t.map.put(try a.dupe(u8, "d.weight"), try own(dense));
    try t.map.put(try a.dupe(u8, "d.bias"), try own(bias));

    const ql = try t.lin(s, "l", .{});
    const dl = try t.lin(s, "d", .{});
    try testing.expect(denseBiased(&.{dl}));
    try testing.expect(!denseBiased(&.{ql}));

    var ones: [64]f32 = @splat(1.0 / 64.0);
    const x = try g.fromF32(&ones, &.{ 1, 64 });
    const yq = try toF32(a, try ql.f(&g, x), s);
    defer a.free(yq);
    const yd = try toF32(a, try dl.f(&g, x), s);
    defer a.free(yd);
    for (yq, yd) |p, r| try testing.expectApproxEqAbs(r, p, 0.01);

    t.scheme = .{ .bits = 4, .group_size = 64 };
    try testing.expectError(error.QuantizedGeometry, t.quantOf("l", .{}));
    t.scheme = null;
    try testing.expectError(error.UndeclaredQuantization, t.quantOf("l", .{}));
    try testing.expectEqual(@as(?Quant, null), try t.quantOf("d", .{}));
}
