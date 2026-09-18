//! Prism Hadamard packs (prism-ml Ternary-Bonsai-2, `prism_hadamard_qwen35`):
//! every packed linear stores its codes in a rotated basis, so the matmul reads
//! `H(signs * x)` per `block`-wide stripe, H the Sylvester Walsh-Hadamard
//! butterfly scaled 1/sqrt(block). The embedding table is the inverse side:
//! the gathered rows become `signs * H(row)`. Signs ship as `<linear>.signs`.
//! `Registry` keys them on the weight handle; `Transformer.qmatmul` and
//! `rawEmbedding` apply them. The input transform is memoized on (x, signs):
//! q/k/v, gate/up and the GDN qkv/z share one normed activation.
const std = @import("std");
const mlx = @import("mlx.zig");

pub const SIGNS_SUFFIX = ".signs";

/// One 128-thread group per `block`-wide stripe, element `(sg*P + p)*32 + lane`
/// (P = J/4): lane bits butterfly via `simd_shuffle_xor`, p bits in registers,
/// the two simdgroup bits through one threadgroup exchange.
const KERNEL_SOURCE =
    \\constexpr int S = 4;
    \\constexpr int P = J / S;
    \\threadgroup float tg[J * 32];
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint base = threadgroup_position_in_grid.x * (J * 32);
    \\uint col = base % x_shape[x_ndim - 1];
    \\float v[P];
    \\for (int p = 0; p < P; ++p) {
    \\  uint i = (sg * P + p) * 32 + lane;
    \\  float e = float(x[base + i]);
    \\  v[p] = INV ? e : e * signs[col + i];
    \\}
    \\for (int h = 1; h < P; h <<= 1) {
    \\  for (int p = 0; p < P; ++p) {
    \\    if ((p & h) == 0) { float a = v[p], b = v[p + h]; v[p] = a + b; v[p + h] = a - b; }
    \\  }
    \\}
    \\for (uint m = 1; m < 32; m <<= 1) {
    \\  float sgn = (lane & m) ? -1.0f : 1.0f;
    \\  for (int p = 0; p < P; ++p) v[p] = fma(sgn, v[p], simd_shuffle_xor(v[p], m));
    \\}
    \\for (int p = 0; p < P; ++p) tg[(sg * P + p) * 32 + lane] = v[p];
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float scale = rsqrt(float(J * 32));
    \\for (uint p = sg; p < P; p += S) {
    \\  float w[S];
    \\  for (int q = 0; q < S; ++q) w[q] = tg[(q * P + p) * 32 + lane];
    \\  float a0 = w[0] + w[1], a1 = w[0] - w[1], a2 = w[2] + w[3], a3 = w[2] - w[3];
    \\  w[0] = a0 + a2; w[2] = a0 - a2; w[1] = a1 + a3; w[3] = a1 - a3;
    \\  for (int q = 0; q < S; ++q) {
    \\    uint i = (q * P + p) * 32 + lane;
    \\    float e = w[q] * scale;
    \\    if (INV) e *= signs[col + i];
    \\    y[base + i] = static_cast<T>(e);
    \\  }
    \\}
;

var kernel_cache: ?mlx.mlx_fast_metal_kernel = null;

fn kernel() !mlx.mlx_fast_metal_kernel {
    if (kernel_cache) |k| return k;
    const in_names = [_][*:0]const u8{ "x", "signs" };
    const out_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("msv_hadamard_block", in_vec, out_vec, KERNEL_SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel_cache = k;
    return k;
}

/// Blocks the kernel holds (J/4 floats per lane, J*32 floats of threadgroup memory).
fn kernelBlock(block: c_int) bool {
    return block == 512 or block == 1024 or block == 2048;
}

fn transformKernel(x: mlx.mlx_array, signs: mlx.mlx_array, block: c_int, inverse: bool, n_total: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const shape = mlx.getShape(x);
    const dt = mlx.mlx_array_dtype(x);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, shape.ptr, shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, @divExact(n_total, block) * 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "J", @divExact(block, 32)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "INV", inverse));
    const inputs = [_]mlx.mlx_array{ x, signs };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try kernel(), in_vec, config, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&y, outs, 0));
    return y;
}

/// Input side: `H(signs * x)`; inverse: `signs * H(x)`. Computed in f32 like
/// the reference, returned in x's dtype. Caller owns the result.
pub fn transform(x: mlx.mlx_array, signs: mlx.mlx_array, block: c_int, inverse: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    if (kernelBlock(block)) {
        const sh = mlx.getShape(x);
        var total: c_int = 1;
        for (sh) |d| total *= d;
        if (sh.len > 0 and @rem(sh[sh.len - 1], block) == 0) return transformKernel(x, signs, block, inverse, total, s);
    }
    return transformComposed(x, signs, block, inverse, s);
}

fn transformComposed(x: mlx.mlx_array, signs: mlx.mlx_array, block: c_int, inverse: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(x);
    if (shape.len == 0 or shape.len >= 16) return error.RhtShape;
    const n = shape[shape.len - 1];
    if (block <= 0 or @rem(n, block) != 0) return error.RhtShape;

    var src = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(src);
    if (inverse) {
        try mlx.check(mlx.mlx_astype(&src, x, .float32, s));
    } else {
        try mlx.check(mlx.mlx_multiply(&src, x, signs, s)); // f32 signs promote
    }

    var blocked_shape: [16]c_int = undefined;
    @memcpy(blocked_shape[0 .. shape.len - 1], shape[0 .. shape.len - 1]);
    blocked_shape[shape.len - 1] = @divExact(n, block);
    blocked_shape[shape.len] = block;
    var blocked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blocked);
    try mlx.check(mlx.mlx_reshape(&blocked, src, &blocked_shape, shape.len + 1, s));

    var had = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(had);
    try mlx.check(mlx.mlx_hadamard_transform(&had, blocked, mlx.mlx_optional_float.none(), s));

    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, had, shape.ptr, shape.len, s));

    var signed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(signed);
    const pre_cast = if (inverse) blk: {
        try mlx.check(mlx.mlx_multiply(&signed, flat, signs, s));
        break :blk signed;
    } else flat;

    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, pre_cast, mlx.mlx_array_dtype(x), s));
    return out;
}

/// Residual add + RMS norm + input rotation: `sum = a + b`, `normed = w * rms(sum)`,
/// `rot = H(signs * normed)`. One 128-thread group per 1024 block; each group
/// re-reads the whole row for the sum of squares instead of syncing with the
/// others. Rounds through T exactly where the composed ops would.
const NORM_ROTATE_SOURCE =
    \\constexpr int BLK = 1024;
    \\constexpr int H = NB * BLK;
    \\threadgroup float tg[BLK];
    \\threadgroup float red[4];
    \\uint t = thread_position_in_threadgroup.x;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sgb = t / 32;
    \\uint blk = threadgroup_position_in_grid.x;
    \\uint row = threadgroup_position_in_grid.y;
    \\const device uint4* a4 = (const device uint4*)(a + row * H);
    \\const device uint4* b4 = (const device uint4*)(b + row * H);
    \\float own[8];
    \\float ss = 0.f;
    \\for (int k = 0; k < NB; ++k) {
    \\  uint4 av = a4[k * 128 + t];
    \\  uint4 bv = HAS_ADD ? b4[k * 128 + t] : uint4(0);
    \\  thread const T* ae = (thread const T*)&av;
    \\  thread const T* be = (thread const T*)&bv;
    \\  for (int q = 0; q < 8; ++q) {
    \\    float e = float(ae[q]);
    \\    if (HAS_ADD) e = float(static_cast<T>(e + float(be[q])));
    \\    ss += e * e;
    \\    if (k == int(blk)) own[q] = e;
    \\  }
    \\}
    \\ss = simd_sum(ss);
    \\if (lane == 0) red[sgb] = ss;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float inv = metal::precise::rsqrt((red[0] + red[1] + red[2] + red[3]) / H + eps);
    \\uint cb = blk * BLK + 8 * t;
    \\uint rb = row * H + cb;
    \\if (HAS_ADD) { T sv[8]; for (int q = 0; q < 8; ++q) sv[q] = static_cast<T>(own[q]); *(device uint4*)(sum + rb) = *(thread uint4*)sv; }
    \\uint4 wv = *(const device uint4*)(w + cb);
    \\thread const T* we = (thread const T*)&wv;
    \\float4 s0 = *(const device float4*)(signs + cb), s1 = *(const device float4*)(signs + cb + 4);
    \\float sgn8[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
    \\float v[8];
    \\T nv[8];
    \\for (int q = 0; q < 8; ++q) { nv[q] = we[q] * static_cast<T>(own[q] * inv); v[q] = float(nv[q]) * sgn8[q]; }
    \\if (WRITE_NORMED) *(device uint4*)(normed + rb) = *(thread uint4*)nv;
    \\for (int h = 1; h < 8; h <<= 1)
    \\  for (int q = 0; q < 8; ++q) if ((q & h) == 0) { float x0 = v[q], x1 = v[q + h]; v[q] = x0 + x1; v[q + h] = x0 - x1; }
    \\for (uint m = 1; m < 32; m <<= 1) {
    \\  float sgn = (lane & m) ? -1.0f : 1.0f;
    \\  for (int q = 0; q < 8; ++q) v[q] = fma(sgn, v[q], simd_shuffle_xor(v[q], m));
    \\}
    \\for (int q = 0; q < 8; ++q) tg[8 * t + q] = v[q];
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const float scale = 1.0f / 32.0f;
    \\for (uint base = t; base < 256; base += 128) {
    \\  float q0 = tg[base], q1 = tg[base + 256], q2 = tg[base + 512], q3 = tg[base + 768];
    \\  float a0 = q0 + q1, a1 = q0 - q1, a2 = q2 + q3, a3 = q2 - q3;
    \\  uint ob = row * H + blk * BLK + base;
    \\  rot[ob] = static_cast<T>((a0 + a2) * scale);
    \\  rot[ob + 256] = static_cast<T>((a1 + a3) * scale);
    \\  rot[ob + 512] = static_cast<T>((a0 - a2) * scale);
    \\  rot[ob + 768] = static_cast<T>((a1 - a3) * scale);
    \\}
;

var norm_rotate_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const NormRotate = struct {
    sum: ?mlx.mlx_array,
    normed: ?mlx.mlx_array,
    rot: mlx.mlx_array,
};

/// Every threadgroup re-reads its whole row for the sum of squares (NB× read
/// amplification), so the kernel serves decode/verify widths only; prefill
/// keeps the composed chain.
pub const NORM_ROTATE_MAX_ROWS: c_int = 16;

/// Null when the shape is outside the kernel (caller keeps the composed ops).
/// `b` null = no residual add; `want_normed` also returns the unrotated norm.
pub fn normRotate(a: mlx.mlx_array, b: ?mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, signs: mlx.mlx_array, block: c_int, want_normed: bool, s: mlx.mlx_stream) !?NormRotate {
    if (block != 1024 or eps.ctx == null) return null;
    const sh = mlx.getShape(a);
    if (sh.len == 0 or sh.len > 4) return null;
    const hidden = sh[sh.len - 1];
    if (@rem(hidden, block) != 0) return null;
    const dt = mlx.mlx_array_dtype(a);
    if (mlx.mlx_array_dtype(w) != dt) return null;
    if (b) |bb| if (mlx.mlx_array_dtype(bb) != dt) return null;
    var rows: c_int = 1;
    for (sh[0 .. sh.len - 1]) |d| rows *= d;
    if (rows > NORM_ROTATE_MAX_ROWS) return null;

    if (norm_rotate_kernel == null) {
        const in_names = [_][*:0]const u8{ "a", "b", "w", "eps", "signs" };
        const out_names = [_][*:0]const u8{ "sum", "normed", "rot" };
        const in_vec = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const k = mlx.mlx_fast_metal_kernel_new("msv_norm_rotate", in_vec, out_vec, NORM_ROTATE_SOURCE, "", true, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        norm_rotate_kernel = k;
    }
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, sh.ptr, sh.len, dt));
    const nb = @divExact(hidden, block);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, nb * 128, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NB", nb));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "HAS_ADD", b != null));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(config, "WRITE_NORMED", want_normed));
    const inputs = [_]mlx.mlx_array{ a, b orelse a, w, eps, signs };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, norm_rotate_kernel.?, in_vec, config, s));
    var res: [3]mlx.mlx_array = undefined;
    for (&res, 0..) |*r, i| {
        r.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(r, outs, i));
    }
    if (b == null) _ = mlx.mlx_array_free(res[0]);
    if (!want_normed) _ = mlx.mlx_array_free(res[1]);
    return .{
        .sum = if (b != null) res[0] else null,
        .normed = if (want_normed) res[1] else null,
        .rot = res[2],
    };
}

fn retained(a: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, a));
    return r;
}

/// Identity of the UNDERLYING array: `mlx_array_set` copies the wrapper, the
/// shape storage lives in the shared descriptor, which a retained handle pins.
fn identity(a: mlx.mlx_array) usize {
    return @intFromPtr(mlx.mlx_array_shape(a));
}

/// Lazy f32 view of a sign vector; the caller evals a batch of these at once
/// and hands each to `register`.
pub fn signsF32(a: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var f = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&f, a, .float32, s));
    return f;
}

fn signsHash(f: mlx.mlx_array) !u64 {
    const n: usize = @intCast(mlx.mlx_array_size(f));
    const data = mlx.mlx_array_data_float32(f) orelse return error.RhtSignsRead;
    return std.hash.Wyhash.hash(n, std.mem.sliceAsBytes(data[0..n]));
}

pub const Registry = struct {
    allocator: std.mem.Allocator,
    block: c_int,
    /// Every rotated weight's biases are exactly -scales (checked at load).
    bias_is_neg_scale: bool = true,
    map: std.AutoHashMapUnmanaged(usize, mlx.mlx_array) = .{},
    /// One retained handle per DISTINCT sign vector, so siblings reading one
    /// activation hit the memo below.
    pool: std.AutoHashMapUnmanaged(u64, mlx.mlx_array) = .{},
    /// The pooled sign vector per input width, for fused producers. A width
    /// with two distinct vectors is unusable there: a producer seeds the memo
    /// under one vector and the consumer under the other would rotate twice.
    by_width: std.AutoHashMapUnmanaged(c_int, mlx.mlx_array) = .{},
    width_conflict: std.AutoHashMapUnmanaged(c_int, void) = .{},
    memo_key: usize = 0,
    memo_signs: ?*anyopaque = null,
    memo_x: mlx.mlx_array = .{ .ctx = null },
    memo_out: mlx.mlx_array = .{ .ctx = null },

    pub fn init(allocator: std.mem.Allocator, block: u32) Registry {
        return .{ .allocator = allocator, .block = @intCast(block) };
    }

    pub fn deinit(self: *Registry) void {
        var it = self.pool.valueIterator();
        while (it.next()) |a| _ = mlx.mlx_array_free(a.*);
        self.pool.deinit(self.allocator);
        self.by_width.deinit(self.allocator);
        self.width_conflict.deinit(self.allocator);
        self.map.deinit(self.allocator);
        self.dropMemo();
    }

    pub fn count(self: *const Registry) usize {
        return self.map.count();
    }

    pub fn conflictingWidths(self: *const Registry) usize {
        return self.width_conflict.count();
    }

    /// Binds `signs` to `w`, copied into the pool (the load-time weights map
    /// that handed them out is freed after load). `signs_f32` is an EVALUATED
    /// f32 copy (`signsF32`), hashed to dedup the pool.
    pub fn register(self: *Registry, w: mlx.mlx_array, signs: mlx.mlx_array, signs_f32: mlx.mlx_array) !void {
        if (w.ctx == null) return error.RhtNullWeight;
        const key = @intFromPtr(w.ctx);
        if (self.map.contains(key)) return;
        const h = try signsHash(signs_f32);
        const shared = self.pool.get(h) orelse blk: {
            const kept = try retained(signs);
            try self.pool.put(self.allocator, h, kept);
            break :blk kept;
        };
        try self.map.put(self.allocator, key, shared);
        const width: c_int = @intCast(mlx.mlx_array_size(signs));
        if (self.by_width.get(width)) |have| {
            if (have.ctx != shared.ctx) try self.width_conflict.put(self.allocator, width, {});
        } else try self.by_width.put(self.allocator, width, shared);
    }

    /// The one sign vector every rotated weight of `width` reads, or null.
    pub fn signsFor(self: *const Registry, width: c_int) ?mlx.mlx_array {
        if (self.width_conflict.contains(width)) return null;
        return self.by_width.get(width);
    }

    pub fn get(self: *const Registry, w: mlx.mlx_array) ?mlx.mlx_array {
        if (w.ctx == null or self.map.count() == 0) return null;
        return self.map.get(@intFromPtr(w.ctx));
    }

    /// The memo pins the last input and its rotation (a prefill chunk's worth
    /// of activation); a forward drops it on exit.
    pub fn dropMemo(self: *Registry) void {
        if (self.memo_x.ctx != null) _ = mlx.mlx_array_free(self.memo_x);
        if (self.memo_out.ctx != null) _ = mlx.mlx_array_free(self.memo_out);
        self.memo_x = .{ .ctx = null };
        self.memo_out = .{ .ctx = null };
        self.memo_key = 0;
        self.memo_signs = null;
    }

    /// Hands `rot` to the next `applyIn(x, signs)` without a dispatch: a fused
    /// producer already computed it.
    pub fn seed(self: *Registry, x: mlx.mlx_array, signs: mlx.mlx_array, rot: mlx.mlx_array) !void {
        self.dropMemo();
        self.memo_x = try retained(x);
        self.memo_out = try retained(rot);
        self.memo_key = identity(x);
        self.memo_signs = signs.ctx;
    }

    /// Input-side transform, memoized on (x identity, signs handle). Caller
    /// owns the result.
    pub fn applyIn(self: *Registry, x: mlx.mlx_array, signs: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        if (self.memo_out.ctx != null and self.memo_key == identity(x) and self.memo_signs == signs.ctx) {
            return retained(self.memo_out);
        }
        const out = try transform(x, signs, self.block, false, s);
        self.dropMemo();
        self.memo_x = try retained(x);
        self.memo_out = try retained(out);
        self.memo_key = identity(x);
        self.memo_signs = signs.ctx;
        return out;
    }
};

// ── tests ──

fn refHadamard(v: []f32) void {
    var stride: usize = 1;
    while (stride < v.len) : (stride <<= 1) {
        for (0..v.len) |lane| {
            if (lane & stride == 0) {
                const a = v[lane];
                const b = v[lane | stride];
                v[lane] = a + b;
                v[lane | stride] = a - b;
            }
        }
    }
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(v.len)));
    for (v) |*e| e.* *= scale;
}

test "rht: block-1024 transform (kernel + composed) matches the reference butterfly on both sides" {
    const s = mlx.gpuStream();
    const rows: usize = 2;
    const n: usize = 2048;
    const block: usize = 1024;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const x = try std.testing.allocator.alloc(f32, rows * n);
    defer std.testing.allocator.free(x);
    for (x) |*e| e.* = rnd.floatNorm(f32);
    var signs: [n]f32 = undefined;
    for (&signs) |*e| e.* = if (rnd.boolean()) 1.0 else -1.0;

    const x_shape = [_]c_int{ @intCast(rows), @intCast(n) };
    const s_shape = [_]c_int{@intCast(n)};
    const xa = mlx.mlx_array_new_data(x.ptr, &x_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(xa);
    const sa = mlx.mlx_array_new_data(&signs, &s_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(sa);

    const want = try std.testing.allocator.alloc(f32, rows * n);
    defer std.testing.allocator.free(want);
    for ([_]bool{ false, true }) |inverse| {
        for ([_]bool{ true, false }) |fused| {
        const got = if (fused) try transform(xa, sa, @intCast(block), inverse, s) else try transformComposed(xa, sa, @intCast(block), inverse, s);
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(got));
        for (0..rows * n) |i| want[i] = if (inverse) x[i] else x[i] * signs[i % n];
        var c: usize = 0;
        while (c < rows * n) : (c += block) refHadamard(want[c .. c + block]);
        if (inverse) for (0..rows * n) |i| {
            want[i] *= signs[i % n];
        };
        const data = mlx.mlx_array_data_float32(got).?;
        for (0..rows * n) |i| try std.testing.expectApproxEqAbs(want[i], data[i], 1e-4);
        }
    }
}

test "rht: two distinct sign vectors at one width dedup the pool and disable signsFor" {
    const s = mlx.gpuStream();
    const n: usize = 1024;
    var sg1: [n]f32 = undefined;
    var sg2: [n]f32 = undefined;
    for (&sg1, &sg2, 0..) |*a, *b, i| {
        a.* = if (i % 3 == 0) -1.0 else 1.0;
        b.* = -a.*;
    }
    const vs = [_]c_int{@intCast(n)};
    const s1 = mlx.mlx_array_new_data(&sg1, &vs, 1, .float32);
    defer _ = mlx.mlx_array_free(s1);
    const s1b = mlx.mlx_array_new_data(&sg1, &vs, 1, .float32);
    defer _ = mlx.mlx_array_free(s1b);
    const s2 = mlx.mlx_array_new_data(&sg2, &vs, 1, .float32);
    defer _ = mlx.mlx_array_free(s2);
    const w_shape = [_]c_int{ 8, 64 };
    var ws: [3]mlx.mlx_array = undefined;
    for (&ws) |*w| w.* = mlx.mlx_array_new_data(&sg1, &w_shape, 2, .float32);
    defer for (ws) |w| {
        _ = mlx.mlx_array_free(w);
    };
    var reg = Registry.init(std.testing.allocator, 1024);
    defer reg.deinit();
    for (ws, [_]mlx.mlx_array{ s1, s1b, s2 }) |w, sg| {
        const f = try signsF32(sg, s);
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_array_eval(f));
        try reg.register(w, sg, f);
        if (sg.ctx == s1b.ctx) {
            try std.testing.expectEqual(@as(usize, 1), reg.pool.count());
            try std.testing.expect(reg.signsFor(@intCast(n)) != null);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), reg.pool.count());
    try std.testing.expect(reg.signsFor(@intCast(n)) == null);
    try std.testing.expect(reg.get(ws[2]) != null);
}

test "rht: normRotate matches add -> rms_norm -> transform" {
    const s = mlx.gpuStream();
    const n: usize = 3072;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    var a: [2 * n]f32 = undefined;
    var b: [2 * n]f32 = undefined;
    var w: [n]f32 = undefined;
    var sg: [n]f32 = undefined;
    for (&a) |*e| e.* = rnd.floatNorm(f32);
    for (&b) |*e| e.* = rnd.floatNorm(f32);
    for (&w) |*e| e.* = 1.0 + 0.1 * rnd.floatNorm(f32);
    for (&sg) |*e| e.* = if (rnd.boolean()) 1.0 else -1.0;
    const xs = [_]c_int{ 2, @intCast(n) };
    const vs = [_]c_int{@intCast(n)};
    const af = mlx.mlx_array_new_data(&a, &xs, 2, .float32);
    defer _ = mlx.mlx_array_free(af);
    const bf = mlx.mlx_array_new_data(&b, &xs, 2, .float32);
    defer _ = mlx.mlx_array_free(bf);
    const wf = mlx.mlx_array_new_data(&w, &vs, 1, .float32);
    defer _ = mlx.mlx_array_free(wf);
    const sa = mlx.mlx_array_new_data(&sg, &vs, 1, .float32);
    defer _ = mlx.mlx_array_free(sa);
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    var bb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bb);
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&ab, af, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&bb, bf, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&wb, wf, .bfloat16, s));
    const eps_v = [_]f32{1e-6};
    const eps = mlx.mlx_array_new_data(&eps_v, &[_]c_int{}, 0, .float32);
    defer _ = mlx.mlx_array_free(eps);

    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, ab, bb, s));
    var normed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normed);
    try mlx.check(mlx.mlx_fast_rms_norm(&normed, sum, wb, 1e-6, s));
    const want = try transformComposed(normed, sa, 1024, false, s);
    defer _ = mlx.mlx_array_free(want);

    const got = (try normRotate(ab, bb, wb, eps, sa, 1024, true, s)).?;
    defer _ = mlx.mlx_array_free(got.rot);
    defer _ = mlx.mlx_array_free(got.sum.?);
    defer _ = mlx.mlx_array_free(got.normed.?);
    for ([_][2]mlx.mlx_array{ .{ want, got.rot }, .{ normed, got.normed.? }, .{ sum, got.sum.? } }) |pair| {
        var d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d);
        var d32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(d32);
        try mlx.check(mlx.mlx_subtract(&d, pair[0], pair[1], s));
        try mlx.check(mlx.mlx_astype(&d32, d, .float32, s));
        try mlx.check(mlx.mlx_array_eval(d32));
        const data = mlx.mlx_array_data_float32(d32).?;
        var worst: f32 = 0;
        for (data[0 .. 2 * n]) |e| worst = @max(worst, @abs(e));
        try std.testing.expect(worst < 0.02); // one bf16 ulp at |x| ~ 4
    }
}
