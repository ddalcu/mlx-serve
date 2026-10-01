//! Residual add fused with the NEXT block's RMSNorm, for decode-width rows:
//! `h_new = h + delta`, `hn = rmsnorm(h_new) * w`. The MoE arm computes delta
//! itself: the f32 router-weighted sum over the K expert outputs (rounded
//! like the composed chain) plus the shared expert's output. One dispatch
//! replaces up to six. Same structure as TensorFold's `add_norm` (MIT).
const std = @import("std");
const mlx = @import("mlx.zig");

// Mirrors MLX's `rms_norm` kernels so the normed output is bit-equal to
// `add -> fast::rms_norm`: FOUR consecutive elements per thread per pass,
// per-simdgroup sums folded through a 32-slot table, precise rsqrt,
// `w * T(x * inv)` in the output dtype. Up to 4096 wide that is MLX's
// single-row kernel (TN = ceil(D/4) threads, one pass); wider, its looped one
// (1024 threads striding 4096 elements a pass).
fn source(comptime moe: bool) [:0]const u8 {
    return std.fmt.comptimePrint(
        \\const uint t = thread_position_in_threadgroup.x;
        \\const uint r = threadgroup_position_in_grid.x;
        \\const uint lane = thread_index_in_simdgroup;
        \\const uint sg = simdgroup_index_in_threadgroup;
        \\constexpr int NC = (D + TN * 4 - 1) / (TN * 4);
        \\threadgroup float local_sums[32];
        \\float hv[NC * 4];
        \\float acc = 0.0f;
        \\for (int p = 0; p < NC; p++)
        \\for (int i = 0; i < 4; i++) {{
        \\    const int c = p * TN * 4 + int(t) * 4 + i;
        \\    hv[p * 4 + i] = 0.0f;
        \\    if (c < D) {{
        \\        const size_t at = size_t(r) * D + c;
        \\        float delta;
        \\        {s}
        \\        const T hn = T(float(H[at]) + delta);
        \\        H_NEW[at] = hn;
        \\        hv[p * 4 + i] = float(hn);
        \\    }}
        \\    acc += hv[p * 4 + i] * hv[p * 4 + i];
        \\}}
        \\acc = simd_sum(acc);
        \\if (sg == 0) local_sums[lane] = 0.0f;
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\if (lane == 0) local_sums[sg] = acc;
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\const float total = simd_sum(local_sums[lane]);
        \\const float inv = metal::precise::rsqrt(total / float(D) + eps);
        \\for (int p = 0; p < NC; p++)
        \\for (int i = 0; i < 4; i++) {{
        \\    const int c = p * TN * 4 + int(t) * 4 + i;
        \\    if (c < D) HN[size_t(r) * D + c] = T(float(W[c]) * float(T(hv[p * 4 + i] * inv)));
        \\}}
    , .{if (moe)
        // The chain's own rounding points: f32 multiply-adds over the K slots
        // in slot order, `.astype` to T, the shared expert added in T.
        \\{
        \\    float routed = 0.0f;
        \\    for (int e = 0; e < K; e++) routed += float(Y[(size_t(r) * K + e) * D + c]) * WT[r * K + e];
        \\    delta = float(T(routed));
        \\    if (SHARED) delta = float(T(delta + float(XS[at])));
        \\}
    else
        \\delta = float(X[at]);
    });
}

const PLAIN_SOURCE = source(false);
const MOE_SOURCE = source(true);

var plain_kernel: ?mlx.mlx_fast_metal_kernel = null;
var moe_kernel: ?mlx.mlx_fast_metal_kernel = null;
const CfgKey = struct { rows: c_int, d: c_int, k: c_int, shared: bool, dt: mlx.mlx_dtype, tn: c_int };
var plain_key: ?CfgKey = null;
var plain_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };
var moe_key: ?CfgKey = null;
var moe_cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null };

// MLX's single-row kernel serves widths up to its looped limit (4096) with
// ceil(D/4) threads; its looped one runs 1024 (two passes cover 8192).
fn threadsFor(d: c_int) ?c_int {
    if (d <= 0 or d > 8192) return null;
    if (d <= 4096) return @divTrunc(d + 3, 4);
    return if (looped_ok == false) null else loopedNt();
}
const LOOPED_NT: c_int = 1024;
pub var looped_nt_override: ?c_int = null; // test seam: 2048 exceeds every GPU's limit
fn loopedNt() c_int {
    return looped_nt_override orelse LOOPED_NT;
}
// Whether this GPU's pipeline runs the looped width's threadgroup; null = not dispatched yet.
var looped_ok: ?bool = null;

/// Whether the fused add+norm serves this width and dtype (callers decide the
/// path up front so a decline never recomputes routing).
pub fn eligible(d: c_int, dt: mlx.mlx_dtype, s: mlx.mlx_stream) bool {
    if (!mlx.streamIsGpu(s)) return false;
    if (dt != .bfloat16 and dt != .float16) return false;
    return threadsFor(d) != null;
}

fn makeKernel(name: [*:0]const u8, ins: []const [*:0]const u8, src: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const outs = [_][*:0]const u8{ "H_NEW", "HN" };
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, src, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

fn buildConfig(key: CfgKey, shape: []const c_int) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const tn = key.tn;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, shape.ptr, shape.len, key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, shape.ptr, shape.len, key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, tn * key.rows, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, tn, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", key.dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "D", key.d));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TN", tn));
    if (key.k > 0) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "K", key.k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "SHARED", @intFromBool(key.shared)));
    }
    return c;
}

pub const Out = struct { h: mlx.mlx_array, normed: mlx.mlx_array };

fn apply(kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, key: CfgKey, arrs: []const mlx.mlx_array, s: mlx.mlx_stream) !?Out {
    const v = mlx.mlx_vector_array_new_data(arrs.ptr, arrs.len);
    defer _ = mlx.mlx_vector_array_free(v);
    var o = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(o);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&o, kernel, v, cfg, s));
    var out: Out = .{ .h = mlx.mlx_array_new(), .normed = mlx.mlx_array_new() };
    errdefer {
        _ = mlx.mlx_array_free(out.h);
        _ = mlx.mlx_array_free(out.normed);
    }
    try mlx.check(mlx.mlx_vector_array_get(&out.h, o, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.normed, o, 1));
    // A pipeline's thread limit is per GPU and known only once MLX builds it: the
    // looped width's first dispatch evaluates here, a refusal comes off the latch
    // and the caller keeps the two ops. Any other error stays latched.
    if (key.tn > 256 and looped_ok == null and !mlx.errorPending()) {
        _ = mlx.mlx_eval(o);
        if (mlx.takeErrorIf("maximum allowed threads per threadgroup")) {
            looped_ok = false;
            _ = mlx.mlx_array_free(out.h);
            _ = mlx.mlx_array_free(out.normed);
            return null;
        }
        if (!mlx.errorPending()) looped_ok = true;
    }
    return out;
}

/// h, x: [B,S,D]; w: [D]; eps: 0-dim f32. Null outside the envelope.
pub fn addNorm(h: mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, s: mlx.mlx_stream) !?Out {
    const sh = mlx.getShape(h);
    if (sh.len != 3) return null;
    const dt = mlx.mlx_array_dtype(h);
    if (!eligible(sh[2], dt, s)) return null;
    if (mlx.mlx_array_dtype(x) != dt or mlx.mlx_array_dtype(w) != dt) return null;
    if (plain_kernel == null) plain_kernel = try makeKernel("msv_add_norm", &[_][*:0]const u8{ "H", "X", "W", "eps" }, PLAIN_SOURCE);
    const key = CfgKey{ .rows = sh[0] * sh[1], .d = sh[2], .k = 0, .shared = false, .dt = dt, .tn = threadsFor(sh[2]).? };
    if (plain_key == null or !std.meta.eql(plain_key.?, key)) {
        const c = try buildConfig(key, sh);
        if (plain_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(plain_cfg);
        plain_cfg = c;
        plain_key = key;
    }
    return try apply(plain_kernel.?, plain_cfg, key, &[_]mlx.mlx_array{ h, x, w, eps }, s);
}

/// h: [B,S,D]; y: [B*S, K, D] per-expert outputs; wt: [B*S, K] f32 router
/// weights; shared: the shared expert's output [B*S, D] (or null). Null
/// outside the envelope.
pub fn moeCombineAddNorm(h: mlx.mlx_array, y: mlx.mlx_array, wt: mlx.mlx_array, shared: ?mlx.mlx_array, w: mlx.mlx_array, eps: mlx.mlx_array, s: mlx.mlx_stream) !?Out {
    const sh = mlx.getShape(h);
    if (sh.len != 3) return null;
    const dt = mlx.mlx_array_dtype(h);
    if (!eligible(sh[2], dt, s)) return null;
    const ysh = mlx.getShape(y);
    const wsh = mlx.getShape(wt);
    if (ysh.len != 3 or wsh.len != 2) return null;
    const rows = sh[0] * sh[1];
    if (ysh[0] != rows or ysh[2] != sh[2] or wsh[0] != rows or wsh[1] != ysh[1]) return null;
    if (mlx.mlx_array_dtype(y) != dt or mlx.mlx_array_dtype(w) != dt or mlx.mlx_array_dtype(wt) != .float32) return null;
    if (shared) |xs| {
        const xsh = mlx.getShape(xs);
        if (xsh.len != 2 or xsh[0] != rows or xsh[1] != sh[2] or mlx.mlx_array_dtype(xs) != dt) return null;
    }
    if (moe_kernel == null) moe_kernel = try makeKernel("msv_moe_combine_add_norm", &[_][*:0]const u8{ "H", "Y", "WT", "XS", "W", "eps" }, MOE_SOURCE);
    const key = CfgKey{ .rows = rows, .d = sh[2], .k = wsh[1], .shared = shared != null, .dt = dt, .tn = threadsFor(sh[2]).? };
    if (moe_key == null or !std.meta.eql(moe_key.?, key)) {
        const c = try buildConfig(key, sh);
        if (moe_cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(moe_cfg);
        moe_cfg = c;
        moe_key = key;
    }
    // With no shared expert `h` stands in for XS; SHARED=0 never reads it.
    return try apply(moe_kernel.?, moe_cfg, key, &[_]mlx.mlx_array{ h, y, wt, shared orelse h, w, eps }, s);
}

/// h, x [1, rows, d] and w [d]: bf16 normals, scale 3.
fn randTriple(rows: c_int, d: c_int, s: mlx.mlx_stream) ![3]mlx.mlx_array {
    var arrs: [3]mlx.mlx_array = undefined;
    for (&arrs, 0..) |*a, i| {
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, 11 + i));
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        const shape: []const c_int = if (i == 2) &.{d} else &.{ 1, rows, d };
        try mlx.check(mlx.mlx_random_normal(&f, shape.ptr, shape.len, .float32, 0.0, 3.0, key, s));
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(a, f, .bfloat16, s));
    }
    return arrs;
}

test "addNorm declines, and stays declined, when the GPU refuses the looped width's threadgroup" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    looped_nt_override = 2048;
    looped_ok = null;
    defer {
        looped_nt_override = null;
        looped_ok = null;
        var buf: [256]u8 = undefined;
        _ = mlx.takeError(&buf);
    }
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);
    const arrs = try randTriple(1, 5120, s);
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try std.testing.expect((try addNorm(arrs[0], arrs[1], arrs[2], eps, s)) == null);
    try std.testing.expect(!eligible(5120, .bfloat16, s));
    try std.testing.expect(eligible(4096, .bfloat16, s));
    try std.testing.expect(!mlx.errorPending());
}

test "addNorm is bit-equal to add -> fast::rms_norm on both sides of MLX's looped limit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);
    for ([_]c_int{ 2688, 4096, 4100, 5120, 8192 }) |d| for ([_]c_int{ 1, 16 }) |rows| {
        const arrs = try randTriple(rows, d, s);
        defer for (arrs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const got = (try addNorm(arrs[0], arrs[1], arrs[2], eps, s)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(got.h);
        defer _ = mlx.mlx_array_free(got.normed);
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_add(&sum, arrs[0], arrs[1], s));
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, sum, arrs[2], 1e-6, s));
        for ([_][2]mlx.mlx_array{ .{ sum, got.h }, .{ normed, got.normed } }) |pair| {
            var e = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(e);
            try mlx.check(mlx.mlx_array_equal(&e, pair[0], pair[1], false, s));
            try mlx.check(mlx.mlx_array_eval(e));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, e));
            if (!same) std.debug.print("add_norm differs at D={d} rows={d}\n", .{ d, rows });
            try std.testing.expect(same);
        }
    };
}
