//! sdk/blocks/linear/verify_qmm_nax.
//! Owns vqmm_nax_avail_cache, verifyQmmNaxAvailable, vqmm_tile_cache, verifyQmmTile, vqmm_nax_env_cache, naxLaneEnvEnabled, and 19 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const verify_qmm = @import("verify_qmm.zig");
const computeNaxAvailable = @import("../../../mlx/device.zig").computeNaxAvailable;
const gpuArchitecture = @import("../../../mlx/device.zig").gpuArchitecture;
const macosProductVersion = @import("../../../mlx/device.zig").macosProductVersion;
const VqmmTile = @import("verify_qmm.zig").VqmmTile;
const cachedScalarInt = @import("verify_qmm.zig").cachedScalarInt;
const verifyQmmEnabled = @import("verify_qmm.zig").verifyQmmEnabled;
const vqmmLaneFor = @import("verify_qmm.zig").vqmmLaneFor;
const vqmmTileFrom = @import("verify_qmm.zig").vqmmTileFrom;

var vqmm_nax_avail_cache: ?bool = null;

/// M5-class NAX units present: "applegpu_g17" GPU + macOS >= 26.2.
/// diag force-gpu-family-fallback pretends the units are absent so
/// an M5 can rehearse the exact M1-M4 plain-SIMD path (QA switch, mirrored
/// from MTPLX). Cached for the process lifetime (inference-thread caller
/// discipline, same as the kernel caches).
pub fn verifyQmmNaxAvailable() bool {
    if (vqmm_nax_probe_override) |v| return v;
    if (vqmm_nax_avail_cache) |v| return v;
    const ok = computeNaxAvailable();
    vqmm_nax_avail_cache = ok;
    return ok;
}

var vqmm_tile_cache: ?VqmmTile = null;

/// The matmul2d tile this process dispatches.
pub fn verifyQmmTile() VqmmTile {
    if (vqmm_nax_probe_override) |v| return if (v) .nax else .off;
    if (vqmm_tile_cache) |v| return v;
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    const tile = vqmmTileFrom(verifyQmmNaxAvailable(), arch, ver);
    vqmm_tile_cache = tile;
    return tile;
}

/// Where the NAX tile takes over from the plain-SIMD lanes: the byte-per-value q8 tile wins already at M=7 (+6.09%
/// over the plain lane, five thermally-settled counterbalanced pairs); packed q4/q6 still lose there, so M=8.
pub fn naxMinMForBits(bits: u32) c_int {
    return if (bits == 8) 7 else 8;
}

fn naxMinM() c_int {
    return naxMinMForBits(4);
}

/// Exact NAX-dispatch predicate, pure over the runtime gates and shape. Keep
/// controller policy behind this seam so a takeover-width tweak or ineligible
/// projection geometry can never advertise a NAX cost profile while verifyQmm
/// would actually fall through to stock qmm.
pub fn verifyQmmNaxEnabledForMFrom(
    m: c_int,
    K: c_int,
    N: c_int,
    verify_on: bool,
    available: bool,
    min_m: c_int,
) bool {
    return verify_on and vqmmLaneFor(m, K, N, available, min_m) == .nax;
}

/// Runtime form of verifyQmmNaxEnabledForMFrom.
pub fn verifyQmmNaxEnabledForM(m: c_int, K: c_int, N: c_int) bool {
    return verifyQmmNaxEnabledForMFrom(
        m,
        K,
        N,
        verifyQmmEnabled(),
        verifyQmmNaxAvailable(),
        naxMinM(),
    );
}

pub const VQMM_NAX_HEADER: [:0]const u8 = kernelSource("kernels/linear/vqmm_nax_header.metal");

/// MTPLX's _build_kernel_m16_nax_ktmpl body VERBATIM (nax_verify.py,
/// Apache-2.0; single-brace form of their f-string). The tensor
/// extents/strides encode dflash's working matmul2d convention — treat
/// them as opaque and correct, do not re-derive. Geometry: threadgroup =
/// 256 threads = 8 simdgroups; each simdgroup owns a K/8 chunk and stages
/// a dequantized 16x32 B tile in threadgroup memory (each of its 32 lanes
/// dequants ONE output column's 16 K-values from MLX's packed byte stream per
/// iteration); matmul2d (16x32x16 multiply_accumulate, execution_simdgroup)
/// multiplies the row-padded activation tile against it into a cooperative
/// fp32 accumulator; after the K loop the 8 partial C tiles reduce across
/// the threadgroup. T/GS/KCONST arrive as template args (MLX name-mangles
/// template values into the host_name, so one kernel object serves every
/// specialization — the GDN Dk/Dv precedent).
pub const VQMM_NAX_SOURCE: [:0]const u8 = kernelSource("kernels/linear/vqmm_nax.metal");

var vqmm_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// ONLY call where `verifyQmmTile()` is live — see the never-build rule.
fn getVerifyQmmNaxKernel() !mlx.mlx_fast_metal_kernel {
    if (vqmm_nax_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_vqmm_nax_m16",
        in_vec,
        out_vec,
        VQMM_NAX_SOURCE,
        VQMM_NAX_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_nax_kernel = kernel;
    return kernel;
}

/// The M-padding half of the NAX host contract, split out so it can be
/// exercised hermetically on non-M5 machines (the scaffolding test wraps
/// it around STOCK qmm — zero pad rows must produce zero output rows):
/// collapse the caller's leading shape to [m, K] (mirrors MTPLX's
/// x.reshape(m, k) so row padding is well-defined for every batch shape),
/// then zero-pad the row axis to the tile's row height. Owned handle.
pub fn naxPadRows(s: mlx.mlx_stream, x: mlx.mlx_array, m: c_int, tile: c_int, K: c_int, xd: mlx.mlx_dtype) !mlx.mlx_array {
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    const x2_shape = [_]c_int{ m, K };
    try mlx.check(mlx.mlx_reshape(&x2, x, &x2_shape, 2, s));
    var x16 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(x16);
    if (m < tile) {
        var zero = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zero);
        const zdims = [_]c_int{};
        try mlx.check(mlx.mlx_zeros(&zero, &zdims, 0, xd, s));
        const pad_axes = [_]c_int{0};
        const pad_low = [_]c_int{0};
        const pad_high = [_]c_int{tile - m};
        try mlx.check(mlx.mlx_pad(&x16, x2, &pad_axes, 1, &pad_low, 1, &pad_high, 1, zero, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&x16, x2));
    }
    return x16;
}

/// The slice-back half: first m rows of the [tile, N] output (pad rows
/// dropped). Owned handle.
pub fn naxSliceRows(s: mlx.mlx_stream, y16: mlx.mlx_array, m: c_int, tile: c_int, N: c_int) !mlx.mlx_array {
    var ym = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ym);
    if (m < tile) {
        const start = [_]c_int{ 0, 0 };
        const stop = [_]c_int{ m, N };
        const strides = [_]c_int{ 1, 1 };
        try mlx.check(mlx.mlx_slice(&ym, y16, &start, 2, &stop, 2, &strides, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&ym, y16));
    }
    return ym;
}

/// Run the NAX m16 tile: pad the activations to the fixed [16, K] tile
/// ("weight streaming dominates so padded rows are nearly free" — the
/// source ledger), grid (256, N/32), slice y back to the caller's M rows.
pub fn runVerifyQmmNax(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    m: c_int,
    tile: c_int,
    K: c_int,
    N: c_int,
    xd: mlx.mlx_dtype,
    xsh: []const c_int,
) !?mlx.mlx_array {
    if (!verify_qmm.vqmm_nax_engaged) {
        verify_qmm.vqmm_nax_engaged = true;
        log.info("[vqmm] NAX verify lane engaged at {d}-bit ({s} tile, {d} rows): M={d} K={d} N={d} gs={d}\n", .{ bits, @tagName(verifyQmmTile()), tile, m, K, N, group_size });
    }
    const x16 = try naxPadRows(s, x, m, tile, K, xd);
    defer _ = mlx.mlx_array_free(x16);

    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ tile, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256, @divExact(N, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KCONST", K));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BMT", tile));

    const inputs_arr = [_]mlx.mlx_array{ x16, w, sc, bi, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmNaxKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y16);
    try mlx.check(mlx.mlx_vector_array_get(&y16, outputs_vec, 0));

    // Drop the pad rows, restore the caller's leading shape with N last.
    const ym = try naxSliceRows(s, y16, m, tile, N);
    defer _ = mlx.mlx_array_free(ym);

    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, ym, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

/// Test seam: force the NAX availability probe (null = real probe). Only
/// ever force FALSE on non-G17 machines — forcing true would let dispatch
/// build a kernel whose matmul2d pipeline cannot be created off M5-class
/// hardware.
pub var vqmm_nax_probe_override: ?bool = null;

// ── tests ──

/// Family-wide NAX readiness: the verify family is on and the hardware probe passes. This intentionally says nothing about a specific
/// M/K/N dispatch; controller decisions use verifyQmmNaxEnabledForM instead.
pub fn verifyQmmNaxEnabled() bool {
    return verifyQmmEnabled() and verifyQmmNaxAvailable();
}

