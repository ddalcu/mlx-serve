//! sdk/blocks/moe/gather_qmv.
//! Owns gatherQmvSource, GQMV_BODY_AFFINE, GQMV_BODY_NVFP4, GQMV_AFFINE_HEADER, GQMV_NVFP4_HEADER, gatherQmvGateUpSource, and 28 more.

const std = @import("std");
const env = @import("../../../env.zig");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../util/log.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const moe_fp4 = @import("moe_fp4.zig");
const fused_activations = @import("../mlp/fused_activations.zig");
const testing = std.testing;
const testRandWeightBf16 = @import("../../testing/parity.zig").testRandWeightBf16;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;
const cachedScalarInt = @import("../linear/verify_qmm.zig").cachedScalarInt;
const clampedSwiGLU = @import("../mlp/activations.zig").clampedSwiGLU;
const fusedSwiGLU = @import("../mlp/fused_activations.zig").fusedSwiGLU;
const swigluSigTable = @import("../mlp/fused_activations.zig").swigluSigTable;
const moeCudaFormat = @import("gather_qmv_rows.zig").moeCudaFormat;

// ── MoE decode gather-qmv kernel ──
// MLX's `gather_qmm` at decode width costs O(EXPERT BANK SIZE), not O(work), in
// any C/C++ process (it is flat under pip-Python with the identical dylib —
// measured E-sweep in the `MoE decode gather` µbench: 30/50/77/300 us at banks
// of 8/25/50/201 MB, all reading the SAME 10 experts). 141 gathers/token x 300 us
// is the whole 42 ms/token Laguna decode. `take` from the same 201 MB bank does
// NOT pay it, so indexed reads are fine; the penalty is specific to gather_qmm's
// addressing.
//
// The shipped workaround (`batchedExpertMm`: take + broadcast + batched qmm)
// dodges it but pays a materialization round-trip — read 9.8 MB of experts,
// write 9.8 MB, read 9.8 MB back = 3x the ideal traffic, which is exactly its
// measured 70 us against a 67 us bandwidth prediction. It cannot go faster.
//
// This kernel reads the bank IN PLACE with GPU-resident indices: one simdgroup
// per (expert slot, output row), each lane striding the packed row and reducing
// with simd_sum. Ideal traffic, no copy, no host sync for the top-K indices.
//
// `X_PER_EXPERT` picks the input layout: 0 = one shared x [K] (gate/up
// projections, every expert sees the same token), 1 = x [TOPK, K] (the down
// projection, where each expert consumes its own activation).
//
// `NVFP4` picks the dequant arithmetic: false = affine (q * scale + bias, the
// mlx-quantized checkpoints), true = NVFP4 (e2m1 nibbles + uint8 e4m3 group
// scales, no biases — poolside's Laguna checkpoints).
fn gatherQmvSource(comptime nvfp4: bool, comptime x_per_expert: bool) [:0]const u8 {
    // Everything is indexed off the raw buffer names rather than through
    // explicitly address-spaced pointers: MLX generates the signature from each
    // input's ACTUAL dtype and puts arrays with fewer than 8 elements in
    // `constant` instead of `device` (backend/common/metal_kernel.cpp), so a
    // hand-declared `const device T*` would mismatch on both counts. `x` in
    // particular arrives as float32 on Laguna while scales/biases are bf16.
    //
    // A uint4 (128-bit) load variant was measured and is NOT kept: 38.9 us vs
    // 37.3 for the scalar form. At ~5 ALU ops per dequantized value this loop
    // is ALU-bound, not load-width bound, so widening the loads only adds code.
    return std.fmt.comptimePrint(kernelSource("kernels/moe/gather_qmv.fmt.metal"), .{
        if (x_per_expert) "(size_t)e * (size_t)K" else "0",
        if (nvfp4) GQMV_BODY_NVFP4 else GQMV_BODY_AFFINE,
        // NVFP4 decodes both halves scaled down by a power of two (2^14 for the
        // e2m1 nibble, 2^8 for the e4m3 scale). Multiplying by a power of two is
        // exact in float, so the two constants fold into ONE 2^22 multiply on
        // the finished dot product instead of two per value.
        if (nvfp4) " * 4194304.0f" else "",
    });
}

pub const GQMV_BODY_AFFINE = kernelSource("kernels/moe/gqmv_body_affine.metal");

// The group scale is constant across a whole packed word (GS 16 >= VPW 8), so
// the word's four partial dot products are summed unscaled and the scale is
// applied once per word rather than once per value.
pub const GQMV_BODY_NVFP4 = kernelSource("kernels/moe/gqmv_body_nvfp4.metal");

// An e2m1 nibble and an e4m3 byte each become a `half` by shifting their bits
// straight into the half exponent/mantissa/sign fields — no LUT, no branches,
// exact for all codes including zero, subnormals and negative zero. The e2m1
// field layout (2-bit exp, 1-bit mantissa) lands in half's low exponent bits
// and top mantissa bit, which yields the true value scaled by 2^-14; e4m3's
// (4-bit exp, 3-bit mantissa) yields it scaled by 2^-8. Callers fold both back
// in with one multiply. Matches MLX's own `fp4.h` / `fp8.h` conversions, whose
// sign handling is a branch instead of a shift.
// One packed unit of quantized values as a uint32: a word for 2/4/8-bit; at
// 3-bit, mx.quantize packs 8 values into 3 bytes (value i at bit 3i), so the
// unit is a byte triple and the body's `>> (i * BITS)` walk is unchanged.
pub const GQMV_AFFINE_HEADER = kernelSource("kernels/moe/gqmv_affine_header.metal");

pub const GQMV_NVFP4_HEADER = GQMV_AFFINE_HEADER ++ moe_fp4.E2M1_HEADER ++ moe_fp4.E2M1X2_HEADER ++ kernelSource("kernels/moe/gqmv_nvfp4_header.metal");

// ── Fused routed gate+up+SwiGLU (one kernel for what was three) ──
//
// The decode expert path ran gate gather -> up gather -> activation as three
// dispatches, all on the critical chain into down_proj. This computes BOTH dot
// products in the SAME simdgroup and applies the activation before it writes,
// so the x element loaded for the gate FMA is reused for the up FMA and the two
// dequant chains interleave instead of running as separate launches.
//
// It halves the simdgroup count rather than the work (each does two rows'
// worth), which is the right trade here: 4096 simdgroups over 40 cores was ~100
// per core, so there is occupancy to spend on instruction-level parallelism.
//
// Bit-exactness comes from rounding each accumulator to T FIRST — exactly where
// `gatherQmv` writes `T(acc)` — and then running the same table-lookup SwiGLU
// `fusedSwiGLU` does. No f32 shortcut through the activation.
fn gatherQmvGateUpSource(comptime nvfp4: bool) [:0]const u8 {
    return std.fmt.comptimePrint(kernelSource("kernels/moe/gather_qmv_gateup.fmt.metal"), .{
        if (nvfp4) GQMV_GATEUP_BODY_NVFP4 else GQMV_GATEUP_BODY_AFFINE,
        if (nvfp4) " * 4194304.0f" else "",
        if (nvfp4) " * 4194304.0f" else "",
    });
}

pub const GQMV_GATEUP_BODY_AFFINE = kernelSource("kernels/moe/gqmv_gateup_body_affine.metal");

pub const GQMV_GATEUP_BODY_NVFP4 = kernelSource("kernels/moe/gqmv_gateup_body_nvfp4.metal");

pub const GQMV_GATEUP_SOURCES = [2][:0]const u8{ gatherQmvGateUpSource(false), gatherQmvGateUpSource(true) };
const GQMV_GATEUP_NAMES = [2][*:0]const u8{ "mlxserve_moe_gather_qmv_gateup", "mlxserve_moe_gather_qmv_gateup_nvfp4" };
var gqmv_gateup_kernels: [2]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getGatherQmvGateUpKernel(nvfp4: bool) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @intFromBool(nvfp4);
    if (gqmv_gateup_kernels[mi]) |k| return k;
    const affine_inputs = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size" };
    const nvfp4_inputs = [_][*:0]const u8{ "x", "wg_q", "g_scales", "wu_q", "u_scales", "inds", "sigtab", "K_size", "N_size" };
    const input_names: []const [*:0]const u8 = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_GATEUP_NAMES[mi],
        in_vec,
        out_vec,
        GQMV_GATEUP_SOURCES[mi],
        if (nvfp4) GQMV_NVFP4_HEADER else GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_gateup_kernels[mi] = kernel;
    return kernel;
}

/// One quant group per pack: a word for 2/4/8-bit, a byte triple (8 values) for 3-bit.
pub fn affineQmvUnitOk(bits: u32, group_size: u32) bool {
    return switch (bits) {
        2, 4, 8 => group_size % (32 / bits) == 0,
        3 => group_size % 8 == 0,
        else => false,
    };
}

pub fn affineQmvPackUnits(K: c_int, bits: u32) c_int {
    return if (bits == 3) @divExact(K, 8) else @divExact(K * @as(c_int, @intCast(bits)), 32);
}

const GateUpCfgKey = struct { topk: c_int, n: c_int, k: c_int, bits: u32, gs: u32, dtype: mlx.mlx_dtype, limit: u32 };
var gateup_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gateup_cfg_key: GateUpCfgKey = std.mem.zeroes(GateUpCfgKey);
var gqmv_gateup_engaged: bool = false;


pub fn gatherQmvGateUpEnabled() bool {
    return gqmv_gateup_override orelse true;
}

/// One kernel for `silu(gather(gate) @ x) * (gather(up) @ x)` at decode width.
/// Returns [TOPK, N] in x's dtype, or null when anything is outside the
/// supported set (caller falls back to the two separate gathers + activation).
pub fn gatherQmvGateUp(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gw: mlx.mlx_array,
    gs_arr: mlx.mlx_array,
    gb: mlx.mlx_array,
    uw: mlx.mlx_array,
    us_arr: mlx.mlx_array,
    ub: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
    /// A whole-number `swiglu_limit` the activation clamps at (0 = none).
    limit: u32,
) !?mlx.mlx_array {
    if (!gatherQmvGateUpEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    const nvfp4 = switch (mode) {
        .affine => false,
        .nvfp4 => true,
        else => return null,
    };
    if (nvfp4) {
        if (bits != 4 or group_size != 16) return null;
        if (mlx.mlx_array_dtype(gs_arr) != .uint8 or mlx.mlx_array_dtype(us_arr) != .uint8) return null;
        if (gb.ctx != null or ub.ctx != null) return null;
    } else {
        if (!affineQmvUnitOk(bits, group_size)) return null;
        if (gb.ctx == null or ub.ctx == null) return null;
    }
    if (gs_arr.ctx == null or us_arr.ctx == null) return null;
    // The activation is a 16-bit table lookup, exactly as `fusedSwiGLU` gates
    // it; f32 activations keep the unfused path so the two can never disagree.
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    // gate and up must be the SAME geometry — the kernel walks one index set
    // through both banks.
    const wsh = mlx.getShape(gw);
    const ush = mlx.getShape(uw);
    if (wsh.len != 3 or ush.len != 3) return null;
    for (wsh, ush) |a, b| if (a != b) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    if (xelems != K) return null;
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    const key = GateUpCfgKey{ .topk = topk, .n = N, .k = K, .bits = bits, .gs = group_size, .dtype = xd, .limit = limit };
    if (gateup_cfg == null or !std.meta.eql(gateup_cfg_key, key)) {
        if (gateup_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, N, topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KP", affineQmvPackUnits(K, bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "LIMIT", @intCast(limit)));
        gateup_cfg = config;
        gateup_cfg_key = key;
    }

    const sigtab = try swigluSigTable(s, xd, std.heap.c_allocator);
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const affine_inputs = [_]mlx.mlx_array{ x, gw, gs_arr, gb, uw, us_arr, ub, inds, sigtab, K_arr, N_arr };
    const nvfp4_inputs = [_]mlx.mlx_array{ x, gw, gs_arr, uw, us_arr, inds, sigtab, K_arr, N_arr };
    const inputs_arr: []const mlx.mlx_array = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvGateUpKernel(nvfp4);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gateup_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    if (!gqmv_gateup_engaged) {
        gqmv_gateup_engaged = true;
        log.info("[moe] fused gate+up+SwiGLU kernel engaged: E_topk={d} inter={d} bits={d} gs={d}\n", .{ topk, N, bits, group_size });
    }
    return y;
}

// Indexed [nvfp4][x_per_expert].
pub const GQMV_SOURCES = [2][2][:0]const u8{
    .{ gatherQmvSource(false, false), gatherQmvSource(false, true) },
    .{ gatherQmvSource(true, false), gatherQmvSource(true, true) },
};
const GQMV_NAMES = [2][2][*:0]const u8{
    .{ "mlxserve_moe_gather_qmv", "mlxserve_moe_gather_qmv_px" },
    .{ "mlxserve_moe_gather_qmv_nvfp4", "mlxserve_moe_gather_qmv_nvfp4_px" },
};
var gqmv_kernels: [2][2]?mlx.mlx_fast_metal_kernel = @splat(@splat(null));

fn getGatherQmvKernel(nvfp4: bool, x_per_expert: bool) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @intFromBool(nvfp4);
    const xi: usize = @intFromBool(x_per_expert);
    if (gqmv_kernels[mi][xi]) |k| return k;
    // NVFP4 carries no biases, so that input is absent from its signature
    // rather than passed as a dummy — MLX builds the signature from the input
    // list, and a phantom array would be one more buffer bound per dispatch.
    const affine_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "K_size", "N_size" };
    const nvfp4_inputs = [_][*:0]const u8{ "x", "w_q", "scales", "inds", "K_size", "N_size" };
    const input_names: []const [*:0]const u8 = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        GQMV_NAMES[mi][xi],
        in_vec,
        out_vec,
        GQMV_SOURCES[mi][xi],
        if (nvfp4) GQMV_NVFP4_HEADER else GQMV_AFFINE_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gqmv_kernels[mi][xi] = kernel;
    return kernel;
}

/// Everything a `gatherQmv` config bakes in; one slot per input layout, since
/// the gate/up (shared-x) and down (per-expert-x) calls alternate every layer.
const GqmvCfgKey = struct {
    topk: c_int,
    n: c_int,
    bits: u32,
    gs: u32,
    dtype: mlx.mlx_dtype,
    x_per_expert: bool,
    act2: bool = false,
};
const GqmvCfgSlot = struct {
    cfg: ?mlx.mlx_fast_metal_kernel_config = null,
    key: GqmvCfgKey = std.mem.zeroes(GqmvCfgKey),
};
var gqmv_cfg_cache: [2]GqmvCfgSlot = .{ .{}, .{} };

pub var gqmv_engaged: bool = false;
pub var decode_ladder_engaged: bool = false;

/// Decode-width gathered qmv over an expert bank, read in place.
/// `x`: [K] shared (x_per_expert=false) or [TOPK, K] (true), bf16/fp16.
/// `w`/`sc`/`bi`: the affine bank, [E, N, K*BITS/32] / [E, N, K/GS].
/// `inds`: [TOPK] uint32 bank rows. Returns [TOPK, N], or null when the shape
/// or quant geometry is outside the supported set (caller falls back).
pub fn gatherQmv(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
    x_per_expert: bool,
) !?mlx.mlx_array {
    return gatherQmvAct(s, x, w, sc, bi, inds, bits, group_size, mode, x_per_expert, false);
}

/// `gatherQmv` with an optional ReLU^2 on the rounded output (`act_relu2`):
/// `T(max(T(acc), 0)^2)`, the composed chain's bits.
pub fn gatherQmvAct(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    inds: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    mode: QuantMode,
    x_per_expert: bool,
    act_relu2: bool,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    // Two dequant arithmetics, one kernel shape. Any other mode (mxfp4/mxfp8,
    // whose scales are e8m0 over different group sizes) falls back — running a
    // bank through the wrong dequant is silently wrong, not a crash.
    const nvfp4 = switch (mode) {
        .affine => false,
        .nvfp4 => true,
        else => return null,
    };
    if (nvfp4) {
        // NVFP4 is bits 4 / group 16 by definition, with uint8 e4m3 scales and
        // no biases; anything else claiming the mode is not a bank we can read.
        if (bits != 4 or group_size != 16) return null;
        if (mlx.mlx_array_dtype(sc) != .uint8) return null;
        if (bi.ctx != null) return null;
    } else {
        // Only widths where one uint32 holds a whole number of values (5/6-bit
        // affine is byte-packed differently — those fall back).
        if (!affineQmvUnitOk(bits, group_size)) return null;
        if (bi.ctx == null) return null;
    }
    if (sc.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16 and xd != .float32) return null;
    if (mlx.mlx_array_dtype(inds) != .uint32) return null;

    const wsh = mlx.getShape(w);
    if (wsh.len != 3) return null;
    const N: c_int = wsh[1];
    const K: c_int = @intCast(@divExact(@as(u32, @intCast(wsh[2])) * 32, bits));
    if (@rem(K, @as(c_int, @intCast(group_size))) != 0) return null;
    const ish = mlx.getShape(inds);
    if (ish.len != 1) return null;
    const topk: c_int = ish[0];
    const xsh = mlx.getShape(x);
    var xelems: i64 = 1;
    for (xsh) |d| xelems *= d;
    const want: i64 = if (x_per_expert) @as(i64, topk) * K else K;
    if (xelems != want) return null;

    // One simdgroup per (slot, row); 8 simdgroups to a threadgroup.
    const sgs_per_tg: c_int = 8;
    if (@rem(N, sgs_per_tg) != 0) return null;

    // CACHED, not rebuilt per call. This runs three times per MoE layer, and a
    // fresh config is ~10 FFI calls: measured `ops/forward` 3226 vs 2329 and
    // +0.4 ms of CPU graph build per token on Laguna XS, which is roughly 60%
    // of the margin this path was being charged against stock `gather_qmm`.
    const key = GqmvCfgKey{ .topk = topk, .n = N, .bits = bits, .gs = group_size, .dtype = xd, .x_per_expert = x_per_expert, .act2 = act_relu2 };
    const slot = &gqmv_cfg_cache[@intFromBool(x_per_expert)];
    if (slot.cfg == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const y_shape = [_]c_int{ topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &y_shape, 2, xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, N, topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgs_per_tg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", xd));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(group_size)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(bits)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ACT2", act_relu2));
        slot.* = .{ .cfg = cfg, .key = key };
    }
    const config = slot.cfg.?;

    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);
    const affine_inputs = [_]mlx.mlx_array{ x, w, sc, bi, inds, K_arr, N_arr };
    const nvfp4_inputs = [_]mlx.mlx_array{ x, w, sc, inds, K_arr, N_arr };
    const inputs_arr: []const mlx.mlx_array = if (nvfp4) &nvfp4_inputs else &affine_inputs;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getGatherQmvKernel(nvfp4, x_per_expert);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs_vec, 0));
    return y;
}

pub var gqmv_gateup_override: ?bool = null;

// ── tests ──

test "fused gate+up+SwiGLU expert kernel is bit-identical to the split gatherQmv path" {
    // The split path is `gatherQmv(gate)` + `gatherQmv(up)` + `fusedSwiGLU`.
    // Fusing them may not change a single rounding: each accumulator is still
    // rounded to T before the activation, and the activation is still the
    // table. Covers both dequant arithmetics and both live top-K widths.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6A7E0FF5);
    const rnd = prng.random();
    gqmv_gateup_override = true;
    defer gqmv_gateup_override = null;
    fused_activations.swiglu_fused_override = true;
    defer fused_activations.swiglu_fused_override = null;

    const E: c_int = 8;
    const N: c_int = 64;
    const K: c_int = 256;
    const TOPK: c_int = 3;

    for ([_]struct { nvfp4: bool, bits: u32, gs: u32 }{
        .{ .nvfp4 = false, .bits = 4, .gs = 64 },
        .{ .nvfp4 = true, .bits = 4, .gs = 16 },
        .{ .nvfp4 = false, .bits = 3, .gs = 128 },
    }) |cfg| {
        const nvfp4 = cfg.nvfp4;
        const bits = cfg.bits;
        const gs = cfg.gs;
        const mode: QuantMode = if (nvfp4) .nvfp4 else .affine;

        // Two independent banks, so a kernel that read the gate bank twice
        // could not pass.
        var banks: [2]struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array } = undefined;
        var made: usize = 0;
        defer for (banks[0..made]) |b| {
            _ = mlx.mlx_array_free(b.w);
            _ = mlx.mlx_array_free(b.sc);
            if (b.bi.ctx != null) _ = mlx.mlx_array_free(b.bi);
        };
        for (0..2) |bi_idx| {
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ E, N, K }, s);
            defer _ = mlx.mlx_array_free(wb);
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), if (nvfp4) "nvfp4" else "affine", .{}, s));
            var wq = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            var sc = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&sc, triple, 1));
            var bias = mlx.mlx_array{ .ctx = null };
            if (!nvfp4) {
                bias = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(&bias, triple, 2));
            }
            banks[bi_idx] = .{ .w = wq, .sc = sc, .bi = bias };
            made += 1;
        }

        const xbuf = try allocator.alloc(f32, @intCast(K));
        defer allocator.free(xbuf);
        for (xbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        const xsh = [_]c_int{K};
        const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 1, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

        var idx: [3]u32 = .{ 5, 0, 3 };
        const ish = [_]c_int{TOPK};
        const inds = mlx.mlx_array_new_data(&idx, &ish, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds);

        // Reference: the two separate gathers, then the fused activation.
        const g2 = (try gatherQmv(s, x, banks[0].w, banks[0].sc, banks[0].bi, inds, bits, gs, mode, false)) orelse
            return error.GatherQmvDeclined;
        defer _ = mlx.mlx_array_free(g2);
        const up2 = (try gatherQmv(s, x, banks[1].w, banks[1].sc, banks[1].bi, inds, bits, gs, mode, false)) orelse
            return error.GatherQmvDeclined;
        defer _ = mlx.mlx_array_free(up2);
        const ref = (try fusedSwiGLU(s, g2, up2)) orelse return error.FusedSwigluDeclined;
        defer _ = mlx.mlx_array_free(ref);

        const got = (try gatherQmvGateUp(s, x, banks[0].w, banks[0].sc, banks[0].bi, banks[1].w, banks[1].sc, banks[1].bi, inds, bits, gs, mode, 0)) orelse
            return error.FusedGateUpDeclined;
        defer _ = mlx.mlx_array_free(got);
        // A swiglu_limit of 1 clamps these magnitudes: bit for bit clampedSwiGLU over the split gathers.
        const clamped = (try gatherQmvGateUp(s, x, banks[0].w, banks[0].sc, banks[0].bi, banks[1].w, banks[1].sc, banks[1].bi, inds, bits, gs, mode, 1)) orelse
            return error.FusedGateUpDeclined;
        defer _ = mlx.mlx_array_free(clamped);
        const clamp_ref = try clampedSwiGLU(s, g2, up2, 1.0);
        defer _ = mlx.mlx_array_free(clamp_ref);
        {
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, clamp_ref, clamped, false, s));
            try mlx.check(mlx.mlx_array_eval(eq));
            var eq_v = false;
            try mlx.check(mlx.mlx_array_item_bool(&eq_v, eq));
            try testing.expect(eq_v);
        }

        const cnt: usize = @intCast(TOPK * N);
        const ref_host = try allocator.alloc(f32, cnt);
        defer allocator.free(ref_host);
        const got_host = try allocator.alloc(f32, cnt);
        defer allocator.free(got_host);
        try testReadF32(ref, ref_host, s);
        try testReadF32(got, got_host, s);
        try testing.expectEqualSlices(f32, ref_host, got_host);
    }
}

test "moe decode: the CUDA fused arm takes fp banks with plain SiLU, and declines what it does not apply" {
    try std.testing.expectEqual(@as(?@import("moe_cuda.zig").Format, .nvfp4), moeCudaFormat(1, .nvfp4, .nvfp4, .nvfp4, true, false, false, false));
    try std.testing.expectEqual(@as(?@import("moe_cuda.zig").Format, .mxfp8), moeCudaFormat(8, .mxfp8, .mxfp8, .mxfp8, true, false, false, false));
    try std.testing.expect(moeCudaFormat(17, .nvfp4, .nvfp4, .nvfp4, true, false, false, false) == null);
    try std.testing.expect(moeCudaFormat(1, .affine, .affine, .affine, true, false, false, false) == null);
    try std.testing.expect(moeCudaFormat(1, .nvfp4, .nvfp4, .mxfp8, true, false, false, false) == null);
    try std.testing.expect(moeCudaFormat(1, .nvfp4, .nvfp4, .nvfp4, false, false, false, false) == null);
    try std.testing.expect(moeCudaFormat(1, .nvfp4, .nvfp4, .nvfp4, true, true, false, false) == null);
    try std.testing.expect(moeCudaFormat(1, .nvfp4, .nvfp4, .nvfp4, true, false, true, false) == null);
    try std.testing.expect(moeCudaFormat(1, .nvfp4, .nvfp4, .nvfp4, true, false, false, true) == null);
}

/// Max |a - ref| as f32, for the gatherQmv no-worse-than-reference parity tests.
fn gqmvMaxAbsErr(a: mlx.mlx_array, ref: mlx.mlx_array, str: mlx.mlx_stream) !f32 {
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    try mlx.check(mlx.mlx_astype(&a32, a, .float32, str));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_subtract(&d, a32, ref, str));
    var ad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ad);
    try mlx.check(mlx.mlx_abs(&ad, d, str));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ad, false, str));
    try mlx.check(mlx.mlx_array_eval(mx));
    var out: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&out, mx));
    return out;
}

test "gatherQmv is no worse than stock gather_qmm vs fp32 dequant ground truth" {
    // Per the kernel-testing rule: assert NO-WORSE-THAN-REFERENCE against fp32
    // dequant ground truth, never kernel-vs-kernel agreement (both round to
    // bf16 in different orders, so exact agreement is not a real invariant).
    // Covers BOTH input layouts: the shared-x gate/up projection and the
    // per-expert-x down projection.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9E3779B9);
    const rnd = prng.random();

    const E: c_int = 8;
    const N: c_int = 64;
    const K: c_int = 256;
    const TOPK: c_int = 3;

    for ([_]struct { bits: u32, gs: u32 }{
        .{ .bits = 2, .gs = 64 },
        .{ .bits = 3, .gs = 64 },
        .{ .bits = 3, .gs = 128 },
        .{ .bits = 4, .gs = 64 },
        .{ .bits = 8, .gs = 32 },
    }) |cfg| {
        // fp32 is the LIVE Laguna case: expert_x arrives as float32 while the
        // bank's scales/biases are bf16. A bf16-only test missed it entirely.
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |xdt| {
            for ([_]bool{ false, true }) |x_per_expert| {
                // ── quantized bank [E,N,K] ──
                const wb = try testRandWeightBf16(rnd, &[_]c_int{ E, N, K }, s);
                defer _ = mlx.mlx_array_free(wb);
                var triple = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(triple);
                try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", .{}, s));
                var wq = mlx.mlx_array_new();
                var wsc = mlx.mlx_array_new();
                var wbi = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wq);
                defer _ = mlx.mlx_array_free(wsc);
                defer _ = mlx.mlx_array_free(wbi);
                try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
                try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
                try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

                // ── x: [K] shared, or [TOPK,K] per expert ──
                const xrows: c_int = if (x_per_expert) TOPK else 1;
                const xcnt: usize = @intCast(xrows * K);
                const xbuf = try allocator.alloc(f32, xcnt);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
                const xsh = [_]c_int{ xrows, K };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 2, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, xdt, s));

                const idxvals = [_]u32{ 5, 0, 3 };
                const idxsh = [_]c_int{TOPK};
                const inds = mlx.mlx_array_new_data(&idxvals, &idxsh, 1, .uint32);
                defer _ = mlx.mlx_array_free(inds);

                // ── fp32 ground truth: dequantize the bank, take the 3 experts,
                // and do the matmul entirely in float32.
                var deq = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(deq);
                try mlx.check(mlx.mlx_dequantize(&deq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", .{}, .{}, s));
                var deq32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(deq32);
                try mlx.check(mlx.mlx_astype(&deq32, deq, .float32, s));
                var sel = mlx.mlx_array_new(); // [TOPK,N,K]
                defer _ = mlx.mlx_array_free(sel);
                try mlx.check(mlx.mlx_take_axis(&sel, deq32, inds, 0, s));
                var selT = mlx.mlx_array_new(); // [TOPK,K,N]
                defer _ = mlx.mlx_array_free(selT);
                const tax = [_]c_int{ 0, 2, 1 };
                try mlx.check(mlx.mlx_transpose_axes(&selT, sel, &tax, 3, s));
                // x as [TOPK,1,K] (broadcast the shared row when needed)
                var x32r = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x32r);
                const x3 = [_]c_int{ xrows, 1, K };
                try mlx.check(mlx.mlx_reshape(&x32r, x32, &x3, 3, s));
                var x32b = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x32b);
                const xb3 = [_]c_int{ TOPK, 1, K };
                try mlx.check(mlx.mlx_broadcast_to(&x32b, x32r, &xb3, 3, s));
                var truth = mlx.mlx_array_new(); // [TOPK,1,N]
                defer _ = mlx.mlx_array_free(truth);
                try mlx.check(mlx.mlx_matmul(&truth, x32b, selT, s));
                const tflat = [_]c_int{ TOPK, N };
                var truth2 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(truth2);
                try mlx.check(mlx.mlx_reshape(&truth2, truth, &tflat, 2, s));

                // ── stock gather_qmm, same layouts ──
                var xg = mlx.mlx_array_new(); // [TOPK,1,K]
                defer _ = mlx.mlx_array_free(xg);
                {
                    var xr = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(xr);
                    try mlx.check(mlx.mlx_reshape(&xr, x, &x3, 3, s));
                    try mlx.check(mlx.mlx_broadcast_to(&xg, xr, &xb3, 3, s));
                }
                var stock3 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(stock3);
                const lhs3 = [_]u32{ 0, 1, 2 };
                const lhs_used = if (x_per_expert) blk: {
                    break :blk mlx.mlx_array_new_data(&lhs3, &idxsh, 1, .uint32);
                } else blk: {
                    const zeros = [_]u32{ 0, 0, 0 };
                    break :blk mlx.mlx_array_new_data(&zeros, &idxsh, 1, .uint32);
                };
                defer _ = mlx.mlx_array_free(lhs_used);
                try mlx.check(mlx.mlx_gather_qmm(&stock3, xg, wq, wsc, wbi, lhs_used, inds, true, mlx.mlx_optional_int.some(@intCast(cfg.gs)), mlx.mlx_optional_int.some(@intCast(cfg.bits)), "affine", false, s));
                var stock2 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(stock2);
                try mlx.check(mlx.mlx_reshape(&stock2, stock3, &tflat, 2, s));

                // ── ours ──
                const maybe = try gatherQmv(s, x, wq, wsc, wbi, inds, cfg.bits, cfg.gs, .affine, x_per_expert);
                try testing.expect(maybe != null);
                const ours = maybe.?;
                defer _ = mlx.mlx_array_free(ours);

                const err_ours = try gqmvMaxAbsErr(ours, truth2, s);
                const err_stock = try gqmvMaxAbsErr(stock2, truth2, s);
                // No worse than stock against the same fp32 ground truth (small
                // slack for bf16 accumulation-order differences).
                testing.expect(err_ours <= err_stock * 1.05 + 1e-3) catch |e| {
                    std.debug.print("\n[gather-qmv] bits={d} gs={d} xdt={any} per_expert_x={} ours_err={d:.6} stock_err={d:.6}\n", .{ cfg.bits, cfg.gs, xdt, x_per_expert, err_ours, err_stock });
                    return e;
                };
            }
        }
    }
}

test "gatherQmv nvfp4 is no worse than stock gather_qmm vs fp32 dequant ground truth" {
    // Laguna's upstream checkpoints ship NVFP4 routed experts, so the affine-only
    // gatherQmv declined them and decode fell back to take + batched
    // quantized_matmul (~3x the ideal traffic). Same no-worse-than-reference
    // discipline as the affine test: fp32 dequant ground truth, never
    // kernel-vs-kernel agreement.
    //
    // NVFP4 is bits=4 / group_size=16 by definition: packed-u32 e2m1 weights,
    // uint8 e4m3 scales, NO biases.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5BD1E995);
    const rnd = prng.random();

    const E: c_int = 8;
    const N: c_int = 64;
    const K: c_int = 256;
    const TOPK: c_int = 3;
    const null_bi = mlx.mlx_array{ .ctx = null };

    // fp32 x is the LIVE Laguna case (expert activations arrive float32 while
    // the bank's scales are uint8); bf16 covers the shared-x gate/up shape.
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |xdt| {
        for ([_]bool{ false, true }) |x_per_expert| {
            // ── nvfp4 bank [E,N,K] ──
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ E, N, K }, s);
            defer _ = mlx.mlx_array_free(wb);
            var pair = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(pair);
            try mlx.check(mlx.mlx_quantize(&pair, wb, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{}, s));
            var wq = mlx.mlx_array_new();
            var wsc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            defer _ = mlx.mlx_array_free(wsc);
            try mlx.check(mlx.mlx_vector_array_get(&wq, pair, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, pair, 1));

            // ── x: [K] shared, or [TOPK,K] per expert ──
            const xrows: c_int = if (x_per_expert) TOPK else 1;
            const xcnt: usize = @intCast(xrows * K);
            const xbuf = try allocator.alloc(f32, xcnt);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xsh = [_]c_int{ xrows, K };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 2, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, xdt, s));

            const idxvals = [_]u32{ 5, 0, 3 };
            const idxsh = [_]c_int{TOPK};
            const inds = mlx.mlx_array_new_data(&idxvals, &idxsh, 1, .uint32);
            defer _ = mlx.mlx_array_free(inds);

            // ── fp32 ground truth: dequantize the bank in float32, take the 3
            // experts, matmul entirely in float32.
            var deq32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(deq32);
            try mlx.check(mlx.mlx_dequantize(&deq32, wq, wsc, null_bi, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
            var sel = mlx.mlx_array_new(); // [TOPK,N,K]
            defer _ = mlx.mlx_array_free(sel);
            try mlx.check(mlx.mlx_take_axis(&sel, deq32, inds, 0, s));
            var selT = mlx.mlx_array_new(); // [TOPK,K,N]
            defer _ = mlx.mlx_array_free(selT);
            const tax = [_]c_int{ 0, 2, 1 };
            try mlx.check(mlx.mlx_transpose_axes(&selT, sel, &tax, 3, s));
            var x32r = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32r);
            const x3 = [_]c_int{ xrows, 1, K };
            try mlx.check(mlx.mlx_reshape(&x32r, x32, &x3, 3, s));
            var x32b = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x32b);
            const xb3 = [_]c_int{ TOPK, 1, K };
            try mlx.check(mlx.mlx_broadcast_to(&x32b, x32r, &xb3, 3, s));
            var truth = mlx.mlx_array_new(); // [TOPK,1,N]
            defer _ = mlx.mlx_array_free(truth);
            try mlx.check(mlx.mlx_matmul(&truth, x32b, selT, s));
            const tflat = [_]c_int{ TOPK, N };
            var truth2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(truth2);
            try mlx.check(mlx.mlx_reshape(&truth2, truth, &tflat, 2, s));

            // ── stock gather_qmm, same layouts ──
            var xg = mlx.mlx_array_new(); // [TOPK,1,K]
            defer _ = mlx.mlx_array_free(xg);
            {
                var xr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xr);
                try mlx.check(mlx.mlx_reshape(&xr, x, &x3, 3, s));
                try mlx.check(mlx.mlx_broadcast_to(&xg, xr, &xb3, 3, s));
            }
            var stock3 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stock3);
            const lhs3 = [_]u32{ 0, 1, 2 };
            const lhs_used = if (x_per_expert) blk: {
                break :blk mlx.mlx_array_new_data(&lhs3, &idxsh, 1, .uint32);
            } else blk: {
                const zeros = [_]u32{ 0, 0, 0 };
                break :blk mlx.mlx_array_new_data(&zeros, &idxsh, 1, .uint32);
            };
            defer _ = mlx.mlx_array_free(lhs_used);
            try mlx.check(mlx.mlx_gather_qmm(&stock3, xg, wq, wsc, null_bi, lhs_used, inds, true, mlx.mlx_optional_int.some(16), mlx.mlx_optional_int.some(4), "nvfp4", false, s));
            var stock2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stock2);
            try mlx.check(mlx.mlx_reshape(&stock2, stock3, &tflat, 2, s));

            // ── ours ──
            const maybe = try gatherQmv(s, x, wq, wsc, null_bi, inds, 4, 16, .nvfp4, x_per_expert);
            try testing.expect(maybe != null);
            const ours = maybe.?;
            defer _ = mlx.mlx_array_free(ours);

            const err_ours = try gqmvMaxAbsErr(ours, truth2, s);
            const err_stock = try gqmvMaxAbsErr(stock2, truth2, s);
            testing.expect(err_ours <= err_stock * 1.05 + 1e-3) catch |e| {
                std.debug.print("\n[gather-qmv-nvfp4] xdt={any} per_expert_x={} ours_err={d:.6} stock_err={d:.6}\n", .{ xdt, x_per_expert, err_ours, err_stock });
                return e;
            };
        }
    }
}

test "MoE decode gather µbench (diag moe-gather-ubench)" {
    // Reproduces the Laguna decode expert-gather with OUR self-built MLX, to
    // compare against the pip-MLX Python sim (single gather ~133us, 47-layer
    // MoE ~7.5ms). If this is far slower, the gap is our MLX build/runtime;
    // if it matches, the gap is the live server context (stream/alloc/pipeline).
    if (env.get(.MOE_GATHER_UBENCH) == null) return error.SkipZigTest;
    const io_util = @import("../../util/io_util.zig");
    const tio = testing.io;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    // Trace mode: run the gather and qmv sections in long sustained bursts so an
    // Instruments Metal System Trace shows a clean, comparable GPU timeline.
    const gather_trace = env.get(.MOE_GATHER_TRACE) != null;
    const E: c_int = 256;
    const N: c_int = 1024; // moe_intermediate (gate/up out)
    const K: c_int = 3072; // hidden
    const TOPK: c_int = 10;
    const GS = 64;
    const BITS = 2;

    // Build a quantized [E,N,K] gate/up bank and a [E,K,N] down bank.
    const makeBank = struct {
        fn go(a: std.mem.Allocator, e: c_int, out: c_int, in: c_int, str: mlx.mlx_stream, rndp: *std.Random.DefaultPrng) ![3]mlx.mlx_array {
            const cnt: usize = @intCast(e * out * in);
            const buf = try a.alloc(f32, cnt);
            defer a.free(buf);
            const rr = rndp.random();
            for (buf) |*v| v.* = rr.float(f32) - 0.5;
            const wsh = [_]c_int{ e, out, in };
            const w32 = mlx.mlx_array_new_data(buf.ptr, &wsh, 3, .float32);
            defer _ = mlx.mlx_array_free(w32);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, str));
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", .{}, str));
            var wq = mlx.mlx_array_new();
            var wsc = mlx.mlx_array_new();
            var wbi = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
            for ([_]mlx.mlx_array{ wq, wsc, wbi }) |aa| try mlx.check(mlx.mlx_array_eval(aa));
            return .{ wq, wsc, wbi };
        }
    }.go;

    var prng = std.Random.DefaultPrng.init(0xF00D);
    const gate = try makeBank(allocator, E, N, K, s, &prng);
    const down = try makeBank(allocator, E, K, N, s, &prng);
    defer for (gate ++ down) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // x [1,1,3072], inds [1,1,10]
    const xbuf = try allocator.alloc(f32, @intCast(K));
    defer allocator.free(xbuf);
    for (xbuf) |*v| v.* = prng.random().float(f32) - 0.5;
    const xsh = [_]c_int{ 1, 1, K };
    const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xsh, 3, .float32);
    defer _ = mlx.mlx_array_free(x32);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
    const idxvals = [_]i32{ 0, 5, 9, 13, 40, 77, 120, 200, 3, 88 };
    const idxsh = [_]c_int{ 1, 1, TOPK };
    const inds = mlx.mlx_array_new_data(&idxvals, &idxsh, 3, .int32);
    defer _ = mlx.mlx_array_free(inds);
    const no_idx = mlx.mlx_array{ .ctx = null };

    _ = no_idx;
    // One decode gather at our 5-D shape, optionally biasing x by `add` to
    // defeat common-subexpression dedup when many are built in one graph.
    const oneGather = struct {
        fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, bank: [3]mlx.mlx_array, ind: mlx.mlx_array, add: f32) !mlx.mlx_array {
            const sh5 = [_]c_int{ 1, 1, 1, 1, K };
            var xe = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xe);
            try mlx.check(mlx.mlx_reshape(&xe, xin, &sh5, 5, str));
            if (add != 0.0) {
                const sc = mlx.mlx_array_new_float(add);
                defer _ = mlx.mlx_array_free(sc);
                var xa = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&xa, xe, sc, str));
                _ = mlx.mlx_array_free(xe);
                xe = xa;
            }
            var out5 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_gather_qmm(&out5, xe, bank[0], bank[1], bank[2], .{ .ctx = null }, ind, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", false, str));
            return out5;
        }
    }.go;

    var sw = io_util.Stopwatch.init(tio);
    // (1) single gather, 400 iters (cached) — compare to pip-MLX Python 133us
    {
        var i: usize = 0;
        while (i < 20) : (i += 1) {
            const g = try oneGather(s, x, gate, inds, 0.0);
            try mlx.check(mlx.mlx_array_eval(g));
            _ = mlx.mlx_array_free(g);
        }
        sw.reset();
        const ITER = 400;
        i = 0;
        while (i < ITER) : (i += 1) {
            const g = try oneGather(s, x, gate, inds, 0.0);
            try mlx.check(mlx.mlx_array_eval(g));
            _ = mlx.mlx_array_free(g);
        }
        const us = @as(f64, @floatFromInt(sw.read())) / 1000.0 / @as(f64, @floatFromInt(ITER));
        std.debug.print("\n[moe-gather-ubench] single gather_qmm (our MLX): {d:.1} us/call\n", .{us});
    }
    // (2) 141 INDEPENDENT gathers (varied x → no CSE) in ONE eval — measures
    // how well our MLX overlaps them (pip-MLX 47-layer MoE sim was ~7.5ms).
    {
        const NG = 141;
        var warm: usize = 0;
        while (warm < 3) : (warm += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneGather(s, x, gate, inds, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        sw.reset();
        const ITER: usize = if (gather_trace) 200 else 20;
        var i: usize = 0;
        while (i < ITER) : (i += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneGather(s, x, gate, inds, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
        std.debug.print("[moe-gather-ubench] 141 independent gathers, one eval (our MLX): {d:.2} ms  ({d:.1} us/gather)\n", .{ ms, ms * 1000.0 / @as(f64, NG) });
    }
    // (3) 141 INDEPENDENT quantized_matmul (dense qmv, NOT gather) in one eval —
    // does plain qmv overlap in our build where gather does not? If yes, an
    // engine-level workaround (extract experts + qmv) can dodge the gather bug.
    {
        // one 2D quantized weight [N, K] for qmv (transpose=true → x@w^T)
        const wn2: usize = @intCast(N * K);
        const wbuf2 = try allocator.alloc(f32, wn2);
        defer allocator.free(wbuf2);
        for (wbuf2) |*v| v.* = prng.random().float(f32) - 0.5;
        const wsh2 = [_]c_int{ N, K };
        const w2_32 = mlx.mlx_array_new_data(wbuf2.ptr, &wsh2, 2, .float32);
        defer _ = mlx.mlx_array_free(w2_32);
        var w2b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w2b);
        try mlx.check(mlx.mlx_astype(&w2b, w2_32, .bfloat16, s));
        var trip = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(trip);
        try mlx.check(mlx.mlx_quantize(&trip, w2b, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", .{}, s));
        var q = mlx.mlx_array_new();
        var sc = mlx.mlx_array_new();
        var bi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q);
        defer _ = mlx.mlx_array_free(sc);
        defer _ = mlx.mlx_array_free(bi);
        try mlx.check(mlx.mlx_vector_array_get(&q, trip, 0));
        try mlx.check(mlx.mlx_vector_array_get(&sc, trip, 1));
        try mlx.check(mlx.mlx_vector_array_get(&bi, trip, 2));
        for ([_]mlx.mlx_array{ q, sc, bi }) |a| try mlx.check(mlx.mlx_array_eval(a));

        const oneQmv = struct {
            fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, wq_: mlx.mlx_array, wsc_: mlx.mlx_array, wbi_: mlx.mlx_array, add: f32) !mlx.mlx_array {
                var xe = mlx.mlx_array_new();
                if (add != 0.0) {
                    const scl = mlx.mlx_array_new_float(add);
                    defer _ = mlx.mlx_array_free(scl);
                    try mlx.check(mlx.mlx_add(&xe, xin, scl, str));
                } else {
                    try mlx.check(mlx.mlx_astype(&xe, xin, .bfloat16, str));
                }
                defer _ = mlx.mlx_array_free(xe);
                var out = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(&out, xe, wq_, wsc_, wbi_, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", str));
                return out;
            }
        }.go;

        const NG = 141;
        var warm: usize = 0;
        while (warm < 3) : (warm += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneQmv(s, x, q, sc, bi, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        sw.reset();
        const ITER: usize = if (gather_trace) 200 else 20;
        var i: usize = 0;
        while (i < ITER) : (i += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneQmv(s, x, q, sc, bi, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
        std.debug.print("[moe-gather-ubench] 141 independent quantized_matmul (qmv), one eval (our MLX): {d:.2} ms  ({d:.1} us/qmv)\n", .{ ms, ms * 1000.0 / @as(f64, NG) });
    }
    // (4) FULL WORKAROUND: take 10 experts from the [256,N,K] bank + broadcast x
    // + BATCHED quantized_matmul. Includes the extraction cost. If fast, this is
    // the decode drop-in for gather_qmm.
    {
        const idx10 = [_]i32{ 0, 5, 9, 13, 40, 77, 120, 200, 3, 88 };
        const idx10sh = [_]c_int{10};
        const inds10 = mlx.mlx_array_new_data(&idx10, &idx10sh, 1, .int32);
        defer _ = mlx.mlx_array_free(inds10);
        // take from the real 256-expert `gate` bank each call (extraction cost)
        const oneBatched = struct {
            fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, bank: [3]mlx.mlx_array, ind: mlx.mlx_array, add: f32) !mlx.mlx_array {
                var wq_ = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wq_);
                var wsc_ = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wsc_);
                var wbi_ = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wbi_);
                try mlx.check(mlx.mlx_take_axis(&wq_, bank[0], ind, 0, str));
                try mlx.check(mlx.mlx_take_axis(&wsc_, bank[1], ind, 0, str));
                try mlx.check(mlx.mlx_take_axis(&wbi_, bank[2], ind, 0, str));
                const sh3 = [_]c_int{ 1, 1, K };
                var xe = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_reshape(&xe, xin, &sh3, 3, str));
                if (add != 0.0) {
                    const scl = mlx.mlx_array_new_float(add);
                    defer _ = mlx.mlx_array_free(scl);
                    var xa = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&xa, xe, scl, str));
                    _ = mlx.mlx_array_free(xe);
                    xe = xa;
                }
                defer _ = mlx.mlx_array_free(xe);
                const bsh = [_]c_int{ 10, 1, K };
                var xbc = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xbc);
                try mlx.check(mlx.mlx_broadcast_to(&xbc, xe, &bsh, 3, str));
                var out = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_quantized_matmul(&out, xbc, wq_, wsc_, wbi_, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", str));
                return out; // [10,1,N]
            }
        }.go;

        const NG = 141;
        var warm: usize = 0;
        while (warm < 3) : (warm += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneBatched(s, x, gate, inds10, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        sw.reset();
        const ITER: usize = if (gather_trace) 200 else 20;
        var i: usize = 0;
        while (i < ITER) : (i += 1) {
            var acc = mlx.mlx_array_new();
            var first = true;
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = try oneBatched(s, x, gate, inds10, @as(f32, @floatFromInt(j)) * 0.001);
                if (first) {
                    acc = g;
                    first = false;
                } else {
                    var na = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&na, acc, g, s));
                    _ = mlx.mlx_array_free(acc);
                    _ = mlx.mlx_array_free(g);
                    acc = na;
                }
            }
            try mlx.check(mlx.mlx_array_eval(acc));
            _ = mlx.mlx_array_free(acc);
        }
        const ms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
        std.debug.print("[moe-gather-ubench] 141 BATCHED qmv over 10 experts, one eval (our MLX): {d:.2} ms  ({d:.1} us/batched)\n", .{ ms, ms * 1000.0 / @as(f64, NG) });
    }
    // (4b) OUR gatherQmv kernel: reads the bank IN PLACE with GPU-resident
    // indices, so it moves the ideal 9.8 MB instead of the batched path's 3x.
    // Target is pip-Python's gather number (~22 us), not the batched 70 us.
    {
        const idx10 = [_]u32{ 0, 5, 9, 13, 40, 77, 120, 200, 3, 88 };
        const idx10sh = [_]c_int{TOPK};
        const inds_u32 = mlx.mlx_array_new_data(&idx10, &idx10sh, 1, .uint32);
        defer _ = mlx.mlx_array_free(inds_u32);
        const xk = [_]c_int{K};
        var xflat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xflat);
        try mlx.check(mlx.mlx_reshape(&xflat, x, &xk, 1, s));
        try mlx.check(mlx.mlx_array_eval(xflat));

        const NG = 141;
        const ITER: usize = if (gather_trace) 200 else 20;
        var pass: usize = 0;
        while (pass < ITER + 1) : (pass += 1) {
            if (pass == 1) sw.reset();
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            var j: usize = 0;
            while (j < NG) : (j += 1) {
                const g = (try gatherQmv(s, xflat, gate[0], gate[1], gate[2], inds_u32, BITS, GS, .affine, false)).?;
                defer _ = mlx.mlx_array_free(g);
                try mlx.check(mlx.mlx_vector_array_append_value(vec, g));
            }
            try mlx.check(mlx.mlx_eval(vec));
        }
        const gms = @as(f64, @floatFromInt(sw.read())) / 1.0e6 / @as(f64, @floatFromInt(ITER));
        std.debug.print("[moe-gather-ubench] 141 OUR gatherQmv (in-place bank), one eval: {d:.2} ms  ({d:.1} us/gather)\n", .{ gms, gms * 1000.0 / @as(f64, NG) });
    }
    // (5) ISOLATION MATRIX. A solo gather is ~144 us but 141-in-one-graph is
    // ~349 us EACH — batching makes each gather SLOWER than running it alone,
    // so this is not merely "fails to overlap". Two candidate serializers, each
    // testable independently (2x2):
    //   - add-chain: the accumulate in (2) is a serial dependency chain, and
    //     every dependent op forces a FULL `memoryBarrier(BarrierScopeBuffers)`
    //     in MLX's concurrent command encoder (backend/metal/device.cpp
    //     maybeInsertBarrier) — one barrier per link drains the whole GPU.
    //   - implicit lhs_indices: `gather_qmm` with a null lhs runs
    //     `indices_or_default` (ops.cpp), which emits a per-call
    //     arange+reshape. The gather then READS that fresh arange output, so
    //     each gather is data-dependent on a kernel dispatched immediately
    //     before it => another guaranteed barrier per gather. Plain qmv has no
    //     index prep at all, which is exactly why it overlaps.
    // Passing an explicit lhs (broadcastable zeros) hoists the index array out
    // of the per-call graph and should remove the second barrier entirely.
    {
        const NG = 141;
        // Explicit lhs_indices, built ONCE and reused by every gather: uint32
        // zeros shaped [1,1,TOPK] so it broadcasts against rhs without work.
        const lhs_vals = [_]u32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }; // TOPK zeros
        const lhs_sh = [_]c_int{ 1, 1, TOPK };
        const lhs_idx = mlx.mlx_array_new_data(&lhs_vals, &lhs_sh, 3, .uint32);
        defer _ = mlx.mlx_array_free(lhs_idx);
        try mlx.check(mlx.mlx_array_eval(lhs_idx));

        const oneGatherLhs = struct {
            fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, bank: [3]mlx.mlx_array, lhs: mlx.mlx_array, ind: mlx.mlx_array, add: f32) !mlx.mlx_array {
                const sh5 = [_]c_int{ 1, 1, 1, 1, K };
                var xe = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xe);
                try mlx.check(mlx.mlx_reshape(&xe, xin, &sh5, 5, str));
                if (add != 0.0) {
                    const sc = mlx.mlx_array_new_float(add);
                    defer _ = mlx.mlx_array_free(sc);
                    var xa = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_add(&xa, xe, sc, str));
                    _ = mlx.mlx_array_free(xe);
                    xe = xa;
                }
                var out5 = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_gather_qmm(&out5, xe, bank[0], bank[1], bank[2], lhs, ind, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", false, str));
                return out5;
            }
        }.go;

        // Run one cell of the matrix: `explicit_lhs` x `chain` (add-chain vs a
        // flat vector eval of 141 independent results).
        const cell = struct {
            fn go(str: mlx.mlx_stream, xin: mlx.mlx_array, bank: [3]mlx.mlx_array, lhs: mlx.mlx_array, ind: mlx.mlx_array, explicit_lhs: bool, chain: bool, iters: usize, watch: *io_util.Stopwatch) !f64 {
                var pass: usize = 0;
                var total_ns: u64 = 0;
                // one warmup pass then `iters` timed passes
                while (pass < iters + 1) : (pass += 1) {
                    if (pass == 1) watch.reset();
                    var acc = mlx.mlx_array_new();
                    var first = true;
                    const vec = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(vec);
                    var j: usize = 0;
                    while (j < NG) : (j += 1) {
                        const bias = @as(f32, @floatFromInt(j)) * 0.001;
                        const g = if (explicit_lhs)
                            try oneGatherLhs(str, xin, bank, lhs, ind, bias)
                        else
                            try oneGather(str, xin, bank, ind, bias);
                        if (!chain) {
                            try mlx.check(mlx.mlx_vector_array_append_value(vec, g));
                            _ = mlx.mlx_array_free(g);
                            continue;
                        }
                        if (first) {
                            acc = g;
                            first = false;
                        } else {
                            var na = mlx.mlx_array_new();
                            try mlx.check(mlx.mlx_add(&na, acc, g, str));
                            _ = mlx.mlx_array_free(acc);
                            _ = mlx.mlx_array_free(g);
                            acc = na;
                        }
                    }
                    if (chain) {
                        try mlx.check(mlx.mlx_array_eval(acc));
                        _ = mlx.mlx_array_free(acc);
                    } else {
                        try mlx.check(mlx.mlx_eval(vec));
                    }
                }
                total_ns = watch.read();
                return @as(f64, @floatFromInt(total_ns)) / 1.0e6 / @as(f64, @floatFromInt(iters));
            }
        }.go;

        const ITER: usize = if (gather_trace) 200 else 20;
        std.debug.print("[moe-gather-ubench] --- isolation matrix (141 gathers, one eval) ---\n", .{});
        for ([_]bool{ false, true }) |explicit_lhs| {
            for ([_]bool{ true, false }) |chain| {
                const ms = try cell(s, x, gate, lhs_idx, inds, explicit_lhs, chain, ITER, &sw);
                std.debug.print("[moe-gather-ubench]   lhs={s:<8} accum={s:<9} {d:6.2} ms  ({d:.1} us/gather)\n", .{
                    if (explicit_lhs) "explicit" else "implicit",
                    if (chain) "add-chain" else "none",
                    ms,
                    ms * 1000.0 / @as(f64, NG),
                });
            }
        }

        // (6) FULLY INDEPENDENT: every input pre-materialized, so the 141
        // gathers have NO producer op in the graph at all — no add feeding x,
        // no arange feeding the indices, no accumulate consuming the outputs.
        // If these still fail to overlap, no data dependency (and therefore no
        // MLX barrier) is responsible and the cost is the kernel itself.
        // The qmv control runs on the SAME pre-materialized x list.
        {
            var xs: [NG]mlx.mlx_array = undefined;
            for (&xs, 0..) |*slot, j| {
                const sh5 = [_]c_int{ 1, 1, 1, 1, K };
                var xr = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xr);
                try mlx.check(mlx.mlx_reshape(&xr, x, &sh5, 5, s));
                const scl = mlx.mlx_array_new_float(@as(f32, @floatFromInt(j)) * 0.001);
                defer _ = mlx.mlx_array_free(scl);
                var xa = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_add(&xa, xr, scl, s));
                try mlx.check(mlx.mlx_array_eval(xa)); // materialize: no producer left
                slot.* = xa;
            }
            defer for (xs) |a| {
                _ = mlx.mlx_array_free(a);
            };

            const indep = struct {
                fn go(str: mlx.mlx_stream, xlist: []const mlx.mlx_array, bank: [3]mlx.mlx_array, lhs: mlx.mlx_array, ind: mlx.mlx_array, iters: usize, watch: *io_util.Stopwatch) !f64 {
                    var pass: usize = 0;
                    while (pass < iters + 1) : (pass += 1) {
                        if (pass == 1) watch.reset();
                        const vec = mlx.mlx_vector_array_new();
                        defer _ = mlx.mlx_vector_array_free(vec);
                        for (xlist) |xj| {
                            var out5 = mlx.mlx_array_new();
                            defer _ = mlx.mlx_array_free(out5);
                            try mlx.check(mlx.mlx_gather_qmm(&out5, xj, bank[0], bank[1], bank[2], lhs, ind, true, mlx.mlx_optional_int.some(GS), mlx.mlx_optional_int.some(BITS), "affine", false, str));
                            try mlx.check(mlx.mlx_vector_array_append_value(vec, out5));
                        }
                        try mlx.check(mlx.mlx_eval(vec));
                    }
                    return @as(f64, @floatFromInt(watch.read())) / 1.0e6 / @as(f64, @floatFromInt(iters));
                }
            }.go;

            const ms_impl = try indep(s, &xs, gate, .{ .ctx = null }, inds, ITER, &sw);
            std.debug.print("[moe-gather-ubench]   lhs=implicit inputs=premat {d:6.2} ms  ({d:.1} us/gather)\n", .{ ms_impl, ms_impl * 1000.0 / @as(f64, NG) });
            const ms_expl = try indep(s, &xs, gate, lhs_idx, inds, ITER, &sw);
            std.debug.print("[moe-gather-ubench]   lhs=explicit inputs=premat {d:6.2} ms  ({d:.1} us/gather)\n", .{ ms_expl, ms_expl * 1000.0 / @as(f64, NG) });

            // (7) BANK-SIZE SWEEP. Every cell below reads the SAME 10 experts
            // (identical bytes touched, identical FLOPs) — only the size of the
            // bank they are indexed out of changes. A flat curve means the
            // gather kernel is uniformly slow; a curve that rises with E means
            // the cost is bank-extent addressing (TLB/cache), not the math.
            const E_SWEEP = [_]c_int{ 10, 32, 64, 256 };
            // MLX's default wired limit is 0 (backend/metal/allocator.cpp), so
            // the residency set wires NOTHING and every expert bank lives in
            // `unwired_set_`. Run the sweep at the default and again with the
            // banks wired: if the O(E) slope collapses when wired, the cost is
            // per-dispatch residency/page validation of the bound allocation,
            // not the gather math.
            // Python's sweep runs from a clean allocator (active=0/cache=0) and
            // is FLAT in E; ours runs after ~11 GB of buffer cache has piled up
            // from building the banks. Sweep both ways: `clear` drops MLX's
            // cached MTLBuffers first.
            for ([_]bool{ false, true }) |clear_first| {
                if (clear_first) try mlx.check(mlx.mlx_clear_cache());
                var prev: usize = 0;
                const wired: usize = 0;
                try mlx.check(mlx.mlx_set_wired_limit(&prev, wired));
                var a0: usize = 0;
                var c0: usize = 0;
                try mlx.check(mlx.mlx_get_active_memory(&a0));
                try mlx.check(mlx.mlx_get_cache_memory(&c0));
                std.debug.print("[moe-gather-ubench]   -- clear_cache_first={} (active={d} MB cache={d} MB) --\n", .{ clear_first, a0 >> 20, c0 >> 20 });
                for (E_SWEEP) |e| {
                    var sweep_prng = std.Random.DefaultPrng.init(0xC0FFEE);
                    const bank = try makeBank(allocator, e, N, K, s, &sweep_prng);
                    defer for (bank) |a| {
                        _ = mlx.mlx_array_free(a);
                    };
                    // Re-apply the limit so banks allocated just now get pulled
                    // out of unwired_set_ by ResidencySet::resize.
                    var p2: usize = 0;
                    try mlx.check(mlx.mlx_set_wired_limit(&p2, 0));
                    try mlx.check(mlx.mlx_set_wired_limit(&p2, wired));
                    // indices must stay in range for the small banks; keep the same
                    // COUNT (10) so the work per gather is identical across cells.
                    var iv: [10]i32 = undefined;
                    for (&iv, 0..) |*slot, j| slot.* = @intCast(@mod(@as(c_int, @intCast(j)), e));
                    const isz = [_]c_int{ 1, 1, TOPK };
                    const ind_e = mlx.mlx_array_new_data(&iv, &isz, 3, .int32);
                    defer _ = mlx.mlx_array_free(ind_e);
                    const ms_e = try indep(s, &xs, bank, lhs_idx, ind_e, ITER, &sw);
                    const mb = @as(f64, @floatFromInt(@as(i64, e) * N * K)) * 0.25 / 1.0e6;
                    std.debug.print("[moe-gather-ubench]   E={d:<4} bank={d:5.0} MB   {d:6.2} ms  ({d:.1} us/gather)\n", .{ e, mb, ms_e, ms_e * 1000.0 / @as(f64, NG) });
                }
            }
            var restore: usize = 0;
            try mlx.check(mlx.mlx_set_wired_limit(&restore, 0));
            var act: usize = 0;
            var cch: usize = 0;
            var pk: usize = 0;
            try mlx.check(mlx.mlx_get_active_memory(&act));
            try mlx.check(mlx.mlx_get_cache_memory(&cch));
            try mlx.check(mlx.mlx_get_peak_memory(&pk));
            std.debug.print("[moe-gather-ubench]   mem: active={d} MB cache={d} MB peak={d} MB\n", .{ act >> 20, cch >> 20, pk >> 20 });
        }
    }
}
