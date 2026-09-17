const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const DOWN_KERNEL_MIN_ROWS: usize = 2;

/// Measured crossover: at one row the kernel's in-dispatch k-reduction tail loses to
/// `gather_mm`; from two rows the expert reads hide it.
pub fn downKernelPreferred(rows: usize) bool {
    return rows >= DOWN_KERNEL_MIN_ROWS;
}

pub const MAX_ROWS: c_int = 16;
pub const MAX_TOPK: c_int = 32;

pub var gateup_lpr: c_int = 8;
pub var down_lpr: c_int = 8;
pub var gateup_rows_per_lane: c_int = 0;
pub var down_rows_per_lane: c_int = 0;

const GATEUP_PREFERRED_NPT: c_int = 2;
const DOWN_PREFERRED_NPT: c_int = 4;
const DOWN_STAGE_BUDGET_BYTES: c_int = 8192;
pub var down_slot_groups: c_int = 0;
pub var down_stage_activation: bool = true;

const GATEUP_SOURCE: [:0]const u8 =
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint tile = threadgroup_position_in_grid.x;
    \\uint pair = threadgroup_position_in_grid.z;
    \\
    \\constexpr uint ROWS = 32u / uint(LPR);
    \\constexpr uint KV = uint(KDIM) / 8u;
    \\constexpr int IT = int((KV + uint(LPR) - 1u) / uint(LPR));
    \\
    \\uint sub = lane % uint(LPR);
    \\uint rl = lane / uint(LPR);
    \\uint n0 = ((tile * uint(SGS) + sg) * ROWS + rl) * uint(NPT);
    \\uint r = pair / uint(TOPK);
    \\
    \\uint eid = uint(slots[pair]);
    \\size_t gbase = (size_t)eid * (size_t)(2 * NDIM) * (size_t)KDIM + (size_t)n0 * (size_t)KDIM;
    \\const device uint4* gp = (const device uint4*)(slab + gbase);
    \\const device uint4* pp = (const device uint4*)(slab + gbase + (size_t)NDIM * (size_t)KDIM);
    \\const device uint4* xp = (const device uint4*)(x + (size_t)r * (size_t)KDIM);
    \\constexpr uint RSTRIDE = uint(KDIM) / 8u;
    \\
    \\float4 gl[NPT], gh[NPT], ul[NPT], uh[NPT];
    \\#pragma clang loop unroll(full)
    \\for (int t = 0; t < NPT; ++t) { gl[t] = float4(0.0f); gh[t] = float4(0.0f); ul[t] = float4(0.0f); uh[t] = float4(0.0f); }
    \\for (int i = 0; i < IT; ++i) {
    \\  uint vi = sub + uint(LPR) * uint(i);
    \\  bool ok = vi < KV;
    \\  uint vs = ok ? vi : 0u;
    \\  uint4 xw = ok ? xp[vs] : uint4(0u);
    \\  float4 xlo = as_type<float4>(xw << 16);
    \\  float4 xhi = as_type<float4>(xw & 0xffff0000u);
    \\  #pragma clang loop unroll(full)
    \\  for (int t = 0; t < NPT; ++t) {
    \\    uint4 gw = ok ? gp[uint(t) * RSTRIDE + vs] : uint4(0u);
    \\    uint4 uw = ok ? pp[uint(t) * RSTRIDE + vs] : uint4(0u);
    \\    gl[t] += xlo * as_type<float4>(gw << 16);
    \\    gh[t] += xhi * as_type<float4>(gw & 0xffff0000u);
    \\    ul[t] += xlo * as_type<float4>(uw << 16);
    \\    uh[t] += xhi * as_type<float4>(uw & 0xffff0000u);
    \\  }
    \\}
    \\#pragma clang loop unroll(full)
    \\for (int t = 0; t < NPT; ++t) {
    \\  float4 gt = gl[t] + gh[t];
    \\  float4 ut = ul[t] + uh[t];
    \\  float acc_g = (gt.x + gt.y) + (gt.z + gt.w);
    \\  float acc_u = (ut.x + ut.y) + (ut.z + ut.w);
    \\  for (uint d = uint(LPR) / 2u; d >= 1u; d >>= 1u) {
    \\    acc_g += simd_shuffle_xor(acc_g, ushort(d));
    \\    acc_u += simd_shuffle_xor(acc_u, ushort(d));
    \\  }
    \\  if (sub == 0u) {
    \\    float act = acc_g / (1.0f + metal::exp(-acc_g));
    \\    h[(size_t)pair * (size_t)NDIM + (size_t)(n0 + uint(t))] = T(act * acc_u);
    \\  }
    \\}
;

const DOWNRED_SOURCE: [:0]const u8 =
    \\threadgroup float partial[TOPK * (32 / LPR) * NPT];
    \\threadgroup uint4 hstage[STAGE ? SSG * (KDIM / 8) : 1];
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint tile = threadgroup_position_in_grid.x;
    \\uint r = threadgroup_position_in_grid.z;
    \\
    \\constexpr uint ROWS = 32u / uint(LPR);
    \\constexpr uint BLK = ROWS * uint(NPT);
    \\constexpr uint KV = uint(KDIM) / 8u;
    \\constexpr int IT = int((KV + uint(LPR) - 1u) / uint(LPR));
    \\constexpr uint SPS = uint(TOPK) / uint(SSG);
    \\constexpr uint RSTRIDE = uint(KDIM) / 8u;
    \\
    \\uint sub = lane % uint(LPR);
    \\uint rl = lane / uint(LPR);
    \\uint n0 = tile * BLK + rl * uint(NPT);
    \\const device uint4* hbase = (const device uint4*)(h + (size_t)r * (size_t)TOPK * (size_t)KDIM);
    \\threadgroup uint4* hs = hstage + (STAGE ? sg * KV : 0u);
    \\
    \\for (uint c = 0; c < SPS; ++c) {
    \\  uint slot = sg * SPS + c;
    \\  uint eid = uint(slots[r * uint(TOPK) + slot]);
    \\  const device uint4* dp = (const device uint4*)(down + (size_t)eid * (size_t)NDIM * (size_t)KDIM + (size_t)n0 * (size_t)KDIM);
    \\  const device uint4* hp = hbase + (size_t)slot * (size_t)RSTRIDE;
    \\  if (STAGE) {
    \\    for (uint v = lane; v < KV; v += 32u) hs[v] = hp[v];
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\  }
    \\  float4 al[NPT], ah[NPT];
    \\  #pragma clang loop unroll(full)
    \\  for (int t = 0; t < NPT; ++t) { al[t] = float4(0.0f); ah[t] = float4(0.0f); }
    \\  for (int i = 0; i < IT; ++i) {
    \\    uint vi = sub + uint(LPR) * uint(i);
    \\    bool ok = vi < KV;
    \\    uint vs = ok ? vi : 0u;
    \\    uint4 hw = ok ? (STAGE ? hs[vs] : hp[vs]) : uint4(0u);
    \\    float4 hlo = as_type<float4>(hw << 16);
    \\    float4 hhi = as_type<float4>(hw & 0xffff0000u);
    \\    #pragma clang loop unroll(full)
    \\    for (int t = 0; t < NPT; ++t) {
    \\      uint4 dw = ok ? dp[uint(t) * RSTRIDE + vs] : uint4(0u);
    \\      al[t] += hlo * as_type<float4>(dw << 16);
    \\      ah[t] += hhi * as_type<float4>(dw & 0xffff0000u);
    \\    }
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int t = 0; t < NPT; ++t) {
    \\    float4 at = al[t] + ah[t];
    \\    float acc = (at.x + at.y) + (at.z + at.w);
    \\    for (uint d = uint(LPR) / 2u; d >= 1u; d >>= 1u) acc += simd_shuffle_xor(acc, ushort(d));
    \\    if (sub == 0u) partial[slot * BLK + rl * uint(NPT) + uint(t)] = acc;
    \\  }
    \\  if (STAGE && SPS > 1) simdgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0u && lane < BLK) {
    \\  float total = 0.0f;
    \\  for (uint k = 0; k < uint(TOPK); ++k) {
    \\    total += weights[r * uint(TOPK) + k] * partial[k * BLK + lane];
    \\  }
    \\  y[(size_t)r * (size_t)NDIM + (size_t)(tile * BLK + lane)] = T(total);
    \\}
;

var gateup_kernel: ?mlx.mlx_fast_metal_kernel = null;
var downred_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn CfgCache(comptime Key: type, comptime CAP: usize) type {
    return struct {
        const Self = @This();
        keys: [CAP]Key = @splat(std.mem.zeroes(Key)),
        cfgs: [CAP]?mlx.mlx_fast_metal_kernel_config = @splat(null),
        used: [CAP]u64 = @splat(0),
        tick: u64 = 0,

        fn get(self: *Self, key: Key) ?mlx.mlx_fast_metal_kernel_config {
            for (self.cfgs, 0..) |c, i| {
                if (c != null and std.meta.eql(self.keys[i], key)) {
                    self.tick += 1;
                    self.used[i] = self.tick;
                    return c.?;
                }
            }
            return null;
        }

        fn put(self: *Self, key: Key, cfg: mlx.mlx_fast_metal_kernel_config) void {
            var victim: usize = 0;
            var oldest: u64 = std.math.maxInt(u64);
            for (self.cfgs, 0..) |c, i| {
                if (c == null) {
                    victim = i;
                    break;
                }
                if (self.used[i] < oldest) {
                    oldest = self.used[i];
                    victim = i;
                }
            }
            if (self.cfgs[victim]) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
            self.cfgs[victim] = cfg;
            self.keys[victim] = key;
            self.tick += 1;
            self.used[victim] = self.tick;
        }
    };
}

const GateUpKey = struct { r: c_int, topk: c_int, n: c_int, k: c_int, lpr: c_int, sgs: c_int, npt: c_int };
var gateup_cfgs: CfgCache(GateUpKey, 8) = .{};
var gateup_engaged: bool = false;

const DownRedKey = struct { r: c_int, topk: c_int, n: c_int, k: c_int, lpr: c_int, npt: c_int, ssg: c_int, stage: bool };
var downred_cfgs: CfgCache(DownRedKey, 8) = .{};
var downred_engaged: bool = false;

fn getGateUpKernel() !mlx.mlx_fast_metal_kernel {
    if (gateup_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "slab", "slots" };
    const output_names = [_][*:0]const u8{"h"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_expert_bf16_gateup_swiglu",
        in_vec,
        out_vec,
        GATEUP_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gateup_kernel = kernel;
    return kernel;
}

fn getDownReduceKernel() !mlx.mlx_fast_metal_kernel {
    if (downred_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "h", "down", "slots", "weights" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "mlxserve_expert_bf16_down_reduce",
        in_vec,
        out_vec,
        DOWNRED_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    downred_kernel = kernel;
    return kernel;
}

fn autoSlotGroups(topk: c_int, k: c_int) c_int {
    var d = topk;
    while (d > 1) : (d -= 1) {
        if (@rem(topk, d) == 0 and d * k * 2 <= DOWN_STAGE_BUDGET_BYTES) return d;
    }
    return 1;
}

fn autoRowsPerLane(n: c_int, rows: c_int, preferred: c_int, max_block: c_int) c_int {
    var npt = preferred;
    while (npt > 1) : (npt -= 1) {
        if (rows * npt <= max_block and @rem(n, rows * npt) == 0) return npt;
    }
    return 1;
}

fn simdgroupsPerGroup(tiles: c_int) c_int {
    var sgs: c_int = 8;
    while (sgs > 1) : (sgs -= 1) {
        if (@rem(tiles, sgs) == 0) return sgs;
    }
    return 1;
}

fn checkSlots(s: mlx.mlx_stream, slots: mlx.mlx_array, rows: c_int, topk: c_int, slab_slots: c_int) !void {
    if (mlx.mlx_array_dtype(slots) != .int32) return error.SlotsNotInt32;
    const sh = mlx.getShape(slots);
    if (sh.len != 2 or sh[0] != rows or sh[1] != topk) return error.SlotsShapeMismatch;
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, slots, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const ptr = mlx.mlx_array_data_int32(contig) orelse return error.SlotsUnreadable;
    const count: usize = @intCast(rows * topk);
    for (ptr[0..count]) |id| {
        if (id < 0 or id >= slab_slots) return error.SlotOutOfRange;
    }
}

pub fn gateUpSwiglu(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    slab: mlx.mlx_array,
    slots: mlx.mlx_array,
) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return error.MetalKernelNeedsGpuStream;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(slab) != .bfloat16) return error.ExpectedBf16;
    const xsh = mlx.getShape(x);
    if (xsh.len != 2) return error.BadActivationShape;
    const wsh = mlx.getShape(slab);
    if (wsh.len != 3) return error.BadSlabShape;
    const U = wsh[0];
    if (U <= 0) return error.EmptySlab;
    if (@rem(wsh[1], 2) != 0) return error.BadSlabShape;
    const N = @divExact(wsh[1], 2);
    const K = wsh[2];
    if (K != xsh[1]) return error.HiddenWidthMismatch;
    if (@rem(K, 8) != 0) return error.UnsupportedHiddenWidth;
    const R = xsh[0];
    if (R < 1 or R > MAX_ROWS) return error.RowsOutOfRange;
    const ssh = mlx.getShape(slots);
    if (ssh.len != 2) return error.SlotsShapeMismatch;
    const topk = ssh[1];
    if (topk < 1 or topk > MAX_TOPK) return error.TopKOutOfRange;
    const lpr = gateup_lpr;
    if (lpr != 1 and lpr != 2 and lpr != 4 and lpr != 8 and lpr != 16 and lpr != 32) return error.UnsupportedLaneSplit;
    const lane_rows = @divExact(@as(c_int, 32), lpr);
    const npt = if (gateup_rows_per_lane > 0) gateup_rows_per_lane else autoRowsPerLane(N, lane_rows, GATEUP_PREFERRED_NPT, 1024);
    if (npt > 8) return error.UnsupportedRowBlock;
    const rows_per_sg = lane_rows * npt;
    if (@rem(N, rows_per_sg) != 0) return error.UnsupportedIntermediateWidth;
    const tiles = @divExact(N, rows_per_sg);
    const sgs = simdgroupsPerGroup(tiles);
    try checkSlots(s, slots, R, topk, U);

    const key = GateUpKey{ .r = R, .topk = topk, .n = N, .k = K, .lpr = lpr, .sgs = sgs, .npt = npt };
    const cached = gateup_cfgs.get(key) orelse blk: {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const out_shape = [_]c_int{ R, topk, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * @divExact(tiles, sgs), sgs, R * topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, sgs, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NDIM", N));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KDIM", K));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LPR", lpr));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SGS", sgs));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NPT", npt));
        gateup_cfgs.put(key, cfg);
        break :blk cfg;
    };

    const inputs_arr = [_]mlx.mlx_array{ x, slab, slots };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getGateUpKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cached, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!gateup_engaged) {
        gateup_engaged = true;
        log.info("[expert-bf16] gate+up+SwiGLU slab kernel engaged: rows={d} topk={d} inter={d} hidden={d} lanes={d} rpl={d}\n", .{ R, topk, N, K, lpr, npt });
    }
    return out;
}

pub fn downReduce(
    s: mlx.mlx_stream,
    h: mlx.mlx_array,
    down: mlx.mlx_array,
    slots: mlx.mlx_array,
    weights: mlx.mlx_array,
) !mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return error.MetalKernelNeedsGpuStream;
    if (mlx.mlx_array_dtype(h) != .bfloat16 or mlx.mlx_array_dtype(down) != .bfloat16) return error.ExpectedBf16;
    if (mlx.mlx_array_dtype(weights) != .float32) return error.ExpectedF32Weights;
    const hsh = mlx.getShape(h);
    if (hsh.len != 3) return error.BadActivationShape;
    const dsh = mlx.getShape(down);
    if (dsh.len != 3) return error.BadSlabShape;
    const U = dsh[0];
    if (U <= 0) return error.EmptySlab;
    const N = dsh[1];
    const K = dsh[2];
    if (K != hsh[2]) return error.IntermediateWidthMismatch;
    if (@rem(K, 8) != 0) return error.UnsupportedIntermediateWidth;
    const R = hsh[0];
    if (R < 1 or R > MAX_ROWS) return error.RowsOutOfRange;
    const topk = hsh[1];
    if (topk < 1 or topk > MAX_TOPK) return error.TopKOutOfRange;
    const wsh = mlx.getShape(weights);
    if (wsh.len != 2 or wsh[0] != R or wsh[1] != topk) return error.WeightsShapeMismatch;
    const lpr = down_lpr;
    if (lpr != 1 and lpr != 2 and lpr != 4 and lpr != 8 and lpr != 16 and lpr != 32) return error.UnsupportedLaneSplit;
    const lane_rows = @divExact(@as(c_int, 32), lpr);
    const stage = down_stage_activation;
    const npt = if (down_rows_per_lane > 0) down_rows_per_lane else autoRowsPerLane(N, lane_rows, DOWN_PREFERRED_NPT, 32);
    if (npt > 8) return error.UnsupportedRowBlock;
    const rows_per_sg = lane_rows * npt;
    if (rows_per_sg > 32) return error.UnsupportedRowBlock;
    if (@rem(N, rows_per_sg) != 0) return error.UnsupportedHiddenWidth;
    try checkSlots(s, slots, R, topk, U);

    const ssg = if (down_slot_groups > 0) down_slot_groups else if (stage) autoSlotGroups(topk, K) else topk;
    if (ssg > topk or @rem(topk, ssg) != 0) return error.UnsupportedSlotGrouping;
    const key = DownRedKey{ .r = R, .topk = topk, .n = N, .k = K, .lpr = lpr, .npt = npt, .ssg = ssg, .stage = stage };
    const cached = downred_cfgs.get(key) orelse blk: {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        const out_shape = [_]c_int{ R, N };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 2, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * @divExact(N, rows_per_sg), ssg, R));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, ssg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", topk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NDIM", N));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KDIM", K));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LPR", lpr));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NPT", npt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SSG", ssg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "STAGE", @intFromBool(stage)));
        downred_cfgs.put(key, cfg);
        break :blk cfg;
    };

    const inputs_arr = [_]mlx.mlx_array{ h, down, slots, weights };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getDownReduceKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cached, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!downred_engaged) {
        downred_engaged = true;
        log.info("[expert-bf16] down+reduce slab kernel engaged: rows={d} topk={d} hidden={d} inter={d} lanes={d} rpl={d} slot_groups={d} staged={}\n", .{ R, topk, N, K, lpr, npt, ssg, stage });
    }
    return out;
}

const testing = std.testing;

fn bfToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

const Bf16Array = struct {
    arr: mlx.mlx_array,
    bits: []u16,

    fn deinit(self: *Bf16Array, alloc: std.mem.Allocator) void {
        _ = mlx.mlx_array_free(self.arr);
        alloc.free(self.bits);
    }
};

fn randBf16(alloc: std.mem.Allocator, s: mlx.mlx_stream, rnd: std.Random, shape: []const c_int) !Bf16Array {
    var count: usize = 1;
    for (shape) |d| count *= @intCast(d);
    const vals = try alloc.alloc(f32, count);
    defer alloc.free(vals);
    for (vals) |*v| v.* = rnd.float(f32) - 0.5;
    return fromF32(alloc, s, vals, shape);
}

fn fromF32(alloc: std.mem.Allocator, s: mlx.mlx_stream, vals: []const f32, shape: []const c_int) !Bf16Array {
    const a32 = mlx.mlx_array_new_data(vals.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(a32);
    var ab = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_astype(&ab, a32, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(ab));
    const src = mlx.mlx_array_data_bfloat16(ab) orelse return error.Bf16Unreadable;
    const bits = try alloc.alloc(u16, vals.len);
    @memcpy(bits, src[0..vals.len]);
    return .{ .arr = ab, .bits = bits };
}

fn readBf16(alloc: std.mem.Allocator, arr: mlx.mlx_array) ![]f32 {
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, arr, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(contig));
    const n = mlx.mlx_array_size(contig);
    const src = mlx.mlx_array_data_bfloat16(contig) orelse return error.Bf16Unreadable;
    const out = try alloc.alloc(f32, n);
    for (out, 0..) |*v, i| v.* = bfToF32(src[i]);
    return out;
}

const ErrStats = struct {
    max: f32 = 0,
    rms: f32 = 0,
    finite: bool = true,
};

fn errStats(got: []const f32, truth: []const f32) ErrStats {
    var st = ErrStats{};
    var acc: f64 = 0;
    for (got, truth) |g, t| {
        if (!std.math.isFinite(g)) st.finite = false;
        const d = @abs(g - t);
        if (d > st.max) st.max = d;
        acc += @as(f64, d) * @as(f64, d);
    }
    st.rms = @floatCast(@sqrt(acc / @as(f64, @floatFromInt(got.len))));
    return st;
}

fn siluF32(v: f32) f32 {
    return v / (1.0 + @exp(-v));
}

fn gateUpTruth(
    alloc: std.mem.Allocator,
    xb: []const u16,
    slab: []const u16,
    slots: []const i32,
    R: usize,
    topk: usize,
    N: usize,
    K: usize,
) ![]f32 {
    const out = try alloc.alloc(f32, R * topk * N);
    for (0..R) |r| {
        for (0..topk) |k| {
            const slot: usize = @intCast(slots[r * topk + k]);
            for (0..N) |n| {
                const gbase = (slot * 2 * N + n) * K;
                const ubase = (slot * 2 * N + N + n) * K;
                var ga: f32 = 0;
                var ua: f32 = 0;
                for (0..K) |j| {
                    const xv = bfToF32(xb[r * K + j]);
                    ga += xv * bfToF32(slab[gbase + j]);
                    ua += xv * bfToF32(slab[ubase + j]);
                }
                out[(r * topk + k) * N + n] = siluF32(ga) * ua;
            }
        }
    }
    return out;
}

fn downTruth(
    alloc: std.mem.Allocator,
    hb: []const u16,
    slab: []const u16,
    slots: []const i32,
    w: []const f32,
    R: usize,
    topk: usize,
    N: usize,
    K: usize,
) ![]f32 {
    const out = try alloc.alloc(f32, R * N);
    for (0..R) |r| {
        for (0..N) |n| {
            var total: f32 = 0;
            for (0..topk) |k| {
                const slot: usize = @intCast(slots[r * topk + k]);
                const dbase = (slot * N + n) * K;
                var acc: f32 = 0;
                for (0..K) |j| {
                    acc += bfToF32(hb[(r * topk + k) * K + j]) * bfToF32(slab[dbase + j]);
                }
                total += w[r * topk + k] * acc;
            }
            out[r * N + n] = total;
        }
    }
    return out;
}

fn sliceTranspose(s: mlx.mlx_stream, slab: mlx.mlx_array, row0: c_int, row1: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(slab);
    const strides = [_]c_int{ 1, 1, 1 };
    var raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(raw);
    try mlx.check(mlx.mlx_slice(&raw, slab, &[_]c_int{ 0, row0, 0 }, 3, &[_]c_int{ sh[0], row1, sh[2] }, 3, &strides, 3, s));
    var t = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, raw, &[_]c_int{ 0, 2, 1 }, 3, s));
    return t;
}

fn gatherMmGateUp(s: mlx.mlx_stream, x: mlx.mlx_array, slab: mlx.mlx_array, slots: mlx.mlx_array, R: c_int, K: c_int, N: c_int) !mlx.mlx_array {
    const no_idx = mlx.mlx_array{ .ctx = null };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &[_]c_int{ R, 1, 1, K }, 4, s));
    const gv = try sliceTranspose(s, slab, 0, N);
    defer _ = mlx.mlx_array_free(gv);
    const uv = try sliceTranspose(s, slab, N, 2 * N);
    defer _ = mlx.mlx_array_free(uv);
    var g4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g4);
    try mlx.check(mlx.mlx_gather_mm(&g4, x4, gv, no_idx, slots, false, s));
    var uu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(uu);
    try mlx.check(mlx.mlx_gather_mm(&uu, x4, uv, no_idx, slots, false, s));
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_squeeze(&g, g4, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_squeeze(&u, uu, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var act = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act);
    try mlx.check(mlx.mlx_multiply(&act, g, sig, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_multiply(&out, act, u, s));
    return out;
}

fn gatherMmDownReduce(s: mlx.mlx_stream, h: mlx.mlx_array, down: mlx.mlx_array, slots: mlx.mlx_array, w: mlx.mlx_array, R: c_int, topk: c_int, K: c_int) !mlx.mlx_array {
    const no_idx = mlx.mlx_array{ .ctx = null };
    var h4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h4);
    try mlx.check(mlx.mlx_reshape(&h4, h, &[_]c_int{ R, topk, 1, K }, 4, s));
    var dv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dv);
    try mlx.check(mlx.mlx_transpose_axes(&dv, down, &[_]c_int{ 0, 2, 1 }, 3, s));
    var d4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d4);
    try mlx.check(mlx.mlx_gather_mm(&d4, h4, dv, no_idx, slots, false, s));
    var d3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d3);
    try mlx.check(mlx.mlx_squeeze(&d3, d4, s));
    var wb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wb);
    try mlx.check(mlx.mlx_astype(&wb, w, .bfloat16, s));
    var we = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(we);
    try mlx.check(mlx.mlx_expand_dims(&we, wb, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d3, we, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, -2, false, s));
    return out;
}

fn randomSlots(alloc: std.mem.Allocator, rnd: std.Random, R: usize, topk: usize, U: usize) ![]i32 {
    const slots = try alloc.alloc(i32, R * topk);
    for (slots) |*v| v.* = @intCast(rnd.uintLessThan(usize, U));
    for (0..R) |r| {
        slots[r * topk + 1] = slots[r * topk];
        if (topk > 4) slots[r * topk + 4] = slots[r * topk + 3];
    }
    if (R > 1) {
        for (0..topk) |k| slots[topk + k] = slots[k];
    }
    return slots;
}

fn runGateUpShape(alloc: std.mem.Allocator, s: mlx.mlx_stream, rnd: std.Random, U: c_int, R: c_int, topk: c_int, N: c_int, K: c_int, verbose: bool) !void {
    var x = try randBf16(alloc, s, rnd, &[_]c_int{ R, K });
    defer x.deinit(alloc);
    var slab = try randBf16(alloc, s, rnd, &[_]c_int{ U, 2 * N, K });
    defer slab.deinit(alloc);
    const slots_host = try randomSlots(alloc, rnd, @intCast(R), @intCast(topk), @intCast(U));
    defer alloc.free(slots_host);
    const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ R, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(slots);

    const truth = try gateUpTruth(alloc, x.bits, slab.bits, slots_host, @intCast(R), @intCast(topk), @intCast(N), @intCast(K));
    defer alloc.free(truth);

    const kout = try gateUpSwiglu(s, x.arr, slab.arr, slots);
    defer _ = mlx.mlx_array_free(kout);
    const kvals = try readBf16(alloc, kout);
    defer alloc.free(kvals);

    const gout = try gatherMmGateUp(s, x.arr, slab.arr, slots, R, K, N);
    defer _ = mlx.mlx_array_free(gout);
    const gvals = try readBf16(alloc, gout);
    defer alloc.free(gvals);

    const ks = errStats(kvals, truth);
    const gs = errStats(gvals, truth);
    if (verbose) std.debug.print("gate_up U={d} R={d} k={d} N={d} K={d}: kernel max={e:.3} rms={e:.3} | gather_mm max={e:.3} rms={e:.3}\n", .{ U, R, topk, N, K, ks.max, ks.rms, gs.max, gs.rms });
    try testing.expect(ks.finite);
    try testing.expect(gs.finite);
    try testing.expect(ks.max <= gs.max);
    try testing.expect(ks.rms <= gs.rms);
}

fn runDownShape(alloc: std.mem.Allocator, s: mlx.mlx_stream, rnd: std.Random, U: c_int, R: c_int, topk: c_int, N: c_int, K: c_int, verbose: bool) !void {
    var h = try randBf16(alloc, s, rnd, &[_]c_int{ R, topk, K });
    defer h.deinit(alloc);
    var slab = try randBf16(alloc, s, rnd, &[_]c_int{ U, N, K });
    defer slab.deinit(alloc);
    const slots_host = try randomSlots(alloc, rnd, @intCast(R), @intCast(topk), @intCast(U));
    defer alloc.free(slots_host);
    const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ R, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(slots);
    const w_host = try alloc.alloc(f32, @intCast(R * topk));
    defer alloc.free(w_host);
    for (w_host) |*v| v.* = rnd.float(f32);
    const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ R, topk }, 2, .float32);
    defer _ = mlx.mlx_array_free(w);

    const truth = try downTruth(alloc, h.bits, slab.bits, slots_host, w_host, @intCast(R), @intCast(topk), @intCast(N), @intCast(K));
    defer alloc.free(truth);

    const kout = try downReduce(s, h.arr, slab.arr, slots, w);
    defer _ = mlx.mlx_array_free(kout);
    const kvals = try readBf16(alloc, kout);
    defer alloc.free(kvals);

    const gout = try gatherMmDownReduce(s, h.arr, slab.arr, slots, w, R, topk, K);
    defer _ = mlx.mlx_array_free(gout);
    const gvals = try readBf16(alloc, gout);
    defer alloc.free(gvals);

    const ks = errStats(kvals, truth);
    const gs = errStats(gvals, truth);
    if (verbose) std.debug.print("down U={d} R={d} k={d} N={d} K={d}: kernel max={e:.3} rms={e:.3} | gather_mm max={e:.3} rms={e:.3}\n", .{ U, R, topk, N, K, ks.max, ks.rms, gs.max, gs.rms });
    try testing.expect(ks.finite);
    try testing.expect(gs.finite);
    try testing.expect(ks.max <= gs.max);
    try testing.expect(ks.rms <= gs.rms);
}

test "bf16 slab gate+up+SwiGLU is no worse than the gather_mm reference against fp32 truth" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xB16C0DE1);
    const rnd = prng.random();
    for ([_]c_int{ 16, 64, 512 }) |U| {
        for ([_]c_int{ 1, 2, 4, 8, 16 }) |R| {
            try runGateUpShape(alloc, s, rnd, U, R, 10, 64, 128, benchEnabled());
        }
    }
}

test "bf16 slab down+reduce is no worse than the gather_mm reference against fp32 truth" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xD0117EED);
    const rnd = prng.random();
    for ([_]c_int{ 16, 64, 512 }) |U| {
        for ([_]c_int{ 1, 2, 4, 8, 16 }) |R| {
            try runDownShape(alloc, s, rnd, U, R, 10, 128, 64, benchEnabled());
        }
    }
}

test "bf16 slab kernels hold parity at the Qwen3.8-Flash-Next expert geometry" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9E77E57);
    const rnd = prng.random();
    try runGateUpShape(alloc, s, rnd, 8, 1, 10, 640, 2560, benchEnabled());
    try runDownShape(alloc, s, rnd, 8, 1, 10, 2560, 640, benchEnabled());
}

const PARITY_VARIANTS = [_]struct { lanes: c_int, rpl: c_int, groups: c_int, stage: bool }{
    .{ .lanes = 8, .rpl = 0, .groups = 0, .stage = true },
    .{ .lanes = 8, .rpl = 0, .groups = 0, .stage = false },
    .{ .lanes = 1, .rpl = 1, .groups = 10, .stage = true },
    .{ .lanes = 2, .rpl = 2, .groups = 5, .stage = true },
    .{ .lanes = 8, .rpl = 0, .groups = 0, .stage = true },
    .{ .lanes = 8, .rpl = 4, .groups = 10, .stage = false },
    .{ .lanes = 8, .rpl = 4, .groups = 10, .stage = true },
    .{ .lanes = 8, .rpl = 8, .groups = 10, .stage = true },
    .{ .lanes = 8, .rpl = 4, .groups = 2, .stage = true },
    .{ .lanes = 8, .rpl = 4, .groups = 1, .stage = true },
    .{ .lanes = 16, .rpl = 2, .groups = 2, .stage = true },
    .{ .lanes = 32, .rpl = 4, .groups = 1, .stage = true },
};

test "bf16 slab kernels hold parity at every lane split and unroll width" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1A4E5);
    const rnd = prng.random();
    const saved_gl = gateup_lpr;
    const saved_dl = down_lpr;
    const saved_gu = gateup_rows_per_lane;
    const saved_du = down_rows_per_lane;
    const saved_gr = down_slot_groups;
    const saved_st = down_stage_activation;
    defer {
        gateup_lpr = saved_gl;
        down_lpr = saved_dl;
        gateup_rows_per_lane = saved_gu;
        down_rows_per_lane = saved_du;
        down_slot_groups = saved_gr;
        down_stage_activation = saved_st;
    }
    for (PARITY_VARIANTS) |v| {
        gateup_lpr = v.lanes;
        down_lpr = v.lanes;
        gateup_rows_per_lane = v.rpl;
        down_rows_per_lane = v.rpl;
        down_slot_groups = v.groups;
        down_stage_activation = v.stage;
        try runGateUpShape(alloc, s, rnd, 8, 2, 10, 32, 128, false);
        try runDownShape(alloc, s, rnd, 8, 2, 10, 64, 56, false);
    }
}

test "bf16 slab down+reduce sums the top-k partials in ascending k order" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    const U: c_int = 4;
    const R: c_int = 1;
    const topk: c_int = 10;
    const N: c_int = 8;
    const K: c_int = 8;

    const hvals = try alloc.alloc(f32, @intCast(R * topk * K));
    defer alloc.free(hvals);
    @memset(hvals, 0);
    for (0..@intCast(topk)) |k| hvals[k * @as(usize, @intCast(K))] = 1.0;
    var h = try fromF32(alloc, s, hvals, &[_]c_int{ R, topk, K });
    defer h.deinit(alloc);

    const dvals = try alloc.alloc(f32, @intCast(U * N * K));
    defer alloc.free(dvals);
    @memset(dvals, 0);
    for (0..@intCast(U)) |u| {
        for (0..@intCast(N)) |n| dvals[(u * @as(usize, @intCast(N)) + n) * @as(usize, @intCast(K))] = 1.0;
    }
    var slab = try fromF32(alloc, s, dvals, &[_]c_int{ U, N, K });
    defer slab.deinit(alloc);

    const slots_host = try alloc.alloc(i32, @intCast(R * topk));
    defer alloc.free(slots_host);
    @memset(slots_host, 0);
    const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ R, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(slots);

    const w_host = try alloc.alloc(f32, @intCast(R * topk));
    defer alloc.free(w_host);
    @memset(w_host, 0);
    w_host[0] = 1.0;
    w_host[1] = 1.0e20;
    w_host[2] = -1.0e20;
    const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ R, topk }, 2, .float32);
    defer _ = mlx.mlx_array_free(w);

    const kout = try downReduce(s, h.arr, slab.arr, slots, w);
    defer _ = mlx.mlx_array_free(kout);
    const kvals = try readBf16(alloc, kout);
    defer alloc.free(kvals);
    for (kvals) |v| try testing.expectEqual(@as(f32, 0.0), v);
}

test "bf16 slab down kernel is preferred only at or past the measured crossover" {
    try testing.expect(!downKernelPreferred(0));
    try testing.expect(!downKernelPreferred(1));
    try testing.expect(downKernelPreferred(2));
    try testing.expect(downKernelPreferred(4));
    try testing.expect(downKernelPreferred(8));
    try testing.expect(downKernelPreferred(@intCast(MAX_ROWS)));
    try testing.expectEqual(@as(usize, 2), DOWN_KERNEL_MIN_ROWS);
}

test "bf16 slab kernels decline outside their contract" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xDEC11E);
    const rnd = prng.random();
    const U: c_int = 4;
    const N: c_int = 16;
    const K: c_int = 32;
    const topk: c_int = 4;

    var slab = try randBf16(alloc, s, rnd, &[_]c_int{ U, 2 * N, K });
    defer slab.deinit(alloc);
    var down = try randBf16(alloc, s, rnd, &[_]c_int{ U, N, K });
    defer down.deinit(alloc);
    var x = try randBf16(alloc, s, rnd, &[_]c_int{ 1, K });
    defer x.deinit(alloc);
    var hh = try randBf16(alloc, s, rnd, &[_]c_int{ 1, topk, K });
    defer hh.deinit(alloc);

    const ok_host = try alloc.alloc(i32, @intCast(topk));
    defer alloc.free(ok_host);
    @memset(ok_host, 1);
    const ok_slots = mlx.mlx_array_new_data(ok_host.ptr, &[_]c_int{ 1, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(ok_slots);
    const w_host = try alloc.alloc(f32, @intCast(topk));
    defer alloc.free(w_host);
    @memset(w_host, 0.5);
    const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ 1, topk }, 2, .float32);
    defer _ = mlx.mlx_array_free(w);

    var empty = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(empty);
    const strides = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(&empty, slab.arr, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 0, 2 * N, K }, 3, &strides, 3, s));
    try testing.expectError(error.EmptySlab, gateUpSwiglu(s, x.arr, empty, ok_slots));

    var empty_down = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(empty_down);
    try mlx.check(mlx.mlx_slice(&empty_down, down.arr, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 0, N, K }, 3, &strides, 3, s));
    try testing.expectError(error.EmptySlab, downReduce(s, hh.arr, empty_down, ok_slots, w));

    var wide = try randBf16(alloc, s, rnd, &[_]c_int{ MAX_ROWS + 1, K });
    defer wide.deinit(alloc);
    const wide_host = try alloc.alloc(i32, @intCast((MAX_ROWS + 1) * topk));
    defer alloc.free(wide_host);
    @memset(wide_host, 0);
    const wide_slots = mlx.mlx_array_new_data(wide_host.ptr, &[_]c_int{ MAX_ROWS + 1, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(wide_slots);
    try testing.expectError(error.RowsOutOfRange, gateUpSwiglu(s, wide.arr, slab.arr, wide_slots));
    var wide_h = try randBf16(alloc, s, rnd, &[_]c_int{ MAX_ROWS + 1, topk, K });
    defer wide_h.deinit(alloc);
    const wide_w_host = try alloc.alloc(f32, @intCast((MAX_ROWS + 1) * topk));
    defer alloc.free(wide_w_host);
    @memset(wide_w_host, 0.5);
    const wide_w = mlx.mlx_array_new_data(wide_w_host.ptr, &[_]c_int{ MAX_ROWS + 1, topk }, 2, .float32);
    defer _ = mlx.mlx_array_free(wide_w);
    try testing.expectError(error.RowsOutOfRange, downReduce(s, wide_h.arr, down.arr, wide_slots, wide_w));

    const bad_host = try alloc.alloc(i32, @intCast(topk));
    defer alloc.free(bad_host);
    @memset(bad_host, 0);
    bad_host[topk - 1] = U;
    const bad_slots = mlx.mlx_array_new_data(bad_host.ptr, &[_]c_int{ 1, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(bad_slots);
    try testing.expectError(error.SlotOutOfRange, gateUpSwiglu(s, x.arr, slab.arr, bad_slots));
    try testing.expectError(error.SlotOutOfRange, downReduce(s, hh.arr, down.arr, bad_slots, w));

    const neg_host = try alloc.alloc(i32, @intCast(topk));
    defer alloc.free(neg_host);
    @memset(neg_host, 0);
    neg_host[0] = -1;
    const neg_slots = mlx.mlx_array_new_data(neg_host.ptr, &[_]c_int{ 1, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(neg_slots);
    try testing.expectError(error.SlotOutOfRange, gateUpSwiglu(s, x.arr, slab.arr, neg_slots));
}

const BenchClock = struct {
    io: std.Io,
    start: std.Io.Timestamp,
    mark_ns: u64 = 0,

    fn init() BenchClock {
        const io = std.Io.Threaded.global_single_threaded.io();
        return .{ .io = io, .start = std.Io.Timestamp.now(io, .boot) };
    }

    fn lap(self: *BenchClock) u64 {
        const cum: u64 = @intCast(self.start.untilNow(self.io, .boot).nanoseconds);
        const d = cum - self.mark_ns;
        self.mark_ns = cum;
        return d;
    }
};

fn benchEnabled() bool {
    const raw = std.c.getenv("EXPERT_BF16_BENCH") orelse return false;
    return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
}

fn medianUs(samples: []u64) f64 {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    const mid = samples.len / 2;
    const ns: f64 = if (samples.len % 2 == 1)
        @floatFromInt(samples[mid])
    else
        (@as(f64, @floatFromInt(samples[mid - 1])) + @as(f64, @floatFromInt(samples[mid]))) / 2.0;
    return ns / 1000.0;
}

fn distinctSlots(alloc: std.mem.Allocator, R: c_int, topk: c_int, U: c_int) ![]i32 {
    const n: usize = @intCast(R * topk);
    const slots = try alloc.alloc(i32, n);
    const stride: i32 = @divTrunc(U, @as(i32, @intCast(n)));
    for (slots, 0..) |*v, i| v.* = @as(i32, @intCast(i)) * stride;
    return slots;
}

const Variant = struct { lanes: c_int, rpl: c_int, groups: c_int = 0, stage: bool = false };
const GATEUP_SWEEP = [_]Variant{
    .{ .lanes = 8, .rpl = 0 },
    .{ .lanes = 8, .rpl = 2 },
    .{ .lanes = 16, .rpl = 2 },
    .{ .lanes = 32, .rpl = 4 },
};
const DOWN_SWEEP = [_]Variant{
    .{ .lanes = 8, .rpl = 0, .groups = 0, .stage = true },
    .{ .lanes = 8, .rpl = 4, .groups = 10, .stage = false },
    .{ .lanes = 8, .rpl = 4, .groups = 10, .stage = true },
    .{ .lanes = 8, .rpl = 8, .groups = 10, .stage = true },
    .{ .lanes = 8, .rpl = 4, .groups = 5, .stage = true },
    .{ .lanes = 8, .rpl = 4, .groups = 2, .stage = true },
    .{ .lanes = 8, .rpl = 4, .groups = 1, .stage = true },
    .{ .lanes = 16, .rpl = 2, .groups = 10, .stage = true },
    .{ .lanes = 32, .rpl = 4, .groups = 10, .stage = true },
};
const REPS: usize = 8;

fn benchLayer(alloc: std.mem.Allocator, s: mlx.mlx_stream, rnd: std.Random, U: c_int, R: c_int, roof_gbs: f64) !void {
    const topk: c_int = 10;
    const N: c_int = 640;
    const H: c_int = 2560;

    var x = try randBf16(alloc, s, rnd, &[_]c_int{ R, H });
    defer x.deinit(alloc);
    var gu = try randBf16(alloc, s, rnd, &[_]c_int{ U, 2 * N, H });
    defer gu.deinit(alloc);
    var dn = try randBf16(alloc, s, rnd, &[_]c_int{ U, H, N });
    defer dn.deinit(alloc);
    var hact = try randBf16(alloc, s, rnd, &[_]c_int{ R, topk, N });
    defer hact.deinit(alloc);
    const w_host = try alloc.alloc(f32, @intCast(R * topk));
    defer alloc.free(w_host);
    for (w_host) |*v| v.* = rnd.float(f32);
    const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ R, topk }, 2, .float32);
    defer _ = mlx.mlx_array_free(w);

    var slot_hosts: [REPS][]i32 = undefined;
    var slot_arrs: [REPS]mlx.mlx_array = undefined;
    const per_rep: usize = @intCast(R * topk);
    for (0..REPS) |rep| {
        slot_hosts[rep] = try alloc.alloc(i32, per_rep);
        for (slot_hosts[rep], 0..) |*v, i| v.* = @intCast((rep * per_rep + i) % @as(usize, @intCast(U)));
        slot_arrs[rep] = mlx.mlx_array_new_data(slot_hosts[rep].ptr, &[_]c_int{ R, topk }, 2, .int32);
    }
    defer for (0..REPS) |rep| {
        _ = mlx.mlx_array_free(slot_arrs[rep]);
        alloc.free(slot_hosts[rep]);
    };

    const laps: usize = 50;
    const warm: usize = 10;
    const arms = 2 * (2 + GATEUP_SWEEP.len + DOWN_SWEEP.len);
    const t = try alloc.alloc(u64, arms * laps);
    defer alloc.free(t);

    const saved_gu = gateup_lpr;
    const saved_dn = down_lpr;
    const saved_guu = gateup_rows_per_lane;
    const saved_dnu = down_rows_per_lane;
    const saved_grp = down_slot_groups;
    const saved_stg = down_stage_activation;
    defer {
        down_stage_activation = saved_stg;
        gateup_lpr = saved_gu;
        down_lpr = saved_dn;
        gateup_rows_per_lane = saved_guu;
        down_rows_per_lane = saved_dnu;
        down_slot_groups = saved_grp;
    }

    var timer = BenchClock.init();
    for (0..warm + laps) |i| {
        var arm: usize = 0;
        const rec = struct {
            fn f(buf: []u64, a: usize, lp: usize, wm: usize, idx: usize, v: u64) void {
                if (idx >= wm) buf[a * lp + (idx - wm)] = v;
            }
        }.f;
        _ = timer.lap();
        {
            const a = try gatherMmGateUp(s, x.arr, gu.arr, slot_arrs[0], R, H, N);
            defer _ = mlx.mlx_array_free(a);
            try mlx.check(mlx.mlx_array_eval(a));
        }
        rec(t, arm, laps, warm, i, timer.lap());
        arm += 1;
        for (GATEUP_SWEEP) |v| {
            gateup_lpr = v.lanes;
            gateup_rows_per_lane = v.rpl;
            const a = try gateUpSwiglu(s, x.arr, gu.arr, slot_arrs[0]);
            try mlx.check(mlx.mlx_array_eval(a));
            _ = mlx.mlx_array_free(a);
            rec(t, arm, laps, warm, i, timer.lap());
            arm += 1;
        }
        {
            const a = try gatherMmDownReduce(s, hact.arr, dn.arr, slot_arrs[0], w, R, topk, N);
            defer _ = mlx.mlx_array_free(a);
            try mlx.check(mlx.mlx_array_eval(a));
        }
        rec(t, arm, laps, warm, i, timer.lap());
        arm += 1;
        for (DOWN_SWEEP) |v| {
            down_lpr = v.lanes;
            down_rows_per_lane = v.rpl;
            down_slot_groups = v.groups;
            down_stage_activation = v.stage;
            const a = try downReduce(s, hact.arr, dn.arr, slot_arrs[0], w);
            try mlx.check(mlx.mlx_array_eval(a));
            _ = mlx.mlx_array_free(a);
            rec(t, arm, laps, warm, i, timer.lap());
            arm += 1;
        }

        _ = timer.lap();
        {
            const v = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(v);
            for (0..REPS) |rep| {
                const a = try gatherMmGateUp(s, x.arr, gu.arr, slot_arrs[rep], R, H, N);
                defer _ = mlx.mlx_array_free(a);
                _ = mlx.mlx_vector_array_append_value(v, a);
            }
            try mlx.check(mlx.mlx_eval(v));
        }
        rec(t, arm, laps, warm, i, timer.lap() / REPS);
        arm += 1;
        for (GATEUP_SWEEP) |vr| {
            gateup_lpr = vr.lanes;
            gateup_rows_per_lane = vr.rpl;
            const v = mlx.mlx_vector_array_new();
            for (0..REPS) |rep| {
                const a = try gateUpSwiglu(s, x.arr, gu.arr, slot_arrs[rep]);
                _ = mlx.mlx_vector_array_append_value(v, a);
                _ = mlx.mlx_array_free(a);
            }
            try mlx.check(mlx.mlx_eval(v));
            _ = mlx.mlx_vector_array_free(v);
            rec(t, arm, laps, warm, i, timer.lap() / REPS);
            arm += 1;
        }
        {
            const v = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(v);
            for (0..REPS) |rep| {
                const a = try gatherMmDownReduce(s, hact.arr, dn.arr, slot_arrs[rep], w, R, topk, N);
                defer _ = mlx.mlx_array_free(a);
                _ = mlx.mlx_vector_array_append_value(v, a);
            }
            try mlx.check(mlx.mlx_eval(v));
        }
        rec(t, arm, laps, warm, i, timer.lap() / REPS);
        arm += 1;
        for (DOWN_SWEEP) |vr| {
            down_lpr = vr.lanes;
            down_rows_per_lane = vr.rpl;
            down_slot_groups = vr.groups;
            down_stage_activation = vr.stage;
            const v = mlx.mlx_vector_array_new();
            for (0..REPS) |rep| {
                const a = try downReduce(s, hact.arr, dn.arr, slot_arrs[rep], w);
                _ = mlx.mlx_vector_array_append_value(v, a);
                _ = mlx.mlx_array_free(a);
            }
            try mlx.check(mlx.mlx_eval(v));
            _ = mlx.mlx_vector_array_free(v);
            rec(t, arm, laps, warm, i, timer.lap() / REPS);
            arm += 1;
        }
    }

    const pairs: f64 = @floatFromInt(R * topk);
    const gu_bytes: f64 = pairs * @as(f64, @floatFromInt(2 * N)) * @as(f64, @floatFromInt(H)) * 2.0;
    const dn_bytes: f64 = pairs * @as(f64, @floatFromInt(H)) * @as(f64, @floatFromInt(N)) * 2.0;

    const report = struct {
        fn f(mode: []const u8, label: []const u8, U_: c_int, R_: c_int, lanes: c_int, rpl: c_int, groups: c_int, stage: bool, kib: f64, us: f64, bytes: f64, roof: f64) void {
            const gbs = bytes / (us * 1000.0);
            std.debug.print("U={d:>3} R={d:>2} {s:<10} {s:<18} lanes={d:>2} rpl={d:>1} g={d:>2} staged={d} tgmem={d:>5.2}KiB {d:>8.1} us {d:>6.1} GB/s {d:>5.1}% roof\n", .{ U_, R_, mode, label, lanes, rpl, groups, @intFromBool(stage), kib, us, gbs, 100.0 * gbs / roof });
        }
    }.f;

    for ([_][]const u8{ "solo", "pipelined" }, 0..) |mode, m| {
        const b0 = m * (2 + GATEUP_SWEEP.len + DOWN_SWEEP.len);
        report(mode, "gate_up gather_mm", U, R, 0, 0, 0, false, 0, medianUs(t[b0 * laps ..][0..laps]), gu_bytes, roof_gbs);
        for (GATEUP_SWEEP, 0..) |v, j| {
            report(mode, "gate_up kernel", U, R, v.lanes, v.rpl, 0, false, 0, medianUs(t[(b0 + 1 + j) * laps ..][0..laps]), gu_bytes, roof_gbs);
        }
        const b1 = b0 + 1 + GATEUP_SWEEP.len;
        report(mode, "down gather_mm", U, R, 0, 0, 0, false, 0, medianUs(t[b1 * laps ..][0..laps]), dn_bytes, roof_gbs);
        for (DOWN_SWEEP, 0..) |v, j| {
            const kib = if (v.stage) @as(f64, @floatFromInt(v.groups * N)) * 2.0 / 1024.0 else 0;
            report(mode, "down kernel", U, R, v.lanes, v.rpl, v.groups, v.stage, kib, medianUs(t[(b1 + 1 + j) * laps ..][0..laps]), dn_bytes, roof_gbs);
        }
    }
}

fn measureRoof(alloc: std.mem.Allocator, s: mlx.mlx_stream, rnd: std.Random) !f64 {
    const elems: c_int = 512 * 1024 * 1024;
    var big = try randBf16(alloc, s, rnd, &[_]c_int{ 1024, @divExact(elems, 1024) });
    defer big.deinit(alloc);
    const laps: usize = 50;
    var t = try alloc.alloc(u64, laps);
    defer alloc.free(t);
    var timer = BenchClock.init();
    for (0..laps + 10) |i| {
        _ = timer.lap();
        var r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(r);
        try mlx.check(mlx.mlx_sum(&r, big.arr, false, s));
        try mlx.check(mlx.mlx_array_eval(r));
        const d = timer.lap();
        if (i >= 10) t[i - 10] = d;
    }
    const us = medianUs(t);
    const bytes: f64 = @as(f64, @floatFromInt(elems)) * 2.0;
    const roof = bytes / (us * 1000.0);
    std.debug.print("roof: mlx_sum over {d:.2} GB bf16 = {d:.1} us = {d:.1} GB/s\n", .{ bytes / 1.0e9, us, roof });
    return roof;
}

test "bf16 slab kernel microbench" {
    if (!benchEnabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xBE0C4);
    const rnd = prng.random();
    const roof = try measureRoof(alloc, s, rnd);
    for ([_]c_int{ 128, 512 }) |U| {
        for ([_]c_int{ 1, 4 }) |R| {
            try benchLayer(alloc, s, rnd, U, R, roof);
            _ = mlx.mlx_clear_cache();
        }
    }
}
