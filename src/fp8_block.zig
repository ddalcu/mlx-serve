//! FP8 e4m3fn linears served from their source bytes: `[N, K]` u8 codes plus
//! one f32 scale per 128x128 tile, the weight being `code * scale` in f32 as
//! the checkpoint defines it. Decode rows take a GEMV that multiplies in f32;
//! wider inputs dequantize one weight to bf16 scratch for MLX's matmul, which
//! is the bf16-dequant route's exact arithmetic.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

pub const BLOCK: u32 = 128;

/// Widest input the GEMV serves; a wider one dequantizes into scratch.
pub var gemv_max_rows: c_int = 16;
/// Widest input that reads x straight from device memory; wider stages it. Every row of a
/// direct input keeps its one-row arithmetic, so this bounds a MiMo verify's width.
pub var gemv_direct_max_rows: c_int = 8;
/// Geometry overrides for the microbench sweep; 0 = the measured default.
pub var gemv_rows_per_sg: c_int = 0;
pub var gemv_sgs: c_int = 0;
pub var gemv_stage_tiles: c_int = 0;

/// How a stored weight's rows map onto its outputs. MiMo's QKV stacks `tp`
/// rank-local slabs of `[q | k | v]` rows and tiles each slab's scales on its
/// own, so a slab's last tile may be partial; a plain linear is one part.
pub const RowSplit = struct {
    tp: u32 = 1,
    parts: [3]u32,
    /// Factor on the third output, rounded to the output dtype first: the
    /// `v * scalar` MiMo applies before caching (`attention_value_scale`).
    v_scale: f32 = 1.0,

    pub fn dense(rows: u32) RowSplit {
        return .{ .parts = .{ rows, 0, 0 } };
    }

    /// The one rank count in {8, 4} that divides every part and tiles each
    /// rank's rows into exactly `scale_rows` grid rows.
    pub fn qkv(q: u64, k: u64, v: u64, scale_rows: u64) !RowSplit {
        var found: ?RowSplit = null;
        for ([_]u64{ 8, 4 }) |tp| {
            if (q % tp != 0 or k % tp != 0 or v % tp != 0) continue;
            const per = (q + k + v) / tp;
            if (scale_rows != tp * ((per + BLOCK - 1) / BLOCK)) continue;
            if (found != null) return error.AmbiguousQkvTensorParallelism;
            found = .{
                .tp = @intCast(tp),
                .parts = .{
                    std.math.cast(u32, q / tp) orelse return error.InvalidQkvGeometry,
                    std.math.cast(u32, k / tp) orelse return error.InvalidQkvGeometry,
                    std.math.cast(u32, v / tp) orelse return error.InvalidQkvGeometry,
                },
            };
        }
        return found orelse error.InvalidQkvGeometry;
    }

    pub fn rowsPerRank(self: RowSplit) u32 {
        return self.parts[0] + self.parts[1] + self.parts[2];
    }

    pub fn blocksPerRank(self: RowSplit) u32 {
        return (self.rowsPerRank() + BLOCK - 1) / BLOCK;
    }

    pub fn outputs(self: RowSplit) usize {
        return if (self.parts[1] == 0 and self.parts[2] == 0) 1 else 3;
    }

    pub fn partRows(self: RowSplit, part: usize) u32 {
        return self.tp * self.parts[part];
    }
};

/// Bit-exact e4m3fn decode of one packed word (4 codes) into 4 floats in byte
/// order: fp8.h's `(b & 127) << 7` half with the sign as the half sign bit;
/// IEEE multiply is sign-magnitude symmetric, so `* 256h` is exact for every
/// code including -0 (0x80).
pub const E4M3_HEADER =
    \\static inline float4 sushi_e4m3_decode4(uint w) {
    \\    uint lo = ((w & 0x007F007Fu) << 7) | ((w & 0x00800080u) << 8);
    \\    uint hs = w >> 8;
    \\    uint hi = ((hs & 0x007F007Fu) << 7) | ((hs & 0x00800080u) << 8);
    \\    half2 h02 = as_type<half2>(lo) * half2((half)256.0f);
    \\    half2 h13 = as_type<half2>(hi) * half2((half)256.0f);
    \\    return float4(float(h02.x), float(h13.x), float(h02.y), float(h13.y));
    \\}
;

const GEOMETRY =
    \\constexpr uint RPR = uint(P0 + P1 + P2);
    \\constexpr uint NROWS = uint(TP) * RPR;
    \\constexpr uint KC = uint(K) / 16u;
    \\constexpr uint KB = uint(K) / 128u;
    \\
;

/// One simdgroup owns NR stored rows for all M input rows.
const GEMV_HEAD = GEOMETRY ++
    \\uint lane = thread_index_in_simdgroup;
    \\uint row0 = (threadgroup_position_in_grid.x * uint(SGS) + simdgroup_index_in_threadgroup) * uint(NR);
    \\const device uint4* wp[NR];
    \\uint srow[NR];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < NR; ++r) {
    \\  uint o = metal::min(row0 + uint(r), NROWS - 1u);
    \\  uint rank = o / RPR;
    \\  srow[r] = (rank * uint(BPR) + (o - rank * RPR) / 128u) * KB;
    \\  wp[r] = (const device uint4*)(w + (size_t)o * (size_t)K);
    \\}
    \\float acc[M][NR];
    \\#pragma clang loop unroll(full)
    \\for (int m = 0; m < M; ++m) {
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < NR; ++r) acc[m][r] = 0.0f;
    \\}
    \\
;

/// Lanes stride each row in 16-byte chunks (a chunk never straddles a tile)
/// and read x from device memory: the narrow widths.
const GEMV_DIRECT =
    \\for (uint c = lane; c < KC; c += 32u) {
    \\  uint4 wr[NR];
    \\  float sc[NR];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < NR; ++r) {
    \\    wr[r] = wp[r][c];
    \\    sc[r] = scales[srow[r] + (c >> 3)];
    \\  }
    \\  #pragma clang loop unroll(full)
    \\  for (int i = 0; i < 4; ++i) {
    \\    float4 wv[NR];
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < NR; ++r) wv[r] = sushi_e4m3_decode4(wr[r][i]) * sc[r];
    \\    #pragma clang loop unroll(full)
    \\    for (int m = 0; m < M; ++m) {
    \\      float4 xv = float4(*((const device vec<T, 4>*)(x + (size_t)m * (size_t)K + c * 16u + uint(i) * 4u)));
    \\      #pragma clang loop unroll(full)
    \\      for (int r = 0; r < NR; ++r) acc[m][r] += dot(xv, wv[r]);
    \\    }
    \\  }
    \\}
    \\
;

/// x staged per S tiles in threadgroup memory, shared by the SGS*NR rows of
/// the threadgroup: past a few rows, re-reading x per row outweighs the weight.
const GEMV_STAGED =
    \\threadgroup float xs[M * 128 * S];
    \\uint tid = simdgroup_index_in_threadgroup * 32u + lane;
    \\for (uint k0 = 0; k0 < uint(K); k0 += 128u * uint(S)) {
    \\  for (uint i = tid; i < uint(M) * 128u * uint(S); i += 32u * uint(SGS)) {
    \\    uint m = i / (128u * uint(S));
    \\    xs[i] = float(x[(size_t)m * (size_t)K + k0 + (i - m * 128u * uint(S))]);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  #pragma clang loop unroll(full)
    \\  for (int b = 0; b < S; ++b) {
    \\    uint col = k0 + uint(b) * 128u + lane * 4u;
    \\    float4 wv[NR];
    \\    #pragma clang loop unroll(full)
    \\    for (int r = 0; r < NR; ++r)
    \\      wv[r] = sushi_e4m3_decode4(((const device uint*)wp[r])[col >> 2]) * scales[srow[r] + (col >> 7)];
    \\    #pragma clang loop unroll(full)
    \\    for (int m = 0; m < M; ++m) {
    \\      float4 xv = *((threadgroup const float4*)(xs + m * 128 * S + b * 128) + lane);
    \\      #pragma clang loop unroll(full)
    \\      for (int r = 0; r < NR; ++r) acc[m][r] += dot(xv, wv[r]);
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\
;

const GEMV_TAIL =
    \\#pragma clang loop unroll(full)
    \\for (int m = 0; m < M; ++m) {
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < NR; ++r) {
    \\    float total = simd_sum(acc[m][r]);
    \\    uint o = row0 + uint(r);
    \\    if (lane == 0u && o < NROWS) {
    \\      uint rank = o / RPR;
    \\      uint local = o - rank * RPR;
    \\
;

const GEMV_STORE_1 =
    \\      y[(size_t)m * NROWS + o] = T(total);
    \\    }
    \\  }
    \\}
;

const GEMV_STORE_3 =
    \\      if (local < uint(P0)) yq[(size_t)m * (TP * P0) + rank * P0 + local] = T(total);
    \\      else if (local < uint(P0 + P1)) yk[(size_t)m * (TP * P1) + rank * P1 + (local - P0)] = T(total);
    \\      else yv[(size_t)m * (TP * P2) + rank * P2 + (local - P0 - P1)] = T(float(T(total)) * as_type<float>(uint(VSB)));
    \\    }
    \\  }
    \\}
;

/// One thread per 16-byte chunk: `bf16(code * scale)`, the value the bf16
/// route stored, written row-major into the output its row belongs to.
const DEQUANT_BODY = GEOMETRY ++
    \\uint c = thread_position_in_grid.x;
    \\uint o = thread_position_in_grid.y;
    \\if (c >= KC || o >= NROWS) return;
    \\uint rank = o / RPR;
    \\uint local = o - rank * RPR;
    \\float sc = scales[(rank * uint(BPR) + local / 128u) * KB + (c >> 3)];
    \\uint4 wr = ((const device uint4*)(w + (size_t)o * (size_t)K))[c];
    \\
;

const DEQUANT_DST_1 =
    \\device T* dst = y + (size_t)o * (size_t)K;
    \\
;

const DEQUANT_DST_3 =
    \\device T* dst = local < uint(P0) ? yq + (size_t)(rank * P0 + local) * (size_t)K
    \\    : local < uint(P0 + P1) ? yk + (size_t)(rank * P1 + local - P0) * (size_t)K
    \\    : yv + (size_t)(rank * P2 + local - P0 - P1) * (size_t)K;
    \\
;

const DEQUANT_TAIL =
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < 4; ++i) {
    \\  *((device vec<T, 4>*)(dst + c * 16u + uint(i) * 4u)) = vec<T, 4>(sushi_e4m3_decode4(wr[i]) * sc);
    \\}
;

const Kind = enum { gemv, staged, dequant };

const KernelSlot = struct {
    kernel: ?mlx.mlx_fast_metal_kernel = null,
    engaged: bool = false,
};

var kernels: [3][2]KernelSlot = @splat(@splat(.{}));

fn getKernel(kind: Kind, three: bool) !mlx.mlx_fast_metal_kernel {
    const slot = &kernels[@backingInt(kind)][@intFromBool(three)];
    if (slot.kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w", "scales" };
    const outs1 = [_][*:0]const u8{"y"};
    const outs3 = [_][*:0]const u8{ "yq", "yk", "yv" };
    const inputs: []const [*:0]const u8 = if (kind == .dequant) input_names[1..] else &input_names;
    const outputs: []const [*:0]const u8 = if (three) &outs3 else &outs1;
    const source: [*:0]const u8 = switch (kind) {
        .gemv => if (three) GEMV_HEAD ++ GEMV_DIRECT ++ GEMV_TAIL ++ GEMV_STORE_3 else GEMV_HEAD ++ GEMV_DIRECT ++ GEMV_TAIL ++ GEMV_STORE_1,
        .staged => if (three) GEMV_HEAD ++ GEMV_STAGED ++ GEMV_TAIL ++ GEMV_STORE_3 else GEMV_HEAD ++ GEMV_STAGED ++ GEMV_TAIL ++ GEMV_STORE_1,
        .dequant => if (three) DEQUANT_BODY ++ DEQUANT_DST_3 ++ DEQUANT_TAIL else DEQUANT_BODY ++ DEQUANT_DST_1 ++ DEQUANT_TAIL,
    };
    const name: [*:0]const u8 = switch (kind) {
        .gemv => if (three) "sushi_fp8_block_gemv3" else "sushi_fp8_block_gemv",
        .staged => if (three) "sushi_fp8_block_gemv_staged3" else "sushi_fp8_block_gemv_staged",
        .dequant => if (three) "sushi_fp8_block_dequant3" else "sushi_fp8_block_dequant",
    };
    const in_vec = mlx.mlx_vector_string_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outputs.ptr, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, E4M3_HEADER, true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    slot.kernel = kernel;
    return kernel;
}

const CfgKey = struct {
    kind: Kind,
    dtype: mlx.mlx_dtype,
    m: c_int,
    k: c_int,
    tp: u32,
    parts: [3]u32,
    nr: c_int,
    sgs: c_int,
    tiles: c_int,
    /// f32 bits of `RowSplit.v_scale` rounded to `dtype`.
    vsb: u32 = 0,
};

const CFG_CAP = 64;
var cfg_keys: [CFG_CAP]CfgKey = undefined;
var cfg_vals: [CFG_CAP]?mlx.mlx_fast_metal_kernel_config = @splat(null);
var cfg_next: usize = 0;

fn cachedConfig(key: CfgKey) !mlx.mlx_fast_metal_kernel_config {
    for (cfg_vals, 0..) |c, i| {
        if (c != null and std.meta.eql(cfg_keys[i], key)) return c.?;
    }
    const cfg = try buildConfig(key);
    const victim = cfg_next % CFG_CAP;
    if (cfg_vals[victim]) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
    cfg_keys[victim] = key;
    cfg_vals[victim] = cfg;
    cfg_next += 1;
    return cfg;
}

fn buildConfig(key: CfgKey) !mlx.mlx_fast_metal_kernel_config {
    const split = RowSplit{ .tp = key.tp, .parts = key.parts };
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const nrows: c_int = @intCast(split.tp * split.rowsPerRank());
    for (0..split.outputs()) |p| {
        const rows: c_int = @intCast(split.partRows(p));
        const shape = [_]c_int{ if (key.kind == .dequant) rows else key.m, if (key.kind == .dequant) key.k else rows };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 2, key.dtype));
    }
    switch (key.kind) {
        .gemv, .staged => {
            const rows_per_group = key.nr * key.sgs;
            const groups = @divTrunc(nrows + rows_per_group - 1, rows_per_group);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * key.sgs * groups, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32 * key.sgs, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "M", key.m));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NR", key.nr));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SGS", key.sgs));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "S", key.tiles));
        },
        .dequant => {
            const kc = @divExact(key.k, 16);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, kc, nrows, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(kc, 256), 1, 1));
        },
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", key.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", key.k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TP", @intCast(split.tp)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "P0", @intCast(split.parts[0])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "P1", @intCast(split.parts[1])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "P2", @intCast(split.parts[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BPR", @intCast(split.blocksPerRank())));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "VSB", @bitCast(key.vsb)));
    return cfg;
}

fn apply(
    s: mlx.mlx_stream,
    kind: Kind,
    inputs: []const mlx.mlx_array,
    key: CfgKey,
    split: RowSplit,
    out: []mlx.mlx_array,
) !void {
    const three = split.outputs() == 3;
    const kernel = try getKernel(kind, three);
    const cfg = try cachedConfig(key);
    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, s));
    if (mlx.mlx_vector_array_size(out_vec) != split.outputs()) return error.MetalKernelBadOutputCount;
    @memset(out[0..split.outputs()], .{});
    errdefer freeOutputs(out[0..split.outputs()]);
    for (out[0..split.outputs()], 0..) |*o, i| {
        try mlx.check(mlx.mlx_vector_array_get(o, out_vec, i));
    }
    const slot = &kernels[@backingInt(kind)][@intFromBool(three)];
    if (!slot.engaged) {
        slot.engaged = true;
        log.info("[fp8] block-128 {s} engaged: rows={d} in={d} tp={d} parts={d}/{d}/{d} ({s})\n", .{
            @tagName(kind), key.m, key.k, split.tp, split.parts[0], split.parts[1], split.parts[2], @tagName(key.dtype),
        });
    }
}

fn freeOutputs(out: []mlx.mlx_array) void {
    for (out) |*a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
        a.* = .{};
    }
}

fn checkWeight(s: mlx.mlx_stream, w: mlx.mlx_array, scales: mlx.mlx_array, split: RowSplit, outs: usize) !c_int {
    if (!mlx.streamIsGpu(s)) return error.MetalKernelNeedsGpuStream;
    if (mlx.mlx_array_dtype(w) != .uint8) return error.Fp8WeightNotU8;
    if (mlx.mlx_array_dtype(scales) != .float32) return error.Fp8ScalesNotF32;
    const wsh = mlx.getShape(w);
    const ssh = mlx.getShape(scales);
    if (wsh.len != 2 or ssh.len != 2) return error.Fp8ShapeMismatch;
    const k = wsh[1];
    if (k <= 0 or @rem(k, BLOCK) != 0) return error.Fp8ShapeMismatch;
    if (wsh[0] != split.tp * split.rowsPerRank()) return error.Fp8ShapeMismatch;
    if (ssh[1] != @divExact(k, BLOCK) or ssh[0] < split.tp * split.blocksPerRank()) return error.Fp8ScaleGridMismatch;
    if (outs < split.outputs()) return error.Fp8OutputCount;
    return k;
}

fn gemvKey(dtype: mlx.mlx_dtype, m: c_int, k: c_int, split: RowSplit) CfgKey {
    const direct = m <= gemv_direct_max_rows;
    // Past four rows two stored rows per simdgroup share each x read; no row's sum moves.
    const nr: c_int = if (gemv_rows_per_sg > 0) gemv_rows_per_sg else if (!direct) 4 else if (m <= 4) 1 else 2;
    const sgs: c_int = if (gemv_sgs > 0) gemv_sgs else if (!direct) 8 else if (m <= 4) 2 else 8;
    var tiles: c_int = if (gemv_stage_tiles > 0) gemv_stage_tiles else 1;
    if (@rem(k, 128 * tiles) != 0) tiles = 1;
    return .{ .kind = if (direct) .gemv else .staged, .dtype = dtype, .m = m, .k = k, .tp = split.tp, .parts = split.parts, .nr = nr, .sgs = sgs, .tiles = if (direct) 0 else tiles, .vsb = @bitCast(roundedTo(dtype, split.v_scale)) };
}

/// `v` as `mlx_astype` would store it in `dtype`, widened back to f32.
fn roundedTo(dtype: mlx.mlx_dtype, v: f32) f32 {
    return switch (dtype) {
        .bfloat16 => bfToF32(bf16Rne(v)),
        .float16 => @floatCast(@as(f16, @floatCast(v))),
        else => v,
    };
}

/// `out[p] = x @ W_p^T` for each output of `split`, shaped `x.shape[:-1] + [rows_p]`.
/// Caller frees `out[0..split.outputs()]`.
pub fn project(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    scales: mlx.mlx_array,
    split: RowSplit,
    out: []mlx.mlx_array,
) !void {
    const k = try checkWeight(s, w, scales, split, out.len);
    @memset(out[0..split.outputs()], .{});
    errdefer freeOutputs(out[0..split.outputs()]);

    const xsh = mlx.getShape(x);
    if (xsh.len == 0 or xsh[xsh.len - 1] != k) return error.Fp8ShapeMismatch;
    const x_dtype = mlx.mlx_array_dtype(x);
    if (x_dtype != .bfloat16 and x_dtype != .float16 and x_dtype != .float32) return error.Fp8ActivationDtype;
    var lead: [8]c_int = undefined;
    if (xsh.len > lead.len) return error.Fp8ShapeMismatch;
    var m: c_int = 1;
    for (xsh[0 .. xsh.len - 1], 0..) |d, i| {
        lead[i] = d;
        m *= d;
    }
    if (m <= 0) return error.Fp8ShapeMismatch;

    if (m <= gemv_max_rows) {
        const key = gemvKey(x_dtype, m, k, split);
        var flat: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
        defer for (&flat) |*a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a.*);
        };
        try apply(s, key.kind, &.{ x, w, scales }, key, split, &flat);
        for (0..split.outputs()) |p| {
            lead[xsh.len - 1] = @intCast(split.partRows(p));
            try mlx.check(mlx.mlx_reshape(&out[p], flat[p], &lead, xsh.len, s));
        }
        return;
    }

    var dense: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
    defer for (&dense) |*a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a.*);
    };
    try dequantize(s, w, scales, split, &dense);
    for (0..split.outputs()) |p| {
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, dense[p], s));
        try mlx.check(mlx.mlx_matmul(&out[p], x, wt, s));
    }
    if (split.outputs() == 3 and split.v_scale != 1.0) {
        const raw = mlx.mlx_array_new_float(split.v_scale);
        defer _ = mlx.mlx_array_free(raw);
        var scalar = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scalar);
        try mlx.check(mlx.mlx_astype(&scalar, raw, mlx.mlx_array_dtype(out[2]), s));
        var scaled = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&scaled, out[2], scalar, s));
        _ = mlx.mlx_array_free(out[2]);
        out[2] = scaled;
    }
}

/// Each output's rows as dense bf16 `[rows_p, K]`, rounded once from the f32
/// product. Caller frees `out[0..split.outputs()]`.
pub fn dequantize(s: mlx.mlx_stream, w: mlx.mlx_array, scales: mlx.mlx_array, split: RowSplit, out: []mlx.mlx_array) !void {
    const k = try checkWeight(s, w, scales, split, out.len);
    const key = CfgKey{ .kind = .dequant, .dtype = .bfloat16, .m = 0, .k = k, .tp = split.tp, .parts = split.parts, .nr = 0, .sgs = 0, .tiles = 0 };
    try apply(s, .dequant, &.{ w, scales }, key, split, out);
}

/// The dense case: one output, `x @ W^T`.
pub fn linear(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, scales: mlx.mlx_array) !mlx.mlx_array {
    const wsh = mlx.getShape(w);
    if (wsh.len != 2 or wsh[0] <= 0) return error.Fp8ShapeMismatch;
    var out: [1]mlx.mlx_array = .{.{}};
    try project(s, x, w, scales, RowSplit.dense(@intCast(wsh[0])), &out);
    return out[0];
}

const testing = std.testing;

fn e4m3Value(code: u8) f64 {
    const sign: f64 = if (code & 0x80 != 0) -1 else 1;
    const e: i32 = (code >> 3) & 15;
    const m: f64 = @floatFromInt(code & 7);
    if (e == 0) return sign * (m / 8.0) * std.math.pow(f64, 2, -6);
    return sign * (1.0 + m / 8.0) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

fn bf16Rne(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn bfToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

/// A synthetic FP8 weight in the source layout, scales in the real range.
const TestWeight = struct {
    codes: []u8,
    scales: []f32,
    n: usize,
    k: usize,
    split: RowSplit,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,

    fn init(alloc: std.mem.Allocator, rnd: std.Random, split: RowSplit, k: usize) !TestWeight {
        const n: usize = split.tp * split.rowsPerRank();
        const srows: usize = split.tp * split.blocksPerRank();
        const codes = try alloc.alloc(u8, n * k);
        for (codes) |*c| {
            c.* = rnd.int(u8);
            if (c.* & 0x7f == 0x7f) c.* ^= 1;
        }
        const scales = try alloc.alloc(f32, srows * (k / BLOCK));
        for (scales) |*v| v.* = 2e-5 + rnd.float(f32) * 1e-3;
        return .{
            .codes = codes,
            .scales = scales,
            .n = n,
            .k = k,
            .split = split,
            .w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ @intCast(n), @intCast(k) }, 2, .uint8),
            .sc = mlx.mlx_array_new_data(scales.ptr, &[_]c_int{ @intCast(srows), @intCast(k / BLOCK) }, 2, .float32),
        };
    }

    fn deinit(self: *TestWeight, alloc: std.mem.Allocator) void {
        _ = mlx.mlx_array_free(self.w);
        _ = mlx.mlx_array_free(self.sc);
        alloc.free(self.codes);
        alloc.free(self.scales);
    }

    fn scaleOf(self: *const TestWeight, row: usize, col: usize) f32 {
        const rpr = self.split.rowsPerRank();
        const rank = row / rpr;
        const srow = rank * self.split.blocksPerRank() + (row - rank * rpr) / BLOCK;
        return self.scales[srow * (self.k / BLOCK) + col / BLOCK];
    }

    /// The source's own weight: the f32 product of code and tile scale.
    fn value(self: *const TestWeight, row: usize, col: usize) f32 {
        return @as(f32, @floatCast(e4m3Value(self.codes[row * self.k + col]))) * self.scaleOf(row, col);
    }

    /// Stored row `row`'s (output, row within output).
    fn destOf(self: *const TestWeight, row: usize) struct { part: usize, row: usize } {
        const rpr = self.split.rowsPerRank();
        const rank = row / rpr;
        var local = row - rank * rpr;
        for (0..3) |p| {
            if (local < self.split.parts[p]) return .{ .part = p, .row = rank * self.split.parts[p] + local };
            local -= self.split.parts[p];
        }
        unreachable;
    }
};

const ErrStats = struct { max: f64 = 0, rms: f64 = 0, max_rel_summands: f64 = 0, finite: bool = true };

/// Error of `got` (one output's [m, rows] values) against the fp64 truth of the
/// source weights, each element also relative to its sum |w*x|.
fn errStats(tw: *const TestWeight, x: []const f32, m: usize, part: usize, got: []const f32) ErrStats {
    var st = ErrStats{};
    var acc: f64 = 0;
    var count: usize = 0;
    const rows = tw.split.partRows(part);
    for (0..tw.n) |row| {
        const d = tw.destOf(row);
        if (d.part != part) continue;
        for (0..m) |mi| {
            var truth: f64 = 0;
            var summands: f64 = 0;
            for (0..tw.k) |j| {
                const term = @as(f64, x[mi * tw.k + j]) * e4m3Value(tw.codes[row * tw.k + j]) * @as(f64, tw.scaleOf(row, j));
                truth += term;
                summands += @abs(term);
            }
            const g: f64 = got[mi * rows + d.row];
            if (!std.math.isFinite(g)) st.finite = false;
            const e = @abs(g - truth);
            st.max = @max(st.max, e);
            if (summands > 0) st.max_rel_summands = @max(st.max_rel_summands, e / summands);
            acc += e * e;
            count += 1;
        }
    }
    st.rms = @sqrt(acc / @as(f64, @floatFromInt(count)));
    return st;
}

fn readAs32(alloc: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    const src = mlx.mlx_array_data_float32(c) orelse return error.TestUnreadable;
    return alloc.dupe(f32, src[0..n]);
}

fn randomX(alloc: std.mem.Allocator, rnd: std.Random, m: usize, k: usize) ![]f32 {
    const x = try alloc.alloc(f32, m * k);
    for (x) |*v| v.* = bfToF32(bf16Rne((rnd.float(f32) - 0.5) * 8.0));
    return x;
}

/// The bf16-dequant route this module replaces: weights rounded to bf16 on
/// the host, uploaded `[N, K]`, and MLX's matmul against the transposed view.
fn bf16Route(alloc: std.mem.Allocator, s: mlx.mlx_stream, tw: *const TestWeight, part: usize, xa: mlx.mlx_array) !mlx.mlx_array {
    const rows = tw.split.partRows(part);
    const bits = try alloc.alloc(u16, rows * tw.k);
    defer alloc.free(bits);
    for (0..tw.n) |row| {
        const d = tw.destOf(row);
        if (d.part != part) continue;
        for (0..tw.k) |j| bits[d.row * tw.k + j] = bf16Rne(tw.value(row, j));
    }
    const wb = mlx.mlx_array_new_data(bits.ptr, &[_]c_int{ @intCast(rows), @intCast(tw.k) }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(wb);
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wb, s));
    var y = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&y, xa, wt, s));
    return y;
}

fn uploadX(s: mlx.mlx_stream, x: []const f32, m: usize, k: usize, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const a = mlx.mlx_array_new_data(x.ptr, &[_]c_int{ 1, @intCast(m), @intCast(k) }, 3, .float32);
    defer _ = mlx.mlx_array_free(a);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, a, dtype, s));
    return out;
}

const PARITY_SEEDS = [_]u64{ 0xF8B10C, 0x5EED8, 0xE4A3 };

const PARITY_SPLITS = [_]RowSplit{
    RowSplit.dense(256),
    .{ .tp = 4, .parts = .{ 96, 24, 16 } },
    .{ .tp = 2, .parts = .{ 128, 96, 64 } },
};

test "fp8 block GEMV is no worse than the bf16-dequant route against fp64 truth" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    for (PARITY_SEEDS) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        for (PARITY_SPLITS) |split| {
            var tw = try TestWeight.init(alloc, rnd, split, 512);
            defer tw.deinit(alloc);
            for ([_]usize{ 1, 2, 3, 8, 16 }) |m| {
                const x = try randomX(alloc, rnd, m, tw.k);
                defer alloc.free(x);
                const xb = try uploadX(s, x, m, tw.k, .bfloat16);
                defer _ = mlx.mlx_array_free(xb);
                var out: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
                try project(s, xb, tw.w, tw.sc, split, &out);
                defer for (out[0..split.outputs()]) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                for (0..split.outputs()) |p| {
                    try testing.expectEqualSlices(c_int, &[_]c_int{ 1, @intCast(m), @intCast(split.partRows(p)) }, mlx.getShape(out[p]));
                    const got = try readAs32(alloc, s, out[p]);
                    defer alloc.free(got);
                    const ref = try bf16Route(alloc, s, &tw, p, xb);
                    defer _ = mlx.mlx_array_free(ref);
                    const want = try readAs32(alloc, s, ref);
                    defer alloc.free(want);
                    const ks = errStats(&tw, x, m, p, got);
                    const rs = errStats(&tw, x, m, p, want);
                    try testing.expect(ks.finite and rs.finite);
                    try testing.expect(ks.rms <= rs.rms);
                    try testing.expect(ks.max <= rs.max);
                }
            }
        }
    }
}

test "fp8 block GEMV in f32 errs by no more than an f32 accumulation over the summands" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    for (PARITY_SEEDS) |seed| {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        var tw = try TestWeight.init(alloc, rnd, PARITY_SPLITS[1], 1024);
        defer tw.deinit(alloc);
        for ([_]usize{ 1, 4, 16 }) |m| {
            const x = try randomX(alloc, rnd, m, tw.k);
            defer alloc.free(x);
            const xf = try uploadX(s, x, m, tw.k, .float32);
            defer _ = mlx.mlx_array_free(xf);
            var out: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
            try project(s, xf, tw.w, tw.sc, tw.split, &out);
            defer for (out) |a| {
                _ = mlx.mlx_array_free(a);
            };
            const ceiling = @as(f64, @floatFromInt(tw.k + 2)) * std.math.pow(f64, 2, -24);
            for (0..3) |p| {
                const got = try readAs32(alloc, s, out[p]);
                defer alloc.free(got);
                const st = errStats(&tw, x, m, p, got);
                try testing.expect(st.finite);
                try testing.expect(st.max_rel_summands <= ceiling);
            }
        }
    }
}

test "fp8 block dequant writes the bf16 route's bytes for every code" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xDE0A7);
    const rnd = prng.random();
    const split = RowSplit{ .tp = 2, .parts = .{ 128, 96, 64 } };
    var tw = try TestWeight.init(alloc, rnd, split, 256);
    defer tw.deinit(alloc);
    // Every finite code, both signs, in every scale tile.
    for (tw.codes, 0..) |*c, i| {
        c.* = @truncate(i *% 7);
        if (c.* & 0x7f == 0x7f) c.* ^= 1;
    }
    const w = mlx.mlx_array_new_data(tw.codes.ptr, &[_]c_int{ @intCast(tw.n), @intCast(tw.k) }, 2, .uint8);
    defer _ = mlx.mlx_array_free(w);
    var out: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
    try dequantize(s, w, tw.sc, split, &out);
    defer for (out) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var views: [3][]const u16 = undefined;
    for (out, 0..) |a, p| {
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(a));
        try mlx.check(mlx.mlx_array_eval(a));
        views[p] = (mlx.mlx_array_data_bfloat16(a) orelse return error.TestUnreadable)[0 .. split.partRows(p) * tw.k];
    }
    for (0..tw.n) |row| {
        const d = tw.destOf(row);
        for (0..tw.k) |j| try testing.expectEqual(bf16Rne(tw.value(row, j)), views[d.part][d.row * tw.k + j]);
    }
}

test "fp8 block QKV split routes rank-local rows and their partial tiles" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    for ([_]u32{ 4, 8 }) |tp| {
        const q_per = 128;
        const k_per = 96;
        const v_per = 64;
        const split = try RowSplit.qkv(q_per * tp, k_per * tp, v_per * tp, tp * 3);
        try testing.expectEqual(tp, split.tp);
        const k: usize = 128;
        const n: usize = tp * (q_per + k_per + v_per);
        const codes = try alloc.alloc(u8, n * k);
        defer alloc.free(codes);
        @memset(codes, 0x38);
        var scales: [24]f32 = undefined;
        for (&scales, 0..) |*v, i| v.* = @floatFromInt(i + 1);
        const w = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ @intCast(n), @intCast(k) }, 2, .uint8);
        defer _ = mlx.mlx_array_free(w);
        const sc = mlx.mlx_array_new_data(&scales, &[_]c_int{ @intCast(tp * 3), 1 }, 2, .float32);
        defer _ = mlx.mlx_array_free(sc);
        // One GEMV width and one dequant width; x = ones makes each output
        // 128 * (its row's tile scale).
        for ([_]usize{ 1, 24 }) |m| {
            const ones = try alloc.alloc(f32, m * k);
            defer alloc.free(ones);
            @memset(ones, 1.0);
            const xb = try uploadX(s, ones, m, k, .bfloat16);
            defer _ = mlx.mlx_array_free(xb);
            var out: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
            try project(s, xb, w, sc, split, &out);
            defer for (out) |a| {
                _ = mlx.mlx_array_free(a);
            };
            const parts = [_][]const f32{
                try readAs32(alloc, s, out[0]),
                try readAs32(alloc, s, out[1]),
                try readAs32(alloc, s, out[2]),
            };
            defer for (parts) |p| alloc.free(p);
            const last = m - 1;
            for (0..tp) |rank| {
                const tile: f32 = @floatFromInt(rank * 3);
                for (0..q_per) |r| try testing.expectEqual(128 * (tile + 1), parts[0][last * tp * q_per + rank * q_per + r]);
                for (0..k_per) |r| try testing.expectEqual(128 * (tile + 2), parts[1][last * tp * k_per + rank * k_per + r]);
                for (0..v_per) |r| {
                    const want: f32 = if (r < 32) tile + 2 else tile + 3;
                    try testing.expectEqual(128 * want, parts[2][last * tp * v_per + rank * v_per + r]);
                }
            }
        }
    }
}

test "fp8 block QKV folds the value scale into V as the composed multiply rounds it" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x7A1E);
    const rnd = prng.random();
    var tw = try TestWeight.init(alloc, rnd, PARITY_SPLITS[1], 512);
    defer tw.deinit(alloc);
    var scaled_split = tw.split;
    scaled_split.v_scale = 0.707;
    // The direct GEMV, the staged GEMV and the dequant route.
    for ([_]usize{ 1, 3, 8, 16, 24 }) |m| {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 }) |dt| {
            const x = try randomX(alloc, rnd, m, tw.k);
            defer alloc.free(x);
            const xa = try uploadX(s, x, m, tw.k, dt);
            defer _ = mlx.mlx_array_free(xa);
            var plain: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
            try project(s, xa, tw.w, tw.sc, tw.split, &plain);
            defer for (plain) |a| {
                _ = mlx.mlx_array_free(a);
            };
            var folded: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
            try project(s, xa, tw.w, tw.sc, scaled_split, &folded);
            defer for (folded) |a| {
                _ = mlx.mlx_array_free(a);
            };
            const raw = mlx.mlx_array_new_float(0.707);
            defer _ = mlx.mlx_array_free(raw);
            var scalar = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(scalar);
            try mlx.check(mlx.mlx_astype(&scalar, raw, mlx.mlx_array_dtype(plain[2]), s));
            var composed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(composed);
            try mlx.check(mlx.mlx_multiply(&composed, plain[2], scalar, s));
            try testing.expectEqual(mlx.mlx_array_dtype(plain[2]), mlx.mlx_array_dtype(folded[2]));
            const refs = [_]mlx.mlx_array{ plain[0], plain[1], composed };
            for (refs, folded) |want_arr, got_arr| {
                const want = try readAs32(alloc, s, want_arr);
                defer alloc.free(want);
                const got = try readAs32(alloc, s, got_arr);
                defer alloc.free(got);
                try testing.expectEqualSlices(f32, want, got);
            }
        }
    }
}

fn gpuRandomFp8(s: mlx.mlx_stream, split: RowSplit, k: c_int, seed: u64) !struct { w: mlx.mlx_array, sc: mlx.mlx_array } {
    const n: c_int = @intCast(split.tp * split.rowsPerRank());
    const srows: c_int = @intCast(split.tp * split.blocksPerRank());
    var k0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k0);
    try mlx.check(mlx.mlx_random_key(&k0, seed));
    var k1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k1);
    try mlx.check(mlx.mlx_random_key(&k1, seed + 1));
    // Codes 0..253 skip both NaN encodings (0x7f, 0xff).
    var bits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bits);
    const lo = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_int(254);
    defer _ = mlx.mlx_array_free(hi);
    try mlx.check(mlx.mlx_random_randint(&bits, lo, hi, &[_]c_int{ n, k }, 2, .int32, k0, s));
    const nan_lo = mlx.mlx_array_new_int(0x7f);
    defer _ = mlx.mlx_array_free(nan_lo);
    var is_nan = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(is_nan);
    try mlx.check(mlx.mlx_equal(&is_nan, bits, nan_lo, s));
    const zero = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero);
    var fixed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(fixed);
    try mlx.check(mlx.mlx_where(&fixed, is_nan, zero, bits, s));
    var w = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_astype(&w, fixed, .uint8, s));
    const s_lo = mlx.mlx_array_new_float(2e-5);
    defer _ = mlx.mlx_array_free(s_lo);
    const s_hi = mlx.mlx_array_new_float(1e-3);
    defer _ = mlx.mlx_array_free(s_hi);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, s_lo, s_hi, &[_]c_int{ srows, @divExact(k, 128) }, 2, .float32, k1, s));
    var sc = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_contiguous(&sc, u, false, s));
    try mlx.check(mlx.mlx_array_eval(w));
    try mlx.check(mlx.mlx_array_eval(sc));
    return .{ .w = w, .sc = sc };
}

test "fp8 block direct GEMV keeps each row's one-row arithmetic up to 8 rows at MiMo's trunk shapes" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const alloc = testing.allocator;
    const shapes = [_]struct { split: RowSplit, k: c_int }{
        .{ .split = .{ .tp = 4, .parts = .{ 3072, 192, 128 }, .v_scale = 0.707 }, .k = 4096 },
        .{ .split = .{ .tp = 4, .parts = .{ 3072, 384, 256 }, .v_scale = 0.707 }, .k = 4096 },
        .{ .split = RowSplit.dense(16384), .k = 4096 },
        .{ .split = RowSplit.dense(4096), .k = 16384 },
    };
    for (shapes, 0..) |shape, si| {
        const tw = try gpuRandomFp8(s, shape.split, shape.k, 0xD1EC7 + si);
        defer _ = mlx.mlx_array_free(tw.w);
        defer _ = mlx.mlx_array_free(tw.sc);
        var prng = std.Random.DefaultPrng.init(0x8F0 + si);
        const x = try randomX(alloc, prng.random(), 8, @intCast(shape.k));
        defer alloc.free(x);
        var rows: usize = 2;
        while (rows <= 8) : (rows += 1) {
            const xb = try uploadX(s, x[0 .. rows * @as(usize, @intCast(shape.k))], rows, @intCast(shape.k), .bfloat16);
            defer _ = mlx.mlx_array_free(xb);
            try testing.expectEqual(Kind.gemv, gemvKey(.bfloat16, @intCast(rows), shape.k, shape.split).kind);
            var wide: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
            try project(s, xb, tw.w, tw.sc, shape.split, &wide);
            defer for (wide[0..shape.split.outputs()]) |a| {
                _ = mlx.mlx_array_free(a);
            };
            for (0..rows) |r| {
                const k: usize = @intCast(shape.k);
                const x1 = try uploadX(s, x[r * k .. (r + 1) * k], 1, k, .bfloat16);
                defer _ = mlx.mlx_array_free(x1);
                var one: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
                try project(s, x1, tw.w, tw.sc, shape.split, &one);
                defer for (one[0..shape.split.outputs()]) |a| {
                    _ = mlx.mlx_array_free(a);
                };
                for (0..shape.split.outputs()) |p| {
                    const n = shape.split.partRows(p);
                    const got = try readAs32(alloc, s, wide[p]);
                    defer alloc.free(got);
                    const want = try readAs32(alloc, s, one[p]);
                    defer alloc.free(want);
                    try testing.expectEqualSlices(f32, want, got[r * n .. (r + 1) * n]);
                }
            }
        }
    }
}

test "fp8 block refuses weights outside its contract by name" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const codes: [128 * 128]u8 = @splat(0x38);
    const w = mlx.mlx_array_new_data(&codes, &[_]c_int{ 128, 128 }, 2, .uint8);
    defer _ = mlx.mlx_array_free(w);
    const one = [_]f32{1};
    const sc = mlx.mlx_array_new_data(&one, &[_]c_int{ 1, 1 }, 2, .float32);
    defer _ = mlx.mlx_array_free(sc);
    const xs: [128]f32 = @splat(1);
    const x = mlx.mlx_array_new_data(&xs, &[_]c_int{ 1, 128 }, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    var out: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
    try testing.expectError(error.Fp8WeightNotU8, project(s, x, sc, sc, RowSplit.dense(128), &out));
    try testing.expectError(error.Fp8ShapeMismatch, project(s, x, w, sc, RowSplit.dense(64), &out));
    try testing.expectError(error.Fp8ScaleGridMismatch, project(s, x, w, sc, .{ .tp = 2, .parts = .{ 32, 16, 16 } }, &out));
    try testing.expectError(error.InvalidQkvGeometry, RowSplit.qkv(128, 96, 64, 5));
    const xw = mlx.mlx_array_new_data(&xs, &[_]c_int{ 2, 64 }, 2, .float32);
    defer _ = mlx.mlx_array_free(xw);
    try testing.expectError(error.Fp8ShapeMismatch, project(s, xw, w, sc, RowSplit.dense(128), &out));
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

fn medianNs(samples: []u64) f64 {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    return @floatFromInt(samples[samples.len / 2]);
}

/// One trunk linear at MiMo's geometry in all three storages the A/B compares.
const BenchShape = struct { name: []const u8, split: RowSplit, k: c_int, fp8: bool = true };

const BENCH_SHAPES = [_]BenchShape{
    .{ .name = "qkv global", .split = .{ .tp = 4, .parts = .{ 3072, 192, 128 } }, .k = 4096 },
    .{ .name = "qkv sliding", .split = .{ .tp = 4, .parts = .{ 3072, 384, 256 } }, .k = 4096 },
    .{ .name = "L0 gate/up", .split = RowSplit.dense(16384), .k = 4096 },
    .{ .name = "L0 down", .split = RowSplit.dense(4096), .k = 16384 },
    .{ .name = "o_proj", .split = RowSplit.dense(4096), .k = 8192, .fp8 = false },
    .{ .name = "lm_head", .split = RowSplit.dense(152576), .k = 4096, .fp8 = false },
};

const BENCH_COPIES = 6;

/// `fp8_staged` stages x at every width, against the arm `fp8` takes there.
const BenchArm = enum { bf16, fp8, fp8_staged, affine8 };

const BenchWeights = struct {
    fp8_w: [BENCH_COPIES]mlx.mlx_array = @splat(.{}),
    fp8_s: [BENCH_COPIES]mlx.mlx_array = @splat(.{}),
    /// Per copy, per output: the bf16 route's `[in, out]` view and the affine
    /// triple, as the served paths bind them.
    bf16_t: [BENCH_COPIES][3]mlx.mlx_array = @splat(@splat(.{})),
    aq: [BENCH_COPIES][3][3]mlx.mlx_array = @splat(@splat(@splat(.{}))),

    fn deinit(self: *BenchWeights) void {
        for (&self.fp8_w) |a| _ = mlx.mlx_array_free(a);
        for (&self.fp8_s) |a| _ = mlx.mlx_array_free(a);
        for (&self.bf16_t) |row| for (row) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (&self.aq) |row| for (row) |t| for (t) |a| {
            _ = mlx.mlx_array_free(a);
        };
    }
};

fn benchWeights(s: mlx.mlx_stream, shape: BenchShape, seed: u64) !BenchWeights {
    var bw = BenchWeights{};
    errdefer bw.deinit();
    const n: c_int = @intCast(shape.split.tp * shape.split.rowsPerRank());
    const srows: c_int = @intCast(shape.split.tp * shape.split.blocksPerRank());
    for (0..BENCH_COPIES) |c| {
        var rk = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rk);
        try mlx.check(mlx.mlx_random_key(&rk, seed + c));
        var bits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bits);
        const lo = mlx.mlx_array_new_int(0);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_int(126);
        defer _ = mlx.mlx_array_free(hi);
        try mlx.check(mlx.mlx_random_randint(&bits, lo, hi, &[_]c_int{ n, shape.k }, 2, .int32, rk, s));
        try mlx.check(mlx.mlx_astype(&bw.fp8_w[c], bits, .uint8, s));
        const one = mlx.mlx_array_new_float(3e-4);
        defer _ = mlx.mlx_array_free(one);
        try mlx.check(mlx.mlx_full(&bw.fp8_s[c], &[_]c_int{ srows, @divExact(shape.k, 128) }, 2, one, .float32, s));
        var dense: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
        defer for (&dense) |a| {
            _ = mlx.mlx_array_free(a);
        };
        try dequantize(s, bw.fp8_w[c], bw.fp8_s[c], shape.split, &dense);
        for (0..shape.split.outputs()) |p| {
            try mlx.check(mlx.mlx_array_eval(dense[p]));
            try mlx.check(mlx.mlx_transpose(&bw.bf16_t[c][p], dense[p], s));
            var parts = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(parts);
            try mlx.check(mlx.mlx_quantize(&parts, dense[p], mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));
            for (0..3) |i| {
                try mlx.check(mlx.mlx_vector_array_get(&bw.aq[c][p][i], parts, i));
                try mlx.check(mlx.mlx_array_eval(bw.aq[c][p][i]));
            }
        }
        try mlx.check(mlx.mlx_array_eval(bw.fp8_w[c]));
        try mlx.check(mlx.mlx_array_eval(bw.fp8_s[c]));
    }
    return bw;
}

fn benchOp(s: mlx.mlx_stream, arm: BenchArm, shape: BenchShape, bw: *const BenchWeights, c: usize, x: mlx.mlx_array, v: mlx.mlx_vector_array) !void {
    var outs: [3]mlx.mlx_array = .{ .{}, .{}, .{} };
    defer for (&outs) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    switch (arm) {
        .fp8 => try project(s, x, bw.fp8_w[c], bw.fp8_s[c], shape.split, &outs),
        .fp8_staged => {
            const armed = gemv_direct_max_rows;
            defer gemv_direct_max_rows = armed;
            gemv_direct_max_rows = 0;
            try project(s, x, bw.fp8_w[c], bw.fp8_s[c], shape.split, &outs);
        },
        .bf16 => for (0..shape.split.outputs()) |p| {
            outs[p] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_matmul(&outs[p], x, bw.bf16_t[c][p], s));
        },
        .affine8 => for (0..shape.split.outputs()) |p| {
            outs[p] = mlx.mlx_array_new();
            const t = bw.aq[c][p];
            try mlx.check(mlx.mlx_quantized_matmul(&outs[p], x, t[0], t[1], t[2], true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", s));
        },
    }
    for (outs[0..shape.split.outputs()]) |a| _ = mlx.mlx_vector_array_append_value(v, a);
}

fn armBytes(arm: BenchArm, shape: BenchShape) f64 {
    const nk: f64 = @as(f64, @floatFromInt(shape.split.tp * shape.split.rowsPerRank())) * @as(f64, @floatFromInt(shape.k));
    return switch (arm) {
        .bf16 => 2 * nk,
        .fp8, .fp8_staged => nk + @as(f64, @floatFromInt(shape.split.tp * shape.split.blocksPerRank())) * @as(f64, @floatFromInt(shape.k)) / 128.0 * 4.0,
        .affine8 => nk + nk / 64.0 * 4.0,
    };
}

fn benchCell(s: mlx.mlx_stream, shape: BenchShape, bw: *const BenchWeights, m: c_int, arms: []const BenchArm, laps: usize) !void {
    const alloc = testing.allocator;
    var rk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rk);
    try mlx.check(mlx.mlx_random_key(&rk, 7));
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_random_normal(&xf, &[_]c_int{ 1, m, shape.k }, 3, .float32, 0, 1, rk, s));
    try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
    try mlx.check(mlx.mlx_array_eval(x));
    const t = try alloc.alloc(u64, arms.len * laps);
    defer alloc.free(t);
    var clock = BenchClock.init();
    for (0..laps + 3) |lap| {
        for (arms, 0..) |arm, ai| {
            const v = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(v);
            for (0..BENCH_COPIES) |c| try benchOp(s, arm, shape, bw, c, x, v);
            _ = clock.lap();
            try mlx.check(mlx.mlx_eval(v));
            const d = clock.lap();
            if (lap >= 3) t[ai * laps + lap - 3] = d;
        }
    }
    for (arms, 0..) |arm, ai| {
        const us = medianNs(t[ai * laps ..][0..laps]) / BENCH_COPIES / 1000.0;
        const armed = gemv_direct_max_rows;
        if (arm == .fp8_staged) gemv_direct_max_rows = 0;
        const key = gemvKey(.bfloat16, m, shape.k, shape.split);
        gemv_direct_max_rows = armed;
        std.debug.print("[fp8-ubench] {s:<12} M={d:>5} {s:<8} {d:>9.1} us {d:>7.1} GB/s  ({s} NR={d} SGS={d} S={d})\n", .{
            shape.name, m, @tagName(arm), us, armBytes(arm, shape) / (us * 1000.0), @tagName(key.kind), key.nr, key.sgs, key.tiles,
        });
    }
}

test "fp8 block microbench vs bf16 and affine-8 at MiMo's trunk shapes (SUSHI_FP8_UBENCH=1)" {
    const raw = std.c.getenv("SUSHI_FP8_UBENCH") orelse return error.SkipZigTest;
    if (std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0")) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const sweep = std.c.getenv("SUSHI_FP8_UBENCH_SWEEP") != null;
    const only = std.c.getenv("SUSHI_FP8_UBENCH_SHAPE");
    for (BENCH_SHAPES, 0..) |shape, si| {
        if (only) |name| if (!std.mem.eql(u8, std.mem.sliceTo(name, 0), shape.name)) continue;
        var bw = try benchWeights(s, shape, 0xF8 + si * 16);
        defer bw.deinit();
        const arms: []const BenchArm = if (shape.fp8) &.{ .bf16, .fp8, .affine8 } else &.{ .bf16, .affine8 };
        for ([_]c_int{ 1, 2, 4, 8, 16, 512, 2048 }) |m| {
            try benchCell(s, shape, &bw, m, arms, if (m > 16) 5 else 30);
        }
        if (shape.fp8) for ([_]c_int{ 4, 5, 6, 7, 8 }) |m| try benchCell(s, shape, &bw, m, &.{ .fp8, .fp8_staged }, 60);
        if (sweep and shape.fp8) {
            const armed = gemv_direct_max_rows;
            defer {
                gemv_rows_per_sg = 0;
                gemv_sgs = 0;
                gemv_stage_tiles = 0;
                gemv_direct_max_rows = armed;
            }
            // The direct arm's geometry at verify widths: rows per simdgroup and simdgroups per
            // group move no row's arithmetic, only how many rows share each x read.
            for ([_]c_int{ 5, 6, 7, 8 }) |m| for ([_]c_int{ 1, 2, 4, 8 }) |nr| for ([_]c_int{ 2, 4, 8 }) |sg| {
                gemv_direct_max_rows = gemv_max_rows;
                gemv_rows_per_sg = nr;
                gemv_sgs = sg;
                try benchCell(s, shape, &bw, m, &.{.fp8}, 30);
            };
            if (std.mem.eql(u8, std.mem.sliceTo(std.c.getenv("SUSHI_FP8_UBENCH_SWEEP").?, 0), "direct")) continue;
            for ([_]c_int{ 4, 8, 16 }) |m| for ([_]c_int{ 2, 4, 8 }) |nr| for ([_]c_int{ 4, 8 }) |sg| for ([_]c_int{ 1, 2 }) |tiles| {
                gemv_direct_max_rows = 0;
                gemv_rows_per_sg = nr;
                gemv_sgs = sg;
                gemv_stage_tiles = tiles;
                try benchCell(s, shape, &bw, m, &.{.fp8}, 20);
            };
        }
        _ = mlx.mlx_clear_cache();
    }
}
