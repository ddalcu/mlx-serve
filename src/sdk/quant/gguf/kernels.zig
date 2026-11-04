//! Metal kernels over raw ggml blocks. A weight is a uint8 MLX array
//! [rows, row_bytes] holding the GGUF tensor bytes untouched.
const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const device = @import("../../../mlx/device.zig");
const quants = @import("quants.zig");
pub const GgmlType = quants.GgmlType;

pub const Error = error{ UnsupportedType, BadShape, MetalKernelCompileFailed, MetalKernelBadOutputCount };

/// Up to this many activation rows go through the quantized matvec (decode,
/// spec verify, decode batches): one kernel decodes each weight once for all of them.
pub const MATVEC_MAX_ROWS = 8;

/// A block is a scale head plus a 16, 32 or 192 byte payload, and the block
/// sizes (17, 18, 20, 22, 24, 34, 210 bytes) put every payload at an
/// unaligned address: byte loads. These types are stored SPLIT at load: a
/// row holds all its payloads first, then all its heads (same bytes, same
/// row length), so payloads sit at multiples of 16 and come in as aligned
/// 16-byte loads. IQ4_NL 0.044 to 0.037 ms and Q4_0 to MLX parity on the
/// 4B's ffn_up (measured). `payload_at`: the payload's offset in a GGUF block.
pub const Split = struct { head: u32, payload: u32, payload_at: u32 };

pub fn splitLayout(ty: GgmlType) ?Split {
    return switch (ty) {
        .q4_0, .iq4_nl => .{ .head = 2, .payload = 16, .payload_at = 2 },
        .q4_1 => .{ .head = 4, .payload = 16, .payload_at = 4 },
        .q5_0 => .{ .head = 6, .payload = 16, .payload_at = 6 },
        .q5_1 => .{ .head = 8, .payload = 16, .payload_at = 8 },
        .q8_0 => .{ .head = 2, .payload = 32, .payload_at = 2 },
        .mxfp4 => .{ .head = 1, .payload = 16, .payload_at = 1 },
        // ql (128) + qh (64) first, scales (16) + d (2) last.
        .q6_k => .{ .head = 18, .payload = 192, .payload_at = 0 },
        else => null,
    };
}

/// Whether rows of `in` weights of this type are stored split.
pub fn isSplit(ty: GgmlType, in: usize) bool {
    return splitLayout(ty) != null and in % ty.blockElems() == 0;
}

/// Bytes per stored row of `in` weights: split rows are padded so every row
/// starts 16-aligned (the padding is under a block, so the block count still
/// follows from the row length).
pub fn rowBytes(ty: GgmlType, in: usize) usize {
    const raw = in / ty.blockElems() * ty.blockBytes();
    return if (isSplit(ty, in)) (raw + 15) / 16 * 16 else raw;
}

/// Rows of `in` weights from the GGUF (interleaved) layout to the split one, `rowBytes` each.
pub fn packRows(ty: GgmlType, in: usize, dst: []u8, src: []const u8) void {
    const l = splitLayout(ty).?;
    const nb = in / ty.blockElems();
    const bb = ty.blockBytes();
    const row = nb * bb;
    const drow = rowBytes(ty, in);
    const head_at: usize = if (l.payload_at == 0) l.payload else 0;
    var r: usize = 0;
    while (r < src.len / row) : (r += 1) {
        const s = src[r * row ..][0..row];
        const d = dst[r * drow ..][0..drow];
        for (0..nb) |b| {
            @memcpy(d[b * l.payload ..][0..l.payload], s[b * bb + l.payload_at ..][0..l.payload]);
            @memcpy(d[nb * l.payload + b * l.head ..][0..l.head], s[b * bb + head_at ..][0..l.head]);
        }
        @memset(d[row..], 0);
    }
}
/// Up to this many through the tile mat-mat kernel, which decodes weights as it
/// goes. Past it, materializing the weight once and using MLX's GEMM is
/// cheaper: in a live model the crossover sits between ~215 and ~515 prompt
/// tokens (an isolated `zig build bench -Dm=...` puts it lower, near 200).
pub const TILE_MAX_ROWS = 384;

/// The emitters, the decoder of one type and its codebooks: MLX hashes a
/// kernel's whole source on every dispatch (~6 us for the 90 KB of every
/// table and decoder together, measured), so a kernel carries only its own.
pub fn headerFor(comptime ty: GgmlType) [:0]const u8 {
    const parts: []const []const u8 = switch (ty) {
        .q4_0, .q4_1, .q5_0, .q5_1 => &.{"blocks/q4_legacy"},
        .q8_0 => &.{"blocks/q8_0"},
        .mxfp4 => &.{"blocks/mxfp4"},
        .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => &.{"blocks/k_quants"},
        .iq4_nl, .iq4_xs => &.{ "tables/kvalues_iq4nl", "blocks/iq4" },
        .iq2_xxs => &.{ "tables/iq2xxs_grid", "blocks/iq2_xxs" },
        .iq2_xs => &.{ "tables/iq2xs_grid", "blocks/iq2_xs" },
        .iq2_s => &.{ "tables/iq2s_grid", "blocks/iq2_s" },
        .iq3_xxs => &.{ "tables/iq3xxs_grid", "blocks/iq3_xxs" },
        .iq3_s => &.{ "tables/iq3s_grid", "blocks/iq3_s" },
        .iq1_s, .iq1_m => &.{ "tables/iq1s_grid_gpu", "blocks/iq1" },
        else => unreachable,
    };
    comptime var h: [:0]const u8 = @embedFile("metal/blocks/common.h");
    inline for (parts) |p| h = h ++ @embedFile("metal/" ++ p ++ ".h");
    return h;
}

/// The matrix units' tensor ops, for `gather_nax`.
const nax_include = "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n";

pub const Kind = enum { dequant, matvec, matvec_cols, gather, gather_swiglu, matmat, gather_tile, gather_nax, gather_nax_gu };

/// Runs the tile for this threadgroup's token tile: the last tile of a prompt
/// may hold fewer 8-token groups.
const tile_dispatch =
    @embedFile("metal/kernels/tile_dispatch.metal");

/// Weight rows per simdgroup of the matvec, for `per` activation rows per
/// kernel instance. Many short threads pull more memory bandwidth than one
/// long thread per row (MLX's own qmv does 4 rows); 8 keeps the lockstep
/// stripes of the matvec even for every unit count that is a multiple of 4.
/// A thread keeps rows x per running sums, held to ~16 floats so they stay in
/// registers.
pub fn matvecRows(per: c_int) c_int {
    return if (per <= 2) 8 else if (per <= 4) 4 else if (per <= 5) 2 else 1;
}

/// Activation rows one kernel instance takes (the rest of `m` runs as more
/// instances on the grid's z axis): the largest divisor of m up to 8. An
/// instance decodes each weight once for all its rows; past 5 rows it keeps
/// one weight row a simdgroup (`matvecRows`) so its accumulators stay in
/// registers.
pub fn rowsPerInstance(m: c_int) c_int {
    var per: c_int = @min(m, 8);
    while (@mod(m, per) != 0) per -= 1;
    return per;
}
pub const SIMDGROUPS = 2;
pub const SIMD_WIDTH = 32;
/// Simdgroups a decode matvec should put in flight before it starts packing rows.
pub const MIN_SIMDGROUPS = 512;
/// MoE prefill: (token, expert) pairs per expert from which the bank is
/// materialized for gather_mm, and the largest bf16 bank that gets that
/// transient copy (one bank at a time); a larger bank stays on the tile kernel.
const DEQUANT_MOE_PAIRS_PER_EXPERT = 24;
const DEQUANT_MOE_MAX_BYTES: u64 = 2 << 30;

/// Tokens per tile of the matmat kernel.
const TILE_TOKENS = 32;

/// The tile routine for GR 8-token groups, a macro so every loop bound is a
/// compile-time constant: with a runtime bound the accumulators spill and the
/// kernel runs 4x slower (measured). ~18 live simdgroup matrices is the
/// register budget, so no hoisting the wm loads and no 64-token tiles either.
/// GG_W = the weight matrix, GG_STORE(t, g) = where 8x8 block (t, g) of the
/// result goes, [GG_T0, GG_T1) the 8-token groups to multiply (the rest
/// keep their zeros; the loop bound stays GR so the accumulators stay in
/// registers).
const tile_macro = blk: {
    const body =
        @embedFile("metal/kernels/tile_macro.metal");
    @setEvalBranchQuota(100_000);
    var buf: [body.len * 2]u8 = undefined;
    const n = std.mem.replace(u8, body, "\n", " \\\n", &buf);
    break :blk buf[0 .. body.len + 2 * n].* ++ "\n".*;
};

comptime {
    std.debug.assert(TILE_TOKENS == 32); // the kernel instantiates GG_TILE for 4, 1, 2 and 3 groups
}

/// Weight rows per tile of the matmat kernel (one decoding thread each): as
/// many as fit the 32 KiB of threadgroup memory Apple GPUs give, TILE_ROWS x
/// UNIT floats, up to the 32 threads of a simdgroup.
fn tileRows(ty: GgmlType) c_int {
    return @intCast(@min(SIMD_WIDTH, (32 << 10) / (ty.unitElems() * @sizeOf(f32)) / 8 * 8));
}

/// A threadgroup is SIMDGROUPS simdgroups. Each simdgroup owns RPT rows.
/// Its 32 threads run in LOCKSTEP, so a loop costs every thread the
/// longest trip count: striping each row on its own wastes lanes whenever
/// 32 doesn't divide the units of a row (48 units = 75%, Q6_K at K = 2560
/// is 20 units = 62%; rotating the stripes per row changes nothing,
/// measured). So the RPT rows are striped as ONE run of RPT * units items,
/// even whenever RPT * units / 32 is whole. simd_sum adds the lanes up.
/// GG_W = first weight row of the instance, GG_X = its first activation row.
const matvec_body =
    @embedFile("metal/kernels/matvec_body.metal");

/// The matvec for 32-weight units, lane-major over a row's units instead: a lane loads its unit's activations into
/// registers once and decodes RPT weight rows against them (ggml's layout), where the striped body reads them again
/// per row. Needs a row of at least 32 units and few activation rows (32 x M floats of registers).
const matvec_cols_body =
    @embedFile("metal/kernels/matvec_cols.metal");

fn colsServes(ty: GgmlType, in: c_int, per: c_int) bool {
    return ty.unitElems() == 32 and per <= 2 and @divExact(in, 32) >= SIMD_WIDTH;
}

/// An IQ type's codebook, read at data-dependent indices: from `constant` memory those reads serialize, so the
/// matvec and tile kinds stage it in threadgroup memory first (as ggml's own Metal kernels do).
fn gridOf(ty: GgmlType) ?struct { table: []const u8, elem: []const u8, len: u32, halves: bool = false } {
    // IQ1's 8 KB grid stays in `constant`: staged, it costs the tile kernels more occupancy than its reads cost. The
    // IQ2 grids of 8 magnitudes an entry stage as two half4 each (exact): a lookup skips the byte unpack.
    return switch (ty) {
        .iq2_xxs => .{ .table = "iq2xxs_grid", .elem = "uint64_t", .len = 256, .halves = true },
        .iq2_xs => .{ .table = "iq2xs_grid", .elem = "uint64_t", .len = 512, .halves = true },
        .iq2_s => .{ .table = "iq2s_grid", .elem = "uint64_t", .len = 1024 },
        .iq3_xxs => .{ .table = "iq3xxs_grid", .elem = "uint32_t", .len = 256 },
        .iq3_s => .{ .table = "iq3s_grid", .elem = "uint32_t", .len = 512 },
        else => null,
    };
}

/// The matvec kinds' codebook in threadgroup memory, `DEQ_AT` reading it there.
fn stagedGrid(comptime ty: GgmlType) []const u8 {
    const g = comptime gridOf(ty) orelse return "";
    if (g.halves) return std.fmt.comptimePrint(@embedFile("metal/kernels/staged_grid_half.fmt.metal"), .{ 2 * g.len, g.len, g.table, @tagName(ty) });
    return std.fmt.comptimePrint(
        \\threadgroup {s} gg_grid[{d}];
        \\for (uint i = simdgroup_index_in_threadgroup * threads_per_simdgroup + thread_index_in_simdgroup; i < {d}; i += simdgroups_per_threadgroup * threads_per_simdgroup) gg_grid[i] = {s}[i];
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\#undef DEQ_AT
        \\#define DEQ_AT(w, row, blk, u, o) deq_{s}((w) + (size_t(row) * NB + (blk)) * BB, u, o, gg_grid)
        \\
    , .{ g.elem, g.len, g.len, g.table, @tagName(ty) });
}

/// The block constants of `ty` (BE, BB, UNIT, UPB, NB of the template K) and DEQ_AT(w, row, blk, u, o): decode unit
/// u of block blk of row `row` of weight w. SPLIT (template const, see isSplit) picks the row layout. `slice`: a
/// kernel built on 32-weight units, where a wider K-quant unit decodes a 32-weight slice at a time (`deq_<type>_s`).
fn prelude(comptime ty: GgmlType, comptime slice: bool) [:0]const u8 {
    const sliced = slice and ty.unitElems() > 32;
    const name = @tagName(ty) ++ if (sliced) "_s" else "";
    const layout = splitLayout(ty) orelse Split{ .head = 0, .payload = 0, .payload_at = 0 };
    const deq_at = if (layout.head > 0)
        std.fmt.comptimePrint(@embedFile("metal/kernels/deq_at_split.fmt.metal"), .{ name, layout.payload, layout.head, layout.payload, name })
    else
        std.fmt.comptimePrint(@embedFile("metal/kernels/deq_at.fmt.metal"), .{name});
    return std.fmt.comptimePrint(@embedFile("metal/kernels/prelude.fmt.metal"), .{ ty.blockElems(), ty.blockBytes(), if (sliced) 32 else ty.unitElems(), tileRows(ty), TILE_TOKENS }) ++ deq_at;
}

/// The kinds built on 32-weight units (the decode gather keeps a wide unit's own matvec, the proof tapes' order).
fn unit32(comptime kind: Kind) bool {
    return kind == .gather_nax or kind == .gather_nax_gu;
}

pub fn source(comptime kind: Kind, comptime ty: GgmlType) [:0]const u8 {
    return prelude(ty, unit32(kind)) ++ switch (kind) {
        // One thread per unit.
        .dequant => @embedFile("metal/kernels/dequant.metal"),
        // Instance z takes activation rows [z * M, z * M + M).
        .matvec => "#define GG_W w\n#define GG_X (x + size_t(z) * M * K)\n" ++ stagedGrid(ty) ++ matvec_body,
        .matvec_cols => "#define GG_W w\n#define GG_X (x + size_t(z) * M * K)\n" ++ stagedGrid(ty) ++ matvec_cols_body,
        // MoE: instance z is one (token, expert) pair, `ids[z]` its expert in
        // the bank w [experts, N, row bytes] and z / XDIV its activation row
        // (XDIV = experts per token, 1 when the rows come already repeated).
        .gather => "#define GG_W (w + size_t(ids[z]) * N * ROWB)\n#define GG_X (x + size_t(z / XDIV) * K)\n" ++ stagedGrid(ty) ++ (if (ty.unitElems() == 32) "if ((K / UNIT) % 32 == 0) {\n" ++ @embedFile("metal/kernels/gather_body.metal") ++ "} else {\n" ++ matvec_body ++ "}\n" else matvec_body),
        // The gather's matvec over a gate and an up bank at once, then SwiGLU on the pair (`gatherSwiGLU`).
        .gather_swiglu => stagedGrid(ty) ++ @embedFile("metal/kernels/gather_swiglu.metal"),
        // x [M, K] f32 times w^T, M a multiple of 8 and N of TILE_ROWS. One
        // threadgroup (= one simdgroup, whose threads run simdgroup_matrix ops
        // TOGETHER) per output tile of TILE_TOKENS tokens x TILE_ROWS rows.
        // Thread t decodes row t of the tile, a unit at a time, into shared
        // threadgroup memory (thread-local arrays spill, this doesn't), then the
        // group multiplies 8x8 blocks against x. The last tile of a prompt may
        // hold fewer 8-token groups.
        .matmat => "threadgroup float tile[TILE_ROWS * UNIT];\n" ++ stagedGrid(ty) ++ tile_macro ++
            @embedFile("metal/kernels/matmat.metal") ++ tile_dispatch,
        // MoE prefill: x [M, K] f32 holds the rows already gathered per
        // (token, expert) pair, `ids` [rows of x, unpadded] their experts, as
        // the sorted gather_qmm layout. A token tile runs the tile routine once
        // per distinct expert in it, over just the 8-token groups that hold
        // its rows, into a staging buffer, and the rows of that expert are
        // copied out. Every real row belongs to exactly one pass, the zero
        // pad rows to none.
        .gather_tile => "threadgroup float tile[TILE_ROWS * UNIT];\nthreadgroup float stage[TILE_TOKENS * TILE_ROWS];\n" ++ stagedGrid(ty) ++ tile_macro ++
            @embedFile("metal/kernels/gather_tile_head.metal") ++ tile_dispatch ++
            @embedFile("metal/kernels/gather_tile_tail.metal"),
        // MoE prefill on the matrix units (`gatherNax`).
        .gather_nax => "threadgroup float buf[128 * (2 * UNIT + 8) / 2];\n" ++ stagedGrid(ty) ++
            @embedFile("metal/kernels/gather_nax.metal"),
        // The same over a gate and an up bank, the clipped SwiGLU on the pair (`gatherNaxSwiGLU`).
        .gather_nax_gu => "threadgroup float buf[128 * (2 * UNIT + 8) / 2];\n" ++ stagedGrid(ty) ++
            @embedFile("metal/kernels/gather_nax_gu.metal"),
    };
}

/// A kind's input names, in its wrapper's order.
pub fn inputNames(comptime kind: Kind) []const [*:0]const u8 {
    return switch (kind) {
        .dequant => &.{"w"},
        .matvec, .matvec_cols, .matmat => &.{ "x", "w" },
        .gather, .gather_tile => &.{ "x", "w", "ids" },
        .gather_nax => &.{ "x", "w", "offs", "tiles" },
        .gather_nax_gu => &.{ "x", "wg", "wu", "offs", "tiles", "wts", "lim" },
        .gather_swiglu => &.{ "x", "wg", "wu", "ids", "sigtab" },
    };
}

/// A kind's header: the type's decoders, after the matrix units' tensor ops for the NAX gathers.
pub fn kernelHeader(comptime kind: Kind, comptime ty: GgmlType) [:0]const u8 {
    return comptime if (kind == .gather_nax or kind == .gather_nax_gu) nax_include ++ headerFor(ty) else headerFor(ty);
}

/// One compiled kernel per (kind, type), built on first use. Only the
/// inference thread calls MLX, so no locking.
fn kernelFor(comptime kind: Kind, comptime ty: GgmlType) Error!mlx.mlx_fast_metal_kernel {
    const Cache = struct {
        // Zig only makes this a separate static per instantiation if it captures the params.
        const key = .{ kind, ty };
        var kernel: ?mlx.mlx_fast_metal_kernel = null;
    };
    _ = Cache.key;
    if (Cache.kernel) |k| return k;
    const input_names = inputNames(kind);
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(
        "gguf_" ++ @tagName(kind) ++ "_" ++ @tagName(ty),
        in_vec,
        out_vec,
        comptime source(kind, ty),
        comptime kernelHeader(kind, ty),
        true,
        false,
    );
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    Cache.kernel = k;
    return k;
}

/// A plugin's own kernel over weights of `ty`: `body` runs after the type's prelude, DEQ_AT and staged grid, `header`
/// after its decoders; its template ints add the K, ROWB and SPLIT of `layout`. Built once per (name, ty).
pub fn pluginKernel(comptime name: [:0]const u8, comptime ins: []const [*:0]const u8, comptime body: [:0]const u8, comptime header: [:0]const u8, ty: GgmlType) Error!mlx.mlx_fast_metal_kernel {
    inline for (all_types) |t| if (t == ty) return pluginKernelFor(name, ins, body, header, t);
    return error.UnsupportedType;
}

/// `pluginKernel`'s source and header for `ty`.
pub fn pluginSource(comptime ty: GgmlType, comptime body: [:0]const u8) [:0]const u8 {
    return comptime prelude(ty, true) ++ stagedGrid(ty) ++ body;
}

pub fn pluginHeader(comptime ty: GgmlType, comptime header: [:0]const u8) [:0]const u8 {
    return comptime headerFor(ty) ++ header;
}

fn pluginKernelFor(comptime name: [:0]const u8, comptime ins: []const [*:0]const u8, comptime body: [:0]const u8, comptime header: [:0]const u8, comptime ty: GgmlType) Error!mlx.mlx_fast_metal_kernel {
    const Cache = struct {
        const key = .{ name, ty };
        var kernel: ?mlx.mlx_fast_metal_kernel = null;
    };
    _ = Cache.key;
    if (Cache.kernel) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const output_names = [_][*:0]const u8{"out"};
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new(name ++ "_" ++ @tagName(ty), in_vec, out_vec, pluginSource(ty, body), pluginHeader(ty, header), true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    Cache.kernel = k;
    return k;
}

/// Every type with a decoder in metal/blocks.h.
pub const all_types = [_]GgmlType{ .q4_0, .q4_1, .q5_0, .q5_1, .q8_0, .mxfp4, .q2_k, .q3_k, .q4_k, .q5_k, .q6_k, .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_nl, .iq4_xs, .iq1_s, .iq1_m };

fn kernel(kind: Kind, ty: GgmlType) Error!mlx.mlx_fast_metal_kernel {
    inline for (all_types) |t| {
        if (t == ty) switch (kind) {
            inline else => |k| return kernelFor(k, t),
        };
    }
    return error.UnsupportedType;
}

pub const Const = struct { [*:0]const u8, c_int };

fn apply(
    k: mlx.mlx_fast_metal_kernel,
    inputs: []const mlx.mlx_array,
    out_shape: []const c_int,
    dtype: mlx.mlx_dtype,
    /// Total threads and threadgroup size, per axis.
    grid: [3]c_int,
    group: [3]c_int,
    /// Compile-time ints of the source (K = in_features, N = rows, ...):
    /// constant loop bounds, and no per-dispatch shape buffers (what `x_shape`
    /// in the source would cost).
    consts: []const Const,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, out_shape.ptr, out_shape.len, dtype));
    for (consts) |c| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, c[0], c[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, grid[0], grid[1], grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, group[0], group[1], group[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", dtype));

    const in_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, k, in_vec, config, s));
    if (mlx.mlx_vector_array_size(out_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, out_vec, 0));
    return out;
}

/// Weight geometry from its byte shape: rows and in_features.
fn geometry(ty: GgmlType, w: mlx.mlx_array) Error!struct { rows: c_int, in: c_int, row_bytes: c_int } {
    if (!ty.isQuantized()) return error.UnsupportedType;
    const ws = mlx.getShape(w);
    if (ws.len != 2 or mlx.mlx_array_dtype(w) != .uint8) return error.BadShape;
    const in = try inFeatures(ty, ws[1]);
    return .{ .rows = ws[0], .in = in, .row_bytes = ws[1] };
}

/// Weights per row from the stored row length (see rowBytes).
fn inFeatures(ty: GgmlType, row_bytes: c_int) Error!c_int {
    const bb: c_int = @intCast(ty.blockBytes());
    const in = @divFloor(row_bytes, bb) * @as(c_int, @intCast(ty.blockElems()));
    if (in == 0 or rowBytes(ty, @intCast(in)) != row_bytes) return error.BadShape;
    return in;
}

/// The layout constants every kernel takes.
pub fn layoutConsts(ty: GgmlType, in: c_int, row_bytes: c_int) [3]Const {
    return .{ .{ "K", in }, .{ "ROWB", row_bytes }, .{ "SPLIT", @intFromBool(isSplit(ty, @intCast(in))) } };
}

/// y = x w[ids]^T over an expert bank w [experts, rows, row_bytes], the drop-in
/// for `mlx_gather_qmm`: x [..., 1, in_features], `rhs_idx` picks the expert
/// of every output, `lhs_idx` (optional, same size) its row of x. Without it
/// x's rows have to lead rhs_idx's dims (decode: x [B, S, 1, 1, in] against
/// [B, S, top_k] experts). Returns rhs_idx's shape + [1, rows].
/// Decode runs a matvec per (token, expert) pair; past MATVEC_MAX_ROWS pairs
/// with x's rows already gathered (prefill), the tile kernel decodes each
/// expert's weights once per token tile instead of once per pair.
pub fn gatherLinear(info: Info, x: mlx.mlx_array, w: mlx.mlx_array, lhs_idx: mlx.mlx_array, rhs_idx: mlx.mlx_array, sorted: bool, s: mlx.mlx_stream) !mlx.mlx_array {
    const ty = info.ty;
    const ws = mlx.getShape(w);
    if (!ty.isQuantized() or info.tiled != null) return error.UnsupportedType;
    const p = try pairsOf(ty, x, w, rhs_idx);
    const rows = p.rows;
    const in = p.in;
    const x_rows = p.x_rows;
    const os = mlx.getShape(rhs_idx);
    const n_out = p.n_out;
    const out_shape = p.out_shape;

    var x_in = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_in);
    var xdiv: c_int = 1;
    if (lhs_idx.ctx != null) {
        if (mlx.mlx_array_size(lhs_idx) != mlx.mlx_array_size(rhs_idx)) return error.BadShape;
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ x_rows, in }, 2, s));
        var lhs_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lhs_flat);
        try mlx.check(mlx.mlx_reshape(&lhs_flat, lhs_idx, &[_]c_int{n_out}, 1, s));
        try mlx.check(mlx.mlx_take_axis(&x_in, flat, lhs_flat, 0, s));
    } else {
        // Row r of x serves outputs [r * xdiv, (r + 1) * xdiv): x's batch dims
        // lead rhs_idx's, however either side is reshaped ([B, S, top_k] or [B * S, top_k]).
        if (@mod(n_out, x_rows) != 0) return error.BadShape;
        xdiv = @divExact(n_out, x_rows);
        try mlx.check(mlx.mlx_array_set(&x_in, x));
    }
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, rhs_idx, .uint32, s));

    // Long prefills: materialize the bank once and let MLX's gather_mm do the
    // GEMMs. The decode costs one pass over the bank, the tile kernel one per
    // (expert, token tile); past ~16 pairs per expert the pass is cheaper
    // (bench: 3.6x at 1700 tokens x 8 experts of 256), as long as the bf16
    // bank stays under a GB. Past that cap the matrix units' gather.
    const bank_bytes = @as(u64, @intCast(ws[0])) * @as(u64, @intCast(rows)) * @as(u64, @intCast(in)) * 2;
    const dequant_bank = n_out >= DEQUANT_MOE_PAIRS_PER_EXPERT * ws[0] and bank_bytes <= DEQUANT_MOE_MAX_BYTES;
    if (xdiv == 1 and dequant_bank) {
        var flat_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_w);
        try mlx.check(mlx.mlx_reshape(&flat_w, w, &[_]c_int{ ws[0] * rows, ws[2] }, 2, s));
        const wd = try dequantize(ty, flat_w, mlx.mlx_array_dtype(x), s);
        defer _ = mlx.mlx_array_free(wd);
        var bank = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bank);
        try mlx.check(mlx.mlx_reshape(&bank, wd, &[_]c_int{ ws[0], rows, in }, 3, s));
        var bank_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bank_t);
        try mlx.check(mlx.mlx_transpose_axes(&bank_t, bank, &[_]c_int{ 0, 2, 1 }, 3, s));
        var x3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x3);
        try mlx.check(mlx.mlx_reshape(&x3, x_in, &[_]c_int{ n_out, 1, in }, 3, s));
        var ids_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_flat);
        try mlx.check(mlx.mlx_reshape(&ids_flat, ids, &[_]c_int{n_out}, 1, s));
        var y3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y3);
        try mlx.check(mlx.mlx_gather_mm(&y3, x3, bank_t, .{ .ctx = null }, ids_flat, sorted, s));
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_reshape(&y, y3, &out_shape, os.len + 2, s));
        return y;
    }

    if (sorted and xdiv == 1 and n_out >= @max(64, NAX_PAIRS_PER_EXPERT * ws[0]) and naxGathers(rows, in, mlx.mlx_array_dtype(x))) {
        const y2 = try gatherNax(ty, x_in, w, ids, rows, in, ws[2], s);
        defer _ = mlx.mlx_array_free(y2);
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_reshape(&y, y2, &out_shape, os.len + 2, s));
        return y;
    }

    // The tile kernel pays one weight decode per (expert, token tile) but runs
    // few threadgroups; it wins once an expert averages a whole 8-token group
    // per tile (256 experts: ~256 prompt tokens, measured, see bench).
    if (xdiv == 1 and n_out >= 8 * ws[0] and @mod(rows, tileRows(ty)) == 0) {
        var x2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x2);
        try mlx.check(mlx.mlx_reshape(&x2, x_in, &[_]c_int{ n_out, in }, 2, s));
        var ids_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_flat);
        try mlx.check(mlx.mlx_reshape(&ids_flat, ids, &[_]c_int{n_out}, 1, s));
        const y2 = try tileGlue(.gather_tile, ty, x2, &.{ w, ids_flat }, rows, s);
        defer _ = mlx.mlx_array_free(y2);
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_reshape(&y, y2, &out_shape, os.len + 2, s));
        return y;
    }

    const groups = @divFloor(rows + GATHER_RPT * GATHER_SIMDGROUPS - 1, GATHER_RPT * GATHER_SIMDGROUPS);
    return apply(try kernel(.gather, ty), &.{ x_in, w, ids }, out_shape[0 .. os.len + 2], mlx.mlx_array_dtype(x), .{ SIMD_WIDTH, GATHER_SIMDGROUPS * groups, n_out }, .{ SIMD_WIDTH, GATHER_SIMDGROUPS, 1 }, &(layoutConsts(ty, in, ws[2]) ++ [_]Const{ .{ "N", rows }, .{ "M", 1 }, .{ "RPT", GATHER_RPT }, .{ "XDIV", xdiv } }), s);
}

/// Pairs per expert from which `gatherNax` beats the per-pair matvec: it reads each expert once per 64-row tile.
const NAX_PAIRS_PER_EXPERT = 4;

/// Whether `gatherNax` serves `ty` at `rows` outputs per expert and `in` inputs of `dt`: a GPU with the matrix
/// units, whole 128-output tiles, 64-weight K steps, 16-bit activations.
fn naxGathers(rows: c_int, in: c_int, dt: mlx.mlx_dtype) bool {
    return device.naxAvailable() and @mod(rows, 128) == 0 and @mod(in, 64) == 0 and (dt == .bfloat16 or dt == .float16);
}

/// x_in [n >= 64, (1,) in] each (token, expert) pair's row and ids [n] its expert, sorted, through the bank w on the
/// matrix units: each expert's run is cut into 64-row tiles (`offs`, `tiles`: the runs' and the tiles' starts).
/// [n, rows] in x_in's dtype.
fn gatherNax(ty: GgmlType, x_in: mlx.mlx_array, w: mlx.mlx_array, ids: mlx.mlx_array, rows: c_int, in: c_int, row_bytes: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const n: c_int = @intCast(mlx.mlx_array_size(ids));
    const e = mlx.getShape(w)[0];
    var runs: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (runs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try runStarts(ids, e, 64, &runs, s);
    const groups = @divFloor(n + 63, 64) + e;
    const x2 = try pairRows(x_in, n, in, s);
    defer _ = mlx.mlx_array_free(x2);
    return apply(try kernel(.gather_nax, ty), &.{ x2, w, runs[0], runs[1] }, &.{ n, rows }, mlx.mlx_array_dtype(x_in), .{ rows, groups, 1 }, .{ 128, 1, 1 }, &(layoutConsts(ty, in, row_bytes) ++ [_]Const{ .{ "N", rows }, .{ "E", e } }), s);
}

/// x as [n, in]: the NAX gathers read x_shape[0] as the pair count, whatever batch dims the caller's x carries.
fn pairRows(x: mlx.mlx_array, n: c_int, in: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_reshape(&r, x, &[_]c_int{ n, in }, 2, s));
    return r;
}

/// Over sorted expert ids [n] of a bank of e experts: `out[0]` [e + 1] each expert's first row, `out[1]` [e + 1] its
/// first `per`-row piece (int32).
fn runStarts(ids: mlx.mlx_array, e: c_int, per: c_int, out: *[2]mlx.mlx_array, s: mlx.mlx_stream) !void {
    const n: c_int = @intCast(mlx.mlx_array_size(ids));
    var arrays: [11]mlx.mlx_array = @splat(.{ .ctx = null });
    defer for (arrays) |a| {
        if (a.ctx != null) _ = mlx.mlx_array_free(a);
    };
    const col = &arrays[0];
    const experts = &arrays[1];
    const hit = &arrays[2];
    const hits = &arrays[3];
    const counts = &arrays[4];
    const runs = &arrays[5];
    const zero = &arrays[6];
    const offs = &arrays[7];
    const up = &arrays[8];
    const pieces = &arrays[9];
    const tiles = &arrays[10];
    try mlx.check(mlx.mlx_reshape(col, ids, &[_]c_int{ n, 1 }, 2, s));
    try mlx.check(mlx.mlx_arange(experts, 0, @floatFromInt(e), 1, .uint32, s));
    try mlx.check(mlx.mlx_equal(hit, col.*, experts.*, s));
    try mlx.check(mlx.mlx_sum_axis(hits, hit.*, 0, false, s));
    try mlx.check(mlx.mlx_astype(counts, hits.*, .int32, s));
    try mlx.check(mlx.mlx_cumsum(runs, counts.*, 0, false, true, s));
    try mlx.check(mlx.mlx_zeros(zero, &[_]c_int{1}, 1, .int32, s));
    try concat(offs, &.{ zero.*, runs.* }, s);
    // Pieces per run: (count + per - 1) / per.
    const round_up = mlx.mlx_array_new_int(per - 1);
    defer _ = mlx.mlx_array_free(round_up);
    const per_a = mlx.mlx_array_new_int(per);
    defer _ = mlx.mlx_array_free(per_a);
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_add(&sum, counts.*, round_up, s));
    try mlx.check(mlx.mlx_floor_divide(up, sum, per_a, s));
    try mlx.check(mlx.mlx_cumsum(pieces, up.*, 0, false, true, s));
    try concat(tiles, &.{ zero.*, pieces.* }, s);
    try mlx.check(mlx.mlx_array_set(&out[0], offs.*));
    try mlx.check(mlx.mlx_array_set(&out[1], tiles.*));
}

fn concat(out: *mlx.mlx_array, parts: []const mlx.mlx_array, s: mlx.mlx_stream) !void {
    const v = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_concatenate_axis(out, v, 0, s));
}

/// An expert bank's rows are scattered reads: four a simdgroup beat eight (swept in situ on IQ banks), eight
/// simdgroups a threadgroup so its staged IQ grid serves 32 rows (two left the grid copy larger than the rows' bytes).
pub const GATHER_RPT = 4;
pub const GATHER_SIMDGROUPS = 8;

/// A gather's (token, expert) pairs over the bank w [experts, rows, row_bytes]: x [..., 1, in] and rhs_idx's shape
/// + [1, rows] out.
const Pairs = struct { rows: c_int, in: c_int, x_rows: c_int, n_out: c_int, out_shape: [8]c_int };

fn pairsOf(ty: GgmlType, x: mlx.mlx_array, w: mlx.mlx_array, rhs_idx: mlx.mlx_array) !Pairs {
    const ws = mlx.getShape(w);
    if (ws.len != 3 or mlx.mlx_array_dtype(w) != .uint8) return error.BadShape;
    const in = try inFeatures(ty, ws[2]);
    const xs = mlx.getShape(x);
    if (xs.len < 2 or xs[xs.len - 1] != in or xs[xs.len - 2] != 1) return error.BadShape;
    const os = mlx.getShape(rhs_idx);
    var p: Pairs = .{ .rows = ws[1], .in = in, .x_rows = @intCast(@divExact(mlx.mlx_array_size(x), @as(usize, @intCast(in)))), .n_out = @intCast(mlx.mlx_array_size(rhs_idx)), .out_shape = undefined };
    if (os.len + 2 > p.out_shape.len or p.n_out == 0) return error.BadShape;
    @memcpy(p.out_shape[0..os.len], os);
    p.out_shape[os.len] = 1;
    p.out_shape[os.len + 1] = p.rows;
    return p;
}

/// silu(x w_gate[ids]^T) * (x w_up[ids]^T) in one pass over both banks: `gatherLinear`'s matvec per (token, expert)
/// pair for each projection, rounded to x's dtype as it writes them, then MLX's SwiGLU chain, bit for bit (`sigtab`:
/// MLX's sigmoid of every 16-bit pattern of x's dtype). x's rows lead rhs_idx's dims, as in `gatherLinear` without
/// lhs indices. Null where `gatherLinear` would not run its matvec (prefill-sized pair counts), or the banks differ.
pub fn gatherSwiGLU(gate: Info, up: Info, x: mlx.mlx_array, wg: mlx.mlx_array, wu: mlx.mlx_array, rhs_idx: mlx.mlx_array, sigtab: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    const ty = gate.ty;
    if (up.ty != ty or !ty.isQuantized() or gate.tiled != null or up.tiled != null) return null;
    const dt = mlx.mlx_array_dtype(x);
    if (dt != .bfloat16 and dt != .float16) return null;
    const ws = mlx.getShape(wg);
    if (!std.mem.eql(c_int, ws, mlx.getShape(wu))) return null;
    const p = try pairsOf(ty, x, wg, rhs_idx);
    if (@mod(p.n_out, p.x_rows) != 0) return error.BadShape;
    const xdiv = @divExact(p.n_out, p.x_rows);
    if (xdiv == 1 and p.n_out >= 8 * ws[0]) return null;
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_astype(&ids, rhs_idx, .uint32, s));
    const groups = @divFloor(p.rows + GATHER_RPT * SIMDGROUPS - 1, GATHER_RPT * SIMDGROUPS);
    const nd = mlx.getShape(rhs_idx).len + 2;
    return try apply(try kernel(.gather_swiglu, ty), &.{ x, wg, wu, ids, sigtab }, p.out_shape[0..nd], dt, .{ SIMD_WIDTH, SIMDGROUPS * groups, p.n_out }, .{ SIMD_WIDTH, SIMDGROUPS, 1 }, &(layoutConsts(ty, p.in, ws[2]) ++ [_]Const{ .{ "N", p.rows }, .{ "RPT", GATHER_RPT }, .{ "XDIV", xdiv } }), s);
}

/// A sorted MoE call's gate and up on the matrix units with the clipped SwiGLU fused: x [n, (1,) in] each pair's row
/// (sorted by expert, `ids` [n]), banks `wg`, `wu` of one type, `wts` [n] f32 each pair's weight, `lim` [1] f32 the clip
/// (0: none). [n, rows] in x's dtype; null where `gatherLinear` would not take the matrix units.
pub fn gatherNaxSwiGLU(gate: Info, up: Info, x: mlx.mlx_array, wg: mlx.mlx_array, wu: mlx.mlx_array, ids: mlx.mlx_array, wts: mlx.mlx_array, lim: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    const ty = gate.ty;
    if (up.ty != ty or gate.tiled != null or up.tiled != null or !std.mem.eql(c_int, mlx.getShape(wg), mlx.getShape(wu))) return null;
    const ws = mlx.getShape(wg);
    if (ws.len != 3) return error.BadShape;
    const rows = ws[1];
    const n: c_int = @intCast(mlx.mlx_array_size(ids));
    const in: c_int = @intCast(mlx.getShape(x)[mlx.getShape(x).len - 1]);
    if (@as(c_int, @intCast(mlx.mlx_array_size(x))) != n * in or @mod(in, @as(c_int, @intCast(ty.blockElems()))) != 0 or rowBytes(ty, @intCast(in)) != ws[2]) return error.BadShape;
    if (n < @max(64, NAX_PAIRS_PER_EXPERT * ws[0])) return null;
    if (!naxGathers(rows, in, mlx.mlx_array_dtype(x))) return null;
    var runs: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (runs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var ids32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids32);
    try mlx.check(mlx.mlx_astype(&ids32, ids, .uint32, s));
    try runStarts(ids32, ws[0], 64, &runs, s);
    const groups = @divFloor(n + 63, 64) + ws[0];
    const x2 = try pairRows(x, n, in, s);
    defer _ = mlx.mlx_array_free(x2);
    return try apply(try kernel(.gather_nax_gu, ty), &.{ x2, wg, wu, runs[0], runs[1], wts, lim }, &.{ n, rows }, mlx.mlx_array_dtype(x), .{ rows * 2, groups, 1 }, .{ 128, 1, 1 }, &(layoutConsts(ty, in, ws[2]) ++ [_]Const{ .{ "N", rows }, .{ "E", ws[0] } }), s);
}

/// w [rows, row_bytes] -> [rows, in_features] in `dtype`.
pub fn dequantize(ty: GgmlType, w: mlx.mlx_array, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const g = try geometry(ty, w);
    const n_units = g.rows * @divExact(g.in, @as(c_int, @intCast(ty.unitElems())));
    return apply(try kernel(.dequant, ty), &.{w}, &.{ g.rows, g.in }, dtype, .{ n_units, 1, 1 }, .{ @min(n_units, 64), 1, 1 }, &layoutConsts(ty, g.in, g.row_bytes), s);
}

/// x [..., in_features] times w^T -> [..., rows], straight off the blocks.
pub fn matvec(ty: GgmlType, x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const g = try geometry(ty, w);
    const xs = mlx.getShape(x);
    var out_shape: [8]c_int = undefined;
    if (xs.len == 0 or xs.len > out_shape.len or xs[xs.len - 1] != g.in) return error.BadShape;
    @memcpy(out_shape[0 .. xs.len - 1], xs[0 .. xs.len - 1]);
    out_shape[xs.len - 1] = g.rows;
    const m: c_int = @intCast(@divExact(mlx.mlx_array_size(x), @as(usize, @intCast(g.in))));
    const per = rowsPerInstance(m);
    const cols = colsServes(ty, g.in, per);
    const rpt = matvecRpt(cols, per, g.rows);
    const groups = @divFloor(g.rows + rpt * SIMDGROUPS - 1, rpt * SIMDGROUPS);
    const k = try kernel(if (cols) .matvec_cols else .matvec, ty);
    return apply(k, &.{ x, w }, out_shape[0..xs.len], mlx.mlx_array_dtype(x), .{ SIMD_WIDTH, SIMDGROUPS * groups, @divExact(m, per) }, .{ SIMD_WIDTH, SIMDGROUPS, 1 }, &(layoutConsts(ty, g.in, g.row_bytes) ++ [_]Const{ .{ "N", g.rows }, .{ "M", per }, .{ "RPT", rpt } }), s);
}

/// Weight rows per simdgroup of the matvec (`cols`: the column kernel) for `per` activation rows into `n` outputs;
/// the wrapper and the kernel contract both read it. Small weights are latency bound, so fewer rows a simdgroup put
/// more of them in flight; the column kernel holds a unit's activations in registers and keeps the most at two rows.
pub fn matvecRpt(cols: bool, per: c_int, n: c_int) c_int {
    var rpt = if (cols) 2 else matvecRows(per);
    while (rpt > 1 and @divFloor(n, rpt) < MIN_SIMDGROUPS) rpt = @divExact(rpt, 2);
    return rpt;
}

/// x [M, in_features] times w^T -> [M, rows], rows a multiple of tileRows.
/// Runs in f32: simdgroup matrices need one element type, f16 buys no speed
/// here (measured) and costs ~1e-3 relative accuracy. Three dispatches: a
/// prefill runs ~250 of these back to back, so the cast + zero-pad to whole
/// groups of 8 tokens is one kernel and the un-pad + cast back another,
/// not the 6 stock ops they would take.
pub fn matmat(ty: GgmlType, x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const g = try geometry(ty, w);
    const xs = mlx.getShape(x);
    if (xs.len != 2 or xs[1] != g.in or @mod(g.rows, tileRows(ty)) != 0) return error.BadShape;
    return tileGlue(.matmat, ty, x, &.{w}, g.rows, s);
}

/// x [M, K] through a tile kernel whose inputs after x are `rest`, rows a
/// multiple of tileRows. The glue around the kernel, see `matmat`.
fn tileGlue(comptime kind: Kind, ty: GgmlType, x: mlx.mlx_array, rest: []const mlx.mlx_array, rows: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const wshape = mlx.getShape(rest[0]);
    const row_bytes = wshape[wshape.len - 1];
    const xs = mlx.getShape(x);
    const m = xs[0];
    const k = xs[1];
    const m_pad = @divFloor(m + 7, 8) * 8;
    const dtype = mlx.mlx_array_dtype(x);
    const glue = dtype != .float32 or m_pad != m;

    const x32 = if (glue) try apply(try glueKernel(.prep), &.{x}, &.{ m_pad, k }, .float32, .{ k, m_pad, 1 }, .{ @min(k, 64), 1, 1 }, &.{}, s) else x;
    defer if (glue) {
        _ = mlx.mlx_array_free(x32);
    };
    var inputs: [3]mlx.mlx_array = undefined;
    inputs[0] = x32;
    @memcpy(inputs[1 .. 1 + rest.len], rest);
    const y32 = try apply(try kernel(kind, ty), inputs[0 .. 1 + rest.len], &.{ m_pad, rows }, .float32, .{ SIMD_WIDTH * @divExact(rows, tileRows(ty)), @divFloor(m_pad + TILE_TOKENS - 1, TILE_TOKENS), 1 }, .{ SIMD_WIDTH, 1, 1 }, &(layoutConsts(ty, k, row_bytes) ++ [_]Const{.{ "N", rows }}), s);
    if (!glue) return y32;
    defer _ = mlx.mlx_array_free(y32);
    const n_out = m * rows;
    return apply(try glueKernel(.finish), &.{y32}, &.{ m, rows }, dtype, .{ n_out, 1, 1 }, .{ @min(n_out, 64), 1, 1 }, &.{}, s);
}

const Glue = enum { prep, finish };

/// prep: x [M, K] of any float type -> f32 [M rounded up to 8, K], zero padded.
/// finish: f32 [M padded, N] -> T [M, N], the padding rows dropped.
fn glueKernel(comptime which: Glue) Error!mlx.mlx_fast_metal_kernel {
    const Cache = struct {
        const key = which;
        var kernel: ?mlx.mlx_fast_metal_kernel = null;
    };
    _ = Cache.key;
    if (Cache.kernel) |k| return k;
    const input_names = [_][*:0]const u8{"x"};
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("gguf_" ++ @tagName(which), in_vec, out_vec, switch (which) {
        .prep => @embedFile("metal/kernels/prep.metal"),
        // Rows are contiguous, so the first M * N values are the unpadded result.
        .finish => @embedFile("metal/kernels/finish.metal"),
    }, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    Cache.kernel = k;
    return k;
}

/// How a GGUF weight travels through mlx-serve's (weight, scales, biases)
/// triples: the weight is the raw uint8 bytes and `scales` is this sentinel, a
/// bool array whose SHAPE carries the metadata: [type+1] or [type+1, nk, r].
/// No MLX quant mode has bool scales, and reading a shape costs nothing.
pub const Info = struct {
    ty: GgmlType,
    /// Set when the weight's INPUT is laid out in llama.cpp's tiled value-head
    /// order (see arch/qwen35.zig): the activations get tiled before the matmul.
    tiled: ?struct { nk: c_int, r: c_int } = null,
};

pub fn sentinel(info: Info, s: mlx.mlx_stream) !mlx.mlx_array {
    const id: c_int = @intCast(@backingInt(info.ty) + 1);
    const shape: []const c_int = if (info.tiled) |t| &.{ id, t.nk, t.r } else &.{id};
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_zeros(&out, shape.ptr, shape.len, .bool_, s));
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// Null for anything that is not a GGUF weight, cheap enough for every matmul.
pub fn infoOf(w: mlx.mlx_array, sc: mlx.mlx_array) ?Info {
    if (sc.ctx == null or w.ctx == null) return null;
    if (mlx.mlx_array_dtype(sc) != .bool_ or mlx.mlx_array_dtype(w) != .uint8) return null;
    const sh = mlx.getShape(sc);
    if (sh.len != 1 and sh.len != 3) return null;
    const ty: GgmlType = @fromBackingInt(@intCast(@as(u32, @intCast(sh[0] - 1))));
    return .{ .ty = ty, .tiled = if (sh.len == 3) .{ .nk = sh[1], .r = sh[2] } else null };
}

/// y = x w^T for x [..., in_features], the drop-in for a quantized matmul.
pub fn linear(info: Info, x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const g = try geometry(info.ty, w);
    const xs = mlx.getShape(x);
    if (xs.len == 0 or xs[xs.len - 1] != g.in) return error.BadShape;
    const m: c_int = @intCast(@divExact(mlx.mlx_array_size(x), @as(usize, @intCast(g.in))));

    var xt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xt);
    if (info.tiled) |t| {
        // grouped [m, nk, r, d] -> tiled [m, r, nk, d], back in x's own shape
        if (@mod(g.in, t.nk * t.r) != 0) return error.BadShape;
        var heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(heads);
        try mlx.check(mlx.mlx_reshape(&heads, x, &[_]c_int{ m, t.nk, t.r, @divExact(g.in, t.nk * t.r) }, 4, s));
        var swapped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(swapped);
        try mlx.check(mlx.mlx_transpose_axes(&swapped, heads, &[_]c_int{ 0, 2, 1, 3 }, 4, s));
        try mlx.check(mlx.mlx_reshape(&xt, swapped, xs.ptr, xs.len, s));
    }
    const xin = if (info.tiled != null) xt else x;
    if (m <= MATVEC_MAX_ROWS) return matvec(info.ty, xin, w, s);

    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, xin, &[_]c_int{ m, g.in }, 2, s));
    const y2 = try wide(info.ty, x2, w, g.rows, s);
    defer _ = mlx.mlx_array_free(y2);

    var out_shape: [8]c_int = undefined;
    if (xs.len > out_shape.len) return error.BadShape;
    @memcpy(out_shape[0 .. xs.len - 1], xs[0 .. xs.len - 1]);
    out_shape[xs.len - 1] = g.rows;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, &out_shape, xs.len, s));
    return y;
}

/// x [m, in] for m past the matvec range.
fn wide(ty: GgmlType, x: mlx.mlx_array, w: mlx.mlx_array, rows: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    if (mlx.getShape(x)[0] > TILE_MAX_ROWS or @mod(rows, tileRows(ty)) != 0) return dequantMatmul(ty, x, w, s);
    return matmat(ty, x, w, s);
}

/// Fallback for row counts the tile kernel can't take: materialize the weight.
pub fn dequantMatmul(ty: GgmlType, x: mlx.mlx_array, w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const wd = try dequantize(ty, w, mlx.mlx_array_dtype(x), s);
    defer _ = mlx.mlx_array_free(wd);
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, wd, s));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_matmul(&y, x, wt, s));
    return y;
}

/// The ggml fixtures hold 4 blocks per type, used here as a [2 rows x 2 blocks] weight.
const TestWeight = struct {
    ty: GgmlType,
    arr: mlx.mlx_array,
    ref: []f32,
    in: usize,

    fn init(comptime ty: GgmlType) !TestWeight {
        const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
        const bytes = data[0 .. 4 * ty.blockBytes()];
        const ref = try std.testing.allocator.alloc(f32, 4 * ty.blockElems());
        errdefer std.testing.allocator.free(ref);
        try quants.dequantize(ty, bytes, ref);
        // Rows go in the layout the kernels expect.
        const in = 2 * ty.blockElems();
        const shape = [_]c_int{ 2, @intCast(rowBytes(ty, in)) };
        var rows: [4 * 256]u8 = undefined;
        if (isSplit(ty, in)) packRows(ty, in, rows[0 .. 2 * rowBytes(ty, in)], bytes) else @memcpy(rows[0..bytes.len], bytes);
        return .{ .ty = ty, .arr = mlx.mlx_array_new_data(&rows, &shape, 2, .uint8), .ref = ref, .in = in };
    }

    fn deinit(self: TestWeight) void {
        _ = mlx.mlx_array_free(self.arr);
        std.testing.allocator.free(self.ref);
    }
};

fn readF32(arr: mlx.mlx_array, n: usize) ![]const f32 {
    try mlx.check(mlx.mlx_array_eval(arr));
    return (mlx.mlx_array_data_float32(arr) orelse return error.BadShape)[0..n];
}

test "GPU dequant equals the CPU reference for every type" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    inline for (all_types) |ty| {
        const w = try TestWeight.init(ty);
        defer w.deinit();
        const got = try dequantize(ty, w.arr, .float32, s);
        defer _ = mlx.mlx_array_free(got);
        for (try readF32(got, w.ref.len), w.ref, 0..) |g, r, i| {
            if (!std.math.approxEqRel(f32, r, g, 1e-5)) {
                std.debug.print("{s}[{d}]: gpu {d} cpu {d}\n", .{ @tagName(ty), i, g, r });
                return error.TestUnexpectedResult;
            }
        }
    }
}

test "linear equals a CPU dot product on the matvec (1 and several rows) and the 2-row fallback path" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(7);
    inline for (all_types) |ty| {
        const w = try TestWeight.init(ty);
        defer w.deinit();
        for ([_]usize{ 1, 2, 3, 5, 6, 7, MATVEC_MAX_ROWS, MATVEC_MAX_ROWS + 3, 16 }) |m| {
            const xv = try std.testing.allocator.alloc(f32, m * w.in);
            defer std.testing.allocator.free(xv);
            for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
            // 3-D on purpose: the forward passes [batch, seq, hidden].
            const x = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, @intCast(m), @intCast(w.in) }, 3, .float32);
            defer _ = mlx.mlx_array_free(x);

            const y = try linear(.{ .ty = ty }, x, w.arr, s);
            defer _ = mlx.mlx_array_free(y);
            try std.testing.expectEqualSlices(c_int, &.{ 1, @intCast(m), 2 }, mlx.getShape(y));
            const got = try readF32(y, m * 2);
            // Past the matvec range the 2-row weight takes the dequantize + MLX GEMM fallback, whose f32 GEMM is not
            // a full-f32 dot on every GPU (NAX): there the bar is that GEMM over the reference weights.
            var gemm = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gemm);
            if (m > MATVEC_MAX_ROWS) {
                const wref = mlx.mlx_array_new_data(w.ref.ptr, &[_]c_int{ 2, @intCast(w.in) }, 2, .float32);
                defer _ = mlx.mlx_array_free(wref);
                var wt = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(wt);
                try mlx.check(mlx.mlx_transpose(&wt, wref, s));
                try mlx.check(mlx.mlx_matmul(&gemm, x, wt, s));
            }
            const ref_gemm = if (m > MATVEC_MAX_ROWS) try readF32(gemm, m * 2) else &.{};
            for (0..m) |mi| for (0..2) |r| {
                var want: f32 = 0;
                var scale: f32 = 0;
                for (xv[mi * w.in ..][0..w.in], w.ref[r * w.in ..][0..w.in]) |a, b| {
                    want += a * b;
                    scale += @abs(a * b);
                }
                if (m > MATVEC_MAX_ROWS) want = ref_gemm[mi * 2 + r];
                try std.testing.expectApproxEqAbs(want, got[mi * 2 + r], 1e-5 * scale);
            };
        }
    }
}

test "the column-major matvec (32-weight units, rows of 1024 weights and a ragged 1280, 1-2 rows) equals a CPU dot product" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(11);
    const rows = 3;
    // 1280 weights: 40 units, so the first 8 lanes take a second one.
    for ([_]usize{ 1024, 1280 }) |in| inline for (all_types) |ty| if (comptime ty.unitElems() == 32) {
        // The fixture's 4 blocks, cycled over 3 rows of 1024 weights.
        const fixture = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin")[0 .. 4 * ty.blockBytes()];
        const nb = rows * in / ty.blockElems();
        const raw = try a.alloc(u8, nb * ty.blockBytes());
        defer a.free(raw);
        for (0..nb) |b| @memcpy(raw[b * ty.blockBytes() ..][0..ty.blockBytes()], fixture[(b % 4) * ty.blockBytes() ..][0..ty.blockBytes()]);
        const ref = try a.alloc(f32, rows * in);
        defer a.free(ref);
        try quants.dequantize(ty, raw, ref);
        const stored = try a.alloc(u8, rows * rowBytes(ty, in));
        defer a.free(stored);
        if (isSplit(ty, in)) packRows(ty, in, stored, raw) else @memcpy(stored, raw);
        const w = mlx.mlx_array_new_data(stored.ptr, &[_]c_int{ rows, @intCast(rowBytes(ty, in)) }, 2, .uint8);
        defer _ = mlx.mlx_array_free(w);
        for ([_]usize{ 1, 2 }) |m| {
            try std.testing.expect(colsServes(ty, @intCast(in), @intCast(m)));
            const xv = try a.alloc(f32, m * in);
            defer a.free(xv);
            for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
            const x = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, @intCast(m), @intCast(in) }, 3, .float32);
            defer _ = mlx.mlx_array_free(x);
            const y = try linear(.{ .ty = ty }, x, w, s);
            defer _ = mlx.mlx_array_free(y);
            const got = try readF32(y, m * rows);
            for (0..m) |mi| for (0..rows) |r| {
                var want: f32 = 0;
                var scale: f32 = 0;
                for (xv[mi * in ..][0..in], ref[r * in ..][0..in]) |xa, wb| {
                    want += xa * wb;
                    scale += @abs(xa * wb);
                }
                try std.testing.expectApproxEqAbs(want, got[mi * rows + r], 1e-5 * scale);
            };
        }
    };
}

test "tile mat-mat kernel (32 rows, whole and padded token tiles) equals a CPU dot product" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(23);
    inline for (all_types) |ty| {
        // The 4 fixture blocks 8 times over: 32 rows of one block each.
        const be = ty.blockElems();
        const bb = ty.blockBytes();
        const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
        const n_rows = 32;
        var bytes: [n_rows * 256]u8 = undefined;
        for (0..n_rows / 4) |i| @memcpy(bytes[4 * i * bb ..][0 .. 4 * bb], data[0 .. 4 * bb]);
        var ref: [4 * 256]f32 = undefined;
        try quants.dequantize(ty, data[0 .. 4 * bb], ref[0 .. 4 * be]);
        var rows: [n_rows * 256]u8 = undefined;
        if (isSplit(ty, be)) packRows(ty, be, rows[0 .. n_rows * rowBytes(ty, be)], bytes[0 .. n_rows * bb]) else @memcpy(rows[0 .. n_rows * bb], bytes[0 .. n_rows * bb]);
        const w = mlx.mlx_array_new_data(&rows, &[_]c_int{ n_rows, @intCast(rowBytes(ty, be)) }, 2, .uint8);
        defer _ = mlx.mlx_array_free(w);

        for ([_]usize{ 32, 41, 9 }) |m| {
            const xv = try std.testing.allocator.alloc(f32, m * be);
            defer std.testing.allocator.free(xv);
            for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
            const x = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ @intCast(m), @intCast(be) }, 2, .float32);
            defer _ = mlx.mlx_array_free(x);
            const y = try linear(.{ .ty = ty }, x, w, s);
            defer _ = mlx.mlx_array_free(y);
            try std.testing.expectEqualSlices(c_int, &.{ @intCast(m), n_rows }, mlx.getShape(y));
            const got = try readF32(y, m * n_rows);
            for (0..m) |mi| for (0..n_rows) |r| {
                var want: f32 = 0;
                var scale: f32 = 0;
                for (xv[mi * be ..][0..be], ref[(r % 4) * be ..][0..be]) |a, b| {
                    want += a * b;
                    scale += @abs(a * b);
                }
                if (@abs(want - got[mi * n_rows + r]) > 1e-5 * scale) {
                    std.debug.print("{s} m={d} token {d} row {d}: got {d} want {d}\n", .{ @tagName(ty), m, mi, r, got[mi * n_rows + r], want });
                    return error.TestUnexpectedResult;
                }
            };
        }
    }
}

test "gatherLinear routes every output to its expert, with and without lhs indices" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(5);
    inline for ([_]GgmlType{ .iq2_xxs, .q5_k, .iq4_nl }) |ty| {
        // The 4 fixture blocks as a bank of 2 experts x 2 rows x 1 block.
        const tw = try TestWeight.init(ty);
        defer tw.deinit();
        const in = ty.blockElems();
        const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
        var rows: [4 * 256]u8 = undefined;
        if (isSplit(ty, in)) packRows(ty, in, rows[0 .. 4 * rowBytes(ty, in)], data[0 .. 4 * ty.blockBytes()]) else @memcpy(rows[0 .. 4 * ty.blockBytes()], data[0 .. 4 * ty.blockBytes()]);
        const bank = mlx.mlx_array_new_data(&rows, &[_]c_int{ 2, 2, @intCast(rowBytes(ty, in)) }, 3, .uint8);
        defer _ = mlx.mlx_array_free(bank);

        const xv = try std.testing.allocator.alloc(f32, 3 * in);
        defer std.testing.allocator.free(xv);
        for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
        const Case = struct { x_shape: []const c_int, lhs: ?[]const u32, rhs: []const u32, rhs_shape: []const c_int, x_row: []const usize };
        const cases = [_]Case{
            // decode layout: 3 tokens, top-2 experts each
            .{ .x_shape = &.{ 1, 3, 1, 1, @intCast(in) }, .lhs = null, .rhs = &.{ 0, 1, 1, 1, 1, 0 }, .rhs_shape = &.{ 1, 3, 2 }, .x_row = &.{ 0, 0, 1, 1, 2, 2 } },
            // sorted prefill layout: rows picked by lhs
            .{ .x_shape = &.{ 3, 1, @intCast(in) }, .lhs = &.{ 2, 0, 0, 1 }, .rhs = &.{ 0, 0, 1, 1 }, .rhs_shape = &.{4}, .x_row = &.{ 2, 0, 0, 1 } },
        };
        for (cases) |c| {
            const x = mlx.mlx_array_new_data(xv.ptr, c.x_shape.ptr, @intCast(c.x_shape.len), .float32);
            defer _ = mlx.mlx_array_free(x);
            const rhs = mlx.mlx_array_new_data(c.rhs.ptr, c.rhs_shape.ptr, @intCast(c.rhs_shape.len), .uint32);
            defer _ = mlx.mlx_array_free(rhs);
            const lhs = if (c.lhs) |l| mlx.mlx_array_new_data(l.ptr, c.rhs_shape.ptr, @intCast(c.rhs_shape.len), .uint32) else mlx.mlx_array{ .ctx = null };
            defer if (c.lhs != null) {
                _ = mlx.mlx_array_free(lhs);
            };
            const y = try gatherLinear(.{ .ty = ty }, x, bank, lhs, rhs, c.lhs != null, s);
            defer _ = mlx.mlx_array_free(y);
            const ys = mlx.getShape(y);
            try std.testing.expectEqualSlices(c_int, c.rhs_shape, ys[0..c.rhs_shape.len]);
            try std.testing.expectEqualSlices(c_int, &.{ 1, 2 }, ys[c.rhs_shape.len..]);
            const got = try readF32(y, c.rhs.len * 2);
            for (c.rhs, c.x_row, 0..) |e, xr, t| for (0..2) |r| {
                var want: f32 = 0;
                var scale: f32 = 0;
                for (xv[xr * in ..][0..in], tw.ref[(e * 2 + r) * in ..][0..in]) |a, b| {
                    want += a * b;
                    scale += @abs(a * b);
                }
                try std.testing.expectApproxEqAbs(want, got[t * 2 + r], 1e-5 * scale);
            };
        }
    }
}

test "the decode gather (rows of whole 32-unit strides and not, a partial row tile) equals a CPU dot product for every type" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(17);
    const a = std.testing.allocator;
    const n_exp = 3;
    const n_rows = 6;
    // 2 tokens, top-2: the decode layout, each token's row shared by its experts.
    const ids = [_]u32{ 2, 0, 1, 2 };
    inline for (all_types) |ty| {
        inline for (.{ 1024, 768 }) |k| {
            const be = ty.blockElems();
            const bb = ty.blockBytes();
            const nb = k / be;
            const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
            var ref: [4 * 256]f32 = undefined;
            try quants.dequantize(ty, data[0 .. 4 * bb], ref[0 .. 4 * be]);
            const raw = try a.alloc(u8, n_exp * n_rows * nb * bb);
            defer a.free(raw);
            for (0..n_exp) |e| for (0..n_rows) |r| for (0..nb) |j| @memcpy(raw[((e * n_rows + r) * nb + j) * bb ..][0..bb], data[((r + e + j) % 4) * bb ..][0..bb]);
            const rb = rowBytes(ty, k);
            const packed_rows = try a.alloc(u8, n_exp * n_rows * rb);
            defer a.free(packed_rows);
            if (isSplit(ty, k)) packRows(ty, k, packed_rows, raw) else @memcpy(packed_rows, raw);
            const bank = mlx.mlx_array_new_data(packed_rows.ptr, &[_]c_int{ n_exp, n_rows, @intCast(rb) }, 3, .uint8);
            defer _ = mlx.mlx_array_free(bank);
            var xv: [2 * k]f32 = undefined;
            for (&xv) |*v| v.* = prng.random().float(f32) - 0.5;
            const x = mlx.mlx_array_new_data(&xv, &[_]c_int{ 1, 2, 1, 1, k }, 5, .float32);
            defer _ = mlx.mlx_array_free(x);
            const idv = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, 2, 2 }, 3, .uint32);
            defer _ = mlx.mlx_array_free(idv);
            const y = try gatherLinear(.{ .ty = ty }, x, bank, .{ .ctx = null }, idv, false, s);
            defer _ = mlx.mlx_array_free(y);
            try std.testing.expectEqualSlices(c_int, &.{ 1, 2, 2, 1, n_rows }, mlx.getShape(y));
            const got = try readF32(y, ids.len * n_rows);
            for (ids, 0..) |e, p| for (0..n_rows) |r| {
                var want: f32 = 0;
                var scale: f32 = 0;
                for (0..nb) |j| for (0..be) |i| {
                    const prod = xv[(p / 2) * k + j * be + i] * ref[((r + e + j) % 4) * be + i];
                    want += prod;
                    scale += @abs(prod);
                };
                if (@abs(want - got[p * n_rows + r]) > 2e-5 * scale) {
                    std.debug.print("{s} k {d} pair {d} expert {d} row {d}: got {d} want {d}\n", .{ @tagName(ty), k, p, e, r, got[p * n_rows + r], want });
                    return error.TestUnexpectedResult;
                }
            };
        }
    }
}

test "gathered tile path (sorted and unsorted prefill rows, padded tile) equals a CPU dot product" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(29);
    inline for ([_]GgmlType{ .iq4_nl, .q6_k, .iq2_xxs }) |ty| {
        // 4 experts x 32 rows of one block each; expert e's row r is fixture block (r + e) % 4.
        const be = ty.blockElems();
        const bb = ty.blockBytes();
        const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
        const n_exp = 4;
        const n_rows = 32;
        var bytes: [n_exp * n_rows * 256]u8 = undefined;
        for (0..n_exp) |e| for (0..n_rows) |r| @memcpy(bytes[(e * n_rows + r) * bb ..][0..bb], data[((r + e) % 4) * bb ..][0..bb]);
        var ref: [4 * 256]f32 = undefined;
        try quants.dequantize(ty, data[0 .. 4 * bb], ref[0 .. 4 * be]);
        var rows: [n_exp * n_rows * 256]u8 = undefined;
        if (isSplit(ty, be)) packRows(ty, be, rows[0 .. n_exp * n_rows * rowBytes(ty, be)], bytes[0 .. n_exp * n_rows * bb]) else @memcpy(rows[0 .. n_exp * n_rows * bb], bytes[0 .. n_exp * n_rows * bb]);
        const bank = mlx.mlx_array_new_data(&rows, &[_]c_int{ n_exp, n_rows, @intCast(rowBytes(ty, be)) }, 3, .uint8);
        defer _ = mlx.mlx_array_free(bank);

        // 41 rows: the tile path (one whole tile plus a padded one); 80: the
        // materialized-bank path. Sorted ids first, then a shuffled copy.
        inline for ([_]usize{ 41, 80 }) |m| {
            var ids: [m]u32 = undefined;
            for (&ids, 0..) |*id, i| id.* = @intCast(i * n_exp / m);
            var shuffled = ids;
            prng.random().shuffle(u32, &shuffled);
            const xv = try std.testing.allocator.alloc(f32, m * be);
            defer std.testing.allocator.free(xv);
            for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
            const x = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ m, 1, @intCast(be) }, 3, .float32);
            defer _ = mlx.mlx_array_free(x);
            for ([_][]const u32{ &ids, &shuffled }) |order| {
                const rhs = mlx.mlx_array_new_data(order.ptr, &[_]c_int{m}, 1, .uint32);
                defer _ = mlx.mlx_array_free(rhs);
                const y = try gatherLinear(.{ .ty = ty }, x, bank, .{ .ctx = null }, rhs, order.ptr == &ids, s);
                defer _ = mlx.mlx_array_free(y);
                try std.testing.expectEqualSlices(c_int, &.{ m, 1, n_rows }, mlx.getShape(y));
                const got = try readF32(y, m * n_rows);
                for (order, 0..) |e, t| for (0..n_rows) |r| {
                    var want: f32 = 0;
                    var scale: f32 = 0;
                    for (xv[t * be ..][0..be], ref[((r + e) % 4) * be ..][0..be]) |a, b| {
                        want += a * b;
                        scale += @abs(a * b);
                    }
                    if (@abs(want - got[t * n_rows + r]) > 1e-5 * scale) {
                        std.debug.print("{s} m={d} token {d} expert {d} row {d}: got {d} want {d}\n", .{ @tagName(ty), m, t, e, r, got[t * n_rows + r], want });
                        return error.TestUnexpectedResult;
                    }
                };
            }
        }
    }
}

test "NAX gather (ragged sorted runs: one empty, one past a tile, gathered rows, short and long) equals a CPU dot product for every type" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu() or !device.naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(31);
    const a = std.testing.allocator;
    const n_exp = 4;
    const n_rows = 128;
    const k = 512;
    // Expert 1's run is empty, expert 2's spans two tiles (then five), expert 3's ends a tile short (then a run of
    // three tiles whose last window is clamped to the end of x).
    for ([_][n_exp]usize{ .{ 5, 0, 45, 30 }, .{ 5, 0, 300, 150 } }) |runs| {
    var m: usize = 0;
    for (runs) |len| m += len;
    const ids = try a.alloc(u32, m);
    defer a.free(ids);
    var at: usize = 0;
    for (runs, 0..) |len, e| for (0..len) |_| {
        ids[at] = @intCast(e);
        at += 1;
    };
    inline for (all_types) |ty| {
        const be = ty.blockElems();
        const bb = ty.blockBytes();
        const nb = k / be;
        const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
        var ref: [4 * 256]f32 = undefined;
        try quants.dequantize(ty, data[0 .. 4 * bb], ref[0 .. 4 * be]);
        // Expert e's row r, block j: fixture block (r + e + j) % 4.
        const raw = try a.alloc(u8, n_exp * n_rows * nb * bb);
        defer a.free(raw);
        for (0..n_exp) |e| for (0..n_rows) |r| for (0..nb) |j| @memcpy(raw[((e * n_rows + r) * nb + j) * bb ..][0..bb], data[((r + e + j) % 4) * bb ..][0..bb]);
        const rb = rowBytes(ty, k);
        const packed_rows = try a.alloc(u8, n_exp * n_rows * rb);
        defer a.free(packed_rows);
        if (isSplit(ty, k)) packRows(ty, k, packed_rows, raw) else @memcpy(packed_rows, raw);
        const bank = mlx.mlx_array_new_data(packed_rows.ptr, &[_]c_int{ n_exp, n_rows, @intCast(rb) }, 3, .uint8);
        defer _ = mlx.mlx_array_free(bank);
        // bf16 activations, the values the reference multiplies too.
        const xv = try a.alloc(f32, m * k);
        defer a.free(xv);
        for (xv) |*v| v.* = @floatCast(@as(f32, @bitCast(@as(u32, @bitCast(prng.random().float(f32) - 0.5)) & 0xFFFF0000)));
        const x32 = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ @intCast(m), 1, k }, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
        const idv = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(m)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(idv);
        const y = try gatherLinear(.{ .ty = ty }, x, bank, .{ .ctx = null }, idv, true, s);
        defer _ = mlx.mlx_array_free(y);
        try std.testing.expectEqualSlices(c_int, &.{ @intCast(m), 1, n_rows }, mlx.getShape(y));
        const got = try testValues(y, s);
        defer a.free(got);
        // The same pairs behind a leading batch dim: the gather reads them as pairs all the same.
        var x4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x4);
        try mlx.check(mlx.mlx_reshape(&x4, x, &[_]c_int{ 1, @intCast(m), 1, k }, 4, s));
        const y4 = try gatherLinear(.{ .ty = ty }, x4, bank, .{ .ctx = null }, idv, true, s);
        defer _ = mlx.mlx_array_free(y4);
        const got4 = try testValues(y4, s);
        defer a.free(got4);
        try std.testing.expectEqualSlices(f32, got, got4);
        for (ids, 0..) |e, t| for (0..n_rows) |r| {
            var want: f32 = 0;
            var scale: f32 = 0;
            for (0..nb) |j| for (0..be) |i| {
                const prod = xv[t * k + j * be + i] * ref[((r + e + j) % 4) * be + i];
                want += prod;
                scale += @abs(prod);
            };
            // The weights round to bf16 on the matrix units, and so does the output.
            if (@abs(want - got[t * n_rows + r]) > 8e-3 * scale) {
                std.debug.print("{s} {d} pairs, pair {d} expert {d} row {d}: got {d} want {d}\n", .{ @tagName(ty), m, t, e, r, got[t * n_rows + r], want });
                return error.TestUnexpectedResult;
            }
        };
    }
    }
}

test "NAX gate+up (clipped SwiGLU times each pair's weight) equals the two gathers and the activation, for every type" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu() or !device.naxAvailable()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(37);
    const a = std.testing.allocator;
    const n_exp = 4;
    const n_rows = 128;
    const k = 512;
    const runs = [n_exp]usize{ 5, 0, 45, 30 };
    const m = 80;
    var ids: [m]u32 = undefined;
    var at: usize = 0;
    for (runs, 0..) |len, e| for (0..len) |_| {
        ids[at] = @intCast(e);
        at += 1;
    };
    var wv: [m]f32 = undefined;
    for (&wv) |*v| v.* = 0.05 + prng.random().float(f32);
    inline for (all_types) |ty| {
        const be = ty.blockElems();
        const bb = ty.blockBytes();
        const nb = k / be;
        const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
        var banks: [2]mlx.mlx_array = undefined;
        for (&banks, 0..) |*bk, salt| {
            const raw = try a.alloc(u8, n_exp * n_rows * nb * bb);
            defer a.free(raw);
            for (0..n_exp) |e| for (0..n_rows) |r| for (0..nb) |j| @memcpy(raw[((e * n_rows + r) * nb + j) * bb ..][0..bb], data[((r + 2 * e + j + salt) % 4) * bb ..][0..bb]);
            const rb = rowBytes(ty, k);
            const packed_rows = try a.alloc(u8, n_exp * n_rows * rb);
            defer a.free(packed_rows);
            if (isSplit(ty, k)) packRows(ty, k, packed_rows, raw) else @memcpy(packed_rows, raw);
            bk.* = mlx.mlx_array_new_data(packed_rows.ptr, &[_]c_int{ n_exp, n_rows, @intCast(rb) }, 3, .uint8);
        }
        defer for (banks) |bk| {
            _ = mlx.mlx_array_free(bk);
        };
        var xv: [m * k]f32 = undefined;
        for (&xv) |*v| v.* = (prng.random().float(f32) - 0.5) * 4;
        const x32 = mlx.mlx_array_new_data(&xv, &[_]c_int{ m, 1, k }, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
        const idv = mlx.mlx_array_new_data(&ids, &[_]c_int{m}, 1, .uint32);
        defer _ = mlx.mlx_array_free(idv);
        const wts = mlx.mlx_array_new_data(&wv, &[_]c_int{m}, 1, .float32);
        defer _ = mlx.mlx_array_free(wts);
        const g = try gatherLinear(.{ .ty = ty }, x, banks[0], .{ .ctx = null }, idv, true, s);
        defer _ = mlx.mlx_array_free(g);
        const u = try gatherLinear(.{ .ty = ty }, x, banks[1], .{ .ctx = null }, idv, true, s);
        defer _ = mlx.mlx_array_free(u);
        const gv = try testValues(g, s);
        defer a.free(gv);
        const uv = try testValues(u, s);
        defer a.free(uv);
        for ([_]f32{ 0, 0.5 }) |l| {
            const lim = mlx.mlx_array_new_data(&l, &[_]c_int{1}, 1, .float32);
            defer _ = mlx.mlx_array_free(lim);
            const y = (try gatherNaxSwiGLU(.{ .ty = ty }, .{ .ty = ty }, x, banks[0], banks[1], idv, wts, lim, s)).?;
            defer _ = mlx.mlx_array_free(y);
            try std.testing.expectEqualSlices(c_int, &.{ m, n_rows }, mlx.getShape(y));
            const got = try testValues(y, s);
            defer a.free(got);
            for (0..m) |t| for (0..n_rows) |r| {
                var gg = gv[t * n_rows + r];
                var uu = uv[t * n_rows + r];
                if (l > 0) {
                    gg = @min(gg, l);
                    uu = std.math.clamp(uu, -l, l);
                }
                const want = gg / (1 + @exp(-gg)) * uu * wv[t];
                // One bf16 rounding of the same f32 product.
                if (@abs(want - got[t * n_rows + r]) > 0x1p-7 * @abs(want) + 1e-6) {
                    std.debug.print("{s} lim {d} pair {d} row {d}: got {d} want {d}\n", .{ @tagName(ty), l, t, r, got[t * n_rows + r], want });
                    return error.TestUnexpectedResult;
                }
            };
        }
    }
}

test "NAX gather ubench: the DeepSeek-V4.1 Coder's expert banks at 2048- and 8192-token chunks, against the tile kernel and MLX's 2-bit affine" {
    const env = @import("../../../env.zig");
    if (env.get(.GGUF_NAX_UBENCH) == null or !device.naxAvailable()) return error.SkipZigTest;
    const iou = @import("../../util/io_util.zig");
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(5);
    const cases = [_]struct { ty: GgmlType, rows: c_int, in: c_int, experts: u32, pairs: usize }{
        .{ .ty = .iq2_xxs, .rows = 2304, .in = 5120, .experts = 192, .pairs = 2048 * 6 },
        .{ .ty = .iq1_m, .rows = 5120, .in = 2304, .experts = 192, .pairs = 2048 * 6 },
        // The Coder's 8k chunk: 192 experts, ~256 pairs each.
        .{ .ty = .iq2_xxs, .rows = 2304, .in = 5120, .experts = 192, .pairs = 8192 * 6 },
        .{ .ty = .iq1_m, .rows = 5120, .in = 2304, .experts = 192, .pairs = 8192 * 6 },
    };
    inline for (cases) |c| {
        const experts = c.experts;
        const pairs = c.pairs;
        const ids = try a.alloc(u32, pairs);
        defer a.free(ids);
        for (ids) |*id| id.* = prng.random().uintLessThan(u32, experts);
        std.mem.sort(u32, ids, {}, std.sort.asc(u32));
        const data = @embedFile("fixtures/" ++ @tagName(c.ty) ++ ".bin");
        const bb = c.ty.blockBytes();
        const rb = rowBytes(c.ty, c.in);
        const bank_bytes = try a.alloc(u8, experts * @as(usize, @intCast(c.rows)) * rb);
        defer a.free(bank_bytes);
        for (0..bank_bytes.len / bb) |i| @memcpy(bank_bytes[i * bb ..][0..bb], data[(i % 4) * bb ..][0..bb]);
        const bank = mlx.mlx_array_new_data(bank_bytes.ptr, &[_]c_int{ experts, c.rows, @intCast(rb) }, 3, .uint8);
        defer _ = mlx.mlx_array_free(bank);
        const xv = try a.alloc(f32, pairs * @as(usize, @intCast(c.in)));
        defer a.free(xv);
        for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
        const x32 = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ pairs, 1, c.in }, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
        const idv = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{pairs}, 1, .uint32);
        defer _ = mlx.mlx_array_free(idv);
        var lhv = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lhv);
        try mlx.check(mlx.mlx_arange(&lhv, 0, pairs, 1, .uint32, s));
        try mlx.check(mlx.mlx_array_eval(x));
        // What the MLX packs run: 2-bit group-128 affine experts through MLX's gather_qmm.
        var qw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qw);
        var qs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qs);
        try mlx.check(mlx.mlx_zeros(&qw, &[_]c_int{ experts, c.rows, @divExact(c.in * 2, 32) }, 3, .uint32, s));
        try mlx.check(mlx.mlx_zeros(&qs, &[_]c_int{ experts, c.rows, @divExact(c.in, 128) }, 3, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(qw));
        try mlx.check(mlx.mlx_array_eval(qs));
        var qb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qb);
        try mlx.check(mlx.mlx_zeros(&qb, &[_]c_int{ experts, c.rows, @divExact(c.in, 128) }, 3, .bfloat16, s));
        try mlx.check(mlx.mlx_array_eval(qb));
        const sdk_nax = @import("../../blocks/moe/gather_qmm_nax.zig");
        var ms: [4]f64 = undefined;
        for (0..4) |arm| {
            var times: [5]f64 = undefined;
            for (0..2 + times.len) |rep| {
                const t0 = iou.nowUsMonotonic(io);
                var y = mlx.mlx_array_new();
                switch (arm) {
                    0 => y = try gatherLinear(.{ .ty = c.ty }, x, bank, .{ .ctx = null }, idv, true, s),
                    1 => y = try gatherLinear(.{ .ty = c.ty }, x, bank, lhv, idv, false, s),
                    2 => try mlx.check(mlx.mlx_gather_qmm(&y, x, qw, qs, qs, .{ .ctx = null }, idv, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", true, s)),
                    else => y = (try sdk_nax.sortedGatherPrompt(x, qw, qs, qb, idv, 2, 128, false, true, s)).?,
                }
                defer _ = mlx.mlx_array_free(y);
                try mlx.check(mlx.mlx_array_eval(y));
                if (rep >= 2) times[rep - 2] = @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / 1e3;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            ms[arm] = times[2];
        }
        const flops = 2.0 * @as(f64, @floatFromInt(pairs)) * @as(f64, @floatFromInt(c.rows)) * @as(f64, @floatFromInt(c.in));
        std.debug.print("[gguf-nax-ubench] {s} {d}x{d}, {d} pairs over {d} experts: NAX {d:.2} ms ({d:.0} TFLOPS), unsorted (tile kernel) {d:.2} ms; 2-bit g128 affine: MLX gather_qmm {d:.2} ms, ours (sortedGatherPrompt) {d:.2} ms\n", .{ @tagName(c.ty), c.rows, c.in, pairs, experts, ms[0], flops / ms[0] / 1e9, ms[1], ms[2], ms[3] });
    }
}

test "decode gather ubench: one token through 6 of 192 experts, 40 layers in one graph, against MLX's 2-bit g128" {
    const env = @import("../../../env.zig");
    if (env.get(.GGUF_NAX_UBENCH) == null or mlx.noGpuBackend()) return error.SkipZigTest;
    const iou = @import("../../util/io_util.zig");
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(9);
    const experts = 192;
    const k = 6;
    const layers = 40;
    const cases = [_]struct { ty: GgmlType, rows: c_int, in: c_int, xk: c_int }{ .{ .ty = .iq2_xxs, .rows = 2304, .in = 5120, .xk = 1 }, .{ .ty = .iq1_m, .rows = 5120, .in = 2304, .xk = k } };
    inline for (cases) |c| {
        const data = @embedFile("fixtures/" ++ @tagName(c.ty) ++ ".bin");
        const bb = c.ty.blockBytes();
        const rb = rowBytes(c.ty, c.in);
        const bank_bytes = try a.alloc(u8, experts * @as(usize, @intCast(c.rows)) * rb);
        defer a.free(bank_bytes);
        for (0..bank_bytes.len / bb) |i| @memcpy(bank_bytes[i * bb ..][0..bb], data[(i % 4) * bb ..][0..bb]);
        const bank = mlx.mlx_array_new_data(bank_bytes.ptr, &[_]c_int{ experts, c.rows, @intCast(rb) }, 3, .uint8);
        defer _ = mlx.mlx_array_free(bank);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ 1, c.xk, 1, c.in }, 4, .bfloat16, 0, 1, .{ .ctx = null }, s));
        var qw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qw);
        var qs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qs);
        try mlx.check(mlx.mlx_zeros(&qw, &[_]c_int{ experts, c.rows, @divExact(c.in * 2, 32) }, 3, .uint32, s));
        try mlx.check(mlx.mlx_zeros(&qs, &[_]c_int{ experts, c.rows, @divExact(c.in, 128) }, 3, .bfloat16, s));
        var idx: [layers]mlx.mlx_array = undefined;
        for (&idx) |*v| {
            var ids: [k]u32 = undefined;
            for (&ids) |*id| id.* = prng.random().uintLessThan(u32, experts);
            v.* = mlx.mlx_array_new_data(&ids, &[_]c_int{ 1, k }, 2, .uint32);
        }
        defer for (idx) |v| {
            _ = mlx.mlx_array_free(v);
        };
        for ([_]mlx.mlx_array{ x, qw, qs, bank }) |v| try mlx.check(mlx.mlx_array_eval(v));
        var us: [2]f64 = undefined;
        for (0..2) |arm| {
            var times: [5]f64 = undefined;
            for (0..4 + times.len) |rep| {
                const outs = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(outs);
                const t0 = iou.nowUsMonotonic(io);
                for (idx) |iv| {
                    var y = mlx.mlx_array_new();
                    if (arm == 0) y = try gatherLinear(.{ .ty = c.ty }, x, bank, .{ .ctx = null }, iv, false, s) else try mlx.check(mlx.mlx_gather_qmm(&y, x, qw, qs, qs, .{ .ctx = null }, iv, true, mlx.mlx_optional_int.some(128), mlx.mlx_optional_int.some(2), "affine", false, s));
                    defer _ = mlx.mlx_array_free(y);
                    try mlx.check(mlx.mlx_vector_array_append_value(outs, y));
                }
                try mlx.check(mlx.mlx_eval(outs));
                if (rep >= 4) times[rep - 4] = @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / layers;
            }
            std.mem.sort(f64, &times, {}, std.sort.asc(f64));
            us[arm] = times[2];
        }
        const gbytes = @as(f64, @floatFromInt(k * @as(usize, @intCast(c.rows)) * rb)) / 1e3;
        std.debug.print("[gguf-decode-ubench] {s} {d}x{d}: GGUF gather {d:.1} us ({d:.0} GB/s), MLX 2-bit g128 gather_qmm {d:.1} us\n", .{ @tagName(c.ty), c.rows, c.in, us[0], gbytes / us[0], us[1] });
    }
}

/// The test banks: 4 experts of `test_bank_rows` one-block rows cut from the type's 4 fixture blocks, `shift` picking
/// another order.
const test_bank_rows = 36;

fn testBank(comptime ty: GgmlType, shift: usize) mlx.mlx_array {
    const be = ty.blockElems();
    const bb = ty.blockBytes();
    const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
    const n = 4 * test_bank_rows;
    var bytes: [n * 256]u8 = undefined;
    for (0..4) |e| for (0..test_bank_rows) |r| @memcpy(bytes[(e * test_bank_rows + r) * bb ..][0..bb], data[((r + e + shift) % 4) * bb ..][0..bb]);
    var rows: [n * 256]u8 = undefined;
    if (isSplit(ty, be)) packRows(ty, be, rows[0 .. n * rowBytes(ty, be)], bytes[0 .. n * bb]) else @memcpy(rows[0 .. n * bb], bytes[0 .. n * bb]);
    return mlx.mlx_array_new_data(&rows, &[_]c_int{ 4, test_bank_rows, @intCast(rowBytes(ty, be)) }, 3, .uint8);
}

/// `y`'s values as f32 (exact for a 16-bit float), owned by the caller.
fn testValues(y: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var y32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y32);
    try mlx.check(mlx.mlx_astype(&y32, y, .float32, s));
    return std.testing.allocator.dupe(f32, try readF32(y32, mlx.mlx_array_size(y)));
}

// Bar: one pass over both banks writes exactly what the two gathers and MLX's SwiGLU chain write.
test "gatherSwiGLU equals two gathers and MLX's SwiGLU bit for bit, at the decode and the sorted verify layouts" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const fused_activations = @import("../../blocks/mlp/fused_activations.zig");
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(17);
    inline for ([_]GgmlType{ .iq2_xxs, .iq2_xs, .iq2_s, .iq3_xxs, .iq3_s, .iq4_nl, .q4_k, .q6_k }) |ty| {
        const k: c_int = @intCast(ty.blockElems());
        const n_rows = test_bank_rows;
        const banks = [2]mlx.mlx_array{ testBank(ty, 0), testBank(ty, 3) };
        defer for (banks) |b| {
            _ = mlx.mlx_array_free(b);
        };
        const Case = struct { x_shape: []const c_int, ids: []const u32, ids_shape: []const c_int };
        const cases = [_]Case{
            // decode: one token's top-3 experts
            .{ .x_shape = &.{ 1, 1, 1, 1, k }, .ids = &.{ 2, 0, 3 }, .ids_shape = &.{ 1, 1, 3 } },
            // sorted verify: x already repeated per pair, pairs sorted by expert
            .{ .x_shape = &.{ 5, 1, k }, .ids = &.{ 0, 0, 1, 3, 3 }, .ids_shape = &.{5} },
        };
        for (cases) |c| for ([_]mlx.mlx_dtype{ .bfloat16, .float16 }) |dt| {
            var n_x: usize = 1;
            for (c.x_shape) |d| n_x *= @intCast(d);
            const xv = try std.testing.allocator.alloc(f32, n_x);
            defer std.testing.allocator.free(xv);
            for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
            const x32 = mlx.mlx_array_new_data(xv.ptr, c.x_shape.ptr, @intCast(c.x_shape.len), .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, dt, s));
            const ids = mlx.mlx_array_new_data(c.ids.ptr, c.ids_shape.ptr, @intCast(c.ids_shape.len), .uint32);
            defer _ = mlx.mlx_array_free(ids);

            const g = try gatherLinear(.{ .ty = ty }, x, banks[0], .{ .ctx = null }, ids, false, s);
            defer _ = mlx.mlx_array_free(g);
            const u = try gatherLinear(.{ .ty = ty }, x, banks[1], .{ .ctx = null }, ids, false, s);
            defer _ = mlx.mlx_array_free(u);
            var sig = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sig);
            try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
            var act = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(act);
            try mlx.check(mlx.mlx_multiply(&act, g, sig, s));
            var want = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(want);
            try mlx.check(mlx.mlx_multiply(&want, act, u, s));

            const got = (try gatherSwiGLU(.{ .ty = ty }, .{ .ty = ty }, x, banks[0], banks[1], ids, try fused_activations.sigmoidTableFor(s, dt), s)).?;
            defer _ = mlx.mlx_array_free(got);
            try std.testing.expectEqual(dt, mlx.mlx_array_dtype(got));
            try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
            const want_v = try testValues(want, s);
            defer std.testing.allocator.free(want_v);
            const got_v = try testValues(got, s);
            defer std.testing.allocator.free(got_v);
            try std.testing.expectEqual(c.ids.len * n_rows, got_v.len);
            try std.testing.expectEqualSlices(f32, want_v, got_v);
        };
    }
}

test "split rows (16 blocks per row, packed at load) match the CPU reference for every split type" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(31);
    inline for (all_types) |ty| if (comptime splitLayout(ty) != null) {
        // 4 rows x 16 blocks (the 4 fixture blocks over and over, shifted per row): 512 weights per row make every head region a multiple of 16 bytes.
        const be = ty.blockElems();
        const bb = ty.blockBytes();
        const data = @embedFile("fixtures/" ++ @tagName(ty) ++ ".bin");
        const n_rows = 4;
        const nb = 16;
        const in = nb * be;
        try std.testing.expect(isSplit(ty, in));
        var src: [n_rows * nb * 256]u8 = undefined;
        for (0..n_rows * nb) |i| @memcpy(src[i * bb ..][0..bb], data[((i + i / nb) % 4) * bb ..][0..bb]);
        var pk: [n_rows * (nb * 256 + 16)]u8 = undefined;
        packRows(ty, in, pk[0 .. n_rows * rowBytes(ty, in)], src[0 .. n_rows * nb * bb]);
        var ref: [4 * 256]f32 = undefined;
        try quants.dequantize(ty, data[0 .. 4 * bb], ref[0 .. 4 * be]);
        const w = mlx.mlx_array_new_data(&pk, &[_]c_int{ n_rows, @intCast(rowBytes(ty, in)) }, 2, .uint8);
        defer _ = mlx.mlx_array_free(w);
        const xv = try std.testing.allocator.alloc(f32, in);
        defer std.testing.allocator.free(xv);
        for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
        const x = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ 1, @intCast(in) }, 2, .float32);
        defer _ = mlx.mlx_array_free(x);
        const y = try linear(.{ .ty = ty }, x, w, s);
        defer _ = mlx.mlx_array_free(y);
        const got = try readF32(y, n_rows);
        for (0..n_rows) |r| {
            var want: f32 = 0;
            var scale: f32 = 0;
            for (0..nb) |blk| for (0..be) |j| {
                const term = xv[blk * be + j] * ref[((r * nb + blk + r) % 4) * be + j];
                want += term;
                scale += @abs(term);
            };
            if (@abs(want - got[r]) > 1e-5 * scale) {
                std.debug.print("{s} split row {d}: got {d} want {d}\n", .{ @tagName(ty), r, got[r], want });
                return error.TestUnexpectedResult;
            }
        }
        // The whole-weight decode reads the split rows too.
        const wd = try dequantize(ty, w, .float32, s);
        defer _ = mlx.mlx_array_free(wd);
        const dq = try readF32(wd, n_rows * in);
        for (0..n_rows) |r| for (0..nb) |blk| for (0..be) |j| {
            try std.testing.expectApproxEqRel(ref[((r * nb + blk + r) % 4) * be + j], dq[r * in + blk * be + j], 1e-5);
        };
    };
}

test "half precision activations keep their dtype and stay close to f32" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const w = try TestWeight.init(.iq4_nl);
    defer w.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    for ([_]mlx.mlx_dtype{ .float16, .bfloat16 }) |dtype| for ([_]usize{ 1, MATVEC_MAX_ROWS + 3 }) |m| {
        const xv = try std.testing.allocator.alloc(f32, m * w.in);
        defer std.testing.allocator.free(xv);
        for (xv) |*v| v.* = prng.random().float(f32) - 0.5;
        const x32 = mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ @intCast(m), @intCast(w.in) }, 2, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, dtype, s));

        const want = try linear(.{ .ty = .iq4_nl }, x32, w.arr, s);
        defer _ = mlx.mlx_array_free(want);
        const y = try linear(.{ .ty = .iq4_nl }, x, w.arr, s);
        defer _ = mlx.mlx_array_free(y);
        try std.testing.expectEqual(dtype, mlx.mlx_array_dtype(y));
        var y32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y32);
        try mlx.check(mlx.mlx_astype(&y32, y, .float32, s));
        // Error budget scales with the magnitude of the summed terms, not the (possibly cancelled) result.
        const got = try readF32(y32, m * 2);
        for (try readF32(want, m * 2), 0..) |r, i| {
            var scale: f32 = 0;
            for (xv[i / 2 * w.in ..][0..w.in], w.ref[i % 2 * w.in ..][0..w.in]) |a, b| scale += @abs(a * b);
            try std.testing.expectApproxEqAbs(r, got[i], 0.005 * scale);
        }
    };
}

test "shape and type mistakes are errors, not GPU faults" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const w = try TestWeight.init(.iq4_nl);
    defer w.deinit();
    try std.testing.expectError(error.BadShape, dequantize(.q4_k, w.arr, .float32, s));
    try std.testing.expectError(error.UnsupportedType, dequantize(.f16, w.arr, .float32, s));
    const x = mlx.mlx_array_new_data(&[_]f32{ 1, 2, 3 }, &[_]c_int{ 1, 3 }, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    try std.testing.expectError(error.BadShape, linear(.{ .ty = .iq4_nl }, x, w.arr, s));
}

test "sentinel round-trips the type and a tiled input matches a manual head swap" {
    if (!mlx.metalKernelsAvailable() or mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const w = try TestWeight.init(.q8_0); // in = 64
    defer w.deinit();

    const plain = try sentinel(.{ .ty = .q3_k }, s);
    defer _ = mlx.mlx_array_free(plain);
    try std.testing.expectEqual(GgmlType.q3_k, infoOf(w.arr, plain).?.ty);
    try std.testing.expect(infoOf(w.arr, plain).?.tiled == null);
    try std.testing.expect(infoOf(w.arr, mlx.mlx_array_new()) == null);
    try std.testing.expect(infoOf(plain, plain) == null);

    // nk=2, r=2, d=16: grouped heads [A0 A1 B0 B1] must reach the weight as [A0 B0 A1 B1].
    const tiled = try sentinel(.{ .ty = .q8_0, .tiled = .{ .nk = 2, .r = 2 } }, s);
    defer _ = mlx.mlx_array_free(tiled);
    const info = infoOf(w.arr, tiled).?;
    var xv: [64]f32 = undefined;
    for (&xv, 0..) |*v, i| v.* = @floatFromInt(i % 7);
    var xt: [64]f32 = undefined;
    for (0..4) |h| @memcpy(xt[16 * h ..][0..16], xv[16 * ([_]usize{ 0, 2, 1, 3 })[h] ..][0..16]);
    const x = mlx.mlx_array_new_data(&xv, &[_]c_int{ 1, 64 }, 2, .float32);
    defer _ = mlx.mlx_array_free(x);
    const x_manual = mlx.mlx_array_new_data(&xt, &[_]c_int{ 1, 64 }, 2, .float32);
    defer _ = mlx.mlx_array_free(x_manual);
    const got = try linear(info, x, w.arr, s);
    defer _ = mlx.mlx_array_free(got);
    const want = try linear(.{ .ty = .q8_0 }, x_manual, w.arr, s);
    defer _ = mlx.mlx_array_free(want);
    try std.testing.expectEqualSlices(f32, try readF32(want, 2), try readF32(got, 2));
}
