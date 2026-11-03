//! sdk/blocks/attention/decode_attn_quant.
//! Owns AttnDqCopy, attnDqUseNvfp4, attn_dq_engaged, attn_dq_nvfp4_engaged, and 11 more.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;
const Weights = @import("../../weights/resident.zig").Weights;
const attn256MaxDiff = @import("attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("attn256.zig").attn256RandBf16;

// ── Decode-only dense-attention requant (mlxfast native-affine class) ──
//
// A dense bf16/f16 attention projection reads its whole weight per decode
// token at the DRAM wall; on Laguna (bf16 attention over nvfp4 experts) the
// four projections are ~3 GB/token of a ~4 GB/token read. An INT8 group-32
// affine side copy halves that traffic. LOSSY BY DESIGN — a real
// requantization of weights the checkpoint ships dense, so this is a
// user-facing toggle (--decode-attn-quant / app Settings), never a sanctioned
// bit-identical class. Prefill, spec-verify and every other multi-token
// forward keep the authoritative dense weight (a quality anchor: KV built by
// prefill stays exact; only decode-step projections carry the perturbation).
pub const AttnDqCopy = struct {
    w: mlx.mlx_array, // packed codes, [out, in*bits/32] u32
    s: mlx.mlx_array,
    b: mlx.mlx_array, // null-ctx for nvfp4 (no biases by construction)
    // The dispatch reads the copy's OWN geometry — a tail layer's nvfp4 copy
    // and an early layer's int8 copy ride the same qmatmulBits call.
    bits: u32 = 8,
    group_size: u32 = 32,
    mode: QuantMode = .affine,
};

/// Which side-copy quantization a layer gets under --decode-attn-quant. Late
/// layers amplify quantization error far less (mlxfast measured ~15x lower on
/// layers >= 32 of 40), so the tail can afford real nvfp4-g16 (half the int8
/// bytes) while early layers keep int8-g32. Default tail = the last 20% of
/// layers (`attnDqNvfp4From`).
pub fn attnDqUseNvfp4(layer: u32, num_layers: u32) bool {
    if (num_layers == 0) return false;
    return layer >= attnDqNvfp4From(num_layers);
}

pub fn attnDqNvfp4From(num_layers: u32) u32 {
    return (num_layers * 4) / 5;
}

pub var attn_dq_engaged: bool = false;
pub var attn_dq_nvfp4_engaged: bool = false;

/// Default ON ( characterization on Laguna XS: decode forward
/// 13.285 -> 10.205 ms median, 3/3 paired boots both orders; greedy outputs
/// across six code/reasoning prompts reached identical answers with
/// wording-level divergence only). Flip together with the mirrored Swift
/// `ServerOptions` default.
pub const DECODE_ATTN_QUANT_DEFAULT = true;

/// Process-wide under `-Dunits` (sdk/util/shared.zig): one copy in the host.
const CliState = struct {
    /// Set by the CLI (`--decode-attn-quant` / `--no-decode-attn-quant`) before any load, so every serve path
    /// (headless included) sees one value.
    decode_attn_quant_flag: ?bool = null,
};
pub const cli_shared = @import("../../util/shared.zig").Shared(CliState, "decode_attn_quant_cli", .{});
pub inline fn cli() *CliState {
    return cli_shared.get();
}

/// Whether the user EXPLICITLY asked for --decode-attn-quant (CLI flag, app
/// toggle) as opposed to riding the silent default. Archs whose
/// characterization showed ANSWER MOVEMENT under the requant gate on this
/// form (dsv4 comp_in: one duplicated paragraph on 1/7 greedy prompts,
/// ); laguna's clean characterization keeps the plain default.
pub fn decodeAttnQuantExplicit() bool {
    if (decode_attn_quant_override) |v| return v; // test seam counts as explicit
    return cli().decode_attn_quant_flag orelse false;
}

pub fn decodeAttnQuantEnabled() bool {
    if (decode_attn_quant_override) |v| return v;
    return cli().decode_attn_quant_flag orelse DECODE_ATTN_QUANT_DEFAULT;
}

/// Whether `--decode-attn-quant` has anything to requantize: a text-trunk attention
/// projection shipped dense. Fully quantized packs decode exactly as without it.
pub fn weightsHaveDenseAttnProj(weights: *const Weights) bool {
    const projs = [_][]const u8{ ".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight", ".q_k_v_proj.weight", ".wq_du.weight", ".wk_dv.weight", ".wv_dv.weight", ".wo_ud.weight", ".wr_du.weight" };
    var it = weights.map.iterator();
    while (it.next()) |e| {
        const name = e.key_ptr.*;
        if (std.mem.indexOf(u8, name, "attn") == null) continue;
        if (std.mem.indexOf(u8, name, "vision") != null or std.mem.indexOf(u8, name, "visual") != null or std.mem.indexOf(u8, name, "audio") != null) continue;
        for (projs) |suffix| {
            if (!std.mem.endsWith(u8, name, suffix)) continue;
            var buf: [512]u8 = undefined;
            const scales = std.fmt.bufPrint(&buf, "{s}.scales", .{name[0 .. name.len - ".weight".len]}) catch break;
            if (decodeAttnQuantEligible(mlx.getShape(e.value_ptr.*), mlx.mlx_array_dtype(e.value_ptr.*), weights.map.contains(scales))) return true;
        }
    }
    return false;
}

/// Whether a loaded dense projection weight (pre-transposed [in, out]) can
/// carry an INT8-g32 side copy. The conditions are the quantized dispatch's
/// own: dense, a float dtype the requant helps, 2-D, in-dim in whole groups.
pub fn decodeAttnQuantEligible(w_shape: []const c_int, dtype: mlx.mlx_dtype, has_scales: bool) bool {
    if (has_scales) return false; // already quantized
    if (dtype != .bfloat16 and dtype != .float16) return false;
    if (w_shape.len != 2) return false;
    return @rem(w_shape[0], 32) == 0;
}

/// Quantize a dense pre-transposed [in, out] weight into a side copy
/// ([out, in] packed — served by `mlx_quantized_matmul` with transpose=true,
/// same as any checkpoint-quantized linear). `nvfp4` picks real nvfp4-g16
/// (half the int8 bytes, e4m3 scales, no biases) over the default INT8-g32
/// affine. Materialized eagerly so the build never rides a token's graph.

pub fn buildAttnDqCopy(s: mlx.mlx_stream, w: mlx.mlx_array, nvfp4: bool) !?AttnDqCopy {
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    const perm = [_]c_int{ 1, 0 };
    try mlx.check(mlx.mlx_transpose_axes(&wt, w, &perm, 2, s));
    var wt_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt_c);
    try mlx.check(mlx.mlx_contiguous(&wt_c, wt, false, s));
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    const gs: c_int = if (nvfp4) 16 else 32;
    const bits: c_int = if (nvfp4) 4 else 8;
    try mlx.check(mlx.mlx_quantize(
        &parts,
        wt_c,
        mlx.mlx_optional_int.some(gs),
        mlx.mlx_optional_int.some(bits),
        if (nvfp4) "nvfp4" else "affine",
        .{ .ctx = null },
        s,
    ));
    if (mlx.mlx_vector_array_size(parts) != @as(usize, if (nvfp4) 2 else 3)) return null;
    var copy = AttnDqCopy{
        .w = mlx.mlx_array_new(),
        .s = mlx.mlx_array_new(),
        .b = if (nvfp4) .{ .ctx = null } else mlx.mlx_array_new(),
        .bits = @intCast(bits),
        .group_size = @intCast(gs),
        .mode = if (nvfp4) .nvfp4 else .affine,
    };
    try mlx.check(mlx.mlx_vector_array_get(&copy.w, parts, 0));
    try mlx.check(mlx.mlx_vector_array_get(&copy.s, parts, 1));
    if (!nvfp4) try mlx.check(mlx.mlx_vector_array_get(&copy.b, parts, 2));
    _ = mlx.mlx_array_eval(copy.w);
    _ = mlx.mlx_array_eval(copy.s);
    if (copy.b.ctx != null) _ = mlx.mlx_array_eval(copy.b);
    return copy;
}

pub var decode_attn_quant_override: ?bool = null; // test seam

// ── tests ──

const qmatmulBits = @import("../linear/qmatmul.zig").qmatmulBits;

test "decode-attn-quant: eligibility is the dispatch's own conditions" {
    const t = std.testing;
    // Laguna o_proj shape, loaded pre-transposed [in, out].
    try t.expect(decodeAttnQuantEligible(&.{ 8192, 2048 }, .bfloat16, false));
    try t.expect(decodeAttnQuantEligible(&.{ 2048, 8192 }, .float16, false));
    // Already quantized: the side copy would be a re-quantization of a
    // requantization — decline.
    try t.expect(!decodeAttnQuantEligible(&.{ 8192, 2048 }, .bfloat16, true));
    // f32 dense weights stay put (VAE-class tensors; never attention).
    try t.expect(!decodeAttnQuantEligible(&.{ 8192, 2048 }, .float32, false));
    // Partial trailing group / non-matrix shapes decline.
    try t.expect(!decodeAttnQuantEligible(&.{ 8190, 2048 }, .bfloat16, false));
    try t.expect(!decodeAttnQuantEligible(&.{8192}, .bfloat16, false));
}

test "decode-attn-quant: ships ON (measured); the override seam reaches the gate" {
    const t = std.testing;
    // The shipped default is a single constant the Swift ServerOptions default
    // mirrors — flip both together.
    try t.expect(DECODE_ATTN_QUANT_DEFAULT == true);
    decode_attn_quant_override = false;
    defer decode_attn_quant_override = null;
    try t.expect(!decodeAttnQuantEnabled());
}

test "decode-attn-quant: int8 side copy tracks the dense projection (and is honestly lossy)" {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xDA9);
    const rnd = prng.random();

    // Dense pre-transposed [in, out] = [256, 192], decode row x = [1, 1, 256].
    const w_shape = [_]c_int{ 256, 192 };
    const w = try attn256RandBf16(rnd, &w_shape, s);
    defer _ = mlx.mlx_array_free(w);
    const x_shape = [_]c_int{ 1, 1, 256 };
    const x = try attn256RandBf16(rnd, &x_shape, s);
    defer _ = mlx.mlx_array_free(x);

    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_matmul(&dense, x, w, s));

    for ([_]bool{ false, true }) |nvfp4| {
        const copy = (try buildAttnDqCopy(s, w, nvfp4)) orelse return error.SideCopyDeclined;
        defer {
            _ = mlx.mlx_array_free(copy.w);
            _ = mlx.mlx_array_free(copy.s);
            if (copy.b.ctx != null) _ = mlx.mlx_array_free(copy.b);
        }
        try std.testing.expectEqual(@as(u32, if (nvfp4) 4 else 8), copy.bits);
        try std.testing.expectEqual(@as(u32, if (nvfp4) 16 else 32), copy.group_size);
        try std.testing.expect(copy.mode == (if (nvfp4) QuantMode.nvfp4 else QuantMode.affine));
        try std.testing.expect((copy.b.ctx == null) == nvfp4);
        const quant = try qmatmulBits(x, copy.w, copy.s, copy.b, copy.bits, copy.group_size, copy.mode, s);
        defer _ = mlx.mlx_array_free(quant);

        // A transposed or mis-grouped side copy produces garbage at ~output
        // magnitude; int8-g32 error on a 256-wide dot is a couple of percent
        // and nvfp4-g16 a few times that. |dense| here is ~4 (256
        // uniform(-.5,.5) products), so the bands are loose lossy bands that
        // still fail loudly on orientation bugs.
        const max_diff = try attn256MaxDiff(dense, quant, s);
        const band: f32 = if (nvfp4) 0.6 else 0.15;
        try std.testing.expect(max_diff < band);
        // And it must actually be the quantized path: bit-identical output
        // would mean the hook silently dispatched the dense weight.
        try std.testing.expect(max_diff > 0.0);
    }
}

test "decode-attn-quant: nvfp4 tail boundary is the last 20% of layers" {
    const t = std.testing;
    // Laguna XS: 40 layers -> nvfp4 from layer 32 (mlxfast's measured split).
    try t.expect(!attnDqUseNvfp4(31, 40));
    try t.expect(attnDqUseNvfp4(32, 40));
    try t.expect(attnDqUseNvfp4(39, 40));
    // Degenerate layer count never divides by zero or engages.
    try t.expect(!attnDqUseNvfp4(0, 0));
}

test "weightsHaveDenseAttnProj: decode-attn-quant applies only to a dense text attention projection" {
    const allocator = std.testing.allocator;
    const s = mlx.gpuStream();
    const put = struct {
        fn add(w: *Weights, name: []const u8, dtype: mlx.mlx_dtype, st: mlx.mlx_stream) !void {
            const shape = [_]c_int{ 64, 64 };
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&arr, &shape, 2, dtype, st));
            try w.map.put(try w.allocator.dupe(u8, name), arr);
        }
    }.add;

    var quantized = Weights.init(allocator);
    defer quantized.deinit();
    try put(&quantized, "model.layers.0.self_attn.q_proj.weight", .uint32, s);
    try put(&quantized, "model.layers.0.self_attn.q_proj.scales", .bfloat16, s);
    try put(&quantized, "vision_tower.layers.0.self_attn.q_proj.weight", .bfloat16, s);
    try std.testing.expect(!weightsHaveDenseAttnProj(&quantized));

    var dense = Weights.init(allocator);
    defer dense.deinit();
    try put(&dense, "model.layers.3.self_attn.o_proj.weight", .bfloat16, s);
    try std.testing.expect(weightsHaveDenseAttnProj(&dense));

    var inkling = Weights.init(allocator);
    defer inkling.deinit();
    try put(&inkling, "model.layers.0.attn.wo_ud.weight", .bfloat16, s);
    try std.testing.expect(weightsHaveDenseAttnProj(&inkling));
}
