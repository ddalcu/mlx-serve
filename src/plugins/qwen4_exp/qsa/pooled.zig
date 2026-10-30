//! The pooled indexer keys: their rope (`QsaPooledRope`, the pool-rope kernel) and the pointer-keyed cache invalidated
//! by an atomic mark.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const mrope = @import("../../../sdk/vision/mrope.zig");
const log = @import("../../../sdk/util/log.zig");
const ForwardCtx = @import("../../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../../engines/mlx/kv/cache.zig").testing;
const scalarOf = @import("../../../engines/mlx/load/weights.zig").scalarOf;
const SSMCacheEntry = @import("../../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const Trunk = @import("../trunk.zig").Trunk;
const materializedOwnedCopy = @import("../../../mlx/array.zig").materializedOwnedCopy;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const testRandUniformBf16 = @import("../../../sdk/testing/parity.zig").testRandUniformBf16;
const qsaProbeAllFinite = @import("../../../sdk/blocks/attention/sparse_block/probe.zig").qsaProbeAllFinite;
const ssmFreeQsaState = @import("history.zig").ssmFreeQsaState;
const tieBias = @import("../../../sdk/blocks/attention/sparse_block/select.zig").tieBias;

/// The QSA pooled-block RoPE tables, shared by every full-attention layer and across
/// forwards while the key holds. The key is the block position (`base`, `step`, `n`, dtype)
/// plus, on the M-RoPE arm, the position table compared by pointer and length; the pointer
/// is never dereferenced. The owner slot frees that table only after
/// `markQsaPooledRopeStale`, so a same-address table cannot hit. Connection threads write
/// only the atomic `stale`; the inference thread frees.
pub const QsaPooledRope = struct {
    base: c_int = 0,
    step: c_int = 0,
    n: c_int = 0,
    dtype: mlx.mlx_dtype = .float32,
    mrope: bool = false,
    mrope_pos: ?[]const i32 = null,
    mrope_total: usize = 0,
    mrope_delta: i32 = 0,
    cos: mlx.mlx_array = .{ .ctx = null },
    sin: mlx.mlx_array = .{ .ctx = null },
    /// Rebuild counter.
    builds: usize = 0,
    stale: std.atomic.Value(bool) = .init(false),

    pub fn deinit(self: *QsaPooledRope) void {
        if (self.cos.ctx != null) _ = mlx.mlx_array_free(self.cos);
        if (self.sin.ctx != null) _ = mlx.mlx_array_free(self.sin);
        self.cos = .{ .ctx = null };
        self.sin = .{ .ctx = null };
        self.mrope_pos = null;
        self.stale.store(false, .monotonic);
    }
};

pub fn qsaPooledTableEq(a: ?[]const i32, b: ?[]const i32) bool {
    const aa = a orelse return b == null;
    const bb = b orelse return false;
    return aa.ptr == bb.ptr and aa.len == bb.len;
}

/// Does every complete-block visibility check pass for every query row of a call starting at
/// cache position `offset`? Block `b` is complete at `p` iff `b*ratio + ratio - 1 <= p`, so
/// the whole sheet is all-true iff `offset >= nb*ratio-1`. Always true at decode width.
pub fn qsaAllBlocksVisible(offset: c_int, nb: c_int, ratio: c_int) bool {
    return offset >= nb * ratio - 1;
}

/// The nb-keyed constants of the QSA block selection, built once per (nb, ratio) instead of
/// once per full-attention layer. A key mismatch rebuilds.
pub const QsaBlockConsts = struct {
    nb: c_int = 0,
    ratio: c_int = 0,
    /// int32 `[1, nb]`: the last cache position of each complete block.
    blk_end2: mlx.mlx_array = .{ .ctx = null },
    /// f32 `[nb]`: `b * 1e-7`, the index-descending bias reproducing torch.topk's lower-index-wins.
    tie_bias: mlx.mlx_array = .{ .ctx = null },
    /// Rebuild counter.
    builds: usize = 0,

    pub fn deinit(self: *QsaBlockConsts) void {
        if (self.blk_end2.ctx != null) _ = mlx.mlx_array_free(self.blk_end2);
        if (self.tie_bias.ctx != null) _ = mlx.mlx_array_free(self.tie_bias);
        self.blk_end2 = .{ .ctx = null };
        self.tie_bias = .{ .ctx = null };
        self.nb = 0;
        self.ratio = 0;
    }

    /// After this returns, `blk_end2`/`tie_bias` are borrowed handles valid until the next `ensure` with a different key.
    pub fn ensure(self: *QsaBlockConsts, s: mlx.mlx_stream, nb: c_int, ratio: c_int) !void {
        if (self.blk_end2.ctx != null and self.nb == nb and self.ratio == ratio) return;
        self.deinit();
        errdefer self.deinit();
        var blk_end = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(blk_end);
        try mlx.check(mlx.mlx_arange(&blk_end, @floatFromInt(ratio - 1), @floatFromInt(nb * ratio), @floatFromInt(ratio), .int32, s));
        self.blk_end2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&self.blk_end2, blk_end, &[_]c_int{ 1, nb }, 2, s));
        self.tie_bias = try tieBias(s, nb);
        self.nb = nb;
        self.ratio = ratio;
        self.builds += 1;
    }
};

pub fn resetQsaPooledRope(self: *Trunk) void {
    self.qsa_pooled_rope.deinit();
}

pub fn markQsaPooledRopeStale(self: *Trunk) void {
    self.qsa_pooled_rope.stale.store(true, .release);
}

// ── Fused QSA pooled-key upkeep ──
pub var qsa_idx_rope_fused_override: ?bool = null;

/// The test seam's false runs the composed pooled-key chain.
pub fn qsaIdxRopeFusedEnabled() bool {
    return qsa_idx_rope_fused_override orelse true;
}

// Mirrors each rounding of the composed chain it replaces, so it is bit-identical:
//   mean  = f32 sum in row order r = 0..R-1, times f32(1/R) (MLX
//           NumberOfElements), then bf16. That order is MLX's only while
//           R <= QSA_POOL_SERIAL_MAX_R (see there);
//   norm  = `rms_single_row` at D = 128 (32 lanes x 4 reads, simd_sum, precise
//           rsqrt, `w * T(x * inv)`);
//   rope  = the bf16 cos/sin table and bf16(bf16(x*cos) + bf16(rot*sin)), with
//           each product and the sum rounded as MLX's separate binary kernels do.
// One simdgroup per pooled row; lane l owns dims 4l..4l+3, and rotate_half's
// partner (d +- HALF) is HALF/4 lanes away at the same slot. The block count
// `nblk` varies per call (1 at decode, up to 512 per prefill chunk), so it is a
// scalar INPUT; the templates (T, R, RD, HALF) are fixed per model and name the
// one pipeline MLX compiles.
pub const QSA_POOL_ROPE_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_pool_rope.metal");

/// Largest block ratio whose f32 block sum MLX folds in row order 0..R-1, the
/// order the kernel uses. The mean over axis 2 of `[B, nb, R, 128]` is a strided
/// reduce; below 32 rows `strided_reduce_general_dispatch` picks
/// `col_reduce_small` with threadgroup_y = min(8, R). Up to 8 rows each y-thread
/// owns ONE row and lid.y 0 folds them 0..R-1 (R == 1 is a copy). At 9..31 each
/// thread sums rows y, y+8, ... first, and at 32+ the looped/2pass kernels
/// reduce in simdgroup trees, so a sum that cancels can differ in f32.
const QSA_POOL_SERIAL_MAX_R: c_int = 8;

var qsa_pool_rope_kernel: ?mlx.mlx_fast_metal_kernel = null;
var qsa_pool_rope_engaged: bool = false;
const QsaPoolRopeCfgKey = struct { b: c_int, nb: c_int, r: c_int, rd: c_int };
/// Int template values handed to MLX so far (T is always bf16); each distinct
/// tuple is a pipeline compile.
const QsaPoolRopeTpl = [4]c_int;
var qsa_pool_rope_tpls: [64]QsaPoolRopeTpl = undefined;
var qsa_pool_rope_tpl_count: usize = 0;

fn qsaPoolRopeNoteTemplate(t: QsaPoolRopeTpl) void {
    const kept = @min(qsa_pool_rope_tpl_count, qsa_pool_rope_tpls.len);
    for (qsa_pool_rope_tpls[0..kept]) |seen| {
        if (std.meta.eql(seen, t)) return;
    }
    // Past the table, every unseen tuple still counts: an overcount, never a miss.
    if (kept < qsa_pool_rope_tpls.len) qsa_pool_rope_tpls[kept] = t;
    qsa_pool_rope_tpl_count += 1;
}
var qsa_pool_rope_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var qsa_pool_rope_cfg_key: QsaPoolRopeCfgKey = std.mem.zeroes(QsaPoolRopeCfgKey);

fn getQsaPoolRopeKernel() !mlx.mlx_fast_metal_kernel {
    if (qsa_pool_rope_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "kb", "norm_w", "cosv", "sinv", "eps", "inv_r", "nblk" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("mlxserve_qsa_pool_rope", in_vec, out_vec, QSA_POOL_ROPE_SOURCE, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_pool_rope_kernel = kernel;
    return kernel;
}

/// Pooled block keys in one launch: f32 block mean → bf16 → rms_norm(w) →
/// partial RoPE with the bf16 cos/sin table `[nb, rope_dims]`. Null → the
/// caller runs the composed chain (`qsaPoolNorm` + `qsaPooledRopeComposed`).
pub fn qsaPoolNormRopeFused(
    s: mlx.mlx_stream,
    kb4: mlx.mlx_array, // [B, nb, R, D] bf16
    norm_w: mlx.mlx_array, // [D] bf16
    eps: f32,
    cosv: mlx.mlx_array, // [nb, rope_dims] bf16
    sinv: mlx.mlx_array,
    rope_dims: c_int,
) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const ksh = mlx.getShape(kb4);
    const wsh = mlx.getShape(norm_w);
    const csh = mlx.getShape(cosv);
    if (ksh.len != 4 or ksh[3] != 128 or ksh[1] < 1 or ksh[2] < 1 or ksh[2] > QSA_POOL_SERIAL_MAX_R) return null;
    if (wsh.len != 1 or wsh[0] != 128) return null;
    // rotate_half's partner must sit whole lanes away: HALF a multiple of 4.
    if (rope_dims <= 0 or rope_dims > 128 or @mod(rope_dims, 8) != 0) return null;
    if (csh.len != 2 or csh[0] != ksh[1] or csh[1] != rope_dims) return null;
    if (!std.mem.eql(c_int, csh, mlx.getShape(sinv))) return null;
    inline for (.{ kb4, norm_w, cosv, sinv }) |arr| {
        if (mlx.mlx_array_dtype(arr) != .bfloat16) return null;
    }
    const key = QsaPoolRopeCfgKey{ .b = ksh[0], .nb = ksh[1], .r = ksh[2], .rd = rope_dims };
    if (qsa_pool_rope_cfg == null or !std.meta.eql(qsa_pool_rope_cfg_key, key)) {
        if (qsa_pool_rope_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        qsa_pool_rope_cfg = null;
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const out_shape = [_]c_int{ ksh[0], ksh[1], 128 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, ksh[0] * ksh[1], 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        const tpl_names = [_][*:0]const u8{ "R", "RD", "HALF" };
        const tpl_vals = [_]c_int{ ksh[2], rope_dims, @divExact(rope_dims, 2) };
        var noted: QsaPoolRopeTpl = .{ 0, 0, 0, 0 };
        for (tpl_names, tpl_vals, 0..) |n, v, i| {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, n, v));
            noted[i] = v;
        }
        qsaPoolRopeNoteTemplate(noted);
        qsa_pool_rope_cfg = config;
        qsa_pool_rope_cfg_key = key;
    }
    const kernel = try getQsaPoolRopeKernel();
    const eps_a = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_a);
    // MLX's mean multiplies by NumberOfElements: f32(1.0 / n) computed in f64.
    const inv_r = mlx.mlx_array_new_float(@floatCast(1.0 / @as(f64, @floatFromInt(ksh[2]))));
    defer _ = mlx.mlx_array_free(inv_r);
    const nblk = mlx.mlx_array_new_int(ksh[1]);
    defer _ = mlx.mlx_array_free(nblk);
    const inputs_arr = [_]mlx.mlx_array{ kb4, norm_w, cosv, sinv, eps_a, inv_r, nblk };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, qsa_pool_rope_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!qsa_pool_rope_engaged) {
        qsa_pool_rope_engaged = true;
        log.info("[qsa] fused pooled-key upkeep engaged (blocks={d} ratio={d} rope={d})\n", .{ ksh[1], ksh[2], rope_dims });
    }
    return out;
}

// ── tests ──

/// Elements whose f32 bit patterns differ (bf16 → f32 is exact, so this is bf16 bit identity, ±0 and NaN included).
fn qsaIdxBitMismatches(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !usize {
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    var b_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&b_c, b, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    var bad: usize = 0;
    for (0..n) |i| {
        if (@as(u32, @bitCast(ad[i])) != @as(u32, @bitCast(bd[i]))) bad += 1;
    }
    return bad;
}

/// Raw index keys `[B, nb, ratio, 128]` bf16 with magnitudes spread over 10^lo_exp..10^hi_exp.
fn qsaIdxTestKeys(rnd: std.Random, b: c_int, nb: c_int, ratio: c_int, lo_exp: f32, hi_exp: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = [_]c_int{ b, nb, ratio, 128 };
    const n: usize = @intCast(b * nb * ratio * 128);
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    for (data) |*x| {
        const mag = std.math.pow(f32, 10.0, lo_exp + (hi_exp - lo_exp) * rnd.float(f32));
        x.* = if (rnd.boolean()) mag else -mag;
    }
    const f = mlx.mlx_array_new_data(data.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

/// Fused vs composed pooled keys for one case; returns the bit-mismatch count (fails if the kernel declines).
fn qsaIdxRopeCase(xfm: *Trunk, kb4: mlx.mlx_array, norm_w: mlx.mlx_array, base: c_int, table_scale: ?f32) !usize {
    const s = xfm.s;
    const rope_dims: c_int = 64;
    const nb = mlx.getShape(kb4)[1];
    const ratio: f32 = @floatFromInt(mlx.getShape(kb4)[2]);
    var cs = try xfm.ropeCosSinFromFreqs(rope_dims, try xfm.ropeInvFreq(rope_dims, xfm.config.rope_theta), @floatFromInt(base), ratio, nb, .bfloat16);
    defer _ = mlx.mlx_array_free(cs.cos);
    defer _ = mlx.mlx_array_free(cs.sin);
    if (table_scale) |m| {
        // YaRN tables arrive pre-multiplied by mscale in bf16; the kernel must read them as-is.
        const ms = try scalarOf(m, .bfloat16, s);
        defer _ = mlx.mlx_array_free(ms);
        var c2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&c2, cs.cos, ms, s));
        _ = mlx.mlx_array_free(cs.cos);
        cs.cos = c2;
        var s2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&s2, cs.sin, ms, s));
        _ = mlx.mlx_array_free(cs.sin);
        cs.sin = s2;
    }
    const pn = try xfm.qsaPoolNorm(kb4, norm_w);
    defer _ = mlx.mlx_array_free(pn);
    const want = try xfm.qsaPooledRopeComposed(pn, rope_dims, cs.cos, cs.sin);
    defer _ = mlx.mlx_array_free(want);
    const got = (try qsaPoolNormRopeFused(s, kb4, norm_w, xfm.config.rms_norm_eps, cs.cos, cs.sin, rope_dims)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(got);
    try std.testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    return qsaIdxBitMismatches(want, got, s);
}

fn qsaIdxRopeTestXfm(s: mlx.mlx_stream) Trunk {
    var xfm: Trunk = undefined;
    xfm.s = s;
    xfm.allocator = std.testing.allocator;
    xfm.config = .{};
    xfm.config.rms_norm_eps = 1e-6;
    xfm.config.rope_theta = 10_000_000.0;
    return xfm;
}

// Bar: the fused pooled-key kernel is bit-identical to the composed chain at every
// block count, position, table scale and key dynamic range the indexer can hand it.
test "qsa pooled keys: fused mean+norm+rope kernel is bit-identical to the composed chain" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var xfm = qsaIdxRopeTestXfm(s);
    var prng = std.Random.DefaultPrng.init(0x1D8_0BE5);
    const rnd = prng.random();
    const norm_w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(norm_w);
    // { batch, blocks, base position, lo exp, hi exp }
    const cases = [_][5]f32{
        .{ 1, 1, 0, -1, 0.5 }, // first block: cos 1, sin 0
        .{ 1, 1, 2052, -1, 0.5 }, // verify S=4 just past the budget
        .{ 1, 1, 32772, -1, 0.5 }, // past 32k
        .{ 1, 1, 131080, -1, 0.5 }, // past 128k
        .{ 1, 1, 1_000_000, -1, 0.5 }, // 1M context
        .{ 1, 2, 16000, -1, 0.5 }, // S=8 verify: two blocks
        .{ 1, 512, 0, -1, 0.5 }, // 2048-token prefill chunk
        .{ 1, 37, 8196, -3, 4 }, // wide range: f32 block sums round
        .{ 2, 5, 4096, -1, 0.5 }, // batch 2
    };
    for (cases) |c| {
        const kb4 = try qsaIdxTestKeys(rnd, @intFromFloat(c[0]), @intFromFloat(c[1]), 4, c[3], c[4], s);
        defer _ = mlx.mlx_array_free(kb4);
        const bad = try qsaIdxRopeCase(&xfm, kb4, norm_w, @intFromFloat(c[2]), null);
        if (bad != 0) std.debug.print("qsa idx rope case {any}: {d} mismatches\n", .{ c, bad });
        try std.testing.expectEqual(@as(usize, 0), bad);
    }
    // YaRN-scaled table.
    const kb4 = try qsaIdxTestKeys(rnd, 1, 3, 4, -1, 0.5, s);
    defer _ = mlx.mlx_array_free(kb4);
    try std.testing.expectEqual(@as(usize, 0), try qsaIdxRopeCase(&xfm, kb4, norm_w, 65540, 1.2071));
}

// Bar: a strided kb4 view (the live k_raw slice of the [S, (n+1)*128] indexer
// projection, and the leftover+chunk concat) gives the same bits as a packed one.
test "qsa pooled keys: fused kernel reads a strided raw-key view bit-identically" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var xfm = qsaIdxRopeTestXfm(s);
    var prng = std.Random.DefaultPrng.init(0x5171_DE);
    const rnd = prng.random();
    const norm_w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const qk = try attn256RandBf16(rnd, &[_]c_int{ 1, 12, 5 * 128 }, s);
    defer _ = mlx.mlx_array_free(qk);
    var k_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_raw);
    try mlx.check(mlx.mlx_slice(&k_raw, qk, &[_]c_int{ 0, 0, 4 * 128 }, 3, &[_]c_int{ 1, 12, 5 * 128 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var kb4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kb4);
    try mlx.check(mlx.mlx_reshape(&kb4, k_raw, &[_]c_int{ 1, 3, 4, 128 }, 4, s));
    try std.testing.expectEqual(@as(usize, 0), try qsaIdxRopeCase(&xfm, kb4, norm_w, 2048, null));
}

// Bar: shapes the kernel does not mirror decline (null) so the caller keeps the chain.
test "qsa pooled keys: fused kernel declines non-bf16 weights and non-128 heads" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var xfm = qsaIdxRopeTestXfm(s);
    var prng = std.Random.DefaultPrng.init(0xDEC1);
    const rnd = prng.random();
    const cs = try xfm.ropeCosSinFromFreqs(64, try xfm.ropeInvFreq(64, xfm.config.rope_theta), 0, 4, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(cs.cos);
    defer _ = mlx.mlx_array_free(cs.sin);
    const kb4 = try qsaIdxTestKeys(rnd, 1, 1, 4, -1, 0.5, s);
    defer _ = mlx.mlx_array_free(kb4);
    const w_bf = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(w_bf);
    var w32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w32);
    try mlx.check(mlx.mlx_astype(&w32, w_bf, .float32, s));
    try std.testing.expect((try qsaPoolNormRopeFused(s, kb4, w32, 1e-6, cs.cos, cs.sin, 64)) == null);
    const kb64 = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, 4, 64 }, s);
    defer _ = mlx.mlx_array_free(kb64);
    const w64 = try testRandUniformBf16(rnd, &[_]c_int{64}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(w64);
    try std.testing.expect((try qsaPoolNormRopeFused(s, kb64, w64, 1e-6, cs.cos, cs.sin, 64)) == null);
    // Sanity: the supported shape does engage.
    const got = (try qsaPoolNormRopeFused(s, kb4, w_bf, 1e-6, cs.cos, cs.sin, 64)) orelse return error.FusedDeclined;
    _ = mlx.mlx_array_free(got);
}

/// Keys `[1, nb, ratio, 128]` whose f32 block sums depend on the add order: even
/// columns hold 2^24, -2^24 (half a block later) and 1s, rotated per column and
/// block, so row-order and tree-order sums disagree; odd columns are small noise.
fn qsaIdxCancelKeys(rnd: std.Random, nb: c_int, ratio: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const r: usize = @intCast(ratio);
    const n: usize = @as(usize, @intCast(nb)) * r * 128;
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    for (0..@intCast(nb)) |b| {
        for (0..128) |c| {
            for (0..r) |row| {
                const at = (b * r + row) * 128 + c;
                if (c % 2 == 1) {
                    data[at] = rnd.float(f32) - 0.5;
                    continue;
                }
                const p = (row + c / 2 + b) % r;
                data[at] = if (r > 1 and p == 0) 16777216.0 else if (r > 1 and p == r / 2) -16777216.0 else 1.0;
            }
        }
    }
    const shape = [_]c_int{ 1, nb, ratio, 128 };
    const f = mlx.mlx_array_new_data(data.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f, .bfloat16, s));
    return out;
}

// Bar: at every ratio the kernel serves (1..8), order-sensitive block sums still
// match the composed mean bit for bit; past 8, MLX's reduce order differs, so
// the kernel declines and the composed chain runs.
test "qsa pooled keys: fused kernel matches order-sensitive sums at ratios 1..8, declines past 8" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var xfm = qsaIdxRopeTestXfm(s);
    var prng = std.Random.DefaultPrng.init(0xCA2CE1);
    const rnd = prng.random();
    const norm_w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(norm_w);
    for ([_]c_int{ 1, 2, 3, 4, 8 }) |ratio| {
        const kb4 = try qsaIdxCancelKeys(rnd, 9, ratio, s);
        defer _ = mlx.mlx_array_free(kb4);
        const bad = try qsaIdxRopeCase(&xfm, kb4, norm_w, 4096, null);
        if (bad != 0) std.debug.print("qsa idx cancel ratio {d}: {d} mismatches\n", .{ ratio, bad });
        try std.testing.expectEqual(@as(usize, 0), bad);
        const rk = try qsaIdxTestKeys(rnd, 1, 5, ratio, -3, 4, s);
        defer _ = mlx.mlx_array_free(rk);
        try std.testing.expectEqual(@as(usize, 0), try qsaIdxRopeCase(&xfm, rk, norm_w, 8192, null));
    }
    for ([_]c_int{ 9, 16, 32 }) |ratio| {
        const kb4 = try qsaIdxCancelKeys(rnd, 3, ratio, s);
        defer _ = mlx.mlx_array_free(kb4);
        const cs = try xfm.ropeCosSinFromFreqs(64, try xfm.ropeInvFreq(64, xfm.config.rope_theta), 4096, @floatFromInt(ratio), 3, .bfloat16);
        defer _ = mlx.mlx_array_free(cs.cos);
        defer _ = mlx.mlx_array_free(cs.sin);
        try std.testing.expect((try qsaPoolNormRopeFused(s, kb4, norm_w, 1e-6, cs.cos, cs.sin, 64)) == null);
        // The composed chain the caller falls back to still serves the shape.
        const pn = try xfm.qsaPoolNorm(kb4, norm_w);
        defer _ = mlx.mlx_array_free(pn);
        const want = try xfm.qsaPooledRopeComposed(pn, 64, cs.cos, cs.sin);
        defer _ = mlx.mlx_array_free(want);
        try std.testing.expect(qsaProbeAllFinite(s, want));
        try std.testing.expectEqualSlices(c_int, &[_]c_int{ 1, 3, 128 }, mlx.getShape(want));
    }
}

// Bar: the block count is a runtime input, so decode (1 block), verify (2) and a
// prefill chunk (512) share ONE pipeline and stay bit-identical.
test "qsa pooled keys: block counts 1, 2, 7, 512 share one fused pipeline" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var xfm = qsaIdxRopeTestXfm(s);
    var prng = std.Random.DefaultPrng.init(0x919E);
    const rnd = prng.random();
    const norm_w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, s);
    defer _ = mlx.mlx_array_free(norm_w);
    var after_first: ?usize = null;
    for ([_]c_int{ 1, 2, 7, 512, 1 }) |nb| {
        const kb4 = try qsaIdxTestKeys(rnd, 1, nb, 4, -1, 0.5, s);
        defer _ = mlx.mlx_array_free(kb4);
        try std.testing.expectEqual(@as(usize, 0), try qsaIdxRopeCase(&xfm, kb4, norm_w, 4096, null));
        if (after_first == null) after_first = qsaPoolRopePipelineCount();
        try std.testing.expectEqual(after_first.?, qsaPoolRopePipelineCount());
    }
}

// Bar: a CPU stream never reaches the Metal kernel; the caller keeps the chain.
test "qsa pooled keys: fused kernel declines a non-GPU stream" {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try std.testing.expect(!mlx.streamIsGpu(cpu));
    var xfm = qsaIdxRopeTestXfm(cpu);
    var prng = std.Random.DefaultPrng.init(0xC9C);
    const rnd = prng.random();
    const norm_w = try testRandUniformBf16(rnd, &[_]c_int{128}, 0.25, 1.75, cpu);
    defer _ = mlx.mlx_array_free(norm_w);
    const kb4 = try qsaIdxTestKeys(rnd, 1, 2, 4, -1, 0.5, cpu);
    defer _ = mlx.mlx_array_free(kb4);
    const cs = try xfm.ropeCosSinFromFreqs(64, try xfm.ropeInvFreq(64, xfm.config.rope_theta), 0, 4, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(cs.cos);
    defer _ = mlx.mlx_array_free(cs.sin);
    const before = qsaPoolRopePipelineCount();
    try std.testing.expect((try qsaPoolNormRopeFused(cpu, kb4, norm_w, 1e-6, cs.cos, cs.sin, 64)) == null);
    try std.testing.expectEqual(before, qsaPoolRopePipelineCount());
}

/// Distinct kernel pipelines (template tuples) the fused pooled-key helper has requested.
pub fn qsaPoolRopePipelineCount() usize {
    return qsa_pool_rope_tpl_count;
}

test "qsa pooled bank: capacity append equals the concat, and the f32 score operand is rebuilt only when a block completes" {
    // The bf16 bank must stay byte-for-byte the old `concatenate`; 12 layers of one forward reuse one f32 build.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();

    var t: Trunk = undefined;

    t.s = s;
    t.allocator = testing.allocator;
    t.qsa_score_bank_builds = 0;

    const hd: c_int = 16;
    var entry: SSMCacheEntry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    defer ssmFreeQsaState(&entry);

    var prng = std.Random.DefaultPrng.init(0x9A17_3D02);
    const rnd = prng.random();
    var reference = mlx.mlx_array{ .ctx = null };
    defer if (reference.ctx != null) {
        _ = mlx.mlx_array_free(reference);
    };
    var nb: c_int = 0;
    var step_i: usize = 0;
    while (step_i < 400) : (step_i += 1) {
        const add: c_int = if (step_i == 0) 5 else 1;
        const new3 = try attn256RandBf16(rnd, &[_]c_int{ 1, add, hd }, s);
        defer _ = mlx.mlx_array_free(new3);
        var next_ref = mlx.mlx_array_new();
        if (reference.ctx != null) {
            const parts = [_]mlx.mlx_array{ reference, new3 };
            const vec = mlx.mlx_vector_array_new_data(&parts, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&next_ref, vec, 1, s));
            _ = mlx.mlx_array_free(reference);
        } else {
            _ = mlx.mlx_array_free(next_ref);
            next_ref = try materializedOwnedCopy(s, new3);
        }
        reference = next_ref;

        try t.qsaAppendPooled(&entry, new3, nb);
        nb += add;
        try testing.expectEqual(nb, mlx.getShape(entry.qsa_pooled)[1]);
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(entry.qsa_pooled, reference, s));
    }
    try testing.expect(mlx.getShape(entry.qsa_pooled_buf)[1] > nb);

    const before = t.qsa_score_bank_builds;
    const bank = try t.qsaScoreBank(&entry, 1, nb, hd);
    try testing.expectEqual(before + 1, t.qsa_score_bank_builds);
    for (0..11) |_| {
        const again = try t.qsaScoreBank(&entry, 1, nb, hd);
        try testing.expect(again.ctx == bank.ctx);
    }
    try testing.expectEqual(before + 1, t.qsa_score_bank_builds);
    try testing.expectEqual(@as(c_int, hd), mlx.getShape(bank)[2]);
    try testing.expectEqual(nb, mlx.getShape(bank)[3]);

    {
        var k_rope = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_rope);
        try mlx.check(mlx.mlx_reshape(&k_rope, entry.qsa_pooled, &[_]c_int{ 1, 1, nb, hd }, 4, s));
        var kt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kt);
        try mlx.check(mlx.mlx_transpose_axes(&kt, k_rope, &[_]c_int{ 0, 1, 3, 2 }, 4, s));
        var k32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k32);
        try mlx.check(mlx.mlx_astype(&k32, kt, .float32, s));
        try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(bank, k32, s));
    }

    const one = try attn256RandBf16(rnd, &[_]c_int{ 1, 1, hd }, s);
    defer _ = mlx.mlx_array_free(one);
    try t.qsaAppendPooled(&entry, one, nb);
    _ = try t.qsaScoreBank(&entry, 1, nb + 1, hd);
    try testing.expectEqual(before + 2, t.qsa_score_bank_builds);
    try testing.expectEqual(nb + 1, mlx.getShape(entry.qsa_score_bank)[3]);
}

test "qsa pooled rope: one cos/sin build per forward, not one per full-attention layer" {
    // The tables are a property of the forward, not of the layer.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();

    var t: Trunk = undefined;

    t.s = s;
    t.allocator = testing.allocator;
    t.config = .{};
    t.config.head_dim = 256;
    t.config.partial_rotary_factor = 0.5;
    t.config.rope_theta = 10_000_000.0;
    t.config.mrope_section = .{ 11, 11, 10 };
    t.rope_freqs_yarn = null;
    t.yarn_inv_freq = null;
    t.fwd_gen = 0;
    t.qsa_pooled_rope = .{};
    defer t.qsa_pooled_rope.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    var seq_off: usize = 0;
    var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &seq_off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };

    const rope_dims: c_int = 128;
    const base: c_int = 96;
    const step: c_int = 4; // = indexer_compress_ratio
    const n: c_int = 5;

    const cs = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 1), t.qsa_pooled_rope.builds);
    for (0..11) |_| {
        const again = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
        try testing.expect(again.cos.ctx == cs.cos.ctx and again.sin.ctx == cs.sin.ctx);
    }
    try testing.expectEqual(@as(usize, 1), t.qsa_pooled_rope.builds);

    var data: [@as(usize, @intCast(n)) * 128]f32 = undefined;
    for (&data, 0..) |*d, i| d.* = @floatFromInt((i % 37) + 1);
    const xshape = [_]c_int{ 1, 1, n, 128 };
    const xf = mlx.mlx_array_new_data(&data, &xshape, 4, .float32);
    defer _ = mlx.mlx_array_free(xf);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, xf, .bfloat16, s));
    const hoisted = try t.ropeApplyCosSin(x, rope_dims, cs.cos, cs.sin);
    defer _ = mlx.mlx_array_free(hoisted);
    const per_layer = try t.ropeAtPositions(x, rope_dims, t.config.rope_theta, @floatFromInt(base), @floatFromInt(step), n);
    defer _ = mlx.mlx_array_free(per_layer);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(hoisted, per_layer, s));

    // A new forward at the same position hits: the forward counter is not in the key.
    t.fwd_gen += 1;
    _ = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 1), t.qsa_pooled_rope.builds);
    _ = try t.qsaPooledCosSin(&ctx, rope_dims, base + step, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 2), t.qsa_pooled_rope.builds);

    // The M-RoPE arm is a different table at the same (base, step, n).
    const pos = try testing.allocator.alloc(i32, 3 * 256);
    defer testing.allocator.free(pos);
    for (pos, 0..) |*p, i| p.* = @intCast(i % 256);
    ctx.mrope_pos = pos;
    ctx.mrope_total = 256;
    const m = try t.qsaPooledCosSin(&ctx, rope_dims, base + step, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 3), t.qsa_pooled_rope.builds);
    const direct = try t.mropeCosSinAt(.{ .pos = pos, .total = 256, .delta = 0 }, @intCast(base + step), @intCast(step), @intCast(n), .bfloat16);
    defer _ = mlx.mlx_array_free(direct.cos);
    defer _ = mlx.mlx_array_free(direct.sin);
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.cos, direct.cos, s));
    try testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(m.sin, direct.sin, s));
    for (0..11) |_| _ = try t.qsaPooledCosSin(&ctx, rope_dims, base + step, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 3), t.qsa_pooled_rope.builds);
}

test "qsa pooled rope: a mark from another thread is observed by the next lookup" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();

    var t: Trunk = undefined;

    t.s = s;
    t.allocator = testing.allocator;
    t.config = .{};
    t.config.head_dim = 256;
    t.config.partial_rotary_factor = 0.5;
    t.config.rope_theta = 10_000_000.0;
    t.config.mrope_section = .{ 11, 11, 10 };
    t.rope_freqs_yarn = null;
    t.yarn_inv_freq = null;
    t.fwd_gen = 0;
    t.qsa_pooled_rope = .{};
    defer t.qsa_pooled_rope.deinit();

    var cache = try KVCache.init(testing.allocator, 1);
    defer cache.deinit();
    var seq_off: usize = 0;
    var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &seq_off, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null };

    const rope_dims: c_int = 128;
    const base: c_int = 96;
    const step: c_int = 4;
    const n: c_int = 5;

    _ = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 1), t.qsa_pooled_rope.builds);

    const Worker = struct {
        fn run(xfm: *Trunk) void {
            xfm.markQsaPooledRopeStale();
        }
    };
    const th = try std.Thread.spawn(.{}, Worker.run, .{&t});
    th.join();
    try testing.expect(t.qsa_pooled_rope.stale.load(.acquire));
    _ = try t.qsaPooledCosSin(&ctx, rope_dims, base, step, n, .bfloat16);
    try testing.expectEqual(@as(usize, 2), t.qsa_pooled_rope.builds);
}

test "qsa block constants: one build per nb, not one per full-attention layer" {
    // Twelve full-attention layers asked for the same nb-keyed aranges; `builds` is the meter.
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var qc: QsaBlockConsts = .{};
    defer qc.deinit();

    try qc.ensure(s, 6, 4);
    try testing.expectEqual(@as(usize, 1), qc.builds);
    const first_end = qc.blk_end2.ctx;
    const first_bias = qc.tie_bias.ctx;
    for (0..11) |_| try qc.ensure(s, 6, 4); // layers 2..12 of the same forward
    try testing.expectEqual(@as(usize, 1), qc.builds);
    try testing.expect(qc.blk_end2.ctx == first_end);
    try testing.expect(qc.tie_bias.ctx == first_bias);

    var ends = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ends);
    try mlx.check(mlx.mlx_astype(&ends, qc.blk_end2, .float32, s));
    try mlx.check(mlx.mlx_array_eval(ends));
    const ev = mlx.mlx_array_data_float32(ends).?;
    for (0..6) |b| try testing.expectEqual(@as(f32, @floatFromInt(4 * b + 3)), ev[b]);
    try mlx.check(mlx.mlx_array_eval(qc.tie_bias));
    const bv = mlx.mlx_array_data_float32(qc.tie_bias).?;
    for (0..6) |b| try testing.expectEqual(@as(f32, @floatFromInt(b)) * 1e-7, bv[b]);

    try qc.ensure(s, 7, 4);
    try testing.expectEqual(@as(usize, 2), qc.builds);
    try testing.expectEqual(@as(c_int, 7), mlx.getShape(qc.blk_end2)[1]);
    try qc.ensure(s, 7, 8);
    try testing.expectEqual(@as(usize, 3), qc.builds);
}
