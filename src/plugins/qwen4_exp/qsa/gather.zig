//! QSA prefill attention over the gathered blocks (`gatherQsa256`, the hd-256 kernel and its NAX twin), their
//! floors, eligibility and transient bills.

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;
const mlx = @import("../../../mlx/ffi.zig");
const log = @import("../../../sdk/util/log.zig");
const arms = @import("arms.zig");
const transformer = @import("../../../engines/mlx/transformer.zig");
const macosProductVersion = @import("../../../mlx/device.zig").macosProductVersion;
const macosVersionAtLeast = @import("../../../mlx/device.zig").macosVersionAtLeast;
const QsaCfgCache = @import("../../../mlx/kernel_cache.zig").QsaCfgCache;
const ATTN256_KERNEL_HEADER = @import("../../../sdk/blocks/attention/attn256.zig").ATTN256_KERNEL_HEADER;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const attn256Base16Aligned = @import("../../../sdk/blocks/attention/attn256.zig").attn256Base16Aligned;
const attn256MaxDiff = @import("../../../sdk/blocks/attention/attn256.zig").attn256MaxDiff;
const attn256RandBf16 = @import("../../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const attn256Reference = @import("../../../sdk/blocks/attention/attn256.zig").attn256Reference;
const attn256SliceLastDim = @import("../../../sdk/blocks/attention/attn256.zig").attn256SliceLastDim;
const verifyQmmNaxAvailable = @import("../../../sdk/blocks/linear/verify_qmm_nax.zig").verifyQmmNaxAvailable;
const ShapeKey = @import("../../../sdk/blocks/moe/router_kernel.zig").ShapeKey;
const qwen4Standin = @import("../../../sdk/diag/standin.zig").qwen4Standin;
const qsaEngagedBit = @import("arms.zig").qsaEngagedBit;
const qsaMaskFromBlocks = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskFromBlocks;
const qsaAttnKernelEnabled = @import("sparse_attn.zig").qsaAttnKernelEnabled;
const qsaFusedVerifyServes = @import("sparse_attn.zig").qsaFusedVerifyServes;
const qsaMaskArm = @import("../../../sdk/blocks/attention/sparse_block/mask.zig").qsaMaskArm;
const qsaProbeAllFinite = @import("../../../sdk/blocks/attention/sparse_block/probe.zig").qsaProbeAllFinite;
const qsaProbeCorruptFirst = @import("../../../sdk/blocks/attention/sparse_block/probe.zig").qsaProbeCorruptFirst;
const qsaProbeLcgBf16 = @import("../../../sdk/blocks/attention/sparse_block/probe.zig").qsaProbeLcgBf16;
const qsaProbeMaxAbsDiff = @import("../../../sdk/blocks/attention/sparse_block/probe.zig").qsaProbeMaxAbsDiff;
const qsaVerifyGatherEnabled = @import("verify_gather.zig").qsaVerifyGatherEnabled;
const qsaVerifyGatherMinKvFor = @import("verify_gather.zig").qsaVerifyGatherMinKvFor;


/// The block-gather kernel serves the qwen4_exp QSA prefill; the test seam's false
/// runs the dense [S, kv] mask arm.
pub fn qsaGatherEnabled() bool {
    return transformer.qsa_gather_override orelse true;
}

/// Below this many keys the mask arm wins: it reads at most kv rows per
/// 64-row tile, the gather reads 2051 rows per token unshared (ladder: 8k
/// -3%, 16k +4%, 32k +19%, 64k +42%).
pub const QSA_GATHER_MIN_KV_DEFAULT: c_int = 8192;

pub fn qsaGatherMinKv() c_int {
    return transformer.qsa_gather_min_kv_override orelse QSA_GATHER_MIN_KV_DEFAULT;
}

/// The prefill floor where the tensor-unit QSA kernel serves: it beats the mask arm from the
/// first sparse chunk (Flash Next, M5 Ultra, first 8192-row chunk: QSA stage 634 -> 120 ms per
/// chunk). At or below (top-512 + 1) * 4 keys every block is visible and there is nothing to skip.
pub const QSA_GATHER_MIN_KV_NAX: c_int = 2052;

/// The kv floor for prefill widths. The test override wins; otherwise the NAX floor when the
/// tensor-unit kernel serves, else the stock gather's floor.
/// Decode and verify widths keep `qsaGatherMinKv`: their gathers do not use the NAX kernel.
pub fn qsaPrefillGatherMinKv() c_int {
    if (transformer.qsa_gather_min_kv_override != null) return qsaGatherMinKv();
    if (!qsaNaxEnabled() or !qsaNaxOsOk() or !verifyQmmNaxAvailable()) return qsaGatherMinKv();
    return QSA_GATHER_MIN_KV_NAX;
}

pub const QSA_GATHER_BK_DEFAULT: c_int = 32;
pub var qsa_gather_bk_override: ?c_int = null;

/// Key tile of the gather kernel: 16 or 32.
pub fn qsaGatherBk() c_int {
    return qsa_gather_bk_override orelse QSA_GATHER_BK_DEFAULT;
}


/// Decode-width QSA gather (S==1): reuse the sorted block selection and read
/// only the selected blocks + incomplete tail instead of the full KV — O(topk)
/// per step, never O(kv). The test seam's false runs the dense bool-mask arm
/// (same selection, full-KV SDPA).
pub fn qsaDecodeGatherEnabled() bool {
    return transformer.qsa_decode_gather_override orelse true;
}

pub var qsa_batched_gather_logged: bool = false;

pub fn qsaBatchedGatherEnabled() bool {
    return transformer.qsa_batched_gather_override orelse true;
}

pub fn qsaBatchedGatherOn(seq_len: c_int) bool {
    const st = qwen4Standin();
    if (st.attn_qsa or st.attn_sdpa) return false;
    if (!qsaBatchedGatherEnabled()) return false;
    if (!qsaGatherEnabled()) return false;
    if (seq_len <= 1) return qsaDecodeGatherEnabled();
    // Verify widths: the fused kernel or its union-gather fallback (`qsaBatchedGatherFloor`
    // says which one, and where).
    return qsaVerifyGatherEnabled() or qsaAttnKernelEnabled();
}

pub fn qsaBatchedGatherFloor(seq_len: c_int, quantized: bool) c_int {
    // The fused verify kernel reads O(budget) rows per row at every kv the indexer is active.
    if (qsaFusedVerifyServes(quantized, seq_len)) return 0;
    const verify_w = seq_len >= 2 and seq_len < FUSED256_MIN_Q_LEN;
    // Neither verify arm serves this KV type: the dense mask at every kv.
    if (verify_w and !qsaVerifyGatherEnabled()) return std.math.maxInt(c_int);
    if (seq_len >= 2) return @max(qsaGatherMinKv(), qsaVerifyGatherMinKvFor(quantized));
    return qsaGatherMinKv();
}


/// The newest SSM checkpoint takes a view of the slot's live
/// history at commit (`handoffQsaHistoryToLatest`): one copy per (slot ∪ entry).
pub fn qsaHistoryShareEnabled() bool {
    return transformer.qsa_history_share_override orelse true;
}

/// Virtual key index -> cache position for one query's block selection:
/// the `sel_len` tokens of its sorted blocks, then its own incomplete tail.
pub const ATTN_QSA256_KERNEL_HEADER = ATTN256_KERNEL_HEADER ++ kernelSource("plugins/qwen4_exp/kernels/attn_qsa256_header.metal");

/// msv_attn_qsa256: one threadgroup per (query token, kv head, batch). The
/// direct-index approach (gather each query's selected blocks, never a dense
/// [S, kv] mask) follows oMLX's Qwen4 exact QSA prefill (jundot/omlx PR
/// #3244, Apache-2.0); no code was ported — the kernel body is derived from
/// our own msv_attn_p256 (whose K/V staging their kernel credits in turn).
/// token's `gqa` query heads are the tile rows (NSG simdgroups x 8); its keys
/// are gathered by the token's sorted selected blocks (`blocks[b, s, :]`,
/// RATIO tokens each, INT_MAX past the row's count) followed by its own
/// incomplete tail — O(budget) per token, never O(kv). Fragment math,
/// staging and online softmax mirror msv_attn_p256.
pub const ATTN_QSA256_KERNEL_SOURCE = kernelSource("plugins/qwen4_exp/kernels/attn_qsa256.metal");

var attn_qsa256_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;

fn getAttnQsa256Kernel() !mlx.mlx_fast_metal_kernel {
    if (attn_qsa256_kernel_cached) |kk| return kk;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "scl", "blocks" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_attn_qsa256",
        in_vec,
        out_vec,
        ATTN_QSA256_KERNEL_SOURCE,
        ATTN_QSA256_KERNEL_HEADER,
        false, // K/V are cache views (see msv_attn_p256)
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    attn_qsa256_kernel_cached = kernel;
    return kernel;
}

// The Metal JIT of a cooperative-tensor kernel happens at first eval, not at
// construction, so `qsaNaxArm` runs it once (tiny apply + eval, inference
// thread) and a failure latches the stock gather for the process.
var qsa_nax_os_cached: ?bool = null;
var qsa_nax_kernel_cached: ?mlx.mlx_fast_metal_kernel = null;
pub var qsa_nax_override: ?bool = null;

pub fn qsaNaxEnabled() bool {
    if (qsaNaxProbeFailed()) return false;
    return qsa_nax_override orelse true;
}

fn qsaNaxOsSupported(version: []const u8) bool {
    return macosVersionAtLeast(version, 26, 3);
}

fn qsaNaxOsOk() bool {
    if (qsa_nax_os_cached) |v| return v;
    var version_buf: [64]u8 = undefined;
    const supported = qsaNaxOsSupported(macosProductVersion(&version_buf) orelse "");
    qsa_nax_os_cached = supported;
    return supported;
}

pub fn qsaNaxEligibleFrom(
    nax_available: bool,
    os_ver: []const u8,
    q_dtype: mlx.mlx_dtype,
    k_dtype: mlx.mlx_dtype,
    v_dtype: mlx.mlx_dtype,
    hd: c_int,
    gqa: c_int,
    q_len: c_int,
) bool {
    if (!nax_available) return false;
    if (!qsaNaxOsSupported(os_ver)) return false;
    if (q_dtype != .bfloat16 or k_dtype != .bfloat16 or v_dtype != .bfloat16) return false;
    if (hd != 256) return false;
    if (gqa != 12) return false;
    if (q_len < 16) return false;
    return true;
}

pub fn qsaNaxEligible(
    q_dtype: mlx.mlx_dtype,
    k_dtype: mlx.mlx_dtype,
    v_dtype: mlx.mlx_dtype,
    hd: c_int,
    gqa: c_int,
    q_len: c_int,
) bool {
    if (qsaNaxProbeFailed()) return false;
    return qsaNaxEligibleFrom(
        verifyQmmNaxAvailable(),
        if (qsaNaxOsOk()) "26.3" else "26.2",
        q_dtype,
        k_dtype,
        v_dtype,
        hd,
        gqa,
        q_len,
    );
}

var qsa_nax_probe_state: std.atomic.Value(u8) = .init(0);
pub var qsa_nax_probe_fail_override: bool = false;
pub var qsa_nax_probe_flip_override: bool = false;
var qsa_nax_kernel_builds: usize = 0;
pub var qsa_gather_used_nax: bool = false;
pub var qsa_gather_engaged_stock: u32 = 0;
pub var qsa_gather_engaged_nax: u32 = 0;

pub fn qsaNaxProbeFailed() bool {
    return qsa_nax_probe_state.load(.acquire) == 1;
}

pub fn qsaNaxArm() void {
    if (qsa_nax_probe_fail_override) {
        qsa_nax_probe_state.store(1, .release);
        return;
    }
    if (qsa_nax_probe_state.load(.acquire) != 0) return;
    if (qsa_nax_override) |v| if (!v) return;
    if (!verifyQmmNaxAvailable()) return;
    if (!qsaNaxOsOk()) return;
    const ok = qsaNaxRunProbe();
    qsa_nax_probe_state.store(if (ok) 2 else 1, .release);
}

const QSA_NAX_PROBE_FLOOR: f32 = 4.9e-4;

fn qsaGatherProbeDispatch(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    blocks: mlx.mlx_array,
    ratio: c_int,
    nax: bool,
) ?mlx.mlx_array {
    const kernel = (if (nax) getQsaNaxKernel() else getAttnQsa256Kernel()) catch return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const gqa: c_int = @divTrunc(qs[1], ks[1]);
    const nsg: c_int = @divTrunc(gqa + 7, 8);
    const bk: c_int = if (nax) 32 else qsaGatherBk();
    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const o_shape = [_]c_int{ qs[0], qs[1], qs[2], 256 };
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .bfloat16) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_set_grid(config, qs[2] * 32, ks[1] * nsg, qs[0]) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", nsg) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BK", bk) != 0) return null;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RATIO", ratio) != 0) return null;
    const inputs_arr = [_]mlx.mlx_array{ q, k, v, scl, blocks };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    if (mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s) != 0) return null;
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return null;
    var out = mlx.mlx_array_new();
    if (mlx.mlx_vector_array_get(&out, outputs_vec, 0) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    if (mlx.mlx_array_eval(out) != 0) {
        _ = mlx.mlx_array_free(out);
        return null;
    }
    return out;
}

fn qsaNaxRunProbe() bool {
    if (mlx.noGpuBackend()) return false;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return false;
    const q = qsaProbeLcgBf16(s, &[_]c_int{ 1, 24, 16, 256 }, 0x9A7E016) orelse return false;
    defer _ = mlx.mlx_array_free(q);
    const k = qsaProbeLcgBf16(s, &[_]c_int{ 1, 2, 16, 256 }, 0x51A2002) orelse return false;
    defer _ = mlx.mlx_array_free(k);
    const v = qsaProbeLcgBf16(s, &[_]c_int{ 1, 2, 16, 256 }, 0xC0FFEE3) orelse return false;
    defer _ = mlx.mlx_array_free(v);
    var blk: [16 * 512]i32 = undefined;
    for (0..16) |r| {
        const complete = (r + 1) / 4;
        for (0..512) |b| blk[r * 512 + b] = if (b < complete) @intCast(b) else std.math.maxInt(i32);
    }
    const bshape = [_]c_int{ 1, 16, 512 };
    const blocks = mlx.mlx_array_new_data(&blk, &bshape, 3, .int32);
    defer _ = mlx.mlx_array_free(blocks);
    const scale: f32 = 1.0 / 16.0;
    const ratio: c_int = 4;
    const stock = qsaGatherProbeDispatch(s, q, k, v, scale, blocks, ratio, false) orelse return false;
    defer _ = mlx.mlx_array_free(stock);
    var nax = qsaGatherProbeDispatch(s, q, k, v, scale, blocks, ratio, true) orelse return false;
    defer _ = mlx.mlx_array_free(nax);
    if (qsa_nax_probe_flip_override) {
        const flipped = qsaProbeCorruptFirst(s, nax) orelse return false;
        _ = mlx.mlx_array_free(nax);
        nax = flipped;
    }
    const ref = qsaGatherF32Reference(q, k, v, scale, blocks, ratio, s) catch return false;
    defer _ = mlx.mlx_array_free(ref);
    if (!qsaProbeAllFinite(s, nax) or !qsaProbeAllFinite(s, stock) or !qsaProbeAllFinite(s, ref)) {
        log.info("[qsa-gather] disabled: probe mismatch (non-finite)\n", .{});
        return false;
    }
    const max_stock = qsaProbeMaxAbsDiff(s, stock, ref) orelse return false;
    const max_nax = qsaProbeMaxAbsDiff(s, nax, ref) orelse return false;
    if (!(max_nax <= @max(1.5 * max_stock, QSA_NAX_PROBE_FLOOR))) {
        log.info("[qsa-gather] disabled: probe mismatch (nax vs stock)\n", .{});
        return false;
    }
    return true;
}

pub const QSA_NAX_SOURCE = kernelSource("plugins/qwen4_exp/kernels/qsa_nax.metal");
pub const QSA_NAX_HEADER = kernelSource("plugins/qwen4_exp/kernels/qsa_nax_header.metal");

fn getQsaNaxKernel() !mlx.mlx_fast_metal_kernel {
    if (!verifyQmmNaxAvailable()) return error.NaxUnavailable;
    if (qsa_nax_kernel_cached) |kernel| return kernel;
    const names = [_][*:0]const u8{ "q", "k", "v", "scl", "blocks" };
    const outputs = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&names, names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outputs, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "msv_qsa_nax_precise",
        in_vec,
        out_vec,
        QSA_NAX_SOURCE,
        QSA_NAX_HEADER,
        false,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    qsa_nax_kernel_cached = kernel;
    qsa_nax_kernel_builds += 1;
    return kernel;
}

const QsaGatherCfgKey = struct {
    q_shape: ShapeKey,
    h_kv: c_int,
    nsg: c_int,
    bk: c_int,
    ratio: c_int,
    nax: bool,
    dtype: mlx.mlx_dtype,
};
var qsa_gather_cfgs = QsaCfgCache(QsaGatherCfgKey, 1){};

/// Block-gathered QSA attention (qwen4_exp prefill): `blocks` is the
/// [B, S, KB] int32 selection, each row ascending with INT_MAX past its
/// count; `ratio` tokens per block. Same envelope as the p256 kernel (hd
/// 256, bf16, q <= k). Null = declined (kill switch, dtype, geometry) — the
/// caller expands the selection to the dense mask arm.
pub fn gatherQsa256(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    blocks: mlx.mlx_array,
    ratio: c_int,
) !?mlx.mlx_array {
    if (!qsaGatherEnabled()) return null;
    if (!mlx.streamIsGpu(s)) return null;
    if (blocks.ctx == null or mlx.mlx_array_ndim(blocks) != 3 or mlx.mlx_array_dtype(blocks) != .int32) return null;
    if (mlx.mlx_array_ndim(q) != 4 or mlx.mlx_array_ndim(k) != 4 or mlx.mlx_array_ndim(v) != 4) return null;
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const vs = mlx.getShape(v);
    const bs = mlx.getShape(blocks);
    if (qs[3] != 256 or ks[3] != 256 or vs[3] != 256) return null;
    if (ks[1] <= 0 or @rem(qs[1], ks[1]) != 0) return null;
    const gqa: c_int = @divTrunc(qs[1], ks[1]);
    if (gqa > 64) return null;
    if (ks[2] < qs[2] or ks[2] != vs[2] or ks[1] != vs[1] or ks[0] != qs[0] or vs[0] != qs[0]) return null;
    if (bs[0] != qs[0] or bs[1] != qs[2] or bs[2] <= 0) return null;
    if (ratio <= 0) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or mlx.mlx_array_dtype(k) != .bfloat16 or mlx.mlx_array_dtype(v) != .bfloat16) return null;

    // K/V stage through `uint4` loads in the stock kernel: an unaligned
    // materialized view base reads wrong values; decline and the caller's
    // mask arm runs the same shape (same bar as `fusedSdpa256Impl`).
    if (!attn256Base16Aligned(k) or !attn256Base16Aligned(v)) return null;

    qsaNaxArm();
    const qst = mlx.mlx_array_strides(q);
    const kst = mlx.mlx_array_strides(k);
    const vst = mlx.mlx_array_strides(v);
    const bst = mlx.mlx_array_strides(blocks);
    const nax_aligned = qst[3] == 1 and kst[3] == 1 and vst[3] == 1 and bst[2] == 1 and
        qst[1] % 8 == 0 and qst[2] % 8 == 0 and
        kst[1] % 8 == 0 and kst[2] % 8 == 0 and
        vst[1] % 8 == 0 and vst[2] % 8 == 0;
    var use_nax = nax_aligned and ratio == 4 and qs[0] == 1 and qs[1] == 24 and ks[1] == 2 and bs[2] == 512 and
        qsaNaxEnabled() and qsaNaxEligible(
        mlx.mlx_array_dtype(q),
        mlx.mlx_array_dtype(k),
        mlx.mlx_array_dtype(v),
        qs[3],
        gqa,
        qs[2],
    );
    const kernel = blk: {
        if (use_nax) {
            break :blk getQsaNaxKernel() catch {
                qsa_nax_probe_state.store(1, .release);
                use_nax = false;
                break :blk getAttnQsa256Kernel() catch return null;
            };
        }
        break :blk getAttnQsa256Kernel() catch return null;
    };
    qsa_gather_used_nax = use_nax;
    const one = [_]c_int{1};
    const scl_data = [_]f32{scale};
    const scl = mlx.mlx_array_new_data(&scl_data, &one, 1, .float32);
    defer _ = mlx.mlx_array_free(scl);
    const nsg: c_int = @divTrunc(gqa + 7, 8);
    const bk: c_int = if (use_nax) 32 else qsaGatherBk();
    const key = QsaGatherCfgKey{
        .q_shape = ShapeKey.from(qs),
        .h_kv = ks[1],
        .nsg = nsg,
        .bk = bk,
        .ratio = ratio,
        .nax = use_nax,
        .dtype = .bfloat16,
    };
    const cfgs: [1]mlx.mlx_fast_metal_kernel_config = qsa_gather_cfgs.get(key) orelse blk: {
        const config = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(config);
        const o_shape = [_]c_int{ qs[0], qs[1], qs[2], 256 };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &o_shape, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, qs[2] * 32, ks[1] * nsg, qs[0]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, nsg, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", nsg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BK", bk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "RATIO", ratio));
        qsa_gather_cfgs.put(key, .{config});
        break :blk .{config};
    };

    const inputs_arr = [_]mlx.mlx_array{ q, k, v, scl, blocks };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfgs[0], s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    const engaged_bit: u5 = qsaEngagedBit(.prefill_gather, qs[2]) + @as(u5, if (use_nax) 16 else 0);
    if (arms.qsa_engaged_bits.take(engaged_bit)) {
        const tgmem: usize = if (use_nax)
            2 * 8 * 32 * 4
        else
            @as(usize, @intCast(bk + 8)) * 256 * 2;
        if (use_nax) {
            qsa_gather_engaged_nax += 1;
            log.info("[qsa-gather] engaged: msv_qsa_nax_precise S={d} kv={d} blocks={d} bk={d} tgmem={d}\n", .{ qs[2], ks[2], bs[2], bk, tgmem });
        } else {
            qsa_gather_engaged_stock += 1;
            log.info("[qsa-gather] engaged: msv_attn_qsa256 S={d} kv={d} blocks={d} bk={d} tgmem={d}\n", .{ qs[2], ks[2], bs[2], bk, tgmem });
        }
    }
    return out;
}

fn qsaGatherRowL(p: c_int, kb: c_int, ratio: c_int) c_int {
    const complete: c_int = @divTrunc(p + 1, ratio);
    const sel_len: c_int = @min(complete, kb) * ratio;
    const tail_start: c_int = complete * ratio;
    return sel_len + p + 1 - tail_start;
}

fn qsaGatherRowPos(blk_row: []const i32, vi: c_int, p: c_int, kb: c_int, ratio: c_int) u32 {
    const complete: c_int = @divTrunc(p + 1, ratio);
    const sel_len: c_int = @min(complete, kb) * ratio;
    const tail_start: c_int = complete * ratio;
    if (vi < sel_len) {
        const b: c_int = @divTrunc(vi, ratio);
        return @intCast(blk_row[@intCast(b)] * ratio + (vi - b * ratio));
    }
    return @intCast(tail_start + (vi - sel_len));
}

fn qsaGatherF32Reference(
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    blocks: mlx.mlx_array,
    ratio: c_int,
    stream: mlx.mlx_stream,
) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    const B = qs[0];
    const Hq = qs[1];
    const qL = qs[2];
    const hd = qs[3];
    const Hk = ks[1];
    const kL = ks[2];
    const kb = bs[2];
    const gqa: c_int = @divTrunc(Hq, Hk);
    if (B != 1) return error.ShapeMismatch;
    try mlx.check(mlx.mlx_array_eval(blocks));
    const bp = mlx.mlx_array_data_int32(blocks) orelse return error.InvalidDtype;
    const nq: usize = @intCast(qL);
    const nkb: usize = @intCast(kb);
    var lmax: c_int = 1;
    for (0..nq) |si| {
        const p: c_int = kL - qL + @as(c_int, @intCast(si));
        lmax = @max(lmax, qsaGatherRowL(p, kb, ratio));
    }
    const nl: usize = @intCast(lmax);
    const ta = std.heap.page_allocator;
    const idx = try ta.alloc(u32, nq * nl);
    defer ta.free(idx);
    const vis = try ta.alloc(bool, nq * nl);
    defer ta.free(vis);
    @memset(idx, 0);
    @memset(vis, false);
    for (0..nq) |si| {
        const p: c_int = kL - qL + @as(c_int, @intCast(si));
        const L = qsaGatherRowL(p, kb, ratio);
        const row = bp[si * nkb .. si * nkb + nkb];
        var vi: c_int = 0;
        while (vi < L) : (vi += 1) {
            idx[si * nl + @as(usize, @intCast(vi))] = qsaGatherRowPos(row, vi, p, kb, ratio);
            vis[si * nl + @as(usize, @intCast(vi))] = true;
        }
    }
    const ishape = [_]c_int{ qL, lmax };
    const vshape = [_]c_int{ 1, 1, qL, lmax };
    const idx_arr = mlx.mlx_array_new_data(idx.ptr, &ishape, 2, .uint32);
    defer _ = mlx.mlx_array_free(idx_arr);
    const vis_arr = mlx.mlx_array_new_data(vis.ptr, &vshape, 4, .bool_);
    defer _ = mlx.mlx_array_free(vis_arr);

    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    var k32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k32);
    var v32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v32);
    try mlx.check(mlx.mlx_astype(&q32, q, .float32, stream));
    try mlx.check(mlx.mlx_astype(&k32, k, .float32, stream));
    try mlx.check(mlx.mlx_astype(&v32, v, .float32, stream));

    var kg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kg);
    var vg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vg);
    try mlx.check(mlx.mlx_take_axis(&kg, k32, idx_arr, 2, stream));
    try mlx.check(mlx.mlx_take_axis(&vg, v32, idx_arr, 2, stream));

    var kge = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kge);
    var vge = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vge);
    try mlx.check(mlx.mlx_expand_dims(&kge, kg, 2, stream));
    try mlx.check(mlx.mlx_expand_dims(&vge, vg, 2, stream));
    const bcast_shape = [_]c_int{ B, Hk, gqa, qL, lmax, hd };
    var kgb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kgb);
    var vgb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vgb);
    try mlx.check(mlx.mlx_broadcast_to(&kgb, kge, &bcast_shape, 6, stream));
    try mlx.check(mlx.mlx_broadcast_to(&vgb, vge, &bcast_shape, 6, stream));
    const gshape = [_]c_int{ B, Hq, qL, lmax, hd };
    var kgr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kgr);
    var vgr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vgr);
    try mlx.check(mlx.mlx_reshape(&kgr, kgb, &gshape, 5, stream));
    try mlx.check(mlx.mlx_reshape(&vgr, vgb, &gshape, 5, stream));

    var qe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qe);
    try mlx.check(mlx.mlx_expand_dims(&qe, q32, 3, stream));
    var dots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dots);
    try mlx.check(mlx.mlx_multiply(&dots, qe, kgr, stream));
    var scores_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_u);
    try mlx.check(mlx.mlx_sum_axis(&scores_u, dots, 4, false, stream));
    const scl = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scl);
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_multiply(&scores, scores_u, scl, stream));

    const ninf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(ninf);
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try mlx.check(mlx.mlx_where(&masked, vis_arr, scores, ninf, stream));
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, masked, 3, true, stream));
    var pe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pe);
    try mlx.check(mlx.mlx_expand_dims(&pe, probs, 4, stream));
    var wv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wv);
    try mlx.check(mlx.mlx_multiply(&wv, pe, vgr, stream));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum_axis(&out, wv, 3, false, stream));
    return out;
}

// ── tests ──

test "qsa prefill floor: the NAX floor only where the tensor-unit kernel serves, overrides win" {
    defer qsa_nax_override = null;
    defer transformer.qsa_gather_min_kv_override = null;
    qsa_nax_override = false;
    try std.testing.expectEqual(qsaGatherMinKv(), qsaPrefillGatherMinKv());
    qsa_nax_override = true;
    if (qsaNaxOsOk() and verifyQmmNaxAvailable())
        try std.testing.expectEqual(QSA_GATHER_MIN_KV_NAX, qsaPrefillGatherMinKv());
    transformer.qsa_gather_min_kv_override = 5000;
    try std.testing.expectEqual(@as(c_int, 5000), qsaPrefillGatherMinKv());
}

pub fn qsaNaxProbeReset() void {
    qsa_nax_probe_state.store(0, .release);
}

pub fn qsaNaxKernelBuilds() usize {
    return qsa_nax_kernel_builds;
}

/// A QSA-shaped bool mask [1,1,qL,kL]: causal, bottom-right aligned, with a
/// random half of the complete 4-token blocks dropped per row and the row's
/// own tail always visible — the shape `qsaMask` emits.
pub fn attn256QsaMask(rnd: std.Random, qL: c_int, kL: c_int) !mlx.mlx_array {
    const n: usize = @intCast(qL * kL);
    const buf = try std.testing.allocator.alloc(bool, n);
    defer std.testing.allocator.free(buf);
    const q_off = kL - qL;
    for (0..@intCast(qL)) |r| {
        const p: c_int = q_off + @as(c_int, @intCast(r));
        const tail_start: c_int = 4 * @divTrunc(p + 1, 4);
        for (0..@intCast(kL)) |c| {
            const ci: c_int = @intCast(c);
            const blk_vis = (@rem(ci, 4) == 0 and rnd.boolean()) or (@rem(ci, 4) != 0 and buf[r * @as(usize, @intCast(kL)) + c - 1]);
            buf[r * @as(usize, @intCast(kL)) + c] = ci <= p and (ci >= tail_start or blk_vis);
        }
    }
    const shape = [_]c_int{ 1, 1, qL, kL };
    return mlx.mlx_array_new_data(buf.ptr, &shape, 4, .bool_);
}

test "qsaMaskArm: on CUDA the bool mask runs as an additive one, bit-identical to MLX's own conversion" {
    if (!mlx.cudaAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x5a17);
    const rnd = prng.random();
    for ([_][2]c_int{ .{ 70, 193 }, .{ 256, 1024 }, .{ 872, 7016 } }) |dims| {
        const q_shape = [_]c_int{ 1, 24, dims[0], 256 };
        const kv_shape = [_]c_int{ 1, 2, dims[1], 256 };
        const q = try attn256RandBf16(rnd, &q_shape, s);
        defer _ = mlx.mlx_array_free(q);
        const k = try attn256RandBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &kv_shape, s);
        defer _ = mlx.mlx_array_free(v);
        const mask = try attn256QsaMask(rnd, dims[0], dims[1]);
        defer _ = mlx.mlx_array_free(mask);
        const got = try qsaMaskArm(s, q, k, v, 1.0 / 16.0, mask);
        defer _ = mlx.mlx_array_free(got);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k, v, 1.0 / 16.0, "array", mask, .{ .ctx = null }, false, s));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, got, ref, false, s));
        try mlx.check(mlx.mlx_array_eval(eq));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try std.testing.expect(same);
    }
}

/// A sorted qwen4 block selection [1,qL,kb] int32 for rows straddling the
/// "every block fits" boundary (INT_MAX past each row's count), plus the
/// equivalent dense [1,1,qL,kL] mask the composed reference consumes.
pub const QsaBlockFixture = struct {
    blocks: mlx.mlx_array,
    mask: mlx.mlx_array,
    pub fn build(rnd: std.Random, qL: c_int, kL: c_int, kb: c_int, ratio: c_int) !QsaBlockFixture {
        const ta = std.testing.allocator;
        const nq: usize = @intCast(qL);
        const nk: usize = @intCast(kL);
        const nkb: usize = @intCast(kb);
        const blocks = try ta.alloc(i32, nq * nkb);
        defer ta.free(blocks);
        const mask = try ta.alloc(bool, nq * nk);
        defer ta.free(mask);
        const picked = try ta.alloc(bool, @intCast(@divTrunc(kL, ratio) + 1));
        defer ta.free(picked);
        const q_off = kL - qL;
        for (0..nq) |r| {
            const p: c_int = q_off + @as(c_int, @intCast(r));
            const complete: c_int = @divTrunc(p + 1, ratio);
            const count: c_int = @min(complete, kb);
            @memset(picked, false);
            var n: c_int = 0;
            while (n < count) {
                const b: usize = rnd.uintLessThan(usize, @intCast(complete));
                if (picked[b]) continue;
                picked[b] = true;
                n += 1;
            }
            var w: usize = 0;
            for (0..@intCast(complete)) |b| if (picked[b]) {
                blocks[r * nkb + w] = @intCast(b);
                w += 1;
            };
            while (w < nkb) : (w += 1) blocks[r * nkb + w] = std.math.maxInt(i32);
            for (0..nk) |c| {
                const ci: c_int = @intCast(c);
                mask[r * nk + c] = ci <= p and (ci >= complete * ratio or picked[@intCast(@divTrunc(ci, ratio))]);
            }
        }
        const bshape = [_]c_int{ 1, qL, kb };
        const mshape = [_]c_int{ 1, 1, qL, kL };
        return .{
            .blocks = mlx.mlx_array_new_data(blocks.ptr, &bshape, 3, .int32),
            .mask = mlx.mlx_array_new_data(mask.ptr, &mshape, 4, .bool_),
        };
    }
    pub fn deinit(self: *QsaBlockFixture) void {
        _ = mlx.mlx_array_free(self.blocks);
        _ = mlx.mlx_array_free(self.mask);
    }
};

fn qsaNaxAssertNoWorseThanStock(
    stream: mlx.mlx_stream,
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    blocks: mlx.mlx_array,
    ratio: c_int,
    floor: f32,
) !void {
    const ref = try qsaGatherF32Reference(q, k, v, scale, blocks, ratio, stream);
    defer _ = mlx.mlx_array_free(ref);
    qsa_nax_override = false;
    const stock = (try gatherQsa256(stream, q, k, v, scale, blocks, ratio)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(stock);
    qsa_nax_override = true;
    const nax = (try gatherQsa256(stream, q, k, v, scale, blocks, ratio)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(nax);
    try std.testing.expect(qsa_gather_used_nax);
    const max_stock = try attn256MaxDiff(stock, ref, stream);
    const max_nax = try attn256MaxDiff(nax, ref, stream);
    std.debug.print("max_stock={e} max_nax={e} floor={e} bar={e}\n", .{ max_stock, max_nax, floor, @max(1.5 * max_stock, floor) });
    try std.testing.expect(max_nax <= @max(1.5 * max_stock, floor));
}

fn qsaBoostLastTileK(
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    blocks: mlx.mlx_array,
    ratio: c_int,
    stream: mlx.mlx_stream,
) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    const ks = mlx.getShape(k);
    const bs = mlx.getShape(blocks);
    const Hq = qs[1];
    const qL = qs[2];
    const hd = qs[3];
    const Hk = ks[1];
    const kL = ks[2];
    const kb = bs[2];
    const gqa: c_int = @divTrunc(Hq, Hk);
    var qc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(qc);
    var kc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kc);
    try mlx.check(mlx.mlx_contiguous(&qc, q, false, stream));
    try mlx.check(mlx.mlx_contiguous(&kc, k, false, stream));
    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    var k32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k32);
    try mlx.check(mlx.mlx_astype(&q32, qc, .float32, stream));
    try mlx.check(mlx.mlx_astype(&k32, kc, .float32, stream));
    try mlx.check(mlx.mlx_array_eval(q32));
    try mlx.check(mlx.mlx_array_eval(k32));
    try mlx.check(mlx.mlx_array_eval(blocks));
    const qp = mlx.mlx_array_data_float32(q32) orelse return error.InvalidDtype;
    const kp = mlx.mlx_array_data_float32(k32) orelse return error.InvalidDtype;
    const bp = mlx.mlx_array_data_int32(blocks) orelse return error.InvalidDtype;
    const nq: usize = @intCast(qL);
    const nkv: usize = @intCast(kL);
    const nd: usize = @intCast(hd);
    const nkb: usize = @intCast(kb);
    const nhk: usize = @intCast(Hk);
    const buf = try std.testing.allocator.alloc(f32, nhk * nkv * nd);
    defer std.testing.allocator.free(buf);
    @memcpy(buf, kp[0 .. nhk * nkv * nd]);
    const bk_tile: c_int = 32;
    for (0..nq) |si| {
        const p: c_int = kL - qL + @as(c_int, @intCast(si));
        const L = qsaGatherRowL(p, kb, ratio);
        const t0: c_int = L - @min(bk_tile, L);
        const row = bp[si * nkb .. si * nkb + nkb];
        var hk: usize = 0;
        while (hk < nhk) : (hk += 1) {
            const h: usize = hk * @as(usize, @intCast(gqa));
            const qbase = (h * nq + si) * nd;
            var vi: c_int = t0;
            while (vi < L) : (vi += 1) {
                const pos: usize = @intCast(qsaGatherRowPos(row, vi, p, kb, ratio));
                const kbase = (hk * nkv + pos) * nd;
                for (0..nd) |d| buf[kbase + d] += 40.0 * qp[qbase + d];
            }
        }
    }
    const kshape = [_]c_int{ 1, Hk, kL, hd };
    const k_host = mlx.mlx_array_new_data(buf.ptr, &kshape, 4, .float32);
    defer _ = mlx.mlx_array_free(k_host);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, k_host, .bfloat16, stream));
    return out;
}

test "qsaNaxOsSupported: cooperative input tensors require macOS 26.3" {
    try std.testing.expect(!qsaNaxOsSupported("26.2"));
    try std.testing.expect(qsaNaxOsSupported("26.3"));
    try std.testing.expect(qsaNaxOsSupported("26.5"));
    try std.testing.expect(!qsaNaxOsSupported(""));
}

test "qsaNaxEligible: kernel predicate names gqa 12, hd 256, q_len>=16, bf16, NAX, macOS 26.3" {
    const dt_bf16 = mlx.mlx_dtype.bfloat16;
    const dt_f32 = mlx.mlx_dtype.float32;
    try std.testing.expect(qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(qsaNaxEligibleFrom(true, "26.5", dt_bf16, dt_bf16, dt_bf16, 256, 12, 40));
    try std.testing.expect(!qsaNaxEligibleFrom(false, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.2", dt_bf16, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "", dt_bf16, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_f32, dt_bf16, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_f32, dt_bf16, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_f32, 256, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 128, 12, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 8, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 16, 16));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 12, 15));
    try std.testing.expect(!qsaNaxEligibleFrom(true, "26.3", dt_bf16, dt_bf16, dt_bf16, 256, 12, 1));
}

test "gatherQsa256 NAX: precise sparse attention and stock fallback" {
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    qsa_nax_override = true;
    defer qsa_nax_override = null;
    if (!qsaNaxEnabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    defer qsa_gather_bk_override = null;
    var prng = std.Random.DefaultPrng.init(0x9a7e);
    const rnd = prng.random();

    // Early rows have fewer visible blocks than the selection capacity.
    const qL: c_int = 40;
    const kL: c_int = 101;
    const kb: c_int = 512;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;

    const ref = try attn256Reference(q, k, v, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref);
    for ([_]bool{ false, true }) |enabled| {
        qsa_nax_override = enabled;
        const out = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expect(try attn256MaxDiff(out, ref, s) < 0.005);
    }
    try std.testing.expect(qsa_gather_used_nax);
    var smaller = try QsaBlockFixture.build(rnd, qL, kL, 6, 4);
    defer smaller.deinit();
    const fallback = (try gatherQsa256(s, q, k, v, scale, smaller.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(fallback);
    try std.testing.expect(!qsa_gather_used_nax);
}

test "gatherQsa256: a last-axis slice with an unaligned base declines and reads the same shape as the fallback" {
    // The stock `msv_attn_qsa256` kernel stages K/V in `uint4` loads. A
    // `[3,259)` slice of a padded `[1,2,101,264]` cache has a 16-byte row
    // stride (264) but a 6-byte base, so the loads read wrong values. The
    // gate must decline it and the caller's fallback (`qsaMaskArm`) must
    // serve the same shape. `[0,256)` is the aligned twin: it keeps the
    // stock kernel.
    const s = mlx.gpuStream();
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    qsa_nax_override = false;
    defer qsa_nax_override = null;
    defer qsa_gather_bk_override = null;
    var prng = std.Random.DefaultPrng.init(0x0FF_5E7);
    const rnd = prng.random();

    const qL: c_int = 40;
    const kL: c_int = 101;
    const kb: c_int = 512;
    const q = try attn256RandBf16(rnd, &[_]c_int{ 1, 24, qL, 256 }, s);
    defer _ = mlx.mlx_array_free(q);
    const k_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kL, 264 }, s);
    defer _ = mlx.mlx_array_free(k_pad);
    const v_pad = try attn256RandBf16(rnd, &[_]c_int{ 1, 2, kL, 264 }, s);
    defer _ = mlx.mlx_array_free(v_pad);
    try mlx.check(mlx.mlx_array_eval(k_pad));
    try mlx.check(mlx.mlx_array_eval(v_pad));
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;

    const k = try attn256SliceLastDim(k_pad, 3, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256SliceLastDim(v_pad, 3, s);
    defer _ = mlx.mlx_array_free(v);

    // The unaligned view must not reach the kernel; the fallback arm reads
    // the same shape and matches the composed reference within the
    // neighboring tests' bar.
    try std.testing.expect((try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) == null);
    const ref = try attn256Reference(q, k, v, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref);
    const arm = try qsaMaskArm(s, q, k, v, scale, fx.mask);
    defer _ = mlx.mlx_array_free(arm);
    try std.testing.expect(try attn256MaxDiff(arm, ref, s) < 0.005);

    // An aligned twin keeps the stock (non-NAX) fast path.
    const k_al = try attn256SliceLastDim(k_pad, 0, s);
    defer _ = mlx.mlx_array_free(k_al);
    const v_al = try attn256SliceLastDim(v_pad, 0, s);
    defer _ = mlx.mlx_array_free(v_al);
    const fast = (try gatherQsa256(s, q, k_al, v_al, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(fast);
    try std.testing.expect(!qsa_gather_used_nax);
    const ref_al = try attn256Reference(q, k_al, v_al, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref_al);
    try std.testing.expect(try attn256MaxDiff(fast, ref_al, s) < 0.005);
}

test "gatherQsa256 NAX: max error vs f32 gather-softmax is at most 1.5x stock" {
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    qsa_nax_override = true;
    defer qsa_nax_override = null;
    if (!qsaNaxEnabled()) return error.SkipZigTest;
    try std.testing.expect(qsaNaxEligibleFrom(true, "26.3", .bfloat16, .bfloat16, .bfloat16, 256, 12, 40));
    const s = mlx.gpuStream();
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x51a2);
    const rnd = prng.random();
    const qL: c_int = 40;
    const kL: c_int = 101;
    const kb: c_int = 512;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;
    try qsaNaxAssertNoWorseThanStock(s, q, k, v, scale, fx.blocks, 4, 4.9e-4);
}

test "gatherQsa256 NAX: online softmax rescale across 32-key tiles vs f32 gather-softmax" {
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    qsa_nax_override = true;
    defer qsa_nax_override = null;
    if (!qsaNaxEnabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0xc0ff);
    const rnd = prng.random();
    const qL: c_int = 40;
    const kL: c_int = 203;
    const kb: c_int = 512;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k0 = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k0);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const k = try qsaBoostLastTileK(q, k0, fx.blocks, 4, s);
    defer _ = mlx.mlx_array_free(k);
    const scale: f32 = 1.0 / 16.0;
    try qsaNaxAssertNoWorseThanStock(s, q, k, v, scale, fx.blocks, 4, 4.9e-4);
}

test "gatherQsa256 NAX: top-512 selection and partial causal tail vs f32 gather-softmax" {
    if (!verifyQmmNaxAvailable() or !qsaNaxOsOk()) return error.SkipZigTest;
    const saved_nax = qsa_nax_override;
    qsa_nax_override = true;
    defer qsa_nax_override = saved_nax;
    if (!qsaNaxEnabled()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0x4020);
    const rnd = prng.random();
    const qL: c_int = 16;
    const kb: c_int = 512;
    const qshape = [_]c_int{ 1, 24, qL, 256 };
    const q = try attn256RandBf16(rnd, &qshape, s);
    defer _ = mlx.mlx_array_free(q);
    for ([_]c_int{ 2053, 2153 }) |kL| {
        const kvshape = [_]c_int{ 1, 2, kL, 256 };
        const k = try attn256RandBf16(rnd, &kvshape, s);
        defer _ = mlx.mlx_array_free(k);
        const v = try attn256RandBf16(rnd, &kvshape, s);
        defer _ = mlx.mlx_array_free(v);
        var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
        defer fx.deinit();
        try qsaNaxAssertNoWorseThanStock(s, q, k, v, 1.0 / 16.0, fx.blocks, 4, 4.9e-4);
    }
}

test "gatherQsa256 NAX: probe failure latches stock gather bit-identical to the stock arm" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    const saved_nax = qsa_nax_override;
    const saved_fail = qsa_nax_probe_fail_override;
    defer {
        qsa_nax_override = saved_nax;
        qsa_nax_probe_fail_override = saved_fail;
        qsaNaxProbeReset();
    }
    var prng = std.Random.DefaultPrng.init(0xa11);
    const rnd = prng.random();
    const qL: c_int = 16;
    const kL: c_int = 16;
    const kb: c_int = 512;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;
    qsa_nax_override = false;
    const stock = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(stock);
    qsa_nax_probe_fail_override = true;
    qsaNaxProbeReset();
    qsa_nax_override = true;
    const fallback = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(fallback);
    try std.testing.expect(!qsa_gather_used_nax);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(stock, fallback, s));
}

test "gatherQsa256 NAX: a probe mismatch latches the stock gather" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    const saved_nax = qsa_nax_override;
    const saved_fail = qsa_nax_probe_fail_override;
    const saved_flip = qsa_nax_probe_flip_override;
    defer {
        qsa_nax_override = saved_nax;
        qsa_nax_probe_fail_override = saved_fail;
        qsa_nax_probe_flip_override = saved_flip;
        qsaNaxProbeReset();
    }
    var prng = std.Random.DefaultPrng.init(0xa11);
    const rnd = prng.random();
    const qL: c_int = 16;
    const kL: c_int = 16;
    const kb: c_int = 512;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;
    qsa_nax_override = false;
    const stock = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(stock);
    qsa_nax_probe_fail_override = false;
    qsa_nax_probe_flip_override = true;
    qsaNaxProbeReset();
    qsa_nax_override = true;
    const fallback = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(fallback);
    try std.testing.expect(qsaNaxProbeFailed());
    try std.testing.expect(!qsa_gather_used_nax);
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(stock, fallback, s));
}

test "gatherQsa256 NAX: probe passes on this machine" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    const saved_nax = qsa_nax_override;
    const saved_fail = qsa_nax_probe_fail_override;
    const saved_flip = qsa_nax_probe_flip_override;
    defer {
        qsa_nax_override = saved_nax;
        qsa_nax_probe_fail_override = saved_fail;
        qsa_nax_probe_flip_override = saved_flip;
        qsaNaxProbeReset();
    }
    qsa_nax_probe_fail_override = false;
    qsa_nax_probe_flip_override = false;
    qsa_nax_override = true;
    qsaNaxProbeReset();
    qsaNaxArm();
    try std.testing.expect(!qsaNaxProbeFailed());
    try std.testing.expect(qsaNaxEnabled());
}

test "gatherQsa256 NAX: latch is consulted before NAX kernel construction" {
    const saved_fail = qsa_nax_probe_fail_override;
    const saved_nax = qsa_nax_override;
    defer {
        qsa_nax_probe_fail_override = saved_fail;
        qsa_nax_override = saved_nax;
        qsaNaxProbeReset();
    }
    qsa_nax_probe_fail_override = true;
    qsaNaxProbeReset();
    qsaNaxArm();
    try std.testing.expect(qsaNaxProbeFailed());
    try std.testing.expect(!qsaNaxEnabled());
    const builds = qsaNaxKernelBuilds();
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    qsa_nax_override = true;
    var prng = std.Random.DefaultPrng.init(0xb0);
    const rnd = prng.random();
    const qL: c_int = 16;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, qL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, qL, 512, 4);
    defer fx.deinit();
    const out = (try gatherQsa256(s, q, k, v, 1.0 / 16.0, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(out);
    try std.testing.expect(!qsa_gather_used_nax);
    try std.testing.expectEqual(builds, qsaNaxKernelBuilds());
}

test "gatherQsa256: declines a non-GPU stream before kernel selection" {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try std.testing.expect(!mlx.streamIsGpu(cpu));
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    var prng = std.Random.DefaultPrng.init(0xc0);
    const rnd = prng.random();
    const qL: c_int = 16;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, qL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, cpu);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, cpu);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, cpu);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, qL, 4, 4);
    defer fx.deinit();
    try std.testing.expect((try gatherQsa256(cpu, q, k, v, 1.0 / 16.0, fx.blocks, 4)) == null);
}

test "gatherQsa256: stock then NAX each fire one engaged line" {
    if (!verifyQmmNaxAvailable()) return error.SkipZigTest;
    if (!qsaNaxOsOk()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    const saved_nax = qsa_nax_override;
    defer qsa_nax_override = saved_nax;
    arms.qsa_engaged_bits.reset();
    qsa_gather_engaged_stock = 0;
    qsa_gather_engaged_nax = 0;
    var prng = std.Random.DefaultPrng.init(0xd0);
    const rnd = prng.random();
    const qL: c_int = 16;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, qL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, qL, 512, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;
    qsa_nax_override = false;
    const a = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(a);
    qsa_nax_override = true;
    qsa_nax_probe_fail_override = false;
    qsaNaxProbeReset();
    const b = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
    defer _ = mlx.mlx_array_free(b);
    try std.testing.expectEqual(@as(u32, 1), qsa_gather_engaged_stock);
    try std.testing.expectEqual(@as(u32, 1), qsa_gather_engaged_nax);
}

test "gatherQsa256: block-gathered QSA parity vs composed 'array' SDPA over the expanded mask (gqa 12, sentinel rows, 16/32 key tiles)" {
    const s = mlx.gpuStream();
    transformer.qsa_gather_override = true;
    defer transformer.qsa_gather_override = null;
    qsa_nax_override = false;
    defer qsa_nax_override = null;
    defer qsa_gather_bk_override = null;
    var prng = std.Random.DefaultPrng.init(0x9a7e);
    const rnd = prng.random();

    // qwen4_exp geometry (24/2 heads, hd 256, ratio 4) with a tiny block
    // budget so the chunk's rows cross from "every block fits" to top-k.
    const qL: c_int = 40;
    const kL: c_int = 101;
    const kb: c_int = 6;
    const q_shape = [_]c_int{ 1, 24, qL, 256 };
    const kv_shape = [_]c_int{ 1, 2, kL, 256 };
    const q = try attn256RandBf16(rnd, &q_shape, s);
    defer _ = mlx.mlx_array_free(q);
    const k = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try attn256RandBf16(rnd, &kv_shape, s);
    defer _ = mlx.mlx_array_free(v);
    var fx = try QsaBlockFixture.build(rnd, qL, kL, kb, 4);
    defer fx.deinit();
    const scale: f32 = 1.0 / 16.0;

    const ref = try attn256Reference(q, k, v, scale, "array", fx.mask, s);
    defer _ = mlx.mlx_array_free(ref);
    for ([_]c_int{ 16, 32 }) |bk| {
        qsa_gather_bk_override = bk;
        const out = (try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) orelse return error.GatherDeclined;
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expect(!qsa_gather_used_nax);
        try std.testing.expect(try attn256MaxDiff(out, ref, s) < 0.005);
    }

    // The production expansion (fallback arm + fixture trace) is the same mask.
    const expanded = try qsaMaskFromBlocks(s, fx.blocks, kL, 4);
    defer _ = mlx.mlx_array_free(expanded);
    var ours32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ours32);
    try mlx.check(mlx.mlx_astype(&ours32, expanded, .float32, s));
    var ref32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref32);
    try mlx.check(mlx.mlx_astype(&ref32, fx.mask, .float32, s));
    try std.testing.expectEqual(@as(f32, 0), try attn256MaxDiff(ours32, ref32, s));

    // Kill switch declines; so does a non-int32 selection.
    transformer.qsa_gather_override = false;
    try std.testing.expect((try gatherQsa256(s, q, k, v, scale, fx.blocks, 4)) == null);
    transformer.qsa_gather_override = true;
    try std.testing.expect((try gatherQsa256(s, q, k, v, scale, ours32, 4)) == null);
}
