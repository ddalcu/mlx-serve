//! A wide prompt layer's hyper-connection stages with fewer passes over the stream (`[L, hc, D]`, 671 MB of bf16 at 16K
//! rows).
//!
//! Unfused, the model runs each side's stage as separate kernels. The attention side: hc_mix (the 24 mixes of
//! every row) reads the stream, then hc_norm (the split, and the collapse of the stream with the previous mix
//! rms-normed) reads it again (and RA_FUSED's read-ahead norm a third time). The FFN side: the hc expand writes the new
//! stream x1 (the x1 region), hc_mix reads it, hc_norm reads it again. Fused: the attention side's mixes and norm (and
//! the read-ahead's norm) are one launch, `kernels/hc_mix_norm.metal`, its mix threadgroups hc_mix's statements and
//! each row group's norm threadgroup hc_norm's (it runs beside the row group's mix threadgroups, which read the same
//! rows); the FFN side's expand and norm are one launch, `kernels/hc_expand_norm.metal` (one threadgroup a row: each
//! thread expands its columns of every copy with the expand's fmas and collapses them from its registers), hc_mix then
//! reads the new stream; both sides take the split from the mixes after (hc_split, the decode stage's own kernel:
//! hc_norm's split statements). The same reductions: every sum over the same threads (`hc_tg`), the same simd sums, the
//! same partials in the same order; a GPU that runs a fused kernel at fewer threads than `hc_tg` refuses it by name at
//! load (the stock kernels serve).

const std = @import("std");
const kernelSource = @import("../../../mlx/kernels.zig").source;

/// The attention side's launch (hc_mix's threadgroups, then each row group's hc_norm threadgroup).
pub const mix_norm_source = kernelSource("plugins/deepseek_v41/kernels/hc_mix_norm.metal");
/// The FFN side's launch; the header picks where the new stream goes.
pub const expand_norm_source = kernelSource("plugins/deepseek_v41/kernels/hc_expand_norm.metal");
pub const expand_norm_fresh_header = "#define CARRIER_INIT\n#define X1_AT(i) x1[i]\n";
pub const expand_norm_into_header = "#define CARRIER_INIT if (tid == 0 && l == 0) carrier[0] = 0;\n#define X1_AT(i) ((device T*)dst)[i]\n";

test "hc_fuse kernel texts name no address space for an input's pointer" {
    // MLX binds an input of fewer than 8 elements (`post`, `cpre` and `eps` at one row) as `const constant T*`, a larger
    // one as `const device T*`: a pointer taken from an input is `auto` (o1a1's `const device float*` from `post` did not
    // compile at one row); only `dst`, a stream region, is cast to be written.
    for ([_][]const u8{ mix_norm_source, expand_norm_source }) |src| {
        try std.testing.expect(std.mem.indexOf(u8, src, "device float*") == null);
        try std.testing.expect(std.mem.indexOf(u8, src, "constant float*") == null);
    }
    try std.testing.expect(std.mem.indexOf(u8, expand_norm_source, "const auto pr = post + l * HC;") != null);
    try std.testing.expect(std.mem.indexOf(u8, expand_norm_source, "const auto cb = comb + l * HC * HC;") != null);
}

test "hc_fuse kernel texts carry the stock kernels' statements" {
    // hc_mix.metal's accumulations and its mix, hc_norm.metal's collapse, rounding, sum and scale, the expand's fmas.
    for ([_][]const u8{
        "for (int k = tid; k < N; k += TG) {",
        "for (int j = 0; j < M; ++j) d[r][j] += v * fv[j];",
        "ss[r] += v * v;",
        "z[(l0 + r) * MIX + m0 + j] = a * metal::precise::rsqrt(b / float(N) + eps[0]) * scale[m0 + j] + base[m0 + j];",
        "for (int j = 0; j < HC; ++j) h += cp[j] * float(x[(size_t(l) * HC + j) * D + d]);",
        "const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);",
    }) |s| try std.testing.expect(std.mem.indexOf(u8, mix_norm_source, s) != null);
    for ([_][]const u8{
        "for (int j = 0; j < HC; ++j) acc = fma(cb[j * HC + i], sv[j], acc);",
        "const T o = T(fma(pr[i], xv, acc));",
        "h += cp[i] * float(o);",
        "h_out[o] = w[d] * T(float(h_out[o]) * inv);",
    }) |s| try std.testing.expect(std.mem.indexOf(u8, expand_norm_source, s) != null);
}
