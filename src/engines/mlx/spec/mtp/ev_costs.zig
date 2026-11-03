//! MTP expected-value cost tables per profile.
const std = @import("std");
const env = @import("../../../../env.zig");
const mtp_mod = @import("../../../../sdk/spec/mtp/mtp.zig");


/// Round-cost model in units of the fixed round cost (verify-forward
/// floor + round eval/read + commit ≈ 1.0 ≈ 32 ms on the 27B since the
/// deferred history append). Ratios are machine-stable where absolute ms
/// are not. Refit via diag mtp-trace on Qwen3.6-27B GDN (M4 Max,
/// saturated fixed depths, same-session sweep AFTER the
/// deferred-append round shape landed): T(1)=42.0, T(3)=62.6,
/// T(6)=111.6, T(7)=114.9 ms — the surface is PIECEWISE: ~10.3 ms
/// marginal per position in the flat verify region (seq <= 4), ~16.3
/// ms/pos for positions 4-6, and position 7 nearly free (+3.3 ms —
/// verify seq 8 rides the same row tile as 5-7), averaged into
/// per_pos_hi. ATTRIBUTION (the GDN-vs-qmm µbench in transformer.zig,
/// diag gdn-ubench): the ladder is ~90% qmm ROW-COUNT cost — the
/// GDN recurrence kernel is nearly flat over verify widths (0.13→0.26 ms
/// per dispatch, T 2→64) and contributes <1 ms/round; the earlier
/// "GDN sequential width ramp" reading was a mis-attribution.
/// The old linear ~1.5 ms/pos model came from a depth-6 run whose
/// windowed controller was silently demoting underneath — never fit
/// costs from a run whose realized m_avg you didn't check.
pub const MtpEvCosts = struct {
    draft: f32, // one sequential MTP-head step (fwd + draft lm_head)
    per_pos_lo: f32, // marginal verify+capture per position, flat region
    per_pos_hi: f32, // ... beyond flat_max (qmm row-tile ramp)
    flat_max: u32, // last draft index in the flat verify region
    sync: f32, // the chunk-A confidence read-back
    /// First draft depth whose verify forward lands on the M5 NAX tile.
    /// Zero disables the third region. Depth k verifies k+1 rows, so the
    /// default NAX M=8 takeover starts at draft position 7.
    nax_from: u32 = 0,
    per_pos_nax: f32 = 0.0,
    /// Serial-floor cost in ms at kv ~= 0 (the probe's width-1 rung).
    /// Zero — every hand-typed table — makes the kv term below a no-op,
    /// so the tables keep behaving exactly as they did.
    floor_ms: f32 = 0,
    /// Context length the marginals above were FITTED AT. The kv term
    /// scales relative to THIS, not to zero: refit #4 was measured at 8K,
    /// so its 0.20 already contains 8K of KV read and re-scaling it from
    /// a kv~=0 floor would discount it twice. Zero = unknown, which
    /// disables the kv term for that surface rather than guessing.
    kv_ref_tokens: u32 = 0,
    /// Learned per-KV-token round cost B (ms/token), the term that makes
    /// wide speculation approach FREE at long context.
    ///
    /// A verify forward of width k reads the weights once and the KV
    /// once, both SHARED across all k query rows; only arithmetic scales
    /// with k. So T(k, L) ~= W + B*L + C(k) and T(k,L)/T(1,L) -> 1 as L
    /// grows: the optimal width RISES with context, and every constant
    /// this controller ever shipped was fitted at ONE context length.
    /// Learned online rather than probed — a boot probe measures at
    /// kv ~= 0, where B is invisible, and probing it would mean
    /// allocating a 32k KV cache purely to time it.
    kv_ms_per_token: f32 = 0,
};
/// refit #4, AFTER the hd-256 causal sdpa split
/// (`splitCausalSdpa` — dense verify q 6..9 now rides the vector path,
/// invalidating the ramp the old hi partially priced): same-session
/// saturated ECHO sweep, Jundot oQ4e 27B @8K cold reps
/// (--prefix-cache-entries 0), M4 Max, forced depths, saturated (m_avg==N) trace windows only — T(1)=44.6,
/// T(2)=51.0, T(3)=59.2, T(4)=68.2, T(6)=95.4, T(8)=142.3 ms → floor
/// ≈ 38.2 ms. Composite marginals in floor units: k<=4 ≈ 0.20 (6.4-9.0
/// ms/pos — flat_max moves 3 → 4, the old hi over-priced k4 at 0.34 and
/// under-drafted it on moderate content), k5-6 ≈ 0.36 (13.6 ms/pos),
/// k7-8 ≈ 0.62 (23.5 ms/pos — the plain-SIMD verify-qmm register cliff
/// at M 8/9, expressed through the generic third region; only reachable
/// when --mtp-max-depth forces past the generic cap of 6). The G17 NAX
/// tables below predate the sdpa split — refit them on an M5 when one
/// is available. draft/verify split not separately identifiable; only
/// the sums enter the controller.
/// (Refit #3, post round-pipelining, for the record:
/// T(1)=45.4, T(2)=53.3 → .10/.10/.24@3/.01 on a ~37.9 ms floor.
/// Refit #2, post-verify-qmm: T(1)=44.6, T(3)=54.4,
/// T(6)=89.9 → .06/.06/.24/.02 on a ~40 ms floor.)
pub const MTP_EV_DEFAULT_COSTS: MtpEvCosts = .{ .draft = 0.10, .per_pos_lo = 0.10, .per_pos_hi = 0.26, .flat_max = 4, .sync = 0.01, .nax_from = 7, .per_pos_nax = 0.52 };
/// M5 Max/G17 refit, same-session saturated fixed-depth
/// sweep after the NAX m16 verify lane landed: T(1..4) ~= 41.35 ms,
/// T(6)=62.15 ms, T(8)=68.39 ms. In floor units this identifies
/// draft+hi ~= .21 and draft+nax ~= .10; T(8)/T(6) is reproduced by
/// 2.19/1.99 = 1.1005 (measured 1.1004). The profile is selected only
/// for the calibrated dense Qwen3.6-27B homogeneous affine-4/gs-64
/// checkpoint with its native affine-8/gs-32 sidecar and successfully
/// built affine-3/gs-64 draft head, when the trunk lm_head routes both
/// M=8 and M=9 through NAX; every other combination retains DEFAULT.
pub const MTP_EV_G17_NAX_COSTS: MtpEvCosts = .{
    .draft = 0.06,
    .per_pos_lo = 0.06,
    .per_pos_hi = 0.15,
    .flat_max = 3,
    .sync = 0.02,
    .nax_from = 7,
    .per_pos_nax = 0.04,
};
/// M5 Max/G17 affine-4/gs-32 sidecar refit. A saturated
/// fixed-depth sweep gave T(1)=36.04, T(3)=43.01, T(6)=62.56, and
/// T(8)=66.06 ms. The fitted composite marginals (`draft + verify`) are
/// .107/.200/.054; depths 4 and 5 independently validate the rounded
/// .11/.20/.05 surface after matching-baseline correction. The split
/// between draft and verify is not separately identifiable.
pub const MTP_EV_G17_NAX_Q4_GS32_COSTS: MtpEvCosts = .{
    .draft = 0.03,
    .per_pos_lo = 0.08,
    .per_pos_hi = 0.17,
    .flat_max = 3,
    .sync = 0.02,
    .nax_from = 7,
    .per_pos_nax = 0.02,
};
/// M5 Max/G17 uniform affine-4/gs-64 Qwen3.8-27B refit.
/// Temperature-gated, reversed two-pass saturated echo gave T(1)=
/// 33.61-33.91, T(3)=38.455-38.465, T(6)=56.555-57.000, and NAX T(8)=
/// 61.890-61.925 ms/round. NAX-off T(8)=88.535-88.610 while depth-6
/// on/off stayed within 0.8%, isolating the M>=8 lane. A 31.4 ms fitted
/// floor gives rounded composite marginals .075/.195/.08. The profile
/// stays distinct because this trunk is uniform q4, not mixed q4/q5/q6.
pub const MTP_EV_G17_NAX_Q4_GS64_COSTS: MtpEvCosts = .{
    .draft = 0.02,
    .per_pos_lo = 0.055,
    .per_pos_hi = 0.175,
    .flat_max = 3,
    .sync = 0.01,
    .nax_from = 7,
    .per_pos_nax = 0.06,
};
/// M5 Max/G17 uniform affine-6/gs-64 Qwen3.8-27B refit.
/// Temperature-gated, six-pass counterbalanced calibration gave median
/// traced T(1)=44.98, T(3)=49.895, T(6)=64.495, and NAX T(8)=76.79
/// ms/round. A 42.52 ms fitted floor yields rounded composite marginals
/// .06/.115/.145 and reproduces T(8)/T(6): 1.815/1.525=1.1902 versus
/// 1.1907 measured. Depth 8 was 108.86 tok/s versus 102.93 at depth 7.
pub const MTP_EV_G17_NAX_Q6_GS64_COSTS: MtpEvCosts = .{
    .draft = 0.02,
    .per_pos_lo = 0.04,
    .per_pos_hi = 0.095,
    .flat_max = 3,
    .sync = 0.01,
    .nax_from = 7,
    .per_pos_nax = 0.125,
};
/// M5 Max/G17 uniform affine-8/gs-64 Qwen3.8-27B refit.
/// Five settled counterbalanced pairs found the q8-specific NAX takeover
/// at verify M=7: depth 6 rose from 73.42 to 77.89 tok/s (+6.09%), while
/// M=6 was -1.14%. Median traced NAX T(6)=85.94 ms and T(8)=90.68 ms over
/// the 53.0575 ms floor give normalized 1.62/1.71. The additive controller
/// cannot encode the one-time M7 kernel transition separately, so k=5..6
/// are a positive .19 bridge and the subsequent k=7..8 NAX marginal is
/// .045. This reproduces both measured endpoints without negative costs.
/// The profile is revoked when q8 NAX is off.
pub const MTP_EV_G17_NAX_Q8_GS64_COSTS: MtpEvCosts = .{
    .draft = 0.005,
    .per_pos_lo = 0.055,
    .per_pos_hi = 0.185,
    .flat_max = 4,
    .sync = 0.01,
    .nax_from = 7,
    .per_pos_nax = 0.04,
};
/// M5 Max/G17 oQ4e mixed affine q4/q5/q6, gs64 refit.
/// Saturated deterministic echo, same-session fixed widths:
/// T(1)=34.37, T(3)=40.31, T(6)=61.20, T(8)=62.79 ms/round.
/// Normalizing the inferred 31.40 ms floor gives composite marginals
/// .095/.220/.025. The split below is arbitrary but positive; only
/// `draft + per_pos_*` enters the controller. The profile is selected
/// solely for the exact resident oQ4e layer-role fingerprint.
pub const MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS: MtpEvCosts = .{
    .draft = 0.02,
    .per_pos_lo = 0.075,
    .per_pos_hi = 0.20,
    .flat_max = 3,
    .sync = 0.01,
    .nax_from = 7,
    .per_pos_nax = 0.005,
};
/// qwen4_exp (Qwen3.8-Flash-Next 125B-A6B) on G17, M5 Max 40-core refit
/// (/28). Fitted at ~8.5k context like every prior surface
/// (a short-context fit planned too shallow at 8.5k, measured -4% —
/// every constant this controller ever shipped was fitted at ONE
/// context length, and 8K is the house one). Forced-depth saturated
/// echo (the test_mtp_equivalence echo appended to ~8.3k of neutral
/// filler, temp 0, `--prefix-cache-entries 0`, round-cost persist off),
/// depths {1,2,3,4,6} x two reversed passes x 3 reps, medians of
/// per-request round_ms: T(1)=25.41, T(2)=31.18, T(3)=35.70,
/// T(4)=41.33, T(6)=53.58 ms -> 20.34 ms floor, composite
/// (draft + per-position) marginals .257/.303 (SSE 0.28), shipped
/// house-style below as `.draft = 0.02` plus per-position .237/.283 —
/// composite minus draft, the same split every calibrated surface
/// carries. Short-context fit for the record: T(1..4,6,8)=
/// 23.61/28.12/33.91/38.87/50.16/62.70 -> floor 18.24, composite
/// .283/.310/.344 (the .344 ships as per-position `.325`).
/// A verify row on this arch is BYTES — a second row reads its own
/// experts — so the marginals sit 3-5x above the dense-sidecar surfaces
/// and there is NO NAX flattening: routed expert banks never ride the
/// vqmm NAX lane (that is the M5 plan's grouped-expert item, and this
/// table is its before-number). `per_pos_nax` carries the short-context
/// k>=7 STEEPENING (+11%/pos, a measurement not a discount) and is
/// inert under this profile's default cap of 6. sync stays nominal per
/// the dry-spell-gate doctrine (measured chunk-boundary sync ~1.7 ms is
/// governed by the realized-rate gate, not the prior). draft/verify
/// split arbitrary but positive; only the sums enter the controller.
pub const MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS: MtpEvCosts = .{
    .draft = 0.02,
    .per_pos_lo = 0.237,
    .per_pos_hi = 0.283,
    .flat_max = 4,
    .sync = 0.02,
    .nax_from = 7,
    .per_pos_nax = 0.325,
};

/// M5 Max mixed4/8 refit after the single-stream verifier
/// fast paths. Affine8 target/head KV, 3-bit draft shortlist, ~8.6k
/// context; forced depths {1,2,3,4,5,6,8}, three echo repetitions in
/// each of two reversed boot orders. Median full-round EMAs (ms):
/// 24.145/28.970/34.035/38.470/44.310/51.970/61.600.
/// Fit floor=19.037 and position slopes=4.983/8.018/4.815 ms;
/// max relative residual 1.30%. Depth6 verifies seven positions and
/// leaves the optimized width2..6 envelope, hence flat_max=5.
/// Only composite marginals are identifiable; retain draft=.02 and
/// nominal sync=.02. Runtime fingerprint and KV guards live in mtp.zig.
/// Evidence and reproduction: notes/MTP-SINGLE-VERIFY.md.
pub const MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS: MtpEvCosts = .{
    .draft = 0.02,
    .per_pos_lo = 0.241753,
    .per_pos_hi = 0.401180,
    .flat_max = 5,
    .sync = 0.02,
    .nax_from = 7,
    .per_pos_nax = 0.232929,
};

/// The cost surface measured for this runtime profile.
pub fn mtpEvCosts(profile: mtp_mod.MtpCostProfile) MtpEvCosts {
    return switch (profile) {
        .generic => MTP_EV_DEFAULT_COSTS,
        .g17_nax_q8_gs32 => MTP_EV_G17_NAX_COSTS,
        .g17_nax_q4_gs32 => MTP_EV_G17_NAX_Q4_GS32_COSTS,
        .g17_nax_q4_gs64 => MTP_EV_G17_NAX_Q4_GS64_COSTS,
        .g17_nax_q6_gs64 => MTP_EV_G17_NAX_Q6_GS64_COSTS,
        .g17_nax_q8_gs64 => MTP_EV_G17_NAX_Q8_GS64_COSTS,
        .g17_nax_oq4e_q4_gs64 => MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS,
        .g17_nax_qwen4_q4_gs64 => MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS,
        .g17_nax_qwen4_mixed_4_8_gs64 => MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS,
    };
}
