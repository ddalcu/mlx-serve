//! The DFlash yield-gate minimum.

const std = @import("std");
const generate_mod = @import("../generate.zig");
const testing = std.testing;

/// Scale the measured M5/block-16 break-even to the effective number of
/// draft positions. `block_size` includes the always-emitted anchor. Thinking
/// is the actual resolved request mode; tools are neither necessary nor
/// sufficient for a reasoning preamble.
pub fn dflashGateMinimum(block_size: u32, enable_thinking: bool, moe_target: bool) f32 {
    const drafts = block_size -| 1;
    if (drafts == 0) return 0;
    const calibrated_min = if (enable_thinking)
        generate_mod.Generator.DFLASH_THINKING_GATE_MIN_ACCEPTED_PER_ROUND
    else
        generate_mod.Generator.DFLASH_GATE_MIN_ACCEPTED_PER_ROUND;
    const scaled = calibrated_min * @as(f32, @floatFromInt(drafts)) / 15.0;
    // The bar is really "round cost / serial step cost", and the whole
    // calibration above was measured on DENSE trunks where one verify forward
    // costs about one serial step. A sparse trunk breaks that: an A1B MoE
    // decodes ~1B of weights per token but its verify reads every expert the
    // block's positions route to, so a round costs far more than a step while
    // the bar stayed at 0.53 and nothing ever disabled.
    //
    // Measured LFM2.5-8B-A1B + its DSpark sidecar, M4 Max, block 5, greedy:
    // novel prose accepts 1.40/round and runs 171 tok/s against 199 serial
    // (round cost = 1.40 x 199/171 = 1.63 steps), while an echo prompt accepts
    // 4.00 and runs 273. A floor of 1.8 disables the losing class after its
    // three warmup rounds and leaves the winning one untouched.
    if (!moe_target) return scaled;
    return @max(scaled, generate_mod.Generator.DFLASH_MOE_GATE_MIN_ACCEPTED_PER_ROUND);
}

// ── tests ──

test "DFlash gate policy follows effective block width and resolved thinking" {
    // block_size includes the always-emitted anchor, so block 16 has 15 draft
    // positions and block 7 has 6. The M5 calibration is normalized to that
    // actual draft width instead of being imposed as an absolute threshold on
    // machines without the wide verification lane.
    try testing.expectApproxEqAbs(@as(f32, 2.0), dflashGateMinimum(16, false, false), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dflashGateMinimum(16, true, false), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.8), dflashGateMinimum(7, false, false), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 0.4), dflashGateMinimum(7, true, false), 0.0001);
    try testing.expectEqual(@as(f32, 0), dflashGateMinimum(1, false, false));

    // A SPARSE target's verify reads every expert its block routes to, so the
    // dense width scaling under-bars it: at block 5 the scaled value is 0.53
    // and LFM2.5-8B-A1B measured break-even at 1.63 accepted/round. The floor
    // binds there and in the thinking arm, and never lowers a bar the width
    // scaling already set higher.
    try testing.expectApproxEqAbs(@as(f32, 1.8), dflashGateMinimum(5, false, true), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 1.8), dflashGateMinimum(5, true, true), 0.0001);
    try testing.expectApproxEqAbs(@as(f32, 2.0), dflashGateMinimum(16, false, true), 0.0001);
    try testing.expectEqual(@as(f32, 0), dflashGateMinimum(1, false, true));
}
