//! engines/mlx/kv/sliding.
//! Owns SLIDING_TRIM_UNBOUNDED, slidingTailSpan, sliding_block_trim_cached, slidingBlockTrimEnabled, sliding_block_trim_logged, sliding_ring_off_for_test, and 1 more.

const std = @import("std");
const env = @import("../../../env.zig");
const log = @import("../../../sdk/util/log.zig");
const ModelConfig = @import("../../../sdk/config/model.zig").ModelConfig;
const FUSED256_MIN_Q_LEN = @import("../../../sdk/blocks/attention/attn256.zig").FUSED256_MIN_Q_LEN;
const fused256Enabled = @import("../../../sdk/blocks/attention/attn256.zig").fused256Enabled;
const SlidingView = @import("../../../sdk/blocks/attention/kv_view.zig").SlidingView;

pub const SLIDING_TRIM_UNBOUNDED = @import("../../../sdk/bases/blocks/sliding.zig").SLIDING_TRIM_UNBOUNDED;

pub const slidingTailSpan = @import("../../../sdk/bases/blocks/sliding.zig").slidingTailSpan;

var sliding_block_trim_logged: bool = false; // one-shot log guard

/// Test seam: the full-buffer reference arm of the sliding-ring equivalence test.
pub var sliding_ring_off_for_test = false;

pub fn slidingViewFor(cfg: *const ModelConfig, total_kv: c_int, seq_len: c_int) SlidingView {
    // Keyed on the kernel's OWN floor: it declines everything below
    // FUSED256_MIN_Q_LEN, so a spec-verify block (4-8 wide) is never
    // band-masked in-kernel and is free to take the trim. The predicate used
    // to say `seq_len >= 2`, which declined the trim for exactly those blocks
    // and then fell through to the explicit-mask path anyway — gemma got
    // nothing from the trim at all.
    const band = fused256Enabled() and cfg.head_dim == 256 and seq_len >= FUSED256_MIN_Q_LEN;
    // A sliding ring holds only the tail, so it always trims.
    const width: usize = if (band) 1 else SLIDING_TRIM_UNBOUNDED;
    const span: u32 = if (cfg.has_sliding_window)
        slidingTailSpan(cfg.sliding_window, @intCast(seq_len), width)
    else
        0;
    const kv_len: c_int = if (span > 0) @min(total_kv, @as(c_int, @intCast(span))) else total_kv;
    // One-shot: an A/B arm is proven by an engagement line in its own log, not
    // by the env it was launched with. Fires only when the BLOCK half of the
    // trim did something — decode has always trimmed to the window.
    if (!sliding_block_trim_logged and span > cfg.sliding_window and kv_len < total_kv) {
        sliding_block_trim_logged = true;
        log.info("[sliding] block trim engaged: window={d} q_len={d} span={d} kv={d} of {d}\n", .{ cfg.sliding_window, seq_len, span, kv_len, total_kv });
    }
    return .{ .span = span, .kv_len = kv_len, .band_in_kernel = band };
}
