//! `MLX_SERVE_DSV41_FLASH_ATTN` (read at load; unset, empty or `1`: on; `0`: off, the exact gather served; anything
//! else is refused by name): a wide prompt layer's attention span as one kernel (kernels/attn_flash.metal) in place of
//! the gather's three launches (qk_t, sm, pv) and the ring's scores and probabilities (scratch_attn.zig).
//!
//! One threadgroup is a query row and `hgn` groups of 16 heads, which share the row's keys (one KV head; the keys are the
//! values). The keys go in blocks of 128 (the window's rows, then the picks in their order); per block each simdgroup
//! scores its 32 keys for its 16 heads on the tensor units, the block's max joins the running max, exp(scale * s - m)
//! goes to threadgroup memory as f16, and the simdgroup accumulates its 128 output dims over the block, its f32
//! accumulator scaled by exp(m_old - m_new). The sink starts the state (m = sink, l = 1). Epilogue: O / l, the last 64
//! dims inverse-roped, stored grouped, or straight into the run region as pv's heads twin does.
//!
//! NOT exact: the probabilities round to f16 against the running max before the division (the gather's sm divides
//! first), the accumulator is rescaled in f32, and the sum's order differs.

pub const plan = @import("flash_plan.zig");
pub const Mode = plan.Mode;
pub const Error = plan.Error;
pub const parse = plan.parse;
pub const hgn = plan.hgn;
pub const source = plan.source;
pub const regionTwin = plan.regionTwin;

/// Off until the plugin's load configures it (on unless the setting says otherwise): a harness that loads the model
/// without the plugin's configure keeps the gather's words.
/// This file's process state, one copy across `-Dunits` units.
const State = struct {
    mode: Mode = .off,
    explicit: bool = false,
};
pub const shared_state = @import("../../../sdk/util/shared.zig").Shared(State, "dsv41_flash_attn", .{});
pub inline fn ps() *State {
    return shared_state.get();
}
/// The setting named on itself (`=1`), not by default: a span the kernel cannot take is refused, not the gather's.

pub fn configure(value: ?[]const u8) Error!void {
    ps().mode = try parse(value);
    ps().explicit = plan.explicitOn(value);
}

