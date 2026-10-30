//! MiMo's split-K decode attention, now `sdk/blocks/attention/dec_attn.zig`: the names this plugin and the engine import.

const moved = @import("../../sdk/blocks/attention/dec_attn.zig");

pub const MAX_ROWS = moved.MAX_ROWS;
pub const attention = moved.attention;
