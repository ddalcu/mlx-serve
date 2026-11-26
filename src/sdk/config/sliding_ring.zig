//! The rows a sliding layer's ring cache keeps (`abi.CacheFacts.sliding_ring`). MLX-free: plugin configs read it.

/// Rows past the window: spec rollback plus the prefix-cache matches that land near an entry's end.
pub const slack: u32 = 2048;

/// The ring a layer attending its last `window` keys keeps; 0 (no window) = the full buffer.
pub fn rows(window: u32) u32 {
    return if (window > 0) window + slack else 0;
}

/// A ring drops rows in whole blocks of this many (`KVCache.ring_drop_align`).
pub const drop_align: u32 = 32;

/// Rows a ring's copy at the prompt's end holds (`KVCache.ringTail`): the window, `tail_slack` more (a next prompt
/// matches a few tokens short of the last one's end) and up to one drop block, where the copy starts.
pub const tail_slack: u32 = 32;

pub fn tailRows(window: u32) u32 {
    return if (window > 0) window + tail_slack + drop_align else 0;
}
