//! The rows a sliding layer's ring cache keeps (`abi.CacheFacts.sliding_ring`). MLX-free: plugin configs read it.

/// Rows past the window: spec rollback plus the prefix-cache matches that land near an entry's end.
pub const slack: u32 = 2048;

/// The ring a layer attending its last `window` keys keeps; 0 (no window) = the full buffer.
pub fn rows(window: u32) u32 {
    return if (window > 0) window + slack else 0;
}
