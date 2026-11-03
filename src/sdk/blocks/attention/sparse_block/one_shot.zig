//! Log-once gates for the sparse-block kernels' engagement lines, keyed by the query-width bucket.

const FUSED256_MIN_Q_LEN = @import("../attn256.zig").FUSED256_MIN_Q_LEN;

/// One-shot gate keyed by a small bit index. `take` returns true the FIRST
/// time each bit is asked for.
pub const OneShotBits = struct {
    seen: u32 = 0,

    pub fn take(self: *OneShotBits, bit: u5) bool {
        const m: u32 = @as(u32, 1) << bit;
        if (self.seen & m != 0) return false;
        self.seen |= m;
        return true;
    }
    pub fn reset(self: *OneShotBits) void {
        self.seen = 0;
    }
};

/// Query-width bucket: decode (1), verify (2 .. FUSED256_MIN_Q_LEN-1),
/// prefill (>= FUSED256_MIN_Q_LEN). The meters are one-shot PER BUCKET so a
/// decode-width engagement at 16k cannot vouch for a verify-width call at
/// 62k — which is exactly the confusion a single global bool created.
pub fn widthBucket(seq_len: c_int) u2 {
    if (seq_len <= 1) return 0;
    if (seq_len < FUSED256_MIN_Q_LEN) return 1;
    return 2;
}
