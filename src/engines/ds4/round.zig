//! One ds4 decode round: the next token drawn from the session's logits, and ds4's speculative arm that commits it
//! with the drafts it verifies after it.

const std = @import("std");

/// ds4 MTP speculative-decode defaults. `draft_tokens` MUST be > 1 to engage (ds4 gates on it); margin 3.0 mirrors
/// ds4's own default acceptance margin. A round holds the verified token + up to ds4's 16-draft cap.
pub const mtp_draft_tokens: c_int = 4;
pub const mtp_margin: f32 = 3.0;
pub const max_round_tokens: usize = 17;

/// Whether a ds4 decode step should use speculative decode: the engine reports >1 ready draft tokens and sampling
/// is greedy (ds4's spec path is argmax-based; temp>0 falls back to the normal sampler). The draft count is the
/// readiness signal for BOTH support kinds (legacy MTP reports its count only when `mtp_ready`, DSpark its block
/// size only when `--dspark` armed the runtime), so a `has_mtp` conjunct would leave DSpark unreachable. A
/// support-GGUF draft verifies by argmax, so it serves greedy requests only; the in-checkpoint head has a sampled arm.
pub fn shouldSpeculate(draft_tokens: c_int, temperature: f32, embedded: bool) bool {
    return draft_tokens > 1 and (temperature <= 0.0 or embedded);
}

pub const Draw = struct {
    temperature: f32,
    top_k: u32,
    top_p: f32,
    min_p: f32,
    rng: *u64,

    fn topK(d: Draw) c_int {
        return @intCast(@min(d.top_k, std.math.maxInt(c_int)));
    }
};

/// The token the session's logits give next: argmax at temperature <= 0, else drawn. Negative: the sampler failed.
pub fn draw(session: anytype, d: Draw) i32 {
    if (d.temperature <= 0.0) return session.argmax();
    return session.sample(d.temperature, d.topK(), d.top_p, d.min_p, d.rng);
}

/// Commits the drawn `first` and the drafts ds4 accepts after it, at most `remaining` tokens, advancing the
/// session's KV itself (no separate eval). The committed ids, `first` first, in `out`.
pub fn speculate(engine: anytype, session: anytype, first: i32, d: Draw, remaining: u32, out: *[max_round_tokens]i32) ![]const i32 {
    const cap: i32 = @intCast(@max(@as(u32, 1), @min(remaining, std.math.maxInt(i32))));
    const n = if (d.temperature <= 0.0)
        try session.evalSpeculative(first, cap, engine.eosToken(), out[0..])
    else
        try session.evalSpeculativeSampled(first, cap, engine.eosToken(), d.temperature, d.topK(), d.top_p, d.min_p, d.rng, out[0..]);
    return out[0..@intCast(@max(n, 0))];
}

// ── tests ──

test "shouldSpeculate: >1 draft tokens + greedy (legacy MTP and DSpark), sampled only on the in-checkpoint head" {
    try std.testing.expect(shouldSpeculate(4, 0.0, false));
    try std.testing.expect(shouldSpeculate(2, -1.0, false));
    // DSpark's block size (e.g. 16) engages the same way.
    try std.testing.expect(shouldSpeculate(16, 0.0, false));
    // No ready draft (0), or 1 draft token, or sampling → regular decode.
    try std.testing.expect(!shouldSpeculate(0, 0.0, false));
    try std.testing.expect(!shouldSpeculate(1, 0.0, false));
    try std.testing.expect(!shouldSpeculate(4, 0.7, false));
    // The in-checkpoint head (Qwen3.8 Flash Next, GLM 5.x) drafts under sampling too.
    try std.testing.expect(shouldSpeculate(2, 0.7, true));
    try std.testing.expect(!shouldSpeculate(0, 0.7, true));
}
