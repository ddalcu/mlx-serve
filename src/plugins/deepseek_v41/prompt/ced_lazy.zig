//! MLX_SERVE_DSV41_CED_LAZY (on by default, =0 off; exact): a prompt chunk's CED replay (`cedTail`: the decoder layers over
//! its last window, run for its last row's logits and the decoder's window rows) is left unevaluated when another
//! replaying chunk of the armed prompt follows, and dropped when that chunk comes (`State.ced_lazy`). That chunk's
//! replay writes every window row a later read reaches (its band stops at its own first row), and the engine reads no
//! prompt chunk's logits. Any other reader (a decode or verify step, a chunk that runs the full forward, a snapshot,
//! the returned logits) evaluates the replay as it stands: the same graph, so the same words.
//!
//! A dropped replay leaves the decoder layers' window rings and the DSpark stages' rings as zeros before the next
//! replay writes them: their slots outside the window (16 of 144) then hold zeros where the eager path holds the dropped
//! replay's rows, which no read reaches. Gain: the 17.6K card's chunk loop (8,192, 8,192, 1,254 rows) replays three
//! tails, and two of them (20 decoder layers over 128 rows each) never run.

const std = @import("std");
const env = @import("../../../env.zig");
const lev = @import("../levers.zig");

pub fn on() bool {
    return lev.onUnlessZero(.CED_LAZY);
}

/// Whether a replaying chunk `ids` at `p0` leaves its replay to a reader: it is the armed `prompt`'s, at least a
/// replaying chunk's rows of the prompt (`window + min_skip`) remain before its last token (which the engine forwards
/// alone), and no snapshot is due at the chunk's end.
pub fn defers(prompt: []const u32, ids: []const u32, p0: usize, window: usize, min_skip: usize, snap_at: ?usize) bool {
    const p1 = p0 + ids.len;
    if (p1 > prompt.len or !std.mem.eql(u32, ids, prompt[p0..p1])) return false;
    if (snap_at) |at| if (at == p1) return false;
    return prompt.len -| (p1 + 1) >= window + min_skip;
}

test "ced_lazy defers: a chunk of the armed prompt with a replaying chunk's rows left before the last token, no snapshot at its end" {
    var prompt: [1000]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i % 97);
    const w = 128;
    const skip = 256;
    // 1000 - 1 - 600 = 399 rows remain: a replaying chunk's 384 fit; at 616 only 383 do.
    try std.testing.expect(defers(&prompt, prompt[200..600], 200, w, skip, null));
    try std.testing.expect(defers(&prompt, prompt[200..615], 200, w, skip, null));
    try std.testing.expect(!defers(&prompt, prompt[200..616], 200, w, skip, null));
    // The chunk the last token follows, a snapshot due at its end, ids not the armed prompt's, or no prompt armed.
    try std.testing.expect(!defers(&prompt, prompt[0..999], 0, w, skip, null));
    try std.testing.expect(!defers(&prompt, prompt[200..600], 200, w, skip, 600));
    try std.testing.expect(defers(&prompt, prompt[200..600], 200, w, skip, 999));
    var other = prompt;
    other[300] +%= 1;
    try std.testing.expect(!defers(&prompt, other[200..600], 200, w, skip, null));
    try std.testing.expect(!defers(&.{}, prompt[200..600], 200, w, skip, null));
    try std.testing.expect(!defers(prompt[0..500], prompt[200..600], 200, w, skip, null));
}
