//! qwen4_exp MTP rerank wiring guard.
const std = @import("std");
const testing = std.testing;


// ── tests ──


test "the qwen4 rerank draft feeds the MIXER output, never the pre-mixer stream" {
    // The one way this change can be wrong without failing to compile. On the
    // sidecar arm `hidden_next` IS the lm_head's input, so the obvious wiring
    // (`draftSelect(xfm, step_out.hidden_next, ...)`) is correct there and
    // silently wrong on qwen4_exp, whose `hidden_next` is the pre-mixer
    // `[B,S,hc*H]` stream — 4x the width the head consumes. The chain must
    // therefore prefer `rerank_x` when a step publishes one.
    const source = @import("../../engines/mlx/generate.zig").all_source;
    const build_at = std.mem.indexOf(u8, source, "fn mtpChainBuild(") orelse return error.MissingChainBuild;
    const end = std.mem.indexOfPos(u8, source, build_at, "\n/// Fire the chain's built graphs") orelse source.len;
    const body = source[build_at..end];

    const sel = std.mem.indexOf(u8, body, "head.draftSelect(") orelse return error.MissingDraftSelect;
    const line_start = std.mem.lastIndexOfScalar(u8, body[0..sel], '\n') orelse 0;
    const line_end = std.mem.indexOfScalarPos(u8, body, sel, '\n') orelse body.len;
    const call = body[line_start..line_end];
    // The rerank input is the resolved vector, never `hidden_next` directly.
    try testing.expect(std.mem.indexOf(u8, call, "rerank_x") != null);
    try testing.expect(std.mem.indexOf(u8, call, "hidden_next") == null);
    try testing.expect(std.mem.indexOf(u8, body, "step_out.rerank_x.ctx != null") != null);

    // ...and a `.mixed` step is what asks for it: a rerank draft must not
    // collapse onto `.none` (the history-append mode), which produces no
    // mixer output at all.
    try testing.expect(std.mem.indexOf(u8, body, "if (use_rerank) .mixed else .logits") != null);

    // The qwen4 arm maps `.mixed` onto the projection that SKIPS the lm_head
    // but still runs the mixer. If it ever mapped to `.none`, `rerank_x` would
    // be null and the chain would silently hand over the pre-mixer stream.
    const step_at = std.mem.indexOf(u8, source, "fn qwen4Step(") orelse return error.MissingQwen4Step;
    const step_end = std.mem.indexOfPos(u8, source, step_at, "\n    /// One head forward") orelse source.len;
    const step_body = source[step_at..step_end];
    try testing.expect(std.mem.indexOf(u8, step_body, ".mixed => .mixed_last_row") != null);
    try testing.expect(std.mem.indexOf(u8, step_body, ".rerank_x = out.mixed") != null);

    // And the owned mixer vector is freed by the chain, not leaked per step.
    try testing.expect(std.mem.indexOf(u8, body, "mlx_array_free(step_out.rerank_x)") != null);
}
