//! Continuing a partial assistant reply: `continuationRequested`, `continuationPrefill`.

const std = @import("std");

const Message = @import("types.zig").Message;

/// Whether this conversation SHAPE is a continuation: it ends with an
/// assistant message carrying prose. The one predicate every surface asks, so
/// "what counts as continuable" cannot differ between the endpoint that
/// implies it and the endpoint that takes a flag — and so the answer matches
/// what `continuationPrefill` will actually do with it.
pub fn continuationRequested(messages: []const Message) bool {
    return continuationPrefill(messages, true) != null;
}

/// The partial assistant reply a continuation request is asking to extend, or
/// null when this is an ordinary turn.
///
/// The flag describes the SHAPE of the request, so a client that sets it on a
/// conversation not ending in an assistant message gets an ordinary turn —
/// never the user's own words committed into the assistant's mouth.
///
/// The prefill is trimmed of trailing whitespace: a tokenizer merges a trailing
/// space into the word that follows, so a prompt ending in one tokenizes
/// differently than the same text does mid-reply. (Anthropic's API rejects such
/// a prefill outright; trimming keeps the request serviceable, and only the
/// PROMPT is trimmed — the stored reply keeps its whitespace.)
pub fn continuationPrefill(messages: []const Message, continue_final: bool) ?[]const u8 {
    if (!continue_final or messages.len == 0) return null;
    const last = messages[messages.len - 1];
    if (!std.mem.eql(u8, last.role, "assistant")) return null;
    // A reply that ended in a tool call has no prose to resume — splicing a
    // continuation onto it would re-run the parse chain over half a call.
    if (last.tool_calls != null) return null;
    const trimmed = std.mem.trimEnd(u8, last.content, " \t\r\n");
    return if (trimmed.len > 0) trimmed else null;
}
