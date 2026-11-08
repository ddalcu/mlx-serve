//! What the example engine serves and says, MLX-free: the model folders it names (no built-in serves them while it is
//! loaded), the `model_type` its probe reads, and its reply in the byte tokenizer's ids.

const std = @import("std");

/// The model folders it serves, by name.
pub const models = [_][]const u8{"_example_engine"};
/// What `probe` asks of a model's config.json.
pub const model_type = "_example_engine";
/// Streamed one byte a token, over and over: a request without an EOS runs to its `max_tokens`.
pub const reply = "Hello from an engine plugin. ";
/// Its one option: another reply.
pub const reply_option = "_EXAMPLE_ENGINE_REPLY";

/// Token `i` of `text`.
pub fn token(text: []const u8, i: u32) u32 {
    return text[i % text.len];
}

test "the reply cycles, one byte a token" {
    try std.testing.expectEqual(@as(u32, 'H'), token(reply, 0));
    try std.testing.expectEqual(@as(u32, 'H'), token(reply, reply.len));
}
