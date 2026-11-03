//! What a request asks of a spoken reply (`engines/mlx/generate/speech.zig`); unset fields are the model's
//! `SpeechFacts`.

pub const Mode = enum { text, audio, interleaved };

pub const Request = struct {
    /// Null: from the prompt's last token, as the reference.
    mode: ?Mode = null,
    temperature: ?f32 = null,
    top_k: ?u32 = null,
    repetition_penalty: ?f32 = null,
    /// A fixed MaskGIT reveal order, frame after frame (tests); empty = random.
    order: []const u32 = &.{},
};
