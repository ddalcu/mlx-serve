//! Which media modality a `model_type` is, or none, as its media plugin declares it: what discovery (what is servable),
//! the media load arms (which engine slot) and `/v1/models` (each row's `modality`) read.

const std = @import("std");
const served = @import("../media_jobs/plugin/served.zig");

/// Detected from `config.json`'s `model_type` and carried on the load path so the registry installs the right engine
/// slot and the server dispatches the right endpoint.
pub const Modality = enum {
    image,
    audio,
    video,
    mesh,
    /// Typed decisions (`/v1/decisions`, Laya or Kev): scorers, not generators,
    /// but they ride the media plumbing (engine slot, inference-thread job).
    decision,
    /// Speaker diarization (Nemotron 3 Diarization): no route of its own, the
    /// speakers of a `diarized_json` transcription.
    diarization,

    pub fn capability(self: Modality) []const u8 {
        return switch (self) {
            .image => "image",
            .audio => "audio",
            .video => "video",
            .mesh => "3d",
            .decision => "decisions",
            .diarization => "diarization",
        };
    }

    /// Static, borrowed-static `ModelConfig.model_type` marker for each
    /// modality. Stable string literals (never freed) — `ModelConfig`
    /// treats `model_type` as borrowed-static, so a heap dupe is wrong here.
    pub fn modelType(self: Modality) []const u8 {
        return switch (self) {
            .image => "flux2",
            .audio => "qwen3_tts",
            .video => "AudioVideo",
            .mesh => "hunyuan3d_2_1",
            .decision => "laya",
            .diarization => "nemotron_diarization",
        };
    }
};

/// The modality of `model_type`, or null for an LM/embedding arch: its media plugin's (a type it claims, or the type
/// it reports for a pack claimed by its files, as `Modality.modelType`'s markers are).
pub fn fromType(model_type: []const u8) ?Modality {
    return served.modalityAs(Modality, served.byName(model_type) orelse return null);
}

// ── tests ──

test "every served media type has its plugin's modality, and an LM arch has none" {
    for (served.tables, served.types) |t, mt| try std.testing.expectEqual(served.modalityAs(Modality, t), fromType(mt).?);
    for ([_][]const u8{ "gemma4", "qwen3", "llama", "deepseek_v4", "bert", "qwen3_asr" }) |mt|
        try std.testing.expect(fromType(mt) == null);
    // Every marker the load path stamps reads back as its own modality.
    for (std.enums.values(Modality)) |m| try std.testing.expectEqual(m, fromType(m.modelType()).?);
}
