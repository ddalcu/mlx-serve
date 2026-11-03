//! Our SDK's root, reached by relative `@import` from inside the main module until P4/P6 makes it a module (the
//! build's `sdk` module is still mlx-stream's). MLX-free, so a plugin's `config.zig` may import it.

pub const plugin = @import("plugin.zig");
pub const Plugin = plugin.Plugin;
pub const Kind = plugin.Kind;
pub const claimsByType = plugin.claimsByType;
pub const mediaClaimsByType = plugin.mediaClaimsByType;

/// The tiny pack a plugin hands the conformance suite (`conformanceFixture`).
pub const fixture = @import("testing/fixture.zig");
/// A fixture's safetensors built tensor by tensor (F32, BF16, packed words).
pub const tensors = @import("testing/tensors.zig");
/// A fixture's byte-level `tokenizer.json`.
pub const byte_tokenizer = @import("testing/byte_tokenizer.zig");
/// A fixture's tiny qwen3_5 trunk (config + weights).
pub const tiny_qwen35 = @import("testing/tiny_qwen35.zig");

test {
    _ = plugin;
    _ = fixture;
    _ = tensors;
    _ = byte_tokenizer;
}
