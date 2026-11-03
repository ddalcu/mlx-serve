//! EmbeddingGemma 2: the Gemma 4 text trunk as a bidirectional encoder (`gemma4/bidirectional.zig`): per-layer inputs
//! from the embeddings alone, the inclusive band, image and video rows spliced at their placeholders, the
//! `embedding_projection` head behind `EmbedHooks.project`. Served on `/v1/embeddings`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const encoder = @import("../gemma4/bidirectional.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &encoder.bill,
    .load = encoder.loadAs(.embedding_gemma2),
    .unload = &encoder.unload,
    .forward = &encoder.tableForward,
    .cache = &.{ .layout = &config.layout },
    .embed = &.{ .project = &encoder.project },
    .weights = &.{},
};

/// The release's tiny checkpoint (`tests/dump_embeddinggemma2_fixtures.py tiny`, f32, its image tower beside it).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes = @embedFile("../../fixtures/embedding_gemma2_tiny_config.json") },
        .{ .name = "model.safetensors", .bytes = @embedFile("../../fixtures/embedding_gemma2_tiny.safetensors") },
    }) };
}

// ── tests ──

test {
    _ = config;
}
