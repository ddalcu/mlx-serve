//! `claim.zig`'s real-pack test: it parses the config and tokenizer the claimed GGUF stands in for, through the engine.

const std = @import("std");
const env = @import("../../../env.zig");
const claim = @import("claim.zig");
const claimState = claim.claimState;
const servablePath = claim.servablePath;
const weightBytes = claim.weightBytes;

test "real GGUF: stands in for a model dir, and --engine turns it off (a real GGUF, env-gated)" {
    const model = std.mem.span(env.get(.MLX_SERVE_GGUF_TEST_MODEL) orelse return error.SkipZigTest);
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const model_mod = @import("../../config/model.zig");
    const tokenizer_mod = @import("../../tokenizer/tokenizer.zig");
    const prev = claimState().enabled;
    defer claimState().enabled = prev;
    claimState().enabled = true;

    var config = try model_mod.parseConfig(io, allocator, model);
    defer config.deinit(allocator);
    try std.testing.expectEqual(model_mod.QuantMode.gguf, config.quant_mode);
    try std.testing.expect(config.num_hidden_layers > 0 and config.quant_bits > 0);

    var tok = try tokenizer_mod.loadTokenizer(io, allocator, model);
    defer tok.deinit();
    const text = "Hello, GGUF on MLX! 12345 héllo";
    const ids = try tok.encode(allocator, text);
    defer allocator.free(ids);
    const back = try tok.decode(allocator, ids, false);
    defer allocator.free(back);
    try std.testing.expectEqualStrings(text, back);
    try std.testing.expect(tok.special_tokens.count() > 0);

    try std.testing.expect(weightBytes(io, model).? > 1 << 20);
    claimState().enabled = false;
    try std.testing.expectEqual(@as(?[]u8, null), servablePath(io, allocator, model));
}
