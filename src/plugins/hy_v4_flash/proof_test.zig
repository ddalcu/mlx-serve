//! Parity proof and contract tests for the hy_v4_flash plugin.

const std = @import("std");
const testing = std.testing;
const abi = @import("../../abi/plugin.zig");
const plugin = @import("plugin.zig");
const config = @import("config.zig");
const hy4 = @import("hy_v4_flash.zig");
const fixture = @import("../../sdk/testing/fixture.zig");

test "hy_v4_flash: table declares hy_v4_flash and refuses foreign types" {
    const claims = plugin.table.claims;
    const mem: fixture.MemDir = .{ .files = &.{} };
    const pd = mem.packDir();
    const claim = claims(abi.Str.of("hy_v4_flash"), abi.Str.of("{}"), &pd);
    try testing.expect(claim != .no);

    const not_claim = claims(abi.Str.of("hy_v3"), abi.Str.of("{}"), &pd);
    try testing.expect(not_claim == .no);
}

test "hy_v4_flash: config parse sets module-owned decode caps and merges eos 120025" {
    const cfg_json =
        \\{
        \\  "model_type": "hy_v4_flash",
        \\  "vocab_size": 120832,
        \\  "hidden_size": 3072,
        \\  "num_hidden_layers": 42,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 32,
        \\  "head_dim": 96,
        \\  "num_experts_per_tok": 16,
        \\  "moe_intermediate_size": 1536,
        \\  "eos_token_id": [2, 120020]
        \\}
    ;
    var parsed: abi.Parsed = .{};
    const mem: fixture.MemDir = .{ .files = &.{} };
    const pd = mem.packDir();
    const res = config.parse(&pd, abi.Str.of(cfg_json), abi.Str.of("{}"), null, true, &parsed);
    try testing.expectEqual(abi.Verdict.ok, res);
    defer config.freeParsed(&parsed);

    try testing.expect(parsed.caps.owns_decode_state);
    try testing.expect(!parsed.caps.prefix_cache);
    try testing.expect(parsed.caps.moe);
    try testing.expectEqualSlices(u32, &.{ 2, 120020, config.hy_eos }, parsed.shell.eos[0..parsed.shell.eos_count]);

    const c = config.of(&parsed);
    try testing.expectEqual(@as(usize, 3072), c.hidden);
    try testing.expectEqual(@as(usize, 42), c.layers);
    try testing.expectEqual(@as(usize, 120832), c.vocab);
}

test "hy_v4_flash: layout reflects indexed vs sliding attention layers" {
    const cfg_json =
        \\{
        \\  "model_type": "hy_v4_flash",
        \\  "vocab_size": 120832,
        \\  "hidden_size": 3072,
        \\  "num_hidden_layers": 3,
        \\  "num_attention_heads": 32,
        \\  "num_key_value_heads": 32,
        \\  "head_dim": 96,
        \\  "num_experts_per_tok": 16,
        \\  "moe_intermediate_size": 1536,
        \\  "sliding_window": 512,
        \\  "layer_types": ["indexed_attention", "sliding_attention", "indexed_attention"]
        \\}
    ;
    var parsed: abi.Parsed = .{};
    const mem: fixture.MemDir = .{ .files = &.{} };
    const pd = mem.packDir();
    const res = config.parse(&pd, abi.Str.of(cfg_json), abi.Str.of("{}"), null, true, &parsed);
    try testing.expectEqual(abi.Verdict.ok, res);
    defer config.freeParsed(&parsed);

    var layers: [3]abi.LayerCache = undefined;
    const n = plugin.layout(&parsed, null, &layers, 3);
    try testing.expectEqual(@as(u32, 3), n);
    try testing.expectEqual(abi.LayerKind.full, layers[0].kind);
    try testing.expectEqual(abi.LayerKind.sliding, layers[1].kind);
    try testing.expectEqual(@as(u32, 512), layers[1].window);
    try testing.expectEqual(abi.LayerKind.full, layers[2].kind);
}
