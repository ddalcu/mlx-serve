//! Parity proof for the qwen3_5 plugin: the table's forward over its own load of a tiny random pack (2 GDN + 2 gated
//! attention layers, hd-128 linear heads, hd-256 attention) is bit-equal to the legacy layered forward — prompt chunks
//! cold and warm, decode, captured verify rows — on logits and every recurrent state and capture, and two slots
//! batched (decode, verify rows, decode) bit-equal to the legacy GDN batched driver; the legacy values recorded to
//! `fixtures/twin-<seed>.tape`. Dense and MoE, 4-bit and bf16.

const std = @import("std");
const fixture = @import("../../sdk/bases/proof/fixture.zig");
const pack = @import("../../sdk/bases/proof/qwen35_pack.zig");
const twin = @import("../qwen3/twin_test.zig");
const qwen3_5 = @import("plugin.zig");

const testing = std.testing;

const T = twin.Twin(qwen3_5);

const solo = [_]twin.Step{
    .{ .mode = .prefill, .ids = &twin.seq(20, 3) },
    .{ .mode = .prefill, .ids = &twin.seq(70, 11) },
    .{ .mode = .decode, .ids = &.{88} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 200 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .verify, .ids = &.{ 9, 33, 77, 140, 2 } },
    .{ .mode = .decode, .ids = &.{64} },
    .{ .mode = .prefill, .ids = &twin.seq(6, 5) },
    .{ .mode = .decode, .ids = &.{17} },
};

fn open(p: *fixture.Pack, comptime moe: bool, comptime quantized: bool, comptime experts: c_int, comptime seed: u64) !*T {
    try pack.writePack(p, moe, quantized, experts);
    const name = std.fmt.comptimePrint("fixtures/twin-{x}.tape", .{seed});
    const t = try T.open(p, pack.config(moe, quantized, experts), &.{}, @embedFile(name));
    return t;
}

fn soloCase(comptime moe: bool, comptime quantized: bool, comptime experts: c_int, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    const t = try open(&p, moe, quantized, experts, seed);
    defer t.close();
    const l = try t.lane();
    defer l.deinit();
    for (solo) |st| try t.step(l, st);
    try t.expectDone();
}

fn batchedCase(comptime moe: bool, comptime seed: u64) !void {
    var p = fixture.Pack.init(seed);
    defer p.deinit();
    const t = try open(&p, moe, true, 8, seed);
    defer t.close();
    // Split the merged states back after every tick, so each slot's states can be compared.
    const lanes = [2]*twin.Lane{ try t.lane(), try t.lane() };
    defer for (lanes) |l| l.deinit();
    try t.step(lanes[0], .{ .mode = .prefill, .ids = &twin.seq(20, 3) });
    try t.step(lanes[1], .{ .mode = .prefill, .ids = &.{ 42, 101, 7, 9, 250, 3, 77 } });
    try t.batched(lanes, .{ &.{77}, &.{3} });
    try t.batched(lanes, .{ &.{ 5, 61, 200 }, &.{ 9, 33, 140 } });
    try t.batched(lanes, .{ &.{190}, &.{64} });
    try t.expectDone();
}

test "qwen3_5 plugin: a dense 4-bit pack through the table is bit-equal to the legacy layered forward" {
    try soloCase(false, true, 8, 0x35c1);
}

test "qwen3_5 plugin: a dense bf16 pack (decode attention side copies) is bit-equal" {
    try soloCase(false, false, 8, 0x35c2);
}

test "qwen3_5 plugin: a qwen3_5_moe 4-bit pack (shared-expert gate as the router kernel's vector) is bit-equal" {
    try soloCase(true, true, 64, 0x35c3);
}

test "qwen3_5 plugin: a qwen3_5_moe bf16 pack (dense experts) is bit-equal" {
    try soloCase(true, false, 8, 0x35c4);
}

test "qwen3_5 plugin: dense batched rows are bit-equal to the legacy GDN batched driver" {
    try batchedCase(false, 0x35c5);
}

test "qwen3_5 plugin: qwen3_5_moe batched rows are bit-equal to the legacy GDN batched driver" {
    try batchedCase(true, 0x35c6);
}
