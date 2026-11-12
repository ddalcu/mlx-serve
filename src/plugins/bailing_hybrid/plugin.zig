//! Ling 3.0 (`bailing_hybrid`, BailingMoeV3): KDA linear attention (a per-channel bounded-sigmoid gate) beside
//! naive MLA every `layer_group_size`-th layer, a grouped sigmoid MoE with one ungated shared expert. Its own forward
//! (`table_forward.zig`) over the trunk GLM-5.3-Flash's plugin loads (`glm5_next/trunk.zig`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const host = @import("../glm5_next/host.zig");
const trunk = @import("../glm5_next/trunk.zig");
const toModelConfig = @import("../glm5_next/model_config.zig").toModelConfig;
const prefillNeededAtChunk = @import("../../engines/mlx/admission/prefill_chunk.zig").prefillNeededAtChunk;
const fwd = @import("table_forward.zig");
const fixture = @import("../glm5_next/fixture.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.ModelTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &host.unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout },
    .weights = &.{},
    .spec = &.{ .trunk_weight = &host.trunkWeight },
};

/// KDA layers keep a recurrent state; an MLA layer caches every head's expanded key (nope | rope) and value.
pub fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const first: u32 = if (span) |s| s.first else 0;
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    for (out[0..n], first..) |*l, li| l.* = if (c.isLinearLayer(@intCast(li)))
        .{ .kind = .linear }
    else
        .{ .kind = .full, .kv_heads = c.num_attention_heads, .k_dim = c.mla_qk_nope_head_dim + c.mla_qk_rope_head_dim, .v_dim = c.mla_v_head_dim };
    return n;
}

/// MLA decompresses its latent to every attention head before the write, so the cache bills per head. A streaming pack
/// (`Config.streamed_bank`) bills its prompt here (`prefillBytes`): the engine bills no streamed model's prompt, and
/// sizes the bank's room from this term.
fn bill(p: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    const c = config.of(p);
    const qk: u32 = c.mla_qk_nope_head_dim + c.mla_qk_rope_head_dim;
    out.* = .{
        .kv_per_token = @as(u64, c.attnLayerCount()) * c.num_attention_heads * (qk + c.mla_v_head_dim) * 2,
        .ssm_checkpoint = c.ssmCheckpointBytes(2),
        .prefill_score_width = qk,
        .prefill_bytes = if (c.streamed_bank) &prefillBytes else null,
    };
}

/// A streaming pack's prompt: the engine's own prompt bill of the pack's config as an unstreamed model's (one output
/// token past the prompt; the bank's slot rows are its plugin's bill, apart).
fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, kv_bits: u8, warm: u64) callconv(.c) u64 {
    const cfg = toModelConfig(config.of(p)) catch return std.math.maxInt(u64);
    return prefillNeededAtChunk(&cfg, seq, 1, kv_bits, chunk, .{ .matched_tokens = warm });
}

fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    return host.load(plugin.name, ctx, out);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m = host.of(mp);
    if (!abi.declaresMode(m.caps, f.mode)) return abi.Verdict.refuse(.unsupported, "bailing_hybrid: forward mode not declared");
    const ops = f.cache_ops orelse return abi.Verdict.refuse(.unsupported, "bailing_hybrid: a forward without cache_ops");
    // A streaming pack's bank: its phase before the trunk runs, its slot rows settled after the outputs are evaluated.
    trunk.bankPhase(m, f.rows) catch |e| return abi.Verdict.named(.failed, @errorName(e));
    const v = host.forwardOne(plugin.name, fwd, m, ops, f, out);
    return if (v.isOk()) host.settleBank(m, f, out) else v;
}

/// The tiny pack the conformance suite runs: KDA, MLA, the dense first layer and the grouped MoE.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.lingFiles(a, 11) };
}

// ── tests ──

test {
    _ = config;
}

test "bailing_hybrid bill: a streaming pack bills its prompt as the engine bills the same config unstreamed; another pack leaves it to the engine" {
    const json = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json");
    const mem: sdk.fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = json }, .{ .name = "experts.bin", .bytes = "" } } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try std.testing.expect(config.parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    var b: abi.Bill = .{};
    bill(&p, &.{ .context = 32768 }, &b);
    const cfg = try toModelConfig(config.of(&p));
    const want = prefillNeededAtChunk(&cfg, 18_380, 1, 16, 4096, .{});
    try std.testing.expect(want > 0);
    try std.testing.expectEqual(want, b.prefill_bytes.?(&p, 18_380, 4096, 16, 0));
    try std.testing.expect(p.caps.prompt_last_row and p.caps.streamed_weights);

    const tiny = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json");
    const tmem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = tiny }} };
    const tdir = tmem.packDir();
    var q: abi.Parsed = .{};
    try std.testing.expect(config.parse(&tdir, abi.Str.of(tiny), .{}, null, true, &q).isOk());
    defer config.freeParsed(&q);
    var bq: abi.Bill = .{};
    bill(&q, &.{ .context = 32768 }, &bq);
    try std.testing.expect(bq.prefill_bytes == null and !q.caps.prompt_last_row);
}
