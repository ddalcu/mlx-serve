//! Ling 3.0 (`bailing_hybrid`, BailingMoeV3): KDA linear attention (a per-channel bounded-sigmoid gate) beside
//! naive MLA every `layer_group_size`-th layer, a grouped sigmoid MoE with one ungated shared expert. Its own forward
//! (`table_forward.zig`) over the trunk GLM-5.3-Flash's plugin loads (`glm5_next/trunk.zig`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const host = @import("../glm5_next/host.zig");
const trunk = @import("../glm5_next/trunk.zig");
const sdk_bill = @import("../../sdk/bill/prefill.zig");
const dq_gemm = @import("../../sdk/blocks/linear/prefill_dq_gemm.zig");
const residual_norm = @import("../../sdk/blocks/norms/residual_norm.zig");
const Family = @import("../glm5_next/config.zig").Config;
const run = @import("../../sdk/bases/run.zig");
const fwd = @import("table_forward.zig");
const head = @import("head.zig");
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
    .weights = &.{ .resident_bytes = &residentBytes },
    .spec = &.{ .trunk_weight = &host.trunkWeight, .head = &head.hooks },
    .meters = &host.meters,
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

/// The pack's shards, and for a streaming pack with a NextN layer its draft readout (`trunk.draftReadout`), built at
/// load from the plugin's EXL3 head.
fn residentBytes(p: *const abi.Parsed, _: abi.Str, _: abi.Str, shards: u64, out: *u64) callconv(.c) abi.Verdict {
    const c = config.of(p);
    out.* = shards + if (c.streamed_bank and c.nextn_layers > 0) trunk.draftReadoutBytes(c.vocab_size, c.hidden_size, trunk.readoutBits(c)) else 0;
    return .ok;
}

/// A streaming pack's prompt: the engine's prompt bill of the pack's config as an unstreamed model's (one output token
/// past the prompt, no warm restore credited; the bank's slot rows are its plugin's bill, apart), priced from
/// `sdk/bill/prefill.zig` so the plugin's closure stays in sdk/. `bill_test.zig` pins it to the engine's
/// `prefillNeededAtChunk` of the same config.
pub fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, kv_bits: u8, _: u64) callconv(.c) u64 {
    const c = config.of(p);
    const mla = c.mla_kv_lora_rank > 0;
    const qk: u64 = @as(u64, c.mla_qk_nope_head_dim) + c.mla_qk_rope_head_dim;
    const attn: u64 = c.attnLayerCount();
    // ModelConfig.kvBytesPerToken: MLA caches every head's expanded key and value, other attention its KV heads.
    const kv_per_tok: u64 = if (mla) attn * c.num_attention_heads * (qk + c.mla_v_head_dim) * 2 else attn * c.num_key_value_heads * 2 * c.head_dim * 2;
    // bills.prefillFfnWidth.
    const moe_w: u64 = @as(u64, c.moe_intermediate_size) * @max(@as(u64, c.num_experts_per_tok), 1) + c.shared_expert_intermediate_size;
    const dense_w: u64 = if (c.intermediate_size_declared) c.intermediate_size else 0;
    const ffn: u64 = if (@max(dense_w, moe_w) > 0) @max(dense_w, moe_w) else c.intermediate_size;
    // bills.prefillStreamBytesPerToken off GLM: the KDA layers' q/k/v inside one eval window, the MoE gather's top_k replica.
    const cadence: u64 = run.PREFILL_EVAL_CADENCE_DEFAULT;
    var stream: u64 = 0;
    const linear: u64 = @as(u64, c.num_hidden_layers) -| attn;
    if (c.linear_num_value_heads > 0 and linear > 0)
        stream += @min(linear, cadence + 1) * (2 * @as(u64, c.linear_num_key_heads) * c.linear_key_head_dim + @as(u64, c.linear_num_value_heads) * c.linear_value_head_dim) * 2;
    if (c.num_experts > 0)
        stream += cadence * @max(@as(u64, c.num_experts_per_tok), 1) * 2 * (@as(u64, c.hidden_size) + c.moe_intermediate_size) * 2;
    // bills.prefillDequantWeightBytes.
    const dq: u64 = if (c.quant_bits == 0 or c.quant_mode != .affine or !dq_gemm.prefillDqGemmEnabled()) 0 else 3 * ffn * c.hidden_size * 2;
    return sdk_bill.prefillMemoryNeeded(seq, c.num_attention_heads, c.num_key_value_heads, kv_per_tok, c.head_dim, if (mla) qk else c.head_dim, c.hidden_size, ffn, kv_bits, chunk, seq, stream, dq, .{ .dq_min_rows = dq_gemm.prefillDqGemmMinRows(c.quant_bits) });
}

fn load(ctx: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
    return host.load(plugin.name, ctx, out);
}

pub fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
    const m = host.of(mp);
    if (!abi.declaresMode(m.caps, f.mode)) return abi.Verdict.refuse(.unsupported, "bailing_hybrid: forward mode not declared");
    const ops = f.cache_ops orelse return abi.Verdict.refuse(.unsupported, "bailing_hybrid: a forward without cache_ops");
    m.serial_rows = serialVerify(m.c, f.mode, f.rows);
    defer m.serial_rows = false;
    // A streaming pack's bank: its phase before the trunk runs, its slot rows settled after the outputs are evaluated.
    // A serial-rows verify calls it a decode lane at a time.
    trunk.bankPhase(m, if (m.serial_rows) 1 else f.rows) catch |e| return abi.Verdict.named(.failed, @errorName(e));
    const v = host.forwardOne(plugin.name, fwd, m, ops, f, out);
    return if (v.isOk()) host.settleBank(m, f, out) else v;
}

/// A streaming pack's verify runs serial rows (`Model.serial_rows`): every row carries the bytes serial decode gives
/// it, so greedy MTP and PLD drafting emit serial decode's text whatever widths the planner picks. Up to the fused
/// add+norm's rows: a wider verify takes another norm than a decode step.
pub fn serialVerify(c: *const Family, mode: abi.Mode, rows: u32) bool {
    return c.streamed_bank and mode == .verify and @as(i64, rows) <= residual_norm.add_norm_max_rows;
}

/// The tiny pack the conformance suite runs: KDA, MLA, the dense first layer and the grouped MoE.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try fixture.lingFiles(a, 11) };
}

// ── tests ──

test {
    _ = config;
}

test "bailing_hybrid serial verify: a streaming pack's verify of up to 16 rows; never a prompt, a decode or an MLX pack" {
    const json = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json");
    const mem: sdk.fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = json }, .{ .name = "experts.bin", .bytes = "" } } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try std.testing.expect(config.parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    const c = config.of(&p);
    try std.testing.expect(serialVerify(c, .verify, 1) and serialVerify(c, .verify, 9) and serialVerify(c, .verify, 16));
    try std.testing.expect(!serialVerify(c, .verify, 17) and !serialVerify(c, .prefill, 9) and !serialVerify(c, .decode, 1));
    const tiny = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json");
    const tmem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = tiny }} };
    const tdir = tmem.packDir();
    var q: abi.Parsed = .{};
    try std.testing.expect(config.parse(&tdir, abi.Str.of(tiny), .{}, null, true, &q).isOk());
    defer config.freeParsed(&q);
    try std.testing.expect(!serialVerify(config.of(&q), .verify, 9));
}

test "bailing_hybrid resident bytes: a streaming pack with a NextN layer adds its draft readout to the shards; Ling 3.0 the shards" {
    const json = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.1-flash-exl3-2.60bpw.json");
    const mem: sdk.fixture.MemDir = .{ .files = &.{ .{ .name = "config.json", .bytes = json }, .{ .name = "experts.bin", .bytes = "" } } };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try std.testing.expect(config.parse(&dir, abi.Str.of(json), .{}, null, true, &p).isOk());
    defer config.freeParsed(&p);
    var got: u64 = 0;
    try std.testing.expect(residentBytes(&p, .{}, .{}, 1000, &got).isOk());
    try std.testing.expectEqual(@as(u64, 1000 + 362_151_936), got);
    const tiny = @embedFile("../../fixtures/model-configs/bailing_hybrid-ling-3.0-tiny-4bit.json");
    const tmem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = tiny }} };
    const tdir = tmem.packDir();
    var q: abi.Parsed = .{};
    try std.testing.expect(config.parse(&tdir, abi.Str.of(tiny), .{}, null, true, &q).isOk());
    defer config.freeParsed(&q);
    try std.testing.expect(residentBytes(&q, .{}, .{}, 1000, &got).isOk());
    try std.testing.expectEqual(@as(u64, 1000), got);
}
