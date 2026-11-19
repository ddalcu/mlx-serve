//! hy_v4_flash's model table. Plan 01 lands the claim, the parse and the bill; the load arms are stubs that
//! refuse with the plan that fills them, so a load fails by name instead of pretending.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
pub const config = @import("config.zig");

pub const plugin = sdk.Plugin{
    .name = "hy_v4",
    .kind = .model,
    .claims = &config.model_types,
};

pub const table = abi.ModelTable{
    .name = abi.Str.of("hy_v4"),
    .claims = config.claims,
    .parse = &config.parse,
    .free_parsed = &config.freeParsed,
    .bill = &bill,
    .load = &load,
    .unload = &unload,
    .forward = &forward,
    .cache = &.{ .layout = &layout },
    .weights = &.{},
};

const Verdict = abi.Verdict;

/// The conformance pack: the tiny hy_v4_flash config with a matching micro safetensors set, sized so every
/// projection, expert and engram table exists at its config shape. Plan 02's loader is what fills it in; until
/// then it is absent — the plugin stays in the registry's claim sweep and skips the load/forward checks.
pub const fixtureMayBeAbsent = true;
pub fn conformanceFixture(a: std.mem.Allocator) !?sdk.fixture.Fixture {
    _ = a;
    return null;
}

/// The Plan-01 bill. It prices the arch from the config; the load-side test's oracle is the pack's real
/// shard census (TensorFold 4-bit: 129.9 GB total, 39.7 GB of it Engram, which the stream bank owns).
fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    _ = shape;
    const c = config.of(p);
    const d: u64 = c.hidden_size;
    const q: u64 = @max(c.quant_bits, 1);
    var load_bytes: u64 = 0;
    // Embed and lm_head quantize too (g64 rows), despite enable_lm_head_fp32 upstream.
    load_bytes += 2 * @as(u64, c.vocab_size) * d * (q + 2) / 8;
    const att_bytes = (d * c.q_lora_rank + @as(u64, c.q_lora_rank) * c.num_attention_heads * 256 +
        d * (c.kv_lora_rank + c.qk_rope_head_dim) +
        @as(u64, (c.kv_lora_rank + c.qk_rope_head_dim)) * c.num_attention_heads * 256 +
        @as(u64, c.num_attention_heads) * c.v_head_dim * d + @as(u64, d) * c.num_attention_heads * c.v_head_dim) * (q + 2) / 8;
    // The compressed indexer's query/key projections (tiny; same affine layout).
    const idx_bytes = @as(u64, c.index_n_heads) * c.index_head_dim * d * (q + 2) / 8 + @as(u64, d) * 256 * (q + 2) / 8;
    // Router gate (f32 rows) + bias ledger, iHC f32 tables, per-arm norms.
    const route_bytes = @as(u64, d) * c.n_routed_experts * 4 + @as(u64, c.n_routed_experts) * 4 + 800_000;
    for (0..c.num_hidden_layers) |li| {
        load_bytes += att_bytes;
        if (c.dsa_layers[li]) load_bytes += idx_bytes;
        if (li < c.first_k_dense_replace) {
            load_bytes += 3 * d * c.intermediate_size * (q + 2) / 8; // the layer-0 dense MLP
        } else {
            // Packed exactly as the pack ships it: gate+up fused into [512, 3072, 192] U32 nibbles (4-bit)
            // with g32 bf16 scale+bias, down [512, 1536, 192]. No fused SwiGLU is possible, so it is never billed.
            load_bytes += @as(u64, c.n_routed_experts) * (3072 * 192 * 4 + 2 * 3072 * 24 * 2 + 1536 * 192 * 4 + 2 * 1536 * 24 * 2);
            load_bytes += @as(u64, c.n_shared_experts) * 3 * d * c.intermediate_size * (q + 2) / 8; // shared expert
            load_bytes += route_bytes;
        }
    }
    if (c.has_vision) load_bytes += c.vision_depth * c.vision_hidden_size * (c.vision_hidden_size * 4 + c.vision_intermediate_size * 2) * 2; // bf16 tower + projector
    for (0..c.dfly_target_layer_count) |_| {
        load_bytes += d * (c.q_lora_rank + @as(u64, c.num_attention_heads) * 256 + c.kv_lora_rank + 8192 + 9216 * 3) * 2; // bf16 draft layer
    }
    load_bytes += 4 * 3 * d * 4; // hc_head, f32
    // Plan 04 owns the indexer's pooled-key state; Plan 02 bills the Engram tables as a stream bank. Until then the
    // latent KV row reads 576 bf16 for the DSA layers (full length) and the SWA layers keep a 512-token ring.
    const kv_row: u64 = (c.kv_lora_rank + c.qk_rope_head_dim) * 2;
    const swa: u64 = c.num_hidden_layers - c.dsa_layer_count;
    out.* = .{
        .kv_per_token = kv_row * c.dsa_layer_count,
        .fixed_slot_bytes = swa * c.sliding_window * kv_row +
            @as(u64, 4) * (8 * c.hc_mult + @as(u64, 4) * 8) * c.hidden_size * 4 + // iHC streams + engram per-token state
            @as(u64, c.dsa_layer_count) * 2048 * 4, // the indexer's latest top-k set per slot
        .load_bytes = load_bytes,
    };
}

fn layout(p: *const abi.Parsed, span: ?*const abi.LayerSpan, out: [*]abi.LayerCache, cap: u32) callconv(.c) u32 {
    const c = config.of(p);
    const n = @min(if (span) |s| s.count else c.num_hidden_layers, cap);
    for (out[0..n], 0..) |*l, i| l.* = .{
        .kind = if (c.dsa_layers[i]) .full else .sliding,
        .window = if (c.dsa_layers[i]) 0 else c.sliding_window,
        .kv_heads = 1,
        .k_dim = c.kv_lora_rank + c.qk_rope_head_dim,
        .v_dim = c.kv_lora_rank,
    };
    return n;
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    _ = ctx;
    out.* = Verdict.refuse(.unsupported, "hy_v4_flash: the weight loader lands in Plan 02 (a pack with no safetensors is refused sooner, by the SDK loader)");
    return null;
}

fn unload(m: *anyopaque) callconv(.c) void {
    _ = m;
}

fn forward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    _ = f;
    _ = out;
    _ = m;
    return Verdict.refuse(.unsupported, "hy_v4_flash: the forward lands in Plan 03");
}

// ── tests ──

const testing = std.testing;

test "hy_v4_flash table: the bill covers shards without the Engram and names the KV row" {
    var p: abi.Parsed = .{};
    try testing.expect(config.parseJson(@embedFile("../../fixtures/model-configs/hy_v4-flash.json"), true, &p).isOk());
    defer config.freeParsed(&p);
    var b: abi.Bill = .{};
    bill(&p, &.{}, &b);
    // The bill's oracle is the pack's shard census (measured 89.5 GB over 129.9 GB total, the rest being
    // the 39.7 GB Engram): the arithmetic may sit within 15% of it.
    try testing.expect(@abs(@as(i64, @intCast(b.load_bytes)) - 89_500_000_000) < 13_500_000_000);
    try testing.expectEqual(@as(u64, 576 * 2 * 12), b.kv_per_token);
    try testing.expect(b.fixed_slot_bytes > 30 * 512 * 576 * 2);
}
