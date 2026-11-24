//! hy_v4_flash's model table: the claim, the parse, the bill, the load and the forward.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const vision_mod = @import("vision.zig");
const trunk_mod = @import("trunk.zig");
const log = @import("../../sdk/util/log.zig");
const mlx = @import("../../mlx/ffi.zig");

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
    .media = &.{ .tower = &vision_mod.tower, .prepare = &vision_mod.prepare },
    .weights = &.{ .resident_bytes = &residentBytes },
};

const Verdict = abi.Verdict;

/// The conformance pack: the tiny hy_v4_flash config with a matching micro safetensors set (fixture.zig).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{ .files = try @import("fixture.zig").build(a) };
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
    const idx_bytes = @as(u64, c.index_n_heads) * c.index_head_dim * c.q_lora_rank * (q + 2) / 8 + @as(u64, d) * 256 * (q + 2) / 8;
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
            load_bytes += @as(u64, c.n_shared_experts) * 3 * d * c.moe_intermediate_size * (q + 2) / 8; // shared expert
            load_bytes += route_bytes;
        }
    }
    if (c.has_vision) load_bytes += c.vision_depth * c.vision_hidden_size * (c.vision_hidden_size * 4 + c.vision_intermediate_size * 2) * 2; // bf16 tower + projector
    for (0..c.dfly_target_layer_count) |_| {
        load_bytes += d * (c.q_lora_rank + @as(u64, c.num_attention_heads) * 256 + c.kv_lora_rank + 8192 + 9216 * 3) * 2; // bf16 draft layer
    }
    load_bytes += 4 * 3 * d * 4; // hc_head, f32
    // The KV row the forward writes: the latent K row carries `[latent | k_rot]` (576 halves) — plus the
    // indexer's per-token key (128) on DSA layers — and the V slot holds the same latent (512 halves) for MLA's
    // post-reduction lift.
    const kv_row: u64 = (2 * c.kv_lora_rank + c.qk_rope_head_dim) * 2;
    const kv_row_dsa: u64 = kv_row + @as(u64, c.index_head_dim) * 2;
    const swa: u64 = c.num_hidden_layers - c.dsa_layer_count;
    out.* = .{
        .kv_per_token = kv_row_dsa * c.dsa_layer_count,
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
        // DSA layers cache their token keys beside the latent (the compressed indexer pools them).
        .k_dim = c.kv_lora_rank + c.qk_rope_head_dim + if (c.dsa_layers[i]) c.index_head_dim else 0,
        .v_dim = c.kv_lora_rank,
    };
    return n;
}

/// A shape/missing failure reads as a sentence naming the tensor: the static prefix plus the name
/// `validate` stopped on, kept in a global that outlives the call (the host reads `why` after the return).
fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return weights.load(ctx) catch |e| {
        log.err("hy_v4: load failed: {t}\n", .{e});
        out.* = switch (e) {
            error.MissingWeight, error.WrongShape => weights.failureVerdict(e),
            error.TableMissing, error.TableSplit, error.TableHeader, error.TableRows, error.TableGeometry, error.TableOpen, error.IndexMissing, error.EngramCount => weights.failureVerdict(e),
            error.UnsupportedWeightSource => Verdict.refuse(.unsupported, "hy_v4_flash: a weight source other than the resident map"),
            else => Verdict.named(.failed, @errorName(e)),
        };
        return null;
    };
}

fn unload(m: *anyopaque) callconv(.c) void {
    const model: *weights.Model = @ptrCast(@alignCast(m));
    model.deinit();
    std.heap.c_allocator.destroy(model);
}

/// The preflight's resident bytes: the shard census less the Engram tables, which the stream bank owns
/// (their ranges open through `weights.Engram`; the load never maps them). A pack that disagrees — no
/// table where the config says one — is refused by name at load, not here.
fn residentBytes(p: *const abi.Parsed, _: abi.Str, _: abi.Str, shards: u64, out: *u64) callconv(.c) Verdict {
    out.* = shards - engramPackedBytes(config.of(p));
    return .ok;
}

/// The packed bytes of every Engram table the config places: one row is 12 U32 words (48 B, 96 4-bit
/// codes) plus three bf16 scales and three bf16 biases (12 B) = 60 B. `rows` comes from the config's
/// head-vocab lists; the test below pins the arithmetic against the pack (4 × ~165 M rows = 39.6 GB), and
/// the tiny-pack load test pins the ranges the load opens.
fn engramPackedBytes(c: *const config.Config) u64 {
    if (c.engram_layer_count == 0) return 0;
    const row_bytes = @divExact(c.engram_head_dim * c.quant_bits, 32) * 4 + 2 * (3 * 2);
    var bytes: u64 = 0;
    for (c.engram_table_rows) |rows| bytes += rows * row_bytes;
    return bytes;
}

fn forward(m: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const model: *weights.Model = @ptrCast(@alignCast(m));
    if (!abi.declaresMode(model.trunk.caps, f.mode)) return Verdict.refuse(.unsupported, "hy_v4_flash: forward mode not declared");
    if (f.cache_ops == null) return Verdict.refuse(.unsupported, "hy_v4_flash: a forward without cache_ops");
    out.* = model.trunk.forward(f) catch |e| {
        log.err("hy_v4: forward failed: {t}\n", .{e});
        if (mlx.takeError(&error_buf)) |text| log.err("hy_v4: latched MLX error: {s}\n", .{text});
        return Verdict.named(.failed, @errorName(e));
    };
    return .ok;
}

var error_buf: [512]u8 = undefined;

// ── tests ──

test "hy_v4 conformance: the tiny pack claims, loads and forwards in every declared mode" {
    try @import("../../sdk/testing/conformance.zig").expectPlugin(@This());
}

const testing = std.testing;

test "hy_v4_flash table: the bill covers shards without the Engram and names the KV row" {
    var p: abi.Parsed = .{};
    try testing.expect(config.parseJson(@embedFile("../../fixtures/model-configs/hy_v4-flash.json"), true, &p).isOk());
    defer config.freeParsed(&p);
    var b: abi.Bill = .{};
    bill(&p, &.{}, &b);
    // The bill's oracle is the pack's shard census (129.9 GB total, 39.60 GB of Engram the stream owns, so
    // 90.3 GB resident): the arithmetic may sit within 15% of it.
    try testing.expect(@abs(@as(i64, @intCast(b.load_bytes)) - 90_300_000_000) < 13_500_000_000);
    try testing.expectEqual(@as(u64, (2 * 512 + 64 + 128) * 2 * 12), b.kv_per_token);
    try testing.expect(b.fixed_slot_bytes > 30 * 512 * (2 * 512 + 64) * 2);
}

test "hy_v4 table: the Engram stream's packed bytes match the pack's table census" {
    var p: abi.Parsed = .{};
    try testing.expect(config.parseJson(@embedFile("../../fixtures/model-configs/hy_v4-flash.json"), true, &p).isOk());
    defer config.freeParsed(&p);
    // The pack's four tables hold 660,006,416 rows between them; a row is 12 U32 words + 3 bf16 scales +
    // 3 bf16 biases = 60 B on disk (measured from the shard headers). The bill must subtract exactly this.
    const rows: u64 = 164_994_380 + 164_999_008 + 165_004_086 + 165_008_942;
    const want = rows * 60;
    const got = engramPackedBytes(config.of(&p));
    try testing.expectApproxEqRel(@as(f64, @floatFromInt(want)), @as(f64, @floatFromInt(got)), 0.001);
    // And the resident bill equals the shard census minus that figure.
    var b: abi.Bill = .{};
    bill(&p, &.{}, &b);
    var resident: u64 = 0;
    try testing.expect(residentBytes(&p, .{}, .{}, 129_900_000_000, &resident).isOk());
    try testing.expectApproxEqRel(@as(f64, @floatFromInt(129_900_000_000 - want)), @as(f64, @floatFromInt(resident)), 0.001);
}
