//! Gemma 4 text packs (`gemma4`, `gemma4_text`, `gemma4_unified(_text)`, mlx-lm's layout): the claim and the parse into
//! `Parsed` plus the shared Gemma decoder `Config` (`plugins/gemma3/config.zig`). MLX-free. Served bit-equal: the
//! dense decoder (KV-sharing tail, K=V full layers, value norms, dual head widths, proportional rope, per-layer
//! embeddings, layer scalars) and the 26B-A4B experts (sigma-MoE beside a dense MLP), with image and audio rows at
//! their placeholders (a unified pack's encoder-free embedder rows too); declined `unsupported`: drafter configs.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const gemma = @import("../gemma3/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "gemma4",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/gemma-4-e4b-it-4bit"},
};

pub const model_types = [_][]const u8{ "gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text" };

pub const claims = sdk.claimsByType(&model_types);

/// Every block of the dense decoder serves batched rows (`plugin.zig` checks it); a pack with experts decodes serial,
/// as the engine's layered forward does.
pub const batches_decode = true;

pub const Config = gemma.Config;
pub const unsupported = gemma.unsupported;
pub const freeParsed = gemma.freeParsed;
pub const of = gemma.of;
pub const layout = gemma.layout;
const field = gemma.field;

/// Keys only a draft model's config carries (DSpark / DFlash blocks, EAGLE-3 heads): not a trunk this serves.
const drafter_keys = [_][]const u8{ "block_size", "mask_token_id", "target_layer_ids", "draft_num_hidden_layers" };

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    var d: gemma.Doc = undefined;
    const opened = gemma.Doc.open(config_json, &d);
    if (!opened.isOk()) return opened;
    defer d.close();
    if (span != null) return unsupported("a layer span (the cluster split)");
    const known = for (model_types) |t| {
        if (std.mem.eql(u8, t, d.model_type)) break true;
    } else false;
    if (!known) return unsupported("another model_type");
    for (drafter_keys) |k| if (field(d.cfg, k) != null) return unsupported("a draft model's config");
    if (field(d.root, "pooling_mode") != null or dir.has("1_Pooling/config.json")) return unsupported("an embedding pack");
    if (gemma.foreignQuant(d.root)) return unsupported("a quantization_config naming a quant_method");

    var c: Config = .{};
    gemma.readCommon(d.root, d.cfg, &c) catch return unsupported("a field out of range");
    gemma.readQuant(d.root, &c) catch return unsupported("quantization");
    // The engine's Gemma 4 arm: one decoder tag, nested prefix, plain norms, value norms, no rope scale.
    c.model_type = "gemma4";
    c.weight_prefix = "language_model.model";
    c.norm_has_offset = false;
    c.has_v_norm = true;
    c.rope_scaling_factor = 1.0;
    if (c.num_experts > 0) {
        if (c.num_experts_per_tok == 0 or c.num_experts_per_tok > c.num_experts or c.moe_intermediate_size == 0)
            return unsupported("an expert geometry out of range");
        // The engine's layered forward reads no per-layer inputs and shares no KV.
        if (c.hidden_size_per_layer_input > 0 or c.num_kv_shared_layers > 0) return unsupported("experts with per-layer inputs or KV sharing");
    }
    if (c.vocab_size == 0 or c.vocab_size > 1 << 22 or c.hidden_size == 0 or c.hidden_size > 1 << 16 or
        c.num_hidden_layers == 0 or c.num_hidden_layers > gemma.max_layer_types or c.num_attention_heads == 0 or
        c.num_key_value_heads == 0 or c.intermediate_size == 0 or c.head_dim == 0 or c.head_dim % 2 != 0 or
        c.head_dim > 512 or c.global_head_dim > 512 or c.global_head_dim % 2 != 0 or
        c.num_kv_shared_layers >= c.num_hidden_layers or (c.has_sliding_window and c.sliding_window_pattern == 0) or
        c.partial_rotary_factor_global <= 0 or c.partial_rotary_factor_global > 1)
        return unsupported("dimensions out of range");
    const v = gemma.finish(dir, d.root, c, stub, .{ .batches_decode = batches_decode and c.num_experts == 0, .moe = c.num_experts > 0, .exact_resumed = true, .dflash_capture = true, .prefill_mask = true }, out);
    if (!v.isOk()) return v;
    // Gemma 4 opens its own `<|channel>thought` block when asked to think.
    out.chat.reasoning_dialect = .gemma_channel;
    gemma.readMedia(d.root, out) catch {
        freeParsed(out);
        return unsupported("a placeholder id");
    };
    return .ok;
}

// ── tests ──

const testing = std.testing;
const fixture = sdk.fixture;

fn parseJson(json: []const u8, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(json), .{}, null, true, out);
}

const e2b_like =
    \\{"model_type":"gemma4","text_config":{"model_type":"gemma4_text","hidden_size":1536,"intermediate_size":6144,
    \\"num_hidden_layers":6,"num_attention_heads":8,"num_key_value_heads":1,"head_dim":256,"global_head_dim":512,
    \\"num_kv_shared_layers":2,"hidden_size_per_layer_input":256,"sliding_window":512,"vocab_size":262144,
    \\"layer_types":["sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","full_attention"],
    \\"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"proportional","partial_rotary_factor":0.25},
    \\"sliding_attention":{"rope_theta":10000.0}},"final_logit_softcapping":30.0,"tie_word_embeddings":true},
    \\"eos_token_id":[1,106,50]}
;

test "gemma4 config: a nested text config parses into the Gemma 4 traits, its KV-shared tail and the stop set" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(e2b_like, &p).isOk());
    defer freeParsed(&p);
    const c = of(&p);
    try testing.expect(c.has_v_norm and !c.norm_has_offset and c.rope_proportional and c.dualHeadDim());
    try testing.expectEqual(@as(?u32, 3), c.kvSource(4));
    try testing.expectEqual(@as(?u32, 2), c.kvSource(5));
    try testing.expectEqualSlices(u32, &.{ 1, 106, 50 }, p.shell.eos[0..p.shell.eos_count]);
    try testing.expect(p.caps.batches_decode and p.caps.exact_resumed);
    var lay: [6]abi.LayerCache = undefined;
    try testing.expectEqual(@as(u32, 6), layout(&p, null, &lay, 6));
    try testing.expect(lay[2].kind == .full and lay[2].k_dim == 512 and lay[4].kv_source == 3 and lay[0].window == 512);
}

test "gemma4 config: experts with per-layer inputs and the drafter fixtures are declined" {
    inline for (.{
        "{\"model_type\":\"gemma4_text\",\"num_experts\":128,\"top_k_experts\":8,\"moe_intermediate_size\":64,\"hidden_size_per_layer_input\":8}",
        @embedFile("../../fixtures/model-configs/gemma4-text.json"),
        @embedFile("../../fixtures/model-configs/gemma4-unified-text.json"),
    }) |json| {
        var p: abi.Parsed = .{};
        try testing.expect(parseJson(json, &p).code == .unsupported and p.private == null);
    }
}
