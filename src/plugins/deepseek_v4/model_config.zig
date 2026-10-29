//! The engine's `ModelConfig` from the family's `Config`, for the modules and the weight loader that still read it:
//! the same names one to one plus the family's constants. Pinned field for field against the engine's own parse.

const std = @import("std");
const model = @import("../../sdk/config/model.zig");
const Config = @import("config.zig").Config;

/// `dir` becomes the config's owned `pack_dir` (V4.1 opens its Engram tables there); `ModelConfig.deinit` frees it.
pub fn toModelConfig(a: std.mem.Allocator, c: *const Config, dir: ?[]const u8) !model.ModelConfig {
    var m: model.ModelConfig = .{};
    inline for (@typeInfo(Config).@"struct".field_names) |name| {
        if (comptime std.mem.eql(u8, name, "gen") or std.mem.eql(u8, name, "dsv41_streamed") or std.mem.eql(u8, name, "dsv41_exl3_bank")) continue;
        const src = @field(c, name);
        const Dst = @TypeOf(@field(m, name));
        @field(m, name) = if (@TypeOf(src) == Dst) src else std.meta.stringToEnum(Dst, @tagName(src)).?;
    }
    m.weight_prefix = ""; // the release ships bare names (embed.weight, layers.N....)
    m.norm_has_offset = false;
    m.scale_embeddings = false;
    m.has_pre_ff_norm = false;
    m.has_qk_norm = false; // q-norm is on the lora rank + an unweighted per-head RMS, in the module
    m.hidden_act = .silu;
    m.legacy.caps.prefix_cache = false;
    if (c.isV41()) m.legacy.thinking = .on;
    if (c.isV41()) if (dir) |d| {
        m.pack_dir = try a.dupe(u8, d);
    };
    return m;
}

// ── tests ──

const testing = std.testing;
const sdk = @import("../../sdk/root.zig");
const abi = @import("../../abi/plugin.zig");
const v4_config = @import("config.zig");
const v41_config = @import("../deepseek_v41/config.zig");

/// The plugin's parse, through `toModelConfig`, equals the engine's parse of the same bytes in every field.
fn expectEngineParse(json: []const u8, v41: bool) !void {
    var want = try model.parseConfigFromJson(testing.allocator, json);
    defer want.deinit(testing.allocator);
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    const v = if (v41) v41_config.parse(&dir, .of(json), .{}, null, true, &p) else v4_config.parse(&dir, .of(json), .{}, null, true, &p);
    try testing.expect(v.isOk());
    defer v4_config.freeParsed(&p);
    var got = try toModelConfig(testing.allocator, v4_config.of(&p), null);
    defer got.deinit(testing.allocator);
    try testing.expectEqualDeep(want, got);
}

/// The DSV4 miniature the oracle fabricates (tests/dsv4_mlx_ref.py): every arm, the DSpark stages.
const mini_config =
    \\{"model_type": "deepseek_v4", "hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4,
    \\ "num_key_value_heads": 1, "head_dim": 96, "qk_rope_head_dim": 32, "q_lora_rank": 32, "o_lora_rank": 16,
    \\ "o_groups": 2, "sliding_window": 8, "compress_ratios": [0, 4, 16, 4, 0, 0, 0], "compress_rope_theta": 160000.0,
    \\ "rope_theta": 10000.0, "rope_scaling": {"factor": 16, "original_max_position_embeddings": 64, "beta_fast": 32,
    \\ "beta_slow": 1, "type": "yarn"}, "index_n_heads": 2, "index_head_dim": 32, "index_topk": 4, "n_routed_experts": 8,
    \\ "num_experts_per_tok": 2, "num_hash_layers": 1, "n_shared_experts": 1, "moe_intermediate_size": 32,
    \\ "routed_scaling_factor": 1.5, "swiglu_limit": 10.0, "norm_topk_prob": true, "scoring_func": "sqrtsoftplus",
    \\ "topk_method": "noaux_tc", "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6, "rms_norm_eps": 1e-6,
    \\ "vocab_size": 64, "max_position_embeddings": 4096, "bos_token_id": 0, "eos_token_id": 1, "dspark_block_size": 3,
    \\ "dspark_noise_token_id": 63, "dspark_target_layer_ids": [1, 3], "dspark_markov_rank": 16,
    \\ "quantization": {"group_size": 32, "bits": 8, "mode": "affine"}}
;

test "deepseek parse: equals the engine's ModelConfig on every V4 and V4.1 config it claims" {
    try expectEngineParse(v4_config.release_config, false);
    try expectEngineParse(mini_config, false);
    try expectEngineParse(model.dsv41_release_config, true);
    try expectEngineParse(@embedFile("../../fixtures/model-configs/deepseek_v41-iq-mlx-3.0bpw.json"), true);
    try expectEngineParse(@embedFile("../../fixtures/model-configs/deepseek_v41-oq2.5e-mtp.json"), true);
    // A flat V4.1 (no text_config), null keys, a tied head, a nullable window: the readers' edge semantics.
    try expectEngineParse(
        \\{"model_type":"deepseek_v41","num_hidden_layers":3,"num_key_value_heads":1,"hc_mult":4,"sliding_window":null,
        \\ "compress_ratios":[0,2,2],"kv_source_layer_ids":[1],"index_source_layer_ids":[1],"rope_theta":null,
        \\ "tie_word_embeddings":true,"eos_token_id":[1,7],"max_position_embeddings":null,"image_token_id":5}
    , true);
}

test "deepseek parse: what the engine refuses the plugin refuses" {
    inline for (.{
        .{ "{\"model_type\":\"deepseek_v4\",\"num_key_value_heads\":2}", false },
        .{ "{\"model_type\":\"deepseek_v4\",\"num_key_value_heads\":1,\"topk_method\":\"greedy\"}", false },
        .{ "{\"model_type\":\"deepseek_v4\",\"num_key_value_heads\":1,\"vocab_size\":-1}", false },
        .{ "{\"model_type\":\"deepseek_v41\",\"num_key_value_heads\":1,\"num_hidden_layers\":2,\"hc_mult\":4,\"compress_ratios\":[0,3]}", true },
        .{ "{\"model_type\":\"deepseek_v41\",\"num_key_value_heads\":1,\"num_hidden_layers\":2,\"hc_mult\":2,\"compress_ratios\":[0,0]}", true },
    }) |case| {
        try testing.expect(std.meta.isError(model.parseConfigFromJson(testing.allocator, case[0])));
        const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = case[0] }} };
        const dir = mem.packDir();
        var p: abi.Parsed = .{};
        const v = if (case[1]) v41_config.parse(&dir, .of(case[0]), .{}, null, true, &p) else v4_config.parse(&dir, .of(case[0]), .{}, null, true, &p);
        try testing.expect(!v.isOk() and p.private == null);
    }
}
