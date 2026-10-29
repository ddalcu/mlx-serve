//! Released MLX packs' config.json (`fixtures/mlx_lm/`, per-tensor quant overrides and Prism's `modules` trimmed to
//! two) through the claim and the parse a served load runs. mlx-lm's `save_config` writes the quant block twice
//! (`quantization` and `quantization_config`); each config here carries both, and a `quantization_config` naming a
//! `quant_method` is refused by name.

const std = @import("std");
const abi = @import("../../../abi/plugin.zig");
const MemDir = @import("../../../sdk/root.zig").fixture.MemDir;
const sdk_plugin = @import("../../../sdk/plugin.zig");
const plugin_config = @import("config.zig");
const served = @import("served.zig");

const testing = std.testing;

const Case = struct { file: []const u8, json: []const u8 };

fn case(comptime file: []const u8) Case {
    return .{ .file = file, .json = @embedFile("fixtures/mlx_lm/" ++ file ++ ".json") };
}

fn deepseek(comptime file: []const u8) Case {
    return .{ .file = file, .json = @embedFile("../../../plugins/deepseek_v4/fixtures/real/" ++ file ++ ".json") };
}

fn gemma(comptime file: []const u8) Case {
    return .{ .file = file, .json = @embedFile("../../../fixtures/model-configs/" ++ file ++ ".json") };
}

fn pinned(comptime file: []const u8) Case {
    return .{ .file = file, .json = @embedFile("../../../fixtures/model-configs/" ++ file ++ ".json") };
}

const cases = [_]Case{
    case("mlx-community_Llama-3.2-1B-Instruct-4bit"),
    case("mlx-community_Mistral-7B-Instruct-v0.3-4bit"),
    case("mlx-community_Qwen2.5-0.5B-Instruct-4bit"),
    case("mlx-community_Qwen3-0.6B-4bit"),
    case("mlx-community_Qwen3-30B-A3B-4bit"),
    case("mlx-community_Qwen3.5-0.8B-MLX-4bit"),
    case("mlx-community_Qwen3.6-35B-A3B-4bit"),
    case("mlx-community_Qwen3-Next-80B-A3B-Instruct-4bit"),
    case("mlx-community_Qwen3-ASR-0.6B-8bit"),
    case("prism-ml_Ternary-Bonsai-2-27B-mlx-2bit"),
    .{ .file = "minicpmv4_7", .json = @embedFile("../../../fixtures/model-configs/minicpmv4_7.json") },
    .{ .file = "qwen4_exp-iq-mlx-4.7bpw", .json = @embedFile("../../../fixtures/model-configs/qwen4_exp-iq-mlx-4.7bpw.json") },
    case("mlx-community_K2-Horizon-7B-oQ6e"),
    case("abenzerps_Spark-X2.5-4B-MLX-4bit"),
    case("poolside_Laguna-XS-2.1-NVFP4-mlx"),
    case("RecursiveSI_flint-int4"),
    case("here-be-dragons-ai_Kolibri-1-MLX-3bit"),
    case("mlx-community_Laguna-S-2.1-oQ4e-fast"),
    case("ddalcu_Muse-Glimmer-30B-MLX-Serve-4bit"),
    case("ddalcu_Muse-Glimmer-30B-text"),
    case("mlx-community_gpt-oss-20b-MXFP4-Q8"),
    case("mlx-community_MiMo-V2-Flash-4bit"),
    case("mlx-community_all-MiniLM-L6-v2-4bit"),
    case("mlx-community_LFM2-1.2B-4bit"),
    case("mlx-community_LFM2-8B-A1B-4bit"),
    case("LiquidAI_LFM3-1.7B-MLX-8bit"),
    case("LiquidAI_LFM3-VL-2B-MLX-8bit"),
    case("LiquidAI_LFM3-Audio-2B-MLX-8bit"),
    case("LiquidAI_LFM3-Omni-2B-MLX-8bit"),
    case("mlx-community_LFM2.5-VL-1.6B-4bit"),
    case("mlx-community_LFM2-VL-1.6B-4bit"),
    case("mlx-community_Hy3-oQ2e"),
    case("mlx-community_DeepSeek-V4.1-Flash-1017-oQ4e-mtp"),
    // DeepSeek-V4-Flash-4bit is the withdrawn preview, refused by both parses (`deepseek_v4/config.zig`).
    deepseek("mlx-community_DeepSeek-V4-Flash-0731-2.4bit-mixed"),
    case("mlx-community_Inkling-Small-mlx-2bit"),
    case("mlx-community_Qwen3.8-Flash-Next-4bit"),
    case("Jundot_Qwen3.8-Flash-Next-oQ4e-mtp"),
    pinned("glm5_next-mlx-community-4bit"),
    pinned("bailing_hybrid-ling-3.0-tiny-4bit"),
    gemma("gemma-3-1b-it-4bit"),
    gemma("gemma-3-4b-it-4bit"),
    gemma("gemma-4-e4b-it-4bit"),
    gemma("embeddinggemma-300m-8bit"),
    gemma("gemma-4-12b-it-4bit"),
};

/// Served types no released mlx-lm config here reaches yet.
const uncovered = [_][]const u8{
    // No mlx-lm pack of these: the original release (refused by name), and the configs the families pin themselves.
    "mimo_v2",          "nemotron_h_puzzle",
    "nemotron_h_omni",  "gemma4_text",
    "embedding_gemma2", "gemma4_unified_text",
    "qwen3_moe_text",   "qwen4_exp_text",
    "qwen3_5_text",     "qwen3_5_moe_text",
    "glm5_next_text",
    // mlx-community's Nemotron-Nano-9B-v2 has a bare `Infinity`, which the legacy parse refuses too.
      "nemotron_h",
};

/// `json` with `quantization_config` a copy of its `quantization` block, as mlx-lm writes it.
fn withCopy(a: std.mem.Allocator, json: []const u8) ![]u8 {
    var v = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
    if (v.object.get("quantization")) |q| try v.object.put(a, "quantization_config", q);
    return std.json.Stringify.valueAlloc(a, v, .{});
}

fn verdict(a: std.mem.Allocator, json: []const u8) !abi.Verdict {
    const mem: MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = json }} };
    const dir = mem.packDir();
    const mt = try sdk_plugin.modelTypeOf(a, json);
    const table = plugin_config.claimant(mt, json, &dir) orelse return abi.Verdict.refuse(.unsupported, "no claimant");
    var p: abi.Parsed = .{};
    const v = table.parse(&dir, abi.Str.of(json), .{}, null, false, &p);
    if (v.isOk()) table.free_parsed(&p);
    return v;
}

test "mlx-lm configs: every served type's released pack, quantization_config beside quantization, is claimed and parsed" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    for (cases) |c| {
        const json = try withCopy(a, c.json);
        const v = try verdict(a, json);
        if (!v.isOk()) std.debug.print("{s}: {s}\n", .{ c.file, v.why.slice() });
        try testing.expect(v.isOk());
        try seen.put(a, try sdk_plugin.modelTypeOf(a, json), {});
    }
    for (served.types) |t| {
        if (seen.contains(t)) continue;
        for (uncovered) |u| {
            if (std.mem.eql(u8, u, t)) break;
        } else {
            std.debug.print("{s}: served, no released config here\n", .{t});
            return error.Uncovered;
        }
    }
}

test "mlx-lm configs: a quantization_config naming a quant_method (openai/gpt-oss-20b's mxfp4 release) is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v = try verdict(arena.allocator(), @embedFile("fixtures/mlx_lm/openai_gpt-oss-20b.json"));
    try testing.expectEqual(abi.Verdict.Code.unsupported, v.code);
    try testing.expectEqualStrings("gpt_oss: a quantization_config naming a quant_method", v.why.slice());
}

test {
    _ = @import("../../../sdk/config/quantization_config.zig");
}
