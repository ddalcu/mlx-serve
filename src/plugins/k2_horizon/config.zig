//! K2-Horizon dense packs (`k2_horizon`): llama's layout and parse with grouped RMS norms (`layernorm_num_groups`),
//! plus the family's chat facts (thinking on, its effort words, the reasoning field its template requires) and its
//! `<ifm|…>` marker aliases. MLX-free. MoVA sizes are refused `unsupported`.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const llama = @import("../llama/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "k2_horizon",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"LLM360/K2-Horizon-7B"},
};

pub const model_types = [_][]const u8{"k2_horizon"};

pub const claims = sdk.claimsByType(&model_types);

pub const batches_decode = llama.batches_decode;

pub const Config = llama.Config;
pub const of = llama.of;
pub const freeParsed = llama.freeParsed;

const family: llama.Family = .{ .name = "k2_horizon", .model_types = &model_types, .grouped_norms = true };

/// The `<ifm|…>` specials as the `<think>` / GLM tool tags they stand for.
const aliases = blk: {
    const pairs = [_][2][]const u8{
        .{ "<ifm|think>", "<think>" },             .{ "</ifm|think>", "</think>" },
        .{ "<ifm|think_fast>", "<think>" },        .{ "</ifm|think_fast>", "</think>" },
        .{ "<ifm|think_faster>", "<think>" },      .{ "</ifm|think_faster>", "</think>" },
        .{ "<ifm|tool_calls>", "<tool_calls>" },   .{ "</ifm|tool_calls>", "</tool_calls>" },
        .{ "<ifm|tool_call>", "<tool_call>" },     .{ "</ifm|tool_call>", "</tool_call>" },
        .{ "<ifm|arg_key>", "<arg_key>" },         .{ "</ifm|arg_key>", "</arg_key>" },
        .{ "<ifm|arg_value>", "<arg_value>" },     .{ "</ifm|arg_value>", "</arg_value>" },
    };
    var out: [pairs.len]abi.MarkerAlias = undefined;
    for (pairs, &out) |p, *o| o.* = .{ .from = abi.Str.of(p[0]), .to = abi.Str.of(p[1]) };
    break :blk out;
};

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    const v = llama.parseFamily(family, dir, config_json, span, stub, out);
    if (!v.isOk()) return v;
    out.chat.thinking_default = .on;
    out.chat.effort_vocab = .k2;
    out.chat.reasoning_round_trip = .required_field;
    out.tokenizer.marker_aliases = .of(&aliases);
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

const k2_7b =
    \\{"model_type":"k2_horizon","hidden_size":4096,"intermediate_size":12288,"num_hidden_layers":36,
    \\ "num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"hidden_act":"silu","rms_norm_eps":1e-06,
    \\ "vocab_size":250624,"layernorm_num_groups":4,"query_key_norm":false,"attention_gate_func":null,
    \\ "num_experts":0,"sliding_window":null,"tie_word_embeddings":false,"max_position_embeddings":524288,
    \\ "rope_head_dim":128,"rope_parameters":{"rope_theta":10000000.0,"rope_type":"default"}}
;

test "k2_horizon config: the 7B config parses its grouped norms, rope and chat facts" {
    var p: abi.Parsed = .{};
    try testing.expect(parseJson(k2_7b, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqual(@as(u32, 4), of(&p).norm_groups);
    try testing.expectEqual(@as(f32, 10000000.0), of(&p).rope_theta);
    try testing.expect(p.chat.thinking_default == .on and p.chat.effort_vocab == .k2);
    try testing.expectEqual(@as(usize, 14), p.tokenizer.marker_aliases.len);
    try testing.expectEqualStrings("<think>", p.tokenizer.marker_aliases.slice()[0].to.slice());
}

test "k2_horizon config: MoVA, q/k norms and an attention gate are refused unsupported" {
    inline for (.{ "\"num_experts\":0", "\"query_key_norm\":false", "\"attention_gate_func\":null" }, .{ "\"num_experts\":8", "\"query_key_norm\":true", "\"attention_gate_func\":\"sigmoid\"" }) |from, to| {
        const json = comptime blk: {
            @setEvalBranchQuota(100_000);
            const at = std.mem.indexOf(u8, k2_7b, from).?;
            break :blk k2_7b[0..at] ++ to ++ k2_7b[at + from.len ..];
        };
        var p: abi.Parsed = .{};
        try testing.expectEqual(abi.Verdict.Code.unsupported, parseJson(json, &p).code);
    }
}
