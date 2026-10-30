//! The refusal a connection thread last met from an engine that does not declare what its request needs, named
//! once in the 400 (`refusalMessage`).

const std = @import("std");
const served_mod = @import("served.zig");
const Kind = served_mod.Kind;
const SharedThreadLocal = @import("../../sdk/util/shared.zig").SharedThreadLocal;

pub const Refused = ?struct { kind: Kind, cap: []const u8 };

/// Set by the engine on the connection thread that submitted; read by `mapGenerationError` on the same thread.
pub const refused_tls = SharedThreadLocal(Refused, "engine_refused", null);

pub fn refused() *Refused {
    return refused_tls.get();
}

/// The 400 for `refused`: the engine, the feature, the fix.
pub fn refusalMessage(buf: []u8) []const u8 {
    const r = refused().* orelse return "This model's engine does not serve this request.";
    refused().* = null;
    const fix = if (r.kind == .ane) "Use a GPU model for that." else "Use an MLX model for that.";
    return std.fmt.bufPrint(buf, "This model runs on {s}, which does not serve {s}. {s}", .{ served_mod.displayName(r.kind), feature(r.cap), fix }) catch
        "This model's engine does not serve this request.";
}

fn feature(cap: []const u8) []const u8 {
    const names = .{
        .{ "penalties", "repetition, frequency or presence penalties" },
        .{ "logit_bias", "logit_bias" },
        .{ "logprobs", "logprobs" },
        .{ "grammar", "structured output (response_format, grammars)" },
        .{ "think_budget", "a reasoning budget" },
        .{ "tool_gate", "forced tool calls (tool_choice)" },
        .{ "prompt_logprobs", "prompt logprobs" },
    };
    inline for (names) |n| if (std.mem.eql(u8, cap, n[0])) return n[1];
    return cap;
}

// ── tests ──

const testing = std.testing;

test "refusalMessage: names the engine, the feature and the fix, once" {
    var buf: [320]u8 = undefined;
    refused().* = .{ .kind = .ane, .cap = "tool_gate" };
    try testing.expectEqualStrings("This model runs on the Neural Engine, which does not serve forced tool calls (tool_choice). Use a GPU model for that.", refusalMessage(&buf));
    try testing.expect(refused().* == null);
    refused().* = .{ .kind = .llama, .cap = "penalties" };
    try testing.expectEqualStrings("This model runs on the embedded llama.cpp engine (GGUF), which does not serve repetition, frequency or presence penalties. Use an MLX model for that.", refusalMessage(&buf));
}
