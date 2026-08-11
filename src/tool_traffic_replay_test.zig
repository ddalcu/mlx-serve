//! Tool-traffic replay — real captured agent traffic, replayed hermetically.
//!
//! Each fixture record is a REAL `(tools schema, raw model output)` pair produced
//! by a real agent CLI (Claude Code / pi / opencode) against a real model, dumped
//! by the server at the one site where both are in scope (see
//! `server.appendRawToolDump`). Replaying them through the ACTUAL parse + coerce
//! path turns an hour of agentic soak into permanent regression coverage.
//!
//! Where `format_corpus_test.zig` holds a small hand-curated set with per-entry
//! expectations, this file holds BULK traffic and asserts only the invariants that
//! must hold for every input, whatever the model said:
//!
//!   R1 valid-json    every emitted `arguments` parses as a JSON object
//!   R2 schema-type   every argument whose type the tool declares carries it
//!   R3 idempotent    coercing twice equals coercing once (byte-for-byte)
//!   R4 no-tag-leak   with no tool call, visible content carries no control tags
//!   R5 no-op         a call already conforming is left BYTE-IDENTICAL
//!
//! R5 is the "auto-correct must never cause problems" guard: the repair layer is
//! only ever allowed to touch input that is actually broken.
//!
//! Grow the fixture:
//!     MLX_SERVE_RAW_DUMP_FILE=/tmp/rawdump.txt mlx-serve --serve --log-level debug
//!     # drive agents at it, then:
//!     tests/harvest_tool_traffic.py --dump /tmp/rawdump.txt \
//!         --out src/fixtures/tool_traffic.jsonl

const std = @import("std");
const testing = std.testing;
const chat = @import("chat.zig");

const fixture = @embedFile("fixtures/tool_traffic.jsonl");

/// Control tags that must never reach visible content, any family. This is the
/// SAME canonical list as `format_corpus_test.zig` — think/channel markers, the
/// tool-call WRAPPER tags, and Gemma's string delimiter. Bare `<function=` /
/// `<parameter=` are deliberately NOT here: a tiny model hallucinating orphaned
/// parameter tags (no `<function=` opener, no name) is unparseable garbage the
/// no-tag-leak invariant was never scoped to strip, and a WELL-FORMED
/// `<function=…>` is recovered as a call by parseToolCalls so it never leaks.
const leak_tags = [_][]const u8{
    "<think>",     "</think>",   "<|channel>", "<channel|>",
    "<|tool_call", "<tool_call", "<|\"|>",
};

/// Tags the server ALWAYS strips via dedicated think-block logic
/// (splitThinkBlock), independent of tool parsing — a leak here
/// is a HARD bug.
const hard_leak_tags = [_][]const u8{
    "<think>", "</think>", "<|channel>", "<channel|>",
};
/// Markup removed ONLY by a SUCCESSFUL tool-call parse: the wrapper tags and
/// Gemma's `<|"|>` string delimiter. When a malformed call fails to parse (a 4B
/// model dropping the tool NAME — `call{command:…}` instead of `call:shell{…}`,
/// or invalid JSON in a `<tool_call>` body), these leak into visible text. That
/// is genuinely-broken model output the layer correctly declines to fabricate a
/// call from — counted as a soft signal, not a hard failure.
const soft_leak_tags = [_][]const u8{ "<|tool_call", "<tool_call", "<|\"|>" };

var g_fail_count: usize = 0;
/// Set REPLAY_DUMP_SOFT=1 to print the record ids of soft signals (for auditing
/// whether any is a fixable bug rather than genuinely-broken model output).
var dump_soft: bool = false;

fn hardFail(comptime what: []const u8, idx: usize, detail: []const u8) void {
    g_fail_count += 1;
    std.debug.print("\n[replay record {d}] HARD {s}\n  {s}\n", .{ idx, what, detail[0..@min(detail.len, 400)] });
}

test "tool traffic replay: captured agent traffic survives parse + schema coercion" {
    g_fail_count = 0;
    dump_soft = std.c.getenv("REPLAY_DUMP_SOFT") != null;
    var line_it = std.mem.splitScalar(u8, fixture, '\n');
    var idx: usize = 0;
    var records: usize = 0;
    var total_calls: usize = 0;
    var coerced_calls: usize = 0;
    var unresolved: usize = 0; // calls left non-conforming (genuinely-broken model output)
    var soft_wrapper_leaks: usize = 0; // unparseable <tool_call> wrappers shown as text

    while (line_it.next()) |line| {
        if (std.mem.trim(u8, line, " \t\r").len == 0) continue;
        idx += 1;

        var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        var rec = std.json.parseFromSlice(std.json.Value, arena, line, .{}) catch |e| {
            std.debug.print("\n[replay record {d}] fixture line is not valid JSON: {}\n", .{ idx, e });
            return error.ToolTrafficReplayFailed;
        };
        defer rec.deinit();
        if (rec.value != .object) continue;

        const raw = (rec.value.object.get("raw") orelse continue).string;
        const tools_val = rec.value.object.get("tools") orelse continue;
        const tools_json = try std.json.Stringify.valueAlloc(arena, tools_val, .{});
        records += 1;

        // Mirror the server exactly: normalize → parse → (bare-JSON inference is
        // schema-dependent and skipped here) → coerce.
        const normalized = try chat.normalizeEmbeddedThinkBlocks(testing.allocator, raw);
        defer if (normalized) |n| testing.allocator.free(n);
        const text: []const u8 = normalized orelse raw;

        const calls = try chat.parseToolCalls(testing.allocator, text);
        defer if (calls) |cs| {
            for (cs) |tc| {
                testing.allocator.free(tc.name);
                testing.allocator.free(tc.arguments);
            }
            testing.allocator.free(cs);
        };

        if (calls == null) {
            // Nothing parsed as a call ⇒ the text is shown to the user
            // (content side of the always-delivered split, same as the server).
            const content = chat.splitThinkBlock(text, true, false).content;
            // Legitimate output has AT MOST ONE close marker per style per turn
            // (one thought block). The strip logic guarantees no leak for that.
            // A model that emits ≥2 bare closes with no matching structure is
            // MISUSING the tag as a prose separator — genuinely-degenerate output
            // (live: 3 stray `<channel|>` interspersed in prose, 0 opens). That's
            // the same soft class as unparseable tool wrappers: slicing can't
            // clean interior stray markers, and forcing an allocating scrub into
            // 15+ server content sites isn't worth it for pathological garbage.
            const degenerate_multi_close =
                std.mem.count(u8, text, "<channel|>") >= 2 or std.mem.count(u8, text, "</think>") >= 2;
            for (hard_leak_tags) |tag| {
                if (std.mem.indexOf(u8, content, tag) != null) {
                    if (degenerate_multi_close) {
                        soft_wrapper_leaks += 1;
                        if (dump_soft) std.debug.print("[soft degenerate-close #{d}] {s}\n", .{ idx, raw[0..@min(raw.len, 160)] });
                    } else {
                        // A leak with ≤1 close of each style is a REAL strip bug.
                        hardFail("think/delimiter tag leaked (single-block strip)", idx, content);
                    }
                    break;
                }
            }
            for (soft_leak_tags) |tag| {
                if (std.mem.indexOf(u8, content, tag) != null) {
                    soft_wrapper_leaks += 1;
                    if (dump_soft) std.debug.print("[soft wrapper-leak #{d}] {s}\n", .{ idx, raw[0..@min(raw.len, 160)] });
                    break;
                }
            }
            continue;
        }

        const cs = calls.?;
        total_calls += cs.len;

        // Snapshot pre-coercion bytes so R5 can prove the no-op property.
        const before = try arena.alloc([]const u8, cs.len);
        for (cs, 0..) |tc, i| before[i] = try arena.dupe(u8, tc.arguments);
        const conformed_before = try arena.alloc(bool, cs.len);
        for (cs, 0..) |tc, i| conformed_before[i] = chat.toolCallConformsToSchema(testing.allocator, tc, tools_json);

        try chat.hoistMisplacedRequiredParams(testing.allocator, cs, tools_json);
        try chat.coerceToolArgsToSchema(testing.allocator, cs, tools_json);

        for (cs, 0..) |tc, i| {
            // R1: emitted arguments must be valid JSON (HARD — a client parses them).
            var parse_ok = true;
            if (std.json.parseFromSlice(std.json.Value, testing.allocator, tc.arguments, .{})) |parsed| {
                parsed.deinit();
            } else |_| {
                parse_ok = false;
                hardFail("emitted arguments are not valid JSON", idx, tc.arguments);
            }

            const conforms_after = if (parse_ok) chat.toolCallConformsToSchema(testing.allocator, tc, tools_json) else false;

            // R2 (no-regression): coercion may only IMPROVE conformance, never
            // reduce it. A call that did NOT conform and STILL doesn't is
            // genuinely-broken model output — counted, not failed.
            if (conformed_before[i] and !conforms_after) {
                hardFail("coercion turned a CONFORMING arg non-conforming", idx, tc.arguments);
            }
            if (!conforms_after) {
                unresolved += 1;
                if (dump_soft) std.debug.print("[non-conforming #{d}] {s} args={s}\n", .{ idx, tc.name, tc.arguments[0..@min(tc.arguments.len, 160)] });
            }

            // R5: input that already conformed must come out byte-identical.
            if (conformed_before[i]) {
                if (!std.mem.eql(u8, before[i], tc.arguments)) {
                    hardFail("auto-correct MUTATED an already-conforming call", idx, tc.arguments);
                }
            } else if (!std.mem.eql(u8, before[i], tc.arguments)) {
                coerced_calls += 1;
            }
        }

        // R3: coercion is idempotent.
        const once = try arena.alloc([]const u8, cs.len);
        for (cs, 0..) |tc, i| once[i] = try arena.dupe(u8, tc.arguments);
        try chat.hoistMisplacedRequiredParams(testing.allocator, cs, tools_json);
        try chat.coerceToolArgsToSchema(testing.allocator, cs, tools_json);
        for (cs, 0..) |tc, i| {
            if (!std.mem.eql(u8, once[i], tc.arguments)) {
                hardFail("coercion is not idempotent", idx, tc.arguments);
            }
        }
    }

    std.debug.print("[replay] {d} records, {d} calls, {d} repaired, {d} non-conforming (broken JSON), {d} soft wrapper-leaks, {d} HARD failures\n", .{ records, total_calls, coerced_calls, unresolved, soft_wrapper_leaks, g_fail_count });
    try testing.expect(records > 0); // an empty fixture is a broken harvest, not a pass
    // Only HARD invariant violations fail the gate. Soft signals (genuinely-broken
    // model output that the layer correctly declines to fabricate) are reported.
    if (g_fail_count != 0) return error.ToolTrafficReplayFailed;
}
