//! echoes, envelope, event emitters, storeResponse

const std = @import("std");
const chat_mod = @import("../../chat/chat.zig");
const responses_mod = @import("model.zig");
const io_util = @import("../../../sdk/util/io_util.zig");
const nowSecs = io_util.nowSecs;

const Conn = @import("../../http/conn.zig").Conn;
const shouldParseToolCalls = @import("../../chat/tools/pipeline.zig").shouldParseToolCalls;
const getOrInitResponseStore = @import("store.zig").getOrInitResponseStore;
const formatTimingsObject = @import("../format.zig").formatTimingsObject;
const jsonEscape = @import("../format.zig").jsonEscape;
const sendResponsesEvent = @import("events.zig").sendResponsesEvent;

pub fn renderResponsesToolsEcho(allocator: std.mem.Allocator, tools_val: ?std.json.Value) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    try buf.append(allocator, '[');
    var emitted: usize = 0;
    if (tools_val) |tv| if (tv == .array) {
        for (tv.array.items) |tool_val| {
            if (tool_val != .object) continue;
            const tool = tool_val.object;
            const t = if (tool.get("type")) |x| (if (x == .string) x.string else "") else "";
            if (!std.mem.eql(u8, t, "function")) continue;

            // Accept both flat (Responses API) and nested-under-"function" (chat-completions) shapes.
            var fn_obj_opt: ?std.json.ObjectMap = null;
            if (tool.get("function")) |fv| if (fv == .object) {
                fn_obj_opt = fv.object;
            };

            const name_v: ?std.json.Value = tool.get("name") orelse if (fn_obj_opt) |fo| fo.get("name") else null;
            const desc_v: ?std.json.Value = tool.get("description") orelse if (fn_obj_opt) |fo| fo.get("description") else null;
            const params_v: ?std.json.Value = tool.get("parameters") orelse if (fn_obj_opt) |fo| fo.get("parameters") else null;
            const strict_v: ?std.json.Value = tool.get("strict") orelse if (fn_obj_opt) |fo| fo.get("strict") else null;

            if (emitted > 0) try buf.append(allocator, ',');
            emitted += 1;
            try buf.appendSlice(allocator, "{\"type\":\"function\",\"name\":");
            if (name_v) |nv| if (nv == .string) {
                const e = try jsonEscape(allocator, nv.string);
                defer allocator.free(e);
                try buf.appendSlice(allocator, e);
            } else {
                try buf.appendSlice(allocator, "\"\"");
            } else {
                try buf.appendSlice(allocator, "\"\"");
            }
            try buf.appendSlice(allocator, ",\"description\":");
            if (desc_v) |dv| if (dv == .string) {
                const e = try jsonEscape(allocator, dv.string);
                defer allocator.free(e);
                try buf.appendSlice(allocator, e);
            } else {
                try buf.appendSlice(allocator, "null");
            } else {
                try buf.appendSlice(allocator, "null");
            }
            try buf.appendSlice(allocator, ",\"parameters\":");
            if (params_v) |pv| if (pv == .object) {
                try responses_mod.serializeJsonValue(allocator, &buf, pv);
            } else {
                try buf.appendSlice(allocator, "null");
            } else {
                try buf.appendSlice(allocator, "null");
            }
            try buf.appendSlice(allocator, ",\"strict\":");
            if (strict_v) |sv| if (sv == .bool) {
                try buf.appendSlice(allocator, if (sv.bool) "true" else "false");
            } else {
                try buf.appendSlice(allocator, "null");
            } else {
                try buf.appendSlice(allocator, "null");
            }
            try buf.append(allocator, '}');
        }
    };
    try buf.append(allocator, ']');
    return try buf.toOwnedSlice(allocator);
}

pub fn renderResponsesToolChoiceEcho(allocator: std.mem.Allocator, tc_val: ?std.json.Value) ![]const u8 {
    if (tc_val) |v| switch (v) {
        .string => |s| {
            if (std.mem.eql(u8, s, "auto") or std.mem.eql(u8, s, "none") or std.mem.eql(u8, s, "required")) {
                return try std.fmt.allocPrint(allocator, "\"{s}\"", .{s});
            }
        },
        .object => |obj| {
            const t = if (obj.get("type")) |x| (if (x == .string) x.string else "") else "";
            if (std.mem.eql(u8, t, "function")) {
                var name: []const u8 = "";
                if (obj.get("name")) |x| {
                    if (x == .string) name = x.string;
                } else if (obj.get("function")) |fv| if (fv == .object) {
                    if (fv.object.get("name")) |nv| if (nv == .string) {
                        name = nv.string;
                    };
                };
                if (name.len > 0) {
                    const esc = try jsonEscape(allocator, name);
                    defer allocator.free(esc);
                    return try std.fmt.allocPrint(allocator, "{{\"type\":\"function\",\"name\":{s}}}", .{esc});
                }
                return try allocator.dupe(u8, "{\"type\":\"function\"}");
            }
        },
        else => {},
    };
    return try allocator.dupe(u8, "\"auto\"");
}

pub fn renderResponsesTextEcho(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]const u8 {
    if (root.get("text")) |tv| if (tv == .object) {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try responses_mod.serializeJsonValue(allocator, &buf, tv);
        return try buf.toOwnedSlice(allocator);
    };
    if (root.get("response_format")) |rf| if (rf == .object) {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try buf.appendSlice(allocator, "{\"format\":");
        try responses_mod.serializeJsonValue(allocator, &buf, rf);
        try buf.append(allocator, '}');
        return try buf.toOwnedSlice(allocator);
    };
    return try allocator.dupe(u8, "{\"format\":{\"type\":\"text\"}}");
}

pub fn renderResponsesReasoningEcho(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]const u8 {
    var effort_buf: [32]u8 = undefined;
    var effort_str: []const u8 = "null";
    var summary_str: []const u8 = "null";
    if (root.get("reasoning")) |rv| if (rv == .object) {
        if (rv.object.get("effort")) |ev| if (ev == .string) {
            const s = ev.string;
            const valid = std.mem.eql(u8, s, "minimal") or
                std.mem.eql(u8, s, "low") or
                std.mem.eql(u8, s, "medium") or
                std.mem.eql(u8, s, "high");
            if (valid) {
                effort_str = std.fmt.bufPrint(&effort_buf, "\"{s}\"", .{s}) catch "null";
            }
        };
        if (rv.object.get("summary")) |sv| if (sv == .string) {
            const s = sv.string;
            const valid = std.mem.eql(u8, s, "auto") or
                std.mem.eql(u8, s, "concise") or
                std.mem.eql(u8, s, "detailed");
            if (valid) {
                // share buffer is fine since alloc happens immediately after
                if (std.mem.eql(u8, s, "auto")) summary_str = "\"auto\"" else if (std.mem.eql(u8, s, "concise")) summary_str = "\"concise\"" else summary_str = "\"detailed\"";
            }
        };
    };
    return try std.fmt.allocPrint(allocator, "{{\"effort\":{s},\"summary\":{s}}}", .{ effort_str, summary_str });
}

pub fn renderResponsesMetadataEcho(allocator: std.mem.Allocator, root: std.json.ObjectMap) ![]const u8 {
    if (root.get("metadata")) |mv| if (mv == .object) {
        var buf = std.ArrayList(u8).empty;
        errdefer buf.deinit(allocator);
        try responses_mod.serializeJsonValue(allocator, &buf, mv);
        return try buf.toOwnedSlice(allocator);
    };
    return try allocator.dupe(u8, "{}");
}

/// Echoed-back fields that round out the OpenAI Responses envelope.
/// Owned by the caller (POST handler); freed after the final envelope is built.
pub const ResponseEcho = struct {
    // Pre-rendered JSON fragments (raw object/array text — caller owns).
    tools_json: []const u8 = "[]",
    tool_choice_json: []const u8 = "\"auto\"",
    text_json: []const u8 = "{\"format\":{\"type\":\"text\"}}",
    reasoning_json: []const u8 = "{\"effort\":null,\"summary\":null}",
    metadata_json: []const u8 = "{}",
    // Plain values; serialized inline.
    instructions: ?[]const u8 = null,
    truncation: []const u8 = "disabled",
    service_tier: []const u8 = "default",
    safety_identifier: ?[]const u8 = null,
    prompt_cache_key: ?[]const u8 = null,
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    presence_penalty: f32 = 0.0,
    frequency_penalty: f32 = 0.0,
    top_logprobs: u32 = 0,
    parallel_tool_calls: bool = true,
    background: bool = false,
    max_output_tokens: ?u32 = null,
    max_tool_calls: ?u32 = null,
};

/// Build the Responses envelope JSON body. Used for both the response.created
/// skeleton (in_progress, output:[]) and the final response.completed body.
/// Shape matches the OpenAI Responses API ResponseResource schema.
pub fn buildResponsesEnvelope(
    io: std.Io,
    allocator: std.mem.Allocator,
    esc_resp_id: []const u8,
    esc_model: []const u8,
    status_str: []const u8,
    output_json: []const u8,
    input_tokens: u32,
    output_tokens: u32,
    cached_input_tokens: u32,
    reasoning_output_tokens: u32,
    should_store: bool,
    prev_id: ?[]const u8,
    incomplete: bool,
    completed: bool,
    echo: ResponseEcho,
    /// Iteration 1: timings extension. The Responses spec doesn't model
    /// per-stage timings, so we attach a sibling `timings` block at the
    /// envelope root with the same shape as /v1/chat/completions. Pass
    /// zeroes to omit the field (legacy callers stay shape-compatible).
    prefill_ns: u64,
    decode_ns: u64,
    tokenize_ns: u64,
    completion_tokens_for_timings: u32,
) ![]const u8 {
    var buf = std.ArrayList(u8).empty;
    errdefer buf.deinit(allocator);
    var num_buf: [32]u8 = undefined;

    try buf.append(allocator, '{');
    try buf.appendSlice(allocator, "\"id\":");
    try buf.appendSlice(allocator, esc_resp_id);

    try buf.appendSlice(allocator, ",\"object\":\"response\",\"created_at\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{nowSecs(io)}));

    if (completed) {
        try buf.appendSlice(allocator, ",\"completed_at\":");
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{nowSecs(io)}));
    } else {
        try buf.appendSlice(allocator, ",\"completed_at\":null");
    }

    try buf.appendSlice(allocator, ",\"status\":\"");
    try buf.appendSlice(allocator, status_str);
    try buf.append(allocator, '"');

    if (incomplete) {
        try buf.appendSlice(allocator, ",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}");
    } else {
        try buf.appendSlice(allocator, ",\"incomplete_details\":null");
    }

    try buf.appendSlice(allocator, ",\"model\":");
    try buf.appendSlice(allocator, esc_model);

    if (prev_id) |pid| {
        const esc_pid = try jsonEscape(allocator, pid);
        defer allocator.free(esc_pid);
        try buf.appendSlice(allocator, ",\"previous_response_id\":");
        try buf.appendSlice(allocator, esc_pid);
    } else {
        try buf.appendSlice(allocator, ",\"previous_response_id\":null");
    }

    if (echo.instructions) |s| {
        const esc = try jsonEscape(allocator, s);
        defer allocator.free(esc);
        try buf.appendSlice(allocator, ",\"instructions\":");
        try buf.appendSlice(allocator, esc);
    } else {
        try buf.appendSlice(allocator, ",\"instructions\":null");
    }

    try buf.appendSlice(allocator, ",\"output\":");
    try buf.appendSlice(allocator, output_json);

    try buf.appendSlice(allocator, ",\"error\":null");

    try buf.appendSlice(allocator, ",\"tools\":");
    try buf.appendSlice(allocator, echo.tools_json);

    try buf.appendSlice(allocator, ",\"tool_choice\":");
    try buf.appendSlice(allocator, echo.tool_choice_json);

    try buf.appendSlice(allocator, ",\"truncation\":\"");
    try buf.appendSlice(allocator, echo.truncation);
    try buf.append(allocator, '"');

    try buf.appendSlice(allocator, ",\"parallel_tool_calls\":");
    try buf.appendSlice(allocator, if (echo.parallel_tool_calls) "true" else "false");

    try buf.appendSlice(allocator, ",\"text\":");
    try buf.appendSlice(allocator, echo.text_json);

    try buf.appendSlice(allocator, ",\"top_p\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.top_p}));
    try buf.appendSlice(allocator, ",\"presence_penalty\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.presence_penalty}));
    try buf.appendSlice(allocator, ",\"frequency_penalty\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.frequency_penalty}));
    try buf.appendSlice(allocator, ",\"top_logprobs\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.top_logprobs}));
    try buf.appendSlice(allocator, ",\"temperature\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{echo.temperature}));

    try buf.appendSlice(allocator, ",\"reasoning\":");
    try buf.appendSlice(allocator, echo.reasoning_json);

    try buf.appendSlice(allocator, ",\"usage\":{\"input_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{input_tokens}));
    try buf.appendSlice(allocator, ",\"output_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{output_tokens}));
    try buf.appendSlice(allocator, ",\"total_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{input_tokens + output_tokens}));
    try buf.appendSlice(allocator, ",\"input_tokens_details\":{\"cached_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{cached_input_tokens}));
    try buf.appendSlice(allocator, "},\"output_tokens_details\":{\"reasoning_tokens\":");
    try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{reasoning_output_tokens}));
    try buf.appendSlice(allocator, "}}");

    if (echo.max_output_tokens) |n| {
        try buf.appendSlice(allocator, ",\"max_output_tokens\":");
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{n}));
    } else {
        try buf.appendSlice(allocator, ",\"max_output_tokens\":null");
    }

    if (echo.max_tool_calls) |n| {
        try buf.appendSlice(allocator, ",\"max_tool_calls\":");
        try buf.appendSlice(allocator, try std.fmt.bufPrint(&num_buf, "{d}", .{n}));
    } else {
        try buf.appendSlice(allocator, ",\"max_tool_calls\":null");
    }

    try buf.appendSlice(allocator, ",\"store\":");
    try buf.appendSlice(allocator, if (should_store) "true" else "false");

    try buf.appendSlice(allocator, ",\"background\":");
    try buf.appendSlice(allocator, if (echo.background) "true" else "false");

    try buf.appendSlice(allocator, ",\"service_tier\":\"");
    try buf.appendSlice(allocator, echo.service_tier);
    try buf.append(allocator, '"');

    try buf.appendSlice(allocator, ",\"metadata\":");
    try buf.appendSlice(allocator, echo.metadata_json);

    if (echo.safety_identifier) |s| {
        const esc = try jsonEscape(allocator, s);
        defer allocator.free(esc);
        try buf.appendSlice(allocator, ",\"safety_identifier\":");
        try buf.appendSlice(allocator, esc);
    } else {
        try buf.appendSlice(allocator, ",\"safety_identifier\":null");
    }

    if (echo.prompt_cache_key) |s| {
        const esc = try jsonEscape(allocator, s);
        defer allocator.free(esc);
        try buf.appendSlice(allocator, ",\"prompt_cache_key\":");
        try buf.appendSlice(allocator, esc);
    } else {
        try buf.appendSlice(allocator, ",\"prompt_cache_key\":null");
    }

    // Iteration 1 timings extension. Reuses the chat-completions
    // formatter so any future field added there appears on Responses too
    // without a second touch point.
    const timings_obj = try formatTimingsObject(allocator, input_tokens, cached_input_tokens, completion_tokens_for_timings, prefill_ns, decode_ns, tokenize_ns);
    defer allocator.free(timings_obj);
    if (timings_obj.len > 0) {
        try buf.appendSlice(allocator, ",\"timings\":");
        try buf.appendSlice(allocator, timings_obj);
    }

    try buf.append(allocator, '}');
    return try buf.toOwnedSlice(allocator);
}

/// Emit output_item.added (type=reasoning) + reasoning_summary_part.added.
/// Pair with `emitResponsesReasoningEnd` after deltas are streamed.
pub fn emitResponsesReasoningStart(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    {
        const item_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.added","output_index":{d},"item":{{"type":"reasoning","id":{s},"summary":[]}}}}
        , .{ output_index, esc_id });
        defer allocator.free(item_added);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.added", item_added);
    }
    {
        const part_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.reasoning_summary_part.added","item_id":{s},"output_index":{d},"summary_index":0,"part":{{"type":"summary_text","text":""}}}}
        , .{ esc_id, output_index });
        defer allocator.free(part_added);
        try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_part.added", part_added);
    }
}

/// Emit a single reasoning_summary_text.delta event with the given chunk.
pub fn emitResponsesReasoningDelta(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    delta_text: []const u8,
) !void {
    if (delta_text.len == 0) return;
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_delta = try jsonEscape(allocator, delta_text);
    defer allocator.free(esc_delta);
    const delta = try std.fmt.allocPrint(allocator,
        \\{{"type":"response.reasoning_summary_text.delta","item_id":{s},"output_index":{d},"summary_index":0,"delta":{s}}}
    , .{ esc_id, output_index, esc_delta });
    defer allocator.free(delta);
    try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_text.delta", delta);
}

/// Emit reasoning_summary_text.done + reasoning_summary_part.done + output_item.done.
pub fn emitResponsesReasoningEnd(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    full_text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, full_text);
    defer allocator.free(esc_text);
    {
        const done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.reasoning_summary_text.done","item_id":{s},"output_index":{d},"summary_index":0,"text":{s}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(done);
        try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_text.done", done);
    }
    {
        const part_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.reasoning_summary_part.done","item_id":{s},"output_index":{d},"summary_index":0,"part":{{"type":"summary_text","text":{s}}}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(part_done);
        try sendResponsesEvent(allocator, stream, seq, "response.reasoning_summary_part.done", part_done);
    }
    {
        const item_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.done","output_index":{d},"item":{{"type":"reasoning","id":{s},"summary":[{{"type":"summary_text","text":{s}}}]}}}}
        , .{ output_index, esc_id, esc_text });
        defer allocator.free(item_done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.done", item_done);
    }
}

/// Emit a full reasoning output item in one shot. Used when the entire
/// reasoning text is known up-front (non-streaming generation paths).
pub fn emitResponsesReasoningEvents(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    reasoning_text: []const u8,
) !void {
    try emitResponsesReasoningStart(allocator, stream, seq, output_index, item_id);
    try emitResponsesReasoningDelta(allocator, stream, seq, output_index, item_id, reasoning_text);
    try emitResponsesReasoningEnd(allocator, stream, seq, output_index, item_id, reasoning_text);
}

/// Emit the SSE event sequence for a function_call output item: output_item.added,
/// function_call_arguments.delta (single full-args delta), .done, output_item.done.
pub fn emitResponsesFunctionCallEvents(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    fc_id: []const u8,
    call_id: []const u8,
    name: []const u8,
    arguments_json: []const u8,
    namespace: ?[]const u8,
) !void {
    const esc_id = try jsonEscape(allocator, fc_id);
    defer allocator.free(esc_id);
    const esc_call = try jsonEscape(allocator, call_id);
    defer allocator.free(esc_call);
    const esc_name = try jsonEscape(allocator, name);
    defer allocator.free(esc_name);
    const esc_args = try jsonEscape(allocator, arguments_json);
    defer allocator.free(esc_args);
    // A namespace-qualified call carries "namespace" so the client resolves it
    // by (namespace, name); flat calls serialize no such field.
    var ns_field: []const u8 = "";
    var ns_owned: ?[]u8 = null;
    defer if (ns_owned) |o| allocator.free(o);
    if (namespace) |ns| {
        const esc_ns = try jsonEscape(allocator, ns);
        defer allocator.free(esc_ns);
        ns_owned = try std.fmt.allocPrint(allocator, ",\"namespace\":{s}", .{esc_ns});
        ns_field = ns_owned.?;
    }

    {
        const item_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.added","output_index":{d},"item":{{"type":"function_call","id":{s},"call_id":{s},"name":{s}{s},"arguments":"","status":"in_progress"}}}}
        , .{ output_index, esc_id, esc_call, esc_name, ns_field });
        defer allocator.free(item_added);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.added", item_added);
    }
    {
        const delta = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.function_call_arguments.delta","item_id":{s},"output_index":{d},"delta":{s}}}
        , .{ esc_id, output_index, esc_args });
        defer allocator.free(delta);
        try sendResponsesEvent(allocator, stream, seq, "response.function_call_arguments.delta", delta);
    }
    {
        const done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.function_call_arguments.done","item_id":{s},"output_index":{d},"arguments":{s}}}
        , .{ esc_id, output_index, esc_args });
        defer allocator.free(done);
        try sendResponsesEvent(allocator, stream, seq, "response.function_call_arguments.done", done);
    }
    {
        const item_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.done","output_index":{d},"item":{{"type":"function_call","id":{s},"call_id":{s},"name":{s}{s},"arguments":{s},"status":"completed"}}}}
        , .{ output_index, esc_id, esc_call, esc_name, ns_field, esc_args });
        defer allocator.free(item_done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.done", item_done);
    }
}

/// Emit output_item.added (type=message) + content_part.added.
/// Pair with `emitResponsesMessageEnd` after output_text.delta events stream.
pub fn emitResponsesMessageStart(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    {
        const item_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.added","output_index":{d},"item":{{"type":"message","id":{s},"role":"assistant","status":"in_progress","content":[]}}}}
        , .{ output_index, esc_id });
        defer allocator.free(item_added);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.added", item_added);
    }
    {
        const part_added = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.content_part.added","item_id":{s},"output_index":{d},"content_index":0,"part":{{"type":"output_text","text":"","annotations":[]}}}}
        , .{ esc_id, output_index });
        defer allocator.free(part_added);
        try sendResponsesEvent(allocator, stream, seq, "response.content_part.added", part_added);
    }
}

/// Emit a single output_text.delta event for an in-progress message.
pub fn emitResponsesMessageDelta(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    delta_text: []const u8,
) !void {
    if (delta_text.len == 0) return;
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_delta = try jsonEscape(allocator, delta_text);
    defer allocator.free(esc_delta);
    const delta = try std.fmt.allocPrint(allocator,
        \\{{"type":"response.output_text.delta","item_id":{s},"output_index":{d},"content_index":0,"delta":{s}}}
    , .{ esc_id, output_index, esc_delta });
    defer allocator.free(delta);
    try sendResponsesEvent(allocator, stream, seq, "response.output_text.delta", delta);
}

/// Emit output_text.done + content_part.done + output_item.done.
pub fn emitResponsesMessageEnd(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    full_text: []const u8,
) !void {
    const esc_id = try jsonEscape(allocator, item_id);
    defer allocator.free(esc_id);
    const esc_text = try jsonEscape(allocator, full_text);
    defer allocator.free(esc_text);
    {
        const done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_text.done","item_id":{s},"output_index":{d},"content_index":0,"text":{s}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_text.done", done);
    }
    {
        const part_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.content_part.done","item_id":{s},"output_index":{d},"content_index":0,"part":{{"type":"output_text","text":{s},"annotations":[]}}}}
        , .{ esc_id, output_index, esc_text });
        defer allocator.free(part_done);
        try sendResponsesEvent(allocator, stream, seq, "response.content_part.done", part_done);
    }
    {
        const item_done = try std.fmt.allocPrint(allocator,
            \\{{"type":"response.output_item.done","output_index":{d},"item":{{"type":"message","id":{s},"role":"assistant","status":"completed","content":[{{"type":"output_text","text":{s},"annotations":[]}}]}}}}
        , .{ output_index, esc_id, esc_text });
        defer allocator.free(item_done);
        try sendResponsesEvent(allocator, stream, seq, "response.output_item.done", item_done);
    }
}

/// Emit a full message output item in one shot. Used when the entire visible
/// text is known up-front (non-streaming generation paths).
pub fn emitResponsesMessageEvents(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    output_index: u32,
    item_id: []const u8,
    text: []const u8,
) !void {
    try emitResponsesMessageStart(allocator, stream, seq, output_index, item_id);
    try emitResponsesMessageDelta(allocator, stream, seq, output_index, item_id, text);
    try emitResponsesMessageEnd(allocator, stream, seq, output_index, item_id, text);
}

/// Persist a finished response to the in-memory store. The stored history is
/// the input messages plus a non-loop-cut assistant turn, deep-copied into
/// the entry's arena so it stays valid across requests.
pub fn storeResponse(
    io: std.Io,
    gpa: std.mem.Allocator,
    resp_id: []const u8,
    model_name: []const u8,
    status_str: []const u8,
    body_json: []const u8,
    input_messages: []const chat_mod.Message,
    visible_text: []const u8,
    reasoning_text: ?[]const u8,
    tool_calls: ?[]const chat_mod.ToolCall,
    finish_details: ?[]const u8,
) !void {
    const sr = try gpa.create(responses_mod.StoredResponse);
    errdefer gpa.destroy(sr);
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    // Keep the response retrievable, but never replay a loop-cut assistant turn.
    const keep_assistant = shouldParseToolCalls(finish_details);

    // Deep-copy input messages.
    const total_msgs = input_messages.len + @intFromBool(keep_assistant);
    const history = try a.alloc(chat_mod.Message, total_msgs);
    for (input_messages, 0..) |m, i| {
        history[i] = .{
            .role = try a.dupe(u8, m.role),
            .content = try a.dupe(u8, m.content),
            .tool_call_id = if (m.tool_call_id) |tid| try a.dupe(u8, tid) else null,
            .tool_calls = if (m.tool_calls) |tcs| blk: {
                const copied = try a.alloc(chat_mod.ToolCall, tcs.len);
                for (tcs, 0..) |tc, j| copied[j] = .{
                    .id = try a.dupe(u8, tc.id),
                    .name = try a.dupe(u8, tc.name),
                    .arguments = try a.dupe(u8, tc.arguments),
                };
                break :blk copied;
            } else null,
            // images: not preserved across requests (would require deep-copying pixel buffers).
            .images = null,
        };
    }
    if (keep_assistant) {
        // Build the assistant message that produced this response.
        var assistant_text_parts = std.ArrayList(u8).empty;
        defer assistant_text_parts.deinit(a);
        if (reasoning_text) |rt| {
            try assistant_text_parts.appendSlice(a, "<think>");
            try assistant_text_parts.appendSlice(a, rt);
            try assistant_text_parts.appendSlice(a, "</think>");
        }
        try assistant_text_parts.appendSlice(a, visible_text);
        const assistant_content = try a.dupe(u8, assistant_text_parts.items);

        var assistant_tool_calls: ?[]chat_mod.ToolCall = null;
        if (tool_calls) |tcs| if (tcs.len > 0) {
            const arr = try a.alloc(chat_mod.ToolCall, tcs.len);
            for (tcs, 0..) |tc, i| {
                arr[i] = .{
                    .id = try a.dupe(u8, tc.id),
                    .name = try a.dupe(u8, tc.name),
                    .arguments = try a.dupe(u8, tc.arguments),
                };
            }
            assistant_tool_calls = arr;
        };

        history[total_msgs - 1] = .{
            .role = try a.dupe(u8, "assistant"),
            .content = assistant_content,
            .tool_calls = assistant_tool_calls,
        };
    }

    sr.* = .{
        .id = try a.dupe(u8, resp_id),
        .created_at = nowSecs(io),
        .model = try a.dupe(u8, model_name),
        .status = try a.dupe(u8, status_str),
        .body_json = try a.dupe(u8, body_json),
        .history = history,
        .arena = arena,
    };
    errdefer sr.release();

    const store = getOrInitResponseStore(io, gpa);
    try store.put(sr);
}

// ── Tests ──
