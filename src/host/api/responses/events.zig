//! sendResponsesEvent

const std = @import("std");

const Conn = @import("../../http/conn.zig").Conn;
const sendAnthropicEvent = @import("../anthropic.zig").sendAnthropicEvent;

/// Wrap a Responses-API streaming event payload with `sequence_number` (which
/// the OpenAI Responses streaming schema requires on every event), then send.
/// The `payload` is expected to be a JSON object string ending in `}`.
pub fn sendResponsesEvent(
    allocator: std.mem.Allocator,
    stream: *Conn,
    seq: *u64,
    event_name: []const u8,
    payload: []const u8,
) !void {
    if (payload.len < 2 or payload[0] != '{' or payload[payload.len - 1] != '}') {
        // Malformed payload — fall back to raw send (defensive; should not happen).
        try sendAnthropicEvent(stream, event_name, payload);
        return;
    }
    var num_buf: [32]u8 = undefined;
    const num_str = try std.fmt.bufPrint(&num_buf, "{d}", .{seq.*});
    seq.* += 1;
    // Detect whether the object has any existing fields (decides leading comma).
    var has_fields = false;
    for (payload[1 .. payload.len - 1]) |c| {
        if (!std.ascii.isWhitespace(c)) {
            has_fields = true;
            break;
        }
    }
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);
    try buf.appendSlice(allocator, payload[0 .. payload.len - 1]);
    if (has_fields) try buf.append(allocator, ',');
    try buf.appendSlice(allocator, "\"sequence_number\":");
    try buf.appendSlice(allocator, num_str);
    try buf.append(allocator, '}');
    try sendAnthropicEvent(stream, event_name, buf.items);
}
