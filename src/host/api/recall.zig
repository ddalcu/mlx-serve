//! `GET`/`DELETE /v1/recall` — the persistent token-id record's status and
//! emptying (sdk/recall, ported from TensorFold's recall endpoints).
//!
//! A STATUS read like `/props`: dispatched before any model resolution, so a
//! poll never cold-loads a model, and it answers with no model at all (an
//! empty `records` array) when Recall is off or nothing has opened a record.

const std = @import("std");
const format = @import("format.zig");
const Conn = @import("../http/conn.zig").Conn;
const respond = @import("../http/respond.zig");
const errors = @import("../http/errors.zig");
const auth = @import("../http/auth.zig");
const mlx_engine = @import("../engines/mlx.zig");

/// The record's trust lane: loopback peers, or --recall-shared. Keyed and
/// LAN peers must not even see the record's shape (ids held, landing rates)
/// — the counts themselves say what was repeated.
fn laneOk(stream: *const Conn) bool {
    return mlx_engine.recallLane(auth.peerIsLoopback(stream));
}

pub fn handleGet(allocator: std.mem.Allocator, stream: *Conn) !void {
    if (!laneOk(stream)) {
        try errors.sendErrorResponse(allocator, stream, "403 Forbidden", "forbidden",
            "the record is loopback-only while an API key is active; start the server with --recall-shared to share it", 403);
        return;
    }
    const rows = mlx_engine.recallRows(allocator);
    defer allocator.free(rows);

    var body = std.ArrayList(u8).empty;
    defer body.deinit(allocator);
    const w = &body;
    try w.appendSlice(allocator, "{\"recall\":{\"enabled\":");
    try w.appendSlice(allocator, if (mlx_engine.recallLane(auth.peerIsLoopback(stream))) "true" else "false");
    try w.appendSlice(allocator, ",\"records\":[");
    for (rows, 0..) |s, i| {
        if (i > 0) try w.append(allocator, ',');
        const model = try format.jsonEscape(allocator, s.model);
        defer allocator.free(model);
        try w.print(allocator, "{{\"model\":{s},\"writes\":{},\"read_only\":{},\"ids\":{d},\"disk_bytes\":{d},\"ram_bytes\":{d},\"cap_bytes\":{d},\"added\":{d},\"held\":{d},\"rates\":[{d:.3},{d:.3},{d:.3}],\"probe_every\":{d},\"proposals\":{d},\"proposed\":{d},\"accepted\":{d}}}", .{
            model,                     s.writes,   s.read_only, s.ids,
            s.disk_bytes,             s.ram_bytes, s.cap_bytes, s.added,
            s.held,                    s.rates[0], s.rates[1],   s.rates[2],
            s.probe_every,             s.proposals, s.proposed,  s.accepted,
        });
    }
    try w.appendSlice(allocator, "]}}");

    try respond.sendResponse(stream, "200 OK", "application/json", body.items);
}

pub fn handleDelete(allocator: std.mem.Allocator, stream: *Conn) !void {
    if (!laneOk(stream)) {
        try errors.sendErrorResponse(allocator, stream, "403 Forbidden", "forbidden",
            "the record is loopback-only while an API key is active; start the server with --recall-shared to share it", 403);
        return;
    }
    const out = mlx_engine.recallEmpty(allocator);
    defer {
        for (out.rows) |row| {
            allocator.free(row.model);
            allocator.free(row.dir);
        }
        allocator.free(out.rows);
    }
    if (out.rows.len == 0) {
        // No record ever opened (recall off, or no request has run): nothing
        // to empty is not an error, same shape as the GET.
        try respond.sendResponse(stream, "200 OK", "application/json", "{\"deleted\":true,\"records\":[]}");
        return;
    }

    var body = std.ArrayList(u8).empty;
    defer body.deinit(allocator);
    const w = &body;
    try w.appendSlice(allocator, "{\"deleted\":");
    try w.appendSlice(allocator, if (out.all_failed) "false" else "true");
    try w.appendSlice(allocator, ",\"records\":[");
    for (out.rows, 0..) |row, i| {
        if (i > 0) try w.append(allocator, ',');
        const model = try format.jsonEscape(allocator, row.model);
        defer allocator.free(model);
        const problem = try format.jsonEscape(allocator, row.problem);
        defer allocator.free(problem);
        try w.print(allocator, "{{\"model\":{s},\"deleted\":{},\"problem\":{s}}}", .{ model, row.ok, problem });
    }
    try w.appendSlice(allocator, "]}");

    if (out.all_failed) {
        try errors.sendErrorResponse(allocator, stream, "409 Conflict", "conflict", "no record could be emptied (read-only, or a chunk file could not be removed)", 409);
        return;
    }
    try respond.sendResponse(stream, "200 OK", "application/json", body.items);
}
