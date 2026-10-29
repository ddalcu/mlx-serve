//! `multipart/form-data` parsing (RFC 7578) over an already-buffered request
//! body. Pure data — no I/O, no allocation: every `Part` field is a slice into
//! the caller's body, so a 40 MB image upload costs nothing extra here.
//!
//! Exists because the OpenAI image-EDIT surface (`POST /v1/images/edits`) is
//! multipart, unlike every other endpoint we serve (all JSON). Generic on
//! purpose: `/v1/audio/transcriptions` is the same shape when it lands.

const std = @import("std");

pub const Part = struct {
    /// The form field name (`Content-Disposition: form-data; name="…"`).
    name: []const u8,
    /// Present when the part was uploaded as a file.
    filename: ?[]const u8,
    content_type: ?[]const u8,
    /// Raw bytes, verbatim — binary-safe (the trailing CRLF before the next
    /// boundary belongs to the framing, not the value, and is stripped).
    data: []const u8,
};

/// RFC 2046 caps a boundary at 70 chars; +2 for the leading `--`.
const MAX_DELIM = 72;

/// Extract the `boundary=` parameter from a Content-Type header VALUE (quoted
/// or bare). Returns null when this isn't a multipart body.
pub fn boundaryFromContentType(ct: []const u8) ?[]const u8 {
    if (indexOfIgnoreCase(ct, "multipart/form-data") == null) return null;
    const at = indexOfIgnoreCase(ct, "boundary=") orelse return null;
    var v = ct[at + "boundary=".len ..];
    while (v.len > 0 and (v[0] == ' ' or v[0] == '\t')) v = v[1..];
    if (v.len > 0 and v[0] == '"') {
        v = v[1..];
        const end = std.mem.indexOfScalar(u8, v, '"') orelse return null;
        return if (end == 0) null else v[0..end];
    }
    var end: usize = 0;
    while (end < v.len and v[end] != ';' and v[end] != ' ' and v[end] != '\r' and v[end] != '\n') end += 1;
    if (end == 0 or end > MAX_DELIM - 2) return null;
    return v[0..end];
}

/// Iterate the parts of a multipart body. Use through a pointer:
/// `var it = try Iterator.init(body, boundary); while (it.next()) |p| {…}`.
pub const Iterator = struct {
    body: []const u8,
    delim_buf: [MAX_DELIM]u8,
    delim_len: usize,
    pos: usize,
    done: bool,

    pub fn init(body: []const u8, boundary: []const u8) !Iterator {
        if (boundary.len == 0 or boundary.len > MAX_DELIM - 2) return error.InvalidBoundary;
        var self = Iterator{ .body = body, .delim_buf = undefined, .delim_len = boundary.len + 2, .pos = 0, .done = false };
        self.delim_buf[0] = '-';
        self.delim_buf[1] = '-';
        @memcpy(self.delim_buf[2..][0..boundary.len], boundary);
        return self;
    }

    fn delim(self: *const Iterator) []const u8 {
        return self.delim_buf[0..self.delim_len];
    }

    pub fn next(self: *Iterator) ?Part {
        if (self.done) return null;
        const d = self.delim();
        const start = std.mem.indexOfPos(u8, self.body, self.pos, d) orelse {
            self.done = true;
            return null;
        };
        var p = start + d.len;
        // `--boundary--` closes the body.
        if (p + 2 <= self.body.len and self.body[p] == '-' and self.body[p + 1] == '-') {
            self.done = true;
            return null;
        }
        // Skip the CRLF (or bare LF — some clients) after the delimiter.
        if (p + 2 <= self.body.len and self.body[p] == '\r' and self.body[p + 1] == '\n') {
            p += 2;
        } else if (p < self.body.len and self.body[p] == '\n') {
            p += 1;
        } else {
            self.done = true;
            return null;
        }
        const hdr_end = std.mem.indexOfPos(u8, self.body, p, "\r\n\r\n") orelse {
            self.done = true;
            return null;
        };
        const headers = self.body[p..hdr_end];
        const data_start = hdr_end + 4;
        // The value runs to the CRLF that introduces the next delimiter. A body
        // that contains the delimiter bytes mid-value can't happen — that's what
        // makes the boundary a boundary — but an unterminated body can, so fall
        // back to the end of the buffer rather than dropping the part.
        const next_delim = std.mem.indexOfPos(u8, self.body, data_start, d);
        var data_end = next_delim orelse self.body.len;
        if (next_delim != null and data_end >= data_start + 2 and
            self.body[data_end - 2] == '\r' and self.body[data_end - 1] == '\n') data_end -= 2;
        self.pos = next_delim orelse self.body.len;
        if (next_delim == null) self.done = true;

        var part = Part{ .name = "", .filename = null, .content_type = null, .data = self.body[data_start..data_end] };
        var lines = std.mem.splitSequence(u8, headers, "\r\n");
        while (lines.next()) |line| {
            if (indexOfIgnoreCase(line, "content-disposition:") == 0) {
                part.name = paramValue(line, "name=") orelse "";
                part.filename = paramValue(line, "filename=");
            } else if (indexOfIgnoreCase(line, "content-type:")) |ci| {
                if (ci == 0) part.content_type = std.mem.trim(u8, line["content-type:".len..], " \t");
            }
        }
        return part;
    }
};

/// `name="value"` (quoted) or `name=value` out of a header line.
///
/// The match must sit at a PARAMETER boundary (start of line, or after a `;`).
/// A plain substring search finds `name=` inside `filename=`, so a client that
/// orders the two the other way round — RFC 7578 fixes no order — had its field
/// name come back as the filename, and the part stopped being recognized.
fn paramValue(line: []const u8, key: []const u8) ?[]const u8 {
    var from: usize = 0;
    const at = blk: while (from < line.len) {
        const rel = indexOfIgnoreCase(line[from..], key) orelse return null;
        const abs = from + rel;
        var j = abs;
        while (j > 0 and (line[j - 1] == ' ' or line[j - 1] == '\t')) j -= 1;
        if (j == 0 or line[j - 1] == ';') break :blk abs;
        from = abs + 1;
    } else return null;
    var v = line[at + key.len ..];
    if (v.len > 0 and v[0] == '"') {
        v = v[1..];
        const end = std.mem.indexOfScalar(u8, v, '"') orelse return null;
        return v[0..end];
    }
    var end: usize = 0;
    while (end < v.len and v[end] != ';' and v[end] != ' ') end += 1;
    return if (end == 0) null else v[0..end];
}

/// Case-insensitive substring search (std.ascii has no indexOfIgnoreCase here).
fn indexOfIgnoreCase(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (haystack.len < needle.len) return null;
    var i: usize = 0;
    outer: while (i + needle.len <= haystack.len) : (i += 1) {
        for (needle, 0..) |c, j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(c)) continue :outer;
        }
        return i;
    }
    return null;
}

// ── Tests ──

const testing = std.testing;

fn buildForm(allocator: std.mem.Allocator, boundary: []const u8, parts: []const struct { hdr: []const u8, data: []const u8 }) ![]u8 {
    var b: std.ArrayList(u8) = .empty;
    errdefer b.deinit(allocator);
    for (parts) |p| {
        try b.appendSlice(allocator, "--");
        try b.appendSlice(allocator, boundary);
        try b.appendSlice(allocator, "\r\n");
        try b.appendSlice(allocator, p.hdr);
        try b.appendSlice(allocator, "\r\n\r\n");
        try b.appendSlice(allocator, p.data);
        try b.appendSlice(allocator, "\r\n");
    }
    try b.appendSlice(allocator, "--");
    try b.appendSlice(allocator, boundary);
    try b.appendSlice(allocator, "--\r\n");
    return b.toOwnedSlice(allocator);
}

test "boundaryFromContentType: bare, quoted, and non-multipart" {
    try testing.expectEqualStrings("abc123", boundaryFromContentType("multipart/form-data; boundary=abc123").?);
    try testing.expectEqualStrings("abc123", boundaryFromContentType("multipart/form-data; boundary=\"abc123\"").?);
    // Header casing and parameter order are the client's choice.
    try testing.expectEqualStrings("X-Y", boundaryFromContentType("Multipart/Form-Data; charset=utf-8; BOUNDARY=X-Y").?);
    try testing.expectEqualStrings("z", boundaryFromContentType("multipart/form-data; boundary=z; other=1").?);
    try testing.expectEqual(@as(?[]const u8, null), boundaryFromContentType("application/json"));
    try testing.expectEqual(@as(?[]const u8, null), boundaryFromContentType("multipart/form-data"));
}

test "Iterator: fields, files, and binary data with embedded CRLF" {
    const a = testing.allocator;
    // A PNG-ish payload containing CRLF and a near-miss of the delimiter — the
    // exact bytes a naive line-splitting parser truncates on.
    const binary = "\x89PNG\r\n\x1a\n--BOUND-not-really\r\nstill data\x00\xff";
    const form = try buildForm(a, "BOUNDARY", &.{
        .{ .hdr = "Content-Disposition: form-data; name=\"prompt\"", .data = "make it winter" },
        .{ .hdr = "Content-Disposition: form-data; name=\"image\"; filename=\"dog.png\"\r\nContent-Type: image/png", .data = binary },
        .{ .hdr = "Content-Disposition: form-data; name=\"n\"", .data = "1" },
    });
    defer a.free(form);

    var it = try Iterator.init(form, "BOUNDARY");
    const p1 = it.next().?;
    try testing.expectEqualStrings("prompt", p1.name);
    try testing.expectEqualStrings("make it winter", p1.data);
    try testing.expectEqual(@as(?[]const u8, null), p1.filename);

    const p2 = it.next().?;
    try testing.expectEqualStrings("image", p2.name);
    try testing.expectEqualStrings("dog.png", p2.filename.?);
    try testing.expectEqualStrings("image/png", p2.content_type.?);
    try testing.expectEqualStrings(binary, p2.data);

    const p3 = it.next().?;
    try testing.expectEqualStrings("n", p3.name);
    try testing.expectEqualStrings("1", p3.data);

    try testing.expectEqual(@as(?Part, null), it.next());
    try testing.expectEqual(@as(?Part, null), it.next()); // stays done
}

test "paramValue keys on the PARAMETER, not a substring of a longer one" {
    // `name=` also occurs inside `fileNAME=`. RFC 7578 fixes no order for the
    // two, and a client that emits filename first made the field name come back
    // as the FILENAME — so the part stopped being recognized as `image` and the
    // edit request 400'd with "missing image".
    const a = testing.allocator;
    for ([_][]const u8{
        "Content-Disposition: form-data; name=\"image\"; filename=\"dog.png\"",
        "Content-Disposition: form-data; filename=\"dog.png\"; name=\"image\"",
    }) |hdr| {
        const form = try buildForm(a, "B", &.{.{ .hdr = hdr, .data = "PNGDATA" }});
        defer a.free(form);
        var it = try Iterator.init(form, "B");
        const p = it.next().?;
        try testing.expectEqualStrings("image", p.name);
        try testing.expectEqualStrings("dog.png", p.filename.?);
    }
}

test "Iterator: repeated field names are yielded in order (multi-image edit)" {
    const a = testing.allocator;
    const form = try buildForm(a, "B", &.{
        .{ .hdr = "Content-Disposition: form-data; name=\"image[]\"; filename=\"a.png\"", .data = "AAA" },
        .{ .hdr = "Content-Disposition: form-data; name=\"image[]\"; filename=\"b.png\"", .data = "BBB" },
    });
    defer a.free(form);
    var it = try Iterator.init(form, "B");
    try testing.expectEqualStrings("AAA", it.next().?.data);
    try testing.expectEqualStrings("BBB", it.next().?.data);
    try testing.expectEqual(@as(?Part, null), it.next());
}

test "Iterator: empty value, missing final boundary, and junk are survivable" {
    const a = testing.allocator;
    const form = try buildForm(a, "B", &.{
        .{ .hdr = "Content-Disposition: form-data; name=\"empty\"", .data = "" },
    });
    defer a.free(form);
    var it = try Iterator.init(form, "B");
    const p = it.next().?;
    try testing.expectEqualStrings("empty", p.name);
    try testing.expectEqualStrings("", p.data);
    try testing.expectEqual(@as(?Part, null), it.next());

    // Truncated upload (no closing delimiter): keep what arrived instead of
    // silently returning nothing — the caller's field validation decides.
    const cut = "--B\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nhalf";
    var it2 = try Iterator.init(cut, "B");
    try testing.expectEqualStrings("half", it2.next().?.data);
    try testing.expectEqual(@as(?Part, null), it2.next());

    // Body that isn't multipart at all → no parts, no crash.
    var it3 = try Iterator.init("{\"prompt\":\"hi\"}", "B");
    try testing.expectEqual(@as(?Part, null), it3.next());

    try testing.expectError(error.InvalidBoundary, Iterator.init("x", ""));
}
