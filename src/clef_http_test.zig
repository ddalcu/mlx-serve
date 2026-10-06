const std = @import("std");
const V = std.json.Value;
const a = std.testing.allocator;
var max_reference_error: f64 = 0;

pub fn readFile(io: std.Io, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var buffer: [4096]u8 = undefined;
    var reader = f.reader(io, &buffer);
    return reader.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
}

fn request(io: std.Io, base: []const u8, path: []const u8, body: ?[]const u8, status: u16) !std.json.Parsed(V) {
    const url = try std.fmt.allocPrint(a, "{s}{s}", .{ base, path });
    defer a.free(url);
    var argv: std.ArrayList([]const u8) = .empty;
    defer argv.deinit(a);
    try argv.appendSlice(a, &.{ "/usr/bin/curl", "--silent", "--show-error", "--max-time", "180", "--write-out", "\n%{http_code}", url });
    if (body) |b| try argv.appendSlice(a, &.{ "--header", "Content-Type: application/json", "--data-raw", b });
    const result = try std.process.run(a, io, .{ .argv = argv.items, .stdout_limit = .limited(4 * 1024 * 1024), .stderr_limit = .limited(65536) });
    defer a.free(result.stdout);
    defer a.free(result.stderr);
    try std.testing.expectEqual(std.process.Child.Term{ .exited = 0 }, result.term);
    const split = std.mem.lastIndexOfScalar(u8, result.stdout, '\n') orelse return error.MissingHttpStatus;
    const actual = try std.fmt.parseInt(u16, result.stdout[split + 1 ..], 10);
    if (actual != status) std.debug.print("{s}: expected {d}, got {d}: {s}\n", .{ path, status, actual, result.stdout[0..split] });
    try std.testing.expectEqual(status, actual);
    return std.json.parseFromSlice(V, a, result.stdout[0..split], .{ .allocate = .alloc_always });
}

pub fn compare(expected: V, actual: V) !void {
    // Shared bf16 prefill reductions amplify in the head; head-only parity uses 0.001.
    return compareWithin(expected, actual, 0.075);
}

pub fn oraclePath() []const u8 {
    return if (std.c.getenv("CLEF_TEST_EXPECTED")) |path| std.mem.span(path) else "tests/fixtures/clef/expected.json";
}

fn compareWithin(expected: V, actual: V, tolerance: f64) !void {
    switch (expected) {
        .float => |n| {
            const value: f64 = switch (actual) {
                .float => |x| x,
                .integer => |x| @floatFromInt(x),
                else => return error.ExpectedNumber,
            };
            try std.testing.expectApproxEqAbs(n, value, tolerance);
            if (tolerance > 0) max_reference_error = @max(max_reference_error, @abs(n - value));
        },
        .integer => |n| try std.testing.expectEqual(n, actual.integer),
        .string => |s| try std.testing.expectEqualStrings(s, actual.string),
        .object => |obj| {
            if (actual != .object) return error.ExpectedObject;
            var it = obj.iterator();
            while (it.next()) |entry| {
                const value = actual.object.get(entry.key_ptr.*) orelse return error.MissingField;
                compareWithin(entry.value_ptr.*, value, tolerance) catch |err| {
                    std.debug.print("clef reference mismatch in {s}\n", .{entry.key_ptr.*});
                    return err;
                };
            }
        },
        else => return error.UnexpectedFixtureValue,
    }
}

test "clef HTTP: discovery, both APIs, images, invalid inputs, isolation and unload (CLEF_HTTP_URL)" {
    const base = std.mem.span(std.c.getenv("CLEF_HTTP_URL") orelse return error.SkipZigTest);
    var threaded = std.Io.Threaded.init(a, .{});
    defer threaded.deinit();
    const io = threaded.io();
    const fixture_text = try readFile(io, "tests/fixtures/clef/request.json");
    defer a.free(fixture_text);
    const fixture_image = try readFile(io, "tests/fixtures/clef/image-request.json");
    defer a.free(fixture_image);
    const fixture_negative = try readFile(io, "tests/fixtures/clef/negative-request.json");
    defer a.free(fixture_negative);
    const oracle_bytes = try readFile(io, oraclePath());
    defer a.free(oracle_bytes);
    const oracle = try std.json.parseFromSlice(V, a, oracle_bytes, .{});
    defer oracle.deinit();
    const image = try readFile(io, "tests/fixtures/robot.png");
    defer a.free(image);
    const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(image.len));
    defer a.free(encoded);
    _ = std.base64.standard.Encoder.encode(encoded, image);

    const listing = try request(io, base, "/v1/models", null, 200);
    defer listing.deinit();
    for ([_][]const u8{ "clef-flash-4bit", "clef-flash-8bit", "clef-4bit", "clef-8bit" }) |pack| {
        max_reference_error = 0;
        const id = try std.fmt.allocPrint(a, "mlx-community/{s}", .{pack});
        defer a.free(id);
        var found = false;
        for (listing.value.object.get("data").?.array.items) |entry| {
            if (!std.mem.eql(u8, entry.object.get("id").?.string, id)) continue;
            found = true;
            var decisions = false;
            for (entry.object.get("capabilities").?.array.items) |cap| {
                if (std.mem.eql(u8, cap.string, "decisions")) decisions = true;
                try std.testing.expect(!std.mem.eql(u8, cap.string, "chat"));
            }
            try std.testing.expect(decisions);
        }
        try std.testing.expect(found);
        const model_body = try std.fmt.allocPrint(a, "{{\"model\":\"{s}\"}}", .{id});
        defer a.free(model_body);
        const loaded = try request(io, base, "/v1/load-model", model_body, 200);
        loaded.deinit();
        var first_text: ?std.json.Parsed(V) = null;
        defer if (first_text) |p| p.deinit();
        for ([_][]const u8{ fixture_text, fixture_image, fixture_negative }, [_][]const u8{ "text", "image", "negative" }, 0..) |fixture, kind, fi| {
            var p = try std.json.parseFromSlice(V, a, fixture, .{});
            defer p.deinit();
            try p.value.object.put(p.arena.allocator(), "model", .{ .string = id });
            if (fi == 1) {
                var items: std.array_list.Managed(V) = .init(p.arena.allocator());
                try items.append(.{ .string = encoded });
                try p.value.object.put(p.arena.allocator(), "images", .{ .array = items });
            }
            const body = try std.json.Stringify.valueAlloc(a, p.value, .{});
            defer a.free(body);
            const expected = oracle.value.object.get(pack).?.object.get(kind).?;
            for ([_][]const u8{ "/v1/decisions", "/v1/systemone" }) |path| {
                const response = try request(io, base, path, body, 200);
                var retain = false;
                defer if (!retain) response.deinit();
                try std.testing.expectEqualStrings(id, response.value.object.get("model").?.string);
                try compare(expected.object.get("answers").?, response.value.object.get("answers").?);
                try compare(expected.object.get("usage").?, response.value.object.get("usage").?);
                if (fi == 0 and first_text == null) {
                    first_text = response;
                    retain = true;
                }
            }
        }
        for ([_][]const u8{
            "\"questions\":{}",
            "\"questions\":{\"q\":{\"type\":\"choice\",\"criteria\":[]}}",
            "\"questions\":{\"q\":{\"type\":\"noul\"}},\"truncate\":\"false\"",
            "\"questions\":{\"q\":{\"type\":\"noul\"}},\"images\":[\"invalid\"]",
            "\"questions\":{\"q\":{\"type\":\"noul\"}},\"videos\":[\"invalid\"]",
            "\"questions\":{\"q\":{\"type\":\"noul\"}},\"media_kwargs\":{\"unsupported\":true}",
        }) |fields| {
            const body = try std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"state\":\"hello\",{s}}}", .{ id, fields });
            defer a.free(body);
            const rejected = try request(io, base, "/v1/systemone", body, 400);
            defer rejected.deinit();
            try std.testing.expect(rejected.value.object.get("error") != null);
        }
        const chat_body = try std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"messages\":[{{\"role\":\"user\",\"content\":\"hello\"}}]}}", .{id});
        defer a.free(chat_body);
        const rejected_chat = try request(io, base, "/v1/chat/completions", chat_body, 400);
        rejected_chat.deinit();
        const instruction = try a.alloc(u8, 40000);
        defer a.free(instruction);
        for (0..20000) |i| @memcpy(instruction[2 * i ..][0..2], "x ");
        const oversized_body = try std.fmt.allocPrint(a, "{{\"model\":\"{s}\",\"state\":\"\",\"questions\":{{\"q\":{{\"type\":\"noul\",\"instructions\":\"{s}\"}}}}}}", .{ id, instruction });
        defer a.free(oversized_body);
        const oversized = try request(io, base, "/v1/decisions", oversized_body, 400);
        oversized.deinit();
        // Image requests and invalid requests must not leave state in the next text request.
        var again = try std.json.parseFromSlice(V, a, fixture_text, .{});
        defer again.deinit();
        try again.value.object.put(again.arena.allocator(), "model", .{ .string = id });
        const again_body = try std.json.Stringify.valueAlloc(a, again.value, .{});
        defer a.free(again_body);
        const repeated = try request(io, base, "/v1/decisions", again_body, 200);
        defer repeated.deinit();
        try compare(oracle.value.object.get(pack).?.object.get("text").?.object.get("answers").?, repeated.value.object.get("answers").?);
        try compareWithin(first_text.?.value.object.get("answers").?, repeated.value.object.get("answers").?, 0);
        const unloaded = try request(io, base, "/v1/unload-model", model_body, 200);
        unloaded.deinit();
        const reloaded = try request(io, base, "/v1/load-model", model_body, 200);
        reloaded.deinit();
        const after_reload = try request(io, base, "/v1/decisions", again_body, 200);
        defer after_reload.deinit();
        try compareWithin(repeated.value.object.get("answers").?, after_reload.value.object.get("answers").?, 0);
        const final_unload = try request(io, base, "/v1/unload-model", model_body, 200);
        final_unload.deinit();
        std.debug.print("clef HTTP passed: {s} (max reference difference {d:.4})\n", .{ pack, max_reference_error });
    }
}
