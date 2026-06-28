//! Standalone HTTP server for native FLUX.2 text-to-image (mirrors tts_server.zig).
//!
//! FLUX is a serial, request→file engine (no token streaming) like the ds4/llama
//! embedded engines. Load the three sub-models + tokenizer once on this thread,
//! then answer `POST /v1/images/generations` with a base64 PNG.
//!
//! `main.zig` routes here when `config.json` indicates an mflux FLUX.2 model.

const std = @import("std");
const mlx = @import("mlx.zig");
const flux = @import("flux.zig");
const png = @import("png.zig");
const tok_mod = @import("tokenizer.zig");
const log = @import("log.zig");
const server_mod = @import("server.zig");

const Conn = server_mod.Conn;

const PAD_TOKEN: i32 = 151643; // Qwen2/3 pad token
const SEQ_LEN: usize = 512; // mflux Qwen3 tokenizer max_length

pub fn runFluxServe(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    host: []const u8,
    port: u16,
) !void {
    log.info("mlx-serve (native FLUX.2 image engine)\n", .{});
    log.info("[args] model: {s}\n", .{model_dir});

    const s = mlx.mlx_default_gpu_stream_new();
    var te = try flux.loadTextEncoder(io, allocator, s, model_dir);
    defer te.deinit();
    var dit = try flux.loadDit(io, allocator, s, model_dir);
    defer dit.deinit();
    var vae = try flux.loadVae(io, allocator, s, model_dir);
    defer vae.deinit();

    // Tokenizer lives in the `tokenizer/` subdir for FLUX.2.
    const tok_dir = try std.fmt.allocPrint(allocator, "{s}/tokenizer", .{model_dir});
    defer allocator.free(tok_dir);
    var tokenizer = try tok_mod.loadTokenizerAny(io, allocator, tok_dir);
    defer tokenizer.deinit();
    log.info("[flux] models + tokenizer ready\n", .{});

    const model_id = basename(model_dir);

    var ip4_bytes: [4]u8 = .{ 0, 0, 0, 0 };
    {
        var parts = std.mem.splitScalar(u8, host, '.');
        var idx: usize = 0;
        while (parts.next()) |part| {
            if (idx >= 4) break;
            ip4_bytes[idx] = std.fmt.parseInt(u8, part, 10) catch 0;
            idx += 1;
        }
    }
    const ip_addr: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = ip4_bytes, .port = port } };
    var server = try ip_addr.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    log.info("\nFLUX image server listening on http://{s}:{d}\n", .{ host, port });
    log.info("  POST /v1/images/generations  {{\"prompt\":\"...\",\"size\":\"1024x1024\"}}\n", .{});

    var ctx = ServeCtx{ .te = &te, .dit = &dit, .vae = &vae, .tok = &tokenizer, .s = s };

    while (true) {
        const accepted = server.accept(io) catch |err| {
            log.err("accept error: {}\n", .{err});
            continue;
        };
        var conn: Conn = undefined;
        conn.init(accepted, io);
        handleOne(io, allocator, &ctx, &conn, model_id) catch |err| {
            log.warn("[flux] request error: {}\n", .{err});
        };
        conn.flush() catch {};
        conn.close();
    }
}

const ServeCtx = struct {
    te: *flux.TextEncoder,
    dit: *flux.Dit,
    vae: *flux.Vae,
    tok: *tok_mod.Tokenizer,
    s: mlx.mlx_stream,
};

fn handleOne(io: std.Io, allocator: std.mem.Allocator, ctx: *ServeCtx, conn: *Conn, model_id: []const u8) !void {
    var hdr_buf: [16 * 1024]u8 = undefined;
    var total_read: usize = 0;
    var content_length: ?usize = null;
    var header_end_pos: usize = 0;
    while (total_read < hdr_buf.len) {
        const n = try conn.read(hdr_buf[total_read..]);
        if (n == 0) break;
        total_read += n;
        if (std.mem.indexOf(u8, hdr_buf[0..total_read], "\r\n\r\n")) |he| {
            header_end_pos = he + 4;
            content_length = findContentLength(hdr_buf[0..he]);
            break;
        }
    }
    if (header_end_pos == 0) return;

    const cl = content_length orelse 0;
    const total_size = header_end_pos + cl;
    if (total_size > 4 * 1024 * 1024) return sendError(conn, 413, "request too large");
    const buf = try allocator.alloc(u8, total_size);
    defer allocator.free(buf);
    @memcpy(buf[0..total_read], hdr_buf[0..total_read]);
    while (total_read < total_size) {
        const n = try conn.read(buf[total_read..total_size]);
        if (n == 0) break;
        total_read += n;
    }
    const request = buf[0..total_read];
    const first_line_end = std.mem.indexOf(u8, request, "\r\n") orelse return;
    const first_line = request[0..first_line_end];
    const body = if (header_end_pos <= total_read) request[header_end_pos..total_read] else "";

    var it = std.mem.tokenizeScalar(u8, first_line, ' ');
    const method = it.next() orelse return;
    const path = it.next() orelse return;

    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/health")) {
        return sendJson(conn, "{\"status\":\"ok\"}");
    }
    if (std.mem.eql(u8, method, "GET") and std.mem.eql(u8, path, "/v1/models")) {
        var b: std.ArrayList(u8) = .empty;
        defer b.deinit(allocator);
        try b.appendSlice(allocator, "{\"object\":\"list\",\"data\":[{\"id\":\"");
        try b.appendSlice(allocator, model_id);
        try b.appendSlice(allocator, "\",\"object\":\"model\",\"capabilities\":[\"image\"]}]}");
        return sendJson(conn, b.items);
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/images/generations")) {
        const prompt_raw = extractJsonString(body, "prompt") orelse {
            return sendError(conn, 400, "missing 'prompt'");
        };
        const prompt = try jsonUnescape(allocator, prompt_raw);
        defer allocator.free(prompt);
        if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");

        // The engine is currently validated/supported at 1024x1024 only (the DiT
        // latent grid is fixed); other sizes crash the VAE reshape, so we force
        // 1024 until multi-resolution lands. `size` is accepted but normalized.
        const width: u32 = 1024;
        const height: u32 = 1024;
        if (extractJsonString(body, "size")) |size| {
            if (!std.mem.eql(u8, size, "1024x1024")) {
                log.warn("[flux] size '{s}' not supported yet; using 1024x1024\n", .{size});
            }
        }
        const seed: u64 = extractJsonInt(body, "seed") orelse 42;
        const steps: u32 = @intCast(extractJsonInt(body, "steps") orelse 4);

        log.info("[flux] generating {d}x{d} steps={d}: {d} chars\n", .{ width, height, steps, prompt.len });
        const png_bytes = generatePng(io, allocator, ctx, prompt, width, height, seed, steps) catch |err| {
            log.err("[flux] generation failed: {}\n", .{err});
            return sendError(conn, 500, "generation failed");
        };
        defer allocator.free(png_bytes);

        // {"data":[{"b64_json":"..."}]}
        const b64_len = std.base64.standard.Encoder.calcSize(png_bytes.len);
        const b64 = try allocator.alloc(u8, b64_len);
        defer allocator.free(b64);
        _ = std.base64.standard.Encoder.encode(b64, png_bytes);

        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(allocator);
        try out.appendSlice(allocator, "{\"created\":0,\"data\":[{\"b64_json\":\"");
        try out.appendSlice(allocator, b64);
        try out.appendSlice(allocator, "\"}]}");
        log.info("[flux] -> {d} PNG bytes ({d} b64)\n", .{ png_bytes.len, b64.len });
        return sendBytesJson(conn, allocator, out.items);
    }
    return sendError(conn, 404, "not found");
}

/// Tokenize the prompt (Qwen3 chat template) and run the FLUX pipeline → PNG bytes.
fn generatePng(io: std.Io, allocator: std.mem.Allocator, ctx: *ServeCtx, prompt: []const u8, width: u32, height: u32, seed: u64, steps: u32) ![]u8 {
    _ = io;
    // mflux Qwen3 chat template (enable_thinking=False adds an empty <think> block).
    const templated = try std.fmt.allocPrint(allocator, "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", .{prompt});
    defer allocator.free(templated);

    const enc = try ctx.tok.encode(allocator, templated); // []u32, handles special tokens
    defer allocator.free(enc);

    // Right-pad / truncate to SEQ_LEN; build attention mask.
    var ids = try allocator.alloc(i32, SEQ_LEN);
    defer allocator.free(ids);
    var mask = try allocator.alloc(i32, SEQ_LEN);
    defer allocator.free(mask);
    const real = @min(enc.len, SEQ_LEN);
    for (0..SEQ_LEN) |i| {
        if (i < real) {
            ids[i] = @intCast(enc[i]);
            mask[i] = 1;
        } else {
            ids[i] = PAD_TOKEN;
            mask[i] = 0;
        }
    }

    const img = try flux.generate(ctx.te, ctx.dit, ctx.vae, ids, mask, seed, steps, height, width);
    defer _ = mlx.mlx_array_free(img);

    // Extract RGB8 from [1,3,H,W] in [0,1] (mirror writePpm).
    const cf = blk: {
        var c = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&c, img, false, ctx.s));
        break :blk c;
    };
    defer _ = mlx.mlx_array_free(cf);
    _ = mlx.mlx_array_eval(cf);
    const sh = mlx.getShape(cf); // [1,3,H,W]
    const H: usize = @intCast(sh[2]);
    const W: usize = @intCast(sh[3]);
    const d = mlx.mlx_array_data_float32(cf) orelse return error.NoData;
    const rgb = try allocator.alloc(u8, W * H * 3);
    defer allocator.free(rgb);
    const plane = W * H;
    for (0..H) |y| for (0..W) |x| {
        const o = (y * W + x) * 3;
        for (0..3) |c| {
            const v = d[c * plane + y * W + x];
            rgb[o + c] = @intFromFloat(std.math.clamp(v * 255.0, 0, 255));
        }
    };
    return png.encodeRgb(allocator, rgb, @intCast(W), @intCast(H));
}

// ── HTTP helpers (self-contained; mirror tts_server.zig) ──

fn sendBytesJson(conn: *Conn, allocator: std.mem.Allocator, json: []const u8) !void {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(allocator);
    try hdr.appendSlice(allocator, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ");
    var num: [20]u8 = undefined;
    const ns = std.fmt.bufPrint(&num, "{d}", .{json.len}) catch unreachable;
    try hdr.appendSlice(allocator, ns);
    try hdr.appendSlice(allocator, "\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n");
    try conn.writeAllNoFlush(hdr.items);
    try conn.writeAll(json);
}

fn sendJson(conn: *Conn, json: []const u8) !void {
    var hdr: [256]u8 = undefined;
    const head = std.fmt.bufPrint(&hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", .{json.len}) catch return;
    try conn.writeAllNoFlush(head);
    try conn.writeAll(json);
}

fn sendError(conn: *Conn, code: u16, msg: []const u8) !void {
    var body_buf: [256]u8 = undefined;
    const body = std.fmt.bufPrint(&body_buf, "{{\"error\":{{\"message\":\"{s}\"}}}}", .{msg}) catch return;
    var hdr: [256]u8 = undefined;
    const head = std.fmt.bufPrint(&hdr, "HTTP/1.1 {d} Error\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ code, body.len }) catch return;
    try conn.writeAllNoFlush(head);
    try conn.writeAll(body);
}

fn findContentLength(headers: []const u8) ?usize {
    var lower_buf: [16 * 1024]u8 = undefined;
    const n = @min(headers.len, lower_buf.len);
    for (headers[0..n], 0..) |c, i| lower_buf[i] = std.ascii.toLower(c);
    const key = "content-length:";
    const idx = std.mem.indexOf(u8, lower_buf[0..n], key) orelse return null;
    var rest = headers[idx + key.len ..];
    var sp: usize = 0;
    while (sp < rest.len and (rest[sp] == ' ' or rest[sp] == '\t')) sp += 1;
    rest = rest[sp..];
    const end = std.mem.indexOfAny(u8, rest, "\r\n ") orelse rest.len;
    return std.fmt.parseInt(usize, rest[0..end], 10) catch null;
}

fn extractJsonString(body: []const u8, key: []const u8) ?[]const u8 {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return null;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return null;
    var i = ki + key_pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    if (i >= body.len or body[i] != '"') return null;
    i += 1;
    const start = i;
    while (i < body.len) : (i += 1) {
        if (body[i] == '\\') {
            i += 1;
            continue;
        }
        if (body[i] == '"') return body[start..i];
    }
    return null;
}

fn extractJsonInt(body: []const u8, key: []const u8) ?u64 {
    var key_pat_buf: [64]u8 = undefined;
    const key_pat = std.fmt.bufPrint(&key_pat_buf, "\"{s}\"", .{key}) catch return null;
    const ki = std.mem.indexOf(u8, body, key_pat) orelse return null;
    var i = ki + key_pat.len;
    while (i < body.len and (body[i] == ' ' or body[i] == ':' or body[i] == '\t')) i += 1;
    const start = i;
    while (i < body.len and (std.ascii.isDigit(body[i]))) i += 1;
    if (i == start) return null;
    return std.fmt.parseInt(u64, body[start..i], 10) catch null;
}

fn jsonUnescape(allocator: std.mem.Allocator, raw: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    var i: usize = 0;
    while (i < raw.len) : (i += 1) {
        if (raw[i] != '\\') {
            try out.append(allocator, raw[i]);
            continue;
        }
        i += 1;
        if (i >= raw.len) break;
        switch (raw[i]) {
            'n' => try out.append(allocator, '\n'),
            't' => try out.append(allocator, '\t'),
            'r' => try out.append(allocator, '\r'),
            '"' => try out.append(allocator, '"'),
            '\\' => try out.append(allocator, '\\'),
            '/' => try out.append(allocator, '/'),
            else => try out.append(allocator, raw[i]),
        }
    }
    return out.toOwnedSlice(allocator);
}

fn basename(path: []const u8) []const u8 {
    var p = path;
    while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    if (std.mem.lastIndexOfScalar(u8, p, '/')) |i| return p[i + 1 ..];
    return p;
}

test "extractJsonInt parses seed/steps" {
    try std.testing.expectEqual(@as(?u64, 7), extractJsonInt("{\"seed\": 7}", "seed"));
    try std.testing.expectEqual(@as(?u64, 20), extractJsonInt("{\"steps\":20,\"x\":1}", "steps"));
    try std.testing.expectEqual(@as(?u64, null), extractJsonInt("{\"prompt\":\"hi\"}", "seed"));
}
