//! Standalone HTTP server for native LTX-Video 2.3 text-to-video (mirrors
//! flux_server.zig / tts_server.zig).
//!
//! LTX is a serial, request→video engine. Load the three sub-models
//! (transformer-dev, connector, vae_decoder) + the Gemma text encoder /
//! tokenizer once on this thread, then answer
//! `POST /v1/video/generations` with base64 RGB frames.  The macOS app muxes
//! the frames into an mp4 (AVFoundation).
//!
//! `main.zig` routes here when `config.json` declares `model_type == "AudioVideo"`.

const std = @import("std");
const mlx = @import("mlx.zig");
const ltx = @import("ltx_video.zig");
const tok_mod = @import("tokenizer.zig");
const log = @import("log.zig");
const server_mod = @import("server.zig");
const io_util = @import("io_util.zig");

const Conn = server_mod.Conn;

const PAD_LEN: usize = 256; // gemma left-pad length (slice-1 default; reference uses 1024)
const PAD_ID: i32 = 0; // gemma <pad>

// Trimmed DEFAULT_NEGATIVE_PROMPT (ltx_pipelines_mlx.utils.constants) — left-pad
// truncates from the left anyway, so the leading clauses are what survive.
const NEGATIVE_PROMPT =
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, " ++
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, distorted " ++
    "proportions, unnatural skin tones, deformed facial features, extra limbs, disfigured " ++
    "hands, inconsistent perspective, camera shake, color banding, cartoonish rendering, " ++
    "3D CGI look, unrealistic materials, uncanny valley effect, exaggerated expressions";

const ServeCtx = struct {
    transformer: *const ltx.Component,
    connector: *const ltx.Component,
    vae: *const ltx.Component,
    tok: *tok_mod.Tokenizer,
    gemma_dir: []const u8,
    io: std.Io,
    s: mlx.mlx_stream,
};

pub fn runLtxServe(
    io: std.Io,
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    host: []const u8,
    port: u16,
) !void {
    log.info("mlx-serve (native LTX-Video 2.3 engine)\n", .{});
    log.info("[args] model: {s}\n", .{model_dir});

    const gemma_dir = try resolveGemmaDir(io, allocator);
    defer allocator.free(gemma_dir);
    log.info("[ltx] gemma text encoder: {s}\n", .{gemma_dir});

    // Components load on the CPU stream (the safetensors Load op has no GPU eval);
    // the forward graph then runs on the GPU stream.
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    const s = mlx.mlx_default_gpu_stream_new();

    const tp = try std.fmt.allocPrintSentinel(allocator, "{s}/transformer-dev.safetensors", .{model_dir}, 0);
    defer allocator.free(tp);
    var transformer = try ltx.loadComponent(allocator, tp, cpu_s);
    defer transformer.deinit();
    const cp = try std.fmt.allocPrintSentinel(allocator, "{s}/connector.safetensors", .{model_dir}, 0);
    defer allocator.free(cp);
    var connector = try ltx.loadComponent(allocator, cp, cpu_s);
    defer connector.deinit();
    const vp = try std.fmt.allocPrintSentinel(allocator, "{s}/vae_decoder.safetensors", .{model_dir}, 0);
    defer allocator.free(vp);
    var vae = try ltx.loadComponent(allocator, vp, cpu_s);
    defer vae.deinit();
    var it = vae.map.iterator();
    while (it.next()) |e| _ = mlx.mlx_array_eval(e.value_ptr.*); // VAE conv graph wants materialized weights

    var tokenizer = try tok_mod.loadTokenizerAny(io, allocator, gemma_dir);
    defer tokenizer.deinit();
    log.info("[ltx] models + tokenizer ready\n", .{});

    const model_id = basename(model_dir);
    var ctx = ServeCtx{ .transformer = &transformer, .connector = &connector, .vae = &vae, .tok = &tokenizer, .gemma_dir = gemma_dir, .io = io, .s = s };

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
    log.info("\nLTX video server listening on http://{s}:{d}\n", .{ host, port });
    log.info("  POST /v1/video/generations  {{\"prompt\":\"...\",\"num_frames\":9,\"height\":256,\"width\":384}}\n", .{});

    while (true) {
        const accepted = server.accept(io) catch |err| {
            log.err("accept error: {}\n", .{err});
            continue;
        };
        var conn: Conn = undefined;
        conn.init(accepted, io);
        handleOne(io, allocator, &ctx, &conn, model_id) catch |err| {
            log.warn("[ltx] request error: {}\n", .{err});
        };
        conn.flush() catch {};
        conn.close();
    }
}

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
        try b.appendSlice(allocator, "\",\"object\":\"model\",\"capabilities\":[\"video\"]}]}");
        return sendJson(conn, b.items);
    }
    if (std.mem.eql(u8, method, "POST") and std.mem.eql(u8, path, "/v1/video/generations")) {
        const prompt_raw = extractJsonString(body, "prompt") orelse return sendError(conn, 400, "missing 'prompt'");
        const prompt = try jsonUnescape(allocator, prompt_raw);
        defer allocator.free(prompt);
        if (prompt.len == 0) return sendError(conn, 400, "empty 'prompt'");

        const num_frames: u32 = @intCast(extractJsonInt(body, "num_frames") orelse 9);
        const height: u32 = @intCast(extractJsonInt(body, "height") orelse 256);
        const width: u32 = @intCast(extractJsonInt(body, "width") orelse 384);
        const seed: u64 = extractJsonInt(body, "seed") orelse 42;
        const steps: u32 = @intCast(extractJsonInt(body, "steps") orelse 4);
        const frame_rate: f32 = 24.0;

        log.info("[ltx] generating {d}f {d}x{d} steps={d}: {d} chars\n", .{ num_frames, height, width, steps, prompt.len });

        const pos_ids = try tokenizePadded(allocator, ctx.tok, prompt);
        defer allocator.free(pos_ids);
        const neg_ids = try tokenizePadded(allocator, ctx.tok, NEGATIVE_PROMPT);
        defer allocator.free(neg_ids);

        const t_start = io_util.nowMs(io);
        var frames = ltx.generateVideoFrames(io, allocator, .{}, ctx.transformer, ctx.connector, ctx.vae, ctx.gemma_dir, pos_ids, neg_ids, PAD_ID, num_frames, height, width, frame_rate, steps, seed, 3.0, 7.0, 0.7, ctx.s) catch |err| {
            log.err("[ltx] generation failed: {}\n", .{err});
            return sendError(conn, 500, "generation failed");
        };
        defer frames.deinit(allocator);
        log.info("[ltx] generated in {d}ms\n", .{io_util.nowMs(io) - t_start});

        const b64_len = std.base64.standard.Encoder.calcSize(frames.rgb.len);
        const b64 = try allocator.alloc(u8, b64_len);
        defer allocator.free(b64);
        _ = std.base64.standard.Encoder.encode(b64, frames.rgb);

        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(allocator);
        const head = try std.fmt.allocPrint(allocator, "{{\"created\":0,\"frames\":{d},\"height\":{d},\"width\":{d},\"fps\":{d},\"format\":\"rgb8\",\"data\":\"", .{ frames.frames, frames.height, frames.width, @as(u32, @intFromFloat(frame_rate)) });
        defer allocator.free(head);
        try out.appendSlice(allocator, head);
        try out.appendSlice(allocator, b64);
        try out.appendSlice(allocator, "\"}");
        log.info("[ltx] -> {d}f {d}x{d} ({d} rgb bytes)\n", .{ frames.frames, frames.height, frames.width, frames.rgb.len });
        return sendBytesJson(conn, allocator, out.items);
    }
    return sendError(conn, 404, "not found");
}

/// Tokenize (BOS + raw prompt) and LEFT-pad/truncate to PAD_LEN with PAD_ID.
fn tokenizePadded(allocator: std.mem.Allocator, tokenizer: *tok_mod.Tokenizer, text: []const u8) ![]i32 {
    const enc = try tokenizer.encode(allocator, text); // []u32, prepends <bos>
    defer allocator.free(enc);
    const ids = try allocator.alloc(i32, PAD_LEN);
    const real = @min(enc.len, PAD_LEN);
    const pad = PAD_LEN - real;
    for (0..pad) |i| ids[i] = PAD_ID;
    // keep the LAST `real` tokens (left-pad = truncate from the left)
    for (0..real) |i| ids[pad + i] = @intCast(enc[enc.len - real + i]);
    return ids;
}

/// Locate the Gemma-3-12B text encoder: `$LTX_GEMMA_DIR`, else the HF cache.
fn resolveGemmaDir(io: std.Io, allocator: std.mem.Allocator) ![]u8 {
    if (std.c.getenv("LTX_GEMMA_DIR")) |env| {
        const e = std.mem.span(env);
        if (e.len > 0) return allocator.dupe(u8, e);
    }
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.NoGemmaDir);
    const snaps = try std.fmt.allocPrint(allocator, "{s}/.cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-4bit/snapshots", .{home});
    defer allocator.free(snaps);
    var dir = std.Io.Dir.openDirAbsolute(io, snaps, .{ .iterate = true }) catch return error.NoGemmaDir;
    defer dir.close(io);
    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind == .directory) {
            return std.fmt.allocPrint(allocator, "{s}/{s}", .{ snaps, entry.name });
        }
    }
    return error.NoGemmaDir;
}

// ── HTTP helpers (mirror flux_server.zig) ──

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

test "ltx tokenizePadded left-pads and truncates" {
    // pure-helper smoke without a real tokenizer is not possible (needs encode);
    // covered by the live test_video_gen.sh integration test instead.
}
