//! Media requests served through a plugin's table, on the inference thread: the route check, the LoRA stack, the
//! uploads, the request bill against live GPU headroom, the job with chat interleaved at its boundaries (previews
//! included), the failure arm and the modality's response; decision requests queued together run as one batch.
//! No backend logic: the plugin validates its own fields.

const std = @import("std");
const abi = @import("../../../abi/media.zig");
const mlx = @import("../../../mlx/ffi.zig");
const lora_mod = @import("../../../sdk/weights/lora.zig");
const gen = @import("../gen.zig");
const sse = @import("../sse.zig");
const media_progress = @import("../../../sdk/media/progress.zig");
const log = @import("../../../sdk/util/log.zig");
const server_mod = @import("../../http/server.zig");
const engine = @import("engine.zig");
const Engine = engine.Engine;

const Conn = server_mod.Conn;

/// Serves `route` on `e`'s table; the response (or nothing, when the client left) is written to `conn`.
pub fn handle(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, e: *Engine, route: gen.GenRoute, model_id: []const u8) !void {
    const t = e.table;
    if (routeHint(e, route)) |hint| return gen.sendError(conn, 400, hint);

    var path_bufs: [lora_mod.MAX_LORAS][]u8 = undefined;
    var scales: [lora_mod.MAX_LORAS]f32 = undefined;
    const n = gen.parseLoraFields(allocator, body, &path_bufs, &scales, server_mod.peerIsLoopback(conn)) catch |err| switch (err) {
        error.TooManyLoraPaths => return gen.sendError(conn, 400, "too many 'lora_paths' (max 8)"),
        error.BadLoraPathsJson => return gen.sendError(conn, 400, "invalid 'lora_paths' (must be a JSON array of strings)"),
        error.BadLoraScalesJson => return gen.sendError(conn, 400, "invalid 'lora_scales' (numbers, comma/space separated, or a JSON array)"),
        error.LoraPathsLocalOnly => return gen.sendError(conn, 403, gen.LORA_LOCAL_ONLY_MSG),
        error.OutOfMemory => return err,
    };
    defer for (path_bufs[0..n]) |p| allocator.free(p);
    if (t.set_loras == null and n > 0) return gen.sendError(conn, 400, "this model does not take LoRA adapters ('lora_path' / 'lora_paths')");
    if (t.set_loras) |set| {
        var loras: [lora_mod.MAX_LORAS]abi.Lora = undefined;
        for (path_bufs[0..n], scales[0..n], loras[0..n]) |p, s, *l| l.* = .{ .path = .of(p), .scale = s };
        const v = set(e.handle, .of(loras[0..n]));
        if (!v.isOk()) return gen.sendError(conn, 400, v.why.slice());
    }

    var inputs: std.ArrayList([]u8) = .empty;
    defer {
        for (inputs.items) |b| allocator.free(b);
        inputs.deinit(allocator);
    }
    try collectInputs(allocator, body, t.modality, &inputs);
    const spans = try allocator.alloc(abi.Str, inputs.items.len);
    defer allocator.free(spans);
    for (inputs.items, spans) |b, *s| s.* = .of(b);

    var job = Job.init(allocator, conn, body, model_id, t.modality != .decision);
    job.wire();
    job.req.inputs = .of(spans);
    job.req.dir = &e.pack;
    const had_error = mlx.errorPending();
    const v = t.step(e.handle, &job.req, &job.sink.progress());
    if (!v.isOk() and job.sink.left) {
        log.info("[gen] {s} job cancelled: the client left\n", .{t.name.slice()});
        return;
    }
    if (!v.isOk()) return job.fail(v);
    // A failure latched during the job is the job's, even when the plugin answered ok: never a 200 over it.
    if (!had_error and mlx.errorPending()) return job.fail(abi.Verdict.refuse(.failed, "generation failed"));
    var out: abi.Str = .{};
    const ev = t.encode(e.handle, &out);
    if (!ev.isOk()) return job.fail(ev);
    log.info("[gen] {s} -> {d} bytes\n", .{ t.name.slice(), out.len });
    var extra: abi.Str = .{};
    if (t.extras) |x| x(e.handle, &extra);
    return job.send(t.modality, out.slice(), extra.slice());
}

/// One queued request of a merged decision pass.
pub const Queued = struct { allocator: std.mem.Allocator, conn: *Conn, body: []const u8 };

/// Decision requests queued together, answered in one `step_batch` pass (one verdict and result each); a table
/// without `step_batch` answers them one by one.
pub fn handleMany(e: *Engine, model_id: []const u8, queued: []const Queued) void {
    const t = e.table;
    const batch = t.step_batch orelse {
        for (queued) |q| handle(q.allocator, q.conn, q.body, e, .decisions, model_id) catch |err|
            log.warn("[gen] {s} job failed: {s}\n", .{ t.name.slice(), @errorName(err) });
        return;
    };
    const a = std.heap.c_allocator;
    const jobs = a.alloc(Job, queued.len) catch return failAll(queued);
    defer a.free(jobs);
    const reqs = a.alloc(abi.Request, queued.len) catch return failAll(queued);
    defer a.free(reqs);
    const out = a.alloc(abi.Verdict, queued.len) catch return failAll(queued);
    defer a.free(out);
    const results = a.alloc(abi.Str, queued.len) catch return failAll(queued);
    defer a.free(results);
    for (queued, jobs, reqs) |q, *j, *r| {
        j.* = Job.init(q.allocator, q.conn, q.body, model_id, false);
        j.wire();
        j.req.dir = &e.pack;
        r.* = j.req;
    }
    const t0 = std.Io.Timestamp.now(queued[0].conn.io, .boot);
    const turn: abi.Progress = .{ .ctx = null, .report = quietReport, .boundary = chatTurn };
    batch(e.handle, .of(reqs), out.ptr, results.ptr, &turn);
    const ms = @as(f64, @floatFromInt(t0.untilNow(queued[0].conn.io, .boot).nanoseconds)) / 1e6;
    if (queued.len > 1)
        log.info("[decision] {s}: {d} requests merged in one pass, {d:.1} ms\n", .{ t.name.slice(), queued.len, ms })
    else
        log.info("[decision] {s}: 1 request in {d:.1} ms\n", .{ t.name.slice(), ms });
    for (jobs, out, results) |*j, v, r| {
        (if (v.isOk()) j.send(t.modality, r.slice(), "") else j.fail(v)) catch |err|
            log.warn("[gen] response not sent: {s}\n", .{@errorName(err)});
    }
}

fn failAll(queued: []const Queued) void {
    for (queued) |q| gen.sendError(q.conn, 500, "out of memory") catch {};
}

fn quietReport(_: ?*anyopaque, _: abi.Str, _: u32, _: u32) callconv(.c) void {}

/// A merged pass belongs to no one client: its boundary gives chat its turn and never stops the pass.
fn chatTurn(_: ?*anyopaque) callconv(.c) bool {
    if (media_progress.genYield().*) |y| y.step(y.ctx);
    return false;
}

/// The merge weight of a queued decision request: its question count (1 for a body the plugin will refuse).
pub fn questionCount(body: []const u8) u32 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const v = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), body, .{}) catch return 1;
    if (v != .object) return 1;
    const q = v.object.get("questions") orelse return 1;
    return if (q == .object) @intCast(@max(1, q.object.count())) else 1;
}

/// Speaker activity over `pcm` (f32 mono, 16 kHz) through a diarization plugin: one probability per speaker per frame
/// (owned by `a`) and the speaker count; null after logging a failure.
pub fn diarize(e: *Engine, a: std.mem.Allocator, pcm: []const f32) ?struct { probs: []f32, speakers: u32 } {
    const t = e.table;
    const inputs = [_]abi.Str{.of(std.mem.sliceAsBytes(pcm))};
    var msg_buf: [512]u8 = undefined;
    var msg_len: usize = 0;
    const mach = engine.machine();
    const req: abi.Request = .{ .body = .of("{}"), .inputs = .of(&inputs), .message = .{ .ptr = &msg_buf, .cap = msg_buf.len, .len = &msg_len }, .machine = &mach, .dir = &e.pack };
    const turn: abi.Progress = .{ .ctx = null, .report = quietReport, .boundary = chatTurn };
    const v = t.step(e.handle, &req, &turn);
    var out: abi.Str = .{};
    const ev = if (v.isOk()) t.encode(e.handle, &out) else v;
    if (!ev.isOk()) {
        log.err("  diarization failed: {s}\n", .{if (msg_len > 0) msg_buf[0..msg_len] else ev.why.slice()});
        return null;
    }
    const bytes = out.slice();
    if (bytes.len < 4 or (bytes.len - 4) % 4 != 0) {
        log.err("  diarization failed: {s} returned {d} bytes\n", .{ t.name.slice(), bytes.len });
        return null;
    }
    const probs = a.alloc(f32, (bytes.len - 4) / 4) catch return null;
    @memcpy(std.mem.sliceAsBytes(probs), bytes[4..]);
    return .{ .probs = probs, .speakers = std.mem.readInt(u32, bytes[0..4], .little) };
}

/// The 400 for an audio route the pack does not serve, naming the route it does; null when it serves `route`.
fn routeHint(e: *const Engine, route: gen.GenRoute) ?[]const u8 {
    if (e.table.modality != .audio) return null;
    const r = e.caps.routes;
    const served = switch (route) {
        .speech => r.speech,
        .music => r.music,
        .sound => r.sound,
        else => true,
    };
    if (served) return null;
    return if (r.music) gen.MUSIC_ROUTE_HINT else if (r.sound) gen.SOUND_ROUTE_HINT else gen.TTS_ROUTE_HINT;
}

/// The uploads in body order, base64-decoded (a WebP arrives as a PNG of the same pixels): image = `image` then each
/// `ref_images` entry, mesh = `image` then each `images` entry. An entry that is not a base64 string arrives empty:
/// the plugin names it, in its own order of checks.
fn collectInputs(a: std.mem.Allocator, body: []const u8, modality: abi.Modality, out: *std.ArrayList([]u8)) !void {
    if (modality != .image and modality != .mesh) return;
    const parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch return; // the plugin names a bad body
    defer parsed.deinit();
    if (parsed.value != .object) return;
    const o = parsed.value.object;
    if (o.get("image")) |v| try out.append(a, try decoded(a, v));
    const list = o.get(if (modality == .image) "ref_images" else "images") orelse return;
    if (list != .array) return;
    for (list.array.items) |v| try out.append(a, try decoded(a, v));
}

fn decoded(a: std.mem.Allocator, v: std.json.Value) ![]u8 {
    if (v != .string) return a.alloc(u8, 0);
    return gen.imageB64Alloc(a, v.string) catch |err| if (err == error.OutOfMemory) err else a.alloc(u8, 0);
}

/// One request's state through its job: the table `Request` (with the host's message buffer), the progress sink and
/// the response. `wire` must run once the job sits at its final address (the request points into it).
const Job = struct {
    allocator: std.mem.Allocator,
    req: abi.Request,
    sink: Sink,
    msg_buf: [512]u8 = undefined,
    msg_len: usize = 0,
    mach: abi.Machine = .{},

    /// `streams`: the modality answers `"stream": true` with SSE (decisions never do).
    fn init(allocator: std.mem.Allocator, conn: *Conn, body: []const u8, model_id: []const u8, streams: bool) Job {
        return .{
            .allocator = allocator,
            .req = .{ .body = .of(body), .model_id = .of(model_id) },
            .sink = .{ .sctx = gen.videoStreamCtx(conn, allocator, body, streams and sse.bodyWantsTrue(body, "stream")) },
        };
    }

    fn wire(self: *Job) void {
        self.req.message = .{ .ptr = &self.msg_buf, .cap = self.msg_buf.len, .len = &self.msg_len };
        self.mach = engine.machine();
        self.req.machine = &self.mach;
        self.sink.prog = self.sink.sctx.progress();
    }

    /// The refusal's text: what the plugin composed, else its static reason.
    fn text(self: *const Job, v: abi.Verdict) []const u8 {
        return if (self.msg_len > 0) self.msg_buf[0..self.msg_len] else v.why.slice();
    }

    /// The failure arm: a refusal of the request is a 400, the machine's capacity a 503, anything else a 500 (or a
    /// 503 when the latched MLX failure says memory); once the stream began, an SSE `error` event.
    fn fail(self: *Job, v: abi.Verdict) !void {
        const conn = self.sink.sctx.conn;
        const msg = self.text(v);
        log.err("[gen] job failed: {s}\n", .{msg});
        const refusal = v.code == .unsupported or v.code == .too_large;
        if (self.sink.sctx.stream and (self.sink.began or !refusal)) {
            try self.sink.begin();
            sse.sendError(conn, msg);
            return;
        }
        const status: u16 = if (refusal) 400 else if (v.code == .busy) 503 else gen.jobFailureStatus(error.MediaJobFailed);
        return gen.sendError(conn, status, msg);
    }

    /// The modality's response for `bytes` (as `encode` defines them); `extra` = the table's `extras`.
    fn send(self: *Job, modality: abi.Modality, bytes: []const u8, extra: []const u8) !void {
        try self.sink.begin();
        const conn = self.sink.sctx.conn;
        const stream = self.sink.sctx.stream;
        return switch (modality) {
            .audio => gen.sendWavWith(self.allocator, conn, bytes, stream, extra),
            .image => gen.sendPngResult(self.allocator, conn, bytes, stream),
            .mesh => gen.sendGlbResult(self.allocator, conn, bytes, stream),
            .video => gen.sendJsonMembers(self.allocator, conn, bytes, stream),
            .decision => gen.sendBytesJson(conn, self.allocator, bytes),
            // No route reaches a diarizer (`diarize` serves the transcription companion).
            .diarization => unreachable,
        };
    }
};

/// The job's progress sink: SSE headers go out with the first event (a request the plugin refuses before any
/// progress is a plain 400), chat runs at each boundary, previews ride the stream, a client that left is remembered.
const Sink = struct {
    sctx: sse.StreamCtx,
    prog: sse.Progress = undefined,
    began: bool = false,
    left: bool = false,

    fn progress(self: *Sink) abi.Progress {
        const o = self.prog.preview_opts;
        return .{
            .ctx = self,
            .report = report,
            .boundary = boundary,
            .preview_opts = .{ .enabled = self.prog.wantsPreview(), .frames = o.frames, .max_side = o.max_side },
            .preview = if (self.prog.wantsPreview()) preview else null,
        };
    }

    fn begin(self: *Sink) !void {
        if (self.began) return;
        self.began = true;
        if (self.sctx.stream) try self.sctx.conn.writeAll(sse.headers);
    }

    /// An empty `stage` begins the response (SSE headers) without an event: a job that answered a streamed request
    /// from its first step reports its failures as SSE events from there on.
    fn report(ctx: ?*anyopaque, stage: abi.Str, n: u32, total: u32) callconv(.c) void {
        const self: *Sink = @ptrCast(@alignCast(ctx.?));
        self.begin() catch {
            self.sctx.cancelled = true;
            return;
        };
        if (stage.len > 0) self.prog.emit(stage.slice(), n, total);
    }

    fn preview(ctx: ?*anyopaque, stage: abi.Str, n: u32, total: u32, frame: *const abi.PreviewFrame) callconv(.c) void {
        const self: *Sink = @ptrCast(@alignCast(ctx.?));
        self.begin() catch {
            self.sctx.cancelled = true;
            return;
        };
        self.prog.emitPreview(stage.slice(), n, total, .{ .jpeg = @constCast(frame.bytes.slice()), .w = frame.w, .h = frame.h, .mime = frame.mime.slice() });
    }

    fn boundary(ctx: ?*anyopaque) callconv(.c) bool {
        const self: *Sink = @ptrCast(@alignCast(ctx.?));
        if (self.prog.boundary()) self.left = true;
        return self.left;
    }
};

// ── tests ──

const testing = std.testing;
const served_mod = @import("served.zig");
const sa_plugin = @import("../../../plugins/stable_audio3/plugin.zig");
const test_conn = @import("test_conn.zig");

const capture = test_conn.capture;
const captureMany = test_conn.captureMany;

test "plugin arm: a stable_audio3 pack is the plugin's, billed as the legacy arm billed it" {
    const a = testing.allocator;
    const io = testing.io;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fx = try sa_plugin.conformanceFixture(arena.allocator());
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "t5gemma-b-b-ul2");
    for (fx.files) |f| try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const path = pbuf[0..try tmp.dir.realPath(io, &pbuf)];

    // Identity and bills: the pack is the plugin's, billed as the legacy arm billed it (every safetensors, resident).
    const mt = gen.peekModelType(io, a, path).?;
    defer a.free(mt);
    try testing.expectEqualStrings("stable_audio3", mt);
    try testing.expectEqual(gen.Modality.audio, gen.detectModality(io, a, path).?);
    const legacy_bill = gen.sumSafetensorsAt(io, path);
    try testing.expect(legacy_bill > 0);
    try testing.expectEqual(legacy_bill, gen.estimatePeakResidentBytes(io, path, mt));

    const pe = try Engine.load(io, a, path, served_mod.byName(mt).?, mlx.mlx_default_cpu_stream_new());
    defer pe.deinit();
    try testing.expectEqual(@as(u64, 0), pe.requestPeak(fx.body));
}

test "plugin arm: chat gets one turn per step boundary, from the driver, through a backend's progress" {
    var sv: [2]std.posix.fd_t = undefined;
    try testing.expect(std.c.socketpair(1, 1, 0, &sv) == 0);
    defer _ = std.c.close(sv[1]);
    defer _ = std.c.close(sv[0]);
    var conn: Conn = undefined;
    conn.init(.{ .socket = .{ .handle = sv[0], .address = @import("test_conn.zig").loopback } }, testing.io);
    const Count = struct {
        var turns: u32 = 0;
        fn step(_: *anyopaque) void {
            turns += 1;
        }
        fn serve(_: *anyopaque, _: *const std.atomic.Value(bool)) void {}
        fn wake(_: *anyopaque) void {}
    };
    var dummy: u8 = 0;
    const hook = sse.Yield{ .ctx = &dummy, .step = Count.step, .serve_until = Count.serve, .wake = Count.wake };
    media_progress.genYield().* = &hook;
    defer media_progress.genYield().* = null;
    var sink = Sink{ .sctx = .{ .conn = &conn, .stream = false } };
    sink.prog = sink.sctx.progress();
    const p = sink.progress();
    const backend = @import("../../../sdk/media/job.zig").progressOf(&p);
    try testing.expect(!backend.boundary());
    try testing.expectEqual(@as(u32, 1), Count.turns);
}

/// A table whose behaviour each test sets: decision, video or diarization jobs over no weights.
const Fake = struct {
    var result: []const u8 = "";
    var request_peak: u64 = 0;
    var refused_load: ?[]const u8 = null;
    var handle_byte: u8 = 0;
    const jpeg = "\xff\xd8not-a-real-jpeg";

    fn table(m: abi.Modality, batch: bool) abi.MediaTable {
        return .{
            .name = .of("fake"),
            .claims = claims,
            .required_file = .of("x"),
            .modality = m,
            .caps = caps,
            .bill_load = billLoad,
            .bill_request = billRequest,
            .load = load,
            .unload = unload,
            .step = step,
            .encode = encode,
            .step_batch = if (batch) stepBatch else null,
        };
    }
    fn claims(_: abi.Str, _: *const abi.PackDir) callconv(.c) abi.Claim {
        return .no;
    }
    fn caps(_: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
        out.* = .{};
    }
    fn billLoad(_: *const abi.PackDir, _: *const abi.Machine) callconv(.c) u64 {
        return 1;
    }
    fn billRequest(_: *const abi.PackDir, _: *const abi.Machine, _: abi.Str) callconv(.c) u64 {
        return request_peak;
    }
    fn load(_: *const abi.LoadCtx, out: *abi.Verdict) callconv(.c) ?*anyopaque {
        if (refused_load) |n| {
            out.* = abi.Verdict.named(.failed, n);
            return null;
        }
        return &handle_byte;
    }
    fn unload(_: *anyopaque) callconv(.c) void {}
    /// "bad" in the body is refused with a composed text naming the model; otherwise one step with a preview. Every
    /// request carries the engine's pack.
    fn one(req: *const abi.Request) abi.Verdict {
        if (req.dir == null) return abi.Verdict.refuse(.failed, "no pack dir on the request");
        if (std.mem.indexOf(u8, req.body.slice(), "bad") != null)
            return req.refuseFmt(.unsupported, "bad request", "bad request for {s}", .{req.model_id.slice()});
        return .ok;
    }
    fn step(_: *anyopaque, req: *const abi.Request, p: *const abi.Progress) callconv(.c) abi.Verdict {
        const v = one(req);
        if (!v.isOk()) return v;
        p.report(p.ctx, .of("denoise"), 0, 2);
        if (p.preview) |f| f(p.ctx, .of("denoise"), 1, 2, &.{ .bytes = .of(jpeg), .w = 4, .h = 2 });
        if (p.boundary(p.ctx)) return abi.Verdict.refuse(.failed, "the client left");
        return .ok;
    }
    fn encode(_: *anyopaque, out: *abi.Str) callconv(.c) abi.Verdict {
        out.* = .of(result);
        return .ok;
    }
    /// Each merged request answers its own body back.
    fn stepBatch(_: *anyopaque, reqs: abi.Span(abi.Request), out: [*]abi.Verdict, results: [*]abi.Str, _: *const abi.Progress) callconv(.c) void {
        for (reqs.slice(), 0..) |*r, i| {
            out[i] = one(r);
            results[i] = r.body;
        }
    }
};

/// A fake engine over an empty folder; `deinit` it.
fn fakeEngine(dir: *testing.TmpDir, t: *const abi.MediaTable) !*Engine {
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const path = pbuf[0..try dir.dir.realPath(testing.io, &pbuf)];
    return Engine.load(testing.io, testing.allocator, path, t, mlx.mlx_default_cpu_stream_new());
}

fn sendJsonBody(conn: *Conn, a: std.mem.Allocator, json: []const u8) !void {
    return gen.sendBytesJson(conn, a, json);
}
fn sendErr(conn: *Conn, code: u16, msg: []const u8) !void {
    return gen.sendError(conn, code, msg);
}

test "plugin arm: the reserve is the table's request bill as is, below the load bill too; 0 leaves the load bill" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const t = Fake.table(.mesh, false);
    const e = try fakeEngine(&tmp, &t);
    defer e.deinit();
    defer Fake.request_peak = 0;
    for ([_]u64{ 0, 7 << 30 }) |peak| {
        Fake.request_peak = peak;
        try testing.expectEqual(peak, e.requestPeak("{}"));
    }
}

test "plugin arm: merged decision requests run as one batch, each answered with its own body or composed refusal" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const t = Fake.table(.decision, true);
    const e = try fakeEngine(&tmp, &t);
    defer e.deinit();
    const bodies = [_][]const u8{ "{\"questions\":{\"a\":{},\"b\":{}}}", "{\"bad\":1}" };
    const Run = struct {
        fn f(conns: *[2]Conn, eng: *Engine, al: std.mem.Allocator, bs: []const []const u8) void {
            var q: [2]Queued = undefined;
            for (&q, conns, bs) |*x, *c, b| x.* = .{ .allocator = al, .conn = c, .body = b };
            handleMany(eng, "laya-x", &q);
        }
    };
    const got = try captureMany(a, 2, Run.f, .{ e, a, &bodies });
    defer for (got) |g| a.free(g);
    const ok = try capture(a, sendJsonBody, .{ a, bodies[0] });
    defer a.free(ok);
    const bad = try capture(a, sendErr, .{ 400, "bad request for laya-x" });
    defer a.free(bad);
    try testing.expectEqualStrings(ok, got[0]);
    try testing.expectEqualStrings(bad, got[1]);
    try testing.expectEqual(@as(u32, 2), questionCount(bodies[0]));
    try testing.expectEqual(@as(u32, 1), questionCount("{\"questions\":[]}"));
}

fn videoRoute(conn: *Conn, a: std.mem.Allocator, body: []const u8, e: *Engine) !void {
    return handle(a, conn, body, e, .video, "v");
}
/// What the legacy video handlers wrote for the same job: SSE headers, a progress event, a preview event, the result.
fn legacyVideo(conn: *Conn, a: std.mem.Allocator, body: []const u8, members: []const u8) !void {
    const stream = sse.bodyWantsTrue(body, "stream");
    var sctx = gen.videoStreamCtx(conn, a, body, stream);
    const p = sctx.progress();
    if (stream) try conn.writeAll(sse.headers);
    p.emit("denoise", 0, 2);
    if (p.wantsPreview()) p.emitPreview("denoise", 1, 2, .{ .jpeg = @constCast(Fake.jpeg), .w = 4, .h = 2 });
    return gen.sendJsonMembers(a, conn, members, stream);
}

test "plugin arm: a video job writes the legacy bytes, previews and JSON members included" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const t = Fake.table(.video, false);
    const e = try fakeEngine(&tmp, &t);
    defer e.deinit();
    Fake.result = "\"frames\":1,\"height\":2,\"width\":4,\"fps\":24,\"format\":\"rgb8\",\"data\":\"AAAA\"";
    for ([_][]const u8{ "{}", "{\"stream\":true}", "{\"stream\":true,\"preview\":true}", "{\"bad\":1,\"stream\":true}" }) |body| {
        const got = try capture(a, videoRoute, .{ a, body, e });
        defer a.free(got);
        const want = if (std.mem.indexOf(u8, body, "bad") != null)
            try capture(a, sendErr, .{ 400, "bad request for v" })
        else
            try capture(a, legacyVideo, .{ a, body, Fake.result });
        defer a.free(want);
        try testing.expectEqualStrings(want, got);
    }
    // The members writer is the legacy body: `{"created":0,` + members + `}` as one JSON response.
    const whole = try std.fmt.allocPrint(a, "{{\"created\":0,{s}}}", .{Fake.result});
    defer a.free(whole);
    const one = try capture(a, sendJsonBody, .{ a, whole });
    defer a.free(one);
    const parts = try capture(a, legacyVideo, .{ a, "{}", Fake.result });
    defer a.free(parts);
    try testing.expectEqualStrings(one, parts);
}

fn audioRoute(conn: *Conn, a: std.mem.Allocator, body: []const u8, e: *Engine) !void {
    return handle(a, conn, body, e, .speech, "w");
}
fn legacyWav(conn: *Conn, a: std.mem.Allocator, body: []const u8, wav: []const u8, extra: []const u8) !void {
    const stream = sse.bodyWantsTrue(body, "stream");
    var sctx = gen.videoStreamCtx(conn, a, body, stream);
    const p = sctx.progress();
    if (stream) try conn.writeAll(sse.headers);
    p.emit("denoise", 0, 2);
    return gen.sendWavWith(a, conn, wav, stream, extra);
}

test "plugin arm: an audio result's extras ride the streamed complete event beside the WAV, never the plain body" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const Extras = struct {
        fn f(_: *anyopaque, out: *abi.Str) callconv(.c) void {
            out.* = .of(",\"abc\":\"X:1\"");
        }
    };
    var t = Fake.table(.audio, false);
    t.extras = Extras.f;
    const e = try fakeEngine(&tmp, &t);
    defer e.deinit();
    e.caps.routes.speech = true;
    Fake.result = "RIFFwav";
    for ([_][]const u8{ "{}", "{\"stream\":true}" }) |body| {
        const got = try capture(a, audioRoute, .{ a, body, e });
        defer a.free(got);
        const want = try capture(a, legacyWav, .{ a, body, Fake.result, ",\"abc\":\"X:1\"" });
        defer a.free(want);
        try testing.expectEqualStrings(want, got);
    }
}

test "plugin arm: a refused load keeps the plugin's own error name; a diarizer returns its speakers and probabilities" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const t = Fake.table(.diarization, false);
    Fake.refused_load = "MissingWeight";
    try testing.expectError(error.MediaPluginLoadFailed, fakeEngine(&tmp, &t));
    try testing.expectEqualStrings("MissingWeight", @import("engine.zig").loadErrorName(error.MediaPluginLoadFailed));
    try testing.expectEqualStrings("FileNotFound", @import("engine.zig").loadErrorName(error.FileNotFound));
    Fake.refused_load = null;
    const e = try fakeEngine(&tmp, &t);
    defer e.deinit();
    var res: [4 + 4 * 4]u8 = undefined;
    std.mem.writeInt(u32, res[0..4], 2, .little);
    const probs = [_]f32{ 0.25, 0.75, 1, 0 };
    @memcpy(res[4..], std.mem.sliceAsBytes(&probs));
    Fake.result = &res;
    const r = diarize(e, testing.allocator, &.{ 0, 0.5 }).?;
    defer testing.allocator.free(r.probs);
    try testing.expectEqual(@as(u32, 2), r.speakers);
    try testing.expectEqualSlices(f32, &probs, r.probs);
}
