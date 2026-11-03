//! The decision plugins' shared request handling (Laya, Kev, D1 and Clef on `/v1/decisions`): the host's limits, the
//! body parsed and bounded as the endpoint has always done it, every refusal named.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const job = @import("../../sdk/media/job.zig");
const laya = @import("laya.zig");
const mlx = @import("../../mlx/ffi.zig");

const Verdict = abi.Verdict;

/// The request bounds in force, from `LoadCtx.settings_json` (`MLX_SERVE_LAYA_MAX_QUESTIONS` /
/// `_MAX_INPUT_TOKENS` on the host); a missing or non-positive value keeps the default.
pub const Limits = struct {
    max_questions: usize = 64,
    max_input_tokens: usize = 32 * 1024,

    pub fn parse(settings_json: []const u8) Limits {
        var l: Limits = .{};
        var buf: [1024]u8 = undefined;
        var fba = std.heap.FixedBufferAllocator.init(&buf);
        const v = std.json.parseFromSliceLeaky(std.json.Value, fba.allocator(), settings_json, .{}) catch return l;
        if (v != .object) return l;
        inline for (.{ "max_questions", "max_input_tokens" }) |k| if (v.object.get(k)) |x| {
            if (x == .integer and x.integer > 0) @field(l, k) = @intCast(x.integer);
        };
        return l;
    }
};

/// `{"state": <string|object|array>, "questions": {id: {...}}, ...}`; `parsed` owns the values.
pub const Body = struct {
    parsed: std.json.Parsed(std.json.Value),
    state: std.json.Value,
    questions: std.json.Value,

    pub fn deinit(self: *Body) void {
        self.parsed.deinit();
    }

    pub fn get(self: *const Body, key: []const u8) ?std.json.Value {
        return self.parsed.value.object.get(key);
    }
};

/// `pre` checks a backend's own field before the shared ones (Clef's `truncate`), as the endpoint always has.
pub fn parse(a: std.mem.Allocator, body: []const u8, comptime pre: ?fn (std.json.ObjectMap) ?Verdict) union(enum) { ok: Body, bad: Verdict } {
    laya.checkJsonDepth(body) catch |err| return .{ .bad = Verdict.named(.unsupported, laya.errorMessage(err).?) };
    var parsed = laya.parseRequestJson(a, body) catch return .{ .bad = Verdict.refuse(.unsupported, "request body is not valid JSON") };
    if (parsed.value != .object) {
        parsed.deinit();
        return .{ .bad = Verdict.refuse(.unsupported, "request body must be a JSON object") };
    }
    const obj = parsed.value.object;
    if (pre) |f| if (f(obj)) |v| {
        parsed.deinit();
        return .{ .bad = v };
    };
    const state = obj.get("state") orelse {
        parsed.deinit();
        return .{ .bad = Verdict.refuse(.unsupported, "missing 'state'") };
    };
    const questions = obj.get("questions") orelse {
        parsed.deinit();
        return .{ .bad = Verdict.refuse(.unsupported, "missing 'questions'") };
    };
    return .{ .ok = .{ .parsed = parsed, .state = state, .questions = questions } };
}

/// The refusal for a failed parse or predict: the limits in force, then the backend's named errors, else the failure.
pub fn refusal(req: *const abi.Request, l: Limits, err: anyerror, comptime errorMessage: fn (anyerror) ?[]const u8) Verdict {
    return switch (err) {
        error.TooManyQuestions => req.refuseFmt(.unsupported, "too many questions in one request (MLX_SERVE_LAYA_MAX_QUESTIONS)", "too many questions in one request (limit {d}, MLX_SERVE_LAYA_MAX_QUESTIONS)", .{l.max_questions}),
        error.TooManyInputTokens => req.refuseFmt(.unsupported, "the questions total too many input tokens (MLX_SERVE_LAYA_MAX_INPUT_TOKENS); split them over several requests", "the questions total more than {d} input tokens (MLX_SERVE_LAYA_MAX_INPUT_TOKENS); split them over several requests", .{l.max_input_tokens}),
        else => if (errorMessage(err)) |m| Verdict.named(.unsupported, m) else job.failure(err, "decision forward failed"),
    };
}

/// A decision plugin answering one request at a time (Kev, D1, Clef: legacy ran them serially). `B` is the backend:
/// `Engine` (`load(io, a, dir, s)`, `deinit`, `parseQuestions`), `Questions`, `Extra` with `prepare(e, a, body)` /
/// `freeExtra(a, extra)` (what a request carries beside its questions), `predict(e, a, model_id, body, qs, extra,
/// max_input_tokens)`, `refusal(req, limits, err)`, and `pre` for `parse`.
pub fn Single(comptime name: []const u8, comptime B: type) type {
    return struct {
        const gpa = std.heap.c_allocator;

        const Engine = struct {
            e: *B.Engine,
            stream: mlx.mlx_stream,
            owns_stream: bool,
            limits: Limits,
            answer: []u8 = &.{},
        };

        pub fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
            const io = std.Io.Threaded.global_single_threaded.io();
            const owns = ctx.stream.ctx == null;
            const s: mlx.mlx_stream = if (owns) mlx.mlx_default_gpu_stream_new() else .{ .ctx = ctx.stream.ctx };
            const e = B.Engine.load(io, gpa, ctx.model_dir.slice(), s) catch |err| {
                if (owns) _ = mlx.mlx_stream_free(s);
                out.* = Verdict.named(.failed, @errorName(err));
                return null;
            };
            const self = gpa.create(Engine) catch {
                e.deinit();
                if (owns) _ = mlx.mlx_stream_free(s);
                out.* = Verdict.refuse(.failed, name ++ ": out of memory");
                return null;
            };
            self.* = .{ .e = e, .stream = s, .owns_stream = owns, .limits = .parse(ctx.settings_json.slice()) };
            return self;
        }

        pub fn unload(ep: *anyopaque) callconv(.c) void {
            const self: *Engine = @ptrCast(@alignCast(ep));
            gpa.free(self.answer);
            self.e.deinit();
            if (self.owns_stream) _ = mlx.mlx_stream_free(self.stream);
            gpa.destroy(self);
        }

        pub fn step(ep: *anyopaque, r: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
            const self: *Engine = @ptrCast(@alignCast(ep));
            var body = switch (parse(gpa, r.body.slice(), B.pre)) {
                .ok => |b| b,
                .bad => |v| return v,
            };
            defer body.deinit();
            const extra = B.prepare(self.e, gpa, &body) catch |err| return B.refusal(r, self.limits, err);
            defer B.freeExtra(gpa, extra);
            var qs = self.e.parseQuestions(gpa, body.questions, self.limits.max_questions) catch |err| return B.refusal(r, self.limits, err);
            defer qs.deinit(gpa);
            const p = job.progressOf(progress);
            if (p.boundary()) return job.client_left;
            const out = B.predict(self.e, gpa, r.model_id.slice(), &body, &qs, extra, self.limits.max_input_tokens) catch |err| return B.refusal(r, self.limits, err);
            p.emit("decide", 1, 1);
            gpa.free(self.answer);
            self.answer = out;
            return .ok;
        }

        pub fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
            const self: *Engine = @ptrCast(@alignCast(ep));
            if (self.answer.len == 0) return Verdict.refuse(.failed, name ++ ": no finished job to encode");
            out.* = .of(self.answer);
            return .ok;
        }
    };
}

// ── tests ──

const testing = std.testing;

test "decision job: the body's shape is refused by name before any engine runs; the limits name their values" {
    inline for (.{
        .{ "[1]", "request body must be a JSON object" },
        .{ "{\"questions\":{}}", "missing 'state'" },
        .{ "{\"state\":\"s\"}", "missing 'questions'" },
        .{ "{", "request body is not valid JSON" },
    }) |c| try testing.expectEqualStrings(c[1], parse(testing.allocator, c[0], null).bad.why.slice());

    var buf: [256]u8 = undefined;
    var len: usize = 0;
    const req: abi.Request = .{ .body = .{}, .message = .{ .ptr = &buf, .cap = buf.len, .len = &len } };
    const l = Limits.parse("{\"max_questions\":3,\"max_input_tokens\":0,\"batch_window_us\":0}");
    try testing.expectEqual(@as(usize, 32 * 1024), l.max_input_tokens);
    _ = refusal(&req, l, error.TooManyQuestions, laya.errorMessage);
    try testing.expectEqualStrings("too many questions in one request (limit 3, MLX_SERVE_LAYA_MAX_QUESTIONS)", buf[0..len]);
    _ = refusal(&req, l, error.TooManyInputTokens, laya.errorMessage);
    try testing.expectEqualStrings("the questions total more than 32768 input tokens (MLX_SERVE_LAYA_MAX_INPUT_TOKENS); split them over several requests", buf[0..len]);
    try testing.expectEqualStrings("each question must be an object", refusal(&req, l, error.QuestionNotObject, laya.errorMessage).why.slice());
    try testing.expectEqual(Verdict.Code.failed, refusal(&req, l, error.Unexpected, laya.errorMessage).code);
}
