//! Laya typed decisions on the media table: `POST /v1/decisions` with a state and questions in, the reference
//! `predict` JSON out (an mmBERT/ModernBERT encoder + decision head). Requests queued together answer in one pass.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const mlx = @import("../../mlx/ffi.zig");
const laya = @import("laya.zig");
const decision = @import("decision_job.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = &config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .decision,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .step = &step,
    .encode = &encode,
    .step_batch = &stepBatch,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Engine = struct {
    e: *laya.Engine,
    stream: mlx.mlx_stream,
    owns_stream: bool,
    limits: decision.Limits,
    /// The last pass's answers, one per request (empty for a refused one).
    answers: [][]u8 = &.{},

    fn freeAnswers(self: *Engine) void {
        for (self.answers) |x| gpa.free(x);
        gpa.free(self.answers);
        self.answers = &.{};
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const owns = ctx.stream.ctx == null;
    const s: mlx.mlx_stream = if (owns) mlx.mlx_default_gpu_stream_new() else .{ .ctx = ctx.stream.ctx };
    const e = laya.Engine.load(io, gpa, ctx.model_dir.slice(), s) catch |err| {
        if (owns) _ = mlx.mlx_stream_free(s);
        out.* = Verdict.named(.failed, @errorName(err));
        return null;
    };
    const self = gpa.create(Engine) catch {
        e.deinit();
        if (owns) _ = mlx.mlx_stream_free(s);
        out.* = Verdict.refuse(.failed, "laya: out of memory");
        return null;
    };
    self.* = .{ .e = e, .stream = s, .owns_stream = owns, .limits = .parse(ctx.settings_json.slice()) };
    return self;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const self: *Engine = @ptrCast(@alignCast(ep));
    self.freeAnswers();
    self.e.deinit();
    if (self.owns_stream) _ = mlx.mlx_stream_free(self.stream);
    gpa.destroy(self);
}

/// A refusal of a validation or limit error: the checkpoint's option limit, then the shared ones.
fn refusal(self: *const Engine, req: *const abi.Request, err: anyerror) Verdict {
    var buf: [160]u8 = undefined;
    if (self.e.limitMessage(&buf, err)) |m| return req.refuseFmt(.unsupported, "a question has too many options for the token budget", "{s}", .{m});
    return decision.refusal(req, self.limits, err, laya.errorMessage);
}

/// Every request of one pass through `predictMany`; a request refused before the forward keeps its verdict.
fn run(self: *Engine, reqs: []const abi.Request, out: []Verdict, progress: *const abi.Progress) void {
    self.freeAnswers();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const fail = struct {
        fn all(o: []Verdict, v: Verdict) void {
            for (o) |*x| x.* = v;
        }
    }.all;
    self.answers = gpa.alloc([]u8, reqs.len) catch return fail(out, Verdict.refuse(.busy, "out of memory"));
    @memset(self.answers, &.{});
    const jobs = a.alloc(laya.Engine.Job, reqs.len) catch return fail(out, Verdict.refuse(.busy, "out of memory"));
    const live = a.alloc(usize, reqs.len) catch return fail(out, Verdict.refuse(.busy, "out of memory"));
    var n: usize = 0;
    for (reqs, out, 0..) |*r, *v, i| {
        const body = switch (decision.parse(a, r.body.slice(), null)) {
            .ok => |b| b,
            .bad => |bad| {
                v.* = bad;
                continue;
            },
        };
        const qs = a.create(laya.Questions) catch {
            v.* = Verdict.refuse(.busy, "out of memory");
            continue;
        };
        qs.* = self.e.parseQuestions(a, body.questions, self.limits.max_questions) catch |err| {
            v.* = refusal(self, r, err);
            continue;
        };
        jobs[n] = .{ .a = gpa, .model_id = r.model_id.slice(), .state = body.state, .questions = qs, .max_input_tokens = self.limits.max_input_tokens };
        live[n] = i;
        n += 1;
    }
    if (n == 0) return;
    const p = job.progressOf(progress);
    if (p.boundary()) {
        for (live[0..n]) |i| out[i] = job.client_left;
        return;
    }
    self.e.predictMany(jobs[0..n]);
    p.emit("decide", 1, 1);
    for (jobs[0..n], live[0..n]) |j, i| {
        if (j.result) |answer| {
            self.answers[i] = answer;
            out[i] = .ok;
        } else |err| out[i] = refusal(self, &reqs[i], err);
    }
}

fn step(ep: *anyopaque, r: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const self: *Engine = @ptrCast(@alignCast(ep));
    var v: [1]Verdict = undefined;
    run(self, r[0..1], &v, progress);
    return v[0];
}

fn stepBatch(ep: *anyopaque, reqs: abi.Span(abi.Request), out: [*]Verdict, results: [*]abi.Str, progress: *const abi.Progress) callconv(.c) void {
    const self: *Engine = @ptrCast(@alignCast(ep));
    const rs = reqs.slice();
    run(self, rs, out[0..rs.len], progress);
    for (results[0..rs.len], 0..) |*x, i| x.* = if (i < self.answers.len) .of(self.answers[i]) else .{};
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const self: *Engine = @ptrCast(@alignCast(ep));
    if (self.answers.len != 1 or self.answers[0].len == 0) return Verdict.refuse(.failed, "laya: no finished job to encode");
    out.* = .of(self.answers[0]);
    return .ok;
}

/// A tiny random Laya: a two-layer ModernBERT at 32 (one global, one sliding layer) over a byte-level tokenizer, one
/// head layer, the scorer and a two-action head.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const d = 32;
    const inter = 32;
    const vocab = 260;
    var t = sdk.tensors.Tensors.init(a, 51);
    try t.rand("encoder.embeddings.tok_embeddings.weight", .{}, &.{ vocab, d }, .F32, 0.5);
    try t.fill("encoder.embeddings.norm.weight", .{}, &.{d}, .F32, 1);
    try t.fill("encoder.final_norm.weight", .{}, &.{d}, .F32, 1);
    try t.rand("type_emb.weight", .{}, &.{ 3, d }, .F32, 0.5);
    for (0..2) |i| {
        if (i > 0) try t.fill("encoder.layers.{d}.attn_norm.weight", .{i}, &.{d}, .F32, 1);
        try t.rand("encoder.layers.{d}.attn.Wqkv.weight", .{i}, &.{ 3 * d, d }, .F32, 0.2);
        try t.rand("encoder.layers.{d}.attn.Wo.weight", .{i}, &.{ d, d }, .F32, 0.2);
        try t.fill("encoder.layers.{d}.mlp_norm.weight", .{i}, &.{d}, .F32, 1);
        try t.rand("encoder.layers.{d}.mlp.Wi.weight", .{i}, &.{ 2 * inter, d }, .F32, 0.2);
        try t.rand("encoder.layers.{d}.mlp.Wo.weight", .{i}, &.{ d, inter }, .F32, 0.2);
    }
    inline for (.{ "norm1", "norm2" }) |n| {
        try t.fill("head.layers.0." ++ n ++ ".weight", .{}, &.{d}, .F32, 1);
        try t.fill("head.layers.0." ++ n ++ ".bias", .{}, &.{d}, .F32, 0);
    }
    inline for (.{ .{ "self_attn.in_proj", 3 * d, d }, .{ "self_attn.out_proj", d, d }, .{ "linear1", 4 * d, d }, .{ "linear2", d, 4 * d } }) |l| {
        try t.rand("head.layers.0." ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .F32, 0.2);
        try t.rand("head.layers.0." ++ l[0] ++ ".bias", .{}, &.{l[1]}, .F32, 0.1);
    }
    try t.fill("scorer.layers.0.weight", .{}, &.{d}, .F32, 1);
    try t.fill("scorer.layers.0.bias", .{}, &.{d}, .F32, 0);
    inline for (.{ .{ "scorer.layers.1", d, d }, .{ "scorer.layers.3", 1, d }, .{ "act_head.layers.0", 256, d + 4 }, .{ "act_head.layers.2", 2, 256 } }) |l| {
        try t.rand(l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .F32, 0.2);
        try t.rand(l[0] ++ ".bias", .{}, &.{l[1]}, .F32, 0.1);
    }
    try t.fill("temperature", .{}, &.{3}, .F32, 1);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "encoder/config.json", .bytes = 
            \\{"model_type":"modernbert","hidden_activation":"gelu","hidden_size":32,"intermediate_size":32,
            \\ "num_attention_heads":2,"num_hidden_layers":2,"layer_types":["full_attention","sliding_attention"],
            \\ "local_attention":16,"max_position_embeddings":256,"norm_eps":1e-05,"vocab_size":260}
            },
            .{ .name = "rl_agent_config.json", .bytes = "{\"head_layers\":1,\"max_len\":192,\"head_max_len\":64,\"act_costs\":{\"escalate\":0.5},\"temperature\":[1.0,1.0,1.0]}" },
            .{ .name = "tokenizer/tokenizer_config.json", .bytes = "{\"cls_token\":\"<bos>\",\"sep_token\":\"<eos>\",\"pad_token\":\"<pad>\",\"mask_token\":\"<mask>\"}" },
            .{ .name = "tokenizer/tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{
                .{ .id = 256, .content = "<bos>" }, .{ .id = 257, .content = "<eos>" },
                .{ .id = 258, .content = "<pad>" }, .{ .id = 259, .content = "<mask>" },
            }) },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
        }),
        .body =
        \\{"state":"the order arrived broken","questions":{"q":{"type":"choice","instructions":"refund?","criteria":["yes","no"]}}}
        ,
    };
}

// ── tests ──

const testing = std.testing;

test "laya plugin: a decision on the tiny checkpoint is deterministic JSON; a bad question is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    const parsed = try std.json.parseFromSlice(std.json.Value, arena.allocator(), first, .{});
    try testing.expect(parsed.value == .object);
    try testing.expectEqualStrings(first, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("question 'type' must be one of choice, score, noul", (try pack.run("{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"rank\"}}}")).refused.why.slice());
}

test "laya plugin: queued requests answer in one pass, each with its own answer or refusal, echoing the model id" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    var buf: [2][256]u8 = undefined;
    var len = [2]usize{ 0, 0 };
    const reqs = [_]abi.Request{
        .{ .body = .of(pack.fx.body), .model_id = .of("laya-tiny") },
        .{ .body = .of("{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"rank\"}}}"), .message = .{ .ptr = &buf[1], .cap = 256, .len = &len[1] } },
    };
    var out: [2]Verdict = undefined;
    var results: [2]abi.Str = undefined;
    const Quiet = struct {
        fn report(_: ?*anyopaque, _: abi.Str, _: u32, _: u32) callconv(.c) void {}
        fn boundary(_: ?*anyopaque) callconv(.c) bool {
            return false;
        }
    };
    table.step_batch.?(pack.engine, .of(&reqs), &out, &results, &.{ .ctx = null, .report = Quiet.report, .boundary = Quiet.boundary });
    try testing.expect(out[0].isOk() and !out[1].isOk());
    try testing.expectEqualStrings("question 'type' must be one of choice, score, noul", out[1].why.slice());
    try testing.expect(std.mem.indexOf(u8, results[0].slice(), "\"laya-tiny\"") != null);
    try testing.expectEqual(@as(usize, 0), results[1].len);
    _ = buf[0];
}

test {
    _ = decision;
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
