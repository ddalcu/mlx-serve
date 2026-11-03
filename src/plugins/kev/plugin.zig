//! Kev typed decisions on the media table: `POST /v1/decisions` through a qwen3_5 trunk and Kev's pointer head; the
//! state runs once, each question from a KV/SSM snapshot. One request at a time.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const kev = @import("kev.zig");
const decision = @import("../laya/decision_job.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const Backend = struct {
    pub const Engine = kev.Engine;
    pub const pre = null;
    pub const Extra = void;

    pub fn prepare(_: *Engine, _: std.mem.Allocator, _: *const decision.Body) !void {}
    pub fn freeExtra(_: std.mem.Allocator, _: void) void {}

    pub fn predict(e: *Engine, a: std.mem.Allocator, model_id: []const u8, body: *const decision.Body, qs: *const kev.Questions, _: void, max_input_tokens: usize) ![]u8 {
        return e.predict(a, model_id, body.state, qs, max_input_tokens);
    }

    pub fn refusal(req: *const abi.Request, l: decision.Limits, err: anyerror) abi.Verdict {
        return decision.refusal(req, l, err, kev.errorMessage);
    }
};

const J = decision.Single(plugin.name, Backend);

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = &config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .decision,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &J.load,
    .unload = &J.unload,
    .step = &J.step,
    .encode = &J.encode,
};

/// A tiny random Kev: a two-layer qwen3_5 trunk (a GatedDeltaNet layer, then gated attention) over a byte tokenizer
/// whose five delimiters are added tokens, and a pointer head at 32.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    var t = sdk.tensors.Tensors.init(a, 71);
    try sdk.tiny_qwen35.weights(&t);
    var head = sdk.tensors.Tensors.init(a, 72);
    inline for (.{ "q", "k" }) |n| {
        try head.rand(n ++ ".weight", .{}, &.{ 32, sdk.tiny_qwen35.hidden }, .F32, 0.1);
        try head.rand(n ++ ".bias", .{}, &.{32}, .F32, 0.1);
    }
    const specials = [_]sdk.byte_tokenizer.Special{
        .{ .id = 256, .content = "<|fim_prefix|>" }, .{ .id = 257, .content = "<|fim_middle|>" },
        .{ .id = 258, .content = "<|box_start|>" },  .{ .id = 259, .content = "<|box_end|>" },
        .{ .id = 260, .content = "<|fim_suffix|>" },
    };
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = sdk.tiny_qwen35.config },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &specials) },
            .{ .name = "tokenizer_config.json", .bytes = "{}" },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
            .{ .name = "kev_head.safetensors", .bytes = try head.bytes() },
            .{ .name = "kev_config.json", .bytes = 
            \\{"format":"kev","format_version":1,"head_dim":32,"temperature":1.5,"delimiters":{"state":"<|fim_prefix|>",
            \\ "question":"<|fim_middle|>","option":"<|box_start|>","option_end":"<|box_end|>","decide":"<|fim_suffix|>"}}
            },
        }),
        .body =
        \\{"state":"the order arrived broken","questions":{"refund":{"type":"noul","instructions":"refund?"},"tone":{"type":"choice","instructions":"tone","criteria":["calm","angry"]}}}
        ,
    };
}

// ── tests ──

const testing = std.testing;

test "kev plugin: a decision on the tiny pack is deterministic JSON echoing the model id; a bad question is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    const parsed = try std.json.parseFromSlice(std.json.Value, arena.allocator(), first, .{});
    try testing.expect(parsed.value.object.get("answers") != null);
    try testing.expectEqualStrings(first, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("question 'type' must be one of choice, score, noul", (try pack.run("{\"state\":\"s\",\"questions\":{\"q\":{\"type\":\"rank\"}}}")).refused.why.slice());
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());

// Bar: every branch's final-norm hidden rows bit-equal to the legacy trunk's, recorded in `fixtures/trunk.tape`.
test "kev plugin: the served qwen3_5 trunk scores branches bit-equal to the engine's own trunk" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const mlx = @import("../../mlx/ffi.zig");
    const tape_mod = @import("../../sdk/testing/tape.zig");
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fx = try conformanceFixture(arena.allocator());
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for (fx.files) |f| try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    var pbuf: [512]u8 = undefined;
    const dir = pbuf[0..try tmp.dir.realPath(io, &pbuf)];

    const state = [_]u32{ 256, 116, 104, 101, 32, 111, 114, 100, 101, 114, 10, 83, 58, 32, 120 };
    const branches = [_][]const u32{ &.{ 257, 81, 58, 32, 65 }, &.{ 257, 81, 58, 32, 66, 67, 258 }, &.{ 260, 10 } };
    const Collect = struct {
        out: [branches.len]mlx.mlx_array = @splat(.{ .ctx = null }),
        fn run(self: *@This(), i: usize, h: mlx.mlx_array) !void {
            self.out[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&self.out[i], h));
        }
    };
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const eng = try kev.Engine.load(io, a, dir, s);
    defer eng.deinit();
    var got: Collect = .{};
    try kev.scoreBranches(a, &eng.trunk, &state, &branches, &got, Collect.run);
    defer for (got.out) |x| {
        _ = mlx.mlx_array_free(x);
    };

    var tape = try tape_mod.Tape.player(@embedFile("fixtures/trunk.tape"));
    for (got.out) |x| try tape.expectNext(x, false);
    try tape.expectDone();
}
