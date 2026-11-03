//! D1 typed decisions on the media table: `POST /v1/decisions` through LiquidAI's D1-3B (an LFM2-VL trunk); the answer
//! is a softmax over each option's token group at the LM head, no new weights. Text only; one request at a time.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const d1 = @import("d1.zig");
const decision = @import("../laya/decision_job.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const Backend = struct {
    pub const Engine = d1.Engine;
    pub const pre = null;
    pub const Extra = void;

    pub fn prepare(_: *Engine, _: std.mem.Allocator, body: *const decision.Body) !void {
        if (body.get("images") != null) return error.D1TextOnly;
    }
    pub fn freeExtra(_: std.mem.Allocator, _: void) void {}

    pub fn predict(e: *Engine, a: std.mem.Allocator, model_id: []const u8, body: *const decision.Body, qs: *const d1.Questions, _: void, max_input_tokens: usize) ![]u8 {
        return e.predict(a, model_id, body.state, qs, max_input_tokens);
    }

    pub fn refusal(req: *const abi.Request, l: decision.Limits, err: anyerror) abi.Verdict {
        if (err == error.D1TextOnly) return abi.Verdict.refuse(.unsupported, "D1 decisions are text-only in this build");
        return decision.refusal(req, l, err, d1.errorMessage);
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

const H = 128;
const INTER = 256;
const VOCAB = 260;

/// A tiny random D1: a two-layer LFM2 trunk (a gated short conv, then attention) at 128 under LFM2-VL's config with
/// the card's `auto_map`, over a byte tokenizer with the chat markers as added tokens.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    var t = sdk.tensors.Tensors.init(a, 81);
    const p = "language_model.model.";
    try t.rand(p ++ "embed_tokens.weight", .{}, &.{ VOCAB, H }, .BF16, 0.5);
    try t.rand(p ++ "embedding_norm.weight", .{}, &.{H}, .BF16, 0.1);
    for (0..2) |i| {
        try t.rand(p ++ "layers.{d}.operator_norm.weight", .{i}, &.{H}, .BF16, 0.1);
        try t.rand(p ++ "layers.{d}.ffn_norm.weight", .{i}, &.{H}, .BF16, 0.1);
        try t.rand(p ++ "layers.{d}.feed_forward.w1.weight", .{i}, &.{ INTER, H }, .BF16, 0.08);
        try t.rand(p ++ "layers.{d}.feed_forward.w3.weight", .{i}, &.{ INTER, H }, .BF16, 0.08);
        try t.rand(p ++ "layers.{d}.feed_forward.w2.weight", .{i}, &.{ H, INTER }, .BF16, 0.06);
    }
    try t.rand(p ++ "layers.0.conv.in_proj.weight", .{}, &.{ 3 * H, H }, .BF16, 0.08);
    try t.rand(p ++ "layers.0.conv.conv.weight", .{}, &.{ H, 3, 1 }, .BF16, 0.5);
    try t.rand(p ++ "layers.0.conv.out_proj.weight", .{}, &.{ H, H }, .BF16, 0.08);
    inline for (.{ .{ "q_proj", 4 * 32, H }, .{ "k_proj", 2 * 32, H }, .{ "v_proj", 2 * 32, H }, .{ "out_proj", H, 4 * 32 } }) |l|
        try t.rand(p ++ "layers.1.self_attn." ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .BF16, 0.08);
    try t.rand(p ++ "layers.1.self_attn.q_layernorm.weight", .{}, &.{32}, .BF16, 0.1);
    try t.rand(p ++ "layers.1.self_attn.k_layernorm.weight", .{}, &.{32}, .BF16, 0.1);
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = 
            \\{"model_type":"lfm2_vl","auto_map":{"AutoModel":"modeling_d1.D1Model"},"tie_word_embeddings":true,
            \\ "text_config":{"model_type":"lfm2","hidden_size":128,"intermediate_size":256,"num_hidden_layers":2,
            \\ "num_attention_heads":4,"num_key_value_heads":2,"vocab_size":260,"norm_eps":1e-5,"rope_theta":1000000.0,
            \\ "max_position_embeddings":4096,"conv_L_cache":3,"layer_types":["conv","full_attention"],"tie_word_embeddings":true}}
            },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{
                .{ .id = 256, .content = "<|startoftext|>" }, .{ .id = 257, .content = "<|im_start|>" },
                .{ .id = 258, .content = "<|im_end|>" },      .{ .id = 259, .content = "<|pad|>" },
            }) },
            .{ .name = "tokenizer_config.json", .bytes = "{}" },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
        }),
        // Byte tokens hold every letter and digit as one token, never "yes": choice and score only.
        .body =
        \\{"state":{"order":"arrived broken"},"questions":{"tone":{"type":"choice","instructions":"tone","criteria":["calm","angry"]},"urgency":{"type":"score","instructions":"how urgent","criteria":["low","mid","high"]}}}
        ,
    };
}

// ── tests ──

const testing = std.testing;

test "d1 plugin: a decision on the tiny pack is deterministic JSON; images are refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    const parsed = try std.json.parseFromSlice(std.json.Value, arena.allocator(), first, .{});
    try testing.expect(parsed.value.object.get("answers") != null);
    try testing.expectEqualStrings(first, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("D1 decisions are text-only in this build", (try pack.run("{\"state\":\"s\",\"questions\":{},\"images\":[]}")).refused.why.slice());
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());

// Bar: the served LFM2 trunk scores branches bit-equal to the engine's deleted legacy trunk: every branch's last-row
// logits through the trunk head, replayed from `fixtures/trunk.tape`.
test "d1 plugin: the served LFM2 trunk scores branches bit-equal to the engine's own trunk" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const mlx = @import("../../mlx/ffi.zig");
    const kev = @import("../kev/kev.zig");
    const Tape = @import("../../sdk/testing/tape.zig").Tape;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const fx = try conformanceFixture(arena.allocator());
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for (fx.files) |f| try tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
    var pbuf: [512]u8 = undefined;
    const dir = pbuf[0..try tmp.dir.realPath(io, &pbuf)];

    const state = [_]u32{ 256, 257, 72, 101, 108, 108, 111, 10, 83, 58, 32, 120 };
    const branches = [_][]const u32{ &.{ 81, 58, 32, 65 }, &.{ 81, 58, 32, 66, 67 }, &.{ 258, 10 } };
    const Collect = struct {
        eng: *d1.Engine,
        out: [branches.len]mlx.mlx_array = undefined,
        fn run(self: *@This(), i: usize, h: mlx.mlx_array) !void {
            const sh = mlx.getShape(h);
            var last = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(last);
            try mlx.check(mlx.mlx_slice(&last, h, &[_]c_int{ 0, sh[1] - 1, 0 }, 3, &[_]c_int{ 1, sh[1], sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, self.eng.trunk.s));
            self.out[i] = try self.eng.headLogits(last);
            try mlx.check(mlx.mlx_array_eval(self.out[i]));
        }
    };
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const eng = try d1.Engine.load(io, a, dir, s);
    defer eng.deinit();
    var got: Collect = .{ .eng = eng };
    try kev.scoreBranches(a, &eng.trunk, &state, &branches, &got, Collect.run);
    defer for (got.out) |x| {
        _ = mlx.mlx_array_free(x);
    };

    var tape = try Tape.player(@embedFile("fixtures/trunk.tape"));
    for (got.out) |x| try tape.expectNext(x, false);
    try tape.expectDone();
}
