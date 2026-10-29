//! Cloudflare Clef joint schema decisions on the media table: `POST /v1/decisions` through a qwen3_5 trunk (with its
//! Qwen3-VL tower when the pack has one) and the joint head; images ride the body. One request at a time.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const clef = @import("clef.zig");
const laya = @import("../laya/laya.zig");
const chat = @import("../../host/chat/chat.zig");
const decision = @import("../laya/decision_job.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

const Backend = struct {
    pub const Engine = clef.Engine;
    pub const Extra = struct { truncate: bool, images: []chat.ImageData };

    pub fn pre(obj: std.json.ObjectMap) ?abi.Verdict {
        const v = obj.get("truncate") orelse return null;
        return if (v == .bool) null else abi.Verdict.refuse(.unsupported, "truncate must be a boolean");
    }

    pub fn prepare(e: *Engine, a: std.mem.Allocator, body: *const decision.Body) !Extra {
        const t = body.get("truncate");
        return .{ .truncate = if (t) |v| v.bool else true, .images = try e.prepareImages(a, body.parsed.value) };
    }

    pub fn freeExtra(a: std.mem.Allocator, x: Extra) void {
        for (x.images) |im| a.free(im.pixels);
        a.free(x.images);
    }

    pub fn predict(e: *Engine, a: std.mem.Allocator, model_id: []const u8, body: *const decision.Body, qs: *const clef.Questions, x: Extra, max_input_tokens: usize) ![]u8 {
        return e.predict(a, model_id, body.state, qs, max_input_tokens, x.truncate, x.images);
    }

    fn message(err: anyerror) ?[]const u8 {
        return clef.errorMessage(err) orelse laya.errorMessage(err);
    }

    pub fn refusal(req: *const abi.Request, l: decision.Limits, err: anyerror) abi.Verdict {
        if (err == error.TooManyInputTokens) return req.refuseFmt(.unsupported, "Clef schema and state exceed the context limit", "Clef schema and state exceed the {d}-token context limit", .{@min(clef.MAX_LENGTH, l.max_input_tokens)});
        return decision.refusal(req, l, err, message);
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

/// The tiny trunk's config as a Clef pack declares it: a vision config and M-RoPE sections (the pack ships no tower).
const config_json =
    \\{"model_type":"qwen3_5","image_token_id":263,"vision_config":{"depth":1},"text_config":{"model_type":"qwen3_5",
    \\ "hidden_size":256,"intermediate_size":512,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
    \\ "head_dim":256,"vocab_size":264,"rms_norm_eps":1e-6,"full_attention_interval":2,"linear_num_key_heads":2,
    \\ "linear_num_value_heads":8,"linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4,
    \\ "max_position_embeddings":4096,"rope_parameters":{"rope_theta":100000.0,"partial_rotary_factor":0.25,
    \\ "mrope_section":[11,11,10],"mrope_interleaved":true},"tie_word_embeddings":false},"tie_word_embeddings":false}
;

/// A tiny random Clef: `sdk/testing/tiny_qwen35` (no vision tower) and a joint head at 32 with one routing layer and
/// one decoder layer.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const hidden = sdk.tiny_qwen35.hidden;
    const w = 32;
    const ff = 64;
    var t = sdk.tensors.Tensors.init(a, 91);
    try sdk.tiny_qwen35.weights(&t);
    var h = sdk.tensors.Tensors.init(a, 92);
    const Norm = struct {
        fn add(x: *sdk.tensors.Tensors, comptime base: []const u8, args: anytype, n: u64) !void {
            try x.fill(base ++ ".weight", args, &.{n}, .F32, 1);
            try x.fill(base ++ ".bias", args, &.{n}, .F32, 0);
        }
        fn lin(x: *sdk.tensors.Tensors, comptime base: []const u8, args: anytype, in: u64, out: u64, bias: bool) !void {
            try x.rand(base ++ ".weight", args, &.{ out, in }, .F32, 0.2);
            if (bias) try x.rand(base ++ ".bias", args, &.{out}, .F32, 0.1);
        }
        fn attn(x: *sdk.tensors.Tensors, comptime base: []const u8, args: anytype, n: u64) !void {
            try x.rand(base ++ ".in_proj_weight", args, &.{ 3 * n, n }, .F32, 0.2);
            try x.rand(base ++ ".in_proj_bias", args, &.{3 * n}, .F32, 0.1);
            try x.rand(base ++ ".out_proj.weight", args, &.{ n, n }, .F32, 0.2);
            try x.rand(base ++ ".out_proj.bias", args, &.{n}, .F32, 0.1);
        }
    };
    try Norm.add(&h, "hidden_norm", .{}, hidden);
    inline for (.{ "memory_projection", "question_projection", "option_question_projection", "global_projection", "option_context_projection", "option_lexical_projection" }) |n|
        try Norm.lin(&h, n, .{}, hidden, w, false);
    try h.rand("type_embedding.weight", .{}, &.{ 3, w }, .F32, 0.2);
    inline for (.{ "option_summary_norm", "field_norm", "option_norm" }) |n| try Norm.add(&h, n, .{}, w);
    try Norm.lin(&h, "residual_scorer.0", .{}, 4 * w, w, true);
    try Norm.lin(&h, "residual_scorer.3", .{}, w, 1, true);
    inline for (.{ "prior_logit_scale", "joint_logit_scale", "residual_gate" }) |n| try h.fill(n, .{}, &.{}, .F32, 1);
    inline for (.{ "query_norm", "memory_norm", "feedforward_norm" }) |n| try Norm.add(&h, "evidence_layers.0." ++ n, .{}, w);
    try Norm.attn(&h, "evidence_layers.0.attention", .{}, w);
    try Norm.lin(&h, "evidence_layers.0.feedforward.0", .{}, w, ff, true);
    try Norm.lin(&h, "evidence_layers.0.feedforward.3", .{}, ff, w, true);
    inline for (.{ "norm1", "norm2", "norm3" }) |n| try Norm.add(&h, "layers.0." ++ n, .{}, w);
    inline for (.{ "self_attn", "multihead_attn" }) |n| try Norm.attn(&h, "layers.0." ++ n, .{}, w);
    try Norm.lin(&h, "layers.0.linear1", .{}, w, ff, true);
    try Norm.lin(&h, "layers.0.linear2", .{}, ff, w, true);
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = config_json },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{
                .{ .id = 256, .content = "<|im_start|>" }, .{ .id = 257, .content = "<|im_end|>" },
                .{ .id = 258, .content = "<think>" },      .{ .id = 259, .content = "</think>" },
            }) },
            .{ .name = "tokenizer_config.json", .bytes = "{}" },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
            .{ .name = "joint_head.safetensors", .bytes = try h.bytes() },
            .{ .name = "joint_head_config.json", .bytes = "{\"hidden_size\":256,\"width\":32,\"routing_layers\":1,\"layers\":1,\"heads\":2,\"feedforward\":64}" },
        }),
        .body =
        \\{"state":"the order arrived broken","questions":{"refund":{"type":"noul","instructions":"refund?"},"tone":{"type":"choice","instructions":"tone","criteria":{"calm":"","angry":""}}}}
        ,
    };
}

// ── tests ──

const testing = std.testing;

test "clef plugin: a decision on the tiny pack is deterministic JSON; truncate and images are checked by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    const parsed = try std.json.parseFromSlice(std.json.Value, arena.allocator(), first, .{});
    try testing.expect(parsed.value.object.get("answers") != null);
    try testing.expectEqualStrings(first, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("truncate must be a boolean", (try pack.run("{\"truncate\":1}")).refused.why.slice());
    try testing.expectEqualStrings("this Clef checkpoint has no vision tower", (try pack.run("{\"state\":\"s\",\"questions\":{},\"images\":[\"AAAA\"]}")).refused.why.slice());
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
