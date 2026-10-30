//! MOSS-Image on the media table: text to image over its Qwen3.5 encoder, single-stream DiT and FLUX.2 small VAE,
//! CFG 3 by default with the negative branch skipping a block. A job is one prompt; the result is a PNG.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const image_request = @import("../../sdk/media/image_request.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const LoraSet = @import("../../sdk/media/lora_set.zig").LoraSet;
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const moss = @import("moss_image.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .image,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .set_loras = &setLoras,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Engine = struct {
    m: *moss.Engine,
    loras: LoraSet = .{},
    png: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const m = moss.Engine.loadOn(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch |err| {
        out.* = job.failure(err, "moss_image: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        m.deinit();
        out.* = Verdict.refuse(.failed, "moss_image: out of memory");
        return null;
    };
    e.* = .{ .m = m };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.loras.clear(&e.m.dit);
    e.m.deinit();
    gpa.free(e.png);
    gpa.destroy(e);
}

/// Adapters name the checkpoint's own modules (`layers.N.proj.q`).
fn setLoras(ep: *anyopaque, loras: abi.Span(abi.Lora)) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    return e.loras.set(gpa, loras.slice(), .generic, &e.m.dit);
}

const spec: image_request.Spec = .{ .guidance = true, .default_guidance = config.DEFAULT_GUIDANCE };

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return Verdict.refuse(.unsupported, "invalid JSON body");
    defer body.deinit();
    var inputs: [16][]const u8 = undefined;
    const in = req.inputs.slice();
    if (in.len > inputs.len) return Verdict.named(.unsupported, spec.too_many_refs);
    for (in, 0..) |s, i| inputs[i] = s.slice();
    const j = switch (image_request.parse(.{ .obj = body.parsed.value.object }, inputs[0..in.len], spec)) {
        .ok => |j| j,
        .bad => |v| return v,
    };
    const w = image_request.clamp16(j.req_w);
    const h = image_request.clamp16(j.req_h);
    const steps = e.m.stepsFor(j.steps);
    const sp = job.progressOf(progress);
    // A canvas past the default the load reserved can run past the working set, and that Metal OOM mid-denoise is
    // uncatchable: refused by name.
    const need = config.genTransientBytes(w, h);
    const room = job.gpuHeadroom();
    if (need > room) log.info("[image] bill refused: {d}x{d} needs {d} MB, headroom {d} MB\n", .{ w, h, need >> 20, room >> 20 });
    if (image_request.canvasRefusal(req, w, h, need, room)) |v| return v;
    const img = e.m.generateImage(j.prompt, w, h, j.seed, steps, .{ .guidance_scale = j.guidance_scale, .negative_prompt = j.negative_prompt }, sp) catch |err|
        return if (err == error.Cancelled) job.client_left else job.failure(err, "generation failed");
    defer _ = mlx.mlx_array_free(img);
    // The image is lazy until here: a Metal failure in the denoise surfaces in the encode.
    const png = image_pixels.toPng(gpa, img, e.m.s) catch |err| return job.failure(err, "generation failed");
    gpa.free(e.png);
    e.png = png;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.png.len == 0) return Verdict.refuse(.failed, "moss_image: no finished job to encode");
    out.* = .of(e.png);
    return .ok;
}

// ── the tiny pack ──

const specials = [_]sdk.byte_tokenizer.Special{ .{ .id = 256, .content = "<|im_start|>" }, .{ .id = 257, .content = "<|im_end|>" }, .{ .id = 258, .content = "<think>" }, .{ .id = 259, .content = "</think>" } };

/// A seeded random pack in the release's layout: a 2-layer Qwen3.5 encoder (full attention), a 2-layer DiT
/// (dim 64, 4 heads) and a 4-level VAE decoder at 32 channels. A 256x256 canvas is 16x16 latent tokens.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    const te_h = 32;
    const d = 64;
    const ffn = 96;
    const lc = 128;
    const vocab = 256 + specials.len;

    var te = T.init(a, 11);
    try te.rand("language_model.model.embed_tokens.weight", .{}, &.{ vocab, te_h }, .BF16, 0.5);
    try te.fill("language_model.model.norm.weight", .{}, &.{te_h}, .BF16, 1);
    for (0..2) |i| {
        try te.fill("language_model.model.layers.{d}.input_layernorm.weight", .{i}, &.{te_h}, .BF16, 1);
        try te.fill("language_model.model.layers.{d}.post_attention_layernorm.weight", .{i}, &.{te_h}, .BF16, 1);
        try te.rand("language_model.model.layers.{d}.self_attn.q_proj.weight", .{i}, &.{ 2 * 2 * 16, te_h }, .BF16, 0.2);
        try te.rand("language_model.model.layers.{d}.self_attn.k_proj.weight", .{i}, &.{ 16, te_h }, .BF16, 0.2);
        try te.rand("language_model.model.layers.{d}.self_attn.v_proj.weight", .{i}, &.{ 16, te_h }, .BF16, 0.2);
        try te.rand("language_model.model.layers.{d}.self_attn.o_proj.weight", .{i}, &.{ te_h, 2 * 16 }, .BF16, 0.2);
        try te.fill("language_model.model.layers.{d}.self_attn.q_norm.weight", .{i}, &.{16}, .BF16, 1);
        try te.fill("language_model.model.layers.{d}.self_attn.k_norm.weight", .{i}, &.{16}, .BF16, 1);
        try te.rand("language_model.model.layers.{d}.mlp.gate_proj.weight", .{i}, &.{ 64, te_h }, .BF16, 0.2);
        try te.rand("language_model.model.layers.{d}.mlp.up_proj.weight", .{i}, &.{ 64, te_h }, .BF16, 0.2);
        try te.rand("language_model.model.layers.{d}.mlp.down_proj.weight", .{i}, &.{ te_h, 64 }, .BF16, 0.2);
    }
    const te_cfg = try std.fmt.allocPrint(a,
        \\{{"model_type":"qwen3_5","text_config":{{"model_type":"qwen3_5_text","hidden_size":{d},"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":64,"vocab_size":{d},"rms_norm_eps":1e-6,"rope_theta":10000.0,"full_attention_interval":1,"max_position_embeddings":4096,
        \\ "linear_num_key_heads":1,"linear_num_value_heads":1,"linear_key_head_dim":16,"linear_value_head_dim":16,"linear_conv_kernel_dim":4}}}}
    , .{ te_h, vocab });

    var dit = T.init(a, 12);
    try dit.rand("video_embedder.weight", .{}, &.{ d, lc }, .BF16, 0.1);
    try dit.rand("video_embedder.bias", .{}, &.{d}, .BF16, 0.1);
    try dit.rand("text_embedder.proj.weight", .{}, &.{ d, te_h }, .BF16, 0.2);
    for (0..2) |i| {
        inline for (.{ "norm1", "norm2" }) |n| try dit.fill("text_embedder.blocks.{d}." ++ n ++ ".weight", .{i}, &.{d}, .BF16, 1);
        inline for (.{ "q_norm", "k_norm" }) |n| try dit.fill("text_embedder.blocks.{d}." ++ n ++ ".weight", .{i}, &.{16}, .BF16, 1);
        try dit.rand("text_embedder.blocks.{d}.qkv.weight", .{i}, &.{ 3 * d, d }, .BF16, 0.12);
        try dit.rand("text_embedder.blocks.{d}.proj.weight", .{i}, &.{ d, d }, .BF16, 0.12);
        try dit.rand("text_embedder.blocks.{d}.ffn_gate.weight", .{i}, &.{ ffn, d }, .BF16, 0.12);
        try dit.rand("text_embedder.blocks.{d}.ffn_up.weight", .{i}, &.{ ffn, d }, .BF16, 0.12);
        try dit.rand("text_embedder.blocks.{d}.ffn_down.weight", .{i}, &.{ d, ffn }, .BF16, 0.1);
    }
    try dit.rand("time_embedding.0.weight", .{}, &.{ d, 256 }, .BF16, 0.06);
    try dit.rand("time_embedding.0.bias", .{}, &.{d}, .BF16, 0.1);
    try dit.rand("time_embedding.2.weight", .{}, &.{ d, d }, .BF16, 0.12);
    try dit.rand("shared_block_modulation.weight", .{}, &.{ 6 * d, d }, .BF16, 0.1);
    try dit.rand("video_head_modulation.weight", .{}, &.{ 2 * d, d }, .BF16, 0.1);
    try dit.fill("final_norm_video.weight", .{}, &.{d}, .BF16, 1);
    try dit.rand("final_linear_video.weight", .{}, &.{ lc, d }, .BF16, 0.1);
    for (0..2) |i| {
        inline for (.{ "norm1", "norm2" }) |n| try dit.fill("layers.{d}." ++ n ++ ".weight", .{i}, &.{d}, .BF16, 1);
        inline for (.{ "q_norm", "k_norm" }) |n| try dit.fill("layers.{d}." ++ n ++ ".weight", .{i}, &.{16}, .BF16, 1);
        inline for (.{ "q", "k", "v", "o" }) |n| try dit.rand("layers.{d}.proj." ++ n ++ ".weight", .{i}, &.{ d, d }, .BF16, 0.12);
        try dit.rand("layers.{d}.proj.o.bias", .{i}, &.{d}, .BF16, 0.05);
        try dit.rand("layers.{d}.proj.ffn_up_gate.weight", .{i}, &.{ 2 * ffn, d }, .BF16, 0.12);
        try dit.rand("layers.{d}.proj.ffn_down.weight", .{i}, &.{ d, ffn }, .BF16, 0.1);
    }

    const c = 32;
    var vae = T.init(a, 13);
    try vae.rand("bn.running_mean", .{}, &.{lc}, .F32, 0.1);
    try vae.fill("bn.running_var", .{}, &.{lc}, .F32, 1);
    try vae.rand("post_quant_conv.weight", .{}, &.{ 32, 32, 1, 1 }, .F32, 0.2);
    try vae.rand("post_quant_conv.bias", .{}, &.{32}, .F32, 0.05);
    try vae.rand("decoder.conv_in.weight", .{}, &.{ c, 32, 3, 3 }, .F32, 0.06);
    try vae.rand("decoder.conv_in.bias", .{}, &.{c}, .F32, 0.05);
    for (0..2) |i| try resnet(&vae, "decoder.mid_block.resnets.{d}", .{i}, c);
    try vae.fill("decoder.mid_block.attentions.0.group_norm.weight", .{}, &.{c}, .F32, 1);
    try vae.fill("decoder.mid_block.attentions.0.group_norm.bias", .{}, &.{c}, .F32, 0);
    inline for (.{ "to_q", "to_k", "to_v", "to_out.0" }) |n| {
        try vae.rand("decoder.mid_block.attentions.0." ++ n ++ ".weight", .{}, &.{ c, c }, .F32, 0.15);
        try vae.rand("decoder.mid_block.attentions.0." ++ n ++ ".bias", .{}, &.{c}, .F32, 0.05);
    }
    for (0..4) |bi| {
        try resnet(&vae, "decoder.up_blocks.{d}.resnets.0", .{bi}, c);
        if (bi < 3) {
            try vae.rand("decoder.up_blocks.{d}.upsamplers.0.conv.weight", .{bi}, &.{ c, c, 3, 3 }, .F32, 0.06);
            try vae.rand("decoder.up_blocks.{d}.upsamplers.0.conv.bias", .{bi}, &.{c}, .F32, 0.05);
        }
    }
    try vae.fill("decoder.conv_norm_out.weight", .{}, &.{c}, .F32, 1);
    try vae.fill("decoder.conv_norm_out.bias", .{}, &.{c}, .F32, 0);
    try vae.rand("decoder.conv_out.weight", .{}, &.{ 3, c, 3, 3 }, .F32, 0.06);
    try vae.rand("decoder.conv_out.bias", .{}, &.{3}, .F32, 0.05);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "model_index.json", .bytes = "{\"_class_name\":\"MOSSImagePipeline\"}" },
            .{ .name = "tokenizer/tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &specials) },
            .{ .name = "text_encoder/config.json", .bytes = te_cfg },
            .{ .name = "text_encoder/model.safetensors", .bytes = try te.bytes() },
            .{ .name = "transformer/config.json", .bytes = "{\"dim\":64,\"ffn_dim\":96,\"num_heads\":4,\"num_layers\":2,\"text_dim\":32,\"latent_channels\":128}" },
            .{ .name = "transformer/diffusion_pytorch_model.safetensors", .bytes = try dit.bytes() },
            .{ .name = "vae/config.json", .bytes = "{\"batch_norm_eps\":0.0001}" },
            .{ .name = config.required_file, .bytes = try vae.bytes() },
        }),
        .body = "{\"prompt\":\"a red cat\",\"size\":\"256x256\",\"steps\":2,\"seed\":3,\"negative_prompt\":\"blurry\"}",
    };
}

fn resnet(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, c: u64) !void {
    try t.fill(fmt ++ ".norm1.weight", args, &.{c}, .F32, 1);
    try t.fill(fmt ++ ".norm1.bias", args, &.{c}, .F32, 0);
    try t.rand(fmt ++ ".conv1.weight", args, &.{ c, c, 3, 3 }, .F32, 0.06);
    try t.rand(fmt ++ ".conv1.bias", args, &.{c}, .F32, 0.05);
    try t.fill(fmt ++ ".norm2.weight", args, &.{c}, .F32, 1);
    try t.fill(fmt ++ ".norm2.bias", args, &.{c}, .F32, 0);
    try t.rand(fmt ++ ".conv2.weight", args, &.{ c, c, 3, 3 }, .F32, 0.06);
    try t.rand(fmt ++ ".conv2.bias", args, &.{c}, .F32, 0.05);
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack;
const golden = @import("../../sdk/testing/golden.zig");

test "moss_image: a job on the tiny pack is a 256x256 PNG; a variation is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const png = (try pack.run(pack.fx.body)).ok;
    try testing.expectEqual(image_pixels.Size{ .w = 256, .h = 256 }, image_pixels.nativeSize(png).?);
    const refused = (try pack.run("{\"prompt\":\"a\",\"image\":\"x\"}")).refused;
    try testing.expect(refused.code == .unsupported);
}

test "moss_image: a lazy image whose evaluation fails at the PNG encode is refused by class, never a stale image" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    // The job's checked mlx calls, counted until two runs agree: the encode's eval of the lazy image is the last two.
    var ops: u64 = 0;
    for (0..3) |_| {
        const n0 = mlx.op_count.load(.monotonic);
        _ = (try pack.run(pack.fx.body)).ok;
        const n = mlx.op_count.load(.monotonic) - n0;
        if (n == ops) break;
        ops = n;
    }
    var stale: [512]u8 = undefined;
    _ = mlx.takeError(&stale);
    defer _ = mlx.takeError(&stale);
    mlx.armLatchingFaultWithForTest(ops - 1, "[METAL] Command buffer execution failed: Insufficient Memory. at array.cpp:352");
    defer mlx.armLatchingFaultForTest(0);
    const v = (try pack.run(pack.fx.body)).refused;
    try testing.expect(mlx.latchingFaultFiredForTest());
    try testing.expectEqual(Verdict.Code.busy, v.code);
    try testing.expectEqualStrings("generation failed", v.why.slice());
}

test "moss_image: the tiny pack's pipeline (CFG with a skipped block) is bit-equal to its golden run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    e.m.uncond_skip = 1;
    const img = try e.m.generateImage("a red cat", 256, 256, 3, 2, .{ .guidance_scale = 3.0, .negative_prompt = "blurry" }, null);
    defer _ = mlx.mlx_array_free(img);
    try mlx.check(mlx.mlx_array_eval(img));
    try golden.expect("moss_image_pipeline", &.{.{ .name = "image", .array = img }});
}
