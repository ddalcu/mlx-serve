//! Ming-Image on the media table: text to image over its MoE MLLM prompt conditioning, single-stream DiT and the
//! Qwen-Image VAE decoding RGBA. A job is one prompt at the release's nearest aspect bucket, or an edit of one
//! picture at its own bucket, resized back to its aspect; the result is a PNG with its alpha.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const image_request = @import("../../sdk/media/image_request.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const ming = @import("ming_image.zig");
const llm = @import("llm.zig");
const vision = @import("vision.zig");
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
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Engine = struct {
    m: *ming.Engine,
    png: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const m = ming.Engine.loadOn(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch |err| {
        out.* = job.failure(err, "ming_image: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        m.deinit();
        out.* = Verdict.refuse(.failed, "ming_image: out of memory");
        return null;
    };
    e.* = .{ .m = m };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.m.deinit();
    gpa.free(e.png);
    gpa.destroy(e);
}

const spec: image_request.Spec = .{ .guidance = true, .default_guidance = config.DEFAULT_GUIDANCE, .edit_ref_cap = 1, .too_many_refs = "Ming-Image edits one picture: 'ref_images' is not supported" };

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return Verdict.refuse(.unsupported, "invalid JSON body");
    defer body.deinit();
    var inputs: [16][]const u8 = undefined;
    const in = req.inputs.slice();
    if (in.len > inputs.len) return Verdict.named(.unsupported, spec.too_many_refs);
    for (in, 0..) |s, i| inputs[i] = s.slice();
    var sp = spec;
    sp.edit = e.m.edit != null;
    const j = switch (image_request.parse(.{ .obj = body.parsed.value.object }, inputs[0..in.len], sp)) {
        .ok => |j| j,
        .bad => |v| return v,
    };
    if (j.negative_prompt.len > 0) return Verdict.named(.unsupported, "'negative_prompt' is not supported by Ming-Image (its guidance contrasts the prompt with an empty condition)");
    if (j.mode == .edit) return editJob(e, req, &j, progress);
    if (promptRefusal(e, j.prompt, 0)) |v| return v;
    const dims = config.bucket(j.req_w, j.req_h);
    if (dims.w != j.req_w or dims.h != j.req_h) log.info("[ming-image] asked {d}x{d} -> bucket {d}x{d}\n", .{ j.req_w, j.req_h, dims.w, dims.h });
    const need = config.genTransientBytes(dims.w, dims.h);
    const room = job.gpuHeadroom();
    if (need > room) log.info("[image] bill refused: {d}x{d} needs {d} MB, headroom {d} MB\n", .{ dims.w, dims.h, need >> 20, room >> 20 });
    if (image_request.canvasRefusal(req, dims.w, dims.h, need, room)) |v| return v;
    const img = e.m.generateImage(j.prompt, dims.w, dims.h, j.seed, j.steps, .{ .guidance_scale = j.guidance_scale }, job.progressOf(progress)) catch |err|
        return if (err == error.Cancelled) job.client_left else job.failure(err, "generation failed");
    defer _ = mlx.mlx_array_free(img);
    // The image is lazy until here: a Metal failure in the denoise surfaces in the encode.
    const png = image_pixels.toPng(gpa, img, e.m.s) catch |err| return job.failure(err, "generation failed");
    gpa.free(e.png);
    e.png = png;
    return .ok;
}

/// A request's own fault, named before it is billed against live headroom: a prompt past the encoder's cap.
fn promptRefusal(e: *Engine, prompt: []const u8, picture: u32) ?Verdict {
    const ids = e.m.enc.promptIds(gpa, prompt, picture) catch |err| return switch (err) {
        error.PromptTooLong => Verdict.named(.unsupported, "'prompt' is longer than the 4096 tokens Ming-Image reads (with its chat template and the picture); shorten it"),
        else => job.failure(err, "generation failed"),
    };
    gpa.free(ids);
    return null;
}

/// An edit: the picture at its bucket in the release's edit table, the result resized (LANCZOS) to its aspect.
fn editJob(e: *Engine, req: *const abi.Request, j: *const image_request.Job, progress: *const abi.Progress) Verdict {
    const ed = &e.m.edit.?;
    if (j.size_given) return Verdict.named(.unsupported, "'size' is not supported for a Ming-Image edit: the result keeps the picture's aspect at about 1 MP");
    const src = image_pixels.decodeRgba(gpa, j.image.?) orelse return Verdict.refuse(.unsupported, "could not decode 'image' (PNG/JPEG/WebP supported)");
    defer src.deinit(gpa);
    if (promptRefusal(e, j.prompt, vision.tokensFor(ed.vcfg, src.h, src.w))) |v| return v;
    const sz = config.editSizes(src.w, src.h, config.EDIT_HIGHRES);
    // The reference rides as a second frame: the DiT's sequence doubles.
    const need = config.genTransientBytes(sz.canvas.w, sz.canvas.h * 2);
    const room = job.gpuHeadroom();
    if (image_request.canvasRefusal(req, sz.canvas.w, sz.canvas.h, need, room)) |v| return v;
    log.info("[ming-image] edit {d}x{d} -> canvas {d}x{d} -> {d}x{d}\n", .{ src.w, src.h, sz.canvas.w, sz.canvas.h, sz.out.w, sz.out.h });
    const img = e.m.editImage(j.prompt, src.pix, src.w, src.h, config.EDIT_HIGHRES, j.seed, j.steps, .{ .guidance_scale = j.guidance_scale }, job.progressOf(progress)) catch |err|
        return if (err == error.Cancelled) job.client_left else job.failure(err, "generation failed");
    defer _ = mlx.mlx_array_free(img);
    const px = image_pixels.toPixels8(gpa, img, e.m.s) catch |err| return job.failure(err, "generation failed");
    defer px.deinit(gpa);
    const out: image_pixels.Pixels8 = if (px.w == sz.out.w and px.h == sz.out.h) px else .{
        .px = vision.resizeRgba(gpa, px.px, px.h, px.w, sz.out.h, sz.out.w, .lanczos) catch |err| return job.failure(err, "generation failed"),
        .w = sz.out.w,
        .h = sz.out.h,
        .channels = px.channels,
    };
    defer if (out.px.ptr != px.px.ptr) gpa.free(out.px);
    const png = image_pixels.encodePng(gpa, out) catch |err| return job.failure(err, "generation failed");
    gpa.free(e.png);
    e.png = png;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.png.len == 0) return Verdict.refuse(.failed, "ming_image: no finished job to encode");
    out.* = .of(e.png);
    return .ok;
}

// ── the tiny pack ──

const specials = [_]sdk.byte_tokenizer.Special{ .{ .id = 256, .content = "<role>" }, .{ .id = 257, .content = "</role>" }, .{ .id = 258, .content = "<|role_end|>" } };
const patch_token = 259; // start 260, end 261
const vocab = 262;
const T = sdk.tensors.Tensors;

fn dense(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64, bias: bool, dt: anytype) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in }, dt, 0.8 / @sqrt(@as(f32, @floatFromInt(in))));
    if (bias) try t.rand(fmt ++ ".bias", args, &.{out}, dt, 0.05);
}

/// An affine 4-bit projection (group 64), mlx-community's layout: random codes, scales and zero points that
/// centre them, the scales stored in `dt`.
fn quant(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64, dt: anytype) !void {
    const sc: f32 = 0.2 / @sqrt(@as(f32, @floatFromInt(in)));
    try t.words(fmt ++ ".weight", args, &.{ out, in / 8 });
    try t.fill(fmt ++ ".scales", args, &.{ out, in / 64 }, dt, sc);
    try t.fill(fmt ++ ".biases", args, &.{ out, in / 64 }, dt, -7.5 * sc);
}

fn conv(t: *T, comptime fmt: []const u8, args: anytype, shape: []const u64) !void {
    var fan: u64 = 1;
    for (shape[1..]) |d| fan *= d;
    try t.rand(fmt ++ ".weight", args, shape, .F32, 0.8 / @sqrt(@as(f32, @floatFromInt(fan))));
    try t.rand(fmt ++ ".bias", args, &.{shape[0]}, .F32, 0.05);
}

fn vaeRes(t: *T, comptime fmt: []const u8, args: anytype, c: u64) !void {
    try t.fill(fmt ++ ".norm1.gamma", args, &.{ c, 1, 1, 1 }, .F32, 1);
    try conv(t, fmt ++ ".conv1", args, &.{ c, c, 3, 3, 3 });
    try t.fill(fmt ++ ".norm2.gamma", args, &.{ c, 1, 1, 1 }, .F32, 1);
    try conv(t, fmt ++ ".conv2", args, &.{ c, c, 3, 3, 3 });
}

fn ditBlock(t: *T, comptime pfx: []const u8, i: usize, d: u64, hd: u64, ffn: u64, ada: u64, modulated: bool) !void {
    inline for (.{ "attention_norm1", "attention_norm2", "ffn_norm1", "ffn_norm2" }) |n| try t.fill(pfx ++ ".{d}." ++ n ++ ".weight", .{i}, &.{d}, .BF16, 1);
    inline for (.{ "norm_q", "norm_k" }) |n| try t.fill(pfx ++ ".{d}.attention." ++ n ++ ".weight", .{i}, &.{hd}, .BF16, 1);
    inline for (.{ "to_q", "to_k", "to_v", "to_out.0" }) |n| try dense(t, pfx ++ ".{d}.attention." ++ n, .{i}, d, d, false, .BF16);
    try dense(t, pfx ++ ".{d}.feed_forward.w1", .{i}, ffn, d, false, .BF16);
    try dense(t, pfx ++ ".{d}.feed_forward.w3", .{i}, ffn, d, false, .BF16);
    try dense(t, pfx ++ ".{d}.feed_forward.w2", .{i}, d, ffn, false, .BF16);
    if (modulated) try dense(t, pfx ++ ".{d}.adaLN_modulation.0", .{i}, 4 * d, ada, true, .BF16);
}

/// A seeded random pack in the release's tree: a 3-layer Bailing MoE (one dense layer, 8 4-bit experts in 2 groups,
/// a 2x2 query block), a 2-layer Qwen2 connector stored f32 as the release's is (its down projections 4-bit with f32
/// scales, as a pack quantized from it as stored), a 2-block DiT (dim 64, 2 heads, one refiner each) and the
/// Qwen-Image VAE at 16 channels decoding RGBA. A 512x512 canvas is 32x32 DiT tokens.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const h = 64;
    const hd = 64;
    var m = T.init(a, 41);
    try m.rand("model.model.word_embeddings.weight", .{}, &.{ vocab, h }, .BF16, 0.5);
    try m.fill("model.model.norm.weight", .{}, &.{h}, .BF16, 1);
    try m.rand("vision.blocks.0.attn.qkv.weight", .{}, &.{ 8, 8 }, .BF16, 0.1); // skipped at load
    for (0..3) |i| {
        try m.fill("model.model.layers.{d}.input_layernorm.weight", .{i}, &.{h}, .BF16, 1);
        try m.fill("model.model.layers.{d}.post_attention_layernorm.weight", .{i}, &.{h}, .BF16, 1);
        try m.fill("model.model.layers.{d}.attention.q_norm.weight", .{i}, &.{hd}, .BF16, 1);
        try m.fill("model.model.layers.{d}.attention.k_norm.weight", .{i}, &.{hd}, .BF16, 1);
        try dense(&m, "model.model.layers.{d}.attention.query_key_value", .{i}, (2 + 2) * hd, h, false, .BF16);
        try dense(&m, "model.model.layers.{d}.attention.dense", .{i}, h, 2 * hd, false, .BF16);
        if (i == 0) {
            inline for (.{ "gate_proj", "up_proj" }) |p| try dense(&m, "model.model.layers.{d}.mlp." ++ p, .{i}, 48, h, false, .BF16);
            try dense(&m, "model.model.layers.{d}.mlp.down_proj", .{i}, h, 48, false, .BF16);
            continue;
        }
        inline for (.{ "gate", "image_gate", "audio_gate" }) |gn| {
            try dense(&m, "model.model.layers.{d}.mlp." ++ gn, .{i}, 8, h, false, .BF16);
            try m.rand("model.model.layers.{d}.mlp." ++ gn ++ ".expert_bias", .{i}, &.{8}, .F32, 0.05);
        }
        // Quantized as mlx-community's packs are (MLX's CPU gather_mm takes f32 only).
        for (0..8) |e| {
            inline for (.{ "gate_proj", "up_proj" }) |p| try quant(&m, "model.model.layers.{d}.mlp.experts.{d}." ++ p, .{ i, e }, 64, h, .BF16);
            try quant(&m, "model.model.layers.{d}.mlp.experts.{d}.down_proj", .{ i, e }, h, 64, .BF16);
        }
        inline for (.{ "gate_proj", "up_proj" }) |p| try dense(&m, "model.model.layers.{d}.mlp.shared_experts." ++ p, .{i}, 16, h, false, .BF16);
        try dense(&m, "model.model.layers.{d}.mlp.shared_experts.down_proj", .{i}, h, 16, false, .BF16);
    }
    const mllm_cfg = std.fmt.comptimePrint(
        \\{{"model_type":"bailingmm_moe_v2_lite","llm_config":{{"hidden_size":{d},"num_hidden_layers":3,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":{d},
        \\ "partial_rotary_factor":1.0,"rope_theta":600000,"rms_norm_eps":1e-6,"num_experts":8,"num_experts_per_tok":2,"n_group":2,"topk_group":1,
        \\ "routed_scaling_factor":2.5,"first_k_dense_replace":1,"image_patch_token":{d},"vocab_size":{d}}}}}
    , .{ h, hd, patch_token, vocab });

    const ch = 32;
    var c = T.init(a, 42);
    try c.fill("model.norm.weight", .{}, &.{ch}, .F32, 1);
    try c.rand("model.embed_tokens.weight", .{}, &.{ vocab, ch }, .F32, 0.5);
    for (0..2) |i| {
        try c.fill("model.layers.{d}.input_layernorm.weight", .{i}, &.{ch}, .F32, 1);
        try c.fill("model.layers.{d}.post_attention_layernorm.weight", .{i}, &.{ch}, .F32, 1);
        try dense(&c, "model.layers.{d}.self_attn.q_proj", .{i}, 2 * 16, ch, true, .F32);
        try dense(&c, "model.layers.{d}.self_attn.k_proj", .{i}, 16, ch, true, .F32);
        try dense(&c, "model.layers.{d}.self_attn.v_proj", .{i}, 16, ch, true, .F32);
        try dense(&c, "model.layers.{d}.self_attn.o_proj", .{i}, ch, 2 * 16, false, .F32);
        inline for (.{ "gate_proj", "up_proj" }) |p| try dense(&c, "model.layers.{d}.mlp." ++ p, .{i}, 64, ch, false, .F32);
        // Quantized from the f32 release as stored: f32 scales.
        try quant(&c, "model.layers.{d}.mlp.down_proj", .{i}, ch, 64, .F32);
    }

    const d = 64;
    const cap = 24;
    var mlp = T.init(a, 43);
    try mlp.rand("query_tokens_dict.2x2", .{}, &.{ 4, h }, .BF16, 0.5);
    try dense(&mlp, "proj_in", .{}, ch, h, true, .BF16);
    try dense(&mlp, "proj_out", .{}, cap, ch, true, .BF16);
    try mlp.fill("proj_directvlm.0.weight", .{}, &.{3 * h}, .BF16, 1);
    try dense(&mlp, "proj_directvlm.1", .{}, d, 3 * h, true, .BF16);

    var dit = T.init(a, 44);
    try dense(&dit, "all_x_embedder.2-1", .{}, d, 64, true, .BF16);
    try dense(&dit, "t_embedder.mlp.0", .{}, 128, 256, true, .BF16);
    try dense(&dit, "t_embedder.mlp.2", .{}, d, 128, true, .BF16);
    try dit.fill("cap_embedder.0.weight", .{}, &.{cap}, .BF16, 1);
    try dense(&dit, "cap_embedder.1", .{}, d, cap, true, .BF16);
    try dense(&dit, "all_final_layer.2-1.adaLN_modulation.1", .{}, d, d, true, .BF16);
    try dense(&dit, "all_final_layer.2-1.linear", .{}, 64, d, true, .BF16);
    try ditBlock(&dit, "noise_refiner", 0, d, 32, 96, d, true);
    try ditBlock(&dit, "context_refiner", 0, d, 32, 96, d, false);
    for (0..2) |i| try ditBlock(&dit, "layers", i, d, 32, 96, d, true);

    const vc = 16;
    var vae = T.init(a, 45);
    try conv(&vae, "post_quant_conv", .{}, &.{ 16, 16, 1, 1, 1 });
    try conv(&vae, "decoder.conv_in", .{}, &.{ vc, 16, 3, 3, 3 });
    for (0..2) |i| try vaeRes(&vae, "decoder.mid_block.resnets.{d}", .{i}, vc);
    try vae.fill("decoder.mid_block.attentions.0.norm.gamma", .{}, &.{ vc, 1, 1 }, .F32, 1);
    try conv(&vae, "decoder.mid_block.attentions.0.to_qkv", .{}, &.{ 3 * vc, vc, 1, 1 });
    try conv(&vae, "decoder.mid_block.attentions.0.proj", .{}, &.{ vc, vc, 1, 1 });
    for (0..4) |bi| {
        for (0..3) |ri| try vaeRes(&vae, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri }, vc);
        if (bi < 3) try conv(&vae, "decoder.up_blocks.{d}.upsamplers.0.resample.1", .{bi}, &.{ vc, vc, 3, 3 });
    }
    try vae.fill("decoder.norm_out.gamma", .{}, &.{ vc, 1, 1, 1 }, .F32, 1);
    try conv(&vae, "decoder.conv_out", .{}, &.{ 4, vc, 3, 3, 3 });

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = "{\"model_type\":\"bailingmm_moe_v2_lite\"}" },
            .{ .name = "mllm/config.json", .bytes = mllm_cfg },
            .{ .name = "mllm/tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &specials) },
            .{ .name = "mllm/model.safetensors", .bytes = try m.bytes() },
            .{ .name = "connector/config.json", .bytes = "{\"hidden_size\":32,\"num_hidden_layers\":2,\"num_attention_heads\":2,\"num_key_value_heads\":1,\"rms_norm_eps\":1e-6,\"rope_theta\":1000000.0}" },
            .{ .name = "connector/model.safetensors", .bytes = try c.bytes() },
            .{ .name = "mlp/config.json", .bytes = "{\"use_identity_mlp\":true,\"use_vlm_directvlm_condition\":true,\"use_learnable_token_condition\":true,\"diffusion_c_input_dim\":24,\"text_encoder_norm\":false,\"connector_norm\":false,\"img_gen_scales\":[2],\"selected_hidden_states_layers\":[1,2,3],\"diffusion_inner_dim\":64}" },
            .{ .name = "mlp/model.safetensors", .bytes = try mlp.bytes() },
            .{ .name = "scheduler/scheduler_config.json", .bytes = "{\"shift\":6.0}" },
            .{ .name = "transformer/config.json", .bytes = "{\"_class_name\":\"DiffusionTransformer\",\"all_patch_size\":[2],\"all_f_patch_size\":[1],\"in_channels\":16,\"dim\":64,\"n_layers\":2,\"n_refiner_layers\":1,\"n_heads\":2,\"n_kv_heads\":2,\"norm_eps\":1e-5,\"qk_norm\":true,\"cap_feat_dim\":24,\"rope_theta\":256.0,\"t_scale\":1000.0,\"axes_dims\":[8,12,12],\"axes_lens\":[1024,64,64],\"alignment_padding_mode\":\"zero_masked\",\"multi_frame_output\":false}" },
            .{ .name = "transformer/diffusion_pytorch_model.safetensors", .bytes = try dit.bytes() },
            .{ .name = "vae/config.json", .bytes = "{\"_class_name\":\"AutoencoderKLQwenImage\",\"z_dim\":16,\"input_channels\":4,\"scaling_factor\":8.0064,\"shift_factor\":0.0}" },
            .{ .name = config.required_file, .bytes = try vae.bytes() },
        }),
        .body = "{\"prompt\":\"a red cat\",\"size\":\"512x512\",\"steps\":3,\"seed\":3}",
    };
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack;

test "ming_image: a job on the tiny pack is an RGBA PNG at its bucket; a negative prompt is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const png = (try pack.run(pack.fx.body)).ok;
    try testing.expectEqual(image_pixels.Size{ .w = 512, .h = 512 }, image_pixels.nativeSize(png).?);
    try testing.expectEqual(@as(u8, 6), png[25]); // IHDR colour type: truecolour with alpha
    const refused = (try pack.run("{\"prompt\":\"a\",\"negative_prompt\":\"b\"}")).refused;
    try testing.expect(refused.code == .unsupported);
}

test "ming_image: a prompt past the encoder's token cap is refused by name, never cut short" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const long = try arena.allocator().alloc(u8, llm.MAX_PROMPT_TOKENS);
    @memset(long, 'x');
    const body = try std.fmt.allocPrint(arena.allocator(), "{{\"prompt\":\"{s}\"}}", .{long});
    switch (try pack.run(body)) {
        .refused => |v| {
            try testing.expect(v.code == .unsupported);
            try testing.expect(std.mem.indexOf(u8, v.why.slice(), "4096 tokens") != null);
        },
        else => return error.TestUnexpectedResult,
    }
}

test "ming_image: an edit on a pack without the vision tower is refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    switch (try pack.run("{\"prompt\":\"x\",\"mode\":\"edit\",\"image\":\"x\"}")) {
        .refused => |v| try testing.expect(std.mem.indexOf(u8, v.why.slice(), "Ming-Image model with its vision tower") != null),
        else => return error.TestUnexpectedResult,
    }
}

test "ming_image: guidance past 1 runs the zeroed-condition branch and moves the image" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    var imgs: [2]mlx.mlx_array = undefined;
    for (&imgs, [_]f32{ 1.0, 3.0 }) |*img, gs| img.* = try e.m.generateImage("a red cat", 512, 512, 3, 2, .{ .guidance_scale = gs }, null);
    defer for (imgs) |x| {
        _ = mlx.mlx_array_free(x);
    };
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_array_equal(&diff, imgs[0], imgs[1], false, e.m.s));
    try mlx.check(mlx.mlx_array_eval(diff));
    var same: bool = true;
    try mlx.check(mlx.mlx_array_item_bool(&same, diff));
    try testing.expect(!same);
}

test "ming_image: both caption streams reach the DiT in bf16 from a connector quantized with f32 scales" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    var g = @import("../../mlx/ops.zig").Graph.init(testing.allocator, e.m.s);
    defer g.deinit();
    const ids = try e.m.enc.promptIds(testing.allocator, "a red cat", 0);
    defer testing.allocator.free(ids);
    const cond = try e.m.enc.encode(&g, ids, null);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(cond.learn));
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(cond.direct));
}

test "ming_image: a prompt's ids are the release's chat text around its system turn" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    const ids = try e.m.enc.promptIds(testing.allocator, "a red cat", 0);
    defer testing.allocator.free(ids);
    const text = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off<|role_end|><role>HUMAN</role>a red cat<|role_end|><role>ASSISTANT</role>";
    const want = try e.m.enc.tok.encode(testing.allocator, text);
    defer testing.allocator.free(want);
    try testing.expectEqualSlices(u32, want, ids);
}
