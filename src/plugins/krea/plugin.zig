//! Krea-2-Turbo on the media table: text to image and variation (SDEdit renoise at `strength`) over its Qwen3-VL
//! encoder (12 tapped layers), single-stream DiT and the Qwen-Image VAE. A job is one prompt; the result is a PNG.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const image_request = @import("../../sdk/media/image_request.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const LoraSet = @import("../../sdk/media/lora_set.zig").LoraSet;
const lora_mod = @import("../../sdk/weights/lora.zig");
const mlx = @import("../../mlx/ffi.zig");
const krea = @import("krea.zig");
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
    k: *krea.Engine,
    loras: LoraSet = .{},
    png: []u8 = &.{},

    fn dit(e: *Engine) Dit {
        return .{ .d = &e.k.dit };
    }
};

/// The DiT as `LoraSet` drives it.
const Dit = struct {
    d: *krea.Dit,
    pub fn attachLora(self: Dit, stack: *const lora_mod.Stack) u32 {
        return krea.attachLora(self.d, stack);
    }
    pub fn detachLora(self: Dit) void {
        krea.detachLora(self.d);
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const k = krea.Engine.loadOn(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch |err| {
        out.* = job.failure(err, "krea: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        k.deinit();
        out.* = Verdict.refuse(.failed, "krea: out of memory");
        return null;
    };
    e.* = .{ .k = k };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.loras.clear(e.dit());
    e.k.deinit();
    gpa.free(e.png);
    gpa.destroy(e);
}

/// Adapters name the DiT's own modules (`blocks.N.attn.wq`, the text-fusion blocks, the embedders).
fn setLoras(ep: *anyopaque, loras: abi.Span(abi.Lora)) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    return e.loras.set(gpa, loras.slice(), .krea2, e.dit());
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return Verdict.refuse(.unsupported, "invalid JSON body");
    defer body.deinit();
    var inputs: [1][]const u8 = undefined;
    const in = req.inputs.slice();
    const spec: image_request.Spec = .{ .img2img = e.k.vae_enc != null, .cond_weights = config.cond_weight_count };
    if (in.len > inputs.len) return Verdict.named(.unsupported, spec.too_many_refs);
    for (in, 0..) |s, i| inputs[i] = s.slice();
    const j = switch (image_request.parse(.{ .obj = body.parsed.value.object }, inputs[0..in.len], spec)) {
        .ok => |j| j,
        .bad => |v| return v,
    };
    const w = image_request.clamp16(j.req_w);
    const h = image_request.clamp16(j.req_h);
    const steps = if (j.steps == 0) config.default_steps else j.steps;
    var opts: krea.GenOpts = .{ .cond_gain = j.cond_gain, .cond_weights = j.condWeights() };
    // Variation shares the output's latent grid: the source is cover-cropped to it, never stretched.
    const init = if (j.mode == .variation)
        image_pixels.decodeVariation(gpa, j.image.?, h, w, j.strength) orelse return Verdict.refuse(.unsupported, "could not decode 'image' (PNG/JPEG supported)")
    else
        null;
    defer if (init) |p| {
        _ = mlx.mlx_array_free(p);
    };
    if (init) |p| {
        opts.init_image = p;
        opts.start_step = image_request.img2imgStartStep(steps, j.strength);
    }
    const sp = job.progressOf(progress);
    const img = e.k.generateImageOpts(gpa, j.prompt, w, h, j.seed, steps, opts, sp) catch |err|
        return if (err == error.Cancelled) job.client_left else job.failure(err, "generation failed");
    defer _ = mlx.mlx_array_free(img);
    // The image is lazy until here: a Metal failure in the denoise surfaces in the encode.
    const png = image_pixels.toPng(gpa, img, e.k.s) catch |err| return job.failure(err, "generation failed");
    gpa.free(e.png);
    e.png = png;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.png.len == 0) return Verdict.refuse(.failed, "krea: no finished job to encode");
    out.* = .of(e.png);
    return .ok;
}

// ── the tiny pack ──

const specials = [_]sdk.byte_tokenizer.Special{ .{ .id = 256, .content = "<|im_start|>" }, .{ .id = 257, .content = "<|im_end|>" } };

/// The prompt template pads with Qwen's <|endoftext|>, so the embedding table keeps the rows past it.
const te_vocab = 151644;

const T = sdk.tensors.Tensors;

/// A dense projection in PyTorch layout [out, in], optional bias.
fn dense(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64, bias: bool) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in }, .BF16, 0.8 / @sqrt(@as(f32, @floatFromInt(in))));
    if (bias) try t.rand(fmt ++ ".bias", args, &.{out}, .BF16, 0.05);
}

/// An affine 4-bit projection (group 64): random codes, scales and zero points that centre them.
fn quant(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64) !void {
    const sc: f32 = 0.2 / @sqrt(@as(f32, @floatFromInt(in)));
    try t.words(fmt ++ ".weight", args, &.{ out, in / 8 });
    try t.fill(fmt ++ ".scales", args, &.{ out, in / 64 }, .BF16, sc);
    try t.fill(fmt ++ ".biases", args, &.{ out, in / 64 }, .BF16, -7.5 * sc);
}

/// One attention + SwiGLU pair of the DiT's naming (`attn.wq..wo`, `qknorm`, `mlp.gate/up/down`).
fn attnMlp(t: *T, comptime pfx: []const u8, args: anytype, dim: u64, hd: u64, mlp: u64, quantized: bool) !void {
    inline for (.{ "wq", "wk", "wv", "gate", "wo" }) |n| {
        if (quantized) try quant(t, pfx ++ ".attn." ++ n, args, dim, dim) else try dense(t, pfx ++ ".attn." ++ n, args, dim, dim, false);
    }
    try t.fill(pfx ++ ".attn.qknorm.qnorm.scale", args, &.{hd}, .BF16, 0);
    try t.fill(pfx ++ ".attn.qknorm.knorm.scale", args, &.{hd}, .BF16, 0);
    try t.fill(pfx ++ ".prenorm.scale", args, &.{dim}, .BF16, 0);
    try t.fill(pfx ++ ".postnorm.scale", args, &.{dim}, .BF16, 0);
    inline for (.{ "gate", "up" }) |n| {
        if (quantized) try quant(t, pfx ++ ".mlp." ++ n, args, mlp, dim) else try dense(t, pfx ++ ".mlp." ++ n, args, mlp, dim, false);
    }
    if (quantized) try quant(t, pfx ++ ".mlp.down", args, dim, mlp) else try dense(t, pfx ++ ".mlp.down", args, dim, mlp, false);
}

/// A causal conv3d in PyTorch layout [out, in, kt, kh, kw].
fn conv3(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64, k: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in, k, k, k }, .F32, 0.8 / @sqrt(@as(f32, @floatFromInt(k * k * in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .F32, 0.05);
}

fn conv2(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64, k: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in, k, k }, .F32, 0.8 / @sqrt(@as(f32, @floatFromInt(k * k * in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .F32, 0.05);
}

fn res(t: *T, comptime fmt: []const u8, args: anytype, c: u64) !void {
    try t.fill(fmt ++ ".norm1.gamma", args, &.{ c, 1, 1, 1 }, .F32, 1);
    try conv3(t, fmt ++ ".conv1", args, c, c, 3);
    try t.fill(fmt ++ ".norm2.gamma", args, &.{ c, 1, 1, 1 }, .F32, 1);
    try conv3(t, fmt ++ ".conv2", args, c, c, 3);
}

fn mid(t: *T, comptime side: []const u8, c: u64) !void {
    for (0..2) |i| try res(t, side ++ ".mid_block.resnets.{d}", .{i}, c);
    try t.fill(side ++ ".mid_block.attentions.0.norm.gamma", .{}, &.{ c, 1, 1 }, .F32, 1);
    try conv2(t, side ++ ".mid_block.attentions.0.to_qkv", .{}, 3 * c, c, 1);
    try conv2(t, side ++ ".mid_block.attentions.0.proj", .{}, c, c, 1);
    try t.fill(side ++ ".norm_out.gamma", .{}, &.{ c, 1, 1, 1 }, .F32, 1);
}

/// A seeded random pack in the release's layout: a 36-layer Qwen3-VL text stack (hidden 64, so all 12 taps
/// exist), a DiT of two quantized blocks (features 128, one 128-wide head) with dense text fusion, and the
/// Qwen-Image VAE at 32 channels with its encoder. A 256x256 canvas is 16x16 tokens beside the prompt rows.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const te_h = 64;
    var te = T.init(a, 31);
    try te.rand("embed_tokens.weight", .{}, &.{ te_vocab, te_h }, .BF16, 0.5);
    for (0..36) |i| {
        try te.fill("layers.{d}.input_layernorm.weight", .{i}, &.{te_h}, .BF16, 1);
        try te.fill("layers.{d}.post_attention_layernorm.weight", .{i}, &.{te_h}, .BF16, 1);
        try te.fill("layers.{d}.self_attn.q_norm.weight", .{i}, &.{32}, .BF16, 1);
        try te.fill("layers.{d}.self_attn.k_norm.weight", .{i}, &.{32}, .BF16, 1);
        try dense(&te, "layers.{d}.self_attn.q_proj", .{i}, 64, te_h, false);
        try dense(&te, "layers.{d}.self_attn.k_proj", .{i}, 32, te_h, false);
        try dense(&te, "layers.{d}.self_attn.v_proj", .{i}, 32, te_h, false);
        try dense(&te, "layers.{d}.self_attn.o_proj", .{i}, te_h, 64, false);
        try dense(&te, "layers.{d}.mlp.gate_proj", .{i}, 128, te_h, false);
        try dense(&te, "layers.{d}.mlp.up_proj", .{i}, 128, te_h, false);
        try dense(&te, "layers.{d}.mlp.down_proj", .{i}, te_h, 128, false);
    }

    const d = 128;
    var dit = T.init(a, 32);
    try dense(&dit, "first", .{}, d, 64, true);
    try dense(&dit, "tmlp.0", .{}, d, 256, true);
    try dense(&dit, "tmlp.2", .{}, d, d, true);
    try dense(&dit, "tproj.1", .{}, 6 * d, d, true);
    for (0..2) |i| {
        try dit.rand("blocks.{d}.mod.lin", .{i}, &.{6 * d}, .BF16, 0.1);
        try attnMlp(&dit, "blocks.{d}", .{i}, d, 128, 384, true);
    }
    for (0..2) |i| try attnMlp(&dit, "txtfusion.layerwise_blocks.{d}", .{i}, te_h, 32, 256, false);
    for (0..2) |i| try attnMlp(&dit, "txtfusion.refiner_blocks.{d}", .{i}, te_h, 32, 256, false);
    try dense(&dit, "txtfusion.projector", .{}, 1, 12, false);
    try dit.fill("txtmlp.0.scale", .{}, &.{te_h}, .BF16, 0);
    try dense(&dit, "txtmlp.1", .{}, d, te_h, true);
    try dense(&dit, "txtmlp.3", .{}, d, d, true);
    try dit.fill("last.norm.scale", .{}, &.{d}, .BF16, 0);
    try dense(&dit, "last.linear", .{}, 64, d, true);
    try dit.rand("last.modulation.lin", .{}, &.{2 * d}, .BF16, 0.1);

    const c = 32;
    var vae = T.init(a, 33);
    try conv3(&vae, "post_quant_conv", .{}, 16, 16, 1);
    try conv3(&vae, "decoder.conv_in", .{}, c, 16, 3);
    try mid(&vae, "decoder", c);
    for (0..4) |bi| {
        for (0..3) |ri| try res(&vae, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri }, c);
        if (bi < 3) try conv2(&vae, "decoder.up_blocks.{d}.upsamplers.0.resample.1", .{bi}, c, c, 3);
    }
    try conv3(&vae, "decoder.conv_out", .{}, 3, c, 3);
    try conv3(&vae, "encoder.conv_in", .{}, c, 3, 3);
    for (0..11) |bi| {
        if (bi == 2 or bi == 5 or bi == 8) try conv2(&vae, "encoder.down_blocks.{d}.resample.1", .{bi}, c, c, 3) else try res(&vae, "encoder.down_blocks.{d}", .{bi}, c);
    }
    try mid(&vae, "encoder", c);
    try conv3(&vae, "encoder.conv_out", .{}, 32, c, 3);
    try conv3(&vae, "quant_conv", .{}, 32, 32, 1);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = "{\"model_type\":\"krea2_turbo\"}" },
            .{ .name = "tokenizer/tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &specials) },
            .{ .name = "text_encoder/model.safetensors", .bytes = try te.bytes() },
            .{ .name = "transformer_mixed_4_8.safetensors", .bytes = try dit.bytes() },
            .{ .name = "vae/model.safetensors", .bytes = try vae.bytes() },
        }),
        .body = "{\"prompt\":\"a red cat\",\"size\":\"256x256\",\"steps\":2,\"seed\":3}",
    };
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack;
const golden = @import("../../sdk/testing/golden.zig");
const png_mod = @import("../../sdk/media/png.zig");

fn gradientPng(a: std.mem.Allocator) ![]u8 {
    const w = 64;
    const h = 48;
    var rgb: [w * h * 3]u8 = undefined;
    for (0..h) |y| for (0..w) |x| {
        const o = (y * w + x) * 3;
        rgb[o] = @intCast(x * 4);
        rgb[o + 1] = @intCast(y * 5);
        rgb[o + 2] = @intCast((x + y) * 2);
    };
    return png_mod.encodeRgb(a, &rgb, w, h);
}

test "krea: a job on the tiny pack is a 256x256 PNG; an edit and a wrong tap count refuse" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const png = (try pack.run(pack.fx.body)).ok;
    try testing.expectEqual(image_pixels.Size{ .w = 256, .h = 256 }, image_pixels.nativeSize(png).?);
    try testing.expect((try pack.run("{\"prompt\":\"a\",\"mode\":\"edit\",\"image\":\"x\"}")).refused.code == .unsupported);
    try testing.expect((try pack.run("{\"prompt\":\"a\",\"cond_weights\":[1,2,3]}")).refused.code == .unsupported);
}

test "krea: the tiny pack's pipeline (rebalanced text, a variation) is bit-equal to its golden run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var pack = try MediaPack(@This()).open(a);
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    const weights = [_]f32{ 1, 0.5, 1.5, 1, 1, 0.8, 1.2, 1, 1, 1, 0.9, 1.1 };
    const t2i = try e.k.generateImageOpts(a, "a red cat", 256, 256, 3, 2, .{ .cond_gain = 1.25, .cond_weights = &weights }, null);
    defer _ = mlx.mlx_array_free(t2i);
    const init = image_pixels.decodeCover(a, try gradientPng(a), 256, 256).?;
    defer _ = mlx.mlx_array_free(init);
    const variation = try e.k.generateImageOpts(a, "a cat", 256, 256, 7, 3, .{ .init_image = init, .start_step = 1 }, null);
    defer _ = mlx.mlx_array_free(variation);
    try golden.expect("krea_pipeline", &.{ .{ .name = "t2i", .array = t2i }, .{ .name = "variation", .array = variation } });
}
