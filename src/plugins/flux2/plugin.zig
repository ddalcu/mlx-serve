//! FLUX.2 klein (4B, 9B, and the undistilled base 9B with real CFG) on the media table: text to image, variation
//! (SDEdit renoise at `strength`) and instruction edits with up to four in-context references, each at its own grid.
//! A job is one prompt; the result is a PNG.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const image_request = @import("../../sdk/media/image_request.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const log = @import("../../sdk/util/log.zig");
const LoraSet = @import("../../sdk/media/lora_set.zig").LoraSet;
const lora_mod = @import("../../sdk/weights/lora.zig");
const mlx = @import("../../mlx/ffi.zig");
const flux = @import("flux.zig");
const engine_mod = @import("engine.zig");
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
    f: engine_mod.Engine,
    loras: LoraSet = .{},
    png: []u8 = &.{},

    /// The DiT as `LoraSet` drives it.
    fn dit(e: *Engine) Dit {
        return .{ .d = &e.f.dit };
    }
};

const Dit = struct {
    d: *flux.Dit,
    pub fn attachLora(self: Dit, stack: *const lora_mod.Stack) u32 {
        return flux.attachLora(self.d, stack);
    }
    pub fn detachLora(self: Dit) void {
        flux.detachLora(self.d);
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const e = gpa.create(Engine) catch {
        out.* = Verdict.refuse(.failed, "flux2: out of memory");
        return null;
    };
    const f = engine_mod.Engine.loadOn(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }, if (ctx.machine) |m| m.low_mem else false) catch |err| {
        gpa.destroy(e);
        out.* = job.failure(err, "flux2: the pack does not load");
        return null;
    };
    e.* = .{ .f = f };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.loras.clear(e.dit());
    e.f.deinit();
    gpa.free(e.png);
    gpa.destroy(e);
}

/// Adapters in diffusers, BFL or kohya spelling resolve onto the DiT's own modules.
fn setLoras(ep: *anyopaque, loras: abi.Span(abi.Lora)) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    return e.loras.set(gpa, loras.slice(), .flux2, e.dit());
}

const decode_refusal = Verdict.refuse(.unsupported, "could not decode 'image' (PNG/JPEG supported)");
const ref_decode_refusal = Verdict.refuse(.unsupported, "could not decode a 'ref_images' entry (PNG/JPEG supported)");

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return Verdict.refuse(.unsupported, "invalid JSON body");
    defer body.deinit();
    var inputs: [config.edit_ref_cap][]const u8 = undefined;
    const in = req.inputs.slice();
    const has_enc = e.f.vae_enc != null;
    const spec: image_request.Spec = .{
        .img2img = has_enc,
        .edit = has_enc,
        .guidance = true,
        .edit_ref_cap = config.edit_ref_cap,
        .cond_weights = config.cond_weight_count,
    };
    if (in.len > inputs.len) return Verdict.named(.unsupported, spec.too_many_refs);
    for (in, 0..) |s, i| inputs[i] = s.slice();
    const j = switch (image_request.parse(.{ .obj = body.parsed.value.object }, inputs[0..in.len], spec)) {
        .ok => |j| j,
        .bad => |v| return v,
    };
    var w = image_request.clamp32(j.req_w);
    var h = image_request.clamp32(j.req_h);

    var pix: [config.edit_ref_cap]mlx.mlx_array = undefined;
    var pix_n: usize = 0;
    defer for (pix[0..pix_n]) |p| {
        _ = mlx.mlx_array_free(p);
    };
    var opts: engine_mod.Opts = .{
        .strength = j.strength,
        .cond_gain = j.cond_gain,
        .cond_weights = j.condWeights(),
        .guidance_scale = j.guidance_scale,
        .negative_prompt = j.negative_prompt,
    };
    if (opts.cond_weights) |cw| log.info("[image] rebalance: gain={d:.2} weights={d}\n", .{ j.cond_gain, cw.len });
    switch (j.mode) {
        .generate => {},
        // Variation shares the output's latent grid: the source is cover-cropped to it, never stretched.
        .variation => {
            pix[0] = image_pixels.decodeVariation(gpa, j.image.?, h, w, j.strength) orelse return decode_refusal;
            pix_n = 1;
            opts.init_image = pix[0];
        },
        // Each reference keeps its own aspect at ~1 MP on its own grid. No size = the primary's resolution; a size
        // is honoured as asked (within the cap).
        .edit => {
            const nat = image_pixels.nativeSize(j.image.?) orelse return decode_refusal;
            for (0..1 + j.refs.len) |i| {
                const bytes = if (i == 0) j.image.? else j.refs[i - 1];
                const n = image_pixels.nativeSize(bytes) orelse return if (i == 0) decode_refusal else ref_decode_refusal;
                const rd = image_request.fitRefDims(n.w, n.h);
                pix[pix_n] = image_pixels.decodeCover(gpa, bytes, rd.h, rd.w) orelse return if (i == 0) decode_refusal else ref_decode_refusal;
                pix_n += 1;
                if (i == 0)
                    log.info("[image] edit: reference {d}x{d} -> {d}x{d} (in-context conditioning)\n", .{ n.w, n.h, rd.w, rd.h })
                else
                    log.info("[image] edit ref {d}: {d}x{d} -> {d}x{d}\n", .{ i + 1, n.w, n.h, rd.w, rd.h });
            }
            const fit = if (j.size_given)
                image_request.fitWithinCap(w, h, config.max_dim)
            else
                image_request.resolveEditTargetSize(nat.w, nat.h, nat.w, nat.h, config.max_dim);
            const tw = image_request.clamp32(fit.w);
            const th = image_request.clamp32(fit.h);
            if (tw != w or th != h)
                log.info("[image] edit: target {d}x{d} -> {d}x{d} (primary reference is {d}x{d}, size {s})\n", .{ w, h, tw, th, nat.w, nat.h, if (j.size_given) "requested" else "matched to source" });
            w = tw;
            h = th;
            opts.edit_images = pix[0..pix_n];
        },
    }
    const steps = if (j.steps == 0) config.default_steps else j.steps;
    const sp = job.progressOf(progress);
    const img = e.f.generateImage(gpa, j.prompt, w, h, j.seed, steps, opts, sp) catch |err|
        return if (err == error.Cancelled) job.client_left else job.failure(err, "generation failed");
    defer _ = mlx.mlx_array_free(img);
    // The image is lazy until here: a Metal failure in the denoise surfaces in the encode.
    const png = image_pixels.toPng(gpa, img, e.f.s) catch |err| return job.failure(err, "generation failed");
    gpa.free(e.png);
    e.png = png;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.png.len == 0) return Verdict.refuse(.failed, "flux2: no finished job to encode");
    out.* = .of(e.png);
    return .ok;
}

// ── the tiny pack ──

const specials = [_]sdk.byte_tokenizer.Special{ .{ .id = 256, .content = "<|im_start|>" }, .{ .id = 257, .content = "<|im_end|>" }, .{ .id = 258, .content = "<think>" }, .{ .id = 259, .content = "</think>" } };

/// The pad id the prompt template fills to its 512 tokens indexes the embedding table, so the table keeps Qwen3's
/// rows past it.
const te_vocab = 151644;

/// An affine 4-bit linear (group 64) in mflux's layout: random codes, scales and zero points that centre them.
fn qlinear(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, out: u64, in: u64, bias: bool) !void {
    const sc: f32 = 0.2 / @sqrt(@as(f32, @floatFromInt(in)));
    try t.words(fmt ++ ".weight", args, &.{ out, in / 8 });
    try t.fill(fmt ++ ".scales", args, &.{ out, in / 64 }, .BF16, sc);
    try t.fill(fmt ++ ".biases", args, &.{ out, in / 64 }, .BF16, -7.5 * sc);
    if (bias) try t.rand(fmt ++ ".bias", args, &.{out}, .BF16, 0.05);
}

fn conv(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, out: u64, k: u64, in: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, k, k, in }, .BF16, 0.8 / @sqrt(@as(f32, @floatFromInt(k * k * in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .BF16, 0.05);
}

fn resnet(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, c: u64) !void {
    try t.fill(fmt ++ ".norm1.weight", args, &.{c}, .BF16, 1);
    try t.fill(fmt ++ ".norm1.bias", args, &.{c}, .BF16, 0);
    try conv(t, fmt ++ ".conv1", args, c, 3, c);
    try t.fill(fmt ++ ".norm2.weight", args, &.{c}, .BF16, 1);
    try t.fill(fmt ++ ".norm2.bias", args, &.{c}, .BF16, 0);
    try conv(t, fmt ++ ".conv2", args, c, 3, c);
}

fn midBlock(t: *sdk.tensors.Tensors, comptime side: []const u8, c: u64) !void {
    for (0..2) |i| try resnet(t, side ++ ".mid_block.resnets.{d}", .{i}, c);
    try t.fill(side ++ ".mid_block.attentions.0.group_norm.weight", .{}, &.{c}, .BF16, 1);
    try t.fill(side ++ ".mid_block.attentions.0.group_norm.bias", .{}, &.{c}, .BF16, 0);
    inline for (.{ "to_q", "to_k", "to_v", "to_out" }) |n| try qlinear(t, side ++ ".mid_block.attentions.0." ++ n, .{}, c, c, true);
    try t.fill(side ++ ".conv_norm_out.weight", .{}, &.{c}, .BF16, 1);
    try t.fill(side ++ ".conv_norm_out.bias", .{}, &.{c}, .BF16, 0);
}

/// A seeded random pack in mflux's klein layout: a 27-layer Qwen3 encoder (hidden 64, so layers 9/18/27 tap), a DiT
/// of one double and one single block (inner 128, one 128-wide head), and the VAE at 64 channels with its encoder.
/// A 256x256 canvas is 16x16 latent tokens beside the 512 text tokens.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    const te_h = 64;
    const hd = 32;
    var te = T.init(a, 21);
    try qlinear(&te, "embed_tokens", .{}, te_vocab, te_h, false);
    try te.fill("norm.weight", .{}, &.{te_h}, .BF16, 1);
    for (0..27) |i| {
        try te.fill("layers.{d}.input_layernorm.weight", .{i}, &.{te_h}, .BF16, 1);
        try te.fill("layers.{d}.post_attention_layernorm.weight", .{i}, &.{te_h}, .BF16, 1);
        try te.fill("layers.{d}.self_attn.q_norm.weight", .{i}, &.{hd}, .BF16, 1);
        try te.fill("layers.{d}.self_attn.k_norm.weight", .{i}, &.{hd}, .BF16, 1);
        try qlinear(&te, "layers.{d}.self_attn.q_proj", .{i}, 2 * hd, te_h, false);
        try qlinear(&te, "layers.{d}.self_attn.k_proj", .{i}, hd, te_h, false);
        try qlinear(&te, "layers.{d}.self_attn.v_proj", .{i}, hd, te_h, false);
        try qlinear(&te, "layers.{d}.self_attn.o_proj", .{i}, te_h, 2 * hd, false);
        try qlinear(&te, "layers.{d}.mlp.gate_proj", .{i}, 128, te_h, false);
        try qlinear(&te, "layers.{d}.mlp.up_proj", .{i}, 128, te_h, false);
        try qlinear(&te, "layers.{d}.mlp.down_proj", .{i}, te_h, 128, false);
    }

    const d = 128;
    const ff = 3 * d;
    var dit = T.init(a, 22);
    try qlinear(&dit, "x_embedder", .{}, d, 128, false);
    try qlinear(&dit, "context_embedder", .{}, d, 3 * te_h, false);
    try qlinear(&dit, "time_guidance_embed.linear_1", .{}, d, 256, false);
    try qlinear(&dit, "time_guidance_embed.linear_2", .{}, d, d, false);
    try qlinear(&dit, "double_stream_modulation_img.linear", .{}, 6 * d, d, false);
    try qlinear(&dit, "double_stream_modulation_txt.linear", .{}, 6 * d, d, false);
    try qlinear(&dit, "single_stream_modulation.linear", .{}, 3 * d, d, false);
    try qlinear(&dit, "norm_out.linear", .{}, 2 * d, d, false);
    try qlinear(&dit, "proj_out", .{}, 128, d, false);
    inline for (.{ "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out" }) |n|
        try qlinear(&dit, "transformer_blocks.0." ++ n, .{}, d, d, false);
    inline for (.{ "norm_q", "norm_k", "norm_added_q", "norm_added_k" }) |n| try dit.fill("transformer_blocks.0.attn." ++ n ++ ".weight", .{}, &.{d}, .BF16, 1);
    inline for (.{ "ff", "ff_context" }) |n| {
        try qlinear(&dit, "transformer_blocks.0." ++ n ++ ".linear_in", .{}, 2 * ff, d, false);
        try qlinear(&dit, "transformer_blocks.0." ++ n ++ ".linear_out", .{}, d, ff, false);
    }
    try qlinear(&dit, "single_transformer_blocks.0.attn.to_qkv_mlp_proj", .{}, 3 * d + 2 * ff, d, false);
    try qlinear(&dit, "single_transformer_blocks.0.attn.to_out", .{}, d, d + ff, false);
    inline for (.{ "norm_q", "norm_k" }) |n| try dit.fill("single_transformer_blocks.0.attn." ++ n ++ ".weight", .{}, &.{d}, .BF16, 1);

    const c = 64;
    var vae = T.init(a, 23);
    try vae.rand("bn.running_mean", .{}, &.{128}, .F32, 0.1);
    try vae.fill("bn.running_var", .{}, &.{128}, .F32, 1);
    try conv(&vae, "post_quant_conv", .{}, 32, 1, 32);
    try conv(&vae, "decoder.conv_in", .{}, c, 3, 32);
    try midBlock(&vae, "decoder", c);
    for (0..4) |bi| {
        for (0..3) |ri| try resnet(&vae, "decoder.up_blocks.{d}.resnets.{d}", .{ bi, ri }, c);
        if (bi < 3) try conv(&vae, "decoder.up_blocks.{d}.upsamplers.0.conv", .{bi}, c, 3, c);
    }
    try conv(&vae, "decoder.conv_out", .{}, 3, 3, c);
    try conv(&vae, "encoder.conv_in", .{}, c, 3, 3);
    for (0..4) |bi| {
        for (0..2) |ri| try resnet(&vae, "encoder.down_blocks.{d}.resnets.{d}", .{ bi, ri }, c);
        if (bi < 3) try conv(&vae, "encoder.down_blocks.{d}.downsamplers.0.conv", .{bi}, c, 3, c);
    }
    try midBlock(&vae, "encoder", c);
    try conv(&vae, "encoder.conv_out", .{}, 64, 3, c);
    try conv(&vae, "quant_conv", .{}, 64, 1, 64);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = "{\"model_type\":\"flux2-klein-4b\"}" },
            .{ .name = "tokenizer/tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &specials) },
            .{ .name = "text_encoder/model.safetensors", .bytes = try te.bytes() },
            .{ .name = "transformer/model.safetensors", .bytes = try dit.bytes() },
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

/// A 64x48 gradient PNG (a reference of its own aspect).
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

test "flux2: a job on the tiny pack is a 256x256 PNG; an edit keeps the reference's aspect; bad fields refuse" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const png = (try pack.run(pack.fx.body)).ok;
    try testing.expectEqual(image_pixels.Size{ .w = 256, .h = 256 }, image_pixels.nativeSize(png).?);

    const ref = try gradientPng(arena.allocator());
    const in = [_]abi.Str{.of(ref)};
    const req: abi.Request = .{ .body = .of("{\"prompt\":\"blue\",\"mode\":\"edit\",\"image\":\"x\",\"size\":\"320x256\",\"steps\":1}"), .inputs = .of(&in) };
    const quiet: abi.Progress = .{ .ctx = null, .report = struct {
        fn f(_: ?*anyopaque, _: abi.Str, _: u32, _: u32) callconv(.c) void {}
    }.f, .boundary = struct {
        fn f(_: ?*anyopaque) callconv(.c) bool {
            return false;
        }
    }.f };
    try testing.expect(table.step(pack.engine, &req, &quiet).isOk());
    var out: abi.Str = .{};
    try testing.expect(table.encode(pack.engine, &out).isOk());
    // An explicit size is honoured as asked, whatever the reference's aspect.
    try testing.expectEqual(image_pixels.Size{ .w = 320, .h = 256 }, image_pixels.nativeSize(out.slice()).?);

    const bad = (try pack.run("{\"prompt\":\"a\",\"cond_weights\":[1,2]}")).refused;
    try testing.expect(bad.code == .unsupported);
}

test "flux2: the tiny pack's pipeline (CFG, cond rebalance, an edit) is bit-equal to its golden run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var pack = try MediaPack(@This()).open(a);
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    const weights = [_]f32{ 1.0, 0.5, 1.5 };
    const t2i = try e.f.generateImage(a, "a red cat", 256, 256, 3, 2, .{ .guidance_scale = 2.5, .negative_prompt = "blurry", .cond_gain = 1.25, .cond_weights = &weights }, null);
    defer _ = mlx.mlx_array_free(t2i);
    const ref = image_pixels.decodeCover(a, try gradientPng(a), 96, 128).?;
    defer _ = mlx.mlx_array_free(ref);
    const edit = try e.f.generateImage(a, "make it blue", 256, 256, 5, 2, .{ .edit_images = &.{ref} }, null);
    defer _ = mlx.mlx_array_free(edit);
    const init = image_pixels.decodeCover(a, try gradientPng(a), 256, 256).?;
    defer _ = mlx.mlx_array_free(init);
    const variation = try e.f.generateImage(a, "a cat", 256, 256, 7, 3, .{ .init_image = init, .strength = 0.7 }, null);
    defer _ = mlx.mlx_array_free(variation);
    try mlx.check(mlx.mlx_array_eval(t2i));
    try golden.expect("flux2_pipeline", &.{ .{ .name = "t2i", .array = t2i }, .{ .name = "edit", .array = edit }, .{ .name = "variation", .array = variation } });
}
