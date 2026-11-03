//! Qwen-Image-2.1 on the media table: text to image and variation over its Qwen3-VL encoder, block-causal
//! single-stream DiT and 64-channel /16 VAE, real CFG on request, RGBA on request; with the tower in its
//! text_encoder/ (the edit pack), instruction edits over up to ten references (raw upload bytes). The result is a PNG.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const image_request = @import("../../sdk/media/image_request.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const LoraSet = @import("../../sdk/media/lora_set.zig").LoraSet;
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const qi = @import("qwen_image.zig");
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
    m: *qi.Engine,
    loras: LoraSet = .{},
    png: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const staged = config.staged(ctx.dir, ctx.machine orelse &abi.Machine{});
    const m = qi.Engine.loadOn(io, gpa, ctx.model_dir.slice(), staged, .{ .ctx = ctx.stream.ctx }) catch |err| {
        out.* = job.failure(err, "qwen_image: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        m.deinit();
        out.* = Verdict.refuse(.failed, "qwen_image: out of memory");
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

fn setLoras(ep: *anyopaque, loras: abi.Span(abi.Lora)) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    return e.loras.set(gpa, loras.slice(), .qwen_image21, &e.m.dit);
}

const decode_refusal = Verdict.refuse(.unsupported, "could not decode 'image' (PNG/JPEG/WebP supported)");
const ref_decode_refusal = Verdict.refuse(.unsupported, "could not decode a 'ref_images' entry (PNG/JPEG/WebP supported)");

/// What the GPU can still take: Metal's working set less live MLX memory.
fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return Verdict.refuse(.unsupported, "invalid JSON body");
    defer body.deinit();
    var inputs: [config.edit_ref_cap][]const u8 = undefined;
    const in = req.inputs.slice();
    const spec: image_request.Spec = .{
        .img2img = true,
        .edit = e.m.supportsEdit(),
        .guidance = true,
        .edit_ref_cap = config.edit_ref_cap,
        .transparent = true,
        .ref_resolution = true,
        .too_many_refs = "too many reference images ('ref_images' takes at most 9 beside 'image')",
    };
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
    var gen: qi.GenOpts = .{ .guidance_scale = j.guidance_scale, .negative_prompt = j.negative_prompt, .transparent = j.transparent };
    const img = switch (j.mode) {
        .generate => blk: {
            if (canvasRefusal(req, w, h)) |v| return v;
            break :blk e.m.generateImage(gpa, j.prompt, w, h, j.seed, steps, gen, sp);
        },
        // Variation shares the output's latent grid: the source is cover-cropped to it, never stretched.
        .variation => blk: {
            const pix = image_pixels.decodeVariation(gpa, j.image.?, h, w, j.strength) orelse return Verdict.refuse(.unsupported, "could not decode 'image' (PNG/JPEG supported)");
            defer _ = mlx.mlx_array_free(pix);
            if (canvasRefusal(req, w, h)) |v| return v;
            gen.init_image = pix;
            gen.start_step = image_request.img2imgStartStep(steps, j.strength);
            break :blk e.m.generateImage(gpa, j.prompt, w, h, j.seed, steps, gen, sp);
        },
        .edit => return edit(e, &j, sp),
    } catch |err| return if (err == error.Cancelled) job.client_left else job.failure(err, "generation failed");
    return finish(e, img);
}

/// A canvas past the default the load reserved can run past the working set, and that Metal OOM mid-denoise is
/// uncatchable: refused by name.
fn canvasRefusal(req: *const abi.Request, w: u32, h: u32) ?Verdict {
    const need = config.genTransientBytes(w, h);
    const room = job.gpuHeadroom();
    if (need > room) log.info("[image] bill refused: {d}x{d} needs {d} MB, headroom {d} MB\n", .{ w, h, need >> 20, room >> 20 });
    return image_request.canvasRefusal(req, w, h, need, room);
}

/// The edit output follows the LAST condition image; its working set (references + target) is checked against the
/// GPU's live headroom, and the prefix cache rides only where it fits too.
fn edit(e: *Engine, j: *const image_request.Job, sp: anytype) Verdict {
    var refs: [config.edit_ref_cap][]const u8 = undefined;
    refs[0] = j.image.?;
    for (j.refs, 1..) |r, i| refs[i] = r;
    const n = 1 + j.refs.len;
    var last: image_pixels.Size = undefined;
    for (refs[0..n], 0..) |r, i| last = image_pixels.nativeSize(r) orelse return if (i == 0) decode_refusal else ref_decode_refusal;
    image_pixels.logByteRefs(refs[0..n]);
    const dims = config.editDims(if (j.size_given) .{ .w = j.req_w, .h = j.req_h } else null, last.w, last.h);
    if (dims.w != j.req_w or dims.h != j.req_h)
        log.info("[image] edit: target {d}x{d} -> {d}x{d} (last reference is {d}x{d}, size {s})\n", .{ j.req_w, j.req_h, dims.w, dims.h, last.w, last.h, if (j.size_given) "requested" else "matched to source" });
    const cfg = e.m.dit_cfg;
    const base = config.editTransientBytes(@intCast(n), j.ref_resolution, dims.w, dims.h);
    const prefix = config.editPrefixBytes(@intCast(n), j.ref_resolution, cfg.layers, cfg.hidden(), if (j.guidance_scale != 1) 2 else 1);
    const steps = qi.resolveSteps(null, j.steps);
    const room = job.gpuHeadroom();
    const cache = config.editPlan(steps, base, prefix, room) orelse {
        log.info("[image] edit bill refused: {d} refs at refres {d} + {d}x{d} target needs {d} MB over the reserve, headroom {d} MB\n", .{
            n, j.ref_resolution, dims.w, dims.h, (base -| config.EDIT_TRANSIENT_BYTES) >> 20, room >> 20,
        });
        return Verdict.refuse(.unsupported, "this edit's working set (references + target) needs more GPU memory than is free — lower 'ref_resolution' or the reference count, or shrink 'size'");
    };
    if (steps > 1 and !cache) log.info("[image] edit prefix cache does not fit; using uncached forward (cached bill {d} MB over reserve, headroom {d} MB)\n", .{
        ((base +| prefix) -| config.EDIT_TRANSIENT_BYTES) >> 20, room >> 20,
    });
    const img = e.m.editImage(gpa, j.prompt, refs[0..n], dims.w, dims.h, j.seed, steps, .{
        .guidance_scale = j.guidance_scale,
        .negative_prompt = j.negative_prompt,
        .ref_resolution = j.ref_resolution,
        .transparent = j.transparent,
        .prefix_cache = cache,
    }, sp) catch |err| return if (err == error.Cancelled) job.client_left else job.failure(err, "generation failed");
    return finish(e, img);
}

fn finish(e: *Engine, img: mlx.mlx_array) Verdict {
    defer _ = mlx.mlx_array_free(img);
    // The image is lazy until here: a Metal failure in the denoise surfaces in the encode.
    const png = image_pixels.toPng(gpa, img, e.m.s) catch |err| return job.failure(err, "generation failed");
    gpa.free(e.png);
    e.png = png;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.png.len == 0) return Verdict.refuse(.failed, "qwen_image: no finished job to encode");
    out.* = .of(e.png);
    return .ok;
}

// ── the tiny pack ──

const specials = [_]sdk.byte_tokenizer.Special{ .{ .id = 256, .content = "<|im_start|>" }, .{ .id = 257, .content = "<|im_end|>" } };

const mf_plugin = @import("../mage_flow/plugin.zig");

/// An affine 4-bit linear (group 64), no bias.
fn qlinear(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, out: u64, in: u64) !void {
    const sc: f32 = 0.2 / @sqrt(@as(f32, @floatFromInt(in)));
    try t.words(fmt ++ ".weight", args, &.{ out, in / 8 });
    try t.fill(fmt ++ ".scales", args, &.{ out, in / 64 }, .BF16, sc);
    try t.fill(fmt ++ ".biases", args, &.{ out, in / 64 }, .BF16, -7.5 * sc);
}

fn linear(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, out: u64, in: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in }, .BF16, 0.8 / @sqrt(@as(f32, @floatFromInt(in))));
}

/// A conv in PyTorch's OIHW layout (the loader transposes), f32.
fn conv(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, out: u64, in: u64, k: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in, k, k }, .F32, 0.8 / @sqrt(@as(f32, @floatFromInt(k * k * in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .F32, 0.05);
}

fn res(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, in: u64, out: u64) !void {
    try t.fill(fmt ++ ".norm1.gamma", args, &.{in}, .F32, 1);
    try conv(t, fmt ++ ".conv1", args, out, in, 3);
    try t.fill(fmt ++ ".norm2.gamma", args, &.{out}, .F32, 1);
    try conv(t, fmt ++ ".conv2", args, out, out, 3);
    if (in != out) try conv(t, fmt ++ ".conv_shortcut", args, out, in, 1);
}

fn mid(t: *sdk.tensors.Tensors, comptime side: []const u8, c: u64) !void {
    try res(t, side ++ ".mid_block.resnets.0", .{}, c, c);
    try res(t, side ++ ".mid_block.resnets.1", .{}, c, c);
    try t.fill(side ++ ".mid_block.attentions.0.norm.gamma", .{}, &.{c}, .F32, 1);
    try conv(t, side ++ ".mid_block.attentions.0.to_qkv", .{}, 3 * c, c, 1);
    try conv(t, side ++ ".mid_block.attentions.0.proj", .{}, c, c, 1);
}

const z = 16;
const mult = [_]u64{ 1, 1, 2, 2, 2 };
const vae_cfg =
    \\{"base_dim":8,"decoder_base_dim":8,"z_dim":16,"num_res_blocks":1,"dim_mult":[1,1,2,2,2],"temperal_downsample":[false,true,true,false],
    \\"latents_mean":[0.1,-0.2,0.3,0,0.05,-0.1,0.2,0,0.1,-0.3,0,0.2,-0.05,0.1,0,-0.1],
    \\"latents_std":[1.5,2,1.2,1,1.8,2.2,1.1,1.4,1.6,2.4,1.3,1,1.9,2.1,1.2,1.7]}
;

/// A seeded random pack in the diffusers layout: the 64-wide Qwen3-VL text stack (mage_flow's tiny encoder), a
/// 2-block DiT (hidden 64, two heads of 32, a 4-bit MLP) and the VAE at 8/16 channels, encoder included (variation).
/// A 256x256 canvas is 16x16 latent tokens.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    var te = T.init(a, 51);
    try mf_plugin.textEncoder(&te, "model.language_model.", 256 + specials.len, 64);

    const d = 64;
    var dit = T.init(a, 52);
    try linear(&dit, "img_in", .{}, d, z);
    try dit.fill("txt_in.text_norm.weight", .{}, &.{64}, .BF16, 0);
    try linear(&dit, "txt_in.in_layer", .{}, d, 64);
    try linear(&dit, "txt_in.out_layer", .{}, d, d);
    try linear(&dit, "time_text_embed.timestep_embedder.linear_1", .{}, d, 256);
    try linear(&dit, "time_text_embed.timestep_embedder.linear_2", .{}, d, d);
    try linear(&dit, "modulation.1", .{}, 4 * d, d);
    try linear(&dit, "norm_out.linear", .{}, d, d);
    try linear(&dit, "proj_out", .{}, z, d);
    for (0..2) |i| {
        inline for (.{ "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0" }) |n| try linear(&dit, "transformer_blocks.{d}." ++ n, .{i}, d, d);
        inline for (.{ "attn.norm_q", "attn.norm_k" }) |n| try dit.fill("transformer_blocks.{d}." ++ n ++ ".weight", .{i}, &.{32}, .BF16, 1);
        try qlinear(&dit, "transformer_blocks.{d}.img_mlp.proj", .{i}, 3 * d, d);
        try qlinear(&dit, "transformer_blocks.{d}.img_mlp.gate_layer", .{i}, 3 * d, d);
        try qlinear(&dit, "transformer_blocks.{d}.img_mlp.out", .{i}, d, 3 * d);
    }

    var vae = T.init(a, 53);
    const dec = 8;
    try conv(&vae, "post_quant_conv", .{}, z, z, 1);
    try conv(&vae, "decoder.conv_in", .{}, dec * mult[4], z, 3);
    try mid(&vae, "decoder", dec * mult[4]);
    var c_in: u64 = dec * mult[4];
    for (0..5) |i| {
        const c_out = dec * mult[4 - i];
        for (0..2) |r| try res(&vae, "decoder.up_blocks.{d}.resnets.{d}", .{ i, r }, if (r == 0) c_in else c_out, c_out);
        if (i < 4) try conv(&vae, "decoder.up_blocks.{d}.upsampler.resample.1", .{i}, c_out, c_out, 3);
        c_in = c_out;
    }
    try vae.fill("decoder.norm_out.gamma", .{}, &.{c_in}, .F32, 1);
    try conv(&vae, "decoder.conv_out", .{}, 4, c_in, 3);

    try conv(&vae, "encoder.conv_in", .{}, dec, 4, 3);
    c_in = dec;
    for (0..5) |i| {
        const c_out = dec * mult[i];
        try res(&vae, "encoder.down_blocks.{d}.resnets.0", .{i}, c_in, c_out);
        if (i < 4) try conv(&vae, "encoder.down_blocks.{d}.downsampler.resample.1", .{i}, c_out, c_out, 3);
        c_in = c_out;
    }
    try mid(&vae, "encoder", c_in);
    try vae.fill("encoder.norm_out.gamma", .{}, &.{c_in}, .F32, 1);
    try conv(&vae, "encoder.conv_out", .{}, 2 * z, c_in, 3);
    try conv(&vae, "quant_conv", .{}, 2 * z, 2 * z, 1);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "model_index.json", .bytes = "{\"_class_name\":\"QwenImage21Pipeline\"}" },
            .{ .name = "processor/tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &specials) },
            .{ .name = "text_encoder/config.json", .bytes = "{}" },
            .{ .name = "text_encoder/model.safetensors", .bytes = try te.bytes() },
            .{ .name = "transformer/config.json", .bytes = "{\"num_layers\":2,\"num_attention_heads\":2,\"attention_head_dim\":32,\"in_channels\":16,\"out_channels\":16,\"context_in_dim\":64,\"mlp_ratio\":3,\"axes_dims_rope\":[8,12,12]}" },
            .{ .name = "transformer/model.safetensors", .bytes = try dit.bytes() },
            .{ .name = "vae/config.json", .bytes = vae_cfg },
            .{ .name = "vae/model.safetensors", .bytes = try vae.bytes() },
        }),
        .body = "{\"prompt\":\"a red cat\",\"size\":\"256x256\",\"steps\":2,\"seed\":3,\"guidance_scale\":2.5,\"negative_prompt\":\"blurry\"}",
    };
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack;
const golden = @import("../../sdk/testing/golden.zig");

test "qwen_image: a job on the tiny pack is a 256x256 PNG, RGBA on request; an edit without the tower is refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const png = (try pack.run(pack.fx.body)).ok;
    try testing.expectEqual(image_pixels.Size{ .w = 256, .h = 256 }, image_pixels.nativeSize(png).?);
    const rgba = (try pack.run("{\"prompt\":\"a\",\"size\":\"256x256\",\"steps\":1,\"transparent\":true}")).ok;
    try testing.expectEqual(@as(u8, 6), rgba[25]); // IHDR colour type 6: RGBA
    const refused = (try pack.run("{\"prompt\":\"a\",\"mode\":\"edit\",\"image\":\"x\"}")).refused;
    try testing.expect(refused.code == .unsupported);
}

test "qwen_image: the tiny pack's pipeline (CFG, variation) is bit-equal to its golden run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    const t2i = try e.m.generateImage(gpa, "a red cat", 256, 256, 3, 2, .{ .guidance_scale = 2.5, .negative_prompt = "blurry" }, null);
    defer _ = mlx.mlx_array_free(t2i);
    var src = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(src);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 11));
    const lo = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(lo);
    const hi = mlx.mlx_array_new_float(1);
    defer _ = mlx.mlx_array_free(hi);
    const sh = [_]c_int{ 1, 3, 256, 256 };
    try mlx.check(mlx.mlx_random_uniform(&src, lo, hi, &sh, 4, .float32, key, e.m.s));
    const v = try e.m.generateImage(gpa, "a cat", 256, 256, 7, 3, .{ .init_image = src, .start_step = 1 }, null);
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_array_eval(t2i));
    try mlx.check(mlx.mlx_array_eval(v));
    try golden.expect("qwen_image_pipeline", &.{ .{ .name = "t2i", .array = t2i }, .{ .name = "variation", .array = v } });
}
