//! MageFlow on the media table: Turbo text to image over its Qwen3-VL encoder, double-stream DiT and DiCo VAE, and
//! the Edit checkpoint's multi-reference editor (raw upload bytes, resized by the engine). The result is a PNG.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const image_request = @import("../../sdk/media/image_request.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const mlx = @import("../../mlx/ffi.zig");
const mf = @import("mage_flow.zig");
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
    m: *mf.Engine,
    png: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const m = mf.Engine.loadOn(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch |err| {
        out.* = job.failure(err, "mage_flow: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        m.deinit();
        out.* = Verdict.refuse(.failed, "mage_flow: out of memory");
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

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return Verdict.refuse(.unsupported, "invalid JSON body");
    defer body.deinit();
    var inputs: [16][]const u8 = undefined;
    const in = req.inputs.slice();
    const spec: image_request.Spec = .{ .edit = e.m.supportsEdit(), .edit_ref_cap = config.edit_ref_cap };
    if (in.len > inputs.len) return Verdict.named(.unsupported, spec.too_many_refs);
    for (in, 0..) |s, i| inputs[i] = s.slice();
    const j = switch (image_request.parse(.{ .obj = body.parsed.value.object }, inputs[0..in.len], spec)) {
        .ok => |j| j,
        .bad => |v| return v,
    };
    var w = image_request.clamp16(j.req_w);
    var h = image_request.clamp16(j.req_h);
    var refs: [config.edit_ref_cap][]const u8 = undefined;
    var n_refs: usize = 0;
    if (j.mode == .edit) {
        // The edit runs AT the target grid and resizes every reference onto it: the primary's aspect, at the
        // requested budget or at its own size.
        const nat = image_pixels.nativeSize(j.image.?) orelse return Verdict.refuse(.unsupported, "could not decode 'image' (PNG/JPEG/WebP supported)");
        for (j.refs) |r| if (image_pixels.nativeSize(r) == null) return Verdict.refuse(.unsupported, "could not decode a 'ref_images' entry (PNG/JPEG/WebP supported)");
        const fit = if (j.size_given)
            image_request.resolveEditTargetSize(nat.w, nat.h, w, h, config.max_dim)
        else
            image_request.resolveEditTargetSize(nat.w, nat.h, nat.w, nat.h, config.max_dim);
        const tw = image_request.clamp16(fit.w);
        const th = image_request.clamp16(fit.h);
        if (tw != w or th != h)
            log.info("[image] edit: target {d}x{d} -> {d}x{d} (primary reference is {d}x{d}, size {s})\n", .{ w, h, tw, th, nat.w, nat.h, if (j.size_given) "requested" else "matched to source" });
        w = tw;
        h = th;
        refs[0] = j.image.?;
        for (j.refs) |r| {
            n_refs += 1;
            refs[n_refs] = r;
        }
        n_refs += 1;
        image_pixels.logByteRefs(refs[0..n_refs]);
    }
    const sp = job.progressOf(progress);
    const img = (if (n_refs != 0)
        e.m.editImage(gpa, j.prompt, refs[0..n_refs], w, h, j.seed, j.steps, sp)
    else
        e.m.generateImage(gpa, j.prompt, w, h, j.seed, j.steps, sp)) catch |err|
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
    if (e.png.len == 0) return Verdict.refuse(.failed, "mage_flow: no finished job to encode");
    out.* = .of(e.png);
    return .ok;
}

// ── the tiny pack ──

const specials = [_]sdk.byte_tokenizer.Special{ .{ .id = 256, .content = "<|im_start|>" }, .{ .id = 257, .content = "<|im_end|>" } };

/// Qwen3-VL's text stack at a 64-wide hidden: its 36 layers and 32/8 heads of 128 are the encoder's own (shared
/// with Qwen-Image), so every layer aliases layer 0's bytes. `pfx` is the pack's prefix.
pub fn textEncoder(t: *sdk.tensors.Tensors, comptime pfx: []const u8, vocab: u64, comptime hidden: u64) !void {
    const inter = 64;
    try t.rand(pfx ++ "embed_tokens.weight", .{}, &.{ vocab, hidden }, .BF16, 0.5);
    try t.fill(pfx ++ "norm.weight", .{}, &.{hidden}, .BF16, 1);
    const shapes = .{
        .{ "input_layernorm", [_]u64{hidden} },     .{ "post_attention_layernorm", [_]u64{hidden} },
        .{ "self_attn.q_norm", [_]u64{128} },       .{ "self_attn.k_norm", [_]u64{128} },
        .{ "self_attn.q_proj", [_]u64{ 4096, hidden } }, .{ "self_attn.k_proj", [_]u64{ 1024, hidden } },
        .{ "self_attn.v_proj", [_]u64{ 1024, hidden } }, .{ "self_attn.o_proj", [_]u64{ hidden, 4096 } },
        .{ "mlp.gate_proj", [_]u64{ inter, hidden } }, .{ "mlp.up_proj", [_]u64{ inter, hidden } },
        .{ "mlp.down_proj", [_]u64{ hidden, inter } },
    };
    inline for (shapes) |sh| {
        const name = pfx ++ "layers.0." ++ sh[0] ++ ".weight";
        const shape: [sh[1].len]u64 = sh[1];
        if (shape.len == 1) try t.fill(name, .{}, &shape, .BF16, 1) else try t.rand(name, .{}, &shape, .BF16, 0.1);
        for (1..36) |i| try t.alias(pfx ++ "layers.{d}." ++ sh[0] ++ ".weight", .{i}, name);
    }
}

fn linear(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, out: u64, in: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in }, .BF16, 0.6 / @sqrt(@as(f32, @floatFromInt(in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .BF16, 0.05);
}

/// An affine 4-bit linear (group 64) with a bias.
fn qlinear(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, out: u64, in: u64) !void {
    const sc: f32 = 0.2 / @sqrt(@as(f32, @floatFromInt(in)));
    try t.words(fmt ++ ".weight", args, &.{ out, in / 8 });
    try t.fill(fmt ++ ".scales", args, &.{ out, in / 64 }, .BF16, sc);
    try t.fill(fmt ++ ".biases", args, &.{ out, in / 64 }, .BF16, -7.5 * sc);
    try t.rand(fmt ++ ".bias", args, &.{out}, .BF16, 0.05);
}

/// A conv in PyTorch's OIHW layout (the loader transposes), f32.
fn conv(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, out: u64, in: u64, k: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in, k, k }, .F32, 0.8 / @sqrt(@as(f32, @floatFromInt(k * k * in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .F32, 0.05);
}

fn vlinear(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, out: u64, in: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in }, .F32, 0.8 / @sqrt(@as(f32, @floatFromInt(in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .F32, 0.05);
}

fn norm(t: *sdk.tensors.Tensors, comptime fmt: []const u8, args: anytype, c: u64) !void {
    try t.fill(fmt ++ ".weight", args, &.{c}, .F32, 1);
    try t.fill(fmt ++ ".bias", args, &.{c}, .F32, 0);
}

/// A seeded random Turbo pack in the diffusers layout: the 64-wide encoder above, a one-block DiT (hidden 64, two
/// heads of 32, one 4-bit MLP), and the DiCo VAE decoder at 32 channels (8-wide pixel decoder). A 256x256 canvas is
/// 16x16 latent tokens.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    const te_h = 64;
    var te = T.init(a, 41);
    try textEncoder(&te, "model.language_model.", 256 + specials.len, te_h);

    const d = 64;
    var dit = T.init(a, 42);
    try linear(&dit, "img_in", .{}, d, 128);
    try dit.fill("txt_norm.weight", .{}, &.{te_h}, .BF16, 1);
    try linear(&dit, "txt_in", .{}, d, te_h);
    try linear(&dit, "time_text_embed.timestep_embedder.linear_1", .{}, d, 256);
    try linear(&dit, "time_text_embed.timestep_embedder.linear_2", .{}, d, d);
    const b = "transformer_blocks.0.";
    try linear(&dit, b ++ "img_mod.1", .{}, 6 * d, d);
    try linear(&dit, b ++ "txt_mod.1", .{}, 6 * d, d);
    inline for (.{ "to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_out.0", "to_add_out" }) |n| try linear(&dit, b ++ "attn." ++ n, .{}, d, d);
    inline for (.{ "norm_q", "norm_k", "norm_added_q", "norm_added_k" }) |n| try dit.fill(b ++ "attn." ++ n ++ ".weight", .{}, &.{32}, .BF16, 1);
    try qlinear(&dit, b ++ "img_mlp.net.0.proj", .{}, 2 * d, d);
    try qlinear(&dit, b ++ "img_mlp.net.2", .{}, d, 2 * d);
    try linear(&dit, b ++ "txt_mlp.net.0.proj", .{}, 2 * d, d);
    try linear(&dit, b ++ "txt_mlp.net.2", .{}, d, 2 * d);
    try linear(&dit, "norm_out.linear", .{}, 2 * d, d);
    try linear(&dit, "proj_out", .{}, 128, d);

    const c = 32;
    const hx = 8;
    var vae = T.init(a, 43);
    const dec = "pipeline.y_embedder.decoder.";
    try conv(&vae, dec ++ "conv_in", .{}, c, 128, 3);
    for ([_]usize{ 0, 2, 4 }) |i| {
        try norm(&vae, dec ++ "block.{d}.norm1", .{i}, c);
        try conv(&vae, dec ++ "block.{d}.conv1", .{i}, c, c, 3);
        try norm(&vae, dec ++ "block.{d}.norm2", .{i}, c);
        try conv(&vae, dec ++ "block.{d}.conv2", .{i}, c, c, 3);
    }
    for ([_]usize{ 1, 3 }) |i| {
        try norm(&vae, dec ++ "block.{d}.norm", .{i}, c);
        inline for (.{ "q", "k", "v", "proj_out" }) |n| try conv(&vae, dec ++ "block.{d}." ++ n, .{i}, c, c, 1);
    }
    try norm(&vae, dec ++ "norm_out", .{}, c);
    try conv(&vae, dec ++ "conv_out", .{}, c, c, 3);
    try vlinear(&vae, "pipeline.t_embedder.mlp.0", .{}, c, 256);
    try vlinear(&vae, "pipeline.t_embedder.mlp.2", .{}, c, c);
    try conv(&vae, "pipeline.s_embedder.proj2", .{}, c, 128 + c, 1);
    for (0..21) |i| {
        try conv(&vae, "pipeline.blocks.{d}.conv1", .{i}, c, c, 1);
        try conv(&vae, "pipeline.blocks.{d}.conv2", .{i}, c, 1, 3);
        try conv(&vae, "pipeline.blocks.{d}.conv3", .{i}, c, c, 1);
        try conv(&vae, "pipeline.blocks.{d}.ca.1", .{i}, c, c, 1);
        try conv(&vae, "pipeline.blocks.{d}.conv4", .{i}, 2 * c, c, 1);
        try conv(&vae, "pipeline.blocks.{d}.conv5", .{i}, c, 2 * c, 1);
        try vlinear(&vae, "pipeline.blocks.{d}.adaLN_modulation.1", .{i}, 6 * c, c);
    }
    try conv(&vae, "pipeline.y_embedder_x", .{}, hx * 256, c, 1);
    try vlinear(&vae, "pipeline.x_embedder.embedder.0", .{}, hx, 3 + hx + 64);
    try vlinear(&vae, "pipeline.dec_net.cond_embed", .{}, hx * 256, c);
    try vlinear(&vae, "pipeline.dec_net.input_proj", .{}, hx, hx);
    for (0..3) |i| {
        try norm(&vae, "pipeline.dec_net.res_blocks.{d}.in_ln", .{i}, hx);
        try vlinear(&vae, "pipeline.dec_net.res_blocks.{d}.mlp.0", .{i}, hx, hx);
        try vlinear(&vae, "pipeline.dec_net.res_blocks.{d}.mlp.2", .{i}, hx, hx);
        try vlinear(&vae, "pipeline.dec_net.res_blocks.{d}.adaLN_modulation.1", .{i}, 3 * hx, hx);
    }
    try vae.fill("pipeline.final_layer.norm.weight", .{}, &.{hx}, .F32, 1);
    try vlinear(&vae, "pipeline.final_layer.linear", .{}, 3, hx);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "model_index.json", .bytes = "{\"_class_name\":\"MageFlowPipeline\",\"_mage_flow_version\":\"0.1.0\"}" },
            .{ .name = "text_encoder/config.json", .bytes = "{}" },
            .{ .name = "text_encoder/tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &specials) },
            .{ .name = "text_encoder/model.safetensors", .bytes = try te.bytes() },
            .{ .name = "transformer/config.json", .bytes = "{\"in_channels\":128,\"context_in_dim\":64,\"hidden_size\":64,\"num_heads\":2,\"depth\":1,\"mlp_ratio\":2.0,\"axes_dim\":[8,12,12],\"theta\":10000}" },
            .{ .name = "transformer/model.safetensors", .bytes = try dit.bytes() },
            .{ .name = "vae/config.json", .bytes = "{}" },
            .{ .name = "vae/model.safetensors", .bytes = try vae.bytes() },
        }),
        .body = "{\"prompt\":\"a red cat\",\"size\":\"256x256\",\"steps\":2,\"seed\":3}",
    };
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack;
const golden = @import("../../sdk/testing/golden.zig");

test "mage_flow: a job on the tiny pack is a 256x256 PNG; an edit on the Turbo pack and a variation are refused" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const png = (try pack.run(pack.fx.body)).ok;
    try testing.expectEqual(image_pixels.Size{ .w = 256, .h = 256 }, image_pixels.nativeSize(png).?);
    const edit = (try pack.run("{\"prompt\":\"a\",\"mode\":\"edit\",\"image\":\"x\"}")).refused;
    try testing.expect(edit.code == .unsupported);
    const variation = (try pack.run("{\"prompt\":\"a\",\"image\":\"x\"}")).refused;
    try testing.expect(variation.code == .unsupported);
}

test "mage_flow: the tiny pack's Turbo pipeline is bit-equal to its golden run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    const img = try e.m.generateImage(gpa, "a red cat", 256, 256, 3, 2, null);
    defer _ = mlx.mlx_array_free(img);
    try mlx.check(mlx.mlx_array_eval(img));
    try golden.expect("mage_flow_pipeline", &.{.{ .name = "image", .array = img }});
}
