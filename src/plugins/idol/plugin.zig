//! Idol on the media table: one image to a 3D Gaussian splat, returned as a GLB: by default a mesh of one textured
//! disc per Gaussian (every viewer draws it solid), or with `geometry: "splat"` the ratified `KHR_gaussian_splatting`
//! extension on one POINTS primitive.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const mesh_request = @import("../../sdk/media/mesh_request.zig");
const t2_plugin = @import("../trellis2/plugin.zig");
const idol = @import("idol.zig");
const splat = @import("splat.zig");
const image = @import("image.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .mesh,
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
    m: *idol.Engine,
    glb: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const m = idol.Engine.loadOn(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch |err| {
        log.err("[idol] load failed: {s}\n", .{@errorName(err)});
        out.* = job.failure(err, "idol: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        m.deinit();
        out.* = Verdict.refuse(.failed, "idol: out of memory");
        return null;
    };
    e.* = .{ .m = m };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.m.deinit();
    gpa.free(e.glb);
    gpa.destroy(e);
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

const Geometry = enum { discs, splat };
const Parsed = union(enum) { ok: struct { o: idol.Opts, geometry: Geometry }, bad: Verdict };

/// The request's fields: `steps`, `seed`, `guidance_scale` (`cfg_strength`), `rescale_t`, `num_gaussians`, `mode`
/// (`synthetic` renders and generated images, `real` photos), `crop_margin` and `geometry` (`discs`, the default: a
/// triangle mesh any viewer draws; `splat`: the `KHR_gaussian_splatting` points). Each wrong value is named.
pub fn parseOpts(body: *const job.Body) Parsed {
    var o = idol.Opts{ .seed = body.uint("seed") orelse 42, .steps = config.DEFAULT_STEPS };
    if (mesh_request.has(&body.parsed.value.object, "steps")) {
        const n = body.uint("steps") orelse return .{ .bad = bad("'steps' must be a whole number in [1,200]") };
        if (n < 1 or n > config.MAX_STEPS) return .{ .bad = bad("'steps' must be a whole number in [1,200]") };
        o.steps = @intCast(n);
    }
    inline for (.{ "guidance_scale", "cfg_strength" }) |k| if (mesh_request.has(&body.parsed.value.object, k)) {
        const g = body.float(k) orelse return .{ .bad = bad("'guidance_scale' must be a number in [0,20]") };
        if (!(g >= 0 and g <= 20)) return .{ .bad = bad("'guidance_scale' must be a number in [0,20]") };
        o.cfg_strength = @floatCast(g);
    };
    if (mesh_request.has(&body.parsed.value.object, "rescale_t")) {
        const r = body.float("rescale_t") orelse return .{ .bad = bad("'rescale_t' must be a number in [1,10]") };
        if (!(r >= 1 and r <= 10)) return .{ .bad = bad("'rescale_t' must be a number in [1,10]") };
        o.rescale_t = @floatCast(r);
    }
    if (mesh_request.has(&body.parsed.value.object, "num_gaussians")) {
        const n = body.uint("num_gaussians") orelse return .{ .bad = bad("'num_gaussians' must be a whole number in [32768,1048576]") };
        if (n < config.MIN_GAUSSIANS or n > config.MAX_GAUSSIANS) return .{ .bad = bad("'num_gaussians' must be a whole number in [32768,1048576]") };
        o.num_gaussians = @intCast(n - n % 32);
    }
    if (body.str("mode")) |m| {
        if (std.mem.eql(u8, m, "real")) {
            o.margin = image.MARGIN_REAL;
        } else if (!std.mem.eql(u8, m, "synthetic")) return .{ .bad = bad("'mode' must be \"synthetic\" (renders, generated images) or \"real\" (photographs)") };
    } else if (mesh_request.has(&body.parsed.value.object, "mode")) return .{ .bad = bad("'mode' must be \"synthetic\" (renders, generated images) or \"real\" (photographs)") };
    if (mesh_request.has(&body.parsed.value.object, "crop_margin")) {
        const c = body.float("crop_margin") orelse return .{ .bad = bad("'crop_margin' must be a number in [1,4]") };
        if (!(c >= 1 and c <= 4)) return .{ .bad = bad("'crop_margin' must be a number in [1,4]") };
        o.margin = c;
    }
    var geometry: Geometry = .discs;
    if (mesh_request.has(&body.parsed.value.object, "geometry")) {
        const g = body.str("geometry") orelse return .{ .bad = bad("'geometry' must be \"discs\" or \"splat\"") };
        geometry = std.meta.stringToEnum(Geometry, g) orelse return .{ .bad = bad("'geometry' must be \"discs\" or \"splat\"") };
    }
    return .{ .ok = .{ .o = o, .geometry = geometry } };
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    var views: mesh_request.Views = .{};
    defer views.deinit(gpa);
    if (mesh_request.decode(gpa, &body.parsed.value.object, req.inputs.slice(), 1, &views)) |v| return v;
    const img = views.list.items[0];
    const parsed_req = switch (parseOpts(&body)) {
        .ok => |o| o,
        .bad => |v| return v,
    };
    const opts = parsed_req.o;
    log.info("[mesh] generating Idol steps={d} gaussians={d} cfg={d:.1} rescale_t={d:.1} seed={d}\n", .{ opts.steps, opts.num_gaussians, opts.cfg_strength, opts.rescale_t, opts.seed });
    var gen = e.m.generate(img.pix, img.w, img.h, opts, job.progressOf(progress)) catch |err| return t2_plugin.jobFailure(err);
    defer gen.deinit(gpa);
    if (job.latched()) |v| return v;
    const glb = (switch (parsed_req.geometry) {
        .discs => splat.writeDiscGlb(gpa, &gen.splats),
        .splat => splat.writeGlb(gpa, &gen.splats),
    }) catch |err| return job.failure(err, "generation failed");
    gpa.free(e.glb);
    e.glb = glb;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.glb.len == 0) return Verdict.refuse(.failed, "idol: no finished job to encode");
    out.* = .of(e.glb);
    return .ok;
}

// ── the tiny pack ──

const T = sdk.tensors.Tensors;
const C: u64 = 128; // channels of the three transformers: two heads of 64
const HEADS: u64 = 2;
const COND: u64 = 16; // the latent width the decoders read
const DINO: u64 = 64;
const VAE_C: u64 = 64;

fn lin(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64, bias: bool) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in }, .F32, 0.5 / @sqrt(@as(f32, @floatFromInt(in))));
    if (bias) try t.rand(fmt ++ ".bias", args, &.{out}, .F32, 0.05);
}

fn gamma(t: *T, comptime fmt: []const u8, args: anytype) !void {
    try t.fill(fmt, args, &.{ HEADS, 64 }, .F32, 1);
}

fn mha(t: *T, comptime p: []const u8, args: anytype, cross: bool) !void {
    if (cross) {
        try lin(t, p ++ ".to_q", args, C, C, true);
        try lin(t, p ++ ".to_kv", args, 2 * C, COND, true);
    } else try lin(t, p ++ ".to_qkv", args, 3 * C, C, true);
    try lin(t, p ++ ".to_out", args, C, C, true);
    try gamma(t, p ++ ".q_rms_norm.gamma", args);
    try gamma(t, p ++ ".k_rms_norm.gamma", args);
}

fn mlp(t: *T, comptime p: []const u8, args: anytype) !void {
    try lin(t, p ++ ".mlp.0", args, 4 * C, C, true);
    try lin(t, p ++ ".mlp.2", args, C, 4 * C, true);
}

fn dBlock(t: *T, comptime p: []const u8, args: anytype, modulated: bool) !void {
    try lin(t, p ++ ".attn.qkv", args, 3 * C, C, true);
    try lin(t, p ++ ".attn.out", args, C, C, true);
    try gamma(t, p ++ ".attn.q_norm.gamma", args);
    try gamma(t, p ++ ".attn.k_norm.gamma", args);
    try mlp(t, p ++ ".mlp", args);
    if (modulated) {
        try t.rand(p ++ ".shift_table", args, &.{ 1, 6 * C }, .F32, 0.05);
    } else inline for (.{ ".norm1", ".norm2" }) |n| {
        try t.fill(p ++ n ++ ".weight", args, &.{C}, .F32, 1);
        try t.fill(p ++ n ++ ".bias", args, &.{C}, .F32, 0);
    }
}

fn repo(t: *T, comptime p: []const u8, args: anytype) !void {
    try t.fill(p ++ ".norm.weight", args, &.{C}, .F32, 1);
    try t.fill(p ++ ".norm.bias", args, &.{C}, .F32, 0);
    try lin(t, p ++ ".gate_map", args, C / 8, C, false);
    try lin(t, p ++ ".content_map", args, C / 8, C, false);
    try lin(t, p ++ ".final_map", args, 3 * HEADS, C / 8, false);
    try t.rand(p ++ ".freqs_0", args, &.{10}, .F32, 1);
    try t.rand(p ++ ".freqs_1", args, &.{10}, .F32, 1);
    try t.rand(p ++ ".freqs_2", args, &.{12}, .F32, 1);
}

fn denoiserPack(a: std.mem.Allocator) ![]u8 {
    var t = T.init(a, 31);
    try lin(&t, "input_layer", .{}, C, 16, true);
    try lin(&t, "cond_embedder", .{}, C, DINO, true);
    try lin(&t, "cond_embedder2", .{}, C, 128, true);
    try lin(&t, "t_embedder.mlp.0", .{}, C, 256, true);
    try lin(&t, "t_embedder.mlp.2", .{}, C, C, true);
    try lin(&t, "adaLN_modulation.1", .{}, 6 * C, C, true);
    try lin(&t, "out_layer", .{}, 16, C, true);
    try lin(&t, "cam_out_layer", .{}, 5, C, true);
    try lin(&t, "cam_refiner.mlp.0", .{}, C, 5, true);
    try lin(&t, "cam_refiner.mlp.2", .{}, C, C, true);
    try t.rand("shift_table", .{}, &.{ 1, 2, C }, .F32, 0.05);
    try t.values("pos_pe", .{}, &.{ 1, 64, 3 }, .F32, try sobol(a, 64));
    try dBlock(&t, "noise_refiner.0", .{}, true);
    try dBlock(&t, "context_refiner.0", .{}, false);
    try dBlock(&t, "blocks.0", .{}, true);
    try repo(&t, "noise_repo_layers.0", .{});
    try repo(&t, "context_repo_layers.0", .{});
    try repo(&t, "repo_layers.0", .{});
    return t.bytes();
}

/// 64 fixed positions in [0, 1]^3 (a van der Corput triple stands in for the scrambled Sobol sequence).
fn sobol(a: std.mem.Allocator, n: usize) ![]f32 {
    const out = try a.alloc(f32, n * 3);
    for (0..n) |i| for ([_]u32{ 2, 3, 5 }, 0..) |base, d| {
        var f: f32 = 1;
        var r: f32 = 0;
        var k: u32 = @intCast(i + 1);
        while (k > 0) : (k /= base) {
            f /= @floatFromInt(base);
            r += f * @as(f32, @floatFromInt(k % base));
        }
        out[i * 3 + d] = r;
    };
    return out;
}

fn octreePack(a: std.mem.Allocator) ![]u8 {
    var t = T.init(a, 32);
    try lin(&t, "in_proj", .{}, C, 3, true);
    try lin(&t, "input_layer", .{}, C, C, true);
    try lin(&t, "l_embedder.mlp.0", .{}, C, 256, true);
    try lin(&t, "l_embedder.mlp.2", .{}, C, C, true);
    try lin(&t, "adaLN_modulation.1", .{}, 6 * C, C, true);
    try lin(&t, "out_proj", .{}, 8, C, true);
    try mha(&t, "blocks.0.cross_attn", .{}, true);
    try mlp(&t, "blocks.0.mlp", .{});
    return t.bytes();
}

fn gaussianPack(a: std.mem.Allocator) ![]u8 {
    var t = T.init(a, 33);
    try lin(&t, "in_proj", .{}, C, 3, true);
    try lin(&t, "input_layer", .{}, C, C, true);
    try lin(&t, "out_proj", .{}, 480, C, true);
    try mha(&t, "blocks.0.self_attn", .{}, false);
    try mha(&t, "blocks.0.cross_attn", .{}, true);
    try mlp(&t, "blocks.0.mlp", .{});
    try t.fill("blocks.0.norm2.weight", .{}, &.{C}, .F32, 1);
    try t.fill("blocks.0.norm2.bias", .{}, &.{C}, .F32, 0);
    try t.rand("points_offset_perturbation", .{}, &.{ 32, 3 }, .F32, 0.5);
    try t.fill("base_offset_scale", .{}, &.{}, .F32, 0.5);
    return t.bytes();
}

fn dinoPack(a: std.mem.Allocator) ![]u8 {
    var t = T.init(a, 34);
    try t.rand("embeddings.cls_token", .{}, &.{ 1, 1, DINO }, .F32, 0.1);
    try t.rand("embeddings.register_tokens", .{}, &.{ 1, 4, DINO }, .F32, 0.1);
    try t.rand("embeddings.patch_embeddings.weight", .{}, &.{ DINO, 3, 16, 16 }, .F32, 0.05);
    try t.rand("embeddings.patch_embeddings.bias", .{}, &.{DINO}, .F32, 0.05);
    inline for (.{ "norm1", "norm2" }) |n| {
        try t.fill("layer.0." ++ n ++ ".weight", .{}, &.{DINO}, .F32, 1);
        try t.fill("layer.0." ++ n ++ ".bias", .{}, &.{DINO}, .F32, 0);
    }
    inline for (.{ "q_proj", "k_proj", "v_proj", "o_proj" }) |n| try lin(&t, "layer.0.attention." ++ n, .{}, DINO, DINO, !std.mem.eql(u8, n, "k_proj"));
    inline for (.{ "gate_proj", "up_proj" }) |n| try lin(&t, "layer.0.mlp." ++ n, .{}, 2 * DINO, DINO, true);
    try lin(&t, "layer.0.mlp.down_proj", .{}, DINO, 2 * DINO, true);
    inline for (.{ "layer_scale1", "layer_scale2" }) |n| try t.fill("layer.0." ++ n ++ ".lambda1", .{}, &.{DINO}, .F32, 1);
    try t.fill("norm.weight", .{}, &.{DINO}, .F32, 1);
    try t.fill("norm.bias", .{}, &.{DINO}, .F32, 0);
    return t.bytes();
}

fn conv(t: *T, comptime fmt: []const u8, args: anytype, out: u64, k: u64, in: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, k, k, in }, .BF16, 0.8 / @sqrt(@as(f32, @floatFromInt(k * k * in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .BF16, 0.05);
}

fn resnet(t: *T, comptime fmt: []const u8, args: anytype, c: u64) !void {
    inline for (.{ ".norm1", ".norm2" }) |n| {
        try t.fill(fmt ++ n ++ ".weight", args, &.{c}, .BF16, 1);
        try t.fill(fmt ++ n ++ ".bias", args, &.{c}, .BF16, 0);
    }
    try conv(t, fmt ++ ".conv1", args, c, 3, c);
    try conv(t, fmt ++ ".conv2", args, c, 3, c);
}

/// An affine 4-bit linear (group 64) with a bias, in mflux's layout: the VAE's attention reads only quantized ones.
fn qlinear(t: *T, comptime fmt: []const u8, out: u64, in: u64) !void {
    const sc: f32 = 0.2 / @sqrt(@as(f32, @floatFromInt(in)));
    try t.words(fmt ++ ".weight", .{}, &.{ out, in / 8 });
    try t.fill(fmt ++ ".scales", .{}, &.{ out, in / 64 }, .BF16, sc);
    try t.fill(fmt ++ ".biases", .{}, &.{ out, in / 64 }, .BF16, -7.5 * sc);
    try t.rand(fmt ++ ".bias", .{}, &.{out}, .BF16, 0.05);
}

/// The FLUX.2 VAE encoder at 64 channels in mflux's layout (channels-last convs), with its batch-norm statistics.
fn vaePack(a: std.mem.Allocator) ![]u8 {
    const c = VAE_C;
    var t = T.init(a, 35);
    try t.rand("bn.running_mean", .{}, &.{128}, .F32, 0.1);
    try t.fill("bn.running_var", .{}, &.{128}, .F32, 1);
    try conv(&t, "encoder.conv_in", .{}, c, 3, 3);
    for (0..4) |bi| {
        for (0..2) |ri| try resnet(&t, "encoder.down_blocks.{d}.resnets.{d}", .{ bi, ri }, c);
        if (bi < 3) try conv(&t, "encoder.down_blocks.{d}.downsamplers.0.conv", .{bi}, c, 3, c);
    }
    for (0..2) |i| try resnet(&t, "encoder.mid_block.resnets.{d}", .{i}, c);
    try t.fill("encoder.mid_block.attentions.0.group_norm.weight", .{}, &.{c}, .BF16, 1);
    try t.fill("encoder.mid_block.attentions.0.group_norm.bias", .{}, &.{c}, .BF16, 0);
    inline for (.{ "to_q", "to_k", "to_v", "to_out" }) |n| try qlinear(&t, "encoder.mid_block.attentions.0." ++ n, c, c);
    try t.fill("encoder.conv_norm_out.weight", .{}, &.{c}, .BF16, 1);
    try t.fill("encoder.conv_norm_out.bias", .{}, &.{c}, .BF16, 0);
    try conv(&t, "encoder.conv_out", .{}, 64, 3, c);
    try conv(&t, "quant_conv", .{}, 64, 1, 64);
    return t.bytes();
}

const tiny_config =
    \\{"model_type":"idol","image_size":64,"max_voxel_level":3,
    \\"denoiser":{"name":"LatentSeqMMFlowModel","args":{"q_token_length":64,"in_channels":16,"cam_channels":5,"model_channels":128,
    \\"cond_channels":64,"cond2_channels":128,"num_blocks":1,"num_refiner_blocks":1,"num_heads":2,"mlp_ratio":4}},
    \\"decoder":{"name":"OctreeProbabilityFixedlenDecoder","args":{"model_channels":128,"cond_channels":16,"num_blocks":1,"num_heads":2,"mlp_ratio":4}},
    \\"decoder_gs":{"name":"GaussianFixedlenDecoder","args":{"in_channels":3,"model_channels":128,"cond_channels":16,"num_blocks":1,"num_heads":2,"mlp_ratio":4,
    \\"representation_config":{"num_gaussians":32,"lr":{"_xyz":1.0,"_features_dc":1.0,"_opacity":1.0,"_scaling":1.0,"_rotation":0.1},
    \\"perturbe_size":1.5,"3d_filter_kernel_size":0.0009,"scaling_bias":0.004,"opacity_bias":0.1}}}}
;

const tiny_dino =
    \\{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":1,"num_attention_heads":1,"patch_size":16,
    \\"num_register_tokens":4,"rope_theta":100.0,"use_gated_mlp":true,"layer_norm_eps":1e-5}
;

/// A 64-pixel pack: every model at the smallest width its code takes (three 128-wide transformers of one block, a
/// one-layer gated DINOv3, a 64-channel VAE encoder); a 64 x 64 cutout is 16 patch tokens beside the 5 prefix rows.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = tiny_config },
            .{ .name = "denoiser.safetensors", .bytes = try denoiserPack(a) },
            .{ .name = "decoder.safetensors", .bytes = try octreePack(a) },
            .{ .name = "decoder_gs.safetensors", .bytes = try gaussianPack(a) },
            .{ .name = "dinov3/config.json", .bytes = tiny_dino },
            .{ .name = "dinov3/model.safetensors", .bytes = try dinoPack(a) },
            .{ .name = "vae/model.safetensors", .bytes = try vaePack(a) },
        }),
        .body = "{\"image\":\"x\",\"steps\":2,\"seed\":3,\"num_gaussians\":32768}",
        .inputs = try a.dupe([]const u8, &.{try t2_plugin.cutout(a)}),
    };
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack;
const mlx = @import("../../mlx/ffi.zig");
const scope = @import("../../mlx/scope.zig");
const weights = @import("../../sdk/weights/load.zig");
const w8a8 = @import("../../sdk/blocks/dit/w8a8.zig");
const den = @import("denoiser.zig");
const layers = @import("layers.zig");
const dinov3 = @import("../../sdk/vision/dinov3.zig");

fn parsed(json: []const u8) !Parsed {
    var b = job.Body.parse(testing.allocator, json).?;
    defer b.deinit();
    return parseOpts(&b);
}

test "idol: request fields are read, rounded and refused by name" {
    const d = (try parsed("{}")).ok.o;
    try testing.expectEqual(@as(u32, 50), d.steps);
    try testing.expectEqual(@as(f32, 3), d.cfg_strength);
    try testing.expectEqual(@as(u32, 262144), d.num_gaussians);
    const fp = (try parsed("{\"steps\":8,\"guidance_scale\":2.5,\"rescale_t\":2,\"num_gaussians\":100010,\"mode\":\"real\",\"seed\":7,\"geometry\":\"splat\"}")).ok;
    const f = fp.o;
    try testing.expectEqual(Geometry.splat, fp.geometry);
    try testing.expectEqual(Geometry.discs, (try parsed("{}")).ok.geometry);
    try testing.expectEqual(@as(u32, 8), f.steps);
    try testing.expectEqual(@as(f32, 2.5), f.cfg_strength);
    try testing.expectEqual(@as(u32, 100000), f.num_gaussians);
    try testing.expectEqual(image.MARGIN_REAL, f.margin);
    try testing.expectEqual(@as(u64, 7), f.seed);
    const cases = [_][2][]const u8{
        .{ "{\"steps\":0}", "'steps'" },
        .{ "{\"steps\":2.5}", "'steps'" },
        .{ "{\"guidance_scale\":-1}", "'guidance_scale'" },
        .{ "{\"rescale_t\":0.5}", "'rescale_t'" },
        .{ "{\"num_gaussians\":100}", "'num_gaussians'" },
        .{ "{\"mode\":\"cartoon\"}", "'mode'" },
        .{ "{\"mode\":3}", "'mode'" },
        .{ "{\"crop_margin\":9}", "'crop_margin'" },
        .{ "{\"geometry\":\"mesh\"}", "'geometry'" },
    };
    for (cases) |c| try testing.expect(std.mem.indexOf(u8, (try parsed(c[0])).bad.why.slice(), c[1]) != null);
}

test "idol: a job on the tiny pack is a splat GLB of the requested size; bad requests refuse by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const one = [_]abi.Str{.of(pack.fx.inputs[0])};
    // The default is the disc mesh: triangles and an atlas, no splat extension.
    const discs = (try pack.runInputs(pack.fx.body, &one)).ok;
    try testing.expectEqualStrings("glTF", discs[0..4]);
    const djson = discs[20 .. 20 + std.mem.readInt(u32, discs[12..16], .little)];
    try testing.expect(std.mem.indexOf(u8, djson, "\"mode\":4") != null and std.mem.indexOf(u8, djson, "KHR_gaussian_splatting") == null);
    // `geometry: "splat"` is the extension on one POINTS primitive of the requested size.
    const glb = (try pack.runInputs("{\"image\":\"x\",\"steps\":2,\"seed\":3,\"num_gaussians\":32768,\"geometry\":\"splat\"}", &one)).ok;
    try testing.expectEqualStrings("glTF", glb[0..4]);
    const jlen = std.mem.readInt(u32, glb[12..16], .little);
    try testing.expect(std.mem.indexOf(u8, glb[20 .. 20 + jlen], "KHR_gaussian_splatting") != null);
    try testing.expect(std.mem.indexOf(u8, glb[20 .. 20 + jlen], "\"count\":32768") != null);
    try testing.expectEqualStrings("missing 'image' (base64 PNG/JPEG of the subject)", (try pack.runInputs("{}", &.{})).refused.why.slice());
    try testing.expectEqualStrings("'steps' must be a whole number in [1,200]", (try pack.runInputs("{\"image\":\"x\",\"steps\":0}", &one)).refused.why.slice());
}

/// The tiny pack's denoiser on the GPU, with a step's stream and the unconditional (all-zero) image features.
const TinyDenoiser = struct {
    arena: std.heap.ArenaAllocator,
    tmp: testing.TmpDir,
    w: weights.Weights,
    d: den.Denoiser,
    s: mlx.mlx_stream,
    sc: scope.Scope,
    noise: []f32,

    const n = 21;

    fn open(self: *TinyDenoiser) !void {
        const a = testing.allocator;
        self.arena = std.heap.ArenaAllocator.init(a);
        errdefer self.arena.deinit();
        self.tmp = testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        try self.tmp.dir.writeFile(testing.io, .{ .sub_path = "denoiser.safetensors", .data = try denoiserPack(self.arena.allocator()) });
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        self.w = try weights.loadWeightsFile(a, pbuf[0..try self.tmp.dir.realPath(testing.io, &pbuf)], "denoiser.safetensors");
        errdefer self.w.deinit();
        try layers.castLinears(a, &self.w, &.{ "blocks.", "noise_refiner.", "context_refiner." }, .float16);
        self.d = try den.Denoiser.load(a, &self.w, .{ .q_token_length = 64, .model_channels = 128, .cond_channels = DINO, .num_blocks = 1, .num_refiner_blocks = 1, .num_heads = HEADS }, .float16);
        self.s = mlx.gpuStream();
        self.sc = scope.Scope.init(a, self.s);
        var rng = std.Random.DefaultPrng.init(5);
        self.noise = try self.arena.allocator().alloc(f32, n * DINO + 64 * 16 + 5);
        for (self.noise) |*v| v.* = rng.random().floatNorm(f32);
    }

    fn close(self: *TinyDenoiser) void {
        self.sc.deinit();
        _ = mlx.mlx_stream_free(self.s);
        self.d.deinit();
        self.w.deinit();
        self.tmp.cleanup();
        self.arena.deinit();
    }

    fn stream(self: *TinyDenoiser) !den.Denoiser.Stream {
        const sc = &self.sc;
        return self.d.stream(sc, try sc.fromF32(self.noise[n * DINO ..][0 .. 64 * 16], &.{ 1, 64, 16 }), try sc.fromF32(self.noise[self.noise.len - 5 ..], &.{ 1, 1, 5 }), 700);
    }

    fn zeros(self: *TinyDenoiser) ![2]mlx.mlx_array {
        return .{ try self.sc.zeros(&.{ 1, n, DINO }, .float32), try self.sc.zeros(&.{ 1, n, 128 }, .float32) };
    }
};

test "idol: identical image tokens run as two keys with a summed bias and keep the velocity; distinct ones stay" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    var t: TinyDenoiser = undefined;
    try t.open();
    defer t.close();
    const d = &t.d;
    const sc = &t.sc;
    const s = t.s;
    const n = TinyDenoiser.n;
    const z1, const z2 = try t.zeros();
    const st = try t.stream();

    const short = try d.context(sc, z1, z2);
    try testing.expectEqual(@as(c_int, 2), mlx.getShape(short.tokens)[1]);
    const full = blk: {
        den.collapse_disabled_for_test = true;
        defer den.collapse_disabled_for_test = false;
        break :blk try d.context(sc, z1, z2);
    };
    try testing.expectEqual(@as(c_int, n), mlx.getShape(full.tokens)[1]);
    const vs = try d.joint(sc, st, short);
    const vf = try d.joint(sc, st, full);
    for ([_][2]mlx.mlx_array{ .{ vs.latent, vf.latent }, .{ vs.camera, vf.camera } }) |pair| {
        const g = try w8a8.agreement(pair[0], pair[1], s);
        try testing.expect(g.cos > 0.99999);
        try testing.expectApproxEqAbs(@as(f64, 1), g.rms_ratio, 1e-3);
    }
    const distinct = try d.context(sc, try sc.fromF32(t.noise[0 .. n * DINO], &.{ 1, n, DINO }), z2);
    try testing.expectEqual(@as(c_int, n), mlx.getShape(distinct.tokens)[1]);
    try testing.expect(distinct.bias.mask.ctx == null and distinct.bias.base2.ctx == null);
}

test "idol: under --w8a8 the attention runs int8, the collapsed tokens' key bias in it, close to sdpa" {
    if (!@import("../../mlx/device.zig").naxAvailable() or mlx.virtualGpu()) return error.SkipZigTest;
    var t: TinyDenoiser = undefined;
    try t.open();
    defer t.close();
    const z1, const z2 = try t.zeros();
    const st = try t.stream();
    const short = try t.d.context(&t.sc, z1, z2);
    try testing.expect(short.bias.base2.ctx != null);
    const prev = w8a8.w8a8State().enabled;
    defer w8a8.w8a8State().enabled = prev;
    var v: [2]den.Denoiser.Velocity = undefined;
    for (&v, 0..) |*out, arm| {
        w8a8.w8a8State().enabled = arm == 1;
        out.* = try t.d.joint(&t.sc, st, short);
        try mlx.check(mlx.mlx_array_eval(out.latent));
    }
    const err = try w8a8.relRms(v[1].latent, v[0].latent, t.s);
    try testing.expect(err > 0); // the int8 route engaged: sdpa twice is identical
    try testing.expect(err < 0.01);
}

test "idol: DINOv3 loaded in bf16 runs in bf16 and keeps the f32 encoder's tokens" {
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const a = testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "config.json", .data = tiny_dino });
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "model.safetensors", .data = try dinoPack(arena.allocator()) });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = pbuf[0..try tmp.dir.realPath(testing.io, &pbuf)];
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = scope.Scope.init(a, s);
    defer sc.deinit();
    var rng = std.Random.DefaultPrng.init(9);
    const px = try arena.allocator().alloc(f32, 64 * 64 * 3);
    for (px) |*v| v.* = rng.random().floatNorm(f32);
    const pixels = try sc.fromF32(px, &.{ 1, 64, 64, 3 });
    var out: [2]mlx.mlx_array = undefined;
    for (&out, [_]mlx.mlx_dtype{ .float32, .bfloat16 }) |*o, dt| {
        var e = try dinov3.Encoder.open(testing.io, a, dir, dt);
        defer e.deinit();
        o.* = try e.forward(&sc, pixels);
        try mlx.check(mlx.mlx_array_eval(o.*));
        try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(o.*));
    }
    const g = try w8a8.agreement(out[1], out[0], s);
    try testing.expect(g.cos < 0.9999999); // bf16 engaged: an f32 run would be identical
    try testing.expect(g.cos > 0.999);
}
