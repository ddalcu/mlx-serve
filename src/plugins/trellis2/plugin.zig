//! TRELLIS.2-4B on the media table: one RGBA cutout to a textured PBR GLB (`.mesh`). The GPU half decodes the
//! voxels (every model loaded for its stage and freed), the CPU half meshes, unwraps and bakes on a worker.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const mesh_request = @import("../../sdk/media/mesh_request.zig");
const mlx = @import("../../mlx/ffi.zig");
const tp = @import("pipeline.zig");
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
    m: *tp.Engine,
    glb: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const m = tp.Engine.loadOn(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch |err| {
        out.* = job.failure(err, "trellis2: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        m.deinit();
        out.* = Verdict.refuse(.failed, "trellis2: out of memory");
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

fn int(body: *const job.Body, key: []const u8) ?i64 {
    const v = body.get(key) orelse return null;
    return if (v == .integer) v.integer else null;
}

pub const Parsed = union(enum) { ok: tp.Opts, bad: Verdict };

/// `resolution` (512 only where the pack ships its 512 texture flow, else 1024 or 1536), `steps` (omitted keeps the
/// published 12), `seed`, `decimation_target` (faces), `texture_size`, and the finish: `roughness_scale`, `clearcoat`.
pub fn parseOpts(body: *const job.Body, has_512: bool) Parsed {
    var o = tp.Opts{ .seed = body.uint("seed") orelse 42 };
    if (int(body, "resolution")) |r| {
        if (r < 0 or r > std.math.maxInt(u32) or !tp.resolutionServed(@intCast(r), has_512))
            return .{ .bad = if (has_512) bad("'resolution' must be 512, 1024 or 1536") else bad("'resolution' must be 1024 or 1536 (this pack has no 512 texture model)") };
        o.resolution = @intCast(r);
    }
    if (int(body, "steps")) |n| {
        if (n < 1 or n > 50) return .{ .bad = bad("'steps' must be in [1,50]") };
        o.steps = @intCast(n);
    }
    if (int(body, "decimation_target")) |f| {
        if (f < 10_000 or f > 2_000_000) return .{ .bad = bad("'decimation_target' must be in [10000,2000000] faces") };
        o.decimation_target = @intCast(f);
    }
    if (int(body, "texture_size")) |t| {
        if (t != 512 and t != 1024 and t != 2048 and t != 4096) return .{ .bad = bad("'texture_size' must be 512, 1024, 2048 or 4096") };
        o.texture_size = @intCast(t);
    }
    if (body.float("roughness_scale")) |r| {
        if (!(r >= 0.05 and r <= 2)) return .{ .bad = bad("'roughness_scale' multiplies the model's roughness and must be in [0.05,2]") };
        o.roughness_scale = @floatCast(r);
    }
    if (body.float("clearcoat")) |c| {
        if (!(c >= 0 and c <= 1)) return .{ .bad = bad("'clearcoat' must be in [0,1]") };
        o.clearcoat = @floatCast(c);
    }
    return .{ .ok = o };
}

/// A resolution whose peak exceeds what the GPU can still take, refused by name with both numbers.
pub fn resolutionRefusal(req: *const abi.Request, resolution: u32) ?Verdict {
    const need = config.peakBytes(resolution);
    const room = job.gpuHeadroom();
    if (need <= room) return null;
    const gb: f64 = 1 << 30;
    return req.refuseFmt(.unsupported, "this resolution needs more GPU memory than is free: send a lower 'resolution'", "resolution {d} needs ~{d:.1} GB of free GPU memory but {d:.1} GB is free: send a lower 'resolution' (1024 needs ~{d:.0} GB)", .{
        resolution, @as(f64, @floatFromInt(need)) / gb, @as(f64, @floatFromInt(room)) / gb, @as(f64, @floatFromInt(config.peakBytes(1024))) / gb,
    });
}

/// A failed mesh job: the client left, a subject or structure the model cannot use (by name), or the machine.
pub fn jobFailure(err: anyerror) Verdict {
    return switch (err) {
        error.Cancelled => job.client_left,
        error.EmptySubject => bad("the image has no subject: its alpha channel is empty (send a cutout on a transparent background, or a photo whose background can be cut away)"),
        error.EmptyStructure => bad("no occupied voxels came out of this image: try another seed or a clearer, larger subject"),
        else => job.failure(err, "generation failed"),
    };
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    var views: mesh_request.Views = .{};
    defer views.deinit(gpa);
    if (mesh_request.decode(gpa, &body.parsed.value.object, req.inputs.slice(), 1, &views)) |v| return v;
    const img = views.list.items[0];
    const opts = switch (parseOpts(&body, e.m.hasResolution512())) {
        .ok => |o| o,
        .bad => |v| return v,
    };
    if (resolutionRefusal(req, opts.resolution)) |v| return v;
    const out = e.m.generateGlb(gpa, img.pix, img.w, img.h, opts, job.progressOf(progress)) catch |err| return jobFailure(err);
    if (job.latched()) |v| {
        gpa.free(out);
        return v;
    }
    gpa.free(e.glb);
    e.glb = out;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.glb.len == 0) return Verdict.refuse(.failed, "trellis2: no finished job to encode");
    out.* = .of(e.glb);
    return .ok;
}

// ── the tiny pack ──

const T = sdk.tensors.Tensors;

fn lin(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64, dt: sdk.tensors.Dtype) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in }, dt, 0.8 / @sqrt(@as(f32, @floatFromInt(in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, dt, 0.05);
}

fn norm(t: *T, comptime fmt: []const u8, args: anytype, c: u64) !void {
    try t.fill(fmt ++ ".weight", args, &.{c}, .F32, 1);
    try t.fill(fmt ++ ".bias", args, &.{c}, .F32, 0);
}

/// A flow DiT: 32 wide (two heads of 16), one block, conditioned on 32-wide image tokens; blocks bf16 as published.
/// `proj` (Pixal3D) nests the cross-attention under `cross_attn_block` beside its projection of `proj` features.
pub fn flowDit(t: *T, in: u64, out: u64, comptime proj: ?u64) !void {
    const c = 32;
    try lin(t, "input_layer", .{}, c, in, .F32);
    try lin(t, "t_embedder.mlp.0", .{}, c, 256, .F32);
    try lin(t, "t_embedder.mlp.2", .{}, c, c, .F32);
    try lin(t, "adaLN_modulation.1", .{}, 6 * c, c, .F32);
    try lin(t, "out_layer", .{}, out, c, .F32);
    const b = "blocks.0.";
    try t.rand(b ++ "modulation", .{}, &.{6 * c}, .BF16, 0.1);
    try t.fill(b ++ "norm2.weight", .{}, &.{c}, .BF16, 1);
    try t.fill(b ++ "norm2.bias", .{}, &.{c}, .BF16, 0);
    try lin(t, b ++ "self_attn.to_qkv", .{}, 3 * c, c, .BF16);
    try lin(t, b ++ "self_attn.to_out", .{}, c, c, .BF16);
    const cross = if (proj != null) b ++ "cross_attn.cross_attn_block." else b ++ "cross_attn.";
    try lin(t, cross ++ "to_q", .{}, c, c, .BF16);
    try lin(t, cross ++ "to_kv", .{}, 2 * c, 32, .BF16);
    try lin(t, cross ++ "to_out", .{}, c, c, .BF16);
    inline for (.{ b ++ "self_attn.q_rms_norm", b ++ "self_attn.k_rms_norm", cross ++ "q_rms_norm", cross ++ "k_rms_norm" }) |n|
        try t.fill(n ++ ".gamma", .{}, &.{ 2, 16 }, .BF16, 1);
    if (proj) |p| try lin(t, b ++ "cross_attn.proj_linear", .{}, c, p, .BF16);
    try lin(t, b ++ "mlp.mlp.0", .{}, 64, c, .BF16);
    try lin(t, b ++ "mlp.mlp.2", .{}, c, 64, .BF16);
}

pub const flow_json = "{\"args\":{\"pe_mode\":\"rope\",\"share_mod\":true,\"qk_rms_norm\":true,\"qk_rms_norm_cross\":true}}";
const flow_json_proj = "{\"args\":{\"pe_mode\":\"rope\",\"share_mod\":true,\"qk_rms_norm\":true,\"qk_rms_norm_cross\":true,\"image_attn_mode\":\"proj\"}}";
pub const decoder_json = "{\"args\":{\"model_channels\":[16,8],\"num_blocks\":[1,1]}}";

fn conv3(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, 3, 3, 3, in }, .F32, 0.8 / @sqrt(@as(f32, @floatFromInt(27 * in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .F32, 0.05);
}

fn convNext(t: *T, comptime fmt: []const u8, args: anytype, c: u64) !void {
    try conv3(t, fmt ++ ".conv", args, c, c);
    try norm(t, fmt ++ ".norm", args, c);
    try lin(t, fmt ++ ".mlp.0", args, 4 * c, c, .F32);
    try lin(t, fmt ++ ".mlp.2", args, c, 4 * c, .F32);
}

/// A sparse SC-VAE decoder at 16 then 8 channels: one subdivision. The shape decoder predicts it (7 outputs), the
/// texture decoder follows it (6).
pub fn sparseDecoder(t: *T, outputs: u64, predicts: bool) !void {
    try lin(t, "from_latent", .{}, 16, 32, .F32);
    try convNext(t, "blocks.0.0", .{}, 16);
    try norm(t, "blocks.0.1.norm1", .{}, 16);
    try conv3(t, "blocks.0.1.conv1", .{}, 64, 16);
    try conv3(t, "blocks.0.1.conv2", .{}, 8, 8);
    if (predicts) try lin(t, "blocks.0.1.to_subdiv", .{}, 8, 16, .F32);
    try convNext(t, "blocks.1.0", .{}, 8);
    try lin(t, "output_layer", .{}, outputs, 8, .F32);
}

fn dense3(t: *T, comptime fmt: []const u8, args: anytype, out: u64, in: u64, scale: f32) !void {
    try t.rand(fmt ++ ".weight", args, &.{ out, in, 3, 3, 3 }, .F32, scale / @sqrt(@as(f32, @floatFromInt(27 * in))));
    try t.rand(fmt ++ ".bias", args, &.{out}, .F32, 0.05);
}

fn ssRes(t: *T, comptime fmt: []const u8, args: anytype, c: u64) !void {
    try norm(t, fmt ++ ".norm1", args, c);
    try dense3(t, fmt ++ ".conv1", args, c, c, 0.8);
    try norm(t, fmt ++ ".norm2", args, c);
    try dense3(t, fmt ++ ".conv2", args, c, c, 0.8);
}

/// The sparse-structure decoder: 16^3 x 8 to 64^3 logits through two pixel shuffles; its output bias keeps a few
/// hundred cells occupied.
pub fn ssDecoder(t: *T) !void {
    try dense3(t, "input_layer", .{}, 8, 8, 0.8);
    for (0..2) |i| try ssRes(t, "middle_block.{d}", .{i}, 8);
    try ssRes(t, "blocks.0", .{}, 8);
    try ssRes(t, "blocks.1", .{}, 8);
    try dense3(t, "blocks.2.conv", .{}, 32, 8, 0.8);
    try ssRes(t, "blocks.3", .{}, 4);
    try ssRes(t, "blocks.4", .{}, 4);
    try dense3(t, "blocks.5.conv", .{}, 32, 4, 0.8);
    try ssRes(t, "blocks.6", .{}, 4);
    try ssRes(t, "blocks.7", .{}, 4);
    try norm(t, "out_layer.0", .{}, 4);
    try t.rand("out_layer.2.weight", .{}, &.{ 1, 4, 3, 3, 3 }, .F32, 0.3);
    try t.fill("out_layer.2.bias", .{}, &.{1}, .F32, -4);
}

/// DINOv3 at 32 wide, one layer, two heads of 16, one register.
pub fn dino(t: *T) !void {
    try t.rand("embeddings.cls_token", .{}, &.{ 1, 1, 32 }, .F32, 0.5);
    try t.rand("embeddings.register_tokens", .{}, &.{ 1, 1, 32 }, .F32, 0.5);
    try t.rand("embeddings.patch_embeddings.weight", .{}, &.{ 32, 3, 16, 16 }, .F32, 0.05);
    try t.rand("embeddings.patch_embeddings.bias", .{}, &.{32}, .F32, 0.05);
    const l = "layer.0.";
    try norm(t, l ++ "norm1", .{}, 32);
    try norm(t, l ++ "norm2", .{}, 32);
    inline for (.{ "q_proj", "v_proj", "o_proj" }) |n| try lin(t, l ++ "attention." ++ n, .{}, 32, 32, .F32);
    try t.rand(l ++ "attention.k_proj.weight", .{}, &.{ 32, 32 }, .F32, 0.14);
    try t.fill(l ++ "layer_scale1.lambda1", .{}, &.{32}, .F32, 0.5);
    try t.fill(l ++ "layer_scale2.lambda1", .{}, &.{32}, .F32, 0.5);
    try lin(t, l ++ "mlp.up_proj", .{}, 64, 32, .F32);
    try lin(t, l ++ "mlp.down_proj", .{}, 32, 64, .F32);
}

pub const dino_json = "{\"hidden_size\":32,\"intermediate_size\":64,\"num_hidden_layers\":1,\"num_attention_heads\":2,\"patch_size\":16,\"num_register_tokens\":1}";

/// A 64x64 cutout: an opaque disc on a transparent background.
pub fn cutout(a: std.mem.Allocator) ![]u8 {
    const n = 64;
    const px = try a.alloc(u8, n * n * 4);
    for (0..n) |y| for (0..n) |x| {
        const dx = @as(f32, @floatFromInt(x)) - 31.5;
        const dy = @as(f32, @floatFromInt(y)) - 31.5;
        const inside = dx * dx + dy * dy < 22 * 22;
        px[(y * n + x) * 4 ..][0..4].* = if (inside) .{ @intCast(x * 4), 200, @intCast(y * 4), 255 } else .{ 0, 0, 0, 0 };
    };
    return @import("../../sdk/media/png.zig").encodeRgba(a, px, n, n);
}

pub fn normJson(a: std.mem.Allocator) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.writeAll("{\"mean\":[");
    for (0..32) |i| try w.writer.print("{s}{d}", .{ if (i == 0) "" else ",", @as(f32, @floatFromInt(i % 5)) * 0.01 });
    try w.writer.writeAll("],\"std\":[");
    for (0..32) |i| try w.writer.print("{s}{d}", .{ if (i == 0) "" else ",", 1.0 + @as(f32, @floatFromInt(i % 3)) * 0.1 });
    try w.writer.writeAll("]}");
    return w.toOwnedSlice();
}

/// The tiny pack's files in the published layout, every model at the smallest width its code takes: DINOv3, the SS
/// flow and decoder, the shape and texture flows (the 1024 names share the 512 bytes), and the two sparse decoders
/// with one subdivision. Pixal3D's (`pixal`) flows project the pixel features (32 wide off the patch map, 64 with
/// NAF), its SS flow says `proj`, its texture decoder keeps its own name, and it has no 512 texture flow.
pub fn tinyPack(a: std.mem.Allocator, comptime pixal: bool) !std.ArrayList(sdk.fixture.File) {
    var ss_flow = T.init(a, 61);
    try flowDit(&ss_flow, 8, 8, if (pixal) 32 else null);
    var shape_flow = T.init(a, 62);
    try flowDit(&shape_flow, 32, 32, if (pixal) 64 else null);
    var tex_flow = T.init(a, 63);
    try flowDit(&tex_flow, 64, 32, if (pixal) 64 else null);
    var ss_dec = T.init(a, 64);
    try ssDecoder(&ss_dec);
    var shape_dec = T.init(a, 65);
    try sparseDecoder(&shape_dec, 7, true);
    var tex_dec = T.init(a, 66);
    try sparseDecoder(&tex_dec, 6, false);
    var dn = T.init(a, 67);
    try dino(&dn);
    const tex_dec_name = if (pixal) "ckpts/tex_dec_next_dc_f16c32_fp16" else "ckpts/tex_dec";
    const tex512 = if (pixal) "" else ",\"tex_slat_flow_model_512\":\"ckpts/tex_flow_512\"";
    const nj = try normJson(a);
    const pipe = try std.fmt.allocPrint(a,
        \\{{"name":"Trellis2ImageTo3DPipeline","args":{{"models":{{"sparse_structure_decoder":"microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16","sparse_structure_flow_model":"ckpts/ss_flow","shape_slat_decoder":"ckpts/shape_dec","shape_slat_flow_model_512":"ckpts/shape_flow_512","shape_slat_flow_model_1024":"ckpts/shape_flow_1024","tex_slat_decoder":"{s}"{s},"tex_slat_flow_model_1024":"ckpts/tex_flow_1024"}},"shape_slat_normalization":{s},"tex_slat_normalization":{s}}}}}
    , .{ tex_dec_name, tex512, nj, nj });
    const shape_bytes = try shape_flow.bytes();
    const tex_bytes = try tex_flow.bytes();
    var files: std.ArrayList(sdk.fixture.File) = .empty;
    try files.appendSlice(a, &.{
        .{ .name = "pipeline.json", .bytes = pipe },
        .{ .name = "dinov3/config.json", .bytes = dino_json },
        .{ .name = "dinov3/model.safetensors", .bytes = try dn.bytes() },
        .{ .name = "ckpts/ss_flow.json", .bytes = if (pixal) flow_json_proj else flow_json },
        .{ .name = "ckpts/ss_flow.safetensors", .bytes = try ss_flow.bytes() },
        .{ .name = "ckpts/shape_flow_512.json", .bytes = flow_json },
        .{ .name = "ckpts/shape_flow_512.safetensors", .bytes = shape_bytes },
        .{ .name = "ckpts/shape_flow_1024.json", .bytes = flow_json },
        .{ .name = "ckpts/shape_flow_1024.safetensors", .bytes = shape_bytes },
        .{ .name = "ckpts/tex_flow_1024.json", .bytes = flow_json },
        .{ .name = "ckpts/tex_flow_1024.safetensors", .bytes = tex_bytes },
        .{ .name = "ckpts/shape_dec.json", .bytes = decoder_json },
        .{ .name = "ckpts/shape_dec.safetensors", .bytes = try shape_dec.bytes() },
        .{ .name = tex_dec_name ++ ".json", .bytes = decoder_json },
        .{ .name = tex_dec_name ++ ".safetensors", .bytes = try tex_dec.bytes() },
        .{ .name = config.required_file, .bytes = try ss_dec.bytes() },
    });
    if (!pixal) try files.appendSlice(a, &.{
        .{ .name = "ckpts/tex_flow_512.json", .bytes = flow_json },
        .{ .name = "ckpts/tex_flow_512.safetensors", .bytes = tex_bytes },
    });
    return files;
}

/// The job runs the 512 pipeline at one step, a 512 atlas and the least face target.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    var files = try tinyPack(a, false);
    return .{
        .files = try files.toOwnedSlice(a),
        .body = "{\"image\":\"x\",\"resolution\":512,\"steps\":1,\"texture_size\":512,\"decimation_target\":10000,\"seed\":3}",
        .inputs = try a.dupe([]const u8, &.{try cutout(a)}),
    };
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack;
const golden = @import("../../sdk/testing/golden.zig");

fn parsed(json: []const u8, has_512: bool) !Parsed {
    var b = job.Body.parse(testing.allocator, json).?;
    defer b.deinit();
    return parseOpts(&b, has_512);
}

test "trellis2: request fields take their defaults and bounds, and a bad value names its field" {
    const d = (try parsed("{}", true)).ok;
    try testing.expectEqual(@as(u32, 1024), d.resolution);
    try testing.expectEqual(@as(u32, 0), d.steps); // the published 12
    try testing.expectEqual(@as(u64, 42), d.seed);
    try testing.expectEqual(@as(f32, 1.0), d.roughness_scale);
    try testing.expectEqual(@as(f32, 0.0), d.clearcoat);
    const o = (try parsed("{\"resolution\":1536,\"steps\":8,\"seed\":7,\"decimation_target\":50000,\"texture_size\":4096,\"roughness_scale\":0.4,\"clearcoat\":1}", false)).ok;
    try testing.expectEqual(@as(u32, 1536), o.resolution);
    try testing.expectEqual(@as(u32, 8), o.steps);
    try testing.expectEqual(@as(u32, 50000), o.decimation_target);
    try testing.expectEqual(@as(u32, 4096), o.texture_size);
    try testing.expectApproxEqAbs(@as(f32, 0.4), o.roughness_scale, 1e-6);
    try testing.expectEqual(@as(f32, 1.0), o.clearcoat);
    try testing.expectEqual(@as(u32, 512), (try parsed("{\"resolution\":512}", true)).ok.resolution);
    const bads = [_][2][]const u8{
        .{ "{\"resolution\":512}", "no 512 texture model" },
        .{ "{\"steps\":51}", "'steps'" },
        .{ "{\"decimation_target\":10}", "'decimation_target'" },
        .{ "{\"texture_size\":1000}", "'texture_size'" },
        .{ "{\"roughness_scale\":0}", "'roughness_scale'" },
        .{ "{\"roughness_scale\":2.5}", "'roughness_scale'" },
        .{ "{\"clearcoat\":-0.1}", "'clearcoat'" },
    };
    for (bads) |c| try testing.expect(std.mem.indexOf(u8, (try parsed(c[0], false)).bad.why.slice(), c[1]) != null);
}

test "trellis2: a job whose GPU stage latched a failure is refused by class, never a GLB of zeros" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const in = [_]abi.Str{.of(pack.fx.inputs[0])};
    var stale: [512]u8 = undefined;
    _ = mlx.takeError(&stale);
    defer _ = mlx.takeError(&stale);
    mlx.latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at array.cpp:352");
    const v = (try pack.runInputs(pack.fx.body, &in)).refused;
    try testing.expectEqual(Verdict.Code.busy, v.code);
    try testing.expectEqualStrings("generation failed", v.why.slice());
}

test "trellis2: a job on the tiny pack is a GLB; other views, fields and resolutions are refused by name" {
    if (mlx.virtualGpu()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const in = [_]abi.Str{.of(pack.fx.inputs[0])};
    const glb = (try pack.runInputs(pack.fx.body, &in)).ok;
    try testing.expectEqualStrings("glTF", glb[0..4]);
    const cases = [_][2][]const u8{
        .{ "{}", "missing 'image' (base64 PNG/JPEG of the subject)" },
        .{ "{\"images\":[\"x\",\"y\"]}", "'images' (several views of one object) is served by Pixal3D; this model takes one 'image'" },
        .{ "{\"image\":\"x\",\"resolution\":768}", "'resolution' must be 512, 1024 or 1536" },
        .{ "{\"image\":\"x\",\"steps\":0}", "'steps' must be in [1,50]" },
        .{ "{\"image\":\"x\",\"clearcoat\":2}", "'clearcoat' must be in [0,1]" },
    };
    for (cases) |c| try testing.expectEqualStrings(c[1], (try pack.runInputs(c[0], &in)).refused.why.slice());
}

/// The voxels the GPU half decodes, as arrays a golden compares.
pub fn voxelArrays(v: *const tp.Voxels) [3]golden.Named {
    const n: c_int = @intCast(v.coords.len / 4);
    return .{
        .{ .name = "coords", .array = mlx.mlx_array_new_data(v.coords.ptr, &[_]c_int{ n, 4 }, 2, .int32) },
        .{ .name = "raw", .array = mlx.mlx_array_new_data(v.raw.ptr, &[_]c_int{ n, 7 }, 2, .float32) },
        .{ .name = "attrs", .array = mlx.mlx_array_new_data(v.attrs.ptr, &[_]c_int{ n, 6 }, 2, .float32) },
    };
}

test "trellis2: the tiny pack's voxels (structure, shape, texture) are bit-equal to its golden run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    const img = image_pixels.decodeRgba(gpa, pack.fx.inputs[0]).?;
    defer img.deinit(gpa);
    var vox = try e.m.generateVoxels(gpa, img.pix, img.w, img.h, .{ .resolution = 512, .steps = 2, .seed = 3 }, null);
    defer vox.deinit(gpa);
    const arrays = voxelArrays(&vox);
    defer for (arrays) |x| {
        _ = mlx.mlx_array_free(x.array);
    };
    try golden.expect("trellis2_voxels", &arrays);
}
