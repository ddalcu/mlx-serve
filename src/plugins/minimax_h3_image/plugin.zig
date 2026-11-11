//! MiniMax-H3-Image on the media table (`/v1/images/generations`, `/v1/images/edits`): text to image and multi-reference
//! editing (`image` + `ref_images`, `<Picture i>` in the prompt) over H3's text encoder and DiT, a six-step PDD student,
//! and the release's 2D image VAE. A request smaller than `min_generation_area` is generated on a larger canvas at its
//! aspect and resized back. The result is a PNG.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const log = @import("../../sdk/util/log.zig");
const mlx = @import("../../mlx/ffi.zig");
const image_request = @import("../../sdk/media/image_request.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const qvis = @import("../../sdk/vision/qwen_vision.zig");
const frame_in = @import("../../sdk/media/frame_in.zig");
const h3_plugin = @import("../minimax_h3/plugin.zig");
const h3_bill = @import("../minimax_h3/config.zig");
const h3v = @import("../minimax_h3/vision.zig");
const image = @import("image.zig");
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
const A = mlx.mlx_array;

/// Edit references keep their own size within these short-edge bounds (the VAE), the tower sees them at 1024.
const REF_SHORT_EDGE = [2]u32{ 1024, 2048 };
const VISION_SHORT_EDGE: u32 = 1024;

const Engine = struct {
    model_dir: []u8,
    stream: mlx.mlx_stream,
    owns_stream: bool,
    machine: abi.Machine,
    geometry: config.Geometry,
    png: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    var buf: [1 << 20]u8 = undefined;
    var why: Verdict = .{};
    const cfg = ctx.dir.readFile("config.json", &buf, &why) orelse "{}";
    const geometry = config.geometry(gpa, cfg) catch |err| {
        out.* = Verdict.named(.failed, @errorName(err));
        return null;
    };
    const e = gpa.create(Engine) catch {
        out.* = Verdict.refuse(.failed, "minimax_h3_image: out of memory");
        return null;
    };
    const owns = ctx.stream.ctx == null;
    e.* = .{
        .model_dir = gpa.dupe(u8, ctx.model_dir.slice()) catch {
            gpa.destroy(e);
            out.* = Verdict.refuse(.failed, "minimax_h3_image: out of memory");
            return null;
        },
        .stream = if (owns) mlx.mlx_default_gpu_stream_new() else .{ .ctx = ctx.stream.ctx },
        .owns_stream = owns,
        .machine = if (ctx.machine) |m| m.* else .{},
        .geometry = geometry,
    };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    gpa.free(e.png);
    gpa.free(e.model_dir);
    if (e.owns_stream) _ = mlx.mlx_stream_free(e.stream);
    gpa.destroy(e);
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

/// One upload decoded for the VAE (its bounded size, cropped top-left to whole 16-pixel cells) and for the tower.
fn decodeRef(bytes: []const u8, s: mlx.mlx_stream) ?image.Ref {
    const nat = image_pixels.nativeSize(bytes) orelse return null;
    const vae_sz = config.boundShortEdge(nat.w, nat.h, REF_SHORT_EDGE[0], REF_SHORT_EDGE[1]);
    const full = frame_in.pm1(gpa, bytes, vae_sz.h, vae_sz.w, .float32, s) orelse return null;
    defer _ = mlx.mlx_array_free(full);
    var px = mlx.mlx_array_new();
    const ch: c_int = @intCast(vae_sz.h / 16 * 16);
    const cw: c_int = @intCast(vae_sz.w / 16 * 16);
    if (mlx.mlx_slice(&px, full, &[_]c_int{ 0, 0, 0, 0, 0 }, 5, &[_]c_int{ 1, 3, 1, ch, cw }, 5, &[_]c_int{ 1, 1, 1, 1, 1 }, 5, s) != 0) {
        _ = mlx.mlx_array_free(px);
        return null;
    }
    const vb = config.boundShortEdge(nat.w, nat.h, VISION_SHORT_EDGE, VISION_SHORT_EDGE);
    const vfit = h3v.fitCanvas(vb.h, vb.w);
    const vision = frame_in.pm1(gpa, bytes, vfit.h, vfit.w, .bfloat16, s) orelse {
        _ = mlx.mlx_array_free(px);
        return null;
    };
    return .{ .pixels = px, .vision = vision };
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    const dir = req.dir orelse return Verdict.refuse(.failed, "minimax_h3_image: the host gave no pack dir");
    var body = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    var inputs: [config.edit_ref_cap][]const u8 = undefined;
    const in = req.inputs.slice();
    const spec: image_request.Spec = .{ .edit = true, .edit_ref_cap = config.edit_ref_cap, .too_many_refs = "too many reference images (MiniMax-H3-Image takes at most 16, 'image' included)" };
    if (in.len > inputs.len) return Verdict.named(.unsupported, spec.too_many_refs);
    for (in, 0..) |x, i| inputs[i] = x.slice();
    const j = switch (image_request.parse(.{ .obj = body.parsed.value.object }, inputs[0..in.len], spec)) {
        .ok => |j| j,
        .bad => |v| return v,
    };
    if (j.mode == .variation) return bad("image-to-image (mode:\"variation\") is not available for MiniMax-H3-Image — send mode:\"edit\" with an instruction");
    const n_steps = e.geometry.n_sigmas - 1;
    if (j.steps != 0 and j.steps != n_steps) return req.refuseFmt(.unsupported, "this pack runs a fixed step count — omit 'steps'", "this pack is a distilled student that runs exactly {d} steps — omit 'steps' or send {d}", .{ n_steps, n_steps });
    var min_area = config.DEFAULT_MIN_AREA;
    if (body.get("min_generation_area")) |v| if (v != .null) {
        const n = body.uint("min_generation_area") orelse return bad("'min_generation_area' must be an integer in [1024, 16777216]");
        if (n < config.MIN_AREA_RANGE[0] or n > config.MIN_AREA_RANGE[1]) return bad("'min_generation_area' must be an integer in [1024, 16777216]");
        min_area = @intCast(n);
    };

    // An edit keeps the primary reference's aspect: at the requested budget, or at its own size.
    var out_w = j.req_w;
    var out_h = j.req_h;
    var srcs: [config.edit_ref_cap][]const u8 = undefined;
    var n_refs: usize = 0;
    if (j.mode == .edit) {
        srcs[0] = j.image.?;
        for (j.refs) |r| {
            n_refs += 1;
            srcs[n_refs] = r;
        }
        n_refs += 1;
        for (srcs[0..n_refs], 0..) |r, i| {
            const sz = image_pixels.nativeSize(r) orelse return req.refuseFmt(.unsupported, "a reference image could not be decoded (PNG/JPEG/WebP supported)", "could not decode reference {d} (PNG/JPEG/WebP supported)", .{i + 1});
            if (@min(sz.w, sz.h) < config.CANVAS_MULTIPLE) return req.refuseFmt(.unsupported, "a reference image needs at least 32 pixels on each side", "reference {d} is {d}x{d}: a reference needs at least 32 pixels on each side", .{ i + 1, sz.w, sz.h });
        }
        const nat = image_pixels.nativeSize(srcs[0]).?;
        const fit = if (j.size_given)
            image_request.resolveEditTargetSize(nat.w, nat.h, out_w, out_h, config.max_dim)
        else
            image_request.fitWithinCap(nat.w, nat.h, config.max_dim);
        out_w = fit.w;
        out_h = fit.h;
    }
    if (out_w < 16 or out_h < 16 or out_w > config.max_dim or out_h > config.max_dim) return bad("'size' sides must be in [16, 6272]");
    const canvas = config.canvasFor(out_w, out_h, min_area);
    if (canvas.w > config.max_dim or canvas.h > config.max_dim) return bad("the generation canvas passes 6272 on a side: lower 'min_generation_area' or pick a less extreme aspect");

    // The DiT stage is priced on its own sequence before any stage loads: a Metal OOM mid-step is uncatchable.
    var ref_dims: [config.edit_ref_cap][2]u32 = undefined;
    for (srcs[0..n_refs], 0..) |r, i| {
        const nat = image_pixels.nativeSize(r).?;
        const b = config.boundShortEdge(nat.w, nat.h, REF_SHORT_EDGE[0], REF_SHORT_EDGE[1]);
        ref_dims[i] = .{ b.w, b.h };
    }
    const rows = config.requestRows(canvas.w, canvas.h, ref_dims[0..n_refs]);
    const activations = h3_bill.activationBytes(rows, 0);
    const mem = h3_plugin.memNow(req);
    const dit = h3_bill.ditResidentBytes(h3_bill.ditFileBytes(dir, &e.machine), true);
    const bill = h3_bill.canvasBill(dit, activations, mem.host_avail, mem.gpu_limit, mem.gpu_active, mlx.cudaAvailable());
    if (dit > 0 and bill.avail > 0 and bill.need > bill.avail) {
        const gb = 1024.0 * 1024.0 * 1024.0;
        return req.refuseFmt(.unsupported, "the canvas does not fit in memory: use a smaller size or fewer references", "a {d}x{d} canvas with {d} reference(s) needs ~{d:.1} GB ({d:.1} GB DiT + {d:.1} GB for {d} sequence rows) but {d:.1} GB is available: use a smaller size, a lower 'min_generation_area' or fewer references", .{
            canvas.w,                                   canvas.h,                                n_refs,
            @as(f64, @floatFromInt(bill.need)) / gb,    @as(f64, @floatFromInt(dit)) / gb,       @as(f64, @floatFromInt(activations)) / gb,
            rows,                                       @as(f64, @floatFromInt(bill.avail)) / gb,
        });
    }

    const s = e.stream;
    var refs: [config.edit_ref_cap]image.Ref = undefined;
    var decoded: usize = 0;
    defer for (refs[0..decoded]) |r| {
        _ = mlx.mlx_array_free(r.pixels);
        _ = mlx.mlx_array_free(r.vision);
    };
    for (srcs[0..n_refs], 0..) |r, i| {
        refs[i] = decodeRef(r, s) orelse return req.refuseFmt(.unsupported, "a reference image could not be decoded (PNG/JPEG/WebP supported)", "could not decode reference {d} (PNG/JPEG/WebP supported)", .{i + 1});
        decoded += 1;
    }
    if (canvas.w != out_w or canvas.h != out_h)
        log.info("[minimax-h3-image] {d}x{d} generated on a {d}x{d} canvas (min_generation_area {d})\n", .{ out_w, out_h, canvas.w, canvas.h, min_area });

    const io = std.Io.Threaded.global_single_threaded.io();
    const rgb = image.generate(gpa, io, .{ .dir = e.model_dir }, &e.geometry, .{
        .prompt = j.prompt,
        .width = canvas.w,
        .height = canvas.h,
        .seed = j.seed,
        .refs = refs[0..n_refs],
    }, job.progressOf(progress), s) catch |err| {
        if (err == error.Cancelled) return job.client_left;
        log.warn("[minimax-h3-image] generation failed: {s}\n", .{@errorName(err)});
        return job.failure(err, "MiniMax-H3-Image generation failed");
    };
    defer _ = mlx.mlx_array_free(rgb);
    const png = encodeOutput(gpa, rgb, out_w, out_h, s) catch |err| return job.failure(err, "MiniMax-H3-Image generation failed");
    gpa.free(e.png);
    e.png = png;
    return .ok;
}

/// The canvas as PNG at the requested size: truncated to u8 first and Lanczos-resized as the release's processor does.
fn encodeOutput(a: std.mem.Allocator, rgb: mlx.mlx_array, out_w: u32, out_h: u32, s: mlx.mlx_stream) ![]u8 {
    var p = try image_pixels.toPixels8(a, rgb, s);
    defer p.deinit(a);
    if (p.w != out_w or p.h != out_h) {
        const sized = try qvis.resizeInterleavedPil(a, p.px, p.h, p.w, out_h, out_w, p.channels, .lanczos);
        a.free(p.px);
        p = .{ .px = sized, .w = out_w, .h = out_h, .channels = p.channels };
    }
    return image_pixels.encodePng(a, p);
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.png.len == 0) return Verdict.refuse(.failed, "minimax_h3_image: no finished job to encode");
    out.* = .of(e.png);
    return .ok;
}

// ── the tiny pack ──

/// H3's tiny DiT without its audio stream and two PDD heads, H3's tiny text encoder, and a one-block 2D image VAE
/// decoder at 32; a 64x64 request generated as asked (`min_generation_area` at its floor), two steps on CPU.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    var dit = try h3_plugin.fixtureDit(a, false, 2);
    var te = try h3_plugin.fixtureTextEncoder(a);
    const vd = 32; // decoder width: 2 heads x 16
    var vae = sdk.tensors.Tensors.init(a, 104);
    try vae.rand("decoder.x_embedder.weight", .{}, &.{ vd, 24 }, .F32, 0.2);
    try vae.rand("decoder.x_embedder.bias", .{}, &.{vd}, .F32, 0.1);
    try vae.rand("decoder.register_tokens", .{}, &.{ 1, 4, vd }, .F32, 0.2);
    try vae.fill("decoder.norm_out.weight", .{}, &.{vd}, .F32, 1);
    try vae.fill("decoder.norm_out.bias", .{}, &.{vd}, .F32, 0);
    try vae.rand("decoder.proj_out.weight", .{}, &.{ 768, vd }, .F32, 0.2);
    try vae.rand("decoder.proj_out.bias", .{}, &.{768}, .F32, 0.1);
    try vae.rand("decoder.mask_token", .{}, &.{ 1, 1, vd }, .F32, 0.2);
    try vae.rand("post_quant_conv.weight", .{}, &.{ 24, 24, 1, 1 }, .F32, 0.2);
    try vae.rand("post_quant_conv.bias", .{}, &.{24}, .F32, 0.1);
    try vae.fill("latents_mean", .{}, &.{24}, .F32, 0);
    try vae.fill("latents_std", .{}, &.{24}, .F32, 1);
    const vp = "decoder.transformer_blocks.0.";
    inline for (.{ .{ "attn.to_qkv", 3 * vd, vd }, .{ "attn.to_out", vd, vd }, .{ "ff.w1", 8 * vd, vd }, .{ "ff.w2", vd, 4 * vd } }) |l| {
        try vae.rand(vp ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .F32, 0.2);
        try vae.rand(vp ++ l[0] ++ ".bias", .{}, &.{l[1]}, .F32, 0.1);
    }
    inline for (.{ "norm1.weight", "norm2.weight", "scale1", "scale2" }) |n| try vae.fill(vp ++ n, .{}, &.{vd}, .F32, 1);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"minimax_h3_image","partition":"image","tasks":["t2i","edit"],"pdd_sigmas":[1.0,0.5,0.0],
            \\ "transformer":{"hidden_size":64,"num_layers":1,"token_refiner_num_layers":1,"num_attention_heads":2,
            \\  "attention_head_dim":32,"ffn_hidden_size":64,"text_dim":64,"timestep_input_dim":32,
            \\  "time_embed_hidden_size":64,"time_embed_dim":32,"rope_inv_freq_len":4},
            \\ "text_encoder":{"hidden":64,"layers":1,"heads":2,"kv_heads":1,"head_dim":32,"intermediate":128},
            \\ "image_vae":{"layers":1,"heads":2,"head_dim":16}}
            },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{}) },
            .{ .name = "transformer.safetensors", .bytes = try dit.bytes() },
            .{ .name = "text_encoder.safetensors", .bytes = try te.bytes() },
            .{ .name = "image_vae.safetensors", .bytes = try vae.bytes() },
        }),
        .body = "{\"prompt\":\"a red ball\",\"size\":\"64x64\",\"seed\":3,\"min_generation_area\":1024}",
    };
}

// ── tests ──

const testing = std.testing;

/// The PNG of a job that must succeed; a refusal fails the test with its reason.
fn okPng(pack: anytype, body: []const u8) ![]const u8 {
    return switch (try pack.run(body)) {
        .ok => |png| png,
        .refused => |v| {
            std.debug.print("refused: {s}\n", .{v.why.slice()});
            return error.Refused;
        },
    };
}

test "minimax_h3_image plugin: a seeded image on the tiny pack is a deterministic PNG; bad fields are refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, try okPng(&pack, pack.fx.body));
    try testing.expectEqual(image_pixels.Size{ .w = 64, .h = 64 }, image_pixels.nativeSize(first).?);
    try testing.expectEqualStrings(first, try okPng(&pack, pack.fx.body));
    inline for (.{
        .{ "{}", "missing 'prompt'" },
        .{ "{\"prompt\":\"x\",\"steps\":4}", "this pack runs a fixed step count — omit 'steps'" },
        .{ "{\"prompt\":\"x\",\"min_generation_area\":12}", "'min_generation_area' must be an integer in [1024, 16777216]" },
        .{ "{\"prompt\":\"x\",\"size\":\"8000x512\"}", "'size' sides must be in [16, 6272]" },
        .{ "{\"prompt\":\"x\",\"mode\":\"edit\"}", "mode:\"edit\" needs an 'image' to edit" },
    }) |c| try testing.expectEqualStrings(c[1], (try pack.run(c[0])).refused.why.slice());
    const junk = [_]abi.Str{.of("not an image")};
    try testing.expectEqualStrings("a reference image could not be decoded (PNG/JPEG/WebP supported)", (try pack.runInputs("{\"prompt\":\"x\",\"mode\":\"edit\",\"image\":\"bm90\"}", &junk)).refused.why.slice());
}

test "minimax_h3_image plugin: a smaller request is generated on the minimum-area canvas and resized back" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    // 32x64 at 4096 px: the canvas doubles to 64x128 and the PNG comes back at the requested size.
    const png = try okPng(&pack, "{\"prompt\":\"x\",\"size\":\"32x64\",\"seed\":1,\"min_generation_area\":8192}");
    try testing.expectEqual(image_pixels.Size{ .w = 32, .h = 64 }, image_pixels.nativeSize(png).?);
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
