//! Hunyuan3D-2.1 on the media table: one photo to a mesh GLB. The shape engine (DINOv2 conditioner, flow DiT,
//! ShapeVAE + marching cubes) stays loaded; `texture: true` adds the paint stage, loaded per request from its own
//! folder and freed after.

const std = @import("std");
const env = @import("../../env.zig");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const mesh_request = @import("../../sdk/media/mesh_request.zig");
const mlx = @import("../../mlx/ffi.zig");
const sse = @import("../../sdk/media/progress.zig");
const glb = @import("../../sdk/mesh/glb.zig");
const hy3d = @import("hunyuan3d.zig");
const paint = @import("paint.zig");
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
    m: *hy3d.Engine,
    s: mlx.mlx_stream,
    /// The paint stage's folder, null when it is not installed.
    paint_dir: ?[]u8,
    glb: []u8 = &.{},
};

const PAINT_SUBDIR = "paint";
const PAINT_SIBLING = "hunyuan3d-2-1-paint-8bit";

fn hasConfig(io: std.Io, dir: []const u8) bool {
    if (dir.len == 0 or !std.fs.path.isAbsolute(dir)) return false;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const cfg = std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}) catch return false;
    std.Io.Dir.accessAbsolute(io, cfg, .{}) catch return false;
    return true;
}

/// The paint stage's folder: `HY3D_PAINT_DIR` (the only candidate when set), else `<pack>/paint` (one download),
/// else the converter's sibling `<parent>/hunyuan3d-2-1-paint-8bit`. Caller frees.
pub fn paintDir(io: std.Io, a: std.mem.Allocator, shape_dir: []const u8) ?[]u8 {
    if (env.get(.HY3D_PAINT_DIR)) |v| {
        const p = std.mem.span(v);
        return if (hasConfig(io, p)) a.dupe(u8, p) catch null else null;
    }
    const sub = std.fs.path.join(a, &.{ shape_dir, PAINT_SUBDIR }) catch return null;
    if (hasConfig(io, sub)) return sub;
    a.free(sub);
    const parent = std.fs.path.dirname(shape_dir) orelse return null;
    const sib = std.fs.path.join(a, &.{ parent, PAINT_SIBLING }) catch return null;
    if (hasConfig(io, sib)) return sib;
    a.free(sib);
    return null;
}

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const s: mlx.mlx_stream = .{ .ctx = ctx.stream.ctx };
    const dir = ctx.model_dir.slice();
    const m = hy3d.Engine.loadOn(io, gpa, dir, s) catch |err| {
        out.* = job.failure(err, "hunyuan3d: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        m.deinit();
        out.* = Verdict.refuse(.failed, "hunyuan3d: out of memory");
        return null;
    };
    e.* = .{ .m = m, .s = s, .paint_dir = paintDir(io, gpa, dir) };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.m.deinit();
    if (e.paint_dir) |p| gpa.free(p);
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

const Job = struct { shape: hy3d.MeshOpts, texture: bool, paint: paint.PaintOpts };
const Parsed = union(enum) { ok: Job, bad: Verdict };

/// `steps` (1..100, 30), `octree_resolution` (64..512, 256), `guidance_scale` (0..20, 5), `seed`, and `texture`
/// (the paint stage, `texture_steps` 1..100); TRELLIS.2's finish fields are refused by name.
fn parseJob(body: *const job.Body, has_paint: bool) Parsed {
    const obj = &body.parsed.value.object;
    if (mesh_request.has(obj, "roughness_scale") or mesh_request.has(obj, "clearcoat"))
        return .{ .bad = bad("'roughness_scale' and 'clearcoat' are TRELLIS.2 and Pixal3D fields (a Hunyuan3D texture's finish comes from its paint)") };
    var o = Job{ .shape = .{ .steps = 30, .guidance = 5, .seed = body.uint("seed") orelse 42, .octree_resolution = 256 }, .texture = body.flag("texture"), .paint = .{} };
    if (int(body, "steps")) |n| {
        if (n < 1 or n > 100) return .{ .bad = bad("'steps' must be in [1,100]") };
        o.shape.steps = @intCast(n);
    }
    if (int(body, "octree_resolution")) |r| {
        if (r < 64 or r > 512) return .{ .bad = bad("'octree_resolution' must be in [64,512]") };
        o.shape.octree_resolution = @intCast(r);
    }
    if (body.float("guidance_scale")) |g| {
        if (!(g >= 0 and g <= 20)) return .{ .bad = bad("'guidance_scale' must be in [0,20]") };
        o.shape.guidance = @floatCast(g);
    }
    // Never a silent untextured downgrade: the request asked for THIS output.
    if (o.texture and !has_paint)
        return .{ .bad = bad("texture requested but the paint weights are not installed (run tests/convert_hunyuan3d_paint_weights.py, or set HY3D_PAINT_DIR)") };
    o.paint.seed = o.shape.seed;
    if (int(body, "texture_steps")) |n| if (n >= 1 and n <= 100) {
        o.paint.steps = @intCast(n);
    };
    return .{ .ok = o };
}

fn failed(err: anyerror) Verdict {
    return if (err == error.Cancelled) job.client_left else job.failure(err, "generation failed");
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    var views: mesh_request.Views = .{};
    defer views.deinit(gpa);
    if (mesh_request.decode(gpa, &body.parsed.value.object, req.inputs.slice(), 1, &views)) |v| return v;
    const img = views.list.items[0];
    const j = switch (parseJob(&body, e.paint_dir != null)) {
        .ok => |o| o,
        .bad => |v| return v,
    };
    const prog = job.progressOf(progress);
    var mesh = e.m.generateMeshRaw(gpa, img.pix, img.w, img.h, j.shape, prog) catch |err| return failed(err);
    defer mesh.deinit(gpa);
    const out = if (j.texture) blk: {
        // The shape stage reads its grid on the host: a failed one is not worth minutes of paint.
        if (job.latched()) |v| return v;
        const io = std.Io.Threaded.global_single_threaded.io();
        const p = paint.PaintEngine.loadOn(io, gpa, e.paint_dir.?, e.s) catch |err| return failed(err);
        defer p.deinit();
        break :blk p.paintMeshToGlb(gpa, &mesh, img.pix, img.w, img.h, j.paint, prog) catch |err| return failed(err);
    } else sse.offload(glb.writeGlb, .{ gpa, &mesh }) catch |err| return failed(err);
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
    if (e.glb.len == 0) return Verdict.refuse(.failed, "hunyuan3d: no finished job to encode");
    out.* = .of(e.glb);
    return .ok;
}

// ── the tiny pack ──

const T = sdk.tensors.Tensors;

const H = 32; // DiT, DINOv2 and ShapeVAE width
const HD = 16; // head width (two heads everywhere)
const EMBED = 8; // latent channels
const INTER = 64; // MoE expert width

fn lin(t: *T, comptime name: []const u8, args: anytype, out: u64, in: u64, bias: bool) !void {
    try t.rand(name ++ ".weight", args, &.{ out, in }, .F32, 0.8 / @sqrt(@as(f32, @floatFromInt(in))));
    if (bias) try t.rand(name ++ ".bias", args, &.{out}, .F32, 0.05);
}

fn norm(t: *T, comptime name: []const u8, args: anytype, c: u64) !void {
    try t.fill(name ++ ".weight", args, &.{c}, .F32, 1);
    try t.fill(name ++ ".bias", args, &.{c}, .F32, 0);
}

fn mlp(t: *T, comptime name: []const u8, args: anytype, inter: u64) !void {
    try lin(t, name ++ ".fc1", args, inter, H, true);
    try lin(t, name ++ ".fc2", args, H, inter, true);
}

/// DINOv2 at 32 wide, one layer, a 56-pixel input (4x4 patches of 14).
fn conditioner(t: *T) !void {
    try t.rand("cls_token", .{}, &.{ 1, 1, H }, .F32, 0.5);
    try t.rand("pos_embed", .{}, &.{ 1, 17, H }, .F32, 0.1);
    try t.rand("patch_embed.weight", .{}, &.{ H, 3, 14, 14 }, .F32, 0.05);
    try t.rand("patch_embed.bias", .{}, &.{H}, .F32, 0.05);
    inline for (.{ "q", "k", "v", "out" }) |n| try lin(t, "layers.0.attn." ++ n, .{}, H, H, true);
    try norm(t, "layers.0.norm1", .{}, H);
    try norm(t, "layers.0.norm2", .{}, H);
    try t.fill("layers.0.ls1", .{}, &.{H}, .F32, 0.5);
    try t.fill("layers.0.ls2", .{}, &.{H}, .F32, 0.5);
    try mlp(t, "layers.0.mlp", .{}, 4 * H);
    try norm(t, "norm", .{}, H);
}

fn attn(t: *T, comptime pfx: []const u8, i: usize, kv_in: u64) !void {
    try lin(t, pfx ++ ".q", .{i}, H, H, false);
    try lin(t, pfx ++ ".k", .{i}, H, kv_in, false);
    try lin(t, pfx ++ ".v", .{i}, H, kv_in, false);
    try lin(t, pfx ++ ".out", .{i}, H, H, true);
    try t.fill(pfx ++ ".q_norm.weight", .{i}, &.{HD}, .F32, 1);
    try t.fill(pfx ++ ".k_norm.weight", .{i}, &.{HD}, .F32, 1);
}

/// Two experts, 8-bit in groups of 32 as the shipped pack (MLX's CPU gather_mm hangs on f16, its gather_qmm runs):
/// codes uniform over a range of about ±0.8/sqrt(in), f16 scales and offsets, an f32 bias.
fn experts(t: *T, comptime name: []const u8, out: u64, in: u64) !void {
    const half = 0.8 / @sqrt(@as(f32, @floatFromInt(in)));
    try t.words(name ++ ".weight", .{}, &.{ 2, out, in / 4 });
    try t.fill(name ++ ".scales", .{}, &.{ 2, out, in / 32 }, .F16, 2 * half / 255);
    try t.fill(name ++ ".biases", .{}, &.{ 2, out, in / 32 }, .F16, -half);
    try t.rand(name ++ ".bias", .{}, &.{ 2, out }, .F32, 0.05);
}

/// The flow DiT at depth 3: block 0 pushes its skip, block 2 pops it and is the one MoE block (two experts).
fn dit(t: *T) !void {
    try lin(t, "x_embedder", .{}, H, EMBED, true);
    try lin(t, "t_embedder.mlp1", .{}, 2 * H, H, true);
    try lin(t, "t_embedder.mlp2", .{}, H, 2 * H, true);
    for (0..3) |i| {
        inline for (.{ "norm1", "norm2", "norm3" }) |n| try norm(t, "blocks.{d}." ++ n, .{i}, H);
        try attn(t, "blocks.{d}.attn1", i, H);
        try attn(t, "blocks.{d}.attn2", i, H);
        if (i < 2) try mlp(t, "blocks.{d}.mlp", .{i}, 4 * H);
    }
    try t.rand("blocks.2.moe.gate.weight", .{}, &.{ 2, H }, .F32, 0.3);
    try experts(t, "blocks.2.moe.experts.fc1", INTER, H);
    try experts(t, "blocks.2.moe.experts.fc2", H, INTER);
    try mlp(t, "blocks.2.moe.shared", .{}, INTER);
    try lin(t, "blocks.2.skip.linear", .{}, H, 2 * H, true);
    try norm(t, "blocks.2.skip.norm", .{}, H);
    try norm(t, "final.norm", .{}, H);
    try lin(t, "final.linear", .{}, EMBED, H, true);
}

fn vaeAttn(t: *T, comptime pfx: []const u8) !void {
    inline for (.{ "q", "k", "v" }) |n| try lin(t, pfx ++ "." ++ n, .{}, H, H, false);
    try lin(t, pfx ++ ".out", .{}, H, H, true);
    try norm(t, pfx ++ ".q_norm", .{}, HD);
    try norm(t, pfx ++ ".k_norm", .{}, HD);
}

/// The ShapeVAE decoder: one self-attention block over the latents, then the geometry head's cross-attention
/// (Fourier features of 2 frequencies) to one SDF channel, whose bias keeps a surface inside the volume.
fn vae(t: *T) !void {
    try lin(t, "post_kl", .{}, H, EMBED, true);
    try norm(t, "blocks.0.ln1", .{}, H);
    try norm(t, "blocks.0.ln2", .{}, H);
    try vaeAttn(t, "blocks.0.attn");
    try mlp(t, "blocks.0.mlp", .{}, 4 * H);
    try lin(t, "geo.query_proj", .{}, H, 15, true);
    inline for (.{ "geo.ln1", "geo.ln2", "geo.ln3", "geo.ln_post" }) |n| try norm(t, n, .{}, H);
    try vaeAttn(t, "geo.attn");
    try mlp(t, "geo.mlp", .{}, 4 * H);
    try t.rand("geo.out_proj.weight", .{}, &.{ 1, H }, .F32, 0.3);
    try t.fill("geo.out_proj.bias", .{}, &.{1}, .F32, 0.9);
}

pub const config_json =
    \\{"model_type":"hunyuan3d_2_1","hidden_size":32,"depth":3,"num_heads":2,"context_dim":32,"num_latents":64,
    \\ "embed_dim":8,"vae_width":32,"vae_heads":2,"vae_decoder_layers":1,"num_freqs":2,"num_moe_layers":1,
    \\ "num_experts":2,"moe_top_k":1,"dino_hidden":32,"dino_layers":1,"dino_heads":2,"dino_patch":14,
    \\ "dino_image_size":56}
;

/// A seeded random shape pack at the smallest widths the code takes; the job runs two steps at octree 64. The paint
/// stage's UNet is SD2's fixed geometry (320..1280 channels), so no tiny paint pack exists: `texture` is refused here.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    var c = T.init(a, 71);
    try conditioner(&c);
    var d = T.init(a, 72);
    try dit(&d);
    var v = T.init(a, 73);
    try vae(&v);
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes = config_json },
            .{ .name = "conditioner.safetensors", .bytes = try c.bytes() },
            .{ .name = "dit.safetensors", .bytes = try d.bytes() },
            .{ .name = "vae.safetensors", .bytes = try v.bytes() },
        }),
        .body = "{\"image\":\"x\",\"steps\":2,\"octree_resolution\":64,\"seed\":3}",
        .inputs = try a.dupe([]const u8, &.{try @import("../trellis2/plugin.zig").cutout(a)}),
    };
}

// ── tests ──

const testing = std.testing;
const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack;
const golden = @import("../../sdk/testing/golden.zig");

test "hunyuan3d: the paint stage resolves inside the pack first, then as the local converter's sibling, else none" {
    const io = testing.io;
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = buf[0..try tmp.dir.realPath(io, &buf)];
    for ([_][]const u8{ "combined", "combined/paint", "local/hunyuan3d-2-1-8bit", "local/hunyuan3d-2-1-paint-8bit", "bare/shape-only" }) |rel| {
        try tmp.dir.createDirPath(io, rel);
        const cfg = try std.fs.path.join(a, &.{ rel, "config.json" });
        defer a.free(cfg);
        try tmp.dir.writeFile(io, .{ .sub_path = cfg, .data = "{}" });
    }
    const cases = [_][2][]const u8{ .{ "combined", "combined/paint" }, .{ "local/hunyuan3d-2-1-8bit", "local/hunyuan3d-2-1-paint-8bit" } };
    for (cases) |c| {
        const shape = try std.fs.path.join(a, &.{ root, c[0] });
        defer a.free(shape);
        const want = try std.fs.path.join(a, &.{ root, c[1] });
        defer a.free(want);
        const got = paintDir(io, a, shape).?;
        defer a.free(got);
        try testing.expectEqualStrings(want, got);
    }
    const bare = try std.fs.path.join(a, &.{ root, "bare/shape-only" });
    defer a.free(bare);
    try testing.expect(paintDir(io, a, bare) == null);
}

test "hunyuan3d: a job whose GPU stage latched a failure is refused by class, never a GLB of zeros" {
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

test "hunyuan3d: a job on the tiny pack is a GLB; fields and a missing paint stage are refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const in = [_]abi.Str{.of(pack.fx.inputs[0])};
    const out = (try pack.runInputs(pack.fx.body, &in)).ok;
    try testing.expectEqualStrings("glTF", out[0..4]);
    const cases = [_][2][]const u8{
        .{ "{\"images\":[\"x\",\"y\"]}", "'images' (several views of one object) is served by Pixal3D; this model takes one 'image'" },
        .{ "{\"image\":\"x\",\"clearcoat\":0.5}", "'roughness_scale' and 'clearcoat' are TRELLIS.2 and Pixal3D fields (a Hunyuan3D texture's finish comes from its paint)" },
        .{ "{\"image\":\"x\",\"steps\":0}", "'steps' must be in [1,100]" },
        .{ "{\"image\":\"x\",\"octree_resolution\":32}", "'octree_resolution' must be in [64,512]" },
        .{ "{\"image\":\"x\",\"guidance_scale\":21}", "'guidance_scale' must be in [0,20]" },
        .{ "{\"image\":\"x\",\"texture\":true}", "texture requested but the paint weights are not installed (run tests/convert_hunyuan3d_paint_weights.py, or set HY3D_PAINT_DIR)" },
    };
    for (cases) |c| try testing.expectEqualStrings(c[1], (try pack.runInputs(c[0], &in)).refused.why.slice());
}

test "hunyuan3d: the tiny pack's latent and SDF grid are bit-equal to its golden run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const e: *Engine = @ptrCast(@alignCast(pack.engine));
    const img = @import("../../sdk/media/image_pixels.zig").decodeRgba(gpa, pack.fx.inputs[0]).?;
    defer img.deinit(gpa);
    const c = e.m.cfg;
    const pixels = try hy3d.preprocessImage(gpa, img.pix, img.w, img.h, c.dino_image_size);
    defer gpa.free(pixels);
    const side: c_int = @intCast(c.dino_image_size);
    const parr = mlx.mlx_array_new_data(pixels.ptr, &[_]c_int{ 1, 3, side, side }, 4, .float32);
    defer _ = mlx.mlx_array_free(parr);
    const cond = try e.m.dino.encode(parr);
    defer _ = mlx.mlx_array_free(cond);
    const latent = try hy3d.denoise(&e.m.dit, gpa, cond, 3, 5, 3, null, null);
    defer _ = mlx.mlx_array_free(latent);
    const set = try hy3d.vaeDecodeLatentSet(&e.m.vae, latent);
    defer _ = mlx.mlx_array_free(set);
    const grid = try hy3d.decodeVolumeHierarchical(&e.m.vae, gpa, set, 64, hy3d.VOLUME_BOUND, 0, 32768, null);
    defer gpa.free(grid);
    const g = mlx.mlx_array_new_data(grid.ptr, &[_]c_int{@intCast(grid.len)}, 1, .float32);
    defer _ = mlx.mlx_array_free(g);
    try golden.expect("hunyuan3d_shape", &.{ .{ .name = "cond", .array = cond }, .{ .name = "latent", .array = latent }, .{ .name = "grid", .array = g } });
}
