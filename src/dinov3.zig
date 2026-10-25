//! DINOv3 ViT image encoder (Meta; `facebook/dinov3-vitl16-pretrain-lvd1689m`) as TRELLIS.2 reads it:
//! every token after the last block and a NON-affine LayerNorm (not the checkpoint's own final norm),
//! CLS + register tokens first, then one token per patch.
//!
//! Matches transformers' `DINOv3ViTModel` (published layout `layer.N.*`): a 16x16 patch conv, CLS and
//! register tokens, 2-axis RoPE on the PATCH tokens only (centres normalized to [-1, 1], rotate-half
//! pairing), pre-norm blocks with per-channel layer scale. Runs f32, as the reference extractor does.
//! Parity: `dinov3 oracle` against transformers on a tiny random model (tests/dump_trellis2_fixtures.py).

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const scope_mod = @import("mlx_scope.zig");
const qvis = @import("qwen_vision.zig");

const A = mlx.mlx_array;
const Scope = scope_mod.Scope;
const Linear = scope_mod.Linear;
const Weights = model_mod.Weights;
const NONE = A{ .ctx = null };

const FINAL_LN_EPS: f32 = 1e-5;
pub const IMAGENET_MEAN = [3]f32{ 0.485, 0.456, 0.406 };
pub const IMAGENET_STD = [3]f32{ 0.229, 0.224, 0.225 };

pub const Cfg = struct {
    hidden: usize,
    heads: usize,
    layers: usize,
    mlp: usize,
    patch: usize,
    registers: usize,
    rope_theta: f32,
    eps: f32,
};

const Layer = struct {
    norm1_w: A,
    norm1_b: A,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    ls1: A,
    norm2_w: A,
    norm2_b: A,
    up: Linear,
    down: Linear,
    ls2: A,
};

pub const Encoder = struct {
    gpa: std.mem.Allocator,
    cfg: Cfg,
    w: Weights,
    cls: A,
    registers: A,
    /// `[hidden, patch, patch, 3]` (NHWC conv layout) and its bias.
    patch_w: A,
    patch_b: A,
    layers: []Layer,
    owned: std.ArrayList(A) = .empty,

    /// `<dir>/config.json` + `<dir>/model.safetensors`.
    pub fn open(io: std.Io, gpa: std.mem.Allocator, dir: []const u8) !Encoder {
        const cfg_path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
        defer gpa.free(cfg_path);
        const bytes = try readFile(io, gpa, cfg_path);
        defer gpa.free(bytes);
        const Raw = struct {
            hidden_size: usize,
            intermediate_size: usize,
            num_hidden_layers: usize,
            num_attention_heads: usize,
            patch_size: usize = 16,
            num_register_tokens: usize = 0,
            rope_theta: f32 = 100,
            use_gated_mlp: bool = false,
            layer_norm_eps: f32 = 1e-5,
        };
        var parsed = try std.json.parseFromSlice(Raw, gpa, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const r = parsed.value;
        if (r.use_gated_mlp) return error.UnsupportedDinov3;
        if (r.hidden_size % r.num_attention_heads != 0 or (r.hidden_size / r.num_attention_heads) % 4 != 0) return error.UnsupportedDinov3;
        const w = try model_mod.loadWeightsFile(gpa, dir, "model.safetensors");
        return load(gpa, w, .{
            .hidden = r.hidden_size,
            .heads = r.num_attention_heads,
            .layers = r.num_hidden_layers,
            .mlp = r.intermediate_size,
            .patch = r.patch_size,
            .registers = r.num_register_tokens,
            .rope_theta = r.rope_theta,
            .eps = r.layer_norm_eps,
        });
    }

    /// Takes ownership of `w`.
    pub fn load(gpa: std.mem.Allocator, w: Weights, cfg: Cfg) !Encoder {
        var e = Encoder{ .gpa = gpa, .cfg = cfg, .w = w, .cls = undefined, .registers = undefined, .patch_w = undefined, .patch_b = undefined, .layers = try gpa.alloc(Layer, 0) };
        errdefer e.deinit();
        const hid: u32 = @intCast(cfg.hidden);
        e.cls = try scope_mod.getW(&e.w, "embeddings.cls_token", .{});
        e.registers = try scope_mod.getW(&e.w, "embeddings.register_tokens", .{});
        e.patch_b = try scope_mod.getW(&e.w, "embeddings.patch_embeddings.bias", .{});
        // Conv2d weight [out, in, kh, kw] -> [out, kh, kw, in] for MLX's channels-last conv.
        const s = mlx.gpuStream();
        defer _ = mlx.mlx_stream_free(s);
        var pw = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(&pw, try scope_mod.getW(&e.w, "embeddings.patch_embeddings.weight", .{}), &[_]c_int{ 0, 2, 3, 1 }, 4, s));
        errdefer _ = mlx.mlx_array_free(pw);
        try mlx.check(mlx.mlx_array_eval(pw));
        try e.owned.append(gpa, pw);
        e.patch_w = pw;
        // The published layout keeps the encoder at the top level; transformers >= 5 nests it under `model.`.
        const nested = e.w.get("layer.0.norm1.weight") == null;
        gpa.free(e.layers);
        e.layers = try gpa.alloc(Layer, cfg.layers);
        inline for (.{ false, true }) |use_nested| {
            if (nested == use_nested) {
                const pre = if (use_nested) "model.layer.{d}" else "layer.{d}";
                for (e.layers, 0..) |*l, i| l.* = .{
                    .norm1_w = try scope_mod.getW(&e.w, pre ++ ".norm1.weight", .{i}),
                    .norm1_b = try scope_mod.getW(&e.w, pre ++ ".norm1.bias", .{i}),
                    .q = try scope_mod.linear(&e.w, pre ++ ".attention.q_proj", .{i}, hid),
                    .k = try scope_mod.linear(&e.w, pre ++ ".attention.k_proj", .{i}, hid),
                    .v = try scope_mod.linear(&e.w, pre ++ ".attention.v_proj", .{i}, hid),
                    .o = try scope_mod.linear(&e.w, pre ++ ".attention.o_proj", .{i}, hid),
                    .ls1 = try scope_mod.getW(&e.w, pre ++ ".layer_scale1.lambda1", .{i}),
                    .norm2_w = try scope_mod.getW(&e.w, pre ++ ".norm2.weight", .{i}),
                    .norm2_b = try scope_mod.getW(&e.w, pre ++ ".norm2.bias", .{i}),
                    .up = try scope_mod.linear(&e.w, pre ++ ".mlp.up_proj", .{i}, hid),
                    .down = try scope_mod.linear(&e.w, pre ++ ".mlp.down_proj", .{i}, @intCast(cfg.mlp)),
                    .ls2 = try scope_mod.getW(&e.w, pre ++ ".layer_scale2.lambda1", .{i}),
                };
            }
        }
        return e;
    }

    pub fn deinit(self: *Encoder) void {
        self.gpa.free(self.layers);
        for (self.owned.items) |a| _ = mlx.mlx_array_free(a);
        self.owned.deinit(self.gpa);
        self.w.deinit();
    }

    fn headDim(self: *const Encoder) usize {
        return self.cfg.hidden / self.cfg.heads;
    }

    /// `[N, head_dim]` cos and sin for the patch grid: the (y, x) centre of each patch in [-1, 1],
    /// `2 pi * centre * inv_freq`, the two axes' angles side by side and the whole row tiled twice.
    fn ropeTables(self: *const Encoder, sc: *Scope, hp: usize, wp: usize) !struct { cos: A, sin: A } {
        const hd = self.headDim();
        const quarter = hd / 4;
        const n = hp * wp;
        const cos_t = try sc.a.alloc(f32, n * hd);
        defer sc.a.free(cos_t);
        const sin_t = try sc.a.alloc(f32, n * hd);
        defer sc.a.free(sin_t);
        for (0..hp) |i| for (0..wp) |j| {
            const cy = 2.0 * ((@as(f32, @floatFromInt(i)) + 0.5) / @as(f32, @floatFromInt(hp))) - 1.0;
            const cx = 2.0 * ((@as(f32, @floatFromInt(j)) + 0.5) / @as(f32, @floatFromInt(wp))) - 1.0;
            const row = (i * wp + j) * hd;
            for (0..2) |axis| for (0..quarter) |f| {
                const inv = 1.0 / std.math.pow(f32, self.cfg.rope_theta, @as(f32, @floatFromInt(f)) * (4.0 / @as(f32, @floatFromInt(hd))));
                const ang = 2.0 * std.math.pi * (if (axis == 0) cy else cx) * inv;
                const col = axis * quarter + f;
                cos_t[row + col] = @cos(ang);
                sin_t[row + col] = @sin(ang);
                cos_t[row + hd / 2 + col] = @cos(ang);
                sin_t[row + hd / 2 + col] = @sin(ang);
            };
        };
        const shape = [_]c_int{ @intCast(n), @intCast(hd) };
        return .{ .cos = try sc.fromF32(cos_t, &shape), .sin = try sc.fromF32(sin_t, &shape) };
    }

    fn rotateHalf(sc: *Scope, x: A) !A {
        const d: c_int = mlx.getShape(x)[3];
        const x1 = try sc.slice(x, 3, 0, @divExact(d, 2));
        const x2 = try sc.slice(x, 3, @divExact(d, 2), d);
        return sc.concat(&.{ try sc.mulS(x2, -1), x1 }, 3);
    }

    /// RoPE on the patch tokens of `x` `[1, H, L, D]`; the prefix (CLS + registers) passes through.
    fn rope(self: *const Encoder, sc: *Scope, x: A, cos: A, sin: A) !A {
        const l: c_int = mlx.getShape(x)[2];
        const p: c_int = @intCast(1 + self.cfg.registers);
        const patches = try sc.slice(x, 2, p, l);
        const rot = try sc.add(try sc.mul(patches, cos), try sc.mul(try rotateHalf(sc, patches), sin));
        return sc.concat(&.{ try sc.slice(x, 2, 0, p), rot }, 2);
    }

    /// `image` `[1, H, W, 3]` f32, ImageNet-normalized, H and W multiples of the patch size.
    /// Returns `[1, 1 + registers + (H/patch)*(W/patch), hidden]` f32.
    pub fn forward(self: *const Encoder, sc: *Scope, image: A) !A {
        const sh = mlx.getShape(image);
        const hp: usize = @intCast(@divExact(sh[1], @as(c_int, @intCast(self.cfg.patch))));
        const wp: usize = @intCast(@divExact(sh[2], @as(c_int, @intCast(self.cfg.patch))));
        const hid: c_int = @intCast(self.cfg.hidden);
        const heads: c_int = @intCast(self.cfg.heads);
        const hd: c_int = @intCast(self.headDim());
        var o = mlx.mlx_array_new();
        const conv = try sc.res(mlx.mlx_conv2d(&o, image, self.patch_w, @intCast(self.cfg.patch), @intCast(self.cfg.patch), 0, 0, 1, 1, 1, sc.s), &o);
        const patches = try sc.reshape(try sc.add(conv, self.patch_b), &.{ 1, @intCast(hp * wp), hid });
        var h = try sc.concat(&.{ try asLike(sc, self.cls, patches), try asLike(sc, self.registers, patches), patches }, 1);
        const l: c_int = mlx.getShape(h)[1];
        const tables = try self.ropeTables(sc, hp, wp);
        const perm = [_]c_int{ 0, 2, 1, 3 };
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(self.headDim())));
        for (self.layers) |*ly| {
            const x = try sc.layerNorm(h, try asLike(sc, ly.norm1_w, h), try asLike(sc, ly.norm1_b, h), self.cfg.eps);
            const q = try self.rope(sc, try sc.transpose(try sc.reshape(try ly.q.forward(sc, x), &.{ 1, l, heads, hd }), &perm), tables.cos, tables.sin);
            const k = try self.rope(sc, try sc.transpose(try sc.reshape(try ly.k.forward(sc, x), &.{ 1, l, heads, hd }), &perm), tables.cos, tables.sin);
            const v = try sc.transpose(try sc.reshape(try ly.v.forward(sc, x), &.{ 1, l, heads, hd }), &perm);
            const att = try sc.reshape(try sc.transpose(try sc.sdpa(q, k, v, scale), &perm), &.{ 1, l, hid });
            h = try sc.add(h, try sc.mul(try ly.o.forward(sc, att), try asLike(sc, ly.ls1, h)));
            const y = try sc.layerNorm(h, try asLike(sc, ly.norm2_w, h), try asLike(sc, ly.norm2_b, h), self.cfg.eps);
            const m = try ly.down.forward(sc, try sc.geluErf(try ly.up.forward(sc, y)));
            h = try sc.add(h, try sc.mul(m, try asLike(sc, ly.ls2, h)));
        }
        return sc.layerNorm(h, NONE, NONE, FINAL_LN_EPS);
    }
};

fn asLike(sc: *Scope, x: A, like: A) !A {
    const dt = mlx.mlx_array_dtype(like);
    return if (mlx.mlx_array_dtype(x) == dt) x else sc.astype(x, dt);
}

fn readFile(io: std.Io, gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(gpa, .limited(1 << 20));
}

/// What `DinoV3FeatureExtractor` feeds the model: the RGB8 image (`w` x `h`) Lanczos-resized to
/// `size` x `size`, scaled to [0, 1] and ImageNet-normalized, as `[size, size, 3]` f32 (NHWC). Caller frees.
pub fn preprocess(gpa: std.mem.Allocator, rgb: []const u8, w: u32, h: u32, size: u32) ![]f32 {
    const resized = try qvis.resizeInterleavedPil(gpa, rgb, h, w, size, size, 3, .lanczos);
    defer gpa.free(resized);
    const out = try gpa.alloc(f32, resized.len);
    for (out, resized, 0..) |*o, v, i| o.* = (@as(f32, @floatFromInt(v)) / 255.0 - IMAGENET_MEAN[i % 3]) / IMAGENET_STD[i % 3];
    return out;
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "dinov3 preprocess: a flat image keeps its value through the resize and is ImageNet-normalized" {
    const gpa = testing.allocator;
    const src = try gpa.alloc(u8, 5 * 7 * 3);
    defer gpa.free(src);
    const px = [3]u8{ 255, 128, 0 };
    for (src, 0..) |*v, i| v.* = px[i % 3];
    const out = try preprocess(gpa, src, 7, 5, 4);
    defer gpa.free(out);
    try testing.expectEqual(@as(usize, 4 * 4 * 3), out.len);
    for (0..16) |p| {
        try testing.expectApproxEqAbs((1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0], out[p * 3], 1e-5);
        try testing.expectApproxEqAbs((128.0 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1], out[p * 3 + 1], 1e-5);
        try testing.expectApproxEqAbs((0.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2], out[p * 3 + 2], 1e-5);
    }
}

test "dinov3 oracle: every token matches transformers' DINOv3ViTModel on a tiny random model (TRELLIS2_FIXTURES)" {
    const gpa = testing.allocator;
    const fixtures = std.mem.span(std.c.getenv("TRELLIS2_FIXTURES") orelse return error.SkipZigTest);
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try std.fmt.allocPrint(gpa, "{s}/dinov3_tiny", .{fixtures});
    defer gpa.free(dir);
    var enc = try Encoder.open(io, gpa, dir);
    defer enc.deinit();
    var fx = try model_mod.loadWeightsFile(gpa, fixtures, "dinov3.safetensors");
    defer fx.deinit();

    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    const image = try sc.transpose(fx.get("image").?, &.{ 0, 2, 3, 1 }); // NCHW -> NHWC
    const got = try enc.forward(&sc, image);
    const want = fx.get("out").?;
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    var c1 = mlx.mlx_array_new();
    const g = try sc.res(mlx.mlx_contiguous(&c1, got, false, s), &c1);
    var c2 = mlx.mlx_array_new();
    const w = try sc.res(mlx.mlx_contiguous(&c2, want, false, s), &c2);
    try mlx.check(mlx.mlx_array_eval(g));
    try mlx.check(mlx.mlx_array_eval(w));
    const gd = (mlx.mlx_array_data_float32(g) orelse return error.MlxError)[0..mlx.mlx_array_size(g)];
    const wd = (mlx.mlx_array_data_float32(w) orelse return error.MlxError)[0..mlx.mlx_array_size(w)];
    var worst: f64 = 0;
    var dot: f64 = 0;
    var gg: f64 = 0;
    var ww: f64 = 0;
    for (gd, wd) |a, b| {
        dot += @as(f64, a) * b;
        gg += @as(f64, a) * a;
        ww += @as(f64, b) * b;
        worst = @max(worst, @abs(@as(f64, a) - b));
    }
    std.debug.print("dinov3 tiny: cos {d:.7} max abs err {e:.2}\n", .{ dot / @sqrt(gg * ww), worst });
    try testing.expect(dot / @sqrt(gg * ww) > 0.99999);
    try testing.expect(worst < 1e-3);
}

/// A directory holding `model.safetensors` with ONE tensor that no loader reads, so every open fails after it owns the weights.
fn writeUselessCheckpoint(td: *std.testing.TmpDir) !void {
    const io = std.testing.io;
    const header = "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
    var buf: [8 + header.len + 4]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8..][0..header.len], header);
    @memset(buf[8 + header.len ..], 0);
    try td.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &buf });
}

test "dinov3: a checkpoint missing its tensors is a named error and frees its weights once" {
    const io = std.testing.io;
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    try writeUselessCheckpoint(&td);
    try td.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"hidden_size\":64,\"intermediate_size\":128,\"num_hidden_layers\":1,\"num_attention_heads\":4}" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try td.dir.realPath(io, &path_buf)];
    if (Encoder.open(io, testing.allocator, dir)) |*e| {
        var enc = e.*;
        enc.deinit();
        return error.TestUnexpectedResult;
    } else |_| {}
}

