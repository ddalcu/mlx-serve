//! EmbeddingGemma 2 embedding tests over the tiny fixture (text, image, video): the served model (the plugin through
//! the engine's driver) and the image tower against the transformers reference.
const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const transformer_mod = @import("../../engines/mlx/transformer.zig");
const model_mod = @import("../../sdk/config/model.zig");
const Transformer = transformer_mod.Transformer;
const testing = std.testing;
const computeEmbeddingsBatch = @import("../../engines/mlx/embed/pooling.zig").computeEmbeddingsBatch;
const computeEmbeddingsBatchWith = @import("../../engines/mlx/embed/pooling.zig").computeEmbeddingsBatchWith;
const plugin_driver = @import("../../engines/mlx/plugin/driver.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const FsPackDir = @import("../../host/registry/pack_dir.zig").FsPackDir;
const abi = @import("../../abi/plugin.zig");
const encoder = @import("../gemma4/bidirectional.zig");


// ── tests ──


fn rowCosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * y;
        na += @as(f64, x) * x;
        nb += @as(f64, y) * y;
    }
    return dot / (@sqrt(na) * @sqrt(nb));
}

/// The tiny EmbeddingGemma 2 checkpoint (`tests/dump_embeddinggemma2_fixtures.py tiny`) written to a temp dir and loaded.
const Eg2Tiny = struct {
    tmp: std.testing.TmpDir,
    allocator: std.mem.Allocator,
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,
    path_buf: [512]u8 = undefined,
    path_len: usize = 0,

    fn open(allocator: std.mem.Allocator, io: std.Io) !Eg2Tiny {
        return openWith(allocator, io, false);
    }

    /// With the image tower's tensors, as a server that serves images loads them.
    fn openWithVision(allocator: std.mem.Allocator, io: std.Io) !Eg2Tiny {
        return openWith(allocator, io, true);
    }

    fn openWith(allocator: std.mem.Allocator, io: std.Io, vision: bool) !Eg2Tiny {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = @embedFile("../../fixtures/embedding_gemma2_tiny_config.json") });
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = @embedFile("../../fixtures/embedding_gemma2_tiny.safetensors") });
        var fx: Eg2Tiny = .{ .tmp = tmp, .allocator = allocator, .config = undefined, .weights = undefined };
        fx.path_len = try tmp.dir.realPath(io, &fx.path_buf);
        fx.config = try model_mod.parseConfig(io, allocator, fx.path());
        errdefer fx.config.deinit(allocator);
        fx.weights = if (vision) try model_mod.loadWeightsWithVision(io, allocator, fx.path()) else try model_mod.loadWeights(io, allocator, fx.path());
        errdefer fx.weights.deinit();
        model_mod.resolveWeightPrefix(&fx.config, &fx.weights);
        return fx;
    }

    fn path(self: *const Eg2Tiny) []const u8 {
        return self.path_buf[0..self.path_len];
    }

    /// The served model: the plugin's load and forward behind the engine's shell.
    fn served(self: *Eg2Tiny, io: std.Io) !Transformer {
        try testing.expect(self.config.plugin != null);
        return plugin_driver.load(io, self.allocator, &self.config, self.path(), mlx.gpuStream());
    }

    fn deinit(self: *Eg2Tiny) void {
        self.weights.deinit();
        self.config.deinit(self.allocator);
        self.tmp.cleanup();
    }
};

test "EmbeddingGemma 2: forward and pooled embeddings match the transformers reference on a tiny checkpoint" {
    // Random weights in the release's layout (full-attention layers wider, inclusive band radius 2,
    // projection-only per-layer inputs, layer scalars) and HF's float32 outputs. The 12-token rows cross the
    // band, so an exclusive radius moves their embedding to cosine 0.84.
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;

    var fx = try Eg2Tiny.open(allocator, io);
    defer fx.deinit();
    var xfm = try fx.served(io);
    defer xfm.deinit();

    const Expected = struct {
        sequences: []const []const u32,
        norm: []const []const []const f32,
        embedding: []const []const f32,
    };
    const parsed = try std.json.parseFromSlice(Expected, allocator, @embedFile("../../fixtures/embedding_gemma2_tiny_expected.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const want = parsed.value;

    // The production path: one padded, key-masked batch, pooled, projected, normalized.
    const batch = try computeEmbeddingsBatch(allocator, &xfm, want.sequences);
    defer {
        for (batch) |r| allocator.free(r);
        allocator.free(batch);
    }
    for (batch, want.embedding, 0..) |got, ref, i| {
        try testing.expectEqual(ref.len, got.len);
        const cos = rowCosine(got, ref);
        testing.expect(cos > 0.998) catch |e| {
            std.debug.print("row {d}: embedding cosine {d:.6}\n", .{ i, cos });
            return e;
        };
    }

    // A shipped pack's head is bf16, and normalizing its bf16 output leaves the norm off 1: the projection
    // hands the normalization f32.
    const head = &@as(*encoder.Loaded, @ptrCast(@alignCast(xfm.plugin.?.model))).base.weights.projection.?.first.w;
    const head_f32 = head.*;
    var head_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head_bf16);
    try mlx.check(mlx.mlx_astype(&head_bf16, head_f32, .bfloat16, xfm.s));
    head.* = head_bf16;
    defer head.* = head_f32;
    const bf16_head = try computeEmbeddingsBatch(allocator, &xfm, want.sequences);
    defer {
        for (bf16_head) |r| allocator.free(r);
        allocator.free(bf16_head);
    }
    for (bf16_head, 0..) |row, i| {
        var sumsq: f64 = 0;
        for (row) |x| sumsq += @as(f64, x) * x;
        testing.expect(@abs(@sqrt(sumsq) - 1.0) < 1e-5) catch |e| {
            std.debug.print("row {d}: norm {d:.6} with a bf16 head\n", .{ i, @sqrt(sumsq) });
            return e;
        };
    }

    // Each row alone has no padding to hide behind; its final-normed token states are the pre-projection
    // hidden the pooling reads, so the layers are checked row by row rather than only through the mean.
    for (want.sequences, want.norm, 0..) |seq, ref_rows, i| {
        var ids: [32]i32 = undefined;
        for (seq, 0..) |t, k| ids[k] = @intCast(t);
        const shape = [_]c_int{ 1, @intCast(seq.len) };
        const input = mlx.mlx_array_new_data(&ids, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);
        const hidden = try xfm.forwardEmbeddingMasked(input, null, null);
        defer _ = mlx.mlx_array_free(hidden);
        var f32_hidden = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32_hidden);
        try mlx.check(mlx.mlx_astype(&f32_hidden, hidden, .float32, xfm.s));
        try mlx.check(mlx.mlx_array_eval(f32_hidden));
        const data = mlx.mlx_array_data_float32(f32_hidden).?;
        for (ref_rows, 0..) |ref, t| {
            const cos = rowCosine(data[t * ref.len ..][0..ref.len], ref);
            testing.expect(cos > 0.995) catch |e| {
                std.debug.print("row {d} token {d}: state cosine {d:.6}\n", .{ i, t, cos });
                return e;
            };
        }
    }
}

const vision_mod = @import("../../engines/mlx/vision_encoder.zig");
const qwen_vision_budget = @import("../../sdk/vision/qwen_vision.zig");

/// The soft-token budget the tiny fixture's image and clip were processed at (70 is the smallest a processor accepts).
const eg2_fixture_budget: u32 = 70;

/// The media half of the tiny fixture's reference outputs (`tests/dump_embeddinggemma2_fixtures.py`).
const Eg2Media = struct {
    media: struct {
        image: struct { h: u32, w: u32, rgb: []const u8, target: [2]u32, tokens: u32, soft: []const []const f32 },
        image_big: struct { budget: u32, target: [2]u32, tokens: u32, soft: []const []const f32 },
        frames: struct { h: u32, w: u32, rgb: []const []const u8, target: [2]u32, tokens: u32, soft: []const []const f32 },
        image_ids: []const u32,
        image_embedding: []const f32,
        text_image_ids: []const u32,
        text_image_embedding: []const f32,
        video_ids: []const u32,
        video_embedding: []const f32,
        mixed_ids: []const u32,
        mixed_embedding: []const f32,
    },
};

/// The fixture's image through the production preprocessing at `budget` soft tokens.
fn eg2FixtureImage(allocator: std.mem.Allocator, b64: []const u8, h: u32, w: u32, budget: u32) !qwen_vision_budget.UnitImage {
    const rgb = try allocator.alloc(u8, @as(usize, h) * w * 3);
    defer allocator.free(rgb);
    try std.base64.standard.Decoder.decode(rgb, b64);
    return qwen_vision_budget.budgetPixels(allocator, rgb, h, w, 16, 3, budget);
}

fn eg2ImageRows(allocator: std.mem.Allocator, enc: *vision_mod.VisionEncoder, m: anytype, budget: u32, target: [2]u32) !mlx.mlx_array {
    const img = try eg2FixtureImage(allocator, m.rgb, m.h, m.w, budget);
    defer allocator.free(img.pixels);
    try testing.expectEqual(target[0], img.h);
    try testing.expectEqual(target[1], img.w);
    const shape = [_]c_int{ 1, 3, @intCast(img.h), @intCast(img.w) };
    const pixels = mlx.mlx_array_new_data(img.pixels.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(pixels);
    return enc.forward(pixels);
}

/// A video as the scheduler hands it to the tower: every frame's CHW pixels back to back, viewed as patch rows.
fn eg2VideoRows(allocator: std.mem.Allocator, enc: *vision_mod.VisionEncoder, m: anytype) !mlx.mlx_array {
    var bytes = std.ArrayList(u8).empty;
    defer bytes.deinit(allocator);
    var grid_h: u32 = 0;
    var grid_w: u32 = 0;
    for (m.rgb) |frame_b64| {
        const img = try eg2FixtureImage(allocator, frame_b64, m.h, m.w, eg2_fixture_budget);
        defer allocator.free(img.pixels);
        try bytes.appendSlice(allocator, img.pixels);
        grid_h = img.h / 16;
        grid_w = img.w / 16;
    }
    const frames: u32 = @intCast(m.rgb.len);
    const rows: c_int = @intCast(frames * grid_h * grid_w);
    const shape = [_]c_int{ rows, 768 };
    const patches = mlx.mlx_array_new_data(bytes.items.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(patches);
    return enc.forwardVideoPatches(patches, frames, grid_h, grid_w);
}

/// The reference's soft-token rows, one list after another, as the [1, n, hidden] array the splice reads.
fn eg2ReferenceRows(allocator: std.mem.Allocator, lists: []const []const []const f32) !mlx.mlx_array {
    var flat = std.ArrayList(f32).empty;
    defer flat.deinit(allocator);
    var n: usize = 0;
    for (lists) |rows| for (rows) |row| {
        try flat.appendSlice(allocator, row);
        n += 1;
    };
    const shape = [_]c_int{ 1, @intCast(n), @intCast(flat.items.len / n) };
    return mlx.mlx_array_new_data(flat.items.ptr, &shape, 3, .float32);
}

fn eg2Concat(parts: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const vec = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, s));
    return out;
}

test "EmbeddingGemma 2: the image tower yields the reference's soft tokens" {
    // At the checkpoint's own budget and at twice it: a grid larger than `default_output_length` has no padded
    // shape to fit and must simply run.
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;

    var fx = try Eg2Tiny.openWithVision(allocator, io);
    defer fx.deinit();
    var enc = try vision_mod.VisionEncoder.init(allocator, fx.config, &fx.weights);
    defer enc.deinit();
    const parsed = try std.json.parseFromSlice(Eg2Media, allocator, @embedFile("../../fixtures/embedding_gemma2_tiny_expected.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const media = parsed.value.media;

    const Case = struct { budget: u32, target: [2]u32, tokens: u32, soft: []const []const f32 };
    const cases = [_]Case{
        .{ .budget = eg2_fixture_budget, .target = media.image.target, .tokens = media.image.tokens, .soft = media.image.soft },
        .{ .budget = media.image_big.budget, .target = media.image_big.target, .tokens = media.image_big.tokens, .soft = media.image_big.soft },
    };
    for (cases) |want| {
        const rows = try eg2ImageRows(allocator, &enc, media.image, want.budget, want.target);
        defer _ = mlx.mlx_array_free(rows);
        var f32_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32_rows);
        try mlx.check(mlx.mlx_astype(&f32_rows, rows, .float32, enc.s));
        try mlx.check(mlx.mlx_array_eval(f32_rows));
        const shape = mlx.getShape(f32_rows);
        try testing.expectEqual(@as(c_int, 1), shape[0]);
        try testing.expectEqual(@as(c_int, @intCast(want.tokens)), shape[1]);
        try testing.expectEqual(@as(c_int, @intCast(fx.config.hidden_size)), shape[2]);
        const data = mlx.mlx_array_data_float32(f32_rows).?;
        for (want.soft, 0..) |ref, t| {
            const cos = rowCosine(data[t * ref.len ..][0..ref.len], ref);
            testing.expect(cos > 0.999) catch |e| {
                std.debug.print("budget {d}, soft token {d}: cosine {d:.6}\n", .{ want.budget, t, cos });
                return e;
            };
        }
    }
}

test "EmbeddingGemma 2: the tower's soft tokens are the same padded and masked (Gemma 4) as unpadded and fused" {
    // Gemma 4 pads every image to its one budget and masks the padding out of eager attention; EmbeddingGemma 2
    // runs the bare grid through the fused kernel. Masked padding is a no-op, so the two arms of the one tower
    // must agree, and a break in either shows here without a Gemma 4 checkpoint.
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;

    var fx = try Eg2Tiny.openWithVision(allocator, io);
    defer fx.deinit();
    var as_gemma4 = fx.config;
    as_gemma4.model_type = "gemma4";
    var padded = try vision_mod.VisionEncoder.init(allocator, as_gemma4, &fx.weights);
    defer padded.deinit();
    var bare = try vision_mod.VisionEncoder.init(allocator, fx.config, &fx.weights);
    defer bare.deinit();
    const parsed = try std.json.parseFromSlice(Eg2Media, allocator, @embedFile("../../fixtures/embedding_gemma2_tiny_expected.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value.media;

    const a = try eg2ImageRows(allocator, &padded, m.image, eg2_fixture_budget, m.image.target);
    defer _ = mlx.mlx_array_free(a);
    const b = try eg2ImageRows(allocator, &bare, m.image, eg2_fixture_budget, m.image.target);
    defer _ = mlx.mlx_array_free(b);
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a, .float32, padded.s));
    try mlx.check(mlx.mlx_astype(&b32, b, .float32, bare.s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    try testing.expectEqual(mlx.getShape(a32)[1], mlx.getShape(b32)[1]);
    const hidden: usize = @intCast(mlx.getShape(a32)[2]);
    const rows: usize = @intCast(mlx.getShape(a32)[1]);
    const da = mlx.mlx_array_data_float32(a32).?;
    const db = mlx.mlx_array_data_float32(b32).?;
    for (0..rows) |t| {
        const cos = rowCosine(da[t * hidden ..][0..hidden], db[t * hidden ..][0..hidden]);
        testing.expect(cos > 0.99999) catch |e| {
            std.debug.print("soft token {d}: padded vs bare cosine {d:.6}\n", .{ t, cos });
            return e;
        };
    }
}

test "EmbeddingGemma 2: image, video and mixed prompts embed like the reference" {
    // The reference splices each modality's rows at its placeholder ids; here one scatter takes every item's
    // rows in PROMPT order, and the mixed prompt puts the video before the image on purpose.
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;

    var fx = try Eg2Tiny.openWithVision(allocator, io);
    defer fx.deinit();
    var xfm = try fx.served(io);
    defer xfm.deinit();
    var enc = try vision_mod.VisionEncoder.init(allocator, fx.config, &fx.weights);
    defer enc.deinit();
    const parsed = try std.json.parseFromSlice(Eg2Media, allocator, @embedFile("../../fixtures/embedding_gemma2_tiny_expected.json"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value.media;

    const image = try eg2ImageRows(allocator, &enc, m.image, eg2_fixture_budget, m.image.target);
    defer _ = mlx.mlx_array_free(image);
    const video = try eg2VideoRows(allocator, &enc, m.frames);
    defer _ = mlx.mlx_array_free(video);
    try testing.expectEqual(@as(c_int, @intCast(m.frames.rgb.len * m.frames.tokens)), mlx.getShape(video)[1]);
    const video_then_image = try eg2Concat(&.{ video, image }, xfm.s);
    defer _ = mlx.mlx_array_free(video_then_image);

    const cases = [_]struct { name: []const u8, ids: []const u32, rows: mlx.mlx_array, want: []const f32 }{
        .{ .name = "image", .ids = m.image_ids, .rows = image, .want = m.image_embedding },
        .{ .name = "text+image", .ids = m.text_image_ids, .rows = image, .want = m.text_image_embedding },
        .{ .name = "video", .ids = m.video_ids, .rows = video, .want = m.video_embedding },
        .{ .name = "video then image", .ids = m.mixed_ids, .rows = video_then_image, .want = m.mixed_embedding },
    };
    for (cases) |c| {
        const seqs = [_][]const u32{c.ids};
        const got = try computeEmbeddingsBatchWith(allocator, &xfm, &seqs, c.rows);
        defer {
            for (got) |r| allocator.free(r);
            allocator.free(got);
        }
        const cos = rowCosine(got[0], c.want);
        testing.expect(cos > 0.99) catch |e| {
            std.debug.print("{s}: embedding cosine {d:.6}\n", .{ c.name, cos });
            return e;
        };
    }

    // The reference's own soft tokens through the same splice and trunk: this leg has no tower in it, so it holds
    // the float32 bar the text rows do (the tower casts its inputs to bf16, which a random toy amplifies), and it
    // can tell the prompt's row order from the kinds' (mean pooling alone hardly can).
    const image_seq = [_][]const u32{m.image_ids};
    const mixed_seq = [_][]const u32{m.mixed_ids};
    const ref_image = try eg2ReferenceRows(allocator, &.{m.image.soft});
    defer _ = mlx.mlx_array_free(ref_image);
    const ref_ordered = try eg2ReferenceRows(allocator, &.{ m.frames.soft, m.image.soft });
    defer _ = mlx.mlx_array_free(ref_ordered);
    const ref_swapped = try eg2ReferenceRows(allocator, &.{ m.image.soft, m.frames.soft });
    defer _ = mlx.mlx_array_free(ref_swapped);
    var exact_cos: [3]f64 = undefined;
    for ([_]struct { seq: []const []const u32, rows: mlx.mlx_array, want: []const f32 }{
        .{ .seq = &image_seq, .rows = ref_image, .want = m.image_embedding },
        .{ .seq = &mixed_seq, .rows = ref_ordered, .want = m.mixed_embedding },
        .{ .seq = &mixed_seq, .rows = ref_swapped, .want = m.mixed_embedding },
    }, 0..) |c, k| {
        const got = try computeEmbeddingsBatchWith(allocator, &xfm, c.seq, c.rows);
        defer {
            for (got) |r| allocator.free(r);
            allocator.free(got);
        }
        exact_cos[k] = rowCosine(got[0], c.want);
    }
    try testing.expect(exact_cos[0] > 0.9995);
    try testing.expect(exact_cos[1] > 0.9995);
    try testing.expect(exact_cos[2] < exact_cos[1] - 0.0005);

    // A prompt whose placeholders outnumber the rows would wrap the scatter silently; two sequences cannot share rows.
    const two = [_][]const u32{ m.image_ids, m.image_ids };
    try testing.expectError(error.MultimodalNeedsOneSequence, computeEmbeddingsBatchWith(allocator, &xfm, &two, image));
}

test "EmbeddingGemma 2: a tower missing a tensor is a load error, a missing tower is not" {
    // A pack with its tower cut to pieces would otherwise crash the server at load; a text-only pack (no tower at
    // all) is the benign opt-out the loader already handles.
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;
    {
        var fx = try Eg2Tiny.openWithVision(testing.allocator, io);
        defer fx.deinit();
        fx.weights.remove("vision_tower.encoder.layers.1.self_attn.k_proj.linear.weight");
        try testing.expectError(error.MissingWeight, vision_mod.VisionEncoder.init(testing.allocator, fx.config, &fx.weights));
    }
    {
        var fx = try Eg2Tiny.open(testing.allocator, io);
        defer fx.deinit();
        try testing.expectError(error.MissingVisionWeights, vision_mod.VisionEncoder.init(testing.allocator, fx.config, &fx.weights));
    }
}

test "EmbeddingGemma 2: a checkpoint missing its head or per-layer-input weights is a load error" {
    // Without the head the server would hand out 512-wide un-projected vectors under an arch that promises 768,
    // and a layer without its input block or scalar would crash the first forward.
    const io = std.Io.Threaded.global_single_threaded.io();
    if (mlx.noGpuBackend()) return;
    for ([_][]const u8{
        "language_model.embedding_projection.weight",
        "language_model.ple.per_layer_model_projection.weight",
        "language_model.ple.per_layer_projection_norm.weight",
        "language_model.layers.3.ple_block.per_layer_projection.weight",
        "language_model.layers.2.layer_scalar",
    }) |name| {
        var fx = try Eg2Tiny.open(testing.allocator, io);
        defer fx.deinit();
        fx.weights.remove(name);
        var res = Resident.adopt(fx.weights);
        defer res.deinit();
        fx.weights = model_mod.Weights.init(testing.allocator);
        const src = res.source();
        var fs = try FsPackDir.open(io, fx.path());
        defer fs.close();
        const dir = fs.packDir();
        const b = fx.config.plugin.?;
        var why: abi.Verdict = .{};
        try testing.expect(b.table.load(&.{ .model_dir = .of(fx.path()), .dir = &dir, .parsed = &b.parsed, .stream = .{ .ctx = mlx.gpuStream().ctx }, .weights = &src }, &why) == null);
        try testing.expectEqualStrings("MissingWeight", why.why.slice());
    }
}
