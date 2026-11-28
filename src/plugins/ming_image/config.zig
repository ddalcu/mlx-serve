//! Ming-Image's claims, caps, bills and canvas buckets: MLX-free (discovery and admission run them with no device).
//! The pack is inclusionAI's component tree (`mllm/`, `connector/`, `mlp/`, `transformer/`, `vae/`, `scheduler/`)
//! under a root config.json whose `model_type` (`bailingmm_moe_v2_lite`) the Ming omni chat models share: the
//! image components decide the claim.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");
const pack_claims = @import("../../sdk/media/pack_claims.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "ming_image",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"inclusionAI/Ming-Image-0.2-Design"},
};

pub const model_types = [_][]const u8{"bailingmm_moe_v2_lite"};

/// The last component a pull lands; the pack is invisible until it does.
pub const required_file = "vae/diffusion_pytorch_model.safetensors";

pub const DEFAULT_STEPS: u32 = 12;
/// `pos + s·(pos − neg)` runs only past 1 (the release's own rule), so 1 is one forward per step.
pub const DEFAULT_GUIDANCE: f32 = 1.0;
pub const VAE_DOWNSAMPLE: u32 = 8;
pub const max_dim = 4096;

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    if (!std.mem.eql(u8, model_type.slice(), model_types[0])) return .no;
    if (!pack_claims.jsonStringIs(dir, "transformer/config.json", &.{"_class_name"}, "DiffusionTransformer")) return .no;
    return if (dir.has("mlp/config.json") and dir.has("connector/config.json")) .by_type else .no;
}

/// Text to image, and one-picture edits where the pack ships the MLLM's vision tower and its processor config. The
/// layer-decomposition checkpoint shares the layout; it is not served.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    const edits = dir.has("mllm/preprocessor_config.json");
    out.* = .{ .guidance = true, .default_guidance = DEFAULT_GUIDANCE, .max_dim = max_dim, .edit = edits, .edit_ref_cap = if (edits) 1 else 0 };
}

/// The canvas a request runs at: the release's aspect bucket nearest the asked ratio, from the table whose
/// resolution is nearest √(w·h). The model trained on these sizes only.
pub fn bucket(w: u32, h: u32) Dims {
    const res = std.math.sqrt(@as(f64, @floatFromInt(w)) * @as(f64, @floatFromInt(h)));
    var table: []const Bucket = &buckets_1024;
    var best_res: f64 = std.math.inf(f64);
    for ([_]struct { r: f64, t: []const Bucket }{ .{ .r = 512, .t = &buckets_512 }, .{ .r = 1024, .t = &buckets_1024 }, .{ .r = 2048, .t = &buckets_2048 } }) |c| {
        if (@abs(c.r - res) < best_res) {
            best_res = @abs(c.r - res);
            table = c.t;
        }
    }
    return nearest(table, w, h);
}

/// The table entry whose height/width ratio is nearest `h/w` (the first on a tie, as `get_closest_ratio`).
fn nearest(table: []const Bucket, w: u32, h: u32) Dims {
    const ratio = @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(w));
    var pick = table[0];
    for (table) |b| if (@abs(b.ratio - ratio) < @abs(pick.ratio - ratio)) {
        pick = b;
    };
    return .{ .w = pick.w, .h = pick.h };
}

pub const Dims = struct { w: u32, h: u32 };

/// The edit table: the release's image-edit task runs at the 1024 buckets.
pub const EDIT_HIGHRES: u32 = 1024;

/// An edit's canvas and output (`process_ratio`): the bucket nearest the source's ratio in the `highres` table,
/// and the size the result is resized to: the source's aspect, scaled to cover the bucket on its tighter side.
pub fn editSizes(w: u32, h: u32, highres: u32) struct { canvas: Dims, out: Dims } {
    const table: []const Bucket = switch (highres) {
        512 => &buckets_512,
        2048 => &buckets_2048,
        else => &buckets_1024,
    };
    const c = nearest(table, w, h);
    const fw: f64 = @floatFromInt(w);
    const fh: f64 = @floatFromInt(h);
    const ch: f64 = @floatFromInt(c.h);
    const cw: f64 = @floatFromInt(c.w);
    const out: Dims = if (ch / fh > cw / fw) .{ .w = @intFromFloat(@trunc(fw * ch / fh)), .h = c.h } else .{ .w = c.w, .h = @intFromFloat(@trunc(fh * cw / fw)) };
    return .{ .canvas = c, .out = out };
}
const Bucket = struct { ratio: f64, h: u32, w: u32 };

/// `bailingmm_utils.process_ratio`'s tables: height/width ratio → [height, width].
const buckets_512 = [_]Bucket{
    .{ .ratio = 0.25, .h = 256, .w = 1024 },  .{ .ratio = 0.26, .h = 256, .w = 992 },  .{ .ratio = 0.27, .h = 256, .w = 960 },
    .{ .ratio = 0.28, .h = 256, .w = 928 },   .{ .ratio = 0.32, .h = 288, .w = 896 },  .{ .ratio = 0.33, .h = 288, .w = 864 },
    .{ .ratio = 0.35, .h = 288, .w = 832 },   .{ .ratio = 0.4, .h = 320, .w = 800 },   .{ .ratio = 0.42, .h = 320, .w = 768 },
    .{ .ratio = 0.48, .h = 352, .w = 736 },   .{ .ratio = 0.5, .h = 352, .w = 704 },   .{ .ratio = 0.52, .h = 352, .w = 672 },
    .{ .ratio = 0.5455, .h = 384, .w = 704 }, .{ .ratio = 0.57, .h = 384, .w = 672 },  .{ .ratio = 0.6, .h = 384, .w = 640 },
    .{ .ratio = 0.65, .h = 416, .w = 640 },   .{ .ratio = 0.68, .h = 416, .w = 608 },  .{ .ratio = 0.72, .h = 416, .w = 576 },
    .{ .ratio = 0.78, .h = 448, .w = 576 },   .{ .ratio = 0.82, .h = 448, .w = 544 },  .{ .ratio = 0.88, .h = 480, .w = 544 },
    .{ .ratio = 0.94, .h = 480, .w = 512 },   .{ .ratio = 1.0, .h = 512, .w = 512 },   .{ .ratio = 1.07, .h = 512, .w = 480 },
    .{ .ratio = 1.13, .h = 544, .w = 480 },   .{ .ratio = 1.21, .h = 544, .w = 448 },  .{ .ratio = 1.29, .h = 576, .w = 448 },
    .{ .ratio = 1.38, .h = 576, .w = 416 },   .{ .ratio = 1.46, .h = 608, .w = 416 },  .{ .ratio = 1.5385, .h = 640, .w = 416 },
    .{ .ratio = 1.67, .h = 640, .w = 384 },   .{ .ratio = 1.75, .h = 672, .w = 384 },  .{ .ratio = 1.8333, .h = 704, .w = 384 },
    .{ .ratio = 2.0, .h = 704, .w = 352 },    .{ .ratio = 2.09, .h = 736, .w = 352 },  .{ .ratio = 2.4, .h = 768, .w = 320 },
    .{ .ratio = 2.5, .h = 800, .w = 320 },    .{ .ratio = 2.89, .h = 832, .w = 288 },  .{ .ratio = 3.0, .h = 864, .w = 288 },
    .{ .ratio = 3.11, .h = 896, .w = 288 },   .{ .ratio = 3.62, .h = 928, .w = 256 },  .{ .ratio = 3.75, .h = 960, .w = 256 },
    .{ .ratio = 3.88, .h = 992, .w = 256 },   .{ .ratio = 4.0, .h = 1024, .w = 256 },
};

const buckets_1024 = [_]Bucket{
    .{ .ratio = 0.25, .h = 512, .w = 2048 },  .{ .ratio = 0.26, .h = 512, .w = 1984 },   .{ .ratio = 0.27, .h = 512, .w = 1920 },
    .{ .ratio = 0.28, .h = 512, .w = 1856 },  .{ .ratio = 0.32, .h = 576, .w = 1792 },   .{ .ratio = 0.33, .h = 576, .w = 1728 },
    .{ .ratio = 0.35, .h = 576, .w = 1664 },  .{ .ratio = 0.4, .h = 640, .w = 1600 },    .{ .ratio = 0.42, .h = 640, .w = 1536 },
    .{ .ratio = 0.48, .h = 704, .w = 1472 },  .{ .ratio = 0.5, .h = 704, .w = 1408 },    .{ .ratio = 0.52, .h = 704, .w = 1344 },
    .{ .ratio = 0.5581, .h = 768, .w = 1376 }, .{ .ratio = 0.5625, .h = 720, .w = 1280 }, .{ .ratio = 0.5647, .h = 768, .w = 1360 },
    .{ .ratio = 0.57, .h = 768, .w = 1344 },  .{ .ratio = 0.6, .h = 768, .w = 1280 },    .{ .ratio = 0.622, .h = 816, .w = 1312 },
    .{ .ratio = 0.625, .h = 800, .w = 1280 }, .{ .ratio = 0.65, .h = 832, .w = 1280 },   .{ .ratio = 0.6582, .h = 832, .w = 1264 },
    .{ .ratio = 0.6667, .h = 832, .w = 1248 }, .{ .ratio = 0.6709, .h = 848, .w = 1264 }, .{ .ratio = 0.68, .h = 832, .w = 1216 },
    .{ .ratio = 0.7013, .h = 864, .w = 1232 }, .{ .ratio = 0.72, .h = 832, .w = 1152 },  .{ .ratio = 0.7467, .h = 896, .w = 1200 },
    .{ .ratio = 0.75, .h = 864, .w = 1152 },  .{ .ratio = 0.7568, .h = 896, .w = 1184 },  .{ .ratio = 0.78, .h = 896, .w = 1152 },
    .{ .ratio = 0.8, .h = 896, .w = 1120 },   .{ .ratio = 0.8056, .h = 928, .w = 1152 },  .{ .ratio = 0.82, .h = 896, .w = 1088 },
    .{ .ratio = 0.88, .h = 960, .w = 1088 },  .{ .ratio = 0.94, .h = 960, .w = 1024 },   .{ .ratio = 0.9846, .h = 1024, .w = 1040 },
    .{ .ratio = 1.0, .h = 1024, .w = 1024 },  .{ .ratio = 1.07, .h = 1024, .w = 960 },   .{ .ratio = 1.13, .h = 1088, .w = 960 },
    .{ .ratio = 1.21, .h = 1088, .w = 896 },  .{ .ratio = 1.2414, .h = 1152, .w = 928 },  .{ .ratio = 1.25, .h = 1120, .w = 896 },
    .{ .ratio = 1.2807, .h = 1168, .w = 912 }, .{ .ratio = 1.29, .h = 1152, .w = 896 },   .{ .ratio = 1.3333, .h = 1152, .w = 864 },
    .{ .ratio = 1.3393, .h = 1200, .w = 896 }, .{ .ratio = 1.38, .h = 1152, .w = 832 },   .{ .ratio = 1.46, .h = 1216, .w = 832 },
    .{ .ratio = 1.4906, .h = 1264, .w = 848 }, .{ .ratio = 1.5, .h = 1248, .w = 832 },    .{ .ratio = 1.6, .h = 1280, .w = 800 },
    .{ .ratio = 1.67, .h = 1280, .w = 768 },  .{ .ratio = 1.75, .h = 1344, .w = 768 },   .{ .ratio = 1.7708, .h = 1360, .w = 768 },
    .{ .ratio = 1.7778, .h = 1280, .w = 720 }, .{ .ratio = 2.0, .h = 1408, .w = 704 },    .{ .ratio = 2.09, .h = 1472, .w = 704 },
    .{ .ratio = 2.4, .h = 1536, .w = 640 },   .{ .ratio = 2.5, .h = 1600, .w = 640 },    .{ .ratio = 2.89, .h = 1664, .w = 576 },
    .{ .ratio = 3.0, .h = 1728, .w = 576 },   .{ .ratio = 3.11, .h = 1792, .w = 576 },   .{ .ratio = 3.62, .h = 1856, .w = 512 },
    .{ .ratio = 3.75, .h = 1920, .w = 512 },  .{ .ratio = 3.88, .h = 1984, .w = 512 },   .{ .ratio = 4.0, .h = 2048, .w = 512 },
};

const buckets_2048 = [_]Bucket{
    .{ .ratio = 0.25, .h = 1024, .w = 4096 },  .{ .ratio = 0.26, .h = 1024, .w = 3968 },  .{ .ratio = 0.27, .h = 1024, .w = 3840 },
    .{ .ratio = 0.28, .h = 1024, .w = 3712 },  .{ .ratio = 0.32, .h = 1152, .w = 3584 },  .{ .ratio = 0.33, .h = 1152, .w = 3456 },
    .{ .ratio = 0.35, .h = 1152, .w = 3328 },  .{ .ratio = 0.4, .h = 1280, .w = 3200 },   .{ .ratio = 0.42, .h = 1280, .w = 3072 },
    .{ .ratio = 0.48, .h = 1408, .w = 2944 },  .{ .ratio = 0.5, .h = 1408, .w = 2816 },   .{ .ratio = 0.52, .h = 1408, .w = 2688 },
    .{ .ratio = 0.5625, .h = 1440, .w = 2560 }, .{ .ratio = 0.57, .h = 1536, .w = 2688 }, .{ .ratio = 0.6, .h = 1536, .w = 2560 },
    .{ .ratio = 0.6667, .h = 1664, .w = 2496 }, .{ .ratio = 0.68, .h = 1664, .w = 2432 }, .{ .ratio = 0.72, .h = 1664, .w = 2304 },
    .{ .ratio = 0.75, .h = 1824, .w = 2432 },  .{ .ratio = 0.78, .h = 1792, .w = 2304 },  .{ .ratio = 0.7917, .h = 1824, .w = 2304 },
    .{ .ratio = 0.8, .h = 1792, .w = 2240 },   .{ .ratio = 0.82, .h = 1792, .w = 2176 },  .{ .ratio = 0.88, .h = 1920, .w = 2176 },
    .{ .ratio = 0.94, .h = 1920, .w = 2048 },  .{ .ratio = 1.0, .h = 2048, .w = 2048 },   .{ .ratio = 1.07, .h = 2048, .w = 1920 },
    .{ .ratio = 1.13, .h = 2176, .w = 1920 },  .{ .ratio = 1.21, .h = 2176, .w = 1792 },  .{ .ratio = 1.25, .h = 2240, .w = 1792 },
    .{ .ratio = 1.2632, .h = 2304, .w = 1824 }, .{ .ratio = 1.29, .h = 2304, .w = 1792 }, .{ .ratio = 1.3333, .h = 2432, .w = 1824 },
    .{ .ratio = 1.38, .h = 2304, .w = 1664 },  .{ .ratio = 1.46, .h = 2432, .w = 1664 },  .{ .ratio = 1.5, .h = 2496, .w = 1664 },
    .{ .ratio = 1.67, .h = 2560, .w = 1536 },  .{ .ratio = 1.75, .h = 2688, .w = 1536 },  .{ .ratio = 1.7778, .h = 2560, .w = 1440 },
    .{ .ratio = 2.0, .h = 2816, .w = 1408 },   .{ .ratio = 2.09, .h = 2944, .w = 1408 },  .{ .ratio = 2.4, .h = 3072, .w = 1280 },
    .{ .ratio = 2.5, .h = 3200, .w = 1280 },   .{ .ratio = 2.89, .h = 3328, .w = 1152 },  .{ .ratio = 3.0, .h = 3456, .w = 1152 },
    .{ .ratio = 3.11, .h = 3584, .w = 1152 },  .{ .ratio = 3.62, .h = 3712, .w = 1024 },  .{ .ratio = 3.75, .h = 3840, .w = 1024 },
    .{ .ratio = 3.88, .h = 3968, .w = 1024 },  .{ .ratio = 4.0, .h = 4096, .w = 1024 },   .{ .ratio = 0.2941, .h = 1120, .w = 3808 },
    .{ .ratio = 0.3043, .h = 1120, .w = 3680 }, .{ .ratio = 0.3679, .h = 1248, .w = 3392 }, .{ .ratio = 0.3846, .h = 1280, .w = 3328 },
    .{ .ratio = 0.433, .h = 1344, .w = 3104 }, .{ .ratio = 0.4574, .h = 1376, .w = 3008 }, .{ .ratio = 0.4681, .h = 1408, .w = 3008 },
    .{ .ratio = 0.5465, .h = 1504, .w = 2752 }, .{ .ratio = 0.6296, .h = 1632, .w = 2592 }, .{ .ratio = 0.6582, .h = 1664, .w = 2528 },
    .{ .ratio = 0.7432, .h = 1760, .w = 2368 }, .{ .ratio = 0.8551, .h = 1888, .w = 2208 }, .{ .ratio = 0.9692, .h = 2016, .w = 2080 },
    .{ .ratio = 1.0317, .h = 2080, .w = 2016 }, .{ .ratio = 1.1695, .h = 2208, .w = 1888 }, .{ .ratio = 1.3455, .h = 2368, .w = 1760 },
    .{ .ratio = 1.5192, .h = 2528, .w = 1664 }, .{ .ratio = 1.5882, .h = 2592, .w = 1632 }, .{ .ratio = 1.8298, .h = 2752, .w = 1504 },
    .{ .ratio = 1.913, .h = 2816, .w = 1472 }, .{ .ratio = 2.1364, .h = 3008, .w = 1408 }, .{ .ratio = 2.186, .h = 3008, .w = 1376 },
    .{ .ratio = 2.3095, .h = 3104, .w = 1344 }, .{ .ratio = 2.6, .h = 3328, .w = 1280 },  .{ .ratio = 2.7179, .h = 3392, .w = 1248 },
    .{ .ratio = 3.2857, .h = 3680, .w = 1120 }, .{ .ratio = 3.4, .h = 3808, .w = 1120 },
};

/// A canvas's working memory past the weights: the bf16 VAE decode sets it (about 3 KiB per output pixel), at least
/// the prompt encode and a denoise step.
const PIXEL_BYTES: u64 = 3 << 10;
const BASE_BYTES: u64 = 2 << 30;

pub fn genTransientBytes(w: u32, h: u32) u64 {
    return BASE_BYTES + @as(u64, w) * h * PIXEL_BYTES;
}

/// The weights the engine holds: every component but the MLLM's vision tower, LM head and unused gates, which the
/// files carry (billed anyway: a few GB over), plus a 1024² request.
pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    const sum = pack_bytes.residentWeights(dir, m, false);
    if (sum == 0) return 0; // unknown pack: never block
    return sum + genTransientBytes(1024, 1024);
}

/// The canvas is the job's to refuse against live headroom (the load reserved a 1024² one).
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");

test "ming_image config: claimed by type only with the image components; a chat sibling is not" {
    const files = [_]fixture.File{
        .{ .name = "config.json", .bytes = "{\"model_type\":\"bailingmm_moe_v2_lite\"}" },
        .{ .name = "transformer/config.json", .bytes = "{\"_class_name\":\"DiffusionTransformer\"}" },
        .{ .name = "mlp/config.json", .bytes = "{}" },
        .{ .name = "connector/config.json", .bytes = "{}" },
    };
    const mem: fixture.MemDir = .{ .files = &files };
    const dir = mem.packDir();
    try testing.expectEqual(abi.Claim.by_type, claims(.of("bailingmm_moe_v2_lite"), &dir));
    try testing.expectEqual(abi.Claim.no, claims(.of("flux2"), &dir));
    const omni: fixture.MemDir = .{ .files = files[0..1] };
    try testing.expectEqual(abi.Claim.no, claims(.of("bailingmm_moe_v2_lite"), &omni.packDir()));
}

test "ming_image config: an edit runs at the source's bucket and resizes back to the source's aspect" {
    // `process_ratio(h, w, highres)` from the release.
    const cases = [_]struct { w: u32, h: u32, hr: u32, cw: u32, ch: u32, ow: u32, oh: u32 }{
        .{ .w = 150, .h = 110, .hr = 512, .cw = 576, .ch = 416, .ow = 576, .oh = 422 },
        .{ .w = 1920, .h = 1080, .hr = 1024, .cw = 1280, .ch = 720, .ow = 1280, .oh = 720 },
        .{ .w = 1024, .h = 1536, .hr = 1024, .cw = 832, .ch = 1248, .ow = 832, .oh = 1248 },
        .{ .w = 2560, .h = 640, .hr = 1024, .cw = 2048, .ch = 512, .ow = 2048, .oh = 512 },
    };
    for (cases) |c| {
        const e = editSizes(c.w, c.h, c.hr);
        try testing.expectEqual(Dims{ .w = c.cw, .h = c.ch }, e.canvas);
        try testing.expectEqual(Dims{ .w = c.ow, .h = c.oh }, e.out);
    }
}

test "ming_image config: a request size snaps to the release's nearest bucket" {
    try testing.expectEqual(Dims{ .w = 1024, .h = 1024 }, bucket(1024, 1024));
    try testing.expectEqual(Dims{ .w = 2048, .h = 2048 }, bucket(2048, 2048));
    try testing.expectEqual(Dims{ .w = 512, .h = 512 }, bucket(512, 512));
    try testing.expectEqual(Dims{ .w = 1024, .h = 1024 }, bucket(1000, 1000));
    try testing.expectEqual(Dims{ .w = 1280, .h = 720 }, bucket(1280, 720));
    try testing.expectEqual(Dims{ .w = 768, .h = 1344 }, bucket(768, 1344));
    try testing.expectEqual(Dims{ .w = 1344, .h = 768 }, bucket(1344, 768));
}
