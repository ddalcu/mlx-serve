//! Qwen-Image-2.1's claims, caps and bills: MLX-free (discovery and admission run them with no device). The
//! mlx-community layout ships no root config.json; its `model_index.json` names the pipeline.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const bill = @import("../../sdk/media/bill.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");
const pack_claims = @import("../../sdk/media/pack_claims.zig");
const image_request = @import("../../sdk/media/image_request.zig");

/// [0] is the type discovery reports for a configless (mlx-community) pack.
pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "qwen_image",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"mlx-community/Qwen-Image-2.1-4bit"},
};

pub const model_types = [_][]const u8{"qwen_image21"};

/// No marker (as discovery has always treated these packs).
pub const required_file = "";

pub const VAE_DOWNSAMPLE: u32 = 16;
pub const MAX_PROMPT_TOKENS: u64 = 2048;
pub const max_dim = 2048;
/// References one edit takes, the primary included: each ~1 MP reference adds ~4096 DiT tokens.
pub const edit_ref_cap = 10;

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    const mt = model_type.slice();
    if (std.mem.startsWith(u8, mt, "qwen_image")) return .by_type;
    // The 2.0 family spells "QwenImagePipeline": a different architecture, never matched.
    if (mt.len == 0 and pack_claims.modelIndexClassIs(dir, "QwenImage21Pipeline")) return .by_marker;
    return .no;
}

/// The Qwen3-VL tower in `text_encoder/` (`model.visual.*`, or mlx-community's `vision_tower.*`): the pack edits.
pub fn towerPresent(dir: *const abi.PackDir) bool {
    return pack_claims.shardsName(dir, "text_encoder", "model.visual.") or pack_claims.shardsName(dir, "text_encoder", "\"vision_tower.");
}

/// Variation always (the VAE encoder loads on first use), real CFG, transparency; edits with the tower.
pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    const edits = towerPresent(dir);
    out.* = .{
        .img2img = true,
        .edit = edits,
        .edit_raw_bytes = edits,
        .guidance = true,
        .default_guidance = 1.0,
        .max_dim = max_dim,
        .edit_ref_cap = if (edits) edit_ref_cap else 0,
        .lora_arch = .of("qwen_image21"),
    };
}

// ── bills ──

/// What a text-to-image request holds above the weights. The denoise is the peak (the banded VAE decode stays
/// below it) and grows per joint token: the canvas's latents plus a full-length prompt.
const GEN_TOKEN_BYTES: u64 = 640 << 10;
const GEN_BASE_BYTES: u64 = 1 << 30;

pub fn genTransientBytes(width: u32, height: u32) u64 {
    const latents: u64 = @as(u64, width / VAE_DOWNSAMPLE) * (height / VAE_DOWNSAMPLE);
    return GEN_BASE_BYTES + (latents + MAX_PROMPT_TOKENS) * GEN_TOKEN_BYTES;
}

/// The load plan reserves a default-size request.
const GEN_TRANSIENT_BYTES: u64 = genTransientBytes(1024, 1024);

/// An edit denoise plus the f32 VAE decode, on top of the weights; the tower rides the text encoder's bytes.
pub const EDIT_TRANSIENT_BYTES: u64 = 6 << 30;

/// The tower's encode working set coexists with the loaded encoder during conditioning: the staged bill must not
/// let the stage max() absorb it.
const EDIT_ENCODE_WS_BYTES: u64 = 2 << 30;

/// The text encoder is half the pack and idle for the whole denoise, so a machine the full set would crowd (past
/// 3/4 of the GPU working set) loads it per request and frees it before the first DiT forward. The engine and the
/// bill read this ONE answer.
pub fn stagesTextEncoder(weights: u64, working_set: u64) bool {
    if (working_set == 0) return false;
    return (weights + GEN_TRANSIENT_BYTES) / 3 > working_set / 4;
}

const Split = struct { te: u64, rest: u64 };

/// The pack's text encoder and everything else, as `--w8a8` leaves them resident.
fn split(dir: *const abi.PackDir, m: *const abi.Machine) Split {
    const te = bill.safetensorsIn(dir, "text_encoder");
    return .{ .te = te, .rest = pack_bytes.residentWeights(dir, m, true) -| te };
}

/// Whether this pack stages its text encoder on this machine.
pub fn staged(dir: *const abi.PackDir, m: *const abi.Machine) bool {
    const w = split(dir, m);
    return stagesTextEncoder(w.te + w.rest, m.gpu_working_set);
}

/// A staged encoder is a stage beside the DiT and VAE; the edit pack's encode stage carries the tower's working set
/// and its denoise the heavier edit transient.
pub fn peakBytes(text_encoder: u64, rest: u64, working_set: u64, has_tower: bool) u64 {
    const transient: u64 = if (has_tower) EDIT_TRANSIENT_BYTES else GEN_TRANSIENT_BYTES;
    if (stagesTextEncoder(text_encoder + rest, working_set)) {
        const te_stage: u64 = if (has_tower) text_encoder +| EDIT_ENCODE_WS_BYTES else text_encoder;
        return rest + @max(te_stage, transient);
    }
    return text_encoder + rest + transient;
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    const w = split(dir, m);
    return peakBytes(w.te, w.rest, m.gpu_working_set, towerPresent(dir));
}

/// A text-to-image canvas past the default the load reserved can run past the working set, and that Metal OOM
/// mid-denoise is uncatchable. An edit's bill needs its references' sizes: the job checks it (`editPlan`).
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}

/// The request-scope edit transient: the widest attention's f32 scores (heads x q x joint kv), the allocation that
/// hangs the first denoise step with no error when it runs past the working set, plus ~65 MB per 1024-conditioning
/// reference.
const EDIT_TEXT_TOKENS: u64 = 2600; // ti2i prompt budget upper bound
const EDIT_HEADS: u64 = 32; // the checkpoint's num_attention_heads

pub fn editTransientBytes(refs: u32, ref_resolution: u32, out_w: u32, out_h: u32) u64 {
    const rt: u64 = @as(u64, ref_resolution / 16) * (ref_resolution / 16);
    const ot: u64 = @as(u64, out_w / 16) * (out_h / 16);
    const joint: u64 = @as(u64, refs) * rt + ot + EDIT_TEXT_TOKENS;
    const scores: u64 = EDIT_HEADS * @max(ot, rt) * joint * 4;
    const persistent: u64 = @as(u64, refs) * rt * (65 << 20) / 4096;
    return scores + persistent;
}

/// BF16 K/V for all layers beside the first step's attention; each CFG branch keeps its own text and references.
pub fn editPrefixBytes(refs: u32, ref_resolution: u32, layers: u64, hidden: u64, branches: u32) u64 {
    const rt: u64 = @as(u64, ref_resolution / 16) * (ref_resolution / 16);
    const prefix = @as(u64, refs) * rt + EDIT_TEXT_TOKENS;
    return prefix * layers * hidden * 2 * 2 * branches;
}

/// Prefer the cached edit, but run uncached when only that fits; null when neither does.
pub fn editPlan(steps: u32, base_bill: u64, prefix_bill: u64, headroom: u64) ?bool {
    if (base_bill -| EDIT_TRANSIENT_BYTES > headroom) return null;
    return steps > 1 and (base_bill +| prefix_bill) -| EDIT_TRANSIENT_BYTES <= headroom;
}

/// An edit's output. A size FLOORS to the DiT's /32 grid (a ceiling, never an upscale); none = the LAST condition
/// image's aspect at the default 1024² (diffusers calculate_dimensions + its own /32 round); then [32, max_dim].
pub fn editDims(explicit: ?image_request.Dims, last_w: u32, last_h: u32) image_request.Dims {
    var w: u32 = undefined;
    var h: u32 = undefined;
    if (explicit) |e| {
        w = e.w / 32 * 32;
        h = e.h / 32 * 32;
    } else {
        const t = editTargetSize(last_w, last_h, 1024);
        w = snap32(t.w);
        h = snap32(t.h);
    }
    return .{ .w = std.math.clamp(w, 32, max_dim), .h = std.math.clamp(h, 32, max_dim) };
}

/// diffusers calculate_dimensions: source aspect at output_resolution², round() each side.
pub fn editTargetSize(src_w: u32, src_h: u32, output_resolution: u32) image_request.Dims {
    const wf: f64 = @floatFromInt(src_w);
    const hf: f64 = @floatFromInt(src_h);
    const res: f64 = @floatFromInt(output_resolution);
    const w: u32 = @intFromFloat(@round(@sqrt(res * res * (wf / hf))));
    // Multiply by src_h before dividing by src_w: an exact .5 quotient (a 4:3 source) stays exact in f64.
    const h: u32 = @intFromFloat(@round(@as(f64, @floatFromInt(w)) * hf / wf));
    return .{ .w = w, .h = h };
}

/// round(x/32)*32 with a u64 intermediate (x+16 can wrap u32).
fn snap32(x: u32) u32 {
    const r: u64 = (@as(u64, x) + 16) / 32 * 32;
    return @intCast(@min(r, @as(u64, std.math.maxInt(u32)) / 32 * 32));
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../../sdk/testing/fixture.zig");
const tensors = @import("../../sdk/testing/tensors.zig");

test "qwen_image config: claimed by type or by its model_index; the tower decides the edit" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var lang = tensors.Tensors.init(arena.allocator(), 1);
    try lang.fill("model.language_model.norm.weight", .{}, &.{2}, .BF16, 1);
    var tower = tensors.Tensors.init(arena.allocator(), 1);
    try tower.fill("vision_tower.patch_embed.proj.weight", .{}, &.{2}, .BF16, 1);
    const t2i: fixture.MemDir = .{ .files = &.{
        .{ .name = "model_index.json", .bytes = "{\"_class_name\":\"QwenImage21Pipeline\"}" },
        .{ .name = "text_encoder/model.safetensors", .bytes = try lang.bytes() },
    } };
    const d = t2i.packDir();
    try testing.expectEqual(abi.Claim.by_marker, claims(.{}, &d));
    try testing.expectEqual(abi.Claim.by_type, claims(.of("qwen_image21"), &d));
    try testing.expectEqual(abi.Claim.no, claims(.of("qwen3_tts"), &d));
    var c: abi.MediaCaps = .{};
    caps(&d, &c);
    try testing.expect(c.img2img and c.guidance and !c.edit and c.edit_ref_cap == 0);
    const edit: fixture.MemDir = .{ .files = &.{.{ .name = "text_encoder/model-1.safetensors", .bytes = try tower.bytes() }} };
    const e = edit.packDir();
    caps(&e, &c);
    try testing.expect(c.edit and c.edit_raw_bytes and c.edit_ref_cap == 10);
    const other: fixture.MemDir = .{ .files = &.{.{ .name = "model_index.json", .bytes = "{\"_class_name\":\"QwenImagePipeline\"}" }} };
    try testing.expectEqual(abi.Claim.no, claims(.{}, &other.packDir()));
}

test "qwen_image config: the load bill covers every request; canvas and edit are checked by the job" {
    const mem: fixture.MemDir = .{ .files = &.{} };
    const d = mem.packDir();
    const m: abi.Machine = .{};
    try testing.expectEqual(@as(u64, 0), billRequest(&d, &m, .of("{\"size\":\"2048x2048\"}")));
    try testing.expectEqual(@as(?bool, null), editPlan(4, EDIT_TRANSIENT_BYTES + 2, 0, 1));
    try testing.expectEqual(@as(?bool, false), editPlan(4, EDIT_TRANSIENT_BYTES + 1, 2, 2));
    try testing.expectEqual(@as(?bool, true), editPlan(4, EDIT_TRANSIENT_BYTES + 1, 1, 2));
    try testing.expectEqual(@as(?bool, false), editPlan(1, EDIT_TRANSIENT_BYTES, 0, 0));
}

test "qwen_image config: the encoder stages past three quarters of the working set, or when forced" {
    const gb: u64 = 1 << 30;
    try testing.expect(!stagesTextEncoder(10 * gb, 0));
    try testing.expect(!stagesTextEncoder(10 * gb, 64 * gb));
    try testing.expect(stagesTextEncoder(40 * gb, 48 * gb));
    // Staged, the encoder and the denoise are stages beside the resident DiT and VAE.
    try testing.expectEqual(15 * gb + GEN_TRANSIENT_BYTES, peakBytes(5 * gb, 10 * gb, 0, false));
    try testing.expectEqual(15 * gb, peakBytes(5 * gb, 10 * gb, 8 * gb, false));
    try testing.expectEqual(17 * gb, peakBytes(5 * gb, 10 * gb, 8 * gb, true));
}

test "qwen_image config: edit dims follow the last reference, a size floors to /32" {
    try testing.expectEqual(image_request.Dims{ .w = 1184, .h = 896 }, editDims(null, 4032, 3024));
    try testing.expectEqual(image_request.Dims{ .w = 512, .h = 736 }, editDims(.{ .w = 527, .h = 750 }, 1, 1));
    try testing.expectEqual(image_request.Dims{ .w = 2048, .h = 32 }, editDims(.{ .w = 4000, .h = 10 }, 1, 1));
}

test "qwen_image config: the text-to-image bill covers the measured peak at every canvas" {
    const GiB: f64 = 1 << 30;
    // Peaks above the weights with a full-length prompt, the same on the 4- and 8-bit packs.
    const measured = [_]struct { w: u32, h: u32, peak: f64 }{ .{ .w = 1024, .h = 1024, .peak = 4.40 }, .{ .w = 2048, .h = 2048, .peak = 11.85 } };
    for (measured) |m| {
        const gib = @as(f64, @floatFromInt(genTransientBytes(m.w, m.h))) / GiB;
        try testing.expect(gib >= m.peak and gib < m.peak * 1.1);
    }
}

test "qwen_image config: an edit pack bills the tower's heavier transient on the same staging answer" {
    const GB: u64 = 1 << 30;
    try testing.expectEqual(@as(u64, 6 << 30), EDIT_TRANSIENT_BYTES);
    // Staged 8-bit pack (TE 9 + rest 9) on a ~22 GB working set: the TE stage (TE + the tower's 2 GiB) dominates.
    try testing.expectEqual(20 * GB, peakBytes(9 * GB, 9 * GB, 22 * GB, true));
    // Resident: everything coexists.
    try testing.expectEqual(24 * GB, peakBytes(9 * GB, 9 * GB, 48 * GB, true));
    try testing.expectEqual(18 * GB + GEN_TRANSIENT_BYTES, peakBytes(9 * GB, 9 * GB, 48 * GB, false));
    // Staged 4-bit pack (TE 2 + rest 4): the edit transient is the biggest stage.
    try testing.expectEqual(10 * GB, peakBytes(2 * GB, 4 * GB, 11 * GB, true));
    try testing.expectEqual(4 * GB + GEN_TRANSIENT_BYTES, peakBytes(2 * GB, 4 * GB, 11 * GB, false));
}

test "qwen_image config: the edit transient scales with refs x ref tokens; the plan picks cached, uncached or none" {
    const GB: u64 = 1 << 30;
    try testing.expect(editTransientBytes(1, 1024, 256, 256) < EDIT_TRANSIENT_BYTES);
    const five = editTransientBytes(5, 1024, 256, 256);
    try testing.expect(five > EDIT_TRANSIENT_BYTES and five < 20 * GB);
    // 10 references plus a 2048x2048 target is past any working set (#496).
    const ten = editTransientBytes(10, 1024, 2048, 2048);
    try testing.expect(ten > 100 * GB);
    try testing.expect(editTransientBytes(10, 512, 2048, 2048) * 2 < ten);
    const base = EDIT_TRANSIENT_BYTES + 2 * GB;
    const prefix = 4 * GB;
    try testing.expectEqual(@as(?bool, true), editPlan(20, base, prefix, 6 * GB));
    try testing.expectEqual(@as(?bool, false), editPlan(20, base, prefix, 2 * GB));
    try testing.expectEqual(@as(?bool, null), editPlan(20, base, prefix, 2 * GB - 1));
    try testing.expectEqual(@as(?bool, null), editPlan(20, base, prefix, 0));
    try testing.expectEqual(@as(?bool, true), editPlan(20, 1 * GB, 2 * GB, 0));
    try testing.expectEqual(@as(?bool, false), editPlan(20, base, std.math.maxInt(u64), 6 * GB));
}


test "qwen_image config: the edit prefix bill counts every layer and each CFG branch" {
    const one = editPrefixBytes(1, 1024, 32, 4096, 1); // the release's 32 layers x 4096 wide
    try testing.expectEqual(@as(u64, (4096 + 2600) * 32 * 4096 * 4), one);
    try testing.expectEqual(2 * one, editPrefixBytes(1, 1024, 32, 4096, 2));
    try testing.expectEqual(@as(u64, 0), editPrefixBytes(10, 1024, 32, 4096, 0));
    try testing.expect(editPrefixBytes(10, 1024, 32, 4096, 2) > 40 << 30);
    try testing.expect(editPrefixBytes(10, 512, 32, 4096, 1) < editPrefixBytes(10, 1024, 32, 4096, 1));
    try testing.expectEqual(@as(u64, (4096 + 2600) * 2 * 32 * 4), editPrefixBytes(1, 1024, 2, 32, 1));
}
