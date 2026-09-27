//! Qwen-Image-2.1 instruction EDIT — the ti2i (text+images→image) VLM
//! conditioning path. The t2i half (`qwen_image.zig`) conditions the DiT on
//! text alone; an edit feeds the SAME Qwen3-VL text encoder a joint sequence:
//! references spliced at `<|image_pad|>` runs, DeepStack features added at LM
//! layers 0/1/2, and the hidden taken BEFORE the LM's final RMS norm (what the
//! DiT was trained on — diffusers neutralizes that norm with a forward hook).
//!
//! Scope: the VLM side — templating, preprocessing, the tower, and
//! `encodeTi2i`; the DiT-side joint construction and the denoise loop live
//! in `qwen_image.zig` (`Engine.editImage`).
//!
//! Differences from mage's `encodeEdit` (each load-bearing, pinned by the
//! `QwenImage edit` env-gated oracles against
//! `tests/dump_qwen_image21_fixtures.py`):
//!  - the LM rope is 3-D INTERLEAVED M-RoPE (mrope_section 24/20/20, theta 5e6,
//!    text (p,p,p), image pads (t,h,w) grid positions per transformers 5.17
//!    `get_rope_index`), not mage's 1-D positions;
//!  - NO final RMS norm and NO cap: the drop is the ti2i system-prefix token
//!    count (14 on the shipped template), not mage's 64/2048 constants;
//!  - the attention mask is causal + padding only (no bidirectional image
//!    attention in transformers 5.17 Qwen3-VL).

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const tok_mod = @import("tokenizer.zig");
const mage_flow = @import("mage_flow.zig");
const mrope = @import("mrope.zig");
const qvis = @import("qwen_vision.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;
const A = mlx.mlx_array;
const TextEncoder = mage_flow.TextEncoder;
const VisionTower = mage_flow.VisionTower;

/// `<|image_pad|>` — also `TextEncoder`'s splice marker (private there).
const TE_IMAGE_TOKEN: i32 = 151655;
const IMAGE_PAD_U32: u32 = 151655;
const VIDEO_PAD_U32: u32 = 151656;
const VISION_START_U32: u32 = 151652;
/// The LM's rotary width and base — shared by the tiny fixture pack and the
/// real 8B (mage's `TextEncoder` hardcodes the same geometry).
const TE_HEAD_DIM: usize = 128;
const TE_THETA: f64 = 5_000_000.0;
/// transformers 5.17 Qwen3-VL interleaved mrope_section (text/h/w lanes).
const MROPE_SECTION: [3]u32 = .{ 24, 20, 20 };
/// The Qwen3-VL tower's position table is checkpoint-constant: 48×48 = 2304.
const VIT_POS_TABLE: usize = 2304;
const VIT_PATCH: u32 = 16;
const VIT_MERGE: u32 = 2;
/// RGB patch row width: in_ch(3) × temporal(2) × patch(16)².
const VIT_PATCH_IN: usize = 1536;

// ── mlx primitives (file-local, mirroring qwen_image.zig) ──

inline fn free(a: A) void {
    _ = mlx.mlx_array_free(a);
}
inline fn astype(x: A, dt: mlx.mlx_dtype, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}
inline fn reshape(x: A, shape: []const c_int, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}
inline fn contig(x: A, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_contiguous(&o, x, false, s));
    return o;
}
/// `[1, seq, C] → [1, end-start, C]`, materialized (a live slice pins its parent).
fn sliceSeq(x: A, start: c_int, end: c_int, s: S) !A {
    const sh = mlx.getShape(x);
    const lo = [_]c_int{ 0, start, 0 };
    const hi = [_]c_int{ sh[0], end, sh[2] };
    const st = [_]c_int{ 1, 1, 1 };
    var o = mlx.mlx_array_new();
    defer free(o);
    try mlx.check(mlx.mlx_slice(&o, x, &lo, sh.len, &hi, sh.len, &st, sh.len, s));
    return contig(o, s);
}

// ── ti2i prompt template ──

/// The pipeline's system prompt (both templates open with it verbatim).
pub const TI2I_SYS_PROMPT = "Comprehend and analyze the provided prompt.";
/// The ti2i system prefix. Its TOKEN COUNT is the conditioning drop — the
/// number of leading hidden rows removed before the sequence reaches the DiT
/// (`edit_drop_idx` in the fixture; 14 on the shipped tokenizer).
pub const TI2I_SYSTEM_PREFIX = "<|im_start|>system\n" ++ TI2I_SYS_PROMPT ++ "<|im_end|>\n";
const USER_PREFIX = "<|im_start|>user\n";
const TAIL = "<|im_end|>\n<|im_start|>assistant\n";

/// Render the ti2i prompt (diffusers `prompt_template_ti2i`, the `<imageN>`
/// chain substituted for N references): the instruction lands DIRECTLY after
/// the last `<|vision_end|>`, no separator. An empty/whitespace prompt becomes
/// `" "` — Qwen has no BOS, so an empty string would leave the encoder with
/// nothing to read. The NEGATIVE prompt renders through this same template
/// (the pipeline has no separate negative variant; only the text differs).
pub fn buildTi2iPrompt(allocator: std.mem.Allocator, prompt: []const u8, n_images: usize) ![]u8 {
    const body = if (std.mem.trim(u8, prompt, " \t\r\n").len == 0) " " else prompt;
    var sb: std.ArrayList(u8) = .empty;
    errdefer sb.deinit(allocator);
    try sb.appendSlice(allocator, TI2I_SYSTEM_PREFIX);
    try sb.appendSlice(allocator, USER_PREFIX);
    var num: [16]u8 = undefined;
    for (0..n_images) |k| {
        if (k > 0) try sb.append(allocator, ' ');
        try sb.appendSlice(allocator, "<image");
        try sb.appendSlice(allocator, try std.fmt.bufPrint(&num, "{d}", .{k + 1}));
        try sb.appendSlice(allocator, "><|vision_start|><|image_pad|><|vision_end|>");
    }
    try sb.appendSlice(allocator, body);
    try sb.appendSlice(allocator, TAIL);
    return sb.toOwnedSlice(allocator);
}

/// Expand each single `<|image_pad|>` placeholder to its merged-grid token
/// count (`t·h·w/4` per reference) — the processor's placeholder expansion.
/// The placeholder count must equal the reference count or the splice below
/// would mismatch the vision features.
pub fn expandImagePads(allocator: std.mem.Allocator, ids: []const i32, slots_per_image: []const usize) ![]i32 {
    var npads: usize = 0;
    for (ids) |t| {
        if (t == TE_IMAGE_TOKEN) npads += 1;
    }
    if (npads != slots_per_image.len) {
        logPadCountMismatch(npads, slots_per_image.len);
        return error.QwenEditPadCountMismatch;
    }
    var total: usize = ids.len;
    for (slots_per_image) |n| total += n - 1;
    const out = try allocator.alloc(i32, total);
    var oi: usize = 0;
    var si: usize = 0;
    for (ids) |t| {
        if (t != TE_IMAGE_TOKEN) {
            out[oi] = t;
            oi += 1;
            continue;
        }
        const n = slots_per_image[si];
        si += 1;
        @memset(out[oi .. oi + n], TE_IMAGE_TOKEN);
        oi += n;
    }
    return out;
}

fn logPadCountMismatch(npads: usize, nslots: usize) void {
    log.err("[qwen-image-edit] {d} <|image_pad|> placeholders but {d} reference images\n", .{ npads, nslots });
}

// ── reference-image preprocessing ──

/// One reference image, resized once at the caller's target (W,H) — the same
/// resize feeds both consumers (the pipeline's "one resize per ref").
pub const EditImage = struct {
    /// [1,3,H,W] f32 [0,1] — alpha-composited over WHITE (the tower's copy).
    rgb: A,
    /// [1,4,H,W] f32 [-1,1] — the VAE reads all four channels.
    vae_in: A,
    w: u32,
    h: u32,

    pub fn deinit(self: *EditImage) void {
        free(self.rgb);
        free(self.vae_in);
    }
};

/// Decode an encoded reference (PNG/JPEG/WebP) to RGBA and preprocess at the
/// target dims. `(w, h)` are the RESIZE TARGET — the caller computes them
/// (aspect at output_resolution², /32-snapped), never this file.
pub fn prepareEditImage(allocator: std.mem.Allocator, s: S, bytes: []const u8, w: u32, h: u32) !EditImage {
    const stb = @import("stb");
    var sw: c_int = 0;
    var sh: c_int = 0;
    var ch: c_int = 0;
    const src = stb.stbi_load_from_memory(bytes.ptr, @intCast(bytes.len), &sw, &sh, &ch, 4) orelse
        return error.ImageDecodeFailed;
    defer stb.stbi_image_free(src);
    if (sw <= 0 or sh <= 0) return error.ImageDecodeFailed;
    const n: usize = @as(usize, @intCast(sw)) * @as(usize, @intCast(sh)) * 4;
    return prepareEditRgba(allocator, s, src[0..n], @intCast(sw), @intCast(sh), w, h);
}

/// The pure RGBA path over already-decoded bytes (interleaved RGBA8,
/// `src_w*src_h*4`). ONE Pillow-compatible LANCZOS resize of the 4-channel
/// source; the VLM copy is then composited over white, the VAE copy keeps the
/// alpha. Both normalized from u8 exactly as the reference processors do.
pub fn prepareEditRgba(allocator: std.mem.Allocator, s: S, rgba: []const u8, src_w: u32, src_h: u32, tgt_w: u32, tgt_h: u32) !EditImage {
    const a = allocator;
    const plane: usize = @as(usize, tgt_w) * tgt_h;
    if (rgba.len != @as(usize, src_w) * src_h * 4) return error.InvalidImageBuffer;

    // PIL resizes RGBA PREMULTIPLIED: pm = MULDIV255(c, a), resample all four
    // channels, then un-premultiply (floor div by alpha, clipped; alpha 0 keeps
    // the numerator). qwen_vision's Pillow-compatible separable kernel rides
    // the per-pass u8 quantization, matching within the oracle tolerance.
    const premul_buf = try a.alloc(u8, rgba.len);
    defer a.free(premul_buf);
    for (rgba, 0..) |v, i| premul_buf[i] = v;
    for (0..@as(usize, src_w) * src_h) |i| {
        const al: i64 = rgba[i * 4 + 3];
        for (0..3) |c| {
            premul_buf[i * 4 + c] = @intCast(muldiv255(rgba[i * 4 + c], al));
        }
    }
    const resized = try qvis.resizeInterleavedPil(a, premul_buf, src_h, src_w, tgt_h, tgt_w, 4, .lanczos);
    defer a.free(resized);
    // Un-premultiply the RGB planes in place (alpha plane is final).
    for (0..plane) |i| {
        const al: i64 = resized[i * 4 + 3];
        if (al == 0) continue;
        for (0..3) |c| {
            const v = @min(@as(i64, 255), @divFloor(@as(i64, resized[i * 4 + c]) * 255, al));
            resized[i * 4 + c] = @intCast(v);
        }
    }

    const rgb_buf = try a.alloc(f32, plane * 3);
    defer a.free(rgb_buf);
    const vae_buf = try a.alloc(f32, plane * 4);
    defer a.free(vae_buf);
    for (0..plane) |i| {
        const r: i64 = resized[i * 4 + 0];
        const g: i64 = resized[i * 4 + 1];
        const b: i64 = resized[i * 4 + 2];
        const al: i64 = resized[i * 4 + 3];
        // PIL paste-with-mask blend, dst = white: out = MULDIV255(255, 255-a)
        // + MULDIV255(src, a) — exact per-channel u8 arithmetic.
        rgb_buf[i] = @as(f32, @floatFromInt(muldiv255(255, 255 - al) + muldiv255(r, al))) / 255.0;
        rgb_buf[plane + i] = @as(f32, @floatFromInt(muldiv255(255, 255 - al) + muldiv255(g, al))) / 255.0;
        rgb_buf[2 * plane + i] = @as(f32, @floatFromInt(muldiv255(255, 255 - al) + muldiv255(b, al))) / 255.0;
        for (0..4) |c| vae_buf[c * plane + i] = @as(f32, @floatFromInt(resized[i * 4 + c])) / 127.5 - 1.0;
    }

    const rgb_shape = [_]c_int{ 1, 3, @intCast(tgt_h), @intCast(tgt_w) };
    const vae_shape = [_]c_int{ 1, 4, @intCast(tgt_h), @intCast(tgt_w) };
    const rgb_raw = mlx.mlx_array_new_data(rgb_buf.ptr, &rgb_shape, 4, .float32);
    defer free(rgb_raw);
    const vae_raw = mlx.mlx_array_new_data(vae_buf.ptr, &vae_shape, 4, .float32);
    defer free(vae_raw);
    return .{
        .rgb = try contig(rgb_raw, s),
        .vae_in = try contig(vae_raw, s),
        .w = tgt_w,
        .h = tgt_h,
    };
}

/// PIL's `MULDIV255` blend rounding: round(x·y/255) with the carry form.
fn muldiv255(x: i64, y: i64) i64 {
    const t = x * y + 128;
    return (t + (t >> 8)) >> 8;
}

/// Patchify the composited RGB into the tower's `pixel_values`
/// [grid_h·grid_w, 1536]: 16/16 patches in merge-block token order, feature
/// layout [C, tps, py, px] with the temporal slots duplicating the frame
/// (Qwen's own image processor does this for a still image).
pub fn vlmPixelValues(allocator: std.mem.Allocator, s: S, rgb: A, grid_h: u32, grid_w: u32) !A {
    const a = allocator;
    const sh = mlx.getShape(rgb);
    if (sh[0] != 1 or sh[1] != 3) return error.InvalidRgbShape;
    const rh: u32 = grid_h * VIT_PATCH;
    const rw: u32 = grid_w * VIT_PATCH;
    if (sh[2] != @as(c_int, @intCast(rh)) or sh[3] != @as(c_int, @intCast(rw))) return error.InvalidRgbShape;

    const plane: usize = @as(usize, rh) * rw;
    const chw = try a.alloc(f32, plane * 3);
    defer a.free(chw);
    {
        const c = try astype(rgb, .float32, s);
        defer free(c);
        try mlx.check(mlx.mlx_array_eval(c));
        const d = mlx.mlx_array_data_float32(c) orelse return error.NoData;
        @memcpy(chw, d[0 .. plane * 3]);
    }
    // The processor's (x/255 - 0.5)/0.5 over u8-derived [0,1] values.
    for (chw) |*v| v.* = v.* * 2.0 - 1.0;

    const npatch: usize = @as(usize, grid_h) * grid_w;
    const pv_buf = try a.alloc(f32, npatch * VIT_PATCH_IN);
    defer a.free(pv_buf);
    qvis.buildPixelValues(pv_buf, chw, 3, rh, rw, VIT_PATCH, 2, VIT_MERGE);
    const shape = [_]c_int{ @intCast(npatch), VIT_PATCH_IN };
    const raw = mlx.mlx_array_new_data(pv_buf.ptr, &shape, 2, .float32);
    defer free(raw);
    return contig(raw, s);
}

// ── 3-D interleaved M-RoPE for the LM layers ──

/// cos/sin [seq, 128] (rotate-half duplicated halves) in the compute dtype:
/// text tokens advance (p,p,p) on the three axes, image-pad tokens get
/// (t,h,w) grid positions per image — transformers 5.17 `get_rope_index`
/// (a faithful single-sequence port exists: `mrope.getRopeIndex`). The 64
/// half-lanes map to axes via `mrope.interleavedSelector` (h claims 1,4,…;
/// w claims 2,5,…, each bounded by section·3; the rest stay t).
fn buildEditRope(allocator: std.mem.Allocator, ids: []const i32, grids: []const [3]i64, dtype: mlx.mlx_dtype, s: S) !struct { cos: A, sin: A } {
    const a = allocator;
    const seq = ids.len;
    const tokens = try a.alloc(u32, seq);
    defer a.free(tokens);
    for (ids, 0..) |t, i| tokens[i] = @intCast(t);
    const images = try a.alloc(mrope.ImageGrid, grids.len);
    defer a.free(images);
    for (grids, 0..) |g, i| images[i] = .{ .t = @intCast(g[0]), .h = @intCast(g[1]), .w = @intCast(g[2]) };

    var ri = try mrope.getRopeIndex(a, tokens, images, &.{}, IMAGE_PAD_U32, VIDEO_PAD_U32, VISION_START_U32, VIT_MERGE);
    defer ri.deinit();

    const pos = try a.alloc(i32, 3 * seq);
    defer a.free(pos);
    @memcpy(pos[0..seq], ri.pos[0]);
    @memcpy(pos[seq .. 2 * seq], ri.pos[1]);
    @memcpy(pos[2 * seq .. 3 * seq], ri.pos[2]);
    const ctx = mrope.PositionContext{ .pos = pos, .total = seq, .delta = ri.delta };

    const half = TE_HEAD_DIM / 2;
    var inv_freq: [half]f64 = undefined;
    mrope.computeInvFreq(&inv_freq, TE_HEAD_DIM, TE_THETA);
    var sel: [half]u8 = undefined;
    mrope.interleavedSelector(&sel, MROPE_SECTION);

    const cos_buf = try a.alloc(f32, seq * TE_HEAD_DIM);
    defer a.free(cos_buf);
    const sin_buf = try a.alloc(f32, seq * TE_HEAD_DIM);
    defer a.free(sin_buf);
    mrope.fillCosSin(cos_buf, sin_buf, ctx, 0, 1, seq, &inv_freq, &sel, TE_HEAD_DIM, 1.0);

    const shape = [_]c_int{ @intCast(seq), TE_HEAD_DIM };
    const cos_raw = mlx.mlx_array_new_data(cos_buf.ptr, &shape, 2, .float32);
    defer free(cos_raw);
    const sin_raw = mlx.mlx_array_new_data(sin_buf.ptr, &shape, 2, .float32);
    defer free(sin_raw);
    return .{ .cos = try astype(cos_raw, dtype, s), .sin = try astype(sin_raw, dtype, s) };
}

// ── the joint encode ──

/// ti2i conditioning: hidden states [1, n, hidden] PRE-final-norm plus the
/// image-pad mask the DiT's token metadata needs (1 at pad positions,
/// post-drop, aligned with `hidden`'s rows).
pub const EditCond = struct {
    hidden: A,
    pad_mask: []i32,
    n: usize,

    pub fn deinit(self: *EditCond, allocator: std.mem.Allocator) void {
        free(self.hidden);
        allocator.free(self.pad_mask);
    }
};

/// Encode the expanded ti2i sequence: embed → splice-REPLACE the merged tower
/// features at the pad positions → 36-layer forward with DeepStack
/// scatter-ADD at layers 0/1/2 → drop the first `drop` rows from the hidden
/// AND the pad mask identically. NO final RMS norm: the DiT reads the hidden
/// before the LM's norm (fixture `edit_hidden` pins it pre-norm). The pad
/// count and the merged feature count must agree — a mismatch is a
/// preprocessing bug, and the scatter would die on the shape inside mlx.
pub fn encodeTi2i(allocator: std.mem.Allocator, s: S, te: *TextEncoder, vit: *const VisionTower, ids: []const i32, mask: []const i32, pixel_values: A, grids: []const [3]i64, drop: usize) !EditCond {
    const a = allocator;
    const seq: c_int = @intCast(ids.len);

    // Token embeddings [seq, hidden].
    const id_shape = [_]c_int{seq};
    const id_arr = mlx.mlx_array_new_data(ids.ptr, &id_shape, 1, .int32);
    defer free(id_arr);
    var taken = mlx.mlx_array_new();
    defer free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, te.embed_table, id_arr, 0, s));

    // Vision tower → merged + DeepStack features.
    const vout = try vit.forward(pixel_values, grids);
    defer free(vout.merged);
    defer for (vout.deepstack) |d| free(d);

    // Visual-token positions (host scan of the placeholder runs).
    var poslist: std.ArrayList(i32) = .empty;
    defer poslist.deinit(a);
    for (ids, 0..) |id, i| if (id == TE_IMAGE_TOKEN) try poslist.append(a, @intCast(i));
    const nvis: c_int = @intCast(poslist.items.len);
    // The placeholder run and the merged features come from two independent
    // paths (prompt templating vs the ViT grid). A mismatch is a preprocessing
    // bug, and the scatter would die on the shape inside mlx — fail honestly.
    if (nvis != mlx.getShape(vout.merged)[0]) {
        log.err("[qwen-image-edit] {d} <|image_pad|> tokens but {d} vision features\n", .{ nvis, mlx.getShape(vout.merged)[0] });
        return error.QwenEditVisionTokenMismatch;
    }
    const pos_shape = [_]c_int{nvis};
    const pos_arr = mlx.mlx_array_new_data(poslist.items.ptr, &pos_shape, 1, .int32);
    defer free(pos_arr);

    // Splice-REPLACE the placeholder embeddings with the merged vision rows.
    const merged_dt = try astype(vout.merged, te.dtype, s);
    defer free(merged_dt);
    const replaced = try mage_flow.scatterRows(taken, pos_arr, merged_dt, false, s);
    defer free(replaced);
    var x = try reshape(replaced, &[_]c_int{ 1, seq, te.hidden }, s);

    const attn_mask = try mage_flow.buildTeMask(a, mask, seq, te.dtype, s);
    defer free(attn_mask);
    const rope = try buildEditRope(a, ids, grids, te.dtype, s);
    defer {
        free(rope.cos);
        free(rope.sin);
    }

    for (&te.layers, 0..) |*layer, i| {
        const nx = try mage_flow.teLayerForward(layer, x, attn_mask, rope.cos, rope.sin, seq, s);
        free(x);
        x = nx;
        if (i < 3) { // DeepStack scatter-ADD at LM layers 0/1/2 (Qwen3-VL).
            const ds_dt = try astype(vout.deepstack[i], te.dtype, s);
            defer free(ds_dt);
            const xf = try reshape(x, &[_]c_int{ seq, te.hidden }, s);
            defer free(xf);
            const scat = try mage_flow.scatterRows(xf, pos_arr, ds_dt, true, s);
            defer free(scat);
            const nx2 = try reshape(scat, &[_]c_int{ 1, seq, te.hidden }, s);
            free(x);
            x = nx2;
        }
    }

    // NO final RMS norm: the DiT reads the hidden BEFORE the LM's norm.
    const start: usize = @min(drop, ids.len);
    const hidden = try sliceSeq(x, @intCast(start), seq, s);
    errdefer free(hidden);
    try mlx.check(mlx.mlx_array_eval(hidden));
    free(x);

    const n: usize = ids.len - start;
    const pad_mask = try a.alloc(i32, n);
    for (ids[start..], 0..) |t, i| pad_mask[i] = if (t == TE_IMAGE_TOKEN) 1 else 0;
    return .{ .hidden = hidden, .pad_mask = pad_mask, .n = n };
}

// ── loader ──

/// Parse the tower geometry from {model_dir}/text_encoder/config.json's
/// `vision_config` — never hardcoded (the tiny fixture pack and the real 8B
/// differ in every number). Named errors when the block is absent.
pub fn parseTowerVitConfig(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !mage_flow.VitConfig {
    const a = allocator;
    const path = try std.fmt.allocPrint(a, "{s}/text_encoder/config.json", .{model_dir});
    defer a.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)) catch return error.QwenImageConfigMissing;
    defer a.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    const v = parsed.value;
    const vis = if (v == .object) v.object.get("vision_config") else null;
    if (vis == null or vis.? != .object) return error.QwenEditNoVisionConfig;
    const cfg = mage_flow.VitConfig{
        .hidden = @intCast(try jsonInt(vis.?, "hidden_size")),
        .heads = @intCast(try jsonInt(vis.?, "num_heads")),
        .inter = @intCast(try jsonInt(vis.?, "intermediate_size")),
        .depth = try jsonInt(vis.?, "depth"),
        .out = @intCast(try jsonInt(vis.?, "out_hidden_size")),
        .deepstack = try jsonDeepstack(vis.?),
        .prefix = "model.visual",
    };
    // The position table is checkpoint-constant (48×48) across Qwen3-VL sizes;
    // a config that says otherwise is not this tower.
    const pos = try jsonInt(vis.?, "num_position_embeddings");
    if (pos != VIT_POS_TABLE) return error.QwenEditTowerGeometryMismatch;
    return cfg;
}

fn jsonInt(v: std.json.Value, key: []const u8) !usize {
    if (v != .object) return error.QwenEditBadVisionConfig;
    const e = v.object.get(key) orelse return error.QwenEditBadVisionConfig;
    return switch (e) {
        .integer => |i| if (i >= 0) @intCast(i) else error.QwenEditBadVisionConfig,
        else => error.QwenEditBadVisionConfig,
    };
}

fn jsonDeepstack(v: std.json.Value) ![3]usize {
    if (v != .object) return error.QwenEditBadVisionConfig;
    const e = v.object.get("deepstack_visual_indexes") orelse return error.QwenEditBadVisionConfig;
    const arr = switch (e) {
        .array => |x| x.items,
        else => return error.QwenEditBadVisionConfig,
    };
    if (arr.len != 3) return error.QwenEditBadVisionConfig;
    var out: [3]usize = undefined;
    for (arr, 0..) |item, i| out[i] = switch (item) {
        .integer => |iv| if (iv >= 0) @intCast(iv) else return error.QwenEditBadVisionConfig,
        else => return error.QwenEditBadVisionConfig,
    };
    return out;
}

/// Open the text_encoder weights ONCE (vision keys KEPT — `loadWeights` drops
/// them as a `--no-vision` tower) and build the LM + tower from that one map.
pub fn loadTeWithTower(io: std.Io, allocator: std.mem.Allocator, s: S, model_dir: []const u8, dtype: mlx.mlx_dtype) !struct { te: TextEncoder, vit: VisionTower } {
    const a = allocator;
    // ONE weight-map open, vision keys KEPT (the plain text loader drops the
    // tower keys as a `--no-vision` tower) — both consumers share it.
    var w = try VisionTower.openWeights(io, a, model_dir);
    defer w.deinit();
    var cfg = parseTowerVitConfig(io, a, model_dir) catch |e| switch (e) {
        error.QwenEditNoVisionConfig => {
            // No config: still tell a tower-without-its-config apart from no
            // tower at all (both spellings).
            if (w.get("model.visual.patch_embed.proj.weight") != null or
                w.get("vision_tower.patch_embed.proj.weight") != null)
                return error.QwenEditTowerConfigMissing;
            return error.QwenEditNoTower;
        },
        else => return e,
    };
    // The tower's spelling is whichever the pack shipped: diffusers/ddalcu
    // `model.visual.*`, mlx-community `vision_tower.*`. Geometry comes from
    // vision_config either way (the parse is spelling-independent).
    if (w.get("model.visual.patch_embed.proj.weight") != null) {
        cfg.prefix = "model.visual";
    } else if (w.get("vision_tower.patch_embed.proj.weight") != null) {
        cfg.prefix = "vision_tower";
    } else return error.QwenEditNoTower; // config present, tower weights absent
    var te = try TextEncoder.loadFrom(a, s, &w, dtype);
    errdefer te.deinit();
    var vit = try VisionTower.loadFrom(a, s, &w, cfg, dtype);
    errdefer vit.deinit();
    // The loaded pos table must match the parsed geometry (a config/weights
    // disagreement is a silent wrong-shape tower).
    const pe = mlx.getShape(vit.pos_embed);
    if (pe[0] != VIT_POS_TABLE or pe[1] != cfg.hidden) return error.QwenEditTowerGeometryMismatch;
    return .{ .te = te, .vit = vit };
}

// ── Tests ──

const testing = std.testing;

test "QwenImage edit ti2i prompt template" {
    const a = testing.allocator;
    const p1 = try buildTi2iPrompt(a, "make the sky red", 1);
    defer a.free(p1);
    try testing.expectEqualStrings(
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n" ++
            "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>make the sky red<|im_end|>\n" ++
            "<|im_start|>assistant\n",
        p1,
    );
    // Second and later references are separated by ONE space before <imageN>.
    const p2 = try buildTi2iPrompt(a, "x", 2);
    defer a.free(p2);
    try testing.expectEqualStrings(
        "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n" ++
            "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|> " ++
            "<image2><|vision_start|><|image_pad|><|vision_end|>x<|im_end|>\n" ++
            "<|im_start|>assistant\n",
        p2,
    );
    // An empty prompt renders " " — Qwen has no BOS to read otherwise.
    const pe = try buildTi2iPrompt(a, "", 1);
    defer a.free(pe);
    try testing.expect(std.mem.endsWith(u8, pe, "<|vision_end|> <|im_end|>\n<|im_start|>assistant\n"));
}

test "QwenImage edit expandImagePads expansion counts" {
    const a = testing.allocator;
    const ids = [_]i32{ 11, TE_IMAGE_TOKEN, 22, TE_IMAGE_TOKEN, 33 };
    const out = try expandImagePads(a, &ids, &.{ 2, 3 });
    defer a.free(out);
    try testing.expectEqualSlices(i32, &.{
        11, TE_IMAGE_TOKEN, TE_IMAGE_TOKEN, 22, TE_IMAGE_TOKEN, TE_IMAGE_TOKEN, TE_IMAGE_TOKEN, 33,
    }, out);
    // The placeholder count must equal the reference count, both ways.
    try testing.expectError(error.QwenEditPadCountMismatch, expandImagePads(a, &ids, &.{2}));
    try testing.expectError(error.QwenEditPadCountMismatch, expandImagePads(a, &ids, &.{ 2, 3, 4 }));
}

const Fixture = struct {
    dir: []const u8,
    fx: Weights,

    fn open() !Fixture {
        const dir = std.mem.span(std.c.getenv("QWEN_IMAGE_TEST_MODEL") orelse return error.SkipZigTest);
        const path = std.mem.span(std.c.getenv("QWEN_IMAGE_FIXTURE") orelse return error.SkipZigTest);
        return .{ .dir = dir, .fx = try model_mod.loadWeightsSingleFile(testing.allocator, path) };
    }
    fn get(self: *const Fixture, key: []const u8) !A {
        return self.fx.get(key) orelse error.MissingFixtureTensor;
    }
};

/// Fixture int32 tensor → owned host slice.
fn readIds(a: std.mem.Allocator, arr: A, s: S) ![]i32 {
    const c = try astype(arr, .int32, s);
    defer free(c);
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    const d = mlx.mlx_array_data_int32(c) orelse return error.NoData;
    const out = try a.alloc(i32, n);
    @memcpy(out, d[0..n]);
    return out;
}

/// Fixture f32 tensor → owned host slice.
fn readF32(a: std.mem.Allocator, arr: A, s: S) ![]f32 {
    const c = try astype(arr, .float32, s);
    defer free(c);
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    const d = mlx.mlx_array_data_float32(c) orelse return error.NoData;
    const out = try a.alloc(f32, n);
    @memcpy(out, d[0..n]);
    return out;
}

const Parity = struct { cos: f32, rms_ratio: f32 };

fn parity(got: A, want: A, s: S) !Parity {
    const g32 = try astype(got, .float32, s);
    defer free(g32);
    const g = try reshape(g32, &[_]c_int{-1}, s);
    defer free(g);
    const w = try reshape(want, &[_]c_int{-1}, s);
    defer free(w);
    const dot = struct {
        fn f(x: A, y: A, st: S) !f32 {
            const p = try mulHost(x, y, st);
            defer free(p);
            var o = mlx.mlx_array_new();
            defer free(o);
            try mlx.check(mlx.mlx_sum_axis(&o, p, 0, false, st));
            var v: f32 = 0;
            try mlx.check(mlx.mlx_array_item_float32(&v, o));
            return v;
        }
    }.f;
    const gg = try dot(g, g, s);
    const ww = try dot(w, w, s);
    return .{ .cos = (try dot(g, w, s)) / (@sqrt(gg) * @sqrt(ww)), .rms_ratio = @sqrt(gg / ww) };
}

fn mulHost(x: A, y: A, st: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, x, y, st));
    return o;
}

fn expectParity(name: []const u8, got: A, want: A, s: S) !void {
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(got));
    const p = try parity(got, want, s);
    std.debug.print("[qwen-image-edit] {s}: cos={d:.6} rms_ratio={d:.6}\n", .{ name, p.cos, p.rms_ratio });
    try testing.expect(p.cos > 0.9999);
    try testing.expectApproxEqAbs(@as(f32, 1.0), p.rms_ratio, 1e-3);
}

/// Max-abs in 8-bit pixel units: `scale` converts the tensors' own units to
/// u8 steps (255 for [0,1], 127.5 for [-1,1]); the bar is 2/255.
fn expectMaxAbs(name: []const u8, got: []const f32, want: []const f32, scale: f32) !void {
    try testing.expectEqual(got.len, want.len);
    var mx: f32 = 0;
    for (got, want) |g, w| mx = @max(mx, @abs(g - w) * scale);
    std.debug.print("[qwen-image-edit] {s}: max-abs = {d:.4}/255\n", .{ name, mx });
    try testing.expect(mx <= 2.0);
}

fn fixtureGrids(a: std.mem.Allocator, f: *const Fixture, s: S) ![][3]i64 {
    const gids = try readIds(a, try f.get("edit_grids"), s);
    defer a.free(gids);
    const nimg = gids.len / 3;
    const grids = try a.alloc([3]i64, nimg);
    for (0..nimg) |i| grids[i] = .{ gids[i * 3], gids[i * 3 + 1], gids[i * 3 + 2] };
    return grids;
}

/// The exact edit fixture's prompt and negative prompt (dump script pins).
const EDIT_PROMPT = "make the fox wear a tiny hat";
const EDIT_NEG_PROMPT = "blurry";

test "QwenImage edit tokenization (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    const tok_dir = try std.fmt.allocPrint(a, "{s}/processor", .{f.dir});
    defer a.free(tok_dir);
    var tok = try tok_mod.loadTokenizerAny(io, a, tok_dir);
    defer tok.deinit();

    const grids = try fixtureGrids(a, &f, s);
    defer a.free(grids);
    // Slots per reference: t·h·w / merge² (the merged-token count).
    const slots = try a.alloc(usize, grids.len);
    defer a.free(slots);
    for (grids, 0..) |g, i| slots[i] = @intCast(@divTrunc(g[0] * g[1] * g[2], 4));
    try testing.expectEqualSlices(usize, &.{ 1036, 1024 }, slots);

    // Positive: render → tokenize → expand == edit_input_ids EXACT.
    const pos_text = try buildTi2iPrompt(a, EDIT_PROMPT, grids.len);
    defer a.free(pos_text);
    const pos_enc = try tok.encode(a, pos_text);
    defer a.free(pos_enc);
    const pos_ids = try a.alloc(i32, pos_enc.len);
    defer a.free(pos_ids);
    for (pos_enc, 0..) |t, i| pos_ids[i] = @intCast(t);
    const expanded = try expandImagePads(a, pos_ids, slots);
    defer a.free(expanded);
    const want = try readIds(a, try f.get("edit_input_ids"), s);
    defer a.free(want);
    try testing.expectEqualSlices(i32, want, expanded);

    // Negative renders through the same template, only the text differs.
    const neg_text = try buildTi2iPrompt(a, EDIT_NEG_PROMPT, grids.len);
    defer a.free(neg_text);
    const neg_enc = try tok.encode(a, neg_text);
    defer a.free(neg_enc);
    const neg_ids = try a.alloc(i32, neg_enc.len);
    defer a.free(neg_ids);
    for (neg_enc, 0..) |t, i| neg_ids[i] = @intCast(t);
    const neg_expanded = try expandImagePads(a, neg_ids, slots);
    defer a.free(neg_expanded);
    const neg_want = try readIds(a, try f.get("edit_neg_input_ids"), s);
    defer a.free(neg_want);
    try testing.expectEqualSlices(i32, neg_want, neg_expanded);

    // The system-prefix token count IS the conditioning drop.
    const prefix_enc = try tok.encode(a, TI2I_SYSTEM_PREFIX);
    defer a.free(prefix_enc);
    const drop = try readIds(a, try f.get("edit_drop_idx"), s);
    defer a.free(drop);
    try testing.expectEqual(@as(usize, @intCast(drop[0])), prefix_enc.len);
}

test "QwenImage edit preprocessing (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();

    // Target dims off the fixture tensor shapes: image 1 resized to the
    // edit_resized_rgb dims; image 2 to its (64,64) grid's pixels.
    const want_rgb = try readF32(a, try f.get("edit_resized_rgb"), s);
    defer a.free(want_rgb);
    const tgt_h: u32 = @intCast(mlx.getShape(try f.get("edit_resized_rgb"))[0]);
    const tgt_w: u32 = @intCast(mlx.getShape(try f.get("edit_resized_rgb"))[1]);
    const grids = try fixtureGrids(a, &f, s);
    defer a.free(grids);

    // RGBA fixture values are u8-valued floats.
    const rgba1 = try readF32(a, try f.get("edit_source_rgba"), s);
    defer a.free(rgba1);
    const src1_h: u32 = @intCast(mlx.getShape(try f.get("edit_source_rgba"))[0]);
    const src1_w: u32 = @intCast(mlx.getShape(try f.get("edit_source_rgba"))[1]);
    const rgba1_u8 = try a.alloc(u8, rgba1.len);
    defer a.free(rgba1_u8);
    for (rgba1, 0..) |v, i| rgba1_u8[i] = @intFromFloat(v);
    const rgba2 = try readF32(a, try f.get("edit_source_rgba_2"), s);
    defer a.free(rgba2);
    const rgba2_u8 = try a.alloc(u8, rgba2.len);
    defer a.free(rgba2_u8);
    for (rgba2, 0..) |v, i| rgba2_u8[i] = @intFromFloat(v);
    const src2_h: u32 = @intCast(mlx.getShape(try f.get("edit_source_rgba_2"))[0]);
    const src2_w: u32 = @intCast(mlx.getShape(try f.get("edit_source_rgba_2"))[1]);
    const tgt2: u32 = @intCast(grids[1][1] * VIT_PATCH);

    var e1 = try prepareEditRgba(a, s, rgba1_u8, src1_w, src1_h, tgt_w, tgt_h);
    defer e1.deinit();
    try testing.expectEqualSlices(c_int, &.{ 1, 3, @intCast(tgt_h), @intCast(tgt_w) }, mlx.getShape(e1.rgb));
    try testing.expectEqualSlices(c_int, &.{ 1, 4, @intCast(tgt_h), @intCast(tgt_w) }, mlx.getShape(e1.vae_in));

    // rgb [1,3,H,W] vs the HWC fixture, in u8 units.
    const rgb1 = try readF32(a, e1.rgb, s);
    defer a.free(rgb1);
    const rgb_hwc = try a.alloc(f32, rgb1.len);
    defer a.free(rgb_hwc);
    // mine is [0,1]; the fixture stores composited u8 — compare in u8 units.
    for (0..rgb1.len / 3) |i| {
        for (0..3) |c| rgb_hwc[i * 3 + c] = rgb1[c * (rgb1.len / 3) + i] * 255.0;
    }
    try expectMaxAbs("resized rgb", rgb_hwc, want_rgb, 1.0);
    // vae_in [-1,1] vs the fixture's normalized resize.
    const vae_want = try readF32(a, try f.get("edit_vae_input"), s);
    defer a.free(vae_want);
    const vae1 = try readF32(a, e1.vae_in, s);
    defer a.free(vae1);
    try expectMaxAbs("vae input", vae1, vae_want, 127.5);

    // pixel_values per image, in the fixture's row order.
    const pv_want = try readF32(a, try f.get("edit_pixel_values"), s);
    defer a.free(pv_want);
    const pv1 = try vlmPixelValues(a, s, e1.rgb, @intCast(grids[0][1]), @intCast(grids[0][2]));
    defer free(pv1);
    try testing.expectEqualSlices(c_int, &.{ @intCast(grids[0][1] * grids[0][2]), VIT_PATCH_IN }, mlx.getShape(pv1));
    const pv1_host = try readF32(a, pv1, s);
    defer a.free(pv1_host);
    const n1: usize = pv1_host.len;
    try expectMaxAbs("pixel values 1", pv1_host, pv_want[0..n1], 127.5);

    var e2 = try prepareEditRgba(a, s, rgba2_u8, src2_w, src2_h, tgt2, tgt2);
    defer e2.deinit();
    const pv2 = try vlmPixelValues(a, s, e2.rgb, @intCast(grids[1][1]), @intCast(grids[1][2]));
    defer free(pv2);
    const pv2_host = try readF32(a, pv2, s);
    defer a.free(pv2_host);
    try expectMaxAbs("pixel values 2", pv2_host, pv_want[n1..], 127.5);

    // The ENCODED path decodes to the same bytes: PNG round-trip → identical
    // outputs to the raw-RGBA path.
    const png_mod = @import("png.zig");
    const enc1 = try png_mod.encodeRgba(a, rgba1_u8, src1_w, src1_h);
    defer a.free(enc1);
    var e1d = try prepareEditImage(a, s, enc1, tgt_w, tgt_h);
    defer e1d.deinit();
    const rgb1d = try readF32(a, e1d.rgb, s);
    defer a.free(rgb1d);
    try testing.expectEqualSlices(f32, rgb1, rgb1d);
}

fn slotsRow(grids: []const [3]i64, i: usize) usize {
    return @intCast(@divTrunc(grids[i][0] * grids[i][1] * grids[i][2], 4));
}

test "QwenImage edit tower bisect (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var loaded = try loadTeWithTower(io, a, s, f.dir, .float32);
    defer loaded.te.deinit();
    defer loaded.vit.deinit();
    const grids = try fixtureGrids(a, &f, s);
    defer a.free(grids);

    const pv = try f.get("edit_pixel_values");
    const out = try loaded.vit.forward(pv, grids);
    defer free(out.merged);
    defer for (out.deepstack) |d| free(d);
    try expectParity("tower merged", out.merged, try f.get("edit_vit_merged"), s);
    const names = [_][]const u8{ "tower deepstack 0", "tower deepstack 1", "tower deepstack 2" };
    for (out.deepstack, names, 0..) |d, name, i| {
        const key = try std.fmt.allocPrint(a, "edit_vit_deepstack_{d}", .{i});
        defer a.free(key);
        try expectParity(name, d, try f.get(key), s);
    }

    // pack-notower: same pack minus the tower — the loader refuses by name.
    const notower = try std.fmt.allocPrint(a, "{s}-notower", .{f.dir});
    defer a.free(notower);
    if (std.Io.Dir.openDirAbsolute(io, notower, .{})) |*d| {
        d.close(io);
        try testing.expectError(error.QwenEditNoTower, loadTeWithTower(io, a, s, notower, .float32));
    } else |_| {}
}

test "QwenImage edit loader takes the mlx-community pack spelling (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    // pack-mcspell: `vision_tower.*` tower + `language_model.model.*` LM keys,
    // embed + pos_embed tables 4-bit gs64 — mlx-community's pack shape.
    const mc = try std.fmt.allocPrint(a, "{s}-mcspell", .{f.dir});
    defer a.free(mc);
    if (std.Io.Dir.openDirAbsolute(io, mc, .{})) |*d| {
        d.close(io);
    } else |_| return; // variant not dumped on this box
    var loaded = try loadTeWithTower(io, a, s, mc, .float32);
    defer loaded.te.deinit();
    defer loaded.vit.deinit();
    // The pos-table shape check inside loadTeWithTower already ran; the
    // dequantized table carries its DENSE logical shape.
    const pe = mlx.getShape(loaded.vit.pos_embed);
    try testing.expectEqual(@as(c_int, @intCast(VIT_POS_TABLE)), pe[0]);
    try testing.expectEqual(@as(c_int, 128), pe[1]); // vision_config hidden_size
    // The quantized embed dequantized; hidden derived from scales*64.
    try testing.expectEqual(@as(c_int, 64), loaded.te.hidden);
}

test "QwenImage edit VLM encode (env-gated)" {
    var f = try Fixture.open();
    defer f.fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const s = mlx.mlx_default_gpu_stream_new();

    var loaded = try loadTeWithTower(io, a, s, f.dir, .float32);
    defer loaded.te.deinit();
    defer loaded.vit.deinit();
    const grids = try fixtureGrids(a, &f, s);
    defer a.free(grids);
    const pv = try f.get("edit_pixel_values");

    const drop_ids = try readIds(a, try f.get("edit_drop_idx"), s);
    defer a.free(drop_ids);
    const drop: usize = @intCast(drop_ids[0]);

    const ids = try readIds(a, try f.get("edit_input_ids"), s);
    defer a.free(ids);
    const mask = try readIds(a, try f.get("edit_attention_mask"), s);
    defer a.free(mask);
    var cond = try encodeTi2i(a, s, &loaded.te, &loaded.vit, ids, mask, pv, grids, drop);
    defer cond.deinit(a);
    try testing.expectEqual(ids.len - drop, cond.n);
    try expectParity("edit hidden", cond.hidden, try f.get("edit_hidden"), s);
    const pad_want = try readIds(a, try f.get("edit_pad_mask"), s);
    defer a.free(pad_want);
    try testing.expectEqualSlices(i32, pad_want, cond.pad_mask);

    // Negative: same images, shorter instruction, same drop.
    const neg_ids = try readIds(a, try f.get("edit_neg_input_ids"), s);
    defer a.free(neg_ids);
    const neg_mask = try readIds(a, try f.get("edit_neg_attention_mask"), s);
    defer a.free(neg_mask);
    var neg_cond = try encodeTi2i(a, s, &loaded.te, &loaded.vit, neg_ids, neg_mask, pv, grids, drop);
    defer neg_cond.deinit(a);
    try testing.expectEqual(neg_ids.len - drop, neg_cond.n);
    try expectParity("edit neg hidden", neg_cond.hidden, try f.get("edit_neg_hidden"), s);
}
