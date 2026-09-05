//! Load a SINGLE-FILE SDXL checkpoint (the Civitai / A1111 / SGM distribution)
//! by converting its LDM key naming into the diffusers layout the SDXL engine
//! already binds.
//!
//! `sdxl_pipeline.Engine.load` reads a diffusers multi-folder repo: `unet/`,
//! `vae/`, `text_encoder/`, `text_encoder_2/`, each with its own `config.json`
//! and diffusers-named weights. A single-file checkpoint has none of that — one
//! `.safetensors` blob, LDM key naming (`model.diffusion_model.*`,
//! `first_stage_model.*`, `conditioner.embedders.{0,1}.*`), no configs, no
//! tokenizer files. This module bridges the two: it loads the blob into a
//! `Weights` map, then produces a SECOND `Weights` map whose keys are exactly
//! what the four component loaders request, so `Engine.loadSingleFile` can reuse
//! the same `*FromWeights` binding paths the folder loader uses.
//!
//! The conversion is a faithful port of diffusers' `convert_ldm_unet_checkpoint`
//! / `convert_ldm_vae_checkpoint` / `convert_open_clip_checkpoint`. Three shapes
//! are more than a rename:
//!
//!   - OpenCLIP bigG packs Q/K/V into ONE `in_proj_weight` `[3d, d]`; diffusers
//!     wants three `[d, d]` linears. `attn.in_proj_*` SPLITS 3-ways on axis 0.
//!   - OpenCLIP `text_projection` is `[hidden, proj]` applied as `pooled @ P`;
//!     HF `CLIPTextModelWithProjection` stores `[proj, hidden]` and applies
//!     `pooled @ W.T`. So `W = P.T` — the one TRANSPOSE in the CLIP path. The
//!     tower loader transposes it back at bind, so `tower.text_projection == P`.
//!   - LDM VAE attention stores Q/K/V/proj_out as 1x1 CONVS `[c, c, 1, 1]`;
//!     diffusers wants `[c, c]` linears, so those SQUEEZE. (SDXL's UNet
//!     `proj_in`/`proj_out` are already linear — `use_linear_projection` — but
//!     the squeeze is rank-gated so a conv-style checkpoint still loads.)
//!
//! ORACLE: the SDXL subsystem has no executed-diffusers reference (see
//! `sdxl.zig`). The conversion's ground truth is instead the FOLDER loader: for
//! any model shipped both ways, this converter's output must equal
//! `model.loadWeights(<repo>/<component>)` tensor-for-tensor. That parity test
//! lives at the bottom, env-gated on a real checkpoint pair.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const nn = @import("sdxl_nn.zig");
const sdxl = @import("sdxl.zig");
const log = @import("log.zig");

const Weights = model_mod.Weights;
const S = mlx.mlx_stream;

/// The standard OpenAI CLIP BPE (openai/clip-vit-large-patch14), embedded so a
/// single-file checkpoint — which ships no tokenizer — is self-contained. Both
/// SDXL towers use byte-identical vocab and merges; they differ only in the pad
/// token, so one copy serves both with a per-tower `pad_id`.
pub const CLIP_VOCAB_JSON = @embedFile("assets/clip_tokenizer/vocab.json");
pub const CLIP_MERGES_TXT = @embedFile("assets/clip_tokenizer/merges.txt");

/// CLIP-L pads with `<|endoftext|>` (49407); bigG pads with `!` (0). See
/// `sdxl_tokenizer.zig`'s header — padding both the same way is silently wrong.
pub const CLIP_L_PAD_ID: u32 = 49407;
pub const CLIP_BIGG_PAD_ID: u32 = 0;

// SDXL UNet geometry, fixed. layers_per_block=2, so each down/up STAGE spans
// (layers_per_block + 1) flat LDM blocks — two resnets plus the sampler slot.
const LAYERS_PER_BLOCK: usize = 2;
const BLOCKS_PER_STAGE: usize = LAYERS_PER_BLOCK + 1; // 3

// ════════════════════════════════════════════════════════════════════════
// Detection
// ════════════════════════════════════════════════════════════════════════

fn hasPrefix(w: *const Weights, prefix: []const u8) bool {
    var it = w.map.keyIterator();
    while (it.next()) |k| if (std.mem.startsWith(u8, k.*, prefix)) return true;
    return false;
}

/// True when a single-file `Weights` map looks like an LDM **SDXL** checkpoint.
///
/// Three markers together, none sufficient alone: the LDM UNet trunk
/// (`model.diffusion_model.input_blocks`), the SDXL-only micro-conditioning
/// embedding (`label_emb` — SD 1.5 has no `add_embedding`), and the SECOND text
/// encoder (`conditioner.embedders.1` — bigG, which is what makes it XL rather
/// than 1.5). A repo missing any one cannot be served by the XL engine.
pub fn isLdmSdxl(w: *const Weights) bool {
    return w.get("model.diffusion_model.input_blocks.0.0.weight") != null and
        w.get("model.diffusion_model.label_emb.0.0.weight") != null and
        hasPrefix(w, "conditioner.embedders.1.model.");
}

/// What a single-file checkpoint declares about how it was TRAINED, read from
/// zero-size marker tensors rather than any config.
///
/// This is the A1111/ComfyUI convention and it is the ONLY trustworthy source
/// for these two facts. NoobAI-XL V-Pred ships `v_pred` + `ztsnr` markers in the
/// checkpoint — and its own diffusers export declares
/// `"prediction_type": "epsilon"`, which is WRONG for it. A config that lies is
/// worse than no config: read as epsilon, a v-prediction model still produces a
/// plausible image, just systematically washed out, with nothing to error on.
/// So the markers win wherever they are present.
pub const TrainingMarkers = struct {
    /// `v_pred` — v-prediction rather than epsilon.
    v_prediction: bool = false,
    /// `ztsnr` — trained with zero terminal SNR.
    zero_snr: bool = false,

    pub fn any(self: TrainingMarkers) bool {
        return self.v_prediction or self.zero_snr;
    }
};

/// Zero-size marker tensor names, in the spelling the ecosystem writes them.
pub const V_PRED_MARKER = "v_pred";
pub const ZTSNR_MARKER = "ztsnr";

/// Read the training markers from a loaded LDM `Weights` map.
pub fn markersOf(w: *const Weights) TrainingMarkers {
    return .{
        .v_prediction = w.get(V_PRED_MARKER) != null,
        .zero_snr = w.get(ZTSNR_MARKER) != null,
    };
}

/// Read the training markers from a safetensors HEADER (its JSON tensor map).
/// Substring search over the raw header, so a caller can feed a bounded prefix
/// of a multi-GB file — same discipline as `sdxl.headerDeclaresLdmSdxl`. The
/// names are quoted so a `v_pred` inside some longer key cannot false-positive.
pub fn markersFromHeader(header_bytes: []const u8) TrainingMarkers {
    return .{
        .v_prediction = std.mem.indexOf(u8, header_bytes, "\"" ++ V_PRED_MARKER ++ "\"") != null,
        .zero_snr = std.mem.indexOf(u8, header_bytes, "\"" ++ ZTSNR_MARKER ++ "\"") != null,
    };
}

// ════════════════════════════════════════════════════════════════════════
// Array transforms
// ════════════════════════════════════════════════════════════════════════

/// A freeable copy of `src` at its own dtype — the "move" case, so the target
/// map owns an array the source map can still free independently.
fn retain(src: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_astype(&o, src, mlx.mlx_array_dtype(src), s));
    return o;
}

/// Squeeze a `[c, c, 1, 1]` 1x1 conv down to a `[c, c]` linear; leave anything
/// already 2-D untouched. Rank-gated so a linear `proj_in` is a no-op copy.
fn squeezeIfRank4(src: mlx.mlx_array, s: S) !mlx.mlx_array {
    if (mlx.mlx_array_ndim(src) != 4) return retain(src, s);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_squeeze(&o, src, s));
    return o;
}

/// Transpose a 2-D matrix (axes swap). Materialized contiguous so the stored
/// buffer is row-major — a downstream raw-pointer read would otherwise see the
/// parent's strides (the `materializedOwnedCopy` gotcha).
fn transpose2d(src: mlx.mlx_array, s: S) !mlx.mlx_array {
    var t = mlx.mlx_array_new();
    const axes = [_]c_int{ 1, 0 };
    try mlx.check(mlx.mlx_transpose_axes(&t, src, &axes, 2, s));
    defer _ = mlx.mlx_array_free(t);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_contiguous(&o, t, false, s));
    return o;
}

/// The i-th of `n` equal slices of `src` along axis 0, materialized contiguous.
fn splitPart(src: mlx.mlx_array, n: c_int, i: usize, s: S) !mlx.mlx_array {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_split(&vec, src, n, 0, s));
    var part = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&part, vec, i));
    defer _ = mlx.mlx_array_free(part);
    var o = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_contiguous(&o, part, false, s));
    return o;
}

// ════════════════════════════════════════════════════════════════════════
// Sub-key mappers — return an allocated diffusers suffix, or null to drop.
// ════════════════════════════════════════════════════════════════════════

/// LDM `ResBlock` sub-key → diffusers `ResnetBlock2D` sub-key (UNet flavour).
fn mapUnetResnetSub(a: std.mem.Allocator, sub: []const u8) !?[]u8 {
    const pairs = [_][2][]const u8{
        .{ "in_layers.0", "norm1" },
        .{ "in_layers.2", "conv1" },
        .{ "emb_layers.1", "time_emb_proj" },
        .{ "out_layers.0", "norm2" },
        .{ "out_layers.3", "conv2" },
        .{ "skip_connection", "conv_shortcut" },
    };
    return remapHead(a, sub, &pairs);
}

/// LDM `ResnetBlock` sub-key → diffusers (VAE flavour). Only `nin_shortcut`
/// differs from the natural names; everything else is already diffusers-spelled.
fn mapVaeResnetSub(a: std.mem.Allocator, sub: []const u8) !?[]u8 {
    const pairs = [_][2][]const u8{
        .{ "nin_shortcut", "conv_shortcut" },
    };
    return remapHead(a, sub, &pairs);
}

/// LDM `AttnBlock` sub-key → diffusers VAE attention sub-key.
fn mapVaeAttnSub(a: std.mem.Allocator, sub: []const u8) !?[]u8 {
    const pairs = [_][2][]const u8{
        .{ "norm", "group_norm" },
        .{ "q", "to_q" },
        .{ "k", "to_k" },
        .{ "v", "to_v" },
        .{ "proj_out", "to_out.0" },
    };
    return remapHead(a, sub, &pairs);
}

/// If `sub` begins with `from` at a dot boundary (or IS `from`), return
/// `to` + the remainder. If no pair matches, return `sub` verbatim (copied).
fn remapHead(a: std.mem.Allocator, sub: []const u8, pairs: []const [2][]const u8) !?[]u8 {
    for (pairs) |p| {
        const from = p[0];
        const to = p[1];
        if (std.mem.eql(u8, sub, from)) return try a.dupe(u8, to);
        if (sub.len > from.len and sub[from.len] == '.' and std.mem.startsWith(u8, sub, from)) {
            return try std.fmt.allocPrint(a, "{s}{s}", .{ to, sub[from.len..] });
        }
    }
    return try a.dupe(u8, sub);
}

/// The integer field `s[start..]` up to the next '.', plus the index of that dot
/// (relative to `s`). `null` when there is no integer or no trailing dot.
fn intField(s: []const u8) ?struct { val: usize, dot: usize } {
    const dot = std.mem.indexOfScalar(u8, s, '.') orelse return null;
    const val = std.fmt.parseInt(usize, s[0..dot], 10) catch return null;
    return .{ .val = val, .dot = dot };
}

// ════════════════════════════════════════════════════════════════════════
// Insert helpers — put a transformed array under a diffusers key.
// ════════════════════════════════════════════════════════════════════════

const Xform = enum { move, squeeze, transpose };

fn insert(dst: *Weights, key: []const u8, src: mlx.mlx_array, xf: Xform, s: S) !void {
    const arr = switch (xf) {
        .move => try retain(src, s),
        .squeeze => try squeezeIfRank4(src, s),
        .transpose => try transpose2d(src, s),
    };
    errdefer _ = mlx.mlx_array_free(arr);
    try dst.put(key, arr);
}

/// True for the two attention projections diffusers stores as linears but a
/// conv-style checkpoint stores as 1x1 convs — squeeze-on-insert, rank-gated.
fn attnProjIsSqueezable(sub: []const u8) bool {
    return std.mem.eql(u8, sub, "proj_in.weight") or std.mem.eql(u8, sub, "proj_out.weight");
}

// ════════════════════════════════════════════════════════════════════════
// UNet:  model.diffusion_model.*  →  diffusers unet keys
// ════════════════════════════════════════════════════════════════════════

fn convertUnet(a: std.mem.Allocator, dst: *Weights, rest: []const u8, src: mlx.mlx_array, s: S) !void {
    // Fixed singletons.
    const singles = [_][2][]const u8{
        .{ "input_blocks.0.0.weight", "conv_in.weight" },
        .{ "input_blocks.0.0.bias", "conv_in.bias" },
        .{ "time_embed.0.weight", "time_embedding.linear_1.weight" },
        .{ "time_embed.0.bias", "time_embedding.linear_1.bias" },
        .{ "time_embed.2.weight", "time_embedding.linear_2.weight" },
        .{ "time_embed.2.bias", "time_embedding.linear_2.bias" },
        .{ "label_emb.0.0.weight", "add_embedding.linear_1.weight" },
        .{ "label_emb.0.0.bias", "add_embedding.linear_1.bias" },
        .{ "label_emb.0.2.weight", "add_embedding.linear_2.weight" },
        .{ "label_emb.0.2.bias", "add_embedding.linear_2.bias" },
        .{ "out.0.weight", "conv_norm_out.weight" },
        .{ "out.0.bias", "conv_norm_out.bias" },
        .{ "out.2.weight", "conv_out.weight" },
        .{ "out.2.bias", "conv_out.bias" },
    };
    for (singles) |p| if (std.mem.eql(u8, rest, p[0])) return insert(dst, p[1], src, .move, s);

    if (std.mem.startsWith(u8, rest, "input_blocks.")) {
        return convertUnetIoBlock(a, dst, rest["input_blocks.".len..], src, s, .down);
    }
    if (std.mem.startsWith(u8, rest, "output_blocks.")) {
        return convertUnetIoBlock(a, dst, rest["output_blocks.".len..], src, s, .up);
    }
    if (std.mem.startsWith(u8, rest, "middle_block.")) {
        return convertUnetMid(a, dst, rest["middle_block.".len..], src, s);
    }
    // Anything else (unexpected) is dropped silently — the loader will name any
    // key it actually needs and cannot find.
}

const IoSide = enum { down, up };

fn convertUnetIoBlock(a: std.mem.Allocator, dst: *Weights, tail: []const u8, src: mlx.mlx_array, s: S, side: IoSide) !void {
    // tail = "{i}.{j}.<sub>"
    const fi = intField(tail) orelse return;
    const i = fi.val;
    const after_i = tail[fi.dot + 1 ..];
    const fj = intField(after_i) orelse return;
    const j = fj.val;
    const sub = after_i[fj.dot + 1 ..];

    switch (side) {
        .down => {
            if (i == 0) return; // conv_in handled as a singleton
            const block_id = (i - 1) / BLOCKS_PER_STAGE;
            const layer_id = (i - 1) % BLOCKS_PER_STAGE;
            if (j == 0 and std.mem.startsWith(u8, sub, "op.")) {
                // Downsampler conv.  op.weight -> conv.weight
                const key = try std.fmt.allocPrint(a, "down_blocks.{d}.downsamplers.0.conv{s}", .{ block_id, sub["op".len..] });
                defer a.free(key);
                return insert(dst, key, src, .move, s);
            }
            if (j == 0) {
                const msub = (try mapUnetResnetSub(a, sub)) orelse return;
                defer a.free(msub);
                const key = try std.fmt.allocPrint(a, "down_blocks.{d}.resnets.{d}.{s}", .{ block_id, layer_id, msub });
                defer a.free(key);
                return insert(dst, key, src, .move, s);
            }
            // j == 1: attention
            const key = try std.fmt.allocPrint(a, "down_blocks.{d}.attentions.{d}.{s}", .{ block_id, layer_id, sub });
            defer a.free(key);
            const xf: Xform = if (attnProjIsSqueezable(sub)) .squeeze else .move;
            return insert(dst, key, src, xf, s);
        },
        .up => {
            const block_id = i / BLOCKS_PER_STAGE;
            const layer_id = i % BLOCKS_PER_STAGE;
            if (j >= 1 and std.mem.startsWith(u8, sub, "conv.")) {
                // Upsampler conv — the only bare `conv.` at j>=1.
                const key = try std.fmt.allocPrint(a, "up_blocks.{d}.upsamplers.0.{s}", .{ block_id, sub });
                defer a.free(key);
                return insert(dst, key, src, .move, s);
            }
            if (j == 0) {
                const msub = (try mapUnetResnetSub(a, sub)) orelse return;
                defer a.free(msub);
                const key = try std.fmt.allocPrint(a, "up_blocks.{d}.resnets.{d}.{s}", .{ block_id, layer_id, msub });
                defer a.free(key);
                return insert(dst, key, src, .move, s);
            }
            // j == 1: attention
            const key = try std.fmt.allocPrint(a, "up_blocks.{d}.attentions.{d}.{s}", .{ block_id, layer_id, sub });
            defer a.free(key);
            const xf: Xform = if (attnProjIsSqueezable(sub)) .squeeze else .move;
            return insert(dst, key, src, xf, s);
        },
    }
}

fn convertUnetMid(a: std.mem.Allocator, dst: *Weights, tail: []const u8, src: mlx.mlx_array, s: S) !void {
    // tail = "{k}.<sub>", k in {0: resnet, 1: attention, 2: resnet}
    const fk = intField(tail) orelse return;
    const k = fk.val;
    const sub = tail[fk.dot + 1 ..];
    switch (k) {
        0, 2 => {
            const which: usize = if (k == 0) 0 else 1;
            const msub = (try mapUnetResnetSub(a, sub)) orelse return;
            defer a.free(msub);
            const key = try std.fmt.allocPrint(a, "mid_block.resnets.{d}.{s}", .{ which, msub });
            defer a.free(key);
            return insert(dst, key, src, .move, s);
        },
        1 => {
            const key = try std.fmt.allocPrint(a, "mid_block.attentions.0.{s}", .{sub});
            defer a.free(key);
            const xf: Xform = if (attnProjIsSqueezable(sub)) .squeeze else .move;
            return insert(dst, key, src, xf, s);
        },
        else => return,
    }
}

// ════════════════════════════════════════════════════════════════════════
// VAE:  first_stage_model.*  →  diffusers vae keys (DECODER + post_quant_conv)
// ════════════════════════════════════════════════════════════════════════

fn convertVae(a: std.mem.Allocator, dst: *Weights, rest: []const u8, src: mlx.mlx_array, s: S, num_up: usize) !void {
    // Decode-only: the encoder half and quant_conv never load.
    if (std.mem.startsWith(u8, rest, "encoder.")) return;
    if (std.mem.startsWith(u8, rest, "quant_conv.")) return;

    if (std.mem.startsWith(u8, rest, "post_quant_conv.")) {
        return insert(dst, rest, src, .move, s); // top-level, unchanged
    }
    if (!std.mem.startsWith(u8, rest, "decoder.")) return;
    const dtail = rest["decoder.".len..];

    // Fixed decoder singletons / renames.
    if (std.mem.startsWith(u8, dtail, "conv_in.")) {
        const key = try std.fmt.allocPrint(a, "decoder.{s}", .{dtail});
        defer a.free(key);
        return insert(dst, key, src, .move, s);
    }
    if (std.mem.startsWith(u8, dtail, "conv_out.")) {
        const key = try std.fmt.allocPrint(a, "decoder.{s}", .{dtail});
        defer a.free(key);
        return insert(dst, key, src, .move, s);
    }
    if (std.mem.startsWith(u8, dtail, "norm_out.")) {
        const key = try std.fmt.allocPrint(a, "decoder.conv_norm_out{s}", .{dtail["norm_out".len..]});
        defer a.free(key);
        return insert(dst, key, src, .move, s);
    }

    // Mid block.
    if (std.mem.startsWith(u8, dtail, "mid.block_1.")) {
        return vaeResnetInto(a, dst, "decoder.mid_block.resnets.0", dtail["mid.block_1.".len..], src, s);
    }
    if (std.mem.startsWith(u8, dtail, "mid.block_2.")) {
        return vaeResnetInto(a, dst, "decoder.mid_block.resnets.1", dtail["mid.block_2.".len..], src, s);
    }
    if (std.mem.startsWith(u8, dtail, "mid.attn_1.")) {
        const asub = (try mapVaeAttnSub(a, dtail["mid.attn_1.".len..])) orelse return;
        defer a.free(asub);
        const key = try std.fmt.allocPrint(a, "decoder.mid_block.attentions.0.{s}", .{asub});
        defer a.free(key);
        // q/k/v/proj_out are 1x1 convs in LDM; group_norm stays as-is.
        const xf: Xform = if (std.mem.startsWith(u8, asub, "to_")) .squeeze else .move;
        return insert(dst, key, src, xf, s);
    }

    // Up blocks — LDM `decoder.up.{n}` is REVERSED vs diffusers `up_blocks`.
    if (std.mem.startsWith(u8, dtail, "up.")) {
        const utail = dtail["up.".len..];
        const fn_ = intField(utail) orelse return;
        const n = fn_.val;
        const block_id = num_up - 1 - n;
        const after_n = utail[fn_.dot + 1 ..];
        if (std.mem.startsWith(u8, after_n, "block.")) {
            const btail = after_n["block.".len..];
            const fm = intField(btail) orelse return;
            const m = fm.val;
            const bsub = btail[fm.dot + 1 ..];
            const prefix = try std.fmt.allocPrint(a, "decoder.up_blocks.{d}.resnets.{d}", .{ block_id, m });
            defer a.free(prefix);
            return vaeResnetInto(a, dst, prefix, bsub, src, s);
        }
        if (std.mem.startsWith(u8, after_n, "upsample.conv.")) {
            const key = try std.fmt.allocPrint(a, "decoder.up_blocks.{d}.upsamplers.0.conv{s}", .{ block_id, after_n["upsample.conv".len..] });
            defer a.free(key);
            return insert(dst, key, src, .move, s);
        }
    }
}

fn vaeResnetInto(a: std.mem.Allocator, dst: *Weights, prefix: []const u8, sub: []const u8, src: mlx.mlx_array, s: S) !void {
    const msub = (try mapVaeResnetSub(a, sub)) orelse return;
    defer a.free(msub);
    const key = try std.fmt.allocPrint(a, "{s}.{s}", .{ prefix, msub });
    defer a.free(key);
    return insert(dst, key, src, .move, s);
}

/// Number of decoder up-stages: max `decoder.up.{n}` index + 1. SDXL's VAE has
/// four; reading it from the checkpoint keeps a differently-sized VAE loadable.
fn countVaeUpStages(w: *const Weights) usize {
    var max_n: ?usize = null;
    var it = w.map.keyIterator();
    while (it.next()) |k| {
        const key = k.*;
        const pfx = "first_stage_model.decoder.up.";
        if (!std.mem.startsWith(u8, key, pfx)) continue;
        const fn_ = intField(key[pfx.len..]) orelse continue;
        max_n = if (max_n) |mx| @max(mx, fn_.val) else fn_.val;
    }
    return if (max_n) |mx| mx + 1 else 0;
}

// ════════════════════════════════════════════════════════════════════════
// CLIP-L:  conditioner.embedders.0.transformer.*  →  diffusers text_encoder
// ════════════════════════════════════════════════════════════════════════

fn convertClipL(a: std.mem.Allocator, dst: *Weights, rest: []const u8, src: mlx.mlx_array, s: S) !void {
    _ = a;
    // After the prefix strip these are already `text_model.*` diffusers keys.
    // The one exception is the int `position_ids` buffer diffusers omits.
    if (std.mem.eql(u8, rest, "text_model.embeddings.position_ids")) return;
    if (!std.mem.startsWith(u8, rest, "text_model.")) return;
    return insert(dst, rest, src, .move, s);
}

// ════════════════════════════════════════════════════════════════════════
// bigG:  conditioner.embedders.1.model.*  →  diffusers text_encoder_2
// ════════════════════════════════════════════════════════════════════════

fn convertClipG(a: std.mem.Allocator, dst: *Weights, rest: []const u8, src: mlx.mlx_array, s: S) !void {
    // Top-level (non-`transformer.`) tensors.
    if (std.mem.eql(u8, rest, "positional_embedding")) {
        return insert(dst, "text_model.embeddings.position_embedding.weight", src, .move, s);
    }
    if (std.mem.eql(u8, rest, "token_embedding.weight")) {
        return insert(dst, "text_model.embeddings.token_embedding.weight", src, .move, s);
    }
    if (std.mem.startsWith(u8, rest, "ln_final.")) {
        const key = try std.fmt.allocPrint(a, "text_model.final_layer_norm{s}", .{rest["ln_final".len..]});
        defer a.free(key);
        return insert(dst, key, src, .move, s);
    }
    if (std.mem.eql(u8, rest, "text_projection")) {
        // OpenCLIP `pooled @ P` vs HF `pooled @ W.T`  =>  W = P.T.
        return insert(dst, "text_projection.weight", src, .transpose, s);
    }
    if (std.mem.eql(u8, rest, "logit_scale")) return; // unused by the tower

    // Per-layer: transformer.resblocks.{i}.<sub>
    if (!std.mem.startsWith(u8, rest, "transformer.resblocks.")) return;
    const ltail = rest["transformer.resblocks.".len..];
    const fi = intField(ltail) orelse return;
    const i = fi.val;
    const sub = ltail[fi.dot + 1 ..];
    const base = try std.fmt.allocPrint(a, "text_model.encoder.layers.{d}", .{i});
    defer a.free(base);

    // in_proj packs Q/K/V on axis 0 — split into three linears.
    if (std.mem.eql(u8, sub, "attn.in_proj_weight") or std.mem.eql(u8, sub, "attn.in_proj_bias")) {
        const field = if (std.mem.endsWith(u8, sub, "weight")) "weight" else "bias";
        const names = [_][]const u8{ "q_proj", "k_proj", "v_proj" };
        for (names, 0..) |nm, idx| {
            const part = try splitPart(src, 3, idx, s);
            errdefer _ = mlx.mlx_array_free(part);
            const key = try std.fmt.allocPrint(a, "{s}.self_attn.{s}.{s}", .{ base, nm, field });
            defer a.free(key);
            try dst.put(key, part);
        }
        return;
    }

    const renames = [_][2][]const u8{
        .{ "ln_1", "layer_norm1" },
        .{ "ln_2", "layer_norm2" },
        .{ "mlp.c_fc", "mlp.fc1" },
        .{ "mlp.c_proj", "mlp.fc2" },
        .{ "attn.out_proj", "self_attn.out_proj" },
    };
    for (renames) |p| {
        const from = p[0];
        if (sub.len > from.len and sub[from.len] == '.' and std.mem.startsWith(u8, sub, from)) {
            const key = try std.fmt.allocPrint(a, "{s}.{s}{s}", .{ base, p[1], sub[from.len..] });
            defer a.free(key);
            return insert(dst, key, src, .move, s);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// Top-level conversion
// ════════════════════════════════════════════════════════════════════════

/// The converted checkpoint, split by component. The two CLIP towers CANNOT
/// share a map: both encoders use byte-identical `text_model.encoder.layers.N.*`
/// key names, so a single flat map would have bigG silently overwrite CLIP-L
/// (the folder layout keeps them in separate `text_encoder/` and
/// `text_encoder_2/` dirs for exactly this reason). UNet and VAE keys never
/// collide with each other, so they share `main`.
pub const Converted = struct {
    main: Weights, // unet + vae, diffusers keys
    clip_l: Weights, // CLIP-L, `text_model.*`
    clip_g: Weights, // bigG, `text_model.*`

    pub fn deinit(self: *Converted) void {
        self.main.deinit();
        self.clip_l.deinit();
        self.clip_g.deinit();
    }
};

/// Build the diffusers-keyed component maps from an LDM single-file `Weights`.
/// The caller owns and must `deinit` the result; `src` is read, never consumed.
pub fn convert(allocator: std.mem.Allocator, src: *const Weights) !Converted {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);

    var out = Converted{
        .main = Weights.init(allocator),
        .clip_l = Weights.init(allocator),
        .clip_g = Weights.init(allocator),
    };
    errdefer out.deinit();

    const num_up = countVaeUpStages(src);

    var it = src.map.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const arr = entry.value_ptr.*;
        if (std.mem.startsWith(u8, key, "model.diffusion_model.")) {
            try convertUnet(allocator, &out.main, key["model.diffusion_model.".len..], arr, s);
        } else if (std.mem.startsWith(u8, key, "first_stage_model.")) {
            try convertVae(allocator, &out.main, key["first_stage_model.".len..], arr, s, num_up);
        } else if (std.mem.startsWith(u8, key, "conditioner.embedders.0.transformer.")) {
            try convertClipL(allocator, &out.clip_l, key["conditioner.embedders.0.transformer.".len..], arr, s);
        } else if (std.mem.startsWith(u8, key, "conditioner.embedders.1.model.")) {
            try convertClipG(allocator, &out.clip_g, key["conditioner.embedders.1.model.".len..], arr, s);
        }
    }

    log.info("[sdxl] single-file convert: {d} LDM tensors -> main {d} / clip_l {d} / clip_g {d} (vae up-stages={d})\n", .{
        src.count(), out.main.count(), out.clip_l.count(), out.clip_g.count(), num_up,
    });
    return out;
}

/// Find a single-file LDM SDXL checkpoint in `model_dir`; return its ABSOLUTE
/// path (allocated, caller frees) or null. Root-level `.safetensors` only,
/// matched on the header markers — the same question `peekSdxlSingleFile` asks
/// during discovery, answered here with the filename the loader needs.
pub fn findLdmSdxlFile(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !?[]u8 {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    const cap = 4 * 1024 * 1024;
    const buf = try allocator.alloc(u8, cap);
    defer allocator.free(buf);
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        var file = dir.openFile(io, entry.name, .{}) catch continue;
        defer file.close(io);
        var rbuf: [4096]u8 = undefined;
        var rs = file.reader(io, &rbuf);
        const n = rs.interface.readSliceShort(buf) catch continue;
        if (n <= 8) continue;
        if (sdxl.headerDeclaresLdmSdxl(buf[8..n])) {
            return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, entry.name });
        }
    }
    return null;
}

// ════════════════════════════════════════════════════════════════════════
// Tests — hermetic key-mapping invariants. See ORACLE note in the header: the
// end-to-end fidelity check is the folder-parity test, env-gated below.
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

test "remapHead swaps a dotted head and copies the tail" {
    const a = testing.allocator;
    const pairs = [_][2][]const u8{.{ "in_layers.0", "norm1" }};
    const got = (try remapHead(a, "in_layers.0.weight", &pairs)).?;
    defer a.free(got);
    try testing.expectEqualStrings("norm1.weight", got);

    // No match -> verbatim copy.
    const same = (try remapHead(a, "conv2.bias", &pairs)).?;
    defer a.free(same);
    try testing.expectEqualStrings("conv2.bias", same);
}

test "intField reads an index and locates its dot" {
    const f = intField("12.1.norm.weight").?;
    try testing.expectEqual(@as(usize, 12), f.val);
    try testing.expectEqual(@as(usize, 2), f.dot);
    try testing.expect(intField("nope.weight") == null);
}

test "unet resnet sub-keys map to diffusers names" {
    const a = testing.allocator;
    const cases = [_][2][]const u8{
        .{ "in_layers.0.weight", "norm1.weight" },
        .{ "in_layers.2.bias", "conv1.bias" },
        .{ "emb_layers.1.weight", "time_emb_proj.weight" },
        .{ "out_layers.0.weight", "norm2.weight" },
        .{ "out_layers.3.weight", "conv2.weight" },
        .{ "skip_connection.bias", "conv_shortcut.bias" },
    };
    for (cases) |c| {
        const got = (try mapUnetResnetSub(a, c[0])).?;
        defer a.free(got);
        try testing.expectEqualStrings(c[1], got);
    }
}

fn mkArr(dims: []const c_int) mlx.mlx_array {
    var total: usize = 1;
    for (dims) |d| total *= @intCast(d);
    const buf = testing.allocator.alloc(f32, total) catch unreachable;
    defer testing.allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = @floatFromInt(i); // distinct values so a split is checkable
    // mlx_array_new_data COPIES shape-worth of bytes, so the buffer can free.
    return mlx.mlx_array_new_data(buf.ptr, dims.ptr, @intCast(dims.len), .float32);
}

test "convert maps the block-index-sensitive keys to their diffusers targets" {
    const a = testing.allocator;
    var src = Weights.init(a);
    defer src.deinit();

    // (ldm key, dims) — dims only need the right RANK for the transform.
    const r4 = [_]c_int{ 2, 2, 1, 1 };
    const r2 = [_]c_int{ 2, 2 };
    try src.put("model.diffusion_model.input_blocks.0.0.weight", mkArr(&r4));
    try src.put("model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn1.to_q.weight", mkArr(&r2));
    try src.put("model.diffusion_model.input_blocks.3.0.op.weight", mkArr(&r4));
    try src.put("model.diffusion_model.input_blocks.4.0.in_layers.0.weight", mkArr(&r2));
    try src.put("model.diffusion_model.output_blocks.2.2.conv.weight", mkArr(&r4));
    try src.put("model.diffusion_model.output_blocks.0.0.out_layers.3.weight", mkArr(&r4));
    try src.put("model.diffusion_model.middle_block.1.proj_in.weight", mkArr(&r2));
    try src.put("model.diffusion_model.middle_block.2.in_layers.2.weight", mkArr(&r4));
    // VAE — up.{n} is reversed; num_up derives from the max index present.
    try src.put("first_stage_model.decoder.up.3.block.0.conv1.weight", mkArr(&r4));
    try src.put("first_stage_model.decoder.up.0.upsample.conv.weight", mkArr(&r4));
    try src.put("first_stage_model.decoder.mid.attn_1.q.weight", mkArr(&r4));
    try src.put("first_stage_model.decoder.mid.block_1.nin_shortcut.weight", mkArr(&r4));
    // CLIP-L: prefix strip only.
    try src.put("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight", mkArr(&r2));
    // bigG: in_proj splits 3-ways; text_projection transposes.
    try src.put("conditioner.embedders.1.model.transformer.resblocks.5.attn.in_proj_weight", mkArr(&[_]c_int{ 6, 2 }));
    try src.put("conditioner.embedders.1.model.text_projection", mkArr(&r2));

    var dst = try convert(a, &src);
    defer dst.deinit();

    // (map, expected diffusers key)
    const main_keys = [_][]const u8{
        "conv_in.weight",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.weight",
        "down_blocks.0.downsamplers.0.conv.weight",
        "down_blocks.1.resnets.0.norm1.weight",
        "up_blocks.0.upsamplers.0.conv.weight",
        "up_blocks.0.resnets.0.conv2.weight",
        "mid_block.attentions.0.proj_in.weight",
        "mid_block.resnets.1.conv1.weight",
        "decoder.up_blocks.0.resnets.0.conv1.weight", // up.3 -> block (4-1-3)=0
        "decoder.up_blocks.3.upsamplers.0.conv.weight", // up.0 -> block (4-1-0)=3
        "decoder.mid_block.attentions.0.to_q.weight",
        "decoder.mid_block.resnets.0.conv_shortcut.weight",
    };
    for (main_keys) |k| {
        if (dst.main.get(k) == null) {
            std.debug.print("missing main key: {s}\n", .{k});
            return error.MissingConvertedKey;
        }
    }
    // CLIP-L map: prefix-stripped text_model.* only.
    try testing.expect(dst.clip_l.get("text_model.embeddings.token_embedding.weight") != null);
    // bigG map: in_proj split 3-ways, text_projection transposed.
    const g_keys = [_][]const u8{
        "text_model.encoder.layers.5.self_attn.q_proj.weight",
        "text_model.encoder.layers.5.self_attn.k_proj.weight",
        "text_model.encoder.layers.5.self_attn.v_proj.weight",
        "text_projection.weight",
    };
    for (g_keys) |k| {
        if (dst.clip_g.get(k) == null) {
            std.debug.print("missing bigG key: {s}\n", .{k});
            return error.MissingConvertedKey;
        }
    }
    // The two towers must NOT share a map: bigG's layer-5 q_proj is absent from
    // CLIP-L's map (the collision that a single flat map would have hidden).
    try testing.expect(dst.clip_l.get("text_model.encoder.layers.5.self_attn.q_proj.weight") == null);
    // The mid attention 1x1 conv squeezed to a 2-D linear.
    try testing.expectEqual(@as(usize, 2), mlx.mlx_array_ndim(dst.main.get("decoder.mid_block.attentions.0.to_q.weight").?));
}

test "training markers are read from the checkpoint, not a config" {
    const a = testing.allocator;
    // A v-pred + ztsnr checkpoint (NoobAI-XL V-Pred's actual shape).
    var w = Weights.init(a);
    defer w.deinit();
    try w.put("model.diffusion_model.input_blocks.0.0.weight", mlx.mlx_array_new());
    try w.put(V_PRED_MARKER, mlx.mlx_array_new());
    try w.put(ZTSNR_MARKER, mlx.mlx_array_new());
    const m = markersOf(&w);
    try testing.expect(m.v_prediction);
    try testing.expect(m.zero_snr);
    try testing.expect(m.any());

    // A plain epsilon checkpoint carries neither.
    var e = Weights.init(a);
    defer e.deinit();
    try e.put("model.diffusion_model.input_blocks.0.0.weight", mlx.mlx_array_new());
    const me = markersOf(&e);
    try testing.expect(!me.v_prediction);
    try testing.expect(!me.zero_snr);
    try testing.expect(!me.any());
}

test "markersFromHeader keys on the QUOTED name so a longer key cannot match" {
    const vpred = "{\"v_pred\":{\"shape\":[0]},\"ztsnr\":{\"shape\":[0]}}";
    const m = markersFromHeader(vpred);
    try testing.expect(m.v_prediction and m.zero_snr);

    // A key that merely CONTAINS the marker name is not the marker.
    const decoy = "{\"model.diffusion_model.v_pred_thing.weight\":{}}";
    const d = markersFromHeader(decoy);
    try testing.expect(!d.v_prediction);
    try testing.expect(!d.zero_snr);
}

test "the marker tensors are not converted into the diffusers maps" {
    // They are zero-size sentinels, not weights — a converted map that carried
    // one would hand a shapeless array to a binder.
    const a = testing.allocator;
    var src = Weights.init(a);
    defer src.deinit();
    try src.put(V_PRED_MARKER, mkArr(&[_]c_int{ 1, 1 }));
    try src.put(ZTSNR_MARKER, mkArr(&[_]c_int{ 1, 1 }));
    var dst = try convert(a, &src);
    defer dst.deinit();
    for ([_]*Weights{ &dst.main, &dst.clip_l, &dst.clip_g }) |m| {
        try testing.expect(m.get(V_PRED_MARKER) == null);
        try testing.expect(m.get(ZTSNR_MARKER) == null);
    }
}

test "headerDeclaresLdmSdxl needs every marker" {
    const yes = "{\"model.diffusion_model.input_blocks.0.0.weight\":{},\"model.diffusion_model.label_emb.0.0.weight\":{},\"conditioner.embedders.1.model.ln_final.weight\":{}}";
    try testing.expect(sdxl.headerDeclaresLdmSdxl(yes));
    const no = "{\"model.diffusion_model.input_blocks.0.0.weight\":{}}"; // SD 1.5-ish
    try testing.expect(!sdxl.headerDeclaresLdmSdxl(no));
}

fn shapeEql(x: mlx.mlx_array, y: mlx.mlx_array) bool {
    const nx = mlx.mlx_array_ndim(x);
    if (nx != mlx.mlx_array_ndim(y)) return false;
    const sx = mlx.mlx_array_shape(x);
    const sy = mlx.mlx_array_shape(y);
    for (0..nx) |i| if (sx[i] != sy[i]) return false;
    return true;
}

// THE ORACLE (see the file header). Given the SAME SDXL model in BOTH forms —
// a single-file LDM checkpoint and its diffusers folder — the converter's
// output must reproduce, key-for-key and shape-for-shape, what the folder
// loader reads from disk. A diffusers conversion of the same checkpoint is the
// same weights, so this catches every mis-mapped index, the in_proj split, the
// VAE up-block reversal, and the 1x1-conv squeezes.
//
//   SDXL_SINGLE_FILE=~/models/illustrious.safetensors \
//   SDXL_DIFFUSERS_DIR=~/models/sdxl-base-1.0 \
//     zig build test -Dtest-filter="sdxl single-file folder parity"
test "sdxl single-file folder parity: converted keys match the diffusers layout" {
    const sf = std.c.getenv("SDXL_SINGLE_FILE") orelse return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("SDXL_DIFFUSERS_DIR") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    var ldm = try model_mod.loadWeightsSingleFile(a, std.mem.span(sf));
    defer ldm.deinit();
    try testing.expect(isLdmSdxl(&ldm));
    var conv = try convert(a, &ldm);
    defer conv.deinit();

    const Component = struct { sub: []const u8, map: *Weights, drop_prefix: []const []const u8 };
    const comps = [_]Component{
        .{ .sub = "unet", .map = &conv.main, .drop_prefix = &.{} },
        // The converter is decode-only: skip the folder VAE's encoder half.
        .{ .sub = "vae", .map = &conv.main, .drop_prefix = &.{ "encoder.", "quant_conv." } },
        .{ .sub = "text_encoder", .map = &conv.clip_l, .drop_prefix = &.{"text_model.embeddings.position_ids"} },
        .{ .sub = "text_encoder_2", .map = &conv.clip_g, .drop_prefix = &.{"text_model.embeddings.position_ids"} },
    };

    var mismatches: usize = 0;
    for (comps) |c| {
        const cdir = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, c.sub });
        defer a.free(cdir);
        var folder = try model_mod.loadWeights(io, a, cdir);
        defer folder.deinit();

        var checked: usize = 0;
        var it = folder.map.iterator();
        while (it.next()) |e| {
            const key = e.key_ptr.*;
            var drop = false;
            for (c.drop_prefix) |p| if (std.mem.startsWith(u8, key, p)) {
                drop = true;
            };
            if (drop) continue;
            checked += 1;
            if (c.map.get(key)) |got| {
                if (!shapeEql(got, e.value_ptr.*)) {
                    std.debug.print("[{s}] SHAPE mismatch: {s}\n", .{ c.sub, key });
                    mismatches += 1;
                }
            } else {
                std.debug.print("[{s}] MISSING in converted: {s}\n", .{ c.sub, key });
                mismatches += 1;
            }
        }
        std.debug.print("[{s}] checked {d} folder keys\n", .{ c.sub, checked });
    }
    try testing.expectEqual(@as(usize, 0), mismatches);
}

test "isLdmSdxl needs all three markers" {
    const a = testing.allocator;
    var w = Weights.init(a);
    defer w.deinit();
    // A hand-built map of empty arrays — detection reads keys only.
    const keys = [_][]const u8{
        "model.diffusion_model.input_blocks.0.0.weight",
        "model.diffusion_model.label_emb.0.0.weight",
        "conditioner.embedders.1.model.ln_final.weight",
    };
    for (keys) |k| try w.put(k, mlx.mlx_array_new());
    try testing.expect(isLdmSdxl(&w));

    var missing = Weights.init(a);
    defer missing.deinit();
    // SD 1.5 shape: UNet + CLIP-L but no label_emb, no bigG.
    try missing.put("model.diffusion_model.input_blocks.0.0.weight", mlx.mlx_array_new());
    try missing.put("conditioner.embedders.0.transformer.text_model.final_layer_norm.weight", mlx.mlx_array_new());
    try testing.expect(!isLdmSdxl(&missing));
}
