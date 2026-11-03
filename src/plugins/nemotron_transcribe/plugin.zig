//! Nemotron 3.5 Transcribe (`nemo_speechlm`): the nemotron_h plugin's hybrid trunk (the checkpoint's `llm_config`, weights
//! under `llm.`) with the perception module's rows spliced at the audio placeholders. The trunk's table is nemotron_h's
//! as it is; this plugin adds the tower, the prompt's row run and the load around it.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const reader = @import("../../sdk/bases/reader.zig");
const quant_abi = @import("../../abi/quant.zig");
const json = @import("../../sdk/config/json.zig");
const nh = @import("../nemotron_h/plugin.zig");
pub const config = @import("config.zig");
const perception = @import("perception.zig");

pub const plugin = config.plugin;

const Verdict = abi.Verdict;
const gpa = std.heap.c_allocator;

pub const table = blk: {
    var t = nh.table;
    t.name = abi.Str.of(plugin.name);
    t.claims = config.claims;
    t.parse = &config.parse;
    t.free_parsed = &config.freeParsed;
    t.load = &load;
    t.unload = &unload;
    t.forward = &forward;
    t.spec = &spec_hooks;
    t.media = &.{ .tower = &tower, .prepare = &prepare, .word_times = &wordTimes, .diarize = &diarize };
    break :blk t;
};

// The trunk's MTP head (`nemotron_h`'s, read from `llm.mtp.*` in this pack) and its table for drafters, on the trunk
// the wrapper holds: each hook gets the wrapper's pointer, never nemotron_h's.
const trunk_spec = nh.table.spec.?;
const trunk_head = trunk_spec.head.?;

const head_hooks: abi.HeadHooks = blk: {
    var h = trunk_head.*;
    h.load = &headLoad;
    h.activate = &headActivate;
    h.release = &headRelease;
    h.forward = &headForward;
    h.truncate = &headTruncate;
    h.adopt = &headAdopt;
    h.mark = &headMark;
    h.draft_select = &headSelect;
    break :blk h;
};

const spec_hooks: abi.SpecHooks = .{ .head = &head_hooks, .trunk_weight = &trunkWeight };

fn trunkOf(mp: *anyopaque) *anyopaque {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    return m.trunk;
}

fn trunkWeight(mp: *anyopaque, which: abi.TrunkWeight, out: *quant_abi.Weight) callconv(.c) Verdict {
    return trunk_spec.trunk_weight.?(trunkOf(mp), which, out);
}
fn headLoad(mp: *anyopaque, ctx: *const abi.LoadCtx) callconv(.c) Verdict {
    return trunk_head.load(trunkOf(mp), ctx);
}
fn headActivate(mp: *anyopaque, slot: u32) callconv(.c) Verdict {
    return trunk_head.activate(trunkOf(mp), slot);
}
fn headRelease(mp: *anyopaque, slot: u32) callconv(.c) void {
    trunk_head.release(trunkOf(mp), slot);
}
fn headForward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    return trunk_head.forward(trunkOf(mp), f, out);
}
fn headTruncate(mp: *anyopaque, slot: u32, len: u64) callconv(.c) void {
    trunk_head.truncate(trunkOf(mp), slot, len);
}
fn headAdopt(mp: *anyopaque, slot: u32, snap: abi.Handle) callconv(.c) Verdict {
    return trunk_head.adopt(trunkOf(mp), slot, snap);
}
fn headMark(mp: *anyopaque, slot: u32) callconv(.c) void {
    trunk_head.mark(trunkOf(mp), slot);
}
fn headSelect(mp: *anyopaque, logits: abi.Array, out: *abi.Array) callconv(.c) Verdict {
    return trunk_head.draft_select(trunkOf(mp), logits, out);
}

pub const Loaded = struct {
    trunk: *nh.Loaded,
    tower: perception.Tower,
    /// The tokenizer's `<|audio|>`: what the prompt's rows hold (`Parsed.placeholder_ids.audio`).
    audio_id: u32,
    /// The in-table id the trunk gathers in their place (`config.spliceId`) and the splice matches.
    splice_id: u32,
    owned: reader.Owned = .{},
    /// The last `word_times` / `diarize` answer, held until the next call or the unload.
    ms: []u32 = &.{},
    frames: []f32 = &.{},
};

const CONFIG_CAP = 4 << 20;

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    return loadModel(ctx) catch |e| {
        out.* = .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
        return null;
    };
}

fn loadModel(ctx: *const abi.LoadCtx) !*Loaded {
    const trunk = try nh.loadModel(ctx);
    errdefer nh.table.unload(trunk);
    // `Parsed.private` is nemotron_h's `Config`, which keeps no perception block: read the file again for it.
    const bytes = try gpa.alloc(u8, CONFIG_CAP);
    defer gpa.free(bytes);
    var why: Verdict = .ok;
    const text = ctx.dir.readFile("config.json", bytes, &why) orelse return error.ConfigUnreadable;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = try json.parseRoot(arena.allocator(), text);
    const audio = try config.parseAudio(root);

    const m = try gpa.create(Loaded);
    errdefer gpa.destroy(m);
    m.* = .{ .trunk = trunk, .tower = undefined, .audio_id = ctx.parsed.placeholder_ids.audio, .splice_id = config.spliceId(ctx.parsed.shell.vocab_size) };
    errdefer m.owned.deinit();
    m.tower = try perception.Tower.load(ctx, audio, nh.config.of(ctx.parsed).block_json, &m.owned);
    // `forward` swaps the prompt's `<|audio|>` rows for this id before the trunk reads them.
    trunk.base.placeholders.audio = m.splice_id;
    return m;
}

fn unload(mp: *anyopaque) callconv(.c) void {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    gpa.free(m.ms);
    gpa.free(m.frames);
    m.tower.deinit();
    m.owned.deinit();
    nh.table.unload(m.trunk);
    gpa.destroy(m);
}

fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    // The prompt's rows are `<|audio|>`, id `vocab_size`: one past the embedding table. The splice overwrites those
    // rows, so the gather reads a stand-in token (`splice_id`) there instead. The engine hands its ids as a device
    // array (`Forward.ids`, `tokens` empty); a direct caller hands host tokens. Device ids cannot be searched without
    // a sync, so that arm rewrites every prompt chunk (a few tiny ops) where the host arm skips an audio-free one.
    if (f.mode == .decode) return nh.table.forward(m.trunk, f, out);
    var swapped_forward = f.*;
    if (f.ids.ctx != null) {
        var g = ops.Graph.init(gpa, m.tower.s);
        defer g.deinit();
        const ids: ops.Array = .{ .ctx = f.ids.ctx };
        const hit = g.equal(ids, g.scalarLike(ids, @floatFromInt(m.audio_id)) catch return oom()) catch return oom();
        const swapped = g.where(hit, g.scalarLike(ids, @floatFromInt(m.splice_id)) catch return oom(), ids) catch return oom();
        swapped_forward.ids = .{ .ctx = swapped.ctx };
        return nh.table.forward(m.trunk, &swapped_forward, out);
    }
    const ids = f.tokens.slice();
    if (std.mem.indexOfScalar(u32, ids, m.audio_id) == null) return nh.table.forward(m.trunk, f, out);
    const swapped = gpa.alloc(u32, ids.len) catch return oom();
    defer gpa.free(swapped);
    for (swapped, ids) |*s, id| s.* = if (id == m.audio_id) m.splice_id else id;
    swapped_forward.tokens = .of(swapped);
    return nh.table.forward(m.trunk, &swapped_forward, out);
}

fn oom() Verdict {
    return Verdict.refuse(.failed, "nemotron_transcribe: the prompt ids could not be rewritten");
}

fn wordTimes(mp: *anyopaque, pcm: abi.Span(f32), units: abi.Span(abi.Str), activity: abi.Span(f32), speakers: u32, out: *abi.Span(u32)) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    const words = gpa.alloc([]const u8, units.len) catch return oom();
    defer gpa.free(words);
    for (units.slice(), words) |u, *w| w.* = u.slice();
    const ms = m.tower.wordTimes(pcm.slice(), words, activity.slice(), speakers) catch |e| return Verdict.named(.failed, @errorName(e));
    gpa.free(m.ms);
    m.ms = ms;
    out.* = .of(ms);
    return .ok;
}

fn diarize(mp: *anyopaque, pcm: abi.Span(f32), speakers: *u32, out: *abi.Span(f32)) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    const frames = m.tower.speakerFrames(pcm.slice()) catch |e| return Verdict.named(.failed, @errorName(e));
    gpa.free(m.frames);
    m.frames = frames;
    speakers.* = m.tower.cfg.speakers;
    out.* = .of(frames);
    return .ok;
}

/// The clip's samples as the tower reads them (`pixels`, f32 mono 16 kHz); `rows` its encoder rows.
fn prepare(_: *const abi.Parsed, in: *const abi.MediaInput, _: u64, out: *abi.MediaItem) callconv(.c) Verdict {
    if (in.kind != .audio) return Verdict.refuse(.unsupported, "nemotron_transcribe: audio only");
    if (in.sample_rate != config.SAMPLE_RATE) return Verdict.refuse(.failed, "nemotron_transcribe: speech is 16 kHz mono samples");
    const n = in.samples.len;
    const rows = perception.rows(n);
    if (rows == 0) return Verdict.refuse(.failed, "nemotron_transcribe: the clip is shorter than one 10 ms frame");
    for (in.samples.slice()) |x| if (!std.math.isFinite(x)) return Verdict.refuse(.failed, "nemotron_transcribe: the clip holds non-finite samples");
    out.* = .{
        .kind = .audio,
        .rows = @intCast(rows),
        .pixels = .{ .ctx = mlx.mlx_array_new_data(in.samples.slice().ptr, &[_]c_int{@intCast(n)}, 1, .float32).ctx },
    };
    return .ok;
}

fn tower(mp: *anyopaque, item: *const abi.MediaItem, out: *abi.Array) callconv(.c) Verdict {
    const m: *Loaded = @ptrCast(@alignCast(mp));
    if (item.kind != .audio) return Verdict.refuse(.unsupported, "nemotron_transcribe: audio only");
    const pixels: mlx.mlx_array = .{ .ctx = item.pixels.ctx };
    mlx.check(mlx.mlx_array_eval(pixels)) catch return Verdict.refuse(.failed, "nemotron_transcribe: the clip could not be read");
    const data = mlx.mlx_array_data_float32(pixels) orelse return Verdict.refuse(.failed, "nemotron_transcribe: the clip could not be read");
    const rows = m.tower.forward(data[0..mlx.mlx_array_size(pixels)]) catch return Verdict.refuse(.failed, "nemotron_transcribe: the perception module failed");
    out.* = .{ .ctx = rows.ctx };
    return .ok;
}

/// The tiny CTC head's non-blank classes.
pub const CTC_CLASSES = 12;

/// The tiny pack the conformance suite runs: nemotron_h's tiny trunk under `llm.`, a one-layer encoder per expert.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const d = 32; // the ASR expert's width
    const sd = 16; // the speaker expert's width
    const mels = 16;
    const tf = 8;
    var t = sdk.tensors.Tensors.init(a, 5);
    try nh.trunkTensors(&t, "llm.backbone", "llm.lm_head");
    // A mel filterbank is non-negative: a negative band power would make its log NaN.
    try t.fill("perception.preprocessor.fb", .{}, &.{ 1, mels, 17 }, .F32, 0.05);
    try t.fill("perception.preprocessor.window", .{}, &.{24}, .F32, 0.5);
    try t.rand("perception.proj.weight", .{}, &.{ 64, d }, .F32, 0.1);
    try t.rand("perception.proj.bias", .{}, &.{64}, .F32, 0.1);
    try t.fill("perception.encoder.asr_norm.weight", .{}, &.{d}, .F32, 1);
    try t.fill("perception.encoder.asr_norm.bias", .{}, &.{d}, .F32, 0);
    try t.fill("perception.encoder.diar_norm.weight", .{}, &.{8}, .F32, 1);
    try t.fill("perception.encoder.diar_norm.bias", .{}, &.{8}, .F32, 0);
    try encoderTensors(&t, "perception.encoder.asr_encoder", d, 2 * d, mels * 8, true);
    try encoderTensors(&t, "perception.encoder.diarization_model.encoder", sd, 2 * sd, mels * 8, false);
    try encoderTensors(&t, "ctc_timestamp.decoder.transformer", d, d / 2, 0, true);
    try t.rand("ctc_timestamp.decoder.upsample.weight", .{}, &.{ d, d, 2 }, .F32, 0.2);
    try t.rand("ctc_timestamp.decoder.upsample.bias", .{}, &.{d}, .F32, 0.1);
    try t.rand("ctc_timestamp.decoder.decoder_layers.0.weight", .{}, &.{ CTC_CLASSES + 1, d, 1 }, .F32, 0.3);
    try t.rand("ctc_timestamp.decoder.decoder_layers.0.bias", .{}, &.{CTC_CLASSES + 1}, .F32, 0.1);
    const sm = "perception.encoder.diarization_model.sortformer_modules";
    try t.rand(sm ++ ".encoder_proj.weight", .{}, &.{ tf, sd }, .F32, 0.2);
    try t.rand(sm ++ ".encoder_proj.bias", .{}, &.{tf}, .F32, 0.1);
    try t.rand(sm ++ ".subpixel_upsample.weight", .{}, &.{ tf * 8, 3, tf }, .F32, 0.2);
    try t.rand(sm ++ ".subpixel_upsample.bias", .{}, &.{tf * 8}, .F32, 0.1);
    try t.rand(sm ++ ".first_hidden_to_hidden.weight", .{}, &.{ tf, tf }, .F32, 0.3);
    try t.rand(sm ++ ".first_hidden_to_hidden.bias", .{}, &.{tf}, .F32, 0.1);
    try t.rand(sm ++ ".single_hidden_to_spks.weight", .{}, &.{ 8, tf }, .F32, 0.3);
    try t.rand(sm ++ ".single_hidden_to_spks.bias", .{}, &.{8}, .F32, 0.1);
    try t.rand(sm ++ ".learnable_sil_emb", .{}, &.{sd}, .F32, 0.1);
    const cfg = try std.fmt.allocPrint(a,
        \\{{"model_type":"nemo_speechlm","audio_token_id":64,"eos_token_id":[2],"llm_config":{s},
        \\ "ctc_timestamps":{{"decoder_config":{{"feat_in":{d},"num_classes":{d},"use_transformer":true,"n_heads":2,"n_layers":1,
        \\  "qk_norm":true,"ff_expansion":0.5,"rotary_fraction":0.5,"residual":true,"residual_scale":0.1,"upsampling_factor":2,
        \\  "upsampling_mode":"conv_transpose"}}}},
        \\ "perception":{{"output_dim":64,"preprocessor":{{"sample_rate":16000,"normalize":null,"window_size":0.0015,
        \\  "window_stride":0.01,"window":"hann","features":16,"n_fft":32,"log":true,"frame_splicing":1}}}},
        \\ "pe_encoder_config":{{"asr_normalize_type":"per_feature","speaker_feature_config_version":1,
        \\  "speaker_feature_mode":"thresholded","speaker_activity_threshold":0.5,"online_inference_length":6,
        \\  "chunk_left_context":2,"chunk_right_context":2,"diar_fifo_len":3,"diar_spkcache_len":20,"diar_spkcache_update_period":4,
        \\  "asr_encoder_cfg":{{"d_model":{d},"n_heads":2,"n_layers":1,"qk_norm":true,"ff_expansion":2,"subsampling_factor":8,
        \\   "rotary_fraction":0.5}},
        \\  "diarization_model_cfg":{{"high_resolution":true,"sortformer_modules":{{"num_spks":8}},
        \\   "encoder":{{"d_model":{d},"n_heads":2,"n_layers":1,"ff_expansion":2,"subsampling_factor":8}}}}}}}}
    , .{ nh.fixture_trunk_json, d, CTC_CLASSES, d, sd });
    return .{ .files = try a.dupe(sdk.fixture.File, &.{
        .{ .name = "config.json", .bytes = cfg },
        .{ .name = "model.safetensors", .bytes = try t.bytes() },
    }) };
}

/// One tiny feature-stacking encoder under `base`.
fn encoderTensors(t: *sdk.tensors.Tensors, comptime base: []const u8, d: u64, ff: u64, stack_in: u64, qk_norm: bool) !void {
    const hd = d / 2;
    if (stack_in > 0) try t.rand(base ++ ".pre_encode.proj.weight", .{}, &.{ d, stack_in }, .F32, 0.05);
    try t.fill(base ++ ".embed_norm.weight", .{}, &.{d}, .F32, 1);
    try t.fill(base ++ ".embed_norm.bias", .{}, &.{d}, .F32, 0);
    try t.fill(base ++ ".final_norm.weight", .{}, &.{d}, .F32, 1);
    try t.fill(base ++ ".final_norm.bias", .{}, &.{d}, .F32, 0);
    try t.fill(base ++ ".layers.0.norm1.weight", .{}, &.{d}, .F32, 1);
    try t.fill(base ++ ".layers.0.norm1.bias", .{}, &.{d}, .F32, 0);
    try t.fill(base ++ ".layers.0.norm2.weight", .{}, &.{d}, .F32, 1);
    try t.fill(base ++ ".layers.0.norm2.bias", .{}, &.{d}, .F32, 0);
    try t.rand(base ++ ".layers.0.attn.w_qkv.weight", .{}, &.{ 3 * d, d }, .F32, 0.1);
    try t.rand(base ++ ".layers.0.attn.out_proj.weight", .{}, &.{ d, d }, .F32, 0.1);
    try t.rand(base ++ ".layers.0.attn.out_proj.bias", .{}, &.{d}, .F32, 0.05);
    if (qk_norm) {
        try t.fill(base ++ ".layers.0.attn.q_norm.weight", .{}, &.{hd}, .F32, 1);
        try t.fill(base ++ ".layers.0.attn.q_norm.bias", .{}, &.{hd}, .F32, 0);
        try t.fill(base ++ ".layers.0.attn.k_norm.weight", .{}, &.{hd}, .F32, 1);
        try t.fill(base ++ ".layers.0.attn.k_norm.bias", .{}, &.{hd}, .F32, 0);
    }
    try t.rand(base ++ ".layers.0.ffn.linear1.weight", .{}, &.{ ff, d }, .F32, 0.1);
    try t.rand(base ++ ".layers.0.ffn.linear1.bias", .{}, &.{ff}, .F32, 0.05);
    try t.rand(base ++ ".layers.0.ffn.linear2.weight", .{}, &.{ d, ff }, .F32, 0.1);
    try t.rand(base ++ ".layers.0.ffn.linear2.bias", .{}, &.{d}, .F32, 0.05);
}

// ── tests ──

test {
    _ = config;
    _ = perception;
    _ = @import("ctc_align.zig");
}
