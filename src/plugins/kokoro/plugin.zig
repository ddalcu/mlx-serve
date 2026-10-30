//! Kokoro-82M on the media table: `/v1/audio/speech` with 54 blendable built-in voices and no cloning. A job is
//! the body's text, phonemized and spoken in one forward per window; the result is a 24 kHz mono WAV.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const kokoro = @import("kokoro.zig");
const mlx = @import("../../mlx/ffi.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(""), // no completeness marker (legacy had none)
    .modality = .audio,
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

/// OpenAI's `input` limit for `/v1/audio/speech`; longer text is spoken window by window up to it.
pub const max_input = 4096;

const Engine = struct {
    k: *kokoro.Engine,
    wav: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const k = kokoro.Engine.load(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch {
        out.* = Verdict.refuse(.failed, "kokoro: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        k.deinit();
        out.* = Verdict.refuse(.failed, "kokoro: out of memory");
        return null;
    };
    e.* = .{ .k = k };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.k.deinit();
    gpa.free(e.wav);
    gpa.destroy(e);
}

/// A speech request, validated. `ref_audio` is Qwen3-TTS's control: refused here, never ignored.
const Speech = struct {
    text: []const u8,
    voice: []const u8 = kokoro.DEFAULT_VOICE,
    speed: f32 = 1.0,
    seed: u64 = 0,

    fn parse(b: *const job.Body) union(enum) { ok: Speech, bad: Verdict } {
        if (b.flag("warm_only")) return .{ .bad = bad("this model does not support voice cloning; nothing to warm") };
        const text = b.str("input") orelse b.str("text") orelse return .{ .bad = bad("missing 'input'") };
        if (text.len == 0) return .{ .bad = bad("empty 'input'") };
        if ((std.unicode.utf8CountCodepoints(text) catch text.len) > max_input) return .{ .bad = bad("'input' is longer than 4096 characters") };
        if (b.str("ref_audio")) |r| if (r.len > 0) return .{ .bad = bad("this model does not support voice cloning; use 'voice' to pick or blend a built-in voice") };
        var s: Speech = .{ .text = text };
        if (b.str("voice")) |v| if (v.len > 0) {
            s.voice = v;
        };
        const speed: f32 = @floatCast(b.float("speed") orelse 1.0);
        if (!(speed > 0.0) or speed > 5.0) return .{ .bad = bad("'speed' must be in (0, 5]") };
        s.speed = speed;
        s.seed = b.uint("seed") orelse 0;
        return .{ .ok = s };
    }
};

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    const sp = switch (Speech.parse(&body)) {
        .ok => |s| s,
        .bad => |v| return v,
    };
    if (!e.k.hasVoice(sp.voice)) return req.refuseFmt(.unsupported, "unknown voice", "unknown voice '{s}'; see /v1/models for the available voices", .{sp.voice});
    progress.report(progress.ctx, .{}, 0, 0);
    if (progress.boundary(progress.ctx)) return job.client_left;
    const wav = e.k.synthesizeWav(sp.text, sp.voice, sp.speed, sp.seed) catch |err| return switch (err) {
        error.EmptyKokoroInput => bad("'input' has no speakable text"),
        else => job.failure(err, "synthesis failed"),
    };
    gpa.free(e.wav);
    e.wav = wav;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.wav.len == 0) return Verdict.refuse(.failed, "kokoro: no finished job to encode");
    out.* = .of(e.wav);
    return .ok;
}

/// A tiny random Kokoro: hidden 16, one ALBERT layer, one text-encoder layer, two decode blocks, a two-stage
/// generator (rates 2 x 2, n_fft 4, hop 1) and one voice; "hello" is the only word its lexicon knows.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    var t = sdk.tensors.Tensors.init(a, 7);
    const nt = 8; // phoneme vocab
    const h = 16; // hidden_dim
    const st = 128; // style half (the voice row is 256 wide)
    const e = 8; // ALBERT embedding
    const bh = 16; // ALBERT hidden
    const inter = 32;
    const max_dur = 4;
    const w = 24; // decoder width
    const r = 4; // asr_res channels
    const c = 16; // generator initial channels
    const nf = 6; // n_fft + 2

    try t.rand("bert.embeddings.word_embeddings.weight", .{}, &.{ nt, e }, .F32, 0.3);
    try t.rand("bert.embeddings.position_embeddings.weight", .{}, &.{ 32, e }, .F32, 0.3);
    try t.rand("bert.embeddings.token_type_embeddings.weight", .{}, &.{ 2, e }, .F32, 0.3);
    try norm(&t, "bert.embeddings.LayerNorm", e, "weight", "bias");
    try lin(&t, "bert.encoder.embedding_hidden_mapping_in", bh, e);
    const layer = "bert.encoder.albert_layer_groups.0.albert_layers.0";
    inline for (.{ "query", "key", "value", "dense" }) |p| try lin(&t, layer ++ ".attention." ++ p, bh, bh);
    try norm(&t, layer ++ ".attention.LayerNorm", bh, "weight", "bias");
    try lin(&t, layer ++ ".ffn", inter, bh);
    try lin(&t, layer ++ ".ffn_output", bh, inter);
    try norm(&t, layer ++ ".full_layer_layer_norm", bh, "weight", "bias");
    try lin(&t, "bert_encoder", h, bh);

    try t.rand("text_encoder.embedding.weight", .{}, &.{ nt, h }, .F32, 0.3);
    try conv(&t, "text_encoder.cnn.0.0", h, h, 3);
    try norm(&t, "text_encoder.cnn.0.1", h, "gamma", "beta");
    try lstm(&t, "text_encoder.lstm", h, h / 2);
    try lstm(&t, "predictor.text_encoder.lstms.0", h + st, h / 2);
    try lin(&t, "predictor.text_encoder.lstms.1.fc", 2 * h, st);
    try lstm(&t, "predictor.lstm", h + st, h / 2);
    try lin(&t, "predictor.duration_proj.linear_layer", max_dur, h);
    try lstm(&t, "predictor.shared", h + st, h / 2);
    inline for (.{ "F0", "N" }) |p| {
        try resBlk(&t, "predictor." ++ p ++ ".0", h, h, false);
        try resBlk(&t, "predictor." ++ p ++ ".1", h, h / 2, true);
        try resBlk(&t, "predictor." ++ p ++ ".2", h / 2, h / 2, false);
        try conv(&t, "predictor." ++ p ++ "_proj", 1, h / 2, 1);
    }

    try conv(&t, "decoder.F0_conv", 1, 1, 3);
    try conv(&t, "decoder.N_conv", 1, 1, 3);
    try conv(&t, "decoder.asr_res.0", r, h, 1);
    try resBlk(&t, "decoder.encode", h + 2, w, false);
    try resBlk(&t, "decoder.decode.0", w + r + 2, w, false);
    try resBlk(&t, "decoder.decode.1", w + r + 2, c, true);

    const g = "decoder.generator";
    try lin(&t, g ++ ".m_source.l_linear", 1, 9);
    try conv(&t, g ++ ".noise_convs.0", c / 2, nf, 4);
    try conv(&t, g ++ ".noise_convs.1", c / 4, nf, 1);
    try resBlock1(&t, g ++ ".noise_res.0", c / 2);
    try resBlock1(&t, g ++ ".noise_res.1", c / 4);
    try t.rand(g ++ ".ups.0.weight", .{}, &.{ c, c / 2, 4 }, .F32, 0.2);
    try t.rand(g ++ ".ups.0.bias", .{}, &.{c / 2}, .F32, 0.1);
    try t.rand(g ++ ".ups.1.weight", .{}, &.{ c / 2, c / 4, 4 }, .F32, 0.2);
    try t.rand(g ++ ".ups.1.bias", .{}, &.{c / 4}, .F32, 0.1);
    inline for (0..3) |i| try resBlock1(&t, std.fmt.comptimePrint(g ++ ".resblocks.{d}", .{i}), c / 2);
    inline for (3..6) |i| try resBlock1(&t, std.fmt.comptimePrint(g ++ ".resblocks.{d}", .{i}), c / 4);
    try conv(&t, g ++ ".conv_post", nf, c / 4, 7);

    var v = sdk.tensors.Tensors.init(a, 8);
    try v.rand("af_heart", .{}, &.{ 8, 1, 2 * st }, .F32, 0.3);

    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"kokoro","hidden_dim":16,"style_dim":128,"n_layer":1,"max_dur":4,"text_encoder_kernel_size":3,
            \\ "plbert":{"hidden_size":16,"num_attention_heads":2,"intermediate_size":32,"num_hidden_layers":1,"max_position_embeddings":32},
            \\ "istftnet":{"upsample_rates":[2,2],"upsample_kernel_sizes":[4,4],"upsample_initial_channel":16,"gen_istft_n_fft":4,"gen_istft_hop_size":1},
            \\ "vocab":{"h":1,"ɛ":2,"l":3,"O":4," ":5,".":6,"ˈ":7}}
            },
            .{ .name = "g2p/us_gold.json", .bytes = "{\"hello\":\"hɛlˈO\"}" },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
            .{ .name = "voices.safetensors", .bytes = try v.bytes() },
        }),
        .body = "{\"input\":\"hello\",\"voice\":\"af_heart\",\"seed\":3}",
    };
}

const Tensors = sdk.tensors.Tensors;

fn lin(t: *Tensors, comptime p: []const u8, out: u64, in: u64) !void {
    try t.rand(p ++ ".weight", .{}, &.{ out, in }, .F32, 0.2);
    try t.rand(p ++ ".bias", .{}, &.{out}, .F32, 0.1);
}

fn conv(t: *Tensors, comptime p: []const u8, out: u64, in: u64, k: u64) !void {
    try t.rand(p ++ ".weight", .{}, &.{ out, in, k }, .F32, 0.2);
    try t.rand(p ++ ".bias", .{}, &.{out}, .F32, 0.1);
}

fn norm(t: *Tensors, comptime p: []const u8, n: u64, comptime w: []const u8, comptime b: []const u8) !void {
    try t.fill(p ++ "." ++ w, .{}, &.{n}, .F32, 1);
    try t.fill(p ++ "." ++ b, .{}, &.{n}, .F32, 0);
}

fn lstm(t: *Tensors, comptime p: []const u8, in: u64, hidden: u64) !void {
    inline for (.{ "", "_reverse" }) |sfx| {
        try t.rand(p ++ ".weight_ih_l0" ++ sfx, .{}, &.{ 4 * hidden, in }, .F32, 0.2);
        try t.rand(p ++ ".weight_hh_l0" ++ sfx, .{}, &.{ 4 * hidden, hidden }, .F32, 0.2);
        try t.rand(p ++ ".bias_ih_l0" ++ sfx, .{}, &.{4 * hidden}, .F32, 0.1);
        try t.rand(p ++ ".bias_hh_l0" ++ sfx, .{}, &.{4 * hidden}, .F32, 0.1);
    }
}

/// `AdainResBlk1d`: a learned shortcut when the width changes, a depthwise pool when it upsamples.
fn resBlk(t: *Tensors, comptime p: []const u8, in: u64, out: u64, upsample: bool) !void {
    try lin(t, p ++ ".norm1.fc", 2 * in, 128);
    if (upsample) {
        try t.rand(p ++ ".pool.weight", .{}, &.{ in, 1, 3 }, .F32, 0.2);
        try t.rand(p ++ ".pool.bias", .{}, &.{in}, .F32, 0.1);
    }
    try conv(t, p ++ ".conv1", out, in, 3);
    try lin(t, p ++ ".norm2.fc", 2 * out, 128);
    try conv(t, p ++ ".conv2", out, out, 3);
    if (in != out) try t.rand(p ++ ".conv1x1.weight", .{}, &.{ out, in, 1 }, .F32, 0.2);
}

/// The generator's `AdaINResBlock1`: three AdaIN + Snake + conv stages at width `ch`, kernel 3.
fn resBlock1(t: *Tensors, comptime p: []const u8, ch: u64) !void {
    inline for (0..3) |j| {
        const n = std.fmt.comptimePrint("{d}", .{j});
        inline for (.{ "1", "2" }) |k| {
            try lin(t, p ++ ".adain" ++ k ++ "." ++ n ++ ".fc", 2 * ch, 128);
            try t.fill(p ++ ".alpha" ++ k ++ "." ++ n, .{}, &.{ 1, ch, 1 }, .F32, 1);
            try conv(t, p ++ ".convs" ++ k ++ "." ++ n, ch, ch, 3);
        }
    }
}

// ── tests ──

const testing = std.testing;

test "kokoro plugin: a request is validated before any forward, by name" {
    inline for (.{
        .{ "{\"warm_only\":true}", "this model does not support voice cloning; nothing to warm" },
        .{ "{}", "missing 'input'" },
        .{ "{\"input\":\"\"}", "empty 'input'" },
        .{ "{\"input\":\"hi\",\"ref_audio\":\"UklG\"}", "this model does not support voice cloning; use 'voice' to pick or blend a built-in voice" },
        .{ "{\"input\":\"hi\",\"speed\":0}", "'speed' must be in (0, 5]" },
    }) |c| {
        var b = job.Body.parse(testing.allocator, c[0]).?;
        defer b.deinit();
        const v = Speech.parse(&b).bad;
        try testing.expectEqualStrings(c[1], v.why.slice());
    }
    var b = job.Body.parse(testing.allocator, "{\"text\":\"hi\",\"voice\":\"\",\"speed\":2,\"seed\":5}").?;
    defer b.deinit();
    const s = Speech.parse(&b).ok;
    try testing.expectEqualStrings(kokoro.DEFAULT_VOICE, s.voice);
    try testing.expect(s.speed == 2 and s.seed == 5);
}

test "kokoro plugin: a seeded job on the tiny pack is a deterministic WAV" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("RIFF", first[0..4]);
    try testing.expectEqualSlices(u8, first, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("unknown voice", (try pack.run("{\"input\":\"hello\",\"voice\":\"zz\"}")).refused.why.slice());
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
