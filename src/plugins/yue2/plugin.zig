//! YuE2-3B on the media table: `/v1/audio/music-generations` from style tags and lyrics (an ABC score planned or
//! given, semantic ids, flow-matching latents, a tiled Oobleck decode); the result is a 48 kHz stereo WAV.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const json_str = @import("../../sdk/util/json_text.zig");
const yue2 = @import("yue2.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .audio,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .step = &step,
    .encode = &encode,
    .extras = &extras,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Engine = struct {
    y: *yue2.Engine,
    song: ?yue2.Song = null,
    /// `,"abc":"…"` for the song's score, or empty.
    members: std.ArrayList(u8) = .empty,

    fn dropSong(e: *Engine) void {
        e.members.clearRetainingCapacity();
        const s = e.song orelse return;
        gpa.free(s.wav);
        if (s.abc) |t| gpa.free(t);
        e.song = null;
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const y = yue2.Engine.load(io, gpa, ctx.model_dir.slice()) catch {
        out.* = Verdict.refuse(.failed, "yue2: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        y.deinit();
        out.* = Verdict.refuse(.failed, "yue2: out of memory");
        return null;
    };
    e.* = .{ .y = y };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.dropSong();
    e.members.deinit(gpa);
    e.y.deinit();
    gpa.destroy(e);
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

/// ACE-Step's controls YuE2 cannot honour: each refused by name, never ignored.
const foreign_fields = [_][]const u8{ "bpm", "keyscale", "timesignature", "vocal_language", "ref_audio", "src_audio", "task", "instrumental" };

/// A song request, validated (the strings borrow the body).
fn parse(b: *const job.Body) union(enum) { ok: yue2.Request, bad: Verdict } {
    inline for (foreign_fields) |f| if (b.get(f)) |v| {
        const present = v == .string or v == .integer or (v == .bool and v.bool);
        if (present) return .{ .bad = bad("'" ++ f ++ "' is not supported by YuE2 (style and lyrics go in 'prompt' and 'lyrics'; a cover needs your own melody score in 'abc')") };
    };
    const prompt = b.str("prompt") orelse return .{ .bad = bad("missing 'prompt' (style tags: genre, instruments, mood, vocal)") };
    if (prompt.len == 0) return .{ .bad = bad("empty 'prompt'") };
    const lyrics = b.str("lyrics") orelse return .{ .bad = bad("missing 'lyrics' (YuE2 is lyric-conditioned; structure tags like [Verse] go on their own lines)") };
    if (std.mem.trim(u8, lyrics, " \t\r\n").len == 0) return .{ .bad = bad("empty 'lyrics'") };
    const cot = if (b.str("cot")) |c| (yue2.Cot.parse(c) orelse return .{ .bad = bad("'cot' must be one of full, melody, off") }) else yue2.Cot.full;
    var req: yue2.Request = .{ .style = prompt, .lyrics = lyrics, .cot = cot };
    if (b.str("abc")) |abc| {
        if (std.mem.trim(u8, abc, " \t\r\n").len == 0) return .{ .bad = bad("empty 'abc'") };
        if (cot == .off) return .{ .bad = bad("'abc' needs 'cot' melody or full — cot off renders without a score") };
        req.abc = abc;
    }
    const seconds = b.uint("duration_seconds") orelse yue2.MAX_SECONDS;
    if (seconds < yue2.MIN_SECONDS or seconds > yue2.MAX_SECONDS) return .{ .bad = bad("'duration_seconds' must be in [5,360]") };
    req.max_frames = @as(u32, @intCast(seconds)) * yue2.FRAME_RATE;
    const steps = b.uint("steps") orelse yue2.DEFAULT_STEPS;
    if (steps < 1 or steps > yue2.MAX_STEPS) return .{ .bad = bad("'steps' must be in [1,100]") };
    req.steps = @intCast(steps);
    if (b.float("cfg_scale")) |g| {
        if (!(g >= 0 and g <= yue2.MAX_CFG)) return .{ .bad = bad("'cfg_scale' must be in [0,20]") };
        req.cfg_scale = @floatCast(g);
    }
    if (b.uint("seed")) |s| req.seed = s;
    return .{ .ok = req };
}

fn step(ep: *anyopaque, r: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, r.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    const req = switch (parse(&body)) {
        .ok => |q| q,
        .bad => |v| return v,
    };
    const fits = e.y.fits(gpa, req) catch |err| return job.failure(err, "music generation failed");
    if (!fits) return bad("style + lyrics + score leave no room in the 24576-token context");
    const song = e.y.generate(gpa, req, job.progressOf(progress)) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        else => job.failure(err, "music generation failed"),
    };
    e.dropSong();
    e.song = song;
    if (song.abc) |score| {
        e.members.appendSlice(gpa, ",\"abc\":") catch return job.failure(error.OutOfMemory, "music generation failed");
        json_str.appendJsonString(gpa, &e.members, score) catch return job.failure(error.OutOfMemory, "music generation failed");
    }
    return .ok;
}

/// The score the song was rendered from, beside the WAV in a streamed result.
fn extras(ep: *anyopaque, out: *abi.Str) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    out.* = .of(e.members.items);
}

/// The WAV.
fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    const s = e.song orelse return Verdict.refuse(.failed, "yue2: no finished job to encode");
    out.* = .of(s.wav);
    return .ok;
}

/// A tiny random YuE2: hidden 64, two layers (AR + NAR paths), a 4-bit `lm_head` and AR `q_proj` beside bf16, a
/// byte-level tiktoken, and a four-stage decoder whose strides multiply to the 1920-sample hop.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    var t = T.init(a, 11);
    const hidden = 64;
    const hd = 32;
    const heads = 2;
    const ffn = 96;
    const vocab = 184704;
    const latent = 64;
    const paths = .{
        .{ "input_layernorm", "self_attn", "post_attention_layernorm", "mlp", true },
        .{ "nar_input_layernorm", "nar_self_attn", "nar_pre_mlp_layernorm", "nar_mlp", false },
    };
    for (0..2) |li| inline for (paths) |p| {
        try t.fill("model.layers.{d}." ++ p[0] ++ ".weight", .{li}, &.{hidden}, .BF16, 1);
        try t.fill("model.layers.{d}." ++ p[2] ++ ".weight", .{li}, &.{hidden}, .BF16, 1);
        const at = "model.layers.{d}." ++ p[1];
        if (p[4]) try q4(&t, at ++ ".q_proj", .{li}, heads * hd, hidden) else try t.rand(at ++ ".q_proj.weight", .{li}, &.{ heads * hd, hidden }, .BF16, 0.1);
        try t.rand(at ++ ".k_proj.weight", .{li}, &.{ hd, hidden }, .BF16, 0.1);
        try t.rand(at ++ ".v_proj.weight", .{li}, &.{ hd, hidden }, .BF16, 0.1);
        try t.rand(at ++ ".o_proj.weight", .{li}, &.{ hidden, heads * hd }, .BF16, 0.1);
        try t.fill(at ++ ".q_norm.weight", .{li}, &.{hd}, .BF16, 1);
        try t.fill(at ++ ".k_norm.weight", .{li}, &.{hd}, .BF16, 1);
        const mp = "model.layers.{d}." ++ p[3];
        try t.rand(mp ++ ".gate_proj.weight", .{li}, &.{ ffn, hidden }, .BF16, 0.1);
        try t.rand(mp ++ ".up_proj.weight", .{li}, &.{ ffn, hidden }, .BF16, 0.1);
        try t.rand(mp ++ ".down_proj.weight", .{li}, &.{ hidden, ffn }, .BF16, 0.1);
    };
    try t.rand("model.embed_tokens.weight", .{}, &.{ vocab, hidden }, .BF16, 1);
    try t.fill("model.norm.weight", .{}, &.{hidden}, .BF16, 1);
    try q4(&t, "lm_head", .{}, vocab, hidden);
    try t.rand("llm2vae.weight", .{}, &.{ latent, hidden }, .BF16, 0.1);
    try t.rand("llm2vae.bias", .{}, &.{latent}, .BF16, 0.1);
    try t.rand("vae2llm.weight", .{}, &.{ hidden, latent }, .BF16, 0.1);
    try t.rand("time_embedder.fc1.weight", .{}, &.{ hidden, 256 }, .BF16, 0.1);
    try t.rand("time_embedder.fc2.weight", .{}, &.{ hidden, hidden }, .BF16, 0.1);
    try t.rand("latent_pos_embed.pe", .{}, &.{ 256, hidden }, .BF16, 1);

    // Decoder stages run strides in reverse ([8,8,6,5] -> 5, 6, 8, 8) at 8 channels after a 16-channel stem.
    var v = T.init(a, 12);
    const strides = [_]u64{ 5, 6, 8, 8 };
    try v.rand("layers.0.weight", .{}, &.{ 16, 7, latent }, .F32, 0.05);
    try v.rand("layers.0.bias", .{}, &.{16}, .F32, 0.1);
    for (strides, 1..) |s, li| {
        const cin: u64 = if (li == 1) 16 else 8;
        try snakeParams(&v, "layers.{d}.layers.0", .{li}, cin);
        try v.rand("layers.{d}.layers.1.weight", .{li}, &.{ 8, 2 * s, cin }, .F32, 0.2);
        try v.rand("layers.{d}.layers.1.bias", .{li}, &.{8}, .F32, 0.1);
        for (2..5) |u| {
            try snakeParams(&v, "layers.{d}.layers.{d}.layers.0", .{ li, u }, 8);
            try v.rand("layers.{d}.layers.{d}.layers.1.weight", .{ li, u }, &.{ 8, 7, 8 }, .F32, 0.15);
            try v.rand("layers.{d}.layers.{d}.layers.1.bias", .{ li, u }, &.{8}, .F32, 0.1);
            try snakeParams(&v, "layers.{d}.layers.{d}.layers.2", .{ li, u }, 8);
            try v.rand("layers.{d}.layers.{d}.layers.3.weight", .{ li, u }, &.{ 8, 1, 8 }, .F32, 0.3);
            try v.rand("layers.{d}.layers.{d}.layers.3.bias", .{ li, u }, &.{8}, .F32, 0.1);
        }
    }
    try snakeParams(&v, "layers.5", .{}, 8);
    try v.rand("layers.6.weight", .{}, &.{ 2, 7, 8 }, .F32, 0.15);
    try v.rand("layers.6.bias", .{}, &.{2}, .F32, 0.1);

    var tik: std.ArrayList(u8) = .empty;
    for (0..256) |i| {
        var b64: [4]u8 = undefined;
        try tik.print(a, "{s} {d}\n", .{ std.base64.standard.Encoder.encode(&b64, &.{@as(u8, @intCast(i))}), i });
    }
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"yue2","hidden_size":64,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
            \\ "head_dim":32,"intermediate_size":96,"vocab_size":184704,"latent_dim":64,"max_latent_frames":256}
            },
            .{ .name = "vae_config.json", .bytes = "{\"decoder_config\":{\"strides\":[8,8,6,5],\"latent_dim\":64},\"decode_core_frames\":64,\"decode_halo_frames\":2}" },
            .{ .name = "qwen.tiktoken", .bytes = tik.items },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
            .{ .name = "vae.safetensors", .bytes = try v.bytes() },
        }),
        .body = "{\"prompt\":\"pop\",\"lyrics\":\"[verse]\\nla la\",\"cot\":\"off\",\"duration_seconds\":5,\"steps\":2,\"seed\":3}",
    };
}

/// A 4-bit affine projection at group 32: random packed words, bf16 scales and biases.
fn q4(t: *sdk.tensors.Tensors, comptime p: []const u8, args: anytype, out: u64, in: u64) !void {
    try t.words(p ++ ".weight", args, &.{ out, in / 8 });
    try t.rand(p ++ ".scales", args, &.{ out, in / 32 }, .BF16, 0.01);
    try t.rand(p ++ ".biases", args, &.{ out, in / 32 }, .BF16, 0.01);
}

fn snakeParams(t: *sdk.tensors.Tensors, comptime p: []const u8, args: anytype, ch: u64) !void {
    try t.rand(p ++ ".alpha", args, &.{ch}, .BF16, 0.3);
    try t.rand(p ++ ".beta", args, &.{ch}, .BF16, 0.3);
}

// ── tests ──

const testing = std.testing;

test "yue2 plugin: a request is validated before any forward, by name" {
    inline for (.{
        .{ "{\"bpm\":120}", "'bpm' is not supported by YuE2 (style and lyrics go in 'prompt' and 'lyrics'; a cover needs your own melody score in 'abc')" },
        .{ "{\"lyrics\":\"x\"}", "missing 'prompt' (style tags: genre, instruments, mood, vocal)" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\" \"}", "empty 'lyrics'" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"cot\":\"no\"}", "'cot' must be one of full, melody, off" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"cot\":\"off\",\"abc\":\"X:1\"}", "'abc' needs 'cot' melody or full — cot off renders without a score" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"duration_seconds\":4}", "'duration_seconds' must be in [5,360]" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"steps\":101}", "'steps' must be in [1,100]" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"cfg_scale\":21}", "'cfg_scale' must be in [0,20]" },
    }) |c| {
        var b = job.Body.parse(testing.allocator, c[0]).?;
        defer b.deinit();
        try testing.expectEqualStrings(c[1], parse(&b).bad.why.slice());
    }
    var b = job.Body.parse(testing.allocator, "{\"prompt\":\"p\",\"lyrics\":\"x\",\"bpm\":false,\"duration_seconds\":6}").?;
    defer b.deinit();
    const r = parse(&b).ok;
    try testing.expect(r.cot == .full and r.max_frames == 6 * yue2.FRAME_RATE and r.steps.? == yue2.DEFAULT_STEPS and r.seed == 831001);
}

test "yue2 plugin: a seeded song on the tiny pack is a deterministic stereo WAV" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("RIFF", first[0..4]);
    try testing.expectEqualSlices(u8, first, (try pack.run(pack.fx.body)).ok);
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
