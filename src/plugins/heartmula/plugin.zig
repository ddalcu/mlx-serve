//! HeartMuLa-oss-3B on the media table: `/v1/audio/music-generations` from style tags and lyrics, a 48 kHz stereo
//! WAV. ACE-Step's controls and the reference-audio slot are named refusals.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const mlx = @import("../../mlx/ffi.zig");
const job = @import("../../sdk/media/job.zig");
const heartmula = @import("heartmula.zig");
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
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;

const Engine = struct {
    e: *heartmula.Engine,
    wav: []u8 = &.{},
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    const h = heartmula.Engine.load(io, gpa, ctx.model_dir.slice(), .{ .ctx = ctx.stream.ctx }) catch {
        out.* = Verdict.refuse(.failed, "heartmula: the pack does not load");
        return null;
    };
    const e = gpa.create(Engine) catch {
        h.deinit();
        out.* = Verdict.refuse(.failed, "heartmula: out of memory");
        return null;
    };
    e.* = .{ .e = h };
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    gpa.free(e.wav);
    e.e.deinit();
    gpa.destroy(e);
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

/// Fields other music backends take that HeartMuLa cannot honour: each refused by name, never ignored. heartlib
/// raises on `ref_audio`; how training fed an instrumental track is not documented.
const foreign_fields = [_][]const u8{ "bpm", "keyscale", "timesignature", "vocal_language", "ref_audio", "src_audio", "instrumental" };

/// A song request, validated (the strings borrow the body).
fn parse(b: *const job.Body) union(enum) { ok: heartmula.Request, bad: Verdict } {
    inline for (foreign_fields) |f| if (b.get(f)) |v| {
        const present = v == .string or v == .integer or v == .float or (v == .bool and v.bool);
        if (present) return .{ .bad = bad("'" ++ f ++ "' is not supported by HeartMuLa (style tags go in 'prompt', lyrics with [Verse]-style tags in 'lyrics')") };
    };
    if (b.str("task")) |t| if (!std.mem.eql(u8, t, "text2music"))
        return .{ .bad = bad("'task' is not supported by HeartMuLa (text2music only)") };
    const prompt = b.str("prompt") orelse return .{ .bad = bad("missing 'prompt' (comma-separated style tags: genre, instruments, mood, vocal)") };
    if (std.mem.trim(u8, prompt, " \t\r\n").len == 0) return .{ .bad = bad("empty 'prompt'") };
    const lyrics = b.str("lyrics") orelse return .{ .bad = bad("missing 'lyrics' (HeartMuLa is lyric-conditioned; structure tags like [Verse] go on their own lines)") };
    if (std.mem.trim(u8, lyrics, " \t\r\n").len == 0) return .{ .bad = bad("empty 'lyrics'") };
    var req: heartmula.Request = .{ .tags = prompt, .lyrics = lyrics };
    const seconds = b.uint("duration_seconds") orelse heartmula.DEFAULT_SECONDS;
    if (seconds < heartmula.MIN_SECONDS or seconds > heartmula.MAX_SECONDS) return .{ .bad = bad("'duration_seconds' must be in [5,360]") };
    req.seconds = @intCast(seconds);
    const steps = b.uint("steps") orelse heartmula.DEFAULT_STEPS;
    if (steps < 4 or steps > 50) return .{ .bad = bad("'steps' must be in [4,50]") };
    req.steps = @intCast(steps);
    if (b.float("cfg_scale")) |g| {
        if (!(g >= 1 and g <= 10)) return .{ .bad = bad("'cfg_scale' must be in [1,10]") };
        req.sampling.cfg = @floatCast(g);
    }
    if (b.float("temperature")) |t| {
        if (!(t > 0 and t <= 5)) return .{ .bad = bad("'temperature' must be in (0,5]") };
        req.sampling.temperature = @floatCast(t);
    }
    if (b.get("top_k") != null) {
        const k = b.uint("top_k") orelse 0;
        if (k < 1 or k > 1000) return .{ .bad = bad("'top_k' must be in [1,1000]") };
        req.sampling.top_k = @intCast(k);
    }
    req.sampling.seed = b.uint("seed") orelse 42;
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
    const wav = e.e.generate(gpa, req, job.progressOf(progress)) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        error.PromptTooLong => bad("prompt + lyrics + 'duration_seconds' exceed the backbone's 8192 positions (shorten the lyrics or the duration)"),
        else => job.failure(err, "music generation failed"),
    };
    gpa.free(e.wav);
    e.wav = wav;
    return .ok;
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.wav.len == 0) return Verdict.refuse(.failed, "heartmula: no finished job to encode");
    out.* = .of(e.wav);
    return .ok;
}

/// A tiny random HeartMuLa over a byte tokenizer: a one-layer backbone and depth decoder at 32 (llama3 RoPE, a 4-bit
/// text table and codebook-0 head), a one-block-per-stack DiT at 16/32, and the published 1920-sample scalar decoder
/// at one to 32 channels. The song stops at the first frame holding a code past `audio_eos_id`.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    const d = 32;
    const ffn = 64;
    const av = 64; // audio vocab per codebook
    var lm = T.init(a, 21);
    try q4(&lm, "text_embeddings", .{}, 258, d);
    try lm.rand("audio_embeddings.weight", .{}, &.{ 8 * av, d }, .BF16, 0.5);
    try lm.rand("unconditional_text_embedding.weight", .{}, &.{ 1, d }, .BF16, 0.5);
    try lm.rand("muq_linear.bias", .{}, &.{d}, .BF16, 0.5);
    try lm.rand("projection.weight", .{}, &.{ d, d }, .BF16, 0.2);
    try q4(&lm, "codebook0_head", .{}, av, d);
    for (0..7) |i| try lm.rand("audio_head.{d}.weight", .{i}, &.{ av, d }, .BF16, 0.5);
    inline for (.{ "backbone", "decoder" }) |p| {
        try lm.fill(p ++ ".norm.weight", .{}, &.{d}, .BF16, 1);
        try lm.fill(p ++ ".layers.0.input_layernorm.weight", .{}, &.{d}, .BF16, 1);
        try lm.fill(p ++ ".layers.0.post_attention_layernorm.weight", .{}, &.{d}, .BF16, 1);
        inline for (.{ .{ "self_attn.q_proj", d, d }, .{ "self_attn.k_proj", 16, d }, .{ "self_attn.v_proj", 16, d }, .{ "self_attn.o_proj", d, d }, .{ "mlp.gate_proj", ffn, d }, .{ "mlp.up_proj", ffn, d }, .{ "mlp.down_proj", d, ffn } }) |l|
            try lm.rand(p ++ ".layers.0." ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .BF16, 0.2);
    }

    var cd = T.init(a, 22);
    const e = "flow_matching.estimator.";
    for (0..8) |q| try cd.rand("flow_matching.vq_embed.layers.{d}.codebook", .{q}, &.{ av, 4 }, .F32, 1);
    try cd.rand("flow_matching.vq_embed.project_out.weight", .{}, &.{ 16, 4 }, .F32, 0.3);
    try cd.rand("flow_matching.vq_embed.project_out.bias", .{}, &.{16}, .F32, 0.1);
    try cd.rand("flow_matching.cond_feature_emb.weight", .{}, &.{ 16, 16 }, .F32, 0.3);
    try cd.rand("flow_matching.cond_feature_emb.bias", .{}, &.{16}, .F32, 0.1);
    try projectLayer(&cd, e ++ "proj_in", 16, 32);
    try projectLayer(&cd, e ++ "connection_proj", 32, 48);
    try projectLayer(&cd, e ++ "proj_out", 8, 32);
    inline for (.{ "transformer_blocks", "transformer_blocks_2" }, .{ "adaln_single", "adaln_single_2" }, .{ "scale_shift_table", "scale_shift_table_2" }, .{ 16, 32 }) |bp, ap, tp, dim| {
        try cd.rand(e ++ tp, .{}, &.{ 2, dim }, .F32, 0.1);
        try cd.rand(e ++ ap ++ ".emb.timestep_embedder.linear_1.weight", .{}, &.{ dim, 512 }, .BF16, 0.05);
        try cd.rand(e ++ ap ++ ".emb.timestep_embedder.linear_1.bias", .{}, &.{dim}, .F32, 0.1);
        try cd.rand(e ++ ap ++ ".emb.timestep_embedder.linear_2.weight", .{}, &.{ dim, dim }, .BF16, 0.2);
        try cd.rand(e ++ ap ++ ".emb.timestep_embedder.linear_2.bias", .{}, &.{dim}, .F32, 0.1);
        try cd.rand(e ++ ap ++ ".linear.weight", .{}, &.{ 6 * dim, dim }, .BF16, 0.1);
        try cd.rand(e ++ ap ++ ".linear.bias", .{}, &.{6 * dim}, .F32, 0.1);
        const b = e ++ bp ++ ".0.";
        try cd.fill(b ++ "attn_norm.weight", .{}, &.{dim}, .F32, 1);
        try cd.fill(b ++ "mlp_norm.weight", .{}, &.{dim}, .F32, 1);
        try cd.rand(b ++ "scale_shift_table", .{}, &.{ 6, dim }, .F32, 0.1);
        inline for (.{ "attn.q_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj" }) |n| try cd.rand(b ++ n ++ ".weight", .{}, &.{ dim, dim }, .BF16, 0.2);
        try cd.rand(b ++ "mlp.gate.weight", .{}, &.{ 2 * dim, dim }, .BF16, 0.2);
        try cd.rand(b ++ "mlp.up.weight", .{}, &.{ 2 * dim, dim }, .BF16, 0.2);
        try cd.rand(b ++ "mlp.down.weight", .{}, &.{ dim, 2 * dim }, .BF16, 0.2);
    }
    const sq = "scalar_model.decoder.";
    try cd.rand(sq ++ "0.weight", .{}, &.{ 32, 5, 4 }, .F32, 0.2);
    try cd.rand(sq ++ "0.bias", .{}, &.{32}, .F32, 0.1);
    const factors = [_]u64{ 5, 4, 4, 4, 3 };
    for (factors, 1..) |f, bi| {
        const cin: u64 = @as(u64, 64) >> @intCast(bi);
        const cout = cin / 2;
        try cd.rand(sq ++ "{d}.up_conv.layer.weight", .{bi}, &.{ cout, 2 * f, cin }, .F32, 0.2);
        try cd.rand(sq ++ "{d}.up_conv.layer.bias", .{bi}, &.{cout}, .F32, 0.1);
        for (0..5) |u| {
            try cd.rand(sq ++ "{d}.convs.{d}.conv1.weight", .{ bi, u }, &.{ cout, 7, cout }, .F32, 0.2);
            try cd.rand(sq ++ "{d}.convs.{d}.conv1.bias", .{ bi, u }, &.{cout}, .F32, 0.1);
            try cd.rand(sq ++ "{d}.convs.{d}.conv2.weight", .{ bi, u }, &.{ cout, 1, cout }, .F32, 0.3);
            try cd.rand(sq ++ "{d}.convs.{d}.conv2.bias", .{ bi, u }, &.{cout}, .F32, 0.1);
            try cd.fill(sq ++ "{d}.convs.{d}.activation1.weight", .{ bi, u }, &.{1}, .F32, 0.25);
            try cd.fill(sq ++ "{d}.convs.{d}.activation2.weight", .{ bi, u }, &.{1}, .F32, 0.25);
        }
    }
    try cd.rand(sq ++ "6.conv.weight", .{}, &.{ 1, 7, 1 }, .F32, 0.3);
    try cd.rand(sq ++ "6.conv.bias", .{}, &.{1}, .F32, 0.1);
    try cd.fill(sq ++ "6.activation.weight", .{}, &.{1}, .F32, 0.25);
    try cd.rand(sq ++ "7.weight", .{}, &.{ 1, 7, 1 }, .F32, 0.3);
    try cd.rand(sq ++ "7.bias", .{}, &.{1}, .F32, 0.1);

    const specials = [_]sdk.byte_tokenizer.Special{ .{ .id = 256, .content = "<|begin_of_text|>" }, .{ .id = 257, .content = "<|end_of_text|>" } };
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"heartmula","audio_num_codebooks":8,"audio_vocab_size":64,"text_vocab_size":258,
            \\ "backbone":{"num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,
            \\   "intermediate_size":64,"max_position_embeddings":512,"rope_theta":500000.0,
            \\   "rope_scaling":{"rope_type":"llama3","factor":32.0}},
            \\ "decoder":{"num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,
            \\   "intermediate_size":64,"max_position_embeddings":16,"rope_theta":500000.0},
            \\ "codec":{"dim":16,"codebook_size":64,"codebook_dim":4,"num_quantizers":8,"attention_head_dim":8,"in_channels":32,
            \\   "num_attention_heads":2,"num_layers":1,"num_layers_2":1,"out_channels":8,"sample_rate":48000,"num_samples":2,
            \\   "upsample_factors":[5,4,4,4,3],"upsample_kernel_sizes":[10,8,8,8,6],"latent_hidden_dim":4,"delay_kernel_size":5,
            \\   "init_channel":1,"res_kernel_size":7,"default_kernel_size":7},
            \\ "text_bos_id":256,"text_eos_id":257,"audio_eos_id":61,"empty_id":0}
            },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &specials) },
            .{ .name = "model.safetensors", .bytes = try lm.bytes() },
            .{ .name = "codec.safetensors", .bytes = try cd.bytes() },
        }),
        .body = "{\"prompt\":\"pop, piano\",\"lyrics\":\"[Verse]\\nla la\",\"duration_seconds\":5,\"steps\":4,\"seed\":3}",
    };
}

/// A 4-bit affine table or projection at group 32: random packed words, bf16 scales and biases.
fn q4(t: *sdk.tensors.Tensors, comptime p: []const u8, args: anytype, out: u64, in: u64) !void {
    try t.words(p ++ ".weight", args, &.{ out, in / 8 });
    try t.rand(p ++ ".scales", args, &.{ out, in / 32 }, .BF16, 0.05);
    try t.rand(p ++ ".biases", args, &.{ out, in / 32 }, .BF16, 0.05);
}

/// heartlib's ProjectLayer: a k=3 Conv1d (MLX layout) then a Linear.
fn projectLayer(t: *sdk.tensors.Tensors, comptime p: []const u8, out: u64, in: u64) !void {
    try t.rand(p ++ ".ffn_1.weight", .{}, &.{ out, 3, in }, .BF16, 0.1);
    try t.rand(p ++ ".ffn_1.bias", .{}, &.{out}, .F32, 0.1);
    try t.rand(p ++ ".ffn_2.weight", .{}, &.{ out, out }, .BF16, 0.2);
    try t.rand(p ++ ".ffn_2.bias", .{}, &.{out}, .F32, 0.1);
}

// ── tests ──

const testing = std.testing;

test "heartmula plugin: a request is validated before any forward, by name" {
    inline for (.{
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"ref_audio\":\"a.wav\"}", "'ref_audio' is not supported by HeartMuLa (style tags go in 'prompt', lyrics with [Verse]-style tags in 'lyrics')" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"instrumental\":true}", "'instrumental' is not supported by HeartMuLa (style tags go in 'prompt', lyrics with [Verse]-style tags in 'lyrics')" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"task\":\"cover\"}", "'task' is not supported by HeartMuLa (text2music only)" },
        .{ "{\"lyrics\":\"x\"}", "missing 'prompt' (comma-separated style tags: genre, instruments, mood, vocal)" },
        .{ "{\"prompt\":\"p\"}", "missing 'lyrics' (HeartMuLa is lyric-conditioned; structure tags like [Verse] go on their own lines)" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\" \"}", "empty 'lyrics'" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"duration_seconds\":4}", "'duration_seconds' must be in [5,360]" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"steps\":3}", "'steps' must be in [4,50]" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"cfg_scale\":0.5}", "'cfg_scale' must be in [1,10]" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"temperature\":0}", "'temperature' must be in (0,5]" },
        .{ "{\"prompt\":\"p\",\"lyrics\":\"x\",\"top_k\":0}", "'top_k' must be in [1,1000]" },
    }) |c| {
        var b = job.Body.parse(testing.allocator, c[0]).?;
        defer b.deinit();
        try testing.expectEqualStrings(c[1], parse(&b).bad.why.slice());
    }
    var b = job.Body.parse(testing.allocator, "{\"prompt\":\"p\",\"lyrics\":\"x\",\"instrumental\":false,\"task\":\"text2music\"}").?;
    defer b.deinit();
    const r = parse(&b).ok;
    try testing.expect(r.seconds == heartmula.DEFAULT_SECONDS and r.steps == heartmula.DEFAULT_STEPS and r.sampling.cfg == heartmula.DEFAULT_CFG and r.sampling.top_k == 50);
}

test "heartmula plugin: a seeded song on the tiny pack is a deterministic stereo WAV" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    try testing.expectEqualStrings("RIFF", first[0..4]);
    try testing.expect(first.len > 44);
    try testing.expectEqualSlices(u8, first, (try pack.run(pack.fx.body)).ok);
}

test "heartmula plugin: the load bill is both weight files plus the working set" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const fx = try conformanceFixture(arena.allocator());
    const mem: sdk.fixture.MemDir = .{ .files = fx.files };
    const bill = config.billLoad(&mem.packDir(), &.{});
    try testing.expect(bill > config.gen_buffer_bytes);
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
