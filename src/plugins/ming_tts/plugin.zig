//! Ming-3.0-tts-0.5b on the media table: `/v1/audio/speech` from text, with built-in voices, voice design, every task of
//! the reference and zero-shot cloning from a reference WAV. The result is a 44.1 kHz mono WAV.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const log = @import("../../sdk/util/log.zig");
const wav = @import("../../sdk/media/wav.zig");
const mlx = @import("../../mlx/ffi.zig");
const engine = @import("engine.zig");
const text = @import("text.zig");
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

const Plugin = struct {
    engine: engine.Engine,
    out: []const u8 = &.{},

    fn setOut(p: *Plugin, bytes: []const u8) void {
        gpa.free(p.out);
        p.out = bytes;
    }
};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const buf = a.alloc(u8, 1 << 20) catch {
        out.* = Verdict.refuse(.failed, "ming_tts: out of memory");
        return null;
    };
    const cfg = ctx.dir.readFile("config.json", buf, out) orelse {
        out.* = Verdict.refuse(.failed, "ming_tts: the pack has no readable config.json");
        return null;
    };
    const dims = config.parseDims(a, cfg) catch |e| {
        out.* = switch (e) {
            error.QuantizedPack => Verdict.refuse(.unsupported, "ming_tts: quantized packs are not supported; use the bf16 inclusionAI/Ming-3.0-tts-0.5b"),
            error.InvalidConfig => Verdict.refuse(.unsupported, "ming_tts: the pack's config.json declares an unusable geometry"),
        };
        return null;
    };
    const p = gpa.create(Plugin) catch {
        out.* = Verdict.refuse(.failed, "ming_tts: out of memory");
        return null;
    };
    p.* = .{ .engine = engine.Engine.init(io, gpa, .{ .ctx = ctx.stream.ctx }, ctx.model_dir.slice(), dims) catch |e| {
        log.err("[ming_tts] load: {s}\n", .{@errorName(e)});
        gpa.destroy(p);
        out.* = Verdict.refuse(.failed, "ming_tts: the pack does not load");
        return null;
    } };
    return p;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const p: *Plugin = @ptrCast(@alignCast(ep));
    p.setOut(&.{});
    p.engine.deinit();
    gpa.destroy(p);
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

const known_fields = [_][]const u8{
    "model",       "input",    "text",      "stream",   "response_format", "speed",    "task",
    "voice",       "language", "ref_audio", "ref_text", "instruction",     "controls", "seed",
    "pause_scale",
};

/// A string field: a wrong type is a refusal, blank or `null` is absent.
fn optStr(req: *const abi.Request, b: *const job.Body, comptime key: []const u8, out: *?[]const u8) ?Verdict {
    const v = b.get(key) orelse return null;
    switch (v) {
        .null => {},
        .string => |s| if (text.strip(s).len > 0) {
            out.* = text.strip(s);
        },
        else => return req.refuseFmt(.unsupported, "a field is not a string", "'{s}' must be a string", .{key}),
    }
    return null;
}

/// `controls`: string values; a key the task model has no slot for is kept for the validator to refuse by name.
fn readControls(req: *const abi.Request, a: std.mem.Allocator, b: *const job.Body, r: *engine.GenerateRequest) ?Verdict {
    const v = b.get("controls") orelse return null;
    if (v == .null) return null;
    if (v != .object) return bad("'controls' must be an object of strings");
    var unknown: std.ArrayList([]const u8) = .empty;
    var it = v.object.iterator();
    while (it.next()) |kv| {
        const k = kv.key_ptr.*;
        if (kv.value_ptr.* != .string) return req.refuseFmt(.unsupported, "a control is not a string", "'controls.{s}' must be a string", .{k});
        const s = kv.value_ptr.string;
        var known = false;
        inline for (@typeInfo(text.Controls).@"struct".field_names) |f| if (std.mem.eql(u8, k, f)) {
            @field(r.controls, f) = s;
            known = true;
        };
        if (!known) unknown.append(a, k) catch return bad("out of memory");
    }
    r.unknown_controls = unknown.items;
    return null;
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const p: *Plugin = @ptrCast(@alignCast(ep));
    var body = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer body.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();

    var fields = body.parsed.value.object.iterator();
    while (fields.next()) |kv| {
        const name = kv.key_ptr.*;
        for (known_fields) |f| {
            if (std.mem.eql(u8, f, name)) break;
        } else return req.refuseFmt(.unsupported, "a request field is not supported", "'{s}' is not a field this model honors", .{name});
    }
    if (body.get("response_format")) |f| if (f != .string or !std.mem.eql(u8, f.string, "wav")) return bad("'response_format' must be \"wav\"");
    if (body.get("speed")) |f| if (!(f == .integer and f.integer == 1) and !(f == .float and f.float == 1.0)) return bad("'speed' must be 1: use 'controls' with the BASIC task");

    const input = body.str("input") orelse body.str("text") orelse return bad("missing 'input'");
    if (text.isBlank(input)) return bad("empty 'input'");
    const chars = std.unicode.utf8CountCodepoints(input) catch return bad("'input' is not valid UTF-8");
    if (chars > engine.MAX_TEXT_CHARS) return req.refuseFmt(.unsupported, "'input' is too long", "'input' is {d} characters; the limit is {d}", .{ chars, engine.MAX_TEXT_CHARS });

    var r: engine.GenerateRequest = .{ .task = "", .text = input };
    var task: ?[]const u8 = null;
    if (optStr(req, &body, "task", &task)) |v| return v;
    if (optStr(req, &body, "voice", &r.voice)) |v| return v;
    if (optStr(req, &body, "language", &r.language)) |v| return v;
    if (optStr(req, &body, "instruction", &r.instruction)) |v| return v;
    if (optStr(req, &body, "ref_text", &r.prompt_text)) |v| return v;
    if (r.language) |l| if (std.ascii.eqlIgnoreCase(l, "auto")) {
        r.language = null;
    };
    if (readControls(req, a, &body, &r)) |v| return v;

    var ref_b64: ?[]const u8 = null;
    if (optStr(req, &body, "ref_audio", &ref_b64)) |v| return v;
    if (ref_b64) |b64| r.prompt_wav = job.base64Decode(a, b64) catch return bad("ref_audio is not valid base64");

    if (body.get("seed")) |s| switch (s) {
        .null => {},
        .integer => |n| {
            if (n < 0) return bad("'seed' must not be negative");
            r.seed = @intCast(n);
        },
        else => return bad("'seed' must be an integer"),
    };
    if (body.get("pause_scale")) |s| {
        r.pause_scale = switch (s) {
            .null => null,
            .integer => |n| @floatFromInt(n),
            .float => |f| @floatCast(f),
            else => return bad("'pause_scale' must be a number"),
        };
        if (r.pause_scale) |x| if (!(x >= 0 and x <= engine.MAX_PAUSE_SCALE)) return bad("'pause_scale' must be between 0 and 4");
    }

    const e = &p.engine;
    e.prog = job.progressOf(progress);
    defer e.prog = null;
    const saved = e.useOptions(r);
    defer e.restoreOptions(saved);
    var d: engine.Diag = .{};
    const samples = run(e, task, &r, &d) catch |err| return switch (err) {
        error.BadRequest => req.refuseFmt(.unsupported, "the request does not fit its task", "{s}", .{d.msg()}),
        error.Cancelled => job.client_left,
        error.BadWav, error.EmptyPrompt => bad("ref_audio is not a decodable WAV"),
        error.PromptTooLong => bad("ref_audio is longer than the 20 s limit"),
        error.NoSpeakerEncoder => bad("ref_audio requires campplus.onnx or campplus.safetensors in the pack"),
        else => job.failure(err, "synthesis failed"),
    };
    defer gpa.free(samples);
    const bytes = wav.encodePcm16Mono(gpa, samples, engine.SAMPLE_RATE) catch |err| return job.failure(err, "WAV encoding failed");
    p.setOut(bytes);
    return .ok;
}

/// With a `task`, its field rules; without one, the reference picks the path: a clone, or a built-in or designed voice.
fn run(e: *engine.Engine, task: ?[]const u8, r: *engine.GenerateRequest, d: *engine.Diag) ![]f32 {
    if (task) |t| {
        r.task = t;
        return e.generate(r.*, d);
    }
    if (!r.controls.isEmpty() or r.unknown_controls.len > 0) return d.set("'controls' needs a 'task'", .{});
    if (r.prompt_wav) |w| {
        if (r.voice != null or r.instruction != null) return d.set("'ref_audio' clones its own voice: leave out 'voice' and 'instruction'", .{});
        if (r.language != null) {
            r.task = "MULTILINGUAL";
            return e.generate(r.*, d);
        }
        return e.clone(r.text, w, r.prompt_text, r.seed, d);
    }
    if (r.prompt_text != null) return d.set("'ref_text' needs 'ref_audio'", .{});
    if (r.language != null) {
        if (r.voice != null or r.instruction != null) return d.set("'language' needs 'ref_audio' when a 'voice' or 'instruction' sets the voice", .{});
        r.task = "TTS";
        return e.generate(r.*, d);
    }
    return e.tts(r.text, r.voice, r.instruction, r.seed, d);
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const p: *Plugin = @ptrCast(@alignCast(ep));
    if (p.out.len == 0) return Verdict.refuse(.failed, "ming_tts: no finished job to encode");
    out.* = .of(p.out);
    return .ok;
}

/// A tiny random Ming: every stack one layer wide at hidden 32 (DiT 32), four heads where the reference has more, a
/// byte-level tokenizer with the contract's special ids. No CAM++ weights, so a clone is refused by name. The stop
/// head's bias ends a seeded job after its fifth step.
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const T = sdk.tensors.Tensors;
    var t = T.init(a, 41);
    const h = 32;
    const d = 32;
    try t.rand("model.model.embed_tokens.weight", .{}, &.{ 151936, h }, .BF16, 0.5);
    try qwenStack(&t, "model.model", 1);
    try t.rand("stop_head.weight", .{}, &.{ 2, h }, .BF16, 0.01);
    try t.values("stop_head.bias", .{}, &.{2}, .BF16, &.{ -8, 8 });
    try lin(&t, "spk_head", h, 192, true);
    const F = "flowloss.cfm.model";
    try ditStack(&t, F);
    try lin(&t, F ++ ".t_embedder.time_mlp.0", d, 256, true);
    try lin(&t, F ++ ".t_embedder.time_mlp.2", d, d, true);
    try lin(&t, F ++ ".x_embedder", d, 64, true);
    try lin(&t, F ++ ".c_embedder.cond_embedder", d, h, true);
    try t.fill(F ++ ".final_layer.norm_final.weight", .{}, &.{d}, .BF16, 1);
    try lin(&t, F ++ ".final_layer.linear", 64, d, true);
    const A = "linear_proj_audio";
    try ditStack(&t, A);
    try lin(&t, A ++ ".x_embedder", d, 64, true);
    try t.rand(A ++ ".word_embedder.weight", .{}, &.{ 1, d }, .BF16, 0.2);
    try t.fill(A ++ ".final_layer.norm_final.weight", .{}, &.{d}, .BF16, 1);
    try lin(&t, A ++ ".final_layer.linear", h, d, true);
    try qwenStack(&t, "audio.encoder.encoder", 1);
    try qwenStack(&t, "audio.encoder.aggregator", 4);
    try lin(&t, "audio.encoder.fc1", h, 882, true);
    try lin(&t, "audio.encoder.fc2", h, h, true);
    try lin(&t, "audio.encoder.fc3", 128, h, true);
    try t.rand("audio.encoder.cls_embed", .{}, &.{ 1, 1, h }, .BF16, 0.2);
    try qwenStack(&t, "audio.decoder.decoder", 1);
    try lin(&t, "audio.decoder.fc1", h, 64, true);
    try lin(&t, "audio.decoder.head.out", 3530, h, true);
    try t.fill("audio.decoder.head.istft.window", .{}, &.{3528}, .BF16, 0.5);
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"dense","aggregator_config":{"depth":1,"hidden_size":32,"num_heads":2},
            \\ "ditar_config":{"depth":1,"hidden_size":32,"num_heads":2},
            \\ "llm_config":{"num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"rope_theta":1e6},
            \\ "audio_tokenizer_config":{
            \\  "enc_kwargs":{"backbone":{"num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"rope_theta":1e6,"use_sliding_window":true,"sliding_window":64}},
            \\  "dec_kwargs":{"backbone":{"num_hidden_layers":1,"hidden_size":32,"num_attention_heads":2,"num_key_value_heads":1,"rope_theta":1e6,"use_sliding_window":true,"sliding_window":64}}}}
            },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{
                .{ .id = 151643, .content = "<|endoftext|>" },
                .{ .id = 151644, .content = "<|im_start|>" },
                .{ .id = 151645, .content = "<|im_end|>" },
                .{ .id = 151652, .content = "<|vision_start|>" },
                .{ .id = 151653, .content = "<|vision_end|>" },
                .{ .id = 151654, .content = "<|vision_pad|>" },
                .{ .id = 151705, .content = "<audioPatch>" },
                .{ .id = 151706, .content = "<audio>" },
            }) },
            .{ .name = "model.safetensors", .bytes = try t.bytes() },
        }),
        .body = "{\"input\":\"hi there\",\"voice\":\"UK_George\",\"seed\":1}",
    };
}

fn lin(t: *sdk.tensors.Tensors, comptime p: []const u8, out: u64, in: u64, bias: bool) !void {
    try t.rand(p ++ ".weight", .{}, &.{ out, in }, .BF16, 0.05);
    if (bias) try t.rand(p ++ ".bias", .{}, &.{out}, .BF16, 0.01);
}

fn qwenStack(t: *sdk.tensors.Tensors, comptime prefix: []const u8, n: usize) !void {
    const h = 32;
    for (0..n) |i| {
        try t.fill(prefix ++ ".layers.{d}.input_layernorm.weight", .{i}, &.{h}, .BF16, 1);
        try t.fill(prefix ++ ".layers.{d}.post_attention_layernorm.weight", .{i}, &.{h}, .BF16, 1);
        try t.rand(prefix ++ ".layers.{d}.self_attn.q_proj.weight", .{i}, &.{ h, h }, .BF16, 0.05);
        try t.rand(prefix ++ ".layers.{d}.self_attn.q_proj.bias", .{i}, &.{h}, .BF16, 0.01);
        try t.rand(prefix ++ ".layers.{d}.self_attn.k_proj.weight", .{i}, &.{ 16, h }, .BF16, 0.05);
        try t.rand(prefix ++ ".layers.{d}.self_attn.k_proj.bias", .{i}, &.{16}, .BF16, 0.01);
        try t.rand(prefix ++ ".layers.{d}.self_attn.v_proj.weight", .{i}, &.{ 16, h }, .BF16, 0.05);
        try t.rand(prefix ++ ".layers.{d}.self_attn.v_proj.bias", .{i}, &.{16}, .BF16, 0.01);
        try t.rand(prefix ++ ".layers.{d}.self_attn.o_proj.weight", .{i}, &.{ h, h }, .BF16, 0.05);
        try t.rand(prefix ++ ".layers.{d}.mlp.gate_proj.weight", .{i}, &.{ 64, h }, .BF16, 0.05);
        try t.rand(prefix ++ ".layers.{d}.mlp.up_proj.weight", .{i}, &.{ 64, h }, .BF16, 0.05);
        try t.rand(prefix ++ ".layers.{d}.mlp.down_proj.weight", .{i}, &.{ h, 64 }, .BF16, 0.05);
    }
    try t.fill(prefix ++ ".norm.weight", .{}, &.{h}, .BF16, 1);
}

fn ditStack(t: *sdk.tensors.Tensors, comptime prefix: []const u8) !void {
    const d = 32;
    try t.fill(prefix ++ ".blocks.0.norm1.weight", .{}, &.{d}, .BF16, 1);
    try t.fill(prefix ++ ".blocks.0.norm2.weight", .{}, &.{d}, .BF16, 1);
    inline for (.{ "to_q", "to_k", "to_v", "to_out.0" }) |n| try lin(t, prefix ++ ".blocks.0.attn." ++ n, d, d, true);
    try lin(t, prefix ++ ".blocks.0.mlp.ff.0.0", 128, d, true);
    try lin(t, prefix ++ ".blocks.0.mlp.ff.2", d, 128, true);
    try t.rand(prefix ++ ".rotary_embed.inv_freq", .{}, &.{8}, .F32, 0.5);
}

// ── tests ──

const testing = std.testing;
const media_pack = @import("../../sdk/testing/media_pack.zig");

test {
    _ = @import("audio.zig");
    _ = @import("fr.zig");
    _ = @import("text.zig");
}

test "ming_tts plugin: a job on the tiny pack is a 44.1 kHz mono WAV, repeatable by seed" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var pack = try media_pack.MediaPack(@This()).open(a);
    defer pack.close();
    const first = try a.dupe(u8, (try pack.run("{\"input\":\"hi there\",\"voice\":\"UK_George\",\"seed\":7}")).ok);
    const again = (try pack.run("{\"input\":\"hi there\",\"voice\":\"UK_George\",\"seed\":7}")).ok;
    try testing.expectEqualSlices(u8, first, again);
    const d = try wav.decode(a, first);
    try testing.expectEqual(@as(u32, 44100), d.sample_rate);
    try testing.expectEqual(@as(u16, 1), d.channels);
}

test "ming_tts plugin: a request is refused by name, never answered in another voice" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try media_pack.MediaPack(@This()).open(arena.allocator());
    defer pack.close();
    const cases = [_]struct { []const u8, []const u8 }{
        .{ "{}", "missing 'input'" },
        .{ "{\"input\":\"  \"}", "empty 'input'" },
        .{ "{\"input\":\"hi\",\"voice\":\"Nobody\"}", "the request does not fit its task" },
        .{ "{\"input\":\"hi\",\"ref_audio\":\"!!\"}", "ref_audio is not valid base64" },
        .{ "{\"input\":\"hi\",\"ref_audio\":\"UklG\"}", "ref_audio is not a decodable WAV" },
        .{ "{\"input\":\"hi\",\"ref_text\":\"x\"}", "the request does not fit its task" },
        .{ "{\"input\":\"hi\",\"task\":\"NOPE\"}", "the request does not fit its task" },
        .{ "{\"input\":\"hi\",\"pause_scale\":9}", "'pause_scale' must be between 0 and 4" },
        .{ "{\"input\":\"hi\",\"speed\":2}", "'speed' must be 1: use 'controls' with the BASIC task" },
        .{ "{\"input\":\"hi\",\"response_format\":\"mp3\"}", "'response_format' must be \"wav\"" },
        .{ "{\"input\":\"hi\",\"warm_only\":true}", "a request field is not supported" },
    };
    for (cases) |c| {
        const o = try pack.run(c[0]);
        try testing.expectEqualStrings(c[1], o.refused.why.slice());
    }
}
