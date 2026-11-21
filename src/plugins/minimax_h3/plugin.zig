//! MiniMax-H3 text-to-audio-video on the media table (`/v1/video/generations`): FL2VA keyframes, REF2VA references,
//! the Turbo LoRA, chained windows and FastH3 Omni's PDD student. The request surface is NARROWER than LTX's: no CFG
//! scale and no pipeline mode, frames on a 17k+5 ladder; anything this backend cannot honor is a named refusal. The
//! result is the response's JSON members (24 fps RGB, 32 kHz stereo PCM16).

const std = @import("std");
const abi = @import("../../abi/media.zig");
const sdk = @import("../../sdk/root.zig");
const job = @import("../../sdk/media/job.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const ops = @import("../../mlx/ops.zig");
const lora = @import("../../sdk/weights/lora.zig");
const image_pixels = @import("../../sdk/media/image_pixels.zig");
const video_out = @import("../../sdk/media/video_out.zig");
const frame_in = @import("../../sdk/media/frame_in.zig");
const wav = @import("../../sdk/media/wav.zig");
const h3 = @import("h3.zig");
const vsa = @import("vsa.zig");
pub const config = @import("config.zig");

pub const plugin = config.plugin;

pub const table = abi.MediaTable{
    .name = abi.Str.of(plugin.name),
    .claims = config.claims,
    .required_file = abi.Str.of(config.required_file),
    .modality = .video,
    .caps = &config.caps,
    .bill_load = &config.billLoad,
    .bill_request = &config.billRequest,
    .load = &load,
    .unload = &unload,
    .set_loras = &setLoras,
    .step = &step,
    .encode = &encode,
};

const gpa = std.heap.c_allocator;
const Verdict = abi.Verdict;
const A = mlx.mlx_array;

const Engine = struct {
    model_dir: []u8,
    stream: mlx.mlx_stream,
    owns_stream: bool,
    machine: abi.Machine,
    geometry: h3.Geometry,
    /// The REF2VA partition (the pack's declared tasks; its files are the FL2VA ones).
    supports_refs: bool,
    /// A pack declaring ref2va without t2va (FastH3 Omni) needs an image or video reference on every request.
    supports_t2va: bool,
    pdd: ?h3.Pdd,
    vsa: ?vsa.Contract,
    /// The text encoder and DiT kept loaded between requests while memory allows.
    resident: h3.Resident,
    /// The request's style LoRAs (`set_loras`), stacked with Turbo inside `generate`.
    lora_paths: [lora.MAX_LORAS][]u8 = undefined,
    lora_scales: [lora.MAX_LORAS]f32 = undefined,
    lora_n: usize = 0,
    result: []u8 = &.{},

    fn freeLoras(e: *Engine) void {
        for (e.lora_paths[0..e.lora_n]) |p| gpa.free(p);
        e.lora_n = 0;
    }

    fn path(e: *const Engine, comptime name: []const u8) ![]u8 {
        return std.fmt.allocPrint(gpa, "{s}/" ++ name, .{e.model_dir});
    }

};

fn load(ctx: *const abi.LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque {
    var buf: [1 << 20]u8 = undefined;
    var why: Verdict = .{};
    const cfg = ctx.dir.readFile("config.json", &buf, &why) orelse "{}";
    const fail = struct {
        fn named(o: *Verdict, err: anyerror) ?*anyopaque {
            o.* = Verdict.named(.failed, @errorName(err));
            return null;
        }
    }.named;
    const geometry = config.geometry(gpa, cfg) catch |err| return fail(out, err);
    const pdd = h3.Pdd.fromConfig(gpa, cfg) catch |err| return fail(out, err);
    const contract = vsa.Contract.fromConfig(gpa, cfg) catch |err| return fail(out, err);
    const e = gpa.create(Engine) catch return fail(out, error.OutOfMemory);
    const owns = ctx.stream.ctx == null;
    e.* = .{
        .model_dir = gpa.dupe(u8, ctx.model_dir.slice()) catch {
            gpa.destroy(e);
            return fail(out, error.OutOfMemory);
        },
        .stream = if (owns) mlx.mlx_default_gpu_stream_new() else .{ .ctx = ctx.stream.ctx },
        .owns_stream = owns,
        .machine = if (ctx.machine) |m| m.* else .{},
        .geometry = geometry,
        .supports_refs = config.declaresTask(gpa, cfg, "ref2va"),
        .supports_t2va = config.declaresTask(gpa, cfg, "t2va"),
        .pdd = pdd,
        .vsa = contract,
        .resident = h3.Resident.init(gpa),
    };
    h3.registerResident(&e.resident);
    return e;
}

fn unload(ep: *anyopaque) callconv(.c) void {
    const e: *Engine = @ptrCast(@alignCast(ep));
    h3.unregisterResident(&e.resident);
    _ = e.resident.release();
    e.freeLoras();
    gpa.free(e.result);
    gpa.free(e.model_dir);
    if (e.owns_stream) _ = mlx.mlx_stream_free(e.stream);
    gpa.destroy(e);
}

/// The request's adapters, proven HERE: the DiT loads per request, so `loadFile`'s own check would be reached minutes
/// into a generation. They stack with Turbo, which `generate` puts first.
fn setLoras(ep: *anyopaque, loras: abi.Span(abi.Lora)) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    e.freeLoras();
    for (loras.slice()) |l| {
        lora.validatePath(l.path.slice()) catch return Verdict.refuse(.unsupported, "'lora_paths' must be absolute paths to readable .safetensors files");
        e.lora_paths[e.lora_n] = gpa.dupe(u8, l.path.slice()) catch return Verdict.refuse(.busy, "out of memory");
        e.lora_scales[e.lora_n] = l.scale;
        e.lora_n += 1;
    }
    return .ok;
}

fn bad(comptime why: []const u8) Verdict {
    return Verdict.refuse(.unsupported, why);
}

/// `default` when omitted, else held to [lo, hi] before any cast.
fn stepsIn(b: *const job.Body, default: u32, lo: u32, hi: u32) ?u32 {
    const v = b.get("steps") orelse return default;
    const n: i64 = switch (v) {
        .integer => |i| i,
        .null => return default,
        else => return null,
    };
    return if (n < lo or n > hi) null else @intCast(n);
}

fn u32Field(b: *const job.Body, key: []const u8, default: u32) u32 {
    return @intCast(@min(b.uint(key) orelse default, std.math.maxInt(u32)));
}

/// `key` holds a value (absent and null are unset).
fn fieldSet(b: *const job.Body, key: []const u8) bool {
    const v = b.get(key) orelse return false;
    return v != .null;
}

fn pm1Frame(bytes: []const u8, h: u32, w: u32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ?A {
    return frame_in.pm1(gpa, bytes, h, w, dt, s);
}

const Keyframes = struct {
    buf: [2]h3.Keyframe = undefined,
    n: usize = 0,

    fn deinit(k: *Keyframes) void {
        for (k.buf[0..k.n]) |kf| {
            _ = mlx.mlx_array_free(kf.pixels);
        }
    }
};

/// FL2VA keyframes: first = the geometry anchor (bf16), last = f32, both cover-cropped to the canvas. NOT graceful:
/// the user asked for THIS frame.
fn keyframes(b: *const job.Body, width: u32, height: u32, s: mlx.mlx_stream, out: *Keyframes) ?Verdict {
    inline for (.{ .{ "first_frame_image", h3.KeyframeAnchor.first, mlx.mlx_dtype.bfloat16 }, .{ "last_frame_image", h3.KeyframeAnchor.last, mlx.mlx_dtype.float32 } }) |k| {
        if (b.str(k[0])) |b64| if (b64.len > 0) {
            const bytes = job.imageUpload(gpa, b64) catch return bad("keyframe image is not valid base64");
            defer gpa.free(bytes);
            const undecodable = bad("keyframe image could not be decoded (PNG/JPEG expected)");
            const pixels = pm1Frame(bytes, height, width, k[2], s) orelse return undecodable;
            out.buf[out.n] = .{ .anchor = k[1], .pixels = pixels };
            out.n += 1;
            log.info("[video] minimax-h3 {s} keyframe conditioning engaged ({d}x{d})\n", .{ @tagName(k[1]), width, height });
        };
    }
    return null;
}

/// A reference WAV as the audio VAE reads it: 32 kHz stereo interleaved.
fn wav32kStereo(b64: []const u8) ![]f32 {
    const bytes = try job.base64Decode(gpa, b64);
    defer gpa.free(bytes);
    const dec = try wav.decode(gpa, bytes);
    defer gpa.free(dec.pcm);
    const stereo = try wav.toStereoInterleaved(gpa, dec.pcm, dec.channels);
    defer gpa.free(stereo);
    return wav.resampleLinear(gpa, stereo, 2, dec.sample_rate, @import("audio.zig").SAMPLE_RATE);
}

/// `[2, L]` f32 from interleaved stereo: the channels ride the batch axis.
fn planar(pcm: []const f32) !A {
    const frames = pcm.len / 2;
    const p = try gpa.alloc(f32, pcm.len);
    defer gpa.free(p);
    for (0..frames) |i| {
        p[i] = pcm[i * 2];
        p[frames + i] = pcm[i * 2 + 1];
    }
    return mlx.mlx_array_new_data(p.ptr, &[_]c_int{ 2, @intCast(frames) }, 2, .float32);
}

/// Every `stride`-th frame at `th`x`tw`: `[1,3,T,H,W]` for the VAE (`cat_axis` 2) or `[n,3,H,W]` for the tower (0).
fn videoFrames(frames: []const []u8, th: u32, tw: u32, stride: u32, cat_axis: c_int, s: mlx.mlx_stream) !?A {
    var g = ops.Graph.init(gpa, s);
    defer g.deinit();
    var parts: std.ArrayList(A) = .empty;
    defer parts.deinit(gpa);
    var i: usize = 0;
    while (i < frames.len) : (i += stride) {
        const f = g.adopt(pm1Frame(frames[i], th, tw, .bfloat16, s) orelse return null) catch return null;
        try parts.append(gpa, if (cat_axis == 2) f else try g.reshape(f, &.{ 1, 3, @intCast(th), @intCast(tw) }));
    }
    if (parts.items.len == 0) return null;
    return g.escape(try g.concat(parts.items, cat_axis));
}

/// The REF2VA references decoded into `out` in `resolveRefs` order; a reference the server cannot use is a named
/// refusal, never dropped (the generation would succeed and ignore what the user asked it to follow).
fn references(req: *const abi.Request, b: *const job.Body, gen_w: u32, gen_h: u32, gen_frames: u32, s: mlx.mlx_stream, out: *std.ArrayList(h3.RefMedia)) !?Verdict {
    const mode: h3.RefImageSizing = blk: {
        const raw = b.str("ref_image_size") orelse break :blk .match;
        if (std.mem.eql(u8, raw, "match")) break :blk .match;
        if (std.mem.eql(u8, raw, "max")) break :blk .max;
        return bad("'ref_image_size' must be \"match\" or \"max\"");
    };
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var images: std.ArrayList([]u8) = .empty;
    var videos: std.ArrayList([][]u8) = .empty;
    var tracks: std.ArrayList(?[]f32) = .empty;
    var audios: std.ArrayList([]f32) = .empty;
    defer for (tracks.items) |t| if (t) |x| gpa.free(x);
    defer for (audios.items) |x| gpa.free(x);

    if (b.get("ref_images")) |v| if (v == .array) for (v.array.items, 0..) |item, i| {
        if (item != .string) return bad("'ref_images' must be an array of base64 PNG/JPEG strings");
        const bytes = job.imageUpload(a, item.string) catch return req.refuseFmt(.unsupported, "a 'ref_images' entry is not valid base64", "'ref_images'[{d}] is not valid base64", .{i});
        try images.append(a, bytes);
    };
    if (b.get("ref_videos")) |v| if (v == .array) for (v.array.items, 0..) |item, vi| {
        if (item != .object) return bad("'ref_videos' must be an array of {\"frames\":[…],\"audio\":\"…\"} objects");
        const fr = item.object.get("frames") orelse return req.refuseFmt(.unsupported, "a 'ref_videos' entry needs a 'frames' array", "'ref_videos'[{d}] needs a 'frames' array of base64 PNG/JPEG strings", .{vi});
        if (fr != .array) return req.refuseFmt(.unsupported, "a 'ref_videos' entry needs a 'frames' array", "'ref_videos'[{d}] needs a 'frames' array of base64 PNG/JPEG strings", .{vi});
        const frames = try a.alloc([]u8, fr.array.items.len);
        for (fr.array.items, frames, 0..) |f, *dst, fi| {
            if (f != .string) return req.refuseFmt(.unsupported, "a 'ref_videos' frames array is malformed", "'ref_videos'[{d}].frames must be an array of base64 PNG/JPEG strings", .{vi});
            dst.* = job.imageUpload(a, f.string) catch return req.refuseFmt(.unsupported, "a 'ref_videos' frame is not valid base64", "'ref_videos'[{d}].frames[{d}] is not valid base64", .{ vi, fi });
        }
        try videos.append(a, frames);
        var track: ?[]f32 = null;
        if (item.object.get("audio")) |au| if (au == .string and au.string.len > 0) {
            track = wav32kStereo(au.string) catch return req.refuseFmt(.unsupported, "a 'ref_videos' soundtrack could not be decoded", "'ref_videos'[{d}].audio must be a PCM16/PCM24/float32 WAV", .{vi});
        };
        try tracks.append(a, track);
    };
    if (b.get("ref_audios")) |v| if (v == .array) for (v.array.items, 0..) |item, i| {
        if (item != .string) return bad("'ref_audios' must be an array of base64 WAV strings");
        const pcm = wav32kStereo(item.string) catch return req.refuseFmt(.unsupported, "a 'ref_audios' entry could not be decoded", "'ref_audios'[{d}] must be a PCM16/PCM24/float32 WAV", .{i});
        try audios.append(a, pcm);
    };
    if (images.items.len + videos.items.len + audios.items.len == 0) return null;

    // Source dimensions are all `resolveRefs` sizes from: an unmeasurable reference is refused here, by field.
    const in_i = try a.alloc(h3.RefInput, images.items.len);
    for (images.items, in_i, 0..) |bytes, *r, i| {
        const sz = image_pixels.nativeSize(bytes) orelse return req.refuseFmt(.unsupported, "a 'ref_images' entry could not be decoded (PNG/JPEG expected)", "could not decode 'ref_images'[{d}] (PNG/JPEG expected)", .{i});
        r.* = .{ .kind = .image, .w = sz.w, .h = sz.h };
    }
    const in_v = try a.alloc(h3.RefInput, videos.items.len);
    for (videos.items, tracks.items, in_v, 0..) |fr, track, *r, i| {
        if (fr.len == 0) return req.refuseFmt(.unsupported, "a 'ref_videos' entry has no frames", "'ref_videos'[{d}] has no frames", .{i});
        const sz = image_pixels.nativeSize(fr[0]) orelse return req.refuseFmt(.unsupported, "a 'ref_videos' frame could not be decoded (PNG/JPEG expected)", "could not decode 'ref_videos'[{d}].frames[0] (PNG/JPEG expected)", .{i});
        // A reference longer than the generation is cut before it snaps to the ladder: the output cannot reach it.
        r.* = .{ .kind = .video, .w = sz.w, .h = sz.h, .frames = @min(@as(u32, @intCast(fr.len)), gen_frames), .soundtrack_samples = if (track) |p| @intCast(p.len / 2) else null };
    }
    const in_a = try a.alloc(h3.RefInput, audios.items.len);
    for (audios.items, in_a) |pcm, *r| r.* = .{ .kind = .audio, .audio_samples = @intCast(pcm.len / 2) };

    var res = switch (try h3.resolveRefs(gpa, in_i, in_v, in_a, gen_w, gen_h, mode)) {
        .ok => |r| r,
        .reject => |why| return Verdict.named(.unsupported, why.message()),
    };
    defer res.deinit();
    // Decoded at the canvases the resolver picked: the VAE canvas for the DiT, the Qwen canvas for the tower.
    for (res.refs) |r| {
        try out.append(gpa, .{ .ref = r });
        const media = &out.items[out.items.len - 1];
        switch (r.kind) {
            .image => {
                const bytes = images.items[r.src_index];
                const vfit = h3.h3v.fitCanvas(r.canvas.h, r.canvas.w);
                const undecodable = req.refuseFmt(.unsupported, "a 'ref_images' entry could not be decoded (PNG/JPEG expected)", "could not decode 'ref_images'[{d}] (PNG/JPEG expected)", .{r.src_index});
                media.pixels = pm1Frame(bytes, r.canvas.h, r.canvas.w, .bfloat16, s) orelse return undecodable;
                media.vision = pm1Frame(bytes, vfit.h, vfit.w, .bfloat16, s) orelse return undecodable;
            },
            .video => {
                const fr = videos.items[r.src_index];
                const vfit = h3.h3v.fitCanvas(r.canvas.h, r.canvas.w);
                const undecodable = req.refuseFmt(.unsupported, "a 'ref_videos' frame could not be decoded (PNG/JPEG expected)", "could not decode a 'ref_videos'[{d}] frame (PNG/JPEG expected)", .{r.src_index});
                media.pixels = try videoFrames(fr[0..r.frames], r.canvas.h, r.canvas.w, 1, 2, s) orelse return undecodable;
                media.vision = try videoFrames(fr[0..r.frames], vfit.h, vfit.w, h3.qwenFrameStride(), 0, s) orelse return undecodable;
                if (tracks.items[r.src_index]) |pcm| media.waveform = try planar(pcm);
            },
            .audio => media.waveform = try planar(audios.items[r.src_index]),
        }
    }
    return null;
}

/// What `init_video` must match, from the request and the pack.
const InitNeeds = struct {
    width: u32,
    height: u32,
    /// The snapped frame count (17k+5) and what the request asked for.
    frame_count: u32,
    requested_frames: u32,
    audio_t: u32,
    steps: u32,
    pdd: bool,
    chain_windows: u32,
    audio_vae: bool,
};

/// The decoded `init_video`, owned until `deinit`.
const InitMedia = struct {
    pixels: ?A = null,
    waveform: ?A = null,
    step: u32 = 0,

    fn deinit(m: *InitMedia) void {
        if (m.pixels) |x| _ = mlx.mlx_array_free(x);
        if (m.waveform) |x| _ = mlx.mlx_array_free(x);
    }

    fn clip(m: *const InitMedia) ?h3.InitClip {
        const px = m.pixels orelse return null;
        return .{ .pixels = px, .waveform = m.waveform, .step = m.step };
    }
};

/// A soundtrack trimmed or zero-padded to `want` stereo frames, when it is within one audio latent frame (25 ms) of
/// it; null otherwise. Caller owns.
fn fitSoundtrack(pcm: []const f32, want: usize) !?[]f32 {
    const have = pcm.len / 2;
    const slack = @import("audio.zig").HOP_LENGTH;
    if (@max(have, want) - @min(have, want) > slack) return null;
    const out = try gpa.alloc(f32, want * 2);
    @memset(out, 0);
    const n = @min(have, want) * 2;
    @memcpy(out[0..n], pcm[0..n]);
    return out;
}

/// `init_video` + `init_step` (video-to-video) decoded into `out`; a clip the run cannot start from is a named
/// refusal. Neither field set leaves `out` empty.
fn initClip(req: *const abi.Request, b: *const job.Body, need: InitNeeds, s: mlx.mlx_stream, out: *InitMedia) !?Verdict {
    const has_clip = fieldSet(b, "init_video");
    const has_step = fieldSet(b, "init_step");
    if (!has_clip and !has_step) return null;
    if (!has_step) return bad("'init_video' needs 'init_step': the step of the schedule to start from (1 to steps-1; lower keeps less of the clip)");
    if (!has_clip) return bad("'init_step' needs 'init_video': the clip to start from ({\"frames\":[…],\"audio\":\"…\"})");
    if (need.pdd) return bad("'init_video' is not supported on a PDD student pack (FastH3 Omni): its fused heads run a fixed schedule from pure noise — load a REF2VA, FL2VA or Flash pack");
    if (need.chain_windows > 1) return bad("'init_video' cannot ride chained windows: send it with 'chain_windows' 1");

    const sv = b.get("init_step").?;
    if (need.steps < 2) return req.refuseFmt(.unsupported, "'init_step' needs a run of at least 2 steps", "'init_step' needs a run of at least 2 steps; this run has {d}", .{need.steps});
    const k: i64 = if (sv == .integer) sv.integer else return req.refuseFmt(.unsupported, "'init_step' must be an integer", "'init_step' must be an integer in [1, {d}] for this {d}-step run", .{ need.steps - 1, need.steps });
    if (k < 1 or k >= need.steps) return req.refuseFmt(.unsupported, "'init_step' is outside [1, steps-1]", "'init_step' must be in [1, {d}] for this {d}-step run; got {d}", .{ need.steps - 1, need.steps, k });

    const v = b.get("init_video").?;
    if (v != .object) return bad("'init_video' must be an object {\"frames\":[base64 PNG/JPEG…],\"audio\":\"<base64 WAV>\"}");
    const fr = v.object.get("frames") orelse return bad("'init_video' needs a 'frames' array of base64 PNG/JPEG strings");
    if (fr != .array) return bad("'init_video' needs a 'frames' array of base64 PNG/JPEG strings");
    if (fr.array.items.len != need.frame_count)
        return req.refuseFmt(.unsupported, "'init_video' has the wrong number of frames", "'init_video' has {d} frames; this request needs exactly {d} ('num_frames' {d} on the 17k+5 ladder)", .{ fr.array.items.len, need.frame_count, need.requested_frames });

    // The soundtrack first: a bad one is refused before any frame is decoded. Without one, a pack with an audio VAE
    // starts from silence, so the audio rows are noised on the same clock as the video's. What lands in `out` is
    // owned by it: the caller's `deinit` frees whatever a refusal below leaves in it.
    const want: usize = @as(usize, need.audio_t) * @import("audio.zig").HOP_LENGTH;
    const au: std.json.Value = v.object.get("audio") orelse .null;
    if (au != .null) {
        if (au != .string) return bad("'init_video'.audio must be a base64 WAV string");
        if (!need.audio_vae) return bad("'init_video'.audio needs the pack's audio VAE (audio_vae.safetensors), which this pack does not ship");
        const pcm = wav32kStereo(au.string) catch return bad("'init_video'.audio must be a PCM16/PCM24/float32 WAV");
        defer gpa.free(pcm);
        const fitted = try fitSoundtrack(pcm, want) orelse
            return req.refuseFmt(.unsupported, "'init_video'.audio is not the clip's length", "'init_video'.audio is {d} samples at 32 kHz; this clip's soundtrack is {d} (within 800)", .{ pcm.len / 2, want });
        defer gpa.free(fitted);
        if (pcm.len / 2 != want) log.info("[video] minimax-h3 init_video soundtrack fitted from {d} to {d} samples\n", .{ pcm.len / 2, want });
        out.waveform = try planar(fitted);
    } else if (need.audio_vae) {
        const silence = try gpa.alloc(f32, want * 2);
        defer gpa.free(silence);
        @memset(silence, 0);
        log.info("[video] minimax-h3 init_video: no soundtrack, treated as silence\n", .{});
        out.waveform = try planar(silence);
    }

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const frames = try a.alloc([]u8, fr.array.items.len);
    for (fr.array.items, frames, 0..) |f, *dst, i| {
        if (f != .string) return bad("'init_video'.frames must be an array of base64 PNG/JPEG strings");
        dst.* = job.imageUpload(a, f.string) catch return req.refuseFmt(.unsupported, "an 'init_video' frame is not valid base64", "'init_video'.frames[{d}] is not valid base64", .{i});
        const sz = image_pixels.nativeSize(dst.*) orelse return req.refuseFmt(.unsupported, "an 'init_video' frame could not be decoded (PNG/JPEG expected)", "could not decode 'init_video'.frames[{d}] (PNG/JPEG expected)", .{i});
        if (sz.w != need.width or sz.h != need.height)
            return req.refuseFmt(.unsupported, "an 'init_video' frame is not the request's width x height", "'init_video'.frames[{d}] is {d}x{d}; every frame must be exactly {d}x{d} (the request's width x height)", .{ i, sz.w, sz.h, need.width, need.height });
    }

    out.pixels = try videoFrames(frames, need.height, need.width, 1, 2, s) orelse
        return bad("an 'init_video' frame could not be decoded (PNG/JPEG expected)");
    out.step = @intCast(k);
    return null;
}

/// Live memory: host RAM (what the OS will grant, and the total) from the request's machine, and the GPU's
/// working-set room. Unknown host RAM keeps nothing resident and skips the staged-canvas check.
pub const Mem = struct { host_avail: u64 = 0, total: u64 = 0, gpu_limit: u64, gpu_active: u64 };

pub fn memNow(req: *const abi.Request) Mem {
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    const m = req.machine orelse &abi.Machine{};
    return .{ .host_avail = m.ram_available, .total = m.ram_total, .gpu_limit = mlx.maxRecommendedWorkingSet(), .gpu_active = active };
}

fn step(ep: *anyopaque, req: *const abi.Request, progress: *const abi.Progress) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    const dir = req.dir orelse return Verdict.refuse(.failed, "minimax_h3: the host gave no pack dir");
    var b = job.Body.parse(gpa, req.body.slice()) orelse return bad("invalid JSON body");
    defer b.deinit();
    const prompt = b.str("prompt") orelse return bad("missing 'prompt'");
    if (prompt.len == 0) return bad("empty 'prompt'");
    // `cfg_scale` / `stg_scale` / `pipeline` are ignored, not refused: the app sends them for every video backend and
    // H3 is CFG-distilled and single-pipeline, so there is nothing they could have selected.

    const turbo = b.flag("turbo");
    const pdd_steps: ?u32 = if (e.pdd) |p| p.steps() else null;
    const two_time = if (e.geometry.two_time) |*tt| tt else null;
    if (turbo and (pdd_steps != null or two_time != null)) return bad("'turbo' is the base model's few-step adapter — this pack is already a distilled student and runs its own trained steps");
    if (two_time != null and e.lora_n > 0) return bad("'lora_paths' targets the base model's blocks — a two-time pack has dropped and factored some of them");
    if (turbo and dir.fileBytes("turbo_lora.safetensors") == 0) return bad("this pack has no turbo_lora.safetensors — download minimax_h3_turbo_4step_ema_ckpt850.safetensors from hf.co/larryvrh/MiniMax-H3-Turbo-Lora (Apache-2.0) into the model folder as turbo_lora.safetensors");

    const width = u32Field(&b, "width", h3.geometry.DEFAULT_SIDE);
    const height = u32Field(&b, "height", h3.geometry.DEFAULT_SIDE);
    const steps = if (pdd_steps) |n|
        stepsIn(&b, n, n, n) orelse return req.refuseFmt(.unsupported, "this pack runs a fixed step count — omit 'steps'", "this pack is a distilled student that runs exactly {d} steps — omit 'steps' or send {d}", .{ n, n })
    else if (two_time) |tt| tts: {
        const n = stepsIn(&b, tt.default_steps, 1, h3.TwoTime.MAX_STEPS) orelse 0;
        if (tt.grid(n) == null) return req.refuseFmt(.unsupported, "this pack samples only the step counts it was trained for", "this pack samples only the step counts it was trained for ({any}) — omit 'steps' for {d}", .{ tt.steps[0..tt.n_grids], tt.default_steps });
        break :tts n;
    } else
        stepsIn(&b, if (turbo) 4 else 30, 1, 100) orelse return bad("'steps' must be in [1,100]");
    const seed = b.uint("seed") orelse 0;
    const requested_frames = u32Field(&b, "num_frames", h3.geometry.DEFAULT_FRAMES);
    if (width % 32 != 0 or height % 32 != 0) return bad("width and height must be multiples of 32");
    if (turbo and steps < 4) return bad("turbo needs at least 4 steps (the distillation's own floor, and its default)");

    // Chained windows ride FL2VA keyframe conditioning: `num_frames` is per window.
    const chain_windows = u32Field(&b, "chain_windows", 1);
    if (chain_windows < 1 or chain_windows > 6) return bad("chain_windows must be 1-6");
    if (chain_windows > 1 and e.supports_refs) return bad("chained windows ride FL2VA keyframe conditioning — the REF2VA pack cannot serve them; load an FL2VA checkpoint");

    // Snapped to the model's own ladder, and said so: a silently different length drifts the client's audio mux.
    const shape = h3.temporalShape(requested_frames);
    const delivered = h3.chainDeliveredFrames(chain_windows, shape.frame_count);
    const rgb = video_out.rgbBytes(delivered, width, height);
    if (rgb > video_out.MAX_RGB_BYTES) return req.refuseFmt(.unsupported, video_out.TOO_LARGE, video_out.TOO_LARGE_FMT, .{ delivered, width, height, rgb / (1024 * 1024), video_out.MAX_RGB_BYTES / (1024 * 1024) });

    const s = e.stream;
    var kfs: Keyframes = .{};
    defer kfs.deinit();
    if (keyframes(&b, width, height, s, &kfs)) |v| return v;

    var refs: std.ArrayList(h3.RefMedia) = .empty;
    defer {
        for (refs.items) |*m| m.deinit();
        refs.deinit(gpa);
    }
    const has_refs = fieldSet(&b, "ref_images") or fieldSet(&b, "ref_videos") or fieldSet(&b, "ref_audios");
    if (has_refs and !e.supports_refs) return bad("this MiniMax-H3 pack does not support references (it declares no 'ref2va' task) — load a REF2VA checkpoint");
    if (has_refs) if (references(req, &b, width, height, shape.frame_count, s, &refs) catch |err| return job.failure(err, "MiniMax-H3 generation failed")) |v| return v;
    if (e.supports_refs and !e.supports_t2va) {
        const visual = for (refs.items) |m| {
            if (m.ref.kind != .audio) break true;
        } else false;
        if (!visual) return bad("this pack generates from references — attach at least one image or video ('ref_images' / 'ref_videos')");
    }

    var init: InitMedia = .{};
    defer init.deinit();
    const init_t0 = std.Io.Timestamp.now(std.Io.Threaded.global_single_threaded.io(), .boot);
    if (initClip(req, &b, .{
        .width = width,
        .height = height,
        .frame_count = shape.frame_count,
        .requested_frames = requested_frames,
        .audio_t = shape.audio_t,
        .steps = steps,
        .pdd = pdd_steps != null,
        .chain_windows = chain_windows,
        .audio_vae = dir.fileBytes("audio_vae.safetensors") > 0,
    }, s, &init) catch |err| return job.failure(err, "MiniMax-H3 generation failed")) |v| return v;
    if (init.pixels != null) log.info("[video] minimax-h3 init_video decoded from the request in {d} ms (base64 + image decode)\n", .{
        @as(u64, @intCast(init_t0.untilNow(std.Io.Threaded.global_single_threaded.io(), .boot).nanoseconds)) / 1_000_000,
    });

    // The request's own activations price the resident plan; when it does not fit, the staged DiT stage is checked
    // here, before any stage loads, since a Metal OOM mid-step is uncatchable.
    const distilled = turbo or pdd_steps != null or two_time != null;
    const fast: ?bool = if (b.get("fast")) |v| (if (v == .bool) v.bool else null) else null;
    const speed = h3.resolveSpeed(fast, distilled);
    const bcast_row: u64 = if (speed.bcast_k > 0) @as(u64, e.geometry.dit.num_layers) * e.geometry.dit.hidden_size * 2 else 0;
    const rows = config.requestRows(width, height, shape.frame_count, @intCast(kfs.n));
    const activations = config.activationBytes(rows, bcast_row);
    const mem = memNow(req);
    const vae_file = dir.fileBytes("video_vae.safetensors");
    const encode_held = if (init.pixels != null) config.encodeHeldBytes(width, height, shape.frame_count) else h3.geometry.encodePassBytes();
    const plan = config.plan(.{
        .sizes = .{
            .te = dir.fileBytes("text_encoder.safetensors"),
            .dit_file = config.ditFileBytes(dir, &e.machine),
            .video_vae = vae_file,
            .audio_vae = dir.fileBytes("audio_vae.safetensors"),
            .turbo_lora = dir.fileBytes("turbo_lora.safetensors"),
        },
        // A two-time DiT's AdaLN takes a 16-wide embedding: precompute frees next to nothing of it.
        .precompute = two_time == null,
        .turbo = turbo,
        .may_keep = config.residentEnabled() and !e.machine.ane_video,
        .cache_now = e.resident.bytes,
        .host_avail = mem.host_avail,
        .total_ram = mem.total,
        .gpu_limit = mem.gpu_limit,
        .gpu_active = mem.gpu_active,
        .weights_page = mlx.cudaAvailable(),
        .activations = activations,
        .encode_held = encode_held,
        .init = init.pixels != null,
        .billed_lift = config.billedEncodeLift(vae_file, encode_held, &e.machine),
    });
    // The staged plan starts from a clean slate.
    if (plan.resident == null) _ = e.resident.release();
    if (plan.refused) |r| {
        const gb = 1024.0 * 1024.0 * 1024.0;
        return switch (r.what) {
            .encode => req.refuseFmt(.unsupported, "the 'init_video' encode does not fit in memory", "the 'init_video' encode needs ~{d:.1} GB (the video VAE and its 256-px tile passes) but {d:.1} GB is available", .{ @as(f64, @floatFromInt(r.need)) / gb, @as(f64, @floatFromInt(r.avail)) / gb }),
            .canvas => canvas: {
                const cache = bcast_row > 0;
                break :canvas req.refuseFmt(.unsupported, "the canvas does not fit in memory: use a smaller canvas or fewer frames", "{d}x{d} x {d} frames needs ~{d:.1} GB ({d:.1} GB DiT{s} + {d:.1} GB for {d} sequence rows{s}) but {d:.1} GB is available: use a smaller canvas or fewer frames{s}", .{
                    width,                                     height,                                 shape.frame_count,
                    @as(f64, @floatFromInt(r.need)) / gb,      @as(f64, @floatFromInt(r.dit)) / gb,    if (mlx.cudaAvailable()) " paged to RAM" else "",
                    @as(f64, @floatFromInt(activations)) / gb, rows,                                   if (cache) " with the fast recipe's attention cache" else "",
                    @as(f64, @floatFromInt(r.avail)) / gb,     if (cache) ", or \"turbo\", which keeps no attention cache" else "",
                });
            },
        };
    }
    if (init.pixels != null) log.info("[minimax-h3] init_video encode: pool lift {d:.1} GB beside one tile pass and the clip ({d:.1} GB)\n", .{ @as(f64, @floatFromInt(plan.lift)) / (1024.0 * 1024.0 * 1024.0), @as(f64, @floatFromInt(encode_held)) / (1024.0 * 1024.0 * 1024.0) });

    const paths: h3.GenPaths = .{
        .tokenizer_dir = e.model_dir,
        .text_encoder = e.path("text_encoder.safetensors") catch return job.failure(error.OutOfMemory, "out of memory"),
        .dit = e.path("transformer.safetensors") catch return job.failure(error.OutOfMemory, "out of memory"),
        .vae = e.path("video_vae.safetensors") catch return job.failure(error.OutOfMemory, "out of memory"),
        .audio_vae = if (dir.fileBytes("audio_vae.safetensors") > 0) e.path("audio_vae.safetensors") catch null else null,
        .turbo_lora = e.path("turbo_lora.safetensors") catch null,
    };
    defer for ([_]?[]const u8{ paths.text_encoder, paths.dit, paths.vae, paths.audio_vae, paths.turbo_lora }) |p| if (p) |x| gpa.free(x);

    var paths_c: [lora.MAX_LORAS][]const u8 = undefined;
    for (e.lora_paths[0..e.lora_n], 0..) |p, i| paths_c[i] = p;
    const io = std.Io.Threaded.global_single_threaded.io();
    var res = h3.generate(gpa, io, paths, .{
        .geometry = e.geometry,
        .resident = if (plan.resident != null) &e.resident else null,
        .resident_bytes = plan.resident orelse 0,
        .encode_lift = plan.lift,
        .prompt = prompt,
        .width = width,
        .height = height,
        .frames = requested_frames,
        .steps = steps,
        .seed = seed,
        .speed = speed,
        .turbo = turbo,
        .pdd = e.pdd,
        .vsa = e.vsa,
        .lora_paths = paths_c[0..e.lora_n],
        .lora_scales = e.lora_scales[0..e.lora_n],
        .chain_windows = chain_windows,
        .keyframes = kfs.buf[0..kfs.n],
        .refs = refs.items,
        .init = init.clip(),
    }, job.progressOf(progress), s) catch |err| return switch (err) {
        error.Cancelled => job.client_left,
        // The LoRA failures are the user's to fix, each with its own remedy.
        error.LoraNoMatch => bad("a LoRA has no modules matching MiniMax-H3's DiT — wrong architecture for this adapter?"),
        error.BadLoraPath => bad("'lora_paths' must be absolute paths to .safetensors files"),
        error.TooManyLoras => bad("too many LoRA adapters (max 8, and turbo takes one of the slots)"),
        error.TurboLoraIncomplete => bad("turbo_lora.safetensors ships modules MiniMax-H3's DiT does not have — replace it with minimax_h3_turbo_4step_ema_ckpt850.safetensors from hf.co/larryvrh/MiniMax-H3-Turbo-Lora"),
        else => job.failure(err, "MiniMax-H3 generation failed"),
    };
    defer res.deinit();
    const members = encodeResult(&res, s) catch |err| return job.failure(err, "MiniMax-H3 generation failed");
    gpa.free(e.result);
    e.result = members;
    return .ok;
}

/// [1,3,F,H,W] in [-1,1] as RGB8 frames, and the soundtrack as PCM16 when the pack has an audio VAE.
fn encodeResult(res: *const h3.GenResult, s: mlx.mlx_stream) ![]u8 {
    const rgb = try h3.pixelsToRgb8(gpa, res, s);
    defer gpa.free(rgb);
    var pcm: ?[]u8 = null;
    defer if (pcm) |p| gpa.free(p);
    if (res.audio) |wave| pcm = try h3.audioToPcm16(gpa, wave, s);
    const audio: ?video_out.Audio = if (pcm) |p| .{ .sample_rate = @import("audio.zig").SAMPLE_RATE, .channels = 2, .pcm16 = p } else null;
    return video_out.members(gpa, res.frame_count, res.height, res.width, h3.FPS, rgb, audio);
}

fn encode(ep: *anyopaque, out: *abi.Str) callconv(.c) Verdict {
    const e: *Engine = @ptrCast(@alignCast(ep));
    if (e.result.len == 0) return Verdict.refuse(.failed, "minimax_h3: no finished job to encode");
    out.* = .of(e.result);
    return .ok;
}

/// The tiny DiT the conformance packs share: 64 wide (two 32-wide heads, one block + one refiner), `out_heads`
/// stacked video output heads, and the audio stream unless `audio` is false.
pub fn fixtureDit(a: std.mem.Allocator, audio: bool, out_heads: u64) !sdk.tensors.Tensors {
    const H = 64;
    const heads = 2;
    const hd = 32;
    const ffn = 64;
    const ted = 32; // time_embed_dim
    var dit = sdk.tensors.Tensors.init(a, 101);
    try dit.rand("video_patch_proj.weight", .{}, &.{ H, 96 }, .F32, 0.1);
    try dit.rand("video_patch_proj.bias", .{}, &.{H}, .F32, 0.1);
    if (audio) {
        try dit.rand("audio_patch_proj.weight", .{}, &.{ H, 32 }, .F32, 0.1);
        try dit.rand("audio_patch_proj.bias", .{}, &.{H}, .F32, 0.1);
    }
    try dit.rand("condition_proj.weight", .{}, &.{ H, H }, .BF16, 0.1);
    try dit.rand("condition_proj.bias", .{}, &.{H}, .BF16, 0.1);
    try dit.rand("time_embedder.proj_in.weight", .{}, &.{ H, 32 }, .F32, 0.1);
    try dit.rand("time_embedder.proj_in.bias", .{}, &.{H}, .F32, 0.1);
    try dit.rand("time_embedder.proj_out.weight", .{}, &.{ ted, H }, .F32, 0.1);
    try dit.rand("time_embedder.proj_out.bias", .{}, &.{ted}, .F32, 0.1);
    try dit.values("rope.inv_freq", .{}, &.{4}, .F32, &.{ 1.0, 0.3, 0.1, 0.03 });
    inline for (.{ "blocks.0", "token_refiner.blocks.0" }) |p| {
        try dit.rand(p ++ ".attn.qkv_proj.weight", .{}, &.{ 3 * heads * hd, H }, .BF16, 0.1);
        try dit.rand(p ++ ".attn.out_proj.weight", .{}, &.{ H, heads * hd }, .BF16, 0.1);
        try dit.fill(p ++ ".attn.q_norm.weight", .{}, &.{hd}, .BF16, 1);
        try dit.fill(p ++ ".attn.k_norm.weight", .{}, &.{hd}, .BF16, 1);
        try dit.rand(p ++ ".mlp.fc1.weight", .{}, &.{ 2 * ffn, H }, .BF16, 0.1);
        try dit.rand(p ++ ".mlp.fc2.weight", .{}, &.{ H, ffn }, .BF16, 0.1);
        try dit.fill(p ++ ".norm1.weight", .{}, &.{H}, .BF16, 1);
        try dit.fill(p ++ ".norm2.weight", .{}, &.{H}, .BF16, 1);
    }
    try dit.rand("blocks.0.adaln_proj.linear.weight", .{}, &.{ 18 * H, ted }, .BF16, 0.05);
    try dit.rand("blocks.0.adaln_proj.linear.bias", .{}, &.{18 * H}, .BF16, 0.05);
    try dit.fill("token_refiner.final_norm.weight", .{}, &.{H}, .BF16, 1);
    try dit.rand("final_layer.adaln_proj.linear.weight", .{}, &.{ 2 * H, ted }, .BF16, 0.05);
    try dit.rand("final_layer.adaln_proj.linear.bias", .{}, &.{2 * H}, .BF16, 0.05);
    try dit.fill("final_layer.norm.weight", .{}, &.{H}, .BF16, 1);
    try dit.rand("final_layer.video_out.weight", .{}, &.{ 96 * out_heads, H }, .F32, 0.1);
    try dit.rand("final_layer.video_out.bias", .{}, &.{96 * out_heads}, .F32, 0.1);
    if (audio) {
        try dit.rand("final_layer.audio_out.weight", .{}, &.{ 32, H }, .F32, 0.1);
        try dit.rand("final_layer.audio_out.bias", .{}, &.{32}, .F32, 0.1);
    }
    return dit;
}

/// The tiny text encoder the conformance packs share: one Qwen3 layer at 64 over a byte tokenizer.
pub fn fixtureTextEncoder(a: std.mem.Allocator) !sdk.tensors.Tensors {
    const H = 64;
    const hd = 32;
    const vocab = 260;
    var te = sdk.tensors.Tensors.init(a, 102);
    try te.rand("model.embed_tokens.weight", .{}, &.{ vocab, H }, .BF16, 0.5);
    const tp = "model.layers.0.";
    try te.fill(tp ++ "input_layernorm.weight", .{}, &.{H}, .BF16, 1);
    try te.fill(tp ++ "post_attention_layernorm.weight", .{}, &.{H}, .BF16, 1);
    inline for (.{ .{ "self_attn.q_proj", 2 * hd, H }, .{ "self_attn.k_proj", hd, H }, .{ "self_attn.v_proj", hd, H }, .{ "self_attn.o_proj", H, 2 * hd }, .{ "mlp.gate_proj", 128, H }, .{ "mlp.up_proj", 128, H }, .{ "mlp.down_proj", H, 128 } }) |l|
        try te.rand(tp ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .BF16, 0.1);
    try te.fill(tp ++ "self_attn.q_norm.weight", .{}, &.{hd}, .BF16, 1);
    try te.fill(tp ++ "self_attn.k_norm.weight", .{}, &.{hd}, .BF16, 1);
    return te;
}

/// A tiny random H3: a one-layer DiT at 64 (two 32-wide heads, one refiner block), a one-layer text encoder at 64
/// over a byte tokenizer, a one-block video VAE decoder at 32 and no audio VAE (the clip comes back silent).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    var dit = try fixtureDit(a, true, 1);
    var te = try fixtureTextEncoder(a);
    return .{
        .files = try a.dupe(sdk.fixture.File, &.{
            .{ .name = "config.json", .bytes =
            \\{"model_type":"minimax_h3","partition":"fl2va","tasks":["t2va","fl2va"],
            \\ "transformer":{"hidden_size":64,"num_layers":1,"token_refiner_num_layers":1,"num_attention_heads":2,
            \\  "attention_head_dim":32,"ffn_hidden_size":64,"text_dim":64,"timestep_input_dim":32,
            \\  "time_embed_hidden_size":64,"time_embed_dim":32,"rope_inv_freq_len":4},
            \\ "text_encoder":{"hidden":64,"layers":1,"heads":2,"kv_heads":1,"head_dim":32,"intermediate":128},
            \\ "video_vae":{"layers":1,"heads":2,"head_dim":16}}
            },
            .{ .name = "tokenizer.json", .bytes = try sdk.byte_tokenizer.json(a, &.{}) },
            .{ .name = "transformer.safetensors", .bytes = try dit.bytes() },
            .{ .name = "text_encoder.safetensors", .bytes = try te.bytes() },
            .{ .name = "video_vae.safetensors", .bytes = try fixtureVae(a, false) },
        }),
        .body = "{\"prompt\":\"a red ball rolls\",\"width\":64,\"height\":64,\"num_frames\":5,\"steps\":2,\"seed\":3,\"fast\":false}",
    };
}

/// The fixture's video VAE: a one-block decoder at 32 and, with `encoder`, the checkpoint's own encoder (its widths are
/// fixed, so it is full size) appended after it, every same-shaped weight sharing one random draw.
fn fixtureVae(a: std.mem.Allocator, encoder: bool) ![]u8 {
    const vd = 32; // decoder width: 2 heads x 16
    var vae = sdk.tensors.Tensors.init(a, 103);
    try vae.rand("decoder.x_embedder.weight", .{}, &.{ vd, 24 }, .BF16, 0.2);
    try vae.rand("decoder.x_embedder.bias", .{}, &.{vd}, .BF16, 0.1);
    try vae.rand("decoder.register_tokens", .{}, &.{ 1, 4, vd }, .BF16, 0.2);
    try vae.fill("decoder.norm_out.weight", .{}, &.{vd}, .BF16, 1);
    try vae.fill("decoder.norm_out.bias", .{}, &.{vd}, .BF16, 0);
    try vae.rand("decoder.proj_out.weight", .{}, &.{ 3072, vd }, .BF16, 0.2);
    try vae.rand("decoder.proj_out.bias", .{}, &.{3072}, .BF16, 0.1);
    try vae.rand("decoder.mask_token", .{}, &.{ 1, 1, vd }, .BF16, 0.2);
    try vae.rand("post_quant_conv.weight", .{}, &.{ 24, 24, 1, 1, 1 }, .BF16, 0.2);
    try vae.rand("post_quant_conv.bias", .{}, &.{24}, .BF16, 0.1);
    try vae.fill("latents_mean", .{}, &.{24}, .BF16, 0);
    try vae.fill("latents_std", .{}, &.{24}, .BF16, 1);
    const vp = "decoder.transformer_blocks.0.";
    inline for (.{ .{ "attn.to_qkv", 3 * vd, vd }, .{ "attn.to_out", vd, vd }, .{ "ff.w1", 8 * vd, vd }, .{ "ff.w2", vd, 4 * vd } }) |l| {
        try vae.rand(vp ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .BF16, 0.2);
        try vae.rand(vp ++ l[0] ++ ".bias", .{}, &.{l[1]}, .BF16, 0.1);
    }
    inline for (.{ "norm1.weight", "norm2.weight", "scale1", "scale2" }) |n| try vae.fill(vp ++ n, .{}, &.{vd}, .BF16, 1);
    if (encoder) try fixtureVaeEncoder(&vae);
    return vae.bytes();
}

fn fixtureVaeEncoder(vae: *sdk.tensors.Tensors) !void {
    const V = @import("vae.zig");
    var drawn: std.ArrayList(struct { o: u64, i: u64, k: u64, name: []const u8 }) = .empty;
    const conv = struct {
        fn f(t: *sdk.tensors.Tensors, seen: anytype, name: []const u8, o: u64, i: u64, k: u64) !void {
            for (seen.items) |d| if (d.o == o and d.i == i and d.k == k) return t.alias("{s}", .{name}, d.name);
            const kd: []const u64 = if (k == 3) &.{ o, i, 3, 3, 3 } else &.{ o, i, 1, 1, 1 };
            try t.rand("{s}", .{name}, kd, .BF16, 0.02);
            try seen.append(t.a, .{ .o = o, .i = i, .k = k, .name = name });
        }
    }.f;
    const vec = struct {
        fn f(t: *sdk.tensors.Tensors, name: []const u8, n: u64, v: f32) !void {
            try t.fill("{s}", .{name}, &.{n}, .BF16, v);
        }
    }.f;
    const a = vae.a;
    try conv(vae, &drawn, "encoder.conv_in.weight", V.ENC_CH, 3, 3);
    try vec(vae, "encoder.conv_in.bias", V.ENC_CH, 0);
    for (0..V.ENC_LEVELS) |lv| {
        for (0..V.ENC_RES_BLOCKS) |bi| {
            const in_ch: u64 = if (bi == 0) V.encBlockIn(lv) else V.encBlockMid(lv);
            const mid: u64 = V.encBlockMid(lv);
            const p = try std.fmt.allocPrint(a, "encoder.down.{d}.block.{d}.", .{ lv, bi });
            try vec(vae, try std.mem.concat(a, u8, &.{ p, "norm1.weight" }), in_ch, 1);
            try vec(vae, try std.mem.concat(a, u8, &.{ p, "norm1.bias" }), in_ch, 0);
            try conv(vae, &drawn, try std.mem.concat(a, u8, &.{ p, "conv1.weight" }), mid, in_ch, 3);
            try vec(vae, try std.mem.concat(a, u8, &.{ p, "conv1.bias" }), mid, 0);
            try vec(vae, try std.mem.concat(a, u8, &.{ p, "norm2.weight" }), mid, 1);
            try vec(vae, try std.mem.concat(a, u8, &.{ p, "norm2.bias" }), mid, 0);
            try conv(vae, &drawn, try std.mem.concat(a, u8, &.{ p, "conv2.weight" }), mid, mid, 3);
            try vec(vae, try std.mem.concat(a, u8, &.{ p, "conv2.bias" }), mid, 0);
            if (in_ch != mid) {
                try conv(vae, &drawn, try std.mem.concat(a, u8, &.{ p, "nin_shortcut.weight" }), mid, in_ch, 1);
                try vec(vae, try std.mem.concat(a, u8, &.{ p, "nin_shortcut.bias" }), mid, 0);
            }
        }
        if (V.encHasDown(lv)) {
            const mid: u64 = V.encBlockMid(lv);
            try conv(vae, &drawn, try std.fmt.allocPrint(a, "encoder.down.{d}.downsample.conv.weight", .{lv}), mid, mid, 3);
            try vec(vae, try std.fmt.allocPrint(a, "encoder.down.{d}.downsample.conv.bias", .{lv}), mid, 0);
        }
    }
    const top: u64 = V.encBlockMid(V.ENC_LEVELS - 1);
    try vec(vae, "encoder.norm_out.weight", top, 1);
    try vec(vae, "encoder.norm_out.bias", top, 0);
    try conv(vae, &drawn, "encoder.conv_out.weight", 48, top, 3);
    try vec(vae, "encoder.conv_out.bias", 48, 0);
    try conv(vae, &drawn, "quant_conv.weight", 48, 48, 1);
    try vec(vae, "quant_conv.bias", 48, 0);
}

// ── tests ──

const testing = std.testing;

test "minimax_h3 plugin: a seeded clip on the tiny pack is deterministic JSON members; bad fields are refused by name" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const first = try arena.allocator().dupe(u8, (try pack.run(pack.fx.body)).ok);
    try testing.expect(std.mem.startsWith(u8, first, "\"frames\":5,\"height\":64,\"width\":64,\"fps\":24,\"format\":\"rgb8\",\"data\":\""));
    try testing.expectEqualStrings(first, (try pack.run(pack.fx.body)).ok);
    inline for (.{
        .{ "{}", "missing 'prompt'" },
        .{ "{\"prompt\":\"x\",\"width\":70}", "width and height must be multiples of 32" },
        .{ "{\"prompt\":\"x\",\"steps\":0}", "'steps' must be in [1,100]" },
        .{ "{\"prompt\":\"x\",\"chain_windows\":7}", "chain_windows must be 1-6" },
        .{ "{\"prompt\":\"x\",\"ref_images\":[\"AAAA\"]}", "this MiniMax-H3 pack does not support references (it declares no 'ref2va' task) — load a REF2VA checkpoint" },
        .{ "{\"prompt\":\"x\",\"turbo\":true}", "this pack has no turbo_lora.safetensors — download minimax_h3_turbo_4step_ema_ckpt850.safetensors from hf.co/larryvrh/MiniMax-H3-Turbo-Lora (Apache-2.0) into the model folder as turbo_lora.safetensors" },
        .{ "{\"prompt\":\"x\",\"first_frame_image\":\"!!\"}", "keyframe image is not valid base64" },
    }) |c| try testing.expectEqualStrings(c[1], (try pack.run(c[0])).refused.why.slice());
}

test "minimax_h3 plugin: a canvas past the live memory is refused before any stage loads, naming its numbers" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    const m: abi.Machine = .{ .ram_total = 1 << 30, .ram_available = 1 << 20 };
    var buf: [512]u8 = undefined;
    var len: usize = 0;
    const dir = pack.mem.packDir();
    const req: abi.Request = .{ .body = .of(pack.fx.body), .machine = &m, .dir = &dir, .message = .{ .ptr = &buf, .cap = buf.len, .len = &len } };
    const Quiet = struct {
        fn report(_: ?*anyopaque, _: abi.Str, _: u32, _: u32) callconv(.c) void {}
        fn boundary(_: ?*anyopaque) callconv(.c) bool {
            return false;
        }
    };
    const v = table.step(pack.engine, &req, &.{ .ctx = null, .report = Quiet.report, .boundary = Quiet.boundary });
    try testing.expect(v.code == .unsupported);
    try testing.expect(std.mem.startsWith(u8, buf[0..len], "64x64 x 5 frames needs ~"));
    try testing.expect(std.mem.endsWith(u8, buf[0..len], "GB is available: use a smaller canvas or fewer frames"));
}

test "minimax_h3 plugin: a reference the server cannot use is refused by name, never dropped" {
    var buf: [256]u8 = undefined;
    var len: usize = 0;
    const m: abi.Machine = .{};
    const req: abi.Request = .{ .body = .of("{}"), .machine = &m, .message = .{ .ptr = &buf, .cap = buf.len, .len = &len } };
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const cases = [_]struct { body: []const u8, needle: []const u8 }{
        .{ .body = "{\"ref_images\":[\"QUJD\",\"!!!!\"]}", .needle = "'ref_images'[1]" },
        .{ .body = "{\"ref_videos\":[{\"frames\":[\"QUJD\",\"QUJD\",\"!!!!\"]}]}", .needle = "frames[2]" },
        .{ .body = "{\"ref_audios\":[\"!!!!\"]}", .needle = "'ref_audios'[0]" },
        .{ .body = "{\"ref_videos\":[{\"frames\":[\"QUJD\"],\"audio\":\"QUJD\"}]}", .needle = "'ref_videos'[0].audio" },
        .{ .body = "{\"ref_videos\":[{\"frames\":[\"QUJD\",1]}]}", .needle = "'ref_videos'[0].frames" },
        .{ .body = "{\"ref_images\":[\"QUJD\"],\"ref_image_size\":\"huge\"}", .needle = "ref_image_size" },
        .{ .body = "{\"ref_images\":[\"QUJDRA==\"]}", .needle = "could not decode 'ref_images'[0]" },
    };
    for (cases) |c| {
        var b = job.Body.parse(testing.allocator, c.body).?;
        defer b.deinit();
        var out: std.ArrayList(h3.RefMedia) = .empty;
        defer {
            for (out.items) |*r| r.deinit();
            out.deinit(gpa);
        }
        len = 0;
        const v = (try references(&req, &b, 864, 480, 124, s, &out)) orelse return error.ExpectedRejection;
        const msg = if (len > 0) buf[0..len] else v.why.slice();
        testing.expect(std.mem.indexOf(u8, msg, c.needle) != null) catch |e| {
            std.debug.print("{s}: {s}\n", .{ c.body, msg });
            return e;
        };
    }
    // No reference fields at all is not a rejection: the feature is simply off.
    var none_b = job.Body.parse(testing.allocator, "{\"prompt\":\"x\"}").?;
    defer none_b.deinit();
    var none: std.ArrayList(h3.RefMedia) = .empty;
    defer none.deinit(gpa);
    try testing.expect((try references(&req, &none_b, 864, 480, 124, s, &none)) == null);
    try testing.expectEqual(@as(usize, 0), none.items.len);
}

test "minimax_h3 plugin: reference audio arrives as 32 kHz stereo in the encoder's planar shape" {
    // The audio VAE is mono and takes the stereo channels on the batch axis, so interleaved -> [2, L] is a
    // de-interleave, not a reshape (a reshape runs and encodes a channel-swapped chirp).
    const inter = [_]f32{ 0.0, 0.5, 0.1, 0.6, 0.2, 0.7 };
    const arr = try planar(&inter);
    defer _ = mlx.mlx_array_free(arr);
    try testing.expectEqualSlices(c_int, &.{ 2, 3 }, mlx.getShape(arr));
    try mlx.check(mlx.mlx_array_eval(arr));
    const d = mlx.mlx_array_data_float32(arr).?;
    for ([_]f32{ 0.0, 0.1, 0.2, 0.5, 0.6, 0.7 }, 0..) |v, i| try testing.expectApproxEqAbs(v, d[i], 1e-6);
    // A 16 kHz mono clip comes back at the VAE's 32 kHz in stereo: the latent-frame count is computed from the
    // rate, so a clip left at its own rate halves the reference's length.
    var mono: [1600]f32 = undefined;
    for (&mono, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 100)) / 100.0;
    const wav_bytes = try wav.encodePcm16(testing.allocator, &mono, 16000, 1);
    defer testing.allocator.free(wav_bytes);
    const b64 = try testing.allocator.alloc(u8, std.base64.standard.Encoder.calcSize(wav_bytes.len));
    defer testing.allocator.free(b64);
    _ = std.base64.standard.Encoder.encode(b64, wav_bytes);
    const pcm = try wav32kStereo(b64);
    defer gpa.free(pcm);
    try testing.expectEqual(@as(usize, 3200 * 2), pcm.len);
    // Duplicated, not summed: both channels carry the mono signal.
    try testing.expectApproxEqAbs(pcm[0], pcm[1], 1e-6);
    try testing.expectApproxEqAbs(pcm[200], pcm[201], 1e-6);
}

/// `n` base64 PNG frames at `w`x`h`, each a different gradient, as a JSON array.
fn testFrames(a: std.mem.Allocator, n: usize, w: u32, h: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.append(a, '[');
    const rgb = try a.alloc(u8, @as(usize, w) * h * 3);
    for (0..n) |f| {
        for (0..h) |y| for (0..w) |x| {
            const p = (y * w + x) * 3;
            rgb[p] = @intCast((x * 4 + f * 40) % 256);
            rgb[p + 1] = @intCast((y * 4) % 256);
            rgb[p + 2] = @intCast((x + y + f * 17) % 256);
        };
        const png = try @import("../../sdk/media/png.zig").encodeRgb(a, rgb, w, h);
        const b64 = try a.alloc(u8, std.base64.standard.Encoder.calcSize(png.len));
        _ = std.base64.standard.Encoder.encode(b64, png);
        try out.print(a, "{s}\"{s}\"", .{ if (f == 0) "" else ",", b64 });
    }
    try out.append(a, ']');
    return out.items;
}

fn testWav(a: std.mem.Allocator, samples: usize) ![]u8 {
    const pcm = try a.alloc(f32, samples * 2);
    for (pcm, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.01) * 0.3;
    const bytes = try wav.encodePcm16(a, pcm, 32000, 2);
    const b64 = try a.alloc(u8, std.base64.standard.Encoder.calcSize(bytes.len));
    _ = std.base64.standard.Encoder.encode(b64, bytes);
    return b64;
}

test "minimax_h3 plugin: init_video and init_step are refused by name unless the clip matches the run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: [512]u8 = undefined;
    var len: usize = 0;
    const m: abi.Machine = .{};
    const req: abi.Request = .{ .body = .of("{}"), .machine = &m, .message = .{ .ptr = &buf, .cap = buf.len, .len = &len } };
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const five = try testFrames(a, 5, 64, 64);
    const four = try testFrames(a, 4, 64, 64);
    const narrow = try testFrames(a, 5, 32, 64);
    const audio_t = h3.temporalShape(5).audio_t;
    const want_samples = audio_t * @import("audio.zig").HOP_LENGTH;
    const base: InitNeeds = .{ .width = 64, .height = 64, .frame_count = 5, .requested_frames = 5, .audio_t = audio_t, .steps = 2, .pdd = false, .chain_windows = 1, .audio_vae = true };
    var pdd = base;
    pdd.pdd = true;
    var chained = base;
    chained.chain_windows = 2;
    var one_step = base;
    one_step.steps = 1;
    var silent_pack = base;
    silent_pack.audio_vae = false;
    const cases = [_]struct { body: []const u8, need: InitNeeds, needle: []const u8 }{
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_video\":{{\"frames\":{s}}}}}", .{five}), .need = base, .needle = "'init_video' needs 'init_step'" },
        .{ .body = "{\"init_step\":1}", .need = base, .needle = "'init_step' needs 'init_video'" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s}}}}}", .{five}), .need = pdd, .needle = "PDD student" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s}}}}}", .{five}), .need = chained, .needle = "chained windows" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s}}}}}", .{five}), .need = one_step, .needle = "at least 2 steps; this run has 1" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":0,\"init_video\":{{\"frames\":{s}}}}}", .{five}), .need = base, .needle = "must be in [1, 1] for this 2-step run; got 0" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":2,\"init_video\":{{\"frames\":{s}}}}}", .{five}), .need = base, .needle = "got 2" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":\"1\",\"init_video\":{{\"frames\":{s}}}}}", .{five}), .need = base, .needle = "'init_step' must be an integer" },
        .{ .body = "{\"init_step\":1,\"init_video\":[\"QUJD\"]}", .need = base, .needle = "'init_video' must be an object" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s}}}}}", .{four}), .need = base, .needle = "has 4 frames; this request needs exactly 5" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s}}}}}", .{narrow}), .need = base, .needle = "'init_video'.frames[0] is 32x64; every frame must be exactly 64x64" },
        .{ .body = "{\"init_step\":1,\"init_video\":{\"frames\":[\"QUJD\",\"!!!!\",\"QUJD\",\"QUJD\",\"QUJD\"]}}", .need = base, .needle = "could not decode 'init_video'.frames[0]" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s},\"audio\":\"{s}\"}}}}", .{ five, try testWav(a, want_samples) }), .need = silent_pack, .needle = "needs the pack's audio VAE" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s},\"audio\":\"QUJD\"}}}}", .{five}), .need = base, .needle = "'init_video'.audio must be a PCM16" },
        .{ .body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s},\"audio\":\"{s}\"}}}}", .{ five, try testWav(a, want_samples / 2) }), .need = base, .needle = "this clip's soundtrack is" },
        // A bad soundtrack is refused before a bad frame is decoded.
        .{ .body = "{\"init_step\":1,\"init_video\":{\"frames\":[\"QUJD\",\"!!!!\",\"QUJD\",\"QUJD\",\"QUJD\"],\"audio\":\"QUJD\"}}", .need = base, .needle = "'init_video'.audio must be a PCM16" },
    };
    for (cases) |c| {
        var b = job.Body.parse(testing.allocator, c.body).?;
        defer b.deinit();
        var out: InitMedia = .{};
        defer out.deinit();
        len = 0;
        const v = (try initClip(&req, &b, c.need, s, &out)) orelse {
            std.debug.print("accepted: {s}\n", .{c.needle});
            return error.ExpectedRejection;
        };
        const msg = if (len > 0) buf[0..len] else v.why.slice();
        testing.expect(std.mem.indexOf(u8, msg, c.needle) != null) catch |e| {
            std.debug.print("want '{s}', got '{s}'\n", .{ c.needle, msg });
            return e;
        };
    }
    // Neither field: the feature is off.
    {
        var b = job.Body.parse(testing.allocator, "{\"prompt\":\"x\"}").?;
        defer b.deinit();
        var out: InitMedia = .{};
        defer out.deinit();
        try testing.expect((try initClip(&req, &b, base, s, &out)) == null);
        try testing.expect(out.clip() == null);
    }
    // A matching clip, its soundtrack 10 samples long: decoded at the canvas, the track fitted to the clip's length.
    {
        const body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s},\"audio\":\"{s}\"}}}}", .{ five, try testWav(a, want_samples + 10) });
        var b = job.Body.parse(testing.allocator, body).?;
        defer b.deinit();
        var out: InitMedia = .{};
        defer out.deinit();
        try testing.expect((try initClip(&req, &b, base, s, &out)) == null);
        const clip = out.clip().?;
        try testing.expectEqual(@as(u32, 1), clip.step);
        try testing.expectEqualSlices(c_int, &.{ 1, 3, 5, 64, 64 }, mlx.getShape(clip.pixels));
        try testing.expectEqualSlices(c_int, &.{ 2, @intCast(want_samples) }, mlx.getShape(clip.waveform.?));
    }
    // No soundtrack: silence of the clip's length on a pack with an audio VAE, none on a pack without one.
    {
        const body = try std.fmt.allocPrint(a, "{{\"init_step\":1,\"init_video\":{{\"frames\":{s}}}}}", .{five});
        for ([_]InitNeeds{ base, silent_pack }) |need| {
            var b = job.Body.parse(testing.allocator, body).?;
            defer b.deinit();
            var out: InitMedia = .{};
            defer out.deinit();
            try testing.expect((try initClip(&req, &b, need, s, &out)) == null);
            const clip = out.clip().?;
            if (!need.audio_vae) {
                try testing.expect(clip.waveform == null);
                continue;
            }
            const track = clip.waveform orelse return error.NoSilence;
            try testing.expectEqualSlices(c_int, &.{ 2, @intCast(want_samples) }, mlx.getShape(track));
            try mlx.check(mlx.mlx_array_eval(track));
            const data = mlx.mlx_array_data_float32(track).?[0 .. 2 * want_samples];
            for (data) |x| try testing.expectEqual(@as(f32, 0), x);
        }
    }
}

test "minimax_h3 plugin: init_step without init_video is refused through the step" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var pack = try MediaPack.open(arena.allocator());
    defer pack.close();
    try testing.expect(std.mem.startsWith(u8, (try pack.run("{\"prompt\":\"x\",\"width\":64,\"height\":64,\"num_frames\":5,\"steps\":2,\"init_step\":1}")).refused.why.slice(), "'init_step' needs 'init_video'"));
}

test "minimax_h3 plugin: a soundtrack within one latent frame is fitted to the clip, a longer miss is not" {
    const pcm = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 }; // 3 stereo frames
    const longer = (try fitSoundtrack(&pcm, 5)).?;
    defer gpa.free(longer);
    try testing.expectEqualSlices(f32, &.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0, 0, 0, 0 }, longer);
    const shorter = (try fitSoundtrack(&pcm, 2)).?;
    defer gpa.free(shorter);
    try testing.expectEqualSlices(f32, &.{ 0.1, 0.2, 0.3, 0.4 }, shorter);
    try testing.expect((try fitSoundtrack(&pcm, 3 + @import("audio.zig").HOP_LENGTH + 1)) == null);
}

/// The conformance pack with the video VAE's encoder: what an `init_video` run needs.
const EncoderPackRoot = struct {
    pub const plugin = config.plugin;
    pub const table = @import("plugin.zig").table;
    pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
        var fx = try @import("plugin.zig").conformanceFixture(a);
        const files = try a.dupe(sdk.fixture.File, fx.files);
        for (files) |*f| if (std.mem.eql(u8, f.name, "video_vae.safetensors")) {
            f.bytes = try fixtureVae(a, true);
        };
        fx.files = files;
        return fx;
    }
};

test "minimax_h3 plugin: an init_video encode that cannot fit even unlifted is refused by name before any stage loads" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var pack = try @import("../../sdk/testing/media_pack.zig").MediaPack(EncoderPackRoot).open(a);
    defer pack.close();
    const body = try std.fmt.allocPrint(a, "{{\"prompt\":\"x\",\"width\":32,\"height\":32,\"num_frames\":5,\"steps\":2,\"init_step\":1,\"init_video\":{{\"frames\":{s}}}}}", .{try testFrames(a, 5, 32, 32)});
    // 4 GB free holds the tiny DiT stage but not one 256-px tile pass.
    const m: abi.Machine = .{ .ram_total = 64 << 30, .ram_available = 4 << 30 };
    var buf: [512]u8 = undefined;
    var len: usize = 0;
    const dir = pack.mem.packDir();
    const req: abi.Request = .{ .body = .of(body), .machine = &m, .dir = &dir, .message = .{ .ptr = &buf, .cap = buf.len, .len = &len } };
    const Quiet = struct {
        fn report(_: ?*anyopaque, _: abi.Str, _: u32, _: u32) callconv(.c) void {}
        fn boundary(_: ?*anyopaque) callconv(.c) bool {
            return false;
        }
    };
    const v = table.step(pack.engine, &req, &.{ .ctx = null, .report = Quiet.report, .boundary = Quiet.boundary });
    try testing.expect(v.code == .unsupported);
    try testing.expect(std.mem.startsWith(u8, buf[0..len], "the 'init_video' encode needs ~"));
    try testing.expect(std.mem.endsWith(u8, buf[0..len], "4.0 GB is available"));
}

test "minimax_h3 plugin: an init_video run starts from the clip at init_step; a run without one is unchanged" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var plain = try MediaPack.open(a);
    defer plain.close();
    const no_init = try a.dupe(u8, (try plain.run(plain.fx.body)).ok);

    var pack = try @import("../../sdk/testing/media_pack.zig").MediaPack(EncoderPackRoot).open(a);
    defer pack.close();
    // The encoder's weights change nothing for a run that does not ask for it.
    try testing.expectEqualStrings(no_init, (try pack.run(pack.fx.body)).ok);

    // 32x32: the encoder is the checkpoint's full width, and its CPU convs at 64x64 cost gigabytes.
    const plain_32 = "{\"prompt\":\"a red ball rolls\",\"width\":32,\"height\":32,\"num_frames\":5,\"steps\":2,\"seed\":3,\"fast\":false";
    const no_init_32 = try a.dupe(u8, (try pack.run(plain_32 ++ "}")).ok);
    const body = try std.fmt.allocPrint(a, "{s},\"init_step\":1,\"init_video\":{{\"frames\":{s}}}}}", .{ plain_32, try testFrames(a, 5, 32, 32) });
    const first = switch (try pack.run(body)) {
        .ok => |o| try a.dupe(u8, o),
        .refused => |v| {
            std.debug.print("init run refused: {s}\n", .{v.why.slice()});
            return error.InitRunRefused;
        },
    };
    try testing.expect(std.mem.startsWith(u8, first, "\"frames\":5,\"height\":32,\"width\":32,\"fps\":24,\"format\":\"rgb8\",\"data\":\""));
    try testing.expectEqual(no_init_32.len, first.len);
    try testing.expect(!std.mem.eql(u8, no_init_32, first));
    try testing.expectEqualStrings(first, (try pack.run(body)).ok);
}

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
