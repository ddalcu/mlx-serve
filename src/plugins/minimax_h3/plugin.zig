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

/// Live memory: host RAM (what the OS will grant, and the total) from the request's machine, and the GPU's
/// working-set room. Unknown host RAM keeps nothing resident and skips the staged-canvas check.
const Mem = struct { host_avail: u64 = 0, total: u64 = 0, gpu_limit: u64, gpu_active: u64 };

fn memNow(req: *const abi.Request) Mem {
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    const m = req.machine orelse &abi.Machine{};
    return .{ .host_avail = m.ram_available, .total = m.ram_total, .gpu_limit = mlx.maxRecommendedWorkingSet(), .gpu_active = active };
}

/// The resident cache for this request, or null for the staged plan (whatever was held is released first, so the
/// staged peak starts from a clean slate). File sizes are read per request: a file can land while the server is up.
fn residentFor(e: *Engine, dir: *const abi.PackDir, activations: u64, mem: Mem) ?struct { r: *h3.Resident, bytes: u64 } {
    const need = config.residentBytes(dir.fileBytes("text_encoder.safetensors"), config.ditFileBytes(dir, &e.machine), dir.fileBytes("video_vae.safetensors") + dir.fileBytes("audio_vae.safetensors"), dir.fileBytes("turbo_lora.safetensors"), activations);
    const avail = config.availBytes(mem.host_avail, mem.gpu_limit, mem.gpu_active);
    if (!(config.residentEnabled() and !e.machine.ane_video and config.keepResident(avail, e.resident.bytes, need, config.residentMargin(mem.total)))) {
        _ = e.resident.release();
        return null;
    }
    return .{ .r = &e.resident, .bytes = need };
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
    if (turbo and pdd_steps != null) return bad("'turbo' is the base model's few-step adapter — this pack is already a distilled student and runs its own trained steps");
    if (turbo and dir.fileBytes("turbo_lora.safetensors") == 0) return bad("this pack has no turbo_lora.safetensors — download minimax_h3_turbo_4step_ema_ckpt850.safetensors from hf.co/larryvrh/MiniMax-H3-Turbo-Lora (Apache-2.0) into the model folder as turbo_lora.safetensors");

    const width = u32Field(&b, "width", 256);
    const height = u32Field(&b, "height", 256);
    const steps = if (pdd_steps) |n|
        stepsIn(&b, n, n, n) orelse return req.refuseFmt(.unsupported, "this pack runs a fixed step count — omit 'steps'", "this pack is a distilled student that runs exactly {d} steps — omit 'steps' or send {d}", .{ n, n })
    else
        stepsIn(&b, if (turbo) 4 else 30, 1, 100) orelse return bad("'steps' must be in [1,100]");
    const seed = b.uint("seed") orelse 0;
    const requested_frames = u32Field(&b, "num_frames", 56);
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

    // The request's own activations price the resident plan; when it does not fit, the staged DiT stage is checked
    // here, before any stage loads, since a Metal OOM mid-step is uncatchable.
    const distilled = turbo or pdd_steps != null;
    const fast: ?bool = if (b.get("fast")) |v| (if (v == .bool) v.bool else null) else null;
    const speed = h3.resolveSpeed(fast, distilled);
    const bcast_row: u64 = if (speed.bcast_k > 0) @as(u64, e.geometry.dit.num_layers) * e.geometry.dit.hidden_size * 2 else 0;
    const rows = config.requestRows(width, height, shape.frame_count, @intCast(kfs.n));
    const activations = config.activationBytes(rows, bcast_row);
    const mem = memNow(req);
    const plan = residentFor(e, dir, activations, mem);
    if (plan == null) {
        const dit = config.ditResidentBytes(config.ditFileBytes(dir, &e.machine), true) + (if (turbo) dir.fileBytes("turbo_lora.safetensors") else 0);
        const bill = config.canvasBill(dit, activations, mem.host_avail, mem.gpu_limit, mem.gpu_active, mlx.cudaAvailable());
        if (dit > 0 and bill.avail > 0 and bill.need > bill.avail) {
            const gb = 1024.0 * 1024.0 * 1024.0;
            const cache = bcast_row > 0;
            return req.refuseFmt(.unsupported, "the canvas does not fit in memory: use a smaller canvas or fewer frames", "{d}x{d} x {d} frames needs ~{d:.1} GB ({d:.1} GB DiT{s} + {d:.1} GB for {d} sequence rows{s}) but {d:.1} GB is available: use a smaller canvas or fewer frames{s}", .{
                width,                                              height,                                    shape.frame_count,
                @as(f64, @floatFromInt(bill.need)) / gb,            @as(f64, @floatFromInt(dit)) / gb,         if (mlx.cudaAvailable()) " paged to RAM" else "",
                @as(f64, @floatFromInt(activations)) / gb,          rows,                                      if (cache) " with the fast recipe's attention cache" else "",
                @as(f64, @floatFromInt(bill.avail)) / gb,           if (cache) ", or \"turbo\", which keeps no attention cache" else "",
            });
        }
    }

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
        .resident = if (plan) |p| p.r else null,
        .resident_bytes = if (plan) |p| p.bytes else 0,
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

/// A tiny random H3: a one-layer DiT at 64 (two 32-wide heads, one refiner block), a one-layer text encoder at 64
/// over a byte tokenizer, a one-block video VAE decoder at 32 and no audio VAE (the clip comes back silent).
pub fn conformanceFixture(a: std.mem.Allocator) !sdk.fixture.Fixture {
    const H = 64;
    const heads = 2;
    const hd = 32;
    const ffn = 64;
    const ted = 32; // time_embed_dim
    const vocab = 260;
    var dit = sdk.tensors.Tensors.init(a, 101);
    try dit.rand("video_patch_proj.weight", .{}, &.{ H, 96 }, .F32, 0.1);
    try dit.rand("video_patch_proj.bias", .{}, &.{H}, .F32, 0.1);
    try dit.rand("audio_patch_proj.weight", .{}, &.{ H, 32 }, .F32, 0.1);
    try dit.rand("audio_patch_proj.bias", .{}, &.{H}, .F32, 0.1);
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
    try dit.rand("final_layer.video_out.weight", .{}, &.{ 96, H }, .F32, 0.1);
    try dit.rand("final_layer.video_out.bias", .{}, &.{96}, .F32, 0.1);
    try dit.rand("final_layer.audio_out.weight", .{}, &.{ 32, H }, .F32, 0.1);
    try dit.rand("final_layer.audio_out.bias", .{}, &.{32}, .F32, 0.1);

    var te = sdk.tensors.Tensors.init(a, 102);
    try te.rand("model.embed_tokens.weight", .{}, &.{ vocab, H }, .BF16, 0.5);
    const tp = "model.layers.0.";
    try te.fill(tp ++ "input_layernorm.weight", .{}, &.{H}, .BF16, 1);
    try te.fill(tp ++ "post_attention_layernorm.weight", .{}, &.{H}, .BF16, 1);
    inline for (.{ .{ "self_attn.q_proj", 2 * hd, H }, .{ "self_attn.k_proj", hd, H }, .{ "self_attn.v_proj", hd, H }, .{ "self_attn.o_proj", H, 2 * hd }, .{ "mlp.gate_proj", 128, H }, .{ "mlp.up_proj", 128, H }, .{ "mlp.down_proj", H, 128 } }) |l|
        try te.rand(tp ++ l[0] ++ ".weight", .{}, &.{ l[1], l[2] }, .BF16, 0.1);
    try te.fill(tp ++ "self_attn.q_norm.weight", .{}, &.{hd}, .BF16, 1);
    try te.fill(tp ++ "self_attn.k_norm.weight", .{}, &.{hd}, .BF16, 1);

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
            .{ .name = "video_vae.safetensors", .bytes = try vae.bytes() },
        }),
        .body = "{\"prompt\":\"a red ball rolls\",\"width\":64,\"height\":64,\"num_frames\":5,\"steps\":2,\"seed\":3,\"fast\":false}",
    };
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

const MediaPack = @import("../../sdk/testing/media_pack.zig").MediaPack(@This());
