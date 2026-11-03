//! The media plugin table (`MediaTable`): image, audio, video and mesh backends, and the per-request decision and
//! diarization jobs. Same rules as `plugin.zig`: extern layouts, C calling convention, `size` first, no globals.

const Host = @import("host.zig").Host;
const std = @import("std");
const common = @import("common.zig");

pub const Span = common.Span;
pub const Str = common.Str;
pub const Verdict = common.Verdict;
pub const Stream = common.Stream;
pub const Claim = common.Claim;
pub const PackDir = common.PackDir;

pub const version: u32 = 1;

pub const Modality = enum(u8) { image, audio, video, mesh, decision, diarization };

/// The audio routes a pack serves.
pub const Routes = extern struct {
    size: u32 = @sizeOf(Routes),
    speech: bool = false,
    music: bool = false,
    sound: bool = false,
};

/// What one pack supports. Read from the pack (a distilled and a base checkpoint of one family differ), no MLX.
pub const MediaCaps = extern struct {
    size: u32 = @sizeOf(MediaCaps),
    img2img: bool = false,
    edit: bool = false,
    /// The edit takes the upload's bytes as sent (no decode by the host).
    edit_raw_bytes: bool = false,
    /// Real CFG: `guidance_scale` and `negative_prompt`.
    guidance: bool = false,
    cond_weights: bool = false,
    multiview: bool = false,
    default_guidance: f32 = 0,
    /// The largest side in pixels; 0 = no cap.
    max_dim: u32 = 0,
    /// Reference images an edit takes; 0 = edits take none.
    edit_ref_cap: u32 = 0,
    /// The LoRA key family the pack's adapters are written for; empty = no LoRA.
    lora_arch: Str = .{},
    routes: Routes = .{},
};

/// The machine and launch facts a bill reads (the host fills them; a bill never reaches MLX).
pub const Machine = extern struct {
    size: u32 = @sizeOf(Machine),
    /// Metal's recommended working set in bytes; 0 = unknown.
    gpu_working_set: u64 = 0,
    /// `--w8a8`: media DiT linears load as int8 copies that replace the packed weights.
    w8a8: bool = false,
    /// `--ane-image` / `--ane-video` / `--ane-audio`: that modality's DiT stays packed for the ANE seam.
    ane_image: bool = false,
    ane_video: bool = false,
    ane_audio: bool = false,
    /// iOS, `MLXSERVE_LOWMEM=1`, or RAM at most 17 GiB: stage the large encoders per request.
    low_mem: bool = false,
    /// Host RAM in bytes, total and what the OS will grant now (free + file-backed + purgeable); 0 = unknown.
    ram_total: u64 = 0,
    ram_available: u64 = 0,
};

/// Valid during `load` only.
pub const LoadCtx = extern struct {
    size: u32 = @sizeOf(LoadCtx),
    model_dir: Str,
    dir: *const PackDir,
    settings_json: Str = .{},
    stream: Stream = .{},
    /// The host table (P2 step 8); opaque until it lands.
    host: ?*const Host = null,
    machine: ?*const Machine = null,
};

pub const Lora = extern struct {
    size: u32 = @sizeOf(Lora),
    path: Str,
    scale: f32 = 1,
};

/// Host memory a plugin writes a refusal text into when the text is composed at run time (a limit's number, the
/// field value it names). The host answers with it instead of `Verdict.why` when `len.*` is non-zero.
pub const Message = extern struct {
    size: u32 = @sizeOf(Message),
    ptr: ?[*]u8 = null,
    cap: usize = 0,
    len: ?*usize = null,
};

/// One job. `inputs` by modality, in body order: image = `image` then each `ref_images` entry, mesh = `image` or each
/// `images` entry (base64-decoded; a WebP arrives as a PNG of the same pixels), diarization = the clip as f32 mono
/// PCM at 16 kHz (native byte order); audio, video and decision jobs read their uploads from `body`.
pub const Request = extern struct {
    size: u32 = @sizeOf(Request),
    body: Str,
    inputs: Span(Str) = .{},
    /// The id the request resolved to, for responses that echo it.
    model_id: Str = .{},
    message: Message = .{},
    /// The machine facts read when this step began (live RAM included); null outside the host.
    machine: ?*const Machine = null,
    /// The pack, valid for the engine's lifetime (the job reads files by name instead of re-opening the path).
    dir: ?*const PackDir = null,

    /// A refusal whose text is composed now: formatted into the host's `message` (truncated to fit), with `fallback`
    /// as the static `why` for a host that gave no buffer.
    pub fn refuseFmt(self: *const Request, code: Verdict.Code, comptime fallback: []const u8, comptime fmt: []const u8, args: anytype) Verdict {
        const m = self.message;
        if (m.ptr) |p| if (m.len) |len| {
            var w = std.Io.Writer.fixed(p[0..m.cap]);
            var n: usize = undefined;
            if (w.print(fmt, args)) |_| n = w.end else |_| {
                // Cut to fit, never inside a UTF-8 sequence.
                n = w.end;
                var k: usize = 0;
                while (k < 3 and k < n and (p[n - 1 - k] & 0xC0) == 0x80) k += 1;
                if (k < n and (std.unicode.utf8ByteSequenceLength(p[n - 1 - k]) catch 1) > k + 1) n -= k + 1;
            }
            len.* = n;
        };
        return Verdict.refuse(code, fallback);
    }
};

/// The client's per-step preview request (#208): off unless it asked and the response streams.
pub const PreviewOpts = extern struct {
    size: u32 = @sizeOf(PreviewOpts),
    enabled: bool = false,
    /// Temporal slices in one filmstrip (1..8).
    frames: u32 = 1,
    /// Longest side in pixels; 0 = the latent's own size.
    max_side: u32 = 256,
};

/// One encoded preview, valid during the `preview` call.
pub const PreviewFrame = extern struct {
    size: u32 = @sizeOf(PreviewFrame),
    bytes: Str,
    mime: Str = Str.of("image/jpeg"),
    w: u32,
    h: u32,
};

/// The host's callbacks into a running job, on the GPU thread.
pub const Progress = extern struct {
    size: u32 = @sizeOf(Progress),
    ctx: ?*anyopaque,
    /// A step's progress; an empty `stage` only begins the response (a streamed one reports later failures as events).
    report: *const fn (ctx: ?*anyopaque, stage: Str, step: u32, total: u32) callconv(.c) void,
    /// Between steps: the host serves chat for its share of the step, then answers whether the client left (stop).
    boundary: *const fn (ctx: ?*anyopaque) callconv(.c) bool,
    preview_opts: PreviewOpts = .{},
    /// A step's progress with its preview; null (or `preview_opts.enabled` false) = previews are not wanted.
    preview: ?*const fn (ctx: ?*anyopaque, stage: Str, step: u32, total: u32, frame: *const PreviewFrame) callconv(.c) void = null,
};

/// A media plugin's table, exported as `exportName(name, version)`. `claims`, `caps` and the bills never reach MLX.
pub const MediaTable = extern struct {
    size: u32 = @sizeOf(MediaTable),
    version: u32 = version,
    name: Str,
    /// `model_type` is empty for a configless pack (classified from its index files).
    claims: *const fn (model_type: Str, dir: *const PackDir) callconv(.c) Claim,
    /// The file whose presence makes a claimed pack complete; until it lands the pack is invisible. Empty: no marker,
    /// the pack is complete as discovered.
    required_file: Str,
    modality: Modality,
    caps: *const fn (dir: *const PackDir, out: *MediaCaps) callconv(.c) void,
    /// The load peak in bytes.
    bill_load: *const fn (dir: *const PackDir, m: *const Machine) callconv(.c) u64,
    /// One request's whole peak from its body, as the scheduler reserves it (`GenRequest.peak_bytes`); 0 = the load
    /// bill covers it. A request past what the GPU can take is the plugin's to refuse by name, inside `step`.
    bill_request: *const fn (dir: *const PackDir, m: *const Machine, body: Str) callconv(.c) u64,
    load: *const fn (ctx: *const LoadCtx, out: *Verdict) callconv(.c) ?*anyopaque,
    unload: *const fn (e: *anyopaque) callconv(.c) void,
    /// Installs the stack for the next job (summed, never merged); an empty span clears it.
    set_loras: ?*const fn (e: *anyopaque, loras: Span(Lora)) callconv(.c) Verdict = null,
    /// Runs one job to its result, calling `progress` at every step boundary.
    step: *const fn (e: *anyopaque, req: *const Request, progress: *const Progress) callconv(.c) Verdict,
    /// The last job's result, in memory the plugin owns until its next `step` or `unload`: image = PNG, audio = WAV,
    /// mesh = GLB, video = the response's JSON members (`"frames":…,"data":"…"`, no braces), decision = the response
    /// JSON, diarization = the speaker count (u32 little-endian) then f32 probabilities [frames][speakers].
    encode: *const fn (e: *anyopaque, out: *Str) callconv(.c) Verdict,
    /// Several queued jobs in one pass (cross-request merging): `out[i]` is job i's verdict, `results[i]` its result
    /// (as `encode`'s). Null: the host runs `step` and `encode` per job.
    step_batch: ?*const fn (e: *anyopaque, reqs: Span(Request), out: [*]Verdict, results: [*]Str, progress: *const Progress) callconv(.c) void = null,
    /// The last job's extra response members (`,"key":value`, valid as `encode`'s result) that a streamed audio
    /// result's `complete` event carries beside the WAV; null or empty = none.
    extras: ?*const fn (e: *anyopaque, out: *Str) callconv(.c) void = null,
};

// ── tests ──

const testing = std.testing;

test "abi media: every struct opens with its size" {
    inline for (.{ Routes, MediaCaps, Machine, Message, PreviewOpts }) |T| {
        const v: T = .{};
        try testing.expectEqual(@as(u32, @sizeOf(T)), v.size);
    }
    inline for (.{ Routes, MediaCaps, Machine, LoadCtx, Lora, Message, Request, PreviewOpts, PreviewFrame, Progress, MediaTable }) |T|
        try testing.expectEqual(@as(usize, 0), @offsetOf(T, "size"));
}

test "abi media: a composed refusal lands in the host's buffer, truncated to fit; without one the fallback stands" {
    var buf: [16]u8 = undefined;
    var len: usize = 0;
    const req: Request = .{ .body = .{}, .message = .{ .ptr = &buf, .cap = buf.len, .len = &len } };
    const v = req.refuseFmt(.unsupported, "too many", "limit {d}, got {d}", .{ 64, 65 });
    try testing.expectEqualStrings("limit 64, got 65", buf[0..len]);
    try testing.expect(v.code == .unsupported and std.mem.eql(u8, "too many", v.why.slice()));
    _ = req.refuseFmt(.unsupported, "x", "{s}", .{"a much longer text than sixteen"});
    try testing.expectEqualStrings("a much longer te", buf[0..len]);
    _ = req.refuseFmt(.unsupported, "x", "{s}", .{"fifteen bytes:!\u{e9}"});
    try testing.expectEqualStrings("fifteen bytes:!", buf[0..len]);
    const bare: Request = .{ .body = .{} };
    try testing.expectEqualStrings("too many", bare.refuseFmt(.unsupported, "too many", "limit {d}", .{1}).why.slice());
}
