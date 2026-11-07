//! Ming-3.0-tts-0.5b: the request pipeline from a task and its fields to mono f32 samples at 44.1 kHz. Long text runs
//! fragment by fragment, so GPU memory does not grow with the text. An Engine runs on the inference thread only.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
pub const model = @import("model.zig");
pub const text = @import("text.zig");
pub const audio = @import("audio.zig");
pub const campplus = @import("campplus.zig");
const config = @import("config.zig");
const log = @import("../../sdk/util/log.zig");
const progress = @import("../../sdk/media/progress.zig");
const tokenizer_mod = @import("../../sdk/tokenizer/tokenizer.zig");
const tokenizer_load = @import("../../sdk/tokenizer/load.zig");

pub const Diag = text.Diag;
pub const SAMPLE_RATE = model.SAMPLE_RATE;
const Arr = model.Arr;
const G = model.G;

pub const MAX_PROMPT_SECONDS: f32 = 20.0;
/// Request text limit (Unicode code points), about an hour of speech. The output buffer takes about 10.6 MB per minute
/// of audio.
pub const MAX_TEXT_CHARS: usize = 50_000;
pub const V2_TASKS = [_][]const u8{ "TTS", "MULTILINGUAL", "BASIC", "EMOTION", "DIALECT", "IP", "FREEFORM", "VOICE_DESIGN" };

/// One request with an explicit task. Strings are borrowed.
pub const GenerateRequest = struct {
    task: []const u8,
    text: []const u8,
    language: ?[]const u8 = null,
    prompt_wav: ?[]const u8 = null,
    prompt_text: ?[]const u8 = null,
    voice: ?[]const u8 = null,
    instruction: ?[]const u8 = null,
    controls: text.Controls = .{},
    /// Control keys the request named that `Controls` has no field for.
    unknown_controls: []const []const u8 = &.{},
    seed: ?u64 = 0,
    /// Overrides `Engine.pause_scale` for this request.
    pause_scale: ?f32 = null,
};

/// Speech-to-speech gap between consecutive fragments, in seconds, by
/// boundary (`Engine.join`). Measured from the end of one fragment's speech
/// to the start of the next one's, so it includes `LEAD_PAD_S + TAIL_PAD_S`.
pub const PAUSE_CLAUSE_S: f32 = 0.2;
pub const PAUSE_SENTENCE_S: f32 = 0.4;
pub const PAUSE_PARAGRAPH_S: f32 = 0.85;
/// Audio kept before a fragment's speech onset and after its offset at a
/// join (`audio.speechBounds`), so quiet onsets and decays are not cut.
pub const LEAD_PAD_S: f32 = 0.04;
pub const TAIL_PAD_S: f32 = 0.08;
/// Raised-cosine fade on each cut edge.
pub const FADE_S: f32 = 0.01;
/// Upper bound for `pause_scale`.
pub const MAX_PAUSE_SCALE: f32 = 4;

pub const Engine = struct {
    /// The running job's progress and cancel probe.
    prog: ?progress.Progress = null,
    /// Bring reference audio to the presets' speech level before use
    /// (`audio.normalizePrompt`). Off reproduces upstream exactly.
    normalize_prompt: bool = true,
    /// Soft-limit output above 0.9 instead of hard clipping at 1.0
    /// (`audio.softLimit`). Off reproduces upstream exactly.
    soft_limit: bool = true,
    /// Regenerate fragments with an implausible speaking rate
    /// (`generateGuarded`). Off reproduces upstream exactly.
    guard: bool = true,
    /// Join fragments with a pause chosen by the boundary between them
    /// (clause, sentence, paragraph; see PAUSE_*_S), cutting the model's own
    /// edge silence. Off concatenates fragments as upstream does.
    join: bool = true,
    /// Multiplies the boundary pauses for the next `synthesize` calls (0 to
    /// MAX_PAUSE_SCALE). The gap never goes below LEAD_PAD_S + TAIL_PAD_S.
    pause_scale: f32 = 1,
    /// Free MLX's buffer cache after each fragment of a multi-fragment text.
    /// Fragments have different lengths, so cached buffers rarely get reused
    /// and the cache otherwise grows with the text.
    clear_cache: bool = true,
    a: std.mem.Allocator,
    io: std.Io,
    s: model.S,
    model: *model.Model,
    tok: tokenizer_mod.Tokenizer,
    /// Null when the pack has no CAM++ weights: cloning is refused by name.
    speaker: ?*campplus.Campplus,
    /// Last CAM++ embedding, keyed by a hash of the 16 kHz input: a client
    /// often reuses one reference for many takes, and CAM++ costs ~90 ms on the CPU.
    spk_cache: ?struct { key: u64, emb: []f32 } = null,

    pub fn init(io: std.Io, a: std.mem.Allocator, s: model.S, model_dir: []const u8, dims: config.Dims) !Engine {
        const m = try model.Model.load(a, s, model_dir, dims);
        errdefer m.deinit();
        var tok = try tokenizer_load.loadTokenizerAny(io, a, model_dir);
        errdefer tok.deinit();
        try validateTokenizer(a, &tok);
        // CAM++ runs on the CPU stream: Metal fp32 convolutions drift ~3e-3 from onnxruntime, enough to change a bf16
        // clone's trajectory.
        const sp = try loadSpeaker(io, a, model_dir);
        return .{ .a = a, .io = io, .s = s, .model = m, .tok = tok, .speaker = sp };
    }

    /// The Hugging Face checkpoint ships campplus.onnx; a campplus.safetensors (mlx-audio's conversion) wins when present.
    fn loadSpeaker(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !?*campplus.Campplus {
        const cpu = mlx.mlx_default_cpu_stream_new();
        const st = try std.fmt.allocPrint(a, "{s}/campplus.safetensors", .{model_dir});
        defer a.free(st);
        if (std.Io.Dir.accessAbsolute(io, st, .{})) |_| {
            return try campplus.Campplus.load(a, cpu, st);
        } else |_| {}
        const onnx = try std.fmt.allocPrint(a, "{s}/campplus.onnx", .{model_dir});
        defer a.free(onnx);
        std.Io.Dir.accessAbsolute(io, onnx, .{}) catch return null;
        return try campplus.Campplus.loadOnnx(io, a, cpu, onnx);
    }

    pub fn deinit(e: *Engine) void {
        if (e.spk_cache) |c| e.a.free(c.emb);
        if (e.speaker) |sp| sp.deinit();
        e.tok.deinit();
        e.model.deinit();
    }

    /// Map a request with an explicit task to `synthesize` and run it (owned samples).
    pub fn generate(e: *Engine, r: GenerateRequest, d: *Diag) ![]f32 {
        const in = try mapGenerate(r, d);
        const saved = e.useOptions(r);
        defer e.restoreOptions(saved);
        return e.synthesize(in, r.seed);
    }

    pub const Options = struct { pause_scale: f32 };

    /// Apply a request's `pause_scale`; returns the previous value.
    pub fn useOptions(e: *Engine, r: GenerateRequest) Options {
        const saved: Options = .{ .pause_scale = e.pause_scale };
        if (r.pause_scale) |v| e.pause_scale = v;
        return saved;
    }

    pub fn restoreOptions(e: *Engine, o: Options) void {
        e.pause_scale = o.pause_scale;
    }

    /// No task, no reference: built-in voice, voice design, or the default voice.
    pub fn tts(e: *Engine, txt: []const u8, voice: ?[]const u8, instruction: ?[]const u8, seed: ?u64, d: *Diag) ![]f32 {
        var in: text.TaskInput = .{ .task = "IP", .text = txt, .use_zero_spk_emb = true };
        if (voice) |v| {
            if (!text.isVoice(v)) return d.set("unknown voice '{s}'; " ++ text.VOICE_HINT, .{v});
            in.controls.ip = v;
        } else if (instruction) |ins| {
            in.task = "FREEFORM";
            in.instruction = ins;
        } else {
            // The model has no neutral speaker; plain TTS uses a default built-in voice.
            in.controls.ip = if (isChineseUtf8(txt)) "灵小柔" else "UK_George";
        }
        return e.synthesizeChecked(in, seed, d);
    }

    /// No task, a reference: zero-shot clone from a WAV clip (+ transcript).
    pub fn clone(e: *Engine, txt: []const u8, prompt_wav: []const u8, prompt_text: ?[]const u8, seed: ?u64, d: *Diag) ![]f32 {
        // fr/de/es/ja/ko text through the zh/en path loses its accents and gets English number expansion ("Bonjour à
        // tous" becomes "Bonjour ， tous"). Route it to MULTILINGUAL, which needs the transcript.
        if (prompt_text != null) if (text.detectLanguage(txt)) |lang| {
            var md: Diag = .{};
            const in: text.TaskInput = .{ .task = "MULTILINGUAL", .text = txt, .language_id = lang, .prompt_wav = prompt_wav, .prompt_text = prompt_text, .use_spk_emb = true };
            if (text.buildRequest(e.a, in, &md)) |req| {
                req.deinit(e.a);
                return e.synthesize(in, seed);
            } else |_| {}
        };
        return e.synthesizeChecked(.{
            .task = "TTS",
            .text = txt,
            // null lets buildRequest detect fr/de/es/ja/ko.
            .language_id = if (isChineseUtf8(txt)) "Chinese" else if (text.detectLanguage(txt) != null) null else "English",
            .prompt_wav = prompt_wav,
            .prompt_text = prompt_text,
            .use_spk_emb = true,
        }, seed, d);
    }

    fn synthesizeChecked(e: *Engine, in: text.TaskInput, seed: ?u64, d: *Diag) ![]f32 {
        // Validate first so the 400 carries the upstream message.
        const req = try text.buildRequest(e.a, in, d);
        req.deinit(e.a);
        return e.synthesize(in, seed);
    }

    /// The reference's `MingTTS.synthesize` kwargs -> waveform (owned, 44.1 kHz). Call `text.buildRequest` first for the
    /// refusal's message; here a bad request is `error.BadRequest`.
    pub fn synthesize(e: *Engine, in: text.TaskInput, seed: ?u64) ![]f32 {
        const a = e.a;
        if (seed) |sd| try mlx.check(mlx.mlx_random_seed(sd));
        var d: Diag = .{};
        const req = try text.buildRequest(a, in, &d);
        defer req.deinit(a);

        var prompt_latent: ?Arr = null;
        defer if (prompt_latent) |x| {
            _ = mlx.mlx_array_free(x);
        };
        var spk: ?[]f32 = null; // 192 floats
        defer if (spk) |x| a.free(x);
        if (req.use_zero_spk_emb) {
            spk = try a.alloc(f32, model.SPK_DIM);
            @memset(spk.?, 0);
        } else if (req.prompt_wav) |bytes| {
            const loaded = try audio.decodeMono(a, bytes);
            defer a.free(loaded.pcm);
            if (loaded.pcm.len == 0) return error.EmptyPrompt;
            if (@as(f32, @floatFromInt(loaded.pcm.len)) > MAX_PROMPT_SECONDS * @as(f32, @floatFromInt(loaded.sample_rate))) return error.PromptTooLong;
            if (e.normalize_prompt) _ = audio.normalizePrompt(loaded.pcm, loaded.sample_rate);
            if (req.use_spk_emb) {
                if (e.speaker == null) return error.NoSpeakerEncoder;
                const w16 = try audio.resample(a, loaded.pcm, loaded.sample_rate, 16000);
                defer a.free(w16);
                spk = try e.speakerEmbed(w16);
            }
            const w44 = try audio.resample(a, loaded.pcm, loaded.sample_rate, SAMPLE_RATE);
            defer a.free(w44);
            prompt_latent = try e.encodePromptLatent(w44);
        }
        var prompt_text: ?[]const u8 = if (prompt_latent != null) req.prompt_text else null;
        // The reference transcript gets the same French spelling as the text, so
        // "20 secondes" in it matches the "vingt secondes" that was spoken.
        const fr_prompt_text: ?[]u8 = if (prompt_text != null and text.isFrench(req.text_language))
            try @import("fr.zig").normalize(a, prompt_text.?)
        else
            null;
        defer if (fr_prompt_text) |x| a.free(x);
        if (fr_prompt_text) |x| prompt_text = x;
        if (prompt_text == null) if (prompt_latent) |x| {
            _ = mlx.mlx_array_free(x);
            prompt_latent = null;
        };

        var frags = try text.prepareLongText(a, req.text, req.text_language, 24);
        defer text.freeFragments(a, &frags);
        var zero = req.use_zero_spk_emb;
        const reuse_first = req.taskname == .STYLE and frags.items.len > 1;
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(a);
        var reused_text: ?[]u8 = null;
        defer if (reused_text) |x| a.free(x);
        var units_total: usize = 0;
        for (frags.items) |f| units_total += text.spokenUnits(f.text);
        var units_done: usize = 0;
        e.report(0, units_total);
        const last = frags.items.len - 1;
        for (frags.items, 0..) |f, i| {
            if (e.boundary()) return error.Cancelled;
            const frag = f.text;
            const chunk = try e.generateGuarded(req, frag, spk, prompt_latent, prompt_text, zero);
            defer a.free(chunk);
            const trimmed = chunk[0..audio.trimTrailingSilence(chunk, SAMPLE_RATE).len];
            if (e.soft_limit) audio.softLimit(trimmed);
            if (reuse_first and i == 0) {
                // VOICE_DESIGN/FREEFORM: later fragments clone the first one's voice.
                const w16 = try audio.resample(a, trimmed, SAMPLE_RATE, 16000);
                defer a.free(w16);
                if (spk) |x| a.free(x);
                spk = null;
                if (e.speaker == null) return error.NoSpeakerEncoder;
                spk = try e.speaker.?.embed(a, w16);
                if (prompt_latent) |x| _ = mlx.mlx_array_free(x);
                prompt_latent = null;
                prompt_latent = try e.encodePromptLatent(trimmed);
                reused_text = try a.dupe(u8, frag);
                prompt_text = reused_text;
                zero = false;
            }
            // Joins (`join`): cut each side of a fragment boundary to its speech
            // plus a short pad, fade the cut edges, and put a pause between.
            // The start of the first and the end of the last fragment keep the
            // model's own silence, so one-fragment text is unchanged.
            // Pads shorter than LEAD_PAD_S / TAIL_PAD_S are made up with
            // silence, so the speech-to-speech gap is the boundary's pause.
            var seg = trimmed;
            const joined = e.join and last > 0;
            var tail_fill: usize = 0;
            if (joined) {
                const b = audio.speechBounds(trimmed, SAMPLE_RATE);
                if (b.end > b.start) {
                    const lo = if (i > 0) b.start -| nsamp(LEAD_PAD_S) else 0;
                    const hi = if (i < last) @min(trimmed.len, b.end + nsamp(TAIL_PAD_S)) else trimmed.len;
                    seg = trimmed[lo..hi];
                    if (i > 0) {
                        audio.fade(seg, @min(nsamp(FADE_S), b.start - lo), .in);
                        try e.emitSilence(&out, nsamp(LEAD_PAD_S) - (b.start - lo));
                    }
                    if (i < last) {
                        audio.fade(seg, nsamp(FADE_S), .out);
                        tail_fill = nsamp(TAIL_PAD_S) - (hi - b.end);
                    }
                }
            }
            try e.emit(&out, seg);
            if (joined and i < last) {
                const gap = e.pauseSeconds(f.after);
                try e.emitSilence(&out, tail_fill + nsamp(@max(0, gap - LEAD_PAD_S - TAIL_PAD_S)));
            }
            if (e.clear_cache and last > 0) _ = mlx.mlx_clear_cache();
            units_done += text.spokenUnits(frag);
            e.report(units_done, units_total);
        }
        return out.toOwnedSlice(a);
    }

    fn pauseSeconds(e: *const Engine, b: text.Boundary) f32 {
        const base: f32 = switch (b) {
            .none => 0,
            .clause => PAUSE_CLAUSE_S,
            .sentence => PAUSE_SENTENCE_S,
            .paragraph => PAUSE_PARAGRAPH_S,
        };
        return base * std.math.clamp(e.pause_scale, 0, MAX_PAUSE_SCALE);
    }

    fn boundary(e: *Engine) bool {
        const p = e.prog orelse return false;
        return p.boundary();
    }

    fn report(e: *Engine, done: usize, total: usize) void {
        if (e.prog) |p| p.emit("speech", @intCast(done), @intCast(total));
    }

    fn emit(e: *Engine, out: *std.ArrayList(f32), samples: []const f32) !void {
        try out.appendSlice(e.a, samples);
    }

    fn emitSilence(e: *Engine, out: *std.ArrayList(f32), n: usize) !void {
        try out.appendNTimes(e.a, 0, n);
    }

    /// `generateFragment` with a check for failed generations: a fragment
    /// whose speaking rate is implausible (`plausible`) is generated again
    /// with fresh noise, and then once more without the reference latents
    /// (speaker embedding only). The most plausible attempt is returned.
    /// Long, noisy references (e.g. a 17 s room recording) can make the model
    /// run to the step cap or stop after one word for most seeds, while the
    /// same reference used as speaker embedding only is fine. Departs from
    /// upstream (`guard` off reproduces it).
    fn generateGuarded(e: *Engine, req: text.Request, frag: []const u8, spk: ?[]const f32, prompt_latent: ?Arr, prompt_text: ?[]const u8, zero: bool) ![]f32 {
        const a = e.a;
        var best = try e.generateFragment(req, frag, spk, prompt_latent, prompt_text, zero);
        if (!e.guard) return best;
        const units = text.spokenUnits(frag);
        var best_score = plausibility(units, best.len);
        var attempt: usize = 1;
        while (best_score > 0 and attempt < 3) : (attempt += 1) {
            const drop_latent = attempt == 2 and prompt_text != null and spk != null;
            if (attempt == 2 and !drop_latent and prompt_text != null) break;
            const next = try e.generateFragment(req, frag, spk, if (drop_latent) null else prompt_latent, if (drop_latent) null else prompt_text, zero);
            const sc = plausibility(units, next.len);
            if (sc < best_score) {
                a.free(best);
                best = next;
                best_score = sc;
            } else a.free(next);
        }
        return best;
    }

    /// 0 when `samples` of audio is a plausible length for `units` spoken
    /// units (1.2 to 12 units/s; fragments under 4 units always pass), else
    /// how far outside that range the rate is (log ratio).
    fn plausibility(units: usize, samples: usize) f64 {
        if (units < 4) return 0;
        const secs = @max(@as(f64, @floatFromInt(samples)) / SAMPLE_RATE, 1e-3);
        const rate = @as(f64, @floatFromInt(units)) / secs;
        if (rate < 1.2) return @log(1.2 / rate);
        if (rate > 12) return @log(rate / 12);
        return 0;
    }

    /// core.generate: one fragment → untrimmed waveform (owned).
    fn generateFragment(e: *Engine, req: text.Request, frag: []const u8, spk: ?[]const f32, prompt_latent_in: ?Arr, prompt_text: ?[]const u8, zero: bool) ![]f32 {
        const a = e.a;
        const max_steps = @min(@as(usize, 1000), model.BASE_STEPS + model.STEPS_PER_UNIT * text.spokenUnits(frag));
        const pl: ?Arr = if (prompt_text == null) null else prompt_latent_in;
        var g = G.init(a, e.s);
        defer g.deinit();
        const spk_arr: ?Arr = if (spk) |x| try g.fromF32(x, &.{ 1, @intCast(model.SPK_DIM) }) else null;
        const view: RequestView = .{ .prompt = req.prompt, .text = frag, .instruction = req.instruction, .prompt_text = prompt_text, .use_zero_spk_emb = zero };
        const built = try e.buildInputs(&g, view, spk_arr, pl);
        defer a.free(built.ids);
        const lat = try model.sampleLatents(e.model, built.embeds, pl, .{ .max_steps = max_steps, .cfg = req.cfg, .sigma = req.sigma, .temperature = req.temperature });
        _ = try g.keep(lat);
        return e.model.vaeDecode(&g, lat);
    }

    /// CAM++ embedding of 16 kHz audio (owned), from the one-entry cache when
    /// the samples match the last call.
    fn speakerEmbed(e: *Engine, w16: []const f32) ![]f32 {
        const key = std.hash.Wyhash.hash(w16.len, std.mem.sliceAsBytes(w16));
        if (e.spk_cache) |c| if (c.key == key) return e.a.dupe(f32, c.emb);
        const emb = try e.speaker.?.embed(e.a, w16);
        errdefer e.a.free(emb);
        const keep = try e.a.dupe(f32, emb);
        if (e.spk_cache) |c| e.a.free(c.emb);
        e.spk_cache = .{ .key = key, .emb = keep };
        return emb;
    }

    /// pad_prompt + encode_prompt_latent: 44.1 kHz audio → latents [1, T, 64] (owned).
    pub fn encodePromptLatent(e: *Engine, wav44: []const f32) !Arr {
        const padded = try padPrompt(e.a, wav44);
        defer e.a.free(padded);
        var g = G.init(e.a, e.s);
        defer g.deinit();
        const x = try promptInput(&g, e.model, try g.fromF32(padded, &.{@intCast(padded.len)}));
        const h = try e.model.vaeEncodeH(&g, x);
        const half = @divExact(mlx.getShape(h)[2], 2);
        const shape = [_]c_int{ 1, mlx.getShape(h)[1], half };
        const nz = try g.normal(&shape, mlx.mlx_array_dtype(h));
        const lat = try model.Model.vaePosterior(&g, h, nz);
        try model.evalAll(&.{lat});
        return model.own(lat);
    }

    pub const Built = struct { ids: []i32, embeds: Arr };

    /// core.build_inputs: token ids + embeddings with the speaker slot and
    /// prompt latents spliced in. `embeds` lives in `g`.
    pub fn buildInputs(e: *Engine, g: *G, r: RequestView, spk_emb: ?Arr, prompt_latent: ?Arr) !Built {
        const a = e.a;
        const m = e.model;
        var proj_spk: ?Arr = null;
        if (spk_emb) |se| {
            const p = try m.spk_head.f(g, try g.astype(try g.reshape(se, &.{ 1, -1 }), m.dtype));
            proj_spk = if (r.use_zero_spk_emb) try g.zeros(mlx.getShape(p), mlx.mlx_array_dtype(p)) else p;
        }
        var proj_lat: ?Arr = null;
        if (prompt_latent != null and r.prompt_text != null) {
            const pl = try g.astype(try g.reshape(prompt_latent.?, &.{ -1, 1, 64 }), m.dtype);
            const agg = try m.aggregate(g, pl); // [T, 1, 896]
            proj_lat = try g.reshape(agg, &.{ 1, -1, mlx.getShape(agg)[2] });
        }
        const steps: usize = if (proj_lat) |x| @intCast(mlx.getShape(x)[1]) else 0;
        const ii = try text.buildInputIds(a, &e.tok, r.prompt, r.text, if (proj_spk != null) 1 else 0, r.instruction, r.prompt_text, steps);
        defer a.free(ii.speaker_positions);
        errdefer a.free(ii.ids);
        var emb = try m.embedIds(g, ii.ids);
        const edt = mlx.mlx_array_dtype(emb);
        const L: c_int = @intCast(ii.ids.len);
        if (proj_spk) |ps| {
            const pos: c_int = @intCast(ii.speaker_positions[0]);
            const row = try g.astype(try g.reshape(ps, &.{ 1, 1, -1 }), edt);
            emb = try g.concat(&.{ try g.sliceAxis(emb, 1, 0, pos, 1), row, try g.sliceAxis(emb, 1, pos + 1, L, 1) }, 1);
        }
        if (proj_lat) |pl| {
            const lp: c_int = @intCast(ii.latent_position.?);
            emb = try g.concat(&.{ try g.sliceAxis(emb, 1, 0, lp, 1), try g.astype(pl, edt) }, 1);
        }
        return .{ .ids = ii.ids, .embeds = emb };
    }
};

/// The request fields `buildInputs` reads.
pub const RequestView = struct {
    prompt: []const u8,
    text: []const u8,
    instruction: ?[]const u8,
    prompt_text: ?[]const u8,
    use_zero_spk_emb: bool,
};

fn nsamp(s: f32) usize {
    return @intFromFloat(@round(@max(s, 0) * SAMPLE_RATE));
}

/// Front-pad to a multiple of 3528 samples (hop × VAE patch), owned.
pub fn padPrompt(a: std.mem.Allocator, w: []const f32) ![]f32 {
    const r = w.len % model.PAD_ALIGN;
    const pad = if (r == 0) 0 else model.PAD_ALIGN - r;
    const out = try a.alloc(f32, w.len + pad);
    @memset(out[0..pad], 0);
    @memcpy(out[pad..], w);
    return out;
}

/// Round to the model dtype like the reference (waveform.to(bf16)), then f32: [1, N].
pub fn promptInput(g: *G, m: *model.Model, w: Arr) !Arr {
    return g.astype(try g.astype(try g.reshape(w, &.{ 1, -1 }), m.dtype), .float32);
}

fn isChineseUtf8(s: []const u8) bool {
    var it = (std.unicode.Utf8View.init(s) catch return false).iterator();
    while (it.nextCodepoint()) |c| if (text.isCjk(c)) return true;
    return false;
}

/// prompt_contract.validate_tokenizer: the trained ids, each a single token.
fn validateTokenizer(a: std.mem.Allocator, tok: *const tokenizer_mod.Tokenizer) !void {
    const want = [_]struct { []const u8, u32 }{
        .{ "<|endoftext|>", 151643 },    .{ "<|im_start|>", 151644 },   .{ "<|im_end|>", 151645 },
        .{ "<|vision_start|>", 151652 }, .{ "<|vision_end|>", 151653 }, .{ "<|vision_pad|>", 151654 },
        .{ "<audioPatch>", 151705 },     .{ "<audio>", 151706 },
    };
    for (want) |w| {
        const ids = try tok.encode(a, w[0]);
        defer a.free(ids);
        if (ids.len != 1 or ids[0] != w[1]) {
            log.err("[ming_tts] tokenizer does not match the Ming-3.0-tts dense contract: {s} -> {any} (want {d})\n", .{ w[0], ids, w[1] });
            return error.TokenizerMismatch;
        }
    }
}

/// Each task's required and forbidden fields, then the reference's own validation, so a refusal carries its message.
pub fn mapGenerate(r: GenerateRequest, d: *Diag) !text.TaskInput {
    var tb: [32]u8 = undefined;
    const t = text.strip(r.task);
    const n = @min(t.len, tb.len);
    for (t[0..n], 0..) |c, i| tb[i] = std.ascii.toUpper(c);
    const task = tb[0..n];
    var known = false;
    for (V2_TASKS) |x| if (std.mem.eql(u8, x, task)) {
        known = true;
    };
    if (!known) return d.set("task must be one of ['TTS', 'MULTILINGUAL', 'BASIC', 'EMOTION', 'DIALECT', 'IP', 'FREEFORM', 'VOICE_DESIGN']", .{});
    const has_controls = !r.controls.isEmpty() or r.unknown_controls.len > 0;
    const Field = struct { name: []const u8, present: bool };
    const fields = [_]Field{
        .{ .name = "ref_audio", .present = r.prompt_wav != null },
        .{ .name = "ref_text", .present = r.prompt_text != null },
        .{ .name = "language", .present = r.language != null },
        .{ .name = "voice", .present = r.voice != null },
        .{ .name = "instruction", .present = r.instruction != null },
        .{ .name = "controls", .present = has_controls },
    };
    const Rule = struct { task: []const u8, required: []const []const u8, forbidden: []const []const u8 };
    const rules = [_]Rule{
        .{ .task = "TTS", .required = &.{}, .forbidden = &.{ "voice", "instruction", "controls" } },
        .{ .task = "MULTILINGUAL", .required = &.{ "ref_audio", "ref_text", "language" }, .forbidden = &.{ "voice", "instruction", "controls" } },
        .{ .task = "BASIC", .required = &.{ "ref_audio", "controls" }, .forbidden = &.{ "ref_text", "language", "voice", "instruction" } },
        .{ .task = "EMOTION", .required = &.{ "ref_audio", "controls" }, .forbidden = &.{ "ref_text", "language", "voice", "instruction" } },
        .{ .task = "DIALECT", .required = &.{ "ref_audio", "controls" }, .forbidden = &.{ "ref_text", "language", "voice", "instruction" } },
        .{ .task = "IP", .required = &.{"voice"}, .forbidden = &.{ "ref_audio", "ref_text", "language", "instruction", "controls" } },
        .{ .task = "FREEFORM", .required = &.{"instruction"}, .forbidden = &.{ "ref_audio", "ref_text", "language", "voice", "controls" } },
        .{ .task = "VOICE_DESIGN", .required = &.{}, .forbidden = &.{ "ref_audio", "ref_text", "language", "voice" } },
    };
    for (rules) |rule| {
        if (!std.mem.eql(u8, rule.task, task)) continue;
        for (rule.required) |f| for (fields) |fd| if (std.mem.eql(u8, fd.name, f) and !fd.present) return d.set("{s} requires {s}", .{ rule.task, f });
        for (rule.forbidden) |f| for (fields) |fd| if (std.mem.eql(u8, fd.name, f) and fd.present) return d.set("{s} does not accept {s}", .{ rule.task, f });
    }
    const task_name: []const u8 = for (V2_TASKS) |x| {
        if (std.mem.eql(u8, x, task)) break x;
    } else unreachable;
    var in: text.TaskInput = .{ .task = task_name, .text = r.text, .language_id = r.language, .controls = r.controls, .unknown_controls = r.unknown_controls };
    if (std.mem.eql(u8, task, "IP") or std.mem.eql(u8, task, "FREEFORM") or std.mem.eql(u8, task, "VOICE_DESIGN")) in.use_zero_spk_emb = true;
    if (std.mem.eql(u8, task, "IP")) {
        if (!text.isVoice(r.voice.?)) return d.set("unknown voice '{s}'; " ++ text.VOICE_HINT, .{r.voice.?});
        in.controls = .{ .ip = r.voice };
    } else if (std.mem.eql(u8, task, "FREEFORM")) {
        in.instruction = r.instruction;
    } else if (std.mem.eql(u8, task, "VOICE_DESIGN")) {
        if (r.instruction != null and has_controls) return d.set("VOICE_DESIGN takes controls.style or instruction, not both", .{});
        if (r.instruction) |ins| in.controls = .{ .style = ins };
    } else if (r.prompt_wav) |p| {
        in.prompt_wav = p;
        in.prompt_text = r.prompt_text;
        in.use_spk_emb = true;
    }
    var tmp_arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer tmp_arena.deinit();
    _ = try text.buildRequest(tmp_arena.allocator(), in, d);
    return in;
}
