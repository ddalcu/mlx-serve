//! NVIDIA Nemotron 3 Diarization (`nemotron_diarization`, mlx-audio's pack
//! layout): who speaks when, up to eight speakers, one probability per speaker
//! every 10 ms, the speakers numbered in the order they first speak. NeMo
//! log-mels, eight to an encoder frame, feed a 31-layer RoPE transformer; a
//! sub-pixel convolution returns to 10 ms. Long audio runs in chunks that also
//! attend to a FIFO of the latest frames and an Arrival-Order Speaker Cache, so
//! a voice keeps its channel across the recording: NVIDIA's offline preset, as
//! the official model in transformers runs it.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const model_mod = @import("../../sdk/weights/load.zig");
const scope_mod = @import("../../mlx/scope.zig");

const Scope = scope_mod.Scope;
const Linear = scope_mod.Linear;
const getW = scope_mod.getW;
const linear = scope_mod.linear;
const A = mlx.mlx_array;
const S = mlx.mlx_stream;

pub const SAMPLE_RATE: u32 = 16000;

/// Which frames a full speaker cache keeps (the official `_compress`).
pub const CachePolicy = struct {
    len: u32,
    silence_per_speaker: u32,
    score_threshold: f32,
    latest_boost: f32,
    strong_boost_rate: f32,
    weak_boost_rate: f32,
    min_positive_rate: f32,
};

pub const Config = struct {
    mels: u32,
    d_model: u32,
    layers: u32,
    heads: u32,
    stack: u32,
    ff: u32,
    rope_base: f32,
    head_dim: u32,
    speakers: u32,
    chunk_len: u32,
    right_context: u32,
    fifo_len: u32,
    update_period: u32,
    cache: CachePolicy,
    n_fft: u32,
    win: u32,
    hop: u32,
    preemphasis: f32,
};

/// mlx-audio's `config.json`.
const ConfigFile = struct {
    num_speakers: u32,
    output_subsampling_factor: u32 = 1,
    encoder_config: struct {
        feat_in: u32,
        d_model: u32,
        n_layers: u32,
        n_heads: u32,
        subsampling_factor: u32,
        ff_expansion: f32,
        qkv_bias: bool = false,
        qk_norm: bool = false,
        pre_block_norm: bool = true,
        xscaling: bool = false,
        rope_base: f32 = 10000,
        rotary_fraction: f32 = 1,
    },
    modules_config: struct {
        tf_d_model: u32,
        chunk_len: u32,
        fifo_len: u32,
        spkcache_len: u32,
        spkcache_update_period: u32,
        chunk_left_context: u32 = 0,
        chunk_right_context: u32,
        spkcache_sil_frames_per_spk: u32,
        pred_score_threshold: f32,
        scores_boost_latest: f32,
        strong_boost_rate: f32,
        weak_boost_rate: f32,
        min_pos_scores_rate: f32,
        use_aosc: bool = true,
        use_learnable_sil_emb: bool = true,
    },
    processor_config: struct {
        feature_size: u32,
        sampling_rate: u32,
        hop_length: u32,
        n_fft: u32,
        win_length: u32,
        preemphasis: f32,
    },
};

/// The checkpoint's shape, refused by name where the port has no answer.
pub fn parseConfig(a: std.mem.Allocator, bytes: []const u8) !Config {
    const parsed = std.json.parseFromSlice(ConfigFile, a, bytes, .{ .ignore_unknown_fields = true }) catch return error.DiarizerConfigInvalid;
    defer parsed.deinit();
    const f = parsed.value;
    const e = f.encoder_config;
    const m = f.modules_config;
    const p = f.processor_config;
    if (e.qkv_bias or e.qk_norm or !e.pre_block_norm or e.xscaling or e.rotary_fraction != 1 or m.chunk_left_context != 0 or
        !m.use_aosc or !m.use_learnable_sil_emb or f.output_subsampling_factor != 1 or p.sampling_rate != SAMPLE_RATE or
        p.win_length > p.n_fft or e.feat_in != p.feature_size)
        return error.DiarizerConfigUnsupported;
    if (f.num_speakers == 0 or e.n_heads == 0 or e.d_model % e.n_heads != 0 or e.subsampling_factor == 0 or m.chunk_len == 0 or
        m.spkcache_len < (1 + m.spkcache_sil_frames_per_spk) * f.num_speakers)
        return error.DiarizerConfigInvalid;
    return .{
        .mels = p.feature_size,
        .d_model = e.d_model,
        .layers = e.n_layers,
        .heads = e.n_heads,
        .stack = e.subsampling_factor,
        .ff = @intFromFloat(@as(f32, @floatFromInt(e.d_model)) * e.ff_expansion),
        .rope_base = e.rope_base,
        .head_dim = e.d_model / e.n_heads,
        .speakers = f.num_speakers,
        .chunk_len = m.chunk_len,
        .right_context = m.chunk_right_context,
        .fifo_len = m.fifo_len,
        .update_period = m.spkcache_update_period,
        .cache = .{
            .len = m.spkcache_len,
            .silence_per_speaker = m.spkcache_sil_frames_per_spk,
            .score_threshold = m.pred_score_threshold,
            .latest_boost = m.scores_boost_latest,
            .strong_boost_rate = m.strong_boost_rate,
            .weak_boost_rate = m.weak_boost_rate,
            .min_positive_rate = m.min_pos_scores_rate,
        },
        .n_fft = p.n_fft,
        .win = p.win_length,
        .hop = p.hop_length,
        .preemphasis = p.preemphasis,
    };
}

/// The frames a cache over `probs` (`n` frames of `speakers` probabilities)
/// keeps, `policy.len` of them: grouped by speaker, in order within one, `n`
/// for a silence slot. Among equal scores the earlier frame wins.
pub fn aoscKeep(a: std.mem.Allocator, probs: []const f32, speakers: usize, policy: CachePolicy) ![]u32 {
    const n = probs.len / speakers;
    const sil: usize = policy.silence_per_speaker;
    const rows = n + sil;
    const budget: f32 = @floatFromInt(policy.len / speakers - sil);
    const min_positive: usize = @intFromFloat(@floor(budget * policy.min_positive_rate));
    const ln_half = @log(@as(f32, 0.5));
    const scores = try a.alloc(f32, speakers * rows);
    defer a.free(scores);
    const ninf = -std.math.inf(f32);
    const thr = policy.score_threshold;
    for (0..n) |t| {
        const p = probs[t * speakers ..][0..speakers];
        var complements: f32 = 0;
        for (p) |x| complements += @log(@max(1 - x, thr));
        for (p, 0..) |x, s| {
            const v = @log(@max(x, thr)) - @log(@max(1 - x, thr)) + complements - ln_half;
            scores[s * rows + t] = if (x > 0.5) v else ninf;
        }
    }
    for (0..speakers) |s| {
        const row = scores[s * rows ..][0..n];
        var positive: usize = 0;
        for (row) |v| positive += @intFromBool(v > 0);
        // A speaker with enough frames of its own drops its overlapped ones.
        if (positive >= min_positive) {
            for (row, 0..) |*v, t| {
                if (!(v.* > 0) and probs[t * speakers + s] > 0.5) v.* = ninf;
            }
        }
        // The frames popped from the FIFO are past the cache's length.
        for (row[@min(policy.len, n)..]) |*v| v.* += policy.latest_boost;
    }
    const order = try a.alloc(u32, @max(n, speakers * rows));
    defer a.free(order);
    for ([_]struct { rate: f32, boost: f32 }{ .{ .rate = policy.strong_boost_rate, .boost = -2 * ln_half }, .{ .rate = policy.weak_boost_rate, .boost = -ln_half } }) |b| {
        const k = @min(n, @as(usize, @intFromFloat(@floor(budget * b.rate))));
        for (0..speakers) |s| {
            const row = scores[s * rows ..][0..n];
            const idx = order[0..n];
            for (idx, 0..) |*i, t| i.* = @intCast(t);
            std.sort.pdq(u32, idx, @as([]const f32, row), higherFirst);
            for (idx[0..k]) |t| row[t] += b.boost;
        }
    }
    for (0..speakers) |s| @memset(scores[s * rows + n ..][0..sil], std.math.inf(f32));
    const all = order[0 .. speakers * rows];
    for (all, 0..) |*i, j| i.* = @intCast(j);
    std.sort.pdq(u32, all, @as([]const f32, scores), higherFirst);
    const keep = try a.alloc(u32, policy.len);
    errdefer a.free(keep);
    const sentinel: u32 = @intCast(speakers * rows);
    for (keep, all[0..policy.len]) |*k, j| k.* = if (scores[j] == ninf) sentinel else j;
    std.sort.pdq(u32, keep, {}, std.sort.asc(u32));
    for (keep) |*k| k.* = if (k.* == sentinel) @intCast(n) else @intCast(@min(k.* % rows, n));
    return keep;
}

fn higherFirst(scores: []const f32, x: u32, y: u32) bool {
    return scores[x] > scores[y] or (scores[x] == scores[y] and x < y);
}

const Layer = struct {
    norm1_w: A,
    norm1_b: A,
    qkv: Linear,
    out: Linear,
    norm2_w: A,
    norm2_b: A,
    ff1: Linear,
    ff2: Linear,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    s: S,
    cfg: Config,
    w: model_mod.Weights,
    layers: []Layer,
    stack: Linear,
    embed_norm_w: A,
    embed_norm_b: A,
    final_norm_w: A,
    final_norm_b: A,
    proj: Linear,
    up_w: A,
    up_b: A,
    dense: Linear,
    out: Linear,
    silence: A,
    /// Owned: the Slaney filterbank `[mels, n_fft/2+1]` and, on the host, the
    /// Hann window centered in `n_fft`.
    fb: A,
    window: []f32,
    /// The activations' dtype: the pack's float weights'.
    dtype: mlx.mlx_dtype,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !*Engine {
        return loadOn(io, allocator, model_dir, mlx.mlx_default_gpu_stream_new());
    }

    /// `load` running on `s` (borrowed).
    pub fn loadOn(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, s: S) !*Engine {
        var dir = try std.Io.Dir.openDirAbsolute(io, model_dir, .{});
        defer dir.close(io);
        const cfg_bytes = try dir.readFileAlloc(io, "config.json", allocator, .limited(1 << 20));
        defer allocator.free(cfg_bytes);
        const cfg = try parseConfig(allocator, cfg_bytes);
        // A missing file is refused here, by name, rather than as an MLX read error.
        dir.access(io, "model.safetensors", .{}) catch return error.DiarizerWeightsMissing;

        const self = try allocator.create(Engine);
        errdefer allocator.destroy(self);
        self.allocator = allocator;
        self.s = s;
        self.cfg = cfg;
        self.w = try model_mod.loadWeightsFile(allocator, model_dir, "model.safetensors");
        errdefer self.w.deinit();
        try scope_mod.evalAll(&self.w);
        self.layers = try allocator.alloc(Layer, cfg.layers);
        errdefer allocator.free(self.layers);
        const w = &self.w;
        const d = cfg.d_model;
        for (self.layers, 0..) |*l, i| l.* = .{
            .norm1_w = try getW(w, "encoder.layers.{d}.norm1.weight", .{i}),
            .norm1_b = try getW(w, "encoder.layers.{d}.norm1.bias", .{i}),
            .qkv = try linear(w, "encoder.layers.{d}.attn.w_qkv", .{i}, d),
            .out = try linear(w, "encoder.layers.{d}.attn.out_proj", .{i}, d),
            .norm2_w = try getW(w, "encoder.layers.{d}.norm2.weight", .{i}),
            .norm2_b = try getW(w, "encoder.layers.{d}.norm2.bias", .{i}),
            .ff1 = try linear(w, "encoder.layers.{d}.ffn.linear1", .{i}, d),
            .ff2 = try linear(w, "encoder.layers.{d}.ffn.linear2", .{i}, cfg.ff),
        };
        self.stack = try linear(w, "encoder.pre_encode.proj", .{}, cfg.mels * cfg.stack);
        self.embed_norm_w = try getW(w, "encoder.embed_norm.weight", .{});
        self.embed_norm_b = try getW(w, "encoder.embed_norm.bias", .{});
        self.final_norm_w = try getW(w, "encoder.final_norm.weight", .{});
        self.final_norm_b = try getW(w, "encoder.final_norm.bias", .{});
        self.proj = try linear(w, "sortformer_modules.encoder_proj", .{}, d);
        self.up_w = try getW(w, "sortformer_modules.subpixel_upsample.weight", .{});
        self.up_b = try getW(w, "sortformer_modules.subpixel_upsample.bias", .{});
        const tf = @as(u32, @intCast(mlx.getShape(self.up_w)[2]));
        self.dense = try linear(w, "sortformer_modules.first_hidden_to_hidden", .{}, tf);
        self.out = try linear(w, "sortformer_modules.single_hidden_to_spks", .{}, tf);
        self.silence = try getW(w, "sortformer_modules.learnable_sil_emb", .{});
        self.dtype = mlx.mlx_array_dtype(self.embed_norm_w);

        var sc = Scope.init(allocator, self.s);
        defer sc.deinit();
        self.fb = sc.out(try sc.reshape(try getW(w, "preprocessor.fb", .{}), &.{ @intCast(cfg.mels), @intCast(cfg.n_fft / 2 + 1) }));
        errdefer _ = mlx.mlx_array_free(self.fb);
        const win = try getW(w, "preprocessor.window", .{});
        try mlx.check(mlx.mlx_array_eval(win));
        const taps = (mlx.mlx_array_data_float32(win) orelse return error.DiarizerWindow)[0..cfg.win];
        self.window = try allocator.alloc(f32, cfg.n_fft);
        @memset(self.window, 0);
        @memcpy(self.window[(cfg.n_fft - cfg.win) / 2 ..][0..cfg.win], taps);
        log.info("[diarize] engine ready ({d} tensors, {d} layers, {d} speakers)\n", .{ self.w.count(), cfg.layers, cfg.speakers });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.allocator.free(self.window);
        _ = mlx.mlx_array_free(self.fb);
        self.allocator.free(self.layers);
        self.w.deinit();
        self.allocator.destroy(self);
    }

    /// Speaker activity every 10 ms of `pcm` (16 kHz mono): `pcm.len / hop`
    /// rows of `cfg.speakers` probabilities, owned by `a`.
    pub fn diarize(self: *Engine, a: std.mem.Allocator, pcm: []const f32) ![]f32 {
        const c = self.cfg;
        const frames = pcm.len / c.hop;
        const out = try a.alloc(f32, frames * c.speakers);
        errdefer a.free(out);
        var st = Stream{ .e = self, .a = a };
        defer st.deinit();
        var done: usize = 0;
        while (done < frames) {
            const n = @min(c.chunk_len * c.stack, frames - done);
            const window = @min((c.chunk_len + c.right_context) * c.stack, frames - done);
            try st.step(pcm, done, n, window, out[done * c.speakers ..][0 .. n * c.speakers]);
            done += n;
        }
        _ = mlx.mlx_clear_cache();
        return out;
    }

    /// NeMo's log-mel of frames `first .. first + count` of `pcm`: windows
    /// centered on each hop over the pre-emphasized clip, zero past its ends.
    /// `[count, mels]` f32.
    pub fn melFrames(self: *const Engine, sc: *Scope, pcm: []const f32, first: usize, count: usize) !A {
        const c = self.cfg;
        const host = try sc.a.alloc(f32, count * c.n_fft);
        defer sc.a.free(host);
        for (0..count) |f| {
            const base = @as(i64, @intCast((first + f) * c.hop)) - c.n_fft / 2;
            for (host[f * c.n_fft ..][0..c.n_fft], self.window, 0..) |*h, wv, k| {
                const i = base + @as(i64, @intCast(k));
                const x: f32 = if (i < 0 or i >= pcm.len) 0 else pcm[@intCast(i)] - (if (i > 0) c.preemphasis * pcm[@intCast(i - 1)] else 0);
                h.* = x * wv;
            }
        }
        const framed = try sc.fromF32(host, &.{ @intCast(count), @intCast(c.n_fft) });
        var sp = mlx.mlx_array_new();
        const spec = try sc.res(mlx.mlx_fft_rfft(&sp, framed, @intCast(c.n_fft), 1, mlx.MLX_FFT_NORM_BACKWARD, self.s), &sp);
        var mg = mlx.mlx_array_new();
        const mag = try sc.res(mlx.mlx_abs(&mg, spec, self.s), &mg);
        var lg = mlx.mlx_array_new();
        const mel = try sc.addS(try sc.lin(try sc.mul(mag, mag), self.fb, null), 1.0 / 16777216.0);
        const feats = try sc.res(mlx.mlx_log(&lg, mel, self.s), &lg);
        try mlx.check(mlx.mlx_array_eval(feats));
        return feats;
    }

    /// The encoder's input for frames `first .. first + count`: eight log-mel
    /// frames to a row, the last group zero-padded. `[1, rows, d_model]`.
    fn inputRows(self: *const Engine, sc: *Scope, pcm: []const f32, first: usize, count: usize) !A {
        const c = self.cfg;
        const pad = (c.stack - count % c.stack) % c.stack;
        var feats = try self.melFrames(sc, pcm, first, count);
        if (pad > 0) feats = try sc.padAxis(feats, 0, 0, @intCast(pad));
        const stacked = try sc.reshape(feats, &.{ 1, @intCast((count + pad) / c.stack), @intCast(c.mels * c.stack) });
        return self.stack.forward(sc, try sc.astype(stacked, self.dtype));
    }

    /// Speaker probabilities `[rows * stack, speakers]` (f32) for the encoder input `x`.
    fn speakerProbs(self: *const Engine, sc: *Scope, x: A) !A {
        const c = self.cfg;
        const rows = mlx.getShape(x)[1];
        var h = try sc.layerNorm(x, self.embed_norm_w, self.embed_norm_b, 1e-5);
        for (self.layers) |*l| {
            h = try sc.add(h, try self.attention(sc, l, try sc.layerNorm(h, l.norm1_w, l.norm1_b, 1e-5)));
            const n2 = try sc.layerNorm(h, l.norm2_w, l.norm2_b, 1e-5);
            h = try sc.add(h, try l.ff2.forward(sc, try sc.geluErf(try l.ff1.forward(sc, n2))));
        }
        h = try sc.layerNorm(h, self.final_norm_w, self.final_norm_b, 1e-5);
        const p = try self.proj.forward(sc, h);
        const tf = mlx.getShape(p)[2];
        const up = try sc.reshape(try sc.add(try sc.conv1d(p, self.up_w, 1, 1, 1), self.up_b), &.{ rows * @as(c_int, @intCast(c.stack)), tf });
        const hidden = try self.dense.forward(sc, try sc.relu(up));
        // The sigmoid in f32: the reference's, and the cache scores tell close frames apart.
        return sc.sigmoid(try sc.astype(try self.out.forward(sc, try sc.relu(hidden)), .float32));
    }

    fn attention(self: *const Engine, sc: *Scope, l: *const Layer, x: A) !A {
        const c = self.cfg;
        const t = mlx.getShape(x)[1];
        const h: c_int = @intCast(c.heads);
        const hd: c_int = @intCast(c.head_dim);
        const qkv = try sc.transpose(try sc.reshape(try l.qkv.forward(sc, x), &.{ 1, t, 3, h, hd }), &.{ 2, 0, 3, 1, 4 });
        var parts: [3]A = undefined;
        for (&parts, 0..) |*p, i| p.* = try sc.reshape(try sc.slice(qkv, 0, @intCast(i), @intCast(i + 1)), &.{ 1, h, t, hd });
        const q = try sc.ropeAt(parts[0], hd, c.rope_base, 0);
        const k = try sc.ropeAt(parts[1], hd, c.rope_base, 0);
        const o = try sc.sdpa(q, k, parts[2], 1 / @sqrt(@as(f32, @floatFromInt(c.head_dim))));
        return l.out.forward(sc, try sc.merge(o));
    }
};

/// The state chunks hand on: the speaker cache and the FIFO (encoder input
/// rows, owned), and the cache's probabilities once it is compressed.
const Stream = struct {
    e: *Engine,
    a: std.mem.Allocator,
    cache: ?A = null,
    fifo: ?A = null,
    cache_probs: []f32 = &.{},
    compressed: bool = false,

    fn deinit(st: *Stream) void {
        if (st.cache) |x| _ = mlx.mlx_array_free(x);
        if (st.fifo) |x| _ = mlx.mlx_array_free(x);
        st.a.free(st.cache_probs);
    }

    fn rowsOf(x: ?A) usize {
        return if (x) |v| @intCast(mlx.getShape(v)[1]) else 0;
    }

    /// One chunk: `n` frames from `first` written to `dst`, read with the
    /// look-ahead up to `window` frames; then its frames join the FIFO, and
    /// the FIFO's oldest the cache.
    fn step(st: *Stream, pcm: []const f32, first: usize, n: usize, window: usize, dst: []f32) !void {
        const e = st.e;
        const c = e.cfg;
        const spk: usize = c.speakers;
        const f: usize = c.stack;
        var sc = Scope.init(st.a, e.s);
        defer sc.deinit();
        const chunk = try e.inputRows(&sc, pcm, first, window);
        const nc = rowsOf(st.cache);
        const nf = rowsOf(st.fifo);
        var parts: [3]A = undefined;
        var np: usize = 0;
        for ([_]?A{ st.cache, st.fifo }) |x| if (x) |v| {
            parts[np] = v;
            np += 1;
        };
        parts[np] = chunk;
        const probs = try e.speakerProbs(&sc, if (np == 0) chunk else try sc.concat(parts[0 .. np + 1], 1));
        try mlx.check(mlx.mlx_array_eval(probs));
        const high = (mlx.mlx_array_data_float32(probs) orelse return error.DiarizerOutput)[0..mlx.mlx_array_size(probs)];
        @memcpy(dst, high[(nc + nf) * f * spk ..][0 .. n * spk]);

        // The chunk's own frames join the FIFO; its look-ahead opens the next chunk.
        const joined = (n + f - 1) / f;
        const own = try sc.slice(chunk, 1, 0, @intCast(joined));
        const fifo = if (st.fifo) |v| try sc.concat(&.{ v, own }, 1) else own;
        const queued = nf + joined;
        if (queued <= c.fifo_len) return st.replace(&sc, st.cache, fifo);
        const pop = @min(queued, @max(c.update_period, queued - c.fifo_len));
        // Encoder-rate probabilities: each row's mean over its 10 ms frames.
        const total = nc + pop;
        const low = try st.a.alloc(f32, total * spk);
        defer st.a.free(low);
        for (0..total) |r| for (0..spk) |s| {
            var sum: f32 = 0;
            for (0..f) |k| sum += high[(r * f + k) * spk + s];
            low[r * spk + s] = sum / @as(f32, @floatFromInt(f));
        };
        // A compressed cache is out of order: only its stored probabilities describe it.
        if (st.compressed) @memcpy(low[0 .. nc * spk], st.cache_probs);
        const popped = try sc.slice(fifo, 1, 0, @intCast(pop));
        const rest: ?A = if (pop < queued) try sc.slice(fifo, 1, @intCast(pop), @intCast(queued)) else null;
        const grown = if (st.cache) |v| try sc.concat(&.{ v, popped }, 1) else popped;
        if (total <= c.cache.len) return st.replace(&sc, grown, rest);

        const keep = try aoscKeep(st.a, low, spk, c.cache);
        defer st.a.free(keep);
        const probs_kept = try st.a.alloc(f32, keep.len * spk);
        for (keep, 0..) |k, i| {
            const row = probs_kept[i * spk ..][0..spk];
            if (k == total) @memset(row, 0) else @memcpy(row, low[k * spk ..][0..spk]);
        }
        st.a.free(st.cache_probs);
        st.cache_probs = probs_kept;
        st.compressed = true;
        const d: c_int = @intCast(c.d_model);
        const table = try sc.concat(&.{ try sc.reshape(grown, &.{ @intCast(total), d }), try sc.reshape(try sc.astype(e.silence, e.dtype), &.{ 1, d }) }, 0);
        const ids = try sc.keep(mlx.mlx_array_new_data(keep.ptr, &[_]c_int{@intCast(keep.len)}, 1, .uint32));
        const cache = try sc.reshape(try sc.take(table, ids), &.{ 1, @intCast(keep.len), d });
        return st.replace(&sc, cache, rest);
    }

    /// Hold `cache` and `fifo` past the step's scope, evaluated.
    fn replace(st: *Stream, sc: *Scope, cache: ?A, fifo: ?A) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        const next = [2]?A{ if (cache) |v| sc.out(v) else null, if (fifo) |v| sc.out(v) else null };
        for (next) |x| {
            if (x) |v| _ = mlx.mlx_vector_array_append_value(vec, v);
        }
        mlx.check(mlx.mlx_eval(vec)) catch |err| {
            for (next) |x| {
                if (x) |v| _ = mlx.mlx_array_free(v);
            }
            return err;
        };
        if (st.cache) |v| _ = mlx.mlx_array_free(v);
        if (st.fifo) |v| _ = mlx.mlx_array_free(v);
        st.cache = next[0];
        st.fifo = next[1];
    }
};

// ── Tests ──

const testing = std.testing;

const POLICY = CachePolicy{ .len = 264, .silence_per_speaker = 1, .score_threshold = 0.25, .latest_boost = 0.05, .strong_boost_rate = 0.75, .weak_boost_rate = 1.5, .min_positive_rate = 0.5 };

test "parseConfig reads mlx-audio's config and refuses what the port lacks" {
    const a = testing.allocator;
    const base =
        \\{"model_type":"nemotron_diarization","num_speakers":8,"output_subsampling_factor":1,
        \\"encoder_config":{"feat_in":128,"d_model":512,"n_layers":31,"n_heads":8,"subsampling_factor":8,"ff_expansion":4.0,
        \\"qkv_bias":false,"qk_norm":false,"pre_block_norm":true,"xscaling":false,"rope_base":10000.0,"rotary_fraction":1.0},
        \\"modules_config":{"tf_d_model":192,"chunk_len":340,"fifo_len":40,"spkcache_len":264,"spkcache_update_period":300,
        \\"chunk_left_context":0,"chunk_right_context":40,"spkcache_sil_frames_per_spk":1,"pred_score_threshold":0.25,
        \\"scores_boost_latest":0.05,"strong_boost_rate":0.75,"weak_boost_rate":1.5,"min_pos_scores_rate":0.5,
        \\"use_aosc":true,"use_learnable_sil_emb":true},
        \\"processor_config":{"feature_size":128,"sampling_rate":16000,"hop_length":160,"n_fft":512,"win_length":400,"preemphasis":0.97},
        \\"quantization":{"group_size":64,"bits":8}}
    ;
    const c = try parseConfig(a, base);
    try testing.expectEqual(@as(u32, 2048), c.ff);
    try testing.expectEqual(@as(u32, 64), c.head_dim);
    try testing.expectEqual(@as(u32, 8), c.speakers);
    try testing.expectEqual(@as(u32, 300), c.update_period);
    try testing.expectEqual(POLICY, c.cache);
    const qk = try std.mem.replaceOwned(u8, a, base, "\"qk_norm\":false", "\"qk_norm\":true");
    defer a.free(qk);
    try testing.expectError(error.DiarizerConfigUnsupported, parseConfig(a, qk));
    const rate = try std.mem.replaceOwned(u8, a, base, "\"sampling_rate\":16000", "\"sampling_rate\":8000");
    defer a.free(rate);
    try testing.expectError(error.DiarizerConfigUnsupported, parseConfig(a, rate));
}

test "aoscKeep keeps the frames the official compression keeps" {
    // transformers' `_compress` on this case (2 speakers, a cache of 8): frames
    // 0 2 10 and 6 8 11, each speaker's block closed by its silence slot.
    const probs = [_]f32{
        0.95, 0.02, 0.91, 0.04, 0.97, 0.01, 0.10, 0.08, 0.88, 0.70, 0.60, 0.65,
        0.03, 0.93, 0.05, 0.86, 0.02, 0.99, 0.30, 0.20, 0.93, 0.03, 0.01, 0.96,
    };
    const policy = CachePolicy{ .len = 8, .silence_per_speaker = 1, .score_threshold = 0.25, .latest_boost = 0.05, .strong_boost_rate = 0.75, .weak_boost_rate = 1.5, .min_positive_rate = 0.5 };
    const keep = try aoscKeep(testing.allocator, &probs, 2, policy);
    defer testing.allocator.free(keep);
    try testing.expectEqualSlices(u32, &.{ 0, 2, 10, 12, 6, 8, 11, 12 }, keep);
}

test "aoscKeep fills the cache with silence when too few frames hold speech" {
    // Two speech frames for speaker 1, nothing else: the rest are silence slots
    // (transformers' `_compress` keeps the same).
    var probs: [20 * 2]f32 = @splat(0.01);
    probs[3 * 2 + 1] = 0.9;
    probs[15 * 2 + 1] = 0.8;
    const policy = CachePolicy{ .len = 8, .silence_per_speaker = 1, .score_threshold = 0.25, .latest_boost = 0.05, .strong_boost_rate = 0.75, .weak_boost_rate = 1.5, .min_positive_rate = 0.5 };
    const keep = try aoscKeep(testing.allocator, &probs, 2, policy);
    defer testing.allocator.free(keep);
    try testing.expectEqualSlices(u32, &.{ 20, 3, 15, 20, 20, 20, 20, 20 }, keep);
}

// Parity, the references executed by tests/dump_nemotron_diarization_fixtures.py:
//   NEMOTRON_DIARIZATION_MODEL=<pack> NEMOTRON_DIARIZATION_FIXTURE=<file> \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="nemotron diarization"
test "nemotron diarization: features, speaker activity and the cache match the references" {
    const fix_path = envPath("NEMOTRON_DIARIZATION_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = envPath("NEMOTRON_DIARIZATION_MODEL") orelse return error.SkipZigTest;
    const a = testing.allocator;
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();
    const engine = try Engine.load(testing.io, a, model_dir);
    defer engine.deinit();
    const pcm = try fixture(&fx, "pcm");
    const spk = engine.cfg.speakers;

    // mlx-audio's log-mel of the first window, [mels, frames].
    const want_feats = try fixture(&fx, "features");
    const count = want_feats.len / engine.cfg.mels;
    var sc = Scope.init(a, engine.s);
    defer sc.deinit();
    const feats = try engine.melFrames(&sc, pcm, 0, count);
    const got_feats = mlx.mlx_array_data_float32(feats).?[0 .. count * engine.cfg.mels];
    var feat_diff: f32 = 0;
    for (0..count) |t| for (0..engine.cfg.mels) |m| {
        feat_diff = @max(feat_diff, @abs(got_feats[t * engine.cfg.mels + m] - want_feats[m * count + t]));
    };

    const got = try engine.diarize(a, pcm);
    defer a.free(got);
    const mlx_probs = try fixture(&fx, "mlx_probs");
    const hf_probs = try fixture(&fx, "hf_probs");
    try testing.expectEqual(mlx_probs.len, got.len);
    const Agree = struct {
        fn of(x: []const f32, y: []const f32, n_spk: usize) struct { active: f64, speaker: f64, max: f32 } {
            var same: usize = 0;
            var spoken: usize = 0;
            var same_spk: usize = 0;
            var worst: f32 = 0;
            for (0..x.len / n_spk) |t| {
                const rx = x[t * n_spk ..][0..n_spk];
                const ry = y[t * n_spk ..][0..n_spk];
                for (rx, ry) |p, q| {
                    same += @intFromBool((p > 0.5) == (q > 0.5));
                    worst = @max(worst, @abs(p - q));
                }
                if (std.mem.max(f32, ry) > 0.5) {
                    spoken += 1;
                    same_spk += @intFromBool(std.mem.indexOfMax(f32, rx) == std.mem.indexOfMax(f32, ry));
                }
            }
            return .{ .active = @as(f64, @floatFromInt(same)) / @as(f64, @floatFromInt(x.len)), .speaker = @as(f64, @floatFromInt(same_spk)) / @as(f64, @floatFromInt(@max(spoken, 1))), .max = worst };
        }
    };
    const vs_mlx = Agree.of(got, mlx_probs, spk);
    const vs_hf = Agree.of(got, hf_probs, spk);
    const mlx_vs_hf = Agree.of(mlx_probs, hf_probs, spk);

    var same_keep: usize = 0;
    var cases: usize = 0;
    var name_buf: [32]u8 = undefined;
    while (fx.get(try std.fmt.bufPrint(&name_buf, "aosc_probs_{d}", .{cases})) != null) : (cases += 1) {
        const probs = try fixture(&fx, try std.fmt.bufPrint(&name_buf, "aosc_probs_{d}", .{cases}));
        const want = try fixtureI32(&fx, try std.fmt.bufPrint(&name_buf, "aosc_frames_{d}", .{cases}));
        const keep = try aoscKeep(a, probs, spk, engine.cfg.cache);
        defer a.free(keep);
        const n: i32 = @intCast(probs.len / spk);
        var equal = true;
        for (keep, want) |k, wv| equal = equal and @as(i32, @intCast(k)) == (if (wv < 0) n else wv);
        same_keep += @intFromBool(equal);
    }
    std.debug.print("[nemotron-diarization] features max |diff| {d:.5}; vs mlx-audio: active {d:.4}, speaker {d:.4}, max {d:.4}; vs official: active {d:.4}, speaker {d:.4} (mlx-audio {d:.4}, {d:.4}); cache {d}/{d} compressions equal\n", .{ feat_diff, vs_mlx.active, vs_mlx.speaker, vs_mlx.max, vs_hf.active, vs_hf.speaker, mlx_vs_hf.active, mlx_vs_hf.speaker, same_keep, cases });
    // Near-silent bins sit at the log's floor, where float rounding is a few 1e-4.
    try testing.expect(feat_diff < 5e-3);
    try testing.expectEqual(cases, same_keep);
    try testing.expect(vs_mlx.active > 0.995 and vs_mlx.speaker > 0.995);
    try testing.expect(vs_hf.active >= mlx_vs_hf.active - 0.002 and vs_hf.speaker >= mlx_vs_hf.speaker - 0.002);
}

fn envPath(name: [*:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    const p = std.mem.sliceTo(raw, 0);
    return if (p.len == 0) null else p;
}

fn fixture(fx: *model_mod.Weights, name: []const u8) ![]const f32 {
    const arr = fx.get(name) orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(arr));
    return (mlx.mlx_array_data_float32(arr) orelse return error.MissingFixtureTensor)[0..mlx.mlx_array_size(arr)];
}

fn fixtureI32(fx: *model_mod.Weights, name: []const u8) ![]const i32 {
    const arr = fx.get(name) orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(arr));
    return (mlx.mlx_array_data_int32(arr) orelse return error.MissingFixtureTensor)[0..mlx.mlx_array_size(arr)];
}
