//! Qwen3-ASR audio tower: mlx-audio's `qwen3_asr.AudioEncoder` in the packs' layout.
//!
//! Whisper log-mel (n_fft 400, hop 160, 128 Slaney mels), three stride-2 3x3
//! convs over 100-frame chunks, sinusoidal positions restarting in every chunk,
//! encoder layers that attend within fixed windows of chunks, then two linears
//! into the text width. Activations stay f32 over the bf16 weights, as in the
//! reference, whose f32 features promote every op.

const std = @import("std");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const ltx_audio = @import("ltx_audio.zig");
const log = @import("log.zig");
const Scope = @import("mlx_scope.zig").Scope;

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const Config = model_mod.QwenAudioConfig;

pub const SAMPLE_RATE: u32 = 16000;
const N_FFT = 400;
const HOP = 160;
/// mlx-audio pads a shorter clip to one second.
const MIN_SAMPLES = SAMPLE_RATE;
/// Conv chunks per dispatch group: bounds the first conv's f32 output (~200 MB).
const CONV_GROUP = 32;
const LN_EPS = 1e-5;

/// Output length of one 3x3, stride-2, pad-1 conv.
fn convLen(n: usize) usize {
    return (n - 1) / 2 + 1;
}

fn convLen3(n: usize) usize {
    return convLen(convLen(convLen(n)));
}

/// Soft tokens for `frames` mel frames cut into `chunk`-frame conv chunks
/// (the reference's `_get_feat_extract_output_lengths` at chunk 100).
pub fn tokenCount(frames: usize, chunk: usize) usize {
    const rem = frames % chunk;
    return frames / chunk * convLen3(chunk) + if (rem > 0) convLen3(rem) else 0;
}

/// Soft tokens for `samples` of 16 kHz audio.
pub fn tokensForSamples(samples: usize, cfg: Config) usize {
    return tokenCount(@max(samples, MIN_SAMPLES) / HOP, 2 * cfg.n_window);
}

const Lin = struct { w: A, b: ?A };

const Layer = struct {
    ln1_w: A,
    ln1_b: A,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln2_w: A,
    ln2_b: A,
    fc1: Lin,
    fc2: Lin,
};

pub const QwenAudio = struct {
    cfg: Config,
    s: S,
    allocator: std.mem.Allocator,
    convs: [3]Lin,
    conv_out: A,
    layers: []Layer,
    ln_post_w: A,
    ln_post_b: A,
    proj1: Lin,
    proj2: Lin,
    // Owned: the Slaney filterbank `[n_fft/2+1, mels]`, the periodic Hann window,
    // and the position rows one conv chunk reaches `[convLen3(chunk), d_model]`.
    mel_t: A,
    window: A,
    pos: A,

    /// Weight handles are BORROWED from `weights`.
    pub fn init(allocator: std.mem.Allocator, cfg: Config, weights: *const model_mod.Weights) !QwenAudio {
        const s = mlx.gpuStream();
        const layers = try allocator.alloc(Layer, cfg.layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*l, i| l.* = .{
            .ln1_w = try must(weights, "layers.{d}.self_attn_layer_norm.weight", .{i}),
            .ln1_b = try must(weights, "layers.{d}.self_attn_layer_norm.bias", .{i}),
            .q = try lin(weights, "layers.{d}.self_attn.q_proj", .{i}),
            .k = try lin(weights, "layers.{d}.self_attn.k_proj", .{i}),
            .v = try lin(weights, "layers.{d}.self_attn.v_proj", .{i}),
            .o = try lin(weights, "layers.{d}.self_attn.out_proj", .{i}),
            .ln2_w = try must(weights, "layers.{d}.final_layer_norm.weight", .{i}),
            .ln2_b = try must(weights, "layers.{d}.final_layer_norm.bias", .{i}),
            .fc1 = try lin(weights, "layers.{d}.fc1", .{i}),
            .fc2 = try lin(weights, "layers.{d}.fc2", .{i}),
        };
        var self = QwenAudio{
            .cfg = cfg,
            .s = s,
            .allocator = allocator,
            .convs = .{ try lin(weights, "conv2d1", .{}), try lin(weights, "conv2d2", .{}), try lin(weights, "conv2d3", .{}) },
            .conv_out = try must(weights, "conv_out.weight", .{}),
            .layers = layers,
            .ln_post_w = try must(weights, "ln_post.weight", .{}),
            .ln_post_b = try must(weights, "ln_post.bias", .{}),
            .proj1 = try lin(weights, "proj1", .{}),
            .proj2 = try lin(weights, "proj2", .{}),
            .mel_t = .{ .ctx = null },
            .window = .{ .ctx = null },
            .pos = .{ .ctx = null },
        };
        errdefer self.freeOwned();
        try self.buildTables();
        try self.evalWeights();
        log.info("Audio encoder: Qwen3-ASR (layers={d}, d_model={d}, heads={d}, out={d})\n", .{ cfg.layers, cfg.d_model, cfg.heads, cfg.output_dim });
        return self;
    }

    pub fn deinit(self: *QwenAudio) void {
        self.freeOwned();
        self.allocator.free(self.layers);
    }

    fn freeOwned(self: *QwenAudio) void {
        for ([_]A{ self.mel_t, self.window, self.pos }) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        }
    }

    fn buildTables(self: *QwenAudio) !void {
        const a = self.allocator;
        const fb = try ltx_audio.slaneyMelFilterbank(a, self.cfg.mel_bins, N_FFT, SAMPLE_RATE);
        defer _ = mlx.mlx_array_free(fb);
        self.mel_t = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_transpose_axes(&self.mel_t, fb, &[_]c_int{ 1, 0 }, 2, self.s));

        var win: [N_FFT]f32 = undefined;
        for (&win, 0..) |*w, i| w.* = @floatCast(0.5 * (1.0 - @cos(2.0 * std.math.pi * @as(f64, @floatFromInt(i)) / N_FFT)));
        self.window = mlx.mlx_array_new_data(&win, &[_]c_int{N_FFT}, 1, .float32);

        // SinusoidalPositionEmbedding: [sin | cos] of p * exp(-i * ln(10000) / (d/2 - 1)).
        const rows = convLen3(2 * self.cfg.n_window);
        const d: usize = self.cfg.d_model;
        const half = d / 2;
        const tbl = try a.alloc(f32, rows * d);
        defer a.free(tbl);
        const inc = @log(10000.0) / @as(f64, @floatFromInt(half - 1));
        for (0..rows) |p| for (0..half) |i| {
            const t = @as(f64, @floatFromInt(p)) * @exp(-inc * @as(f64, @floatFromInt(i)));
            tbl[p * d + i] = @floatCast(@sin(t));
            tbl[p * d + half + i] = @floatCast(@cos(t));
        };
        self.pos = mlx.mlx_array_new_data(tbl.ptr, &[_]c_int{ @intCast(rows), @intCast(d) }, 2, .float32);
    }

    fn evalWeights(self: *QwenAudio) !void {
        var list = std.ArrayList(A).empty;
        defer list.deinit(self.allocator);
        const a = self.allocator;
        try list.appendSlice(a, &.{ self.conv_out, self.ln_post_w, self.ln_post_b, self.mel_t, self.window, self.pos });
        for (self.convs) |l| try appendLin(a, &list, l);
        for ([_]Lin{ self.proj1, self.proj2 }) |l| try appendLin(a, &list, l);
        for (self.layers) |l| {
            try list.appendSlice(a, &.{ l.ln1_w, l.ln1_b, l.ln2_w, l.ln2_b });
            for ([_]Lin{ l.q, l.k, l.v, l.o, l.fc1, l.fc2 }) |x| try appendLin(a, &list, x);
        }
        const vec = mlx.mlx_vector_array_new_data(list.items.ptr, list.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
    }

    /// Soft tokens `[1, n, output_dim]` (bf16) for 16 kHz mono f32-LE PCM.
    pub fn forward(self: *QwenAudio, pcm_bytes: []const u8) !A {
        const pcm = try self.allocator.alloc(f32, pcm_bytes.len / 4);
        defer self.allocator.free(pcm);
        @memcpy(std.mem.sliceAsBytes(pcm), pcm_bytes[0 .. pcm.len * 4]);
        const feats = try self.features(pcm);
        defer _ = mlx.mlx_array_free(feats);
        const rows = try self.encode(feats);
        defer _ = mlx.mlx_array_free(rows);
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const sh = mlx.getShape(rows);
        const out = try sc.reshape(try sc.astype(rows, .bfloat16), &.{ 1, sh[0], sh[1] });
        return sc.out(out);
    }

    /// WhisperFeatureExtractor: log-mel `[mels, samples / 160]` (f32), the
    /// reference's own normalization (8 dB below the clip's peak, then (x+4)/4).
    pub fn features(self: *QwenAudio, pcm: []const f32) !A {
        const a = self.allocator;
        const n = @max(pcm.len, MIN_SAMPLES);
        const buf = try a.alloc(f32, n);
        defer a.free(buf);
        @memset(buf, 0);
        @memcpy(buf[0..pcm.len], pcm);
        const padded = try ltx_audio.reflectPadCenterF32(a, buf, N_FFT / 2);
        defer a.free(padded);
        // The centered STFT has n/160 + 1 frames; the extractor drops the last.
        const frames = n / HOP;

        var sc = Scope.init(a, self.s);
        defer sc.deinit();
        const wave = try sc.fromF32(padded, &.{@intCast(padded.len)});
        var fr = mlx.mlx_array_new();
        const framed = try sc.res(mlx.mlx_as_strided(&fr, wave, &[_]c_int{ @intCast(frames), N_FFT }, 2, &[_]i64{ HOP, 1 }, 2, 0, self.s), &fr);
        var sp = mlx.mlx_array_new();
        const spec = try sc.res(mlx.mlx_fft_rfft(&sp, try sc.mul(framed, self.window), N_FFT, 1, mlx.MLX_FFT_NORM_BACKWARD, self.s), &sp);
        var mg = mlx.mlx_array_new();
        const mag = try sc.res(mlx.mlx_abs(&mg, spec, self.s), &mg);
        const power = try sc.mul(mag, mag);
        var cl = mlx.mlx_array_new();
        const mel = try sc.res(mlx.mlx_maximum(&cl, try sc.matmul(power, self.mel_t), try sc.scalar(1e-10), self.s), &cl);
        var lg = mlx.mlx_array_new();
        const log10 = try sc.mulS(try sc.res(mlx.mlx_log(&lg, mel, self.s), &lg), 1.0 / @log(10.0));
        var mx = mlx.mlx_array_new();
        const peak = try sc.res(mlx.mlx_max(&mx, log10, false, self.s), &mx);
        var fl = mlx.mlx_array_new();
        const floored = try sc.res(mlx.mlx_maximum(&fl, log10, try sc.addS(peak, -8.0), self.s), &fl);
        const norm = try sc.mulS(try sc.addS(floored, 4.0), 0.25);
        const out = sc.out(try sc.transpose(norm, &.{ 1, 0 }));
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    /// The tower over log-mel `[mels, frames]`: soft tokens `[n, output_dim]` (f32).
    pub fn encode(self: *QwenAudio, feats: A) !A {
        const cfg = self.cfg;
        const chunk: usize = 2 * cfg.n_window;
        const frames: usize = @intCast(mlx.getShape(feats)[1]);
        const n_tok = tokenCount(frames, chunk);
        const n_chunks = (frames + chunk - 1) / chunk;
        const per_chunk = convLen3(chunk);

        var x = try self.frontend(feats, frames, n_chunks);
        defer _ = mlx.mlx_array_free(x);

        // Windows of whole chunks attend only within themselves (the reference's block mask).
        const win = convLen3(@min(frames, chunk)) * (cfg.n_window_infer / chunk);
        const n_win = (n_tok + win - 1) / win;
        const d: c_int = @intCast(cfg.d_model);
        var mask: A = .{ .ctx = null };
        defer if (mask.ctx != null) {
            _ = mlx.mlx_array_free(mask);
        };
        {
            var sc = Scope.init(self.allocator, self.s);
            defer sc.deinit();
            var rows = try sc.slice(try sc.reshape(x, &.{ @intCast(n_chunks * per_chunk), d }), 0, 0, @intCast(n_tok));
            if (n_win * win > n_tok) {
                rows = try sc.padAxis(rows, 0, 0, @intCast(n_win * win - n_tok));
                const keep = try self.allocator.alloc(bool, n_win * win);
                defer self.allocator.free(keep);
                for (keep, 0..) |*k, i| k.* = i < n_tok;
                mask = mlx.mlx_array_new_data(keep.ptr, &[_]c_int{ @intCast(n_win), 1, 1, @intCast(win) }, 4, .bool_);
            }
            const windows = sc.out(try sc.reshape(rows, &.{ @intCast(n_win), @intCast(win), d }));
            _ = mlx.mlx_array_free(x);
            x = windows;
        }
        for (self.layers) |*l| {
            const next = try self.layer(l, x, mask);
            _ = mlx.mlx_array_free(x);
            x = next;
            try mlx.check(mlx.mlx_array_eval(x));
        }

        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const rows = try sc.slice(try sc.reshape(x, &.{ @intCast(n_win * win), d }), 0, 0, @intCast(n_tok));
        const h = try sc.layerNorm(rows, self.ln_post_w, self.ln_post_b, LN_EPS);
        const p1 = try sc.geluErf(try sc.lin(h, self.proj1.w, self.proj1.b));
        const out = sc.out(try sc.lin(p1, self.proj2.w, self.proj2.b));
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    /// Conv chunks -> `[n_chunks, convLen3(chunk), d_model]` with positions added,
    /// CONV_GROUP chunks per dispatch.
    fn frontend(self: *QwenAudio, feats: A, frames: usize, n_chunks: usize) !A {
        const chunk: usize = 2 * self.cfg.n_window;
        const mels: c_int = @intCast(self.cfg.mel_bins);
        var parts = std.ArrayList(A).empty;
        defer {
            for (parts.items) |p| _ = mlx.mlx_array_free(p);
            parts.deinit(self.allocator);
        }
        {
            var sc = Scope.init(self.allocator, self.s);
            defer sc.deinit();
            // [mels, frames] -> zero-padded whole chunks [n_chunks, mels, chunk, 1] (NHWC).
            const padded = try sc.padAxis(feats, 1, 0, @intCast(n_chunks * chunk - frames));
            const chunks = try sc.reshape(try sc.transpose(try sc.reshape(padded, &.{ mels, @intCast(n_chunks), @intCast(chunk) }), &.{ 1, 0, 2 }), &.{ @intCast(n_chunks), mels, @intCast(chunk), 1 });
            var g: usize = 0;
            while (g < n_chunks) : (g += CONV_GROUP) {
                var gs = Scope.init(self.allocator, self.s);
                defer gs.deinit();
                var x = try gs.slice(chunks, 0, @intCast(g), @intCast(@min(g + CONV_GROUP, n_chunks)));
                for (self.convs) |c| x = try gs.geluErf(try gs.add(try gs.conv2d(x, c.w, 2, 1), c.b.?));
                // [b, freq, time, ch] -> [b, time, ch * freq] -> conv_out, + positions.
                const sh = mlx.getShape(x);
                const flat = try gs.reshape(try gs.transpose(x, &.{ 0, 2, 3, 1 }), &.{ sh[0], sh[2], sh[3] * sh[1] });
                const part = gs.out(try gs.add(try gs.lin(flat, self.conv_out, null), self.pos));
                parts.append(self.allocator, part) catch |e| {
                    _ = mlx.mlx_array_free(part);
                    return e;
                };
                try mlx.check(mlx.mlx_array_eval(part));
            }
        }
        if (parts.items.len == 1) {
            const only = parts.items[0];
            parts.clearRetainingCapacity();
            return only;
        }
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        return sc.out(try sc.concat(parts.items, 0));
    }

    /// One pre-LN encoder layer over `[windows, win, d]`.
    fn layer(self: *QwenAudio, l: *const Layer, x: A, mask: A) !A {
        const hd: c_int = @intCast(self.cfg.d_model / self.cfg.heads);
        const heads: c_int = @intCast(self.cfg.heads);
        var sc = Scope.init(self.allocator, self.s);
        defer sc.deinit();
        const h = try sc.layerNorm(x, l.ln1_w, l.ln1_b, LN_EPS);
        const scaling: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
        const q = try sc.heads(try sc.mulS(try sc.lin(h, l.q.w, l.q.b), scaling), heads, hd);
        const k = try sc.heads(try sc.lin(h, l.k.w, l.k.b), heads, hd);
        const v = try sc.heads(try sc.lin(h, l.v.w, l.v.b), heads, hd);
        const att = try sc.sdpaMasked(q, k, v, 1.0, mask);
        const x1 = try sc.add(x, try sc.lin(try sc.merge(att), l.o.w, l.o.b));
        const h2 = try sc.layerNorm(x1, l.ln2_w, l.ln2_b, LN_EPS);
        const ff = try sc.lin(try sc.geluErf(try sc.lin(h2, l.fc1.w, l.fc1.b)), l.fc2.w, l.fc2.b);
        return sc.out(try sc.add(x1, ff));
    }
};

fn appendLin(a: std.mem.Allocator, list: *std.ArrayList(A), l: Lin) !void {
    try list.append(a, l.w);
    if (l.b) |b| try list.append(a, b);
}

fn must(weights: *const model_mod.Weights, comptime fmt: []const u8, args: anytype) !A {
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "audio_tower." ++ fmt, args) catch return error.MissingWeight;
    return weights.get(key) orelse {
        log.warn("MISSING AUDIO TOWER WEIGHT: {s}\n", .{key});
        return error.MissingWeight;
    };
}

fn lin(weights: *const model_mod.Weights, comptime fmt: []const u8, args: anytype) !Lin {
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "audio_tower." ++ fmt ++ ".bias", args) catch return error.MissingWeight;
    return .{ .w = try must(weights, fmt ++ ".weight", args), .b = weights.get(key) };
}

// ── Tests ──

const testing = std.testing;

test "tokenCount equals the reference's _get_feat_extract_output_lengths" {
    const cases = [_][2]usize{
        .{ 1, 1 },   .{ 2, 1 },   .{ 3, 1 },   .{ 7, 1 },    .{ 8, 1 },     .{ 9, 2 },        .{ 99, 13 },
        .{ 100, 13 }, .{ 101, 14 }, .{ 180, 23 }, .{ 199, 26 }, .{ 200, 26 }, .{ 680, 88 }, .{ 2113, 275 },
        .{ 120000, 15600 },
    };
    for (cases) |c| try testing.expectEqual(c[1], tokenCount(c[0], 100));
    // A clip under a second is padded to one first, as mlx-audio does.
    try testing.expectEqual(@as(usize, 13), tokensForSamples(800, .{}));
}

// Parity vs the EXECUTED reference: mlx-audio's qwen3_asr on the pack's own
// weights (tests/dump_qwen3_asr_fixtures.py), so a diff is a layout/math bug.
//
//   QWEN3_ASR_MODEL=~/.mlx-serve/models/mlx-community/Qwen3-ASR-0.6B-8bit \
//   QWEN3_ASR_FIXTURE=~/claude-tmp/qwen3-asr/speech_fixture.safetensors \
//   zig build test -Doptimize=ReleaseFast -Dtest-filter="qwen3-asr"
fn envPath(name: [*:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    const p = std.mem.sliceTo(raw, 0);
    return if (p.len == 0) null else p;
}

test "qwen3-asr: features and audio tower match the reference" {
    const fix_path = envPath("QWEN3_ASR_FIXTURE") orelse return error.SkipZigTest;
    const model_dir = envPath("QWEN3_ASR_MODEL") orelse return error.SkipZigTest;
    const a = testing.allocator;
    var config = try model_mod.parseConfig(testing.io, a, model_dir);
    defer config.deinit(a);
    var weights = try model_mod.loadModelWeights(testing.io, a, model_dir, &config, true);
    defer weights.deinit();
    var fx = try model_mod.loadWeightsSingleFile(a, fix_path);
    defer fx.deinit();
    var qa = try QwenAudio.init(a, config.qwen_audio.?, &weights);
    defer qa.deinit();

    const pcm_arr = fx.get("pcm") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(pcm_arr));
    const pcm = mlx.mlx_array_data_float32(pcm_arr).?[0..mlx.mlx_array_size(pcm_arr)];
    const feats = try qa.features(pcm);
    defer _ = mlx.mlx_array_free(feats);
    const want_f = fx.get("features") orelse return error.MissingFixtureTensor;
    try testing.expectEqualSlices(c_int, mlx.getShape(want_f), mlx.getShape(feats));
    const fd = try maxAbsDiff(feats, want_f, qa.s);
    std.debug.print("[qwen3-asr] features {any} max|diff|={e}\n", .{ mlx.getShape(feats), fd });
    // f32 GPU FFT against numpy's f64 one, after a log and a /4.
    try testing.expect(fd < 1e-3);

    // The tower on the reference's own features, so its diff is the tower's alone.
    const rows = try qa.encode(want_f);
    defer _ = mlx.mlx_array_free(rows);
    const want = fx.get("audio_embeds") orelse return error.MissingFixtureTensor;
    try testing.expectEqualSlices(c_int, mlx.getShape(want), mlx.getShape(rows));
    const c = try cosine(rows, want, qa.s);
    const r = try rmsRatio(rows, want, qa.s);
    std.debug.print("[qwen3-asr] audio_embeds {any} cos={d:.6} rms_ratio={d:.5}\n", .{ mlx.getShape(rows), c, r });
    // Same weights, same f32 math, a different op order: near-identical.
    try testing.expect(c > 0.9999 and r > 0.999 and r < 1.001);
}

fn reduceSum(x: A, s: S) !f32 {
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    var o = mlx.mlx_array_new();
    const t = try sc.res(mlx.mlx_sum(&o, try sc.astype(x, .float32), false, s), &o);
    try mlx.check(mlx.mlx_array_eval(t));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, t));
    return v;
}

fn maxAbsDiff(x: A, y: A, s: S) !f32 {
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    var ab = mlx.mlx_array_new();
    const d = try sc.res(mlx.mlx_abs(&ab, try sc.sub(try sc.astype(x, .float32), try sc.astype(y, .float32)), s), &ab);
    var m = mlx.mlx_array_new();
    const t = try sc.res(mlx.mlx_max(&m, d, false, s), &m);
    try mlx.check(mlx.mlx_array_eval(t));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, t));
    return v;
}

fn cosine(x: A, y: A, s: S) !f32 {
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const xf = try sc.astype(x, .float32);
    const yf = try sc.astype(y, .float32);
    const dot = try reduceSum(try sc.mul(xf, yf), s);
    return dot / @sqrt(try reduceSum(try sc.mul(xf, xf), s) * try reduceSum(try sc.mul(yf, yf), s));
}

fn rmsRatio(x: A, y: A, s: S) !f32 {
    var sc = Scope.init(testing.allocator, s);
    defer sc.deinit();
    const xf = try sc.astype(x, .float32);
    const yf = try sc.astype(y, .float32);
    return @sqrt(try reduceSum(try sc.mul(xf, xf), s) / try reduceSum(try sc.mul(yf, yf), s));
}
