//! LFM3's speech codec decoder (the pack's `audio_tokenizer/`, mlx-vlm's `lfm3_audio/codec.py`): each frame's codes
//! dequantize by FSQ (a code is one digit per level), project to the decoder width and repeat `stft_frame_stack`
//! times (12.5 Hz frames into 75 Hz STFT frames); an 8-layer LFM3 trunk (`trunk.zig`: convs and windowed causal
//! attention, sandwich norms) and a linear give each STFT frame's log-magnitude and phase; an inverse STFT overlap-adds
//! them into 24 kHz ("same" padding: trimmed both ends, divided by the window envelope). The trunk is causal with a
//! bounded reach, so a `Stream` decodes each chunk over a fixed left context and equals the whole decode.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const reader = @import("../../sdk/bases/reader.zig");
const load_weights = @import("../../sdk/weights/load.zig");
const Resident = @import("../../sdk/weights/resident.zig").Resident;
const trunk_mod = @import("trunk.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const gpa = std.heap.c_allocator;

pub const dir_name = "audio_tokenizer";
pub const max_levels = 8;

pub const Codec = struct {
    levels: [max_levels]u32 = @splat(0),
    n_levels: u32 = 0,
    n_q: u32 = 12,
    stack: u32 = 6,
    n_fft: u32 = 1280,
    hop: u32 = 320,
    sample_rate: u32 = 24000,
    /// The decoder's attention window in STFT frames (`decoder_window`).
    window_rows: u32 = 30,
    /// Earlier frames a chunk is decoded over: the trunk's reach, in frames.
    context: u32 = 0,
    project_out: Linear = .{},
    from_quantizer: Linear = .{},
    trunk: trunk_mod.Trunk = undefined,
    lin: Linear = .{},
    window: Array = .{},
    s: mlx.mlx_stream = .{ .ctx = null },

    /// The codec beside the model (`audio_tokenizer/`): its config and decode-side tensors; null when absent.
    pub fn load(ctx: *const abi.LoadCtx, owned: *reader.Owned) !?Codec {
        if (!ctx.dir.exists(ctx.dir.ctx, abi.Str.of(dir_name ++ "/model.safetensors"))) return null;
        var c: Codec = .{ .s = .{ .ctx = ctx.stream.ctx } };
        try c.readConfig(ctx.dir);
        var path: [std.fs.max_path_bytes]u8 = undefined;
        var res = Resident.adopt(try load_weights.loadWeightsSingleFile(gpa, try std.fmt.bufPrint(&path, "{s}/" ++ dir_name ++ "/model.safetensors", .{ctx.model_dir.slice()})));
        defer res.deinit();
        const src = res.source();
        var lctx = ctx.*;
        lctx.weights = &src;
        var r = try reader.Reader.open(&lctx, "{}", "decoder");
        defer r.close();
        // The reference names FSQ's projection `quantizer.fsq.project_out` before its sanitize.
        const fsq = if (r.has("quantizer.project_out.weight")) "quantizer.project_out" else "quantizer.fsq.project_out";
        c.project_out = try biased(&r, owned, fsq, c.n_q * c.n_levels);
        c.from_quantizer = try biased(&r, owned, "from_quantizer", @intCast(ops.shape(c.project_out.bias)[0]));
        const qdim: u32 = @intCast(ops.shape(c.from_quantizer.bias)[0]);
        // The backbone's shape is fixed by the reference class: sliding layers 2, 4, 6, convs elsewhere.
        var attention: [8]bool = @splat(false);
        for ([_]usize{ 2, 4, 6 }) |i| attention[i] = true;
        c.trunk = try trunk_mod.Trunk.load(&r, owned, "decoder.lfm", &attention, qdim, 1e-5, 1e6, .{ .causal = c.window_rows });
        errdefer c.trunk.deinit();
        c.lin = try biased(&r, owned, "decoder.lin", qdim);
        const window = try r.need("decoder.istft.window");
        c.window = window;
        try owned.own(window);
        // Each conv layer reaches its kernel minus one rows back, each attention layer its window minus one.
        var reach: u32 = 0;
        for (c.trunk.layers) |L| reach += switch (L.mixer) {
            .conv => |cv| @as(u32, @intCast(ops.shape(cv.conv)[1])) - 1,
            .attn => c.window_rows - 1,
        };
        c.context = (reach + c.stack - 1) / c.stack;
        return c;
    }

    fn readConfig(c: *Codec, dir: *const abi.PackDir) !void {
        var buf: [1 << 15]u8 = undefined;
        var why: abi.Verdict = .{};
        const bytes = dir.readFile(dir_name ++ "/config.json", &buf, &why) orelse return error.CodecConfigUnreadable;
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.CodecConfigUnreadable;
        const o = parsed.value.object;
        const int = struct {
            fn f(obj: std.json.ObjectMap, k: []const u8, d: u32) u32 {
                const v = obj.get(k) orelse return d;
                return if (v == .integer and v.integer > 0) @intCast(v.integer) else d;
            }
        }.f;
        c.n_q = int(o, "n_q", 12);
        c.stack = int(o, "stft_frame_stack", 6);
        c.n_fft = int(o, "stft_n_fft", 1280);
        c.hop = int(o, "stft_hop_length", 320);
        c.window_rows = int(o, "decoder_window", 30);
        c.sample_rate = int(o, "sampling_rate", 24000);
        if (int(o, "stft_win_length", c.n_fft) != c.n_fft or c.n_fft % c.hop != 0) return error.UnsupportedCodec;
        const levels = o.get("fsq_levels") orelse return error.UnsupportedCodec;
        if (levels != .array or levels.array.items.len == 0 or levels.array.items.len > max_levels) return error.UnsupportedCodec;
        for (levels.array.items, 0..) |l, i| c.levels[i] = if (l == .integer and l.integer > 1) @intCast(l.integer) else return error.UnsupportedCodec;
        c.n_levels = @intCast(levels.array.items.len);
    }

    pub fn deinit(c: *Codec) void {
        c.trunk.deinit();
    }

    /// Codes of one frame range [0, codebook_size) of the codec: `prod(levels)`.
    pub fn codebookSize(c: *const Codec) u32 {
        var n: u32 = 1;
        for (c.levels[0..c.n_levels]) |l| n *= l;
        return n;
    }

    /// FSQ's latents `[frames, n_q * levels]` (f32, host) of `codes` (`frames x n_q`, frame after frame).
    fn latents(c: *const Codec, a: std.mem.Allocator, codes: []const u32) ![]f32 {
        const nl = c.n_levels;
        const out = try a.alloc(f32, codes.len * nl);
        for (codes, 0..) |code, i| {
            var basis: u32 = 1;
            for (c.levels[0..nl], 0..) |l, j| {
                const half: f32 = @floatFromInt(l / 2);
                out[i * nl + j] = (@as(f32, @floatFromInt((code / basis) % l)) - half) / half;
                basis *= l;
            }
        }
        return out;
    }

    /// The STFT frames `[rows, n_fft]` (f32, windowed time domain, host) of frames `[from, codes.len / n_q)`, decoded
    /// over `context` frames before `from`.
    fn stftFrames(c: *const Codec, a: std.mem.Allocator, codes: []const u32, from: usize) ![]f32 {
        const total = codes.len / c.n_q;
        const start = from -| c.context;
        var g = Graph.init(gpa, c.s);
        defer g.deinit();
        const lat = try c.latents(a, codes[start * c.n_q ..]);
        defer a.free(lat);
        const frames: c_int = @intCast(total - start);
        // The two projections run in f32 and round once into the trunk, which amplifies its input's rounding.
        const dt = mlx.mlx_array_dtype(.{ .ctx = c.project_out.w.ctx });
        var x = try g.fromSlice(f32, lat, &.{ 1, frames, @intCast(c.n_q * c.n_levels) });
        x = try g.astype(try c.from_quantizer.forward(&g, try c.project_out.forward(&g, x)), dt);
        x = try g.repeat(x, @intCast(c.stack), 1);
        const h = try c.trunk.forward(&g, x, @intCast(start * c.stack));
        const keep_from: c_int = @intCast((from - start) * c.stack);
        const rows = try g.sliceAxis(try g.astype(try c.lin.forward(&g, h), .float32), 1, keep_from, frames * @as(c_int, @intCast(c.stack)));
        const bins: c_int = @intCast(c.n_fft / 2 + 1);
        const mag = try g.exp(try g.sliceAxis(rows, 2, 0, bins));
        const angle = try g.sliceAxis(rows, 2, bins, 2 * bins);
        const re = try g.mul(mag, try g.cos(angle));
        const im = try g.mul(mag, try g.sin(angle));
        const i_unit = try g.adopt(mlx.mlx_array_new_complex(0, 1));
        const spec = try g.add(re, try g.mul(im, i_unit));
        const time = try g.call(mlx.mlx_fft_irfft, .{ spec, @as(c_int, @intCast(c.n_fft)), @as(c_int, -1), mlx.MLX_FFT_NORM_BACKWARD });
        const windowed = try g.mul(time, try g.astype(c.window, .float32));
        try mlx.check(mlx.mlx_array_eval(windowed));
        const n: usize = @intCast(mlx.mlx_array_size(windowed));
        const data = mlx.mlx_array_data_float32(windowed) orelse return error.NoData;
        return a.dupe(f32, data[0..n]);
    }

    /// The whole utterance's 24 kHz samples (`frames * stack * hop`), as the reference's `decode`.
    pub fn decode(c: *const Codec, a: std.mem.Allocator, codes: []const u32) ![]f32 {
        var st = try Stream.init(a, c);
        defer st.deinit();
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(a);
        try st.push(codes, &out);
        try st.finish(&out);
        return out.toOwnedSlice(a);
    }
};

/// A streaming decode: frames pushed as they arrive, samples appended as they become final (no later frame reaches
/// them), the trimmed tail on `finish`.
pub const Stream = struct {
    a: std.mem.Allocator,
    c: *const Codec,
    codes: std.ArrayList(u32) = .empty,
    /// The overlap-add of every STFT frame so far, and the window envelope under it, from sample 0 (untrimmed).
    audio: std.ArrayList(f32) = .empty,
    envelope: std.ArrayList(f32) = .empty,
    /// Untrimmed samples handed out so far.
    emitted: usize = 0,
    stft_frames: usize = 0,

    pub fn init(a: std.mem.Allocator, c: *const Codec) !Stream {
        return .{ .a = a, .c = c };
    }

    pub fn deinit(st: *Stream) void {
        st.codes.deinit(st.a);
        st.audio.deinit(st.a);
        st.envelope.deinit(st.a);
    }

    fn trim(st: *const Stream) usize {
        return (st.c.n_fft - st.c.hop) / 2;
    }

    /// Decodes `codes` (whole frames of `n_q` codes) and appends the samples that became final to `out`.
    pub fn push(st: *Stream, codes: []const u32, out: *std.ArrayList(f32)) !void {
        if (codes.len == 0) return;
        if (codes.len % st.c.n_q != 0) return error.PartialFrame;
        const from = st.codes.items.len / st.c.n_q;
        try st.codes.appendSlice(st.a, codes);
        const frames = try st.c.stftFrames(st.a, st.codes.items, from);
        defer st.a.free(frames);
        const n_fft = st.c.n_fft;
        const hop = st.c.hop;
        const window = try st.windowSquared();
        defer st.a.free(window);
        const rows = frames.len / n_fft;
        for (0..rows) |r| {
            const at = (st.stft_frames + r) * hop;
            const need = at + n_fft;
            if (st.audio.items.len < need) {
                try st.audio.appendNTimes(st.a, 0, need - st.audio.items.len);
                try st.envelope.appendNTimes(st.a, 0, need - st.envelope.items.len);
            }
            for (0..n_fft) |i| {
                st.audio.items[at + i] += frames[r * n_fft + i];
                st.envelope.items[at + i] += window[i];
            }
        }
        st.stft_frames += rows;
        // A sample before the next STFT frame's start is reached by no later frame.
        try st.emit(st.stft_frames * hop, out);
    }

    /// Appends the samples still held, up to the end less the trim.
    pub fn finish(st: *Stream, out: *std.ArrayList(f32)) !void {
        try st.emit(st.audio.items.len -| st.trim(), out);
    }

    fn emit(st: *Stream, upto: usize, out: *std.ArrayList(f32)) !void {
        const from = @max(st.emitted, st.trim());
        if (upto <= from) return;
        for (st.audio.items[from..upto], st.envelope.items[from..upto]) |s, e| try out.append(st.a, s / e);
        st.emitted = upto;
    }

    fn windowSquared(st: *const Stream) ![]f32 {
        const w = st.c.window;
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, .{ .ctx = w.ctx }, .float32, st.c.s));
        try mlx.check(mlx.mlx_array_eval(f));
        const data = mlx.mlx_array_data_float32(f) orelse return error.NoData;
        const out = try st.a.alloc(f32, st.c.n_fft);
        for (out, data[0..st.c.n_fft]) |*o, x| o.* = x * x;
        return out;
    }
};

fn biased(r: *reader.Reader, owned: *reader.Owned, base: []const u8, in_dim: u32) !Linear {
    var l = try r.proj(base, in_dim);
    var nb: [160]u8 = undefined;
    l.bias = try r.need(try std.fmt.bufPrint(&nb, "{s}.bias", .{base}));
    try owned.linear(l);
    return l;
}
