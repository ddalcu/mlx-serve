//! Nemotron 3.5 Transcribe's perception module (mlx-audio's `nemotron_transcribe/perception.py`, NeMo's
//! `ParallelExpertEncoderPT`): NeMo log-mels, a feature-stacking RoPE Transformer (the ASR expert, fed the mels
//! normalized per feature over the whole clip) and a Sortformer speaker encoder (the speaker expert, fed the RAW mels:
//! `perception.py` normalizes for the ASR expert only) over 40 s windows with context, the speaker activity fused into
//! the ASR rows through a sinusoidal kernel, then a linear projection to the trunk.
//! The speaker half is `nemotron_diarization`'s streaming (FIFO + arrival-order speaker cache) with left context.
//! Activations run in the weights' dtype (bf16), the speaker probabilities and the cache scores in f32.
//! The CTC timestamp head reads the ASR expert's rows (before the fusion's norm) in f32.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const bases = @import("../../sdk/bases/bases.zig");
const reader = @import("../../sdk/bases/reader.zig");
const config = @import("config.zig");
const diarization = @import("../nemotron_diarization/diarization.zig");
const tokenizer_load = @import("../../sdk/tokenizer/load.zig");
const Tokenizer = @import("../../sdk/tokenizer/tokenizer.zig").Tokenizer;
const ctc_align = @import("ctc_align.zig");
const log = @import("../../sdk/util/log.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const Linear = bases.blocks.linear.Linear;
const gpa = std.heap.c_allocator;

const LN_EPS = 1e-5;
const LOG_GUARD = 1.0 / 16777216.0;
const MEL_BLOCK = 4096;

/// Encoder rows of a clip of `samples`: one per `stack` mel frames, the last group padded.
pub fn rows(samples: usize) usize {
    const frames = samples / config.HOP;
    return (frames + config.STACK - 1) / config.STACK;
}

const Layer = struct {
    norm1_w: Array,
    norm1_b: Array,
    qkv: Linear,
    q_norm_w: Array = .{},
    q_norm_b: Array = .{},
    k_norm_w: Array = .{},
    k_norm_b: Array = .{},
    out: Linear,
    norm2_w: Array,
    norm2_b: Array,
    ff1: Linear,
    ff2: Linear,
};

const Encoder = struct {
    cfg: config.Encoder,
    /// The feature-stacking projection; null for the CTC head's bridge (it reads rows, not features).
    stack: ?Linear,
    embed_w: Array,
    embed_b: Array,
    final_w: Array,
    final_b: Array,
    layers: []Layer,

    fn load(r: *reader.Reader, owned: *reader.Owned, base: []const u8, cfg: config.Encoder, stack_in: u32) !Encoder {
        var nb: [192]u8 = undefined;
        const layers = try gpa.alloc(Layer, cfg.layers);
        errdefer gpa.free(layers);
        const d = cfg.d_model;
        for (layers, 0..) |*l, i| {
            l.* = .{
                .norm1_w = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.norm1.weight", .{ base, i })),
                .norm1_b = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.norm1.bias", .{ base, i })),
                .qkv = try lin(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.attn.w_qkv", .{ base, i }), d, false),
                .out = try lin(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.attn.out_proj", .{ base, i }), d, true),
                .norm2_w = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.norm2.weight", .{ base, i })),
                .norm2_b = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.norm2.bias", .{ base, i })),
                .ff1 = try lin(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.ffn.linear1", .{ base, i }), d, true),
                .ff2 = try lin(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.ffn.linear2", .{ base, i }), cfg.ff, true),
            };
            if (cfg.qk_norm) {
                l.q_norm_w = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.attn.q_norm.weight", .{ base, i }));
                l.q_norm_b = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.attn.q_norm.bias", .{ base, i }));
                l.k_norm_w = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.attn.k_norm.weight", .{ base, i }));
                l.k_norm_b = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.layers.{d}.attn.k_norm.bias", .{ base, i }));
            }
        }
        return .{
            .cfg = cfg,
            .stack = if (stack_in == 0) null else try lin(r, owned, try std.fmt.bufPrint(&nb, "{s}.pre_encode.proj", .{base}), stack_in, false),
            .embed_w = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.embed_norm.weight", .{base})),
            .embed_b = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.embed_norm.bias", .{base})),
            .final_w = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.final_norm.weight", .{base})),
            .final_b = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.final_norm.bias", .{base})),
            .layers = layers,
        };
    }

    fn deinit(e: *Encoder) void {
        gpa.free(e.layers);
    }

    /// `[mels, frames]` features to `[1, ceil(frames / stack), d_model]`: `stack` frames side by side, the last group
    /// zero-padded, then the projection.
    fn stackRows(e: *const Encoder, g: *Graph, feats: Array, stack: u32, dt: mlx.mlx_dtype) !Array {
        const sh = ops.shape(feats);
        const pad: c_int = @intCast((stack - @as(u32, @intCast(sh[1])) % stack) % stack);
        var x = try g.transpose(feats, &.{ 1, 0 });
        if (pad > 0) x = try g.padAxis(x, 0, 0, pad);
        const grouped = try g.reshape(x, &.{ 1, @divExact(sh[1] + pad, @as(c_int, @intCast(stack))), sh[0] * @as(c_int, @intCast(stack)) });
        return e.stack.?.forward(g, try g.astype(grouped, dt));
    }

    fn forward(e: *const Encoder, g: *Graph, x: Array) !Array {
        var h = try g.layerNorm(x, e.embed_w, e.embed_b, LN_EPS);
        for (e.layers) |*l| {
            h = try g.add(h, try e.attention(g, l, try g.layerNorm(h, l.norm1_w, l.norm1_b, LN_EPS)));
            const n2 = try g.layerNorm(h, l.norm2_w, l.norm2_b, LN_EPS);
            h = try g.add(h, try l.ff2.forward(g, try gelu(g, try l.ff1.forward(g, n2))));
        }
        return g.layerNorm(h, e.final_w, e.final_b, LN_EPS);
    }

    fn attention(e: *const Encoder, g: *Graph, l: *const Layer, x: Array) !Array {
        const c = e.cfg;
        const t = ops.shape(x)[1];
        const heads: c_int = @intCast(c.heads);
        const hd: c_int = @intCast(c.d_model / c.heads);
        const qkv = try g.reshape(try l.qkv.forward(g, x), &.{ 1, t, 3, heads, hd });
        var parts: [3]Array = undefined;
        for (&parts, 0..) |*p, i| p.* = try g.transpose(try g.reshape(try g.sliceAxis(qkv, 2, @intCast(i), @intCast(i + 1)), &.{ 1, t, heads, hd }), &.{ 0, 2, 1, 3 });
        var q = parts[0];
        var k = parts[1];
        if (c.qk_norm) {
            q = try g.layerNorm(q, l.q_norm_w, l.q_norm_b, LN_EPS);
            k = try g.layerNorm(k, l.k_norm_w, l.k_norm_b, LN_EPS);
        }
        const dims: c_int = @intFromFloat(@as(f32, @floatFromInt(hd)) * c.rotary_fraction);
        q = try g.rope(q, dims, c.rope_base, 0);
        k = try g.rope(k, dims, c.rope_base, 0);
        const o = try g.sdpa(q, k, parts[2], 1 / @sqrt(@as(f32, @floatFromInt(hd))), .none);
        return l.out.forward(g, try g.reshape(try g.transpose(o, &.{ 0, 2, 1, 3 }), &.{ 1, t, @intCast(c.d_model) }));
    }
};

/// Exact GELU in f32, rounded once to the activation dtype.
fn gelu(g: *Graph, x: Array) !Array {
    const dt = ops.dtype(x);
    const f = try g.astype(x, .float32);
    const y = try g.mul(try g.mulScalar(f, 0.5), try g.addScalar(try g.erf(try g.mulScalar(f, std.math.sqrt1_2)), 1.0));
    return g.astype(y, dt);
}

fn relu(g: *Graph, x: Array) !Array {
    return g.maximum(x, try g.scalarLike(x, 0));
}

/// `<spk:N>` as its N; null for any other unit.
pub fn speakerTag(u: []const u8) ?i32 {
    if (u.len < 7 or !std.ascii.eqlIgnoreCase(u[0..5], "<spk:") or u[u.len - 1] != '>') return null;
    return std.fmt.parseInt(i32, u[5 .. u.len - 1], 10) catch null;
}

/// The CTC timestamp head: ASR rows `[1, T, d]` to log-probabilities of the columns a transcript reads.
const Ctc = struct {
    cfg: config.Ctc,
    bridge: Encoder,
    /// ConvTranspose1d weight `[out, k, in]` and bias.
    up_w: Array,
    up_b: Array,
    /// The pointwise projection `[classes + 1, d]` (the blank last) and bias, f32.
    cls_w: Array,
    cls_b: Array,
    /// Null when the pack ships no alignment tokenizer: the head still runs on token ids.
    tok: ?Tokenizer,

    fn load(ctx: *const abi.LoadCtx, r: *reader.Reader, owned: *reader.Owned, cfg: config.Ctc) !Ctc {
        const base = "ctc_timestamp.decoder";
        var c: Ctc = undefined;
        c.cfg = cfg;
        c.bridge = try Encoder.load(r, owned, base ++ ".transformer", cfg.enc, 0);
        errdefer c.bridge.deinit();
        var g = Graph.init(gpa, r.s);
        defer g.deinit();
        // torch's ConvTranspose1d weight is [in, out, k]; MLX's is [out, k, in].
        c.up_w = g.escape(try g.contiguous(try g.transpose(try arr(r, owned, base ++ ".upsample.weight"), &.{ 1, 2, 0 })));
        try owned.own(c.up_w);
        c.up_b = try arr(r, owned, base ++ ".upsample.bias");
        const w = try arr(r, owned, base ++ ".decoder_layers.0.weight");
        const sh = ops.shape(w);
        c.cls_w = g.escape(try g.reshape(w, &.{ sh[0], sh[1] }));
        try owned.own(c.cls_w);
        c.cls_b = try arr(r, owned, base ++ ".decoder_layers.0.bias");
        if (sh[0] != cfg.classes + 1) return error.CtcClasses;
        try ops.eval(c.up_w);
        try ops.eval(c.cls_w);
        const dir = try std.fmt.allocPrint(gpa, "{s}/alignment_tokenizer", .{ctx.model_dir.slice()});
        defer gpa.free(dir);
        c.tok = tokenizer_load.loadTokenizer(std.Io.Threaded.global_single_threaded.io(), gpa, dir) catch null;
        return c;
    }

    fn deinit(c: *Ctc) void {
        c.bridge.deinit();
        if (c.tok) |*tk| tk.deinit();
    }

    /// Log-probabilities `[1, 2T, cols]` f32 of the blank and `keep` over ASR rows `states` `[1, T, d]` (a window).
    fn columns(c: *const Ctc, g: *Graph, states: Array, keep: Array) !Array {
        const x = try g.astype(states, .float32);
        const up = try g.add(try g.convTranspose1d(x, c.up_w, @intCast(c.cfg.upsample), 0, 1, 0, 1), c.up_b);
        const bridged = try c.bridge.forward(g, up);
        const mixed = try g.add(up, try g.mulScalar(bridged, c.cfg.residual));
        const logits = try g.add(try g.matmul(mixed, try g.transpose(c.cls_w, &.{ 1, 0 })), c.cls_b);
        const lse = try g.call(mlx.mlx_logsumexp_axis, .{ logits, @as(c_int, 2), true });
        return g.sub(try g.take(logits, keep, 2), lse);
    }
};

pub const Tower = struct {
    cfg: config.Audio,
    s: mlx.mlx_stream,
    dtype: mlx.mlx_dtype,
    asr: Encoder,
    spk: Encoder,
    /// The Slaney filterbank transposed `[n_fft / 2 + 1, mels]` f32; the Hann window centered in `n_fft` (host).
    fb: Array,
    window: []f32,
    asr_norm_w: Array,
    asr_norm_b: Array,
    diar_norm_w: Array,
    diar_norm_b: Array,
    /// The sinusoidal speaker kernel `[speakers, d_model]` f32.
    kernel: Array,
    proj: Linear,
    spk_proj: Linear,
    up_w: Array,
    up_b: Array,
    dense: Linear,
    head: Linear,
    silence: Array,
    ctc: ?Ctc = null,

    pub fn load(ctx: *const abi.LoadCtx, cfg: config.Audio, block_json: []const u8, owned: *reader.Owned) !Tower {
        var r = try reader.Reader.open(ctx, block_json, "perception");
        defer r.close();
        var t: Tower = undefined;
        t.cfg = cfg;
        t.s = r.s;
        t.asr = try Encoder.load(&r, owned, "perception.encoder.asr_encoder", cfg.asr, cfg.mels * cfg.stack);
        errdefer t.asr.deinit();
        t.spk = try Encoder.load(&r, owned, "perception.encoder.diarization_model.encoder", cfg.spk, cfg.mels * cfg.stack);
        errdefer t.spk.deinit();
        const sm = "perception.encoder.diarization_model.sortformer_modules";
        var nb: [192]u8 = undefined;
        t.asr_norm_w = try arr(&r, owned, "perception.encoder.asr_norm.weight");
        t.asr_norm_b = try arr(&r, owned, "perception.encoder.asr_norm.bias");
        t.diar_norm_w = try arr(&r, owned, "perception.encoder.diar_norm.weight");
        t.diar_norm_b = try arr(&r, owned, "perception.encoder.diar_norm.bias");
        t.dtype = mlx.mlx_array_dtype(t.asr_norm_w);
        t.proj = try lin(&r, owned, "perception.proj", cfg.asr.d_model, true);
        t.spk_proj = try lin(&r, owned, try std.fmt.bufPrint(&nb, "{s}.encoder_proj", .{sm}), cfg.spk.d_model, true);
        t.up_w = try arr(&r, owned, try std.fmt.bufPrint(&nb, "{s}.subpixel_upsample.weight", .{sm}));
        t.up_b = try arr(&r, owned, try std.fmt.bufPrint(&nb, "{s}.subpixel_upsample.bias", .{sm}));
        const tf: u32 = @intCast(ops.shape(t.up_w)[2]);
        t.dense = try lin(&r, owned, try std.fmt.bufPrint(&nb, "{s}.first_hidden_to_hidden", .{sm}), tf, true);
        t.head = try lin(&r, owned, try std.fmt.bufPrint(&nb, "{s}.single_hidden_to_spks", .{sm}), tf, true);
        t.silence = try arr(&r, owned, try std.fmt.bufPrint(&nb, "{s}.learnable_sil_emb", .{sm}));

        var g = Graph.init(gpa, r.s);
        defer g.deinit();
        const bins: c_int = @intCast(cfg.n_fft / 2 + 1);
        const fb = try arr(&r, owned, "perception.preprocessor.fb");
        t.fb = g.escape(try g.transpose(try g.astype(try g.reshape(fb, &.{ @intCast(cfg.mels), bins }), .float32), &.{ 1, 0 }));
        try owned.own(t.fb);
        try ops.eval(t.fb);
        const win = try g.astype(try arr(&r, owned, "perception.preprocessor.window"), .float32);
        try ops.eval(win);
        if (mlx.mlx_array_size(win) != cfg.win) return error.WindowRead;
        const taps = (mlx.mlx_array_data_float32(win) orelse return error.WindowRead)[0..cfg.win];
        t.window = try gpa.alloc(f32, cfg.n_fft);
        errdefer gpa.free(t.window);
        @memset(t.window, 0);
        @memcpy(t.window[(cfg.n_fft - cfg.win) / 2 ..][0..cfg.win], taps);
        t.kernel = try speakerKernel(&g, cfg);
        try owned.own(t.kernel);
        t.ctc = null;
        if (cfg.ctc) |cc| if (r.has("ctc_timestamp.decoder.decoder_layers.0.weight")) {
            t.ctc = Ctc.load(ctx, &r, owned, cc) catch |e| blk: {
                log.warn("[nemotron_transcribe] no word timestamps: {s}\n", .{@errorName(e)});
                break :blk null;
            };
        };
        return t;
    }

    pub fn deinit(t: *Tower) void {
        if (t.ctc) |*c| c.deinit();
        t.asr.deinit();
        t.spk.deinit();
        gpa.free(t.window);
    }

    /// NeMo's `[mels, frames]` log-mel over `pcm`: windows centered on each hop over the pre-emphasized clip, zero past
    /// its ends, one frame per `hop` samples (the partial last hop is dropped). f32, owned by `g`.
    fn features(t: *const Tower, g: *Graph, pcm: []const f32) !Array {
        const c = t.cfg;
        const total = pcm.len / c.hop;
        var host = try gpa.alloc(f32, @min(total, MEL_BLOCK) * c.n_fft);
        defer gpa.free(host);
        var blocks: std.ArrayList(Array) = .empty;
        defer blocks.deinit(gpa);
        var done: usize = 0;
        while (done < total) {
            const n = @min(MEL_BLOCK, total - done);
            for (0..n) |f| {
                const base = @as(i64, @intCast((done + f) * c.hop)) - @as(i64, c.n_fft / 2);
                for (host[f * c.n_fft ..][0..c.n_fft], t.window, 0..) |*h, w, k| {
                    const i = base + @as(i64, @intCast(k));
                    const x: f32 = if (i < 0 or i >= pcm.len) 0 else pcm[@intCast(i)] - (if (i > 0) c.preemphasis * pcm[@intCast(i - 1)] else 0);
                    h.* = x * w;
                }
            }
            const framed = try g.fromSlice(f32, host[0 .. n * c.n_fft], &.{ @intCast(n), @intCast(c.n_fft) });
            const spec = try g.call(mlx.mlx_fft_rfft, .{ framed, @as(c_int, @intCast(c.n_fft)), @as(c_int, 1), mlx.MLX_FFT_NORM_BACKWARD });
            const mag = try g.abs(spec);
            const mel = try g.addScalar(try g.matmul(try g.square(mag), t.fb), LOG_GUARD);
            const block = try g.log(mel);
            try ops.eval(block);
            try blocks.append(gpa, block);
            done += n;
        }
        return g.transpose(try g.concat(blocks.items, 0), &.{ 1, 0 });
    }

    /// What one pass over a clip yields: the trunk's rows, the ASR encoder's states, the speakers' per-frame activity.
    const Want = struct { rows: bool = false, states: bool = false, speakers: bool = false };
    const Pass = struct {
        /// `[rows, out_dim]` after the fusion and `proj` (caller owns).
        rows: ?mlx.mlx_array = null,
        /// `[1, rows, d]` ASR encoder states (caller owns).
        states: ?mlx.mlx_array = null,
        /// `[frames][speakers]` f32, one row per 10 ms (gpa-owned).
        speakers: []f32 = &.{},
    };

    /// The clip's rows `[rows, out_dim]` (caller owns) from its 16 kHz mono samples.
    pub fn forward(t: *const Tower, pcm: []const f32) !mlx.mlx_array {
        return (try t.pass(pcm, .{ .rows = true })).rows.?;
    }

    /// The windows of `pcm` (`chunk` rows each with `left`/`right` rows of context), each through the experts `want` asks.
    fn pass(t: *const Tower, pcm: []const f32, want: Want) !Pass {
        const c = t.cfg;
        const sc = c.stream;
        const need_asr = want.rows or want.states;
        const need_spk = want.rows or want.speakers;
        var g = Graph.init(gpa, t.s);
        defer g.deinit();
        const feats = try t.features(&g, pcm);
        try ops.eval(feats);
        const total: usize = @intCast(ops.shape(feats)[1]);
        const normalized = if (need_asr) try g.astype(try normalize(&g, feats), t.dtype) else feats;
        try ops.eval(normalized);

        var st = Stream{ .t = t };
        defer st.deinit();
        var outs: std.ArrayList(Array) = .empty;
        var states: std.ArrayList(Array) = .empty;
        defer {
            for (outs.items) |o| _ = mlx.mlx_array_free(o);
            outs.deinit(gpa);
            for (states.items) |o| _ = mlx.mlx_array_free(o);
            states.deinit(gpa);
        }
        var hi: std.ArrayList(f32) = .empty;
        errdefer hi.deinit(gpa);
        const stack: usize = c.stack;
        const size = @as(usize, sc.chunk) * stack;
        const left = @as(usize, sc.left) * stack;
        const right = @as(usize, sc.right) * stack;
        var start: usize = 0;
        while (start < total) : (start += size) {
            var cg = Graph.init(gpa, t.s);
            defer cg.deinit();
            const end = @min(start + size, total);
            const lo = start -| left;
            const hi_end = @min(total, end + right);
            const first: c_int = @intCast((start - lo) / stack);
            const count = (end + stack - 1) / stack - (start + stack - 1) / stack;

            var window_states: ?Array = null;
            if (need_asr) {
                const stacked = try t.asr.stackRows(&cg, try cg.sliceAxis(normalized, 1, @intCast(lo), @intCast(hi_end)), c.stack, t.dtype);
                const encoded_all = try t.asr.forward(&cg, stacked);
                window_states = try cg.sliceAxis(encoded_all, 1, first, first + @as(c_int, @intCast(count)));
            }
            if (want.states) {
                try ops.eval(window_states.?);
                try states.append(gpa, cg.escape(window_states.?));
            }
            if (need_spk) {
                const window = try t.spk.stackRows(&cg, try cg.sliceAxis(feats, 1, @intCast(lo), @intCast(hi_end)), c.stack, t.dtype);
                const probs = try gpa.alloc(f32, count * c.speakers);
                defer gpa.free(probs);
                const frames = if (want.speakers) try hi.addManyAsSlice(gpa, count * stack * c.speakers) else null;
                try st.step(&cg, window, (start - lo) / stack, count, probs, frames);
                if (want.rows) {
                    const fused = try t.fuse(&cg, window_states.?, try cg.fromSlice(f32, probs, &.{ 1, @intCast(count), @intCast(c.speakers) }));
                    try ops.eval(fused);
                    try outs.append(gpa, cg.escape(fused));
                }
            }
        }
        var out: Pass = .{};
        if (want.rows) {
            const joined = try g.concat(outs.items, 1);
            const rows_out = try t.proj.forward(&g, joined);
            const sh = ops.shape(rows_out);
            out.rows = g.escape(try g.reshape(rows_out, &.{ sh[1], sh[2] }));
        }
        errdefer if (out.rows) |r| {
            _ = mlx.mlx_array_free(r);
        };
        if (want.states) out.states = g.escape(try g.concat(states.items, 1));
        if (want.speakers) {
            const kept = @min(total * c.speakers, hi.items.len);
            hi.shrinkRetainingCapacity(kept);
            out.speakers = try hi.toOwnedSlice(gpa);
        }
        return out;
    }

    /// Speaker activity `[frames][speakers]` (10 ms rows, gpa-owned) over the clip, the speaker expert alone.
    pub fn speakerFrames(t: *const Tower, pcm: []const f32) ![]f32 {
        return (try t.pass(pcm, .{ .speakers = true })).speakers;
    }

    /// The transcript's `units` (its words in spoken order, `<spk:N>` turn tags among them) timed over `pcm` by the CTC
    /// head: per unit a start and an end in ms and the Sortformer column of its speaker (`NO_SPEAKER` for none), 3 u32
    /// each; a tag's entry is zeros. `activity` is the clip's `[frames][speakers]` 10 ms speaker probabilities (empty:
    /// no speaker information). gpa-owned.
    pub fn wordTimes(t: *const Tower, pcm: []const f32, units: []const []const u8, activity: []const f32, speakers: u32) ![]u32 {
        const ctc = &(t.ctc orelse return error.NoCtcHead);
        const tok = &(ctc.tok orelse return error.NoAlignmentTokenizer);
        const words = try gpa.alloc(TokenWord, units.len);
        defer gpa.free(words);
        var n: usize = 0;
        var tag: i32 = -1;
        const of_unit = try gpa.alloc(i64, units.len);
        defer gpa.free(of_unit);
        defer for (words[0..n]) |w| gpa.free(w.ids);
        for (units, of_unit) |u, *slot| {
            if (speakerTag(u)) |x| {
                tag = x;
                slot.* = -1;
                continue;
            }
            // Each word alone ("▁word"), as NeMo tokenizes a stream's words.
            words[n] = .{ .tag = tag, .ids = try tok.encode(gpa, u) };
            slot.* = @intCast(n);
            n += 1;
        }
        const timed = try t.alignTokens(pcm, words[0..n], activity, speakers);
        defer gpa.free(timed);
        const out = try gpa.alloc(u32, 3 * units.len);
        errdefer gpa.free(out);
        for (of_unit, 0..) |wi, i| {
            if (wi < 0) {
                @memset(out[3 * i ..][0..3], 0);
                continue;
            }
            const x = timed[@intCast(wi)];
            out[3 * i] = @intFromFloat(@round(x.start * 1000));
            out[3 * i + 1] = @intFromFloat(@round(x.end * 1000));
            out[3 * i + 2] = if (x.speaker < 0) NO_SPEAKER else @intCast(x.speaker);
        }
        return out;
    }

    pub const TokenWord = struct { tag: i32, ids: []u32 };
    pub const NO_SPEAKER: u32 = std.math.maxInt(u32);

    /// `wordTimes` over token ids (vocabulary ids below the head's classes).
    pub fn alignTokens(t: *const Tower, pcm: []const f32, words: []const TokenWord, activity: []const f32, speakers: u32) ![]ctc_align.Timed {
        const ctc = &(t.ctc orelse return error.NoCtcHead);
        for (words) |w| for (w.ids) |id| if (id >= ctc.cfg.classes) return error.TokenOutOfRange;
        // Compact columns: the blank first, then the distinct tokens in order of id.
        var columns: std.ArrayList(u32) = .empty;
        defer columns.deinit(gpa);
        try columns.append(gpa, ctc.cfg.classes);
        for (words) |w| for (w.ids) |id| try columns.append(gpa, id);
        std.mem.sort(u32, columns.items[1..], {}, std.sort.asc(u32));
        var uniq: usize = 1;
        for (columns.items[1..]) |id| {
            if (id != columns.items[uniq - 1] or uniq == 1) {
                columns.items[uniq] = id;
                uniq += 1;
            }
        }
        columns.shrinkRetainingCapacity(uniq);
        const target = try gpa.alloc(ctc_align.Word, words.len);
        defer gpa.free(target);
        var built: usize = 0;
        defer for (target[0..built]) |w| gpa.free(w.tokens);
        for (words, target) |w, *o| {
            const cols = try gpa.alloc(u32, w.ids.len);
            for (w.ids, cols) |id, *c| c.* = @intCast(std.mem.indexOfScalar(u32, columns.items, id).?);
            o.* = .{ .tag = w.tag, .tokens = cols };
            built += 1;
        }

        var g = Graph.init(gpa, t.s);
        defer g.deinit();
        const pass_out = try t.pass(pcm, .{ .states = true });
        const all_states = pass_out.states.?;
        defer _ = mlx.mlx_array_free(all_states);
        const n_rows: usize = @intCast(ops.shape(all_states)[1]);
        const keep = try g.fromSlice(u32, columns.items, &.{@intCast(columns.items.len)});
        const up: usize = ctc.cfg.upsample;
        const ncol = columns.items.len;
        const frames = n_rows * up;
        const lp = try gpa.alloc(f32, frames * ncol);
        defer gpa.free(lp);
        const sc = t.cfg.stream;
        var core: usize = 0;
        while (core < n_rows) : (core += sc.chunk) {
            var cg = Graph.init(gpa, t.s);
            defer cg.deinit();
            const end = @min(core + sc.chunk, n_rows);
            const lo = core -| sc.left;
            const hi = @min(n_rows, end + sc.right);
            const window = try cg.sliceAxis(all_states, 1, @intCast(lo), @intCast(hi));
            const logp = try ctc.columns(&cg, window, keep);
            const own = try cg.sliceAxis(logp, 1, @intCast((core - lo) * up), @intCast((end - lo) * up));
            try ops.eval(own);
            const host = (mlx.mlx_array_data_float32(own) orelse return error.CtcOutput)[0 .. (end - core) * up * ncol];
            @memcpy(lp[core * up * ncol ..][0..host.len], host);
        }

        // The speakers: the 10 ms activity pooled onto the encoder's rows, its > 0.5 labels as they are.
        var spk: ?ctc_align.Speakers = null;
        var spk_rows: []f32 = &.{};
        defer gpa.free(spk_rows);
        var labels: []u8 = &.{};
        defer gpa.free(labels);
        if (speakers > 0 and activity.len >= speakers) {
            const count: usize = speakers;
            const diar_frames = activity.len / count;
            const stack: usize = t.cfg.stack;
            const n_spk_rows = (diar_frames + stack - 1) / stack;
            spk_rows = try gpa.alloc(f32, n_spk_rows * count);
            for (0..n_spk_rows) |r| for (0..count) |s| {
                var sum: f32 = 0;
                const hi_f = @min(diar_frames, (r + 1) * stack);
                for (r * stack..hi_f) |f| sum += activity[f * count + s];
                spk_rows[r * count + s] = sum / @as(f32, @floatFromInt(hi_f - r * stack));
            };
            labels = try gpa.alloc(u8, diar_frames * count);
            for (activity[0 .. diar_frames * count], labels) |x, *l| l.* = @intFromBool(x > 0.5);
            spk = .{ .rows = spk_rows, .n_rows = n_spk_rows, .count = count, .diar = labels, .diar_frames = diar_frames, .duration = @as(f64, @floatFromInt(pcm.len)) / @as(f64, @floatFromInt(config.SAMPLE_RATE)) };
        }
        const step = @as(f64, @floatFromInt(config.HOP * t.cfg.stack / up)) / @as(f64, @floatFromInt(config.SAMPLE_RATE));
        return ctc_align.alignWords(gpa, lp, frames, ncol, target, spk, step, ctc.cfg.timing);
    }

    /// ASR rows plus the speaker kernel weighted by each frame's normalized speaker activity.
    fn fuse(t: *const Tower, g: *Graph, states: Array, probs_f32: Array) !Array {
        var p = try g.astype(probs_f32, t.dtype);
        if (t.cfg.thresholded) p = try g.astype(try g.greater(p, try g.scalarLike(p, t.cfg.threshold)), t.dtype);
        const targets = try g.layerNorm(p, t.diar_norm_w, t.diar_norm_b, LN_EPS);
        const infusion = try g.matmul(targets, try g.astype(t.kernel, t.dtype));
        const scaled = try g.mulScalar(infusion, t.cfg.kernel_scale);
        return g.add(try g.layerNorm(states, t.asr_norm_w, t.asr_norm_b, LN_EPS), scaled);
    }

    /// Speaker probabilities `[rows * stack, speakers]` (f32) for the speaker encoder's rows `x`.
    fn speakerProbs(t: *const Tower, g: *Graph, x: Array) !Array {
        const h = try t.spk.forward(g, x);
        const p = try t.spk_proj.forward(g, h);
        const n = ops.shape(p)[1];
        const tf = ops.shape(p)[2];
        const up = try g.add(try g.conv1d(p, t.up_w, 1, 1, 1, 1), t.up_b);
        const flat = try g.reshape(up, &.{ n * @as(c_int, @intCast(t.cfg.stack)), tf });
        const hidden = try t.dense.forward(g, try relu(g, flat));
        return g.sigmoid(try g.astype(try t.head.forward(g, try relu(g, hidden)), .float32));
    }
};

/// NeMo's per-feature normalization over time: mean and the unbiased std (+ 1e-5), the first frame as the mean's
/// reference for the f32 sum.
fn normalize(g: *Graph, feats: Array) !Array {
    const frames = ops.shape(feats)[1];
    const reference = try g.sliceAxis(feats, 1, 0, 1);
    const mean = try g.add(reference, try g.mean(try g.sub(feats, reference), 1, true));
    const centered = try g.sub(feats, mean);
    const variance = try g.mulScalar(try g.sum(try g.square(centered), 1, true), 1.0 / @as(f32, @floatFromInt(@max(1, frames - 1))));
    return g.div(centered, try g.addScalar(try g.sqrt(variance), 1e-5));
}

/// `[speakers, d_model]`: sin and cos of `speaker * 10000^(-2i / d_model)`, interleaved.
fn speakerKernel(g: *Graph, cfg: config.Audio) !Array {
    const d = cfg.asr.d_model;
    const host = try gpa.alloc(f32, cfg.speakers * d);
    defer gpa.free(host);
    for (0..cfg.speakers) |s| for (0..d / 2) |i| {
        const freq = @exp(@as(f32, @floatFromInt(2 * i)) * -(@log(10000.0) / @as(f32, @floatFromInt(d))));
        const angle = @as(f32, @floatFromInt(s)) * freq;
        host[s * d + 2 * i] = @sin(angle);
        host[s * d + 2 * i + 1] = @cos(angle);
    };
    return g.escape(try g.fromSlice(f32, host, &.{ @intCast(cfg.speakers), @intCast(d) }));
}

/// The speaker encoder's state between windows: the cache and the FIFO (encoder rows, owned), and the cache's own
/// probabilities once it is compressed (a compressed cache is out of order, so only they describe it).
const Stream = struct {
    t: *const Tower,
    cache: ?Array = null,
    fifo: ?Array = null,
    cache_probs: []f32 = &.{},
    compressed: bool = false,

    fn deinit(st: *Stream) void {
        if (st.cache) |x| _ = mlx.mlx_array_free(x);
        if (st.fifo) |x| _ = mlx.mlx_array_free(x);
        gpa.free(st.cache_probs);
    }

    fn rowsOf(x: ?Array) usize {
        return if (x) |v| @intCast(ops.shape(v)[1]) else 0;
    }

    /// One window `chunk` (`left` rows of context, then `n` rows of its own, then the look-ahead): the probabilities of
    /// its own `n` rows (the mean of each row's `stack` frames) into `dst`; its own rows then join the FIFO and the
    /// FIFO's oldest the cache.
    fn step(st: *Stream, g: *Graph, chunk: Array, left: usize, n: usize, dst: []f32, frames: ?[]f32) !void {
        const t = st.t;
        const c = t.cfg;
        const sc = c.stream;
        const spk: usize = c.speakers;
        const f: usize = c.stack;
        const nc = rowsOf(st.cache);
        const nf = rowsOf(st.fifo);
        var parts: [3]Array = undefined;
        var np: usize = 0;
        for ([_]?Array{ st.cache, st.fifo }) |x| if (x) |v| {
            parts[np] = v;
            np += 1;
        };
        parts[np] = chunk;
        const probs = try t.speakerProbs(g, if (np == 0) chunk else try g.concat(parts[0 .. np + 1], 1));
        try ops.eval(probs);
        const high = (mlx.mlx_array_data_float32(probs) orelse return error.SpeakerOutput)[0..mlx.mlx_array_size(probs)];
        const first = nc + nf + left;
        for (0..n) |r| {
            for (0..spk) |s| dst[r * spk + s] = rowMean(high, first + r, s, f, spk);
        }
        if (frames) |fr| for (0..n * f) |k| @memcpy(fr[k * spk ..][0..spk], high[(first * f + k) * spk ..][0..spk]);

        const own = try g.sliceAxis(chunk, 1, @intCast(left), @intCast(left + n));
        const fifo = if (st.fifo) |v| try g.concat(&.{ v, own }, 1) else own;
        const queued = nf + n;
        if (queued <= sc.fifo) return st.replace(g, st.cache, fifo);
        // mlx-audio's `streaming_step`: the update period (300) outsizes the FIFO (40), so its first overflow pops all of it.
        const pop = @min(queued, @max(sc.period, queued - sc.fifo));
        const total = nc + pop;
        const low = try gpa.alloc(f32, total * spk);
        defer gpa.free(low);
        for (0..total) |r| {
            const row = if (r < nc) r else if (r - nc < nf) r else first + (r - nc - nf);
            for (0..spk) |s| low[r * spk + s] = rowMean(high, row, s, f, spk);
        }
        if (st.compressed) {
            if (st.cache_probs.len != nc * spk) return error.SpeakerCache;
            @memcpy(low[0 .. nc * spk], st.cache_probs);
        }
        const popped = try g.sliceAxis(fifo, 1, 0, @intCast(pop));
        const rest: ?Array = if (pop < queued) try g.sliceAxis(fifo, 1, @intCast(pop), @intCast(queued)) else null;
        const grown = if (st.cache) |v| try g.concat(&.{ v, popped }, 1) else popped;
        if (total <= sc.cache) return st.replace(g, grown, rest);

        const policy: diarization.CachePolicy = .{
            .len = sc.cache,
            .silence_per_speaker = sc.silence_per_speaker,
            .score_threshold = sc.score_threshold,
            .latest_boost = sc.latest_boost,
            .strong_boost_rate = sc.strong_boost_rate,
            .weak_boost_rate = sc.weak_boost_rate,
            .min_positive_rate = sc.min_positive_rate,
        };
        const keep = try diarization.aoscKeep(gpa, low, spk, policy);
        defer gpa.free(keep);
        const kept = try gpa.alloc(f32, keep.len * spk);
        for (keep, 0..) |k, i| {
            const row = kept[i * spk ..][0..spk];
            if (k == total) @memset(row, 0) else @memcpy(row, low[k * spk ..][0..spk]);
        }
        gpa.free(st.cache_probs);
        st.cache_probs = kept;
        st.compressed = true;
        const d: c_int = @intCast(c.spk.d_model);
        const table = try g.concat(&.{ try g.reshape(grown, &.{ @intCast(total), d }), try g.reshape(try g.astype(t.silence, t.dtype), &.{ 1, d }) }, 0);
        const ids = try g.fromSlice(u32, keep, &.{@intCast(keep.len)});
        const cache = try g.reshape(try g.take(table, ids, 0), &.{ 1, @intCast(keep.len), d });
        return st.replace(g, cache, rest);
    }

    /// Hold `cache` and `fifo` past the window's graph, evaluated.
    fn replace(st: *Stream, g: *Graph, cache: ?Array, fifo: ?Array) !void {
        const next = [2]?Array{ if (cache) |v| g.escape(v) else null, if (fifo) |v| g.escape(v) else null };
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (next) |x| if (x) |v| {
            _ = mlx.mlx_vector_array_append_value(vec, v);
        };
        mlx.check(mlx.mlx_eval(vec)) catch |err| {
            for (next) |x| if (x) |v| {
                _ = mlx.mlx_array_free(v);
            };
            return err;
        };
        if (st.cache) |v| _ = mlx.mlx_array_free(v);
        if (st.fifo) |v| _ = mlx.mlx_array_free(v);
        st.cache = next[0];
        st.fifo = next[1];
    }
};

/// Speaker `s`'s mean probability over the `f` frames of encoder row `row`.
fn rowMean(high: []const f32, row: usize, s: usize, f: usize, spk: usize) f32 {
    var sum: f32 = 0;
    for (0..f) |k| sum += high[(row * f + k) * spk + s];
    return sum / @as(f32, @floatFromInt(f));
}

fn lin(r: *reader.Reader, owned: *reader.Owned, base: []const u8, in_dim: u32, bias: bool) !Linear {
    var l = try r.proj(base, in_dim);
    try owned.linear(l);
    if (bias) {
        var nb: [224]u8 = undefined;
        l.bias = try arr(r, owned, try std.fmt.bufPrint(&nb, "{s}.bias", .{base}));
    }
    return l;
}

fn arr(r: *reader.Reader, owned: *reader.Owned, full: []const u8) !Array {
    const a = try r.need(full);
    try owned.own(a);
    return a;
}

// ── tests ──

test "nemotron_transcribe perception: a clip's rows are its hops grouped by the stacking factor" {
    try std.testing.expectEqual(@as(usize, 0), rows(159));
    try std.testing.expectEqual(@as(usize, 1), rows(160));
    try std.testing.expectEqual(@as(usize, 1), rows(1280));
    try std.testing.expectEqual(@as(usize, 2), rows(1440));
    try std.testing.expectEqual(@as(usize, 300), rows(16000 * 24));
}
