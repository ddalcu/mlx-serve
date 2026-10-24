//! DeepSeek-V4.1 (deepseek_v41), served from MLX ops. Its own module beside
//! deepseek_v4: V4.1 shares compressed KV and index keys across layers,
//! prefilters candidate blocks, staggers the hyper-connections (each sublayer
//! collapses with the mix the previous one computed), adds Engram, and has no
//! hash-routed layers. References: DeepSeek's inference/model.py (rev
//! dba1be0a) and mlx-lm's deepseek_v41.py; fixtures from
//! tests/dump_dsv41_fixtures.py.
//!
//! The request's state lives on the model, so one request runs at a time.
//! A chunk runs one layer at a time over all of its spans: each layer's
//! experts are read once per chunk.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const transformer = @import("transformer.zig");
const gather_nax = @import("gather_qmm_nax.zig");
const log = @import("log.zig");
const engram = @import("dsv41_engram.zig");
const glm5 = @import("glm5_next.zig");
const mxfp8_qmv = @import("mxfp8_qmv.zig");
const dsv4 = @import("deepseek_v4.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const ModelConfig = model.ModelConfig;
const QuantParams = transformer.QuantParams;
const none: A = .{ .ctx = null };

/// Rows one attention span handles: the compressed rows each query gathers
/// are its largest transient ([span, index_topk, head_dim] f32).
pub const SPAN: usize = 256;

/// Threads in a hyper-connection threadgroup before `probeHcThreads` finds what the GPU grants.
const HC_TG: usize = 1024;

/// Rows one MoE call handles: a layer's expert banks stream once per call.
const MOE_ROWS: usize = 2048;
/// Window rows kept past the window, so a verify chunk can write its rows
/// and still roll back to any prefix.
const RING_SLACK: usize = 16;
/// A prompt chunk replays through the decoder (`cedTail`) only when at least this many of its rows skip it.
const CED_MIN_SKIP: usize = 256;

// ── op scope: every intermediate a forward makes, freed together ──────────

const Scope = struct {
    a: std.mem.Allocator,
    s: S,
    items: std.ArrayList(A) = .empty,

    fn init(a: std.mem.Allocator, s: S) Scope {
        return .{ .a = a, .s = s };
    }
    fn deinit(self: *Scope) void {
        for (self.items.items) |x| _ = mlx.mlx_array_free(x);
        self.items.deinit(self.a);
    }
    fn keep(self: *Scope, x: A) !A {
        self.items.append(self.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }
    fn res(self: *Scope, rc: c_int, o: *const A) !A {
        mlx.check(rc) catch |e| {
            _ = mlx.mlx_array_free(o.*);
            return e;
        };
        return self.keep(o.*);
    }
    /// A handle on `x` that outlives the scope (caller frees).
    fn out(_: *Scope, x: A) A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }
    fn op1(sc: *Scope, comptime func: anytype, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(func(&o, x, sc.s), &o);
    }
    fn op2(sc: *Scope, comptime func: anytype, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(func(&o, x, y, sc.s), &o);
    }
    fn add(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_add, x, y);
    }
    fn sub(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_subtract, x, y);
    }
    fn mul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_multiply, x, y);
    }
    fn div(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_divide, x, y);
    }
    fn matmul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_matmul, x, y);
    }
    fn f(sc: *Scope, v: f32) !A {
        return sc.keep(mlx.mlx_array_new_float(v));
    }
    fn u(sc: *Scope, v: u32) !A {
        return sc.keep(mlx.mlx_array_new_data(&v, &[_]c_int{}, 0, .uint32));
    }
    fn i(sc: *Scope, v: i32) !A {
        return sc.keep(mlx.mlx_array_new_int(v));
    }
    fn ints(sc: *Scope, data: []const i32, shape: []const c_int) !A {
        return sc.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .int32));
    }
    fn astype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        if (mlx.mlx_array_dtype(x) == dt) return x;
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_astype(&o, x, dt, sc.s), &o);
    }
    fn view(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_view(&o, x, dt, sc.s), &o);
    }
    fn reshape(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    fn transpose(sc: *Scope, x: A, axes: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, sc.s), &o);
    }
    fn t2(sc: *Scope, x: A) !A {
        return sc.transpose(x, &.{ 1, 0 });
    }
    fn broadcast(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_broadcast_to(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    fn expand(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_expand_dims(&o, x, axis, sc.s), &o);
    }
    /// x[..., lo:hi, ...] on one axis (negative axis counts from the end).
    fn slice(sc: *Scope, x: A, axis_in: isize, lo: usize, hi: usize) !A {
        const sh = mlx.getShape(x);
        const axis: usize = if (axis_in < 0) @intCast(@as(isize, @intCast(sh.len)) + axis_in) else @intCast(axis_in);
        var start: [8]c_int = @splat(0);
        var stop: [8]c_int = undefined;
        const st: [8]c_int = @splat(1);
        @memcpy(stop[0..sh.len], sh);
        start[axis] = @intCast(lo);
        stop[axis] = @intCast(hi);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice(&o, x, &start, sh.len, &stop, sh.len, &st, sh.len, sc.s), &o);
    }
    fn concat(sc: *Scope, xs: []const A, axis: c_int) !A {
        if (xs.len == 1) return xs[0];
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_concatenate_axis(&o, vec, axis, sc.s), &o);
    }
    fn stack(sc: *Scope, xs: []const A, axis: c_int) !A {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_stack_axis(&o, vec, axis, sc.s), &o);
    }
    fn take(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_axis(&o, x, idx, axis, sc.s), &o);
    }
    fn takeAlong(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_along_axis(&o, x, idx, axis, sc.s), &o);
    }
    fn reduce(sc: *Scope, comptime f_: anytype, x: A, axis: c_int, keep_: bool) !A {
        var o = mlx.mlx_array_new();
        return sc.res(f_(&o, x, axis, keep_, sc.s), &o);
    }
    fn sum(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_sum_axis, x, axis, keep_);
    }
    fn mean(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_mean_axis, x, axis, keep_);
    }
    fn max(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_max_axis, x, axis, keep_);
    }
    fn argmax(sc: *Scope, x: A, axis: c_int) !A {
        return sc.reduce(mlx.mlx_argmax_axis, x, axis, false);
    }
    fn softmax(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_softmax_axis(&o, x, axis, true, sc.s), &o);
    }
    /// The first `k` positions of `x` along the last axis, largest first (unordered).
    fn topk(sc: *Scope, x: A, k: usize) !A {
        const neg = try sc.op1(mlx.mlx_negative, x);
        var o = mlx.mlx_array_new();
        const part = try sc.res(mlx.mlx_argpartition_axis(&o, neg, @intCast(k - 1), -1, sc.s), &o);
        return sc.slice(part, -1, 0, k);
    }
    fn where(sc: *Scope, c: A, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_where(&o, c, x, y, sc.s), &o);
    }
    fn rms(sc: *Scope, x: A, w: A, eps: f32) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rms_norm(&o, x, w, eps, sc.s), &o);
    }
    fn arange(sc: *Scope, lo: usize, hi: usize) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_arange(&o, @floatFromInt(lo), @floatFromInt(hi), 1, .int32, sc.s), &o);
    }
    fn zeros(sc: *Scope, shape: []const c_int, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, sc.s), &o);
    }
    fn clip(sc: *Scope, x: A, lo: f32, hi: f32) !A {
        return sc.op2(mlx.mlx_minimum, try sc.op2(mlx.mlx_maximum, x, try sc.f(lo)), try sc.f(hi));
    }
    fn sigmoid(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_sigmoid, x);
    }
};

fn dim(x: A, axis: isize) usize {
    const sh = mlx.getShape(x);
    const ax: usize = if (axis < 0) @intCast(@as(isize, @intCast(sh.len)) + axis) else @intCast(axis);
    return @intCast(sh[ax]);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

// ── weights ──────────────────────────────────────────────────────────────

/// One linear: a plain weight `[out, in]`, or a quantized one whose mode,
/// bits and group size are read off its own geometry.
pub const Lin = struct {
    w: A,
    s: A = none,
    b: A = none,
    qp: QuantParams = .{ .bits = 0, .group_size = 0 },

    fn apply(self: *const Lin, sc: *Scope, x: A) !A {
        if (self.s.ctx == null) return sc.matmul(x, try sc.t2(self.w));
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_quantized_matmul(&o, x, self.w, self.s, self.b, true, mlx.mlx_optional_int.some(@intCast(self.qp.group_size)), mlx.mlx_optional_int.some(@intCast(self.qp.bits)), self.qp.mode.cstr(), sc.s), &o);
    }

    /// Rows `ids` of the table, dequantized to `dt`.
    fn rows(self: *const Lin, sc: *Scope, ids: A, dt: mlx.mlx_dtype) !A {
        if (self.s.ctx == null) return sc.astype(try sc.take(self.w, ids, 0), dt);
        return sc.keep(try transformer.gatherQuantizedRows(sc.s, self.w, self.s, self.b, ids, self.qp.group_size, self.qp.bits, self.qp.mode, none, .{ .value = dt, .has_value = true }));
    }
};

const Hc = struct {
    fn_w: A, // f32 [mix, hc * dim]
    scale: A, // f32 [mix]: scale[0] for pre, [1] post, [2] comb
    base: A, // f32 [mix]
};

/// `wkv` above ratio 1: the kv rows then the gate rows as one linear whose
/// f32 input meets f32 math (scales widened to f32), since pooling runs in f32.
const Comp = struct { wkv: Lin, norm: A };
const Idx = struct { wq_b: Lin, weights_proj: Lin, wk: ?Lin, k_norm: A };

/// Stacked routed experts `[E, out, in]`; `gather_qmm` reads them by index.
const Bank = struct {
    w: A,
    s: A = none,
    b: A = none,
    qp: QuantParams = .{ .bits = 0, .group_size = 0 },
};

/// `gu`: gate and up concatenated along their output rows at load (same
/// format), so one gather reads both; `gate` and `up` are then unset.
/// `views`: one-expert views of `gu` and `down` that the decode kernels bind
/// in their place (they address experts from the buffer pointer): MLX bills
/// a command buffer for each bound array's size, and a whole bank forced a
/// commit at every kernel.
const Experts = struct { gate: Bank, up: Bank, down: Bank, gu: ?Bank = null, views: ?[2]Bank = null };

const Layer = struct {
    attn_norm: A,
    ffn_norm: A,
    q_norm: A,
    kv_norm: A,
    hc_attn: Hc,
    hc_ffn: Hc,
    wq_a: Lin,
    wq_b: Lin,
    wkv: Lin,
    /// Grouped output LoRA as `[o_groups, o_lora_rank, in]` views.
    wo_a: Lin,
    wo_b: Lin,
    sink: A, // f32 [1, heads, 1]
    ratio: u8,
    comp: ?Comp,
    idx: ?Idx,
    gate_w: A, // [E, dim], read in f32 by `GATE_SOURCE`
    gate_bias: A, // f32 [1, E]
    gate_bias_vl: ?A = null, // f32 [1, E]: the bias an image span's rows pick experts by (`noaux_tc_for_vl`)
    top_k: usize,
    shared: [3]Lin, // w1 (gate), w3 (up), w2 (down)
    shared_gu: ?Lin = null, // w1 and w3 as one linear; then shared[0..2] are unset
    q_kv: ?Lin = null, // a trunk layer's wq_a rows then wkv rows; then wq_a and wkv are unset
    experts: Experts,
};

const EngramLayer = struct {
    layer: usize,
    wkv: Lin,
    qk: A, // f32 [hc, dim]: q_weight * k_weight
    table: engram.Table,
};

const DsparkW = struct {
    main_proj: Lin,
    main_norm: A,
    norm: A, // the last stage's
    markov_embed: A, // [vocab, rank]
    markov_head: A, // f32 [vocab, rank]
    conf: A, // f32 [1, dim + rank]
};

/// Rope as MLX `fast_rope` periods (`1 / inv_freq`), and their negation
/// for the inverse rotation.
const Rope = struct { fwd: A, inv: A };

pub const Dsv41Model = struct {
    gpa: std.mem.Allocator,
    s: S,
    cfg: ModelConfig,
    act: mlx.mlx_dtype,
    embed: Lin,
    head: Lin,
    norm: A,
    /// The trunk, then the DSpark stages.
    layers: []Layer,
    n_layers: usize,
    vocab: usize,
    n_mtp: usize = 0,
    ds_block: usize = 0,
    /// Rows per attention span (tests narrow it to drive several spans).
    span: usize = SPAN,
    /// Rows per MoE call: spans whose rows fit are routed and run together.
    moe_rows: usize = MOE_ROWS,
    /// CED: the decoder's first layer (`cedSplit`), the fewest rows a replaying chunk skips, and how much has
    /// replayed.
    ced_split: ?usize = null,
    ced_min_skip: usize = CED_MIN_SKIP,
    ced_stats: struct { chunks: u64 = 0, skipped: u64 = 0 } = .{},
    /// Threads per threadgroup of the hc kernels, all four alike so the wide and decode-width stages sum in one order.
    hc_tg: usize = HC_TG,
    /// Bytes per indexer score block (tests narrow it to drive several).
    /// Test-only: named intermediates of a one-span forward.
    trace: ?*Trace = null,
    /// Test-only: skip the QAT round-trips (the reference's plain-math mode).
    qat_off: bool = false,
    dspark: ?DsparkW = null,
    hash: ?engram.Hash = null,
    token_map: []u32 = &.{},
    engrams: []EngramLayer = &.{},
    /// The request's prompt, whose later chunks' Engram rows are read while an earlier one runs.
    prompt: []u32 = &.{},
    ahead: ?*engram.Prefetch = null,
    rope_plain: Rope,
    rope_yarn: Rope,
    attn_dec_kernel: mlx.mlx_fast_metal_kernel,
    attn_split_kernel: mlx.mlx_fast_metal_kernel,
    attn_merge_kernel: mlx.mlx_fast_metal_kernel,
    attn_softmax_kernel: mlx.mlx_fast_metal_kernel,
    /// Wide spans score and weight rows in f16 (`attnWide`); `MLX_SERVE_DSV41_ATTN_F16=0` keeps the f32 chain.
    attn_f16: bool = true,
    idx_score_kernel: mlx.mlx_fast_metal_kernel,
    idx_tile_kernel: mlx.mlx_fast_metal_kernel,
    gate_kernel: mlx.mlx_fast_metal_kernel,
    topk_kernel: mlx.mlx_fast_metal_kernel,
    dense_f32x_kernel: mlx.mlx_fast_metal_kernel,
    route_kernel: mlx.mlx_fast_metal_kernel,
    hc_mix_kernel: mlx.mlx_fast_metal_kernel,
    hc_norm_kernel: mlx.mlx_fast_metal_kernel,
    hc_stage_kernel: mlx.mlx_fast_metal_kernel,
    hc_stage_x_kernel: mlx.mlx_fast_metal_kernel,
    hc_split_kernel: mlx.mlx_fast_metal_kernel,
    rope_kernel: mlx.mlx_fast_metal_kernel,
    kv_row_kernel: mlx.mlx_fast_metal_kernel,
    engram_kernel: mlx.mlx_fast_metal_kernel,
    moe_gu_kernel: mlx.mlx_fast_metal_kernel,
    moe_dn_kernel: mlx.mlx_fast_metal_kernel,
    moe_act_kernel: mlx.mlx_fast_metal_kernel,
    moe_combine_kernel: mlx.mlx_fast_metal_kernel,
    /// One-token decode's routed-gather input indices (what `gather_qmm` builds
    /// from an arange per call otherwise): gate/up `[1, k]` zeros, down `[1, k]` 0..k-1.
    lhs_gu: A = none,
    lhs_dn: A = none,
    /// `qat` per kind, compiled at init (null: the op chain).
    /// Per trunk layer: its decode-width stretches compiled (`Region`).
    regions: []LayerRegions = &.{},
    /// Arrays made at load (casts, reshapes, stacks); freed in `deinit`.
    owned: std.ArrayList(A) = .empty,
    dec_state: ?State = null,
    pair: ?Pair = null,
    /// Bumped whenever `dec_state` is replaced: a snapshot names the state it was taken of.
    state_gen: u64 = 0,
    /// The last prompt's state one token short of its end (`resumePrompt`).
    snap: ?Snap = null,
    /// Where this prompt's snapshot is taken (`armSnapshot`), and its tokens.
    snap_at: ?usize = null,
    snap_ids: []u32 = &.{},
    /// The image rows inside the snapshot's ids (`visionKey`): a prefix of the same ids with other pixels never resumes,
    /// and an image added AFTER the prefix does not stop it.
    snap_vision_key: u64 = 0,
    ds_conf_thr: f32 = -std.math.inf(f32),
    /// The adaptive verify policy reads each draft's confidence; otherwise only a finite `ds_conf_thr` does.
    ds_want_conf: bool = false,
    ds_prof: ?dsv4.DsparkProfile = null,
    ds_prof_head_ns: u64 = 0,
    ds_prof_comp_sync_ns: u64 = 0,

    fn dropAhead(self: *Dsv41Model) void {
        if (self.ahead) |a| a.finish();
        self.ahead = null;
    }

    pub fn deinit(self: *Dsv41Model) void {
        if (self.dec_state) |*st| st.deinit();
        if (self.snap) |*sn| sn.deinit(self.gpa);
        self.gpa.free(self.snap_ids);
        self.dropAhead();
        self.gpa.free(self.prompt);
        for (self.engrams) |*e| e.table.close();
        self.gpa.free(self.engrams);
        self.gpa.free(self.token_map);
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(self.gpa);
        self.gpa.free(self.layers);
        _ = mlx.mlx_fast_metal_kernel_free(self.attn_dec_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.attn_split_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.attn_merge_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.attn_softmax_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.idx_score_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.idx_tile_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.gate_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.route_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.hc_mix_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.topk_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.dense_f32x_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.hc_norm_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.hc_stage_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.hc_stage_x_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.hc_split_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.moe_gu_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.moe_dn_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.moe_act_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.moe_combine_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.rope_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.kv_row_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(self.engram_kernel);
        freeRegions(self);
        self.gpa.destroy(self);
    }

    fn own(self: *Dsv41Model, x: A) !A {
        self.owned.append(self.gpa, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    /// How many of `ids` are image span positions.
    fn imageRows(self: *const Dsv41Model, ids: []const u32) usize {
        return if (self.cfg.image_token_id == 0) 0 else std.mem.count(u32, ids, &.{self.cfg.image_token_id});
    }

    /// A token's Engram id; an image span's positions take no part in an n-gram. Without the request's image
    /// rows the image token is text, as in the reference's text-only forward.
    fn compressedId(self: *const Dsv41Model, t: u32, images: bool) u32 {
        if (images and self.cfg.image_token_id != 0 and t == self.cfg.image_token_id) return engram.Hash.DEAD;
        return if (t < self.token_map.len) self.token_map[t] else 0;
    }

    fn held(self: *const Dsv41Model, li: usize) bool {
        const p = self.pair orelse return true;
        return p.holds(li);
    }

    fn isKvSource(self: *const Dsv41Model, l: usize) bool {
        return l < self.n_layers and (self.cfg.dsv41_kv_sources >> @intCast(l)) & 1 != 0;
    }
    fn isIndexSource(self: *const Dsv41Model, l: usize) bool {
        return l < self.n_layers and (self.cfg.dsv41_index_sources >> @intCast(l)) & 1 != 0;
    }
    fn ratio(self: *const Dsv41Model, l: usize) usize {
        return if (l < self.n_layers) self.cfg.dsv4_compress_ratios[l] else 0;
    }
    /// The latest layer at or above `l` that matches `pred` (config parse
    /// proved one exists wherever a compressed layer asks).
    fn latest(self: *const Dsv41Model, l: usize, comptime pred: fn (*const Dsv41Model, usize) bool) usize {
        var j = l + 1;
        while (j > 0) : (j -= 1) {
            if (pred(self, j - 1)) return j - 1;
        }
        unreachable;
    }
    fn isOwner(self: *const Dsv41Model, l: usize) bool {
        return self.isKvSource(l) and self.isIndexSource(l);
    }
    fn dsparkTarget(self: *const Dsv41Model, l: usize) ?usize {
        for (self.cfg.dsv4_dspark_target_layers[0..self.cfg.dsv4_n_dspark_target_layers], 0..) |t, k| {
            if (t == l) return k;
        }
        return null;
    }
};

/// CED (the tech report's causal encoder-decoder): the decoder's first layer, when the layout has one. Every layer from
/// the last KV source on reads ratio-1 rows, so a prefill chunk runs them over its last `sliding_window` rows only.
/// Engram layers must sit in the encoder and the DSpark targets in the decoder, where the rows they read exist.
pub fn cedSplit(c: *const ModelConfig) ?usize {
    if (!c.has_sliding_window or c.sliding_window == 0 or c.dsv41_kv_sources == 0) return null;
    const n: usize = c.num_hidden_layers;
    const split: usize = 63 - @clz(c.dsv41_kv_sources);
    if (split == 0 or split >= n or n > c.dsv4_compress_ratios.len) return null;
    for (split..n) |l| if (c.dsv4_compress_ratios[l] != 1) return null;
    for (c.dsv41_engram_layers[0..c.dsv41_n_engram_layers]) |l| if (l >= split) return null;
    for (c.dsv4_dspark_target_layers[0..c.dsv4_n_dspark_target_layers]) |l| if (l < split) return null;
    return split;
}

/// `MLX_SERVE_DSV41_HEAD8=0` serves a dense LM head as the pack ships it instead of requantizing it to 8-bit.
pub fn head8Wanted() bool {
    return head8From(std.c.getenv("MLX_SERVE_DSV41_HEAD8"));
}

fn head8From(v: ?[*:0]const u8) bool {
    return v == null or v.?[0] != '0';
}

var ced_env: ?bool = null;
/// `MLX_SERVE_DSV41_CED=0` keeps every prefill chunk on the full forward.
fn cedEnabled() bool {
    if (ced_env == null) {
        const v = std.c.getenv("MLX_SERVE_DSV41_CED");
        ced_env = v == null or v.?[0] != '0';
    }
    return ced_env.?;
}

/// The rows of a chunk the decoder runs over (its last `sliding_window`) when the chunk replays: last-row logits
/// only, no verify block, no pair, and at least `ced_min_skip` rows to skip. A chunk that does not replay runs the
/// full forward, its window continuing from the rows the last replay wrote.
fn cedTail(m: *const Dsv41Model, st: *const State, L: usize, want: Want) ?usize {
    if (m.ced_split == null or want != .last or m.pair != null or st.verify != null) return null;
    const w: usize = m.cfg.sliding_window;
    if (L < w + m.ced_min_skip or !cedEnabled()) return null;
    return w;
}

const NameBuf = [192]u8;

const Trace = struct {
    gpa: std.mem.Allocator,
    items: std.ArrayList(struct { name: []u8, arr: A }) = .empty,

    fn put(self: *Trace, comptime fmt: []const u8, args: anytype, x: A) !void {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        try self.items.append(self.gpa, .{ .name = try std.fmt.allocPrint(self.gpa, fmt, args), .arr = o });
    }

    fn deinit(self: *Trace) void {
        for (self.items.items) |it| {
            self.gpa.free(it.name);
            _ = mlx.mlx_array_free(it.arr);
        }
        self.items.deinit(self.gpa);
    }
};

fn getW(w: *const model.Weights, comptime fmt: []const u8, args: anytype) !A {
    var buf: NameBuf = undefined;
    const name = std.fmt.bufPrint(&buf, fmt, args) catch return error.NameTooLong;
    return w.get(name) orelse {
        log.err("deepseek_v41: missing weight {s}\n", .{name});
        return error.MissingWeight;
    };
}

fn getOpt(w: *const model.Weights, comptime fmt: []const u8, args: anytype) ?A {
    var buf: NameBuf = undefined;
    const name = std.fmt.bufPrint(&buf, fmt, args) catch return null;
    return w.get(name);
}

fn getLin(w: *const model.Weights, in_dim: usize, comptime base: []const u8, args: anytype) !Lin {
    const wt = try getW(w, base ++ ".weight", args);
    const sc = getOpt(w, base ++ ".scales", args) orelse return .{ .w = wt };
    const bi = getOpt(w, base ++ ".biases", args) orelse none;
    const qp = transformer.quantParamsFromGeometry(wt, sc, bi.ctx != null, @intCast(in_dim)) orelse {
        var buf: NameBuf = undefined;
        log.err("deepseek_v41: {s} has a quantized layout this build does not read\n", .{std.fmt.bufPrint(&buf, base, args) catch base});
        return error.Dsv41QuantGeometry;
    };
    return .{ .w = wt, .s = sc, .b = bi, .qp = qp };
}

/// A dense weight that an mxfp8 code plus e8m0 scale per 32 columns holds
/// exactly (fp8 blocks a converter widened to bf16) as that pair, half the
/// bytes; null when any value would move.
fn exactMxfp8(s: S, w: A) !?[2]A {
    if (mlx.mlx_array_ndim(w) != 2 or @mod(dim(w, -1), 32) != 0) return null;
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, w, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, s));
    var q = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_vector_array_get(&q, qv, 0));
    var sc = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_vector_array_get(&sc, qv, 1));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_dequantize(&d, q, sc, none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = mlx.mlx_array_dtype(w), .has_value = true }, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, d, w, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    if (!same) {
        _ = mlx.mlx_array_free(q);
        _ = mlx.mlx_array_free(sc);
        return null;
    }
    const pair = [_]A{ q, sc };
    try evalAll(&pair);
    return pair;
}

/// `l` (dense) as affine `bits` over groups of `gs`, the dense weight released.
fn quantizeLin(m: *Dsv41Model, w: *model.Weights, l: Lin, bits: u32, gs: u32) !Lin {
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, l.w, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", none, m.s));
    var t: [3]A = undefined;
    for (&t, 0..) |*a, i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, qv, i));
        a.* = try m.own(o);
    }
    try evalAll(&t);
    release(m, w, l.w);
    return .{ .w = t[0], .s = t[1], .b = t[2], .qp = .{ .bits = bits, .group_size = gs } };
}

fn bankOf(l: Lin) Bank {
    return .{ .w = l.w, .s = l.s, .b = l.b, .qp = l.qp };
}

/// `x` cast to `dt` and evaluated, owned by the model.
fn ownCast(m: *Dsv41Model, x: A, dt: mlx.mlx_dtype) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, m.s));
    _ = try m.own(o);
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

fn ownReshape(m: *Dsv41Model, x: A, shape: []const c_int) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, m.s));
    _ = try m.own(o);
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

fn loadHc(m: *Dsv41Model, w: *const model.Weights, comptime pfx: []const u8, li: usize, comptime which: []const u8) !Hc {
    const hc = m.cfg.dsv4_hc_mult;
    const scale = try ownCast(m, try getW(w, pfx ++ ".{d}.hc_" ++ which ++ "_scale", .{li}), .float32);
    var ids: [64]i32 = undefined;
    const mix = (2 + hc) * hc;
    for (0..mix) |k| ids[k] = if (k < hc) 0 else if (k < 2 * hc) 1 else 2;
    const idx = mlx.mlx_array_new_data(&ids, &[_]c_int{ci(mix)}, 1, .int32);
    defer _ = mlx.mlx_array_free(idx);
    var sv = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&sv, scale, idx, 0, m.s));
    _ = try m.own(sv);
    try mlx.check(mlx.mlx_array_eval(sv));
    return .{
        .fn_w = try ownCast(m, try getW(w, pfx ++ ".{d}.hc_" ++ which ++ "_fn", .{li}), .float32),
        .scale = sv,
        .base = try ownCast(m, try getW(w, pfx ++ ".{d}.hc_" ++ which ++ "_base", .{li}), .float32),
    };
}

/// Routed experts: stacked banks under the pack's names, or per-expert
/// tensors stacked here.
/// `a` and `b` (same quantized format) as one array of their rows stacked
/// along `axis`; the originals released (owned, or the weights map's).
fn fuseRows(m: *Dsv41Model, w: *model.Weights, a: Lin, b: Lin, axis: c_int) !?Lin {
    if (a.s.ctx == null or b.s.ctx == null or a.qp.bits != b.qp.bits or a.qp.group_size != b.qp.group_size or a.qp.mode != b.qp.mode or (a.b.ctx == null) != (b.b.ctx == null)) return null;
    var out: Lin = .{ .w = none, .qp = a.qp };
    inline for (.{ "w", "s", "b" }) |f| {
        const x = @field(a, f);
        if (x.ctx != null) {
            const vec = mlx.mlx_vector_array_new_data(&[_]A{ x, @field(b, f) }, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_concatenate_axis(&o, vec, axis, m.s));
            @field(out, f) = try m.own(o);
            try mlx.check(mlx.mlx_array_eval(o));
        }
    }
    inline for (.{ "w", "s", "b" }) |f| {
        release(m, w, @field(a, f));
        release(m, w, @field(b, f));
    }
    return out;
}

/// `kv` and `gate` as `Comp.wkv` holds them above ratio 1: rows fused when the
/// formats agree (affine scales and biases widened to f32; dense rows as
/// stored, `denseF32x` reads them), else both dequantized to f32 rows.
fn poolLin(m: *Dsv41Model, w: *model.Weights, kv: Lin, gate: Lin) !Lin {
    if (try fuseRows(m, w, kv, gate, 0)) |f| {
        var out = f;
        if (f.qp.mode == .affine) inline for (.{ "s", "b" }) |n| if (@field(f, n).ctx != null) {
            @field(out, n) = try m.own(try evalOne(try astypeArr(@field(f, n), .float32, m.s)));
            release(m, w, @field(f, n));
        };
        return out;
    }
    const dense = kv.s.ctx == null and gate.s.ctx == null and mlx.mlx_array_dtype(kv.w) == mlx.mlx_array_dtype(gate.w);
    var rows: [2]A = undefined;
    for ([_]Lin{ kv, gate }, &rows) |l, *r| {
        var d = mlx.mlx_array_new();
        if (dense) try mlx.check(mlx.mlx_copy(&d, l.w, m.s)) else if (l.s.ctx == null) try mlx.check(mlx.mlx_astype(&d, l.w, .float32, m.s)) else try mlx.check(mlx.mlx_dequantize(&d, l.w, l.s, l.b, mlx.mlx_optional_int.some(@intCast(l.qp.group_size)), mlx.mlx_optional_int.some(@intCast(l.qp.bits)), l.qp.mode.cstr(), none, .{ .value = .float32, .has_value = true }, m.s));
        r.* = d;
    }
    defer for (rows) |r| {
        _ = mlx.mlx_array_free(r);
    };
    const vec = mlx.mlx_vector_array_new_data(&rows, 2);
    defer _ = mlx.mlx_vector_array_free(vec);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, vec, 0, m.s));
    const out: Lin = .{ .w = try m.own(try evalOne(o)) };
    inline for (.{ "w", "s", "b" }) |n| {
        release(m, w, @field(kv, n));
        release(m, w, @field(gate, n));
    }
    return out;
}

fn astypeArr(x: A, dt: mlx.mlx_dtype, s: S) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}

fn evalOne(x: A) !A {
    mlx.check(mlx.mlx_array_eval(x)) catch |e| {
        _ = mlx.mlx_array_free(x);
        return e;
    };
    return x;
}

/// Free `x` wherever it is held: the model's owned list or the weights map.
fn release(m: *Dsv41Model, w: *model.Weights, x: A) void {
    if (x.ctx == null) return;
    for (m.owned.items, 0..) |o, i| if (o.ctx == x.ctx) {
        _ = mlx.mlx_array_free(m.owned.swapRemove(i));
        return;
    };
    var it = w.map.iterator();
    while (it.next()) |kv| if (kv.value_ptr.ctx == x.ctx) {
        w.remove(kv.key_ptr.*);
        return;
    };
}

fn fuseExperts(m: *Dsv41Model, w: *model.Weights, e: Experts) !Experts {
    const g: Lin = .{ .w = e.gate.w, .s = e.gate.s, .b = e.gate.b, .qp = e.gate.qp };
    const u: Lin = .{ .w = e.up.w, .s = e.up.s, .b = e.up.b, .qp = e.up.qp };
    const f = (try fuseRows(m, w, g, u, 1)) orelse return e;
    const gu = bankOf(f);
    return .{ .gate = .{ .w = none }, .up = .{ .w = none }, .down = e.down, .gu = gu, .views = .{ try oneExpert(m, gu), try oneExpert(m, e.down) } };
}

fn oneExpert(m: *Dsv41Model, b: Bank) !Bank {
    var out = b;
    inline for (.{ "w", "s", "b" }) |f| if (@field(b, f).ctx != null) {
        const a = @field(b, f);
        var v = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&v, a, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, ci(dim(a, 1)), ci(dim(a, 2)) }, 3, &[_]c_int{ 1, 1, 1 }, 3, m.s));
        @field(out, f) = try m.own(v);
    };
    return out;
}

fn loadExperts(m: *Dsv41Model, w: *model.Weights, comptime pfx: []const u8, li: usize, n_experts: usize) !Experts {
    const d = m.cfg.hidden_size;
    const inter = m.cfg.moe_intermediate_size;
    if (getOpt(w, pfx ++ ".{d}.ffn.experts.gate_proj.weight", .{li}) != null) return .{
        .gate = bankOf(try getLin(w, d, pfx ++ ".{d}.ffn.experts.gate_proj", .{li})),
        .up = bankOf(try getLin(w, d, pfx ++ ".{d}.ffn.experts.up_proj", .{li})),
        .down = bankOf(try getLin(w, inter, pfx ++ ".{d}.ffn.experts.down_proj", .{li})),
    };
    if (getOpt(w, pfx ++ ".{d}.ffn.experts.w1.weight", .{li}) != null) return .{
        .gate = bankOf(try getLin(w, d, pfx ++ ".{d}.ffn.experts.w1", .{li})),
        .up = bankOf(try getLin(w, d, pfx ++ ".{d}.ffn.experts.w3", .{li})),
        .down = bankOf(try getLin(w, inter, pfx ++ ".{d}.ffn.experts.w2", .{li})),
    };
    if (getOpt(w, pfx ++ ".{d}.ffn.experts.0.w1.weight", .{li}) != null) {
        var banks: [3]Bank = undefined;
        inline for (.{ "w1", "w3", "w2" }, 0..) |proj, k| {
            const in_dim = if (k == 2) inter else d;
            var parts: [3]std.ArrayList(A) = .{ .empty, .empty, .empty };
            defer for (&parts) |*p| p.deinit(m.gpa);
            var first: Lin = undefined;
            for (0..n_experts) |e| {
                const l = try getLin(w, in_dim, pfx ++ ".{d}.ffn.experts.{d}." ++ proj, .{ li, e });
                if (e == 0) first = l;
                if (l.qp.bits != first.qp.bits or l.qp.group_size != first.qp.group_size or l.qp.mode != first.qp.mode or (l.b.ctx == null) != (first.b.ctx == null)) return error.Dsv41QuantGeometry;
                try parts[0].append(m.gpa, l.w);
                if (l.s.ctx != null) try parts[1].append(m.gpa, l.s);
                if (l.b.ctx != null) try parts[2].append(m.gpa, l.b);
            }
            var bank: Bank = .{ .w = undefined, .qp = first.qp };
            inline for (.{ .{ "w", "weight" }, .{ "s", "scales" }, .{ "b", "biases" } }, 0..) |f, p| {
                if (parts[p].items.len > 0) {
                    @field(bank, f[0]) = try m.own(try mlx.stackEvaluated(parts[p].items, m.s));
                    // The stack holds the experts now: the map's copies go, or the bank is resident twice.
                    for (0..n_experts) |e| {
                        var buf: NameBuf = undefined;
                        w.remove(std.fmt.bufPrint(&buf, pfx ++ ".{d}.ffn.experts.{d}." ++ proj ++ "." ++ f[1], .{ li, e }) catch return error.NameTooLong);
                    }
                }
            }
            banks[k] = bank;
        }
        return .{ .gate = banks[0], .up = banks[1], .down = banks[2] };
    }
    log.err("deepseek_v41: {s}.{d} has no routed experts\n", .{ pfx, li });
    return error.MissingWeight;
}

fn loadLayer(m: *Dsv41Model, w: *model.Weights, comptime pfx: []const u8, li: usize, trunk_li: ?usize) !Layer {
    const c = &m.cfg;
    const d = c.hidden_size;
    const heads = c.num_attention_heads;
    const hd = c.head_dim;
    const og = c.dsv4_o_groups;
    const ol = c.dsv4_o_lora_rank;
    const ratio: u8 = if (trunk_li) |t| c.dsv4_compress_ratios[t] else 0;
    const is_kv = if (trunk_li) |t| m.isKvSource(t) else false;
    const is_ix = if (trunk_li) |t| m.isIndexSource(t) else false;
    var wo_a = try getLin(w, heads * hd / og, pfx ++ ".{d}.attn.wo_a", .{li});
    if (wo_a.s.ctx == null) if (try exactMxfp8(m.s, wo_a.w)) |pair| {
        // A decode step reads every wo_a: half the bytes, the same values.
        wo_a = .{ .w = try m.own(pair[0]), .s = try m.own(pair[1]), .qp = .{ .bits = 8, .group_size = 32, .mode = .mxfp8 } };
        var buf: NameBuf = undefined;
        w.remove(try std.fmt.bufPrint(&buf, pfx ++ ".{d}.attn.wo_a.weight", .{li}));
    };
    inline for (.{ "w", "s", "b" }) |field| {
        const x = @field(wo_a, field);
        if (x.ctx != null) @field(wo_a, field) = try ownReshape(m, x, &.{ ci(og), ci(ol), @intCast(dim(x, -1)) });
    }
    const sink = blk: {
        const raw = try getW(w, pfx ++ ".{d}.attn.attn_sink", .{li});
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&r, raw, &[_]c_int{ 1, ci(heads), 1 }, 3, m.s));
        defer _ = mlx.mlx_array_free(r);
        break :blk try ownCast(m, r, .float32);
    };
    const n_experts: usize = if (trunk_li == null) c.dsv41_dspark_experts else c.num_experts;
    var ly: Layer = .{
        .attn_norm = try ownCast(m, try getW(w, pfx ++ ".{d}.attn_norm.weight", .{li}), m.act),
        .ffn_norm = try ownCast(m, try getW(w, pfx ++ ".{d}.ffn_norm.weight", .{li}), m.act),
        .q_norm = try ownCast(m, try getW(w, pfx ++ ".{d}.attn.q_norm.weight", .{li}), m.act),
        .kv_norm = try ownCast(m, try getW(w, pfx ++ ".{d}.attn.kv_norm.weight", .{li}), m.act),
        .hc_attn = try loadHc(m, w, pfx, li, "attn"),
        .hc_ffn = try loadHc(m, w, pfx, li, "ffn"),
        .wq_a = try getLin(w, d, pfx ++ ".{d}.attn.wq_a", .{li}),
        .wq_b = try getLin(w, c.dsv4_q_lora_rank, pfx ++ ".{d}.attn.wq_b", .{li}),
        .wkv = try getLin(w, d, pfx ++ ".{d}.attn.wkv", .{li}),
        .wo_a = wo_a,
        .wo_b = try getLin(w, og * ol, pfx ++ ".{d}.attn.wo_b", .{li}),
        .sink = sink,
        .ratio = ratio,
        .comp = if (is_kv) .{
            .wkv = if (ratio > 1) try poolLin(m, w, try getLin(w, d, pfx ++ ".{d}.attn.compressor.wkv", .{li}), try getLin(w, d, pfx ++ ".{d}.attn.compressor.wgate", .{li})) else try getLin(w, d, pfx ++ ".{d}.attn.compressor.wkv", .{li}),
            .norm = try ownCast(m, try getW(w, pfx ++ ".{d}.attn.compressor.norm.weight", .{li}), m.act),
        } else null,
        .idx = if (is_ix) .{
            .wq_b = try getLin(w, c.dsv4_q_lora_rank, pfx ++ ".{d}.attn.indexer.wq_b", .{li}),
            .weights_proj = try getLin(w, d, pfx ++ ".{d}.attn.indexer.weights_proj", .{li}),
            .wk = if (is_kv) try getLin(w, hd, pfx ++ ".{d}.attn.indexer.wk", .{li}) else null,
            .k_norm = if (is_kv) try ownCast(m, try getW(w, pfx ++ ".{d}.attn.indexer.k_norm.weight", .{li}), m.act) else none,
        } else null,
        .gate_w = try getW(w, pfx ++ ".{d}.ffn.gate.weight", .{li}),
        .gate_bias = try ownReshape(m, try ownCast(m, try getW(w, pfx ++ ".{d}.ffn.gate.bias", .{li}), .float32), &.{ 1, ci(n_experts) }),
        .top_k = if (trunk_li == null) c.dsv41_dspark_top_k else c.num_experts_per_tok,
        .shared = .{
            try getLin(w, d, pfx ++ ".{d}.ffn.shared_experts.w1", .{li}),
            try getLin(w, d, pfx ++ ".{d}.ffn.shared_experts.w3", .{li}),
            try getLin(w, c.moe_intermediate_size, pfx ++ ".{d}.ffn.shared_experts.w2", .{li}),
        },
        .experts = try fuseExperts(m, w, try loadExperts(m, w, pfx, li, n_experts)),
    };
    if (trunk_li != null) if (getOpt(w, pfx ++ ".{d}.ffn.gate.bias_vl", .{li})) |b| {
        ly.gate_bias_vl = try ownReshape(m, try ownCast(m, b, .float32), &.{ 1, ci(n_experts) });
    };
    if (try fuseRows(m, w, ly.shared[0], ly.shared[1], 0)) |f| {
        ly.shared_gu = f;
        ly.shared[0] = .{ .w = none };
        ly.shared[1] = .{ .w = none };
    }
    // DSpark's stages also project a window row alone (`kvRow`).
    if (trunk_li != null) if (try fuseRows(m, w, ly.wq_a, ly.wkv, 0)) |f| {
        ly.q_kv = f;
        ly.wq_a = .{ .w = none };
        ly.wkv = .{ .w = none };
    };
    return ly;
}

/// `inference/model.py` `precompute_freqs_cis` as fast_rope periods: YaRN
/// fades the dims past the training context by `factor` when `orig > 0`.
fn buildRope(m: *Dsv41Model, base: f64, orig: u32, factor: f64, beta_fast: f64, beta_slow: f64) !Rope {
    const rd = m.cfg.dsv4_rope_head_dim;
    var fwd: [256]f32 = undefined;
    var inv: [256]f32 = undefined;
    const dimf: f64 = @floatFromInt(rd);
    var low: f64 = 0;
    var high: f64 = 0;
    if (orig > 0) {
        const corr = struct {
            fn f(rot: f64, dd: f64, b: f64, o: f64) f64 {
                return dd * @log(o / (rot * 2.0 * std.math.pi)) / (2.0 * @log(b));
            }
        }.f;
        const o: f64 = @floatFromInt(orig);
        low = @max(@floor(corr(beta_fast, dimf, base, o)), 0);
        high = @min(@ceil(corr(beta_slow, dimf, base, o)), dimf - 1);
    }
    for (0..rd / 2) |k| {
        const e: f32 = @as(f32, @floatFromInt(2 * k)) / @as(f32, @floatFromInt(rd));
        var fr: f32 = 1.0 / std.math.pow(f32, @floatCast(base), e);
        if (orig > 0) {
            const ramp: f32 = @floatCast(std.math.clamp((@as(f64, @floatFromInt(k)) - low) / @max(high - low, 1e-3), 0, 1));
            const smooth = 1 - ramp;
            fr = fr / @as(f32, @floatCast(factor)) * (1 - smooth) + fr * smooth;
        }
        fwd[k] = 1.0 / fr;
        inv[k] = -fwd[k];
    }
    const shape = [_]c_int{ci(rd / 2)};
    return .{
        .fwd = try m.own(mlx.mlx_array_new_data(&fwd, &shape, 1, .float32)),
        .inv = try m.own(mlx.mlx_array_new_data(&inv, &shape, 1, .float32)),
    };
}

/// Decode attention for one (row, head) per threadgroup: the window rows the
/// query sees (read from the ring after this step's write) then its selected
/// compressed rows, f32 online softmax split over NSG simdgroups, the sink in
/// the denominator only. meta = {p0, L, K, window, cap, n_comp}.
const ATTN_DEC_BODY =
    \\const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int i = threadgroup_position_in_grid.y, h = threadgroup_position_in_grid.x;
    \\const int p0 = meta[0], K = meta[2], window = meta[3], cap = meta[4], nc = meta[5];
    \\const int H = q_shape[1];
    \\constexpr int V = D / 32;
    \\const int pos = p0 + i;
    \\const int lo = metal::max(0, pos - window + 1);
    \\const int nw = pos - lo + 1;
    \\float qv[V], acc[V];
    \\for (int v = 0; v < V; ++v) { qv[v] = float(q[(i * H + h) * D + lane * V + v]); acc[v] = 0.0f; }
    \\float mx = -INFINITY, den = 0.0f;
    \\const float scale = scal[0];
    \\// Split sp of S takes rows [t_lo, t_hi) of the window + selected list.
    \\const int sp = threadgroup_position_in_grid.z, chunk = (nw + K + S - 1) / S;
    \\const int t_lo = sp * chunk, t_hi = metal::min(nw + K, t_lo + chunk);
    \\for (int t = t_lo + sg; t < t_hi; t += NSG) {
    \\  const device KT* row;
    \\  if (t < nw) row = ring + size_t((lo + t) % cap) * D;
    \\  else {
    \\    const int c = topk[i * K + t - nw];
    \\    if (c < 0 || c >= nc) continue;
    \\    row = comp + size_t(c) * D;
    \\  }
    \\  float kv[V], dot = 0.0f;
    \\  for (int v = 0; v < V; ++v) { kv[v] = float(row[lane * V + v]); dot += qv[v] * kv[v]; }
    \\  const float s = metal::simd_sum(dot) * scale;
    \\  const float nm = metal::max(mx, s);
    \\  const float corr = metal::exp(mx - nm), p = metal::exp(s - nm);
    \\  den = den * corr + p;
    \\  for (int v = 0; v < V; ++v) acc[v] = acc[v] * corr + p * kv[v];
    \\  mx = nm;
    \\}
    \\threadgroup float t_acc[NSG * D];
    \\threadgroup float t_mx[NSG], t_den[NSG];
    \\for (int v = 0; v < V; ++v) t_acc[sg * D + lane * V + v] = acc[v];
    \\if (lane == 0) { t_mx[sg] = mx; t_den[sg] = den; }
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
;
const ATTN_DEC_SOURCE = ATTN_DEC_BODY ++
    \\float M = sink[h];
    \\for (int g = 0; g < NSG; ++g) M = metal::max(M, t_mx[g]);
    \\float total = metal::exp(sink[h] - M);
    \\float w[NSG];
    \\for (int g = 0; g < NSG; ++g) { w[g] = t_den[g] > 0.0f ? metal::exp(t_mx[g] - M) : 0.0f; total += t_den[g] * w[g]; }
    \\for (int d = sg * 32 + lane; d < D; d += NSG * 32) {
    \\  float o = 0.0f;
    \\  for (int g = 0; g < NSG; ++g) o += t_acc[g * D + d] * w[g];
    \\  out[(i * H + h) * D + d] = dsv41_unrope<OT, RD, D>(OT(o / total), d, p0 + i, periods);
    \\}
;

/// `ropeTail` with `attnOut`'s inverse periods on a head channel the attention
/// just wrote (`v`, at position `pos`): pairs are adjacent lanes, so every lane
/// of the simdgroup must call it. `P`: MLX passes `periods` as a constant
/// pointer under 8 elements, else a device one.
const UNROPE_HEADER =
    \\template <typename OT, int RD, int D, typename P>
    \\inline OT dsv41_unrope(OT v, int d, int pos, P periods) {
    \\  const float pv = metal::simd_shuffle_xor(float(v), ushort(1));
    \\  if (RD == 0 || d < D - RD) return v;
    \\  const bool even = (d - (D - RD)) % 2 == 0;
    \\  const float x1 = even ? float(v) : pv, x2 = even ? pv : float(v);
    \\  const float theta = float(pos) * (1.0f / periods[(d - (D - RD)) / 2]);
    \\  const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
    \\  return OT(even ? x1 * cs - x2 * sn : x1 * sn + x2 * cs);
    \\}
;

/// `ATTN_DEC_SOURCE` over one of S splits of its rows: the split's max, sum and
/// unnormalized output (`ATTN_MERGE_SOURCE` finishes it).
const ATTN_SPLIT_SOURCE = ATTN_DEC_BODY ++
    \\float M = -INFINITY, l = 0.0f, w[NSG];
    \\for (int g = 0; g < NSG; ++g) M = metal::max(M, t_mx[g]);
    \\for (int g = 0; g < NSG; ++g) { w[g] = t_den[g] > 0.0f ? metal::exp(t_mx[g] - M) : 0.0f; l += t_den[g] * w[g]; }
    \\const size_t base = (size_t(i) * H + h) * S + sp;
    \\for (int d = sg * 32 + lane; d < D; d += NSG * 32) {
    \\  float o = 0.0f;
    \\  for (int g = 0; g < NSG; ++g) o += t_acc[g * D + d] * w[g];
    \\  pacc[base * D + d] = o;
    \\}
    \\if (sg == 0 && lane == 0) { pm[base] = M; pl[base] = l; }
;

/// The S partials of one (row, head) `ATTN_SPLIT_SOURCE` wrote, merged with
/// the sink in the denominator, then `attnOut`'s inverse rope (as
/// `ATTN_DEC_SOURCE` ends); one thread a channel.
const ATTN_MERGE_SOURCE =
    \\const int d = thread_position_in_threadgroup.x, h = threadgroup_position_in_grid.y, i = threadgroup_position_in_grid.z;
    \\const int H = pm_shape[1];
    \\const size_t base = (size_t(i) * H + h) * S;
    \\float M = sink[h];
    \\for (int j = 0; j < S; ++j) M = metal::max(M, pm[base + j]);
    \\float total = metal::exp(sink[h] - M), o = 0.0f;
    \\for (int j = 0; j < S; ++j) {
    \\  const float w = pl[base + j] > 0.0f ? metal::exp(pm[base + j] - M) : 0.0f;
    \\  total += pl[base + j] * w;
    \\  o += pacc[(base + j) * D + d] * w;
    \\}
    \\out[(size_t(i) * H + h) * D + d] = dsv41_unrope<OT, RD, D>(OT(o / total), d, meta[0] + i, periods);
;

/// Swept on M5 Ultra over 4..8 simdgroups and 1..4 rows a pass: 8 x 1.
const ATTN_DEC_NSG = 8;

/// A wide span's softmax, one simdgroup per (row, head): the raw window scores
/// `lw` [L*H, Wk] under the band (query l sees columns (wp + l - window, wp + l]),
/// the raw selected scores `lc` [L, H, K] where the pick is >= 0, scaled by
/// `scal` here in f32, max and sum in f32 with the sink in the denominator
/// only, the normalized probabilities written in `OT`. NJ columns a lane.
/// meta = {wp, window, Wk, K}.
const ATTN_SOFTMAX_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int h = thread_position_in_grid.x / 32, l = thread_position_in_grid.y;
    \\const int H = sink_shape[1];
    \\const int wp = meta[0], window = meta[1], Wk = meta[2], K = meta[3];
    \\const int N = Wk + K;
    \\const float scale = scal[0];
    \\const size_t rw = size_t(l) * H + h;
    \\float s[NJ];
    \\float mx = -INFINITY;
    \\for (int j = 0; j < NJ; ++j) {
    \\  const int c = lane + 32 * j;
    \\  float v = -INFINITY;
    \\  if (c < Wk) { if (c <= wp + l && c + window > wp + l) v = scale * float(lw[rw * Wk + c]); }
    \\  else if (c < N) { if (topk[size_t(l) * K + c - Wk] >= 0) v = scale * float(lc[rw * K + c - Wk]); }
    \\  s[j] = v;
    \\  mx = metal::max(mx, v);
    \\}
    \\mx = metal::simd_max(mx);
    \\const float sk = sink[h];
    \\const float M = metal::max(mx, sk);
    \\float den = 0.0f;
    \\for (int j = 0; j < NJ; ++j) { const float p = metal::exp(s[j] - M); s[j] = p; den += p; }
    \\den = metal::simd_sum(den) + metal::exp(sk - M);
    \\const float inv = 1.0f / den;
    \\for (int j = 0; j < NJ; ++j) {
    \\  const int c = lane + 32 * j;
    \\  if (c < Wk) pw[rw * Wk + c] = OT(s[j] * inv);
    \\  else if (c < N) pc[rw * K + c - Wk] = OT(s[j] * inv);
    \\}
;

/// The most window + selected columns `attnWide` scores (NJ <= 32 floats a lane); wider spans keep the f32 chain.
const ATTN_SOFTMAX_MAX_COLS = 1024;

/// The indexer's scores `[L, nb]`: sum over heads of weight * relu(q_h . k_n),
/// a weight being its projection in f32 times `wsc`, keys read in their cached
/// dtype, -inf past the groups row l's query sees ((p0 + l + 1) / ratio).
/// meta = {p0, ratio}.
const IDX_SCORE_SOURCE =
    \\const int n = thread_position_in_grid.x, l = thread_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, tgn = threads_per_threadgroup.x;
    \\const int nb = keys_shape[0];
    \\threadgroup float qs[IH * IHD];
    \\for (uint k = tid; k < IH * IHD; k += tgn) qs[k] = float(q[l * IH * IHD + k]);
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (n >= nb) return;
    \\float dots[IH];
    \\for (int h = 0; h < IH; ++h) dots[h] = 0.0f;
    \\const device KT* kr = keys + size_t(n) * IHD;
    \\for (int j = 0; j < IHD; ++j) {
    \\  const float kv = float(kr[j]);
    \\  for (int h = 0; h < IH; ++h) dots[h] += qs[h * IHD + j] * kv;
    \\}
    \\float s = 0.0f;
    \\for (int h = 0; h < IH; ++h) s += (float(w[l * IH + h]) * wsc[0]) * metal::max(dots[h], 0.0f);
    \\const int seen = (meta[0] + l + 1) / meta[1];
    \\scores[size_t(l) * nb + n] = n < seen ? s : -INFINITY;
;

/// `IDX_SCORE_SOURCE` for a wide block: one threadgroup a (query, KT-key tile), its four simdgroups
/// each KT/4 keys x every head on 8x8 matrix ops over the keys staged in threadgroup memory; a head
/// tile is relu'd, weighted and summed as it completes, so no [L, IH, nb] scratch. A tile the query
/// cannot see is -inf without the work. `q` arrives in the keys' dtype: the rope kernel rounded it
/// to that dtype before widening, so the cast is exact.
const IDX_TILE_SOURCE =
    \\const int l = threadgroup_position_in_grid.y, n0 = threadgroup_position_in_grid.x * KT;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int nb = keys_shape[0];
    \\const int seen = (meta[0] + l + 1) / meta[1];
    \\const int end = metal::min(nb, n0 + KT);
    \\if (n0 >= seen) {
    \\  for (int n = n0 + int(tid); n < end; n += 128) scores[size_t(l) * nb + n] = -INFINITY;
    \\  return;
    \\}
    \\constexpr int RT = IH / 8, CT = KT / 32, C4 = IHD / 4;
    \\using T4 = metal::vec<T, 4>;
    \\threadgroup T kt[KT * IHD];
    \\threadgroup float wt[IH], red[4][64];
    \\for (uint i = tid; i < KT * C4; i += 128) {
    \\  const int n = n0 + int(i / C4);
    \\  ((threadgroup T4*)kt)[i] = n < nb ? ((const device T4*)keys)[size_t(n) * C4 + i % C4] : T4(0);
    \\}
    \\if (tid < IH) wt[tid] = float(w[size_t(l) * IH + tid]) * wsc[0];
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\const device T* qr = q + size_t(l) * IH * IHD;
    \\const threadgroup T* kr = kt + sg * (KT / 4) * IHD;
    \\metal::simdgroup_matrix<float, 8, 8> acc[RT][CT];
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < RT; ++r)
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < CT; ++c) acc[r][c] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\for (int d = 0; d < IHD; d += 8) {
    \\  metal::simdgroup_matrix<T, 8, 8> a[RT], b[CT];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT; ++r) metal::simdgroup_load(a[r], qr + r * 8 * IHD + d, IHD);
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < CT; ++c) metal::simdgroup_load(b[c], kr + c * 8 * IHD + d, IHD, metal::ulong2(0, 0), true);
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < RT; ++r)
    \\    #pragma clang loop unroll(full)
    \\    for (int c = 0; c < CT; ++c) metal::simdgroup_multiply_accumulate(acc[r][c], a[r], b[c], acc[r][c]);
    \\}
    \\// A finished tile's rows are heads: relu, weight and sum them through this simdgroup's scratch.
    \\// Threadgroup barriers: on M1 a simdgroup barrier let lanes read the scratch before the store landed.
    \\const int col = lane & 7, row = (lane >> 3) * 2;
    \\float s[CT];
    \\#pragma clang loop unroll(full)
    \\for (int c = 0; c < CT; ++c) s[c] = 0.0f;
    \\#pragma clang loop unroll(full)
    \\for (int r = 0; r < RT; ++r)
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < CT; ++c) {
    \\    metal::simdgroup_store(acc[r][c], red[sg], 8);
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\    float p = metal::max(red[sg][row * 8 + col], 0.0f) * wt[r * 8 + row] + metal::max(red[sg][(row + 1) * 8 + col], 0.0f) * wt[r * 8 + row + 1];
    \\    p += metal::simd_shuffle_xor(p, 8);
    \\    p += metal::simd_shuffle_xor(p, 16);
    \\    s[c] += p;
    \\    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  }
    \\if (lane < 8)
    \\  for (int c = 0; c < CT; ++c) {
    \\    const int n = n0 + int(sg) * (KT / 4) + c * 8 + col;
    \\    if (n < nb) scores[size_t(l) * nb + n] = n < seen ? s[c] : -INFINITY;
    \\  }
;

/// Router logits `[L, E]` in f32 from bf16 rows: one simdgroup an expert,
/// 8 inputs a lane per step.
const GATE_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int e = threadgroup_position_in_grid.x * SGS + simdgroup_index_in_threadgroup;
    \\const int l = threadgroup_position_in_grid.y;
    \\const int E = w_shape[0], D = w_shape[1];
    \\if (e >= E) return;
    \\const device auto* wr = w + size_t(e) * D;
    \\const device auto* xr = x + size_t(l) * D;
    \\float acc = 0.0f;
    \\for (int k = lane * 8; k < D; k += 256)
    \\  for (int j = 0; j < 8; ++j) acc += float(xr[k + j]) * float(wr[k + j]);
    \\acc = metal::simd_sum(acc);
    \\if (lane == 0) logits[size_t(l) * E + e] = acc;
;

/// One row a threadgroup, one expert a thread: sqrt-softplus scores, the
/// bias-picked top K (lowest index on ties), their scores normalized
/// (NORM) and scaled.
const ROUTE_SOURCE =
    \\const int e = thread_position_in_threadgroup.x, l = threadgroup_position_in_grid.y;
    \\const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int E = logits_shape[1];
    \\threadgroup float t_s[1024], t_v[32];
    \\threadgroup int t_i[32], t_pick[K];
    \\float s = 0.0f, v = -INFINITY;
    \\if (e < E) {
    \\  const float z = logits[size_t(l) * E + e];
    \\  const float y = metal::exp(-metal::abs(z));
    \\  const float lp = y < 1e-4f ? y * (1.0f - y * (0.5f - y / 3.0f)) : metal::log(1.0f + y);
    \\  s = metal::sqrt(metal::max(z, 0.0f) + lp);
    \\  v = s + bias[(bias_shape[0] == 1 ? 0 : l) * E + e];
    \\  t_s[e] = s;
    \\}
    \\for (int r = 0; r < K; ++r) {
    \\  const float bv = metal::simd_max(v);
    \\  const int bi = metal::simd_min(v == bv && e < E ? e : 0x7fffffff);
    \\  if (lane == 0) { t_v[sg] = bv; t_i[sg] = bi; }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (e == 0) {
    \\    const int nsg = (threads_per_threadgroup.x + 31) / 32;
    \\    float best = t_v[0]; int at = t_i[0];
    \\    for (int g = 1; g < nsg; ++g) if (t_v[g] > best || (t_v[g] == best && t_i[g] < at)) { best = t_v[g]; at = t_i[g]; }
    \\    t_pick[r] = at;
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (e == t_pick[r]) v = -INFINITY;
    \\}
    \\if (e == 0) {
    \\  float sum = 0.0f;
    \\  for (int r = 0; r < K; ++r) sum += t_s[t_pick[r]];
    \\  const float div = NORM ? sum + 1e-20f : 1.0f;
    \\  for (int r = 0; r < K; ++r) {
    \\    idx[l * K + r] = uint(t_pick[r]);
    \\    wts[l * K + r] = t_s[t_pick[r]] / div * scal[0];
    \\  }
    \\}
;

/// `x` [L, K] f32 times dense rows `w` [N, K] (any float dtype) in f32,
/// RPS rows a simdgroup: the product the reference computes after widening w.
const DENSE_F32X_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int n0 = thread_position_in_grid.y * RPS, l = thread_position_in_grid.z;
    \\const int K = x_shape[1], N = w_shape[0];
    \\float acc[RPS];
    \\for (int r = 0; r < RPS; ++r) acc[r] = 0.0f;
    \\const device float* xr = x + size_t(l) * K;
    \\for (int k = int(lane) * 4; k < K; k += 128) {
    \\  const float4 xv = *(const device float4*)(xr + k);
    \\  for (int r = 0; r < RPS; ++r) {
    \\    const device auto* wr = w + size_t(n0 + r) * K + k;
    \\    acc[r] += xv.x * float(wr[0]) + xv.y * float(wr[1]) + xv.z * float(wr[2]) + xv.w * float(wr[3]);
    \\  }
    \\}
    \\for (int r = 0; r < RPS; ++r) {
    \\  const float v = metal::simd_sum(acc[r]);
    \\  if (lane == 0 && n0 + r < N) out[size_t(l) * N + n0 + r] = v;
    \\}
;
const dense_f32x_rps = 4;

/// The K largest of each row of `x` [L, n] f32 (K <= n), as their indices in
/// ascending order, ties to the lower index: a radix select of the order-
/// preserving uint keys, 8 bits a pass, then one pass writing every key above
/// the threshold and the first ties. One threadgroup a row.
const TOPK_SOURCE =
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int l = threadgroup_position_in_grid.y, n = x_shape[1];
    \\const auto xr = x + size_t(l) * n;
    \\threadgroup atomic_uint hist[256];
    \\threadgroup uint t_sel[2], t_part[TG / 32];
    \\uint prefix = 0, krem = K;
    \\for (int pass = 0; pass < 4; ++pass) {
    \\  const int shift = 24 - 8 * pass;
    \\  const uint hmask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
    \\  for (uint b = tid; b < 256; b += TG) atomic_store_explicit(&hist[b], 0u, metal::memory_order_relaxed);
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  for (int i = int(tid); i < n; i += TG) {
    \\    const uint key = dsv41_key(xr[i]);
    \\    if ((key & hmask) == prefix) atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, metal::memory_order_relaxed);
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  if (sg == 0) {
    \\    // Bins 255..0 in lane order of 8 bins each: the digit where the count from the top reaches krem.
    \\    uint c[8], tot = 0;
    \\    for (int j = 0; j < 8; ++j) { c[j] = atomic_load_explicit(&hist[255 - (lane * 8 + j)], metal::memory_order_relaxed); tot += c[j]; }
    \\    const uint before = metal::simd_prefix_exclusive_sum(tot);
    \\    const bool mine = before < krem && before + tot >= krem;
    \\    if (mine) {
    \\      uint acc = before;
    \\      int j = 0;
    \\      for (; j < 8; ++j) { if (acc + c[j] >= krem) break; acc += c[j]; }
    \\      t_sel[0] = prefix | (uint(255 - (lane * 8 + j)) << shift);
    \\      t_sel[1] = krem - acc;
    \\    }
    \\  }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  prefix = t_sel[0];
    \\  krem = t_sel[1];
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\}
    \\// Contiguous index chunks a thread, so the output keeps index order.
    \\const int C = (n + TG - 1) / TG, i0 = int(tid) * C, i1 = metal::min(n, i0 + C);
    \\uint gt = 0, eq = 0;
    \\for (int i = i0; i < i1; ++i) { const uint key = dsv41_key(xr[i]); gt += key > prefix; eq += key == prefix; }
    \\// Exclusive scans across the threadgroup: ties before this chunk, then picks before it.
    \\uint s = metal::simd_prefix_exclusive_sum(eq);
    \\if (lane == 31) t_part[sg] = s + eq;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\uint off = 0;
    \\for (uint g = 0; g < sg; ++g) off += t_part[g];
    \\const uint eq_before = s + off;
    \\const uint take = eq_before >= krem ? 0u : metal::min(eq, krem - eq_before);
    \\const uint sel = gt + take;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\s = metal::simd_prefix_exclusive_sum(sel);
    \\if (lane == 31) t_part[sg] = s + sel;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\off = 0;
    \\for (uint g = 0; g < sg; ++g) off += t_part[g];
    \\uint pos = s + off, taken = 0;
    \\for (int i = i0; i < i1; ++i) {
    \\  const uint key = dsv41_key(xr[i]);
    \\  if (key > prefix || (key == prefix && taken < take)) {
    \\    taken += key == prefix;
    \\    out[size_t(l) * K + pos++] = i;
    \\  }
    \\}
;

const TOPK_HEADER =
    \\inline uint dsv41_key(float f) {
    \\  const uint u = as_type<uint>(f);
    \\  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    \\}
;
const topk_tg = 1024;

/// `hcSplit`'s projection: per row, the 24 mixing logits of the flattened
/// stream, rms-scaled, times scale plus base. One threadgroup an R-row x M-mix
/// tile, so each fn and stream load serves the tile; every (row, mix) dot keeps
/// `HC_STAGE_SOURCE`'s order (thread tid the terms k = tid + c * TG in c order,
/// one simd sum, the simdgroups in order), which is what makes the two bit-equal.
const HC_MIX_SOURCE =
    \\const int m0 = threadgroup_position_in_grid.x * M, l0 = threadgroup_position_in_grid.y * R;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int N = fn_shape[1], MIX = fn_shape[0], L = x_shape[0];
    \\float d[R][M], ss[R];
    \\for (int r = 0; r < R; ++r) { ss[r] = 0.0f; for (int j = 0; j < M; ++j) d[r][j] = 0.0f; }
    \\for (int k = tid; k < N; k += TG) {
    \\  float fv[M];
    \\  #pragma clang loop unroll(full)
    \\  for (int j = 0; j < M; ++j) fv[j] = fn[size_t(m0 + j) * N + k];
    \\  #pragma clang loop unroll(full)
    \\  for (int r = 0; r < R; ++r) {
    \\    const float v = l0 + r < L ? float(x[size_t(l0 + r) * N + k]) : 0.0f;
    \\    #pragma clang loop unroll(full)
    \\    for (int j = 0; j < M; ++j) d[r][j] += v * fv[j];
    \\    ss[r] += v * v;
    \\  }
    \\}
    \\threadgroup float t_d[R][M][TG / 32], t_s[R][TG / 32];
    \\for (int r = 0; r < R; ++r) {
    \\  for (int j = 0; j < M; ++j) { const float v = metal::simd_sum(d[r][j]); if (lane == 0) t_d[r][j][sg] = v; }
    \\  const float v = metal::simd_sum(ss[r]);
    \\  if (lane == 0) t_s[r][sg] = v;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (tid < R * M) {
    \\  const int r = tid / M, j = tid % M;
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int g = 0; g < TG / 32; ++g) { a += t_d[r][j][g]; b += t_s[r][g]; }
    \\  if (l0 + r < L) z[(l0 + r) * MIX + m0 + j] = a * metal::precise::rsqrt(b / float(N) + eps[0]) * scale[m0 + j] + base[m0 + j];
    \\}
;

/// `HC_MIX_SOURCE`'s tile: rows a threadgroup, and the mixes a threadgroup
/// (the largest of these that divides the mix count).
const hc_mix_rows = 4;
const hc_mix_cols = [_]usize{ 6, 4, 3, 2, 1 };

/// One hyper-connection stage as two kernels. The staggered collapse reads the PREVIOUS mix (`cpre`), so in
/// `HC_STAGE_SOURCE` threadgroup 0 collapses and rms-norms the stream without waiting for this stage's
/// mixes, while threadgroups 1..MIX each take one mix's dot; the FFN side rebuilds the stream from the
/// block output in every threadgroup and threadgroup 0 stores it. The split (`HC_SPLIT_SOURCE`) is read
/// only later, so the next matmul runs beside it.
const HC_STAGE_HEAD =
    \\const uint role = threadgroup_position_in_grid.x;
    \\const int l = threadgroup_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int D = x_shape[2], N = fn_shape[1];
    \\threadgroup float t_a[TG / 32], t_b[TG / 32];
    \\const device auto* xr = x + size_t(l) * HC * D;
    \\float v[HC][ND];
    \\for (int c = 0; c < ND; ++c) {
    \\  const int d = int(tid) + c * TG;
    \\  for (int i = 0; i < HC; ++i) v[i][c] = 0.0f;
;

/// The stream is `x` itself.
const HC_STAGE_LOAD_PLAIN =
    \\  if (d < D) for (int i = 0; i < HC; ++i) v[i][c] = float(xr[i * D + d]);
    \\}
;

/// The stream is `hcExpand(x, y, previous split)`, rounded to T where the expand rounds it.
const HC_STAGE_LOAD_EXPAND =
    \\  if (d < D) {
    \\    const auto pr = spost + l * HC;
    \\    const auto cb = scomb + l * HC * HC;
    \\    float sv[HC];
    \\    for (int j = 0; j < HC; ++j) sv[j] = float(xr[j * D + d]);
    \\    const float yv = float(y[size_t(l) * D + d]);
    \\    for (int i = 0; i < HC; ++i) {
    \\      float acc = 0.0f;
    \\      for (int j = 0; j < HC; ++j) acc = metal::fma(cb[j * HC + i], sv[j], acc);
    \\      const T o = T(metal::fma(pr[i], yv, acc));
    \\      v[i][c] = float(o);
    \\      if (role == 0) x1[(size_t(l) * HC + i) * D + d] = o;
    \\    }
    \\  }
    \\}
;

const HC_STAGE_TAIL =
    \\if (role == 0) {
    \\  float cp[HC];
    \\  for (int j = 0; j < HC; ++j) cp[j] = cpre[l * HC + j];
    \\  float ss = 0.0f;
    \\  T hv[ND];
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < ND; ++c) {
    \\    float h = 0.0f;
    \\    for (int j = 0; j < HC; ++j) h += cp[j] * v[j][c];
    \\    hv[c] = T(h);
    \\    ss += float(hv[c]) * float(hv[c]);
    \\  }
    \\  ss = metal::simd_sum(ss);
    \\  if (lane == 0) t_a[sg] = ss;
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  float tot = 0.0f;
    \\  for (int g = 0; g < TG / 32; ++g) tot += t_a[g];
    \\  const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < ND; ++c) {
    \\    const int d = int(tid) + c * TG;
    \\    if (d < D) h_out[size_t(l) * D + d] = w[d] * T(float(hv[c]) * inv);
    \\  }
    \\  return;
    \\}
    \\const int m = int(role) - 1;
    \\const device float* fr = fn + size_t(m) * N;
    \\float dd = 0.0f, sq = 0.0f;
    \\#pragma clang loop unroll(full)
    \\for (int i = 0; i < HC; ++i)
    \\  #pragma clang loop unroll(full)
    \\  for (int c = 0; c < ND; ++c) {
    \\    const int d = int(tid) + c * TG;
    \\    const float u = v[i][c];
    \\    if (d < D) dd += u * fr[i * D + d];
    \\    sq += u * u;
    \\  }
    \\dd = metal::simd_sum(dd); sq = metal::simd_sum(sq);
    \\if (lane == 0) { t_a[sg] = dd; t_b[sg] = sq; }
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\if (tid == 0) {
    \\  float a = 0.0f, b = 0.0f;
    \\  for (int g = 0; g < TG / 32; ++g) { a += t_a[g]; b += t_b[g]; }
    \\  z[l * ((2 + HC) * HC) + m] = a * metal::precise::rsqrt(b / float(N) + eps[0]) * scale[m] + base[m];
    \\}
;

/// The hc split of the mixes `z` (pre = sigmoid + eps, post = 2 sigmoid, Sinkhorn comb); one simdgroup a row.
const HC_SPLIT_SOURCE =
    \\const int l = threadgroup_position_in_grid.y;
    \\const uint lane = thread_index_in_threadgroup;
    \\constexpr int MIX = (2 + HC) * HC;
    \\const float e = eps[1];
    \\const device float* zr = z + l * MIX;
    \\if (lane < HC) {
    \\  pre[l * HC + lane] = 1.0f / (1.0f + metal::exp(-zr[lane])) + e;
    \\  post[l * HC + lane] = 2.0f / (1.0f + metal::exp(-zr[HC + lane]));
    \\}
    \\// Lane a * HC + b holds comb[a][b]; each sum gathers its HC terms in the serial order.
    \\const bool act = lane < HC * HC;
    \\const ushort a = act ? lane / HC : 0, b = act ? lane % HC : 0;
    \\float v = zr[2 * HC + a * HC + b];
    \\float mx = metal::simd_shuffle(v, a * HC);
    \\for (ushort k = 1; k < HC; ++k) mx = metal::max(mx, metal::simd_shuffle(v, a * HC + k));
    \\v = metal::exp(v - mx);
    \\float sm = 0.0f;
    \\for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, a * HC + k);
    \\v = v / sm + e;
    \\for (int t = 0; t < 2 * ITERS - 1; ++t) {
    \\  const bool rows = t % 2 == 1;
    \\  sm = 0.0f;
    \\  for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, rows ? a * HC + k : k * HC + b);
    \\  v = v / (sm + e);
    \\}
    \\if (act) comb[l * HC * HC + lane] = v;
;

const HC_STAGE_SOURCE = HC_STAGE_HEAD ++ HC_STAGE_LOAD_PLAIN ++ HC_STAGE_TAIL;
const HC_STAGE_X_SOURCE = HC_STAGE_HEAD ++ HC_STAGE_LOAD_EXPAND ++ HC_STAGE_TAIL;
const HC_STAGE_INS = [_][*:0]const u8{ "x", "cpre", "w", "fn", "scale", "base", "eps" };
const HC_STAGE_X_INS = [_][*:0]const u8{ "x", "y", "spost", "scomb", "cpre", "w", "fn", "scale", "base", "eps" };
const HC_STAGE_OUTS = [_][*:0]const u8{ "h_out", "z" };
const HC_STAGE_X_OUTS = [_][*:0]const u8{ "h_out", "z", "x1" };

/// The hc split (pre = sigmoid + eps, post = 2 sigmoid, Sinkhorn comb; one simdgroup), then the stream collapsed with
/// `cpre` (f32, rounded to T like `hcCollapse`) and rms-normed like MLX's
/// `rms_norm`. One threadgroup a row.
const HC_NORM_SOURCE =
    \\const int l = threadgroup_position_in_grid.y;
    \\const uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int D = x_shape[2];
    \\constexpr int MIX = (2 + HC) * HC;
    \\const float e = eps[1];
    \\if (sg == 0) {
    \\  // Lane a * HC + b holds comb[a][b]; each sum gathers its HC terms in the serial order.
    \\  const device float* zr = z + l * MIX;
    \\  if (lane < HC) {
    \\    pre[l * HC + lane] = 1.0f / (1.0f + metal::exp(-zr[lane])) + e;
    \\    post[l * HC + lane] = 2.0f / (1.0f + metal::exp(-zr[HC + lane]));
    \\  }
    \\  const bool act = lane < HC * HC;
    \\  const ushort a = act ? lane / HC : 0, b = act ? lane % HC : 0;
    \\  float v = zr[2 * HC + a * HC + b];
    \\  float mx = metal::simd_shuffle(v, a * HC);
    \\  for (ushort k = 1; k < HC; ++k) mx = metal::max(mx, metal::simd_shuffle(v, a * HC + k));
    \\  v = metal::exp(v - mx);
    \\  float sm = 0.0f;
    \\  for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, a * HC + k);
    \\  v = v / sm + e;
    \\  for (int t = 0; t < 2 * ITERS - 1; ++t) {
    \\    const bool rows = t % 2 == 1;
    \\    sm = 0.0f;
    \\    for (ushort k = 0; k < HC; ++k) sm += metal::simd_shuffle(v, rows ? a * HC + k : k * HC + b);
    \\    v = v / (sm + e);
    \\  }
    \\  if (act) comb[l * HC * HC + lane] = v;
    \\}
    \\float cp[HC];
    \\for (int j = 0; j < HC; ++j) cp[j] = cpre[l * HC + j];
    \\float ss = 0.0f;
    \\for (int d = tid; d < D; d += TG) {
    \\  float h = 0.0f;
    \\  for (int j = 0; j < HC; ++j) h += cp[j] * float(x[(size_t(l) * HC + j) * D + d]);
    \\  const T hb = T(h);
    \\  h_out[size_t(l) * D + d] = hb;
    \\  ss += float(hb) * float(hb);
    \\}
    \\threadgroup float t_s[TG / 32];
    \\ss = metal::simd_sum(ss);
    \\if (lane == 0) t_s[sg] = ss;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\float tot = 0.0f;
    \\for (int g = 0; g < TG / 32; ++g) tot += t_s[g];
    \\const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
    \\for (int d = tid; d < D; d += TG) {
    \\  const size_t o = size_t(l) * D + d;
    \\  h_out[o] = w[d] * T(float(h_out[o]) * inv);
    \\}
;

/// R affine rows of one bank against `x` (K wide), across one simdgroup:
/// a lane's x chunk is prescaled by 2^-(BITS i) so a packed value is masked,
/// never shifted, and each group adds scale * sum(x q) + bias * sum(x).
/// MX: mxfp8 rows (BITS 8, GS 32: e4m3 bytes, an e8m0 exponent a group, `bi` unread).
const MOE_HEADER = transformer.GQMV_AFFINE_HEADER ++ mxfp8_qmv.HEADER ++
    \\template <int BITS, int GS, int K, int R, bool MX, typename X, typename S>
    \\inline void dsv41_dots(const device uint32_t* w, const device S* sc, const device S* bi, thread const size_t* rows,
    \\                       const device X* x, uint lane, thread float* acc) {
    \\  constexpr int VPW = BITS == 3 ? 8 : 32 / BITS;
    \\  constexpr int KW = K * BITS / 32, KG = K / GS, KP = K / VPW;
    \\  constexpr uint MASK = (1u << BITS) - 1u;
    \\  for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    \\  for (int pk = int(lane); pk < KP; pk += 32) {
    \\    const int kb = pk * VPW, gi = kb / GS;
    \\    float xp[VPW], xs = 0.0f;
    \\    for (int i = 0; i < VPW; ++i) { const float v = float(x[kb + i]); xs += v; xp[i] = MX ? v : v * (1.0f / float(1u << (BITS * i))); }
    \\    for (int r = 0; r < R; ++r) {
    \\      const uint32_t q = mlxserve_qpack<BITS>(w, rows[r] * KW, pk);
    \\      if (MX) {
    \\        // e8m0 as 2^(e - 127), times the 2^8 the e4m3 bit placement leaves out.
    \\        const uint e = uint(sc[rows[r] * KG + gi]);
    \\        const float2 lo = mlxserve_e4m3x2(q), hi = mlxserve_e4m3x2(q >> 8);
    \\        acc[r] += as_type<float>(e == 0u ? 0x400000u : (e << 23)) * 256.0f * (xp[0] * lo.x + xp[1] * hi.x + xp[2] * lo.y + xp[3] * hi.y);
    \\      } else {
    \\        float dq = 0.0f;
    \\        for (int i = 0; i < VPW; ++i) dq += xp[i] * float(q & (MASK << (BITS * i)));
    \\        acc[r] += float(sc[rows[r] * KG + gi]) * dq + float(bi[rows[r] * KG + gi]) * xs;
    \\      }
    \\    }
    \\  }
    \\  for (int r = 0; r < R; ++r) acc[r] = metal::simd_sum(acc[r]);
    \\}
;

/// One token's routed gate+up (the fused bank: expert e's up rows sit N rows
/// after its gate rows) and, as slot TOPK, the shared expert's; RPS rows a
/// simdgroup. Each accumulator rounds to T where `gather_qmm` writes it, then
/// `clampedSwiglu` times the slot's routing weight (1 for the shared) in f32.
const MOE_GU_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint n0 = thread_position_in_grid.y * RPS, e = thread_position_in_grid.z;
    \\float a[2 * RPS];
    \\size_t rows[2 * RPS];
    \\const size_t eb = e < TOPK ? size_t(inds[e]) * 2 * N + n0 : n0;
    \\for (int r = 0; r < RPS; ++r) { rows[r] = eb + r; rows[RPS + r] = eb + N + r; }
    \\if (e < TOPK) dsv41_dots<BITS, GS, K, 2 * RPS, false>(w, scales, biases, rows, x, lane, a);
    \\else dsv41_dots<SBITS, SGS, K, 2 * RPS, SMX>(sw, sscales, sbiases, rows, x, lane, a);
    \\const float wt = e < TOPK ? wts[e] : 1.0f;
    \\if (lane == 0) for (int r = 0; r < RPS; ++r) {
    \\  float g = float(T(a[r])), u = float(T(a[RPS + r]));
    \\  if (lim[0] > 0.0f) { g = metal::min(g, lim[0]); u = metal::clamp(u, -lim[0], lim[0]); }
    \\  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
    \\  out[size_t(e) * N + n0 + r] = T((g * (g < 0.0f ? y : 1.0f - y)) * u * wt);
    \\}
;

/// One token's routed down projections summed in f32 in slot order, plus the
/// shared expert's (`withShared`), then the hc expand (`hcExpand`) into the
/// new stream; RPS rows a simdgroup. Every rounding where the composed path
/// takes it.
const MOE_DN_SOURCE =
    \\const uint lane = thread_index_in_simdgroup;
    \\const uint n0 = thread_position_in_grid.y * RPS;
    \\float tot[RPS], a[RPS];
    \\size_t rows[RPS];
    \\for (int r = 0; r < RPS; ++r) tot[r] = 0.0f;
    \\for (int e = 0; e < TOPK; ++e) {
    \\  for (int r = 0; r < RPS; ++r) rows[r] = size_t(inds[e]) * N + n0 + r;
    \\  dsv41_dots<BITS, GS, K, RPS, false>(w, scales, biases, rows, x + size_t(e) * K, lane, a);
    \\  for (int r = 0; r < RPS; ++r) tot[r] += float(T(a[r]));
    \\}
    \\for (int r = 0; r < RPS; ++r) rows[r] = n0 + r;
    \\dsv41_dots<SBITS, SGS, K, RPS, SMX>(sw, sscales, sbiases, rows, x + size_t(TOPK) * K, lane, a);
    \\if (lane == 0) for (int r = 0; r < RPS; ++r) {
    \\  const uint d = n0 + r;
    \\  const float y = float(T(tot[r] + float(T(a[r]))));
    \\  for (int i = 0; i < HC; ++i) {
    \\    float acc = 0.0f;
    \\    for (int j = 0; j < HC; ++j) acc = metal::fma(comb[j * HC + i], float(res[j * N + d]), acc);
    \\    out[i * N + d] = T(metal::fma(post[i], y, acc));
    \\  }
    \\}
;

/// A prefill MoE call's SwiGLU, one element a thread: row `r` of the fused
/// gate+up gather `y` (`[n, 2N]`, gate then up) times its routing weight, the
/// f32 math of `clampedSwiglu` in its order, rounded once to T.
const MOE_ACT_SOURCE =
    \\const uint i = thread_position_in_grid.x;
    \\const uint row = i / N, c = i % N;
    \\if (row >= uint(y_shape[0])) return;
    \\const device T* yr = y + size_t(row) * 2 * N;
    \\float g = float(yr[c]), u = float(yr[N + c]);
    \\if (lim[0] > 0.0f) { g = metal::min(g, lim[0]); u = metal::clamp(u, -lim[0], lim[0]); }
    \\const float s = 1.0f / (1.0f + metal::precise::exp(metal::abs(g)));
    \\out[i] = T(g * (g < 0.0f ? s : 1.0f - s) * u * wts[row]);
;

/// A prefill MoE call's combine, one element a thread: the row's K sorted down
/// rows (`inv` names them) summed in f32 plus the shared expert, rounded once to
/// T; what unsort, the f32 sum over K and `addShared` did in five passes.
const MOE_COMBINE_SOURCE =
    \\const uint i = thread_position_in_grid.x;
    \\const uint row = i / D, c = i % D;
    \\if (row >= uint(sh_shape[0])) return;
    \\float acc = 0.0f;
    \\for (int j = 0; j < K; ++j) acc += float(dn[size_t(inv[row * K + j]) * D + c]);
    \\out[i] = T(acc + float(sh[i]));
;

const MOE_ACT_INS = [_][*:0]const u8{ "y", "wts", "lim" };
const MOE_COMBINE_INS = [_][*:0]const u8{ "dn", "inv", "sh" };

/// One output element a thread: rows of ROW elements, the last RD of each
/// D-wide vector roped (MLX `fast::rope`'s traditional pairs and math), then
/// QK's activation round trip over blocks inside one simdgroup (`Qat`:
/// 1 fp8 / 32 / ue8m0, 2 fp4 / 32 / ue8m0, 3 fp4 / 16 / e4m3), rounded to
/// T, written as OT.
const ROPE_TAIL_SOURCE =
    \\const uint i = thread_position_in_grid.x;
    \\const int D = x_shape[x_ndim - 1];
    \\const int c = i % D, row = i / ROW;
    \\const int head = D - RD;
    \\float r = float(x[i]);
    \\if (c >= head) {
    \\  const int p = (c - head) / 2;
    \\  const size_t b = i - (c - head) % 2;
    \\  const float x1 = float(x[b]), x2 = float(x[b + 1]);
    \\  const float theta = scal[0] * float(row + off) * (1.0f / periods[p]);
    \\  const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
    \\  r = float(T((c - head) % 2 == 0 ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
    \\}
    \\if (QK != 0) {
    \\  float amax = metal::abs(r);
    \\  for (ushort m = (QK == 3 ? 8 : 16); m > 0; m >>= 1) amax = metal::max(amax, metal::simd_shuffle_xor(amax, m));
    \\  float scale;
    \\  if (QK == 1) scale = dsv41_pow2ceil(metal::max(amax, 1e-4f) * (1.0f / 448.0f));
    \\  else if (QK == 2) scale = dsv41_pow2ceil(metal::max(amax, 6.0f * 1.17549435e-38f) * (1.0f / 6.0f));
    \\  else scale = dsv41_fp8_roundtrip(metal::max(amax, 6.0f * 0.001953125f) / 6.0f);
    \\  r = QK == 1 ? dsv41_fp8_roundtrip(metal::clamp(r / scale, -448.0f, 448.0f)) * scale
    \\             : dsv41_e2m1(metal::clamp(r / scale, -6.0f, 6.0f)) * scale;
    \\}
    \\out[i] = OT(T(r));
;

/// e4m3 encode/decode exactly as MLX's `fp8_e4m3` (which takes it from
/// PyTorch's `Float8_e4m3fn`): saturating, round to nearest even.
const FP8_HEADER =
    \\inline float dsv41_fp8_roundtrip(float f) {
    \\  uint fb = as_type<uint>(f);
    \\  const uint sign = fb & 0x80000000u;
    \\  fb ^= sign;
    \\  uchar bits;
    \\  if (fb >= (543u << 21)) bits = 0x7E;
    \\  else if (fb < (121u << 23)) {
    \\    const uint dm = 141u << 23;
    \\    bits = uchar(as_type<uint>(as_type<float>(fb) + as_type<float>(dm)) - dm);
    \\  } else {
    \\    const uint odd = (fb >> 20) & 1u;
    \\    fb += (uint(7 - 127) << 23) + 0x7FFFFu;
    \\    fb += odd;
    \\    bits = uchar(fb >> 20);
    \\  }
    \\  bits |= uchar(sign >> 24);
    \\  const ushort v = bits & 127;
    \\  const ushort u = (v << 7) | (((v + 1) >> 7) << 14) | (ushort((bits >> 7) & 1) << 15);
    \\  return float(as_type<half>(u)) * 256.0f;
    \\}
    \\inline float dsv41_pow2ceil(float a) {
    \\  const uint ab = as_type<uint>(a);
    \\  return as_type<float>((((ab >> 23) & 0xFFu) + ((ab & 0x7FFFFFu) != 0u ? 1u : 0u)) << 23);
    \\}
    \\inline float dsv41_e2m1(float v) {
    \\  const float m = metal::abs(v);
    \\  const int k = int(m > 0.25f) + int(m > 1.25f) + int(m > 2.5f) + int(m > 5.0f) + int(m >= 0.75f) + int(m >= 1.75f) + int(m >= 3.5f);
    \\  const float lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    \\  return metal::sign(v) * lut[k];
    \\}
;

/// `kvRow` after its projection, one threadgroup (D threads) a row: MLX's
/// rms_norm, the tail rope, then (QAT) the fp8 / 32 / ue8m0 round trip.
const KV_ROW_SOURCE =
    \\const int l = threadgroup_position_in_grid.y, d = thread_position_in_threadgroup.x;
    \\const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int D = x_shape[1];
    \\const size_t o = size_t(l) * D + d;
    \\const float xv = float(x[o]);
    \\threadgroup float t_s[32];
    \\threadgroup float t_y[1024];
    \\const float ss = metal::simd_sum(xv * xv);
    \\if (lane == 0) t_s[sg] = ss;
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\float tot = 0.0f;
    \\for (int g = 0; g < D / 32; ++g) tot += t_s[g];
    \\const float inv = metal::precise::rsqrt(tot / float(D) + eps[0]);
    \\const T y = w[d] * T(xv * inv);
    \\t_y[d] = float(y);
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\float r = float(y);
    \\const int head = D - RD;
    \\if (d >= head) {
    \\  const int p = (d - head) / 2, b = d - (d - head) % 2;
    \\  const float x1 = t_y[b], x2 = t_y[b + 1];
    \\  const float theta = float(l + off) * (1.0f / periods[p]);
    \\  const float cs = metal::fast::cos(theta), sn = metal::fast::sin(theta);
    \\  r = float(T((d - head) % 2 == 0 ? x1 * cs - x2 * sn : x1 * sn + x2 * cs));
    \\}
    \\if (QAT) {
    \\  const float scale = dsv41_pow2ceil(metal::max(metal::simd_max(metal::abs(r)), 1e-4f) * (1.0f / 448.0f));
    \\  r = dsv41_fp8_roundtrip(metal::clamp(r / scale, -448.0f, 448.0f)) * scale;
    \\}
    \\out[o] = T(r);
;

fn buildKernel(name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, src: [*:0]const u8, header: [*:0]const u8) !mlx.mlx_fast_metal_kernel {
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

const IDX_INS = [_][*:0]const u8{ "q", "keys", "w", "meta", "wsc" };

fn buildIdxScoreKernel() !mlx.mlx_fast_metal_kernel {
    return buildKernel("dsv41_index_scores", &IDX_INS, &.{"scores"}, IDX_SCORE_SOURCE, "");
}

const ATTN_INS = [_][*:0]const u8{ "q", "ring", "comp", "topk", "sink", "meta", "scal" };
const ATTN_SOFTMAX_INS = [_][*:0]const u8{ "lw", "lc", "topk", "sink", "meta", "scal" };
const ATTN_DEC_INS = ATTN_INS ++ [_][*:0]const u8{"periods"};
const ATTN_MERGE_INS = [_][*:0]const u8{ "pacc", "pm", "pl", "sink", "meta", "periods" };

/// A two-Mac pair's share of the trunk: the worker runs layers [lo, hi), the host every other one.
/// A cut never splits what layers share (`pairCutOk`), and the worker holds no Engram layer.
pub const Pair = struct {
    lo: usize,
    hi: usize,
    worker: bool,

    pub fn holds(p: Pair, li: usize) bool {
        return (li >= p.lo and li < p.hi) == p.worker;
    }
};

/// Host: hands one step's stream `[L, hc, dim]` and mix `[L, hc]` to the worker at `Pair.lo` and
/// returns both as they stand after `Pair.hi` (caller frees); `snap`: the host snapshots when
/// the step ends, so the worker does too; `verify`: a DSpark verify chunk, so the worker arms
/// the same rollback anchors. Set by distributed.zig.
pub var pair_exchange: ?*const fn (x: A, pre: A, p0: usize, snap: bool, verify: bool) anyerror![2]A = null;

/// The highest set bit of `mask` at or below `l`, or -1.
fn latestBit(mask: u64, l: usize) i64 {
    var j: i64 = @intCast(l);
    while (j >= 0) : (j -= 1) if ((mask >> @intCast(j)) & 1 != 0) return j;
    return -1;
}

/// Whether a pair may cut the trunk before layer `x`: no compressed layer at or past it reads
/// compressed rows, index picks or candidate blocks made below it.
pub fn pairCutOk(c: *const ModelConfig, x: usize) bool {
    const n = c.num_hidden_layers;
    if (x == 0 or x == n) return true;
    if (x > n) return false;
    const cand = c.dsv41_candidate_source;
    for (x..n) |l| {
        if (c.dsv4_compress_ratios[l] == 0) continue;
        if (latestBit(c.dsv41_kv_sources, l) < x or latestBit(c.dsv41_index_sources, l) < x) return false;
        if (cand >= 0 and x > cand and l > cand and (c.dsv41_index_sources >> @intCast(l)) & 1 != 0) return false;
    }
    return true;
}

fn engramIn(c: *const ModelConfig, lo: usize, hi: usize) bool {
    for (c.dsv41_engram_layers[0..c.dsv41_n_engram_layers]) |l| if (l >= lo and l < hi) return true;
    return false;
}

/// The worker's layers: the longest valid run holding no Engram layer that stays within its
/// share of the two Macs' free memory (`mine`, `theirs`, any unit), else the shortest one.
pub fn pairRange(c: *const ModelConfig, mine: u64, theirs: u64) ?[2]usize {
    const n = c.num_hidden_layers;
    const share = @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(theirs)) / @as(f64, @floatFromInt(@max(mine + theirs, 1)));
    var best: ?[2]usize = null;
    for (0..n) |a| {
        if (!pairCutOk(c, a)) continue;
        for (a + 1..n + 1) |b| {
            if (!pairCutOk(c, b) or engramIn(c, a, b)) continue;
            const len: f64 = @floatFromInt(b - a);
            const cur = best orelse {
                best = .{ a, b };
                continue;
            };
            const cur_len: f64 = @floatFromInt(cur[1] - cur[0]);
            const fits = len <= share;
            const cur_fits = cur_len <= share;
            if ((fits and (!cur_fits or len > cur_len)) or (!fits and !cur_fits and len < cur_len)) best = .{ a, b };
        }
    }
    return best;
}

pub const InitOpts = struct {
    /// The activation dtype; tests run f32 against the f32 reference.
    act: mlx.mlx_dtype = .bfloat16,
    /// Load the DSpark stages when the checkpoint ships them.
    dspark: bool = false,
    /// This process's share of a two-Mac pair.
    pair: ?Pair = null,
    /// A dense head served 8-bit (affine, g64): half the bytes every token reads. Lossy.
    head_8bit: bool = false,
};

pub fn init(gpa: std.mem.Allocator, cfg: *const ModelConfig, w: *model.Weights, s: S, opts: InitOpts) !*Dsv41Model {
    const m = try gpa.create(Dsv41Model);
    errdefer gpa.destroy(m);
    const n_layers = cfg.num_hidden_layers;
    const n_stages: usize = if (opts.dspark and cfg.dsv4_dspark_block_size > 0 and getOpt(w, "mtp.0.main_proj.weight", .{}) != null)
        cfg.dsv4_n_compress_ratios - n_layers
    else
        0;
    m.* = .{
        .gpa = gpa,
        .s = s,
        .cfg = cfg.*,
        .act = opts.act,
        .embed = undefined,
        .head = undefined,
        .norm = undefined,
        .layers = try gpa.alloc(Layer, n_layers + n_stages),
        .n_layers = n_layers,
        .vocab = cfg.vocab_size,
        .rope_plain = undefined,
        .rope_yarn = undefined,
        .attn_dec_kernel = try buildKernel("dsv41_attn_decode", &ATTN_DEC_INS, &.{"out"}, ATTN_DEC_SOURCE, UNROPE_HEADER),
        .attn_split_kernel = try buildKernel("dsv41_attn_split", &ATTN_INS, &.{ "pacc", "pm", "pl" }, ATTN_SPLIT_SOURCE, ""),
        .attn_merge_kernel = try buildKernel("dsv41_attn_merge", &ATTN_MERGE_INS, &.{"out"}, ATTN_MERGE_SOURCE, UNROPE_HEADER),
        .attn_softmax_kernel = try buildKernel("dsv41_attn_softmax", &ATTN_SOFTMAX_INS, &.{ "pw", "pc" }, ATTN_SOFTMAX_SOURCE, ""),
        .attn_f16 = if (std.c.getenv("MLX_SERVE_DSV41_ATTN_F16")) |v| v[0] != '0' else true,
        .idx_score_kernel = try buildIdxScoreKernel(),
        .idx_tile_kernel = try buildKernel("dsv41_index_tile", &IDX_INS, &.{"scores"}, IDX_TILE_SOURCE, ""),
        .gate_kernel = try buildKernel("dsv41_gate", &.{ "x", "w" }, &.{"logits"}, GATE_SOURCE, ""),
        .topk_kernel = try buildKernel("dsv41_topk", &.{"x"}, &.{"out"}, TOPK_SOURCE, TOPK_HEADER),
        .dense_f32x_kernel = try buildKernel("dsv41_dense_f32x", &.{ "x", "w" }, &.{"out"}, DENSE_F32X_SOURCE, ""),
        .route_kernel = try buildKernel("dsv41_route", &.{ "logits", "bias", "scal" }, &.{ "idx", "wts" }, ROUTE_SOURCE, ""),
        .hc_mix_kernel = try buildKernel("dsv41_hc_mix", &.{ "x", "fn", "scale", "base", "eps" }, &.{"z"}, HC_MIX_SOURCE, ""),
        .hc_norm_kernel = try buildKernel("dsv41_hc_norm", &.{ "z", "x", "cpre", "w", "eps" }, &.{ "pre", "post", "comb", "h_out" }, HC_NORM_SOURCE, ""),
        .hc_stage_kernel = try buildKernel("dsv41_hc_stage", &HC_STAGE_INS, &HC_STAGE_OUTS, HC_STAGE_SOURCE, ""),
        .hc_stage_x_kernel = try buildKernel("dsv41_hc_stage_x", &HC_STAGE_X_INS, &HC_STAGE_X_OUTS, HC_STAGE_X_SOURCE, ""),
        .hc_split_kernel = try buildKernel("dsv41_hc_split", &.{ "z", "eps" }, &.{ "pre", "post", "comb" }, HC_SPLIT_SOURCE, ""),
        .rope_kernel = try buildKernel("dsv41_rope_tail", &.{ "x", "periods", "off", "scal" }, &.{"out"}, ROPE_TAIL_SOURCE, FP8_HEADER),
        .kv_row_kernel = try buildKernel("dsv41_kv_row", &.{ "x", "w", "periods", "off", "eps" }, &.{"out"}, KV_ROW_SOURCE, FP8_HEADER),
        .engram_kernel = try buildKernel("dsv41_engram", &ENGRAM_INS, &.{"out"}, ENGRAM_SOURCE, ""),
        .moe_gu_kernel = try buildKernel("dsv41_moe_gu", &MOE_GU_INS, &.{"out"}, MOE_GU_SOURCE, MOE_HEADER),
        .moe_dn_kernel = try buildKernel("dsv41_moe_dn", &MOE_DN_INS, &.{"out"}, MOE_DN_SOURCE, MOE_HEADER),
        .moe_act_kernel = try buildKernel("dsv41_moe_act", &MOE_ACT_INS, &.{"out"}, MOE_ACT_SOURCE, ""),
        .moe_combine_kernel = try buildKernel("dsv41_moe_combine", &MOE_COMBINE_INS, &.{"out"}, MOE_COMBINE_SOURCE, ""),
    };
    m.cfg.pack_dir = null; // borrowed; the caller's config owns it
    m.pair = opts.pair;
    const worker = if (opts.pair) |p| p.worker else false;
    if (opts.pair) |p| if (p.worker and engramIn(cfg, p.lo, p.hi)) return error.DsvPairEngram;
    errdefer {
        freeRegions(m);
        for (m.engrams) |*e| e.table.close();
        gpa.free(m.engrams);
        gpa.free(m.token_map);
        for (m.owned.items) |x| _ = mlx.mlx_array_free(x);
        m.owned.deinit(gpa);
        gpa.free(m.layers);
        _ = mlx.mlx_fast_metal_kernel_free(m.attn_dec_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.attn_split_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.attn_merge_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.attn_softmax_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.idx_score_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.idx_tile_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.gate_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.route_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.hc_mix_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.topk_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.dense_f32x_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.hc_norm_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.hc_stage_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.hc_stage_x_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.hc_split_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.rope_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.kv_row_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.engram_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.moe_gu_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.moe_dn_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.moe_act_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(m.moe_combine_kernel);
    }
    if (worker) {
        m.embed = .{ .w = none };
        m.head = .{ .w = none };
        m.norm = none;
    } else {
        m.embed = try getLin(w, cfg.hidden_size, "embed", .{});
        m.head = try getLin(w, cfg.hidden_size, "head", .{});
        if (m.head.s.ctx == null) {
            if (opts.head_8bit) m.head = try quantizeLin(m, w, m.head, 8, 64);
            log.info("deepseek_v41: lm head {s}\n", .{if (opts.head_8bit) "requantized to 8-bit (MLX_SERVE_DSV41_HEAD8=0 keeps it dense)" else "dense, as shipped"});
        }
        m.norm = try ownCast(m, try getW(w, "norm.weight", .{}), m.act);
    }
    m.rope_plain = try buildRope(m, cfg.rope_theta, 0, 1, 0, 0);
    m.rope_yarn = try buildRope(m, cfg.dsv4_compress_rope_theta, if (cfg.rope_yarn) cfg.yarn_orig_max_pos else 0, cfg.yarn_factor, cfg.yarn_beta_fast, cfg.yarn_beta_slow);
    for (0..n_layers) |li| m.layers[li] = if (m.held(li)) try loadLayer(m, w, "layers", li, li) else std.mem.zeroes(Layer);
    m.ced_split = cedSplit(cfg);
    try probeHcThreads(gpa, m);
    try compileRegions(m);
    if (n_stages > 0) {
        for (0..n_stages) |si| m.layers[n_layers + si] = try loadLayer(m, w, "mtp", si, null);
        const last = n_stages - 1;
        const targets = cfg.dsv4_n_dspark_target_layers;
        m.dspark = .{
            .main_proj = try getLin(w, cfg.hidden_size * targets, "mtp.0.main_proj", .{}),
            .main_norm = try ownCast(m, try getW(w, "mtp.0.main_norm.weight", .{}), m.act),
            .norm = try ownCast(m, try getW(w, "mtp.{d}.norm.weight", .{last}), m.act),
            .markov_embed = try getW(w, "mtp.{d}.markov_head.embed.weight", .{last}),
            .markov_head = try ownCast(m, try getW(w, "mtp.{d}.markov_head.head.weight", .{last}), .float32),
            .conf = try ownCast(m, try getW(w, "mtp.{d}.confidence_head.proj.weight", .{last}), .float32),
        };
        m.n_mtp = n_stages;
        m.ds_block = cfg.dsv4_dspark_block_size;
        m.ds_conf_thr = dsv4.dsparkConfThreshold();
    }
    if (!worker) if (cfg.pack_dir) |dir| try initEngram(m, w, dir);
    {
        const k: usize = cfg.num_experts_per_tok;
        var ks: [64]u32 = undefined;
        for (ks[0..k], 0..) |*v, i| v.* = @intCast(i);
        m.lhs_dn = try m.own(mlx.mlx_array_new_data(&ks, &[_]c_int{ 1, ci(k) }, 2, .uint32));
        @memset(ks[0..k], 0);
        m.lhs_gu = try m.own(mlx.mlx_array_new_data(&ks, &[_]c_int{ 1, ci(k) }, 2, .uint32));
    }
    try evalWeights(m);
    log.info("deepseek_v41: {d} layers, engram {d} layers, DSpark {d} stages\n", .{ n_layers, m.engrams.len, n_stages });
    return m;
}

/// Every array the model reads, evaluated one layer at a time. A compiled region captures its layer's weights,
/// and a lazy load in its trace would read the weight from disk into a fresh buffer on every call.
fn evalWeights(m: *Dsv41Model) !void {
    var xs: std.ArrayList(A) = .empty;
    defer xs.deinit(m.gpa);
    for (m.layers) |*ly| {
        xs.clearRetainingCapacity();
        try collectArrays(m.gpa, &xs, ly.*);
        try evalAll(xs.items);
        try wireLoaded(m);
    }
    xs.clearRetainingCapacity();
    try collectArrays(m.gpa, &xs, .{ m.embed, m.head, m.norm, m.dspark });
    for (m.engrams) |e| try collectArrays(m.gpa, &xs, .{ e.wkv, e.qk });
    try evalAll(xs.items);
}

/// Wires what has loaded so far: the residency policy takes the new buffers into the set (the scheduler applies
/// it only after init), and a GPU command buffer makes the set resident. Until then the weights are ordinary
/// pages, and a pack near the RAM size had them compressed until the kernel killed the process.
fn wireLoaded(m: *const Dsv41Model) !void {
    _ = mlx.applyWiredPolicy();
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_add(&o, one, one, m.s));
    try mlx.check(mlx.mlx_array_eval(o));
}

/// The arrays inside `v` (structs, optionals and arrays of them).
fn collectArrays(gpa: std.mem.Allocator, xs: *std.ArrayList(A), v: anytype) !void {
    const T = @TypeOf(v);
    if (T == A) {
        if (v.ctx != null) try xs.append(gpa, v);
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names) |name| try collectArrays(gpa, xs, @field(v, name)),
        .optional => if (v) |x| try collectArrays(gpa, xs, x),
        .array => for (v) |x| try collectArrays(gpa, xs, x),
        else => {},
    }
}

/// The prompt pass's working set over `floor`: a chunk runs one layer at a time over spans; the whole
/// chunk's stream (in and out), the picks layers hand down and the DSpark rows
/// stay live across layers, and one span at a time adds its gathered
/// compressed rows, attention scores and an indexer score block, one MoE call its transients.
/// State is the compressed caches per token plus the windows per layer.
pub fn prefillBytes(config: *const ModelConfig, seq: u64, chunk: u64, floor: u64) u64 {
    const c: u64 = @min(chunk, @max(seq, 1));
    const span: u64 = @min(SPAN, c);
    const d: u64 = config.hidden_size;
    const hd: u64 = config.head_dim;
    const topk: u64 = config.dsv4_index_topk;
    const window: u64 = config.sliding_window;
    const cand_block: u64 = @max(config.dsv41_candidate_block_size, 1);
    const state = seq * config.kvBytesPerToken() + config.num_hidden_layers * (window + 16) * hd * 2;
    const held = c * (2 * config.dsv4_hc_mult * d * 2 + topk * 4 + seq / cand_block + config.dsv4_n_dspark_target_layers * d * 2);
    // The chunk's query heads and attention heads (roped, gathered by `attnOut`) live for the layer.
    const proj = c * config.num_attention_heads * hd * 2 * 4;
    const attn = proj + span * topk * hd * 4 * 2 + span * config.num_attention_heads * ((window + topk) * 4 * 2 + hd * 4 * 2);
    const idx_bytes = span * seq * 4 * 3;
    const moe_bytes = @min(c, MOE_ROWS) * config.num_experts_per_tok * (2 * config.moe_intermediate_size + d) * 4;
    return (state + held + attn + idx_bytes + moe_bytes + floor) * 5 / 4;
}

/// The DSpark stages load when the pack ships them, unless `--no-mtp` or the model's `mtp` setting turn them off
/// (`applyModelSettings` folds both into `mtp_override`); V4's `MLX_SERVE_DSV4_DSPARK` turns them on either way.
pub fn dsparkWanted(cfg: *const ModelConfig) bool {
    // On a pair DSpark drafts on the host, from the target layers' rows: they must be its own.
    if (cfg.dist_dsv41) |p| {
        if (p.worker) return false;
        for (cfg.dsv4_dspark_target_layers[0..cfg.dsv4_n_dspark_target_layers]) |t| if (t >= p.lo and t < p.hi) return false;
    }
    const env = std.c.getenv("MLX_SERVE_DSV4_DSPARK");
    return (cfg.mtp_override orelse true) or (env != null and env.?[0] != '0');
}

/// What a load holds of the pack: its safetensors bytes less the Engram tables it preads and the tensors it
/// drops (the vision tower on a text load; the DSpark stages when they do not load; a tensor's copy in a shard the
/// index does not name for it).
pub fn residentDiskBytes(gpa: std.mem.Allocator, dir: []const u8, cfg: *const ModelConfig, shards: u64) !u64 {
    var total = shards -| try droppedBytes(gpa, dir, dsparkWanted(cfg), cfg.has_vision);
    for (cfg.dsv41_engram_layers[0..cfg.dsv41_n_engram_layers], 0..) |l, k| {
        var t = try engram.Table.open(gpa, dir, k, l, cfg.dsv41_engram_rows[k], cfg.dsv41_engram_head_dim);
        defer t.close();
        total -|= t.shardBytes();
    }
    return total;
}

fn droppedBytes(gpa: std.mem.Allocator, dir_path: []const u8, dspark: bool, vision: bool) !u64 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 << 20)) catch return 0;
    const idx = std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch return 0;
    const wm = (if (idx == .object) idx.object.get("weight_map") else null) orelse return 0;
    if (wm != .object) return 0;
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (wm.object.values()) |v| if (v == .string) try files.put(a, v.string, {});
    var total: u64 = 0;
    for (files.keys()) |name| {
        const f = dir.openFile(io, name, .{}) catch continue;
        defer f.close(io);
        var rb: [8192]u8 = undefined;
        var rs = f.reader(io, &rb);
        const len = rs.interface.takeInt(u64, .little) catch continue;
        if (len > 64 << 20) continue;
        const hdr = try a.alloc(u8, @intCast(len));
        rs.interface.readSliceAll(hdr) catch continue;
        const h = std.json.parseFromSliceLeaky(std.json.Value, a, hdr, .{}) catch continue;
        if (h != .object) continue;
        var it = h.object.iterator();
        while (it.next()) |e| {
            const key = e.key_ptr.*;
            const bare = if (std.mem.startsWith(u8, key, model.dsv41_text_prefix)) key[model.dsv41_text_prefix.len..] else key;
            const owner = wm.object.get(key);
            const foreign = owner != null and owner.? == .string and !std.mem.eql(u8, owner.?.string, name);
            if (!foreign and (vision or !model.dsv41VisionKey(bare)) and (dspark or !std.mem.startsWith(u8, bare, "mtp."))) continue;
            const off = (if (e.value_ptr.* == .object) e.value_ptr.object.get("data_offsets") else null) orelse continue;
            if (off != .array or off.array.items.len != 2 or off.array.items[0] != .integer or off.array.items[1] != .integer) continue;
            total += @intCast(@max(off.array.items[1].integer - off.array.items[0].integer, 0));
        }
    }
    return total;
}

/// Engram for the pack in `dir`: the token map, the hash, and each layer's
/// table and projections. Without Engram layers in the config this is a no-op.
fn initEngram(m: *Dsv41Model, w: *const model.Weights, dir: []const u8) !void {
    const c = &m.cfg;
    if (c.dsv41_n_engram_layers == 0) return;
    m.token_map = try engram.loadTokenMap(m.gpa, dir, c.vocab_size, c.dsv41_engram_compressed_vocab);
    if (c.dsv41_engram_pad_id >= m.token_map.len) return error.EngramTokenMap;
    m.hash = try engram.Hash.init(c, m.token_map[c.dsv41_engram_pad_id]);
    const n = c.dsv41_n_engram_layers;
    m.engrams = try m.gpa.alloc(EngramLayer, n);
    var opened: usize = 0;
    errdefer {
        for (m.engrams[0..opened]) |*e| e.table.close();
        m.gpa.free(m.engrams);
        m.engrams = &.{};
    }
    const cols = m.hash.?.cols();
    for (0..n) |k| {
        const l: usize = c.dsv41_engram_layers[k];
        const qw = try getW(w, "layers.{d}.engram.q_weight", .{l});
        const kw = try getW(w, "layers.{d}.engram.k_weight", .{l});
        var prod = mlx.mlx_array_new();
        {
            var q32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q32);
            var k32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k32);
            try mlx.check(mlx.mlx_astype(&q32, qw, .float32, m.s));
            try mlx.check(mlx.mlx_astype(&k32, kw, .float32, m.s));
            try mlx.check(mlx.mlx_multiply(&prod, q32, k32, m.s));
        }
        _ = try m.own(prod);
        try mlx.check(mlx.mlx_array_eval(prod));
        m.engrams[k] = .{
            .layer = l,
            .wkv = try getLin(w, cols * c.dsv41_engram_head_dim, "layers.{d}.engram.wkv", .{l}),
            .qk = prod,
            .table = try engram.Table.open(m.gpa, dir, k, l, c.dsv41_engram_rows[k], c.dsv41_engram_head_dim),
        };
        opened += 1;
    }
}

// ── per-request state ────────────────────────────────────────────────────

/// Append-only rows on the GPU, growing by a quarter; a rollback is offset-only.
const Rows = struct {
    buf: A = none,
    used: usize = 0,
    cap: usize = 0,
    width: usize,
    dtype: mlx.mlx_dtype,

    fn deinit(self: *Rows) void {
        if (self.buf.ctx != null) _ = mlx.mlx_array_free(self.buf);
    }

    fn append(self: *Rows, x: A, s: S) !void {
        const n = dim(x, 0);
        if (n == 0) return;
        if (self.used + n > self.cap) {
            const cap = @max(self.used + n, @max(64, self.cap + self.cap / 4));
            var z = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ ci(cap), ci(self.width) }, 2, self.dtype, s));
            if (self.used > 0) {
                const old = try self.view(self.used, s);
                defer _ = mlx.mlx_array_free(old);
                const zeros = z;
                defer _ = mlx.mlx_array_free(zeros);
                var up = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice_update(&up, zeros, old, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(self.used), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
                z = up;
            }
            if (self.buf.ctx != null) _ = mlx.mlx_array_free(self.buf);
            self.buf = z;
            self.cap = cap;
        }
        var xc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xc);
        try mlx.check(mlx.mlx_astype(&xc, x, self.dtype, s));
        var up = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice_update(&up, self.buf, xc, &[_]c_int{ ci(self.used), 0 }, 2, &[_]c_int{ ci(self.used + n), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
        _ = mlx.mlx_array_free(self.buf);
        self.buf = up;
        self.used += n;
    }

    /// Rows [0, n) (caller frees).
    fn view(self: *const Rows, n: usize, s: S) !A {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&o, self.buf, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
        return o;
    }
};

/// The window's rows by absolute position in a ring of window + slack rows.
const Ring = struct {
    buf: A,
    cap: usize,
    width: usize,

    fn init(cap: usize, width: usize, dt: mlx.mlx_dtype, s: S) !Ring {
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ ci(cap), ci(width) }, 2, dt, s));
        return .{ .buf = z, .cap = cap, .width = width };
    }

    fn deinit(self: *Ring) void {
        _ = mlx.mlx_array_free(self.buf);
    }

    /// Rows for positions [p0, p0 + n); only the last `cap` can survive.
    fn write(self: *Ring, x: A, p0: usize, s: S) !void {
        const n = dim(x, 0);
        const keep_n = @min(n, self.cap);
        const first = p0 + n - keep_n;
        var done: usize = 0;
        while (done < keep_n) {
            const slot = (first + done) % self.cap;
            const run = @min(keep_n - done, self.cap - slot);
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ ci(n - keep_n + done), 0 }, 2, &[_]c_int{ ci(n - keep_n + done + run), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var cast = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cast);
            try mlx.check(mlx.mlx_astype(&cast, part, mlx.mlx_array_dtype(self.buf), s));
            var up = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice_update(&up, self.buf, cast, &[_]c_int{ ci(slot), 0 }, 2, &[_]c_int{ ci(slot + run), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            _ = mlx.mlx_array_free(self.buf);
            self.buf = up;
            done += run;
        }
    }

    /// Rows for positions [lo, hi) in order; `hi - lo <= cap`.
    fn read(self: *const Ring, sc: *Scope, lo: usize, hi: usize) !A {
        std.debug.assert(hi - lo <= self.cap);
        var idx: [512]i32 = undefined;
        for (lo..hi, 0..) |p, k| idx[k] = @intCast(p % self.cap);
        return sc.take(self.buf, try sc.ints(idx[0 .. hi - lo], &.{ci(hi - lo)}), 0);
    }
};

const LState = struct {
    win: Ring,
    comp: ?Rows = null,
    idx_k: ?Rows = null,
    /// Ratio > 1 compressors: the open group's kv and gate rows, f32.
    pend_kv: A = none,
    pend_gate: A = none,

    fn deinit(self: *LState) void {
        self.win.deinit();
        if (self.comp) |*r| r.deinit();
        if (self.idx_k) |*r| r.deinit();
        if (self.pend_kv.ctx != null) _ = mlx.mlx_array_free(self.pend_kv);
        if (self.pend_gate.ctx != null) _ = mlx.mlx_array_free(self.pend_gate);
    }
};

/// What a verify chunk needs to keep only a prefix: each pooling
/// compressor's rows from its open group's first token on, and the chunk's
/// DSpark main-hidden rows (committed only once the accepted count is known).
const Verify = struct {
    n0: usize,
    comp_used: []usize,
    idx_used: []usize,
    pend_kv: []A,
    pend_gate: []A,
    mh: A = none,
};

pub const State = struct {
    gpa: std.mem.Allocator,
    n: usize = 0,
    layers: []LState,
    /// The request's compressed token ids (Engram look-back).
    hist: std.ArrayList(u32) = .empty,
    /// Per DSpark stage: the main-stream KV rows of committed positions.
    ds: []Ring = &.{},
    /// The last committed position's main hidden `[1, targets * dim]`.
    mh: A = none,
    verify: ?Verify = null,

    pub fn deinit(self: *State) void {
        for (self.layers) |*l| l.deinit();
        self.gpa.free(self.layers);
        for (self.ds) |*r| r.deinit();
        self.gpa.free(self.ds);
        self.hist.deinit(self.gpa);
        if (self.mh.ctx != null) _ = mlx.mlx_array_free(self.mh);
        self.dropVerify();
    }

    fn dropVerify(self: *State) void {
        const v = self.verify orelse return;
        for (v.pend_kv) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (v.pend_gate) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        if (v.mh.ctx != null) _ = mlx.mlx_array_free(v.mh);
        self.gpa.free(v.comp_used);
        self.gpa.free(v.idx_used);
        self.gpa.free(v.pend_kv);
        self.gpa.free(v.pend_gate);
        self.verify = null;
    }
};

fn arrayBytes(a: A) u64 {
    if (a.ctx == null) return 0;
    return @as(u64, mlx.mlx_array_size(a)) * @as(u64, mlx.mlx_array_itemsize(a));
}

/// The decode state's resident bytes: every layer's window ring, compressed and indexer rows (their
/// capacity, as allocated), open pooling group, and the DSpark windows. What `kv_cache_bytes` reports for
/// this module, whose state lives outside the KVCache.
pub fn stateBytes(st: *const State) u64 {
    var bytes: u64 = 0;
    for (st.layers) |*l| {
        bytes += arrayBytes(l.win.buf) + arrayBytes(l.pend_kv) + arrayBytes(l.pend_gate);
        if (l.comp) |r| bytes += arrayBytes(r.buf);
        if (l.idx_k) |r| bytes += arrayBytes(r.buf);
    }
    for (st.ds) |r| bytes += arrayBytes(r.buf);
    return bytes + arrayBytes(st.mh);
}

pub fn initState(m: *const Dsv41Model, gpa: std.mem.Allocator) !State {
    const layers = try gpa.alloc(LState, m.n_layers);
    var made: usize = 0;
    errdefer {
        for (layers[0..made]) |*l| l.deinit();
        gpa.free(layers);
    }
    const hd = m.cfg.head_dim;
    const cap = m.cfg.sliding_window + RING_SLACK;
    for (layers, 0..) |*l, li| {
        l.* = .{ .win = try Ring.init(cap, hd, m.act, m.s) };
        made += 1;
        if (m.isKvSource(li)) l.comp = .{ .width = hd, .dtype = m.act };
        if (m.isOwner(li)) l.idx_k = .{ .width = m.cfg.dsv4_index_head_dim, .dtype = m.act };
    }
    var st: State = .{ .gpa = gpa, .layers = layers };
    if (m.n_mtp > 0) {
        st.ds = try gpa.alloc(Ring, m.n_mtp);
        for (st.ds, 0..) |*r, k| r.* = Ring.init(cap, hd, m.act, m.s) catch |e| {
            for (st.ds[0..k]) |*q| q.deinit();
            gpa.free(st.ds);
            return e;
        };
    }
    return st;
}

// ── numerics ──────────────────────────────────────────────────────────────

/// The reference's activation quant round trips (`act_quant`, `fp4_act_quant`):
/// fp8 / 32 / ue8m0, fp4 / 32 / ue8m0, fp4 / 16 / e4m3 (`ROPE_TAIL_SOURCE`).
const Qat = enum { fp8_ue8m0, fp4_ue8m0, fp4_e4m3 };

/// Rope on the last `rd` channels of `x` `[L, D]` or `[L, H, D]`, row k at
/// position `scale * (offset + k)`, adjacent pairs, in f32, then `qk`'s round
/// trip (`ROPE_TAIL_SOURCE`); the result in `out_dt` (null: x's dtype).
/// `offset`: an int32 scalar (an input of a compiled stretch).
fn ropeTail(sc: *Scope, m: *const Dsv41Model, x: A, periods: A, offset: A, scale: f32) !A {
    return ropeQat(sc, m, x, periods, offset, scale, null, null);
}

fn ropeQat(sc: *Scope, m: *const Dsv41Model, x: A, periods: A, offset: A, scale: f32, qk: ?Qat, out_dt: ?mlx.mlx_dtype) !A {
    const n: usize = mlx.mlx_array_size(x);
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{scale}, &[_]c_int{1}, 1, .float32));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const sh = mlx.getShape(x);
    const dt = mlx.mlx_array_dtype(x);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, out_dt orelse dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(n), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @intCast(@min(n, 256)), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RD", @intCast(m.cfg.dsv4_rope_head_dim)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROW", ci(n / dim(x, 0))));
    const kind: c_int = if (m.qat_off or qk == null) 0 else @as(c_int, @intFromEnum(qk.?)) + 1;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "QK", kind));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", out_dt orelse dt));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, periods, offset, scal }, 4);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.rope_kernel, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

fn ropeFor(m: *const Dsv41Model, li: usize) *const Rope {
    return if (m.ratio(li) > 0) &m.rope_yarn else &m.rope_plain;
}

const Split = struct { pre: A, post: A, comb: A };
const Normed = struct { sp: Split, h: A };

/// `hcSplit` of `x` plus the rms-normed collapse of `x` with `cpre` (the
/// previous mix: the staggered hyper-connections). A decode-width block is one
/// stage (`hcStage`), a wide one the mix and norm kernels.
fn hcPreNorm(sc: *Scope, m: *const Dsv41Model, x: A, cpre: A, hc: *const Hc, norm_w: A) !Normed {
    if (dim(x, 0) <= region_rows) {
        const r = try hcStage(sc, m, x, null, cpre, hc, norm_w);
        return .{ .sp = r.sp, .h = r.h };
    }
    return hcPreNormWide(sc, m, x, cpre, hc, norm_w);
}

/// `hcPreNorm` for blocks of many rows, where throughput rules: the mix and norm kernels.
fn hcPreNormWide(sc: *Scope, m: *const Dsv41Model, x: A, cpre: A, hc: *const Hc, norm_w: A) !Normed {
    const L = dim(x, 0);
    const mix = dim(hc.fn_w, 0);
    const tg = m.hc_tg;
    const cols = for (hc_mix_cols) |c| {
        if (mix % c == 0) break c;
    } else unreachable;
    const eps = try hcEps(sc, m);
    var zv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(zv);
    {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(mix) }, 2, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(mix / cols * tg), ci((L + hc_mix_rows - 1) / hc_mix_rows), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", ci(tg)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "R", hc_mix_rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "M", ci(cols)));
        const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, hc.fn_w, hc.scale, hc.base, eps }, 5);
        defer _ = mlx.mlx_vector_array_free(iv);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&zv, m.hc_mix_kernel, iv, cfg, sc.s));
    }
    var z = mlx.mlx_array_new();
    _ = try sc.res(mlx.mlx_vector_array_get(&z, zv, 0), &z);
    return hcNorm(sc, m, z, x, cpre, norm_w, eps);
}

/// Metal caps a register-heavy kernel's threadgroup below 1024 on some GPUs (M1/M2): launches every hc kernel once at
/// `m.hc_tg` threads and halves it until they run.
fn probeHcThreads(gpa: std.mem.Allocator, m: *Dsv41Model) !void {
    while (true) {
        probeHcOnce(gpa, m) catch |e| {
            if (e != error.MlxError or m.hc_tg <= 32 or !mlx.takeErrorIf("Thread group size")) return e;
            m.hc_tg /= 2;
            log.info("[dsv41] hc kernels: {d} threads per threadgroup on this GPU\n", .{m.hc_tg});
            continue;
        };
        return;
    }
}

fn probeHcOnce(gpa: std.mem.Allocator, m: *const Dsv41Model) !void {
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    const D = ci(m.cfg.hidden_size);
    const nh = ci(m.cfg.dsv4_hc_mult);
    const mix = (2 + nh) * nh;
    const x = try sc.zeros(&.{ 1, nh, D }, m.act);
    const y = try sc.zeros(&.{ 1, D }, m.act);
    const w = try sc.zeros(&.{D}, m.act);
    const cpre = try sc.zeros(&.{ 1, nh }, .float32);
    const hc: Hc = .{ .fn_w = try sc.zeros(&.{ mix, nh * D }, .float32), .scale = try sc.zeros(&.{mix}, .float32), .base = try sc.zeros(&.{mix}, .float32) };
    const sp: Split = .{ .pre = none, .post = try sc.zeros(&.{ 1, nh }, .float32), .comb = try sc.zeros(&.{ 1, nh, nh }, .float32) };
    const wide = try hcPreNormWide(&sc, m, x, cpre, &hc, w);
    const plain = try hcStage(&sc, m, x, null, cpre, &hc, w);
    const expand = try hcStage(&sc, m, x, .{ .y = y, .sp = &sp }, cpre, &hc, w);
    try evalAll(&.{ wide.h, wide.sp.comb, plain.h, plain.sp.comb, expand.h, expand.x1, expand.sp.comb });
}

fn hcEps(sc: *Scope, m: *const Dsv41Model) !A {
    return sc.keep(mlx.mlx_array_new_data(&[_]f32{ m.cfg.rms_norm_eps, m.cfg.dsv4_hc_eps }, &[_]c_int{2}, 1, .float32));
}

const Staged = struct { x1: A, sp: Split, h: A };
const Expand = struct { y: A, sp: *const Split };

/// `hcPreNorm` (with `ex`: `hcExpandPreNorm`, the stream being `hcExpand(x, y, ex.sp)`) as
/// `HC_STAGE_SOURCE` and `HC_SPLIT_SOURCE`.
fn hcStage(sc: *Scope, m: *const Dsv41Model, x: A, ex: ?Expand, cpre: A, hc: *const Hc, norm_w: A) !Staged {
    const tg = m.hc_tg;
    const L = dim(x, 0);
    const D = dim(x, 2);
    const nh = m.cfg.dsv4_hc_mult;
    const mix = dim(hc.fn_w, 0);
    const dt = mlx.mlx_array_dtype(x);
    if (mix != (2 + nh) * nh) return error.Dsv41HcStage;
    const eps = try hcEps(sc, m);
    var outs: [3]A = undefined; // h_out, z, x1
    {
        const cfg = mlx.mlx_fast_metal_kernel_config_new();
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(D) }, 2, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(mix) }, 2, .float32));
        if (ex != null) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, mlx.getShape(x).ptr, 3, dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((mix + 1) * tg), ci(L), 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", ci(tg)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", ci(nh)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ND", ci((D + tg - 1) / tg)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
        const iv = if (ex) |e|
            mlx.mlx_vector_array_new_data(&[_]A{ x, try sc.astype(e.y, dt), e.sp.post, e.sp.comb, cpre, norm_w, hc.fn_w, hc.scale, hc.base, eps }, 10)
        else
            mlx.mlx_vector_array_new_data(&[_]A{ x, cpre, norm_w, hc.fn_w, hc.scale, hc.base, eps }, 7);
        defer _ = mlx.mlx_vector_array_free(iv);
        var ov = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ov);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, if (ex != null) m.hc_stage_x_kernel else m.hc_stage_kernel, iv, cfg, sc.s));
        for (outs[0..if (ex != null) 3 else 2], 0..) |*o, k| {
            o.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(o, ov, k));
            _ = try sc.keep(o.*);
        }
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(nh) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(nh) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(nh), ci(nh) }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, ci(L), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", ci(nh)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(m.cfg.dsv4_hc_sinkhorn_iters)));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ outs[1], eps }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.hc_split_kernel, iv, cfg, sc.s));
    var g: [3]A = undefined;
    for (&g, 0..) |*o, k| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, k));
        _ = try sc.keep(o.*);
    }
    return .{ .x1 = if (ex != null) outs[2] else x, .sp = .{ .pre = g[0], .post = g[1], .comb = g[2] }, .h = outs[0] };
}

/// `hcPreNorm` for the stream `hcExpand(x, y, sp)` makes, which it returns beside the rest. At decode
/// width the expand is rebuilt wherever the stage reads it, saving its dispatch from the chain; a wide
/// block pays for that rebuild in every mix, so it expands once.
fn hcExpandPreNorm(sc: *Scope, m: *const Dsv41Model, x: A, y: A, sp: *const Split, cpre: A, hc: *const Hc, norm_w: A) !Staged {
    if (dim(x, 0) <= region_rows) return hcStage(sc, m, x, .{ .y = y, .sp = sp }, cpre, hc, norm_w);
    const x1 = try hcExpand(sc, x, y, sp);
    const r = try hcPreNormWide(sc, m, x1, cpre, hc, norm_w);
    return .{ .x1 = x1, .sp = r.sp, .h = r.h };
}

/// The hc split of the mixes `z` and the collapse of `x` with `cpre`, rms-normed.
fn hcNorm(sc: *Scope, m: *const Dsv41Model, z: A, x: A, cpre: A, norm_w: A, eps: A) !Normed {
    const L = dim(x, 0);
    const D = dim(x, 2);
    const nh = m.cfg.dsv4_hc_mult;
    const tg = m.hc_tg;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(nh) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(nh) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(nh), ci(nh) }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(D) }, 2, mlx.mlx_array_dtype(x)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(tg), ci(L), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", ci(tg)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", ci(nh)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(m.cfg.dsv4_hc_sinkhorn_iters)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(x)));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ z, x, cpre, norm_w, eps }, 5);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.hc_norm_kernel, iv, cfg, sc.s));
    var outs: [4]A = undefined;
    for (&outs, 0..) |*o, k| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, k));
        _ = try sc.keep(o.*);
    }
    return .{ .sp = .{ .pre = outs[0], .post = outs[1], .comb = outs[2] }, .h = outs[3] };
}

/// Collapse the hc copies `[L, hc, D]` with `pre` `[L, hc]`, in f32.
fn hcCollapse(sc: *Scope, x: A, pre: A) !A {
    const w = try sc.expand(pre, 1); // [L, 1, hc]
    const y = try sc.matmul(w, try sc.astype(x, .float32)); // [L, 1, D]
    return sc.astype(try sc.reshape(y, &.{ ci(dim(x, 0)), ci(dim(x, 2)) }), mlx.mlx_array_dtype(x));
}

/// `out[k] = post[k] * y + sum_j comb[j, k] * x[j]` — the new stream.
fn hcExpand(sc: *Scope, x: A, y: A, sp: *const Split) !A {
    const L = dim(x, 0);
    const x4 = try sc.reshape(x, &.{ 1, ci(L), ci(dim(x, 1)), ci(dim(x, 2)) });
    const c: glm5.Collapsed = .{ .x = none, .post = sp.post, .comb = sp.comb };
    const o = try sc.keep(try glm5.expand(sc.s, try sc.expand(y, 0), x4, &c));
    return sc.reshape(o, mlx.getShape(x));
}

fn clampedSwiglu(sc: *Scope, gate: A, up: A, limit: f32) !A {
    var g = try sc.astype(gate, .float32);
    var u_ = try sc.astype(up, .float32);
    if (limit > 0) {
        g = try sc.op2(mlx.mlx_minimum, g, try sc.f(limit));
        u_ = try sc.clip(u_, -limit, limit);
    }
    return sc.mul(try sc.mul(g, try sc.sigmoid(g)), u_);
}

// ── attention ──────────────────────────────────────────────────────────────

/// What attention layers hand down within one span of one chunk.
const Shared = struct {
    topk: A = none, // [L, k] compressed indices, -1 = none
    cand: A = none, // [L, n_blocks] candidate blocks
    img: A = none, // [L] bool: the rows inside an image span (`Spans.img` for this span)
    /// A replay tail's first position: the window band stops there, the decode-width graphs are skipped and no
    /// compressor runs (the split layer's read the whole chunk).
    floor: ?usize = null,
};

/// `x` normed attention input `[L, dim]` at positions [p0, p0 + L). With `pre` (the rows' projections,
/// made over the whole chunk by `wideLayer`) the heads `[L, H, hd]` come back before `attnOut`.
fn attention(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, x: A, p0: usize, sh: *Shared, pre: ?Proj) !A {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const L = dim(x, 0);
    const hd = c.head_dim;
    const ls = &st.layers[li];
    const pos = try sc.i(ci(p0));
    const compiled = pre == null and sh.floor == null and compiledAt(m, L);

    var pj: Proj = undefined;
    if (pre) |p| pj = p else if (compiled) {
        var a: [3]A = undefined;
        try runRegion(sc, m, li, .q_kv, &.{ x, pos }, &a);
        pj = .{ .q = a[0], .kv = a[1], .qr = a[2] };
    } else pj = try qkvProj(sc, m, ly, ropeFor(m, li).fwd, x, pos);
    const q = pj.q;
    const qr = pj.qr;
    const kv = pj.kv;

    const window = c.sliding_window;
    var keys_w = none;
    var wp: usize = 0;
    if (compiled) try ls.win.write(kv, p0, sc.s) else {
        const prev_lo = @max(sh.floor orelse 0, if (p0 + 1 > window) p0 + 1 - window else 0);
        const prev = if (p0 > prev_lo) try ls.win.read(sc, prev_lo, p0) else none;
        wp = p0 - prev_lo;
        keys_w = if (prev.ctx != null) try sc.concat(&.{ prev, kv }, 0) else kv;
        try ls.win.write(kv, p0, sc.s);
    }

    var comp_rows = none;
    var topk = none;
    if (ly.ratio > 0) {
        const r: usize = ly.ratio;
        const clen = (p0 + L) / r;
        if (ly.comp) |*cw| if (sh.floor == null) try compress(sc, m, st, li, cw, x, p0);
        if (ly.idx) |*iw| {
            sh.topk = if (clen == 0) none else try indexer(sc, m, st, li, iw, x, qr, p0, sh);
        }
        topk = sh.topk;
        if (clen > 0 and topk.ctx != null) {
            const src = m.latest(li, Dsv41Model.isKvSource);
            comp_rows = try sc.keep(try st.layers[src].comp.?.view(clen, sc.s));
        }
    }

    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    if (compiled) {
        const o = try attnDecode(sc, m, ly, q, ls.win, comp_rows, topk, p0, scale, ropeFor(m, li).inv);
        var y: [1]A = undefined;
        try runRegion(sc, m, li, .attn_out, &.{o}, &y);
        return y[0];
    }
    const K = if (comp_rows.ctx != null) dim(topk, 1) else 0;
    const f16_core = m.attn_f16 and (m.act == .bfloat16 or m.act == .float16) and dim(keys_w, 0) + K <= ATTN_SOFTMAX_MAX_COLS and mlx.streamIsGpu(sc.s);
    const o = if (f16_core)
        try attnWide(sc, m, ly.sink, q, keys_w, wp, comp_rows, topk, scale, m.act)
    else
        try attnWideF32(sc, ly.sink, q, keys_w, try windowBand(sc, wp, L, window), comp_rows, topk, scale, m.act);
    if (m.trace) |t| {
        try t.put("sa_q_{d}", .{li}, q);
        try t.put("sa_kv_{d}", .{li}, keys_w);
        try t.put("sa_o_{d}", .{li}, o);
    }

    if (pre != null) return o;
    return attnOut(sc, m, ly, ropeFor(m, li).inv, o, pos);
}

/// `[L, wp + L]`: query i sees the window rows (wp + i - window, wp + i].
fn windowBand(sc: *Scope, wp: usize, L: usize, window: usize) !A {
    const qpos = try sc.reshape(try sc.arange(wp, wp + L), &.{ ci(L), 1 });
    const kpos = try sc.reshape(try sc.arange(0, wp + L), &.{ 1, ci(wp + L) });
    return sc.op2(mlx.mlx_logical_and, try sc.op2(mlx.mlx_less_equal, kpos, qpos), try sc.op2(mlx.mlx_greater, kpos, try sc.sub(qpos, try sc.i(@intCast(window)))));
}

/// A span's scores, softmax and weighted rows as the f32 op chain: the window
/// band `band` [L, Wk], then the selected compressed rows (`topk` < 0 = none);
/// the per-head sink joins the softmax denominator only.
fn attnWideF32(sc: *Scope, sink: A, q: A, keys_w: A, band: A, comp_rows: A, topk: A, scale: f32, out_dt: mlx.mlx_dtype) !A {
    const qf = try sc.astype(q, .float32);
    const kw = try sc.astype(keys_w, .float32);
    var logits = try sc.matmul(qf, try sc.t2(kw)); // [L, H, Wk]
    var mask = try sc.expand(band, 1);
    var g = none;
    if (comp_rows.ctx != null) {
        const safe = try sc.op2(mlx.mlx_maximum, topk, try sc.i(0));
        g = try sc.astype(try sc.take(comp_rows, safe, 0), .float32); // [L, k, hd]
        const lc = try sc.matmul(qf, try sc.transpose(g, &.{ 0, 2, 1 })); // [L, H, k]
        logits = try sc.concat(&.{ logits, lc }, -1);
        mask = try sc.concat(&.{ mask, try sc.expand(try sc.op2(mlx.mlx_greater_equal, topk, try sc.i(0)), 1) }, -1);
    }
    logits = try sc.where(mask, try sc.mul(logits, try sc.f(scale)), try sc.f(-std.math.inf(f32)));
    const mx = try sc.op2(mlx.mlx_maximum, try sc.max(logits, -1, true), sink);
    const p = try sc.op1(mlx.mlx_exp, try sc.sub(logits, mx));
    const denom = try sc.add(try sc.sum(p, -1, true), try sc.op1(mlx.mlx_exp, try sc.sub(sink, mx)));
    const wk = dim(keys_w, 0);
    var o = try sc.matmul(try sc.slice(p, -1, 0, wk), kw);
    if (g.ctx != null) o = try sc.add(o, try sc.matmul(try sc.slice(p, -1, wk, dim(p, -1)), g));
    return sc.astype(try sc.div(o, denom), out_dt); // [L, H, hd]
}

/// `attnWideF32` with f16 GEMMs and one softmax kernel (`ATTN_SOFTMAX_SOURCE`) that finishes the scale in f32 and writes
/// normalized f16 probabilities, so no sum leaves f16's range. Operands convert to f16 exactly (fp8/fp4 codes times
/// power-of-two scales) and the keys are rms-normed, so nothing nears f16's maximum.
fn attnWide(sc: *Scope, m: *const Dsv41Model, sink: A, q: A, keys_w: A, wp: usize, comp_rows: A, topk: A, scale: f32, out_dt: mlx.mlx_dtype) !A {
    const L = dim(q, 0);
    const H = dim(q, 1);
    const D = dim(q, 2);
    const Wk = dim(keys_w, 0);
    const K = if (comp_rows.ctx != null) dim(topk, 1) else 0;
    const h16 = mlx.mlx_dtype.float16;
    // q carries the power of two under `scale` (exact), so the raw f16 scores sit within 2x of the
    // logits and overflow only where the logits would; the kernel applies the rest in f32.
    const pow2 = @exp2(@floor(@log2(scale)));
    const qh = try sc.astype(try sc.mul(q, try sc.astype(try sc.f(pow2), mlx.mlx_array_dtype(q))), h16);
    const kw = try sc.astype(keys_w, h16);
    const lw = try sc.matmul(try sc.reshape(qh, &.{ ci(L * H), ci(D) }), try sc.t2(kw)); // [L*H, Wk]
    var g = none;
    var lc = lw;
    var picks = none;
    if (K > 0) {
        picks = try sc.astype(topk, .int32);
        const safe = try sc.op2(mlx.mlx_maximum, picks, try sc.i(0));
        g = try sc.take(try sc.astype(comp_rows, h16), safe, 0); // [L, K, D]
        lc = try sc.matmul(qh, try sc.transpose(g, &.{ 0, 2, 1 })); // [L, H, K]
    }
    const meta_v = [_]i32{ @intCast(wp), @intCast(m.cfg.sliding_window), @intCast(Wk), @intCast(K) };
    const meta = try sc.ints(&meta_v, &.{meta_v.len});
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{scale / pow2}, &[_]c_int{1}, 1, .float32));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L * H), ci(Wk) }, 2, h16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(if (K > 0) L else 1), ci(if (K > 0) H else 1), ci(@max(K, 1)) }, 3, h16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(H * 32), ci(L), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(@min(H * 32, 128)), 1, 1));
    // NJ rounded to a multiple of 4: a handful of kernel variants per process, not one per span width.
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NJ", ci((Wk + K + 127) / 128 * 4)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", h16));
    const inputs = [_]A{ lw, lc, if (K > 0) picks else meta, sink, meta, scal };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.attn_softmax_kernel, iv, cfg, sc.s));
    var pw = mlx.mlx_array_new();
    _ = try sc.res(mlx.mlx_vector_array_get(&pw, ov, 0), &pw);
    var pc = mlx.mlx_array_new();
    _ = try sc.res(mlx.mlx_vector_array_get(&pc, ov, 1), &pc);
    var o = try sc.reshape(try sc.matmul(pw, kw), &.{ ci(L), ci(H), ci(D) });
    if (K > 0) o = try sc.add(o, try sc.matmul(pc, g));
    return sc.astype(o, out_dt);
}

/// The indexer's scores `[L, nb]` (`IDX_SCORE_SOURCE`): a decode block keeps the scalar kernel's bits,
/// a wide block whose geometry tiles by 8 rides the matrix kernel.
fn indexScores(sc: *Scope, m: *const Dsv41Model, q: A, keys: A, weights: A, wscale: f32, p0: usize, ratio: usize) !A {
    const ih = dim(q, 1);
    const ihd = dim(q, 2);
    if (dim(q, 0) > region_rows and dim(keys, 0) > 0 and ih % 8 == 0 and ih <= 128 and ihd % 8 == 0)
        return indexScoresTiled(sc, m, q, keys, weights, wscale, p0, ratio);
    return indexScoresScalar(sc, m, q, keys, weights, wscale, p0, ratio);
}

fn indexScoresTiled(sc: *Scope, m: *const Dsv41Model, q: A, keys: A, weights: A, wscale: f32, p0: usize, ratio: usize) !A {
    const L = dim(q, 0);
    const nb = dim(keys, 0);
    const dt = mlx.mlx_array_dtype(keys);
    const kt: usize = if (dt == .float32) 32 else 64; // the staged tile stays at 16 KB
    const meta = try sc.ints(&.{ @intCast(p0), @intCast(ratio) }, &.{2});
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(nb) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((nb + kt - 1) / kt * 128), ci(L), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IH", ci(dim(q, 1))));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IHD", ci(dim(q, 2))));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KT", ci(kt)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const inputs = [_]A{ try sc.astype(q, dt), keys, weights, meta, try sc.keep(mlx.mlx_array_new_data(&[_]f32{wscale}, &[_]c_int{1}, 1, .float32)) };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.idx_tile_kernel, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

fn indexScoresScalar(sc: *Scope, m: *const Dsv41Model, q: A, keys: A, weights: A, wscale: f32, p0: usize, ratio: usize) !A {
    const L = dim(q, 0);
    const nb = dim(keys, 0);
    const meta = try sc.ints(&.{ @intCast(p0), @intCast(ratio) }, &.{2});
    const tg: usize = 256;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(nb) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((nb + tg - 1) / tg * tg), ci(L), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IH", ci(dim(q, 1))));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IHD", ci(dim(q, 2))));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "KT", mlx.mlx_array_dtype(keys)));
    const inputs = [_]A{ q, keys, weights, meta, try sc.keep(mlx.mlx_array_new_data(&[_]f32{wscale}, &[_]c_int{1}, 1, .float32)) };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.idx_score_kernel, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// `attention`'s scores, softmax and weighted rows for a decode block as one
/// kernel (`ATTN_DEC_SOURCE`); the ring already holds this step's rows. With
/// `inv`, the heads leave inverse-roped (`attnOut`'s first step).
fn attnDecode(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, q: A, win: Ring, comp_rows: A, topk: A, p0: usize, scale: f32, inv: ?A) !A {
    const K = if (comp_rows.ctx != null) dim(topk, 1) else 0;
    return attnDecodeSplit(sc, m, ly, q, win, comp_rows, topk, p0, scale, attnSplits(m.cfg.sliding_window + K), inv);
}

fn attnDecodeSplit(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, q: A, win: Ring, comp_rows: A, topk: A, p0: usize, scale: f32, splits: c_int, inv: ?A) !A {
    const rd: c_int = if (inv != null) @intCast(m.cfg.dsv4_rope_head_dim) else 0;
    const periods = inv orelse try sc.zeros(&.{1}, .float32);
    const L = dim(q, 0);
    const H = dim(q, 1);
    const D = dim(q, 2);
    const K = if (comp_rows.ctx != null) dim(topk, 1) else 0;
    const meta_v = [_]i32{ @intCast(p0), @intCast(L), @intCast(K), @intCast(m.cfg.sliding_window), @intCast(win.cap), if (K > 0) @intCast(dim(comp_rows, 0)) else 0 };
    const meta = try sc.ints(&meta_v, &.{meta_v.len});
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{scale}, &[_]c_int{1}, 1, .float32));
    const comp = if (K > 0) try sc.astype(comp_rows, mlx.mlx_array_dtype(win.buf)) else win.buf;
    const ids = if (K > 0) try sc.astype(topk, .int32) else meta;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    if (splits == 1) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(H), ci(D) }, 3, m.act));
    } else {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(H), splits, ci(D) }, 4, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(H), splits }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(H), splits }, 3, .float32));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(H * ATTN_DEC_NSG * 32), ci(L), splits));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ATTN_DEC_NSG * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "D", ci(D)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSG", ATTN_DEC_NSG));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "S", splits));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "KT", mlx.mlx_array_dtype(win.buf)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", m.act));
    if (splits == 1) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RD", rd));
    const inputs = [_]A{ q, win.buf, comp, ids, ly.sink, meta, scal, periods };
    const iv = mlx.mlx_vector_array_new_data(&inputs, if (splits == 1) inputs.len else inputs.len - 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, if (splits == 1) m.attn_dec_kernel else m.attn_split_kernel, iv, cfg, sc.s));
    if (splits == 1) {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
    }
    var parts: [3]A = undefined;
    for (&parts, 0..) |*x, k| {
        x.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(x, ov, k));
        _ = try sc.keep(x.*);
    }
    const mc = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(mc);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(mc, &[_]c_int{ ci(L), ci(H), ci(D) }, 3, m.act));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(mc, ci(D), ci(H), ci(L)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(mc, ci(D), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "D", ci(D)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "S", splits));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(mc, "OT", m.act));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(mc, "RD", rd));
    const mv = mlx.mlx_vector_array_new_data(&[_]A{ parts[0], parts[1], parts[2], ly.sink, meta, periods }, 6);
    defer _ = mlx.mlx_vector_array_free(mv);
    var mo = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(mo);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&mo, m.attn_merge_kernel, mv, mc, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, mo, 0), &o);
}

/// Threadgroups a (row, head) splits its `rows` (window + selected) into: one
/// per 160 rows, swept on M5 Ultra at 80..1000.
fn attnSplits(rows: usize) c_int {
    return @intCast(@max(1, @min(8, rows / 160)));
}

const Proj = struct { q: A, kv: A, qr: A };

/// Rows at `pos` -> the roped query heads, the window row (roped, fp8
/// round-trip) and the query latent the indexer reads.
fn qkvProj(sc: *Scope, m: *Dsv41Model, ly: *const Layer, periods: A, x: A, pos: A) !Proj {
    const c = &m.cfg;
    const qa, const kv = if (ly.q_kv) |*f| blk: {
        const y = try f.apply(sc, x);
        const n = dim(y, -1);
        break :blk .{ try sc.slice(y, -1, 0, n - c.head_dim), try sc.slice(y, -1, n - c.head_dim, n) };
    } else .{ try ly.wq_a.apply(sc, x), try ly.wkv.apply(sc, x) };
    const qr = try sc.rms(qa, ly.q_norm, c.rms_norm_eps);
    const q = try ropeTail(sc, m, try sc.reshape(try ly.wq_b.apply(sc, qr), &.{ ci(dim(x, 0)), ci(c.num_attention_heads), ci(c.head_dim) }), periods, pos, 1.0);
    return .{ .q = q, .kv = try kvRow(sc, m, ly, periods, kv, pos), .qr = qr };
}

/// The projected window row `kv` normed, roped and fp8 round-tripped.
fn kvRow(sc: *Scope, m: *Dsv41Model, ly: *const Layer, periods: A, kv: A, pos: A) !A {
    const L = dim(kv, 0);
    const D = dim(kv, 1);
    const eps = try sc.keep(mlx.mlx_array_new_data(&[_]f32{m.cfg.rms_norm_eps}, &[_]c_int{1}, 1, .float32));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(D) }, 2, mlx.mlx_array_dtype(kv)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(D), ci(L), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(D), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RD", @intCast(m.cfg.dsv4_rope_head_dim)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "QAT", !m.qat_off));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", mlx.mlx_array_dtype(kv)));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ kv, ly.kv_norm, periods, pos, eps }, 5);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.kv_row_kernel, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// Attention heads `o` [L, H, hd] at `pos` back to the stream: the inverse
/// rope, the grouped `wo_a`, then `wo_b`.
fn attnOut(sc: *Scope, m: *Dsv41Model, ly: *const Layer, inv: A, o: A, pos: A) !A {
    return attnOutRoped(sc, m, ly, try ropeTail(sc, m, o, inv, pos, 1.0));
}

/// `attnOut` of heads already inverse-roped.
fn attnOutRoped(sc: *Scope, m: *Dsv41Model, ly: *const Layer, r: A) !A {
    const c = &m.cfg;
    const L = dim(r, 0);
    const og = c.dsv4_o_groups;
    const gin = c.num_attention_heads * c.head_dim / og;
    const ob = try sc.transpose(try sc.reshape(r, &.{ ci(L), ci(og), ci(gin) }), &.{ 1, 0, 2 }); // [og, L, in]
    const oa = if (ly.wo_a.s.ctx == null)
        try sc.matmul(ob, try sc.transpose(ly.wo_a.w, &.{ 0, 2, 1 }))
    else blk: {
        var ro = mlx.mlx_array_new();
        break :blk try sc.res(mlx.mlx_quantized_matmul(&ro, ob, ly.wo_a.w, ly.wo_a.s, ly.wo_a.b, true, mlx.mlx_optional_int.some(@intCast(ly.wo_a.qp.group_size)), mlx.mlx_optional_int.some(@intCast(ly.wo_a.qp.bits)), ly.wo_a.qp.mode.cstr(), sc.s), &ro);
    };
    return ly.wo_b.apply(sc, try sc.astype(try sc.reshape(try sc.transpose(oa, &.{ 1, 0, 2 }), &.{ ci(L), -1 }), m.act));
}

/// `x` [L, K] f32 through the dense linear `l` in f32: decode rows read `l`
/// as stored (`DENSE_F32X_SOURCE`); wider inputs widen it (MLX promotes).
fn denseF32x(sc: *Scope, m: *const Dsv41Model, l: *const Lin, x: A) !A {
    const L = dim(x, 0);
    const N = dim(l.w, 0);
    const K = dim(x, 1);
    if (L > region_rows or l.s.ctx != null or mlx.mlx_array_dtype(l.w) == .float32 or K % 128 != 0 or N % dense_f32x_rps != 0 or !mlx.streamIsGpu(sc.s)) return l.apply(sc, x);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(N) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, ci(N / dense_f32x_rps), ci(L)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RPS", dense_f32x_rps));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, l.w }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.dense_f32x_kernel, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// A KV source's compressor: pools each completed group of `ratio` tokens
/// into one latent (softmax gate in f32 above ratio 1), publishes index keys
/// from the latent before rope when this layer owns them, then ropes the
/// latent at its group's first position and stores it fp4-simulated.
fn compress(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, cw: *const Comp, x: A, p0: usize) !void {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const ls = &st.layers[li];
    const r: usize = ly.ratio;
    var latent = none;
    if (r == 1) {
        latent = try sc.rms(try cw.wkv.apply(sc, x), cw.norm, c.rms_norm_eps);
    } else {
        // Pooling runs in f32, weights included (the reference promotes them).
        const xf = try sc.astype(x, .float32);
        const hd = c.head_dim;
        const kvg = try denseF32x(sc, m, &cw.wkv, xf);
        var kv = try sc.slice(kvg, -1, 0, hd);
        var gate = try sc.slice(kvg, -1, hd, 2 * hd);
        if (ls.pend_kv.ctx != null) {
            kv = try sc.concat(&.{ ls.pend_kv, kv }, 0);
            gate = try sc.concat(&.{ ls.pend_gate, gate }, 0);
        }
        if (st.verify) |*v| {
            v.pend_kv[li] = sc.out(kv);
            v.pend_gate[li] = sc.out(gate);
        }
        const total = dim(kv, 0);
        const g = total / r;
        const full = g * r;
        const new_kv = if (full < total) sc.out(try sc.slice(kv, 0, full, total)) else none;
        const new_gate = if (full < total) sc.out(try sc.slice(gate, 0, full, total)) else none;
        if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
        if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
        ls.pend_kv = new_kv;
        ls.pend_gate = new_gate;
        if (g == 0) return;
        const kg = try sc.reshape(try sc.slice(kv, 0, 0, full), &.{ ci(g), ci(r), ci(hd) });
        const gg = try sc.reshape(try sc.slice(gate, 0, 0, full), &.{ ci(g), ci(r), ci(hd) });
        const pooled = try sc.sum(try sc.mul(kg, try sc.softmax(gg, 1)), 1, false);
        latent = try sc.rms(try sc.astype(pooled, m.act), cw.norm, c.rms_norm_eps);
    }
    const g0 = p0 / r;
    if (ly.idx) |*iw| if (iw.wk) |*wk| {
        var k = try sc.rms(try wk.apply(sc, latent), iw.k_norm, c.rms_norm_eps);
        k = try ropeQat(sc, m, k, m.rope_yarn.fwd, try sc.i(ci(g0)), @floatFromInt(r), .fp4_ue8m0, null);
        try ls.idx_k.?.append(k, sc.s);
    };
    const rot = try ropeQat(sc, m, latent, m.rope_yarn.fwd, try sc.i(ci(g0)), @floatFromInt(r), .fp4_e4m3, null);
    try ls.comp.?.append(rot, sc.s);
}

/// The `k` largest scores of each row as int32 indices (ascending, ties to
/// the lower index); `Scope.topk` off the GPU.
fn topkRows(sc: *Scope, m: *const Dsv41Model, x: A, k: usize) !A {
    if (!mlx.streamIsGpu(sc.s) or mlx.mlx_array_ndim(x) != 2) return sc.astype(try sc.topk(x, k), .int32);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(dim(x, 0)), ci(k) }, 2, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, topk_tg, ci(dim(x, 0)), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, topk_tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TG", topk_tg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", ci(k)));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{try sc.astype(x, .float32)}, 1);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.topk_kernel, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// The top-`index_topk` compressed entries each query keeps (-1 = none),
/// scored against the latest index-key owner's keys. The candidate source
/// also publishes its pinned top blocks; later index sources score only
/// inside them.
fn indexer(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, iw: *const Idx, x: A, qr: A, p0: usize, sh: *Shared) !A {
    const c = &m.cfg;
    const L = dim(x, 0);
    const r: usize = m.ratio(li);
    const nb = (p0 + L) / r;
    const ih = c.dsv4_index_n_heads;
    const ihd = c.dsv4_index_head_dim;
    const owner = m.latest(li, Dsv41Model.isOwner);
    const keys = try sc.keep(try st.layers[owner].idx_k.?.view(nb, sc.s));
    const q = try ropeQat(sc, m, try sc.reshape(try iw.wq_b.apply(sc, qr), &.{ ci(L), ci(ih), ci(ihd) }), m.rope_yarn.fwd, try sc.i(ci(p0)), 1.0, .fp4_ue8m0, .float32);
    const wscale = 1.0 / @sqrt(@as(f32, @floatFromInt(ihd))) / @sqrt(@as(f32, @floatFromInt(ih)));
    // A query sees the groups that end at or before it.
    var scores = try indexScores(sc, m, q, keys, try iw.weights_proj.apply(sc, x), wscale, p0, r); // [L, nb]
    const lens_h = try sc.a.alloc(i32, L);
    defer sc.a.free(lens_h);
    for (lens_h, 0..) |*v, l| v.* = @intCast((p0 + l + 1) / r);
    const lens_i = try sc.ints(lens_h, &.{ ci(L), 1 });
    const ninf = try sc.f(-std.math.inf(f32));

    const cand_src = c.dsv41_candidate_source;
    const bs: usize = c.dsv41_candidate_block_size;
    const n_blocks = (nb + bs - 1) / bs;
    const kb: usize = c.dsv41_candidate_topk_blocks;
    if (cand_src >= 0 and li == @as(usize, @intCast(cand_src)) and n_blocks <= kb) {
        // Every block is kept; the ones it would mask are all -inf in the later sources too (same ratio).
        sh.cand = none;
    } else if (cand_src >= 0 and li == @as(usize, @intCast(cand_src))) {
        var padded = scores;
        if (n_blocks * bs > nb) padded = try sc.concat(&.{ scores, try sc.broadcast(ninf, &.{ ci(L), ci(n_blocks * bs - nb) }) }, -1);
        var blocks = try sc.max(try sc.reshape(padded, &.{ ci(L), ci(n_blocks), ci(bs) }), -1, false);
        // The block holding the query's newest group is only partly full: pin it.
        const last = try sc.op1(mlx.mlx_floor, try sc.div(try sc.astype(try sc.sub(lens_i, try sc.i(1)), .float32), try sc.f(@floatFromInt(bs))));
        const bidx = try sc.reshape(try sc.arange(0, n_blocks), &.{ 1, ci(n_blocks) });
        blocks = try sc.where(try sc.op2(mlx.mlx_equal, try sc.astype(bidx, .float32), last), try sc.f(std.math.inf(f32)), blocks);
        const chosen = try topkRows(sc, m, blocks, kb);
        const reach = try sc.op2(mlx.mlx_greater, try sc.takeAlong(blocks, chosen, -1), ninf);
        var o = mlx.mlx_array_new();
        sh.cand = try sc.res(mlx.mlx_put_along_axis(&o, try sc.zeros(&.{ ci(L), ci(n_blocks) }, .bool_), chosen, reach, -1, sc.s), &o);
    } else if (cand_src >= 0 and li > @as(usize, @intCast(cand_src)) and sh.cand.ctx != null) {
        var o = mlx.mlx_array_new();
        const rep = try sc.res(mlx.mlx_repeat_axis(&o, sh.cand, ci(bs), -1, sc.s), &o);
        scores = try sc.where(try sc.slice(rep, -1, 0, nb), scores, ninf);
    }
    const k = @min(@as(usize, c.dsv4_index_topk), nb);
    const idx = try topkRows(sc, m, scores, k);
    // Every pick is a group the query sees once all of the chunk's rows see all nb groups.
    const picked = if (lens_h[0] == nb) idx else try sc.where(try sc.op2(mlx.mlx_less, idx, lens_i), idx, try sc.i(-1));
    if (m.trace) |t| {
        var o = mlx.mlx_array_new();
        try t.put("topk_{d}", .{li}, try sc.res(mlx.mlx_sort_axis(&o, picked, -1, sc.s), &o));
    }
    return picked;
}

// ── MoE ──────────────────────────────────────────────────────────────────

fn gatherMm(sc: *Scope, x: A, b: *const Bank, lhs: A, idx: A, sorted: bool) !A {
    var o = mlx.mlx_array_new();
    if (b.s.ctx == null) return sc.res(mlx.mlx_gather_mm(&o, x, try sc.transpose(b.w, &.{ 0, 2, 1 }), lhs, idx, sorted, sc.s), &o);
    return sc.res(mlx.mlx_gather_qmm(&o, x, b.w, b.s, b.b, lhs, idx, true, mlx.mlx_optional_int.some(@intCast(b.qp.group_size)), mlx.mlx_optional_int.some(@intCast(b.qp.bits)), b.qp.mode.cstr(), sorted, sc.s), &o);
}

fn gateUp(sc: *Scope, e: *const Experts, x: A, lhs: A, idx: A, sorted: bool) ![2]A {
    const gu = e.gu orelse return .{ try gatherMm(sc, x, &e.gate, lhs, idx, sorted), try gatherMm(sc, x, &e.up, lhs, idx, sorted) };
    const y = try gatherMm(sc, x, &gu, lhs, idx, sorted);
    const n = dim(y, -1) / 2;
    return .{ try sc.slice(y, -1, 0, n), try sc.slice(y, -1, n, 2 * n) };
}

const MOE_GU_INS = [_][*:0]const u8{ "x", "w", "scales", "biases", "sw", "sscales", "sbiases", "inds", "wts", "lim" };
const MOE_DN_INS = [_][*:0]const u8{ "x", "w", "scales", "biases", "sw", "sscales", "sbiases", "inds", "res", "post", "comb" };

/// One token's FFN after its norm, as two kernels (`MOE_GU_SOURCE`,
/// `MOE_DN_SOURCE`): the routed and shared experts and the hc expand of `x1`
/// with `sp`, the new stream `hcExpand` returns; null where they do not apply
/// (rows, formats, an f32 stream).
fn ffnDecode(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, h: A, idx: A, wts: A, x1: A, sp: *const Split) !?A {
    const gu = ly.experts.gu orelse return null;
    const sgu = ly.shared_gu orelse return null;
    const dn = &ly.experts.down;
    const sdn = &ly.shared[2];
    if (dim(h, 0) != 1 or mlx.mlx_array_dtype(h) == .float32 or !mlx.streamIsGpu(sc.s)) return null;
    for ([_]QuantParams{ gu.qp, dn.qp, sgu.qp, sdn.qp }, [_]A{ gu.b, dn.b, sgu.b, sdn.b }, 0..) |qp, b, i| {
        if (i >= 2 and qp.mode == .mxfp8 and qp.bits == 8 and qp.group_size == 32) continue;
        if (b.ctx == null or qp.mode != .affine) return null;
        if (!(qp.bits == 2 or qp.bits == 3 or qp.bits == 4 or qp.bits == 8) or qp.group_size % (if (qp.bits == 3) 8 else 32 / qp.bits) != 0) return null;
    }
    const k = dim(idx, 1);
    const d = dim(h, 1);
    const inter = @divExact(dim(gu.w, 1), 2);
    if (inter % moe_rps != 0 or d % moe_rps != 0) return null;
    const lim = try sc.keep(mlx.mlx_array_new_data(&[_]f32{m.cfg.dsv4_swiglu_limit}, &[_]c_int{1}, 1, .float32));
    const flat = try sc.reshape(idx, &.{ci(k)});
    const dt = mlx.mlx_array_dtype(h);
    const vgu, const vdn = ly.experts.views orelse .{ gu, dn.* };
    // mxfp8 has no biases: its scales fill the input the MX dot never reads.
    const sgb = if (sgu.b.ctx == null) sgu.s else sgu.b;
    const sdb = if (sdn.b.ctx == null) sdn.s else sdn.b;
    const act = try moeKernel(sc, m.moe_gu_kernel, &.{ h, vgu.w, vgu.s, vgu.b, sgu.w, sgu.s, sgb, flat, try sc.reshape(wts, &.{ci(k)}), lim }, &.{ ci(k + 1), ci(inter) }, dt, gu.qp, sgu.qp, d, inter, k);
    return try moeKernel(sc, m.moe_dn_kernel, &.{ act, vdn.w, vdn.s, vdn.b, sdn.w, sdn.s, sdb, flat, x1, sp.post, sp.comb }, mlx.getShape(x1), dt, dn.qp, sdn.qp, inter, d, k);
}

/// Rows a simdgroup in the MoE kernels: swept on M5 Ultra over 1..8 rows and 1..4 simdgroups a threadgroup.
const moe_rps = 2;

fn moeKernel(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, ins: []const A, shape: []const c_int, dt: mlx.mlx_dtype, qp: QuantParams, sqp: QuantParams, K: usize, N: usize, topk: usize) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, dt));
    const gu = shape.len == 2;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, ci(N / moe_rps), if (gu) ci(topk + 1) else 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "RPS", moe_rps));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", ci(K)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", ci(N)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "TOPK", ci(topk)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BITS", @intCast(qp.bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS", @intCast(qp.group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SBITS", @intCast(sqp.bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SGS", @intCast(sqp.group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "SMX", sqp.mode == .mxfp8));
    if (!gu) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", shape[1]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

fn routed(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A, idx: A, wts: A) !A {
    const L = dim(x, 0);
    const k = dim(idx, 1);
    const d = dim(x, 1);
    const limit = m.cfg.dsv4_swiglu_limit;
    const e = &ly.experts;
    const uidx = try sc.astype(idx, .uint32);
    const xe = try sc.reshape(x, &.{ ci(L), 1, 1, ci(d) });
    const one = L == 1 and m.lhs_gu.ctx != null and k == dim(m.lhs_gu, 1);
    const gt, const up = try gateUp(sc, e, xe, if (one) m.lhs_gu else none, uidx, false); // [L, k, 1, inter] each
    const act = try sc.astype(try sc.mul(try clampedSwiglu(sc, gt, up, limit), try sc.reshape(wts, &.{ ci(L), ci(k), 1, 1 })), m.act);
    const dn = try gatherMm(sc, act, &e.down, if (one) m.lhs_dn else none, uidx, false); // [L, k, 1, d]
    return sc.sum(try sc.astype(try sc.reshape(dn, &.{ ci(L), ci(k), ci(d) }), .float32), 1, false);
}

/// A sorted gather of `b` for rows `idx` (one run per expert) on the NAX port where it serves the bank
/// (affine 2..8-bit, bf16 rows), else MLX's; `row_map` reads `x`'s rows in place instead of a `take`.
fn sortedBank(sc: *Scope, x: A, row_map: ?A, b: *const Bank, idx: A) !A {
    if (transformer.verifyQmmNaxAvailable() and b.s.ctx != null and (b.qp.mode == .affine or b.qp.mode.isFp4())) {
        const nax = if (row_map) |map|
            try gather_nax.sortedGatherMapped(x, map, b.w, b.s, b.b, idx, b.qp.bits, b.qp.group_size, b.qp.mode.isFp4(), true, sc.s)
        else
            try gather_nax.sortedGather(x, b.w, b.s, b.b, idx, b.qp.bits, b.qp.group_size, b.qp.mode.isFp4(), true, sc.s);
        if (nax) |y| return sc.keep(y);
    }
    const xs = if (row_map) |map| try sc.take(x, map, 0) else x;
    return gatherMm(sc, xs, b, none, idx, true);
}

const Tmpl = struct { name: [*:0]const u8, v: c_int };

/// One thread an output element, 256 a threadgroup.
fn flatKernel(sc: *Scope, kern: mlx.mlx_fast_metal_kernel, ins: []const A, shape: []const c_int, dt: mlx.mlx_dtype, total: usize, tmpl: []const Tmpl) !A {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, shape.ptr, shape.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((total + 255) / 256 * 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    for (tmpl) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, t.v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    const iv = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// The routed experts over `L` rows at once plus the shared expert `sh`, as `[L, d]` in `x`'s dtype:
/// the n = L k pairs sorted by expert so each bank streams once, gate+up and down as sorted gathers
/// (`sortedBank`), the SwiGLU x weight and the unsort + k-sum + shared add one kernel each.
fn moePrefill(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A, idx: A, wts: A, sh: A) !A {
    const L = dim(x, 0);
    const k = dim(idx, 1);
    const d = dim(x, 1);
    const n = L * k;
    const e = &ly.experts;
    const dt = mlx.mlx_array_dtype(x);
    const flat = try sc.reshape(try sc.astype(idx, .uint32), &.{ci(n)});
    var oo = mlx.mlx_array_new();
    const order = try sc.res(mlx.mlx_argsort_axis(&oo, flat, 0, sc.s), &oo);
    var oi = mlx.mlx_array_new();
    const inv = try sc.res(mlx.mlx_argsort_axis(&oi, order, 0, sc.s), &oi);
    const rows = try sc.op2(mlx.mlx_floor_divide, order, try sc.u(@intCast(k)));
    const sidx = try sc.take(flat, order, 0);
    const x3 = try sc.reshape(x, &.{ ci(L), 1, ci(d) });
    const y = if (e.gu) |*gu| try sortedBank(sc, x3, rows, gu, sidx) else try sc.concat(&.{ try sortedBank(sc, x3, rows, &e.gate, sidx), try sortedBank(sc, x3, rows, &e.up, sidx) }, -1); // [n, 1, 2 inter]
    const inter = dim(y, -1) / 2;
    const lim = try sc.keep(mlx.mlx_array_new_data(&[_]f32{m.cfg.dsv4_swiglu_limit}, &[_]c_int{1}, 1, .float32));
    const w_sorted = try sc.take(try sc.reshape(wts, &.{ci(n)}), order, 0);
    const act = try flatKernel(sc, m.moe_act_kernel, &.{ try sc.reshape(y, &.{ ci(n), ci(2 * inter) }), w_sorted, lim }, &.{ ci(n), 1, ci(inter) }, dt, n * inter, &.{.{ .name = "N", .v = ci(inter) }});
    const dn = try sortedBank(sc, act, null, &e.down, sidx); // [n, 1, d]
    return flatKernel(sc, m.moe_combine_kernel, &.{ try sc.reshape(dn, &.{ ci(n), ci(d) }), inv, try sc.astype(sh, dt) }, &.{ ci(L), ci(d) }, dt, L * d, &.{ .{ .name = "D", .v = ci(d) }, .{ .name = "K", .v = ci(k) } });
}

const Routes = struct { idx: A, wts: A };

/// sqrt-softplus scores; the bias picks the top-k, the unbiased scores
/// (normalized, times the routed scale) weigh them.
/// f32 router logits `x @ gate_w^T`: the GEMV kernel at decode width, one f32 matmul once the rows are many.
fn gateLogits(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A) !A {
    const L = dim(x, 0);
    const E = dim(ly.gate_w, 0);
    if (L > region_rows) {
        const w = try sc.astype(ly.gate_w, .float32);
        return sc.matmul(try sc.astype(x, .float32), try sc.transpose(w, &.{ 1, 0 }));
    }
    const sgs: usize = 8;
    var lo = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(lo);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(E) }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci((E + sgs - 1) / sgs * 32), ci(sgs * L), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, ci(sgs), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SGS", ci(sgs)));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, ly.gate_w }, 2);
    defer _ = mlx.mlx_vector_array_free(iv);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&lo, m.gate_kernel, iv, cfg, sc.s));
    var logits = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&logits, lo, 0), &logits);
}

/// `img` (`[L]` bool, or none) names the rows inside an image span, which pick their experts by `gate_bias_vl`.
fn route(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A, img: A) !Routes {
    const L = dim(x, 0);
    const E = dim(ly.gate_w, 0);
    const K = ly.top_k;
    const logits = try gateLogits(sc, m, ly, x);
    const bias = if (img.ctx != null and ly.gate_bias_vl != null) try sc.where(try sc.expand(img, -1), ly.gate_bias_vl.?, ly.gate_bias) else ly.gate_bias;
    const scal = try sc.keep(mlx.mlx_array_new_data(&[_]f32{m.cfg.router_scaling_factor}, &[_]c_int{1}, 1, .float32));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(K) }, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), ci(K) }, 2, .float32));
    const tg = (E + 31) / 32 * 32;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(tg), ci(L), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", ci(K)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "NORM", m.cfg.moe_route_norm and K > 1));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ logits, bias, scal }, 3);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.route_kernel, iv, cfg, sc.s));
    var idx = mlx.mlx_array_new();
    var wts = mlx.mlx_array_new();
    return .{ .idx = try sc.res(mlx.mlx_vector_array_get(&idx, ov, 0), &idx), .wts = try sc.res(mlx.mlx_vector_array_get(&wts, ov, 1), &wts) };
}

/// The routed sum `y` (f32) plus the shared expert.
fn withShared(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A, y: A) !A {
    return addShared(sc, m, y, try shared(sc, m, ly, x));
}

fn shared(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A) !A {
    const g, const u = if (ly.shared_gu) |*gu| blk: {
        const y = try gu.apply(sc, x);
        const n = dim(y, -1) / 2;
        break :blk .{ try sc.slice(y, -1, 0, n), try sc.slice(y, -1, n, 2 * n) };
    } else .{ try ly.shared[0].apply(sc, x), try ly.shared[1].apply(sc, x) };
    const sh = try clampedSwiglu(sc, g, u, m.cfg.dsv4_swiglu_limit);
    return ly.shared[2].apply(sc, try sc.astype(sh, m.act));
}

fn addShared(sc: *Scope, m: *const Dsv41Model, y: A, sy: A) !A {
    return sc.astype(try sc.add(y, try sc.astype(sy, .float32)), m.act);
}

fn moe(sc: *Scope, m: *const Dsv41Model, li: usize, x: A, img: A) !A {
    const ly = &m.layers[li];
    const r = try route(sc, m, ly, x, img);
    if (dim(x, 0) > region_rows) return moePrefill(sc, m, ly, x, r.idx, r.wts, try shared(sc, m, ly, x));
    return withShared(sc, m, ly, x, try routed(sc, m, ly, x, r.idx, r.wts));
}

// ── Engram ──────────────────────────────────────────────────────────────

/// The gated write of the n-gram rows into the stream `[L, hc, dim]`: `wkv`
/// turns the rows into one key per copy plus a shared value, and the gate is
/// sigmoid of the signed sqrt of a per-copy normalized stream-key dot.
fn engramApply(sc: *Scope, m: *const Dsv41Model, e: *const EngramLayer, x: A, rows: A) !A {
    const kv = try e.wkv.apply(sc, try sc.astype(rows, m.act));
    if (dim(x, 0) <= region_rows and mlx.streamIsGpu(sc.s)) return engramGate(sc, m, e.qk, x, kv);
    return engramGateOps(sc, m, e.qk, x, kv);
}

/// `engramApply` after `wkv` as the op chain.
fn engramGateOps(sc: *Scope, m: *const Dsv41Model, qk: A, x: A, kv: A) !A {
    const c = &m.cfg;
    const L = dim(x, 0);
    const hc = c.dsv4_hc_mult;
    const d = c.hidden_size;
    const key = try sc.reshape(try sc.astype(try sc.slice(kv, -1, 0, hc * d), .float32), &.{ ci(L), ci(hc), ci(d) });
    const value = try sc.astype(try sc.slice(kv, -1, hc * d, (hc + 1) * d), .float32);
    const h = try sc.astype(x, .float32);
    const eps = try sc.f(c.rms_norm_eps);
    const rs_h = try sc.op1(mlx.mlx_rsqrt, try sc.add(try sc.mean(try sc.op1(mlx.mlx_square, h), -1, false), eps));
    const rs_k = try sc.op1(mlx.mlx_rsqrt, try sc.add(try sc.mean(try sc.op1(mlx.mlx_square, key), -1, false), eps));
    const dot = try sc.mul(try sc.mul(try sc.sum(try sc.mul(try sc.mul(h, qk), key), -1, false), try sc.mul(rs_h, rs_k)), try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(d)))));
    const mag = try sc.op1(mlx.mlx_sqrt, try sc.op2(mlx.mlx_maximum, try sc.op1(mlx.mlx_abs, dot), try sc.f(1e-6)));
    const gate = try sc.sigmoid(try sc.where(try sc.op2(mlx.mlx_less, dot, try sc.f(0)), try sc.op1(mlx.mlx_negative, mag), mag));
    const out = try sc.add(h, try sc.mul(try sc.expand(gate, -1), try sc.expand(value, 1)));
    return sc.astype(out, mlx.mlx_array_dtype(x));
}

/// `engramGateOps` as one kernel (`ENGRAM_SOURCE`), a threadgroup a (row, copy).
fn engramGate(sc: *Scope, m: *const Dsv41Model, qk: A, x: A, kv: A) !A {
    const c = &m.cfg;
    const d = c.hidden_size;
    const tg = mlxRowReduceThreads(d);
    const cst_v = [_]f32{ @floatCast(1.0 / @as(f64, @floatFromInt(d))), c.rms_norm_eps, 1.0 / @sqrt(@as(f32, @floatFromInt(d))) };
    const cst = try sc.keep(mlx.mlx_array_new_data(&cst_v, &[_]c_int{cst_v.len}, 1, .float32));
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const sh = mlx.getShape(x);
    const dt = mlx.mlx_array_dtype(x);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, sh.ptr, sh.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(tg), ci(c.dsv4_hc_mult), ci(dim(x, 0))));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, ci(tg), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", dt));
    const iv = mlx.mlx_vector_array_new_data(&[_]A{ x, kv, qk, cst }, 4);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.engram_kernel, iv, cfg, sc.s));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, ov, 0), &o);
}

/// The threads MLX's sum gives a contiguous row of `n` floats (`threadgroup_size_from_row_size`);
/// at 64 or fewer one thread adds the row in order (`row_reduce_small`).
fn mlxRowReduceThreads(n: usize) usize {
    if (n <= 64) return 1;
    if (n <= 512) return 32;
    if (n <= 1024) return 128;
    return @min(1024, ((n + 3) / 4 + 31) / 32 * 32);
}

/// `engramGateOps` for one (row l, copy j) of the stream `x` [L, hc, D], `kv` the `wkv` rows
/// [L, (hc + 1) * D]: the three row sums in MLX's order (`mlxRowReduceThreads`: each thread its
/// 4-wide blocks then its tail, simd sums, then the simdgroup partials), then every op of the
/// chain in its order, uncontracted, so the bits are the chain's. cst = {1 / D, eps, 1 / sqrt(D)}.
const ENGRAM_SOURCE =
    \\#pragma clang fp contract(off)
    \\const uint t = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    \\const int TG = threads_per_threadgroup.x, j = threadgroup_position_in_grid.y, l = threadgroup_position_in_grid.z;
    \\const int HC = qk_shape[0], D = qk_shape[1];
    \\const size_t xo = (size_t(l) * HC + j) * D, ko = size_t(l) * (HC + 1) * D + size_t(j) * D, vo = size_t(l) * (HC + 1) * D + size_t(HC) * D;
    \\threadgroup float part[3][32];
    \\threadgroup float gate_s;
    \\float sh = 0.0f, sk = 0.0f, sd = 0.0f;
    \\const int blocks = D / (TG * 4), extra = D - blocks * TG * 4;
    \\for (int b = 0; b <= blocks; ++b) {
    \\  const int c0 = b * TG * 4 + int(t) * 4;
    \\  const int n = b < blocks ? 4 : metal::clamp(extra - int(t) * 4, 0, 4);
    \\  for (int i = 0; i < n; ++i) {
    \\    const float hv = float(x[xo + c0 + i]), kk = float(kv[ko + c0 + i]);
    \\    const float h2 = hv * hv, k2 = kk * kk, hq = hv * qk[size_t(j) * D + c0 + i];
    \\    const float p = hq * kk;
    \\    sh = h2 + sh;
    \\    sk = k2 + sk;
    \\    sd = p + sd;
    \\  }
    \\}
    \\if (TG > 1) {
    \\  sh = metal::simd_sum(sh);
    \\  sk = metal::simd_sum(sk);
    \\  sd = metal::simd_sum(sd);
    \\}
    \\if (TG > 32) {
    \\  if (lane == 0) { part[0][sg] = sh; part[1][sg] = sk; part[2][sg] = sd; }
    \\  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\  const bool own = int(t) < TG / 32;
    \\  sh = metal::simd_sum(own ? part[0][t] : 0.0f);
    \\  sk = metal::simd_sum(own ? part[1][t] : 0.0f);
    \\  sd = metal::simd_sum(own ? part[2][t] : 0.0f);
    \\}
    \\if (t == 0) {
    \\  const float mh = sh * cst[0], mk = sk * cst[0];
    \\  const float eh = mh + cst[1], ek = mk + cst[1];
    \\  const float rh = metal::precise::rsqrt(eh), rk = metal::precise::rsqrt(ek);
    \\  const float rr = rh * rk;
    \\  const float d1 = sd * rr;
    \\  const float dot = d1 * cst[2];
    \\  const float ad = metal::abs(dot);
    \\  const float mg = metal::precise::sqrt(metal::isnan(ad) || ad > 1e-6f ? ad : 1e-6f);
    \\  const float s = dot < 0.0f ? -mg : mg;
    \\  const float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(s)));
    \\  gate_s = s < 0.0f ? y : 1.0f - y;
    \\}
    \\threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    \\const float g = gate_s;
    \\for (int c = t; c < D; c += TG) {
    \\  const float gv = g * float(kv[vo + c]);
    \\  out[xo + c] = OT(float(x[xo + c]) + gv);
    \\}
;

const ENGRAM_INS = [_][*:0]const u8{ "x", "kv", "qk", "cst" };

// ── the forward ──────────────────────────────────────────────────────────

const Want = enum { last, all };

/// One block over one span: the staggered hyper-connections around
/// attention and the MoE. Returns the new stream; `pre` becomes the FFN's
/// mix, which the next layer's attention collapses with.
fn block(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, x: A, pre: *A, p0: usize, sh: *Shared, ds_main: ?A) !A {
    if (ds_main == null and sh.floor == null and sh.img.ctx == null and li < m.regions.len and compiledAt(m, dim(x, 0))) {
        var a: [4]A = undefined; // the attention's pre, post, comb; its input
        try runRegion(sc, m, li, .attn_in, &.{ x, pre.* }, &a);
        const ao = try attention(sc, m, st, li, a[3], p0, sh, null);
        var o: [2]A = undefined;
        try runRegion(sc, m, li, .ffn, &.{ x, ao, a[0], a[1], a[2] }, &o);
        pre.* = o[1];
        return o[0];
    }
    const mid = try blockAttn(sc, m, st, li, x, pre.*, p0, sh, ds_main);
    const y = try moe(sc, m, li, mid.h, sh.img);
    if (m.trace) |t| try t.put("ffn_{d}", .{li}, y);
    pre.* = mid.sf.pre;
    return hcExpand(sc, mid.x1, y, &mid.sf);
}

/// A block up to its FFN: the stream after attention, the FFN's mix and the
/// normed FFN input.
const Mid = struct { x1: A, sf: Split, h: A };

fn blockAttn(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, x: A, pre: A, p0: usize, sh: *Shared, ds_main: ?A) !Mid {
    const ly = &m.layers[li];
    const pa = try hcPreNorm(sc, m, x, pre, &ly.hc_attn, ly.attn_norm);
    const sa = pa.sp;
    var h = pa.h;
    if (m.trace) |t| {
        try t.put("hc_pre_{d}", .{li}, sa.pre);
        try t.put("hc_post_{d}", .{li}, sa.post);
        try t.put("hc_comb_{d}", .{li}, sa.comb);
        try t.put("attn_in_{d}", .{li}, h);
    }
    h = if (ds_main) |mk| try dsparkAttention(sc, m, li, h, mk, p0) else try attention(sc, m, st, li, h, p0, sh, null);
    if (m.trace) |t| try t.put("attn_{d}", .{li}, h);
    const pf = try hcExpandPreNorm(sc, m, x, h, &sa, sa.pre, &ly.hc_ffn, ly.ffn_norm);
    return .{ .x1 = pf.x1, .sf = pf.sp, .h = pf.h };
}

fn onehotPre(sc: *Scope, m: *const Dsv41Model, L: usize) !A {
    const hc = m.cfg.dsv4_hc_mult;
    const first = try sc.op2(mlx.mlx_equal, try sc.arange(0, hc), try sc.i(0));
    return sc.broadcast(try sc.astype(first, .float32), &.{ ci(L), ci(hc) });
}

/// `t`'s one integer value; waits for the batch that computes `t` only (no new op on it).
fn hostToken(t: A) !u32 {
    try mlx.check(mlx.mlx_array_eval(t));
    return switch (mlx.mlx_array_dtype(t)) {
        .uint32 => (mlx.mlx_array_data_uint32(t) orelse return error.NoData)[0],
        .int32 => @intCast((mlx.mlx_array_data_int32(t) orelse return error.NoData)[0]),
        else => error.Dsv41GpuIdsUnsupported,
    };
}

fn evalAll(xs: []const A) !void {
    const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// Advance `st` by `ids` and return the logits (f32, `[1, V]` for the last
/// position or `[L, V]` for every one; caller frees). Chunks wider than one
/// span run a layer at a time over all spans, evaluating per layer.
pub fn extend(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, ids: []const u32, want: Want) !A {
    return extendIds(m, gpa, st, .{ .host = ids }, want, null);
}

/// A chunk's tokens: on the host, or one decode token still on the GPU (a
/// sampler's output, any integer dtype).
pub const Ids = union(enum) { host: []const u32, gpu: A };

/// The Engram row ids of `ids` (host hash) for the chunk, appended to the history; caller frees.
fn engramIds(m: *const Dsv41Model, gpa: std.mem.Allocator, st: *State, ids: []const u32, images: bool) ![]u32 {
    const hs = &(m.hash orelse return &.{});
    const comp_ids = try gpa.alloc(u32, ids.len);
    defer gpa.free(comp_ids);
    for (ids, comp_ids) |t, *o| o.* = m.compressedId(t, images);
    const out = try gpa.alloc(u32, ids.len * hs.n_layers * hs.cols());
    errdefer gpa.free(out);
    hs.rows(st.hist.items, comp_ids, out);
    try st.hist.appendSlice(st.gpa, comp_ids);
    return out;
}

/// The Engram rows of `ids` (the chunk at `st.n - ids.len`, whose hash is `rows`) as a prefetch: the one the
/// previous chunk started for it, or a new one; either way the prompt's next chunk is read behind. `m.ahead`
/// keeps a prefetch that still has a chunk to hand over; the caller finishes the other.
fn engramPrefetch(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, tables: []const *const engram.Table, ids: []const u32, rows: []const u32, span: usize, images: bool) !*engram.Prefetch {
    const hs = &m.hash.?;
    const p0 = st.n;
    const L = ids.len;
    const p1 = p0 + L;
    const mine = p1 <= m.prompt.len and std.mem.eql(u32, ids, m.prompt[p0..p1]);
    const adopted: ?*engram.Prefetch = if (m.ahead) |a| if (mine and a.covers(p0, L)) a else null else null;
    errdefer if (adopted) |a| a.finish();
    if (m.ahead != null and adopted == null) m.dropAhead();
    m.ahead = null;

    // The next chunk is expected as wide as this one; hist already holds this chunk.
    const w = if (mine) @min(m.prompt.len - p1, L) else 0;
    const next = try gpa.alloc(u32, if (w > region_rows) w * hs.n_layers * hs.cols() else 0);
    defer gpa.free(next);
    if (next.len > 0) {
        const comp = try gpa.alloc(u32, w);
        defer gpa.free(comp);
        for (m.prompt[p1..][0..w], comp) |t, *o| o.* = m.compressedId(t, images);
        hs.rows(st.hist.items, comp, next);
    }
    const ahead_spec: engram.Prefetch.Spec = .{ .at = p1, .len = w, .ids = next };
    if (adopted) |a| {
        if (next.len > 0) m.ahead = try engram.Prefetch.start(gpa, tables, &.{ahead_spec}, hs.cols(), span);
        return a;
    }
    const cur_spec: engram.Prefetch.Spec = .{ .at = p0, .len = L, .ids = rows };
    const pf = try engram.Prefetch.start(gpa, tables, if (next.len > 0) &.{ cur_spec, ahead_spec } else &.{cur_spec}, hs.cols(), span);
    if (next.len > 0) m.ahead = pf;
    return pf;
}

/// `extend` over `ids`. A GPU token goes out as its own batch and the layers
/// before the first Engram layer are queued on it before the host reads it
/// back for the hash: the GPU never waits on that round trip.
/// An image's rows for the chunk's `image_token_id` positions: `rows` is the request's `[1, N, dim]` (or
/// `[N, dim]`) soft tokens in prompt order, `offset` the rows earlier chunks consumed.
pub const VisionRows = struct { rows: A, offset: usize };

fn extendIds(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, ids: Ids, want: Want, vision: ?VisionRows) !A {
    const c = &m.cfg;
    const L = switch (ids) {
        .host => |h| h.len,
        .gpu => |t| mlx.mlx_array_size(t),
    };
    const p0 = st.n;
    const hc = c.dsv4_hc_mult;
    const d = c.hidden_size;
    if (ids == .gpu) {
        if (L != 1 or m.pair != null) return error.Dsv41GpuIdsUnsupported;
        const tv = mlx.mlx_vector_array_new_value(ids.gpu);
        defer _ = mlx.mlx_vector_array_free(tv);
        try mlx.check(mlx.mlx_async_eval(tv));
    }

    // Engram row ids for the whole chunk (host hash); the rows themselves are
    // read span by span at their layer.
    const images = vision != null;
    var eng_ids: []u32 = if (ids == .host) try engramIds(m, gpa, st, ids.host, images) else &.{};
    defer gpa.free(eng_ids);
    const ncols: usize = if (m.hash) |*hs| hs.cols() else 0;

    // What crosses layers: each span's stream and mix, the picks attention
    // layers hand down, and the DSpark main rows; then the same for the
    // replay tail's spans when the chunk has one.
    const span = m.span;
    const n_spans = (L + span - 1) / span;
    const tail = cedTail(m, st, L, want);
    const n_tail = if (tail) |t| (t + span - 1) / span else 0;
    const targets: usize = if (m.n_mtp > 0) c.dsv4_n_dspark_target_layers else 0;
    const held = try gpa.alloc(A, (n_spans + n_tail) * (4 + targets));
    defer {
        for (held) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        gpa.free(held);
    }
    @memset(held, none);
    var sp = spansOf(held, n_spans, targets, span, L, p0);
    sp.eng_ids = eng_ids;
    sp.ncols = ncols;
    var img_mask: A = none;
    defer if (img_mask.ctx != null) {
        _ = mlx.mlx_array_free(img_mask);
    };
    {
        var sc = Scope.init(gpa, m.s);
        defer sc.deinit();
        const id_arr = switch (ids) {
            .host => |h| try sc.ints(@ptrCast(h), &.{ci(L)}),
            .gpu => |t| try sc.reshape(t, &.{1}),
        };
        var emb = try m.embed.rows(&sc, id_arr, m.act);
        if (vision) |vr| if (ids == .host and c.image_token_id != 0 and std.mem.indexOfScalar(u32, ids.host, c.image_token_id) != null) {
            // The span's positions take the encoder's rows (delimiters included) in order, and are marked
            // for the VL routing bias and the Engram pass-through.
            const idx = try gpa.alloc(i32, L);
            defer gpa.free(idx);
            var k = vr.offset;
            for (ids.host, idx) |t, *o| {
                o.* = if (t == c.image_token_id) ci(k) else 0;
                k += @intFromBool(t == c.image_token_id);
            }
            if (k * d > mlx.mlx_array_size(vr.rows)) return error.Dsv41VisionRowsShort;
            const tok = try sc.i(@intCast(c.image_token_id));
            try mlx.check(mlx.mlx_equal(&img_mask, id_arr, tok, m.s));
            sp.img = img_mask;
            const picked = try sc.astype(try sc.take(try sc.reshape(vr.rows, &.{ -1, ci(d) }), try sc.ints(idx, &.{ci(L)}), 0), m.act);
            emb = try sc.where(try sc.expand(img_mask, -1), picked, emb);
        };
        for (0..n_spans) |k| {
            const lo, const hi = sp.bounds(k);
            hold(&sp.xs[k], try sc.broadcast(try sc.expand(try sc.slice(emb, 0, lo, hi), 1), &.{ ci(hi - lo), ci(hc), ci(d) }));
            hold(&sp.pres[k], try onehotPre(&sc, m, hi - lo));
        }
    }

    // The chunk's Engram rows come off the SSD on a thread while the layers before them run (a
    // decode token's once it is read back), and the prompt's next chunk is read behind them: the
    // previous call already did the same for this one.
    const eng_on = m.pair == null and m.engrams.len > 0;
    const eng_tables = try gpa.alloc(*const engram.Table, if (eng_on) m.engrams.len else 0);
    defer gpa.free(eng_tables);
    for (eng_tables, m.engrams[0..eng_tables.len]) |*t, *e| t.* = &e.table;
    var prefetch: ?*engram.Prefetch = null;
    defer if (prefetch) |pf| if (m.ahead != pf) pf.finish();
    if (eng_on and ids == .host) prefetch = try engramPrefetch(m, gpa, st, eng_tables, ids.host, eng_ids, span, images);
    sp.prefetch = prefetch;

    var from: usize = 0;
    if (ids == .gpu and m.engrams.len > 0) {
        var first = m.n_layers;
        for (m.engrams) |e| first = @min(first, e.layer);
        try runLayers(m, gpa, st, &sp, held, 0, first);
        var ev: std.ArrayList(A) = .empty;
        defer ev.deinit(gpa);
        for (held) |x| if (x.ctx != null) try ev.append(gpa, x);
        for (st.layers[0..first]) |*ls| try appendLayerState(gpa, &ev, ls);
        const vec = mlx.mlx_vector_array_new_data(ev.items.ptr, ev.items.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_async_eval(vec));
        const tok = try hostToken(ids.gpu);
        eng_ids = try engramIds(m, gpa, st, &.{tok}, images);
        sp.eng_ids = eng_ids;
        prefetch = try engramPrefetch(m, gpa, st, eng_tables, &.{tok}, eng_ids, span, images);
        sp.prefetch = prefetch;
        from = first;
    }
    if (m.pair) |p| {
        try runLayers(m, gpa, st, &sp, held, 0, p.lo);
        try exchange(m, gpa, &sp, p0, m.snap_at != null and m.snap_at.? == p0 + L, if (st.verify) |v| v.n0 == p0 else false);
        from = p.hi;
    }
    const enc_hi = if (tail != null) m.ced_split.? else m.n_layers;
    try runLayers(m, gpa, st, &sp, held, from, enc_hi);
    if (tail) |tl| {
        var tsp = spansOf(held[n_spans * (4 + targets) ..], n_tail, targets, span, tl, p0 + L - tl);
        tsp.floor = tsp.p0;
        try cedReplayIn(m, gpa, st, &sp, &tsp, enc_hi);
        m.ced_stats.chunks += 1;
        m.ced_stats.skipped += L - tl;
        log.info("[dsv41] CED prefill: {d} rows encoder-only, replay {d} rows\n", .{ L - tl, tl });
        try runLayers(m, gpa, st, &tsp, held, enc_hi, m.n_layers);
        sp = tsp;
    }
    st.n = p0 + L;

    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    if (m.n_mtp > 0) {
        var rows: std.ArrayList(A) = .empty;
        defer rows.deinit(gpa);
        for (0..sp.xs.len) |k| try rows.append(gpa, try sc.concat(sp.mh[k * targets ..][0..targets], -1));
        const all = try sc.concat(rows.items, 0); // [L, targets * dim]
        if (st.verify) |*v| {
            v.mh = sc.out(all);
        } else try commitMain(&sc, m, st, all, sp.p0);
    }

    // The final collapse uses the last layer's FFN mix.
    var h: A = undefined;
    if (want == .last) {
        const k = sp.xs.len - 1;
        const rows = dim(sp.xs[k], 0);
        h = try hcCollapse(&sc, try sc.slice(sp.xs[k], 0, rows - 1, rows), try sc.slice(sp.pres[k], 0, rows - 1, rows));
    } else {
        var parts: std.ArrayList(A) = .empty;
        defer parts.deinit(gpa);
        for (0..sp.xs.len) |k| try parts.append(gpa, try hcCollapse(&sc, sp.xs[k], sp.pres[k]));
        h = try sc.concat(parts.items, 0);
    }
    const logits = try sc.astype(try m.head.apply(&sc, try sc.rms(h, m.norm, c.rms_norm_eps)), .float32);
    var ev: std.ArrayList(A) = .empty;
    defer ev.deinit(gpa);
    try ev.append(gpa, logits);
    for (st.layers) |*ls| try appendLayerState(gpa, &ev, ls);
    for (st.ds) |r| try ev.append(gpa, r.buf);
    const vec = mlx.mlx_vector_array_new_data(ev.items.ptr, ev.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_async_eval(vec));
    return sc.out(logits);
}

/// Layers [lo, hi) over every span of `sp`, one layer at a time; `held` is every array crossing
/// layers, evaluated per layer when there are several spans.
fn runLayers(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, sp: *const Spans, held: []A, lo: usize, hi: usize) !void {
    const n_spans = sp.xs.len;
    // Wide spans run a layer's row-wise stages over the whole chunk, attention's core per span.
    const wide = n_spans > 1 and sp.span > region_rows and m.trace == null;
    for (lo..hi) |li| {
        if (wide) try wideLayer(m, gpa, st, sp, li) else for (0..n_spans) |k| {
            // A span's intermediates are dropped as soon as it is built, so the
            // layer's evaluation frees each one when its consumers are done.
            var sc = Scope.init(gpa, m.s);
            defer sc.deinit();
            var x = try spanIn(&sc, m, gpa, sp, li, k);
            const a, const b = sp.bounds(k);
            var sh: Shared = .{ .topk = sp.topks[k], .cand = sp.cands[k], .floor = sp.floor, .img = try sp.imgSlice(&sc, a, b) };
            var pre = sp.pres[k];
            x = try block(&sc, m, st, li, x, &pre, sp.p0 + k * sp.span, &sh, null);
            if (m.trace) |t| try t.put("stream_{d}", .{li}, x);
            hold(&sp.xs[k], x);
            hold(&sp.pres[k], pre);
            hold(&sp.topks[k], sh.topk);
            hold(&sp.cands[k], sh.cand);
        }
        const decode = n_spans == 1 and sp.L <= region_rows;
        if (n_spans > 1 or (decode and li + 1 < hi)) {
            var ev: std.ArrayList(A) = .empty;
            defer ev.deinit(gpa);
            for (held) |x| if (x.ctx != null) try ev.append(gpa, x);
            try appendLayerState(gpa, &ev, &st.layers[li]);
            if (!decode) try evalAll(ev.items) else {
                // A decode step's graph takes the host about as long as the GPU
                // takes to run it: hand each layer over as it is built.
                const vec = mlx.mlx_vector_array_new_data(ev.items.ptr, ev.items.len);
                defer _ = mlx.mlx_vector_array_free(vec);
                try mlx.check(mlx.mlx_async_eval(vec));
            }
        }
    }
}

/// Host: every span's stream and mix to the worker, and back as they stand after its layers.
/// The picks and candidates restart at the next source layer (a valid cut is one).
fn exchange(m: *Dsv41Model, gpa: std.mem.Allocator, sp: *const Spans, p0: usize, snap: bool, verify: bool) !void {
    const xchg = pair_exchange orelse return error.DsvPairNotJoined;
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    const back = try xchg(try sc.concat(sp.xs, 0), try sc.concat(sp.pres, 0), p0, snap, verify);
    _ = try sc.keep(back[0]);
    _ = try sc.keep(back[1]);
    for (0..sp.xs.len) |k| {
        const lo, const hi = sp.bounds(k);
        hold(&sp.xs[k], try sc.slice(back[0], 0, lo, hi));
        hold(&sp.pres[k], try sc.slice(back[1], 0, lo, hi));
        hold(&sp.topks[k], none);
        hold(&sp.cands[k], none);
    }
}

/// Worker: its layers over one step the host handed over (`x` [L, hc, dim], `pre` [L, hc]),
/// returning both as they stand after them (caller frees). The state follows the host's: a step
/// at 0 starts a new one, a step behind it rewinds to the snapshot, and `snap` takes one after.
/// A verify chunk arms the rollback anchors; the next step's position says how much of it the
/// host kept.
pub fn workerStep(m: *Dsv41Model, gpa: std.mem.Allocator, x: A, pre: A, p0: usize, snap: bool, verify: bool) ![2]A {
    const p = m.pair orelse return error.DsvNotPaired;
    if (m.dec_state) |*prev| if (prev.verify) |v| {
        if (p0 >= v.n0 and p0 < prev.n) try rollbackVerify(m, prev, p0 - v.n0);
        prev.dropVerify();
    };
    if (p0 == 0 or m.dec_state == null) {
        if (m.dec_state) |*old| old.deinit();
        m.dec_state = null;
        m.state_gen += 1;
        m.dec_state = try initState(m, gpa);
    } else if (p0 != m.dec_state.?.n and !try restoreAt(m, p0)) return error.DistPeerDiverged;
    const st = &m.dec_state.?;
    const L = dim(x, 0);
    const n_spans = (L + m.span - 1) / m.span;
    const held = try gpa.alloc(A, n_spans * 4);
    defer {
        for (held) |h| if (h.ctx != null) {
            _ = mlx.mlx_array_free(h);
        };
        gpa.free(held);
    }
    @memset(held, none);
    const sp = spansOf(held, n_spans, 0, m.span, L, p0);
    {
        var sc = Scope.init(gpa, m.s);
        defer sc.deinit();
        for (0..n_spans) |k| {
            const lo, const hi = sp.bounds(k);
            hold(&sp.xs[k], try sc.slice(x, 0, lo, hi));
            hold(&sp.pres[k], try sc.slice(pre, 0, lo, hi));
        }
    }
    if (verify) try armVerify(st, m.n_layers);
    try runLayers(m, gpa, st, &sp, held, p.lo, p.hi);
    st.n = p0 + L;
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    const xo = try sc.concat(sp.xs, 0);
    const po = try sc.concat(sp.pres, 0);
    var ev: std.ArrayList(A) = .empty;
    defer ev.deinit(gpa);
    try ev.appendSlice(gpa, &.{ xo, po });
    for (st.layers[p.lo..p.hi]) |*ls| try appendLayerState(gpa, &ev, ls);
    try evalAll(ev.items);
    if (snap) {
        m.snap_at = st.n;
        try snapshotIfDue(m);
    }
    return .{ sc.out(xo), sc.out(po) };
}

/// Layer `li` over a wide chunk: the row-wise stages (hc mixes, projections, `attnOut`, the FFN half) run
/// once over all of the chunk's rows, attention's state-bound core per span, the MoE over `moe_rows` at a
/// time. A span-wide linear fills a few threadgroups per GPU core; the chunk's rows fill the GPU. New streams and mixes land in `sp.xs` / `sp.pres`.
fn wideLayer(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, sp: *const Spans, li: usize) !void {
    const n_spans = sp.xs.len;
    const ly = &m.layers[li];
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    const parts = try gpa.alloc(A, n_spans);
    defer gpa.free(parts);
    for (parts, 0..) |*p, k| p.* = try spanIn(&sc, m, gpa, sp, li, k);
    const x = try sc.concat(parts, 0);
    const pa = try hcPreNormWide(&sc, m, x, try sc.concat(sp.pres, 0), &ly.hc_attn, ly.attn_norm);
    const pos = try sc.i(ci(sp.p0));
    const pj = try qkvProj(&sc, m, ly, ropeFor(m, li).fwd, pa.h, pos);
    const heads = try gpa.alloc(A, n_spans);
    defer gpa.free(heads);
    for (heads, 0..) |*o, k| {
        const lo, const hi = sp.bounds(k);
        // A span's core intermediates are dropped once built, as in the per-span path.
        var ssc = Scope.init(gpa, m.s);
        defer ssc.deinit();
        var sh: Shared = .{ .topk = sp.topks[k], .cand = sp.cands[k], .floor = sp.floor };
        const span_pj: Proj = .{ .q = try ssc.slice(pj.q, 0, lo, hi), .kv = try ssc.slice(pj.kv, 0, lo, hi), .qr = try ssc.slice(pj.qr, 0, lo, hi) };
        o.* = try sc.keep(ssc.out(try attention(&ssc, m, st, li, try ssc.slice(pa.h, 0, lo, hi), sp.p0 + lo, &sh, span_pj)));
        hold(&sp.topks[k], sh.topk);
        hold(&sp.cands[k], sh.cand);
    }
    const y = try attnOut(&sc, m, ly, ropeFor(m, li).inv, try sc.concat(heads, 0), pos);
    const pf = try hcExpandPreNorm(&sc, m, x, y, &pa.sp, pa.sp.pre, &ly.hc_ffn, ly.ffn_norm);
    const group = @max(1, m.moe_rows / sp.span);
    var k0: usize = 0;
    while (k0 < n_spans) : (k0 += group) {
        const k1 = @min(n_spans, k0 + group);
        const lo = k0 * sp.span;
        const hi = sp.bounds(k1 - 1)[1];
        var gsc = Scope.init(gpa, m.s);
        defer gsc.deinit();
        const yg = try moe(&gsc, m, li, try gsc.slice(pf.h, 0, lo, hi), try sp.imgSlice(&gsc, lo, hi));
        const sf: Split = .{ .pre = try gsc.slice(pf.sp.pre, 0, lo, hi), .post = try gsc.slice(pf.sp.post, 0, lo, hi), .comb = try gsc.slice(pf.sp.comb, 0, lo, hi) };
        const xg = try hcExpand(&gsc, try gsc.slice(pf.x1, 0, lo, hi), yg, &sf);
        for (k0..k1) |k| {
            const a, const b = sp.bounds(k);
            hold(&sp.xs[k], try gsc.slice(xg, 0, a - lo, b - lo));
            hold(&sp.pres[k], try gsc.slice(pf.sp.pre, 0, a, b));
        }
    }
}

/// What crosses layers in `extend`: each span's stream and mix, the picks
/// attention layers hand down, the DSpark main rows, and the Engram row ids.
const Spans = struct {
    xs: []A,
    pres: []A,
    topks: []A,
    cands: []A,
    mh: []A,
    span: usize,
    L: usize,
    p0: usize,
    eng_ids: []const u32,
    ncols: usize,
    prefetch: ?*engram.Prefetch = null,
    /// A replay tail's first position (`Shared.floor`).
    floor: ?usize = null,
    /// `[L]` bool, the chunk's rows inside an image span (none for text): they route by the VL bias and
    /// Engram leaves them untouched.
    img: A = none,

    fn bounds(sp: *const Spans, k: usize) [2]usize {
        const lo = k * sp.span;
        return .{ lo, @min(sp.L, lo + sp.span) };
    }

    fn imgSlice(sp: *const Spans, sc: *Scope, lo: usize, hi: usize) !A {
        return if (sp.img.ctx != null) try sc.slice(sp.img, 0, lo, hi) else none;
    }
};

/// `n` spans over rows [p0, p0 + L) in `held`'s slots: streams, mixes, picks, candidates, then `targets` DSpark
/// main rows per span.
fn spansOf(held: []A, n: usize, targets: usize, span: usize, L: usize, p0: usize) Spans {
    return .{ .xs = held[0..n], .pres = held[n .. 2 * n], .topks = held[2 * n .. 3 * n], .cands = held[3 * n .. 4 * n], .mh = held[4 * n .. 4 * n + n * targets], .span = span, .L = L, .p0 = p0, .eng_ids = &.{}, .ncols = 0 };
}

/// CED: layer `li`'s compressor over every row of `sp` (the decoder's global KV and index keys exist at every
/// position), then `sp`'s last rows become `tsp`'s spans, the decoder's input; `sp`'s streams are released. The rows
/// before the tail get no decoder window KV, which no tail query reaches: the tail's band stops at its first row.
fn cedReplayIn(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, sp: *const Spans, tsp: *const Spans, li: usize) !void {
    const ly = &m.layers[li];
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    for (sp.xs, sp.pres, 0..) |x, pre, k| {
        const pa = try hcPreNormWide(&sc, m, x, pre, &ly.hc_attn, ly.attn_norm);
        try compress(&sc, m, st, li, &ly.comp.?, pa.h, sp.p0 + k * sp.span);
    }
    // The tail's rows come from the spans it overlaps.
    const skip = sp.L - tsp.L;
    const k0 = skip / sp.span;
    const x = try sc.concat(sp.xs[k0..], 0);
    const pre = try sc.concat(sp.pres[k0..], 0);
    const off = skip - k0 * sp.span;
    for (0..tsp.xs.len) |k| {
        const lo, const hi = tsp.bounds(k);
        hold(&tsp.xs[k], try sc.slice(x, 0, off + lo, off + hi));
        hold(&tsp.pres[k], try sc.slice(pre, 0, off + lo, off + hi));
    }
    for ([_][]A{ sp.xs, sp.pres, sp.topks, sp.cands }) |slots| for (slots) |*s| hold(s, none);
}

/// Span `k`'s stream entering layer `li`: its Engram write, and its DSpark
/// main row when the layer is a target.
fn spanIn(sc: *Scope, m: *Dsv41Model, gpa: std.mem.Allocator, sp: *const Spans, li: usize, k: usize) !A {
    const lo, const hi = sp.bounds(k);
    var x = sp.xs[k];
    for (m.engrams, 0..) |*e, ek| {
        if (e.layer != li) continue;
        const nc = sp.ncols;
        const rows = if (sp.prefetch) |pf| blk: {
            const raw = try pf.take(sp.p0, ek, k);
            defer raw.deinit(gpa);
            break :blk try sc.keep(try e.table.dequant(raw, (hi - lo) * nc, sc.s));
        } else blk: {
            const mine = try gpa.alloc(u32, (hi - lo) * nc);
            defer gpa.free(mine);
            for (lo..hi) |t| @memcpy(mine[(t - lo) * nc ..][0..nc], sp.eng_ids[(t * m.engrams.len + ek) * nc ..][0..nc]);
            break :blk try sc.keep(try e.table.gather(gpa, mine, sc.s));
        };
        const y = try engramApply(sc, m, e, x, try sc.reshape(rows, &.{ ci(hi - lo), -1 }));
        // An image span's rows get no Engram write (the reference shuts their gate).
        x = if (sp.img.ctx != null) try sc.where(try sc.expand(try sc.expand(try sp.imgSlice(sc, lo, hi), -1), -1), x, y) else y;
        if (m.trace) |t| try t.put("engram_{d}", .{li}, x);
    }
    if (m.n_mtp > 0) if (m.dsparkTarget(li)) |t| {
        hold(&sp.mh[k * m.cfg.dsv4_n_dspark_target_layers + t], try sc.astype(try sc.mean(try sc.astype(x, .float32), 1, false), m.act));
    };
    return x;
}

// ── compiled decode stretches ────────────────────────────────────────────

/// Blocks of at most this many rows (decode, DSpark and copy verify) take the decode-width kernels, whose grids carry the
/// rows, and run the stretches around attention's state as compiled graphs, one per layer: what precedes attention, its
/// projections, and the FFN half of the block.
const region_rows = 16;

const Region = enum { attn_in, q_kv, attn_out, ffn };

const LayerRegions = struct {
    m: *Dsv41Model,
    li: usize,
    fns: [std.enums.values(Region).len]mlx.mlx_closure = @splat(.{}),
};

fn compiledAt(m: *const Dsv41Model, rows: usize) bool {
    return rows <= region_rows and m.regions.len > 0 and m.trace == null;
}

fn compileRegions(m: *Dsv41Model) !void {
    m.regions = try m.gpa.alloc(LayerRegions, m.n_layers);
    for (m.regions, 0..) |*r, li| r.* = .{ .m = m, .li = li };
    for (m.regions) |*r| inline for (comptime std.enums.values(Region)) |kind| {
        const raw = mlx.mlx_closure_new_func_payload(&RegionBody(kind).call, @ptrCast(r), null);
        defer _ = mlx.mlx_closure_free(raw);
        try mlx.check(mlx.mlx_compile(&r.fns[@intFromEnum(kind)], raw, false));
    };
}

fn freeRegions(m: *Dsv41Model) void {
    for (m.regions) |r| for (r.fns) |f| if (f.ctx != null) {
        _ = mlx.mlx_closure_free(f);
    };
    m.gpa.free(m.regions);
    m.regions = &.{};
}

fn runRegion(sc: *Scope, m: *const Dsv41Model, li: usize, comptime kind: Region, args: []const A, out: []A) !void {
    const in = mlx.mlx_vector_array_new_data(args.ptr, args.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var res = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(res);
    try mlx.check(mlx.mlx_closure_apply(&res, m.regions[li].fns[@intFromEnum(kind)], in));
    for (out, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        o.* = try sc.res(mlx.mlx_vector_array_get(&x, res, i), &x);
    }
}

fn RegionBody(comptime kind: Region) type {
    return struct {
        fn call(res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
            const r: *const LayerRegions = @ptrCast(@alignCast(payload.?));
            run(r.m, r.li, res, in) catch return -1;
            return 0;
        }

        fn run(m: *Dsv41Model, li: usize, res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array) !void {
            const ly = &m.layers[li];
            var a: [8]A = undefined;
            const n = mlx.mlx_vector_array_size(in);
            for (a[0..n], 0..) |*x, i| {
                x.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(x, in, i));
            }
            defer for (a[0..n]) |x| {
                _ = mlx.mlx_array_free(x);
            };
            var sc = Scope.init(std.heap.page_allocator, m.s);
            defer sc.deinit();
            switch (kind) {
                // x, the previous FFN mix -> the attention's mix and its normed input
                .attn_in => {
                    const pa = try hcPreNorm(&sc, m, a[0], a[1], &ly.hc_attn, ly.attn_norm);
                    res.* = mlx.mlx_vector_array_new_data(&.{ pa.sp.pre, pa.sp.post, pa.sp.comb, pa.h }, 4);
                },
                // the attention's input and position -> its query, window row and query latent
                .q_kv => {
                    const pj = try qkvProj(&sc, m, ly, ropeFor(m, li).fwd, a[0], a[1]);
                    res.* = mlx.mlx_vector_array_new_data(&.{ pj.q, pj.kv, pj.qr }, 3);
                },
                // the inverse-roped attention heads -> the attention's output
                .attn_out => res.* = mlx.mlx_vector_array_new_data(&.{try attnOutRoped(&sc, m, ly, a[0])}, 1),
                // x, the attention's output and mix -> the new stream and the FFN's pre
                .ffn => {
                    const sa: Split = .{ .pre = a[2], .post = a[3], .comb = a[4] };
                    const pf = try hcExpandPreNorm(&sc, m, a[0], a[1], &sa, a[2], &ly.hc_ffn, ly.ffn_norm);
                    const x1 = pf.x1;
                    const r = try route(&sc, m, ly, pf.h, none);
                    const y = try ffnDecode(&sc, m, ly, pf.h, r.idx, r.wts, x1, &pf.sp) orelse try hcExpand(&sc, x1, try withShared(&sc, m, ly, pf.h, try routed(&sc, m, ly, pf.h, r.idx, r.wts)), &pf.sp);
                    res.* = mlx.mlx_vector_array_new_data(&.{ y, pf.sp.pre }, 2);
                },
            }
        }
    };
}

/// Point `slot` at `x` (a new handle), releasing what it held.
fn hold(slot: *A, x: A) void {
    if (slot.ctx == x.ctx) return;
    var o = mlx.mlx_array_new();
    if (x.ctx != null) _ = mlx.mlx_array_set(&o, x) else {
        _ = mlx.mlx_array_free(o);
        o = none;
    }
    if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
    slot.* = o;
}

fn appendLayerState(gpa: std.mem.Allocator, ev: *std.ArrayList(A), ls: *const LState) !void {
    try ev.append(gpa, ls.win.buf);
    if (ls.comp) |r| if (r.buf.ctx != null) try ev.append(gpa, r.buf);
    if (ls.idx_k) |r| if (r.buf.ctx != null) try ev.append(gpa, r.buf);
    if (ls.pend_kv.ctx != null) try ev.append(gpa, ls.pend_kv);
    if (ls.pend_gate.ctx != null) try ev.append(gpa, ls.pend_gate);
}

/// A fresh request: drop the old state and prefill `ids`.
pub fn prefill(m: *Dsv41Model, gpa: std.mem.Allocator, ids: []const u32, vision: ?VisionRows) !A {
    if (m.dec_state) |*st| st.deinit();
    m.dec_state = null;
    m.state_gen += 1;
    m.dec_state = try initState(m, gpa);
    return extendResumable(m, gpa, ids, vision);
}

// ── prefix resume: one conversation ──────────────────────────────────────

/// What a resume restores: every buffer a later step replaces rather than
/// appends to is held by reference (a ring's first write after copies it
/// once), and append-only rows by their count.
const Snap = struct {
    gen: u64,
    vision_key: u64,
    /// The state's position: `ids.len` on the host; a worker keeps no ids.
    n: usize,
    ids: []u32,
    layers: []LSnap,
    ds: []A,
    mh: A,
    hist: []u32,

    const LSnap = struct { win: A, comp_used: usize, idx_used: usize, pend_kv: A, pend_gate: A };

    fn deinit(self: *Snap, gpa: std.mem.Allocator) void {
        for (self.layers) |l| for ([_]A{ l.win, l.pend_kv, l.pend_gate }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (self.ds) |a| _ = mlx.mlx_array_free(a);
        if (self.mh.ctx != null) _ = mlx.mlx_array_free(self.mh);
        gpa.free(self.layers);
        gpa.free(self.ds);
        gpa.free(self.hist);
        gpa.free(self.ids);
    }
};

fn ref(x: A) A {
    var o = mlx.mlx_array_new();
    if (x.ctx == null) {
        _ = mlx.mlx_array_free(o);
        return none;
    }
    _ = mlx.mlx_array_set(&o, x);
    return o;
}

/// The state `prompt`'s run reaches one token short of its end is kept for
/// the next request (`snapshotIfDue`); restores the last one when it is a
/// prefix of `prompt` and returns its length (0: a cold prompt). A restore
/// always leaves a token to forward.
pub fn resumePrompt(m: *Dsv41Model, prompt: []const u32, vision_rows: ?A) !usize {
    m.dropAhead();
    m.gpa.free(m.prompt);
    m.prompt = &.{};
    m.prompt = try m.gpa.dupe(u32, prompt);
    const n = try restore(m, prompt, vision_rows);
    m.gpa.free(m.snap_ids);
    m.snap_ids = &.{};
    m.snap_at = null;
    m.snap_vision_key = 0;
    if (prompt.len >= 2) {
        m.snap_ids = try m.gpa.dupe(u32, prompt[0 .. prompt.len - 1]);
        m.snap_at = prompt.len - 1;
        m.snap_vision_key = try visionKey(vision_rows, m.imageRows(m.snap_ids));
    }
    return n;
}

/// What a snapshot keys its images by: a hash of the first `n` soft-token rows (0 for none), `n` being the image
/// positions inside the ids it covers, so the rows of an image after them do not enter. Without rows the image
/// token is text.
pub fn visionKey(rows: ?A, n: usize) !u64 {
    if (n == 0) return 0;
    const r = rows orelse return 0;
    const d = mlx.getShape(r)[mlx.getShape(r).len - 1];
    if (n * @as(usize, @intCast(d)) > mlx.mlx_array_size(r)) return error.Dsv41VisionRowsShort;
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_reshape(&f, r, &[_]c_int{ -1, d }, 2, mlx.gpuStream()));
    try mlx.check(mlx.mlx_slice(&f, f, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n), d }, 2, &[_]c_int{ 1, 1 }, 2, mlx.gpuStream()));
    try mlx.check(mlx.mlx_astype(&f, f, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    const data = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return std.hash.Wyhash.hash(0x5e1f, std.mem.sliceAsBytes(data[0..mlx.mlx_array_size(f)]));
}

fn restore(m: *Dsv41Model, prompt: []const u32, vision_rows: ?A) !usize {
    const sn = &(m.snap orelse return 0);
    const st = &(m.dec_state orelse return 0);
    if (sn.gen != m.state_gen or sn.ids.len >= prompt.len or !std.mem.eql(u32, sn.ids, prompt[0..sn.ids.len])) return 0;
    if (sn.vision_key != try visionKey(vision_rows, m.imageRows(sn.ids))) return 0;
    try applySnap(st, sn);
    return st.n;
}

/// Worker: rewinds to its snapshot at `n`, where the host restored its own.
fn restoreAt(m: *Dsv41Model, n: usize) !bool {
    const sn = &(m.snap orelse return false);
    const st = &(m.dec_state orelse return false);
    if (sn.gen != m.state_gen or sn.n != n) return false;
    try applySnap(st, sn);
    return true;
}

fn applySnap(st: *State, sn: *const Snap) !void {
    st.dropVerify();
    for (st.layers, sn.layers) |*l, s| {
        hold(&l.win.buf, s.win);
        if (l.comp) |*r| r.used = s.comp_used;
        if (l.idx_k) |*r| r.used = s.idx_used;
        hold(&l.pend_kv, s.pend_kv);
        hold(&l.pend_gate, s.pend_gate);
    }
    for (st.ds, sn.ds) |*r, b| hold(&r.buf, b);
    hold(&st.mh, sn.mh);
    st.hist.clearRetainingCapacity();
    try st.hist.appendSlice(st.gpa, sn.hist);
    st.n = sn.n;
}

/// Takes the armed snapshot once the state stands at its position.
fn snapshotIfDue(m: *Dsv41Model) !void {
    const at = m.snap_at orelse return;
    const st = &(m.dec_state orelse return);
    if (st.n != at) return;
    m.snap_at = null;
    if (m.snap) |*old| old.deinit(m.gpa);
    m.snap = null;
    const layers = try m.gpa.alloc(Snap.LSnap, st.layers.len);
    errdefer m.gpa.free(layers);
    const ds = try m.gpa.alloc(A, st.ds.len);
    errdefer m.gpa.free(ds);
    const hist = try m.gpa.dupe(u32, st.hist.items);
    for (layers, st.layers) |*o, *l| o.* = .{
        .win = ref(l.win.buf),
        .comp_used = if (l.comp) |r| r.used else 0,
        .idx_used = if (l.idx_k) |r| r.used else 0,
        .pend_kv = ref(l.pend_kv),
        .pend_gate = ref(l.pend_gate),
    };
    for (ds, st.ds) |*o, r| o.* = ref(r.buf);
    m.snap = .{ .gen = m.state_gen, .vision_key = m.snap_vision_key, .n = st.n, .ids = m.snap_ids, .layers = layers, .ds = ds, .mh = ref(st.mh), .hist = hist };
    m.snap_ids = &.{};
}

/// `extend` that takes the armed snapshot on the way: a chunk crossing its
/// position runs as two.
/// One decode token still on the GPU (`Ids.gpu`), where `extendResumable` takes host ids.
pub fn extendResumableGpu(m: *Dsv41Model, gpa: std.mem.Allocator, tok: A) !A {
    try snapshotIfDue(m);
    const out = try extendIds(m, gpa, &m.dec_state.?, .{ .gpu = tok }, .last, null);
    try snapshotIfDue(m);
    return out;
}

pub fn extendResumable(m: *Dsv41Model, gpa: std.mem.Allocator, ids: []const u32, vision: ?VisionRows) !A {
    try snapshotIfDue(m);
    const st = &m.dec_state.?;
    if (m.snap_at) |at| if (at > st.n and at < st.n + ids.len) {
        const k = at - st.n;
        const head = try extendIds(m, gpa, st, .{ .host = ids[0..k] }, .last, vision);
        _ = mlx.mlx_array_free(head);
        try snapshotIfDue(m);
        // The tail's rows follow the head's.
        var rest = vision;
        if (rest) |*v| v.offset += std.mem.count(u32, ids[0..k], &.{m.cfg.image_token_id});
        return extendIds(m, gpa, st, .{ .host = ids[k..] }, .last, rest);
    };
    const out = try extendIds(m, gpa, st, .{ .host = ids }, .last, vision);
    try snapshotIfDue(m);
    return out;
}

// ── DSpark ──────────────────────────────────────────────────────────────

/// Each stage's window rows for committed positions [p0, p0 + n): the main
/// stream `main_norm(main_proj(hidden))` through the stage's own kv path.
fn commitMain(sc: *Scope, m: *Dsv41Model, st: *State, mh: A, p0: usize) !void {
    const ds = &(m.dspark orelse return);
    const n = dim(mh, 0);
    if (st.mh.ctx != null) _ = mlx.mlx_array_free(st.mh);
    st.mh = sc.out(try sc.slice(mh, 0, n - 1, n));
    const main_x = try sc.rms(try ds.main_proj.apply(sc, mh), ds.main_norm, m.cfg.rms_norm_eps);
    for (st.ds, 0..) |*ring, si| {
        const st_ly = &m.layers[m.n_layers + si];
        const kv = try kvRow(sc, m, st_ly, m.rope_plain.fwd, try st_ly.wkv.apply(sc, main_x), try sc.i(ci(p0)));
        try ring.write(kv, p0, sc.s);
    }
}

/// A stage's attention for the draft rows at [p0, p0 + B): every row sees
/// the stage's window of committed main rows and every draft row.
fn dsparkAttention(sc: *Scope, m: *Dsv41Model, li: usize, x: A, main_keys: A, p0: usize) !A {
    const ly = &m.layers[li];
    const hd = m.cfg.head_dim;
    const pos = try sc.i(ci(p0));
    const pj = try qkvProj(sc, m, ly, m.rope_plain.fwd, x, pos);
    const q = pj.q;
    const keys = try sc.astype(try sc.concat(&.{ main_keys, pj.kv }, 0), .float32);
    const qf = try sc.astype(q, .float32);
    const logits = try sc.mul(try sc.matmul(qf, try sc.t2(keys)), try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(hd)))));
    const mx = try sc.op2(mlx.mlx_maximum, try sc.max(logits, -1, true), ly.sink);
    const p = try sc.op1(mlx.mlx_exp, try sc.sub(logits, mx));
    const denom = try sc.add(try sc.sum(p, -1, true), try sc.op1(mlx.mlx_exp, try sc.sub(ly.sink, mx)));
    const o = try sc.astype(try sc.div(try sc.matmul(p, keys), denom), m.act);
    return attnOut(sc, m, ly, m.rope_plain.inv, o, pos);
}

pub const DsparkDraft = dsv4.DsparkDraft;
pub const DsparkRound = dsv4.DsparkRound;
pub const DsparkPhases = dsv4.DsparkPhases;

/// The draft for the block at [n, n + B): `[t1, noise…]` through the stages,
/// the last stage's collapse, the trunk head over its norm, then the Markov
/// bigram bias chained position by position on the GPU (greedy), and the
/// confidence head's logit per position; it truncates the block only when a threshold is set.
pub fn dsparkDraft(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32) !DsparkDraft {
    return draftRows(m, gpa, st, t1, false);
}

/// `keep_logits` also copies the Markov-biased rows to the host.
fn draftRows(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32, keep_logits: bool) !DsparkDraft {
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    const c = &m.cfg;
    const B = m.ds_block;
    const n = st.n;
    const hc = c.dsv4_hc_mult;
    const d = c.hidden_size;
    const ds = &m.dspark.?;
    var ids: [16]i32 = undefined;
    for (0..B) |k| ids[k] = @intCast(if (k == 0) t1 else c.dsv4_dspark_noise_token_id);
    const emb = try m.embed.rows(&sc, try sc.ints(ids[0..B], &.{ci(B)}), m.act);
    var x = try sc.broadcast(try sc.expand(emb, 1), &.{ ci(B), ci(hc), ci(d) });
    var pre = try onehotPre(&sc, m, B);
    const window = @min(c.sliding_window, n);
    var dummy: Shared = .{};
    for (0..m.n_mtp) |si| {
        const keys = try st.ds[si].read(&sc, n - window, n);
        x = try block(&sc, m, st, m.n_layers + si, x, &pre, n, &dummy, keys);
    }
    const hout = try hcCollapse(&sc, x, pre);
    const logits = try sc.astype(try m.head.apply(&sc, try sc.rms(hout, ds.norm, c.rms_norm_eps)), .float32);
    var cur = try sc.ints(ids[0..1], &.{1});
    const want_conf = m.ds_want_conf or std.math.isFinite(m.ds_conf_thr);
    var toks: [16]A = undefined;
    var confs: [16]A = undefined;
    var rows_k: [16]A = undefined;
    for (0..B) |k| {
        const me = try sc.astype(try sc.take(ds.markov_embed, cur, 0), .float32); // [1, rank]
        const row = try sc.add(try sc.slice(logits, 0, k, k + 1), try sc.matmul(me, try sc.t2(ds.markov_head)));
        rows_k[k] = row;
        if (want_conf) confs[k] = try sc.matmul(try sc.concat(&.{ try sc.astype(try sc.slice(hout, 0, k, k + 1), .float32), me }, -1), try sc.t2(ds.conf));
        cur = try sc.astype(try sc.argmax(row, -1), .int32);
        toks[k] = cur;
    }
    const all = try sc.concat(toks[0..B], 0);
    const rows_all = if (keep_logits) try sc.concat(rows_k[0..B], 0) else all;
    var ev = [_]A{ all, if (want_conf) try sc.concat(confs[0..B], 0) else all, rows_all };
    try evalAll(&ev);
    const out_ids = try gpa.alloc(u32, B + 1);
    errdefer gpa.free(out_ids);
    out_ids[0] = t1;
    const data = mlx.mlx_array_data_int32(all) orelse return error.NoData;
    for (0..B) |k| out_ids[k + 1] = @intCast(data[k]);
    const conf = try gpa.alloc(f32, B);
    errdefer gpa.free(conf);
    var len = B;
    if (want_conf) {
        @memcpy(conf, (mlx.mlx_array_data_float32(try sc.astype(ev[1], .float32)) orelse return error.NoData)[0..B]);
        for (conf, 0..) |v, k| {
            if (v < m.ds_conf_thr) {
                len = k;
                break;
            }
        }
    } else @memset(conf, 0);
    const host_logits = try gpa.alloc(f32, if (keep_logits) B * m.vocab else 0);
    if (keep_logits) @memcpy(host_logits, (mlx.mlx_array_data_float32(rows_all) orelse return error.NoData)[0 .. B * m.vocab]);
    return .{ .ids = out_ids, .len = len, .logits = host_logits, .confidence = conf };
}

pub const DsparkPending = struct {
    vl_g: A,
    b: usize,
    verify: [16]u32,
    phases: DsparkPhases = .{},

    pub fn lapVerify(_: *DsparkPending, _: *const Dsv41Model) void {}

    pub fn deinit(self: *DsparkPending) void {
        _ = mlx.mlx_array_free(self.vl_g);
    }
};

pub fn dsparkBegin(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32) !DsparkPending {
    var d = try dsparkDraft(m, gpa, st, t1);
    defer d.deinit(gpa);
    return dsparkBeginWith(m, gpa, st, t1, &d);
}

/// Most drafts one verify chunk carries: its rows and `t1` fit the windows' slack.
pub const VERIFY_MAX_DRAFTS: usize = RING_SLACK - 1;

/// Arm the rollback anchors and append `[t1, drafts…]` as a verify chunk;
/// the returned `vl_g` holds every position's logits, lazily. The drafts
/// may be the stages' block or a copied continuation (`VERIFY_MAX_DRAFTS`).
pub fn dsparkBeginWith(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32, d: *const DsparkDraft) !DsparkPending {
    const b = d.len;
    if (b > VERIFY_MAX_DRAFTS) return error.TooManyDrafts;
    try armVerify(st, m.n_layers);
    var ids: [16]u32 = undefined;
    ids[0] = t1;
    @memcpy(ids[1 .. b + 1], d.ids[1 .. b + 1]);
    const vl = extend(m, gpa, st, ids[0 .. b + 1], .all) catch |e| {
        st.dropVerify();
        return e;
    };
    return .{ .vl_g = vl, .b = b, .verify = ids };
}

/// The rollback anchors a verify chunk restores to.
fn armVerify(st: *State, nl: usize) !void {
    var v: Verify = .{
        .n0 = st.n,
        .comp_used = try st.gpa.alloc(usize, nl),
        .idx_used = try st.gpa.alloc(usize, nl),
        .pend_kv = try st.gpa.alloc(A, nl),
        .pend_gate = try st.gpa.alloc(A, nl),
    };
    for (st.layers, 0..) |*ls, li| {
        v.comp_used[li] = if (ls.comp) |r| r.used else 0;
        v.idx_used[li] = if (ls.idx_k) |r| r.used else 0;
        v.pend_kv[li] = none;
        v.pend_gate[li] = none;
    }
    st.verify = v;
}

/// Back to `keep_n` positions past the verify chunk's start: offsets roll back and each pooling
/// compressor's open group is re-cut from the chunk's rows.
fn rollbackVerify(m: *Dsv41Model, st: *State, keep_n: usize) !void {
    var sc = Scope.init(st.gpa, m.s);
    defer sc.deinit();
    const v = &st.verify.?;
    const end = v.n0 + keep_n;
    for (st.layers, 0..) |*ls, li| {
        const r = m.ratio(li);
        if (ls.comp) |*rows| rows.used = v.comp_used[li] + (end / r - v.n0 / r);
        if (ls.idx_k) |*rows| rows.used = v.idx_used[li] + (end / r - v.n0 / r);
        if (r > 1 and v.pend_kv[li].ctx != null) {
            const first = v.n0 - v.n0 % r; // position of pend_kv[li] row 0
            const lo = end - end % r - first;
            const hi = end - first;
            if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
            if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
            ls.pend_kv = if (hi > lo) sc.out(try sc.slice(v.pend_kv[li], 0, lo, hi)) else none;
            ls.pend_gate = if (hi > lo) sc.out(try sc.slice(v.pend_gate[li], 0, lo, hi)) else none;
        }
    }
    st.n = end;
    if (st.hist.items.len > end) st.hist.shrinkRetainingCapacity(end);
}

/// Keep `accepted + 1` of the verify chunk's positions: offsets roll back,
/// each pooling compressor's open group is re-cut from the chunk's rows, and
/// the kept positions' main rows go to the DSpark windows.
pub fn dsparkFinish(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, pending: *DsparkPending, accepted: usize, next_token: u32) !DsparkRound {
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    defer st.dropVerify();
    const v = &st.verify.?;
    const keep_n = accepted + 1;
    if (keep_n < pending.b + 1) try rollbackVerify(m, st, keep_n);
    if (v.mh.ctx != null) try commitMain(&sc, m, st, try sc.slice(v.mh, 0, 0, keep_n), v.n0);
    const tokens = try gpa.alloc(u32, keep_n);
    @memcpy(tokens, pending.verify[0..keep_n]);
    var ph = pending.phases;
    ph.accepted = @intCast(accepted);
    ph.committed = @intCast(keep_n);
    ph.submitted = @intCast(pending.b);
    return .{ .tokens = tokens, .next_token = next_token, .accepted = @intCast(accepted), .phases = ph };
}

pub fn dsparkObserve(m: *Dsv41Model, ph: DsparkPhases) void {
    if (m.ds_prof) |*p| {
        p.observe(ph);
        if (p.rounds % 16 == 0) p.report();
    }
}

/// One greedy round: draft, verify, keep the longest prefix the trunk's own
/// argmax agrees with, and the trunk's token after it.
pub fn dsparkRound(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32, accepted_cap: usize) !DsparkRound {
    var d = try dsparkDraft(m, gpa, st, t1);
    defer d.deinit(gpa);
    return dsparkRoundWith(m, gpa, st, t1, &d, accepted_cap);
}

/// `dsparkRound` over a given draft (a copied continuation).
pub fn dsparkRoundWith(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32, d: *const DsparkDraft, accepted_cap: usize) !DsparkRound {
    var pending = try dsparkBeginWith(m, gpa, st, t1, d);
    defer pending.deinit();
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, pending.vl_g, -1, false, m.s));
    var am32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am32);
    try mlx.check(mlx.mlx_astype(&am32, am, .int32, m.s));
    try mlx.check(mlx.mlx_array_eval(am32));
    const best = mlx.mlx_array_data_int32(am32) orelse return error.NoData;
    var accepted: usize = 0;
    while (accepted < pending.b and @as(u32, @intCast(best[accepted])) == d.ids[accepted + 1]) accepted += 1;
    accepted = @min(accepted, accepted_cap);
    return dsparkFinish(m, gpa, st, &pending, accepted, @intCast(best[accepted]));
}

// ── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// A tiny pack from tests/dump_dsv41_fixtures.py, its reference outputs,
/// and the model in f32. MLX runs f32 GEMMs as TF32 on M5 unless told not
/// to, and reads the switch once, at the process's first matmul: a full-suite
/// run sets `MLX_ENABLE_TF32=0` itself. The reference is plain f32.
const Tiny = struct {
    cfg: ModelConfig,
    w: model.Weights,
    fx: model.Weights,
    m: *Dsv41Model,

    fn open(root: []const u8, pack: []const u8, fixture: []const u8) !Tiny {
        return openAct(root, pack, fixture, .float32);
    }

    fn openAct(root: []const u8, pack: []const u8, fixture: []const u8, act: mlx.mlx_dtype) !Tiny {
        _ = setenv("MLX_ENABLE_TF32", "0", 1);
        const gpa = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const dir = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ root, pack });
        defer gpa.free(dir);
        var cfg = try model.parseConfig(io, gpa, dir);
        errdefer cfg.deinit(gpa);
        var w = try model.loadModelWeights(io, gpa, dir, &cfg, false);
        errdefer w.deinit();
        var fx = try model.loadWeightsFile(gpa, dir, fixture);
        errdefer fx.deinit();
        const m = try init(gpa, &cfg, &w, mlx.gpuStream(), .{ .act = act, .dspark = true });
        return .{ .cfg = cfg, .w = w, .fx = fx, .m = m };
    }

    fn deinit(self: *Tiny) void {
        self.m.deinit();
        self.fx.deinit();
        self.w.deinit();
        self.cfg.deinit(testing.allocator);
    }

    fn host(self: *const Tiny, name: []const u8) ![]f32 {
        return hostF32(self.fx.get(name) orelse return error.MissingFixture);
    }

    fn ids(self: *const Tiny) ![]u32 {
        const a = self.fx.get("ids") orelse return error.MissingFixture;
        try mlx.check(mlx.mlx_array_eval(a));
        const d = mlx.mlx_array_data_int32(a) orelse return error.NoData;
        const out = try testing.allocator.alloc(u32, mlx.mlx_array_size(a));
        for (out, 0..) |*o, k| o.* = @intCast(d[k]);
        return out;
    }
};

fn hostI32(a: A) ![]i32 {
    try mlx.check(mlx.mlx_array_eval(a));
    const p = mlx.mlx_array_data_int32(a) orelse return error.NoData;
    return testing.allocator.dupe(i32, p[0..mlx.mlx_array_size(a)]);
}

fn hostF32(a: A) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, mlx.gpuStream()));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, c, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return testing.allocator.dupe(f32, d[0..mlx.mlx_array_size(f)]);
}

/// Rows of `got` against `want` (`rows` x `width`): every row's cosine and
/// norm ratio, and the largest error against the reference's peak.
fn expectRowsClose(label: []const u8, got: []const f32, want: []const f32, width: usize) !void {
    try testing.expectEqual(want.len, got.len);
    var peak: f64 = 0;
    for (want) |v| peak = @max(peak, @abs(v));
    var worst_cos: f64 = 1;
    var worst_ratio: f64 = 1;
    var worst_err: f64 = 0;
    var argmax_miss: usize = 0;
    for (0..want.len / width) |r| {
        const g = got[r * width ..][0..width];
        const w = want[r * width ..][0..width];
        var dot: f64 = 0;
        var gg: f64 = 0;
        var ww: f64 = 0;
        var ag: usize = 0;
        var aw: usize = 0;
        for (g, w, 0..) |a, b, k| {
            dot += @as(f64, a) * b;
            gg += @as(f64, a) * a;
            ww += @as(f64, b) * b;
            worst_err = @max(worst_err, @abs(@as(f64, a) - b) / peak);
            if (a > g[ag]) ag = k;
            if (b > w[aw]) aw = k;
        }
        worst_cos = @min(worst_cos, dot / @sqrt(gg * ww));
        const ratio = @sqrt(gg / ww);
        if (@abs(ratio - 1) > @abs(worst_ratio - 1)) worst_ratio = ratio;
        if (ag != aw) argmax_miss += 1;
    }
    std.debug.print("{s}: min cos {d:.7} rms ratio {d:.6} max err {e:.2} argmax misses {d}\n", .{ label, worst_cos, worst_ratio, worst_err, argmax_miss });
    try testing.expect(worst_cos > 0.9999);
    try testing.expect(@abs(worst_ratio - 1) < 1e-3);
    try testing.expectEqual(@as(usize, 0), argmax_miss);
}

fn logitsHost(m: *Dsv41Model, st: *State, ids: []const u32, want: Want) ![]f32 {
    const l = try extend(m, testing.allocator, st, ids, want);
    defer _ = mlx.mlx_array_free(l);
    return hostF32(l);
}

test "dsv41 fixture: prefill, chunked prefill, spans and decode match DeepSeek's reference (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    for ([_][]const u8{ "pipe", "repack" }) |pack| {
        var t = try Tiny.open(root, pack, "fixture.safetensors");
        defer t.deinit();
        const ids = try t.ids();
        defer gpa.free(ids);
        const V = t.m.vocab;
        const full_ref = try t.host("logits_full");
        defer gpa.free(full_ref);
        const dec_ref = try t.host("logits_decode");
        defer gpa.free(dec_ref);
        const n_prompt = ids.len - dec_ref.len / V;
        var label: [64]u8 = undefined;

        var st = try initState(t.m, gpa);
        const full = try logitsHost(t.m, &st, ids, .all);
        st.deinit();
        defer gpa.free(full);
        try expectRowsClose(try std.fmt.bufPrint(&label, "{s} full prefill", .{pack}), full, full_ref, V);

        // Chunk boundaries inside a ratio-2 group and inside the window.
        st = try initState(t.m, gpa);
        var chunked: std.ArrayList(f32) = .empty;
        defer chunked.deinit(gpa);
        var lo: usize = 0;
        for ([_]usize{ 13, 14, 31, ids.len }) |hi| {
            const part = try logitsHost(t.m, &st, ids[lo..hi], .all);
            defer gpa.free(part);
            try chunked.appendSlice(gpa, part);
            lo = hi;
        }
        st.deinit();
        try expectRowsClose(try std.fmt.bufPrint(&label, "{s} chunked prefill", .{pack}), chunked.items, full_ref, V);

        // Several spans per layer (layer-major) give the same rows.
        t.m.span = 7;
        st = try initState(t.m, gpa);
        const spanned = try logitsHost(t.m, &st, ids, .all);
        st.deinit();
        t.m.span = SPAN;
        defer gpa.free(spanned);
        try expectRowsClose(try std.fmt.bufPrint(&label, "{s} 7-row spans", .{pack}), spanned, full_ref, V);

        // The MoE over several spans at once (9-row spans, two or all of them per call).
        t.m.span = 9;
        for ([_]usize{ 18, 4096 }) |moe_rows| {
            t.m.moe_rows = moe_rows;
            st = try initState(t.m, gpa);
            const wide = try logitsHost(t.m, &st, ids, .all);
            st.deinit();
            defer gpa.free(wide);
            try expectRowsClose(try std.fmt.bufPrint(&label, "{s} {d}-row MoE blocks", .{ pack, moe_rows }), wide, full_ref, V);
        }
        t.m.span = SPAN;
        t.m.moe_rows = MOE_ROWS;

        st = try initState(t.m, gpa);
        defer st.deinit();
        gpa.free(try logitsHost(t.m, &st, ids[0..n_prompt], .last));
        var dec: std.ArrayList(f32) = .empty;
        defer dec.deinit(gpa);
        const draft_ids = t.fx.get("draft_ids");
        const draft_ref = if (draft_ids != null) try t.host("draft_logits") else &.{};
        defer if (draft_ids != null) gpa.free(draft_ref);
        var drafted: std.ArrayList(f32) = .empty;
        defer drafted.deinit(gpa);
        for (n_prompt..ids.len) |i| {
            const row = try logitsHost(t.m, &st, ids[i .. i + 1], .last);
            defer gpa.free(row);
            try dec.appendSlice(gpa, row);
            if (draft_ids == null) continue;
            // The draft continues from the reference's own pick.
            const ref_row = dec_ref[(i - n_prompt) * V ..][0..V];
            var t1: u32 = 0;
            for (ref_row, 0..) |v, k| {
                if (v > ref_row[t1]) t1 = @intCast(k);
            }
            var d = try draftRows(t.m, gpa, &st, t1, true);
            defer d.deinit(gpa);
            try drafted.appendSlice(gpa, d.logits);
        }
        try expectRowsClose(try std.fmt.bufPrint(&label, "{s} decode", .{pack}), dec.items, dec_ref, V);
        if (draft_ids != null) try expectRowsClose(try std.fmt.bufPrint(&label, "{s} DSpark draft", .{pack}), drafted.items, draft_ref, V);
    }
}

test "dsv41 fixture ladder: with the QAT round-trips off every layer matches in f32 (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    for ([_][]const u8{ "pipe", "repack" }) |pack| {
        var t = try Tiny.open(root, pack, "fixture_noqat.safetensors");
        defer t.deinit();
        t.m.qat_off = true;
        const ids = try t.ids();
        defer gpa.free(ids);
        var tr: Trace = .{ .gpa = gpa };
        defer tr.deinit();
        t.m.trace = &tr;
        var st = try initState(t.m, gpa);
        defer st.deinit();
        gpa.free(try logitsHost(t.m, &st, ids, .all));
        t.m.trace = null;
        for (st.layers, 0..) |*ls, li| {
            inline for (.{ "comp", "idx_k" }, .{ "comp_{d}", "idxk_{d}" }) |field, name| {
                if (@field(ls, field)) |r| {
                    const v = try r.view(r.used, mlx.gpuStream());
                    defer _ = mlx.mlx_array_free(v);
                    try tr.put(name, .{li}, v);
                }
            }
        }
        std.debug.print("{s}:\n", .{pack});
        for (tr.items.items) |it| {
            const want = t.host(it.name) catch continue;
            defer gpa.free(want);
            const got = try hostF32(it.arr);
            defer gpa.free(got);
            if (std.mem.startsWith(u8, it.name, "topk")) {
                var diff: usize = 0;
                for (got, want) |a, b| diff += @intFromBool(a != b);
                std.debug.print("  {s:<10} {d}/{d} picks differ\n", .{ it.name, diff, got.len });
                try testing.expectEqual(@as(usize, 0), diff);
                continue;
            }
            var dot: f64 = 0;
            var gg: f64 = 0;
            var ww: f64 = 0;
            var worst: f64 = 0;
            var peak: f64 = 0;
            for (got, want) |a, b| {
                dot += @as(f64, a) * b;
                gg += @as(f64, a) * a;
                ww += @as(f64, b) * b;
                worst = @max(worst, @abs(@as(f64, a) - b));
                peak = @max(peak, @abs(b));
            }
            std.debug.print("  {s:<10} cos {d:.7} ratio {d:.5} max err {e:.2}\n", .{ it.name, dot / @sqrt(gg * ww), @sqrt(gg / ww), worst / peak });
            try testing.expect(worst / peak < 1e-4);
        }
    }
}

fn argmaxRow(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, k| {
        if (v > row[best]) best = k;
    }
    return @intCast(best);
}

/// Greedy tokens from `st` after `prompt` (prefilled here), `n` of them.
fn serialGreedy(m: *Dsv41Model, st: *State, prompt: []const u32, n: usize, out: []u32) !u32 {
    var row = try logitsHost(m, st, prompt, .last);
    var t = argmaxRow(row);
    for (0..n) |k| {
        testing.allocator.free(row);
        out[k] = t;
        row = try logitsHost(m, st, &.{t}, .last);
        t = argmaxRow(row);
    }
    testing.allocator.free(row);
    return t;
}

test "dsv41 decode: a token still on the GPU decodes as its host id does, Engram layers included (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    try testing.expect(t.m.engrams.len > 0);
    const ids = try t.ids();
    defer gpa.free(ids);
    var host_st = try initState(t.m, gpa);
    defer host_st.deinit();
    var gpu_st = try initState(t.m, gpa);
    defer gpu_st.deinit();
    var lh = try extend(t.m, gpa, &host_st, ids, .last);
    defer _ = mlx.mlx_array_free(lh);
    var lg = try extend(t.m, gpa, &gpu_st, ids, .last);
    defer _ = mlx.mlx_array_free(lg);
    for (0..6) |_| {
        const tok = try argmaxHost(lh);
        const nh = try extend(t.m, gpa, &host_st, &.{tok}, .last);
        _ = mlx.mlx_array_free(lh);
        lh = nh;
        var g = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(g);
        try mlx.check(mlx.mlx_argmax_axis(&g, lg, -1, false, mlx.gpuStream()));
        const ng = try extendIds(t.m, gpa, &gpu_st, .{ .gpu = g }, .last, null);
        _ = mlx.mlx_array_free(lg);
        lg = ng;
        const a = try hostF32(lh);
        defer gpa.free(a);
        const b = try hostF32(lg);
        defer gpa.free(b);
        for (a, b) |x, y| try testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
    }
    try testing.expectEqualSlices(u32, host_st.hist.items, gpu_st.hist.items);
}

test "dsv41 fixture: DSpark rounds emit serial greedy; rollback and full accept leave serial state (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const m = t.m;
    try testing.expect(m.n_mtp > 0);
    const ids = try t.ids();
    defer gpa.free(ids);
    const prompt = ids[0..32];
    const N = 24;
    const B = m.ds_block;
    var serial: [N + 16]u32 = undefined;
    var st_s = try initState(m, gpa);
    defer st_s.deinit();
    const next_s = try serialGreedy(m, &st_s, prompt, N, &serial);

    var st_d = try initState(m, gpa);
    defer st_d.deinit();
    const first = try logitsHost(m, &st_d, prompt, .last);
    var tok = argmaxRow(first);
    gpa.free(first);
    var got: usize = 0;
    var rounds: usize = 0;
    var partial: usize = 0;
    var emitted: [N]u32 = undefined;
    while (got < N) {
        var r = try dsparkRound(m, gpa, &st_d, tok, N - got - 1);
        defer r.deinit(gpa);
        @memcpy(emitted[got .. got + r.tokens.len], r.tokens);
        got += r.tokens.len;
        rounds += 1;
        if (r.accepted < B and got < N) partial += 1;
        tok = r.next_token;
    }
    try testing.expectEqualSlices(u32, serial[0..N], &emitted);
    try testing.expectEqual(next_s, tok);
    try testing.expectEqual(st_s.n, st_d.n);
    try testing.expect(partial > 0); // the random head's drafts are mostly rejected

    // Every cache a rejected tail touched is back where serial decode left it.
    var ds = try draftRows(m, gpa, &st_s, tok, true);
    defer ds.deinit(gpa);
    var dd = try draftRows(m, gpa, &st_d, tok, true);
    defer dd.deinit(gpa);
    try expectRowsClose("draft after rollbacks", dd.logits, ds.logits, m.vocab);

    // A draft the trunk agrees with in full commits every row.
    var cont: [16]u32 = undefined;
    var st_c = try initState(m, gpa);
    defer st_c.deinit();
    var full_prompt: [64]u32 = undefined;
    @memcpy(full_prompt[0..prompt.len], prompt);
    @memcpy(full_prompt[prompt.len .. prompt.len + N], serial[0..N]);
    const after = try serialGreedy(m, &st_c, full_prompt[0 .. prompt.len + N], B + 1, &cont);
    const ids_full = try gpa.alloc(u32, B + 1);
    ids_full[0] = tok;
    @memcpy(ids_full[1..], cont[1 .. B + 1]);
    var inj: DsparkDraft = .{ .ids = ids_full, .len = B, .logits = try gpa.alloc(f32, 0), .confidence = try gpa.alloc(f32, 0) };
    defer inj.deinit(gpa);
    try testing.expectEqual(tok, cont[0]);
    var pending = try dsparkBeginWith(m, gpa, &st_d, tok, &inj);
    defer pending.deinit();
    var r = try dsparkFinish(m, gpa, &st_d, &pending, B, after);
    defer r.deinit(gpa);
    try testing.expectEqual(st_c.n, st_d.n);
    var dc = try draftRows(m, gpa, &st_c, after, true);
    defer dc.deinit(gpa);
    var dd2 = try draftRows(m, gpa, &st_d, after, true);
    defer dd2.deinit(gpa);
    try expectRowsClose("draft after full accept", dd2.logits, dc.logits, m.vocab);
}

test "dsv41 fixture: full-width copy rounds emit serial greedy and leave serial state (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const m = t.m;
    try testing.expect(m.n_mtp > 0);
    const ids = try t.ids();
    defer gpa.free(ids);
    const prompt = ids[0..32];
    const W = VERIFY_MAX_DRAFTS;
    const N = 3 * (W + 1);
    var serial: [N]u32 = undefined;
    var st_s = try initState(m, gpa);
    defer st_s.deinit();
    const next_s = try serialGreedy(m, &st_s, prompt, N, &serial);

    var st_d = try initState(m, gpa);
    defer st_d.deinit();
    const first = try logitsHost(m, &st_d, prompt, .last);
    var tok = argmaxRow(first);
    gpa.free(first);
    var got: usize = 0;
    var emitted: [N]u32 = undefined;
    // A copy that breaks after `good` drafts, then one the trunk takes whole, then stage rounds.
    for ([_]usize{ 8, W }) |good| {
        const ids_c = try gpa.alloc(u32, W + 1);
        ids_c[0] = tok;
        @memcpy(ids_c[1..], serial[got + 1 .. got + W + 1]);
        if (good < W) ids_c[good + 1] = (ids_c[good + 1] + 1) % @as(u32, @intCast(m.vocab));
        var copy: DsparkDraft = .{ .ids = ids_c, .len = W, .logits = &.{}, .confidence = &.{} };
        defer copy.deinit(gpa);
        var r = try dsparkRoundWith(m, gpa, &st_d, tok, &copy, N);
        defer r.deinit(gpa);
        try testing.expectEqual(@as(u32, @intCast(good)), r.accepted);
        @memcpy(emitted[got .. got + r.tokens.len], r.tokens);
        got += r.tokens.len;
        tok = r.next_token;
    }
    while (got < N) {
        var r = try dsparkRound(m, gpa, &st_d, tok, N - got - 1);
        defer r.deinit(gpa);
        @memcpy(emitted[got .. got + r.tokens.len], r.tokens);
        got += r.tokens.len;
        tok = r.next_token;
    }
    try testing.expectEqualSlices(u32, &serial, &emitted);
    try testing.expectEqual(next_s, tok);
    try testing.expectEqual(st_s.n, st_d.n);
    var ds = try draftRows(m, gpa, &st_s, tok, true);
    defer ds.deinit(gpa);
    var dd = try draftRows(m, gpa, &st_d, tok, true);
    defer dd.deinit(gpa);
    try expectRowsClose("draft after copy rounds", dd.logits, ds.logits, m.vocab);
}

test "dsv41 fixture: verifies cut to any prefix, plain steps between rounds, emit serial greedy (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const m = t.m;
    try testing.expect(m.n_mtp > 0);
    const ids = try t.ids();
    defer gpa.free(ids);
    const prompt = ids[0..32];
    const N = 28;
    var serial: [N + 16]u32 = undefined;
    var st_s = try initState(m, gpa);
    defer st_s.deinit();
    const next_s = try serialGreedy(m, &st_s, prompt, N, &serial);

    var st_d = try initState(m, gpa);
    defer st_d.deinit();
    const first = try logitsHost(m, &st_d, prompt, .last);
    var tok = argmaxRow(first);
    gpa.free(first);
    var got: usize = 0;
    var emitted: [N]u32 = undefined;
    // Drafts verified per round; null = a plain decode step on a GPU token, as the generator's skipped rounds run.
    const plan = [_]?usize{ 0, 2, null, null, 5, 1, null, 3, 0, 4 };
    var k: usize = 0;
    while (got < N) : (k += 1) {
        if (plan[k % plan.len]) |drafts| {
            var d = try dsparkDraft(m, gpa, &st_d, tok);
            defer d.deinit(gpa);
            for (d.confidence) |c| try testing.expect(std.math.isFinite(c));
            d.len = @min(drafts, d.len);
            var r = try dsparkRoundWith(m, gpa, &st_d, tok, &d, N - got - 1);
            defer r.deinit(gpa);
            @memcpy(emitted[got .. got + r.tokens.len], r.tokens);
            got += r.tokens.len;
            tok = r.next_token;
        } else {
            emitted[got] = tok;
            got += 1;
            const v: i32 = @intCast(tok);
            const g = mlx.mlx_array_new_data(&v, &[_]c_int{1}, 1, .int32);
            defer _ = mlx.mlx_array_free(g);
            const l = try extendIds(m, gpa, &st_d, .{ .gpu = g }, .last, null);
            defer _ = mlx.mlx_array_free(l);
            const row = try hostF32(l);
            defer gpa.free(row);
            tok = argmaxRow(row);
        }
    }
    try testing.expectEqualSlices(u32, serial[0..N], &emitted);
    try testing.expectEqual(next_s, tok);
    try testing.expectEqual(st_s.n, st_d.n);
    var ds = try draftRows(m, gpa, &st_s, tok, true);
    defer ds.deinit(gpa);
    var dd = try draftRows(m, gpa, &st_d, tok, true);
    defer dd.deinit(gpa);
    try expectRowsClose("draft after cut verifies and plain steps", dd.logits, ds.logits, m.vocab);
}

test "dsv41 fixture: bf16 activations stay within bf16 noise before routing can flip (DSV41_TINY)" {
    // From the first MoE on, a random tiny model's bf16-rounded router inputs
    // flip near-tied picks, each a large per-row change; layer 0's attention
    // carries the dtype plumbing without that chaos.
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    for ([_][]const u8{ "pipe", "repack" }) |pack| {
        var t = try Tiny.openAct(root, pack, "fixture.safetensors", .bfloat16);
        defer t.deinit();
        const ids = try t.ids();
        defer gpa.free(ids);
        var tr: Trace = .{ .gpa = gpa };
        defer tr.deinit();
        t.m.trace = &tr;
        var st = try initState(t.m, gpa);
        defer st.deinit();
        const logits = try logitsHost(t.m, &st, ids, .all);
        defer gpa.free(logits);
        t.m.trace = null;
        for (logits) |v| try testing.expect(std.math.isFinite(v));
        for (tr.items.items) |it| {
            if (!std.mem.eql(u8, it.name, "attn_0")) continue;
            const want = try t.host(it.name);
            defer gpa.free(want);
            const got = try hostF32(it.arr);
            defer gpa.free(got);
            var dot: f64 = 0;
            var gg: f64 = 0;
            var ww: f64 = 0;
            for (got, want) |a, b| {
                dot += @as(f64, a) * b;
                gg += @as(f64, a) * a;
                ww += @as(f64, b) * b;
            }
            const cos = dot / @sqrt(gg * ww);
            std.debug.print("{s} bf16 {s}: cos {d:.6}\n", .{ pack, it.name, cos });
            // bf16 noise flips a few fp8 window codes.
            try testing.expect(cos > 0.999);
        }
    }
}

test "dsv41 fixture: the QAT round-trips are bit-exact with the reference's, midpoints included (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var fx = try model.loadWeightsFile(gpa, root, "qat.safetensors");
    defer fx.deinit();
    var m: Dsv41Model = undefined;
    m.qat_off = false;
    m.cfg = .{};
    m.cfg.dsv4_rope_head_dim = 0; // no channel roped: the round trips alone
    m.rope_kernel = try buildKernel("dsv41_rope_tail", &.{ "x", "periods", "off", "scal" }, &.{"out"}, ROPE_TAIL_SOURCE, FP8_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.rope_kernel);
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const x = fx.get("x").?;
    const periods = try sc.keep(mlx.mlx_array_new_data(&[_]f32{1}, &[_]c_int{1}, 1, .float32));
    const cases = .{ .{ "fp8_ue8m0_32", Qat.fp8_ue8m0 }, .{ "fp4_ue8m0_32", Qat.fp4_ue8m0 }, .{ "fp4_e4m3_16", Qat.fp4_e4m3 } };
    inline for (cases) |cs| {
        const got = try hostF32(try ropeQat(&sc, &m, x, periods, try sc.i(0), 0, cs[1], null));
        defer gpa.free(got);
        const want = try hostF32(fx.get(cs[0]).?);
        defer gpa.free(want);
        for (got, want) |a, b| try testing.expectEqual(@as(u32, @bitCast(b)), @as(u32, @bitCast(a)));
    }
}

test "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" {
    // DSV41_PPL = a pack dir, DSV41_PPL_TEXT = the corpus text (wikitext-2-raw test,
    // rows joined as stored), DSV41_PPL_HEAD8 serves the head 8-bit as the server
    // does. PipeNetwork's method: windows of 2048 tokens from the start, each scored
    // on its own 2047 next-token predictions.
    const dir = std.mem.span(std.c.getenv("DSV41_PPL") orelse return error.SkipZigTest);
    const text_path = std.mem.span(std.c.getenv("DSV41_PPL_TEXT") orelse return error.SkipZigTest);
    const max_win = if (std.c.getenv("DSV41_PPL_WINDOWS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else std.math.maxInt(usize);
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var cfg = try model.parseConfig(io, gpa, dir);
    defer cfg.deinit(gpa);
    var w = try model.loadModelWeights(io, gpa, dir, &cfg, false);
    defer w.deinit();
    const m = try init(gpa, &cfg, &w, mlx.gpuStream(), .{ .head_8bit = std.c.getenv("DSV41_PPL_HEAD8") != null });
    defer m.deinit();
    var tok = try @import("tokenizer.zig").loadTokenizer(io, gpa, dir);
    defer tok.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    const seq = 2048;
    const n_win = @min(ids.len / seq, max_win);
    std.debug.print("[ppl] {d} tokens, {d} windows x {d}\n", .{ ids.len, n_win, seq });
    var nll_sum: f64 = 0;
    var scored: usize = 0;
    for (0..n_win) |win| {
        const wi = ids[win * seq ..][0..seq];
        var st = try initState(m, gpa);
        defer st.deinit();
        const logits = try extend(m, gpa, &st, wi, .all); // [seq, V] f32
        defer _ = mlx.mlx_array_free(logits);
        var sc = Scope.init(gpa, m.s);
        defer sc.deinit();
        const pred = try sc.slice(logits, 0, 0, seq - 1);
        var lo = mlx.mlx_array_new();
        const lse = try sc.res(mlx.mlx_logsumexp_axis(&lo, pred, -1, false, sc.s), &lo);
        const next = try sc.reshape(try sc.keep(mlx.mlx_array_new_data(wi[1..].ptr, &[_]c_int{seq - 1}, 1, .uint32)), &.{ seq - 1, 1 });
        const picked = try sc.reshape(try sc.takeAlong(pred, next, -1), &.{seq - 1});
        var so = mlx.mlx_array_new();
        const total = try sc.res(mlx.mlx_sum(&so, try sc.sub(lse, picked), false, sc.s), &so);
        try mlx.check(mlx.mlx_array_eval(total));
        const v: f64 = (mlx.mlx_array_data_float32(total) orelse return error.NoData)[0];
        nll_sum += v;
        scored += seq - 1;
        std.debug.print("[ppl] window {d}: nll {d:.4} ppl so far {d:.4}\n", .{ win, v, @exp(nll_sum / @as(f64, @floatFromInt(scored))) });
    }
    std.debug.print("[ppl] perplexity {d:.4} over {d} tokens\n", .{ @exp(nll_sum / @as(f64, @floatFromInt(scored))), scored });
}

/// A layer whose FFN is random banks, routed at `bits` and shared at 8 bits
/// (affine, or mxfp8 as the iQ packs ship it), for the fused FFN's tests.
const FfnFixture = struct {
    sc: Scope,
    m: Dsv41Model,
    ly: Layer,
    dq: [4]A, // f32 dequantized: routed gate+up, routed down, shared gate+up, shared down

    fn init(gpa: std.mem.Allocator, E: c_int, d: c_int, inter: c_int, bits: u32, shared_mode: model.QuantMode) !*FfnFixture {
        const f = try gpa.create(FfnFixture);
        errdefer gpa.destroy(f);
        f.sc = Scope.init(gpa, mlx.gpuStream());
        f.m = undefined;
        f.m.cfg = .{};
        f.m.cfg.dsv4_swiglu_limit = 10;
        f.m.lhs_gu = none;
        f.m.moe_gu_kernel = try buildKernel("dsv41_moe_gu", &MOE_GU_INS, &.{"out"}, MOE_GU_SOURCE, MOE_HEADER);
        f.m.moe_dn_kernel = try buildKernel("dsv41_moe_dn", &MOE_DN_INS, &.{"out"}, MOE_DN_SOURCE, MOE_HEADER);
        f.m.moe_act_kernel = try buildKernel("dsv41_moe_act", &MOE_ACT_INS, &.{"out"}, MOE_ACT_SOURCE, "");
        f.m.moe_combine_kernel = try buildKernel("dsv41_moe_combine", &MOE_COMBINE_INS, &.{"out"}, MOE_COMBINE_SOURCE, "");
        f.m.act = .bfloat16;
        f.ly = undefined;
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, 11));
        const shapes = [_][3]c_int{ .{ E, 2 * inter, d }, .{ E, d, inter }, .{ 1, 2 * inter, d }, .{ 1, d, inter } };
        var q: [4]Lin = undefined;
        for (shapes, 0..) |sh, j| {
            const b: u32 = if (j < 2) bits else 8;
            const mode: model.QuantMode = if (j < 2) .affine else shared_mode;
            const gs: u32 = if (mode == .mxfp8) 32 else 64;
            var w = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w);
            try mlx.check(mlx.mlx_random_normal(&w, &sh, 3, .bfloat16, 0, 0.02, key, f.sc.s));
            var qv = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(qv);
            try mlx.check(mlx.mlx_quantize(&qv, w, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(b)), mode.cstr(), none, f.sc.s));
            var t: [3]A = .{ none, none, none };
            const n = mlx.mlx_vector_array_size(qv);
            for (t[0..n], 0..) |*a, i| {
                a.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(a, qv, i));
                _ = try f.sc.keep(a.*);
                if (j >= 2) a.* = try f.sc.reshape(a.*, mlx.getShape(a.*)[1..]); // the shared linears are 2-D
            }
            try evalAll(t[0..n]);
            q[j] = .{ .w = t[0], .s = t[1], .b = t[2], .qp = .{ .bits = b, .group_size = gs, .mode = mode } };
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_dequantize(&o, t[0], t[1], t[2], mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(b)), mode.cstr(), none, .{ .value = .float32, .has_value = true }, f.sc.s));
            f.dq[j] = try f.sc.keep(o);
        }
        f.ly.experts = .{ .gate = .{ .w = none }, .up = .{ .w = none }, .down = bankOf(q[1]), .gu = bankOf(q[0]) };
        f.ly.shared_gu = q[2];
        f.ly.shared = .{ .{ .w = none }, .{ .w = none }, q[3] };
        return f;
    }

    fn deinit(f: *FfnFixture, gpa: std.mem.Allocator) void {
        _ = mlx.mlx_fast_metal_kernel_free(f.m.moe_gu_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(f.m.moe_dn_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(f.m.moe_act_kernel);
        _ = mlx.mlx_fast_metal_kernel_free(f.m.moe_combine_kernel);
        f.sc.deinit();
        gpa.destroy(f);
    }

    /// The composed path the FFN region takes without the kernels.
    fn composed(f: *FfnFixture, h: A, idx: A, wts: A, x1: A, sp: *const Split) !A {
        return hcExpand(&f.sc, x1, try withShared(&f.sc, &f.m, &f.ly, h, try routed(&f.sc, &f.m, &f.ly, h, idx, wts)), sp);
    }
};

test "dsv41 ffn decode: the two kernels are no further from f32 truth than the composed path, every routed width, affine and mxfp8 shared" {
    const gpa = testing.allocator;
    const k = 6;
    const d = 512;
    const inter = 384;
    for ([_]u32{ 2, 3, 4, 8 }) |bits| for ([_]model.QuantMode{ .affine, .mxfp8 }) |smode| {
        const f = try FfnFixture.init(gpa, 16, d, inter, bits, smode);
        defer f.deinit(gpa);
        const sc = &f.sc;
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, 3));
        var hx = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&hx, &[_]c_int{ 1, d }, 2, .bfloat16, 0, 1, key, sc.s));
        const h = try sc.keep(hx);
        const idx = try sc.keep(mlx.mlx_array_new_data(&[_]u32{ 3, 0, 15, 12, 5, 9 }, &[_]c_int{ 1, k }, 2, .uint32));
        const wts = try sc.keep(mlx.mlx_array_new_data(&[_]f32{ 0.9, 0.4, 0.31, 0.2, 0.12, 0.05 }, &[_]c_int{ 1, k }, 2, .float32));
        // A zero stream with post[0] = 1: stream 0 of the result is the FFN output rounded once.
        const sp: Split = .{
            .pre = none,
            .post = try sc.keep(mlx.mlx_array_new_data(&[_]f32{ 1, 0.5, 2, -1 }, &[_]c_int{ 1, 4 }, 2, .float32)),
            .comb = try sc.keep(mlx.mlx_array_new_data(&[_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 }, &[_]c_int{ 1, 4, 4 }, 3, .float32)),
        };
        const zero = try sc.zeros(&.{ 1, 4, d }, .bfloat16);
        const fused = (try ffnDecode(sc, &f.m, &f.ly, h, idx, wts, zero, &sp)).?;
        const comp = try f.composed(h, idx, wts, zero, &sp);
        // f32 truth over the dequantized weights.
        const flat = try sc.reshape(idx, &.{k});
        const hf = try sc.reshape(try sc.astype(h, .float32), &.{ d, 1 });
        const hu = try sc.reshape(try sc.matmul(try sc.take(f.dq[0], flat, 0), hf), &.{ k, 2 * inter });
        const a32 = try sc.mul(try clampedSwiglu(sc, try sc.slice(hu, 1, 0, inter), try sc.slice(hu, 1, inter, 2 * inter), 10), try sc.reshape(wts, &.{ k, 1 }));
        const routed_t = try sc.sum(try sc.matmul(try sc.take(f.dq[1], flat, 0), try sc.reshape(a32, &.{ k, inter, 1 })), 0, false);
        const su = try sc.reshape(try sc.matmul(f.dq[2], hf), &.{ 1, 2 * inter });
        const s32 = try clampedSwiglu(sc, try sc.slice(su, 1, 0, inter), try sc.slice(su, 1, inter, 2 * inter), 10);
        const truth = try sc.add(try sc.reshape(routed_t, &.{ 1, d }), try sc.matmul(s32, try sc.t2(f.dq[3])));
        const fo = try hostF32(try sc.slice(fused, 1, 0, 1));
        defer gpa.free(fo);
        const co = try hostF32(try sc.slice(comp, 1, 0, 1));
        defer gpa.free(co);
        const t = try hostF32(truth);
        defer gpa.free(t);
        var ef: f64 = 0;
        var ec: f64 = 0;
        var tt: f64 = 0;
        for (fo, co, t) |a, b, r| {
            ef += (a - r) * (a - r);
            ec += (b - r) * (b - r);
            tt += r * r;
        }
        std.debug.print("bits {d}, shared {s}: fused-truth {e:.2}, composed-truth {e:.2} (rel rms)\n", .{ bits, @tagName(smode), @sqrt(ef / tt), @sqrt(ec / tt) });
        try testing.expect(@sqrt(ef / tt) < 1e-2);
        try testing.expect(@sqrt(ef) <= 1.25 * @sqrt(ec) + 1e-4 * @sqrt(tt));
        // The other streams: post and comb exactly as `hcExpand` applies them.
        var sx = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&sx, &[_]c_int{ 1, 4, d }, 3, .bfloat16, 0, 1, key, sc.s));
        const x1 = try sc.keep(sx);
        const y0 = try sc.astype(try sc.slice(fused, 1, 0, 1), .bfloat16); // [1, 1, d]
        const want = try hostF32(try hcExpand(sc, x1, try sc.reshape(y0, &.{ 1, d }), &sp));
        defer gpa.free(want);
        const got = try hostF32((try ffnDecode(sc, &f.m, &f.ly, h, idx, wts, x1, &sp)).?);
        defer gpa.free(got);
        for (got, want) |a, b| try testing.expectEqual(@as(u32, @bitCast(b)), @as(u32, @bitCast(a)));
    };
}

test "dsv41 ffn prefill: the sorted-gather MoE with its activation and combine kernels is no further from f32 truth than the composed path at 2048 rows, every routed width" {
    const gpa = testing.allocator;
    const L = 2048;
    const k = 6;
    const d = 512;
    const inter = 384;
    const E = 16;
    for ([_]u32{ 2, 3, 4, 8 }) |bits| {
        const f = try FfnFixture.init(gpa, E, d, inter, bits, .affine);
        defer f.deinit(gpa);
        const sc = &f.sc;
        var key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(key);
        try mlx.check(mlx.mlx_random_key(&key, 7));
        var hx = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&hx, &[_]c_int{ L, d }, 2, .bfloat16, 0, 1, key, sc.s));
        const h = try sc.keep(hx);
        // Skewed routing (higher experts draw more rows), k distinct experts a row, weights in (0.05, 1.05).
        var rng = std.Random.DefaultPrng.init(bits);
        var ids: [L * k]u32 = undefined;
        var ws: [L * k]f32 = undefined;
        for (0..L) |l| {
            var used: [E]bool = @splat(false);
            var j: usize = 0;
            while (j < k) {
                const e: usize = @min(E - 1, @as(usize, @intFromFloat(@sqrt(rng.random().float(f32)) * E)));
                if (used[e]) continue;
                used[e] = true;
                ids[l * k + j] = @intCast(e);
                ws[l * k + j] = 0.05 + rng.random().float(f32);
                j += 1;
            }
        }
        const idx = try sc.keep(mlx.mlx_array_new_data(&ids, &[_]c_int{ L, k }, 2, .uint32));
        const wts = try sc.keep(mlx.mlx_array_new_data(&ws, &[_]c_int{ L, k }, 2, .float32));
        const fused = try moePrefill(sc, &f.m, &f.ly, h, idx, wts, try shared(sc, &f.m, &f.ly, h));
        const comp = try withShared(sc, &f.m, &f.ly, h, try routed(sc, &f.m, &f.ly, h, idx, wts));
        // f32 truth over the dequantized weights: every expert on every row, the picks gathered.
        const hf = try sc.astype(h, .float32);
        const hu = try sc.matmul(hf, try sc.transpose(f.dq[0], &.{ 0, 2, 1 })); // [E, L, 2 inter]
        const a32 = try clampedSwiglu(sc, try sc.slice(hu, -1, 0, inter), try sc.slice(hu, -1, inter, 2 * inter), 10);
        const dn = try sc.transpose(try sc.matmul(a32, try sc.transpose(f.dq[1], &.{ 0, 2, 1 })), &.{ 1, 0, 2 }); // [L, E, d]
        const picked = try sc.takeAlong(dn, try sc.reshape(idx, &.{ L, k, 1 }), 1); // [L, k, d]
        const routed_t = try sc.sum(try sc.mul(picked, try sc.reshape(wts, &.{ L, k, 1 })), 1, false);
        const su = try sc.matmul(hf, try sc.t2(f.dq[2]));
        const s32 = try clampedSwiglu(sc, try sc.slice(su, -1, 0, inter), try sc.slice(su, -1, inter, 2 * inter), 10);
        const truth = try sc.add(routed_t, try sc.matmul(s32, try sc.t2(f.dq[3])));
        const fo = try hostF32(fused);
        defer gpa.free(fo);
        const co = try hostF32(comp);
        defer gpa.free(co);
        const t = try hostF32(truth);
        defer gpa.free(t);
        var ef: f64 = 0;
        var ec: f64 = 0;
        var tt: f64 = 0;
        for (fo, co, t) |a, b, r| {
            ef += (a - r) * (a - r);
            ec += (b - r) * (b - r);
            tt += r * r;
        }
        std.debug.print("bits {d}: prefill-truth {e:.2}, composed-truth {e:.2} (rel rms), nax {any}\n", .{ bits, @sqrt(ef / tt), @sqrt(ec / tt), transformer.verifyQmmNaxAvailable() });
        try testing.expect(@sqrt(ef / tt) < 1e-2);
        try testing.expect(@sqrt(ef) <= 1.25 * @sqrt(ec) + 1e-4 * @sqrt(tt));
    }
}

test "dsv41 bench: one-token FFN as a dependent chain, composed vs kernels (DSV41_BENCH_MOEK)" {
    // DSV41_BENCH_MOEK = "<gate/up bits><down bits>[m]", e.g. 23; the shared expert is 8-bit, mxfp8 with `m`.
    const spec = std.mem.span(std.c.getenv("DSV41_BENCH_MOEK") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("io_util.zig");
    const E = 384;
    const d = 5120;
    const inter = 2304;
    const k = 6;
    const wb = [2]u32{ if (spec.len >= 2) spec[0] - '0' else 2, if (spec.len >= 2) spec[1] - '0' else 3 };
    const smode: model.QuantMode = if (std.mem.endsWith(u8, spec, "m")) .mxfp8 else .affine;
    const f = try FfnFixture.init(gpa, E, d, inter, wb[0], smode);
    defer f.deinit(gpa);
    if (wb[1] != wb[0]) {
        const g = try FfnFixture.init(gpa, E, d, inter, wb[1], smode);
        defer g.deinit(gpa);
        f.ly.experts.down = g.ly.experts.down;
        for ([_]A{ g.ly.experts.down.w, g.ly.experts.down.s, g.ly.experts.down.b }) |a| {
            var o = mlx.mlx_array_new();
            _ = mlx.mlx_array_set(&o, a);
            _ = try f.sc.keep(o);
        }
    }
    const sc = &f.sc;
    var rng = std.Random.DefaultPrng.init(3);
    const wts = try sc.keep(mlx.mlx_array_new_data(&[_]f32{ 0.3, 0.2, 0.2, 0.1, 0.1, 0.1 }, &[_]c_int{ 1, k }, 2, .float32));
    const sp: Split = .{
        .pre = none,
        .post = try sc.keep(mlx.mlx_array_new_data(&[_]f32{ 1, 0.5, 0.5, 0.5 }, &[_]c_int{ 1, 4 }, 2, .float32)),
        .comb = try sc.keep(mlx.mlx_array_new_data(&[_]f32{ 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25 }, &[_]c_int{ 1, 4, 4 }, 3, .float32)),
    };
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    const steps = 200;
    for (0..2) |arm| for (0..3) |_| {
        var x0 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&x0, &[_]c_int{ 1, 4, d }, 3, .bfloat16, 0, 1, key, sc.s));
        var x = try sc.keep(x0);
        try evalAll(&.{x});
        var ids: [steps][k]u32 = undefined;
        for (&ids) |*r| for (r) |*v| {
            v.* = rng.random().uintLessThan(u32, E);
        };
        const t0 = iou.nowUsMonotonic(io);
        for (&ids) |*r| {
            const idx = try sc.keep(mlx.mlx_array_new_data(r, &[_]c_int{ 1, k }, 2, .uint32));
            const h = try sc.reshape(try sc.slice(x, 1, 0, 1), &.{ 1, d });
            x = if (arm == 1) (try ffnDecode(sc, &f.m, &f.ly, h, idx, wts, x, &sp)).? else try f.composed(h, idx, wts, x, &sp);
        }
        try evalAll(&.{x});
        const us = @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / steps;
        std.debug.print("[moek] {s} {d}/{d}-bit + {s} shared: {d:.1} us/token-layer\n", .{ if (arm == 1) "kernels " else "composed", wb[0], wb[1], @tagName(smode), us });
    };
}

test "dsv41 topk: the kernel picks the k largest of each row in index order, ties to the lower index" {
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    m.topk_kernel = try buildKernel("dsv41_topk", &.{"x"}, &.{"out"}, TOPK_SOURCE, TOPK_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.topk_kernel);
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    for ([_][2]usize{ .{ 3, 3 }, .{ 100, 7 }, .{ 4100, 512 }, .{ 8192, 512 }, .{ 70000, 512 }, .{ 600, 600 } }) |nk| {
        const n, const k = nk;
        const rows = 2;
        const v = try gpa.alloc(f32, rows * n);
        defer gpa.free(v);
        // Row 0 distinct values with some -inf; row 1 coarse values, so many ties.
        for (v[0..n], 0..) |*x, i| x.* = if (i % 13 == 4) -std.math.inf(f32) else r.float(f32) * 2 - 1;
        for (v[n..], 0..) |*x, i| x.* = @floatFromInt(@as(i32, @intCast((i * 7919) % 5)) - 2);
        const x = try sc.keep(mlx.mlx_array_new_data(v.ptr, &[_]c_int{ rows, ci(n) }, 2, .float32));
        const got = try hostI32(try topkRows(&sc, &m, x, k));
        defer gpa.free(got);
        for (0..rows) |row| {
            const vals = v[row * n ..][0..n];
            // The reference: indices sorted by value descending, then index ascending; its first k, in index order.
            const order = try gpa.alloc(u32, n);
            defer gpa.free(order);
            for (order, 0..) |*o, i| o.* = @intCast(i);
            std.mem.sort(u32, order, vals, struct {
                fn lt(c: []const f32, a: u32, b: u32) bool {
                    return c[a] > c[b] or (c[a] == c[b] and a < b);
                }
            }.lt);
            const want = order[0..k];
            std.mem.sort(u32, want, {}, std.sort.asc(u32));
            for (want, got[row * k ..][0..k]) |w, g| try testing.expectEqual(@as(i32, @intCast(w)), g);
        }
    }
}

test "dsv41 index scores: the matrix kernel scores what the scalar kernel scores and picks its blocks, near-ties aside" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    m.idx_score_kernel = try buildIdxScoreKernel();
    defer _ = mlx.mlx_fast_metal_kernel_free(m.idx_score_kernel);
    m.idx_tile_kernel = try buildKernel("dsv41_index_tile", &IDX_INS, &.{"scores"}, IDX_TILE_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.idx_tile_kernel);
    m.topk_kernel = try buildKernel("dsv41_topk", &.{"x"}, &.{"out"}, TOPK_SOURCE, TOPK_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.topk_kernel);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    // The release's indexer geometry and the tiny fixture's; keys bf16 and f32; nb off the tile
    // grid, a query that sees one tile only, a start inside a tile.
    for ([_][2]c_int{ .{ 32, 128 }, .{ 8, 32 } }) |g| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |kdt| for ([_][3]usize{ .{ 9, 5, 0 }, .{ 70, 130, 37 }, .{ 40, 700, 1300 } }) |c| {
        const ih, const ihd = g;
        const L, const nb, const p0 = c;
        const ratio: usize = 2;
        var t: [3]A = undefined;
        for ([_][]const c_int{ &.{ ci(L), ih, ihd }, &.{ ci(nb), ihd }, &.{ ci(L), ih } }, &t, 0..) |sh, *o, i| {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&r, sh.ptr, sh.len, if (i == 2) .float32 else kdt, 0, 1, key, sc.s));
            o.* = try sc.keep(r);
        }
        // q as the rope kernel hands it over: rounded to the keys' dtype, widened to f32.
        const q = try sc.astype(t[0], .float32);
        const ws: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(ihd * ih)));
        const want_a = try indexScoresScalar(&sc, &m, q, t[1], t[2], ws, p0, ratio);
        const got_a = try indexScoresTiled(&sc, &m, q, t[1], t[2], ws, p0, ratio);
        // The folded weight scale is the product the indexer used to make first. Not bit for bit:
        // whether the kernel contracts `float(w) * scale * q` into an FMA is the Metal compiler's
        // call per GPU (bit-equal on M5, one rounding apart on the macos-26 CI runner).
        const wb = try sc.astype(t[2], .bfloat16);
        for ([2]bool{ false, true }) |tiled| {
            const f = if (tiled) &indexScoresTiled else &indexScoresScalar;
            const pre = try hostF32(try f(&sc, &m, q, t[1], try sc.mul(try sc.astype(wb, .float32), try sc.f(ws)), 1.0, p0, ratio));
            defer gpa.free(pre);
            const fold = try hostF32(try f(&sc, &m, q, t[1], wb, ws, p0, ratio));
            defer gpa.free(fold);
            for (pre, fold) |a, b| {
                if (std.math.isInf(a)) try testing.expectEqual(a, b) else try testing.expectApproxEqAbs(a, b, 1e-5 * @max(1.0, @abs(a)));
            }
        }
        const want = try hostF32(want_a);
        defer gpa.free(want);
        const got = try hostF32(got_a);
        defer gpa.free(got);
        try testing.expectEqual(L * nb, got.len);
        for (want, got) |w, v| {
            if (std.math.isInf(w)) {
                try testing.expectEqual(w, v);
            } else {
                try testing.expectApproxEqAbs(w, v, 1e-4 * @max(1.0, @abs(w)));
            }
        }
        // Each kernel's picks sit within tolerance of the other's k-th best score: the picks are
        // identical except where the k-th place is a near-tie.
        const sorted = try gpa.alloc(f32, nb);
        defer gpa.free(sorted);
        for ([_]usize{ 4, 512 }) |kk| {
            const k = @min(kk, nb);
            const pw = try hostI32(try topkRows(&sc, &m, want_a, k));
            defer gpa.free(pw);
            const pg = try hostI32(try topkRows(&sc, &m, got_a, k));
            defer gpa.free(pg);
            for ([_][]const f32{ want, got }, [_][]const i32{ pg, pw }) |judge, picks| {
                for (0..L) |l| {
                    @memcpy(sorted, judge[l * nb ..][0..nb]);
                    std.mem.sort(f32, sorted, {}, std.sort.desc(f32));
                    const kth = sorted[k - 1];
                    const tol = 1e-4 * @max(1.0, @abs(kth));
                    for (picks[l * k ..][0..k]) |p| try testing.expect(judge[l * nb + @as(usize, @intCast(p))] >= kth - tol);
                }
            }
        }
    };
}

test "dsv41 decode attention: rows split across threadgroups give the one-threadgroup heads" {
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.sliding_window = 128;
    m.act = .bfloat16;
    m.attn_dec_kernel = try buildKernel("dsv41_attn_decode", &ATTN_DEC_INS, &.{"out"}, ATTN_DEC_SOURCE, UNROPE_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_dec_kernel);
    m.attn_split_kernel = try buildKernel("dsv41_attn_split", &ATTN_INS, &.{ "pacc", "pm", "pl" }, ATTN_SPLIT_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_split_kernel);
    m.attn_merge_kernel = try buildKernel("dsv41_attn_merge", &ATTN_MERGE_INS, &.{"out"}, ATTN_MERGE_SOURCE, UNROPE_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_merge_kernel);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 4));
    var ly: Layer = undefined;
    var t: [4]A = undefined;
    for ([_][]const c_int{ &.{8}, &.{ 144, 512 }, &.{ 700, 512 }, &.{ 2, 8, 512 } }, &t) |sh, *o| {
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&r, sh.ptr, sh.len, if (sh.len == 1) .float32 else .bfloat16, 0, 1, key, sc.s));
        o.* = try sc.keep(r);
    }
    ly.sink = t[0];
    const win: Ring = .{ .buf = t[1], .cap = 144, .width = 512 };
    // 300 selected rows, a few of them unset (-1) or past the cache.
    var ids: [2 * 300]i32 = undefined;
    for (&ids, 0..) |*v, i| v.* = if (i % 37 == 5) -1 else if (i % 41 == 7) 900 else @intCast((i * 13) % 700);
    const topk = try sc.ints(&ids, &.{ 2, 300 });
    const one = try hostF32(try attnDecodeSplit(&sc, &m, &ly, t[3], win, t[2], topk, 500, 0.05, 1, null));
    defer gpa.free(one);
    for ([_]c_int{ 2, 4 }) |splits| {
        const got = try hostF32(try attnDecodeSplit(&sc, &m, &ly, t[3], win, t[2], topk, 500, 0.05, splits, null));
        defer gpa.free(got);
        for (got, one) |a, b| try testing.expectApproxEqAbs(b, a, 1e-2 * @max(1.0, @abs(b)));
    }
}

test "dsv41 decode attention: the folded inverse rope is ropeTail of the heads, bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.sliding_window = 128;
    m.cfg.dsv4_rope_head_dim = 64;
    m.qat_off = true;
    m.attn_dec_kernel = try buildKernel("dsv41_attn_decode", &ATTN_DEC_INS, &.{"out"}, ATTN_DEC_SOURCE, UNROPE_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_dec_kernel);
    m.attn_split_kernel = try buildKernel("dsv41_attn_split", &ATTN_INS, &.{ "pacc", "pm", "pl" }, ATTN_SPLIT_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_split_kernel);
    m.attn_merge_kernel = try buildKernel("dsv41_attn_merge", &ATTN_MERGE_INS, &.{"out"}, ATTN_MERGE_SOURCE, UNROPE_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_merge_kernel);
    m.rope_kernel = try buildKernel("dsv41_rope_tail", &.{ "x", "periods", "off", "scal" }, &.{"out"}, ROPE_TAIL_SOURCE, FP8_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.rope_kernel);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    var per: [32]f32 = undefined;
    for (&per, 0..) |*v, k| v.* = -std.math.pow(f32, 10000, @as(f32, @floatFromInt(k)) / 32);
    const inv = try sc.keep(mlx.mlx_array_new_data(&per, &[_]c_int{32}, 1, .float32));
    var ids: [3 * 300]i32 = undefined;
    for (&ids, 0..) |*v, i| v.* = if (i % 37 == 5) -1 else @intCast((i * 13) % 700);
    const topk = try sc.ints(&ids, &.{ 3, 300 });
    for ([_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 }) |act| {
        m.act = act;
        var ly: Layer = undefined;
        var t: [4]A = undefined;
        for ([_][]const c_int{ &.{8}, &.{ 144, 512 }, &.{ 700, 512 }, &.{ 3, 8, 512 } }, &t) |sh, *o| {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&r, sh.ptr, sh.len, if (sh.len == 1) .float32 else act, 0, 1, key, sc.s));
            o.* = try sc.keep(r);
        }
        ly.sink = t[0];
        const win: Ring = .{ .buf = t[1], .cap = 144, .width = 512 };
        // Splits 1 = the one-threadgroup kernel; 2 and 4 = the merge.
        for ([_]c_int{ 1, 2, 4 }) |splits| {
            const p0 = 3001;
            const heads = try attnDecodeSplit(&sc, &m, &ly, t[3], win, t[2], topk, p0, 0.05, splits, null);
            const want = try hostF32(try ropeTail(&sc, &m, heads, inv, try sc.i(p0), 1.0));
            defer gpa.free(want);
            const got = try hostF32(try attnDecodeSplit(&sc, &m, &ly, t[3], win, t[2], topk, p0, 0.05, splits, inv));
            defer gpa.free(got);
            for (got, want) |a, b| try testing.expectEqual(@as(u32, @bitCast(b)), @as(u32, @bitCast(a)));
        }
    }
}

test "dsv41 engram: the decode-width kernel writes the op chain's stream bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.dsv4_hc_mult = 4;
    m.engram_kernel = try buildKernel("dsv41_engram", &ENGRAM_INS, &.{"out"}, ENGRAM_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.engram_kernel);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 7));
    // Each row-reduce width MLX picks (1, 32, 128, 1024 threads); 4 to 32 rows, where its sum switches kernel.
    for ([_]usize{ 64, 128, 1024, 5120 }) |d| for ([_]usize{ 1, 3, 8 }) |L| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |xdt| {
        m.cfg.hidden_size = @intCast(d);
        m.act = if (xdt == .float32) .float16 else .bfloat16;
        var t: [3]A = undefined;
        for ([_][]const c_int{ &.{ ci(L), 4, ci(d) }, &.{ ci(L), ci(5 * d) }, &.{ 4, ci(d) } }, [_]mlx.mlx_dtype{ xdt, m.act, .float32 }, &t) |sh, dt, *o| {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&r, sh.ptr, sh.len, dt, 0, 1, key, sc.s));
            o.* = try sc.keep(r);
        }
        const want = try hostF32(try engramGateOps(&sc, &m, t[2], t[0], t[1]));
        defer gpa.free(want);
        const got = try hostF32(try engramGate(&sc, &m, t[2], t[0], t[1]));
        defer gpa.free(got);
        for (want, got) |a, b| try testing.expectEqual(@as(u32, @bitCast(a)), @as(u32, @bitCast(b)));
    };
}

test "dsv41 wide attention: the f16 core gives the f32 chain's heads at the release's span shape" {
    // Bar: per (row, head) cosine, norm ratio and peak error against the f32 chain, both read in f32.
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.sliding_window = 128;
    m.act = .bfloat16;
    m.attn_softmax_kernel = try buildKernel("dsv41_attn_softmax", &ATTN_SOFTMAX_INS, &.{ "pw", "pc" }, ATTN_SOFTMAX_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_softmax_kernel);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 11));
    const Case = struct { L: usize, H: usize, D: usize, wp: usize, nc: usize, K: usize, kw: f32 = 1 };
    for ([_]Case{
        .{ .L = 256, .H = 64, .D = 512, .wp = 127, .nc = 4096, .K = 512 }, // a full span
        .{ .L = 256, .H = 64, .D = 512, .wp = 0, .nc = 0, .K = 0 }, // the first span, window only
        .{ .L = 5, .H = 6, .D = 64, .wp = 3, .nc = 40, .K = 3 }, // odd grids
        .{ .L = 64, .H = 8, .D = 128, .wp = 16, .nc = 200, .K = 16, .kw = 8 }, // keys 8x wider than the norm leaves
    }) |cs| {
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(cs.D)));
        var t: [4]A = undefined;
        // Sink f32; queries 4x wider than the keys so the scaled logits spread like attention's (std ~4).
        const shapes = [_][]const c_int{ &.{ 1, ci(cs.H), 1 }, &.{ ci(cs.L), ci(cs.H), ci(cs.D) }, &.{ ci(cs.wp + cs.L), ci(cs.D) }, &.{ ci(@max(cs.nc, 1)), ci(cs.D) } };
        const widths = [_]f32{ 1, 4, cs.kw, cs.kw };
        for (shapes, widths, &t, 0..) |sh, width, *o, k| {
            var r = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&r, sh.ptr, sh.len, if (k == 0) .float32 else .bfloat16, 0, width, key, sc.s));
            o.* = try sc.keep(r);
        }
        const sink = t[0];
        const q = t[1];
        const keys_w = t[2];
        const comp = if (cs.K > 0) t[3] else none;
        var topk = none;
        if (cs.K > 0) {
            const ids = try gpa.alloc(i32, cs.L * cs.K);
            defer gpa.free(ids);
            for (ids, 0..) |*v, i| v.* = if (i % 37 == 5) -1 else @intCast((i * 13) % cs.nc);
            topk = try sc.ints(ids, &.{ ci(cs.L), ci(cs.K) });
        }
        const band = try windowBand(&sc, cs.wp, cs.L, m.cfg.sliding_window);
        const want = try hostF32(try attnWideF32(&sc, sink, q, keys_w, band, comp, topk, scale, .float32));
        defer gpa.free(want);
        const got = try hostF32(try attnWide(&sc, &m, sink, q, keys_w, cs.wp, comp, topk, scale, .float32));
        defer gpa.free(got);
        for (got) |v| try testing.expect(std.math.isFinite(v));
        var peak: f64 = 0;
        for (want) |v| peak = @max(peak, @abs(v));
        var worst_cos: f64 = 1;
        var worst_ratio: f64 = 1;
        var worst_err: f64 = 0;
        for (0..cs.L * cs.H) |r| {
            const g = got[r * cs.D ..][0..cs.D];
            const w = want[r * cs.D ..][0..cs.D];
            var dot: f64 = 0;
            var gg: f64 = 0;
            var ww: f64 = 0;
            for (g, w) |a, b| {
                dot += @as(f64, a) * b;
                gg += @as(f64, a) * a;
                ww += @as(f64, b) * b;
                worst_err = @max(worst_err, @abs(@as(f64, a) - b) / peak);
            }
            worst_cos = @min(worst_cos, dot / @sqrt(gg * ww));
            const ratio = @sqrt(gg / ww);
            if (@abs(ratio - 1) > @abs(worst_ratio - 1)) worst_ratio = ratio;
        }
        std.debug.print("wide attention L={d} H={d} K={d}: min cos {d:.7} rms ratio {d:.6} max err/peak {e:.2}\n", .{ cs.L, cs.H, cs.K, worst_cos, worst_ratio, worst_err });
        const wide_keys = cs.kw > 1;
        try testing.expect(worst_cos > @as(f64, if (wide_keys) 0.9995 else 0.9999));
        try testing.expect(@abs(worst_ratio - 1) < @as(f64, if (wide_keys) 2e-2 else 5e-3));
        try testing.expect(worst_err < @as(f64, if (wide_keys) 2e-2 else 1e-2));
    }
}

test "dsv41 bench: one-token decode attention as a dependent chain (DSV41_BENCH_ATTN)" {
    if (std.c.getenv("DSV41_BENCH_ATTN") == null) return error.SkipZigTest;
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("io_util.zig");
    var sc = Scope.init(gpa, s);
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.sliding_window = 128;
    m.act = .bfloat16;
    m.attn_dec_kernel = try buildKernel("dsv41_attn_decode", &ATTN_DEC_INS, &.{"out"}, ATTN_DEC_SOURCE, UNROPE_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_dec_kernel);
    m.attn_split_kernel = try buildKernel("dsv41_attn_split", &ATTN_INS, &.{ "pacc", "pm", "pl" }, ATTN_SPLIT_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_split_kernel);
    m.attn_merge_kernel = try buildKernel("dsv41_attn_merge", &ATTN_MERGE_INS, &.{"out"}, ATTN_MERGE_SOURCE, UNROPE_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.attn_merge_kernel);
    var ly: Layer = undefined;
    ly.sink = try sc.zeros(&.{64}, .float32);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 9));
    const rnd = struct {
        fn f(sc_: *Scope, key_: A, shape: []const c_int) !A {
            var o = mlx.mlx_array_new();
            return sc_.res(mlx.mlx_random_normal(&o, shape.ptr, shape.len, .bfloat16, 0, 1, key_, sc_.s), &o);
        }
    };
    const win: Ring = .{ .buf = try rnd.f(&sc, key, &.{ 144, 512 }), .cap = 144, .width = 512 };
    for ([_]usize{ 0, 512 }) |k| {
        const nc: usize = 8192;
        const comp = try rnd.f(&sc, key, &.{ @intCast(nc), 512 });
        var ids: [512]i32 = undefined;
        for (&ids, 0..) |*v, i| v.* = @intCast((i * 13) % nc);
        const topk = if (k > 0) try sc.ints(ids[0..k], &.{ 1, @intCast(k) }) else none;
        var q = try rnd.f(&sc, key, &.{ 1, 64, 512 });
        try evalAll(&.{ q, win.buf, comp });
        const steps = 300;
        var best: [2]f64 = .{ std.math.inf(f64), std.math.inf(f64) };
        for (0..3) |_| {
            const t0 = iou.nowUsMonotonic(io);
            for (0..steps) |_| q = try attnDecode(&sc, &m, &ly, q, win, if (k > 0) comp else none, topk, 4096, 0.04, null);
            const t1 = iou.nowUsMonotonic(io);
            try evalAll(&.{q});
            best[0] = @min(best[0], @as(f64, @floatFromInt(t1 - t0)) / steps);
            best[1] = @min(best[1], @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t1)) / steps);
        }
        std.debug.print("[attn] window 128 + {d} selected rows: build {d:.1} us/call, eval {d:.1} us/call\n", .{ k, best[0], best[1] });
    }
}

test "dsv41 bench: one hyper-connection stage as a dependent chain (DSV41_BENCH_HC = rows)" {
    const rows_s = std.mem.span(std.c.getenv("DSV41_BENCH_HC") orelse return error.SkipZigTest);
    const L: c_int = try std.fmt.parseInt(c_int, rows_s, 10);
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const iou = @import("io_util.zig");
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    try hcTestModel(&m);
    defer hcTestModelFree(&m);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 3));
    var t: [8]A = undefined;
    for ([_][]const c_int{ &.{ L, 4, 5120 }, &.{ 24, 20480 }, &.{24}, &.{24}, &.{5120}, &.{ L, 5120 }, &.{ L, 4 }, &.{ L, 4, 4 } }, &t, 0..) |sh, *o, i| {
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&r, sh.ptr, sh.len, if (i == 0 or i == 4 or i == 5) .bfloat16 else .float32, 0, if (i == 1) 0.01 else 1, key, sc.s));
        o.* = try sc.keep(r);
    }
    const hc: Hc = .{ .fn_w = t[1], .scale = t[2], .base = t[3] };
    var x = t[0];
    var pre = try onehotPre(&sc, &m, @intCast(L));
    try evalAll(&t);
    var best: f64 = std.math.inf(f64);
    for (0..3) |_| {
        const t0 = iou.nowUsMonotonic(io);
        for (0..200) |_| {
            const r = try hcExpandPreNorm(&sc, &m, x, t[5], &.{ .pre = t[6], .post = t[6], .comb = t[7] }, pre, &hc, t[4]);
            pre = r.sp.pre;
            x = try sc.add(r.x1, try sc.mul(try sc.expand(r.h, 1), try sc.f(1e-3)));
        }
        try evalAll(&.{x});
        best = @min(best, @as(f64, @floatFromInt(iou.nowUsMonotonic(io) - t0)) / 200);
    }
    std.debug.print("[hc] {d} rows: expand+mix+norm+add chain {d:.1} us/call\n", .{ L, best });
}

/// A model with only the hc kernels built, at the release's widths.
fn hcTestModel(m: *Dsv41Model) !void {
    m.cfg = .{};
    m.cfg.hidden_size = 5120;
    m.cfg.dsv4_hc_mult = 4;
    m.cfg.dsv4_hc_sinkhorn_iters = 20;
    m.s = mlx.gpuStream();
    m.act = .bfloat16;
    m.hc_tg = HC_TG;
    m.hc_mix_kernel = try buildKernel("dsv41_hc_mix", &.{ "x", "fn", "scale", "base", "eps" }, &.{"z"}, HC_MIX_SOURCE, "");
    m.hc_norm_kernel = try buildKernel("dsv41_hc_norm", &.{ "z", "x", "cpre", "w", "eps" }, &.{ "pre", "post", "comb", "h_out" }, HC_NORM_SOURCE, "");
    m.hc_stage_kernel = try buildKernel("dsv41_hc_stage", &HC_STAGE_INS, &HC_STAGE_OUTS, HC_STAGE_SOURCE, "");
    m.hc_stage_x_kernel = try buildKernel("dsv41_hc_stage_x", &HC_STAGE_X_INS, &HC_STAGE_X_OUTS, HC_STAGE_X_SOURCE, "");
    m.hc_split_kernel = try buildKernel("dsv41_hc_split", &.{ "z", "eps" }, &.{ "pre", "post", "comb" }, HC_SPLIT_SOURCE, "");
}

fn hcTestModelFree(m: *Dsv41Model) void {
    for ([_]mlx.mlx_fast_metal_kernel{ m.hc_mix_kernel, m.hc_norm_kernel, m.hc_stage_kernel, m.hc_stage_x_kernel, m.hc_split_kernel }) |k| _ = mlx.mlx_fast_metal_kernel_free(k);
}

test "dsv41 state bytes: the window rings and the compressed rows as allocated" {
    const gpa = testing.allocator;
    const s = mlx.gpuStream();
    const layers = try gpa.alloc(LState, 2);
    defer gpa.free(layers);
    layers[0] = .{ .win = try Ring.init(144, 512, .bfloat16, s) };
    layers[1] = .{ .win = try Ring.init(144, 512, .bfloat16, s), .comp = .{ .width = 512, .dtype = .bfloat16 } };
    var st: State = .{ .gpa = gpa, .layers = layers };
    defer for (layers) |*l| l.deinit();
    try testing.expectEqual(@as(u64, 2 * 144 * 512 * 2), stateBytes(&st));
    var row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(row);
    try mlx.check(mlx.mlx_zeros(&row, &[_]c_int{ 1, 512 }, 2, .bfloat16, s));
    try layers[1].comp.?.append(row, s); // the first append allocates 64 rows of capacity
    try testing.expectEqual(@as(u64, 2 * 144 * 512 * 2 + 64 * 512 * 2), stateBytes(&st));
}

test "dsv41 router: the wide f32 matmul gives the logits the decode kernel gives" {
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    m.gate_kernel = try buildKernel("dsv41_gate", &.{ "x", "w" }, &.{"logits"}, GATE_SOURCE, "");
    defer _ = mlx.mlx_fast_metal_kernel_free(m.gate_kernel);
    const E = 384;
    const D = 5120;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 5));
    var xr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&xr, &[_]c_int{ 40, D }, 2, .bfloat16, 0, 1, key, sc.s));
    const x = try sc.keep(xr);
    var wr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&wr, &[_]c_int{ E, D }, 2, .bfloat16, 0, 0.02, key, sc.s));
    var ly: Layer = undefined;
    ly.gate_w = try sc.keep(wr);
    const wide = try hostF32(try gateLogits(&sc, &m, &ly, x));
    defer gpa.free(wide);
    var narrow: [E * 8]f32 = undefined;
    var worst: f64 = 0;
    var peak: f64 = 0;
    for (0..5) |blk| {
        const rows = try sc.slice(x, 0, @intCast(blk * 8), @intCast(blk * 8 + 8));
        const got = try hostF32(try gateLogits(&sc, &m, &ly, rows));
        defer gpa.free(got);
        @memcpy(&narrow, got);
        for (narrow, wide[blk * 8 * E ..][0 .. 8 * E]) |a, b| {
            peak = @max(peak, @abs(@as(f64, a)));
            worst = @max(worst, @abs(@as(f64, a) - b));
        }
    }
    try testing.expect(worst / peak < 1e-5);
}

test "dsv41 kv row: a head-width row launches and is normed by its own rms on this GPU" {
    if (!mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    m.cfg = .{};
    m.cfg.rms_norm_eps = 1e-6;
    m.cfg.dsv4_rope_head_dim = 64;
    m.qat_off = true;
    m.kv_row_kernel = try buildKernel("dsv41_kv_row", &.{ "x", "w", "periods", "off", "eps" }, &.{"out"}, KV_ROW_SOURCE, FP8_HEADER);
    defer _ = mlx.mlx_fast_metal_kernel_free(m.kv_row_kernel);
    const L = 3;
    const D = 512;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 11));
    var xr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_normal(&xr, &[_]c_int{ L, D }, 2, .bfloat16, 0, 1, key, sc.s));
    const x = try sc.keep(xr);
    var ly: Layer = undefined;
    ly.kv_norm = try sc.keep(mlx.mlx_array_new_data(&@as([D]f32, @splat(1)), &[_]c_int{D}, 1, .float32));
    ly.kv_norm = try sc.astype(ly.kv_norm, .bfloat16);
    const periods = try sc.keep(mlx.mlx_array_new_data(&@as([32]f32, @splat(1)), &[_]c_int{32}, 1, .float32));
    const got = try hostF32(try kvRow(&sc, &m, &ly, periods, x, try sc.i(0)));
    defer gpa.free(got);
    const xs = try hostF32(x);
    defer gpa.free(xs);
    // Columns before the rope tail are the row times 1/rms.
    for (0..L) |l| {
        var ss: f32 = 0;
        for (xs[l * D ..][0..D]) |v| ss += v * v;
        const inv = 1.0 / @sqrt(ss / D + 1e-6);
        for (0..D - 64) |d| try testing.expectApproxEqAbs(xs[l * D + d] * inv, got[l * D + d], 1e-2 * @max(1.0, @abs(xs[l * D + d] * inv)));
    }
}

test "dsv41 hc: a threadgroup the GPU refuses is halved until the kernels launch" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    var m: Dsv41Model = undefined;
    try hcTestModel(&m);
    defer hcTestModelFree(&m);
    m.hc_tg = 4 * HC_TG; // above any Metal cap
    try probeHcThreads(testing.allocator, &m);
    try testing.expect(m.hc_tg >= 32 and m.hc_tg <= HC_TG and m.hc_tg % 32 == 0);
    const granted = m.hc_tg;
    try probeHcThreads(testing.allocator, &m);
    try testing.expectEqual(granted, m.hc_tg);
}

test "dsv41 hc: a decode-width stage gives the wide path's mixes, split, norm and expanded stream bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const gpa = testing.allocator;
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    var m: Dsv41Model = undefined;
    try hcTestModel(&m);
    defer hcTestModelFree(&m);
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 9));
    try probeHcThreads(gpa, &m);
    // The width the GPU grants and a narrower one: the stage and the wide kernels agree at both.
    const granted = m.hc_tg;
    for ([_]usize{ granted, 256 }) |tg| {
        m.hc_tg = tg;
        for ([_]c_int{ 1, 5, 8, 17, 100, 300, 1027 }) |L| {
            var t: [8]A = undefined;
            for ([_][]const c_int{ &.{ L, 4, 5120 }, &.{ L, 5120 }, &.{ L, 4 }, &.{ L, 4, 4 }, &.{ 24, 20480 }, &.{24}, &.{5120}, &.{ L, 4 } }, &t, 0..) |sh, *o, i| {
                var r = mlx.mlx_array_new();
                const dt: mlx.mlx_dtype = if (i <= 1 or i == 6) .bfloat16 else .float32;
                try mlx.check(mlx.mlx_random_normal(&r, sh.ptr, sh.len, dt, 0, if (i == 4) 0.01 else 1, key, sc.s));
                o.* = try sc.keep(r);
            }
            const hc: Hc = .{ .fn_w = t[4], .scale = t[5], .base = t[5] };
            const sp: Split = .{ .pre = t[2], .post = t[2], .comb = t[3] };
            const cpre = t[7];
            const want_plain = try hcPreNormWide(&sc, &m, t[0], cpre, &hc, t[6]);
            const x1 = try hcExpand(&sc, t[0], t[1], &sp);
            const want_x = try hcPreNormWide(&sc, &m, x1, cpre, &hc, t[6]);
            const got_plain = try hcStage(&sc, &m, t[0], null, cpre, &hc, t[6]);
            const got_x = try hcStage(&sc, &m, t[0], .{ .y = t[1], .sp = &sp }, cpre, &hc, t[6]);
            const pairs = [_][2]A{
                .{ want_plain.h, got_plain.h },     .{ want_plain.sp.pre, got_plain.sp.pre }, .{ want_plain.sp.post, got_plain.sp.post }, .{ want_plain.sp.comb, got_plain.sp.comb },
                .{ x1, got_x.x1 },                  .{ want_x.h, got_x.h },                   .{ want_x.sp.pre, got_x.sp.pre },           .{ want_x.sp.post, got_x.sp.post },
                .{ want_x.sp.comb, got_x.sp.comb },
            };
            for (pairs) |pr| {
                var eq = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(eq);
                try mlx.check(mlx.mlx_array_equal(&eq, pr[0], pr[1], false, sc.s));
                var same = false;
                mlx.check(mlx.mlx_array_eval(eq)) catch |e| {
                    var msg: [512]u8 = undefined;
                    std.debug.print("hc rows={d} tg={d}: {s}\n", .{ L, tg, mlx.takeError(&msg) orelse "no MLX message" });
                    return e;
                };
                try mlx.check(mlx.mlx_array_item_bool(&same, eq));
                try testing.expect(same);
            }
        }
    }
}

test "dsv41 wo_a: fp8 blocks widened to bf16 serve as mxfp8 unchanged, a true bf16 weight stays dense" {
    const s = mlx.gpuStream();
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 7));
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_random_normal(&r, &[_]c_int{ 64, 128 }, 2, .bfloat16, 0, 1, key, s));
    try testing.expect(try exactMxfp8(s, r) == null);
    var qv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qv);
    try mlx.check(mlx.mlx_quantize(&qv, r, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, s));
    var q0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q0);
    var s0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(s0);
    try mlx.check(mlx.mlx_vector_array_get(&q0, qv, 0));
    try mlx.check(mlx.mlx_vector_array_get(&s0, qv, 1));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_dequantize(&wide, q0, s0, none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .bfloat16, .has_value = true }, s));
    const pair = (try exactMxfp8(s, wide)).?;
    defer for (pair) |x| {
        _ = mlx.mlx_array_free(x);
    };
    var back = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(back);
    try mlx.check(mlx.mlx_dequantize(&back, pair[0], pair[1], none, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", none, .{ .value = .bfloat16, .has_value = true }, s));
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, back, wide, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_eval(eq));
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    try testing.expect(same);
}

test "dsv41 decode step: serial one-token steps as the server pipelines them (DSV41_STEP)" {
    // DSV41_STEP = a pack dir; DSV41_STEP_CTX prompt tokens first (default 1024).
    const dir = std.mem.span(std.c.getenv("DSV41_STEP") orelse return error.SkipZigTest);
    const ctx = if (std.c.getenv("DSV41_STEP_CTX")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 1024;
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var cfg = try model.parseConfig(io, gpa, dir);
    defer cfg.deinit(gpa);
    var w = try model.loadModelWeights(io, gpa, dir, &cfg, false);
    defer w.deinit();
    const m = try init(gpa, &cfg, &w, mlx.gpuStream(), .{ .head_8bit = true });
    defer m.deinit();
    const prompt = try gpa.alloc(u32, ctx);
    defer gpa.free(prompt);
    for (prompt, 0..) |*t, k| t.* = @intCast((k * 7919 + 13) % @min(m.vocab, 100000));
    var st = try initState(m, gpa);
    defer st.deinit();
    const tok: u32 = blk: {
        const l = try extend(m, gpa, &st, prompt, .last);
        defer _ = mlx.mlx_array_free(l);
        break :blk try argmaxHost(l);
    };
    const n = 256;
    const warm = 64; // the first steps only time the host queueing ahead of the GPU
    const iou = @import("io_util.zig");
    var total: [n]i64 = undefined;
    // As the server decodes: the next token stays on the GPU (`extendIds` reads it back).
    var g = mlx.mlx_array_new_data(&tok, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(g);
    for (0..n) |k| {
        const t0 = iou.nowUsMonotonic(io);
        const l = try extendIds(m, gpa, &st, .{ .gpu = g }, .last, null);
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_argmax_axis(&o, l, -1, false, mlx.gpuStream()));
        _ = mlx.mlx_array_free(g);
        g = o;
        _ = mlx.mlx_array_free(l);
        total[k] = iou.nowUsMonotonic(io) - t0;
    }
    // Pipelined steps hand time to each other: the mean is the throughput.
    const steady = total[warm..];
    var sum: i64 = 0;
    for (steady) |t| sum += t;
    std.mem.sort(i64, steady, {}, std.sort.asc(i64));
    const mean: f64 = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(steady.len)) / 1000;
    std.debug.print("[step] ctx {d}: mean step {d:.2} ms, p10 {d:.2}, p90 {d:.2} -> {d:.1} tok/s\n", .{
        ctx, mean, @as(f64, @floatFromInt(steady[steady.len / 10])) / 1000, @as(f64, @floatFromInt(steady[steady.len * 9 / 10])) / 1000, 1000.0 / mean,
    });
}

test "dsv41 DSpark round: draft, verify and finish timed apart, greedy (DSV41_ROUND)" {
    // DSV41_ROUND = a pack dir; DSV41_STEP_CTX prompt tokens first (default 1024).
    const dir = std.mem.span(std.c.getenv("DSV41_ROUND") orelse return error.SkipZigTest);
    const ctx = if (std.c.getenv("DSV41_STEP_CTX")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 1024;
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var cfg = try model.parseConfig(io, gpa, dir);
    defer cfg.deinit(gpa);
    var w = try model.loadModelWeights(io, gpa, dir, &cfg, false);
    defer w.deinit();
    const m = try init(gpa, &cfg, &w, mlx.gpuStream(), .{ .head_8bit = true, .dspark = true });
    defer m.deinit();
    const prompt = try gpa.alloc(u32, ctx);
    defer gpa.free(prompt);
    for (prompt, 0..) |*t, k| t.* = @intCast((k * 7919 + 13) % @min(m.vocab, 100000));
    if (std.c.getenv("DSV41_ROUND_TEXT")) |tp| { // real prose: generated n-grams are new to the Engram tables
        var tok = try @import("tokenizer.zig").loadTokenizer(io, gpa, dir);
        defer tok.deinit();
        const off = if (std.c.getenv("DSV41_ROUND_OFFSET")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 0;
        const text = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(tp), gpa, .limited(64 << 20));
        defer gpa.free(text);
        const ids = try tok.encode(gpa, text[off..@min(text.len, off + ctx * 8)]);
        defer gpa.free(ids);
        if (ids.len < ctx) return error.SkipZigTest;
        @memcpy(prompt, ids[0..ctx]);
    }
    var st = try initState(m, gpa);
    defer st.deinit();
    var t1: u32 = blk: {
        const l = try extend(m, gpa, &st, prompt, .last);
        defer _ = mlx.mlx_array_free(l);
        break :blk try argmaxHost(l);
    };
    const iou = @import("io_util.zig");
    const n = 48;
    const warm = 8;
    var ms: [3]f64 = @splat(0); // draft (build + run + read), verify (+ argmax read), finish
    var accepted: usize = 0;
    for (0..n) |k| {
        const t0 = iou.nowUsMonotonic(io);
        var d = try dsparkDraft(m, gpa, &st, t1);
        defer d.deinit(gpa);
        const t_d = iou.nowUsMonotonic(io);
        var pending = try dsparkBeginWith(m, gpa, &st, t1, &d);
        defer pending.deinit();
        var sc = Scope.init(gpa, m.s);
        defer sc.deinit();
        const best = try hostI32(try sc.astype(try sc.argmax(pending.vl_g, -1), .int32));
        defer gpa.free(best);
        const t_v = iou.nowUsMonotonic(io);
        var a: usize = 0;
        while (a < pending.b and @as(u32, @intCast(best[a])) == d.ids[a + 1]) a += 1;
        var r = try dsparkFinish(m, gpa, &st, &pending, a, @intCast(best[a]));
        r.deinit(gpa);
        t1 = @intCast(best[a]);
        if (k < warm) continue;
        ms[0] += @floatFromInt(t_d - t0);
        ms[1] += @floatFromInt(t_v - t_d);
        ms[2] += @floatFromInt(iou.nowUsMonotonic(io) - t_v);
        accepted += a;
    }
    const rounds: f64 = @floatFromInt(n - warm);
    for (&ms) |*v| v.* /= rounds * 1000;
    std.debug.print("[round] ctx {d}: draft {d:.2} ms, verify {d:.2} ms, finish {d:.2} ms, {d:.2} accepted/round -> {d:.1} tok/s\n", .{
        ctx, ms[0], ms[1], ms[2], @as(f64, @floatFromInt(accepted)) / rounds, (@as(f64, @floatFromInt(accepted)) / rounds + 1) * 1000 / (ms[0] + ms[1] + ms[2]),
    });
}

test "dsv41 verify width: time of a verify block per draft count (DSV41_VERIFY_ROWS)" {
    // DSV41_VERIFY_ROWS = a pack dir; DSV41_STEP_CTX prompt tokens first (default 2048).
    const dir = std.mem.span(std.c.getenv("DSV41_VERIFY_ROWS") orelse return error.SkipZigTest);
    const ctx = if (std.c.getenv("DSV41_STEP_CTX")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 2048;
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var cfg = try model.parseConfig(io, gpa, dir);
    defer cfg.deinit(gpa);
    var w = try model.loadModelWeights(io, gpa, dir, &cfg, false);
    defer w.deinit();
    const m = try init(gpa, &cfg, &w, mlx.gpuStream(), .{ .head_8bit = true, .dspark = true });
    defer m.deinit();
    const prompt = try gpa.alloc(u32, ctx);
    defer gpa.free(prompt);
    for (prompt, 0..) |*t, k| t.* = @intCast((k * 7919 + 13) % @min(m.vocab, 100000));
    var st = try initState(m, gpa);
    defer st.deinit();
    const t1: u32 = blk: {
        const l = try extend(m, gpa, &st, prompt, .last);
        defer _ = mlx.mlx_array_free(l);
        break :blk try argmaxHost(l);
    };
    const iou = @import("io_util.zig");
    for ([_]usize{ 1, 3, 5, 7, 8, 9, 10, 11, 12, 15 }) |b| {
        const ids = try gpa.alloc(u32, b + 1);
        for (ids, 0..) |*t, k| t.* = @intCast((k * 104729 + 7) % @min(m.vocab, 100000));
        var d: DsparkDraft = .{ .ids = ids, .len = b, .logits = &.{}, .confidence = &.{} };
        defer d.deinit(gpa);
        var total: i64 = 0;
        var cold: i64 = 0;
        const reps = 12;
        for (0..reps + 3) |k| {
            const t0 = iou.nowUsMonotonic(io);
            var pending = try dsparkBeginWith(m, gpa, &st, t1, &d);
            defer pending.deinit();
            try mlx.check(mlx.mlx_array_eval(pending.vl_g));
            const dt = iou.nowUsMonotonic(io) - t0;
            var r = try dsparkFinish(m, gpa, &st, &pending, 0, 0);
            r.deinit(gpa);
            if (k == 0) cold = dt;
            if (k >= 3) total += dt;
        }
        std.debug.print("[verify] ctx {d} drafts {d} rows {d}: {d:.2} ms warm, {d:.0} ms first call\n", .{ ctx, b, b + 1, @as(f64, @floatFromInt(total)) / reps / 1000, @as(f64, @floatFromInt(cold)) / 1000 });
    }
}

fn argmaxHost(l: A) !u32 {
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_argmax_axis(&o, l, -1, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(o));
    return @intCast((mlx.mlx_array_data_uint32(o) orelse return error.NoData)[0]);
}

test "dsv41 resume: a prompt that extends the last one resumes its state and matches a cold prefill (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const a = try t.ids();
    defer gpa.free(a);
    const m = t.m;
    // Turn 2: the first prompt plus a reply and a new question, or the first prompt again.
    const b = try gpa.alloc(u32, a.len + 6);
    defer gpa.free(b);
    @memcpy(b[0..a.len], a);
    @memcpy(b[a.len..], &[_]u32{ 5, 9, 2, 7, 3, 11 });
    for ([_][]const u32{ b, a }) |p| {
        // Turn 1: the prompt, then three decode steps.
        try testing.expectEqual(@as(usize, 0), try resumePrompt(m, a, null));
        _ = mlx.mlx_array_free(try prefill(m, gpa, a, null));
        for ([_]u32{ 5, 9, 13 }) |tok| _ = mlx.mlx_array_free(try extendResumable(m, gpa, &.{tok}, null));
        const n = try resumePrompt(m, p, null);
        try testing.expectEqual(a.len - 1, n);
        const wl = try extendResumable(m, gpa, p[n..], null);
        defer _ = mlx.mlx_array_free(wl);
        const warm = try hostF32(wl);
        defer gpa.free(warm);
        m.snap.?.deinit(gpa);
        m.snap = null;
        try testing.expectEqual(@as(usize, 0), try resumePrompt(m, p, null));
        const cl = try prefill(m, gpa, p, null);
        defer _ = mlx.mlx_array_free(cl);
        const cold = try hostF32(cl);
        defer gpa.free(cold);
        try testing.expectEqualSlices(f32, cold, warm);
    }
}

test "dsv41 image token: without image rows it is text, as in the reference's text-only forward; resume included (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const a = try t.ids();
    defer gpa.free(a);
    const m = t.m;
    const b = try gpa.alloc(u32, a.len + 3);
    defer gpa.free(b);
    @memcpy(b[0..a.len], a);
    @memcpy(b[a.len..], &[_]u32{ 5, 9, 2 });
    // The logits with no image token, then with a prompt id named the image token.
    var want: ?[]f32 = null;
    defer if (want) |w| gpa.free(w);
    for ([_]u32{ 0, a[1] }) |img| {
        m.cfg.image_token_id = img;
        if (m.snap) |*sn| sn.deinit(gpa);
        m.snap = null;
        try testing.expectEqual(@as(usize, 0), try resumePrompt(m, a, null));
        _ = mlx.mlx_array_free(try prefill(m, gpa, a, null));
        const n = try resumePrompt(m, b, null);
        try testing.expectEqual(a.len - 1, n);
        const l = try extendResumable(m, gpa, b[n..], null);
        defer _ = mlx.mlx_array_free(l);
        const got = try hostF32(l);
        if (want) |w| {
            defer gpa.free(got);
            try testing.expectEqualSlices(f32, w, got);
        } else want = got;
    }
}

test "dsv41 CED split:the decoder starts at the last KV source when everything after it reads its ratio-1 rows" {
    var c: ModelConfig = .{ .model_type = "deepseek_v41", .num_hidden_layers = 40, .sliding_window = 128 };
    for (0..40) |l| c.dsv4_compress_ratios[l] = if (l < 2) 0 else if (l < 20) 2 else 1;
    c.dsv41_kv_sources = (1 << 2) | (1 << 8) | (1 << 14) | (1 << 20);
    c.dsv41_engram_layers[0..2].* = .{ 1, 14 };
    c.dsv41_n_engram_layers = 2;
    c.dsv4_dspark_target_layers[0..3].* = .{ 37, 38, 39 };
    c.dsv4_n_dspark_target_layers = 3;
    try testing.expectEqual(@as(?usize, 20), cedSplit(&c));
    // A DSpark target, an Engram layer, a pooling layer or no window: no split.
    c.dsv4_dspark_target_layers[0] = 19;
    try testing.expectEqual(@as(?usize, null), cedSplit(&c));
    c.dsv4_dspark_target_layers[0] = 37;
    c.dsv41_engram_layers[1] = 20;
    try testing.expectEqual(@as(?usize, null), cedSplit(&c));
    c.dsv41_engram_layers[1] = 14;
    c.dsv4_compress_ratios[30] = 2;
    try testing.expectEqual(@as(?usize, null), cedSplit(&c));
    c.dsv4_compress_ratios[30] = 1;
    c.has_sliding_window = false;
    try testing.expectEqual(@as(?usize, null), cedSplit(&c));
    // The tiny fixture's layout.
    var t: ModelConfig = .{ .model_type = "deepseek_v41", .num_hidden_layers = 8, .sliding_window = 8 };
    t.dsv4_compress_ratios[0..8].* = .{ 0, 0, 2, 2, 2, 1, 1, 1 };
    t.dsv41_kv_sources = (1 << 2) | (1 << 5);
    t.dsv41_engram_layers[0..2].* = .{ 1, 2 };
    t.dsv41_n_engram_layers = 2;
    try testing.expectEqual(@as(?usize, 5), cedSplit(&t));
}

/// `m`'s last-row logits for `ids` fed in `chunks` to a fresh state, then one row per decode token of `next`,
/// `[1 + next.len, V]`; with `tr`, the forward's trace.
fn cedArm(m: *Dsv41Model, ids: []const u32, chunks: []const usize, next: []const u32, tr: ?*Trace) ![]f32 {
    const gpa = testing.allocator;
    var st = try initState(m, gpa);
    defer st.deinit();
    m.trace = tr;
    defer m.trace = null;
    var out: std.ArrayList(f32) = .empty;
    defer out.deinit(gpa);
    var lo: usize = 0;
    for (chunks) |n| {
        const row = try logitsHost(m, &st, ids[lo .. lo + n], .last);
        defer gpa.free(row);
        lo += n;
        if (lo == ids.len) try out.appendSlice(gpa, row);
    }
    try testing.expectEqual(ids.len, lo);
    for (next) |tok| {
        const row = try logitsHost(m, &st, &.{tok}, .last);
        defer gpa.free(row);
        try out.appendSlice(gpa, row);
    }
    return out.toOwnedSlice(gpa);
}

fn traceLastRow(tr: *const Trace, name: []const u8) ![]f32 {
    for (tr.items.items) |it| {
        if (!std.mem.eql(u8, it.name, name)) continue;
        const all = try hostF32(it.arr);
        defer testing.allocator.free(all);
        const width = all.len / dim(it.arr, 0);
        return testing.allocator.dupe(f32, all[all.len - width ..]);
    }
    return error.MissingTrace;
}

test "dsv41 CED: a wide chunk replays its last window through the decoder, the same whatever the chunking before it; the kill switch keeps the full forward (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const m = t.m;
    const split = m.ced_split orelse return error.TestUnexpectedResult;
    const w: usize = m.cfg.sliding_window;
    var st0 = try initState(m, gpa);
    defer st0.deinit();
    try testing.expectEqual(@as(?usize, null), cedTail(m, &st0, w + CED_MIN_SKIP - 1, .last));
    try testing.expectEqual(@as(?usize, w), cedTail(m, &st0, w + CED_MIN_SKIP, .last));
    try testing.expectEqual(@as(?usize, null), cedTail(m, &st0, w + CED_MIN_SKIP, .all));
    m.ced_min_skip = w; // a chunk of 2 windows replays
    const ids = try t.ids();
    defer gpa.free(ids);
    const N = ids.len;
    const V = m.vocab;
    const next = [_]u32{ 5, 9, 13 };
    defer {
        _ = setenv("MLX_SERVE_DSV41_CED", "1", 1);
        ced_env = null;
    }

    // The full forward, and the chunk replayed as one piece; traced once each.
    var tr: [2]Trace = .{ .{ .gpa = gpa }, .{ .gpa = gpa } };
    defer for (&tr) |*x| x.deinit();
    var plain: [2][]f32 = undefined;
    defer for (plain) |p| gpa.free(p);
    for (0..2) |ced| {
        _ = setenv("MLX_SERVE_DSV41_CED", if (ced == 1) "1" else "0", 1);
        ced_env = null;
        plain[ced] = try cedArm(m, ids, &.{N}, &next, null);
        gpa.free(try cedArm(m, ids, &.{N}, &next, &tr[ced]));
    }
    const full = plain[0];
    const one = plain[1];
    try testing.expectEqual(@as(u64, 2), m.ced_stats.chunks);
    try testing.expectEqual(@as(u64, 2 * (N - w)), m.ced_stats.skipped);
    for (one) |v| try testing.expect(std.math.isFinite(v));

    // Bar: the first decoder layer's last row is the full forward's (its window is whole); later layers read truncated tails.
    var name: [32]u8 = undefined;
    const attn_name = try std.fmt.bufPrint(&name, "attn_{d}", .{split});
    const a_full = try traceLastRow(&tr[0], attn_name);
    defer gpa.free(a_full);
    const a_ced = try traceLastRow(&tr[1], attn_name);
    defer gpa.free(a_ced);
    var peak: f64 = 0;
    var worst: f64 = 0;
    for (a_full, a_ced) |a, b| {
        peak = @max(peak, @abs(@as(f64, a)));
        worst = @max(worst, @abs(@as(f64, a) - b));
    }
    std.debug.print("{s} last row: CED vs full max err {e:.2}\n", .{ attn_name, worst / peak });
    try testing.expect(worst / peak < 1e-4);
    for (0..1 + next.len) |r| {
        var dot: f64 = 0;
        var gg: f64 = 0;
        var ww: f64 = 0;
        for (one[r * V ..][0..V], full[r * V ..][0..V]) |a, b| {
            dot += @as(f64, a) * b;
            gg += @as(f64, a) * a;
            ww += @as(f64, b) * b;
        }
        const cos = dot / @sqrt(gg * ww);
        std.debug.print("CED vs full forward, row {d}: cos {d:.5}\n", .{ r, cos });
        try testing.expect(cos > 0.99);
    }

    // Bar: chunking before the replaying chunk, and a short last chunk, leave the same logits.
    const chunked = try cedArm(m, ids, &.{ 13, 17, N - 30 }, &next, null);
    defer gpa.free(chunked);
    try expectRowsClose("CED chunking 13/17/rest", chunked, one, V);
    try testing.expectEqual(@as(u64, 4), m.ced_stats.chunks);
    const short_a = try cedArm(m, ids, &.{ N - 10, 10 }, &next, null);
    defer gpa.free(short_a);
    const short_b = try cedArm(m, ids, &.{ 20, N - 30, 10 }, &next, null);
    defer gpa.free(short_b);
    try expectRowsClose("CED short last chunk", short_b, short_a, V);
    try testing.expectEqual(@as(u64, 7), m.ced_stats.chunks);

    // Several spans per tail.
    m.span = 4;
    const spanned = try cedArm(m, ids, &.{N}, &next, null);
    defer gpa.free(spanned);
    m.span = SPAN;
    try expectRowsClose("CED 4-row spans", spanned, one, V);

    // The kill switch: the full forward, nothing replayed.
    _ = setenv("MLX_SERVE_DSV41_CED", "0", 1);
    ced_env = null;
    const off = try cedArm(m, ids, &.{N}, &next, null);
    defer gpa.free(off);
    try testing.expectEqualSlices(f32, full, off);
    try testing.expectEqual(@as(u64, 8), m.ced_stats.chunks);
}

test "dsv41 CED resume: a prompt that extends the last one replays the tail a cold prefill replays, DSpark windows included (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const m = t.m;
    try testing.expect(m.ced_split != null and m.n_mtp > 0);
    m.ced_min_skip = m.cfg.sliding_window;
    const ids = try t.ids();
    defer gpa.free(ids);
    const a = ids[0..40];
    // Turn 2 adds more than a window: its suffix replays on its own.
    const b = try gpa.alloc(u32, a.len + 20);
    defer gpa.free(b);
    @memcpy(b[0..a.len], a);
    for (b[a.len..], 0..) |*v, k| v.* = @intCast((k * 7 + 3) % 90);
    var got: [2][]f32 = undefined;
    var drafted: [2][]f32 = undefined;
    for (0..2) |warm| {
        if (warm == 1) {
            // Turn 1: prompt `a`, then three decode steps.
            try testing.expectEqual(@as(usize, 0), try resumePrompt(m, a, null));
            _ = mlx.mlx_array_free(try prefill(m, gpa, a, null));
            for ([_]u32{ 5, 9, 13 }) |tok| _ = mlx.mlx_array_free(try extendResumable(m, gpa, &.{tok}, null));
        } else if (m.snap) |*sn| {
            sn.deinit(gpa);
            m.snap = null;
        }
        const n = try resumePrompt(m, b, null);
        try testing.expectEqual(if (warm == 1) a.len - 1 else 0, n);
        const before = m.ced_stats.chunks;
        const l = if (warm == 1) try extendResumable(m, gpa, b[n..], null) else try prefill(m, gpa, b, null);
        defer _ = mlx.mlx_array_free(l);
        try testing.expectEqual(before + 1, m.ced_stats.chunks);
        const d = try extendResumable(m, gpa, &.{7}, null);
        defer _ = mlx.mlx_array_free(d);
        got[warm] = try hostF32(d);
        var dr = try draftRows(m, gpa, &m.dec_state.?, 11, true);
        defer dr.deinit(gpa);
        drafted[warm] = try gpa.dupe(f32, dr.logits);
    }
    defer for (got) |g| gpa.free(g);
    defer for (drafted) |g| gpa.free(g);
    try testing.expectEqualSlices(f32, got[0], got[1]);
    try testing.expectEqualSlices(f32, drafted[0], drafted[1]);
}

test "dsv41 engram: a chunk's rows are read behind the one before it and give the same logits (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const base = try t.ids();
    defer gpa.free(base);
    const m = t.m;
    // Three chunks, each wider than `region_rows`, so every one has a wide chunk to read ahead for.
    const cuts = [_]usize{ region_rows + 1, 2 * (region_rows + 1), 3 * (region_rows + 1) };
    const ids = try gpa.alloc(u32, cuts[2]);
    defer gpa.free(ids);
    for (ids, 0..) |*id, k| id.* = base[k % base.len];
    var logits: [2][]f32 = undefined;
    for (&logits, 0..) |*out, ahead| {
        _ = try resumePrompt(m, ids, null);
        if (ahead == 0) { // the prompt is unknown: every chunk reads its own rows
            m.gpa.free(m.prompt);
            m.prompt = &.{};
        }
        var lo: usize = 0;
        for (cuts, 0..) |hi, c| {
            const l = if (c == 0) try prefill(m, gpa, ids[lo..hi], null) else try extendResumable(m, gpa, ids[lo..hi], null);
            defer _ = mlx.mlx_array_free(l);
            if (c + 1 == cuts.len) out.* = try hostF32(l);
            // After a chunk, the next one's rows are coming; after the last, nothing.
            try testing.expectEqual(ahead == 1 and c + 1 < cuts.len, m.ahead != null);
            if (m.ahead) |a| try testing.expect(a.covers(hi, @min(cuts[c + 1] - hi, hi - lo)));
            lo = hi;
        }
    }
    defer for (logits) |l| gpa.free(l);
    try testing.expectEqualSlices(f32, logits[0], logits[1]);
}

test "dsv41 init: every weight the model reads is evaluated before a region traces it (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "pipe", "fixture.safetensors");
    defer t.deinit();
    var xs: std.ArrayList(A) = .empty;
    defer xs.deinit(gpa);
    for (t.m.layers) |ly| try collectArrays(gpa, &xs, ly);
    try collectArrays(gpa, &xs, .{ t.m.embed, t.m.head, t.m.norm, t.m.dspark });
    for (t.m.engrams) |e| try collectArrays(gpa, &xs, .{ e.wkv, e.qk });
    try testing.expect(xs.items.len > 100);
    for (xs.items) |x| {
        var avail = false;
        try mlx.check(mlx._mlx_array_is_available(&avail, x));
        try testing.expect(avail);
    }
}

test "dsv41 per-expert packs: banks stacked at load serve as the stacked pack, the per-expert copies dropped (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "pipe", "fixture.safetensors");
    defer t.deinit();
    const ids = try t.ids();
    defer gpa.free(ids);
    const sl = try prefill(t.m, gpa, ids, null);
    defer _ = mlx.mlx_array_free(sl);
    const stacked = try hostF32(sl);
    defer gpa.free(stacked);

    // The same pack under TensorFold's names: one tensor per expert and projection.
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try std.fmt.allocPrint(gpa, "{s}/pipe", .{root});
    defer gpa.free(dir);
    var w = try model.loadModelWeights(io, gpa, dir, &t.cfg, false);
    defer w.deinit();
    var banks: std.ArrayList([]const u8) = .empty;
    defer banks.deinit(gpa);
    var it = w.map.keyIterator();
    while (it.next()) |k| if (std.mem.indexOf(u8, k.*, ".ffn.experts.") != null and std.mem.indexOf(u8, k.*, "_proj.") != null) try banks.append(gpa, k.*);
    const s = mlx.gpuStream();
    for (banks.items) |key| {
        const at = std.mem.indexOf(u8, key, ".ffn.experts.").? + ".ffn.experts.".len;
        const proj = key[at..std.mem.indexOfScalarPos(u8, key, at, '.').?];
        const short = if (std.mem.eql(u8, proj, "gate_proj")) "w1" else if (std.mem.eql(u8, proj, "up_proj")) "w3" else "w2";
        const x = w.get(key).?;
        const sh = [_]c_int{ ci(dim(x, 1)), ci(dim(x, 2)) };
        for (0..dim(x, 0)) |e| {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ ci(e), 0, 0 }, 3, &[_]c_int{ ci(e + 1), sh[0], sh[1] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var flat = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&flat, part, &sh, 2, s));
            try mlx.check(mlx.mlx_array_eval(flat));
            const name = try std.fmt.allocPrint(gpa, "{s}{d}.{s}{s}", .{ key[0..at], e, short, key[at + proj.len ..] });
            try w.map.put(name, flat);
        }
    }
    for (banks.items) |key| w.remove(key);

    const m = try init(gpa, &t.cfg, &w, s, .{ .act = .float32, .dspark = true });
    defer m.deinit();
    const pl = try prefill(m, gpa, ids, null);
    defer _ = mlx.mlx_array_free(pl);
    const per_expert = try hostF32(pl);
    defer gpa.free(per_expert);
    try testing.expectEqualSlices(f32, stacked, per_expert);
    try testing.expect(w.get("layers.0.ffn.experts.0.w1.weight") == null);
    try testing.expect(w.get("layers.0.ffn.experts.1.w2.scales") == null);
}

test "dsv41 oMLX layouts: each linear's mode, bits and group come off its own geometry" {
    const s = mlx.gpuStream();
    const Case = struct { w: []const c_int, sc: []const c_int, fp: bool, in: u32, bits: u32, group: u32, mode: model.QuantMode };
    // Jundot oQ3e: 3-bit experts, 6-bit wo_b, the Engram projection in mxfp8, the DSpark experts in mxfp4.
    const cases = [_]Case{
        .{ .w = &.{ 384, 2304, 480 }, .sc = &.{ 384, 2304, 80 }, .fp = false, .in = 5120, .bits = 3, .group = 64, .mode = .affine },
        .{ .w = &.{ 5120, 1536 }, .sc = &.{ 5120, 128 }, .fp = false, .in = 8192, .bits = 6, .group = 64, .mode = .affine },
        .{ .w = &.{ 25600, 1536 }, .sc = &.{ 25600, 192 }, .fp = true, .in = 6144, .bits = 8, .group = 32, .mode = .mxfp8 },
        .{ .w = &.{ 128, 2304, 640 }, .sc = &.{ 128, 2304, 160 }, .fp = true, .in = 5120, .bits = 4, .group = 32, .mode = .mxfp4 },
    };
    for (cases) |c| {
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        try mlx.check(mlx.mlx_zeros(&w, c.w.ptr, c.w.len, .uint32, s));
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        try mlx.check(mlx.mlx_zeros(&sc, c.sc.ptr, c.sc.len, if (c.fp) .uint8 else .bfloat16, s));
        const qp = transformer.quantParamsFromGeometry(w, sc, !c.fp, c.in).?;
        try testing.expectEqual(c.bits, qp.bits);
        try testing.expectEqual(c.group, qp.group_size);
        try testing.expectEqual(c.mode, qp.mode);
    }
}

test "dsv41 bill: a tensor's copy in a shard the index does not name for it is not held" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const shards = [_][2][]const u8{
        .{ "a.safetensors", "{\"language_model.x\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":[0,16]},\"language_model.y\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[16,20]}}" },
        .{ "b.safetensors", "{\"language_model.x\":{\"dtype\":\"U8\",\"shape\":[8],\"data_offsets\":[0,8]}}" },
    };
    for (shards) |sh| {
        var buf: [512]u8 = @splat(0);
        std.mem.writeInt(u64, buf[0..8], sh[1].len, .little);
        @memcpy(buf[8..][0..sh[1].len], sh[1]);
        try tmp.dir.writeFile(io, .{ .sub_path = sh[0], .data = buf[0 .. 8 + sh[1].len + 20] });
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"language_model.x\":\"b.safetensors\",\"language_model.y\":\"a.safetensors\"}}" });
    var pbuf: [1024]u8 = undefined;
    const dir = pbuf[0..try tmp.dir.realPath(io, &pbuf)];
    try testing.expectEqual(@as(u64, 16), try droppedBytes(testing.allocator, dir, true, false));
}

test "dsv41 rows: growth keeps every appended row" {
    const s = mlx.gpuStream();
    var r: Rows = .{ .width = 4, .dtype = .float32 };
    defer r.deinit();
    var want: [300 * 4]f32 = undefined;
    for (&want, 0..) |*v, i| v.* = @floatFromInt(i);
    var at: usize = 0;
    // Chunks past the 64-row start and every quarter's growth after it.
    for ([_]usize{ 50, 40, 70, 90, 50 }) |n| {
        const x = mlx.mlx_array_new_data(want[at * 4 ..].ptr, &[_]c_int{ ci(n), 4 }, 2, .float32);
        defer _ = mlx.mlx_array_free(x);
        try r.append(x, s);
        at += n;
    }
    const v = try r.view(at, s);
    defer _ = mlx.mlx_array_free(v);
    const got = try hostF32(v);
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(f32, want[0 .. at * 4], got);
}

test "dsv41 pair: DSpark on the host drafts what the whole model drafts, rollbacks included (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    // The fixture's only Engram-free worker run (5..8) holds its DSpark targets; move them below
    // it, for the whole model and the pair alike.
    var c = t.cfg;
    c.dsv4_dspark_target_layers[0..3].* = .{ 2, 3, 4 };
    const pair: Pair = .{ .lo = 5, .hi = 8, .worker = false };
    var cp = c;
    cp.dist_dsv41 = pair;
    try testing.expect(dsparkWanted(&cp));
    // A load moves what it repacks (DSpark experts, fused gate/up) out of its weight map: each model loads its own.
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try std.fmt.allocPrint(gpa, "{s}/repack", .{root});
    defer gpa.free(dir);
    var w_whole = try model.loadModelWeights(io, gpa, dir, &c, false);
    defer w_whole.deinit();
    var w_host = try model.loadModelWeights(io, gpa, dir, &c, false);
    defer w_host.deinit();
    const whole = try init(gpa, &c, &w_whole, mlx.gpuStream(), .{ .act = .float32, .dspark = true });
    defer whole.deinit();
    try testing.expect(whole.n_mtp > 0);
    const host = try init(gpa, &c, &w_host, mlx.gpuStream(), .{ .act = .float32, .dspark = true, .pair = pair });
    defer host.deinit();
    var wp = pair;
    wp.worker = true;
    var w_worker = try model.loadModelWeights(io, gpa, dir, &c, false);
    defer w_worker.deinit();
    const worker = try init(gpa, &c, &w_worker, mlx.gpuStream(), .{ .act = .float32, .pair = wp });
    defer worker.deinit();
    const Xchg = struct {
        var w: *Dsv41Model = undefined;
        fn call(x: A, pre: A, p0: usize, snap: bool, verify: bool) anyerror![2]A {
            return workerStep(w, testing.allocator, x, pre, p0, snap, verify);
        }
    };
    Xchg.w = worker;
    pair_exchange = &Xchg.call;
    defer pair_exchange = null;

    const ids = try t.ids();
    defer gpa.free(ids);
    const prompt = ids[0..32];
    const N = 24;
    var emitted: [2][N + 16]u32 = undefined;
    var partial: usize = 0;
    var after: [2]u32 = undefined;
    for ([_]*Dsv41Model{ whole, host }, 0..) |m, which| {
        var st = try initState(m, gpa);
        defer st.deinit();
        const first = try logitsHost(m, &st, prompt, .last);
        var tok = argmaxRow(first);
        gpa.free(first);
        var got: usize = 0;
        while (got < N) {
            var r = try dsparkRound(m, gpa, &st, tok, N - got - 1);
            defer r.deinit(gpa);
            @memcpy(emitted[which][got .. got + r.tokens.len], r.tokens);
            got += r.tokens.len;
            if (r.accepted < m.ds_block and got < N) partial += 1;
            tok = r.next_token;
        }
        after[which] = tok;
    }
    try testing.expectEqualSlices(u32, emitted[0][0..N], emitted[1][0..N]);
    try testing.expectEqual(after[0], after[1]);
    try testing.expect(partial > 0); // the worker rolled back with the host
}

test "dsv41 pair cuts: valid only where no layer past the cut reads what a layer below it made" {
    var c: ModelConfig = .{ .model_type = "deepseek_v41", .num_hidden_layers = 8 };
    c.dsv4_compress_ratios[0..8].* = .{ 0, 0, 2, 2, 2, 1, 1, 1 };
    c.dsv41_kv_sources = (1 << 2) | (1 << 5);
    c.dsv41_index_sources = (1 << 2) | (1 << 5) | (1 << 7);
    c.dsv41_candidate_source = 5;
    c.dsv41_engram_layers[0..2].* = .{ 1, 2 };
    c.dsv41_n_engram_layers = 2;
    for (0..9) |x| try testing.expectEqual(x == 0 or x == 1 or x == 2 or x == 5 or x == 8, pairCutOk(&c, x));
    // The worker's share of 40% of memory is 3.2 layers: the run within it, holding no Engram layer.
    try testing.expectEqual([2]usize{ 5, 8 }, pairRange(&c, 60, 40).?);
}

test "dsv41 pair: a host and a worker split at a valid cut give the unsplit model's logits, resume included (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    // A load moves what it repacks out of its weight map: each model loads its own.
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try std.fmt.allocPrint(gpa, "{s}/repack", .{root});
    defer gpa.free(dir);
    var w_host = try model.loadModelWeights(io, gpa, dir, &t.cfg, false);
    defer w_host.deinit();
    var w_worker = try model.loadModelWeights(io, gpa, dir, &t.cfg, false);
    defer w_worker.deinit();
    const host = try init(gpa, &t.cfg, &w_host, mlx.gpuStream(), .{ .act = .float32, .pair = .{ .lo = 5, .hi = 8, .worker = false } });
    defer host.deinit();
    const worker = try init(gpa, &t.cfg, &w_worker, mlx.gpuStream(), .{ .act = .float32, .pair = .{ .lo = 5, .hi = 8, .worker = true } });
    defer worker.deinit();
    const Xchg = struct {
        var w: *Dsv41Model = undefined;
        fn call(x: A, pre: A, p0: usize, snap: bool, verify: bool) anyerror![2]A {
            return workerStep(w, testing.allocator, x, pre, p0, snap, verify);
        }
    };
    Xchg.w = worker;
    pair_exchange = &Xchg.call;
    defer pair_exchange = null;
    // Several spans per chunk on both sides.
    for ([_]*Dsv41Model{ t.m, host, worker }) |m| m.span = 4;

    const a = try t.ids();
    defer gpa.free(a);
    const b = try gpa.alloc(u32, a.len + 6);
    defer gpa.free(b);
    @memcpy(b[0..a.len], a);
    @memcpy(b[a.len..], &[_]u32{ 5, 9, 2, 7, 3, 11 });
    // Turn 1 (prefill + three decode steps), then turn 2 resumed from the snapshot.
    var got: [2][]f32 = undefined;
    for ([_]*Dsv41Model{ t.m, host }, 0..) |m, which| {
        _ = try resumePrompt(m, a, null);
        _ = mlx.mlx_array_free(try prefill(m, gpa, a, null));
        for ([_]u32{ 5, 9, 13 }) |tok| _ = mlx.mlx_array_free(try extendResumable(m, gpa, &.{tok}, null));
        const n = try resumePrompt(m, b, null);
        try testing.expectEqual(a.len - 1, n);
        const l = try extendResumable(m, gpa, b[n..], null);
        defer _ = mlx.mlx_array_free(l);
        got[which] = try hostF32(l);
    }
    defer for (got) |g| gpa.free(g);
    try expectRowsClose("pair vs whole", got[1], got[0], t.cfg.vocab_size);
}

test "dsv41 head: the 8-bit head is the default, MLX_SERVE_DSV41_HEAD8=0 keeps the head as the pack ships it" {
    try testing.expect(head8From(null));
    try testing.expect(head8From("1"));
    try testing.expect(!head8From("0"));
}
