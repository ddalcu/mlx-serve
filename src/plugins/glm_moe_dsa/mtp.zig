//! GLM-5.3's multi-token prediction head (the release's layer `num_hidden_layers`, `num_nextn_predict_layers` = 1) and
//! the draft lane's state over it. Pair p is (the token at p + 1, the target's final-normed hidden at p); the head
//! computes `h = layer(eh_proj(cat[enorm(embed(token)), hnorm(hidden)]))` and its logits `lm_head(shared_head.norm(h))`
//! with the target's embedding and head (transformers `MtpLayer` / `MtpModel`, modeling_layers.py; vLLM
//! `DeepSeekMultiTokenPredictorLayer`). The layer is the trunk's decoder layer (MLA with a full indexer, a routed MLP
//! with its shared expert) over its own cache: pair p's latent, rope key and indexer key at position p.
//!
//! A round's first step appends every pending pair (the pairs the target verified since the last round) and runs the
//! layer for the last one, which drafts the first token. Depth k > 1 runs the same layer on (the draft k - 1,
//! shared_head.norm(h) of the step before) at position p + k - 1, the last pair's p. `index_share_for_mtp_iteration`
//! (vLLM's `set_skip_topk`): such a step runs no indexer and attends the keys the first step's row selected, every
//! committed pair's key while they fit `index_topk`; it appends nothing, so a rejected draft leaves the cache as it was.
//!
//! The residents are the published BF16 tensors of `mtp/mtp-residents.safetensors`, bound as they are: `kv_b_proj`
//! split into the per-head `embed_q` / `unembed_out` by a reshape and slices (views of the same bytes, the reference's
//! unquantized sanitize). The routed experts are a bank kind bound at comptime: `.exl3` (the served lane: the EXL3
//! mini-expert records of `mtp/mtp-experts.bin`, resident, through sushi's EXL3 MoE on the host's `mlx_host`) or
//! `.dense` (the CPU lane's fixture: BF16 `mlp.switch_mlp.*` tensors in the residents file).

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const sdk = @import("sdk");
const log = sdk.log;
const glm = @import("glm_moe_dsa.zig");
const graph = @import("graph.zig");
const cache_mod = @import("cache.zig");
const settings = @import("settings.zig");
const acceptance = @import("../../sdk/weights/streamed/sdk_ext.zig").acceptance;
const exl3 = @import("exl3_bank.zig");
const topk_mod = @import("topk.zig");
const bo = @import("../../sdk/weights/streamed/build_flags.zig");
/// The acceptance-by-top-probability histogram (`Counts.decided`, `accepted_by_p`, `ProbLine`) and the per-step
/// reduction it reads: a diagnostic, in profile builds (`build_flags.glm_mtp_histograms`) and the tests only.
pub const histograms: bool = bo.glm_mtp_histograms or @import("builtin").is_test;

const G = graph.G;
const T = G.T;
const Diag = glm.Diag;

pub const dir_name = exl3.mtp_dir;
pub const residents_file = "mtp-residents.safetensors";
pub const bank_file = exl3.mtp_sidecar_file;
/// The deepest draft a round verifies (depth + 1 rows of top-8 routes in one decode-lane call).
pub const max_depth = settings.Config.max_mtp_depth;
/// Prompt pairs whose keys one append covers (the prompt pass's catch-up, in chunks).
pub const append_rows: u32 = 2048;
/// Serial steps whose hidden the lane keeps for a later round; past them the lane stops tracking the request.
pub const max_pending: u32 = 64;

pub const BankKind = enum { exl3, dense };

pub const Refusal = error{ MtpPackMissing, MtpConfig, MtpTensorMissing, MtpTensorDtype, MtpTensorShape, MtpTensorUnexpected, MtpBankSize };

fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// The MTP layer as a one-layer model of the trunk's config: index 0, a full indexer, a routed MLP. Never deinit'd
/// (it borrows nothing it frees).
pub fn layerConfig(c: *const glm.Config) glm.Config {
    var m = c.*;
    m.n_layers = 1;
    m.indexer_types = @splat(.shared);
    m.indexer_types[0] = .full;
    m.mlp_types = @splat(.dense);
    m.mlp_types[0] = .sparse;
    m.owned_overrides = &.{};
    m.owned_paths = &.{};
    return m;
}

/// The release's config as the lane needs it: an MTP layer, and a selection shared across a round's steps when the
/// lane drafts more than one token (the lane implements `index_share_for_mtp_iteration` only).
pub fn checkConfig(c: *const glm.Config, depth: u32, diag: ?*Diag) Refusal!void {
    if (c.n_nextn < 1) return refuse(diag, error.MtpConfig, "config: num_nextn_predict_layers is {d} (the draft lane needs the release's MTP layer)", .{c.n_nextn});
    if (depth > 1 and !c.index_share_mtp) return refuse(diag, error.MtpConfig, "config: index_share_for_mtp_iteration is not true (a draft past the first reuses the first step's selection; mtp_depth {d})", .{depth});
}

// ── the pack's MTP directory, host side ──

/// The MTP layer's residents as published (BF16; the indexer, the router and its bias as stored), names in `a`.
/// `.dense` adds the routed experts' `mlp.switch_mlp.{gate,up,down}_proj.weight` `[E, out, in]` (the CPU lane's).
pub fn residentSpec(a: std.mem.Allocator, c: *const glm.Config, kind: BankKind) ![]glm.Param {
    var out: std.ArrayList(glm.Param) = .empty;
    const bf16 = &[_]glm.StDtype{.BF16};
    const S = struct {
        fn add(o: *std.ArrayList(glm.Param), al: std.mem.Allocator, l: u32, rest: []const u8, dtypes: []const glm.StDtype, shape: []const u64) !void {
            var p: glm.Param = .{ .name = try std.fmt.allocPrint(al, "model.layers.{d}.{s}", .{ l, rest }), .dtypes = dtypes, .rank = @intCast(shape.len) };
            @memcpy(p.shape[0..shape.len], shape);
            try o.append(al, p);
        }
    };
    const l = c.n_layers;
    const h: u64 = c.hidden_size;
    const heads: u64 = c.n_heads;
    const si: u64 = @as(u64, c.moe_intermediate_size) * c.n_shared_experts;
    const e: u64 = c.n_routed_experts;
    for ([_][]const u8{ "enorm.weight", "hnorm.weight", "shared_head.norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight" }) |n| try S.add(&out, a, l, n, bf16, &.{h});
    try S.add(&out, a, l, "eh_proj.weight", bf16, &.{ h, 2 * h });
    try S.add(&out, a, l, "self_attn.q_a_proj.weight", bf16, &.{ c.q_lora_rank, h });
    try S.add(&out, a, l, "self_attn.q_a_layernorm.weight", bf16, &.{c.q_lora_rank});
    try S.add(&out, a, l, "self_attn.q_b_proj.weight", bf16, &.{ heads * c.qHeadDim(), c.q_lora_rank });
    try S.add(&out, a, l, "self_attn.kv_a_proj_with_mqa.weight", bf16, &.{ c.kv_lora_rank + c.qk_rope_head_dim, h });
    try S.add(&out, a, l, "self_attn.kv_a_layernorm.weight", bf16, &.{c.kv_lora_rank});
    try S.add(&out, a, l, "self_attn.kv_b_proj.weight", bf16, &.{ heads * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank });
    try S.add(&out, a, l, "self_attn.o_proj.weight", bf16, &.{ h, heads * c.v_head_dim });
    try S.add(&out, a, l, "self_attn.indexer.wq_b.weight", &glm.float_dtypes, &.{ @as(u64, c.index_n_heads) * c.index_head_dim, c.q_lora_rank });
    try S.add(&out, a, l, "self_attn.indexer.wk.weight", &glm.float_dtypes, &.{ c.index_head_dim, h });
    try S.add(&out, a, l, "self_attn.indexer.k_norm.weight", &glm.float_dtypes, &.{c.index_head_dim});
    try S.add(&out, a, l, "self_attn.indexer.k_norm.bias", &glm.float_dtypes, &.{c.index_head_dim});
    try S.add(&out, a, l, "self_attn.indexer.weights_proj.weight", &glm.float_dtypes, &.{ c.index_n_heads, h });
    try S.add(&out, a, l, "mlp.gate.weight", &glm.float_dtypes, &.{ e, h });
    try S.add(&out, a, l, "mlp.gate.e_score_correction_bias", &glm.float_dtypes, &.{e});
    try S.add(&out, a, l, "mlp.shared_experts.gate_proj.weight", bf16, &.{ si, h });
    try S.add(&out, a, l, "mlp.shared_experts.up_proj.weight", bf16, &.{ si, h });
    try S.add(&out, a, l, "mlp.shared_experts.down_proj.weight", bf16, &.{ h, si });
    if (kind == .dense) {
        const inter: u64 = c.moe_intermediate_size;
        try S.add(&out, a, l, "mlp.switch_mlp.gate_proj.weight", bf16, &.{ e, inter, h });
        try S.add(&out, a, l, "mlp.switch_mlp.up_proj.weight", bf16, &.{ e, inter, h });
        try S.add(&out, a, l, "mlp.switch_mlp.down_proj.weight", bf16, &.{ e, h, inter });
    }
    return out.toOwnedSlice(a);
}

/// The residents file against the spec from its header alone: every spec tensor present in one of its dtypes at its
/// exact shape, nothing else. Returns its tensor bytes.
pub fn checkResidents(spec: []const glm.Param, ck: *const glm.Checkpoint, diag: ?*Diag) Refusal!u64 {
    for (spec) |p| {
        const t = ck.tensors.get(p.name) orelse return refuse(diag, error.MtpTensorMissing, "{s}/{s}: {s} missing", .{ dir_name, residents_file, p.name });
        if (std.mem.indexOfScalar(glm.StDtype, p.dtypes, t.dtype) == null) return refuse(diag, error.MtpTensorDtype, "{s}: dtype {t}, want {any}", .{ p.name, t.dtype, p.dtypes });
        if (t.rank != p.rank or !std.mem.eql(u64, t.shape[0..t.rank], p.shapeOf())) return refuse(diag, error.MtpTensorShape, "{s}: shape {any}, want {any}", .{ p.name, t.shape[0..t.rank], p.shapeOf() });
    }
    var bytes: u64 = 0;
    for (ck.tensors.keys(), ck.tensors.values()) |name, t| {
        for (spec) |p| {
            if (std.mem.eql(u8, p.name, name)) break;
        } else return refuse(diag, error.MtpTensorUnexpected, "{s}/{s}: {s} is not a tensor of the MTP layer this build binds", .{ dir_name, residents_file, name });
        bytes += t.end - t.begin;
    }
    return bytes;
}

/// What the lane holds besides its KV, from the pack's headers and manifest (the bill's terms), as MLX allocates it: the
/// residents (each tensor its buffer), and the experts' arrays (each bank layer's segment over its minis) with their slot
/// table (`Exl3.base`, an i32 per expert). The lane measures them at its open (`Lane.measured`).
pub const Facts = struct { resident_bytes: u64, expert_bytes: u64 };

/// The MTP directory of the pack at `dir` checked from its headers (the residents against the spec, the EXL3
/// manifest under the trunk bank's rules, `exl3.Bank.openMtp`): refused by name when it is missing.
pub fn facts(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, kind: BankKind, diag: ?*Diag) !Facts {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}/{s}", .{ dir, dir_name, residents_file }, 0);
    if (std.c.access(path.ptr, std.c.F_OK) != 0) return refuse(diag, error.MtpPackMissing, "{s}: no MTP layer in this pack (mtp_depth > 0 needs {s}/ beside the shards: scripts/glm53_pack.py mtp)", .{ path, dir_name });
    var ck = glm.Checkpoint.openFile(gpa, path, diag) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return refuse(diag, error.MtpPackMissing, "{s}: {s}", .{ path, @errorName(e) }),
    };
    defer ck.deinit();
    _ = try checkResidents(try residentSpec(a, c, kind), &ck, diag);
    const page = std.heap.pageSize();
    const resident_bytes = glm.bufferBytes(&ck, page);
    const expert_bytes = switch (kind) {
        .dense => 0,
        .exl3 => blk: {
            var b = try exl3.Bank.openMtp(a, io, try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, dir_name }), c, diag);
            defer b.deinit();
            var n: u64 = 0;
            for (b.layers) |l| for (l.segments) |sg| {
                n += cache_mod.bufferBytes(@as(u64, l.minis) * l.n_held * sg.length, page);
            };
            break :blk n + cache_mod.bufferBytes(@as(u64, c.n_routed_experts) * @sizeOf(i32), page);
        },
    };
    return .{ .resident_bytes = resident_bytes, .expert_bytes = expert_bytes };
}

const host_caps = @import("../../sdk/weights/streamed/host_caps.zig");
const sushi = if (host_caps.exl3) @import("mlx_host").exl3.sushi else @compileError("glm_moe_dsa: this host exports no exl3 (host_caps): the reference must sit behind host_caps.glm");

// ── the routed experts ──

/// The MTP layer's routed experts of bank kind `kind`: `moe(x [n, hidden], indices [n, k] int32, scores [n, k] f32)`
/// = the weighted sum `[n, hidden]` in x's dtype.
pub fn Experts(comptime kind: BankKind) type {
    return switch (kind) {
        .dense => Dense,
        .exl3 => Exl3,
    };
}

/// BF16 experts `[E, out, in]` (the fixture's), each routed expert's matrices taken by id and multiplied in a batch.
pub const Dense = struct {
    gate: T,
    up: T,
    down: T,

    pub fn open(_: std.mem.Allocator, _: std.Io, _: []const u8, c: *const glm.Config, w: *const sdk.Weights, diag: ?*Diag) !Dense {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.mlp.switch_mlp", .{c.n_layers});
        return .{ .gate = try get(w, p, "gate_proj.weight", diag), .up = try get(w, p, "up_proj.weight", diag), .down = try get(w, p, "down_proj.weight", diag) };
    }

    pub fn deinit(_: *Dense) void {}

    pub fn moe(self: *const Dense, g: *G, x: T, indices: T, scores: T) !T {
        const k: c_int = g.shapeOf(indices).dim(1);
        const n: c_int = g.shapeOf(x).dim(0);
        const h: c_int = g.shapeOf(x).dim(1);
        const flat = try g.reshape(indices, &.{-1});
        const x4 = try g.reshape(x, &.{ n, 1, 1, h });
        const P = struct {
            fn proj(gg: *G, w: T, idx: T, v: T, kk: c_int) !T {
                const ws = try gg.take(w, idx, 0);
                const s = gg.shapeOf(ws);
                return gg.matmul(v, try graph.swapLast(gg, try gg.reshape(ws, &.{ -1, kk, s.dim(1), s.dim(2) })));
            }
        };
        const act = try g.mul(try g.silu(try P.proj(g, self.gate, flat, x4, k)), try P.proj(g, self.up, flat, x4, k));
        const y = try g.reshape(try P.proj(g, self.down, flat, act, k), &.{ n, k, h });
        return g.astype(try g.sum(try g.mul(y, try g.expandDims(scores, -1)), -2, false), g.dtypeOf(y));
    }
};

/// The EXL3 bank: each bank layer (one K) its nine components as resident arrays `[minis, ...]` (the records'
/// segments, bytes unchanged; the trellis read as uint16), and per expert its first mini's slot across the bank
/// layers. A routed expert is its `tp` minis at the expert's score (`scripts/glm53_pack.py`).
pub const Exl3 = struct {
    banks: [2]sushi.Bank = undefined,
    n_banks: usize = 0,
    /// int32 [E]: the expert's first mini slot (its bank layer's offset + local x tp).
    base: T,
    tp: c_int,
    /// Every array this bank made.
    owned: std.ArrayList(T) = .empty,
    gpa: std.mem.Allocator,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const glm.Config, _: *const sdk.Weights, diag: ?*Diag) !Exl3 {
        const mdir = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ dir, dir_name });
        defer gpa.free(mdir);
        // Its sidecar past the page cache (the experts are resident in MLX arrays; a cached copy would only take the
        // box's memory).
        var m = try exl3.Bank.openMtp(gpa, io, mdir, c, diag);
        defer m.deinit();
        var self: Exl3 = .{ .base = undefined, .tp = @intCast(m.tp), .gpa = gpa };
        errdefer self.deinit();
        var slot_base: [2]u32 = .{ 0, 0 };
        var next: u32 = 0;
        for (m.layers, 0..) |l, b| {
            // An empty bank layer holds no expert: no arrays, no group of the MoE.
            if (m.slots[b].empty()) continue;
            const n_minis = l.minis * l.n_held;
            slot_base[b] = next;
            next += n_minis;
            var arrs: [9]T = undefined;
            for (l.segments, 0..) |sg, ci| {
                // On mapped pages, unmapped at the copy's end: a large block freed to libc's malloc can stay in the
                // process's footprint (its large cache), outside every term of the bill.
                const host = try std.heap.page_allocator.alloc(u8, @intCast(@as(u64, n_minis) * sg.length));
                defer std.heap.page_allocator.free(host);
                for (0..n_minis) |mi| {
                    const off = l.base_offset + mi * l.record_bytes + sg.offset;
                    sdk.io_util.readAligned(m.sidecar.fd, host[mi * sg.length ..][0..@intCast(sg.length)], off) catch return refuse(diag, error.MtpBankSize, "{s}: short read at {d}", .{ m.sidecar_path, off });
                }
                var shape: [4]c_int = undefined;
                shape[0] = @intCast(n_minis);
                for (sg.shape[0..sg.rank], 0..) |x, i| shape[i + 1] = @intCast(x);
                const arr = mlx.mlx_array_new_data(host.ptr, &shape, @intCast(1 + sg.rank), exl3.mlxDtype(sg.dtype));
                if (arr.ctx == null) return error.MlxError;
                try self.owned.append(gpa, arr);
                arrs[ci] = arr;
            }
            // code = trellis, rout = svh (the output side), rin = suh (the input side).
            self.banks[self.n_banks] = .{
                .gate = .{ .trellis = arrs[0], .svh = arrs[1], .suh = arrs[2] },
                .up = .{ .trellis = arrs[3], .svh = arrs[4], .suh = arrs[5] },
                .down = .{ .trellis = arrs[6], .svh = arrs[7], .suh = arrs[8] },
            };
            self.n_banks += 1;
        }
        const base = try gpa.alloc(i32, m.n_experts);
        defer gpa.free(base);
        for (base, 0..) |*x, e| {
            const b = m.streamLayer(0, @intCast(e));
            x.* = @intCast(slot_base[b] + m.local[b * m.n_experts + e] * m.tp);
        }
        const shape = [_]c_int{@intCast(base.len)};
        self.base = mlx.mlx_array_new_data(base.ptr, &shape, 1, .int32);
        if (self.base.ctx == null) return error.MlxError;
        try self.owned.append(gpa, self.base);
        var n: u64 = 0;
        for (m.layers) |l| n += @as(u64, l.minis) * l.n_held * l.logical_bytes;
        log.info("glm_moe_dsa: MTP experts {d} bank layers, {d} minis, {d} B resident ({s})\n", .{ m.layers.len, next, n, m.sidecar_path });
        return self;
    }

    pub fn deinit(self: *Exl3) void {
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(self.gpa);
        self.n_banks = 0;
    }

    /// The minis' slots and scores of the routed experts `[1, n, k * tp]`, then sushi's EXL3 MoE over the bank layers.
    pub fn moe(self: *const Exl3, g: *G, x: T, indices: T, scores: T) !T {
        const sh = g.shapeOf(indices);
        const n = sh.dim(0);
        const k = sh.dim(1);
        const h = g.shapeOf(x).dim(1);
        const first = try g.reshape(try g.take(self.base, try g.reshape(indices, &.{-1}), 0), &.{ n, k, 1 });
        const ranks = try g.reshape(try g.arange(0, @floatFromInt(self.tp), 1, .int32), &.{ 1, 1, self.tp });
        const slots = try g.reshape(try g.add(first, ranks), &.{ 1, n, k * self.tp });
        const sc = try g.reshape(try g.broadcastTo(try g.reshape(scores, &.{ n, k, 1 }), &.{ n, k, self.tp }), &.{ 1, n, k * self.tp });
        const y = try g.adopt(try sushi.moeGroups(g.s, try g.reshape(x, &.{ 1, n, h }), self.banks[0..self.n_banks], slots, sc, .mcg, false));
        return g.reshape(y, &.{ n, h });
    }
};

fn get(w: *const sdk.Weights, prefix: []const u8, rest: []const u8, diag: ?*Diag) !T {
    var buf: [192]u8 = undefined;
    return w.need(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ prefix, rest }), diag, error.MtpTensorMissing, "not in the MTP residents");
}

// ── the head ──

/// The MTP layer's residents bound (the decoder layer's projections dense, `embed_q` / `unembed_out` as views of
/// `kv_b_proj`), the views kept by the head.
pub const Head = struct {
    layer: graph.Layer,
    eh_proj: T,
    enorm: T,
    hnorm: T,
    head_norm: T,
    /// The views of `kv_b_proj` (embed_q, unembed_out), released with the head.
    views: [2]T,

    pub fn bind(g: *G, w: *const sdk.Weights, c: *const glm.Config, diag: ?*Diag) !Head {
        var pb: [64]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}", .{c.n_layers});
        const d = graph.QLinear.dense;
        const m0 = g.mark();
        defer g.resetTo(m0);
        // kv_b_proj [heads * (nope + v), kv_lora] -> [heads, nope + v, kv_lora]; embed_q's weight is the nope rows
        // swapped (mlx-lm's `MultiLinear` [heads, kv_lora, nope]), unembed_out's the v rows [heads, v, kv_lora].
        const heads: c_int = @intCast(c.n_heads);
        const nope: c_int = @intCast(c.qk_nope_head_dim);
        const vd: c_int = @intCast(c.v_head_dim);
        const kvr: c_int = @intCast(c.kv_lora_rank);
        const v3 = try g.reshape(try get(w, p, "self_attn.kv_b_proj.weight", diag), &.{ heads, nope + vd, kvr });
        const eq = g.keep(try graph.swapLast(g, try g.slice(v3, &.{ 0, 0, 0 }, &.{ heads, nope, kvr }, &.{ 1, 1, 1 })));
        errdefer g.release(eq);
        const uo = g.keep(try g.slice(v3, &.{ 0, nope, 0 }, &.{ heads, nope + vd, kvr }, &.{ 1, 1, 1 }));
        errdefer g.release(uo);
        const at = struct {
            fn f(ww: *const sdk.Weights, pre: []const u8, rest: []const u8, dg: ?*Diag) !T {
                return get(ww, pre, rest, dg);
            }
        }.f;
        return .{
            .layer = .{
                .input_norm = try at(w, p, "input_layernorm.weight", diag),
                .post_norm = try at(w, p, "post_attention_layernorm.weight", diag),
                .q_a = d(try at(w, p, "self_attn.q_a_proj.weight", diag)),
                .q_a_norm = try at(w, p, "self_attn.q_a_layernorm.weight", diag),
                .q_b = d(try at(w, p, "self_attn.q_b_proj.weight", diag)),
                .kv_a = d(try at(w, p, "self_attn.kv_a_proj_with_mqa.weight", diag)),
                .kv_a_norm = try at(w, p, "self_attn.kv_a_layernorm.weight", diag),
                .embed_q = d(eq),
                .unembed_out = d(uo),
                .o = d(try at(w, p, "self_attn.o_proj.weight", diag)),
                .indexer = .{
                    .wq_b = try at(w, p, "self_attn.indexer.wq_b.weight", diag),
                    .wk = try at(w, p, "self_attn.indexer.wk.weight", diag),
                    .k_norm_w = try at(w, p, "self_attn.indexer.k_norm.weight", diag),
                    .k_norm_b = try at(w, p, "self_attn.indexer.k_norm.bias", diag),
                    .weights_proj = try at(w, p, "self_attn.indexer.weights_proj.weight", diag),
                },
                .router = .{ .w = try at(w, p, "mlp.gate.weight", diag), .bias = try at(w, p, "mlp.gate.e_score_correction_bias", diag) },
                .shared = .{ .gate = d(try at(w, p, "mlp.shared_experts.gate_proj.weight", diag)), .up = d(try at(w, p, "mlp.shared_experts.up_proj.weight", diag)), .down = d(try at(w, p, "mlp.shared_experts.down_proj.weight", diag)) },
            },
            .eh_proj = try at(w, p, "eh_proj.weight", diag),
            .enorm = try at(w, p, "enorm.weight", diag),
            .hnorm = try at(w, p, "hnorm.weight", diag),
            .head_norm = try at(w, p, "shared_head.norm.weight", diag),
            .views = .{ eq, uo },
        };
    }

    pub fn deinit(self: *Head, g: *G) void {
        for (self.views) |v| g.release(v);
    }
};

// ── acceptance ──

/// A round's decision on the verify rows' logits `[w, vocab]` (row i scores the token after row i's): how many
/// drafts the target accepts and the token after them (the correction, or the bonus when all are accepted).
pub const Decision = struct { accepted: u32, next: u32 };

/// `drafts` against `logits` under `mode` and the request's sampling (`base`: the absolute position of row 0, the
/// sampled draws' key). Greedy: exact accepts a draft equal to its row's argmax; typical accepts one whose probability
/// at temperature 1 exceeds min(eps, delta * exp(-entropy)) (the host's `sdk.acceptance.typicalThreshold`); the
/// correction and the bonus are the argmax. Sampled: every draft lane's rules (`sdk_ext.acceptance`): exact is the
/// sample-then-verify rule (`sampledRows`: every row's token the tempered, filtered row's position-keyed draw, a draft
/// accepted when it is that token: the serial step's tokens); typical (`typicalRows`) tests the same p against its own entropy and draws
/// the correction and the bonus from p.
/// `prepare`, one eval of its arrays, `DecisionArrays.read`.
pub fn decide(g: *G, logits: T, drafts: []const u32, mode: sdk.acceptance.Mode, sampling: sdk.SamplingParams, base: u64) !Decision {
    const p = try prepare(g, logits, drafts, mode, sampling, base);
    try g.evalAll(&.{ p.tok, p.flags orelse p.tok });
    return p.read(g);
}

/// A decision's device arrays on the verify rows' logits (`prepare`): each row's token (`tok`, uint32 [w]: the
/// argmax or the draw) and the drafted rows' acceptance flags (`flags`, bool [drafts]; null: greedy exact, whose drafts
/// the host compares with `tok`).
pub const DecisionArrays = struct {
    tok: T,
    flags: ?T = null,
    w: usize,
    drafts: [max_depth]u32 = undefined,
    n_drafts: usize,

    /// The decision from the arrays the caller's eval produced, read in place: no wait on the GPU (`G.readU32`;
    /// `error.NotEvaluated` when that eval did not include `tok` and `flags`).
    pub fn read(p: *const DecisionArrays, g: *G) !Decision {
        var tok: [max_depth + 1]u32 = undefined;
        var flags: [max_depth]bool = undefined;
        _ = try g.readU32(p.tok, tok[0..p.w]);
        if (p.flags) |f| _ = try g.readBool(f, flags[0..p.n_drafts]);
        var a: u32 = 0;
        while (a < p.n_drafts) : (a += 1) {
            const ok = if (p.flags != null) flags[a] else p.drafts[a] == tok[a];
            if (!ok) break;
        }
        return .{ .accepted = a, .next = tok[a] };
    }
};

/// `decide`'s graph, not evaluated: the caller evaluates `tok` and `flags` (a round: with the verify's own outputs).
pub fn prepare(g: *G, logits: T, drafts: []const u32, mode: sdk.acceptance.Mode, sampling: sdk.SamplingParams, base: u64) !DecisionArrays {
    const w: usize = @intCast(g.shapeOf(logits).dim(0));
    std.debug.assert(w == drafts.len + 1 and w <= max_depth + 1);
    const typ: ?acceptance.Typical = switch (mode) {
        .typical => |t| t,
        .exact => null,
        else => return error.MtpAcceptanceNotImplemented,
    };
    var p: DecisionArrays = .{ .tok = undefined, .w = w, .n_drafts = drafts.len };
    @memcpy(p.drafts[0..drafts.len], drafts);
    if (sampling.greedy()) {
        p.tok = try g.argmax(logits, -1);
        if (typ) |t| if (drafts.len > 0) {
            p.flags = try acceptance.typicalFlags(g, logits, drafts, t);
        };
    } else {
        const sm = acceptance.Params.of(sampling);
        const sr = if (typ) |t| try acceptance.typicalRows(g, logits, sm, base, drafts, t) else try acceptance.sampledRows(g, logits, sm, base, drafts);
        p.tok = sr.tok;
        p.flags = sr.accept;
    }
    return p;
}

// ── the lane ──

/// The lane's counters over one request (the host's `sdk.DraftStats` and the request's line).
pub const Counts = struct {
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    generated: u64 = 0,
    /// Rounds the lane ran without drafts (the budget's last tokens, or a request it did not track).
    serial: u64 = 0,
    wall_ns: u64 = 0,
    /// The rounds' time in the drafts (the MTP steps) and in the verify: its forward, its one eval (which computes the
    /// decision's arrays too, `prepare`) and the stream's flush. The decision's read and the commit are the rest of
    /// `wall_ns`.
    draft_ns: u64 = 0,
    verify_ns: u64 = 0,
    /// The largest rise of MLX's high-water mark over a round's start (the round's transient, measured).
    peak_rise: u64 = 0,
    /// The verify's demand reads in bytes, and of them the reads whose expert no kept row routes (the first row of
    /// the layer's call that routes it is past the accepted ones): the rejected rows' own reads.
    verify_bytes: u64 = 0,
    rejected_bytes: u64 = 0,
    /// Per tenth of the head's top probability at a draft step: the steps the decision reached (the accepted ones
    /// and the first rejected one) and the accepted ones.
    decided: [10]u64 = @splat(0),
    accepted_by_p: [10]u64 = @splat(0),
};

/// The tenth of `p` (a probability): 0 .. 9.
pub fn tenth(p: f32) usize {
    return @min(@as(usize, @intFromFloat(@max(p, 0) * 10)), 9);
}

/// The lane's line at the request's end.
pub const Line = struct {
    depth: u32,
    mode: []const u8,
    delta: f32,
    c: Counts,

    pub fn format(p: Line, w: *std.Io.Writer) std.Io.Writer.Error!void {
        const rounds: f64 = @floatFromInt(@max(p.c.rounds, 1));
        const rate: f64 = if (p.c.drafted == 0) 0 else 100 * @as(f64, @floatFromInt(p.c.accepted)) / @as(f64, @floatFromInt(p.c.drafted));
        const s = @as(f64, @floatFromInt(p.c.wall_ns)) / 1e9;
        try w.print("glm_moe_dsa: mtp depth {d} acceptance {s}", .{ p.depth, p.mode });
        if (std.mem.eql(u8, p.mode, "typical")) try w.print(" (delta {d})", .{p.delta});
        try w.print(": {d} rounds ({d} without drafts), {d} drafted, {d} accepted ({d:.1}%), {d:.2} accepted / {d:.2} tokens per round, {d} tokens in {d:.2} s ({d:.1} tok/s; drafts {d:.2} s, verify {d:.2} s; round peak {d:.0} MB)", .{
            p.c.rounds,                                           p.c.serial,                                            p.c.drafted, p.c.accepted, rate,
            @as(f64, @floatFromInt(p.c.accepted)) / rounds,       @as(f64, @floatFromInt(p.c.generated)) / rounds,       p.c.generated, s,
            if (s == 0) 0 else @as(f64, @floatFromInt(p.c.generated)) / s, @as(f64, @floatFromInt(p.c.draft_ns)) / 1e9, @as(f64, @floatFromInt(p.c.verify_ns)) / 1e9,
            @as(f64, @floatFromInt(p.c.peak_rise)) / 1e6,
        });
        const gb = @as(f64, @floatFromInt(p.c.verify_bytes)) / 1e9;
        try w.print(", verify reads {d:.2} GB ({d:.3} GB per emitted token), {d:.2} GB of it for rejected rows only", .{
            gb, if (p.c.generated == 0) 0 else gb / @as(f64, @floatFromInt(p.c.generated)), @as(f64, @floatFromInt(p.c.rejected_bytes)) / 1e9,
        });
    }
};

/// The lane's acceptance by the head's top probability, at the request's end.
pub const ProbLine = struct {
    c: Counts,

    pub fn format(p: ProbLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.writeAll("glm_moe_dsa: mtp acceptance by the head's top probability (accepted / decided):");
        for (p.c.decided, p.c.accepted_by_p, 0..) |d, acc, i| try w.print("{s} {d}.{d} {d}/{d}", .{ if (i == 0) "" else ",", i / 10, i % 10, acc, d });
    }
};

/// The MTP head over its cache, with bank kind `kind`. The cache holds the pairs 0 .. `cache.len - 1`; `pending` the
/// target's final-normed hidden of the positions `cache.len ..` whose pairs are not appended yet (their next tokens
/// arrive later): `cache.len + n_pending` is the target's length while the lane tracks the request.
pub fn Lane(comptime kind: BankKind) type {
    return struct {
        const Self = @This();
        pub const Bank = Experts(kind);

        gpa: std.mem.Allocator,
        /// The MTP layer as a one-layer config (its cache's and its attention's).
        cfg1: glm.Config,
        weights: sdk.Weights,
        head: Head,
        bank: Bank,
        cache: graph.Cache,
        pending: ?T = null,
        n_pending: u32 = 0,
        depth: u32,
        mode: sdk.acceptance.Mode,
        counts: Counts = .{},
        /// The head's top probability at each step of the last round's drafts.
        top_p: [max_depth]f32 = @splat(0),
        /// Logged once per request that the lane stopped tracking.
        untracked_logged: bool = false,
        /// The indexer's top-k route the module built at its construction (`graph.Routes.topk`; null: argpartition).
        topk: ?*const topk_mod.Kernels = null,
        /// MLX's active bytes the open added (its residents, its experts' arrays and their slot table), measured once:
        /// the module holds them to the bill's term (`glm_moe_dsa_bill.Mtp`).
        measured_bytes: u64 = 0,

        /// The lane over the pack at `dir` (its `mtp/`), `depth` drafts a round, `cap` positions in its cache.
        pub fn open(gpa: std.mem.Allocator, io: std.Io, g: *G, dir: []const u8, c: *const glm.Config, depth: u32, mode: sdk.acceptance.Mode, cap: u32, nocache: bool, diag: ?*Diag) !*Self {
            try checkConfig(c, depth, diag);
            _ = try facts(gpa, io, dir, c, kind, diag);
            const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}/{s}", .{ dir, dir_name, residents_file }, 0);
            defer gpa.free(path);
            var w = sdk.Weights.init(gpa);
            errdefer w.deinit();
            const cpu = mlx.mlx_default_cpu_stream_new();
            defer _ = mlx.mlx_stream_free(cpu);
            try sdk.loader.file(gpa, &w, path.ptr, cpu, .{ .nocache = nocache });
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.* = .{ .gpa = gpa, .cfg1 = layerConfig(c), .weights = w, .head = undefined, .bank = undefined, .cache = undefined, .depth = depth, .mode = mode };
            // MLX's cache emptied first: a buffer it holds would serve one of the lane's arrays at the cached buffer's
            // size (up to two pages more), and the lane's bytes are billed at their own.
            _ = mlx.mlx_clear_cache();
            var active0: usize = 0;
            _ = mlx.mlx_get_active_memory(&active0);
            self.head = try Head.bind(g, &self.weights, c, diag);
            errdefer self.head.deinit(g);
            self.bank = try Bank.open(gpa, io, dir, c, &self.weights, diag);
            errdefer self.bank.deinit();
            self.cache = try graph.Cache.init(gpa, &self.cfg1, cap);
            // The residents resident now (a lazy load would land in the first round), and their bytes as MLX counts them.
            try self.weights.materialize();
            var active1: usize = 0;
            _ = mlx.mlx_get_active_memory(&active1);
            self.measured_bytes = active1 -| active0;
            log.info("glm_moe_dsa: MTP lane depth {d}, acceptance {s}, {d} residents ({s}), {d} B resident with its experts (measured)\n", .{ depth, sdk.acceptance.name(mode), self.weights.count(), path, self.measured_bytes });
            return self;
        }

        pub fn deinit(self: *Self, g: *G) void {
            if (self.pending) |p| g.release(p);
            self.cache.deinit(g);
            self.bank.deinit();
            self.head.deinit(g);
            self.weights.deinit();
            self.gpa.destroy(self);
        }

        /// The pairs appended plus the positions pending: the target length the lane tracks.
        pub fn tracked(self: *const Self) u64 {
            return @as(u64, self.cache.len) + self.n_pending;
        }

        /// The final-normed hidden `[rows, hidden]` of the target positions after the tracked ones, pending. Past
        /// `max_pending` serial positions (a request the lane does not draft) the lane stops tracking: the rows are
        /// dropped and `tracked` stays behind the target.
        pub fn pend(self: *Self, g: *G, target_len: u64, hidden: T, rows: u32, serial: bool) !void {
            if (self.tracked() + rows != target_len) return;
            if (serial and self.n_pending + rows > max_pending) {
                if (!self.untracked_logged) log.info("glm_moe_dsa: MTP lane: {d} serial steps past its last round, the lane stops tracking this request\n", .{self.n_pending});
                self.untracked_logged = true;
                return;
            }
            const m = g.mark();
            defer g.resetTo(m);
            const all = if (self.pending) |p| try g.concat(&.{ p, hidden }, 0) else hidden;
            const kept = g.keep(all);
            try g.evalAll(&.{kept});
            if (self.pending) |p| g.release(p);
            self.pending = kept;
            self.n_pending += rows;
        }

        /// The pairs input `eh_proj(cat[enorm(embed(tokens)), hnorm(hidden)])` `[n, hidden]`.
        fn input(self: *const Self, g: *G, tw: *const graph.Weights, tokens: []const u32, hidden: T) !T {
            const c = &self.cfg1;
            const e = try graph.rmsNorm(g, try graph.embed(g, tw, tokens), self.head.enorm, c.rms_norm_eps);
            const hn = try graph.rmsNorm(g, try g.astype(hidden, g.dtypeOf(e)), self.head.hnorm, c.rms_norm_eps);
            return graph.linear(g, try g.concat(&.{ e, hn }, 1), self.head.eh_proj);
        }

        /// The pending pairs whose next token `tokens` gives (`tokens[i]` follows pending row i), at most
        /// `tokens.len`, appended in chunks of `append_rows`; the rest stay pending.
        pub fn append(self: *Self, g: *G, tw: *const graph.Weights, tokens: []const u32) !void {
            const n: u32 = @intCast(@min(tokens.len, self.n_pending));
            if (n == 0) return;
            const lw = &self.head.layer;
            var at: u32 = 0;
            while (at < n) {
                const end = @min(at + append_rows, n);
                const m = g.mark();
                defer g.resetTo(m);
                const x = try self.input(g, tw, tokens[at..end], try graph.rowSlice(g, self.pending.?, @intCast(at), @intCast(end)));
                const a_in = try graph.rmsNorm(g, x, lw.input_norm, self.cfg1.rms_norm_eps);
                const k = try graph.keysOf(g, &self.cfg1, lw, a_in, self.cache.len, end - at);
                try self.cache.append(g, 0, k.latent, k.k_pe, k.index);
                self.cache.commit(end - at);
                var bufs: [3]T = undefined;
                try g.evalAll(self.cache.buffers(0, &bufs));
                at = end;
            }
            const m = g.mark();
            defer g.resetTo(m);
            const old = self.pending.?;
            self.pending = if (n == self.n_pending) null else g.keep(try rowCopy(g, old, n, self.n_pending));
            if (self.pending) |p| try g.evalAll(&.{p});
            g.release(old);
            self.n_pending -= n;
        }

        /// The layer's output for one row `x [1, hidden]` at position `pos` over the cache as it stands, with the
        /// selection `topk` (null: every key).
        fn row(self: *const Self, g: *G, x: T, pos: u32, topk: ?T) !T {
            const c = &self.cfg1;
            const lw = &self.head.layer;
            const a_in = try graph.rmsNorm(g, x, lw.input_norm, c.rms_norm_eps);
            const q = try graph.queryOf(g, c, lw, a_in, pos, 1);
            const o = try graph.absorbedRow(g, c, lw, q.q_nope, q.q_pe, try self.cache.latentView(g, 0), try self.cache.ropeView(g, 0), topk);
            const attn = try graph.qlinear(g, try g.reshape(try g.transposeAxes(o, &.{ 0, 2, 1, 3 }), &.{ 1, @as(c_int, @intCast(c.n_heads * c.v_head_dim)) }), lw.o);
            const h1 = try g.add(x, attn);
            const x2 = try graph.rmsNorm(g, h1, lw.post_norm, c.rms_norm_eps);
            const r = try graph.route(g, c, lw.router.?, x2);
            const f = try g.add(try self.bank.moe(g, x2, r.indices, r.weights), try graph.mlp(g, lw.shared.?, x2));
            return g.add(h1, f);
        }

        /// The selection of the cache's last pair's row `x` (its own indexer row over every index key), null while
        /// the keys fit `index_topk`; kept, the caller releases it.
        fn ownSelection(self: *Self, g: *G, a: std.mem.Allocator, x: T, pos: u32) !?T {
            const c = &self.cfg1;
            const lw = &self.head.layer;
            const a_in = try graph.rmsNorm(g, x, lw.input_norm, c.rms_norm_eps);
            const q = try graph.queryOf(g, c, lw, a_in, pos, 1);
            var carry = try graph.Carry.init(a, c, 1, pos + 1, false);
            defer carry.deinit(g);
            try graph.select(g, c, lw.indexer.?, a_in, q.iq.?, try self.cache.indexView(g, 0), pos, 1, &carry, self.topk);
            return if (carry.topk[0]) |t| g.keep(t) else null;
        }

        /// A round's drafts after `t1` (the token after the target's last position): the pending pairs appended (their
        /// tokens `after` then `t1`), the first step on the last of them, then `depth - 1` steps on the drafts. `force`
        /// (a test's) replaces the first drafts. `logits_out` (a test's, `depth` x vocab) gets each step's logits.
        pub fn draft(self: *Self, g: *G, a: std.mem.Allocator, tw: *const graph.Weights, after: []const u32, t1: u32, depth: u32, out: []u32, force: []const u32, logits_out: ?[]f32) !void {
            std.debug.assert(after.len + 1 == self.n_pending and out.len >= depth and depth >= 1);
            const tokens = try a.alloc(u32, self.n_pending);
            defer a.free(tokens);
            @memcpy(tokens[0..after.len], after);
            tokens[after.len] = t1;
            // Every pending pair but the last leaves only its keys; the last one's row is the first step's.
            try self.append(g, tw, tokens[0 .. tokens.len - 1]);
            const c = &self.cfg1;
            const m0 = g.mark();
            defer g.resetTo(m0);
            var x = try self.input(g, tw, tokens[tokens.len - 1 ..], self.pending.?);
            const lw = &self.head.layer;
            const a_in = try graph.rmsNorm(g, x, lw.input_norm, c.rms_norm_eps);
            const k = try graph.keysOf(g, c, lw, a_in, self.cache.len, 1);
            try self.cache.append(g, 0, k.latent, k.k_pe, k.index);
            self.cache.commit(1);
            g.release(self.pending.?);
            self.pending = null;
            self.n_pending = 0;
            const pos = self.cache.len - 1;
            const topk = try self.ownSelection(g, a, x, pos);
            defer if (topk) |t| g.release(t);
            const vocab: usize = c.vocab_size;
            for (0..depth) |step| {
                const s: u32 = @intCast(step);
                const h = try self.row(g, x, pos + s, topk);
                const normed = g.keep(try graph.rmsNorm(g, h, self.head.head_norm, c.rms_norm_eps));
                defer g.release(normed);
                const lg = try graph.qlinear(g, normed, tw.lm_head);
                if (logits_out) |lo| _ = try g.hostF32(try g.astype(lg, .float32), lo[step * vocab ..][0..vocab]);
                // The argmax; where the histogram is compiled in, the head's top probability `exp(max - logsumexp)` read
                // with it.
                const lf = try g.reshape(try g.astype(lg, .float32), &.{-1});
                const am = try g.argmax(lf, 0);
                var tok: [1]u32 = undefined;
                if (comptime histograms) {
                    const top = try g.exp(try g.sub(try g.max(lf, 0, false), try g.logsumexp(lf, 0, false)));
                    try g.evalAll(&.{ am, top });
                    var tp: [1]f32 = undefined;
                    _ = try g.hostF32(top, &tp);
                    self.top_p[step] = tp[0];
                } else try g.evalAll(&.{am});
                _ = try g.hostU32(am, &tok);
                out[step] = if (step < force.len) force[step] else tok[0];
                if (step + 1 < depth) {
                    x = try self.input(g, tw, out[step..][0..1], normed);
                }
            }
            var bufs: [3]T = undefined;
            try g.evalAll(self.cache.buffers(0, &bufs));
        }

        /// The lane's part of a prefix restore: `n` target positions match the new prompt. Returns the positions the
        /// target keeps (at most `n`) so that the lane tracks exactly them: its pairs past them dropped (and a pair
        /// formed with the token after the match), its pending rows past them dropped.
        pub fn keepFor(self: *Self, g: *G, n: u64) u64 {
            const p: u64 = self.cache.len;
            var k: u64 = @min(n, p + self.n_pending);
            if (k > 0 and p >= k and k == n) k -= 1;
            const keep_pairs = @min(p, k);
            self.cache.truncateTo(@intCast(keep_pairs));
            const rows: u32 = @intCast(k - keep_pairs);
            if (rows == 0) {
                if (self.pending) |x| g.release(x);
                self.pending = null;
                self.n_pending = 0;
            } else if (rows < self.n_pending) {
                const m = g.mark();
                defer g.resetTo(m);
                const old = self.pending.?;
                const kept = rowCopy(g, old, 0, rows) catch return k;
                g.evalAll(&.{kept}) catch return k;
                self.pending = g.keep(kept);
                g.release(old);
                self.n_pending = rows;
            }
            self.untracked_logged = false;
            return k;
        }

        pub fn line(self: *const Self) Line {
            return .{ .depth = self.depth, .mode = sdk.acceptance.name(self.mode), .delta = switch (self.mode) {
                .typical => |t| t.delta,
                else => 0,
            }, .c = self.counts };
        }
    };
}

/// Rows [lo, hi) of `x` (its axis n - 2) as a copy, for the lane's pending rows: a slice's view keeps all of `x`
/// alive (one pending row would hold the prompt call's whole 16,387-row hidden into the decode).
fn rowCopy(g: *G, x: T, lo: u32, hi: u32) !T {
    return g.take(x, try g.arange(@floatFromInt(lo), @floatFromInt(hi), 1, .uint32), @intCast(g.shapeOf(x).n - 2));
}

const testing = std.testing;

test "glm mtp: the lane's kept pending rows are a copy: the rows' values, and the source's buffer leaves MLX's active memory at its release" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(testing.allocator, s);
    defer g.deinit();
    const rows = 4096;
    const hidden = 1024;
    const m = g.mark();
    const x = g.keep(try g.reshape(try g.arange(0, rows * hidden, 1, .float32), &.{ 1, rows, hidden }));
    try g.evalAll(&.{x});
    g.resetTo(m);
    var before: usize = 0;
    _ = mlx.mlx_get_active_memory(&before);
    const y = g.keep(try rowCopy(&g, x, rows - 2, rows));
    try g.evalAll(&.{y});
    g.resetTo(m);
    defer g.release(y);
    g.release(x);
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    // the source's 16 MiB left (a view would keep them: no fall); the copy's two rows stay, rounded up by the allocator
    try testing.expect(before -| after >= rows * hidden * 4 / 2);
    var got: [2 * hidden]f32 = undefined;
    _ = try g.hostF32(try g.reshape(y, &.{-1}), &got);
    for (got, 0..) |v, i| try testing.expectEqual(@as(f32, @floatFromInt((rows - 2) * hidden + i)), v);
    g.resetTo(m);
}

test "glm mtp: the route limit gives depth 5; the MTP layer config is one full, routed layer" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    try testing.expectEqual(@as(u32, 5), max_depth);
    var c = try glm.Config.parse(testing.allocator, try tinyText(), null, null);
    defer c.deinit(testing.allocator);
    const m = layerConfig(&c);
    try testing.expectEqual(@as(u32, 1), m.n_layers);
    try testing.expect(m.isFull(0) and m.isSparse(0));
    try testing.expectEqual(@as(u32, 1), m.nSparse());
}

test "glm mtp: the release's MTP residents (23 BF16 tensors, the box's file) pass the spec; the config's flags gate the lane" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const text = try glm.testConfigJson(testing.allocator, "");
    defer testing.allocator.free(text);
    var c = try glm.Config.parse(testing.allocator, text, &glm.glm53, null);
    defer c.deinit(testing.allocator);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const spec = try residentSpec(arena.allocator(), &c, .exl3);
    try testing.expectEqual(@as(usize, 23), spec.len);
    var bytes: u64 = 0;
    for (spec) |p| {
        var n: u64 = 2;
        for (p.shapeOf()) |d| n *= d;
        bytes += n;
    }
    // The box's mtp-residents.safetensors: 578,488,832 tensor bytes (the bias F32: 256 x 4 for 256 x 2 here).
    try testing.expectEqual(@as(u64, 578_488_832 - 512), bytes);
    try testing.expectEqualStrings("model.layers.78.enorm.weight", spec[0].name);
    try checkConfig(&c, 5, null);
    var no = c;
    no.index_share_mtp = false;
    try checkConfig(&no, 1, null);
    var diag: Diag = .{};
    try testing.expectError(error.MtpConfig, checkConfig(&no, 2, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "index_share_for_mtp_iteration") != null);
    no.n_nextn = 0;
    try testing.expectError(error.MtpConfig, checkConfig(&no, 1, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "num_nextn_predict_layers") != null);
}

var tiny_text_buf: [8192]u8 = undefined;

fn tinyText() ![]const u8 {
    var fba = std.heap.FixedBufferAllocator.init(&tiny_text_buf);
    return glm.tinyConfigJson(fba.allocator(), glm.tiny_quant);
}

test "glm mtp: the request's line (the ABBA harness reads it): depth, acceptance, rounds, drafted, accepted, rate, per round, tok/s" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const l: Line = .{ .depth = 3, .mode = "exact", .delta = 0, .c = .{ .rounds = 348, .drafted = 1042, .accepted = 676, .generated = 1024, .wall_ns = 207_560_000_000, .draft_ns = 9_000_000_000, .verify_ns = 190_000_000_000, .peak_rise = 312_400_000, .verify_bytes = 2_100_000_000_000, .rejected_bytes = 230_000_000_000, .decided = .{ 9, 0, 0, 0, 0, 0, 0, 0, 30, 990 }, .accepted_by_p = .{ 1, 0, 0, 0, 0, 0, 0, 0, 20, 655 } } };
    const s = try std.fmt.allocPrint(testing.allocator, "{f}", .{l});
    defer testing.allocator.free(s);
    try testing.expectEqualStrings("glm_moe_dsa: mtp depth 3 acceptance exact: 348 rounds (0 without drafts), 1042 drafted, 676 accepted (64.9%), 1.94 accepted / 2.94 tokens per round, 1024 tokens in 207.56 s (4.9 tok/s; drafts 9.00 s, verify 190.00 s; round peak 312 MB), verify reads 2100.00 GB (2.051 GB per emitted token), 230.00 GB of it for rejected rows only", s);
    const ps = try std.fmt.allocPrint(testing.allocator, "{f}", .{ProbLine{ .c = l.c }});
    defer testing.allocator.free(ps);
    try testing.expectEqualStrings("glm_moe_dsa: mtp acceptance by the head's top probability (accepted / decided): 0.0 1/9, 0.1 0/0, 0.2 0/0, 0.3 0/0, 0.4 0/0, 0.5 0/0, 0.6 0/0, 0.7 0/0, 0.8 20/30, 0.9 655/990", ps);
    try testing.expectEqual(@as(usize, 0), tenth(0.0999));
    try testing.expectEqual(@as(usize, 1), tenth(0.1));
    try testing.expectEqual(@as(usize, 9), tenth(1.0));
    const t: Line = .{ .depth = 2, .mode = "typical", .delta = 0.3, .c = .{ .rounds = 2, .drafted = 4, .accepted = 3, .generated = 5, .wall_ns = 1_000_000_000 } };
    const u = try std.fmt.allocPrint(testing.allocator, "{f}", .{t});
    defer testing.allocator.free(u);
    try testing.expect(std.mem.startsWith(u8, u, "glm_moe_dsa: mtp depth 2 acceptance typical (delta 0.3): 2 rounds"));
}

test "glm mtp: the pack's mtp/ opens under the trunk bank's rules: the MTP layer's two bank layers, its experts map, the refusals by name" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    const a = testing.allocator;
    var c = try glm.Config.parse(a, try tinyText(), null, null);
    defer c.deinit(a);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = try exl3.tmpRoot(&tmp, &rbuf);
    const img = try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{ .mtp_only = true });
    defer a.free(img);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const mdir = try std.fmt.allocPrint(arena.allocator(), "{s}/{s}", .{ root, dir_name });
    var diag: Diag = .{};
    var m = exl3.Bank.openMtp(a, testing.io, mdir, &c, &diag) catch |e| {
        std.debug.print("refused: {s}\n", .{diag.message()});
        return e;
    };
    defer m.deinit();
    // The MTP layer's K3 and K4 bank layers (experts even and odd, 4 minis each) as the bank's one routed layer.
    try testing.expectEqual(@as(usize, 2), m.layers.len);
    try testing.expectEqual(@as(u32, c.n_layers), m.layers[0].model_layer);
    try testing.expectEqual(@as(u32, 3), m.layers[0].k);
    try testing.expectEqual(@as(u32, 32), m.layers[1].minis * m.layers[1].n_held);
    try testing.expectEqual(@as(u32, 1), m.streamLayer(0, 7));
    try testing.expectEqual(@as(u16, 3), m.local[m.n_experts + 7]);
    try testing.expectEqual(@as(u64, 16), m.mini_inter);
    try testing.expectEqual(@as(u64, img.len), m.sidecar.size);
    // A wrong codebook multiplier, a short sidecar, a config of other dims.
    a.free(try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{ .mtp_only = true, .multiplier = 0x83DCD12D }));
    try testing.expectError(error.QuantNotImplemented, exl3.Bank.openMtp(a, testing.io, mdir, &c, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "multiplier") != null);
    a.free(try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{ .mtp_only = true, .truncate_file = true }));
    try testing.expectError(error.SidecarGeometry, exl3.Bank.openMtp(a, testing.io, mdir, &c, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), bank_file) != null);
    a.free(try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{ .mtp_only = true }));
    var other = c;
    other.hidden_size = 256;
    try testing.expectError(error.DimsMismatch, exl3.Bank.openMtp(a, testing.io, mdir, &other, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "dims") != null);
    // The trunk's manifest is not the MTP directory's: its first bank layers are the routed layers'.
    a.free(try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{}));
    var sub = try tmp.dir.openDir(testing.io, dir_name, .{});
    defer sub.close(testing.io);
    try sub.writeFile(testing.io, .{ .sub_path = exl3.mtp_manifest_file, .data = try tmp.dir.readFileAlloc(testing.io, exl3.manifest_file, arena.allocator(), .limited(1 << 20)) });
    try testing.expectError(error.LayerGeometry, exl3.Bank.openMtp(a, testing.io, mdir, &c, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "bank layer entry 0") != null);
    // A uniform pack's MTP directory: one K3 bank layer of all 16 experts, the second stream layer empty.
    a.free(try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{ .mtp_only = true, .k3_experts = c.n_routed_experts }));
    var m1 = try exl3.Bank.openMtp(a, testing.io, mdir, &c, &diag);
    defer m1.deinit();
    try testing.expectEqual(@as(usize, 2), m1.layers.len);
    try testing.expectEqual(@as(u32, 64), m1.layers[0].minis * m1.layers[0].n_held);
    try testing.expectEqual(@as(u32, 0), m1.layers[1].n_held);
    try testing.expectEqual(@as(u32, 0), m1.streamLayer(0, 7));
    try testing.expectEqual(@as(u16, 7), m1.local[7]);
    // Two ranks: the same experts in minis of twice the width.
    a.free(try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{ .mtp_only = true, .tp = 2 }));
    var m2 = try exl3.Bank.openMtp(a, testing.io, mdir, &c, &diag);
    defer m2.deinit();
    try testing.expectEqual(@as(u32, 16), m2.layers[0].minis * m2.layers[0].n_held);
    try testing.expectEqual(@as(u64, 32), m2.mini_inter);
}

test "glm mtp: the EXL3 bank's arrays are the records' segments byte for byte, [minis, ...], and each expert's first slot is its bank offset + local x tp" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = testing.allocator;
    var c = try glm.Config.parse(a, try tinyText(), null, null);
    defer c.deinit(a);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = try exl3.tmpRoot(&tmp, &rbuf);
    const img = try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{ .mtp_only = true });
    defer a.free(img);
    var w = sdk.Weights.init(a);
    defer w.deinit();
    var b = try Exl3.open(a, testing.io, root, &c, &w, null);
    defer b.deinit();
    try testing.expectEqual(@as(usize, 2), b.n_banks);
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var m = try exl3.Bank.openMtp(a, testing.io, try std.fmt.allocPrint(arena.allocator(), "{s}/{s}", .{ root, dir_name }), &c, null);
    defer m.deinit();
    for (m.layers, 0..) |l, bi| {
        const n_minis = l.minis * l.n_held;
        const bank = b.banks[bi];
        const arrs = [9]mlx.mlx_array{ bank.gate.trellis, bank.gate.svh, bank.gate.suh, bank.up.trellis, bank.up.svh, bank.up.suh, bank.down.trellis, bank.down.svh, bank.down.suh };
        for (l.segments, arrs, 0..) |sg, arr, ci| {
            try mlx.check(mlx.mlx_array_eval(arr));
            try testing.expectEqual(if (ci % 3 == 0) mlx.mlx_dtype.uint16 else mlx.mlx_dtype.float16, mlx.mlx_array_dtype(arr));
            try testing.expectEqual(@as(usize, n_minis) * sg.length, (mlx.mlx_array_size(arr) * mlx.mlx_array_itemsize(arr)));
            const bytes = mlx.mlx_array_data_uint8(arr).?[0..(mlx.mlx_array_size(arr) * mlx.mlx_array_itemsize(arr))];
            for (0..n_minis) |mi| {
                const off = l.base_offset + mi * l.record_bytes + sg.offset;
                try testing.expectEqualSlices(u8, img[off..][0..sg.length], bytes[mi * sg.length ..][0..sg.length]);
            }
        }
    }
    try mlx.check(mlx.mlx_array_eval(b.base));
    const base = mlx.mlx_array_data_int32(b.base).?;
    for (0..c.n_routed_experts) |e| try testing.expectEqual(@as(i32, @intCast(if (e % 2 == 0) (e / 2) * 4 else 32 + (e / 2) * 4)), base[e]);
}

test "glm mtp: the EXL3 bank's MoE on the GPU equals sushi's host decode of the same records (each routed expert = its 4 minis at its score)" {
    if (comptime !host_caps.glm) return error.SkipZigTest;
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const a = testing.allocator;
    // hidden 128 and minis of 128 (inter 512 over 4 ranks): the kernels' Hadamard blocks.
    const base_text = try tinyText();
    const text = try std.mem.replaceOwned(u8, a, base_text, "\"moe_intermediate_size\":64,", "\"moe_intermediate_size\":512,");
    defer a.free(text);
    var c = try glm.Config.parse(a, text, null, null);
    defer c.deinit(a);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var rbuf: [512]u8 = undefined;
    const root = try exl3.tmpRoot(&tmp, &rbuf);
    const img = try exl3.writeSynth(a, testing.io, tmp.dir, &c, .{ .mtp_only = true, .signs = true });
    defer a.free(img);
    var w = sdk.Weights.init(a);
    defer w.deinit();
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var g = try G.init(a, s);
    defer g.deinit();
    var b = try Exl3.open(a, testing.io, root, &c, &w, null);
    defer b.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var m = try exl3.Bank.openMtp(a, testing.io, try std.fmt.allocPrint(arena.allocator(), "{s}/{s}", .{ root, dir_name }), &c, null);
    defer m.deinit();
    const h = c.hidden_size;
    const n = 2;
    const k = 8;
    var rng = std.Random.DefaultPrng.init(5);
    var xs: [n * 128]f32 = undefined;
    for (&xs) |*v| v.* = rng.random().floatNorm(f32) * 0.5;
    const ids = [n * k]i32{ 0, 3, 5, 6, 9, 10, 12, 15, 1, 2, 4, 7, 8, 11, 13, 14 };
    var sc: [n * k]f32 = undefined;
    for (&sc) |*v| v.* = 0.05 + rng.random().float(f32) * 0.2;
    const xb = try g.astype(try g.hostArray(std.mem.sliceAsBytes(&xs), &.{ n, @intCast(h) }, .float32), .bfloat16);
    var xin: [n * 128]f32 = undefined;
    _ = try g.hostF32(try g.astype(xb, .float32), &xin);
    const y = try b.moe(&g, xb, try g.hostArray(std.mem.sliceAsBytes(&ids), &.{ n, k }, .int32), try g.hostArray(std.mem.sliceAsBytes(&sc), &.{ n, k }, .float32));
    var got: [n * 128]f32 = undefined;
    _ = try g.hostF32(try g.astype(y, .float32), &got);
    // The host reference: sushi's own decode of each mini's three projections from the records (the EXL3 quant's).
    var ids16: [n * k]u16 = undefined;
    for (&ids16, ids) |*d, e| d.* = @intCast(e);
    var want: [n * 128]f32 = undefined;
    @import("exl3_quant.zig").hostMoe(img, &m, 0, &xin, &ids16, &sc, k, &want);
    var scale: f32 = 0;
    var worst: f32 = 0;
    for (want, got) |x, y_| {
        scale = @max(scale, @abs(x));
        worst = @max(worst, @abs(x - y_));
    }
    std.debug.print("glm mtp exl3: GPU MoE against the host decode, max |delta| {d:.5} at outputs up to {d:.3}\n", .{ worst, scale });
    try testing.expect(scale > 0.1);
    try testing.expect(worst <= 0.03 * scale);
}
