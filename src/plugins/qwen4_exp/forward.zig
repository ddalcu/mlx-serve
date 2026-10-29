//! The qwen4_exp trunk forward (`forwardQwen4With`): embeddings tiled into hyper-connection streams, the n-gram PLE,
//! GDN and QSA attention layers under hc reads and writes, the final mixer; also the grouped verify rows and a
//! pipeline rank's stream forward. Env-gated fixture tests against the HF reference live here.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const mrope = @import("../../sdk/vision/mrope.zig");
const model_mod = @import("../../sdk/config/model.zig");
const log = @import("../../sdk/util/log.zig");
const gather_qmv = @import("../../sdk/blocks/moe/gather_qmv.zig");
const mtp = @import("mtp.zig");
const standin = @import("../../sdk/diag/standin.zig");
const verify_projection = @import("../../sdk/blocks/linear/verify_projection.zig");
const MAX_BATCH_ROWS = @import("../../engines/mlx/batched/layered.zig").MAX_BATCH_ROWS;
const layerCap = @import("../../engines/mlx/diag/profile.zig").layerCap;
const env = @import("../../env.zig");
const profileFwdOn = @import("prof.zig").profileFwdOn;
const PREFILL_EVAL_CADENCE_DEFAULT = @import("../../engines/mlx/eval_cadence.zig").PREFILL_EVAL_CADENCE_DEFAULT;
const evalCadencePoint = @import("../../engines/mlx/eval_cadence.zig").evalCadencePoint;
const ladderStepMulti = @import("../../engines/mlx/eval_cadence.zig").ladderStepMulti;
const prefillEvalCadenceApplies = @import("../../engines/mlx/eval_cadence.zig").prefillEvalCadenceApplies;
const prefillEvalCadenceFor = @import("../../engines/mlx/eval_cadence.zig").prefillEvalCadenceFor;
const MoeLayerWeights = @import("../../engines/mlx/layer_weights.zig").MoeLayerWeights;
const CaptureLayers = @import("../../engines/mlx/forward_ctx.zig").CaptureLayers;
const ForwardCtx = @import("../../engines/mlx/forward_ctx.zig").ForwardCtx;
const KVCache = @import("../../engines/mlx/kv/cache.zig").KVCache;
const testing = @import("../../engines/mlx/kv/cache.zig").testing;
const SSMCacheEntry = @import("../../engines/mlx/ssm/entry.zig").SSMCacheEntry;
const ssmFreeSpecCapture = @import("../../engines/mlx/ssm/entry.zig").ssmFreeSpecCapture;
const ssmRollbackFromCapture = @import("../../engines/mlx/ssm/entry.zig").ssmRollbackFromCapture;
const nextSsmMemberGen = @import("../../engines/mlx/ssm/group.zig").nextSsmMemberGen;
const ssmLiveAux = @import("../../engines/mlx/ssm/group.zig").ssmLiveAux;
const Trunk = @import("trunk.zig").Trunk;
const attn256RandBf16 = @import("../../sdk/blocks/attention/attn256.zig").attn256RandBf16;
const verifyLmHeadProjection = @import("../../sdk/blocks/linear/verify_projection.zig").verifyLmHeadProjection;
const moeRouterFusedEnabled = @import("../../sdk/blocks/moe/router_kernel.zig").moeRouterFusedEnabled;
const endMropeChunk = @import("../../engines/mlx/methods/mrope.zig").endMropeChunk;
const qwen4Standin = @import("../../sdk/diag/standin.zig").qwen4Standin;
const standinRef = @import("../../sdk/diag/standin.zig").standinRef;
const Qwen4TiedCmp = @import("../../sdk/testing/parity.zig").Qwen4TiedCmp;
const qwen4CompareRows = @import("../../sdk/testing/parity.zig").qwen4CompareRows;
const qwen4CompareTied = @import("../../sdk/testing/parity.zig").qwen4CompareTied;
const qwen4ReadF32 = @import("../../sdk/testing/parity.zig").qwen4ReadF32;
const HcRead = @import("hc.zig").HcRead;
const HcPending = @import("hc_fused.zig").HcPending;
const Qwen4Trace = @import("mtp.zig").Qwen4Trace;
const PleArmFixture = @import("ple.zig").PleArmFixture;
const expectPleHistoryEqual = @import("ple.zig").expectPleHistoryEqual;
const pleArmEntry = @import("ple.zig").pleArmEntry;
const pleArmIds = @import("ple.zig").pleArmIds;
const qwen4BatchedPleForward = @import("ple.zig").qwen4BatchedPleForward;
const Qwen4AttnProf = @import("prof.zig").Qwen4AttnProf;
const Qwen4FwdProf = @import("prof.zig").Qwen4FwdProf;
const ssmFreeQsaState = @import("qsa/history.zig").ssmFreeQsaState;

/// qwen4_exp's batched decode builds each step's graph while the GPU idles, so its ladder ships ON at this stride.
/// Serial decode and other arches keep `decode_async_ladder`.
pub const QWEN4_DECODE_ASYNC_LADDER_DEFAULT: u32 = 4;

pub fn qwen4VerifyRowsEligible(self: *Trunk, token_rows: []const mlx.mlx_array, ctxs: []const *ForwardCtx) bool {
    // A tensor shard's MoE is half of each expert until `moeTensorParallel` adds the
    // peer's half; the grouped arm calls `moeVerifyRows` directly and would skip it.
    if (self.dist_tp != null) return false;
    if (self.qwen4 == null or !mlx.streamIsGpu(self.s) or !moeRouterFusedEnabled() or
        token_rows.len < 2 or token_rows.len > 8 or token_rows.len != ctxs.len or
        self.config.hidden_size < 32 or self.config.num_experts_per_tok >= 32)
    {
        return false;
    }
    if (Trunk.qwen4_trace != null or profileFwdOn()) return false;
    const si = qwen4Standin();
    if (si.gdn or si.attn or si.mlp or si.gdn_recur or si.gdn_proj or si.attn_qsa or si.attn_sdpa or si.hc or si.moe_shared or si.moe_router or si.moe_gateup or si.moe_down) return false;
    for (self.moe_layers) |lw| switch (lw.mlp) {
        .moe => |mw| if (mw.expert_bias != null or mw.router_scale != null or mw.per_expert_scale != null) return false,
        else => return false,
    };
    var total_tokens: u64 = 0;
    for (token_rows, ctxs) |tokens, ctx| {
        const shape = mlx.getShape(tokens);
        if (shape.len != 2 or shape[0] != 1 or shape[1] < 1 or ctx.ssm_entries == null or ctx.mrope_pos != null or ctx.vision_embeddings != null or
            ctx.capture_hidden != null or ctx.capture_hidden_all != null or ctx.capture_stream_all != null or ctx.capture_layers != null or ctx.batch_slots != null)
        {
            return false;
        }
        total_tokens += @intCast(shape[1]);
    }
    if (self.config.num_experts == 0 or self.config.num_experts_per_tok == 0) return false;
    return (total_tokens * self.config.num_experts_per_tok) / self.config.num_experts < 4;
}

pub fn forwardQwen4VerifyRows(
    self: *Trunk,
    token_rows: []const mlx.mlx_array,
    ctxs: []const *ForwardCtx,
    out_logits: []mlx.mlx_array,
    out_last: []mlx.mlx_array,
    out_all: []mlx.mlx_array,
) !void {
    const op_start = mlx.op_count.load(.monotonic);
    const cfg = &self.config;
    const layers = self.moe_layers;
    const hc: c_int = @intCast(cfg.hc_count);
    const Row = struct {
        ctx: *ForwardCtx,
        tokens: mlx.mlx_array,
        entries: []SSMCacheEntry,
        offset: usize,
        seq: c_int,
        generation: u64,
        h: mlx.mlx_array,
        pending: ?HcPending = null,
    };
    var rows: [MAX_BATCH_ROWS]Row = undefined;
    var initialized: usize = 0;
    defer for (rows[0..initialized]) |*row| {
        if (row.pending) |*pending| pending.deinit();
        if (row.h.ctx != null) _ = mlx.mlx_array_free(row.h);
    };
    var generation = self.fwd_gen;
    for (token_rows, ctxs, 0..) |tokens, ctx, i| {
        generation +%= 1;
        self.fwd_gen = generation;
        const seq = mlx.getShape(tokens)[1];
        const emb = try self.embedding(tokens);
        defer _ = mlx.mlx_array_free(emb);
        var h = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_tile(&h, emb, &.{ 1, 1, hc }, 3, self.s));
        errdefer _ = mlx.mlx_array_free(h);
        if (self.qwen4_stream_f32 or env.onCached(&qwen4_stream_f32_env, .QWEN4_STREAM_F32)) {
            var h32 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&h32, h, .float32, self.s));
            _ = mlx.mlx_array_free(h);
            h = h32;
        }
        rows[i] = .{
            .ctx = ctx,
            .tokens = tokens,
            .entries = ctx.ssm_entries.?,
            .offset = ctx.moe_seq_offset.*,
            .seq = seq,
            .generation = generation,
            .h = h,
        };
        initialized += 1;
    }

    self.spec_capture_ssm = true;
    defer self.spec_capture_ssm = false;
    for (0..layerCap(cfg.num_hidden_layers)) |layer_idx| {
        const layer = &layers[layer_idx];
        var group_projection = true;
        if (group_projection) {
            var total: c_int = 0;
            for (rows[0..initialized]) |row| {
                group_projection = group_projection and row.seq >= 2 and row.seq <= 6;
                total += row.seq;
            }
            group_projection = group_projection and total <= 12;
        }
        var attn_reads: [MAX_BATCH_ROWS]HcRead = undefined;
        var attn_count: usize = 0;
        defer for (attn_reads[0..attn_count]) |*read| read.deinit();
        var attn_inputs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        var attn_entries: [MAX_BATCH_ROWS]*SSMCacheEntry = undefined;
        var generations: [MAX_BATCH_ROWS]u64 = undefined;
        for (rows[0..initialized], 0..) |*row, i| {
            self.fwd_gen = row.generation;
            if (layer.ple) |*pw| try self.addVerifyPle(row, pw, layer_idx, i);
        }
        const joined_attn = try self.hcReadVerifyRows(rows[0..initialized], &layer.hc_attn.?);
        defer if (joined_attn) |values| {
            for (values) |*value| if (value.mixed.ctx != null) value.deinit();
            self.allocator.free(values);
        };
        for (rows[0..initialized], 0..) |*row, i| {
            self.fwd_gen = row.generation;
            var read = if (joined_attn) |values| blk: {
                const value = values[i];
                values[i] = .{ .mixed = .{}, .inj = .{} };
                break :blk value;
            } else try self.hcReadPending(&row.h, &layer.hc_attn.?, 1, row.seq, &row.pending);
            if (group_projection) {
                attn_reads[i] = read;
                attn_count += 1;
                attn_inputs[i] = read.mixed;
                attn_entries[i] = &row.entries[layer_idx];
                generations[i] = row.generation;
            } else {
                defer read.deinit();
                const attn = switch (layer.attn) {
                    .linear => |la| try self.gatedDeltaNet(read.mixed, &la, &row.entries[layer_idx], layer_idx, 1, row.seq, true),
                    .full => |fa| try self.qwen4AttnWith(row.ctx, read.mixed, &fa, &row.entries[layer_idx], @intCast(layer_idx), @intCast(row.offset), 0, 1, row.seq, true),
                };
                defer _ = mlx.mlx_array_free(attn);
                try self.hcWriteOrDefer(&row.h, attn, read.inj, 1, row.seq, .{ .ctx = null }, &row.pending);
            }
        }
        if (group_projection) {
            const projected = switch (layer.attn) {
                .linear => |*la| try self.gdnVerifyRows(attn_inputs[0..initialized], attn_entries[0..initialized], generations[0..initialized], layer_idx, la),
                .full => |*fa| try self.attnVerifyRows(attn_inputs[0..initialized], attn_entries[0..initialized], ctxs, generations[0..initialized], layer_idx, fa),
            };
            defer if (projected) |values| {
                for (values) |value| _ = mlx.mlx_array_free(value);
                self.allocator.free(values);
            };
            for (rows[0..initialized], attn_reads[0..initialized], 0..) |*row, read, i| {
                self.fwd_gen = row.generation;
                const attn = if (projected) |values| try standinRef(values[i]) else switch (layer.attn) {
                    .linear => |*la| try self.gatedDeltaNet(read.mixed, la, &row.entries[layer_idx], layer_idx, 1, row.seq, true),
                    .full => |*fa| try self.qwen4AttnWith(row.ctx, read.mixed, fa, &row.entries[layer_idx], @intCast(layer_idx), @intCast(row.offset), 0, 1, row.seq, true),
                };
                defer _ = mlx.mlx_array_free(attn);
                try self.hcWriteOrDefer(&row.h, attn, read.inj, 1, row.seq, .{ .ctx = null }, &row.pending);
            }
        }

        var reads: [MAX_BATCH_ROWS]HcRead = undefined;
        var read_count: usize = 0;
        defer for (reads[0..read_count]) |*read| read.deinit();
        var inputs: [MAX_BATCH_ROWS]mlx.mlx_array = undefined;
        const joined_mlp = try self.hcReadVerifyRows(rows[0..initialized], &layer.hc_mlp.?);
        defer if (joined_mlp) |values| {
            for (values) |*value| if (value.mixed.ctx != null) value.deinit();
            self.allocator.free(values);
        };
        for (rows[0..initialized], 0..) |*row, i| {
            self.fwd_gen = row.generation;
            reads[i] = if (joined_mlp) |values| blk: {
                const value = values[i];
                values[i] = .{ .mixed = .{}, .inj = .{} };
                break :blk value;
            } else try self.hcReadPending(&row.h, &layer.hc_mlp.?, 1, row.seq, &row.pending);
            read_count += 1;
            inputs[i] = reads[i].mixed;
        }
        const mw = switch (layer.mlp) {
            .moe => |*weights| weights,
            else => unreachable,
        };
        const mlp = try self.moeVerifyRows(inputs[0..initialized], mw);
        defer {
            for (mlp) |value| _ = mlx.mlx_array_free(value);
            self.allocator.free(mlp);
        }
        for (rows[0..initialized], mlp, reads[0..initialized]) |*row, value, read| {
            self.fwd_gen = row.generation;
            try self.hcWriteOrDefer(&row.h, value, read.inj, 1, row.seq, .{ .ctx = null }, &row.pending);
        }
    }

    var join_lmhead = true;
    var lmhead_widths: [MAX_BATCH_ROWS]c_int = undefined;
    if (join_lmhead) {
        var total: c_int = 0;
        for (rows[0..initialized], 0..) |row, i| {
            join_lmhead = join_lmhead and !row.ctx.skip_lm_head and row.seq >= 2 and row.seq <= 6;
            lmhead_widths[i] = row.seq;
            total += row.seq;
        }
        join_lmhead = join_lmhead and total <= 12;
    }
    var lmhead_inputs: [MAX_BATCH_ROWS]mlx.mlx_array = @splat(.{});
    defer for (lmhead_inputs[0..initialized]) |value| {
        if (value.ctx != null) _ = mlx.mlx_array_free(value);
    };
    var completed: usize = 0;
    errdefer for (0..completed) |i| {
        _ = mlx.mlx_array_free(out_logits[i]);
        _ = mlx.mlx_array_free(out_last[i]);
        _ = mlx.mlx_array_free(out_all[i]);
    };
    for (rows[0..initialized], 0..) |*row, i| {
        self.fwd_gen = row.generation;
        try self.hcFlush(&row.h, 1, row.seq, &row.pending);
        row.ctx.moe_seq_offset.* += @intCast(row.seq);
        var all = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&all, row.h));
        errdefer _ = mlx.mlx_array_free(all);
        const shape = mlx.getShape(row.h);
        var last = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&last, row.h, &.{ 0, shape[1] - 1, 0 }, 3, &.{ 1, shape[1], shape[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
        errdefer _ = mlx.mlx_array_free(last);
        const mix = try self.hcReadPending(&row.h, &self.qwen4_mixer, 1, row.seq, &row.pending);
        _ = mlx.mlx_array_free(row.h);
        row.h = .{ .ctx = null };
        if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
        const logits: mlx.mlx_array = if (join_lmhead) blk: {
            lmhead_inputs[i] = mix.mixed;
            break :blk .{};
        } else if (row.ctx.skip_lm_head)
            mix.mixed
        else blk: {
            defer _ = mlx.mlx_array_free(mix.mixed);
            break :blk try self.lmHeadProject(mix.mixed);
        };
        out_logits[i] = logits;
        out_last[i] = last;
        out_all[i] = all;
        completed += 1;
    }
    if (join_lmhead) {
        const vec = mlx.mlx_vector_array_new_data(&lmhead_inputs, initialized);
        defer _ = mlx.mlx_vector_array_free(vec);
        var joined = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(joined);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, self.s));
        if (try verifyLmHeadProjection(self.s, joined, self.lm_head_w, self.lm_head_s, self.lm_head_b, lmhead_widths[0..initialized])) |logits| {
            // The block stays published for the accept side; `out_logits`
            // are views of it either way.
            self.verify_joined_logits = logits;
            var offset: c_int = 0;
            for (rows[0..initialized], 0..) |row, i| {
                try mlx.check(mlx.mlx_slice(&out_logits[i], logits, &.{ 0, offset, 0 }, 3, &.{ 1, offset + row.seq, mlx.getShape(logits)[2] }, 3, &.{ 1, 1, 1 }, 3, self.s));
                offset += row.seq;
            }
            verify_projection.mtp_verify_lmhead_rows_calls +%= 1;
            const Once = struct {
                var logged = false;
            };
            if (!Once.logged) {
                Once.logged = true;
                log.info("[batched] vocabulary verify projection engaged (slots={d})\n", .{initialized});
            }
        } else {
            for (0..initialized) |i| out_logits[i] = try self.lmHeadProject(lmhead_inputs[i]);
        }
    }
    self.fwd_gen = generation;
    const group_ops = mlx.op_count.load(.monotonic) - op_start;
    mtp.mtp_verify_moe_group_graphs +%= 1;
    mtp.mtp_verify_moe_group_last_ops = group_ops;
}

/// Qwen3.8-Flash-Next forward: embeddings tiled into `hc` residual
/// streams; every block reads a gated mix of the streams and writes back
/// through per-stream scalar gates; the n-gram PLE adds to the streams
/// before its layer; the final mixer replaces model.norm.
/// A later pipeline rank's forward: its layers over the `[B, L, hc*hidden]`
/// stream the previous rank handed over, returning the stream for the next one.
pub fn forwardQwen4Stream(self: *Trunk, ctx: *ForwardCtx, stream: mlx.mlx_array) !mlx.mlx_array {
    ctx.stream_in = stream;
    defer ctx.stream_in = null;
    return self.forwardQwen4With(ctx, .{ .ctx = null });
}

pub fn forwardQwen4With(self: *Trunk, ctx: *ForwardCtx, token_ids: mlx.mlx_array) !mlx.mlx_array {
    const stream_in = ctx.stream_in;
    self.fwd_gen +%= 1; // per-forward QSA scratch key
    const ml = self.moe_layers;
    const cfg = &self.config;
    const offset = ctx.moe_seq_offset.*;
    const entries = ctx.ssm_entries orelse return error.MissingSsmEntries;
    const hc: c_int = @intCast(cfg.hc_count);

    self.spec_capture_ssm = ctx.capture_ssm_seq;
    defer self.spec_capture_ssm = false;

    var h = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(h);
    if (stream_in) |st| {
        try mlx.check(mlx.mlx_array_set(&h, st));
    } else {
        // Vision rows splice into the `hidden`-wide embeddings BEFORE the
        // hyper-connection tile (the reference masked_scatters, then repeats).
        var emb = try self.embedding(token_ids);
        {
            errdefer _ = mlx.mlx_array_free(emb);
            emb = try self.applyVisionEmbeddingsWith(ctx, emb, token_ids);
        }
        defer _ = mlx.mlx_array_free(emb);
        const reps = [_]c_int{ 1, 1, hc };
        try mlx.check(mlx.mlx_tile(&h, emb, &reps, 3, self.s));
    }
    const x_shape = mlx.getShape(h);
    const batch: c_int = x_shape[0];
    const seq_len: c_int = x_shape[1];
    const is_prefill = seq_len > 1;

    if (self.qwen4_stream_f32 or env.onCached(&qwen4_stream_f32_env, .QWEN4_STREAM_F32)) {
        var h32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&h32, h, .float32, self.s));
        _ = mlx.mlx_array_free(h);
        h = h32;
    }
    // M-RoPE chunk tables: read by every full-attn layer AND the QSA
    // indexer's queries (its pooled block keys take a strided build).
    try self.beginMropeChunk(ctx, @intCast(offset), @intCast(seq_len), mlx.mlx_array_dtype(h));
    defer endMropeChunk(ctx);
    var dt = mlx.DtypeTrace.begin("qwen4", h, switch (ml[0].attn) {
        .full => |f| f.q_w,
        .linear => |la| la.qkv_w,
    });

    const eval_cadence = prefillEvalCadenceFor(
        PREFILL_EVAL_CADENCE_DEFAULT,
        cfg,
        @intCast(seq_len),
        @as(u64, @intCast(offset)) + @as(u64, @intCast(seq_len)),
        ctx.cache.config.scheme != .off,
    );

    var prof = try Qwen4FwdProf.init(seq_len, h);
    var pending: ?HcPending = null;
    defer if (pending) |*pd| pd.deinit();

    const ladder = if (ctx.batch_slots != null) self.decode_async_ladder_qwen4 else self.decode_async_ladder;
    var ladder_blocked = false;
    for (0..layerCap(cfg.num_hidden_layers)) |layer_idx| {
        const li: u32 = @intCast(layer_idx);
        const lw = &ml[layer_idx];
        const entry = &entries[layer_idx];

        if (lw.ple) |*pw| {
            try self.hcFlush(&h, batch, seq_len, &pending);
            const add = try self.pleForward(ctx, h, token_ids, pw, entry, layer_idx, batch, seq_len);
            defer _ = mlx.mlx_array_free(add);
            var h_ple = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_add(&h_ple, h, add, self.s));
            _ = mlx.mlx_array_free(h);
            h = h_ple;
            try prof.lap(h, .ple);
        }

        const si = qwen4Standin();
        var pre: HcRead = if (si.hc) try self.hcReadStandin(h, batch, seq_len) else try self.hcReadPending(&h, &lw.hc_attn.?, batch, seq_len, &pending);
        defer pre.deinit();
        try prof.lap(pre.mixed, .hc_read);
        const attn_out = switch (lw.attn) {
            .linear => |la| if (si.gdn) try standinRef(pre.mixed) else try self.gatedDeltaNet(pre.mixed, &la, entry, layer_idx, batch, seq_len, is_prefill),
            .full => |fa| if (si.attn) try standinRef(pre.mixed) else try self.qwen4AttnWith(ctx, pre.mixed, &fa, entry, li, @intCast(offset), 0, batch, seq_len, is_prefill),
        };
        defer _ = mlx.mlx_array_free(attn_out);
        try prof.lap(attn_out, if (lw.attn == .linear) .gdn else .attn);
        if (layer_idx == 3) if (Trunk.qwen4_trace) |tr| Qwen4Trace.set(&tr.attn3_out, attn_out);
        if (layer_idx == 0) if (Trunk.qwen4_trace) |tr| {
            Qwen4Trace.set(&tr.mixed_attn, pre.mixed);
            Qwen4Trace.set(&tr.inj_attn, pre.inj);
            Qwen4Trace.set(&tr.attn_out, attn_out);
        };
        if (si.hc) h = try self.hcWrite(h, attn_out, pre.inj, batch, seq_len) else try self.hcWriteOrDefer(&h, attn_out, pre.inj, batch, seq_len, lw.hc_mlp.?.inject_flat, &pending);
        if (prof.timing) try self.hcFlush(&h, batch, seq_len, &pending);
        try prof.lap(h, .hc_write);

        var pre2: HcRead = if (si.hc) try self.hcReadStandin(h, batch, seq_len) else try self.hcReadPending(&h, &lw.hc_mlp.?, batch, seq_len, &pending);
        defer pre2.deinit();
        try prof.lap(pre2.mixed, .hc_read);
        const mlp_out = if (si.mlp) try standinRef(pre2.mixed) else switch (lw.mlp) {
            .moe => |*mw| if (self.dist_tp) |tp| try self.moeTensorParallel(tp, pre2.mixed, mw, li) else try self.moeMLP(pre2.mixed, mw),
            .dense => |*dw| try self.denseMLP(pre2.mixed, dw),
        };
        defer _ = mlx.mlx_array_free(mlp_out);
        try prof.lap(mlp_out, .mlp);
        if (layer_idx == 0) if (Trunk.qwen4_trace) |tr| {
            Qwen4Trace.set(&tr.mixed_mlp, pre2.mixed);
            Qwen4Trace.set(&tr.inj_mlp, pre2.inj);
            Qwen4Trace.set(&tr.mlp_out, mlp_out);
        };
        if (si.hc) h = try self.hcWrite(h, mlp_out, pre2.inj, batch, seq_len) else try self.hcWriteOrDefer(&h, mlp_out, pre2.inj, batch, seq_len, if (layer_idx + 1 < layerCap(cfg.num_hidden_layers)) ml[layer_idx + 1].hc_attn.?.inject_flat else .{ .ctx = null }, &pending);
        if (prof.timing) try self.hcFlush(&h, batch, seq_len, &pending);
        try prof.lap(h, .hc_write);
        prof.endLayer(if (lw.ple != null) .ple else if (lw.attn == .linear) .gdn else .attn);
        if (ladder != 0 and seq_len == 1 and (layer_idx + 1) % ladder == 0 and !ladder_blocked) {
            // A host-filled PLE leaf reads as zeros until `flushDeferredPle`, so fill it before the first early
            // eval; while its ids are still a lazy sample, skip the ladder for this forward rather than sync.
            if (ctx.ple_pending) |pp| if (!pp.gpu) {
                var ids_avail = false;
                if (mlx._mlx_array_is_available(&ids_avail, pp.token_ids) == 0 and ids_avail) try self.flushDeferredPle(ctx) else ladder_blocked = true;
            };
            // `mlp_out` rides along: the HC write that folds it into `h` is deferred to the next read.
            if (!ladder_blocked) ladderStepMulti(&.{ h, mlp_out }, ladder, layer_idx, seq_len);
        }

        if (ctx.capture_layers) |cl| {
            for (cl.ids, cl.out) |cid, *slot| {
                if (cid == li) {
                    try self.hcFlush(&h, batch, seq_len, &pending);
                    _ = mlx.mlx_array_set(slot, h);
                }
            }
        }
        if (is_prefill and prefillEvalCadenceApplies(seq_len) and ((layer_idx + 1) % eval_cadence == 0 or layer_idx + 1 == layerCap(cfg.num_hidden_layers))) {
            try evalCadencePoint(h, ctx.ssm_entries);
        }
        dt.layer(h, layer_idx);
    }

    if (stream_in != null or self.dist_peer != null) try self.hcFlush(&h, batch, seq_len, &pending);
    if (stream_in != null) {
        ctx.moe_seq_offset.* += @intCast(seq_len);
        return h;
    }
    if (self.dist_peer) |peer| {
        // The hop's sends are queued before its receive, so the PLE leaf must be filled now.
        if (ctx.ple_pending != null) try self.flushDeferredPle(ctx);
        const back = try peer.hop(h, offset);
        _ = mlx.mlx_array_free(h);
        h = back;
    }
    if (ctx.capture_stream_all != null or ctx.capture_hidden != null or ctx.capture_hidden_all != null) try self.hcFlush(&h, batch, seq_len, &pending);
    ctx.moe_seq_offset.* += @intCast(seq_len);
    dt.end(h);
    prof.report(seq_len, @as(usize, @intCast(offset)) + @as(usize, @intCast(seq_len)), ctx.capture_ssm_seq, cfg.num_hidden_layers - cfg.attnCacheLayerCount(), cfg.attnCacheLayerCount());
    Qwen4AttnProf.flush(seq_len, @as(usize, @intCast(offset)) + @as(usize, @intCast(seq_len)));
    if (ctx.capture_stream_all) |target| _ = mlx.mlx_array_set(target, h);
    // On this arch the spec "hidden" IS the pre-mixer stream: the MTP head
    // consumes `[B, L, hc*hidden]`, never the mixed 2560 (vLLM/SGLang).
    if (ctx.capture_hidden) |target| {
        const hs = mlx.getShape(h);
        const last = hs[1] - 1;
        const start = [_]c_int{ 0, last, 0 };
        const stop = [_]c_int{ hs[0], hs[1], hs[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sliced, h, &start, 3, &stop, 3, &strides, 3, self.s));
        _ = mlx.mlx_array_set(target, sliced);
        _ = mlx.mlx_array_free(sliced);
    }
    if (ctx.capture_hidden_all) |target_all| _ = mlx.mlx_array_set(target_all, h);

    const mix = try self.hcReadPending(&h, &self.qwen4_mixer, batch, seq_len, &pending);
    // The function-scope errdefer reads `h` at unwind time, so surrender the handle where it is released.
    _ = mlx.mlx_array_free(h);
    h = .{ .ctx = null };
    if (mix.inj.ctx != null) _ = mlx.mlx_array_free(mix.inj);
    const final = mix.mixed;
    if (ctx.skip_lm_head) return final;
    errdefer _ = mlx.mlx_array_free(final);
    const logits = try self.lmHeadProject(final);
    _ = mlx.mlx_array_free(final);
    return logits;
}

/// Forward-pass diagnostics for qwen4_exp. QWEN4_PROFILE_FWD=1 (S 2..16) or
/// =all: per-block GPU ms with a sync per block, never on by default;
/// QWEN4_OPCOUNT=1: FFI op attribution per block, no syncs.
var qwen4_stream_f32_env: ?bool = null;

// ── tests ──

// ── qwen4_exp fixture parity (env-gated: QWEN4_TEST_MODEL + QWEN4_FIXTURE) ──

test "qwen4 fixture vision: tower parity, pre-tile splice + M-RoPE prefill/decode, QSA mask at 3-D block positions (QWEN4_FIXTURE_VISION)" {
    // `tests/dump_qwen4_exp_fixtures.py build|dump --vision` + the converter's
    // `--add-vision --src`. Proves, together: the tower on this arch's names,
    // the splice BEFORE the hyper-connection tile, the M-RoPE chunk tables
    // reaching attention AND the QSA indexer, and the indexer's pooled block
    // keys roped at 3-D block-start positions (the layer-3 mask is EXACT).
    const qwen_vision = @import("../../sdk/vision/qwen_vision.zig");
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("QWEN4_FIXTURE_VISION") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    try testing.expect(config.qwen_vision and config.image_token_id != 0);
    var weights = try model_mod.loadWeightsWithVision(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.qwen4_stream_f32 = true;
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var qv = try qwen_vision.QwenVision.init(allocator, .of(&config), &weights);
    defer qv.deinit();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: usize = mlx.mlx_array_size(ids_arr);
    const ids_src = mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable;
    const v: usize = @intCast(config.vocab_size);
    const ref_full = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_full);
    const ref_last = try qwen4ReadF32(allocator, fx.get("logits_prefill_last") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_last);
    const ref_dec = try qwen4ReadF32(allocator, fx.get("logits_decode") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_dec);
    const ties = try Qwen4Ties.read(allocator, &fx, s);
    defer ties.deinit(allocator);
    const t_dec: usize = ref_dec.len / v;
    const t_pre: usize = T - t_dec;
    // Our own full-forward logits, the bar for the chunked arm below.
    const ours_full = try allocator.alloc(f32, T * v);
    defer allocator.free(ours_full);

    // [a] tower: pixel_values → post-merger rows vs the reference pooler_output.
    const grid_arr = fx.get("image_grid_thw") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(grid_arr));
    const grid = mlx.mlx_array_data_int32(grid_arr) orelse return error.Unreadable;
    const gt: u32 = @intCast(grid[0]);
    const gh: u32 = @intCast(grid[1]);
    const gw: u32 = @intCast(grid[2]);
    const px = fx.get("pixel_values") orelse return error.MissingFixtureTensor;
    const emb = try qv.forward(px, gh, gw); // [1, N/merge², out_hidden]
    defer _ = mlx.mlx_array_free(emb);
    const ref_emb = try qwen4ReadF32(allocator, fx.get("image_embeds") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_emb);
    const our_emb = try qwen4ReadF32(allocator, emb, s);
    defer allocator.free(our_emb);
    try testing.expectEqual(ref_emb.len, our_emb.len);
    const hid: usize = @intCast(config.qv_out_hidden);
    const n_img: usize = ref_emb.len / hid;
    const re = qwen4CompareRows(our_emb, ref_emb, n_img, hid);
    std.debug.print("[qwen4 fixture vision] tower {d}x{d}x{d} -> {d} rows: min row cos {d:.5}\n", .{ gt, gh, gw, n_img, re.min_cos });
    try testing.expect(re.min_cos > 0.99);

    // [b] 3-D positions: our get_rope_index == the reference table + delta.
    const ref_pos_arr = fx.get("position_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ref_pos_arr));
    const ref_pos = mlx.mlx_array_data_int32(ref_pos_arr) orelse return error.Unreadable;
    const ref_delta_arr = fx.get("rope_delta") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ref_delta_arr));
    const ref_delta = (mlx.mlx_array_data_int32(ref_delta_arr) orelse return error.Unreadable)[0];
    const host_ids = try allocator.alloc(u32, T);
    defer allocator.free(host_ids);
    for (0..T) |i| host_ids[i] = @intCast(ids_src[i]);
    const grids = [_]mrope.ImageGrid{.{ .t = gt, .h = gh, .w = gw }};
    var ri = try mrope.getRopeIndex(allocator, host_ids, &grids, &.{}, config.image_token_id, config.video_token_id, config.vision_start_token_id, config.qv_merge);
    defer ri.deinit();
    try testing.expectEqual(ref_delta, ri.delta);
    const pos_flat = try allocator.alloc(i32, 3 * T);
    defer allocator.free(pos_flat);
    for (0..3) |axis| for (0..T) |i| {
        pos_flat[axis * T + i] = ri.pos[axis][i];
        try testing.expectEqual(ref_pos[axis * T + i], ri.pos[axis][i]);
    };
    var n_pad_total: usize = 0;
    for (host_ids) |t| {
        if (t == config.image_token_id) n_pad_total += 1;
    }
    try testing.expectEqual(n_img, n_pad_total);

    // [c] one full-prompt forward: streams, final logits, layer-3 QSA mask EXACT.
    {
        const shape = [_]c_int{ 1, @intCast(T) };
        const ids = mlx.mlx_array_new_data(@ptrCast(ids_src), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        const ctx_slot = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
        defer ctx_slot.deinit(allocator);
        var ctx = ctx_slot.ctx;
        ctx.vision_embeddings = emb;
        ctx.mrope_pos = pos_flat;
        ctx.mrope_total = T;
        ctx.mrope_delta = ri.delta;
        const n_layers: usize = config.num_hidden_layers;
        const cap_ids = try allocator.alloc(u32, n_layers);
        defer allocator.free(cap_ids);
        const cap_out = try allocator.alloc(mlx.mlx_array, n_layers);
        defer allocator.free(cap_out);
        for (0..n_layers) |i| {
            cap_ids[i] = @intCast(i);
            cap_out[i] = mlx.mlx_array_new();
        }
        defer for (cap_out) |o| {
            _ = mlx.mlx_array_free(o);
        };
        var cap_layers: CaptureLayers = .{ .ids = cap_ids, .out = cap_out };
        ctx.capture_layers = &cap_layers;
        var final_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(final_all);
        ctx.capture_hidden_all = &final_all;
        var trace: Trunk.Qwen4Trace = .{};
        defer trace.deinit();
        Trunk.qwen4_trace = &trace;
        defer Trunk.qwen4_trace = null;
        const logits = try xfm.forwardWith(&ctx, ids);
        defer _ = mlx.mlx_array_free(logits);
        for (0..n_layers) |i| {
            var nb: [32]u8 = undefined;
            const key = std.fmt.bufPrint(&nb, "stream_{d}", .{i + 1}) catch unreachable;
            const ref = fx.get(key) orelse continue;
            if (i + 1 >= n_layers or cap_out[i].ctx == null) continue;
            const ref_h = try qwen4ReadF32(allocator, ref, s);
            defer allocator.free(ref_h);
            const our_h = try qwen4ReadF32(allocator, cap_out[i], s);
            defer allocator.free(our_h);
            if (our_h.len != ref_h.len) {
                std.debug.print("[qwen4 fixture vision] stream_{d}: size {d} vs ref {d}\n", .{ i + 1, our_h.len, ref_h.len });
                continue;
            }
            const w: usize = ref_h.len / T;
            const rc = qwen4CompareRows(our_h, ref_h, T, w);
            std.debug.print("[qwen4 fixture vision] after layer {d}: min row cos {d:.5}\n", .{ i, rc.min_cos });
            if (rc.min_cos < 0.999 and i == 3) {
                for (0..T) |row| {
                    const rr = qwen4CompareRows(our_h[row * w ..][0..w], ref_h[row * w ..][0..w], 1, w);
                    std.debug.print("    row {d}: cos {d:.4}\n", .{ row, rr.min_cos });
                }
            }
        }
        // The reference's own cos/sin table [T, rope_dims] vs our chunk build.
        if (fx.get("rope_cos")) |ref_cos_arr| {
            const ref_cos = try qwen4ReadF32(allocator, ref_cos_arr, s);
            defer allocator.free(ref_cos);
            const ours_cs = try xfm.mropeCosSinAt(.{ .pos = pos_flat, .total = T, .delta = ri.delta }, 0, 1, T, .float32);
            defer _ = mlx.mlx_array_free(ours_cs.cos);
            defer _ = mlx.mlx_array_free(ours_cs.sin);
            const our_cos = try qwen4ReadF32(allocator, ours_cs.cos, s);
            defer allocator.free(our_cos);
            try testing.expectEqual(ref_cos.len, our_cos.len);
            var max_d: f32 = 0;
            for (ref_cos, our_cos) |a, b| max_d = @max(max_d, @abs(a - b));
            std.debug.print("[qwen4 fixture vision] M-RoPE cos table vs reference: max |d| {e:.3}\n", .{max_d});
            try testing.expect(max_d < 1e-4);
        }
        if (fx.get("l3_attn_out")) |ref_a| if (trace.attn3_out.ctx != null) {
            const ref_h = try qwen4ReadF32(allocator, ref_a, s);
            defer allocator.free(ref_h);
            const our_h = try qwen4ReadF32(allocator, trace.attn3_out, s);
            defer allocator.free(our_h);
            if (our_h.len == ref_h.len) {
                const w: usize = ref_h.len / T;
                const rc = qwen4CompareRows(our_h, ref_h, T, w);
                std.debug.print("[qwen4 fixture vision] l3 attn out: min row cos {d:.5}\n", .{rc.min_cos});
                if (rc.min_cos < 0.999) for (0..T) |row| {
                    const rr = qwen4CompareRows(our_h[row * w ..][0..w], ref_h[row * w ..][0..w], 1, w);
                    std.debug.print("    attn row {d}: cos {d:.4}\n", .{ row, rr.min_cos });
                };
            } else std.debug.print("[qwen4 fixture vision] l3 attn out: size {d} vs ref {d}\n", .{ our_h.len, ref_h.len });
        };
        // The mask is the direct guard for the pooled-key angles: a scalar
        // rope on the image blocks picks different blocks.
        const ref_mask = try qwen4ReadF32(allocator, fx.get("l3_qsa_mask") orelse return error.MissingFixtureTensor, s);
        defer allocator.free(ref_mask);
        try testing.expect(trace.qsa_mask.ctx != null);
        const our_mask = try qwen4ReadF32(allocator, trace.qsa_mask, s);
        defer allocator.free(our_mask);
        try testing.expectEqual(ref_mask.len, our_mask.len);
        var mask_bad: usize = 0;
        var mask_tied_rows: usize = 0;
        for (0..T) |row| {
            if (ties.rowTied(row)) {
                mask_tied_rows += 1;
                continue;
            }
            for (0..T) |c| {
                if (ref_mask[row * T + c] != our_mask[row * T + c]) mask_bad += 1;
            }
        }
        std.debug.print("[qwen4 fixture vision] layer-3 QSA mask: {d} mismatched over {d} decisive rows ({d} acquitted)\n", .{ mask_bad, T - mask_tied_rows, mask_tied_rows });
        try testing.expect(mask_tied_rows < T / 2);
        try testing.expectEqual(@as(usize, 0), mask_bad);
        if (fx.get("stream_pre_mixer")) |ref_pm| {
            const ref_h = try qwen4ReadF32(allocator, ref_pm, s);
            defer allocator.free(ref_h);
            const our_h = try qwen4ReadF32(allocator, final_all, s);
            defer allocator.free(our_h);
            if (our_h.len == ref_h.len) {
                const w: usize = ref_h.len / T;
                const rc = qwen4CompareRows(our_h, ref_h, T, w);
                std.debug.print("[qwen4 fixture vision] pre-mixer stream: min row cos {d:.5}\n", .{rc.min_cos});
                if (rc.min_cos < 0.999) for (0..T) |row| {
                    const rr = qwen4CompareRows(our_h[row * w ..][0..w], ref_h[row * w ..][0..w], 1, w);
                    std.debug.print("    row {d}: cos {d:.4}\n", .{ row, rr.min_cos });
                };
            }
        }
        const ours = try qwen4ReadF32(allocator, logits, s);
        defer allocator.free(ours);
        try testing.expectEqual(T * v, ours.len);
        @memcpy(ours_full, ours);
        const r = qwen4CompareTied("logits", ours, ref_full, T, v, 0, ties);
        std.debug.print("[qwen4 fixture vision] full prefill T={d}: min cos {d:.5} ({d} rows acquitted) argmax {d}/{d} decisive rows\n", .{ T, r.min_cos, r.tied, r.agree, r.decisive });
        try testing.expect(r.min_cos > 0.995);
        // A k < E fixture (`build --topk`) ties in most trunk rows across
        // every MoE layer; its selection coverage is the MTP head's ONE
        // MoE layer (decisive floor above). k = E fixtures keep the T/2 floor.
        try testing.expect(r.decisive >= (if (ties.route_gap != null) @as(usize, 2) else T / 2) and r.agree == r.decisive);
    }

    // [d] fresh state: prefill in TWO chunks split inside the image span
    // (`vision_splice_offset` + pooled block keys built across chunks), then
    // decode past the table (positions abs + delta) vs the reference.
    {
        var cache = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache.deinit();
        var off: usize = 0;
        const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = emb };
        ctx.mrope_pos = pos_flat;
        ctx.mrope_total = T;
        ctx.mrope_delta = ri.delta;
        var first_pad: usize = 0;
        while (first_pad < T and host_ids[first_pad] != config.image_token_id) first_pad += 1;
        const t1: usize = first_pad + n_img / 2;
        try testing.expect(t1 < t_pre);
        const c1 = mlx.mlx_array_new_data(@ptrCast(ids_src), &[_]c_int{ 1, @intCast(t1) }, 2, .int32);
        defer _ = mlx.mlx_array_free(c1);
        const l1 = try xfm.forwardWith(&ctx, c1);
        _ = mlx.mlx_array_free(l1);
        ctx.vision_splice_offset = t1 - first_pad;
        const c2 = mlx.mlx_array_new_data(@ptrCast(ids_src + t1), &[_]c_int{ 1, @intCast(t_pre - t1) }, 2, .int32);
        defer _ = mlx.mlx_array_free(c2);
        const plog = try xfm.forwardWith(&ctx, c2);
        defer _ = mlx.mlx_array_free(plog);
        ctx.vision_splice_offset = 0;
        const pours = try qwen4ReadF32(allocator, plog, s);
        defer allocator.free(pours);
        // The chunking invariant is OUR full forward: the split lands inside
        // the image and the last row may be a reference near-tie, but chunked
        // == unchunked is ours to keep (chunk-boundary rounding only).
        const last = pours[(t_pre - t1 - 1) * v ..][0..v];
        const rself = qwen4CompareRows(last, ours_full[(t_pre - 1) * v ..][0..v], 1, v);
        const rp = qwen4CompareTied("prefill-last", last, ref_last, 1, v, t_pre - 1, ties);
        std.debug.print("[qwen4 fixture vision] chunked prefill {d}+{d}: last-row cos vs our full forward {d:.5}, vs reference {d:.5}\n", .{ t1, t_pre - t1, rself.min_cos, rp.min_cos });
        try testing.expect(rself.min_cos > 0.999);
        try testing.expect(rp.min_cos > 0.995);

        var dec: Qwen4TiedCmp = .{ .min_cos = 1.0, .agree = 0, .decisive = 0, .tied = 0 };
        for (0..t_dec) |k| {
            const dshape = [_]c_int{ 1, 1 };
            const did = mlx.mlx_array_new_data(@ptrCast(ids_src + t_pre + k), &dshape, 2, .int32);
            defer _ = mlx.mlx_array_free(did);
            const dlog = try xfm.forwardWith(&ctx, did);
            defer _ = mlx.mlx_array_free(dlog);
            const dours = try qwen4ReadF32(allocator, dlog, s);
            defer allocator.free(dours);
            const rd = qwen4CompareTied("decode", dours, ref_dec[k * v ..][0..v], 1, v, t_pre + k, ties);
            if (rd.min_cos < dec.min_cos) dec.min_cos = rd.min_cos;
            dec.agree += rd.agree;
            dec.decisive += rd.decisive;
            dec.tied += rd.tied;
        }
        std.debug.print("[qwen4 fixture vision] decode x{d}: min cos {d:.5} ({d} acquitted) argmax {d}/{d} decisive rows\n", .{ t_dec, dec.min_cos, dec.tied, dec.agree, dec.decisive });
        try testing.expect(dec.min_cos > 0.995);
        try testing.expect(dec.decisive >= (if (ties.route_gap != null) @as(usize, 0) else 2) and dec.agree == dec.decisive);
    }
}

test "qwen4 fixture bf16: residual streams track a bf16 reference (QWEN4_FIXTURE_BF16)" {
    // `tests/dump_qwen4_exp_fixtures.py` with QWEN4_REF_DTYPE=bf16. The shipped
    // checkpoint runs bf16 residual streams; against the f32 fixture that path
    // reads 0.905 at the logits (bf16 rounding compounds through the tiny
    // random model), against a bf16 reference the streams agree to 0.9998.
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("QWEN4_FIXTURE_BF16") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadModelWeights(io, allocator, std.mem.span(model_dir), &config, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    const ids_shape = mlx.getShape(ids_arr);
    const T: usize = @intCast(ids_shape[ids_shape.len - 1]);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    const tshape = [_]c_int{ 1, @intCast(T) };
    try mlx.check(mlx.mlx_reshape(&ids, ids_arr, &tshape, 2, s));

    const ctx_slot = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
    defer ctx_slot.deinit(allocator);
    var ctx = ctx_slot.ctx;
    const n_layers: usize = config.num_hidden_layers;
    const cap_ids = try allocator.alloc(u32, n_layers);
    defer allocator.free(cap_ids);
    const cap_out = try allocator.alloc(mlx.mlx_array, n_layers);
    defer allocator.free(cap_out);
    for (0..n_layers) |i| {
        cap_ids[i] = @intCast(i);
        cap_out[i] = mlx.mlx_array_new();
    }
    defer for (cap_out) |o| {
        _ = mlx.mlx_array_free(o);
    };
    var cap_layers: CaptureLayers = .{ .ids = cap_ids, .out = cap_out };
    ctx.capture_layers = &cap_layers;
    const logits = try xfm.forwardWith(&ctx, ids);
    defer _ = mlx.mlx_array_free(logits);
    var checked: usize = 0;
    for (1..n_layers) |i| {
        var nb: [32]u8 = undefined;
        const key = std.fmt.bufPrint(&nb, "stream_{d}", .{i}) catch unreachable;
        const ref = fx.get(key) orelse continue;
        const ref_h = try qwen4ReadF32(allocator, ref, s);
        defer allocator.free(ref_h);
        const our_h = try qwen4ReadF32(allocator, cap_out[i - 1], s);
        defer allocator.free(our_h);
        if (our_h.len != ref_h.len) continue;
        const rc = qwen4CompareRows(our_h, ref_h, T, ref_h.len / T);
        std.debug.print("[qwen4 fixture bf16] after layer {d}: min row cos {d:.5}\n", .{ i - 1, rc.min_cos });
        try testing.expect(rc.min_cos > 0.999);
        checked += 1;
    }
    try testing.expect(checked >= n_layers - 2);
    // Logits are printed, not asserted: a tiny random model's logits are
    // chaotic between any two bf16 renderings (0.89 vs bf16-torch measured);
    // the streams above are the bf16 bar.
    if (fx.get("logits_full")) |ref_arr| {
        const ref_full = try qwen4ReadF32(allocator, ref_arr, s);
        defer allocator.free(ref_full);
        const ours = try qwen4ReadF32(allocator, logits, s);
        defer allocator.free(ours);
        const v: usize = @intCast(config.vocab_size);
        if (ours.len == ref_full.len and ref_full.len == T * v) {
            const rc = qwen4CompareRows(ours, ref_full, T, v);
            std.debug.print("[qwen4 fixture bf16] logits (informational): min row cos {d:.4} argmax {d}/{d}\n", .{ rc.min_cos, rc.argmax_agree, T });
        }
    }
}

pub const Qwen4RowCmp = @import("../../sdk/testing/parity.zig").Qwen4RowCmp;
pub const Qwen4Ties = @import("../../sdk/testing/parity.zig").Qwen4Ties;

/// One per-request slot for the qwen4 batched-vs-serial test.
pub const Qwen4TestSlot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []SSMCacheEntry,
    ctx: ForwardCtx = undefined,
    /// The MTP head's per-request stash, so a committed-state bar can assert the trunk
    /// verify left it where the solo round did.
    head: Trunk.Qwen4MtpState,

    pub fn init(alloc: std.mem.Allocator, n_layers: u32) !*Qwen4TestSlot {
        const sl = try alloc.create(Qwen4TestSlot);
        const cache = try KVCache.init(alloc, n_layers);
        const head_cache = try KVCache.init(alloc, 1);
        sl.* = .{
            .cache = cache,
            .entries = try alloc.alloc(SSMCacheEntry, n_layers),
            .head = .{
                .cache = head_cache,
                .entry = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false },
            },
        };
        for (sl.entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        sl.ctx = .{ .cache = &sl.cache, .moe_seq_offset = &sl.off, .ssm_entries = sl.entries, .ssm_member_gen = nextSsmMemberGen(), .capture_hidden = null, .vision_embeddings = null };
        return sl;
    }
    pub fn deinit(sl: *Qwen4TestSlot, alloc: std.mem.Allocator) void {
        for (sl.entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        }
        alloc.free(sl.entries);
        _ = mlx.mlx_array_free(sl.head.entry.conv_state);
        _ = mlx.mlx_array_free(sl.head.entry.ssm_state);
        sl.head.cache.deinit();
        sl.cache.deinit();
        alloc.destroy(sl);
    }
    pub fn forward(sl: *Qwen4TestSlot, xfm: anytype, ids: []const i32) !mlx.mlx_array {
        const shape = [_]c_int{ 1, @intCast(ids.len) };
        const arr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(arr);
        return xfm.forwardWith(&sl.ctx, arr);
    }
};

fn qwen4Rows(arr: mlx.mlx_array) c_int {
    return if (arr.ctx != null) mlx.getShape(arr)[1] else -1;
}

test "qwen4 batched decode: one forwardMoeBatchedDecode tick == two serial ticks (GDN + PLE history + QSA keys/mask)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("QWEN4_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.qwen4_stream_f32 = true;
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: usize = mlx.mlx_array_size(ids_arr);
    const ids = (mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable)[0..T];
    const v: usize = @intCast(config.vocab_size);
    const n_layers = config.num_hidden_layers;

    // Two prompts of DIFFERENT length (pads the shorter slot's KV) whose
    // continuations run every slot past the indexer budget.
    const pa: usize = T / 2;
    const pb: usize = pa - 3;
    const K: usize = T - pa;
    const a_ids = ids[0 .. pa + K];
    const b_ids = ids[3 .. 3 + pb + K];

    const a_s = try Qwen4TestSlot.init(allocator, n_layers);
    defer a_s.deinit(allocator);
    const b_s = try Qwen4TestSlot.init(allocator, n_layers);
    defer b_s.deinit(allocator);
    const a_b = try Qwen4TestSlot.init(allocator, n_layers);
    defer a_b.deinit(allocator);
    const b_b = try Qwen4TestSlot.init(allocator, n_layers);
    defer b_b.deinit(allocator);
    _ = mlx.mlx_array_free(try a_s.forward(&xfm, a_ids[0..pa]));
    _ = mlx.mlx_array_free(try b_s.forward(&xfm, b_ids[0..pb]));
    _ = mlx.mlx_array_free(try a_b.forward(&xfm, a_ids[0..pa]));
    _ = mlx.mlx_array_free(try b_b.forward(&xfm, b_ids[0..pb]));
    std.debug.print("[qwen4 batched] prefill A {d} B {d}: cache.step A {d}, offset A {d}\n", .{ pa, pb, a_s.cache.step, a_s.off });

    var min_cos: f64 = 1.0;
    var checked: usize = 0;
    var agreed: usize = 0;
    for (0..K) |k| {
        const ta = a_ids[pa + k];
        const tb = b_ids[pb + k];
        const la = try a_s.forward(&xfm, &.{ta});
        defer _ = mlx.mlx_array_free(la);
        const lb = try b_s.forward(&xfm, &.{tb});
        defer _ = mlx.mlx_array_free(lb);

        const ctxs = [_]*ForwardCtx{ &a_b.ctx, &b_b.ctx };
        const offs = [_]u32{ @intCast(a_b.off), @intCast(b_b.off) };
        const out = try xfm.forwardMoeBatchedDecode(&.{ @intCast(ta), @intCast(tb) }, &ctxs, &offs, null);
        defer {
            for (out) |o| _ = mlx.mlx_array_free(o);
            allocator.free(out);
        }
        // The scheduler advances each slot's own position after the tick.
        a_b.off += 1;
        b_b.off += 1;
        try testing.expectEqual(a_s.off, a_b.off);
        try testing.expectEqual(b_s.off, b_b.off);

        inline for (.{ la, lb }, .{ out[0], out[1] }, .{ "A", "B" }) |ser, bat, tag| {
            const sh = try qwen4ReadF32(allocator, ser, s);
            defer allocator.free(sh);
            const bh = try qwen4ReadF32(allocator, bat, s);
            defer allocator.free(bh);
            try testing.expectEqual(sh.len, bh.len);
            const rc = qwen4CompareRows(sh, bh, 1, v);
            if (rc.min_cos < min_cos) min_cos = rc.min_cos;
            // argmax must agree wherever the serial top-2 is decisive (3%).
            var top: f32 = -std.math.inf(f32);
            var second: f32 = -std.math.inf(f32);
            for (sh) |x| {
                if (x > top) {
                    second = top;
                    top = x;
                } else if (x > second) second = x;
            }
            if (@abs(top - second) > 0.03 * @abs(top)) {
                checked += 1;
                agreed += rc.argmax_agree;
            }
            if (rc.min_cos < 0.999) std.debug.print("[qwen4 batched] step {d} slot {s}: cos {d:.5} argmax {d}\n", .{ k, tag, rc.min_cos, rc.argmax_agree });
        }
        // Per-slot state moved identically: PLE history, QSA key rows, pooled blocks.
        for (0..n_layers) |l| {
            inline for (.{ .{ a_s, a_b }, .{ b_s, b_b } }) |pair| {
                const es = &pair[0].entries[l];
                const eb = &pair[1].entries[l];
                try testing.expectEqual(es.ple_prev_valid, eb.ple_prev_valid);
                try testing.expectEqualSlices(u32, &es.ple_prev, &eb.ple_prev);
                try testing.expectEqual(qwen4Rows(es.aux_state), qwen4Rows(ssmLiveAux(eb)));
                try testing.expectEqual(qwen4Rows(es.qsa_pooled), qwen4Rows(eb.qsa_pooled));
            }
        }
    }
    std.debug.print("[qwen4 batched] {d} ticks x 2 slots: min cos {d:.5}, argmax {d}/{d} decisive rows\n", .{ K, min_cos, agreed, checked });
    try testing.expect(min_cos > 0.995);
    try testing.expect(checked >= 2 and agreed == checked);
    // Both slots crossed the budget: the batched QSA mask path ran.
    var pooled = false;
    for (b_b.entries) |*e| if (e.qsa_pooled.ctx != null) {
        pooled = true;
    };
    try testing.expect(pooled);
}

test "qwen4 prefill: one lazy 64-token forward gives the same logits every time (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadModelWeights(io, allocator, std.mem.span(model_dir), &config, false);
    defer weights.deinit();
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    // No eval inside the forward: the whole prefill streams through the GPU as one pipeline.
    var toks: [64]i32 = undefined;
    for (&toks, 0..) |*v, i| v.* = @intCast(100 + (i * 37) % 5000);
    var first: ?mlx.mlx_array = null;
    defer if (first) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var differing: usize = 0;
    for (0..5) |_| {
        const fresh = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
        defer fresh.deinit(allocator);
        const ids = mlx.mlx_array_new_data(&toks, &[_]c_int{ 1, 64 }, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        var ctx = fresh.ctx;
        const lg = try xfm.forwardWith(&ctx, ids);
        defer _ = mlx.mlx_array_free(lg);
        var f = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&f, lg, .float32, s));
        try mlx.check(mlx.mlx_array_eval(f));
        if (first) |a| {
            defer _ = mlx.mlx_array_free(f);
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, f, a, false, s));
            try mlx.check(mlx.mlx_array_eval(eq));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            if (!same) differing += 1;
        } else first = f;
    }
    try testing.expectEqual(@as(usize, 0), differing);
}

test "qwen4 deferred PLE: pipelined decode AND MTP verify widths match the direct forward (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    const State = struct {
        cache: KVCache,
        off: usize = 0,
        entries: []SSMCacheEntry,
        fn init(a: std.mem.Allocator, n_layers: u32) !@This() {
            const entries = try a.alloc(SSMCacheEntry, n_layers);
            for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
            const cache = try KVCache.init(a, n_layers);
            return .{ .cache = cache, .entries = entries };
        }
        fn deinit(self: *@This(), a: std.mem.Allocator) void {
            for (self.entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                ssmFreeQsaState(e);
            }
            a.free(self.entries);
            self.cache.deinit();
        }
        fn ctx(self: *@This()) ForwardCtx {
            return .{ .cache = &self.cache, .moe_seq_offset = &self.off, .ssm_entries = self.entries, .capture_hidden = null, .vision_embeddings = null };
        }
    };
    var direct = try State.init(allocator, config.num_hidden_layers);
    defer direct.deinit(allocator);
    var deferred = try State.init(allocator, config.num_hidden_layers);
    defer deferred.deinit(allocator);
    var dctx = direct.ctx();
    var lctx = deferred.ctx();

    const prompt = [_]i32{ 5, 17, 42, 9, 23, 8, 31, 2 };
    const pshape = [_]c_int{ 1, prompt.len };
    for ([_]*ForwardCtx{ &dctx, &lctx }) |c| {
        const pids = mlx.mlx_array_new_data(&prompt, &pshape, 2, .int32);
        defer _ = mlx.mlx_array_free(pids);
        const l = try xfm.forwardWith(c, pids);
        try mlx.check(mlx.mlx_array_eval(l));
        _ = mlx.mlx_array_free(l);
    }

    // Two decode steps: the deferred arm builds on an unfilled leaf, then
    // fills it before eval. Both steps must match bit for bit (the second one
    // proves the n-gram history advanced at flush time).
    const steps = [_]i32{ 11, 4 };
    for (steps) |tok| {
        const dshape = [_]c_int{ 1, 1 };
        const did = mlx.mlx_array_new_data(&tok, &dshape, 2, .int32);
        defer _ = mlx.mlx_array_free(did);
        const dl = try xfm.forwardWith(&dctx, did);
        defer _ = mlx.mlx_array_free(dl);

        lctx.ple_defer = true;
        const ll = try xfm.forwardWith(&lctx, did);
        defer _ = mlx.mlx_array_free(ll);
        lctx.ple_defer = false;
        try testing.expect(lctx.ple_pending != null);
        try xfm.flushDeferredPle(&lctx);
        try testing.expect(lctx.ple_pending == null);

        const a = try qwen4ReadF32(allocator, dl, s);
        defer allocator.free(a);
        const b = try qwen4ReadF32(allocator, ll, s);
        defer allocator.free(b);
        try testing.expectEqualSlices(f32, a, b);
    }

    // Verify-width arm: an MTP round builds `[1, 1+m]` over a chain of LAZY
    // drafts with per-position SSM capture on. Deferred must match eager in
    // logits AND in the rollback state the capture leaves behind — the PLE
    // layer's own conv capture runs later in the same forward and keys on
    // `spec_ple_len`, which the deferred gather has not written yet.
    {
        const rows = [_]i32{ 13, 6, 27, 3, 19, 40, 12 };
        for ([_]usize{ 4, 7 }) |w| {
            const vshape = [_]c_int{ 1, @intCast(w) };
            const vids = mlx.mlx_array_new_data(&rows, &vshape, 2, .int32);
            defer _ = mlx.mlx_array_free(vids);

            dctx.capture_ssm_seq = true;
            const dl = try xfm.forwardWith(&dctx, vids);
            defer _ = mlx.mlx_array_free(dl);
            dctx.capture_ssm_seq = false;

            lctx.capture_ssm_seq = true;
            lctx.ple_defer = true;
            const ll = try xfm.forwardWith(&lctx, vids);
            defer _ = mlx.mlx_array_free(ll);
            lctx.ple_defer = false;
            lctx.capture_ssm_seq = false;
            try testing.expect(lctx.ple_pending != null);
            try xfm.flushDeferredPle(&lctx);
            try testing.expect(lctx.ple_pending == null);

            const a = try qwen4ReadF32(allocator, dl, s);
            defer allocator.free(a);
            const b = try qwen4ReadF32(allocator, ll, s);
            defer allocator.free(b);
            try testing.expectEqualSlices(f32, a, b);

            var saw_ple = false;
            for (direct.entries, deferred.entries) |*de, *le| {
                try testing.expectEqual(de.spec_ple_len, le.spec_ple_len);
                try testing.expectEqual(de.ple_prev_valid, le.ple_prev_valid);
                try testing.expectEqualSlices(u32, &de.ple_prev, &le.ple_prev);
                // A deferred gather that never claimed the slot leaves this
                // null while eager holds a capture: `ssmRollbackFromCapture`
                // would then silently take the non-PLE branch.
                try testing.expectEqual(de.spec_ple_input.ctx == null, le.spec_ple_input.ctx == null);
                if (de.spec_ple_len > 0) {
                    saw_ple = true;
                    try testing.expect(de.spec_ple_len >= @as(u32, @intCast(w)));
                    const n: usize = de.spec_ple_len;
                    try testing.expectEqualSlices(u32, de.spec_ple_tokens[0..n], le.spec_ple_tokens[0..n]);
                    try testing.expect(de.spec_ple_input.ctx != null);
                }
            }
            try testing.expect(saw_ple);
            for (direct.entries) |*e| ssmFreeSpecCapture(e);
            for (deferred.entries) |*e| ssmFreeSpecCapture(e);
        }
    }

    // Load-bearing arm: an unflushed leaf is NOT the direct forward.
    {
        const tok: i32 = 7;
        const dshape = [_]c_int{ 1, 1 };
        const did = mlx.mlx_array_new_data(&tok, &dshape, 2, .int32);
        defer _ = mlx.mlx_array_free(did);
        const dl = try xfm.forwardWith(&dctx, did);
        defer _ = mlx.mlx_array_free(dl);
        lctx.ple_defer = true;
        const ll = try xfm.forwardWith(&lctx, did);
        defer _ = mlx.mlx_array_free(ll);
        lctx.ple_defer = false;
        const a = try qwen4ReadF32(allocator, dl, s);
        defer allocator.free(a);
        const b = try qwen4ReadF32(allocator, ll, s);
        defer allocator.free(b);
        xfm.discardDeferredPle(&lctx);
        try testing.expect(!std.mem.eql(f32, a, b));
    }
}

test "qwen4 fixture: full prefill, chunked prefill + decode past the QSA budget match the HF reference" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("QWEN4_FIXTURE") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, allocator, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.qwen4 != null);
    // A `build --yarn` pack carries `rope_type: "yarn"` in its config: the same
    // fixture then measures the context extension end to end (attention, QSA
    // indexer, M-RoPE table, MTP head) at positions past the trained window.
    // Red probe: the YaRN fixture against the no-YaRN pack (same weights)
    // fails at the first check (MTP head cos 0.971 vs the 0.995 bar).
    try testing.expectEqual(config.rope_yarn, xfm.yarnActive());
    if (config.rope_yarn) std.debug.print("[qwen4 fixture] YaRN factor {d:.1} window {d} -> {d}\n", .{ config.yarn_factor, config.yarn_orig_max_pos, config.max_position_embeddings });
    // The fixture renders the reference in f32; run our streams in f32 too so
    // the bar measures the math. The shipped bf16 path is pinned by the
    // `qwen4 fixture bf16` test below (streams vs a bf16 reference) — a tiny
    // random model's final logits are chaotic between any two bf16 renderings.
    xfm.qwen4_stream_f32 = true;
    xfm.compileQwen4Hc(); // the served path: fused hyper-connection tails
    xfm.compileGdnGate();
    xfm.compileMoeRouting();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    const ref_full = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_full);
    const ref_last = try qwen4ReadF32(allocator, fx.get("logits_prefill_last") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_last);
    const ref_dec = try qwen4ReadF32(allocator, fx.get("logits_decode") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_dec);
    const ties = try Qwen4Ties.read(allocator, &fx, s);
    defer ties.deinit(allocator);
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const T: usize = mlx.mlx_array_size(ids_arr);
    const ids_src = mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable;
    const v: usize = @intCast(config.vocab_size);
    const t_dec: usize = ref_dec.len / v;
    const t_pre: usize = T - t_dec;

    // [1] one full-prompt forward.
    {
        const shape = [_]c_int{ 1, @intCast(T) };
        const ids = mlx.mlx_array_new_data(@ptrCast(ids_src), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids);
        const ctx_slot = try Qwen4TestSlot.init(allocator, config.num_hidden_layers);
        defer ctx_slot.deinit(allocator);
        var ctx = ctx_slot.ctx;
        // Per-layer residual streams (fixture stream_i = INPUT of layer i —
        // stream_0 is the tiled embedding — and stream_{n} = the final mixer
        // output), so our layer-i output is compared with stream_{i+1}.
        const n_layers: usize = config.num_hidden_layers;
        const cap_ids = try allocator.alloc(u32, n_layers);
        defer allocator.free(cap_ids);
        const cap_out = try allocator.alloc(mlx.mlx_array, n_layers);
        defer allocator.free(cap_out);
        for (0..n_layers) |i| {
            cap_ids[i] = @intCast(i);
            cap_out[i] = mlx.mlx_array_new();
        }
        defer for (cap_out) |o| {
            _ = mlx.mlx_array_free(o);
        };
        var cap_layers: CaptureLayers = .{ .ids = cap_ids, .out = cap_out };
        ctx.capture_layers = &cap_layers;
        var final_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(final_all);
        ctx.capture_hidden_all = &final_all;
        var stream_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stream_all);
        ctx.capture_stream_all = &stream_all;
        if (fx.get("l1_ngram_ids")) |ref_ids_arr| {
            const st = xfm.qwen4.?;
            var f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(f);
            try mlx.check(mlx.mlx_astype(&f, ref_ids_arr, .int32, s));
            try mlx.check(mlx.mlx_array_eval(f));
            const ref_ids = mlx.mlx_array_data_int32(f) orelse return error.Unreadable;
            const nh: usize = st.hash.n_heads;
            const host_ids = try allocator.alloc(u32, T);
            defer allocator.free(host_ids);
            for (0..T) |i| host_ids[i] = @intCast(ids_src[i]);
            const rows = try allocator.alloc(i64, T * nh);
            defer allocator.free(rows);
            var prev: [8]u32 = @splat(st.hash.eos);
            st.hash.rowIds(prev[0 .. st.hash.ngram_size - 1], host_ids, rows);
            var bad: usize = 0;
            for (0..T * nh) |i| {
                if (rows[i] != ref_ids[i]) {
                    if (bad < 6) std.debug.print("[qwen4 fixture] ngram id t={d} h={d}: ours {d} ref {d}\n", .{ i / nh, i % nh, rows[i], ref_ids[i] });
                    bad += 1;
                }
            }
            std.debug.print("[qwen4 fixture] ngram ids: {d}/{d} mismatched (eos {d}, mults {d} {d} {d}, vocab {d} {d} {d} {d})\n", .{ bad, T * nh, st.hash.eos, st.hash.multipliers[0], st.hash.multipliers[1], st.hash.multipliers[2], st.hash.vocab[0], st.hash.vocab[1], st.hash.vocab[2], st.hash.vocab[3] });
        }
        var trace: Trunk.Qwen4Trace = .{};
        defer trace.deinit();
        Trunk.qwen4_trace = &trace;
        defer Trunk.qwen4_trace = null;
        const logits = try xfm.forwardWith(&ctx, ids);
        defer _ = mlx.mlx_array_free(logits);
        // MTP head: stream rows 0..T-2 + tokens 1..T-1 at positions 1.. → logits for 2..T.
        if (fx.get("mtp_logits")) |ref_mtp_arr| if (xfm.qwen4_mtp != null) {
            const ref_mtp = try qwen4ReadF32(allocator, ref_mtp_arr, s);
            defer allocator.free(ref_mtp);
            const tm: c_int = @intCast(T - 1);
            var stream_prev = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stream_prev);
            const st_shape = mlx.getShape(stream_all);
            try mlx.check(mlx.mlx_slice(&stream_prev, stream_all, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, tm, st_shape[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const tshape = [_]c_int{ 1, tm };
            const next_ids = mlx.mlx_array_new_data(@ptrCast(ids_src + 1), &tshape, 2, .int32);
            defer _ = mlx.mlx_array_free(next_ids);
            try xfm.qwen4MtpReset();
            const mo = try xfm.qwen4MtpForward(stream_prev, next_ids, 1, null, .all_rows);
            defer _ = mlx.mlx_array_free(mo.logits);
            defer _ = mlx.mlx_array_free(mo.stream);
            const ours_m = try qwen4ReadF32(allocator, mo.logits, s);
            defer allocator.free(ours_m);
            // The head's own router ties are acquitted by its own margins.
            const head_ties = Qwen4Ties{
                .qsa_gap = null,
                .route_gap = if (fx.get("mtp_route_gap")) |g| try qwen4ReadF32(allocator, g, s) else null,
                .logit_margin = null,
            };
            defer head_ties.deinit(allocator);
            const rm = qwen4CompareTied("mtp", ours_m, ref_mtp, T - 1, v, 0, head_ties);
            std.debug.print("[qwen4 fixture] MTP head T-1={d}: min cos {d:.5} ({d} rows acquitted) argmax {d}/{d} decisive rows\n", .{ T - 1, rm.min_cos, rm.tied, rm.agree, rm.decisive });
            try testing.expect(rm.min_cos > 0.995);
            try testing.expect(rm.decisive >= (T - 1) / 2);
        };
        inline for (.{ "l0_mixed_attn", "l0_inj_attn", "l0_attn_out", "l0_mixed_mlp", "l0_inj_mlp", "l0_mlp_out", "l1_ple_emb", "l1_ple_out", "l3_qsa_mask", "l3_attn_out" }, .{ &trace.mixed_attn, &trace.inj_attn, &trace.attn_out, &trace.mixed_mlp, &trace.inj_mlp, &trace.mlp_out, &trace.ple_emb, &trace.ple_out, &trace.qsa_mask, &trace.attn3_out }) |key, ours_arr| {
            if (fx.get(key)) |ref| if (ours_arr.ctx != null) {
                const ref_h = try qwen4ReadF32(allocator, ref, s);
                defer allocator.free(ref_h);
                const our_h = try qwen4ReadF32(allocator, ours_arr.*, s);
                defer allocator.free(our_h);
                if (our_h.len == ref_h.len) {
                    const w: usize = ref_h.len / T;
                    const rc = qwen4CompareRows(our_h, ref_h, T, w);
                    const r0 = qwen4CompareRows(our_h, ref_h, 1, w);
                    std.debug.print("[qwen4 fixture] {s}: min row cos {d:.5} (row0 {d:.5})\n", .{ key, rc.min_cos, r0.min_cos });
                    if (rc.min_cos < 0.999) {
                        for (0..T) |row| {
                            const rr = qwen4CompareRows(our_h[row * w ..][0..w], ref_h[row * w ..][0..w], 1, w);
                            std.debug.print("    row {d}: cos {d:.4}\n", .{ row, rr.min_cos });
                        }
                    }
                } else std.debug.print("[qwen4 fixture] {s}: size {d} vs ref {d}\n", .{ key, our_h.len, ref_h.len });
            };
        }
        for (0..n_layers + 1) |i| {
            var nb: [32]u8 = undefined;
            const key = std.fmt.bufPrint(&nb, "stream_{d}", .{i}) catch unreachable;
            const ref = fx.get(key) orelse continue;
            const ref_h = try qwen4ReadF32(allocator, ref, s);
            defer allocator.free(ref_h);
            if (i == 0) continue;
            const our_arr = if (i < n_layers) cap_out[i - 1] else final_all;
            if (our_arr.ctx == null) continue;
            const our_h = try qwen4ReadF32(allocator, our_arr, s);
            defer allocator.free(our_h);
            if (our_h.len != ref_h.len) {
                std.debug.print("[qwen4 fixture] stream_{d}: size {d} vs ref {d}\n", .{ i, our_h.len, ref_h.len });
                continue;
            }
            const w: usize = ref_h.len / T;
            const rc = qwen4CompareRows(our_h, ref_h, T, w);
            std.debug.print("[qwen4 fixture] after layer {d}: min row cos {d:.5}\n", .{ i - 1, rc.min_cos });
        }
        if (fx.get("stream_pre_mixer")) |ref_pm| {
            const ref_h = try qwen4ReadF32(allocator, ref_pm, s);
            defer allocator.free(ref_h);
            const our_h = try qwen4ReadF32(allocator, final_all, s);
            defer allocator.free(our_h);
            if (our_h.len == ref_h.len) {
                const w: usize = ref_h.len / T;
                const rc = qwen4CompareRows(our_h, ref_h, T, w);
                std.debug.print("[qwen4 fixture] pre-mixer stream (layer {d} out): min row cos {d:.5}\n", .{ n_layers - 1, rc.min_cos });
                if (rc.min_cos < 0.999) for (0..T) |row| {
                    const rr = qwen4CompareRows(our_h[row * w ..][0..w], ref_h[row * w ..][0..w], 1, w);
                    std.debug.print("    row {d}: cos {d:.4}\n", .{ row, rr.min_cos });
                };
            }
        }
        const ours = try qwen4ReadF32(allocator, logits, s);
        defer allocator.free(ours);
        try testing.expectEqual(T * v, ours.len);
        const r = qwen4CompareTied("logits", ours, ref_full, T, v, 0, ties);
        std.debug.print("[qwen4 fixture] full prefill T={d}: min cos {d:.5} ({d} rows acquitted) argmax {d}/{d} decisive rows\n", .{ T, r.min_cos, r.tied, r.agree, r.decisive });
        try testing.expect(r.min_cos > 0.995);
        // A k < E fixture (`build --topk`) ties in most trunk rows across
        // every MoE layer; its selection coverage is the MTP head's ONE
        // MoE layer (decisive floor above). k = E fixtures keep the T/2 floor.
        try testing.expect(r.decisive >= (if (ties.route_gap != null) @as(usize, 2) else T / 2) and r.agree == r.decisive);
    }

    // [2] fresh state: prefill the first t_pre tokens, then decode one at a
    // time past the indexer budget (the sparse mask + key history path).
    {
        var cache = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache.deinit();
        var off: usize = 0;
        const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = null };

        const pshape = [_]c_int{ 1, @intCast(t_pre) };
        const pids = mlx.mlx_array_new_data(@ptrCast(ids_src), &pshape, 2, .int32);
        defer _ = mlx.mlx_array_free(pids);
        const plog = try xfm.forwardWith(&ctx, pids);
        defer _ = mlx.mlx_array_free(plog);
        const pours = try qwen4ReadF32(allocator, plog, s);
        defer allocator.free(pours);
        const rp = qwen4CompareTied("prefill-last", pours[(t_pre - 1) * v ..], ref_last, 1, v, t_pre - 1, ties);
        std.debug.print("[qwen4 fixture] prefill {d}: last-row cos {d:.5}\n", .{ t_pre, rp.min_cos });
        try testing.expect(rp.min_cos > 0.995);

        var dec: Qwen4TiedCmp = .{ .min_cos = 1.0, .agree = 0, .decisive = 0, .tied = 0 };
        for (0..t_dec) |k| {
            const dshape = [_]c_int{ 1, 1 };
            const did = mlx.mlx_array_new_data(@ptrCast(ids_src + t_pre + k), &dshape, 2, .int32);
            defer _ = mlx.mlx_array_free(did);
            const dlog = try xfm.forwardWith(&ctx, did);
            defer _ = mlx.mlx_array_free(dlog);
            const dours = try qwen4ReadF32(allocator, dlog, s);
            defer allocator.free(dours);
            const rd = qwen4CompareTied("decode", dours, ref_dec[k * v ..][0..v], 1, v, t_pre + k, ties);
            if (rd.min_cos < dec.min_cos) dec.min_cos = rd.min_cos;
            dec.agree += rd.agree;
            dec.decisive += rd.decisive;
            dec.tied += rd.tied;
        }
        std.debug.print("[qwen4 fixture] decode x{d}: min cos {d:.5} ({d} acquitted) argmax {d}/{d} decisive rows\n", .{ t_dec, dec.min_cos, dec.tied, dec.agree, dec.decisive });
        try testing.expect(dec.min_cos > 0.995);
        try testing.expect(dec.decisive >= (if (ties.route_gap != null) @as(usize, 0) else 2) and dec.agree == dec.decisive);
    }

    // [3] spec-verify rollback: prefill P, a CAPTURING verify of 4 tokens,
    // roll back to 1 accepted (GDN state + PLE window/tokens + QSA key
    // histories), continue 4 tokens — must match a fresh forward.
    {
        const P: usize = 10;
        const VL: usize = 4;
        const ACC: u32 = 1;
        const cont: usize = 4;
        var cache = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache.deinit();
        var off: usize = 0;
        const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = null };
        const pids = mlx.mlx_array_new_data(@ptrCast(ids_src), &[_]c_int{ 1, @intCast(P) }, 2, .int32);
        defer _ = mlx.mlx_array_free(pids);
        _ = mlx.mlx_array_free(try xfm.forwardWith(&ctx, pids));
        const kv_step_snap = cache.step;
        ctx.capture_ssm_seq = true;
        const vids = mlx.mlx_array_new_data(@ptrCast(ids_src + P), &[_]c_int{ 1, @intCast(VL) }, 2, .int32);
        defer _ = mlx.mlx_array_free(vids);
        _ = mlx.mlx_array_free(try xfm.forwardWith(&ctx, vids));
        ctx.capture_ssm_seq = false;
        try cache.truncate(P + 1 + ACC, s);
        cache.step = kv_step_snap;
        for (entries) |*e| try ssmRollbackFromCapture(e, ACC, @intCast(VL), s);
        for (entries) |*e| ssmFreeSpecCapture(e);
        off = P + 1 + ACC;
        const kept = P + 1 + ACC;
        const cids = mlx.mlx_array_new_data(@ptrCast(ids_src + kept), &[_]c_int{ 1, @intCast(cont) }, 2, .int32);
        defer _ = mlx.mlx_array_free(cids);
        const clog = try xfm.forwardWith(&ctx, cids);
        defer _ = mlx.mlx_array_free(clog);
        const ours_c = try qwen4ReadF32(allocator, clog, s);
        defer allocator.free(ours_c);

        var cache2 = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache2.deinit();
        var off2: usize = 0;
        const entries2 = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries2);
        for (entries2) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries2) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx2: ForwardCtx = .{ .cache = &cache2, .moe_seq_offset = &off2, .ssm_entries = entries2, .capture_hidden = null, .vision_embeddings = null };
        const fids = mlx.mlx_array_new_data(@ptrCast(ids_src), &[_]c_int{ 1, @intCast(kept + cont) }, 2, .int32);
        defer _ = mlx.mlx_array_free(fids);
        const flog = try xfm.forwardWith(&ctx2, fids);
        defer _ = mlx.mlx_array_free(flog);
        const fresh = try qwen4ReadF32(allocator, flog, s);
        defer allocator.free(fresh);
        const rr = qwen4CompareRows(ours_c, fresh[kept * v ..], cont, v);
        std.debug.print("[qwen4 fixture] rollback (P={d}, verify {d}, accept {d}, +{d}): min cos {d:.5} argmax {d}/{d}\n", .{ P, VL, ACC, cont, rr.min_cos, rr.argmax_agree, rr.rows });
        try testing.expect(rr.min_cos > 0.999);
        try testing.expect(rr.argmax_agree == rr.rows);
    }

    // [4] cross-round MTP head bookkeeping (mirrors mtpChainBuild / Phase 5a
    // stash / consume-time truncate): prefill history, two rounds of
    // draft → verify(capture) → accept a → trunk rollback → stash, then the
    // third round's merged (stash ++ t1) head forward must match a fresh head
    // over the full committed rows. Head KV + aux keys + pooled blocks all
    // ride qwen4MtpTruncate.
    if (xfm.qwen4_mtp != null) {
        const P: usize = 8;
        const M: usize = 2; // drafts per round
        const accs = [_]u32{ 1, 0 };
        const H = struct {
            fn rows(st: mlx.mlx_stream, arr: mlx.mlx_array, from: usize, to: usize) !mlx.mlx_array {
                const sh = mlx.getShape(arr);
                var out = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice(&out, arr, &[_]c_int{ 0, @intCast(from), 0 }, 3, &[_]c_int{ sh[0], @intCast(to), sh[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
                return out;
            }
            fn cat(st: mlx.mlx_stream, parts: []const mlx.mlx_array) !mlx.mlx_array {
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (parts) |p| _ = mlx.mlx_vector_array_append_value(vec, p);
                var out = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, st));
                return out;
            }
            fn ids(vals: []const i32) mlx.mlx_array {
                const sh = [_]c_int{ 1, @intCast(vals.len) };
                return mlx.mlx_array_new_data(@ptrCast(vals.ptr), &sh, 2, .int32);
            }
            fn argmaxLast(alloc: std.mem.Allocator, logits: mlx.mlx_array, st: mlx.mlx_stream, vv: usize) !i32 {
                const f = try qwen4ReadF32(alloc, logits, st);
                defer alloc.free(f);
                const row = f[f.len - vv ..];
                var best: usize = 0;
                for (row, 0..) |x, i| if (x > row[best]) {
                    best = i;
                };
                return @intCast(best);
            }
        };
        // Trunk: prefill P with the pre-mixer stream captured.
        var cache = try KVCache.init(allocator, config.num_hidden_layers);
        defer cache.deinit();
        var off: usize = 0;
        const entries = try allocator.alloc(SSMCacheEntry, config.num_hidden_layers);
        defer allocator.free(entries);
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        defer for (entries) |*e| {
            ssmFreeSpecCapture(e);
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            ssmFreeQsaState(e);
        };
        var ctx: ForwardCtx = .{ .cache = &cache, .moe_seq_offset = &off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = null };
        var h_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_all);
        ctx.capture_hidden_all = &h_all;
        const pids = H.ids(ids_src[0..P]);
        defer _ = mlx.mlx_array_free(pids);
        _ = mlx.mlx_array_free(try xfm.forwardWith(&ctx, pids));

        // Head history like the prefill chunk loop: rows (h[j], id[j+1]),
        // j < P-1, at position 1 (the last prompt row is round 1's job).
        try xfm.qwen4MtpReset();
        {
            const hp = try H.rows(s, h_all, 0, P - 1);
            defer _ = mlx.mlx_array_free(hp);
            const tp = H.ids(ids_src[1..P]);
            defer _ = mlx.mlx_array_free(tp);
            const o = try xfm.qwen4MtpForward(hp, tp, 1, null, .all_rows);
            _ = mlx.mlx_array_free(o.logits);
            _ = mlx.mlx_array_free(o.stream);
        }
        // Committed rows for the fresh-head replay: hidden + token per row.
        var fresh_h: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (fresh_h.items) |a| _ = mlx.mlx_array_free(a);
            fresh_h.deinit(allocator);
        }
        var fresh_ids: std.ArrayList(i32) = .empty;
        defer fresh_ids.deinit(allocator);
        try fresh_h.append(allocator, try H.rows(s, h_all, 0, P - 1));
        try fresh_ids.appendSlice(allocator, ids_src[1..P]);

        var last_hidden = try H.rows(s, h_all, P - 1, P);
        defer _ = mlx.mlx_array_free(last_hidden);
        var t1: i32 = ids_src[P];
        var next_free: usize = P + 1; // fixture ids handed out as "sampled" tokens
        var stash_ids: std.ArrayList(i32) = .empty;
        defer stash_ids.deinit(allocator);
        var stash_h: ?mlx.mlx_array = null;
        defer if (stash_h) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var stash_off0: usize = 0;
        var off0: usize = xfm.qwen4_mtp.?.seq_offset;
        var last_logits: mlx.mlx_array = .{ .ctx = null };
        defer if (last_logits.ctx != null) {
            _ = mlx.mlx_array_free(last_logits);
        };

        for (accs, 0..) |acc, round| {
            // Chain build (i = 0 merges the pending stash; i > 0 chains the
            // head's own stream).
            var drafts: [M]i32 = undefined;
            var h_chain: ?mlx.mlx_array = null;
            defer if (h_chain) |a| {
                _ = mlx.mlx_array_free(a);
            };
            for (0..M) |i| {
                const prev_tok: i32 = if (i == 0) t1 else drafts[i - 1];
                const o = if (i == 0 and stash_h != null) blk: {
                    try xfm.qwen4MtpTruncate(stash_off0);
                    try stash_ids.append(allocator, prev_tok);
                    const mids = H.ids(stash_ids.items);
                    defer _ = mlx.mlx_array_free(mids);
                    const mh = try H.cat(s, &.{ stash_h.?, last_hidden });
                    defer _ = mlx.mlx_array_free(mh);
                    _ = mlx.mlx_array_free(stash_h.?);
                    stash_h = null;
                    stash_ids.clearRetainingCapacity();
                    break :blk try xfm.qwen4MtpForward(mh, mids, @intCast(stash_off0 + 1), null, .all_rows);
                } else blk: {
                    const one = [_]i32{prev_tok};
                    const tid = H.ids(&one);
                    defer _ = mlx.mlx_array_free(tid);
                    break :blk try xfm.qwen4MtpForward(if (i == 0) last_hidden else h_chain.?, tid, @intCast(off0 + i + 1), null, .all_rows);
                };
                if (h_chain) |a| _ = mlx.mlx_array_free(a);
                h_chain = try H.rows(s, o.stream, @intCast(mlx.getShape(o.stream)[1] - 1), @intCast(mlx.getShape(o.stream)[1]));
                _ = mlx.mlx_array_free(o.stream);
                drafts[i] = try H.argmaxLast(allocator, o.logits, s, v);
                _ = mlx.mlx_array_free(o.logits);
            }
            // Verify [t1, drafts] with capture, roll the trunk back to acc.
            var vids_buf: [1 + M]i32 = undefined;
            vids_buf[0] = t1;
            for (drafts, 0..) |d, i| vids_buf[1 + i] = d;
            const kv_step_snap = cache.step;
            const base = off; // the trunk's position; qwen4 never advances `cache.step`
            ctx.capture_ssm_seq = true;
            var vh_all = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vh_all);
            ctx.capture_hidden_all = &vh_all;
            const vids = H.ids(&vids_buf);
            defer _ = mlx.mlx_array_free(vids);
            _ = mlx.mlx_array_free(try xfm.forwardWith(&ctx, vids));
            ctx.capture_ssm_seq = false;
            ctx.capture_hidden_all = null;
            try cache.truncate(base + 1 + acc, s);
            cache.step = kv_step_snap;
            for (entries) |*e| try ssmRollbackFromCapture(e, acc, @intCast(1 + M), s);
            for (entries) |*e| ssmFreeSpecCapture(e);
            off = base + 1 + acc;
            // Phase 5a stash: ids [t1, drafts[0..acc]], hidden [last_hidden, vh[0..acc]].
            try stash_ids.append(allocator, t1);
            for (drafts[0..acc]) |d| try stash_ids.append(allocator, d);
            const vh_acc = try H.rows(s, vh_all, 0, acc);
            defer _ = mlx.mlx_array_free(vh_acc);
            stash_h = if (acc == 0) try H.cat(s, &.{last_hidden}) else try H.cat(s, &.{ last_hidden, vh_acc });
            stash_off0 = off0;
            off0 = stash_off0 + 1 + acc;
            // Fresh replay rows mirror the stash.
            try fresh_h.append(allocator, try H.cat(s, &.{last_hidden}));
            try fresh_ids.append(allocator, t1);
            if (acc > 0) {
                try fresh_h.append(allocator, try H.rows(s, vh_all, 0, acc));
                try fresh_ids.appendSlice(allocator, drafts[0..acc]);
            }
            // Phase 5b: last_hidden = vh[acc], t1 = the "sampled" next token.
            _ = mlx.mlx_array_free(last_hidden);
            last_hidden = try H.rows(s, vh_all, acc, acc + 1);
            t1 = ids_src[next_free];
            next_free += 1;
            _ = round;
        }
        // Round 3's merged forward: stash ++ [t1].
        try xfm.qwen4MtpTruncate(stash_off0);
        try stash_ids.append(allocator, t1);
        const n_merged = stash_ids.items.len;
        {
            const mids = H.ids(stash_ids.items);
            defer _ = mlx.mlx_array_free(mids);
            const mh = try H.cat(s, &.{ stash_h.?, last_hidden });
            defer _ = mlx.mlx_array_free(mh);
            const o = try xfm.qwen4MtpForward(mh, mids, @intCast(stash_off0 + 1), null, .all_rows);
            _ = mlx.mlx_array_free(o.stream);
            last_logits = o.logits;
        }
        const ours_x = try qwen4ReadF32(allocator, last_logits, s);
        defer allocator.free(ours_x);
        try testing.expectEqual(n_merged * v, ours_x.len);
        // Fresh head over every committed row in one forward.
        try fresh_h.append(allocator, try H.cat(s, &.{last_hidden}));
        try fresh_ids.append(allocator, t1);
        try xfm.qwen4MtpReset();
        const fh = try H.cat(s, fresh_h.items);
        defer _ = mlx.mlx_array_free(fh);
        const fids = H.ids(fresh_ids.items);
        defer _ = mlx.mlx_array_free(fids);
        try testing.expectEqual(@as(c_int, @intCast(fresh_ids.items.len)), mlx.getShape(fh)[1]);
        const fo = try xfm.qwen4MtpForward(fh, fids, 1, null, .all_rows);
        defer _ = mlx.mlx_array_free(fo.logits);
        defer _ = mlx.mlx_array_free(fo.stream);
        const fresh_x = try qwen4ReadF32(allocator, fo.logits, s);
        defer allocator.free(fresh_x);
        const n_rows = fresh_ids.items.len;
        const rx = qwen4CompareRows(ours_x, fresh_x[(n_rows - n_merged) * v ..], n_merged, v);
        std.debug.print("[qwen4 fixture] MTP cross-round (P={d}, rounds {d}, acc {d}/{d}, merged {d} rows of {d}): min cos {d:.5} argmax {d}/{d}\n", .{ P, accs.len, accs[0], accs[1], n_merged, n_rows, rx.min_cos, rx.argmax_agree, rx.rows });
        try testing.expect(rx.min_cos > 0.999);
        try testing.expect(rx.argmax_agree == rx.rows);
    }
}

/// A decode-width `forwardQwen4With` over dense synthetic weights with every block a stand-in: the graph is the
/// embedding, one host-gathered PLE layer, the final mixer and the lm_head. Ladder strides start at the shipped
/// defaults (serial off, batched `QWEN4_DECODE_ASYNC_LADDER_DEFAULT`).
const Qwen4LadderRig = struct {
    fx: PleArmFixture,
    layer_bytes: [LAYERS * @sizeOf(MoeLayerWeights)]u8 align(@alignOf(MoeLayerWeights)),
    arrs: [16]mlx.mlx_array,
    n: usize,

    const LAYERS = 4;
    const HIDDEN = 128;
    const HC = 2;
    const RANK = 16;
    const VOCAB = 64;

    const Slot = struct {
        entries: [LAYERS]SSMCacheEntry,
        off: usize,

        fn init() Slot {
            var s: Slot = .{ .entries = undefined, .off = 0 };
            for (&s.entries) |*e| e.* = pleArmEntry();
            return s;
        }

        fn deinit(self: *Slot) void {
            for (&self.entries) |*e| ssmFreeQsaState(e);
        }
    };

    const Step = struct { logits: []f32, pending: bool };

    fn init(self: *Qwen4LadderRig) !void {
        try self.fx.init(null);
        self.n = 0;
        try self.fx.arm(false);
        const x = self.fx.xfm();
        const st = x.qwen4.?;
        const emb_dim: c_int = @intCast(st.table.dim * st.hash.n_heads);
        x.config = .{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } }, .hidden_size = HIDDEN, .hc_count = HC, .num_hidden_layers = LAYERS, .vocab_size = VOCAB };
        var prng = std.Random.DefaultPrng.init(59);
        const r = prng.random();
        x.emb_w = try self.weight(r, &.{ VOCAB, HIDDEN });
        x.lm_head_w = try self.weight(r, &.{ VOCAB, HIDDEN });
        var ones = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{HIDDEN}, 1, .bfloat16, x.s));
        self.keep(ones);
        x.ones_hidden = ones;
        const none: mlx.mlx_array = .{ .ctx = null };
        x.qwen4_mixer = .{
            .norm_w = try self.weight(r, &.{ HC, HIDDEN }),
            .down_w = try self.weight(r, &.{ HC * HIDDEN, RANK }),
            .down_s = none,
            .down_b = none,
            .up_w = try self.weight(r, &.{ RANK, HC * HIDDEN }),
            .up_s = none,
            .up_b = none,
        };
        self.layer_bytes = @splat(0);
        const layers: [*]MoeLayerWeights = @ptrCast(&self.layer_bytes);
        for (layers[0..LAYERS]) |*lw| {
            lw.hc_attn = null;
            lw.hc_mlp = null;
            lw.ple = null;
        }
        layers[0].ple = .{
            .key_w = try self.weight(r, &.{ emb_dim, HC * HIDDEN }),
            .key_s = none,
            .key_b = none,
            .value_w = try self.weight(r, &.{ emb_dim, HIDDEN }),
            .value_s = none,
            .value_b = none,
            .norm_key = try self.weight(r, &.{ HC, HIDDEN }),
            .norm_query = try self.weight(r, &.{ HC, HIDDEN }),
            .norm_conv = try self.weight(r, &.{ HC, HIDDEN }),
            .conv_w = try self.weight(r, &.{ HC * HIDDEN, @intCast(x.config.ple_conv_kernel), 1 }),
        };
        x.moe_layers = layers[0..LAYERS];
        x.decode_async_ladder = 0;
        x.decode_async_ladder_qwen4 = QWEN4_DECODE_ASYNC_LADDER_DEFAULT;
    }

    fn deinit(self: *Qwen4LadderRig) void {
        for (self.arrs[0..self.n]) |a| _ = mlx.mlx_array_free(a);
        self.fx.deinit();
    }

    fn keep(self: *Qwen4LadderRig, a: mlx.mlx_array) void {
        self.arrs[self.n] = a;
        self.n += 1;
    }

    fn weight(self: *Qwen4LadderRig, r: std.Random, shape: []const c_int) !mlx.mlx_array {
        const a = try attn256RandBf16(r, shape, self.fx.xfm().s);
        self.keep(a);
        return a;
    }

    /// One serial decode step on a deferred PLE leaf at serial ladder stride `ladder`; reports whether the leaf was
    /// still pending when the build returned, then fills it and reads the logits.
    fn step(self: *Qwen4LadderRig, slot: *Slot, ids: mlx.mlx_array, ladder: u32) !Step {
        const x = self.fx.xfm();
        x.decode_async_ladder = ladder;
        defer x.decode_async_ladder = 0;
        var ctx: ForwardCtx = .{ .cache = @ptrCast(&self.fx.cache_bytes), .moe_seq_offset = &slot.off, .ssm_entries = &slot.entries, .capture_hidden = null, .vision_embeddings = null, .ple_defer = true };
        const logits = try x.forwardQwen4With(&ctx, ids);
        defer _ = mlx.mlx_array_free(logits);
        const pending = ctx.ple_pending != null;
        try x.flushDeferredPle(&ctx);
        return .{ .logits = try qwen4ReadF32(testing.allocator, logits, x.s), .pending = pending };
    }

    fn prefill(self: *Qwen4LadderRig, sl: *Qwen4TestSlot, ids: []const i32) !void {
        const shape = [_]c_int{ 1, @intCast(ids.len) };
        const arr = mlx.mlx_array_new_data(@ptrCast(ids.ptr), &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(arr);
        const logits = try self.fx.xfm().forwardQwen4With(&sl.ctx, arr);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
    }

    /// One batched decode step over `slots` at batched ladder stride `ladder`, filling (or discarding) the leaf.
    fn batchedStep(self: *Qwen4LadderRig, slots: []const *Qwen4TestSlot, ids: []const i32, ladder: u32, fill: bool) !Step {
        const x = self.fx.xfm();
        x.decode_async_ladder_qwen4 = ladder;
        defer x.decode_async_ladder_qwen4 = QWEN4_DECODE_ASYNC_LADDER_DEFAULT;
        const run = try qwen4BatchedPleForward(x, slots, ids, 1, false, true, fill);
        defer _ = mlx.mlx_array_free(run.logits);
        return .{ .logits = try qwen4ReadF32(testing.allocator, run.logits, x.s), .pending = run.pending };
    }
};

fn expectLogitBitsEqual(want: []const f32, got: []const f32) !void {
    try testing.expectEqual(want.len, got.len);
    if (std.mem.eql(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got))) return;
    var n: usize = 0;
    var max: f32 = 0;
    for (want, got) |a, b| {
        if (@as(u32, @bitCast(a)) == @as(u32, @bitCast(b))) continue;
        n += 1;
        max = @max(max, @abs(a - b));
    }
    std.debug.print("logits differ in {d}/{d} entries, max |diff| {d}\n", .{ n, want.len, max });
    return error.TestExpectedEqual;
}

test "qwen4 decode ladder: serial decode keeps the default off; an explicit stride fills host ids first, skips lazy ones" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    standin.qwen4_standin_override = .{ .gdn = true, .attn = true, .mlp = true, .hc = true };
    defer standin.qwen4_standin_override = null;
    var rig: Qwen4LadderRig = undefined;
    try rig.init();
    defer rig.deinit();
    const s = rig.fx.xfm().s;
    const strides = [_]u32{ 1, 4 };

    var off = Qwen4LadderRig.Slot.init();
    defer off.deinit();
    var on: [strides.len]Qwen4LadderRig.Slot = .{ Qwen4LadderRig.Slot.init(), Qwen4LadderRig.Slot.init() };
    defer for (&on) |*sl| sl.deinit();
    var lazy = Qwen4LadderRig.Slot.init();
    defer lazy.deinit();
    const zero = mlx.mlx_array_new_int(0);
    defer _ = mlx.mlx_array_free(zero);
    for ([_]u32{ 11, 42, 7 }) |tok| {
        const ids = pleArmIds(&[_]u32{tok}, 1);
        defer _ = mlx.mlx_array_free(ids);
        // The batched default does not reach a serial forward.
        gather_qmv.decode_ladder_engaged = false;
        const want = try rig.step(&off, ids, 0);
        defer testing.allocator.free(want.logits);
        try testing.expect(want.pending);
        try testing.expect(!gather_qmv.decode_ladder_engaged);

        // Host-readable ids: the guard fills the leaf mid-build and the ladder fires.
        for (strides, &on) |stride, *sl| {
            gather_qmv.decode_ladder_engaged = false;
            const got = try rig.step(sl, ids, stride);
            defer testing.allocator.free(got.logits);
            try expectLogitBitsEqual(want.logits, got.logits);
            try testing.expect(!got.pending);
            try testing.expect(gather_qmv.decode_ladder_engaged);
        }

        // Ids still a lazy sample (pipelined decode): the ladder stands down instead of syncing on them.
        var lazy_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lazy_ids);
        try mlx.check(mlx.mlx_add(&lazy_ids, ids, zero, s));
        gather_qmv.decode_ladder_engaged = false;
        const got = try rig.step(&lazy, lazy_ids, 1);
        defer testing.allocator.free(got.logits);
        try expectLogitBitsEqual(want.logits, got.logits);
        try testing.expect(got.pending);
        try testing.expect(!gather_qmv.decode_ladder_engaged);
    }
}

test "qwen4 decode ladder: batched N=2 decode fills the PLE leaf first, logits and per-slot history bit-identical to off" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    standin.qwen4_standin_override = .{ .gdn = true, .attn = true, .mlp = true, .hc = true };
    defer standin.qwen4_standin_override = null;
    var rig: Qwen4LadderRig = undefined;
    try rig.init();
    defer rig.deinit();
    const alloc = testing.allocator;
    // Arms: off, on at stride 1, on at the shipped stride, and an unfilled leaf.
    const strides = [_]u32{ 0, 1, QWEN4_DECODE_ASYNC_LADDER_DEFAULT };
    const prompts = [2][]const i32{ &.{ 5, 17, 42, 9 }, &.{ 11, 4, 19 } };
    var slots: [strides.len + 1][2]*Qwen4TestSlot = undefined;
    var made: usize = 0;
    defer for (0..made) |k| slots[k / 2][k % 2].deinit(alloc);
    for (&slots) |*arm| for (arm, prompts) |*sl, p| {
        sl.* = try Qwen4TestSlot.init(alloc, Qwen4LadderRig.LAYERS);
        made += 1;
        try rig.prefill(sl.*, p);
    };

    const steps = [_][2]i32{ .{ 13, 6 }, .{ 27, 3 }, .{ 40, 12 } };
    for (steps, 0..) |ids, k| {
        const want = try rig.batchedStep(&slots[0], &ids, strides[0], true);
        defer alloc.free(want.logits);
        try testing.expect(want.pending);
        if (k == 0) {
            // The harness has teeth: a leaf left unfilled changes the logits.
            const zeroed = try rig.batchedStep(&slots[strides.len], &ids, strides[0], false);
            defer alloc.free(zeroed.logits);
            try testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(want.logits), std.mem.sliceAsBytes(zeroed.logits)));
        }
        for (strides[1..], slots[1..strides.len]) |stride, *arm| {
            gather_qmv.decode_ladder_engaged = false;
            const got = try rig.batchedStep(arm, &ids, stride, true);
            defer alloc.free(got.logits);
            try expectLogitBitsEqual(want.logits, got.logits);
            try testing.expect(!got.pending);
            try testing.expect(gather_qmv.decode_ladder_engaged);
            for (slots[0], arm.*) |a, b| {
                for (a.entries, b.entries) |*ea, *eb| try expectPleHistoryEqual(ea, eb);
            }
        }
    }
}
