//! The engine's process-wide settings: set once from the launch flags (`apply`), and reported to `/props`.

const std = @import("std");
const host = @import("../../host/engines/mlx.zig");
const props = @import("../../host/api/props.zig");
const LoadedModel = @import("../../host/registry/registry.zig").LoadedModel;
const decode_attn_quant = @import("../../sdk/blocks/attention/decode_attn_quant.zig");
const ane_mod = @import("../../sdk/ane/offload.zig");
const qwen4_mod = @import("../../plugins/qwen4_exp/qwen4_exp.zig");
const generate_mod = @import("generate.zig");
const scheduler_mod = @import("scheduler.zig");
const prefill_interleave = @import("prefill_interleave.zig");
const distributed = @import("cluster/distributed.zig");
const sleep_inhibit = @import("runtime/sleep_inhibit.zig");
const depth_bounds = @import("spec/mtp_depth_bounds.zig");
const mlx_stream = @import("modules/stream.zig");

pub fn settings(out: *host.Settings) void {
    out.* = .{
        .prefill_chunk = generate_mod.prefillCli().override,
        .max_mtp_ctx = generate_mod.max_mtp_ctx,
        .prefill_decode_share = prefill_interleave.prefillDecodeShare(),
        .mtp_head_kv_quant = @import("transformer.zig").cli().mtp_head_kv_quant,
        .explicit_prefill_chunk = generate_mod.explicitPrefillChunkU32(),
    };
}

pub fn gauges(out: *host.Gauges) void {
    out.* = .{
        .ane_int8_bytes = ane_mod.aneState().live_int8_bytes.load(.monotonic),
        .ane_layers = ane_mod.aneState().live_layers.load(.monotonic),
        .ngram_warm_bytes = qwen4_mod.warmBytes().live_warm_bytes.load(.monotonic),
    };
}

pub fn modelProps(lm: *const LoadedModel, out: *host.ModelProps) void {
    const x = lm.transformer;
    const native = if (x) |t| t.mlx_stream_ext else null;
    out.* = .{
        .decode_attn_quant = decode_attn_quant.decodeAttnQuantEnabled() and (if (x) |t| t.dense_attn_proj else false),
        .mtp_loaded = lm.mtp != null or (if (x) |t| t.dsparkStages() > 0 else false),
        .mtp_acceptance = if (lm.config) |c| c.mtpAcceptance(generate_mod.mtp_acceptance_default) else generate_mod.mtp_acceptance_default,
        .mtp_greedy_tail = generate_mod.mtpGreedyTailFor(if (lm.config) |c| c.mtp_greedy_tail_override else null),
        .native_lane = if (native) |m| mlx_stream.laneName(m) else "",
        .native_block = if (native) |m| mlx_stream.blockSize(m) else 0,
    };
}

/// The Neural Engine prefill object (absent when the offload is off), then the n-gram warm object (absent when no
/// table is warming).
pub fn modelPropsJson(lm: *const LoadedModel, gpa: std.mem.Allocator) ?[]u8 {
    return modelPropsJsonOrErr(lm, gpa) catch null;
}

fn modelPropsJsonOrErr(lm: *const LoadedModel, gpa: std.mem.Allocator) ![]u8 {
    const ane_json = blk: {
        if (lm.transformer) |x| if (x.ane_prefill) |eng| {
            var stats: [ane_mod.MAX_UNITS]props.AneUnitStat = undefined;
            for (eng.units, 0..) |*u, i| stats[i] = .{
                .instance = u.instance,
                .evals = u.evals_ok.load(.monotonic),
                .eval_failures = u.evals_failed.load(.monotonic),
            };
            break :blk try props.anePropsJson(gpa, "channel", eng.coveredLayers(), eng.coveredGdnLayers(), eng.rows, eng.chunk_rows, eng.share, eng.int8_bytes, stats[0..eng.units.len]);
        };
        break :blk try gpa.dupe(u8, "");
    };
    defer gpa.free(ane_json);
    const w = qwen4_mod.warmBytes();
    const ngram_json = try props.ngramWarmPropsJson(gpa, w.live_warm_bytes.load(.acquire), w.live_warm_total.load(.acquire));
    defer gpa.free(ngram_json);
    return std.mem.concat(gpa, u8, &.{ ane_json, ngram_json });
}

pub fn clusterJson(gpa: std.mem.Allocator) ?[]u8 {
    return distributed.propsJson(gpa) catch null;
}

/// The launch flags that set the engine's own globals.
pub fn apply(f: *const host.Flags) void {
    scheduler_mod.no_vision_global = f.no_vision;
    sleep_inhibit.setEnabled(f.prevent_sleep);
    scheduler_mod.skip_mem_preflight = f.skip_mem_preflight;
    if (f.ple_gpu) @import("../../plugins/qwen4_exp/ple_gpu.zig").pleGpu().enabled = true;
    depth_bounds.active = .{ .min = f.mtp_depth_min, .max = f.mtp_depth_max };
    generate_mod.mtp_greedy_tail_default = f.mtp_greedy_tail;
    generate_mod.max_mtp_ctx = f.max_mtp_ctx;
    generate_mod.mtp_history_window_override = f.mtp_history_window;
    if (f.prefill_chunk) |w| {
        generate_mod.prefillCli().override = w;
        generate_mod.prefillCli().explicit = true;
    }
    generate_mod.prefill_trace_force = f.prefill_trace;
    generate_mod.prefillCli().machine_cap = @import("admission/prefill_chunk.zig").autoPrefillChunkCap(@import("../../mlx/ffi.zig").cudaAvailable());
    scheduler_mod.prefill_decode_share = f.prefill_decode_share;
    @import("transformer.zig").cli().mtp_head_kv_quant = f.mtp_head_kv_quant;
    generate_mod.mtp_acceptance_default = f.mtp_acceptance;
}
