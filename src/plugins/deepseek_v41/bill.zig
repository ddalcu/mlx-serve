//! DeepSeek-V4.1's memory bill: its compressed caches per token, the keys a prefill query reads, and the prompt
//! pass's working set (`dsv41_mod.prefillBytes`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const cfg_mod = @import("config.zig");
const dsv41_mod = @import("deepseek_v41.zig");
const mlx = @import("../../mlx/ffi.zig");
const system = @import("../../sdk/util/system.zig");

const PREFILL_RUNTIME_FLOOR_BYTES = @import("../../sdk/bill/prefill.zig").PREFILL_RUNTIME_FLOOR_BYTES;

/// A prompt pass's working set (`dsv41_mod.prefillBytes`) over the plugin's config.
pub fn prefillAt(c: *const cfg_mod.Config, seq: u64, chunk: u64) u64 {
    return dsv41_mod.prefillBytes(c, kvPerToken(c), seq, chunk, PREFILL_RUNTIME_FLOOR_BYTES);
}

// ── a streamed bank (`experts.bin`) ──

/// The prompt positions a streamed bank's slot rows are billed for when the model sets no `ctx_size`.
pub const stream_context: u64 = 16384;
/// The contexts a streamed bank tries when the model sets no `ctx_size`, largest first.
pub const stream_context_rungs = [_]u64{ 262144, 131072, 65536, 32768 };
/// The fewest prompt slot rows an auto context keeps.
pub const stream_min_rows: u32 = 16;
/// The rows of one streamed prompt call: each layer's routed experts are read once a call.
pub const stream_call_rows: usize = 16384;
/// Positions a request generates past its prompt (the decode phase's state).
pub const stream_generation: u64 = 8192;
/// What the box keeps unplanned beside the host's wired-limit margin: the stop under the ceiling, and the host side
/// of the process (its heap and the stream's read buffers).
pub const stream_host_bytes: u64 = 2_000_000_000 + (1 << 30);

/// The EXL3 source is shared by two independently retained request states.
pub const stream_active_slots = cfg_mod.stream_active_slots;

/// Backing capacity of one state: append buffers grow by a quarter, rings include verify slack, and a pending
/// pooling slice may retain the whole prompt call that produced it. Snapshots retain another set after append.
pub fn streamRetainedBytes(c: *const cfg_mod.Config, context: u64) u64 {
    const hd: u64 = c.head_dim;
    const stages: u64 = c.dsv4_n_compress_ratios -| c.num_hidden_layers;
    var backing: u64 = (c.num_hidden_layers + stages) *| (c.sliding_window + @as(u64, 16)) *| hd *| 2;
    for (0..c.num_hidden_layers) |li| {
        const ratio: u64 = c.dsv4_compress_ratios[li];
        if (ratio == 0 or (c.dsv41_kv_sources >> @intCast(li)) & 1 == 0) continue;
        const rows = (context +| (ratio - 1)) / ratio;
        const cap = @max(64, rows +| rows / 4 +| 1);
        backing +|= cap *| hd *| 2;
        if ((c.dsv41_index_sources >> @intCast(li)) & 1 != 0) backing +|= cap *| c.dsv4_index_head_dim *| 2;
        if (ratio > 1) backing +|= stream_call_rows *| hd *| 8;
    }
    backing +|= c.dsv4_n_dspark_target_layers *| @as(u64, c.hidden_size) *| 2;
    // The live and snapshot histories/prompt ids also remain owned while another slot runs.
    return backing *| 2 +| context *| 32;
}

/// Zero slots retains the single-flight bill; bounded slots share one workspace and one active growth term.
pub fn streamReserveAt(c: *const cfg_mod.Config, context: u64, chunk: u64, slots: u8) u64 {
    if (slots == 0) return prefillAt(c, context, chunk);
    const old_state = context *| kvPerToken(c) +| @as(u64, c.num_hidden_layers) *| (c.sliding_window + @as(u64, 16)) *| c.head_dim *| 2;
    const retained = streamRetainedBytes(c, context);
    const workspace = prefillAt(c, context, chunk) -| (old_state *| 5 / 4);
    return @as(u64, slots) *| retained +| workspace +| retained / 2;
}

/// The bytes a streamed bank's slot rows may take in each phase: the GPU working set less `wired_margin` (the host's
/// `--wired-margin-gib`) and `stream_host_bytes`, the memory in use now (the residents loaded, every other process)
/// and the phase's working set at `context` positions (a prompt: one `stream_call_rows` call's; decode: the state at
/// `context + stream_generation`).
pub fn streamBudget(config: *const cfg_mod.Config, context: u64, wired_margin: u64, active_slots: u8) struct { prompt: u64, decode: u64 } {
    const in_use = system.getTotalMemBytes() -| system.getAvailableMemBytes();
    const box = @as(u64, mlx.maxRecommendedWorkingSet()) -| wired_margin -| stream_host_bytes -| in_use;
    return .{
        .prompt = box -| streamReserveAt(config, context, stream_call_rows, active_slots),
        .decode = box -| streamReserveAt(config, context + stream_generation, 1, active_slots),
    };
}

/// The table's `bill`: only the KV sources' compressed latents and index keys grow per token (bf16); the windows
/// are a fixed ring per slot inside the module's state.
pub fn bill(p: *const abi.Parsed, shape: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    _ = shape;
    out.* = .{ .kv_per_token = kvPerToken(cfg_mod.of(p)), .prefill_attn_keys = &attnKeys, .prefill_bytes = &prefillBytes };
}

pub fn kvPerToken(c: *const cfg_mod.Config) u64 {
    var sum: u64 = 0;
    for (0..c.num_hidden_layers) |l| {
        const r: u64 = c.dsv4_compress_ratios[l];
        if (r == 0 or (c.dsv41_kv_sources >> @intCast(l)) & 1 == 0) continue;
        sum += c.head_dim / r;
        if ((c.dsv41_index_sources >> @intCast(l)) & 1 != 0) sum += c.dsv4_index_head_dim / r;
    }
    return sum * 2;
}

/// Every layer reads its window plus the indexer's top-k compressed entries, plus the sink.
fn attnKeys(p: *const abi.Parsed, seq: u64) callconv(.c) u64 {
    const c = cfg_mod.of(p);
    return @min(seq, @as(u64, c.sliding_window) + c.dsv4_index_topk + 1);
}

fn prefillBytes(p: *const abi.Parsed, seq: u64, chunk: u32, kv_bits: u8, warm: u64) callconv(.c) u64 {
    _ = kv_bits;
    _ = warm;
    return prefillAt(cfg_mod.of(p), seq, chunk);
}

test "deepseek_v41 streamed reserve: two retained contexts share one workspace and include snapshots and growth" {
    var c: cfg_mod.Config = .{};
    c.num_hidden_layers = 2;
    c.hidden_size = 5120;
    c.head_dim = 512;
    c.sliding_window = 256;
    c.num_attention_heads = 64;
    c.dsv4_hc_mult = 4;
    c.dsv4_index_topk = 512;
    c.dsv4_index_head_dim = 128;
    c.dsv4_n_compress_ratios = 2;
    c.dsv4_compress_ratios[0] = 1;
    c.dsv4_compress_ratios[1] = 2;
    c.dsv41_kv_sources = 3;
    c.dsv41_index_sources = 3;
    const ctx: u64 = 196608;
    const retained = streamRetainedBytes(&c, ctx);
    try std.testing.expect(retained > 2 * ctx * kvPerToken(&c));
    const one = streamReserveAt(&c, ctx, stream_call_rows, 1);
    const two = streamReserveAt(&c, ctx, stream_call_rows, 2);
    try std.testing.expectEqual(retained, two - one);
    try std.testing.expect(two < one * 2);
    try std.testing.expect(two > prefillAt(&c, ctx, stream_call_rows));
    try std.testing.expect(streamRetainedBytes(&c, ctx + stream_generation) > retained);
    c.dsv4_n_compress_ratios = 5;
    c.dsv4_n_dspark_target_layers = 3;
    try std.testing.expect(streamRetainedBytes(&c, ctx) > retained);
}

test "deepseek_v41 streamed reserve: legacy policy retains the exact upstream workspace bill" {
    var c: cfg_mod.Config = .{};
    c.num_hidden_layers = 2;
    c.hidden_size = 5120;
    c.head_dim = 512;
    c.sliding_window = 128;
    c.num_attention_heads = 64;
    c.dsv4_hc_mult = 4;
    c.dsv4_index_topk = 512;
    c.dsv4_index_head_dim = 128;
    c.dsv4_n_compress_ratios = 2;
    c.dsv4_compress_ratios[0] = 1;
    c.dsv4_compress_ratios[1] = 2;
    c.dsv41_kv_sources = 3;
    c.dsv41_index_sources = 3;
    for ([_]u64{ 16384, 196608, 204800 }) |context| {
        for ([_]u64{ 1, stream_call_rows }) |chunk| {
            try std.testing.expectEqual(prefillAt(&c, context, chunk), streamReserveAt(&c, context, chunk, 0));
            try std.testing.expect(streamReserveAt(&c, context, chunk, stream_active_slots) > streamReserveAt(&c, context, chunk, 0));
        }
    }
}
