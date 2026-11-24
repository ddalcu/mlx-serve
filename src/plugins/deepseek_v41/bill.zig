//! DeepSeek-V4.1's memory bill: its compressed caches per token, the keys a prefill query reads, and the prompt
//! pass's working set (`dsv41_mod.prefillBytes`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const env = @import("../../env.zig");
const cfg_mod = @import("config.zig");
const dsv41_mod = @import("deepseek_v41.zig");
const mlx = @import("../../mlx/ffi.zig");
const system = @import("../../sdk/util/system.zig");
const outside_model = @import("../../sdk/quant/outside_model.zig");

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

/// The Engram LRU beside the host's bytes: it fills on every load of a pack with Engram layers.
fn engramStore(config: *const cfg_mod.Config) u64 {
    return if (config.dsv41_n_engram_layers > 0) @import("decode/engram_decode.zig").billBytes(env.span(.MLX_SERVE_DSV41_ENGRAM_LRU_MB)) else 0;
}

/// The memory a bank's budget reads: `total - available` and the server's footprint.
pub fn reading() outside_model.Reading {
    return .{ .in_use = system.getTotalMemBytes() -| system.getAvailableMemBytes(), .footprint = @as(u64, system.getAppMemFootprintMb()) << 20 };
}

/// The memory in use at the bank's open, `now`, settled from the load's `start` (`outside_model.settled`: macOS shows
/// the residents' new buffers a second or two late); as read where the weights live in device memory (the footprint
/// is the host's there).
pub fn inUseAt(now: outside_model.Reading, start: outside_model.Reading, device_weights: bool) u64 {
    return if (device_weights) now.in_use else outside_model.settled(now, start).in_use;
}

/// The bytes a streamed bank's slot rows may take in each phase: the GPU working set less `wired_margin` (the host's
/// `--wired-margin-gib`) and `stream_host_bytes`, the memory in use (`inUseAt`: the residents loaded, every other
/// process) and the phase's working set at `context` positions (a prompt: one `stream_call_rows` call's; decode: the
/// state at `context + stream_generation`).
pub fn streamBudget(config: *const cfg_mod.Config, context: u64, wired_margin: u64, in_use: u64) struct { prompt: u64, decode: u64 } {
    const box = @as(u64, mlx.maxRecommendedWorkingSet()) -| wired_margin -| stream_host_bytes -| engramStore(config) -| in_use;
    return .{
        .prompt = box -| prefillAt(config, context, stream_call_rows),
        .decode = box -| prefillAt(config, context + stream_generation, 1),
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

test "dsv41 bill: a bank's memory in use settles from the load's start; a reading that covers it, or device weights, stand as read" {
    const MiB: u64 = 1 << 20;
    // The page counts have not yet shown the residents' new buffers: in use reads below the server's footprint.
    const lag: outside_model.Reading = .{ .in_use = 13_112 * MiB, .footprint = 23_582 * MiB };
    const start: outside_model.Reading = .{ .in_use = 14_000 * MiB, .footprint = 1_000 * MiB };
    try std.testing.expectEqual(13_000 * MiB + 23_582 * MiB, inUseAt(lag, start, false));
    try std.testing.expectEqual(13_112 * MiB, inUseAt(lag, start, true));
    const live: outside_model.Reading = .{ .in_use = 36_585 * MiB, .footprint = 23_582 * MiB };
    try std.testing.expectEqual(36_585 * MiB, inUseAt(live, start, false));
}

test "dsv41 bill: a bank bills the Engram LRU whenever the pack has Engram layers" {
    var c: cfg_mod.Config = .{ .model_type = "deepseek_v41" };
    try std.testing.expectEqual(@as(u64, 0), engramStore(&c));
    c.dsv41_n_engram_layers = 1;
    try std.testing.expect(engramStore(&c) > 0);
}
