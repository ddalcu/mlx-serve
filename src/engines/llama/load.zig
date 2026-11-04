//! Opening a llama.cpp engine and its context from the settings that shape a load: sequences, KV quant, prefill
//! batch and MTP drafts.

const std = @import("std");
const model_discovery = @import("../../host/registry/discovery.zig");
const log = @import("../../sdk/util/log.zig");
const ffi = @import("ffi.zig");

/// Sequences one context holds at most.
pub const max_seqs = 64;

pub const Settings = @import("../../sdk/quant/gguf/llama_settings.zig").Settings;
pub const LlamaKvQuant = @import("../../sdk/quant/gguf/llama_settings.zig").LlamaKvQuant;

pub fn Opened(comptime bridge: type) type {
    return struct { engine: *bridge.LlamaEngine, ctx: *bridge.LlamaContext };
}

/// Opens the GGUF at `path` on `bridge` (`bridge.zig`, or a test double) with a context of `s.seqs` sequences of
/// `ctx_size` tokens each. An `mtp-*.gguf` head beside the model wins over the trunk's own NextN heads.
pub fn open(comptime bridge: type, io: std.Io, gpa: std.mem.Allocator, path: []const u8, s: Settings, ctx_size: u32) !Opened(bridge) {
    log.info("[llama] opening engine: {s}\n", .{path});
    const mtp_path: ?[]u8 = if (s.mtp_drafts > 0) model_discovery.findLlamaMtpSidecar(io, gpa, path) else null;
    defer if (mtp_path) |p| gpa.free(p);
    if (mtp_path) |p| log.info("[llama] MTP draft head: {s}\n", .{p});
    const engine = try bridge.LlamaEngine.open(gpa, path, .{
        .mtp_path = mtp_path,
        .load_mtp = s.mtp_drafts > 0,
    });
    errdefer engine.close();
    log.info("[llama] engine ready (EOS={d}, n_vocab={d})\n", .{ engine.eosToken(), engine.nVocab() });

    const ctx = try engine.createContext(.{
        .ctx_size = @intCast(ctx_size),
        .n_seq = s.seqs,
        .type_k = s.kv_quant.ggmlType(),
        .type_v = s.kv_quant.ggmlType(),
        .ubatch = s.ubatch,
        .mtp_drafts = s.mtp_drafts,
    });
    log.info("[llama] context: {d} sequences x {d} tokens, KV {s}, MTP drafts {d}\n", .{
        ctx.seqs.len, ctx_size, s.kv_quant.label(), ctx.mtpDrafts(),
    });
    return .{ .engine = engine, .ctx = ctx };
}
