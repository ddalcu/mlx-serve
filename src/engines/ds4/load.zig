//! Opening a ds4 engine from the launch flags that shape it: the MTP support GGUF beside the model, DSpark, the
//! in-checkpoint head, SSD streaming and the prefill chunk.

const std = @import("std");
const model_discovery = @import("../../host/registry/discovery.zig");
const log = @import("../../sdk/util/log.zig");
const round = @import("round.zig");

/// The flags (`--ds4-mtp`, `--dspark`, `--ssd-streaming`, `--prefill-chunk`); the adapter's `settings_json`.
pub const Flags = struct {
    mtp: bool = true,
    dspark: bool = false,
    ssd_streaming: bool = false,
    prefill_chunk: u32 = 0,
};

/// Opens the GGUF at `path` on `bridge` (`bridge.zig`, or a test double). ds4 refuses `--mtp` together with
/// `--ssd-streaming`, so the support GGUF is only sought when streaming is off.
pub fn open(comptime bridge: type, io: std.Io, gpa: std.mem.Allocator, path: []const u8, flags: Flags) !*bridge.Ds4Engine {
    log.info("[ds4] opening engine: {s}\n", .{path});
    const mtp_path: ?[]u8 = if (flags.mtp and !flags.ssd_streaming) model_discovery.findDs4MtpSidecar(io, gpa, path) else null;
    defer if (mtp_path) |p| gpa.free(p);
    if (mtp_path) |p| log.info("[ds4] MTP draft head: {s}\n", .{p});
    const engine = try bridge.Ds4Engine.open(gpa, path, .{
        .warm_weights = true,
        .ssd_streaming = flags.ssd_streaming,
        .prefill_chunk = flags.prefill_chunk,
        .mtp_path = mtp_path,
        .mtp_draft_tokens = if (mtp_path != null) round.mtp_draft_tokens else 0,
        .mtp_margin = round.mtp_margin,
        .dspark = flags.dspark,
        .embedded_mtp = flags.mtp and !flags.ssd_streaming and bridge.ggufDeclaresEmbeddedMtp(io, gpa, path),
    });
    // `draft_tokens` is the spec-readiness signal for both support kinds (MTP count, or the DSpark block size).
    log.info("[ds4] engine ready (EOS={d}, has_mtp={}, draft_tokens={d})\n", .{ engine.eosToken(), engine.hasMtp(), engine.mtpDraftTokens() });
    return engine;
}
