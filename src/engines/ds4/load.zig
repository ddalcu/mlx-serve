//! Opening a ds4 engine from the launch flags that shape it: the MTP support GGUF beside the model, DSpark, the
//! in-checkpoint head, SSD streaming and the prefill chunk.

const std = @import("std");
const model_discovery = @import("../../host/registry/discovery.zig");
const log = @import("../../sdk/util/log.zig");
const round = @import("round.zig");
const engine_pick = @import("../../sdk/quant/gguf/engine_pick.zig");

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
    const want = flags.mtp and !flags.ssd_streaming;
    const spec: engine_pick.Spec = if (want) bridge.ggufSpec(io, gpa, path) else .{};
    const found: ?[]u8 = if (want) model_discovery.findDs4MtpSidecar(io, gpa, path) else null;
    defer if (found) |p| gpa.free(p);
    if (found) |p| if (spec.support_gguf) log.info("[ds4] MTP draft head: {s}\n", .{p}) else log.info("[ds4] this model takes no support GGUF in ds4, skipping {s} (--mlx-gguf drafts with it)\n", .{p});
    const mtp_path = if (spec.support_gguf) found else null;
    const engine = try bridge.Ds4Engine.open(gpa, path, .{
        .warm_weights = true,
        .ssd_streaming = flags.ssd_streaming,
        .prefill_chunk = flags.prefill_chunk,
        .mtp_path = mtp_path,
        .mtp_draft_tokens = if (mtp_path != null) round.mtp_draft_tokens else 0,
        .mtp_margin = round.mtp_margin,
        .dspark = flags.dspark,
        .embedded_mtp = spec.embedded_mtp,
    });
    // `draft_tokens` is the spec-readiness signal for both support kinds (MTP count, or the DSpark block size).
    log.info("[ds4] engine ready (EOS={d}, has_mtp={}, draft_tokens={d})\n", .{ engine.eosToken(), engine.hasMtp(), engine.mtpDraftTokens() });
    return engine;
}

test "open: a support GGUF beside the model reaches ds4 only for an arch that takes one" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "m.gguf", .data = "" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m-DSpark-support.gguf", .data = "" });
    var root: [std.fs.max_path_bytes]u8 = undefined;
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/m.gguf", .{root[0..try tmp.dir.realPath(io, &root)]});
    inline for (.{ true, false }) |takes| {
        const B = struct {
            pub fn ggufSpec(_: std.Io, _: std.mem.Allocator, _: []const u8) engine_pick.Spec {
                return .{ .support_gguf = takes };
            }
            pub const Ds4Engine = struct {
                pub fn open(_: std.mem.Allocator, _: []const u8, opts: anytype) !*Ds4Engine {
                    return if ((opts.mtp_path != null) == takes) error.Opened else error.TestUnexpectedResult;
                }
                pub fn eosToken(_: *Ds4Engine) i32 {
                    unreachable;
                }
                pub fn hasMtp(_: *Ds4Engine) bool {
                    unreachable;
                }
                pub fn mtpDraftTokens(_: *Ds4Engine) c_int {
                    unreachable;
                }
            };
        };
        try std.testing.expectError(error.Opened, open(B, io, std.testing.allocator, path, .{}));
    }
}
