//! LTX-Video's claims, caps and bill, MLX-free: an `AudioVideo` config.json beside `connector.safetensors`; the
//! resident engine billed as the pack minus the transformer variant it never holds, plus the text encoder's stage.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const env = @import("../../env.zig");
const sdk = @import("../../sdk/root.zig");
const bill = @import("../../sdk/media/bill.zig");
const pack_bytes = @import("../../sdk/media/pack_bytes.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "ltx",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"dgrauet/ltx-2.3-mlx-q4"},
};

pub const model_types = [_][]const u8{"AudioVideo"};

pub const claims = sdk.mediaClaimsByType(&model_types);

/// A generic "AudioVideo" config is not an LTX pack without its connector.
pub const required_file = "connector.safetensors";

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{};
}

/// LTX's plan. Its engine is RESIDENT (transformer + connector + VAEs + audio
/// stay on the engine for its lifetime), with two corrections the directory
/// sum cannot make:
///
///   `spare_transformer` — packs ship `transformer-dev` AND
///   `transformer-distilled` (~10.5 GiB each) and `ensureTransformer` frees one
///   BEFORE loading the other, precisely so they never coexist. The sum bills a
///   phantom.
///
///   `text_encoder` — the 2.3 Gemma encoder is a SEPARATE shared repo, loaded by
///   `ltx_video.gemmaCapture` per generation and freed when it returns. Being
///   outside the model dir, the sum bills 7.5 GiB at zero, and it is resident
///   on top of the whole engine while it runs.
///
/// No activation term: nothing has been measured for this backend, and
/// inventing one would newly refuse loads that work today.
pub fn peakBytes(dir_sum: u64, spare_transformer: u64, text_encoder: u64) u64 {
    return (dir_sum -| spare_transformer) + text_encoder;
}

/// One transformer variant as resident (its int8 copy under `--w8a8`).
fn variantBytes(dir: *const abi.PackDir, m: *const abi.Machine, name: []const u8) u64 {
    return (if (m.w8a8) pack_bytes.int8Resident(dir, name) else null) orelse dir.fileBytes(name);
}

/// The 2.3 text encoder, resolved like the engine's `resolveGemmaDir` (`LTX_GEMMA_DIR`, then the shared download
/// under `~/.mlx-serve/models`): outside the pack, so read here from the filesystem. Absent → 0.
pub fn externalTextEncoderBytes() u64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var buf: [1024]u8 = undefined;
    if (env.span(.LTX_GEMMA_DIR)) |e| {
        if (std.fs.path.isAbsolute(e)) return dirSafetensors(io, e);
    }
    const home = std.mem.span(env.get(.HOME) orelse return 0);
    for ([_][]const u8{ "mlx-community/gemma-3-12b-it-4bit", "gemma-3-12b-it-4bit" }) |rel| {
        const p = std.fmt.bufPrint(&buf, "{s}/.mlx-serve/models/{s}", .{ home, rel }) catch continue;
        const n = dirSafetensors(io, p);
        if (n > 0) return n;
    }
    return 0;
}

fn dirSafetensors(io: std.Io, path: []const u8) u64 {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return 0; // openDirAbsolute UB class
    var d = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var total: u64 = 0;
    var it = d.iterate();
    while (it.next(io) catch null) |e| {
        if (!std.mem.endsWith(u8, e.name, ".safetensors")) continue;
        const st = d.statFile(io, e.name, .{}) catch continue;
        if (st.kind == .file) total += @intCast(st.size);
    }
    return total;
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    const sum = bill.safetensorsIn(dir, "");
    if (sum == 0) return 0; // unknown dir -> never block
    const spare = @min(variantBytes(dir, m, "transformer-dev.safetensors"), variantBytes(dir, m, "transformer-distilled.safetensors"));
    return peakBytes(sum -| pack_bytes.ditBytes(dir, false) + pack_bytes.ditBytes(dir, m.w8a8), spare, externalTextEncoderBytes());
}

/// Nothing measured above the resident engine (see `peakBytes`).
pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = dir;
    _ = m;
    _ = body;
    return 0;
}

// ── tests ──

const testing = std.testing;

test "ltx bill: one transformer variant resident, the text encoder's stage on top" {
    const gb: u64 = 1 << 30;
    const variant = 10 * gb;
    const dir_sum = 2 * variant + 6 * gb;
    try testing.expectEqual(dir_sum - variant + 7 * gb, peakBytes(dir_sum, variant, 7 * gb));
    try testing.expectEqual(dir_sum - variant, peakBytes(dir_sum, variant, 0));
}

test "LTX bills ONE transformer variant, plus the text encoder its dir cannot see" {
    const MB: u64 = 1024 * 1024;
    // Real dgrauet/ltx-2.3-mlx-q4 sizes. Both transformer variants ship at
    // 10.54 GiB and `ensureTransformer` frees one BEFORE loading the other, so
    // the sum bills a phantom. The Gemma text encoder is a SEPARATE repo
    // (mlx-community/gemma-3-12b-it-4bit, 7.5 GiB) loaded per generation on top
    // of the resident engine, so the dir sum bills it at zero.
    const dir_sum: u64 = 30_318 * MB; // all eight files
    const variant: u64 = 10_793 * MB;
    const gemma: u64 = 7_680 * MB;

    const peak = peakBytes(dir_sum, variant, gemma);
    try std.testing.expectEqual(dir_sum - variant + gemma, peak);
    // Strictly below the sum-of-dir bill it replaces on a two-variant pack…
    try std.testing.expect(peak < dir_sum + dir_sum / 10);
    // …but a one-variant pack must go UP, not down: the text encoder is real
    // and was billed at nothing. Under-billing is the uncatchable-OOM side.
    try std.testing.expect(peakBytes(dir_sum - variant, 0, gemma) > dir_sum - variant);
    // A missing/unfound encoder dir contributes nothing rather than guessing.
    try std.testing.expectEqual(dir_sum - variant, peakBytes(dir_sum, variant, 0));

    // LTX 2.5 ships its text encoder INSIDE the pack, and `sumSafetensorsIn`
    // recurses one level — so `dir_sum` already carries it. It is resident
    // only while the engine is (loaded per generation, freed on return), so
    // the peak is still sum-minus-spare and the encoder must NOT also ride in
    // as a stage: that bills 6.8 GiB twice and refuses loads that fit.
    const te_in_pack: u64 = 6_800 * MB;
    const sum_25: u64 = dir_sum + te_in_pack;
    try std.testing.expectEqual(sum_25 - variant, peakBytes(sum_25, variant, 0));
    try std.testing.expect(peakBytes(sum_25, variant, te_in_pack) > sum_25 - variant);
}
