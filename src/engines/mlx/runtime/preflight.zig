//! Load preflight: resident and on-disk bills, the eviction gate estimate, available memory.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ple_gpu = @import("../../../plugins/qwen4_exp/ple_gpu.zig");
const gen_mod = @import("../../../host/media_jobs/gen.zig");
const mtp_mod = @import("../../../sdk/spec/mtp/mtp.zig");
const model_mod = @import("../../../sdk/config/model.zig");
const plugin_driver = @import("../plugin/driver.zig");
const abi = @import("../../../abi/plugin.zig");
const log = @import("../../../sdk/util/log.zig");
const modelopt_pack = @import("../../../sdk/quant/nvfp4/modelopt_pack.zig");
const model_registry_mod = @import("../../../host/registry/registry.zig");
const model_discovery = @import("../../../host/registry/discovery.zig");
const mlx_gguf = @import("../../../sdk/quant/gguf/claim.zig");
const status = @import("../../../host/status.zig");
const streamed = @import("../../../sdk/weights/streamed.zig");
const mlx_stream = @import("../modules/stream.zig");
const testing = std.testing;

/// A load whose own bill exceeds the memory cap is refused whatever else is resident.
pub fn loadExceedsMemCap(estimated: u64, max_resident_mem: u64) bool {
    return max_resident_mem != 0 and estimated > max_resident_mem;
}

const TEXT_GATE_MARGIN_CAP_BYTES: u64 = 12 * 1024 * 1024 * 1024;

/// The post-load residency bill the eviction gate reserves, in bytes.
///
/// A media entry is billed by its BACKEND: a staged-residency model
/// (`minimax_h3` runs its text encoder and FREES it before the DiT loads) never
/// holds the sum of every safetensors in its directory. The gate runs BEFORE
/// the media preflight that already knows this, so a disagreement here is a
/// refusal the preflight never gets to overturn — H3's 37.55 GiB sum against a
/// 48 GB Mac's 30.0 GiB auto cap made the model permanently unloadable while
/// its real 22.83 GiB peak fit with 7 GB to spare (#126).
///
/// `media_peak == 0` means "not a media model, or a directory we could not
/// read" and keeps the original ladder byte-for-byte: a text bill is weights
/// only, so it takes 10% headroom for KV / vision / drafter overhead, up to
/// `TEXT_GATE_MARGIN_CAP_BYTES`: that overhead does not grow with the weights.
///
/// A media peak takes NONE. Those are text-model concepts — a media engine has
/// no KV cache and no drafter — and `gen.estimatePeakResidentBytes` already
/// carries an explicit transient term for the stage that generates. Stacking a
/// second blanket margin on it billed H3's 8-bit pack ~27% over its measured
/// peak, which is the difference between loading and not on a 48 GB Mac. It
/// also makes the gate agree with `genLoadResidentBytes`, which has always
/// committed the bare peak.
pub fn gateEstimateBytes(media_peak: u64, bytes_on_disk: ?u64, num_hidden_layers: u32, hidden_size: u32) u64 {
    if (media_peak > 0) return media_peak;
    const base: u64 = if (bytes_on_disk) |b|
        b
    else
        @as(u64, num_hidden_layers) * @as(u64, hidden_size) * 4 * 4;
    return base + @min(base / 10, TEXT_GATE_MARGIN_CAP_BYTES);
}

/// Where a media peak lives. Unified memory holds it in one pool. CUDA keeps file-loaded
/// weights in managed memory backed by host RAM and the engine's working buffers on the GPU,
/// so there the weights (capped by the peak: a staged pack never holds its whole sum) are
/// RAM's and only the rest of the peak is the GPU's.
pub fn mediaBill(peak: u64, weights_on_disk: u64, weights_page: bool) struct { ram: u64, gpu: u64 } {
    if (!weights_page or weights_on_disk == 0) return .{ .ram = peak, .gpu = 0 };
    const ram = @min(peak, weights_on_disk);
    return .{ .ram = ram, .gpu = peak - ram };
}

/// What a media load COMMITS to the residency budget once ready. Same estimator
/// the gate reserved against, so reserve and commit can only differ by the
/// gate's headroom: committing the dir sum instead parked H3 in the budget at
/// 14.7 GB more than it can ever hold, evicting live LLMs for bytes nobody was
/// using (#126, secondary 1).
pub fn genLoadResidentBytes(media_peak: u64, bytes_on_disk: u64) u64 {
    if (media_peak > 0) return media_peak;
    return bytes_on_disk;
}

/// Bytes MLX holds right now across every resident model (an allocator counter, safe off the inference thread).
pub fn mlxActiveBytes() u64 {
    var n: usize = 0;
    _ = mlx.mlx_get_active_memory(&n);
    return n;
}

/// The n-gram table is resident, wired, only on the `--ple-gpu` arm, and sits outside the
/// `.safetensors` sum; the host gather only faults in the rows it reads.
pub fn pleTableBill(io: std.Io, config: *const model_mod.ModelConfig) u64 {
    if (!ple_gpu.pleGpu().enabled) return 0;
    if (config.embedded_ple_payload_bytes) |bytes| return bytes;
    const p = config.ngram_table_path orelse return 0;
    const st = std.Io.Dir.cwd().statFile(io, p, .{}) catch return 0;
    return @intCast(st.size);
}

/// Sum of `*.safetensors` bytes in `model_dir` — the MLX weight footprint used
/// by the load pre-flight. Returns 0 if the dir can't be read (treated as
/// "unknown" by the caller, which then skips the check). Symlinked weights
/// count (statFile follows links) — an HF hub-cache snapshot is ALL symlinks.
pub fn modelDiskBytes(io: std.Io, model_dir: []const u8) u64 {
    if (mlx_gguf.weightBytes(io, model_dir)) |bytes| return bytes;
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch return 0;
    defer dir.close(io);
    // A pack's index names the shards the loader reads; a stray shard beside
    // them (issue #274) is dead weight and must not be billed.
    var referenced: ?std.StringHashMapUnmanaged(void) = model_discovery.indexShardSet(io, dir);
    defer if (referenced) |*r| model_discovery.freeShardSet(r);
    var it = dir.iterate();
    var total: u64 = 0;
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        if (referenced) |r| if (!r.contains(entry.name)) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind != .file) continue;
        total += @intCast(st.size);
    }
    if (referenced) |r| {
        var kit = r.keyIterator();
        while (kit.next()) |name| {
            if (std.mem.indexOfScalar(u8, name.*, '/') == null or std.mem.indexOf(u8, name.*, "..") != null) continue;
            const st = dir.statFile(io, name.*, .{}) catch continue;
            if (st.kind == .file) total += @intCast(st.size);
        }
    }
    return total;
}

/// A ModelOpt pack also bills the banks' decoded scales, which are smaller on disk.
pub fn residentModelDiskBytes(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, config: *const model_mod.ModelConfig) !u64 {
    const shards = try shardResidentBytes(allocator, io, model_dir, config);
    const whole = if (config.modelopt) shards +| modelopt_pack.bankExpansionBytes(model_dir) else shards;
    return @import("../cluster/distributed.zig").hostShare(model_dir, config, whole);
}

fn shardResidentBytes(allocator: std.mem.Allocator, io: std.Io, model_dir: []const u8, config: *const model_mod.ModelConfig) !u64 {
    if (config.mlxStream()) return mlx_stream.loadBytes(allocator, io, config);
    var shards = modelDiskBytes(io, model_dir);
    // An expert bank's plugin bills its rows (`resident_bytes`).
    if (config.stream_experts_override) |st| if (!config.expert_bank) {
        // The routed experts stay on disk; their slot rows are resident.
        const s = try streamed.Streamed.packBytes(allocator, model_dir, st.slotRows(config.num_experts));
        shards = shards -| s.experts +| s.slots;
    };
    const b = config.plugin orelse return shards;
    const f = b.table.weights.resident_bytes orelse return shards;
    var settings: [128]u8 = undefined;
    var out = shards;
    const v = f(&b.parsed, .of(model_dir), .of(plugin_driver.loadSettings(&settings, config)), shards, &out);
    if (!v.isOk()) {
        log.err("[preflight] {s}: {s}\n", .{ b.table.name.slice(), v.why.slice() });
        return error.ResidentBytesRefused;
    }
    return out;
}

pub fn residentGateBytes(validated: ?u64, discovered: ?u64, mlx_text: bool) ?u64 {
    if (!mlx_text) return discovered;
    if (validated) |bytes| if (bytes > 0) return bytes;
    return discovered orelse 0;
}

pub fn committedTextBytes(model_bytes: u64, config: *const model_mod.ModelConfig) u64 {
    if (model_bytes > 0) return model_bytes;
    return @as(u64, config.num_hidden_layers) * @as(u64, config.hidden_size) * 4 * 4;
}

/// Which "available memory" figure the preflight should trust. On iOS the
/// host-wide number is meaningless — the OS keeps RAM full (file cache,
/// jetsam-evictable background apps) and will evict on our behalf, so the
/// per-PROCESS jetsam headroom (`os_proc_available_memory`, nonzero only on
/// iOS) is the figure that decides whether the load survives. Live bug: an
/// 8 GB iPhone reported ~4 GB host-free and the preflight refused a 3.6 GB
/// model that fit comfortably inside the ~6.4 GB process limit.
/// `gpu_limit` = Metal's working-set limit (0 = unknown): a lowered `iogpu.wired_limit_mb` makes
/// it bind below free RAM, and weights past it OOM in warmup instead of refusing by name.
/// Memory a load of `need` bytes can use. A resident media cache is opportunistic, so a load that
/// would be refused makes it let go first and the figure is read again.
pub fn availForLoad(io: std.Io, need: u64) u64 {
    var avail = loadAvail(io);
    if (memInsufficientForLoad(need, avail)) {
        const released = gen_mod.releaseMediaResidency();
        if (released > 0) {
            model_registry_mod.generations().last_unload.record(model_registry_mod.monotonicMs(), released);
            avail = loadAvail(io);
        }
    }
    if (!waitForUnload(need, avail, model_registry_mod.generations().last_unload, model_registry_mod.monotonicMs())) return avail;
    return settledAvail(need, avail, io, loadAvail, pauseLoadAvail);
}

const UNLOAD_SETTLE_MS = model_registry_mod.UNLOAD_SETTLE_MS;

/// Whether a load refused at `avail` waits for memory still coming back: only right after an
/// unload whose bytes could close the gap. Any other refusal is instant, since the wait stalls every stream.
fn waitForUnload(need: u64, avail: u64, mark: model_registry_mod.UnloadMark, now_ms: i64) bool {
    if (mark.bytes == 0 or now_ms - mark.at_ms > UNLOAD_SETTLE_MS) return false;
    return !memInsufficientForLoad(need, avail +| mark.bytes);
}

fn loadAvail(_: std.Io) u64 {
    return effectiveAvailableBytes(status.getAvailableMemBytes(), status.getProcAvailableMemBytes(), loadGpuLimit());
}

/// RAM a load can still take beside everything resident: what the OS grants less the reserve it
/// keeps (iOS's per-process headroom already is what the process may take). Null when unreadable.
pub fn loadRamRoom() ?u64 {
    const proc = status.getProcAvailableMemBytes();
    if (proc > 0) return proc;
    const host = status.getAvailableMemBytes();
    if (host == 0) return null;
    return host -| status.osReserveBytes(status.getTotalMemBytes(), mlx.weightsInDeviceMemory());
}

fn pauseLoadAvail(io: std.Io) void {
    std.Io.sleep(io, .fromMilliseconds(100), .real) catch {};
}

/// Memory an unload just freed reaches the OS over a second or two (150 GB of wired weights on an
/// M5 Ultra), or all at once ~700 ms late (16 GB M4): a reading too low for the load is re-read
/// every pause, for at most 5 s; two seconds without growth leave the refusal standing.
fn settledAvail(need: u64, first: u64, ctx: anytype, comptime measure: fn (@TypeOf(ctx)) u64, comptime pause: fn (@TypeOf(ctx)) void) u64 {
    const step: u64 = 64 << 20; // below this the reading is noise, not memory coming back
    var avail = first;
    var flat: u32 = 0;
    var reads: u32 = 0;
    while (memInsufficientForLoad(need, avail) and flat < 20 and reads < 50) : (reads += 1) {
        pause(ctx);
        const next = measure(ctx);
        if (next > avail +| step) {
            avail = next;
            flat = 0;
        } else flat += 1;
    }
    return avail;
}

/// The GPU ceiling on loaded weights. MLX's CUDA backend keeps file-loaded arrays in managed
/// memory that pages to host RAM, so there RAM is the bound (and an allocation failure is a
/// catchable error, not Metal's process-killing OOM); 0 = none.
fn loadGpuLimit() u64 {
    return if (mlx.cudaAvailable()) 0 else mlx.maxRecommendedWorkingSet();
}

fn effectiveAvailableBytes(host_avail: u64, proc_avail: u64, gpu_limit: u64) u64 {
    const avail = if (proc_avail > 0) proc_avail else host_avail;
    return if (gpu_limit > 0) @min(avail, gpu_limit) else avail;
}

pub fn memInsufficientForLoad(weights_bytes: u64, avail_bytes: u64) bool {
    if (weights_bytes == 0 or avail_bytes == 0) return false;
    // Headroom over the weights for warmup buffers + a baseline KV cache (see
    // `loadRequirementBytes`). `avail_bytes` (status.getAvailableMemBytes) is
    // what the OS grants a new wiring, never total − used. --skip-mem-preflight bypasses.
    return avail_bytes < loadRequirementBytes(weights_bytes);
}

/// Bytes of the MTP head sidecar the loader will read beside `model_dir`'s shards
/// (`mtp.sidecar_rel_paths`; a sidecar ships at its serving width): 0 when the head is
/// in the checkpoint (its shards are billed already), absent, or not going to load.
pub fn mtpSidecarDiskBytes(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8, mtp_enabled: bool) u64 {
    if (!mtp_enabled) return 0;
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return 0;
    defer dir.close(io);
    const rel = mtp_mod.resolveMtpSidecarInDir(io, allocator, dir) orelse return 0;
    const st = dir.statFile(io, rel, .{}) catch return 0;
    return @intCast(st.size);
}

/// Resident bytes a sidecar takes once loaded: `bits` per weight plus a bf16
/// (scale, bias) pair per group of 64 for a dense DFlash assistant quantized
/// at load; its file size otherwise (`bits == 0`).
pub fn drafterResidentBytes(disk_bytes: u64, bits: u32) u64 {
    if (bits == 0) return disk_bytes;
    return disk_bytes * (2 * bits + 1) / 32;
}

const PreflightVerdict = enum { fits, drop_drafter, refuse };

/// The load preflight over the model AND its sidecar: a sidecar the checkpoint
/// brought along (`drafter_optional`) is dropped before the model is refused.
pub fn preflightVerdict(weights_bytes: u64, drafter_bytes: u64, avail_bytes: u64, drafter_optional: bool) PreflightVerdict {
    if (!memInsufficientForLoad(weights_bytes + drafter_bytes, avail_bytes)) return .fits;
    if (drafter_optional and drafter_bytes > 0 and !memInsufficientForLoad(weights_bytes, avail_bytes)) return .drop_drafter;
    return .refuse;
}

/// Total free memory a load demands: the model's own peak plus the headroom the
/// guard wants for warmup buffers and a baseline KV cache.
///
/// It exists so a REFUSAL can quote the number it actually compared. The media
/// preflight used to say "generation peaks at ~4.3 GB but only 5.4 GB is free"
/// — two figures that, read together, say the load should have worked. It was
/// refused for the headroom, which the sentence never mentioned, so the user
/// went hunting for a problem that wasn't there. Same class as
/// the context-overflow 400: a rejection has to state the bar it set.
pub fn loadRequirementBytes(weights_bytes: u64) u64 {
    const HEADROOM_CAP: u64 = 6 * 1024 * 1024 * 1024;
    const headroom: u64 = @min(weights_bytes / 8, HEADROOM_CAP) + 1024 * 1024 * 1024;
    return weights_bytes + headroom;
}

// ── tests ──

test "pleTableBill: the GPU arm bills the n-gram table, embedded shards included; the host gather bills nothing" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = "0123456789" });
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const path = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/ngram_table.bin", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(path);

    const was = ple_gpu.pleGpu().enabled;
    defer ple_gpu.pleGpu().enabled = was;
    var config: model_mod.ModelConfig = .{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } }, .ngram_table_path = path };
    ple_gpu.pleGpu().enabled = false;
    try std.testing.expectEqual(@as(u64, 0), pleTableBill(io, &config));
    ple_gpu.pleGpu().enabled = true;
    try std.testing.expectEqual(@as(u64, 10), pleTableBill(io, &config));
    // An embedded pack bills its shards' payload; its `ngram_table.bin` path names no file.
    config.embedded_ple_payload_bytes = 120;
    try std.testing.expectEqual(@as(u64, 120), pleTableBill(io, &config));
    config = .{ .model_type = "qwen4_exp", .legacy = .{ .caps = .{ .long_ctx = true } } };
    try std.testing.expectEqual(@as(u64, 0), pleTableBill(io, &config));
}

test "residentModelDiskBytes: a ModelOpt pack is billed its decoded expert banks" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try modelopt_pack.writeMiniPack(a, io, tmp.dir, .{ .layers = 2, .experts = 3, .cols = 64, .scale_suffix = "weight_scale.nvfp4_csf_fixed", .scale_len = 10 });
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const path = try std.fmt.allocPrint(a, "{s}/.zig-cache/tmp/{s}", .{ std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr))), tmp.sub_path });
    defer a.free(path);
    const on_disk = modelDiskBytes(io, path);
    var config: model_mod.ModelConfig = .{ .model_type = "glm5_next" };
    try std.testing.expectEqual(on_disk, try residentModelDiskBytes(a, io, path, &config));
    config.modelopt = true;
    try std.testing.expectEqual(on_disk + 18 * 18, try residentModelDiskBytes(a, io, path, &config));
}

test "GGUF and unavailable media peak retain discovery bytes at the eviction gate" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(@as(?u64, 100 * GB), residentGateBytes(null, 100 * GB, false));
    try std.testing.expectEqual(110 * GB, gateEstimateBytes(0, residentGateBytes(null, 100 * GB, false), 48, 2560));
    try std.testing.expectEqual(24 * GB, gateEstimateBytes(24 * GB, residentGateBytes(null, 100 * GB, false), 0, 0));
    try std.testing.expectEqual(@as(?u64, 70 * GB), residentGateBytes(70 * GB, 100 * GB, true));
    try std.testing.expectEqual(@as(?u64, 100 * GB), residentGateBytes(0, 100 * GB, true));
    try std.testing.expectEqual(@as(?u64, 0), residentGateBytes(0, null, true));
    try std.testing.expectEqual(@as(?u64, 0), residentGateBytes(null, 0, false));
}

test "a text pack's gate margin stops growing with its weights" {
    const GiB: u64 = 1024 * 1024 * 1024;
    // DeepSeek-V4.1 oQ2.5e: 203.83 GiB of resident weights must fit a 222.72 GiB cap with nothing else loaded.
    const ds: u64 = 218_858_232_429;
    try std.testing.expect(gateEstimateBytes(0, ds, 43, 4096) < 222 * GiB);
    // Up to 100 GiB the margin is still 10%.
    try std.testing.expectEqual(@as(u64, 66 * GiB), gateEstimateBytes(0, 60 * GiB, 48, 2560));
}

test "a load over the memory cap on its own is named as such, not as a busy registry" {
    const GiB: u64 = 1024 * 1024 * 1024;
    try std.testing.expect(loadExceedsMemCap(225 * GiB, 222 * GiB));
    try std.testing.expect(!loadExceedsMemCap(200 * GiB, 222 * GiB));
    try std.testing.expect(!loadExceedsMemCap(225 * GiB, 0));
}

test "residentModelDiskBytes: a served pack is billed by its plugin's resident_bytes, which sees the load settings" {
    const T = struct {
        fn resident(_: *const abi.Parsed, _: abi.Str, settings_json: abi.Str, shards: u64, out: *u64) callconv(.c) abi.Verdict {
            if (std.mem.eql(u8, settings_json.slice(), "{\"mtp\":false}")) return abi.Verdict.refuse(.failed, "mtp off");
            out.* = shards - 7;
            return .ok;
        }
    };
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "0123456789" });
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = buf[0..try tmp.dir.realPath(io, &buf)];
    const table: abi.ModelTable = .{ .name = .of("t"), .claims = undefined, .parse = undefined, .free_parsed = undefined, .bill = undefined, .load = undefined, .unload = undefined, .forward = undefined, .cache = undefined, .weights = &.{ .resident_bytes = &T.resident } };
    var binding: @import("../plugin/config.zig").Binding = .{ .table = &table };
    var config: model_mod.ModelConfig = .{ .num_hidden_layers = 2, .hidden_size = 64 };
    try std.testing.expectEqual(@as(u64, 10), try residentModelDiskBytes(std.testing.allocator, io, dir, &config));
    config.plugin = &binding;
    const snapshot = try residentModelDiskBytes(std.testing.allocator, io, dir, &config);
    try std.testing.expectEqual(@as(u64, 3), snapshot);
    try std.testing.expectEqual(snapshot, committedTextBytes(snapshot, &config));
    try std.testing.expectEqual(snapshot + snapshot / 10, gateEstimateBytes(0, snapshot, config.num_hidden_layers, config.hidden_size));
    config.mtp_override = false;
    try std.testing.expectError(error.ResidentBytesRefused, residentModelDiskBytes(std.testing.allocator, io, dir, &config));
}

test "modelDiskBytes follows HF-cache symlinks (a snapshot dir measured ZERO)" {
    // A model served straight out of the HuggingFace hub cache is a snapshot
    // dir of SYMLINKS into ../../blobs. Skipping .sym_link entries measured a
    // 121 GB checkpoint at 0 bytes, and memInsufficientForLoad treats 0 as
    // "unknown" → the preflight waved the load through into 34 GB of swap.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "blobs");
    try tmp.dir.writeFile(io, .{ .sub_path = "blobs/abc123", .data = "0123456789abcdef" });
    try tmp.dir.createDirPath(io, "snapshots/rev");
    try tmp.dir.symLink(io, "../../blobs/abc123", "snapshots/rev/model.safetensors", .{});
    // A dangling link (blob pruned) is skipped, never an error…
    try tmp.dir.symLink(io, "../../blobs/gone", "snapshots/rev/model-00002.safetensors", .{});
    // …and a symlink to a DIRECTORY must not be summed (statFile follows it).
    try tmp.dir.symLink(io, "../../blobs", "snapshots/rev/dir.safetensors", .{});

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const snap = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/snapshots/rev", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(snap);

    try std.testing.expectEqual(@as(u64, 16), modelDiskBytes(io, snap));
}

test "modelDiskBytes bills only the shards the index names (issue #274)" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "m");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model-00001-of-00002.safetensors", .data = "0123456789" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model-00002-of-00002.safetensors", .data = "01234" });
    // Dead weight: present on disk, referenced by nothing.
    try tmp.dir.writeFile(io, .{ .sub_path = "m/stray.safetensors", .data = "0123456789abcdef0123456789abcdef" });
    try tmp.dir.writeFile(io, .{ .sub_path = "m/model.safetensors.index.json", .data =
        \\{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors","c":"model-00001-of-00002.safetensors"}}
    });

    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}/m", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(dir);
    try std.testing.expectEqual(@as(u64, 15), modelDiskBytes(io, dir));
}

test "settledAvail: a refused reading waits while freed memory still reaches the OS" {
    const GB: u64 = 1024 * 1024 * 1024;
    const Seq = struct {
        vals: []const u64,
        i: usize = 0,
        fn measure(self: *@This()) u64 {
            const v = self.vals[@min(self.i, self.vals.len - 1)];
            self.i += 1;
            return v;
        }
        fn pause(_: *@This()) void {}
    };
    // An unload's 150 GB unwires over a few readings: the load fits once it lands.
    var climbing = Seq{ .vals = &.{ 80 * GB, 140 * GB, 200 * GB, 230 * GB } };
    try std.testing.expectEqual(200 * GB, settledAvail(180 * GB, 66 * GB, &climbing, Seq.measure, Seq.pause));
    // A 16 GB M4 unloading 7.3 GB: flat for ~700 ms, then the whole release lands at once.
    var stepped = Seq{ .vals = &.{ 5 * GB, 5 * GB, 5 * GB, 5 * GB, 5 * GB, 5 * GB, 5 * GB, 11 * GB } };
    try std.testing.expectEqual(11 * GB, settledAvail(6 * GB, 5 * GB, &stepped, Seq.measure, Seq.pause));
    // Nothing being freed: the refusal stands after two seconds of flat readings, not five.
    var flat = Seq{ .vals = &.{66 * GB} };
    try std.testing.expectEqual(66 * GB, settledAvail(180 * GB, 66 * GB, &flat, Seq.measure, Seq.pause));
    try std.testing.expectEqual(@as(usize, 20), flat.i);
    // A load that already fits never waits.
    var fits = Seq{ .vals = &.{0} };
    try std.testing.expectEqual(200 * GB, settledAvail(10 * GB, 200 * GB, &fits, Seq.measure, Seq.pause));
    try std.testing.expectEqual(@as(usize, 0), fits.i);
}

test "waitForUnload: only a load an unload just freed room for waits" {
    const GB: u64 = 1024 * 1024 * 1024;
    const now: i64 = 100_000;
    const freed: model_registry_mod.UnloadMark = .{ .at_ms = now - 500, .bytes = 150 * GB };
    try std.testing.expect(waitForUnload(180 * GB, 66 * GB, freed, now));
    try std.testing.expect(!waitForUnload(180 * GB, 66 * GB, .{}, now)); // no unload yet
    try std.testing.expect(!waitForUnload(180 * GB, 66 * GB, .{ .at_ms = now - 60_000, .bytes = 150 * GB }, now)); // long settled
    try std.testing.expect(!waitForUnload(500 * GB, 66 * GB, freed, now)); // even the freed bytes cannot fit it
    var taken = freed;
    taken.consume(150 * GB); // a load committed into the freed room
    try std.testing.expect(!waitForUnload(180 * GB, 66 * GB, taken, now));
}

test "effectiveAvailableBytes is capped by the GPU working-set limit" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(36 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB));
    try std.testing.expect(memInsufficientForLoad(70 * GB, effectiveAvailableBytes(98 * GB, 0, 36 * GB)));
}

test "effectiveAvailableBytes prefers the per-process jetsam headroom when present" {
    const GB: u64 = 1024 * 1024 * 1024;
    try std.testing.expectEqual(6 * GB, effectiveAvailableBytes(4 * GB, 6 * GB, 0)); // iOS: proc wins
    try std.testing.expectEqual(4 * GB, effectiveAvailableBytes(4 * GB, 0, 0)); // macOS: proc query = 0 → host
    try std.testing.expectEqual(@as(u64, 0), effectiveAvailableBytes(0, 0, 0)); // both unknown → 0 (never blocks)
}

test "the preflight bills an MTP sidecar the index never names, never an in-checkpoint head twice" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // A minimal safetensors file whose header carries one MTP key: 8-byte LE header length + JSON + data.
    const header = "{\"mtp.fc.weight\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}";
    var buf: [8 + header.len + 4]u8 = undefined;
    std.mem.writeInt(u64, buf[0..8], header.len, .little);
    @memcpy(buf[8 .. 8 + header.len], header);
    @memset(buf[8 + header.len ..], 0);
    try tmp.dir.createDirPath(io, "mtp");
    try tmp.dir.writeFile(io, .{ .sub_path = "mtp/weights.safetensors", .data = &buf });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"a\":\"model.safetensors\"}}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = "0123456789" });
    var cwd_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_ptr = std.c.getcwd(&cwd_buf, cwd_buf.len) orelse return error.NoCwd;
    const cwd = std.mem.span(@as([*:0]const u8, @ptrCast(cwd_ptr)));
    const dir = try std.fmt.allocPrint(std.testing.allocator, "{s}/.zig-cache/tmp/{s}", .{ cwd, tmp.sub_path });
    defer std.testing.allocator.free(dir);
    // The shard sum sees only what the index names; the sidecar is its own bill, and only when the head loads.
    try std.testing.expectEqual(@as(u64, 10), modelDiskBytes(io, dir));
    try std.testing.expectEqual(@as(u64, buf.len), mtpSidecarDiskBytes(io, std.testing.allocator, dir, true));
    try std.testing.expectEqual(@as(u64, 0), mtpSidecarDiskBytes(io, std.testing.allocator, dir, false));
    try tmp.dir.deleteFile(io, "mtp/weights.safetensors");
    try std.testing.expectEqual(@as(u64, 0), mtpSidecarDiskBytes(io, std.testing.allocator, dir, true));
}

test "media bill: CUDA checks the weights against RAM and only the working set against the GPU" {
    const GB: u64 = 1024 * 1024 * 1024;
    // Music3 8-bit: 13 GB of weights + its 6 GB generation buffer.
    const buffer = @import("../../../plugins/music3/config.zig").gen_buffer_bytes;
    const peak = 13 * GB + buffer;
    const cuda = mediaBill(peak, 13 * GB, true);
    try std.testing.expectEqual(13 * GB, cuda.ram);
    try std.testing.expectEqual(buffer, cuda.gpu);
    // Unified memory holds both in one pool, as before.
    const unified = mediaBill(peak, 13 * GB, false);
    try std.testing.expectEqual(peak, unified.ram);
    try std.testing.expectEqual(@as(u64, 0), unified.gpu);
    // A staged pack peaks below its file sum: the peak bounds the RAM bill, nothing is left for the GPU.
    const staged = mediaBill(15 * GB, 40 * GB, true);
    try std.testing.expectEqual(15 * GB, staged.ram);
    try std.testing.expectEqual(@as(u64, 0), staged.gpu);
}

test "a refusal quotes the number it actually compared" {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;

    // Live report FLUX.2-klein 4B refused with "generation peaks at
    // ~4.3 GB but only 5.4 GB is free" — two numbers that say the load should
    // have worked. The guard was right (it also wants ~1.5 GB of headroom for
    // warmup buffers) but the message quoted the PEAK, so the user went looking
    // for a problem that wasn't there and then tried the same prompt in the
    // other pane. What a refusal must state is the TOTAL it demanded.
    const peak: u64 = 4300 * MB;
    const avail: u64 = 5400 * MB;
    try std.testing.expect(memInsufficientForLoad(peak, avail));
    try std.testing.expect(loadRequirementBytes(peak) > avail);

    // The requirement IS the comparison — not a second formula that can drift
    // from it. At exactly the requirement a load is allowed; a byte under is not.
    try std.testing.expect(!memInsufficientForLoad(peak, loadRequirementBytes(peak)));
    try std.testing.expect(memInsufficientForLoad(peak, loadRequirementBytes(peak) - 1));
    try std.testing.expect(!memInsufficientForLoad(42 * GB, loadRequirementBytes(42 * GB)));
}

test "the eviction gate bills a media entry its BACKEND peak, never the dir's safetensors sum" {
    // #126. MiniMax-H3's four safetensors sum to 37.55 GiB, but the text
    // encoder RUNS AND IS FREED before the DiT loads, so the real staged peak
    // is 22.83 GiB. `doLoadGenOnInferenceThread`'s preflight already bills it
    // correctly — and never gets the chance, because this gate runs first and
    // billed the sum. On a 48 GB Mac (auto cap = 80% of a 38338 MB wired limit
    // = 30.0 GiB) that refused every load, permanently, on an idle server with
    // nothing to evict and nothing to wait for.
    const disk_sum: u64 = 40_316_668_515; // te + dit + video_vae + audio_vae
    const staged_peak: u64 = 24_511_876_594; // max(te, dit) + both vaes
    const cap: u64 = 30 * 1024 * 1024 * 1024;

    try testing.expect(gateEstimateBytes(0, disk_sum, 0, 0) > cap); // the bug
    try testing.expect(gateEstimateBytes(staged_peak, disk_sum, 0, 0) <= cap); // the fix

    // A media peak OUTRANKS bytes_on_disk — the whole point is that the two
    // disagree. It must not be averaged, summed or maxed with it, and it takes
    // no headroom: the KV/vision/drafter margin is a TEXT concept, the media
    // estimator carries its own transient term, and the commit
    // (`genLoadResidentBytes`) has always parked the bare peak.
    try testing.expectEqual(staged_peak, gateEstimateBytes(staged_peak, disk_sum, 0, 0));
    try testing.expectEqual(gateEstimateBytes(staged_peak, disk_sum, 0, 0), genLoadResidentBytes(staged_peak, disk_sum));

    // Non-media entries are BYTE-UNCHANGED: a zero peak means "not a media
    // model" (or a dir we could not read), and the old ladder stands.
    try testing.expectEqual(@as(u64, 1100), gateEstimateBytes(0, 1000, 0, 0));
    try testing.expectEqual(@as(u64, 0), gateEstimateBytes(0, 0, 0, 0));
    // No bytes_on_disk → the layers × hidden × 16 fallback, unchanged.
    const fallback: u64 = 32 * 4096 * 16;
    try testing.expectEqual(fallback + fallback / 10, gateEstimateBytes(0, null, 32, 4096));
}

test "a media model commits the residency the gate reserved" {
    // Secondary #1 of the issue: the gate reserved the staged peak and then
    // `markReadyLocked` committed the DIR SUM, so H3 sat in the budget at
    // 37.55 GB — 14.7 GB more than it can ever hold — and would evict a
    // genuinely-resident LLM to make room for bytes nobody was using.
    // Reserve-then-commit must read the same estimator, so the only difference
    // between them is the gate's 10% headroom.
    const staged_peak: u64 = 24_511_876_594;
    const disk_sum: u64 = 40_316_668_515;
    try testing.expectEqual(staged_peak, genLoadResidentBytes(staged_peak, disk_sum));
    // A backend with no staged plan keeps the sum (peak == sum there anyway).
    try testing.expectEqual(disk_sum, genLoadResidentBytes(0, disk_sum));
    // Neither known → 0, as before.
    try testing.expectEqual(@as(u64, 0), genLoadResidentBytes(0, 0));
}

test "a sidecar bills its loaded width, and only an in-dir one is dropped to fit" {
    const GB: u64 = 1024 * 1024 * 1024;
    // A dense bf16 assistant quantized at load: bits per weight plus a bf16
    // (scale, bias) pair per group of 64; a packed or dense-loaded one its own bytes.
    try std.testing.expectEqual(@as(u64, 32 * 9), drafterResidentBytes(32 * 32, 4));
    try std.testing.expectEqual(@as(u64, 32 * 17), drafterResidentBytes(32 * 32, 8));
    try std.testing.expectEqual(@as(u64, 1000), drafterResidentBytes(1000, 0));
    // 20 GB of weights want ~23.5 GB; a 3 GB drafter pushes that past 26 GB.
    try std.testing.expectEqual(PreflightVerdict.fits, preflightVerdict(20 * GB, 3 * GB, 30 * GB, true));
    try std.testing.expectEqual(PreflightVerdict.drop_drafter, preflightVerdict(20 * GB, 3 * GB, 25 * GB, true));
    try std.testing.expectEqual(PreflightVerdict.refuse, preflightVerdict(20 * GB, 3 * GB, 25 * GB, false));
    try std.testing.expectEqual(PreflightVerdict.refuse, preflightVerdict(20 * GB, 3 * GB, 20 * GB, true));
    try std.testing.expectEqual(PreflightVerdict.fits, preflightVerdict(20 * GB, 0, 25 * GB, true));
}

test "memInsufficientForLoad: headroom + unknown-query guards" {
    const GB: u64 = 1024 * 1024 * 1024;
    const MB: u64 = 1024 * 1024;
    // A 6.9 GB 4-bit model with ~10 GB available (free + speculative + the file
    // cache the OS will grant) — what a 16 GB Mac reports pre-load. Needs ~8.8 GB
    // (weights + weights/8 + 1 GB for warmup + baseline KV) → loads.
    try std.testing.expect(!memInsufficientForLoad(6900 * MB, 10 * GB));
    // Restart-into-pressure: 42 GB weights, only 44 GB free → needs ~46, refuse.
    try std.testing.expect(memInsufficientForLoad(42 * GB, 44 * GB));
    // #45 end-to-end: a prior 7 GB model resident in the anon set leaves 3 GB of
    // reclaimable classes — the guard must refuse the second 7 GB load, or warmup
    // dies in an uncatchable Metal OOM instead of this 400.
    try std.testing.expect(memInsufficientForLoad(7 * GB, 3 * GB));
    // Plenty of headroom → allow.
    try std.testing.expect(!memInsufficientForLoad(42 * GB, 86 * GB));
    // Exactly weights, no headroom → refuse.
    try std.testing.expect(memInsufficientForLoad(42 * GB, 42 * GB));
    // Unknown figures (query failed / size unknown) → never block.
    try std.testing.expect(!memInsufficientForLoad(0, 44 * GB));
    try std.testing.expect(!memInsufficientForLoad(42 * GB, 0));

    // A PROPORTIONAL margin becomes impossible at the top of the range. Our own
    // DeepSeek-V4-Flash mirror is 109.7 GB of weights and a 128 GB Mac reports
    // ~118 GB available with nothing else loaded — but weights/8 demanded 14.7
    // GB of headroom, i.e. 124.4 GB, which that machine cannot have. The guard
    // refused to load the flagship checkpoint on exactly the hardware its model
    // card names, while `--skip-mem-preflight` booted it repeatedly and served
    // 6.7K-token prefills with ~8.6 GB to spare. Headroom covers warmup
    // buffers + a baseline KV cache, and neither scales with a MoE's total
    // weights (13B active here) — so the proportional term is CAPPED.
    try std.testing.expect(!memInsufficientForLoad(109_730 * MB, 118_330 * MB));
    // Still refuses when the box genuinely cannot fit it.
    try std.testing.expect(memInsufficientForLoad(109_730 * MB, 112 * GB));
}

test "availForLoad: only a load that would be refused makes media residency let go" {
    // A box with no headroom refuses even a 1-byte load (GitHub's 7 GB runners): nothing "fits anywhere" there.
    if (memInsufficientForLoad(1, loadAvail(testing.io))) return error.SkipZigTest;
    var held = @import("../../../plugins/minimax_h3/h3.zig").Resident.init(std.testing.allocator);
    held.bytes = 1000;
    @import("../../../plugins/minimax_h3/h3.zig").registerResident(&held);
    defer @import("../../../plugins/minimax_h3/h3.zig").unregisterResident(&held);
    _ = availForLoad(testing.io, 1); // fits anywhere: the cache stays
    try std.testing.expectEqual(@as(u64, 1000), held.bytes);
    _ = availForLoad(testing.io, 1 << 50); // can never fit: the cache is released before the refusal
    try std.testing.expectEqual(@as(u64, 0), held.bytes);
}

test "a refusal right after a two-model eviction waits for the memory to come back" {
    const GiB: u64 = 1024 * 1024 * 1024;
    var mark: model_registry_mod.UnloadMark = .{};
    mark.record(1_000, 100 * GiB);
    mark.record(1_100, 17 * GiB);
    try std.testing.expect(waitForUnload(211 * GiB, 108 * GiB, mark, 1_200));
}

test "residentModelDiskBytes: with stream_experts a pack bills its residents and slot rows, not the experts" {
    if (!@import("build_options").mlx_stream) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = std.testing.tmpDir(.{});
    defer td.cleanup();
    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    try @import("../../../sdk/weights/streamed/mlx_pack_test.zig").writePack(dir);
    var config: model_mod.ModelConfig = .{ .num_experts = @import("../../../sdk/weights/streamed/mlx_pack_test.zig").E, .stream_experts_override = .{ .rows = 2 } };
    const b = try streamed.Streamed.packBytes(std.testing.allocator, dir, 2);
    try std.testing.expectEqual(modelDiskBytes(io, dir) - b.experts + b.slots, try residentModelDiskBytes(std.testing.allocator, io, dir, &config));
    config.stream_experts_override = null;
    try std.testing.expectEqual(modelDiskBytes(io, dir), try residentModelDiskBytes(std.testing.allocator, io, dir, &config));
}
