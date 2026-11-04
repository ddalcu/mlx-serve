//! Where a pack keeps its MTP head: a sidecar file beside the trunk, or `mtp.*` tensors in the checkpoint itself.
//! Header and index reads only; no weight loads.

const std = @import("std");
const testing = std.testing;

/// Sidecar file layouts we accept, in priority order. The native layout wins
/// so a repo shipping several keeps loading exactly what it loaded before.
/// Root-level names are what others publish (mutual compat: their
/// loader accepts our `mtp/weights.safetensors` too).
pub const sidecar_rel_paths = [_][]const u8{
    "mtp/weights.safetensors", // mlx-serve native (ddalcu repos, build_mtp_sidecar.py)
    "mtp.safetensors", // others
    "model-mtp.safetensors", // others
    "optiq/mtp.safetensors", // oMLX OptiQ (delta-encoded norms — folded at load)
    "mtp_head.safetensors", // Nemotron-H heads (sevren-ai packs, bare keys)
    "mtp-4bit.safetensors", // Nemotron-H heads (TensorFold's Vontra packs)
};

/// Relative path (one of `sidecar_rel_paths`) of the first sidecar file under
/// `dir` whose HEADER carries a marker key, or null when the model ships no
/// loadable MTP head. The marker gate is the same one discovery and the
/// in-checkpoint shard sweep apply — a file at a sidecar PATH is only a
/// sidecar when it provably contains a head this loader can bind (DeepSeek-V4
/// mirrors ship their own dsv4-shaped MTP module at `model-mtp.safetensors`;
/// claiming it by NAME alone sent the qwen-shaped loader into
/// MissingMtpWeight on every `--mtp` boot, live).
pub fn resolveMtpSidecarInDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) ?[]const u8 {
    for (&sidecar_rel_paths) |rel| {
        const st = dir.statFile(io, rel, .{}) catch continue;
        if (st.size == 0) continue;
        if (!safetensorsHeaderHasMtpHead(io, allocator, dir, rel)) continue;
        return rel;
    }
    return null;
}

/// Where a model's MTP head lives.
pub const MtpSource = union(enum) {
    /// A separate sidecar file — rel path, one of `sidecar_rel_paths`.
    sidecar_file: []const u8,
    /// Inside the main checkpoint safetensors (Qwen HF releases and oMLX
    /// oQ4e-class conversions ship `[language_model.]mtp.*` in the trunk
    /// shards). Loading reads ONLY the shards the index names for mtp keys.
    in_checkpoint,
};

/// Marker projections that prove a LOADABLE head: `fc` (Qwen dense/MoE
/// layouts) or `eh_proj` (hy3). Discovery and the shard sweep both gate on
/// this same set, so a checkpoint with stray `mtp.*` auxiliaries but no
/// marker never claims a head it can't bind.
const mtp_marker_keys = [_][]const u8{
    "mtp.fc.weight",
    "language_model.mtp.fc.weight",
    "mtp.eh_proj.weight",
    "language_model.mtp.eh_proj.weight",
    "attn.mixer.q_proj.weight", // Nemotron-H head (bare keys, dense `eh_proj`)
    "layers.0.eh_proj.weight", // Nemotron-H head (TensorFold names)
};

/// Any tensor belonging to the head (either root prefix).
fn isMtpHeadKey(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "mtp.") or
        std.mem.indexOf(u8, key, ".mtp.") != null;
}

/// Sanity bound for index.json / safetensors headers (the Jundot 27B index
/// is ~212 KB; headers of the largest checkpoints stay well under this).
pub const checkpoint_header_limit: usize = 64 * 1024 * 1024;

/// Parse a sharded checkpoint's `model.safetensors.index.json` and return
/// the unique shard basenames holding MTP-head tensors, in first-seen order
/// (caller frees each name + the slice). Empty when the checkpoint carries
/// no loadable head — the sweep is gated on `mtp_marker_keys`, so partial
/// auxiliaries never produce a doomed load.
pub fn mtpShardsFromIndexJson(allocator: std.mem.Allocator, bytes: []const u8) ![][]u8 {
    var out: std.ArrayList([]u8) = .empty;
    errdefer {
        for (out.items) |sh| allocator.free(sh);
        out.deinit(allocator);
    }
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch
        return out.toOwnedSlice(allocator);
    defer parsed.deinit();
    if (parsed.value != .object) return out.toOwnedSlice(allocator);
    const weight_map = parsed.value.object.get("weight_map") orelse
        return out.toOwnedSlice(allocator);
    if (weight_map != .object) return out.toOwnedSlice(allocator);

    var has_marker = false;
    for (&mtp_marker_keys) |marker| {
        if (weight_map.object.get(marker) != null) {
            has_marker = true;
            break;
        }
    }
    if (!has_marker) return out.toOwnedSlice(allocator);

    var it = weight_map.object.iterator();
    outer: while (it.next()) |entry| {
        if (!isMtpHeadKey(entry.key_ptr.*)) continue;
        if (entry.value_ptr.* != .string) continue;
        const shard = entry.value_ptr.string;
        for (out.items) |seen| {
            if (std.mem.eql(u8, seen, shard)) continue :outer;
        }
        try out.append(allocator, try allocator.dupe(u8, shard));
    }
    return out.toOwnedSlice(allocator);
}

/// Read a file under `dir` fully (bounded); null on absence or overflow.
pub fn readDirFileAlloc(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, sub_path: []const u8, limit: usize) ?[]u8 {
    const f = dir.openFile(io, sub_path, .{}) catch return null;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    return rs.interface.allocRemaining(allocator, .limited(limit)) catch null;
}

/// True when the sharded index names an in-checkpoint head.
fn indexJsonHasMtpHead(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) bool {
    const bytes = readDirFileAlloc(io, allocator, dir, "model.safetensors.index.json", checkpoint_header_limit) orelse return false;
    defer allocator.free(bytes);
    const shards = mtpShardsFromIndexJson(allocator, bytes) catch return false;
    defer {
        for (shards) |sh| allocator.free(sh);
        allocator.free(shards);
    }
    return shards.len > 0;
}

/// Single-file checkpoints have no index — peek the safetensors JSON header
/// (8-byte LE length prefix) for a marker key, without touching tensor data.
/// Marker names are plain ASCII (dots/letters), so a quoted substring scan
/// is exact — no JSON-escape variants exist for them.
fn safetensorsHeaderHasMtpHead(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, sub_path: []const u8) bool {
    const f = dir.openFile(io, sub_path, .{}) catch return false;
    defer f.close(io);
    var rb: [8192]u8 = undefined;
    var rs = f.reader(io, &rb);
    const header_len = rs.interface.takeInt(u64, .little) catch return false;
    if (header_len == 0 or header_len > checkpoint_header_limit) return false;
    const header = allocator.alloc(u8, @intCast(header_len)) catch return false;
    defer allocator.free(header);
    rs.interface.readSliceAll(header) catch return false;
    for (&mtp_marker_keys) |marker| {
        var quoted_buf: [64]u8 = undefined;
        const quoted = std.fmt.bufPrint(&quoted_buf, "\"{s}\"", .{marker}) catch continue;
        if (std.mem.indexOf(u8, header, quoted) != null) return true;
    }
    return false;
}

/// Resolve where (if anywhere) this model's MTP head lives. A sidecar file
/// always outranks an in-checkpoint head so repos shipping both keep
/// loading exactly what they loaded before.
pub fn resolveMtpSource(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) ?MtpSource {
    if (resolveMtpSidecarInDir(io, allocator, dir)) |rel| return .{ .sidecar_file = rel };
    if (indexJsonHasMtpHead(io, allocator, dir)) return .in_checkpoint;
    if (safetensorsHeaderHasMtpHead(io, allocator, dir, "model.safetensors")) return .in_checkpoint;
    return null;
}

/// Qwen3.8-Flash-Next's head is the checkpoint's own layer, loaded by the
/// trunk (`loadQwen4Mtp`), never through `resolveMtpSource`.
const qwen4_mtp_marker = "\"language_model.mtp.fc_hidden.weight\"";
const qwen4_root_mtp_marker = "\"mtp.fc_hidden.weight\"";

test "qwen4 root MTP head is advertised from the shard index" {
    const io = testing.io;
    const allocator = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"mtp.fc_hidden.weight\":\"model-00001.safetensors\"}}" });
    try testing.expect(dirAdvertisesMtp(io, allocator, tmp.dir));
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try tmp.dir.realPath(io, &path_buf);
    try testing.expect(!hasMtpHead(io, allocator, path_buf[0..path_len]));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"mtp.fc_embedding.weight\":\"model-00001.safetensors\"}}" });
    try testing.expect(!dirAdvertisesMtp(io, allocator, tmp.dir));
}

fn indexJsonHasQwen4Mtp(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) bool {
    const bytes = readDirFileAlloc(io, allocator, dir, "model.safetensors.index.json", checkpoint_header_limit) orelse return false;
    defer allocator.free(bytes);
    return std.mem.indexOf(u8, bytes, qwen4_mtp_marker) != null or
        std.mem.indexOf(u8, bytes, qwen4_root_mtp_marker) != null;
}

/// Advertisement probe: ANY head the server can run, including qwen4's.
pub fn dirAdvertisesMtp(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) bool {
    return resolveMtpSource(io, allocator, dir) != null or indexJsonHasQwen4Mtp(io, allocator, dir);
}

/// True when `model_dir` carries a head for the generic MTP loader.
/// `model_dir` is absolute (same contract as `model.parseConfig`).
pub fn hasMtpHead(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) bool {
    if (model_dir.len == 0 or !std.fs.path.isAbsolute(model_dir)) return false;
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    return resolveMtpSource(io, allocator, dir) != null;
}
