//! Plan 05 — model discovery (Phase 1 minimal).
//!
//! Walks a directory looking for subdirectories that contain `config.json`,
//! treats each as a discoverable model. Used by `--model-dir` to enumerate
//! candidate models for `/v1/models` listing without loading them.
//!
//! v1 scope: discovery + listing only. Request routing still goes to the
//! single `--model` loaded at startup; if the user passes `--model-dir`
//! WITHOUT `--model`, we pick the first discovered model as the loaded one
//! and surface the rest as `loaded:false` siblings.
//!
//! On-demand load + LRU eviction live in plan 05 phases 2-5 and depend on
//! plan 01 Phase 0 (detangling Transformer state).

const std = @import("std");
const log = @import("log.zig");

/// Architecture allow-list for discovery. Must stay in sync with the
/// `model_type` branches in `model.zig:parseConfigFromJson`. Discovery
/// silently skips any subdirectory whose `config.json` declares a
/// `model_type` outside this list — that prevents `--model-dir` from
/// picking up partially-downloaded or unsupported checkpoints (e.g. a
/// `deepseek_v4` directory next to gemma/qwen ones) which would otherwise
/// crash the server when the tokenizer for the unknown arch is loaded.
///
/// `gemma4_assistant` is deliberately excluded: those are speculative-
/// decoding drafters, not standalone primary models. Bare `gemma4`/`qwen3`
/// drafters can't decode on their own, and users shouldn't see them in
/// `/v1/models`.
const supported_model_types = [_][]const u8{
    "gemma3",
    "gemma4",       "gemma4_text",
    "qwen3",        "qwen3_5",        "qwen3_5_text",
    "qwen3_5_moe",  "qwen3_5_moe_text",
    "qwen3_next",
    "llama",        "mistral",
    "lfm2",         // also matches any "lfm2*" prefix (lfm2_vl etc. when added)
    "nemotron_h",
    "bert",
    "deepseek_v4",
};

fn isSupportedModelType(model_type: []const u8) bool {
    if (std.mem.startsWith(u8, model_type, "lfm2")) return true;
    for (supported_model_types) |t| {
        if (std.mem.eql(u8, model_type, t)) return true;
    }
    return false;
}

/// Outcome of reading a candidate's config.json. Discovery treats any
/// non-`.supported` result as "skip this directory."
const ConfigPeek = union(enum) {
    supported: []const u8, // owned dupe of model_type
    unsupported_arch: []const u8, // owned dupe of model_type
    unsupported_quant: []const u8, // owned dupe of quantization.mode
    missing_or_unparseable,
};

/// Peek at a candidate's `config.json`: classify by `model_type` and
/// `quantization.mode`. Discovery uses this to filter out:
///   - unsupported archs (e.g. deepseek_v4, which crashes the tokenizer)
///   - non-affine quantizations (e.g. nvfp4 / NVIDIA FP4 — same
///     `model_type` as a normal MLX qwen, but a weight layout our
///     safetensors loader can't decode and which aborts on tensor read).
///
/// Returned strings are owned by `allocator`; the caller frees them via
/// the helpers in `freeConfigPeek`.
fn peekConfig(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, entry_name: []const u8) ConfigPeek {
    var sub = dir.openDir(io, entry_name, .{}) catch return .missing_or_unparseable;
    defer sub.close(io);
    var file = sub.openFile(io, "config.json", .{}) catch return .missing_or_unparseable;
    defer file.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const bytes = rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch return .missing_or_unparseable;
    defer allocator.free(bytes);
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return .missing_or_unparseable;
    defer parsed.deinit();
    const root = parsed.value.object;
    const mt_val = root.get("model_type") orelse return .missing_or_unparseable;
    if (mt_val != .string) return .missing_or_unparseable;
    if (!isSupportedModelType(mt_val.string)) {
        const dup = allocator.dupe(u8, mt_val.string) catch return .missing_or_unparseable;
        return .{ .unsupported_arch = dup };
    }
    // Quantization gate: if a model declares a `quantization.mode`, accept
    // only the standard MLX `affine` scheme. Models without a quantization
    // block (bf16 / unquantized) pass through.
    if (root.get("quantization")) |q_val| {
        if (q_val == .object) {
            if (q_val.object.get("mode")) |mode_val| {
                if (mode_val == .string and !std.mem.eql(u8, mode_val.string, "affine")) {
                    const dup = allocator.dupe(u8, mode_val.string) catch return .missing_or_unparseable;
                    return .{ .unsupported_quant = dup };
                }
            }
        }
    }
    return .{ .supported = allocator.dupe(u8, mt_val.string) catch return .missing_or_unparseable };
}

pub const DiscoveredModel = struct {
    /// Model id (subdirectory basename, e.g. "gemma-4-e4b-it-4bit").
    id: []const u8,
    /// Absolute path to the model directory.
    path: []const u8,
    /// Approximate weight size on disk in bytes (sum of *.safetensors). Used
    /// later by eviction; null if scan failed.
    bytes_on_disk: ?u64,
};

pub const DiscoveryResult = struct {
    models: []DiscoveredModel,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *DiscoveryResult) void {
        for (self.models) |*m| {
            self.allocator.free(m.id);
            self.allocator.free(m.path);
        }
        self.allocator.free(self.models);
    }
};

/// Scan `model_dir` for subdirectories containing `config.json`.
/// Returns DiscoveryResult; caller owns memory via deinit().
/// Symlinks followed; permission errors on individual subdirs skipped silently.
pub fn discoverModels(io: std.Io, allocator: std.mem.Allocator, model_dir: []const u8) !DiscoveryResult {
    var dir = std.Io.Dir.openDirAbsolute(io, model_dir, .{ .iterate = true }) catch |err| {
        return err;
    };
    defer dir.close(io);

    var found = std.ArrayList(DiscoveredModel).empty;
    errdefer {
        for (found.items) |*m| {
            allocator.free(m.id);
            allocator.free(m.path);
        }
        found.deinit(allocator);
    }

    var iter = dir.iterate();
    while (try iter.next(io)) |entry| {
        if (entry.kind != .directory and entry.kind != .sym_link) continue;
        if (entry.name.len == 0 or entry.name[0] == '.') continue;

        // Confirm it has a config.json
        var sub = dir.openDir(io, entry.name, .{}) catch continue;
        defer sub.close(io);

        const cfg_stat = sub.statFile(io, "config.json", .{}) catch continue;
        if (cfg_stat.kind != .file) continue;

        // Filter by supported model_type AND quantization scheme. Catches:
        //   - partially-downloaded checkpoints (missing/garbage config)
        //   - unsupported arches (e.g. deepseek_v4, MLA + indexer)
        //   - unsupported quants (e.g. nvfp4, which shares model_type with
        //     standard MLX qwens but uses a different weight layout)
        // before they reach the tokenizer/weight loaders.
        switch (peekConfig(io, allocator, dir, entry.name)) {
            .missing_or_unparseable => {
                log.info("[discovery] skip {s}: config.json missing or unparseable", .{entry.name});
                continue;
            },
            .unsupported_arch => |mt| {
                defer allocator.free(mt);
                log.info("[discovery] skip {s}: unsupported model_type '{s}'", .{ entry.name, mt });
                continue;
            },
            .unsupported_quant => |mode| {
                defer allocator.free(mode);
                log.info("[discovery] skip {s}: unsupported quantization mode '{s}' (only 'affine' supported)", .{ entry.name, mode });
                continue;
            },
            .supported => |mt| allocator.free(mt),
        }

        // Compute weight bytes (sum of *.safetensors sizes) — best-effort.
        var bytes: u64 = 0;
        var bytes_ok = false;
        var sub_iter_dir = dir.openDir(io, entry.name, .{ .iterate = true }) catch null;
        if (sub_iter_dir) |*sd| {
            defer sd.close(io);
            var sd_iter = sd.iterate();
            while (sd_iter.next(io) catch null) |sub_entry| {
                if (sub_entry.kind != .file) continue;
                if (!std.mem.endsWith(u8, sub_entry.name, ".safetensors")) continue;
                const st = sd.statFile(io, sub_entry.name, .{}) catch continue;
                bytes += @intCast(st.size);
                bytes_ok = true;
            }
        }

        const id = try allocator.dupe(u8, entry.name);
        const path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ trimTrailingSlash(model_dir), entry.name });
        try found.append(allocator, .{
            .id = id,
            .path = path,
            .bytes_on_disk = if (bytes_ok) bytes else null,
        });
    }

    // Stable order: by id ascending, so listing is deterministic.
    std.sort.pdq(DiscoveredModel, found.items, {}, lessThanById);

    return .{
        .models = try found.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

fn trimTrailingSlash(s: []const u8) []const u8 {
    var p = s;
    while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    return p;
}

fn lessThanById(_: void, a: DiscoveredModel, b: DiscoveredModel) bool {
    return std.mem.lessThan(u8, a.id, b.id);
}

// ── Tests ──

const testing = std.testing;

test "trimTrailingSlash" {
    try testing.expectEqualStrings("foo", trimTrailingSlash("foo/"));
    try testing.expectEqualStrings("foo", trimTrailingSlash("foo//"));
    try testing.expectEqualStrings("foo", trimTrailingSlash("foo"));
    try testing.expectEqualStrings("", trimTrailingSlash("//"));
}

test "lessThanById sorts ascending" {
    const a: DiscoveredModel = .{ .id = "a", .path = "x", .bytes_on_disk = null };
    const b: DiscoveredModel = .{ .id = "b", .path = "x", .bytes_on_disk = null };
    try testing.expect(lessThanById({}, a, b));
    try testing.expect(!lessThanById({}, b, a));
    try testing.expect(!lessThanById({}, a, a));
}
