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
    "gemma4_unified", "gemma4_unified_text",
    "qwen3",        "qwen3_5",        "qwen3_5_text",
    "qwen3_5_moe",  "qwen3_5_moe_text",
    "qwen3_moe",    "qwen3_moe_text",
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

/// Quantization modes the MLX loader supports. Must stay in sync with
/// `model.zig:QuantMode` (discovery deliberately avoids importing model.zig,
/// which would drag the mlx FFI into this filesystem-only module).
const supported_quant_modes = [_][]const u8{ "affine", "nvfp4", "mxfp4", "mxfp8" };

fn isSupportedQuantMode(mode: []const u8) bool {
    for (supported_quant_modes) |m| {
        if (std.mem.eql(u8, mode, m)) return true;
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
///   - unsupported quantization modes (anything outside
///     `supported_quant_modes` — affine, nvfp4, mxfp4, mxfp8).
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
    // only the schemes the loader supports. Models without a quantization
    // block (bf16 / unquantized) pass through.
    if (root.get("quantization")) |q_val| {
        if (q_val == .object) {
            if (q_val.object.get("mode")) |mode_val| {
                if (mode_val == .string and !isSupportedQuantMode(mode_val.string)) {
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
    /// `model_type` peeked from config.json (e.g. "bert"), so registry stubs
    /// can advertise arch-derived capabilities before a cold load. Empty
    /// when unknown.
    model_type: []const u8 = "",
};

pub const DiscoveryResult = struct {
    models: []DiscoveredModel,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *DiscoveryResult) void {
        for (self.models) |*m| {
            self.allocator.free(m.id);
            self.allocator.free(m.path);
            if (m.model_type.len > 0) self.allocator.free(m.model_type);
        }
        self.allocator.free(self.models);
    }
};

/// True if a `.gguf` basename is the DeepSeek-V4-Flash model served by the ds4
/// engine (case-insensitive `deepseek-v4-flash` prefix). Every other GGUF routes
/// to the generic llama.cpp engine — libllama can't load the DSV4-Flash
/// architecture, which is why ds4 exists. Mirrors the Swift app's
/// `isSupportedDsv4Gguf` so client and server agree on which GGUFs are ds4.
pub fn isDs4GgufBasename(name: []const u8) bool {
    const prefix = "deepseek-v4-flash";
    if (name.len < prefix.len) return false;
    for (prefix, 0..) |c, i| {
        if (std.ascii.toLower(name[i]) != c) return false;
    }
    return true;
}

/// True if a `.gguf` basename is a multimodal-projection sidecar (CLIP
/// vision / audio encoder packaged separately so the language model can
/// reference it at runtime). llama.cpp tooling, ollama, and LM Studio all
/// use the `mmproj-*` prefix for this; `llama_model_load_from_file` refuses
/// them with `unsupported model architecture: 'clip'`. Filtering them out
/// at directory-pick time lets a user point at a model folder (which
/// commonly ships both the LLM and the mmproj sidecar — Gemma 4 VL, Qwen
/// 3.6 VL, etc.) and have the right file get loaded.
///
/// Match is a case-insensitive `mmproj` prefix + `.gguf` suffix. `mmproj.gguf`
/// itself matches; `model-mmproj.gguf` (suffix, not prefix) does NOT —
/// only basenames starting with the prefix are sidecars in the wild.
pub fn isMmprojGgufBasename(basename: []const u8) bool {
    if (basename.len < 7 or !std.mem.endsWith(u8, basename, ".gguf")) return false;
    const prefix = "mmproj";
    if (basename.len < prefix.len) return false;
    for (basename[0..prefix.len], prefix) |c, p| {
        if (std.ascii.toLower(c) != p) return false;
    }
    return true;
}

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
            if (m.model_type.len > 0) allocator.free(m.model_type);
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
        //   - unsupported quants (modes outside supported_quant_modes)
        // before they reach the tokenizer/weight loaders.
        const model_type: []const u8 = switch (peekConfig(io, allocator, dir, entry.name)) {
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
                log.info("[discovery] skip {s}: unsupported quantization mode '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)", .{ entry.name, mode });
                continue;
            },
            .supported => |mt| mt, // ownership moves to the DiscoveredModel
        };
        errdefer if (model_type.len > 0) allocator.free(model_type);

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
            .model_type = model_type,
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

pub const ProbeResult = struct {
    /// Owned dupe of the supported model_type.
    model_type: []const u8,
    /// Sum of *.safetensors bytes; null if the scan failed.
    bytes_on_disk: ?u64,
};

/// Validate an arbitrary absolute model directory the way discovery would
/// (config.json present, supported model_type and quant mode) and report its
/// weight bytes. Used by /v1/load-model's register-by-path branch for models
/// OUTSIDE the --model-dir scan — e.g. the app's auto-downloaded embedding
/// encoder, which lands wherever the download root is regardless of which
/// org dir the chat model came from.
pub fn probeModelDir(io: std.Io, allocator: std.mem.Allocator, abs_path: []const u8) !ProbeResult {
    const trimmed = trimTrailingSlash(abs_path);
    const base = std.fs.path.basename(trimmed);
    const parent = std.fs.path.dirname(trimmed) orelse return error.InvalidModelPath;
    if (base.len == 0 or parent.len == 0) return error.InvalidModelPath;

    var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return error.ModelDirNotFound;
    defer dir.close(io);

    const model_type: []const u8 = switch (peekConfig(io, allocator, dir, base)) {
        .missing_or_unparseable => return error.ModelDirNotFound,
        .unsupported_arch => |mt| {
            allocator.free(mt);
            return error.UnsupportedArch;
        },
        .unsupported_quant => |mode| {
            allocator.free(mode);
            return error.UnsupportedQuantMode;
        },
        .supported => |mt| mt,
    };
    errdefer allocator.free(model_type);

    var bytes: u64 = 0;
    var bytes_ok = false;
    var sub = dir.openDir(io, base, .{ .iterate = true }) catch null;
    if (sub) |*sd| {
        defer sd.close(io);
        var it = sd.iterate();
        while (it.next(io) catch null) |entry| {
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
            const st = sd.statFile(io, entry.name, .{}) catch continue;
            bytes += @intCast(st.size);
            bytes_ok = true;
        }
    }
    return .{ .model_type = model_type, .bytes_on_disk = if (bytes_ok) bytes else null };
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

test "isDs4GgufBasename routes DSV4 to ds4 and everything else to llama" {
    // DeepSeek-V4-Flash → ds4 (case-insensitive).
    try testing.expect(isDs4GgufBasename("DeepSeek-V4-Flash-Q4_K_M.gguf"));
    try testing.expect(isDs4GgufBasename("deepseek-v4-flash-bf16.gguf"));
    // Any other GGUF → llama.cpp engine.
    try testing.expect(!isDs4GgufBasename("qwen2.5-0.5b-instruct-q4_k_m.gguf"));
    try testing.expect(!isDs4GgufBasename("Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"));
    try testing.expect(!isDs4GgufBasename("deepseek-v3-chat.gguf")); // V3, not V4-Flash
    try testing.expect(!isDs4GgufBasename("short.gguf"));
}

test "isMmprojGgufBasename catches the multimodal-projection sidecars" {
    // Real mmproj files seen in the wild (Gemma 4 VL, Qwen 3.6 VL, ...).
    try testing.expect(isMmprojGgufBasename("mmproj-gemma-4-E4B-it-BF16.gguf"));
    try testing.expect(isMmprojGgufBasename("mmproj-gemma-4-E2B-it-BF16.gguf"));
    try testing.expect(isMmprojGgufBasename("mmproj-F32.gguf"));
    try testing.expect(isMmprojGgufBasename("mmproj-Qwen3.6-27B-VL-BF16.gguf"));
    // Case-insensitive on the prefix only.
    try testing.expect(isMmprojGgufBasename("MMPROJ-foo.gguf"));
    try testing.expect(isMmprojGgufBasename("MmProj-bar.gguf"));
    // Bare prefix.gguf — also a sidecar.
    try testing.expect(isMmprojGgufBasename("mmproj.gguf"));

    // Real LLM .gguf — must NOT match (this is the regression class:
    // pre-fix, the directory-picker grabbed the alphabetically-first
    // file and that file was the mmproj sidecar).
    try testing.expect(!isMmprojGgufBasename("gemma-4-E4B-it-Q4_K_M.gguf"));
    try testing.expect(!isMmprojGgufBasename("Qwen3.5-4B-IQ4_NL.gguf"));
    try testing.expect(!isMmprojGgufBasename("DeepSeek-V4-Flash-Q4_K_M.gguf"));
    // Not a .gguf → not a sidecar.
    try testing.expect(!isMmprojGgufBasename("mmproj-readme.md"));
    try testing.expect(!isMmprojGgufBasename("mmproj"));
    // Suffix-only — model-mmproj.gguf is NOT the convention.
    try testing.expect(!isMmprojGgufBasename("model-mmproj.gguf"));
}

test "isSupportedModelType accepts qwen3_moe (Qwen3-30B-A3B)" {
    // Regression for the "[discovery] skip ...: unsupported model_type
    // 'qwen3_moe'" warning: Qwen3-30B-A3B / Qwen3-Coder-30B-A3B must be
    // discoverable by the model manager, not silently skipped.
    try testing.expect(isSupportedModelType("qwen3_moe"));
    try testing.expect(isSupportedModelType("qwen3_moe_text"));
    // Sibling arches still recognized.
    try testing.expect(isSupportedModelType("qwen3_5_moe"));
    try testing.expect(isSupportedModelType("qwen3"));
    // A genuinely unknown arch is still rejected.
    try testing.expect(!isSupportedModelType("totally_made_up_arch"));
}

test "isSupportedQuantMode accepts nvfp4 (issue #24), rejects unknown" {
    // Regression for "[discovery] skip ...: unsupported quantization mode
    // 'nvfp4'": nvfp4 / mxfp4 / mxfp8 checkpoints are loadable and must be
    // discoverable.
    try testing.expect(isSupportedQuantMode("affine"));
    try testing.expect(isSupportedQuantMode("nvfp4"));
    try testing.expect(isSupportedQuantMode("mxfp4"));
    try testing.expect(isSupportedQuantMode("mxfp8"));
    try testing.expect(!isSupportedQuantMode("fp99"));
}
