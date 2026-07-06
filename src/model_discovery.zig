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
    "gemma3",       "gemma3_text",
    "gemma4",       "gemma4_text",
    "gemma4_unified", "gemma4_unified_text",
    "diffusion_gemma",
    "qwen2",
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

/// Native media-generation archs (image / audio / video / 3D), served by the
/// unified engines in `gen.zig`. Recognized here so `--model-dir` discovery
/// and `/v1/load-model` by-path accept them; the modality engine (not the MLX
/// transformer) handles the load. Kept as inline string checks so this module
/// stays filesystem-only (no mlx/gen import). Mirrors `gen.modalityFromType`.
pub fn isMediaModelType(model_type: []const u8) bool {
    return std.mem.startsWith(u8, model_type, "flux2") or
        std.mem.startsWith(u8, model_type, "krea") or
        std.mem.eql(u8, model_type, "qwen3_tts") or
        std.mem.eql(u8, model_type, "acestep") or
        std.mem.eql(u8, model_type, "AudioVideo") or
        std.mem.startsWith(u8, model_type, "hunyuan3d");
}

fn isSupportedModelType(model_type: []const u8) bool {
    if (std.mem.startsWith(u8, model_type, "lfm2")) return true;
    if (isMediaModelType(model_type)) return true;
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
    // Media models manage their own per-component quantization (the top-level
    // config may declare a mode the MLX loader doesn't, e.g. a DiT scheme), so
    // they bypass the LM quant gate below.
    if (isMediaModelType(mt_val.string)) {
        return .{ .supported = allocator.dupe(u8, mt_val.string) catch return .missing_or_unparseable };
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

/// Result of scanning a directory for LLM `.gguf` files (mmproj sidecars
/// excluded). `pick` is the alphabetically-smallest LLM gguf basename — the
/// same deterministic file `resolveGgufFile` loads — so callers can report
/// the bytes that will actually become resident, not the sum of every quant
/// in a multi-quant repo.
const GgufScan = struct {
    pick: ?[]u8 = null,
    pick_bytes: u64 = 0,
    saw_mmproj: bool = false,
};

/// Scan an iterable dir for LLM `.gguf` entries. Symlinked files count
/// (statFile follows links) — users symlink multi-GB weights rather than
/// copy them. Caller frees `pick`.
fn scanLlmGguf(io: std.Io, allocator: std.mem.Allocator, dir: *std.Io.Dir) !GgufScan {
    var scan: GgufScan = .{};
    errdefer if (scan.pick) |p| allocator.free(p);
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".gguf")) continue;
        if (isMmprojGgufBasename(entry.name)) {
            scan.saw_mmproj = true;
            continue;
        }
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind != .file) continue;
        if (scan.pick == null or std.mem.lessThan(u8, entry.name, scan.pick.?)) {
            if (scan.pick) |p| allocator.free(p);
            scan.pick = try allocator.dupe(u8, entry.name);
            scan.pick_bytes = @intCast(st.size);
        }
    }
    return scan;
}

/// True if `path` points at a .gguf file or a directory containing an LLM
/// one (mmproj sidecars don't count — a folder holding only an mmproj file
/// is not a valid LLM path). Accepts directories so users can pass the
/// canonical `~/.mlx-serve/models/<owner>/<repo>/` shape.
///
/// Empty / non-absolute paths (e.g. headless boot with no --model) return
/// false — guarded BEFORE `openDirAbsolute`, which ASSERTS the path is
/// absolute (`unreachable` on "") and in ReleaseFast that's UB that
/// miscompiles the caller (see the openDirAbsolute gotcha in CLAUDE.md).
pub fn isGgufModelPath(io: std.Io, path: []const u8) bool {
    if (path.len == 0 or !std.fs.path.isAbsolute(path)) return false;
    // A direct .gguf file path always routes to the gguf branch so
    // `resolveGgufFile` can emit a precise error if it's actually an
    // mmproj sidecar (falling through to the MLX path would produce an
    // opaque "no config.json" failure instead).
    if (std.mem.endsWith(u8, path, ".gguf")) return true;
    var dir = std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true }) catch return false;
    defer dir.close(io);
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".gguf")) continue;
        if (isMmprojGgufBasename(entry.name)) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind == .file) return true;
    }
    return false;
}

/// Resolve the actual .gguf file path. When `path` is a directory, return
/// the alphabetically-smallest non-mmproj `.gguf` entry within it (caller
/// frees) — deterministic so "load order depends on readdir(3) iteration
/// order" can't happen and the user can predict which quant loads when both
/// `Q4_K_M.gguf` and `Q8_0.gguf` sit in one folder. When `path` is already
/// a file, return a dup. Errors:
///   error.NoGgufFile         — no .gguf files at all
///   error.OnlyMmprojGgufFile — directory (or path) had only mmproj sidecars
///
/// Does NOT log on error — the caller decides whether the error is "fatal
/// user load" (then call `logResolveGgufError`) or "silent probe".
pub fn resolveGgufFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.mem.endsWith(u8, path, ".gguf")) {
        if (isMmprojGgufBasename(std.fs.path.basename(path))) {
            return error.OnlyMmprojGgufFile;
        }
        return allocator.dupe(u8, path);
    }
    var dir = try std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true });
    defer dir.close(io);
    const scan = try scanLlmGguf(io, allocator, &dir);
    if (scan.pick) |p| {
        defer allocator.free(p);
        return std.fmt.allocPrint(allocator, "{s}/{s}", .{ trimTrailingSlash(path), p });
    }
    if (scan.saw_mmproj) return error.OnlyMmprojGgufFile;
    return error.NoGgufFile;
}

/// Emit a user-facing, actionable error message for the failures
/// `resolveGgufFile` can return. Call from the fatal load path; probes
/// should NOT log and let the eventual resolveGgufFile re-attempt surface
/// the error once.
pub fn logResolveGgufError(path: []const u8, err: anyerror) void {
    switch (err) {
        error.OnlyMmprojGgufFile => {
            // Discriminate between "user pointed at the mmproj file
            // directly" and "directory had only mmproj sidecars".
            if (std.mem.endsWith(u8, path, ".gguf")) {
                log.err("'{s}' is an mmproj sidecar (CLIP vision/audio encoder), not an LLM. Point at the language-model .gguf (typically the same directory, e.g. `*-Q4_K_M.gguf`).\n", .{path});
            } else {
                log.err("'{s}' contains only mmproj sidecars (multimodal projection / CLIP encoders). Download or move the matching language-model .gguf (e.g. `*-Q4_K_M.gguf`) into this directory.\n", .{path});
            }
        },
        error.NoGgufFile => log.err("'{s}' contains no .gguf files.\n", .{path}),
        else => log.err("resolveGgufFile('{s}'): {s}\n", .{ path, @errorName(err) }),
    }
}

/// Coarse classification of a model for UX surfaces — the `mlx-serve list`
/// TYPE column and the `run` chat-REPL preflight. Mirrors how serving
/// actually routes the directory (media engines, embedded GGUF engines,
/// encoder-only, drafter sidecars).
pub const ModelKind = enum {
    chat,
    image,
    audio,
    video,
    mesh,
    embed,
    drafter,
    unsupported,

    /// Short label for the `list` TYPE column.
    pub fn label(self: ModelKind) []const u8 {
        return switch (self) {
            .chat => "chat",
            .image => "image",
            .audio => "audio",
            .video => "video",
            .mesh => "3d",
            .embed => "embed",
            .drafter => "drafter",
            .unsupported => "unsupported",
        };
    }

    /// Human phrase for refusal messages ("'X' is <describe>").
    pub fn describe(self: ModelKind) []const u8 {
        return switch (self) {
            .chat => "a chat model",
            .image => "an image generation model",
            .audio => "an audio generation model",
            .video => "a video generation model",
            .mesh => "a 3D generation model",
            .embed => "an embedding encoder (use /v1/embeddings)",
            .drafter => "a speculative-decoding drafter sidecar, not a standalone model (load it via --drafter beside a Gemma 4 target)",
            .unsupported => "an architecture mlx-serve does not support",
        };
    }

    /// The generation endpoint that DOES serve this kind, when one exists.
    pub fn genEndpoint(self: ModelKind) ?[]const u8 {
        return switch (self) {
            .image => "/v1/images/generations",
            .audio => "/v1/audio/speech (TTS) or /v1/audio/music-generations (music)",
            .video => "/v1/video/generations",
            .mesh => "/v1/3d/generations",
            else => null,
        };
    }
};

/// Map a config.json `model_type` to its ModelKind. "gguf" is the synthetic
/// type discovery assigns to GGUF dirs — chat via the embedded engines.
pub fn modelKindFromType(model_type: []const u8) ModelKind {
    if (std.mem.eql(u8, model_type, "bert")) return .embed;
    if (std.mem.endsWith(u8, model_type, "_assistant")) return .drafter;
    if (std.mem.startsWith(u8, model_type, "flux2") or
        std.mem.startsWith(u8, model_type, "krea")) return .image;
    if (std.mem.eql(u8, model_type, "qwen3_tts") or
        std.mem.eql(u8, model_type, "acestep")) return .audio;
    if (std.mem.eql(u8, model_type, "AudioVideo")) return .video;
    if (std.mem.startsWith(u8, model_type, "hunyuan3d")) return .mesh;
    if (std.mem.eql(u8, model_type, "gguf")) return .chat;
    if (isSupportedModelType(model_type)) return .chat;
    return .unsupported;
}

/// Classify an ABSOLUTE model dir. An LLM `.gguf` wins (embedded chat
/// engine — same precedence as routing); else config.json's model_type;
/// null when the dir is not model-shaped at all.
pub fn classifyModelPath(io: std.Io, allocator: std.mem.Allocator, abs_path: []const u8) ?ModelKind {
    if (abs_path.len == 0 or !std.fs.path.isAbsolute(abs_path)) return null;
    if (isGgufModelPath(io, abs_path)) return .chat;
    const trimmed = trimTrailingSlash(abs_path);
    const base = std.fs.path.basename(trimmed);
    const parent = std.fs.path.dirname(trimmed) orelse return null;
    if (base.len == 0 or parent.len == 0) return null;
    var dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return null;
    defer dir.close(io);
    return switch (peekConfig(io, allocator, dir, base)) {
        .missing_or_unparseable => null,
        // Raw model_type still classifies the KIND even when serving would
        // skip it (drafters, vit, ...) — that's exactly what the label is for.
        .unsupported_arch => |mt| blk: {
            defer allocator.free(mt);
            break :blk modelKindFromType(mt);
        },
        .unsupported_quant => |mode| blk: {
            allocator.free(mode);
            break :blk .unsupported;
        },
        .supported => |mt| blk: {
            defer allocator.free(mt);
            break :blk modelKindFromType(mt);
        },
    };
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
    return discoverModelsInDir(io, allocator, dir, model_dir);
}

/// Core scan over an already-open root. Two layouts are recognized:
///   <root>/<model>/config.json               → id "<model>"
///   <root>/<org>/<model>/config.json         → id "<org>/<model>"
/// The second is the HF-style layout `~/.mlx-serve/models` uses (the app's
/// DownloadManager and `mlx-serve pull` both write there).
pub fn discoverModelsInDir(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir, model_dir: []const u8) !DiscoveryResult {
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

        const is_model_dir = try tryAddModel(io, allocator, dir, entry.name, "", model_dir, &found);
        if (is_model_dir) continue;

        // No config.json at this level — maybe an org dir (org/repo layout).
        var org = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
        defer org.close(io);
        var org_iter = org.iterate();
        while (org_iter.next(io) catch null) |org_entry| {
            if (org_entry.kind != .directory and org_entry.kind != .sym_link) continue;
            if (org_entry.name.len == 0 or org_entry.name[0] == '.') continue;
            _ = try tryAddModel(io, allocator, org, org_entry.name, entry.name, model_dir, &found);
        }
    }

    // Stable order: by id ascending, so listing is deterministic.
    std.sort.pdq(DiscoveredModel, found.items, {}, lessThanById);

    return .{
        .models = try found.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

/// Inspect `parent/<name>` as a model-dir candidate; append to `found` if
/// it holds an LLM `.gguf` or a supported config.json. Returns true when
/// the dir was model-shaped (gguf present, or config.json present — even
/// if unsupported), so callers know not to descend into it looking for an
/// org layout.
fn tryAddModel(
    io: std.Io,
    allocator: std.mem.Allocator,
    parent: std.Io.Dir,
    name: []const u8,
    id_prefix: []const u8,
    model_dir: []const u8,
    found: *std.ArrayList(DiscoveredModel),
) !bool {
    var sub = parent.openDir(io, name, .{ .iterate = true }) catch return false;
    defer sub.close(io);

    var bytes: u64 = 0;
    var bytes_ok = false;

    // GGUF first (issue #59) — mirrors `--model` routing, where isGgufPath
    // is checked BEFORE any config.json parse ("GGUF files bypass the MLX
    // dispatch entirely"). Pulled GGUF repos usually ship no config.json at
    // all, and the ones that do (unsloth) ship the ORIGINAL model's — an
    // MLX classification would cold-load into a missing-safetensors failure.
    const model_type: []const u8 = blk: {
        const scan = scanLlmGguf(io, allocator, &sub) catch GgufScan{};
        if (scan.pick) |p| {
            allocator.free(p);
            bytes = scan.pick_bytes;
            bytes_ok = true;
            break :blk try allocator.dupe(u8, "gguf");
        }

        const cfg_stat = sub.statFile(io, "config.json", .{}) catch return false;
        if (cfg_stat.kind != .file) return false;

        // Filter by supported model_type AND quantization scheme. Catches:
        //   - partially-downloaded checkpoints (missing/garbage config)
        //   - unsupported arches (e.g. deepseek_v4, MLA + indexer)
        //   - unsupported quants (modes outside supported_quant_modes)
        // before they reach the tokenizer/weight loaders.
        break :blk switch (peekConfig(io, allocator, parent, name)) {
            .missing_or_unparseable => {
                log.info("[discovery] skip {s}: config.json missing or unparseable", .{name});
                return true;
            },
            .unsupported_arch => |mt| {
                defer allocator.free(mt);
                log.info("[discovery] skip {s}: unsupported model_type '{s}'", .{ name, mt });
                return true;
            },
            .unsupported_quant => |mode| {
                defer allocator.free(mode);
                log.info("[discovery] skip {s}: unsupported quantization mode '{s}' (supported: affine, nvfp4, mxfp4, mxfp8)", .{ name, mode });
                return true;
            },
            .supported => |mt| mt, // ownership moves to the DiscoveredModel
        };
    };
    errdefer if (model_type.len > 0) allocator.free(model_type);

    // Compute weight bytes (sum of *.safetensors sizes) — best-effort.
    // GGUF entries already carry the picked file's size instead.
    if (!bytes_ok) {
        var sub_iter_dir = parent.openDir(io, name, .{ .iterate = true }) catch null;
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
    }

    const id = if (id_prefix.len > 0)
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ id_prefix, name })
    else
        try allocator.dupe(u8, name);
    errdefer allocator.free(id);
    const path = if (id_prefix.len > 0)
        try std.fmt.allocPrint(allocator, "{s}/{s}/{s}", .{ trimTrailingSlash(model_dir), id_prefix, name })
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ trimTrailingSlash(model_dir), name });
    try found.append(allocator, .{
        .id = id,
        .path = path,
        .bytes_on_disk = if (bytes_ok) bytes else null,
        .model_type = model_type,
    });
    return true;
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

    // GGUF first — same precedence as tryAddModel and `--model` routing.
    gguf: {
        var sub = dir.openDir(io, base, .{ .iterate = true }) catch break :gguf;
        defer sub.close(io);
        const scan = scanLlmGguf(io, allocator, &sub) catch break :gguf;
        if (scan.pick) |p| {
            allocator.free(p);
            return .{
                .model_type = try allocator.dupe(u8, "gguf"),
                .bytes_on_disk = scan.pick_bytes,
            };
        }
    }

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

/// Metadata an `unloaded` stub can advertise via `/v1/models` without faulting
/// in weights — all sourced from `config.json` (+ chat-template presence). Lets
/// clients see context window, dims, MoE-ness, and capabilities (tools/vision)
/// before a cold load. `found=false` means config.json couldn't be read/parsed.
pub const StubMeta = struct {
    found: bool = false,
    vocab_size: u32 = 0,
    hidden_size: u32 = 0,
    num_hidden_layers: u32 = 0,
    max_position_embeddings: u32 = 0,
    quant_bits: u32 = 0,
    is_moe: bool = false,
    has_vision: bool = false,
    has_chat: bool = false,
};

fn jsonU32(obj: std.json.ObjectMap, key: []const u8) u32 {
    if (obj.get(key)) |v| {
        if (v == .integer and v.integer > 0) return @intCast(v.integer);
    }
    return 0;
}

/// Pure: extract `StubMeta` from raw config.json bytes. `has_chat_template` is
/// supplied by the caller (the template lives outside config.json), and
/// determines chat/tool capabilities — gated off for encoder-only (`bert`)
/// archs. Mirrors the loaded-path capability rules in `server.renderModelEntry`
/// (chat-template presence ⇒ chat/tool_use/streaming/json_schema). Returns
/// `.found=false` on any parse failure.
pub fn parseStubMeta(allocator: std.mem.Allocator, config_json: []const u8, has_chat_template: bool) StubMeta {
    var meta: StubMeta = .{};
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, config_json, .{}) catch return meta;
    defer parsed.deinit();
    if (parsed.value != .object) return meta;
    const root = parsed.value.object;
    meta.found = true;
    meta.vocab_size = jsonU32(root, "vocab_size");
    meta.hidden_size = jsonU32(root, "hidden_size");
    meta.num_hidden_layers = jsonU32(root, "num_hidden_layers");
    meta.max_position_embeddings = jsonU32(root, "max_position_embeddings");
    if (root.get("quantization")) |q| {
        if (q == .object) meta.quant_bits = jsonU32(q.object, "bits");
    }
    meta.is_moe = jsonU32(root, "num_experts") > 0 or
        jsonU32(root, "num_local_experts") > 0 or
        jsonU32(root, "n_routed_experts") > 0;
    const mt: []const u8 = if (root.get("model_type")) |v|
        (if (v == .string) v.string else "")
    else
        "";
    // Vision: a `vision_config` block on a non-`_text` arch (the `_text` guard
    // skips text-only quantized checkpoints with a vestigial block).
    meta.has_vision = root.get("vision_config") != null and !std.mem.endsWith(u8, mt, "_text");
    const is_encoder = std.mem.eql(u8, mt, "bert");
    meta.has_chat = has_chat_template and !is_encoder;
    return meta;
}

/// Read `StubMeta` for the model directory at `abs_path` (config.json + a
/// chat-template-presence check). Best-effort: any I/O / parse failure yields
/// `.found=false`. Called per unloaded entry from `/v1/models` — cheap (small
/// JSON files) and that endpoint isn't hot.
pub fn readStubMeta(io: std.Io, allocator: std.mem.Allocator, abs_path: []const u8) StubMeta {
    const trimmed = trimTrailingSlash(abs_path);
    const base = std.fs.path.basename(trimmed);
    const parent = std.fs.path.dirname(trimmed) orelse return .{};
    if (base.len == 0 or parent.len == 0) return .{};
    var parent_dir = std.Io.Dir.openDirAbsolute(io, parent, .{}) catch return .{};
    defer parent_dir.close(io);
    var dir = parent_dir.openDir(io, base, .{}) catch return .{};
    defer dir.close(io);

    var file = dir.openFile(io, "config.json", .{}) catch return .{};
    defer file.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = file.reader(io, &rbuf);
    const bytes = rs.interface.allocRemaining(allocator, .limited(4 * 1024 * 1024)) catch return .{};
    defer allocator.free(bytes);

    return parseStubMeta(allocator, bytes, hasChatTemplate(io, allocator, dir));
}

/// True if the model dir ships a chat template — a `chat_template.jinja` file,
/// or a `tokenizer_config.json` that carries a `chat_template` key. Cheap proxy
/// for "this is an instruct/chat model" used to gate chat/tool capabilities on
/// unloaded stubs.
fn hasChatTemplate(io: std.Io, allocator: std.mem.Allocator, dir: std.Io.Dir) bool {
    if (dir.statFile(io, "chat_template.jinja", .{})) |st| {
        if (st.kind == .file) return true;
    } else |_| {}
    var f = dir.openFile(io, "tokenizer_config.json", .{}) catch return false;
    defer f.close(io);
    var rbuf: [4096]u8 = undefined;
    var rs = f.reader(io, &rbuf);
    const bytes = rs.interface.allocRemaining(allocator, .limited(8 * 1024 * 1024)) catch return false;
    defer allocator.free(bytes);
    return std.mem.indexOf(u8, bytes, "\"chat_template\"") != null;
}

fn lessThanById(_: void, a: DiscoveredModel, b: DiscoveredModel) bool {
    return std.mem.lessThan(u8, a.id, b.id);
}

// ── Tests ──

const testing = std.testing;

test "discoverModels finds flat and org/repo model dirs" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "flat-model");
    try tmp.dir.writeFile(io, .{ .sub_path = "flat-model/config.json", .data = "{\"model_type\":\"gemma3\"}" });
    // HF-style org/repo layout — the DownloadManager / `mlx-serve pull`
    // convention. Must be discovered with id "org/name".
    try tmp.dir.createDirPath(io, "mlx-community/nested-model");
    try tmp.dir.writeFile(io, .{ .sub_path = "mlx-community/nested-model/config.json", .data = "{\"model_type\":\"qwen3\"}" });
    // Junk that must not surface: an org dir with a non-model child, and a
    // dot-dir.
    try tmp.dir.createDirPath(io, "empty-org/not-a-model");
    try tmp.dir.createDirPath(io, ".hidden/whatever");

    var result = try discoverModelsInDir(io, allocator, tmp.dir, "/models-root");
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 2), result.models.len);
    try std.testing.expectEqualStrings("flat-model", result.models[0].id);
    try std.testing.expectEqualStrings("/models-root/flat-model", result.models[0].path);
    try std.testing.expectEqualStrings("gemma3", result.models[0].model_type);
    try std.testing.expectEqualStrings("mlx-community/nested-model", result.models[1].id);
    try std.testing.expectEqualStrings("/models-root/mlx-community/nested-model", result.models[1].path);
    try std.testing.expectEqualStrings("qwen3", result.models[1].model_type);
}

test "discoverModels finds GGUF dirs without config.json (issue #59)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // Pulled GGUF repo layout: `<org>/<repo>/<quant>.gguf`, NO config.json
    // (bartowski / unsloth-style multi-quant repos ship only .gguf files).
    // `mlx-serve list` already counts these as models; discovery must agree.
    try tmp.dir.createDirPath(io, "bartowski/some-model-GGUF");
    try tmp.dir.writeFile(io, .{ .sub_path = "bartowski/some-model-GGUF/model-IQ2_M.gguf", .data = "0123" });
    try tmp.dir.writeFile(io, .{ .sub_path = "bartowski/some-model-GGUF/model-Q4_K_M.gguf", .data = "01234567" });
    // Flat single-file layout.
    try tmp.dir.createDirPath(io, "flat-gguf");
    try tmp.dir.writeFile(io, .{ .sub_path = "flat-gguf/tiny.gguf", .data = "x" });
    // mmproj sidecar ONLY → not an LLM dir, must not be discovered.
    try tmp.dir.createDirPath(io, "sidecar-only");
    try tmp.dir.writeFile(io, .{ .sub_path = "sidecar-only/mmproj-foo.gguf", .data = "x" });
    // Interrupted pull → .partial is not a .gguf, must not be discovered.
    try tmp.dir.createDirPath(io, "partial");
    try tmp.dir.writeFile(io, .{ .sub_path = "partial/model.gguf.partial", .data = "x" });

    var result = try discoverModelsInDir(io, allocator, tmp.dir, "/root");
    defer result.deinit();

    try testing.expectEqual(@as(usize, 2), result.models.len);
    try testing.expectEqualStrings("bartowski/some-model-GGUF", result.models[0].id);
    try testing.expectEqualStrings("gguf", result.models[0].model_type);
    // bytes = the file the loader will pick (alphabetically-smallest LLM
    // .gguf — resolveGgufFile's rule), NOT the sum of every quant in the dir.
    try testing.expectEqual(@as(?u64, 4), result.models[0].bytes_on_disk);
    try testing.expectEqualStrings("flat-gguf", result.models[1].id);
    try testing.expectEqualStrings("gguf", result.models[1].model_type);
}

test "discoverModels: a .gguf beside config.json wins (mirrors --model routing)" {
    // `--model <dir>` checks isGgufPath BEFORE parsing config.json, so a dir
    // holding both routes to the embedded engine. Discovery must classify it
    // identically — some GGUF repos (unsloth) ship the original config.json
    // next to the quants, and an MLX classification would cold-load into a
    // missing-safetensors failure.
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "both");
    try tmp.dir.writeFile(io, .{ .sub_path = "both/config.json", .data = "{\"model_type\":\"qwen3\"}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "both/model-Q4_K_M.gguf", .data = "01234567" });

    var result = try discoverModelsInDir(io, allocator, tmp.dir, "/root");
    defer result.deinit();

    try testing.expectEqual(@as(usize, 1), result.models.len);
    try testing.expectEqualStrings("gguf", result.models[0].model_type);
}

test "probeModelDir accepts a GGUF dir (register-by-path / /api/pull)" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "some-model-GGUF");
    try tmp.dir.writeFile(io, .{ .sub_path = "some-model-GGUF/model-Q4_K_M.gguf", .data = "01234567" });

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const abs = try std.fmt.allocPrint(allocator, "{s}/some-model-GGUF", .{path_buf[0..root_len]});
    defer allocator.free(abs);

    const probe = try probeModelDir(io, allocator, abs);
    defer allocator.free(probe.model_type);
    try testing.expectEqualStrings("gguf", probe.model_type);
    try testing.expectEqual(@as(?u64, 8), probe.bytes_on_disk);
}

test "resolveGgufFile: deterministic pick, mmproj filtering, precise errors" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "multi");
    try tmp.dir.writeFile(io, .{ .sub_path = "multi/model-Q8_0.gguf", .data = "x" });
    try tmp.dir.writeFile(io, .{ .sub_path = "multi/model-Q4_K_M.gguf", .data = "x" });
    try tmp.dir.writeFile(io, .{ .sub_path = "multi/mmproj-model.gguf", .data = "x" });
    try tmp.dir.createDirPath(io, "sidecar-only");
    try tmp.dir.writeFile(io, .{ .sub_path = "sidecar-only/mmproj-foo.gguf", .data = "x" });
    try tmp.dir.createDirPath(io, "empty");

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..root_len];

    // Directory: alphabetically-smallest non-mmproj .gguf wins.
    const multi = try std.fmt.allocPrint(allocator, "{s}/multi", .{root});
    defer allocator.free(multi);
    const picked = try resolveGgufFile(io, allocator, multi);
    defer allocator.free(picked);
    try testing.expect(std.mem.endsWith(u8, picked, "/multi/model-Q4_K_M.gguf"));
    try testing.expect(isGgufModelPath(io, multi));

    // Direct file path: dup'd through; mmproj file rejected precisely.
    const direct = try std.fmt.allocPrint(allocator, "{s}/multi/model-Q8_0.gguf", .{root});
    defer allocator.free(direct);
    const direct_res = try resolveGgufFile(io, allocator, direct);
    defer allocator.free(direct_res);
    try testing.expectEqualStrings(direct, direct_res);
    const mmproj_file = try std.fmt.allocPrint(allocator, "{s}/multi/mmproj-model.gguf", .{root});
    defer allocator.free(mmproj_file);
    try testing.expectError(error.OnlyMmprojGgufFile, resolveGgufFile(io, allocator, mmproj_file));

    // Sidecar-only dir vs genuinely empty dir: distinct errors, and neither
    // counts as a GGUF model path.
    const sidecar = try std.fmt.allocPrint(allocator, "{s}/sidecar-only", .{root});
    defer allocator.free(sidecar);
    try testing.expectError(error.OnlyMmprojGgufFile, resolveGgufFile(io, allocator, sidecar));
    try testing.expect(!isGgufModelPath(io, sidecar));
    const empty = try std.fmt.allocPrint(allocator, "{s}/empty", .{root});
    defer allocator.free(empty);
    try testing.expectError(error.NoGgufFile, resolveGgufFile(io, allocator, empty));
    try testing.expect(!isGgufModelPath(io, empty));

    // Empty / relative paths: never a GGUF path (the openDirAbsolute
    // ReleaseFast-UB guard).
    try testing.expect(!isGgufModelPath(io, ""));
    try testing.expect(!isGgufModelPath(io, "relative/dir"));
}

test "modelKindFromType labels every family (list TYPE column + run preflight)" {
    // Chat: MLX archs, the synthetic gguf type, and DiffusionGemma (which
    // was missing from the discovery allowlist despite being servable).
    try testing.expectEqual(ModelKind.chat, modelKindFromType("gemma4"));
    try testing.expectEqual(ModelKind.chat, modelKindFromType("qwen3_5_moe"));
    try testing.expectEqual(ModelKind.chat, modelKindFromType("gguf"));
    try testing.expectEqual(ModelKind.chat, modelKindFromType("diffusion_gemma"));
    try testing.expect(isSupportedModelType("diffusion_gemma"));
    // Media modalities.
    try testing.expectEqual(ModelKind.image, modelKindFromType("flux2-klein-4b"));
    try testing.expectEqual(ModelKind.image, modelKindFromType("krea2_turbo"));
    try testing.expectEqual(ModelKind.audio, modelKindFromType("qwen3_tts"));
    try testing.expectEqual(ModelKind.audio, modelKindFromType("acestep"));
    try testing.expectEqual(ModelKind.video, modelKindFromType("AudioVideo"));
    try testing.expectEqual(ModelKind.mesh, modelKindFromType("hunyuan3d_2_1"));
    // Encoders, drafter sidecars, and genuinely unsupported archs.
    try testing.expectEqual(ModelKind.embed, modelKindFromType("bert"));
    try testing.expectEqual(ModelKind.drafter, modelKindFromType("gemma4_assistant"));
    try testing.expectEqual(ModelKind.drafter, modelKindFromType("gemma4_unified_assistant"));
    try testing.expectEqual(ModelKind.unsupported, modelKindFromType("vit"));
    // Labels stay column-friendly.
    try testing.expectEqualStrings("3d", ModelKind.mesh.label());
    try testing.expectEqualStrings("chat", ModelKind.chat.label());
}

test "classifyModelPath: gguf/media/drafter dirs classify; junk is null" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    try tmp.dir.createDirPath(io, "g");
    try tmp.dir.writeFile(io, .{ .sub_path = "g/model-Q4_K_M.gguf", .data = "x" });
    try tmp.dir.createDirPath(io, "img");
    try tmp.dir.writeFile(io, .{ .sub_path = "img/config.json", .data = "{\"model_type\":\"flux2-klein-4b\"}" });
    try tmp.dir.createDirPath(io, "drafter");
    try tmp.dir.writeFile(io, .{ .sub_path = "drafter/config.json", .data = "{\"model_type\":\"gemma4_assistant\"}" });
    try tmp.dir.createDirPath(io, "junk");

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &path_buf);
    const root = path_buf[0..root_len];

    const cases = .{
        .{ "g", ModelKind.chat },
        .{ "img", ModelKind.image },
        .{ "drafter", ModelKind.drafter },
    };
    inline for (cases) |c| {
        const p = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, c[0] });
        defer allocator.free(p);
        try testing.expectEqual(c[1], classifyModelPath(io, allocator, p).?);
    }
    const junk = try std.fmt.allocPrint(allocator, "{s}/junk", .{root});
    defer allocator.free(junk);
    try testing.expect(classifyModelPath(io, allocator, junk) == null);
    try testing.expect(classifyModelPath(io, allocator, "") == null);
    try testing.expect(classifyModelPath(io, allocator, "rel/path") == null);
}

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

test "isSupportedModelType accepts native media archs (image/audio/video)" {
    // Unified media-gen: FLUX (flux2*), Qwen3-TTS, LTX-Video (AudioVideo) load
    // through the registry now, so discovery + by-path must accept them.
    try testing.expect(isSupportedModelType("flux2-klein-4b"));
    try testing.expect(isSupportedModelType("flux2"));
    try testing.expect(isSupportedModelType("qwen3_tts"));
    try testing.expect(isSupportedModelType("AudioVideo"));
    try testing.expect(isMediaModelType("flux2-klein-9b"));
    try testing.expect(isMediaModelType("krea2_turbo"));
    try testing.expect(isSupportedModelType("krea2_turbo"));
    try testing.expect(isMediaModelType("hunyuan3d_2_1"));
    try testing.expect(isSupportedModelType("hunyuan3d_2_1"));
    try testing.expect(isMediaModelType("acestep"));
    try testing.expect(isSupportedModelType("acestep"));
    try testing.expect(!isMediaModelType("gemma4"));
}

test "isSupportedModelType accepts gemma3_text (text-only Gemma3ForCausalLM)" {
    // Regression for "[discovery] skip ...: unsupported model_type
    // 'gemma3_text'": text-only Gemma 3 abliterated checkpoints
    // (mlx-community/gemma-3-12b-it-qat-abliterated-lm-4bit) ship a flat
    // top-level model_type "gemma3_text" and must be discoverable, not skipped.
    try testing.expect(isSupportedModelType("gemma3_text"));
    try testing.expect(isSupportedModelType("gemma3"));
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

test "parseStubMeta extracts dims/ctx/quant/MoE + chat/vision capabilities" {
    const a = testing.allocator;
    // MoE chat model (Qwen3-Coder-30B-A3B shape), chat template present.
    {
        const json =
            \\{"model_type":"qwen3_moe","vocab_size":151936,"hidden_size":2048,
            \\"num_hidden_layers":48,"max_position_embeddings":262144,
            \\"num_experts":128,"num_experts_per_tok":8,"quantization":{"bits":8,"group_size":64}}
        ;
        const m = parseStubMeta(a, json, true);
        try testing.expect(m.found);
        try testing.expectEqual(@as(u32, 151936), m.vocab_size);
        try testing.expectEqual(@as(u32, 2048), m.hidden_size);
        try testing.expectEqual(@as(u32, 48), m.num_hidden_layers);
        try testing.expectEqual(@as(u32, 262144), m.max_position_embeddings);
        try testing.expectEqual(@as(u32, 8), m.quant_bits);
        try testing.expect(m.is_moe);
        try testing.expect(m.has_chat); // template present, not encoder
        try testing.expect(!m.has_vision);
    }
    // Dense model, no template → no chat caps; no quant block → 0 bits.
    {
        const json =
            \\{"model_type":"qwen2","hidden_size":5120,"num_attention_heads":40,
            \\"max_position_embeddings":32768}
        ;
        const m = parseStubMeta(a, json, false);
        try testing.expect(m.found);
        try testing.expect(!m.is_moe);
        try testing.expect(!m.has_chat);
        try testing.expectEqual(@as(u32, 0), m.quant_bits);
        try testing.expectEqual(@as(u32, 32768), m.max_position_embeddings);
    }
    // Vision: vision_config on a non-_text arch → has_vision.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"gemma4\",\"vision_config\":{\"hidden_size\":1152}}", true);
        try testing.expect(m.has_vision);
    }
    // …but a `_text` arch with a vestigial vision_config must NOT report vision.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"qwen3_5_moe_text\",\"vision_config\":{}}", true);
        try testing.expect(!m.has_vision);
    }
    // Encoder (bert): chat/tool caps suppressed even with a template present.
    {
        const m = parseStubMeta(a, "{\"model_type\":\"bert\",\"hidden_size\":384}", true);
        try testing.expect(!m.has_chat);
    }
    // Malformed → found=false.
    {
        const m = parseStubMeta(a, "not json", true);
        try testing.expect(!m.found);
    }
}
