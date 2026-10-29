//! Which `.gguf` in a folder is the language model: companion files (mmproj, tokenizer, MTP heads) and embedding
//! GGUFs never are, and of the rest the alphabetically smallest is, so the pick is deterministic.

const std = @import("std");
const gguf_meta = @import("engine_pick.zig");

/// Result of scanning a directory for LLM `.gguf` files (mmproj sidecars
/// excluded). `pick` is the alphabetically-smallest LLM gguf basename — the
/// same deterministic file `resolveGgufFile` loads — so callers can report
/// the bytes that will actually become resident, not the sum of every quant
/// in a multi-quant repo.
pub const GgufScan = struct {
    pick: ?[]u8 = null,
    pick_bytes: u64 = 0,
    saw_mmproj: bool = false,
    /// An embedding GGUF (`gguf_meta.declaresPooling`) was passed over.
    saw_embedding: bool = false,
};

/// Scan an iterable dir for LLM `.gguf` entries. Symlinked files count
/// (statFile follows links) — users symlink multi-GB weights rather than
/// copy them. Caller frees `pick`.
pub fn scanLlmGguf(io: std.Io, allocator: std.mem.Allocator, dir: *std.Io.Dir) !GgufScan {
    var scan: GgufScan = .{};
    errdefer if (scan.pick) |p| allocator.free(p);
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".gguf")) continue;
        // Sidecars are never candidates. `saw_mmproj` stays mmproj-specific —
        // it only drives the "this folder holds ONLY a CLIP encoder" error text.
        if (isGgufSidecarBasename(entry.name)) {
            if (isMmprojGgufBasename(entry.name)) scan.saw_mmproj = true;
            continue;
        }
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind != .file) continue;
        if (gguf_meta.fileDeclaresPooling(io, dir.*, entry.name)) {
            scan.saw_embedding = true;
            continue;
        }
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
        if (isGgufSidecarBasename(entry.name)) continue;
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
///   error.EmbeddingGgufFile  — only embedding GGUFs (or the path is one)
///
/// Does NOT log on error — the caller decides whether the error is "fatal
/// user load" (then call `logResolveGgufError`) or "silent probe".
pub fn resolveGgufFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.mem.endsWith(u8, path, ".gguf")) {
        if (isMmprojGgufBasename(std.fs.path.basename(path))) {
            return error.OnlyMmprojGgufFile;
        }
        if (gguf_meta.fileDeclaresPooling(io, std.Io.Dir.cwd(), path)) return error.EmbeddingGgufFile;
        return allocator.dupe(u8, path);
    }
    var dir = try std.Io.Dir.openDirAbsolute(io, path, .{ .iterate = true });
    defer dir.close(io);
    const scan = try scanLlmGguf(io, allocator, &dir);
    if (scan.pick) |p| {
        defer allocator.free(p);
        return std.fmt.allocPrint(allocator, "{s}/{s}", .{ trimTrailingSlash(path), p });
    }
    if (scan.saw_embedding) return error.EmbeddingGgufFile;
    if (scan.saw_mmproj) return error.OnlyMmprojGgufFile;
    return error.NoGgufFile;
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

/// True if `basename` is a non-LLM `.gguf` COMPANION file rather than a
/// language-model quant. Two kinds ship today:
///
///   - `mmproj-*.gguf`  — multimodal-projection / CLIP encoder (see above).
///   - `*tokenizer*.gguf` — audio/speech tokenizer shipped beside a TTS model
///     (real: `qwen3-tts-tokenizer-f16.gguf` next to `qwen3-tts-0.6b-f16.gguf`).
///
/// `scanLlmGguf` picks the alphabetically-smallest candidate, so a folder whose
/// tokenizer happens to sort first would otherwise be loaded AS the LLM. Mirrors
/// the Swift `DownloadManager.isGgufSidecar` — the macOS app lists every quant
/// in a folder as a separately selectable model, so the two must agree on which
/// files are models or the app offers one the server can't load.
/// True for an MTP draft-head GGUF — the llama.cpp / ds4 speculative-decode
/// sidecar (e.g. `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`). NOT loadable as a
/// chat model: it's a dependency of the main quant, loaded beside it by the ds4
/// engine for a faster decode. Matched as a delimited `-MTP-` token so a real
/// model that merely contains the letters "mtp" isn't caught.
pub fn isMtpGgufBasename(basename: []const u8) bool {
    if (!std.mem.endsWith(u8, basename, ".gguf")) return false;
    // Delimited tokens only, so a chat quant whose scheme name merely
    // contains the letters can't match. Covers the legacy MTP draft head
    // (`…-MTP-….gguf`) AND the 0731 DSpark stage bundle
    // (`DeepSeek-V4-Flash-DSpark-support.gguf`) — ds4 loads either via the
    // same --mtp slot and classifies by tensors. Swift mirror:
    // DownloadManager.isGgufSidecar — keep in sync.
    return std.ascii.startsWithIgnoreCase(basename, "mtp-") or // llama.cpp convert --mtp
        asciiContainsIgnoreCase(basename, "-mtp-") or asciiContainsIgnoreCase(basename, "-mtp.") or
        asciiContainsIgnoreCase(basename, "-dspark-") or asciiContainsIgnoreCase(basename, "-dspark.");
}

pub fn isGgufSidecarBasename(basename: []const u8) bool {
    if (!std.mem.endsWith(u8, basename, ".gguf")) return false;
    if (isMmprojGgufBasename(basename)) return true;
    if (asciiContainsIgnoreCase(basename, "tokenizer")) return true;
    return isMtpGgufBasename(basename);
}

pub fn asciiContainsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or haystack.len < needle.len) return false;
    var i: usize = 0;
    outer: while (i + needle.len <= haystack.len) : (i += 1) {
        for (haystack[i..][0..needle.len], needle) |c, n| {
            if (std.ascii.toLower(c) != n) continue :outer;
        }
        return true;
    }
    return false;
}

pub fn trimTrailingSlash(s: []const u8) []const u8 {
    var p = s;
    while (p.len > 0 and p[p.len - 1] == '/') p = p[0 .. p.len - 1];
    return p;
}
