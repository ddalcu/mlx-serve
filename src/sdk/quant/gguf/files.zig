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
    // A split set's first shard is its metadata: the set is billed whole.
    if (scan.pick) |p| if (splitSet(p)) |set| {
        scan.pick_bytes = 0;
        var again = dir.iterate();
        while (again.next(io) catch null) |entry| {
            if (!std.mem.startsWith(u8, entry.name, set.stem) or !std.mem.endsWith(u8, entry.name, set.tail)) continue;
            if (entry.name.len != p.len) continue;
            const st = dir.statFile(io, entry.name, .{}) catch continue;
            scan.pick_bytes += @intCast(st.size);
        }
    };
    return scan;
}

/// `<stem>-00001-of-NNNNN.gguf` as (`<stem>-`, `-of-NNNNN.gguf`), null for any other name.
pub fn splitSet(basename: []const u8) ?struct { stem: []const u8, tail: []const u8 } {
    const tail_len = "-of-00000.gguf".len;
    const first = "-00001";
    if (basename.len < first.len + tail_len) return null;
    const tail = basename[basename.len - tail_len ..];
    if (!std.mem.startsWith(u8, tail, "-of-") or !std.mem.endsWith(u8, tail, ".gguf")) return null;
    for (tail[4..9]) |c| if (!std.ascii.isDigit(c)) return null;
    const head = basename[0 .. basename.len - tail_len];
    if (!std.mem.endsWith(u8, head, first) or std.mem.eql(u8, tail[4..9], "00001")) return null;
    return .{ .stem = head[0 .. head.len - first.len + 1], .tail = tail };
}

test "a split set is picked by its first shard and billed whole" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for ([_][]const u8{ "M-00001-of-00003.gguf", "M-00002-of-00003.gguf", "M-00003-of-00003.gguf" }, [_]usize{ 10, 200, 3000 }) |name, n| {
        const bytes = try std.testing.allocator.alloc(u8, n);
        defer std.testing.allocator.free(bytes);
        @memset(bytes, 0);
        try tmp.dir.writeFile(io, .{ .sub_path = name, .data = bytes });
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "M-other.gguf.txt", .data = "x" });
    const scan = try scanLlmGguf(io, std.testing.allocator, &tmp.dir);
    defer std.testing.allocator.free(scan.pick.?);
    try std.testing.expectEqualStrings("M-00001-of-00003.gguf", scan.pick.?);
    try std.testing.expectEqual(@as(u64, 3210), scan.pick_bytes);
    try std.testing.expect(splitSet("M-00002-of-00003.gguf") == null);
    try std.testing.expect(splitSet("M-00001-of-00001.gguf") == null);
    try std.testing.expect(splitSet("Q4_K_M.gguf") == null);
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
///   - `mmproj-*.gguf` / `*-Vision.gguf` — a vision tower (`isVisionGgufBasename`).
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
        std.ascii.startsWithIgnoreCase(basename, "dspark-") or // Unsloth's DeepSeek-V4.1 stage bundle
        asciiContainsIgnoreCase(basename, "-mtp-") or asciiContainsIgnoreCase(basename, "-mtp.") or
        asciiContainsIgnoreCase(basename, "-dspark-") or asciiContainsIgnoreCase(basename, "-dspark.");
}

/// Full path to an MTP draft-head GGUF for a model file, or null: the
/// model's own folder first, then an `MTP/` folder inside it or beside it
/// (unsloth keeps each quant in its own folder and the heads in `MTP/`).
/// Caller owns the returned slice.
pub fn findMtpSidecar(io: std.Io, allocator: std.mem.Allocator, model_file_path: []const u8) ?[]u8 {
    const dir_path = std.fs.path.dirname(model_file_path) orelse return null;
    const skip = std.fs.path.basename(model_file_path);
    if (firstMtpGgufIn(io, allocator, dir_path, skip)) |p| return p;
    for ([_]?[]const u8{ dir_path, std.fs.path.dirname(dir_path) }) |base| {
        const mtp_dir = std.fs.path.join(allocator, &.{ base orelse continue, "MTP" }) catch return null;
        defer allocator.free(mtp_dir);
        if (firstMtpGgufIn(io, allocator, mtp_dir, skip)) |p| return p;
    }
    return null;
}

/// The first MTP draft-head GGUF in `dir_path` by name, other than `skip`.
pub fn firstMtpGgufIn(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8, skip: []const u8) ?[]u8 {
    return firstGgufIn(io, allocator, dir_path, skip, isMtpGgufBasename);
}

/// True for a vision tower GGUF shipped beside a language model: an `mmproj-*` file, or ds4's `*-Vision.gguf`
/// (a suffix only: `Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf` is a model).
pub fn isVisionGgufBasename(basename: []const u8) bool {
    const tail = "-vision.gguf";
    return isMmprojGgufBasename(basename) or
        (basename.len >= tail.len and std.ascii.eqlIgnoreCase(basename[basename.len - tail.len ..], tail));
}

/// Full path to the vision tower GGUF beside a model file, else one folder up (a repo with one folder per quant ships
/// its mmproj beside them), or null. Caller owns the returned slice.
pub fn findVisionSidecar(io: std.Io, allocator: std.mem.Allocator, model_file_path: []const u8) ?[]u8 {
    const dir_path = std.fs.path.dirname(model_file_path) orelse return null;
    const skip = std.fs.path.basename(model_file_path);
    if (firstGgufIn(io, allocator, dir_path, skip, isVisionGgufBasename)) |p| return p;
    return firstGgufIn(io, allocator, std.fs.path.dirname(dir_path) orelse return null, skip, isVisionGgufBasename);
}

/// The first GGUF in `dir_path` by name that `kind` accepts, other than `skip`.
fn firstGgufIn(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8, skip: []const u8, comptime kind: fn ([]const u8) bool) ?[]u8 {
    // openDir asserts (→ ReleaseFast UB) on a non-absolute path.
    if (dir_path.len == 0 or !std.fs.path.isAbsolute(dir_path)) return null;
    var dir = std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var name_buf: [std.fs.max_name_bytes]u8 = undefined;
    var best: ?[]const u8 = null;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (!kind(entry.name) or std.mem.eql(u8, entry.name, skip)) continue;
        if (best != null and !std.mem.lessThan(u8, entry.name, best.?)) continue;
        const st = dir.statFile(io, entry.name, .{}) catch continue;
        if (st.kind != .file or entry.name.len > name_buf.len) continue;
        @memcpy(name_buf[0..entry.name.len], entry.name);
        best = name_buf[0..entry.name.len];
    }
    return std.fs.path.join(allocator, &.{ dir_path, best orelse return null }) catch null;
}

pub fn isGgufSidecarBasename(basename: []const u8) bool {
    if (!std.mem.endsWith(u8, basename, ".gguf")) return false;
    if (isVisionGgufBasename(basename)) return true;
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

test "a vision tower GGUF is a sidecar, never the model, and is found beside it" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    try std.testing.expect(isGgufSidecarBasename("DeepSeek-V4.1-Flash-1017-Vision.gguf") and isVisionGgufBasename("mmproj-gemma-4-E4B-it-BF16.gguf"));
    try std.testing.expect(!isGgufSidecarBasename("Llama-3.2-11B-Vision-Instruct-Q4_K_M.gguf"));
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for ([_][]const u8{ "A-1017-Vision.gguf", "B-IQ2.gguf", "B-DSpark-support.gguf" }) |name| try tmp.dir.writeFile(io, .{ .sub_path = name, .data = "x" });
    const scan = try scanLlmGguf(io, a, &tmp.dir);
    defer a.free(scan.pick.?);
    try std.testing.expectEqualStrings("B-IQ2.gguf", scan.pick.?);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const model = try std.fs.path.join(a, &.{ root, "B-IQ2.gguf" });
    defer a.free(model);
    const tower = findVisionSidecar(io, a, model) orelse return error.TestUnexpectedResult;
    defer a.free(tower);
    try std.testing.expectEqualStrings("A-1017-Vision.gguf", std.fs.path.basename(tower));
    try tmp.dir.deleteFile(io, "A-1017-Vision.gguf");
    try std.testing.expect(findVisionSidecar(io, a, model) == null);
}

test "a quant folder's vision tower is found one level up, as a repo ships its mmproj beside the quant folders" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    try tmp.dir.createDirPath(io, "IQ2_XXS");
    try tmp.dir.writeFile(io, .{ .sub_path = "IQ2_XXS/M-IQ2_XXS-00001-of-00006.gguf", .data = "x" });
    try tmp.dir.writeFile(io, .{ .sub_path = "mmproj-M-BF16.gguf", .data = "x" });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    const model = try std.fs.path.join(a, &.{ root, "IQ2_XXS", "M-IQ2_XXS-00001-of-00006.gguf" });
    defer a.free(model);
    const tower = findVisionSidecar(io, a, model) orelse return error.TestUnexpectedResult;
    defer a.free(tower);
    try std.testing.expectEqualStrings("mmproj-M-BF16.gguf", std.fs.path.basename(tower));
    // One beside the model wins over the folder above.
    try tmp.dir.writeFile(io, .{ .sub_path = "IQ2_XXS/mmproj-own.gguf", .data = "x" });
    const own = findVisionSidecar(io, a, model) orelse return error.TestUnexpectedResult;
    defer a.free(own);
    try std.testing.expectEqualStrings("mmproj-own.gguf", std.fs.path.basename(own));
}
