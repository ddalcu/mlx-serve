//! CLI subcommands — `mlx-serve run|pull|list <model>` — Ollama-grade
//! ergonomics for the terminal.
//!
//!   mlx-serve run gemma4        # download if missing, serve, drop into a REPL
//!   mlx-serve pull qwen3.6      # download only
//!   mlx-serve list              # what's on disk
//!
//! Short names resolve through a curated alias table (mirroring the MLX
//! Core app catalog in ChatModels.swift); anything containing '/' is
//! treated as a HuggingFace repo id directly ("org/repo", with optional
//! "hf.co/" prefix and ":tag" suffix). Downloads land in
//! `~/.mlx-serve/models/<org>/<repo>` — the single source of truth shared
//! with the app's DownloadManager and the server's media-dep resolution.
//!
//! Transport is the system `curl` (always present on macOS): rock-solid
//! TLS/redirect/HTTP2 handling, `-C -` resume, `--create-dirs`, and a free
//! progress bar on the CLI path. Pure helpers (alias resolution, tree-JSON
//! parsing, file filtering, REPL body/line codecs) are hermetically tested
//! here; only the thin curl/spawn wrappers need a live network.

const std = @import("std");
const ollama = @import("ollama.zig");
const model_discovery = @import("model_discovery.zig");
const log = @import("log.zig");

// ── Alias table ─────────────────────────────────────────────────────────

pub const Alias = struct {
    /// Short name before the ':', e.g. "gemma4".
    name: []const u8,
    /// Tag after the ':'; empty = selectable only by full name:tag.
    tag: []const u8,
    repo: []const u8,
    /// Picked when the user gives the bare name with no tag.
    is_default: bool = false,
    /// Non-empty: restrict the download to this single .gguf artifact.
    gguf_file: []const u8 = "",
};

/// Mirrors the app catalog (`gemmaModelOptions` in ChatModels.swift).
/// Bare-name defaults pick the 4-bit build that fits the widest range of
/// Macs for that family.
pub const aliases = [_]Alias{
    .{ .name = "gemma4", .tag = "e2b", .repo = "mlx-community/gemma-4-e2b-it-4bit" },
    .{ .name = "gemma4", .tag = "e2b-8bit", .repo = "mlx-community/gemma-4-e2b-it-8bit" },
    .{ .name = "gemma4", .tag = "e4b", .repo = "mlx-community/gemma-4-e4b-it-4bit", .is_default = true },
    .{ .name = "gemma4", .tag = "e4b-8bit", .repo = "mlx-community/gemma-4-e4b-it-8bit" },
    .{ .name = "gemma4", .tag = "12b", .repo = "mlx-community/gemma-4-12b-it-4bit" },
    .{ .name = "gemma4", .tag = "12b-8bit", .repo = "mlx-community/gemma-4-12b-it-8bit" },
    .{ .name = "gemma4", .tag = "26b", .repo = "mlx-community/gemma-4-26b-a4b-it-4bit" },
    .{ .name = "gemma4", .tag = "26b-8bit", .repo = "mlx-community/gemma-4-26b-a4b-it-8bit" },
    .{ .name = "gemma4", .tag = "31b", .repo = "mlx-community/gemma-4-31b-it-4bit" },
    .{ .name = "gemma4", .tag = "31b-8bit", .repo = "mlx-community/gemma-4-31b-it-8bit" },
    .{ .name = "gemma3", .tag = "12b", .repo = "mlx-community/gemma-3-12b-it-4bit", .is_default = true },
    // Qwen 3.6 27B ships an MTP sidecar the server auto-loads for
    // multi-token speculative decode — the best default experience.
    .{ .name = "qwen3.6", .tag = "27b", .repo = "ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve", .is_default = true },
    .{ .name = "qwen3.5", .tag = "0.8b", .repo = "mlx-community/Qwen3.5-0.8B-MLX-4bit", .is_default = true },
    .{ .name = "deepseek-v4", .tag = "flash", .repo = "antirez/deepseek-v4-gguf", .is_default = true, .gguf_file = "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf" },
    .{ .name = "bge-small", .tag = "en", .repo = "mlx-community/bge-small-en-v1.5-8bit", .is_default = true },
};

pub const Resolved = struct {
    repo: []const u8,
    gguf_file: []const u8 = "",
};

/// Short name / repo ref → HF repo id. Accepts:
///   "gemma4" / "gemma4:12b"           (alias table)
///   "org/repo" / "org/repo:tag"       (direct, tag stripped)
///   "hf.co/org/repo", "huggingface.co/org/repo"
/// Returns null for unknown alias-shaped names (no '/').
pub fn resolveShortName(name: []const u8) ?Resolved {
    var n = name;
    for ([_][]const u8{ "hf.co/", "huggingface.co/", "https://huggingface.co/" }) |prefix| {
        if (std.ascii.startsWithIgnoreCase(n, prefix)) {
            n = n[prefix.len..];
            break;
        }
    }
    n = ollama.stripTag(n);
    if (n.len == 0) return null;
    if (std.mem.indexOfScalar(u8, n, '/') != null) {
        // Direct org/repo reference.
        return .{ .repo = n };
    }
    // Alias lookup: "name" or "name:tag" (tag was stripped above — redo the
    // split on the ORIGINAL string so alias tags still work).
    var base = name;
    var tag: []const u8 = "";
    if (std.mem.lastIndexOfScalar(u8, name, ':')) |ci| {
        base = name[0..ci];
        tag = name[ci + 1 ..];
    }
    if (std.mem.eql(u8, tag, "latest")) tag = "";
    for (aliases) |a| {
        if (!std.ascii.eqlIgnoreCase(a.name, base)) continue;
        if (tag.len == 0) {
            if (a.is_default) return .{ .repo = a.repo, .gguf_file = a.gguf_file };
        } else if (std.ascii.eqlIgnoreCase(a.tag, tag)) {
            return .{ .repo = a.repo, .gguf_file = a.gguf_file };
        }
    }
    return null;
}

/// `~/.mlx-serve/models/<org>/<repo>` — the single models root shared with
/// the app's DownloadManager.
pub fn modelDestPath(allocator: std.mem.Allocator, home: []const u8, repo: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/.mlx-serve/models/{s}", .{ home, repo });
}

pub fn modelsRootPath(allocator: std.mem.Allocator, home: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator, "{s}/.mlx-serve/models", .{home});
}

fn homeDir() []const u8 {
    return std.mem.span(std.c.getenv("HOME") orelse return "/tmp");
}

// ── HF tree listing ─────────────────────────────────────────────────────

pub const RepoFile = struct {
    path: []u8,
    size: u64,
};

pub fn freeRepoFiles(allocator: std.mem.Allocator, files: []RepoFile) void {
    for (files) |f| allocator.free(f.path);
    allocator.free(files);
}

/// Parse the HF `/api/models/<repo>/tree/main?recursive=true` JSON array.
/// LFS entries report the real artifact size under `lfs.size`.
pub fn parseTreeJson(allocator: std.mem.Allocator, json: []const u8) ![]RepoFile {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch return error.InvalidTree;
    defer parsed.deinit();
    if (parsed.value != .array) return error.InvalidTree;

    var files = std.ArrayList(RepoFile).empty;
    errdefer {
        for (files.items) |f| allocator.free(f.path);
        files.deinit(allocator);
    }
    for (parsed.value.array.items) |item| {
        if (item != .object) continue;
        const obj = item.object;
        const t = obj.get("type") orelse continue;
        if (t != .string or !std.mem.eql(u8, t.string, "file")) continue;
        const p = obj.get("path") orelse continue;
        if (p != .string) continue;
        var size: u64 = 0;
        if (obj.get("lfs")) |lfs| {
            if (lfs == .object) {
                if (lfs.object.get("size")) |s| {
                    if (s == .integer and s.integer > 0) size = @intCast(s.integer);
                }
            }
        }
        if (size == 0) {
            if (obj.get("size")) |s| {
                if (s == .integer and s.integer > 0) size = @intCast(s.integer);
            }
        }
        try files.append(allocator, .{ .path = try allocator.dupe(u8, p.string), .size = size });
    }
    return files.toOwnedSlice(allocator);
}

/// Chat-default file selection (mirrors the app's `FileSelection.chatDefault`):
/// top-level files + the `mtp/` spec-decode sidecar; repo housekeeping and
/// demo assets are skipped.
pub fn shouldDownload(path: []const u8) bool {
    if (path.len == 0 or path[0] == '.') return false;
    if (std.mem.indexOfScalar(u8, path, '/')) |_| {
        return std.mem.startsWith(u8, path, "mtp/");
    }
    const skip_exact = [_][]const u8{ "README.md", "LICENSE", "LICENSE.txt", "USE_POLICY.md" };
    for (skip_exact) |s| {
        if (std.ascii.eqlIgnoreCase(path, s)) return false;
    }
    const skip_ext = [_][]const u8{ ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".md" };
    for (skip_ext) |ext| {
        if (path.len > ext.len and std.ascii.eqlIgnoreCase(path[path.len - ext.len ..], ext)) return false;
    }
    return true;
}

// ── Pull ────────────────────────────────────────────────────────────────

pub const Reporter = struct {
    impl: *anyopaque,
    /// One human-readable status line per call (no trailing newline).
    reportFn: *const fn (impl: *anyopaque, line: []const u8) void,

    pub fn say(self: Reporter, comptime fmt: []const u8, args: anytype) void {
        var buf: [512]u8 = undefined;
        const line = std.fmt.bufPrint(&buf, fmt, args) catch return;
        self.reportFn(self.impl, line);
    }
};

/// Appends the shared curl argv prefix; returns the owned Authorization
/// header string when HF_TOKEN is set (caller frees).
fn curlBaseArgs(list: *std.ArrayList([]const u8), allocator: std.mem.Allocator) !?[]u8 {
    try list.appendSlice(allocator, &.{ "curl", "-fL", "--retry", "3", "--retry-delay", "2" });
    if (std.c.getenv("HF_TOKEN")) |tok| {
        const header = try std.fmt.allocPrint(allocator, "Authorization: Bearer {s}", .{std.mem.span(tok)});
        try list.appendSlice(allocator, &.{ "-H", header });
        return header;
    }
    return null;
}

/// GET a small HTTPS document (the tree listing) via curl; returns stdout.
fn curlFetch(allocator: std.mem.Allocator, io: std.Io, url: []const u8) ![]u8 {
    var argv = std.ArrayList([]const u8).empty;
    defer argv.deinit(allocator);
    const header_storage = try curlBaseArgs(&argv, allocator);
    defer if (header_storage) |h| allocator.free(h);
    try argv.appendSlice(allocator, &.{ "-s", url });
    const result = std.process.run(allocator, io, .{
        .argv = argv.items,
        .stdout_limit = .limited(64 * 1024 * 1024),
    }) catch return error.FetchFailed;
    defer allocator.free(result.stderr);
    errdefer allocator.free(result.stdout);
    switch (result.term) {
        // Quiet on failure — callers report (the REPL health poll EXPECTS
        // failures while the server boots).
        .exited => |code| if (code != 0) return error.FetchFailed,
        else => return error.FetchFailed,
    }
    return result.stdout;
}

/// Download one file to `<dest_dir>/<file>` via curl (`-C -` resume onto a
/// .partial, atomic rename on success). `show_progress` inherits stderr so
/// the terminal gets curl's progress bar; the server-side pull passes false.
fn curlDownload(allocator: std.mem.Allocator, io: std.Io, url: []const u8, dest_path: []const u8, show_progress: bool) !void {
    const partial = try std.fmt.allocPrint(allocator, "{s}.partial", .{dest_path});
    defer allocator.free(partial);

    var argv = std.ArrayList([]const u8).empty;
    defer argv.deinit(allocator);
    const header_storage = try curlBaseArgs(&argv, allocator);
    defer if (header_storage) |h| allocator.free(h);
    try argv.appendSlice(allocator, &.{ "--create-dirs", "-C", "-", "-o", partial });
    if (show_progress) {
        try argv.append(allocator, "--progress-bar");
    } else {
        try argv.append(allocator, "-sS");
    }
    try argv.append(allocator, url);

    var child = std.process.spawn(io, .{
        .argv = argv.items,
        .stdin = .ignore,
        .stdout = .ignore,
        .stderr = if (show_progress) .inherit else .ignore,
    }) catch return error.DownloadFailed;
    const term = child.wait(io) catch return error.DownloadFailed;
    switch (term) {
        .exited => |code| if (code != 0) return error.DownloadFailed,
        else => return error.DownloadFailed,
    }
    std.Io.Dir.renameAbsolute(partial, dest_path, io) catch return error.DownloadFailed;
}

fn fileSizeAt(io: std.Io, dir_path: []const u8, rel: []const u8) ?u64 {
    if (dir_path.len == 0 or !std.fs.path.isAbsolute(dir_path)) return null;
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return null;
    defer dir.close(io);
    const st = dir.statFile(io, rel, .{}) catch return null;
    if (st.kind != .file) return null;
    return st.size;
}

/// True when the model directory already holds a COMPLETE, loadable
/// checkpoint. "config.json exists" is NOT enough: an interrupted `pull`
/// (Ctrl-C mid-weights) leaves config.json + *.partial, and treating that
/// as present skipped the resume and fed a weightless dir to the loader
/// (live SIGSEGV — see tests/test_partial_download.sh). Complete means: no
/// .partial leftovers anywhere (top level or one subdir deep, e.g.
/// mtp/weights.safetensors.partial), plus config.json AND at least one
/// .safetensors for MLX dirs — or any .gguf, which is self-contained.
pub fn modelPresent(io: std.Io, dir_path: []const u8) bool {
    if (dir_path.len == 0 or !std.fs.path.isAbsolute(dir_path)) return false;
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true }) catch return false;
    defer dir.close(io);
    return modelPresentInDir(io, dir);
}

fn modelPresentInDir(io: std.Io, dir: std.Io.Dir) bool {
    var has_config = false;
    var has_safetensors = false;
    var has_gguf = false;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        switch (entry.kind) {
            .file => {
                if (std.mem.endsWith(u8, entry.name, ".partial")) return false;
                if (std.mem.eql(u8, entry.name, "config.json")) has_config = true;
                if (std.mem.endsWith(u8, entry.name, ".safetensors")) has_safetensors = true;
                if (std.mem.endsWith(u8, entry.name, ".gguf")) has_gguf = true;
            },
            .directory => {
                // One level deep is enough for the pull layouts (mtp/ is the
                // only subdir the chat-default selection downloads into).
                var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
                defer sub.close(io);
                var sit = sub.iterate();
                while (sit.next(io) catch null) |se| {
                    if (se.kind == .file and std.mem.endsWith(u8, se.name, ".partial")) return false;
                }
            },
            else => {},
        }
    }
    if (has_gguf) return true;
    return has_config and has_safetensors;
}

/// Download `resolved.repo` into `dest_dir`. Skips files already complete
/// on disk (size match), resumes partials, reports per-file progress.
pub fn pullRepo(allocator: std.mem.Allocator, io: std.Io, resolved: Resolved, dest_dir: []const u8, reporter: Reporter, show_progress: bool) !void {
    reporter.say("pulling manifest for {s}", .{resolved.repo});
    const tree_url = try std.fmt.allocPrint(allocator, "https://huggingface.co/api/models/{s}/tree/main?recursive=true", .{resolved.repo});
    defer allocator.free(tree_url);
    const tree_json = curlFetch(allocator, io, tree_url) catch {
        reporter.say("error: could not list {s} (check the name, your network, or HF_TOKEN for gated repos)", .{resolved.repo});
        return error.PullFailed;
    };
    defer allocator.free(tree_json);
    const files = parseTreeJson(allocator, tree_json) catch {
        reporter.say("error: unexpected listing for {s}", .{resolved.repo});
        return error.PullFailed;
    };
    defer freeRepoFiles(allocator, files);

    var wanted: usize = 0;
    var total_bytes: u64 = 0;
    for (files) |f| {
        if (!wantedFile(resolved, f.path)) continue;
        wanted += 1;
        total_bytes += f.size;
    }
    if (wanted == 0) {
        reporter.say("error: {s} has no downloadable model files", .{resolved.repo});
        return error.PullFailed;
    }
    reporter.say("{d} files, {d} MB total", .{ wanted, total_bytes / (1024 * 1024) });

    var idx: usize = 0;
    for (files) |f| {
        if (!wantedFile(resolved, f.path)) continue;
        idx += 1;
        const dest_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dest_dir, f.path });
        defer allocator.free(dest_path);
        if (f.size > 0) {
            if (fileSizeAt(io, dest_dir, f.path)) |have| {
                if (have == f.size) {
                    reporter.say("[{d}/{d}] {s} — already complete", .{ idx, wanted, f.path });
                    continue;
                }
            }
        }
        reporter.say("[{d}/{d}] pulling {s} ({d} MB)", .{ idx, wanted, f.path, f.size / (1024 * 1024) });
        const url = try std.fmt.allocPrint(allocator, "https://huggingface.co/{s}/resolve/main/{s}", .{ resolved.repo, f.path });
        defer allocator.free(url);
        curlDownload(allocator, io, url, dest_path, show_progress) catch {
            reporter.say("error: download failed for {s} (partial kept — rerun to resume)", .{f.path});
            return error.PullFailed;
        };
    }
    reporter.say("success: {s} ready", .{resolved.repo});
}

fn wantedFile(resolved: Resolved, path: []const u8) bool {
    if (resolved.gguf_file.len > 0) {
        // Single-artifact GGUF repos: just that file (plus nothing else).
        return std.mem.eql(u8, path, resolved.gguf_file);
    }
    return shouldDownload(path);
}

// ── Commands ────────────────────────────────────────────────────────────

fn stderrReport(impl: *anyopaque, line: []const u8) void {
    _ = impl;
    log.info("{s}\n", .{line});
}

var stderr_reporter_dummy: u8 = 0;
const stderr_reporter = Reporter{ .impl = &stderr_reporter_dummy, .reportFn = &stderrReport };

/// Resolve + download-if-missing; returns the local model dir (owned).
/// Exits the process with a friendly message on unknown names.
pub fn ensureModelAvailable(allocator: std.mem.Allocator, io: std.Io, name: []const u8) ![]u8 {
    // A path that exists locally is used as-is.
    if (std.fs.path.isAbsolute(name)) return allocator.dupe(u8, name);
    const resolved = resolveShortName(name) orelse {
        log.err("unknown model '{s}'\n", .{name});
        printKnownAliases(io);
        std.process.exit(1);
    };
    const dest = try modelDestPath(allocator, homeDir(), resolved.repo);
    errdefer allocator.free(dest);
    if (modelPresent(io, dest)) return dest;
    try pullRepo(allocator, io, resolved, dest, stderr_reporter, true);
    return dest;
}

pub fn cmdPull(allocator: std.mem.Allocator, io: std.Io, name: []const u8) !void {
    const dir = try ensureModelAvailable(allocator, io, name);
    defer allocator.free(dir);
    log.info("model at {s}\n", .{dir});
    log.info("run it: mlx-serve run {s}\n", .{name});
}

fn printKnownAliases(io: std.Io) void {
    _ = io;
    log.err("known short names (or use any HuggingFace 'org/repo'):\n", .{});
    for (aliases) |a| {
        if (a.is_default) {
            log.err("  {s} (= {s}:{s}) -> {s}\n", .{ a.name, a.name, a.tag, a.repo });
        } else {
            log.err("  {s}:{s} -> {s}\n", .{ a.name, a.tag, a.repo });
        }
    }
}

/// `mlx-serve list` — models on disk under ~/.mlx-serve/models.
pub fn cmdList(allocator: std.mem.Allocator, io: std.Io) !void {
    const root = try modelsRootPath(allocator, homeDir());
    defer allocator.free(root);

    var out_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &out_buf);
    const w = &stdout_w.interface;
    defer w.flush() catch {};

    var dir = std.Io.Dir.openDirAbsolute(io, root, .{ .iterate = true }) catch {
        try w.print("no models yet (looked in {s})\ntry: mlx-serve pull gemma4\n", .{root});
        return;
    };
    defer dir.close(io);

    try w.print("{s: <56} {s: <12} {s: >10}\n", .{ "NAME", "TYPE", "SIZE" });
    var count: usize = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind != .directory) continue;
        if (entry.name.len == 0 or entry.name[0] == '.') continue;
        var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
        defer sub.close(io);
        if (isModelDir(io, &sub)) {
            try printModelRow(io, allocator, w, &sub, entry.name, root);
            count += 1;
            continue;
        }
        // org/ level: one more hop down.
        var sub_it = sub.iterate();
        while (sub_it.next(io) catch null) |sub_entry| {
            if (sub_entry.kind != .directory) continue;
            var leaf = sub.openDir(io, sub_entry.name, .{ .iterate = true }) catch continue;
            defer leaf.close(io);
            if (!isModelDir(io, &leaf)) continue;
            var name_buf: [512]u8 = undefined;
            const full = std.fmt.bufPrint(&name_buf, "{s}/{s}", .{ entry.name, sub_entry.name }) catch continue;
            try printModelRow(io, allocator, w, &leaf, full, root);
            count += 1;
        }
    }
    if (count == 0) {
        try w.print("(none) — try: mlx-serve pull gemma4\n", .{});
    }
}

fn isModelDir(io: std.Io, dir: *std.Io.Dir) bool {
    if (dir.statFile(io, "config.json", .{})) |st| {
        if (st.kind == .file) return true;
    } else |_| {}
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".gguf")) return true;
    }
    return false;
}

fn printModelRow(io: std.Io, allocator: std.mem.Allocator, w: *std.Io.Writer, dir: *std.Io.Dir, name: []const u8, root: []const u8) !void {
    const bytes = dirBytesOneLevel(io, dir);
    // TYPE from the same classification serving uses (gguf → chat via the
    // embedded engines, media modalities, embed, drafter, unsupported) so
    // the list is honest about which rows `run` can actually chat with.
    const abs = std.fmt.allocPrint(allocator, "{s}/{s}", .{ root, name }) catch null;
    defer if (abs) |a| allocator.free(a);
    const kind_label: []const u8 = blk: {
        const a = abs orelse break :blk "?";
        const kind = model_discovery.classifyModelPath(io, allocator, a) orelse break :blk "?";
        break :blk kind.label();
    };
    var size_buf: [32]u8 = undefined;
    try w.print("{s: <56} {s: <12} {s: >10}\n", .{ name, kind_label, formatSize(&size_buf, bytes) });
}

/// Sum file bytes in a model dir INCLUDING one level of subdirectories —
/// media bundles keep their weights in transformer/ vae/ text_encoder/ etc.
/// (the same one-level layout assumption `modelPresent` makes). Top-level-
/// only summing showed a 7 GB FLUX bundle as "6 KB".
fn dirBytesOneLevel(io: std.Io, dir: *std.Io.Dir) u64 {
    var bytes: u64 = 0;
    var it = dir.iterate();
    while (it.next(io) catch null) |entry| {
        switch (entry.kind) {
            .file, .sym_link => {
                const st = dir.statFile(io, entry.name, .{}) catch continue;
                if (st.kind == .file) bytes += @intCast(st.size);
            },
            .directory => {
                var sub = dir.openDir(io, entry.name, .{ .iterate = true }) catch continue;
                defer sub.close(io);
                var sit = sub.iterate();
                while (sit.next(io) catch null) |se| {
                    if (se.kind != .file) continue;
                    const st = sub.statFile(io, se.name, .{}) catch continue;
                    bytes += @intCast(st.size);
                }
            },
            else => {},
        }
    }
    return bytes;
}

pub fn formatSize(buf: []u8, bytes: u64) []const u8 {
    const gb = 1024 * 1024 * 1024;
    const mb = 1024 * 1024;
    if (bytes >= gb) {
        const whole = bytes / gb;
        const tenth = (bytes % gb) * 10 / gb;
        return std.fmt.bufPrint(buf, "{d}.{d} GB", .{ whole, tenth }) catch "?";
    }
    if (bytes >= mb) return std.fmt.bufPrint(buf, "{d} MB", .{bytes / mb}) catch "?";
    return std.fmt.bufPrint(buf, "{d} KB", .{bytes / 1024}) catch "?";
}

// ── REPL (mlx-serve run) ────────────────────────────────────────────────
//
// The REPL is deliberately a real HTTP client against the server's own
// /api/chat endpoint (via curl, streaming NDJSON) — it dogfoods the Ollama
// surface on every keystroke instead of poking internal functions.

pub const Turn = struct {
    role: []const u8,
    content: []const u8,
};

/// /api/chat request body for the REPL conversation so far.
pub fn buildReplChatBody(allocator: std.mem.Allocator, history: []const Turn) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll("{\"model\":\"mlx-serve\",\"stream\":true,\"messages\":[");
    for (history, 0..) |turn, i| {
        if (i > 0) try w.writeAll(",");
        try w.writeAll("{\"role\":");
        try ollama.writeJsonString(w, turn.role);
        try w.writeAll(",\"content\":");
        try ollama.writeJsonString(w, turn.content);
        try w.writeAll("}");
    }
    try w.writeAll("]}");
    return allocator.dupe(u8, out.written());
}

pub const ReplDelta = struct {
    /// Owned by caller.
    content: []u8,
    done: bool,
    eval_count: u64 = 0,
    eval_duration_ns: u64 = 0,
    err: ?[]u8 = null,
};

/// One NDJSON line from /api/chat → the piece the REPL prints.
pub fn parseReplLine(allocator: std.mem.Allocator, line: []const u8) ?ReplDelta {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const root = parsed.value.object;
    if (root.get("error")) |e| {
        if (e == .string) {
            return .{
                .content = allocator.dupe(u8, "") catch return null,
                .done = true,
                .err = allocator.dupe(u8, e.string) catch return null,
            };
        }
    }
    var content: []const u8 = "";
    if (root.get("message")) |m| {
        if (m == .object) {
            if (m.object.get("content")) |c| {
                if (c == .string) content = c.string;
            }
        }
    }
    const done = if (root.get("done")) |d| (d == .bool and d.bool) else false;
    var eval_count: u64 = 0;
    var eval_ns: u64 = 0;
    if (done) {
        if (root.get("eval_count")) |v| {
            if (v == .integer and v.integer > 0) eval_count = @intCast(v.integer);
        }
        if (root.get("eval_duration")) |v| {
            if (v == .integer and v.integer > 0) eval_ns = @intCast(v.integer);
        }
    }
    return .{
        .content = allocator.dupe(u8, content) catch return null,
        .done = done,
        .eval_count = eval_count,
        .eval_duration_ns = eval_ns,
    };
}

/// Interactive loop on the calling thread. Waits for the server to answer
/// /health, then reads prompts from stdin and streams /api/chat responses.
/// Returns when the user exits (/bye or EOF); caller shuts the server down.
pub fn runRepl(allocator: std.mem.Allocator, io: std.Io, port: u16) !void {
    const health_url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/health", .{port});
    defer allocator.free(health_url);
    // Big checkpoints take a while to fault in; poll patiently.
    var waited_ms: u64 = 0;
    while (waited_ms < 15 * 60 * 1000) {
        if (curlFetch(allocator, io, health_url)) |body| {
            allocator.free(body);
            break;
        } else |_| {}
        std.Io.sleep(io, .fromMilliseconds(500), .real) catch {};
        waited_ms += 500;
    }

    const chat_url = try std.fmt.allocPrint(allocator, "http://127.0.0.1:{d}/api/chat", .{port});
    defer allocator.free(chat_url);

    var out_buf: [4096]u8 = undefined;
    var stdout_w = std.Io.File.stdout().writer(io, &out_buf);
    const w = &stdout_w.interface;
    try w.writeAll("\n>>> chat is live — /bye to exit\n");
    try w.flush();

    var history = std.ArrayList(Turn).empty;
    defer {
        for (history.items) |t| allocator.free(t.content);
        history.deinit(allocator);
    }

    var stdin_buf: [16 * 1024]u8 = undefined;
    var stdin_r = std.Io.File.stdin().reader(io, &stdin_buf);
    const r = &stdin_r.interface;

    while (true) {
        try w.writeAll(">>> ");
        try w.flush();
        const line = r.takeDelimiter('\n') catch break orelse break;
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (trimmed.len == 0) continue;
        if (std.mem.eql(u8, trimmed, "/bye") or std.mem.eql(u8, trimmed, "/exit") or std.mem.eql(u8, trimmed, "/quit")) break;

        try history.append(allocator, .{ .role = "user", .content = try allocator.dupe(u8, trimmed) });
        const body = try buildReplChatBody(allocator, history.items);
        defer allocator.free(body);

        const reply = streamOneTurn(allocator, io, chat_url, body, w) catch |err| {
            try w.print("\n[error: {s}]\n", .{@errorName(err)});
            try w.flush();
            continue;
        };
        try history.append(allocator, .{ .role = "assistant", .content = reply });
        try w.writeAll("\n");
        try w.flush();
    }
}

/// POST the body, stream NDJSON, print content deltas as they arrive.
/// Returns the full assistant reply (owned).
fn streamOneTurn(allocator: std.mem.Allocator, io: std.Io, url: []const u8, body: []const u8, w: *std.Io.Writer) ![]u8 {
    var child = std.process.spawn(io, .{
        .argv = &.{ "curl", "-sN", "-X", "POST", "-H", "Content-Type: application/json", "--data-binary", "@-", url },
        .stdin = .pipe,
        .stdout = .pipe,
        .stderr = .ignore,
    }) catch return error.CurlSpawnFailed;
    defer child.kill(io);

    {
        var in_buf: [4096]u8 = undefined;
        var stdin_w = child.stdin.?.writer(io, &in_buf);
        try stdin_w.interface.writeAll(body);
        try stdin_w.interface.flush();
        child.stdin.?.close(io);
        child.stdin = null;
    }

    var full = std.ArrayList(u8).empty;
    errdefer full.deinit(allocator);

    var out_buf: [64 * 1024]u8 = undefined;
    var stdout_r = child.stdout.?.reader(io, &out_buf);
    const r = &stdout_r.interface;
    while (true) {
        const line = r.takeDelimiter('\n') catch break orelse break;
        if (line.len == 0) continue;
        const delta = parseReplLine(allocator, line) orelse continue;
        defer allocator.free(delta.content);
        if (delta.err) |e| {
            defer allocator.free(e);
            try w.print("[server error: {s}]", .{e});
            try w.flush();
            break;
        }
        if (delta.content.len > 0) {
            try w.writeAll(delta.content);
            try w.flush();
            try full.appendSlice(allocator, delta.content);
        }
        if (delta.done) {
            if (delta.eval_count > 0 and delta.eval_duration_ns > 0) {
                const tok_s = @as(f64, @floatFromInt(delta.eval_count)) * 1e9 / @as(f64, @floatFromInt(delta.eval_duration_ns));
                try w.print("\n[{d} tokens, {d:.1} tok/s]", .{ delta.eval_count, tok_s });
            }
            break;
        }
    }
    _ = child.wait(io) catch {};
    return full.toOwnedSlice(allocator);
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

test "cli: resolveShortName aliases, tags, org/repo, hf.co, unknown" {
    // Bare alias picks the family default.
    try testing.expectEqualStrings("mlx-community/gemma-4-e4b-it-4bit", resolveShortName("gemma4").?.repo);
    try testing.expectEqualStrings("ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve", resolveShortName("qwen3.6").?.repo);
    // Tagged alias.
    try testing.expectEqualStrings("mlx-community/gemma-4-12b-it-4bit", resolveShortName("gemma4:12b").?.repo);
    try testing.expectEqualStrings("mlx-community/gemma-4-26b-a4b-it-8bit", resolveShortName("GEMMA4:26B-8BIT").?.repo);
    // :latest behaves like bare.
    try testing.expectEqualStrings("mlx-community/gemma-4-e4b-it-4bit", resolveShortName("gemma4:latest").?.repo);
    // Direct org/repo passthrough, tag stripped, hf.co prefixes stripped.
    try testing.expectEqualStrings("org/repo", resolveShortName("org/repo").?.repo);
    try testing.expectEqualStrings("org/repo", resolveShortName("org/repo:latest").?.repo);
    try testing.expectEqualStrings("org/repo", resolveShortName("hf.co/org/repo").?.repo);
    try testing.expectEqualStrings("org/repo", resolveShortName("https://huggingface.co/org/repo").?.repo);
    // GGUF single-artifact alias carries its file restriction.
    const ds = resolveShortName("deepseek-v4").?;
    try testing.expect(ds.gguf_file.len > 0);
    // Unknown alias-shaped name → null.
    try testing.expect(resolveShortName("doesnotexist") == null);
    try testing.expect(resolveShortName("gemma4:nosuchtag") == null);
}

test "cli: modelDestPath layout" {
    const allocator = testing.allocator;
    const p = try modelDestPath(allocator, "/Users/x", "org/repo");
    defer allocator.free(p);
    try testing.expectEqualStrings("/Users/x/.mlx-serve/models/org/repo", p);
}

test "cli: shouldDownload chat-default selection" {
    try testing.expect(shouldDownload("config.json"));
    try testing.expect(shouldDownload("model.safetensors"));
    try testing.expect(shouldDownload("model-00001-of-00002.safetensors"));
    try testing.expect(shouldDownload("tokenizer.json"));
    try testing.expect(shouldDownload("chat_template.jinja"));
    try testing.expect(shouldDownload("mtp/weights.safetensors"));
    try testing.expect(!shouldDownload(".gitattributes"));
    try testing.expect(!shouldDownload("README.md"));
    try testing.expect(!shouldDownload("assets/demo.png"));
    try testing.expect(!shouldDownload("banner.png"));
    try testing.expect(!shouldDownload("vae/weights.safetensors")); // media subdirs are app-bundle territory
}

test "cli: modelPresentInDir requires a COMPLETE checkpoint" {
    // Regression: an interrupted `pull` (Ctrl-C mid-weights) leaves
    // config.json + model.safetensors.partial. modelPresent used to return
    // true on config.json alone, so the rerun skipped the resume and fed a
    // weightless dir to the loader (SIGSEGV). Present now means: no .partial
    // leftovers anywhere (top level or one subdir deep), and config.json +
    // >=1 .safetensors (MLX) or any .gguf.
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // config.json alone (weights never started): not present.
    try tmp.dir.createDirPath(io, "a");
    try tmp.dir.writeFile(io, .{ .sub_path = "a/config.json", .data = "{}" });
    {
        var d = try tmp.dir.openDir(io, "a", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // config.json + interrupted weights: not present (the user's live repro).
    try tmp.dir.writeFile(io, .{ .sub_path = "a/model.safetensors.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "a", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // Complete single-file checkpoint: present.
    try tmp.dir.createDirPath(io, "b");
    try tmp.dir.writeFile(io, .{ .sub_path = "b/config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "b/model.safetensors", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "b", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(modelPresentInDir(io, d));
    }

    // Complete weights but another file still partial (e.g. tokenizer.json):
    // not present — resume must finish the pull.
    try tmp.dir.writeFile(io, .{ .sub_path = "b/tokenizer.json.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "b", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // Interrupted sidecar one subdir deep (mtp/weights.safetensors.partial):
    // not present.
    try tmp.dir.createDirPath(io, "c/mtp");
    try tmp.dir.writeFile(io, .{ .sub_path = "c/config.json", .data = "{}" });
    try tmp.dir.writeFile(io, .{ .sub_path = "c/model.safetensors", .data = "x" });
    try tmp.dir.writeFile(io, .{ .sub_path = "c/mtp/weights.safetensors.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "c", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }

    // GGUF: the file itself is the checkpoint (no config.json needed)…
    try tmp.dir.createDirPath(io, "g");
    try tmp.dir.writeFile(io, .{ .sub_path = "g/model-Q4_K_M.gguf", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "g", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(modelPresentInDir(io, d));
    }

    // …but a partial GGUF is not.
    try tmp.dir.createDirPath(io, "h");
    try tmp.dir.writeFile(io, .{ .sub_path = "h/model-Q4_K_M.gguf.partial", .data = "x" });
    {
        var d = try tmp.dir.openDir(io, "h", .{ .iterate = true });
        defer d.close(io);
        try testing.expect(!modelPresentInDir(io, d));
    }
}

test "cli: parseTreeJson uses lfs size and skips directories" {
    const allocator = testing.allocator;
    const files = try parseTreeJson(allocator,
        \\[{"type":"directory","path":"mtp","size":0},
        \\ {"type":"file","path":"config.json","size":1234},
        \\ {"type":"file","path":"model.safetensors","size":135,"lfs":{"size":5300000000,"sha256":"x"}}]
    );
    defer freeRepoFiles(allocator, files);
    try testing.expectEqual(@as(usize, 2), files.len);
    try testing.expectEqualStrings("config.json", files[0].path);
    try testing.expectEqual(@as(u64, 1234), files[0].size);
    try testing.expectEqual(@as(u64, 5_300_000_000), files[1].size);
}

test "cli: buildReplChatBody and parseReplLine round-trip" {
    const allocator = testing.allocator;
    const history = [_]Turn{
        .{ .role = "user", .content = "hi \"there\"\n" },
        .{ .role = "assistant", .content = "hello" },
    };
    const body = try buildReplChatBody(allocator, &history);
    defer allocator.free(body);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const msgs = parsed.value.object.get("messages").?.array.items;
    try testing.expectEqual(@as(usize, 2), msgs.len);
    try testing.expectEqualStrings("hi \"there\"\n", msgs[0].object.get("content").?.string);
    try testing.expect(parsed.value.object.get("stream").?.bool);

    const d1 = parseReplLine(allocator, "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"Hey\"},\"done\":false}").?;
    defer allocator.free(d1.content);
    try testing.expectEqualStrings("Hey", d1.content);
    try testing.expect(!d1.done);

    const d2 = parseReplLine(allocator, "{\"model\":\"m\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\",\"eval_count\":50,\"eval_duration\":2000000000}").?;
    defer allocator.free(d2.content);
    try testing.expect(d2.done);
    try testing.expectEqual(@as(u64, 50), d2.eval_count);

    const d3 = parseReplLine(allocator, "{\"error\":\"boom\"}").?;
    defer allocator.free(d3.content);
    defer if (d3.err) |e| allocator.free(e);
    try testing.expect(d3.done);
    try testing.expectEqualStrings("boom", d3.err.?);
}

test "cli: formatSize" {
    var buf: [32]u8 = undefined;
    try testing.expectEqualStrings("5.2 GB", formatSize(&buf, 5_600_000_000));
    try testing.expectEqualStrings("35 MB", formatSize(&buf, 36_700_160));
    try testing.expectEqualStrings("2 KB", formatSize(&buf, 2048));
}

test "cli: dirBytesOneLevel counts weight subdirs (FLUX bundle showed 6 KB)" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();

    // Media-bundle shape: small top-level config + the actual weights one
    // level down in transformer/ and vae/ (the modelPresent layout).
    try tmp.dir.createDirPath(io, "m/transformer");
    try tmp.dir.createDirPath(io, "m/vae");
    try tmp.dir.writeFile(io, .{ .sub_path = "m/config.json", .data = "{}" }); // 2 bytes
    try tmp.dir.writeFile(io, .{ .sub_path = "m/transformer/w.safetensors", .data = "0123456789" }); // 10
    try tmp.dir.writeFile(io, .{ .sub_path = "m/vae/w.safetensors", .data = "0123" }); // 4

    var m = try tmp.dir.openDir(io, "m", .{ .iterate = true });
    defer m.close(io);
    try testing.expectEqual(@as(u64, 16), dirBytesOneLevel(io, &m));
}
