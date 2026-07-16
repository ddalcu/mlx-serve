//! `mlx-serve --version` report — the versions of the app and every embedded
//! engine, WITHOUT booting the HTTP server. The macOS app spawns
//! `mlx-serve --version` as a one-shot subprocess and parses this so Settings
//! can show engine versions without a running server (Swift side:
//! `EngineVersions.parse`). Keep this a pure formatter — main.zig gathers the
//! runtime values (`mlx_version()`, `ggml_version()`/`ggml_commit()`) and the
//! build-time pins (`build_options`) and calls `writeReport`.

const std = @import("std");

/// Every version string surfaced by `--version`. Runtime-queryable ones
/// (`mlx`, `ggml`, `ggml_commit`) come from the linked libraries; the rest are
/// build-time pins (Homebrew mlx-c, the pinned ds4 submodule commit, the
/// fetch-llama.sh `LLAMA_TAG`, the compiled GGUF file-format version).
pub const Info = struct {
    /// mlx-serve app version (`build_options.version`).
    app: []const u8,
    /// MLX core, from `mlx_version()` at runtime.
    mlx: []const u8,
    /// mlx-c C bindings, a Homebrew pin (no runtime API).
    mlx_c: []const u8,
    /// ggml library version, from `ggml_version()` at runtime.
    ggml: []const u8,
    /// ggml short commit, from `ggml_commit()`. May be empty.
    ggml_commit: []const u8,
    /// llama.cpp release tag (`bNNNN`), the fetch-llama.sh pin.
    llama_tag: []const u8,
    /// GGUF file-format version (compiled `GGUF_VERSION`).
    gguf_format: []const u8,
    /// Pinned ds4 submodule short commit (no runtime API).
    ds4_commit: []const u8,
};

/// Render one `name value` line per component in a stable order. Machine-
/// parseable: the first whitespace-delimited token is the component name, the
/// remainder is its version (which may itself contain spaces, e.g.
/// `ggml 0.16.0 (47c786924)`). A pin with no value collapses to `unknown` so
/// every line always has a value token.
pub fn writeReport(w: *std.Io.Writer, info: Info) !void {
    try w.print("mlx-serve {s}\n", .{val(info.app)});
    try w.print("mlx {s}\n", .{val(info.mlx)});
    try w.print("mlx-c {s}\n", .{val(info.mlx_c)});
    if (info.ggml_commit.len > 0) {
        try w.print("ggml {s} ({s})\n", .{ val(info.ggml), info.ggml_commit });
    } else {
        try w.print("ggml {s}\n", .{val(info.ggml)});
    }
    try w.print("llama.cpp {s}\n", .{val(info.llama_tag)});
    try w.print("gguf {s}\n", .{val(info.gguf_format)});
    try w.print("ds4 {s}\n", .{val(info.ds4_commit)});
}

/// Allocate the report as a string (test/caller convenience).
pub fn report(allocator: std.mem.Allocator, info: Info) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try writeReport(&out.writer, info);
    return allocator.dupe(u8, out.written());
}

/// A blank pin reads as `unknown` — never an empty value token, so the Swift
/// parser always gets `name` + `version`.
fn val(s: []const u8) []const u8 {
    return if (s.len == 0) "unknown" else s;
}

test "version: report renders one name-value line per component" {
    const s = try report(std.testing.allocator, .{
        .app = "26.7.9",
        .mlx = "0.32.0",
        .mlx_c = "0.6.0",
        .ggml = "0.16.0",
        .ggml_commit = "47c786924",
        .llama_tag = "b9999",
        .gguf_format = "3",
        .ds4_commit = "80ebbc3",
    });
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(
        \\mlx-serve 26.7.9
        \\mlx 0.32.0
        \\mlx-c 0.6.0
        \\ggml 0.16.0 (47c786924)
        \\llama.cpp b9999
        \\gguf 3
        \\ds4 80ebbc3
        \\
    , s);
}

test "version: empty ggml commit drops the parenthetical; blank pins read as unknown" {
    const s = try report(std.testing.allocator, .{
        .app = "26.7.9",
        .mlx = "0.32.0",
        .mlx_c = "", // build.sh couldn't resolve it (dev build)
        .ggml = "0.16.0",
        .ggml_commit = "",
        .llama_tag = "b9999",
        .gguf_format = "3",
        .ds4_commit = "",
    });
    defer std.testing.allocator.free(s);
    try std.testing.expectEqualStrings(
        \\mlx-serve 26.7.9
        \\mlx 0.32.0
        \\mlx-c unknown
        \\ggml 0.16.0
        \\llama.cpp b9999
        \\gguf 3
        \\ds4 unknown
        \\
    , s);
}
