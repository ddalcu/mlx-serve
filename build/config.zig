//! Configure-time facts every graph reads: the version stamps and engine pins checked into the tree, the macOS SDK and
//! webp the macOS graph links, and the checks that a staged dependency is there before anything compiles.

const std = @import("std");

/// Where the macOS graph finds the SDK's frameworks and webp.
pub const Host = struct {
    /// The macOS SDK's frameworks dir: setting any non-default target field disables Zig's native SDK detection.
    frameworks: ?[]const u8,
    /// `-Dwebp-prefix`: a copy of webp's include/ and lib/ (a box without Homebrew), else null.
    webp_prefix: ?[]const u8,
    /// webp's prefix: the copy, else Homebrew's.
    brew_prefix: []const u8,

    /// Resolves the SDK and webp, and on a macOS target checks webp's version and the staged mlx before any step runs.
    pub fn init(b: *std.Build, target: std.Build.ResolvedTarget) Host {
        const frameworks = macosFrameworks(b, target);
        // Unset: Homebrew's prefix, its version read from `brew`. Set: a directory holding a copy of the formula's
        // include/ and lib/, its version read from the copy's lib/pkgconfig.
        const webp_prefix = b.option([]const u8, "webp-prefix", "Directory holding a copy of webp's include/ and lib/ (unset: Homebrew)");
        if (target.result.os.tag == .macos) {
            if (webp_prefix) |dir| verifyCopiedDeps(b, dir) else verifyBrewDeps(b);
            verifyMlxStage(b);
        }
        return .{ .frameworks = frameworks, .webp_prefix = webp_prefix, .brew_prefix = webp_prefix orelse "/opt/homebrew" };
    }
};

fn macosFrameworks(b: *std.Build, target: std.Build.ResolvedTarget) ?[]const u8 {
    if (target.result.os.tag != .macos) return null;
    var code: u8 = undefined;
    const stdout = b.runAllowFail(
        &.{ "xcrun", "--sdk", "macosx", "--show-sdk-path" },
        &code,
        .inherit,
    ) catch return null;
    const sdk = std.mem.trim(u8, stdout, " \n\r\t");
    if (sdk.len == 0) return null;
    return b.fmt("{s}/System/Library/Frameworks", .{sdk});
}

pub fn buildRootHandle(b: *std.Build) std.Io.Dir {
    return b.root.root_dir.handle;
}

/// The llama.cpp tag staged by scripts/fetch-llama.sh (it writes LLAMA_TAG to
/// `lib/llama/.version`). Read at configure time so a plain `zig build` reports
/// the real tag without app/build.sh having to pass `--llama-tag`. Returns null
/// (→ "unknown") when llama hasn't been fetched yet.
pub fn readLlamaTag(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/llama/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    const trimmed = std.mem.trim(u8, bytes, " \t\r\n");
    return if (trimmed.len == 0) null else b.dupe(trimmed);
}

/// `CFBundleShortVersionString` out of the checked-in app/Info.plist — the one
/// place the current CalVer is committed (app/build.sh stamps it on every real
/// build). Read at configure time so a plain `zig build` reports the same
/// version the last shipped build did, instead of a made-up literal.
pub fn readAppVersion(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "app/Info.plist",
        b.allocator,
        .limited(64 * 1024),
    ) catch return null;
    const key = "<key>CFBundleShortVersionString</key>";
    const at = std.mem.indexOf(u8, bytes, key) orelse return null;
    const open = std.mem.indexOfPos(u8, bytes, at + key.len, "<string>") orelse return null;
    const start = open + "<string>".len;
    const end = std.mem.indexOfPos(u8, bytes, start, "</string>") orelse return null;
    const v = std.mem.trim(u8, bytes[start..end], " \t\r\n");
    return if (v.len > 0) b.dupe(v) else null;
}

/// The ds4 submodule's checked-out commit, short, read without git: its `.git` file names the
/// git dir, whose detached HEAD holds the commit.
pub fn readDs4Commit(b: *std.Build) ?[]const u8 {
    const root = buildRootHandle(b);
    const link = root.readFileAlloc(b.graph.io, "lib/ds4/.git", b.allocator, .limited(512)) catch return null;
    const line = std.mem.trim(u8, link, " \t\r\n");
    const prefix = "gitdir: ";
    if (!std.mem.startsWith(u8, line, prefix)) return null;
    const git_dir = line[prefix.len..];
    const head_path = b.pathJoin(&.{ if (std.fs.path.isAbsolute(git_dir)) git_dir else b.pathJoin(&.{ "lib/ds4", git_dir }), "HEAD" });
    const head = root.readFileAlloc(b.graph.io, head_path, b.allocator, .limited(256)) catch return null;
    const sha = std.mem.trim(u8, head, " \t\r\n");
    if (sha.len < 7 or std.mem.startsWith(u8, sha, "ref:")) return null;
    return b.dupe(sha[0..7]);
}

/// The pinned mlx-c revision from lib/mlx/.version (written by
/// scripts/build-mlx.sh as "mlx=<sha> mlxc=<sha> target=<ver>"), surfaced in
/// `mlx-serve --version`. Returns null (→ "unknown") when not staged yet.
pub fn readMlxcPin(b: *std.Build) ?[]const u8 {
    const bytes = buildRootHandle(b).readFileAlloc(
        b.graph.io,
        "lib/mlx/.version",
        b.allocator,
        .limited(256),
    ) catch return null;
    var it = std.mem.tokenizeScalar(u8, std.mem.trim(u8, bytes, " \t\r\n"), ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "mlxc=")) return b.dupe(tok["mlxc=".len..]);
    }
    return null;
}

/// Module `mlx_stamp`: the staged MLX build's stamp (commits, patch inputs, CUDA arch) as
/// `bytes`, keying the CUDA kernel cache (`mlx.exportPtxCacheDir`); "" when nothing is staged.
/// Copied through the build graph: a configure-time read is cached, so a rebuilt MLX kept the old stamp.
pub fn mlxStampModule(b: *std.Build, stamp_path: []const u8) *std.Build.Module {
    const wf = b.addWriteFiles();
    const src = if (buildRootHandle(b).access(b.graph.io, stamp_path, .{})) |_| blk: {
        _ = wf.addCopyFile(b.path(stamp_path), "version");
        break :blk "pub const bytes = @embedFile(\"version\");\n";
    } else |_| "pub const bytes = \"\";\n";
    return b.createModule(.{ .root_source_file = wf.add("mlx_stamp.zig", src) });
}

/// Configure-time check that scripts/build-mlx.sh has staged the pinned
/// mlx/mlx-c build. Mirrors verifyBrewDeps: fail loudly with the fix, never
/// let the linker produce a confusing -lmlxc error (or silently pick up a
/// leftover brew copy from /opt/homebrew/lib).
fn verifyMlxStage(b: *std.Build) void {
    const stage_ok = blk: {
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/libmlxc.dylib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/lib/mlx.metallib", .{}) catch break :blk false;
        buildRootHandle(b).access(b.graph.io, "lib/mlx/.version", .{}) catch break :blk false;
        break :blk true;
    };
    if (!stage_ok) {
        std.debug.print(
            "\n[mlx-serve] lib/mlx is not staged (self-built mlx + mlx-c). Run:\n" ++
                "  git submodule update --init lib/mlx-src lib/mlxc-src && ./scripts/build-mlx.sh\n\n",
            .{},
        );
        std.process.exit(1);
    }
}

/// A Homebrew formula the macOS graph links: its name, the pkg-config file and header a copied prefix must hold, and
/// the version floor.
const BrewDep = struct { name: []const u8, pc: []const u8, header: []const u8, min: std.SemanticVersion };

const required_brew_deps = [_]BrewDep{
    // mlx + mlx-c are NOT brew deps anymore: they are pinned submodules built
    // by scripts/build-mlx.sh (see addMlxLib) so the NAX kernels ship enabled.
    .{ .name = "webp", .pc = "libwebp", .header = "webp/decode.h", .min = .{ .major = 1, .minor = 6, .patch = 0 } },
};

/// `-Dwebp-prefix=<dir>`: the copy must hold each dependency's header and its pkg-config file, whose `Version:` line
/// meets the same floor the Homebrew check applies.
fn verifyCopiedDeps(b: *std.Build, dir: []const u8) void {
    for (required_brew_deps) |dep| {
        const header = b.fmt("{s}/include/{s}", .{ dir, dep.header });
        std.Io.Dir.cwd().access(b.graph.io, header, .{}) catch {
            std.debug.print("\n[mlx-serve] -Dwebp-prefix={s}: {s} not found\n\n", .{ dir, header });
            std.process.exit(1);
        };
        const pc_path = b.fmt("{s}/lib/pkgconfig/{s}.pc", .{ dir, dep.pc });
        const pc = std.Io.Dir.cwd().readFileAlloc(b.graph.io, pc_path, b.allocator, .limited(16 * 1024)) catch {
            std.debug.print("\n[mlx-serve] -Dwebp-prefix={s}: {s} not found (the copy must include lib/pkgconfig)\n\n", .{ dir, pc_path });
            std.process.exit(1);
        };
        var ver_str: ?[]const u8 = null;
        var lines = std.mem.splitScalar(u8, pc, '\n');
        while (lines.next()) |line| {
            if (std.mem.startsWith(u8, line, "Version:")) ver_str = std.mem.trim(u8, line["Version:".len..], " \t\r");
        }
        const have = std.SemanticVersion.parse(ver_str orelse "") catch {
            std.debug.print("[mlx-serve] cannot read '{s}' version from {s}\n", .{ dep.name, pc_path });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[mlx-serve] -Dwebp-prefix={s}: '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}\n\n",
                .{ dir, dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        }
    }
}

fn verifyBrewDeps(b: *std.Build) void {
    for (required_brew_deps) |dep| {
        var code: u8 = undefined;
        const stdout = b.runAllowFail(
            &.{ "brew", "list", "--versions", dep.name },
            &code,
            .inherit,
        ) catch {
            std.debug.print(
                "\n[mlx-serve] missing Homebrew dependency '{s}' (>= {d}.{d}.{d}). Install with: brew install webp\n\n",
                .{ dep.name, dep.min.major, dep.min.minor, dep.min.patch },
            );
            std.process.exit(1);
        };
        const trimmed = std.mem.trim(u8, stdout, " \n\r\t");
        const space = std.mem.indexOfScalar(u8, trimmed, ' ') orelse {
            std.debug.print("[mlx-serve] cannot parse `brew list --versions {s}` output: {s}\n", .{ dep.name, trimmed });
            std.process.exit(1);
        };
        var ver_str = trimmed[space + 1 ..];
        // Strip Homebrew revision suffix (e.g., "0.6.0_2" -> "0.6.0").
        if (std.mem.indexOfScalar(u8, ver_str, '_')) |us| ver_str = ver_str[0..us];
        const have = std.SemanticVersion.parse(ver_str) catch {
            std.debug.print("[mlx-serve] cannot parse '{s}' version '{s}'\n", .{ dep.name, ver_str });
            std.process.exit(1);
        };
        if (have.order(dep.min) == .lt) {
            std.debug.print(
                "\n[mlx-serve] Homebrew '{s}' is {d}.{d}.{d}; need >= {d}.{d}.{d}. Run: brew upgrade {s}\n\n",
                .{ dep.name, have.major, have.minor, have.patch, dep.min.major, dep.min.minor, dep.min.patch, dep.name },
            );
            std.process.exit(1);
        }
    }
}
