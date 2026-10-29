//! sdk.expert.bank_file: what a bank's open does around its own manifest rules (its `adopt`): the manifest read and
//! parsed, the sidecar opened for the reader at the records' bytes, the quant claim's peek. Each failure is a refusal
//! by name. GLM-5.3's two banks open through `open`; DeepSeek-V4.1's bank opens its sidecar through `openSidecar`.

const std = @import("std");
const sdk = @import("sdk");
const Diag = sdk.Diag;
const io = @import("io.zig");

/// Sets `diag` (when given) and returns `err`: a refusal by name.
pub fn refuse(diag: ?*Diag, err: anytype, comptime fmt: []const u8, args: anytype) @TypeOf(err) {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// A file name with no directory part: the sidecar the manifest names, beside it.
pub fn plainName(s: []const u8) bool {
    return s.len > 0 and std.mem.indexOfScalar(u8, s, '/') == null and !std.mem.eql(u8, s, ".") and !std.mem.eql(u8, s, "..");
}

const max_manifest_bytes = 64 << 20;

/// `dir`'s manifest `file`, parsed as `T` in `a`, an arena (the parse keeps slices of the text).
pub fn readManifest(comptime T: type, a: std.mem.Allocator, fs: std.Io, dir: []const u8, comptime file: []const u8, diag: ?*Diag) !T {
    const path = try std.fmt.allocPrint(a, "{s}/" ++ file, .{dir});
    const text = std.Io.Dir.cwd().readFileAlloc(fs, path, a, .limited(max_manifest_bytes)) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.FileNotFound => return refuse(diag, error.ManifestMissing, file ++ ": not found in {s}", .{dir}),
        else => return refuse(diag, error.ManifestMissing, file ++ ": {s}", .{@errorName(e)}),
    };
    return std.json.parseFromSliceLeaky(T, a, text, .{ .ignore_unknown_fields = true }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return refuse(diag, error.ManifestSyntax, file ++ ": {s}", .{@errorName(e)}),
    };
}

/// How the sidecar file's size must compare with the records' bytes: GLM-5.3's banks hold exactly them, DeepSeek-V4.1's
/// at least them.
pub const SizeRule = enum { exact, at_least };

/// The read pool's descriptor for `path` (`io.openUncached`: no symlink, a regular file, the page cache bypassed), its
/// size against the records' `size` under `rule`.
pub fn openSidecar(path: [:0]const u8, size: u64, rule: SizeRule, diag: ?*Diag) !io.UncachedFd {
    var errno: c_int = 0;
    const f = io.openUncached(path.ptr, &errno) catch |e| return switch (e) {
        error.NotFound => refuse(diag, error.SidecarMissing, "{s}: not found", .{path}),
        error.OpenFailed => refuse(diag, error.SidecarOpen, "{s}: open(O_RDONLY | O_NOFOLLOW) failed, errno {d}", .{ path, errno }),
        error.StatFailed => refuse(diag, error.SidecarOpen, "{s}: fstat failed", .{path}),
        error.NotRegularFile => refuse(diag, error.SidecarGeometry, "{s} is not a regular file", .{path}),
        error.NoCacheRefused => refuse(diag, error.SidecarOpen, "{s}: F_NOCACHE / F_RDAHEAD refused", .{path}),
    };
    errdefer f.close();
    switch (rule) {
        .exact => if (f.size != size) return refuse(diag, error.SidecarGeometry, "{s} is {d} B, the manifest's sidecar {d} B", .{ path, f.size, size }),
        .at_least => if (f.size < size) return refuse(diag, error.SidecarGeometry, "{s} is {d} B < {d}", .{ path, f.size, size }),
    }
    return f;
}

/// A bank's open around its own rules: `dir`'s manifest `file` parsed as `Manifest`, then `Bank.adopt` (the manifest's
/// rules against the model's `config`; it returns the records' bytes) and, with `sidecar`, the sidecar the manifest
/// names opened at exactly those bytes. A refusal frees what the bank took (`Bank.deinit`).
pub fn open(comptime Bank: type, comptime Manifest: type, comptime file: []const u8, allocator: std.mem.Allocator, fs: std.Io, dir: []const u8, config: anytype, diag: ?*Diag, sidecar: bool) !Bank {
    if (!std.fs.path.isAbsolute(dir)) return refuse(diag, error.BankDirNotAbsolute, "bank dir \"{s}\" is not an absolute path", .{dir});
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const m = try readManifest(Manifest, a, fs, dir, file, diag);
    var bank: Bank = .{ .allocator = allocator };
    errdefer bank.deinit();
    const size = try bank.adopt(a, &m, config, diag);
    if (!sidecar) return bank;
    bank.sidecar_path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir, m.sidecar.file }, 0);
    bank.sidecar = try openSidecar(bank.sidecar_path, size, .exact, diag);
    return bank;
}

/// A bank's description for its quant's claim (`sdk_ext.quant.BankPeek`): its `quantization` object (JSON text) and its
/// dims, in `a`.
pub fn peek(a: std.mem.Allocator, quantization: []const u8, hidden: u64, inter: u64, n_experts: u64, n_layers: u64) !sdk.GroupPeek {
    return .{ .quantization = try std.json.parseFromSliceLeaky(std.json.Value, a, quantization, .{}), .hidden = hidden, .inter = inter, .n_experts = n_experts, .n_layers = n_layers, .layers = &.{} };
}
