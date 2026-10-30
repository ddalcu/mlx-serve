//! mlx/device.
//! Owns naxArchGeneration, DeviceGen, device_gen_cache, deviceGeneration, naxArchSupportedFrom, naxArchIsG17, and 11 more.

const std = @import("std");
const env = @import("../env.zig");
const builtin = @import("builtin");
const mlx = @import("ffi.zig");
const ane_offload = @import("../sdk/ane/offload.zig");
const testing = std.testing;

// ── NAX m16 verify tile (M5-class matrix units; see the section comment
// above for provenance, gating, and the never-build-off-probe rule) ──

/// Case-insensitive prefix match on the M5-class GPU family identifier.
/// Prefix (not equality) is MTPLX's shipping behavior — device variants
/// report suffixed forms ("applegpu_g17s", "applegpu_g17d").
pub fn naxArchGeneration(arch: []const u8) struct { gen: u32, phone: bool } {
    if (arch.len < 3) return .{ .gen = 0, .phone = false };
    var buf: [128]u8 = undefined;
    const n = @min(arch.len, buf.len);
    for (arch[0..n], 0..) |c, i| buf[i] = std.ascii.toLower(c);
    const a = buf[0..n];
    // Generic compiler targets such as air64_v27 do not name a GPU generation.
    const suffix = if (std.mem.startsWith(u8, a, "applegpu_g")) a[10..] else if (a[0] == 'g') a[1..] else return .{ .gen = 0, .phone = false };
    var digits: usize = 0;
    while (digits < suffix.len and std.ascii.isDigit(suffix[digits])) digits += 1;
    if (digits == 0 or suffix.len - digits > 1) return .{ .gen = 0, .phone = false };
    if (digits < suffix.len and !std.ascii.isAlphabetic(suffix[digits])) return .{ .gen = 0, .phone = false };
    return .{
        .gen = std.fmt.parseInt(u32, suffix[0..digits], 10) catch 0,
        .phone = digits < suffix.len and suffix[digits] == 'p',
    };
}

pub const DeviceGen = struct { gen: u32, phone: bool };

var device_gen_cache: ?DeviceGen = null;

/// The device's GPU generation, parsed once. Kernel plans keyed on generation (`qmv2.planFor`, the
/// down+reduce lane count) read this one cache.
pub fn deviceGeneration() DeviceGen {
    if (device_gen_override) |g| return g;
    if (device_gen_cache) |g| return g;
    var buf: [128]u8 = undefined;
    const parsed = naxArchGeneration(gpuArchitecture(&buf) orelse "");
    const g: DeviceGen = .{ .gen = parsed.gen, .phone = parsed.phone };
    device_gen_cache = g;
    return g;
}

pub fn naxArchSupportedFrom(arch: []const u8) bool {
    const parsed = naxArchGeneration(arch);
    const floor: u32 = if (parsed.phone) 18 else 17;
    return parsed.gen >= floor;
}

pub fn naxArchIsG17(arch: []const u8) bool {
    return naxArchSupportedFrom(arch);
}

/// "26.4"/"26.4.1"-style product version at least req_major.req_minor.
/// Unparseable components read as 0 (mirrors MTPLX: int() failures fall
/// back to 0, so garbage can never satisfy the floor).
pub fn macosVersionAtLeast(ver: []const u8, req_major: u32, req_minor: u32) bool {
    var it = std.mem.splitScalar(u8, ver, '.');
    const major = std.fmt.parseInt(u32, it.first(), 10) catch 0;
    const minor: u32 = if (it.next()) |mn| (std.fmt.parseInt(u32, mn, 10) catch 0) else 0;
    return major > req_major or (major == req_major and minor >= req_minor);
}

/// The whole availability gate, pure over its inputs (mirror of MTPLX's
/// nax_available()): not force-fallback, G17-class GPU, macOS >= 26.2 (the
/// MetalPerformancePrimitives floor).
pub fn naxAvailableFrom(force_fallback: bool, arch: []const u8, os_ver: []const u8) bool {
    if (force_fallback) return false;
    if (!naxArchIsG17(arch)) return false;
    return macosVersionAtLeast(os_ver, 26, 2);
}

/// Human-readable NAX status for `--version` / the app's Settings ("nax"
/// line, first token on/off, remainder the reason). Hardware + OS is the
/// whole story for OUR binaries: the bundled MLX always ships the NAX
/// kernels (build-mlx.sh + tests/test_mlx_staged_nax.sh assert them), so
/// this mirrors MLX's stock-op is_nax_available() gate (GPU gen >= 17,
/// macOS >= 26.2 — that symbol is not exported, hence the mirror).
/// Deliberately ignores the verifyQmm-lane QA env switches: those scope
/// our custom kernel lane, not MLX's stock dispatch.
pub fn naxStatusFrom(arch: []const u8, os_ver: []const u8) []const u8 {
    if (!naxArchIsG17(arch)) return "off (requires M5-class GPU)";
    if (!macosVersionAtLeast(os_ver, 26, 2)) return "off (requires macOS 26.2+)";
    return "on (M5 neural accelerators)";
}

/// naxStatusFrom over the real device: mlx device-info arch + sysctl OS
/// version (same sources as the verifyQmm probe).
pub fn naxStatus() []const u8 {
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    return naxStatusFrom(arch, ver);
}

extern "c" fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int;

/// "kern.osproductversion" → "26.4"-style string (the sysctl mirror of
/// Python's platform.mac_ver()[0]).
pub fn macosProductVersion(buf: []u8) ?[]const u8 {
    if (comptime !builtin.os.tag.isDarwin()) return null;
    var len: usize = buf.len;
    if (sysctlbyname("kern.osproductversion", buf.ptr, &len, null, 0) != 0) return null;
    var n = @min(len, buf.len);
    while (n > 0 and buf[n - 1] == 0) n -= 1;
    if (n == 0) return null;
    return buf[0..n];
}

/// GPU architecture identifier off mlx device info ("applegpu_g16" on the
/// M4 Max). The returned pointer from mlx is borrowed from the info object,
/// so the string is copied into the caller's buffer before the info frees.
pub fn gpuArchitecture(buf: []u8) ?[]const u8 {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) != 0) return null;
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return null;
    var cstr: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") != 0) return null;
    const arch = std.mem.span(cstr);
    if (arch.len == 0 or arch.len > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    return buf[0..arch.len];
}

pub fn computeNaxAvailable() bool {
    var force = false;
    if (env.get(.FORCE_GPU_FAMILY_FALLBACK)) |p| {
        const v = std.mem.span(p);
        force = v.len > 0 and v[0] == '1';
    }
    var arch_buf: [128]u8 = undefined;
    const arch = gpuArchitecture(&arch_buf) orelse "";
    var ver_buf: [64]u8 = undefined;
    const ver = macosProductVersion(&ver_buf) orelse "";
    return naxAvailableFrom(force, arch, ver);
}

var nax_avail_cache: ?bool = null;
/// MLX's own NAX gate (M5-class GPU AND macOS >= 26.2), read once.
pub fn naxAvailable() bool {
    if (nax_avail_cache == null) nax_avail_cache = computeNaxAvailable();
    return nax_avail_cache.?;
}

/// Total physical RAM (`hw.memsize`); 0 when the read fails (gates that
/// consume this must treat 0 as "unknown", never as "tiny machine").
pub fn totalMemBytes() u64 {
    return ane_offload.totalMemBytes();
}

/// Test seam: pin the GPU generation every generation-keyed plan reads (null = the device's).
pub var device_gen_override: ?DeviceGen = null;

// ── tests ──

test "NAX availability rejects compiler versions and malformed GPU identifiers" {
    for ([_][]const u8{ "air64_v27", "AIR64_V27", "metal_27", "foo17", "gfoobar27", "g17dd", "applegpu_g", "applegpu_g17_extra" }) |arch| {
        try testing.expect(!naxAvailableFrom(false, arch, "26.6.2"));
    }
    try testing.expect(naxAvailableFrom(false, "AppleGPU_G17S", "26.6.2"));
    try testing.expect(naxAvailableFrom(false, "g18p", "26.6.2"));
    try testing.expect(!naxAvailableFrom(false, "g17p", "26.6.2"));
}

test "NAX availability probe: G17 prefix + macOS 26.2 floor + fallback rehearsal (pure parts)" {
    // Mirrors MTPLX's shipping nax_available() gate exactly (nax_verify.py).
    // Arch: case-insensitive prefix match on "applegpu_g17".
    try testing.expect(naxArchIsG17("applegpu_g17s"));
    try testing.expect(naxArchIsG17("AppleGPU_G17"));
    try testing.expect(naxArchIsG17("applegpu_g17"));
    try testing.expect(!naxArchIsG17("applegpu_g16")); // M4-class
    try testing.expect(!naxArchIsG17("applegpu_g13"));
    try testing.expect(!naxArchIsG17(""));
    try testing.expect(naxArchIsG17("g17"));

    // macOS floor: >= 26.2 (where MetalPerformancePrimitives ships).
    try testing.expect(macosVersionAtLeast("26.2", 26, 2));
    try testing.expect(macosVersionAtLeast("26.4.1", 26, 2));
    try testing.expect(macosVersionAtLeast("27.0", 26, 2));
    try testing.expect(macosVersionAtLeast("27", 26, 2));
    try testing.expect(!macosVersionAtLeast("26.1", 26, 2));
    try testing.expect(!macosVersionAtLeast("26", 26, 2));
    try testing.expect(!macosVersionAtLeast("25.5", 26, 2));
    // Unparseable components read as 0 and never satisfy the floor (MTPLX
    // semantics: int() failures fall back to 0).
    try testing.expect(!macosVersionAtLeast("", 26, 2));
    try testing.expect(!macosVersionAtLeast("garbage", 26, 2));
    try testing.expect(!macosVersionAtLeast("26.x", 26, 2));

    // The combined gate.
    try testing.expect(naxAvailableFrom(false, "applegpu_g17d", "26.2"));
    try testing.expect(!naxAvailableFrom(false, "applegpu_g16", "26.2"));
    try testing.expect(!naxAvailableFrom(false, "applegpu_g17d", "26.1"));
    // diag force-gpu-family-fallback QA rehearsal: an M5 pretends the
    // units are absent so the exact M1-M4 plain-SIMD path runs there.
    try testing.expect(!naxAvailableFrom(true, "applegpu_g17d", "26.4"));
}

test "naxStatusFrom: on for G17 + 26.2, off names the missing leg (--version / Settings display)" {
    // Hardware + OS both satisfied → active. Kernels-present is a build-time
    // invariant for our binaries (build-mlx.sh + tests/test_mlx_staged_nax.sh),
    // so hardware capability == MLX's stock-op is_nax_available() gate.
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusFrom("applegpu_g17s", "26.2"));
    try testing.expectEqualStrings("on (M5 neural accelerators)", naxStatusFrom("applegpu_g17d", "27.0"));
    // Pre-M5 GPU: the GPU is the blocker regardless of OS.
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusFrom("applegpu_g16s", "26.4"));
    try testing.expectEqualStrings("off (requires M5-class GPU)", naxStatusFrom("", "26.4"));
    // M5 on an old OS: the OS is the blocker (bundle min is 26.2 anyway, but
    // a dev build can run anywhere).
    try testing.expectEqualStrings("off (requires macOS 26.2+)", naxStatusFrom("applegpu_g17s", "26.1"));
}

test "naxArchSupportedFrom: desktop gen>=17, phone gen>=18" {
    try std.testing.expect(naxArchSupportedFrom("applegpu_g17s"));
    try std.testing.expect(naxArchSupportedFrom("applegpu_g17"));
    try std.testing.expect(naxArchSupportedFrom("g17"));
    try std.testing.expect(naxArchSupportedFrom("applegpu_g18s"));
    try std.testing.expect(naxArchSupportedFrom("g18"));
    try std.testing.expect(!naxArchSupportedFrom("applegpu_g16s"));
    try std.testing.expect(!naxArchSupportedFrom("g16"));
    try std.testing.expect(!naxArchSupportedFrom("applegpu_g17p"));
    try std.testing.expect(!naxArchSupportedFrom("g17p"));
    try std.testing.expect(naxArchSupportedFrom("applegpu_g18p"));
    try std.testing.expect(naxArchSupportedFrom("g18p"));
    try std.testing.expect(!naxArchSupportedFrom("garbage"));
    try std.testing.expect(!naxArchSupportedFrom(""));
}
