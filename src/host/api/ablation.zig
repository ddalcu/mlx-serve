//! Per-request directional ablation (Arditi et al. 2024): the profiles `--ablation-profiles` loads at launch, the key
//! `--ablation-key-file` names, and a request's `"ablation": {"profile", "strength"}` field resolved against them.
//! A profile is a directory: `manifest.json` plus `direction.safetensors`, one `[hidden]` direction normalized here.

const std = @import("std");
const log = @import("../../sdk/util/log.zig");
const Request = @import("../../sdk/request/ablation.zig").Request;
const Conn = @import("../http/conn.zig").Conn;
const http_server = @import("../http/server.zig");
const LoadedModel = @import("../registry/registry.zig").LoadedModel;

pub const manifest_format = "mlx-serve-ablation";
pub const manifest_version = 1;
/// The request header that carries the ablation key.
pub const key_header = "x-ablation-key";
const min_key_len = 32;
const max_hidden = 1 << 16;

pub const Profile = struct {
    name: []const u8,
    /// Unit direction `[hidden]`; empty when the profile was refused.
    direction: []const f32 = &.{},
    hidden: u32 = 0,
    /// Lowercase hex sha256 of the exact `config.json` bytes of the pack the profile was computed on (the field name LoRA
    /// overlays use for the same binding).
    config_sha256: [64]u8 = @splat('0'),
    /// Content hash of the manifest and tensor bytes.
    hash: u64 = 0,
    /// Why the profile was refused at load; null = usable.
    refused: ?[]const u8 = null,
};

/// The profiles and the key digest, read once at launch and never written after.
pub const Set = struct {
    arena: std.heap.ArenaAllocator,
    profiles: []Profile,
    key_digest: [32]u8,

    pub fn deinit(self: *Set) void {
        self.arena.deinit();
    }

    pub fn find(self: *const Set, name: []const u8) ?*const Profile {
        for (self.profiles) |*p| if (std.mem.eql(u8, p.name, name)) return p;
        return null;
    }

    /// Whether `presented` is the key: sha256 digests compared in constant time, so neither length nor content leaks.
    pub fn keyMatches(self: *const Set, presented: []const u8) bool {
        var d: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(trim(presented), &d, .{});
        return std.crypto.timing_safe.eql([32]u8, d, self.key_digest);
    }
};

pub const LaunchError = error{ AblationFlagsIncomplete, AblationKeyFileRefused, AblationProfilesUnreadable, OutOfMemory };

/// Both flags or neither; one alone is a launch error naming the missing one.
pub fn launch(gpa: std.mem.Allocator, io: std.Io, profiles_dir: ?[]const u8, key_file: ?[]const u8) LaunchError!?Set {
    if (profiles_dir == null and key_file == null) return null;
    if (profiles_dir == null or key_file == null) {
        log.err("[ablation] {s} needs {s} too\n", if (profiles_dir == null) .{ "--ablation-key-file", "--ablation-profiles" } else .{ "--ablation-profiles", "--ablation-key-file" });
        return error.AblationFlagsIncomplete;
    }
    var set: Set = .{ .arena = .init(gpa), .profiles = &.{}, .key_digest = undefined };
    errdefer set.deinit();
    set.key_digest = readKey(set.arena.allocator(), key_file.?) catch |e| {
        log.err("[ablation] key file {s} refused: {s}\n", .{ key_file.?, keyRule(e) });
        return error.AblationKeyFileRefused;
    };
    set.profiles = loadProfiles(set.arena.allocator(), io, profiles_dir.?) catch {
        log.err("[ablation] profiles directory {s} is unreadable\n", .{profiles_dir.?});
        return error.AblationProfilesUnreadable;
    };
    var usable: usize = 0;
    for (set.profiles) |p| {
        if (p.refused) |why| log.err("[ablation] profile '{s}' refused: {s}\n", .{ p.name, why }) else usable += 1;
    }
    if (usable == 0) log.warn("[ablation] no usable profile in {s}\n", .{profiles_dir.?});
    log.info("[ablation] enabled: {d} profile(s) from {s}\n", .{ usable, profiles_dir.? });
    return set;
}

const KeyError = error{ NotRegular, NotOwner, GroupOrOtherAccess, TooShort, Unreadable, OutOfMemory };

fn keyRule(e: KeyError) []const u8 {
    return switch (e) {
        error.NotRegular => "not a regular file",
        error.NotOwner => "not owned by the server's user",
        error.GroupOrOtherAccess => "mode must be 0600 (no group or other access)",
        error.TooShort => "the key must be at least 32 bytes",
        error.Unreadable, error.OutOfMemory => "unreadable",
    };
}

/// The key file's sha256: a regular file the server's user owns, no group or other bits, at least 32 bytes trimmed.
/// One open without following a symlink, the checks on that descriptor, and the read from it: the file checked is the
/// file read.
fn readKey(a: std.mem.Allocator, path: []const u8) KeyError![32]u8 {
    const path_z = a.dupeSentinel(u8, path, 0) catch return error.OutOfMemory;
    defer a.free(path_z);
    const fd = std.c.open(path_z, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true, .NONBLOCK = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return if (std.c._errno().* == @intFromEnum(std.c.E.LOOP)) error.NotRegular else error.Unreadable;
    defer _ = std.c.close(fd);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0) return error.Unreadable;
    if (st.mode & std.c.S.IFMT != std.c.S.IFREG) return error.NotRegular;
    if (st.uid != std.c.getuid()) return error.NotOwner;
    if (st.mode & 0o077 != 0) return error.GroupOrOtherAccess;
    var buf: [4096]u8 = undefined;
    defer std.crypto.secureZero(u8, &buf);
    var len: usize = 0;
    if (st.size >= buf.len) return error.Unreadable;
    while (len < buf.len) {
        const n = std.c.read(fd, buf[len..].ptr, buf.len - len);
        if (n < 0) {
            if (std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
            return error.Unreadable;
        }
        if (n == 0) break;
        len += @intCast(n);
    }
    const key = trim(buf[0..len]);
    if (key.len < min_key_len) return error.TooShort;
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(key, &d, .{});
    return d;
}

fn trim(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n");
}

/// Every subdirectory with a `manifest.json`, in name order; a bad one is kept with the reason it was refused.
fn loadProfiles(a: std.mem.Allocator, io: std.Io, dir_path: []const u8) ![]Profile {
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var out: std.ArrayList(Profile) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .directory) continue;
        var sub = dir.openDir(io, entry.name, .{}) catch continue;
        defer sub.close(io);
        const manifest = sub.readFileAlloc(io, "manifest.json", a, .limited(1 << 16)) catch |e| switch (e) {
            error.FileNotFound => continue,
            else => {
                try out.append(a, .{ .name = try a.dupe(u8, entry.name), .refused = "manifest.json is unreadable" });
                continue;
            },
        };
        const tensor = sub.readFileAlloc(io, "direction.safetensors", a, .limited(64 << 20)) catch null;
        try out.append(a, try parseProfile(a, try a.dupe(u8, entry.name), manifest, tensor));
    }
    std.mem.sortUnstable(Profile, out.items, {}, struct {
        fn lt(_: void, x: Profile, y: Profile) bool {
            return std.mem.lessThan(u8, x.name, y.name);
        }
    }.lt);
    return out.items;
}

/// A profile from its manifest and tensor bytes; refused (with the reason) rather than failed.
pub fn parseProfile(a: std.mem.Allocator, name: []const u8, manifest: []const u8, tensor: ?[]const u8) !Profile {
    var p: Profile = .{ .name = name };
    const parsed = std.json.parseFromSlice(std.json.Value, a, manifest, .{}) catch {
        p.refused = "manifest.json is not JSON";
        return p;
    };
    const root = if (parsed.value == .object) parsed.value.object else {
        p.refused = "manifest.json is not an object";
        return p;
    };
    const format = root.get("format") orelse std.json.Value{ .null = {} };
    if (format != .string or !std.mem.eql(u8, format.string, manifest_format)) {
        p.refused = "manifest format is not " ++ manifest_format;
        return p;
    }
    const version = root.get("version") orelse std.json.Value{ .null = {} };
    if (version != .integer or version.integer != manifest_version) {
        p.refused = "unknown manifest version";
        return p;
    }
    const hidden_v = root.get("hidden_size") orelse std.json.Value{ .null = {} };
    if (hidden_v != .integer or hidden_v.integer <= 0 or hidden_v.integer > max_hidden) {
        p.refused = "hidden_size is missing or out of range";
        return p;
    }
    p.hidden = @intCast(hidden_v.integer);
    const sha = blk: {
        const v = root.get("base_config_sha256") orelse break :blk null;
        break :blk if (v == .string and v.string.len == 64 and isHex(v.string)) v.string else null;
    } orelse {
        p.refused = "base_config_sha256 is missing or not 64 hex digits";
        return p;
    };
    for (&p.config_sha256, sha) |*o, c| o.* = std.ascii.toLower(c);
    const bytes = tensor orelse {
        p.refused = "direction.safetensors is missing";
        return p;
    };
    const dir = readDirection(a, bytes, p.hidden) catch |e| {
        p.refused = switch (e) {
            error.ZeroDirection => "the direction is zero",
            error.ShapeMismatch => "the direction's shape is not [hidden_size]",
            error.UnsupportedDtype => "the direction's dtype is not F32, F16 or BF16",
            error.OutOfMemory => return error.OutOfMemory,
            else => "direction.safetensors is malformed",
        };
        return p;
    };
    p.direction = dir;
    var h = std.hash.Wyhash.init(0);
    h.update(manifest);
    h.update(bytes);
    p.hash = h.final();
    return p;
}

fn isHex(s: []const u8) bool {
    for (s) |c| if (!std.ascii.isHex(c)) return false;
    return true;
}

const TensorError = error{ Malformed, ShapeMismatch, UnsupportedDtype, ZeroDirection, OutOfMemory };

/// The tensor `direction` of a safetensors file as a unit f32 vector of `hidden` values (`[hidden]` or `[1, hidden]`).
fn readDirection(a: std.mem.Allocator, bytes: []const u8, hidden: u32) TensorError![]f32 {
    if (bytes.len < 8) return error.Malformed;
    const hlen = std.mem.readInt(u64, bytes[0..8], .little);
    if (hlen > bytes.len - 8) return error.Malformed;
    const header = bytes[8..][0..@intCast(hlen)];
    const data = bytes[8 + @as(usize, @intCast(hlen)) ..];
    const parsed = std.json.parseFromSlice(std.json.Value, a, header, .{}) catch return error.Malformed;
    if (parsed.value != .object) return error.Malformed;
    const t = parsed.value.object.get("direction") orelse return error.Malformed;
    if (t != .object) return error.Malformed;
    const dtype = t.object.get("dtype") orelse return error.Malformed;
    const shape = t.object.get("shape") orelse return error.Malformed;
    const offs = t.object.get("data_offsets") orelse return error.Malformed;
    if (dtype != .string or shape != .array or offs != .array or offs.array.items.len != 2) return error.Malformed;
    var n: u64 = 1;
    for (shape.array.items) |d| {
        if (d != .integer or d.integer < 0) return error.Malformed;
        n = std.math.mul(u64, n, @intCast(d.integer)) catch return error.Malformed;
    }
    const dims = shape.array.items;
    const shaped = (dims.len == 1 or (dims.len == 2 and dims[0].integer == 1)) and n == hidden;
    if (!shaped) return error.ShapeMismatch;
    const width: usize = if (std.mem.eql(u8, dtype.string, "F32")) 4 else if (std.mem.eql(u8, dtype.string, "F16") or std.mem.eql(u8, dtype.string, "BF16")) 2 else return error.UnsupportedDtype;
    const lo = offs.array.items[0];
    const hi = offs.array.items[1];
    if (lo != .integer or hi != .integer or lo.integer < 0 or hi.integer < lo.integer or hi.integer > data.len) return error.Malformed;
    const raw = data[@intCast(lo.integer)..@intCast(hi.integer)];
    if (raw.len != hidden * width) return error.ShapeMismatch;
    const out = try a.alloc(f32, hidden);
    var norm: f64 = 0;
    for (out, 0..) |*o, i| {
        o.* = switch (width) {
            4 => @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little)),
            else => if (dtype.string[0] == 'B')
                @bitCast(@as(u32, std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)) << 16)
            else
                @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, raw[i * 2 ..][0..2], .little)))),
        };
        if (!std.math.isFinite(o.*)) return error.Malformed;
        norm += @as(f64, o.*) * o.*;
    }
    if (norm < 1e-12) return error.ZeroDirection;
    const inv: f32 = @floatCast(1.0 / @sqrt(norm));
    for (out) |*o| o.* *= inv;
    return out;
}

/// The model a request runs on, as far as a profile cares.
pub const Served = struct {
    hidden: u32,
    /// The served plugin declares `Caps.ablation`.
    serves: bool,
    /// Lowercase hex sha256 of the pack's `config.json`; null when it could not be read.
    config_sha256: ?[64]u8,
};

/// A refusal: the HTTP status line and a message that never holds the key.
pub const Refusal = struct { status: []const u8, code: u16, msg: []const u8 };

pub const Outcome = union(enum) {
    /// No `ablation` field.
    none,
    /// Strength 0: runs as a standard request; the header still reports the profile.
    standard: Selected,
    run: Selected,
    refused: Refusal,
};

pub const Selected = struct {
    profile: *const Profile,
    strength: f32,

    pub fn request(self: Selected) Request {
        var h = std.hash.Wyhash.init(self.profile.hash);
        h.update(std.mem.asBytes(&self.strength));
        return .{ .direction = self.profile.direction, .strength = self.strength, .key = h.final() | 1 };
    }
};

fn bad(msg: []const u8) Outcome {
    return .{ .refused = .{ .status = "400 Bad Request", .code = 400, .msg = msg } };
}

fn unauthorized(msg: []const u8) Outcome {
    return .{ .refused = .{ .status = "401 Unauthorized", .code = 401, .msg = msg } };
}

/// Resolves a request body's `ablation` field. Checked in order: present, enabled, key, shape, profile, model.
/// `msg_buf` holds a refusal message naming the profile; it outlives the returned outcome.
pub fn resolve(set: ?*const Set, field: ?std.json.Value, presented_key: ?[]const u8, served: Served, msg_buf: []u8) Outcome {
    const v = field orelse return .none;
    const s = set orelse return bad("ablation is not enabled on this server");
    const key = presented_key orelse return unauthorized("ablation key missing (X-Ablation-Key header)");
    if (!s.keyMatches(key)) return unauthorized("ablation key rejected");
    if (v != .object) return bad("ablation must be an object {\"profile\": string, \"strength\": number}");
    var it = v.object.iterator();
    while (it.next()) |kv| {
        if (!std.mem.eql(u8, kv.key_ptr.*, "profile") and !std.mem.eql(u8, kv.key_ptr.*, "strength"))
            return bad(std.fmt.bufPrint(msg_buf, "unknown field 'ablation.{s}'", .{clip(kv.key_ptr.*)}) catch "unknown field in ablation");
    }
    const name_v = v.object.get("profile") orelse return bad("ablation.profile is required");
    if (name_v != .string) return bad("ablation.profile must be a string");
    const strength: f32 = if (v.object.get("strength")) |sv| switch (sv) {
        .float => |f| @floatCast(f),
        .integer => |i| if (i >= 0 and i <= 1) @floatFromInt(i) else -1,
        else => return bad("ablation.strength must be a number in [0, 1]"),
    } else 1.0;
    if (!(strength >= 0 and strength <= 1)) return bad("ablation.strength must be a number in [0, 1]");
    const name = name_v.string;
    const p = s.find(name) orelse return bad(std.fmt.bufPrint(msg_buf, "unknown ablation profile '{s}'", .{clip(name)}) catch "unknown ablation profile");
    if (p.refused) |why| return bad(std.fmt.bufPrint(msg_buf, "ablation profile '{s}' was refused at load: {s}", .{ p.name, why }) catch "ablation profile was refused at load");
    if (!served.serves) return bad("this model does not serve ablation");
    if (p.hidden != served.hidden) return bad(std.fmt.bufPrint(msg_buf, "ablation profile '{s}' is for another model (hidden size {d}, the model's is {d})", .{ p.name, p.hidden, served.hidden }) catch "ablation profile is for another model");
    const sha = served.config_sha256 orelse return bad("ablation: the model's config.json could not be read to check the profile");
    if (!std.mem.eql(u8, &sha, &p.config_sha256)) return bad(std.fmt.bufPrint(msg_buf, "ablation profile '{s}' is for another model (config.json digest differs)", .{p.name}) catch "ablation profile is for another model");
    const sel: Selected = .{ .profile = p, .strength = strength };
    return if (strength == 0) .{ .standard = sel } else .{ .run = sel };
}

fn clip(s: []const u8) []const u8 {
    return s[0..@min(s.len, 64)];
}

/// Lowercase hex sha256 of `<model_dir>/config.json`.
pub fn configDigest(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) ?[64]u8 {
    const path = std.fs.path.join(a, &.{ model_dir, "config.json" }) catch return null;
    defer a.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return null;
    defer a.free(bytes);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

/// The `X-Ablation` response header line (CRLF-terminated) for `sel`; the profile name is a directory name the
/// operator chose, kept to header-safe bytes.
pub fn headerLine(buf: []u8, sel: Selected) []const u8 {
    var name_buf: [64]u8 = undefined;
    const name = clip(sel.profile.name);
    for (name, 0..) |c, i| name_buf[i] = if (c > 0x20 and c < 0x7f and c != ';') c else '_';
    return std.fmt.bufPrint(buf, "X-Ablation: profile={s}; strength={d:.3}; hash={x:0>16}\r\n", .{ name_buf[0..name.len], sel.strength, sel.profile.hash }) catch "";
}

/// Resolves a parsed body's `ablation` field for `conn`'s request on `lm`: a run or a strength-0 request sets the
/// conn's response header (and, for a run, `conn.ablation`); a refusal is returned for the caller to send.
pub fn forRequest(conn: *Conn, root: std.json.ObjectMap, lm: *const LoadedModel) ?Refusal {
    const field = root.get("ablation") orelse return null;
    const set = http_server.g_ablation;
    const config = lm.config orelse return bad("this model does not serve ablation").refused;
    const serves = if (config.plugin) |pb| pb.parsed.caps.ablation else false;
    // The pack's config is read only for a request that holds the key (`resolve` checks it first).
    const authorized = if (set) |s| if (conn.ablation_key) |k| s.keyMatches(k) else false else false;
    var a = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer a.deinit();
    const served: Served = .{
        .hidden = config.hidden_size,
        .serves = serves,
        .config_sha256 = if (authorized and serves) configDigest(conn.io, a.allocator(), lm.path) else null,
    };
    const out = resolve(set, field, conn.ablation_key, served, &conn.ablation_header);
    const sel = switch (out) {
        .none => return null,
        .refused => |r| {
            // The message may sit in the header buffer: keep it, but never send it as a header.
            conn.ablation_header_len = 0;
            return r;
        },
        .standard, .run => |sel| sel,
    };
    if (out == .run) conn.ablation = sel.request();
    conn.ablation_header_len = headerLine(&conn.ablation_header, sel).len;
    log.info("[ablation] profile={s} strength={d:.3}{s}\n", .{ sel.profile.name, sel.strength, if (out == .standard) " (standard run)" else "" });
    return null;
}

/// Whether a raw JSON body carries the field (the Ollama routes, which translate the body and would drop it).
pub fn bodyNamesAblation(a: std.mem.Allocator, body: []const u8) bool {
    const parsed = std.json.parseFromSlice(std.json.Value, a, body, .{}) catch return false;
    defer parsed.deinit();
    return parsed.value == .object and parsed.value.object.get("ablation") != null;
}

/// The surfaces that do not serve the field refuse it rather than ignore it.
pub fn refuseOnSurface(root: std.json.ObjectMap) ?Refusal {
    if (root.get("ablation") == null) return null;
    return bad("ablation is served on /v1/chat/completions and /v1/completions only").refused;
}

// ── tests ──

const testing = std.testing;

fn safetensors(a: std.mem.Allocator, dtype: []const u8, shape: []const u8, data: []const u8) ![]u8 {
    const header = try std.fmt.allocPrint(a, "{{\"direction\":{{\"dtype\":\"{s}\",\"shape\":{s},\"data_offsets\":[0,{d}]}}}}", .{ dtype, shape, data.len });
    defer a.free(header);
    const out = try a.alloc(u8, 8 + header.len + data.len);
    std.mem.writeInt(u64, out[0..8], header.len, .little);
    @memcpy(out[8..][0..header.len], header);
    @memcpy(out[8 + header.len ..], data);
    return out;
}

const sha_a = "abababababababababababababababababababababababababababababababab";
const good_manifest = "{\"format\":\"mlx-serve-ablation\",\"version\":1,\"hidden_size\":4,\"base_config_sha256\":\"" ++ sha_a ++ "\"}";

fn f32Bytes(comptime v: [4]f32) [16]u8 {
    var out: [16]u8 = undefined;
    for (v, 0..) |x, i| std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(x), .little);
    return out;
}

test "ablation profile: a valid manifest and tensor give a unit direction and a content hash" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const t = try safetensors(a, "F32", "[4]", &f32Bytes(.{ 3, 0, 4, 0 }));
    const p = try parseProfile(a, "x", good_manifest, t);
    try testing.expect(p.refused == null);
    try testing.expectApproxEqAbs(@as(f32, 0.6), p.direction[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8), p.direction[2], 1e-6);
    try testing.expect(p.hash != 0);
    const t2 = try safetensors(a, "F32", "[1,4]", &f32Bytes(.{ 1, 0, 0, 0 }));
    try testing.expect((try parseProfile(a, "y", good_manifest, t2)).refused == null);
}

test "ablation profile: each malformed input is refused by name, never loaded" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ok = try safetensors(a, "F32", "[4]", &f32Bytes(.{ 1, 0, 0, 0 }));
    const cases = [_]struct { m: []const u8, t: ?[]const u8, why: []const u8 }{
        .{ .m = "nope", .t = ok, .why = "not JSON" },
        .{ .m = "{\"format\":\"other\",\"version\":1}", .t = ok, .why = "format" },
        .{ .m = "{\"format\":\"mlx-serve-ablation\",\"version\":2,\"hidden_size\":4}", .t = ok, .why = "version" },
        .{ .m = "{\"format\":\"mlx-serve-ablation\",\"version\":1,\"hidden_size\":4}", .t = ok, .why = "base_config_sha256" },
        .{ .m = good_manifest, .t = null, .why = "missing" },
        .{ .m = good_manifest, .t = try safetensors(a, "F32", "[3]", &f32Bytes(.{ 1, 0, 0, 0 })), .why = "shape" },
        .{ .m = good_manifest, .t = try safetensors(a, "I32", "[4]", &f32Bytes(.{ 1, 0, 0, 0 })), .why = "dtype" },
        .{ .m = good_manifest, .t = try safetensors(a, "F32", "[4]", &f32Bytes(.{ 0, 0, 0, 0 })), .why = "zero" },
        .{ .m = good_manifest, .t = "short", .why = "malformed" },
    };
    for (cases) |c| {
        const p = try parseProfile(a, "bad", c.m, c.t);
        const why = p.refused orelse return error.TestExpectedRefusal;
        try testing.expect(std.mem.indexOf(u8, why, c.why) != null);
        try testing.expectEqual(@as(usize, 0), p.direction.len);
    }
}

fn testSet(a: std.mem.Allocator, key: []const u8) !Set {
    var set: Set = .{ .arena = .init(a), .profiles = &.{}, .key_digest = undefined };
    std.crypto.hash.sha2.Sha256.hash(key, &set.key_digest, .{});
    const aa = set.arena.allocator();
    const ps = try aa.alloc(Profile, 2);
    ps[0] = try parseProfile(aa, "good", good_manifest, try safetensors(aa, "F32", "[4]", &f32Bytes(.{ 1, 0, 0, 0 })));
    ps[1] = try parseProfile(aa, "broken", "{}", null);
    set.profiles = ps;
    return set;
}

const test_key = "kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk";

fn json(a: std.mem.Allocator, text: []const u8) !std.json.Value {
    return (try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}));
}

test "ablation request: every refusal row of the failure matrix, and the key never in a message" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var set = try testSet(testing.allocator, test_key);
    defer set.deinit();
    var sha: [64]u8 = undefined;
    @memcpy(&sha, sha_a);
    const served: Served = .{ .hidden = 4, .serves = true, .config_sha256 = sha };
    var buf: [256]u8 = undefined;
    const field = try json(a, "{\"profile\":\"good\",\"strength\":0.5}");
    const Case = struct { set: ?*const Set, field: []const u8, key: ?[]const u8, served: Served, code: u16, says: []const u8 };
    const other: Served = .{ .hidden = 4, .serves = true, .config_sha256 = @splat('c') };
    const cases = [_]Case{
        .{ .set = null, .field = "{\"profile\":\"good\"}", .key = test_key, .served = served, .code = 400, .says = "not enabled" },
        .{ .set = &set, .field = "{\"profile\":\"good\"}", .key = null, .served = served, .code = 401, .says = "missing" },
        .{ .set = &set, .field = "{\"profile\":\"good\"}", .key = "wrong-key-of-another-length", .served = served, .code = 401, .says = "rejected" },
        .{ .set = &set, .field = "\"good\"", .key = test_key, .served = served, .code = 400, .says = "must be an object" },
        .{ .set = &set, .field = "{\"profile\":\"good\",\"strenght\":1}", .key = test_key, .served = served, .code = 400, .says = "unknown field" },
        .{ .set = &set, .field = "{\"strength\":1}", .key = test_key, .served = served, .code = 400, .says = "profile is required" },
        .{ .set = &set, .field = "{\"profile\":7}", .key = test_key, .served = served, .code = 400, .says = "must be a string" },
        .{ .set = &set, .field = "{\"profile\":\"good\",\"strength\":1.5}", .key = test_key, .served = served, .code = 400, .says = "[0, 1]" },
        .{ .set = &set, .field = "{\"profile\":\"good\",\"strength\":-0.1}", .key = test_key, .served = served, .code = 400, .says = "[0, 1]" },
        .{ .set = &set, .field = "{\"profile\":\"good\",\"strength\":\"1\"}", .key = test_key, .served = served, .code = 400, .says = "[0, 1]" },
        .{ .set = &set, .field = "{\"profile\":\"nope\"}", .key = test_key, .served = served, .code = 400, .says = "unknown ablation profile 'nope'" },
        .{ .set = &set, .field = "{\"profile\":\"broken\"}", .key = test_key, .served = served, .code = 400, .says = "refused at load" },
        .{ .set = &set, .field = "{\"profile\":\"good\"}", .key = test_key, .served = .{ .hidden = 4, .serves = false, .config_sha256 = sha }, .code = 400, .says = "does not serve" },
        .{ .set = &set, .field = "{\"profile\":\"good\"}", .key = test_key, .served = .{ .hidden = 8, .serves = true, .config_sha256 = sha }, .code = 400, .says = "hidden size" },
        .{ .set = &set, .field = "{\"profile\":\"good\"}", .key = test_key, .served = other, .code = 400, .says = "digest differs" },
    };
    for (cases) |c| {
        const o = resolve(c.set, try json(a, c.field), c.key, c.served, &buf);
        const r = switch (o) {
            .refused => |r| r,
            else => return error.TestExpectedRefusal,
        };
        try testing.expectEqual(c.code, r.code);
        try testing.expect(std.mem.indexOf(u8, r.msg, c.says) != null);
        try testing.expect(std.mem.indexOf(u8, r.msg, test_key) == null);
        if (c.key) |k| try testing.expect(std.mem.indexOf(u8, r.msg, k) == null);
    }
    try testing.expect(resolve(&set, null, test_key, served, &buf) == .none);
    const run = resolve(&set, field, test_key ++ "\n", served, &buf);
    try testing.expect(run == .run and run.run.strength == 0.5);
    const zero = resolve(&set, try json(a, "{\"profile\":\"good\",\"strength\":0}"), test_key, served, &buf);
    try testing.expect(zero == .standard);
    const dflt = resolve(&set, try json(a, "{\"profile\":\"good\"}"), test_key, served, &buf);
    try testing.expect(dflt == .run and dflt.run.strength == 1.0);
}

test "ablation request: the key folds the strength in, never 0" {
    var set = try testSet(testing.allocator, test_key);
    defer set.deinit();
    const p = set.find("good").?;
    const a: Selected = .{ .profile = p, .strength = 1.0 };
    const b: Selected = .{ .profile = p, .strength = 0.5 };
    try testing.expect(a.request().key != b.request().key);
    try testing.expect(a.request().key != 0);
    try testing.expectEqual(a.request().key, a.request().key);
}

test "ablation launch: one flag alone is refused, neither is off" {
    try testing.expect((try launch(testing.allocator, testing.io, null, null)) == null);
    try testing.expectError(error.AblationFlagsIncomplete, launch(testing.allocator, testing.io, "/nonexistent", null));
    try testing.expectError(error.AblationFlagsIncomplete, launch(testing.allocator, testing.io, null, "/nonexistent"));
}

test "ablation key file: mode, length and trim rules" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var root_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = root_buf[0..try tmp.dir.realPath(testing.io, &root_buf)];
    const a = testing.allocator;
    const path = try std.fs.path.join(a, &.{ root, "key" });
    defer a.free(path);
    const path_z = try a.dupeSentinel(u8, path, 0);
    defer a.free(path_z);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "key", .data = test_key ++ "\n" });
    _ = std.c.chmod(path_z, 0o644);
    try testing.expectError(error.GroupOrOtherAccess, readKey(a, path));
    _ = std.c.chmod(path_z, 0o600);
    var want: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(test_key, &want, .{});
    try testing.expectEqualSlices(u8, &want, &(try readKey(a, path)));
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "key", .data = "short\n" });
    try testing.expectError(error.TooShort, readKey(a, path));
    try testing.expectError(error.NotRegular, readKey(a, root));
    const link = try std.fs.path.join(a, &.{ root, "link" });
    defer a.free(link);
    const link_z = try a.dupeSentinel(u8, link, 0);
    defer a.free(link_z);
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "key", .data = test_key });
    try testing.expect(std.c.symlink(path_z, link_z) == 0);
    try testing.expectError(error.NotRegular, readKey(a, link));
    try testing.expectError(error.Unreadable, readKey(a, "/nonexistent/key"));
}

test "ablation header: names the profile and strength, header-safe" {
    var set = try testSet(testing.allocator, test_key);
    defer set.deinit();
    var buf: [256]u8 = undefined;
    const line = headerLine(&buf, .{ .profile = set.find("good").?, .strength = 0.5 });
    try testing.expect(std.mem.startsWith(u8, line, "X-Ablation: profile=good; strength=0.500; hash="));
    try testing.expect(std.mem.endsWith(u8, line, "\r\n"));
}
