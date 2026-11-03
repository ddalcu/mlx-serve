//! The rate tally's settings and file. Every stream layer keeps, per expert, the rate at which its decode routes request
//! it (`policy.LayerPolicy`: an EMA per route); the top `ratio` of each layer's experts by rate are pinned resident, the
//! rest of the rows run the transition-window policy. The tally file carries the rates from one run to the next: the
//! load reads it (the preload fills the prompt rows hottest first, and prefill evicts for pinned experts only), each
//! request's end writes it. A first run with no file plans as the stream did before the tally. Every model the expert
//! stream serves takes these settings; each arch sets its own default ratio (`Settings.ratio`): GLM-5.3 0.5, where it
//! was measured; DeepSeek-V4.1 0, off until a DeepSeek-V4.1 measurement. A ratio of 0 is no tally at all: the stream
//! gets none, nothing is counted, no file is read or written. The file lives under ~/.mlx-stream/<model key>/; a
//! cell holds it fixed with `expert_tally_file` (`pinned`: read at the load, never written; `off`: no file).

const std = @import("std");
const env = @import("mlx_host").env;
const log = @import("sdk").log;

/// The pinned share of each stream layer's experts, on [0, 1]: 0 pins nothing (every row runs the transition-window
/// policy), 1 pins every expert the rows hold.
pub const default_ratio: f64 = 0.5;
/// The rate's half-life in the layer's decode routes (alpha = 1 - 2^(-1/window)): 32 routes, about 110 tokens of
/// GLM-5.3's MTP rounds (3.5 tokens a round). On two recorded coding decodes replayed at the box's rows
/// (work/glm53-eval/routes/rate_sim.py), with a tally file from the other seed, windows 8 / 16 / 32 / 64 / 128 gave
/// +0.31 / +0.43 / +0.53 / +0.55 / +0.50 points of hit rate at ratio 0.5, and with a file from a prose decode
/// +0.04 / +0.06 / -0.06 / -0.47 / -1.16: 32 is the longest window a file from another workload does not cost.
pub const default_window: u32 = 32;

pub const file_name = "expert-tally.bin";
pub const magic = "MLXSTALY".*;
pub const version: u32 = 1;

/// The file: this header, then `n_layers` x `n_experts` f32 rates (each expert's share of its layer's decode routes in
/// the last request that routed the layer), little-endian.
pub const Header = extern struct {
    magic: [8]u8 = magic,
    version: u32 = version,
    n_layers: u32,
    n_experts: u32,
    window: u32,
};

/// The tally file's use (`expert_tally_file`): `on` reads it at the load and writes it at each request's end; `pinned`
/// reads it and never writes it (a cell's arms all start from the same file); `off` neither reads nor writes one (the
/// tally runs in the process alone, every load a first run).
pub const File = enum { on, pinned, off };

/// The model-settings.json keys: `expert_resident_ratio` (a number on [0, 1]), `expert_tally_window` (an integer of at
/// least 1 route) and `expert_tally_file` ("on", "pinned" or "off"). A value of another type is ignored (the default
/// stays); a number outside its range, or another name, is refused at load by name (`check`).
pub const Settings = struct {
    ratio: f64 = default_ratio,
    window: u32 = default_window,
    file: File = .on,
    ratio_out_of_range: ?f64 = null,
    window_out_of_range: ?i64 = null,
    file_unknown: bool = false,

    /// Whether the stream keeps a tally at all: a ratio of 0 pins nothing, so nothing is counted either.
    pub fn enabled(s: Settings) bool {
        return s.ratio > 0;
    }

    /// Whether the load reads the tally file.
    pub fn readsFile(s: Settings) bool {
        return s.enabled() and s.file != .off;
    }

    /// Whether each request's end writes the tally file.
    pub fn writesFile(s: Settings) bool {
        return s.enabled() and s.file == .on;
    }

    /// The keys of one model's settings object; returns whether either was present.
    pub fn apply(s: *Settings, obj: std.json.ObjectMap) bool {
        var any = false;
        if (obj.get("expert_resident_ratio")) |v| {
            const x: ?f64 = switch (v) {
                .float => |f| f,
                .integer => |i| @floatFromInt(i),
                else => null,
            };
            if (x) |r| {
                if (r >= 0 and r <= 1) s.ratio = r else s.ratio_out_of_range = r;
                any = true;
            }
        }
        if (obj.get("expert_tally_window")) |v| if (v == .integer) {
            if (v.integer >= 1 and v.integer <= std.math.maxInt(u32)) s.window = @intCast(v.integer) else s.window_out_of_range = v.integer;
            any = true;
        };
        if (obj.get("expert_tally_file")) |v| if (v == .string) {
            if (std.meta.stringToEnum(File, v.string)) |f| s.file = f else s.file_unknown = true;
            any = true;
        };
        return any;
    }

    /// The load's refusal of an impossible value, by name (`arch` prefixes the line).
    pub fn check(s: Settings, comptime arch: []const u8) error{ ResidentRatioOutOfRange, TallyWindowOutOfRange, TallyFileUnknown }!void {
        if (s.ratio_out_of_range) |r| {
            log.warn(arch ++ ": load refused: expert_resident_ratio {d} is outside [0, 1] (ResidentRatioOutOfRange)\n", .{r});
            return error.ResidentRatioOutOfRange;
        }
        if (s.window_out_of_range) |w| {
            log.warn(arch ++ ": load refused: expert_tally_window {d} is not a whole number of routes of at least 1 (TallyWindowOutOfRange)\n", .{w});
            return error.TallyWindowOutOfRange;
        }
        if (s.file_unknown) {
            log.warn(arch ++ ": load refused: expert_tally_file names none of on, pinned, off (TallyFileUnknown)\n", .{});
            return error.TallyFileUnknown;
        }
    }
};

/// The pins of a layer of `experts` experts at `ratio`: floor(ratio x experts).
pub fn pinLimit(ratio: f64, experts: u32) u32 {
    const x = @floor(std.math.clamp(ratio, 0, 1) * @as(f64, @floatFromInt(experts)) + 1e-9);
    return @min(experts, @as(u32, @intFromFloat(x)));
}

/// The EMA's factor per route at a half-life of `window` routes.
pub fn alphaOf(window: u32) f32 {
    return @floatCast(1 - std.math.pow(f64, 0.5, 1 / @as(f64, @floatFromInt(@max(window, 1)))));
}

/// The model's tally file: $HOME/.mlx-stream/<key>/expert-tally.bin (`key`: the arch and the model's directory name,
/// `keyOf`). Null without a HOME.
pub fn pathOf(a: std.mem.Allocator, key: []const u8) !?[]u8 {
    const home = env.get(.HOME) orelse return null;
    return try std.fmt.allocPrint(a, "{s}/.mlx-stream/{s}/" ++ file_name, .{ std.mem.span(home), key });
}

/// `<arch>-<the model directory's last component>`, one directory name.
pub fn keyOf(a: std.mem.Allocator, arch: []const u8, model_dir: []const u8) ![]u8 {
    const trimmed = std.mem.trimEnd(u8, model_dir, "/");
    const base = std.fs.path.basename(trimmed);
    const out = try std.fmt.allocPrint(a, "{s}-{s}", .{ arch, if (base.len == 0) "model" else base });
    for (out) |*c| if (c.* == '/' or c.* == 0) {
        c.* = '_';
    };
    return out;
}

pub const Read = union(enum) {
    /// The rates (`n_layers` x `n_experts`, the caller's).
    rates: []f32,
    missing,
    /// A file of another geometry or version, or short: ignored (the run is a first run).
    mismatched,
};

/// The file at `path` for a stream of `n_layers` layers of `n_experts` experts.
pub fn read(a: std.mem.Allocator, io: std.Io, path: []const u8, n_layers: u32, n_experts: u32) !Read {
    const want = @sizeOf(Header) + @as(usize, n_layers) * n_experts * @sizeOf(f32);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(want + 1)) catch |e| switch (e) {
        error.FileNotFound => return .missing,
        error.OutOfMemory => return error.OutOfMemory,
        else => return .mismatched,
    };
    defer a.free(bytes);
    if (bytes.len != want) return .mismatched;
    const h = std.mem.bytesToValue(Header, bytes[0..@sizeOf(Header)]);
    if (!std.mem.eql(u8, &h.magic, &magic) or h.version != version or h.n_layers != n_layers or h.n_experts != n_experts) return .mismatched;
    const rates = try a.alloc(f32, @as(usize, n_layers) * n_experts);
    @memcpy(std.mem.sliceAsBytes(rates), bytes[@sizeOf(Header)..]);
    return .{ .rates = rates };
}

/// `rates` written to `path` (its directory made): a temporary file beside it, then renamed over it.
pub fn write(a: std.mem.Allocator, io: std.Io, path: []const u8, rates: []const f32, n_layers: u32, n_experts: u32, window: u32) !void {
    std.debug.assert(rates.len == @as(usize, n_layers) * n_experts);
    const dir = std.fs.path.dirname(path) orelse return error.BadTallyPath;
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const bytes = try a.alloc(u8, @sizeOf(Header) + rates.len * @sizeOf(f32));
    defer a.free(bytes);
    const h: Header = .{ .n_layers = n_layers, .n_experts = n_experts, .window = window };
    @memcpy(bytes[0..@sizeOf(Header)], std.mem.asBytes(&h));
    @memcpy(bytes[@sizeOf(Header)..], std.mem.sliceAsBytes(rates));
    const tmp = try std.fmt.allocPrint(a, "{s}.tmp-{d}", .{ path, std.c.getpid() });
    defer a.free(tmp);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp, .data = bytes });
    errdefer std.Io.Dir.cwd().deleteFile(io, tmp) catch {};
    try std.Io.Dir.rename(std.Io.Dir.cwd(), tmp, std.Io.Dir.cwd(), path, io);
}
