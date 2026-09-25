//! Directional steering: one unit-norm `hidden`-wide f32 direction per trunk
//! layer, applied at runtime as `y -= scale * d * dot(d, y)`. The attn arm
//! edits the attention block's output before its residual write; the ffn arm
//! edits every branch of the residual after the MLP write. A bank file is raw
//! little-endian f32 `[n_layers][hidden]` with no header.
//!
//! Pure module: the bank registry owns host rows and per-dtype MLX rows,
//! the projection is a plain op chain in the activation dtype. Nothing here
//! reads the Transformer.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");

const MAX_BANKS: u8 = 8;
const MAX_NAME: usize = 1024;
/// Bare registry names stay short; a longer string must be an absolute path.
const MAX_BARE_NAME: usize = 64;
const SCALE_LIMIT: f32 = 100;

/// What one forward applies. `bank` is meaningless when both scales are 0.
pub const Req = struct {
    bank: u8 = 0,
    ffn: f32 = 0,
    attn: f32 = 0,

    pub fn isOff(r: Req) bool {
        return r.ffn == 0 and r.attn == 0;
    }
};

/// Does this `model_type` have steering seams? Answers from the arch tag alone, so a
/// status route can tell "this arch cannot steer" from "the model is not loaded yet".
pub fn archSupported(model_type: []const u8) bool {
    return std.mem.startsWith(u8, model_type, "qwen4_exp");
}

/// The active default and a caller-owned copy of its bank path, read as one unit.
pub const ActivePath = struct { req: Req, path: []const u8 };

const Scales = struct { ffn: f32, attn: f32 };

/// A file with neither scale steers the ffn arm at 1.
pub fn resolveScales(ffn: ?f32, attn: ?f32) Scales {
    if (ffn == null and attn == null) return .{ .ffn = 1, .attn = 0 };
    return .{ .ffn = ffn orelse 0, .attn = attn orelse 0 };
}

pub fn validateScale(v: f32) !void {
    if (!std.math.isFinite(v) or @abs(v) > SCALE_LIMIT) return error.SteeringScaleRange;
}

/// A configured setting: name or absolute path plus scales. Inline storage so
/// the value survives the settings JSON it was parsed from.
pub const Configured = struct {
    name_buf: [MAX_NAME]u8 = undefined,
    name_len: u16 = 0,
    ffn: f32 = 1,
    attn: f32 = 0,

    pub fn init(n: []const u8, ffn: f32, attn: f32) !Configured {
        if (n.len > MAX_NAME or !(std.fs.path.isAbsolute(n) or isBareName(n))) return error.BadSteeringName;
        var c: Configured = .{ .ffn = ffn, .attn = attn, .name_len = @intCast(n.len) };
        @memcpy(c.name_buf[0..n.len], n);
        return c;
    }

    pub fn name(self: *const Configured) []const u8 {
        return self.name_buf[0..self.name_len];
    }
};

/// The per-model setting: absent key = follow the launch flags, `null` = off
/// even with launch flags, object = this configuration. A plain struct (not a
/// union) so the configs and load requests that carry it stay zeroable;
/// all-zero = inherit.
pub const Setting = struct {
    state: enum(u8) { inherit = 0, off, configured } = .inherit,
    cfg: Configured = .{},

    pub const inherit: Setting = .{};
    pub const off: Setting = .{ .state = .off };

    pub fn configured(c: Configured) Setting {
        return .{ .state = .configured, .cfg = c };
    }

    pub fn configuredOrNull(self: Setting) ?Configured {
        return if (self.state == .configured) self.cfg else null;
    }

    /// `null` for a malformed object: the settings file is hand-editable and
    /// a typo must never stop a load (the caller logs and ignores).
    pub fn fromJsonValue(v: std.json.Value) ?Setting {
        switch (v) {
            .null => return off,
            .object => |o| {
                const n = o.get("name") orelse return null;
                if (n != .string) return null;
                return configure(n.string, jsonF32(o, "ffn") catch return null, jsonF32(o, "attn") catch return null) catch null;
            },
            else => return null,
        }
    }
};

fn configure(name: []const u8, ffn: ?f32, attn: ?f32) !Setting {
    const sc = resolveScales(ffn, attn);
    try validateScale(sc.ffn);
    try validateScale(sc.attn);
    return Setting.configured(try Configured.init(name, sc.ffn, sc.attn));
}

/// The launch flags as one setting: no file = `.inherit`; a scale without a file is a
/// named refusal.
pub fn settingFromFlags(file: ?[]const u8, ffn: ?f32, attn: ?f32) !Setting {
    const f = file orelse {
        if (ffn != null or attn != null) return error.SteeringScaleWithoutFile;
        return Setting.inherit;
    };
    return configure(f, ffn, attn);
}

fn jsonF32(o: std.json.ObjectMap, key: []const u8) !?f32 {
    const v = o.get(key) orelse return null;
    return switch (v) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => error.SteeringScaleRange,
    };
}

/// A capture id names one dump directory: short, no separators.
pub fn isCaptureId(n: []const u8) bool {
    return n.len > 0 and n.len <= 64 and isBareName(n);
}

pub fn isBareName(n: []const u8) bool {
    if (n.len == 0 or n.len > MAX_BARE_NAME) return false;
    for (n) |c| {
        const ok = std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-';
        if (!ok) return false;
    }
    return true;
}

/// Always absolute: a relative `HOME` would hand `openDirAbsolute` a relative path (UB).
pub fn registryDir(buf: []u8) ![]const u8 {
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    if (!std.fs.path.isAbsolute(home)) return error.BadSteeringPath;
    return std.fmt.bufPrint(buf, "{s}/.mlx-serve/steering", .{home});
}

/// A bare name maps into the registry dir; an absolute path passes through.
fn resolvePath(buf: []u8, name_or_path: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(name_or_path)) {
        if (name_or_path.len > buf.len) return error.BadSteeringName;
        @memcpy(buf[0..name_or_path.len], name_or_path);
        return buf[0..name_or_path.len];
    }
    if (!isBareName(name_or_path)) return error.BadSteeringName;
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = try registryDir(&dir_buf);
    return std.fmt.bufPrint(buf, "{s}/{s}.f32", .{ dir, name_or_path }) catch error.BadSteeringName;
}

const FileId = struct { size: u64, mtime: i128 };

fn expectedBytes(n_layers: u32, hidden: u32) u64 {
    return @as(u64, n_layers) * hidden * @sizeOf(f32);
}

/// Prove the file on OUR side before mlx sees it (the `lora.validatePath`
/// rule): absolute, a regular file, exactly `n_layers*hidden` f32s.
fn validateFile(io: std.Io, path: []const u8, n_layers: u32, hidden: u32) !FileId {
    if (!std.fs.path.isAbsolute(path)) return error.BadSteeringPath;
    // Stat, never open: opening a FIFO blocks the conn thread until a writer appears.
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return error.BadSteeringPath;
    if (st.kind != .file) return error.BadSteeringPath;
    if (st.size != expectedBytes(n_layers, hidden)) return error.SteeringFileSize;
    return .{ .size = st.size, .mtime = st.mtime.nanoseconds };
}

fn readBank(allocator: std.mem.Allocator, io: std.Io, path: []const u8, n_layers: u32, hidden: u32) ![]f32 {
    const want = expectedBytes(n_layers, hidden);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(want + 1)) catch |e| switch (e) {
        error.OutOfMemory => return e,
        else => return error.BadSteeringPath,
    };
    defer allocator.free(bytes);
    if (bytes.len != want) return error.SteeringFileSize;
    const rows = try allocator.alloc(f32, @as(usize, n_layers) * hidden);
    errdefer allocator.free(rows);
    @memcpy(std.mem.sliceAsBytes(rows), bytes);
    return rows;
}

// ---- bank registry ----

const Bank = struct {
    path: []u8,
    id: FileId,
    host: []f32,
    /// Per-layer `[hidden]` rows in the activation dtype, built on the
    /// inference thread at first use.
    rows: ?[]mlx.mlx_array = null,
    rows_dtype: mlx.mlx_dtype = .float32,
    refs: u32 = 0,
    last_used: u64 = 0,
};

/// What a request body's `steering` field says: `null` = off for this
/// request; an object names a bank (`name` or `file`) and/or scales; scales
/// alone ride the active default's bank.
pub const Spec = struct {
    name: ?[]const u8 = null,
    ffn: ?f32 = null,
    attn: ?f32 = null,
    off: bool = false,

    pub fn fromJsonValue(v: std.json.Value) !Spec {
        switch (v) {
            .null => return .{ .off = true },
            .object => |o| {
                var spec: Spec = .{};
                // Both would leave one silently ignored.
                if (o.get("name") != null and o.get("file") != null) return error.BadSteeringRequest;
                if (o.get("name") orelse o.get("file")) |n| switch (n) {
                    .string => |str| spec.name = str,
                    .null => spec.off = true,
                    else => return error.BadSteeringRequest,
                };
                spec.ffn = try jsonF32(o, "ffn");
                spec.attn = try jsonF32(o, "attn");
                if (spec.ffn) |x| try validateScale(x);
                if (spec.attn) |x| try validateScale(x);
                return spec;
            },
            else => return error.BadSteeringRequest,
        }
    }
};

/// A connection-owned pin: released by `errdefer` until the ref is handed to
/// a slot (`disarm`, done by `Scheduler.submit` once the slot is queued) or
/// to the active default (`swapDefault`).
pub const Reservation = struct {
    banks: ?*Banks = null,
    req: Req = .{},
    armed: bool = false,
    /// A direction-capture request rides with its steering: the dump
    /// directory the tail forward writes into. Owned by the handler.
    capture_dir: ?[]const u8 = null,

    pub fn none(banks: ?*Banks) Reservation {
        return .{ .banks = banks };
    }

    pub fn release(self: *Reservation) void {
        if (!self.armed) return;
        self.armed = false;
        self.banks.?.releaseBank(self.req.bank);
    }

    pub fn disarm(self: *Reservation) void {
        self.armed = false;
    }
};

pub const Banks = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    n_layers: u32,
    hidden: u32,
    /// Guards the table, every `refs` change and `default` together, so a
    /// snapshot of the default pins its bank in the same critical section
    /// as a swap would release it.
    mu: std.Io.Mutex = .init,
    /// Serializes `POST /v1/steering` per model: persist-then-swap runs as
    /// one unit. Lock order is `control_mu` -> `file_mu` and `control_mu` -> `mu`, never the reverse:
    /// `swapDefault` and `release` both take `mu` while this is held.
    control_mu: std.Io.Mutex = .init,
    slots: [MAX_BANKS]?Bank = @splat(null),
    default: Req = .{},
    tick: u64 = 0,
    /// MLX rows of evicted banks, freed by the inference thread (the sole
    /// mlx caller) on its next drain.
    free_later: std.ArrayList([]mlx.mlx_array) = .empty,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, n_layers: u32, hidden: u32) Banks {
        return .{ .allocator = allocator, .io = io, .n_layers = n_layers, .hidden = hidden };
    }

    /// Inference thread (or a test): frees every MLX array.
    pub fn deinit(self: *Banks) void {
        self.drainFreeLater();
        for (&self.slots) |*slot| if (slot.*) |*b| {
            self.freeRows(b);
            self.allocator.free(b.host);
            self.allocator.free(b.path);
            slot.* = null;
        };
        self.free_later.deinit(self.allocator);
    }

    fn refsOf(self: *Banks, idx: u8) u32 {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        return if (self.slots[idx]) |b| b.refs else 0;
    }

    /// The path of a bank this caller PINS. A caller holding no pin must use
    /// `defaultPathInto`: an unpinned slot can be evicted and its path freed between
    /// the read and the use.
    pub fn pathOf(self: *Banks, idx: u8) []const u8 {
        return if (self.slots[idx]) |b| b.path else "";
    }

    /// The active default and a COPY of its path (`path` slices `buf`), read as one locked
    /// unit, so a concurrent swap + eviction cannot free the path this caller formats.
    pub fn defaultPathInto(self: *Banks, buf: []u8) ActivePath {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const req = self.default;
        if (req.isOff()) return .{ .req = req, .path = "" };
        const p = if (self.slots[req.bank]) |b| b.path else "";
        @memcpy(buf[0..p.len], p);
        return .{ .req = req, .path = buf[0..p.len] };
    }

    /// Conn thread. Validates, reads and pins; both scales 0 pins nothing. The name is
    /// validated even then: `POST /v1/steering` persists what it is given.
    pub fn reserve(self: *Banks, name_or_path: []const u8, ffn: f32, attn: f32) !Reservation {
        try validateScale(ffn);
        try validateScale(attn);
        var pbuf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try resolvePath(&pbuf, name_or_path);
        const id = try validateFile(self.io, path, self.n_layers, self.hidden);
        if (ffn == 0 and attn == 0) return Reservation.none(self);

        self.mu.lockUncancelable(self.io);
        if (self.findLocked(path, id)) |idx| {
            self.pinLocked(idx);
            self.mu.unlock(self.io);
            return .{ .banks = self, .req = .{ .bank = idx, .ffn = ffn, .attn = attn }, .armed = true };
        }
        self.mu.unlock(self.io);

        // The read happens outside the lock; a racing insert of the same
        // file is deduped below.
        const host = try readBank(self.allocator, self.io, path, self.n_layers, self.hidden);
        errdefer self.allocator.free(host);
        const path_owned = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(path_owned);

        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        if (self.findLocked(path, id)) |idx| {
            self.allocator.free(host);
            self.allocator.free(path_owned);
            self.pinLocked(idx);
            return .{ .banks = self, .req = .{ .bank = idx, .ffn = ffn, .attn = attn }, .armed = true };
        }
        const idx = try self.freeSlotLocked();
        self.slots[idx] = .{ .path = path_owned, .id = id, .host = host };
        self.pinLocked(idx);
        return .{ .banks = self, .req = .{ .bank = idx, .ffn = ffn, .attn = attn }, .armed = true };
    }

    /// Conn thread: a request's spec against this model. No name = the
    /// active default with the spec's scales laid over it.
    pub fn reserveSpec(self: *Banks, spec: Spec) !Reservation {
        if (spec.off) return Reservation.none(self);
        if (spec.name) |n| {
            const sc = resolveScales(spec.ffn, spec.attn);
            return try self.reserve(n, sc.ffn, sc.attn);
        }
        var snap = self.snapshotDefault();
        if (!snap.armed) return snap;
        if (spec.ffn) |f| snap.req.ffn = f;
        if (spec.attn) |a| snap.req.attn = a;
        if (snap.req.isOff()) {
            snap.release();
            return Reservation.none(self);
        }
        return snap;
    }

    /// Conn thread: the active default, pinned in the same critical section.
    fn snapshotDefault(self: *Banks) Reservation {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        if (self.default.isOff()) return Reservation.none(self);
        self.pinLocked(self.default.bank);
        return .{ .banks = self, .req = self.default, .armed = true };
    }

    /// CONSUMES the reservation: its pin becomes the default's and `res` is disarmed.
    pub fn swapDefault(self: *Banks, res: *Reservation) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        const old = self.default;
        self.default = if (res.armed) res.req else .{};
        res.disarm();
        if (!old.isOff()) self.unpinLocked(old.bank);
    }

    /// Slot-side release (inference thread cleanup drain), exactly once per pin.
    pub fn releaseBank(self: *Banks, idx: u8) void {
        self.mu.lockUncancelable(self.io);
        defer self.mu.unlock(self.io);
        self.unpinLocked(idx);
    }

    fn findLocked(self: *Banks, path: []const u8, id: FileId) ?u8 {
        for (self.slots, 0..) |s, i| if (s) |b| {
            if (b.id.size == id.size and b.id.mtime == id.mtime and std.mem.eql(u8, b.path, path)) return @intCast(i);
        };
        return null;
    }

    fn pinLocked(self: *Banks, idx: u8) void {
        const b = &self.slots[idx].?;
        b.refs += 1;
        self.tick += 1;
        b.last_used = self.tick;
    }

    fn unpinLocked(self: *Banks, idx: u8) void {
        const b = &self.slots[idx].?;
        std.debug.assert(b.refs > 0);
        b.refs -= 1;
    }

    /// An empty slot, else the least recently used unpinned bank (its MLX
    /// rows deferred to the inference thread), else busy.
    fn freeSlotLocked(self: *Banks) !u8 {
        for (self.slots, 0..) |s, i| if (s == null) return @intCast(i);
        var victim: ?u8 = null;
        for (self.slots, 0..) |s, i| if (s) |b| {
            if (b.refs != 0) continue;
            if (victim == null or b.last_used < self.slots[victim.?].?.last_used) victim = @intCast(i);
        };
        const idx = victim orelse return error.SteeringBanksBusy;
        const b = &self.slots[idx].?;
        if (b.rows) |rows| try self.free_later.append(self.allocator, rows);
        b.rows = null;
        self.allocator.free(b.host);
        self.allocator.free(b.path);
        self.slots[idx] = null;
        return idx;
    }

    fn freeRows(self: *Banks, b: *Bank) void {
        if (b.rows) |rows| {
            for (rows) |r| _ = mlx.mlx_array_free(r);
            self.allocator.free(rows);
        }
        b.rows = null;
    }

    /// Inference thread: free the rows eviction handed over.
    pub fn drainFreeLater(self: *Banks) void {
        self.mu.lockUncancelable(self.io);
        var list = self.free_later;
        self.free_later = .empty;
        self.mu.unlock(self.io);
        for (list.items) |rows| {
            for (rows) |r| _ = mlx.mlx_array_free(r);
            self.allocator.free(rows);
        }
        list.deinit(self.allocator);
    }

    /// Inference thread. The caller holds a pin, so the bank cannot be
    /// evicted under it; only this thread ever writes `rows`.
    pub fn rowsFor(self: *Banks, idx: u8, dt: mlx.mlx_dtype, s: mlx.mlx_stream) ![]mlx.mlx_array {
        const b = &self.slots[idx].?;
        if (b.rows) |rows| if (b.rows_dtype == dt) return rows;
        self.freeRows(b);
        const rows = try self.allocator.alloc(mlx.mlx_array, self.n_layers);
        errdefer self.allocator.free(rows);
        var built: usize = 0;
        errdefer for (rows[0..built]) |r| {
            _ = mlx.mlx_array_free(r);
        };
        const shape = [_]c_int{@intCast(self.hidden)};
        for (rows, 0..) |*r, l| {
            r.* = try hostAs(b.host[l * self.hidden ..].ptr, &shape, dt, s);
            built += 1;
            try mlx.check(mlx.mlx_array_eval(r.*));
        }
        b.rows = rows;
        b.rows_dtype = dt;
        return rows;
    }

    /// Inference thread: one `[B,1,..,1,hidden]` direction per batched row,
    /// gathered on the host (B is small, a mixed group is rare).
    pub fn gatherRows(self: *Banks, banks: []const u8, layer: usize, ndim: usize, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
        const hidden: usize = self.hidden;
        const buf = try self.allocator.alloc(f32, banks.len * hidden);
        defer self.allocator.free(buf);
        for (banks, 0..) |idx, r| {
            const b = &self.slots[idx].?;
            @memcpy(buf[r * hidden .. (r + 1) * hidden], b.host[layer * hidden .. (layer + 1) * hidden]);
        }
        var shape: [4]c_int = .{ @intCast(banks.len), 1, 1, 1 };
        shape[ndim - 1] = @intCast(hidden);
        return hostAs(buf.ptr, shape[0..ndim], dt, s);
    }
};

// ---- projection ----

/// Host f32 data as an array of `dt`.
fn hostAs(data: [*]const f32, shape: []const c_int, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_data(data, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, dt, s));
    return out;
}

/// A `[1]` scalar in `dt`: a bare f32 scalar array promotes bf16 operands.
pub fn scalarAs(v: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    return hostAs(&[_]f32{v}, &.{1}, dt, s);
}

/// Per-row scales as a `[B,1,..,1]` array of `dt` (ndim dims).
pub fn columnAs(vals: []const f32, ndim: usize, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape: [4]c_int = .{ @intCast(vals.len), 1, 1, 1 };
    return hostAs(vals.ptr, shape[0..ndim], dt, s);
}

/// `x - scale * d * dot(d, x)` over the last axis.
fn projectLast(x: mlx.mlx_array, d: mlx.mlx_array, scale: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (try projectFused(x, d, scale, s)) |y| return y;
    return projectChain(x, d, scale, s);
}

pub var fused_override: ?bool = null;
var fused_env: ?bool = null;
pub var fold_override: ?bool = null;
var fold_env: ?bool = null;

/// A default-on kill switch: `=0` turns it off; the env is read once.
fn switchOn(override: ?bool, cache: *?bool, name: [*:0]const u8) bool {
    if (override) |v| return v;
    if (cache.*) |v| return v;
    const raw = std.c.getenv(name);
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    cache.* = on;
    return on;
}

/// `MLX_SERVE_STEER_FOLD=0` keeps the edit out of the hyper-connection read (the seams
/// flush and project on their own). `MLX_SERVE_STEER_FUSED=0` turns the fold off too, so
/// that switch alone restores the composed chain at every width.
pub fn foldEnabled() bool {
    return fusedEnabled() and switchOn(fold_override, &fold_env, "MLX_SERVE_STEER_FOLD");
}

fn fusedEnabled() bool {
    return switchOn(fused_override, &fused_env, "MLX_SERVE_STEER_FUSED");
}

// One threadgroup per row; `meta` = {rows per batch element, n directions, n scales}
// (1 = shared, else per batch element). The dot accumulates in f32 and the output
// rounds ONCE, so the bar is fp32 truth, never bit identity with the chain.
const PROJECT_SOURCE =
    \\uint tid = thread_index_in_threadgroup;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint row = threadgroup_position_in_grid.y;
    \\threadgroup float tg[8];
    \\const int b = int(row) / meta[0];
    \\const device T* x = x_in + (size_t)row * (size_t)H;
    \\const device T* d = d_in + (meta[1] > 1 ? (size_t)b * (size_t)H : 0);
    \\device T* y = y_out + (size_t)row * (size_t)H;
    \\float a = 0.0f;
    \\for (int k = int(tid); k < H; k += 256) a += float(x[k]) * float(d[k]);
    \\a = simd_sum(a);
    \\if (lane == 0) tg[sg] = a;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float dot = 0.0f;
    \\for (int g = 0; g < 8; ++g) dot += tg[g];
    \\float coeff = dot * float(sc_in[meta[2] > 1 ? b : 0]);
    \\for (int k = int(tid); k < H; k += 256) y[k] = T(float(x[k]) - coeff * float(d[k]));
;

var project_kernel: ?mlx.mlx_fast_metal_kernel = null;
/// Keyed on the output's full shape: the ffn `[B,S,hc,hidden]` and attn `[B,S,hidden]`
/// arms can share a row count (the `transformer.ShapeKey` rule).
const ProjectCfgKey = struct {
    dims: [8]c_int = @splat(0),
    len: u8 = 0,
    dtype: mlx.mlx_dtype = .bfloat16,

    fn from(sh: []const c_int, dt: mlx.mlx_dtype) ProjectCfgKey {
        var k = ProjectCfgKey{ .dtype = dt };
        for (sh, 0..) |d, i| {
            if (i >= k.dims.len) break;
            k.dims[i] = d;
        }
        k.len = @intCast(@min(sh.len, k.dims.len));
        return k;
    }
};

/// Keyed LRU: the two arms alternate shapes every layer, so one slot would rebuild per layer.
const PROJECT_CFG_SLOTS = 8;
const ProjectCfgEntry = struct { key: ProjectCfgKey, cfg: mlx.mlx_fast_metal_kernel_config, stamp: u64 };
var project_cfgs: [PROJECT_CFG_SLOTS]?ProjectCfgEntry = @splat(null);
var project_cfg_clock: u64 = 0;
const ProjectMetaKey = struct { rpb: c_int, ndir: c_int, nsc: c_int };
var project_meta: ?mlx.mlx_array = null;
var project_meta_key: ProjectMetaKey = undefined;
var project_engaged = false;

fn getProjectKernel() !mlx.mlx_fast_metal_kernel {
    if (project_kernel) |k| return k;
    const inputs = [_][*:0]const u8{ "x_in", "d_in", "sc_in", "meta" };
    const outputs = [_][*:0]const u8{"y_out"};
    const in_vec = mlx.mlx_vector_string_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&outputs, outputs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("mlxserve_steer_project", in_vec, out_vec, PROJECT_SOURCE, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    project_kernel = kernel;
    return kernel;
}

/// The cached config for this exact output shape, building and interning it on a miss.
/// Evicts the least recently used slot; the victim's config is freed only after the new
/// one is built, so a failed build leaves the cache exactly as it was.
fn projectCfgFor(key: ProjectCfgKey, xs: []const c_int, rows: c_int, hidden: c_int, dt: mlx.mlx_dtype) !mlx.mlx_fast_metal_kernel_config {
    project_cfg_clock += 1;
    for (&project_cfgs) |*slot| {
        if (slot.*) |*e| if (std.meta.eql(e.key, key)) {
            e.stamp = project_cfg_clock;
            return e.cfg;
        };
    }
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, xs.ptr, xs.len, dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 256, rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", hidden));
    var victim: usize = 0;
    var oldest: u64 = std.math.maxInt(u64);
    for (project_cfgs, 0..) |slot, i| {
        const st = if (slot) |e| e.stamp else 0;
        if (st < oldest) {
            oldest = st;
            victim = i;
        }
    }
    if (project_cfgs[victim]) |e| _ = mlx.mlx_fast_metal_kernel_config_free(e.cfg);
    project_cfgs[victim] = .{ .key = key, .cfg = cfg, .stamp = project_cfg_clock };
    return cfg;
}

/// The projection as ONE dispatch for every row of `x`. Null → the caller keeps
/// the chain. `d` is `[hidden]` or `[B,1,..,hidden]`, `scale` a scalar or `[B,1,..]`.
fn projectFused(x: mlx.mlx_array, d: mlx.mlx_array, scale: mlx.mlx_array, s: mlx.mlx_stream) !?mlx.mlx_array {
    if (!fusedEnabled() or !mlx.streamIsGpu(s)) return null;
    const dt = mlx.mlx_array_dtype(x);
    // f32 declined: the oracle stream keeps the chain it is compared against.
    if (dt != .bfloat16 and dt != .float16) return null;
    if (mlx.mlx_array_dtype(d) != dt or mlx.mlx_array_dtype(scale) != dt) return null;
    const xs = mlx.getShape(x);
    if (xs.len < 2) return null;
    const hidden = xs[xs.len - 1];
    const batch = xs[0];
    // A direction below 8 elements would bind in `constant`, which the kernel's pointer cannot take.
    if (hidden < 8 or batch < 1) return null;
    const n = mlx.mlx_array_size(x);
    if (n == 0 or n % @as(usize, @intCast(hidden)) != 0) return null;
    const rows: c_int = @intCast(n / @as(usize, @intCast(hidden)));
    if (@rem(rows, batch) != 0) return null;
    const ds = mlx.getShape(d);
    if (ds.len == 0 or ds[ds.len - 1] != hidden) return null;
    const ndir: c_int = @intCast(mlx.mlx_array_size(d) / @as(usize, @intCast(hidden)));
    if (ndir != 1 and (ndir != batch or ds.len != xs.len or ds[0] != batch)) return null;
    const nsc: c_int = @intCast(mlx.mlx_array_size(scale));
    const ss = mlx.getShape(scale);
    if (nsc != 1 and (nsc != batch or ss.len != xs.len or ss[0] != batch)) return null;
    if (ndir > 1) for (ds[1 .. ds.len - 1]) |v| if (v != 1) return null;
    if (nsc > 1) for (ss[1..]) |v| if (v != 1) return null;

    const mkey = ProjectMetaKey{ .rpb = @divExact(rows, batch), .ndir = ndir, .nsc = nsc };
    if (project_meta == null or !std.meta.eql(project_meta_key, mkey)) {
        if (project_meta) |m| _ = mlx.mlx_array_free(m);
        const vals = [_]i32{ mkey.rpb, mkey.ndir, mkey.nsc };
        const msh = [_]c_int{3};
        project_meta = mlx.mlx_array_new_data(&vals, &msh, 1, .int32);
        project_meta_key = mkey;
    }
    const cfg = try projectCfgFor(ProjectCfgKey.from(xs, dt), xs, rows, hidden, dt);
    // A 0-d input binds as a reference the kernel cannot index: hand the scale in 1-D.
    const sc_shape = [_]c_int{nsc};
    var sc1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc1);
    try mlx.check(mlx.mlx_reshape(&sc1, scale, &sc_shape, 1, s));
    const ins = [_]mlx.mlx_array{ x, d, sc1, project_meta.? };
    const in_vec = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var res = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(res);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&res, try getProjectKernel(), in_vec, cfg, s));
    if (mlx.mlx_vector_array_size(res) != 1) return error.MetalKernelBadOutputCount;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, res, 0));
    if (!project_engaged) {
        project_engaged = true;
        log.info("[steering] fused projection engaged: hidden={d} (MLX_SERVE_STEER_FUSED=0 restores the chain)\n", .{hidden});
    }
    return y;
}

/// The composed chain: five dependent dispatches, four roundings in `dt`.
fn projectChain(x: mlx.mlx_array, d: mlx.mlx_array, scale: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, x, d, s));
    var dot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dot);
    try mlx.check(mlx.mlx_sum_axis(&dot, prod, -1, true, s));
    var coeff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(coeff);
    try mlx.check(mlx.mlx_multiply(&coeff, dot, scale, s));
    var sub = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sub);
    try mlx.check(mlx.mlx_multiply(&sub, coeff, d, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&out, x, sub, s));
    return out;
}

/// The ffn arm: every hyper-connection branch of the flat `[B,S,hc*hidden]` stream.
pub fn projectStream(h: mlx.mlx_array, d: mlx.mlx_array, scale: mlx.mlx_array, hc: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(h);
    const shape4 = [_]c_int{ shape[0], shape[1], hc, @divExact(shape[2], hc) };
    var h4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h4);
    try mlx.check(mlx.mlx_reshape(&h4, h, &shape4, 4, s));
    const o4 = try projectLast(h4, d, scale, s);
    defer _ = mlx.mlx_array_free(o4);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, o4, shape.ptr, shape.len, s));
    return out;
}

/// The attn arm: the `[B,S,hidden]` o_proj output.
pub fn projectRow(y: mlx.mlx_array, d: mlx.mlx_array, scale: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    return projectLast(y, d, scale, s);
}

test "steering: launch flags resolve to a setting" {
    try std.testing.expect((try settingFromFlags(null, null, null)).state == .inherit);
    const only_file = (try settingFromFlags("terse", null, null)).cfg;
    try std.testing.expectEqualStrings("terse", only_file.name());
    try std.testing.expectEqual(@as(f32, 1), only_file.ffn);
    try std.testing.expectEqual(@as(f32, 0), only_file.attn);
    const both = (try settingFromFlags("/abs/x.f32", -1, 0.5)).cfg;
    try std.testing.expectEqual(@as(f32, -1), both.ffn);
    try std.testing.expectEqual(@as(f32, 0.5), both.attn);
    try std.testing.expectError(error.SteeringScaleWithoutFile, settingFromFlags(null, 1, null));
    try std.testing.expectError(error.SteeringScaleRange, settingFromFlags("x", 101, null));
    try std.testing.expectError(error.BadSteeringName, settingFromFlags("bad name", null, null));
}

test "steering: a capture id is a short safe path segment" {
    try std.testing.expect(isCaptureId("run1-good-07"));
    try std.testing.expect(isCaptureId("a.b_c"));
    try std.testing.expect(!isCaptureId(""));
    try std.testing.expect(!isCaptureId("../x"));
    try std.testing.expect(!isCaptureId("a/b"));
    try std.testing.expect(!isCaptureId("sp ace"));
    const long: [65]u8 = @splat('a');
    try std.testing.expect(!isCaptureId(&long));
}

test "steering: scale defaults and range" {
    const both = resolveScales(null, null);
    try std.testing.expectEqual(@as(f32, 1.0), both.ffn);
    try std.testing.expectEqual(@as(f32, 0.0), both.attn);
    const attn_only = resolveScales(null, 0.5);
    try std.testing.expectEqual(@as(f32, 0.0), attn_only.ffn);
    try std.testing.expectEqual(@as(f32, 0.5), attn_only.attn);
    try validateScale(100);
    try validateScale(-100);
    try validateScale(0);
    try std.testing.expectError(error.SteeringScaleRange, validateScale(std.math.nan(f32)));
    try std.testing.expectError(error.SteeringScaleRange, validateScale(101));
    try std.testing.expectError(error.SteeringScaleRange, validateScale(-101));
    try std.testing.expect((Req{ .bank = 0, .ffn = 0, .attn = 0 }).isOff());
    try std.testing.expect(!(Req{ .bank = 0, .ffn = 0, .attn = 0.1 }).isOff());
}

test "steering: names resolve into the registry dir, paths pass through, junk is refused" {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const home = std.mem.span(std.c.getenv("HOME") orelse "/tmp");
    const named = try resolvePath(&buf, "terse-v2.1");
    var want_buf: [std.fs.max_path_bytes]u8 = undefined;
    const want = try std.fmt.bufPrint(&want_buf, "{s}/.mlx-serve/steering/terse-v2.1.f32", .{home});
    try std.testing.expectEqualStrings(want, named);
    try std.testing.expectEqualStrings("/abs/dir/x.f32", try resolvePath(&buf, "/abs/dir/x.f32"));
    try std.testing.expectError(error.BadSteeringName, resolvePath(&buf, "foo/bar"));
    try std.testing.expectError(error.BadSteeringName, resolvePath(&buf, "../x"));
    try std.testing.expectError(error.BadSteeringName, resolvePath(&buf, ""));
    try std.testing.expectError(error.BadSteeringName, resolvePath(&buf, "sp ace"));
    const long: [MAX_NAME + 1]u8 = @splat('a');
    try std.testing.expectError(error.BadSteeringName, Configured.init(&long, 1, 0));
    const c = try Configured.init("terse", 1, 0);
    try std.testing.expectEqualStrings("terse", c.name());
}

test "steering: a bank file is proven on our side before mlx sees it" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try tmp.dir.realPath(io, &pbuf);
    const root = pbuf[0..root_len];
    const n_layers: u32 = 3;
    const hidden: u32 = 8;
    var rows: [n_layers * hidden]f32 = undefined;
    for (&rows, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.25;
    try tmp.dir.writeFile(io, .{ .sub_path = "good.f32", .data = std.mem.sliceAsBytes(&rows) });
    try tmp.dir.writeFile(io, .{ .sub_path = "short.f32", .data = std.mem.sliceAsBytes(rows[0 .. rows.len - 1]) });
    const good = try std.fmt.allocPrint(a, "{s}/good.f32", .{root});
    defer a.free(good);
    const short = try std.fmt.allocPrint(a, "{s}/short.f32", .{root});
    defer a.free(short);

    try std.testing.expectError(error.BadSteeringPath, validateFile(io, "relative.f32", n_layers, hidden));
    try std.testing.expectError(error.BadSteeringPath, validateFile(io, root, n_layers, hidden)); // a directory
    // A FIFO is refused without being opened.
    const fifo = try std.fmt.allocPrintSentinel(a, "{s}/pipe.f32", .{root}, 0);
    defer a.free(fifo);
    try std.testing.expectEqual(@as(c_int, 0), struct {
        extern "c" fn mkfifo(path: [*:0]const u8, mode: std.c.mode_t) c_int;
    }.mkfifo(fifo, 0o600));
    try std.testing.expectError(error.BadSteeringPath, validateFile(io, fifo, n_layers, hidden));
    try std.testing.expectError(error.SteeringFileSize, validateFile(io, short, n_layers, hidden));
    const id = try validateFile(io, good, n_layers, hidden);
    try std.testing.expectEqual(@as(u64, n_layers * hidden * 4), id.size);

    const host = try readBank(a, io, good, n_layers, hidden);
    defer a.free(host);
    try std.testing.expectEqual(@as(f32, @floatFromInt(2 * hidden)) * 0.25, host[2 * hidden]);
    try std.testing.expectEqual(@as(f32, @floatFromInt(3 * hidden - 1)) * 0.25, host[host.len - 1]);
}

test "steering: Setting round-trips its three states through JSON" {
    const a = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, a,
        \\{"off": null, "cfg": {"name": "terse", "ffn": -1.5, "attn": 0.25}, "bare": {"name": "x"}, "junk": {"ffn": 1}, "junk2": 7, "badname": {"name": "a b"}, "relname": {"name": "rel/x.f32"}}
    , .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expect(Setting.fromJsonValue(root.get("off").?).?.state == .off);
    const cfg = Setting.fromJsonValue(root.get("cfg").?).?.cfg;
    try std.testing.expectEqualStrings("terse", cfg.name());
    try std.testing.expectEqual(@as(f32, -1.5), cfg.ffn);
    try std.testing.expectEqual(@as(f32, 0.25), cfg.attn);
    const bare = Setting.fromJsonValue(root.get("bare").?).?.cfg;
    try std.testing.expectEqual(@as(f32, 1.0), bare.ffn); // file alone = ffn 1
    try std.testing.expectEqual(@as(f32, 0.0), bare.attn);
    try std.testing.expect(Setting.fromJsonValue(root.get("junk").?) == null); // no name = malformed
    try std.testing.expect(Setting.fromJsonValue(root.get("junk2").?) == null);
    // A name no load could resolve is malformed too, so the launch flags apply.
    try std.testing.expect(Setting.fromJsonValue(root.get("badname").?) == null);
    try std.testing.expect(Setting.fromJsonValue(root.get("relname").?) == null);
}

// ---- bank registry ----

fn writeTestBank(io: std.Io, dir: std.Io.Dir, name: []const u8, n_layers: u32, hidden: u32, seed: f32) !void {
    var rows: [8 * 16]f32 = undefined;
    const n = n_layers * hidden;
    for (rows[0..n], 0..) |*v, i| v.* = seed + @as(f32, @floatFromInt(i));
    try dir.writeFile(io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(rows[0..n]) });
}

const TestBanks = struct {
    tmp: std.testing.TmpDir,
    root: [std.fs.max_path_bytes]u8 = undefined,
    root_len: usize = 0,
    banks: Banks,
    paths: [MAX_BANKS + 2][]u8 = undefined,
    n_paths: usize = 0,

    fn init(n_files: usize) !*TestBanks {
        const a = std.testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const t = try a.create(TestBanks);
        t.* = .{ .tmp = std.testing.tmpDir(.{}), .banks = Banks.init(a, io, 4, 16) };
        t.root_len = try t.tmp.dir.realPath(io, &t.root);
        for (0..n_files) |i| {
            var name_buf: [16]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "b{d}.f32", .{i});
            try writeTestBank(io, t.tmp.dir, name, 4, 16, @floatFromInt(i * 100));
            t.paths[i] = try std.fmt.allocPrint(a, "{s}/{s}", .{ t.root[0..t.root_len], name });
            t.n_paths += 1;
        }
        return t;
    }

    fn deinit(t: *TestBanks) void {
        const a = std.testing.allocator;
        t.banks.deinit();
        for (t.paths[0..t.n_paths]) |p| a.free(p);
        t.tmp.cleanup();
        a.destroy(t);
    }
};

test "steering: a request's steering field parses to a spec" {
    const a = std.testing.allocator;
    var parsed = try std.json.parseFromSlice(std.json.Value, a,
        \\{"off": null, "named": {"name": "terse", "ffn": -1}, "file": {"file": "/abs/x.f32", "attn": 0.5}, "scales": {"ffn": 0, "attn": 0}, "badname": {"name": 7}, "badscale": {"ffn": "x"}, "both": {"name": "a", "file": "/b.f32"}, "list": [1]}
    , .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const off = try Spec.fromJsonValue(root.get("off").?);
    try std.testing.expect(off.off and off.name == null);
    const named = try Spec.fromJsonValue(root.get("named").?);
    try std.testing.expectEqualStrings("terse", named.name.?);
    try std.testing.expectEqual(@as(?f32, -1), named.ffn);
    try std.testing.expectEqual(@as(?f32, null), named.attn);
    const file = try Spec.fromJsonValue(root.get("file").?);
    try std.testing.expectEqualStrings("/abs/x.f32", file.name.?);
    try std.testing.expectEqual(@as(?f32, 0.5), file.attn);
    const scales = try Spec.fromJsonValue(root.get("scales").?);
    try std.testing.expect(scales.name == null and !scales.off);
    try std.testing.expectEqual(@as(?f32, 0), scales.ffn);
    try std.testing.expectError(error.BadSteeringRequest, Spec.fromJsonValue(root.get("badname").?));
    try std.testing.expectError(error.SteeringScaleRange, Spec.fromJsonValue(root.get("badscale").?));
    try std.testing.expectError(error.BadSteeringRequest, Spec.fromJsonValue(root.get("list").?));
    try std.testing.expectError(error.BadSteeringRequest, Spec.fromJsonValue(root.get("both").?));
}

test "steering: a spec without a name rides the default's bank with its own scales" {
    const t = try TestBanks.init(1);
    defer t.deinit();
    // No default: scales alone steer nothing, a name reserves.
    var none = try t.banks.reserveSpec(.{ .ffn = 1 });
    try std.testing.expect(none.req.isOff() and !none.armed);
    none.release();
    var d = try t.banks.reserve(t.paths[0], 0.5, 0);
    const bank = d.req.bank;
    t.banks.swapDefault(&d);
    var inherit = try t.banks.reserveSpec(.{});
    try std.testing.expectEqual(bank, inherit.req.bank);
    try std.testing.expectEqual(@as(f32, 0.5), inherit.req.ffn);
    try std.testing.expectEqual(@as(u32, 2), t.banks.refsOf(bank));
    inherit.release();
    var attn_only = try t.banks.reserveSpec(.{ .attn = 0.25 });
    try std.testing.expectEqual(@as(f32, 0.5), attn_only.req.ffn); // the default's ffn survives
    try std.testing.expectEqual(@as(f32, 0.25), attn_only.req.attn);
    attn_only.release();
    var opt_out = try t.banks.reserveSpec(.{ .ffn = 0, .attn = 0 });
    try std.testing.expect(opt_out.req.isOff() and !opt_out.armed);
    try std.testing.expectEqual(@as(u32, 1), t.banks.refsOf(bank));
    opt_out.release();
    var explicit_off = try t.banks.reserveSpec(.{ .off = true });
    try std.testing.expect(explicit_off.req.isOff());
    explicit_off.release();
    var none0 = Reservation.none(&t.banks);
    t.banks.swapDefault(&none0);
}

test "steering: reserve dedupes by file, refs count guards, release restores" {
    const t = try TestBanks.init(2);
    defer t.deinit();
    var r1 = try t.banks.reserve(t.paths[0], 0.5, 0);
    var r2 = try t.banks.reserve(t.paths[0], 1.0, 0);
    try std.testing.expectEqual(r1.req.bank, r2.req.bank);
    try std.testing.expectEqual(@as(u32, 2), t.banks.refsOf(r1.req.bank));
    try std.testing.expectEqual(@as(f32, 0.5), r1.req.ffn);
    try std.testing.expectEqual(@as(f32, 1.0), r2.req.ffn);
    r1.release();
    r1.release(); // idempotent on the guard side
    try std.testing.expectEqual(@as(u32, 1), t.banks.refsOf(r2.req.bank));
    // Disarmed = ownership moved to a slot; the slot-side release brings it back.
    const idx = r2.req.bank;
    r2.disarm();
    r2.release();
    try std.testing.expectEqual(@as(u32, 1), t.banks.refsOf(idx));
    t.banks.releaseBank(idx);
    try std.testing.expectEqual(@as(u32, 0), t.banks.refsOf(idx));
    // Both scales zero = no bank pinned at all.
    var off = try t.banks.reserve(t.paths[1], 0, 0);
    defer off.release();
    try std.testing.expect(off.req.isOff());
    var occupied: usize = 0;
    for (t.banks.slots) |sl| occupied += @intFromBool(sl != null);
    try std.testing.expectEqual(@as(usize, 1), occupied);
}

test "steering: a full registry refuses by name, evicts LRU once something is unpinned" {
    const t = try TestBanks.init(MAX_BANKS + 1);
    defer t.deinit();
    var held: [MAX_BANKS]Reservation = undefined;
    for (0..MAX_BANKS) |i| held[i] = try t.banks.reserve(t.paths[i], 1, 0);
    try std.testing.expectError(error.SteeringBanksBusy, t.banks.reserve(t.paths[MAX_BANKS], 1, 0));
    // Two unpinned: 5 is the least recently used and sits at the HIGHER index, so an
    // index-order or MRU pick fails.
    const victim = held[5].req.bank;
    held[2].release();
    held[5].release();
    var touch = try t.banks.reserve(t.paths[2], 1, 0);
    touch.release();
    var extra = try t.banks.reserve(t.paths[MAX_BANKS], 1, 0);
    defer extra.release();
    try std.testing.expectEqual(victim, extra.req.bank);
    try std.testing.expectEqualStrings(t.paths[MAX_BANKS], t.banks.pathOf(extra.req.bank));
    try std.testing.expectEqualStrings(t.paths[2], t.banks.pathOf(held[2].req.bank));
    for (&held, 0..) |*h, i| if (i != 2 and i != 5) h.release();
}

test "steering: a snapshot of the default survives a swap and an eviction pass" {
    const t = try TestBanks.init(MAX_BANKS + 1);
    defer t.deinit();
    try std.testing.expect(t.banks.snapshotDefault().req.isOff());
    var a_res = try t.banks.reserve(t.paths[0], 1, 0);
    const a_idx = a_res.req.bank;
    t.banks.swapDefault(&a_res);
    try std.testing.expectEqual(@as(u32, 1), t.banks.refsOf(a_idx));
    var snap = t.banks.snapshotDefault();
    try std.testing.expectEqual(a_idx, snap.req.bank);
    try std.testing.expectEqual(@as(u32, 2), t.banks.refsOf(a_idx));
    var b_res = try t.banks.reserve(t.paths[1], 1, 0);
    const b_idx = b_res.req.bank;
    t.banks.swapDefault(&b_res);
    try std.testing.expectEqual(@as(u32, 1), t.banks.refsOf(a_idx)); // the snapshot's pin
    try std.testing.expectEqual(@as(u32, 1), t.banks.refsOf(b_idx));
    // Fill every other slot and ask for one more: A is pinned, so the pass may not touch it.
    var held: [MAX_BANKS]Reservation = undefined;
    var n_held: usize = 0;
    for (2..MAX_BANKS + 1) |i| {
        held[n_held] = t.banks.reserve(t.paths[i], 1, 0) catch |e| {
            try std.testing.expectEqual(error.SteeringBanksBusy, e);
            break;
        };
        n_held += 1;
    }
    try std.testing.expectEqual(MAX_BANKS - 2, n_held); // every slot is pinned now
    // Unpin one: the next reserve must evict exactly that slot, never the snapshot's.
    const victim = held[0].req.bank;
    held[0].release();
    held[0] = try t.banks.reserve(t.paths[MAX_BANKS], 1, 0);
    try std.testing.expectEqual(victim, held[0].req.bank);
    try std.testing.expectEqualStrings(t.paths[0], t.banks.pathOf(a_idx));
    try std.testing.expectEqual(@as(u32, 1), t.banks.refsOf(a_idx));
    snap.release();
    try std.testing.expectEqual(@as(u32, 0), t.banks.refsOf(a_idx));
    for (held[0..n_held]) |*h| h.release();
    var noneb = Reservation.none(&t.banks);
    t.banks.swapDefault(&noneb);
    try std.testing.expectEqual(@as(u32, 0), t.banks.refsOf(b_idx));
}

fn hammer(banks: *Banks, paths: []const []u8, iters: usize, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (0..iters) |_| {
        const p = paths[rnd.uintLessThan(usize, paths.len)];
        switch (rnd.uintLessThan(u8, 3)) {
            0 => {
                var r = banks.reserve(p, 1, 0) catch continue;
                r.release();
            },
            1 => {
                var r = banks.reserve(p, 1, 0) catch continue;
                banks.swapDefault(&r);
            },
            else => {
                var s = banks.snapshotDefault();
                s.release();
            },
        }
    }
}

test "steering: concurrent snapshot/swap/reserve leave refs equal to the live guards" {
    const t = try TestBanks.init(4);
    defer t.deinit();
    const paths = t.paths[0..t.n_paths];
    const t1 = try std.Thread.spawn(.{}, hammer, .{ &t.banks, paths, 5000, 1 });
    const t2 = try std.Thread.spawn(.{}, hammer, .{ &t.banks, paths, 5000, 2 });
    t1.join();
    t2.join();
    var noneb = Reservation.none(&t.banks);
    t.banks.swapDefault(&noneb);
    for (0..MAX_BANKS) |i| try std.testing.expectEqual(@as(u32, 0), t.banks.refsOf(@intCast(i)));
}

// ---- projection ----

fn testDot(x: mlx.mlx_array, d: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var prod = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(prod);
    try mlx.check(mlx.mlx_multiply(&prod, x, d, s));
    var p32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p32);
    try mlx.check(mlx.mlx_astype(&p32, prod, .float32, s));
    var sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum);
    try mlx.check(mlx.mlx_sum_axis(&sum, p32, -1, false, s));
    try mlx.check(mlx.mlx_array_eval(sum));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, sum));
    return v;
}

fn testEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var v: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&v, eq));
    return v;
}

fn testRand(rnd: std.Random, shape: []const c_int, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    for (data) |*x| x.* = rnd.float(f32) - 0.5;
    return hostAs(data.ptr, shape, dt, s);
}

fn fillUnit(rnd: std.Random, row: []f32) void {
    var n2: f64 = 0;
    for (row) |*x| {
        x.* = rnd.float(f32) - 0.5;
        n2 += @as(f64, x.*) * x.*;
    }
    const inv: f32 = @floatCast(1.0 / @sqrt(n2));
    for (row) |*x| x.* *= inv;
}

/// A random unit-norm direction, f32 on the host so the bar is exact.
fn testUnitDir(rnd: std.Random, hidden: usize, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const data = try std.testing.allocator.alloc(f32, hidden);
    defer std.testing.allocator.free(data);
    fillUnit(rnd, data);
    return hostAs(data.ptr, &.{@intCast(hidden)}, dt, s);
}

test "steering: projection is identity at scale 0, kills the direction at scale 1, keeps dtype and shape" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const hc: c_int = 4;
    const hidden: usize = 128;
    inline for (.{ mlx.mlx_dtype.bfloat16, mlx.mlx_dtype.float32 }) |dt| {
        const h = try testRand(rnd, &.{ 1, 3, hc * @as(c_int, hidden) }, dt, s);
        defer _ = mlx.mlx_array_free(h);
        const d = try testUnitDir(rnd, hidden, dt, s);
        defer _ = mlx.mlx_array_free(d);
        const zero = try scalarAs(0, dt, s);
        defer _ = mlx.mlx_array_free(zero);
        const one = try scalarAs(1, dt, s);
        defer _ = mlx.mlx_array_free(one);

        const same = try projectStream(h, d, zero, hc, s);
        defer _ = mlx.mlx_array_free(same);
        try std.testing.expect(try testEqual(same, h, s));
        try std.testing.expectEqual(dt, mlx.mlx_array_dtype(same));

        const out = try projectStream(h, d, one, hc, s);
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expectEqual(dt, mlx.mlx_array_dtype(out));
        try std.testing.expectEqualSlices(c_int, mlx.getShape(h), mlx.getShape(out));
        // Per branch: the component along d is gone, up to the rounding of `coeff`.
        const shape4 = [_]c_int{ 1, 3, hc, @intCast(hidden) };
        var h4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h4);
        try mlx.check(mlx.mlx_reshape(&h4, h, &shape4, 4, s));
        var o4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o4);
        try mlx.check(mlx.mlx_reshape(&o4, out, &shape4, 4, s));
        for (0..3) |t| for (0..@as(usize, hc)) |b| {
            const start = [_]c_int{ 0, @intCast(t), @intCast(b), 0 };
            const stop = [_]c_int{ 1, @intCast(t + 1), @intCast(b + 1), @intCast(hidden) };
            const strides = [_]c_int{ 1, 1, 1, 1 };
            var hin = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(hin);
            try mlx.check(mlx.mlx_slice(&hin, h4, &start, 4, &stop, 4, &strides, 4, s));
            var hout = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(hout);
            try mlx.check(mlx.mlx_slice(&hout, o4, &start, 4, &stop, 4, &strides, 4, s));
            const before = try testDot(hin, d, s);
            const after = try testDot(hout, d, s);
            // bf16 rounds every output element (2^-8 relative), so what is
            // left along d is bounded by |x| * 2^-7, not by `before`.
            const bar: f32 = if (dt == .float32) 1e-5 else @sqrt(try testDot(hin, hin, s)) / 128.0;
            try std.testing.expect(@abs(after) < @abs(before) or @abs(before) < bar);
            try std.testing.expect(@abs(after) <= bar);
        };
    }
}

test "steering: mixed rows gather their own direction and scale, an off row is byte-identical" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    const hc: c_int = 4;
    const hidden: usize = 64;
    const dt: mlx.mlx_dtype = .bfloat16;
    const h = try testRand(rnd, &.{ 2, 1, hc * @as(c_int, hidden) }, dt, s);
    defer _ = mlx.mlx_array_free(h);
    const d = try testUnitDir(rnd, hidden, dt, s);
    defer _ = mlx.mlx_array_free(d);
    // Row 0 steered at 1, row 1 off: the same d for both rows, broadcast as [B,1,1,hidden].
    const dshape = [_]c_int{ 1, 1, 1, @intCast(hidden) };
    var d1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d1);
    try mlx.check(mlx.mlx_reshape(&d1, d, &dshape, 4, s));
    const reps = [_]c_int{ 2, 1, 1, 1 };
    var d_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d_rows);
    try mlx.check(mlx.mlx_tile(&d_rows, d1, &reps, 4, s));
    const scales = try columnAs(&.{ 1.0, 0.0 }, 4, dt, s);
    defer _ = mlx.mlx_array_free(scales);
    const out = try projectStream(h, d_rows, scales, hc, s);
    defer _ = mlx.mlx_array_free(out);

    const one = try scalarAs(1, dt, s);
    defer _ = mlx.mlx_array_free(one);
    const uniform = try projectStream(h, d, one, hc, s);
    defer _ = mlx.mlx_array_free(uniform);
    const strides = [_]c_int{ 1, 1, 1 };
    inline for (.{ 0, 1 }) |row| {
        const start = [_]c_int{ row, 0, 0 };
        const stop = [_]c_int{ row + 1, 1, hc * @as(c_int, hidden) };
        var got = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_slice(&got, out, &start, 3, &stop, 3, &strides, 3, s));
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_slice(&want, if (row == 0) uniform else h, &start, 3, &stop, 3, &strides, 3, s));
        try std.testing.expect(try testEqual(got, want, s));
    }
    // The attn arm's row form: [B,S,hidden] with a [hidden] direction.
    const y = try testRand(rnd, &.{ 1, 3, @as(c_int, hidden) }, dt, s);
    defer _ = mlx.mlx_array_free(y);
    const y_out = try projectRow(y, d, one, s);
    defer _ = mlx.mlx_array_free(y_out);
    try std.testing.expectEqualSlices(c_int, mlx.getShape(y), mlx.getShape(y_out));
    const zero = try scalarAs(0, dt, s);
    defer _ = mlx.mlx_array_free(zero);
    const y_same = try projectRow(y, d, zero, s);
    defer _ = mlx.mlx_array_free(y_same);
    try std.testing.expect(try testEqual(y_same, y, s));
}

fn testHostF32(arr: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    const data = mlx.mlx_array_data_float32(c) orelse return error.Unreadable;
    const out = try std.testing.allocator.alloc(f32, n);
    @memcpy(out, data[0..n]);
    return out;
}

test "steering: the fused projection is no farther from fp32 truth than the chain, exact at scale 0, kill switch restores the chain" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    const allocator = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    defer fused_override = null;
    const Case = struct { shape: []const c_int, mixed: bool };
    // The serving shapes: the ffn arm's decode stream, the attn arm's row, a verify
    // width, and a mixed batched group of each form (per-row direction and scale).
    const cases = [_]Case{
        .{ .shape = &.{ 1, 1, 4, 2560 }, .mixed = false },
        .{ .shape = &.{ 1, 1, 2560 }, .mixed = false },
        .{ .shape = &.{ 1, 5, 4, 2560 }, .mixed = false },
        .{ .shape = &.{ 3, 2, 4, 64 }, .mixed = true },
        .{ .shape = &.{ 3, 1, 64 }, .mixed = true },
    };
    inline for (.{ mlx.mlx_dtype.bfloat16, mlx.mlx_dtype.float32 }) |dt| {
        for (cases) |case| {
            const xs = case.shape;
            const hidden: usize = @intCast(xs[xs.len - 1]);
            const batch: usize = @intCast(xs[0]);
            var n: usize = 1;
            for (xs) |v| n *= @intCast(v);
            const rows = n / hidden;
            const rpb = rows / batch;
            const ndir: usize = if (case.mixed) batch else 1;
            const nsc: usize = if (case.mixed) batch else 1;

            // Unit directions on the host, then x = noise + a per-row component ALONG the
            // direction, so every row's edit is well above an output ulp.
            const dhost = try allocator.alloc(f32, ndir * hidden);
            defer allocator.free(dhost);
            for (0..ndir) |r| fillUnit(rnd, dhost[r * hidden .. (r + 1) * hidden]);
            const xhost = try allocator.alloc(f32, n);
            defer allocator.free(xhost);
            for (0..rows) |row| {
                const dr = dhost[(if (ndir > 1) row / rpb else 0) * hidden ..][0..hidden];
                const along: f32 = if (row % 2 == 0) 0.6 else -0.45;
                for (0..hidden) |k| xhost[row * hidden + k] = rnd.float(f32) - 0.5 + along * dr[k];
            }
            const x = try hostAs(xhost.ptr, xs, dt, s);
            defer _ = mlx.mlx_array_free(x);
            var dshape: [4]c_int = .{ @intCast(ndir), 1, 1, 1 };
            const dnd: usize = if (ndir == 1) 1 else xs.len;
            dshape[dnd - 1] = @intCast(hidden);
            const d = try hostAs(dhost.ptr, dshape[0..dnd], dt, s);
            defer _ = mlx.mlx_array_free(d);
            const scale = if (case.mixed) try columnAs(&.{ 1.0, -0.5, 0.0 }, xs.len, dt, s) else try scalarAs(0.75, dt, s);
            defer _ = mlx.mlx_array_free(scale);

            fused_override = false;
            const chain = try projectLast(x, d, scale, s);
            defer _ = mlx.mlx_array_free(chain);
            fused_override = true;
            const via = try projectLast(x, d, scale, s);
            defer _ = mlx.mlx_array_free(via);
            const fused_opt = try projectFused(x, d, scale, s);
            defer if (fused_opt) |f| {
                _ = mlx.mlx_array_free(f);
            };
            // f32 is declined: the chain is the oracle there.
            if (dt == .float32) {
                try std.testing.expect(fused_opt == null);
                try std.testing.expect(try testEqual(via, chain, s));
                continue;
            }
            const fused = fused_opt orelse return error.FusedProjectionDeclined;
            try std.testing.expectEqual(dt, mlx.mlx_array_dtype(fused));
            try std.testing.expectEqualSlices(c_int, xs, mlx.getShape(fused));
            try std.testing.expect(try testEqual(via, fused, s));

            // Truth in f64 from the dt-rounded inputs; the bar is squared error against it,
            // never a chain-vs-kernel diff.
            const xh = try testHostF32(x, s);
            defer allocator.free(xh);
            const dh = try testHostF32(d, s);
            defer allocator.free(dh);
            const sh = try testHostF32(scale, s);
            defer allocator.free(sh);
            const ch = try testHostF32(chain, s);
            defer allocator.free(ch);
            const fh = try testHostF32(fused, s);
            defer allocator.free(fh);
            var se_chain: f64 = 0;
            var se_fused: f64 = 0;
            var edit: f64 = 0;
            for (0..rows) |row| {
                const b = row / rpb;
                const xr = xh[row * hidden .. (row + 1) * hidden];
                const dr = dh[(if (ndir > 1) b else 0) * hidden ..][0..hidden];
                const sc: f64 = sh[if (nsc > 1) b else 0];
                var dot: f64 = 0;
                for (xr, dr) |xv, dv| dot += @as(f64, xv) * dv;
                for (0..hidden) |k| {
                    const t = @as(f64, xr[k]) - sc * dr[k] * dot;
                    const i = row * hidden + k;
                    se_chain += (ch[i] - t) * (ch[i] - t);
                    se_fused += (fh[i] - t) * (fh[i] - t);
                    edit += (xr[k] - t) * (xr[k] - t);
                    if (sc == 0) try std.testing.expectEqual(xr[k], fh[i]);
                }
            }
            try std.testing.expect(std.math.isFinite(se_fused));
            try std.testing.expect(se_fused <= se_chain);
            // The edit itself is far above what either arm loses to rounding.
            try std.testing.expect(edit > 100.0 * se_chain);

            // Scale 0 is the identity to the byte.
            const zero = try scalarAs(0, dt, s);
            defer _ = mlx.mlx_array_free(zero);
            const same = (try projectFused(x, d, zero, s)) orelse return error.FusedProjectionDeclined;
            defer _ = mlx.mlx_array_free(same);
            try std.testing.expect(try testEqual(same, x, s));
        }
    }
}
