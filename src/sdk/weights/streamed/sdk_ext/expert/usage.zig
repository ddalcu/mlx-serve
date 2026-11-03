//! The expert usage profile and the resident seed. A server run with `expert_usage_record: "<path>"` counts, per stream
//! layer, the experts each request routed (`Stream.addUsage`: the prompt pass's routed ids from the policy's prompt counts
//! and the decode routes from its request counts; no counter of its own on any route) and writes them at each request's
//! end to `<path>` (`Recorder`). The file, `expert-usage-v1.json`, goes next to the pack. At the load, `resident_seed`
//! (`usage` | `none`; default `usage` when the pack has the file, `none` when not) reads it once and fills each layer's
//! empty prompt rows with the layer's most used experts (`Stream.seedResidents`), so the first request finds them
//! resident. The seed changes no policy: the seeded experts are residents as a kept resident is (`shrink(.keep)`).
//! `prompt_residents` (`PromptResidents`) is the reverse change's route the arches take by the same name.

const std = @import("std");
const log = @import("sdk").log;

pub const file_name = "expert-usage-v1.json";
/// The file's largest size the load reads (a 64 x 512 profile of u64 counts is ~0.4 MB).
pub const max_file_bytes: usize = 16 << 20;
pub const max_path: usize = 1024;
pub const max_label: usize = 256;

/// The residents a prompt after a decode finds (`prompt_residents`), installed once at construction: `keep` (the
/// default: the residents decode left in the prompt rows stay for the next prompt) or `forget` (the control, by name).
pub const PromptResidents = enum { keep, forget };

/// `prompt_residents`'s install line for `arch`, printed once at construction (the words of glm5_next's).
pub fn promptResidentsLine(comptime arch: []const u8, residents: PromptResidents) []const u8 {
    return switch (residents) {
        .keep => arch ++ " prompt residents: keep (decode's residents stay for the next prompt)",
        .forget => arch ++ " prompt residents: forget (every request starts from empty rows)",
    };
}

/// `resident_seed`: `usage` (the prompt rows filled from the pack's `expert-usage-v1.json` at the load) or `none`.
pub const Seed = enum { usage, none };

/// The model-settings keys `Settings.apply` reads.
pub const keys = [_][]const u8{ "resident_seed", "expert_usage_record", "expert_usage_label" };

/// `resident_seed`, `expert_usage_record` (an absolute path; default off) and `expert_usage_label` (the prompt set's
/// name the file's source carries). A bad value is refused at the load by name (`check`).
pub const Settings = struct {
    /// Null: by the pack (`usage` when it has the file, `none` when not).
    seed: ?Seed = null,
    seed_bad: bool = false,
    record_buf: [max_path]u8 = undefined,
    record_len: usize = 0,
    record_bad: bool = false,
    label_buf: [max_label]u8 = undefined,
    label_len: usize = 0,
    label_bad: bool = false,
    /// The keys were named (whatever their values): an arch without the record refuses either, and a label refuses
    /// without a record.
    record_named: bool = false,
    label_named: bool = false,

    /// The keys of one settings object; returns whether any was present.
    pub fn apply(s: *Settings, obj: std.json.ObjectMap) bool {
        var any = false;
        if (obj.get("resident_seed")) |v| {
            any = true;
            const x: ?Seed = if (v == .string) std.meta.stringToEnum(Seed, v.string) else null;
            if (x) |seed| s.seed = seed else s.seed_bad = true;
        }
        if (obj.get("expert_usage_record")) |v| {
            any = true;
            s.record_named = true;
            if (v == .string and v.string.len > 0 and v.string.len <= max_path and std.fs.path.isAbsolute(v.string)) {
                @memcpy(s.record_buf[0..v.string.len], v.string);
                s.record_len = v.string.len;
            } else s.record_bad = true;
        }
        if (obj.get("expert_usage_label")) |v| {
            any = true;
            s.label_named = true;
            if (v == .string and v.string.len > 0 and v.string.len <= max_label) {
                @memcpy(s.label_buf[0..v.string.len], v.string);
                s.label_len = v.string.len;
            } else s.label_bad = true;
        }
        return any;
    }

    /// The load's refusal of a bad value, by name (`arch` prefixes the line).
    pub fn check(s: *const Settings, comptime arch: []const u8) error{ ResidentSeedSetting, ExpertUsageRecordSetting, ExpertUsageLabelSetting, ExpertUsageLabelWithoutRecord }!void {
        if (s.seed_bad) {
            log.warn(arch ++ ": load refused: resident_seed names none of usage, none (ResidentSeedSetting)\n", .{});
            return error.ResidentSeedSetting;
        }
        if (s.record_bad) {
            log.warn(arch ++ ": load refused: expert_usage_record is not an absolute file path of at most {d} bytes (ExpertUsageRecordSetting)\n", .{max_path});
            return error.ExpertUsageRecordSetting;
        }
        if (s.label_bad) {
            log.warn(arch ++ ": load refused: expert_usage_label is not a string of 1 to {d} bytes (ExpertUsageLabelSetting)\n", .{max_label});
            return error.ExpertUsageLabelSetting;
        }
        if (s.label_named and !s.record_named) {
            log.warn(arch ++ ": load refused: expert_usage_label names the prompt set of a record, and expert_usage_record is not set (ExpertUsageLabelWithoutRecord)\n", .{});
            return error.ExpertUsageLabelWithoutRecord;
        }
    }

    /// `check` for an arch that installs the seed but not the record (MiMo-V2.6, DeepSeek-V4.1): a named
    /// `expert_usage_record` or `expert_usage_label` is refused by name, never ignored.
    pub fn checkSeedOnly(s: *const Settings, comptime arch: []const u8) error{ ResidentSeedSetting, ExpertUsageRecordNotInstalled }!void {
        if (s.seed_bad) {
            log.warn(arch ++ ": load refused: resident_seed names none of usage, none (ResidentSeedSetting)\n", .{});
            return error.ResidentSeedSetting;
        }
        if (s.record_named or s.label_named) {
            log.warn(arch ++ ": load refused: expert_usage_record / expert_usage_label are installed on glm5_next only (ExpertUsageRecordNotInstalled)\n", .{});
            return error.ExpertUsageRecordNotInstalled;
        }
    }

    pub fn recordPath(s: *const Settings) ?[]const u8 {
        return if (s.record_len > 0) s.record_buf[0..s.record_len] else null;
    }

    pub fn label(s: *const Settings) []const u8 {
        return if (s.label_len > 0) s.label_buf[0..s.label_len] else "unlabeled";
    }
};

/// The file's JSON form (the form the recorder writes).
pub const Doc = struct {
    model_type: []const u8,
    layers: u32,
    experts: u32,
    counts: []const []const u64,
    source: Source,

    /// The totals over every session, and each session (one server load that recorded into the file), oldest first.
    pub const Source = struct {
        requests: u64,
        prompt_tokens: u64,
        decode_tokens: u64,
        sessions: []const Session,
    };

    pub const Session = struct {
        prompt_set: []const u8,
        binary_sha256: []const u8,
        requests: u64 = 0,
        prompt_tokens: u64 = 0,
        decode_tokens: u64 = 0,
    };

    /// The checks every reader runs: the v1 form (`ExpertUsageMalformed`), then the model and geometry
    /// (`ExpertUsageMismatch`).
    pub fn check(d: *const Doc, model_type: []const u8, layers: u32, experts: u32) error{ ExpertUsageMalformed, ExpertUsageMismatch }!void {
        if (d.counts.len != d.layers or d.experts == 0 or d.experts > std.math.maxInt(u16) or d.source.sessions.len == 0) return error.ExpertUsageMalformed;
        for (d.counts) |row| if (row.len != d.experts) return error.ExpertUsageMalformed;
        if (!std.mem.eql(u8, d.model_type, model_type) or d.layers != layers or d.experts != experts) return error.ExpertUsageMismatch;
    }
};

/// The file as the readers parse it: `source` in either v1 form. The first form (recorders before the sessions list)
/// is one session's object: `prompt_set`, `binary_sha256` and the totals. The second form is the totals and
/// `sessions`. `doc` gives the second form; a first-form source is its one session. `source` is provenance only: the
/// counts, `layers`, `experts` and `model_type` are what the checks and the seed read.
const ReadDoc = struct {
    model_type: []const u8,
    layers: u32,
    experts: u32,
    counts: []const []const u64,
    source: struct {
        requests: u64,
        prompt_tokens: u64,
        decode_tokens: u64,
        sessions: ?[]const Doc.Session = null,
        prompt_set: ?[]const u8 = null,
        binary_sha256: ?[]const u8 = null,
    },

    /// The second form; a source that is neither form (no sessions and no first-form provenance, or both) is
    /// `ExpertUsageMalformed`. `ar` holds the one session of a first-form source.
    fn doc(r: ReadDoc, ar: std.mem.Allocator) error{ ExpertUsageMalformed, OutOfMemory }!Doc {
        const s = r.source;
        const sessions: []const Doc.Session = if (s.sessions) |ss| blk: {
            if (s.prompt_set != null or s.binary_sha256 != null) return error.ExpertUsageMalformed;
            break :blk ss;
        } else blk: {
            const ps = s.prompt_set orelse return error.ExpertUsageMalformed;
            const sha = s.binary_sha256 orelse return error.ExpertUsageMalformed;
            const one = try ar.alloc(Doc.Session, 1);
            one[0] = .{ .prompt_set = ps, .binary_sha256 = sha, .requests = s.requests, .prompt_tokens = s.prompt_tokens, .decode_tokens = s.decode_tokens };
            break :blk one;
        };
        return .{ .model_type = r.model_type, .layers = r.layers, .experts = r.experts, .counts = r.counts, .source = .{ .requests = s.requests, .prompt_tokens = s.prompt_tokens, .decode_tokens = s.decode_tokens, .sessions = sessions } };
    }
};

/// `text` as the second form, in `ar` (an arena): `ExpertUsageMalformed` when it is not JSON of either v1 form. Every
/// string is a copy in `ar` (the recorder frees `text` and keeps the earlier sessions' strings).
fn readDoc(ar: std.mem.Allocator, text: []const u8) error{ ExpertUsageMalformed, OutOfMemory }!Doc {
    const r = std.json.parseFromSliceLeaky(ReadDoc, ar, text, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.ExpertUsageMalformed,
    };
    return r.doc(ar);
}

/// The record side (`expert_usage_record`): the counts over every request of every session recorded into `path`. At the
/// load the file at `path`, when there is one, is read and checked (`Doc.check`: refused by name, never overwritten) and
/// its counts and sessions are the start; this load is a new session. The file is written once at the load (the
/// directory proven writable, else `ExpertUsageRecordNotWritable`) and at each request's end (a temporary file beside
/// `path`, renamed over it). The arch adds a request's counts (`Stream.addUsage`) into `counts`, then calls `endRequest`.
pub const Recorder = struct {
    a: std.mem.Allocator,
    /// The earlier sessions' strings (the file's).
    arena: std.heap.ArenaAllocator,
    path: []u8,
    model_type: []const u8,
    binary_sha256: [64]u8,
    layers: u32,
    experts: u32,
    /// `experts` per layer, layer order.
    counts: []u64,
    requests: u64 = 0,
    prompt_tokens: u64 = 0,
    decode_tokens: u64 = 0,
    /// Every session, this load's last.
    sessions: []Doc.Session,

    /// `model_type` is static (the arch's name). The running binary's sha256 is read once here.
    pub fn init(a: std.mem.Allocator, io: std.Io, path: []const u8, model_type: []const u8, label: []const u8, layers: u32, experts: u32) !*Recorder {
        const r = try a.create(Recorder);
        errdefer a.destroy(r);
        r.* = .{ .a = a, .arena = .init(a), .path = undefined, .model_type = model_type, .binary_sha256 = selfSha256(a, io), .layers = layers, .experts = experts, .counts = undefined, .sessions = undefined };
        errdefer r.arena.deinit();
        const ar = r.arena.allocator();
        r.path = try ar.dupe(u8, path);
        r.counts = try ar.alloc(u64, @as(usize, layers) * experts);
        @memset(r.counts, 0);
        var earlier: []const Doc.Session = &.{};
        const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(max_file_bytes)) catch |e| switch (e) {
            error.FileNotFound => null,
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                log.warn("expert usage record: load refused: {s} is not readable: {s} (ExpertUsageMalformed)\n", .{ path, @errorName(e) });
                return error.ExpertUsageMalformed;
            },
        };
        if (text) |t| {
            defer a.free(t);
            // A first-form file is written back in the second form, its session kept as session 1.
            const parsed = readDoc(ar, t) catch |e| {
                if (e == error.OutOfMemory) return error.OutOfMemory;
                log.warn("expert usage record: load refused: {s} is not the v1 form (ExpertUsageMalformed); move it away to start a new record\n", .{path});
                return error.ExpertUsageMalformed;
            };
            parsed.check(model_type, layers, experts) catch |e| {
                log.warn("expert usage record: load refused: {s}: {s} (the file must be the v1 form of {s} with {d} layers x {d} experts)\n", .{ path, @errorName(e), model_type, layers, experts });
                return e;
            };
            for (parsed.counts, 0..) |row, l| @memcpy(r.counts[l * experts ..][0..experts], row);
            r.requests = parsed.source.requests;
            r.prompt_tokens = parsed.source.prompt_tokens;
            r.decode_tokens = parsed.source.decode_tokens;
            earlier = parsed.source.sessions;
        }
        r.sessions = try ar.alloc(Doc.Session, earlier.len + 1);
        @memcpy(r.sessions[0..earlier.len], earlier);
        r.sessions[earlier.len] = .{ .prompt_set = try ar.dupe(u8, label), .binary_sha256 = &r.binary_sha256 };
        r.write(io) catch |e| {
            log.warn("expert usage record: load refused: {s} is not writable: {s} (ExpertUsageRecordNotWritable)\n", .{ path, @errorName(e) });
            return error.ExpertUsageRecordNotWritable;
        };
        return r;
    }

    pub fn deinit(r: *Recorder) void {
        r.arena.deinit();
        r.a.destroy(r);
    }

    /// This load's session.
    pub fn session(r: *const Recorder) *Doc.Session {
        return &r.sessions[r.sessions.len - 1];
    }

    /// One request's end (its counts already added): the totals, then the file written.
    pub fn endRequest(r: *Recorder, io: std.Io, prompt_tokens: u64, decode_tokens: u64) !void {
        r.requests += 1;
        r.prompt_tokens += prompt_tokens;
        r.decode_tokens += decode_tokens;
        const ss = r.session();
        ss.requests += 1;
        ss.prompt_tokens += prompt_tokens;
        ss.decode_tokens += decode_tokens;
        try r.write(io);
    }

    pub fn write(r: *const Recorder, io: std.Io) !void {
        const rows = try r.a.alloc([]const u64, r.layers);
        defer r.a.free(rows);
        for (rows, 0..) |*row, l| row.* = r.counts[l * r.experts ..][0..r.experts];
        const doc: Doc = .{ .model_type = r.model_type, .layers = r.layers, .experts = r.experts, .counts = rows, .source = .{ .requests = r.requests, .prompt_tokens = r.prompt_tokens, .decode_tokens = r.decode_tokens, .sessions = r.sessions } };
        const json = try std.json.Stringify.valueAlloc(r.a, doc, .{});
        defer r.a.free(json);
        const tmp = try std.fmt.allocPrint(r.a, "{s}.tmp-{d}", .{ r.path, std.c.getpid() });
        defer r.a.free(tmp);
        // Before the write: a write that fails part way leaves no temporary file.
        errdefer std.Io.Dir.cwd().deleteFile(io, tmp) catch {};
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = tmp, .data = json });
        try std.Io.Dir.rename(std.Io.Dir.cwd(), tmp, std.Io.Dir.cwd(), r.path, io);
    }
};

/// The running binary's sha256 in hex, read in 1 MiB parts; 64 '0' when it cannot be read (the file then says so).
fn selfSha256(a: std.mem.Allocator, io: std.Io) [64]u8 {
    var out: [64]u8 = @splat('0');
    const path = std.process.executablePathAlloc(io, a) catch return out;
    defer a.free(path);
    const f = std.Io.Dir.cwd().openFile(io, path, .{}) catch return out;
    defer f.close(io);
    const buf = a.alloc(u8, 1 << 20) catch return out;
    defer a.free(buf);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var off: u64 = 0;
    while (true) {
        const n = f.readPositionalAll(io, buf, off) catch return out;
        h.update(buf[0..n]);
        off += n;
        if (n < buf.len) break;
    }
    const d = h.finalResult();
    _ = std.fmt.bufPrint(&out, "{x}", .{&d}) catch unreachable;
    return out;
}

/// The seed side: a profile read from the pack (`load`), checked against the stream's geometry once.
pub const Profile = struct {
    a: std.mem.Allocator,
    layers: u32,
    experts: u32,
    counts: []u64,
    /// "<last session's prompt set>[ + <k> earlier sessions], <requests> requests, <tokens> tokens, binary <sha 8>".
    source_buf: [max_label + 160]u8 = undefined,
    source_len: usize = 0,
    /// `order`'s scratch (one layer's ids).
    order_buf: []u16,

    pub fn deinit(p: *Profile) void {
        p.a.free(p.counts);
        p.a.free(p.order_buf);
    }

    pub fn source(p: *const Profile) []const u8 {
        return p.source_buf[0..p.source_len];
    }

    /// Layer `layer`'s experts of positive count, most used first (ties by the lower id), at most `rows`.
    pub fn order(p: *Profile, layer: u32, rows: u32) []const u16 {
        const c = p.counts[@as(usize, layer) * p.experts ..][0..p.experts];
        var n: usize = 0;
        for (c, 0..) |x, e| if (x > 0) {
            p.order_buf[n] = @intCast(e);
            n += 1;
        };
        const ids = p.order_buf[0..n];
        std.sort.pdq(u16, ids, c, struct {
            fn hotter(cs: []const u64, x: u16, y: u16) bool {
                return if (cs[x] != cs[y]) cs[x] > cs[y] else x < y;
            }
        }.hotter);
        return ids[0..@min(n, rows)];
    }
};

/// `Stream.seedResidents`'s order: each layer's most used experts of `p`, at most the layer's prompt rows.
pub const Order = struct {
    p: *Profile,
    rows: []const u32,

    pub fn of(o: Order, layer: u32) []const u16 {
        return o.p.order(layer, o.rows[layer]);
    }
};

/// The seed's result: the residents per layer (least and most), the prompt rows per layer (least and most), the records
/// read (`seedLine`).
pub const Seeded = struct {
    min: u32,
    max: u32,
    rows_min: u32,
    rows_max: u32,
    records: u64,

    /// From the stream's layers after the seed and the prompt rows.
    pub fn of(layers: anytype, rows: []const u32, records: u64) Seeded {
        var out: Seeded = .{ .min = std.math.maxInt(u32), .max = 0, .rows_min = std.math.maxInt(u32), .rows_max = 0, .records = records };
        for (layers, rows) |*ls, r| {
            out.min = @min(out.min, ls.policy.occupancy);
            out.max = @max(out.max, ls.policy.occupancy);
            out.rows_min = @min(out.rows_min, r);
            out.rows_max = @max(out.rows_max, r);
        }
        return out;
    }
};

/// `text` (the file's bytes) as a profile for a stream of `layers` x `experts` of `model_type`: refused by name when it
/// does not parse as the v1 form (`ExpertUsageMalformed`) or names another model or geometry (`ExpertUsageMismatch`).
pub fn parse(a: std.mem.Allocator, text: []const u8, model_type: []const u8, layers: u32, experts: u32) !Profile {
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const d = try readDoc(arena.allocator(), text);
    try d.check(model_type, layers, experts);
    const counts = try a.alloc(u64, @as(usize, layers) * experts);
    errdefer a.free(counts);
    for (d.counts, 0..) |row, l| @memcpy(counts[l * experts ..][0..experts], row);
    const order_buf = try a.alloc(u16, experts);
    var p: Profile = .{ .a = a, .layers = layers, .experts = experts, .counts = counts, .order_buf = order_buf };
    const s = d.source;
    const last = s.sessions[s.sessions.len - 1];
    const sha = last.binary_sha256[0..@min(8, last.binary_sha256.len)];
    var more_buf: [40]u8 = undefined;
    const more = if (s.sessions.len > 1) std.fmt.bufPrint(&more_buf, " + {d} earlier sessions", .{s.sessions.len - 1}) catch "" else "";
    const src = std.fmt.bufPrint(&p.source_buf, "{s}{s}, {d} requests, {d} prompt + {d} decode tokens, binary {s}", .{ last.prompt_set[0..@min(last.prompt_set.len, max_label)], more, s.requests, s.prompt_tokens, s.decode_tokens, sha }) catch "";
    p.source_len = src.len;
    return p;
}

/// The seed the load installs (`resident_seed` against the pack), decided once: a profile, or null (none) with the
/// reason. `resident_seed: usage` named without the file is refused by name (`ExpertUsageMissing`).
pub const Decision = struct {
    profile: ?Profile,
    reason: enum { usage, setting_none, no_file },
};

pub fn decide(a: std.mem.Allocator, io: std.Io, settings: *const Settings, pack_dir: []const u8, comptime arch: []const u8, layers: u32, experts: u32) !Decision {
    if (settings.seed == .none) return .{ .profile = null, .reason = .setting_none };
    const path = try std.fs.path.join(a, &.{ pack_dir, file_name });
    defer a.free(path);
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(max_file_bytes)) catch |e| switch (e) {
        error.FileNotFound => {
            if (settings.seed == .usage) {
                log.warn(arch ++ ": load refused: resident_seed is usage, but {s} is absent (ExpertUsageMissing)\n", .{path});
                return error.ExpertUsageMissing;
            }
            return .{ .profile = null, .reason = .no_file };
        },
        error.OutOfMemory => return error.OutOfMemory,
        else => {
            log.warn(arch ++ ": load refused: {s} is not readable: {s} (ExpertUsageMalformed)\n", .{ path, @errorName(e) });
            return error.ExpertUsageMalformed;
        },
    };
    defer a.free(text);
    const p = parse(a, text, arch, layers, experts) catch |e| {
        log.warn(arch ++ ": load refused: {s}: {s} (the file must be the v1 form of {s} with {d} layers x {d} experts)\n", .{ path, @errorName(e), arch, layers, experts });
        return e;
    };
    return .{ .profile = p, .reason = .usage };
}

/// The seed's install line: `<arch> expert residents seeded: <n> of <rows> rows per layer from expert-usage-v1.json
/// (<source>); <records> records, <bytes> B read in <ms> ms` (n and rows as `a-b` when the layers differ), or `none`.
pub fn seedLine(buf: []u8, comptime arch: []const u8, d: *const Decision, r: Seeded, bytes: u64, ms: u64) []const u8 {
    return switch (d.reason) {
        .setting_none => arch ++ " expert residents seeded: none (resident_seed: none)",
        .no_file => arch ++ " expert residents seeded: none (no " ++ file_name ++ ")",
        .usage => blk: {
            var nb: [24]u8 = undefined;
            var rb: [24]u8 = undefined;
            const n = range(&nb, r.min, r.max);
            const rows = range(&rb, r.rows_min, r.rows_max);
            break :blk std.fmt.bufPrint(buf, arch ++ " expert residents seeded: {s} of {s} rows per layer from " ++ file_name ++ " ({s}); {d} records, {d} B read in {d} ms", .{ n, rows, d.profile.?.source(), r.records, bytes, ms }) catch buf[0..0];
        },
    };
}

fn range(buf: []u8, lo: u32, hi: u32) []const u8 {
    return (if (lo == hi) std.fmt.bufPrint(buf, "{d}", .{lo}) else std.fmt.bufPrint(buf, "{d}-{d}", .{ lo, hi })) catch buf[0..0];
}

const testing = std.testing;

test "expert usage: settings parse, defaults, and bad values refused by name" {
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, "{\"resident_seed\":\"none\",\"expert_usage_record\":\"/tmp/u.json\",\"expert_usage_label\":\"he64+std16k\"}", .{});
    defer parsed.deinit();
    var s: Settings = .{};
    try testing.expect(s.apply(parsed.value.object));
    try s.check("t");
    try testing.expectEqual(Seed.none, s.seed.?);
    try testing.expectEqualStrings("/tmp/u.json", s.recordPath().?);
    try testing.expectEqualStrings("he64+std16k", s.label());
    const d: Settings = .{};
    try testing.expect(d.seed == null and d.recordPath() == null);
    for ([_][]const u8{ "{\"resident_seed\":\"hot\"}", "{\"resident_seed\":1}" }) |t| {
        const p = try std.json.parseFromSlice(std.json.Value, testing.allocator, t, .{});
        defer p.deinit();
        var x: Settings = .{};
        _ = x.apply(p.value.object);
        try testing.expectError(error.ResidentSeedSetting, x.check("t"));
    }
    for ([_][]const u8{ "{\"expert_usage_record\":\"rel/u.json\"}", "{\"expert_usage_record\":\"\"}", "{\"expert_usage_record\":true}" }) |t| {
        const p = try std.json.parseFromSlice(std.json.Value, testing.allocator, t, .{});
        defer p.deinit();
        var x: Settings = .{};
        _ = x.apply(p.value.object);
        try testing.expectError(error.ExpertUsageRecordSetting, x.check("t"));
    }
    // A label: 1 to `max_label` bytes, and only beside a record.
    const Want = struct { text: []const u8, err: anyerror };
    for ([_]Want{
        .{ .text = "{\"expert_usage_record\":\"/tmp/u.json\",\"expert_usage_label\":\"\"}", .err = error.ExpertUsageLabelSetting },
        .{ .text = "{\"expert_usage_record\":\"/tmp/u.json\",\"expert_usage_label\":3}", .err = error.ExpertUsageLabelSetting },
        .{ .text = "{\"expert_usage_label\":\"he64\"}", .err = error.ExpertUsageLabelWithoutRecord },
    }) |w| {
        const p = try std.json.parseFromSlice(std.json.Value, testing.allocator, w.text, .{});
        defer p.deinit();
        var x: Settings = .{};
        _ = x.apply(p.value.object);
        try testing.expectError(w.err, x.check("t"));
    }
    // An arch without the record (MiMo, DeepSeek-V4.1): either key named, any value (the empty label too), is refused.
    for ([_][]const u8{ "{\"expert_usage_label\":\"\"}", "{\"expert_usage_label\":\"x\"}", "{\"expert_usage_record\":\"/tmp/u.json\"}", "{\"expert_usage_record\":1}" }) |t| {
        const p = try std.json.parseFromSlice(std.json.Value, testing.allocator, t, .{});
        defer p.deinit();
        var x: Settings = .{};
        _ = x.apply(p.value.object);
        try testing.expectError(error.ExpertUsageRecordNotInstalled, x.checkSeedOnly("t"));
    }
    try testing.expectEqualStrings("mimo_v2 prompt residents: keep (decode's residents stay for the next prompt)", promptResidentsLine("mimo_v2", .keep));
    try testing.expectEqualStrings("deepseek_v41 prompt residents: forget (every request starts from empty rows)", promptResidentsLine("deepseek_v41", .forget));
}

test "expert usage: the file round-trips (record, write, parse); a later load's record adds to the file as a new session; a malformed or mismatched file, or a path that cannot be written, is refused by name" {
    const a = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    var empty = testing.tmpDir(.{});
    defer empty.cleanup();
    var empty_buf: [std.fs.max_path_bytes]u8 = undefined;
    const empty_dir = empty_buf[0..try empty.dir.realPath(testing.io, &empty_buf)];
    const path = try std.fs.path.join(a, &.{ dir, file_name });
    defer a.free(path);
    const r = try Recorder.init(a, testing.io, path, "glm5_next", "unit", 2, 4);
    defer r.deinit();
    @memcpy(r.counts, &[_]u64{ 5, 0, 9, 1, 0, 0, 3, 3 });
    try r.endRequest(testing.io, 26, 10);
    const text = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(max_file_bytes));
    defer a.free(text);
    var p = try parse(a, text, "glm5_next", 2, 4);
    defer p.deinit();
    try testing.expectEqualSlices(u64, r.counts, p.counts);
    try testing.expect(std.mem.startsWith(u8, p.source(), "unit, 1 requests, 26 prompt + 10 decode tokens, binary "));
    // Most used first, ties by the lower id, zero counts never, at most the rows.
    try testing.expectEqualSlices(u16, &.{ 2, 0, 3 }, p.order(0, 8));
    try testing.expectEqualSlices(u16, &.{ 2, 0 }, p.order(0, 2));
    try testing.expectEqualSlices(u16, &.{ 2, 3 }, p.order(1, 8));
    // The decision from the pack directory: the file found and checked; another geometry or model refused by name.
    var s: Settings = .{};
    var d = try decide(a, testing.io, &s, dir, "glm5_next", 2, 4);
    defer d.profile.?.deinit();
    try testing.expect(d.reason == .usage);
    var lb: [512]u8 = undefined;
    try testing.expect(std.mem.startsWith(u8, seedLine(&lb, "glm5_next", &d, .{ .min = 2, .max = 3, .rows_min = 4, .rows_max = 4, .records = 5 }, 100, 1), "glm5_next expert residents seeded: 2-3 of 4 rows per layer from expert-usage-v1.json (unit, 1 requests,"));
    try testing.expectError(error.ExpertUsageMismatch, decide(a, testing.io, &s, dir, "glm5_next", 3, 4));
    try testing.expectError(error.ExpertUsageMismatch, decide(a, testing.io, &s, dir, "glm5_next", 2, 5));
    try testing.expectError(error.ExpertUsageMismatch, decide(a, testing.io, &s, dir, "mimo_v2", 2, 4));
    try testing.expectError(error.ExpertUsageMalformed, parse(a, "{\"model_type\":\"glm5_next\",\"layers\":2}", "glm5_next", 2, 4));
    try testing.expectError(error.ExpertUsageMalformed, parse(a, "{\"model_type\":\"glm5_next\",\"layers\":2,\"experts\":4,\"counts\":[[1,2,3,4],[1,2]],\"source\":{\"prompt_set\":\"x\",\"requests\":1,\"prompt_tokens\":1,\"decode_tokens\":1,\"binary_sha256\":\"0\"}}", "glm5_next", 2, 4));
    try testing.expectError(error.ExpertUsageMalformed, parse(a, "not json", "glm5_next", 2, 4));
    // A second load recording to the same path: the file's counts and sessions are its start.
    {
        const r2 = try Recorder.init(a, testing.io, path, "glm5_next", "unit2", 2, 4);
        defer r2.deinit();
        try testing.expectEqualSlices(u64, r.counts, r2.counts);
        for (r2.counts, r.counts) |*c, x| c.* += x;
        try r2.endRequest(testing.io, 4, 2);
        const t2 = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(max_file_bytes));
        defer a.free(t2);
        var p2 = try parse(a, t2, "glm5_next", 2, 4);
        defer p2.deinit();
        try testing.expectEqualSlices(u64, &.{ 10, 0, 18, 2, 0, 0, 6, 6 }, p2.counts);
        try testing.expect(std.mem.startsWith(u8, p2.source(), "unit2 + 1 earlier sessions, 2 requests, 30 prompt + 12 decode tokens, binary "));
        // Another model or geometry at the path: refused by name, the file left as it is.
        try testing.expectError(error.ExpertUsageMismatch, Recorder.init(a, testing.io, path, "glm5_next", "x", 3, 4));
        try testing.expectError(error.ExpertUsageMismatch, Recorder.init(a, testing.io, path, "mimo_v2", "x", 2, 4));
        const t3 = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(max_file_bytes));
        defer a.free(t3);
        try testing.expectEqualStrings(t2, t3);
    }
    // A path whose directory is absent: refused at the load by name, no file left.
    const bad = try std.fs.path.join(a, &.{ empty_dir, "absent", file_name });
    defer a.free(bad);
    try testing.expectError(error.ExpertUsageRecordNotWritable, Recorder.init(a, testing.io, bad, "glm5_next", "x", 2, 4));
    // None: by the setting, or no file in the directory (and `usage` named without the file refused by name).
    s.seed = .none;
    try testing.expect((try decide(a, testing.io, &s, dir, "glm5_next", 2, 4)).reason == .setting_none);
    var e: Settings = .{};
    try testing.expect((try decide(a, testing.io, &e, empty_dir, "glm5_next", 2, 4)).reason == .no_file);
    e.seed = .usage;
    try testing.expectError(error.ExpertUsageMissing, decide(a, testing.io, &e, empty_dir, "glm5_next", 2, 4));
    try testing.expectEqualStrings("glm5_next expert residents seeded: none (no expert-usage-v1.json)", seedLine(&lb, "glm5_next", &.{ .profile = null, .reason = .no_file }, .{ .min = 0, .max = 0, .rows_min = 0, .rows_max = 0, .records = 0 }, 0, 0));
}

test "expert usage: the first-form file (one session's source) and the second form (totals + sessions) both parse; a first-form record is written back in the second form, its session kept as session 1" {
    const a = testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    // The header of the GLM-5.3-Flash pack's file (de03530's recorder) and synthesized 42 x 288 counts.
    const layers = 42;
    const experts = 288;
    const rows = try ar.alloc([]const u64, layers);
    for (rows, 0..) |*row, l| {
        const r = try ar.alloc(u64, experts);
        for (r, 0..) |*c, e| c.* = (l * 7 + e * 13) % 997;
        row.* = r;
    }
    const sha = "59d24e84597db2b155e7aaf66f3d78b37cc4c2ef2743f7418108d3b312ff06a0";
    const First = struct {
        model_type: []const u8,
        layers: u32,
        experts: u32,
        counts: []const []const u64,
        source: struct { prompt_set: []const u8, requests: u64, prompt_tokens: u64, decode_tokens: u64, binary_sha256: []const u8 },
    };
    const first = try std.json.Stringify.valueAlloc(ar, First{ .model_type = "glm5_next", .layers = layers, .experts = experts, .counts = rows, .source = .{ .prompt_set = "humaneval0-63+std16k", .requests = 65, .prompt_tokens = 27266, .decode_tokens = 9887, .binary_sha256 = sha } }, .{});
    const one = [_]Doc.Session{.{ .prompt_set = "humaneval0-63+std16k", .binary_sha256 = sha, .requests = 65, .prompt_tokens = 27266, .decode_tokens = 9887 }};
    const second = try std.json.Stringify.valueAlloc(ar, Doc{ .model_type = "glm5_next", .layers = layers, .experts = experts, .counts = rows, .source = .{ .requests = 65, .prompt_tokens = 27266, .decode_tokens = 9887, .sessions = &one } }, .{});
    for ([_][]const u8{ first, second }) |text| {
        var p = try parse(a, text, "glm5_next", layers, experts);
        defer p.deinit();
        for (rows, 0..) |row, l| try testing.expectEqualSlices(u64, row, p.counts[l * experts ..][0..experts]);
        try testing.expectEqualStrings("humaneval0-63+std16k, 65 requests, 27266 prompt + 9887 decode tokens, binary 59d24e84", p.source());
        try testing.expectError(error.ExpertUsageMismatch, parse(a, text, "glm5_next", 41, experts));
        try testing.expectError(error.ExpertUsageMismatch, parse(a, text, "mimo_v2", layers, experts));
    }
    // A source of neither form, of both forms, or with no session: malformed.
    const c = "{\"model_type\":\"glm5_next\",\"layers\":1,\"experts\":2,\"counts\":[[1,2]],\"source\":";
    const ok = c ++ "{\"requests\":1,\"prompt_tokens\":1,\"decode_tokens\":1,\"prompt_set\":\"x\",\"binary_sha256\":\"0\"}}";
    var p_ok = try parse(a, ok, "glm5_next", 1, 2);
    p_ok.deinit();
    for ([_][]const u8{
        c ++ "{\"requests\":1,\"prompt_tokens\":1,\"decode_tokens\":1}}",
        c ++ "{\"requests\":1,\"prompt_tokens\":1,\"decode_tokens\":1,\"prompt_set\":\"x\"}}",
        c ++ "{\"requests\":1,\"prompt_tokens\":1,\"decode_tokens\":1,\"prompt_set\":\"x\",\"binary_sha256\":\"0\",\"sessions\":[{\"prompt_set\":\"x\",\"binary_sha256\":\"0\"}]}}",
        c ++ "{\"requests\":1,\"prompt_tokens\":1,\"decode_tokens\":1,\"sessions\":[]}}",
        c ++ "{\"prompt_set\":\"x\",\"binary_sha256\":\"0\"}}",
    }) |t| try testing.expectError(error.ExpertUsageMalformed, parse(a, t, "glm5_next", 1, 2));
    // A record into a first-form file: the counts and totals are the start, the file becomes the second form, and its
    // one session is session 1.
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = dir_buf[0..try tmp.dir.realPath(testing.io, &dir_buf)];
    const path = try std.fs.path.join(a, &.{ dir, file_name });
    defer a.free(path);
    try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = path, .data = first });
    const r = try Recorder.init(a, testing.io, path, "glm5_next", "unit2", layers, experts);
    defer r.deinit();
    for (rows, 0..) |row, l| try testing.expectEqualSlices(u64, row, r.counts[l * experts ..][0..experts]);
    r.counts[0] += 1;
    try r.endRequest(testing.io, 4, 2);
    const t2 = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(max_file_bytes));
    defer a.free(t2);
    // The second form exactly (no first-form fields left in `source`).
    const strict = try std.json.parseFromSlice(Doc, a, t2, .{});
    defer strict.deinit();
    const s = strict.value.source;
    try testing.expectEqual(@as(usize, 2), s.sessions.len);
    try testing.expectEqualStrings("humaneval0-63+std16k", s.sessions[0].prompt_set);
    try testing.expectEqualStrings(sha, s.sessions[0].binary_sha256);
    try testing.expect(s.sessions[0].requests == 65 and s.sessions[0].prompt_tokens == 27266 and s.sessions[0].decode_tokens == 9887);
    try testing.expectEqualStrings("unit2", s.sessions[1].prompt_set);
    try testing.expect(s.sessions[1].requests == 1 and s.sessions[1].prompt_tokens == 4 and s.sessions[1].decode_tokens == 2);
    try testing.expect(s.requests == 66 and s.prompt_tokens == 27270 and s.decode_tokens == 9889);
    try testing.expectEqual(rows[0][0] + 1, strict.value.counts[0][0]);
}
