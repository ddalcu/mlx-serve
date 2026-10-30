//! The route recorder, a diagnostic that is off until an arch arms it: one request's decode routes, written in one
//! file at the request's end. The stream calls it at its plan, its lookahead, its settle and its grow
//! (`Stream.recorder`); the arch at the request's start (`begin`), at each step's end (`endStep`) and at the request's
//! end (`write`). Off, the stream's cost is one null check per route.
//!
//! What it keeps:
//! - per decode call of a routed layer: its layer, its step, the forward's rows, the routed ids in the router's flat
//!   order (row r's top-k at `[r * k, (r + 1) * k)`), per id what served it (`Flag`), and the lookahead's records for
//!   the next routed layer. A bank of several bank layers per routed layer (`banks_per_layer`, GLM-5.3's EXL3: K3 and
//!   K4) routes a call once per bank layer: the arch opens the call (`call`) and each of its routes fills its ids'
//!   flags in it;
//! - per step: its rows, drafts, accepted drafts, emitted tokens and wall time;
//! - per bank layer (the stream's layers: `banks_per_layer` per routed layer), at the decode handover: the prompt's
//!   routed rows per expert (the policy's prompt counts) and the residents in slot order, with the prompt and decode
//!   rows.
//!
//! The file (little-endian, packed in this order): `Header`; `prompt_rows` [n_layers]u32; `decode_rows`
//! [n_layers]u32; the prompt counts [n_layers][n_experts]u32; the residents [n_layers][n_experts]u16 (slot s of
//! bank layer l at [l][s]; `policy.no_expert` for an empty slot and past the decode rows); `Step` [n_steps]; `Route`
//! [n_routes]. `n_layers` counts bank layers; a route's `layer` is its routed layer.

const std = @import("std");
const policy = @import("policy.zig");

pub const magic = "GLMROUTE".*;
pub const version: u32 = 2;
pub const max_ids = policy.max_route_ids;
/// The lookahead's records per call (`lookahead.max_budget`).
pub const max_spec = 4;
/// A route index for a route the recorder did not keep.
pub const none: u32 = std.math.maxInt(u32);

/// What served one routed id (bits; the same for each repeat of an expert in a route).
pub const Flag = struct {
    /// Resident when the route was planned.
    pub const hit: u8 = 1;
    /// A miss admitted to a persistent row.
    pub const persistent: u8 = 2;
    /// A miss served from the transient window (not admitted).
    pub const transient: u8 = 4;
    /// Its record was read (a miss whose row still held its record reads nothing).
    pub const read: u8 = 8;
    /// Its gate/up range was copied out of a lookahead record (no preadv): the lookahead's claim.
    pub const adopted: u8 = 16;
    /// Its admission evicted a resident.
    pub const evicted: u8 = 32;
    /// Its bytes are in its row: a hit or a skipped load at the plan, a read when it landed.
    pub const settled: u8 = 64;
    /// Served by its routed layer's second bank layer (GLM-5.3's EXL3: the K4 bank).
    pub const bank: u8 = 128;
};

pub const Header = extern struct {
    magic: [8]u8 = magic,
    version: u32 = version,
    n_layers: u32,
    n_experts: u32,
    top_k: u32,
    max_ids: u32 = max_ids,
    prompt_tokens: u32,
    mtp_depth: u32,
    n_routes: u32,
    n_steps: u32,
    route_bytes: u32 = @sizeOf(Route),
    step_bytes: u32 = @sizeOf(Step),
    /// Routes and steps not kept (an allocation failed).
    dropped: u32,
    /// Bank layers per routed layer.
    banks_per_layer: u32,
    reserved: u32 = 0,
};

pub const Step = extern struct {
    first_route: u32,
    n_routes: u32,
    rows: u16,
    drafted: u16,
    accepted: u16,
    emitted: u16,
    wall_us: u32,
    reserved: u32 = 0,
};

pub const Route = extern struct {
    layer: u16,
    n_ids: u8,
    rows: u8,
    step: u32,
    ids: [max_ids]u16,
    flags: [max_ids]u8,
    /// The records read ahead for the next layer (`policy.no_expert`: none).
    spec: [max_spec]u16,
};

comptime {
    std.debug.assert(@sizeOf(Header) == 64 and @sizeOf(Step) == 24 and @sizeOf(Route) == 160);
}

/// Files named so far in this process (`nextFile`): a model loaded again never reuses a name.
var files = std.atomic.Value(u32).init(0);

/// The next file's sequence number in this process.
pub fn nextFile() u32 {
    return files.fetchAdd(1, .monotonic);
}

/// The diagnostic's switch, the host's convention (absent or `0`: off), and its directory: a value that starts with
/// `/` names the directory, any other value means `/tmp`.
pub fn dirOf(raw: ?[*:0]const u8) ?[]const u8 {
    const v = raw orelse return null;
    if (v[0] == '0') return null;
    const s = std.mem.span(v);
    return if (s.len > 0 and s[0] == '/') s else "/tmp";
}

pub const Recorder = struct {
    a: std.mem.Allocator,
    /// Bank layers (the stream's layers) and bank layers per routed layer.
    n_layers: u32,
    bpl: u32,
    n_experts: u32,
    top_k: u32,
    prompt_tokens: u32 = 0,
    mtp_depth: u32 = 0,
    prompt_rows: []u32,
    decode_rows: []u32,
    counts: []u32,
    residents: []u16,
    /// Per expert, the flags of the route being recorded (zero between routes).
    scratch: []u8,
    routes: std.ArrayList(Route) = .empty,
    steps: std.ArrayList(Step) = .empty,
    /// The open step: its index and its first route.
    step: u32 = 0,
    first: u32 = 0,
    /// The open call (`call`): the route its bank layers' routes fill.
    open: u32 = none,
    dropped: u32 = 0,

    /// `n_layers` bank layers, `bpl` of them per routed layer.
    pub fn init(a: std.mem.Allocator, n_layers: u32, n_experts: u32, top_k: u32, bpl: u32) !*Recorder {
        if (n_layers == 0 or n_layers > std.math.maxInt(u16) or n_experts == 0 or n_experts >= policy.no_expert or top_k == 0 or bpl == 0 or bpl > 2 or n_layers % bpl != 0) return error.InvalidRecorder;
        const self = try a.create(Recorder);
        errdefer a.destroy(self);
        const nl: usize = n_layers;
        const ne: usize = n_experts;
        const prompt_rows = try a.alloc(u32, nl);
        errdefer a.free(prompt_rows);
        const decode_rows = try a.alloc(u32, nl);
        errdefer a.free(decode_rows);
        const counts = try a.alloc(u32, nl * ne);
        errdefer a.free(counts);
        const residents = try a.alloc(u16, nl * ne);
        errdefer a.free(residents);
        const scratch = try a.alloc(u8, ne);
        @memset(scratch, 0);
        self.* = .{ .a = a, .n_layers = n_layers, .bpl = bpl, .n_experts = n_experts, .top_k = top_k, .prompt_rows = prompt_rows, .decode_rows = decode_rows, .counts = counts, .residents = residents, .scratch = scratch };
        self.begin(0, 0);
        return self;
    }

    pub fn deinit(self: *Recorder) void {
        const a = self.a;
        a.free(self.prompt_rows);
        a.free(self.decode_rows);
        a.free(self.counts);
        a.free(self.residents);
        a.free(self.scratch);
        self.routes.deinit(a);
        self.steps.deinit(a);
        a.destroy(self);
    }

    /// A request's start: what the recorder kept is dropped.
    pub fn begin(self: *Recorder, prompt_tokens: u32, mtp_depth: u32) void {
        self.routes.clearRetainingCapacity();
        self.steps.clearRetainingCapacity();
        self.step = 0;
        self.first = 0;
        self.open = none;
        self.dropped = 0;
        self.prompt_tokens = prompt_tokens;
        self.mtp_depth = mtp_depth;
        @memset(self.prompt_rows, 0);
        @memset(self.decode_rows, 0);
        @memset(self.counts, 0);
        @memset(self.residents, policy.no_expert);
    }

    /// The decode handover, per bank layer: its prompt counts, its residents in slot order (`slots.len` = its decode
    /// rows) and its prompt rows.
    pub fn handover(self: *Recorder, layer: u32, counts: []const u32, slots: []const u16, prompt_rows: u32) void {
        const ne = self.n_experts;
        @memcpy(self.counts[layer * ne ..][0..ne], counts[0..ne]);
        const res = self.residents[layer * ne ..][0..ne];
        @memset(res, policy.no_expert);
        @memcpy(res[0..slots.len], slots);
        self.prompt_rows[layer] = prompt_rows;
        self.decode_rows[layer] = @intCast(slots.len);
    }

    /// A decode call of routed layer `layer` over several bank layers: `ids` as routed (rows x k). Each bank layer's
    /// route fills its ids' flags in it until `endCall`.
    pub fn call(self: *Recorder, layer: u32, ids: []const u16) void {
        self.open = self.add(layer, ids);
    }

    pub fn endCall(self: *Recorder) void {
        self.open = none;
    }

    fn add(self: *Recorder, layer: u32, ids: []const u16) u32 {
        if (ids.len == 0 or ids.len > max_ids) {
            self.dropped += 1;
            return none;
        }
        const index: u32 = @intCast(self.routes.items.len);
        const r = self.routes.addOne(self.a) catch {
            self.dropped += 1;
            return none;
        };
        r.* = .{ .layer = @intCast(layer), .n_ids = @intCast(ids.len), .rows = @intCast(ids.len / self.top_k), .step = self.step, .ids = @splat(policy.no_expert), .flags = @splat(0), .spec = @splat(policy.no_expert) };
        @memcpy(r.ids[0..ids.len], ids);
        return index;
    }

    /// One decode route of bank layer `layer`: `ids` as routed, its plan, and per load (plan order) whether it reads.
    /// Inside a call (`call`) it fills its ids' flags in the call's route; otherwise it is a route of its own. Returns
    /// the route's index (`none` when it is not kept).
    pub fn route(self: *Recorder, layer: u32, ids: []const u16, plan: *const policy.Plan, reads: []const bool) u32 {
        const routed = layer / self.bpl;
        const index = if (self.open != none and self.routes.items[self.open].layer == routed) self.open else self.add(routed, ids);
        if (index == none) return none;
        const r = &self.routes.items[index];
        const sc = self.scratch;
        const side: u8 = if (layer % self.bpl == 1) Flag.bank else 0;
        for (plan.hitsOf()) |e| sc[e] = Flag.hit | Flag.settled | side;
        for (plan.loadsOf(), reads[0..plan.n_loads]) |l, rd| sc[l.expert] = (if (l.persistent) Flag.persistent else Flag.transient) | (if (rd) Flag.read else Flag.settled) | side;
        for (plan.evictionsOf()) |ev| sc[ev.next] |= Flag.evicted;
        for (r.ids[0..r.n_ids], r.flags[0..r.n_ids]) |e, *f| {
            if (sc[e] != 0) f.* = sc[e];
        }
        for (ids) |e| sc[e] = 0;
        return index;
    }

    /// The records route `index`'s lookahead read ahead for the next routed layer.
    pub fn spec(self: *Recorder, index: u32, experts: []const u16) void {
        if (index == none) return;
        const r = &self.routes.items[index];
        const n = @min(experts.len, max_spec);
        @memcpy(r.spec[0..n], experts[0..n]);
    }

    /// A read of route `index` landed: `expert`'s record, and whether its gate/up range was copied out of a lookahead
    /// record.
    pub fn settled(self: *Recorder, index: u32, expert: u16, adopted: bool) void {
        if (index == none) return;
        const r = &self.routes.items[index];
        const f: u8 = Flag.settled | (if (adopted) Flag.adopted else 0);
        for (r.ids[0..r.n_ids], r.flags[0..r.n_ids]) |e, *x| {
            if (e == expert) x.* |= f;
        }
    }

    /// A step's end: the forward's rows, its drafts, the drafts accepted, the tokens it emitted and its wall time.
    pub fn endStep(self: *Recorder, rows: u32, drafted: u32, accepted: u32, emitted: u32, wall_ns: u64) void {
        const n: u32 = @intCast(self.routes.items.len);
        self.steps.append(self.a, .{
            .first_route = self.first,
            .n_routes = n - self.first,
            .rows = @intCast(rows),
            .drafted = @intCast(drafted),
            .accepted = @intCast(accepted),
            .emitted = @intCast(emitted),
            .wall_us = @intCast(@min(wall_ns / std.time.ns_per_us, std.math.maxInt(u32))),
        }) catch {
            self.dropped += 1;
        };
        self.step += 1;
        self.first = n;
    }

    /// Something to write: a step or a route since `begin`.
    pub fn any(self: *const Recorder) bool {
        return self.routes.items.len > 0 or self.steps.items.len > 0;
    }

    /// The file's bytes (the caller frees them).
    pub fn image(self: *const Recorder, a: std.mem.Allocator) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(a);
        const h: Header = .{ .n_layers = self.n_layers, .n_experts = self.n_experts, .top_k = self.top_k, .prompt_tokens = self.prompt_tokens, .mtp_depth = self.mtp_depth, .n_routes = @intCast(self.routes.items.len), .n_steps = @intCast(self.steps.items.len), .dropped = self.dropped, .banks_per_layer = self.bpl };
        try out.appendSlice(a, std.mem.asBytes(&h));
        try out.appendSlice(a, std.mem.sliceAsBytes(self.prompt_rows));
        try out.appendSlice(a, std.mem.sliceAsBytes(self.decode_rows));
        try out.appendSlice(a, std.mem.sliceAsBytes(self.counts));
        try out.appendSlice(a, std.mem.sliceAsBytes(self.residents));
        try out.appendSlice(a, std.mem.sliceAsBytes(self.steps.items));
        try out.appendSlice(a, std.mem.sliceAsBytes(self.routes.items));
        return out.toOwnedSlice(a);
    }

    /// The file at `path` (absolute).
    pub fn write(self: *Recorder, a: std.mem.Allocator, io: std.Io, path: []const u8) !void {
        const bytes = try self.image(a);
        defer a.free(bytes);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
    }
};

/// A route file read back (tests): its header and each part's bytes, in the file's order.
pub const Parts = struct {
    header: Header,
    prompt_rows: []const u8,
    decode_rows: []const u8,
    counts: []const u8,
    residents: []const u8,
    steps: []const u8,
    routes: []const u8,

    /// `img`'s parts at its header's counts; `error.RouteFileSize` when its bytes are not exactly them.
    pub fn of(img: []const u8) error{RouteFileSize}!Parts {
        if (img.len < @sizeOf(Header)) return error.RouteFileSize;
        const h = std.mem.bytesToValue(Header, img[0..@sizeOf(Header)]);
        const per_layer: usize = h.n_layers;
        const per_expert = per_layer * h.n_experts;
        const lens = [_]usize{ per_layer * 4, per_layer * 4, per_expert * 4, per_expert * 2, @as(usize, h.n_steps) * @sizeOf(Step), @as(usize, h.n_routes) * @sizeOf(Route) };
        var total: usize = @sizeOf(Header);
        for (lens) |n| total += n;
        if (img.len != total) return error.RouteFileSize;
        var parts: [lens.len][]const u8 = undefined;
        var off: usize = @sizeOf(Header);
        for (&parts, lens) |*p, n| {
            p.* = img[off..][0..n];
            off += n;
        }
        return .{ .header = h, .prompt_rows = parts[0], .decode_rows = parts[1], .counts = parts[2], .residents = parts[3], .steps = parts[4], .routes = parts[5] };
    }

    pub fn step(p: *const Parts, i: usize) Step {
        return std.mem.bytesToValue(Step, p.steps[i * @sizeOf(Step) ..][0..@sizeOf(Step)]);
    }

    pub fn route(p: *const Parts, i: usize) Route {
        return std.mem.bytesToValue(Route, p.routes[i * @sizeOf(Route) ..][0..@sizeOf(Route)]);
    }
};

const testing = std.testing;

test "expert routes: the switch follows the host's convention, a value starting with / names the directory" {
    try testing.expectEqual(@as(?[]const u8, null), dirOf(null));
    try testing.expectEqual(@as(?[]const u8, null), dirOf("0"));
    try testing.expectEqual(@as(?[]const u8, null), dirOf("0/tmp/x"));
    try testing.expectEqualStrings("/tmp", dirOf("1").?);
    try testing.expectEqualStrings("/tmp", dirOf("").?);
    try testing.expectEqualStrings("/Users/x/routes", dirOf("/Users/x/routes").?);
    const n = nextFile();
    try testing.expectEqual(n + 1, nextFile());
}

test "expert routes: a plan's hits, loads, evictions and reads become each id's flags, the image reads back" {
    const a = testing.allocator;
    const rec = try Recorder.init(a, 2, 16, 2, 1);
    defer rec.deinit();
    rec.begin(37, 1);
    var p = try policy.LayerPolicy.init(a, 16, 3);
    defer p.deinit(a);
    var plan: policy.Plan = .{};
    // The prompt leaves 2 and 1 resident (the seed's least frequent first); the grow adds one empty row.
    p.prepareSeed(&.{ 1, 1, 2 });
    p.plan(&.{ 1, 2 }, .prefill, &plan);
    try p.grow(4);
    rec.handover(0, p.prefill_freq, p.slot_to_expert[0..p.capacity], 3);
    try testing.expectEqualSlices(u16, &.{ 2, 1, policy.no_expert, policy.no_expert }, rec.residents[0..4]);
    try testing.expectEqual(@as(u32, 4), rec.decode_rows[0]);
    // Two rows of top-2: 1 hits, 5 and 7 take the empty rows (7's row still held its record), 5 repeats.
    p.plan(&.{ 1, 5, 7, 5 }, .decode, &plan);
    const first = rec.route(0, &.{ 1, 5, 7, 5 }, &plan, &.{ true, false });
    try testing.expectEqual(@as(u32, 0), first);
    rec.spec(first, &.{ 9, 3 });
    rec.settled(first, 5, true);
    rec.endStep(2, 1, 1, 2, 12_345_678);
    const r = rec.routes.items[0];
    try testing.expectEqual(@as(u8, 2), r.rows);
    try testing.expectEqual(@as(u8, Flag.hit | Flag.settled), r.flags[0]);
    try testing.expectEqual(@as(u8, Flag.persistent | Flag.read | Flag.settled | Flag.adopted), r.flags[1]);
    try testing.expectEqual(r.flags[1], r.flags[3]);
    try testing.expectEqual(@as(u8, Flag.persistent | Flag.settled), r.flags[2]);
    try testing.expectEqualSlices(u16, &.{ 9, 3, policy.no_expert, policy.no_expert }, &r.spec);
    // A full layer: 9 evicts the coldest resident.
    p.plan(&.{ 9, 5 }, .decode, &plan);
    try testing.expectEqual(@as(u32, 1), plan.n_evictions);
    _ = rec.route(1, &.{ 9, 5 }, &plan, &.{true});
    try testing.expectEqual(@as(u8, Flag.persistent | Flag.read | Flag.evicted), rec.routes.items[1].flags[0]);
    rec.endStep(1, 0, 0, 1, 1_000);
    for (rec.scratch) |x| try testing.expectEqual(@as(u8, 0), x);

    const img = try rec.image(a);
    defer a.free(img);
    try testing.expectEqual(@as(usize, 64 + 2 * 2 * 4 + 2 * 16 * 4 + 2 * 16 * 2 + 2 * 24 + 2 * 160), img.len);
    const im = try Parts.of(img);
    const h = im.header;
    try testing.expectEqualSlices(u8, "GLMROUTE", &h.magic);
    try testing.expectEqual(@as(u32, 37), h.prompt_tokens);
    try testing.expectEqual(@as(u32, 2), h.n_routes);
    try testing.expectEqual(@as(u32, 2), h.n_steps);
    try testing.expectEqual(@as(u32, 0), h.dropped);
    try testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, im.counts[4..8], .little));
    try testing.expectEqual(Step{ .first_route = 0, .n_routes = 1, .rows = 2, .drafted = 1, .accepted = 1, .emitted = 2, .wall_us = 12_345 }, im.step(0));
    const r1 = im.route(1);
    try testing.expectEqual(@as(u16, 1), r1.layer);
    try testing.expectEqual(@as(u32, 1), r1.step);
    try testing.expectEqualSlices(u16, &.{ 9, 5 }, r1.ids[0..2]);
}

test "expert routes: a call over two bank layers is one route, each id flagged by its own bank layer's plan" {
    const a = testing.allocator;
    // One routed layer of two bank layers (8 experts each side of the id range): bank layer 0 holds even ids, 1 odd.
    const rec = try Recorder.init(a, 2, 16, 2, 2);
    defer rec.deinit();
    rec.begin(1, 0);
    var even = try policy.LayerPolicy.init(a, 16, 2);
    defer even.deinit(a);
    var odd = try policy.LayerPolicy.init(a, 16, 2);
    defer odd.deinit(a);
    var plan: policy.Plan = .{};
    // The prompt leaves 2 resident in the even bank layer, nothing in the odd one.
    even.prepareSeed(&.{2});
    even.plan(&.{2}, .prefill, &plan);
    rec.handover(0, even.prefill_freq, even.slot_to_expert[0..even.capacity], 2);
    rec.handover(1, odd.prefill_freq, odd.slot_to_expert[0..odd.capacity], 2);
    // Two rows of top-2: [2, 3], [4, 3].
    const ids = [_]u16{ 2, 3, 4, 3 };
    rec.call(0, &ids);
    even.plan(&.{ 2, 4 }, .decode, &plan);
    const k3 = rec.route(0, &.{ 2, 4 }, &plan, &.{true});
    odd.plan(&.{ 3, 3 }, .decode, &plan);
    const k4 = rec.route(1, &.{ 3, 3 }, &plan, &.{true});
    rec.endCall();
    try testing.expectEqual(@as(u32, 0), k3);
    try testing.expectEqual(k3, k4);
    rec.settled(k4, 3, false);
    rec.endStep(2, 1, 0, 1, 0);
    try testing.expectEqual(@as(usize, 1), rec.routes.items.len);
    const r = rec.routes.items[0];
    try testing.expectEqual(@as(u8, 2), r.rows);
    try testing.expectEqualSlices(u16, &ids, r.ids[0..4]);
    try testing.expectEqual(@as(u8, Flag.hit | Flag.settled), r.flags[0]);
    try testing.expectEqual(@as(u8, Flag.persistent | Flag.read | Flag.settled | Flag.bank), r.flags[1]);
    try testing.expectEqual(@as(u8, Flag.persistent | Flag.read), r.flags[2]);
    try testing.expectEqual(r.flags[1], r.flags[3]);
    // Outside a call, a bank layer's route is its own, at its routed layer.
    odd.plan(&.{5}, .decode, &plan);
    try testing.expectEqual(@as(u32, 1), rec.route(1, &.{5}, &plan, &.{true}));
    try testing.expectEqual(@as(u16, 0), rec.routes.items[1].layer);
    const img = try rec.image(a);
    defer a.free(img);
    try testing.expectEqual(@as(u32, 2), (try Parts.of(img)).header.banks_per_layer);
}
