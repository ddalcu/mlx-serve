//! LAN model sharing (v1 — LAN only; WAN rooms layer onto these same seams).
//!
//! Two independent halves, both OFF by default:
//!   • SHARE (`--lan-share <all|id,...>`): advertise this server as a Bonjour
//!     `_mlxserve._tcp` service and open the SHARED INFERENCE SURFACE to
//!     non-loopback clients — a route allowlist (`routeClass`) plus a
//!     shared-model check enforced in server.zig's LAN gate. Admin endpoints
//!     (load/unload, metrics, status page, stored responses) stay host-local.
//!   • DISCOVER (`--lan-discover`): browse for peers, mirror their shared
//!     models into `/v1/models` as `<id>@<peer>` entries, and transparently
//!     proxy any request naming one to its host (byte-for-byte streaming
//!     tunnel). Claude Code / any localhost client gets LAN models for free.
//!
//! Design rules:
//!   - The proxy is a TRANSPORT: no scheduler, no MLX, no inference-thread
//!     involvement. Tunnels run on the calling connection thread; discovery
//!     runs on one browser thread; dns_sd (mDNSResponder, in libSystem) does
//!     all mDNS work — no hand-rolled multicast.
//!   - Loops are impossible by construction: remote entries are never
//!     included in the model list served to LAN peers, and `@peer` ids from
//!     non-loopback clients are denied at the gate (no multi-hop).
//!   - The host sees tunneled prompts in plaintext (it computes on them) —
//!     the Swift Settings pane carries the disclosure.

const std = @import("std");
const crt = @import("../../sdk/util/crt.zig");
const os_mutex = @import("../../sdk/util/os_mutex.zig");
const log = @import("../../sdk/util/log.zig");
const agent_budget = @import("../launch/budget.zig");

pub const SERVICE_TYPE = "_mlxserve._tcp";

// ─────────────────────────────────────────────────────────────────────────────
// Pure policy + codec helpers (hermetic tests at the bottom of this file)
// ─────────────────────────────────────────────────────────────────────────────

/// `<bare>@<peer>` — the id form remote models take in /v1/models. Local
/// registry ids never contain '@' (HF org/repo + dir basenames), and a
/// registered local id that does still wins at the dispatch site (registry
/// peek runs before the remote interception).
pub const RemoteId = struct { bare: []const u8, peer: []const u8 };

pub fn splitRemoteId(id: []const u8) ?RemoteId {
    const at = std.mem.lastIndexOfScalar(u8, id, '@') orelse return null;
    if (at == 0 or at + 1 == id.len) return null;
    return .{ .bare = id[0..at], .peer = id[at + 1 ..] };
}

/// What a non-loopback, non-API-key client may reach while sharing is on.
pub const RouteClass = enum { open, model_gated, denied };

pub fn routeClass(method: []const u8, path: []const u8) RouteClass {
    const eql = std.mem.eql;
    if (eql(u8, method, "OPTIONS")) return .open; // CORS preflight
    if (eql(u8, method, "GET")) {
        for ([_][]const u8{ "/health", "/v1/models", "/api/version", "/launch" }) |p|
            if (eql(u8, path, p)) return .open;
        return .denied;
    }
    if (eql(u8, method, "POST")) {
        for ([_][]const u8{
            "/v1/chat/completions",   "/v1/completions",
            "/v1/messages",           "/v1/responses",
            "/v1/embeddings",         "/v1/images/generations",
            "/v1/images/edits",       "/v1/audio/speech",
            "/v1/audio/music-generations", "/v1/audio/sound-generations",
            "/v1/video/generations",  "/v1/3d/generations",
            "/api/chat",              "/api/generate",
            "/api/embed",             "/api/embeddings",
            "/v1/audio/transcriptions",
        }) |p| if (eql(u8, path, p)) return .model_gated;
        return .denied;
    }
    return .denied;
}

/// Which local models `--lan-share` exposes: `all`, or a csv of ids.
/// Registry ids are `basename` or `org/name`; the app's share list sends
/// basenames — matching is symmetric basename-tolerant so the two can't drift.
pub const SharedSet = struct {
    all: bool = false,
    ids: []const []const u8 = &.{},

    pub fn parse(alloc: std.mem.Allocator, spec: []const u8) !SharedSet {
        const trimmed = std.mem.trim(u8, spec, " \t");
        if (std.ascii.eqlIgnoreCase(trimmed, "all")) return .{ .all = true };
        var list: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (list.items) |s| alloc.free(s);
            list.deinit(alloc);
        }
        var it = std.mem.splitScalar(u8, trimmed, ',');
        while (it.next()) |raw| {
            const id = std.mem.trim(u8, raw, " \t");
            if (id.len > 0) try list.append(alloc, try alloc.dupe(u8, id));
        }
        return .{ .ids = try list.toOwnedSlice(alloc) };
    }

    pub fn deinit(self: *SharedSet, alloc: std.mem.Allocator) void {
        for (self.ids) |id| alloc.free(id);
        alloc.free(self.ids);
        self.* = .{};
    }

    pub fn empty(self: SharedSet) bool {
        return !self.all and self.ids.len == 0;
    }

    pub fn allows(self: SharedSet, id: []const u8) bool {
        if (self.all) return true;
        if (id.len == 0) return false;
        for (self.ids) |entry| {
            if (std.mem.eql(u8, entry, id) or
                std.mem.eql(u8, entry, basename(id)) or
                std.mem.eql(u8, basename(entry), id)) return true;
        }
        return false;
    }

    fn basename(s: []const u8) []const u8 {
        const slash = std.mem.lastIndexOfScalar(u8, s, '/') orelse return s;
        return s[slash + 1 ..];
    }
};

/// Bonjour instance names are arbitrary UTF-8, but the peer name doubles as a
/// model-id suffix — collapse anything outside [A-Za-z0-9._-] to '-' ('@'
/// would break the suffix split; spaces annoy CLIs). Never empty.
pub fn sanitizeName(buf: []u8, raw: []const u8) []const u8 {
    var n: usize = 0;
    var pending_dash = false;
    for (raw) |c| {
        if (std.ascii.isAlphanumeric(c) or c == '.' or c == '_' or c == '-') {
            if (pending_dash and n > 0 and n < buf.len) {
                buf[n] = '-';
                n += 1;
            }
            pending_dash = false;
            if (n < buf.len) {
                buf[n] = c;
                n += 1;
            }
        } else pending_dash = true;
    }
    return if (n == 0) "mac" else buf[0..n];
}

/// Rebuild `body` with the model value replaced by `bare_id`. `model_value`
/// MUST be the slice `server.parseModelFromBody` returned — i.e. alias `body`
/// — so the decide and rewrite layers can never disagree on which bytes are
/// the model field. Anything else (a canonicalised copy) is refused: its
/// offset would be read off an unrelated pointer.
pub fn rewriteModelValue(alloc: std.mem.Allocator, body: []const u8, model_value: []const u8, bare_id: []const u8) ![]u8 {
    const at = @intFromPtr(model_value.ptr);
    const base = @intFromPtr(body.ptr);
    if (at < base or at - base + model_value.len > body.len) return error.ModelNotInBody;
    const start = at - base;
    const out = try alloc.alloc(u8, body.len - model_value.len + bare_id.len);
    @memcpy(out[0..start], body[0..start]);
    @memcpy(out[start..][0..bare_id.len], bare_id);
    @memcpy(out[start + bare_id.len ..], body[start + model_value.len ..]);
    return out;
}

/// Collapse JSON's slash escapes: the optional `\/` to `/` and `\\` to `\`. Swift's
/// JSONSerialization (and PHP's json_encode) escape every slash, so a remote id's org prefix
/// arrives as `ddalcu\/gemma…` from the app while the peer table stores the canonical form —
/// the same `\/` class the load-model handler documents — and a Windows path id arrives as
/// `C:\\Users…`. Returns the input verbatim (zero-copy) when there is nothing to collapse or
/// the scratch buffer is too small.
pub fn unescapeJsonSlashes(buf: []u8, s: []const u8) []const u8 {
    if ((std.mem.indexOf(u8, s, "\\/") == null and std.mem.indexOf(u8, s, "\\\\") == null) or s.len > buf.len) return s;
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\' and i + 1 < s.len and (s[i + 1] == '/' or s[i + 1] == '\\')) i += 1;
        buf[n] = s[i];
        n += 1;
    }
    return buf[0..n];
}

/// TXT record wire format: length-prefixed `key=value` strings.
pub fn txtBuild(buf: []u8, token: []const u8) []const u8 {
    std.debug.assert(buf.len >= 5 + 2 + token.len and token.len <= 253);
    buf[0] = 3;
    @memcpy(buf[1..4], "v=1");
    buf[4] = @intCast(2 + token.len);
    @memcpy(buf[5..7], "t=");
    @memcpy(buf[7..][0..token.len], token);
    return buf[0 .. 7 + token.len];
}

pub fn txtFind(txt: []const u8, key_eq: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < txt.len) {
        const len: usize = txt[i];
        i += 1;
        if (i + len > txt.len) return null; // truncated/hostile record
        const entry = txt[i .. i + len];
        i += len;
        if (std.mem.startsWith(u8, entry, key_eq)) return entry[key_eq.len..];
    }
    return null;
}

/// One shared model as advertised by a peer: its bare id (for request
/// routing) and the ready-to-emit /v1/models entry JSON (id rewritten to
/// `<id>@<peer>`, plus a top-level `"lan_peer"` marker clients badge on).
pub const PeerModel = struct { id: []const u8, entry_json: []const u8 };

pub fn freePeerModels(alloc: std.mem.Allocator, models: []PeerModel) void {
    for (models) |m| {
        alloc.free(m.id);
        alloc.free(m.entry_json);
    }
    alloc.free(models);
}

/// Parse a peer's /v1/models response body into PeerModels. Entries that
/// aren't objects with a string id are skipped; a body without a `data`
/// array is an error (not an mlx-serve peer).
pub fn parsePeerModels(alloc: std.mem.Allocator, body: []const u8, peer: []const u8) ![]PeerModel {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return error.BadPeerJson;
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadPeerJson;
    const data = parsed.value.object.get("data") orelse return error.BadPeerJson;
    if (data != .array) return error.BadPeerJson;

    var out: std.ArrayList(PeerModel) = .empty;
    errdefer {
        for (out.items) |m| {
            alloc.free(m.id);
            alloc.free(m.entry_json);
        }
        out.deinit(alloc);
    }
    for (data.array.items) |*item| {
        if (item.* != .object) continue;
        const id_v = item.object.get("id") orelse continue;
        if (id_v != .string) continue;
        // Never mirror the peer's own REMOTE stubs (entries badged lan_peer):
        // a peer that fails to filter its list — old binary, or a loopback
        // fetch between two servers on one Mac bypassing the non-loopback
        // gate — would otherwise chain re-exports into `@a@b` ids.
        if (item.object.get("lan_peer") != null) continue;
        const bare = try alloc.dupe(u8, id_v.string);
        errdefer alloc.free(bare);
        // Mutations of the parsed tree allocate from ITS arena (freed
        // wholesale by parsed.deinit) — never mix the caller's gpa in.
        const arena = parsed.arena.allocator();
        const full = try std.fmt.allocPrint(arena, "{s}@{s}", .{ bare, peer });
        try item.object.put(arena, "id", .{ .string = full });
        try item.object.put(arena, "lan_peer", .{ .string = peer });
        // A peer's alias resolves only on that peer.
        _ = item.object.orderedRemove("alias");
        try agent_budget.stampRow(arena, &item.object);
        const entry_json = try std.json.Stringify.valueAlloc(alloc, item.*, .{});
        errdefer alloc.free(entry_json);
        try out.append(alloc, .{ .id = bare, .entry_json = entry_json });
    }
    return out.toOwnedSlice(alloc);
}

// ─────────────────────────────────────────────────────────────────────────────
// dns_sd FFI (mDNSResponder client API — exported by libSystem, no extra link)
// ─────────────────────────────────────────────────────────────────────────────

const DNSServiceRef = ?*anyopaque;
const kDNSServiceFlagsMoreComing: u32 = 0x1;
const kDNSServiceFlagsAdd: u32 = 0x2;
const kDNSServiceProtocol_IPv4: u32 = 0x01;

const BrowseReply = *const fn (DNSServiceRef, u32, u32, i32, ?[*:0]const u8, ?[*:0]const u8, ?[*:0]const u8, ?*anyopaque) callconv(.c) void;
const ResolveReply = *const fn (DNSServiceRef, u32, u32, i32, ?[*:0]const u8, ?[*:0]const u8, u16, u16, ?[*]const u8, ?*anyopaque) callconv(.c) void;
const AddrInfoReply = *const fn (DNSServiceRef, u32, u32, i32, ?[*:0]const u8, ?*const std.posix.sockaddr, u32, ?*anyopaque) callconv(.c) void;

extern "c" fn DNSServiceRegister(ref: *DNSServiceRef, flags: u32, interface: u32, name: ?[*:0]const u8, regtype: [*:0]const u8, domain: ?[*:0]const u8, host: ?[*:0]const u8, port_be: u16, txt_len: u16, txt: ?*const anyopaque, cb: ?*const anyopaque, ctx: ?*anyopaque) i32;
/// Linux's dns_sd is Avahi's compat layer, which fails every call (-65537) when avahi-daemon is down.
const daemon_hint = if (@import("builtin").os.tag == .linux) " (is avahi-daemon running? systemctl enable --now avahi-daemon)" else "";
extern "c" fn DNSServiceBrowse(ref: *DNSServiceRef, flags: u32, interface: u32, regtype: [*:0]const u8, domain: ?[*:0]const u8, cb: BrowseReply, ctx: ?*anyopaque) i32;
extern "c" fn DNSServiceResolve(ref: *DNSServiceRef, flags: u32, interface: u32, name: [*:0]const u8, regtype: [*:0]const u8, domain: [*:0]const u8, cb: ResolveReply, ctx: ?*anyopaque) i32;
extern "c" fn DNSServiceGetAddrInfo(ref: *DNSServiceRef, flags: u32, interface: u32, protocol: u32, hostname: [*:0]const u8, cb: AddrInfoReply, ctx: ?*anyopaque) i32;
extern "c" fn DNSServiceRefSockFD(ref: DNSServiceRef) i32;
extern "c" fn DNSServiceProcessResult(ref: DNSServiceRef) i32;
extern "c" fn DNSServiceRefDeallocate(ref: DNSServiceRef) void;

// ─────────────────────────────────────────────────────────────────────────────
// Runtime: advertiser + browser thread + peer table + proxy tunnel
// ─────────────────────────────────────────────────────────────────────────────

pub const Remote = struct { ip4: [4]u8, port: u16 };

/// What a peer says about itself in its `/v1/models` "machine" (behind its API key): its OS,
/// device model id and, when it leads a Thunderbolt pair, that pair's model. Bounded copies.
pub const PeerMachine = struct {
    os: Text(16) = .{},
    model: Text(32) = .{},
    leads: Text(128) = .{},
    joined: bool = false,
    free_mib: u64 = 0,
    total_mib: u64 = 0,

    fn Text(comptime n: usize) type {
        return struct {
            buf: [n]u8 = undefined,
            len: usize = 0,
            fn set(self: *@This(), v: ?std.json.Value) void {
                const str = if (v) |x| (if (x == .string) x.string else "") else "";
                self.len = @min(str.len, n);
                @memcpy(self.buf[0..self.len], str[0..self.len]);
            }
            fn get(self: *const @This()) []const u8 {
                return self.buf[0..self.len];
            }
        };
    }

    fn mib(v: ?std.json.Value) u64 {
        const x = v orelse return 0;
        return if (x == .integer and x.integer > 0) @intCast(x.integer) else 0;
    }

    pub fn parse(alloc: std.mem.Allocator, body: []const u8) PeerMachine {
        var out: PeerMachine = .{};
        var parsed = std.json.parseFromSlice(std.json.Value, alloc, body, .{}) catch return out;
        defer parsed.deinit();
        if (parsed.value != .object) return out;
        const m = parsed.value.object.get("machine") orelse return out;
        if (m != .object) return out;
        out.os.set(m.object.get("os"));
        out.model.set(m.object.get("model"));
        out.leads.set(m.object.get("leads"));
        if (m.object.get("joined")) |j| out.joined = j == .bool and j.bool;
        out.free_mib = mib(m.object.get("free_mib"));
        out.total_mib = mib(m.object.get("total_mib"));
        return out;
    }
};

fn appendPeerAddress(alloc: std.mem.Allocator, buf: *std.ArrayList(u8), display: []const u8, ip4: [4]u8, port: u16, m: *const PeerMachine) !void {
    if (buf.items.len > 0) try buf.append(alloc, ',');
    try buf.print(alloc, "{{\"name\":{f},\"url\":\"http://{d}.{d}.{d}.{d}:{d}\",\"os\":{f},\"model\":{f},\"leads\":{f},\"joined\":{},\"free_mib\":{d},\"total_mib\":{d}}}", .{
        std.json.fmt(display, .{}),             ip4[0], ip4[1], ip4[2], ip4[3], port,
        std.json.fmt(m.os.get(), .{}),          std.json.fmt(m.model.get(), .{}),
        std.json.fmt(m.leads.get(), .{}),       m.joined,
        m.free_mib,                             m.total_mib,
    });
}

/// Whether a refetched peer shows the app the same thing it did.
fn samePeerView(old: *const Peer, models: []const PeerModel, needs_key: bool, m: *const PeerMachine) bool {
    if (old.models.len != models.len or old.needs_key != needs_key or old.machine.joined != m.joined) return false;
    for (old.models, models) |a, b| if (!std.mem.eql(u8, a.id, b.id)) return false;
    return std.mem.eql(u8, old.machine.leads.get(), m.leads.get()) and
        std.mem.eql(u8, old.machine.os.get(), m.os.get()) and std.mem.eql(u8, old.machine.model.get(), m.model.get());
}

const Peer = struct {
    display: []u8, // sanitized instance name — also the hash key
    ip4: [4]u8,
    port: u16,
    models: []PeerModel,
    /// The peer answered 401: it runs with --api-key and we hold no working key.
    needs_key: bool = false,
    machine: PeerMachine = .{},

    pub fn deinit(p: *Peer, alloc: std.mem.Allocator) void {
        freePeerModels(alloc, p.models);
        alloc.free(p.display);
    }
};

const BrowseEvent = struct { name: [:0]u8, domain: [:0]u8 };

const Known = struct { domain: [:0]u8, fails: u8 = 0 };
/// Consecutive failed resolves before a service is forgotten (~4 min at the
/// 10 s refresh cadence). Until then the service keeps retrying quietly.
const KNOWN_MAX_FAILS: u8 = 24;
/// Consecutive failed resolves before an INSTALLED peer leaves the table
/// (~20-30 s at the refresh cadence). One transient dns_sd hiccup — a busy
/// mDNSResponder, an interface appearing/vanishing (VM or docker bridge), a
/// 3 s resolve timeout while the peer's GPU is pinned by a load — must not
/// evict a live peer: its cached ip4:port still tunnels, and eviction turns
/// the next chat into a "LAN peer for this model is offline" 404 (live
/// chats through a proxy alternated success/404 while the peer
/// stayed up and advertising the whole time). A genuinely-gone peer still
/// leaves within PEER_DROP_FAILS refreshes; the tunnel answers 502 honestly
/// if it is picked during the grace window.
const PEER_DROP_FAILS: u8 = 3;

const KnownFailureAction = enum { retain, drop_peer, drop_and_forget };

/// Pure policy: what a known service's consecutive-failure count (AFTER
/// incrementing for the current failure) does to the peer table + registry.
fn knownFailureAction(fails: u8) KnownFailureAction {
    if (fails >= KNOWN_MAX_FAILS) return .drop_and_forget;
    if (fails >= PEER_DROP_FAILS) return .drop_peer;
    return .retain;
}

pub const Options = struct {
    port: u16,
    /// `--lan-share` value (`all` | csv of ids); null = sharing off.
    share_spec: ?[]const u8 = null,
    /// Advertised instance name; null → hostname (".local" stripped).
    name: ?[]const u8 = null,
    discover: bool = false,
    /// `~/.mlx-serve/lan-keys.json`: the key each keyed peer requires.
    keys_path: ?[]const u8 = null,
};

pub const Lan = struct {
    alloc: std.mem.Allocator,
    port: u16,
    discover: bool,
    share: ?SharedSet = null,
    keys_path: ?[:0]u8 = null,
    name_buf: [64]u8 = undefined,
    name: []const u8 = "",
    /// Random per-process token in the TXT record — how a browser recognizes
    /// (and skips) its own advertisement.
    token_hex: [16]u8 = undefined,
    reg_ref: DNSServiceRef = null,
    thread: ?std.Thread = null,
    stop_flag: std.atomic.Value(bool) = .init(false),
    /// pthread mutex, not `std.Io.Mutex`: lookups run on conn threads and the
    /// browser thread, none of which should block through an Io handle for a
    /// micro critical section (same rationale as log.zig's sink_mutex).
    mu: os_mutex.Mutex = .{},
    peers: std.StringHashMap(Peer),
    /// Moves whenever a peer comes, goes, or changes its models or machine: the app re-reads
    /// `/v1/models` when `/props` "registry_generation" (which includes this) moves.
    changes: std.atomic.Value(u64) = .init(0),
    /// Set by `pokeDiscovery` (conn threads); consumed by the browser loop.
    refresh_asap: std.atomic.Value(bool) = .init(false),
    // Browser-thread only (events + refresh both run there — no lock):
    events: std.ArrayList(BrowseEvent) = .empty,
    /// Every service name browse has ever reported, with its consecutive
    /// resolve-failure count. THE retry mechanism: a transient resolve/fetch
    /// hiccup at ADD time must not lose the peer forever (browse won't
    /// re-announce a service that never left), so refresh re-attempts every
    /// known service — installed or not — and only a service that keeps
    /// failing (or turns out to be our own advertisement) is forgotten.
    known: std.StringHashMap(Known),

    pub fn start(alloc: std.mem.Allocator, opts: Options) !*Lan {
        if (comptime @import("builtin").os.tag == .windows) return error.LanUnsupportedOnWindows;
        const l = try alloc.create(Lan);
        errdefer alloc.destroy(l);
        l.* = .{
            .alloc = alloc,
            .port = opts.port,
            .discover = opts.discover,
            .peers = .init(alloc),
            .known = .init(alloc),
        };
        if (opts.keys_path) |p| l.keys_path = try alloc.dupeSentinel(u8, p, 0);
        errdefer if (l.keys_path) |p| alloc.free(p);
        var rnd: [8]u8 = undefined;
        std.c.arc4random_buf(&rnd, rnd.len);
        _ = std.fmt.bufPrint(&l.token_hex, "{x:0>16}", .{std.mem.readInt(u64, &rnd, .big)}) catch unreachable;

        var host_buf: [std.posix.HOST_NAME_MAX]u8 = undefined;
        var raw_name: []const u8 = opts.name orelse std.posix.gethostname(&host_buf) catch "mac";
        if (std.mem.endsWith(u8, raw_name, ".local")) raw_name = raw_name[0 .. raw_name.len - ".local".len];
        l.name = sanitizeName(&l.name_buf, raw_name);

        if (opts.share_spec) |spec| {
            var set = try SharedSet.parse(alloc, spec);
            if (set.empty()) {
                set.deinit(alloc);
                log.warn("[lan] --lan-share matched no models; sharing disabled\n", .{});
            } else {
                l.share = set;
                l.startAdvertise();
            }
        }
        // Spawn whenever sharing was REQUESTED (not only when the initial
        // registration succeeded) — the browser thread's revive loop can
        // heal a registration that failed at boot or died with the daemon.
        if (l.share != null or l.discover)
            l.thread = try std.Thread.spawn(.{}, threadMain, .{l});
        return l;
    }

    pub fn shutdown(l: *Lan) void {
        l.stop_flag.store(true, .release);
        if (l.thread) |th| th.join();
        if (l.reg_ref != null) DNSServiceRefDeallocate(l.reg_ref); // unregisters
        var it = l.peers.valueIterator();
        while (it.next()) |p| p.deinit(l.alloc);
        l.peers.deinit();
        for (l.events.items) |ev| {
            l.alloc.free(ev.name);
            l.alloc.free(ev.domain);
        }
        l.events.deinit(l.alloc);
        var kit = l.known.iterator();
        while (kit.next()) |e| {
            l.alloc.free(e.key_ptr.*);
            l.alloc.free(e.value_ptr.domain);
        }
        l.known.deinit();
        if (l.share) |*s| s.deinit(l.alloc);
        if (l.keys_path) |p| l.alloc.free(p);
        const alloc = l.alloc;
        alloc.destroy(l);
    }

    pub fn sharing(l: *const Lan) bool {
        return l.share != null;
    }

    pub fn sharedAllows(l: *const Lan, id: []const u8) bool {
        return if (l.share) |s| s.allows(id) else false;
    }

    pub const RemoteLookup = union(enum) { found: Remote, peer_unknown, model_unlisted };

    /// Three-state lookup for a `<bare>@<peer>` id. `found` → tunnel it.
    /// `model_unlisted` → the peer answered recently and does NOT offer this
    /// model: definitive, fail fast. `peer_unknown` → the peer isn't in the
    /// table (yet): offline, mid-restart, or discovery still converging —
    /// the proxy WAITS briefly and retries instead of failing instantly
    /// (live: a chat fired while the peer Mac was redeploying — or right
    /// after a local restart — got an instant misleading 404). A peer
    /// installed with an EMPTY model list (mid-boot) counts as unknown so
    /// the wait covers it too.
    pub fn lookupRemote(l: *Lan, id: []const u8) RemoteLookup {
        const rid = splitRemoteId(id) orelse return .peer_unknown;
        l.mu.lock();
        defer l.mu.unlock();
        const p = l.peers.getPtr(rid.peer) orelse return .peer_unknown;
        for (p.models) |m|
            if (std.mem.eql(u8, m.id, rid.bare)) return .{ .found = .{ .ip4 = p.ip4, .port = p.port } };
        return if (p.models.len == 0) .peer_unknown else .model_unlisted;
    }

    /// The key `peer` requires (owned), read fresh so an edit applies at once.
    pub fn peerKey(l: *const Lan, alloc: std.mem.Allocator, peer: []const u8) ?[]u8 {
        const path = l.keys_path orelse return null;
        const json = readSmallFile(alloc, path) orelse return null;
        defer alloc.free(json);
        return peerKeyFromJson(alloc, json, peer);
    }

    /// Append the names of peers that refused us for want of a key, as JSON
    /// strings (sanitized names need no escaping).
    pub fn appendPeersNeedingKey(l: *Lan, alloc: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        l.mu.lock();
        defer l.mu.unlock();
        var it = l.peers.valueIterator();
        while (it.next()) |p| if (p.needs_key) {
            if (buf.items.len > 0) try buf.append(alloc, ',');
            try buf.print(alloc, "\"{s}\"", .{p.display});
        };
    }

    /// Every reachable peer as `{"name":…,"url":"http://a.b.c.d:port"}`, comma-separated.
    pub fn appendPeerAddresses(l: *Lan, alloc: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        l.mu.lock();
        defer l.mu.unlock();
        var it = l.peers.valueIterator();
        while (it.next()) |p| try appendPeerAddress(alloc, buf, p.display, p.ip4, p.port, &p.machine);
    }

    /// Ask the browser thread to re-attempt every known service NOW instead
    /// of at the next 10 s tick — the proxy's convergence wait uses this so
    /// a rebooted peer is picked up within a poll cycle, not a refresh cycle.
    pub fn pokeDiscovery(l: *Lan) void {
        l.refresh_asap.store(true, .release);
    }

    /// Owned copy of the /v1/models entry JSON for a remote id (the
    /// load-model no-op renders it so app flows work unchanged).
    pub fn remoteEntryFor(l: *Lan, alloc: std.mem.Allocator, id: []const u8) ?[]u8 {
        const rid = splitRemoteId(id) orelse return null;
        l.mu.lock();
        defer l.mu.unlock();
        const p = l.peers.getPtr(rid.peer) orelse return null;
        for (p.models) |m|
            if (std.mem.eql(u8, m.id, rid.bare)) return alloc.dupe(u8, m.entry_json) catch null;
        return null;
    }

    /// Append every discovered remote model's entry JSON to a /v1/models
    /// `data` array under construction (comma-managed by buffer length).
    pub fn appendRemoteEntries(l: *Lan, alloc: std.mem.Allocator, buf: *std.ArrayList(u8)) !void {
        l.mu.lock();
        defer l.mu.unlock();
        var it = l.peers.valueIterator();
        while (it.next()) |p| for (p.models) |m| {
            if (buf.items.len > 0) try buf.append(alloc, ',');
            try buf.appendSlice(alloc, m.entry_json);
        };
    }

    fn startAdvertise(l: *Lan) void {
        var txt_buf: [32]u8 = undefined;
        const txt = txtBuild(&txt_buf, &l.token_hex);
        var name_z: [72]u8 = undefined;
        @memcpy(name_z[0..l.name.len], l.name);
        name_z[l.name.len] = 0;
        const err = DNSServiceRegister(&l.reg_ref, 0, 0, @ptrCast(&name_z), SERVICE_TYPE, null, null, std.mem.nativeToBig(u16, l.port), @intCast(txt.len), txt.ptr, null, null);
        if (err != 0) {
            log.warn("[lan] Bonjour registration failed ({d}); sharing not advertised{s}\n", .{ err, daemon_hint });
            l.reg_ref = null;
        } else {
            const n = if (l.share.?.all) "all models" else "selected models";
            log.info("[lan] sharing {s} as \"{s}\" ({s} port {d})\n", .{ n, l.name, SERVICE_TYPE, l.port });
        }
    }

    fn removePeer(l: *Lan, display: []const u8) void {
        l.mu.lock();
        defer l.mu.unlock();
        if (l.peers.fetchRemove(display)) |kv| {
            var p = kv.value;
            p.deinit(l.alloc);
            _ = l.changes.fetchAdd(1, .monotonic);
        }
    }

    pub fn installPeer(l: *Lan, display: []const u8, ip4: [4]u8, port: u16, models: []PeerModel, needs_key: bool, machine: PeerMachine) void {
        const owned = l.alloc.dupe(u8, display) catch {
            freePeerModels(l.alloc, models);
            return;
        };
        l.mu.lock();
        defer l.mu.unlock();
        var moved = true;
        if (l.peers.fetchRemove(display)) |kv| {
            var old = kv.value;
            moved = !samePeerView(&old, models, needs_key, &machine);
            old.deinit(l.alloc);
        }
        if (moved) _ = l.changes.fetchAdd(1, .monotonic);
        const p = Peer{ .display = owned, .ip4 = ip4, .port = port, .models = models, .needs_key = needs_key, .machine = machine };
        l.peers.put(p.display, p) catch {
            var tmp = p;
            tmp.deinit(l.alloc);
        };
    }

    const Attempt = enum { installed, self_ad, failed };

    /// Re-resolve + re-fetch one service. Serves both browse ADDs and the
    /// periodic refresh; a REMOVE event also routes here. A failure NEVER
    /// removes the peer directly — `attemptKnown` owns the removal decision
    /// (grace of PEER_DROP_FAILS consecutive failures), so one transient
    /// dns_sd hiccup can't evict a live peer whose cached ip4:port still
    /// tunnels fine.
    fn resolveAndInstall(l: *Lan, service_name: [:0]const u8, domain: [:0]const u8) Attempt {
        var disp_buf: [64]u8 = undefined;
        const display = sanitizeName(&disp_buf, service_name);

        var res: ResolveOut = .{};
        var ref: DNSServiceRef = null;
        if (DNSServiceResolve(&ref, 0, 0, service_name.ptr, SERVICE_TYPE, domain.ptr, onResolve, &res) != 0) return .failed;
        const resolved = pumpUntil(ref, &res.done, 3000);
        DNSServiceRefDeallocate(ref);
        if (!resolved) {
            log.debug("[lan] resolve timed out for \"{s}\"\n", .{display});
            return .failed;
        }
        if (std.mem.eql(u8, res.token[0..res.token_len], &l.token_hex)) return .self_ad;

        var addr: AddrOut = .{};
        var aref: DNSServiceRef = null;
        res.host[res.host_len] = 0;
        if (DNSServiceGetAddrInfo(&aref, 0, 0, kDNSServiceProtocol_IPv4, @ptrCast(&res.host), onAddr, &addr) != 0) return .failed;
        const addressed = pumpUntil(aref, &addr.done, 3000);
        DNSServiceRefDeallocate(aref);
        if (!addressed or addr.count == 0) {
            log.debug("[lan] no IPv4 for \"{s}\" host \"{s}\"\n", .{ display, res.host[0..res.host_len] });
            return .failed;
        }

        // Try each address, loopback first; the first one that ACCEPTS is the
        // peer's address from now on (the tunnel reuses it).
        var order_buf: [4][4]u8 = undefined;
        const candidates = addr.ordered(&order_buf);
        var ip4: [4]u8 = candidates[0];
        var models: ?[]PeerModel = null;
        var machine: PeerMachine = .{};
        var reachable = false;
        var needs_key = false;
        const key = l.peerKey(l.alloc, display);
        defer if (key) |k| l.alloc.free(k);
        for (candidates) |cand| {
            models = fetchPeerModels(l.alloc, cand, res.port, display, &l.token_hex, key, &machine) catch |err| switch (err) {
                error.PeerUnreachable => {
                    log.debug("[lan] \"{s}\" unreachable at {d}.{d}.{d}.{d}:{d}\n", .{ display, cand[0], cand[1], cand[2], cand[3], res.port });
                    continue;
                },
                // The fetch landed on OURSELVES (stale record of a former
                // self — resolve-time TXT check can't see it). Forget the
                // service; the real record, if any, re-registers via browse.
                error.SelfFetch => return .self_ad,
                error.Unauthorized => blk: {
                    log.debug("[lan] \"{s}\" needs an API key (lan-keys.json)\n", .{display});
                    needs_key = true;
                    break :blk null;
                },
                else => blk: {
                    // Connected but no usable answer (booting, mid-restart):
                    // keep the peer listed empty; the next refresh heals it.
                    log.debug("[lan] model fetch from \"{s}\" failed: {s}\n", .{ display, @errorName(err) });
                    break :blk null;
                },
            };
            ip4 = cand;
            reachable = true;
            break;
        }
        if (!reachable) return .failed;
        const fetched = models orelse {
            l.installPeer(display, ip4, res.port, &.{}, needs_key, .{});
            return .installed;
        };
        const changed = blk: {
            l.mu.lock();
            defer l.mu.unlock();
            const p = l.peers.getPtr(display) orelse break :blk true;
            break :blk p.models.len != fetched.len;
        };
        const count = fetched.len;
        l.installPeer(display, ip4, res.port, fetched, false, machine);
        if (changed)
            log.info("[lan] peer \"{s}\" at {d}.{d}.{d}.{d}:{d} shares {d} models\n", .{ display, ip4[0], ip4[1], ip4[2], ip4[3], res.port, count });
        return .installed;
    }

    /// Attempt one known service and keep its failure bookkeeping. Peer
    /// removal AND forgetting both happen here ONLY (self-ads forget
    /// immediately; an installed peer survives PEER_DROP_FAILS-1 transient
    /// failures; KNOWN_MAX_FAILS forgets the service) — a fresh browse ADD
    /// always re-registers.
    fn attemptKnown(l: *Lan, name: []const u8) void {
        const entry = l.known.getPtr(name) orelse return;
        const name_z = l.alloc.dupeSentinel(u8, name, 0) catch return;
        defer l.alloc.free(name_z);
        switch (l.resolveAndInstall(name_z, entry.domain)) {
            .installed => entry.fails = 0,
            .self_ad => l.forgetKnown(name),
            .failed => {
                entry.fails +|= 1;
                switch (knownFailureAction(entry.fails)) {
                    .retain => {},
                    .drop_peer => {
                        var disp_buf: [64]u8 = undefined;
                        l.removePeer(sanitizeName(&disp_buf, name));
                    },
                    .drop_and_forget => {
                        var disp_buf: [64]u8 = undefined;
                        l.removePeer(sanitizeName(&disp_buf, name));
                        l.forgetKnown(name);
                    },
                }
            },
        }
    }

    fn forgetKnown(l: *Lan, name: []const u8) void {
        if (l.known.fetchRemove(name)) |kv| {
            l.alloc.free(kv.key);
            l.alloc.free(kv.value.domain);
        }
    }

    fn refreshKnown(l: *Lan) void {
        var names: std.ArrayList([]u8) = .empty;
        defer {
            for (names.items) |n| l.alloc.free(n);
            names.deinit(l.alloc);
        }
        var it = l.known.keyIterator();
        while (it.next()) |k|
            names.append(l.alloc, l.alloc.dupe(u8, k.*) catch continue) catch break;
        for (names.items) |name| {
            if (l.stop_flag.load(.acquire)) return;
            l.attemptKnown(name);
        }
    }
};

const ResolveOut = struct {
    done: bool = false,
    port: u16 = 0,
    host: [256]u8 = undefined,
    host_len: usize = 0,
    token: [17]u8 = undefined,
    token_len: usize = 0,
};

/// Resolve callbacks fire once PER INTERFACE — only a SUCCESS may complete
/// the wait (an lo0 error arriving first must not abort a resolve that en0
/// would have answered a millisecond later; `pumpUntil`'s deadline covers
/// the all-interfaces-failed case).
fn onResolve(ref: DNSServiceRef, flags: u32, if_idx: u32, err: i32, fullname: ?[*:0]const u8, hosttarget: ?[*:0]const u8, port_be: u16, txt_len: u16, txt: ?[*]const u8, ctx: ?*anyopaque) callconv(.c) void {
    _ = ref;
    _ = flags;
    _ = if_idx;
    _ = fullname;
    const out: *ResolveOut = @ptrCast(@alignCast(ctx orelse return));
    if (err != 0 or hosttarget == null) return;
    const host = std.mem.span(hosttarget.?);
    if (host.len >= out.host.len) return;
    @memcpy(out.host[0..host.len], host);
    out.host_len = host.len;
    out.port = std.mem.bigToNative(u16, port_be);
    if (txt) |txt_ptr| {
        if (txtFind(txt_ptr[0..txt_len], "t=")) |token| {
            out.token_len = @min(token.len, out.token.len);
            @memcpy(out.token[0..out.token_len], token[0..out.token_len]);
        }
    }
    out.done = true;
}

/// A multi-homed host answers with one A record per interface — collect them
/// ALL (`MoreComing` clearing marks the batch end) so the fetch can try each.
/// Order matters downstream: loopback connects are exempt from macOS Local
/// Network privacy, so for a same-machine peer the lo0 record must win even
/// when the en0 record arrives first (live flake: en0-first resolution had
/// the SYN silently blackholed → 3 s poll timeout, peer never installed).
const AddrOut = struct {
    done: bool = false,
    count: usize = 0,
    ip4s: [4][4]u8 = undefined,

    fn add(out: *AddrOut, ip: [4]u8) void {
        for (out.ip4s[0..out.count]) |seen| if (std.mem.eql(u8, &seen, &ip)) return;
        if (out.count < out.ip4s.len) {
            out.ip4s[out.count] = ip;
            out.count += 1;
        }
    }

    /// Addresses in connect-attempt order: loopback first, then as resolved.
    fn ordered(out: *const AddrOut, buf: *[4][4]u8) []const [4]u8 {
        var n: usize = 0;
        for (out.ip4s[0..out.count]) |ip| if (ip[0] == 127) {
            buf[n] = ip;
            n += 1;
        };
        for (out.ip4s[0..out.count]) |ip| if (ip[0] != 127) {
            buf[n] = ip;
            n += 1;
        };
        return buf[0..n];
    }
};

/// Accumulates every IPv4 record; the batch completes when a callback
/// arrives with `MoreComing` clear AND at least one address landed (errors
/// alone never complete — `pumpUntil`'s deadline covers the all-failed case).
fn onAddr(ref: DNSServiceRef, flags: u32, if_idx: u32, err: i32, hostname: ?[*:0]const u8, address: ?*const std.posix.sockaddr, ttl: u32, ctx: ?*anyopaque) callconv(.c) void {
    _ = ref;
    _ = if_idx;
    _ = hostname;
    _ = ttl;
    const out: *AddrOut = @ptrCast(@alignCast(ctx orelse return));
    blk: {
        const sa = address orelse break :blk;
        if (err != 0 or sa.family != std.posix.AF.INET) break :blk;
        const sin: *const std.posix.sockaddr.in = @ptrCast(@alignCast(sa));
        out.add(@bitCast(sin.addr));
    }
    if (flags & kDNSServiceFlagsMoreComing == 0 and out.count > 0) out.done = true;
}

fn onBrowse(ref: DNSServiceRef, flags: u32, if_idx: u32, err: i32, name: ?[*:0]const u8, regtype: ?[*:0]const u8, domain: ?[*:0]const u8, ctx: ?*anyopaque) callconv(.c) void {
    _ = ref;
    _ = if_idx;
    _ = regtype;
    if (err != 0) return;
    const l: *Lan = @ptrCast(@alignCast(ctx orelse return));
    // ADD and REMOVE both route through resolveAndInstall (see its doc), so
    // the event only needs the service identity.
    log.debug("[lan] browse event: \"{s}\" (flags 0x{x})\n", .{ std.mem.span(name orelse return), flags });
    const n = l.alloc.dupeSentinel(u8, std.mem.span(name.?), 0) catch return;
    const d = l.alloc.dupeSentinel(u8, std.mem.span(domain orelse "local."), 0) catch {
        l.alloc.free(n);
        return;
    };
    l.events.append(l.alloc, .{ .name = n, .domain = d }) catch {
        l.alloc.free(n);
        l.alloc.free(d);
    };
}

/// How often a dead dns_sd ref (browse or advertise) is re-created. Also the
/// retry cadence while mDNSResponder itself is down.
const REVIVE_INTERVAL_MS: i64 = 5_000;

fn threadMain(l: *Lan) void {
    var browse_ref: DNSServiceRef = null;
    defer if (browse_ref != null) DNSServiceRefDeallocate(browse_ref);

    var last_refresh = monoMs();
    var keys_stamp: i64 = if (l.keys_path) |p| fileStamp(p) else 0;
    // Dead refs are REVIVED, never left null forever: an mDNSResponder
    // restart (macOS update, daemon crash) or a sleep/wake cycle can
    // invalidate every dns_sd connection at once, and a permanently-dead
    // browse leaves this server blind to peers that are up and advertising
    // — every remote chat then 404s "peer offline" until a manual restart.
    // `revive_at = 0` makes the first loop iteration do the initial starts.
    var revive_at: i64 = 0;
    while (!l.stop_flag.load(.acquire)) {
        const now_ms = monoMs();
        if (now_ms >= revive_at) {
            revive_at = now_ms + REVIVE_INTERVAL_MS;
            if (l.discover and browse_ref == null) {
                if (DNSServiceBrowse(&browse_ref, 0, 0, SERVICE_TYPE, null, onBrowse, l) != 0) {
                    log.warn("[lan] Bonjour browse failed to start; retrying in {d} s{s}\n", .{ @divTrunc(REVIVE_INTERVAL_MS, 1000), daemon_hint });
                    browse_ref = null;
                } else {
                    log.info("[lan] discovering peers ({s})\n", .{SERVICE_TYPE});
                }
            }
            // Sharing was requested but the registration is gone (failed at
            // boot, or the daemon dropped it) — re-advertise, or peers see
            // this host vanish while it keeps serving.
            if (l.share != null and l.reg_ref == null) l.startAdvertise();
        }
        var fds: [2]std.posix.pollfd = undefined;
        var refs: [2]DNSServiceRef = undefined;
        var n: usize = 0;
        for ([_]DNSServiceRef{ l.reg_ref, browse_ref }) |r| {
            if (r != null) {
                fds[n] = .{ .fd = DNSServiceRefSockFD(r), .events = std.posix.POLL.IN, .revents = 0 };
                refs[n] = r;
                n += 1;
            }
        }
        if (n == 0) {
            crt.sleepMs(500); // this thread has no Io
            continue;
        }
        const ready = std.posix.poll(fds[0..n], 1000) catch break;
        if (ready > 0) for (fds[0..n], refs[0..n]) |fd, r| {
            if (fd.revents == 0) continue;
            // IN with a clean ProcessResult = normal traffic. Anything else
            // (ProcessResult error, or HUP/ERR/NVAL with no data) means the
            // daemon dropped this connection — tear the ref down so the
            // revive above re-creates it, instead of hot-spinning on a dead
            // fd or silently losing discovery/advertising.
            const alive = fd.revents & std.posix.POLL.IN != 0 and DNSServiceProcessResult(r) == 0;
            if (alive) continue;
            if (r == browse_ref) {
                log.warn("[lan] dns_sd browse connection lost; will re-browse\n", .{});
                DNSServiceRefDeallocate(browse_ref);
                browse_ref = null;
            } else if (r == l.reg_ref) {
                log.warn("[lan] dns_sd advertise connection lost; will re-register\n", .{});
                DNSServiceRefDeallocate(l.reg_ref);
                l.reg_ref = null;
            }
        };
        // Drain browse events (appended by onBrowse inside ProcessResult —
        // same thread, so plain iteration is safe even though the resolves
        // below block for seconds). ADD and REMOVE both upsert `known` and
        // attempt immediately; resolvability decides what survives.
        while (l.events.pop()) |ev| {
            defer l.alloc.free(ev.name);
            if (!l.known.contains(ev.name)) blk: {
                const key = l.alloc.dupe(u8, ev.name) catch break :blk;
                l.known.put(key, .{ .domain = ev.domain }) catch {
                    l.alloc.free(key);
                    break :blk;
                };
                l.attemptKnown(ev.name);
                continue;
            }
            l.alloc.free(ev.domain);
            l.attemptKnown(ev.name);
        }
        const now = monoMs();
        var poked = l.refresh_asap.swap(false, .acq_rel);
        // A saved key retries its peer now, not at the next refresh.
        if (l.keys_path) |p| {
            const stamp = fileStamp(p);
            if (stamp != keys_stamp) poked = true;
            keys_stamp = stamp;
        }
        if (l.discover and (poked or now - last_refresh > 10_000)) {
            last_refresh = now;
            l.refreshKnown();
        }
    }
}

const FileStat = struct { mtime_ns: i64, regular: bool };

/// `path` relative to the cwd, or the open `fd` itself when `path` is empty.
/// `std.c.Stat` is Darwin-only in Zig 0.17, so Linux reads `statx`.
fn statFile(fd: std.c.fd_t, path: [*:0]const u8) ?FileStat {
    if (comptime @import("builtin").os.tag == .linux) {
        const linux = std.os.linux;
        var sx: linux.Statx = undefined;
        const flags: u32 = if (path[0] == 0) linux.AT.EMPTY_PATH else 0;
        if (linux.errno(linux.statx(fd, path, flags, .{ .TYPE = true, .MTIME = true }, &sx)) != .SUCCESS) return null;
        return .{ .mtime_ns = sx.mtime.sec * 1_000_000_000 + sx.mtime.nsec, .regular = linux.S.ISREG(sx.mode) };
    }
    var st: std.c.Stat = undefined;
    if ((if (path[0] == 0) std.c.fstat(fd, &st) else std.c.stat(path, &st)) != 0) return null;
    const m = st.mtime();
    return .{
        .mtime_ns = @as(i64, @intCast(m.sec)) * 1_000_000_000 + @as(i64, @intCast(m.nsec)),
        .regular = st.mode & std.c.S.IFMT == std.c.S.IFREG,
    };
}

/// mtime in ns, 0 when the file is absent.
fn fileStamp(path: [*:0]const u8) i64 {
    const st = statFile(std.c.AT.FDCWD, path) orelse return 0;
    return st.mtime_ns;
}

/// Regular files only, opened non-blocking: a FIFO at the path would park the
/// browser thread in `open` until a writer came, and discovery with it.
fn readSmallFile(alloc: std.mem.Allocator, path: [*:0]const u8) ?[]u8 {
    if (comptime @import("builtin").os.tag == .windows) return null; // LAN is off on Windows
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NONBLOCK = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const st = statFile(fd, "") orelse return null;
    if (!st.regular) return null;
    var buf: std.ArrayList(u8) = .empty;
    var chunk: [4096]u8 = undefined;
    while (buf.items.len < 64 * 1024) {
        const n = readFd(fd, &chunk) catch break;
        if (n == 0) return buf.toOwnedSlice(alloc) catch break;
        buf.appendSlice(alloc, chunk[0..n]) catch break;
    }
    buf.deinit(alloc);
    return null;
}

/// `peer`'s key out of lan-keys.json (`{"<peer>": "<key>"}`, written by the
/// app). Null when absent, or when it could break the header it rides in.
pub fn peerKeyFromJson(alloc: std.mem.Allocator, json: []const u8, peer: []const u8) ?[]u8 {
    var parsed = std.json.parseFromSlice(std.json.Value, alloc, json, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const v = parsed.value.object.get(peer) orelse return null;
    if (v != .string or v.string.len == 0) return null;
    for (v.string) |c| if (c < 0x20 or c == 0x7f) return null;
    return alloc.dupe(u8, v.string) catch null;
}

/// The `Authorization` header line for a keyed peer; nothing without a key.
const AuthLine = struct {
    key: ?[]const u8,
    pub fn format(self: AuthLine, w: *std.Io.Writer) std.Io.Writer.Error!void {
        if (self.key) |k| try w.print("Authorization: Bearer {s}\r\n", .{k});
    }
};

/// Monotonic milliseconds without an `Io` handle (this thread has none —
/// same rationale as log.zig's raw-libc sink).
const monoMs = crt.monotonicMs;

/// One dns_sd ref pumped until its callback flips `done` or the deadline hits.
fn pumpUntil(ref: DNSServiceRef, done: *const bool, timeout_ms: i64) bool {
    if (done.*) return true; // answered synchronously (the Linux GetAddrInfo in ane_stub.c)
    const fd = DNSServiceRefSockFD(ref);
    if (fd < 0) return false;
    const deadline = monoMs() + timeout_ms;
    while (!done.*) {
        const remain = deadline - monoMs();
        if (remain <= 0) return false;
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const ready = std.posix.poll(&fds, @intCast(@min(remain, 1000))) catch return false;
        if (ready > 0 and DNSServiceProcessResult(ref) != 0) return false;
    }
    return true;
}

// ── Outbound sockets (raw libc — self-contained, no std.Io runtime) ──

const fd_t = std.c.fd_t;

/// Non-blocking connect with a real deadline: a blocking connect to a
/// powered-off host can hang for the kernel's full SYN-retry budget (~75 s).
pub fn connectTimeout(ip4: [4]u8, port: u16, timeout_ms: i32) !fd_t {
    const fd = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (fd < 0) return error.PeerUnreachable;
    errdefer _ = std.c.close(fd);
    const nonblock: c_int = @bitCast(std.c.O{ .NONBLOCK = true });
    const flags = std.c.fcntl(fd, std.c.F.GETFL);
    _ = std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock);
    var sa: std.posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip4) };
    if (std.c.connect(fd, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0) {
        const eno = std.c._errno().*;
        if (eno != @intFromEnum(std.c.E.INPROGRESS)) {
            log.debug("[lan] connect({d}.{d}.{d}.{d}:{d}) errno={d}\n", .{ ip4[0], ip4[1], ip4[2], ip4[3], port, eno });
            return error.PeerUnreachable;
        }
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.OUT, .revents = 0 }};
        const ready = std.posix.poll(&fds, timeout_ms) catch return error.PeerUnreachable;
        if (ready == 0) {
            log.debug("[lan] connect({d}.{d}.{d}.{d}:{d}) poll timeout\n", .{ ip4[0], ip4[1], ip4[2], ip4[3], port });
            return error.PeerUnreachable;
        }
        var so_err: c_int = 0;
        var so_len: std.posix.socklen_t = @sizeOf(c_int);
        if (std.c.getsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.ERROR, &so_err, &so_len) != 0 or so_err != 0) {
            log.debug("[lan] connect({d}.{d}.{d}.{d}:{d}) SO_ERROR={d}\n", .{ ip4[0], ip4[1], ip4[2], ip4[3], port, so_err });
            return error.PeerUnreachable;
        }
    }
    _ = std.c.fcntl(fd, std.c.F.SETFL, flags); // back to blocking
    return fd;
}

fn writeAllFd(fd: fd_t, data: []const u8) !void {
    var off: usize = 0;
    while (off < data.len) {
        const n = std.c.write(fd, data.ptr + off, data.len - off);
        if (n <= 0) return error.PeerUnreachable;
        off += @intCast(n);
    }
}

/// Read once; 0 on EOF, error on failure/timeout.
fn readFd(fd: fd_t, buf: []u8) !usize {
    const n = std.c.read(fd, buf.ptr, buf.len);
    if (n < 0) return error.ReadFailed;
    return @intCast(n);
}

/// Case-insensitive header lookup in a raw HTTP head. `name_lower` must be
/// lowercase; matches at line starts only. Returns the trimmed value.
pub fn headerValueCI(head: []const u8, name_lower: []const u8) ?[]const u8 {
    var it = std.mem.splitSequence(u8, head, "\r\n");
    while (it.next()) |line| {
        if (line.len < name_lower.len + 1) continue;
        var matches = true;
        for (name_lower, 0..) |c, i| {
            if (std.ascii.toLower(line[i]) != c) {
                matches = false;
                break;
            }
        }
        if (!matches or line[name_lower.len] != ':') continue;
        return std.mem.trim(u8, line[name_lower.len + 1 ..], " \t");
    }
    return null;
}

/// Discovery fetch: GET the peer's (shared-filtered) /v1/models, with its key
/// when we hold one; `error.Unauthorized` when a keyed peer refuses us.
/// `error.SelfFetch` when the response carries OUR OWN process token
/// (`X-MLX-LAN-Token`): a stale Bonjour record of a former self — same name
/// and port, different TXT token, so the resolve-time TXT check can't catch
/// it — resolves back to this very server, and the loopback-first fetch
/// would happily install our own models as a "peer" (live test_lan_share
/// self-mirror after the peer-restart section).
fn fetchPeerModels(alloc: std.mem.Allocator, ip4: [4]u8, port: u16, peer_display: []const u8, own_token: []const u8, key: ?[]const u8, machine: ?*PeerMachine) ![]PeerModel {
    const fd = try connectTimeout(ip4, port, 3000);
    defer _ = std.c.close(fd);
    const tv = std.c.timeval{ .sec = 5, .usec = 0 };
    std.posix.setsockopt(fd, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&tv)) catch {};
    const req = try std.fmt.allocPrint(alloc, "GET /v1/models HTTP/1.1\r\nHost: mlx-serve\r\nConnection: close\r\nX-MLX-LAN: 1\r\n{f}\r\n", .{AuthLine{ .key = key }});
    defer alloc.free(req);
    try writeAllFd(fd, req);
    var resp: std.ArrayList(u8) = .empty;
    defer resp.deinit(alloc);
    var chunk: [16 * 1024]u8 = undefined;
    while (resp.items.len < 8 * 1024 * 1024) {
        const n = readFd(fd, &chunk) catch break;
        if (n == 0) break;
        try resp.appendSlice(alloc, chunk[0..n]);
    }
    const raw = resp.items;
    const line_end = std.mem.indexOf(u8, raw, "\r\n") orelse return error.BadPeerJson;
    if (std.mem.indexOf(u8, raw[0..line_end], " 401") != null) return error.Unauthorized;
    if (std.mem.indexOf(u8, raw[0..line_end], " 200") == null) return error.BadPeerJson;
    const header_end = std.mem.indexOf(u8, raw, "\r\n\r\n") orelse return error.BadPeerJson;
    if (headerValueCI(raw[0..header_end], "x-mlx-lan-token")) |tok| {
        if (std.mem.eql(u8, tok, own_token)) return error.SelfFetch;
    }
    if (machine) |m| m.* = PeerMachine.parse(alloc, raw[header_end + 4 ..]);
    return parsePeerModels(alloc, raw[header_end + 4 ..], peer_display);
}

/// Proxy one request to `remote` and pump the response back byte-for-byte.
/// `conn` needs `writeAll([]const u8) !void` + `peerClosed() bool` — the
/// server's `*Conn` fits, and the duck typing keeps this file
/// server-independent and the pump hermetically testable.
/// `error.PeerUnreachable` is returned BEFORE anything is written to `conn`
/// (the caller can still send a clean 502); any later failure just ends the
/// stream — the client sees a closed socket, the peer sees a disconnect and
/// cancels its slot.
pub fn tunnel(remote: Remote, method: []const u8, raw_path: []const u8, body: []const u8, key: ?[]const u8, conn: anytype) error{PeerUnreachable}!void {
    if (comptime @import("builtin").os.tag == .windows) return error.PeerUnreachable; // LAN is off on Windows
    const fd = connectTimeout(remote.ip4, remote.port, 3000) catch return error.PeerUnreachable;
    defer _ = std.c.close(fd);
    var head_buf: [2048]u8 = undefined;
    const head = std.fmt.bufPrint(
        &head_buf,
        "{s} {s} HTTP/1.1\r\nHost: {d}.{d}.{d}.{d}:{d}\r\nContent-Type: application/json\r\nAccept: */*\r\nConnection: close\r\nX-MLX-LAN: 1\r\n{f}Content-Length: {d}\r\n\r\n",
        .{ method, raw_path, remote.ip4[0], remote.ip4[1], remote.ip4[2], remote.ip4[3], remote.port, AuthLine{ .key = key }, body.len },
    ) catch return error.PeerUnreachable;
    writeAllFd(fd, head) catch return error.PeerUnreachable;
    writeAllFd(fd, body) catch return error.PeerUnreachable;

    // Pump peer → client until peer EOF. The 1 s poll tick doubles as the
    // client-disconnect probe so an abandoned generation is torn down on the
    // peer too (its own disconnect-cancel machinery fires when we close).
    var buf: [16 * 1024]u8 = undefined;
    while (true) {
        var fds = [_]std.posix.pollfd{.{ .fd = fd, .events = std.posix.POLL.IN, .revents = 0 }};
        const ready = std.posix.poll(&fds, 1000) catch return;
        if (ready == 0) {
            if (conn.peerClosed()) return;
            continue;
        }
        const n = readFd(fd, &buf) catch return;
        if (n == 0) return;
        conn.writeAll(buf[0..n]) catch return;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

const t = std.testing;

test "lan: peer address lookup answers on Linux, where Avahi lacks GetAddrInfo" {
    if (comptime @import("builtin").os.tag != .linux) return error.SkipZigTest;
    var addr: AddrOut = .{};
    var aref: DNSServiceRef = null;
    try std.testing.expectEqual(@as(i32, 0), DNSServiceGetAddrInfo(&aref, 0, 0, kDNSServiceProtocol_IPv4, "localhost", onAddr, &addr));
    try std.testing.expect(pumpUntil(aref, &addr.done, 1000));
    try std.testing.expectEqualSlices(u8, &.{ 127, 0, 0, 1 }, &addr.ip4s[0]);
}

test "lan: headerValueCI finds a header case-insensitively and trims the value" {
    const head = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nX-MLX-LAN-Token: deadbeefcafef00d\r\n";
    try t.expectEqualStrings("deadbeefcafef00d", headerValueCI(head, "x-mlx-lan-token").?);
    try t.expectEqualStrings("application/json", headerValueCI(head, "content-type").?);
    try t.expect(headerValueCI(head, "x-missing") == null);
    // Name must match at line start, not mid-header.
    try t.expect(headerValueCI("X-Foo-Bar: 1\r\n", "bar") == null);
}

test "lan: splitRemoteId splits on the LAST @ and rejects degenerate forms" {
    const r = splitRemoteId("gemma-4-e4b-it-4bit@Davids-Mac").?;
    try t.expectEqualStrings("gemma-4-e4b-it-4bit", r.bare);
    try t.expectEqualStrings("Davids-Mac", r.peer);
    // A bare id that itself contains '@' resolves to the last suffix.
    const r2 = splitRemoteId("weird@name@peer1").?;
    try t.expectEqualStrings("weird@name", r2.bare);
    try t.expectEqualStrings("peer1", r2.peer);
    try t.expect(splitRemoteId("no-at-here") == null);
    try t.expect(splitRemoteId("@peer") == null);
    try t.expect(splitRemoteId("model@") == null);
    try t.expect(splitRemoteId("") == null);
}

test "lan: transient resolve failures retain a live peer; only persistent failure drops it" {
    // Grace policy for the browse thread's failure bookkeeping. dns_sd
    // resolves hiccup transiently on a LIVE peer (busy mDNSResponder, a
    // VM/docker bridge interface appearing or vanishing mid-toggle, a 3 s
    // resolve timeout while the peer's GPU is pinned) — one such hiccup
    // must NOT evict the peer from the table: the entry's cached ip4:port
    // still tunnels fine, and eviction turns the next chat into a
    // user-visible "LAN peer for this model is offline" 404. Only a
    // PERSISTENT failure streak drops the peer, and only KNOWN_MAX_FAILS
    // forgets the service name entirely.
    try t.expectEqual(KnownFailureAction.retain, knownFailureAction(1));
    try t.expectEqual(KnownFailureAction.retain, knownFailureAction(PEER_DROP_FAILS - 1));
    try t.expectEqual(KnownFailureAction.drop_peer, knownFailureAction(PEER_DROP_FAILS));
    try t.expectEqual(KnownFailureAction.drop_peer, knownFailureAction(KNOWN_MAX_FAILS - 1));
    try t.expectEqual(KnownFailureAction.drop_and_forget, knownFailureAction(KNOWN_MAX_FAILS));
    try t.expectEqual(KnownFailureAction.drop_and_forget, knownFailureAction(255));
}

test "lan: routeClass allows exactly the shared inference surface" {
    // Open probes.
    try t.expectEqual(RouteClass.open, routeClass("OPTIONS", "/v1/chat/completions"));
    try t.expectEqual(RouteClass.open, routeClass("GET", "/health"));
    try t.expectEqual(RouteClass.open, routeClass("GET", "/v1/models"));
    try t.expectEqual(RouteClass.open, routeClass("GET", "/api/version"));
    try t.expectEqual(RouteClass.open, routeClass("GET", "/launch"));
    // Inference is model-gated on every surface, media included.
    for ([_][]const u8{
        "/v1/chat/completions",        "/v1/completions",
        "/v1/messages",                "/v1/responses",
        "/v1/embeddings",              "/v1/images/generations",
        "/v1/images/edits",            "/v1/audio/speech",
        "/v1/audio/music-generations", "/v1/audio/sound-generations",
        "/v1/video/generations",       "/v1/3d/generations",
        "/api/chat",                   "/api/generate",
        "/api/embed",                  "/api/embeddings",
        "/v1/audio/transcriptions",
    }) |p| try t.expectEqual(RouteClass.model_gated, routeClass("POST", p));
    // Admin/host-local stays denied.
    try t.expectEqual(RouteClass.denied, routeClass("POST", "/v1/load-model"));
    try t.expectEqual(RouteClass.denied, routeClass("POST", "/v1/unload-model"));
    try t.expectEqual(RouteClass.denied, routeClass("POST", "/v1/responses/compact"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/metrics"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/metrics.json"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/v1/responses/resp_123"));
    try t.expectEqual(RouteClass.denied, routeClass("DELETE", "/v1/responses/resp_123"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/api/tags"));
    try t.expectEqual(RouteClass.denied, routeClass("POST", "/api/pull"));
    try t.expectEqual(RouteClass.denied, routeClass("GET", "/props"));
}

test "lan: SharedSet parses all|csv and matches basename-tolerantly" {
    const a = t.allocator;
    var all = try SharedSet.parse(a, "all");
    defer all.deinit(a);
    try t.expect(all.all);
    try t.expect(!all.empty());
    try t.expect(all.allows("anything"));

    var set = try SharedSet.parse(a, "gemma-4-e4b-it-4bit, mlx-community/bge-small-en-v1.5-8bit");
    defer set.deinit(a);
    try t.expect(!set.empty());
    // Exact.
    try t.expect(set.allows("gemma-4-e4b-it-4bit"));
    // Registry id is org/name, share entry is the basename.
    try t.expect(set.allows("some-org/gemma-4-e4b-it-4bit"));
    // Share entry is org/name, registry id is the basename.
    try t.expect(set.allows("bge-small-en-v1.5-8bit"));
    try t.expect(!set.allows("qwen3.6-27b"));
    try t.expect(!set.allows(""));

    var none = try SharedSet.parse(a, " , ");
    defer none.deinit(a);
    try t.expect(none.empty());
    try t.expect(!none.allows("gemma-4-e4b-it-4bit"));
}

test "lan: sanitizeName collapses hostile chars and never returns empty" {
    var buf: [64]u8 = undefined;
    try t.expectEqualStrings("David-s-MacBook-Pro", sanitizeName(&buf, "David's MacBook Pro"));
    try t.expectEqualStrings("mac-2", sanitizeName(&buf, "mac (2)"));
    // '@' must never survive — it's the remote-id delimiter.
    try t.expectEqualStrings("a-b.local", sanitizeName(&buf, "a@b.local"));
    try t.expectEqualStrings("mac", sanitizeName(&buf, "!!!"));
    try t.expectEqualStrings("mac", sanitizeName(&buf, ""));
    try t.expectEqualStrings("plain-name_1.local", sanitizeName(&buf, "plain-name_1.local"));
}

test "lan: rewriteModelValue refuses a model value that is not a slice of the body" {
    const a = t.allocator;
    const body = "{\"model\":\"org\\/m@Studio\",\"messages\":[]}";
    const raw = body[10 .. 10 + "org\\/m@Studio".len];
    var canon_buf: [64]u8 = undefined;
    const canon = unescapeJsonSlashes(&canon_buf, raw);
    try t.expectError(error.ModelNotInBody, rewriteModelValue(a, body, canon, "org/m"));
    const out = try rewriteModelValue(a, body, raw, "org/m");
    defer a.free(out);
    try t.expectEqualStrings("{\"model\":\"org/m\",\"messages\":[]}", out);
}

test "lan: rewriteModelValue refuses an unescaped Windows path id" {
    const body = "{\"model\":\"C:\\\\m@local\"}";
    const canonical = "C:\\m@local"; // an unescaped copy, as handleConnection used to pass
    try t.expectError(error.ModelNotInBody, rewriteModelValue(t.allocator, body, canonical, "C:\\m"));
}

test "lan: rewriteModelValue splices the aliased model value in place" {
    const a = t.allocator;
    const body = "{\"model\":\"qwen3.6-27b@Studio\",\"messages\":[{\"role\":\"user\",\"content\":\"hi @Studio\"}]}";
    const val_start = std.mem.indexOf(u8, body, "qwen3.6-27b@Studio").?;
    const model_value = body[val_start .. val_start + "qwen3.6-27b@Studio".len];
    const out = try rewriteModelValue(a, body, model_value, "qwen3.6-27b");
    defer a.free(out);
    try t.expectEqualStrings(
        "{\"model\":\"qwen3.6-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi @Studio\"}]}",
        out,
    );
}

test "lan: TXT record round-trips the instance token" {
    var buf: [64]u8 = undefined;
    const txt = txtBuild(&buf, "deadbeefcafef00d");
    try t.expect(txt.len > 0);
    try t.expectEqualStrings("deadbeefcafef00d", txtFind(txt, "t=").?);
    try t.expectEqualStrings("1", txtFind(txt, "v=").?);
    try t.expect(txtFind(txt, "x=") == null);
    // Truncated/hostile TXT never panics.
    try t.expect(txtFind(&[_]u8{200}, "t=") == null);
    try t.expect(txtFind(&[_]u8{}, "t=") == null);
}

test "lan: parsePeerModels rewrites ids, adds lan_peer, keeps meta" {
    const a = t.allocator;
    const body =
        \\{"object":"list","data":[
        \\ {"id":"gemma-4-e4b-it-4bit","object":"model","loaded":true,"capabilities":["chat","vision"],"meta":{"context_length":94000}},
        \\ {"alias":"klein","id":"flux2-klein-4bit","object":"model","loaded":false,"capabilities":["image"]},
        \\ {"id":"qwen3.6-27b@SomeoneElse","object":"model","loaded":true,"lan_peer":"SomeoneElse"},
        \\ {"id":42,"object":"junk"}
        \\]}
    ;
    // The lan_peer entry above is the peer's OWN remote stub — a peer that
    // fails to filter its list (old binary, loopback fetch between two
    // servers on one Mac) would otherwise chain re-exports (`@a@b` ids;
    // a third server on the test Mac leaked its mirrors
    // into test_lan_share's servers). Never mirror someone else's mirror.
    const models = try parsePeerModels(a, body, "Studio");
    defer freePeerModels(a, models);
    try t.expectEqual(@as(usize, 2), models.len);
    try t.expectEqualStrings("gemma-4-e4b-it-4bit", models[0].id);
    // Entry JSON carries the suffixed id + the lan_peer badge + original meta.
    try t.expect(std.mem.indexOf(u8, models[0].entry_json, "\"id\":\"gemma-4-e4b-it-4bit@Studio\"") != null);
    try t.expect(std.mem.indexOf(u8, models[0].entry_json, "\"lan_peer\":\"Studio\"") != null);
    try t.expect(std.mem.indexOf(u8, models[0].entry_json, "\"context_length\":94000") != null);
    // A peer on an older build sent no agent budget: the mirror carries the one its context implies.
    try t.expect(std.mem.indexOf(u8, models[0].entry_json, "\"agent_budget\":{\"context\":94000,\"output\":47000,\"compaction_reserve\":20000,\"too_small_for\":{}}") != null);
    try t.expect(std.mem.indexOf(u8, models[1].entry_json, "agent_budget") == null);
    try t.expect(std.mem.indexOf(u8, models[1].entry_json, "\"id\":\"flux2-klein-4bit@Studio\"") != null);
    // The peer's alias resolves only on the peer, so the mirror does not list it.
    try t.expect(std.mem.indexOf(u8, models[1].entry_json, "alias") == null);
    // Not an mlx-serve shape → error, not a crash.
    try t.expectError(error.BadPeerJson, parsePeerModels(a, "{\"nope\":true}", "x"));
    try t.expectError(error.BadPeerJson, parsePeerModels(a, "not json", "x"));
}

test "lan: JSON-escaped slashes canonicalize (Swift clients send org\\/name)" {
    var buf: [256]u8 = undefined;
    // JSONSerialization (Swift, PHP, …) legally escapes '/' as '\/'. The org/
    // prefix of a remote id then misses the byte-compare — live 404 "no longer
    // shares this model" from the app on ddalcu\/gemma-4-e2b…@Davids-Mac-mini.
    try t.expectEqualStrings("ddalcu/gemma-e2b@Mini", unescapeJsonSlashes(&buf, "ddalcu\\/gemma-e2b@Mini"));
    // No escapes → the INPUT slice comes back verbatim (zero-copy).
    const plain = "ddalcu/gemma-e2b@Mini";
    try t.expect(unescapeJsonSlashes(&buf, plain).ptr == plain.ptr);
    // Only the escapes `\/` and `\\` collapse; other backslashes survive.
    try t.expectEqualStrings("a\\b/c", unescapeJsonSlashes(&buf, "a\\b\\/c"));
    // A Windows path id arrives with its backslashes escaped.
    try t.expectEqualStrings("C:\\Users\\m", unescapeJsonSlashes(&buf, "C:\\\\Users\\\\m"));
    // Oversized input degrades to verbatim rather than truncating.
    var tiny: [4]u8 = undefined;
    try t.expectEqualStrings("x\\/y", unescapeJsonSlashes(tiny[0..2], "x\\/y"));
}

test "lan: lookupRemote distinguishes found / unlisted / unknown" {
    const a = t.allocator;
    var l = Lan{ .alloc = a, .port = 0, .discover = true, .peers = .init(a), .known = .init(a) };
    defer {
        var it = l.peers.valueIterator();
        while (it.next()) |p| p.deinit(a);
        l.peers.deinit();
        l.known.deinit();
    }

    // Unknown peer (and non-remote ids) → unknown: the proxy waits for
    // discovery to converge instead of failing instantly.
    try t.expect(l.lookupRemote("gemma@ghost") == .peer_unknown);
    try t.expect(l.lookupRemote("local-model") == .peer_unknown);

    const models = try a.alloc(PeerModel, 1);
    models[0] = .{ .id = try a.dupe(u8, "gemma"), .entry_json = try a.dupe(u8, "{}") };
    l.installPeer("studio", .{ 127, 0, 0, 1 }, 1234, models, false, .{});
    try t.expect(l.lookupRemote("gemma@studio") == .found);
    // The peer answered recently and does NOT offer this model — definitive,
    // fail fast (probes/typos must not burn the wait).
    try t.expect(l.lookupRemote("other@studio") == .model_unlisted);

    // A mid-boot empty install (peer reachable, models not served yet)
    // counts as unknown so the wait covers it too.
    l.installPeer("booting", .{ 127, 0, 0, 1 }, 1235, &.{}, false, .{});
    try t.expect(l.lookupRemote("anything@booting") == .peer_unknown);
}

test "lan: peerKeyFromJson reads one peer's key and refuses header-breaking values" {
    const a = t.allocator;
    const json = "{\"Studio\":\"s3cret\",\"Empty\":\"\",\"Num\":5,\"Evil\":\"a\\r\\nX-Injected: 1\"}";
    const k = peerKeyFromJson(a, json, "Studio").?;
    defer a.free(k);
    try t.expectEqualStrings("s3cret", k);
    try t.expect(peerKeyFromJson(a, json, "Missing") == null);
    try t.expect(peerKeyFromJson(a, json, "Empty") == null);
    try t.expect(peerKeyFromJson(a, json, "Num") == null);
    try t.expect(peerKeyFromJson(a, json, "Evil") == null);
    try t.expect(peerKeyFromJson(a, "not json", "Studio") == null);
}

test "lan: a keys path that is not a regular file reads as absent without blocking" {
    var buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "/tmp/mlx-serve-lan-fifo-{d}", .{std.c.getpid()}, 0);
    const C = struct {
        extern "c" fn mkfifo(path: [*:0]const u8, mode: std.c.mode_t) c_int;
    };
    _ = std.c.unlink(path);
    try t.expect(C.mkfifo(path, 0o600) == 0);
    defer _ = std.c.unlink(path);
    try t.expect(readSmallFile(t.allocator, path) == null);
    try t.expect(readSmallFile(t.allocator, "/tmp") == null);
}

test "lan: fileStamp is a file's mtime, and 0 when it is absent" {
    var buf: [64]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "/tmp/mlx-serve-lan-stamp-{d}", .{std.c.getpid()}, 0);
    _ = std.c.unlink(path);
    try t.expectEqual(@as(i64, 0), fileStamp(path));
    const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(std.c.mode_t, 0o600));
    try t.expect(fd >= 0);
    _ = std.c.close(fd);
    defer _ = std.c.unlink(path);
    try t.expect(fileStamp(path) > 0);
}

test "lan: a peer that answered 401 is listed as needing a key" {
    const a = t.allocator;
    var l = Lan{ .alloc = a, .port = 0, .discover = true, .peers = .init(a), .known = .init(a) };
    defer {
        var it = l.peers.valueIterator();
        while (it.next()) |p| p.deinit(a);
        l.peers.deinit();
        l.known.deinit();
    }
    l.installPeer("open", .{ 127, 0, 0, 1 }, 1, &.{}, false, .{});
    l.installPeer("locked", .{ 127, 0, 0, 1 }, 2, &.{}, true, .{});
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try l.appendPeersNeedingKey(a, &buf);
    try t.expectEqualStrings("\"locked\"", buf.items);
}

/// Duck-typed stand-in for server.Conn in tunnel tests.
const TestSink = struct {
    alloc: std.mem.Allocator,
    buf: std.ArrayList(u8) = .empty,

    pub fn writeAll(self: *TestSink, data: []const u8) !void {
        try self.buf.appendSlice(self.alloc, data);
    }
    pub fn peerClosed(self: *TestSink) bool {
        _ = self;
        return false;
    }
};

fn testListener(port_out: *u16) !fd_t {
    const lst = std.c.socket(std.posix.AF.INET, std.posix.SOCK.STREAM, 0);
    if (lst < 0) return error.Sock;
    errdefer _ = std.c.close(lst);
    var sa: std.posix.sockaddr.in = .{ .port = 0, .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    if (std.c.bind(lst, @ptrCast(&sa), @sizeOf(std.posix.sockaddr.in)) != 0) return error.Sock;
    var sa_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr.in);
    if (std.c.getsockname(lst, @ptrCast(&sa), &sa_len) != 0) return error.Sock;
    port_out.* = std.mem.bigToNative(u16, sa.port);
    if (std.c.listen(lst, 1) != 0) return error.Sock;
    return lst;
}

test "lan: tunnel forwards the rewritten request and pumps a chunked streaming response" {
    const a = t.allocator;
    var port: u16 = 0;
    const lst = try testListener(&port);
    defer _ = std.c.close(lst);

    const FakePeer = struct {
        fn say(c: fd_t, msg: []const u8) void {
            _ = std.c.write(c, msg.ptr, msg.len);
        }
        fn run(listener: fd_t) void {
            const c = std.c.accept(listener, null, null);
            if (c < 0) return;
            defer _ = std.c.close(c);
            var req: [4096]u8 = undefined;
            var got: usize = 0;
            while (got < req.len) {
                const n = readFd(c, req[got..]) catch return;
                if (n == 0) break;
                got += n;
                if (std.mem.indexOf(u8, req[0..got], "\"messages\":[]}") != null) break;
            }
            // The peer must see the BARE id, no trace of the @peer suffix, and its key.
            const rewritten = std.mem.indexOf(u8, req[0..got], "\"model\":\"bare-model\"") != null and
                std.mem.indexOf(u8, req[0..got], "@Studio") == null and
                std.mem.indexOf(u8, req[0..got], "POST /v1/chat/completions HTTP/1.1") != null and
                std.mem.indexOf(u8, req[0..got], "\r\nAuthorization: Bearer s3cret\r\n") != null;
            say(c, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n");
            say(c, if (rewritten) "data: ok\n\n" else "data: WRONG-REQUEST\n\n");
            const ts = std.c.timespec{ .sec = 0, .nsec = 20_000_000 };
            _ = std.c.nanosleep(&ts, null); // force a second pump iteration
            say(c, "data: [DONE]\n\n");
        }
    };
    const th = try std.Thread.spawn(.{}, FakePeer.run, .{lst});
    defer th.join();

    const body = "{\"model\":\"bare-model@Studio\",\"messages\":[]}";
    const vs = std.mem.indexOf(u8, body, "bare-model@Studio").?;
    const rewritten = try rewriteModelValue(a, body, body[vs .. vs + "bare-model@Studio".len], "bare-model");
    defer a.free(rewritten);

    var sink = TestSink{ .alloc = a };
    defer sink.buf.deinit(a);
    try tunnel(.{ .ip4 = .{ 127, 0, 0, 1 }, .port = port }, "POST", "/v1/chat/completions", rewritten, "s3cret", &sink);

    try t.expect(std.mem.indexOf(u8, sink.buf.items, "HTTP/1.1 200 OK") != null);
    try t.expect(std.mem.indexOf(u8, sink.buf.items, "data: ok") != null);
    try t.expect(std.mem.indexOf(u8, sink.buf.items, "data: [DONE]") != null);
    try t.expect(std.mem.indexOf(u8, sink.buf.items, "WRONG-REQUEST") == null);
}

test "lan: tunnel to a dead peer fails before writing anything to the client" {
    const a = t.allocator;
    var port: u16 = 0;
    const lst = try testListener(&port);
    _ = std.c.close(lst); // port now refuses connections

    var sink = TestSink{ .alloc = a };
    defer sink.buf.deinit(a);
    try t.expectError(error.PeerUnreachable, tunnel(.{ .ip4 = .{ 127, 0, 0, 1 }, .port = port }, "POST", "/v1/messages", "{}", null, &sink));
    try t.expectEqual(@as(usize, 0), sink.buf.items.len);
}

test "lan: discovery sends the peer's key, and a keyed peer's 401 reads as Unauthorized" {
    const a = t.allocator;
    var port: u16 = 0;
    const lst = try testListener(&port);
    defer _ = std.c.close(lst);

    const KeyedPeer = struct {
        fn run(listener: fd_t) void {
            for (0..2) |_| {
                const c = std.c.accept(listener, null, null);
                if (c < 0) return;
                defer _ = std.c.close(c);
                var req: [4096]u8 = undefined;
                var got: usize = 0;
                while (got < req.len) {
                    const n = readFd(c, req[got..]) catch return;
                    if (n == 0) break;
                    got += n;
                    if (std.mem.indexOf(u8, req[0..got], "\r\n\r\n") != null) break;
                }
                const reply = if (std.mem.indexOf(u8, req[0..got], "\r\nAuthorization: Bearer s3cret\r\n") != null)
                    "HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"object\":\"list\",\"data\":[{\"id\":\"gemma\"}]}"
                else
                    "HTTP/1.1 401 Unauthorized\r\nConnection: close\r\n\r\n{}";
                _ = std.c.write(c, reply.ptr, reply.len);
            }
        }
    };
    const th = try std.Thread.spawn(.{}, KeyedPeer.run, .{lst});
    defer th.join();

    try t.expectError(error.Unauthorized, fetchPeerModels(a, .{ 127, 0, 0, 1 }, port, "Studio", "own-token", null, null));
    const models = try fetchPeerModels(a, .{ 127, 0, 0, 1 }, port, "Studio", "own-token", "s3cret", null);
    defer freePeerModels(a, models);
    try t.expectEqual(@as(usize, 1), models.len);
    try t.expectEqualStrings("gemma", models[0].id);
}

test "a peer that starts leading a pair moves the change count; a plain refetch does not" {
    const a = std.testing.allocator;
    var l = Lan{ .alloc = a, .port = 0, .discover = true, .peers = .init(a), .known = .init(a) };
    defer {
        var it = l.peers.valueIterator();
        while (it.next()) |p| p.deinit(a);
        l.peers.deinit();
        l.known.deinit();
    }
    l.installPeer("M4Max", .{ 127, 0, 0, 1 }, 1, &.{}, false, .{});
    const first = l.changes.load(.monotonic);
    l.installPeer("M4Max", .{ 127, 0, 0, 1 }, 1, &.{}, false, .{});
    try std.testing.expectEqual(first, l.changes.load(.monotonic));
    l.installPeer("M4Max", .{ 127, 0, 0, 1 }, 1, &.{}, false, PeerMachine.parse(a,
        \\{"machine":{"os":"macos","leads":"Flash","joined":false}}
    ));
    try std.testing.expect(l.changes.load(.monotonic) > first);
}

test "a peer's machine rides from its /v1/models into the app's peer list" {
    const a = std.testing.allocator;
    const m = PeerMachine.parse(a,
        \\{"object":"list","data":[],"machine":{"os":"macos","model":"Mac16,5","leads":"MiniMax-H3","joined":false,"free_mib":98205,"total_mib":131072}}
    );
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(a);
    try appendPeerAddress(a, &buf, "M4Max", .{ 192, 168, 137, 3 }, 11234, &m);
    try appendPeerAddress(a, &buf, "5060-linux", .{ 192, 168, 2, 76 }, 11400, &PeerMachine.parse(a, "{\"data\":[]}"));
    const j = try std.fmt.allocPrint(a, "[{s}]", .{buf.items});
    defer a.free(j);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, j, .{});
    defer parsed.deinit();
    const first = parsed.value.array.items[0].object;
    try std.testing.expectEqualStrings("http://192.168.137.3:11234", first.get("url").?.string);
    try std.testing.expectEqualStrings("Mac16,5", first.get("model").?.string);
    try std.testing.expectEqualStrings("MiniMax-H3", first.get("leads").?.string);
    try std.testing.expectEqual(@as(i64, 98205), first.get("free_mib").?.integer);
    try std.testing.expectEqualStrings("", parsed.value.array.items[1].object.get("os").?.string);
}
