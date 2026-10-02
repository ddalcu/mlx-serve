//! Daemon-free Linux mDNS responder and browser for LAN peer discovery.
//! PTR/SRV/TXT/A discovery only; LAN policy and peer management stay in lan.zig.

const std = @import("std");
const net = @import("lan_net.zig");
const log = @import("log.zig");
const policy = @import("lan.zig");

pub const GROUP_IP4: [4]u8 = .{ 224, 0, 0, 251 };
pub const PORT: u16 = 5353;

pub const RType = struct {
    pub const a: u16 = 1;
    pub const ptr: u16 = 12;
    pub const txt: u16 = 16;
    pub const srv: u16 = 33;
    pub const any: u16 = 255;
};

pub const CLASS_IN: u16 = 1;
/// mDNS overloads the class's top bit: cache-flush on a response record,
/// unicast-reply-requested on a question. Same bit, opposite direction.
pub const CACHE_FLUSH: u16 = 0x8000;

/// TTLs from RFC 6762 §10: two minutes for records tied to a host's address,
/// 75 minutes for the shared PTR. A goodbye overrides both with 0.
const TTL_HOST: u32 = 120;
const TTL_SHARED: u32 = 4500;

pub const ParseError = error{
    /// The packet ended mid-record, a compression pointer led out of bounds or
    /// into a loop, or a name exceeded what DNS permits. Never a panic: these
    /// bytes arrive from anyone on the LAN.
    Malformed,
};

const MAX_NAME = 255;
const MAX_LABEL = 63;
/// A name may traverse at most this many compression pointers before we call
/// it a decompression bomb.
const MAX_JUMPS = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Wire codec (pure — every byte here can come from a hostile LAN neighbour)
// ─────────────────────────────────────────────────────────────────────────────

/// Decode the name at `off` into `out` (dotted, no trailing dot). Returns the
/// name and the offset just past the name AS ENCODED AT `off` — for a
/// compressed name that is right after the pointer, not after its target.
fn readName(msg: []const u8, off: usize, out: []u8) ParseError!struct { name: []const u8, next: usize } {
    var pos = off;
    var next: ?usize = null;
    var jumps: usize = 0;
    var n: usize = 0;
    while (true) {
        if (pos >= msg.len) return error.Malformed;
        const len = msg[pos];
        if (len & 0xc0 == 0xc0) {
            if (pos + 1 >= msg.len) return error.Malformed;
            const target = (@as(usize, len & 0x3f) << 8) | msg[pos + 1];
            if (next == null) next = pos + 2;
            jumps += 1;
            // Both guards matter: `target >= pos` catches a self-pointer and
            // any forward chain, `jumps` catches two names pointing at each
            // other's predecessor.
            if (jumps > MAX_JUMPS or target >= pos or target >= msg.len) return error.Malformed;
            pos = target;
            continue;
        }
        if (len > MAX_LABEL) return error.Malformed;
        if (len == 0) {
            pos += 1;
            break;
        }
        if (pos + 1 + len > msg.len) return error.Malformed;
        if (n != 0) {
            if (n + 1 > out.len) return error.Malformed;
            out[n] = '.';
            n += 1;
        }
        if (n + len > out.len or n + len > MAX_NAME) return error.Malformed;
        @memcpy(out[n..][0..len], msg[pos + 1 ..][0..len]);
        n += len;
        pos += 1 + len;
    }
    return .{ .name = out[0..n], .next = next orelse pos };
}

pub const Question = struct {
    name: []const u8,
    rtype: u16,
    rclass: u16,
};

pub const Record = struct {
    name: []const u8,
    rtype: u16,
    rclass: u16,
    /// TTL 0 is a goodbye: the peer is withdrawing this record.
    ttl: u32,
    rdata: []const u8,
    /// Offset of `rdata` within the message — a PTR/SRV target needs it to
    /// resolve its own compression pointers.
    rdata_off: usize,
};

pub const Parser = struct {
    msg: []const u8,
    is_response: bool,
    qd: u16,
    an: u16,
    pos: usize,
    q_left: u16,
    r_left: u32,

    pub fn init(msg: []const u8) ParseError!Parser {
        if (msg.len < 12) return error.Malformed;
        const flags = std.mem.readInt(u16, msg[2..4], .big);
        const qd = std.mem.readInt(u16, msg[4..6], .big);
        const an = std.mem.readInt(u16, msg[6..8], .big);
        const ns = std.mem.readInt(u16, msg[8..10], .big);
        const ar = std.mem.readInt(u16, msg[10..12], .big);
        return .{
            .msg = msg,
            .is_response = flags & 0x8000 != 0,
            .qd = qd,
            .an = an,
            .pos = 12,
            .q_left = qd,
            // Additionals carry the A record for a peer's SRV target as often
            // as the answer section does, so all three are walked.
            .r_left = @as(u32, an) + ns + ar,
        };
    }

    /// Questions must be drained before records: they share `pos`.
    pub fn nextQuestion(p: *Parser, name_buf: []u8) ParseError!?Question {
        if (p.q_left == 0) return null;
        p.q_left -= 1;
        const nm = try readName(p.msg, p.pos, name_buf);
        if (nm.next + 4 > p.msg.len) return error.Malformed;
        const rtype = std.mem.readInt(u16, p.msg[nm.next..][0..2], .big);
        const rclass = std.mem.readInt(u16, p.msg[nm.next + 2 ..][0..2], .big);
        p.pos = nm.next + 4;
        return .{ .name = nm.name, .rtype = rtype, .rclass = rclass };
    }

    pub fn nextRecord(p: *Parser, name_buf: []u8) ParseError!?Record {
        while (p.q_left != 0) _ = try p.nextQuestion(name_buf);
        if (p.r_left == 0) return null;
        p.r_left -= 1;
        const nm = try readName(p.msg, p.pos, name_buf);
        if (nm.next + 10 > p.msg.len) return error.Malformed;
        const rtype = std.mem.readInt(u16, p.msg[nm.next..][0..2], .big);
        const rclass = std.mem.readInt(u16, p.msg[nm.next + 2 ..][0..2], .big);
        const ttl = std.mem.readInt(u32, p.msg[nm.next + 4 ..][0..4], .big);
        const rdlen = std.mem.readInt(u16, p.msg[nm.next + 8 ..][0..2], .big);
        const rd_off = nm.next + 10;
        if (rd_off + rdlen > p.msg.len) return error.Malformed;
        p.pos = rd_off + rdlen;
        return .{
            .name = nm.name,
            .rtype = rtype,
            .rclass = rclass,
            .ttl = ttl,
            .rdata = p.msg[rd_off..][0..rdlen],
            .rdata_off = rd_off,
        };
    }

    /// Decode a name inside rdata (PTR target, SRV target) — needs the whole
    /// message, because it may be a compression pointer into the header area.
    pub fn nameAt(p: *const Parser, off: usize, out: []u8) ParseError![]const u8 {
        const nm = try readName(p.msg, off, out);
        return nm.name;
    }
};

pub const BuildError = error{NoSpaceLeft};

/// Writes names out in full: compression must be PARSED (peers emit it) but
/// never needs to be EMITTED, and our packets are small.
pub const Builder = struct {
    buf: []u8,
    len: usize = 12,
    counts: [4]u16 = @splat(0), // qd, an, ns, ar

    pub const Kind = enum { query, response };

    pub fn init(buf: []u8, kind: Kind) Builder {
        const b = Builder{ .buf = buf };
        @memset(buf[0..12], 0);
        // QR + AA: an mDNS response is always authoritative. Query id stays 0.
        if (kind == .response) std.mem.writeInt(u16, buf[2..4], 0x8400, .big);
        return b;
    }

    fn put(b: *Builder, bytes: []const u8) BuildError!void {
        if (b.len + bytes.len > b.buf.len) return error.NoSpaceLeft;
        @memcpy(b.buf[b.len..][0..bytes.len], bytes);
        b.len += bytes.len;
    }

    fn putInt(b: *Builder, comptime T: type, v: T) BuildError!void {
        var tmp: [@sizeOf(T)]u8 = undefined;
        std.mem.writeInt(T, &tmp, v, .big);
        try b.put(&tmp);
    }

    fn putName(b: *Builder, name: []const u8) BuildError!void {
        var it = std.mem.splitScalar(u8, name, '.');
        while (it.next()) |label| {
            if (label.len == 0) continue;
            if (label.len > MAX_LABEL) return error.NoSpaceLeft;
            try b.putInt(u8, @intCast(label.len));
            try b.put(label);
        }
        try b.putInt(u8, 0);
    }

    pub fn addQuestion(b: *Builder, name: []const u8, rtype: u16, rclass: u16) BuildError!void {
        try b.putName(name);
        try b.putInt(u16, rtype);
        try b.putInt(u16, rclass);
        b.counts[0] += 1;
    }

    /// Opens a record and reserves its rdlength; returns the offset to patch.
    fn startRecord(b: *Builder, name: []const u8, rtype: u16, rclass: u16, ttl: u32) BuildError!usize {
        try b.putName(name);
        try b.putInt(u16, rtype);
        try b.putInt(u16, rclass);
        try b.putInt(u32, ttl);
        try b.putInt(u16, 0);
        b.counts[1] += 1;
        return b.len - 2;
    }

    fn endRecord(b: *Builder, at: usize) void {
        std.mem.writeInt(u16, b.buf[at..][0..2], @intCast(b.len - at - 2), .big);
    }

    pub fn addPtr(b: *Builder, name: []const u8, ttl: u32, target: []const u8) BuildError!void {
        const at = try b.startRecord(name, RType.ptr, CLASS_IN, ttl);
        try b.putName(target);
        b.endRecord(at);
    }

    pub fn addSrv(b: *Builder, name: []const u8, ttl: u32, port: u16, target: []const u8) BuildError!void {
        const at = try b.startRecord(name, RType.srv, CLASS_IN | CACHE_FLUSH, ttl);
        try b.putInt(u16, 0); // priority
        try b.putInt(u16, 0); // weight
        try b.putInt(u16, port);
        try b.putName(target);
        b.endRecord(at);
    }

    pub fn addTxt(b: *Builder, name: []const u8, ttl: u32, txt: []const u8) BuildError!void {
        const at = try b.startRecord(name, RType.txt, CLASS_IN | CACHE_FLUSH, ttl);
        // Already length-prefixed by policy.txtBuild — this is the wire form.
        if (txt.len == 0) try b.putInt(u8, 0) else try b.put(txt);
        b.endRecord(at);
    }

    pub fn addA(b: *Builder, name: []const u8, ttl: u32, ip4: [4]u8) BuildError!void {
        const at = try b.startRecord(name, RType.a, CLASS_IN | CACHE_FLUSH, ttl);
        try b.put(&ip4);
        b.endRecord(at);
    }

    pub fn finish(b: *Builder) []const u8 {
        for (b.counts, 0..) |c, i| std.mem.writeInt(u16, b.buf[4 + i * 2 ..][0..2], c, .big);
        return b.buf[0..b.len];
    }
};

/// Instance part of `<instance>.<service_type>.local`, or null if `full` is not
/// of that form. Tolerates dots inside the instance name (mDNS allows them; a
/// naive `splitScalar` would truncate "my.box" to "my").
pub fn instanceOf(full: []const u8, service_type: []const u8) ?[]const u8 {
    var suffix_buf: [128]u8 = undefined;
    const suffix = std.fmt.bufPrint(&suffix_buf, ".{s}.local", .{service_type}) catch return null;
    if (full.len < suffix.len or !eqlIgnoreCase(full[full.len - suffix.len ..], suffix)) return null;
    const inst = full[0 .. full.len - suffix.len];
    return if (inst.len == 0) null else inst;
}

/// Does this question ask for something we advertise? A browser sends PTR for
/// the service type; a peer that already has our SRV asks A for our host.
pub fn questionIsOurs(q: Question, service_type: []const u8, instance_full: []const u8, host_fqdn: []const u8) bool {
    if (q.rtype != RType.ptr and q.rtype != RType.srv and
        q.rtype != RType.txt and q.rtype != RType.a and q.rtype != RType.any) return false;
    var type_buf: [128]u8 = undefined;
    const type_name = std.fmt.bufPrint(&type_buf, "{s}.local", .{service_type}) catch return false;
    return eqlIgnoreCase(q.name, type_name) or
        eqlIgnoreCase(q.name, instance_full) or
        eqlIgnoreCase(q.name, host_fqdn);
}

/// DNS names are case-insensitive on the wire, and peers do not preserve ours.
fn eqlIgnoreCase(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| if (std.ascii.toLower(x) != std.ascii.toLower(y)) return false;
    return true;
}

/// Never advertise an address a peer cannot dial. Loopback is the one that
/// matters: it resolves on the peer's box too, at which point our model list
/// looks like theirs.
pub fn publishAddress(ip4: [4]u8) bool {
    if (ip4[0] == 127) return false;
    if (ip4[0] == 0) return false;
    if (ip4[0] == 169 and ip4[1] == 254) return false; // link-local autoconf
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Responder: one socket, advertising and/or browsing
// ─────────────────────────────────────────────────────────────────────────────

/// What we advertise. `txt` is the wire form `policy.txtBuild` produces.
pub const Advertisement = struct {
    instance: []const u8,
    port: u16,
    txt: []const u8,
};

/// A discovered peer service, as much as has been learned so far. Fixed
/// buffers: a table entry must not let a LAN neighbour drive an allocation.
pub const Service = struct {
    name_buf: [96]u8 = undefined,
    name_len: usize = 0,
    txt_buf: [256]u8 = undefined,
    txt_len: usize = 0,
    host_buf: [96]u8 = undefined,
    host_len: usize = 0,
    ip4: ?[4]u8 = null,
    port: u16 = 0,
    /// Set by a TTL-0 record: the peer said goodbye and the entry is dead.
    gone: bool = false,

    pub fn name(s: *const Service) []const u8 {
        return s.name_buf[0..s.name_len];
    }
    pub fn txt(s: *const Service) []const u8 {
        return s.txt_buf[0..s.txt_len];
    }
    fn host(s: *const Service) []const u8 {
        return s.host_buf[0..s.host_len];
    }
    /// Complete enough to dial.
    pub fn resolved(s: *const Service) bool {
        return !s.gone and s.ip4 != null and s.port != 0;
    }
};

pub const MAX_SERVICES = 32;

pub const Options = struct {
    advertise: ?Advertisement = null,
    discover: bool = false,
    service_type: []const u8 = policy.SERVICE_TYPE,
    /// Owner of the A record and target of the SRV.
    host_name: []const u8 = "mlx-serve",
};

pub const Responder = struct {
    sock: net.Socket = net.invalid_socket,
    service_type: []const u8,
    /// Owned copies: `Options` borrows from the caller's frame.
    ad: ?Advertisement = null,
    ad_instance_buf: [96]u8 = undefined,
    ad_txt_buf: [256]u8 = undefined,
    host_buf: [96]u8 = undefined,
    host_len: usize = 0,
    ifaces: [8][4]u8 = undefined,
    iface_count: usize = 0,
    /// Fixed table: a LAN with more than MAX_SERVICES mlx-serve peers is not a
    /// scenario worth allocating for, and a fixed size cannot be grown without
    /// bound by a flood of invented instance names.
    services: [MAX_SERVICES]Service = undefined,
    service_count: usize = 0,
    // Bonjour may send A before SRV, even in a different packet.
    addresses: [MAX_SERVICES]struct { name: [MAX_NAME]u8, len: usize, ip4: [4]u8 } = undefined,
    address_count: usize = 0,

    pub fn init(r: *Responder, opts: Options) !void {
        r.* = .{ .service_type = opts.service_type };
        r.sock = try net.udpBindShared(PORT);
        errdefer net.close(r.sock);

        var ip_buf: [16][4]u8 = undefined;
        for (net.localIp4Addresses(&ip_buf)) |ip| {
            if (!publishAddress(ip) or r.iface_count == r.ifaces.len) continue;
            r.ifaces[r.iface_count] = ip;
            r.iface_count += 1;
            net.joinMulticast(r.sock, GROUP_IP4, ip);
        }
        // Also join on the unspecified interface: on a host whose enumeration
        // came back short, this is the one that works.
        net.joinMulticast(r.sock, GROUP_IP4, .{ 0, 0, 0, 0 });
        net.setMulticastTtl(r.sock, 255); // RFC 6762: mDNS packets leave with TTL 255
        net.setMulticastLoop(r.sock, true); // two instances on one box

        r.host_len = @min(opts.host_name.len, r.host_buf.len);
        @memcpy(r.host_buf[0..r.host_len], opts.host_name[0..r.host_len]);

        if (opts.advertise) |ad| r.setAdvertisement(ad);
        if (opts.discover) r.sendQuery();
    }

    pub fn setAdvertisement(r: *Responder, ad: Advertisement) void {
        const ni = @min(ad.instance.len, r.ad_instance_buf.len);
        @memcpy(r.ad_instance_buf[0..ni], ad.instance[0..ni]);
        const nt = @min(ad.txt.len, r.ad_txt_buf.len);
        @memcpy(r.ad_txt_buf[0..nt], ad.txt[0..nt]);
        r.ad = .{ .instance = r.ad_instance_buf[0..ni], .port = ad.port, .txt = r.ad_txt_buf[0..nt] };
    }

    pub fn deinit(r: *Responder) void {
        // Goodbye (TTL 0) so peers drop us now instead of waiting out the PTR
        // TTL — without it a restart looks like a 75-minute outage.
        if (r.ad != null) r.announce(true);
        net.close(r.sock);
        r.sock = net.invalid_socket;
    }

    pub fn advertising(r: *const Responder) bool {
        return r.ad != null;
    }

    fn hostFqdn(r: *const Responder, out: []u8) []const u8 {
        return std.fmt.bufPrint(out, "{s}.local", .{r.host_buf[0..r.host_len]}) catch out[0..0];
    }

    fn fullName(r: *const Responder, out: []u8) []const u8 {
        const ad = r.ad orelse return out[0..0];
        return std.fmt.bufPrint(out, "{s}.{s}.local", .{ ad.instance, r.service_type }) catch out[0..0];
    }

    /// Our whole record set, on every interface. `goodbye` zeroes every TTL.
    ///
    /// Always the full set whatever was asked: it is one small packet, and it
    /// saves the peer a second round trip for the A record it is about to
    /// need. RFC 6762 §8.3 wants an announcement repeated a few times, which
    /// `announceBurst` does at the points where loss actually costs us.
    fn announce(r: *Responder, goodbye: bool) void {
        const ad = r.ad orelse return;
        var name_buf: [192]u8 = undefined;
        var host_buf: [160]u8 = undefined;
        var type_buf: [128]u8 = undefined;
        const full = r.fullName(&name_buf);
        const fqdn = r.hostFqdn(&host_buf);
        const type_name = std.fmt.bufPrint(&type_buf, "{s}.local", .{r.service_type}) catch return;
        const host_ttl: u32 = if (goodbye) 0 else TTL_HOST;
        const shared_ttl: u32 = if (goodbye) 0 else TTL_SHARED;

        for (r.ifaces[0..r.iface_count]) |iface| {
            var buf: [1400]u8 = undefined;
            var b = Builder.init(&buf, .response);
            b.addPtr(type_name, shared_ttl, full) catch continue;
            b.addSrv(full, host_ttl, ad.port, fqdn) catch continue;
            b.addTxt(full, shared_ttl, ad.txt) catch continue;
            // The A record carries THIS interface's address, so a peer reached
            // over one interface is never handed another's unreachable one.
            b.addA(fqdn, host_ttl, iface) catch continue;
            net.setMulticastInterface(r.sock, iface);
            net.sendTo(r.sock, GROUP_IP4, PORT, b.finish());
        }
    }

    /// RFC 6762 §8.3: announce more than once, because the first multicast can
    /// simply be lost and nothing retransmits it. Cheap insurance — the
    /// alternative is a peer that stays invisible until its next browse.
    pub fn announceBurst(r: *Responder) void {
        if (r.ad == null) return;
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            r.announce(false);
            if (i + 1 < 3) {
                const ts = std.c.timespec{ .sec = 0, .nsec = 250_000_000 };
                _ = std.c.nanosleep(&ts, null);
            }
        }
    }

    pub fn reannounce(r: *Responder) void {
        r.announce(false);
    }

    pub fn sendQuery(r: *Responder) void {
        var type_buf: [128]u8 = undefined;
        const type_name = std.fmt.bufPrint(&type_buf, "{s}.local", .{r.service_type}) catch return;
        var buf: [256]u8 = undefined;
        var b = Builder.init(&buf, .query);
        b.addQuestion(type_name, RType.ptr, CLASS_IN) catch return;
        const pkt = b.finish();
        for (r.ifaces[0..r.iface_count]) |iface| {
            net.setMulticastInterface(r.sock, iface);
            net.sendTo(r.sock, GROUP_IP4, PORT, pkt);
        }
    }

    /// Bonjour can omit additional records: explicitly resolve partial replies.
    pub fn sendResolveQueries(r: *Responder) void {
        for (r.found()) |*svc| {
            if (svc.gone or (svc.resolved() and svc.txt_len != 0)) continue;
            var name_buf: [192]u8 = undefined;
            const full = std.fmt.bufPrint(&name_buf, "{s}.{s}.local", .{ svc.name(), r.service_type }) catch continue;
            var buf: [512]u8 = undefined;
            var b = Builder.init(&buf, .query);
            if (svc.port == 0) b.addQuestion(full, RType.srv, CLASS_IN) catch continue;
            if (svc.txt_len == 0) b.addQuestion(full, RType.txt, CLASS_IN) catch continue;
            if (svc.ip4 == null and svc.host_len != 0) b.addQuestion(svc.host(), RType.a, CLASS_IN) catch continue;
            if (b.counts[0] == 0) continue;
            for (r.ifaces[0..r.iface_count]) |iface| {
                net.setMulticastInterface(r.sock, iface);
                net.sendTo(r.sock, GROUP_IP4, PORT, b.finish());
            }
        }
    }

    /// Drain every datagram available within `timeout_ms`, answering queries
    /// for our advertisement and folding responses into the table.
    pub fn pump(r: *Responder, timeout_ms: i32) void {
        const deadline = net.monoMs() + timeout_ms;
        while (true) {
            const remain = deadline - net.monoMs();
            if (remain <= 0) return;
            if (!net.waitReadable(r.sock, @intCast(@min(remain, 1000)))) return;
            var buf: [9000]u8 = undefined;
            const dg = net.recvFrom(r.sock, &buf) orelse return;
            r.handle(buf[0..dg.len]);
        }
    }

    fn handle(r: *Responder, pkt: []const u8) void {
        var p = Parser.init(pkt) catch return;
        var name_buf: [MAX_NAME]u8 = undefined;
        if (!p.is_response) {
            const ad = r.ad orelse return;
            _ = ad;
            var full_buf: [192]u8 = undefined;
            var host_buf: [160]u8 = undefined;
            const full = r.fullName(&full_buf);
            const fqdn = r.hostFqdn(&host_buf);
            while (p.nextQuestion(&name_buf) catch return) |q| {
                if (!questionIsOurs(q, r.service_type, full, fqdn)) continue;
                r.announce(false);
                return;
            }
            return;
        }
        while (p.nextRecord(&name_buf) catch return) |rec| {
            if (rec.rclass & ~CACHE_FLUSH == CLASS_IN) r.foldRecord(&p, rec);
        }
    }

    /// Fold one answer into the table. Records arrive in any order and across
    /// packets, so an entry is built up by name and only becomes usable once
    /// `resolved()` holds.
    fn foldRecord(r: *Responder, p: *const Parser, rec: Record) void {
        var tmp: [MAX_NAME]u8 = undefined;
        switch (rec.rtype) {
            RType.ptr => {
                const target = p.nameAt(rec.rdata_off, &tmp) catch return;
                const inst = instanceOf(target, r.service_type) orelse return;
                const svc = r.entryFor(inst) orelse return;
                if (rec.ttl == 0) svc.gone = true;
            },
            RType.srv => {
                const inst = instanceOf(rec.name, r.service_type) orelse return;
                if (rec.rdata.len < 6) return;
                const svc = r.entryFor(inst) orelse return;
                if (rec.ttl == 0) {
                    svc.gone = true;
                    return;
                }
                svc.port = std.mem.readInt(u16, rec.rdata[4..6], .big);
                const target = p.nameAt(rec.rdata_off + 6, &tmp) catch return;
                svc.host_len = @min(target.len, svc.host_buf.len);
                @memcpy(svc.host_buf[0..svc.host_len], target[0..svc.host_len]);
                for (r.addresses[0..r.address_count]) |*address| {
                    if (eqlIgnoreCase(address.name[0..address.len], target)) svc.ip4 = address.ip4;
                }
            },
            RType.txt => {
                const inst = instanceOf(rec.name, r.service_type) orelse return;
                const svc = r.entryFor(inst) orelse return;
                if (rec.ttl == 0) {
                    svc.gone = true;
                    return;
                }
                svc.txt_len = @min(rec.rdata.len, svc.txt_buf.len);
                @memcpy(svc.txt_buf[0..svc.txt_len], rec.rdata[0..svc.txt_len]);
            },
            RType.a => {
                if (rec.rdata.len != 4 or rec.ttl == 0) return;
                const ip: [4]u8 = rec.rdata[0..4].*;
                if (!publishAddress(ip)) return;
                const address = for (r.addresses[0..r.address_count]) |*address| {
                    if (eqlIgnoreCase(address.name[0..address.len], rec.name)) break address;
                } else blk: {
                    if (r.address_count == r.addresses.len) break :blk null;
                    const address = &r.addresses[r.address_count];
                    address.len = rec.name.len;
                    @memcpy(address.name[0..address.len], rec.name);
                    r.address_count += 1;
                    break :blk address;
                };
                if (address) |a| a.ip4 = ip;
                // An A record names a HOST, not an instance: attach it to
                // every entry whose SRV target is that host.
                for (r.services[0..r.service_count]) |*svc| {
                    if (svc.host_len != 0 and eqlIgnoreCase(svc.host(), rec.name)) svc.ip4 = ip;
                }
            },
            else => {},
        }
    }

    fn entryFor(r: *Responder, inst: []const u8) ?*Service {
        for (r.services[0..r.service_count]) |*svc| {
            if (eqlIgnoreCase(svc.name(), inst)) return svc;
        }
        if (r.service_count == r.services.len) return null;
        const svc = &r.services[r.service_count];
        svc.* = .{};
        svc.name_len = @min(inst.len, svc.name_buf.len);
        @memcpy(svc.name_buf[0..svc.name_len], inst[0..svc.name_len]);
        r.service_count += 1;
        return svc;
    }

    /// Forget everything learned so far. The caller's peer table is the
    /// durable one (with its own two-tier failure counters); this table only
    /// has to describe the current sweep, and clearing it is what lets a
    /// renamed or moved peer be seen at its new address.
    pub fn clearServices(r: *Responder) void {
        r.service_count = 0;
        r.address_count = 0;
    }

    pub fn found(r: *Responder) []const Service {
        return r.services[0..r.service_count];
    }

    /// Claim `desired`, appending `-2`, `-3`… if another responder already
    /// advertises that name with a DIFFERENT token. Returns the name claimed.
    ///
    /// This is not RFC 6762 §8 probing — there is no tie-breaking and no
    /// defence of the name afterwards. It closes the one case that actually
    /// bites: dns_sd renames a colliding instance for free (`name (2)`), so
    /// without any check at all, two servers on one box — or two hosts with
    /// the same name — would both answer for one instance and a browser would
    /// see one peer with two SRV records and route to whichever answered last.
    /// A one-shot pre-flight costs a few hundred ms at startup and removes
    /// that; a real §8 implementation can replace it without changing callers.
    pub fn claimName(r: *Responder, desired: []const u8, own_token: []const u8, out: []u8) []const u8 {
        var attempt: u8 = 1;
        while (attempt <= 4) : (attempt += 1) {
            const candidate = if (attempt == 1)
                std.fmt.bufPrint(out, "{s}", .{desired}) catch desired
            else
                std.fmt.bufPrint(out, "{s}-{d}", .{ desired, attempt }) catch desired;
            r.clearServices();
            r.sendQuery();
            r.pump(400);
            if (!r.nameTakenByOther(candidate, own_token)) return candidate;
            log.info("[lan] instance name '{s}' is taken on this network; trying '{s}-{d}'\n", .{ candidate, desired, attempt + 1 });
        }
        return std.fmt.bufPrint(out, "{s}", .{desired}) catch desired;
    }

    fn nameTakenByOther(r: *Responder, candidate: []const u8, own_token: []const u8) bool {
        for (r.services[0..r.service_count]) |*svc| {
            if (svc.gone or !eqlIgnoreCase(svc.name(), candidate)) continue;
            // Our own advertisement echoing back is not a conflict.
            const tok = policy.txtFind(svc.txt(), "t=") orelse return true;
            if (!std.mem.eql(u8, tok, own_token)) return true;
        }
        return false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

const testing = std.testing;

test "mdns: a browse query round-trips through build and parse" {
    var buf: [256]u8 = undefined;
    var b = Builder.init(&buf, .query);
    try b.addQuestion("_mlxserve._tcp.local", RType.ptr, CLASS_IN);
    const pkt = b.finish();

    var p = try Parser.init(pkt);
    try testing.expect(!p.is_response);
    var nb: [MAX_NAME]u8 = undefined;
    const q = (try p.nextQuestion(&nb)).?;
    try testing.expectEqualStrings("_mlxserve._tcp.local", q.name);
    try testing.expectEqual(RType.ptr, q.rtype);
    try testing.expectEqual(@as(?Question, null), try p.nextQuestion(&nb));
}

test "mdns: an advertisement round-trips PTR, SRV, TXT and A" {
    var txt_buf: [128]u8 = undefined;
    const txt = policy.txtBuild(&txt_buf, "0123456789abcdef");

    var buf: [1400]u8 = undefined;
    var b = Builder.init(&buf, .response);
    try b.addPtr("_mlxserve._tcp.local", TTL_SHARED, "mac.\x5fmlxserve._tcp.local"[0..0] ++ "mac._mlxserve._tcp.local");
    try b.addSrv("mac._mlxserve._tcp.local", TTL_HOST, 8080, "mac.local");
    try b.addTxt("mac._mlxserve._tcp.local", TTL_SHARED, txt);
    try b.addA("mac.local", TTL_HOST, .{ 192, 168, 1, 40 });
    const pkt = b.finish();

    var p = try Parser.init(pkt);
    try testing.expect(p.is_response);
    var nb: [MAX_NAME]u8 = undefined;
    var tmp: [MAX_NAME]u8 = undefined;
    var saw_ptr = false;
    var saw_srv = false;
    var saw_txt = false;
    var saw_a = false;
    while (try p.nextRecord(&nb)) |rec| {
        switch (rec.rtype) {
            RType.ptr => {
                try testing.expectEqualStrings("mac._mlxserve._tcp.local", try p.nameAt(rec.rdata_off, &tmp));
                saw_ptr = true;
            },
            RType.srv => {
                try testing.expectEqual(@as(u16, 8080), std.mem.readInt(u16, rec.rdata[4..6], .big));
                try testing.expectEqualStrings("mac.local", try p.nameAt(rec.rdata_off + 6, &tmp));
                saw_srv = true;
            },
            RType.txt => {
                // The token must survive the wire byte-for-byte: it is what
                // makes a browser skip its own advertisement, and therefore
                // what makes proxy loops impossible.
                try testing.expectEqualStrings("0123456789abcdef", policy.txtFind(rec.rdata, "t=").?);
                saw_txt = true;
            },
            RType.a => {
                try testing.expectEqualSlices(u8, &.{ 192, 168, 1, 40 }, rec.rdata);
                saw_a = true;
            },
            else => {},
        }
    }
    try testing.expect(saw_ptr and saw_srv and saw_txt and saw_a);
}

test "mdns: compression pointers in a peer's packet decode" {
    // Hand-built: our own Builder never emits compression, so only a
    // hand-rolled packet can prove we PARSE what other stacks emit. Two
    // records — the first names itself inline and points into itself from
    // rdata, the second's NAME is a pointer to the first's.
    var pkt: [96]u8 = undefined;
    @memset(&pkt, 0);
    std.mem.writeInt(u16, pkt[2..4], 0x8400, .big);
    std.mem.writeInt(u16, pkt[6..8], 2, .big); // two answers
    const name_at: u16 = 0xc000 | 12; // "local" lives at offset 12

    // Record 1: name "local" inline, PTR whose rdata is a pointer to it.
    var n: usize = 12;
    pkt[n] = 5;
    @memcpy(pkt[n + 1 ..][0..5], "local");
    pkt[n + 6] = 0;
    n += 7;
    std.mem.writeInt(u16, pkt[n..][0..2], RType.ptr, .big);
    std.mem.writeInt(u16, pkt[n + 2 ..][0..2], CLASS_IN, .big);
    std.mem.writeInt(u32, pkt[n + 4 ..][0..4], 120, .big);
    std.mem.writeInt(u16, pkt[n + 8 ..][0..2], 2, .big);
    const rd1 = n + 10;
    std.mem.writeInt(u16, pkt[rd1..][0..2], name_at, .big);
    n = rd1 + 2;

    // Record 2: name is a COMPRESSION POINTER, rdata is an address.
    std.mem.writeInt(u16, pkt[n..][0..2], name_at, .big);
    n += 2;
    std.mem.writeInt(u16, pkt[n..][0..2], RType.a, .big);
    std.mem.writeInt(u16, pkt[n + 2 ..][0..2], CLASS_IN, .big);
    std.mem.writeInt(u32, pkt[n + 4 ..][0..4], 120, .big);
    std.mem.writeInt(u16, pkt[n + 8 ..][0..2], 4, .big);
    const rd2 = n + 10;
    @memcpy(pkt[rd2..][0..4], &[_]u8{ 192, 168, 1, 40 });

    var p = try Parser.init(pkt[0 .. rd2 + 4]);
    var nb: [MAX_NAME]u8 = undefined;
    var tmp: [MAX_NAME]u8 = undefined;

    const r1 = (try p.nextRecord(&nb)).?;
    try testing.expectEqualStrings("local", r1.name);
    try testing.expectEqualStrings("local", try p.nameAt(r1.rdata_off, &tmp));

    const r2 = (try p.nextRecord(&nb)).?;
    try testing.expectEqualStrings("local", r2.name); // decoded through the pointer
    try testing.expectEqualSlices(u8, &.{ 192, 168, 1, 40 }, r2.rdata);

    try testing.expectEqual(@as(?Record, null), try p.nextRecord(&nb));
}

test "mdns: hostile packets are rejected, never looped on" {
    var nb: [MAX_NAME]u8 = undefined;

    // Too short to hold a header.
    try testing.expectError(error.Malformed, Parser.init("abc"));

    // A pointer to itself, and a forward pointer: both must be refused rather
    // than followed. RFC 1035 requires pointers to point BACKWARDS.
    inline for (.{ @as(u16, 0xc000 | 12), @as(u16, 0xc000 | 30) }) |ptr_val| {
        var pkt: [64]u8 = undefined;
        @memset(&pkt, 0);
        std.mem.writeInt(u16, pkt[2..4], 0x8400, .big);
        std.mem.writeInt(u16, pkt[6..8], 1, .big);
        std.mem.writeInt(u16, pkt[12..14], ptr_val, .big);
        var p = try Parser.init(&pkt);
        try testing.expectError(error.Malformed, p.nextRecord(&nb));
    }

    // A label that runs past the end of the message.
    var pkt: [20]u8 = undefined;
    @memset(&pkt, 0);
    std.mem.writeInt(u16, pkt[2..4], 0x8400, .big);
    std.mem.writeInt(u16, pkt[6..8], 1, .big);
    pkt[12] = 60; // claims 60 bytes of label in an 20-byte packet
    var p = try Parser.init(&pkt);
    try testing.expectError(error.Malformed, p.nextRecord(&nb));
}

test "mdns: instanceOf strips the service suffix and tolerates dots in a name" {
    try testing.expectEqualStrings("mac", instanceOf("mac._mlxserve._tcp.local", "_mlxserve._tcp").?);
    // mDNS instance names may contain dots; a naive split would truncate this.
    try testing.expectEqualStrings("my.box", instanceOf("my.box._mlxserve._tcp.local", "_mlxserve._tcp").?);
    try testing.expectEqual(@as(?[]const u8, null), instanceOf("_mlxserve._tcp.local", "_mlxserve._tcp"));
    try testing.expectEqual(@as(?[]const u8, null), instanceOf("printer._ipp._tcp.local", "_mlxserve._tcp"));
}

test "mdns: we answer the queries a peer actually sends, and nothing else" {
    const st = "_mlxserve._tcp";
    const full = "mac._mlxserve._tcp.local";
    const host = "mac.local";
    // A browser's PTR for the service type.
    try testing.expect(questionIsOurs(.{ .name = "_mlxserve._tcp.local", .rtype = RType.ptr, .rclass = CLASS_IN }, st, full, host));
    // A peer that has our SRV and needs the address behind it. Without this
    // arm the peer resolves the service and can never dial it.
    try testing.expect(questionIsOurs(.{ .name = host, .rtype = RType.a, .rclass = CLASS_IN }, st, full, host));
    // Case-insensitive: peers do not preserve ours.
    try testing.expect(questionIsOurs(.{ .name = "MAC.LOCAL", .rtype = RType.a, .rclass = CLASS_IN }, st, full, host));
    // Someone else's service, and someone else's host.
    try testing.expect(!questionIsOurs(.{ .name = "_ipp._tcp.local", .rtype = RType.ptr, .rclass = CLASS_IN }, st, full, host));
    try testing.expect(!questionIsOurs(.{ .name = "other.local", .rtype = RType.a, .rclass = CLASS_IN }, st, full, host));
}

test "mdns: loopback and link-local are never advertised off-box" {
    // 127.x resolves on the PEER's machine too, at which point our model list
    // looks like theirs.
    try testing.expect(!publishAddress(.{ 127, 0, 0, 1 }));
    try testing.expect(!publishAddress(.{ 169, 254, 3, 4 }));
    try testing.expect(!publishAddress(.{ 0, 0, 0, 0 }));
    try testing.expect(publishAddress(.{ 192, 168, 1, 40 }));
    try testing.expect(publishAddress(.{ 10, 0, 0, 7 }));
}

/// The advertisement TXT used by the socket tests. Built per call: a
/// module-level `var` filled from a `test {}` block is initialized in test
/// ORDER, which is not the order these run in.
fn testTxt(buf: []u8, token: []const u8) []const u8 {
    return policy.txtBuild(buf, token);
}

test "mdns: a responder discovers another responder on this machine" {
    // The one test that exercises the SOCKET path — codec round-trips prove
    // the bytes, not that a peer ever receives them. Two instances on one box
    // is also a supported deployment, so this is not merely a stand-in for a
    // second machine.
    var txt_buf: [64]u8 = undefined;
    const txt = testTxt(&txt_buf, "aaaabbbbccccdddd");

    var adv: Responder = undefined;
    adv.init(.{
        .advertise = .{ .instance = "mdns-test-a", .port = 18080, .txt = txt },
        .host_name = "mdns-test-host",
    }) catch return error.SkipZigTest;
    defer adv.deinit();
    // No routable interface (a sandbox with loopback only) means there is
    // nothing to multicast on and the test has no subject.
    if (adv.iface_count == 0) return error.SkipZigTest;

    var br: Responder = undefined;
    br.init(.{ .discover = true }) catch return error.SkipZigTest;
    defer br.deinit();

    adv.announce(false);

    // Multicast is lossy by nature; retry rather than pin the test to one
    // datagram surviving.
    var tries: usize = 0;
    while (tries < 6) : (tries += 1) {
        br.sendQuery();
        adv.pump(150); // let the advertiser answer the query it just received
        br.pump(300);
        for (br.found()) |*svc| {
            if (!std.mem.eql(u8, svc.name(), "mdns-test-a")) continue;
            if (!svc.resolved()) continue;
            try testing.expectEqual(@as(u16, 18080), svc.port);
            // The token is what makes a browser skip its own advertisement.
            try testing.expectEqualStrings("aaaabbbbccccdddd", policy.txtFind(svc.txt(), "t=").?);
            // Never a loopback address: the peer has to be able to dial it.
            try testing.expect(publishAddress(svc.ip4.?));
            return;
        }
    }
    return error.SkipZigTest; // multicast blocked on this host
}

test "mdns: claimName renames around another instance holding the name" {
    var holder_txt: [64]u8 = undefined;
    var holder: Responder = undefined;
    holder.init(.{
        .advertise = .{ .instance = "mdns-claim", .port = 18081, .txt = testTxt(&holder_txt, "aaaabbbbccccdddd") },
        .host_name = "mdns-test-host",
    }) catch return error.SkipZigTest;
    defer holder.deinit();
    if (holder.iface_count == 0) return error.SkipZigTest;
    holder.announce(false);

    var joiner: Responder = undefined;
    joiner.init(.{ .discover = true }) catch return error.SkipZigTest;
    defer joiner.deinit();

    // A DIFFERENT token: this is somebody else's name, so we must not take it.
    var out: [96]u8 = undefined;
    const claimed = claimWithHolder(&joiner, &holder, "mdns-claim", "1111222233334444", &out);
    if (std.mem.eql(u8, claimed, "mdns-claim")) return error.SkipZigTest; // multicast blocked
    try testing.expectEqualStrings("mdns-claim-2", claimed);
}

/// `claimName` drives its own query/pump cycle; in-process there is no
/// separate thread answering, so the holder is pumped alongside it.
fn claimWithHolder(joiner: *Responder, holder: *Responder, desired: []const u8, token: []const u8, out: []u8) []const u8 {
    var attempt: u8 = 1;
    while (attempt <= 4) : (attempt += 1) {
        const candidate = if (attempt == 1)
            std.fmt.bufPrint(out, "{s}", .{desired}) catch desired
        else
            std.fmt.bufPrint(out, "{s}-{d}", .{ desired, attempt }) catch desired;
        joiner.clearServices();
        joiner.sendQuery();
        holder.pump(150);
        joiner.pump(300);
        if (!joiner.nameTakenByOther(candidate, token)) return candidate;
    }
    return desired;
}

test "mdns: Bonjour A records can precede SRV in one response or separate packets" {
    const full = "studio._mlxserve._tcp.local";
    const host = "studio.local";
    const ip = [4]u8{ 10, 29, 1, 13 };
    var buf: [512]u8 = undefined;
    var r = Responder{ .service_type = policy.SERVICE_TYPE };
    var b = Builder.init(&buf, .response);
    try b.addA(host, 120, ip);
    try b.addSrv(full, 120, 11234, host);
    r.handle(b.finish());
    try testing.expect(r.found()[0].resolved());
    try testing.expectEqual(ip, r.found()[0].ip4.?);

    r.clearServices();
    b = Builder.init(&buf, .response);
    try b.addA(host, 120, ip);
    r.handle(b.finish());
    b = Builder.init(&buf, .response);
    try b.addSrv(full, 120, 11234, host);
    r.handle(b.finish());
    try testing.expect(r.found()[0].resolved());
    try testing.expectEqual(ip, r.found()[0].ip4.?);
}

test "mdns: service suffix matching is case insensitive" {
    try testing.expectEqualStrings("Studio", instanceOf("Studio._MLXSERVE._TCP.LOCAL", policy.SERVICE_TYPE).?);
}
