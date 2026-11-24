//! Direct Thunderbolt 5 RDMA KV-cache transfer endpoints (`/v1/cache/rdma/*`).
//!
//! Control plane gated by the generic `--cache-transfer` / `--cache-transfer-key-file` policy:
//!   - `GET    /v1/cache/rdma/info`                availability, protocol, active distributed state, RDMA devices
//!   - `POST   /v1/cache/rdma/offer`               (src) export + pin + return offer ticket (arms UC responder if QP is RTS)
//!   - `POST   /v1/cache/rdma/accept`              (dst / src) out-of-band UC QP INIT -> RTR -> RTS handshake
//!   - `POST   /v1/cache/rdma/pull`                (dst) version & headroom pre-flight, UC credit-paced pull, attach
//!   - `DELETE /v1/cache/rdma/offer/{id}?clean=1|0` (src) clean=1 immediate release; clean=0 quarantine until expires_ms
//!   - `POST   /v1/cache/rdma/quiesce`             (all) drain & destroy RDMA QPs/contexts cleanly
//!
//! Transport notes (macOS 26.2+ Thunderbolt 5 `librdma.dylib`):
//! - Apple's TB5 RDMA driver supports two-sided `IBV_QPT_UC` SEND/RECV only (`IBV_QPT_RC` returns
//!   `EOPNOTSUPP`, `ibv_reg_mr` returns `rkey=0`).
//! - Each directional QP (`send` vs `recv`) owns one pre-registered 4 MiB staging ring MR created
//!   once at QP setup, giving 0 ms per-transfer MR registration, arbitrary KV cache size support,
//!   and credit-paced 512 KiB x 4 (2 MiB window) streaming over TB5.
//! - End-to-end byte integrity is verified on import by `peer_wire`'s 64-bit Wyhash checksum.
//! - `mlx-serve` makes zero outbound HTTP/TCP connections on this path; any external coordinator
//!   or script can mediate QP metadata exchange over the authenticated HTTP control plane.

const std = @import("std");
const builtin = @import("builtin");
const build_options = @import("build_options");
const log = @import("../../sdk/util/log.zig");
const io_util = @import("../../sdk/util/io_util.zig");
const http_server = @import("../http/server.zig");
const Conn = @import("../http/conn.zig").Conn;
const router = @import("../http/router.zig");
const http_request = @import("../http/request.zig");
const auth = @import("../http/auth.zig");
const sendResponse = @import("../http/respond.zig").sendResponse;
const sendErrorResponse = @import("../http/errors.zig").sendErrorResponse;
const sendErrorResponseWithReason = @import("../http/errors.zig").sendErrorResponseWithReason;
const mlx_engine = @import("../engines/mlx.zig");
const cache_transfer = @import("cache_transfer.zig");
const requests = @import("../gpu_thread/requests.zig");
const LoadedModel = @import("../registry/registry.zig").LoadedModel;
const Scheduler = @import("../gpu_thread/scheduler.zig").Scheduler;

pub const prefix = "/v1/cache/rdma/";
const offer_delete_prefix = "/v1/cache/rdma/offer/";
pub const protocol_id = "uc_ring_v1";

const MIN_EXPIRES_MS: u64 = 1_000;
const MAX_EXPIRES_MS: u64 = 300_000;
const MAX_PULL_DEADLINE_MS: u64 = 300_000;
const MAX_OFFERS: usize = 32;
const MAX_QPS: usize = 32;
const MAX_LIST_DEVICES: usize = 16;
const CONSUMED_RING_LEN: usize = 256;
const QUIESCE_WAIT_MS: i64 = 500;
const DEFAULT_PSN: u32 = 7;

// ── C RDMA shim FFI (compiled on macOS non-iOS only) ─────────────────────────

const has_real_rdma = builtin.os.tag == .macos and !build_options.ios;

pub const CEndpoint = extern struct {
    gid: [16]u8 = @splat(0),
    qpn: u32 = 0,
    psn: u32 = 0,
    lid: u16 = 0,
    gid_index: u8 = 0,
    active_mtu: u8 = 0,
};

const CDev = opaque {};
const CQp = opaque {};

extern fn msv_rdma_available() callconv(.c) c_int;
extern fn msv_rdma_list_devices(out: [*][64]u8, cap: usize) callconv(.c) c_int;
extern fn msv_rdma_dev_open(dev_name: [*:0]const u8, err_buf: ?[*]u8, err_cap: usize) callconv(.c) ?*CDev;
extern fn msv_rdma_dev_close(dev: *CDev) callconv(.c) void;
extern fn msv_rdma_qp_create(dev: *CDev, psn: u32, is_sender: c_int, local_out: *CEndpoint, err_buf: ?[*]u8, err_cap: usize) callconv(.c) ?*CQp;
extern fn msv_rdma_qp_connect(qp: *CQp, peer_gid: *const [16]u8, peer_qpn: u32, peer_psn: u32, peer_lid: u16, err_buf: ?[*]u8, err_cap: usize) callconv(.c) c_int;
extern fn msv_rdma_qp_destroy(qp: *CQp) callconv(.c) void;
extern fn msv_rdma_serve_offer(qp: *CQp, src_buf: *const anyopaque, len: u64, tid_16: *const [16]u8, stop_flag: *volatile c_int, timeout_ms: u64, err_buf: ?[*]u8, err_cap: usize) callconv(.c) c_int;
extern fn msv_rdma_pull_offer(qp: *CQp, dst_buf: *anyopaque, len: u64, tid_16: *const [16]u8, stop_flag: *volatile c_int, timeout_ms: u64, err_buf: ?[*]u8, err_cap: usize) callconv(.c) c_int;

// ── Pluggable Transport VTable (Real IBVerbs + Loopback test transport) ──────

pub const ServeStatus = enum { ok, cancelled, timeout, io_error };
pub const ReadStatus = enum { ok, timeout, io_error };

pub const Transport = struct {
    ctx: *anyopaque,
    available: *const fn (ctx: *anyopaque) bool,
    list_devices: *const fn (ctx: *anyopaque, out: [][64]u8) usize,
    dev_open: *const fn (ctx: *anyopaque, dev_name: []const u8, err_buf: []u8) ?*anyopaque,
    dev_close: *const fn (ctx: *anyopaque, dev: *anyopaque) void,
    qp_create: *const fn (ctx: *anyopaque, dev: *anyopaque, psn: u32, is_sender: bool, local_out: *CEndpoint, err_buf: []u8) ?*anyopaque,
    qp_connect: *const fn (ctx: *anyopaque, qp: *anyopaque, peer_gid: [16]u8, peer_qpn: u32, peer_psn: u32, peer_lid: u16, err_buf: []u8) bool,
    qp_destroy: *const fn (ctx: *anyopaque, qp: *anyopaque) void,
    serve_offer: *const fn (ctx: *anyopaque, qp: *anyopaque, src_buf: []const u8, tid_16: [16]u8, stop_flag: *std.atomic.Value(c_int), timeout_ms: u64, err_buf: []u8) ServeStatus,
    pull_offer: *const fn (ctx: *anyopaque, qp: *anyopaque, dst_buf: []u8, tid_16: [16]u8, stop_flag: *std.atomic.Value(c_int), timeout_ms: u64, err_buf: []u8) ReadStatus,
};

var real_transport_dummy: u8 = 0;

const real_transport: Transport = .{
    .ctx = &real_transport_dummy,
    .available = realAvailable,
    .list_devices = realListDevices,
    .dev_open = realDevOpen,
    .dev_close = realDevClose,
    .qp_create = realQpCreate,
    .qp_connect = realQpConnect,
    .qp_destroy = realQpDestroy,
    .serve_offer = realServeOffer,
    .pull_offer = realPullOffer,
};

fn cErrSlice(buf: []u8) []const u8 {
    return std.mem.sliceTo(buf, 0);
}

fn realAvailable(_: *anyopaque) bool {
    if (comptime !has_real_rdma) return false;
    return msv_rdma_available() != 0;
}

fn realListDevices(_: *anyopaque, out: [][64]u8) usize {
    if (comptime !has_real_rdma) return 0;
    if (out.len == 0) return 0;
    const n = msv_rdma_list_devices(out.ptr, out.len);
    return if (n > 0) @min(@as(usize, @intCast(n)), out.len) else 0;
}

fn realDevOpen(_: *anyopaque, dev_name: []const u8, err_buf: []u8) ?*anyopaque {
    if (comptime !has_real_rdma) return null;
    var name_z: [64]u8 = undefined;
    if (dev_name.len == 0 or dev_name.len >= name_z.len) {
        _ = std.fmt.bufPrint(err_buf, "invalid rdma device name\x00", .{}) catch {};
        return null;
    }
    @memcpy(name_z[0..dev_name.len], dev_name);
    name_z[dev_name.len] = 0;
    const d = msv_rdma_dev_open(@ptrCast(&name_z), err_buf.ptr, err_buf.len) orelse return null;
    return @ptrCast(d);
}

fn realDevClose(_: *anyopaque, dev: *anyopaque) void {
    if (comptime !has_real_rdma) return;
    msv_rdma_dev_close(@ptrCast(@alignCast(dev)));
}

fn realQpCreate(_: *anyopaque, dev: *anyopaque, psn: u32, is_sender: bool, local_out: *CEndpoint, err_buf: []u8) ?*anyopaque {
    if (comptime !has_real_rdma) return null;
    const q = msv_rdma_qp_create(
        @ptrCast(@alignCast(dev)),
        psn,
        if (is_sender) 1 else 0,
        local_out,
        err_buf.ptr,
        err_buf.len,
    ) orelse return null;
    return @ptrCast(q);
}

fn realQpConnect(_: *anyopaque, qp: *anyopaque, peer_gid: [16]u8, peer_qpn: u32, peer_psn: u32, peer_lid: u16, err_buf: []u8) bool {
    if (comptime !has_real_rdma) return false;
    return msv_rdma_qp_connect(@ptrCast(@alignCast(qp)), &peer_gid, peer_qpn, peer_psn, peer_lid, err_buf.ptr, err_buf.len) == 0;
}

fn realQpDestroy(_: *anyopaque, qp: *anyopaque) void {
    if (comptime !has_real_rdma) return;
    msv_rdma_qp_destroy(@ptrCast(@alignCast(qp)));
}

fn realServeOffer(
    _: *anyopaque,
    qp: *anyopaque,
    src_buf: []const u8,
    tid_16: [16]u8,
    stop_flag: *std.atomic.Value(c_int),
    timeout_ms: u64,
    err_buf: []u8,
) ServeStatus {
    if (comptime !has_real_rdma) return .io_error;
    const rc = msv_rdma_serve_offer(
        @ptrCast(@alignCast(qp)),
        src_buf.ptr,
        src_buf.len,
        &tid_16,
        @ptrCast(&stop_flag.raw),
        timeout_ms,
        err_buf.ptr,
        err_buf.len,
    );
    return switch (rc) {
        0 => .ok,
        1 => .cancelled,
        -1 => .timeout,
        else => .io_error,
    };
}

fn realPullOffer(
    _: *anyopaque,
    qp: *anyopaque,
    dst_buf: []u8,
    tid_16: [16]u8,
    stop_flag: *std.atomic.Value(c_int),
    timeout_ms: u64,
    err_buf: []u8,
) ReadStatus {
    if (comptime !has_real_rdma) return .io_error;
    const rc = msv_rdma_pull_offer(
        @ptrCast(@alignCast(qp)),
        dst_buf.ptr,
        dst_buf.len,
        &tid_16,
        @ptrCast(&stop_flag.raw),
        timeout_ms,
        err_buf.ptr,
        err_buf.len,
    );
    return switch (rc) {
        0 => .ok,
        -1 => .timeout,
        else => .io_error,
    };
}

// ── Cache Hooks (real Scheduler by default; injectable for hermetic unit tests) ─

pub const CacheHooks = struct {
    ctx: *anyopaque,
    export_session: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, model: ?[]const u8, session: u64, deadline_ms: ?u64, max_bytes: usize) anyerror!requests.CacheExportOutcome,
    import_body: *const fn (ctx: *anyopaque, allocator: std.mem.Allocator, model: ?[]const u8, body: []const u8, deadline_ms: ?u64) anyerror!requests.CacheImportOutcome,
};

fn acquireResidentModel(io: std.Io, scheduler: *Scheduler, requested_model: ?[]const u8) ?*LoadedModel {
    const reg = scheduler.registry;
    reg.mutex.lockUncancelable(io);
    defer reg.mutex.unlock(io);
    const entry: *LoadedModel = if (requested_model) |req_id| blk: {
        const id = if (req_id.len == 0 or std.mem.eql(u8, req_id, "mlx-serve")) reg.default_id else req_id;
        if (id.len == 0) return null;
        break :blk reg.entries.get(id) orelse return null;
    } else scheduler.current_model orelse return null;
    if (entry.state != .ready) return null;
    _ = entry.refcount.fetchAdd(1, .acq_rel);
    return entry;
}

fn defaultExportSession(io: std.Io, allocator: std.mem.Allocator, model: ?[]const u8, session: u64, deadline_ms: ?u64, max_bytes: usize) !requests.CacheExportOutcome {
    const scheduler = http_server.launch().global_scheduler orelse return error.SchedulerNotReady;
    const lm = acquireResidentModel(io, scheduler, model) orelse return .not_found;
    defer scheduler.release(lm);
    var req: requests.CacheExportRequest = .{
        .model = lm,
        .prompt_ids = &.{},
        .has_tools = false,
        .session = session,
        .deadline_ms = deadline_ms,
        .max_bytes = max_bytes,
        .admit_transfer_budget = true,
        .allocator = allocator,
    };
    return scheduler.exportCache(&req);
}

fn defaultImportBody(io: std.Io, allocator: std.mem.Allocator, model: ?[]const u8, body: []const u8, deadline_ms: ?u64) !requests.CacheImportOutcome {
    const scheduler = http_server.launch().global_scheduler orelse return error.SchedulerNotReady;
    const lm = acquireResidentModel(io, scheduler, model) orelse return .{ .refused = .unsupported_model };
    defer scheduler.release(lm);
    var req: requests.CacheImportRequest = .{
        .model = lm,
        .body = body,
        .deadline_ms = deadline_ms,
        .allocator = allocator,
    };
    return scheduler.importCache(&req);
}

// ── State Manager (host-unit owned; refcounted Dev/QP/Offer lifetimes) ───────

const DevEntry = struct {
    name_buf: [64]u8 = @splat(0),
    name_len: usize = 0,
    handle: *anyopaque,
    /// Protected by `Manager.mu`: 1 for membership in `mgr.devs` + 1 per live `QpEntry`.
    refcount: u32 = 1,

    fn name(self: *const DevEntry) []const u8 {
        return self.name_buf[0..self.name_len];
    }
};

const QpState = enum { init, rts };
const QpRole = enum { send, recv };

const QpEntry = struct {
    peer_id_buf: [64]u8 = @splat(0),
    peer_id_len: usize = 0,
    dev_name_buf: [64]u8 = @splat(0),
    dev_name_len: usize = 0,
    role: QpRole,
    dev: *DevEntry,
    handle: *anyopaque,
    local: CEndpoint,
    state: QpState = .init,
    peer_gid: [16]u8 = @splat(0),
    peer_qpn: u32 = 0,
    peer_psn: u32 = 0,
    in_use: bool = false,
    /// Set when removed from `mgr.qps` while `in_use == true`; destroyed by the owning op on exit.
    retired: bool = false,
    stop_flag: std.atomic.Value(c_int) = .init(0),

    fn peerId(self: *const QpEntry) []const u8 {
        return self.peer_id_buf[0..self.peer_id_len];
    }

    fn devName(self: *const QpEntry) []const u8 {
        return self.dev_name_buf[0..self.dev_name_len];
    }
};

const OfferState = enum { pinned, serving, served, quarantined };

const OfferEntry = struct {
    transfer_id: [32]u8,
    tid_16: [16]u8,
    peer_id_buf: [64]u8 = @splat(0),
    peer_id_len: usize = 0,
    dev_name_buf: [64]u8 = @splat(0),
    dev_name_len: usize = 0,
    buf_ptr: [*]u8,
    buf_len: usize,
    buf_allocator: std.mem.Allocator,
    reserved_export_bytes: u64,
    expires_at_ms: i64,
    timeout_ms: u64,
    state: OfferState = .pinned,
    stop_flag: std.atomic.Value(c_int) = .init(0),
    responder_done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    /// Protected by `Manager.mu`: 1 for membership in `mgr.offers` + 1 per active responder/joiner.
    /// Memory is freed only when `refcount` transitions `1 -> 0`.
    refcount: u32 = 1,

    fn idSlice(self: *const OfferEntry) []const u8 {
        return &self.transfer_id;
    }

    fn peerId(self: *const OfferEntry) []const u8 {
        return self.peer_id_buf[0..self.peer_id_len];
    }

    fn devName(self: *const OfferEntry) []const u8 {
        return self.dev_name_buf[0..self.dev_name_len];
    }

    fn freeResources(self: *OfferEntry, mgr_alloc: std.mem.Allocator) void {
        self.buf_allocator.free(self.buf_ptr[0..self.buf_len]);
        if (self.reserved_export_bytes > 0) {
            http_server.releaseCacheExportBytes(self.reserved_export_bytes);
            self.reserved_export_bytes = 0;
        }
        mgr_alloc.destroy(self);
    }
};

const ResponderCtx = struct {
    mgr: *Manager,
    io: std.Io,
    offer: *OfferEntry,
    qp: *QpEntry,
};

fn responderMain(ctx: ResponderCtx) void {
    var err_buf: [256]u8 = @splat(0);
    const offer = ctx.offer;
    const qp = ctx.qp;
    const status = ctx.mgr.transport.serve_offer(
        ctx.mgr.transport.ctx,
        qp.handle,
        offer.buf_ptr[0..offer.buf_len],
        offer.tid_16,
        &offer.stop_flag,
        offer.timeout_ms,
        &err_buf,
    );
    const qp_broken = (status == .timeout or status == .io_error);
    if (qp_broken) {
        const detail = cErrSlice(&err_buf);
        log.warn("RDMA sender responder error for offer {s}: {s} ({s})\n", .{
            offer.idSlice(),
            @tagName(status),
            if (detail.len > 0) detail else "unknown",
        });
    }

    var free_offer = false;
    var destroy_qp: ?*QpEntry = null;
    var close_dev: ?*DevEntry = null;
    {
        ctx.mgr.mu.lockUncancelable(ctx.io);
        defer ctx.mgr.mu.unlock(ctx.io);
        if (status == .ok and offer.state == .serving) {
            offer.state = .served;
        }
        qp.in_use = false;
        if (qp.retired) {
            destroy_qp = qp;
        } else if (qp_broken) {
            if (ctx.mgr.findQpEntryIndexLocked(qp)) |q_idx| {
                _ = ctx.mgr.qps.swapRemove(q_idx);
                qp.retired = true;
                destroy_qp = qp;
            }
        }
        if (destroy_qp) |dq| {
            close_dev = ctx.mgr.decDevRefLocked(dq.dev);
        }
        offer.responder_done.store(true, .release);
        free_offer = ctx.mgr.decOfferRefLocked(offer);
        ctx.mgr.active_ops -|= 1;
    }

    if (destroy_qp) |dq| {
        ctx.mgr.transport.qp_destroy(ctx.mgr.transport.ctx, dq.handle);
        ctx.mgr.alloc().destroy(dq);
    }
    if (close_dev) |cd| {
        ctx.mgr.transport.dev_close(ctx.mgr.transport.ctx, cd.handle);
        ctx.mgr.alloc().destroy(cd);
    }
    if (free_offer) {
        offer.freeResources(ctx.mgr.alloc());
    }
}

const ConsumedId = struct {
    buf: [128]u8 = @splat(0),
    len: usize = 0,

    fn slice(self: *const ConsumedId) []const u8 {
        return self.buf[0..self.len];
    }
};

const JoinItem = struct {
    offer: *OfferEntry,
    thread: ?std.Thread,
};

pub const Manager = struct {
    mu: std.Io.Mutex = .init,
    transport: Transport = real_transport,
    cache_hooks: ?CacheHooks = null,
    clock_fn: *const fn (std.Io) i64 = io_util.nowMsMonotonic,
    test_available_override: ?bool = null,
    test_distributed_active_override: ?bool = null,
    devs: std.ArrayList(*DevEntry) = .empty,
    qps: std.ArrayList(*QpEntry) = .empty,
    offers: std.ArrayList(*OfferEntry) = .empty,
    consumed: [CONSUMED_RING_LEN]ConsumedId = @splat(.{}),
    consumed_next: usize = 0,
    active_ops: u32 = 0,
    quiescing_count: u32 = 0,

    fn alloc(self: *const Manager) std.mem.Allocator {
        _ = self;
        return std.heap.c_allocator;
    }

    fn nowMs(self: *const Manager, io: std.Io) i64 {
        return self.clock_fn(io);
    }

    pub fn isAvailable(self: *const Manager) bool {
        if (self.test_available_override) |v| return v;
        return self.transport.available(self.transport.ctx);
    }

    fn distributedClusterActive(self: *const Manager) bool {
        if (self.test_distributed_active_override) |v| return v;
        return mlx_engine.api().cluster_active();
    }

    fn listDevices(self: *const Manager, out: [][64]u8) usize {
        if (self.test_available_override) |v| if (!v) return 0;
        return self.transport.list_devices(self.transport.ctx, out);
    }

    fn decDevRefLocked(self: *Manager, dev: *DevEntry) ?*DevEntry {
        _ = self;
        std.debug.assert(dev.refcount > 0);
        dev.refcount -= 1;
        return if (dev.refcount == 0) dev else null;
    }

    fn decOfferRefLocked(self: *Manager, offer: *OfferEntry) bool {
        _ = self;
        std.debug.assert(offer.refcount > 0);
        offer.refcount -= 1;
        return offer.refcount == 0;
    }

    fn findDevLocked(self: *Manager, dev_name: []const u8) ?*DevEntry {
        for (self.devs.items) |d| {
            if (std.mem.eql(u8, d.name(), dev_name)) return d;
        }
        return null;
    }

    fn getOrOpenDevLocked(self: *Manager, dev_name: []const u8, err_buf: []u8) ?*DevEntry {
        if (self.findDevLocked(dev_name)) |d| return d;
        if (self.devs.items.len >= MAX_QPS) {
            _ = std.fmt.bufPrint(err_buf, "too many open RDMA devices\x00", .{}) catch {};
            return null;
        }
        if (dev_name.len == 0 or dev_name.len >= 64) {
            _ = std.fmt.bufPrint(err_buf, "invalid rdma_dev name\x00", .{}) catch {};
            return null;
        }
        const h = self.transport.dev_open(self.transport.ctx, dev_name, err_buf) orelse return null;
        const entry = self.alloc().create(DevEntry) catch {
            self.transport.dev_close(self.transport.ctx, h);
            _ = std.fmt.bufPrint(err_buf, "out of memory storing rdma dev\x00", .{}) catch {};
            return null;
        };
        entry.* = .{ .handle = h, .name_len = dev_name.len, .refcount = 1 };
        @memcpy(entry.name_buf[0..dev_name.len], dev_name);
        self.devs.append(self.alloc(), entry) catch {
            self.transport.dev_close(self.transport.ctx, h);
            self.alloc().destroy(entry);
            _ = std.fmt.bufPrint(err_buf, "out of memory storing rdma dev\x00", .{}) catch {};
            return null;
        };
        return entry;
    }

    fn findQpIndexLocked(self: *Manager, peer_id: []const u8, dev_name: []const u8, role: QpRole) ?usize {
        for (self.qps.items, 0..) |q, idx| {
            if (q.role == role and std.mem.eql(u8, q.peerId(), peer_id) and std.mem.eql(u8, q.devName(), dev_name)) return idx;
        }
        return null;
    }

    fn findQpEntryIndexLocked(self: *Manager, target: *const QpEntry) ?usize {
        for (self.qps.items, 0..) |q, idx| {
            if (q == target) return idx;
        }
        return null;
    }

    fn createQpLocked(
        self: *Manager,
        io: std.Io,
        dev: *DevEntry,
        peer_id: []const u8,
        rdma_dev: []const u8,
        role: QpRole,
        local_out: *CEndpoint,
        err_buf: []u8,
    ) ?*QpEntry {
        if (self.qps.items.len >= MAX_QPS) {
            _ = std.fmt.bufPrint(err_buf, "too many active RDMA QPs\x00", .{}) catch {};
            return null;
        }
        const psn = randomPsn(io);
        const q_handle = self.transport.qp_create(self.transport.ctx, dev.handle, psn, role == .send, local_out, err_buf) orelse return null;
        const qe = self.alloc().create(QpEntry) catch {
            self.transport.qp_destroy(self.transport.ctx, q_handle);
            _ = std.fmt.bufPrint(err_buf, "out of memory allocating qp entry\x00", .{}) catch {};
            return null;
        };
        qe.* = .{
            .role = role,
            .dev = dev,
            .handle = q_handle,
            .local = local_out.*,
            .state = .init,
            .peer_id_len = peer_id.len,
            .dev_name_len = rdma_dev.len,
        };
        @memcpy(qe.peer_id_buf[0..peer_id.len], peer_id);
        @memcpy(qe.dev_name_buf[0..rdma_dev.len], rdma_dev);
        self.qps.append(self.alloc(), qe) catch {
            self.transport.qp_destroy(self.transport.ctx, q_handle);
            self.alloc().destroy(qe);
            _ = std.fmt.bufPrint(err_buf, "out of memory storing qp entry\x00", .{}) catch {};
            return null;
        };
        dev.refcount += 1;
        return qe;
    }

    /// Remove QP at `idx` from `self.qps` and either destroy immediately (if `!in_use`) or mark
    /// `retired = true` so the active op destroys it safely on exit without UAF.
    fn retireQpAtLocked(self: *Manager, idx: usize) struct { qp: ?*QpEntry, dev: ?*DevEntry } {
        const q = self.qps.swapRemove(idx);
        q.stop_flag.store(1, .release);
        q.retired = true;
        if (q.in_use) {
            return .{ .qp = null, .dev = null };
        }
        const close_dev = self.decDevRefLocked(q.dev);
        return .{ .qp = q, .dev = close_dev };
    }

    fn destroyRetiredQp(self: *Manager, qp: ?*QpEntry, dev: ?*DevEntry) void {
        if (qp) |q| {
            self.transport.qp_destroy(self.transport.ctx, q.handle);
            self.alloc().destroy(q);
        }
        if (dev) |d| {
            self.transport.dev_close(self.transport.ctx, d.handle);
            self.alloc().destroy(d);
        }
    }

    fn hasActiveOfferForLinkLocked(self: *Manager, peer_id: []const u8, rdma_dev: []const u8) bool {
        for (self.offers.items) |o| {
            if ((o.state == .pinned or o.state == .serving) and
                std.mem.eql(u8, o.peerId(), peer_id) and std.mem.eql(u8, o.devName(), rdma_dev))
            {
                return true;
            }
        }
        return false;
    }

    fn spawnResponderLocked(self: *Manager, io: std.Io, offer: *OfferEntry, q: *QpEntry) bool {
        if (q.in_use or q.state != .rts or q.retired) return false;
        offer.stop_flag.store(0, .release);
        offer.responder_done.store(false, .release);
        offer.state = .serving;
        offer.refcount += 1; // owned by responderMain
        q.in_use = true;
        self.active_ops += 1;
        const th = std.Thread.spawn(.{}, responderMain, .{ResponderCtx{
            .mgr = self,
            .io = io,
            .offer = offer,
            .qp = q,
        }}) catch {
            q.in_use = false;
            self.active_ops -|= 1;
            offer.refcount -= 1;
            offer.state = .pinned;
            return false;
        };
        offer.thread = th;
        return true;
    }

    fn joinAndReleaseOffers(self: *Manager, io: std.Io, items: []JoinItem) void {
        for (items) |it| {
            if (it.thread) |t| t.join();
            const should_free = blk: {
                self.mu.lockUncancelable(io);
                defer self.mu.unlock(io);
                break :blk self.decOfferRefLocked(it.offer);
            };
            if (should_free) it.offer.freeResources(self.alloc());
        }
    }

    fn checkAndRecordConsumedLocked(self: *Manager, transfer_id: []const u8) bool {
        if (transfer_id.len == 0 or transfer_id.len > 128) return false;
        for (&self.consumed) |*c| {
            if (c.len == transfer_id.len and std.mem.eql(u8, c.slice(), transfer_id)) return false;
        }
        var slot = &self.consumed[self.consumed_next];
        slot.len = transfer_id.len;
        @memcpy(slot.buf[0..transfer_id.len], transfer_id);
        self.consumed_next = (self.consumed_next + 1) % CONSUMED_RING_LEN;
        return true;
    }

    /// Blocking sweep that removes all expired offers, joins any running responders outside the
    /// mutex, and releases their memory in batches until none remain expired.
    pub fn sweepExpired(self: *Manager, io: std.Io) usize {
        var total: usize = 0;
        while (true) {
            var batch: [MAX_OFFERS]JoinItem = undefined;
            var n: usize = 0;
            {
                self.mu.lockUncancelable(io);
                defer self.mu.unlock(io);
                const now_ms = self.nowMs(io);
                var i: usize = 0;
                while (i < self.offers.items.len and n < batch.len) {
                    const o = self.offers.items[i];
                    if (now_ms >= o.expires_at_ms) {
                        _ = self.offers.swapRemove(i);
                        o.stop_flag.store(1, .release);
                        const th = o.thread;
                        o.thread = null;
                        // Transfer the `mgr.offers` reference to `batch[n]`.
                        batch[n] = .{ .offer = o, .thread = th };
                        n += 1;
                    } else {
                        i += 1;
                    }
                }
            }
            if (n == 0) break;
            self.joinAndReleaseOffers(io, batch[0..n]);
            total += n;
        }
        return total;
    }

    /// Non-blocking sweep: reclaims expired offers that either have no thread or whose responder
    /// thread has already reached exit (`responder_done`), and signals `stop_flag` on any
    /// still-running expired responders without blocking on an active transfer.
    pub fn sweepExpiredNonBlocking(self: *Manager, io: std.Io) usize {
        var total: usize = 0;
        while (true) {
            var batch: [MAX_OFFERS]JoinItem = undefined;
            var n: usize = 0;
            {
                self.mu.lockUncancelable(io);
                defer self.mu.unlock(io);
                const now_ms = self.nowMs(io);
                var i: usize = 0;
                while (i < self.offers.items.len and n < batch.len) {
                    const o = self.offers.items[i];
                    if (now_ms >= o.expires_at_ms) {
                        o.stop_flag.store(1, .release);
                        if (o.thread == null or o.responder_done.load(.acquire)) {
                            _ = self.offers.swapRemove(i);
                            const th = o.thread;
                            o.thread = null;
                            batch[n] = .{ .offer = o, .thread = th };
                            n += 1;
                            continue;
                        }
                    }
                    i += 1;
                }
            }
            if (n == 0) break;
            self.joinAndReleaseOffers(io, batch[0..n]);
            total += n;
        }
        return total;
    }

    pub fn quiesce(self: *Manager, io: std.Io) struct { qps_closed: usize, offers_cleared: usize } {
        var offers_cleared: usize = 0;

        self.mu.lockUncancelable(io);
        self.quiescing_count += 1;
        for (self.qps.items) |q| q.stop_flag.store(1, .release);
        for (self.offers.items) |o| o.stop_flag.store(1, .release);
        self.mu.unlock(io);

        defer {
            self.mu.lockUncancelable(io);
            self.quiescing_count -|= 1;
            self.mu.unlock(io);
        }

        // Drain all offers in batches and join their responder threads outside the mutex.
        while (true) {
            var batch: [MAX_OFFERS]JoinItem = undefined;
            var n: usize = 0;
            {
                self.mu.lockUncancelable(io);
                defer self.mu.unlock(io);
                while (self.offers.items.len > 0 and n < batch.len) {
                    const o = self.offers.pop().?;
                    o.stop_flag.store(1, .release);
                    const th = o.thread;
                    o.thread = null;
                    batch[n] = .{ .offer = o, .thread = th };
                    n += 1;
                }
            }
            if (n == 0) break;
            self.joinAndReleaseOffers(io, batch[0..n]);
            offers_cleared += n;
        }

        // Wait briefly for any active pulls / handlers to unwind after stop_flag.
        self.mu.lockUncancelable(io);
        const wait_deadline = io_util.nowMsMonotonic(io) + QUIESCE_WAIT_MS;
        while (self.active_ops > 0 and io_util.nowMsMonotonic(io) < wait_deadline) {
            self.mu.unlock(io);
            std.Io.sleep(io, .fromMilliseconds(5), .real) catch {};
            self.mu.lockUncancelable(io);
        }

        // Retire all QPs and remove all devices in bounded batches outside the mutex so
        // nothing can ever be truncated or leaked. Any QP still `in_use` (or device still
        // referenced by an in-use QP) stays alive via refcount and is destroyed safely by
        // the unwinding op's exit path without UAF even if `active_ops > 0`.
        var qps_closed: usize = 0;
        self.mu.unlock(io);

        while (true) {
            var qps_to_destroy: [MAX_QPS]*QpEntry = undefined;
            var qps_destroy_n: usize = 0;
            var devs_to_close: [MAX_QPS]*DevEntry = undefined;
            var devs_close_n: usize = 0;
            {
                self.mu.lockUncancelable(io);
                defer self.mu.unlock(io);
                while (self.qps.items.len > 0 and qps_destroy_n < qps_to_destroy.len and devs_close_n < devs_to_close.len) {
                    const q = self.qps.pop().?;
                    q.stop_flag.store(1, .release);
                    q.retired = true;
                    qps_closed += 1;
                    if (!q.in_use) {
                        qps_to_destroy[qps_destroy_n] = q;
                        qps_destroy_n += 1;
                        if (self.decDevRefLocked(q.dev)) |d| {
                            devs_to_close[devs_close_n] = d;
                            devs_close_n += 1;
                        }
                    }
                }
                if (qps_destroy_n == 0 and devs_close_n == 0 and self.qps.items.len == 0) break;
            }
            for (qps_to_destroy[0..qps_destroy_n]) |q| {
                self.transport.qp_destroy(self.transport.ctx, q.handle);
                self.alloc().destroy(q);
            }
            for (devs_to_close[0..devs_close_n]) |d| {
                self.transport.dev_close(self.transport.ctx, d.handle);
                self.alloc().destroy(d);
            }
        }

        while (true) {
            var devs_to_close: [MAX_QPS]*DevEntry = undefined;
            var devs_close_n: usize = 0;
            {
                self.mu.lockUncancelable(io);
                defer self.mu.unlock(io);
                while (self.devs.items.len > 0 and devs_close_n < devs_to_close.len) {
                    const d = self.devs.pop().?;
                    if (self.decDevRefLocked(d)) |dead| {
                        devs_to_close[devs_close_n] = dead;
                        devs_close_n += 1;
                    }
                }
                if (devs_close_n == 0 and self.devs.items.len == 0) break;
            }
            for (devs_to_close[0..devs_close_n]) |d| {
                self.transport.dev_close(self.transport.ctx, d.handle);
                self.alloc().destroy(d);
            }
        }

        return .{ .qps_closed = qps_closed, .offers_cleared = offers_cleared };
    }

    pub fn deinitAll(self: *Manager, io: std.Io) void {
        _ = self.quiesce(io);
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        self.qps.deinit(self.alloc());
        self.offers.deinit(self.alloc());
        self.devs.deinit(self.alloc());
        self.consumed = @splat(.{});
        self.consumed_next = 0;
    }

    pub fn activeOfferCount(self: *Manager, io: std.Io) usize {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.offers.items.len;
    }

    pub fn activeQpCount(self: *Manager, io: std.Io) usize {
        self.mu.lockUncancelable(io);
        defer self.mu.unlock(io);
        return self.qps.items.len;
    }
};

var g_manager: Manager = .{};

pub fn isRdmaPath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, prefix);
}

pub fn isAvailable() bool {
    return g_manager.isAvailable();
}

pub fn sweepExpired(io: std.Io) void {
    _ = g_manager.sweepExpired(io);
}

pub fn sweepExpiredNonBlocking(io: std.Io) void {
    _ = g_manager.sweepExpiredNonBlocking(io);
}

pub fn quiesceGlobal(io: std.Io) void {
    _ = g_manager.quiesce(io);
}

// ── Hex / GID / Transfer-ID / Identifier Helpers ─────────────────────────────

fn isSafeIdent(s: []const u8) bool {
    if (s.len == 0 or s.len >= 64) return false;
    for (s) |c| {
        const ok = (c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or
            c == '_' or c == '-' or c == '.' or c == ':';
        if (!ok) return false;
    }
    return true;
}

fn formatGidHex(gid: [16]u8) [32]u8 {
    return std.fmt.bytesToHex(gid, .lower);
}

fn parseGidHex(s: []const u8) ?[16]u8 {
    var out: [16]u8 = undefined;
    if (s.len == 32) {
        _ = std.fmt.hexToBytes(&out, s) catch return null;
        return out;
    }
    var compact: [32]u8 = undefined;
    var n: usize = 0;
    for (s) |c| {
        if (c == ':') continue;
        if (n >= 32) return null;
        compact[n] = c;
        n += 1;
    }
    if (n != 32) return null;
    _ = std.fmt.hexToBytes(&out, &compact) catch return null;
    return out;
}

fn randomTransferId(io: std.Io) struct { hex: [32]u8, raw: [16]u8 } {
    var rnd: [16]u8 = undefined;
    io.random(&rnd);
    return .{ .hex = std.fmt.bytesToHex(rnd, .lower), .raw = rnd };
}

fn randomPsn(io: std.Io) u32 {
    var rnd: [4]u8 = undefined;
    io.random(&rnd);
    return std.mem.readInt(u32, &rnd, .little) & 0x00ffffff;
}

fn transferIdTo16(transfer_id: []const u8) [16]u8 {
    var out: [16]u8 = undefined;
    if (transfer_id.len == 32) {
        if (std.fmt.hexToBytes(&out, transfer_id)) |_| return out else |_| {}
    }
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(transfer_id, &digest, .{});
    @memcpy(&out, digest[0..16]);
    return out;
}

fn percentDecodeSegment(out: []u8, raw: []const u8) ?[]const u8 {
    var i: usize = 0;
    var n: usize = 0;
    while (i < raw.len) {
        if (n >= out.len) return null;
        if (raw[i] == '%') {
            if (i + 2 >= raw.len) return null;
            const hi = std.fmt.charToDigit(raw[i + 1], 16) catch return null;
            const lo = std.fmt.charToDigit(raw[i + 2], 16) catch return null;
            out[n] = (hi << 4) | lo;
            n += 1;
            i += 3;
        } else {
            out[n] = raw[i];
            n += 1;
            i += 1;
        }
    }
    return out[0..n];
}

fn elapsedMsF64(start_us: i64, end_us: i64) f64 {
    const delta_us = @max(end_us - start_us, 0);
    return @as(f64, @floatFromInt(delta_us)) / 1000.0;
}

// ── HTTP Entrypoint ──────────────────────────────────────────────────────────

/// Dispatch `/v1/cache/rdma/*`. Gated by the process-wide `g_cache_transfer` policy:
///   - disabled (default) -> 404
///   - `--cache-transfer` -> loopback peers only (403 for remote; respects `--api-key-strict`)
///   - `--cache-transfer-key-file` -> `Authorization: Bearer <key>` from any peer (401 on mismatch)
pub fn handleRequest(
    allocator: std.mem.Allocator,
    stream: *Conn,
    method: []const u8,
    raw_path: []const u8,
    path: []const u8,
    raw_headers: []const u8,
    body: []const u8,
) !void {
    return handleRequestOn(&g_manager, allocator, stream, method, raw_path, path, raw_headers, body);
}

pub fn handleRequestOn(
    mgr: *Manager,
    allocator: std.mem.Allocator,
    stream: *Conn,
    method: []const u8,
    raw_path: []const u8,
    path: []const u8,
    raw_headers: []const u8,
    body: []const u8,
) !void {
    const verdict = auth.checkCacheTransferAuth(
        http_server.g_cache_transfer,
        stream.stream.socket.address,
        raw_headers,
        raw_path,
    );
    if (verdict != .allowed) {
        try auth.sendCacheTransferDenial(allocator, stream, verdict, true);
        return;
    }

    if (std.mem.eql(u8, path, "/v1/cache/rdma/info")) {
        if (!std.mem.eql(u8, method, "GET")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Method not allowed", 405);
            return;
        }
        return handleInfo(mgr, allocator, stream);
    }

    if (std.mem.eql(u8, path, "/v1/cache/rdma/quiesce")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Method not allowed", 405);
            return;
        }
        return handleQuiesce(mgr, stream);
    }

    if (std.mem.startsWith(u8, path, offer_delete_prefix)) {
        if (!std.mem.eql(u8, method, "DELETE")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Method not allowed", 405);
            return;
        }
        const raw_id = path[offer_delete_prefix.len..];
        const clean_val = auth.queryParamValue(raw_path, "clean");
        const clean = if (clean_val) |v| std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "true") else false;
        return handleDeleteOffer(mgr, allocator, stream, raw_id, clean);
    }

    if (std.mem.eql(u8, path, "/v1/cache/rdma/offer")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Method not allowed", 405);
            return;
        }
        if (mgr.distributedClusterActive()) {
            try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "server_error", "RDMA cache transfer is unavailable while a distributed cluster is active", 503, "rdma_distributed_active");
            return;
        }
        return handleOffer(mgr, allocator, stream, body);
    }

    if (std.mem.eql(u8, path, "/v1/cache/rdma/accept")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Method not allowed", 405);
            return;
        }
        if (mgr.distributedClusterActive()) {
            try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "server_error", "RDMA cache transfer is unavailable while a distributed cluster is active", 503, "rdma_distributed_active");
            return;
        }
        return handleAccept(mgr, allocator, stream, body);
    }

    if (std.mem.eql(u8, path, "/v1/cache/rdma/pull")) {
        if (!std.mem.eql(u8, method, "POST")) {
            try sendErrorResponse(allocator, stream, "405 Method Not Allowed", "invalid_request_error", "Method not allowed", 405);
            return;
        }
        if (mgr.distributedClusterActive()) {
            try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "server_error", "RDMA cache transfer is unavailable while a distributed cluster is active", 503, "rdma_distributed_active");
            return;
        }
        return handlePull(mgr, allocator, stream, body);
    }

    try sendErrorResponse(allocator, stream, "404 Not Found", "not_found", "Unknown RDMA cache endpoint", 404);
}

// ── GET /v1/cache/rdma/info ──────────────────────────────────────────────────

fn handleInfo(mgr: *Manager, allocator: std.mem.Allocator, stream: *Conn) !void {
    const avail = mgr.isAvailable();
    const dist_active = mgr.distributedClusterActive();
    var devs: [MAX_LIST_DEVICES][64]u8 = @splat(@splat(0));
    const n_devs = if (avail) mgr.listDevices(&devs) else 0;

    var out: std.Io.Writer.Allocating = .init(allocator);
    defer out.deinit();
    try out.writer.print("{{\"available\":{},\"protocol\":\"{s}\",\"distributed_active\":{},\"server_version\":{f},\"devices\":[", .{
        avail,
        protocol_id,
        dist_active,
        std.json.fmt(build_options.version, .{}),
    });
    for (devs[0..n_devs], 0..) |*name_buf, i| {
        if (i != 0) try out.writer.writeByte(',');
        try out.writer.print("{f}", .{std.json.fmt(std.mem.sliceTo(name_buf, 0), .{})});
    }
    try out.writer.writeAll("]}");
    try sendResponse(stream, "200 OK", "application/json", out.written());
}

const HttpErr = struct {
    status: []const u8,
    err_type: []const u8,
    message: []const u8,
    code: u32,
    reason: ?[]const u8 = null,

    fn send(self: HttpErr, allocator: std.mem.Allocator, stream: *Conn) !void {
        try sendErrorResponseWithReason(allocator, stream, self.status, self.err_type, self.message, self.code, self.reason);
    }
};

// ── POST /v1/cache/rdma/offer ────────────────────────────────────────────────

fn handleOffer(mgr: *Manager, allocator: std.mem.Allocator, stream: *Conn, body: []const u8) !void {
    if (!mgr.isAvailable()) {
        try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "server_error", "RDMA transport unavailable on this host", 503, "rdma_unavailable");
        return;
    }

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON body", 400, "bad_request");
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400, "bad_request");
        return;
    }
    const obj = parsed.value.object;
    const session_key = getNonEmptyString(obj, "session_key") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing session_key", 400, "bad_request");
        return;
    };
    const peer_id = getNonEmptyString(obj, "peer_id") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing peer_id", 400, "bad_request");
        return;
    };
    const rdma_dev = getNonEmptyString(obj, "rdma_dev") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing rdma_dev", 400, "bad_request");
        return;
    };
    if (!isSafeIdent(peer_id) or !isSafeIdent(rdma_dev)) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid peer_id or rdma_dev identifier", 400, "bad_request");
        return;
    }
    const model_opt = parseOptionalNonEmptyString(obj, "model") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid model", 400, "bad_request");
        return;
    };
    const raw_expires = (parseOptionalPositiveU64(obj, "expires_ms") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid expires_ms", 400, "bad_request");
        return;
    }) orelse 35_000;
    const expires_ms = std.math.clamp(raw_expires, MIN_EXPIRES_MS, MAX_EXPIRES_MS);

    _ = mgr.sweepExpired(stream.io);

    // Reserve an active-op slot and pre-validate RDMA device before running the GPU export.
    var err_buf: [256]u8 = @splat(0);
    const pre_err: ?HttpErr = blk: {
        mgr.mu.lockUncancelable(stream.io);
        defer mgr.mu.unlock(stream.io);
        if (mgr.quiescing_count > 0) {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA transport quiescing", .code = 503, .reason = "rdma_unavailable" };
        }
        if (mgr.offers.items.len >= MAX_OFFERS) {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "Too many pinned RDMA offers", .code = 503, .reason = "rdma_busy" };
        }
        if (mgr.hasActiveOfferForLinkLocked(peer_id, rdma_dev)) {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA link already has an active offer", .code = 503, .reason = "rdma_busy" };
        }
        if (mgr.findQpIndexLocked(peer_id, rdma_dev, .send)) |q_idx| {
            if (mgr.qps.items[q_idx].in_use) {
                break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA sender QP busy", .code = 503, .reason = "rdma_busy" };
            }
        }
        if (mgr.getOrOpenDevLocked(rdma_dev, &err_buf) == null) {
            const detail = cErrSlice(&err_buf);
            const reason: []const u8 = if (std.mem.indexOf(u8, detail, "too many") != null) "rdma_busy" else "rdma_unavailable";
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = if (detail.len > 0) detail else "RDMA device unavailable", .code = 503, .reason = reason };
        }
        mgr.active_ops += 1;
        break :blk null;
    };
    if (pre_err) |e| {
        try e.send(allocator, stream);
        return;
    }
    defer {
        mgr.mu.lockUncancelable(stream.io);
        mgr.active_ops -|= 1;
        mgr.mu.unlock(stream.io);
    }

    const session_tag = Conn.sessionTag(session_key);
    const max_bytes = cache_transfer.max_export_bytes_override orelse router.max_cache_export_bytes;
    const export_alloc = mgr.alloc();
    const outcome = (if (mgr.cache_hooks) |hooks|
        hooks.export_session(hooks.ctx, export_alloc, model_opt, session_tag, null, max_bytes)
    else
        defaultExportSession(stream.io, export_alloc, model_opt, session_tag, null, max_bytes)) catch |err| {
        log.warn("POST /v1/cache/rdma/offer export failed ({s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "503 Service Unavailable", "internal_error", "Cache export failed", 503);
        return;
    };

    const exp = switch (outcome) {
        .not_found => {
            try sendErrorResponseWithReason(allocator, stream, "404 Not Found", "not_found", "No committed entry for this session", 404, "cache_miss");
            return;
        },
        .deadline_passed => {
            try sendErrorResponse(allocator, stream, "408 Request Timeout", "timeout_error", "cache export refused: deadline passed", 408);
            return;
        },
        .transfer_unsupported => {
            try sendErrorResponseWithReason(allocator, stream, "409 Conflict", "conflict", "cache transfer unsupported", 409, "transfer_unsupported");
            return;
        },
        .identity_pending => {
            try sendErrorResponseWithReason(allocator, stream, "409 Conflict", "conflict", "pack identity pending", 409, "identity_pending");
            return;
        },
        .too_large => {
            try sendErrorResponseWithReason(allocator, stream, "413 Payload Too Large", "invalid_request_error", "cache export exceeds max transfer size", 413, "cache_transfer_too_large");
            return;
        },
        .no_memory => {
            try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "insufficient_memory", "cache export refused: insufficient memory headroom", 503, "cache_export_no_memory");
            return;
        },
        .ok => |res| res,
    };

    const reserved_bytes: u64 = if (mgr.cache_hooks == null) @intCast(exp.bytes.len) else 0;
    const buf_ptr: [*]u8 = exp.bytes.ptr;
    const buf_len: usize = exp.bytes.len;
    if (buf_len == 0) {
        export_alloc.free(exp.bytes);
        if (reserved_bytes > 0) http_server.releaseCacheExportBytes(reserved_bytes);
        try sendErrorResponseWithReason(allocator, stream, "404 Not Found", "not_found", "Empty cache export", 404, "cache_miss");
        return;
    }
    var buf_owned = true;
    defer if (buf_owned) {
        export_alloc.free(buf_ptr[0..buf_len]);
        if (reserved_bytes > 0) http_server.releaseCacheExportBytes(reserved_bytes);
    };

    const tid = randomTransferId(stream.io);
    var local_ep: CEndpoint = .{};
    var qp_mode: []const u8 = "accept";
    var failed_offer: ?*OfferEntry = null;

    const pin_err: ?HttpErr = blk: {
        mgr.mu.lockUncancelable(stream.io);
        defer mgr.mu.unlock(stream.io);
        if (mgr.quiescing_count > 0) {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA transport quiescing", .code = 503, .reason = "rdma_unavailable" };
        }
        // Re-check capacity and per-link exclusion under lock #2 to eliminate TOCTOU races.
        if (mgr.offers.items.len >= MAX_OFFERS) {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "Too many pinned RDMA offers", .code = 503, .reason = "rdma_busy" };
        }
        if (mgr.hasActiveOfferForLinkLocked(peer_id, rdma_dev)) {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA link already has an active offer", .code = 503, .reason = "rdma_busy" };
        }
        const dev = mgr.findDevLocked(rdma_dev) orelse {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA device closed", .code = 503, .reason = "rdma_unavailable" };
        };

        var qp_entry: *QpEntry = undefined;
        if (mgr.findQpIndexLocked(peer_id, rdma_dev, .send)) |existing| {
            qp_entry = mgr.qps.items[existing];
            if (qp_entry.in_use) {
                break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA sender QP busy", .code = 503, .reason = "rdma_busy" };
            }
            local_ep = qp_entry.local;
            qp_mode = if (qp_entry.state == .rts) "persistent" else "accept";
        } else {
            qp_entry = mgr.createQpLocked(stream.io, dev, peer_id, rdma_dev, .send, &local_ep, &err_buf) orelse {
                const detail = cErrSlice(&err_buf);
                const reason: []const u8 = if (std.mem.indexOf(u8, detail, "too many") != null) "rdma_busy" else "rdma_unavailable";
                break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = if (detail.len > 0) detail else "ibv_create_qp failed", .code = 503, .reason = reason };
            };
            qp_mode = "accept";
        }

        const now_ms = mgr.nowMs(stream.io);
        const oe = mgr.alloc().create(OfferEntry) catch {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "internal_error", .message = "Out of memory storing offer", .code = 503 };
        };
        oe.* = .{
            .transfer_id = tid.hex,
            .tid_16 = tid.raw,
            .peer_id_len = peer_id.len,
            .dev_name_len = rdma_dev.len,
            .buf_ptr = buf_ptr,
            .buf_len = buf_len,
            .buf_allocator = export_alloc,
            .reserved_export_bytes = reserved_bytes,
            .expires_at_ms = now_ms +| @as(i64, @intCast(expires_ms)),
            .timeout_ms = expires_ms,
            .state = .pinned,
            .refcount = 1,
        };
        @memcpy(oe.peer_id_buf[0..peer_id.len], peer_id);
        @memcpy(oe.dev_name_buf[0..rdma_dev.len], rdma_dev);

        mgr.offers.append(mgr.alloc(), oe) catch {
            mgr.alloc().destroy(oe);
            break :blk .{ .status = "503 Service Unavailable", .err_type = "internal_error", .message = "Out of memory storing offer", .code = 503 };
        };
        buf_owned = false;

        // If the persistent sender QP is already RTS, arm the UC responder BEFORE returning 200
        // so dst:/pull can immediately send its REQ doorbell without racing.
        if (qp_entry.state == .rts) {
            if (!mgr.spawnResponderLocked(stream.io, oe, qp_entry)) {
                _ = mgr.offers.pop();
                failed_offer = oe;
                break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "Failed to arm RDMA responder", .code = 503, .reason = "rdma_unavailable" };
            }
        }
        break :blk null;
    };
    if (failed_offer) |fo| fo.freeResources(mgr.alloc());
    if (pin_err) |e| {
        try e.send(allocator, stream);
        return;
    }

    const gid_hex = formatGidHex(local_ep.gid);
    var out_buf: [1024]u8 = undefined;
    const import_peak: u64 = if (exp.planned_peak > 0) exp.planned_peak else @intCast(buf_len);
    const resp = try std.fmt.bufPrint(
        &out_buf,
        "{{\"transfer_id\":\"{s}\",\"server_version\":{f},\"byte_len\":{d},\"import_peak\":{d},\"token_count\":{d},\"rdma_dev\":\"{s}\",\"qp_mode\":\"{s}\",\"gid\":\"{s}\",\"qpn\":{d},\"psn\":{d},\"regions\":[{{\"addr\":0,\"rkey\":0,\"len\":{d}}}],\"expires_ms\":{d}}}",
        .{
            &tid.hex,
            std.json.fmt(build_options.version, .{}),
            buf_len,
            import_peak,
            exp.n,
            rdma_dev,
            qp_mode,
            &gid_hex,
            local_ep.qpn,
            local_ep.psn,
            buf_len,
            expires_ms,
        },
    );
    try sendResponse(stream, "200 OK", "application/json", resp);
}

// ── POST /v1/cache/rdma/accept ───────────────────────────────────────────────

fn handleAccept(mgr: *Manager, allocator: std.mem.Allocator, stream: *Conn, body: []const u8) !void {
    if (!mgr.isAvailable()) {
        try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "server_error", "RDMA transport unavailable on this host", 503, "rdma_unavailable");
        return;
    }

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON body", 400, "bad_request");
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400, "bad_request");
        return;
    }
    const obj = parsed.value.object;
    const peer_id = getNonEmptyString(obj, "peer_id") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing peer_id", 400, "bad_request");
        return;
    };
    const rdma_dev = getNonEmptyString(obj, "rdma_dev") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing rdma_dev", 400, "bad_request");
        return;
    };
    if (!isSafeIdent(peer_id) or !isSafeIdent(rdma_dev)) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid peer_id or rdma_dev identifier", 400, "bad_request");
        return;
    }
    const reset = (parseOptionalBool(obj, "reset") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid reset boolean", 400, "bad_request");
        return;
    }) orelse false;
    const transfer_id_opt = parseOptionalNonEmptyString(obj, "transfer_id") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid transfer_id", 400, "bad_request");
        return;
    };
    const peer_gid_str = parseOptionalNonEmptyString(obj, "peer_gid") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid peer_gid", 400, "bad_request");
        return;
    };
    const peer_qpn_opt = parseOptionalU32(obj, "peer_qpn") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid peer_qpn", 400, "bad_request");
        return;
    };
    const peer_psn = ((parseOptionalU32(obj, "peer_psn") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid peer_psn", 400, "bad_request");
        return;
    }) orelse DEFAULT_PSN) & 0x00ffffff;
    const peer_lid: u16 = blk: {
        if (obj.get("peer_lid")) |v| {
            if (v == .null) break :blk 0;
            if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u16)) {
                try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid peer_lid (must be 0..65535)", 400, "bad_request");
                return;
            }
            break :blk @intCast(v.integer);
        }
        break :blk 0;
    };

    _ = mgr.sweepExpired(stream.io);

    var err_buf: [256]u8 = @splat(0);
    var local_ep: CEndpoint = .{};
    var status_str: []const u8 = "init";

    // Reset-only call (src self-healing after dst restart): stop & join any responder waiting on
    // the old sender QP (holding an OfferEntry ref while joining so concurrent sweep/delete cannot
    // UAF), recreate a fresh INIT sender QP with a fresh random PSN, and return its endpoint.
    if (reset and peer_gid_str == null and peer_qpn_opt == null) {
        while (true) {
            var join_batch: [MAX_OFFERS]JoinItem = undefined;
            var join_n: usize = 0;
            {
                mgr.mu.lockUncancelable(stream.io);
                defer mgr.mu.unlock(stream.io);
                for (mgr.offers.items) |o| {
                    if (std.mem.eql(u8, o.peerId(), peer_id) and std.mem.eql(u8, o.devName(), rdma_dev)) {
                        o.stop_flag.store(1, .release);
                        if (o.thread) |t| {
                            if (join_n < join_batch.len) {
                                o.thread = null;
                                o.refcount += 1; // pin OfferEntry while joining outside mutex
                                join_batch[join_n] = .{ .offer = o, .thread = t };
                                join_n += 1;
                            }
                        }
                        if (o.state == .serving) o.state = .pinned;
                    }
                }
            }
            if (join_n == 0) break;
            mgr.joinAndReleaseOffers(stream.io, join_batch[0..join_n]);
        }

        var retired_qp: ?*QpEntry = null;
        var retired_dev: ?*DevEntry = null;
        const reset_err: ?HttpErr = blk: {
            mgr.mu.lockUncancelable(stream.io);
            defer mgr.mu.unlock(stream.io);
            if (mgr.quiescing_count > 0) {
                break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA transport quiescing", .code = 503, .reason = "rdma_unavailable" };
            }
            const dev = mgr.getOrOpenDevLocked(rdma_dev, &err_buf) orelse {
                const detail = cErrSlice(&err_buf);
                const reason: []const u8 = if (std.mem.indexOf(u8, detail, "too many") != null) "rdma_busy" else "rdma_unavailable";
                break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = if (detail.len > 0) detail else "RDMA device unavailable", .code = 503, .reason = reason };
            };
            if (mgr.findQpIndexLocked(peer_id, rdma_dev, .send)) |idx| {
                if (mgr.qps.items[idx].in_use) {
                    break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "QP currently busy", .code = 503, .reason = "rdma_busy" };
                }
                const ret = mgr.retireQpAtLocked(idx);
                retired_qp = ret.qp;
                retired_dev = ret.dev;
            }
            if (mgr.createQpLocked(stream.io, dev, peer_id, rdma_dev, .send, &local_ep, &err_buf) == null) {
                const detail = cErrSlice(&err_buf);
                const reason: []const u8 = if (std.mem.indexOf(u8, detail, "too many") != null) "rdma_busy" else "rdma_unavailable";
                break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = if (detail.len > 0) detail else "ibv_create_qp failed", .code = 503, .reason = reason };
            }
            status_str = "init";
            break :blk null;
        };
        mgr.destroyRetiredQp(retired_qp, retired_dev);
        if (reset_err) |e| {
            try e.send(allocator, stream);
            return;
        }
    } else {
        const pg_str = peer_gid_str orelse {
            try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing peer_gid", 400, "bad_request");
            return;
        };
        const peer_gid = parseGidHex(pg_str) orelse {
            try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid peer_gid hex", 400, "bad_request");
            return;
        };
        const peer_qpn = peer_qpn_opt orelse {
            try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing peer_qpn", 400, "bad_request");
            return;
        };

        var retired_qp: ?*QpEntry = null;
        var retired_dev: ?*DevEntry = null;
        var failed_qp: ?*QpEntry = null;
        var failed_dev: ?*DevEntry = null;
        const accept_err: ?HttpErr = blk: {
            mgr.mu.lockUncancelable(stream.io);
            defer mgr.mu.unlock(stream.io);
            if (mgr.quiescing_count > 0) {
                break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA transport quiescing", .code = 503, .reason = "rdma_unavailable" };
            }
            const dev = mgr.getOrOpenDevLocked(rdma_dev, &err_buf) orelse {
                const detail = cErrSlice(&err_buf);
                const reason: []const u8 = if (std.mem.indexOf(u8, detail, "too many") != null) "rdma_busy" else "rdma_unavailable";
                break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = if (detail.len > 0) detail else "RDMA device unavailable", .code = 503, .reason = reason };
            };

            // Explicit role disambiguation:
            //   - `transfer_id` present (and `!reset`): `src:/accept` completing its `.send` QP
            //     handshake and arming that exact offer.
            //   - `transfer_id` absent (or `reset == true`): `dst:/accept` connecting its `.recv` QP.
            const role: QpRole = if (!reset and transfer_id_opt != null) .send else .recv;

            var qp_entry: *QpEntry = undefined;
            if (role == .send) {
                const s_idx = mgr.findQpIndexLocked(peer_id, rdma_dev, .send) orelse {
                    break :blk .{ .status = "409 Conflict", .err_type = "conflict", .message = "No pending INIT sender QP for peer", .code = 409, .reason = "qp_reset" };
                };
                qp_entry = mgr.qps.items[s_idx];
                if (qp_entry.state != .init) {
                    break :blk .{ .status = "409 Conflict", .err_type = "conflict", .message = "Sender QP not in INIT state", .code = 409, .reason = "qp_reset" };
                }
            } else {
                if (mgr.findQpIndexLocked(peer_id, rdma_dev, .recv)) |existing| {
                    const q = mgr.qps.items[existing];
                    if (!reset and q.state == .rts and q.peer_qpn == peer_qpn and q.peer_psn == peer_psn and std.mem.eql(u8, &q.peer_gid, &peer_gid)) {
                        qp_entry = q;
                        local_ep = q.local;
                        status_str = "ready";
                    } else {
                        if (q.in_use) {
                            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "QP currently busy", .code = 503, .reason = "rdma_busy" };
                        }
                        const ret = mgr.retireQpAtLocked(existing);
                        retired_qp = ret.qp;
                        retired_dev = ret.dev;
                        qp_entry = mgr.createQpLocked(stream.io, dev, peer_id, rdma_dev, .recv, &local_ep, &err_buf) orelse {
                            const detail = cErrSlice(&err_buf);
                            const reason: []const u8 = if (std.mem.indexOf(u8, detail, "too many") != null) "rdma_busy" else "rdma_unavailable";
                            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = if (detail.len > 0) detail else "ibv_create_qp failed", .code = 503, .reason = reason };
                        };
                    }
                } else {
                    qp_entry = mgr.createQpLocked(stream.io, dev, peer_id, rdma_dev, .recv, &local_ep, &err_buf) orelse {
                        const detail = cErrSlice(&err_buf);
                        const reason: []const u8 = if (std.mem.indexOf(u8, detail, "too many") != null) "rdma_busy" else "rdma_unavailable";
                        break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = if (detail.len > 0) detail else "ibv_create_qp failed", .code = 503, .reason = reason };
                    };
                }
            }

            if (!std.mem.eql(u8, status_str, "ready")) {
                // On src, verify the target offer is still pinned BEFORE transitioning the QP so a
                // missing/expired offer does not burn the INIT QP.
                var target_offer: ?*OfferEntry = null;
                if (role == .send) {
                    const want_tid = transfer_id_opt.?;
                    for (mgr.offers.items) |o| {
                        if (std.mem.eql(u8, o.idSlice(), want_tid) and o.state == .pinned and o.thread == null and
                            std.mem.eql(u8, o.peerId(), peer_id) and std.mem.eql(u8, o.devName(), rdma_dev))
                        {
                            target_offer = o;
                            break;
                        }
                    }
                    if (target_offer == null) {
                        break :blk .{ .status = "404 Not Found", .err_type = "not_found", .message = "Pinned offer not found for transfer_id", .code = 404, .reason = "cache_miss" };
                    }
                }

                if (!mgr.transport.qp_connect(mgr.transport.ctx, qp_entry.handle, peer_gid, peer_qpn, peer_psn, peer_lid, &err_buf)) {
                    if (mgr.findQpEntryIndexLocked(qp_entry)) |q_idx| {
                        const ret = mgr.retireQpAtLocked(q_idx);
                        failed_qp = ret.qp;
                        failed_dev = ret.dev;
                    }
                    const detail = cErrSlice(&err_buf);
                    break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = if (detail.len > 0) detail else "ibv_modify_qp failed", .code = 503, .reason = "rdma_unavailable" };
                }
                qp_entry.state = .rts;
                qp_entry.peer_gid = peer_gid;
                qp_entry.peer_qpn = peer_qpn;
                qp_entry.peer_psn = peer_psn;
                local_ep = qp_entry.local;
                status_str = "ready";

                if (role == .send) {
                    const o = target_offer.?;
                    if (!mgr.spawnResponderLocked(stream.io, o, qp_entry)) {
                        break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "Failed to arm RDMA responder", .code = 503, .reason = "rdma_unavailable" };
                    }
                }
            }
            break :blk null;
        };
        mgr.destroyRetiredQp(failed_qp, failed_dev);
        mgr.destroyRetiredQp(retired_qp, retired_dev);
        if (accept_err) |e| {
            try e.send(allocator, stream);
            return;
        }
    }

    const gid_hex = formatGidHex(local_ep.gid);
    var out_buf: [512]u8 = undefined;
    const resp = try std.fmt.bufPrint(
        &out_buf,
        "{{\"status\":\"{s}\",\"rdma_dev\":\"{s}\",\"gid\":\"{s}\",\"qpn\":{d},\"psn\":{d}}}",
        .{ status_str, rdma_dev, &gid_hex, local_ep.qpn, local_ep.psn },
    );
    try sendResponse(stream, "200 OK", "application/json", resp);
}

// ── POST /v1/cache/rdma/pull ─────────────────────────────────────────────────

fn handlePull(mgr: *Manager, allocator: std.mem.Allocator, stream: *Conn, body: []const u8) !void {
    if (!mgr.isAvailable()) {
        try sendErrorResponseWithReason(allocator, stream, "503 Service Unavailable", "server_error", "RDMA transport unavailable on this host", 503, "rdma_unavailable");
        return;
    }

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, body, .{}) catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid JSON body", 400, "bad_request");
        return;
    };
    defer parsed.deinit();
    if (parsed.value != .object) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Request body must be a JSON object", 400, "bad_request");
        return;
    }
    const obj = parsed.value.object;
    const transfer_id = getNonEmptyString(obj, "transfer_id") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing transfer_id", 400, "bad_request");
        return;
    };
    if (transfer_id.len > 128) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "transfer_id too long", 400, "bad_request");
        return;
    }
    const peer_id = getNonEmptyString(obj, "peer_id") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing peer_id", 400, "bad_request");
        return;
    };
    const rdma_dev = getNonEmptyString(obj, "rdma_dev") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing rdma_dev", 400, "bad_request");
        return;
    };
    if (!isSafeIdent(peer_id) or !isSafeIdent(rdma_dev)) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid peer_id or rdma_dev identifier", 400, "bad_request");
        return;
    }
    const model_opt = parseOptionalNonEmptyString(obj, "model") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid model", 400, "bad_request");
        return;
    };
    const raw_deadline = (parseOptionalPositiveU64(obj, "deadline_ms") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid deadline_ms", 400, "bad_request");
        return;
    }) orelse 30_000;
    const deadline_ms = std.math.clamp(raw_deadline, 1, MAX_PULL_DEADLINE_MS);
    const offer_val = obj.get("offer") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing offer object", 400, "bad_request");
        return;
    };
    if (offer_val != .object) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "offer must be an object", 400, "bad_request");
        return;
    }
    const offer = offer_val.object;
    const offer_tid = getNonEmptyString(offer, "transfer_id") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing offer.transfer_id", 400, "bad_request");
        return;
    };
    if (!std.mem.eql(u8, transfer_id, offer_tid)) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "transfer_id mismatch", 400, "bad_request");
        return;
    }

    // 1. Optional pre-DMA server_version check -> 409 version_mismatch (0 bytes transferred).
    if (offer.get("server_version")) |sv| {
        if (sv != .null) {
            if (sv != .string or sv.string.len == 0) {
                try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid offer.server_version", 400, "bad_offer");
                return;
            }
            if (!std.mem.eql(u8, sv.string, build_options.version)) {
                try sendErrorResponseWithReason(allocator, stream, "409 Conflict", "conflict", "offer server_version mismatch", 409, "version_mismatch");
                return;
            }
        }
    }

    // 2. Validate byte_len, import_peak, regions, and optional offer.gid/qpn/psn.
    const byte_len = getPositiveU64(offer, "byte_len") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid offer.byte_len", 400, "bad_offer");
        return;
    };
    if (byte_len > router.max_cache_transfer_body_bytes) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "insufficient_memory", "offer byte_len exceeds max cache transfer size", 400, "import_no_room");
        return;
    }
    const parsed_import_peak = parseOptionalPositiveU64(offer, "import_peak") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid offer.import_peak", 400, "bad_offer");
        return;
    };
    const import_peak = @max(parsed_import_peak orelse byte_len, byte_len);

    const regions_val = offer.get("regions") orelse {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Missing offer.regions", 400, "bad_offer");
        return;
    };
    if (regions_val != .array or regions_val.array.items.len == 0 or regions_val.array.items.len > 64) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid offer.regions", 400, "bad_offer");
        return;
    }
    var sum_len: u64 = 0;
    for (regions_val.array.items) |rv| {
        if (rv != .object) {
            try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid region entry", 400, "bad_offer");
            return;
        }
        const ro = rv.object;
        _ = getNonNegativeU64(ro, "addr") orelse {
            try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid region.addr", 400, "bad_offer");
            return;
        };
        _ = (parseOptionalU32(ro, "rkey") catch null) orelse {
            try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid region.rkey", 400, "bad_offer");
            return;
        };
        const r_len = getPositiveU64(ro, "len") orelse {
            try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid region.len", 400, "bad_offer");
            return;
        };
        sum_len = std.math.add(u64, sum_len, r_len) catch {
            try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Region length overflow", 400, "bad_offer");
            return;
        };
    }
    if (sum_len != byte_len) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "offer regions length sum != byte_len", 400, "bad_offer");
        return;
    }

    const offer_qpn = parseOptionalU32(offer, "qpn") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid offer.qpn", 400, "bad_offer");
        return;
    };
    const offer_psn_raw = parseOptionalU32(offer, "psn") catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid offer.psn", 400, "bad_offer");
        return;
    };
    const offer_psn: ?u32 = if (offer_psn_raw) |p| p & 0x00ffffff else null;
    const offer_gid: ?[16]u8 = blk: {
        if (offer.get("gid")) |gv| {
            if (gv == .null) break :blk null;
            if (gv != .string or gv.string.len == 0) {
                try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid offer.gid", 400, "bad_offer");
                return;
            }
            const parsed_gid = parseGidHex(gv.string) orelse {
                try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "Malformed offer.gid hex", 400, "bad_offer");
                return;
            };
            break :blk parsed_gid;
        }
        break :blk null;
    };

    // 3. Pre-DMA headroom + budget check (<1 ms, 0 bytes transferred on refusal -> 400 import_no_room).
    if (!http_server.reserveCacheImportBytes(byte_len)) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "insufficient_memory", "cache import refused: inbound transfer byte budget is full", 400, "import_no_room");
        return;
    }
    defer http_server.releaseCacheImportBytes(byte_len);

    if (!http_request.cacheImportPeakFits(stream.io, import_peak, byte_len, model_opt)) {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "insufficient_memory", "cache import refused: planned import peak exceeds available headroom", 400, "import_no_room");
        return;
    }

    _ = mgr.sweepExpired(stream.io);

    // 4. Acquire connected RTS receiver QP + verify single-use transfer_id under lock (before RDMA).
    var qp_entry: *QpEntry = undefined;
    const acq_err: ?HttpErr = blk: {
        mgr.mu.lockUncancelable(stream.io);
        defer mgr.mu.unlock(stream.io);
        if (mgr.quiescing_count > 0) {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA transport quiescing", .code = 503, .reason = "rdma_unavailable" };
        }
        if (mgr.findDevLocked(rdma_dev) == null) {
            break :blk .{ .status = "409 Conflict", .err_type = "conflict", .message = "RDMA device not initialized for peer QP", .code = 409, .reason = "qp_reset" };
        }
        const q_idx = mgr.findQpIndexLocked(peer_id, rdma_dev, .recv) orelse {
            break :blk .{ .status = "409 Conflict", .err_type = "conflict", .message = "RDMA QP not connected for source peer", .code = 409, .reason = "qp_reset" };
        };
        const q = mgr.qps.items[q_idx];
        if (q.state != .rts) {
            break :blk .{ .status = "409 Conflict", .err_type = "conflict", .message = "RDMA QP not in RTS state", .code = 409, .reason = "qp_reset" };
        }
        if (offer_qpn) |oqpn| {
            if (q.peer_qpn != oqpn) {
                break :blk .{ .status = "409 Conflict", .err_type = "conflict", .message = "RDMA QP peer QPN mismatch (peer restarted)", .code = 409, .reason = "qp_reset" };
            }
        }
        if (offer_psn) |opsn| {
            if (q.peer_psn != opsn) {
                break :blk .{ .status = "409 Conflict", .err_type = "conflict", .message = "RDMA QP peer PSN mismatch (peer restarted)", .code = 409, .reason = "qp_reset" };
            }
        }
        if (offer_gid) |ogid| {
            if (!std.mem.eql(u8, &q.peer_gid, &ogid)) {
                break :blk .{ .status = "409 Conflict", .err_type = "conflict", .message = "RDMA QP peer GID mismatch", .code = 409, .reason = "qp_reset" };
            }
        }
        if (q.in_use) {
            break :blk .{ .status = "503 Service Unavailable", .err_type = "server_error", .message = "RDMA QP busy", .code = 503, .reason = "rdma_busy" };
        }
        if (!mgr.checkAndRecordConsumedLocked(transfer_id)) {
            break :blk .{ .status = "409 Conflict", .err_type = "conflict", .message = "transfer_id already consumed", .code = 409, .reason = "duplicate" };
        }
        q.stop_flag.store(0, .release);
        q.in_use = true;
        mgr.active_ops += 1;
        qp_entry = q;
        break :blk null;
    };
    if (acq_err) |e| {
        try e.send(allocator, stream);
        return;
    }

    var qp_broken = false;
    defer {
        var destroy_qp: ?*QpEntry = null;
        var close_dev: ?*DevEntry = null;
        {
            mgr.mu.lockUncancelable(stream.io);
            defer mgr.mu.unlock(stream.io);
            qp_entry.in_use = false;
            if (qp_entry.retired) {
                destroy_qp = qp_entry;
            } else if (qp_broken) {
                if (mgr.findQpEntryIndexLocked(qp_entry)) |q_idx| {
                    _ = mgr.qps.swapRemove(q_idx);
                    qp_entry.retired = true;
                    destroy_qp = qp_entry;
                }
            }
            if (destroy_qp) |dq| {
                close_dev = mgr.decDevRefLocked(dq.dev);
            }
            mgr.active_ops -|= 1;
        }
        mgr.destroyRetiredQp(destroy_qp, close_dev);
    }

    // 5. Allocate receive buffer (0 ms MR registration — QP uses its pre-registered 4 MiB ring).
    const dst_buf = mgr.alloc().alloc(u8, @intCast(byte_len)) catch {
        try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "insufficient_memory", "Failed to allocate RDMA receive buffer", 400, "import_no_room");
        return;
    };
    defer mgr.alloc().free(dst_buf);

    // 6. Execute credit-paced IBV_QPT_UC pull over Thunderbolt 5.
    var err_buf: [256]u8 = @splat(0);
    const tid_16 = transferIdTo16(transfer_id);
    const t_rdma_start = io_util.nowUsMonotonic(stream.io);
    const now_mono: u64 = @intCast(@max(io_util.nowMsMonotonic(stream.io), 0));
    const deadline_mono = now_mono +| deadline_ms;
    const read_status = mgr.transport.pull_offer(
        mgr.transport.ctx,
        qp_entry.handle,
        dst_buf,
        tid_16,
        &qp_entry.stop_flag,
        deadline_ms,
        &err_buf,
    );
    const t_rdma_end = io_util.nowUsMonotonic(stream.io);

    switch (read_status) {
        .ok => {},
        .timeout => {
            qp_broken = true;
            try sendErrorResponseWithReason(allocator, stream, "504 Gateway Timeout", "timeout_error", "RDMA pull deadline exceeded", 504, "rdma_timeout");
            return;
        },
        .io_error => {
            qp_broken = true;
            const detail = cErrSlice(&err_buf);
            try sendErrorResponseWithReason(allocator, stream, "500 Internal Server Error", "server_error", if (detail.len > 0) detail else "RDMA pull failed", 500, "rdma_read_failed");
            return;
        },
    }

    // 7. Attach received wire envelope to HotPrefixCache via scheduler.importCache
    //    (verifies 64-bit peer_wire Wyhash checksum over all received bytes).
    const t_attach_start = io_util.nowUsMonotonic(stream.io);
    const import_outcome = (if (mgr.cache_hooks) |hooks|
        hooks.import_body(hooks.ctx, allocator, model_opt, dst_buf, deadline_mono)
    else
        defaultImportBody(stream.io, allocator, model_opt, dst_buf, deadline_mono)) catch |err| {
        log.warn("POST /v1/cache/rdma/pull import failed ({s})\n", .{@errorName(err)});
        try sendErrorResponse(allocator, stream, "500 Internal Server Error", "internal_error", "Cache import failed", 500);
        return;
    };
    const t_attach_end = io_util.nowUsMonotonic(stream.io);

    switch (import_outcome) {
        .ok => |stats| {
            const rdma_ms = elapsedMsF64(t_rdma_start, t_rdma_end);
            const attach_ms = elapsedMsF64(t_attach_start, t_attach_end);
            var out_buf: [256]u8 = undefined;
            const resp = try std.fmt.bufPrint(
                &out_buf,
                "{{\"status\":\"imported\",\"bytes\":{d},\"rdma_ms\":{d:.2},\"attach_ms\":{d:.2}}}",
                .{ stats.bytes, rdma_ms, attach_ms },
            );
            try sendResponse(stream, "200 OK", "application/json", resp);
        },
        .refused => |refusal| {
            switch (refusal) {
                .no_memory => try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "insufficient_memory", "cache import refused: insufficient memory headroom", 400, "import_no_room"),
                .stale => try sendErrorResponseWithReason(allocator, stream, "409 Conflict", "conflict", "cache import refused: stale or duplicate entry", 409, "duplicate"),
                .commit_declined => try sendErrorResponseWithReason(allocator, stream, "409 Conflict", "conflict", "cache import refused: commit declined", 409, "commit_declined"),
                .deadline_passed => try sendErrorResponse(allocator, stream, "408 Request Timeout", "timeout_error", "cache import refused: deadline passed", 408),
                .transfer_unsupported => try sendErrorResponseWithReason(allocator, stream, "409 Conflict", "conflict", "cache transfer unsupported", 409, "transfer_unsupported"),
                .identity_pending => try sendErrorResponseWithReason(allocator, stream, "409 Conflict", "conflict", "pack identity pending", 409, "identity_pending"),
                .unsupported_model, .layout_key, .pack_identity, .settings, .quant_config, .import_plan, .import_arrays, .describe_mismatch => try sendErrorResponseWithReason(allocator, stream, "422 Unprocessable Entity", "invalid_request_error", "cache import refused: incompatible snapshot", 422, @tagName(refusal)),
                .bad_envelope, .unknown_version, .truncated_body, .bad_dtype, .bad_shape, .bad_offsets, .bad_checksum, .bad_quant_config => try sendErrorResponseWithReason(allocator, stream, "400 Bad Request", "invalid_request_error", "cache import refused: corrupt envelope", 400, @tagName(refusal)),
            }
        },
    }
}

// ── DELETE /v1/cache/rdma/offer/{id}?clean=1|0 ───────────────────────────────

fn handleDeleteOffer(mgr: *Manager, allocator: std.mem.Allocator, stream: *Conn, raw_id: []const u8, clean: bool) !void {
    var dec_buf: [128]u8 = undefined;
    const tid = percentDecodeSegment(&dec_buf, raw_id) orelse {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Invalid transfer_id", 400);
        return;
    };
    if (tid.len == 0) {
        try sendErrorResponse(allocator, stream, "400 Bad Request", "invalid_request_error", "Empty transfer_id", 400);
        return;
    }

    _ = mgr.sweepExpired(stream.io);

    var join_item: ?JoinItem = null;
    var resp_buf: [256]u8 = undefined;
    const resp: []const u8 = blk: {
        mgr.mu.lockUncancelable(stream.io);
        defer mgr.mu.unlock(stream.io);
        const now_ms = mgr.nowMs(stream.io);
        for (mgr.offers.items, 0..) |o, idx| {
            if (std.mem.eql(u8, o.idSlice(), tid)) {
                o.stop_flag.store(1, .release);
                const th = o.thread;
                o.thread = null;
                if (clean) {
                    _ = mgr.offers.swapRemove(idx);
                    // Transfer `mgr.offers`'s reference to `join_item`.
                    join_item = .{ .offer = o, .thread = th };
                    break :blk "{\"status\":\"released\"}";
                } else {
                    o.state = .quarantined;
                    if (th != null) {
                        // Hold an extra reference while joining outside mutex so a concurrent
                        // sweep/quiesce/clean=1 cannot UAF `o` before `join()` completes.
                        o.refcount += 1;
                        join_item = .{ .offer = o, .thread = th };
                    }
                    const rem = @max(o.expires_at_ms - now_ms, 0);
                    break :blk try std.fmt.bufPrint(&resp_buf, "{{\"status\":\"quarantined\",\"remaining_ms\":{d}}}", .{rem});
                }
            }
        }
        break :blk "{\"status\":\"already_released\"}";
    };

    if (join_item) |item| {
        var single = [1]JoinItem{item};
        mgr.joinAndReleaseOffers(stream.io, &single);
    }
    try sendResponse(stream, "200 OK", "application/json", resp);
}

// ── POST /v1/cache/rdma/quiesce ──────────────────────────────────────────────

fn handleQuiesce(mgr: *Manager, stream: *Conn) !void {
    const res = mgr.quiesce(stream.io);
    var out_buf: [160]u8 = undefined;
    const resp = try std.fmt.bufPrint(
        &out_buf,
        "{{\"status\":\"quiesced\",\"qps_closed\":{d},\"offers_cleared\":{d}}}",
        .{ res.qps_closed, res.offers_cleared },
    );
    try sendResponse(stream, "200 OK", "application/json", resp);
}

// ── JSON Field Helpers ───────────────────────────────────────────────────────

fn getNonEmptyString(obj: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = obj.get(key) orelse return null;
    if (v != .string or v.string.len == 0) return null;
    return v.string;
}

fn parseOptionalNonEmptyString(obj: std.json.ObjectMap, key: []const u8) error{InvalidField}!?[]const u8 {
    const v = obj.get(key) orelse return null;
    if (v == .null) return null;
    if (v != .string or v.string.len == 0) return error.InvalidField;
    return v.string;
}

fn getPositiveU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = obj.get(key) orelse return null;
    if (v != .integer or v.integer <= 0) return null;
    return @intCast(v.integer);
}

fn getNonNegativeU64(obj: std.json.ObjectMap, key: []const u8) ?u64 {
    const v = obj.get(key) orelse return null;
    if (v != .integer or v.integer < 0) return null;
    return @intCast(v.integer);
}

fn parseOptionalPositiveU64(obj: std.json.ObjectMap, key: []const u8) error{InvalidField}!?u64 {
    const v = obj.get(key) orelse return null;
    if (v == .null) return null;
    if (v != .integer or v.integer <= 0) return error.InvalidField;
    return @intCast(v.integer);
}

fn parseOptionalU32(obj: std.json.ObjectMap, key: []const u8) error{InvalidField}!?u32 {
    const v = obj.get(key) orelse return null;
    if (v == .null) return null;
    if (v != .integer or v.integer < 0 or v.integer > std.math.maxInt(u32)) return error.InvalidField;
    return @intCast(v.integer);
}

fn parseOptionalBool(obj: std.json.ObjectMap, key: []const u8) error{InvalidField}!?bool {
    const v = obj.get(key) orelse return null;
    if (v == .null) return null;
    if (v != .bool) return error.InvalidField;
    return v.bool;
}

// ── Hermetic Loopback RDMA Transport & Unit Tests ────────────────────────────

const testing = std.testing;

pub const LoopbackFabric = struct {
    const SLOT_IDLE: u32 = 0;
    const SLOT_ARMED: u32 = 1;
    const SLOT_COPYING: u32 = 2;
    const SLOT_DONE: u32 = 3;

    const FakeQp = struct {
        qpn: u32,
        psn: u32,
        gid: [16]u8,
        is_sender: bool,
        connected: bool = false,
        peer_qpn: u32 = 0,
        peer_psn: u32 = 0,
        peer_gid: [16]u8 = @splat(0),
        alive: bool = true,
        slot_state: std.atomic.Value(u32) = .init(SLOT_IDLE),
        armed_tid: [16]u8 = @splat(0),
        armed_ptr: ?[*]const u8 = null,
        armed_len: usize = 0,
    };
    const FakeDev = struct {
        gid: [16]u8,
    };

    allocator: std.mem.Allocator,
    next_qpn: u32 = 100,
    qps: std.ArrayList(*FakeQp) = .empty,
    devs: std.ArrayList(*FakeDev) = .empty,
    reads_completed: usize = 0,
    bytes_transferred: u64 = 0,
    inject_read_status: ReadStatus = .ok,

    pub fn init(allocator: std.mem.Allocator) LoopbackFabric {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *LoopbackFabric) void {
        for (self.qps.items) |q| self.allocator.destroy(q);
        for (self.devs.items) |d| self.allocator.destroy(d);
        self.qps.deinit(self.allocator);
        self.devs.deinit(self.allocator);
    }

    pub fn transport(self: *LoopbackFabric) Transport {
        return .{
            .ctx = self,
            .available = available,
            .list_devices = listDevices,
            .dev_open = devOpen,
            .dev_close = devClose,
            .qp_create = qpCreate,
            .qp_connect = qpConnect,
            .qp_destroy = qpDestroy,
            .serve_offer = serveOffer,
            .pull_offer = pullOffer,
        };
    }

    fn me(ctx: *anyopaque) *LoopbackFabric {
        return @ptrCast(@alignCast(ctx));
    }

    fn available(_: *anyopaque) bool {
        return true;
    }

    fn listDevices(_: *anyopaque, out: [][64]u8) usize {
        if (out.len == 0) return 0;
        const name = "rdma_en2";
        @memset(&out[0], 0);
        @memcpy(out[0][0..name.len], name);
        return 1;
    }

    fn devOpen(ctx: *anyopaque, dev_name: []const u8, _: []u8) ?*anyopaque {
        const self = me(ctx);
        const d = self.allocator.create(FakeDev) catch return null;
        var gid: [16]u8 = @splat(0);
        gid[10] = 0xff;
        gid[11] = 0xff;
        gid[12] = 192;
        gid[13] = 0;
        gid[14] = 2;
        gid[15] = if (dev_name.len > 0) dev_name[dev_name.len - 1] else 1;
        d.* = .{ .gid = gid };
        self.devs.append(self.allocator, d) catch {
            self.allocator.destroy(d);
            return null;
        };
        return d;
    }

    fn devClose(_: *anyopaque, _: *anyopaque) void {}

    fn qpCreate(ctx: *anyopaque, dev: *anyopaque, psn: u32, is_sender: bool, local_out: *CEndpoint, _: []u8) ?*anyopaque {
        const self = me(ctx);
        const d: *FakeDev = @ptrCast(@alignCast(dev));
        const q = self.allocator.create(FakeQp) catch return null;
        const qpn = self.next_qpn;
        self.next_qpn += 1;
        q.* = .{ .qpn = qpn, .psn = psn & 0x00ffffff, .gid = d.gid, .is_sender = is_sender };
        self.qps.append(self.allocator, q) catch {
            self.allocator.destroy(q);
            return null;
        };
        local_out.* = .{ .gid = d.gid, .qpn = qpn, .psn = q.psn, .lid = 1, .gid_index = 2, .active_mtu = 5 };
        return q;
    }

    fn qpConnect(_: *anyopaque, qp: *anyopaque, peer_gid: [16]u8, peer_qpn: u32, peer_psn: u32, _: u16, _: []u8) bool {
        const q: *FakeQp = @ptrCast(@alignCast(qp));
        if (!q.alive) return false;
        q.connected = true;
        q.peer_gid = peer_gid;
        q.peer_qpn = peer_qpn;
        q.peer_psn = peer_psn & 0x00ffffff;
        return true;
    }

    fn qpDestroy(_: *anyopaque, qp: *anyopaque) void {
        const q: *FakeQp = @ptrCast(@alignCast(qp));
        q.alive = false;
        q.connected = false;
    }

    fn serveOffer(
        _: *anyopaque,
        qp: *anyopaque,
        src_buf: []const u8,
        tid_16: [16]u8,
        stop_flag: *std.atomic.Value(c_int),
        _: u64,
        err_buf: []u8,
    ) ServeStatus {
        const q: *FakeQp = @ptrCast(@alignCast(qp));
        if (!q.alive or !q.connected or !q.is_sender) {
            _ = std.fmt.bufPrint(err_buf, "sender qp not connected\x00", .{}) catch {};
            return .io_error;
        }
        q.armed_tid = tid_16;
        q.armed_ptr = src_buf.ptr;
        q.armed_len = src_buf.len;
        q.slot_state.store(SLOT_ARMED, .release);

        while (true) {
            const st = q.slot_state.load(.acquire);
            if (st == SLOT_DONE) {
                q.armed_ptr = null;
                q.armed_len = 0;
                q.slot_state.store(SLOT_IDLE, .release);
                return .ok;
            }
            if (stop_flag.load(.acquire) != 0) {
                if (q.slot_state.cmpxchgStrong(SLOT_ARMED, SLOT_IDLE, .acq_rel, .acquire) == null) {
                    q.armed_ptr = null;
                    q.armed_len = 0;
                    return .cancelled;
                }
            }
            std.Thread.yield() catch {};
        }
    }

    fn pullOffer(
        ctx: *anyopaque,
        qp: *anyopaque,
        dst_buf: []u8,
        tid_16: [16]u8,
        stop_flag: *std.atomic.Value(c_int),
        _: u64,
        err_buf: []u8,
    ) ReadStatus {
        const self = me(ctx);
        if (self.inject_read_status != .ok) return self.inject_read_status;
        const q: *FakeQp = @ptrCast(@alignCast(qp));
        if (!q.alive or !q.connected or q.is_sender) {
            _ = std.fmt.bufPrint(err_buf, "recv qp not connected\x00", .{}) catch {};
            return .io_error;
        }

        var peer: ?*FakeQp = null;
        for (self.qps.items) |cand| {
            if (cand.alive and cand.connected and cand.is_sender and
                cand.qpn == q.peer_qpn and cand.peer_qpn == q.qpn and
                cand.psn == q.peer_psn and cand.peer_psn == q.psn)
            {
                peer = cand;
                break;
            }
        }
        const src_qp = peer orelse {
            _ = std.fmt.bufPrint(err_buf, "peer sender qp not in RTS or PSN mismatch\x00", .{}) catch {};
            return .io_error;
        };

        var waited: usize = 0;
        while (waited < 40_000) : (waited += 1) {
            if (stop_flag.load(.acquire) != 0) return .timeout;
            if (src_qp.slot_state.cmpxchgStrong(SLOT_ARMED, SLOT_COPYING, .acq_rel, .acquire) == null) {
                if (!std.mem.eql(u8, &src_qp.armed_tid, &tid_16) or src_qp.armed_len != dst_buf.len or src_qp.armed_ptr == null) {
                    src_qp.slot_state.store(SLOT_ARMED, .release);
                    _ = std.fmt.bufPrint(err_buf, "armed offer tid/len mismatch\x00", .{}) catch {};
                    return .io_error;
                }
                @memcpy(dst_buf, src_qp.armed_ptr.?[0..dst_buf.len]);
                src_qp.slot_state.store(SLOT_DONE, .release);
                self.reads_completed += 1;
                self.bytes_transferred += dst_buf.len;
                return .ok;
            }
            std.Thread.yield() catch {};
        }
        _ = std.fmt.bufPrint(err_buf, "timed out waiting for sender responder\x00", .{}) catch {};
        return .timeout;
    }
};

const TestCacheState = struct {
    stored_session: u64 = 0,
    payload: []const u8 = "",
    imported: ?[]u8 = null,
    last_export_model: ?[]const u8 = null,
    last_import_model: ?[]const u8 = null,
    import_refusal: ?requests.CacheImportRefusal = null,
    allocator: std.mem.Allocator,

    fn hooks(self: *TestCacheState) CacheHooks {
        return .{ .ctx = self, .export_session = expFn, .import_body = impFn };
    }

    fn expFn(ctx: *anyopaque, a: std.mem.Allocator, model: ?[]const u8, session: u64, _: ?u64, _: usize) !requests.CacheExportOutcome {
        const self: *TestCacheState = @ptrCast(@alignCast(ctx));
        self.last_export_model = model;
        if (session != self.stored_session or self.payload.len == 0) return .not_found;
        const copy = try a.dupe(u8, self.payload);
        return .{ .ok = .{ .bytes = copy, .n = 42, .export_ms = 1, .planned_peak = copy.len * 2 } };
    }

    fn impFn(ctx: *anyopaque, _: std.mem.Allocator, model: ?[]const u8, body: []const u8, _: ?u64) !requests.CacheImportOutcome {
        const self: *TestCacheState = @ptrCast(@alignCast(ctx));
        self.last_import_model = model;
        if (self.import_refusal) |r| return .{ .refused = r };
        if (self.imported) |old| self.allocator.free(old);
        self.imported = try self.allocator.dupe(u8, body);
        return .{ .ok = .{ .n = 42, .bytes = body.len, .decode_ms = 0, .admit_ms = 0, .import_ms = 1, .commit_ms = 0 } };
    }

    fn deinit(self: *TestCacheState) void {
        if (self.imported) |b| self.allocator.free(b);
    }
};

const TestReply = struct {
    status: u16,
    body: []u8,

    fn deinit(self: TestReply, a: std.mem.Allocator) void {
        a.free(self.body);
    }
};

fn invokeHttpFrom(
    mgr: *Manager,
    a: std.mem.Allocator,
    peer_addr: std.Io.net.IpAddress,
    method: []const u8,
    raw_path: []const u8,
    auth_header: ?[]const u8,
    body: []const u8,
) !TestReply {
    var sockets: [2]std.posix.fd_t = undefined;
    try testing.expectEqual(@as(c_int, 0), std.c.socketpair(1, 1, 0, &sockets));
    defer _ = std.c.close(sockets[1]);
    var conn: Conn = undefined;
    Conn.init(&conn, .{ .socket = .{ .handle = sockets[0], .address = peer_addr } }, testing.io);
    defer conn.close();

    var hdr_buf: [512]u8 = undefined;
    const hdrs = if (auth_header) |h|
        try std.fmt.bufPrint(&hdr_buf, "{s} {s} HTTP/1.1\r\nAuthorization: {s}\r\n", .{ method, raw_path, h })
    else
        try std.fmt.bufPrint(&hdr_buf, "{s} {s} HTTP/1.1\r\n", .{ method, raw_path });
    const path = if (std.mem.indexOfScalar(u8, raw_path, '?')) |q| raw_path[0..q] else raw_path;

    try handleRequestOn(mgr, a, &conn, method, raw_path, path, hdrs, body);
    try conn.flush();
    try testing.expectEqual(@as(c_int, 0), std.c.shutdown(sockets[0], 1));

    var raw_out: [8192]u8 = undefined;
    const n = std.c.recv(sockets[1], &raw_out, raw_out.len, std.c.MSG.WAITALL);
    try testing.expect(n > 0);
    const resp = raw_out[0..@intCast(n)];
    const first_sp = std.mem.indexOfScalar(u8, resp, ' ') orelse return error.MalformedResponse;
    const status = try std.fmt.parseInt(u16, resp[first_sp + 1 .. first_sp + 4], 10);
    const body_part = (std.mem.cut(u8, resp, "\r\n\r\n") orelse return error.MalformedResponse)[1];
    return .{ .status = status, .body = try a.dupe(u8, body_part) };
}

fn invokeHttp(
    mgr: *Manager,
    a: std.mem.Allocator,
    method: []const u8,
    raw_path: []const u8,
    auth_header: ?[]const u8,
    body: []const u8,
) !TestReply {
    return invokeHttpFrom(mgr, a, .{ .ip4 = std.Io.net.Ip4Address.loopback(0) }, method, raw_path, auth_header, body);
}

var test_clock_ms: i64 = 10_000;
fn fakeClock(_: std.Io) i64 {
    return test_clock_ms;
}

fn setupTransferKey(key_str: []const u8) auth.CacheTransferConfig {
    const old = http_server.g_cache_transfer;
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(key_str, &digest, .{});
    http_server.g_cache_transfer = .{ .key = digest };
    return old;
}

test "cache_rdma: generic cache-transfer auth enforces disabled (404), loopback-only (403 remote), and key-file (401)" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    const a = testing.allocator;
    var fabric = LoopbackFabric.init(a);
    defer fabric.deinit();
    var mgr: Manager = .{ .transport = fabric.transport() };
    defer mgr.deinitAll(testing.io);

    const old_ct = http_server.g_cache_transfer;
    defer http_server.g_cache_transfer = old_ct;

    const lan_peer: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 192, 0, 2, 10 }, .port = 0 } };

    // Disabled (default) -> 404 even on loopback.
    http_server.g_cache_transfer = .disabled;
    const r404 = try invokeHttp(&mgr, a, "GET", "/v1/cache/rdma/info", "Bearer secret-key", "");
    defer r404.deinit(a);
    try testing.expectEqual(@as(u16, 404), r404.status);

    // Loopback-only mode admits loopback and rejects non-loopback (403).
    http_server.g_cache_transfer = .loopback;
    const r403 = try invokeHttpFrom(&mgr, a, lan_peer, "GET", "/v1/cache/rdma/info", null, "");
    defer r403.deinit(a);
    try testing.expectEqual(@as(u16, 403), r403.status);

    const r_lo = try invokeHttp(&mgr, a, "GET", "/v1/cache/rdma/info", null, "");
    defer r_lo.deinit(a);
    try testing.expectEqual(@as(u16, 200), r_lo.status);
    try testing.expect(std.mem.indexOf(u8, r_lo.body, "\"protocol\":\"uc_ring_v1\"") != null);
    try testing.expect(std.mem.indexOf(u8, r_lo.body, "\"rdma_en2\"") != null);

    // Key-file mode requires exact Bearer token from any peer (including loopback).
    _ = setupTransferKey("rdma-transfer-key-0123456789abcdef");
    for ([_]?[]const u8{ null, "Bearer wrong-key", "Bearer serving-key", "rdma-transfer-key-0123456789abcdef" }) |hdr| {
        const r = try invokeHttp(&mgr, a, "POST", "/v1/cache/rdma/quiesce", hdr, "{}");
        defer r.deinit(a);
        try testing.expectEqual(@as(u16, 401), r.status);
    }

    const ok_lan = try invokeHttpFrom(&mgr, a, lan_peer, "POST", "/v1/cache/rdma/quiesce", "Bearer rdma-transfer-key-0123456789abcdef", "{}");
    defer ok_lan.deinit(a);
    try testing.expectEqual(@as(u16, 200), ok_lan.status);
    try testing.expect(std.mem.indexOf(u8, ok_lan.body, "\"status\":\"quiesced\"") != null);
}

test "cache_rdma: full offer -> accept -> pull -> delete(clean=1) -> persistent reuse -> quiesce over loopback fabric" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    const a = testing.allocator;
    test_clock_ms = 50_000;

    const old_ct = setupTransferKey("rdma-transfer-key-0123456789abcdef");
    defer http_server.g_cache_transfer = old_ct;
    const auth_hdr = "Bearer rdma-transfer-key-0123456789abcdef";

    var fabric = LoopbackFabric.init(a);
    defer fabric.deinit();

    var payload_buf: [3072]u8 = undefined;
    for (&payload_buf, 0..) |*b, i| b.* = @intCast((i * 31 + 7) & 0xff);
    const payload: []const u8 = &payload_buf;
    var src_cache: TestCacheState = .{ .allocator = a, .stored_session = Conn.sessionTag("sess-alpha"), .payload = payload };
    defer src_cache.deinit();
    var dst_cache: TestCacheState = .{ .allocator = a };
    defer dst_cache.deinit();

    var src_mgr: Manager = .{ .transport = fabric.transport(), .cache_hooks = src_cache.hooks(), .clock_fn = fakeClock };
    defer src_mgr.deinitAll(testing.io);
    var dst_mgr: Manager = .{ .transport = fabric.transport(), .cache_hooks = dst_cache.hooks(), .clock_fn = fakeClock };
    defer dst_mgr.deinitAll(testing.io);

    // 1. Session miss -> 404 with reason="cache_miss"
    const miss = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/offer", auth_hdr, "{\"session_key\":\"unknown-sess\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"expires_ms\":35000}");
    defer miss.deinit(a);
    try testing.expectEqual(@as(u16, 404), miss.status);
    try testing.expect(std.mem.indexOf(u8, miss.body, "\"reason\":\"cache_miss\"") != null);

    // 2. Step 1 Offer (cold QP -> qp_mode="accept"), forwarding optional model selector
    const offer1 = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/offer", auth_hdr, "{\"session_key\":\"sess-alpha\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"model\":\"demo-model\",\"expires_ms\":35000}");
    defer offer1.deinit(a);
    try testing.expectEqual(@as(u16, 200), offer1.status);
    try testing.expectEqual(@as(usize, 1), src_mgr.activeOfferCount(testing.io));

    // Overlapping offer on the same (peer_id, rdma_dev) while offer1 is pinned -> 503 rdma_busy.
    const overlap = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/offer", auth_hdr, "{\"session_key\":\"sess-alpha\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"expires_ms\":35000}");
    defer overlap.deinit(a);
    try testing.expectEqual(@as(u16, 503), overlap.status);
    try testing.expect(std.mem.indexOf(u8, overlap.body, "\"reason\":\"rdma_busy\"") != null);

    const offer1_json = try std.json.parseFromSlice(std.json.Value, a, offer1.body, .{});
    defer offer1_json.deinit();
    const o1 = offer1_json.value.object;
    try testing.expectEqualStrings("accept", o1.get("qp_mode").?.string);
    try testing.expectEqualStrings(build_options.version, o1.get("server_version").?.string);
    const tid1 = o1.get("transfer_id").?.string;
    const src_gid = o1.get("gid").?.string;
    const src_qpn = o1.get("qpn").?.integer;
    const src_psn = o1.get("psn").?.integer;

    // 3. Out-of-band QP accept: dst (with reset=true) then src (with transfer_id)
    const dst_acc_body = try std.fmt.allocPrint(a, "{{\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en3\",\"peer_gid\":\"{s}\",\"peer_qpn\":{d},\"peer_psn\":{d},\"reset\":true}}", .{ src_gid, src_qpn, src_psn });
    defer a.free(dst_acc_body);
    const dst_acc = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, dst_acc_body);
    defer dst_acc.deinit(a);
    try testing.expectEqual(@as(u16, 200), dst_acc.status);
    const dst_acc_json = try std.json.parseFromSlice(std.json.Value, a, dst_acc.body, .{});
    defer dst_acc_json.deinit();
    const dst_gid = dst_acc_json.value.object.get("gid").?.string;
    const dst_qpn = dst_acc_json.value.object.get("qpn").?.integer;
    const dst_psn = dst_acc_json.value.object.get("psn").?.integer;

    const src_acc_body = try std.fmt.allocPrint(a, "{{\"transfer_id\":\"{s}\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"peer_gid\":\"{s}\",\"peer_qpn\":{d},\"peer_psn\":{d}}}", .{ tid1, dst_gid, dst_qpn, dst_psn });
    defer a.free(src_acc_body);
    const src_acc = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, src_acc_body);
    defer src_acc.deinit(a);
    try testing.expectEqual(@as(u16, 200), src_acc.status);

    // 4. Step 2 Pull on dst -> transfers bytes over two-sided UC RDMA and imports into dst_cache.
    const pull1_body = try std.fmt.allocPrint(a, "{{\"transfer_id\":\"{s}\",\"session_key\":\"sess-alpha\",\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en3\",\"model\":\"demo-model\",\"deadline_ms\":10000,\"offer\":{s}}}", .{ tid1, offer1.body });
    defer a.free(pull1_body);
    const pull1 = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/pull", auth_hdr, pull1_body);
    defer pull1.deinit(a);
    try testing.expectEqual(@as(u16, 200), pull1.status);
    try testing.expect(std.mem.indexOf(u8, pull1.body, "\"status\":\"imported\"") != null);
    try testing.expectEqual(@as(usize, 1), fabric.reads_completed);
    try testing.expectEqual(@as(u64, payload.len), fabric.bytes_transferred);
    try testing.expectEqualStrings(payload, dst_cache.imported.?);

    // 5. Replay same transfer_id on dst -> 409 duplicate (zero additional RDMA transfers).
    const replay = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/pull", auth_hdr, pull1_body);
    defer replay.deinit(a);
    try testing.expectEqual(@as(u16, 409), replay.status);
    try testing.expect(std.mem.indexOf(u8, replay.body, "\"reason\":\"duplicate\"") != null);
    try testing.expectEqual(@as(usize, 1), fabric.reads_completed);

    // 6. Step 3 DELETE ?clean=1 -> immediately joins responder & unpins offer
    const del1_path = try std.fmt.allocPrint(a, "/v1/cache/rdma/offer/{s}?clean=1", .{tid1});
    defer a.free(del1_path);
    const del1 = try invokeHttp(&src_mgr, a, "DELETE", del1_path, auth_hdr, "");
    defer del1.deinit(a);
    try testing.expectEqual(@as(u16, 200), del1.status);
    try testing.expect(std.mem.indexOf(u8, del1.body, "\"status\":\"released\"") != null);
    try testing.expectEqual(@as(usize, 0), src_mgr.activeOfferCount(testing.io));

    // 7. Second offer reuses persistent RTS QP directly (qp_mode="persistent", no accept needed).
    const offer2 = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/offer", auth_hdr, "{\"session_key\":\"sess-alpha\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"expires_ms\":5000}");
    defer offer2.deinit(a);
    try testing.expectEqual(@as(u16, 200), offer2.status);
    const offer2_json = try std.json.parseFromSlice(std.json.Value, a, offer2.body, .{});
    defer offer2_json.deinit();
    try testing.expectEqualStrings("persistent", offer2_json.value.object.get("qp_mode").?.string);
    const tid2 = offer2_json.value.object.get("transfer_id").?.string;

    // Pull second offer directly without any accept.
    const pull2_body = try std.fmt.allocPrint(a, "{{\"transfer_id\":\"{s}\",\"session_key\":\"sess-alpha\",\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en3\",\"deadline_ms\":10000,\"offer\":{s}}}", .{ tid2, offer2.body });
    defer a.free(pull2_body);
    const pull2 = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/pull", auth_hdr, pull2_body);
    defer pull2.deinit(a);
    try testing.expectEqual(@as(u16, 200), pull2.status);
    try testing.expectEqual(@as(usize, 2), fabric.reads_completed);

    // 8. Unclean DELETE ?clean=0 quarantines offer until expires_ms elapses.
    const del2_path = try std.fmt.allocPrint(a, "/v1/cache/rdma/offer/{s}?clean=0", .{tid2});
    defer a.free(del2_path);
    const del2 = try invokeHttp(&src_mgr, a, "DELETE", del2_path, auth_hdr, "");
    defer del2.deinit(a);
    try testing.expectEqual(@as(u16, 200), del2.status);
    try testing.expect(std.mem.indexOf(u8, del2.body, "\"status\":\"quarantined\"") != null);
    try testing.expectEqual(@as(usize, 1), src_mgr.activeOfferCount(testing.io));

    // Still pinned before expires_ms; swept once clock advances past expires_ms.
    test_clock_ms += 4_999;
    try testing.expectEqual(@as(usize, 0), src_mgr.sweepExpiredNonBlocking(testing.io));
    try testing.expectEqual(@as(usize, 1), src_mgr.activeOfferCount(testing.io));
    test_clock_ms += 2;
    try testing.expectEqual(@as(usize, 1), src_mgr.sweepExpiredNonBlocking(testing.io));
    try testing.expectEqual(@as(usize, 0), src_mgr.activeOfferCount(testing.io));

    // 9. Quiesce tears down all persistent QPs cleanly.
    try testing.expectEqual(@as(usize, 1), src_mgr.activeQpCount(testing.io));
    const q_resp = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/quiesce", auth_hdr, "{}");
    defer q_resp.deinit(a);
    try testing.expectEqual(@as(u16, 200), q_resp.status);
    try testing.expectEqual(@as(usize, 0), src_mgr.activeQpCount(testing.io));
}

test "cache_rdma: qp_reset self-heal re-arms pinned offer after dst restart" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    const a = testing.allocator;
    test_clock_ms = 60_000;

    const old_ct = setupTransferKey("rdma-transfer-key-0123456789abcdef");
    defer http_server.g_cache_transfer = old_ct;
    const auth_hdr = "Bearer rdma-transfer-key-0123456789abcdef";

    var fabric = LoopbackFabric.init(a);
    defer fabric.deinit();

    const payload = "self-heal-kv-envelope-payload-0123456789abcdef";
    var src_cache: TestCacheState = .{ .allocator = a, .stored_session = Conn.sessionTag("sess-heal"), .payload = payload };
    defer src_cache.deinit();
    var dst_cache: TestCacheState = .{ .allocator = a };
    defer dst_cache.deinit();

    var src_mgr: Manager = .{ .transport = fabric.transport(), .cache_hooks = src_cache.hooks(), .clock_fn = fakeClock };
    defer src_mgr.deinitAll(testing.io);
    var dst_mgr: Manager = .{ .transport = fabric.transport(), .cache_hooks = dst_cache.hooks(), .clock_fn = fakeClock };
    defer dst_mgr.deinitAll(testing.io);

    // Cold offer + accept + delete to establish persistent QP pair.
    const o1 = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/offer", auth_hdr, "{\"session_key\":\"sess-heal\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"expires_ms\":35000}");
    defer o1.deinit(a);
    const o1_j = try std.json.parseFromSlice(std.json.Value, a, o1.body, .{});
    defer o1_j.deinit();
    const tid1 = o1_j.value.object.get("transfer_id").?.string;
    const da1_body = try std.fmt.allocPrint(a, "{{\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"peer_gid\":\"{s}\",\"peer_qpn\":{d},\"peer_psn\":{d},\"reset\":true}}", .{
        o1_j.value.object.get("gid").?.string, o1_j.value.object.get("qpn").?.integer, o1_j.value.object.get("psn").?.integer,
    });
    defer a.free(da1_body);
    const da1 = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, da1_body);
    defer da1.deinit(a);
    const da1_j = try std.json.parseFromSlice(std.json.Value, a, da1.body, .{});
    defer da1_j.deinit();
    const sa1_body = try std.fmt.allocPrint(a, "{{\"transfer_id\":\"{s}\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"peer_gid\":\"{s}\",\"peer_qpn\":{d},\"peer_psn\":{d}}}", .{
        tid1, da1_j.value.object.get("gid").?.string, da1_j.value.object.get("qpn").?.integer, da1_j.value.object.get("psn").?.integer,
    });
    defer a.free(sa1_body);
    const sa1 = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, sa1_body);
    defer sa1.deinit(a);
    const del1_p = try std.fmt.allocPrint(a, "/v1/cache/rdma/offer/{s}?clean=1", .{tid1});
    defer a.free(del1_p);
    const del1 = try invokeHttp(&src_mgr, a, "DELETE", del1_p, auth_hdr, "");
    defer del1.deinit(a);

    // Simulate dst restart (quiesce dst so its recv QP is gone while src still thinks QP is RTS).
    const dq = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/quiesce", auth_hdr, "{}");
    defer dq.deinit(a);

    // src offers on persistent QP -> arms responder on stale QP.
    const o2 = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/offer", auth_hdr, "{\"session_key\":\"sess-heal\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"expires_ms\":35000}");
    defer o2.deinit(a);
    const o2_j = try std.json.parseFromSlice(std.json.Value, a, o2.body, .{});
    defer o2_j.deinit();
    try testing.expectEqualStrings("persistent", o2_j.value.object.get("qp_mode").?.string);
    const tid2 = o2_j.value.object.get("transfer_id").?.string;

    // dst pull -> 409 qp_reset.
    const p2_body = try std.fmt.allocPrint(a, "{{\"transfer_id\":\"{s}\",\"session_key\":\"sess-heal\",\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"deadline_ms\":10000,\"offer\":{s}}}", .{ tid2, o2.body });
    defer a.free(p2_body);
    const p2_fail = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/pull", auth_hdr, p2_body);
    defer p2_fail.deinit(a);
    try testing.expectEqual(@as(u16, 409), p2_fail.status);
    try testing.expect(std.mem.indexOf(u8, p2_fail.body, "\"reason\":\"qp_reset\"") != null);

    // Self-heal: src:/accept(reset=true) -> dst:/accept(reset=true) -> src:/accept(transfer_id) -> retry pull.
    const sr = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, "{\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"reset\":true}");
    defer sr.deinit(a);
    try testing.expectEqual(@as(u16, 200), sr.status);
    const sr_j = try std.json.parseFromSlice(std.json.Value, a, sr.body, .{});
    defer sr_j.deinit();
    const new_src_gid = sr_j.value.object.get("gid").?.string;
    const new_src_qpn = sr_j.value.object.get("qpn").?.integer;
    const new_src_psn = sr_j.value.object.get("psn").?.integer;

    const da2_body = try std.fmt.allocPrint(a, "{{\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"peer_gid\":\"{s}\",\"peer_qpn\":{d},\"peer_psn\":{d},\"reset\":true}}", .{ new_src_gid, new_src_qpn, new_src_psn });
    defer a.free(da2_body);
    const da2 = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, da2_body);
    defer da2.deinit(a);
    try testing.expectEqual(@as(u16, 200), da2.status);
    const da2_j = try std.json.parseFromSlice(std.json.Value, a, da2.body, .{});
    defer da2_j.deinit();

    const sa2_body = try std.fmt.allocPrint(a, "{{\"transfer_id\":\"{s}\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\",\"peer_gid\":\"{s}\",\"peer_qpn\":{d},\"peer_psn\":{d}}}", .{
        tid2, da2_j.value.object.get("gid").?.string, da2_j.value.object.get("qpn").?.integer, da2_j.value.object.get("psn").?.integer,
    });
    defer a.free(sa2_body);
    const sa2 = try invokeHttp(&src_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, sa2_body);
    defer sa2.deinit(a);
    try testing.expectEqual(@as(u16, 200), sa2.status);

    // Retry pull with updated offer QPN + PSN -> succeeds on the SAME still-pinned offer.
    const p2_retry_body = try std.fmt.allocPrint(
        a,
        "{{\"transfer_id\":\"{s}\",\"session_key\":\"sess-heal\",\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"deadline_ms\":10000,\"offer\":{{\"transfer_id\":\"{s}\",\"server_version\":{f},\"byte_len\":{d},\"import_peak\":{d},\"rdma_dev\":\"rdma_en2\",\"gid\":\"{s}\",\"qpn\":{d},\"psn\":{d},\"regions\":[{{\"addr\":0,\"rkey\":0,\"len\":{d}}}],\"expires_ms\":35000}}}}",
        .{ tid2, tid2, std.json.fmt(build_options.version, .{}), payload.len, payload.len * 2, new_src_gid, new_src_qpn, new_src_psn, payload.len },
    );
    defer a.free(p2_retry_body);
    const p2_ok = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/pull", auth_hdr, p2_retry_body);
    defer p2_ok.deinit(a);
    try testing.expectEqual(@as(u16, 200), p2_ok.status);
    try testing.expectEqualStrings(payload, dst_cache.imported.?);

    const del2_p = try std.fmt.allocPrint(a, "/v1/cache/rdma/offer/{s}?clean=1", .{tid2});
    defer a.free(del2_p);
    const del2 = try invokeHttp(&src_mgr, a, "DELETE", del2_p, auth_hdr, "");
    defer del2.deinit(a);
    try testing.expectEqual(@as(u16, 200), del2.status);
}

test "cache_rdma: pull refuses version_mismatch (409), import_no_room (400), malformed gid (400), and validates peer_lid & distributed cluster" {
    if (comptime builtin.os.tag == .windows) return error.SkipZigTest;
    const a = testing.allocator;

    const old_ct = setupTransferKey("rdma-transfer-key-0123456789abcdef");
    defer http_server.g_cache_transfer = old_ct;
    const auth_hdr = "Bearer rdma-transfer-key-0123456789abcdef";

    var fabric = LoopbackFabric.init(a);
    defer fabric.deinit();
    var dst_mgr: Manager = .{ .transport = fabric.transport() };
    defer dst_mgr.deinitAll(testing.io);

    // Version mismatch -> 409 version_mismatch
    const mixed_body = "{\"transfer_id\":\"tid-mixed\",\"session_key\":\"s\",\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"deadline_ms\":5000,\"offer\":{\"transfer_id\":\"tid-mixed\",\"server_version\":\"different-version\",\"byte_len\":1024,\"import_peak\":2048,\"rdma_dev\":\"rdma_en2\",\"regions\":[{\"addr\":4096,\"rkey\":0,\"len\":1024}],\"expires_ms\":5000}}";
    const r_mixed = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/pull", auth_hdr, mixed_body);
    defer r_mixed.deinit(a);
    try testing.expectEqual(@as(u16, 409), r_mixed.status);
    try testing.expect(std.mem.indexOf(u8, r_mixed.body, "\"reason\":\"version_mismatch\"") != null);
    try testing.expectEqual(@as(usize, 0), fabric.reads_completed);

    // Malformed offer.gid -> 400 bad_offer (never silently ignored)
    const bad_gid_body = try std.fmt.allocPrint(a, "{{\"transfer_id\":\"tid-badgid\",\"session_key\":\"s\",\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"deadline_ms\":5000,\"offer\":{{\"transfer_id\":\"tid-badgid\",\"server_version\":{f},\"byte_len\":1024,\"import_peak\":2048,\"rdma_dev\":\"rdma_en2\",\"gid\":\"not-hex\",\"regions\":[{{\"addr\":4096,\"rkey\":0,\"len\":1024}}],\"expires_ms\":5000}}}}", .{std.json.fmt(build_options.version, .{})});
    defer a.free(bad_gid_body);
    const r_badgid = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/pull", auth_hdr, bad_gid_body);
    defer r_badgid.deinit(a);
    try testing.expectEqual(@as(u16, 400), r_badgid.status);
    try testing.expect(std.mem.indexOf(u8, r_badgid.body, "\"reason\":\"bad_offer\"") != null);

    // peer_lid > 65535 -> 400 bad_request (no @intCast panic)
    const bad_lid = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, "{\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"peer_gid\":\"00000000000000000000ffffc0000201\",\"peer_qpn\":100,\"peer_lid\":65536}");
    defer bad_lid.deinit(a);
    try testing.expectEqual(@as(u16, 400), bad_lid.status);

    // Unescaped characters in rdma_dev -> 400 bad_request
    const bad_dev = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, "{\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma\\\"bad\",\"peer_gid\":\"00000000000000000000ffffc0000201\",\"peer_qpn\":100}");
    defer bad_dev.deinit(a);
    try testing.expectEqual(@as(u16, 400), bad_dev.status);

    // Pre-flight import_no_room -> 400 import_no_room with 0 RDMA bytes transferred
    const old_override = http_request.import_headroom_override;
    defer http_request.import_headroom_override = old_override;
    http_request.import_headroom_override = 1000;

    const no_room_body = try std.fmt.allocPrint(a, "{{\"transfer_id\":\"tid-noroom\",\"session_key\":\"s\",\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"deadline_ms\":5000,\"offer\":{{\"transfer_id\":\"tid-noroom\",\"server_version\":{f},\"byte_len\":1024,\"import_peak\":1,\"rdma_dev\":\"rdma_en2\",\"regions\":[{{\"addr\":0,\"rkey\":0,\"len\":1024}}],\"expires_ms\":5000}}}}", .{std.json.fmt(build_options.version, .{})});
    defer a.free(no_room_body);
    const r_noroom = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/pull", auth_hdr, no_room_body);
    defer r_noroom.deinit(a);
    try testing.expectEqual(@as(u16, 400), r_noroom.status);
    try testing.expect(std.mem.indexOf(u8, r_noroom.body, "\"reason\":\"import_no_room\"") != null);
    try testing.expectEqual(@as(usize, 0), fabric.reads_completed);

    // Malformed optional fields (peer_psn=-1, offer.qpn="bad") -> 400 instead of silent fallback.
    const bad_psn = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/accept", auth_hdr, "{\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"peer_gid\":\"00000000000000000000ffffc0000201\",\"peer_qpn\":100,\"peer_psn\":-1}");
    defer bad_psn.deinit(a);
    try testing.expectEqual(@as(u16, 400), bad_psn.status);

    const bad_qpn = try std.fmt.allocPrint(a, "{{\"transfer_id\":\"tid-badqpn\",\"session_key\":\"s\",\"peer_id\":\"peer-a\",\"rdma_dev\":\"rdma_en2\",\"deadline_ms\":5000,\"offer\":{{\"transfer_id\":\"tid-badqpn\",\"server_version\":{f},\"byte_len\":1024,\"import_peak\":2048,\"rdma_dev\":\"rdma_en2\",\"qpn\":\"bad\",\"regions\":[{{\"addr\":0,\"rkey\":0,\"len\":1024}}],\"expires_ms\":5000}}}}", .{std.json.fmt(build_options.version, .{})});
    defer a.free(bad_qpn);
    const r_badqpn = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/pull", auth_hdr, bad_qpn);
    defer r_badqpn.deinit(a);
    try testing.expectEqual(@as(u16, 400), r_badqpn.status);

    // Active distributed cluster -> 503 rdma_distributed_active on offer/accept/pull (info & quiesce still work).
    dst_mgr.test_distributed_active_override = true;
    defer dst_mgr.test_distributed_active_override = null;
    const r_dist = try invokeHttp(&dst_mgr, a, "POST", "/v1/cache/rdma/offer", auth_hdr, "{\"session_key\":\"s\",\"peer_id\":\"peer-b\",\"rdma_dev\":\"rdma_en2\"}");
    defer r_dist.deinit(a);
    try testing.expectEqual(@as(u16, 503), r_dist.status);
    try testing.expect(std.mem.indexOf(u8, r_dist.body, "\"reason\":\"rdma_distributed_active\"") != null);

    const r_info = try invokeHttp(&dst_mgr, a, "GET", "/v1/cache/rdma/info", auth_hdr, "");
    defer r_info.deinit(a);
    try testing.expectEqual(@as(u16, 200), r_info.status);
    try testing.expect(std.mem.indexOf(u8, r_info.body, "\"distributed_active\":true") != null);
}
