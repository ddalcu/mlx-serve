//! ANE prefill-MLP offload (perf-plan-aug-17 P5, `--ane-prefill`).
//!
//! Splits each full-width prefill chunk's dense SwiGLU MLP rows between the
//! GPU (existing MLX chain) and the Apple Neural Engine (private framework
//! via lib/ane, int8 per-row weights, fp16 datapath). v1 scope: qwen3_5
//! family dense MLP only; opt-in, default OFF, LOSSY by design (the
//! decode-attn-quant precedent — quality is A/B'd per arch, bytes are not
//! expected to match).
//!
//! Threading contract: the inference thread stays the sole MLX caller — it
//! does the plane memcpys and mlx array creation; ONLY `msv_ane_mlp_eval`
//! runs on the dedicated ANE thread (one in-flight eval, kick/wait pair).
//!
//! Stage-A measured facts this file encodes (harness at
//! ~/claude-tmp/perf-aug17/p5-ane-mlp): full-MLP-in-one-program parity
//! cos 0.9999 vs fp32; 11.8 TFLOPS eval with the down conv K-chunked
//! (a single K=17408 conv is a 2.6x cliff — chunking lives in
//! lib/ane/ane_mlp.m); fp16 range safe to 16x with the always-on
//! (1/16 .. x16) down-conv wrap.

const std = @import("std");
const log = @import("log.zig");

// ── C ABI (lib/ane/ane_mlp.h) ──

pub const MsvAneMlp = opaque {};
pub const MsvAnePlane = opaque {};
extern fn msv_ane_available() c_int;
extern fn msv_ane_internal_free_disk() u64;
extern fn msv_ane_plane_create(bytes: usize) ?*MsvAnePlane;
extern fn msv_ane_plane_free(p: ?*MsvAnePlane) void;
extern fn msv_ane_plane_base(p: ?*MsvAnePlane) ?[*]f16;
extern fn msv_ane_mlp_create(
    name: [*:0]const u8,
    hidden: u32,
    ffn: u32,
    rows: u32,
    gate_q: [*]const i8,
    gate_s: [*]const f32,
    up_q: [*]const i8,
    up_s: [*]const f32,
    down_q: [*]const i8,
    down_s: [*]const f32,
    input_plane: ?*MsvAnePlane,
    output_plane: ?*MsvAnePlane,
    err: [*]u8,
    err_size: usize,
) ?*MsvAneMlp;
extern fn msv_ane_gdn_create(
    name: [*:0]const u8,
    hidden: u32,
    qkv_out: u32,
    z_out: u32,
    rows: u32,
    qkv_q: [*]const i8,
    qkv_s: [*]const f32,
    z_q: [*]const i8,
    z_s: [*]const f32,
    input_plane: ?*MsvAnePlane,
    output_plane: ?*MsvAnePlane,
    err: [*]u8,
    err_size: usize,
) ?*MsvAneMlp;
extern fn msv_ane_mlp_free(m: ?*MsvAneMlp) void;
extern fn msv_ane_mlp_input(m: ?*MsvAneMlp) ?[*]f16;
extern fn msv_ane_mlp_output(m: ?*MsvAneMlp) ?[*]f16;
extern fn msv_ane_mlp_eval(m: ?*MsvAneMlp, err: [*]u8, err_size: usize) c_int;
extern fn msv_ane_mlp_compile_seconds(m: ?*const MsvAneMlp) f64;
extern fn msv_ane_mlp_cache_hit(m: ?*const MsvAneMlp) c_int;

/// Whether the private AppleNeuralEngine framework is present and usable.
pub fn available() bool {
    return msv_ane_available() != 0;
}

/// Minimum useful ANE tile: below this the per-layer kick/join overhead
/// outweighs the offloaded work (the engagement floor, not a correctness
/// bound — Stage A measured flat TFLOPS down to 512 rows).
pub const ANE_MIN_ROWS: u32 = 256;

/// Rows the ANE takes out of a fixed `chunk_rows`-wide prefill chunk at
/// `share` (0..1): floored to a multiple of 32, clamped so the GPU keeps at
/// least 16 rows, 0 when the tile would be below ANE_MIN_ROWS or the share
/// is degenerate. The 32-row quantum is a measured plane-pitch contract,
/// not a preference: an fp16 plane's per-channel pitch is rows x 2 bytes,
/// and a pitch off the 64-byte grid compiles fine but fails EVERY eval
/// ("Program Inference error") — the v2 share-0.35 collapse was rows
/// 2864 = 16 mod 32 falling back to a full GPU recompute per chunk (A3
/// probe 2026-08-18: 2864/2896 dead, 2880/2912/3264/3680 all ~11.8 TFLOPS,
/// no rate cliff among legal tiles).
pub fn aneShareRows(chunk_rows: u32, share: f32) u32 {
    if (chunk_rows < ANE_MIN_ROWS + 16 or !(share > 0)) return 0;
    const raw: f32 = @as(f32, @floatFromInt(chunk_rows)) * share;
    if (!(raw > 0) or raw != raw) return 0;
    var rows: u32 = @intFromFloat(raw);
    rows -= rows % 32;
    if (rows > chunk_rows - 16) rows = (chunk_rows - 16) - (chunk_rows - 16) % 32;
    if (rows < ANE_MIN_ROWS) return 0;
    return rows;
}

/// Chip name via sysctl ("Apple M3 Ultra"); empty on failure — callers fall
/// to their default row. Canonical copy (dflash.zig delegates here); the GPU
/// arch string cannot tell Ultra from Max, hence the CPU brand.
pub fn chipBrandString(buf: []u8) []const u8 {
    var len: usize = buf.len;
    if (std.c.sysctlbyname("machdep.cpu.brand_string", buf.ptr, &len, null, 0) != 0) return "";
    if (len > 0 and buf[len - 1] == 0) len -= 1;
    return buf[0..len];
}

/// Per-(mode, silicon) default share. Every row is a MEASURED optimum of its
/// own sweep — never interpolated (a share change needs its own A/B):
///   channel / M4 Max: 0.45 (2026-08-18, Qwen3.8-27B oQ4e: 0.40/0.45/0.50
///     ranked 305/311/297 at 16k — rollover at 0.50).
///   channel / M3 Ultra: 0.35 (PR #223 tester, 2026-08-19, same pack 4-bit:
///     0.45 ≈ +1% at 32k — nothing; sweep 0.45/0.40/0.35/0.30 ranked
///     398/421/440/438 at 32k; clean post-reboot 0.35 A/B +0.2%/+8.5%/+13.7%
///     at 8k/16k/32k, reproduced 3x incl. across a reboot — and the smaller
///     share also drops the int8 copy 9.47 → 7.30 GB).
///   row / M4 Max: 0.40 (2026-08-17: 0.30 +10/+14%, 0.40 +12/+18%, 0.50
///     regresses). Row is unmeasured elsewhere and keeps the M4 row.
pub fn defaultShare(mode: Mode, chip: []const u8) f32 {
    if (mode == .row) return 0.40;
    if (std.mem.indexOf(u8, chip, "M3 Ultra") != null) return 0.35;
    return 0.45;
}

/// The ANE's share of each covered projection (channel mode: fraction of
/// output channels; row mode: fraction of chunk token rows).
/// MLX_SERVE_ANE_SPLIT overrides; the default is per (mode, silicon) —
/// see `defaultShare`.
pub fn splitShare() f32 {
    var chip_buf: [128]u8 = undefined;
    const def = defaultShare(splitMode(), chipBrandString(&chip_buf));
    const raw = std.c.getenv("MLX_SERVE_ANE_SPLIT") orelse return def;
    const v = std.fmt.parseFloat(f32, std.mem.sliceTo(raw, 0)) catch return def;
    if (!(v > 0) or v > 1) return def;
    return v;
}

/// ANE prefill is for M4-and-below: on NAX-class GPUs (M5+) the GPU prefill
/// already outruns the seam — measured a LOSS on M5 Max (channel 0.45 median
/// -11%/-7.5% at 16k/32k, PR #223, two testers). MLX_SERVE_ANE_FORCE=1 keeps
/// the build for future silicon measurement (M6 etc.). Pure so it is
/// hermetically testable; the scheduler passes the live NAX probe + env.
pub fn anePrefillAllowed(nax_available: bool, force_env: ?[]const u8) bool {
    if (!nax_available) return true;
    if (force_env) |v| return v.len > 0 and v[0] == '1';
    return false;
}

/// GDN input-projection offload beside the MLP one (v2). MLX_SERVE_ANE_GDN=0
/// keeps `--ane-prefill` MLP-only — the attribution lever for A/Bs.
pub fn gdnEnabled() bool {
    const raw = std.c.getenv("MLX_SERVE_ANE_GDN") orelse return true;
    return !std.mem.eql(u8, std.mem.sliceTo(raw, 0), "0");
}

/// How each projection splits between ANE and GPU (A1, ane-plan-aug-18).
/// `row`: the ANE takes the first `aneShareRows` token rows through the
/// FULL weights (v1/v2 shipped design; int8 copy = 100% of covered
/// weights). `channel`: both units see ALL chunk tokens through sliced
/// weights — the ANE holds output channels [0..k) of gate/up (and qkv/z),
/// the GPU the rest; the down projection contributes PARTIAL sums added at
/// the seam — so the resident int8 copy scales with the share. Channel is
/// the DEFAULT since the 2026-08-18 counterbalanced A/B (27B oQ4e, M4 Max,
/// medians): channel-0.45 306.4/301.1 vs row-0.40 300.7/294.3 vs off
/// 258.2/239.0 at 16k/32k — +18.7%/+26.0% over off at 9.3 GB ANE bytes
/// against row's 20.4. MLX_SERVE_ANE_MODE=row restores the row split.
pub const Mode = enum { row, channel };

pub fn splitMode() Mode {
    const raw = std.c.getenv("MLX_SERVE_ANE_MODE") orelse return .channel;
    if (std.mem.eql(u8, std.mem.sliceTo(raw, 0), "row")) return .row;
    return .channel;
}

/// Channel-slice boundary alignment: a multiple of 128 keeps the GPU-side
/// complement slices legal for every quant geometry we serve (group sizes
/// 32/64/128; packed-word boundaries at any bits in {2,3,4,5,6,8}) and the
/// ANE-side weights nicely tiled.
pub const CHANNEL_ALIGN: u32 = 128;

/// The ANE's slice of a projection's output channels at `share`: floored to
/// CHANNEL_ALIGN, clamped so the GPU keeps at least CHANNEL_ALIGN channels,
/// 0 when the slice degenerates. The down conv's K-chunkability is
/// guaranteed by construction for 128-aligned widths (a power-of-two
/// divisor <= 16 always lands under the K cliff), mirrored here as an
/// assert rather than a walk.
pub fn channelSliceWidth(width: u32, share: f32) u32 {
    if (width < 2 * CHANNEL_ALIGN or !(share > 0)) return 0;
    const raw: f32 = @as(f32, @floatFromInt(width)) * share;
    if (!(raw > 0) or raw != raw) return 0;
    var k: u32 = @intFromFloat(raw);
    k -= k % CHANNEL_ALIGN;
    if (k > width - CHANNEL_ALIGN) k = width - CHANNEL_ALIGN - (width - CHANNEL_ALIGN) % CHANNEL_ALIGN;
    if (k < CHANNEL_ALIGN) return 0;
    return k;
}

/// What the offload will actually cost in bytes, computable from the config
/// BEFORE any dequant: the int8 weight copies (dense MLP: gate/up/down; GDN:
/// the fused qkv+z stack) plus the SHARED fp16 IOSurface planes (A9: evals
/// are strictly serial, so one plane per shape class — input, MLP output,
/// GDN output — serves every program; ~11 GB back on the 27B vs the old
/// per-program pairs). Per-row fp16 scales are noise next to these (out_dim
/// × 2 bytes per weight) and deliberately not billed.
pub fn engineBillBytes(dense_layers: u64, gdn_layers: u64, hidden: u64, ffn: u64, qkv_out: u64, z_out: u64, rows: u64) u64 {
    const dense_int8 = dense_layers * 3 * hidden * ffn;
    const gdn_int8 = gdn_layers * (qkv_out + z_out) * hidden;
    var planes: u64 = 0;
    if (dense_layers > 0 or gdn_layers > 0) planes += hidden * rows * 2; // shared input
    if (dense_layers > 0) planes += hidden * rows * 2; // MLP output
    if (gdn_layers > 0) planes += (qkv_out + z_out) * rows * 2; // GDN output
    return dense_int8 + gdn_int8 + planes;
}

/// Headroom the gate reserves beyond resident weights + the ANE bill: KV
/// cache, the prefill chunk's transient envelope, and OS baseline. 12 GB
/// covers the 27B at the default chunk with margin; a refused load names
/// every number it compared.
pub const GATE_HEADROOM_BYTES: u64 = 12 * 1024 * 1024 * 1024;

/// The per-model admission gate (replaces the v1 flat 96 GB total-RAM
/// check, which refused a 1 GB bill on a 64 GB Mac and said nothing about
/// WHY): the bill is admitted when resident + bill + headroom fits total
/// RAM. An unknown total (sysctl failure) is no information — allow, the
/// server's other memory guards still stand.
pub fn gateAllows(total_mem: u64, resident: u64, bill: u64) bool {
    if (total_mem == 0) return true;
    return resident + bill + GATE_HEADROOM_BYTES <= total_mem;
}

/// The hard floor for starting an ANE build at all: below this much free
/// internal disk even cache RESTORES and the framework's own model saves
/// start failing bare ("Write weightsFilePath failed" in the unified log),
/// and the build ships silent partial coverage — the 2026-08-18 class. The
/// build is SKIPPED with a named refusal instead.
pub const BUILD_DISK_FLOOR_BYTES: u64 = 1 << 30;

/// Free bytes on the internal volume that bounds the ANE compile-session
/// budget (aned's per-compile scratch lives in root tmp for the client's
/// lifetime, wherever OUR staging is). 0 = probe failed, no information.
pub fn internalFreeDiskBytes() u64 {
    return msv_ane_internal_free_disk();
}

/// Live ANE totals across resident engines, for the `--metrics` gauges and
/// anything else that wants "what is the ANE holding right now" without
/// walking the registry. Published by `publishLive` after a successful
/// build, retired by deinit; zero whenever no engine is resident (the
/// zero-when-off metrics invariant).
pub var live_int8_bytes = std.atomic.Value(u64).init(0);
pub var live_layers = std.atomic.Value(u64).init(0);

pub const RowQuant = struct {
    q: []i8,
    s: []f32,

    pub fn deinit(self: *RowQuant, allocator: std.mem.Allocator) void {
        allocator.free(self.q);
        allocator.free(self.s);
    }
};

/// Per-output-row symmetric int8 quantization of a dense row-major [n, k]
/// f32 weight: s[i] = max|row i| / 127, q = round(w / s) clamped to ±127.
/// An all-zero row gets scale 0 and zero codes (dequantizes to zero).
pub fn quantizeRowsInt8(allocator: std.mem.Allocator, w: []const f32, n: usize, k: usize) !RowQuant {
    std.debug.assert(w.len == n * k);
    const q = try allocator.alloc(i8, n * k);
    errdefer allocator.free(q);
    const s = try allocator.alloc(f32, n);
    errdefer allocator.free(s);
    for (0..n) |i| {
        const row = w[i * k .. (i + 1) * k];
        var amax: f32 = 0;
        for (row) |v| {
            const a = @abs(v);
            if (a > amax) amax = a;
        }
        if (amax == 0) {
            s[i] = 0;
            @memset(q[i * k .. (i + 1) * k], 0);
            continue;
        }
        const scale = amax / 127.0;
        s[i] = scale;
        const inv = 1.0 / scale;
        for (row, 0..) |v, j| {
            const r = @round(v * inv);
            q[i * k + j] = @intFromFloat(std.math.clamp(r, -127.0, 127.0));
        }
    }
    return .{ .q = q, .s = s };
}

/// Per-model ANE prefill engine: one compiled MLP program per dense-MLP
/// layer at a FIXED row tile, plus the dedicated eval thread. Built at
/// model load (scheduler), owned by the Transformer, freed on unload.
pub const AnePrefill = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    /// Per-layer MLP handles; null = layer not offloaded (build failure or
    /// non-dense layer). Indexed by layer index.
    layers: []?*MsvAneMlp,
    /// Per-layer GDN input-projection handles (in_proj_qkv + in_proj_z as
    /// one fused program); null = not offloaded (full-attention layer,
    /// combined-proj arch, build failure, or MLX_SERVE_ANE_GDN=0).
    gdn_layers: []?*MsvAneMlp,
    hidden: u32,
    ffn: u32,
    /// GDN projection output widths the compiled programs are built at
    /// (0 = no GDN offload). The seam reads these to slice the output plane.
    gdn_qkv_out: u32 = 0,
    gdn_z_out: u32 = 0,
    /// Shared I/O planes (A9): evals are strictly serial, so every program
    /// of a shape class binds the same surface — one input (hidden x rows,
    /// MLP and GDN alike), one MLP output (hidden x rows), one GDN output
    /// ((qkv+z) x rows) — instead of a pair per program (~11 GB on the 27B).
    plane_in: ?*MsvAnePlane = null,
    plane_mlp_out: ?*MsvAnePlane = null,
    plane_gdn_out: ?*MsvAnePlane = null,
    /// The fixed ANE row tile every compiled program expects (== chunk_rows
    /// in channel mode: both units see all chunk tokens there).
    rows: u32,
    /// The full chunk width the tile was derived from — the forward seam
    /// only engages on chunks of exactly this width.
    chunk_rows: u32,
    /// row = token-row split through full weights; channel = output-channel
    /// split through sliced weights (partial-sum down join).
    mode: Mode = .row,
    /// The share the engine was built at and the int8 bytes it holds —
    /// set by the builder after the layer loop, read by /props and the
    /// metrics gauges. 0 until publishLive.
    share: f32 = 0,
    int8_bytes: u64 = 0,
    published: bool = false,

    thread: ?std.Thread = null,
    mu: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    /// Protected by mu: .idle → .requested(handle) → .done(ok).
    state: enum { idle, requested, done } = .idle,
    pending_handle: ?*MsvAneMlp = null,
    eval_ok: bool = true,
    stop_flag: bool = false,
    eval_err: [512]u8 = @splat(0),
    engaged_logged: bool = false,
    gdn_engaged_logged: bool = false,
    /// Lifetime eval counts, read by /props from the server thread (the M3
    /// Ultra tester could not verify DISPATCH from /props — the one-shot
    /// engagement lines live in the log, but a props probe is what a bench
    /// harness reads). Written on the ANE thread, hence atomics.
    evals_ok: std.atomic.Value(u64) = .init(0),
    evals_failed: std.atomic.Value(u64) = .init(0),

    /// `gdn_qkv_out`/`gdn_z_out` of 0 skips the GDN output plane (MLP-only
    /// engine); non-zero sizes it for the fused qkv+z programs. In channel
    /// mode `ffn`/`gdn_*_out` are the SLICED widths and rows == chunk_rows.
    pub fn init(allocator: std.mem.Allocator, io: std.Io, num_layers: usize, hidden: u32, ffn: u32, rows: u32, chunk_rows: u32, gdn_qkv_out: u32, gdn_z_out: u32, mode: Mode) !*AnePrefill {
        const self = try allocator.create(AnePrefill);
        errdefer allocator.destroy(self);
        const layers = try allocator.alloc(?*MsvAneMlp, num_layers);
        errdefer allocator.free(layers);
        @memset(layers, null);
        const gdn_layers = try allocator.alloc(?*MsvAneMlp, num_layers);
        errdefer allocator.free(gdn_layers);
        @memset(gdn_layers, null);
        const io_bytes = @as(usize, hidden) * rows * 2;
        const plane_in = msv_ane_plane_create(io_bytes) orelse return error.AnePlaneAlloc;
        errdefer msv_ane_plane_free(plane_in);
        const plane_mlp_out = msv_ane_plane_create(io_bytes) orelse return error.AnePlaneAlloc;
        errdefer msv_ane_plane_free(plane_mlp_out);
        var plane_gdn_out: ?*MsvAnePlane = null;
        errdefer msv_ane_plane_free(plane_gdn_out);
        if (gdn_qkv_out > 0 and gdn_z_out > 0) {
            plane_gdn_out = msv_ane_plane_create(@as(usize, gdn_qkv_out + gdn_z_out) * rows * 2) orelse return error.AnePlaneAlloc;
        }
        self.* = .{
            .allocator = allocator,
            .io = io,
            .layers = layers,
            .gdn_layers = gdn_layers,
            .hidden = hidden,
            .ffn = ffn,
            .gdn_qkv_out = if (plane_gdn_out != null) gdn_qkv_out else 0,
            .gdn_z_out = if (plane_gdn_out != null) gdn_z_out else 0,
            .plane_in = plane_in,
            .plane_mlp_out = plane_mlp_out,
            .plane_gdn_out = plane_gdn_out,
            .rows = rows,
            .chunk_rows = chunk_rows,
            .mode = mode,
        };
        self.thread = try std.Thread.spawn(.{}, evalLoop, .{self});
        return self;
    }

    /// Record this engine's built totals into the live gauges (once).
    pub fn publishLive(self: *AnePrefill, share: f32, int8_bytes: u64) void {
        std.debug.assert(!self.published);
        self.share = share;
        self.int8_bytes = int8_bytes;
        self.published = true;
        _ = live_int8_bytes.fetchAdd(int8_bytes, .monotonic);
        _ = live_layers.fetchAdd(@intCast(self.coveredLayers() + self.coveredGdnLayers()), .monotonic);
    }

    pub fn deinit(self: *AnePrefill) void {
        if (self.published) {
            _ = live_int8_bytes.fetchSub(self.int8_bytes, .monotonic);
            _ = live_layers.fetchSub(@intCast(self.coveredLayers() + self.coveredGdnLayers()), .monotonic);
        }
        if (self.thread) |t| {
            self.mu.lockUncancelable(self.io);
            self.stop_flag = true;
            self.cond.broadcast(self.io);
            self.mu.unlock(self.io);
            t.join();
        }
        for (self.layers) |h| msv_ane_mlp_free(h);
        self.allocator.free(self.layers);
        for (self.gdn_layers) |h| msv_ane_mlp_free(h);
        self.allocator.free(self.gdn_layers);
        msv_ane_plane_free(self.plane_in);
        msv_ane_plane_free(self.plane_mlp_out);
        msv_ane_plane_free(self.plane_gdn_out);
        const allocator = self.allocator;
        allocator.destroy(self);
    }

    /// Number of layers with a compiled ANE MLP program.
    pub fn coveredLayers(self: *const AnePrefill) usize {
        var n: usize = 0;
        for (self.layers) |h| {
            if (h != null) n += 1;
        }
        return n;
    }

    /// Number of layers with a compiled GDN input-projection program.
    pub fn coveredGdnLayers(self: *const AnePrefill) usize {
        var n: usize = 0;
        for (self.gdn_layers) |h| {
            if (h != null) n += 1;
        }
        return n;
    }

    pub fn inputPlane(self: *AnePrefill, layer: usize) ?[*]f16 {
        return msv_ane_mlp_input(self.layers[layer]);
    }

    pub fn outputPlane(self: *AnePrefill, layer: usize) ?[*]f16 {
        return msv_ane_mlp_output(self.layers[layer]);
    }

    /// Planes of an arbitrary handle (the GDN seam addresses handles
    /// directly; the MLP helpers above stay for the older seam).
    pub fn planeInput(handle: ?*MsvAneMlp) ?[*]f16 {
        return msv_ane_mlp_input(handle);
    }

    pub fn planeOutput(handle: ?*MsvAneMlp) ?[*]f16 {
        return msv_ane_mlp_output(handle);
    }

    /// Hand a program's filled input plane to the ANE thread. One in-flight
    /// eval; the caller must `wait()` before the next kick.
    pub fn kick(self: *AnePrefill, handle: *MsvAneMlp) void {
        self.mu.lockUncancelable(self.io);
        std.debug.assert(self.state == .idle);
        self.pending_handle = handle;
        self.state = .requested;
        self.cond.broadcast(self.io);
        self.mu.unlock(self.io);
    }

    /// Block until the in-flight eval finishes. Returns false when the eval
    /// failed (the caller falls back to the GPU for those rows this chunk).
    pub fn wait(self: *AnePrefill) bool {
        self.mu.lockUncancelable(self.io);
        while (self.state != .done) self.cond.waitUncancelable(self.io, &self.mu);
        const ok = self.eval_ok;
        self.state = .idle;
        self.mu.unlock(self.io);
        return ok;
    }

    fn evalLoop(self: *AnePrefill) void {
        while (true) {
            self.mu.lockUncancelable(self.io);
            while (self.state != .requested and !self.stop_flag)
                self.cond.waitUncancelable(self.io, &self.mu);
            if (self.stop_flag) {
                self.mu.unlock(self.io);
                return;
            }
            const handle = self.pending_handle;
            self.mu.unlock(self.io);

            const ok = msv_ane_mlp_eval(handle, &self.eval_err, self.eval_err.len) != 0;
            if (ok) {
                _ = self.evals_ok.fetchAdd(1, .monotonic);
            } else {
                _ = self.evals_failed.fetchAdd(1, .monotonic);
                log.warn("[ane] eval failed: {s}\n", .{std.mem.sliceTo(&self.eval_err, 0)});
            }

            self.mu.lockUncancelable(self.io);
            self.eval_ok = ok;
            self.state = .done;
            self.cond.broadcast(self.io);
            self.mu.unlock(self.io);
        }
    }

    /// Build and install one layer's compiled MLP program from row-major
    /// [ffn, hidden] gate/up and [hidden, ffn] down f32 weights.
    pub fn buildLayer(self: *AnePrefill, layer: usize, gate: []const f32, up: []const f32, down: []const f32) !void {
        var gq = try quantizeRowsInt8(self.allocator, gate, self.ffn, self.hidden);
        defer gq.deinit(self.allocator);
        var uq = try quantizeRowsInt8(self.allocator, up, self.ffn, self.hidden);
        defer uq.deinit(self.allocator);
        var dq = try quantizeRowsInt8(self.allocator, down, self.hidden, self.ffn);
        defer dq.deinit(self.allocator);
        var err: [512]u8 = @splat(0);
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrintSentinel(&name_buf, "mlp_l{d}_r{d}", .{ layer, self.rows }, 0) catch "mlp";
        const handle = msv_ane_mlp_create(
            name.ptr,
            self.hidden,
            self.ffn,
            self.rows,
            gq.q.ptr,
            gq.s.ptr,
            uq.q.ptr,
            uq.s.ptr,
            dq.q.ptr,
            dq.s.ptr,
            self.plane_in,
            self.plane_mlp_out,
            &err,
            err.len,
        ) orelse {
            log.warn("[ane] layer {d} program build failed: {s}\n", .{ layer, std.mem.sliceTo(&err, 0) });
            return error.AneBuildFailed;
        };
        self.layers[layer] = handle;
    }

    /// One-shot engagement lines, one PER SEAM — a built-but-never-
    /// dispatched program is exactly the dispatch-hole class, so each seam
    /// proves its own dispatch in the log (the expectNoSpec rule).
    pub fn logEngagedOnce(self: *AnePrefill) void {
        if (self.engaged_logged) return;
        self.engaged_logged = true;
        log.info("[ane] prefill offload engaged: mode={s} mlp={d} rows={d}/{d} (--ane-prefill; MLX_SERVE_ANE_SPLIT sets the share)\n", .{ @tagName(self.mode), self.coveredLayers(), self.rows, self.chunk_rows });
    }

    pub fn logGdnEngagedOnce(self: *AnePrefill) void {
        if (self.gdn_engaged_logged) return;
        self.gdn_engaged_logged = true;
        log.info("[ane] gdn offload engaged: mode={s} {d} layers, rows={d}/{d} (MLX_SERVE_ANE_GDN=0 restores MLP-only)\n", .{ @tagName(self.mode), self.coveredGdnLayers(), self.rows, self.chunk_rows });
    }

    /// Build and install one layer's GDN input-projection program from
    /// row-major [qkv_out, hidden] and [z_out, hidden] f32 weights. The
    /// widths must match what init sized the shared GDN output plane for.
    pub fn buildGdnLayer(self: *AnePrefill, layer: usize, qkv_out: u32, z_out: u32, qkv: []const f32, z: []const f32) !void {
        if (self.plane_gdn_out == null or self.gdn_qkv_out != qkv_out or self.gdn_z_out != z_out)
            return error.AneGdnPlaneMismatch;
        var qq = try quantizeRowsInt8(self.allocator, qkv, qkv_out, self.hidden);
        defer qq.deinit(self.allocator);
        var zq = try quantizeRowsInt8(self.allocator, z, z_out, self.hidden);
        defer zq.deinit(self.allocator);
        var err: [512]u8 = @splat(0);
        var name_buf: [64]u8 = undefined;
        const name = std.fmt.bufPrintSentinel(&name_buf, "gdn_l{d}_r{d}", .{ layer, self.rows }, 0) catch "gdn";
        const handle = msv_ane_gdn_create(
            name.ptr,
            self.hidden,
            qkv_out,
            z_out,
            self.rows,
            qq.q.ptr,
            qq.s.ptr,
            zq.q.ptr,
            zq.s.ptr,
            self.plane_in,
            self.plane_gdn_out,
            &err,
            err.len,
        ) orelse {
            log.warn("[ane] layer {d} gdn program build failed: {s}\n", .{ layer, std.mem.sliceTo(&err, 0) });
            return error.AneBuildFailed;
        };
        self.gdn_layers[layer] = handle;
    }
};

// ── Tests ──

const testing = std.testing;

test "defaultShare: per-silicon channel rows, row mode keeps its M4 optimum" {
    // M3 Ultra channel row (PR #223 tester): 0.45 measured ≈ nothing, 0.35
    // measured +8.5%/+13.7% at 16k/32k — the default must be the measured
    // optimum for the machine, never one machine's number everywhere.
    try std.testing.expectEqual(@as(f32, 0.35), defaultShare(.channel, "Apple M3 Ultra"));
    try std.testing.expectEqual(@as(f32, 0.45), defaultShare(.channel, "Apple M4 Max"));
    try std.testing.expectEqual(@as(f32, 0.45), defaultShare(.channel, "Apple M3 Max"));
    try std.testing.expectEqual(@as(f32, 0.45), defaultShare(.channel, ""));
    // Row mode is only measured on M4; every chip keeps that row.
    try std.testing.expectEqual(@as(f32, 0.40), defaultShare(.row, "Apple M3 Ultra"));
    try std.testing.expectEqual(@as(f32, 0.40), defaultShare(.row, "Apple M4 Max"));
}

test "anePrefillAllowed: NAX-class GPUs refuse the seam unless forced" {
    // No NAX (M1-M4): allowed regardless of the force env.
    try std.testing.expect(anePrefillAllowed(false, null));
    try std.testing.expect(anePrefillAllowed(false, "0"));
    // NAX present (M5+): refused — the GPU prefill measured faster (PR #223).
    try std.testing.expect(!anePrefillAllowed(true, null));
    try std.testing.expect(!anePrefillAllowed(true, "0"));
    try std.testing.expect(!anePrefillAllowed(true, ""));
    // MLX_SERVE_ANE_FORCE=1 keeps the build for future-silicon measurement.
    try std.testing.expect(anePrefillAllowed(true, "1"));
}

test "engineBillBytes: int8 weights + SHARED fp16 planes (A9)" {
    // Evals are strictly serial, so the planes are per SHAPE CLASS, not per
    // program: one input (hidden x rows), one MLP output (hidden x rows),
    // one GDN output ((qkv+z) x rows). Hand-computed small case: dense 2
    // layers (3 weights of [ffn=8, hidden=4] -> 192 int8) + one GDN layer
    // ((6+2)*4 = 32 int8); planes 128 (in) + 128 (mlp out) + 256 (gdn out).
    try testing.expectEqual(@as(u64, 736), engineBillBytes(2, 1, 4, 8, 6, 2, 16));
    // The 27B at full coverage, rows 3264: int8 ~19.7 GiB + ~166 MB planes
    // (the per-program planes billed ~10.3 GiB here before A9).
    try testing.expectEqual(
        @as(u64, 21_313_093_632),
        engineBillBytes(64, 48, 5120, 17408, 10240, 6144, 3264),
    );
    // No GDN coverage bills neither GDN int8 nor a GDN output plane.
    try testing.expectEqual(@as(u64, 448), engineBillBytes(2, 0, 4, 8, 6, 2, 16));
    // No dense coverage bills no MLP output plane.
    try testing.expectEqual(@as(u64, 32 + 128 + 256), engineBillBytes(0, 1, 4, 8, 6, 2, 16));
}

test "gateAllows: per-model bill vs total RAM, unknown total allows" {
    const gib = 1024 * 1024 * 1024;
    // Unknown total (sysctl failure) is no information — allow, the old
    // `total_mem > 0` behavior.
    try testing.expect(gateAllows(0, 16 * gib, 32 * gib));
    // 64 GB Mac, 16 GB resident, ~32 GB bill: 16+32+12 headroom = 60 <= 64.
    try testing.expect(gateAllows(64 * gib, 16 * gib, 32 * gib));
    // 36 GB Mac, same model: refused.
    try testing.expect(!gateAllows(36 * gib, 16 * gib, 32 * gib));
    // A small model on a small Mac passes (the flat 96 GB gate refused it).
    try testing.expect(gateAllows(16 * gib, 1 * gib, 1 * gib));
    // Exact fit allows.
    try testing.expect(gateAllows(60 * gib, 16 * gib, 32 * gib));
}

test "aneShareRows: 32-row floor, GPU remainder, engagement minimum" {
    // 32-row quantum: an fp16 plane's per-channel pitch is rows x 2 bytes,
    // and a pitch off the 64-byte grid fails EVERY eval of the compiled
    // program (the v2 share-0.35 collapse: rows 2864 = 16 mod 32).
    try testing.expectEqual(@as(u32, 2432), aneShareRows(8192, 0.30));
    try testing.expectEqual(@as(u32, 2848), aneShareRows(8192, 0.35)); // was 2864, the live cliff
    try testing.expectEqual(@as(u32, 3264), aneShareRows(8192, 0.40)); // default share unchanged
    try testing.expectEqual(@as(u32, 1216), aneShareRows(4096, 0.30));
    try testing.expectEqual(@as(u32, 0), aneShareRows(8192, 0.0)); // no share
    try testing.expectEqual(@as(u32, 0), aneShareRows(8192, -1.0));
    try testing.expectEqual(@as(u32, 0), aneShareRows(512, 0.30)); // 144 < ANE_MIN_ROWS
    try testing.expectEqual(@as(u32, 0), aneShareRows(64, 0.9)); // chunk too small
    // Oversized share clamps so the GPU keeps >= 16 rows.
    const clamped = aneShareRows(8192, 1.5);
    try testing.expect(clamped <= 8192 - 16 and clamped % 32 == 0 and clamped > 0);
}

test "channelSliceWidth: 128-aligned slice, GPU remainder, degenerate shares" {
    // The 27B geometries at the default share (the spike's probed widths).
    try testing.expectEqual(@as(u32, 6912), channelSliceWidth(17408, 0.40));
    try testing.expectEqual(@as(u32, 4096), channelSliceWidth(10240, 0.40));
    try testing.expectEqual(@as(u32, 2432), channelSliceWidth(6144, 0.40));
    // Degenerate: tiny width, zero/negative share, NaN-safe.
    try testing.expectEqual(@as(u32, 0), channelSliceWidth(128, 0.5));
    try testing.expectEqual(@as(u32, 0), channelSliceWidth(17408, 0.0));
    try testing.expectEqual(@as(u32, 0), channelSliceWidth(17408, -1.0));
    // Oversized share clamps so the GPU keeps >= 128 channels.
    const clamped = channelSliceWidth(17408, 1.5);
    try testing.expect(clamped <= 17408 - 128 and clamped % 128 == 0 and clamped > 0);
    // Every 128-aligned width has a power-of-two K-chunk divisor <= 16
    // landing under the ANE down-conv cliff (mirrors down_chunks_for).
    var k: u32 = 128;
    while (k <= 17408) : (k += 128) {
        var ok = false;
        var n: u32 = 1;
        while (n <= 16) : (n += 1) {
            if (k % n == 0 and k / n <= 4608) ok = true;
        }
        try testing.expect(ok);
    }
}

test "quantizeRowsInt8: round-trip, per-row scales, zero row" {
    const w = [_]f32{
        1.0,  -2.0, 0.5,  0.25, // row 0: amax 2
        0.0,  0.0,  0.0,  0.0, // row 1: all zero
        -0.1, 0.05, 0.02, 0.1, // row 2: amax 0.1
    };
    var rq = try quantizeRowsInt8(testing.allocator, &w, 3, 4);
    defer rq.deinit(testing.allocator);
    try testing.expectApproxEqAbs(@as(f32, 2.0 / 127.0), rq.s[0], 1e-7);
    try testing.expectEqual(@as(f32, 0), rq.s[1]);
    try testing.expectApproxEqAbs(@as(f32, 0.1 / 127.0), rq.s[2], 1e-7);
    // Extremes hit exactly ±127; zero row stays zero codes.
    try testing.expectEqual(@as(i8, -127), rq.q[1]);
    try testing.expectEqual(@as(i8, 0), rq.q[4]);
    try testing.expectEqual(@as(i8, 127), rq.q[11]);
    // Dequantized max error <= half a step per row.
    for (0..3) |i| {
        for (0..4) |j| {
            const deq = @as(f32, @floatFromInt(rq.q[i * 4 + j])) * rq.s[i];
            try testing.expect(@abs(deq - w[i * 4 + j]) <= rq.s[i] * 0.5 + 1e-9);
        }
    }
}
