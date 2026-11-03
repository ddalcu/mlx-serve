//! qwen4_exp forward diagnostics: per-block GPU laps (`QWEN4_PROFILE_FWD`) and the QSA attention profile.

const std = @import("std");
const env = @import("../../env.zig");
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../../sdk/util/log.zig");
const gather_qmv_rows = @import("../../sdk/blocks/moe/gather_qmv_rows.zig");
const ProfClock = @import("../../engines/mlx/diag/profile.zig").ProfClock;

pub const Qwen4FwdProf = struct {
    const Block = enum(u8) { ple, hc_read, gdn, attn, hc_write, mlp };
    timing: bool,
    ops: bool,
    clock: ProfClock,
    ns: [6]u64 = @splat(0),
    layer_ns: u64 = 0,
    kind_ns: [3]u64 = @splat(0), // ple layer, gdn layers, attn layers
    op_mark: u64 = 0,
    op_ns: [6]u64 = @splat(0),

    pub fn init(seq_len: c_int, h: mlx.mlx_array) !Qwen4FwdProf {
        const raw = env.get(.QWEN4_PROFILE_FWD);
        // `=1`: S 2..16 (verify). `=all`: any S, including decode S=1 and prefill.
        const all = raw != null and raw.?[0] == 'a';
        const timing = profileFwdOn() and (all or (seq_len >= 2 and seq_len <= 16));
        var p: Qwen4FwdProf = .{ .timing = timing, .ops = env.on(.QWEN4_OPCOUNT), .clock = ProfClock.init() };
        if (p.ops) {
            gather_qmv_rows.moeStats().moe_rows_fused_layers = 0;
            gather_qmv_rows.moeStats().moe_sorted_layers = 0;
        }
        if (timing) {
            try mlx.check(mlx.mlx_array_eval(h));
            _ = p.clock.lap();
        }
        if (p.ops) p.op_mark = mlx.op_count.load(.monotonic);
        return p;
    }

    pub fn lap(self: *Qwen4FwdProf, arr: mlx.mlx_array, block: Block) !void {
        if (self.ops) {
            const now = mlx.op_count.load(.monotonic);
            self.op_ns[@backingInt(block)] += now - self.op_mark;
            self.op_mark = now;
        }
        if (!self.timing) return;
        try mlx.check(mlx.mlx_array_eval(arr));
        const ns = self.clock.lap();
        self.ns[@backingInt(block)] += ns;
        self.layer_ns += ns;
    }

    pub fn endLayer(self: *Qwen4FwdProf, kind: Block) void {
        const slot: usize = switch (kind) {
            .ple => 0,
            .gdn => 1,
            else => 2,
        };
        self.kind_ns[slot] += self.layer_ns;
        self.layer_ns = 0;
    }

    pub fn report(self: *const Qwen4FwdProf, seq_len: c_int, kv: usize, capture: bool, gdn_layers: u32, attn_layers: u32) void {
        if (self.ops) log.info("[qwen4-ops] S={d} ple {d} hcRead {d} gdn {d} attn {d} hcWrite {d} mlp {d} moeFused {d} moeSorted {d}\n", .{ seq_len, self.op_ns[0], self.op_ns[1], self.op_ns[2], self.op_ns[3], self.op_ns[4], self.op_ns[5], gather_qmv_rows.moeStats().moe_rows_fused_layers, gather_qmv_rows.moeStats().moe_sorted_layers });
        if (!self.timing) return;
        const ms = struct {
            fn f(n: u64) f64 {
                return @as(f64, @floatFromInt(n)) / 1e6;
            }
        }.f;
        log.info("[qwen4-prof] S={d} kv={d} capture={} gdn {d:.3} ms ({d} us, {d} layers, {d:.3}/layer) attn {d:.3} ms ({d} us, {d} layers, {d:.3}/layer) ple-layer {d:.3} ms ({d} us)\n", .{
            seq_len,
            kv,
            capture,
            ms(self.kind_ns[1]),
            self.kind_ns[1] / 1000,
            gdn_layers,
            ms(self.kind_ns[1]) / @as(f64, @floatFromInt(@max(gdn_layers, 1))),
            ms(self.kind_ns[2]),
            self.kind_ns[2] / 1000,
            attn_layers,
            ms(self.kind_ns[2]) / @as(f64, @floatFromInt(@max(attn_layers, 1))),
            ms(self.kind_ns[0]),
            self.kind_ns[0] / 1000,
        });
        log.info("[qwen4-prof] blocks (whole forward): ple {d:.3} ms / {d} us  hcRead {d:.3} ms / {d} us  gdn {d:.3} ms / {d} us  attn {d:.3} ms / {d} us  hcWrite {d:.3} ms / {d} us  mlp {d:.3} ms / {d} us\n", .{
            ms(self.ns[0]), self.ns[0] / 1000,
            ms(self.ns[1]), self.ns[1] / 1000,
            ms(self.ns[2]), self.ns[2] / 1000,
            ms(self.ns[3]), self.ns[3] / 1000,
            ms(self.ns[4]), self.ns[4] / 1000,
            ms(self.ns[5]), self.ns[5] / 1000,
        });
    }
};

/// QWEN4_PROFILE_ATTN=1: split every qwen4_exp full-attention layer of a prefill forward (S > 16)
/// into the QSA indexer, q/k/v projection + rope + KV append, the QSA kernel, and the gate + o_proj
/// tail, with a sync at each boundary. Summed over the forward's layers and logged once per
/// forward next to `[qwen4-prof]`. Diagnostic only, never on by default.
pub const Qwen4AttnProf = struct {
    const Stage = enum(u2) { indexer, proj, qsa, tail };
    var on: ?bool = null;
    pub var active = false;
    var clock: ProfClock = undefined;
    var ns: [4]u64 = @splat(0);
    var layers: u32 = 0;

    pub fn sync(arr: mlx.mlx_array) void {
        if (arr.ctx == null) return;
        mlx.check(mlx.mlx_array_eval(arr)) catch {};
    }

    /// Syncs `x` and starts the clock when profiling this layer; false otherwise.
    pub fn begin(x: mlx.mlx_array, seq_len: c_int) bool {
        if (on == null) on = env.on(.QWEN4_PROFILE_ATTN);
        if (!on.? or seq_len <= 16) return false;
        sync(x);
        clock = ProfClock.init();
        active = true;
        layers += 1;
        return true;
    }

    pub fn lap(arr: mlx.mlx_array, stage: Stage) void {
        sync(arr);
        ns[@backingInt(stage)] += clock.lap();
    }

    pub fn flush(seq_len: c_int, kv: usize) void {
        if (layers == 0) return;
        const ms = struct {
            fn f(n: u64) f64 {
                return @as(f64, @floatFromInt(n)) / 1e6;
            }
        }.f;
        log.info("[qwen4-attn] S={d} kv={d} layers={d} indexer {d:.3} ms  proj+rope+kv {d:.3} ms  qsa {d:.3} ms  gate+o_proj {d:.3} ms\n", .{ seq_len, kv, layers, ms(ns[0]), ms(ns[1]), ms(ns[2]), ms(ns[3]) });
        ns = @splat(0);
        layers = 0;
    }
};

/// Caches for the qwen4 per-layer diagnostic switches: a diag read per layer per forward scans `MLX_SERVE_DIAG`.
pub var qwen4_no_pooled_env: ?bool = null;
pub var qwen4_debug_scores_env: ?bool = null;
pub var qwen4_profile_fwd_env: ?bool = null;
pub var qwen4_profile_qsa_env: ?bool = null;

/// Diag `qwen4-profile-fwd` armed (`1`: verify widths, `all`: every width).
pub fn profileFwdOn() bool {
    if (qwen4_profile_fwd_env == null) qwen4_profile_fwd_env = env.onValue(env.get(.QWEN4_PROFILE_FWD));
    return qwen4_profile_fwd_env.?;
}
