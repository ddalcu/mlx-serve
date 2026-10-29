//! Refusing stand-ins for lib/expert_io's C ABI (the read pool, the event shim) on graphs built
//! without those sources (every target but macOS: Linux, iOS): shared code type-checks there; the pool and the
//! event gate refuse at start.

const mlx = @import("sdk").mlx;
const io = @import("io.zig");

/// `q3ld_*`: `q3ld_abi` answers 0, so `expert_io`'s start refuses before any read.
pub const q3ld = struct {
    pub fn q3ld_spec_config(_: i32, _: ?[*]const u64, _: i32, _: i64, _: i64, _: i64, _: ?[*]i64) c_int {
        return -1;
    }
    pub fn q3ld_spec_streams(_: i32) c_int {
        return -1;
    }
    pub fn q3ld_start(_: i32, _: [*]const u64, _: i64, _: i64, _: [*]i64, _: i64, _: [*]i64, _: i64, _: *[6]i64) c_int {
        return -1;
    }
    pub fn q3ld_submit(_: i32, _: i64, _: i64, _: i32, _: i32, _: i32, _: [*]const i64, _: [*]const [*]const u64, _: [*]const i64, _: i64) c_int {
        return -1;
    }
    pub fn q3ld_submit_scatter(_: i32, _: i32, _: i32, _: [*]const i32, _: [*]const i64, _: [*]const i64, _: [*]const [*]const u64, _: [*]const i64, _: i64) c_int {
        return -1;
    }
    pub fn q3ld_withdraw(_: i64) c_int {
        return -1;
    }
    pub fn q3ld_spec_step_len(_: i32, _: i64, _: i64, _: i32, _: ?[*]const i64, _: i64) i32 {
        return -1;
    }
    pub fn q3ld_spec_step_lens(_: i32, _: i64, _: i64, _: i32, _: ?[*]const i64, _: ?[*]const i64) i32 {
        return -1;
    }
    pub fn q3ld_spec_state(_: *[io.spec_state_w * io.max_spec]i64) i32 {
        return -1;
    }
    pub fn q3ld_seq() i64 {
        return 0;
    }
    pub fn q3ld_wait(_: i64, _: i64) i64 {
        return 0;
    }
    pub fn q3ld_gauge(_: *[6]i64) void {}
    pub fn q3ld_quiesce(_: i64) c_int {
        return 0;
    }
    pub fn q3ld_stop() c_int {
        return 0;
    }
    pub fn q3ld_pre_config(_: i32, _: i32, _: ?[*]const i64) c_int {
        return -1;
    }
    pub fn q3ld_pre_read_lens(_: i32, _: i64, _: i64, _: i32, _: [*]const i64, _: [*]const i64) i32 {
        return -1;
    }
    pub fn q3ld_pre_state(_: *[io.pre_state_w * io.max_pre]i64) i32 {
        return -1;
    }
    pub fn q3ld_ev_config(_: i32, _: u64, _: i64, _: u64) c_int {
        return -1;
    }
    pub fn q3ld_ev_gates(_: i32, _: [*]const u64, _: [*]const i32, _: [*]const i64) i32 {
        return -1;
    }
    pub fn q3ld_ev_release(_: u64) i32 {
        return -1;
    }
    pub fn q3ld_ev_state(_: *[10]i64) i32 {
        return -1;
    }
    pub fn q3ld_monotonic_ns() i64 {
        return 0;
    }
    pub fn q3ld_direct_config(_: i32) c_int {
        return -1;
    }
    pub fn q3ld_abi() i32 {
        return 0;
    }
    pub fn q3ld_counters_n() i32 {
        return 0;
    }
    pub fn q3ld_max_spec() i32 {
        return 0;
    }
    pub fn q3ld_max_pre() i32 {
        return 0;
    }
    pub fn q3ld_max_gates() i32 {
        return 0;
    }
    pub fn q3ld_max_gate_tickets() i32 {
        return 0;
    }
    pub fn q3ld_test_rules(_: i32, _: [*]const i64, _: [*]const i64, _: [*]const i64) void {}
    pub fn q3ld_test_delay(_: u64, _: i64) void {}
    pub fn q3ld_test_ev_log(_: ?[*]i64, _: i64) i64 {
        return 0;
    }
};

/// `dsv41ev_*`: every event creation refuses.
pub const ev = struct {
    pub fn dsv41ev_abi() i32 {
        return 0;
    }
    pub fn dsv41ev_create_metal(_: u64, _: *u64) i32 {
        return -1;
    }
    pub fn dsv41ev_create_host(_: *i64, _: i64) i32 {
        return -1;
    }
    pub fn dsv41ev_wait(_: [*]const mlx.mlx_array, _: usize, _: i32, _: u64, _: ?[*]const mlx.mlx_array, _: usize, _: bool, _: mlx.mlx_stream, _: [*]mlx.mlx_array) c_int {
        return -1;
    }
    pub fn dsv41ev_signal(_: [*]const mlx.mlx_array, _: usize, _: i32, _: u64, _: mlx.mlx_stream, _: [*]mlx.mlx_array) c_int {
        return -1;
    }
    pub fn dsv41ev_value(_: i32) u64 {
        return 0;
    }
    pub fn dsv41ev_stats(_: *[8]i64) void {}
    pub fn dsv41ev_last_error() [*:0]const u8 {
        return "expert_io: not built for this target";
    }
};
