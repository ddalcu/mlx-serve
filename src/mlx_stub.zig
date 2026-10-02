//! mlx surface for builds without MLX.

const std = @import("std");

const real = @import("mlx_types.zig");

pub const mlx_array = real.mlx_array;
pub const mlx_stream = real.mlx_stream;
pub const mlx_device = real.mlx_device;
pub const mlx_string = real.mlx_string;
pub const mlx_vector_array = real.mlx_vector_array;
pub const mlx_device_info = real.mlx_device_info;
pub const mlx_dtype = real.mlx_dtype;
pub const mlx_device_type = real.mlx_device_type;

pub const Error = error{MlxUnavailable};

pub fn check(status: c_int) Error!void {
    if (status != 0) return Error.MlxUnavailable;
}

pub fn noGpuBackend() bool {
    return true;
}

pub fn gpuStream() mlx_stream {
    return .{};
}

pub fn mlx_metal_is_available(res: *bool) c_int {
    res.* = false;
    return 0;
}

pub fn mlx_get_active_memory(out: *usize) c_int {
    out.* = 0;
    return 0;
}
pub fn mlx_get_cache_memory(out: *usize) c_int {
    out.* = 0;
    return 0;
}
pub fn mlx_get_peak_memory(out: *usize) c_int {
    out.* = 0;
    return 0;
}
pub fn mlx_reset_peak_memory() c_int {
    return 0;
}
pub fn mlx_set_cache_limit(out: *usize, _: usize) c_int {
    out.* = 0;
    return 0;
}

pub const WiredMode = real.WiredMode;

pub const WiredPolicyResult = real.WiredPolicyResult;

pub fn applyWiredPolicy() WiredPolicyResult {
    return .{ .mode = .off, .target = null };
}

pub fn mlx_clear_cache() c_int {
    return 0;
}

pub fn mlx_random_seed(_: u64) c_int {
    return 0;
}

pub fn mlx_version(out: *mlx_string) c_int {
    out.* = .{};
    return 0;
}

pub var op_count: std.atomic.Value(u64) = .init(0);

pub fn mlx_string_new() mlx_string {
    return .{};
}
pub fn mlx_string_data(_: mlx_string) [*:0]const u8 {
    return "";
}
pub fn mlx_string_free(_: mlx_string) c_int {
    return 0;
}

pub fn mlx_get_default_device(out: *mlx_device) c_int {
    out.* = .{};
    return 0;
}
pub fn mlx_set_default_device(_: mlx_device) c_int {
    return 0;
}
pub fn mlx_device_new_type(_: mlx_device_type, _: c_int) mlx_device {
    return .{};
}
pub fn mlx_device_free(_: mlx_device) c_int {
    return 0;
}
pub fn mlx_device_info_new() mlx_device_info {
    return .{};
}
pub fn mlx_device_info_get(out: *mlx_device_info, _: mlx_device) c_int {
    out.* = .{};
    return 0;
}
pub fn mlx_device_info_get_size(out: *usize, _: mlx_device_info, _: [*:0]const u8) c_int {
    out.* = 0;
    return 0;
}
pub fn mlx_device_info_free(_: mlx_device_info) c_int {
    return 0;
}

pub fn mlx_array_new() mlx_array {
    return .{};
}
pub fn mlx_array_new_data(_: ?*const anyopaque, _: [*]const c_int, _: c_int, _: mlx_dtype) mlx_array {
    return .{};
}
pub fn mlx_array_free(_: mlx_array) c_int {
    return 0;
}
pub fn mlx_array_eval(_: mlx_array) c_int {
    return -1;
}
pub fn mlx_array_item_int32(out: *i32, _: mlx_array) c_int {
    out.* = 0;
    return -1;
}
pub fn mlx_concatenate_axis(out: *mlx_array, _: mlx_vector_array, _: c_int, _: mlx_stream) c_int {
    out.* = .{};
    return -1;
}
pub fn mlx_vector_array_new_data(_: [*]const mlx_array, _: usize) mlx_vector_array {
    return .{};
}
pub fn mlx_vector_array_free(_: mlx_vector_array) c_int {
    return 0;
}

pub fn getShape(_: mlx_array) []const c_int {
    return &.{};
}

pub fn installErrorHandler() void {}

pub fn maxRecommendedWorkingSet() usize {
    return 0;
}

pub fn peekErrorName() ?[]const u8 {
    return null;
}

pub fn checkErrorDecode() Error!void {}
