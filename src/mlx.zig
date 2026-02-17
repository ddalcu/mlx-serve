// Manual FFI bindings for mlx-c.
// We declare only the functions we need rather than @cImport to avoid
// potential issues with C++ headers and keep the dependency surface explicit.

const std = @import("std");

// ── Opaque handle types ──
pub const mlx_array = extern struct { ctx: ?*anyopaque = null };
pub const mlx_stream = extern struct { ctx: ?*anyopaque = null };
pub const mlx_device = extern struct { ctx: ?*anyopaque = null };
pub const mlx_string = extern struct { ctx: ?*anyopaque = null };
pub const mlx_map_string_to_array = extern struct { ctx: ?*anyopaque = null };
pub const mlx_map_string_to_string = extern struct { ctx: ?*anyopaque = null };
pub const mlx_map_string_to_array_iterator = extern struct { ctx: ?*anyopaque = null, map_ctx: ?*anyopaque = null };
pub const mlx_vector_array = extern struct { ctx: ?*anyopaque = null };

// ── Enums ──
pub const mlx_dtype = enum(c_int) {
    bool_ = 0,
    uint8 = 1,
    uint16 = 2,
    uint32 = 3,
    uint64 = 4,
    int8 = 5,
    int16 = 6,
    int32 = 7,
    int64 = 8,
    float16 = 9,
    float32 = 10,
    float64 = 11,
    bfloat16 = 12,
    complex64 = 13,
};

pub const mlx_device_type = enum(c_int) { cpu = 0, gpu = 1 };

// ── Optional types ──
pub const mlx_optional_int = extern struct {
    value: c_int = 0,
    has_value: bool = false,

    pub fn none() mlx_optional_int {
        return .{ .value = 0, .has_value = false };
    }
    pub fn some(v: c_int) mlx_optional_int {
        return .{ .value = v, .has_value = true };
    }
};

pub const mlx_optional_float = extern struct {
    value: f32 = 0,
    has_value: bool = false,

    pub fn none() mlx_optional_float {
        return .{ .value = 0, .has_value = false };
    }
    pub fn some(v: f32) mlx_optional_float {
        return .{ .value = v, .has_value = true };
    }
};

// ── Extern declarations ──
pub extern "c" fn mlx_version(str: *mlx_string) c_int;

// String
pub extern "c" fn mlx_string_new() mlx_string;
pub extern "c" fn mlx_string_data(str: mlx_string) [*:0]const u8;
pub extern "c" fn mlx_string_free(str: mlx_string) c_int;

// Device
pub extern "c" fn mlx_device_new() mlx_device;
pub extern "c" fn mlx_device_new_type(dtype: mlx_device_type, index: c_int) mlx_device;
pub extern "c" fn mlx_device_free(dev: mlx_device) c_int;
pub extern "c" fn mlx_get_default_device(dev: *mlx_device) c_int;
pub extern "c" fn mlx_set_default_device(dev: mlx_device) c_int;

// Stream
pub extern "c" fn mlx_stream_new() mlx_stream;
pub extern "c" fn mlx_stream_free(s: mlx_stream) c_int;
pub extern "c" fn mlx_default_cpu_stream_new() mlx_stream;
pub extern "c" fn mlx_default_gpu_stream_new() mlx_stream;
pub extern "c" fn mlx_synchronize(s: mlx_stream) c_int;

// Metal
pub extern "c" fn mlx_metal_is_available(res: *bool) c_int;

// Array creation
pub extern "c" fn mlx_array_new() mlx_array;
pub extern "c" fn mlx_array_new_int(val: c_int) mlx_array;
pub extern "c" fn mlx_array_new_float(val: f32) mlx_array;
pub extern "c" fn mlx_array_new_data(data: ?*const anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype) mlx_array;
pub extern "c" fn mlx_array_free(arr: mlx_array) c_int;
pub extern "c" fn mlx_array_set(arr: *mlx_array, src: mlx_array) c_int;

// Array info
pub extern "c" fn mlx_array_tostring(str: *mlx_string, arr: mlx_array) c_int;
pub extern "c" fn mlx_array_ndim(arr: mlx_array) usize;
pub extern "c" fn mlx_array_shape(arr: mlx_array) [*]const c_int;
pub extern "c" fn mlx_array_size(arr: mlx_array) usize;
pub extern "c" fn mlx_array_dtype(arr: mlx_array) mlx_dtype;
pub extern "c" fn mlx_array_eval(arr: mlx_array) c_int;
pub extern "c" fn mlx_array_itemsize(arr: mlx_array) usize;

// Scalar access
pub extern "c" fn mlx_array_item_float32(res: *f32, arr: mlx_array) c_int;
pub extern "c" fn mlx_array_item_int32(res: *i32, arr: mlx_array) c_int;

// Data access
pub extern "c" fn mlx_array_data_float32(arr: mlx_array) ?[*]const f32;
pub extern "c" fn mlx_array_data_int32(arr: mlx_array) ?[*]const i32;
pub extern "c" fn mlx_array_data_uint32(arr: mlx_array) ?[*]const u32;

// Vector array
pub extern "c" fn mlx_vector_array_new() mlx_vector_array;
pub extern "c" fn mlx_vector_array_new_data(data: [*]const mlx_array, size: usize) mlx_vector_array;
pub extern "c" fn mlx_vector_array_free(vec: mlx_vector_array) c_int;
pub extern "c" fn mlx_vector_array_size(vec: mlx_vector_array) usize;
pub extern "c" fn mlx_vector_array_get(res: *mlx_array, vec: mlx_vector_array, idx: usize) c_int;
pub extern "c" fn mlx_vector_array_append_value(vec: mlx_vector_array, val: mlx_array) c_int;

// Map string -> array
pub extern "c" fn mlx_map_string_to_array_new() mlx_map_string_to_array;
pub extern "c" fn mlx_map_string_to_array_free(map: mlx_map_string_to_array) c_int;
pub extern "c" fn mlx_map_string_to_array_get(value: *mlx_array, map: mlx_map_string_to_array, key: [*:0]const u8) c_int;
pub extern "c" fn mlx_map_string_to_array_insert(map: mlx_map_string_to_array, key: [*:0]const u8, value: mlx_array) c_int;

// Map iterator
pub extern "c" fn mlx_map_string_to_array_iterator_new(map: mlx_map_string_to_array) mlx_map_string_to_array_iterator;
pub extern "c" fn mlx_map_string_to_array_iterator_free(it: mlx_map_string_to_array_iterator) c_int;
pub extern "c" fn mlx_map_string_to_array_iterator_next(key: *?[*:0]const u8, value: *mlx_array, it: mlx_map_string_to_array_iterator) c_int;

// Map string -> string
pub extern "c" fn mlx_map_string_to_string_new() mlx_map_string_to_string;
pub extern "c" fn mlx_map_string_to_string_free(map: mlx_map_string_to_string) c_int;

// IO
pub extern "c" fn mlx_load_safetensors(res_0: *mlx_map_string_to_array, res_1: *mlx_map_string_to_string, file: [*:0]const u8, s: mlx_stream) c_int;

// ── Ops ──
pub extern "c" fn mlx_add(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_subtract(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_multiply(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_divide(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_negative(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_matmul(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_square(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_sqrt(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_rsqrt(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_exp(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_log(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_abs(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_tanh(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_erf(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;

pub extern "c" fn mlx_reshape(res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) c_int;
pub extern "c" fn mlx_transpose(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_transpose_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int;
pub extern "c" fn mlx_expand_dims(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_squeeze(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;

pub extern "c" fn mlx_take(res: *mlx_array, a: mlx_array, indices: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_take_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) c_int;

pub extern "c" fn mlx_concatenate_axis(res: *mlx_array, arrays: mlx_vector_array, axis: c_int, s: mlx_stream) c_int;

pub extern "c" fn mlx_softmax_axis(res: *mlx_array, a: mlx_array, axis: c_int, precise: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_argmax_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int;

pub extern "c" fn mlx_mean_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int;

pub extern "c" fn mlx_astype(res: *mlx_array, a: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int;

pub extern "c" fn mlx_where(res: *mlx_array, condition: mlx_array, x: mlx_array, y: mlx_array, s: mlx_stream) c_int;

pub extern "c" fn mlx_arange(res: *mlx_array, start: f64, stop: f64, step: f64, dtype: mlx_dtype, s: mlx_stream) c_int;
pub extern "c" fn mlx_full(res: *mlx_array, shape: [*]const c_int, shape_num: usize, val: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int;
pub extern "c" fn mlx_zeros(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, s: mlx_stream) c_int;

pub extern "c" fn mlx_slice(res: *mlx_array, a: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int;

pub extern "c" fn mlx_triu(res: *mlx_array, x: mlx_array, k: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_tril(res: *mlx_array, x: mlx_array, k: c_int, s: mlx_stream) c_int;

pub extern "c" fn mlx_power(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_less(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_greater(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_greater_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_less_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;

// Quantized matmul
pub extern "c" fn mlx_quantized_matmul(res: *mlx_array, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, transpose_w: bool, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, s: mlx_stream) c_int;

// Dequantize (fallback)
pub extern "c" fn mlx_dequantize(res: *mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, dtype: mlx_optional_dtype, s: mlx_stream) c_int;

pub const mlx_optional_dtype = extern struct {
    value: mlx_dtype = .float32,
    has_value: bool = false,
};

// ── Fast ops ──
pub extern "c" fn mlx_fast_rms_norm(res: *mlx_array, x: mlx_array, weight: mlx_array, eps: f32, s: mlx_stream) c_int;
pub extern "c" fn mlx_fast_rope(res: *mlx_array, x: mlx_array, dims: c_int, traditional: bool, base: mlx_optional_float, scale: f32, offset: c_int, freqs: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_fast_scaled_dot_product_attention(res: *mlx_array, queries: mlx_array, keys: mlx_array, values: mlx_array, scale: f32, mask_mode: [*:0]const u8, mask_arr: mlx_array, sinks: mlx_array, s: mlx_stream) c_int;

// ── Random ──
pub extern "c" fn mlx_random_categorical(res: *mlx_array, logits: mlx_array, axis: c_int, key: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_random_key(res: *mlx_array, seed: u64) c_int;
pub extern "c" fn mlx_random_seed(seed: u64) c_int;

// ── Batch eval ──
pub extern "c" fn mlx_eval(outputs: mlx_vector_array) c_int;

// ── Memory management ──
pub extern "c" fn mlx_set_memory_limit(res: *usize, limit: usize) c_int;
pub extern "c" fn mlx_set_cache_limit(res: *usize, limit: usize) c_int;
pub extern "c" fn mlx_get_active_memory(res: *usize) c_int;
pub extern "c" fn mlx_get_peak_memory(res: *usize) c_int;
pub extern "c" fn mlx_reset_peak_memory() c_int;

// ── Error handler ──
pub const mlx_error_handler_func = ?*const fn ([*:0]const u8, ?*anyopaque) callconv(.c) void;
pub extern "c" fn mlx_set_error_handler(handler: mlx_error_handler_func, data: ?*anyopaque, dtor: ?*const fn (?*anyopaque) callconv(.c) void) void;

// ── Zig helper wrappers ──

/// Get GPU stream (default)
pub fn gpuStream() mlx_stream {
    return mlx_default_gpu_stream_new();
}

/// Print an array for debugging
pub fn printArray(label: []const u8, arr: mlx_array) void {
    _ = mlx_array_eval(arr);
    var str = mlx_string_new();
    _ = mlx_array_tostring(&str, arr);
    const data = mlx_string_data(str);
    std.debug.print("{s}: {s}\n", .{ label, data });
    _ = mlx_string_free(str);
}

/// Get array shape as a Zig slice
pub fn getShape(arr: mlx_array) []const c_int {
    const ndim = mlx_array_ndim(arr);
    if (ndim == 0) return &.{};
    return mlx_array_shape(arr)[0..ndim];
}

/// Check if an mlx-c call succeeded (returns 0 on success)
pub fn check(ret: c_int) !void {
    if (ret != 0) return error.MlxError;
}
