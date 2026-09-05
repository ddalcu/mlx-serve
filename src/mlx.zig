// Manual FFI bindings for mlx-c.
// We declare only the functions we need rather than @cImport to avoid
// potential issues with C++ headers and keep the dependency surface explicit.

const std = @import("std");
const builtin = @import("builtin");
const log = @import("log.zig");

// ── Opaque handle types ──
pub const mlx_array = extern struct { ctx: ?*anyopaque = null };
pub const mlx_stream = extern struct { ctx: ?*anyopaque = null };
pub const mlx_device = extern struct { ctx: ?*anyopaque = null };
pub const mlx_string = extern struct { ctx: ?*anyopaque = null };
pub const mlx_map_string_to_array = extern struct { ctx: ?*anyopaque = null };
pub const mlx_map_string_to_string = extern struct { ctx: ?*anyopaque = null };
pub const mlx_map_string_to_array_iterator = extern struct { ctx: ?*anyopaque = null, map_ctx: ?*anyopaque = null };
pub const mlx_vector_array = extern struct { ctx: ?*anyopaque = null };
pub const mlx_closure = extern struct { ctx: ?*anyopaque = null };

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
pub extern "c" fn mlx_stream_new_device(dev: mlx_device) mlx_stream;
pub extern "c" fn mlx_stream_free(s: mlx_stream) c_int;
pub extern "c" fn mlx_default_cpu_stream_new() mlx_stream;
pub extern "c" fn mlx_default_gpu_stream_new() mlx_stream;
pub extern "c" fn mlx_stream_get_device(dev: *mlx_device, stream: mlx_stream) c_int;
pub extern "c" fn mlx_device_get_type(dtype: *mlx_device_type, dev: mlx_device) c_int;

/// True when the stream targets the GPU (custom Metal kernels require it).
pub fn streamIsGpu(s: mlx_stream) bool {
    var dev = mlx_device{ .ctx = null };
    if (mlx_stream_get_device(&dev, s) != 0) return false;
    defer _ = mlx_device_free(dev);
    var dt: mlx_device_type = .cpu;
    if (mlx_device_get_type(&dt, dev) != 0) return false;
    return dt == .gpu;
}
pub extern "c" fn mlx_synchronize(s: mlx_stream) c_int;

// Metal
pub extern "c" fn mlx_metal_is_available(res: *bool) c_int;

// Array creation
pub extern "c" fn mlx_array_new() mlx_array;
pub extern "c" fn mlx_array_new_int(val: c_int) mlx_array;
pub extern "c" fn mlx_array_new_float(val: f32) mlx_array;
pub extern "c" fn mlx_array_new_bool(val: bool) mlx_array;
pub extern "c" fn mlx_array_new_data(data: ?*const anyopaque, shape: [*]const c_int, dim: c_int, dtype: mlx_dtype) mlx_array;
pub extern "c" fn mlx_array_free(arr: mlx_array) c_int;
pub extern "c" fn mlx_array_set(arr: *mlx_array, src: mlx_array) c_int;

// Array info
pub extern "c" fn mlx_array_tostring(str: *mlx_string, arr: mlx_array) c_int;
pub extern "c" fn mlx_array_ndim(arr: mlx_array) usize;
pub extern "c" fn mlx_array_shape(arr: mlx_array) [*]const c_int;
pub extern "c" fn mlx_array_strides(arr: mlx_array) [*]const usize;
pub extern "c" fn mlx_array_size(arr: mlx_array) usize;
pub extern "c" fn mlx_array_dtype(arr: mlx_array) mlx_dtype;
pub extern "c" fn mlx_array_eval(arr: mlx_array) c_int;
pub extern "c" fn mlx_array_itemsize(arr: mlx_array) usize;

// Scalar access
pub extern "c" fn mlx_array_item_float32(res: *f32, arr: mlx_array) c_int;
pub extern "c" fn mlx_array_item_int32(res: *i32, arr: mlx_array) c_int;

// Data access
pub extern "c" fn mlx_array_data_bool(arr: mlx_array) ?[*]const bool;
pub extern "c" fn mlx_array_data_float32(arr: mlx_array) ?[*]const f32;
pub extern "c" fn mlx_array_data_float16(arr: mlx_array) ?[*]const f16;
pub extern "c" fn mlx_array_data_bfloat16(arr: mlx_array) ?[*]const u16;
pub extern "c" fn mlx_array_data_int32(arr: mlx_array) ?[*]const i32;
pub extern "c" fn mlx_array_data_uint32(arr: mlx_array) ?[*]const u32;
pub extern "c" fn mlx_array_data_uint8(arr: mlx_array) ?[*]const u8;

// Vector array
pub extern "c" fn mlx_vector_array_new() mlx_vector_array;
pub extern "c" fn mlx_vector_array_new_data(data: [*]const mlx_array, size: usize) mlx_vector_array;
pub extern "c" fn mlx_vector_array_free(vec: mlx_vector_array) c_int;
pub extern "c" fn mlx_vector_array_size(vec: mlx_vector_array) usize;
pub extern "c" fn mlx_vector_array_get(res: *mlx_array, vec: mlx_vector_array, idx: usize) c_int;
pub extern "c" fn mlx_vector_array_new_value(val: mlx_array) mlx_vector_array;
pub extern "c" fn mlx_vector_array_append_value(vec: mlx_vector_array, val: mlx_array) c_int;

// Closure + compile
pub extern "c" fn mlx_closure_new_func_payload(
    fun: *const fn (*mlx_vector_array, mlx_vector_array, ?*anyopaque) callconv(.c) c_int,
    payload: ?*anyopaque,
    dtor: ?*const fn (?*anyopaque) callconv(.c) void,
) mlx_closure;
pub extern "c" fn mlx_closure_free(cls: mlx_closure) c_int;
pub extern "c" fn mlx_closure_apply(res: *mlx_vector_array, cls: mlx_closure, input: mlx_vector_array) c_int;
pub extern "c" fn mlx_compile(res: *mlx_closure, fun: mlx_closure, shapeless: bool) c_int;
pub extern "c" fn mlx_detail_compile_clear_cache() c_int;

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
pub extern "c" fn mlx_map_string_to_string_insert(map: mlx_map_string_to_string, key: [*:0]const u8, value: [*:0]const u8) c_int;
pub extern "c" fn mlx_map_string_to_string_get(value: *[*:0]const u8, map: mlx_map_string_to_string, key: [*:0]const u8) c_int;

// IO
pub extern "c" fn mlx_load_safetensors(res_0: *mlx_map_string_to_array, res_1: *mlx_map_string_to_string, file: [*:0]const u8, s: mlx_stream) c_int;
pub extern "c" fn mlx_save_safetensors(file: [*:0]const u8, param: mlx_map_string_to_array, metadata: mlx_map_string_to_string) c_int;

// ── Ops ──
pub extern "c" fn mlx_add(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_subtract(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_multiply(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_divide(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_floor_divide(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_negative(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_maximum(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
// Kokoro's SineGen takes the fractional part of a phase ramp (`x - floor(x)`).
pub extern "c" fn mlx_floor(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_ceil(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_log2(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_round(res: *mlx_array, a: mlx_array, decimals: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_sign(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
// Complex plumbing for Kokoro's iSTFTNet head. mlx-c has NO "build a complex
// array from two real ones" op, so the spectrum is assembled as `re + im·i`
// using a complex SCALAR from `mlx_array_new_complex` (float ⊕ complex
// promotes to complex).
pub extern "c" fn mlx_real(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_imag(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_arctan2(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_array_new_complex(real_val: f32, imag_val: f32) mlx_array;
pub extern "c" fn mlx_minimum(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_matmul(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_square(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_sqrt(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_rsqrt(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_exp(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_log(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_abs(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_tanh(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_cos(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_sin(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_erf(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;

pub extern "c" fn mlx_reshape(res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) c_int;
pub extern "c" fn mlx_transpose(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_transpose_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, s: mlx_stream) c_int;
pub extern "c" fn mlx_expand_dims(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_squeeze(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_broadcast_to(res: *mlx_array, a: mlx_array, shape: [*]const c_int, shape_num: usize, s: mlx_stream) c_int;

pub extern "c" fn mlx_take(res: *mlx_array, a: mlx_array, indices: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_take_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) c_int;

pub extern "c" fn mlx_concatenate_axis(res: *mlx_array, arrays: mlx_vector_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_pad(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, low_pad: [*]const c_int, low_pad_num: usize, high_pad: [*]const c_int, high_pad_num: usize, pad_value: mlx_array, mode: [*:0]const u8, s: mlx_stream) c_int;

pub extern "c" fn mlx_softmax_axis(res: *mlx_array, a: mlx_array, axis: c_int, precise: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_argmax_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int;

pub extern "c" fn mlx_copy(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_sort_axis(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_argsort_axis(res: *mlx_array, a: mlx_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_topk(res: *mlx_array, a: mlx_array, k: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_topk_axis(res: *mlx_array, a: mlx_array, k: c_int, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_cumsum(res: *mlx_array, a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) c_int;

pub extern "c" fn mlx_mean_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int;
// Variance along one axis — the InstanceNorm1d half of Kokoro's AdaIN1d
// (`src/kokoro.zig`). `ddof` 0 = biased/population variance, which is what
// torch's normalization layers use.
pub extern "c" fn mlx_var_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, ddof: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_min_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int;

pub extern "c" fn mlx_astype(res: *mlx_array, a: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int;

pub extern "c" fn mlx_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_remainder(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_where(res: *mlx_array, condition: mlx_array, x: mlx_array, y: mlx_array, s: mlx_stream) c_int;

pub extern "c" fn mlx_arange(res: *mlx_array, start: f64, stop: f64, step: f64, dtype: mlx_dtype, s: mlx_stream) c_int;
pub extern "c" fn mlx_full(res: *mlx_array, shape: [*]const c_int, shape_num: usize, val: mlx_array, dtype: mlx_dtype, s: mlx_stream) c_int;
pub extern "c" fn mlx_zeros(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, s: mlx_stream) c_int;
pub extern "c" fn mlx_ones(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, s: mlx_stream) c_int;

pub extern "c" fn mlx_slice(res: *mlx_array, a: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int;
pub extern "c" fn mlx_slice_update(res: *mlx_array, src: mlx_array, update: mlx_array, start: [*]const c_int, start_num: usize, stop: [*]const c_int, stop_num: usize, strides: [*]const c_int, strides_num: usize, s: mlx_stream) c_int;

pub extern "c" fn mlx_triu(res: *mlx_array, x: mlx_array, k: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_tril(res: *mlx_array, x: mlx_array, k: c_int, s: mlx_stream) c_int;

pub extern "c" fn mlx_power(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_less(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_greater(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_greater_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_less_equal(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;

// Quantized matmul
pub extern "c" fn mlx_quantized_matmul(res: *mlx_array, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, transpose_w: bool, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, s: mlx_stream) c_int;

// Gathered quantized matmul (for MoE expert dispatch)
pub extern "c" fn mlx_gather_qmm(res: *mlx_array, x: mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, lhs_indices: mlx_array, rhs_indices: mlx_array, transpose_w: bool, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, sorted_indices: bool, s: mlx_stream) c_int;
// Dense gathered matmul — the unquantized analog of gather_qmm, used for bf16
// MoE expert dispatch. No transpose flag: `b` must already be [..., in, out].
pub extern "c" fn mlx_gather_mm(res: *mlx_array, a: mlx_array, b: mlx_array, lhs_indices: mlx_array, rhs_indices: mlx_array, sorted_indices: bool, s: mlx_stream) c_int;

// Dequantize (fallback)
pub extern "c" fn mlx_dequantize(res: *mlx_array, w: mlx_array, scales: mlx_array, biases: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, global_scale: mlx_array, dtype: mlx_optional_dtype, s: mlx_stream) c_int;

// Quantize (affine group-wise). Returns a vector_array of [q, scales, biases].
// Used by the KV cache quantization path; see src/kv_quant.zig.
pub extern "c" fn mlx_quantize(res: *mlx_vector_array, w: mlx_array, group_size: mlx_optional_int, bits: mlx_optional_int, mode: [*:0]const u8, global_scale: mlx_array, s: mlx_stream) c_int;

// Additional ops for MoE / GatedDeltaNet
pub extern "c" fn mlx_sigmoid(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_sum_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_max_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_sum(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_conv1d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride: c_int, padding: c_int, dilation: c_int, groups: c_int, s: mlx_stream) c_int;
// FFT — real-input forward transform for the TTS speaker-encoder mel/STFT.
// `mlx_fft_norm`: BACKWARD=0 (no scaling on forward; matches mx.fft.rfft default).
pub const mlx_fft_norm = c_int;
pub const MLX_FFT_NORM_BACKWARD: mlx_fft_norm = 0;
pub extern "c" fn mlx_fft_rfft(res: *mlx_array, a: mlx_array, n: c_int, axis: c_int, norm: mlx_fft_norm, s: mlx_stream) c_int;
// Real-output inverse transform — the iSTFT head of Kokoro's iSTFTNet generator
// (`src/kokoro.zig`). `n` is the OUTPUT length in samples, so the caller states
// the frame size (20) rather than inferring it from the 11-bin input.
pub extern "c" fn mlx_fft_irfft(res: *mlx_array, a: mlx_array, n: c_int, axis: c_int, norm: mlx_fft_norm, s: mlx_stream) c_int;
// 2D/3D conv + transposed conv for the media-generation decoders (VAE/vocoder).
// MLX layout: input NHWC / NDHWC, weight OHWI / ODHWI (out-channels first, in-channels last).
pub extern "c" fn mlx_conv2d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, padding_0: c_int, padding_1: c_int, dilation_0: c_int, dilation_1: c_int, groups: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_conv3d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, stride_2: c_int, padding_0: c_int, padding_1: c_int, padding_2: c_int, dilation_0: c_int, dilation_1: c_int, dilation_2: c_int, groups: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_conv_transpose1d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride: c_int, padding: c_int, dilation: c_int, output_padding: c_int, groups: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_conv_transpose2d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, padding_0: c_int, padding_1: c_int, dilation_0: c_int, dilation_1: c_int, output_padding_0: c_int, output_padding_1: c_int, groups: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_conv_transpose3d(res: *mlx_array, input: mlx_array, weight: mlx_array, stride_0: c_int, stride_1: c_int, stride_2: c_int, padding_0: c_int, padding_1: c_int, padding_2: c_int, dilation_0: c_int, dilation_1: c_int, dilation_2: c_int, output_padding_0: c_int, output_padding_1: c_int, output_padding_2: c_int, groups: c_int, s: mlx_stream) c_int;
// Seed-exact noise for diffusion latents (FLUX): pair with mlx_random_key(seed).
pub extern "c" fn mlx_random_normal(res: *mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, loc: f32, scale: f32, key: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_argpartition_axis(res: *mlx_array, a: mlx_array, kth: c_int, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_take_along_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_put_along_axis(res: *mlx_array, a: mlx_array, indices: mlx_array, values: mlx_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_logical_and(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_logical_or(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_repeat_axis(res: *mlx_array, arr: mlx_array, repeats: c_int, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_tile(res: *mlx_array, arr: mlx_array, reps: [*]const c_int, reps_num: usize, s: mlx_stream) c_int;
pub extern "c" fn mlx_log1p(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_logaddexp(res: *mlx_array, a: mlx_array, b: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_stack_axis(res: *mlx_array, arrays: mlx_vector_array, axis: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_split(res: *mlx_vector_array, a: mlx_array, num_splits: c_int, axis: c_int, s: mlx_stream) c_int;

pub const mlx_optional_dtype = extern struct {
    value: mlx_dtype = .float32,
    has_value: bool = false,
};

// ── Fast ops ──
pub extern "c" fn mlx_fast_rms_norm(res: *mlx_array, x: mlx_array, weight: mlx_array, eps: f32, s: mlx_stream) c_int;
pub extern "c" fn mlx_fast_layer_norm(res: *mlx_array, x: mlx_array, weight: mlx_array, bias: mlx_array, eps: f32, s: mlx_stream) c_int;
pub extern "c" fn mlx_fast_rope(res: *mlx_array, x: mlx_array, dims: c_int, traditional: bool, base: mlx_optional_float, scale: f32, offset: c_int, freqs: mlx_array, s: mlx_stream) c_int;
// `mlx_fast_rope_dynamic` accepts a per-row offset array (shape [B], int32) so a single
// kernel launch handles N requests at different KV positions during batched decode.
pub extern "c" fn mlx_fast_rope_dynamic(res: *mlx_array, x: mlx_array, dims: c_int, traditional: bool, base: mlx_optional_float, scale: f32, offset: mlx_array, freqs: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_fast_scaled_dot_product_attention(res: *mlx_array, queries: mlx_array, keys: mlx_array, values: mlx_array, scale: f32, mask_mode: [*:0]const u8, mask_arr: mlx_array, sinks: mlx_array, force_fused: bool, s: mlx_stream) c_int;

// ── Vector of strings (for custom metal kernels) ──
pub const mlx_vector_string = extern struct { ctx: ?*anyopaque = null };
pub extern "c" fn mlx_vector_string_new() mlx_vector_string;
pub extern "c" fn mlx_vector_string_new_data(data: [*]const [*:0]const u8, size: usize) mlx_vector_string;
pub extern "c" fn mlx_vector_string_free(vec: mlx_vector_string) c_int;

// ── Custom Metal kernels ──
pub const mlx_fast_metal_kernel_config = extern struct { ctx: ?*anyopaque = null };
pub extern "c" fn mlx_fast_metal_kernel_config_new() mlx_fast_metal_kernel_config;
pub extern "c" fn mlx_fast_metal_kernel_config_free(cls: mlx_fast_metal_kernel_config) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_add_output_arg(cls: mlx_fast_metal_kernel_config, shape: [*]const c_int, size: usize, dtype: mlx_dtype) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_set_grid(cls: mlx_fast_metal_kernel_config, g1: c_int, g2: c_int, g3: c_int) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_set_thread_group(cls: mlx_fast_metal_kernel_config, t1: c_int, t2: c_int, t3: c_int) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_add_template_arg_dtype(cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, dtype: mlx_dtype) c_int;
pub extern "c" fn mlx_any_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_add_template_arg_int(cls: mlx_fast_metal_kernel_config, name: [*:0]const u8, value: c_int) c_int;
pub extern "c" fn mlx_fast_metal_kernel_config_set_verbose(cls: mlx_fast_metal_kernel_config, verbose: bool) c_int;

pub const mlx_fast_metal_kernel = extern struct { ctx: ?*anyopaque = null };
pub extern "c" fn mlx_fast_metal_kernel_new(name: [*:0]const u8, input_names: mlx_vector_string, output_names: mlx_vector_string, source: [*:0]const u8, header: [*:0]const u8, ensure_row_contiguous: bool, atomic_outputs: bool) mlx_fast_metal_kernel;
pub extern "c" fn mlx_fast_metal_kernel_free(cls: mlx_fast_metal_kernel) c_int;
pub extern "c" fn mlx_fast_metal_kernel_apply(outputs: *mlx_vector_array, cls: mlx_fast_metal_kernel, inputs: mlx_vector_array, config: mlx_fast_metal_kernel_config, s: mlx_stream) c_int;

// ── Random ──
pub extern "c" fn mlx_random_categorical(res: *mlx_array, logits: mlx_array, axis: c_int, key: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_random_key(res: *mlx_array, seed: u64) c_int;
// Uniform noise — Kokoro's SineGen draws a random initial phase per harmonic.
// Bounds are ARRAYS, unlike mlx_random_normal's scalar loc/scale.
pub extern "c" fn mlx_random_uniform(res: *mlx_array, low: mlx_array, high: mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, key: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_random_seed(seed: u64) c_int;
// Uniform random integers in [low, high) — DiffusionGemma canvas init/renoise.
pub extern "c" fn mlx_random_randint(res: *mlx_array, low: mlx_array, high: mlx_array, shape: [*]const c_int, shape_num: usize, dtype: mlx_dtype, key: mlx_array, s: mlx_stream) c_int;

// ── Diffusion sampler ops ──
pub extern "c" fn mlx_cummax(res: *mlx_array, a: mlx_array, axis: c_int, reverse: bool, inclusive: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_logsumexp_axis(res: *mlx_array, a: mlx_array, axis: c_int, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_all(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_isfinite(res: *mlx_array, a: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_clip(res: *mlx_array, a: mlx_array, a_min: mlx_array, a_max: mlx_array, s: mlx_stream) c_int;
pub extern "c" fn mlx_contiguous(res: *mlx_array, a: mlx_array, allow_col_major: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_mean(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_mean_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_std(res: *mlx_array, a: mlx_array, keepdims: bool, ddof: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_std_axes(res: *mlx_array, a: mlx_array, axes: [*]const c_int, axes_num: usize, keepdims: bool, ddof: c_int, s: mlx_stream) c_int;
pub extern "c" fn mlx_max(res: *mlx_array, a: mlx_array, keepdims: bool, s: mlx_stream) c_int;
pub extern "c" fn mlx_array_item_bool(res: *bool, arr: mlx_array) c_int;

// ── Batch eval ──
pub extern "c" fn mlx_eval(outputs: mlx_vector_array) c_int;
pub extern "c" fn mlx_async_eval(outputs: mlx_vector_array) c_int;

// ── Memory management ──
pub extern "c" fn mlx_clear_cache() c_int;
pub extern "c" fn mlx_set_memory_limit(res: *usize, limit: usize) c_int;
pub extern "c" fn mlx_set_cache_limit(res: *usize, limit: usize) c_int;
pub extern "c" fn mlx_set_wired_limit(res: *usize, limit: usize) c_int;
pub extern "c" fn mlx_get_active_memory(res: *usize) c_int;
pub extern "c" fn mlx_get_cache_memory(res: *usize) c_int;
pub extern "c" fn mlx_get_peak_memory(res: *usize) c_int;
pub extern "c" fn mlx_reset_peak_memory() c_int;

// ── Device info ──
pub const mlx_device_info = extern struct { ctx: ?*anyopaque = null };
pub extern "c" fn mlx_device_info_new() mlx_device_info;
pub extern "c" fn mlx_device_info_get(info: *mlx_device_info, dev: mlx_device) c_int;
pub extern "c" fn mlx_device_info_free(info: mlx_device_info) c_int;
pub extern "c" fn mlx_device_info_get_size(res: *usize, info: mlx_device_info, key: [*:0]const u8) c_int;
pub extern "c" fn mlx_device_info_get_string(res: *[*:0]const u8, info: mlx_device_info, key: [*:0]const u8) c_int;

// ── Error handler ──
pub const mlx_error_handler_func = ?*const fn ([*:0]const u8, ?*anyopaque) callconv(.c) void;
pub extern "c" fn mlx_set_error_handler(handler: mlx_error_handler_func, data: ?*anyopaque, dtor: ?*const fn (?*anyopaque) callconv(.c) void) void;

// ── Zig helper wrappers ──

/// Get the default compute stream. Normally the GPU (Metal) stream. When MLX
/// was built without a Metal backend (e.g. the iOS Simulator slice, where MLX's
/// Metal device can't be constructed), there is no GPU stream — fall back to the
/// CPU stream so the engine runs on the Accelerate backend. The check is cached;
/// on real hardware Metal is always available, so this is the GPU stream as
/// before with no measurable overhead.
var no_gpu_backend_cache: ?bool = null;

/// True when MLX has no Metal/GPU backend (e.g. the iOS Simulator slice built
/// with MLX_BUILD_METAL=OFF). Callers use this to take CPU-only paths and skip
/// GPU-only work (kernel-fusion JIT compilation). Always false on real hardware.
pub fn noGpuBackend() bool {
    if (no_gpu_backend_cache == null) {
        var avail: bool = false;
        _ = mlx_metal_is_available(&avail);
        no_gpu_backend_cache = !avail;
    }
    return no_gpu_backend_cache.?;
}

pub fn gpuStream() mlx_stream {
    return if (noGpuBackend()) mlx_default_cpu_stream_new() else mlx_default_gpu_stream_new();
}

/// Print an array for debugging
pub fn printArray(label: []const u8, arr: mlx_array) void {
    _ = mlx_array_eval(arr);
    var str = mlx_string_new();
    _ = mlx_array_tostring(&str, arr);
    const data = mlx_string_data(str);
    log.debug("{s}: {s}\n", .{ label, data });
    _ = mlx_string_free(str);
}

/// Get array shape as a Zig slice
pub fn getShape(arr: mlx_array) []const c_int {
    const ndim = mlx_array_ndim(arr);
    if (ndim == 0) return &.{};
    return mlx_array_shape(arr)[0..ndim];
}

/// How a residual's dtype moved between two observation points.
pub const DtypeStep = enum { quiet, widened, narrowed, changed };

/// Bytes per element, for deciding whether a dtype change WIDENED the stream.
/// Only the dtypes a residual can actually hold are named; anything else is
/// reported as a plain change rather than guessed at.
pub fn dtypeBytes(d: mlx_dtype) ?u8 {
    return switch (d) {
        .float16, .bfloat16 => 2,
        .float32 => 4,
        .float64 => 8,
        else => null,
    };
}

pub fn dtypeStep(prev: mlx_dtype, cur: mlx_dtype) DtypeStep {
    if (prev == cur) return .quiet;
    const pb = dtypeBytes(prev) orelse return .changed;
    const cb = dtypeBytes(cur) orelse return .changed;
    if (cb > pb) return .widened;
    if (cb < pb) return .narrowed;
    return .changed; // same width, different kernel selection (bf16 vs f16)
}

/// One-shot latches for `[dtype-trace]`, keyed by forward-path name. A vision
/// tower and its text trunk are different paths and latch independently.
var dtype_trace_seen: [12]?[]const u8 = @splat(null);

/// True the FIRST time `path` is seen, false forever after. After the first
/// forward the trace costs nothing — a per-layer FFI dtype read on every token
/// would be a real (if small) tax, and identical repeated lines would bury the
/// one line that matters.
pub fn dtypeTraceArm(path: []const u8) bool {
    for (&dtype_trace_seen) |*slot| {
        if (slot.*) |seen| {
            if (std.mem.eql(u8, seen, path)) return false;
            continue;
        }
        slot.* = path;
        return true;
    }
    return false; // table full — stop tracing rather than spam
}

pub fn resetDtypeTraceForTest() void {
    dtype_trace_seen = @splat(null);
}

/// DIAGNOSTIC (`[dtype-trace]`): watch the residual stream's dtype across one
/// forward pass.
///
/// A residual wider than the weights promotes EVERY projection's weight on
/// read, so the cost lands as a UNIFORM multiple across attention, MLP and
/// lm_head — which reads like a platform problem, not a bug. The Laguna YaRN
/// mscale table (f32 constant multiplied into bf16 q/k) cost 3x and was
/// invisible to op count, kernel choice, KV ablation and the CPU/GPU split.
/// Endpoints alone say THAT it widened; per-layer lines on every layer are
/// unreadable. So: log both endpoints, and in between log only where the dtype
/// actually moves — one line that names the layer responsible.
pub const DtypeTrace = struct {
    path: []const u8,
    on: bool,
    last: mlx_dtype,

    pub fn begin(path: []const u8, h: mlx_array, weight: ?mlx_array) DtypeTrace {
        const on = dtypeTraceArm(path);
        const d = mlx_array_dtype(h);
        if (on) {
            log.info("[dtype-trace] {s}: residual in = {s}, first weight = {s}\n", .{
                path,
                @tagName(d),
                if (weight) |w| (if (w.ctx == null) "null" else @tagName(mlx_array_dtype(w))) else "n/a",
            });
        }
        return .{ .path = path, .on = on, .last = d };
    }

    pub fn layer(self: *DtypeTrace, h: mlx_array, layer_idx: usize) void {
        if (!self.on) return;
        const d = mlx_array_dtype(h);
        const step = dtypeStep(self.last, d);
        if (step == .quiet) return;
        log.info("[dtype-trace] {s}: residual {s} at layer {d}: {s} -> {s}\n", .{
            self.path,
            @tagName(step),
            layer_idx,
            @tagName(self.last),
            @tagName(d),
        });
        self.last = d;
    }

    pub fn end(self: *DtypeTrace, h: mlx_array) void {
        if (!self.on) return;
        log.info("[dtype-trace] {s}: residual out = {s}\n", .{ self.path, @tagName(mlx_array_dtype(h)) });
    }
};

/// Check if an mlx-c call succeeded (returns 0 on success)
/// DIAGNOSTIC op counter. Every graph-building op funnels through `check`, so
/// this counts ops ISSUED (not kernels dispatched — MLX fuses some). Only read
/// by the decode forward probe; the increment is a single relaxed add on a
/// path that already does an FFI call, so it does not move any benchmark.
pub var op_count: std.atomic.Value(u64) = .init(0);

/// TEST-ONLY fault injection at the ONE chokepoint every checked mlx-c call
/// goes through: `fault.arm(k)` makes the k-th SUBSEQUENT call that would have
/// SUCCEEDED report `error.MlxError` instead, so an ownership audit can walk
/// every error path of a function without patching the function. Failing calls
/// are not counted (they already take the error path on their own). The op
/// itself has already executed when `check` sees its return code — the
/// argument is evaluated first — so what is injected is a succeeding op
/// REPORTING failure; the caller's `&out` handle is a valid, freeable array,
/// which is exactly the state its cleanup must survive. Compiled only under
/// `builtin.is_test`, so a release build carries neither the branch nor the
/// symbols.
pub const fault = struct {
    var remaining: u64 = 0; // 0 = disarmed
    var fired: bool = false;

    /// Fail the k-th checked call from here (k >= 1). `arm(0)` is a no-op arm.
    pub fn arm(k: u64) void {
        remaining = k;
        fired = false;
    }
    pub fn disarm() void {
        remaining = 0;
    }
    /// Did the armed fault actually fire? False means the call under test
    /// returned before reaching its k-th checked op — an assertion the test
    /// needs, or the sweep silently stops covering anything.
    pub fn didFire() bool {
        return fired;
    }
    fn hit() bool {
        if (remaining == 0) return false;
        remaining -= 1;
        if (remaining != 0) return false;
        fired = true;
        return true;
    }
};

pub fn check(ret: c_int) !void {
    _ = op_count.fetchAdd(1, .monotonic);
    if (ret != 0) return error.MlxError;
    if (comptime builtin.is_test) {
        if (fault.hit()) return error.MlxError;
    }
}

// ---------------------------------------------------------------------------
// Wired-residency policy (mlxfast notes/47 class).
//
// MLX's Metal residency set has a capacity set by `mlx_set_wired_limit`. Two
// failure shapes bracket the useful setting:
//  * capacity 0 (MLX default): nothing is wired, and the driver re-establishes
//    residency for the whole RAM-resident model on every command buffer —
//    measured upstream as 9-15 ms kernelStart gaps across a prefill.
//  * capacity >> live bytes (our historical `max_recommended_working_set_size`):
//    every transient allocation fits the residency set, so each alloc/evict
//    issues a Metal commit() — per-allocation overhead on the decode path.
// The `fit` policy wires the CURRENT live set with a small slack and no
// headroom: weights migrate into the set in one resize commit, and every later
// transient FAILS the fit test and stays on the commit-free unwired path.
// Re-applied after every load/unload so the capacity tracks the live set.

pub const WiredMode = enum {
    off, // wire nothing (MLX default behavior)
    max, // capacity = max_recommended_working_set_size (historical behavior)
    fit, // capacity = live bytes + slack (zero headroom)

    pub fn fromEnv(value: ?[]const u8) WiredMode {
        const v = value orelse return .max;
        if (std.mem.eql(u8, v, "off") or std.mem.eql(u8, v, "0")) return .off;
        if (std.mem.eql(u8, v, "max")) return .max;
        if (std.mem.eql(u8, v, "fit")) return .fit;
        return .max;
    }
};

/// Zero-headroom capacity for `fit` mode. `set_wired_limit` above the
/// recommended working set is an uncatchable MLX error, so the target is
/// clamped a margin under it; a dead device query or an empty live set
/// declines (null) rather than wiring garbage.
pub fn wiredFitTarget(active_bytes: usize, slack_bytes: usize, max_rec: usize) ?usize {
    if (active_bytes == 0 or max_rec == 0) return null;
    const margin = 256 << 20;
    if (max_rec <= margin) return null;
    const cap = max_rec - margin;
    const target = active_bytes +| slack_bytes;
    return @min(target, cap);
}

pub const WiredPolicyResult = struct { mode: WiredMode, target: ?usize };

pub fn maxRecommendedWorkingSet() usize {
    var dev = mlx_device{ .ctx = null };
    _ = mlx_get_default_device(&dev);
    var info = mlx_device_info_new();
    defer _ = mlx_device_info_free(info);
    if (mlx_device_info_get(&info, dev) != 0) return 0;
    var max_rec: usize = 0;
    if (mlx_device_info_get_size(&max_rec, info, "max_recommended_working_set_size") != 0) return 0;
    return max_rec;
}

/// Apply the wired-residency policy. Call on the inference thread AFTER a
/// model load or unload completes (and from the offline run path after load)
/// so `fit` capacity tracks the live set. The caller logs the result.
pub fn applyWiredPolicy() WiredPolicyResult {
    const mode = WiredMode.fromEnv(if (std.c.getenv("MLX_SERVE_WIRED")) |p| std.mem.span(p) else null);
    if (noGpuBackend()) return .{ .mode = mode, .target = null };
    var prev: usize = 0;
    switch (mode) {
        .off => {
            _ = mlx_set_wired_limit(&prev, 0);
            return .{ .mode = mode, .target = 0 };
        },
        .max => {
            const max_rec = maxRecommendedWorkingSet();
            if (max_rec == 0) return .{ .mode = mode, .target = null };
            _ = mlx_set_wired_limit(&prev, max_rec);
            return .{ .mode = mode, .target = max_rec };
        },
        .fit => {
            // Drop cached (free) buffers first so the resize walk wires only
            // live weights and the capacity leaves no headroom for scratch.
            _ = mlx_clear_cache();
            const max_rec = maxRecommendedWorkingSet();
            var active: usize = 0;
            _ = mlx_get_active_memory(&active);
            const slack_mb: usize = blk: {
                const raw_c = std.c.getenv("MLX_SERVE_WIRED_SLACK_MB") orelse break :blk 64;
                const raw = std.mem.span(raw_c);
                break :blk std.fmt.parseInt(usize, raw, 10) catch 64;
            };
            const target = wiredFitTarget(active, slack_mb << 20, max_rec) orelse
                return .{ .mode = mode, .target = null };
            // Shrink-then-grow forces ResidencySet::resize to re-walk, pulling
            // buffers allocated since the last apply out of the unwired set.
            _ = mlx_set_wired_limit(&prev, 0);
            _ = mlx_set_wired_limit(&prev, target);
            return .{ .mode = mode, .target = target };
        },
    }
}

test "wired mode from env" {
    const t = std.testing;
    // Default (unset) is the historical behavior until the fit-policy A/B
    // picks a winner; unknown values also take the default rather than
    // silently wiring nothing.
    try t.expectEqual(WiredMode.max, WiredMode.fromEnv(null));
    try t.expectEqual(WiredMode.off, WiredMode.fromEnv("off"));
    try t.expectEqual(WiredMode.off, WiredMode.fromEnv("0"));
    try t.expectEqual(WiredMode.max, WiredMode.fromEnv("max"));
    try t.expectEqual(WiredMode.fit, WiredMode.fromEnv("fit"));
    try t.expectEqual(WiredMode.max, WiredMode.fromEnv("banana"));
}

test "wired fit target: zero headroom, clamped, declines empty" {
    const t = std.testing;
    const gb = 1 << 30;
    // Normal: live + slack.
    try t.expectEqual(@as(?usize, 10 * gb + (64 << 20)), wiredFitTarget(10 * gb, 64 << 20, 115 * gb));
    // Live set near the ceiling clamps a margin under max_rec.
    try t.expectEqual(@as(?usize, 115 * gb - (256 << 20)), wiredFitTarget(115 * gb, 64 << 20, 115 * gb));
    // Nothing live / dead query: decline.
    try t.expectEqual(@as(?usize, null), wiredFitTarget(0, 64 << 20, 115 * gb));
    try t.expectEqual(@as(?usize, null), wiredFitTarget(10 * gb, 64 << 20, 0));
}

// ── mlx-c error latch: an MLX failure is an ERROR, never `exit(-1)` ──
//
// mlx-c's DEFAULT error handler is
//
//     static void mlx_error_handler_default_(const char* msg, void* data) {
//       printf("MLX error: %s\n", msg);
//       exit(-1);
//     }
//
// (`lib/mlxc-src/mlx/c/error.cpp`), installed unless someone calls
// `mlx_set_error_handler`. Every mlx-c entry point is
// `try { ... } catch (std::exception& e) { mlx_error(e.what()); return 1; }`,
// so an MLX exception is CAUGHT at the C boundary and then the process is
// killed by the handler before the `return 1` ever reaches us.
//
// That is what a Metal working-set OOM looks like from the outside: mlx
// 0.32.2 stores a failed command buffer's status in `CommandEncoder::error_`
// and re-throws it from `synchronize()`/`get_command_encoder()`
// (`lib/mlx-src/mlx/backend/metal/device.cpp:564-579`), a stream thread's
// exception is captured into `StreamThread::error` and re-thrown on the main
// thread at the next enqueue (`lib/mlx-src/mlx/scheduler.cpp:77-79,99-112`),
// and mlx-c catches it — the live 458,832-token report's
// "…Insufficient Memory… at transforms.cpp:15" is `_mlx_error`'s own
// " at %s:%d" suffix naming `mlx/c/transforms.cpp`, i.e. `mlx_eval`. The
// death was ours to prevent, not Metal's to inflict (issue #353).
//
// So: install a handler that LATCHES the message and returns. The mlx-c call
// then returns 1 as designed, and the engine turns the latch into a named
// error at its next checkpoint. mlx clears its own error state when it
// throws (`Error::check` exchanges the message out), so the process stays
// serviceable for the next request.
var mlx_error_latched = std.atomic.Value(bool).init(false);
var mlx_error_buf: [512]u8 = undefined;
var mlx_error_len: usize = 0;
/// A pthread mutex, not `std.Io.Mutex`, for `log.zig`'s reason and one more:
/// mlx-c calls the handler from whatever thread raised — an MLX stream thread
/// included — and none of them carry an `Io` handle.
var mlx_error_mtx: std.c.pthread_mutex_t = .{};
fn lockErrBuf() void {
    _ = std.c.pthread_mutex_lock(&mlx_error_mtx);
}
fn unlockErrBuf() void {
    _ = std.c.pthread_mutex_unlock(&mlx_error_mtx);
}

/// Classification of a latched mlx-c message. Memory failures must reach the
/// client as a memory 503 (`ModelRegistry.loadErrorFromName` keeps BOTH
/// `error.OutOfMemory` and `error.InsufficientMemory` in that class), so the
/// spelling of the failure decides the error we raise. Every MLX allocation
/// failure and every Metal command-buffer OOM lands in one of these:
///
///   * "[METAL] Command buffer execution failed: Insufficient Memory …"
///     — the working-set abort this latch exists for;
///   * "[malloc] Unable to allocate N bytes." / "[metal::malloc] …" —
///     `MetalAllocator::malloc`'s own throws;
///   * "[metal::malloc] Resource limit (N) exceeded." — buffer-count wall,
///     which is a memory condition from the operator's side too.
pub fn mlxErrorIsMemory(msg: []const u8) bool {
    const needles = [_][]const u8{
        "Insufficient Memory",
        "insufficient memory",
        "Unable to allocate",
        "Resource limit",
        "out of memory",
        "Out of memory",
        "maximum allowed buffer size",
    };
    for (needles) |n| {
        if (std.mem.indexOf(u8, msg, n) != null) return true;
    }
    return false;
}

fn latchMlxError(msg: [*:0]const u8, data: ?*anyopaque) callconv(.c) void {
    _ = data;
    const span = std.mem.span(msg);
    lockErrBuf();
    defer unlockErrBuf();
    // FIRST error wins: mlx preserves the earliest error for the same reason
    // (a poisoned stream produces follow-on noise that hides the cause).
    if (!mlx_error_latched.load(.acquire)) {
        const n = @min(span.len, mlx_error_buf.len);
        @memcpy(mlx_error_buf[0..n], span[0..n]);
        mlx_error_len = n;
        mlx_error_latched.store(true, .release);
    }
    log.err("[mlx] {s}\n", .{span});
}

/// Install the latching handler. Called ONCE from `main()`, next to
/// `server.applyMlxCacheLimit`, for the same reason: a per-path install is how
/// a mode ends up silently keeping the old behaviour.
///
/// `MLX_SERVE_MLX_ERROR_LATCH=0` restores mlx-c's `exit(-1)` handler for
/// same-boot A/Bs and for anyone who would rather have a core than a 503.
pub fn installErrorHandler() void {
    if (std.c.getenv("MLX_SERVE_MLX_ERROR_LATCH")) |p| {
        if (std.mem.eql(u8, std.mem.span(p), "0")) {
            log.info("[mlx] error latch disabled — an MLX error will exit the process\n", .{});
            return;
        }
    }
    mlx_set_error_handler(latchMlxError, null, null);
}

/// True when an MLX call has failed since the last `takeError`.
pub fn errorPending() bool {
    return mlx_error_latched.load(.acquire);
}

/// PEEK the latch WITHOUT consuming it: the NAME of the Zig error a
/// `checkError`/`checkErrorDecode` call would raise right now, or null when
/// nothing is latched. The names are `@errorName`'s own, so
/// `scheduler.Slot.errorNameIsMemory` classifies a peeked failure exactly as
/// it classifies a consumed one and the client still gets the memory 503.
///
/// It exists because a slot FINISH must not answer 200 over a forward that
/// failed (external review of PR #363, item 3) and yet must not STEAL the
/// latch from the decode-tick wrapper, whose job is to fail every OTHER slot
/// in a batched group from that same failure. Peek here, mark this slot, let
/// the wrapper consume.
pub fn peekErrorName() ?[]const u8 {
    if (!mlx_error_latched.load(.acquire)) return null;
    lockErrBuf();
    defer unlockErrBuf();
    return if (mlxErrorIsMemory(mlx_error_buf[0..mlx_error_len])) "OutOfMemory" else "MlxFailure";
}

/// Consume the latched message into `buf`, clearing the latch. Returns null
/// when nothing is latched. The message is COPIED so the caller can format it
/// after another MLX call has run.
pub fn takeError(buf: []u8) ?[]const u8 {
    if (!mlx_error_latched.load(.acquire)) return null;
    lockErrBuf();
    defer unlockErrBuf();
    const n = @min(mlx_error_len, buf.len);
    @memcpy(buf[0..n], mlx_error_buf[0..n]);
    mlx_error_len = 0;
    mlx_error_latched.store(false, .release);
    return buf[0..n];
}

/// One-shot fault injection AT the engine's checkpoint, for builds that are
/// not test builds — `fault.arm` above is `builtin.is_test`-only and the
/// invariant this proves ("an MLX error costs one REQUEST, not the server")
/// can only be observed end to end, with a real model behind a real socket.
///
/// `MLX_SERVE_MLX_FAULT_CHUNK=<n>` latches a synthetic Metal working-set OOM
/// at the n-th `checkError` of the process and then disarms itself, so the
/// NEXT request runs clean and the test can assert both halves. Absent or `0`
/// = off (`diagEnvOn` discipline: an env read with `getenv != null` is armed
/// by `=0`). Resolved once; the armed path costs one relaxed load per prefill
/// chunk.
const FAULT_CHUNK_MSG = "[METAL] Command buffer execution failed: Insufficient Memory (injected by MLX_SERVE_MLX_FAULT_CHUNK). at transforms.cpp:15";
const FAULT_STEP_MSG = "[METAL] Command buffer execution failed: Insufficient Memory (injected by MLX_SERVE_MLX_FAULT_STEP). at transforms.cpp:15";

/// One armed, self-disarming injector. Two instances: one for the prefill
/// chunk checkpoint, one for the decode step checkpoint — separate counters
/// because a test that wants to fail the 3rd DECODE step must not have its
/// count consumed by the prefill's chunks.
const FaultSite = struct {
    env: [:0]const u8,
    msg: [:0]const u8,
    at: ?u64 = null,
    seen: u64 = 0,

    fn target(self: *FaultSite) u64 {
        if (self.at) |v| return v;
        var v: u64 = 0;
        if (std.c.getenv(self.env.ptr)) |pz| {
            v = std.fmt.parseInt(u64, std.mem.span(pz), 10) catch 0;
        }
        self.at = v;
        if (v > 0) log.warn("[mlx] FAULT INJECTION armed: {s}=#{d}\n", .{ self.env, v });
        return v;
    }

    /// Latch this site's message on the n-th call, then disarm — one boot
    /// exercises both the failure and the recovery.
    fn maybeFire(self: *FaultSite) void {
        const t = self.target();
        if (t == 0) return;
        self.seen += 1;
        if (self.seen != t) return;
        self.at = 0;
        latchErrorForTest(self.msg);
    }

    fn reset(self: *FaultSite) void {
        self.at = null;
        self.seen = 0;
    }
};

var fault_chunk = FaultSite{ .env = "MLX_SERVE_MLX_FAULT_CHUNK", .msg = FAULT_CHUNK_MSG };
var fault_step = FaultSite{ .env = "MLX_SERVE_MLX_FAULT_STEP", .msg = FAULT_STEP_MSG };

/// Consume a latched MLX failure as a Zig error: `error.OutOfMemory` for the
/// memory class (named memory 503), `error.MlxFailure` for everything else.
/// Zero cost when nothing is latched — one acquire load.
fn consumeLatch() !void {
    if (!mlx_error_latched.load(.acquire)) return;
    var buf: [512]u8 = undefined;
    const msg = takeError(&buf) orelse return;
    if (mlxErrorIsMemory(msg)) return error.OutOfMemory;
    return error.MlxFailure;
}

/// PREFILL checkpoint: called once per chunk plus once after the last.
pub fn checkError() !void {
    fault_chunk.maybeFire();
    return consumeLatch();
}

/// DECODE checkpoint: called once per decode tick, after the tick's forward.
/// A separate entry point ONLY so the two injectors count separately — a
/// decode-time failure is the same latch and the same errors.
///
/// It exists at all because a prefill-only check attributes nothing: a decode
/// forward that failed left the slot emitting tokens sampled from unwritten
/// buffers, the request finished 200 with garbage, and the latch was handed
/// to the NEXT request's first prefill chunk as ITS 503 (#353 follow-up).
pub fn checkErrorDecode() !void {
    fault_step.maybeFire();
    return consumeLatch();
}

/// Latch a message as if mlx-c had raised it. Tests call it directly; the
/// `MLX_SERVE_MLX_FAULT_CHUNK` injector above calls it once.
pub fn latchErrorForTest(msg: [:0]const u8) void {
    latchMlxError(msg.ptr, null);
}

/// TEST-ONLY reset so one test's injector state cannot leak into another.
pub fn resetFaultChunkForTest() void {
    fault_chunk.reset();
    fault_step.reset();
    var buf: [512]u8 = undefined;
    _ = takeError(&buf);
}

test "an MLX memory failure is classified as a memory error, an argument error is not" {
    const t = std.testing;
    // The exact strings mlx produces for the two OOM shapes issue #353 hit.
    try t.expect(mlxErrorIsMemory("[METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory). at transforms.cpp:15"));
    try t.expect(mlxErrorIsMemory("[malloc] Unable to allocate 8589934592 bytes."));
    try t.expect(mlxErrorIsMemory("[metal::malloc] Resource limit (128000) exceeded."));
    try t.expect(mlxErrorIsMemory("[metal::malloc] Attempting to allocate 99 bytes which is greater than the maximum allowed buffer size of 5 bytes."));
    // A shape/argument bug must NOT be reported to the client as "out of
    // memory" — it is a 500-class engine fault, not something the operator
    // fixes by shortening the prompt.
    try t.expect(!mlxErrorIsMemory("[slice_update] Invalid slice sizes."));
    try t.expect(!mlxErrorIsMemory("[matmul] Last dimension of first input must match."));
}

test "the latch turns an mlx-c error into a Zig error instead of exiting, once" {
    const t = std.testing;
    try t.expect(!errorPending());
    try checkError();

    latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    try t.expect(errorPending());
    // FIRST error wins — a poisoned stream's follow-on noise must not rename
    // the cause.
    latchErrorForTest("[slice_update] Invalid slice sizes.");
    try t.expectError(error.OutOfMemory, checkError());
    // Consumed: the next request starts clean, which is the whole point of
    // not exiting.
    try t.expect(!errorPending());
    try checkError();

    latchErrorForTest("[matmul] Last dimension of first input must match.");
    try t.expectError(error.MlxFailure, checkError());
    try t.expect(!errorPending());
}

test "the injected MLX error costs ONE checkError and the engine keeps working after it" {
    const t = std.testing;
    // The invariant behind issue #353, at the level a hermetic test can reach:
    // an MLX failure is consumed EXACTLY once, and everything after it runs.
    // The end-to-end half of it — a 503 for the failing request and a 200 for
    // the next one on the same server — needs a real model behind a real
    // socket, which is why the injector below is also reachable in a release
    // build through `MLX_SERVE_MLX_FAULT_CHUNK` and pinned by
    // `tests/test_mlx_error_recovery.sh`.
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();

    // Chunk 1 and 2 of a prefill: clean.
    try checkError();
    try checkError();

    // Chunk 3 fails, the way the Metal working-set abort does.
    latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    try t.expectError(error.OutOfMemory, checkError());

    // And the process is still an engine: the latch is clear, the next
    // prefill's chunks pass, and MLX itself still evaluates. (Before the
    // handler was installed there was no "after" — mlx-c called exit(-1).)
    try t.expect(!errorPending());
    try checkError();
    const s = gpuStream();
    const shape = [_]c_int{2};
    const data = [_]f32{ 1.5, 2.5 };
    const a = mlx_array_new_data(@ptrCast(&data), &shape, 1, .float32);
    defer _ = mlx_array_free(a);
    var sum = mlx_array_new();
    defer _ = mlx_array_free(sum);
    try t.expectEqual(@as(c_int, 0), mlx_sum(&sum, a, false, s));
    _ = mlx_array_eval(sum);
    var out: f32 = 0;
    _ = mlx_array_item_float32(&out, sum);
    try t.expectApproxEqAbs(@as(f32, 4.0), out, 1e-6);
    try checkError();
}

test "peekErrorName names the class WITHOUT consuming the latch" {
    const t = std.testing;
    // The slot-finish guard (external review of PR #363, item 3) has to know
    // whether the last forward failed before it publishes a terminator, and
    // has to leave the latch for the tick wrapper that fails the rest of a
    // batched group from the same failure. Consuming here would answer this
    // slot correctly and let its siblings finish 200 on the same dead buffers.
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();

    try t.expect(peekErrorName() == null);

    latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    // Same name the consuming path raises, so the memory classification (and
    // with it the 503) survives the peek.
    try t.expectEqualStrings("OutOfMemory", peekErrorName().?);
    // ...and the latch is STILL there, twice over.
    try t.expectEqualStrings("OutOfMemory", peekErrorName().?);
    try t.expect(errorPending());
    try t.expectError(error.OutOfMemory, checkErrorDecode());
    try t.expect(peekErrorName() == null);

    // A shape bug is not a memory failure and must not be reported as one.
    latchErrorForTest("[matmul] Last dimension of first input must match.");
    try t.expectEqualStrings("MlxFailure", peekErrorName().?);
    try t.expectError(error.MlxFailure, checkErrorDecode());
    try t.expect(peekErrorName() == null);
}

test "a DECODE-time MLX failure is attributed to the decoding request, not the next one" {
    const t = std.testing;
    // The third defect of the #353 branch: `checkError` was consumed ONLY by
    // the prefill chunk loop. A decode forward that failed left the slot
    // sampling from buffers Metal never wrote — the request finished 200 with
    // garbage — and the latch waited for the NEXT request's first prefill
    // chunk, which answered 503 for a failure that was not its own. Two wrong
    // answers from one error.
    resetFaultChunkForTest();
    defer resetFaultChunkForTest();

    // Both checkpoints read the SAME latch: whichever runs first consumes it.
    try checkError();
    try checkErrorDecode();
    latchErrorForTest("[METAL] Command buffer execution failed: Insufficient Memory. at transforms.cpp:15");
    try t.expectError(error.OutOfMemory, checkErrorDecode());
    try t.expect(!errorPending());
    // Consumed by decode means the next prefill is clean — that is the whole
    // point of attributing it where it happened.
    try checkError();

    // The two injectors count SEPARATELY: a test that wants to fail the n-th
    // decode step must not have its count eaten by the prefill's chunks.
    // (Neither is armed here, so both are inert and this asserts the shape.)
    try checkError();
    try checkError();
    try checkErrorDecode();
    try t.expect(!errorPending());
}
