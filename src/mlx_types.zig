//! Backend-independent MLX handle layouts and policy values.

const std = @import("std");

pub const mlx_array = extern struct { ctx: ?*anyopaque = null };

pub const mlx_stream = extern struct { ctx: ?*anyopaque = null };

pub const mlx_device = extern struct { ctx: ?*anyopaque = null };

pub const mlx_string = extern struct { ctx: ?*anyopaque = null };

pub const mlx_map_string_to_array = extern struct { ctx: ?*anyopaque = null };

pub const mlx_map_string_to_string = extern struct { ctx: ?*anyopaque = null };

pub const mlx_map_string_to_array_iterator = extern struct { ctx: ?*anyopaque = null, map_ctx: ?*anyopaque = null };

pub const mlx_vector_array = extern struct { ctx: ?*anyopaque = null };

pub const mlx_closure = extern struct { ctx: ?*anyopaque = null };

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

pub const mlx_device_info = extern struct { ctx: ?*anyopaque = null };

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

pub const WiredPolicyResult = struct { mode: WiredMode, target: ?usize };
