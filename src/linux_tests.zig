test {
    _ = @import("lan.zig");
    _ = @import("lan_mdns.zig");
    _ = @import("lan_net.zig");
    if (@import("build_cfg.zig").gguf_only) _ = @import("gguf_stubs_test.zig");
    _ = @import("kv_quant_config.zig");
    _ = @import("loop_detect.zig");
    _ = @import("generate_common.zig");
}
