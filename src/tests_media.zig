//! Split unit-test root: pure-SDK media/3D/audio/image/video plugins, sdk/media and sdk/mesh.

// pub: Sushi's EXL3 module (src/sdk/quant/exl3/sushi) reaches these through its host root.
pub const mlx = @import("mlx/ffi.zig");
pub const io_util = @import("sdk/util/io_util.zig");
pub const log = @import("sdk/util/log.zig");
pub const env = @import("env.zig");
comptime {
    _ = @import("plugins/registry_test_media.zig");
    _ = @import("engines/mlx/api.zig");
    @import("plugin_host_check.zig").reexports(@This());
}

/// The test process caps MLX's buffer pool as the server's `main` does; `build/test_runner.zig` calls it first.
export fn msv_test_process_init() void {
    @import("mlx/state.zig").applyMlxCacheLimit();
}

test {
    _ = @import("plugins/stable_audio3/plugin.zig");
    _ = @import("host/media_jobs/plugin/flip_test.zig");
    _ = @import("host/media_jobs/plugin/flip_image_test.zig");
    _ = @import("plugins/nemotron_diarization/diarization.zig");
    _ = @import("plugins/kokoro/kokoro.zig");
    _ = @import("plugins/laya/laya.zig");
    _ = @import("plugins/kev/kev.zig");
    _ = @import("plugins/d1/d1.zig");
    _ = @import("plugins/kokoro/g2p.zig");
    _ = @import("sdk/media/wav.zig");
    _ = @import("sdk/media/bill.zig");
    _ = @import("sdk/media/video_out.zig");
    _ = @import("sdk/media/image_pixels.zig");
    _ = @import("sdk/media/config_fields.zig");
    _ = @import("sdk/media/pack_bytes.zig");
    _ = @import("sdk/media/lora_set.zig");
    _ = @import("sdk/media/istft.zig");
    _ = @import("sdk/media/number_words.zig");
    _ = @import("sdk/media/job.zig");
    _ = @import("sdk/media/image_request.zig");
    _ = @import("sdk/media/mesh_request.zig");
    _ = @import("sdk/media/pack_claims.zig");
    _ = @import("sdk/media/graph_call.zig");
    _ = @import("plugins/qwen3_tts/tts.zig");
    _ = @import("plugins/flux2/flux.zig");
    _ = @import("plugins/krea/krea.zig");
    _ = @import("plugins/mage_flow/mage_flow.zig");
    _ = @import("plugins/qwen_image/qwen_image.zig");
    _ = @import("plugins/qwen_image/edit.zig");
    _ = @import("plugins/moss_image/moss_image.zig");
    _ = @import("plugins/ltx/video.zig");
    _ = @import("plugins/ltx/diffvae.zig");
    _ = @import("plugins/ltx/diffvae_kernel.zig");
    _ = @import("plugins/ltx/diffvae_forward.zig");
    _ = @import("plugins/ltx/audio.zig");
    _ = @import("plugins/minimax_h3/h3.zig");
    _ = @import("plugins/minimax_h3/vsa.zig");
    _ = @import("plugins/minimax_h3/cluster.zig");
    _ = @import("plugins/minimax_h3/vision.zig");
    _ = @import("plugins/minimax_h3/vae.zig");
    _ = @import("plugins/minimax_h3/audio.zig");
    _ = @import("sdk/media/png.zig");
    _ = @import("sdk/media/jpeg.zig");
    _ = @import("sdk/media/preview.zig");
    _ = @import("sdk/mesh/marching_cubes.zig");
    _ = @import("sdk/mesh/glb.zig");
    _ = @import("plugins/hunyuan3d/hunyuan3d.zig");
    _ = @import("plugins/acestep/acestep.zig");
    _ = @import("plugins/music3/music3.zig");
    _ = @import("plugins/stable_audio3/stable_audio.zig");
    _ = @import("plugins/yue2/yue2.zig");
    _ = @import("sdk/bases/dit.zig");
    _ = @import("plugins/trellis2/sparse.zig");
    _ = @import("plugins/trellis2/trellis2.zig");
    _ = @import("sdk/vision/dinov3.zig");
    _ = @import("plugins/trellis2/mesh.zig");
    _ = @import("plugins/trellis2/remesh.zig");
    _ = @import("plugins/trellis2/pipeline.zig");
    _ = @import("plugins/pixal3d/naf.zig");
    _ = @import("plugins/pixal3d/pixal3d.zig");
    _ = @import("sdk/mesh/uvwrap.zig");
    _ = @import("sdk/mesh/mesh_simplify.zig");
    _ = @import("plugins/hunyuan3d/paint.zig");
    _ = @import("plugins/hunyuan3d/paint_unet.zig");
    _ = @import("sdk/mesh/rasterize.zig");
    _ = @import("sdk/mesh/texinpaint.zig");
    _ = @import("sdk/mesh/bake.zig");
}
