//! The plugin kit's one Zig module (`mlx_serve`): the ABI tables and the MLX-free SDK an outside plugin builds
//! against. Its import closure is the kit's source (`zig build kit` copies exactly that and refuses one that reaches
//! the engine or the host); MLX itself reaches a plugin as the kit's `mlx_c` module (mlx-c's headers).

pub const abi = struct {
    pub const common = @import("abi/common.zig");
    pub const plugin = @import("abi/plugin.zig");
    pub const media = @import("abi/media.zig");
    pub const quant = @import("abi/quant.zig");
    pub const host = @import("abi/host.zig");
    pub const dylib = @import("abi/dylib.zig");
};

/// `sdk.Plugin`, `claimsByType`, `fixture`, `tensors`, `byte_tokenizer`, as built-in plugins see them.
pub const sdk = @import("sdk/root.zig");
/// The `Host` table's plugin-side accessor (`Host.of(ctx.host)`).
pub const host = @import("sdk/host.zig");
/// `exportManifest`: what the kit's build calls on the plugin root.
pub const dylib = @import("sdk/dylib.zig");
/// `generation_config.json` into `Shell.gen`.
pub const generation = @import("sdk/config/generation.zig");
/// WAV encoding for audio backends.
pub const wav = @import("sdk/media/wav.zig");

/// The MLX bindings `blocks` are written on: their `mlx_array` and `mlx_stream` are `{ ctx }` like mlx-c's, so a
/// plugin converts at the call (`.{ .ctx = x.ctx }`).
pub const mlx = @import("mlx/ffi.zig");

/// Register a quant layout of your own (`foreign.register`) so the blocks' `qmatmul` and expert matmul read it.
pub const foreign = @import("sdk/quant/foreign.zig");

/// Layer blocks the server's own plugins are built from, for a plugin that would rather call them than write the layer
/// on mlx-c. They are versioned with the kit: build against one kit version and pin it, a newer kit may rename or
/// change them. They run on the same MLX the server loads (the plugin's mlx-c calls resolve against it at load).
pub const blocks = struct {
    pub const norms = struct {
        pub const add_norm = @import("sdk/blocks/norms/add_norm.zig");
        pub const group_norm = @import("sdk/blocks/norms/group_norm.zig");
        pub const grouped_rms = @import("sdk/blocks/norms/grouped_rms.zig");
        pub const layer_norm_f32 = @import("sdk/blocks/norms/layer_norm_f32.zig");
        pub const residual_norm = @import("sdk/blocks/norms/residual_norm.zig");
    };
    pub const conv = struct {
        pub const depthwise = @import("sdk/blocks/conv/depthwise.zig");
        pub const gated_conv_step = @import("sdk/blocks/conv/gated_conv_step.zig");
    };
    pub const ssm = struct {
        pub const mamba2_decode = @import("sdk/blocks/ssm/mamba2_decode.zig");
    };
    pub const schedulers = struct {
        pub const flow_match = @import("sdk/blocks/schedulers/flow_match.zig");
    };
    pub const vae = struct {
        pub const conv = @import("sdk/blocks/vae/conv.zig");
        pub const resnet = @import("sdk/blocks/vae/resnet.zig");
        pub const snake = @import("sdk/blocks/vae/snake.zig");
    };
    pub const dit = struct {
        pub const modulate = @import("sdk/blocks/dit/modulate.zig");
        pub const timestep = @import("sdk/blocks/dit/timestep.zig");
        pub const rope = @import("sdk/blocks/dit/rope.zig");
        pub const linear = @import("sdk/blocks/dit/linear.zig");
        pub const dit_kernels = @import("sdk/blocks/dit/dit_kernels.zig");
        pub const int8_attn = @import("sdk/blocks/dit/int8_attn.zig");
        pub const w8a8 = @import("sdk/blocks/dit/w8a8.zig");
        pub const w8a8_bill = @import("sdk/blocks/dit/w8a8_bill.zig");
    };
    pub const embedding = @import("sdk/blocks/embedding.zig");
    pub const linear = struct {
        pub const fused_rows = @import("sdk/blocks/linear/fused_rows.zig");
        pub const prefill_dq_gemm = @import("sdk/blocks/linear/prefill_dq_gemm.zig");
        pub const qmatmul = @import("sdk/blocks/linear/qmatmul.zig");
        pub const qmv_rows = @import("sdk/blocks/linear/qmv_rows.zig");
        pub const verify_projection = @import("sdk/blocks/linear/verify_projection.zig");
        pub const verify_qmm = @import("sdk/blocks/linear/verify_qmm.zig");
        pub const verify_qmm_nax = @import("sdk/blocks/linear/verify_qmm_nax.zig");
    };
    pub const qmm = struct {
        pub const lane_qmm = @import("sdk/blocks/qmm/lane_qmm.zig");
        pub const mxfp8_qmv = @import("sdk/blocks/qmm/mxfp8_qmv.zig");
        pub const qmm_int8 = @import("sdk/blocks/qmm/qmm_int8.zig");
        pub const qmv2 = @import("sdk/blocks/qmm/qmv2.zig");
        pub const qmv_nax2 = @import("sdk/blocks/qmm/qmv_nax2.zig");
        pub const rowqmv = @import("sdk/blocks/qmm/rowqmv.zig");
        pub const simd_qmm = @import("sdk/blocks/qmm/simd_qmm.zig");
    };
    pub const mlp = struct {
        pub const activations = @import("sdk/blocks/mlp/activations.zig");
        pub const dense = @import("sdk/blocks/mlp/dense.zig");
        pub const fused_activations = @import("sdk/blocks/mlp/fused_activations.zig");
        pub const gelu = @import("sdk/blocks/mlp/gelu.zig");
    };
    pub const rope = struct {
        pub const mrope = @import("sdk/blocks/rope/mrope.zig");
        pub const qk_norm_rope = @import("sdk/blocks/rope/qk_norm_rope.zig");
        pub const rope = @import("sdk/blocks/rope/rope.zig");
        pub const scaled = @import("sdk/blocks/rope/scaled.zig");
        pub const yarn = @import("sdk/blocks/rope/yarn.zig");
    };
    pub const attention = struct {
        pub const attn256 = @import("sdk/blocks/attention/attn256.zig");
        pub const batched = @import("sdk/blocks/attention/batched.zig");
        pub const dec_attn = @import("sdk/blocks/attention/dec_attn.zig");
        pub const decode_attn_quant = @import("sdk/blocks/attention/decode_attn_quant.zig");
        pub const gates = @import("sdk/blocks/attention/gates.zig");
        pub const kv_quant = @import("sdk/blocks/attention/kv_quant.zig");
        pub const kv_quant_attn = @import("sdk/blocks/attention/kv_quant_attn.zig");
        pub const kv_view = @import("sdk/blocks/attention/kv_view.zig");
        pub const lane_attn = @import("sdk/blocks/attention/lane_attn.zig");
        pub const masks = @import("sdk/blocks/attention/masks.zig");
        pub const nax_attention = @import("sdk/blocks/attention/nax_attention.zig");
        pub const row_attn = @import("sdk/blocks/attention/row_attn.zig");
        pub const sdpa = @import("sdk/blocks/attention/sdpa.zig");
    };
    pub const moe = struct {
        pub const decode = @import("sdk/blocks/moe/decode.zig");
        pub const down_reduce = @import("sdk/blocks/moe/down_reduce.zig");
        pub const experts = @import("sdk/blocks/moe/experts.zig");
        pub const gather_qmm_nax = @import("sdk/blocks/moe/gather_qmm_nax.zig");
        pub const gather_qmv = @import("sdk/blocks/moe/gather_qmv.zig");
        pub const gather_qmv_rows = @import("sdk/blocks/moe/gather_qmv_rows.zig");
        pub const moe_affine4 = @import("sdk/blocks/moe/moe_affine4.zig");
        pub const moe_fp4 = @import("sdk/blocks/moe/moe_fp4.zig");
        pub const router_kernel = @import("sdk/blocks/moe/router_kernel.zig");
        pub const routing = @import("sdk/blocks/moe/routing.zig");
        pub const verify_kernels = @import("sdk/blocks/moe/verify_kernels.zig");
        pub const weights = @import("sdk/blocks/moe/weights.zig");
    };
    pub const hc = struct {
        pub const hc_decode2 = @import("sdk/blocks/hc/hc_decode2.zig");
        pub const hc_prefill = @import("sdk/blocks/hc/hc_prefill.zig");
    };
    pub const gdn = struct {
        pub const decode = @import("sdk/blocks/gdn/decode.zig");
        pub const decode_fused = @import("sdk/blocks/gdn/decode_fused.zig");
        pub const gate = @import("sdk/blocks/gdn/gate.zig");
        pub const gated_delta_net = @import("sdk/blocks/gdn/gated_delta_net.zig");
        pub const kernels = @import("sdk/blocks/gdn/kernels.zig");
    };
};

test {
    _ = abi.dylib;
    _ = dylib;
}
