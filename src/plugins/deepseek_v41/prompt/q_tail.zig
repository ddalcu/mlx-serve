//! A wide prompt layer's query heads are not copied whole to rope their last channels.
//!
//! The stock path writes the chunk's roped heads `q` [L, 64, 512] bf16 (1.07 GB at 16K rows) from `wq_b`'s output
//! (another 1.07 GB): the rope kernel copies the 448 unroped channels of every head as they are and ropes the last 64.
//! Instead the wide layer keeps `wq_b`'s output as the unroped heads and ropes only the last 64 channels of each head into
//! `qt` [L, 64, 64] (kernels/rope_tail_only.metal: rope_tail.metal's arithmetic on the same pairs); the gather's score
//! kernel (kernels/attn_qk_t.metal) reads the unroped heads for its first seven 64-deep steps and `qt` for the last. The
//! scores' operands are the same bf16 words in the same steps on the same tensor op, so the same scores. A span the
//! gather declines (another shape, the f32 core) gets the whole roped heads as before. One fresh 0.94 GB buffer and its
//! copy pass fewer a layer.
const kernelSource = @import("../../../mlx/kernels.zig").source;

/// The kernels' text (the model builds them at load).
pub const rope_tail_only_source = kernelSource("plugins/deepseek_v41/kernels/rope_tail_only.metal");
pub const attn_qk_t_source = kernelSource("plugins/deepseek_v41/kernels/attn_qk_t.metal");
