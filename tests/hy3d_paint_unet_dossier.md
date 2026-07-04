# Hunyuan3D-2.1 paint 2.5D UNet — implementation dossier (source-verified 2026-07-04)

Companion to `hy3d_paint_weights_contract.md`; every fact below read directly from
`reference/hy3dpaint/hunyuanpaintpbr/{unet/modules.py, unet/attn_processor.py, pipeline.py}`.
This is the P2-7/P2-8 spec for `src/hunyuan3d_paint.zig`.

## Base UNet (SD2.1, use_linear_projection=true)

Standard diffusers UNet2DConditionModel: conv_in (12ch main / 4ch dual) → time_embedding
(sinusoidal 320 → linear_1 → silu → linear_2 → 1280) → down_blocks [CrossAttnDown×3, Down]
(2 resnets + 2 Transformer2DModel each for cross blocks; downsample stride-2 conv after blocks 0-2)
→ mid (resnet, Transformer2DModel, resnet) → up_blocks [Up, CrossAttnUp×3] (3 resnets [+3 attns];
upsample nearest2x+conv after blocks 0-2; skip-concat with down residuals) → conv_norm_out(GN32)
→ silu → conv_out. Resnet: GN32(eps 1e-5) → silu → conv1 → (+ time_emb_proj(silu(temb)) broadcast)
→ GN32 → silu → conv2; conv_shortcut where channels change. Transformer2DModel wrapper:
GN32(norm, eps 1e-6 — diffusers uses 1e-6 here) → LINEAR proj_in ([B,HW,C]) → transformer_blocks
→ LINEAR proj_out → residual add. Timestep embedding: diffusers Timesteps(320, flip_sin_to_cos=True,
downscale_freq_shift=0) — cos-first order, half dim 160.

## Basic2p5DTransformerBlock.forward (modules.py:472-707) — EXACT order

Input hidden [(b·n_pbr·n), l, c]; kwargs: num_in_batch=n (views), mode ("w" ref-write / "r" gen-read),
mva_scale, ref_scale, condition_embed_dict, dino_hidden_states (already ImageProj'd), position_voxel_indices.

1. `norm_hidden = norm1(hidden)` (plain LayerNorm; norm_type is layer_norm)
2. **MDA** (use_mda): reshape [b, n_pbr, n, l, c]; per material m: flatten to [(b·n), l, c],
   self-attn with that material's projections (albedo → attn1.to_q/k/v/out; mr →
   attn1.to_q_mr/… flattened names). PER-VIEW attention (seq = l only). Concat over materials
   → residual add to hidden.
3. **RA write** (mode contains "w" — the DUAL UNet pass): condition_embed_dict[layer_name] =
   rearrange(norm_hidden, "(b n) l c -> b (n l) c"). (Dual pass has no MDA/RA/MA/DINO modules,
   so for it steps 2,4,5,7 use the plain base path — its blocks are plain BasicTransformerBlock.)
4. **RA read** (mode contains "r", use_ra): query = ALBEDO slice of norm_hidden rearranged
   [b, n_pbr, (n·l), c][:,0]; enc = condition_embed_dict[layer_name] [b, l_ref·n_ref, c].
   RefAttnProcessor: q = to_q(query), k = to_k(enc), v = CONCAT([to_v(enc), to_v_mr(enc)], -1);
   heads reshape value with head_dim·n_pbr per head; SDPA (scale head_dim^-0.5); split output
   along last dim per material; per-material to_out / to_out_mr; outputs stacked [b, n_pbr, (n l), c]
   → reshape [(b n_pbr n), l, c]; hidden += ref_scale · out. (ref_scale per CFG batch: 0 uncond / 1 cond.)
5. **MA** (num_in_batch>1, use_ma): seq = rearrange(norm_hidden, "(b n_pbr n) l c -> (b n_pbr) (n l) c").
   attn_multiview self-attn with 3D RoPE on q,k (PoseRoPE). hidden += mva_scale · out (mva_scale 1.0).
6. `norm_hidden = norm2(hidden)`; **attn2** cross to the 77 learned tokens (per material stream);
   hidden += out.
7. **DINO** (use_dino): SAME norm_hidden as step 6; attn_dino cross to dino tokens [*, 1028, 1024]
   (repeated per (n_pbr·n) stream); hidden += out. NOTE zero-DINO batches still add to_out.bias —
   fine, cancels in CFG (both A and B batches compute it identically... NO: A uses zeros, B real;
   bias adds to BOTH A and B equally → cancels in (B−A), stays in A's absolute value exactly as
   the reference's batch-0. Reproduce reference: batch A = dino zeros (bias still added), batch B = real.)
8. `norm_hidden = norm3(hidden)`; **FF** GeGLU: proj → chunk2 (value first, gate second) →
   v·gelu(g) → net.2 linear; hidden += out.

## PoseRoPE (attn_processor.py:367-463)

- Per axis tables: dim_xy = head_dim//8·3 (=24 for head_dim 64), dim_z = head_dim//8·2 (=16).
  freqs = 1/theta^(arange(0,dim,2)/dim) (f32), theta 10000; freqs = outer(grid, freqs);
  cos/sin = repeat_interleave(2, dim=1) → [res, dim].
- Voxel indices [B, L, 3] index the tables per axis; cat x,y,z → [B, L, head_dim] cos/sin.
- apply: x pairs (even,odd): out = x·cos + rotate(x)·sin where rotate = (-x1, x0) interleaved
  (reshape [..., d/2, 2], unbind, stack(-x_imag, x_real), flatten). Computed in f32, cast back to f16.
- Cache: keyed by seq-len (n·l) in position_voxel_indices dict {seq_len: {voxel_indices, voxel_resolution}},
  and per head_dim after first computation. head_dim is 64 everywhere → one table set per level.

## Voxel indices (modules.py:196-274)

From position_maps [B, N, 3, H, W] (the RENDERED position images, values [0,1], bg exactly 1):
for (grid_res, voxel_res) in zip([H,H/2,H/4,H/8],[8H,4H,2H,H]) — with H=64 latent: grids [64,32,16,8],
voxels [512,256,128,64]:
  valid = (position != 1).all(channel); invalid → 0
  average positions over each (H/grid)² cell (sum/count, count clamp min 1)
  cells with count < cellsize²/16 → 0
  clamp [0,1] · (voxel_res−1), round → int indices [B, N, 3, g, g] → rearrange [B, (N·g·g), 3]
  dict key = N·g·g (seq len).

## Dual-UNet ref pass (modules.py:1011-1065)

Once per generation (cached): ref_latents [B·3(CFG), 4, 64, 64] (identical repeat 3× — so ONE
unique forward needed), timestep 0, encoder_hidden_states = learned_text_clip_ref [77,1024],
mode "w", num_in_batch = N_ref = 1. The dual UNet is plain SD2.1 (no 2.5D modules); its blocks
write condition_embed_dict[layer_name] = norm1-output rearranged. 16 entries (one per transformer block).

## Main forward assembly (modules.py:959-1102)

sample [B, N_pbr, N_gen, 4, 64, 64] concat embeds_normal + embeds_position (each [B(·3), N_gen, 4, 64, 64],
unsqueeze/repeat over N_pbr) along channel → 12ch → flatten [(B·N_pbr·N_gen), 12, 64, 64].
encoder_hidden_states = stack([learned_albedo, learned_mr]) [B, N_pbr, 77, 1024] repeated per view.
timestep scalar t for all streams. class_labels None. added_cond None.

## CFG (pipeline.py:590-697 + 298-327)

Triple batch [uncond, ref, full]: ref_scale = [0,1,1]; dino = [zeros, zeros, real]; latents/normal/
position/prompt embeds identical across batches. camera_azims NEVER passed → view_scale = 1.0 →
noise = uncond + g·(full − uncond) algebraically (middle cancels). PORT AS 2-BATCH:
A = {ref_scale 0, dino zeros}, B = {ref_scale 1, dino real}; v = A + g·(B − A). guidance_rescale 0.
Guidance 3.0, steps 30 DDIM (runtime reference uses UniPC-15; we dump fixtures with DDIM).
Scheduler steps on latents[:, :4] (12ch input, 4ch output/state).

## Oracle 4/5 input contract (dump script `--with-unet`, must match Zig oracle tests)

Seeded CPU randn: ref_latents g(1)·0.18215 [1,4,64,64]; latents g(2) [12,4,64,64];
embeds_normal g(3)·0.18215 [6,4,64,64]; embeds_position g(4)·0.18215 [6,4,64,64]; t=999;
dino tokens = oracle-3 fixture (real proj output); position_maps: SKIP RoPE cache → feed
position_voxel_indices=None? NO — PoseRoPE must be exercised: derive position_maps from the
synth image trick instead: use a deterministic synthetic position map [1,6,3,512,512] built like
synth_image_chw (values [0,1], bg=1). Dump: canary condition_embed_dict entries for layers
down_0_0_0 / mid_0_0 / up_3_2_0 (oracle 4, cos>0.995) + final v [12,4,64,64] (oracle 5, cos>0.995).
Single forward, no CFG (ref_scale=1, real dino) — CFG algebra is pinned hermetically.

## raForward war-story (RA value layout — ADJUDICATED by oracle 5, 2026-07-04)

Three candidate readings of RefAttnProcessor2_0's value handling existed during the port:
(a) channel-concat [to_v|to_v_mr] → straight head view → merge heads → slice merged [:C]/[C:];
(b) per-head interleave value [to_v_h|to_v_mr_h] → per-head output split;
(c) channel-concat → straight head view (head h = concat cols [128h,128h+128)) → per-head
output split(64, dim=-1) BEFORE the head merge.

**(c) is the reference** — `value.view(B,L,H,2C/H).transpose` after a dim=-1 cat, then
`torch.split(sdpa_out, head_dim=64, dim=-1)` on the [B,H,L,128] tensor. Consequences worth
staring at: with an ODD head count the middle value-head straddles the material boundary, and
the "albedo" output stream mixes to_v even-half-heads with to_v_mr slices. It looks like a bug;
it is what training saw (to_out/to_out_mr were trained against exactly this scramble). (a) fails
the hermetic 2-head split test; (b) pairs different value columns with each attention head and
was REJECTED empirically. Final: oracle 4 (ref cache) 0.99999/1.00000/0.99998, oracle 5 (full
2.5D step) **1.00000** against fp32-CPU fixtures.

Fixture-side lesson (same day, twice): fp16-MPS dumps cost real cos — the 24,576-token MA
softmax alone read as a 0.9865 "failure" of a correct implementation. Dump UNet/DINO oracle
fixtures fp32-CPU (inputs still fp16-quantized so both sides start identical); see the
CLAUDE.md gotcha "Parity fixtures for fp16-fragile giants".
