# TODO

* Upscale Images & Video using SeedVR2 One-step diffusion DiT + 3D-causal VAE
* Built in Transcriber
* Prompt to Lyrics helper
* Expand Chat Tools to be able to generate media

* Remaining levers: understand the msv_attn_p256 µbench-vs-live gap (kernel wins isolated at kL≤2·qL, loses in the live graph — custom-kernel fusion boundary? then flip fused256CausalMode on); qmm approach re-derived; gate_up/gdn_tail fusions target decode; prefillEvalCadence per-layer sync above 2 GiB scores costs ~1-2% at 32K+ on the 27B (kept for small-RAM peak) 
* Stretch: DSpark/EAGLE3-class trained drafter (ARahim3/mlx-dspark) — needs a per-target trained 5-layer backbone (none exists for Qwen3.6-27B); 