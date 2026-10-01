#!/usr/bin/env python3
"""Reference oracle for the MiMo-ViT tower (src/mimo_vision.zig).

Runs the checkpoint's OWN `MiMoVisionTransformer` (modeling_mimo_v2.py) on the
CPU in f32 and writes what the Zig tests compare against.

  tiny OUT.safetensors --ref DIR
      A 4-block random tower (full, row band, column band, full; sinks on the
      band blocks) on an 8x12 patch grid. One file holds the weights under the
      checkpoint's `visual.*` names and the fixture under `fixture.*`:
      pixel_values, grid_thw, features, and features_sink_column: the same
      tower with the sink as an extra softmax column (SGLang's reading) instead
      of a bias on key 0's logit (the reference's and vLLM's), so the Zig test
      can prove which one it implements. Committed as
      src/fixtures/mimo_vision_tiny.safetensors.

  real OUT.safetensors --ref DIR
      The real tower (`visual.*` read from DIR's shards) on a synthetic
      450x650 RGB image: rgb, the vendor-preprocessed pixel_values, grid_thw,
      the patch embedding, a few block outputs, and features.

DIR is the checkpoint directory (config.json, modeling_mimo_v2.py,
configuration_mimo_v2.py and, for `real`, the shards). Needs torch,
transformers, numpy and safetensors; exit 2 = SKIP when they are missing.
"""
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as e:  # pragma: no cover
    sys.stderr.write(f"SKIP: {e}\n")
    raise SystemExit(2)

PIXEL_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)
PIXEL_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1)

TINY = dict(
    depth=4, hidden_size=32, num_heads=4, num_key_value_heads=2, qk_channels=16,
    intermediate_size=48, out_hidden_size=64, patch_size=4, spatial_merge_size=2,
    temporal_patch_size=2, in_chans=3, hidden_act="silu", use_sink=True,
    fullatt_block_indexes=[0, 3], vit_window_attn_types=[-1, 0, 1, -1],
    visual_token_window_size=4,
)
TINY_GRID = (1, 8, 12)


def import_reference(ref_dir):
    """The checkpoint's modeling module, imported as a package (it uses a relative import)."""
    pkg = Path(tempfile.mkdtemp()) / "mimo_v2_ref"
    pkg.mkdir()
    (pkg / "__init__.py").touch()
    for f in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
        shutil.copyfile(Path(ref_dir) / f, pkg / f)
    sys.path.insert(0, str(pkg.parent))
    import mimo_v2_ref.modeling_mimo_v2 as mm
    return mm


def smart_resize(height, width, factor, min_pixels, max_pixels):
    if min(height, width) < factor:
        scale = factor / min(height, width)
        height, width = int(round(height * scale)), int(round(width * scale))
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return int(h_bar), int(w_bar)


def vendor_pixel_values(rgb, patch, merge, tps, min_pixels, max_pixels):
    """The SGLang/vLLM image path: smart_resize, bilinear, mean/std, flatten."""
    h, w, _ = rgb.shape
    rh, rw = smart_resize(h, w, patch * merge, min_pixels, max_pixels)
    img = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
    img = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
    img = ((img - PIXEL_MEAN) / PIXEL_STD).squeeze(0)
    frames = img.unsqueeze(0).repeat(tps, 1, 1, 1)
    gh, gw = rh // patch, rw // patch
    p = frames.view(1, tps, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    p = p.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(gh * gw, 3 * tps * patch * patch)
    return p.contiguous(), torch.tensor([[1, gh, gw]], dtype=torch.int32)


def sink_column_forward(mm):
    """MiMoVisionAttention.forward with the sink as an extra softmax column."""

    def forward(self, hidden_states, cu_seqlens, position_embeddings, full_attn=False):
        seq_len = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q = qkv[:, :q_dim].view(seq_len, self.num_heads, self.head_dim)
        k = qkv[:, q_dim:q_dim + kv_dim].view(seq_len, self.num_kv_heads, self.head_dim)
        v = qkv[:, q_dim + kv_dim:].view(seq_len, self.num_kv_heads, self.head_dim)
        q, k = mm._apply_rotary_pos_emb_vision(q, k, *position_embeddings)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * self.scaling
        if not full_attn:
            mask = self._build_window_mask(seq_len, q.device, q.dtype)
            if mask is not None:
                scores = scores + mask
        if self.sinks is not None:
            col = self.sinks.view(-1, 1, 1).expand(-1, seq_len, 1).to(scores.dtype)
            probs = torch.softmax(torch.cat([scores, col], dim=-1), dim=-1)[..., :-1]
        else:
            probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hqk,khd->qhd", probs, v).reshape(seq_len, -1)
        return self.proj(out)

    return forward


def tiny(out, ref_dir):
    mm = import_reference(ref_dir)
    torch.manual_seed(20260924)
    tower = mm.MiMoVisionTransformer(mm._as_namespace(dict(TINY))).float().eval()
    with torch.no_grad():
        for name, p in tower.named_parameters():
            p.copy_(torch.randn_like(p) * (0.5 if name.endswith("sinks") else 0.2))
            if name.endswith("sinks"):
                p.add_(4.0)
        # The checkpoint ships the merger without biases: they are zero.
        for b in (tower.merger.ln_q.bias, tower.merger.mlp[0].bias, tower.merger.mlp[2].bias):
            b.zero_()
        # The file stores bf16; the reference runs on exactly those values.
        for p in tower.parameters():
            p.copy_(p.to(torch.bfloat16).float())
    grid = torch.tensor([TINY_GRID], dtype=torch.int32)
    n = TINY_GRID[1] * TINY_GRID[2]
    feat = 3 * TINY["temporal_patch_size"] * TINY["patch_size"] ** 2
    pixel_values = torch.randn(n, feat)
    with torch.no_grad():
        features = tower(pixel_values, grid)
        keep = mm.MiMoVisionAttention.forward
        mm.MiMoVisionAttention.forward = sink_column_forward(mm)
        try:
            features_col = tower(pixel_values, grid)
        finally:
            mm.MiMoVisionAttention.forward = keep
    cos = F.cosine_similarity(features.flatten(), features_col.flatten(), dim=0).item()
    if cos > 0.99:
        raise SystemExit(f"the two sink readings agree (cos {cos:.5f}); the fixture cannot tell them apart")
    tensors = {}
    skip = ("merger.ln_q.bias", "merger.mlp.0.bias", "merger.mlp.2.bias")
    for name, p in tower.state_dict().items():
        if name not in skip:
            tensors[f"visual.{name}"] = p.detach().to(torch.bfloat16).contiguous()
    tensors["fixture.pixel_values"] = pixel_values.contiguous()
    tensors["fixture.grid_thw"] = grid
    tensors["fixture.features"] = features.contiguous()
    tensors["fixture.features_sink_column"] = features_col.contiguous()
    save_file(tensors, out, metadata={"mimo_vision_tiny": json.dumps(TINY)})


def synthetic_rgb(h, w):
    y, x = np.mgrid[0:h, 0:w]
    rgb = np.stack([
        (x * 255 // max(w - 1, 1)),
        (y * 255 // max(h - 1, 1)),
        ((x // 25 + y // 25) % 2) * 200 + 30,
    ], axis=-1).astype(np.uint8)
    rgb[(h // 3):(h // 3 + 40), (w // 4):(3 * w // 4)] = (250, 250, 250)
    rgb[(h // 3 + 10):(h // 3 + 30), (w // 4 + 20):(w // 2)] = (10, 10, 10)
    return rgb


def real(out, ref_dir):
    mm = import_reference(ref_dir)
    ref = Path(ref_dir)
    cfg = json.loads((ref / "config.json").read_text())
    vc = cfg["vision_config"]
    pc = cfg["processor_config"]
    tower = mm.MiMoVisionTransformer(mm._as_namespace(dict(vc))).float().eval()
    index = json.loads((ref / "model.safetensors.index.json").read_text())["weight_map"]
    state = {}
    for key, shard in index.items():
        if key.startswith("visual."):
            with safe_open(str(ref / shard), framework="pt") as f:
                state[key[len("visual."):]] = f.get_tensor(key).float()
    missing, unexpected = tower.load_state_dict(state, strict=False)
    if unexpected or sorted(missing) != ["merger.ln_q.bias", "merger.mlp.0.bias", "merger.mlp.2.bias"]:
        raise SystemExit(f"unexpected tower keys: missing={missing} unexpected={unexpected}")
    with torch.no_grad():
        for b in (tower.merger.ln_q.bias, tower.merger.mlp[0].bias, tower.merger.mlp[2].bias):
            b.zero_()

    rgb = synthetic_rgb(450, 650)
    pixel_values, grid = vendor_pixel_values(
        rgb, vc["patch_size"], vc["spatial_merge_size"], vc["temporal_patch_size"],
        pc["image_min_pixels"], min(pc["image_max_pixels"], 1536 * 1536))
    taps = {}
    hooks = [tower.patch_embed.register_forward_hook(lambda m, i, o: taps.__setitem__("patch_embed", o))]
    for idx in (0, 1, 5, 13, 27):
        hooks.append(tower.blocks[idx].register_forward_hook(
            lambda m, i, o, idx=idx: taps.__setitem__(f"block{idx}", o)))
    with torch.no_grad():
        features = tower(pixel_values, grid)
    for h in hooks:
        h.remove()
    tensors = {
        "rgb": torch.from_numpy(rgb).contiguous(),
        "pixel_values": pixel_values,
        "grid_thw": grid,
        "features": features.contiguous(),
    }
    tensors.update({k: v.contiguous() for k, v in taps.items()})
    save_file(tensors, out)


def main():
    if len(sys.argv) != 5 or sys.argv[1] not in ("tiny", "real") or sys.argv[3] != "--ref":
        sys.stderr.write(__doc__)
        raise SystemExit(1)
    (tiny if sys.argv[1] == "tiny" else real)(sys.argv[2], sys.argv[4])


if __name__ == "__main__":
    main()
