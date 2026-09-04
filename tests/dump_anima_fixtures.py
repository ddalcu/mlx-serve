"""Self-contained reference for Anima's LLMAdapter (comfy/ldm/anima/model.py).

The adapter uses only standard ops (nn.Linear/RMSNorm/LayerNorm/Embedding, plain
F.scaled_dot_product_attention, visible rotate-half RoPE) — no ComfyUI fused
kernels — so this faithful copy needs no ComfyUI install. It loads the real
Anima weights and dumps an fp32 CPU fixture the Zig port is checked against.

Run: venv/bin/python adapter_ref.py <dit.safetensors> <out_dir>
"""
import sys, json, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

PREFIX = "model.diffusion_model.llm_adapter."


def save_safetensors_np(path, tensors):
    """Minimal safetensors writer (F32 / I32) — avoids the safetensors lib's
    numpy-version incompatibility. Format: u64 header len + JSON header + data."""
    _DT = {"float32": "F32", "int32": "I32", "int64": "I64"}
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        header[name] = {"dtype": _DT[str(arr.dtype)], "shape": list(arr.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    hj = json.dumps(header).encode("utf-8")
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(blob)


class ops:
    Linear = nn.Linear
    RMSNorm = nn.RMSNorm
    LayerNorm = nn.LayerNorm
    Embedding = nn.Embedding


# ── verbatim from comfy/ldm/anima/model.py ────────────────────────────────
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.rope_theta = 10000
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner = head_dim * n_heads
        self.n_heads, self.head_dim = n_heads, head_dim
        self.q_proj = ops.Linear(query_dim, inner, bias=False)
        self.q_norm = ops.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = ops.Linear(context_dim, inner, bias=False)
        self.k_norm = ops.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = ops.Linear(context_dim, inner, bias=False)
        self.o_proj = ops.Linear(inner, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, pe=None, pe_ctx=None):
        context = x if context is None else context
        ish = x.shape[:-1]
        csh = context.shape[:-1]
        q = self.q_norm(self.q_proj(x).view(*ish, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).view(*csh, self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(context).view(*csh, self.n_heads, self.head_dim).transpose(1, 2)
        if pe is not None:
            q = apply_rotary_pos_emb(q, *pe)
            k = apply_rotary_pos_emb(k, *pe_ctx)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        o = o.transpose(1, 2).reshape(*ish, -1).contiguous()
        return self.o_proj(o)


class TransformerBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16):
        super().__init__()
        self.norm_self_attn = ops.RMSNorm(model_dim, eps=1e-6)
        self.self_attn = Attention(model_dim, model_dim, num_heads, model_dim // num_heads)
        self.norm_cross_attn = ops.RMSNorm(model_dim, eps=1e-6)
        self.cross_attn = Attention(model_dim, source_dim, num_heads, model_dim // num_heads)
        self.norm_mlp = ops.RMSNorm(model_dim, eps=1e-6)
        self.mlp = nn.Sequential(ops.Linear(model_dim, int(model_dim * 4.0)), nn.GELU(),
                                 ops.Linear(int(model_dim * 4.0), model_dim))

    def forward(self, x, context, tmask=None, smask=None, pe=None, pe_ctx=None):
        x = x + self.self_attn(self.norm_self_attn(x), mask=tmask, pe=pe, pe_ctx=pe)
        x = x + self.cross_attn(self.norm_cross_attn(x), mask=smask, context=context, pe=pe, pe_ctx=pe_ctx)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6, num_heads=16):
        super().__init__()
        self.embed = ops.Embedding(32128, target_dim)
        self.in_proj = nn.Identity()
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([TransformerBlock(source_dim, model_dim, num_heads) for _ in range(num_layers)])
        self.out_proj = ops.Linear(model_dim, target_dim)
        self.norm = ops.RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids):
        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids))
        pos = torch.arange(x.shape[1]).unsqueeze(0)
        pos_ctx = torch.arange(context.shape[1]).unsqueeze(0)
        pe = self.rotary_emb(x, pos)
        pe_ctx = self.rotary_emb(x, pos_ctx)
        for b in self.blocks:
            x = b(x, context, pe=pe, pe_ctx=pe_ctx)
        return self.norm(self.out_proj(x))


def main():
    dit_path, out_dir = sys.argv[1], sys.argv[2]
    torch.manual_seed(0)
    model = LLMAdapter().to(torch.float32).eval()
    # Load the real adapter weights (strip prefix, fp32).
    sd = {}
    with safe_open(dit_path, framework="pt") as f:
        for k in f.keys():
            if k.startswith(PREFIX):
                sd[k[len(PREFIX):]] = f.get_tensor(k).to(torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected: {unexpected[:5]}"
    # buffers (inv_freq) are the only allowed missing.
    assert all("inv_freq" in m for m in missing), f"missing: {missing[:5]}"
    print(f"loaded {len(sd)} adapter tensors; missing(buffers)={len(missing)}")

    L, M = 7, 5  # source (qwen) len, target (t5) len — small for a tight fixture
    qwen_hidden = torch.randn(1, L, 1024, dtype=torch.float32)
    t5_ids = torch.tensor([[3, 100, 2011, 55, 1]], dtype=torch.long)
    with torch.no_grad():
        out = model(qwen_hidden, t5_ids)  # [1, M, 1024]
    print("out shape", tuple(out.shape), "mean", float(out.mean()), "std", float(out.std()))

    np.save(f"{out_dir}/adapter_qwen_hidden.npy", qwen_hidden.numpy())
    np.save(f"{out_dir}/adapter_t5_ids.npy", t5_ids.numpy().astype(np.int32))
    np.save(f"{out_dir}/adapter_out.npy", out.numpy())
    # safetensors fixture the Zig parity test loads (existing loadSafetensorsFile).
    save_safetensors_np(f"{out_dir}/anima_adapter_fixture.safetensors", {
        "qwen_hidden": qwen_hidden.numpy().astype(np.float32),
        "t5_ids": t5_ids.numpy().astype(np.int32),
        "adapter_out": out.numpy().astype(np.float32),
    })
    with open(f"{out_dir}/adapter_manifest.json", "w") as f:
        json.dump({"L": L, "M": M, "hidden": 1024,
                   "out_mean": float(out.mean()), "out_std": float(out.std()),
                   "out_first8": out.flatten()[:8].tolist()}, f, indent=2)
    print("wrote fixtures to", out_dir)


if __name__ == "__main__":
    main()
