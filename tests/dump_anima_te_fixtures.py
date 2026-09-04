"""Self-contained reference for the Anima Qwen3-0.6B text encoder.

Standard causal Qwen3 (comfy Qwen3_06BConfig): 28 layers, hidden 1024, 16 q-heads
/ 8 kv-heads, head_dim 128, per-head RMS qk-norm, SwiGLU, rope theta 1e6, eps 1e-6,
plain (unscaled) embedding, final model.norm applied. Output = last hidden states.

Run: venv/bin/python te_ref.py <qwen_3_06b_base.safetensors> <out_dir>
"""
import sys, json, struct
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

EPS = 1e-6
H, KV, HD, THETA, LAYERS = 16, 8, 128, 1000000.0, 28


def save_st(path, tensors):
    _DT = {"float32": "F32", "int32": "I32"}
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        header[name] = {"dtype": _DT[str(arr.dtype)], "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blob += b; off += len(b)
    hj = json.dumps(header).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj); f.write(blob)


def rmsnorm(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope_tables(L):
    inv = 1.0 / (THETA ** (torch.arange(0, HD, 2).float() / HD))
    freqs = torch.outer(torch.arange(L).float(), inv)  # [L, HD/2]
    emb = torch.cat([freqs, freqs], dim=-1)            # [L, HD]
    return emb.cos(), emb.sin()


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin):  # x [1,heads,L,HD], cos/sin [L,HD]
    return x * cos + rotate_half(x) * sin


def main():
    te_path, out_dir = sys.argv[1], sys.argv[2]
    sd = {}
    with safe_open(te_path, framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k).to(torch.float32)

    torch.manual_seed(0)
    ids = torch.tensor([[3, 100, 2011, 55, 1, 42, 9001]], dtype=torch.long)  # L=7
    L = ids.shape[1]
    x = F.embedding(ids, sd["model.embed_tokens.weight"])  # [1,L,1024]
    cos, sin = rope_tables(L)
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        h = rmsnorm(x, sd[p + "input_layernorm.weight"])
        q = (h @ sd[p + "self_attn.q_proj.weight"].T).view(1, L, H, HD)
        k = (h @ sd[p + "self_attn.k_proj.weight"].T).view(1, L, KV, HD)
        v = (h @ sd[p + "self_attn.v_proj.weight"].T).view(1, L, KV, HD)
        q = rmsnorm(q, sd[p + "self_attn.q_norm.weight"]).transpose(1, 2)  # [1,H,L,HD]
        k = rmsnorm(k, sd[p + "self_attn.k_norm.weight"]).transpose(1, 2)  # [1,KV,L,HD]
        v = v.transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(H // KV, dim=1)
        v = v.repeat_interleave(H // KV, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [1,H,L,HD]
        o = o.transpose(1, 2).reshape(1, L, H * HD)
        o = o @ sd[p + "self_attn.o_proj.weight"].T
        x = x + o
        h = rmsnorm(x, sd[p + "post_attention_layernorm.weight"])
        g = h @ sd[p + "mlp.gate_proj.weight"].T
        u = h @ sd[p + "mlp.up_proj.weight"].T
        m = (F.silu(g) * u) @ sd[p + "mlp.down_proj.weight"].T
        x = x + m
    x = rmsnorm(x, sd["model.norm.weight"])
    print("hidden", tuple(x.shape), "mean", float(x.mean()), "std", float(x.std()))
    save_st(f"{out_dir}/anima_te_fixture.safetensors", {
        "ids": ids.numpy().astype(np.int32),
        "hidden": x.numpy().astype(np.float32),
    })
    print("wrote", f"{out_dir}/anima_te_fixture.safetensors")


if __name__ == "__main__":
    main()
