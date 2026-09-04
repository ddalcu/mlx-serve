"""Self-contained reference for the Anima/Cosmos-Predict2 MiniTrainDIT.

Faithful copy of comfy/ldm/cosmos/predict2.py + position_embedding.py, with the
only ComfyUI-native piece — the fused RMSNorm+RoPE `rms_rope_split_half` — taken
from the real `comfy_kitchen` package (verified on CPU). Everything else is
standard torch. Loads the real DiT weights and dumps an fp32 CPU fixture the Zig
port is checked against.

Run: venv/bin/python dit_ref.py <anima-*.safetensors> <out_dir>
"""
import sys, json, struct, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors import safe_open
import comfy_kitchen as ck

PREFIX = "model.diffusion_model."


def save_st(path, tensors):
    _DT = {"float32": "F32", "int32": "I32"}
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        header[name] = {"dtype": _DT[str(arr.dtype)], "shape": list(arr.shape),
                        "data_offsets": [off, off + len(b)]}
        blob += b; off += len(b)
    hj = json.dumps(header).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj); f.write(blob)


# ── position_embedding.py: VideoRopePosition3DEmb ─────────────────────────
class VideoRopePosition3DEmb(nn.Module):
    def __init__(self, head_dim, len_h, len_w, len_t, base_fps=24,
                 h_extrapolation_ratio=1.0, w_extrapolation_ratio=1.0,
                 t_extrapolation_ratio=1.0, enable_fps_modulation=True):
        super().__init__()
        self.base_fps = base_fps
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_t = dim - 2 * dim_h
        self.dim_spatial_range = torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h
        self.dim_temporal_range = torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t
        self.h_ntk = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk = w_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.t_ntk = t_extrapolation_ratio ** (dim_t / (dim_t - 2))

    def forward(self, B_T_H_W_C, fps=None):
        h_theta = 10000.0 * self.h_ntk
        w_theta = 10000.0 * self.w_ntk
        t_theta = 10000.0 * self.t_ntk
        h_freqs = 1.0 / (h_theta ** self.dim_spatial_range)
        w_freqs = 1.0 / (w_theta ** self.dim_spatial_range)
        t_freqs = 1.0 / (t_theta ** self.dim_temporal_range)
        B, T, H, W, _ = B_T_H_W_C
        seq = torch.arange(max(H, W, T)).float()
        half_h = torch.outer(seq[:H], h_freqs)
        half_w = torch.outer(seq[:W], w_freqs)
        if fps is None or self.enable_fps_modulation is False:
            half_t = torch.outer(seq[:T], t_freqs)
        else:
            half_t = torch.outer(seq[:T] / fps * self.base_fps, t_freqs)
        half_h = torch.stack([half_h.cos(), -half_h.sin(), half_h.sin(), half_h.cos()], dim=-1)
        half_w = torch.stack([half_w.cos(), -half_w.sin(), half_w.sin(), half_w.cos()], dim=-1)
        half_t = torch.stack([half_t.cos(), -half_t.sin(), half_t.sin(), half_t.cos()], dim=-1)
        em = torch.cat([
            repeat(half_t, "t d x -> t h w d x", h=H, w=W),
            repeat(half_h, "h d x -> t h w d x", t=T, w=W),
            repeat(half_w, "w d x -> t h w d x", t=T, h=H),
        ], dim=-2)
        return rearrange(em, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


class GPT2FeedForward(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.activation = nn.GELU()
        self.layer1 = nn.Linear(d, dff, bias=False)
        self.layer2 = nn.Linear(dff, d, bias=False)

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


def torch_attention_op(q, k, v):  # q,k,v [B,S,H,D]
    q = rearrange(q, "b s h d -> b h s d")
    k = rearrange(k, "b s h d -> b h s d")
    v = rearrange(v, "b s h d -> b h s d")
    o = F.scaled_dot_product_attention(q, k, v)
    return rearrange(o, "b h s d -> b s (h d)")


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        self.is_selfattn = context_dim is None
        context_dim = query_dim if context_dim is None else context_dim
        inner = head_dim * n_heads
        self.n_heads, self.head_dim = n_heads, head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.v_norm = nn.Identity()
        self.output_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(self, x, context=None, rope_emb=None):
        context = x if context is None else context
        q = self.q_proj(x); k = self.k_proj(context); v = self.v_proj(context)
        q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim), (q, k, v))
        v = self.v_norm(v)
        if self.is_selfattn and rope_emb is not None:
            q, k = ck.rms_rope_split_half(q, k, rope_emb, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps)
        else:
            q = self.q_norm(q); k = self.k_norm(k)
        o = torch_attention_op(q, k, v)
        return self.output_proj(o)


class Timesteps(nn.Module):
    def __init__(self, n):
        super().__init__(); self.n = n

    def forward(self, t_B_T):
        t = t_B_T.flatten().float()
        half = self.n // 2
        exp = -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        emb = t[:, None].float() * torch.exp(exp)[None, :]
        emb = torch.cat([emb.cos(), emb.sin()], dim=-1)
        return rearrange(emb, "(b t) d -> b t d", b=t_B_T.shape[0], t=t_B_T.shape[1])


class TimestepEmbedding(nn.Module):
    def __init__(self, inf, outf):
        super().__init__()
        self.linear_1 = nn.Linear(inf, outf, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(outf, 3 * outf, bias=False)

    def forward(self, sample):
        emb = self.linear_2(self.activation(self.linear_1(sample)))
        return sample, emb  # (emb_B_T_D=sample, adaln_lora)


class PatchEmbed(nn.Module):
    def __init__(self, ps, pt, inc, outc):
        super().__init__()
        self.ps, self.pt = ps, pt
        self.proj = nn.Sequential()  # index 0 unused (Rearrange), index 1 = Linear
        self.proj.add_module("0", nn.Identity())
        self.proj.add_module("1", nn.Linear(inc * ps * ps * pt, outc, bias=False))

    def forward(self, x):  # [B,C,T,H,W]
        x = rearrange(x, "b c (t r) (h m) (w n) -> b t h w (c r m n)", r=self.pt, m=self.ps, n=self.ps)
        return self.proj[1](x)


class FinalLayer(nn.Module):
    def __init__(self, hidden, ps, pt, outc, adaln_dim=256):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden, ps * ps * pt * outc, bias=False)
        self.hidden = hidden
        self.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, adaln_dim, bias=False),
                                              nn.Linear(adaln_dim, 2 * hidden, bias=False))

    def forward(self, x, emb, adaln_lora):
        shift, scale = (self.adaln_modulation(emb) + adaln_lora[:, :, : 2 * self.hidden]).chunk(2, dim=-1)
        shift = rearrange(shift, "b t d -> b t 1 1 d"); scale = rearrange(scale, "b t d -> b t 1 1 d")
        x = self.layer_norm(x) * (1 + scale) + shift
        return self.linear(x)


class Block(nn.Module):
    def __init__(self, dim, ctx_dim, heads, mlp_ratio=4.0, adaln_dim=256):
        super().__init__()
        self.layer_norm_self_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, None, heads, dim // heads)
        self.layer_norm_cross_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, ctx_dim, heads, dim // heads)
        self.layer_norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(dim, int(dim * mlp_ratio))

        def mod():
            return nn.Sequential(nn.SiLU(), nn.Linear(dim, adaln_dim, bias=False),
                                 nn.Linear(adaln_dim, 3 * dim, bias=False))
        self.adaln_modulation_self_attn = mod()
        self.adaln_modulation_cross_attn = mod()
        self.adaln_modulation_mlp = mod()

    def forward(self, x, emb, ctx, rope_emb, adaln_lora):
        def three(m): return (m(emb) + adaln_lora).chunk(3, dim=-1)
        ssh, ssc, ssg = three(self.adaln_modulation_self_attn)
        csh, csc, csg = three(self.adaln_modulation_cross_attn)
        msh, msc, msg = three(self.adaln_modulation_mlp)
        r = lambda t: rearrange(t, "b t d -> b t 1 1 d")
        B, T, H, W, D = x.shape

        def fn(xx, ln, sc, sh): return ln(xx) * (1 + r(sc)) + r(sh)
        nx = fn(x, self.layer_norm_self_attn, ssc, ssh)
        res = rearrange(self.self_attn(rearrange(nx, "b t h w d -> b (t h w) d"), None, rope_emb=rope_emb),
                        "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x = torch.addcmul(x, r(ssg), res)

        nx = fn(x, self.layer_norm_cross_attn, csc, csh)
        res = rearrange(self.cross_attn(rearrange(nx, "b t h w d -> b (t h w) d"), ctx, rope_emb=rope_emb),
                        "b (t h w) d -> b t h w d", t=T, h=H, w=W)
        x = torch.addcmul(x, r(csg), res)

        nx = fn(x, self.layer_norm_mlp, msc, msh)
        x = torch.addcmul(x, r(msg), self.mlp(nx))
        return x


class MiniTrainDIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.mc = 2048; self.heads = 16; self.ps = 2; self.pt = 1; self.outc = 16
        self.pos = VideoRopePosition3DEmb(self.mc // self.heads, 240 // 2, 240 // 2, 128 // 1,
                                          h_extrapolation_ratio=4.0, w_extrapolation_ratio=4.0,
                                          t_extrapolation_ratio=1.0, enable_fps_modulation=False)
        self.t_embedder = nn.Sequential(Timesteps(self.mc), TimestepEmbedding(self.mc, self.mc))
        self.x_embedder = PatchEmbed(self.ps, self.pt, 17, self.mc)
        self.blocks = nn.ModuleList([Block(self.mc, 1024, self.heads) for _ in range(28)])
        self.final_layer = FinalLayer(self.mc, self.ps, self.pt, self.outc)
        self.t_embedding_norm = nn.RMSNorm(self.mc, eps=1e-6)

    def forward(self, x, timesteps, context):
        B, C, T, H, W = x.shape
        pad = torch.zeros(B, 1, T, H, W, dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)  # concat_padding_mask
        xe = self.x_embedder(x)  # [B,T,H/ps,W/ps,D]
        rope = self.pos((B, xe.shape[1], xe.shape[2], xe.shape[3], xe.shape[4])).unsqueeze(1).unsqueeze(0)
        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        temb, adaln = self.t_embedder[1](self.t_embedder[0](timesteps))
        temb = self.t_embedding_norm(temb)
        xf = xe.float()
        for b in self.blocks:
            xf = b(xf, temb, context, rope, adaln)
        out = self.final_layer(xf, temb, adaln)
        out = rearrange(out, "B T H W (p1 p2 t C) -> B C (T t) (H p1) (W p2)", p1=self.ps, p2=self.ps, t=self.pt)
        return out


def main():
    dit_path, out_dir = sys.argv[1], sys.argv[2]
    torch.manual_seed(0)
    m = MiniTrainDIT().to(torch.float32).eval()
    sd = {}
    with safe_open(dit_path, framework="pt") as f:
        for kk in f.keys():
            if kk.startswith(PREFIX) and "llm_adapter" not in kk:
                sd[kk[len(PREFIX):]] = f.get_tensor(kk).to(torch.float32)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    unexpected = [u for u in unexpected]
    assert not unexpected, f"unexpected: {unexpected[:8]}"
    # allowed missing: the Rearrange placeholder proj.0, RMSNorm has weight so none.
    miss = [x for x in missing if "proj.0" not in x]
    assert not miss, f"missing: {miss[:8]}"
    print(f"loaded {len(sd)} DiT tensors")

    Hl, Wl = 8, 8  # latent spatial (→ 16 tokens after 2x2 patch: 4x4)
    latent = torch.randn(1, 16, 1, Hl, Wl, dtype=torch.float32)
    context = torch.randn(1, 512, 1024, dtype=torch.float32) * 0.2
    timesteps = torch.tensor([0.7], dtype=torch.float32)
    with torch.no_grad():
        out = m(latent, timesteps, context)
    print("out", tuple(out.shape), "mean", float(out.mean()), "std", float(out.std()))
    save_st(f"{out_dir}/anima_dit_fixture.safetensors", {
        "latent": latent.numpy().astype(np.float32),
        "context": context.numpy().astype(np.float32),
        "timesteps": timesteps.numpy().astype(np.float32),
        "dit_out": out.numpy().astype(np.float32),
    })
    print("wrote", f"{out_dir}/anima_dit_fixture.safetensors")


if __name__ == "__main__":
    main()
