"""Self-contained reference for the Qwen-Image (WAN-2.1) VAE, image path
(decoder AND encoder — the encoder half backs img2img's `Vae.encode`).

Faithful copy of comfy/ldm/wan/vae.py's decode+encode classes. For a single
image T=1, so both run one iteration with feat_cache=None — the whole
temporal-caching path is dead — and every CausalConv3d takes the explicit
zero-pad branch (identical output to comfy's autopad fast path). Loads the
real qwen_image_vae weights and dumps an fp32 CPU fixture.

Run:
    venv/bin/python dump_anima_vae_fixtures.py <qwen_image_vae.safetensors> <out_dir> decode
    venv/bin/python dump_anima_vae_fixtures.py <qwen_image_vae.safetensors> <out_dir> encode

`mode` defaults to "decode" (the original CLI shape, unchanged). "encode"
writes `anima_vae_encode_fixture.safetensors` — feed it to `zig build test`
via `ANIMA_VAE=<qwen_image_vae.safetensors> ANIMA_VAE_ENC_FIXTURE=<that file>`
(test "anima: Qwen-Image VAE encode parity vs reference fixture" in anima.zig).
"""
import sys, json, struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open


def save_st(path, tensors):
    _DT = {"float32": "F32"}
    header, blob, off = {}, bytearray(), 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        b = arr.tobytes()
        header[name] = {"dtype": _DT[str(arr.dtype)], "shape": list(arr.shape), "data_offsets": [off, off + len(b)]}
        blob += b; off += len(b)
    hj = json.dumps(header).encode(); hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj))); f.write(hj); f.write(blob)


class CausalConv3d(nn.Conv3d):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._padding = 2 * self.padding[0]
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x, cache_x=None):
        if self._padding > 0:
            pad_shape = list(x.shape)
            pad_shape[2] = self._padding
            padding = torch.zeros(pad_shape, device=x.device, dtype=x.dtype)
            x = torch.cat([padding, x], dim=2)
        return super().forward(x)


class RMS_norm(nn.Module):
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        bd = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *bd) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma.to(x) + (self.bias.to(x) if self.bias is not None else 0)


def vae_attention(q, k, v):
    b, c, h, w = q.shape
    q = q.reshape(b, c, h * w).transpose(1, 2)
    k = k.reshape(b, c, h * w).transpose(1, 2)
    v = v.reshape(b, c, h * w).transpose(1, 2)
    o = F.scaled_dot_product_attention(q, k, v)
    return o.transpose(1, 2).reshape(b, c, h, w)


class Resample(nn.Module):
    def __init__(self, dim, mode):
        super().__init__()
        self.dim, self.mode = dim, mode
        if mode == "upsample2d":
            self.resample = nn.Sequential(nn.Upsample(scale_factor=(2., 2.), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == "upsample3d":
            self.resample = nn.Sequential(nn.Upsample(scale_factor=(2., 2.), mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode in ("downsample2d", "downsample3d"):
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x):  # feat_cache=None (image path): temporal path is dead
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(), CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout), CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        x = vae_attention(q, k, v)
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x + identity


class Decoder3d(nn.Module):
    def __init__(self, dim=96, z_dim=16, output_channels=3, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                 attn_scales=[], temperal_upsample=[True, True, False], dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.middle = nn.Sequential(ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]), ResidualBlock(dims[0], dims[0], dropout))
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "upsample3d" if temperal_upsample[i] else "upsample2d"
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, output_channels, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.upsamples:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class Encoder3d(nn.Module):
    """Mirror of Decoder3d, reversed (comfy/ldm/wan/vae.py Encoder3d)."""

    def __init__(self, dim=96, z_dim=16, input_channels=3, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                 attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0):
        super().__init__()
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv1 = CausalConv3d(input_channels, dims[0], 3, padding=1)
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim), ResidualBlock(out_dim, out_dim, dropout))
        self.head = nn.Sequential(RMS_norm(out_dim, images=False), nn.SiLU(), CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.downsamples:
            x = layer(x)
        for layer in self.middle:
            x = layer(x)
        for layer in self.head:
            x = layer(x)
        return x


class WanVAE(nn.Module):
    def __init__(self, dim=96, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2):
        super().__init__()
        self.encoder = Encoder3d(dim, z_dim * 2, 3, dim_mult, num_res_blocks)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)  # quant_conv
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)  # post_quant_conv
        self.decoder = Decoder3d(dim, z_dim, 3, dim_mult, num_res_blocks)

    def encode(self, x, stages=None):  # x [1,3,1,H,W] in [0,1] (image: T=1) -> mu [1,16,1,H/8,W/8]
        if stages is not None:
            stages["image"] = x[:, :, 0].clone()  # [1,3,H,W] [0,1] — the fixture's "image" input
        x = x * 2 - 1  # [0,1] -> [-1,1], the WAN2.1 VAE's own encode convention
        out = self.encoder(x)
        if stages is not None: stages["encoder_out"] = out[:, :, 0].clone()  # [1,32,h,w]
        enc = self.conv1(out)
        mu, log_var = enc.chunk(2, dim=1)
        if stages is not None: stages["mu"] = mu[:, :, 0].clone()  # [1,16,h,w]
        return mu

    def decode(self, z, stages=None):  # z [1,16,1,H,W]  (image: T=1)
        if stages is not None:
            stages["input"] = z[:, :, 0:1].clone()
            # conv2 weight as [O,I] laid into a fake [1,O,I,1] so the Zig NCHW
            # comparison (transpose 0,3,1,2 of a [1,O,1,I]) lines up.
            wm = self.conv2.weight[:, :, 0, 0, 0]  # [O,I]
            stages["conv2_wmat"] = wm.reshape(1, wm.shape[0], 1, wm.shape[1]).clone()
        x = self.conv2(z)
        x = x[:, :, 0:1, :, :]
        if stages is not None: stages["after_conv2"] = x.clone()
        d = self.decoder
        x = d.conv1(x)
        if stages is not None: stages["after_conv1"] = x.clone()
        for layer in d.middle:
            x = layer(x)
        if stages is not None: stages["after_middle"] = x.clone()
        for layer in d.upsamples:
            x = layer(x)
        if stages is not None: stages["after_ups"] = x.clone()
        for layer in d.head:
            x = layer(x)
        return x


def main():
    vae_path, out_dir = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "decode"
    assert mode in ("decode", "encode"), f"unknown mode {mode!r} (want decode|encode)"
    torch.manual_seed(0)
    m = WanVAE().to(torch.float32).eval()
    dec_prefixes = ("decoder.", "conv2.weight", "conv2.bias")
    enc_prefixes = ("encoder.", "conv1.weight", "conv1.bias")
    want_prefixes = dec_prefixes if mode == "decode" else enc_prefixes
    sd = {}
    with safe_open(vae_path, framework="pt") as f:
        for kk in f.keys():
            if kk.startswith(want_prefixes) or kk in want_prefixes:
                sd[kk] = f.get_tensor(kk).to(torch.float32)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    unexpected = [u for u in unexpected]
    assert not unexpected, f"unexpected: {unexpected[:8]}"
    # The half of the model this mode doesn't use (encoder+conv1 in decode
    # mode, decoder+conv2 in encode mode) is expected to stay uninitialized.
    other_prefixes = enc_prefixes if mode == "decode" else dec_prefixes
    miss = [x for x in missing if "num_batches" not in x and not x.startswith(other_prefixes)]
    assert not miss, f"missing: {miss[:8]}"
    print(f"loaded {len(sd)} VAE tensors ({mode})")

    if mode == "decode":
        latent = torch.randn(1, 16, 1, 8, 8, dtype=torch.float32) * 0.5
        stages = {}
        with torch.no_grad():
            out = m.decode(latent, stages)  # [1,3,1,64,64]
        print("out", tuple(out.shape), "mean", float(out.mean()), "std", float(out.std()))
        tosave = {"latent": latent.numpy().astype(np.float32), "vae_out": out.numpy().astype(np.float32)}
        for kk, vv in stages.items():
            # store NCHW (drop T) so the Zig side can compare in its NCHW output order
            arr = vv[:, :, 0].contiguous().numpy().astype(np.float32)  # [1,C,H,W]
            tosave[kk] = arr
            print(f"  stage {kk}: {tuple(arr.shape)}")
        out_path = f"{out_dir}/anima_vae_fixture.safetensors"
        save_st(out_path, tosave)
        print("wrote", out_path)
    else:
        # Deterministic smooth gradient image, H=W=64 -> latent 8x8 (multiple
        # of 16, matching the smallest real request size).
        HW = 64
        yy, xx = torch.meshgrid(torch.linspace(0, 1, HW), torch.linspace(0, 1, HW), indexing="ij")
        img = torch.stack([xx, yy, 0.5 * (xx + yy)], dim=0).unsqueeze(0).unsqueeze(2)  # [1,3,1,H,W] in [0,1]
        stages = {}
        with torch.no_grad():
            mu = m.encode(img, stages)  # [1,16,1,h,w]
        print("mu", tuple(mu.shape), "mean", float(mu.mean()), "std", float(mu.std()))
        tosave = {}
        for kk, vv in stages.items():
            tosave[kk] = vv.contiguous().numpy().astype(np.float32)  # already dropped T
            print(f"  stage {kk}: {tuple(tosave[kk].shape)}")
        out_path = f"{out_dir}/anima_vae_encode_fixture.safetensors"
        save_st(out_path, tosave)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
