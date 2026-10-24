# Portions adapted from facebookresearch/DiT.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Upstream portions: LICENSES/DiT-CC-BY-NC-4.0.txt.
# Modified for dual-stream, U-shaped conditional image fusion.

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                               Embedding Layers                                #
#################################################################################


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class PatchEmbed(nn.Module):
    """Dynamic 2D image-to-patch embedding."""

    def __init__(
        self,
        patch_size: int = 2,
        in_chans: int = 4,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        ph, pw = self.patch_size
        assert h % ph == 0, f"Input height {h} is not divisible by patch size {ph}"
        assert w % pw == 0, f"Input width {w} is not divisible by patch size {pw}"
        x = self.proj(x)
        hp, wp = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        return x, hp, wp


#################################################################################
#                                RoPE Utilities                                 #
#################################################################################


class RoPE2DCache:
    @staticmethod
    def build(
        h: int,
        w: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
        base: float = 10000.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert (
            head_dim % 4 == 0
        ), f"head_dim={head_dim} must be divisible by 4 for 2D RoPE"
        quarter = head_dim // 4

        y = torch.arange(h, device=device, dtype=torch.float32)
        x = torch.arange(w, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)

        inv_freq = 1.0 / (
            base
            ** (torch.arange(quarter, device=device, dtype=torch.float32) / quarter)
        )
        freqs_h = torch.einsum("n,d->nd", yy, inv_freq)
        freqs_w = torch.einsum("n,d->nd", xx, inv_freq)

        cos_h = torch.cos(freqs_h).to(dtype=dtype)
        sin_h = torch.sin(freqs_h).to(dtype=dtype)
        cos_w = torch.cos(freqs_w).to(dtype=dtype)
        sin_w = torch.sin(freqs_w).to(dtype=dtype)
        return cos_h, sin_h, cos_w, sin_w


def apply_2d_rope(
    x: torch.Tensor,
    cos_h: torch.Tensor,
    sin_h: torch.Tensor,
    cos_w: torch.Tensor,
    sin_w: torch.Tensor,
) -> torch.Tensor:
    """
    x: [B, heads, N, head_dim]
    cos/sin_*: [N, head_dim // 4]
    """
    b, heads, n, d = x.shape
    quarter = d // 4
    assert d % 4 == 0, f"head_dim={d} must be divisible by 4"

    x_h1 = x[..., :quarter]
    x_h2 = x[..., quarter : 2 * quarter]
    x_w1 = x[..., 2 * quarter : 3 * quarter]
    x_w2 = x[..., 3 * quarter :]

    cos_h = cos_h.view(1, 1, n, quarter)
    sin_h = sin_h.view(1, 1, n, quarter)
    cos_w = cos_w.view(1, 1, n, quarter)
    sin_w = sin_w.view(1, 1, n, quarter)

    out_h1 = x_h1 * cos_h - x_h2 * sin_h
    out_h2 = x_h2 * cos_h + x_h1 * sin_h
    out_w1 = x_w1 * cos_w - x_w2 * sin_w
    out_w2 = x_w2 * cos_w + x_w1 * sin_w

    return torch.cat([out_h1, out_h2, out_w1, out_w2], dim=-1)


#################################################################################
#                            Token/Grid Conversions                             #
#################################################################################


def tokens_to_map(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, n, c = x.shape
    assert n == h * w, f"Token number mismatch: {n} != {h} * {w}"
    return x.transpose(1, 2).reshape(b, c, h, w)


def map_to_tokens(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    return x.flatten(2).transpose(1, 2), h, w


#################################################################################
#                               Core Attention Blocks                           #
#################################################################################


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert (
            self.head_dim % 4 == 0
        ), f"head_dim={self.head_dim} must be divisible by 4 for 2D RoPE"
        self.scale = self.head_dim**-0.5
        self.rope_base = rope_base

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        q_hw: Tuple[int, int],
        kv_hw: Tuple[int, int],
    ) -> torch.Tensor:
        b, nq, c = q_in.shape
        nk = kv_in.shape[1]
        qh, qw = q_hw
        kh, kw = kv_hw
        assert nq == qh * qw, f"Query token number mismatch: {nq} != {qh} * {qw}"
        assert nk == kh * kw, f"Key token number mismatch: {nk} != {kh} * {kw}"

        q = self.q(q_in).reshape(b, nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(kv_in).reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(kv_in).reshape(b, nk, self.num_heads, self.head_dim).transpose(1, 2)

        q_cos_h, q_sin_h, q_cos_w, q_sin_w = RoPE2DCache.build(
            qh, qw, self.head_dim, q.device, q.dtype, self.rope_base
        )
        k_cos_h, k_sin_h, k_cos_w, k_sin_w = RoPE2DCache.build(
            kh, kw, self.head_dim, k.device, k.dtype, self.rope_base
        )

        q = apply_2d_rope(q, q_cos_h, q_sin_h, q_cos_w, q_sin_w)
        k = apply_2d_rope(k, k_cos_h, k_sin_h, k_cos_w, k_sin_w)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, nq, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MultiHeadSelfAttention(MultiHeadCrossAttention):
    def forward(self, x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        return super().forward(x, x, hw, hw)


class DualStreamBlock(nn.Module):
    """Lighter dual-stream block with stage-wise adaLN conditioning."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.norm_ir_self = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_vis_self = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn_ir = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=True, rope_base=rope_base
        )
        self.self_attn_vis = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=True, rope_base=rope_base
        )

        self.norm_ir_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_vis_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_ir_from_vis = MultiHeadCrossAttention(
            dim, num_heads=num_heads, qkv_bias=True, rope_base=rope_base
        )
        self.cross_vis_from_ir = MultiHeadCrossAttention(
            dim, num_heads=num_heads, qkv_bias=True, rope_base=rope_base
        )

        self.norm_ir_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_vis_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp_ir = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.mlp_vis = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 18 * dim, bias=True),
        )

    def forward(
        self,
        x_ir: torch.Tensor,
        x_vis: torch.Tensor,
        c: torch.Tensor,
        hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        chunks = self.adaLN_modulation(c).chunk(18, dim=1)
        (
            ir_shift_self,
            ir_scale_self,
            ir_gate_self,
            ir_shift_cross,
            ir_scale_cross,
            ir_gate_cross,
            ir_shift_mlp,
            ir_scale_mlp,
            ir_gate_mlp,
            vis_shift_self,
            vis_scale_self,
            vis_gate_self,
            vis_shift_cross,
            vis_scale_cross,
            vis_gate_cross,
            vis_shift_mlp,
            vis_scale_mlp,
            vis_gate_mlp,
        ) = chunks

        x_ir = x_ir + ir_gate_self.unsqueeze(1) * self.self_attn_ir(
            modulate(self.norm_ir_self(x_ir), ir_shift_self, ir_scale_self),
            hw,
        )
        x_vis = x_vis + vis_gate_self.unsqueeze(1) * self.self_attn_vis(
            modulate(self.norm_vis_self(x_vis), vis_shift_self, vis_scale_self),
            hw,
        )

        x_ir = x_ir + ir_gate_cross.unsqueeze(1) * self.cross_ir_from_vis(
            modulate(self.norm_ir_cross(x_ir), ir_shift_cross, ir_scale_cross),
            x_vis,
            hw,
            hw,
        )
        x_vis = x_vis + vis_gate_cross.unsqueeze(1) * self.cross_vis_from_ir(
            modulate(self.norm_vis_cross(x_vis), vis_shift_cross, vis_scale_cross),
            x_ir,
            hw,
            hw,
        )

        x_ir = x_ir + ir_gate_mlp.unsqueeze(1) * self.mlp_ir(
            modulate(self.norm_ir_mlp(x_ir), ir_shift_mlp, ir_scale_mlp)
        )
        x_vis = x_vis + vis_gate_mlp.unsqueeze(1) * self.mlp_vis(
            modulate(self.norm_vis_mlp(x_vis), vis_shift_mlp, vis_scale_mlp)
        )
        return x_ir, x_vis


class TokenDownsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> Tuple[torch.Tensor, int, int]:
        x = tokens_to_map(x, h, w)
        x = self.proj(x)
        return map_to_tokens(x)


class TokenUpsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        target_h: int,
        target_w: int,
    ) -> Tuple[torch.Tensor, int, int]:
        x = tokens_to_map(x, h, w)
        x = F.interpolate(
            x,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        x = self.proj(x)
        return map_to_tokens(x)


class SkipMerge(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

    def forward(self, up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = torch.cat([up, skip, up - skip, up * skip], dim=-1)
        return self.proj(x)


class FinalFusionHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

    def forward(self, x_ir: torch.Tensor, x_vis: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(torch.cat([x_ir, x_vis], dim=-1)))
        return g * x_ir + (1.0 - g) * x_vis


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


#################################################################################
#                          Dual-stream U-shaped DiT Lite                        #
#################################################################################


class DiT(nn.Module):
    """
    Lighter dual-stream U-shaped DiT with 2D RoPE.

    Compatible API:
        forward(x, t, cond_img1, cond_img2)
    """

    def __init__(
        self,
        input_size: int = 16,
        patch_size: int = 2,
        in_channels: int = 8,
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        learn_sigma: bool = True,
        use_checkpoint: bool = True,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "hidden_size must be divisible by num_heads"
        assert depth >= 6, "use depth >= 6 for the U-shaped hierarchy"

        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.input_size = input_size
        self.use_checkpoint = use_checkpoint

        self.x_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )
        self.ir_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )
        self.vis_embedder = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size,
            bias=True,
        )

        self.x_to_ir = nn.Linear(hidden_size, hidden_size)
        self.x_to_vis = nn.Linear(hidden_size, hidden_size)

        dim0 = hidden_size
        dim1 = hidden_size * 2
        dim2 = hidden_size * 2  # keep bottleneck width modest to save memory

        head0 = num_heads
        head1 = num_heads * 2
        head2 = num_heads * 2

        assert (
            dim1 % head1 == 0 and dim2 % head2 == 0
        ), "Scaled dims must stay divisible by scaled heads"
        assert (
            dim0 // head0
        ) % 4 == 0, "stage0 head_dim must be divisible by 4 for 2D RoPE"
        assert (
            dim1 // head1
        ) % 4 == 0, "stage1 head_dim must be divisible by 4 for 2D RoPE"
        assert (
            dim2 // head2
        ) % 4 == 0, "bottleneck head_dim must be divisible by 4 for 2D RoPE"

        # stage allocation: shallow encoder / mid / decoder
        n0 = max(1, depth // 4)
        n1 = max(1, depth // 4)
        n_mid = max(2, depth - (2 * n0 + 2 * n1))

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.cond_proj0 = nn.Identity()  # hidden_size -> dim0
        self.cond_proj1 = nn.Linear(hidden_size, dim1)  # hidden_size -> dim1
        self.cond_proj2 = nn.Linear(hidden_size, dim2)  # hidden_size -> dim2

        self.stage0 = nn.ModuleList(
            [DualStreamBlock(dim0, head0, mlp_ratio, rope_base) for _ in range(n0)]
        )
        self.down0_ir = TokenDownsample(dim0, dim1)
        self.down0_vis = TokenDownsample(dim0, dim1)

        self.stage1 = nn.ModuleList(
            [DualStreamBlock(dim1, head1, mlp_ratio, rope_base) for _ in range(n1)]
        )
        self.down1_ir = TokenDownsample(dim1, dim2)
        self.down1_vis = TokenDownsample(dim1, dim2)

        self.mid = nn.ModuleList(
            [DualStreamBlock(dim2, head2, mlp_ratio, rope_base) for _ in range(n_mid)]
        )

        self.up1_ir = TokenUpsample(dim2, dim1)
        self.up1_vis = TokenUpsample(dim2, dim1)
        self.merge1_ir = SkipMerge(dim1)
        self.merge1_vis = SkipMerge(dim1)
        self.dec1 = nn.ModuleList(
            [DualStreamBlock(dim1, head1, mlp_ratio, rope_base) for _ in range(n1)]
        )

        self.up0_ir = TokenUpsample(dim1, dim0)
        self.up0_vis = TokenUpsample(dim1, dim0)
        self.merge0_ir = SkipMerge(dim0)
        self.merge0_vis = SkipMerge(dim0)
        self.dec0 = nn.ModuleList(
            [DualStreamBlock(dim0, head0, mlp_ratio, rope_base) for _ in range(n0)]
        )

        self.final_fusion = FinalFusionHead(dim0)
        self.final_layer = FinalLayer(dim0, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.shape[0], -1))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for module in self.modules():
            if isinstance(module, DualStreamBlock):
                nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(module.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _run_blocks(
        self,
        blocks: nn.ModuleList,
        x_ir: torch.Tensor,
        x_vis: torch.Tensor,
        c: torch.Tensor,
        hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for block in blocks:
            if self.use_checkpoint and self.training:
                x_ir, x_vis = torch.utils.checkpoint.checkpoint(
                    block, x_ir, x_vis, c, hw, use_reentrant=False
                )
            else:
                x_ir, x_vis = block(x_ir, x_vis, c, hw)
        return x_ir, x_vis

    def unpatchify(
        self, x: torch.Tensor, h_patches: int, w_patches: int
    ) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        assert h_patches * w_patches == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h_patches, w_patches, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h_patches * p, w_patches * p))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_img1: torch.Tensor,
        cond_img2: torch.Tensor,
    ) -> torch.Tensor:
        x_tok, hp, wp = self.x_embedder(x)
        ir_tok, hp_ir, wp_ir = self.ir_embedder(cond_img1)
        vis_tok, hp_vis, wp_vis = self.vis_embedder(cond_img2)
        assert (
            (hp, wp) == (hp_ir, wp_ir) == (hp_vis, wp_vis)
        ), "Patch grid mismatch among inputs"

        ir_tok = ir_tok + self.x_to_ir(x_tok)
        vis_tok = vis_tok + self.x_to_vis(x_tok)

        c_base = self.t_embedder(t)
        c0 = self.cond_proj0(c_base)
        c1 = self.cond_proj1(c_base)
        c2 = self.cond_proj2(c_base)

        # Encoder stage 0
        ir_tok, vis_tok = self._run_blocks(self.stage0, ir_tok, vis_tok, c0, (hp, wp))
        skip0_ir, skip0_vis = ir_tok, vis_tok

        # Down 0 -> stage 1
        ir_tok, h1, w1 = self.down0_ir(ir_tok, hp, wp)
        vis_tok, _, _ = self.down0_vis(vis_tok, hp, wp)
        ir_tok, vis_tok = self._run_blocks(self.stage1, ir_tok, vis_tok, c1, (h1, w1))
        skip1_ir, skip1_vis = ir_tok, vis_tok

        # Down 1 -> bottleneck
        ir_tok, h2, w2 = self.down1_ir(ir_tok, h1, w1)
        vis_tok, _, _ = self.down1_vis(vis_tok, h1, w1)
        ir_tok, vis_tok = self._run_blocks(self.mid, ir_tok, vis_tok, c2, (h2, w2))

        # Up 1 + skip1
        ir_tok, h1u, w1u = self.up1_ir(ir_tok, h2, w2, h1, w1)
        vis_tok, _, _ = self.up1_vis(vis_tok, h2, w2, h1, w1)

        ir_tok = self.merge1_ir(ir_tok, skip1_ir)
        vis_tok = self.merge1_vis(vis_tok, skip1_vis)
        ir_tok, vis_tok = self._run_blocks(self.dec1, ir_tok, vis_tok, c1, (h1, w1))

        # Up 0 + skip0
        ir_tok, h0u, w0u = self.up0_ir(ir_tok, h1, w1, hp, wp)
        vis_tok, _, _ = self.up0_vis(vis_tok, h1, w1, hp, wp)

        ir_tok = self.merge0_ir(ir_tok, skip0_ir)
        vis_tok = self.merge0_vis(vis_tok, skip0_vis)
        ir_tok, vis_tok = self._run_blocks(self.dec0, ir_tok, vis_tok, c0, (hp, wp))

        fused = self.final_fusion(ir_tok, vis_tok)
        out = self.final_layer(fused, c0)
        out = self.unpatchify(out, hp, wp)
        return out

    def forward_with_cfg(self, x, t, cond_img1, cond_img2, cfg_scale):
        half = x[: len(x) // 2]
        half_cond1 = cond_img1[: len(cond_img1) // 2]
        half_cond2 = cond_img2[: len(cond_img2) // 2]
        combined = torch.cat([half, half], dim=0)
        combined_cond1 = torch.cat([half_cond1, half_cond1], dim=0)
        combined_cond2 = torch.cat([half_cond2, half_cond2], dim=0)
        model_out = self.forward(
            combined, t, cond_img1=combined_cond1, cond_img2=combined_cond2
        )

        if self.in_channels >= 3:
            eps, rest = model_out[:, :3], model_out[:, 3:]
        else:
            eps, rest = (
                model_out[:, : self.in_channels],
                model_out[:, self.in_channels :],
            )

        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                    Configs                                    #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=16, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=16, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=16, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=10, hidden_size=512, patch_size=2, num_heads=8, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=10, hidden_size=512, patch_size=4, num_heads=8, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=10, hidden_size=512, patch_size=8, num_heads=8, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=8, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=8, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=8, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}
