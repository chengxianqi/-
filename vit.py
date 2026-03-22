import torch
import torch.nn as nn
from components import PatchEmbed, AxialRotaryEmbedding, MHA, Mlp, SwiGLU, DropPath

class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, rope=None, drop_path=0., mlp_ratio=4, use_glu=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = MHA(dim, num_heads, qkv_bias=False, rope=rope)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        if use_glu:
            self.mlp = SwiGLU(dim, mult=mlp_ratio)
        else:
            self.mlp = Mlp(dim=dim, mult=mlp_ratio, use_grn=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        x = x + self.drop_path(attn_out)
        mlp_out = self.mlp(self.norm2(x))
        return x + self.drop_path(mlp_out)


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, num_class, dim=768, depth=12, heads=12, drop_path=0., mlp_ratio=4, use_glu=False):
        super().__init__()
        assert dim % heads == 0
        assert img_size % patch_size == 0
        self.patch_embed = PatchEmbed(patch_size, in_chans, dim)
        num_patches = (img_size // patch_size) ** 2
        self.rope = AxialRotaryEmbedding(dim // heads, num_patches)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.Sequential(*[
            EncoderBlock(dim, heads, rope=self.rope, drop_path=dp_rates[i], mlp_ratio=mlp_ratio, use_glu=use_glu)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Linear(dim, num_class)
        self._init_weights()

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        return self.head(self.norm(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.weight)