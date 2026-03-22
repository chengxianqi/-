import math
import torch
import torch.nn as nn
from components import AdaLN, MHA, Mlp, SwiGLU, PatchEmbed, TimestepEmbedding, AxialRotaryEmbedding

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, rope=None, mlp_ratio=4, use_glu=False):
        super().__init__()
        self.norm1 = AdaLN(hidden_size, hidden_size)
        self.attn = MHA(hidden_size, num_heads, qkv_bias=False, rope=rope)
        self.norm2 = AdaLN(hidden_size, hidden_size)
        if use_glu:
            self.mlp = SwiGLU(hidden_size, mult=mlp_ratio)
        else:
            self.mlp = Mlp(hidden_size, mult=mlp_ratio)
        self.gate_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c):
        gate1, gate2 = self.gate_proj(c).chunk(2, dim=1)
        attn_out = self.attn(self.norm1(x, c)) * gate1.unsqueeze(1)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x, c)) * gate2.unsqueeze(1)
        return x + mlp_out


class DiT(nn.Module):
    def __init__(self, input_size=32, patch_size=2, in_chans=3, num_class=10, hidden_size=768, depth=12, heads=12, mlp_ratio=4, use_glu=False):
        super().__init__()
        assert hidden_size % heads == 0
        assert input_size % patch_size == 0
        self.in_chans = in_chans
        self.patch_size = patch_size
        num_patches = (input_size // patch_size) ** 2
        self.x_embedder = PatchEmbed(patch_size, in_chans, hidden_size)
        self.t_embedder = TimestepEmbedding(hidden_size, hidden_size)
        self.y_embedder = nn.Embedding(num_class + 1, hidden_size)
        self.rope = AxialRotaryEmbedding(hidden_size // heads, num_patches)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, heads, rope=self.rope, mlp_ratio=mlp_ratio, use_glu=use_glu)
            for i in range(depth)
        ])
        self.norm = AdaLN(hidden_size, cond_dim=hidden_size)
        self.head = nn.Linear(hidden_size, patch_size * patch_size * in_chans)
        self._init_weights()

    def unpatchify(self, x):
        b, seq_len, _ = x.shape
        c, p = self.in_chans, self.patch_size
        h = w = math.isqrt(seq_len)
        assert h * w == seq_len
        x = x.reshape(b, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        imgs = x.reshape(b, c, h * p, w * p)
        return imgs

    def forward(self, x, t, y):
        x = self.x_embedder(x)
        c = self.t_embedder(t) + self.y_embedder(y)
        for block in self.blocks:
            x = block(x, c)

        x = self.head(self.norm(x, c))
        return self.unpatchify(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, AdaLN):
                nn.init.zeros_(m.linear.weight)

        for block in self.blocks:
            nn.init.zeros_(block.gate_proj[-1].weight)
        
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.y_embedder.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        nn.init.trunc_normal_(self.y_embedder.weight, std=0.02)
        nn.init.zeros_(self.norm.linear.weight)
        nn.init.zeros_(self.head.weight)