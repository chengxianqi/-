import torch
import torch.nn as nn
from components import GQA, SwiGLU, DropPath, RotaryEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope=None, mult=8/3, bias=False):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = GQA(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=False,
            proj_bias=bias,
            rope=rope
        )
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = SwiGLU(dim, mult=mult, bias=bias)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x), causal=True)
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        return x + mlp_out


class DecoderOnlyLM(nn.Module):
    def __init__(self, vocab_size, max_seq_len, dim=768, depth=12, num_heads=12, num_kv_heads=3, bias=False):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        head_dim = dim // num_heads
        self.rope = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len, base=10000)
        self.layers = nn.ModuleList([
            DecoderBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                rope=self.rope,
                bias = bias
            )
            for i in range(depth)
        ])
        self.norm = nn.RMSNorm(dim, eps=1e-6)
        self.lm_head = nn.Linear(dim, vocab_size, bias=bias)
        self._init_weights()

    def forward(self, tokens):
        x = self.tok_embeddings(tokens)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)