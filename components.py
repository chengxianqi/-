import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 3.1 归一化 (Normalization)
# ==========================================

class GRN(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        
    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GroupRMSNorm(nn.Module):
    def __init__(self, groups, channels, eps=1e-6, affine=True):
        super().__init__()
        self.groups = groups
        assert channels % groups == 0
        self.eps = eps
        if affine: self.weight = nn.Parameter(torch.ones(channels))
        else: self.register_parameter("weight", None)
        
    def forward(self, x):
        B, C, H, W = x.shape
        G = self.groups
        x = x.view(B, G, -1)
        inv_rms = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        x = (x * inv_rms).view(B, C, H, W)
        return x if self.weight is None else x * self.weight.view(1, C, 1, 1)


class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim, eps=1e-6, use_rms=False):
        super().__init__()
        norm = nn.RMSNorm if use_rms else nn.LayerNorm
        self.norm = norm(dim, eps=eps, elementwise_affine=False)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, c):
        shift, scale = self.linear(self.silu(c)).chunk(2, dim=-1)
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
        return self.norm(x) * (1 + scale) + shift


class AdaGN(nn.Module):
    def __init__(self, dim, cond_dim, num_groups=32, eps=1e-6, use_rms=False):
        super().__init__()
        norm = GroupRMSNorm if use_rms else nn.GroupNorm
        self.norm = norm(num_groups, dim, eps=eps, affine=False)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(cond_dim, dim * 2)

    def forward(self, x, c):
        shift, scale = self.linear(self.silu(c)).chunk(2, dim=-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        return self.norm(x) * (1 + scale) + shift

# ==========================================
# 3.2 特征变换 (Feature Transforms)
# ==========================================

class Mlp(nn.Module):
    def __init__(self, dim, mult=4, bias=True, act=nn.GELU, drop=0., use_grn=False):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.fc1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.act = act()
        self.grn = GRN(hidden_dim) if use_grn else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()
    
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.drop(self.grn(x))
        return self.fc2(x)


class SwiGLU(nn.Module):
    def __init__(self, dim, mult=8/3, bias=True, act=nn.SiLU, drop=0., use_grn=False):
        super().__init__()
        hidden_dim = math.ceil(dim * mult / 256) * 256
        self.proj = nn.Linear(dim, hidden_dim * 2, bias=bias)
        self.act = nn.SiLU()
        self.grn = GRN(hidden_dim) if use_grn else nn.Identity()
        self.out = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop) if drop > 0. else nn.Identity()

    def forward(self, x):
        x1, x2 = self.proj(x).chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.drop(self.grn(x))
        return self.out(x)

# ==========================================
# 3.3 注意力 (Attention)
# ==========================================

class MHA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, proj_bias=True, rope=None, drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.o_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope
        self.drop = drop

    def forward(self, x, causal=False):
        B, L, D = x.shape
        qkv = self.qkv_proj(x).view(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0, is_causal=causal
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)


class MHA2d(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, proj_bias=True, rope=None, drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.qkv_proj = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.o_proj = nn.Conv2d(dim, dim, 1, bias=proj_bias)
        self.rope = rope
        self.drop = drop

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv_proj(x).view(B, 3, self.num_heads, -1, H * W).permute(1, 0, 2, 4, 3)
        q, k, v = qkv.unbind(dim=0)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0
        )
        out = out.transpose(2, 3).reshape(B, C, H, W)
        return self.o_proj(out)


class GQA(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, qkv_bias=True, proj_bias=True, rope=None, drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.groups = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        kv_dim = num_kv_heads * dim // num_heads
        self.kv_proj = nn.Linear(dim, kv_dim * 2, bias=qkv_bias)
        self.o_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.rope = rope
        self.drop = drop

    def forward(self, x, causal=False):
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, -1).transpose(1, 2)
        kv = self.kv_proj(x).view(B, L, 2, self.num_kv_heads, -1).permute(2, 0, 3, 1, 4)
        if self.groups > 1:
            kv = kv.unsqueeze(3).expand(2, B, self.num_kv_heads, self.groups, L, -1)
            kv = kv.reshape(2, B, self.num_heads, L, -1)
        k, v = kv.unbind(dim=0)
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0, is_causal=causal
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.o_proj(out)

# ==========================================
# 3.4 输入适配 (Input Adapters)
# ==========================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = torch.exp(-math.log(base) * torch.linspace(0, 1, dim // 2))
        t = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):
        B, H, L, Hd = x.shape
        cos = self.cos[None, None, :L, :]
        sin = self.sin[None, None, :L, :]
        x = x.reshape(B, H, L, -1, 2)
        x1, x2 = x[..., 0], x[..., 1]
        x_out = torch.empty_like(x)
        x_out[..., 0] = x1 * cos - x2 * sin
        x_out[..., 1] = x1 * sin + x2 * cos
        return x_out.view(B, H, L, Hd)


class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()
        assert dim % 4 == 0
        grid_size = math.isqrt(max_seq_len)
        assert grid_size * grid_size == max_seq_len
        inv_freq = torch.exp(-math.log(base) * torch.linspace(0, 1, dim // 4))
        t = torch.arange(grid_size, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        freqs_h = freqs[:, None, :].expand(grid_size, grid_size, -1)
        freqs_w = freqs[None, :, :].expand(grid_size, grid_size, -1)
        freqs_2d = torch.cat([freqs_h, freqs_w], dim=-1).reshape(max_seq_len, -1)
        self.register_buffer("cos", freqs_2d.cos(), persistent=False)
        self.register_buffer("sin", freqs_2d.sin(), persistent=False)

    def forward(self, x):
        B, H, L, Hd = x.shape
        cos = self.cos[None, None, :L, :]
        sin = self.sin[None, None, :L, :]
        x = x.reshape(B, H, L, -1, 2)
        x1, x2 = x[..., 0], x[..., 1]
        x_out = torch.empty_like(x)
        x_out[..., 0] = x1 * cos - x2 * sin
        x_out[..., 1] = x1 * sin + x2 * cos
        return x_out.view(B, H, L, Hd)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, out_dim, base=10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = torch.exp(-math.log(base) * torch.linspace(0, 1, dim // 2))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, t):
        if t.ndim == 1: t = t[:, None].float()
        freq = t * self.inv_freq[None, :]
        emb = torch.cat([freq.sin(), freq.cos()], dim=-1)  
        return self.mlp(emb)


class GFF(nn.Module):
    def __init__(self, dim, out_dim, scale=10.0):
        super().__init__()
        assert dim % 2 == 0
        B = torch.randn(dim // 2) * scale
        B = 2 * math.pi * B
        self.register_buffer("B", B)
        self.mlp = nn.Sequential(
            nn.Linear(dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, t):
        if t.ndim == 1: t = t[:, None].float()
        freq = t @ self.B
        emb = torch.cat([freq.sin(), freq.cos()], dim=-1)
        return self.mlp(emb)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)

# ==========================================
# 3.5 正则化 (Regularization)
# ==========================================

class DropPath(nn.Module):
    def __init__(self, drop_prob=0., inplace=False):
        super().__init__()
        self.drop_prob = drop_prob
        self.inplace = inplace

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0: mask.div_(keep_prob)
        return x.mul_(mask) if self.inplace else x * mask


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, channel_last=True):
        super().__init__()
        self.channel_last = channel_last
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        if self.channel_last: return x * self.gamma
        else: return x * self.gamma.view(1, -1, 1, 1)