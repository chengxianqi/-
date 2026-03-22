import torch
import torch.nn as nn
from components import DropPath, LayerNorm2d, SwiGLU, Mlp

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., kernel_size=7, expansion=4, use_glu=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        if use_glu:
            self.mlp = SwiGLU(dim, mult=expansion, use_grn=True)
        else:
            self.mlp = Mlp(dim, mult=expansion, use_grn=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans, num_class, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768), drop_path=0., stem_ks=4, block_ks=7, expansion=4, use_glu=False):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=stem_ks, stride=stem_ks),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i], drop_path=dp_rates[cur + j], kernel_size=block_ks, expansion=expansion, use_glu=use_glu)
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_class)
        self._init_weights()

    def forward(self, x):
        for down, stage in zip(self.downsample_layers, self.stages):
            x = down(x)
            x = stage(x)

        x = x.mean([-2, -1])
        return self.head(self.norm(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
        
        nn.init.zeros_(self.head.weight)