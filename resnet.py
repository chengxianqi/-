import torch
import torch.nn as nn
from components import DropPath

class Bottleneck(nn.Module):
    def __init__(self, in_dim, internal_dim, stride=1, drop_path=0., expansion=4):
        super().__init__()
        out_dim = internal_dim * expansion
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, internal_dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_dim)
        self.conv2 = nn.Conv2d(internal_dim, internal_dim, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(internal_dim)
        self.conv3 = nn.Conv2d(internal_dim, out_dim, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if stride == 1 and in_dim == out_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                *([nn.AvgPool2d(stride)] if stride > 1 else []),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
            )
        self.drop_path = DropPath(drop_path, inplace=True) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x_preact = self.relu(self.bn1(x))
        shortcut = self.shortcut(x_preact)
        x = self.conv1(x_preact)
        x = self.conv2(self.relu(self.bn2(x)))
        x = self.conv3(self.relu(self.bn3(x)))
        return shortcut + self.drop_path(x)


class ResNet(nn.Module):
    def __init__(self, in_chans, num_classes, depths=(3, 4, 6, 3), drop_path=0., stem_dim=64, internal_dims=(64, 128, 256, 512), block_expansion=4):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        mid_dim = stem_dim // 2
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, mid_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            self.act,
            nn.Conv2d(mid_dim, mid_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            self.act,
            nn.Conv2d(mid_dim, stem_dim, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        dp_rates = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        in_dim = stem_dim
        stages = []
        for depth, internal_dim, stride in zip(depths, internal_dims, (1, 2, 2, 2)):
            layers = []
            for i in range(depth):
                block = Bottleneck(
                    in_dim,
                    internal_dim,
                    stride=stride if i == 0 else 1,
                    drop_path=dp_rates[cur],
                    expansion = block_expansion
                )
                cur += 1
                layers.append(block)
                in_dim = internal_dim * block_expansion
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.norm = nn.BatchNorm2d(in_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_dim, num_classes)
        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.act(self.norm(x))
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.conv3.weight)
        
        nn.init.zeros_(self.fc.weight)