import torch
import torch.nn as nn
from components import AdaGN, MHA2d, TimestepEmbedding

class UnetBlock(nn.Module):
    def __init__(self, in_chans, out_chans, embed_chans, use_attn=False, up=False, down=False):
        super().__init__()
        self.use_attn = use_attn
        if up:
            self.updown = nn.Upsample(scale_factor=2.0, mode='nearest')
        elif down:
            self.updown = nn.AvgPool2d(2)
        else:
            self.updown = nn.Identity()
        self.shortcut = nn.Conv2d(in_chans, out_chans, 1) if in_chans != out_chans else nn.Identity()
        self.norm1 = nn.GroupNorm(32, in_chans)
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.norm2 = AdaGN(out_chans, embed_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        if use_attn:
            self.norm3 = nn.GroupNorm(32, out_chans)
            num_heads = out_chans // 64 if out_chans >= 64 else 1
            self.attn = MHA2d(out_chans, num_heads)

    def forward(self, x, emb):
        shortcut = self.shortcut(self.updown(x))
        x = self.conv1(self.updown(self.act(self.norm1(x))))
        x = self.conv2(self.act(self.norm2(x, emb)))
        x = shortcut + x
        return x + self.attn(self.norm3(x)) if self.use_attn else x


class DiffusionUNet(nn.Module):
    def __init__(self, in_chans, num_class=0, num_blocks=2, model_channels=64, channel_mult=(1, 2, 4), with_attn=(False, False, True), embed_dim_mult=4):
        super().__init__()
        embed_dim = embed_dim_mult * model_channels
        self.time_embed = TimestepEmbedding(model_channels, embed_dim)
        self.class_embed = nn.Embedding(num_class + 1, embed_dim)
        skip_chans = [model_channels]
        self.stem = nn.Conv2d(in_chans, model_channels, 3, padding=1)
        self.encoder = nn.ModuleList()
        for i, mult in enumerate(channel_mult):
            out_chans = model_channels * mult
            for _ in range(num_blocks):
                self.encoder.append(UnetBlock(skip_chans[-1], out_chans, embed_dim, use_attn=with_attn[i]))
                skip_chans.append(out_chans)
            if i != len(channel_mult) - 1:
                self.encoder.append(UnetBlock(out_chans, out_chans, embed_dim, down=True))
                skip_chans.append(out_chans)
        self.middle_block1 = UnetBlock(out_chans, out_chans, embed_dim, use_attn=True)
        self.middle_block2 = UnetBlock(out_chans, out_chans, embed_dim)
        self.decoder = nn.ModuleList()
        curr_chans = out_chans
        for i, mult in reversed(list(enumerate(channel_mult))):
            out_chans = model_channels * mult
            for j in range(num_blocks + 1):
                self.decoder.append(UnetBlock(skip_chans.pop() + curr_chans, out_chans, embed_dim, use_attn=with_attn[i]))
                if i != 0 and j == num_blocks:
                    self.decoder.append(UnetBlock(out_chans, out_chans, embed_dim, up=True))
                curr_chans = out_chans
        self.head = nn.Sequential(
            nn.GroupNorm(32, out_chans),
            nn.SiLU(),
            nn.Conv2d(out_chans, in_chans, kernel_size=3, padding=1)
        )
        self._init_weights()

    def forward(self, x, t, c):
        x = self.stem(x)
        emb = self.time_embed(t) + self.class_embed(c)
        skips = [x]
        for block in self.encoder:
            x = block(x, emb)
            skips.append(x)
        x = self.middle_block1(x, emb)
        x = self.middle_block2(x, emb)
        for block in self.decoder:
            if isinstance(block.updown, nn.Upsample):
                x = block(x, emb)
            else:
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, emb)
        return self.head(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
        
        for m in self.modules():
            if isinstance(m, UnetBlock):
                nn.init.zeros_(m.conv2.weight)
            elif isinstance(m, MHA2d):
                nn.init.zeros_(m.o_proj.weight)
        
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        nn.init.zeros_(self.head[-1].weight)