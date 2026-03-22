import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

def get_param_groups(model, weight_decay):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 1:
            no_decay.append(param)
        elif 'embed' in name or 'token' in name:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]
    return param_groups

class EMA(WeightAveraging):
    def __init__(self, decay=0.999, start_step=500, use_buffers=False):
        super().__init__(avg_fn=get_ema_avg_fn(decay), use_buffers=use_buffers)
        self.start_step = start_step

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= self.start_step)