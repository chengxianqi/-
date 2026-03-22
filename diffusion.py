import torch
import lightning as L
import numpy as np
import tempfile
import os
from scipy.optimize import linear_sum_assignment
from torchvision.utils import make_grid, save_image
from torch.utils.data.dataloader import default_collate
from torch_fidelity import calculate_metrics
from tqdm import tqdm

class DiffusionSampler:
    def __init__(self, model, shape, pure_noise_fn, schedule_fn, d_fn):
        self.model = model
        self.shape = shape
        self.pure_noise_fn = pure_noise_fn
        self.schedule_fn = schedule_fn
        self.d_fn = d_fn

    def _cfg(self, x, t, c, guidance_scale):
        if guidance_scale == 1.0:
            return self.model(x, t, c).clone()

        uncond_c = torch.zeros_like(c)
        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        c_cat = torch.cat([c, uncond_c], dim=0)

        out = self.model(x_cat, t_cat, c_cat)
        out_cond, out_uncond = out.chunk(2, dim=0)
        return out_uncond + guidance_scale * (out_cond - out_uncond)

    def _step_euler(self, x, t, t_next, c, guidance_scale, get_d_fn):
        dt = (t_next - t).view(-1, 1, 1, 1)
        d = get_d_fn(x, t, self._cfg(x, t, c, guidance_scale))
        return x + d * dt

    def _step_heun(self, x, t, t_next, c, guidance_scale, get_d_fn):
        dt = (t_next - t).view(-1, 1, 1, 1)
        d = get_d_fn(x, t, self._cfg(x, t, c, guidance_scale))
        x_euler = x + d * dt
        d_next = get_d_fn(x_euler, t_next, self._cfg(x_euler, t_next, c, guidance_scale))
        return x + 0.5 * (d + d_next) * dt

    @torch.inference_mode()
    def sample(self, n, c=None, steps=20, guidance_scale=1.0, solver="heun", device='cuda'):
        self.model.eval().to(device)
        if c is not None:
            c = c.to(device)
            assert c.shape[0] == n
        if c is None: assert guidance_scale == 1.0

        xt = self.pure_noise_fn((n, *self.shape)).to(device)
        schedule = self.schedule_fn(steps).to(device)
        for i in range(steps):
            t = schedule[i].expand(n)
            t_next = schedule[i + 1].expand(n)
            if solver == "heun":
                xt = self._step_heun(xt, t, t_next, c, guidance_scale, self.d_fn)
            else:
                xt = self._step_euler(xt, t, t_next, c, guidance_scale, self.d_fn)
        return xt

class ImageLog(L.Callback):
    def __init__(self, every_n_epochs=10, cfg_scale=1.0):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.cfg_scale = cfg_scale

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n_epochs != 0:
            return

        if trainer.logger:
            num_class = pl_module.config.num_class
            c = torch.arange(num_class) + 1
            imgs = pl_module.sampler.sample(num_class, c=c, guidance_scale=self.cfg_scale, device=pl_module.device)
            imgs = (imgs.clamp(-1, 1) + 1) / 2.0
            grid = make_grid(imgs, nrow=5)
            trainer.logger.experiment.add_image(
                "val/samples_one_per_class",
                grid,
                global_step=trainer.global_step
            )
    

def ot_collate_fn(ot=False):
    def base_collate(batch):
        y, c = default_collate(batch)
        x0 = torch.randn_like(y)
        return x0, y, c

    def ot_collate(batch):
        x0, y, c = base_collate(batch)
        B = y.shape[0]
        x0_flat = x0.view(B, -1)
        y_flat = y.view(B, -1)
        C = torch.cdist(x0_flat, y_flat, p=2).numpy()
        row_ind, col_ind = linear_sum_assignment(C)
        x0_ot = torch.empty_like(x0)
        x0_ot[col_ind] = x0[row_ind]
        return x0_ot, y, c

    return ot_collate if ot else base_collate


class FlowMatching:
    def __init__(self, sig_min=1e-3):
        self.sig_min = sig_min
    
    def forward(self, model, x, t, c):
        return model(x, t * 1000, c)
    
    def loss(self, model, x0, y, c):
        t = torch.sigmoid(torch.randn(y.shape[0], 1, 1, 1, device=y.device))
        sig_t = 1 - (1 - self.sig_min) * t
        xt = sig_t * x0 + t * y
        ut = y - (1 - self.sig_min) * x0
        vt = model(xt, t.view(-1), c)
        loss = ((vt - ut).square()).mean(dim=(1,2,3))
        return loss.mean()
    
    def get_schedule(self, steps):
        return torch.linspace(0, 1, steps + 1)
    
    def get_pure_noise(self, shape):
        return torch.randn(shape)
    
    def get_d(self, x, t, x_hat):
        return x_hat


class EDM:
    def __init__(self, sig_data=0.5, p_mean=-1.2, p_std=1.2, sig_min=0.002, sig_max=80, rho=7):
        self.sig_data, self.p_mean, self.p_std = sig_data, p_mean, p_std
        self.sig_min, self.sig_max, self.rho = sig_min, sig_max, rho
        self.var_data = self.sig_data ** 2

    def forward(self, model, x, sig, c):
        sig = sig.view(-1, 1, 1, 1)
        var = sig.square()
        var_total = var + self.var_data
        c_skip = self.var_data / var_total
        c_out = sig * self.sig_data / var_total.sqrt()
        c_in = 1 / var_total.sqrt()
        c_noise = sig.log() / 4

        F_x = model((c_in * x), c_noise.view(-1), c)
        D_x = c_skip * x + c_out * F_x
        return D_x
    
    def loss(self, model, x0, y, c):
        log_normal = torch.randn(y.shape[0], device=y.device)
        sig = (log_normal * self.p_std + self.p_mean).exp().clamp(self.sig_min, self.sig_max)
        weight = (sig.square() + self.var_data) / (sig * self.sig_data).square()
        n = x0 * sig.view(-1, 1, 1, 1)
        D_yn = model(y + n, sig, c)
        loss = weight * (D_yn - y).square().mean(dim=(1,2,3))
        return loss.mean()

    def get_schedule(self, steps):
        ramp = torch.linspace(0, 1, steps + 1)
        min_inv_rho = self.sig_min ** (1 / self.rho)
        max_inv_rho = self.sig_max ** (1 / self.rho)
        sigs = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigs
    
    def get_pure_noise(self, shape):
        return torch.randn(shape) * self.sig_max
    
    def get_d(self, x, sig, x_hat):
        return (x - x_hat) / sig.view(-1, 1, 1, 1)

def fid_evaluate(litmodel, fid_samples=10000, fid_batch_size=1024, cfg_scale=2.0, device='cuda'):
    num_batches = fid_samples // fid_batch_size
    with tempfile.TemporaryDirectory() as temp_dir:
        for i in tqdm(range(num_batches), desc="Generating images"):
            c = torch.randint(1, litmodel.config.num_class + 1, (fid_batch_size,))
            imgs = litmodel.sampler.sample(fid_batch_size, c=c, guidance_scale=cfg_scale, device=device)
            imgs = (imgs.clamp(-1, 1) + 1) / 2.0
            for j in range(fid_batch_size):
                img_idx = i * fid_batch_size + j
                save_image(imgs[j], os.path.join(temp_dir, f"{img_idx:05d}.png"))

        metrics = calculate_metrics(
            input1='cifar10-train',
            input2=temp_dir,
            cuda=device == 'cuda',
            isc=True,
            fid=True,
            kid=True
        )