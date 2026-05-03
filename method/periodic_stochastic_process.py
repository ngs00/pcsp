import torch
import torch.nn
import math
from scipy.special import erfinv
from method.util import *


class PeriodicStochProc(torch.nn.Module):
    def __init__(self, dim_latent, beta=0.99, num_timesteps=None):
        super(PeriodicStochProc, self).__init__()
        self.prd_error = 1 / math.sqrt(dim_latent)

        if num_timesteps is None:
            # if L = 4n for n = {1, 2, ...}.
            self.scale_real = self.prd_error / (4 * erfinv(beta))
            self.scale_img = self.prd_error / (4 * erfinv(beta))

            # Estimate L.
            self.num_timesteps = int(1 / (dim_latent * self.scale_real**2))
            if self.num_timesteps % 4 != 0:
                self.num_timesteps += 4 - (self.num_timesteps % 4)

            self.alpha_bar, self.alpha_bar_square = make_periodic_steps(self.num_timesteps)
        else:
            self.num_timesteps = num_timesteps
            self.alpha_bar, self.alpha_bar_square = make_periodic_steps(self.num_timesteps)
            self.scale_real = self.prd_error / (math.sqrt(2) * torch.sqrt(1 - self.alpha_bar_square[:, 0]) * erfinv(0.99))
            self.scale_img = self.prd_error / (math.sqrt(2) * torch.sqrt(1 - self.alpha_bar_square[:, 1]) * erfinv(0.99))
            self.scale_real[self.scale_real == torch.inf] = 1e-3
            self.scale_img[self.scale_img == torch.inf] = 1e-3

            self.scale_real = self.prd_error / (2 * erfinv(beta) * torch.max(torch.sqrt(2 * (1 - self.alpha_bar_square[:, 0]))))
            self.scale_img = self.prd_error / (2 * erfinv(beta) * torch.max(torch.sqrt(2 * (1 - self.alpha_bar_square[:, 1]))))

        self.std_real = torch.sqrt(0.5 * (1 - self.alpha_bar_square[:, 0]))
        self.std_img = torch.sqrt(0.5 * (1 - self.alpha_bar_square[:, 1]))
        self.alpha_bar = self.alpha_bar.cuda()
        self.alpha_bar_square = self.alpha_bar_square.cuda()

    def forward(self, g):
        _g = g.repeat_interleave(self.num_timesteps, dim=0)

        real = self.alpha_bar[:, 0].unsqueeze(1).repeat(g.shape[0], 1) * _g
        img = self.alpha_bar[:, 1].unsqueeze(1).repeat(g.shape[0], 1) * _g
        real = real.view(g.shape[0], self.num_timesteps, -1)
        img = img.view(g.shape[0], self.num_timesteps, -1)

        std_real = self.alpha_bar_square[:, 0].unsqueeze(0).unsqueeze(2).repeat(g.shape[0], 1, 1)
        std_real = torch.sqrt(0.5 * (1 - std_real)).repeat(1, 1, g.shape[1])
        # scale_coeff_real = self.scale_real.unsqueeze(0).repeat(g.shape[0], 1).unsqueeze(2)
        sample_real = real + self.scale_real * torch.randn_like(std_real) * std_real

        std_img = self.alpha_bar_square[:, 1].unsqueeze(0).unsqueeze(2).repeat(g.shape[0], 1, 1)
        std_img = torch.sqrt(0.5 * (1 - std_img)).repeat(1, 1, g.shape[1])
        # scale_coeff_img = self.scale_img.unsqueeze(0).repeat(g.shape[0], 1).unsqueeze(2)
        sample_img = img + self.scale_img * torch.randn_like(std_img) * std_img

        return sample_real, sample_img
