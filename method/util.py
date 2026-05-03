import torch
from copy import deepcopy


class EulerVar:
    def __init__(self, theta):
        self.theta = theta if isinstance(theta, torch.Tensor) else torch.tensor(theta, dtype=torch.float)
        self.real = torch.cos(self.theta)
        self.img = torch.sin(self.theta)

        if self.real**2 < 1e-6:
            self.real = torch.tensor(0, dtype=torch.float)

        if self.img**2 < 1e-6:
            self.img = torch.tensor(0, dtype=torch.float)

    def cumprod(self, n):
        return [EulerVar(i * self.theta.item()) for i in range(0, n)]

    def cumprod_square(self, n):
        return [EulerVar(2 * i * self.theta.item()) for i in range(0, n)]


def to_tensor(euler_vars):
    return torch.cat([torch.tensor([e.real, e.img], dtype=torch.float).unsqueeze(0) for e in euler_vars], dim=0)


def make_periodic_steps(num_timesteps):
    var = EulerVar((2 * torch.pi) / num_timesteps)
    euler_cumprod = var.cumprod(num_timesteps)
    euler_cumprod_square = var.cumprod_square(num_timesteps)
    proj = to_tensor(euler_cumprod)
    proj_square = to_tensor(euler_cumprod_square)

    return proj, proj_square
