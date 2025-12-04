# src/utils.py
import torch

def sample_points(rays_o, rays_d, n_samples=64, near=2.0, far=6.0):
    """
    Sample points along rays
    """
    t_vals = torch.linspace(near, far, steps=n_samples, device=rays_o.device)
    points = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[None, :, None]
    return points, t_vals

def volume_rendering(sigma, rgb, deltas):
    """
    Render color from predicted sigma and rgb values
    """
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * deltas)
    transmittance = torch.cumprod(
        torch.cat([torch.ones((alpha.shape[0],1), device=alpha.device), 1-alpha + 1e-10], dim=1), dim=1
    )[:, :-1]
    weights = alpha * transmittance
    pixel_color = torch.sum(weights[:, :, None] * rgb, dim=1)
    return pixel_color

def get_rays(H, W, focal, c2w):
    """
    Simple pinhole camera rays
    """
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    i = i.t().float()
    j = j.t().float()
    dirs = torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], dim=-1)  # camera space
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], dim=-1)  # world space
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d
