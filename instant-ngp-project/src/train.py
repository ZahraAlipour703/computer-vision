import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from .tiny_nerf import TinyNeRF
from .hash_encoder import HashEncoder
from .utils import get_rays, sample_points, volume_rendering


# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, n_rays=1024):
        super().__init__()
        self.n_rays = n_rays

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        rays_o = torch.randn(self.n_rays, 3)
        rays_d = torch.randn(self.n_rays, 3)
        target = torch.rand(self.n_rays, 3)
        return rays_o, rays_d, target

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    encoder = HashEncoder().to(device)
    model = TinyNeRF(input_dim=encoder.n_levels*encoder.n_features_per_level).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=1e-3)

    for epoch in range(100):
        for rays_o, rays_d, target in dataloader:
            rays_o, rays_d, target = rays_o.squeeze(0).to(device), rays_d.squeeze(0).to(device), target.squeeze(0).to(device)
            
            points, t_vals = sample_points(rays_o, rays_d)
            points_flat = points.view(-1, 3)
            points_encoded = encoder(points_flat)
            sigma, rgb = model(points_encoded)
            sigma = sigma.view(points.shape[0], points.shape[1], 1)
            rgb = rgb.view(points.shape[0], points.shape[1], 3)

            deltas = t_vals[1:] - t_vals[:-1]
            deltas = torch.cat([deltas, deltas[:1]])
            pred = volume_rendering(sigma, rgb, deltas)

            loss = torch.mean((pred - target) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train()
