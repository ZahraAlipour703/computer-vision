import torch
import torch.nn as nn

class HashEncoder(nn.Module):
    def __init__(self, n_levels=16, n_features_per_level=2, log2_hashmap_size=19):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hashmap_size = 2 ** log2_hashmap_size
        
        # Feature tables for each level
        self.embeddings = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level) 
            for _ in range(n_levels)
        ])
        
        # Initialize embeddings
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, a=-1e-4, b=1e-4)
    
    def hash_function(self, coords, level):
        # Simple spatial hashing
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device)
        hashed = ((coords.long() * primes).sum(-1)) & (self.hashmap_size - 1)
        return hashed
    
    def forward(self, x):
        # x shape: [B, 3], range [0, 1]
        features = []
        for level in range(self.n_levels):
            scale = 2 ** level
            coords = torch.floor(x * scale)
            hashed = self.hash_function(coords, level)
            f = self.embeddings[level](hashed)
            features.append(f)
        return torch.cat(features, dim=-1)
