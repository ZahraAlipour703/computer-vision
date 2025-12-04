import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyNeRF(nn.Module):
    def __init__(self, input_dim=60, hidden_dim=128, output_dim=4):
        """
        input_dim: dimension of the encoded input (after positional encoding)
        hidden_dim: neurons in each hidden layer
        output_dim: 4 (sigma + RGB)
        """
        super(TinyNeRF, self).__init__()
        
        # Small MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers
        self.fc_sigma = nn.Linear(hidden_dim, 1)   # density
        self.fc_rgb = nn.Linear(hidden_dim, 3)     # color

    def forward(self, x):
        # Pass through MLP with ReLU
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        # Predict density and color
        sigma = F.relu(self.fc_sigma(h))   # density should be >=0
        rgb = torch.sigmoid(self.fc_rgb(h))  # RGB in [0,1]
        
        return sigma, rgb

# Example usage
if __name__ == "__main__":
    x = torch.randn(5, 60)  # batch of 5 points with 60-dim positional encoding
    model = TinyNeRF()
    sigma, rgb = model(x)
    print("Sigma:", sigma)
    print("RGB:", rgb)
