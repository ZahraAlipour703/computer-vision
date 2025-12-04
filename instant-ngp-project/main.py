# main.py

# ---------------------------
# Headless matplotlib setup
# ---------------------------
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

# ---------------------------
# Standard imports
# ---------------------------
from src.train import train
from src.tiny_nerf import TinyNeRF
from src.hash_encoder import HashEncoder
import torch
import numpy as np

# ---------------------------
# Render function
# ---------------------------
def render_example():
    # Example: generate dummy image
    H, W = 400, 400
    pixel_image = np.random.rand(H, W, 3)  # Replace with your actual render

    # Save to file instead of plt.show()
    output_path = "render.png"
    plt.imsave(output_path, pixel_image)
    print(f"Render saved to {output_path}")

# ---------------------------
# Main entry
# ---------------------------
if __name__ == "__main__":
    mode = input("Enter mode (train/render): ").strip().lower()

    if mode == "train":
        train()  # your training function
    elif mode == "render":
        render_example()
    else:
        print("Invalid mode. Choose 'train' or 'render'.")
