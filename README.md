# 🧠 Instant-NGP (TinyNeRF Implementation)

A lightweight, educational reimplementation of **NVIDIA’s Instant Neural Graphics Primitives (Instant-NGP)** — focusing on **Neural Radiance Fields (NeRF)**.  
This project demonstrates how a neural network can **learn a 3D scene from 2D images** and **render novel views** from unseen camera angles.

---

## 🚀 Overview

This project builds a **TinyNeRF** model — a simple MLP network — that takes encoded 3D positions and view directions as input and predicts:
- **Density (σ)** → how much light is absorbed/scattered
- **Color (RGB)** → emitted color at that 3D point

By training on multiple images of a scene (like a checkerboard or object from various angles), it learns a full **3D radiance field**, allowing **photorealistic novel-view rendering**.

---

## 🧩 Features

✅ Modular structure with PyTorch  
✅ Lightweight **TinyNeRF** MLP backbone  
✅ Simple **training & rendering** interface  
✅ Extendable to **HashGrid Encoding** and **Occupancy Grid Sampling**  
✅ Inspired by **NVIDIA Instant-NGP** & **Google TinyNeRF**  

---

## 🗂️ Project Structure

instant-ngp-project/
│
├── main.py # Entry point (train or render)
├── src/
│ ├── hash_encoder.py # Positional or hash encoding
│ ├── tiny_nerf.py # Core NeRF model (MLP)
│ ├── train.py # Training loop and loss
│ └── utils.py # Helper utilities
│
├── data/ # Input images and camera poses
├── outputs/ # Trained models and rendered images
├── venv/ # Virtual environment
└── README.md # Project documentation

---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/instant-ngp-project.git
   cd instant-ngp-project
    cd instant-ngp-project
2. **Create virtual environment**
   ```bash
    python -m venv npgenv
    npgenv\Scripts\activate
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
## How it works
1️⃣ Data Encoding

3D coordinates and viewing directions are first encoded using a positional or hash encoder to capture fine spatial details.

2️⃣ TinyNeRF Model

A small MLP takes these encoded features and predicts:

(color_rgb, density_sigma) = TinyNeRF(encoded_xyz_dir)

3️⃣ Volume Rendering

For each camera ray (through a pixel), the network samples multiple points in 3D space and integrates the predicted colors and densities along that ray to compute the final pixel color.

4️⃣ Training

The model minimizes Mean Squared Error (MSE) between the rendered pixels and the ground truth image pixels.

5️⃣ Rendering

After training, the model can render the same scene from new camera viewpoints — effectively performing 3D reconstruction from 2D inputs.

## Usage
**🏋️ Train the model**
```bash
python main.py
```
When prompted:

Enter mode (train/render): train

**🎨 Render a scene**

After training:
```bash
python main.py
```

Then choose:

Enter mode (train/render): render


A rendered image will be saved as:

outputs/render.png

**🖼️ Example Results**
Input Views	Reconstructed Scene

The network reconstructs the 3D geometry and appearance from multiple 2D images.

**🔧 Future Improvements**

 Implement multi-resolution hash encoding (Instant-NGP style)

 Add occupancy grid sampling for faster convergence

 Support real NeRF datasets (e.g., LLFF, Blender scenes)

 Integrate interactive GUI rendering with Open3D or pythreejs

 Add camera pose optimization for uncalibrated images

**Author**

Developer: Zahra Alipour
📧 Email: zahraalipour.ac@gmail.com

Inspired by:

NVIDIA Instant-NGP (2022)

TinyNeRF (Google Research)
