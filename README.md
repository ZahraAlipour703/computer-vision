<<<<<<< HEAD
# Computer Vision Projects

Welcome to the **Computer Vision Projects** repository! This project showcases a collection of computer vision tools and scripts, including an **ArUco Marker Generation** module with robust fallback implementations. Whether you're an AI researcher, a developer, or a hobbyist, this repository offers practical code and guidance for building advanced computer vision applications.


## Features

- **Robust ArUco Marker Generation:**  
  - Supports various dictionaries (e.g., `DICT_4X4_50`, `DICT_5X5_100`, etc.).
  - Custom fallback implementation if `cv2.aruco.drawMarker` is unavailable.
  - User-configurable marker size, border width, and output file format.

- **Modular Code Structure:**  
  - Easy-to-read and well-documented code.
  - Command-line interface for simple integration into your workflows.

- **Extensible Repository:**  
  - Designed for continuous growth with more computer vision projects and tools.

## Installation

### Prerequisites

- Python 3.6+
- [OpenCV with Contrib Modules](https://pypi.org/project/opencv-contrib-python/)
- [NumPy](https://numpy.org/)

### Setup

1. **Clone the Repository:**    ```bash
   git clone https://github.com/ZahraAlipour703/computer-vision
2. Install Required Python Packages: It's recommended to use a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install opencv-contrib-python numpy
Usage
ArUco Marker Generation
The Aruco-Marker-Generation.py script allows you to generate custom ArUco markers.

Command-Line Arguments
--dict: (Required) Specify the ArUco dictionary type (e.g., DICT_4X4_50).

--id: (Required) Marker ID (must be within the range supported by the chosen dictionary).

--size: (Optional) Size of the marker image in pixels (default: 200).

--output: (Optional) Output filename for the marker image (default: aruco_marker.png).

--border: (Optional) Border width in pixels (default: 1).
Example :
python Aruco-Marker-Generation.py --dict DICT_4X4_50 --id 23 --size 300 --output marker_23.png
##Folder Structure:##
├── Aruco
│   └── Aruco-Marker-Generation.py   # Main script for generating ArUco markers
├── README.md                        # This file





=======
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
>>>>>>> npg
