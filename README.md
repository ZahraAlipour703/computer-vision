🤖 Computer Vision & 3D Reconstruction Projects

Welcome to the Computer Vision Projects repository! This collection showcases two distinct, yet complementary, projects: a set of fundamental ArUco Marker tools and an advanced implementation of a Neural Radiance Field (NeRF) model, inspired by NVIDIA's Instant-NGP.

This repository is designed for AI researchers, developers, and enthusiasts interested in practical code for advanced 2D and 3D computer vision applications.
💡 Repository Overview
Project	Focus	Technology	Key Feature
1. ArUco Tools	2D Computer Vision	OpenCV	Robust marker generation with fallbacks.
2. TinyNeRF (Instant-NGP)	3D Scene Reconstruction	PyTorch	Learn 3D scenes from 2D images for novel-view synthesis.
1️⃣ ArUco Marker Generation Module

This module provides tools for generating custom ArUco markers, which are essential for camera calibration, pose estimation, and augmented reality applications.
🌟 Features

    Robust Generation: Supports various ArUco dictionaries (e.g., DICT_4X4_50, DICT_5X5_100).

    Custom Fallback: Includes a robust fallback implementation for drawing markers if the native cv2.aruco.drawMarker function is unavailable.

    User-Configurable: Easy control over marker size, border width, and output file format.

⚙️ Usage Example

The main script is Aruco/Aruco-Marker-Generation.py.
Bash

# Example: Generate marker ID 23 from dictionary DICT_4X4_50 at 300 pixels
python Aruco/Aruco-Marker-Generation.py --dict DICT_4X4_50 --id 23 --size 300 --output marker_23.png

2️⃣ TinyNeRF (Instant-NGP Inspired)

This project is a lightweight, educational reimplementation of the core concepts behind Neural Radiance Fields (NeRF), focusing on how a simple network can learn a full 3D scene from 2D images.
🧠 How It Works

    Data Encoding: 3D coordinates and viewing directions are encoded (e.g., using positional encoding) to capture fine spatial detail.

    TinyNeRF Model (MLP): A small Multi-Layer Perceptron (MLP) takes these encoded features and predicts Density (σ) and Color (RGB) for a given 3D point.

    Volume Rendering: For every pixel ray, predicted densities and colors along the ray are integrated to calculate the final pixel color, enabling photorealistic rendering of unseen views.

🚀 Key Features

    Modular structure built with PyTorch.

    Lightweight TinyNeRF MLP backbone.

    Simple training & rendering command-line interface.

    Designed to be extensible to more advanced techniques like HashGrid Encoding and Occupancy Grid Sampling.

💻 Usage
Mode	Command	Description
Train	python main.py (and select train)	Trains the NeRF model on input images/poses.
Render	python main.py (and select render)	Renders a novel view using the trained model, saved to outputs/render.png.
🛠️ Installation & Setup
Prerequisites

    Python 3.6+

    OpenCV with Contrib Modules (opencv-contrib-python)

    NumPy

    PyTorch (for the NeRF project)

Setup Steps

    Clone the Repository
    Bash

git clone https://github.com/ZahraAlipour703/computer-vision
cd computer-vision

Create and Activate Virtual Environment
Bash

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

Install Dependencies
Bash

    # Install common CV and NeRF dependencies
    pip install opencv-contrib-python numpy torch
    # Note: Depending on your specific TinyNeRF requirements, you may need a requirements.txt file.

🗂️ Consolidated Folder Structure

computer-vision/
├── Aruco/
│   └── Aruco-Marker-Generation.py    # Main script for ArUco marker generation
│
├── ngp/                                # Instant-NGP / TinyNeRF Project Directory
│   ├── main.py                         # Entry point (train or render)
│   ├── src/                            # Core components
│   │   ├── hash_encoder.py             # Positional or hash encoding
│   │   ├── tiny_nerf.py                # Core NeRF model (MLP)
│   │   ├── train.py                    # Training loop
│   │   └── utils.py                    # Helper utilities
│   ├── data/                           # Input images and camera poses
│   └── outputs/                        # Trained models and rendered images
│
└── README.md                           # This file

📈 Future Improvements

    Implement multi-resolution hash encoding (Instant-NGP style).

    Add occupancy grid sampling for faster convergence in the NeRF model.

    Support standard NeRF datasets (e.g., LLFF, Blender scenes).

    Add camera pose optimization for uncalibrated images in the NeRF pipeline.

✍️ Author

Zahra Alipour

    Email: zahraalipour.ac@gmail.com
