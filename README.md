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





