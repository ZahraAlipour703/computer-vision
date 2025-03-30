#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import sys
import os

def get_aruco_dictionary(dict_flag):
    """
    Retrieve the ArUco dictionary using available function names.
    Tries Dictionary_get first; if unavailable, falls back to getPredefinedDictionary.
    """
    if hasattr(aruco, "Dictionary_get"):
        return aruco.Dictionary_get(dict_flag)
    elif hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(dict_flag)
    else:
        raise RuntimeError("Neither Dictionary_get nor getPredefinedDictionary is available in cv2.aruco.")

def get_marker_bits(dictionary_name):
    """
    Infer the marker bits from the dictionary name.
    For dictionary names like "DICT_4X4_50", "DICT_5X5_100", etc., extract the first number.
    Defaults to 4 if unable to determine.
    """
    if dictionary_name.startswith("DICT_"):
        parts = dictionary_name.split("_")
        if len(parts) >= 2 and "X" in parts[1]:
            try:
                marker_bits = int(parts[1].split("X")[0])
                return marker_bits
            except ValueError:
                pass
    # Default marker bits if not inferred from the name
    return 4

def draw_marker_custom(dictionary, dictionary_name, marker_id, side_pixels, border_bits=1):
    """
    Custom implementation to draw an ArUco marker.
    
    Parameters:
        dictionary: The ArUco dictionary object.
        dictionary_name: The dictionary name string (to infer marker size in bits).
        marker_id: The ID of the marker.
        side_pixels: The desired image size in pixels.
        border_bits: The width (in bits/pixels) of the white border to add.
        
    Returns:
        marker_image: The generated marker as a numpy array.
    """
    # Infer the marker size (in bits) from the dictionary name
    marker_bits = get_marker_bits(dictionary_name)
    
    # Retrieve the raw marker bytes and unpack them into bits.
    marker_bytes = dictionary.bytesList[marker_id]
    bits = np.unpackbits(marker_bytes)[:marker_bits * marker_bits]
    marker_matrix = bits.reshape((marker_bits, marker_bits)).astype(np.uint8) * 255

    # Resize the marker to the desired pixel dimensions using nearest neighbor interpolation.
    marker_image = cv2.resize(marker_matrix, (side_pixels, side_pixels), interpolation=cv2.INTER_NEAREST)

    # Add a white border if requested.
    if border_bits > 0:
        marker_image = cv2.copyMakeBorder(marker_image, border_bits, border_bits, border_bits, border_bits,
                                            cv2.BORDER_CONSTANT, value=255)
    return marker_image

def generate_aruco_marker(dictionary_name, marker_id, marker_size, output_file, border_bits=1):
    """
    Generate an ArUco marker image and save it to a file.
    
    Parameters:
        dictionary_name (str): Key for the desired ArUco dictionary (e.g., 'DICT_4X4_50').
        marker_id (int): Marker ID to generate.
        marker_size (int): Desired output image size in pixels.
        output_file (str): Filename for the saved marker image.
        border_bits (int): Border width in pixels (default: 1).
    """
    # Supported dictionaries mapping
    ARUCO_DICT = {
        "DICT_4X4_50": aruco.DICT_4X4_50,
        "DICT_4X4_100": aruco.DICT_4X4_100,
        "DICT_4X4_250": aruco.DICT_4X4_250,
        "DICT_4X4_1000": aruco.DICT_4X4_1000,
        "DICT_5X5_50": aruco.DICT_5X5_50,
        "DICT_5X5_100": aruco.DICT_5X5_100,
        "DICT_5X5_250": aruco.DICT_5X5_250,
        "DICT_5X5_1000": aruco.DICT_5X5_1000,
        "DICT_6X6_50": aruco.DICT_6X6_50,
        "DICT_6X6_100": aruco.DICT_6X6_100,
        "DICT_6X6_250": aruco.DICT_6X6_250,
        "DICT_6X6_1000": aruco.DICT_6X6_1000,
        "DICT_7X7_50": aruco.DICT_7X7_50,
        "DICT_7X7_100": aruco.DICT_7X7_100,
        "DICT_7X7_250": aruco.DICT_7X7_250,
        "DICT_7X7_1000": aruco.DICT_7X7_1000,
        "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
        # AprilTag dictionaries (if supported by your OpenCV version)
        "DICT_APRILTAG_16h5": aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h10": aruco.DICT_APRILTAG_36h10,
        "DICT_APRILTAG_36h11": aruco.DICT_APRILTAG_36h11,
    }

    if dictionary_name not in ARUCO_DICT:
        raise ValueError("Invalid dictionary name. Available options: {}".format(", ".join(ARUCO_DICT.keys())))

    # Retrieve the dictionary using a compatible function.
    try:
        aruco_dict = get_aruco_dictionary(ARUCO_DICT[dictionary_name])
    except Exception as e:
        raise RuntimeError("Error retrieving dictionary: {}".format(e))

    # Validate marker ID against the dictionary.
    max_id = len(aruco_dict.bytesList) - 1
    if marker_id < 0 or marker_id > max_id:
        raise ValueError("Marker ID must be between 0 and {} for {}. Provided: {}"
                         .format(max_id, dictionary_name, marker_id))

    # Generate marker image: try using built-in drawMarker, otherwise use custom implementation.
    try:
        if hasattr(aruco, "drawMarker"):
            marker_image = aruco.drawMarker(aruco_dict, marker_id, marker_size, borderBits=border_bits)
        else:
            marker_image = draw_marker_custom(aruco_dict, dictionary_name, marker_id, marker_size, border_bits)
    except Exception as e:
        raise RuntimeError("Error generating marker image: {}".format(e))

    # Validate and adjust the output file extension.
    valid_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    root, ext = os.path.splitext(output_file)
    if ext.lower() not in valid_extensions:
        print(f"Warning: The file extension '{ext}' is not recognized. Using default '.png' extension instead.")
        output_file = root + '.png'

    # Save the marker image to file.
    try:
        if not cv2.imwrite(output_file, marker_image):
            raise IOError("Failed to write image to file: {}".format(output_file))
    except Exception as e:
        raise RuntimeError("Error saving marker image: {}".format(e))

    print("ArUco marker successfully generated and saved to:", output_file)

def main():
    parser = argparse.ArgumentParser(description="Generate an ArUco marker image.")
    parser.add_argument("--dict", type=str, required=True,
                        help="ArUco dictionary type (e.g., 'DICT_4X4_50')")
    parser.add_argument("--id", type=int, required=True,
                        help="Marker ID to generate (must be within the dictionary range)")
    parser.add_argument("--size", type=int, default=200,
                        help="Size of the marker image in pixels (default: 200)")
    parser.add_argument("--output", type=str, default="aruco_marker.png",
                        help="Output filename (default: 'aruco_marker.png')")
    parser.add_argument("--border", type=int, default=1,
                        help="Border width in pixels (default: 1)")
    args = parser.parse_args()

    try:
        generate_aruco_marker(args.dict, args.id, args.size, args.output, args.border)
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
