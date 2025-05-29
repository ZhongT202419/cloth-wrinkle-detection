import os
import cv2
import numpy as np

# Path to the dataset
image_dir = r'E:\Clothing Wrinkles\Images\train_preprocessed\class1'

# List all files in the directory
image_files = os.listdir(image_dir)

# Function to load RGB and create dummy depth if depth is missing
def load_rgbd_image(rgb_path, depth_path=None):
    # Load the RGB image
    rgb_img = cv2.imread(rgb_path)

    if rgb_img is None:
        print(f"Failed to load RGB image from: {rgb_path}")
        return None

    if depth_path and os.path.exists(depth_path):
        # Load depth image if available
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            print(f"Failed to load depth image from: {depth_path}")
            return None
        print(f"Loaded depth image from: {depth_path}")
    else:
        # If no depth image is found, create a dummy depth map
        depth_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
        print(f"Depth image not found, creating dummy depth map for: {rgb_path}")

    # Resize depth to match RGB dimensions (if needed)
    depth_img_resized = cv2.resize(depth_img, (rgb_img.shape[1], rgb_img.shape[0]))

    # Combine RGB and depth into a 4-channel image
    rgbd_img = np.dstack((rgb_img, depth_img_resized))

    return rgbd_img

# Iterate over all files and attempt to load RGB and corresponding depth images
for file_name in image_files:
    if file_name.endswith('.jpg'):  # Only process .jpg files
        rgb_image_path = os.path.join(image_dir, file_name)
        # Assume depth image has the same base name but with .png extension
        depth_image_path = rgb_image_path.replace('.jpg', '.png')

        rgbd_img = load_rgbd_image(rgb_image_path, depth_image_path)
        if rgbd_img is not None:
            print(f"Processed RGBD image from: {rgb_image_path}")
        else:
            print(f"Failed to process RGBD image for: {file_name}")
