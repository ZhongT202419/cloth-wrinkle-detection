import os

rgb_image_path = r'E:\Clothing Wrinkles\Images\train_preprocessed\class1\rgb_image.jpg'
depth_image_path = r'E:\Clothing Wrinkles\Images\train_preprocessed\class1\depth_image.png'

# Check if the files exist
if not os.path.exists(rgb_image_path):
    print(f"RGB image not found at: {rgb_image_path}")

if not os.path.exists(depth_image_path):
    print(f"Depth image not found at: {depth_image_path}")


import os

# Path to your dataset
image_dir = r'E:\Clothing Wrinkles\Images\train_preprocessed\class1'

# List files in the directory
image_files = os.listdir(image_dir)

# Let's print out some of the files to ensure they are being read correctly
for i, file_name in enumerate(image_files[:10]):  # Only show first 10 for brevity
    print(f"File {i+1}: {file_name}")
