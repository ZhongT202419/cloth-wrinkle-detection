import os
import shutil

# Paths to your train and validation datasets
train_dir = r"E:\Clothing Wrinkles\Images\train_preprocessed"
valid_dir = r"E:\Clothing Wrinkles\Images\valid_preprocessed"

# Create dummy class folders 'class1' for both train and validation directories
os.makedirs(os.path.join(train_dir, "class1"), exist_ok=True)
os.makedirs(os.path.join(valid_dir, "class1"), exist_ok=True)

# Move all images into the 'class1' subdirectory for training data
for file_name in os.listdir(train_dir):
    if file_name.endswith('.jpg'):  # Ensure only image files are moved
        shutil.move(os.path.join(train_dir, file_name), os.path.join(train_dir, "class1", file_name))

# Move all images into the 'class1' subdirectory for validation data
for file_name in os.listdir(valid_dir):
    if file_name.endswith('.jpg'):  # Ensure only image files are moved
        shutil.move(os.path.join(valid_dir, file_name), os.path.join(valid_dir, "class1", file_name))

print("All images have been moved to 'class1' subdirectories.")
