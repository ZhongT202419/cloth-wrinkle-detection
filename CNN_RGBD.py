import os
import numpy as np
import cv2
import random
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt  # Added import for plotting

# Function to load RGBD images (RGB + Depth)
def load_rgbd_image(rgb_path, depth_path=None):
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        raise FileNotFoundError(f"RGB image not found at path: {rgb_path}")
    
    if depth_path and os.path.exists(depth_path):
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_img is None:
            depth_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
    else:
        # Create a dummy depth map if depth image is missing
        depth_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)
    
    # Resize depth image to match RGB dimensions
    depth_img_resized = cv2.resize(depth_img, (rgb_img.shape[1], rgb_img.shape[0]))
    
    # Stack depth as the fourth channel to RGB
    rgbd_img = np.dstack((rgb_img, depth_img_resized))
    
    return rgbd_img

# Load the base VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False)

# Modify the first Conv2D layer to accept 4 channels (RGB + Depth)
input_shape = (224, 224, 4)
new_input = Input(shape=input_shape)

# Create the first Conv2D layer manually and keep a reference to it
conv1_layer = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='block1_conv1_modified')
x = conv1_layer(new_input)

# Set the weights for the first layer manually (initialize depth weights with zeros)
first_layer_weights = base_model.get_layer("block1_conv1").get_weights()

# Original weights shape: (3, 3, 3, 64)
# New weights shape should be: (3, 3, 4, 64)
# Initialize the weights for the depth channel as zeros
depth_weights = np.zeros_like(first_layer_weights[0][:, :, :1, :])
new_weights = np.concatenate([first_layer_weights[0], depth_weights], axis=2)

# Apply the new weights to the modified layer
conv1_layer.set_weights([new_weights, first_layer_weights[1]])

# Proceed with the rest of the VGG16 layers
# Use the outputs of the modified first layer as inputs to the rest of the model
previous_layer = x
for layer in base_model.layers[2:]:
    # Clone the layer to avoid conflicts
    layer_config = layer.get_config()
    cloned_layer = layer.__class__.from_config(layer_config)
    # Set weights to the cloned layer
    cloned_layer.build(previous_layer.shape)
    cloned_layer.set_weights(layer.get_weights())
    # Apply the cloned layer
    previous_layer = cloned_layer(previous_layer)

# Add custom layers for binary classification
x = Flatten()(previous_layer)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)  # Binary classification

# Final model
model = Model(inputs=new_input, outputs=output)

# Freeze the VGG16 layers except the custom ones
for layer in model.layers:
    if 'block' in layer.name:
        layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Custom data generator using Keras' Sequence class
class RGBDDataGenerator(Sequence):
    def __init__(self, image_dir, batch_size=32, shuffle=True):
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            and not f.startswith('.')  # Exclude hidden files
        ]
        if not self.image_files:
            print(f"No image files found in {image_dir}. Please check the directory and file extensions.")
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_files) / self.batch_size))
    
    def __getitem__(self, index):
        batch_files = self.image_files[index * self.batch_size:(index + 1) * self.batch_size]
        batch_rgbd = []
        batch_labels = []
        for file_name in batch_files:
            rgb_image_path = os.path.join(self.image_dir, file_name)
            # Modify depth image path as per your naming convention
            depth_image_path = os.path.splitext(rgb_image_path)[0] + '_depth.png'  # Adjust if needed
            rgbd_img = load_rgbd_image(rgb_image_path, depth_image_path)
            
            if rgbd_img is None:
                continue  # Skip if the image could not be loaded
            
            # Resize to the input size
            rgbd_img_resized = cv2.resize(rgbd_img, (224, 224))
            
            # Preprocess the RGB channels
            rgb_channels = rgbd_img_resized[..., :3].astype('float32')
            rgb_channels = preprocess_input(rgb_channels)
            
            # Normalize the depth channel
            depth_channel = rgbd_img_resized[..., 3].astype('float32') / 255.0
            depth_channel = np.expand_dims(depth_channel, axis=-1)
            
            # Combine the preprocessed RGB channels and depth channel
            rgbd_img_preprocessed = np.concatenate([rgb_channels, depth_channel], axis=-1)
            
            batch_rgbd.append(rgbd_img_preprocessed)
            
            # Assign label based on filename or some other logic
            label = 1 if 'wrinkle' in file_name.lower() else 0
            batch_labels.append(label)
        
        if not batch_rgbd:
            raise ValueError(f"No data found for batch {index}. Please check your data generator.")
        
        return np.array(batch_rgbd), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.image_files)

# Set training and validation directories
train_dir = r'E:\Clothing Wrinkles\Images\train_preprocessed\class1'
valid_dir = r'E:\Clothing Wrinkles\Images\valid_preprocessed\class1'

# Define training and validation generators
train_generator = RGBDDataGenerator(train_dir, batch_size=32)
valid_generator = RGBDDataGenerator(valid_dir, batch_size=32, shuffle=False)

# Debugging: Print the number of images found
print(f"Number of training images: {len(train_generator.image_files)}")
print(f"Number of validation images: {len(valid_generator.image_files)}")

# Test the data generator by fetching a batch
try:
    batch_rgbd, batch_labels = train_generator[0]
    print(f"Batch RGBD shape: {batch_rgbd.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
except ValueError as e:
    print(e)
    exit()

# Verify the model's input shape
print(f"Model input shape: {model.input_shape}")

# Ensure the data matches the model's expected input shape
if batch_rgbd.shape[1:] != model.input_shape[1:]:
    raise ValueError("Mismatch between model input shape and batch data shape.")

# Calculate steps per epoch
steps_per_epoch = len(train_generator)
validation_steps = len(valid_generator)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=6,  # Set epochs to 6
    validation_data=valid_generator,
    validation_steps=validation_steps
)

# Save the trained model
model.save('rgbd_wrinkle_detection_model.keras')

print("Model training completed.")

# Plot training & validation accuracy values
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('training_accuracy.png')  # Save the accuracy plot
plt.close()

# Plot training & validation loss values
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')  # Save the loss plot
plt.close()
