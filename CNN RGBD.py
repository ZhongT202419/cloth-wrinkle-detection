from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a function to load RGBD images (created earlier)
def load_rgbd_image(rgb_path, depth_path=None):
    rgb_img = cv2.imread(rgb_path)

    if depth_path and os.path.exists(depth_path):
        depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    else:
        depth_img = np.zeros((rgb_img.shape[0], rgb_img.shape[1]), dtype=np.uint8)  # Dummy depth

    depth_img_resized = cv2.resize(depth_img, (rgb_img.shape[1], rgb_img.shape[0]))
    rgbd_img = np.dstack((rgb_img, depth_img_resized))

    return rgbd_img

# Modify the CNN input shape to accept 4 channels (RGB + Depth)
input_shape = (224, 224, 4)  # 4 channels for RGBD

# Load the base VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Build the new model on top of VGG16
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # For binary classification (wrinkles/no wrinkles)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Data Augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load the RGBD images into memory for training (assuming they are stored as .npy arrays or use your loader)
train_dir = r'E:\Clothing Wrinkles\Images\train_preprocessed'
valid_dir = r'E:\Clothing Wrinkles\Images\valid_preprocessed'

# Generator to load RGBD images (you may need to modify to match your setup)
def rgbd_data_generator(image_dir, batch_size=32):
    image_files = os.listdir(image_dir)
    while True:
        batch_rgbd = []
        batch_labels = []  # Assuming binary classification
        for file_name in image_files:
            if file_name.endswith('.jpg'):
                rgb_image_path = os.path.join(image_dir, file_name)
                depth_image_path = rgb_image_path.replace('.jpg', '.png')  # Assuming depth is PNG
                rgbd_img = load_rgbd_image(rgb_image_path, depth_image_path)

                # Resize to the input size (e.g., 224x224) and normalize
                rgbd_img_resized = cv2.resize(rgbd_img, (224, 224))
                batch_rgbd.append(rgbd_img_resized)

                # Assign label based on filename or some other logic
                label = 1 if 'wrinkle' in file_name else 0
                batch_labels.append(label)

                if len(batch_rgbd) == batch_size:
                    yield np.array(batch_rgbd), np.array(batch_labels)
                    batch_rgbd = []
                    batch_labels = []

# Train and validation generators
train_generator = rgbd_data_generator(train_dir, batch_size=32)
valid_generator = rgbd_data_generator(valid_dir, batch_size=32)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Modify based on your dataset size
    epochs=10,
    validation_data=valid_generator,
    validation_steps=50)

# Save the trained model
model.save('rgbd_wrinkle_detection_model.keras')

print("Model training completed.")




import matplotlib.pyplot as plt

# Plot training accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(10, 5))
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

