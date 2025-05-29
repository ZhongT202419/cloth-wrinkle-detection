from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Define paths to preprocessed images
train_dir = r"E:\Clothing Wrinkles\Images\train_preprocessed"
valid_dir = r"E:\Clothing Wrinkles\Images\valid_preprocessed"

# Image dimensions
img_width, img_height = 224, 224  # ResNet/VGG16 input size

# Data Augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Rescale pixel values to [0, 1]
    rotation_range=40,  # Rotation augmentation
    width_shift_range=0.2,  # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    shear_range=0.2,  # Shear
    zoom_range=0.2,  # Zoom
    horizontal_flip=True,  # Horizontal flip
    fill_mode='nearest')  # Fill any pixels created after augmentations

# Validation data is only rescaled (no augmentation)
valid_datagen = ImageDataGenerator(rescale=1.0/255)

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')  # Change class_mode if you have multiple classes

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary')

# Load VGG16 model with pretrained weights (ImageNet), exclude the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers of VGG16, we will not train these layers
for layer in base_model.layers:
    layer.trainable = False

# Build the new model on top of VGG16
model = Sequential()
model.add(base_model)  # Add VGG16 as the base
model.add(Flatten())  # Flatten the output
model.add(Dense(256, activation='relu'))  # Fully connected layer
model.add(Dropout(0.5))  # Dropout for regularization
model.add(Dense(1, activation='sigmoid'))  # Output layer (binary classification)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Display model architecture
model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=25,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size)

# Save the trained model
model.save('wrinkle_detection_model.h5')

# Plot training accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

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
