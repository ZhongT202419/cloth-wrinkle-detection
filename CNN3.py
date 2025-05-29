import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple model
inputs = layers.Input(shape=(28, 28, 1))
x = layers.Flatten()(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images[..., None] / 255.0
test_images = test_images[..., None] / 255.0

# Define a simple generator
def data_generator(images, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    return dataset

train_gen = data_generator(train_images, train_labels)
test_gen = data_generator(test_images, test_labels)

# Print the type of model
print("Type of model:", type(model))

# Train the model
history = model.fit(
    train_gen,
    epochs=1,
    validation_data=test_gen,
    workers=4,
    use_multiprocessing=True
)
