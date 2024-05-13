import numpy as np
import mnist
from tensorflow.keras.models import Sequential  # Using sequential and not model since this network is a linear stack
# of layers
from tensorflow.keras.layers import Dense  # Sequential constructor takes an array of Keras Layers. This is a standard
# feedforward network --> only need the Dense layer
from tensorflow.keras.utils import to_categorical

# Each image in MNIST dataset is 28x28 and has a centered greyscale digit.
# The goal is to flatten the images in to a 784 dimensional vector, which will then use as input to the neural network.
# The output will be one of 10 potential classes (0-9)

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalizing the images. Pixel values from [0,255] to [-0.5, 0.5], to make network easier to train
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model
# Last layer is a Softmax output layer with 10 nodes, one for each class
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

# Compile the model
model.compile(
    optimizer='adam',  # optimizer can be default
    loss='categorical_crossentropy',
    metrics=['accuracy'],  # classification, so metric can be accuracy
)

# Train the model.
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,  # iterations over the entire dataset
    batch_size=32,  # number of samples per gradient update
)

# Evaluate the model.
model.evaluate(
    test_images,
    to_categorical(test_labels)
)

# Save the model to disk.
model.save_weights('model.h5')

# Load the model from disk later
# model.load_weights('model.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1))

# Check our predictions
print(test_labels[:5])
