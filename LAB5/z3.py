"""
This script demonstrates the implementation of basic and complex neural network models
for image classification on the Fashion MNIST dataset.

The Fashion MNIST dataset is loaded, normalized, and transformed into categorical labels.
Two neural network models are created: a basic model and a more complex model with two hidden layers.
Both models are compiled, trained, and evaluated on the test set.

Note: The script assumes that the necessary libraries are installed and the Fashion MNIST dataset
is available through Keras.

"""

import tensorflow as tf
from keras import layers, models
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# Load Fashion MNIST data
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize data (scale pixels to the range [0, 1])
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Transform labels into categorical form (one-hot encoding)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Basic Model
model_basic = models.Sequential()
model_basic.add(layers.Flatten(input_shape=(28, 28)))  # Flatten layer for flattening images
model_basic.add(layers.Dense(128, activation='relu'))  # Dense layer with ReLU activation
model_basic.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
model_basic.add(layers.Dense(10, activation='softmax'))  # Output layer with softmax activation

# Compile the basic model
model_basic.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# Train the basic model
model_basic.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2, verbose=False)

# Evaluate the accuracy of the basic model on the test set
test_loss_basic, test_acc_basic = model_basic.evaluate(test_images, test_labels)
print('Basic Model Test accuracy:', test_acc_basic)

# Complex Model with Two Hidden Layers
model_complex = models.Sequential()
model_complex.add(layers.Flatten(input_shape=(28, 28)))  # Flatten layer for flattening images
model_complex.add(layers.Dense(256, activation='relu'))  # First dense layer with ReLU activation
model_complex.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
model_complex.add(layers.Dense(128, activation='relu'))  # Second dense layer with ReLU activation
model_complex.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
model_complex.add(layers.Dense(10, activation='softmax'))  # Output layer with softmax activation

# Compile the complex model
model_complex.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

# Train the complex model
model_complex.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2, verbose=False)

# Evaluate the accuracy of the complex model on the test set
test_loss_complex, test_acc_complex = model_complex.evaluate(test_images, test_labels)
print('Complex Model Test accuracy:', test_acc_complex)
