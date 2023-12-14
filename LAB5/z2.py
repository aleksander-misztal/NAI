"""
This script demonstrates the implementation of a Convolutional Neural Network (CNN)
for image classification on an Animals dataset.

The dataset is loaded using the TensorFlow ImageDataGenerator, and a CNN model is created.
The model is compiled, and training is performed using data augmentation and normalization.
Callbacks such as EarlyStopping and ModelCheckpoint are used during training.

Note: The dataset should be organized in subdirectories under 'data/Animals' with one subdirectory per class.

"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Define constants
batch_size = 32
img_height = 150
img_width = 150
epochs = 10

# Count the number of subdirectories in 'data/Animals'
num_classes = len(os.listdir('data/Animals'))

# Create a CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory('data/Animals',
                                                    target_size=(img_height, img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory('data/Animals',
                                                         target_size=(img_height, img_width),
                                                         batch_size=batch_size,
                                                         class_mode='categorical',
                                                         subset='validation')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_animal_classifier_model.h5', save_best_only=True)

# Train the model
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=validation_generator,
                    callbacks=[early_stopping, model_checkpoint])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(validation_generator)
print("Test Accuracy:", test_accuracy)

# Save the final model
model.save('final_animal_classifier_model.h5')
