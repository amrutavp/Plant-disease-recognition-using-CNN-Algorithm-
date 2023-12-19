# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:49:17 2022


"""

import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 25
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True,
                zoom_range=0.3,
                shear_range = 0.2    
)

train_generator = train_datagen.flow_from_directory(
        'D:\\CODE\\CODE\\APP\\Dataset\\train',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=32,
        class_mode='categorical',
)

validation_datagen = ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True,
                zoom_range=0.3,
                shear_range = 0.2   
)

validation_generator = validation_datagen.flow_from_directory(
        'D:\\CODE\\CODE\\APP\\Dataset\\val',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=32,
        class_mode='categorical'
)

test_datagen = ImageDataGenerator(
                rescale=1./255,
  
)

test_generator = test_datagen.flow_from_directory(
        'D:\\CODE\\CODE\\APP\\Dataset\\test',
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=32,
        class_mode='categorical'
)

input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 10 

model = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(32, kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    batch_size=32,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=25,
)

model.save("2Leaf_model.h5")

# accuracies
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.savefig('Maxpoolingaccuracies.png')
plt.show()

# loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.savefig('Maxpoolingloss.png')
plt.show()
