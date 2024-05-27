
import os

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
import numpy as np

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#-----1. duplicate data-----

data_path = "dataset/"
base_dir = os.path.join(data_path)

IMAGE_SIZE = 224 #model input size
BATCH_SIZE = 64

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale = 1./255,
        validation_split=0.2)

train_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        subset='training')

val_generator = datagen.flow_from_directory(
        base_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size = BATCH_SIZE,
        subset='validation')

#visualisasi

for _ in range(5):
    img, label = train_generator.next()
    print(img.shape)   #  (1,256,256,3)
    plt.imshow(img[0])
    plt.show()
    
    
for image_batch,label_batch in train_generator:
    break
image_batch.shape, label_batch.shape

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
num_classes=2


model = Sequential()
 
#------------------------------------
# Conv Block 1: 32 Filters, MaxPool.
#------------------------------------
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=IMG_SHAPE))
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1)) 
#------------------------------------
# Conv Block 2: 64 Filters, MaxPool.
#------------------------------------
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the convolutional features.
#------------------------------------
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'],
             )

model.summary()
epochs = 20
history = model.fit(train_generator,
                    epochs=epochs,
                    validation_data=val_generator)
loss, acc = model.evaluate_generator(val_generator, verbose=1)
#start plotting here
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()