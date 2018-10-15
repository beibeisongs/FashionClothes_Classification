# encoding=utf-8
# Date: 2018-10-14
# Author: MJUZY
# Important Reference: https://blog.csdn.net/JinbaoSite/article/details/77435558


from keras.applications import VGG16

""" Download weights URL:

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.1/'
                'vgg16_weights_tf_dim_ordering_tf_kernels.h5')

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
"""

conv_base = VGG16(
    weights="D:/神经网络资料：Python/Learning_Process_Keras/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",    # weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

conv_base.summary()


import os
from keras.preprocessing.image import ImageDataGenerator
from Prepare import base_dir


train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical')


from keras import models
from keras import layers
from keras import optimizers


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))  # adjust the shape of the Array to the shape of the final feature map
model.add(layers.Dense(8, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=100,  # Stands for the total times of the training loop
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=50)

model.save('VGG_8_Clothes_Classes.h5')

import matplotlib.pyplot as plt


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
