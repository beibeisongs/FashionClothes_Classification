# encoding=utf-8
# Date: 2018-10-14
# Author: MJUZY
# Important Referrence:
# https://blog.csdn.net/zpalyq110/article/details/80432827
# https://github.com/Freemanzxp/Image-category-understanding-and-application/blob/master/main/MedicalLargeFine_tuning.py


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
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical')


from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense
from keras import optimizers


model = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels.h5',
              include_top=True)

model.layers.pop()
model.layers.pop()
model.layers.pop()

model.outputs = [model.layers[-1].output]
x = Dense(256, activation='relu')(model.layers[-1].output)
x = Dense(8, activation='softmax')(x)

model = Model(model.input, x)

for layer in model.layers[:15]:
    layer.trainable = False

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
    loss='categorical_crossentropy',
    metrics=['acc'])

model.summary()

history = model.fit_generator(
                    train_generator,
                    steps_per_epoch=100,  # Stands for the total times of the training loop
                    epochs=30,
                    validation_data=validation_generator,
                    validation_steps=50)

model.save('VGG16FineTune_8_Clothes_Classes.h5')

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
