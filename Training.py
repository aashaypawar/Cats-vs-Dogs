from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import numpy as np

img_width = 150
img_height = 150

train_dir = 'train'
test_dir = 'test'
train_sample = 25000
test_sample = 12500
epoch = 100
batch_size = 100

if K.image_data_format() == 'channel_first':
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)

train_imgdatagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_imgdatagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_imgdatagen.flow_from_directory(
    train_dir,
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = 'binary')

test_generator = test_imgdatagen.flow_from_directory(
    test_dir,
    target_size = (img_width,img_height),
    batch_size = batch_size,
    class_mode = 'binary')

model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.summary()

model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_sample // batch_size,
    epochs=epoch,
    validation_data = test_generator,
    validation_steps = test_sample // batch_size)

model.save('new_model.h5')
