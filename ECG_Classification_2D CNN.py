import numpy as np
import keras, os
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization, ELU
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.preprocessing import image


#tf.test.gpu_device_name()

# ECG image Dataset directly  downloaded from https://www.kaggle.com/erhmrai/ecg-image-data


batch_size = 32

IMAGE_SIZE = [128, 128]  # size of image input to the CNN network
train_image_number = 99199 # number of images used for training the network
 
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("D:/Dataset_ECG/train/",
                                          target_size=IMAGE_SIZE,
                                          batch_size = 32,
                                          shuffle=True)

test_dataset = test.flow_from_directory("D:/Dataset_ECG/test/",
                                          target_size=IMAGE_SIZE,
                                          batch_size =32,
                                          shuffle=True)


 # CNN model created using  tensorflow library. The architecture of CNN model is from the paper "The ECG arrhythmia classification using a 2-D convolutional neural network"

model = Sequential()    
model.add(Conv2D(64, (3,3),strides = (1,1), input_shape = IMAGE_SIZE + [3],kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(64, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))

model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(128, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))

model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Conv2D(256, (3,3),strides = (1,1),kernel_initializer='glorot_uniform'))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))

model.add(Flatten())

model.add(Dense(2048))

model.add(keras.layers.ELU())

model.add(BatchNormalization())

model.add(Dropout(0.5))

model.add(Dense(6, activation='softmax'))  # 6 different classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
steps_per_epoch = train_image_number/ batch_size

model.fit_generator(train_dataset,
         steps_per_epoch,
         epochs = 10,
         validation_data = test_dataset)

model.save('D:/Dataset_ECG')     
model.save('ECGModel.h5')