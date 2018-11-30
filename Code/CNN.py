
# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
#import keras
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score
import h5py



### Reproducibility ###
import os
import random as rn
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(29)  # For numpy numbers
rn.seed(29)   # For Python
tf.set_random_seed(29)    #For Tensorflow



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options))

K.set_session(sess)

K.tensorflow_backend._get_available_gpus()


train_path = "/home/ubuntu/Plant-Species1-Recognition/PlantImages/train/"
test_path = "/home/ubuntu/Plant-Species-Recognition/PlantImages/test/"
valid_path = "/home/ubuntu/Plant-Species-Recognition/PlantImages/validation/"


# train_path = "/home/ubuntu/Plant-Species-Recognition/PreprocessedPlantImages/train/"
# test_path = "/home/ubuntu/Plant-Species-Recognition/PreprocessedPlantImages/test/"
# valid_path = "/home/ubuntu/Plant-Species-Recognition/PreprocessedPlantImages/validation/"

#input_dim = 50
input_dim = 100
#input_dim = 200

def preprocess(image):
    norm_im = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    hsv = cv2.cvtColor(norm_im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 0, 0), (100, 255, 255))
    imask = mask > 0
    green = np.zeros_like(norm_im, np.uint8)
    green[imask] = norm_im[imask]
    medblur = cv2.medianBlur(green, 13)
    #return norm_im
    return green
    #return medblur




#Generates batches of tensor image data that images must be in
#F-f-d takes path and puts in batches of normalized data, one-hot encodes classes (class_mode argument)
train_batches = ImageDataGenerator(preprocessing_function = preprocess).flow_from_directory(train_path, target_size=(input_dim, input_dim), batch_size=100)
test_batches = ImageDataGenerator(preprocessing_function = preprocess).flow_from_directory(test_path, target_size=(input_dim, input_dim), batch_size=100)
valid_batches = ImageDataGenerator(preprocessing_function = preprocess).flow_from_directory(valid_path, target_size=(input_dim, input_dim), batch_size=100)

#preprocessing_function = preprocess

print(train_batches.classes)


#### Build and Train CNN ####


model = Sequential([])

model.add(Conv2D(32, (7, 7), input_shape=(input_dim, input_dim, 3), padding='same'))
model.add(BatchNormalization(momentum=0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (7, 7), padding='same'))
model.add(BatchNormalization(momentum=0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization(momentum=0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization(momentum=0.1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(32, (3, 3), padding='same'))
# model.add(BatchNormalization(momentum=0.1))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', name='fc_' + str(2)))
model.add(Dropout(0.1))
model.add(Dense(12, activation='softmax'))

model.summary()

#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True))
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')
callbacks_list = [earlystop]

model.fit_generator(train_batches, steps_per_epoch=325, validation_data = valid_batches, validation_steps=77, epochs=5, callbacks=callbacks_list, verbose=1)
os.system("gpustat")

model.save("11layerCNN-100x100-5epochs-Adam-Normalization.h5")
model.save_weights("11layerCNN-100x100-5epochs-Adam-Normalization-weights.h5")


# # #### Predicting ####
train_ims, train_labels = next(train_batches)
test_ims, test_labels = next(test_batches)
#
train_score = model.evaluate(train_ims, train_labels)
test_score = model.evaluate(test_ims, test_labels)
#
print(train_score)
print(test_score)
#
# # # First column, class is 1d 0 or 1
labels = []
for label in test_labels:
    lab = np.where(label == max(label))[0][0]
    labels.append(lab)



predictions = model.predict_generator(test_batches, steps = 109, verbose = 0)
print(predictions)

preds = []
for prediction in predictions:
    label = np.where(prediction == max(prediction))[0][0]
    preds.append(label)


cm = confusion_matrix(labels, preds)
print(cm)

acc = accuracy_score(labels, preds, normalize=True, sample_weight=None)
print(acc)


print(acc)