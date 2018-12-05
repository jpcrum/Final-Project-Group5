#Packages
import numpy as np
from glob import glob
from tqdm import tqdm
import PIL
import cv2
from keras.utils import np_utils
from keras.preprocessing import image
from sklearn.datasets import load_files
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential

#Fucntion to load data
def load_dataset(path):
    data = load_files(path)
    doc_files = np.array(data['filenames'])
    doc_targets = np_utils.to_categorical(np.array(data['target']), 16)
    return doc_files, doc_targets

#Loading data from filepaths
train_files, train_targets = load_dataset('/home/ubuntu/Deep-Learning/Project/header data/Duplicate_train/header')
valid_files, valid_targets = load_dataset('/home/ubuntu/Deep-Learning/Project/header data/duplicate_valid/header')
test_files, test_targets = load_dataset('/home/ubuntu/Deep-Learning/Project/header data/Duplicate_test/header')
doc_names = [item[20:-1] for item in (glob("/home/ubuntu/Deep-Learning/Project/header data/Duplicate_train/header/*/"))]
#Check the dataset
print('There are %d total doc categories.' % len(doc_names))
print('There are %s total doc images.\n' % len(np.hstack([train_files, valid_files])))
print('There are %d training doc images.' % len(train_files))
print('There are %d validation doc images.' % len(valid_files))
print('There are %d test doc images.'% len(test_files))
#function that converts image into 4D array to facilitate Keras CNN
def convert_4darray(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
#function which converts all images in given path to 4D array for Keras CNN
def convert_4darrays(img_paths):
    list_of_tensors = [convert_4darray(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)
#pre-process the data for Keras
train_tensors = convert_4darrays(train_files).astype('float32')/255
test_tensors = convert_4darrays(test_files).astype('float32')/255
valid_tensors = convert_4darrays(valid_files).astype('float32')/255


#Model CNN
model = Sequential()
# Conv layer 1
model.add(Conv2D(32, (5, 5), strides=(1, 1), use_bias=False,padding='same', activation='relu', input_shape=(224, 224, 3)))
# max pooling layer 1
model.add(MaxPooling2D(pool_size=(3, 3), strides=3))

# Conv layer 2
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
# Mac pooling layer 2
model.add(MaxPooling2D(pool_size=(3, 3), strides=3))

# Conv layer 3
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# Max pooling layer 3
model.add(MaxPooling2D(pool_size=(3, 3), strides=3))

# Conv layer 4
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

# COnv layer 5
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
# Max pooling layer 4
model.add(MaxPooling2D(pool_size=(3, 3), strides=3))

# Flatten layer 1
model.add(Flatten())
model.add(Dense(32, activation='relu'))
# Flatter layer 2
model.add(Dense(32, activation='relu'))
# Predictions
model.add(Dense(16, activation='softmax'))

# Model summary
model.summary()

EPOCHS = 10
#model compile
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#Fititng model
model.fit(train_tensors, train_targets,validation_data=(valid_tensors, valid_targets),epochs=EPOCHS, batch_size=100,  verbose=2)
#get index of predicted document imagefor each image in test set
predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
#Test accuracy
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)