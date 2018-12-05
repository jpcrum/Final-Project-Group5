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
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.applications.vgg16 import decode_predictions
from keras import optimizers

#function to load data from given path
def load_dataset(path):
    data = load_files(path)
    doc_files = np.array(data['filenames'])
    doc_targets = np_utils.to_categorical(np.array(data['target']), 16)
    return doc_files, doc_targets

#Loading data
train_files, train_targets = load_dataset('/home/ubuntu/Deep-Learning/Project/Regional_whole/train')
valid_files, valid_targets = load_dataset('/home/ubuntu/Deep-Learning/Project/Regional_whole/valid')
test_files, test_targets = load_dataset('/home/ubuntu/Deep-Learning/Project/Regional_whole/test')
doc_names = [item[20:-1] for item in (glob("/home/ubuntu/Deep-Learning/Project/Regional_whole/train/*/"))]
#checking the dataset
print('There are %d total doc categories.' % len(doc_names))
print('There are %s total doc images.\n' % len(np.hstack([train_files, valid_files])))
print('There are %d training doc images.' % len(train_files))
print('There are %d validation doc images.' % len(valid_files))
print('There are %d test doc images.'% len(test_files))

#function that converts image into 4D array to facilitate Keras CNN
def convert_4darray(img_path):
    #loads image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    #convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    #convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)
#fucntion which converts all images in given path to 4D array for Keras CNN
def convert_4darrays(img_paths):
    list_of_tensors = [convert_4darray(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

#rescaling the images by dividing eveyr pixel in every image by 255 - preprocess data for Keras
train_tensors = convert_4darrays(train_files).astype('float32')/255
test_tensors = convert_4darrays(test_files).astype('float32')/255
valid_tensors = convert_4darrays(valid_files).astype('float32')/255


#model VGG-16
base_model = VGG16(weights = 'imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
#Flatten layer
x = Flatten()(x)
#softmax layer
predictions = Dense(16, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

#optimizer
#rmsprop= optimizers.RMSprop(lr=0.01, rho=0.9)
#compiling the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
EPOCHS=10
#model fit
model.fit(train_tensors, train_targets,validation_data=(valid_tensors, valid_targets),epochs=EPOCHS, batch_size=64,  verbose=1)
# get index of predicted document for each image in test set
predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
#test accuracy
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

