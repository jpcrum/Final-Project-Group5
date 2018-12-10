import os
import random
import pandas as pd
import numpy as np
import cv2

classes = {'letter': '0', 'form': '1', 'email': '2', 'handwritten': '3', 'advertisement': '4', 'scientific report': '5',
           'scientific publication': '6', 'specification': '7', 'file folder': '8', 'news article': '9', 'budget': '10',
           'invoice': '11', 'presentation': '12', 'questionnaire': '13', 'resume': '14', 'memo': '15'}

def extract_paths(path):
    image_filenames = []
    for root, dirs, files in os.walk(path):
        if len(files) > 0:
            for file in files:
                image_filenames.append(str(root)+os.sep+str(file))
    return image_filenames

filenames_test = extract_paths('/home/ubuntu/MachineLearningII/test/')
filenames_train = extract_paths('/home/ubuntu/MachineLearningII/train/')

def make_csv(files, dataset):
    images_and_labels = []
    for image in files:
        print(image)

        base = os.path.dirname(image).rsplit("/", 1)

        class_lab = base[1]
        label = classes[class_lab]
        image_and_label = [image, label]
        images_and_labels.append(image_and_label)

    df = pd.DataFrame(images_and_labels,columns=['image_paths','labels'])
    #df.to_csv('/home/ubuntu/MachineLearningII/{}_images_and_labels_1_image.csv'.format(dataset), index = False)

make_csv(filenames_test, 'test')
make_csv(filenames_train, 'train')

