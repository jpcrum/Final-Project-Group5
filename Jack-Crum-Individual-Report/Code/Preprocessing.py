
# coding: utf-8

# In[1]:


import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, random
random.seed(29)


# In[2]:


random.seed(29)

def random_photo_per_class(path):
    random_images = []
    for root, dirs, files in os.walk(path):
        if root[-5:] != "train":
            image = random.choice(os.listdir("{}".format(root)))
            random_images.append(str(root)+os.sep+str(image))
    return random_images

random_photo_per_class("C:/Users/sjcrum/Documents/Machine Learning II/Final Project/DocumentImages/train")


# In[3]:


examples = random_photo_per_class("C:/Users/sjcrum/Documents/Machine Learning II/Final Project/DocumentImages/train")


# In[4]:


for image in examples:
    directory = os.path.dirname(image)
    doctype = directory.rsplit('\\', 1)[-1]
    img = cv2.imread(image, 0)

    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ))
    
    plt.subplot(111)
    plt.imshow(res, cmap='Greys_r')
    plt.title('{}'.format(doctype))
    plt.xticks([])
    plt.yticks([])

    plt.show()


# In[5]:


img = cv2.imread('C:/Users/sjcrum/Documents/Machine Learning II/Final Project/DocumentImages/train/advertisement/502599496.tif',0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max()/ cdf.max()

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]

hist2,bins2 = np.histogram(img2.flatten(),256,[0,256])

cdf2 = hist2.cumsum()
cdf_normalized2 = cdf2 * hist2.max()/ cdf2.max()

plt.subplot(221)
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

plt.subplot(222)
plt.imshow(img, cmap='Greys_r')

plt.subplot(223)
plt.plot(cdf_normalized2, color = 'b')
plt.hist(img2.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')

plt.subplot(224)
plt.imshow(img2, cmap='Greys_r')

plt.show()


# In[21]:


image = cv2.imread("C:/Users/sjcrum/Documents/Machine Learning II/Final Project/DocumentImages/test/advertisement/502607274+-7274.tif", 0)

plt.imshow(image, cmap='gray')
plt.show()


# In[15]:


height, width = img.shape


# In[25]:


header = img[0:(int(height*0.33)), 0:width]
footer = img[int(height*0.67):height, 0:width]
lbody = img[int(height*0.33):int(height*0.67), 0:int(width*0.5)]
rbody = img[int(height*0.33):int(height*0.67), int(width*0.5):width]


# In[26]:


plt.imshow(img, cmap = "gray")
plt.show()
plt.imshow(header, cmap = "gray")
plt.show()
plt.imshow(lbody, cmap = "gray")
plt.show()
plt.imshow(rbody, cmap = "gray")
plt.show()
plt.imshow(footer, cmap = "gray")
plt.show()


# In[ ]:


def create_regions(folder, dataset):
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            image_paths.append(str(root)+os.sep+str(file))
            
            
    path = 'C:/Users/sjcrum/Documents/Machine Learning II/Final Project/RegionalImages/{}/'.format(dataset)
    print(path)
    regions = ['whole', 'header', 'footer', 'left_body', 'right_body']
    classes = {'letter': '0', 'form': '1', 'email': '2', 'handwritten': '3', 'advertisement': '4', 'scientific report': '5', 
       'scientific publication': '6', 'specification': '7', 'file folder': '8', 'news article': '9', 'budget': '10', 
       'invoice': '11', 'presentation': '12', 'questionnaire': '13', 'resume': '14', 'memo': '15'}
    
    for region in regions:
        new_path = path + region
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for cl in classes.keys():
            class_path = new_path + '/' + cl
            if not os.path.exists(class_path):
                os.makedirs(class_path)
    
    for image in image_paths:
        img = cv2.imread(image,0)
        height = img.shape[0]
        width = img.shape[1]  
        split_im = image.split('\\', 1)
        print(split_im)
        cv2.imwrite('{}/whole/{}'.format(path, split_im[1]), img)
        header = img[0:(int(height*0.33)), 0:width]
        cv2.imwrite('{}/header/{}'.format(path, split_im[1]), header)
        footer = img[int(height*0.67):height, 0:width]
        cv2.imwrite('{}/footer/{}'.format(path, split_im[1]), footer)
        left_body = img[int(height*0.33):int(height*0.67), 0:int(width*0.5)]
        cv2.imwrite('{}/left_body/{}'.format(path, split_im[1]), left_body)
        right_body = img[int(height*0.33):int(height*0.67), int(width*0.5):width]
        cv2.imwrite('{}/right_body/{}'.format(path, split_im[1]), right_body)

create_regions("C:/Users/sjcrum/Documents/Machine Learning II/Final Project/DocumentImages/train", "train")
create_regions("C:/Users/sjcrum/Documents/Machine Learning II/Final Project/DocumentImages/valid", "valid")


# In[31]:



path = 'C:/Users/sjcrum/Documents/Machine Learning II/Final Project/RegionalImages/'
regions = ['whole', 'header', 'footer', 'left_body', 'right_body']
for region in regions:
    new_path = path + region
    if not os.path.exists(new_path):
        os.makedirs(new_path)

