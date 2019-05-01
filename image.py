import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2
from skimage.filters import threshold_local

imagePath = os.getcwd()+'/testImages/images/'
lablePath = os.getcwd()+'testImages/labels'
image_files = []

def getImages():
    global image_files, imagePath
    for _ in os.listdir(imagePath):
        if os.path.isfile(imagePath+_):
            image_files.append(imagePath+_)

def convertImages(images):
    i = cv2.imread(images)
    i = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
    T = threshold_local(i,999,offset=10,method="gaussian")
    i = (T - i).astype("uint8")*255
    img = Image.fromarray(i).resize((28,28))
    im2arr = np.array(img)/255.0
    im2arr = im2arr.reshape(1,28,28,1)
    return im2arr

getImages()

for x in image_files:
    _ = convertImages(x)
    plt.title(x)
    plt.imshow(np.squeeze(_),cmap='gray')
    plt.show()