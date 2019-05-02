#!/usr/bin/python3

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import logging
import time
from termcolor import colored
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from skimage.filters import threshold_local
from sklearn import svm, metrics, externals

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'

trn_i = []
trn_l = None
val_i = []
val_l = None
model = None
model_path = "models/svm.pkl"
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
    im2arr = np.squeeze(im2arr.reshape(1,28*28,1,1))
    return im2arr

def loadDataset():
    global trn_i, trn_l, val_i, val_l
    mnist = datasets.mnist
    (t_i, trn_l), (v_i, val_l) = mnist.load_data()
    t_i = t_i.reshape((60000, 28, 28, 1))
    v_i = v_i.reshape((10000, 28, 28, 1))
    t_i, v_i = t_i/255.0, v_i/255.0
    for i in range(0,60000):
        trn_i.append(np.squeeze(t_i[i]))
    for i in range(0,10000):
        val_i.append(np.squeeze(v_i[i]))
    trn_i = np.array(trn_i)
    val_i = np.array(val_i)
    trn_i = np.squeeze(trn_i.reshape((60000,28*28,1)))
    val_i = np.squeeze(val_i.reshape((10000,28*28,1)))

def create_model():
    #c=2,gamma=0.01,acc=0.9809,time=36.6 minutes
    return svm.SVC(C=3,kernel='rbf',gamma=0.01,cache_size=12000,probability=True)

def train_model(model):
    global trn_i, trn_l
    start = time.time()
    model.fit(trn_i,trn_l)
    stop = time.time()
    print(colored("Training time : %f min"%((stop-start)/60),'yellow',attrs=['bold']))
    externals.joblib.dump(model,model_path)

loadDataset()
if "-t" in sys.argv:
    # create model
    print(colored("Creating model",'yellow',attrs=['bold']))
    model = create_model()
    # train model
    print(colored("Training model",'yellow',attrs=['bold']))
    train_model(model)
    print("Predicting model")
    predict = model.predict(val_i)
    print( metrics.confusion_matrix(val_l, predict))