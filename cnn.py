#!/usr/bin/python3
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import logging
import time
from termcolor import colored
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import cv2
from skimage.filters import threshold_local

logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

trn_i = None
trn_l = None
val_i = None
val_l = None
model = None
model_path = "models/cnn.h5"
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
    i = (i - T).astype("uint8")*255
    img = Image.fromarray(i).resize((28,28))
    im2arr = 1-np.array(img)/255.0
    im2arr = im2arr.reshape(1,28,28,1)
    return im2arr

def loadDataset():
    global trn_i, trn_l, val_i, val_l
    mnist = datasets.mnist
    (trn_i, trn_l), (val_i, val_l) = mnist.load_data()
    trn_i = trn_i.reshape((60000, 28, 28, 1))
    val_i = val_i.reshape((10000, 28, 28, 1))
    trn_i, val_i = trn_i/255.0, val_i/255.0

def create_model():
    model = models.Sequential()
    # input layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # hidden layers
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # output layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def modelMetrics(data,labels,model):
    #confusion matrix
    pred = []
    x = model.predict(data)
    [pred.append(np.argmax(_)) for _ in x]
    cm = tf.math.confusion_matrix(labels,pred,num_classes=tf.cast(tf.constant(10,tf.int32),tf.int32))
    print(colored("Confusion Matrix:","yellow",attrs=['bold']))
    for row in cm:
        for cell in row:
            print(colored(" %5d "%cell,'yellow',attrs=['bold']),end="")
        print()

def train_validate(model,steps):
    global model_path, trn_i, trn_l, val_i, val_l
    # fit model
    start = time.time()
    history = model.fit(trn_i,trn_l,epochs=steps)
    stop = time.time()
    print(colored("Training time : %f min"%((stop-start)/60),'yellow',attrs=['bold']))
    # save entire model as hdf5
    model.save(model_path)
    # metrics
    acc = history.history['accuracy']
    loss = history.history['loss']
    # graph accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training Metrics')
    plt.xlabel('epoch')
    # graph loss
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.xlabel('epoch')
    plt.show()

def evaluate_model(data,labels,model):
    # model validation
    print(colored("Model Validation:",'yellow',attrs=['bold']))
    loss, acc = model.evaluate(data, labels)
    print(colored("Sparse Categotical Loss = %f"%(loss),'yellow',attrs=['bold']))
    print(colored("Labeling Accuracy = %f"%(acc),'yellow',attrs=['bold']))

def predict(model):
    global image_files
    getImages()
    for _ in image_files:
        x = convertImages(_)
        p = model.predict(x)
        title = ""
        if np.max(p) > 0.90:
            title = np.argmax(p)
            print(colored("%s,%d"%(_,np.argmax(p)),'yellow',attrs=['bold']))
        else:
            title = "∅"
            print(colored("%s,%s"%(_,"∅"),'yellow',attrs=['bold']))   
        plt.title(title)
        plt.imshow(np.squeeze(x),cmap='gray')
        plt.show()
        '''for i in p:
            for j in i:
                print("%.10f"%j)'''

# load dataset
loadDataset()
if "-t" in sys.argv:
    epochs = 0
    if sys.argv[sys.argv.index("-t")+1] is not None:
        try:
            epochs = int(sys.argv[sys.argv.index("-t")+1])
        except:
            epochs = 100
    # create model
    model = create_model()
    model.summary()
    # train model
    train_validate(model,epochs)

if "-p" in sys.argv:
    # fetch saved model
    model = models.load_model(model_path)
    model.summary()
    # evaluate model
    evaluate_model(val_i,val_l,model)
    # confusion matrix 
    modelMetrics(val_i,val_l,model)
    # predict
    predict(model)