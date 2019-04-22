#!/usr/bin/python3
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

trn_i = None
trn_l = None
val_i = None
val_l = None
model = None
model_path = "models/cnn.h5"
imagePath = os.getcwd()+'/testImages/'
image_files = []

def getImages():
    global image_files, imagePath
    for _ in os.listdir(imagePath):
        if os.path.isfile(imagePath+_):
            image_files.append(imagePath+_)

def convertImages(images):
    img = Image.open(images).convert("L")
    img = np.resize(img, (28,28,1))
    im2arr = np.array(img)/255
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
    model.add(layers.Dense(11, activation='softmax'))
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def train_validate(model,steps):
    global model_path, trn_i, trn_l, val_i, val_l
    # fit and validate
    history = model.fit(trn_i,trn_l,epochs=steps,validation_data=(val_i,val_l))
    # save entire model as hdf5
    model.save(model_path)
    # metrics
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # graph accuracy
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training & Validation Metrics')
    plt.xlabel('epoch')
    # graph loss
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.xlabel('epoch')
    plt.show()
    # model validation
    val_loss, val_acc = model.evaluate(val_i, val_l)
    print("Validation Loss =",val_loss)
    print("Validation Accuracy=",val_acc)

def predict(model):
    global image_files
    getImages()
    for _ in image_files:
        x = convertImages(_)
        p = model.predict(x)
        if np.max(p) > 0.90:
            print(_,np.argmax(p))
        else:
            print(_,"NOTA")

if "-t" in sys.argv:
    epochs = 0
    if sys.argv[sys.argv.index("-t")+1] is not None:
        try:
            epochs = int(sys.argv[sys.argv.index("-t")+1])
        except:
            epochs = 100
    # load dataset
    loadDataset()
    # create model
    model = create_model()
    model.summary()
    # train model
    train_validate(model,epochs)

if "-p" in sys.argv:
    # fetch saved model
    model = models.load_model(model_path)
    # predict
    predict(model)