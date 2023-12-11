import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
import glob
from tkinter import *
from PIL import Image, ImageTk
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os
import shutil

class Tree_Data:  
    def load_image():
        #Function to load image
        data_image = []
        for filename in glob.glob('./images/*.jpg'):
            image = Image.open(filename)
            data_image.append(image)
        print("Image loaded successfully")
        
        for image in data_image:
            plt.imshow(image)
            plt.show()

    def train_image():
        #Show 3 train images in Charlock folder
        data_train = []
        for filename in glob.glob('./data/train/Charlock/*.png'):
            image1 = Image.open(filename)
            data_train.append(image1)
        print("Train data loaded successfully")
        for image in data_train[:3]:
            plt.imshow(image)
            plt.show()

    def test_data():
        #Show 3 test images
        data_test = []
        for filename in glob.glob('./data/test/*.png'):
            image2 = Image.open(filename)
            data_test.append(image2)
        print("Test data loaded successfully")
        for image in data_test[:3]:
            plt.imshow(image)
            plt.show()
    def make_train_csv():
        #Delete train csv file if it exists
        if os.path.exists('./model/train.csv'):
            os.remove('./model/train.csv')
        #Function to make train.csv from train folder in new_dataset folder with image_names and image_full_names   
        image_names = []
        image_labels = []
        #Get image names and tree names
        for filename in glob.glob('./new_dataset/train/*/*.png'):
            image_names.append(filename.split('\\')[-1])
            image_labels.append(filename.split('\\')[-2])
        #Create dataframe
        df = pd.DataFrame()
        df['image_names'] = image_names
        df['image_labels'] = image_labels
        df.to_csv('./model/train.csv', index=False)
    def generate_tree_image():
        #Delete all folders and files in new_dataset folder
        folder = './new_dataset/train/'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        #Function to generate 10 tree data images from images folder into train folder of new_dataset
        data_image = []
        for filename in glob.glob('./images/*.jpg'):
            image = Image.open(filename)
            data_image.append(image)
        image_names = ['orange_tree', 'Silver_birch']
        for i in range(10):
            for image , name in zip(data_image, image_names):
                image = image.resize((299, 299))
                image = np.array(image)
                image = image.reshape((1, 299, 299, 3))
                datagen = ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest')
                j = 0

                #Check if folder exists
                if not os.path.exists('./new_dataset/train/' + str(name)):
                    os.makedirs('./new_dataset/train/' + str(name))
                #Save the image with corresponding name and format
                for batch in datagen.flow(image, batch_size=1, save_to_dir='./new_dataset/train/' + str(name) + '/', save_prefix = name, save_format = 'png'):
                    j += 1
                    if j == 1:
                        break
#Set parameters
batch_size = 64
epochs = 30
num_classes = 10

def load_data():
    #Load data from train.csv
    train_data = pd.read_csv('./data/train.csv')
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    train_data.head()

    #Split data into train and test
    train = train_data[:3800]
    test = train_data[3800:]

    #Split data into X_train, y_train, X_test, y_test
    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values

    #Reshape data
    X_train = X_train.reshape(X_train.shape[0], 70, 70, 3)
    X_test = X_test.reshape(X_test.shape[0], 70, 70, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    #Normalize data
    X_train /= 255
    X_test /= 255

    #Convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return X_train, y_train, X_test, y_test

def build_model():
    #Build model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(299, 299, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    #Save model
    model.save('project_model.h5')
    return model

#Evaluate model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), shuffle=True)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

#Prediction 
def predict(model):
    #Load image
    image = Image.open('./images/1.jpg')
    image = image.resize((299, 299))
    image = np.array(image)
    image = image.reshape((1, 299, 299, 3))
    image = image.astype('float32')
    image /= 255

    #Predict image
    result = model.predict(image)
    result = np.argmax(result, axis=1)
    print(result)

    #Show image
    plt.imshow(image.reshape(299, 299, 3))
    plt.show()


