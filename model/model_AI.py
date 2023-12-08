import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
import glob
from tkinter import *
from PIL import Image, ImageTk
from keras.preprocessing.image import ImageDataGenerator


class Tree_Model:  
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
        #Function to make train.csv
        train_data = []
        for filename in glob.glob('./data/train/**/*.png', recursive=True):
            image = cv2.imread(filename)
            image = cv2.resize(image, (70, 70))
            image = image.flatten()
            image = image.tolist()
            image.append(filename.split('/')[-2])
            train_data.append(image)
        print("Train data loaded successfully")
        train_data = pd.DataFrame(train_data)
        train_data.to_csv('./data/train.csv', index=False)
        print("Train.csv created successfully")

    def generate_tree_image():
        #Function to generate 10 tree data images from images folder into new_dataset folder
        data_image = []
        for filename in glob.glob('./images/*.jpg'):
            image = Image.open(filename)
            data_image.append(image)
        
        for i in range(10):
            for image in data_image:
                image = image.resize((70, 70))
                image = np.array(image)
                image = image.reshape((1, 70, 70, 3))
                datagen = ImageDataGenerator(
                    rotation_range=10,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    zoom_range=0.1,
                    horizontal_flip=True,
                    fill_mode='nearest')
                j = 0
                for batch in datagen.flow(image, batch_size=1, save_to_dir='./new_dataset', save_prefix='tree', save_format='jpg'):
                    j += 1
                    if j == 1:
                        break
        print("Tree images generated successfully")

