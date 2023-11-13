import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
import glob
from tkinter import *
from PIL import Image, ImageTk
class Tree_Model():  
    def load_image():
        #Function to load image
        data_image = []
        for filename in glob.glob('../images/*.jpg'):
            image = Image.open(filename)
            data_image.append(image)

    

    def load_data():
        #Function to load data
        new_data = []
        path_to_data = '../data/train/png'
        for filename in glob.glob(path_to_data + '/*.png'):
            image = cv2.imread(filename)
            image = cv2.resize(image, (100, 100))
            new_data.append(image)
        new_data = np.array(new_data)
        print("Data loaded successfully")


    def plot_image():
        #Function to plot image
        for image in data_image:
            image.show()
            render = ImageTk.PhotoImage(image)
            img = Label(image=render)
            img.image = render
            img.place(x=0, y=0)