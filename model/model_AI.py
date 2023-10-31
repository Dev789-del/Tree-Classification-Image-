import glob
from tkinter import *
from PIL import Image, ImageTk

data_image = []

def load_data():
    for filename in glob.glob('../images/*.jpg'):
        image = Image.open(filename)
        data_image.append(image)


def plot_data():
    for image in data_image:
        image.show()
        render = ImageTk.PhotoImage(image)
        img = Label(image=render)
        img.image = render
        img.place(x=0, y=0)

def main():
    load_data()
    plot_data()

if __name__ == '__main__':
    main()