import glob
from PIL import Image

data_image = []

def load_data():
    for filename in glob.glob('data/*.jpg'):
        image = Image.open(filename)
        data_image.append(image)


def plot_data():
    for image in data_image:
        image.show()

def main():
    load_data()
    plot_data()

if __name__ == '__main__':
    main()