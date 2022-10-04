# function for resizing and cropping to 240, 240
from PIL import Image
import os


def resizeAndCrop(imgPath):
    imgPath = os.path.join('images', imgPath)

    im = Image.open(imgPath)

    # resize
    im = im.resize((240, 240))
    os.remove(imgPath)
    # save

    im.save(imgPath)


for imgPath in os.listdir('images'):
    resizeAndCrop(imgPath)
