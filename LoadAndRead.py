import os
import gzip
import shutil
from typing import BinaryIO

import numpy as np

"""
First of all, we should download and read files
First we downloaded MNIST dataset with training set images that contained 60,000 examples 
Secondly, we should unzip the data, read it and than provide some sort of EDA before analysis
"""

# UNZIP the file, than we`ll receive the ubyte file
PATH = 'E:/Pattern Recognition/DigitRecognizer/Digit-Recognizer/data/'
train = 'train-images-idx3-ubyte.gz'
labels = 'train-labels-idx3-ubyte.gz'


def unzip_file(path, prefix):# unzip the file and return new value

    with gzip.open(path+prefix, 'rb') as out:
        f = out.read()
        out.close()

    return f

images = unzip_file(PATH, train)

print(type(images))


# Once we have byte files we can move further with transforming ubtes into readible format e.g. numpy array



def load_ubyte(data):
    intype = np.dtype('int32').newbyteorder('>')
    nbytes = 4 * intype.itemsize
    data = np.fromfile(data, dtype='ubyte')

    magicBytes, nImages, width, height = np.frombuffer(data[:nbytes].tobytes(), intype)

    data = data[nbytes:].astype(dtype='int32').reshape([nImages, width, height])

    return data


imagesArr = load_ubyte(images)
print(imagesArr.size)