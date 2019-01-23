import gzip
import numpy as np
import matplotlib.pyplot as plt

"""
First of all, we should download and read files
First we downloaded MNIST dataset with training set images that contained 60,000 examples 
Secondly, we should unzip the data, read it and than provide some sort of EDA before analysis
"""

# UNZIP the file, than we`ll receive the ubyte file
PATH = 'E:/Pattern Recognition/DigitRecognizer/Digit-Recognizer/data/'
train = 'train-images-idx3-ubyte'
labels = 'train-labels-idx1-ubyte'


#load array of images and labels 
    
def load_array_from_bytes(folder, prefix):
    
    dataType = np.dtype( 'int32' ).newbyteorder( '>' )
    nBytes = 4 * dataType.itemsize

    data = np.fromfile(folder+prefix, dtype='ubyte')
    magicBytes, numImages, w, h = np.frombuffer( data[:nBytes].tobytes(), dataType )
    data = data[nBytes:].astype( dtype = 'float32' ).reshape( [ numImages, w, h ] )
    
    return data

images = load_array_from_bytes(PATH, train)
labs = load_array_from_bytes(PATH, labels)

plt.imshow(images[1, :, :], cmap = 'gray')



















