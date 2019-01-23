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
    
def load_array_from_bytes(folder, prefix, iflabel = False):
    
    dataType = np.dtype( 'int32' ).newbyteorder( '>' )
    
    if iflabel == False:
        
        nBytes = 4 * dataType.itemsize

        data = np.fromfile(folder+prefix, dtype='ubyte')
        magicBytes, numImages, w, h = np.frombuffer( data[:nBytes].tobytes(), dataType )
        data = data[nBytes:].astype( dtype = 'float32' ).reshape( [ numImages, w, h ] )
    
        return data
    
    else: 
        data = np.fromfile(folder+prefix, dtype='ubyte')[2 * dataType.itemsize:]
        
        return data




        

images = load_array_from_bytes(PATH, train) #images array
labs = load_array_from_bytes(PATH, labels, iflabel=True) #labels array

plt.imshow(images[1, :, :], cmap = 'gray')




















