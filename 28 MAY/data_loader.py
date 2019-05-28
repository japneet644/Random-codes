"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip
# from sklearn.model_selection import train_test_split
import numpy as np

def load_data_wrapper():
    f = open('./DATA/8by8modifiedlattices.pkl', 'rb')
    # f = open('./DATA/32by32lattices.pkl', 'rb')
    if (f.read(2) == '\x1f\x8b'):
        f.seek(0)
        return gzip.GzipFile(fileobj=f)
    else:
        f.seek(0)
    training_data = pickle.load(f, encoding="latin1")
    training_inputs = np.reshape(training_data,(320000, 64))  #for x in training_data
    # training_inputs = np.reshape(training_data,(320000, 32,32,1 ))
    #train = np.concatenate((training_inputs[:5000],training_inputs[-5000:]))
    return training_inputs#[:10000]#reshape(10000,8,8)##
