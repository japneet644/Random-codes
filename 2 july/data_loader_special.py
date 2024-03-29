import pickle
import gzip
# from sklearn.model_selection import train_test_split
import numpy as np

def load_data_wrapper():
    f = open('./DATA/latticescomponents.pkl', 'rb')
    # f = open('./DATA/32by32lattices.pkl', 'rb')
    if (f.read(2) == '\x1f\x8b'):
        f.seek(0)
        return gzip.GzipFile(fileobj=f)
    else:
        f.seek(0)
    training_inputs = pickle.load(f, encoding="latin1")
    training_inputs = np.reshape(training_inputs,(320000, 128))
    return training_inputs
