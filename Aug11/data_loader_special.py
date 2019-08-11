import pickle
import gzip
# from sklearn.model_selection import train_test_split
import numpy as np

def load_data_wrapper():
    f = open('./DATA/8by8lattices.pkl', 'rb')
    if (f.read(2) == '\x1f\x8b'):
        f.seek(0)
        return gzip.GzipFile(fileobj=f)
    else:
        f.seek(0)
    training_inputs = pickle.load(f, encoding="latin1")
    # print(np.array(training_inputs).shape) # 32,10000,8,8
    training_inputs = np.reshape(training_inputs[0],(10000, 8,8,1))
    return training_inputs
