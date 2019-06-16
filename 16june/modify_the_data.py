import pickle
import gzip
# from sklearn.model_selection import train_test_split
import numpy as np

f = open('./DATA/8by8lattices.pkl', 'rb')
# f = open('./DATA/32by32lattices.pkl', 'rb')
if (f.read(2) == '\x1f\x8b'):
    f.seek(0)
    gzip.GzipFile(fileobj=f)
else:
    f.seek(0)
training_data = pickle.load(f, encoding="latin1")
training_inputs = np.reshape(training_data,(320000, 64))
training_inputs[training_inputs>0.5] = training_inputs[training_inputs>0.5] - 1
print(training_inputs[0])
output = open('shiftedlattices.pkl', 'wb')
pickle.dump(training_inputs, output)
output.close()
