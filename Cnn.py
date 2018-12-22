import random
import numpy as np

# Third-party libraries

class Network(object):

    def __init__(self):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        # method =[   [ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 5, 5),   poolsize=(2, 2), activation_fn=ReLU)],
        #             [ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5), poolsize=(2, 2), activation_fn=ReLU)],
        #              FullyConnectedLayer( n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        # FullyConnectedLayer(
        #     n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        # SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        # mini_batch_size)]

        self.sizes = [16,20,10]
        self.num_layers = len(self.sizes)
        self.mini_batch_size = 1

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.filters = [[np.random.randn(1,1 ,5,5)],[np.random.randn(1, 1, 5, 5)]]

        self.activation_Conv1 = np.zeros((self.mini_batch_size, 1, 12, 12))
        self.activation_Conv2 = np.zeros((self.mini_batch_size, 1, 4 , 4 ))

    def feedforward(self, image):
        """Return the output of the network if ``a`` is input."""
        # imgage.shape = (mini_batch_size, 1, 28, 28)
        q = 0
        for img in image:
            # img.shape = (1, 28, 28)
            self.activation_Conv1[q] = self.convolution(        img             ,(1,28,28),(1,1,5,5),(1,24,24),2,0,(self.mini_batch_size, 1, 12, 12))
            self.activation_Conv2[q] = self.convolution(self.activation_Conv1[q],(1,12,12),(1,1,5,5),(1,8 ,8 ),2,1,(self.mini_batch_size, 1, 4 , 4 ))
            a = np.reshape(self.activation_Conv2,(16,1))
            print(a)
            q = q + 1
            for b, w in zip(self.biases, self.weights):
                print(a.shape)
                a = np.dot(w, a)+b
                a[a < 0] = 0
        print(a)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # if test_data:
            #     print "Epoch {0}: {1} / {2}".format(
            #         j, self.evaluate(test_data), n_test)
            # else:
            #     print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
    def convolution(self,img,img_shape,filter_shape, output_size ,n,w,final_shape):
        newimg = np.zeros(output_size,dtype = np.float )
        a, b, u, x = 0, 0, filter_shape[-1]//2, 0
        for filter in self.filters[w]:
            for i in range(u,img_shape[-1]-u):
                for j in range(u,img_shape[-1]-u):
                    newimg[x,a,b] = max(0,np.sum(np.array(img[:,i-u:i+u+1,j-u:j+u+1])*filter))
                    b = b + 1
                b = 0
                a = a + 1
            a = 0
            x = x + 1
    #     return newimg
    # def maxpooling(self,img,img_shape,n): #at all layers simuntaneously
        newer  = np.zeros(final_shape[1:],dtype = np.float )
        for k in range(final_shape[1]):
            for i in range(0,output_size[1],n):
                for j in range(0,output_size[2],n):
                    # print(k,i,j)
                    newer[k,i//2,j//2] = np.max(img[k,i:i+n,j:j+n])
        return newer

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
if __name__ == '__main__':
    # arr = np.random.randn((3,8,8))
    img_shape = (1,1,28,28)
    img = np.random.randint(5,size = img_shape)
    N = Network()
    N.feedforward(img)
# filters = [np.random.randn(3,3,3)  for x in range(20)]
# img = [np.random.randn(8,8)  for x in range(3)]
# img_shape = (3,8,8)
# filter_shape = (20,3,3,3)
# img = np.random.randint(5,size = img_shape)
# filters = np.random.randint(5,size = filter_shape)
# def convolution(img,img_shape,filter_shape, output_size):
#     newimg = np.zeros(output_size,dtype = np.int16 )
#     a, b ,u,x = 0, 0, filter_shape[-1]//2,0
#     for filter in filters:
#         for i in range(u,img_shape[-1]-u):
#             for j in range(u,img_shape[-1]-u):
#                 # print(np.sum(np.array(img[:,i-u:i+u+1,j-u:j+u+1])*filter))
#                 newimg[x,a,b] = np.sum(np.array(img[:,i-u:i+u+1,j-u:j+u+1])*filter)
#                 b = b + 1
#             b = 0
#             a = a + 1
#         a = 0
#         x = x + 1
#     return newimg
# convolution(img,(3,8,8),filter_shape,(20,6,6))
