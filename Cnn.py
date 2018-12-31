import random
import numpy as np

# Third-party libraries
IMAGE_SHAPE = (1,28,28)

class Network(object):

    def __init__(self):
        # method =[   [ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), filter_shape=(20, 5, 5),   poolsize=(2, 2), activation_fn=ReLU)],
        #             [ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), filter_shape=(40, 20, 5, 5), poolsize=(2, 2), activation_fn=ReLU)],
        #              FullyConnectedLayer( n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        # FullyConnectedLayer(
        #     n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
        # SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)],
        # mini_batch_size)]

        self.sizes = [20*4*4,20,10]
        self.num_layers = len(self.sizes)
        # self.no_of_images = 1
        self.poolsize = 2
        self.img_shape = [(1,28,28),(5,12,12)]
        self.conv_out_shape = [(5,24,24),(10,8,8)]
        self.pool_out_shape = [(1,5,12,12),(1,10,4,4)]

        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

        self.filter_shape=[(5,1, 5, 5),(10, 10, 5, 5)]
        self.filters = [  [np.random.random_sample(self.filter_shape[0])] , [np.random.random_sample(self.filter_shape[1])]  ]
        self.fbiases = [  [np.random.random_sample(self.filter_shape[0][0])] , [np.random.random_sample(self.filter_shape[1][0])]  ]

        self.activation_Conv1 = np.zeros(self.pool_out_shape[0])
        self.activation_Conv2 = np.zeros(self.pool_out_shape[1])

        self.conv1 = np.zeros(self.conv_out_shape[0])
        self.conv2 = np.zeros(self.conv_out_shape[1])

    def feedforward(self, image):
        """Return the output of the network if ``a`` is input."""
        # imgage.shape = ( 1, 28, 28)
        # img.shape = (1, 28, 28)
        img = np.reshape(image,IMAGE_SHAPE)
        self.activation_Conv1 = self.predict(        img          ,self.img_shape[0],self.filter_shape[0],self.conv_out_shape[0],self.poolsize,0,self.pool_out_shape[0])
        self.activation_Conv2 = self.predict(self.activation_Conv1,self.img_shape[1],self.filter_shape[1],self.conv_out_shape[1],self.poolsize,1,self.pool_out_shape[1])
        a = np.reshape(self.activation_Conv2,(self.sizes[0],1))
        for b, w in zip(self.biases, self.weights):
            # print(a.shape)
            a = np.dot(w, a)+b
            a[a < 0] = 0
        # print(a)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        training_data = list(training_data)
        if test_data:
            test_data = list(test_data)
            print(sum(1 for _ in test_data))
            n_test = sum(1 for _ in test_data)
        n = len(training_data)
        print(n)
        for j in range(epochs):
            q = 0
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n//50000, mini_batch_size)]
            for mini_batch in mini_batches:
                print(q)
                self.update_mini_batch(mini_batch, eta)
                q = q + 1
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        fnabla_w = [  [np.zeros(self.filter_shape[0],   dtype = np.float)] , [np.zeros(self.filter_shape[1],    dtype =np.float)]  ]
        fnabla_b = [  [np.zeros(self.filter_shape[0][0],dtype = np.float)] , [np.zeros(self.filter_shape[1][0], dtype =np.float)]  ]
        # q = 0
        for x,y in  mini_batch:
            u = np.reshape(x,(1,28,28))
            # print(q)
            delta_nabla_b, delta_nabla_w,delta_fnabla_b,delta_fnabla_w = self.backprop(u, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            fnabla_b = [nb+dnb for nb, dnb in zip(fnabla_b, delta_fnabla_b)]
            fnabla_w = [nw+dnw for nw, dnw in zip(fnabla_w, delta_fnabla_w)]
            # q = q + 1
        #Convolutional backup
        #================================================================================
        # print(delta_fnabla_b)
        self.weights = [w-(eta)*nw  for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta)*nb   for b, nb in zip(self.biases, nabla_b)]
        self.filters = [w-(eta)*nw  for w, nw in zip(self.filters, fnabla_w)]
        self.fbiases = [b-(eta)*nb   for b, nb in zip(self.biases, fnabla_b)]

    def backprop(self, image, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        fnabla_w = [  [np.zeros(self.filter_shape[0],dtype =np.float)] , [np.zeros(self.filter_shape[1],dtype =np.float)]  ]
        fnabla_b = [  [np.zeros(self.filter_shape[0][0],dtype =np.float)] , [np.zeros(self.filter_shape[1][0],dtype =np.float)]  ]
        # feedforward
        # img.shape = (1, 28, 28)
        self.conv1, self.activation_Conv1, bdpool1 = self.convolution(        image        ,self.img_shape[0],self.filter_shape[0],self.conv_out_shape[0],self.poolsize,0,self.pool_out_shape[0])
        self.conv2, self.activation_Conv2, bdpool2 = self.convolution(self.activation_Conv1,self.img_shape[1],self.filter_shape[1],self.conv_out_shape[1],self.poolsize,1,self.pool_out_shape[1])
        activations = [np.reshape(self.activation_Conv2,(self.sizes[0],1))] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            a = np.dot(w, activations[-1])+b
            zs.append(a)
            a[a < 0] = 0
            activations.append(a)
        # print(a)
        # backward pass
        z = zs[-1]
        sp = z>0
        delta = self.cost_derivative(activations[-1], y)*sp
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = z>0 # error
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        delta = np.dot(self.weights[0].transpose(), delta)
        delta = np.reshape(delta,self.pool_out_shape[1])
        dconv2 = self.back_prop_pool(delta,1,bdpool2)
        dpool1 = self.back_prop_conv(dconv2,1,fnabla_w,fnabla_b,bdpool2)
        dconv1 = self.back_prop_pool(dpool1,0,bdpool1)
        filtsz = self.filter_shape[0][-1]
        # print(self.conv_out_shape[0][0],self.conv_out_shape[0][1],self.conv_out_shape[0][2])
        for k in range(self.conv_out_shape[0][0]):
            for i in range(self.conv_out_shape[0][1]):
                for j in range(self.conv_out_shape[0][2]):
                    if bdpool1[k,i,j] == True:
                        fnabla_w[0] += dconv1[k,i,j]*image[:,i:i+filtsz,j:j+filtsz]
                        fnabla_b[0] += dconv1[k,i,j]
        return (nabla_b, nabla_w, fnabla_b, fnabla_w)

    def back_prop_pool(self,delta,ln ,bdpool):
        err = np.zeros(self.conv_out_shape[ln])
        n = self.poolsize
        for k in range(self.conv_out_shape[ln][0]):
            for i in range(self.conv_out_shape[ln][1]):
                for j in range(self.conv_out_shape[ln][2]):
                    if bdpool[k,i,j] == True:
                        err[k,i,j] = delta[0,k,i//2,j//2]

        return err

    def back_prop_conv(self,err,ln,fnabla_w,fnabla_b,bdpool):
        filtsz = self.filter_shape[ln][-1]
        dpool1 = np.zeros(self.pool_out_shape[0],dtype =np.float)
        for k in range(self.conv_out_shape[ln][0]):
            for i in range(self.conv_out_shape[ln][1]):
                for j in range(self.conv_out_shape[ln][2]):
                    if bdpool[k,i,j] == True:
                        # print( self.filters[1][k])
                        fnabla_w[ln] += err[k,i,j]*self.conv1[:,i:i+filtsz,j:j+filtsz]
                        fnabla_b[ln] += err[k,i,j]
                        dpool1[0,:,i:i+filtsz,j:j+filtsz] += self.filters[1][0][k]*err[k,i,j]
        return dpool1
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
        bdpool = np.zeros(output_size,dtype = bool )
        a, b, u, x = 0, 0, filter_shape[-1]//2, 0
        for filter,bias in zip(self.filters[w][0],self.fbiases[w][0]):
            for i in range(u,img_shape[-1]-u):
                for j in range(u,img_shape[-1]-u):
                    # print(img.shape)
                    newimg[x,a,b] = max(0,np.sum(img[:,i-u:i+u+1,j-u:j+u+1]*filter + bias))
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
                    newer[k,i//2,j//2] = np.max(newimg[k,i:i+n,j:j+n])
                    h = np.unravel_index(np.argmax(newimg[k,i:i+n,j:j+n]),newimg[k,i:i+n,j:j+n].shape)
                    bdpool[k,i+h[0],j+h[1]] = True
        return newimg, newer, bdpool

    def predict(self,img,img_shape,filter_shape, output_size ,n,w,final_shape):
        newimg = np.zeros(output_size,dtype = np.float )
        a, b, u, x = 0, 0, filter_shape[-1]//2, 0
        for filter,bias in zip(self.filters[w][0],self.fbiases[w][0]):
            for i in range(u,img_shape[-1]-u):
                for j in range(u,img_shape[-1]-u):
                    # print(img.shape, filter.shape,bias.shape)
                    newimg[x,a,b] = max(0,np.sum(img[:,i-u:i+u+1,j-u:j+u+1]*filter + bias))
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
                    newer[k,i//2,j//2] = np.max(newimg[k,i:i+n,j:j+n])
        return newer

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
#
# if __name__ == '__main__':
#     # arr = np.random.randn((3,8,8))
#     img_shape = (1,1,28,28)
#     img = np.random.randint(5,size = img_shape)
#     N = Network()
#     y = np.zeros(10)
#     for _ in range(5):
#         N.update_mini_batch(img,y,0.1)
    # N.feedforward(img)


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
