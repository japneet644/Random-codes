import math
from tkinter import *
import numpy as np
import random

Leap = 1
Length = 5
Nofactions = 6

class Enviroment(object):
    """docstring for Enviroment."""
    def __init__(self,sizes):
        self.pos = [10.0,7.0]
        self.theta = 0
        self.obst = [[[4,2],[3,3],[2,2],[3,1]],[[4+2,2+2],[3+2,3+2],[2+2,2+2],[3+2,1+2]],[[4+2,2],[3+2,3],[4,2],[5,1]],[[0,0],[20,0],[20,20],[0,20]]]
        self.Qs = np.zeros(Nofactions)
        self.count = 0
        self.epsilon = 0.2
        self.reward = 0.0
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.brea = 0
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def next_state(self,action):
        # 0^  1| 2>  3<
        newtheta = self.theta
        newpos = self.pos
        # action = action%4
        if action==0:
            newpos = [self.pos[0] + Leap*math.cos(math.radians(math.radians(self.theta))), self.pos[1]+Leap*math.sin(math.radians(self.theta))]
        elif action == 1:
            newpos = [self.pos[0] - Leap*math.cos(math.radians(self.theta)), self.pos[1]-Leap*math.sin(math.radians(self.theta))]
        elif action==2:
            newpos = [self.pos[0] + Leap*math.cos(math.radians(self.theta-90)), self.pos[1]+Leap*math.sin(math.radians(self.theta-90))]
        elif action==3:
            newpos = [self.pos[0] - Leap*math.cos(math.radians(self.theta-90)), self.pos[1]-Leap*math.sin(math.radians(self.theta-90))]
        elif action==4:
            newtheta += 10
            newtheta %= 180
        elif action==5:
            newtheta -= 10
            newtheta %= 180
        return newpos,newtheta

    def Is_collision(self):
        if(self.pos[0]<0 or self.pos[1]<0 or self.pos[0]>20):
            return 1
        ep = ([self.pos[0] + Length*math.cos(math.radians(self.theta)), self.pos[1]+Length*math.sin(math.radians(self.theta))],[self.pos[0] - Length*math.cos(math.radians(self.theta)), self.pos[1]-Length*math.sin(math.radians(self.theta))])
        for quad in self.obst:
            a = list(zip(quad[:-1],quad[1:]))
            a.append([quad[-1],quad[0]])
            for line in a:
                if intersects(line,ep):
                    return 1;
        return 0;

    def Q_learning(self):

        # print(self.count)
        time =0
        while self.count < 1:
            self.count += 1
            print(self.pos,self.theta,self.reward,self.Qs)
            self.pos = [10.0,7.0]
            time = 0
            self.brea = 0
            while time<6000  and self.brea==0:
                time += 1
                self.epsilon = self.epsilon/(1+self.count//100)
                for a in range(Nofactions):
                    self.Qs[a] = self.feedforward([[self.pos[0]],[self.pos[1]],[3.14*self.theta/180],[a]])

                if random.uniform(0, 1)<(1-self.epsilon):
                    action = random.randint(0,Nofactions-1)
                else:
                    action = np.argmax(self.Qs)
                Qs2 = np.zeros(Nofactions)
                prew_pos,prev_theta = self.pos,self.theta
                for a in range(Nofactions):
                    newpos,newtheta = self.next_state(a)
                    Qs2[a] = self.feedforward([[newpos[0]],[newpos[1]],[3.14*newtheta/180],[a]])
                self.reward = 0
                if self.Is_collision()==1:
                    self.reward = -2
                    self.brea = 1
                elif newpos[1] > 15:
                    self.reward = 1
                    self.brea=1
                elif time==6000:
                    self.reward=-1
                # else:
                #     self.reward = 0.5 #-15+ self.pos[1]
                action2 = np.argmax(Qs2)
                self.Qs[action]=self.Qs[action] + 0.5*(self.reward + Qs2[action2]- self.Qs[action])
                # mini_batch = [[[prew_pos[0]],[prew_pos[1]],[prev_theta],[action]],self.Qs[action] + 0.5*(self.reward + Qs2[action2]- self.Qs[action])]
                self.update_mini_batch([[[prew_pos[0]],[prew_pos[1]],[prev_theta],[action]],self.Qs[action]])
                self.pos,self.theta = self.next_state(action2)
                # print(action2)
        return

# class Network(object):

    # def __init__(self, sizes):


    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs=1, mini_batch_size=1, eta=1,
            test_data=None):
        if test_data: n_test = len(test_data)
        n = len(training_data)
        # for j in xrange(epochs):
        #     # random.shuffle(training_data)
        #     mini_batches = [
        #         training_data[k:k+mini_batch_size]
        #         for k in xrange(0, n, mini_batch_size)]
        #     for mini_batch in mini_batches:
        self.update_mini_batch(training_data, eta)
            # if test_data:
            #     print "Epoch {0}: {1} / {2}".format(
            #         j, self.evaluate(test_data), n_test)
            # else:
            #     print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta=2):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        x, y = mini_batch
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta)*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = [np.array(u) for u in x ]
        activations = [np.array(x)] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # print(delta)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    # def evaluate(self, test_data):
    #     """Return the number of test inputs for which the neural
    #     network outputs the correct result. Note that the neural
    #     network's output is assumed to be the index of whichever
    #     neuron in the final layer has the highest activation."""
    #     test_results = [(np.argmax(self.feedforward(x)), y)
    #                     for (x, y) in test_data]
    #     return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


    #     if abs(math.atan((line[1][1]-line[0][1])/(line[1][0]-line[0][0])) - self.theta) < 0.5 or abs(math.atan((line[1][1]-line[0][1])/(line[1][0]-line[0][0])) - self.theta+180) < 0.5:
    #         if
def intersects(a, b):
    p = [b[0][0]-a[0][0], b[0][1]-a[0][1]]
    q = [a[1][0]-a[0][0], a[1][1]-a[0][1]]
    r = [b[1][0]-b[0][0], b[1][1]-b[0][1]]
    t = (q[1]*p[0] - q[0]*p[1])/(q[0]*r[1] - q[1]*r[0]) \
        if (q[0]*r[1] - q[1]*r[0]) != 0 \
        else (q[1]*p[0] - q[0]*p[1])
    u = (p[0] + t*r[0])/q[0] \
        if q[0] != 0 \
        else (p[1] + t*r[1])/q[1]
    return t >= 0 and t <= 1 and u >= 0 and u <= 1

if __name__ == '__main__':
    E = Enviroment([4,6,6,1])
    # N = Network([4,6,6,1])

    root = Tk()
    canvas = Canvas(root,width=600,height=600)
    canvas.pack()
    E.Q_learning()
    # ep = (100 + 10*(E.pos[0] + Length*math.cos(math.radians(E.theta))),100 +10*( E.pos[1]+Length*math.sin(math.radians(E.theta))), 100 +10*(E.pos[0] - Length*math.cos(math.radians(E.theta))), 100 +10*(E.pos[1]-Length*math.sin(math.radians(E.theta))))
    ep = ((E.pos[0] + Length*math.cos(math.radians(E.theta))),( E.pos[1]+Length*math.sin(math.radians(E.theta))),(E.pos[0] - Length*math.cos(math.radians(E.theta))),(E.pos[1]-Length*math.sin(math.radians(E.theta))))
    canvas.create_line(ep)
    # canvas.create_line(5,24 ,63 ,120,fill="red")
    for quad in E.obst[:-1]:
        a = np.reshape(np.array(quad),8)
        # a = 10*a + 100
        canvas.create_polygon(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7])
    # for action in np.zeros(12)+3:
    #     E.pos,E.theta = E.next_state(action)
    #     ep = ((E.pos[0] + Length*math.cos(math.radians(E.theta))),( E.pos[1]+Length*math.sin(math.radians(E.theta))), (E.pos[0] - Length*math.cos(math.radians(E.theta))), (E.pos[1]-Length*math.sin(math.radians(E.theta))))
    #     canvas.create_line(ep,fill="red")
    # # print(E.Is_collision())
    canvas.scale("all", 0, 0, 30, 30)


    root.mainloop()
    print(E.pos,E.theta,E.reward,E.Qs)
