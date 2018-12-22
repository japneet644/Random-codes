import numpy as np
import random
class neural(object):
    def __init__(self,size):
        self.noflayers = len(size)
        self.size = size
        self.weight = np.array([np.random.rand(x[0],x[1]) for x in zip(size[1:],size[:-1])])
        self.bias = np.array([np.random.rand(x) for x in size[1:]])
        self.activations = np.array([np.zeros([x]) for x in np.append(size[1:],size[-1])])
        self.derivative = np.array([np.zeros([x]) for x in np.append(size[1:],size[-1])])
        self.inp = np.zeros(size[0])
        self.cost = np.array(np.zeros([2,self.size[-1]]))
        self.error = np.array([np.zeros([x]) for x in self.size])
        self.learning_rate = 0.1

    # def costfunction(self,out):
    #     # 0 derivative
    #     for i in range(self.size[-1]):
    #         self.cost[0][i] = out[i] - self.activations[self.size[-1] - 1][i]
    #         self.cost[1][i] = 0.5*((out[i] - self.activations[self.size[-1] - 1][i])**2)
    #     return

    def incomingerror(self,x,y):
        #finds incoming error in x,y th neuron X is HORIZONTAL axis while y is VERTICAL
        e = 0
        # print(self.size[x+1])
        for i in range(self.size[x+1]):#i = 0,1
            e = e + self.weight[x][i][y]*self.error[x+1][i]
        return e

    def updateweights(self,x,y,z):
        if(x>=2):
            self.weight[x-1][y][z] -= self.learning_rate*self.error[x][y]*self.derivative[x-1][y]*self.activations[x-2][z]
        else:
            self.weight[x-1][y][z] -= self.learning_rate*self.error[x][y]*self.derivative[x-1][y]*self.inp[z]
        return

    def backprop(self):
        #backpropagate error
        self.error[-1] = np.array(list(map(lambda x,y:x*y, self.cost[0],self.derivative[-2])))
        for i in range(self.noflayers-2,-1,-1):
            for j in range(self.size[i]):
                self.error[i][j] = self.incomingerror(i,j)
        # print("SELf.error ", self.error)
        for i in range(self.noflayers-1,0,-1):
            for j in range(self.size[i]):
                for k in range(self.size[i-1]):
                    # print(i,j,k)
                    self.updateweights(i,j,k)
        return

    def forwardprop(self,entries,results):
        #inp = np.zeros([])
        for (inp,out) in zip(entries,results):
            self.inp = inp
            for i in range(0,self.noflayers-1):
                for j in range(self.size[i+1]):
                    if(i==0):
                        self.activations[0][j] = sigmoid(self.weight[0][j],self.inp,self.bias[0][j])
                    else:
                        self.activations[i][j] = sigmoid(self.weight[i][j],self.activations[i-1],self.bias[i][j])
                self.derivative[i] =  np.array(list(map(lambda x: x*(1-x), self.activations[i])))
            self.costfunction(out)
            self.activations[-1] = self.cost[1]
            self.derivative[-1] = self.cost[0]
            # print("activations ",A.activations)
            # print('derivative ',A.derivative)
            # print("cost ",self.cost[1])
            self.backprop()
        return

    def output(self,entries):
        #inp = np.zeros([])
        self.inp = entries
        # print("vec",self.weight[0][j],self.inp,self.bias[0][j])
        for i in range(0,self.noflayers-1):
            for j in range(self.size[i+1]):
                if(i==0):
                    self.activations[0][j] = sigmoid(self.weight[0][j],self.inp,self.bias[0][j])
                else:
                    self.activations[i][j] = sigmoid(self.weight[i][j],self.activations[i-1],self.bias[i][j])
        print("activations ",A.activations[-1])
            # print('derivative ',A.derivative)
            # print("cost ",self.cost[1])
            # self.backprop()
        return
def sigmoid(vec,tvec,bias):
    x = 1.0/(1.0 + np.exp(-np.dot(vec,tvec) - bias))
    # print(-np.dot(vec,tvec) - bias)
    return x
#def layerimpulse(inp,weights,bias):
#=========================================
#-----------------------------------------
#=========================================
#XAXIS IS VERTICAL y ASIS IS HORIZONTAL
#=========================================
#-----------------------------------------
#=========================================

if __name__ == '__main__':
    size =[3,4,4,1]   #list(map(int,input().split()))
    A = neural(size)
    a = [[False,False],[False,False],[False,False],[False,False],[False,False]]
    b,x,y = True,True,True
    for _ in range(10):
        if (a[0][0] == True and b == False) or a[0][1] == True and b == True:
            break
        a.pop(0)
        while (x==True and y==True) or (x ==True and a[-1][1] ==True) or (y ==True and a[-1][0] ==True):
            x = random.choice([True,False])
            y = np.random.rand() > .5
        a.append([x,y])
        print([x[0] for x in a])
        print([x[1] for x in a])
        print()
        # b = A.feedforward()
