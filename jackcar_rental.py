import numpy as np
from math import factorial
# import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline


def poissonProb(lambdA,n):
    return (lambdA**n)*np.exp(-lambdA)/factorial(n)

class JacksCarRental:
    def __init__(self):
        self.maxCars = 20
        self.maxMove = 5
        self.nCars = self.maxCars+1 # 21 possibilities
        self.maxMorning = self.nCars+self.maxMove
        self.rentPrice = 10
        self.moveCost = 2
        self.dcRate = 0.9
        self.freeParkingCap = 10
        self.parkingCost = 2
        self.theta = 0.001
        self.valueMat = np.zeros((self.nCars,self.nCars))
        self.policyMat = np.random.randint(6, size=(self.nCars,self.nCars))      # np.zeros((self.nCars,self.nCars),dtype ="int8")# valoue at [i,j] = number of cars that will be transfered when there are i,j cars at location X ,Y respextively
        self.rew1,self.tP1 = self.initProbs(3,3)
        self.rew2,self.tP2 = self.initProbs(4,2)

    def initProbs(self,rent_Lambda,return_Lambda):
        """Returns
        1)rew[i] :- Expected Immediate reward from i cars
        2)tp[i,j] :- Probability that no of cars changes from i to j after an event of rent and return
        """
        rew = np.zeros(self.maxMorning)
        tP =  np.zeros((self.maxMorning,self.nCars))
        rent = 0
        rentProb = 1
        while rentProb > self.theta:
            rentProb = poissonProb(rent_Lambda,rent)
            for n in range(self.maxMorning):
                satRent = min(n,rent)
                rew[n] += self.rentPrice*rentProb*satRent
            ret = 0
            retProb = 1
            while retProb > self.theta:
                retProb = poissonProb(return_Lambda,ret)
                for m in range(self.maxMorning):
                    satRent = min(rent,m)
                    new_n = m+ret-satRent
                    new_n = max(new_n,0)
                    new_n = min(self.nCars-1,new_n)
                    tP[m,new_n] += rentProb*retProb
                ret += 1
            rent += 1
        return rew,tP

    def policyIter(self):
        policyStable = False
        count = 0
        policies = []
        while not policyStable:
            print( 'Policy',count,' =====================')
            print(self.policyMat)
            policies.append(self.policyMat)
            print( 'Evaluating ....')
            self.policyEval()
            print( 'Improving ....')
            policyStable = self.policyImprove()
            count += 1
        return policies


    def policyEval(self):
        diff = 0.1
        while (diff > self.theta):
            diff = 0
            for n1 in range(self.nCars):
                for n2 in range(self.nCars):
                    tmpV = self.valueMat[n1,n2]
                    a = self.policyMat[n1,n2]
                    self.valueMat[n1,n2] = self.calVal(n1,n2,a)
                    diff = max(diff,np.abs(self.valueMat[n1,n2]-tmpV))

    def calVal(self,n1,n2,a):
        """restricting a so that mod(a) = allowed value of cars (after transfer, acc. to policy) at both locations."""
        a = min(a,+n1)
        a = max(a,-n2)
        a = min(+self.maxMove,a)
        a = max(-self.maxMove,a)
        val = -self.moveCost*np.abs(a)
        morning_n1 = n1 - a
        morning_n2 = n2 + a
        for new_n1 in range(self.nCars):
            for new_n2 in range(self.nCars):
                val += (self.tP1[morning_n1,new_n1]*self.tP2[morning_n2,new_n2]*((self.rew1[morning_n1]+self.rew2[morning_n2])+(self.dcRate*self.valueMat[new_n1,new_n2])))
        return val

    def policyImprove(self):
        policyStable = True
        for n1 in range(self.nCars):
            for n2 in range(self.nCars):
                b = self.policyMat[n1,n2]
                self.policyMat[n1,n2] = self.greedyPolicy(n1,n2)
                if b != self.policyMat[n1,n2]:
                    policyStable = False
        return policyStable

    def greedyPolicy(self,n1,n2):
        a_min = max(-self.maxMove,-n2)
        a_max = min(+self.maxMove,+n1)
        bestAction = a_min
        bestValue = float(self.calVal(n1,n2,a_min))
        for a in range(a_min+1,a_max+1):
            val = self.calVal(n1,n2,a)
            if val > bestValue+self.theta:
                bestValue = val
                bestAction = a
        return bestAction

    # def printValues(self,v=None,i=''):
    #     if not v:
    #         v = self.valueMat
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.set_title('Values '+str(i))
    #     aZ = []
    #     aX = []
    #     aY = []
    #     for i in range (jp.nCars):
    #         for j in range (jp.nCars):
    #             aX.append(i)
    #             aY.append(j)
    #             aZ.append(v[i, j])
    #     ax.set_ylabel('# of cars at location 1')
    #     ax.set_xlabel('# of cars at location 2')
    #     ax.scatter(aX, aY, aZ)

    # def printPolicy(self,p,i=''):
    #     plt.figure()
    #     ticks = [0]+['']*(self.maxCars-1)+[self.maxCars]
    #     ax = sns.heatmap(p.astype(int),square=True,xticklabels=ticks,yticklabels=ticks)
    #     ax.set_title('Policy '+str(i))
    #     ax.set_ylabel('# of cars at location 1')
    #     ax.set_xlabel('# of cars at location 2')
    #     ax.invert_yaxis()
    #     cbar = ax.collections[0].colorbar
    #     cbar.set_ticks(np.arange(self.maxMove*2+1)-self.maxMove)
    #     cbar.set_ticklabels(np.arange(self.maxMove*2+1)-self.maxMove)



jp = JacksCarRental()
p = jp.policyIter()

# for i,po in enumerate(p):
#     jp.printPolicy(po,i)

# jp.printValues(None,4)
