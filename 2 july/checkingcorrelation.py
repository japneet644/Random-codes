import numpy as np
from xy import *
import pickle, pprint
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math
J = 1
max_T = 0.5
min_T = 0.1
lattice_shape = (8, 8) #can be changed to (16,16) or (32,32)
steps = 1
iters_per_step = 8000
random_state = 25
T_vals = np.linspace(min_T, max_T, 5)
betas = 1 / T_vals
lattices = []
theta = []
len = []
#Monte Carlo Simulation
for beta in betas:
        lat=[]
        print(1/beta)
        random_state=random_state+1
        xy=XYModelMetropolisSimulation(lattice_shape=lattice_shape,beta=beta,J=J,random_state=random_state)
        for q in range(10):
            xy.simulate(steps,iters_per_step)
            lat.append(xy.L+0)
            mag_x=np.mean(np.cos(2*np.pi*xy.L))
            mag_y=np.mean(np.sin(2*np.pi*xy.L))
            u = math.degrees(math.atan2(mag_y,mag_x))
            # xy.L = np.random.rand(*lattice_shape)
            if u>0:
                theta.append(u-360)
            else:
                theta.append(u)
            print(u,(mag_x**2+mag_y**2)**0.5)
        lattices.append(lat)
        len = xy.get_correlations()
        print('Done')

print(len)
#Saving Data
output = open(str(lattice_shape)+'lattices.pkl', 'wb')
pickle.dump(lattices, output)
output.close()
