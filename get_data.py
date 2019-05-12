import numpy as np
from xy import *
import pickle, pprint
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math

def draw_grid(lattice_size,angle,beta):
    height = 640
    width = 640
    image = Image.new(mode='L', size=(height, width), color=255)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = int(image.width / lattice_size)

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)
    # del draw
    draw.text((10,10), "TEMP %f" %(beta), fill=(100))
    a = step_size//2
    for i in range(0, image.width, step_size):
        for j in range(0, image.height, step_size):
            draw.line(((i+a, j+a) , ( i + a + a*math.cos(angle[i//step_size,j//step_size]), j + a + a*math.sin(angle[i//step_size,j//step_size]))))
            draw.line(((i+a, j+a) , ( i + a - a*math.cos(angle[i//step_size,j//step_size]), j + a - a*math.sin(angle[i//step_size,j//step_size]))))
    image.show()

#Parameters might change
J = 1
max_T = 2
min_T = 0.01
lattice_shape = (8, 8) #can be changed to (16,16) or (32,32)
steps = 1
iters_per_step = 2000
random_state = 25
T_vals = np.linspace(min_T, max_T, 32)
betas = 1 / T_vals
lattices = []
#Monte Carlo Simulation
for beta in betas:
        lat=[]
        print(1/beta)
        random_state=random_state+1
        xy=XYModelMetropolisSimulation(lattice_shape=lattice_shape,beta=beta,J=J,random_state=random_state)
        for q in range(10000):
            xy.simulate(steps,iters_per_step)
            lat.append(xy.L+0)
            # draw_grid(lattice_shape[0],xy.L,1/beta)
        lattices.append(lat)
        print('Done')
#Saving Data
output = open(str(lattice_shape)+'lattices.pkl', 'wb')
pickle.dump(lattices, output)
output.close()
