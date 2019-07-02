import numpy as np
import math

def get_specific_heat(all_lattices,T):
    #Specific Heat of a set of latticess
    # all_lattices = np.array(all_lattices)
    H_vals=[]
    Lshape = all_lattices[0].shape
    # print(Lshape,T,all_lattices)
    for lattice in all_lattices:
        H_vals.append(0.5*np.sum(calculate_H_matrix(lattice)))
    E_var=np.var(H_vals)
    # print(E_var)
    return (E_var) * (T ** (-2))*(1.0/(Lshape[0]*Lshape[1]))

def get_energy(all_lattices):
    # print(Lshape,T,all_lattices)
    H_vals=[]
    for lattice in all_lattices:
        H_vals.append(0.5*np.sum(calculate_H_matrix(lattice)))
    return H_vals

def calculate_H_matrix(lattice):
    #H_matrix is the matrix containing the value of hamiltonian at each site
    H_matrix=np.zeros(lattice.shape)
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            H_matrix[i, j] -= np.cos(2 * np.pi * (lattice[i, j] - lattice[i, (j + 1) % lattice.shape[1]]))
            H_matrix[i, j] -= np.cos(2 * np.pi * (lattice[i, j] - lattice[i, (j - 1) % lattice.shape[1]]))
            H_matrix[i, j] -= np.cos(2 * np.pi * (lattice[i, j] - lattice[(i + 1) % lattice.shape[0], j]))
            H_matrix[i, j] -= np.cos(2 * np.pi * (lattice[i, j] - lattice[(i - 1) % lattice.shape[0], j]))
    return H_matrix

def get_magnetization(lattice):
    #Gives net magnetization of a lattice
    mag_x=np.mean(np.cos(2*np.pi*lattice))
    mag_y=np.mean(np.sin(2*np.pi*lattice))
    return (mag_x**2+mag_y**2)**0.5

def get_magnetization_direction(lattices):
    theta=[]
    for lattice in lattices:
        mag_x=np.mean(np.cos(2*np.pi*lattice))
        mag_y=np.mean(np.sin(2*np.pi*lattice))
        u = math.degrees(math.atan2(mag_y,mag_x))
        if u>0:
            theta.append(u-360)
        else:
            theta.append(u)
    return theta

def get_mean_magnetization(lattices):
    #Mean Magnetization and Standard Deviation of a set of lattices
    mag=[]
    for lattice in lattices:
        mag.append(get_magnetization(lattice))
    return [[mag],np.mean(mag),np.std(mag)]#

def get_vorticity_configuration(lattice):
    #Vorticity configuration is the matrix containing the value of vorticity at each site
    vortex_matrix=np.zeros(lattice.shape)
    l=lattice.shape[0]
    for i in range(lattice.shape[0]):
        for j in range(lattice.shape[1]):
            vortex_matrix[i,j]+=saw(lattice[(i+1)%l,(j+1)%l]-lattice[i,(j+1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i+1)%l,j]-lattice[(i+1)%l,(j+1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i+1)%l,(j-1)%l]-lattice[(i+1)%l,j])
            vortex_matrix[i,j]+=saw(lattice[i,(j-1)%l]-lattice[(i+1)%l,(j-1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i-1)%l,(j-1)%l]-lattice[i,(j-1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i-1)%l,j]-lattice[(i-1)%l,(j-1)%l])
            vortex_matrix[i,j]+=saw(lattice[(i-1)%l,(j+1)%l]-lattice[(i-1)%l,j])
            vortex_matrix[i,j]+=saw(lattice[i,(j+1)%l]-lattice[(i-1)%l,(j+1)%l])
            vortex_matrix[i,j]=round(vortex_matrix[i,j])
    return vortex_matrix

def saw(x):
    #saw function used to calculate vorticity at a site
    if x<=-1/2:
        return x+1
    if x>=1/2:
        return x-1
    else:
        return x
