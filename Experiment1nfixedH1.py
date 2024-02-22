import numpy as np
import pandas
import matplotlib.pyplot as plt
import pickle
import random
np.set_printoptions(threshold=np.inf, suppress=True)


# In[2]:


def generate_a_model(delta,n):
    alphas = 0*np.random.rand(n)+0.05
    alphas = alphas/np.sum(alphas)
    Pb = np.zeros([n,n])
#     E = np.ones([n,n]) # graph
    
    for i in range(n):
      for j in range(n):
        Pb[i,j] = alphas[j]/(alphas[i] + alphas[j])

    # pert = delta*np.random.uniform(low=-0.5, high=0.5, size=(n, n))
    pert = np.triu(np.ones((n, n)) * delta, k=1) - np.tril(np.ones((n, n)) * delta, k=-1)

    P = np.clip(Pb+pert, 0.02, 0.98)
    any_clipped = not np.array_equal(P, Pb+pert)

    np.fill_diagonal(pert, 0)
    np.fill_diagonal(P, 0)

    Z = []
    print('=======================' , k)
    for i in range(n):
        row_Z = []
        for j in range(n):
            if i!=j:
                samples = np.random.binomial(1, P[i][j], k).tolist()
#                 print(samples)
                row_Z.append(samples)
            else:
                row_Z.append([])
        Z.append(row_Z)

#     Z = np.random.rand(n, n * k) < P.flatten().repeat(k).reshape(n, n*k)

#     print("anything clipped: ", any_clipped)
    return P, Z


# In[3]:


def compute_test_statistic(pihat,Z,K):
    n = len(pihat)
    T = 0
    for i in range(n):
        for j in range(n):
            pij_hat = pihat[i] + pihat[j]
            if K[i,j] > 1 and i!=j:
                T += ((pij_hat)**2 * (Z[i, j] * (Z[i, j] - 1))) / (K[i,j] * (K[i,j] - 1))  +    pihat[j]**2    -    2 * pihat[j] * pij_hat * (Z[i,j] / K[i,j])

    return T


# In[4]:


def construct_Z_empirical_and_Kij(Zlist):
    n = len(Zlist)
    Z_empirical = [[0] * n for _ in range(n)]
    Kij = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i!= j:
                count = sum(Zlist[i][j])
                Z_empirical[i][j] = count
                Kij[i][j] = len(Zlist[i][j])
        Kij[i][i] = 2
    return Z_empirical, Kij


# In[5]:


def compute_Tfromlist(Z):
    Zemp , K = construct_Z_empirical_and_Kij(Z)
    n = len(K)
    Zemp = np.array(Zemp)
    K = np.array(K)
#     print(Zemp)
#     print(K)
    # print(np.sum(K, axis=1))
    Shat = np.where((K == 0), 0, Zemp / (K * n))
    for i in range(n):
        Shat[i, i] = 1 - np.sum(Shat[i, :])

    pihat = compute_stationary_distribution(Shat)
    T0 = compute_test_statistic(pihat,Zemp,K)

    return T0


# In[6]:


def compute_error(P): # prints ideal error
  n = P.shape[0]
  S = P/n
  for i in range(n):
      S[i,i] = 1 - np.sum(S[i,:])
  pi1 = compute_stationary_distribution(S)
  np.fill_diagonal(P, 0.5)
  Pi = np.diag(pi1)

  error = Pi.dot(P) + P.dot(Pi) - np.ones((n, n)).dot(Pi)

  # Compute the Frobenius norm squared
  result = np.linalg.norm(error, 'fro')**2

  print("ideal error:", result)
  return


# In[7]:


def compute_stationary_distribution(S):
    eigenvalues, eigenvectors = np.linalg.eig(S.T)
    dominant_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    dominant_eigenvalue = eigenvalues[dominant_eigenvalue_index]
    stationary_dist = np.real_if_close(np.abs(eigenvectors[:, dominant_eigenvalue_index]))

    # Normalize the stationary distribution
    stationary_dist /= np.sum(stationary_dist)

    return stationary_dist


# In[8]:


def permute_entries(Z, K):
    n = len(Z)
    new_Z = [[[] for _ in range(n)] for _ in range(n)]  # Initialize new_Z as a nested list

    for i in range(n):
        for j in range(n):
            if i != j:
                combined_entries = Z[i][j] + [1 - val for val in Z[j][i]]
                
                # Shuffle the elements in the combined list
                random.shuffle(combined_entries)

                # Assign the first K[i][j] elements to Z[i][j]
                new_Z[i][j] = combined_entries[:K[i][j]]

                # Assign the remaining elements to Z[j][i]
                new_Z[j][i] = [1 - val for val in combined_entries[K[i][j]:]]

    return new_Z


# In[9]:


# import copy
# def add_one_to_list(lst):
#     lst2 = copy.deepcopy(lst)
#     lst2.remove(3)
#     lst.append(100)
#     return lst2
# # Define a list
# my_list = [3,1, 2, 3, 3, 4, 5]

# # Call the function to add one to each element of the list
# modified_list = add_one_to_list(my_list)

# # Print the modified list
# print("Original List:", my_list)
# print("Modified List:", modified_list)



# In[10]:


import copy

def cycle_completion(Zlist):
    n = len(Zlist)
    i = np.random.choice(range(n))
    #     Zlist2 = copy.deepcopy(Zlist)
    cycle = [i]
    lll = 0
    success =  False
    while lll < 10*n:
        j = np.random.choice(range(n))
        lll += 1
        # print(i,j,Zlist2) 
        if j != i and  len(Zlist[i][j]) > 0:
            entry_num = np.random.choice(range(len(Zlist[i][j])))
            entry = Zlist[i][j][entry_num]
            
            if entry > 0:
                Zlist[i][j].pop(entry_num)
                Zlist[j][i].append(entry)
                i = j
                lll = 0
                cycle.append(j)
                if i == cycle[0]:
                    success = True
                    break
    if success == False:
         print('I could not commplete the cycle')
            # print(entry,entry[::-1], j,i)
    return Zlist, success


# ## MAIN CODE below
# 
# 

# In[11]:


n_values = np.arange(10, 101, 15)
k_values = [12, 24, 36]
# n_values = [10]
# k_values= [10]
# Lists to store the computed values
# count = 0 
T0mean_values = []
T1_values = []
T2_values = []
T0percentile_5 = []
T0percentile_95 = []
for k in k_values:
    for n in n_values:
        T0a, T1a, T2a = [], [], []
        for aaa in range(100):
            P, Z = generate_a_model(0.22,n) ### H1 = 0
            _, K = construct_Z_empirical_and_Kij(Z)

            T0 = compute_Tfromlist(Z) # Compute test statisitic without any reshuffling
            T0a.append(n*k*T0)            # print(T0)
            if aaa<10:                # Shuffle data and compute T1 and T2
                compute_error(P) # Prints ideal errror (verification)
                print('Model number ',  aaa, 'n = ', n, 'k = ', k)
                permuted_T1 = []
                permuted_T2 = []
                permZ = copy.deepcopy(Z)
                for u in range(200):
                    permZ1 = permute_entries(Z,K) # Home effect shuffling
                    permuted_T1.append(n*k*compute_Tfromlist(permZ1))
                for u in range(200):
                    permZ = permute_entries(permZ,K) # Home effect shuffling
                    num_of_cyclic_shuffling = max(n,50)
                    if u == 0: # Burnout
                        num_of_cyclic_shuffling = 2*n*k                    
                    for _ in range(num_of_cyclic_shuffling): # number of cyclic shuffling
                        Shuffling_success = False # Shuffling success
                        while not Shuffling_success: # sometimes the chain is not irreducible thats why shuffling fails & we restart the cycle 
                            dummy, Shuffling_success = cycle_completion(permZ)
                        permZ = dummy
                    permuted_T2.append(n*k*compute_Tfromlist(permZ))
                    # print(n*k*permuted_T1[-1],n*k*permuted_T2[-1])
                # Calculate the 95% value
                T1_95th = np.percentile(permuted_T1,95)
                T2_95th =  np.percentile(permuted_T2,95)
                print('For this model the values are', n*k*T0,T1_95th, T2_95th)
#                 print(len(permuted_T1[0]), len(permuted_T2))
                print(np.round(permuted_T1, 2))
                print(np.round(permuted_T2, 2))
                T1a.append(T1_95th)
                T2a.append(T2_95th)
            # Calculate mean and percentiles
        T0mean_values.append(np.mean(T0a))
        T0percentile_5.append(np.percentile(T0a, 5))
        T0percentile_95.append(np.percentile(T0a, 95))
        T1_values.append(np.mean(T1a))
        T2_values.append(np.mean(T2a))


# In[ ]:


# Save all the variables using pickle
data = {
'n_values': n_values,
'T0mean_values': T0mean_values,
'T0percentile_5':T0percentile_5,
'T0percentile_95':T0percentile_95,
'T1_values': T1_values,
'T2_values': T2_values,
'k_values': k_values
}

with open(f'Exp1H1_k.pickle', 'wb') as f:
    pickle.dump(data, f)

# Plotting

colors = ['red', 'green', 'blue', 'orange']  # Add more colors if needed 

for k in range(len(k_values)): 
    # Define labels
    label_T0 = f'n*k*T0, k = {k_values[k]}'
    label_T1 = f'n*k*T1, k = {k_values[k]}'
    label_T2 = f'n*k*T2, k = {k_values[k]}'
    
    y = np.array(T0mean_values[k*len(n_values): (k+1)*len(n_values)])
    y5 = np.array(T0percentile_5[k*len(n_values): (k+1)*len(n_values)])
    y95 = np.array(T0percentile_95[k*len(n_values): (k+1)*len(n_values)])
    
    plt.plot(n_values,y,'-*',color=colors[k], label =label_T0, linewidth=2) #mean curve.
    plt.fill_between(n_values, y5, y95, color=colors[k], alpha=.3)

    # Plot T1
    plt.plot(n_values, T1_values[k*len(n_values): (k+1)*len(n_values)], '-o', label=label_T1, color=colors[k], linewidth=3)

    # Plot T2
    plt.plot(n_values, T2_values[k*len(n_values): (k+1)*len(n_values)], '-s', label=label_T2, color=colors[k], linewidth=3)

plt.xlabel('n')
plt.ylim(-5, 5)
plt.ylabel('Threshold Values')
plt.title('Threshold Values for Different n and k')
plt.legend()
plt.grid(True)
plt.show()
# Save the figure with a dynamic filename containing the value of k
filename = f"Exp1H1_k.png"
plt.savefig(filename)
