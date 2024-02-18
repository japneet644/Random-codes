import numpy as np
from itertools import permutations, combinations
from math import factorial

diff = 1

for kk in range(1):
    m = 3
    u = 3
    v = 3
    A = -np.log(np.random.rand(m, u))
    DD = np.diag([1] + [0] * (m - 1))
    B = -np.log(np.random.rand(u, v))
    S1 = np.zeros((m, u))
    S2 = np.zeros((u, v))

    for i in range(m):
        S1[i, :] = A[i, :] / np.sum(A[i, :])

    for i in range(u):
        S2[i, :] = B[i, :] / np.sum(B[i, :])

    t1 = 1
    t2 = 1
    t3 = 1

    gamma0 = 1 / (0.95 * np.random.rand(m) + 0.05)
    gamma0[0] = 1
    PP = permutations(gamma0)

    for j in range(factorial(m)):
        gamma = list(PP[j])
        Gamma = np.diag(gamma)

        t1 = min(t1, np.sum(np.min(Gamma @ S1, axis=0)) / min(gamma0))
        t3 = min(t3, np.sum(np.min(Gamma @ (S1 @ S2), axis=0)) / min(gamma0))

        combinations = list(combinations(range(u), m))
        numCombinations = len(combinations)

        for i in range(numCombinations):
            S21 = S2[combinations[i], :]
            t2 = min(t2, np.sum(np.min(Gamma @ S21, axis=0)) / min(gamma0))

    if 1 - t3 > (1 - t1) * (1 - t2) + 1e-8:
        print('error ')
        print(1 - t3, (1 - t1) * (1 - t2), 1 - np.sum(np.min(S1 @ S2, axis=1)))

    diff = min(diff, abs(1 - t3 - (1 - t1) * (1 - t2)))

print(1 - t3, (1 - t1) * (1 - t2))
