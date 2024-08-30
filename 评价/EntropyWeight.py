import math

import numpy as np


def entropyWeight(mat):
    P = [[0 for x in range(mat.shape[1])] for y in range(mat.shape[0])]
    S = np.sum(mat, axis=0)
    for i in range(mat.shape[0]):
        for j in range(1, mat.shape[1]):
            P[i][j] = mat[i][j] / S[j]
    E = [0 for x in range(mat.shape[1])]
    for i in range(mat.shape[0]):
        for j in range(1, mat.shape[1]):
            if P[i][j] == 0:
                continue
            E[j] += P[i][j] * math.log(P[i][j])
    for i in range(1, mat.shape[1]):
        E[i] = -1 / math.log(mat.shape[0]) * E[i]
    D = [0 for x in range(mat.shape[1])]
    for i in range(1, mat.shape[1]):
        D[i] = 1 - E[i]
    W = [0 for x in range(mat.shape[1])]
    tot = sum(D)
    for i in range(1, mat.shape[1]):
        W[i] = D[i] / tot
    print(W)
    return W
