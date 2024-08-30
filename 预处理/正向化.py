import math

import numpy as np


def minMetrice(mat, cols):
    Max = np.max(mat, axis=0)
    print(Max)
    for i in cols:
        for j in range(len(mat)):
            mat[j][i] = Max[i] - mat[j][i]


def intervalMetrice(mat, cols, l, r):
    Max = [0 for _ in range(mat.shape[1])]
    Min = [math.inf for _ in range(mat.shape[1])]
    for i in cols:
        for j in range(len(mat)):
            Max[i] = max(Max[i], mat[j][i])
            Min[i] = min(Min[i], mat[j][i])
    M = [0 for _ in range(mat.shape[1])]
    for i in cols:
        M[i] = max(l - Min[i], Max[i] - r)
    for i in cols:
        for j in range(len(mat)):
            if mat[j][i] < l:
                mat[j][i] = 1 - (l - mat[j][i]) / M[i]
            elif l <= mat[j][i] <= r:
                mat[j][i] = 1
            else:
                mat[j][i] = 1 - (mat[j][i] - r) / M[i]


def bestScoreMatrices(mat, cols, bs):
    M = [0 for _ in range(mat.shape[1])]
    for i in cols:
        for j in range(len(mat)):
            M[i] = max(M[i], abs(mat[j][i] - bs))
    print(M)
    for i in cols:
        for j in range(len(mat)):
            mat[j][i] = 1 - abs(mat[j][i] - bs) / M[i]
