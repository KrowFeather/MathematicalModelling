import math


def normalization(mat, mtype):
    if mtype == 0: 
        standard = [0 for i in range(mat.shape[1])]
        for i in range(mat.shape[0]):
            for j in range(1, mat.shape[1]):
                standard[j] += mat[i][j] ** 2
        for i in range(len(standard)):
            standard[i] = standard[i] ** 0.5
        for i in range(mat.shape[0]):
            for j in range(1, mat.shape[1]):
                mat[i][j] /= standard[j]
        print(mat)
    else:
        Max = [0 for _ in range(mat.shape[1])]
        Min = [math.inf for _ in range(mat.shape[1])]
        for i in range(mat.shape[0]):
            for j in range(1, mat.shape[1]):
                Max[j] = max(Max[j], mat[i][j])
                Min[j] = min(Min[j], mat[i][j])
        for i in range(mat.shape[0]):
            for j in range(1, mat.shape[1]):
                mat[i][j] = (mat[i][j] - Min[j]) / (Max[j] - Min[j])
        print(mat)
