import copy
import math


def topsis(m, W):
    mat = copy.copy(m)
    for i in range(mat.shape[0]):
        for j in range(1, mat.shape[1]):
            mat[i][j] *= W[j]
    Max = [0 for _ in range(mat.shape[1])]
    Min = [math.inf for _ in range(mat.shape[1])]
    for i in range(mat.shape[0]):
        for j in range(1, mat.shape[1]):
            Max[j] = max(Max[j], mat[i][j])
            Min[j] = min(Min[j], mat[i][j])
    dis_plus = [0 for _ in range(mat.shape[0])]
    dis_minus = [0 for _ in range(mat.shape[0])]
    for i in range(mat.shape[0]):
        for j in range(1, mat.shape[1]):
            dis_plus[i] += (Max[j] - mat[i][j]) ** 2
        dis_plus[i] **= 0.5
    for i in range(mat.shape[0]):
        for j in range(1, mat.shape[1]):
            dis_minus[i] += (Min[j] - mat[i][j]) ** 2
        dis_minus[i] **= 0.5
    Score = [0 for _ in range(mat.shape[0])]
    for i in range(mat.shape[0]):
        Score[i] = dis_minus[i] / (dis_plus[i] + dis_minus[i])
    tot = sum(Score)
    for i in range(len(Score)):
        Score[i] = Score[i] / tot * 100
    print(Score)
    return Score
