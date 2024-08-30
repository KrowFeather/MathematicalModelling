import numpy as np


def checkAHPAvailable(mat):
    n = mat.shape[0]
    print(n)
    eig_val, eig_vec = np.linalg.eig(mat)
    Max_eig = max(eig_val)
    CI = (Max_eig - n) / (n - 1)
    RI = [0.0, 0.0001, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
    CR = CI / RI[n - 1]
    print(CI)
    print(CR)
    if CR < 0.10:
        return True
    else:
        return False


def arithmeticMeanMethod(mat):
    print(mat)
    Sum = np.sum(mat, axis=0)
    n = mat.shape[0]
    std = mat / Sum
    Sumr = np.sum(std, axis=1)
    W = Sumr / n
    return W


def geometricMeanMethod(mat):
    n = mat.shape[0]
    prodA = np.prod(mat, axis=1)
    prodB = np.power(prodA, 1 / n)
    re_prod_A = prodB / np.sum(prodB)
    return re_prod_A


def eigenvalueMethod(mat):
    n = mat.shape[0]
    eigVal, eigVec = np.linalg.eig(mat)
    max_index = np.argmax(eigVal)
    max_vec = eigVec[:, max_index]
    W = max_vec / np.sum(max_vec)
    return W
