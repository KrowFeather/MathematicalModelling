import copy
import math

import numpy as np
import pandas as pd

from utils.DrawPlot import Figure

df = pd.read_excel('./TestDataSet/LogReg.xlsx')
mat = df.to_numpy()
Y = mat[:, mat.shape[1] - 1]
X = mat[:, 1:mat.shape[1] - 1]


def calcW(Y, X, ww, alpha):
    w = copy.copy(ww)
    S = [0 for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        for j in range(len(Y)):
            S[i] += Y[j] * X[j][i]
    for i in range(X.shape[1]):
        S[i] /= len(Y)
    for i in range(X.shape[1]):
        w[i] = ww[i] - alpha * S[i]
    return w


def calcB(Y, alpha, b):
    return b - alpha * (sum(Y) / len(Y))


def YHat(w, X, b):
    f = [0 for _ in range(X.shape[0])]
    for i in range(X.shape[0]):
        x = X[i, :]
        yh = np.matmul(w, x.T) + b
        f[i] = 1 / (1 + math.exp(-1 * yh))
    return np.array(f)


def lossFunc(yHat, Y):
    c = [0 for _ in range(len(Y))]
    for i in range(len(Y)):
        c[i] = Y[i] * math.log(yHat[i]) + (1 - Y[i]) * math.log(1 - yHat[i])
    s = sum(c)
    return -1 * s / len(Y)


def gradientDescent(X, Y, iter, alpha):
    iteration = iter
    m = len(Y)
    w = [1 for _ in range(X.shape[1])]
    b = 1
    loss = []
    while iteration > 0:
        print(iter - iteration + 1)
        yh = YHat(w, X, b)
        loss.append(lossFunc(yh, Y))
        print('loss:', loss[len(loss) - 1])
        newy = yh - Y
        w = calcW(newy, X, w, alpha)
        b = calcB(newy, alpha, b)
        iteration -= 1
    return w, b, loss


iter = 300
a = 0.2
w, b, loss = gradientDescent(X, Y, iter, a)
print('w:', w)
print('b:', b)
fig = Figure()
fig.draw2DScatterPlot(x=list(range(1, len(loss) + 1)), y=loss)
fig.show()
