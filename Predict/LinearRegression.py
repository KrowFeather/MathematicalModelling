import copy

import numpy as np
import pandas as pd

from utils.DrawPlot import Figure

df = pd.read_excel('./TestDataSet/LinReg.xlsx')
mat = df.to_numpy()
Y = mat[:, mat.shape[1] - 1]
X = mat[:, 1:mat.shape[1] - 1]

print('X:', X)
print('Y:', Y)


def getYHat(w, X, b):
    return np.matmul(w, X.T) + b


def getW(ww, X, Y, yhat, alpha):
    w = copy.copy(ww)
    S = [0 for _ in range(len(Y))]
    C = [0 for _ in range(X.shape[1])]
    for i in range(len(Y)):
        S[i] = yhat[i] - Y[i]
    for i in range(X.shape[1]):
        for j in range(len(Y)):
            C[i] += S[j] * X[j][i]
        C[i] /= len(Y)
        w[i] = ww[i] - alpha * C[i]
    return w


def lossFunc(yHat, Y):
    c = [0 for _ in range(len(Y))]
    for i in range(len(Y)):
        c[i] = (yHat[i] - Y[i]) ** 2
    s = sum(c)
    return s / len(Y)


def gradientDescent(X, Y, iter, alpha):
    iterations = iter
    m = len(Y)
    w = [1 for _ in range(X.shape[1])]
    b = 1
    loss = []
    while iterations > 0:
        print(iter - iterations + 1)
        yhat = getYHat(w, X, b)
        w = getW(w, X, Y, yhat, alpha)
        derivB = np.sum(yhat - Y) / m
        b = b - alpha * derivB
        loss.append(lossFunc(yhat, Y))
        print('loss:', loss[len(loss) - 1])
        iterations -= 1
    return w, b, loss


iter = 100
alpha = 0.003
w, b, loss = gradientDescent(X, Y, iter, alpha)
print('w:', w)
print('b:', b)
fig = Figure()
fig.draw2DScatterPlot(x=list(range(1, len(loss) + 1)), y=loss)
fig.show()
