import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel('./dataset/LogReg.xlsx')
mat = df.to_numpy()
Y = mat[:, mat.shape[1] - 1]
X = mat[:, 1:mat.shape[1] - 1]

class LogisticRegression:
    def __init__(self,X,Y,iter,alpha):
        self.alpha = alpha
        self.iter = iter
        self.X = X
        self.Y = Y
    
    def calcW(self,Y, X, ww, alpha):
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


    def calcB(self,Y, alpha, b):
        return b - alpha * (sum(Y) / len(Y))


    def YHat(self,w, X, b):
        f = [0 for _ in range(X.shape[0])]
        for i in range(X.shape[0]):
            x = X[i, :]
            yh = np.matmul(w, x.T) + b
            f[i] = 1 / (1 + math.exp(-1 * yh))
        return np.array(f)


    def lossFunc(self,yHat, Y):
        c = [0 for _ in range(len(Y))]
        for i in range(len(Y)):
            c[i] = Y[i] * math.log(yHat[i]) + (1 - Y[i]) * math.log(1 - yHat[i])
        s = sum(c)
        return -1 * s / len(Y)


    def work(self):
        iteration = self.iter
        m = len(self.Y)
        w = [1 for _ in range(self.X.shape[1])]
        b = 1
        loss = []
        while iteration > 0:
            print(self.iter - iteration + 1)
            yh = self.YHat(w, X, b)
            loss.append(self.lossFunc(yh, Y))
            print('loss:', loss[len(loss) - 1])
            newy = yh - Y
            w = self.calcW(newy, X, w, self.alpha)
            b = self.calcB(newy, self.alpha, b)
            iteration -= 1
        return w, b, loss


iter = 300
a = 0.2
model = LogisticRegression(X,Y,iter,a)
w, b, loss = model.work()
print('w:', w)
print('b:', b)

fig = plt.figure(figsize=(10,6))
plt.plot(loss)
