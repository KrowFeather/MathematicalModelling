import numpy as np
import pandas as pd

from utils.DrawPlot import Figure

df = pd.read_excel('./TestDataSet/LinReg.xlsx')
mat = df.to_numpy()
Y = mat[:, mat.shape[1] - 1]
X = mat[:, 1]

print('X:', X)
print('Y:', Y)


def LinearRegressionOLS(x, y):
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    n = len(x)
    sum_xx = 0
    sum_xy = 0
    for i in range(n):
        sum_xx += x[i] ** 2
        sum_xy += x[i] * y[i]
    w = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
    b = np.mean(y) - np.mean(x) * w
    print(w, b)
    return w, b


w, b = LinearRegressionOLS(X, Y)
fig = Figure()
fig.draw2DScatterPlot(X, Y)
fig.drawLinearFunction(w, b, [np.min(X) - 2, np.max(X) + 2])
fig.show()
