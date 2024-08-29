import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def nonlinear_function(x, a, b, c):
    # 拆分输入变量
    x1, x2, x3, x4 = x
    return a * np.sin(b * x1) + c * np.exp(-x2) + x3 ** 2 * x4

# 输入变量，转置以适应 curve_fit 的要求
x_data = np.array([[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]])

y_data = np.array([1, 2, 3]) # 因变量

# 使用 curve_fit 进行拟合
popt, pcov = curve_fit(nonlinear_function, x_data, y_data)

# 可视化结果
plt.scatter(range(len(y_data)), y_data, label="Data")
plt.plot(range(len(y_data)), nonlinear_function(x_data, *popt), color='red', label="Fitted curve")
plt.legend()
plt.show()
