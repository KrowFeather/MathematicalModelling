from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# 拟合的目标函数
def func(x, a, b):
    return a * x + b

# 生成模拟数据
x_data= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([2.5, 4.5, 4.8, 5.5, 6.0, 7.0, 7.8, 8.0, 9.0, 9.5])
#x_data = np.linspace(0, 4, 50)
#y_data = func(x_data, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=len(x_data))

# 使用curve_fit函数来拟合非线性数据
popt, pcov = curve_fit(func, x_data, y_data)

# 画出原始数据和拟合曲线
plt.scatter(x_data, y_data, label="Data")
plt.plot(x_data, func(x_data, *popt), color='red', label="Fitted curve")
plt.legend()
plt.show()

