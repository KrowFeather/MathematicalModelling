import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 导入数据并设置日期列为索引
# 这里将日期解析为日期格式，并将列重命名为“日期”和“女装”
dta = pd.read_csv('women_dress.csv', parse_dates=['0'])
dta = dta.rename(columns={'0': '日期', '1': '女装'})
dta.set_index('日期', inplace=True)
print(dta)

# 绘制时间序列数据的图表
plt.plot(dta)
plt.show()

# 显示带有固定尺寸的时间序列图
dta.plot(figsize=(10, 4))

# Holt-Winters法应用于时间序列建模
#diff1=dta.diff().dropna()
winter_model = ExponentialSmoothing(dta).fit()
print(winter_model.summary())

# ADF检验（单位根检验）以检查时间序列的平稳性
# ADF(dta)

'''
这里参数lags=5表示只检验滞后五期。
我们可以看到五期的P值全部小于0.05，
说明在0.05的显著性水平下，
该数据不是纯随机序列，可以进行下一步建模。
'''

# 自相关图（确定q）和偏自相关图（确定p）
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)
plt.tight_layout()
plt.show()

'''
ACF图拖尾，PACF图2阶截尾，建议拟合AR(2)模型。
'''

# 一阶差分以使非平稳序列变为平稳序列
diff1 = dta.diff().dropna()
plt.plot(diff1)
plt.show()

diff2= diff1.diff().dropna()
plt.plot(diff2)
plt.show()

diff3=diff2.diff().dropna()
plt.plot(diff3)
plt.show()
# 差分后的自相关图和偏自相关图
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(diff3.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(diff3, lags=40, ax=ax2)
plt.tight_layout()
plt.show()

# 使用ARIMA模型进行拟合，参数order=(p,d,q)
arma_mod20 = ARIMA(dta, order=(1, 3, 3)).fit()  # p=1, d=1, q=1
print(arma_mod20.params)

# 残差检验
arma_mod20.resid.plot(figsize=(10, 3))

# 正态性检验
stats.normaltest(arma_mod20.resid)

# QQ图用于查看残差的正态性
qqplot(arma_mod20.resid, line="q", fit=True)

# Durbin-Watson检验，用于检测残差的自相关性
dw_stat = sm.stats.durbin_watson(arma_mod20.resid.values)
print(f'Durbin-Watson statistic: {dw_stat}')

# Ljung-Box检验，用于进一步检查残差的白噪声特性
lb = lb_test(arma_mod20.resid, return_df=True, lags=5)
print(lb)

# 残差的自相关和偏自相关图
fig = plt.figure(figsize=(8, 5))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(arma_mod20.resid.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(arma_mod20.resid, lags=40, ax=ax2)

# 模型预测
predict_sunspots = arma_mod20.predict(start="1989-01-01", end="1998-12-01")

# 绘制实际值与预测值的比较图
plt.figure(figsize=(10, 4))
plt.plot(dta.index, dta['女装'], label='Actual')
plt.plot(predict_sunspots.index, predict_sunspots, label='Predicted')
plt.legend(['Actual', 'Predicted'])
plt.xlabel('Time (Year)')
plt.ylabel('Women\'s Dress Sales')
plt.show()

# 模型评价：计算MAE和RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    return mae, rmse

mae, rmse = evaluation(dta.to_numpy(), predict_sunspots[:len(dta)].to_numpy())
print(f'MAE: {mae}, RMSE: {rmse}')

# 结束程序

exit(0)