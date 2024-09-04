import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import product
from tqdm import tqdm_notebook
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

# 1. 数据预处理
# 1.3 导入数据
df = pd.read_csv('user_balance_table.csv', parse_dates=['report_date'])
df.head().append(df.tail())
df.shape
df.info()
df.isnull().sum()
# print(df)

# 1.4提取“total_purchase_amt”和“total_redeem_amt”并按日求和汇总
daydf = df.groupby(['report_date'])
totaldf = daydf['total_purchase_amt', 'total_redeem_amt'].sum()
totaldf.head()
# print(totaldf)

# 2. 建模分析
# 2.1 绘制时序图
plt.style.use('ggplot')
fig = plt.subplots(figsize=(15, 5))
ax1 = plt.subplot2grid((1, 1), (0, 0))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
ax1.set_title('时序图', fontsize=18)
totaldf.plot(ax=ax1, linewidth=2, fontsize=14)

# 2.2 使用Dickey-Fuller测试来验证序列平稳性
sm.tsa.seasonal_decompose(totaldf['total_purchase_amt'].values, extrapolate_trend='freq', period=7)
print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(totaldf['total_purchase_amt'])[1])
sm.tsa.seasonal_decompose(totaldf['total_redeem_amt'].values, extrapolate_trend='freq', period=7)
print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(totaldf['total_redeem_amt'])[1])

# 2.3 使用Ljung-Box进行白噪声检验
purchase_value = acorr_ljungbox(totaldf['total_purchase_amt'], lags=1)
redeem_value = acorr_ljungbox(totaldf['total_redeem_amt'], lags=1)
print(purchase_value)
print(redeem_value)

# 2.4 画出时间序列图及其自相关图和偏自相关图的函数
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
    画出时间序列的图形
    y - 时间序列
    lags - 延迟大小
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)

    with plt.style.context(style):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y.values)[1]
        ts_ax.set_title(
            'Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
        plt.rcParams['axes.unicode_minus'] = False
        # plt.show()
tsplot(totaldf.total_purchase_amt, lags=60)
tsplot(totaldf.total_redeem_amt, lags=60)

# 2.5 原序列转化为平稳序列,季节差分设为7天，并设置1阶差分
prc_diff = totaldf.total_purchase_amt - totaldf.total_purchase_amt.shift(7)
tsplot(prc_diff[7:], lags=60)
pca_diff = prc_diff -prc_diff.shift(1)
tsplot(pca_diff[7+1:], lags=60)
red_diff = totaldf.total_redeem_amt - totaldf.total_redeem_amt.shift(7)
tsplot(red_diff[7:], lags=60)
rda_diff = red_diff -red_diff.shift(1)
tsplot(rda_diff[7+1:], lags=60)

# 2.5 确定SARIMA的值
# 获取total_purchase_amt的最优模型
# 设置初始值
ps = range(2, 5)
d = 1
qs = range(2, 5)
Ps = range(0, 2)
D = 1
Qs = range(0, 2)
s = 7  # 季节性长度仍然是7

# 使用所有可能的参数组合创建列表
parameters = product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
len(parameters_list)

pca_results = []
pca_best_aic = float('inf')  # 正无穷
for pca_param in parameters_list:

    #     try:
    pca_model = sm.tsa.statespace.SARIMAX(totaldf.total_purchase_amt, order=(pca_param[0], d, pca_param[1]),
                                          seasonal_order=(pca_param[2], D, pca_param[3], s)).fit()
    #     except ValueError:
    #         print('参数错误：',pca_param)
    #         continue
    pca_aic = pca_model.aic
    if pca_aic < pca_best_aic:
        pca_best_model = pca_model
        pca_best_aic = pca_aic
        pca_best_param = pca_param
    pca_results.append([pca_param, pca_model.aic])
# 输出最优模型
pca_result_table = pd.DataFrame(pca_results)
pca_result_table.columns = ['parameters', 'pca_aic']
print('最优模型：', pca_best_model.summary())

# 获取total_redeem_amt的最优模型
# 设置rda初始值
rps = range(1, 5)
rd = 1
rqs = range(1, 5)
RPs = range(0, 2)
RD = 1
RQs = range(0, 2)
Rs = 7  # 季节性长度仍然是7

# 使用所有可能的参数组合创建列表
rda_parameters = product(rps, rqs, RPs, RQs)
rda_parameters_list = list(rda_parameters)
len(rda_parameters_list)

rda_results = []
rda_best_aic = float('inf')  #正无穷
for rda_param in rda_parameters_list:
    rda_model = sm.tsa.statespace.SARIMAX(totaldf.total_redeem_amt,order=(rda_param[0], rd, rda_param[1]),
                                          seasonal_order=(rda_param[2], RD, rda_param[3], Rs)).fit()
    rda_aic = rda_model.aic
    if rda_aic < rda_best_aic:
        rda_best_model = rda_model
        rda_best_aic = rda_aic
        rda_best_param = rda_param
    rda_results.append([rda_param,rda_model.aic])
#输出最优模型
rda_result_table = pd.DataFrame(rda_results)
rda_result_table.columns = ['parameters','rda_aic']
print('最优模型：',rda_best_model.summary())

# 3. 建模预测
# “total_purchase_amt”序列建模预测实现代码及结果序列
fig=plt.figure(figsize=(20,5))
ax2=fig.add_subplot(111)
purchase=totaldf['total_purchase_amt']
purchasePredict=pca_best_model.predict('2014-09-01','2014-09-30',typ='levels')
purchasePredict.plot(ax=ax2)
purchase.plot(ax=ax2)
plt.show()

# “total_redeem_amt”序列建模预测实现代码及结果序列
fig=plt.figure(figsize=(20,5))
ax3=fig.add_subplot(111)
purchase=totaldf['total_redeem_amt']
redeemPredict=rda_best_model.predict('2014-09-01','2014-09-30',typ='levels')
redeemPredict.plot(ax=ax3)
purchase.plot(ax=ax3)
plt.show()

# 4. 输出预测结果
report_date=list(range(20140901,20140931))
sub=pd.DataFrame()
sub['purchase']=purchasePredict
sub['redeem']=redeemPredict
sub=sub.reset_index()
sub['index']=report_date
sub=sub.rename(columns={'index':'report_date'})
sub[['purchase','redeem']]=sub[['purchase','redeem']].astype(np.int64)
sub=sub.set_index(['report_date'])
sub.to_csv('tc_comp_predict_table.csv',sep=',')