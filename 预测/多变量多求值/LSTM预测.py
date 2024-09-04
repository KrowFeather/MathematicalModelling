import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams['font.sans-serif'] = ['SimHei']

# 假设 data 是你加载的数据
###################----------数据读取------------##########################

# 我们的数据要求不能包括我们的标题
# 我们的数据需要把我们的Y周加入到我们的数据表中

data = pd.read_excel("数据表.xlsx").values


###################---------数据预处理---------###########################

## 选取我们的数据集比例
nn = int(0.8 * data.shape[0])
numTimeStepsTrain = nn

## 选取我们的数据集和测试集
dataTrain = data[:numTimeStepsTrain+1, :]
dataTest = data[numTimeStepsTrain:, :]

## 将我们的数据进行标准化处理，并将对应的mu,sig存储后用于后一步的转换
mu = np.mean(dataTrain, axis=0)
print(mu)
sig = np.std(dataTrain, axis=0)

dataTrainStandardized = (dataTrain - mu) / sig
dataTestStandardized = (dataTest - mu) / sig

## 将我们的X变量和Y变量分别读取出来，计算上一个元素和当前元素之间的关系
XTrain = dataTrainStandardized[:-1, :]
YTrain = dataTrainStandardized[1:, 0]  # 假设Y是第一个特征
XTest = dataTestStandardized[:-1, :]
YTest = dataTestStandardized[1:, 0]  # 假设Y是第一个特征


## 重新构造矩阵结构以满足我们的要求
XTrain = XTrain.reshape((XTrain.shape[0], 1, XTrain.shape[1]))
XTest = XTest.reshape((XTest.shape[0], 1, XTest.shape[1]))

## 设置我们的x输入个数，Y输出个数
numFeatures = XTrain.shape[2]  # 输入特征数
numResponses = 1               # 输出节点数


##########################---------调参---------------##########################

## 调整神经元数量
numHiddenUnits = 10       


model = Sequential([
    LSTM(numHiddenUnits, input_shape=(1, numFeatures)),  # time_steps设置为1
    Dropout(0.2),
    Dense(numResponses)
])
## 调整学习率
optimizer = Adam(learning_rate=0.005) 

model.compile(optimizer=optimizer, loss='mean_squared_error')

########################----------运行--------------##########################
history = model.fit(XTrain, YTrain, epochs=500, batch_size=10, validation_data=(XTest, YTest), verbose=2)

#######################----------结果处理-----------#########################

## 获得我们的预测结果，并将对应的标准化结果返回
YPred_train = model.predict(XTrain) 
YPred_train = YPred_train * sig[0] + mu[0]  # 逆标准化,
YPred_test = model.predict(XTest)
YPred_test = YPred_test * sig[0] + mu[0]  # 逆标准化

## 评价指标，来评价我们的对应的模型
T_train = data[:numTimeStepsTrain, 0]  # 对应于YTrain的时间段
T_test = data[numTimeStepsTrain+1:, 0]   # 对应于YTest的时间段

error1 = np.sqrt(mean_squared_error(T_train, YPred_train))
error2 = np.sqrt(mean_squared_error(T_test, YPred_test))
mae1 = mean_absolute_error(T_train, YPred_train)
mae2 = mean_absolute_error(T_test, YPred_test)

print(f'训练集数据的MAE为：{mae1}')
print(f'验证集数据的MAE为：{mae2}')

##################--------------绘图----------------########################
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(YPred_train, '-s', color=[1, 0, 0], linewidth=1, markersize=5, markerfacecolor=[1, 0, 0])
plt.plot(T_train, '-o', color=[0.6, 0.6, 0.6], linewidth=0.8, markersize=4, markerfacecolor=[0.6, 0.6, 0.6])
plt.legend(['LSTM拟合训练数据', '实际分析数据'], loc='best')
plt.title('LSTM模型预测结果及真实值')
plt.xlabel('样本')
plt.ylabel('数值')
plt.show()

################-------------存储模型------------------####################
model.save('lstm_model.h5')

