# CRITIC评价法，类似于我们的熵权评价法
import numpy as np
import pandas as pd
# x是我们的输入的数据
x=np.array([[9,9,97,38],
           [8,8,97,40],
           [10,8,98,35],
           [9,9,99,30],
           [8,8,96,25]])

# 我们根据我们的数据计算出我们的每一个指标对应的方差
theta_j=[0 for i in range(0,x.shape[1])]
n=x.shape[0]
m=x.shape[1]

# 根据我们的每一个指标计算出对应的
func_j=[0 for i in range(0,x.shape[1])]



for j in range(0,x.shape[1]):
    sum=0
    res=0
    for i in range(0,x.shape[0]):
        sum+=x[i][j]
    sum/=n
    res=0
    for i in range(0,x.shape[0]):
        res+=1.0*(x[i][j]-sum)**2
    #print("res=",res)
    res**=0.5
#    print("res=",res)
    theta_j[j]=res
    #print("sumj=",sum)
print(theta_j)

# 计算出我们的每一个指标之间的相关性矩阵，1-得到的结果就是我们的防擦好
r=np.corrcoef(x,x)
print(r)
for j in range(0,m):
    sum=0
    res=0
    for i in range(0,n):
        res+=1-r[i][j]
    func_j[j]=res

#print(func_j)

c_j=[theta_j[i]*func_j[i] for i in range(0,m)]
print(c_j)

# 根据我们的所有的结果得到我们的最后的评价权重矩阵
sum=np.sum(c_j)
c_j=[c_j[i]/sum for i in range(0,m)]
print(c_j)