# 该方法用于我们得到的多个权重矩阵之间的组合结果
# 根据多个权重矩阵，得到最红的权重矩阵
import numpy as np
import pandas as pd
w=np.array([[0.1405,0.4150,0.3208,0.1237],
            [0.3532,0.3186,0.1726,0.1557],])
# w1,w2,w3
#w=w.T
#x代表我们通过三种方法得到的三种权重,横向量表示来自不同方法
n=w.shape[0]
m=w.shape[1]
w_c=np.zeros((n,n),dtype=float)
w_r=np.zeros((n,1),dtype=float)

#print(type(np.dot(w[0],w[0])))
for i in range(0,n):
    w_r[i]=np.dot(w[i],w[i])
    for j in range(0,n):
        w_c[i][j]=np.dot(w[i],w[j])
w_r=w_r 
minn=np.inf
for i in range(0,n):
    for j in range(0,n):
        minn=min(minn,w_c[i][j])
if(minn<0):
    w_c=(w_c-min(w_c))/sum(w_c-min(w_c))

#print(w_c)
#print(np.linalg.inv(w_c))
#print(w_r)
a=np.dot(np.linalg.inv(w_c),w_r) #矩阵乘法
print(a)