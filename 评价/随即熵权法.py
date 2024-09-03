import random
import numpy as np
import time
import Topsis as tp
# 第一步，设置一个初始化的权值,这一个需要我们通过其他的评价算法得到
w=np.array([0.1405,0.4150,0.3208,0.1237])
# 第二步：定义一个随机的次数,在wo附近不停的随机
w0=np.array([0.1405,0.4150,0.3208,0.1237])
n=w.shape[0]
w1=np.zeros((n,1))

w_all=[]
rand_times=100
for t in range(0,rand_times):
    sum=0
    for i in range(0,n-1):
        random.seed(time.time())
        rand1=random.uniform(0,1)*0.5*w.max()
        #print("rand1={}".format(rand1))
        rand2=random.uniform(0,1)
        if(rand2>0.5):
            rand1=w[i]-rand1
        else:
            rand1=w[i]+rand1
        sum+=rand1
        w1[i]=rand1
    
    #print(w1)
    w2=w1
    w2[n-1]=1-sum
#    print(w2[n-1])
    if(w2[n-1]<0):
        w2=w2-np.full((n,1),w2.min())
    w_all.append(w2)

#print(w_all)

# 对于我们随机生成的所有w进行评价
score_all=[]
for i in range(0,len(w_all)):
    w_test=w_all[i]
    score_all.append(tp.topsis(data,w_test))

score=ave(score_all)

