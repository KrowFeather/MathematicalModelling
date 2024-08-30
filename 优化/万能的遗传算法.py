# 万能的遗传算法

#只需要给定我们的约束，目标函数，一定给你结果
import numpy as np
import random
import copy
city=[(1,2),(2,3),(3,4),(4,5),(5,6)]
n=len(city)
distance=np.zeros((n,n))
print(distance)
print(n)
for i in range(0,n):
    for j in range(0,n):
        distance[i][j]=((int(city[i][1])-int(city[j][1]))**2+(int(city[i][0])-int(city[j][0]))**2)**(0.5)

NP=20 #种群大小
G=500 #遗传代数
L=n #编码长度
PC=0.7 #交叉率
PM=0.3 #变异率
f0=[x for x in range(0,L)]
f=[]

for i in range(0,NP):
    f1=f0.copy()
    random.shuffle(f1)
    f.append(f1)

print(f)