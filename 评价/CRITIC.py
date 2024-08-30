import numpy as np
import pandas as pd
x=np.array([[9,9,97,38],
           [8,8,97,40],
           [10,8,98,35],
           [9,9,99,30],
           [8,8,96,25]])
theta_j=[0 for i in range(0,x.shape[1])]
n=x.shape[0]
m=x.shape[1]
func_j=[0 for i in range(0,x.shape[1])]
for i in range(0,x.shape[0]):
    for j in range(0,x.shape[1]):
        print(x[i][j])

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

sum=np.sum(c_j)
c_j=[c_j[i]/sum for i in range(0,m)]
print(c_j)