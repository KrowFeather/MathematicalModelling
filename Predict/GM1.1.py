import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
df = pd.read_excel('./dataset/ref.xlsx')
X = df.to_numpy()
print(X)

n = X.shape[1]
lmda = [0 for _ in range(n-1)]
for i in range(1,n):
    lmda[i-1] = X[0,i-1]/X[0,i]
print(lmda)

theta = [math.e**(-2/(n+1)),math.e**(2/(n+1))]

valid = True
for v in lmda:
    if theta[0]<=v<=theta[1]:
        continue
    else:
        valid=False
        break
print(valid)


X1 = np.zeros((1,n))
X1[0,0] = X[0,0]
for i in range(1,n):
    X1[0,i] = X1[0,i-1]+X[0,i] 
print(X1)

B = np.ones((n-1,2))
for i in range(n-1):
    B[i,0]= -0.5*(X1[0,i]+X1[0,i+1])
print(B)

Y = X[:,1:]
Y = Y.T
print(Y)

tmp_1 = np.linalg.inv(np.dot(B.T,B))
tmp_2 = np.dot(tmp_1,B.T) 
ans = np.dot(tmp_2,Y)
ans = ans.T
a = ans[0,0]
b = ans[0,1]
print(a,b)

pred = [0 for _ in range(n)]
for i in range(1,n+1):
    pred[i-1] = (X[0,0]-b/a)*math.e**(-a*(i-1))+b/a
print(pred)

measured_X = [0 for _ in range(n)]
measured_X[0]=X[0,0]
for i in range(1,n):
    measured_X[i] = pred[i] - pred[i-1]
print('X predicted:',measured_X)

error = [0 for _ in range(n)]
for i in range(n):
    error[i] = X[0,i]-measured_X[i]
print('error:',error)

crv = [0 for _ in range(n)]
for i in range(n):
    crv[i] = 1 - (1-0.5*a)/(1+0.5*a)*lmda[i-1]
crv[0]=0
print("crv:",crv)

plt.plot(list(range(n)),pred,marker = 'o')
plt.plot(list(range(n)),X1[0,:],marker = 'o')
plt.show()

plt.plot(list(range(n)),measured_X,marker = 'o')
plt.plot(list(range(n)),X[0,:],marker = 'o')
plt.show()