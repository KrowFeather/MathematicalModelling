import random 
n=100000000
cnt=0
for i in range(n):
    x=random.uniform(0,1)
    y=random.uniform(0,1)
    dist=x**2+y**2
    if(dist<=1):
        cnt+=1

print(cnt/n*4)