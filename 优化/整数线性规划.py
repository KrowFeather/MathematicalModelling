from pulp import *
c=[20,10]
a=[[5,4],
   [2,5]]
b=[24,13]
solver=LpProblem('test1',LpMaximize)
x1=LpVariable('x1',lowBound=0,cat=LpInteger)
x2=LpVariable('x2',lowBound=0,cat=LpInteger)
solver+=(5*x1+4*x2<=24,'c1')
solver+=(2*x1+5*x2<=24,'c2')
solver+=(20*x1+10*x2,'obj')
solver.solve()
