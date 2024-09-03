from pulp import *
solver=LpProblem('test1',LpMaximize)
x1=LpVariable('x1',lowBound=0,)
x2=LpVariable('x2',lowBound=0,)
#x3=LpVariable('x3',lowBound=0,)
solver+=(x1+x2>=1,'c1')
#solver+=(280*x1+250*x2+400*x3<=6000,'c2')
solver+=(-(3*x1+2*x2),'obj')
solver.solve()
