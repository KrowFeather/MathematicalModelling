from pulp import *
solver=LpProblem('test1',LpMaximize)
x1=LpVariable('x1',lowBound=0,)
x2=LpVariable('x2',lowBound=0,)
x3=LpVariable('x3',lowBound=0,)
solver+=(1.5*x1+3*x2+5*x3<=600,'c1')
solver+=(280*x1+250*x2+400*x3<=6000,'c2')
solver+=(2*x1+3*x2+4*x3,'obj')
solver.solve()
