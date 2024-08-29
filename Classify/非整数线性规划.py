import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint,linprog

#非线性规划对应模板
class TrustConstrOptimizer:
    def __init__(self, x0):
        """
        初始化优化器。
        
        参数:
        x0 (list or np.array): 初始猜测的变量值。
        """
        self.x0 = np.array(x0)
        
        # 设置变量的上下界：第一个变量x在0到1之间，第二个变量y在-0.5到2之间
        self.bounds = Bounds([0, -0.5], [1.0, 2.0])
        
        # 设置线性约束条件：线性不等式约束，限制 Ax <= b
        self.linear_constraint = LinearConstraint([[1, 2], [2, 1]], [-np.inf, 1], [1, 1])
        
        # 设置非线性约束条件
        self.nonlinear_constraint = NonlinearConstraint(self.cons_f, -np.inf, 1, jac=self.cons_J, hess=self.cons_H)
    
    def rosen(self, x):
        """
        目标函数：Rosenbrock函数，这是一个常用于测试优化算法的函数。
        
        参数:
        x (np.array): 变量值。
        
        返回:
        float: 函数在x处的值。
        """
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)


    def rosen_der(self, x):#可选项
        """
        目标函数的梯度（雅可比矩阵）。
        
        参数:
        x (np.array): 变量值。
        
        返回:
        np.array: 梯度向量，在x处的梯度。
        """
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
        der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
        der[-1] = 200*(x[-1]-x[-2]**2)
        return der

    def rosen_hess(self, x):#可选项
        """
        目标函数的黑塞矩阵（二阶导数矩阵）。
        
        参数:
        x (np.array): 变量值。
        
        返回:
        np.array: 黑塞矩阵，在x处的二阶导数矩阵。
        """
        x = np.asarray(x)
        H = np.diag(-400*x[:-1], 1) - np.diag(400*x[:-1], -1)
        diagonal = np.zeros_like(x)
        diagonal[0] = 1200*x[0]**2-400*x[1]+2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
        H = H + np.diag(diagonal)
        return H

    def cons_f(self, x):
        """
        非线性约束条件的函数定义。
        
        参数:
        x (np.array): 变量值。
        
        返回:
        list: 非线性约束函数的值。
        """
        return [x[0]**2 + x[1], x[0]**2 - x[1]]

    def cons_J(self, x):
        """
        非线性约束函数的雅可比矩阵（梯度）。
        
        参数:
        x (np.array): 变量值。
        
        返回:
        list: 雅可比矩阵，在x处的梯度。
        """
        return [[2*x[0], 1], [2*x[0], -1]]

    def cons_H(self, x, v):
        """
        非线性约束函数的黑塞矩阵（二阶导数矩阵）。
        
        参数:
        x (np.array): 变量值。
        v (np.array): 乘数向量，表示约束条件的权重。
        
        返回:
        np.array: 黑塞矩阵，在x处的二阶导数矩阵。
        """
        return v[0]*np.array([[2, 0], [0, 0]]) + v[1]*np.array([[2, 0], [0, 0]])

    def optimize(self):
        """
        执行优化过程，使用信赖域约束算法（trust-constr）。
        
        返回:
        OptimizeResult: 优化结果，包括优化后的变量值、目标函数值、约束条件等信息。
        """
        res = minimize(self.rosen, self.x0, method='trust-constr',
                       constraints=[self.linear_constraint, self.nonlinear_constraint],
                       options={'verbose': 1}, bounds=self.bounds)
        return res

c=[-5,-4,-6]
a=[[1,-1,1],
   [3,2,4],
   [3,2,0]]
b=[20,42,30]

res=linprog(c,A_ub=a,b_ub=b,method="revised simplex")
np.set_printoptions(suppress=True) 
print(res)
