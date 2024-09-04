#####-------------调库小能手的cftools-python弱化版---------------#######
###################使用方法：##################################
###########1。划到底部导入我们的数据############################
###########2。选择一类合适的模型来进行拟合猜测###################
###########3。如果数据可以拟合，我们会跳出一个图#################
###########4。如果诗句不适合拟合，我们会输出-1报错###############
###########5。等一个有缘人做一个好看的UI########################
##############################################################
import inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
class function:
    def exponential_model_1(x, a, b):
        return a * np.exp(-b * x)
    
    def exponential_model_2(x,a,b,c,d):
        return a*np.exp(-b*x)+c*np.exp(-d*x)

    def fourier_model_1(x,a0,a1,w,b1):
        return a0+a1*np.cos(x*w)+b1*np.sin(x*w)
    
    
    def pow_1(x,a,b):
        return a*np.power(x,b)
    
    def pow_2(x,a,b,c):
        return a*np.power(x,b)+c
    
    def fraction_01(x,p1,q1):
        return (p1)/(x**1+q1)
    def fraction_02(x,p1,q1,q2):
        return (p1)/(x**2+q1*(x**1)+q2)
    def fraction_03(x,p1,q1,q2,q3):
        return (p1)/(x**3+q1*(x**2)+q2*(x**1)+q3)
    def fraction_04(x,p1,q1,q2,q3,q4):
        return (p1)/(x**4+q1*(x**3)+q2*(x**2)+q3*(x**1)+q4)
    def fraction_11(x,p1,p2,q1):
        return (p1*(x**1)+p2)/(x**1+q1)
    def fraction_12(x,p1,p2,q1,q2):
        return (p1*(x**1)+p2)/(x**2+q1*(x**1)+q2)
    def fraction_13(x,p1,p2,q1,q2,q3):
        return (p1*(x**1)+p2)/(x**3+q1*(x**2)+q2*(x**1)+q3)
    def fraction_14(x,p1,p2,q1,q2,q3,q4):
        return (p1*(x**1)+p2)/(x**4+q1*(x**3)+q2*(x**2)+q3*(x**1)+q4)
    def fraction_21(x,p1,p2,p3,q1):
        return (p1*(x**2)+p2*(x**1)+p3)/(x**1+q1)
    def fraction_22(x,p1,p2,p3,q1,q2):
        return (p1*(x**2)+p2*(x**1)+p3)/(x**2+q1*(x**1)+q2)
    def fraction_23(x,p1,p2,p3,q1,q2,q3):
        return (p1*(x**2)+p2*(x**1)+p3)/(x**3+q1*(x**2)+q2*(x**1)+q3)
    def fraction_24(x,p1,p2,p3,q1,q2,q3,q4):
        return (p1*(x**2)+p2*(x**1)+p3)/(x**4+q1*(x**3)+q2*(x**2)+q3*(x**1)+q4)
    def fraction_31(x,p1,p2,p3,p4,q1):
        return (p1*(x**3)+p2*(x**2)+p3*(x**1)+p4)/(x**1+q1)
    def fraction_32(x,p1,p2,p3,p4,q1,q2):
        return (p1*(x**3)+p2*(x**2)+p3*(x**1)+p4)/(x**2+q1*(x**1)+q2)
    def fraction_33(x,p1,p2,p3,p4,q1,q2,q3):
        return (p1*(x**3)+p2*(x**2)+p3*(x**1)+p4)/(x**3+q1*(x**2)+q2*(x**1)+q3)
    def fraction_34(x,p1,p2,p3,p4,q1,q2,q3,q4):
        return (p1*(x**3)+p2*(x**2)+p3*(x**1)+p4)/(x**4+q1*(x**3)+q2*(x**2)+q3*(x**1)+q4)
    def fraction_41(x,p1,p2,p3,p4,p5,q1):
        return (p1*(x**4)+p2*(x**3)+p3*(x**2)+p4*(x**1)+p5)/(x**1+q1)
    def fraction_42(x,p1,p2,p3,p4,p5,q1,q2):
        return (p1*(x**4)+p2*(x**3)+p3*(x**2)+p4*(x**1)+p5)/(x**2+q1*(x**1)+q2)
    def fraction_43(x,p1,p2,p3,p4,p5,q1,q2,q3):
        return (p1*(x**4)+p2*(x**3)+p3*(x**2)+p4*(x**1)+p5)/(x**3+q1*(x**2)+q2*(x**1)+q3)
    def fraction_44(x,p1,p2,p3,p4,p5,q1,q2,q3,q4):
        return (p1*(x**4)+p2*(x**3)+p3*(x**2)+p4*(x**1)+p5)/(x**4+q1*(x**3)+q2*(x**2)+q3*(x**1)+q4)
    def fraction_51(x,p1,p2,p3,p4,p5,p6,q1):
        return (p1*(x**5)+p2*(x**4)+p3*(x**3)+p4*(x**2)+p5*(x**1)+p6)/(x**1+q1)
    def fraction_52(x,p1,p2,p3,p4,p5,p6,q1,q2):
        return (p1*(x**5)+p2*(x**4)+p3*(x**3)+p4*(x**2)+p5*(x**1)+p6)/(x**2+q1*(x**1)+q2)
    def fraction_53(x,p1,p2,p3,p4,p5,p6,q1,q2,q3):
        return (p1*(x**5)+p2*(x**4)+p3*(x**3)+p4*(x**2)+p5*(x**1)+p6)/(x**3+q1*(x**2)+q2*(x**1)+q3)
    def fraction_54(x,p1,p2,p3,p4,p5,p6,q1,q2,q3,q4):
        return (p1*(x**5)+p2*(x**4)+p3*(x**3)+p4*(x**2)+p5*(x**1)+p6)/(x**4+q1*(x**3)+q2*(x**2)+q3*(x**1)+q4)

    def sin_1(x,a1,b1,c1):
        return a1*np.sin(b1*x+c1)
    def sin_2(x,a1,b1,c1,a2,b2,c2):
        return a1*np.sin(b1*x+c1)+a2*np.sin(b2*x+c2)
    def sin_3(x,a1,b1,c1,a2,b2,c2,a3,b3,c3):
        return a1*np.sin(b1*x+c1)+a2*np.sin(b2*x+c2)+a3*np.sin(b3*x+c3)
    def sin_4(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4):
        return a1*np.sin(b1*x+c1)+a2*np.sin(b2*x+c2)+a3*np.sin(b3*x+c3)+a4*np.sin(b4*x+c4)
    def sin_5(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5):
        return a1*np.sin(b1*x+c1)+a2*np.sin(b2*x+c2)+a3*np.sin(b3*x+c3)+a4*np.sin(b4*x+c4)+a5*np.sin(b5*x+c5)
    def sin_6(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6):
        return a1*np.sin(b1*x+c1)+a2*np.sin(b2*x+c2)+a3*np.sin(b3*x+c3)+a4*np.sin(b4*x+c4)+a5*np.sin(b5*x+c5)+a6*np.sin(b6*x+c6)
    def sin_7(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6,a7,b7,c7):
        return a1*np.sin(b1*x+c1)+a2*np.sin(b2*x+c2)+a3*np.sin(b3*x+c3)+a4*np.sin(b4*x+c4)+a5*np.sin(b5*x+c5)+a6*np.sin(b6*x+c6)+a7*np.sin(b7*x+c7)
    def sin_8(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6,a7,b7,c7,a8,b8,c8):
        return a1*np.sin(b1*x+c1)+a2*np.sin(b2*x+c2)+a3*np.sin(b3*x+c3)+a4*np.sin(b4*x+c4)+a5*np.sin(b5*x+c5)+a6*np.sin(b6*x+c6)+a7*np.sin(b7*x+c7)+a8*np.sin(b8*x+c8)
    def gauss_1(x,a1,b1,c1):
        return a1*np.exp(-(((x-b1)/c1)**2))
    def gauss_2(x,a1,b1,c1,a2,b2,c2):
        return a1*np.exp(-(((x-b1)/c1)**2))+a2*np.exp(-(((x-b2)/c2)**2))
    def gauss_3(x,a1,b1,c1,a2,b2,c2,a3,b3,c3):
        return a1*np.exp(-(((x-b1)/c1)**2))+a2*np.exp(-(((x-b2)/c2)**2))+a3*np.exp(-(((x-b3)/c3)**2))
    def gauss_4(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4):
        return a1*np.exp(-(((x-b1)/c1)**2))+a2*np.exp(-(((x-b2)/c2)**2))+a3*np.exp(-(((x-b3)/c3)**2))+a4*np.exp(-(((x-b4)/c4)**2))
    def gauss_5(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5):
        return a1*np.exp(-(((x-b1)/c1)**2))+a2*np.exp(-(((x-b2)/c2)**2))+a3*np.exp(-(((x-b3)/c3)**2))+a4*np.exp(-(((x-b4)/c4)**2))+a5*np.exp(-(((x-b5)/c5)**2))
    def gauss_6(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6):
        return a1*np.exp(-(((x-b1)/c1)**2))+a2*np.exp(-(((x-b2)/c2)**2))+a3*np.exp(-(((x-b3)/c3)**2))+a4*np.exp(-(((x-b4)/c4)**2))+a5*np.exp(-(((x-b5)/c5)**2))+a6*np.exp(-(((x-b6)/c6)**2))
    def gauss_7(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6,a7,b7,c7):
        return a1*np.exp(-(((x-b1)/c1)**2))+a2*np.exp(-(((x-b2)/c2)**2))+a3*np.exp(-(((x-b3)/c3)**2))+a4*np.exp(-(((x-b4)/c4)**2))+a5*np.exp(-(((x-b5)/c5)**2))+a6*np.exp(-(((x-b6)/c6)**2))+a7*np.exp(-(((x-b7)/c7)**2))
    def gauss_8(x,a1,b1,c1,a2,b2,c2,a3,b3,c3,a4,b4,c4,a5,b5,c5,a6,b6,c6,a7,b7,c7,a8,b8,c8):
        return a1*np.exp(-(((x-b1)/c1)**2))+a2*np.exp(-(((x-b2)/c2)**2))+a3*np.exp(-(((x-b3)/c3)**2))+a4*np.exp(-(((x-b4)/c4)**2))+a5*np.exp(-(((x-b5)/c5)**2))+a6*np.exp(-(((x-b6)/c6)**2))+a7*np.exp(-(((x-b7)/c7)**2))+a8*np.exp(-(((x-b8)/c8)**2))

    

    def expweib(x,a,b):
        return a*b*x**{b-1}*np.exp(-a*(x**{b}))


    # def sin_clluster(n):
    #     s="def sin_{}(x,".format(n)
    #     for i in range(1,n):
    #         s+="a{},b{},c{},".format(i,i,i)
    #     s+="a{},b{},c{}):\n\treturn ".format(n,n,n)

    #     for i in range(1,n):
    #         s+="a{}*np.sin(b{}*x+c{})+".format(i,i,i)
        
    #     s+="a{}*np.sin(b{}*x+c{})".format(n,n,n)
    #     return s

    # def fraction(n,m):
    #     s="def fraction_{}{}".format(n,m)
    #     s+="(x,"
    #     for i in range(1,n+2):
    #         s+="p{},".format(i)
    #     for i in range(1,m):
    #         s+="q{},".format(i)
    #     s+="q{}):\n\treturn ".format(m)
    #     s+="("
    #     for i in range(1,n+1):
    #         s+="p{}*(x**{})+".format(i,n+1-i)+""
    #     s+="p{}".format(n+1)
    #     s+=")/("
    #     s+="x**{}+".format(m)
    #     for i in range(1,m):
    #         s+="q{}*(x**{})+".format(i,m-i)
    #     s+='q{})'.format(m)
    #     return s

    # def gauss(n):
    #     s="def gauss_{}".format(n)
    #     s+="(x,"

    #     for i in range(1,n):
    #         s+="a{},b{},c{},".format(i,i,i)

    #     s+="a{},b{},c{}):\n\treturn ".format(n,n,n)
    #     for i in range(1,n):
    #         s+="a{}*np.exp(-(((x-b{})/c{})**2))+".format(i,i,i)
    #     s+="a{}*np.exp(-(((x-b{})/c{})**2))".format(n,n,n)

class fit:
    def __init__(self, x, y) -> None:
        self.xdata = x
        self.ydata = y
        self.methods_dict = {}
        self.get_methods_dict()
    def get_methods_dict(self):
        for name, method in inspect.getmembers(function, inspect.isfunction):
                self.methods_dict[name] = method
        #print(self.methods_dict)
        #print(self.methods_dict['sin_1'](0,1,2,3))
        #return self.methods_dict
    def fit_func(self,func):
        popt,pcov=curve_fit(func,self.xdata,self.ydata)
        x_fit =np.linspace(min(self.xdata),max(self.xdata),1000)
        y_fit=function.exponential_model_1(x_fit,*popt)
        print(popt)
        plt.scatter(self.xdata,self.ydata,label='Data')
        plt.plot(x_fit,y_fit,'r',label='Fit')
        plt.legend()
        plt.show()

    ## 多项式模拟
    def enumerate_poly(self):
        for i in range(1, 10):
            coefficients = np.polyfit(self.xdata, self.ydata, i)
            polynomial = np.poly1d(coefficients)
            plt.scatter(self.xdata, self.ydata)
            plt.plot(self.xdata, polynomial(self.xdata), color='red')
            plt.title(f'Degree {i} Polynomial Fit')
            plt.show()
    
    ## 指数模拟

    def enumerate_exp(self):
        for i in range(1,3):
            name='exponential_model_{}'.format(i)
            func=self.methods_dict[name]
            try:
                self.fit_func(func)
            except:
                print(name+"is not a suitable model!!!")
    
    def enumerate_fourier(self):
        #for i in range(1,3):
            name='fourier_model_1'
            func=self.methods_dict[name]
            try:
                self.fit_func(func)
            except:
                print(name+"is not a suitable model!!!")
    def enumerate_gauss(self):
        function.gauss_1
        for i in range(1,9):
            name='gauss_{}'.format(i)
            func=self.methods_dict[name]
            try:
                self.fit_func(func)
            except:
                print(name+"is not a suitable model!!!")
            
    def enumerate_fraction(self):
        #function.fraction_01
        for i in range(0,6):
            for j in range(1,6):
                name='fraction_{}{}'.format(i,j)
                func=self.methods_dict[name]
                try:
                    self.fit_func(func)
                except:
                    print(name+"is not a suitable model!!!")

    def enumerate_sin(self):
        for i in range(1,9):
            name='sin_{}'.format(i)
            func=self.methods_dict[name]
            try:
                self.fit_func(func)
            except:
                print(name+"is not a suitable model!!!")
        
    #function.expweib
    def enumerate_weib(self):
        name='weib'
        func=function.expweib
        try:
            self.fit_func(func)
        except:
            print(name+"is not a suitable model!!!")

x = np.array([1, 2, 3, 4, 5])
y = np.array([2.5, 4.5, 4.8, 5.5, 6.0])


a=fit(x,y)
a.enumerate_exp()