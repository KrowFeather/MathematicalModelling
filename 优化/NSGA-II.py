# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

# Importing required modules
import math
import random
import matplotlib.pyplot as plt

# First function to optimize, 目标函数1
def function1(x):
    value = -x**2  # 计算目标函数1的值
    return value

# Second function to optimize，目标函数2
def function2(x):
    value = -(x-2)**2  # 计算目标函数2的值
    return value

# Function to find index of list，备用函数，找到list中a的下标在哪里
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i  # 返回元素a在列表中的索引
    return -1  # 如果找不到，返回-1

# Function to sort by values，很弱的排序，o(n^2)
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf  # 将已选中的最小值设为无穷大，避免重复选择
    return sorted_list

# 非支配排序，选取严格比x要打的元素
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]  # 存储每个解支配的解
    front = [[]]  # 存储非支配前沿
    n = [0 for i in range(0, len(values1))]  # 存储每个解被支配的次数
    rank = [0 for i in range(0, len(values1))]  # 存储每个解的等级

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or \
               (values1[p] >= values1[q] and values2[p] > values2[q]) or \
               (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)  # p支配q
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or \
                 (values1[q] >= values1[p] and values2[q] > values2[p]) or \
                 (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1  # q支配p
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)  # 如果p不被任何解支配，将其加入第一前沿

    i = 0
    while(front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if(n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)  # 如果q不被任何解支配，将其加入下一前沿
        i = i + 1
        front.append(Q)

    del front[len(front)-1]  # 删除最后一个空的前沿
    return front

# 计算种群的拥挤程度
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]  # 初始化拥挤距离
    sorted1 = sort_by_values(front, values1[:])  # 按第一个目标函数排序
    sorted2 = sort_by_values(front, values2[:])  # 按第二个目标函数排序
    distance[0] = 4444444444444444  # 边界点的拥挤距离设为无穷大
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted1[k+1]] - values2[sorted1[k-1]]) / (max(values1) - min(values1))
    for k in range(1, len(front)-1):
        distance[k] = distance[k] + (values1[sorted2[k+1]] - values2[sorted2[k-1]]) / (max(values2) - min(values2))
    return distance

# 补充我们的约束（如果存在的话）
def check_constraints(x):
    # 线性约束：x >= -10
    if x < -10:
        return False
    # 非线性约束：x^2 <= 100
    if x**2 > 100:
        return False
    return True

# 种群变异
def crossover(a, b):
    r = random.random()  # 生成一个随机数
    if r > 0.5:
        return mutation((a + b) / 2)  # 交叉操作
    else:
        return mutation((a - b) / 2)

# 种群变异
def mutation(solution):
    mutation_prob = random.random()  # 生成一个随机数
    if mutation_prob < 1:
        solution = min_x + (max_x - min_x) * random.random()  # 变异操作
    while not check_constraints(solution):
        solution = min_x + (max_x - min_x) * random.random()  # 重新生成直到满足约束
    return solution

# 初始化种群的大小，迭代的次数
pop_size = 20
max_gen = 921

# 初始化我们的各个初始解
min_x = -55
max_x = 55
solution = [min_x + (max_x - min_x) * random.random() for i in range(0, pop_size)]  # 生成初始种群

# 确保初始种群满足约束
solution = [x for x in solution if check_constraints(x)]
while len(solution) < pop_size:
    new_solution = min_x + (max_x - min_x) * random.random()
    if check_constraints(new_solution):
        solution.append(new_solution)

gen_no = 0
while(gen_no < max_gen):
    # 计算这一代的所有结果
    function1_values = [function1(solution[i]) for i in range(0, pop_size)]  # 计算第一个目标函数的值
    function2_values = [function2(solution[i]) for i in range(0, pop_size)]  # 计算第二个目标函数的值
    
    # non_dominated 中国表示我们没有被支配的解
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])  # 进行非支配排序
    
    ##-=--------------------------------调试信息------------------------------------------##
    print("The best front for Generation number ", gen_no, " is") 
    for valuez in non_dominated_sorted_solution[0]:
        print(round(solution[valuez], 3), end=" ")  # 打印当前代的最优前沿
    print("\n")
    ##----------------------------------调试信息-----------------------------------------##
    
    # 计算种群的拥挤程度
    crowding_distance_values = []
    for i in range(0, len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))  # 计算拥挤距离
    solution2 = solution[:]
    
    # 生成新的种群
    while(len(solution2) != 2 * pop_size):
        a1 = random.randint(0, pop_size-1)
        b1 = random.randint(0, pop_size-1)
        new_solution = crossover(solution[a1], solution[b1])  # 交叉生成新解
        if check_constraints(new_solution):
            solution2.append(new_solution)
    
    # 计算新的种群的拥挤程度
    function1_values2 = [function1(solution2[i]) for i in range(0, 2 * pop_size)]  # 计算新种群的第一个目标函数值
    function2_values2 = [function2(solution2[i]) for i in range(0, 2 * pop_size)]  # 计算新种群的第二个目标函数值
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])  # 对新种群进行非支配排序
    crowding_distance_values2 = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))  # 计算新种群的拥挤距离
    
    # 计算新答案
    new_solution = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in range(0, len(non_dominated_sorted_solution2[i]))]
        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])  # 按拥挤距离排序
        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0, len(non_dominated_sorted_solution2[i]))]
        front.reverse()
        for value in front:
            new_solution.append(value)
            if(len(new_solution) == pop_size):
                break
        if (len(new_solution) == pop_size):
            break
    solution = [solution2[i] for i in new_solution]  # 更新种群
    gen_no = gen_no + 1  # 增加代数

# 让我们绘制最终的前沿
function1 = [i * -1 for i in function1_values]  # 反转目标函数值以便绘图
function2 = [j * -1 for j in function2_values]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)  # 绘制散点图
plt.show()  # 显示图表