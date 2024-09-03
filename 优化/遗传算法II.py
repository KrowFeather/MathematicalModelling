import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义约束：限制我们的路线的起点必须是哪里
equality_constraints = [
    lambda x: x[0]**2 + x[1]**2 - 100  # 例如：要求一开始必须1
]

# 不等式约束：要求我们的路线上满足某一种特殊要求
inequality_constraints = [

]

# 目标函数，你希望最大化或者最小化的函数，具体可以通过我们的一个
def object_function(x):
    return (x[0]**2 + x[1]**3 + x[0]**4)

######################################遗传算法相关实现##################################################

history_max = -1
history_best_solution = None

# 检查我们的函数是否满足我们的等式，不等式约束
def check_constraints(solution, equality_constraints=None, inequality_constraints=None):
    if equality_constraints:
        for eq in equality_constraints:
            if not np.isclose(eq(solution), 0):
                return False
    if inequality_constraints:
        for ineq in inequality_constraints:
            if ineq(solution) > 0:
                return False
    return True

# 不满足我们的解，我们就修改我们的对应的解
def repair_solution(solution, equality_constraints=None, inequality_constraints=None):
    while not check_constraints(solution, equality_constraints, inequality_constraints):
        solution = np.random.randint(low=0, high=100, size=len(solution))
    return solution

# 计算我们的一个解的对应的适应度
def get_fitness(f):
    global history_max, history_best_solution
    fitness = []
    for solution in f:
        fitness_value = object_function(solution)
        if fitness_value > history_max:
            history_max = fitness_value
            history_best_solution = solution

    for solution in f:
        fitness_value = object_function(solution) / history_max
        fitness.append(fitness_value)
    return np.array(fitness)

# 轮盘赌选择淘汰，其中我们的适应度越高，我们被淘汰的概率阅读
def selection(f, fitness, NP):
    fitness = np.where(fitness <= 0, np.finfo(float).eps, fitness)  # 确保适应度值为正
    fitness = 1 / (fitness + 0.01)  # 适应度越高，被选择的概率越小
    fitness_sum = np.sum(fitness)
    fitness_prob = fitness / fitness_sum
    selected_indices = np.random.choice(range(len(f)), size=NP, p=fitness_prob)
    return f[selected_indices], fitness[selected_indices]

# 交叉操作
def crossover(f, PC, L, N, equality_constraints, inequality_constraints):
    for i in range(0, len(f), 2):
        if np.random.rand() < PC:
            parent1 = f[i]
            parent2 = f[i + 1]
            crossover_point = np.random.randint(0, L)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            child1 = repair_solution(child1, equality_constraints, inequality_constraints)
            child2 = repair_solution(child2, equality_constraints, inequality_constraints)
            f[i] = child1
            f[i + 1] = child2
    return f

# 变异操作
def mutation(f, PM, L, equality_constraints, inequality_constraints):
    for i in range(len(f)):
        if np.random.rand() < PM:
            solution = f[i]
            mutation_point = np.random.randint(0, L)
            solution[mutation_point] = np.random.randint(0, 100)
            solution = repair_solution(solution, equality_constraints, inequality_constraints)
            f[i] = solution
    return f

#########################################运行##################################################
def main():
    ##########################调参#####################################
    N = 2
    NP = 20  # 种群大小
    G = 100  # 遗传代数
    L = 2  # 编码长度
    PC = 0.7  # 交叉率
    PM = 0.3  # 变异率
    # 初始化种群
    f = []
    while len(f) < NP:
        individual = [np.random.randint(low=0, high=100), np.random.randint(low=0, high=100)]
        if check_constraints(individual, equality_constraints=equality_constraints, inequality_constraints=inequality_constraints):
            f.append(individual)

    f = np.array(f)

    Rlength = []
    best_solutions = []

    for gen in range(G):
        # 交叉
        f = crossover(f, PC, L, N, equality_constraints, inequality_constraints)
        # 变异
        f = mutation(f, PM, L, equality_constraints, inequality_constraints)
        # 计算适应度
        fitness = get_fitness(f)
        # 轮盘赌选择
        f, fitness = selection(f, fitness, NP)
        # 保存最优个体
        Rlength.append(min(fitness))
        best_index = np.argmin(fitness)
        R = f[best_index]
        best_solutions.append(R)

        # 绘图
        print(f"Generation {gen + 1}: Best solution = {R}, Best fitness = {min(fitness)}")

    # 输出历史最佳个体
    print(f"History Best Solution: {history_best_solution}, History Best Fitness: {history_max}")

    # 绘制最佳解的适应度值随迭代次数的变化图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(Rlength)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness over Generations')

    # 绘制最佳解的坐标随迭代次数的变化图
    plt.subplot(1, 2, 2)
    best_solutions = np.array(best_solutions)
    plt.plot(best_solutions[:, 0], label='x1')
    plt.plot(best_solutions[:, 1], label='x2')
    plt.xlabel('Generation')
    plt.ylabel('Best Solution Coordinates')
    plt.title('Best Solution Coordinates over Generations')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()