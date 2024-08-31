import numpy as np
import random
import matplotlib.pyplot as plt

# 检查解是否满足约束条件的函数
def check_constraints(solution, equality_constraints=None, inequality_constraints=None):
    # equality_constraints: 列表，包含等式约束的lambda表达式
    # inequality_constraints: 列表，包含不等式约束的lambda表达式
    if equality_constraints:
        for eq in equality_constraints:
            if not np.isclose(eq(solution), 0):
                return False
    if inequality_constraints:
        for ineq in inequality_constraints:
            if ineq(solution) > 0:
                return False
    return True

# 修正解以满足约束条件的函数
def repair_solution(solution, equality_constraints=None, inequality_constraints=None):
    # 示例：将修正策略为返回一个随机解，您可以根据需要调整
    while not check_constraints(solution, equality_constraints, inequality_constraints):
        solution = np.random.permutation(solution)
    return solution

# 计算适应度函数
def get_fitness(f, N, Distance, L):
    fitness = []
    for i in range(len(f)):
        length = Distance[f[i][N-1], f[i][0]]  # 最后一个城市到第一个城市的路径长度
        for j in range(L-1):
            length += Distance[f[i][j], f[i][j+1]]  # 计算闭环路径长度
        fitness.append(length)
    return np.array(fitness)

# 轮盘赌选择
def selection(f, fitness, NP):
    total_fitness = np.sum(1 / fitness)  # 适应度值越小越好
    selection_probabilities = (1 / fitness) / total_fitness
    selected_indices = np.zeros(NP, dtype=int)

    for i in range(NP - 1):
        r = random.random()
        partial_sum = 0
        for j in range(len(selection_probabilities)):
            partial_sum += selection_probabilities[j]
            if r <= partial_sum:
                selected_indices[i] = j
                break

    # 精英选择：保留最好的1个个体,需要自己修改
    elite_indices = np.argmin(fitness)
    selected_indices[NP - 1] = elite_indices

    return f[selected_indices], fitness[selected_indices]

# 交叉操作
def crossover(f, PC, L, N, equality_constraints=None, inequality_constraints=None):
    for nind in range(len(f)):
        if PC > random.random():
            nnper = np.random.permutation(len(f))
            A = f[nnper[0]].copy()
            B = f[nnper[1]].copy()
            w = int(np.ceil(L / 5))  # 交叉点个数
            p = np.random.randint(1, N - w + 1)  # 开始点位
            temp = A[p:p+w]
            y = [np.where(B == temp[k])[0][0] for k in range(w)]
            y.sort()
            ttemp = B[y]
            B[y] = temp
            A[p:p+w] = ttemp

            # 检查并修正交叉后的解
            A = repair_solution(A, equality_constraints, inequality_constraints)
            B = repair_solution(B, equality_constraints, inequality_constraints)

            f = np.vstack([f, A, B])
    return f

# 变异操作
def mutation(f, PM, L, equality_constraints=None, inequality_constraints=None):
    for nind in range(len(f)):
        if PM > random.random():
            chrom = f[nind].copy()
            while True:
                s1 = np.random.randint(1, L)
                s2 = np.random.randint(1, L)
                if s1 < s2:
                    break
            if s1 != 1 and s2 != L:
                new_chrom = np.concatenate([chrom[:s1-1], chrom[s1-1:s2][::-1], chrom[s2:]])
            elif s1 == 1 and s2 != L:
                new_chrom = np.concatenate([chrom[:s2][::-1], chrom[s2:]])
            elif s1 != 1 and s2 == L:
                new_chrom = np.concatenate([chrom[:s1-1], chrom[s1-1:][::-1]])
            else:
                new_chrom = chrom[::-1]

            # 检查并修正变异后的解
            new_chrom = repair_solution(new_chrom, equality_constraints, inequality_constraints)
            f = np.vstack([f, new_chrom])
    return f

# 主程序
def main():
    city = np.array([
        [1304, 2312], [3639, 1315], [4177, 2244], [3712, 1399], [3488, 1535], [3326, 1556],
        [3238, 1229], [4196, 1044], [4312, 790], [4386, 570], [3007, 1970], [2562, 1756],
        [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678], [3918, 2179], [4061, 2370],
        [3780, 2212], [3676, 2578], [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2376],
        [3394, 2643], [3439, 3201], [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826], [2370, 2975]
    ])

    N = len(city)
    Distance = np.zeros((N, N))

    # 计算距离矩阵
    for i in range(N):
        for j in range(N):
            Distance[i, j] = np.linalg.norm(city[i] - city[j])

    NP = 20  # 种群大小
    G = 200  # 遗传代数
    L = N  # 编码长度
    PC = 0.7  # 交叉率
    PM = 0.3  # 变异率

    # 初始化种群
    f = np.array([np.random.permutation(L) for _ in range(NP)])

    # 定义约束（如果有的话）
    equality_constraints = [
        lambda x: x[0]-0  # 例如：要求一开始必须是0
    ]
    inequality_constraints = [
        # lambda x:   # 例如：假设所有解的最大值不超过30
    ]

    Rlength = []

    for gen in range(G):
        # 交叉
        f = crossover(f, PC, L, N, equality_constraints, inequality_constraints)
        # 变异
        f = mutation(f, PM, L, equality_constraints, inequality_constraints)
        # 计算适应度
        fitness = get_fitness(f, N, Distance, L)
        # 轮盘赌选择
        f, fitness = selection(f, fitness, NP)
        # 保存最优个体
        Rlength.append(min(fitness))
        best_index = np.argmin(fitness)
        R = f[best_index]

        # 绘图
        plt.subplot(1, 2, 1)
        for i in range(N-1):
            plt.plot([city[R[i], 0], city[R[i+1], 0]], [city[R[i], 1], city[R[i+1], 1]], 'bo-')
        plt.plot([city[R[N-1], 0], city[R[0], 0]], [city[R[N-1], 1], city[R[0], 1]], 'ro-')
        plt.title(f'优化最短距离: {min(fitness)}')
        plt.subplot(1, 2, 2)
        plt.plot(Rlength)
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')
        plt.title('适应度进化曲线')
        plt.pause(0.001)

    plt.show()

if __name__ == '__main__':
    main()
