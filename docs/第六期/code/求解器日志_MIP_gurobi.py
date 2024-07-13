import numpy as np
import matplotlib.pyplot as plt
from gurobipy import *

# 设置随机种子，确保程序每次运行结果一致
rnd = np.random
rnd.seed(1)

# 定义城市和客户点数量
num_customers = 10
xc = rnd.rand(num_customers + 1) * 200  # 随机生成城市的横坐标，范围[0,200]
yc = rnd.rand(num_customers + 1) * 100  # 随机生成城市的纵坐标，范围[0,100]

# 画图查看生成的城市分布
# plt.plot(xc[0], yc[0], c='r', marker='s')  # 仓库/出发点
# plt.scatter(xc, yc, c='b')  # 客户点

# 定义集合和参数
customers = list(range(1, num_customers + 1))  # 客户点集合
nodes = list(range(0, num_customers + 1))  # 所有点集合（仓库+客户点）
arcs = [(i, j) for i in nodes for j in nodes if i != j]  # 城市之间的弧段
distances = {(i, j): np.hypot(xc[i] - xc[j], yc[i] - yc[j]) for i, j in arcs}  # 计算弧段的长度
max_capacity = 20  # 车最大载重
demands = {i: rnd.randint(1, 10) for i in customers}  # 随机生成客户点的需求量，范围[1,10]

# 创建模型
model = Model('CVRP')

# 日志
model.setParam('OutputFlag', True)
model.setParam('LogFile', 'log_demo_MIP.log')
model.setParam('MIPGap', 0.01)

# 添加变量
x = model.addVars(arcs, vtype=GRB.BINARY, name="x")  # 是否链接ij的二元变量
u = model.addVars(customers, vtype=GRB.CONTINUOUS, name="u")  # 车在客户点的累计载货量

# 设定目标函数：最小化总距离
model.modelSense = GRB.MINIMIZE
model.setObjective(quicksum(x[i, j] * distances[i, j] for i, j in arcs))

# 添加约束条件
model.addConstrs((quicksum(x[i, j] for j in nodes if i != j) == 1) for i in customers)
model.addConstrs((quicksum(x[i, j] for i in nodes if i != j) == 1) for j in customers)
model.addConstrs((x[i, j] == 1) >> (u[i] + demands[j] == u[j]) for i, j in arcs if i != 0 and j != 0)
model.addConstrs((u[i] >= demands[i]) for i in customers)
model.addConstrs((u[i] <= max_capacity) for i in customers)

# 优化模型
model.optimize()

# 输出最优解的所有连线，即xij中值接近1的(i,j)
active_arcs = [arc for arc in arcs if x[arc].x > 0.9]
print(active_arcs)

# 画图显示最优路径
for (i, j) in active_arcs:
    plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='r')
plt.plot(xc[0], yc[0], c='g', marker='s')
plt.scatter(xc, yc, c='b')
plt.show()
