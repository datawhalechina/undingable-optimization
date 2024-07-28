import coptpy as cp
from coptpy import COPT

# 物品信息
weights = [2, 3, 4, 5] # 每件物品的重量
values  = [3, 4, 5, 6] # 每件物品的价值
capacity = 10 # 背包的容量

# 创建 COPT 环境
env = cp.Envr()

# 创建模型
model = env.createModel("knapsack_problem")

# 设置日志输出
model.setLogFile('knapsack_problem_copt.log')

# 添加变量
x = model.addVars(len(weights), lb=0, vtype=COPT.INTEGER, nameprefix="x")

# 设置目标函数
model.setObjective(cp.quicksum(values[i] * x[i] for i in range(len(weights))), COPT.MAXIMIZE)

# 添加约束条件
model.addConstr(cp.quicksum(weights[i] * x[i] for i in range(len(weights))) <= capacity, "weight_limit")

# 模型求解
model.solve()

# 输出结果
if model.status == COPT.OPTIMAL:
    print("Optimal solution found:")
    for v in model.getVars():
        print(f"{v.getName()}: {v.x}")
    print(f"Objective value: {model.objVal}")
else:
    print("No optimal solution found.")