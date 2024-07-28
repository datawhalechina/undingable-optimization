import gurobipy as gp
from gurobipy import GRB

# 物品信息
weights = [2, 3, 4, 5] # 每件物品的重量
values  = [3, 4, 5, 6] # 每件物品的价值
capacity = 10 # 背包的容量

# 创建模型
model = gp.Model("knapsack_problem")

# 添加变量
x = model.addVars(len(weights), lb=0, vtype=GRB.INTEGER, name="x")

# 设置目标函数
model.setObjective(gp.quicksum(values[i] * x[i] for i in range(len(weights))), GRB.MAXIMIZE)

# 添加约束条件
model.addConstr(gp.quicksum(weights[i] * x[i] for i in range(len(weights))) <= capacity, "weight_limit")

# 模型求解
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for v in model.getVars():
        print(f"{v.varName}: {v.x}")
    print(f"Objective value: {model.objVal}")
else:
    print("No optimal solution found.")