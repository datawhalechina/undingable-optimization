import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("mixed_integer_example")

# 添加变量
x = model.addVar(vtype=GRB.CONTINUOUS, name="x")
y = model.addVar(vtype=GRB.CONTINUOUS, name="y")
z = model.addVar(vtype=GRB.INTEGER, name="z")

# 设置目标函数
model.setObjective(3 * x + 4 * y + 5 * z, GRB.MAXIMIZE)

# 添加约束条件
model.addConstr(2 * x + 3 * y + 4 * z <= 10, "c1")
model.addConstr(x + 2 * y + 3 * z <= 7, "c2")

# 初步求解松弛问题（忽略整数约束）
model.optimize()

# 输出松弛问题的结果，作为整数规划的初始解
print("Relaxed solution:")
for v in model.getVars():
    print(f"{v.varName}: {v.x}")

print('\n' + '-' * 80 + '\n')

# 设置整数约束
x.vtype = GRB.INTEGER
y.vtype = GRB.INTEGER

# 重新求解整数规划问题
model.optimize()

# 输出整数规划的结果
if model.status == GRB.OPTIMAL:
    print("Optimal solution:")
    for v in model.getVars():
        print(f"{v.varName}: {v.x}")
    print(f"Objective value: {model.objVal}")
else:
    print("No optimal solution found.")
