import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("MILP_example")

# 添加变量
x1 = model.addVar(vtype=GRB.INTEGER, name="x1")
x2 = model.addVar(vtype=GRB.INTEGER, name="x2")
x3 = model.addVar(vtype=GRB.CONTINUOUS, name="x3")
x4 = model.addVar(vtype=GRB.CONTINUOUS, name="x4")

# 设置目标函数
model.setObjective(10*x1 + 15*x2 + 20*x3 + 25*x4, GRB.MAXIMIZE)

# 添加约束
model.addConstr(2*x1 + 3*x2 + 4*x3 + 5*x4 <= 100, "c1")
model.addConstr(3*x1 + 2*x2 + 5*x3 + 4*x4 <= 120, "c2")

# 优化模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for v in model.getVars():
        print(f"{v.varName} = {v.x}")
    print(f"Objective value: {model.objVal}")
else:
    print("No optimal solution found.")
