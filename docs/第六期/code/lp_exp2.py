from gurobipy import Model, GRB

# 创建模型
m = Model("product_mixed")

# 定义变量
x = m.addVar(lb=0, ub=10, vtype=GRB.CONTINUOUS, name="x")
y = m.addVar(lb=0, ub=20, vtype=GRB.CONTINUOUS, name="y")

# 定义目标函数
m.setObjective(2 * x + 3 * y, GRB.MAXIMIZE)

# 定义约束条件
m.addConstr(x <= 10, "c1")
m.addConstr(y <= 20, "c2")

# 求解模型
m.optimize()

# 打印结果
if m.status == GRB.OPTIMAL:
    print(f"生产 A: {x.x} 件")
    print(f"生产 B: {y.x} 件")
    print(f"最大利润: {m.objVal} 元")
else:
    print("无解")
