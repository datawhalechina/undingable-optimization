import gurobipy as gp
from gurobipy import GRB

try:
    # 创建一个模型
    model = gp.Model("example")

    # 创建变量
    x = model.addVar(name="x", lb=0)
    y = model.addVar(name="y", lb=0)

    # 设置目标函数
    model.setObjective(2 * x + 3 * y, GRB.MAXIMIZE)

    # 添加约束条件
    model.addConstr(x + y <= 4, "c0")
    model.addConstr(x - y >= 1, "c1")

    # 优化模型
    model.optimize()

    # 输出结果
    if model.status == GRB.OPTIMAL:
        print(f"Optimal value for x: {x.X}")
        print(f"Optimal value for y: {y.X}")
        print(f"Optimal objective: {model.objVal}")
    else:
        print("No optimal solution found.")

except gp.GurobiError as e:
    print(f"Gurobi error: {e.errno} - {e}")

except AttributeError:
    print("Encountered an attribute error.")