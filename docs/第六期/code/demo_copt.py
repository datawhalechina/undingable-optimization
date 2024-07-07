import coptpy as cp

try:
    # 创建一个模型
    env = cp.Envr()
    model = env.createModel("example")

    # 创建变量
    x = model.addVar(lb=0, name="x")
    y = model.addVar(lb=0, name="y")

    # 设置目标函数
    model.setObjective(2 * x + 3 * y, cp.COPT.MAXIMIZE)

    # 添加约束条件
    model.addConstr(x + y <= 4, name="c0")
    model.addConstr(x - y >= 1, name="c1")

    # 优化模型
    model.solve()

    # 输出结果
    if model.status == cp.COPT.OPTIMAL:
        print(f"Optimal value for x: {x.x}")
        print(f"Optimal value for y: {y.x}")
        print(f"Optimal objective: {model.objval}")
    else:
        print("No optimal solution found.")

except cp.CoptError as e:
    print(f"Copt error: {e.errno} - {e}")

except AttributeError:
    print("Encountered an attribute error.")