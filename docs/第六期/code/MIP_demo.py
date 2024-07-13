import gurobipy as gp
from gurobipy import GRB

# 参数设置
m = 5  # 设施数量
n = 10  # 客户数量
f = [12, 20, 15, 18, 22]  # 设施建设成本
c = [
    [3, 2, 1, 4, 5, 6, 7, 8, 9, 10],
    [5, 6, 7, 8, 3, 2, 1, 4, 5, 6],
    [8, 9, 10, 11, 7, 6, 5, 4, 3, 2],
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    [1, 4, 2, 3, 5, 7, 9, 11, 13, 15]
]  # 服务成本

# 创建模型
model = gp.Model("facility_location")

# 设置日志参数
model.setParam('LogToConsole', 1)  # 控制台输出日志
model.setParam('LogFile', 'MIP_demo.log')  # 将日志写入文件

# 创建变量
y = model.addVars(m, vtype=GRB.BINARY, name="y")
x = model.addVars(m, n, vtype=GRB.BINARY, name="x")

# 设置目标函数
model.setObjective(gp.quicksum(f[i] * y[i] for i in range(m)) + gp.quicksum(c[i][j] * x[i, j] for i in range(m) for j in range(n)), GRB.MINIMIZE)

# 添加约束条件
for j in range(n):
    model.addConstr(gp.quicksum(x[i, j] for i in range(m)) == 1, name=f"cust_{j}")

for i in range(m):
    for j in range(n):
        model.addConstr(x[i, j] <= y[i], name=f"assign_{i}_{j}")

# 求解模型
model.optimize()

# 打印结果
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.ObjVal}")
    for i in range(m):
        if y[i].X > 0.5:
            print(f"Facility {i} is built.")
            for j in range(n):
                if x[i, j].X > 0.5:
                    print(f"  Customer {j} is served by facility {i}.")
else:
    print("No optimal solution found.")
