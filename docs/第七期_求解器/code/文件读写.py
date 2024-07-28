import gurobipy as gp
from gurobipy import GRB

# 创建模型
model = gp.Model("TESTPROB")

# 添加变量
x1 = model.addVar(lb=0, ub=1, name="x1")
x2 = model.addVar(lb=0, name="x2")

# 设置目标函数
model.setObjective(x1 + 4 * x2, GRB.MAXIMIZE)

# 添加约束
model.addConstr(2 * x1 + x2 <= 5, "LIM1")
model.addConstr(3 * x1 >= 10, "LIM2")
model.addConstr(2 * x2 == 7, "LIM3")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f'Optimal objective value: {model.objVal}')
    for var in model.getVars():
        print(f'{var.varName}: {var.x}')
    model.write('lp_model.sol')
    model.write('lp_model.bas')
else: 
    model.computeIIS()
    model.write('lp_model.ilp')
    print('No optimal solution found.')

# 写入文件
model.write('lp_model.mps')
model.write('lp_model.lp')
model.write('lp_model.prm')
model.write('lp_model.rlp')
model.write('lp_model.json')
print('写入文件成功。')