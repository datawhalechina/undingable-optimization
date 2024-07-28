"""
问题：

  Maximize:
    40x + 50y

  Subject to:
    2x + 4y <= 100
    3x + y <= 50

  where:
    x > = 0
    y > = 0
    
"""


import gurobipy as gp
from gurobipy import GRB

# 创建一个模型
model = gp.Model("Maximize_Profit")

# 创建变量
# name: 变量名；vtype: 变量类型；lb: 下界；ub: 上界
x = model.addVar(name="A", vtype=GRB.CONTINUOUS, lb=0)
y = model.addVar(name="B", vtype=GRB.CONTINUOUS, lb=0)

# 设置目标函数
model.setObjective(40 * x + 50 * y, GRB.MAXIMIZE)

# 添加约束条件
## 变量范围约束可直接在创建变量时指定，更加高效简洁
model.addConstr(2 * x + 4 * y <= 100, "Production_Time") # 生产时间约束
model.addConstr(3 * x + y <= 50, "Raw_Material")  # 原材料约束

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print(f"产品 A 的最优生产数量为: {x.X}")
    print(f"产品 B 的最优生产数量为: {y.X}")
    print(f"最大利润为: {model.objVal}")
else:
    print("未找到最优解。")

