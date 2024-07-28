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

import coptpy as cp
from coptpy import COPT

# 创建 COPT 环境
env = cp.Envr()

# 创建模型
model = env.createModel("Maximize_Profit")

# 创建变量
# name: 变量名；vtype: 变量类型；lb: 下界；ub: 上界
x = model.addVar(lb=0, name="A", vtype=COPT.CONTINUOUS)
y = model.addVar(lb=0, name="B", vtype=COPT.CONTINUOUS)

# 设置目标函数
model.setObjective(40 * x + 50 * y, COPT.MAXIMIZE)

# 添加约束条件
## 变量范围约束可直接在创建变量时指定，更加高效简洁
model.addConstr(2 * x + 4 * y <= 100, "Production_Time")  # 生产时间约束
model.addConstr(3 * x + y <= 50, "Raw_Material")   # 原材料约束

# 求解模型
model.solve()

# 输出结果
if model.status == COPT.OPTIMAL:
    print(f"产品 A 的最优生产数量为: {x.x}")
    print(f"产品 B 的最优生产数量为: {y.x}")
    print(f"最大利润为: {model.objval}")
else: 
    print("未找到最优解。")
