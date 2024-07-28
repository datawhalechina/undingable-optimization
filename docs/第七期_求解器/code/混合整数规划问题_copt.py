import numpy as np
import coptpy as cp
from coptpy import COPT, quicksum

# 设置随机种子，确保程序每次运行结果一致
np.random.seed(1)

# 定义候选设施地点和需求点数量
num_facilities = 5
num_customers  = 10

# 随机生成设施建设成本和运输成本
facility_costs  = np.random.randint(10, 100, size=num_facilities)
transport_costs = np.random.randint(1, 10, size=(num_facilities, num_customers))
demands         = np.random.randint(5, 15, size=num_customers)

# 创建模型
env   = cp.Envr()
model = env.createModel('FacilityLocation')

# 添加变量
x = model.addVars(num_facilities, vtype=COPT.BINARY, nameprefix="x")  # 设施建设决策变量
y = model.addVars(num_facilities, num_customers, vtype=COPT.CONTINUOUS, nameprefix="y")  # 运输数量变量

# 设定目标函数：最小化总成本
model.setObjective(quicksum(facility_costs[i] * x[i] \
                            for i in range(num_facilities)) \
                                + \
                   quicksum(transport_costs[i][j] * y[i, j] \
                            for i in range(num_facilities) \
                            for j in range(num_customers)), \
                    sense=COPT.MINIMIZE)

# 添加约束条件
# 每个需求点的需求必须得到满足
model.addConstrs((quicksum(y[i, j] \
                    for i in range(num_facilities)) == demands[j] \
                    for j in range(num_customers)), \
                    nameprefix="demand")

# 运输数量约束
model.addConstrs((y[i, j] <= demands[j] * x[i] \
                  for i in range(num_facilities) \
                  for j in range(num_customers)), \
                    nameprefix="transport")

# 优化模型
model.solve()

# 输出结果
if model.status == COPT.OPTIMAL:
    print('Total cost: %g' % model.objval)
    print('Facilities built:')
    for i in range(num_facilities):
        if x[i].x > 0.5:
            print(f'Facility {i} is built.')
    print('Transportation plan:')
    for i in range(num_facilities):
        for j in range(num_customers):
            if y[i, j].x > 0.1:
                print(f'Facility {i} transports {y[i, j].x} units to Customer {j}.')
else:
    print('No optimal solution found.')
