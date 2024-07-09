import gurobipy as gp
from gurobipy import GRB

# 设置随机种子和参数
import numpy as np
np.random.seed(0)

# 定义集合
num_factories  = 4
num_warehouses = 3
num_markets    = 5
num_products   = 2

# 生成随机成本和需求数据
production_costs  = np.random.randint(10, 50, size=(num_factories, num_products))
transport_cost_fw = np.random.randint(1, 20, size=(num_factories, num_warehouses, num_products))
transport_cost_wm = np.random.randint(1, 20, size=(num_warehouses, num_markets, num_products))
market_demand     = np.random.randint(10, 100, size=(num_markets, num_products))

# 创建模型
model = gp.Model("supply_chain_network_design")

# 添加变量
x = model.addVars(num_factories, num_products, vtype=GRB.CONTINUOUS, name="x")
y = model.addVars(num_factories, num_warehouses, num_products, vtype=GRB.CONTINUOUS, name="y")
z = model.addVars(num_warehouses, num_markets, num_products, vtype=GRB.CONTINUOUS, name="z")

# 设置目标函数：最小化总成本
model.setObjective(
    gp.quicksum(production_costs[f, p] * x[f, p] for f in range(num_factories) for p in range(num_products)) +
    gp.quicksum(transport_cost_fw[f, w, p] * y[f, w, p] for f in range(num_factories) for w in range(num_warehouses) for p in range(num_products)) +
    gp.quicksum(transport_cost_wm[w, m, p] * z[w, m, p] for w in range(num_warehouses) for m in range(num_markets) for p in range(num_products)),
    GRB.MINIMIZE
)

# 添加约束：市场需求必须得到满足
model.addConstrs(
    (gp.quicksum(z[w, m, p] for w in range(num_warehouses)) == market_demand[m, p] for m in range(num_markets) for p in range(num_products)),
    name="demand_satisfaction"
)

# 添加约束：仓库的流入必须等于流出
model.addConstrs(
    (gp.quicksum(y[f, w, p] for f in range(num_factories)) == gp.quicksum(z[w, m, p] for m in range(num_markets)) for w in range(num_warehouses) for p in range(num_products)),
    name="warehouse_balance"
)

# 添加约束：仓库和市场的供应量不能超过工厂的生产量
model.addConstrs(
    (y[f, w, p] <= x[f, p] for f in range(num_factories) for w in range(num_warehouses) for p in range(num_products)),
    name="production_capacity"
)

# 优化模型
model.optimize()

# 输出结果
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    for f in range(num_factories):
        for p in range(num_products):
            print(f"Factory {f} produces {x[f, p].x} units of product {p}.")
            for w in range(num_warehouses):
                if y[f, w, p].x > 0:
                    print(f"  Ships {y[f, w, p].x} units of product {p} to warehouse {w}.")
    for w in range(num_warehouses):
        for m in range(num_markets):
            for p in range(num_products):
                if z[w, m, p].x > 0:
                    print(f"Warehouse {w} ships {z[w, m, p].x} units of product {p} to market {m}.")
    print(f"Objective value: {model.objVal}")
else:
    print("No optimal solution found.")
