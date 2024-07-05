"""
求解问题：

  Maximize:
    3.1 x + 2.3 y + 1.2 z

  Subject to:
    1.5 x + 1.2 y + 1.8 z <= 2.6
    0.8 x + 0.6 y + 0.9 z >= 1.2

  where:
    0.1 <= x <= 0.6
    0.2 <= y <= 1.5
    0.3 <= z <= 2.8
"""

import gurobipy as gp
from gurobipy import GRB


# 创建模型
model = gp.Model("lp_ex1")

# 添加变量: x, y, z
x = model.addVar(lb=0.1, ub=0.6, name="x", vtype=GRB.CONTINUOUS)
y = model.addVar(lb=0.2, ub=1.5, name="y", vtype=GRB.CONTINUOUS)
z = model.addVar(lb=0.3, ub=2.8, name="z", vtype=GRB.CONTINUOUS)

# Add constraints
model.addConstr(1.5*x + 1.2*y + 1.8*z <= 2.6)
model.addConstr(0.8*x + 0.6*y + 0.9*z >= 1.2)

# Set objective function
model.setObjective(1.2*x + 1.8*y + 2.1*z, GRB.MAXIMIZE)


# Solve the model
model.optimize()

# Analyze solution
if model.status == GRB.OPTIMAL:
  print("Objective value: {}".format(model.objval))
  allvars = model.getVars()

  print("Variable solution:")
  for var in allvars:
    print(" x[{0}]: {1}".format(var.index, var.x))


  # Write model, solution and modified parameters to file
  # model.write("lp_ex1.mps")
  # model.write("lp_ex1.bas")
  # model.write("lp_ex1.sol")
  # model.write("lp_ex1.par")