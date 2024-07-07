
# 第二章 线性规划问题

接下来我们用一个简单的线性规划问题来展示如何用求解器求解优化问题。

### 2.1 问题描述

假设我们有一个工厂，生产两种产品：产品 A 和产品 B。我们希望确定每种产品的生产数量，以最大化总利润。生产每种产品需要消耗一定的资源，且资源是有限的。

下表总结了产品 A 和产品 B 的相关信息：

| 产品     | 利润（元） | 生产时间（小时） | 原材料（单位） |
|----------|-:|-:|-:|
| 产品 A   | 40                | 2                    | 3                  |
| 产品 B   | 50                | 4                    | 1                  |

- 可用的生产时间总共为 100 小时
- 可用的原材料总共为 50 单位

#### 数学模型

##### 目标
最大化总利润 $Z$。

##### 决策变量
- $x$: 生产的产品 A 的数量
- $y$: 生产的产品 B 的数量

##### 目标函数
最大化总利润：
$\text{Maximize } Z = 40x + 50y$

##### 约束条件
1. 生产时间约束：
$2x + 4y \leq 100$
2. 原材料约束：
$3x + y \leq 50$
3. 非负约束：
$x \geq 0$
$y \geq 0$

### 2.2 用 Gurobi 求解

下面是用 Gurobi 求解这个线性规划问题的代码：

```python
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
```

**注意**，模型和约束的名字可以是中文，但出于兼容性考虑，通常不推荐写中文。

求解结果如下：
![Gurobi 求解结果](./images/线性规划_gurobi_求解结果.png)


## 2.3 用 Copt 求解

下面是用 Gurobi 求解这个线性规划问题的代码：

```python
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
```

求解结果：
![Copt 求解结果](./images/线性规划_copt_求解结果.png)

## 2.4 Gurobi 和 Copt 语法比较

从上面的例子我们可以看到 Copt 和 Gurobi 的语法是非常相似的，但也存在一些差异。

下表为两个求解器的语法对比：


| 功能          | Gurobi 语法                           | COPT 语法                            |
|---------------|--------------------------------------|--------------------------------------|
| 导入库        | `import gurobipy as gp` <br> `from gurobipy import GRB`            | `import coptpy as cp` <br> `from coptpy import COPT`    |
|创建环境| - | `env = cp.Envr()` |
| 创建模型      | `model = gp.Model("model_name")`     | `model = cp.Model("model_name")`     |
| 添加变量      | `x = model.addVar(name="x")`         | `x = model.addVar(name="x")`         |
| 设置变量类型  | `x = model.addVar(vtype=GRB.INTEGER)`| `x = model.addVar(vtype=COPT.INTEGER)` |
| 添加多个变量  | `vars = model.addVars(10)`           | `vars = model.addVars(10)`           |
| 设置目标函数  | `model.setObjective(expr, GRB.MAXIMIZE)` | `model.setObjective(expr, COPT.MAXIMIZE)` |
| 添加约束      | `model.addConstr(x + y <= 10)`       | `model.addConstr(x + y <= 10)`       |
| 添加多个约束  | `model.addConstrs((x[i] + y[i] <= 10 for i in range(10)))` | `model.addConstrs((x[i] + y[i] <= 10 for i in range(10)))` |
| 模型求解      | `model.optimize()`                   | `model.solve()`                      |
| 获取变量值    | `x.X`                                | `x.X`                                |
| 获取目标值    | `model.objVal`                       | `model.objVal`                       |
| 读取模型文件  | `model = gp.read("model_file.mps")`  | `model = cp.read("model_file.mps")`  |
| 写出模型文件  | `model.write("model_file.mps")`      | `model.write("model_file.mps")`      |
| 获取约束松弛  | `model.getConstrByName("c1").slack`  | `model.getConstrByName("c1").slack`  |


### 主要区别

- **创建环境**：Copt 在创建模型之前需要通过 `env = cp.Envr()` 创建环境，而 Gurobi 则不需要。
- **模型求解**：Gurobi 使用 `model.optimize()`，COPT 使用 `model.solve()`。
- **变量定义**：Gurobi 中的变量类型和目标函数常量使用 `GRB`，而 COPT 使用 `COPT`。

更多差异请参阅求解器官网文档：
- [Gurobi 求解器官网文档](https://www.gurobi.com/documentation/11.0/)
- [Copt 求解器官网文档](https://www.shanshu.ai/copt-document)


