# 第一章 优化求解器安装指南

首先，我们来了解一下优化求解器。如果每次遇到新的优化问题都要重新编写优化算法，那将会非常低效并且重复造轮子。而**优化求解器已经封装好了各类优化算法**，我们只需构建模型并将其转化为代码交给求解器求解，便可快速高效地获得结果，避免重复劳动。

本文目录: 

- [第一章 优化求解器安装指南](#第一章-优化求解器安装指南)
  - [1.1 求解器简介](#11-求解器简介)
  - [1.2 Gurobi 安装](#12-gurobi-安装)
  - [1.3 Copt 安装](#13-copt-安装)
  - [1.4 商用授权](#14-商用授权)
  - [参考资料](#参考资料)


## 1.1 求解器简介

优化求解器是解决复杂优化问题的高效工具，避免了重复编写算法的繁琐工作，简化了问题求解过程，使得用户能够快速获取最优解。在学术研究、工业应用、金融、物流等领域有广泛应用。以下是一些常用的优化求解器：

- **Gurobi**: 以高性能和易用性著称的商用优化求解器，支持线性规划、整数规划等多种优化问题。
- **CPLEX** : IBM 提供的商用优化求解器，功能强大，广泛应用于工业界。
- **Copt**  : 一个由国内公司开发的高性能的商业优化求解器，专注于提供高效的求解能力。
- **SCIP**  : 一个开源的求解器，特别适用于混合整数规划问题。
- **GLPK**  : GNU 提供的开源线性规划套件，适用于线性规划和混合整数规划问题。
- **Xpress**：一款由FICO开发的高性能优化求解器，用于解决线性规划（LP）、混合整数规划（MIP）、非线性规划（NLP）等复杂优化问题。

## 1.2 Gurobi 安装

在 Python 中安装 Gurobi 需要以下步骤：

1. **安装 Gurobi**: 你可以使用以下命令通过 pip 安装 Gurobi Python 接口：
    ```bash
    pip install gurobipy
    ```

2. **个人用户许可证**: 个人用户在首次使用 Gurobi 时，如果是非商用目的，可以不申请许可证，但求解的模型规模会受到限制，无法求解大规模的问题。

3. [**示例代码**](./code/demo_gurobi.py):
    ```python
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
    ```

## 1.3 Copt 安装

在 Python 中安装 Copt 需要以下步骤：

1. **安装 Copt**: 你可以使用以下命令通过 pip 安装 Copt Python 接口：
    ```bash
    pip install coptpy
    ```

2. **个人用户许可证**: 个人用户在首次使用 Copt 时，如果是非商用目的，可以不申请许可证，但求解的模型规模会受到限制，无法求解大规模的问题。

3. [**示例代码**](./code/demo_copt.py):
    ```python
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
    ```

## 1.4 商用授权

如果需要在商业环境中使用这些优化求解器或求解大规模优化问题，需要在求解器官网上申请商用许可证。以下是一些常用求解器的官网链接：

- [Gurobi 官网](https://www.gurobi.com)
- [Copt 官网](http://www.shanshu.ai)

请根据求解器官网的指示安装软件，申请许可证，以合法使用这些工具进行优化求解。

如果你是**在校学生**，可以申请学术许可证（Academic License），例如 Gurobi 的 Academic License 可以免费试用一年。


## 参考资料
[1] [Gurobi Optimizer Example Tour](https://www.gurobi.com/documentation/9.5/examples/index.html) <br>
[2] [Gurobi Optimizer Reference Manual](https://www.gurobi.com/documentation/9.5/refman/index.html) <br>
[3] [COPT 安装教程](https://www.shanshu.ai/copt-document/detail?docType=1&id=10) <br>
[4] [COPT 代码示例](https://www.shanshu.ai/copt-document/detail?docType=4&id=21)