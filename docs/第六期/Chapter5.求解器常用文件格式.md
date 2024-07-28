# 第五章 求解器常用文件格式

在求解优化问题后，我们通常需要保存求解结果，有时还需要保存模型。这时就涉及到各种文件格式的存储和读取了。

本文目录：

- [第五章 求解器常用文件格式](#第五章-求解器常用文件格式)
  - [5.1 MPS 文件](#51-mps-文件)
    - [MPS 文件示例](#mps-文件示例)
    - [MPS 文件的组成部分](#mps-文件的组成部分)
      - [1. NAME](#1-name)
      - [2. OBJSENSE](#2-objsense)
      - [3. ROWS](#3-rows)
      - [4. COLUMNS](#4-columns)
        - [结构：](#结构)
      - [5. RHS](#5-rhs)
      - [6. BOUNDS](#6-bounds)
      - [7. RANGES](#7-ranges)
      - [8. ENDATA](#8-endata)
  - [5.2 LP 文件](#52-lp-文件)
      - [基本结构](#基本结构)
      - [格式规则](#格式规则)
      - [示例](#示例)
      - [详细说明](#详细说明)
      - [注意事项](#注意事项)
  - [5.3 ILP 文件](#53-ilp-文件)
      - [什么是IIS？](#什么是iis)
      - [使用Gurobi求解ILP文件](#使用gurobi求解ilp文件)
  - [5.4 SOL 文件](#54-sol-文件)
    - [基本结构](#基本结构-1)
    - [文件生成](#文件生成)
    - [读取SOL文件](#读取sol文件)
  - [参考资料](#参考资料)



本文将介绍求解器常用的几种文件格式，帮助大家掌握这些常见文件格式的读取和写入操作。

以 Gurobi 为例，Gurobi 求解器支持多种文本文件格式：
- **MPS**、REW、**LP**、RLP、**ILP**、OPB、DLP 和 DUA 格式用于保存优化模型。
- MST 格式用于存储 MIP 起始数据。将这些数据导入 MIP 模型可以使求解从已知可行的解开始。
- HNT 格式用于存储 MIP 提示。将此数据导入 MIP 模型中可以引导 MIP 搜索朝向高质量的可行解。
- ORD 格式用于保存 MIP 变量分支优先级。将此数据导入 MIP 模型中会影响搜索策略。
- BAS 格式保存单纯形基信息。将此数据导入连续模型中可以使单纯形算法从给定的单纯形基开始。
- **SOL** 和 JSON 解决方案格式用于保存求解信息。后者包括关于优化结果的附加信息。这两种格式只能在模型优化后写入。
- ATTR 格式存储模型的一系列属性，包括（多个）MIP 起始数据、解决方案、基信息、分区、变量提示和分支优先级。
- PRM 格式保存参数值。将此数据导入模型中会更改引用参数的值。

这些文件格式的写入和读取都非常简单，**读取用`read()`函数，写入用`write()`函数**，例如：

```python
import gurobipy as gp

# 读取 MPS 模型文件
model = gp.read('lp_model.mps')

# 保存 MPS 文件
model.write('lp_model.mps')
```

其他文件格式同理，将文件后缀改为对应的格式即可。注意写入读取的路径。

以下深入介绍几种常用文件格式：

## 5.1 MPS 文件

MPS 文件格式（Mathematical Programming System file format）是一种标准化的文本文件格式，用于描述线性规划（LP）和混合整数线性规划（MILP）问题。它广泛用于工业应用，因为它是一种紧凑且易于交换的格式。MPS 文件格式由一系列固定格式的字段组成，这些字段描述了模型的各个部分。

### MPS 文件示例

以下是一个 MPS 文件示例：

```plaintext
NAME          TESTPROB
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  LIM3
COLUMNS
    XONE      COST       1
    XONE      LIM1       2
    XONE      LIM2       3
    YTWO      COST       4
    YTWO      LIM1       1
    YTWO      LIM3       2
RHS
    RHS1      LIM1       5
    RHS1      LIM2      10
    RHS1      LIM3       7
BOUNDS
 UP BND1      XONE       1
 LO BND1      YTWO       0
ENDATA
```

### MPS 文件的组成部分

MPS 文件格式由几个部分组成，每个部分用特定的关键字标识。这些部分通常包括：

1. **NAME**
2. **OBJSENSE**
3. **ROWS**
4. **COLUMNS**
5. **RHS**
6. **BOUNDS**
7. **RANGES**
8. **ENDATA**

以下是对每个部分的详细介绍：

#### 1. NAME
**作用**: 定义模型的名称。这个部分是可选的。

**示例**:
```plaintext
NAME          TESTPROB
```

#### 2. OBJSENSE
**作用**: **可选项**，定义目标函数的优化方向，即最大化（MAX）还是最小化（MIN）。Gurobi默认是最小化。

**示例**:
```plaintext
OBJSENSE MAX
```


#### 3. ROWS
**作用**: 定义目标函数和约束。每一行表示一个约束或目标函数。

**行类型**:
- `N`: 无限制（free），即目标函数（Objective function）
- `L`: 小于等于（<=）约束
- `G`: 大于等于（>=）约束
- `E`: 等于（=）约束

**示例**:
```plaintext
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  LIM3
```

除了`ROWS`外，可选的`LAZYCONS`和`USERCUTS`也有着和`ROWS`相似的结构，都表示约束的类型。

`LAZYCONS`为惰性约束（Lazy Constraints），其本质也是线性约束，区别在于标准约束在求解过程中始终被考虑，而惰性约束只有在特定条件下才会被动态地添加和强制执行，以提高求解效率。

`USERCUTS`部分为用户自定义的割平面（user cuts）。用户割平面是由用户提供的额外约束，用于帮助混合整数规划（MIP）求解器更高效地**缩小可行解空间，尽快找到最优解**。这些切割平面与惰性约束（LAZYCONS）不同，后者是为了确保解的可行性，而用户切割平面则主要用于改进求解器的性能。

#### 4. COLUMNS
**作用**: 定义变量及其在约束和目标函数中的系数。每一行描述一个或两个变量（列）在特定约束（行）中的系数。

##### 结构：
- **列名**：每行的第一个字段是变量（列）的名称。
- **行名和系数**：后面的字段是这个变量在特定约束（行）中的系数。每个非零系数由行名和浮点数值组成。
- **多行连续性**：同一变量（列）的多行必须在文件中连续出现。这意味着如果一个变量在多个约束中有非零系数，这些系数必须在COLUMNS部分的连续行中列出。

**示例**:
```plaintext
COLUMNS
    XONE      COST       1
    XONE      LIM1       2
    XONE      LIM2       3
    YTWO      COST       4
    YTWO      LIM1       1
    YTWO      LIM3       2
```

解读：
- 变量`XONE`在目标函数`COST`中的系数为1。
- 变量`XONE`在约束`LIM1`中的系数为2，在约束`LIM2`中的约束为3。
- 变量`YTWO`在目标函数`COST`中的系数为4。
- 变量`YTWO`在约束`LIM1`中的系数为1，在约束`LIM2`中的约束为2。


#### 5. RHS
**作用**: 定义约束的右端常数项（Right-Hand Side, RHS）。

**示例**:
```plaintext
RHS
    RHS1      LIM1       5
    RHS1      LIM2      10
    RHS1      LIM3       7
```

解读：
- 约束`LIM1`右端常数为5。
- 约束`LIM2`右端常数为10。
- 约束`LIM3`右端常数为7。


#### 6. BOUNDS
**作用**: 定义变量的上下界。默认情况下，每个变量的取值范围为$[0, \infty)$。

  **边界类型**: 
- `UP`    : 上界（Upper bound）
- `LO`    : 下界（Lower bound）
- `FX`    : 固定值（Fixed value）
- `FR`    : 自由变量（Free variable，无上下界）
- `MI`    : 负无穷（Minus infinity）
- `PL`    : 正无穷（Plus infinity）
- `BV`    : 二进制变量（Binary variable）
- `LI`    : 整数变量的下界（lower bound for integer variable）
- `UI`    : 整数变量的上界（upper bound for integer variable）
- `SC`    : 半连续变量的上界（upper bound for semi-continuous variable）
- `SI`    : 半整数的变量的上界（upper bound for semi-integer variable）

**示例**:
```plaintext
BOUNDS
 UP BND1      XONE       1
 LO BND1      YTWO       0
```

解释：
- 变量`XONE`的上界为1。
- 变量`YTWO`的下界为0。

#### 7. RANGES
**作用**: 定义范围约束（Range constraints）。这部分是可选的，用于定义范围约束的上下界。

**示例**:
```plaintext
RANGES
    RNG1      LIM1       1
    RNG2      LIM2       5
```

#### 8. ENDATA
**作用**: 标识 MPS 文件的结束。

**示例**:
```plaintext
ENDATA
```

## 5.2 LP 文件

LP文件格式（Linear Programming File Format）是一种用于描述线性规划和混合整数规划模型的文本格式，相较于MPS格式，它更易于人类阅读和编写。LP文件格式具有灵活性，但也存在一些限制，比如不能保留列的顺序以及系数的精确数值。

#### 基本结构

LP文件由多个部分组成，每个部分用特定的关键词开始，并捕获优化模型的逻辑片段。部分通常按固定顺序排列，尽管有些部分可以互换。以下是LP文件的主要部分：

1. **目标函数（Objective Function）**
2. **约束（Constraints）**
3. **变量界限（Bounds）**
4. **变量类型（Variable Types）**
5. **结束（End）**

#### 格式规则

- **可读性**：LP文件不依赖固定字段宽度，使用换行符和空白字符来分隔对象。
- **注释**：反斜杠（`\`）符号表示注释，注释符号后的内容会被忽略。
- **变量名和约束名**：每个变量必须有一个唯一的名称，名称最长不超过255个字符，不能以数字或特殊字符（如`+`, `-`, `*`, `^`, `<`, `>`, `=`, `(`, `)`, `[`, `]`, `,`, `:`）开头。名称之间需要用空白字符分隔。

#### 示例

以下是一个简单的LP文件示例：

```plaintext
\ LP format example

Maximize
  x + y + z

Subject To
  c0: x + y = 1
  c1: x + 5 y + 2 z <= 10
  qc0: x + y + [ x ^ 2 - 2 x * y + 3 y ^ 2 ] <= 5

Bounds
  0 <= x <= 5
  z >= 2

Generals
  x y z

End
```

#### 详细说明

1. **目标函数（Objective Function）**
   - 以`Maximize`或`Minimize`开始，接下来是目标函数的定义。例如：
   ```plaintext
   Maximize
     x + y + z
   ```
   表示目标函数是最大化 `x + y + z`。

2. **约束（Constraints）**
   - 以`Subject To`开始，每行定义一个约束。例如：
   ```plaintext
   Subject To
     c0: x + y = 1
     c1: x + 5 y + 2 z <= 10
   ```
   定义了两个线性约束，`c0`表示 `x + y = 1`，`c1`表示 `x + 5 y + 2 z <= 10`。
   - 支持二次约束，如：
   ```plaintext
   qc0: x + y + [ x ^ 2 - 2 x * y + 3 y ^ 2 ] <= 5
   ```

3. **变量界限（Bounds）**
   - 以`Bounds`开始，定义每个变量的上下界。例如：
   ```plaintext
   Bounds
     0 <= x <= 5
     z >= 2
   ```
   定义了变量 `x` 的范围是0到5，变量 `z` 的下界是2。

4. **变量类型（Variable Types）**
   - 以`Generals`或`Binary`开始，定义整数变量或二进制变量。例如：
   ```plaintext
   Generals
     x y z
   ```

5. **结束（End）**
   - `End`表示LP文件的结束。

#### 注意事项

- 空白字符：空白字符在LP格式中是必须的。例如，`x+y+z`会被解析为一个单独的变量名，而`x + y + z`则会被解析为三个变量的表达式。
- 保留字：变量名不应与LP文件格式的关键词（如`st`, `bounds`, `min`, `max`, `binary`, `end`）相同。

## 5.3 ILP 文件

ILP文件格式与LP文件格式是相同的。唯一的区别在于它们的使用方式。ILP文件格式专门用于存储已计算出的不可约不一致子系统（IIS）模型。

#### 什么是IIS？

IIS（Irreducible Inconsistent Subsystem），不可约不一致子系统，即最小冲突集合，表示在一个不可行的优化问题中，去除任何一个约束都会使得剩下的子系统变为可行。ILP文件用于存储这些最小冲突集合，用于进一步的分析和调试模型。

#### 使用Gurobi求解ILP文件

以下是使用Gurobi读取和分析ILP文件的Python代码示例：

```python
import gurobipy as gp
from gurobipy import GRB

# 读取ILP文件
model = gp.read("example.ilp")

# 求解模型
model.optimize()

# 输出结果
if model.status == GRB.INFEASIBLE:
    print('Model is infeasible.')
    # 计算IIS并保存
    model.computeIIS()
    model.write("example.ilp")
    print('IIS written to example.ilp')
else:
    print('Model is feasible or optimal.')
```

注意要先计算出 IIS 模型，然后才能将IIS模型写入ILP文件。

## 5.4 SOL 文件

SOL文件格式用于存储优化模型的解。当求解器找到一个可行解或最优解后，可以使用诸如`GRBwrite`等命令将解写入SOL文件。SOL文件包含模型中每个变量的名称和值，每个变量及其对应的值占一行。

### 基本结构

SOL文件的结构非常简单，每行包含一个变量名和它的值。以下是一个示例SOL文件的内容：

```plaintext
# Solution file
x  1.0
y  0.5
z  0.2
```

在这个示例中，`x`、`y`和`z`是变量名，它们对应的解分别是`1.0`、`0.5`和`0.2`。

### 文件生成

SOL文件可以通过调用Gurobi的`write()`函数生成。假设我们已经有一个已优化的模型，可以通过以下代码将解写入SOL文件：

```python
import gurobipy as gp
from gurobipy import GRB

# 创建并优化模型
model = gp.Model()
x = model.addVar(name="x")
y = model.addVar(name="y")
z = model.addVar(name="z")

model.setObjective(x + y + z, GRB.MAXIMIZE)
model.addConstr(x + y == 1, "c0")
model.addConstr(x + 5*y + 2*z <= 10, "c1")

model.optimize()

# 如果模型找到可行解或最优解，写入SOL文件
if model.status == GRB.OPTIMAL or model.status == GRB.SUBOPTIMAL:
    model.write("solution.sol")
```

### 读取SOL文件

读取SOL文件非常简单，以下为两种常见方式：

1. Gurobi求解器读取：`gp.read('solution.sol')`。
2. Python的标准文件操作读取：

```python
solution = {}
with open("solution.sol", "r") as file:
    for line in file:
        if not line.startswith("#"):
            parts = line.split()
            variable = parts[0]
            value = float(parts[1])
            solution[variable] = value

print(solution)
```

## 参考资料
[1] [Gurobi 官网文档](https://docs.gurobi.com/projects/optimizer/en/current/reference/misc/fileformats.html#mps-format) <br>