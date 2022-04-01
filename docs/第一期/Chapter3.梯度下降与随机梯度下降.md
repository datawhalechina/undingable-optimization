# 第三章 梯度下降与随机梯度下降
&emsp;&emsp;线性模型和神经网络的训练通常都可以描述为一个优化问题。即设 $\omega^{(1)},\omega^{(2)},\cdots\omega^{(l)}$ 为优化变量（它们可以是向量、矩阵、张量）。我们通常会遇到求解这样一个优化问题：
$$
\min_{w^{(1)},\cdots ,w^{(l)}}\quad L(w^{(1)},\cdots ,w^{(l)})
$$
&emsp;&emsp;对于这样一个比较简单的无约束优化问题，我们常使用梯度下降算法（Gradient Descent, 缩写`GD`）和随机梯度下降算法（Stochastic Gradient Descent，缩写`SGD`）来寻找最优解。

## 3.1 梯度

&emsp;&emsp;我们常常用函数的一阶信息即梯度去求函数的最优值。上述问题的梯度可以记做
$$
\underbrace{\nabla_{\boldsymbol{w}^{(i)}} L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right) \triangleq \frac{\partial L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right)}{\partial \boldsymbol{w}^{(i)}}}_{\text {两种符号都表示 } L \text { 关于 } \boldsymbol{w}^{(l)} \text { 的梯度 }}, \quad \forall i=1, \cdots, l .
$$
注意梯度 $\displaystyle\nabla _{w^{(i)}}L$ 的形状应该和 $w^{(i)}$ 的形状完全一致。

&emsp;&emsp;如果用`TensorFlow`和`PyTorch` 等深度学习平台，可以不需要关心梯度是如何求出来的。只要定义的函数对某个变量可微，`TensorFlow`和`PyTorch`就可以自动求该函数关于该变量的梯度。但是，我们应该注意在写程序前，检查梯度的形状与变量的形状是否相同。

## 3.2 梯度下降

&emsp;&emsp;定义梯度是一个上升的方向。因此，想要极小化一个函数，很自然地会想到沿着梯度方向的反方向去搜索。沿着梯度反方向称为做梯度下降（GD）。
$$
x_{k+1}=x_{k}+\alpha_{k}*\left(-\nabla f\left(x_{k}\right)\right).
$$
&emsp;&emsp;我们也可以用瞎子爬山的例子来很好的理解梯度下降算法的含义，瞎子爬山可以看做求一个函数的极大值，瞎子在每一步都可以获得当前的坡度（即梯度），但不知道其他各点的任何情况。梯度下降法相当于在爬山中沿着山坡最陡的方向往前爬（或是下山）。

&emsp;&emsp;那么，对于最上面提出的优化问题，可以写出瞎子的梯度下降的算法过程：
$$
w_{\text {new }}^{(i)} \leftarrow w_{\text {now }}^{(i)}-\alpha \cdot \nabla_{w^{(i)}} L\left(w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}\right), \quad \forall i=1, \cdots, l
$$
其中， $\displaystyle w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}$ 为当前需要优化的变量。

&emsp;&emsp;通常称上面式子中的 $\alpha$ 为步长或者是学习率，其设置影响梯度下降算法的收敛速率，最终会影响神经网络的测试准确率，所以 $\alpha$ 需要仔细调整。数学中常常通过线搜索的方式寻找 $\alpha$，可以参考Jorge Nocedel的《Numerical Optimization》文献[1]，这里不再赘述。

&emsp;&emsp;当优化函数是凸的L-利普希茨连续函数时，梯度下降法可以保证收敛性，且收敛速率为 $\displaystyle O(\frac{1}{k})$，$k$为迭代步数。

>  **注：**利普希茨连续的定义是如果函数 $f$ 在区间 $Q$ 上以常数L-利普希茨连续，那么对于 $x, y \in Q$ 有
> $$
> \|f(x)-f(y)\| \leq L\|x-y\|
> $$

&emsp;&emsp;给出一个简单的python程序再来复习一下梯度下降法。
```python
"""
一维问题的梯度下降法示例
"""

def func_1d(x):
    """
    目标函数
    :param x: 自变量，标量
    :return: 因变量，标量
    """
    return x ** 2 + 1


def grad_1d(x):
    """
    目标函数的梯度
    :param x: 自变量，标量
    :return: 因变量，标量
    """
    return x * 2


def gradient_descent_1d(grad, cur_x=0.1, learning_rate=0.01, precision=0.0001, max_iters=10000):
    """
    一维问题的梯度下降法
    :param grad: 目标函数的梯度
    :param cur_x: 当前 x 值，通过参数可以提供初始值
    :param learning_rate: 学习率，也相当于设置的步长
    :param precision: 设置收敛精度
    :param max_iters: 最大迭代次数
    :return: 局部最小值 x*
    """
    for i in range(max_iters):
        grad_cur = grad(cur_x)
        if abs(grad_cur) < precision:
            break  # 当梯度趋近为 0 时，视为收敛
        cur_x = cur_x - grad_cur * learning_rate
        print("第", i, "次迭代：x 值为 ", cur_x)

    print("局部最小值 x =", cur_x)
    return cur_x


if __name__ == '__main__':
    gradient_descent_1d(grad_1d, cur_x=10, learning_rate=0.2, precision=0.000001, max_iters=10000)
```

## 3.3 随机梯度下降

&emsp;&emsp;在需要优化大规模的问题时，计算梯度已经成为了一件非常麻烦的事情。是否能够用梯度样本中的一些例子来近似所有的梯度样本呢？答案是可以的！

&emsp;&emsp;如果目标函数可以写成连加或者期望的形式，那么就可以用随机梯度下降求解最小化问题。

&emsp;&emsp;假设目标函数可以写成 $n$ 项连加形式：
$$
L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right)=\frac{1}{n} \sum_{j=1}^{n} F_{j}\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right)
$$
&emsp;&emsp;其中，函数 $F_j$ 隐含第 $j$ 个训练样本 $(x_j , y_j)$。每次随机从集合 ${1, 2, \cdots , n}$ 中抽取一个整数，记作 $j$。设当前的优化变量为 $w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}$ 计算此处的随机梯度，并且做随机梯度下降：
$$
\mid w_{\text {new }}^{(i)} \leftarrow w_{\text {now }}^{(i)}-\alpha \cdot \underbrace{\nabla_{w^{(i)}} F_{j}\left(w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}\right)}_{\text {随机梯度 }}, \quad \forall i=1, \cdots, l .
$$
&emsp;&emsp;事实上，在实际操作中我们会发现求一些非凸优化问题时使用GD算法，往往会停在鞍点上，无法收敛到局部最优点，这会导致测试的准确率非常低；而SGD可以跳出鞍点，继续向更好的最优点前进。

&emsp;&emsp;令人欣喜的是，SGD也可以保证收敛，具体证明过程比较复杂，感兴趣的话可以阅读文献[4]。这里仅给出SGD收敛的一个**充分条件**：
$$
\sum_{k=1}^{\infty}\alpha_k=\infty,\sum_{k=1}^{\infty}\alpha_k^2<\infty
$$

&emsp;&emsp;最后给出一个简单的python程序复习一下随机梯度下降法。
```python
import numpy as np
import math

# 生成测试数据
x = 2 * np.random.rand(100, 1)  # 随机生成100*1的二维数组，值分别在0~2之间

y = 4 + 3 * x + np.random.randn(100, 1)  # 随机生成100*1的二维数组，值分别在4~11之间

x_b = np.c_[np.ones((100, 1)), x]
print("x矩阵内容如下：\n{}".format(x_b[0:3]))
n_epochs = 100
t0, t1 = 1, 10

m = n_epochs
def learning_schedule(t):  # 模拟实现动态修改步长
    return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        x_i = x_b[random_index:random_index+1]
        y_i = y[random_index:random_index+1]
        gradients = 2 * x_i.T.dot(x_i.dot(theta)-y_i)  # 调用公式
        learning_rate = learning_schedule(epoch * m + i)
        theta = theta - learning_rate * gradients

    if epoch % 30 == 0:
        print("抽样查看：\n{}".format(theta))

print("最终结果：\n{}".format(theta))

# 计算误差
error = math.sqrt(math.pow((theta[0][0] - 4), 2) + math.pow((theta[1][0] - 3), 2))
print("误差：\n{}".format(error))
```

## 参考文献

【1】王树森, 黎彧君, 张志华, 深度强化学习,https://github.com/wangshusen/DRL/blob/master/Notes_CN/DRL.pdf, 2021  
【2】Nocedal, Jorge & Wright, Stephen. (2006). Numerical Optimization. 10.1007/978-0-387-40065-5.   
【3】 Jorge Nocedal§. Optimization Methods for Large-Scale Machine Learning[J]. Siam Review, 2016, 60(2).  
【4】Nemirovski A S , Juditsky A , Lan G , et al. Robust Stochastic Approximation Approach to Stochastic Programming[J]. SIAM Journal on Optimization, 2009, 19(4):1574-1609.