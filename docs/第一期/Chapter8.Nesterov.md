# 第8章 Nesterov 加速算法

&emsp;&emsp;Nesterov加速方法的基本迭代形式为：
$$
\begin{aligned}
v_{t} &=\mu_{t-1} v_{t-1}-\epsilon_{t-1} \nabla g\left(\theta_{t-1}+\mu_{t-1} v_{t-1}\right) \\
\theta_{t} &=\theta_{t-1}+v_{t}
\end{aligned}
$$
&emsp;&emsp;和动量方法的区别在于二者用到了不同点的梯度，动量方法采用的是上一步 $\theta_{t-1}$ 的梯度方向，而Nesterov加速方法则是从 $\theta_{t-1}$ 朝着 $v_{t-1}$ 往前一步。 一种解释是，反正要朝着 $v_{t-1}$ 方向走，不如先利用了这个信息。 接下来推导出第二种等价形式：
$$
\begin{aligned}
\theta_{t} &=\theta_{t-1}+v_{t} \\
&=\theta_{t-1}+\mu_{t-1} v_{t-1}-\epsilon_{t-1} \nabla g\left(\theta_{t-1}+\mu_{t-1} v_{t-1}\right) \\
&=\theta_{t-1}+\mu_{t-1}\left(\theta_{t-1}-\theta_{t-2}\right)-\epsilon_{t-1} \nabla g\left(\theta_{t-1}+\mu_{t-1}\left(\theta_{t-1}-\theta_{t-2}\right)\right)
\end{aligned}
$$
&emsp;&emsp;然后引入中间变量 $y_{t-1}$ ，使得它满足
$$
y_{t-1}=\theta_{t-1}+\mu_{t-1}\left(\theta_{t-1}-\theta_{t-2}\right)
$$
&emsp;&emsp;然后得到第二种等价形式
$$
\begin{aligned}
&\theta_{t}=y_{t-1}-\epsilon_{t-1} \nabla g\left(y_{t-1}\right) \\
&y_{t}=\theta_{t}+\mu_{t}\left(\theta_{t}-\theta_{t-1}\right)
\end{aligned}
$$
&emsp;&emsp;这可以理解为先进行梯度步，然后再进行加速步，这两种形式是完全等价的。