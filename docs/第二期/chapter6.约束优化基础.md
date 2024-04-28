本章内容主要研究的是约束最小化问题：
$$
	\begin{split}
		&\min\limits_{x\in\mathbb{R}} f(x)\\\
		&s.t. \begin{cases}
			c_i(x) = 0, i\in \mathcal{E},\\
			c_i(x) \geq 0, i\in \mathcal{I},
		\end{cases}
	\end{split}
$$


约定$c_i$为定义在$\mathbb{R}$或其子集上的实值函数， $\mathcal{E}$和$\mathcal{I}$分别是等式约束和不等式约束的指标集。于是该问题的可行域为
$$
	\mathcal{X} = \{x\in \mathbb{R}\mid c_i(x)=0, i\in \mathcal{E}\text{且}c_i(x)\geq 0, i\in \mathcal{I}\}.
$$

一个较为明显的做法是将$\mathcal{X}$的示性函数加到目标函数后组成无约束优化问题，但这样造成函数的性质并不是很好。为解决该问题，我们首先要给出约束优化问题的最优性理论，为此我们引入如下定义。

## 约束最优化问题的最优性条件
### 切锥(tangent cone)

给定可行域$\mathcal{X}$及其内部一点$x$，若存在可行序列$\{z_k\}_{k=1}^\infty\subset \mathcal{X}$满足$\lim_{k\to \infty}z_k = x$以及正标量序列$\{t_k\}_{k=1}^\infty$，$t_k\to 0$满足
$$\lim\limits_{k\to \infty}\frac{z_k - x}{t_k} = d,$$
则向量$d$为$\mathcal{X}$在$x$处的一个切向量所有$x$处切向量的集合称为**切锥**，用$T_\mathcal{X}(x)$表示.

> 实际上，这里的切锥即为微分几何中的切空间(tangent space)。

与无约束优化类似，我们要求切锥(可行方向集合)不包含使得目标函数值下降的方向，这就是局部最优点需要满足的必要条件，称为$\textbf{几何最优性条件}$。
### 几何最优性条件
​	假设可行点$x^*\in\mathcal{X}$是问题最小化问题的一个局部极小点，如果$f(x)$和$c_i(x)$，$i\in \mathcal{I}\cup \mathcal{E}$在点$x^*$处是可微的，那么有

$$d^T\nabla f(x^*)\geq 0,\quad d\in T_\mathcal{X}(x^*)$$
等价于

$$T_\mathcal{X}(x^*)\cap \{d\mid \nabla f(x^*)^Td<0\}=\emptyset$$

>上述定理实际上说的是在最优点$x^*$处，线性化可行方向和下降方向的交集为空集，这是显然的。

### 积极集(active set)
​	对于可行点$x\in \mathcal{X}$，在该点处的$\textbf{积极集}$定义为两部分指标的集合，一部分是等式约束对应的指标，另一部分是不等式约束中在该点等式成立的约束所对应的指标，即
$$\mathcal{A}(x) = \mathcal{E}\cup \{i\in\mathcal{I}:c_i = 0\}$$



### 线性无关约束品性LICQ
​	给定可行点$x\in\mathcal{X}$及相应的积极集$\mathcal{A}(x)$.如果积极集对应的约束函数的梯度，即$\nabla c_i(x), i\in\mathcal{A}(x)$，是线性无关的，则称$\textbf{线性无关约束品性(LICQ)}$在点$x$处成立。

### 线性化可行方向锥
​	在点$x$处的\textbf{线性化可行方向锥}定义为
$$
		\mathcal{F}(x) = \begin{cases}
			d\mid \begin{split}
				d^T\nabla c_i(x)=0,\forall i\in \mathcal{E},\\
				d^T\nabla c_i(x)\geq 0,\forall i\in \mathcal{I}\cap \mathcal{A}(x)
			\end{split}
		\end{cases}
$$

> 引理:给定任意可行点$x\in\mathcal{X}$，如果在该点处LICQ成立，则有$T_\mathcal{X} = \mathcal{F}(x).$


直接验证几何最优性条件中交集是空集仍是较为麻烦的，为此我们介绍一个更加方便的方式

### Farkas引理

设$p$和$q$是两个非负整数，给定向量组$\{a_i\in\mathbb{R}^n, i = 1,2,\cdots, p\}$，$\{b_i\in\mathbb{R}^n, i = 1,2,\cdots, q\}$和$c\in\mathbb{R}^n$.满足下列条件：
$$d^Ta_i = 0,\quad i = 1, 2, \cdots, p,$$
$$d^Tb_i \geq 0,\quad i = 1, 2, \cdots, q,$$
$$d^Tc<0$$
的$d$不存在当且仅当存在一组$\lambda_i, i = 1,2,\cdots, p$和$\mu_i\geq 0,i = 1,2, \cdots q$，使得
$$c = \sum\limits_{i=1}^p \lambda_ia_i + \sum\limits_{i=1}^q\mu_i b_i.$$


利用Farkas引理，我们可以把几何最优性条件(交集为空集)写为下面的等价形式：
$$
	-\nabla f(x^*) = \sum\limits_{i\in \mathcal{E}}\lambda_i^* \nabla c_i(x^*)+\sum\limits_{i\in \mathcal{A}(x^*)\cap\mathcal{I}}\lambda_i^*\nabla c_i(x^*),
$$
其中 $\lambda_i^*\in\mathbb{R}, i\in \mathcal{E},\lambda_i^*\geq 0, i\in \mathcal{A}(x^*)\cap\mathcal{I}$.如补充定义$\lambda_i^*=0,i\in \mathcal{I}\backslash\mathcal{A}(x^*)$，那么有
$$
	-\nabla f(x^*) = \sum\limits_{i\in\mathcal{I}\cup\mathcal{E}}\lambda_i^*\nabla c_i(x^*),
$$
这恰好为Lagrange函数关于$x$的一阶最优性条件，另外，对于任意的$i\in \mathcal{I}$，我们注意到
$$
	\lambda_i^*c_i(x^*) = 0.
$$
上式称为**互补松弛条件**，这个条件说明对于不等式约束，以下两种情况至少出现一种：

1. 乘子$\lambda_i\geq 0$
2. 不等式约束失效，即$c_i(x^*>0)$严格成立

一般来讲，如果上述两种情况恰好只有一种满足时，我们称**严格互补松弛条件**成立，一般来说具有严格互补松弛条件的最优值点有较好的性质。

综上，我们有如下的一阶必要条件，也称作KKT条件，并称满足该条件的点$x^*, \lambda^*$为**KKT对**.
### Karush-Kuhn-Tucker条件
设$x^*$是约束优化问题的一个局部最优点，如果
$$T_{\mathcal{X}}(x^*) = \mathcal{F}(x^*)$$
成立，即在该点LICQ成立，则存在Lagrange乘子$\lambda_i^*$使得下列条件成立：

1. 稳定性条件 $\nabla_x L(x^*, \lambda^*) = \nabla f(x^*)-\sum\limits_{i\in \mathcal{I}\cup \mathcal{E}}\lambda_i^*\nabla c_i(x^*) = 0$,
2. 原始可行性条件 $c_i(x^*) = 0,\forall i\in\mathcal{E},$
3. 原始可行性条件 $c_i(x^*) \geq 0,\forall i\in\mathcal{I},$
4. 对偶可行性条件 $\lambda_i^*\geq 0,\forall i\in \mathcal{I}$,
5. 互补松弛条件 $\lambda_i^*c_i(x^*) = 0,\forall i\in \mathcal{I}$


## 练习
求下面约束优化问题的KKT点及相应的乘子
$$
		\begin{split}
			&\min f(x)=x_1^2+x_2\\
			s.t. & -x_1^2-x_2^2+9\geq 0\\
			& -x_1-x_2+1\geq 0
		\end{split}
$$
Solution: 
相应的Lagrange函数为
$$
		\mathcal{L}(x,\lambda) = x_1^2+x_2 - \lambda_1 (-x_1^2-x_2^2+9) - \lambda_2 (-x_1-x_2+1)
$$
KKT条件为
$$
		\begin{split}
			&\frac{\partial\mathcal{L}}{\partial x_1} = 2x_1+2\lambda_1 x_1 +\lambda_2 = 0\\
			&\frac{\partial\mathcal{L}}{\partial x_2} = 1+2\lambda_1 x_2 +\lambda_2 = 0\\
			&\lambda_1 (-x_1^2-x_2^2+9) = 0, \lambda_1\geq 0, -x_1^2-x_2^2+9\geq 0\\
			&\lambda_2 (-x_1-x_2+1)  = 0, \lambda_2\geq 0, -x_1-x_2+1\geq 0
		\end{split}
$$
首先$\lambda_1 = \lambda_2 = 0$无解\\
$\lambda_1 = 0,\lambda_2 \neq 0$得$\lambda_2 = -1$，矛盾\\
$\lambda_1\neq 0, \lambda_2 = 0$得$x = (0, -3)^T,\lambda = (\frac{1}{6},0)^T$\\
$\lambda_1\neq 0, \lambda_2 \neq 0$得$x = (\frac{1\pm \sqrt{17}}{2},\frac{1\mp \sqrt{17}}{2})^T$，有$\lambda_2 = -\frac{1}{2}$矛盾。

综上，只有一个KKT点$(0, -3)^T$及相应的Lagrange乘子$(\frac{1}{6}, 0)^T$。
