# chapter2 线搜索
&emsp;&emsp;最优化问题可以分为无约束最优化问题与约束最优化问题两大类。无约束最优化问题是求一个函数的极值问题，即
$$
\min f(x) \tag{1}
$$
其中$x \in \mathbb{R}^n$为决策变量，$f(x)\in \mathbb{R}$为目标函数。问题$(1)$的解称为最优解，记为$x^{*}$,该点的函数值$f(x^{*})$称为最优值。问题$(1)$的最优解分为全局最优解和局部最优解，本节介绍的线搜索算法是求局部最优解的算法。

<!-- &emsp;&emsp;无约束优化问题是众多优化问题中最基本的问题，它对自变量$x$的取值范围不加限制，所以无需考虑$x$的可行性。对于光滑函数，可以较容易地利用梯度和海瑟矩阵的信息来设计算法；对于非光滑函数，可以利用次梯度来构造迭代格式．无约束优化问题的优化算法主要分为两大类：线搜索类型的优化算法和信赖域类型的优化算法．


它们都是对函数$f(x)$ 在局部进行近似，但处理近似问题的方式不同．线搜索类算法根据搜索方向的不同可以分为梯度类算法、次梯度算法、牛顿算法、拟牛顿算法等．一旦确定了搜索的方向，下一步即沿着该方向寻找下一个迭代点． -->

## 2.1线搜索算法结构
&emsp;&emsp;求解最优化问题的基本方法是迭代算法，即采用逐步逼近的计算方法来逼近问题的精确解的方法。以最小化问题为例，在一个算法中，可先选定一个初始迭代点$x_{0}\in \mathbb{R}^n$，在该迭代点处，确定一个函数值下降的方向，再确定在这个方向上的步长，从而求得下一个迭代点，依次类推，产生一个迭代点列{$x_{k}$}，{$x_{k}$}或其子列应收敛于问题的最优解。当给定的某种终止准则满足时，或者表明$x_{k}$已满足要求的近似最优解的精度，或者表明算法已无力进一步改善迭代点，迭代结束。
线搜索算法的基本结构如下：
$$
\begin{array}{l}
（1） 给定初始点x_{0} \in \mathbb{R^{n}}, k:=0 \\
（2）若在x_{k}点终止准则满足，则输出有关信息，停止迭代  \\
（3）确定f(x)在x_{k}点的下降方向d_{k} \\
（4）计算步长\alpha_{k}，使f(x_{k}+\alpha_{k} d_{k})小于f(x_{k}) \\
（5）令x_{k+1}:=x_{k}+\alpha_{k} d_{k} ，k:=k+1,转（2） \\
\end{array}
$$
其中包含两个基本要素：一是下降的方向；二是步长。不同的方法可以得到不同的下降方向和步长，由此可构成不同的算法，具有以上结构的最优化方法称为线搜索方法。
## 2.2终止准则
&emsp;&emsp;因为局部极小点$x^{*}$是稳定点（假设目标函数$f(x)$的一阶导数存在），可以用$\lVert \nabla{f(x_{k})} \rVert \leqslant \epsilon$作为终止准则，$\epsilon$的大小决定所得迭代点$x_{k}$近似$x^{*}$的精度。但该准则也有一定的局限性，对于在极小点领域内比较陡峭的函数，即使该领域中的点已相当接近极小点，但其梯度值可能仍然较大，从而使迭代难以停止。

&emsp;&emsp;其他终止准则有$\lVert x_{k}-x_{k+1}\rVert \leqslant \epsilon$或$f_{k}-f_{k+1}\leqslant\epsilon$,但这些准则满足只能说明算法这时所进行的迭代对迭代点或迭代点处目标函数值的改善已经非常小，并不能保证$\lVert x_{k}-x_{k+1}\rVert$或$f_{k}-f_{k+1}$一定足够小。

## 2.3搜索方向
<!-- &emsp;&emsp;对任意$d\in \mathbb{R}^n$，且$d\neq0$，若存在$\overline{\alpha}_{k}$，使
$$
f\left(x_{k}+\overline{\alpha}_{k} d\right)<f\left(x_{k}\right)
$$
则称$d$为下降方向。 -->

&emsp;&emsp;在迭代点$x_{k}$处，若存在$\overline{\alpha}_{k}$，使
$$
f\left(x_{k}+\alpha_{k} d\right)<f\left(x_{k}\right), \forall\alpha \in (0, \overline{\alpha}_{k}),
$$
则$d$为$f(x)$在$x_{k}$点的下降方向。

&emsp;&emsp;根据$f\left(x_{k}+\alpha_{k} d\right)$在$x_{k}$点的Taylor展开式，
$$
f\left(x_{k}+\alpha d\right)=f\left(x_{k}\right)+\alpha \nabla f\left(x_{k}\right)^{T}d +O(\lVert \alpha d \rVert^{2})
$$
知，下降方向$d$为满足$\nabla f\left(x_{k}\right)^{T}d < 0$的方向。

&emsp;&emsp;线搜索类算法根据搜索方向的不同可以分为梯度类算法、次梯度算法、牛顿算法、拟牛顿算法等．

<!-- 
&emsp;&emsp;在迭代点$x_{k}$处，若存在$\overline{\alpha}_{k}$，使
$$
f\left(x_{k}+\alpha_{k} d\right)<f\left(x_{k}\right), \forall\alpha \in (0, \overline{\alpha}_{k}),
$$
定义$d^{T}\nabla f(x)$为$f(x)$在点$x$处沿着方向$d$的梯度，记作
$$
d^{T}\nabla f(x)=\nabla f(x)^{T} d
$$


&emsp;&emsp;在迭代点$x_{k}$处，当迭代方向$d_{k}$已知时，求使$f(x_{k}+\alpha d_{k})$沿$d_{k}$方向关于步长$\alpha$取极小值，即
$$
\alpha_{k}=\arg \min \limits_{\alpha} f\left(x_{k}+\alpha d_{k}\right)
$$
，称为精确线搜索准则。 -->
## 2.4搜索步长
&emsp;&emsp;在线搜索方法中，搜索方向的选择有很多，但是步长选择的方法在不同算法中非常相似。设$\phi(\alpha)=f(x_{k}+\alpha d_{k})$，从当前迭代点$x_{k}$出发，沿搜索方向$d_{k}$，需要确定合适的步长$\alpha_{k}$，使$f(x_{k}+\alpha_{k} d_{k})<f(x_{k})$，即$\phi(\alpha_{k})<\phi(0)$。
<!-- ，是关于$\alpha$的一维搜索问题。 -->
<!-- 
主要包含两方面的内容：（1）满足什么样的准则，步长可以接受？（2）有了合适的准则，满足该准则的步长该如何求？
线搜索的目标是选取合适的  使得 φ(ak)尽可能减小.但这一工作并不容易:wk应该使得f充分下降，与此同时不应在寻找上花费过多的计算量.我们需要权衡这两个方面.一个自然的想法是寻找 《 使得 -->
<!-- ### 2.3.1精确线搜索准则 -->
&emsp;&emsp;搜索步长的选择通常需要在目标函数下降量和确定$\alpha_{k}$的计算量之间进行平衡。一个很自然的想法是取$\alpha_{k}$使目标函数$f(x_{k})$沿方向$d_{k}$达到极小，即使得$
\phi( \alpha_{k} )=\operatorname* {m i n}\limits_{\alpha> 0} \phi( \alpha)$，这种方法称为精确线搜索。由于在实际计算中，采用精确线搜索通常需要很大的计算量，且对一般问题而言，实现精确线搜索十分困难，因此在实际应用中较少使用。另一个想法是不去求$
\phi( \alpha)$的最小值点，而是选取$\alpha_{k}$使目标函数得到可接受的下降量$f(x_{k})-f(x_{k}+\alpha_{k} d_{k})$，这种线搜索方法被称为非精确线搜索。非精确线搜索因需要的计算量相对较少更受人们青睐。
<!-- 
因而花费计算较少的非精确线搜索更受人们青睐。理论上精确的最优步长一般很难求得，当变量维度非常高或者$f(x)$非常复杂时，采用这种方法选取$\alpha_{k}$通常需要很大计算量。实际上，当迭代点离最优解尚远时，是没有必要做高精度线搜索的。另外，对一般问题而言，实现精确线搜索是很困难的。如果选取$\alpha_{k}$使目标函数$f$得到可接受的下降量，即使得下降量$f(x_{k})-f(x_{k}+\alpha_{k} d_{k})>0$是用户可接受的，则称为非精确线搜索。 -->
### 2.4.1非精确线搜索准则
&emsp;&emsp;在非精确线搜索算法中，在选取$\alpha_{k}$时仅使$f(x_{k}+\alpha_{k} d_{k})<f(x_{k})$，是不足以确保生成的迭代序列$\{x_{k}\}$收敛到最优解的，选取$\alpha_{k}$需要满足一定的要求，这些要求被称为线搜索准则。线搜索准则的合适与否直接决定了算法的收敛性，若选取不合适的线搜索准则将会导致算法无法收敛至极小值点。例如考虑一维无约束优化问题
$$
\operatorname* {m i n}_{x} f(x)=x^{2}
$$
初始迭代点$x_{0}=1$。由于问题是一维的，下降方向只有{-1，+1}两种。我们选取$d_{k}=-sign(x_{k})$，且只要求选取的步长满足迭代点处函数值单调下降，即
$
f(x_{k}+\alpha d_{k})<f(x_{k})
$
考虑选取如下两种步长：

$$
\alpha_{k, 1}=\frac{1} {3^{k+1}}, \quad\alpha_{k, 2}=1+\frac{2} {3^{k+1}}, 
$$
通过计算可以得到

$$
x_{k}^{1}=\frac{1} {2} \left( 1+\frac{1} {3^{k}} \right), \quad x_{k}^{2}=\frac{(-1 )^{k}} {2} \left( 1+\frac{1} {3^{k}} \right). 
$$

显然，序列$\{f(x_{k}^{1})\}$和序列$\{f(x_{k}^{2})\}$均单调下降，但序列$\{x_{k}^{1}\}$收敛的点不是极小值点，序列$\{x_{k}^{2}\}$则在原点左右振荡，不存在极限，如下图所示。
出现上述情况的原因是在迭代过程中函数值$f(x)$的下降量不够充分以至于算法无法收敛到极小值点.为了避免这种情况发生，必须引入一些更合理的线搜索准则来确保迭代的收敛性.
![image](./images/ch2/fx收敛图.png "image")
<!-- 在选取$\alpha_{k}$时仅使$f(x_{k}+\alpha_{k} d_{k})<f(x_{k})$，是不足以确保生成的迭代序列$/{x_{k}/}$收敛到最优解的。如下图所示，其中目标函数最小值为$f^{*}=-1$,存在一个迭代序列$\{x_{k}\}$,使得$f(x_{k})=5/k,k=0,1,\dots$,虽然每次迭代目标函数都有下降但是$f(x_{k})$收敛于0而不是函数最小值。每一次迭代中目标函数的下降量不足够，使得迭代序列无法收敛到最优解，为避免这种情况，需要使步长的选择满足一定的准则，常见的准则包括Goldstein准则和Wolfe准则。 -->
#### 1. Armijo准则
&emsp;&emsp;Armijo 准则：

$$
f ( x_{k}+\alpha d_{k} ) \leqslant f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k},  \rho\in (0,1)
$$
是一个常用的线搜索准则，引人 Armiio准则的目的是保证每一步迭代充分下降。
Armijo 准则有非常直观的几何含义，它指的是点$(\alpha,\phi(\alpha))$必须在直线
$$l(\alpha)=f ( x_{k} )+\rho\alpha \nabla f(x_{k})^{\mathsf{T}}d_{k}$$
的下方。如下图所示(图中$g=\nabla f (x)$)，区间（0,$\beta_{4}$]和[$\beta_{5}$,$\beta_{6}$]中的点均满足 Armijo 准则。$d$为下降方向，满足$\nabla f ( x_{k} )^{\mathsf{T}} d_{k}<0$， $l(\alpha)$的斜率为负，选取符合 Armiio准则的$\alpha$确实会使得函数值下降。在实际应用中，参数$c_{1}$通常选为一个很小的正数，例如$c_{1}=10^{-3}$，这使得 Armijo 准则非常容易得到满足。但是仅仅使用 Armijo 准则并不能保证迭代的收敛性，因为可行区域中包含了步长$\alpha$接近0的区域，当$\alpha$取值太小时，目标函数值的下降量可能过小，导致序列$\{f(x_{k})\}$的极限值不是极小值，必须避免$\alpha$取值过小，因此Armiio准则需要配合其他准则共同使用。
![Armijo准则](./images/ch2/Armijo准则_1.png "Armijo准则")
#### 2.Goldstein准则
&emsp;&emsp;为了克服 Armijo 准则的缺陷，需要引入其他准则来保证每一步的 $\alpha$不会太小。既然 Armijo 准则只要求点($\alpha,\phi(\alpha)$)必须处在某直线下方，我们也可使用相同的形式使得该点必须处在另一条直线的上方。这就是Armijo-Goldstein 准则，简称 Goldstein 准则：

$$
\begin{aligned} {{f ( x_{k}+\alpha d_{k} )}} & {{} {{} \leqslant f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k},}} \\ {{f ( x_{k}+\alpha d_{k} )}} & {{} {{} \geqslant f ( x_{k} )+( 1-\rho ) \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k},}} \\ \end{aligned} 
$$
其中$\rho\in (0,1/2)$。
同样，Goldstein 准则也有非常直观的几何含义，它指的是点($\alpha,\phi(\alpha)$)必须在两条直线

$$
\begin{aligned} {{l_{1} ( \alpha)}} & {{} {{} {{}=f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k},}}} \\ {{l_{2} ( \alpha)}} & {{} {{} {{}=f ( x_{k} )+( 1-\rho ) \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k}}}} \\ \end{aligned} 
$$
之间。如下图所示，区间[$\beta_{3}$,$\beta_{4}$]和[$\beta_{5}$,$\beta_{6}$]中的点均满足Goldsteimn准则，同时可注意到Goldstein准则确实去掉了过小的$\alpha$。
![Goldstein准则](./images/ch2/Goldstein准则_1.png "Goldstein准则")

#### 3.Wolfe准则
&emsp;&emsp;Goldstein准则能够使得函数值充分下降，但是它可能会避开$\phi(\alpha)$取最小值的区域。为此引人Armijo-Wolfe准则，简称Wolfe准则：
$$
f ( x_{k}+\alpha d_{k} ) \leqslant f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k}, 
$$
$$
\nabla f ( x_{k}+\alpha d_{k} )^{\mathrm{T}} d_{k} \geqslant \sigma \nabla f ( x_{k} )^{\mathrm{T}} d_{k}, 
$$
其中$1>\sigma>\rho>0$为给定常数。在Wolfe准则中，第一个不等式即是 Armijo 准则，而第二个不等式则是 Wolfe 准则的本质要求。 注意到 $\nabla f ( x_{k}+\alpha d_{k} )^{\mathrm{T}} d_{k}$恰好就是$\phi(\alpha)$的导数，Wolfe准则实际要求$\phi(\alpha)$在点$\alpha$处切线的斜率不能小于$\phi(\alpha)$在零点斜率的$\sigma$倍。如下图所示，在区间[$\beta_{7}$,$\beta_{4}$]、[$\beta_{8}$,$\beta_{9}$]和[$\beta_{10}$,$\beta_{6}$]中的点均满足Wolfe 准则。注意到在$\phi(\alpha)$的极小値点$\alpha^{*}$处有$\phi'(\alpha^{*})=\nabla f ( x_{k}+\alpha^{*} d_{k} )^{\mathrm{T}} d_{k}=0$，因此$\alpha^{*}$永远满足第二个不等式。而选择较小的$\rho$可使得$\alpha^{*}$同时满足第一个不等式，即 Wolfe 准则在绝大多数情况下会包含线搜索子问题的精确解。在实际应用中，参数$\sigma$通常取为 0.9。

![Wolfe准则](./images/ch2/Wolfe准则_1.png "Wolfe准则")

&emsp;&emsp;在Wolfe准则中，即使$\sigma$取为0，亦无法保证满足准则的点接近精确线搜索的结果。但若采用下面的强Wolfe准则，$\sigma$取得越小，满足准则的$\alpha$越接近精确线搜索的结果，
$$
f(x_{k}+\alpha d_{k})\leqslant f(x_{k})+\rho \alpha\nabla f ( x_{k} )^{\mathrm{T}}d_{k}\\|\nabla f(x_{k}+\alpha d_{k})^{\mathrm{T}}d_{k}|\leqslant-\sigma \nabla f( x_{k} )^{\mathrm{T}}d_{k}
$$
其中$1>\sigma>\rho>0$。
### 2.4.2收敛性
&emsp;&emsp;
这一小节给出非精确线搜索算法的收敛性。

 $\textbf{Zoutendijk定理:}$考虑一般的迭代格式$x_{k+1}=x_{k}+\alpha_{K}d_{k}$，其中$d_{k}$是搜索方向，$\alpha_{k}$是步长，且在迭代过程中Wolfe准则满足。假设目标函数$f$下有界、连续可微且梯度$L$-利普希茨连续，即
$$
\| \nabla f ( x )-\nabla f ( y ) \| \leqslant L \| x-y \|, \quad\forall\, x, y \in\mathbb{R}^{n}, 
$$
那么

$$
\sum_{k=0}^{\infty} \operatorname{cos}^{2} \theta_{k} \| \nabla f ( x^{k} ) \|^{2} <+\infty, \tag{2}
$$
其中$\operatorname{cos}_{\theta_{k}}$为负梯度$-\nabla f(x_{k})$和下降方向$d_{k}$夹角的余弦，即
$$
\operatorname{cos} \theta_{k}=\frac{-\nabla f ( x_{k} )^{\mathrm{T}} d_{k}} {\| \nabla f ( x_{k} ) \| \| d_{k} \|}. 
$$
不等式(2)也被称为$\textbf{Zoutendijk条件}$。

&emsp;&emsp;$\textbf{证明:}$

&emsp;&emsp;由wolfe条件
$$
\nabla f ( x_{k}+\alpha d_{k} )^{\mathrm{T}} d_{k} \geqslant \sigma \nabla f ( x_{k} )^{\mathrm{T}} d_{k}, 
$$
可得
$$
\Big( \nabla f ( x_{k+1} )-\nabla f ( x_{k} ) \Big)^{\mathsf{T}} d_{k} \geqslant( \sigma-1 ) \nabla f ( x_{k} )^{\mathsf{T}} d^{k}. 
$$
由柯西不等式和梯度L-利普希茨连续性质

$$
\left( \nabla f ( x_{k+1} )-\nabla f ( x_{k} ) \right)^{\mathrm{T}} \! d^{k} \leqslant\| \nabla f ( x_{k+1} )-\nabla f ( x_{k} ) \| \| d_{k} \| \leqslant\alpha_{k} L \| d_{k} \|^{2}. 
$$
结合上述两式可得

$$
\alpha_{k} \geqslant{\frac{\sigma-1} {L}} {\frac{\nabla f ( x_{k} )^{\mathrm{T}} d_{k}} {\| d_{k} \|^{2}}}. 
$$
注意到$\nabla f ( x_{k} )^{\mathrm{T}} d_{k}<0$，将上式代入wolfe准则中的第一个不等式$f ( x_{k}+\alpha d_{k} ) \leqslant f ( x_{k} )+\rho \alpha\nabla f ( x_{k} )^{\mathsf{T}} d_{k}$条件，则

$$
f ( x_{k+1} ) \leqslant f ( x_{k} )+\rho {\frac{\sigma-1} {L}} {\frac{\left( \nabla f ( x_{k} )^{\mathsf{T}} d_{k} \right)^{2}} {\| d_{k} \|^{2}}}. 
$$
根据$\theta_{k}$的定义，此不等式可等价表述为

$$
f ( x_{k+1} ) \leqslant f ( x_{k} )+\rho \frac{\sigma-1} {L} \operatorname{c o s}^{2} \theta_{k} \| \nabla f ( x_{k} ) \|^{2}. 
$$
再关于k求和，有
$$
f ( x_{k+1} ) \leqslant f ( x_{0} )-\rho \frac{1-\sigma} {L} \sum_{j=0}^{k} \operatorname{c o s}^{2} \theta_{j} \| \nabla f ( x_{j} ) \|^{2}. 
$$
又因为函数$f$是下有界的，且由$0<\rho<\sigma<1$可知$\rho(1-\sigma)>0$，因此当$k\rightarrow +\infty$时

$$
\sum_{j=0}^{\infty} \operatorname{cos}^{2} \theta_{j} \| \nabla f ( x_{j} ) \|^{2} <+\infty. 
$$

&emsp;&emsp;Zoutendik定理指出，只要迭代点满足 Wolfe 准则，对梯度利普希茨连续且下有界函数总能推出(2)式成立。实际上采用Goldstein准则也可推出类似的条件。Zoutendik定理刻画了线搜索准则的性质，配合下降方向$d_{k}$的选取方式可以得到最基本的收敛性。


$\textbf{线搜索算法的收敛性:}$

&emsp;&emsp;对于线搜索算法，设$\theta_{k}$为每一步负梯度 $-\nabla f(x_{k})$与下降方向$d_{k}$的夹角，并假设对任意的k，存在常数$\gamma>0$，使得
$$
\theta_{k} < \frac{\pi} {2}-\gamma, 
$$
则在Zoutendik定理成立的条件下，有
$$
\operatorname* {l i m}_{k \to\infty} \nabla f ( x^{k} )=0. 
$$
&emsp;&emsp;$\textbf{证明:}$

&emsp;&emsp;假设结论不成立，即存在子列$\{k_{l}\}$和正常数$\delta>0$，使得

$$
\| \nabla f ( x_{k_{l}} ) \| \geqslant\delta, \quad l=1, 2, \cdots. 
$$
根据$\theta_{k}$的假设，对任意的k,

$$
\operatorname{c o s} \theta_{k} > \operatorname{s i n} \gamma> 0. 
$$
我们仅考虑式(2)的第$k_{l}$项，有

$$
\begin{aligned} {{\sum_{k=0}^{\infty} \operatorname{c o s}^{2} \theta_{k} \| \nabla f ( x_{k} ) \|^{2}}} & {{} \geqslant\sum_{l=1}^{\infty} \operatorname{cos}^{2} \theta_{k_{l}} \| \nabla f ( x_{k_{l}} ) \|^{2}} \\ {} & {{} \geqslant\sum_{l=1}^{\infty} ( \operatorname{sin}^{2} \gamma) \cdot\delta^{2} \to+\infty,} \\ \end{aligned} 
$$
这显然和Zoutendik定理矛盾。因此必有

$$
\operatorname* {l i m}_{k \to\infty} \nabla f ( x^{k} )=0. 
$$
该推论建立在 Zoutendik 条件之上，它的本质要求是每一步的下降方向$d_{k}$和负梯度方向不能趋于正交。这个条件的几何直观明显:当下降方向$d_{k}$和梯度正交时，根据泰勒展开的一阶近似，目标函数值$f(x)$几乎不发生改变，因此要求$d_{k}$与梯度正交方向夹角有一致的下界。

后面会介绍多种d的选取方法，在选取时条件(6.1.8)总得到满足总的来说，推论6.1仅仅给出了最基本的收敛性，而没有更进一步回答算法的收敛速度.这是由于算法收敛速度极大地取决于d的选取.接下来我们将着重介绍如何选取下降方向 d.
## 2.4搜索算法
在实际应用中较少使用.
需要选择的合适步长，使$f(x_{k}+\alpha_{k} d_{k})<f(x_{k})$，同时保证序列${f(x_{k})}$收敛于极小点。
构造辅助函数$\phi(\alpha)=f(x_{k}+\alpha d_{k})$，其中$d_{k}$为给定的下降方向，，$\alpha>0$为该辅助函数的自变量。函数$\phi(\alpha)$的步长，则
$$
\phi(\alpha)=\phi(0)+\alpha \nabla f(x_{k})^{T} d_{k}
$$

&emsp;&emsp;对$\phi(\alpha)$求导，并令其等于0，可得
$$
\nabla f(x_{k})^{T} d_{k}=\phi^{\prime}(\alpha)=0
$$
则
$$
\phi(\alpha)=\phi\left(0\right)+\alpha \nabla f\left(x_{k}\right)^{T} d_{k} +O(\lVert \alpha d_{k} \rVert^{2})
$$


&emsp;&emsp;当$\nabla f\left(x_{k}\right)^{T} d_{k} \neq 0$时，$\phi(\alpha)$在$\alpha=0$处取得极小值，即
线搜索类算法根据搜索方向的不同可以分为梯度类算法、次梯度算法、牛顿算法、拟牛顿算法等．这是因为选取dk 的方法千差万别，但选取αk 的方法在不同算法中非常相似．
### 2.2.1精确线搜索准则
&emsp;&emsp;在迭代点$x_{k}$处，当迭代方向$d_{k}$已知时，使$f(x_)$沿$d_{k}$方向关于步长$\alpha$取极小值，即
$$
\alpha_{k}=\arg \min \limits_{\alpha} f\left(x_{k}+\alpha d_{k}\right)
$$
，称为精确线搜索准则。当变量维度非常高或者$f(x)$非常复杂时，采用这种方法选取
$\alpha_{k}$通常需要很大计算量，在实际应用中较少使用.$d_{k}^{T}\nabla{f(x_{k}+\alpha d_{k})}=0$

### 2.2.2步长准则
&emsp;&emsp;在迭代点$x_{k}$处，当迭代方向$d_{k}$已知时，使$f(x_{k}+\alpha d_{k})$沿$d_{k}$方向关于步长$\alpha$取极小值，即
$$
\alpha_{k}=\arg \min \limits_{\alpha} f\left(x_{k}+\alpha d_{k}\right)
$$
&emsp;&emsp;在实际应用中，由于计算梯度$f(x_{k})$的复杂性，通常采用次梯度$\nabla f(x_{k})$来代替梯度，即

&emsp;&emsp;线搜索算法的基本结构可以分为以下几个步骤：

1. 确定下降方向：根据目标函数的性质，确定下降方向，例如梯度下降法中，下降方向为负梯度方向；

2. 确定步长：根据搜索方向，确定步长，即迭代步长；

3. 更新迭代点：根据搜索方向和步长，更新迭代点；
&emsp;&emsp;其中，终止准则可以分为以下几种：

1. 迭代次数达到预先设定的最大迭代次数；
\text { 计算步长： } \alpha_{k}=\text { 线搜索算法 } \\
\text { 更新迭代点： } x_{k}=x_{k-1}+\alpha_{k} \cdot \boldsymbol{d}_{k} \\
\text { 判断迭代点是否满足终止准则： } \text { 终止准则 } \\
\text { 迭代： } k=k+1, \text { 转 } 2 \\
\end{array}

## 2.3搜索算法
### 2.3.1 回退算法
在优化算法的实现中，寻找一个满足 Armijo 准则的步长是比较容易的,一个最常用的算法是回退法，给定初值&，回退法通过不断以指数方式缩小试探步长，找到第一个满足 Armijo 准则(6.1.2)的点.
>1. 选择初始步长$
\overline{\alpha}$，参数$\gamma,c\in(0,1)$。初始化$\alpha\leftarrow\overline{\alpha}$。
>
>2. while$f(x_{k} + \alpha d_{k}) > f(x_{k}) + c\alpha \nabla f(x_{k})^{\mathbb{T}}d_{k}$ do
>
>3. &emsp;&emsp;令$\alpha \leftarrow \gamma \alpha$
>
>4. end while
>5. 输出$\alpha_{k}=\alpha$

该算法被称为回退法是因为$\alpha$的试验值是由大至小的，它可以确保输出的 “ 能尽量地大.此外算法6.1不会无限进行下去，因为d是一个下降方向，当$\alpha$充分小时，Armijo 准则总是成立的，在实际应用中我们通常也会给α设置一个下界，防止步长过小.
本小节介绍在实际中使用的线搜索算法.之前的讨论已经初步介绍了回退法(算法6.1)，并指出该算法可以用于寻找 Armijo 准则(6.1.2)的步长实际上只要修改一下算法的终止条件，回退法就可以被用在其他线搜索准则之上，例如之前我们提到的两种非单调线搜索准则(6.1.5)和(6.1.6).回退法的实现简单、原理直观，所以它是最常用的线搜索算法之一.然而，回退法的缺点也很明显:第一，它无法保证找到满足 Wolfe 准则的步长，即条件(6.1.4b)不一定成立，但对一些优化算法而言，找到满足 Wolfe 准则的步长是十分必要的;第二，回退法以指数的方式缩小步长，因此对初值¢和参数 γ的选取比较敏感，当丫过大时每一步试探步长改变量很小，此时回退法效率比较低，当丫过小时回退法过于激进，导致最终找到的步长太小，错过了选取大步长的机会:下面简单介绍其他类型的线搜索算法.
### 2.3.2 黄金分割法

```
算法6.1 线搜索回退法
1. 选择初始步长$\a$，参数α∈(0,1)。初始化α←α。
2. while f(x* + αdk) > f(x) + α⋅c⋅f(x)'dk do
3.     令α←α/2
4. end while
5. 输出 αk = α
```


## 2.4收敛性分析
&emsp;&emsp;线搜索算法框架可以分为以下几个步骤：

1. 确定搜索方向：根据目标函数的性质，确定搜索方向，例如梯度下降法中，搜索方向为负梯度方向；

2. 确定步长：根据搜索方向，确定步长，即迭代步长；

3. 更新迭代点：根据搜索方向和步长，更新迭代点；

## 3.1 梯度

&emsp;&emsp;我们常常用函数的一阶信息即梯度去求函数的最优值。上述问题的梯度可以记做
$$
\underbrace{\nabla_{\boldsymbol{w}^{(i)}} L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right) \triangleq \frac{\partial L\left(\boldsymbol{w}^{(1)}, \cdots, \boldsymbol{w}^{(l)}\right)}{\partial \boldsymbol{w}^{(i)}}}_{\text {两种符号都表示 } L \text { 关于 } \boldsymbol{w}^{(l)} \text { 的梯度 }}, \quad \forall i=1, \cdots, l .
$$
注意梯度 $\displaystyle\nabla _{w^{(i)}}L$ 的形状应该和 $w^{(i)}$ 的形状完全一致。

&emsp;&emsp;如果用 `TensorFlow`和 `PyTorch` 等深度学习平台，可以不需要关心梯度是如何求出来的。只要定义的函数对某个变量可微，`TensorFlow`和 `PyTorch`就可以自动求该函数关于该变量的梯度。但是，我们应该注意在写程序前，检查梯度的形状与变量的形状是否相同。

## 3.2 梯度下降

&emsp;&emsp;我们通常规定梯度是函数的最快的上升方向。因此，想要极小化一个函数，很自然地会想到沿着梯度方向的反方向去搜索。沿着梯度反方向称为做梯度下降（GD）。
$$
x_{k+1}=x_{k}+\alpha_{k}*\left(-\nabla f\left(x_{k}\right)\right).
$$
&emsp;&emsp;我们也可以用瞎子爬山的例子来理解梯度下降算法的含义，瞎子爬山可以看做求一个函数的极大值，瞎子在每一步都可以获得当前的坡度（即梯度），但不知道其他各点的任何情况。梯度下降法相当于在爬山中沿着山坡最陡的方向往前爬（或是下山）。

&emsp;&emsp;那么，对于最上面提出的优化问题，可以写出瞎子的梯度下降的算法过程：
$$
w_{\text {new }}^{(i)} \leftarrow w_{\text {now }}^{(i)}-\alpha \cdot \nabla_{w^{(i)}} L\left(w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}\right), \quad \forall i=1, \cdots, l
$$
其中， $\displaystyle w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}$ 为当前需要优化的变量。

&emsp;&emsp;通常称上面式子中的 $\alpha$ 为步长或者是学习率，其设置影响梯度下降算法的收敛速率，最终会影响神经网络的测试准确率，所以 $\alpha$ 需要仔细调整。数学中常常通过线搜索的方式寻找 $\alpha$，可以参考 Jorge Nocedel的《Numerical Optimization》文献[1]，这里不再赘述。

&emsp;&emsp;当优化函数是凸的L-利普希茨连续函数时，梯度下降法可以保证收敛性，且收敛速率为 $\displaystyle O(\frac{1}{k})$，$k$为迭代步数。

>  **注：**利普希茨连续的定义是如果函数 $f$ 在区间 $Q$ 上以常数L-利普希茨连续，那么对于 $x, y \in Q$ 有
> $$
> \|f(x)-f(y)\| \leq L\|x-y\|
> $$

&emsp;&emsp;给出一个简单的 python程序再来复习一下梯度下降法。
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
&emsp;&emsp;其中，函数 $F_j$ 隐含第 $j$ 个训练样本 $(x_j , y_j)$。每次随机从集合 ${1, 2, \cdots , n}$ 中抽取一个整数，记作 $j$。设当前的优化变量为 $w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}$ 计算此处的随机梯度，并对他做随机梯度下降：
$$
\mid w_{\text {new }}^{(i)} \leftarrow w_{\text {now }}^{(i)}-\alpha \cdot \underbrace{\nabla_{w^{(i)}} F_{j}\left(w_{\text {now }}^{(1)}, \cdots, w_{\text {now }}^{(l)}\right)}_{\text {随机梯度 }}, \quad \forall i=1, \cdots, l .
$$
&emsp;&emsp;事实上，在实际操作中我们会发现使用GD算法求一些非凸的优化问题时，程序往往会停在鞍点上，无法收敛到局部最优点，这会导致测试的准确率非常低；而使用 SGD方法可以帮助我们跳出鞍点，继续向更好的最优点前进。

&emsp;&emsp;令人欣喜的是， SGD也可以保证收敛，具体证明过程比较复杂，感兴趣的话可以阅读文献[4]。这里仅给出 SGD收敛的一个**充分条件**：
$$
\sum_{k=1}^{\infty}\alpha_k=\infty,\sum_{k=1}^{\infty}\alpha_k^2<\infty
$$

&emsp;&emsp;最后给出一个简单的 python程序复习一下随机梯度下降法。
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
【3】Jorge Nocedal§. Optimization Methods for Large-Scale Machine Learning[J]. Siam Review, 2016, 60(2).  
【4】Nemirovski A S , Juditsky A , Lan G , et al. Robust Stochastic Approximation Approach to Stochastic Programming[J]. SIAM Journal on Optimization, 2009, 19(4):1574-1609.

