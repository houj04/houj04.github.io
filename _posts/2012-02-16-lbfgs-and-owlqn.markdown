---
layout: post
title: "L-BFGS and OWL-QN"
categories: mypost
---

## LBFGS和OWLQN

在实际遇到的“大”问题中，Hessian矩阵可能会非常大，导致无法计算甚至是无法存储。内存受限的拟牛顿法于是应运而生，它不保存完整的$n \times n$的整个矩阵，而是保存少数一些长度是$n$的向量，用它们来对Hessian矩阵进行近似。

一个著名的内存受限的方法是L-BFGS，L是limited memory，即“内存受限的”。从名字上就可以看出，它是基于BFGS算法的。其主要思想是使用最近几轮的迭代数据来近似Hessian矩阵，同时较早一些的数据就被扔掉了，这样可以大幅度减少内存存储。

先来回顾一下BFGS方法。

在BFGS方法里，每一步要寻找下一个迭代点，这样做：

$$x_{k+1} = x_k + \alpha_kH_k\nabla f_k$$

其中$\alpha_k$是步长，$H_k$则是每一轮都通过下面的公式进行更新：

$$ H_{k+1} = V_k^TH_kV_k + \rho_ks_ks_k^T $$

其中

$$\rho_k = \dfrac{1}{y_k^Ts_k}$$

$$ V_k = I - \rho_ky_ks_k^T $$

$$s_k = x_{k+1} - x_k$$

$$y_k = \nabla f_{k+1} - \nabla f_k$$

在计算下一个迭代点的时候，实际上需要的是$H_k\nabla f_k$。于是有了这样的一种做法，保留最新的$m$个$s_i$和$y_i$，一共$2m$个长度是$n$的向量，然后用一大堆向量内积和加法来近似计算$H_k\nabla f_k$。每到新的一轮，当计算出来新的$s_i$和$y_i$之后，把最老的扔掉，仍然只保留最新的$m$个。这样对内存的占用量就从原来的$n^2$量级大幅度降低到了$mn$量级。从实际近似的效果上来看，$m$的取值在3到20之间有比较好的效果。

下面来说说具体的做法。在第$k$轮，记当前的迭代点是$x_k$，保留的$k$组数据是$\\{s_i, y_i\\}$，其中$i = k-m, \dots, k-1$。

首先选择一个Hessian矩阵的初值$H_k^0$，使用BFGS相同的更新公式一轮一轮地算，把迭代和递归都展开，最后得到的结果是

$$
\begin{aligned}
H_k &= (V_{k-1}^T \cdots V_{k-m}^T) H_k^0 (V_{k-m} \cdots V_{k-1})\\
    &+ \rho_{k-m}(V_{k-1}^T \cdots V_{k-m+1}^T)s_{k-m}s_{k-m}^T(V_{k-m+1} \cdots V_{k-1}) \\
    &+ \rho_{k-m+1}(V_{k-1}^T \cdots V_{k-m+2}^T)s_{k-m+1}s_{k-m+1}^T(V_{k-m+2} \cdots V_{k-1}) \\
    &+ \cdots \\
    &+ \rho_{k-1}s_{k-1}s_{k-1}^T
\end{aligned} $$

从上面的这个表达式，得到了称为two-loop的算法，用来计算$H_k\nabla f_k$，如下：

* 算法7.4（L-BFGS two-loop recursion）
* $q \leftarrow \nabla f_k$
* for $i = k-1, k-2, \dots, k-m$
* $ \quad \alpha_i \leftarrow \rho_is_i^Tq$
* $ \quad q \leftarrow q - \alpha_iy_i$
* end for
* $ r \leftarrow H_k^0q $
* for $i = k-m, k-m+1, \dots, k-1$
* $ \quad \beta \leftarrow \rho_iy_i^Tr$
* $ \quad r \leftarrow r + s_i(\alpha_i - \beta)$
* end for
* 结束。此时$H_k\nabla f_k = r$。

于是，就得到了完整的L-BFGS算法：

* 算法7.5（L-BFGS）
* 设置初始迭代点$x_0$，正整数$m$
* $k \leftarrow 0$
* 重复
* $\quad$选择一个$H_0^k$
* $\quad$使用算法7.4计算搜索方向：$p_k \leftarrow -H_k\nabla f_k$
* $\quad$用Wolfe条件求步长$\alpha_k$，并计算下一个迭代点$x_{k+1}\leftarrow x_k+\alpha_kp_k$
* $\quad$如果$k>m$
* $\quad \quad$删去向量$s_{k-m}$和$y_{k-m}$
* $\quad$计算并保存$s_k\leftarrow x_{k+1}-x_k$和$y_k \leftarrow \nabla f_{k+1} - \nabla f_k$
* $\quad k \leftarrow k+1$
* 直到收敛

这样问题就解决了：对于一个优化问题，可以按照这样一个流程来解：找一个初始迭代点，每一轮计算一个下降方向，使用线性搜索找到一个合适的步长，移动到那个点，计算新的梯度和目标函数，然后把旧的$s=diff(x)$和$y=diff(\nabla f)$扔掉，一直这样下去，直到满足收敛准则。

还有什么问题？有时候模型会出现过拟合问题，因此需要对模型的参数进行限制，常见的方法之一是修改目标函数，把模型的参数也计算到目标函数里面，越复杂的模型对目标函数的“贡献”就越多，那么如果我们希望得到简单的模型，让目标函数下降也就在一定程度上控制了模型的复杂程度。

于是要求解的目标函数就成了这个样子：

$$f(x) = l(x) + r(x)$$

其中，$l(x)$是原来的目标函数，而$r(x)$称为“正规化项”，用来控制模型的复杂程度。正规化项的一个常见的例子是“1-范数”，即：

$$r(x) = C\lVert x \rVert_1 = C \sum_i |x_i|$$

简单来说，就是把训练出来的模型的参数，每个参数取绝对值，然后都加在一起，再乘以一个固定的非负的系数。

紧接着问题来了，绝对值函数不是处处可导的，在某个$x_i = 0$的时候，梯度计算就成了问题，随着$x_i$从某个负值变成正值的时候，它的导数会从$-1$变成$+1$。

但是刚才说的L1正则化也有一些好的性质：（1）它在坐标轴上才不可导，在非坐标轴的任意地方（也就是象限的内部）都是可导的；（2）L1正则化的本质是求绝对值，一阶导数是常数，二阶导数直接就是0，所以不影响Hessian矩阵及其近似，也就是说之前我们提到的BFGS和L-BFGS那些用一大堆向量来近似Hessian矩阵的方法完全不受影响。

所以解决思路就出来了：我们限定在某一个象限内部进行搜索（一维线性搜索），躲开坐标轴，不就行了么。

## Orthant-Wise Limited-memory Quasi-Newton algorithm

先引入两个操作很简单的函数记号：

（1）符号函数：大于0就返回1，小于0就返回-1，否则0。

$$
\begin{equation}
\sigma(x)=
\begin{cases}
   1 &\mbox{if $x>0$}\\
   0 &\mbox{if $x=0$}\\
  -1 &\mbox{if $x<0$}
\end{cases}
\end{equation}
$$

（2）“变号就清零”函数：以$y$为参照，如果$x$在某个维度上的符号和$y$相同，那么该维度上的返回值就是$x$的值，否则就清零。

$$
\begin{equation}
\pi_i(x;y)=
\begin{cases}
   x_i &\mbox{if $\sigma(x_i) = \sigma(y_i)$}\\
   0   &\mbox{otherwise}
\end{cases}
\end{equation}
$$

刚才说了躲开坐标轴，也就是在“限定象限内进行搜索”。但是有的时候当前迭代点就恰好在坐标轴上（比方说迭代初值在原点），所以要特殊考虑那些坐标为$0$的点。

OWL-QN算法中重要的两步是：

（1）选定一个象限。我们用符号$\xi_i$来记录某个象限的第$i$维的符号，所有维度记作$\xi\in\\{-1, 0, 1\\}^n$。那么符合要求的象限（算上平面边界）中的所有的点，就可以记为：

$$\Omega_\xi = \{x\in\mathbb{R}^n:\pi(x;\xi)=x\}$$

简单说，就是“$x$和$\xi$，每一维的符号都一样”。

使用“虚梯度”来代替原始的梯度：

$$
\begin{equation}
\diamond_if(x)=
\begin{cases}
   \partial_i^-f(x) &\mbox{if $\partial_i^-f(x) > 0$}\\
   \partial_i^+f(x) &\mbox{if $\partial_i^+f(x) < 0$}\\
   0   &\mbox{otherwise}
\end{cases}
\end{equation}
$$

其中，左偏导数和右偏导数定义为

$$
\begin{equation}
\partial_i^\pm f(x)=\dfrac{\partial}{\partial x_i}l(x)+
\begin{cases}
   C\sigma(x_i) &\mbox{if $x_i \ne 0$}\\
   \pm C        &\mbox{if $x_i = 0 $}
\end{cases}
\end{equation}
$$

那么我们的象限定义就是

$$
\begin{equation}
\xi_i^k=
\begin{cases}
   \sigma(x_i^k)             &\mbox{if $x_i^k \ne 0$}\\
   \sigma(-\diamond_if(x^k)) &\mbox{if $x_i^k = 0 $}
\end{cases}
\end{equation}
$$

（2）受限的一维搜索：
刚才说了，一维搜索要在限定的象限里面进行，那么产生新的迭代点的公式也就从原来的

$$x_{k+1} = x_k + \alpha_kp_k$$

变成了

$$x^{k+1} = \pi(x^k + \alpha p^k; \xi^k)$$

也就是：当Line Search使得$x^{k+1}$的第$i$个维度$x_i^{k+1}$越过$\xi^k$指定的象限时（符号发生变化），就把$x_i^{k+1}$清零。

完整的OWL-QN算法，如下面的流程所示。

* OWL-QN算法
* 选取初始点$x^0$，$S=\\{\\}$，$Y=\\{\\}$
* for $k=0$ to MaxIters do
* $\quad$计算$v^k=-\diamond f(x^k)$
* $\quad$用L-BFGS的two loop方法，用$S$和$Y$计算$d^k=H^kv^k$
* $\quad$搜索方向$p^k=\pi(d^k;v^k)$
* $\quad$用受限的一维搜索确定$x^{k+1}$
* $\quad$如果满足收敛条件则退出
* $\quad$更新$S$和$Y$：$s^k=x^{k+1}-x^k$，$y^k=\nabla l(x^{k+1})-\nabla l(x^k)$
* end for

