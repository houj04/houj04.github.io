---
layout: post
title: "Newton method and BFGS algorithm"
categories: mypost
---

## 基本概念与数学知识复习

* 正定矩阵：设实对称矩阵$M$，对于所有的非零向量$z=(z_1, z_2, \dots, z_n)^T$，都有$z^TMz>0$，则称矩阵$M$为正定的（positive definite）。

   性质：正定矩阵均可逆，且其逆矩阵也是正定的。

* 泰勒展开：设某函数$f(x)$在某常数$a$的邻域内无限可微，则

$$ f(x) = f(a) + \dfrac{f'(a)}{1!}(x-a) + \dfrac{f''(a)}{2!}(x-a)^2 + \cdots + \dfrac{f^{(n)}(a)}{n!}(x-a)^n $$

* 梯度：设某标量函数$f(x)$，其中$x=(x_1, x_2, \dots, x_n)^T$是一个列向量。定义其梯度为（也是列向量）

$$ \nabla f = \left( \dfrac{\partial f}{\partial x_1}, \dfrac{\partial f}{\partial x_2}, \cdots, \dfrac{\partial f}{\partial x_n} \right) ^T$$

* Hessian矩阵（设实函数$f(x_1, x_2, \dots, x_n)$所有的二阶导数都存在）：

$$
Hessian(f) = \left(
\begin{matrix}
 \dfrac{\partial^2 f}{\partial x_1^2}      & \dfrac{\partial^2 f}{\partial x_1 \partial x_2}      & \cdots & \dfrac{\partial^2 f}{\partial x_1 \partial x_n}      \\
 \dfrac{\partial^2 f}{\partial x_2 \partial x_1}      & \dfrac{\partial^2 f}{\partial x_2^2}      & \cdots & \dfrac{\partial^2 f}{\partial x_2 \partial x_n}      \\
 \vdots & \vdots & \ddots & \vdots \\
 \dfrac{\partial^2 f}{\partial x_n \partial x_1}      & \dfrac{\partial^2 f}{\partial x_n \partial x_2}      & \cdots & \dfrac{\partial^2 f}{\partial x_n^2}      \\
\end{matrix}
\right)
$$

* 泰勒展开和Hessian矩阵的关系，即多元泰勒展开：

$$ y = f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \dfrac{1}{2}\Delta x^T Hessian(x) \Delta x$$

* 实对阵矩阵的谱分解：

（1）正交矩阵：如果某矩阵$A$，满足$A^TA = AA^T = I$，则称$A$是正交矩阵。

（2）正交向量：如果两个向量的点积为$0$，则称它们正交。

（3）$n$阶实对称矩阵$A$必可以正交对角化。记$\lambda_1, \lambda_2, \dots, \lambda_n$是$A$的特征值，则一定存在$n$阶正交阵$Q$和对角阵$\lambda = diag(\lambda_1, \lambda_2, \dots, \lambda_n)$，使得$A=Q\lambda Q^T$成立。或者可以等价地说，$n$阶实对称矩阵$A$必有$n$个两两正交的单位特征向量$q_1, q_2, \dots, q_n$，此时有

$$ A = \lambda_1q_1(q_1)^T + \lambda_2q_2(q_2)^T + \cdots + \lambda_nq_n(q_n)^T $$

* 矩阵求逆的Sherman-Morrison-Woodbury公式：如果某矩阵$A^* = A + UV^T$，那么

$$ (A^*)^{-1} = A^{-1} - A^{-1}U(I+V^TA^{-1}U)^{-1}V^TA^{-1} $$

* 搜索方向：已知某函数$y = f(x)$，其中$x = (x_1, x_2, \dots, x_n)^T$，步长$a$，搜索方向$d$，那么，“沿搜索方向计算”实际上是计算$f(x+ad)$。

* 中值定理：如果函数$f(x)$在$(a, b)$上可导，在$[a, b]$上连续，则必存在$x^* \in [a, b]$，使得$f'(x^*)(b-a) = f(b) - f(a)$。

## 牛顿法

目的：求出某个函数$f(x)$的极值。记极值点为$x^{\*}$，那么$f'(x^*) = 0$。

方法：先找出一个接近极值点的点，记为$x_0$，然后在附近寻找$x_1$，使得$f'(x_1)$更加接近于0。


带来的问题：如果记$x_1 = x_0 + \Delta x$，实际上就是求“偏移量”$\Delta x$。怎么求？

用泰勒展开式：

$$ f(x_1) \approx f(x_0) + f'(x_0)(x_1-x_0) + \dfrac{1}{2}f''(x_0)(x_1-x_0)^2 $$


也就是

$$ f(x_1) \approx f(x_0) + f'(x_0)\Delta x + \dfrac{1}{2}f''(x_0)(\Delta x)^2$$

两边对$\Delta x$求导，然后把约等号换成等号，则有

$$ f'(x_0) + f''(x_0)\Delta x = 0 $$

所以解出

$$ \Delta x = - \dfrac{f'(x_0)}{f''(x_0)} $$

相应的迭代公式就出来了

$$ x_{n+1} = x_n - \dfrac{f'(x_n)}{f''(x_n)} $$

## 牛顿法的高维情况

前面得到一维情况的迭代公式

$$ x_{n+1} = x_n - \dfrac{f'(x_n)}{f''(x_n)} = x_n - \left(f''(x_n)\right)^{-1} \times \left(f'(x_n)\right)$$

高维的时候，用梯度代替导数，用Hessian矩阵代替二阶导数，那么

$$ X_{n+1} = X_n - \left(Hessian(x)\right)^{-1} \times \left(\nabla f(X_n)\right)$$

有什么问题么：Hessian矩阵的逆不好求，并且需要二阶导数。


## 拟牛顿方法

上面的牛顿法的高维情况，“搜索方向”是什么：

$$ \Delta x = -(Hessian(x_n))^{-1}\nabla f(x_n) $$

既然Hessian矩阵的逆不好求，尝试用另外一个矩阵来进行“近似”。

$$ \Delta x = - B_k^{-1} \nabla f(x_k)$$

牛顿法：

$$ f(x_k + \Delta x) \approx f(x_k) + \nabla f(x_k)^T\Delta x+ \dfrac{1}{2}(\Delta x)^TB\Delta x $$

对$\Delta x$求导：

$$ \nabla f(x_k + \Delta x) \approx \nabla f(x_k)+ B\Delta x $$

我们希望这个近似足够好，也就是换成等于号：

$$ \nabla f(x_k + \Delta x) = \nabla f(x_k)+ B\Delta x $$

或者写成

$$ B\Delta x = \nabla f(x_k + \Delta x) - \nabla f(x_k) $$

$$ B \times diff(x) = diff(\nabla f) $$

所谓的拟牛顿方法，就是下面的这样一个流程：

## The Broyden-Fletcher-Goldfarb-Shanno (BFGS) Algorithm

1、初始化，$B_0$设为单位矩阵

2、更新目标向量$x$：

（1）先计算搜索方向：

$$ \Delta x_k = -\alpha_kB_k^{-1}\nabla f(x_k) $$

（2）更新$x$：

$$ x_{k+1} = x_k + \Delta x_k$$

3、计算新的$x$点的梯度$\nabla f(x_{k+1})$，并记$y_k = \nabla f(x_{k+1})-\nabla f(x_k)$，$s_k = x_{k+1} - x_k = \alpha_kp_k$

于是就有

$$ B_{k+1}s_k = y_k$$

4、更新B矩阵：

$$B_{k+1} = B_k + \dfrac{y_ky_k^T}{y_k^Ts_k} - \dfrac{(B_ks_k)(B_ks_k)^T}{s_k^TB_ks_k} $$

这样看起来很好。不过在计算搜索方向的时候还是要求矩阵的逆。所以干脆，直接再做一步，我们直接存$B^{-1}$。

那么

$$ B_{k+1}^{-1} = B_k^{-1} + \dfrac{(s_k^Ty_k+y_k^TB_k^{-1}y_k)(s_ks_k^T)}{(s_k^Ty_k)^2} - \dfrac{B_k^{-1}y_ks_k^T+s_ky_k^TB_k^{-1}}{s_k^Ty_k} $$

于是问题又来了：（1）这么长的一个公式，是怎么推出来的？后边还有一个更长的求$B$矩阵的逆；（2）怎么冒出一个$\alpha$来？

先说第一个。我们要求$B$矩阵是对称且正定的，那么$B_{k+1}-B_k=A$肯定也是对称的。那么由之前的关于“对称矩阵的分解”：

$$ A = \lambda_1q_1(q_1)^T + \lambda_2q_2(q_2)^T + \cdots + \lambda_nq_n(q_n)^T $$

对$A$取近似，只保留特征值大的两项。然后换一下字母以方便计算：

$$A = auu^T + bvv^T$$

那么现在的状况是：

$$
\begin{cases}{}
B_{k+1} = B_k + auu^T + bvv^T \\
B_{k+1}s_k = y_k
\end{cases}
$$

也就是

$$ B_ks_k + auu^Ts_k + bvv^Ts_k = y_k $$

现在我们要找出符合条件的$a$，$u$，$b$，$v$。显然下面的这样一组解是符合要求的：

$$
\begin{cases}{}
v = y_k \\
b \times v^Ts_k = 1 \\
u = -B_ks_k \\
au^TS_k = 1
\end{cases}
$$

然后把这组解代入到之前的递推式$A = auu^T + bvv^T$里面，就得到了$A$。

矩阵的逆可以用Sherman-Morrison-Woodbury公式来计算。计算过程比较复杂，就不列出来了。

## 搜索方向的计算

### 线性搜索法

线性搜索法的每一轮，计算一个搜索方向$p_k$，以及“沿着这个方向走多远”（也叫“步长”），记为$\alpha_k$。于是新的迭代点就是

$$x_{k+1} = x_k + \alpha_kp_k$$

搜索方向就是之前说的“沿……搜索”，一般长这个样子：

$$ p_k = -B_k^{-1} \nabla f(x_k) $$

在牛顿法里面，这里的矩阵$B$就是Hessian矩阵，而在拟牛顿法里，$B$矩阵是用来近似Hessian矩阵的，每一轮都通过一个公式进行更新。

如果$B_k$是正定的，那么

$$ p_k^T\nabla f(x_k) = -\nabla f(x_k)^TB_k^{-1}\nabla f(x_k) < 0 $$

也就是说确实是“下降方向”。

终于开始说步长$\alpha$了。

我们选择步长主要考虑两个因素：（1）确实能使目标函数$f$能够有足够的下降；（2）寻找步长这件事情本身，不能太复杂。

当然，最最理想的情况是这样，我们找到的步长，就是下面这个单变量函数的最小值：

$$ \phi(\alpha) = f(x_k + \alpha p_k) $$

但是这很困难，甚至找到一个“局部最小值”都很困难。

### 非精确的线性搜索

一种常见的线性搜索法是，$\alpha_k$首先一定能让目标函数“足够下降”。即满足下面的不等式，其中$0<c_1<1$：

$$ \phi(\alpha) = f(x_k + \alpha p_k) \le l(\alpha) = f(x_k) + c_1\alpha\nabla f_k^Tp_k $$

$l(\alpha)$是递减的。等式左右两边都是递减的。但是因为$0<c_1<1$，所以在$\alpha$是一个小正数的时候，等式左边“减得比较快”。或者说，直线的方向比切线的方向要更“平”一点。所以$\phi(\alpha)\le l(\alpha)$一定成立。

所以还要有一个限制，即弯曲条件。在$0<c_1<c_2<1$的情况下，要有

$$ \phi'(\alpha_k) = \nabla f(x_k + \alpha_k p_k)^Tp_k \ge c_2\nabla f(x_k)^Tp_k$$

也就是说，$\phi$在$\alpha_k$的斜率，要比初始的$\phi'(\alpha_0)$大$c_2$倍。注意这里的$\nabla f(x_k)^Tp_k$是负的，所以，$\phi$在$\alpha_k$的斜率比调整过（乘以$c_2$）的原始斜率还要“平”一些。

为什么？

如果$\phi'(\alpha)$的斜率“负得很厉害”，那么如果$\alpha$再“前进一些”，则可以让$f$减少得更多。

完整的Wolfe条件，就是上面这两条。合并在一起就可以：

$$ f(x_k + \alpha_k p_k) \le f(x_k) + c_1\alpha_k\nabla f(x_k)^Tp_k $$

$$ \nabla f(x_k + \alpha_k p_k)^Tp_k \ge c_2\nabla f(x_k)^Tp_k$$

还有一个问题，就是，满足上面两条的$\alpha$，一定存在么？

来看这样一个引理：设函数$f$是实值标量函数，连续可微。$p_k$是$x_k$点的下降方向。假设函数$f$在$\\{x_k+\alpha p_k \| \alpha > 0\\}$有下界，那么，如果$0<c_1<c_2<1$，就一定存在这样的“步长区间”，使得Wolfe条件满足。

证明如下：

由已知，$\phi(\alpha)=f(x_k+\alpha p_k)$，对$\forall\alpha>0$有下界。

但是这个直线可没有下界：$l(\alpha) = f(x_k)+\alpha c_1\nabla f(x_k)^Tp_k$

所以它们必相交，至少一次。记$\alpha'$是它们的相交点。

于是，相交时，有

$$ f(x_k+\alpha'p_k) = f(x_k) + \alpha'c_1\nabla f(x_k)^Tp_k $$

那么在$\alpha'$点“之前”，式37就肯定满足，即“直线肯定在曲线上面”。

由中值定理，存在$\alpha'' \in (0,\alpha')$，使得

$$ f(x_k + \alpha' p_k) - f(x_k) = \alpha' \nabla f(x_k + \alpha''p_k)^Tp_k $$

把式39和式40联立，于是

$$ \nabla f(x_k+\alpha''p_k)^Tp_k = c_1\nabla f(x_k)^Tp_k > c_2\nabla f(x_k)^Tp_k $$

上式后面的小于号是因为$0<c_1<c_2$且$\nabla f(x_k)^Tp_k < 0$。

这不正好就是式38么。

所以，在$ \alpha'' $附近的区间，满足Wolfe条件。

但是，还是不知道$\alpha$怎么算……

在一些情况下，我们可以放弃Wolfe条件的后面那个，于是就有了一种叫做backtracking的线性搜索方法。这个方法倒是简单。

1、选取初始值：$\alpha>0$，$\rho\in(0, 1)$，$c\in(0, 1)$。

2、判断是否满足$f(x_k+\alpha p_k)\le f(x_k)+c\alpha\nabla f(x_k)^Tp_k$，如果满足则停止，否则$\alpha = \rho\alpha$。

这实际上是利用了之前所说的：当$\alpha$足够小的时候一定成立。

再后面，BFGS算法现在还有什么问题没有？

当数据的维数很大时，如1M，那么$B$矩阵的边长就是1M。所以一共是1MM个矩阵元素。然后double型是8个字节，所以只存储$B$矩阵本身，就需要8E12字节，即8T。这是不可接受的。

进一步改进：内存有限的BFGS算法（The limited-memory BFGS (L-BFGS or LM-BFGS) algorithm）。

