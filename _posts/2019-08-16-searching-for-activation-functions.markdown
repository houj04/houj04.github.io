---
layout: post
title: "Searching for Activation Functions"
categories: mypost
---

# Searching for Activation Functions

## 论文介绍

* 文章标题：Searching for Activation Functions
* 作者：Prajit Ramachandran, Barret Zoph, Quoc V. Le
* 参考：https://arxiv.org/abs/1710.05941

## 摘要

深度神经网络（deep network）中的激活函数（activation function）的选择，对网络的训练和任务的效果有很大的影响。目前，最成功并且最广泛应用的激活函数是Rectified Linear Unit（ReLU）。尽管有很多手工设计的用来替换ReLU的激活函数被提出，但是没有一个能成功替换掉，因为收益不稳定。在本文的工作中，作者们提出了一个自动化的搜索技术，用来发现新的激活函数。使用一个exhaustive和强化学习（reinforcement learning）结合的方法，作者们发现了多个新的激活函数。作者们通过进行实验性的验证的方法，使用发现的最好的激活函数，来对这些搜索结果进行有效性的验证。实验显示找到的最好的激活函数，$f(x)=x\cdot sigmoid(\beta x)$，起名为Swish，在一系列有挑战性的数据集上，在更深的模型上，表现比ReLU要好。举例来说，简单把ReLU替换成Swish，可以在ImageNet数据集上把Mobile NASNet-A的top-1分类正确率提升0.9%，吧Inception-ResNet-v2提升0.6%。Swish所具有的简单性，以及它和ReLU的相似性，使得在任意的神经网络之哦跟你都可以简单地把ReLU替换成Swish。

## 1 介绍

在每个深度神经网络的核心中，都有一个线性变换（linear transformation）和紧随其后的激活函数$f(\cdot)$。激活函数在训练深度神经网络中起到一个很重要的作用。目前，最成功也是应用最广泛的激活函数是Rectifiled Linear Unit（ReLU），定义是$f(x)=max(x,0)$。ReLu的使用使得现代的神经网络可以进行完全的监督训练（supervised training）。用ReLu的深度神经网络更容易优化，相对于使用sigmoid或者tanh来说，因为当ReLU函数的输入是正的（positive）的时候，梯度是可以直接流过去的。由于它的简单和高效率，ReLU成为了通常情况下的默认激活函数，在整个深度学习（deep learning）界都在使用。

尽管有大量的激活函数被提出，希望用来替代掉ReLu，但是没有一个能获得类似ReLU的广泛的应用。很多使用者喜欢ReLU的简单以及可靠性，因为其他激活函数带来的性能提升在不同的模型和数据集上并不稳定（inconsistent）。

那些被提出来用来替换ReLU的激活函数，都是手工设计的，用来满足某种需求。然而，最近有一些使用搜索技术来自动发现传统的需要手工设计的组建，近期有一些工作展示出非常有效。举例来说，有人使用了强化学习的方法来找到一个可重复的（repilicable）卷积层（convolutional cell），可以在ImageNet数据上超越手工设计的网络结构。

在本文的工作中，作者们使用自动化的搜索技术，用来发现新的激活函数。作者们专注于找到新的标量（scalar）激活函数，即，输入是一个标量，输出也是一个标量。因为这种标量激活函数可以用来替换掉ReLU函数，而无需修改网络结构。使用了结合exhaustive和强化学习的搜索方法，作者们发现了一系列新的激活函数，展现了有前途（promising）的表现。为了进一步验证用搜索的方法来寻找标量激活函数的方法的有效性，作者们用实验评估了找到的最好的激活函数。找到的最好的激活函数，作者们称之为Swish，是$f(x)=x\cdot sigmoid(\beta x)$，这里$\beta$是一个常量或者是一个可以训练的参数。作者们精细的实验展示了Swish可以稳定匹配ReLU或者超过它，在一系列有挑战的领域中，比如图像分类（image classification）和机器翻译（machine translation）。在ImageNet数据集上，用Swish替换掉ReLU，将top-1分类正确率提升了0.9%，用的是Mobile NASNet-A网络，如果用Inception-ResNet-v2网络则可以提升0.6%。这些准确率的提升是很大的，因为整整一年的结构调整（architectural tuning）和扩大（enlarging），从Inveption-ResNet-v2升级到Inveption V3的提升量是1.3%。

## 2 方法

为了使用搜索技术，需要设计一个搜索空间（search space），它需要包含有希望的候选激活函数。在设计搜索空间时候的一个重要的挑战是，要平衡搜索空间的大小和表现力（expressivity）。一个过分受限制的搜索空间将不会包含新的激活函数，而一个过大的搜索空间又很难进行高效率搜索。为了平衡这两个准则（criteria），作者们设计了一个简单的搜索空间，受到了优化器（optimizer）搜索空间的启发，该空间包含用来构建激活函数的一元（unary）和二元（binary）函数。

如原文中的图1所示，图中的激活函数可以看成是重复组合（compose）“核心单元（core unit）”，定义为$b(u_1(x_1),u_2(x_2))$。核心单元有两个标量输入，把每个输入独立通过一个一元函数，然后将两个一元的输出用一个二元函数进行合并，最后输出一个标量。由于目的是找到一个标量激活函数，用来把一个标量转换成另一个标量，那么一元函数的输入限制成这一层（layer）的激活之前的值$x$或者是二元函数的输出。

给定了搜索空间之后，搜索算法的目标是找到这些一元函数和二元函数的有效组合。搜索算法的选择，依赖于搜索空间的大小。如果搜索空间小，例如使用单个核心单元，那么就可以穷举遍历（enumerate）整个搜索空间。如果核心单元重复很多次，那么搜索空间就会极度变大（例如，上升到$10^{12}$这个数量级）使得穷举不可能。

对于大的搜索空间，作者们使用了一个循环神经网络（RNN）控制器，如原文的图2所示。在每一个时间步（timestep）上，控制器预测出激活函数的一个单独的分量。这个预测会送给控制器的下一个时间步，这个过程会一直重复直到每个激活函数的分量都被预测出来。预测出来的字符串将用于构建激活函数。

一旦一个候选的激活函数被搜索算法生成出来，一个用了这个候选的激活函数的“子网络（child network）”在同样的任务上进行训练，例如CIFAR-10数据集上的图像分类任务。在训练之后，子网络的验证集准确率会记下来，并且用来更新搜索算法。在穷举法的情况下，一系列表现最好的激活函数，按照验证集的准确率排序记下来。在RNN控制器的情况下，控制器用强化学习的方法进行训练，来最大化验证集准确率，这里的验证集准确率作为强化学习中的奖励（reward）。训练过程推动控制器生成具有高验证集准确率的激活函数。

## 3 搜索成果

作者们所有的搜索都是在下列情况下执行的：选用ResNet-20作为子网络的结构，数据用CIFAR-10，训练10k步。这个受限制的环境可能潜在会把结果歪曲（skew），因为表现最好的激活函数可能只在小网络上面表现好。尽管如此，作者们会在实验部分说明，多数找到的函数可以在更大的模型上泛化（generalize）。在小的搜索空间上进行了完全的搜索，而在大的搜索空间上使用了RNN控制器。RNN控制器的训练使用了Policy Proximal Optimization，对奖励用了exponential moving average作为基线（用来减少方差）。所考虑使用的一元函数和二元函数的完整列表如下：

* 一元函数：$x$，$-x$，$|x|$，$x^2$，$x^3$，$\sqrt{x}$，$\beta x$，$x+\beta$，$\log(|x|+e)$，
$\exp(x)$，$\sin(x)$，$\cos(x)$，$\sinh(x)$，$\cosh(x)$，
$\tanh(x)$，$\sinh^{-1}(x)$，$\tan^{-1}(x)$，$sinc(x)$，
$\max(x,0)$，$\min(x,0)$，$\sigma(x)$，$\log(1+\exp(x))$，
$\exp(-x^2)$，$erf(x)$，$\beta$

* 二元函数：$x_1+x_2$，$x_1\cdot x_2$，$x_1-x_2$，$\dfrac{x_1}{x_2+e}$，$\max(x_1, x_2)$，$\min(x_1,x_2)$，$\sigma(x_1)\cdot x_2$，
$\exp(-\beta(x_1-x_2)^2)$，$\exp(-\beta|x_1-x_2|)$，$\beta x_1+(1-\beta)x_2$

上面的$\beta$是每个通道（channel）都不同的可训练的参数（trainable parameter），以及$\sigma(x)=(1+exp(-x))^{-1}$是sigmoid函数。不同的搜索空间的构造的时候，使用了不同的用来构建激活函数的核心单元的数量，以及可供候选的不同的一元和二元函数。

原文中的图3展示了搜索获得的最好的新的激活函数。从搜索结果中显示出来的一些值得注意的东西：

* 复杂的激活函数持续不如简单的激活函数，可能是因为增加了优化的复杂度。表现最好的激活函数由1个或2个核心单元组成。

* 表现最好的几个激活函数有一个结构上的共性，都是使用了原始的激活之前的值$x$作为一个输入，直接给到最后的二元函数：$b(x,g(x))$。ReLU函数其实也是符合这个结构的，其中$b(x_1,x_2)=max(x_1,x_2)$，并且$g(x)=0$。

* 搜索里面有一些使用了周期函数（periodic function），如sin和cos。最常见的使用周期函数的方法是用原始未激活的值$x$（或者线性缩放（scale）之后的$x$）与之做加法或者做减法。在激活函数中使用周期函数，这件事情只是在前人的工作中简单尝试过一些。所以这些发现的激活函数为以后的研究提出了一个新的方向。

* 包含了除法的函数倾向于效果不好，因为当分母(denominator）接近于0的时候输出会爆炸（explode）。除法只在某些情况下获得成功：要么分母会远离0（比如$\cosh(x)$），要么接近于0的时候分子（numerator）也接近于0，这样会输出1。

由于这些激活函数是用一个相对小的子网络获得的，这些的性能表现可能不能泛化到较大的模型上。为了检测这几个最好的新的激活函数在不同结构上的表现的鲁棒性（robustness），作者们进行了附加的实验，使用了ResNet-164 (RN)、Wide ResNet 28-10 (WRN)、DenseNet 100-12 (DN)。作者们用TensorFlow实现了上述三个模型，将ReLU函数替换成了上述每一个搜索中获得的表现最好的新激活函数。作者们使用了前人工作中同样的超参数（hyperparameter），例如用带momentum的SGD进行优化，并且参照以前的工作，运行5次然后报告的值是中位数（median）。

运行结果如原文中的表1和表2所示。不管模型结构的变化，8个激活函数中的6个都成功泛化。这6个激活函数中，所有的在ResNet-164上都打平或者超过ReLU。更进一步，找到的激活函数中有两个，$x\cdot\sigma(\beta x)$和$\max(x,\sigma(x))$，在三个模型上都稳定打平或者超过ReLU。

尽管上述结论看起来是有希望的（promising），但是仍然不清楚这些发现的激活函数是否能够在有挑战性的真实世界的数据集上成功替换掉ReLU。为了验证这些搜索的有效性，本文的剩余部分将专注于实验性对激活函数$f(x)=x\cdot\sigma(\beta x)$进行评估，作者们称这个叫做Swish。作者们选择广泛地（extensively）对Swish进行评估，而不是$\max(x,\sigma(x))$，因为早期实验现实了Swish的泛化能力更好。在后续的章节中，作者们分析了Swish的特点，然后进行了一个完整的实验的评估，对Swish、ReLU以及其他候选的基线激活函数进行了对比，使用了一系列的大模型和各种的任务。

## 4 Swish

简要概述一下，Swish的定义是$x\cdot\sigma(\beta x)$，其中$\sigma(z)=(1+\exp(-z))^{-1}$是sigmoid函数，$\beta$是一个常数或者是一个可以训练的参数。原文中的图4展示了在不同的$\beta$下面的Swish的图像。如果$\beta=1$，那么Swish等价于Sigmoid-weighted Linear Unit（SiL），被前人为了强化学习而提出。如果$\beta=0$，Swish变成了scaled linear function$f(x)=\dfrac{x}{2}$。当$\beta\rightarrow\infty$时，sigmoid分量接近于0-1函数，因此Swish逐渐变成接近ReLU函数。这说明Swish可以被大致看成是一个平滑的函数，它非线性（nonlinearly）插值（interpolate）在线性函数（linear function）和ReLU函数之间。而插值的度（degree）则可以被模型来控制，如果$\beta$设置成一个可训练的参数。

和ReLU类似，Swish没有上界（unbounded above）但是有下界（bounded below）。和ReLU不相似的是，Swish是平滑（smooth）的并且是非单调（non-monotonic）的。事实上，Swish的这种非单调性将其区别于多数常见的激活函数。Swish的导数（derivative）是：

$$
\begin{align}
f'(x) &= \sigma(\beta x) + \beta x\cdot\sigma(\beta x)(1-\sigma(\beta x)) \\
&= \sigma(\beta x)+\beta x\cdot\sigma(\beta x)-\beta x \cdot \sigma(\beta x)^2 \\
&= \beta x \cdot \sigma(x) + \sigma(\beta x)(1-\beta x\cdot\sigma(\beta x)) \\
&=\beta f(x) + \sigma(\beta x)(1-\beta f(x))
\end{align}
$$

在不同的$\beta$值的时候，Swish的一阶导数（first derivative）展示在原文中的图5。$\beta$的缩放（scale）控制了一阶导数接近（asymptote）0和1的速度。当$\beta=1$的时候，对于小于1.25（大约值）的输入来说，导数的幅度（magnitude）会小于1。因此，$\beta=1$时候的Swish的成功说明了ReLU的“保留梯度特性（gradient perserving property）”（即，当$x>0$的时候梯度是1）在现代的结构中可能不再是一个有区分度的优势（distinct advantage）。

Swish和ReLU的最大的区别是那个非单调的“颠簸（bump）”，在$x<0$区间中。原文中的图6显示，有一大部分的未激活的值（preactivation）落在了颠簸区（$-5\le x \le 0$）中，说明这个非单调的颠簸是Swish的一个重要部分。这个颠簸的形状可以通过调整$\beta$参数来进行控制。尽管在实践中固定$\beta = 1 $是有效的，在后续的实验部分会展示在某些模型上，对$\beta$进行训练可以更进一步提升性能。原文的图7展示了训练后的$\beta$的值的分布，用的是Mobile NASNet-A模型。训练得到的$\beta$的值从0到1.5都有分布，在$\beta\approx 1$有一个峰值（peak），提示了模型从可训练的$\beta$参数而带来的灵活性中获得了收益。

实践中，在多数深度学习库中，只修改一行代码就可以实现Swish。例如TensorFlow（用`x * tf.sigmoid(beta * x)`），或者是直接`tf.nn.swish(x)`（需要用作者们发布出来的版本）。需要注意的是，如果同时使用了批处理归一化（BatchNorm），那么需要设置scale parameter。某些上层（high level）库会默认关闭scale parameter，因为ReLU函数是piecewise linear的，但是这种设置对Swish是错误的。要训练Swish的网络，作者们发现把训练ReLU网络时候的学习率（learning rate）稍微降低一些会工作得不错。











































