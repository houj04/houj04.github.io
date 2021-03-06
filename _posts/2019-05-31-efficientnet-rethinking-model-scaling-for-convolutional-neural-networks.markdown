---
layout: post
title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
categories: mypost
---

# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

## 论文介绍

* 文章标题：EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
* 作者：Mingxing Tan, Quoc V. Le
* 发表于：ICML2019
* 参考：https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html
* 参考：https://arxiv.org/abs/1905.11946

## 摘要

卷积神经网络（Convolutional Neural Networks，ConvNets）通常是在一个受限的资源预算中进行设计开发，然后如果有更多的资源就进行扩张（scaled up）以获得更高的准确率。在本文中，作者们系统性地研究了模型的扩张，并且发现了：仔细进行网络深度（depth）、宽度（width）和分辨率（resolution）的平衡，可以获得更好的性能。基于这个发现，作者们提出了一种新的扩张方法，可以将深度、宽度、分辨率等等所有的维度进行同样的缩放，使用一种简单但是很高效的复合系数（compound coefficient）。作者们通过将MobileNet和ResNet进行扩张，展示了这种方法的有效性。

更进一步，作者们使用神经网络结构搜索（neural architecture search）来设计一个新的基线网络，并且将其扩张获得了一系列的模型，称为EfficientNet，比前面的卷积神经网络获得了更好的准确率和效率。特别的，作者们提出的EfficientNet-B7模型，在ImageNet数据集上分别获得了领先的84.4%和97.1%的top-1和top-5准确率，并且在推理阶段（inference）比已知的最好的卷积神经网络要小8.4x和快6.1x。作者们提出的EfficientNet们还可以很好进行迁移，并且获得领先的准确率：CIFAR-100数据集91.7%，Flowers数据集98.8%，以及其它3个迁移学习的数据集。论文的源代码已经对外公开。

## 1 介绍

对卷积神经网络进行扩张被广泛用于获得更好的准确率上。举例来说，ResNet可以从ResNet-18扩张到ResNet-200，使用了更多的层。近期，GPipe通过把一个基线模型扩大了4倍获得了领先的ImageNet数据集上84.3%的top-1准确率。但是，就扩张卷积神经网络的过程来说，从来没有被仔细研究过，并且目前有多种方法来做。最常见的方法是对深度进行扩张。另一种稍微不那么常见，但是逐渐流行起来的方法是通过提升图像分辨率来扩张模型。在前人的工作中，通常是把三个维度之一进行扩张：深度、宽度、图像大小。尽管也可以把两个或者三个维度进行扩张，但是任意的扩张就需要乏味的手工调整，并且通常也只能获得次优（sub-optimal）的准确率和效率。

在本文中，作者们将会研究扩张卷积神经网络的过程。特殊地，作者们将研究这个核心问题：是否有一种方法可以对卷积神经网络进行扩张，同时获得更好的准确率和更高的效率？经验上的研究显示，对网络结构的宽度、深度、分辨率的所有维度进行平衡是很重要的，并且很惊奇的是，这种平衡可以通过简单的对它们进行常数的缩放来获得。基于这个发现，作者们提出了一种虽然简单但是有效的复合扩张方法。和传统的实践（任意对这些因子进行缩放）不同的是，作者们提出的方法对网络的宽度、深度和分辨率用统一的缩放方法，使用一系列的固定的缩放因子。举例来说，如果希望使用$2^N$倍的更多的计算资源，那么可以简单地将网络的深度增加$\alpha^N$，宽度增加$\beta^N$，图像大小增加$\gamma^N$，这里$\alpha$、$\beta$、$\gamma$是常量参数，可以用原始的小模型进行一个小的网格搜索（grid search）来确定。原文中的图2展示了作者们提出的扩张方法和传统方法的比较。

直觉上来说，这种复合的扩张方法应该是好使的因为如果输入图像更大，那么网络也就需要更多的层来增加接收区域（receptive field），需要更多的通道（channel）来获得更加精准的模式（pattern）。实际上以前有一些理论和实践的结果都显示了确实在网络宽度和深度之间存在着某种联系。但是就作者们所知道的情况而言，他们是第一个来实际度量一个网络的宽度、深度和分辨率这三者之间的关系的。

作者们的扩张方法在目前的MobileNet和ResNet上面表现都不错。值得注意的是，模型扩张的效果严重依赖于基线网络。为了进一步证实，作者们使用了网络结构搜索来获得一个新的基线网络，并且对它进行了扩张，获得了一组模型，称为EfficientNet。原文中的图1总结了在ImageNet数据集上的性能，作者们提出的EfficientNet显著超过了其它的卷积神经网络。特别的，作者们的EfficientNet-B7超过了最好的存在的GPipe的准确率，使用了8.4x更少的参数，推理的时候快6.1x。和广泛使用的ResNet相比，作者们的EfficientNet-B4将top-1准确率从ResNet-50的76.4%提升到了82.6%，而FLOPS数接近。除了ImageNet之外，EfficientNet在8个广泛使用的数据集里面的5个上都可以很好迁移，并且获得领先的准确率，同时还能比已知的卷积神经网络参数减少，最多能减少21x。

## 2 相关工作

准确率（accuracy）：自从AlexNet在2012年的ImageNet竞赛上夺冠之后，卷积神经网络越来越大，也越来越准确。2014年ImageNet的冠军GoogleNet用6.8M个参数获得了74.8%的top-1准确率。2017年ImageNet的冠军SENet用145M个参数获得了82.7%的top-1准确率。近期，GPipe将ImageNet的验证集top-1指标的最好成绩向前推进到了84.3%，用了557M个参数：它实在太大了，因此只能用一种特殊的流水线并行库，将网络拆分然后将每个部分分到不同的加速器上面。这些模型主要都是为ImageNet而设计的，但是近期的一些研究也显示了更好的ImageNet模型在其他的一系列迁移学习数据集上表现也更好，以及一些其它的计算机视觉任务上，例如目标检测（object detection）。尽管对很多应用来说高准确率很重要，作者们已经遇到了硬件内存的限制，因此更高的准确率提升需要更好的效率。

效率（efficiency）：很深的卷积神经网络通常参数都过多了（over-parameterized）。模型压缩（model compression）是一种用准确率换效率的减小模型大小的方法。由于手机越来越变得无处不在，所以手工定制适合于移动设备的模型也很常见，例如SqueezeNet、MobileNet、ShuffleNet。最近，神经网络结构搜索开始更流行于设计高效的适合移动设备的网络，并且还能获得甚至比手工定制的移动网络更好的效率（通过仔细调整网络的宽度、深度、卷积核的类型和大小等等）。但是，目前仍然不清楚的是如何将这些技术应用于大模型上，它们有显著更大的设计空间以及非常多的调整的消耗。在本文中，作者们意图研究能够超越前人工作的准确率的超大卷积神经网络的模型效率。为了获得这个目标，作者们采取了模型扩张的方法。

模型扩张：根据不同资源的限制，对一个网络进行缩放的做法有很多，例如ResNet可以只调整网络的深度（层数）就做到向下收缩（到ResNet-18）或者扩张（ResNet-200）。而WideResNet和MobileNet则是通过网络的宽度（通道数）进行缩放。另外，通常都认为更大的输入图像的大小会对准确率的提升有帮助，但是需要浪费更多FLOPS。尽管一些前人的研究显示了网络的深度和宽度对卷积神经网络的表现力都很重要，但是仍然有一个开放性的问题：如何高效率地对一个卷积神经网络进行缩放，获得更好的效率和准确率。作者们的工作系统性和实践性地对卷积神经网络进行了研究，对网络的三个维度：宽度、深度、分辨率进行了缩放。

## 3 复合模型缩放

在这一节，会对扩张问题进行公式化，研究不同的方法，并且提出作者们的新的扩张方法。

### 3.1 问题的公式化

一个卷积神经网络的层$i$可以被定义成这样的函数：$Y_i=F_i(X_i)$，这里$F_i$是操作符（operator），$Y_i$是输出的张量（tensor），$X_i$是输入张量，形状是$<H_i, W_i, C_i>$（为了简便起见，省略了批处理的维度）。这里的$H_i$和$W_i$是空间维度（spatial dimension），$C_i$是通道维度（channel dimension）。一个卷积神经网络记作$N$，可以看成是一系列连接在一起的层：$N=F_k\odot\dots\odot F_1\odot F_1(X_1)=\odot_{j=1\dots k}F_j(X_1)$。在实际应用的时候，卷积神经网络的层通常切分成多个阶段（stage），所有的层在每一个阶段中都共享相同的结构：例如，ResNet有5个阶段，所有的层在每个阶段都有相同的卷积类型，除了第一层执行的是下采样（down-sampling）。因此，可以把一个卷积神经网络定义成为

$$ N=\bigodot_{i=1\dots s}F_i^{L_i}\left (X_{<H_i, W_i, C_i>}\right ) $$

上式中$F_i^{L_i}$表示层$F_i$在阶段$i$中重复了$L_i$次。$<H_i, W_i, C_i>$表示层$i$的输入张量$X$的形状。原文中的图2（a）展示了一个卷积神经网络的代表，其中空间维度是逐渐缩小的但是通道维度随着层数而逐渐扩张。例如，从最开始的形状是$<224, 224, 3>$一直到最后的输出$<7, 7, 512>$。

和普通的卷积神经网络的设计专注于找到最好的层的结构$F_i$不同，模型扩张尝试去把网络的长度（$L_i$），宽度（$C_i$）和/或分辨率（$H_i$和$W_i$）进行扩张，而不修改在基线网络里面预定义的$F_i$。通过固定$F_i$的方法，模型扩张把面对新的资源需求限制的设计问题，给简化了，但是仍然有一个很大的设计空间来对每一层探索不同的$L_i$、$C_i$、$H_i$、$W_i$。为了进一步压缩设计空间，作者们限制了所有的层都必须用同样的比例进行缩放。作者们的目的是对任意给定的资源限制都能找到最大化的模型准确率，可以公式化为下面的一个优化问题：

$$
\max_{d, w, r} Accuracy(N(d, w, r)) \\
s.t. N(d, w, r) = \bigodot_{i=1\dots s}F_i^{d\cdot L_i}(X_{<r\cdot H_i, r\cdot W_i, w\cdot C_i>}) \\
memory(N) \le target\_memory \\
FLOPS(N) \le target\_flops \\
$$

上式中$w$、$d$、$r$是用来对网络的宽度、深度、分辨率进行缩放的系数；$F_i$、$L_i$、$H_i$、$W_i$、$C_i$是在基线网络里面预定义的参数（参考原文中的表格1作为例子）。

### 3.2 伸缩维度

式2的主要难点在于，最优化的$d$、$w$、$r$之间有互相的依赖，并且在不同的资源限制下的值会变。由于有这个困难，传统方法主要从以下三个维度对卷积神经网络进行缩放：

深度$d$：对网络深度进行缩放是很多网络结构中最通常使用的方法。直觉上是更深的网络可以抓到更丰富和更复杂的特征，并且在新的任务上泛化能力（generalize）更好。但是，更深的网络也更难训练，因为有梯度消失（vanishing gradient）问题。尽管有一些技术，例如跳边（skip connection），或者是批处理归一化（batch normalization），可以用来缓和一下训练中的问题，但是对非常深的网络来说，准确率的上升并不多。例如，ResNet-1000和ResNet-101的准确率差不多，即使它有明显更多的层。原文中的图3（中）展示了作者们的实际测试，对一个基线模型，用不同的深度系数$d$进行缩放，也说明了对于非常深的网络中逐渐减小的准确率回报。

宽度$w$：对比较小的模型来说，扩大网络的宽度是个常见操作。在其他的一些文献中有讨论，更宽的网络能够抓到更多精细的特征，并且更加容易训练。尽管如此，特别宽并且很浅（shallow）的网络倾向于在获得高层次特征时会比较困难。作者们的实验结果展示在原文的图3（左）里面，可以看出在网络随着更大的$w$变得更宽的时候，准确率迅速饱和（saturate）。

分辨率$r$：给了更高分辨率的输入图像，卷积神经网络可以获得更多精细化的模式。从早期的224x224开始，现在的网络倾向于用299x299或者331x331来获得更好的准确率。近期，GPipe获得了ImageNet的最好结果，它的分辨率是480x480。更高的分辨率，例如600x600，在目标检测中也有广泛的应用。原文中图3（右）展示了对分辨率进行缩放的结果，确实更高的分辨率可以提升准确率，但是对超高分辨率来说，准确率的获得也越来越少。（$r=1.0$表示分辨率224x224，$r=2.5$表示分辨率560x560）

上面的分析给了作者们第一个发现：

发现1：把网络的任意一个维度（宽度、深度、分辨率）提升，都可以获得准确率的提升，但是对于更大的模型，这种提升会衰减。


### 3.3 复合缩放

作者们发现，不同的维度上的缩放，不是独立的。直观上看，对于分辨率高的图像，应该增加网络的深度（这里是不是有typo，应该是宽度？），这样使得更大的接收区域可以帮助抓到在更大图像中才有的包括更多像素的相似特征。对应的，也需要增加网络深度，当分辨率更大的时候，因为要抓住更多精细化的包含更多高分辨率图像中的更多像素点的模式。这些直观上的东西提示应该协调和平衡不同的维度的缩放，而不是传统方法的单维度缩放。

为了验证作者们的直觉，作者们比较了在不同网络深度和分辨率下面缩放宽度的情况，展示在原文中的图4中。如果不修改深度（$d=1.0$）和分辨率（$r=1.0$）而只调整宽度，准确率很快就饱和了。而用更深（$d=2.0$）和更高分辨率（$r=2.0$）的时候，在在同样的FLOPS的消耗下，调整宽度获得了明显更好的准确率。这些结论给出了第二个发现：

发现2：为了最求更好的准确率和效果，很重要的一点是，在调整卷积神经网络的缩放时，要平衡所有的维度：宽度、深度和分辨率。

实际上，一些前人的工作已经在对网络的宽度和深度进行平衡，但是他们都需要大量的人工工作。

在本文中，作者们提出了一种新的复合扩张方法，使用了一个复合的系数$\phi$来对网络的宽度、深度和分辨率统一进行缩放：

$$
depth: d = \alpha^\phi \\
width: w = \beta^\phi \\
resolution: r = \gamma^\phi \\
s.t. \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2 \\
\alpha \ge 1, \beta \ge 1, \gamma \ge 1 
$$

上式中的$\alpha$、$\beta$、$\gamma$是常数，可以用一个小的网格搜索来确定。直观地说，$\phi$是一个用户指定的参数，用来控制模型缩放的时候有多少资源可以用。而$\alpha$、$\beta$、$\gamma$指定了如何把这些多出来的资源分配到网络的宽度、深度和分辨率上。值得注意的是，对于一个正常的卷积操作来说，FLOPS是和$d$、$w^2$、$r^2$成正比的，也就是说，把网络的深度加倍则会导致FLOPS的加倍，而网络宽度的加倍或者分辨率的加倍会导致FLOPS的4倍。由于卷积操作在卷积神经网络里面通常是占主导地位的计算量，所以按照公式3来进行缩放的话，大致会让总的FLOPS上升至$(\alpha\cdot\beta^2\cdot\gamma^2)^\phi$。在本文中，作者们限制了$\alpha\cdot\beta^2\cdot\gamma^2\approx2$，这样对于任意的新的$\phi$，总的FLOPS会大约上升$2^\phi$。

## 4 EfficientNet的结构

由于模型的缩放并不影响基线网络中的层内的操作$F_i$，因此拥有一个好的基线网络也是很重要的。作者们在已经存在的卷积神经网络上面评估了他们提出的方法，但是为了更好地展示方法的有效性，作者们还制作了一个新的适合移动的网络基线，称为EfficientNet。

受前人MNAS工作的启发（注，其实MNAS也是作者的文章，推测是匿名评审的原因，不能暴露自己的身份信息所以写的是“前人”的工作），作者们制作了自己的基线网络，用了一个多目标网络结构搜索，同时优化准确率和FLOPS。特别地，用了相同的搜索空间，用$ACC(m)\times[FLOPS(m)/T]^w$作为优化目标，其中$ACC(m)$和$FLOPS(m)$分别是模型$m$的准确率和FLOPS。$T$是目标FLOPS，$w=-0.07$是超参数，用来控制准确率和FLOPS的平衡。和前人工作有所不同的是，这里本文作者们优化的是FLOPS而不是延迟（latency）因为并不是针对任何的一个特定的硬件设备。作者们的研究产出了一个高效的网络，命名为EfficientNet-B0。由于本文作者使用了和MNAS作者相同的搜索空间，所以网络结构和MNAS也接近，除了EfficientNet-B0稍微大一点，因为设定的FLOPS目标是400M，比MNAS稍微大一点。文中的表格1展示了EfficientNet-B0的网络结构，它的主要的构成块是mobile inverted bottleneck，作者们也增加了aqueeze-and-excitation优化。

从基线的EfficientNet-B0开始，作者们应用了提出的复合扩张的方法，对它进行了两个步骤的扩张：

1、先固定$\phi=1$，假设可用的资源有两倍，基于公式2和公式3，对$\alpha$、$\beta$、$\gamma$进行了一个小的网格搜索，特别的，作者们找到了针对EfficientNet-B0的最好的值是$\alpha=1.2$、$\beta=1.1$、$\gamma=1.15$，限制是$\alpha\cdot\beta^2\cdot\gamma^2\approx 2$。

2、固定$\alpha$、$\beta$、$\gamma$为常数，然后按照公式3将基线网络用不同的$\phi$进行扩张，获得了EfficientNet-B1一直到B7。参考原文中的表格2。

需要注意的是，如果用一个更大的模型，直接在上面搜索$\alpha$、$\beta$、$\gamma$的值，是有可能获得更好的性能的。但是搜索所需要的代价会明显增大。作者们提出的解决方案是仅在小的基线网络上进行搜索（第一步），然后将同样的收缩系数应用于其他模型（第二步）。









