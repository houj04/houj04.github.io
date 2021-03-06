---
layout: post
title: "Deep Compression"
categories: mypost
---

# Deep Compression

## 论文介绍

* 文章标题：Deep Compression: compressing deep neural networks with pruning, trained quantization and huffman coding
* 作者：Song Han, Huizi Mao, William J. Dally
* 发表于：ICLR 2016 （best paper）

## 摘要

神经网络（neural network）需要大量的算力和内存，因此很难部署在硬件资源受限的嵌入式系统（embedded system）上。文章提出了一种叫做“深度压缩（deep compression）”的方法，包含三个阶段：剪枝（pruning）、带训练的量化（trained quantization）和霍夫曼编码（huffman coding）。三个阶段共同作用，可以将神经网络的存储空间减小35x到49x不等，并且没有影响模型的准确度（accuracy）。

首先，对神经网络进行剪枝，只学习出“重要”的连接（connection）。下一步，对权重（weight）进行共享（share）和量化。最后，应用霍夫曼编码。在前两个阶段需要对网络进行重训练（retrain），用来微调（fine tune）剩余的连接和量化中心点（centroid）。

剪枝将连接数减少9x到13x。量化将保存每个连接的比特（bit）数从32降低到5。在ImageNet数据集上，作者提出的方法将AlexNet的存储量降低了35x，从240MB降低到6.9MB，并且没有精度损失。在VGG-16网络上，降低了49x，从552MB降低到11.3MB，也没有精度损失。模型变小后，可以把模型放在SRAM缓存（cache）中，而不是DRAM内存（memory）中了，进而可以加速并且节约能源。

## 1 介绍

深度神经网络在计算机视觉的各种任务上应用广泛并且取得了很好的效果，但是尽管这些神经网络很强大，它们含有的大量的权重需要消耗很多的存储空间和内存带宽。例如，AlexNet Caffemodel超过200MB，VGG-16 Caffemodel超过500MB。这么大的模型在移动（mobile）端上很难进行部署和使用，原因有下面两个：

首先，对于百度（baidu）和脸书（facebook）这样的重视移动互联网的公司，各种手机应用程序都是通过应用程序市场更新的，因此对二进制文件的大小很敏感。例如，超过100MB的应用程序通常只在连接了wifi的情况下才会下载。尽管在移动设备上直接运行深度神经网络有很多优点（隐私保护、对带宽要求低、实时处理），但是超大的体积导致了无法在手机app中使用dnn。

另一个原因是资源消耗。运行大的神经网络需要大量的内存带宽来取权重值，并且需要大量的计算来进行内积（dot product）。访问内存是能源消耗的大头，在45nm CMOS工艺下，一个32位浮点数相加需要消耗0.9pJ，一个32位SRAM cache访问需要5pJ，而一个32位的DRAM内存访问需要640pJ，是加法操作的3个数量级。

体积较大的神经网络会需要很多的DRAM访问，进而带来更多的能量消耗。例如，以每秒20次的频率，运行一个有10亿（1 billion）连接的神经网络，那么仅仅在内存访问上就需要这么大的功率：$(20Hz)(1G)(640pJ) = 12.8W$，这个数值对于很多典型的移动设备来说是难以承受的。

作者们的目标是降低大型神经网络的推理（inference）阶段所需要的存储空间和能量，便于部署到移动设备上。因此提出了称为“深度压缩（deep compression）”的一个三阶段方法：（1）进行剪枝，把冗余（redundant）的连接移除，只保留最有用的连接；（2）将权重进行量化，不同的连接可以共享相同的权重，并且用码表（codebook）来保存权重，网络中只保存下标（index）；（3）用霍夫曼编码，根据有效权重的出现频率不同而进行压缩。

作者们发现，剪枝和量化可以在不互相影响的情况下对网络进行压缩，可以获得令人意外的高压缩率。压缩之后的大小只有几兆字节，因此可以把所有的权重都缓存到芯片上，而不需要每次都从DRAM取。

## 2 剪枝

在以前的文献中就有使用剪枝技术来减少网络的复杂度和防止过拟合（over-fitting）的做法。

首先用正常的训练方法来训练神经网络。然后将“小权重”的连接给剪掉，即：一个网络中，所有权重小于某个阈值的连接，都被去掉。最后对网络进行重训练，将剩余的未被剪掉的稀疏（sparse）连接的权重进行调整。在AlexNet网络上可以剪掉9x的参数个数，在VGG-16上面可以剪掉13x的参数个数。

剪枝后的稀疏结构用传统的稀疏矩阵表示法来表示，例如CSR（compressed sparse row）或者CSC（compressed sparse column），一共需要$2a+n+1$个数。其中$a$是非零元个数，$n$是行数或者列数。

为了继续压缩，不直接存储绝对位置，而是存的下标偏移量，并且把这个偏移量也进行了编码。对于卷积层（conv layer）用8比特，对于全连接层（fc layer）用5比特。当这个下标偏移量超过这几个比特能表示的范围时，用0进行占位。

用一个简单的情况来表示，假设用来表示偏移量的比特数是3，那么偏移量最多只能到2^3=8。那么当需要表示一个11的偏移的时候，在8的位置上放一个0的值和一个3的偏移，这样就能“接着”找到真正的位于11的下一个值。

## 3 量化和权重共享

在对网络进行剪枝之后，作者们继续使用量化和权重共享的方法，降低用来保存每个权重的比特数，进而可以进一步压缩。让某些不同连接共享相同的权重，就能够减少有效权重的总数。随后需要调整一下这些共享权重的权重值。

权重共享的方法如下。假设一个全连接层有4个输入节点和4个输出节点，权重矩阵的形状为4×4。将权重聚类到4个桶（bin）内，每个桶内的权重会共享相同的值，因此对于整个4×4的矩阵来说，只需要存储“桶的编号”即可。

在更新的时候，梯度（gradient）矩阵也按照权重矩阵的分桶方式来进行分桶，相同桶内的梯度会被加和，然后乘以学习率（learning rate）之后和共享的权重值相减。

在剪枝之后的AlexNet网络上，作者们对每一个卷积层用8比特（256个共享权重）进行了量化，对每一个全连接层用5比特（32个共享权重）进行了量化，没有精度损失。

为了计算压缩率，假设有$k$个聚类，那么用来记录“桶的下标”所需要的比特数是$\log_2(k)$。一般来说，一个含有$n$个连接的神经网络，每个连接用$b$个比特来表示，这些连接一共有$k$个共享权重，那么压缩率可以这样计算：

$$ r = \dfrac{nb}{nlog_2(k)+kb} $$

举例来说，某个只有单个隐藏层的神经网络，一共有16个参数，分4个桶。那么在原始情况下，需要用32比特来保存原始的每一个权重，现在（压缩后）只需要4个有效权重，每个仍是32比特。再加上16个2比特来保存下标，因此总的压缩率是$16 * 32/(4 * 32+2 * 16)=3.2$。

### 3.1 权重共享

作者们使用了K均值（k-means）聚类，来找到网络中每个层中的共享权重，所有落在同一个聚类中的权重都共享成同一个值。不同层的权重不共享。这个操作实际上是将$n$个原始权重$W= \\{ w_1, w_2, \dots, w_n \\}$聚类成$k$个类$C=\\{c_1, c_2, \dots, c_k\\}$，优化目标是让类内距离（within-cluster sum of squares，WCSS）最小：

$$ \arg\limits_C\min \sum_{i=1}^k \sum_{w\in c_i} |w-c_i|^2$$

注意，这里的网络是首先已经完全训练好的，在此基础上才进行聚类和共享权重的计算。

### 3.2 共享权重的初始化

聚类中心点的初始化选择，对聚类的质量以及网络的预测精度都有影响。作者们尝试了三种不同的初始化方法：随机、基于密度的、线性初始化。在进行剪枝之后，剩余的权重分布是一个双峰分布（bimodal distribution）。

随机：从数据中随机选择k个值，来初始化中心点。由于双峰分布中有两个峰值，随机初始化方法出来的结果，倾向于在两个峰值附近集中。

基于密度的：把权重的累计密度函数（CDF）在y轴上进行等距离切分，然后在x轴上找到对应的点就作为初始化的值。这种方法也会让聚类结果集中在原始峰值附近。

线性：统计出所有权重的最小值和最大值，然后将整个区间等分，取等分点作为初始化值。相对于前两种做法，这种方法得到的聚类中心点最分散。

根据之前的研究，大权重比小权重的作用更大，但是大权重的个数更少。所以对于随机初始化和基于密度的初始化方法来说，只有很少的中心点具有较大的绝对值（absolute value），进而很难表示出这些较少的但是值较大的权重的特点。而线性初始化就没有这个问题。实验表明线性初始化的效果最好。

### 3.3 前向和后向计算

前面说的“1维k均值聚类”得到的中心点的值是共享权重的权值。在前向计算（feed forward）和后向计算（back-propagation）的时候增加了一个查表的动作，每个连接都有一个下标，指向共享权重表中的某个元素。

后向计算的时候，需要把所有属于同一个桶的梯度都累加在一起，然后再更新权重。

记损失函数（loss funcion）为$L$，第$j$列和第$i$行的权重是$W_{ij}$，$W_{ij}$的中心点下标是$I_{ij}$，第$k$个中心点是$C_k$。用指示函数（indicator function）$\mathbb{1}(.)$的话，该中心点的梯度按照下式计算：

$$ \dfrac{\partial L}{\partial C_k} = \sum_{i, j} \dfrac{\partial L}{\partial W_{ij}} \dfrac{\partial W_{ij}}{\partial C_k} = \sum_{i, j} \dfrac{\partial L}{\partial W_{ij}} \mathbb{1}(I_{ij} = k)$$


## 4 霍夫曼编码

霍夫曼编码是一种常用的无损压缩手段，用变长码字（codeword）对符号进行编码，码表是根据每个符号的出现频率生成的：越频繁出现的符号，编码的长度就越短。

作者们的实验显示，量化之后的权重，和稀疏矩阵的下标的分布是不均匀的，多数量化后的权重都出现在两个峰值附近。而稀疏矩阵的下标的差距很少超过20。霍夫曼编码对这些非均匀分布的数值进行压缩，可以减少20%到30%的空间。














