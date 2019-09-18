# 向量, 矩阵与张量梯度的理论推导

机器学习, 神经网络的反向传播的基础在于梯度计算, 涉及向量(vector)、矩阵(matrix)以及张量(tensor)各类梯度, 明白其中的计算原理和规则具有一定的重要性。[2]中巨细无遗地阐述了与机器学习相关的各类calculus知识, 对于夯实相关基础大有裨益。[1]中从基础出发, 以常见的示例阐明了向量, 矩阵, 张量相关求导的**推导方法**, 极具启发性。本文总结[1]的核心思路如下, 并以一个常见的例子作为练习该推导方法。

## 基本思路

在进行梯度推导时, 从**标量**对**标量**的导数推导出发, 再组合为**标量|向量|矩阵**对**标量|向量|矩阵**的求导<sup>[1]</sup>。

## 例子

<img alt="${\bf X}: B \times N$" src="svgs/d0fb78af684d969978cc430874e9e4d0.svg" align="middle" width="76.3752pt" height="22.557149999999986pt"/>矩阵, <img alt="${\bf W}: N \times M$" src="svgs/9fe2ff9a739a82a0609b6813cb73aebd.svg" align="middle" width="86.33526pt" height="22.557149999999986pt"/>矩阵, <img alt="${\bf Y}: B\times M$" src="svgs/3896a191eb77fc9ace5b6f01ceb5dccd.svg" align="middle" width="79.587585pt" height="22.557149999999986pt"/>矩阵, 而<img alt="$\tilde{ {\bf Y} }$" src="svgs/02c3c2783a29871731d4e15e487956c0.svg" align="middle" width="14.764860000000004pt" height="30.359009999999977pt"/>表示如下: 

<p align="center"><img alt="$$&#10;\tilde{ {\bf Y} } = {\bf X} \cdot {\bf W},&#10;$$" src="svgs/896e0965e2c1aec3de7b375b20b6d9d0.svg" align="middle" width="87.21866999999999pt" height="18.375719999999998pt"/></p>

定义loss为<img alt="$L=\sum_b \sum_m ( \tilde{ y }_ {b, m} - y_ {b, m} )^2$" src="svgs/a9816abdc26b23b1de5bf67e1e3fb444.svg" align="middle" width="189.528405pt" height="26.76201000000001pt"/>。  
以上的例子表示一组Batch size为<img alt="$B$" src="svgs/61e84f854bc6258d4108d08d4c4a0852.svg" align="middle" width="13.293555000000003pt" height="22.46574pt"/>的输入维度为<img alt="$N$" src="svgs/f9c4988898e7f532b9f826a75014ed3c.svg" align="middle" width="14.999985000000004pt" height="22.46574pt"/>, 输出维度为<img alt="$M$" src="svgs/fb97d38bcc19230b0acd442e17db879c.svg" align="middle" width="17.739810000000002pt" height="22.46574pt"/>的数据集通过一层fully connected网络拟合(无bias项)。现在我们需要推导反向传播中需要计算的项<img alt="$\frac{\partial L}{ \partial {\bf W} }$" src="svgs/0fe6da93d2f7c1971887fa42380c616b.svg" align="middle" width="23.155934999999996pt" height="28.926479999999973pt"/>。注意到其中<img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align="middle" width="11.187330000000003pt" height="22.46574pt"/>为标量(scalar), 而<img alt="${\bf W}$" src="svgs/ce01fa56c40e0a3b42fb4a4c6c6d67fe.svg" align="middle" width="19.805940000000003pt" height="22.557149999999986pt"/>为<img alt="$N\times M$" src="svgs/54d384aec447994fc72866673a4796ac.svg" align="middle" width="52.83102000000001pt" height="22.46574pt"/>矩阵(matrix), 我们按照以上的指导原则, 先确定<img alt="$\frac{\partial L}{ \partial w_ {n, m} }$" src="svgs/b64c25958bb7a0966803e06dc9ad912a.svg" align="middle" width="39.15945pt" height="28.926479999999973pt"/>的表达式, 推导如下:  

<p align="center"><img alt="$$&#10;\frac{\partial L}{ \partial w_ {n, m} } = \frac{\partial L}{\partial \tilde{ {\bf Y} }} \frac{ \partial \tilde{ {\bf Y} } }{ \partial w_ {n, m} },&#10;$$" src="svgs/0a2a7d348c7fc5a89c388fb2b2e3e547.svg" align="middle" width="152.60354999999998pt" height="42.279104999999994pt"/></p>

注意到其中第一部分为标量对<img alt="$B\times M$" src="svgs/70d7063384beb3728b7262f661ed6273.svg" align="middle" width="51.124425pt" height="22.46574pt"/>矩阵的梯度, 第二部分为<img alt="$B\times M$" src="svgs/70d7063384beb3728b7262f661ed6273.svg" align="middle" width="51.124425pt" height="22.46574pt"/>矩阵对标量的梯度; 两者结果均为<img alt="$B\times M$" src="svgs/70d7063384beb3728b7262f661ed6273.svg" align="middle" width="51.124425pt" height="22.46574pt"/>矩阵, 两者间的"乘"是指对应项相乘再相加, 即"内积"。接下来我们分别计算这两项。

* 第一项: <img alt="$\frac{\partial L}{\partial \tilde{ {\bf Y} }}$" src="svgs/cbeb8150a337ba5410a36aadc575f7b9.svg" align="middle" width="19.232564999999997pt" height="28.926479999999973pt"/>

同样地, 我们首先拆解为标量对标量的求导问题, 如下: 

<p align="center"><img alt="$$&#10;\begin{aligned}&#10;\frac{ \partial L }{\partial \tilde{ y }_ {i, j} } &amp;= \frac{ \partial }{ \partial \tilde{ y }_ {i, j} } \left[ \sum_b \sum_m ( \tilde{ y }_ {b, m} - y_ {b, m} )^2 \right] \\&#10;&amp;= 2 ( \tilde{ y }_ {i, j} - y_ {i, j} ),&#10;\end{aligned}&#10;$$" src="svgs/a6a93b26036c45f5fde7ebb27455f1ef.svg" align="middle" width="267.05745pt" height="75.2697pt"/></p>

即<img alt="$L$" src="svgs/ddcb483302ed36a59286424aa5e0be17.svg" align="middle" width="11.187330000000003pt" height="22.46574pt"/>对<img alt="$\tilde{ {\bf Y} }$" src="svgs/02c3c2783a29871731d4e15e487956c0.svg" align="middle" width="14.764860000000004pt" height="30.359009999999977pt"/>的<img alt="$(i, j)$" src="svgs/e8873e227619b7a62ee7eb981ef1faea.svg" align="middle" width="33.46497000000001pt" height="24.65759999999998pt"/>分量的导数如上, 对应到矩阵形式则: 

<p align="center"><img alt="$$&#10;\frac{\partial L}{\partial \tilde{ {\bf Y} }} = 2 ( \tilde{ {\bf Y} } - {\bf Y}),&#10;$$" src="svgs/da54e52aafecba5d1fd2023bdbf20f7e.svg" align="middle" width="123.48698999999999pt" height="35.9073pt"/></p>

* 第二项: <img alt="$\frac{ \partial \tilde{ {\bf Y} } }{ \partial w_ {n, m} }$" src="svgs/29b0ddf1dc7aa8f2c5293f0897c279dc.svg" align="middle" width="39.15945pt" height="34.30548pt"/>

这一项是矩阵对标量求导, 同样地, 先按照标量对标量求导处理: 

<p align="center"><img alt="$$&#10;\begin{aligned}&#10;\frac{ \partial \tilde{ y }_ {i, j}  }{ \partial w_ {n, m} } &amp;= \frac{\partial }{ \partial w_ {n, m} } \left[ \sum_ {k=1}^N  x_{i, k} w_{k, j}  \right] \\&#10;&amp;= \frac{\partial }{ \partial w_ {n, m} } \left[ x_ {i, 1} w_ {1, j} + \ldots + x_ {i, n} w_ {n, j} + \ldots + x_ {i, N} w_ {N, j} \right] \\&#10;&amp;= \begin{cases}&#10;&#9;0, &amp; j \ne m \\&#10;&#9;x_ {i, n}, &amp; j = m&#10;   \end{cases}&#10;\end{aligned},&#10;$$" src="svgs/a9901925a557272b2214d976175397a5.svg" align="middle" width="430.90079999999995pt" height="151.16442pt"/></p>

根据以上结果可以扩展到矩阵形式的梯度如下:  

<p align="center"><img alt="$$&#10;\frac{ \partial \tilde{ {\bf Y} } }{ \partial w_ {n, m} } = \left[ &#10;\begin{array}{ccccc}&#10;0 &amp; \ldots &amp; x_{1, n} &amp; \ldots &amp; 0\\&#10;0 &amp; \ldots &amp; x_{2, n} &amp; \ldots &amp; 0\\&#10;\vdots &amp; \ddots &amp; \vdots &amp; \ddots &amp; \vdots\\&#10;0 &amp; \ldots &amp; x_{B, n} &amp; \ldots &amp; 0\\&#10;\end{array}&#10;\right]_ {B\times M},&#10;$$" src="svgs/faa7a16f4aa4dd1245e20bc23b866372.svg" align="middle" width="304.12305pt" height="91.49811pt"/></p>

其中非零列<img alt="$[x_{1, n}, x_{2, n}, \ldots, x_{B, n} ]^\intercal$" src="svgs/b9084340443a56119f6d9e734acb9022.svg" align="middle" width="150.74961000000002pt" height="24.65759999999998pt"/>位于矩阵的第<img alt="$m$" src="svgs/0e51a2dede42189d77627c4d742822c3.svg" align="middle" width="14.433210000000003pt" height="14.155350000000013pt"/>列。  
将第一项和第二项的结果组合可以得到:  

<p align="center"><img alt="$$&#10;\begin{aligned}&#10;\frac{\partial L}{ \partial w_ {n, m} } &amp;= \sum_{i}\sum_{j} 2(\tilde{ {\bf Y} } - {\bf Y} ) \circ  \left[ &#10;\begin{array}{ccccc}&#10;0 &amp; \ldots &amp; x_{1, n} &amp; \ldots &amp; 0\\&#10;0 &amp; \ldots &amp; x_{2, n} &amp; \ldots &amp; 0\\&#10;\vdots &amp; \ddots &amp; \vdots &amp; \ddots &amp; \vdots\\&#10;0 &amp; \ldots &amp; x_{B, n} &amp; \ldots &amp; 0\\&#10;\end{array}&#10;\right]_ {B\times M} \\&#10;&amp;= 2 \sum_{i=1}^B x_{i, n} ( \tilde{y}_ {i, m} - y_{i, m} )\\&#10;&amp;= {\bf X}_ n^\intercal \cdot 2( \tilde{ {\bf Y} } - {\bf Y} )_ m, &#10;\end{aligned}&#10;$$" src="svgs/1d767598ce7ee8a9631346b30062eb85.svg" align="middle" width="435.11324999999994pt" height="175.03199999999998pt"/></p>

其中<img alt="$\circ$" src="svgs/c0463eeb4772bfde779c20d52901d01b.svg" align="middle" width="8.219277000000005pt" height="14.61206999999998pt"/>表示Hadamard积, 即两维度相同的矩阵对应项相乘。<img alt="${\bf X}_ n$" src="svgs/235a0996a628b61cd948d6658effe0af.svg" align="middle" width="22.418220000000005pt" height="22.557149999999986pt"/>与<img alt="$( \tilde{ {\bf Y} } - {\bf Y} )_ m$" src="svgs/1c6ff8d82649513e5a4ca630d5f421bf.svg" align="middle" width="74.07097499999999pt" height="30.359009999999977pt"/>分别表示矩阵<img alt="$X$" src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg" align="middle" width="14.908740000000003pt" height="22.46574pt"/>的第<img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align="middle" width="9.867000000000003pt" height="14.155350000000013pt"/>列和<img alt="$( \tilde{ {\bf Y} } - {\bf Y} )$" src="svgs/b1da593c5026154d7dfdfefd24becd89.svg" align="middle" width="62.406135pt" height="30.359009999999977pt"/>的第<img alt="$m$" src="svgs/0e51a2dede42189d77627c4d742822c3.svg" align="middle" width="14.433210000000003pt" height="14.155350000000013pt"/>列。  
综合以上, 将结果矩阵化可以得到<img alt="$\frac{\partial L}{ \partial {\bf W} }$" src="svgs/0fe6da93d2f7c1971887fa42380c616b.svg" align="middle" width="23.155934999999996pt" height="28.926479999999973pt"/>表示如下:  

<p align="center"><img alt="$$&#10;\begin{aligned}&#10;\frac{\partial L}{ \partial {\bf W} } &amp;= {\bf X}^\intercal \cdot 2 (\tilde{ {\bf Y} } - {\bf Y}) \\&#10;&amp;= {\bf X}^\intercal \frac{\partial L}{\partial \tilde{ {\bf Y} }  }.&#10;\end{aligned}&#10;$$" src="svgs/347ab64a1a2000cfe55ab0e0d42751a4.svg" align="middle" width="158.39076pt" height="76.29467999999999pt"/></p>

**小结**: 对形如<img alt="${\bf C} = {\bf A}\cdot {\bf B}, f({\bf C})$" src="svgs/5b562cb703d07198b352856d820807a2.svg" align="middle" width="118.743735pt" height="24.65759999999998pt"/>标量函数求梯度<img alt="$\partial f / \partial {\bf B}$" src="svgs/a71387d1a3c9803937eb0e4da6bc3e4f.svg" align="middle" width="50.76489pt" height="24.65759999999998pt"/>的结果为:  

<p align="center"><img alt="$$&#10;\frac{\partial f} {\partial {\bf B} } = {\bf A}^\intercal \frac{\partial f}{ \partial {\bf C} }.&#10;$$" src="svgs/a0c84bf34cef695a847bb670ddca797c.svg" align="middle" width="101.339865pt" height="33.812129999999996pt"/></p>

## 参考

[1]. [Vector, Matrix, and Tensor Derivatives](http://cs231n.stanford.edu/vecDerivs.pdf)  
[2]. [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/)