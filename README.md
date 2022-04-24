# RLJSP
Reinforcement learning in Job Scheduling Problem (JSP)


## Papers
- [A DEEP REINFORCEMENT LEARNING BASED SOLUTION FOR FLEXIBLE JOB SHOP SCHEDULING PROBLEM](http://www.ijsimm.com/Full_Papers/Fulltext2021/text20-2_CO7.pdf)
- [Reinforcement Learning for Solving the Vehicle Routing Problem](https://arxiv.org/pdf/1802.04240.pdf)
- [Research on Adaptive Job Shop Scheduling Problems Based on Dueling Double DQN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9218934)


## Problem formulation

## GitHub repo
- [JSP](https://github.com/prosysscience/RL-Job-Shop-Scheduling)
- [A Reinforcement Learning Environment For Job-Shop Scheduling](https://arxiv.org/pdf/2104.03760.pdf)
- [JSSP](https://github.com/mcfadd/Job_Shop_Schedule_Problem)
- [JobShopPRO](https://github.com/paulkastel/JobShopPRO)

## Method summary
### 整数规划常用方法
- 分枝定界法：可求纯或混合整数线性规划
- 割平面法：可求纯或混合整数线性规划
- 隐枚举法：用于求解0-1整数规划，有过滤法和分枝法
- 匈牙利法：解决指派问题（0-1规划特殊情形) 
- 蒙特卡罗法：求解各种类型规划
- [优化算法](https://keson96.github.io/categories/%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95/)
- [常见组合优化问题与求解方法简单介绍](https://zhuanlan.zhihu.com/p/161677525)

## Objectives

- What are the pros/cons of the RL method in solving the JSPs as opposed to classical algorithms?
- How is RL incorporated in solving JSPs?

## Reinforcement Learning for Solving the Vehicle Routing Problem
### Key points
- end-to-end framework (端到端框架)
- near-optical solution (近似最优解)
- parametrized stochastic policy (参数化随机策略)：直接优化用于做决策的策略（policy），假设策略是随机的，且服从一个参数化的策略分布。优点：针对连续场景，不会出现策略退化现象，其目标表达更直接 缺点：没有全局收敛性保障，但能收敛到局部最优。（比较Q-learning/off-policy，针对离散空间，假设policy是deterministic的，而且它的求解空间是函数空间，在时间无穷保证收敛到全局最优解）
- 基于Bello et. al. 2016, 可处理动态改变系统
- RNN解码器+注意机制（attention mechanism）
- 当输入量动态改变时，结果能自适应
- 比传统算法（基于启发式）有了极大提升：61%的案例中搜索到了更优解
- 和vanilla seq-2-seq相比，将编码器的隐状态通过加权，拼接到解码器的隐状态
- 和pointer network相比，当输入量动态改变时，整个网络不再需要被更新


## A DEEP REINFORCEMENT LEARNING BASED SOLUTION FOR FLEXIBLE JOB SHOP SCHEDULING PROBLEM

### Key points
- 端到端框架
- 3D析取图调度
- DRL
- RNN解码器+注意机制（attention mechanism）
- policy gradient （on-policy）
- 改进的pointer network
- 三大模块：scheduling environment(排程环境)、offline learning（线下学习）、online application（线上应用）
- 多job、多machine、动态分析

针对的是柔性作业车间调度问题,有NJ个工件，第i个工件有NOi个工序，NM个机器，对于工序Oih，都有一组机器可以加工该工序Mih，每一道工序的加工时间是给定的，每一台可加工机器同一时间只能处理一个工件。目标就是将工件的工序分配给机器，最小化完工时间。



使用指针网络

编码器根据所选的调度特征对要调度的操作进行编码

然后使用attention机制，在decode每一步，指向一个输入
使用策略梯度算法来优化参数
训练后的模型以连续动作序列的形式实时生成调度解，而无需对每个新问题实例进行重新训练。


启发式：及时安排，但无法保证调度结果最优。

元启发式：通常通过进化算子或粒子运动迭代搜索调度解，如遗传算法（GA）和粒子群优化（PSO）。虽然这些方法可以获得高质量的解，但由于迭代优化时间长，不能满足实时性要求；一旦问题结构发生变化，就需要重新设计通用性差的方法。该算法放弃了寻找最优解，而是试图在合理时间内找到近似可行解。

深度强化学习（DRL）将深度学习（DL）和RL结合起来，实现从感知到行动的端到端学习。

本篇论文的贡献是提出一个自适应调度框架，它结合了端到端DRL和3D析取图。该框架包括三个部分：调度环境、离线学习和在线应用。

调度环境采用三维析取图建模。基于析取图的调度解决方案是首先初始化就绪任务集，然后根据智能体的action将具有最高优先级的作业调度到具有最高优先级的机器。然后将作业从约束网络中移除，并将其后续任务添加到就绪任务集中。重复此过程，直到就绪任务集为空并获得调度结果。这样一个完整的过程叫做episode。

在离线学习阶段，将状态st输入到critic网络和actor网络中，分别输出baseline和action。策略梯度算法用于学习随机策略π，参数为θ。

虽然在学习阶段需要很长时间进行训练，但一旦学习到最优策略，在online application阶段，它可以应用于新的调度问题，并且可以在短时间内获得最优结果。


指针网络，encoder将输入编码得到feature vector，使用decoder并结合attention以自回归方式构造解决方案。

仅当输入传输顺序信息时，才需要RNN。但是对于FJSP这样的组合优化问题，任何操作的随机排列都包含与原始输入相同的信息，并且对最终的调度结果没有影响。所以他们就不用RNN编码器，而是直接用嵌入式输入替换RNN隐藏状态。使用embedding。

model：该模型包含两个主要组件。第一部分是嵌入集，它将输入映射到D维向量空间。第二个组件是解码器，它在每一步指向一个解码器的输入。这里，GRU RNN用于对解码器网络进行建模。

Fo是operation的特征向量，在每个解码步骤t，利用有glimpse的基于上下文context-based的注意机制，使用可变长度对齐向量at从输入中提取相关信息。一般来说，at指定每个输入数据点在下一解码步骤t中的相关性。Ct是context vector，计算嵌入之后的向量fto bar的加权的线性组合，权重就是ato，然后得到这个条件概率。选择一个输入作为下一个解码器的输入的动作会有一系列，从开始选到选完会得到一个序列pai，目标就是找到一个策略p，使得生成序列pai的方式是在满足约束的同时，最小化完工时间。

用RL优化指针网络的参数θ，使用policy gradient方法和随机梯度下降法stochastic gradient descent，使用REINFORCE算法计算梯度，对梯度进行Monte-Carlo采样。使用参数化的baseline来估计预期完工时间通常可以提高学习效率，因此使用critic学习用当前策略得到的预期完工时间来生成baseline。Stochastic gradient descent来训练critic，使用predicted value和实际完工时间的均方误差作目标。

实验：actor用指针网络，由于输入顺序不影响结果，使用了一维卷积层来代替encoder中的RNN，decoder使用GRU RNN。Critic有三个一维卷积层。

其中FJSP被描述为一个序列决策问题。通过改进的指针网络，每个操作都被编码到一个高维的嵌入向量中。通过注意机制，每一个解码步骤都会有一个输入作为动作。从总体、任务和机器的角度选择20个静态特征和24个动态特征，并通过特征组合扩展输入操作，以确保只需一个操作即可同时确定具有最高优先级的作业和机器。

与传统的RL方法相比，该方法本质上是在解空间而不是规则空间中搜索，提高了解的质量。与元启发式算法相比，该算法的优点在于通过离线训练，无需再训练即可在线求解不同规模的调度问题，具有较强的泛化能力和适应性。未来的研究将集中在以下几个方面。首先，我们将对每个超参数进行敏感性分析，以进一步提高溶解的质量。第二，我们将尝试使用更先进的策略梯度方法，如A3C、TRPO等，以增强现有的策略梯度方法。最后，将该方法应用于动态调度环境下的反应式调度。

