# RLJSP
Reinforcement learning in Job Scheduling Problem (JSP)


## Materials
- [A DEEP REINFORCEMENT LEARNING BASED SOLUTION FOR FLEXIBLE JOB SHOP SCHEDULING PROBLEM](http://www.ijsimm.com/Full_Papers/Fulltext2021/text20-2_CO7.pdf)
- [Reinforcement Learning for Solving the Vehicle Routing Problem](https://arxiv.org/pdf/1802.04240.pdf)
- [Research on Adaptive Job Shop Scheduling Problems Based on Dueling Double DQN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9218934)


## Problem formulation



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
