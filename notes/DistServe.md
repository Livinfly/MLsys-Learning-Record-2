# DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving

[论文地址](https://arxiv.org/abs/2401.09670)

## 动机

LLM，prefill 阶段和 decoding 阶段的**资源分配**和**并行策略**耦合，导致训推效率都降低了。

两者会互相干扰。

虽然因为共享模型参数、KV cache，分离会带来通讯量的增加，但是实验证明是值得的。

## 贡献

-   识别并提出了 PD 解耦分离
-   设计了分配算法，自动选择 PD 资源分配
-   全面评估了提出的 DistServe 系统

## 相关

-   Continuous Batching, Chunked Prefill

    能够提高 GPU 利用率，最大化整个系统的吞吐量。变成 TTFT 和 TPOT 之间的 trade-off。
    
    具体地，Chunked Prefill 就是这样 trade-off 的一种方法，本质上，不能解决两者的干涉。

## 分析

由排队论的公式、计算瓶颈、内存瓶颈分析。

Prefill，intra-op

Decode，inter-op



实际部署中：

-   多样的 profill 长度，导致 pipeline bubbles

    设计一套算法，根据工作负载来找到能最小化 bubble 的并行策略

-   通信开销变大，显然 prefill 和 decode 的机器之间需要通信

## 方法

给定模型、负载特征、延迟要求，SLO 目标（percentage of requests that meet TTFT requirement），DistServe 会确定并行策略，计算实例数量的分配，在物理集群中怎么放置。（称为 placement），最大化 goodput 实际吞吐量。

遍历去获得最优配置。

在线调度上，请求先到中央控制器，再分配给**最短等待序列**的 prefill 实例，处理完成后，再选择**负载最小**的 decoding 实例。

根据 GPU 满载来倒推$L_m$，见效 pipeline bubbles；

对于业务中的**过载**情况，使用 pull 方法，而不是 push 方法，把 prefill 实例的内存作为 queuing buffer。
