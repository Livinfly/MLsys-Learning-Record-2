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

    

## 方法





