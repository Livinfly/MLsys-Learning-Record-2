# vLLM 源码学习

>   v0

-   LLMEngine 处理请求前（实例化阶段），会跑一次模拟实验来估计GPU的显存分配预留多少给KV Cache block。
-   LLMEngine 开始处理请求时(add_request)，它会把每个prompt当成一个请求，同时把它包装成一个SequenceGroup对象。
-   在1次推理中，所有seq_group要么一起做prefill，要么一起做decode。
-   调度器中只是给出了物理块的分配方案，并没有实际往物理块中添加数据，添加数据这一步是CacheEngine照着这个方案来实际操作的

>   v1



## 参考资料

[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)

[图解大模型计算加速系列：vLLM源码解析1，整体架构 - 知乎](https://zhuanlan.zhihu.com/p/691045737)

[图解大模型计算加速系列：vLLM源码解析2，调度器策略(Scheduler) - 知乎](https://zhuanlan.zhihu.com/p/692540949)

[图解大模型计算加速系列：vLLM源码解析3，块管理器BlockManager（上篇） - 知乎](https://zhuanlan.zhihu.com/p/700780161)

