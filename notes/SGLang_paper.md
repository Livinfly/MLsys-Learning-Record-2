# SGLang: Efficient Execution of Structured Language Model Programs

## 动机

-   编程复杂，有高级抽象库，但是性能瓶颈大；底层推理引擎又过于繁琐，难以编写复杂程序。
-   推理效率低下，需要控制权来回切换，开销大，特别是在大量小规模的复杂逻辑（多轮对话、工具使用）中。
-   KV cache 的高效利用、共享不充分。

## 困难

-   需要重新整合顶层的功能需求和底层，工程难度大。
-   需要设计新的 KV cache 存储逻辑。

## 贡献

1. SGLang 前端语言

   -   提供了 Python 接口，实际上设计了一种 DSL，把 Python 函数编译成**计算图**。
   -   使得 Python 的**控制流**能直接在 Runtime 运行时中执行，避免了切换，实现「一次启动，全程运行」

2. SGLang 运行时（SRT）

   - RadixAttention

     将所有请求的 KV cache 存储在全局的共享的 Radix Tree 基数树中，提高了缓存的利用率。

     且 PagedAttn 实际上是 RadixAttn 的退化成线性链表的特例。

     同时，采用 longest-shared-prefix-first order 的调度策略来得到更高的缓存命中率。

   - Structure Output

     可以强制按照**JSON**、**正则表达式**格式输出，把约束编译为一个压缩的**有限状态机**（Finite State Machine，FSM）

     在确定的部分，可以直接**跳过**，提升效率。

消融实验，证明吞吐量提高，多场景具有性能优势。

同时将 GPT-4 作为「编译器」，对计算图进行优化（把更可能成为前缀的部分，排到最前面），来增加可共享的前缀长度。

## 总结

SGLang，全新的 LLM Serving System，提供了更细腻 LLM DSL，同时，用 RadixAttn 和 FSM 提升 KV cache 利用率与结构化输出效率。

## 局限 / 未来可能方向

（论文中提到的）

增强前端语言的表达能力。现在牺牲 Python 的动态性了，需要按照 SGLang 设定的编程范式写。

目前是解释器，后续扩展到编译器，同时进行更深度的编译器优化。

与其他编程模型结合。

调度上会存在**饥饿**问题需要优化。

（我的想法）

安全问题，全局的数据结构共享，可能存在各种侧信道攻击的问题。

## 源码阅读

[SGLang](https://github.com/sgl-project/sglang)

[具体代码学习](SGLang.md)