**DeepWalk原理**
DeepWalk本质上是一个图嵌入（Graph Embedding）方法，它的目标是：
把图里的节点映射成向量（embedding），并让“结构上相似的节点”在向量空间中靠得很近。
1. 随机游走 (Random Walks)：在图上走“随机游走路径”，得到一条条节点序列。
2. 训练词向量 (Word2Vec)：把这些序列丢给 Word2Vec（通常用 Skip-gram），让模型学习“一个节点的上下文是谁”。
3. 相似度计算：嵌入向量出来以后，可以做聚类、可视化、推荐、社区发现等任务。

**DeepWalk与Word2Vec的关系**
DeepWalk：把 图结构 转换成“伪文本语料”（节点序列），然后交给 Word2Vec 来训练节点向量。
Word2Vec：把 文本语料（词序列）作为输入，学到词的向量表示。
DeepWalk = 随机游走(生成节点序列) + Word2Vec(学习节点向量)，DeepWalk本质上是Word2Vec在图上的应用。
