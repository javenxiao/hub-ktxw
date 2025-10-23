## DeepWalk与Word2Vec的关系  
**DeepWalk：** 把 图结构 转换成“伪文本语料”（节点序列），然后交给 Word2Vec 来训练节点向量。
**Word2Vec：** 把 文本语料（词序列）作为输入，学到词的向量表示。  
1. DeepWalk 借助 Word2Vec 的能力，将图嵌入问题转化为一个类似 NLP 的问题  
2. DeepWalk 主要用于图嵌入，Word2Vec 主要用于词嵌入。
3. DeepWalk 是对 Word2Vec 的扩展：DeepWalk 将图嵌入问题转化为一个类似于 NLP 的问题，借助 Word2Vec 的能力来学习节点的嵌入。  
4. Word2Vec 是 DeepWalk 的核心组件：DeepWalk 的嵌入学习依赖于 Word2Vec 的训练过程。