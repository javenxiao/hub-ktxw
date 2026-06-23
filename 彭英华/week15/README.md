### DeepResearch.py是对原始DeepResearch.py代码进行改进后的版本，实现了以下两种改进。
- 不同的章节可以并发生成

- 生成的章节内容会通过agent进行审查，并且会通过返回的修改建议进行修改，直至满意为止

#### DeepResearch.md是通过运行DeepResearch.py生成的完整版本的以“Agentic AI在软件开发中的最新应用和挑战”为题的报告

#### dataprepare.py 是通过本地部署mineru将pdf文件解析为markdown文件后，然后对markdown文件进行解析，划分为chunk后存入本地的es中

#### rag.py 是通过调用agent对es检索结果进行rag问答
