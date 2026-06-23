## 一个高精度、端到端的文档公式智能问答系统（Formula-QAS）

### 思路1 
1. 由于$\LaTeX$可以很好表示公式，首先将图片|文字的公式形式转化话结构化的$\LaTeX$公式
2. 建立用户提问与$\LaTeX$公式的关联，使用RAG检索二者的相似度，得到topk待选公式
3. 使用thinking大模型，输入用户提问和检索结果，生成可执行python代码
4. 返回 代码执行结果 或 大模型推理结果

### 思路2
1. 由于$\LaTeX$可以很好表示公式，首先将图片|文字的公式形式转化话结构化的$\LaTeX$公式
2. 用精确的python代码实现公式计算，并搭建mcp服务
3. 建立用户提问与$\LaTeX$公式的关联，使用RAG检索二者的相似度，得到topk待选公式 和 用户输入中的公式参数
4. 待选公式填入tool_filter，调用对应tool，返回结果

### 备选方案
#### 备选方案1
1. langchain提示词模板+云大模型

  ```sequence
  title: langchain 大模型应用
  数据文件->待选公式: pd读取
  用户输入->待选公式: pd读取，相似度计算，top-k
  note over 用户输入,待选公式: 基于Embedding模型的文本编码
  用户输入->Prompt: 填入
  待选公式->Prompt: 填入
  Prompt->messages: PromptTemplate().format_prompt
  messages->返回结果: ChatOpenAI().invoke
  ```

2. HF transformers库，加载本地模型

   ```sequence
   title:langchain+transformers 大模型应用
   数据文件->待选公式:pd读取
   待选公式->Prompt模板: 填入
   用户输入->Prompt模板:填入
   Prompt模板->Qwen对话模板: tokenizer.apply_chat_template()
   Qwen对话模板->模型输入:tokenizer()
   模型输入->模型输出:AutoModelForCausalLM().generate()
   模型输出->放回结果:tokenizer.decode
   ```

3. 文本提取+工作流编排

   - **文本提取**

   ```mermaid
   graph LR
   F[文本提取]
   A(zip压缩文件) -->|ZipFile| B{文件类型}
   B -->|文件类型为.pdf| C(fitz.open.get_text)
   B -->|文件类型为.md| D(.decode'utf-8')
   ```
   - **提示词设计**

   ```mermaid
   graph TD
   公式匹配--> 意图识别 
   意图识别--> 槽位填充
   槽位填充--> 运行前检查
   ```
   - **工作流编排**

   ```mermaid
   graph LR
   title[工作流编排]
   a(语料库)-->|SBert|b(语料句向量)
   c(用户输入)-->|SBert|d(输入句向量)
   b-->|top-k|e(余弦相似度)
   d-->e
   c-->|填入|f(提示词模板)
   e-->|填入|f
   f-->g[ollama]
   g-->h{格式化输出}
   h-->|Y|j(JSON)
   h-->|N|k(正则抢救)
   h-->|ERROR|异常处理
   
   ```
