### RAG实现流程

#### 1.创建知识库

#### 2.创建知识文档

#### 3.解析知识文档

 - 读取文件，遍历每一页内容，进行bge模型embedding编码，保存到ElasticSearch中，并建立chunk_info索引
 - 每页内容还会分割成256字符大小的chunk，分别进行bge模型embedding编码，保存到ElasticSearch中，并建立chunk_info索引
 - 建立chunk_info和document_meta索引
 - 使用前3页内容作为文章摘要，也保存到ElasticSearch中，建立document_meta索引

#### 4.全文检索

 - 用户输入内容后，对消息的内容使用ElasticSearch对指定的知识库进行全文检索
 - 对输入的内容进行直接文字检索一次，得到结果1
 - 对输入的内容进行bge模型embedding编码，再进行语义检索一次，得到结果2
 - 对2次的结果计算各自的分数，按照分数排序放到一个列表
 - 将得到的结果和输入的内容，使用bge-rerank模型进行重排序
 - 得到消息相关的内容

#### 5.输入到大模型得到结果

 - 使用提示词工程
 - 将检索得到的消息相关内容，替换RELATED_DOCUMENT
 - 将用户输入的内容，替换QUESTION
 - 得到的提示词一起输入大模型得到结果
