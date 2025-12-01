
【作业1】对多模态RAG的项目，设计下接口，需要满足：数据管理的接口、多模态检索接口、多模态问答接口
    接口定义定义清楚，传入的参数 + 返回的结果格式
答：数据管理的接口:
    文档管理接口:
        mysql/sqlite数据库存储数据元信息:
        文档类型：markdown,pdf,word,excel
        上传文档接口（upload_document）：
            Args:上传文件
            returns:
                request_id=str(uuid.uuid4()),
                document_id=document_id,
                category=category,
                title=title,
                knowledge_id=knowledge_id,
                file_type=file.content_type,
                response_code=200,
                response_msg="文档添加成功",#状态
                process_status="completed",
                processing_time=time.time() - start_time
        删除文档接口（delete_document）：
            Args:document_id
            returns:
                request_id=str(uuid.uuid4()),
                document_id=document_id,
                knowledge_id=record.knowledge_id,
                category=record.category,
                title=record.title,
                file_type=record.file_type,
                response_code=200,
                response_msg="文档删除成功",#状态
                process_status="completed",
                processing_time=time.time() - start_time
        查询文档接口（query_document）：
            Args:document_title
            returns:
                request_id=str(uuid.uuid4()),
                document_id=knowledge_id,
                category="",
                title="",
                response_code=404,
                response_msg="文档不存在",
                process_status="completed",
                processing_time=time.time() - start_time
    知识库管理接口:
        es/milvus储存embedding向量
        创建知识库接口(create_knowledge_base):
            Args:knowledge_id
            returns:参考文档接口
        删除知识库接口(delete_knowledge_base):
            Args:knowledge_id
            returns:参考文档接口
        修改/更新知识库接口(update_knowledge_base):
            Args:knowledge_id
            returns:参考文档接口
        查询知识库接口(query_knowledge_base):
            Args:title/category/...
            returns:参考文档接口


多模态检索接口:
    输入文字检索(text_query):
        Args:输入文本
        returns:检索结果列表
    上传图片检索(image_query)
        Args:图片文件
        returns:检索结果列表
    文字+图片联合检索(text_image_query)
        Args:图片文件，输入文本
        returns:检索结果列表


多模态问答接口:
    输入文字检索(text_query)
        Args:输入文本
        returns:检索文档id列表,大模型回答
    上传图片检索(image_query)
        Args:图片文件
        returns:检索文档id列表,大模型回答
    文字+图片联合检索(text_image_query)
        Args:图片文件，输入文本
        returns:检索文档id列表,大模型回答



