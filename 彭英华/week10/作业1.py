from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.chat_models import init_chat_model
import os
os.environ["OPENAI_API_KEY"] = "sk-90aa2b7df82745f3a46373cc0ddd0497"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = init_chat_model(model_provider="openai",model="qwen3-vl-plus")
messages = [{"role":"system","content":"你是一个图像分类的专家，你需要根据用户输入的图片判断它是什么动物,不需要多余的输出"},
            {"role":"user","content":[{"type":"text","text":"帮我识别输入的图片是{justfication}"},
                                      {"type":"image","image_url":"https://pic.nximg.cn/file/20230424/34062335_221258900107_2.jpg"}]}]
prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model
response = chain.invoke({"justfication":"cat or dog"})
print(response.content)
