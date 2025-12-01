from langchain_openai import OpenAI
import base64
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage
from langchain.chat_models import init_chat_model
import os


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

os.environ["OPENAI_API_KEY"] = "sk-41357d21198744d38f05c076348fab3a"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = init_chat_model(model_provider="openai",model="qwen3-vl-plus")
messages = [{"role":"system","content":"你是一个图像分类的专家，你需要根据用户输入的图片判断它是什么动物,不需要多余的输出"},
            {"role":"user","content":[{"type":"text","text":"帮我识别输入的图片是{justfication}"},
                                      {"type":"image","image_url":f"data:image/jpeg;base64,"+encode_image("img/cat.jpg")+""}]}]
prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model
response = chain.invoke({"justfication":"cat or dog"})
print(response.content)