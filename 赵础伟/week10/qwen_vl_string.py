from langchain.chat_models import init_chat_model
import os
os.environ["OPENAI_API_KEY"] = "sk-41357d21198744d38f05c076348fab3a"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = init_chat_model(model="qwen3-vl-plus",model_provider="openai",
                        extra_body={
                            'enable_thinking': True,
                            "thinking_budget": 81920}
                        )
messages = [{"role":"system","content":"你是一个信息抽取专家，你需要根据用户所给的图片从中抽取文字，并解析为文本"},
            {"role":"user","content":[{"type":"text","text":"帮我从图片中抽取文字，并解析为文本"},
                                      {"type":"image_url","image_url":"https://c-ssl.duitang.com/uploads/item/201807/11/20180711231551_vwkti.jpeg"}]}]
response = model.invoke(messages)
print(response.content)