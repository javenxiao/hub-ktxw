import openai
import json

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    api_key="sk-2525a42dd******9288574f", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://api.deepseek.com",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
)
print(completion.model_dump_json())


#
# {"id":"6d069f50-352e-453c-bfd8-9cfbff9e11ba","choices":
#     [{"finish_reason":"stop","index":0,"logprobs":null,
#       "message":{"content":"我是DeepSeek-V3，由深度求索公司创造的智能助手！😊 我的目标是帮助你解答问题、提供信息、陪你聊天，或者协助完成各种任务"
#                            "。如果你有任何问题或需要帮助，随时告诉我哦！","refusal":null,"role":"assistant","annotations":null,"audio":null
#           ,"function_call":null,"tool_calls":null}}],"created":1757516444,"model":"deepseek-chat","object":"chat.completion"
#     ,"service_tier":null,"system_fingerprint":"fp_08f168e49b_prod0820_fp8_kvcache",
#  "usage":{"completion_tokens":48,"prompt_tokens":12,"total_tokens":60,"completion_tokens_details":null,
#           "prompt_tokens_details":{"audio_tokens":null,"cached_tokens":0},"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":12}}
