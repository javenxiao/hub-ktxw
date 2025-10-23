import json

from openai import OpenAI


def get_tags(path):
    """
    获取类型、标签
    :param path: 文件名
    :return: 类型、标签字符串
    """
    tags = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tags.append(line.strip())
    return ' / '.join(tags).strip()


intents = get_tags('intents.txt')
# print(intents)
domains = get_tags('domains.txt')
# print(domains)
slots = get_tags('slots.txt')
# print(slots)

output = """
```json
{
    "domain": ,
    "intent": ,
    "slots": {
      "待选实体": "实体名词",
    }
}
```
"""

client = OpenAI(
    api_key='sk-965710e5b13541a493c47ffa1c8c4634',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)


def recognition(content):
    """
    意图识别、领域识别、实体识别
    :param content: 输入内容
    :return: 输出结果 json格式
    """
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {'role': 'system', 'content': f"""
    你是一个专业的意图识别、领域识别、实体识别的专家。
    根据用户输入的内容，抽取意图类型、领域类型、实体标签。
    输出结果只能是从以下待选的类型或者标签中选择：
    待选的意图类型：{intents}
    待选的领域类型：{domains}
    待选的实体标签：{slots}
    输出格式为json格式，domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签
    {output}
    """},
            {'role': 'user', 'content': content}
        ]
    )
    result = completion.choices[0].message.content
    results = result.split('\n')
    if len(results) > 2:
        del results[0]
        del results[-1]
    result = '\n'.join(results)
    try:
        data = json.loads(result)
        return json.dumps(data, ensure_ascii=False)
    except Exception as e:
        print(e)
    return result


# 测试
# if __name__ == '__main__':
#     result = recognition('糖醋鲤鱼怎么做啊？你只负责吃，c则c。')
#     print('识别结果：\n', result)
#     result = recognition('从台州到金华的汽车票。')
#     print('识别结果：\n', result)