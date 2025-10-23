import json
from typing import Union, List

import openai

import config

client = openai.OpenAI(
    api_key=config.API_KEY,
    base_url=config.BASE_URL,
)


def classify_by_prompt(request_texts: Union[str, List[str]]) -> Union[dict, List[dict]]:
    if isinstance(request_texts, str):
        return classify(request_texts)

    classify_result = []
    for request_text in request_texts:
        result_dict = classify(request_text)
        classify_result.append(result_dict)

    return classify_result


def classify(request_text: str) -> dict:
    completion = client.chat.completions.create(
        model=config.MODEL,
        messages=[
            {"role": "system", "content": f"""你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
            - 待选的领域类别：{" / ".join(config.DOMAINS)}
            - 待选的意图类别：{" / ".join(config.INTENTS)}
            - 待选的实体标签：{" / ".join(config.SLOTS)}
            最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。
            {{
                "domain": ,
                "intent": ,
                "slots": {{
                    "待选实体": "实体名词",
                }}
            }}
            """},
            {"role": "user", "content": request_text},

        ],
    )
    result = completion.choices[0].message.content
    result_dict = {'text': request_text}
    try:
        result_dict.update(json.loads(result))
    except json.JSONDecodeError:
        print("ERROR:", result)
        result_dict['error_msg'] = '识别失败'
    return result_dict


if __name__ == "__main__":
    texts = "帮我查询下桂林飞南京的航班"
    classify_result = classify_by_prompt(texts)
    print(json.dumps(classify_result, ensure_ascii=False, indent=4))
