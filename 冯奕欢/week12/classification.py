import numpy as np
import torch
from fastmcp import FastMCP
from typing import Annotated

from transformers import BertTokenizer, BertForSequenceClassification

mcp = FastMCP(
    name="Classification-MCP-Server",
    instructions="""This server contains some api of classification.""",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "../models/iic/nlp_structbert_emotion-classification_chinese-base"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

id2label = model.config.id2label
# print(id2label)
label2id = model.config.label2id
# print(label2id)

@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    print("input ---> ", text)
    if len(text.strip()) == 0:
        return "错误：输入文本为空"
    return predict(text)


def predict(text):
    """
    判断文本的情感类别
    :param text:
    :return:
    """
    input_feature = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=True
    )
    input_ids = input_feature["input_ids"].to(device)
    attention_mask = input_feature["attention_mask"].to(device)
    token_type_ids = input_feature["token_type_ids"].to(device)
    output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    logits = output.logits
    logits = logits.detach().cpu().numpy()
    index = np.argmax(logits, axis=-1)[0]
    if 0 <= index < len(id2label):
        print("classification ---> ", id2label[index])
        return f"**{text}** 属于 **{id2label[index]}** 类别"
    else:
        return "错误：识别错误"


# if __name__ == '__main__':
#     test_content = "新年快乐"
#     print(f"{test_content} ==> ", predict(test_content))
#     test_content = "哇，你怎么这么可爱"
#     print(f"{test_content} ==> ", predict(test_content))