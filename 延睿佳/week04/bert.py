from typing import Union, List

import torch
from transformers import AutoTokenizer, BertForSequenceClassification

from config import BERT_MODEL_PERTRAINED_PATH, BERT_MODEL_PKL_PATH

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 类别名称：0=差评，1=好评
CATEGORY_NAME = ["差评", "好评"]

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)
model = BertForSequenceClassification.from_pretrained(
    BERT_MODEL_PERTRAINED_PATH, num_labels=2
)

# 加载训练好的权重
model.load_state_dict(torch.load(BERT_MODEL_PKL_PATH, map_location=device))
model.to(device)
model.eval()


def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    """
    使用 fine-tuned BERT 模型对外卖评价文本进行分类
    输入可以是单个字符串，也可以是字符串列表
    输出是 "好评" 或 "差评"
    """

    single_input = False
    if isinstance(request_text, str):
        request_text = [request_text]
        single_input = True
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("输入必须是 str 或 List[str]")

    # 编码
    encoding = tokenizer(
        request_text,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    # 推理
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()

    results = [CATEGORY_NAME[x] for x in preds]

    return results[0] if single_input else results
