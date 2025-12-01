#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 Author: Marky
 Time: 2025/11/16 19:51 
 Description:
 作业2：使用云端qwen-vl模型，完成带文字截图的图，文本的解析转换为文本。
"""


import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../../../models/Qwen/Qwen2.5-VL-3B-Instruct/",
    torch_dtype=torch.float32,
    device_map="cpu"
)

processor = AutoProcessor.from_pretrained(
    "../../../models/Qwen/Qwen2.5-VL-3B-Instruct/"
)

image_path = "./img_3.png"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {"type": "text", "text":"识别图中的文字,并输出"}
        ]
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print(output_text)


# 结果：
# Loading checkpoint shards: 100%|██████████| 2/2 [00:16<00:00,  8.42s/it]
# ['用心做人、用爱做事\n\n仁 [英 悲 宽容]\n义 [公正 正义]\n礼 [礼貌 尊重]\n信 [诚实 守信]\n忠 [尽责 坚持]\n孝 [孝顺 感恩]\n廉 [节制 自律]\n耻 [知耻 自省]\n勤 [勤奋 进取]\n俭 [俭朴 惜物]\n勇 [不惧 承担]\n慧 [智慧 聪慧]']


