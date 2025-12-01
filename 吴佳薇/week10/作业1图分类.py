import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 配置4位量化
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "F:/ai/tools/models/Qwen/Qwen2.5-VL-7B-Instruct/",
    quantization_config=quantization_config,
    device_map="auto"
)


# default: Load the model on the available device(s)
'''model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "F:/ai/tools/models/Qwen/Qwen2.5-VL-7B-Instruct/", dtype="auto", device_map="auto"
)'''
'''含义：从本地路径加载预训练的Qwen2.5-VL模型

from_pretrained()：从指定路径加载预训练模型

torch_dtype="auto"：自动选择合适的数据类型（float16/float32）

device_map="auto"：自动将模型分配到可用设备（GPU/CPU）'''


# default processer
processor = AutoProcessor.from_pretrained("F:/ai/tools/models/Qwen/Qwen2.5-VL-7B-Instruct/")#加载与模型配套的处理器，用于处理文本和图像输入

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://pic.nximg.cn/file/20230517/11949215_143004845103_2.jpg",
            },
            {"type": "text", "text": "is it dog or cat in the image"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
'''含义：将对话消息格式化为模型可接受的格式

tokenize=False：只格式化不进行分词

add_generation_prompt=True：添加生成提示符，告诉模型开始生成回复。
text 的内容可能是：
"<|im_start|>user\n<|image|>\nis it dog or cat in the picture<|im_end|>\n<|im_start|>assistant\n"
'''

image_inputs, video_inputs = process_vision_info(messages)
'''含义：从消息中提取并处理视觉信息（图像/视频）

image_inputs：处理后的图像张量

video_inputs：处理后的视频张量（本例中为空）'''

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
'''含义：使用处理器准备完整的模型输入

text=[text]：将格式化文本转为列表

images=image_inputs：添加图像输入

padding=True：自动填充到相同长度

return_tensors="pt"：返回PyTorch张量'''

inputs = inputs.to(device)#将输入数据放到GPU或CPU上运行

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=2)
'''含义：使用模型生成回复

**inputs：解包输入参数

max_new_tokens=2：限制生成2个新token（仅为示例，实际应设更大）

运行结果：generated_ids包含完整的输入+输出token IDs'''

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
'''含义：从完整输出中移除输入部分，只保留模型生成的部分

示例：

输入token数：50

输出总token数：52

修剪后：只取最后2个token（52-50=2）'''

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
'''含义：将token ID转换回可读文本

skip_special_tokens=True：跳过特殊token（如<|endoftext|>）

clean_up_tokenization_spaces=False：保留原始分词空格'''
print(output_text[0])
