#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
 Author: Marky
 Time: 2025/11/16 20:26 
 Description:

作业3: 加载中文的clip模型，只要cpu推理，跑完 01-CLIP模型.ipynb
https://www.modelscope.cn/models/AI-ModelScope/chinese-clip-vit-base-patch16
可以不用原始数据，任意10个图 10个文本，完成图文匹配。

"""

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from tqdm import tqdm_notebook, tqdm
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import transformers
print(f"transformers 版本: {transformers.__version__}")  #  4.57.1

# 加载模型
model = ChineseCLIPModel.from_pretrained("../../../models/chinese-clip-vit-base-patch16")  # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("../../../models/chinese-clip-vit-base-patch16")  # 预处理
model.to("cpu")
# 图像编码
img_image_feat = []

batch_size = 10
img_paths = [
    '0a3f3df75c48c9c54be7ecdaa4ff9b3fca07d09f.jpg',
    '0a4d22bf6639b91acafa23466d121d7236975c6c.jpg',
    '0a4dbf817ecd96700998fafdaa6a4a5d030fcbe8.jpg',
    '0a4ea2d81dccbd87c26d26a41f3f9296caa8856f.jpg',
    '0a04cc16cadcf2a686d216fd162da2ede681f3d5.jpg',
    '0a5a04a3b1e4c6840b19e825b9a06f9f51825c7d.jpg',
    '0a5a567a20f1a08cfc17a5f9af56cd6b7e96f51d.jpg',
    '0a5c60ed13f119b1133392b144633a1fc9500f99.jpg',
    '0a5eb1740ffab396e799c46242d5a24f81c80a2f.jpg',
    '0a6a5bcc8e09b685428a9f12b6fd7acb6b3a87e1.jpg',
    '0a6cb526ac4fc835f2bde94cff56d568c0158fba.jpg',
    'image.jpg'
]

for idx in tqdm_notebook(range(len(img_paths) // batch_size + 1)):
    imgs = [Image.open(path) for path in img_paths[idx * batch_size: (idx + 1) * batch_size]]

    if len(imgs) == 0:
        break

    inputs = processor(images=imgs, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features.data.numpy()
        img_image_feat.append(image_features)

img_image_feat = np.vstack(img_image_feat)
img_image_feat = normalize(img_image_feat)
print(img_image_feat.shape)

# 文字编码
img_texts_feat = []

img_captions = [
    '一位穿着白色裙子的女生手里拿着一顶帽子',
    '有一个男人拿着话筒在唱歌，很投入，灯光效果很鲜艳',
    '一个穿着红色图案运动上衣的男生很开心，脚旁边有一个足球',
    '两个老奶奶在折东西，折了一大箩筐',
    '一位穿着红色衣服的中年妇女在大棚里看蔬菜',
    '一位帅哥和一位美女坐在桌子前面，桌子上面摆放着一瓶红酒，美女含情脉脉的看着帅哥',
    '一个穿着白色运动服的职业人员正在踢足球',
    '一群穿着西装的人在开会，文档都扔到空中了，看来气氛很差',
    '穿着一整套黑色衣服的美女，头在也戴着一副黑色眼睛，看起来很开心',
    '一个中年男士拿着平板在办公',
    '一位穿着粉色衣服的男生和一个美女在跳舞'
   # '有个男生很努力的电脑旁边工作'
]

batch_size = 10
for idx in range(len(img_captions) // batch_size + 1):
    texts = [text for text in img_captions[idx * batch_size: (idx + 1) * batch_size]]

    if len(texts) == 0:
        break

    inputs = processor(text=texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features.data.numpy()
        img_texts_feat.append(text_features)

img_texts_feat = np.vstack(img_texts_feat)
img_texts_feat = normalize(img_texts_feat)
print(img_texts_feat.shape)
#
#
# 相似度计算
query_idx = 10
sim_result = np.dot(img_texts_feat[query_idx], img_image_feat.T) # 矩阵计算
sim_idx = sim_result.argsort()[::-1][1:4]


print('输入文本: ', img_captions[query_idx])
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(Image.open(img_paths[sim_idx[0]]))
plt.xticks([]); plt.yticks([])

plt.subplot(132)
plt.imshow(Image.open(img_paths[sim_idx[1]]))
plt.xticks([]); plt.yticks([])

plt.subplot(133)
plt.imshow(Image.open(img_paths[sim_idx[2]]))
plt.xticks([]); plt.yticks([])
