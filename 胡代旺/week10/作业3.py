import glob, json, os
from PIL import Image
from tqdm import tqdm_notebook
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

img_paths = glob.glob('./imgs/*.jpg')

texts = pd.read_csv("./texts/dataset.csv", header=None)
texts = texts[0]

# print(img_paths)
# print(texts)
model = ChineseCLIPModel.from_pretrained("./model/chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("./model/chinese-clip-vit-base-patch16") # 预处理


img_image_feat = []
for img in img_paths:
    imgs = Image.open(img)
    inputs = processor(images=imgs, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    image_features = image_features.data.numpy()
    img_image_feat.append(image_features)

img_image_feat = np.vstack(img_image_feat)
img_image_feat = normalize(img_image_feat)

img_texts_feat = []
for text in texts:
    text = [text]
    inputs = processor(text=text, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features.data.numpy()
        img_texts_feat.append(text_features)

img_texts_feat = np.vstack(img_texts_feat)
img_texts_feat = normalize(img_texts_feat)

#
sim_result = np.dot(img_image_feat, img_texts_feat.T) # 矩阵计算
sim_idx = sim_result.argsort()[::-1][:,0]

for idx, img in enumerate(img_paths):
    print(f'图片{img.split('\\')[1]}的描述是：{texts[sim_idx[idx]]}')
