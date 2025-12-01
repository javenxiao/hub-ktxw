from PIL import Image
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import glob

img_paths = glob.glob('../data/images/*.jpg')
img_paths.sort()

img_paths = img_paths[:3000]

str="一只冲浪的猫被鲨鱼追逐,一个人在飞,游乐场里的大摆锤,一只带墨镜的鸭子,风车美景,摩天轮,放飞气球放飞梦想,游乐园里旋转秋千,一个可爱的小猪装饰品,为世界杯歌唱"

img_texts = str.split(',')

def img_text_map(img_paths, img_texts):
    #加载模型
    model = ChineseCLIPModel.from_pretrained("../models/chinese-clip-vit-base-patch16") # 中文clip模型
    processor = ChineseCLIPProcessor.from_pretrained("../models/chinese-clip-vit-base-patch16") # 预处理

    imgs = [Image.open(path) for path in img_paths]
    # 处理输入
    inputs = processor(text=img_texts, images=imgs, return_tensors="pt", padding=True)

    # 模型推理
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取特征向量
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    # 归一化特征向量
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(image_embeds, text_embeds.t())
    return similarity_matrix, imgs, img_texts

def getSimilarity(similarity_matrix, image_paths, img_texts):
    for i, img_path in enumerate(image_paths):
        img_name = img_path.split('\\')[-1]
        best_match_idx = torch.argmax(similarity_matrix[i]).item()
        best_similarity = similarity_matrix[i][best_match_idx].item()
        print(f"图像:{img_name}-->文本:{img_texts[best_match_idx]} (相似度: {best_similarity:.4f})")

similarity_matrix, imgs, img_texts = img_text_map(img_paths, img_texts)
getSimilarity(similarity_matrix, img_paths, img_texts)