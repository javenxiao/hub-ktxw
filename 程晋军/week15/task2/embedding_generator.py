import torch
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
from config import EMBEDDING_CONFIG

class EmbeddingGenerator:
    def __init__(self):
        self.use_api = EMBEDDING_CONFIG["use_api"]
        if self.use_api:
            # OpenAI API初始化
            self.client = OpenAI(
                api_key=EMBEDDING_CONFIG["openai_api_key"],
                base_url=EMBEDDING_CONFIG["base_url"]
            )
            self.embedding_model = EMBEDDING_CONFIG["openai_embedding_model"]
        else:
            # 本地模型逻辑不变
            self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_CONFIG["local_model_name"])
            self.model = AutoModel.from_pretrained(EMBEDDING_CONFIG["local_model_name"])
            self.model.eval()

    def get_embedding(self, text: str) -> list[float]:
        if self.use_api:
            # OpenAI API调用
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        else:
            # 本地模型逻辑不变
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy().tolist()

    def batch_get_embedding(self, texts: list[str]) -> list[list[float]]:
        return [self.get_embedding(text) for text in texts]

if __name__ == "__main__":
    embedding_generator = EmbeddingGenerator()
    text = "这是一个测试文本"
    embedding = embedding_generator.get_embedding(text)
    print(embedding)