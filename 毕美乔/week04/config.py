REGEX_RULE = {
    "FilmTele-Play": ["播放", "电视剧"],
    "HomeAppliance-Control": ["空调", "广播"]
}

# CATEGORY_NAME = [
#     'Travel-Query', 'Music-Play', 'FilmTele-Play', 'Video-Play',
#     'Radio-Listen', 'HomeAppliance-Control', 'Weather-Query',
#     'Alarm-Update', 'Calendar-Query', 'TVProgram-Play', 'Audio-Play',
#     'Other'
# ]

CATEGORY_NAME = ['Negative', 'Positive']

TFIDF_MODEL_PKL_PATH = "weights/tfidf_ml.pkl"

# BERT_MODEL_PKL_PATH = "weights/mac-bert.pt"
# BERT_MODEL_PERTRAINED_PATH = "/root/autodl-tmp/chinese-macbert-base/"
BERT_MODEL_PKL_PATH = "weights/bert_waimai_ckpt"
BERT_MODEL_PERTRAINED_PATH = "/home/bmq/.cache/huggingface"

LLM_OPENAI_SERVER_URL = f"http://127.0.0.1:30000/v1" # ollama
LLM_OPENAI_API_KEY = "None"
LLM_MODEL_NAME = "Qwen2.5-3B-Instruct/"
