API_KEY = 'sk-a2aa801ac63a4f0c949dc133a3614786'
BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
USED_MODEL = 'qwen-plus'

HOST = "127.0.0.1"
PORT = 8000

def load_txt(path: str) -> list[str]:
    with open(path, 'r', encoding='utf-8') as f:
        data = [i.strip() for i in f.readlines()]
    return data


DOMAINS = load_txt("./asset/domains.txt")
INTENTS = load_txt("./asset/intents.txt")
SLOTS = load_txt("./asset/slots.txt")
