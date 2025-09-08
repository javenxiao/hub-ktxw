import os.path
import random
import json
import pandas as pd

from config import Config
from transdata import TextClassifyRequest

data_file = os.path.join("..", Config["data_path"])

df = pd.read_csv(data_file, sep=",", encoding="utf-8", header=0)
text = df["review"].sample(n=60, random_state=random.randint(1, 1000)).to_list()
# print(text)

content = ""
for i, t in enumerate(text):
    tcr = TextClassifyRequest(request_id="9999", request_text=t)
    content += f"{tcr.__dict__}\n"

with open("test.json", "w", encoding="utf-8") as f:
    f.write(content)