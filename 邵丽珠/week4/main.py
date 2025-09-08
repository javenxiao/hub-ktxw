import time
import traceback

from fastapi import FastAPI

from model.bert import model_for_bert

app = FastAPI()

@app.get("/train/bert")
def train_by_bert():
    model_for_bert()
