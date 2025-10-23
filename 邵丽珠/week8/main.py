from fastapi import FastAPI
app = FastAPI(
    response_model=None
)

from model.workByPrompt import query
from model.workBytools import queryBytools
from data_schema import TextResponse

@app.post("/queryByPrompt")
def queryByPrompt(req: str) -> str:
    return query(req)


@app.post("/queryByTools")
def queryByTools(req: str) -> TextResponse:
    return queryBytools(req)
