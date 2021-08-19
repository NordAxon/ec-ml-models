from fastapi import FastAPI, HTTPException
import models
from enum import Enum

class ModelName(str, Enum):
    question_answering = "question_answering"
    text_generation = "text_generator"
    sentiment_analysis = "sentiment_analysis"


app = FastAPI()
global model

@app.get("/start/items/{chosen_model}", status_code=200)
def start_model(chosen_model: ModelName):
    global model
    if chosen_model == ModelName.question_answering:
        model = models.QA()
    if chosen_model == ModelName.text_generation:
        model = models.TextGenerator()
    if chosen_model == ModelName.sentiment_analysis:
        model = models.SentimentAnalyser()


@app.get("/qa/items/{question}/{context}")
def qa_pipeline(question: str, context: str, status_code=200):
    try:
        response = model.answer_question(question, context)
        answer = response['answer']
        score = response['score']
        return {'answer':answer, 'score':score}
    except NameError:
        raise HTTPException(status_code=500, detail="Model not working - did you forget to start the model?")