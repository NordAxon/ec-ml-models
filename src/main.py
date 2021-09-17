from fastapi import FastAPI, HTTPException, UploadFile, File
import models
from enum import Enum
from PIL import Image
from io import BytesIO
import numpy as np
from pathlib import Path
import uvicorn
from starlette.responses import RedirectResponse
from pydantic import BaseModel

from utils import read_imagefile

class ModelName(str, Enum):
    question_answering = "question_answering"
    text_generation = "text_generator"
    sentiment_analysis = "sentiment_analysis"
    image_classifier = "image_classifier"

class ModelChoice(BaseModel):
    name: ModelName

class QuestionAnswering(BaseModel):
    context: str
    question: str

class TextContext(BaseModel):
    context: str

app = FastAPI()
global model

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/start/", status_code=200)
def start_model(chosen_model: ModelChoice):
    global model
    if chosen_model.name == ModelName.question_answering:
        model = models.QA()
    elif chosen_model.name == ModelName.text_generation:
        model = models.TextGenerator()
    elif chosen_model.name == ModelName.sentiment_analysis:
        model = models.SentimentAnalyser()
    elif chosen_model.name == ModelName.image_classifier:
        model = models.ImageClassifier()
    else:
        raise HTTPException(status_code=500, detail="Model name not correct, please revise.")


@app.post("/qa/")
def qa_pipeline(question_answering: QuestionAnswering):
    try:
        response = model.answer_question(question_answering.question, question_answering.context)
        answer = response['answer']
        score = response['score']
        return {'answer':answer, 'score':score}
    except NameError:
        raise HTTPException(status_code=500, detail="Model not working - did you forget to start the model?")


@app.post("/text_generation/")
def text_generation(text_gen: TextContext):
    try:
        response = model.generate_text(text_gen.context)[0]
        generated_text = response["generated_text"]
        return {'generated_text': generated_text}
    except NameError:
        raise HTTPException(status_code=500, detail="Model not working - did you forget to start the model?")


@app.post("/sentiment_analysis/")
def sentiment_analysis(text: TextContext):
    try:
        response = model.analyse_text(text.context)[0]
        label = response["label"]
        score = response["score"]
        return {'sentiment_label': label, 'score': score}
    except NameError:
        raise HTTPException(status_code=500, detail="Model not working - did you forget to start the model?")


@app.post("/classify_image/")
async def classify_image(file: UploadFile = File(...)):
    extension_is_correct = Path(file.filename).suffix in (".jpg", ".jpeg", ".png")
    if extension_is_correct:
        try:
            file_contents = await file.read()
            image = read_imagefile(file_contents)
            response = model.classify(image)
            return {key: str(value) for key, value in response.items()}
        except NameError:
            raise HTTPException(status_code=500, detail="Model not working - did you forget to start the model?")
    else:
        raise HTTPException(status_code=500, detail="Image must be jpg or png format")


if __name__ == "__main__":
    uvicorn.run(app, debug=True)