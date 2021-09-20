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

class Image_Classes(BaseModel):
    class_1: str
    class_2: str
    class_3: str

app = FastAPI()
global model
global chosen_model_name

@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@app.post("/start/", status_code=200)
def start_model(chosen_model: ModelChoice):
    global model
    global chosen_model_name
    if chosen_model.name == ModelName.question_answering:
        model = models.QA()
        chosen_model_name = chosen_model.name
    elif chosen_model.name == ModelName.text_generation:
        model = models.TextGenerator()
        chosen_model_name = chosen_model.name
    elif chosen_model.name == ModelName.sentiment_analysis:
        model = models.SentimentAnalyser()
        chosen_model_name = chosen_model.name
    elif chosen_model.name == ModelName.image_classifier:
        model = models.ImageClassifier()
        chosen_model_name = chosen_model.name
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
    try:
        file_contents = await file.read()
        image = read_imagefile(file_contents)
        response = model.classify(image)
        return {key: str(value) for key, value in response.items()}
    except NameError:
        raise HTTPException(status_code=500, detail="Model not working - did you forget to start the model?")
    
@app.put("/change_classes/")
def change_model_classes(new_classes: Image_Classes):
    global model
    global chosen_model_name
    if chosen_model_name == ModelName.image_classifier:
        model = models.ImageClassifier(
            labels = [new_classes.class_1, new_classes.class_2, new_classes.class_3])

if __name__ == "__main__":
    uvicorn.run(app, debug=True)