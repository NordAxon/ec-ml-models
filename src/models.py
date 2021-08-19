from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

class QA:
    
    def __init__(self):
        self.pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
        
    def answer_question(self, question, context):
        return self.pipeline(question=question, context=context)


class TextGenerator:
    
    def __init__(self):
        self.pipeline = pipeline("text-generation", model='distilgpt2')
        
    def generate_text(self, context, min_length=50, max_length=500):
        " Generates text from the context"
        return self.pipeline(context, min_length=min_length, max_length=max_length)

class SentimentAnalyser:
    
    def __init__(self):
        self.pipeline = pipeline("sentiment-analysis")

    def analyse_text(self, text):
        return self.pipeline(text)
