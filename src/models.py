from transformers import pipeline
from transformers import CLIPProcessor, CLIPModel
import PIL

class QA:

    def __init__(self):
        """Constructor of QA class, defining the transformers pipeline
        """
        self.pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

    def answer_question(self, question: str, context: str) -> dict:
        """Runs the input through the pre-defined QA pipeline

        Args:
            question (str): The question to be answered
            context (str): The context provided

        Returns:
            dict: A dictionary containing answer to the question etc.
        """
        return self.pipeline(question=question, context=context)


class TextGenerator:
    
    def __init__(self):
        """Constructor of TextGenerator class, defining the transformers pipeline
        """
        self.pipeline = pipeline("text-generation", model='distilgpt2')

    def generate_text(self, context: str, min_length=50, max_length=500) -> list:
        """Calls the model distilgpt2 that creates a continuation to the context string

        Args:
            context (str): The context to be appended with the model output
            min_length (int, optional): Min length of continuation text. Defaults to 50.
            max_length (int, optional): Max length of continuation text. Defaults to 500.

        Returns:
            list(dict): The generated text in a dict nested in a list
        """
        return self.pipeline(context, min_length=min_length, max_length=max_length)

class SentimentAnalyser:
    
    def __init__(self):
        """Constructor for sentiment analysis, calls the predefined transformers pipeline
        """
        self.pipeline = pipeline("sentiment-analysis")

    def analyse_text(self, text: str) -> list:
        """Analyzes the text sentiment by calling the predefined pipeline

        Args:
            text (str): The text to be sentiment analyzed

        Returns:
            list(dict): The result in a dict nested in a list
        """
        return self.pipeline(text)

class ImageClassifier:

    def __init__(self, labels = ['cat', 'dog', 'banana']):
        """Constructor for a model that is classifying images from classes as initialized

        Args:
            labels (list, optional): List of classes to be added in the model. Defaults to ['cat', 'dog', 'banana'].
        """
        self.labels = labels
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def classify(self, image) -> dict:
        """Classifies an input image

        Args:
            image (): The image to be classified

        Returns:
            dict: How well the image corresponds to the model classes
        """
        inputs = self.processor(text=self.labels, images=image, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1).detach().numpy()[0] # we can take the softmax to get the label probabilities
        return self._yield_output(probs, self.labels)

    def _yield_output(self, probs, labels) -> dict:
        "Returns a dict mapping from label to probability"
        result = {z[0]:z[1] for z in zip(labels, probs)}
        return result

    def change_labels(self, new_labels: list):
        "Changes the class labels that the model uses"
        self.labels = new_labels

    def get_labels(self) -> list:
        "Returns the class labels the model uses at inference"
        return self.labels
