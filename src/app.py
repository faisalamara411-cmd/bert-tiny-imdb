from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

app = FastAPI(title="IMDb Sentiment Analysis API")

model = AutoModelForSequenceClassification.from_pretrained("bert-tiny-imdb")
tokenizer = AutoTokenizer.from_pretrained("bert-tiny-imdb")

class Review(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(review: Review):
    inputs = tokenizer(review.text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = int(np.argmax(logits.numpy()))
    return {"text": review.text, "sentiment": "positive" if prediction==1 else "negative"}

@app.get("/health")
def health():
    return {"status": "ok"}
