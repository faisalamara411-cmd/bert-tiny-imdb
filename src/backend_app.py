from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI(title="BERT Tiny IMDb Inference API")

# Local path to your trained model
model_name = os.path.join(os.path.dirname(__file__), "bert-tiny-imdb")


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class InputText(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "Service is running"}

@app.post("/predict")
def predict(data: InputText):
    inputs = tokenizer(data.text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs).item()

    sentiment = "positive" if label == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": round(float(probs[0][label]), 4),
        "probabilities": {
            "negative": round(float(probs[0][0]), 4),
            "positive": round(float(probs[0][1]), 4)
        }
    }


    

