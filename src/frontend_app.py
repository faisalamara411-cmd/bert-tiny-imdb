import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "prajjwal1/bert-tiny"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

st.title("🎬 IMDb Sentiment Analysis")
st.write("Enter a movie review below and find out whether it's **positive** or **negative**.")

user_input = st.text_area("📝 Enter movie review text:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()

        sentiment = "Positive 😊" if pred == 1 else "Negative 😞"
        confidence = probs[0][pred].item()

        st.success(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.4f}")
        st.write("**Probabilities:**", probs.tolist())
    else:
        st.warning("Please enter some text to predict.")


