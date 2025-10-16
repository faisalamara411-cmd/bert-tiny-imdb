import streamlit as st
import requests

# URL of your FastAPI backend
API_URL = "http://127.0.0.1:8000/predict"


st.title("IMDb Sentiment Analysis")

# Text input from user
user_input = st.text_area("Enter movie review text:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        payload = {"text": user_input}
        try:
            response = requests.post(API_URL, json=payload)
            result = response.json()
            
            st.write("**Sentiment:**", result.get("sentiment"))
            st.write("**Confidence:**", round(result.get("confidence"), 4))
            st.write("**Probabilities:**", result.get("probabilities"))
        except Exception as e:
            st.error(f"Error connecting to API: {e}")
    else:
        st.warning("Please enter some text to predict.")


