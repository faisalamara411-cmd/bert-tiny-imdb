import streamlit as st
import requests

st.title("🎬 IMDb Sentiment Analysis")
st.write("Enter a movie review below and find out whether it's **positive** or **negative**.")

API_URL = "http://imdb-backend:8000/predict"


user_input = st.text_area("📝 Enter movie review text:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        response = requests.post(API_URL, json={"text": user_input})
        if response.status_code == 200:
            result = response.json()
            st.success(f"**Sentiment:** {result['sentiment'].capitalize()}")
            st.write(f"**Confidence:** {result['confidence']:.4f}")
            st.write("**Probabilities:**", result['probabilities'])
        else:
            st.error("Error calling API")
    else:
        st.warning("Please enter some text to predict.")



