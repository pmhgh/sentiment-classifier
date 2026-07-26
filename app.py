import streamlit as st
import re
import pickle

# Load saved model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

st.title("🎬 Movie Review Sentiment Classifier")
st.write("Paste a movie review, get positive/negative prediction")

user_input = st.text_area("Enter your movie review:", height=150)

if st.button("Predict Sentiment"):
    if user_input:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]
        
        sentiment = "😊 POSITIVE" if pred == 1 else "😞 NEGATIVE"
        confidence = prob[pred] * 100
        
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Confidence:** {confidence:.1f}%")
    else:
        st.warning("Please enter a review!")