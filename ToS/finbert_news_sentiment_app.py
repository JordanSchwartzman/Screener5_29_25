import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

st.set_page_config(page_title="FinBERT News Sentiment", layout="centered")

# Load FinBERT model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

tokenizer, model = load_model()

# UI layout
st.title("💼 FinBERT-Powered News Sentiment Analyzer")
st.write("Paste in financial news text to get sentiment insights using a model trained specifically on market-related data.")

text_input = st.text_area("Paste financial news here:", height=300)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    labels = ["negative", "neutral", "positive"]
    return labels[prediction.item()], confidence.item(), probs[0].tolist()

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please paste a news article or sentence.")
    else:
        sentiment, confidence, scores = analyze_sentiment(text_input)
        st.subheader("📊 Sentiment Result")
        st.markdown(f"**Sentiment:** `{sentiment.title()}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}`")

        with st.expander("🔬 Full Score Breakdown"):
            st.json({
                "Positive": round(scores[2], 4),
                "Neutral": round(scores[1], 4),
                "Negative": round(scores[0], 4)
            })

        st.caption("Powered by FinBERT (ProsusAI/finbert)")
