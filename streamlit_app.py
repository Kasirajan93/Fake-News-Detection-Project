import streamlit as st
import torch
import torch.nn.functional as F
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from pathlib import Path

# --------------------------
# Load Model
# --------------------------

MODEL_NAME = "kasirajan10/fake-news-distilbert"

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --------------------------
# UI
# --------------------------

st.title("📰 Fake News Detection System")

st.write(
    "Detect whether a news article is Real or Fake using a fine-tuned DistilBERT model."
)

news_text = st.text_area(
    "Paste News Article",
    height=250
)

if st.button("Predict"):

    if news_text.strip() == "":
        st.warning("Please enter a news article.")
    else:

        inputs = tokenizer(
            news_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Move logits safely to CPU
        logits = outputs.logits.detach().cpu()

        # Calculate probabilities
        probs = torch.softmax(logits, dim=1)

        # Get confidence and prediction
        confidence = probs.max().item() * 100
        prediction = probs.argmax(dim=1).item()

        if prediction == 1:
            st.error(
                f"⚠️ Fake News\n\nConfidence: {confidence:.2f}%"
            )
        else:
            st.success(
                f"✅ Real News\n\nConfidence: {confidence:.2f}%"
            )