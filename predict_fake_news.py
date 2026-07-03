from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load saved model and tokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "distilbert_fake_news"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

# Sample news
news_text = input("Enter News Text:\n")

# Tokenize
inputs = tokenizer(
    news_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=256
)

# Predict
import torch.nn.functional as F

with torch.no_grad():
    outputs = model(**inputs)

probs = F.softmax(outputs.logits, dim=1)

confidence = torch.max(probs).item() * 100

prediction = torch.argmax(probs, dim=1).item()

if prediction == 1:
    print(f"⚠️ Fake News ({confidence:.2f}%)")
else:
    print(f"✅ Real News ({confidence:.2f}%)")