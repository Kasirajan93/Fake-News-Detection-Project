# Fake-News-Detection-Project


## 📸 Application Screenshot

![Fake News Detection App](images/app_screenshot.png)

# 📰 Fake News Detection using DistilBERT

## 📌 Project Overview

Fake news spreads rapidly across digital platforms and can influence public opinion, create misinformation, and cause social harm. This project uses Natural Language Processing (NLP) and Transformer-based Deep Learning to automatically classify news articles as **Real News** or **Fake News**.

The model is built using **DistilBERT**, a lightweight version of BERT that provides high accuracy while reducing computational requirements.

---

## 🚀 Features

- News Article Classification (Real / Fake)
- DistilBERT Fine-Tuning using Hugging Face Transformers
- Streamlit Web Application
- Confidence Score Prediction
- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Baseline Comparison using Logistic Regression
- Interactive User Interface

---

## 📂 Project Structure

```text
FAKE-NEWS-PREDICTION
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── Fake.csv
│   └── True.csv
│
├── models/
│   └── distilbert_fake_news/
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_DistilBERT.ipynb
│
├── src/
│   ├── predict_fake_news.py
│   └── test_samples.txt
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset

### Source

Kaggle Fake and Real News Dataset

Dataset contains:

| Category | Records |
|-----------|----------|
| Fake News | 23,481 |
| Real News | 21,417 |
| Total | 44,898 |

Columns:

- title
- text
- subject
- date

After duplicate removal:

```text
Final Dataset Size: 44,689
```

---

## 🔍 Exploratory Data Analysis

### Key Findings

- No missing values detected
- Dataset is nearly balanced
- 209 duplicate records removed
- Combined title and article text for training

### Label Encoding

| Label | Meaning |
|---------|---------|
| 0 | Real News |
| 1 | Fake News |

---

## ⚙️ Data Preprocessing

The following preprocessing steps were applied:

- Convert text to lowercase
- Remove URLs
- Remove HTML tags
- Remove special characters
- Remove extra spaces

Example:

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

---

# 🤖 Baseline Model

## TF-IDF + Logistic Regression

### Model

```python
TfidfVectorizer(max_features=10000)
LogisticRegression(max_iter=1000)
```

### Result

```text
Accuracy: 98.96%
```

This baseline provides strong performance using traditional machine learning techniques.

---

# 🧠 DistilBERT Model

## Why DistilBERT?

DistilBERT is:

- 40% smaller than BERT
- Faster inference
- Lower memory usage
- Retains approximately 97% of BERT's performance

### Tokenizer

```python
DistilBertTokenizerFast
```

### Model

```python
DistilBertForSequenceClassification
```

### Training Configuration

| Parameter | Value |
|------------|--------|
| Epochs | 1 |
| Batch Size | 4 |
| Max Length | 256 |
| Labels | 2 |
| Framework | Hugging Face Transformers |

---

# 📈 Model Performance

## DistilBERT Results

### Accuracy

```text
99.60%
```

### Validation Loss

```text
0.026588
```

### Classification Report

| Class | Precision | Recall | F1 Score |
|---------|-----------|---------|-----------|
| Real | 1.00 | 0.99 | 1.00 |
| Fake | 0.99 | 1.00 | 1.00 |

### Confusion Matrix

```text
[[932, 7],
 [1, 1060]]
```

### Interpretation

- 932 Real articles correctly classified
- 1060 Fake articles correctly classified
- Only 8 misclassifications out of 2000 samples

---

# 🖥️ Streamlit Application

Launch the application:

```bash
streamlit run app/streamlit_app.py
```

The application allows users to:

- Paste a news article
- Predict whether it is Real or Fake
- View prediction confidence score

Example Output:

```text
⚠️ Fake News
Confidence: 99.6%
```

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/Kasirajan93/Fake-News-Detection-Project.git
```

Navigate into the project:

```bash
cd Fake-News-Detection-Project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/streamlit_app.py
```

---

# 🛠️ Technologies Used

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-Learn

### Deep Learning

- PyTorch
- Hugging Face Transformers

### Web Application

- Streamlit

---

# 📌 Limitations

The model was trained on political and news-domain articles from the Kaggle dataset.

Therefore:

- Performance may decrease on unseen domains
- Scientific articles
- Space research articles
- Entertainment news
- Regional language news

This phenomenon is known as **Domain Shift**.

---

# 🔮 Future Improvements

- Deploy on Streamlit Cloud
- Add Explainable AI (XAI)
- Integrate News API
- Multilingual Fake News Detection
- Real-time News Verification
- Advanced Model Comparison (BERT, RoBERTa, DeBERTa)

---

# 👨‍💻 Author

### Kasi Rajan

Former Ayurvedic Doctor transitioning into Data Science & AI.

### Skills

- Python
- SQL
- Machine Learning
- Deep Learning
- NLP
- Power BI
- Tableau

### GitHub

https://github.com/Kasirajan93

### LinkedIn

https://linkedin.com/in/kasi-rajan-488005349

---

# ⭐ Acknowledgements

- Kaggle Fake and Real News Dataset
- Hugging Face Transformers
- PyTorch
- Streamlit
- Scikit-Learn
