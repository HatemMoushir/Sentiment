# 🌍 Arabic-English Tweet Sentiment Classification

This project trains a sentiment analysis model on **Arabic** and **English** tweets using a multilingual BERT model (`bert-base-multilingual-cased`). It combines datasets from **UCI** and **Stanford Sentiment140** to build a bilingual classifier that detects **positive** or **negative** sentiment.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HatemMoushir/sentiment/blob/main/aren_sentiment_classification.ipynb)

---

## 📌 Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Datasets](#datasets)
- [Data Cleaning](#data-cleaning)
- [Model Setup](#model-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)
- [Notes](#notes)

---

## 🧠 Overview

This notebook builds a binary sentiment classifier for short texts (tweets) in **Arabic** and **English**. It:

- Loads Arabic sentiment tweets from UCI.
- Loads English tweets from Stanford Sentiment140.
- Cleans and preprocesses both datasets.
- Combines the two into one dataset.
- Fine-tunes a pretrained multilingual BERT model.
- Evaluates accuracy, precision, recall, and F1 score.

---

## ✅ Requirements

Install all dependencies with:

```bash
pip install -q datasets transformers evaluate pandas scikit-learn


---

📊 Datasets

Arabic Dataset

Source: UCI Arabic Tweets Sentiment Classification 2024

Labels used: Positive, Negative (Neutral entries removed)


English Dataset

Source: Stanford Sentiment140

Original labels: 0 = Negative, 4 = Positive → mapped to 0/1.


Both datasets are downsampled to 1,000 samples each for quick experimentation.


---

🧼 Data Cleaning

Arabic Text:

Character normalization (e.g., أ → ا, ى → ي, ة → ه)

URL and non-Arabic character removal


English Text:

Lowercasing

Removal of: URLs, mentions, hashtags, punctuation, numbers


# Arabic label mapping
{"Negative": 0, "Positive": 1}


---

🤖 Model Setup

Tokenizer: bert-base-multilingual-cased

Model: AutoModelForSequenceClassification with 2 output labels

Texts are tokenized with max length 128 and padded/truncated accordingly.



---

🏋️‍♂️ Training

Train/test split: 90% training / 10% test

Epochs: 2

Batch size: 16

Evaluation metrics:

Accuracy

Precision

Recall

F1 Score




---

📈 Evaluation

The model is evaluated at the end of each epoch using scikit-learn metrics:

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


---

💾 Saving the Model

After training, the final model and tokenizer are saved locally:

trainer.save_model("sentiment_model_final")
tokenizer.save_pretrained("sentiment_model_final")

---

✍️ Author

Hatem Moushir
📫 h_moushir@hotmail.com
