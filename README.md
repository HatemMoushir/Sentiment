# Sentiment140 Fine-Tuning using
DistilBERT (Stanford Dataset)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HatemMoushir/sentiment/blob/main/sentiment140_distilbert.ipynb)

This project demonstrates how to fine-tune the `distilbert-base-uncased` model on the **Sentiment140** dataset from Stanford using Hugging Face Transformers.

The training was done on a subset of 10,000 tweets (from the full 1.6M dataset) for quick experimentation.

---

## 📦 Dataset

- Source: [Sentiment140](https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)
- Original size: 1.6M labeled tweets
- Classes:  
  - `0` = Negative  
  - `4` = Positive (mapped to `1` during preprocessing)

---

## 🚀 How to Run

1. **Download & Extract Data**
```bash
!wget https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
!unzip trainingandtestdata.zip

2. Install Dependencies



!pip install -q datasets transformers evaluate pandas

3. Preprocess & Prepare Dataset



Load the CSV

Filter labels (0 and 4 only)

Map 4 → 1

Sample 10,000 tweets

Convert to Hugging Face Dataset


4. Tokenize & Split



Tokenization with distilbert-base-uncased

Train/test split: 90% / 10%


5. Fine-tune with Trainer API



2 epochs

Batch size: 16

Accuracy metric


6. Save & Load Model



Model saved in sent140_model/

Reload and use with pipeline



---

🧪 Evaluation Results

Metric	Value

Accuracy (Val) – Epoch 1	83.43%
Accuracy (Val) – Epoch 2	(no improvement) – 83.43%
Training Loss (Epoch 1)	0.4022
Validation Loss (Epoch 1)	0.3784
Training Loss (Epoch 2)	0.2536
Validation Loss (Epoch 2)	0.4179


> ✅ Note: Training was done on a small subset (10k tweets) for faster experimentation on Google Colab.




---

✅ Quick Test

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("sent140_model")
tokenizer = AutoTokenizer.from_pretrained("sent140_model")

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(classifier("This is the best day ever!"))


---

🛠 Built With

Transformers

Datasets

Google Colab

ChatGPT assistance for training script and documentation



---

📜 License

MIT – Feel free to use, modify, and share.


---

✨ Author

Developed by Hatem Moushir
GitHub: @HatemMoushir
