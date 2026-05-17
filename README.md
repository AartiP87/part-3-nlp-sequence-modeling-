# Part 3: NLP and Sequence Modeling Mini Project

## Overview
This project builds an NLP pipeline to classify customer support messages by sentiment:
**positive**, **neutral**, or **negative**.

---

## Folder Structure

```
part-3-nlp-sequence-modeling/
│
├── README.md               ← This file
├── notebook.ipynb          ← Full assignment notebook (Tasks 1–6)
├── requirements.txt        ← Python dependencies
└── results/
    ├── model_evaluation.png    ← Confusion matrices for both models
    ├── model_evaluation.csv    ← Precision / Recall / F1 per class
    └── sample_predictions.txt  ← 25 sample predictions per model
```

---

## Dataset

| Property | Value |
|----------|-------|
| File | `customer_support_text_classification.csv` |
| Records | 1,500 |
| Target column | `sentiment_label` (positive / neutral / negative) |
| Input column | `customer_message` |
| Avg word count | ~12.7 words per message |

**Place the CSV in the same folder as `notebook.ipynb` before running.**

# Use the relevant Part 3 dataset from the shared folder:

https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing

---

## Tasks Covered

| Task | Description |
|------|-------------|
| Task 1 | Dataset Understanding — shape, class distribution, sample records |
| Task 2 | Text Preprocessing — lowercase, clean, tokenise, remove stopwords |
| Task 3 | Text Vectorization — TF-IDF and Bag of Words with explanation |
| Task 4 | Baseline Models — Logistic Regression (TF-IDF) and Naïve Bayes (BoW) |
| Task 5 | LSTM Sequence Model — full trainable architecture with training curves |
| Task 6 | Reflection — RNNs, LSTMs, Attention, and Transformers |

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Jupyter
```bash
jupyter notebook notebook.ipynb
```

### 3. Run on Google Colab
- Upload `notebook.ipynb` to [colab.research.google.com](https://colab.research.google.com)
- Upload the CSV using the sidebar file upload or run:
```python
from google.colab import files
uploaded = files.upload()
```
- Run all cells top to bottom

---

## Models & Results

| Model | Vectorization | Test Accuracy |
|-------|--------------|---------------|
| Logistic Regression | TF-IDF (unigrams + bigrams) | See results/ |
| Naïve Bayes | Bag of Words | See results/ |
| LSTM | Tokenizer sequences + Embedding | Trained in notebook |

---

## Key Concepts

- **TF-IDF** weights rare but informative words higher than common words
- **LSTM** uses forget / input / output gates to preserve long-range context
- **Attention** removes the fixed-length bottleneck in seq2seq models
- **Transformers** process all tokens in parallel via self-attention, enabling massive scale
