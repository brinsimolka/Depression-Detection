# Depression Detection from Reddit Posts (BERT)

Fine-tuned `bert-base-uncased` that classifies an English text as **depressed** or **not depressed**, served through a FastAPI endpoint.

> Educational project. It is not a medical tool and must not be used to diagnose or judge real people.

## Project structure

| File | Role |
|---|---|
| `depression_pipeline.py` | Data cleaning, train/validation/test split, BERT fine-tuning, evaluation, model saving |
| `text_utils.py` | Text cleaning function shared by training and the API |
| `api.py` | FastAPI service that loads the trained model and exposes `/predict` |
| `depression_dataset_reddit_cleaned.csv` | Dataset (7,731 texts, label `is_depression`) |
| `requirements.txt` | Python dependencies |

## Method

1. Remove empty and duplicate texts, light cleaning (URLs, mentions, extra spaces). Stopwords and punctuation are kept because BERT uses them.
2. Stratified split: 70% train, 15% validation, 15% test.
3. Tokenization with the BERT WordPiece tokenizer (max 256 tokens, dynamic padding per batch).
4. Fine-tuning of `bert-base-uncased` with a 2-class head, class-weighted cross-entropy, AdamW (lr 2e-5), warm-up, up to 4 epochs.
5. Best epoch selected on validation F1, early stopping, final evaluation on the untouched test set.

## How to run

### 1. Clone and install

```bash
git clone https://github.com/brinsimolka/Depression-Detection.git
cd Depression-Detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with `source venv/bin/activate`.

### 2. Train the model

```bash
python depression_pipeline.py
```

A GPU is strongly recommended. On Google Colab, select Runtime > Change runtime type > T4 GPU, then run:

```bash
!git clone https://github.com/YOUR_USERNAME/Depression-Detection.git
%cd Depression-Detection
!pip install -q transformers accelerate seaborn scikit-learn
!python depression_pipeline.py
```

Optional: for an external test, download `Reddit_depression_dataset.csv` from
[usmaann/Depression_Severity_Dataset](https://github.com/usmaann/Depression_Severity_Dataset)
and place it in the project folder before training.

Training produces the `depression_bert/` folder (the model and tokenizer), `confusion_matrix.png` and `class_distribution.png`.

### 3. Start the API

```bash
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs`, choose `POST /predict`, click "Try it out" and send:

```json
{"text": "I feel empty and nothing matters anymore"}
```

Example response:

```json
{"label": "depressed", "probability_depression": 0.97, "disclaimer": "Educational tool, not a medical diagnosis."}
```

## Stack

PyTorch, Hugging Face Transformers, scikit-learn, pandas, NumPy, matplotlib, seaborn, FastAPI, Uvicorn.
