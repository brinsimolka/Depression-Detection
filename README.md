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

## Results

Fill in after training:

| Evaluation | Accuracy | F1 (depressed) |
|---|---|---|
| Test split of this dataset | to fill | to fill |
| External test: Reddit Depression Severity (minimum vs rest) | to fill | to fill |

## Limitations

- In this dataset, texts labelled "not depressed" are much shorter than "depressed" ones (median 13 vs 78 words), so a high in-dataset score can partly reflect text length and style rather than depression. The external test on a different dataset is the more reliable measure.
- English only. Binary output, no severity level.
- Trained on Reddit text; behaviour on other writing styles is not guaranteed.

## Run it

```bash
python -m venv venv
venv\Scripts\activate            # Windows (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
python depression_pipeline.py    # GPU strongly recommended (e.g. Google Colab T4)
uvicorn api:app --reload
```

Optional: place `Reddit_depression_dataset.csv` from
[usmaann/Depression_Severity_Dataset](https://github.com/usmaann/Depression_Severity_Dataset)
in the project folder before training to get the external test.

Open `http://127.0.0.1:8000/docs` and call `POST /predict`:

```json
{"text": "I feel empty and nothing matters anymore"}
```

The trained model folder `depression_bert/` is not stored in the repository because of its size (about 440 MB). Run the training script to generate it.

## Stack

PyTorch, Hugging Face Transformers, scikit-learn, pandas, NumPy, matplotlib, seaborn, FastAPI, Uvicorn.
