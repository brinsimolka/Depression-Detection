import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertForSequenceClassification, BertTokenizer

from text_utils import clean_text

MODEL_DIR = "depression_bert"
MAX_LEN = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(device).eval()

app = FastAPI(title="Depression Detection API")


class TextInput(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "Depression Detection API is running"}


@app.post("/predict")
def predict(payload: TextInput):
    enc = tokenizer(clean_text(payload.text), truncation=True, max_length=MAX_LEN,
                    return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=1)[0]
    p_dep = float(probs[1])
    return {
        "label": "depressed" if p_dep >= 0.5 else "not depressed",
        "probability_depression": round(p_dep, 4),
        "disclaimer": "Educational tool, not a medical diagnosis.",
    }
