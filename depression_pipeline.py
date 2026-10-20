import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from transformers import (BertForSequenceClassification, BertTokenizer, DataCollatorWithPadding,
                          EarlyStoppingCallback, Trainer, TrainingArguments)

from text_utils import clean_text

DATA_PATH = "depression_dataset_reddit_cleaned.csv"
EXTERNAL_PATH = "Reddit_depression_dataset.csv"
MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "depression_bert"
MAX_LEN = 256
SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

df = pd.read_csv(DATA_PATH).dropna()
df["text"] = df["clean_text"].map(clean_text)
df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
df["label"] = df["is_depression"].astype(int)
print("Dataset:", df.shape)
print(df["label"].value_counts().to_string())

df["n_words"] = df["text"].str.split().str.len()
print("\nMedian words per class:")
print(df.groupby("label")["n_words"].median().to_string())

sns.countplot(x="label", data=df)
plt.title("Class distribution")
plt.savefig("class_distribution.png", dpi=120)
plt.close()

train_df, tmp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
val_df, test_df = train_test_split(tmp_df, test_size=0.5, stratify=tmp_df["label"], random_state=SEED)
print(f"\ntrain {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


class DepressionDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.enc.items()}
        item["labels"] = self.labels[idx]
        return item


train_ds = DepressionDataset(train_df["text"], train_df["label"])
val_ds = DepressionDataset(val_df["text"], val_df["label"])
test_ds = DepressionDataset(test_df["text"], test_df["label"])

weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_df["label"].values)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
print("Class weights:", weights.round(3))


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, preds), "f1": f1_score(p.label_ids, preds)}


model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    warmup_steps=100,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    logging_steps=25,
    fp16=torch.cuda.is_available(),
    seed=SEED,
    report_to="none",
)

trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
)
trainer.train()

print("\nValidation:", trainer.evaluate())

test_out = trainer.predict(test_ds)
test_preds = np.argmax(test_out.predictions, axis=1)
print("\nTest set")
print(classification_report(test_df["label"], test_preds, target_names=["not depressed", "depressed"], digits=3))

cm = confusion_matrix(test_df["label"], test_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["not depressed", "depressed"], yticklabels=["not depressed", "depressed"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion matrix (test)")
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()

if os.path.exists(EXTERNAL_PATH):
    ext = pd.read_csv(EXTERNAL_PATH).dropna()
    ext["text"] = ext["text"].map(clean_text)
    ext["label"] = (ext["label"].str.lower() != "minimum").astype(int)
    ext_ds = DepressionDataset(ext["text"], ext["label"])
    ext_preds = np.argmax(trainer.predict(ext_ds).predictions, axis=1)
    print("\nExternal test (Reddit severity dataset, minimum vs rest)")
    print(classification_report(ext["label"], ext_preds, target_names=["not depressed", "depressed"], digits=3))
else:
    print(f"\n{EXTERNAL_PATH} not found, external test skipped")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved to", OUTPUT_DIR)
