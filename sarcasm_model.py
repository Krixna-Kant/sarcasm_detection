import pandas as pd
import torch
import re
import emoji
from langdetect import detect
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, classification_report

# Loading Datasets
train_df = pd.read_csv("/content/train.csv")
test_df = pd.read_csv("/content/test.csv")

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = emoji.demojize(text)

    try:
        lang = detect(text)
        if lang == "hi":
            text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    except:
        pass
    
    return text

# Text preprocessing
train_df["Cleaned_Tweet"] = train_df["Tweet"].apply(clean_text)
test_df["Cleaned_Tweet"] = test_df["Tweet"].apply(clean_text)

# Converting labels to numerical values
train_df["Label"] = train_df["Label"].map({"YES": 1, "NO": 0})

# Model Selection
MODEL_NAME = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Tokenization function
def tokenize_text(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")

# Tokenize Data
train_encodings = tokenize_text(train_df["Cleaned_Tweet"].tolist(), tokenizer)

# PyTorch Dataset Class
class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Preparing Datasets
full_dataset = SarcasmDataset(train_encodings, train_df["Label"].tolist())
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none",
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    lr_scheduler_type="cosine",
)

# Trainer API
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Train Model
trainer.train()

# Save Model
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# Prediction Function
def predict_sarcasm(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(predictions, dim=1).item()
    confidence = predictions[0][predicted_class].item()
    return "Sarcastic 😏" if predicted_class == 1 else "Not Sarcastic 🙂", confidence
