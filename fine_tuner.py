import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}


tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# lendo dataframe limpo
df = pd.read_pickle("whatsapp_unirio.pkl")

# autores como números em ordem alfabética
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["author"])

# divide em treino e validação
train_texts, temp_text, train_labels, tem_labels = train_test_split(
    df["message"].tolist(),
    df["label"].tolist(),
    test_size=0.4,
    random_state=42,
    stratify=df["label"]
)

val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_text,
    tem_labels,
    test_size=0.5,
    random_state=42,
    stratify=tem_labels
)

# tokenização
train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=512
)
val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding=True,
    max_length=512
)
test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=512
)

# dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, train_labels)
test_dataset = TextDataset(test_encodings, test_labels)
val_dataset = TextDataset(val_encodings, val_labels)


# 5
num_labels = len(label_encoder.classes_)

model = AutoModelForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased",
    num_labels=num_labels
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()


trainer.evaluate()

test_dataset = TextDataset(test_encodings, test_labels)
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
# decodifica os rótulos
decoded_preds = label_encoder.inverse_transform(preds)
# adiciona as previsões ao df original
df_test = pd.DataFrame({"mensagem": test_texts, "autor_previsto": decoded_preds, "autor_real": test_labels})

# salva o DataFrame com as previsões
df_test.to_csv("predictions.csv", index=False)

# salva o modelo
model.save_pretrained("modelo_treinado")
tokenizer.save_pretrained("modelo_treinado")