import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import json

# トークナイザーとモデルの読み込み
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese', num_labels=2)

# データの読み込み
with open('initial_data.json', 'r') as f:
    initial_data = json.load(f)

# トークナイズ
inputs = tokenizer([d["input"] for d in initial_data], truncation=True, padding=True, return_tensors="pt")
labels = [1 if d["response"] != "不明" else 0 for d in initial_data]

# データセットの作成
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = CustomDataset(inputs, labels)

# トレーニングの設定
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# トレーナーの設定
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# モデルのトレーニング
trainer.train()

# トレーニング済みモデルの保存
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
