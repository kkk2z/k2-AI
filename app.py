from quart import Quart, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

app = Quart(__name__)

# トークナイザーとモデルの読み込み
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese')

# データの読み込み
with open('initial_data.json', 'r') as f:
    data = json.load(f)

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

@app.route('/')
async def home():
    return await render_template('index.html')

@app.route('/chat', methods=['POST'])
async def chat():
    req_data = await request.get_json()
    user_input = req_data['input']
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    response_id = outputs.logits.argmax().item()
    response = data[response_id]['response']

    if response == "[UNK]":
        return jsonify({'response': 'その言葉は知りません。どういう意味ですか？'})

    return jsonify({'response': response})

@app.route('/learn', methods=['POST'])
async def learn():
    req_data = await request.get_json()
    new_word = req_data['new_word']
    meaning = req_data['meaning']

    # データの追加
    new_data = {"input": new_word, "response": meaning}
    data.append(new_data)
    with open('initial_data.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 自己学習と不明単語の処理
    if len(data) > 1000:
        await self_learn()

    return jsonify({'status': 'learned'})

async def self_learn():
    # 自己学習のロジック
    inputs = tokenizer([d["input"] for d in data], truncation=True, padding=True, return_tensors="pt")
    labels = [torch.tensor(1 if d["response"] != "不明" else 0) for d in data]
    train_dataset = CustomDataset(inputs, labels)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained('./results')
    tokenizer.save_pretrained('./results')

    # 不明な単語の自動処理
    unknown_words = [word for word in data if word['response'] == "不明"]
    for word in unknown_words:
        word['response'] = "不明"
    with open('initial_data.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    app.run(debug=True)
