import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from model import MicroLiteModel
from tokenizer import MicroLiteTokenizer

class TextDataset(Dataset):
    """Датасет для обучения"""
    
    def __init__(self, texts, tokenizer, seq_length=32):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Превращаем все тексты в токены
        self.tokens = []
        for text in texts:
            if text.strip():
                tokens = tokenizer.encode(text.strip())
                self.tokens.extend(tokens)
        
        # Создаём последовательности
        self.sequences = []
        for i in range(0, len(self.tokens) - seq_length, seq_length // 2):
            seq = self.tokens[i:i + seq_length]
            if len(seq) == seq_length:
                self.sequences.append(seq)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Вход: все токены кроме последнего
        x = torch.tensor(seq[:-1], dtype=torch.long)
        # Цель: все токены кроме первого (сдвиг вправо)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y

def train():
    # Загружаем токенизатор
    print("📖 Загрузка токенизатора...")
    tokenizer = MicroLiteTokenizer()
    tokenizer.load('../models/tokenizer.json')
    
    # Загружаем датасет
    print("📚 Загрузка датасета...")
    with open('../data/dataset.txt', 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"📊 Найдено {len(texts)} текстов")
    
    # Создаём датасет и загрузчик
    dataset = TextDataset(texts, tokenizer, seq_length=32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    print(f"📊 Создано {len(dataset)} обучающих последовательностей")
    
    # Создаём модель
    print("🧠 Создание модели...")
    model = MicroLiteModel(
        vocab_size=tokenizer.vocab_size,
        embed_size=64,
        num_heads=4,
        num_layers=4,
        max_seq_len=128
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Количество параметров: {total_params:,}")
    
    # Настройки обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"💻 Обучение на {device}")
    print("=" * 50)
    
    # Обучение
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.permute(0, 2, 1), y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch+1}/{epochs}, Потери: {avg_loss:.4f}")
    
    print("=" * 50)
    print("✅ Обучение завершено!")
    
    # Сохраняем модель
    os.makedirs('../models', exist_ok=True)
    model.save('../models/model.pth')
    
    # Тестовая генерация
    print("\n🧪 Тестовая генерация:")
    print("=" * 50)
    test_prompts = ["ёжик", "гриб", "ёжик нашёл"]
    for prompt in test_prompts:
        generated = model.generate(tokenizer, prompt, max_length=30, temperature=0.8)
        print(f"Промпт: {prompt}")
        print(f"Сгенерировано: {generated}")
        print("-" * 30)

if __name__ == "__main__":
    train()