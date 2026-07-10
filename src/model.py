import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
import os

class MicroLiteModel(nn.Module):
    """Микроскопическая языковая модель"""
    
    def __init__(self, vocab_size, embed_size=64, num_heads=4, num_layers=4, max_seq_len=128):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Встраивание токенов
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        
        # Позиционное кодирование (обучаемое)
        self.position_embedding = nn.Embedding(max_seq_len, embed_size)
        
        # Слои трансформера
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Выходной слой
        self.output_layer = nn.Linear(embed_size, vocab_size)
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len] - входные токены
        mask: [seq_len, seq_len] - маска для внимания
        """
        batch_size, seq_len = x.shape
        
        # Встраивание токенов
        token_embeds = self.token_embedding(x)  # [batch_size, seq_len, embed_size]
        
        # Позиционное кодирование
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)  # [batch_size, seq_len, embed_size]
        
        # Суммируем встраивания
        x = token_embeds + pos_embeds  # [batch_size, seq_len, embed_size]
        
        # Трансформер (ожидает [seq_len, batch_size, embed_size])
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, embed_size]
        
        # Создаём маску для внимания (если не передана)
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        
        # Пропускаем через трансформер
        x = self.transformer(x, mask=mask)  # [seq_len, batch_size, embed_size]
        
        # Обратно к [batch_size, seq_len, embed_size]
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, embed_size]
        
        # Выходной слой
        logits = self.output_layer(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate(self, tokenizer, prompt, max_length=50, temperature=1.0, top_k=5):
        """
        Генерация текста по промпту
        """
        self.eval()
        
        # Токенизируем промпт
        input_ids = tokenizer.encode(prompt)
        
        # Добавляем batch dimension
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        generated = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Берём последние max_seq_len токенов
                if len(generated) > self.max_seq_len:
                    context = generated[-self.max_seq_len:]
                else:
                    context = generated
                
                # Превращаем в тензор
                context_tensor = torch.tensor([context], dtype=torch.long)
                
                # Получаем логиты
                logits = self.forward(context_tensor)
                
                # Берём логиты для последнего токена
                next_token_logits = logits[0, -1, :] / temperature
                
                # Применяем top-k
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_values
                
                # Превращаем в вероятности
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Выбираем следующий токен
                next_token = torch.multinomial(probs, 1).item()
                
                # Добавляем в последовательность
                generated.append(next_token)
                
                # Если сгенерирован EOS, останавливаемся
                if next_token == tokenizer.vocab['<EOS>']:
                    break
        
        # Декодируем
        return tokenizer.decode(generated)
    
    def save(self, path):
        """Сохраняет модель"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'embed_size': self.embed_size,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len
        }, path)
        print(f"✅ Модель сохранена в {path}")
    
    @classmethod
    def load(cls, path):
        """Загружает модель"""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            embed_size=checkpoint['embed_size'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            max_seq_len=checkpoint['max_seq_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Модель загружена из {path}")
        return model

# Проверяем работу модели
if __name__ == "__main__":
    # Загружаем токенизатор
    with open('../models/tokenizer.json', 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    # Создаём простой токенизатор для теста
    from tokenizer import MicroLiteTokenizer
    tokenizer = MicroLiteTokenizer()
    tokenizer.vocab = tokenizer_data['vocab']
    tokenizer.vocab_size = tokenizer_data['vocab_size']
    tokenizer.reverse_vocab = {idx: token for token, idx in tokenizer.vocab.items()}
    
    # Создаём модель
    model = MicroLiteModel(
        vocab_size=tokenizer.vocab_size,
        embed_size=64,
        num_heads=4,
        num_layers=4,
        max_seq_len=128
    )
    
    # Проверяем количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Количество параметров: {total_params:,}")
    print(f"📊 Размер модели: ~{total_params * 4 / 1024 / 1024:.2f} MB (в float32)")
    
    # Тестовая генерация
    print("\n🧪 Тестовая генерация:")
    print("=" * 50)
    test_prompt = "ёжик"
    generated = model.generate(tokenizer, test_prompt, max_length=20, temperature=0.8)
    print(f"Промпт: {test_prompt}")
    print(f"Сгенерировано: {generated}")
    print("=" * 50)