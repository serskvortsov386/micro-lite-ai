import re
from collections import Counter
import json
import os

class MicroLiteTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts, vocab_size=500):
        """Создаём словарь из текстов"""
        # Разбиваем текст на слова и знаки препинания
        tokens = []
        for text in texts:
            # Разбиваем по пробелам и знакам препинания
            words = re.findall(r'\w+|[.,!?;]', text.lower())
            tokens.extend(words)
        
        # Считаем частоту токенов
        token_counts = Counter(tokens)
        
        # Создаём словарь: специальные токены + самые частые слова
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        vocab = special_tokens.copy()
        
        # Добавляем самые частые слова (ограничиваем vocab_size)
        most_common = token_counts.most_common(vocab_size - len(special_tokens))
        vocab.extend([token for token, _ in most_common])
        
        # Создаём отображения
        self.vocab = {token: idx for idx, token in enumerate(vocab)}
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
        print(f"✅ Создан словарь размером {self.vocab_size} токенов")
        return self.vocab
    
    def encode(self, text):
        """Превращаем текст в последовательность чисел"""
        # Разбиваем текст на токены
        tokens = re.findall(r'\w+|[.,!?;]', text.lower())
        
        # Превращаем каждый токен в число
        ids = []
        ids.append(self.vocab['<BOS>'])  # Начало предложения
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['<UNK>'])  # Неизвестное слово
        ids.append(self.vocab['<EOS>'])  # Конец предложения
        
        return ids
    
    def decode(self, ids):
        """Превращаем числа обратно в текст"""
        tokens = []
        for idx in ids:
            token = self.reverse_vocab.get(idx, '<UNK>')
            if token in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                continue
            tokens.append(token)
        
        # Восстанавливаем текст (добавляем пробелы где нужно)
        text = ' '.join(tokens)
        text = re.sub(r'\s([.,!?;])', r'\1', text)  # Убираем пробелы перед знаками
        return text
    
    def save(self, path):
        """Сохраняем словарь в файл"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Токенизатор сохранён в {path}")
    
    def load(self, path):
        """Загружаем словарь из файла"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.vocab_size = data['vocab_size']
            self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        print(f"✅ Токенизатор загружен из {path}")
        return self

# Тестируем токенизатор
if __name__ == "__main__":
    # Загружаем датасет
    with open('../data/dataset.txt', 'r', encoding='utf-8') as f:
        texts = f.readlines()
    
    # Создаём токенизатор
    tokenizer = MicroLiteTokenizer()
    tokenizer.build_vocab(texts, vocab_size=200)
    
    # Тестируем
    test_text = "Ёжик ищет большой гриб"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"\n📝 Тест токенизатора:")
    print(f"Исходный текст: {test_text}")
    print(f"Закодировано: {encoded}")
    print(f"Декодировано: {decoded}")
    
    # Сохраняем токенизатор
    os.makedirs('../models', exist_ok=True)
    tokenizer.save('../models/tokenizer.json')