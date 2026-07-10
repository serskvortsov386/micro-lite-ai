import sys
sys.path.append('src')

from tokenizer import MicroLiteTokenizer
from model import MicroLiteModel

# Загружаем токенизатор
tokenizer = MicroLiteTokenizer()
tokenizer.load('models/tokenizer.json')

# Загружаем модель
model = MicroLiteModel.load('models/model.pth')

# Генерация
prompts = [
    "ёжик",
    "гриб",
    "ёжик нашёл большой гриб",
    "белочка",  # Этого персонажа модель не видела в обучении — посмотрим, что сделает!
]

print("=" * 50)
print("🎨 Генерация текстов MicroLiteAI")
print("=" * 50)

for prompt in prompts:
    print(f"\n📝 Промпт: {prompt}")
    generated = model.generate(
        tokenizer, 
        prompt, 
        max_length=40, 
        temperature=0.7,
        top_k=8
    )
    print(f"💬 Ответ: {generated}")
    print("-" * 40)