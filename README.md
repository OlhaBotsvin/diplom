# Проєкт "Розпізнавання продуктів харчування та генерація рецептів"
Цей проєкт дозволяє користувачам завантажувати зображення продуктів харчування, розпізнавати інгредієнти та генерувати рецепти на їх основі.
## Встановлення та налаштування

1. Клонуйте репозиторій:
```bash
git clone https://github.com/OlhaBotsvin/diplom.git
cd diplom
```

2. Встановіть залежності:
```bash
pip install -r requirements.txt
```

3. Створіть файл `.env` та додайте свої ключі API:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```
## Запуск додатку

```bash
python app.py
```
