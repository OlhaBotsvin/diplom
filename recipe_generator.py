from openai import OpenAI
from typing import List, Dict, Any
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RecipeGenerator:
    def __init__(self, openai_client=None):
        """
        Initialize the RecipeGenerator.
        
        Args:
            openai_client: OpenAI client instance (optional, will create new one if not provided)
        """
        # Use provided client or create new one
        if openai_client:
            self.client = openai_client
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
    
    async def generate_recipes(self, ingredients: List[str], difficulty: str = None) -> Dict[str, Any]:
        """
        Generate recipes based on the provided ingredients.
        
        Args:
            ingredients: List of ingredients to use in recipes
            difficulty: Optional difficulty level (легкий, середній, складний)
            
        Returns:
            Dictionary containing recipe data
        """
        try:
            difficulty_guides = {
                "легкий": "прості рецепти, 3-5 кроків, до 30 хвилин, базові техніки приготування",
                "середній": "середня складність, 5-7 кроків, до 60 хвилин, різноманітніші техніки",
                "складний": "складні техніки, 7+ кроків, понад 60 хвилин, комплексні методи приготування"
            }
            
            # Визначаємо рекомендації відповідно до обраної складності
            diff_guide = difficulty_guides.get(difficulty.lower(), difficulty_guides["середній"])
            
            # Покращений промпт з впливом складності
            prompt = f"""
Ти досвідчений шеф-кухар. Створи смачний рецепт зі списку інгредієнтів: {', '.join(ingredients)}.

ЗАВДАННЯ: Створити смачну страву, використовуючи логічну підмножину доступних інгредієнтів.

ОБОВ'ЯЗКОВЕ ПРАВИЛО: Використовуй ТІЛЬКИ інгредієнти зі списку + базові продукти (сіль, перець, цукор, олія, вода).

ПРАВИЛА ПРИГОТУВАННЯ:
1) Продукти, що потребують термічної обробки (м'ясо, риба, яйця, картопля, буряк) - ЗАВЖДИ готуй термічно
2) Використовуй лише логічно сумісні інгредієнти - не обов'язково всі
3) Для продуктів з вказаною кількістю (напр. "10 шт яєць") - бери лише необхідну кількість (2-3 шт)
4) Для продуктів без вказаної кількості - додавай реалістичні пропорції

ЗАБОРОНЕНІ КОМБІНАЦІЇ:
1) Молоко + риба або молоко + огірки(свіжі або консервовані) 
2) Будь-які експериментальні поєднання, що суперечать кулінарній логіці

СКЛАДНІСТЬ РЕЦЕПТУ - {difficulty}: {diff_guide}

ВИМОГИ ДО ФОРМАТУ:
1) Вказуй реалістичний час приготування для кожного етапу
2) Описуй детально способи обробки інгредієнтів
3) Обов'язково включи невикористані інгредієнти до списку "unused_ingredients"

ФОРМАТ ВІДПОВІДІ (JSON):
{{
  "compatible": true,
  "recipes": [
    {{
      "name": "Назва рецепту",
      "ingredients": ["2 шт яйця", "300 г картоплі", "1 шт цибуля", "2 ст.л. олії", "..."],
      "instructions": "1. Перший крок.\\n2. Другий крок.\\n3. Третій крок.",
      "total_time": час у хвилинах,
      "servings": кількість порцій,
      "difficulty": "{difficulty}",
      "serving_suggestions": "Порада щодо подачі.",
      "unused_ingredients": ["помідори", "шинка"]
    }}
  ],
  "message": "Короткий коментар чому деякі інгредієнти не використані."
}}

Якщо створення рецепту неможливе через несумісність усіх інгредієнтів, відповідай СТРОГО у такому форматі:
{{
  "compatible": false,
  "recipes": [],
  "message": "З цих інгредієнтів неможливо створити смачну страву. Спробуйте завантажити фото з іншими продуктами."
}}"""
            
            # Визначаємо параметри запиту
            model = "o4-mini"
            
            # Створюємо параметри запиту без температури
            completion_params = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "response_format": {"type": "json_object"}  # Вказуємо, що очікуємо JSON відповідь
            }
            
            # Виконуємо запит без параметра температури
            completion = self.client.chat.completions.create(**completion_params)
            
            text = completion.choices[0].message.content.strip()
            
            try:
                # Просто парсимо JSON напряму
                recipe_data = json.loads(text)
                return recipe_data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing recipe JSON: {e}")
                # Повертаємо базовий шаблон з повідомленням про помилку
                return {
                    "message": f"На жаль, не вдалося створити рецепт з цих інгредієнтів: {str(e)}",
                    "recipes": []
                }
                
        except Exception as e:
            logger.error(f"Error generating recipes: {str(e)}")
            return {
                "message": f"На жаль, не вдалося створити рецепт з цих інгредієнтів: {str(e)}", 
                "recipes": []
            }