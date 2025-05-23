from google.generativeai import GenerativeModel
import google.generativeai as genai
import base64
from typing import List
from pydantic import BaseModel
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RecognitionResponse(BaseModel):
    ingredients: List[str]

class IngredientRecognizer:
    def __init__(self, api_key=None):
        """
        Initialize the IngredientRecognizer with Gemini API.
        
        Args:
            api_key: Gemini API key (optional, will use env var if not provided)
        """
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Provide it directly or set GEMINI_API_KEY environment variable.")
        
        # Initialize Gemini API
        genai.configure(api_key=self.api_key)
        
        # Create model instance for Gemini 2.0 Flash
        self.model = GenerativeModel(model_name="gemini-2.0-flash")
        
    async def recognize_from_image_bytes(self, image_bytes: bytes) -> List[str]:
        """
        Recognize ingredients from raw image bytes using Gemini.
        
        Args:
            image_bytes: Raw image data
            
        Returns:
            List of recognized ingredients
        """
        try:
            # Encode the image as base64
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            
            # Create prompt with image for Gemini
            prompt = """Ти експерт з розпізнавання продуктів харчування на фотографіях.

ЗАВДАННЯ: Ідентифікувати всі продукти на фото і вказати точну кількість згідно правил.

ПРАВИЛА ВИЗНАЧЕННЯ КІЛЬКОСТІ:
1) ШТУЧНІ ПРОДУКТИ:
   - ЯЙЦЯ: дуже уважно рахуй точну кількість видимих яєць ("4 шт яйця")
   - ФРУКТИ/ОВОЧІ: вказуй кількість цілих одиниць ("3 шт яблука", "2 шт помідори")

2) ПРОДУКТИ БЕЗ КІЛЬКОСТІ:
   - УПАКОВАНІ ПРОДУКТИ: пиши без "шт" і без "пачка" ("масло", "твердий сир", "шинка")
   - РІДИНИ: пиши без кількості ("молоко", "олія", "кефір")
   - СИПКІ ПРОДУКТИ: пиши без кількості ("борошно", "цукор", "рис")

ТОЧНЕ ВИЗНАЧЕННЯ М'ЯСА:
1) Для ЧЕРВОНОГО м'яса:
   - "яловичина" - насичено-червоне, часто з білими прожилками жиру
   - "свинина" - рожево-сіре, часто з білим жиром по краях

2) Для БІЛОГО м'яса:
   - "курятина" - світло-рожева, з тонкою шкірою, часто видно кістки
   - "індичатина" - світліша за курятину, більші шматки

3) Для РИБИ:
   - "риба" - або конкретний вид, якщо впевнений

4) Для ГОТОВИХ М'ЯСНИХ ВИРОБІВ:
   - "шинка" - рожеві нарізані шматки
   - "ковбаса" - циліндрична форма, різні відтінки
   - "сосиски" - менші ніж ковбаса, рівномірного відтінку

КОНКРЕТНІ НАЗВИ ПРОДУКТІВ:
1) СИРИ - розрізняй типи:
   - "кисломолочний сир" - для білого м'якого творогу
   - "твердий сир" - для жовтих твердих сирів
   - "м'який сир" - для бринзи, фети, адигейського

2) М'ЯСО - вказуй конкретний вид:
   - "свинина", "яловичина", "курятина", "індичка"
   - Якщо можливо, вказуй частину: "куряче філе", "свиняча вирізка"

3) МОЛОЧНІ ПРОДУКТИ - розрізняй:
   - "сметана", "йогурт", "кефір", "масло"

ІНСТРУКЦІЇ ТА ЗАБОРОНИ:
1) ІГНОРУЙ написи на упаковках - орієнтуйся на візуальні ознаки
2) НЕ ВИГАДУЙ продукти, яких не бачиш на фото
3) ЗАБОРОНЕНІ ФРАЗИ: "пачка масла", "1 шт сир", "плитка шоколаду", "пакет цукру"
4) Пиши відповідь УКРАЇНСЬКОЮ мовою

ВІДПОВІДАЙ у форматі списку інгредієнтів, розділеного комами.

ПРАВИЛЬНІ ПРИКЛАДИ:
1) "5 шт яєць, масло, твердий сир, цибуля"
2) "2 шт яблука, банан, молоко, сметана"
3) "3 шт помідори, огірок, шинка, пшеничний хліб"

ПОМИЛКИ ТА ВИПРАВЛЕННЯ:
1) Неправильно: "1 шт масло" / Правильно: "масло"
2) Неправильно: "пачка сиру" / Правильно: "твердий сир" або "кисломолочний сир"
3) Неправильно: "3 шт яблука, 1 пакет цукру" / Правильно: "3 шт яблука, цукор"

Перевір свою відповідь перед відправкою."""

            # Call Gemini API with image
            response = self.model.generate_content(
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt},
                            {"inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_image
                            }}
                        ]
                    }
                ],
                generation_config={"temperature": 0.0}
            )
            
            # Extract ingredients from the response
            ingredients_text = response.text.strip()
            
            # Parse the list of ingredients (assuming comma-separated)
            if "," in ingredients_text:
                ingredients = [item.strip().lower() for item in ingredients_text.split(",")]
            else:
                # Handle the case where there are no commas
                ingredients = [item.strip().lower() for item in ingredients_text.split("\n") if item.strip()]
                # If still no items found, just use the whole text
                if not ingredients:
                    ingredients = [ingredients_text.lower()]
            
            # Remove any periods at the end
            ingredients = [ing[:-1] if ing.endswith('.') else ing for ing in ingredients]
            
            return ingredients
            
        except Exception as e:
            logger.error(f"Error recognizing ingredients: {str(e)}")
            raise Exception(f"Error recognizing ingredients with Gemini: {str(e)}")