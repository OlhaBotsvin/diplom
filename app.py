import gradio as gr
import numpy as np
import PIL
import io
import logging
import os
import imghdr
from dotenv import load_dotenv
from ingredient_recognition import IngredientRecognizer
from recipe_generator import RecipeGenerator
from openai import OpenAI

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Завантаження змінних середовища з .env файлу
load_dotenv()

# Перевірка наявності API ключів
gemini_api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not gemini_api_key:
    logger.error("GEMINI_API_KEY не знайдено в .env файлі")
    raise ValueError("GEMINI_API_KEY не знайдено. Додайте ключ у .env файл")

if not openai_api_key:
    logger.error("OPENAI_API_KEY не знайдено в .env файлі")
    raise ValueError("OPENAI_API_KEY не знайдено. Додайте ключ у .env файл")

openai_client = OpenAI(api_key=openai_api_key)

# Ініціалізація класів для розпізнавання інгредієнтів та генерації рецептів
ingredient_recognizer = IngredientRecognizer(api_key=gemini_api_key)  # Gemini для розпізнавання
recipe_generator = RecipeGenerator(openai_client)  # o4-mini для генерації рецептів

# Функція для перевірки зображення
def validate_image(image_path):
    if image_path is None:
        return False, "Будь ласка, завантажте зображення продуктів.", None
    img_type = imghdr.what(image_path)
    if img_type not in ("jpeg", "jpg"):
        return False, "Зображення має бути у форматі JPEG (.jpg, .jpeg).", None
    file_size = os.path.getsize(image_path)
    if file_size > 5 * 1024 * 1024:
        return False, f"Розмір зображення ({file_size/1024/1024:.2f} MB) перевищує 5 MB.", None
    img = PIL.Image.open(image_path)
    width, height = img.size
    if not ((width >= 720 and height >= 1280) or (width >= 1280 and height >= 720)):
        return False, f"Розмір зображення ({width}x{height}) менший за 720×1280 або 1280×720.", None
    return True, "Зображення відповідає вимогам.", img

async def recipe_generation(image_path: str, difficulty: str) -> tuple[str, str, str]:
    """
    Advanced recipe generation that first recognizes ingredients with Gemini and then generates recipes with o4-mini.
    
    Args:
        image: Uploaded image of food ingredients
        difficulty: Difficulty level of the recipe
    
    Returns:
        Tuple of (title, ingredients list, recipe content)
    """
    try:
        # Перевірити, чи обрана складність
        if not difficulty:
            return "# Помилка", "", "Будь ласка, виберіть складність рецепту."

        # Перевірити валідність зображення
        is_valid, message, image = validate_image(image_path)
        if not is_valid:
            return "# Помилка", "", message
        
        # Перетворити зображення в байти для Gemini
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        image_bytes = buffered.getvalue()
        
        logger.info("Розпізнавання інгредієнтів...")
        ingredients = await ingredient_recognizer.recognize_from_image_bytes(image_bytes)
        
        # Перевірка чи розпізнані продукти
        if not ingredients or len(ingredients) == 0:
            return "# Не знайдено продуктів", "", "На зображенні не вдалося розпізнати жодних продуктів харчування. Будь ласка, завантажте інше фото з чітко видимими продуктами."
        
        ingredients_text = "## Розпізнані інгредієнти:\n" + ", ".join(ingredients)
        logger.info(f"Розпізнані інгредієнти: {ingredients}")
        
        difficulty_map = {
            "Легкий": "легкий",
            "Середній": "середній",
            "Важкий": "складний"
        }
        
        logger.info(f"Генерація рецепту для інгредієнтів зі складністю {difficulty}...")
        
        # Генеруємо рецепт з перевіркою сумісності інгредієнтів
        modified_recipe_data = await recipe_generator.generate_recipes(ingredients, difficulty_map.get(difficulty, "середній"))
        
        # Перевіряємо, чи інгредієнти сумісні
        if modified_recipe_data.get("compatible") == False:
            # Якщо інгредієнти несумісні, виводимо просте повідомлення
            message = modified_recipe_data.get("message", "З цих інгредієнтів неможливо створити смачну страву. Спробуйте завантажити фото з іншими продуктами.")
            return "# Несумісні інгредієнти", ingredients_text, f"{message}"
        
        # Якщо рецепти знайдені, форматуємо їх
        if modified_recipe_data.get("recipes"):
            recipe = modified_recipe_data["recipes"][0]  
            recipe_text = f"## {recipe.get('name', 'Рецепт')}\n\n"
            
            recipe_text += "### Інгредієнти:\n"
            for ingredient in recipe.get('ingredients', []):
                recipe_text += f"* {ingredient}\n"
            
            recipe_text += "\n### Інструкції:\n"
            instructions = recipe.get('instructions', 'Інструкції відсутні')
            
            
            if "\\n" in instructions:
                instructions = instructions.replace("\\n", "\n")
                
            recipe_text += instructions
            
            # Форматування інформації про рецепт
            recipe_text += "\n\n### Інформація:\n"
            recipe_text += f"* **Загальний час приготування:** {recipe.get('total_time', 0)} хвилин\n"
            recipe_text += f"* **Кількість порцій:** {recipe.get('servings', 2)}\n"
            
            # Додаємо поради з подачі, якщо є
            if recipe.get('serving_suggestions'):
                recipe_text += f"\n### Подача:\n{recipe.get('serving_suggestions')}"
            
            # Додаємо інформацію про невикористані інгредієнти, якщо такі є
            unused_ingredients = recipe.get('unused_ingredients', [])
            if unused_ingredients and len(unused_ingredients) > 0:
                # Використовуємо повідомлення з пояснення, якщо воно є
                if modified_recipe_data.get("message"):
                    recipe_text += f"\n\n### Примітка:\n{modified_recipe_data.get('message')}"
            
            return "# Рецепт", ingredients_text, recipe_text
        else:
            # Якщо рецепт не знайдено, виводимо повідомлення
            message = modified_recipe_data.get("message", "На жаль, з цих інгредієнтів неможливо створити повноцінний рецепт.")
            return "# Результат аналізу", ingredients_text, f"{message}"
    
    except Exception as e:
        logger.error(f"Помилка в процесі генерації рецепту: {str(e)}")
        return "# Помилка", "", f"Сталася помилка при генерації рецепту: {str(e)}"

def clear_outputs():
    """Функція для очищення всіх полів інтерфейсу"""
    return None, None, "", "", ""

# Функція для створення інтерфейсу Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Генератор рецептів")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Зображення продуктів",
                type="filepath",
                sources=["upload"],
                image_mode="RGB"
            )
            
            gr.Markdown("""
            **Вимоги до зображень:**
            - Формат JPEG (.jpg, .jpeg)
            - Розмір не менше 720×1280 або 1280×720 пікселів
            - Розмір файлу не більше 5 MB
            """)
            
            difficulty = gr.Radio(
                ["Легкий", "Середній", "Важкий"], 
                label="Складність рецепту"
            )
            submit_button = gr.Button("Розпізнати інгредієнти та згенерувати рецепт", variant="primary")
            clear_button = gr.Button("Очистити все")
        
        with gr.Column():
            title_output = gr.Markdown("# Рецепт")
            ingredients_output = gr.Markdown()
            recipe_output = gr.Markdown()
    
    submit_button.click(
        fn=recipe_generation,
        inputs=[image_input, difficulty],
        outputs=[title_output, ingredients_output, recipe_output]
    )
    
    clear_button.click(
        fn=clear_outputs,
        inputs=[],
        outputs=[image_input, difficulty, title_output, ingredients_output, recipe_output]
    )

    gr.Markdown("""
    ### Як користуватися:
    1. Завантажте фотографію наявних продуктів (JPEG формат)
    2. Виберіть бажаний рівень складності рецепту
    3. Натисніть кнопку для розпізнавання інгредієнтів та генерації рецепту
    4. Отримайте список розпізнаних інгредієнтів та рецепт (якщо можливо його створити)
    
    ### Примітка:
    Система автоматично підбирає найкращу комбінацію інгредієнтів для створення смачної страви.<br> 
    Не всі розпізнані інгредієнти будуть використані у рецепті - програма вибере логічне поєднання продуктів.
    """)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)