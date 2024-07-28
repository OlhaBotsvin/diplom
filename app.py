from openai import OpenAI
import PIL
import io
import base64

client = OpenAI()

def process_image(image: PIL.Image.Image) -> str:
    # Convert the image to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    response = client.chat.completions.create(
    model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "Ти професійний шеф-кухар. У мене є фотографія продуктів харчування. Створи реалістичний рецепт, який ти можеш приготувати ТІЛЬКИ з цих продуктів.Рецепт повинен бути їстівним та продукти мають поєднуватися. Якщо продукти неможливо поєднати, скажи про це.",
                },
                {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{img_str}"
                },
                },
            ],
            }
        ],
    )
    return response.choices[0].message.content