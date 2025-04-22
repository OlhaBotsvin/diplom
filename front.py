import gradio as gr
import numpy as np
import PIL
from app import process_image

def greet(text, image:PIL.Image.Image, difficulty:str) -> tuple[str,str]:
    response = process_image(image)
    return "# Рецепт", response

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Markdown("""
            # Генератор рецептів
            Завантажте зображення наявних продуктів.
            """),
            gr.Image(label="Зображення", type="pil"), 
            gr.Radio(["Легкий", "Середній", "Важкий"], label="Складність рецепту", info="Якої складності рецепт?")],
    outputs=[gr.Markdown("""
            # Рецепт
            """), gr.Markdown()],
    allow_flagging="never",
)

demo.launch()

# correct version