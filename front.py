import gradio as gr
import numpy as np

def greet(text, image:np.ndarray, difficulty:str) -> str:
    return difficulty

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Markdown("""
            # Генератор рецептів
            Завантажте зображення наявних продуктів.
            """),
            gr.Image(label="Зображення"), 
            gr.Radio(["Low", "Medium", "Hard"], label="Складність", info="Якої складності рецепт?")],
    outputs=[gr.Textbox(label="Рецепт")],
    allow_flagging="never",
)

demo.launch()