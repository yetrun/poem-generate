# app.py
# -*- coding: utf-8 -*-
"""
诗歌生成（推理版，含前置提示词 & 体裁切换）
"""

import os
import gradio as gr

from poem.config import read_configs
from poem.generator import PoemGenerator

# -------- 读取配置与构造生成器 --------
poem_configs = read_configs()
if not poem_configs:
    raise RuntimeError("未读取到任何体裁配置，请检查 read_configs() 返回值。")

poem_generators = [
    PoemGenerator(
        genre=cfg.genre,
        vectorization_model=cfg.vectorization_model,
        generation_model=cfg.generation_model,
    )
    for cfg in poem_configs
]

# 体裁名列表 & 索引映射（poem_configs[i].genre.genre 即体裁名）
GENRE_NAMES = [cfg.genre.genre_name for cfg in poem_configs]
GENRE_TO_INDEX = {name: i for i, name in enumerate(GENRE_NAMES)}

# --------（可选）一致性校验：每个体裁的模型输出维度应等于其词表大小 --------
for cfg in poem_configs:
    tv = cfg.vectorization_model
    vocab_size = len(tv.get_vocabulary())
    out_dim = cfg.generation_model.output_shape[-1]
    if out_dim != vocab_size:
        raise ValueError(
            f"体裁“{cfg.genre.genre_name}”的模型输出维度({out_dim})与词表大小({vocab_size})不一致，请检查该体裁的配置。"
        )

# -------- 生成与格式化（根据体裁名选择对应生成器） --------
def generate_and_format_ui(prompt: str, temperature: float, genre_name: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return "⚠️ 请至少输入一个起始字。"
    idx = GENRE_TO_INDEX.get(genre_name, 0)
    generator = poem_generators[idx]
    return generator.generate_and_format(prompt, temperature)

def footer_for_genre(genre_name: str) -> str:
    idx = GENRE_TO_INDEX.get(genre_name, 0)
    config = poem_configs[idx]

    genre = poem_configs[idx].genre
    model_path = config.meta.get('model_path', 'N/A')
    vocabulary_path = config.meta.get('vocabulary_path', 'N/A')
    vocabulary_size = len(config.vectorization_model.get_vocabulary())
    return f"**当前体裁**：`{genre.genre_name}` ｜ **行数**：{genre.rows} ｜ **每行字数**：{genre.cols} ｜ **模型文件**：`{model_path}` ｜ **词表文件**：`{vocabulary_path}` ｜ **词表大小**：{vocabulary_size}"

# -------- Gradio UI --------
DESC = "# 诗歌生成（多体裁）"

with gr.Blocks(title="诗歌生成") as demo:
    gr.Markdown(DESC)
    with gr.Row(equal_height=True):
        genre_dd = gr.Dropdown(
            choices=GENRE_NAMES,
            value=GENRE_NAMES[0],
            label="体裁",
            allow_custom_value=False,
            scale=1
        )
        seed = gr.Textbox(label="前置提示词（至少 1 个字）", placeholder="例如：海外", lines=1, scale=3)
    with gr.Row():
        temp = gr.Slider(0, 2.0, value=0.5, step=0.05, label="温度（0=贪心）")
    btn = gr.Button("📝 生成")
    output = gr.Textbox(label="生成结果", lines=6)

    # 点击生成时，将体裁名一并传入
    btn.click(
        fn=generate_and_format_ui,
        inputs=[seed, temp, genre_dd],
        outputs=[output],
        api_name="generate",
    )

    with gr.Row():
        footer_md = gr.Markdown(value=footer_for_genre(GENRE_NAMES[0]))

    genre_dd.change(fn=footer_for_genre, inputs=genre_dd, outputs=footer_md)
    demo.load(fn=footer_for_genre, inputs=genre_dd, outputs=footer_md)

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
