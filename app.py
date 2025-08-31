# app.py
# -*- coding: utf-8 -*-
"""
极简五言绝句生成（推理版，含前置提示词）
- 必填：前置提示词（至少 1 个字）
- 固定总字数：24（包含前置提示词）
- 只展示格式化结果：4 行 × 6 字
- 依赖文件：poetry_vocabulary.txt, lstm_poetry_model.keras
"""
import os
import numpy as np
import gradio as gr

from keras import models, layers

VOCAB_PATH = os.environ.get("POETRY_VOCAB_PATH", "models/poetry_vocabulary.txt")
MODEL_PATH = os.environ.get("POETRY_MODEL_PATH", "models/lstm_poetry_model-epoch50.keras")
TOTAL_CHARS = 24  # 最终总字数，包含前置提示词

# -------- 词表加载 --------
def load_text_vectorization(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到词表：{path}")
    with open(path, "r", encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f]
    if not vocab:
        raise ValueError("词表为空")

    tv = layers.TextVectorization(
        max_tokens=10000,
        standardize=None,
        split=None,  # 直接喂二维数组
        output_mode="int",
        output_sequence_length=24
    )
    tv.set_vocabulary(vocab)
    return tv

text_vectorization = load_text_vectorization(VOCAB_PATH)
VOCAB_SIZE = len(text_vectorization.get_vocabulary())

def is_cjk(ch: str) -> bool:
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"

# -------- 模型加载 --------
model = models.load_model(MODEL_PATH, compile=False)
try:
    out_dim = model.output_shape[-1]
    if out_dim != VOCAB_SIZE:
        raise ValueError(f"模型输出维度({out_dim})与词表大小({VOCAB_SIZE})不一致")
except Exception:
    pass  # 某些模式下拿不到 output_shape，忽略

# Define the sample generate function
def generate_poem(prompt, max_length=24, temperature=1.0):
    """
    Generate a poem based on the start prompt

    Returns:
        A generated poem as a string.
    """
    prompt_inputs = list(prompt)
    generated = text_vectorization(prompt_inputs)[:len(prompt)].numpy().tolist()
    while len(generated) < max_length:
        input_sequence = np.array(generated).reshape(1, -1)
        predictions = model.predict(input_sequence, verbose=0)[0]
        next_token_id = sample(predictions[-1], temperature)
        generated.append(next_token_id)
    return ''.join(text_vectorization.get_vocabulary()[token_id] for token_id in generated)

def sample(predictions, temperature=1.0, eps1=1e-20, eps2=1e-9):
    p = np.asarray(predictions, dtype=np.float64)

    # The two key points: log(p + eps1) divide by (T + eps2)
    logits = np.log(p + eps1) / (float(temperature) + eps2)

    # Subtract the max logit to prevent overflow
    logits -= np.max(logits)

    q = np.exp(logits)
    q /= q.sum()
    return int(np.random.choice(len(q), p=q))

def generate_and_format(prompt, temperature=1.0):
    # 生成正常的古诗
    poem = generate_poem(prompt, temperature=temperature)

    # 格式化为 4×6
    lines = ["".join(poem[i:i+6]) for i in range(0, TOTAL_CHARS, 6)]
    return "\n".join(lines)

# -------- Gradio UI --------
DESC = "# 五言绝句生成\n请输入**至少 1 个起始字**，总字数固定 24（包含起始字），仅展示 4×6 排版结果。"

with gr.Blocks(title="五言绝句生成") as demo:
    gr.Markdown(DESC)
    with gr.Row():
        seed = gr.Textbox(label="前置提示词（至少 1 个字）", placeholder="例如：海外", lines=1)
    with gr.Row():
        temp = gr.Slider(0, 2.0, value=0.5, step=0.05, label="温度（0=贪心）")
    btn = gr.Button("📝 生成")
    output = gr.Textbox(label="生成结果", lines=6)

    btn.click(fn=generate_and_format, inputs=[seed, temp], outputs=[output], api_name="generate")

    # 页脚信息
    with gr.Row():
        gr.Markdown(
            f"**模型文件**：`{MODEL_PATH}` ｜ **词表**：`{VOCAB_PATH}` ｜ **词表大小**：{VOCAB_SIZE}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
