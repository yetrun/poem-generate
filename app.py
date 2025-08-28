# app.py
# -*- coding: utf-8 -*-
"""
极简五言绝句生成（推理版，含前置提示词）
- 必填：前置提示词（至少 1 个字）
- 固定总字数：24（包含前置提示词）
- 只展示格式化结果：4 行 × 6 字
- 依赖文件：poetry_vocabulary.txt, lstm_poetry_model.keras

%pip install gradio
"""
import os
import random
import numpy as np
import gradio as gr

from keras import models

VOCAB_PATH = os.environ.get("POETRY_VOCAB_PATH", "poetry_vocabulary.txt")
MODEL_PATH = os.environ.get("POETRY_MODEL_PATH", "lstm_poetry_model.keras")
TOTAL_CHARS = 24  # 最终总字数，包含前置提示词

# -------- 词表加载 --------
def load_vocab(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到词表：{path}")
    with open(path, "r", encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f]
    if not vocab:
        raise ValueError("词表为空")
    token_to_id = {t: i for i, t in enumerate(vocab)}
    id_to_token = list(vocab)
    pad_like = ("", "[PAD]", "<PAD>", "[pad]", "<pad>")
    unk_like = ("[UNK]", "<UNK>", "[unk]", "<unk>")
    pad_id = next((i for i, t in enumerate(vocab) if t in pad_like), 0)
    unk_id = next((i for i, t in enumerate(vocab) if t in unk_like), 1 if len(vocab) > 1 else 0)
    return vocab, token_to_id, id_to_token, pad_id, unk_id

vocab, token_to_id, id_to_token, PAD_ID, UNK_ID = load_vocab(VOCAB_PATH)
VOCAB_SIZE = len(vocab)

def is_cjk(ch: str) -> bool:
    return len(ch) == 1 and "\u4e00" <= ch <= "\u9fff"

# 允许采样的候选（优先仅单个汉字）
ALLOWED_IDS = [i for i, t in enumerate(id_to_token) if is_cjk(t)]
if not ALLOWED_IDS:
    ALLOWED_IDS = [i for i in range(VOCAB_SIZE) if i not in (PAD_ID, UNK_ID)]

# -------- 模型加载 --------
model = models.load_model(MODEL_PATH, compile=False)
try:
    out_dim = model.output_shape[-1]
    if out_dim != VOCAB_SIZE:
        raise ValueError(f"模型输出维度({out_dim})与词表大小({VOCAB_SIZE})不一致")
except Exception:
    pass  # 某些模式下拿不到 output_shape，忽略

# -------- 编解码 --------
def text_to_ids(text: str):
    return [token_to_id.get(ch, UNK_ID) for ch in list(text or "")]

# -------- 采样（仅温度）--------
def sample_id_from_probs(probs: np.ndarray, temperature: float) -> int:
    probs = np.asarray(probs, dtype="float64")
    mask = np.zeros_like(probs, dtype=bool)
    mask[ALLOWED_IDS] = True
    probs = np.where(mask, np.maximum(probs, 0.0), 0.0)
    s = probs.sum()
    if s <= 0:
        return int(random.choice(ALLOWED_IDS))
    probs = probs / s

    if temperature <= 0:
        return int(np.argmax(probs))

    logits = np.log(probs + 1e-9) / float(temperature)
    logits -= np.max(logits)
    p = np.exp(logits)
    ps = p.sum()
    if ps <= 0:
        return int(random.choice(ALLOWED_IDS))
    p /= ps
    return int(np.random.choice(len(p), p=p))

# -------- 生成（总计 24 字，包含前置提示词）--------
def generate_total_24(seed_text: str, temperature: float) -> str:
    seed_text = (seed_text or "").strip()
    if not seed_text:
        return "⚠️ 请至少输入一个起始字。"

    ids = text_to_ids(seed_text)

    # 先把前置提示词里的汉字放入展示序列
    displayed_chars = [ch for ch in seed_text if is_cjk(ch)]
    displayed_chars = displayed_chars[:TOTAL_CHARS]  # 过长则截断直接返回
    if len(displayed_chars) >= TOTAL_CHARS:
        # 直接格式化输出
        lines = ["".join(displayed_chars[i:i+6]) for i in range(0, TOTAL_CHARS, 6)]
        return "\n".join(lines)

    need = TOTAL_CHARS - len(displayed_chars)

    safety_steps = 2048
    while need > 0 and safety_steps > 0:
        x = np.array([ids], dtype="int32")
        preds = model.predict(x, verbose=0)
        next_probs = preds[0, -1]
        nid = sample_id_from_probs(next_probs, temperature)
        ids.append(nid)
        tok = id_to_token[nid]
        if is_cjk(tok):
            displayed_chars.append(tok)
            need -= 1
        safety_steps -= 1

    # 格式化为 4×6
    if len(displayed_chars) < TOTAL_CHARS:
        # 极端情况下补空格保持排版整齐（一般不会触发）
        displayed_chars += ["　"] * (TOTAL_CHARS - len(displayed_chars))
    lines = ["".join(displayed_chars[i:i+6]) for i in range(0, TOTAL_CHARS, 6)]
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

    btn.click(fn=generate_total_24, inputs=[seed, temp], outputs=[output], api_name="generate")

    # 页脚信息
    with gr.Row():
        gr.Markdown(
            f"**模型文件**：`{MODEL_PATH}` ｜ **词表**：`{VOCAB_PATH}` ｜ **词表大小**：{VOCAB_SIZE}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
