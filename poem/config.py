import os
import json
from keras import Model, models, layers
from poem.genre import Genre

# -------- 词表加载 --------
def load_text_vectorization(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到词表：{path}")
    with open(path, "r", encoding="utf-8") as f:
        vocab = [line.rstrip("\n") for line in f]
    if not vocab:
        raise ValueError("词表为空")

    tv = layers.TextVectorization(
        standardize=None,
        split='character',
        output_mode="int",
        output_sequence_length=None,
        ragged = True
    )
    tv.set_vocabulary(vocab)
    return tv

class PoemConfig:
    def __init__(self, genre: Genre, vectorization_model: layers.TextVectorization,
                 generation_model: Model, meta: dict):
        self.genre = genre
        self.vectorization_model = vectorization_model
        self.generation_model = generation_model
        self.meta = meta

    @staticmethod
    def from_config(config: dict):
        genre_name = config['genre']
        genre = Genre[genre_name]

        vocabulary_path = config['vocabulary_path']
        vectorization_model = load_text_vectorization(vocabulary_path)

        model_path = config['model_path']
        generation_model = models.load_model(model_path, compile=False)

        return PoemConfig(
            genre=genre,
            vectorization_model=vectorization_model,
            generation_model=generation_model,
            meta=config
        )

def read_configs():
    config_path = "poem_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件：{config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dicts = json.load(f)

    configs = [PoemConfig.from_config(cfg) for cfg in config_dicts]
    return configs

if __name__ == "__main__":
    configs = read_configs()
    print(configs[0].meta)