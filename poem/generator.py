import numpy as np
from keras import Model, layers

from poem.genre import Genre


# -------- 采样函数 --------
def sampling(predictions: list, temperature=1.0, eps1=1e-20, eps2=1e-9) -> int:
    p = np.asarray(predictions, dtype=np.float64)

    # The two key points: log(p + eps1) divide by (T + eps2)
    logits = np.log(p + eps1) / (float(temperature) + eps2)

    # Subtract the max logit to prevent overflow
    logits -= np.max(logits)

    q = np.exp(logits)
    q /= q.sum()
    return int(np.random.choice(len(q), p=q))


class PoemGenerator:
    def __init__(self, genre: Genre, vectorization_model: layers.TextVectorization, generation_model: Model):
        self.genre = genre
        self.vectorization_model = vectorization_model
        self.generation_model = generation_model

    def generate_and_format(self, prompt: str, temperature: float = 1.0) -> str:
        poem_text = self.generate(prompt, temperature)
        poem_cols = self.genre.cols
        poem_length = self.genre.length

        lines = ["".join(poem_text[i:i + poem_cols + 1]) for i in range(0, poem_length, poem_cols + 1)]
        return "\n".join(lines)

    def generate(self, prompt: str, temperature: float = 1.0) -> str:
        """
        Generate a poem based on the start prompt

        Returns:
            A generated poem as a string.
        """
        poem_length = self.genre.length
        prompt_ids = self.vectorization_model(prompt)[:len(prompt)]
        generated = list(prompt_ids) # cast to list for appending
        while len(generated) < poem_length:
            input_sequence = np.array(generated).reshape(1, -1)
            predictions = self.generation_model.predict(input_sequence, verbose=0)[0]
            next_token_id = sampling(predictions[-1], temperature)
            generated.append(next_token_id)
        return ''.join(self.vectorization_model.get_vocabulary()[token_id] for token_id in generated)

if __name__ == "__main__":
    from config import PoemConfig

    # Example usage
    config = PoemConfig.from_config({
        'genre': 'WUJUE',
        'vocabulary_path': "models/poetry_vocabulary.txt",
        'model_path': "models/lstm_poetry_model-epoch50.keras"
    })

    poem_generator = PoemGenerator(
        genre=config.genre,
        vectorization_model=config.vectorization_model,
        generation_model=config.generation_model,
    )
    prompt = "春风"
    temperature = 1.0
    poem = poem_generator.generate_and_format(prompt, temperature)
    print(poem)