import os

import keras
from keras import layers, models

from poem.generator import PoemGenerator
from train.config import Config


def build_model(config: Config, vocab_size: int) -> keras.Model:
    inputs = keras.Input(shape=(None,), dtype="int32", name="inputs")
    x = layers.Embedding(input_dim=vocab_size, output_dim=config.embedding_dim, name="embedding")(inputs)
    x = layers.LSTM(config.lstm_units, return_sequences=True, name="lstm")(x)
    x = layers.Dropout(config.dropout_rate, name="dropout")(x)
    outputs = layers.Dense(vocab_size, activation="softmax", name="output")(x)
    return models.Model(inputs=inputs, outputs=outputs, name="lstm_decoder")


def train_model(train_sequences, target_sequences, text_vectorization: layers.TextVectorization, config: Config):
    vocab_list = text_vectorization.get_vocabulary()
    genre = config.genre

    # --- Build model (simple LSTM decoder) ---
    model = build_model(config, len(vocab_list))
    model.summary()

    # --- Train model ---
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.fit(
        train_sequences,
        target_sequences,
        batch_size=config.batch_size,
        epochs=config.epochs
    )

    # --- Save model ---
    os.makedirs('models', exist_ok=True)
    model_path = f'models/{genre.name}_lstm_model-epoch{config.epochs}.keras'
    model.save(model_path)
    print(f"[INFO] Model saved to: {model_path}")

    # --- Generate sample text (demo) ---
    poem_generator = PoemGenerator(
        vectorization_model=text_vectorization,
        generation_model=model,
        genre=genre
    )
    demo_output = poem_generator.generate("海外", temperature=0)
    print("[DEMO] Generated poem:", demo_output)
