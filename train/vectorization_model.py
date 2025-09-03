import os

import pandas as pd
from keras import layers

from poem.genre import Genre


def save_vocabulary(vocabulary: list[str], genre: Genre):
    os.makedirs('models', exist_ok=True)
    vocab_path = f'models/{genre.name}_vocabulary.txt'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for token in vocabulary:
            f.write(f"{token}\n")
    print(f"[INFO] Vocabulary saved to: {vocab_path}")


def build_text_vectorization(poem_texts: pd.Series, poem_length: int):
    # --- Build TextVectorization ---
    text_vectorization = layers.TextVectorization(
        standardize=None,
        split='character',
        output_mode="int",
        output_sequence_length=poem_length
    )
    text_vectorization.adapt(poem_texts)

    print('Vocabulary size:', text_vectorization.vocabulary_size())
    vocab_list = text_vectorization.get_vocabulary()
    print('Vocabulary samples:', ''.join(vocab_list[:20]))

    # Encode & decode a sample
    encoded = text_vectorization(poem_texts.iloc[0])
    decoded = [vocab_list[i] for i in encoded]
    try:
        enc_np = encoded.numpy()
    except Exception:
        # In case of Eager/Graph mode differences
        enc_np = encoded
    print('Encoded:', enc_np)
    print('Decoded:', ''.join(decoded))

    return text_vectorization


def convert_to_tokens(poem_texts, genre: Genre):
    # --- Build TextVectorization ---
    poem_length = genre.length
    text_vectorization = build_text_vectorization(poem_texts, poem_length)
    vocab_list = text_vectorization.get_vocabulary()

    # Encode all poems
    train_token_ids = text_vectorization(poem_texts)
    print('shape of train dataset:', train_token_ids.shape)

    # Save vocabulary
    save_vocabulary(vocab_list, genre)

    # --- Prepare train/target sequences ---
    train_sequences = train_token_ids[:, :-1]
    target_sequences = train_token_ids[:, 1:]
    print("[INFO] train_sequences.shape =", train_sequences.shape,
          " target_sequences.shape =", target_sequences.shape)

    return train_sequences, target_sequences, text_vectorization
