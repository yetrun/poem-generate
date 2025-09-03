# -*- coding: utf-8 -*-
"""
Poetry LSTM Train Pipeline

This script reproduces the end-to-end workflow:
- Load dataset files
- Clean & validate poem texts by genre rules
- Build TextVectorization and save its vocabulary
- Build, train, and save an LSTM decoder model
- Generate a sample poem using the trained model

Requirements:
  - pandas, numpy, tensorflow / keras
  - Local modules:
        from poem.genre import Genre
        from poem.generator import PoemGenerator

Run:
  python train.py -g WUJUE
"""

from train.parse_args import get_config_from_cli
from train.read_dataset import read_poem_text
from train.vectorization_model import convert_to_tokens
from train.generation_model import train_model


def main():
    config = get_config_from_cli()
    train_poem = read_poem_text(config)
    train_sequences, target_sequences, text_vectorization = convert_to_tokens(train_poem, config.genre)
    train_model(train_sequences, target_sequences, text_vectorization, config)


if __name__ == "__main__":
    main()
