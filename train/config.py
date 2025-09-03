from dataclasses import dataclass

from poem.genre import Genre


@dataclass
class Config:
    genre: Genre
    batch_size: int = 256
    epochs: int = 50
    embedding_dim: int = 100
    lstm_units: int = 512
    dropout_rate: float = 0.1
    dataset_number: int = 0
