import os
from typing import List

import pandas as pd

from poem.genre import Genre
from train.config import Config

DATASET_DIRECTORY = 'data/Poetry/诗歌数据集'
VALID_PUNCTUATIONS = set("！，？。")


# ===== File utilities =====
def get_all_files(base_dir: str) -> List[str]:
    """
    Recursively collect all files under base_dir.
    """
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def read_file_to_pandas(file_path: str, genre_name: str) -> pd.Series:
    """
    Read a CSV file and return a Series of poem contents filtered by '体裁' containing genre_name.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    df = pd.read_csv(file_path)
    filter_by_genre = df[df["体裁"].astype(str).str.contains(genre_name, na=False)].copy()
    poems = filter_by_genre['内容']
    return poems


def extract_dynasty_from_filename(file_path: str) -> str:
    """
    Extract the dynasty from the file name.
    """
    base_name = os.path.basename(file_path)
    dynasty = base_name.split('.')[0]
    return dynasty


# ===== Checking functions =====
def check_poem_punctuation(text: str, positions: list[int], valid_punctuations=None) -> bool:
    """
    Check the punctuation in the fixed positions of the single poem text.

    :param text: string, the single poem text
    :param positions: list, the fixed positions to check
    :param valid_punctuations: set, optional, the valid punctuation characters

    :return: bool, whether the text is valid. If False, it means there are other characters
             found in the fixed positions.
    """
    if valid_punctuations is None:
        valid_punctuations = VALID_PUNCTUATIONS

    chars_at_fixed_positions = set(text[i] for i in positions if i < len(text))
    invalid_chars = chars_at_fixed_positions - valid_punctuations
    return not invalid_chars


def check_poem(text: str, rows: int, cols: int) -> bool:
    """
    Check if the poem text is valid by checking both punctuation and length.

    :param text: string, the single poem text
    :param rows: int, number of rows in the poem. E.g. 4 for 绝句
    :param cols: int, number of columns in the single poem row，not including the punctuation. E.g. 5 for 五言

    :return: bool, whether the text is valid
    """
    # Guard: some items may not be string, e.g. float('nan')
    if type(text) is not str:
        return False

    punctuation_positions = [(i + 1) * (cols + 1) - 1 for i in range(rows)]
    poem_length = rows * (cols + 1)

    return len(text) == poem_length and check_poem_punctuation(text, punctuation_positions)


def check_poems(poem_texts: pd.Series, genre: Genre) -> pd.Series:
    """
    Check the poems Series using the above checking functions and return the mask values (True/False) for each poem text.

    :param poem_texts: pd.Series, the series of poem texts, it supports MultiIndex
    :param genre: Genre(Enum), the genre rule applied to check
    :return: pd.Series of bool, the mask values for each poem text, with the same index as input
    """
    from functools import partial

    check_for_current_genre = partial(check_poem, rows=genre.rows, cols=genre.cols)
    # Slice to length BEFORE checking, consistent with the notebook
    mask = poem_texts.str[:genre.length].apply(check_for_current_genre)
    return mask


def report_check_results(mask: pd.Series):
    return len(mask), int(mask.sum()), f"{mask.mean() * 100:.2f}%"


def read_poem_text(config: Config):
    current_genre = config.genre

    # --- Collect files ---
    base_dir = os.path.expanduser(DATASET_DIRECTORY)
    poem_files = get_all_files(base_dir)  # format #{DATASET_DIRECTORY}/XXX.txt
    if not poem_files:
        raise RuntimeError(f"No files found under {base_dir}")

    # --- Demo: single file read & quick checks ---
    demo_file = poem_files[16] if len(poem_files) > 16 else poem_files[0]
    print(f"[INFO] Demo file: {demo_file}")

    one_dynasty_poems = read_file_to_pandas(demo_file, current_genre.genre_name)
    print("[INFO] one_dynasty_poems.shape =", one_dynasty_poems.shape)

    # validate one sample
    sample_ok = check_poem(one_dynasty_poems.iloc[0], current_genre.rows, current_genre.cols)
    print("[INFO] first poem valid? ->", sample_ok)

    mask = check_poems(one_dynasty_poems, current_genre)
    print("[INFO] demo dynasty report:", report_check_results(mask))

    # --- Read ALL files into a MultiIndex Series ---
    list_of_poems = [read_file_to_pandas(file_path, current_genre.genre_name) for file_path in poem_files]
    dynasty_list = [extract_dynasty_from_filename(file) for file in poem_files]
    all_dynasty_poems = pd.concat(list_of_poems, keys=dynasty_list)

    # --- Apply checks across all ---
    mask_all = check_poems(all_dynasty_poems, current_genre)
    total, passed, ratio = report_check_results(mask_all)
    print(f"[INFO] all dynasties report: total={total}, passed={passed}, ratio={ratio}")

    cleaned_poems = all_dynasty_poems[mask_all].str[:current_genre.length]
    print("[INFO] cleaned_poems.shape =", cleaned_poems.shape)

    # --- Only select specific number data samples ---
    if config.dataset_number > 0:
        train_poem = cleaned_poems.sample(n=config.dataset_number, random_state=42)
    else:
        train_poem = cleaned_poems.sample(frac=1.0, random_state=42)
    print(f"[INFO] After sampling, actual train_poems.shape = {train_poem.shape}")

    return train_poem
