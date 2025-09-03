# ---- CLI: genre 必填，其余参数可选 ----
import argparse
from functools import partial

from poem.genre import Genre
from train.config import Config


def _float_0_1(option_name: str, x: str) -> float:
    v = float(x)
    if not (0.0 <= v < 1.0):
        raise argparse.ArgumentTypeError(f"{option_name} 必须在 [0.0, 1.0) 区间内")
    return v

def _non_neg_int(option_name: str, x: str) -> int:
    v = int(x)
    if v < 0:
        raise argparse.ArgumentTypeError(f"{option_name} 必须为非负整数")
    return v

def _parse_genre(s: str) -> Genre:
    s_raw = s.strip()
    s_up = s_raw.upper()
    # 允许直接用枚举名：WUJUE/QIJUE/WULV/QILV
    if s_up in getattr(Genre, "__members__", {}):
        return Genre[s_up]

    # 允许常用别名（大小写不敏感；含中文）
    s_l = s_raw.lower()
    alias = {
        "wujue": Genre.WUJUE,  "五绝": Genre.WUJUE,  "5jue": Genre.WUJUE,
        "qijue": Genre.QIJUE,  "七绝": Genre.QIJUE,  "7jue": Genre.QIJUE,
        "wulv":  Genre.WULV,   "五律": Genre.WULV,   "5lv":  Genre.WULV,  "wülv": Genre.WULV, "wulü": Genre.WULV,
        "qilv":  Genre.QILV,   "七律": Genre.QILV,   "7lv":  Genre.QILV,  "qülv": Genre.QILV, "qilü": Genre.QILV,
    }
    if s_l in alias:
        return alias[s_l]

    raise argparse.ArgumentTypeError(
        f"无法解析体裁: {s_raw}。可用值示例：WUJUE/五绝、QIJUE/七绝、WULV/五律、QILV/七律"
    )


def get_config_from_cli() -> Config:
    parser = argparse.ArgumentParser(
        description="诗歌 LSTM 训练参数（仅 --genre 必填，其余有默认值）"
    )
    parser.add_argument("-g", "--genre", required=True, type=_parse_genre,
                        help="体裁：可用 WUJUE/五绝，QIJUE/七绝，WULV/五律，QILV/七律 等")
    parser.add_argument("-b", "--batch-size", type=int, default=256,
                        help="Batch size（默认 256）")
    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="训练轮数（默认 50）")
    parser.add_argument("--embedding-dim", "--embed-dim", type=int, default=100,
                        help="Embedding 维度（默认 100）")
    parser.add_argument("-u", "--lstm-units", type=int, default=512,
                        help="LSTM 单元数（默认 512）")
    parser.add_argument("-p", "--dropout-rate", "--dropout", type=partial(_float_0_1, "dropout_rate"), default=0.1,
                        help="Dropout 比例，取值 [0,1)（默认 0.1）")
    parser.add_argument("-n", "--dataset-number", type=partial(_non_neg_int, "dataset_number"), default=0,
                        help="限制用于训练的数据条数（默认 0 表示不限制）")

    args = parser.parse_args()
    cfg = Config(**vars(args))
    return cfg
