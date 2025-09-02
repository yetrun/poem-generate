from enum import unique, Enum

@unique
class Genre(Enum):
    WUJUE = ("五言绝句", 4, 5)
    QIJUE = ("七言绝句", 4, 7)
    WULV = ("五言律诗", 8, 5)
    QILV = ("七言律诗", 8, 7)

    def __init__(self, genre_name: str, rows: int, cols: int):
        self.genre_name = genre_name
        self.rows = rows
        self.cols = cols

    @property
    def length(self):
        return self.rows * (self.cols + 1)

if __name__ == "__main__":
    print(Genre['WUJUE'])
