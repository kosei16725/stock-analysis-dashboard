"""列名や画面文言など、アプリ全体で共有する固定値。"""

DATE_COLUMN = "Date"
OPEN_COLUMN = "Open"
HIGH_COLUMN = "High"
LOW_COLUMN = "Low"
CLOSE_COLUMN = "Close"
ADJ_CLOSE_COLUMN = "Adj Close"
VOLUME_COLUMN = "Volume"

REQUIRED_PRICE_COLUMNS = (
    OPEN_COLUMN,
    HIGH_COLUMN,
    LOW_COLUMN,
    CLOSE_COLUMN,
    VOLUME_COLUMN,
)

DISCLAIMER = (
    "本アプリは学習・情報提供を目的としており、"
    "投資判断を推奨・保証するものではありません。"
)


def moving_average_column(window: int) -> str:
    """移動平均の列名を返す。

    Args:
        window: 移動平均を計算する営業日数。

    Returns:
        ``MA_20`` のような列名。

    Raises:
        ValueError: windowが1未満の場合。
    """
    if window < 1:
        raise ValueError("移動平均日数は1以上で指定してください。")
    return f"MA_{window}"
