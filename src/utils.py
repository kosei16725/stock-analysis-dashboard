"""入力検証などの小さな共通処理を提供する。"""

import re


def validate_ticker(ticker: str) -> str:
    """銘柄コードを検証し、前後の空白を除いて返す。

    Args:
        ticker: Yahoo Finance形式の銘柄コード。

    Returns:
        検証済みの銘柄コード。

    Raises:
        ValueError: 空文字または利用できない文字を含む場合。
    """
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("銘柄コードを入力してください。")
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,20}", normalized):
        raise ValueError("銘柄コードの形式が正しくありません。")
    return normalized
