"""Yahoo Financeから株価を取得し、表示用に整形する。"""

from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from src.constants import CLOSE_COLUMN, REQUIRED_PRICE_COLUMNS, moving_average_column
from src.utils import validate_ticker


class StockDataError(RuntimeError):
    """株価データの取得または検証に失敗した場合の例外。"""


def normalize_price_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinanceの列を単一階層へ統一する。

    Args:
        data: yfinanceから返されたDataFrame。
        ticker: 取得対象の銘柄コード。

    Returns:
        単一階層の列を持つDataFrameのコピー。

    Raises:
        StockDataError: MultiIndexから対象銘柄を特定できない場合。
    """
    normalized = data.copy()
    if not isinstance(normalized.columns, pd.MultiIndex):
        return normalized

    # yfinanceのバージョン差により、銘柄が第1・第2階層のどちらにもなり得る。
    for level in range(normalized.columns.nlevels):
        if ticker in normalized.columns.get_level_values(level):
            return normalized.xs(ticker, axis=1, level=level, drop_level=True).copy()

    raise StockDataError("取得データの列から対象銘柄を特定できませんでした。")


def validate_price_data(data: pd.DataFrame) -> None:
    """株価DataFrameが第1段階の処理に利用できるか検証する。

    Args:
        data: 検証対象の株価DataFrame。

    Returns:
        なし。

    Raises:
        StockDataError: データが空、または必須列が不足している場合。
    """
    if data.empty:
        raise StockDataError("株価データが0件でした。銘柄コードや期間を確認してください。")

    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in data.columns]
    if missing:
        raise StockDataError(f"株価データに必要な列がありません: {', '.join(missing)}")


def prepare_price_data(
    data: pd.DataFrame,
    ticker: str,
    moving_average_windows: Sequence[int] = (20, 50),
) -> pd.DataFrame:
    """株価データを検証し、移動平均を追加する。

    Args:
        data: yfinance形式の株価DataFrame。
        ticker: 対象銘柄コード。
        moving_average_windows: 移動平均の日数。

    Returns:
        日付順で、数値列と移動平均を持つDataFrame。

    Raises:
        StockDataError: データが空、列不足、または有効な終値がない場合。
        ValueError: 銘柄コードや移動平均日数が不正な場合。
    """
    checked_ticker = validate_ticker(ticker)
    prepared = normalize_price_columns(data, checked_ticker)
    validate_price_data(prepared)

    prepared = prepared.sort_index()
    for column in REQUIRED_PRICE_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.replace([np.inf, -np.inf], np.nan)

    if prepared[CLOSE_COLUMN].notna().sum() == 0:
        raise StockDataError("有効な終値が含まれていません。")

    for window in moving_average_windows:
        column_name = moving_average_column(window)
        prepared[column_name] = prepared[CLOSE_COLUMN].rolling(
            window=window, min_periods=window
        ).mean()
    return prepared


def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    moving_average_windows: Sequence[int] = (20, 50),
) -> pd.DataFrame:
    """Yahoo Financeから日足データを取得して移動平均を追加する。

    Args:
        ticker: Yahoo Finance形式の銘柄コード。
        period: ``1y`` などの取得期間。
        moving_average_windows: 移動平均の日数。

    Returns:
        整形済みの日足株価DataFrame。

    Raises:
        ValueError: 入力値が不正な場合。
        StockDataError: 通信、取得、またはデータ検証に失敗した場合。
    """
    checked_ticker = validate_ticker(ticker)
    if not period or not period.strip():
        raise ValueError("取得期間を指定してください。")

    try:
        raw_data = yf.download(
            checked_ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise StockDataError(f"Yahoo Financeからのデータ取得に失敗しました: {exc}") from exc

    return prepare_price_data(raw_data, checked_ticker, moving_average_windows)
