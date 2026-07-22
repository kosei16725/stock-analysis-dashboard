"""株価データ取得・整形処理の通信に依存しないテスト。"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.constants import REQUIRED_PRICE_COLUMNS
from src.data_loader import StockDataError, fetch_stock_data, prepare_price_data


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """十分な件数を持つテスト用の日足DataFrameを返す。"""
    index = pd.date_range("2025-01-01", periods=60, freq="B", name="Date")
    return pd.DataFrame(
        {
            "Open": range(100, 160),
            "High": range(101, 161),
            "Low": range(99, 159),
            "Close": range(100, 160),
            "Adj Close": range(100, 160),
            "Volume": range(1_000, 1_060),
        },
        index=index,
    )


def test_prepare_price_data_has_required_columns(sample_price_data: pd.DataFrame) -> None:
    """正常データに必須列と移動平均列が存在することを確認する。"""
    result = prepare_price_data(sample_price_data, "7203.T")

    assert set(REQUIRED_PRICE_COLUMNS).issubset(result.columns)
    assert {"MA_20", "MA_50"}.issubset(result.columns)
    assert result["MA_20"].iloc[-1] == pytest.approx(sum(range(140, 160)) / 20)


def test_prepare_price_data_rejects_empty_dataframe() -> None:
    """空DataFrameを分かりやすい例外として検出する。"""
    with pytest.raises(StockDataError, match="0件"):
        prepare_price_data(pd.DataFrame(), "7203.T")


def test_prepare_price_data_rejects_missing_column(sample_price_data: pd.DataFrame) -> None:
    """必須列不足時に、不足列名を含む例外を返す。"""
    invalid_data = sample_price_data.drop(columns="Volume")

    with pytest.raises(StockDataError, match="Volume"):
        prepare_price_data(invalid_data, "7203.T")


@patch("src.data_loader.yf.download")
def test_fetch_stock_data_without_network(
    mock_download, sample_price_data: pd.DataFrame
) -> None:
    """yfinanceをモックし、通信なしで取得処理をテストする。"""
    mock_download.return_value = sample_price_data

    result = fetch_stock_data("7203.T", period="1y")

    assert len(result) == 60
    assert "MA_50" in result.columns
    mock_download.assert_called_once()


@patch("src.data_loader.yf.download", side_effect=ConnectionError("offline"))
def test_fetch_stock_data_wraps_network_error(mock_download) -> None:
    """通信例外が利用者向けのStockDataErrorへ変換されることを確認する。"""
    with pytest.raises(StockDataError, match="データ取得に失敗"):
        fetch_stock_data("7203.T")
