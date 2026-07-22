"""未来情報を含まない特徴量と目的変数のテスト。"""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.constants import (
    BB_LOWER_20_COLUMN,
    BB_PERCENT_B_20_COLUMN,
    BB_STD_20_COLUMN,
    BB_UPPER_20_COLUMN,
    BB_WIDTH_20_COLUMN,
    EMA_12_COLUMN,
    EMA_26_COLUMN,
    FEATURE_COLUMNS,
    MACD_COLUMN,
    MACD_HISTOGRAM_COLUMN,
    MACD_SIGNAL_COLUMN,
    RSI_14_COLUMN,
    TARGET_COLUMN,
)
from src.features import (
    add_target,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_rsi,
    create_features,
    create_training_data,
)


@pytest.fixture
def price_data() -> pd.DataFrame:
    """特徴量計算に十分な80営業日分のテストデータを返す。"""
    index = pd.date_range("2025-01-01", periods=80, freq="B", name="Date")
    return pd.DataFrame(
        {
            "Close": np.arange(100.0, 180.0),
            "Volume": np.arange(1_000.0, 1_080.0),
            "Optional": np.nan,
        },
        index=index,
    )


def test_target_is_created_from_next_business_day(price_data: pd.DataFrame) -> None:
    """上昇時は1、非上昇時は0、翌日がない行は欠損になることを確認する。"""
    data = price_data.iloc[:3].copy()
    data["Close"] = [100.0, 110.0, 105.0]

    result = add_target(data)

    assert result[TARGET_COLUMN].iloc[0] == 1
    assert result[TARGET_COLUMN].iloc[1] == 0
    assert pd.isna(result[TARGET_COLUMN].iloc[2])


def test_features_do_not_use_future_values(price_data: pd.DataFrame) -> None:
    """将来行を変更しても、それ以前の特徴量が変わらないことを確認する。"""
    changed_future = price_data.copy()
    changed_future.loc[changed_future.index[70]:, "Close"] *= 10
    changed_future.loc[changed_future.index[70]:, "Volume"] *= 10

    original_features = create_features(price_data)
    changed_features = create_features(changed_future)
    comparison_index = price_data.index[:70]

    pdt.assert_frame_equal(
        original_features.loc[comparison_index, list(FEATURE_COLUMNS)],
        changed_features.loc[comparison_index, list(FEATURE_COLUMNS)],
    )


def test_training_data_has_no_missing_target(price_data: pd.DataFrame) -> None:
    """欠損処理後のTargetに欠損値がなく、最終行が除外されることを確認する。"""
    result = create_training_data(price_data)

    assert not result[TARGET_COLUMN].isna().any()
    assert result.index.max() == price_data.index[-2]
    # 学習に使わないOptional列の欠損だけでは、追加の行削除をしない。
    assert result["Optional"].isna().all()


def test_feature_column_names_are_expected(price_data: pd.DataFrame) -> None:
    """作成される特徴量列名と順序が定義どおりであることを確認する。"""
    result = create_features(price_data)
    actual_feature_columns = tuple(
        column for column in result.columns if column in FEATURE_COLUMNS
    )

    assert actual_feature_columns == FEATURE_COLUMNS


def test_rsi_is_between_zero_and_one_hundred(price_data: pd.DataFrame) -> None:
    """計算可能な14日RSIが必ず0以上100以下になることを確認する。"""
    oscillating = price_data["Close"] + np.sin(np.arange(len(price_data))) * 5
    rsi = calculate_rsi(oscillating).dropna()

    assert not rsi.empty
    assert rsi.between(0, 100).all()


def test_rsi_is_high_for_rising_prices_and_low_for_falling_prices() -> None:
    """一方向の上昇系列ではRSIが高く、下落系列では低くなる。"""
    index = pd.date_range("2025-01-01", periods=40, freq="B")
    rising = pd.Series(np.arange(100.0, 140.0), index=index)
    falling = pd.Series(np.arange(140.0, 100.0, -1.0), index=index)

    assert calculate_rsi(rising).iloc[-1] > 70
    assert calculate_rsi(falling).iloc[-1] < 30


def test_rsi_uses_arithmetic_seed_and_wilder_recursion() -> None:
    """混合系列で最初の算術平均と次時点のWilder再帰式を確認する。"""
    close = pd.Series(
        [
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
            46.00,
        ],
        index=pd.date_range("2025-01-01", periods=16, freq="B"),
    )
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    initial_gain = gains.iloc[1:15].mean()
    initial_loss = losses.iloc[1:15].mean()
    expected_first = 100 * initial_gain / (initial_gain + initial_loss)
    next_gain = gains.iloc[15]
    next_loss = losses.iloc[15]
    recursive_gain = (initial_gain * 13 + next_gain) / 14
    recursive_loss = (initial_loss * 13 + next_loss) / 14
    expected_next = 100 * recursive_gain / (recursive_gain + recursive_loss)

    result = calculate_rsi(close)

    assert result.iloc[:14].isna().all()
    assert result.iloc[14] == pytest.approx(expected_first)
    assert result.iloc[15] == pytest.approx(expected_next)

    # 旧実装の単純なewmシードなら異なる値となり、回帰を検出できる。
    ewm_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ewm_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ewm_rsi = 100 * ewm_gain / (ewm_gain + ewm_loss)
    assert not np.isclose(result.iloc[14], ewm_rsi.iloc[14])


def test_macd_signal_and_histogram_relationship(price_data: pd.DataFrame) -> None:
    """MACDとSignal、Histogramが定義どおりの関係になる。"""
    result = calculate_macd(price_data["Close"]).dropna()

    assert not result.empty
    assert np.allclose(
        result[MACD_COLUMN],
        result[EMA_12_COLUMN] - result[EMA_26_COLUMN],
    )
    assert np.allclose(
        result[MACD_HISTOGRAM_COLUMN],
        result[MACD_COLUMN] - result[MACD_SIGNAL_COLUMN],
    )


def test_bollinger_band_order(price_data: pd.DataFrame) -> None:
    """Bollinger Upper、Middle、Lowerの大小関係を確認する。"""
    result = calculate_bollinger_bands(price_data["Close"])
    valid = result.dropna()
    middle = price_data["Close"].rolling(20, min_periods=20).mean().loc[valid.index]

    assert (valid[BB_UPPER_20_COLUMN] >= middle).all()
    assert (middle >= valid[BB_LOWER_20_COLUMN]).all()


def test_bollinger_percent_b_and_band_width(price_data: pd.DataFrame) -> None:
    """%BとBand Widthが20日母標準偏差から正しく計算される。"""
    close = price_data["Close"]
    result = calculate_bollinger_bands(close)
    date = close.index[-1]
    middle = close.iloc[-20:].mean()
    standard_deviation = close.iloc[-20:].std(ddof=0)
    upper = middle + 2 * standard_deviation
    lower = middle - 2 * standard_deviation

    assert result.loc[date, BB_STD_20_COLUMN] == pytest.approx(standard_deviation)
    assert result.loc[date, BB_WIDTH_20_COLUMN] == pytest.approx(
        (upper - lower) / middle
    )
    assert result.loc[date, BB_PERCENT_B_20_COLUMN] == pytest.approx(
        (close.loc[date] - lower) / (upper - lower)
    )


def test_technical_features_avoid_zero_division_and_infinity() -> None:
    """値動きがない系列でもRSIとBollinger指標に無限値を作らない。"""
    index = pd.date_range("2025-01-01", periods=80, freq="B")
    data = pd.DataFrame(
        {"Close": np.full(80, 100.0), "Volume": np.full(80, 1_000.0)},
        index=index,
    )
    result = create_features(data)
    technical_columns = [
        RSI_14_COLUMN,
        EMA_12_COLUMN,
        EMA_26_COLUMN,
        MACD_COLUMN,
        MACD_SIGNAL_COLUMN,
        MACD_HISTOGRAM_COLUMN,
        BB_STD_20_COLUMN,
        BB_UPPER_20_COLUMN,
        BB_LOWER_20_COLUMN,
        BB_WIDTH_20_COLUMN,
        BB_PERCENT_B_20_COLUMN,
    ]

    assert not np.isinf(result[technical_columns].to_numpy(dtype=float)).any()
    assert result[RSI_14_COLUMN].dropna().eq(50.0).all()
    assert result[BB_WIDTH_20_COLUMN].dropna().eq(0.0).all()
    assert result[BB_PERCENT_B_20_COLUMN].dropna().eq(0.5).all()


def test_create_features_does_not_mutate_input(price_data: pd.DataFrame) -> None:
    """特徴量作成後も入力DataFrameが変更されないことを確認する。"""
    original = price_data.copy(deep=True)

    create_features(price_data)

    pdt.assert_frame_equal(price_data, original)
