"""株価の過去・当日情報から機械学習用データを作成する。"""

import numpy as np
import pandas as pd

from src.constants import (
    BB_LOWER_20_COLUMN,
    BB_PERCENT_B_20_COLUMN,
    BB_STD_20_COLUMN,
    BB_UPPER_20_COLUMN,
    BB_WIDTH_20_COLUMN,
    CLOSE_COLUMN,
    DAILY_RETURN_COLUMN,
    EMA_12_COLUMN,
    EMA_26_COLUMN,
    FEATURE_COLUMNS,
    MACD_COLUMN,
    MACD_HISTOGRAM_COLUMN,
    MACD_SIGNAL_COLUMN,
    MA_DEVIATION_COLUMN,
    RETURN_5_COLUMN,
    RETURN_20_COLUMN,
    RSI_14_COLUMN,
    TARGET_COLUMN,
    VOLATILITY_20_COLUMN,
    VOLUME_CHANGE_COLUMN,
    VOLUME_COLUMN,
    moving_average_column,
)


class FeatureEngineeringError(ValueError):
    """特徴量または目的変数を作成できない場合の例外。"""


def _calculate_wilder_average(values: pd.Series, period: int) -> pd.Series:
    """最初の算術平均をシードとしてWilderの再帰平均を計算する。

    Args:
        values: 日付昇順のGainまたはLoss。
        period: 算術平均と再帰更新に用いる期間。

    Returns:
        最初のperiod個を算術平均し、以降をWilder式で更新したSeries。
        有効値がperiod個連続するまでNaN。

    Raises:
        ValueError: periodが1未満の場合。
    """
    if period < 1:
        raise ValueError("Wilder平均の期間は1以上で指定してください。")

    result = pd.Series(np.nan, index=values.index, dtype=float)
    previous_average = np.nan
    for position in range(period, len(values)):
        current_value = values.iloc[position]
        if np.isnan(previous_average):
            initial_values = values.iloc[position - period + 1: position + 1]
            if initial_values.notna().all():
                previous_average = float(initial_values.mean())
            else:
                continue
        elif pd.isna(current_value):
            # 欠損を未来方向へ補完せず、再びperiod個そろうまで計算しない。
            previous_average = np.nan
            continue
        else:
            previous_average = (
                previous_average * (period - 1) + float(current_value)
            ) / period
        result.iloc[position] = previous_average
    return result


def calculate_rsi(close: pd.Series) -> pd.Series:
    """算術平均で初期化した厳密なWilder方式でRSIを計算する。

    最初の14個の上昇幅・下落幅をそれぞれ算術平均する。以降は前日の
    平均を13倍して当日の値を加え、14で割る再帰式を使用する。
    値動きが全くない場合は中立値50とする。

    Args:
        close: 日付昇順の終値Series。

    Returns:
        0以上100以下のRSI。計算期間に満たない先頭行はNaN。

    Raises:
        なし。
    """
    period = 14
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    average_gain = _calculate_wilder_average(gains, period)
    average_loss = _calculate_wilder_average(losses, period)
    total_movement = average_gain + average_loss
    rsi = 100.0 * average_gain / total_movement.where(total_movement != 0)
    rsi = rsi.mask(total_movement == 0, 50.0).clip(lower=0.0, upper=100.0)
    rsi.name = RSI_14_COLUMN
    return rsi


def calculate_macd(
    close: pd.Series,
) -> pd.DataFrame:
    """終値のEMAからMACD、Signal、Histogramを計算する。

    Args:
        close: 日付昇順の終値Series。

    Returns:
        EMA12、EMA26、MACD、Signal、Histogramを持つDataFrame。

    Raises:
        なし。
    """
    short_span, long_span, signal_span = 12, 26, 9
    ema_short = close.ewm(
        span=short_span, adjust=False, min_periods=short_span
    ).mean()
    ema_long = close.ewm(
        span=long_span, adjust=False, min_periods=long_span
    ).mean()
    macd = ema_short - ema_long
    signal = macd.ewm(
        span=signal_span, adjust=False, min_periods=signal_span
    ).mean()
    histogram = macd - signal
    return pd.DataFrame(
        {
            EMA_12_COLUMN: ema_short,
            EMA_26_COLUMN: ema_long,
            MACD_COLUMN: macd,
            MACD_SIGNAL_COLUMN: signal,
            MACD_HISTOGRAM_COLUMN: histogram,
        },
        index=close.index,
    )


def calculate_bollinger_bands(
    close: pd.Series,
) -> pd.DataFrame:
    """20日移動平均と母標準偏差からBollinger Bandsを計算する。

    Args:
        close: 日付昇順の終値Series。

    Returns:
        標準偏差、Upper、Lower、Band Width、%Bを持つDataFrame。
        Middleは既存のMA_20と同じため重複して返さない。バンド幅が0の%Bは
        中立値0.5とする。

    Raises:
        なし。
    """
    window, standard_deviations = 20, 2.0
    middle = close.rolling(window=window, min_periods=window).mean()
    standard_deviation = close.rolling(
        window=window, min_periods=window
    ).std(ddof=0)
    upper = middle + standard_deviations * standard_deviation
    lower = middle - standard_deviations * standard_deviation
    band_range = upper - lower
    band_width = band_range / middle.where(middle != 0)
    percent_b = (close - lower) / band_range.where(band_range != 0)
    percent_b = percent_b.mask((band_range == 0) & band_range.notna(), 0.5)

    return pd.DataFrame(
        {
            BB_STD_20_COLUMN: standard_deviation,
            BB_UPPER_20_COLUMN: upper,
            BB_LOWER_20_COLUMN: lower,
            BB_WIDTH_20_COLUMN: band_width,
            BB_PERCENT_B_20_COLUMN: percent_b,
        },
        index=close.index,
    )


def validate_feature_input(data: pd.DataFrame) -> None:
    """特徴量作成に必要な入力データを検証する。

    Args:
        data: 日付をインデックス、株価を列に持つDataFrame。

    Returns:
        なし。

    Raises:
        FeatureEngineeringError: データが空、またはClose・Volume列がない場合。
    """
    if data.empty:
        raise FeatureEngineeringError("特徴量を作成する株価データが空です。")

    required_columns = (CLOSE_COLUMN, VOLUME_COLUMN)
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise FeatureEngineeringError(
            f"特徴量作成に必要な列がありません: {', '.join(missing)}"
        )


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """当日までに利用できる価格・出来高から特徴量を作成する。

    リターンは現在値と過去値、移動平均とボラティリティは末尾が当日の
    rolling windowを使用するため、翌営業日以降の情報は参照しない。

    Args:
        data: Close列とVolume列を含む日足DataFrame。

    Returns:
        元データに20個の特徴量列を追加した、日付昇順のDataFrame。

    Raises:
        FeatureEngineeringError: 入力が空、または必要列が不足している場合。
    """
    validate_feature_input(data)
    featured = data.sort_index().copy()
    close = pd.to_numeric(featured[CLOSE_COLUMN], errors="coerce")
    volume = pd.to_numeric(featured[VOLUME_COLUMN], errors="coerce")

    # pct_changeは当日と指定日前の値だけを比較し、欠損値を自動補完しない。
    featured[DAILY_RETURN_COLUMN] = close.pct_change(periods=1, fill_method=None)
    featured[RETURN_5_COLUMN] = close.pct_change(periods=5, fill_method=None)
    featured[RETURN_20_COLUMN] = close.pct_change(periods=20, fill_method=None)

    for window in (5, 20, 50):
        column = moving_average_column(window)
        featured[column] = close.rolling(window=window, min_periods=window).mean()

    featured[MA_DEVIATION_COLUMN] = (
        close / featured[moving_average_column(20)] - 1
    )
    featured[VOLATILITY_20_COLUMN] = featured[DAILY_RETURN_COLUMN].rolling(
        window=20, min_periods=20
    ).std()
    featured[VOLUME_CHANGE_COLUMN] = volume.pct_change(periods=1, fill_method=None)
    featured[RSI_14_COLUMN] = calculate_rsi(close)

    macd_features = calculate_macd(close)
    featured.loc[:, macd_features.columns] = macd_features

    bollinger_features = calculate_bollinger_bands(close)
    featured.loc[:, bollinger_features.columns] = bollinger_features

    # 0除算などで生じる無限値は、後段で欠損値として必要な行だけ除外する。
    featured.loc[:, FEATURE_COLUMNS] = featured.loc[:, FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )
    return featured


def add_target(data: pd.DataFrame) -> pd.DataFrame:
    """翌営業日の終値が上昇するかを目的変数Targetとして追加する。

    翌営業日の終値はTargetの作成だけに使用する。翌営業日が存在しない
    最終行は、下落と誤判定せず欠損値のまま残す。

    Args:
        data: Close列を含む日付昇順のDataFrame。

    Returns:
        Target列を追加したDataFrameのコピー。Targetは0、1または欠損値。

    Raises:
        FeatureEngineeringError: Close列が存在しない場合。
    """
    if CLOSE_COLUMN not in data.columns:
        raise FeatureEngineeringError(f"目的変数作成に必要な{CLOSE_COLUMN}列がありません。")

    targeted = data.copy()
    close = pd.to_numeric(targeted[CLOSE_COLUMN], errors="coerce")
    next_close = close.shift(-1)
    valid_target = close.notna() & next_close.notna()

    target = pd.Series(pd.NA, index=targeted.index, dtype="Int8")
    target.loc[valid_target] = (
        next_close.loc[valid_target] > close.loc[valid_target]
    ).astype("int8")
    targeted[TARGET_COLUMN] = target
    return targeted


def create_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """特徴量とTargetがそろった学習用DataFrameを作成する。

    Args:
        data: Close列とVolume列を含む日足DataFrame。

    Returns:
        全特徴量とTargetが欠損していない行だけを残したDataFrame。
        元の価格列にある、学習に使わない欠損値だけでは行を削除しない。

    Raises:
        FeatureEngineeringError: 入力が不正、または有効な学習行がない場合。
    """
    featured = create_features(data)
    targeted = add_target(featured)

    # rollingの先頭行と翌日がない最終行など、学習に必要な列だけで判定する。
    training_data = targeted.dropna(
        subset=[*FEATURE_COLUMNS, TARGET_COLUMN]
    ).copy()
    if training_data.empty:
        raise FeatureEngineeringError(
            "特徴量とTargetがそろう行がありません。50営業日を超えるデータを指定してください。"
        )

    training_data[TARGET_COLUMN] = training_data[TARGET_COLUMN].astype("int8")
    return training_data
