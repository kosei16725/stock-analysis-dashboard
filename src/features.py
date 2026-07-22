"""株価の過去・当日情報から機械学習用データを作成する。"""

import numpy as np
import pandas as pd

from src.constants import (
    CLOSE_COLUMN,
    DAILY_RETURN_COLUMN,
    FEATURE_COLUMNS,
    MA_DEVIATION_COLUMN,
    RETURN_5_COLUMN,
    RETURN_20_COLUMN,
    TARGET_COLUMN,
    VOLATILITY_20_COLUMN,
    VOLUME_CHANGE_COLUMN,
    VOLUME_COLUMN,
    moving_average_column,
)


class FeatureEngineeringError(ValueError):
    """特徴量または目的変数を作成できない場合の例外。"""


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
        元データに9個の特徴量列を追加した、日付昇順のDataFrame。

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
