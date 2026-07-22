"""未来情報を含まない特徴量と目的変数のテスト。"""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.constants import FEATURE_COLUMNS, TARGET_COLUMN
from src.features import add_target, create_features, create_training_data


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
