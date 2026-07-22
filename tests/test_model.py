"""時系列分割とLightGBM学習・評価の通信非依存テスト。"""

import numpy as np
import pandas as pd
import pytest

from src.constants import FEATURE_COLUMNS, TARGET_COLUMN
from src.model import (
    calculate_metrics,
    run_model_pipeline,
    separate_features_target,
    split_time_series,
)


@pytest.fixture
def artificial_training_data() -> pd.DataFrame:
    """順序を逆転させた100営業日分の固定人工データを返す。"""
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=100, freq="B", name="Date")
    values = rng.normal(size=(100, len(FEATURE_COLUMNS)))
    data = pd.DataFrame(values, index=index, columns=FEATURE_COLUMNS)
    # 学習・テストの両期間に0と1が確実に含まれる固定Target。
    data[TARGET_COLUMN] = np.arange(100) % 2
    return data.sort_index(ascending=False)


def test_features_and_target_are_separated(
    artificial_training_data: pd.DataFrame,
) -> None:
    """Xとyが定数で指定した列へ正しく分離されることを確認する。"""
    X, y = separate_features_target(artificial_training_data)

    assert tuple(X.columns) == FEATURE_COLUMNS
    assert y.name == TARGET_COLUMN
    assert TARGET_COLUMN not in X.columns
    assert X.index.equals(y.index)


def test_target_05_raises_value_error(
    artificial_training_data: pd.DataFrame,
) -> None:
    """Targetの0.5がastypeで0へ切り捨てられず拒否されることを確認する。"""
    invalid_data = artificial_training_data.copy()
    invalid_data[TARGET_COLUMN] = invalid_data[TARGET_COLUMN].astype(float)
    invalid_data.iloc[0, invalid_data.columns.get_loc(TARGET_COLUMN)] = 0.5

    with pytest.raises(ValueError, match="厳密に0または1"):
        separate_features_target(invalid_data)


def test_target_19_raises_value_error(
    artificial_training_data: pd.DataFrame,
) -> None:
    """Targetの1.9がastypeで1へ切り捨てられず拒否されることを確認する。"""
    invalid_data = artificial_training_data.copy()
    invalid_data[TARGET_COLUMN] = invalid_data[TARGET_COLUMN].astype(float)
    invalid_data.iloc[0, invalid_data.columns.get_loc(TARGET_COLUMN)] = 1.9

    with pytest.raises(ValueError, match="厳密に0または1"):
        separate_features_target(invalid_data)


@pytest.mark.parametrize("invalid_target", [-1, 2])
def test_non_binary_integer_target_raises_value_error(
    artificial_training_data: pd.DataFrame,
    invalid_target: int,
) -> None:
    """Targetの二値以外の整数を拒否することを確認する。"""
    invalid_data = artificial_training_data.copy()
    invalid_data.iloc[0, invalid_data.columns.get_loc(TARGET_COLUMN)] = invalid_target

    with pytest.raises(ValueError, match="厳密に0または1"):
        separate_features_target(invalid_data)


def test_numeric_string_features_are_converted_without_mutating_input(
    artificial_training_data: pd.DataFrame,
) -> None:
    """数値文字列を数値型Xへ変換し、入力DataFrameは変更しないことを確認する。"""
    string_data = artificial_training_data.copy()
    string_data = string_data.assign(
        **{column: string_data[column].astype(str) for column in FEATURE_COLUMNS}
    )
    original_dtypes = string_data.dtypes.copy()

    X, _ = separate_features_target(string_data)
    result = run_model_pipeline(string_data)

    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
    assert all(
        pd.api.types.is_numeric_dtype(dtype) for dtype in result.X_train.dtypes
    )
    assert string_data.dtypes.equals(original_dtypes)


def test_string_zero_and_one_targets_are_accepted(
    artificial_training_data: pd.DataFrame,
) -> None:
    """意味を変えず数値化できるTarget文字列の0と1を許可する。"""
    string_data = artificial_training_data.copy()
    string_data[TARGET_COLUMN] = string_data[TARGET_COLUMN].astype(str)

    _, y = separate_features_target(string_data)

    assert y.dtype == np.dtype("int8")
    assert set(y.unique()) == {0, 1}


def test_non_numeric_feature_raises_value_error(
    artificial_training_data: pd.DataFrame,
) -> None:
    """数値へ変換できない特徴量文字列を拒否する。"""
    invalid_data = artificial_training_data.copy()
    invalid_data[FEATURE_COLUMNS[0]] = invalid_data[FEATURE_COLUMNS[0]].astype(object)
    invalid_data.iloc[0, 0] = "not-a-number"

    with pytest.raises(ValueError, match="数値へ変換できない"):
        separate_features_target(invalid_data)


@pytest.mark.parametrize("invalid_feature", [np.nan, np.inf, -np.inf])
def test_non_finite_feature_raises_value_error(
    artificial_training_data: pd.DataFrame,
    invalid_feature: float,
) -> None:
    """特徴量のNaNと正負の無限値を拒否する。"""
    invalid_data = artificial_training_data.copy()
    invalid_data.iloc[0, 0] = invalid_feature
    expected_message = "欠損値" if np.isnan(invalid_feature) else "無限値"

    with pytest.raises(ValueError, match=expected_message):
        separate_features_target(invalid_data)


@pytest.mark.parametrize("invalid_target", [np.nan, np.inf, -np.inf])
def test_non_finite_target_raises_value_error(
    artificial_training_data: pd.DataFrame,
    invalid_target: float,
) -> None:
    """TargetのNaNと正負の無限値を分かりやすく拒否する。"""
    invalid_data = artificial_training_data.copy()
    invalid_data[TARGET_COLUMN] = invalid_data[TARGET_COLUMN].astype(float)
    invalid_data.iloc[0, invalid_data.columns.get_loc(TARGET_COLUMN)] = invalid_target
    expected_message = "欠損値" if np.isnan(invalid_target) else "無限値"

    with pytest.raises(ValueError, match=expected_message):
        separate_features_target(invalid_data)


def test_split_preserves_time_order(
    artificial_training_data: pd.DataFrame,
) -> None:
    """逆順入力でも過去80%と未来20%に昇順分割されることを確認する。"""
    X, y = separate_features_target(artificial_training_data)
    X_train, X_test, y_train, y_test = split_time_series(X, y)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert X_train.index.is_monotonic_increasing
    assert X_test.index.is_monotonic_increasing
    assert X_train.index.max() < X_test.index.min()
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)


def test_model_predictions_and_probabilities_match_test_rows(
    artificial_training_data: pd.DataFrame,
) -> None:
    """予測件数と確率範囲がテストデータに対応することを確認する。"""
    result = run_model_pipeline(artificial_training_data)

    assert len(result.predictions) == len(result.X_test)
    assert len(result.probabilities) == len(result.X_test)
    assert np.all((result.probabilities >= 0) & (result.probabilities <= 1))


def test_metric_keys_are_expected(artificial_training_data: pd.DataFrame) -> None:
    """評価結果に指定された5指標が含まれることを確認する。"""
    result = run_model_pipeline(artificial_training_data)

    assert set(result.metrics) == {
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    }


def test_feature_importance_contains_every_feature(
    artificial_training_data: pd.DataFrame,
) -> None:
    """重要度が全特徴量を一度ずつ含み、降順であることを確認する。"""
    result = run_model_pipeline(artificial_training_data)
    importance = result.feature_importance

    assert set(importance["Feature"]) == set(FEATURE_COLUMNS)
    assert len(importance) == len(FEATURE_COLUMNS)
    assert importance["Importance"].is_monotonic_decreasing


def test_too_few_rows_raise_value_error(
    artificial_training_data: pd.DataFrame,
) -> None:
    """分割に必要な最低件数未満では分かりやすいValueErrorになる。"""
    X, y = separate_features_target(artificial_training_data.iloc[:9])

    with pytest.raises(ValueError, match="最低10行"):
        split_time_series(X, y)


def test_roc_auc_is_none_when_test_has_one_class() -> None:
    """正解が片方のクラスだけでも評価処理が停止しないことを確認する。"""
    y_true = pd.Series([1, 1, 1], dtype="int8")
    predictions = np.array([1, 0, 1], dtype="int8")
    probabilities = np.array([0.8, 0.4, 0.7])

    metrics = calculate_metrics(y_true, predictions, probabilities)

    assert metrics["ROC-AUC"] is None
    assert metrics["Accuracy"] == pytest.approx(2 / 3)
