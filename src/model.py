"""時系列分割、LightGBMの学習・予測・評価を提供する。"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.constants import FEATURE_COLUMNS, TARGET_COLUMN

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
MINIMUM_DATA_ROWS = 10

MetricValues = Dict[str, Optional[float]]
DataSplit = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


@dataclass(frozen=True)
class ModelResult:
    """1回の時系列分割による学習・評価結果。"""

    model: LGBMClassifier
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    predictions: np.ndarray
    probabilities: np.ndarray
    metrics: MetricValues
    feature_importance: pd.DataFrame


def separate_features_target(
    training_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """学習用DataFrameを特徴量Xと目的変数yへ分離する。

    Args:
        training_data: FEATURE_COLUMNSとTargetを含むDataFrame。

    Returns:
        定義済みの順序を保った特徴量Xと、整数型の目的変数y。

    Raises:
        ValueError: データが空、必要列不足、欠損、またはTargetが二値でない場合。
    """
    if training_data.empty:
        raise ValueError("学習用データが空です。")

    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required_columns if column not in training_data.columns]
    if missing:
        raise ValueError(f"学習に必要な列がありません: {', '.join(missing)}")

    X = training_data.loc[:, FEATURE_COLUMNS].copy()
    raw_y = training_data.loc[:, TARGET_COLUMN].copy()

    # 数値文字列は許可するが、変換結果をXへ保持してLightGBMへobject型を渡さない。
    try:
        X = X.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("特徴量に数値へ変換できない値が含まれています。") from exc
    if X.isna().any().any():
        raise ValueError("特徴量に欠損値が含まれています。")
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("特徴量に正または負の無限値が含まれています。")

    # "0"・"1"は許可する。小数を整数へ切り捨てる前に元の意味を検証する。
    try:
        numeric_y = pd.to_numeric(raw_y, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Targetは数値の0または1で指定してください。") from exc
    if numeric_y.isna().any():
        raise ValueError("Targetに欠損値が含まれています。")
    if not np.isfinite(numeric_y.to_numpy(dtype=float)).all():
        raise ValueError("Targetに正または負の無限値が含まれています。")
    if not numeric_y.isin([0, 1]).all():
        raise ValueError("Targetは厳密に0または1で指定してください。")

    y = numeric_y.astype("int8")
    return X, y


def split_time_series(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
) -> DataSplit:
    """過去を学習、未来をテストとして時系列順に分割する。

    Args:
        X: 特徴量DataFrame。
        y: Xと同じインデックスを持つ目的変数。
        test_size: 末尾からテストに割り当てる割合。デフォルトは20%。

    Returns:
        X_train、X_test、y_train、y_testの順のタプル。

    Raises:
        ValueError: 比率、件数、インデックス、または期間の前後関係が不正な場合。
    """
    if not 0 < test_size < 1:
        raise ValueError("test_sizeは0より大きく1より小さい値を指定してください。")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("特徴量XとTarget yのインデックスを一致させてください。")
    if len(X) < MINIMUM_DATA_ROWS:
        raise ValueError(
            f"データが少なすぎます。最低{MINIMUM_DATA_ROWS}行必要です。"
        )

    # 同じ並べ替え順をXとyへ適用し、ランダムなシャッフルは行わない。
    order = X.index.argsort(kind="stable")
    sorted_X = X.iloc[order].copy()
    sorted_y = y.iloc[order].copy()

    test_rows = int(np.ceil(len(sorted_X) * test_size))
    split_position = len(sorted_X) - test_rows
    if split_position < 2 or test_rows < 1:
        raise ValueError("学習期間とテスト期間の両方に十分なデータがありません。")

    X_train = sorted_X.iloc[:split_position].copy()
    X_test = sorted_X.iloc[split_position:].copy()
    y_train = sorted_y.iloc[:split_position].copy()
    y_test = sorted_y.iloc[split_position:].copy()

    if X_train.index.max() >= X_test.index.min():
        raise ValueError("学習期間の最終日はテスト期間の開始日より前である必要があります。")
    return X_train, X_test, y_train, y_test


def train_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LGBMClassifier:
    """過去期間だけを使ってLightGBM分類モデルを学習する。

    Args:
        X_train: 学習期間の特徴量。
        y_train: 学習期間のTarget。
        random_state: 再現性を確保する乱数シード。

    Returns:
        学習済みLGBMClassifier。

    Raises:
        ValueError: 学習データが空、件数不一致、または片方のクラスしかない場合。
    """
    if X_train.empty or y_train.empty:
        raise ValueError("学習データが空です。")
    if len(X_train) != len(y_train):
        raise ValueError("X_trainとy_trainの件数が一致しません。")
    if y_train.nunique() < 2:
        raise ValueError("学習データにはTargetの0と1の両方が必要です。")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def predict_labels(model: LGBMClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """テストデータの上昇・非上昇ラベルを予測する。

    Args:
        model: 学習済みLightGBM分類モデル。
        X_test: テスト期間の特徴量。

    Returns:
        0または1の予測ラベル配列。

    Raises:
        ValueError: テストデータが空の場合。
    """
    if X_test.empty:
        raise ValueError("予測するテストデータが空です。")
    return np.asarray(model.predict(X_test), dtype="int8")


def predict_up_probabilities(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """テストデータについて上昇クラス1の確率を予測する。

    Args:
        model: 学習済みLightGBM分類モデル。
        X_test: テスト期間の特徴量。

    Returns:
        0以上1以下の上昇確率配列。

    Raises:
        ValueError: テストデータが空、またはモデルにクラス1がない場合。
    """
    if X_test.empty:
        raise ValueError("予測するテストデータが空です。")

    class_positions = np.flatnonzero(model.classes_ == 1)
    if len(class_positions) != 1:
        raise ValueError("学習済みモデルに上昇クラス1がありません。")
    probabilities = model.predict_proba(X_test)[:, class_positions[0]]
    return np.asarray(probabilities, dtype=float)


def calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> MetricValues:
    """分類予測の評価指標を計算する。

    Args:
        y_true: テスト期間の正解Target。
        predictions: 0または1の予測ラベル。
        probabilities: 上昇クラス1の予測確率。

    Returns:
        Accuracy、Precision、Recall、F1、ROC-AUCを持つ辞書。
        正解が片方のクラスだけの場合、ROC-AUCはNone。

    Raises:
        ValueError: 正解と予測の件数が一致しない場合。
    """
    if len(y_true) != len(predictions) or len(y_true) != len(probabilities):
        raise ValueError("正解ラベル、予測ラベル、予測確率の件数が一致しません。")

    roc_auc: Optional[float]
    if y_true.nunique() < 2:
        roc_auc = None
    else:
        roc_auc = float(roc_auc_score(y_true, probabilities))

    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "ROC-AUC": roc_auc,
    }


def get_feature_importance(model: LGBMClassifier) -> pd.DataFrame:
    """特徴量名と重要度を高い順に返す。

    Args:
        model: FEATURE_COLUMNSで学習したLightGBM分類モデル。

    Returns:
        Feature列とImportance列を持つDataFrame。

    Raises:
        ValueError: モデルの特徴量数がFEATURE_COLUMNSと一致しない場合。
    """
    importances = np.asarray(model.feature_importances_)
    if len(importances) != len(FEATURE_COLUMNS):
        raise ValueError("モデルの特徴量数がFEATURE_COLUMNSと一致しません。")

    importance = pd.DataFrame(
        {"Feature": list(FEATURE_COLUMNS), "Importance": importances.astype(int)}
    )
    return importance.sort_values(
        "Importance", ascending=False, kind="stable", ignore_index=True
    )


def run_model_pipeline(
    training_data: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ModelResult:
    """時系列分割からLightGBMの学習・評価までを一度に実行する。

    Args:
        training_data: 特徴量とTargetを含む学習用DataFrame。
        test_size: 未来側をテストに使う割合。
        random_state: LightGBMの乱数シード。

    Returns:
        学習済みモデル、分割データ、予測、評価結果をまとめたModelResult。

    Raises:
        ValueError: 入力、分割、または学習条件が不正な場合。
    """
    X, y = separate_features_target(training_data)
    X_train, X_test, y_train, y_test = split_time_series(X, y, test_size)
    model = train_classifier(X_train, y_train, random_state)
    predictions = predict_labels(model, X_test)
    probabilities = predict_up_probabilities(model, X_test)
    metrics = calculate_metrics(y_test, predictions, probabilities)
    importance = get_feature_importance(model)

    return ModelResult(
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        predictions=predictions,
        probabilities=probabilities,
        metrics=metrics,
        feature_importance=importance,
    )
