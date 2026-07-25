"""expanding windowによる特徴量セットのウォークフォワード検証を提供する。"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtest import DEFAULT_BUY_THRESHOLD
from src.constants import FEATURE_COLUMNS
from src.feature_selection import (
    build_feature_sets,
    bound_backtest_prices,
    evaluate_feature_set,
    purge_boundary_training_sample,
    rank_features_by_training_gain,
)
from src.model import DEFAULT_RANDOM_STATE, separate_features_target

FEATURE_SET_ORDER = (
    "All 20 Features",
    "Top 15 by Gain Importance",
    "Top 10 by Gain Importance",
    "Baseline 9 Features",
)

FOLD_RESULT_COLUMNS = (
    "Fold",
    "Feature_Set",
    "Feature_Count",
    "Selected_Features",
    "Train_Start",
    "Train_End",
    "Test_Start",
    "Test_End",
    "Backtest_End",
    "Train_Size_Before_Purge",
    "Train_Size_After_Purge",
    "Purged_Sample_Count",
    "Test_Size",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "Total_Return",
    "Annual_Return",
    "Sharpe_Ratio",
    "Max_Drawdown",
    "Win_Rate",
    "Total_Trades",
)

AGGREGATE_RESULT_COLUMNS = (
    "Feature_Set",
    "Fold_Count",
    "Average_Feature_Count",
    "Accuracy_Mean",
    "Accuracy_Std",
    "Precision_Mean",
    "Precision_Std",
    "Recall_Mean",
    "Recall_Std",
    "F1_Mean",
    "F1_Std",
    "ROC_AUC_Mean",
    "ROC_AUC_Std",
    "Total_Return_Mean",
    "Total_Return_Std",
    "Annual_Return_Mean",
    "Annual_Return_Std",
    "Sharpe_Ratio_Mean",
    "Sharpe_Ratio_Std",
    "Max_Drawdown_Mean",
    "Max_Drawdown_Std",
    "Win_Rate_Mean",
    "Win_Rate_Std",
    "Total_Trades_Mean",
    "Total_Trades_Std",
    "Positive_Return_Folds",
    "Positive_Return_Rate",
)

AGGREGATE_METRICS = (
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "Total_Return",
    "Annual_Return",
    "Sharpe_Ratio",
    "Max_Drawdown",
    "Win_Rate",
    "Total_Trades",
)


@dataclass(frozen=True)
class WalkForwardSplit:
    """1つのFoldにおけるパージ済み学習データとテストデータ。"""

    fold: int
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    train_size_before_purge: int
    purged_sample_count: int = 1


@dataclass(frozen=True)
class WalkForwardResult:
    """Fold別結果と特徴量セット別集約結果。"""

    fold_results: pd.DataFrame
    aggregate_results: pd.DataFrame


def _validate_window_size(name: str, value: object, minimum: int) -> int:
    """ウォークフォワードの件数引数を厳密な整数として検証する。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name}は整数で指定してください。")
    integer_value = int(value)
    if integer_value < minimum:
        raise ValueError(f"{name}は{minimum}以上で指定してください。")
    return integer_value


def create_walk_forward_splits(
    X: pd.DataFrame,
    y: pd.Series,
    initial_train_size: int = 100,
    test_size: int = 20,
    step_size: int = 20,
) -> List[WalkForwardSplit]:
    """時系列順を保ったexpanding window Foldを作成する。

    Args:
        X: FEATURE_COLUMNSを持つ特徴量DataFrame。
        y: Xと同じインデックスを持つTarget。
        initial_train_size: Fold 1の分割直前学習件数。2以上。
        test_size: 各Foldの固定テスト件数。1以上。
        step_size: Foldごとに学習末尾を進める件数。1以上。

    Returns:
        学習末尾1件をパージ済みのWalkForwardSplit一覧。
        test_size未満の末尾データはFoldに含めない。

    Raises:
        ValueError: 引数、インデックス、列、件数、または時系列順が不正な場合。
    """
    initial = _validate_window_size("initial_train_size", initial_train_size, 2)
    test_rows = _validate_window_size("test_size", test_size, 1)
    step = _validate_window_size("step_size", step_size, 1)
    if X.empty or y.empty:
        raise ValueError("ウォークフォワード検証データが空です。")
    if len(X) != len(y) or not X.index.equals(y.index):
        raise ValueError("特徴量XとTarget yのインデックスを一致させてください。")
    if X.index.has_duplicates:
        raise ValueError("ウォークフォワード検証の日付に重複があります。")
    if tuple(X.columns) != FEATURE_COLUMNS:
        raise ValueError("ウォークフォワードの特徴量列をFEATURE_COLUMNSと一致させてください。")

    order = X.index.argsort(kind="stable")
    sorted_X = X.iloc[order].copy()
    sorted_y = y.iloc[order].copy()
    splits: List[WalkForwardSplit] = []
    train_end = initial
    fold_number = 1
    while train_end + test_rows <= len(sorted_X):
        X_train_before = sorted_X.iloc[:train_end].copy()
        y_train_before = sorted_y.iloc[:train_end].copy()
        X_test = sorted_X.iloc[train_end: train_end + test_rows].copy()
        y_test = sorted_y.iloc[train_end: train_end + test_rows].copy()
        X_train, y_train = purge_boundary_training_sample(
            X_train_before,
            y_train_before,
        )
        if X_train.index.max() >= X_test.index.min():
            raise ValueError("学習期間の全日付をテスト開始日より前にしてください。")
        splits.append(
            WalkForwardSplit(
                fold=fold_number,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                train_size_before_purge=train_end,
            )
        )
        fold_number += 1
        train_end += step

    if not splits:
        raise ValueError("指定条件では完全なテストFoldを1つも作成できません。")
    return splits


def evaluate_walk_forward_fold(
    split: WalkForwardSplit,
    close_prices: pd.Series,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
    feature_sets: Optional[Mapping[str, Sequence[str]]] = None,
) -> pd.DataFrame:
    """1つのFoldで特徴量順位を決め、各特徴量セットを再学習・評価する。

    Args:
        split: パージ済み学習期間と固定テスト期間を持つFold。
        close_prices: Foldのテスト期間と末尾翌取引日を含む終値。
        buy_threshold: Buy判定の予測確率閾値。
        random_state: 全LightGBMに共通の乱数シード。
        feature_sets: テスト等で明示する特徴量セット。省略時はFoldの学習期間
            だけでGain順位を計算し、標準4セットを作成する。

    Returns:
        4特徴量セットを基本とするFold別評価DataFrame。

    Raises:
        ValueError: 特徴量、期間、翌取引日、学習、または評価が不正な場合。
    """
    if feature_sets is None:
        gain_ranking = rank_features_by_training_gain(
            split.X_train,
            split.y_train,
            random_state,
        )
        selected_sets: Mapping[str, Sequence[str]] = build_feature_sets(gain_ranking)
    else:
        selected_sets = feature_sets
    if not selected_sets:
        raise ValueError("評価する特徴量セットがありません。")

    bounded_prices = bound_backtest_prices(close_prices, split.X_test.index)
    backtest_end = bounded_prices.index.max()
    rows: List[Dict[str, object]] = []
    for feature_set_name, selected_features in selected_sets.items():
        evaluation = evaluate_feature_set(
            feature_set_name=feature_set_name,
            selected_features=selected_features,
            X_train=split.X_train,
            X_test=split.X_test,
            y_train=split.y_train,
            y_test=split.y_test,
            close_prices=bounded_prices,
            buy_threshold=buy_threshold,
            random_state=random_state,
        )
        row = {
            "Fold": split.fold,
            **evaluation,
            "Backtest_End": backtest_end,
            "Train_Size_Before_Purge": split.train_size_before_purge,
            "Train_Size_After_Purge": len(split.X_train),
            "Purged_Sample_Count": split.purged_sample_count,
            "Test_Size": len(split.X_test),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=FOLD_RESULT_COLUMNS)


def _safe_sample_std(series: pd.Series) -> float:
    """有効値が1件以下なら0、それ以外は標本標準偏差を返す。"""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) <= 1:
        return 0.0
    return float(numeric.std(ddof=1))


def aggregate_walk_forward_results(fold_results: pd.DataFrame) -> pd.DataFrame:
    """Fold別結果を特徴量セットごとの平均・標準偏差へ集約する。

    Args:
        fold_results: FOLD_RESULT_COLUMNSを含むFold別DataFrame。

    Returns:
        標準4セット順の集約DataFrame。Foldが1つの場合の標準偏差は0。

    Raises:
        ValueError: 入力が空、必要列不足、または未知の特徴量セットを含む場合。
    """
    if fold_results.empty:
        raise ValueError("集約するFold別結果が空です。")
    required = {
        "Feature_Set",
        "Fold",
        "Feature_Count",
        *AGGREGATE_METRICS,
    }
    missing = sorted(required.difference(fold_results.columns))
    if missing:
        raise ValueError(f"Fold別結果に必要な列がありません: {', '.join(missing)}")
    unknown_sets = sorted(set(fold_results["Feature_Set"]) - set(FEATURE_SET_ORDER))
    if unknown_sets:
        raise ValueError(f"未定義の特徴量セットがあります: {', '.join(unknown_sets)}")

    rows: List[Dict[str, object]] = []
    for feature_set_name in FEATURE_SET_ORDER:
        group = fold_results.loc[fold_results["Feature_Set"] == feature_set_name]
        if group.empty:
            continue
        fold_count = int(group["Fold"].nunique())
        row: Dict[str, object] = {
            "Feature_Set": feature_set_name,
            "Fold_Count": fold_count,
            "Average_Feature_Count": float(group["Feature_Count"].mean()),
        }
        for metric in AGGREGATE_METRICS:
            numeric = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_Mean"] = float(numeric.mean())
            row[f"{metric}_Std"] = _safe_sample_std(numeric)
        positive_folds = int((group["Total_Return"] > 0).sum())
        row["Positive_Return_Folds"] = positive_folds
        row["Positive_Return_Rate"] = positive_folds / fold_count
        rows.append(row)
    return pd.DataFrame(rows, columns=AGGREGATE_RESULT_COLUMNS)


def run_walk_forward_experiment(
    training_data: pd.DataFrame,
    close_prices: pd.Series,
    initial_train_size: int = 100,
    test_size: int = 20,
    step_size: int = 20,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> WalkForwardResult:
    """全Foldの特徴量セット比較と集約を実行する。

    Args:
        training_data: 全FEATURE_COLUMNSとTargetを含む学習用DataFrame。
        close_prices: 全Foldのテスト日と各末尾翌取引日を含む終値。
        initial_train_size: Fold 1の分割直前学習件数。
        test_size: 各Foldの固定テスト件数。
        step_size: Foldごとに学習末尾を進める件数。
        buy_threshold: Buy判定の予測確率閾値。
        random_state: 全LightGBMに共通の乱数シード。

    Returns:
        ``fold_results``と``aggregate_results``を持つWalkForwardResult。

    Raises:
        ValueError: 入力、Fold作成、特徴量選択、学習、または評価に失敗した場合。
    """
    X, y = separate_features_target(training_data)
    splits = create_walk_forward_splits(
        X,
        y,
        initial_train_size=initial_train_size,
        test_size=test_size,
        step_size=step_size,
    )
    fold_frames = [
        evaluate_walk_forward_fold(
            split,
            close_prices,
            buy_threshold=buy_threshold,
            random_state=random_state,
        )
        for split in splits
    ]
    fold_results = pd.concat(fold_frames, ignore_index=True)
    aggregate_results = aggregate_walk_forward_results(fold_results)
    return WalkForwardResult(
        fold_results=fold_results,
        aggregate_results=aggregate_results,
    )
