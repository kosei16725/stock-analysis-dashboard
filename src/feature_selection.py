"""学習期間のGain重要度を使った特徴量セット比較実験を提供する。"""

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from src.backtest import DEFAULT_BUY_THRESHOLD, run_backtest
from src.constants import (
    DAILY_RETURN_COLUMN,
    FEATURE_COLUMNS,
    MA_DEVIATION_COLUMN,
    RETURN_5_COLUMN,
    RETURN_20_COLUMN,
    VOLATILITY_20_COLUMN,
    VOLUME_CHANGE_COLUMN,
    moving_average_column,
)
from src.model import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    calculate_metrics,
    predict_labels,
    predict_up_probabilities,
    separate_features_target,
    split_time_series,
    train_classifier,
)

# Phase 8でテクニカル指標を追加する前に使用していた9特徴量を明示する。
BASELINE_FEATURE_COLUMNS: Tuple[str, ...] = (
    DAILY_RETURN_COLUMN,
    RETURN_5_COLUMN,
    RETURN_20_COLUMN,
    moving_average_column(5),
    moving_average_column(20),
    moving_average_column(50),
    MA_DEVIATION_COLUMN,
    VOLATILITY_20_COLUMN,
    VOLUME_CHANGE_COLUMN,
)

RESULT_COLUMNS = (
    "Feature_Set",
    "Feature_Count",
    "Selected_Features",
    "Train_Start",
    "Train_End",
    "Test_Start",
    "Test_End",
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


def rank_features_by_training_gain(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[str, ...]:
    """学習データだけでLightGBMを学習し、Gain重要度順を返す。

    Args:
        X_train: 過去側の学習特徴量。評価対象のテスト行を含めない。
        y_train: X_trainに対応する学習Target。
        random_state: LightGBMの乱数シード。

    Returns:
        Gain降順、同値時は特徴量名昇順の特徴量名タプル。

    Raises:
        ValueError: 学習列がFEATURE_COLUMNSと一致しない場合、または学習不能な場合。
    """
    if tuple(X_train.columns) != FEATURE_COLUMNS:
        raise ValueError("Gain順位の学習列をFEATURE_COLUMNSと一致させてください。")

    ranking_model = train_classifier(X_train, y_train, random_state)
    feature_names = ranking_model.booster_.feature_name()
    gains = np.asarray(
        ranking_model.booster_.feature_importance(importance_type="gain"),
        dtype=float,
    )
    if feature_names != list(FEATURE_COLUMNS) or len(gains) != len(FEATURE_COLUMNS):
        raise ValueError("Gain重要度とFEATURE_COLUMNSの対応が一致しません。")

    ranking = pd.DataFrame({"Feature": feature_names, "Gain": gains})
    ranking = ranking.sort_values(
        ["Gain", "Feature"],
        ascending=[False, True],
        kind="stable",
        ignore_index=True,
    )
    return tuple(ranking["Feature"])


def select_top_features(
    gain_ranking: Sequence[str],
    count: int,
) -> Tuple[str, ...]:
    """Gain順位の先頭から重複のない指定数の特徴量を選ぶ。

    Args:
        gain_ranking: 学習期間だけから求めた特徴量順位。
        count: 選択する特徴量数。

    Returns:
        順位を維持した特徴量名タプル。

    Raises:
        ValueError: 件数が不正、順位に重複・未知の特徴量がある場合。
    """
    ranking = tuple(gain_ranking)
    if count < 1 or count > len(ranking):
        raise ValueError("選択数はGain順位の特徴量数以内で指定してください。")
    if len(set(ranking)) != len(ranking):
        raise ValueError("Gain順位に重複した特徴量があります。")
    unknown = [feature for feature in ranking if feature not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"Gain順位に未定義の特徴量があります: {', '.join(unknown)}")
    return ranking[:count]


def build_feature_sets(gain_ranking: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """比較対象となる4種類の特徴量セットを作る。

    Args:
        gain_ranking: 学習期間だけから求めた全20特徴量のGain順位。

    Returns:
        セット名をキー、特徴量名タプルを値とする辞書。

    Raises:
        ValueError: 順位が全FEATURE_COLUMNSを一度ずつ含まない場合。
    """
    ranking = tuple(gain_ranking)
    if len(ranking) != len(FEATURE_COLUMNS) or set(ranking) != set(FEATURE_COLUMNS):
        raise ValueError("Gain順位は全FEATURE_COLUMNSを一度ずつ含めてください。")
    return {
        "All 20 Features": tuple(FEATURE_COLUMNS),
        "Top 15 by Gain Importance": select_top_features(ranking, 15),
        "Top 10 by Gain Importance": select_top_features(ranking, 10),
        "Baseline 9 Features": BASELINE_FEATURE_COLUMNS,
    }


def bound_backtest_prices(
    close_prices: pd.Series,
    prediction_index: pd.Index,
) -> pd.Series:
    """予測期間と末尾予測の翌取引日だけに終値を制限する。

    Args:
        close_prices: 実際の取引日をインデックスに持つ終値Series。
        prediction_index: テスト予測日を持つインデックス。

    Returns:
        最初の予測日から、最後の予測日より後にある最初の取引日までの
        終値コピー。

    Raises:
        ValueError: 入力が空、日付が重複、または末尾予測の翌取引日がない場合。
    """
    if close_prices.empty:
        raise ValueError("バックテストに使用する終値データが空です。")
    if prediction_index.empty:
        raise ValueError("バックテストに使用するテスト予測日がありません。")
    if close_prices.index.has_duplicates:
        raise ValueError("終値の日付インデックスに重複があります。")

    sorted_close = close_prices.sort_index().copy()
    first_prediction_date = prediction_index.min()
    last_prediction_date = prediction_index.max()
    following_dates = sorted_close.index[sorted_close.index > last_prediction_date]
    if following_dates.empty:
        raise ValueError(
            "最後のテスト予測を執行する翌営業日の価格が不足しています。"
        )
    next_trading_date = following_dates[0]
    bounded = sorted_close.loc[
        (sorted_close.index >= first_prediction_date)
        & (sorted_close.index <= next_trading_date)
    ].copy()
    return bounded


def purge_boundary_training_sample(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """翌営業日Targetがテスト期間を参照する学習末尾1件を除外する。

    Args:
        X_train: 時系列分割直後の学習特徴量。
        y_train: X_trainと同じインデックスを持つ翌営業日Target。

    Returns:
        末尾1件を除外した学習特徴量とTargetのコピー。

    Raises:
        ValueError: インデックスが不一致、または除外後に学習行が残らない場合。
    """
    if not X_train.index.equals(y_train.index):
        raise ValueError("境界除外前の特徴量とTargetのインデックスが一致しません。")
    if len(X_train) < 2:
        raise ValueError("境界サンプル除外後の学習データが不足します。")
    return X_train.iloc[:-1].copy(), y_train.iloc[:-1].copy()


def evaluate_feature_set(
    feature_set_name: str,
    selected_features: Sequence[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    close_prices: pd.Series,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, object]:
    """指定特徴量だけで再学習し、分類・バックテスト指標を返す。

    Args:
        feature_set_name: 比較表へ表示する特徴量セット名。
        selected_features: 再学習へ使用する重複のない特徴量名。
        X_train: 共通の学習期間特徴量。
        X_test: 共通のテスト期間特徴量。
        y_train: 共通の学習Target。
        y_test: 共通のテストTarget。
        close_prices: バックテストに使う終値。実際の取引日から、最初の
            テスト予測日〜末尾予測の翌取引日へ制限して使用する。
        buy_threshold: Buy判定の予測確率閾値。
        random_state: LightGBMの乱数シード。

    Returns:
        特徴量セット、期間、分類指標、バックテスト指標を持つ辞書。

    Raises:
        ValueError: 特徴量セットが空、重複、未定義、または評価不能な場合。
    """
    features = tuple(selected_features)
    if not features:
        raise ValueError("評価する特徴量セットが空です。")
    if len(set(features)) != len(features):
        raise ValueError("選択特徴量に重複があります。")
    undefined = [feature for feature in features if feature not in FEATURE_COLUMNS]
    if undefined:
        raise ValueError(
            f"FEATURE_COLUMNSに未定義の特徴量があります: {', '.join(undefined)}"
        )
    missing_train = [feature for feature in features if feature not in X_train.columns]
    if missing_train:
        raise ValueError(
            f"学習データに選択特徴量がありません: {', '.join(missing_train)}"
        )
    missing_test = [feature for feature in features if feature not in X_test.columns]
    if missing_test:
        raise ValueError(
            f"テストデータに選択特徴量がありません: {', '.join(missing_test)}"
        )
    backtest_prices = bound_backtest_prices(close_prices, X_test.index)

    model = train_classifier(X_train.loc[:, features], y_train, random_state)
    X_test_selected = X_test.loc[:, features]
    predictions = predict_labels(model, X_test_selected)
    probabilities = predict_up_probabilities(model, X_test_selected)
    classification = calculate_metrics(y_test, predictions, probabilities)
    probability_series = pd.Series(
        probabilities,
        index=X_test.index,
        name="Probability",
    )
    backtest = run_backtest(probability_series, backtest_prices, buy_threshold)

    return {
        "Feature_Set": feature_set_name,
        "Feature_Count": len(features),
        "Selected_Features": features,
        "Train_Start": X_train.index.min(),
        "Train_End": X_train.index.max(),
        "Test_Start": X_test.index.min(),
        "Test_End": X_test.index.max(),
        "Accuracy": classification["Accuracy"],
        "Precision": classification["Precision"],
        "Recall": classification["Recall"],
        "F1": classification["F1"],
        "ROC_AUC": classification["ROC-AUC"],
        "Total_Return": backtest.metrics["Total Return"],
        "Annual_Return": backtest.metrics["Annual Return"],
        "Sharpe_Ratio": backtest.metrics["Sharpe Ratio"],
        "Max_Drawdown": backtest.metrics["Max Drawdown"],
        "Win_Rate": backtest.metrics["Win Rate"],
        "Total_Trades": backtest.metrics["Total Trades"],
    }


def run_feature_selection_experiment(
    training_data: pd.DataFrame,
    close_prices: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """同一の時系列分割で4種類の特徴量セットを比較する。

    特徴量順位は分割後のX_trainだけで決定し、X_testは順位決定後に各
    特徴量セットへ適用する。ランキング用モデルの予測は評価に使用しない。

    Args:
        training_data: 全FEATURE_COLUMNSとTargetを含む学習用DataFrame。
        close_prices: バックテスト期間を含む日足終値。
        test_size: 未来側をテストに使う割合。
        buy_threshold: Buy判定の予測確率閾値。
        random_state: すべてのLightGBMへ共通で渡す乱数シード。

    Returns:
        4特徴量セットの分類・バックテスト比較DataFrame。

    Raises:
        ValueError: 入力、分割、特徴量選択、学習、または評価に失敗した場合。
    """
    X, y = separate_features_target(training_data)
    X_train, X_test, y_train, y_test = split_time_series(X, y, test_size)
    # Targetは翌営業日の終値から作るため、分割直後の学習末尾ラベルは
    # テスト初日の価格を参照する。この1件を全ランキング・評価から除外する。
    X_train, y_train = purge_boundary_training_sample(X_train, y_train)
    gain_ranking = rank_features_by_training_gain(X_train, y_train, random_state)
    feature_sets = build_feature_sets(gain_ranking)

    rows = [
        evaluate_feature_set(
            feature_set_name=name,
            selected_features=features,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            close_prices=close_prices,
            buy_threshold=buy_threshold,
            random_state=random_state,
        )
        for name, features in feature_sets.items()
    ]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)
