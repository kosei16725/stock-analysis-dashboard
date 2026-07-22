"""学習期間だけで行う特徴量セット比較実験の通信非依存テスト。"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import src.feature_selection as feature_selection
from src.constants import FEATURE_COLUMNS, TARGET_COLUMN
from src.feature_selection import (
    BASELINE_FEATURE_COLUMNS,
    RESULT_COLUMNS,
    bound_backtest_prices,
    build_feature_sets,
    evaluate_feature_set,
    purge_boundary_training_sample,
    run_feature_selection_experiment,
    select_top_features,
)
from src.model import separate_features_target, split_time_series


@pytest.fixture
def experiment_data() -> tuple[pd.DataFrame, pd.Series]:
    """100営業日の人工学習データと、翌営業日を含む終値を返す。"""
    rng = np.random.default_rng(20260723)
    index = pd.date_range("2025-01-01", periods=100, freq="B", name="Date")
    values = rng.normal(size=(len(index), len(FEATURE_COLUMNS)))
    training_data = pd.DataFrame(values, index=index, columns=FEATURE_COLUMNS)
    training_data[TARGET_COLUMN] = np.arange(len(index)) % 2

    close_index = pd.date_range("2025-01-01", periods=101, freq="B", name="Date")
    close_prices = pd.Series(
        100.0 + np.arange(len(close_index)) * 0.1 + np.sin(np.arange(len(close_index))),
        index=close_index,
        name="Close",
    )
    return training_data, close_prices


def test_feature_sets_have_expected_counts() -> None:
    """All、Top 15、Top 10、Baselineが指定された特徴量数になる。"""
    feature_sets = build_feature_sets(tuple(reversed(FEATURE_COLUMNS)))

    assert len(feature_sets["All 20 Features"]) == 20
    assert len(feature_sets["Top 15 by Gain Importance"]) == 15
    assert len(feature_sets["Top 10 by Gain Importance"]) == 10
    assert len(feature_sets["Baseline 9 Features"]) == 9
    assert feature_sets["Baseline 9 Features"] == BASELINE_FEATURE_COLUMNS


def test_selected_features_are_unique_and_do_not_mutate_constants() -> None:
    """各セットに重複がなく、FEATURE_COLUMNSの内容を変更しない。"""
    original_features = tuple(FEATURE_COLUMNS)
    feature_sets = build_feature_sets(FEATURE_COLUMNS)

    assert all(len(features) == len(set(features)) for features in feature_sets.values())
    assert tuple(FEATURE_COLUMNS) == original_features


def test_duplicate_gain_ranking_is_rejected() -> None:
    """重複したGain順位から上位特徴量を選択しない。"""
    ranking = (FEATURE_COLUMNS[0], FEATURE_COLUMNS[0], *FEATURE_COLUMNS[2:])

    with pytest.raises(ValueError, match="重複"):
        select_top_features(ranking, 10)


def test_experiment_result_has_required_columns_and_common_period(
    experiment_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """比較表が必要列を持ち、4セットの学習・テスト期間が一致する。"""
    training_data, close_prices = experiment_data

    result = run_feature_selection_experiment(training_data, close_prices)

    assert tuple(result.columns) == RESULT_COLUMNS
    assert result["Feature_Count"].tolist() == [20, 15, 10, 9]
    for column in ("Train_Start", "Train_End", "Test_Start", "Test_End"):
        assert result[column].nunique() == 1
    assert result["Train_End"].iloc[0] < result["Test_Start"].iloc[0]
    assert all(
        len(features) == count
        for features, count in zip(result["Selected_Features"], result["Feature_Count"])
    )


def test_test_data_changes_do_not_change_selected_features(
    experiment_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """未来側20%の値を変更しても学習期間由来のGain選択は変わらない。"""
    training_data, close_prices = experiment_data
    changed_test_data = training_data.copy()
    changed_test_data.loc[changed_test_data.index[-20:], FEATURE_COLUMNS] *= 1000.0

    original = run_feature_selection_experiment(training_data, close_prices)
    changed = run_feature_selection_experiment(changed_test_data, close_prices)

    assert original["Selected_Features"].tolist() == changed[
        "Selected_Features"
    ].tolist()


def test_gain_ranking_receives_training_period_only(
    experiment_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gain順位関数へ未来側テスト20%が渡されないことを確認する。"""
    training_data, close_prices = experiment_data
    captured: dict[str, object] = {}
    original_ranker = feature_selection.rank_features_by_training_gain

    def capture_training_period(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
    ) -> tuple[str, ...]:
        captured["rows"] = len(X_train)
        captured["start"] = X_train.index.min()
        captured["end"] = X_train.index.max()
        captured["target_end"] = y_train.index.max()
        return original_ranker(X_train, y_train, random_state)

    monkeypatch.setattr(
        feature_selection,
        "rank_features_by_training_gain",
        capture_training_period,
    )

    result = feature_selection.run_feature_selection_experiment(
        training_data,
        close_prices,
    )

    assert captured["rows"] == 79
    assert captured["start"] == training_data.index[0]
    assert captured["end"] == training_data.index[78]
    assert captured["target_end"] == training_data.index[78]
    assert captured["end"] < result["Test_Start"].iloc[0]


def test_boundary_target_row_is_purged_without_changing_test_period(
    experiment_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """テスト初日を翌日として持つ学習末尾1件だけを除外する。"""
    training_data, _ = experiment_data
    X, y = separate_features_target(training_data)
    X_train, X_test, y_train, y_test = split_time_series(X, y)
    boundary_index = X_train.index[-1]
    original_test_index = X_test.index.copy()

    purged_X, purged_y = purge_boundary_training_sample(X_train, y_train)

    assert len(purged_X) == len(X_train) - 1
    assert len(purged_y) == len(y_train) - 1
    assert boundary_index not in purged_X.index
    assert boundary_index not in purged_y.index
    assert purged_X.index[-1] < boundary_index < X_test.index[0]
    assert X_test.index.equals(original_test_index)
    assert len(y_test) == 20


def test_all_models_receive_same_purged_training_boundary(
    experiment_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gainランキングと4セットが同じパージ済み学習期間を使う。"""
    training_data, close_prices = experiment_data
    captured_periods: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    original_ranker = feature_selection.rank_features_by_training_gain
    original_evaluator = feature_selection.evaluate_feature_set

    def capture_ranker(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
    ) -> tuple[str, ...]:
        captured_periods.append(
            (X_train.index.min(), X_train.index.max(), len(X_train))
        )
        return original_ranker(X_train, y_train, random_state)

    def capture_evaluator(*args: object, **kwargs: object) -> dict[str, object]:
        X_train = kwargs["X_train"]
        assert isinstance(X_train, pd.DataFrame)
        captured_periods.append(
            (X_train.index.min(), X_train.index.max(), len(X_train))
        )
        return original_evaluator(*args, **kwargs)

    monkeypatch.setattr(
        feature_selection,
        "rank_features_by_training_gain",
        capture_ranker,
    )
    monkeypatch.setattr(feature_selection, "evaluate_feature_set", capture_evaluator)

    result = run_feature_selection_experiment(training_data, close_prices)

    assert len(captured_periods) == 5
    assert len(set(captured_periods)) == 1
    assert captured_periods[0] == (
        training_data.index[0],
        training_data.index[78],
        79,
    )
    assert result["Test_Start"].iloc[0] == training_data.index[80]
    assert result["Test_End"].iloc[0] == training_data.index[99]


def test_experiment_does_not_mutate_inputs_or_feature_columns(
    experiment_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """実験が入力DataFrame、終値Series、FEATURE_COLUMNSを変更しない。"""
    training_data, close_prices = experiment_data
    original_training = training_data.copy(deep=True)
    original_close = close_prices.copy(deep=True)
    original_features = tuple(FEATURE_COLUMNS)

    run_feature_selection_experiment(training_data, close_prices)

    pd.testing.assert_frame_equal(training_data, original_training)
    pd.testing.assert_series_equal(close_prices, original_close)
    assert tuple(FEATURE_COLUMNS) == original_features


@pytest.mark.parametrize("invalid_feature", [TARGET_COLUMN, "Next_Return", "Future_Feature"])
def test_undefined_feature_is_rejected_before_model_training(
    experiment_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
    invalid_feature: str,
) -> None:
    """Targetや未来由来の未定義列を、モデル学習前に具体名付きで拒否する。"""
    training_data, close_prices = experiment_data
    X, y = separate_features_target(training_data)
    X_train, X_test, y_train, y_test = split_time_series(X, y)
    X_train[invalid_feature] = 1.0
    X_test[invalid_feature] = 1.0
    train_mock = Mock(side_effect=AssertionError("モデル学習を呼んではいけません。"))
    monkeypatch.setattr(feature_selection, "train_classifier", train_mock)

    with pytest.raises(ValueError, match=invalid_feature):
        evaluate_feature_set(
            "Invalid",
            (FEATURE_COLUMNS[0], invalid_feature),
            X_train,
            X_test,
            y_train,
            y_test,
            close_prices,
        )

    train_mock.assert_not_called()


def test_defined_feature_must_exist_in_both_train_and_test(
    experiment_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """正規スキーマの列でもテストデータにない場合は学習前に拒否する。"""
    training_data, close_prices = experiment_data
    X, y = separate_features_target(training_data)
    X_train, X_test, y_train, y_test = split_time_series(X, y)
    missing_feature = FEATURE_COLUMNS[0]
    X_test = X_test.drop(columns=missing_feature)
    train_mock = Mock(side_effect=AssertionError("モデル学習を呼んではいけません。"))
    monkeypatch.setattr(feature_selection, "train_classifier", train_mock)

    with pytest.raises(ValueError, match=f"テストデータ.*{missing_feature}"):
        evaluate_feature_set(
            "Missing Test Feature",
            (missing_feature,),
            X_train,
            X_test,
            y_train,
            y_test,
            close_prices,
        )

    train_mock.assert_not_called()


def test_defined_feature_set_is_evaluated_normally(
    experiment_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """FEATURE_COLUMNS内の正常な特徴量セットは従来どおり評価できる。"""
    training_data, close_prices = experiment_data
    X, y = separate_features_target(training_data)
    X_train, X_test, y_train, y_test = split_time_series(X, y)

    result = evaluate_feature_set(
        "Baseline 9 Features",
        BASELINE_FEATURE_COLUMNS,
        X_train,
        X_test,
        y_train,
        y_test,
        close_prices,
    )

    assert result["Feature_Count"] == 9
    assert result["Selected_Features"] == BASELINE_FEATURE_COLUMNS
    assert result["Test_Start"] == X_test.index.min()
    assert result["Test_End"] == X_test.index.max()


def append_future_prices(close_prices: pd.Series, periods: int = 5) -> pd.Series:
    """人工終値の後ろへ、バックテスト対象外となる取引日を追加する。"""
    future_index = pd.date_range(
        close_prices.index[-1],
        periods=periods + 1,
        freq="B",
        name=close_prices.index.name,
    )[1:]
    future = pd.Series(
        np.linspace(200.0, 300.0, periods),
        index=future_index,
        name=close_prices.name,
    )
    return pd.concat([close_prices, future])


def test_backtest_prices_end_at_first_trading_day_after_last_prediction(
    experiment_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """余分な将来取引日を、末尾予測の翌取引日より後へ含めない。"""
    training_data, close_prices = experiment_data
    X, y = separate_features_target(training_data)
    _, X_test, _, _ = split_time_series(X, y)
    extended_close = append_future_prices(close_prices)

    bounded = bound_backtest_prices(extended_close, X_test.index)
    expected_next_date = extended_close.index[extended_close.index > X_test.index.max()][0]

    assert bounded.index[0] == X_test.index.min()
    assert bounded.index[-1] == expected_next_date
    assert (bounded.index > expected_next_date).sum() == 0


def test_extra_future_prices_do_not_change_experiment_metrics(
    experiment_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """評価範囲外の価格を大きく変更しても比較指標が変わらない。"""
    training_data, close_prices = experiment_data
    extended_close = append_future_prices(close_prices)
    changed_future = extended_close.copy()
    changed_future.loc[changed_future.index > close_prices.index[-1]] *= 1000.0

    original = run_feature_selection_experiment(training_data, extended_close)
    changed = run_feature_selection_experiment(training_data, changed_future)
    metric_columns = [
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
    ]

    pd.testing.assert_frame_equal(original[metric_columns], changed[metric_columns])


def test_missing_next_trading_day_raises_value_error(
    experiment_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """最後のテスト予測を執行する翌取引日がなければ明示的に拒否する。"""
    training_data, close_prices = experiment_data
    X, y = separate_features_target(training_data)
    X_train, X_test, y_train, y_test = split_time_series(X, y)
    insufficient_close = close_prices.loc[: X_test.index.max()].copy()

    with pytest.raises(ValueError, match="翌営業日の価格が不足"):
        evaluate_feature_set(
            "Baseline 9 Features",
            BASELINE_FEATURE_COLUMNS,
            X_train,
            X_test,
            y_train,
            y_test,
            insufficient_close,
        )


def test_all_feature_sets_use_same_bounded_backtest_period(
    experiment_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4セットすべてでTest_End直後の同じ取引日までを評価する。"""
    training_data, close_prices = experiment_data
    extended_close = append_future_prices(close_prices)
    captured_periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    original_backtest = feature_selection.run_backtest

    def capture_backtest_period(
        probabilities: pd.Series,
        bounded_close: pd.Series,
        threshold: float,
    ) -> object:
        captured_periods.append((bounded_close.index.min(), bounded_close.index.max()))
        return original_backtest(probabilities, bounded_close, threshold)

    monkeypatch.setattr(feature_selection, "run_backtest", capture_backtest_period)

    result = run_feature_selection_experiment(training_data, extended_close)
    expected_test_end = result["Test_End"].iloc[0]
    expected_next_date = extended_close.index[extended_close.index > expected_test_end][0]

    assert len(captured_periods) == 4
    assert len(set(captured_periods)) == 1
    assert captured_periods[0][0] == result["Test_Start"].iloc[0]
    assert captured_periods[0][1] == expected_next_date
    assert captured_periods[0][1] > expected_test_end
