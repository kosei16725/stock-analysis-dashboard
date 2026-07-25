"""expanding windowウォークフォワード検証の通信非依存テスト。"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import src.feature_selection as feature_selection
import src.walk_forward as walk_forward
from src.constants import FEATURE_COLUMNS, TARGET_COLUMN
from src.model import separate_features_target
from src.walk_forward import (
    AGGREGATE_RESULT_COLUMNS,
    FEATURE_SET_ORDER,
    FOLD_RESULT_COLUMNS,
    WalkForwardSplit,
    aggregate_walk_forward_results,
    create_walk_forward_splits,
    evaluate_walk_forward_fold,
    run_walk_forward_experiment,
)


@pytest.fixture(scope="module")
def walk_forward_data() -> tuple[pd.DataFrame, pd.Series]:
    """3つの完全Foldと翌取引日を作れる165営業日の人工データを返す。"""
    rng = np.random.default_rng(20260723)
    index = pd.date_range("2025-01-01", periods=165, freq="B", name="Date")
    values = rng.normal(size=(len(index), len(FEATURE_COLUMNS)))
    training_data = pd.DataFrame(values, index=index, columns=FEATURE_COLUMNS)
    training_data[TARGET_COLUMN] = np.arange(len(index)) % 2

    close_index = pd.date_range("2025-01-01", periods=166, freq="B", name="Date")
    close_prices = pd.Series(
        100.0
        + np.arange(len(close_index)) * 0.1
        + np.sin(np.arange(len(close_index))),
        index=close_index,
        name="Close",
    )
    return training_data, close_prices


@pytest.fixture(scope="module")
def completed_experiment(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> walk_forward.WalkForwardResult:
    """人工データで実行した標準条件のウォークフォワード結果を返す。"""
    training_data, close_prices = walk_forward_data
    return run_walk_forward_experiment(training_data, close_prices)


def make_splits(
    training_data: pd.DataFrame,
    initial_train_size: int = 100,
    test_size: int = 20,
    step_size: int = 20,
) -> list[WalkForwardSplit]:
    """人工学習データをXとyへ分離してFoldを作る。"""
    X, y = separate_features_target(training_data)
    return create_walk_forward_splits(
        X,
        y,
        initial_train_size=initial_train_size,
        test_size=test_size,
        step_size=step_size,
    )


def test_expanding_folds_are_chronological_and_disjoint(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """各Foldでtrainとtestが重複せず、trainがtestより過去になる。"""
    training_data, _ = walk_forward_data
    splits = make_splits(training_data)

    assert [split.fold for split in splits] == [1, 2, 3]
    for split in splits:
        assert split.X_train.index.is_monotonic_increasing
        assert split.X_test.index.is_monotonic_increasing
        assert split.X_train.index.intersection(split.X_test.index).empty
        assert split.X_train.index.max() < split.X_test.index.min()


def test_default_sizes_and_expanding_window_are_applied(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """initial/test/stepがFold件数とexpanding trainへ反映される。"""
    training_data, _ = walk_forward_data
    splits = make_splits(training_data)

    assert [split.train_size_before_purge for split in splits] == [100, 120, 140]
    assert [len(split.X_train) for split in splits] == [99, 119, 139]
    assert [len(split.X_test) for split in splits] == [20, 20, 20]
    assert splits[1].X_test.index.min() == training_data.index[120]
    assert splits[2].X_test.index.min() == training_data.index[140]


def test_custom_step_size_is_applied(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """step_sizeだけ学習末尾とテスト開始位置が進む。"""
    training_data, _ = walk_forward_data
    splits = make_splits(training_data, step_size=10)

    assert splits[0].train_size_before_purge == 100
    assert splits[1].train_size_before_purge == 110
    assert splits[1].X_test.index.min() == training_data.index[110]


def test_incomplete_final_fold_is_excluded(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """末尾5件だけの不完全なテストFoldを作成しない。"""
    training_data, _ = walk_forward_data
    splits = make_splits(training_data)

    assert len(splits) == 3
    assert splits[-1].X_test.index.max() == training_data.index[159]
    assert training_data.index[160] not in splits[-1].X_test.index


def test_no_complete_fold_raises_value_error(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """initial trainの後に完全なtestを確保できなければ拒否する。"""
    training_data, _ = walk_forward_data

    with pytest.raises(ValueError, match="Foldを1つも"):
        make_splits(training_data, initial_train_size=150, test_size=20)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("initial_train_size", 0),
        ("initial_train_size", 1),
        ("initial_train_size", -1),
        ("test_size", 0),
        ("test_size", -1),
        ("step_size", 0),
        ("step_size", -1),
    ],
)
def test_non_positive_window_arguments_are_rejected(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
    name: str,
    value: int,
) -> None:
    """0、負数、およびinitial trainの1を拒否する。"""
    training_data, _ = walk_forward_data
    kwargs = {"initial_train_size": 100, "test_size": 20, "step_size": 20}
    kwargs[name] = value

    with pytest.raises(ValueError):
        make_splits(training_data, **kwargs)


@pytest.mark.parametrize("invalid_value", [1.5, "20", None])
@pytest.mark.parametrize(
    "name",
    ["initial_train_size", "test_size", "step_size"],
)
def test_non_integer_window_arguments_are_rejected(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
    name: str,
    invalid_value: object,
) -> None:
    """件数引数の小数、文字列、Noneを拒否する。"""
    training_data, _ = walk_forward_data
    kwargs = {"initial_train_size": 100, "test_size": 20, "step_size": 20}
    kwargs[name] = invalid_value

    with pytest.raises(ValueError, match="整数"):
        make_splits(training_data, **kwargs)


@pytest.mark.parametrize(
    "name",
    ["initial_train_size", "test_size", "step_size"],
)
def test_boolean_window_arguments_are_rejected(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
    name: str,
) -> None:
    """boolをPython上の整数として受け入れない。"""
    training_data, _ = walk_forward_data
    kwargs = {"initial_train_size": 100, "test_size": 20, "step_size": 20}
    kwargs[name] = True

    with pytest.raises(ValueError, match="整数"):
        make_splits(training_data, **kwargs)


def test_each_fold_purges_only_boundary_label_and_keeps_test(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """各Foldで末尾Target1件だけを除外し、test件数を維持する。"""
    training_data, _ = walk_forward_data
    splits = make_splits(training_data)

    for split in splits:
        assert split.purged_sample_count == 1
        assert len(split.X_train) == split.train_size_before_purge - 1
        assert len(split.y_train) == split.train_size_before_purge - 1
        assert len(split.X_test) == 20
        assert len(split.y_test) == 20
        boundary_position = split.train_size_before_purge - 1
        assert training_data.index[boundary_position] not in split.X_train.index
        assert split.X_train.index.max() < training_data.index[boundary_position]


def test_ranking_and_four_models_share_same_purged_fold_train(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gainランキングと4モデルへ同一のパージ済みtrain/testを渡す。"""
    training_data, close_prices = walk_forward_data
    split = make_splits(training_data)[0]
    captured: list[tuple[pd.Index, pd.Index]] = []
    original_ranker = walk_forward.rank_features_by_training_gain
    original_evaluator = walk_forward.evaluate_feature_set

    def capture_ranker(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
    ) -> tuple[str, ...]:
        assert X_train.index.equals(y_train.index)
        captured.append((X_train.index.copy(), split.X_test.index.copy()))
        return original_ranker(X_train, y_train, random_state)

    def capture_evaluator(*args: object, **kwargs: object) -> dict[str, object]:
        X_train = kwargs["X_train"]
        X_test = kwargs["X_test"]
        y_train = kwargs["y_train"]
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert X_train.index.equals(y_train.index)
        captured.append((X_train.index.copy(), X_test.index.copy()))
        return original_evaluator(*args, **kwargs)

    monkeypatch.setattr(
        walk_forward,
        "rank_features_by_training_gain",
        capture_ranker,
    )
    monkeypatch.setattr(walk_forward, "evaluate_feature_set", capture_evaluator)

    evaluate_walk_forward_fold(split, close_prices)

    assert len(captured) == 5
    assert all(train.equals(split.X_train.index) for train, _ in captured)
    assert all(test.equals(split.X_test.index) for _, test in captured)


def test_test_changes_do_not_change_fold_selected_features(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """同じFoldのtest特徴量だけを変更してもGain選択が変わらない。"""
    training_data, close_prices = walk_forward_data
    split = make_splits(training_data)[0]
    changed_split = WalkForwardSplit(
        fold=split.fold,
        X_train=split.X_train.copy(),
        X_test=split.X_test * 1000.0,
        y_train=split.y_train.copy(),
        y_test=split.y_test.copy(),
        train_size_before_purge=split.train_size_before_purge,
    )

    original = evaluate_walk_forward_fold(split, close_prices)
    changed = evaluate_walk_forward_fold(changed_split, close_prices)

    assert original["Selected_Features"].tolist() == changed[
        "Selected_Features"
    ].tolist()


def test_feature_set_counts_and_unique_features_in_every_fold(
    completed_experiment: walk_forward.WalkForwardResult,
) -> None:
    """各FoldのAll/Top 15/Top 10/Baseline件数と重複なしを確認する。"""
    result = completed_experiment.fold_results
    expected_counts = {
        "All 20 Features": 20,
        "Top 15 by Gain Importance": 15,
        "Top 10 by Gain Importance": 10,
        "Baseline 9 Features": 9,
    }

    for row in result.itertuples():
        assert row.Feature_Count == expected_counts[row.Feature_Set]
        assert len(row.Selected_Features) == len(set(row.Selected_Features))


@pytest.mark.parametrize(
    "invalid_feature",
    [TARGET_COLUMN, "Next_Return", "Future_Feature"],
)
def test_undefined_features_are_rejected_before_training(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
    monkeypatch: pytest.MonkeyPatch,
    invalid_feature: str,
) -> None:
    """FEATURE_COLUMNS外の代表的な未来列をモデル学習前に拒否する。"""
    training_data, close_prices = walk_forward_data
    split = make_splits(training_data)[0]
    X_train = split.X_train.copy()
    X_test = split.X_test.copy()
    X_train[invalid_feature] = 1.0
    X_test[invalid_feature] = 1.0
    invalid_split = WalkForwardSplit(
        fold=split.fold,
        X_train=X_train,
        X_test=X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        train_size_before_purge=split.train_size_before_purge,
    )
    train_mock = Mock(side_effect=AssertionError("学習を呼んではいけません。"))
    monkeypatch.setattr(feature_selection, "train_classifier", train_mock)

    with pytest.raises(ValueError, match=invalid_feature):
        evaluate_walk_forward_fold(
            invalid_split,
            close_prices,
            feature_sets={"Invalid": (FEATURE_COLUMNS[0], invalid_feature)},
        )

    train_mock.assert_not_called()


def append_future_prices(close_prices: pd.Series, periods: int = 5) -> pd.Series:
    """終値末尾へ評価対象外の取引日を追加する。"""
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


def test_backtest_ends_at_fold_last_prediction_next_trading_day(
    completed_experiment: walk_forward.WalkForwardResult,
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """各FoldのBacktest_EndがTest_End直後の実取引日になる。"""
    _, close_prices = walk_forward_data
    results = completed_experiment.fold_results

    for row in results.itertuples():
        expected = close_prices.index[close_prices.index > row.Test_End][0]
        assert row.Backtest_End == expected
        assert row.Backtest_End > row.Test_End


def test_extra_future_prices_do_not_change_fold_metrics(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fold終了後の価格を大きく変更しても評価指標が変わらない。"""
    training_data, close_prices = walk_forward_data
    split = make_splits(training_data)[0]
    extended = append_future_prices(close_prices)
    changed = extended.copy()
    changed.loc[changed.index > close_prices.index[-1]] *= 1000.0

    original_result = evaluate_walk_forward_fold(split, extended)
    changed_result = evaluate_walk_forward_fold(split, changed)
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

    pd.testing.assert_frame_equal(
        original_result[metric_columns],
        changed_result[metric_columns],
    )


def test_missing_next_trading_day_raises_value_error(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Fold末尾予測の翌取引日がなければ明示的に拒否する。"""
    training_data, close_prices = walk_forward_data
    split = make_splits(training_data)[0]
    insufficient = close_prices.loc[: split.X_test.index.max()]

    with pytest.raises(ValueError, match="翌営業日の価格が不足"):
        evaluate_walk_forward_fold(split, insufficient)


def test_all_feature_sets_share_fold_evaluation_period(
    completed_experiment: walk_forward.WalkForwardResult,
) -> None:
    """同一Foldの4セットでtrain/test/backtest期間が一致する。"""
    results = completed_experiment.fold_results
    period_columns = [
        "Train_Start",
        "Train_End",
        "Test_Start",
        "Test_End",
        "Backtest_End",
        "Train_Size_Before_Purge",
        "Train_Size_After_Purge",
        "Purged_Sample_Count",
        "Test_Size",
    ]

    for _, group in results.groupby("Fold"):
        assert len(group) == 4
        assert all(group[column].nunique() == 1 for column in period_columns)


def test_fold_and_aggregate_result_columns_are_complete(
    completed_experiment: walk_forward.WalkForwardResult,
) -> None:
    """Fold別・集約結果が規定列を規定順序で持つ。"""
    assert tuple(completed_experiment.fold_results.columns) == FOLD_RESULT_COLUMNS
    assert (
        tuple(completed_experiment.aggregate_results.columns)
        == AGGREGATE_RESULT_COLUMNS
    )
    assert completed_experiment.aggregate_results["Feature_Set"].tolist() == list(
        FEATURE_SET_ORDER
    )


def make_single_fold_aggregate_input(total_return: float) -> pd.DataFrame:
    """集約関数の1 Fold境界条件を確認する最小DataFrameを作る。"""
    row: dict[str, object] = {
        "Feature_Set": "All 20 Features",
        "Fold": 1,
        "Feature_Count": 20,
    }
    for metric in walk_forward.AGGREGATE_METRICS:
        row[metric] = total_return if metric == "Total_Return" else 0.5
    return pd.DataFrame([row])


def test_single_fold_standard_deviations_are_zero() -> None:
    """Foldが1件でもすべての標準偏差を0.0として返す。"""
    aggregate = aggregate_walk_forward_results(
        make_single_fold_aggregate_input(total_return=0.1)
    )
    std_columns = [column for column in aggregate if column.endswith("_Std")]

    assert all(aggregate[column].iloc[0] == 0.0 for column in std_columns)


@pytest.mark.parametrize(
    ("returns", "expected_count", "expected_rate"),
    [
        ([0.1, -0.1, 0.2], 2, 2 / 3),
        ([0.0, -0.1], 0, 0.0),
    ],
)
def test_positive_return_fold_count_and_rate(
    returns: list[float],
    expected_count: int,
    expected_rate: float,
) -> None:
    """Total_Returnが正のFold数と割合を正しく集約する。"""
    frames = []
    for fold, total_return in enumerate(returns, start=1):
        frame = make_single_fold_aggregate_input(total_return)
        frame["Fold"] = fold
        frames.append(frame)

    aggregate = aggregate_walk_forward_results(pd.concat(frames, ignore_index=True))

    assert aggregate["Positive_Return_Folds"].iloc[0] == expected_count
    assert aggregate["Positive_Return_Rate"].iloc[0] == pytest.approx(expected_rate)


def test_inputs_and_feature_columns_are_not_mutated(
    walk_forward_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """実験が入力DataFrame、終値Series、FEATURE_COLUMNSを変更しない。"""
    training_data, close_prices = walk_forward_data
    training_copy = training_data.copy(deep=True)
    close_copy = close_prices.copy(deep=True)
    feature_columns_copy = tuple(FEATURE_COLUMNS)

    run_walk_forward_experiment(training_data, close_prices)

    pd.testing.assert_frame_equal(training_data, training_copy)
    pd.testing.assert_series_equal(close_prices, close_copy)
    assert tuple(FEATURE_COLUMNS) == feature_columns_copy
