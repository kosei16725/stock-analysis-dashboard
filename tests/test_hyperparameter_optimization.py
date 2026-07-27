"""Optuna Nested Walk Forward最適化の通信非依存テスト。"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import optuna
import pandas as pd
import pytest

from src.constants import FEATURE_COLUMNS, TARGET_COLUMN
from src.hyperparameter_optimization import (
    FOLD_FAILURE_PENALTY,
    LOW_TRADE_PENALTY,
    OptimizationResult,
    calculate_penalized_sharpe,
    create_optuna_objective,
    create_study,
    run_nested_walk_forward_optimization,
    save_optimization_results,
    suggest_lightgbm_parameters,
)
from src.walk_forward import WalkForwardSplit


def _parameters() -> dict:
    """探索空間内の固定パラメータを返す。"""
    return {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
    }


def _training_data(rows: int = 140) -> pd.DataFrame:
    """全特徴量と二値Targetを持つ人工時系列を作る。"""
    index = pd.bdate_range("2024-01-01", periods=rows)
    data = {
        feature: np.linspace(number, number + 1, rows)
        for number, feature in enumerate(FEATURE_COLUMNS, start=1)
    }
    data[TARGET_COLUMN] = np.arange(rows) % 2
    return pd.DataFrame(data, index=index)


def _close_prices(rows: int = 145) -> pd.Series:
    """評価末尾の翌取引日を含む人工終値を作る。"""
    index = pd.bdate_range("2024-01-01", periods=rows)
    return pd.Series(np.linspace(100.0, 130.0, rows), index=index, name="Close")


def _split() -> WalkForwardSplit:
    """目的関数テスト用のパージ済みFoldを作る。"""
    frame = _training_data(90)
    X = frame.loc[:, FEATURE_COLUMNS]
    y = frame[TARGET_COLUMN]
    return WalkForwardSplit(
        fold=1,
        X_train=X.iloc[:69].copy(),
        X_test=X.iloc[70:90].copy(),
        y_train=y.iloc[:69].copy(),
        y_test=y.iloc[70:90].copy(),
        train_size_before_purge=70,
    )


def _completed_study() -> optuna.Study:
    """全探索パラメータを持つ完了済み1 Trial Studyを作る。"""
    study = create_study()
    study.enqueue_trial(_parameters())

    def objective(trial: optuna.Trial) -> float:
        suggest_lightgbm_parameters(trial)
        trial.set_user_attr("Inner_Fold_Count", 1)
        trial.set_user_attr("Failed_Fold_Count", 0)
        trial.set_user_attr("Total_Penalty", 0.0)
        return 1.25

    study.optimize(objective, n_trials=1)
    return study


def _optimization_result() -> OptimizationResult:
    """保存処理テスト用の最小結果を作る。"""
    trials = pd.DataFrame(
        [
            {
                "Outer_Fold": 1,
                "Trial": 0,
                "Value": 0.5,
                "State": "COMPLETE",
                **_parameters(),
            },
            {
                "Outer_Fold": 1,
                "Trial": 1,
                "Value": 1.25,
                "State": "COMPLETE",
                **_parameters(),
            },
        ]
    )
    return OptimizationResult(
        trials=trials,
        outer_fold_results=pd.DataFrame(),
        comparison_summary=pd.DataFrame(),
        best_parameters_by_fold={1: _parameters()},
        best_scores_by_fold={1: 1.25},
        recommended_parameters=_parameters(),
        recommended_score=1.25,
        parameter_importance={"learning_rate": 0.7, "num_leaves": 0.3},
    )


def test_trial_generates_all_parameters_within_search_space() -> None:
    """Trialが定義済み7パラメータを探索範囲内で生成する。"""
    study = create_study()

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_lightgbm_parameters(trial)
        assert set(parameters) == set(_parameters())
        assert 0.005 <= parameters["learning_rate"] <= 0.2
        assert 15 <= parameters["num_leaves"] <= 127
        assert 3 <= parameters["max_depth"] <= 12
        assert 5 <= parameters["min_child_samples"] <= 100
        assert 0.5 <= parameters["feature_fraction"] <= 1.0
        assert 0.5 <= parameters["bagging_fraction"] <= 1.0
        assert 1 <= parameters["bagging_freq"] <= 10
        return 0.0

    study.optimize(objective, n_trials=1)
    assert study.best_trial.number == 0


def test_best_parameters_are_selected_from_highest_objective() -> None:
    """最大化Studyが最も高い目的関数値のパラメータを選ぶ。"""
    study = create_study()
    study.enqueue_trial({**_parameters(), "num_leaves": 20})
    study.enqueue_trial({**_parameters(), "num_leaves": 80})

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_lightgbm_parameters(trial)
        return float(parameters["num_leaves"])

    study.optimize(objective, n_trials=2)
    assert study.best_params["num_leaves"] == 80
    assert study.best_value == 80.0


@pytest.mark.parametrize(
    ("sharpe", "trades", "expected_score", "expected_penalty"),
    [
        (2.0, 4, 2.0 - LOW_TRADE_PENALTY, LOW_TRADE_PENALTY),
        (2.0, 5, 2.0, 0.0),
        (float("nan"), 10, -10.0, 10.0),
    ],
)
def test_penalty_rules(
    sharpe: float,
    trades: int,
    expected_score: float,
    expected_penalty: float,
) -> None:
    """低取引数と無効Sharpeへ定義済みPenaltyを適用する。"""
    score, penalty = calculate_penalized_sharpe(sharpe, trades)
    assert score == expected_score
    assert penalty == expected_penalty


def test_fold_failure_is_penalized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inner Fold評価失敗をTrial停止ではなくPenaltyとして扱う。"""
    monkeypatch.setattr(
        "src.hyperparameter_optimization.evaluate_parameters_on_fold",
        Mock(side_effect=ValueError("artificial failure")),
    )
    study = create_study()
    study.optimize(
        create_optuna_objective([_split()], _close_prices()),
        n_trials=1,
    )
    assert study.best_value == -FOLD_FAILURE_PENALTY
    assert study.best_trial.user_attrs["Failed_Fold_Count"] == 1


def test_nested_result_keeps_input_unchanged_and_uses_best_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested実行が入力を破壊せずBest ParameterをOuter評価へ渡す。"""
    training_data = _training_data()
    close_prices = _close_prices()
    original_training = training_data.copy(deep=True)
    original_close = close_prices.copy(deep=True)
    study = _completed_study()
    optimize_mock = Mock(return_value=study)
    evaluate_mock = Mock(
        return_value={
            "Accuracy": 0.5,
            "Precision": 0.5,
            "Recall": 0.5,
            "F1": 0.5,
            "ROC_AUC": 0.5,
            "Total_Return": 0.01,
            "Annual_Return": 0.1,
            "Sharpe_Ratio": 1.0,
            "Max_Drawdown": -0.02,
            "Win_Rate": 0.5,
            "Total_Trades": 6,
            "Backtest_End": close_prices.index[-1],
        }
    )
    monkeypatch.setattr(
        "src.hyperparameter_optimization.optimize_inner_folds",
        optimize_mock,
    )
    monkeypatch.setattr(
        "src.hyperparameter_optimization.evaluate_parameters_on_fold",
        evaluate_mock,
    )

    result = run_nested_walk_forward_optimization(
        training_data,
        close_prices,
        n_trials=1,
        outer_initial_train_size=100,
        outer_test_size=20,
        outer_step_size=20,
        inner_initial_train_size=60,
        inner_test_size=20,
        inner_step_size=20,
    )

    pd.testing.assert_frame_equal(training_data, original_training)
    pd.testing.assert_series_equal(close_prices, original_close)
    assert result.recommended_parameters == _parameters()
    assert set(result.outer_fold_results["Model"]) == {"Default", "Tuned"}
    assert any(call.args[2] == study.best_params for call in evaluate_mock.call_args_list)


def test_save_results_creates_csv_json_and_png(tmp_path: Path) -> None:
    """Trial CSV、Best Parameter JSON、3つのPNGを生成する。"""
    paths = save_optimization_results(_optimization_result(), tmp_path)

    assert set(paths) == {
        "trials_csv",
        "best_parameters_json",
        "trial_values_png",
        "objective_history_png",
        "parameter_importance_png",
    }
    assert all(path.exists() for path in paths.values())
    loaded = pd.read_csv(paths["trials_csv"])
    assert len(loaded) == 2
    json_text = paths["best_parameters_json"].read_text(encoding="utf-8")
    assert '"recommended_score": 1.25' in json_text
    for key in (
        "trial_values_png",
        "objective_history_png",
        "parameter_importance_png",
    ):
        assert paths[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("n_trials", [0, -1, 1.5, True])
def test_invalid_trial_count_is_rejected(n_trials: object) -> None:
    """0、負数、非整数、boolのTrial数を拒否する。"""
    with pytest.raises(ValueError, match="n_trials"):
        run_nested_walk_forward_optimization(
            _training_data(),
            _close_prices(),
            n_trials=n_trials,  # type: ignore[arg-type]
        )
