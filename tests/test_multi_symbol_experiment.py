"""複数銘柄ウォークフォワード実験の通信非依存テスト。"""

from pathlib import Path

import pandas as pd
import pytest

import src.multi_symbol_experiment as multi_symbol
from src.walk_forward import (
    FEATURE_SET_ORDER,
    FOLD_RESULT_COLUMNS,
    WalkForwardResult,
    aggregate_walk_forward_results,
)


def make_walk_forward_result(metric_base: float = 0.1) -> WalkForwardResult:
    """2 Fold・4特徴量セットの固定ウォークフォワード結果を作る。"""
    rows = []
    for fold in (1, 2):
        for count, feature_set in zip((20, 15, 10, 9), FEATURE_SET_ORDER):
            metric_value = metric_base + fold * 0.01
            rows.append(
                {
                    "Fold": fold,
                    "Feature_Set": feature_set,
                    "Feature_Count": count,
                    "Selected_Features": tuple(f"Feature_{i}" for i in range(count)),
                    "Train_Start": pd.Timestamp("2025-01-01"),
                    "Train_End": pd.Timestamp("2025-04-30"),
                    "Test_Start": pd.Timestamp("2025-05-02"),
                    "Test_End": pd.Timestamp("2025-05-29"),
                    "Backtest_End": pd.Timestamp("2025-05-30"),
                    "Train_Size_Before_Purge": 100,
                    "Train_Size_After_Purge": 99,
                    "Purged_Sample_Count": 1,
                    "Test_Size": 20,
                    "Accuracy": metric_value,
                    "Precision": metric_value,
                    "Recall": metric_value,
                    "F1": metric_value,
                    "ROC_AUC": metric_value,
                    "Total_Return": metric_value,
                    "Annual_Return": metric_value,
                    "Sharpe_Ratio": metric_value,
                    "Max_Drawdown": -metric_value,
                    "Win_Rate": metric_value,
                    "Total_Trades": fold,
                }
            )
    fold_results = pd.DataFrame(rows, columns=FOLD_RESULT_COLUMNS)
    return WalkForwardResult(
        fold_results=fold_results,
        aggregate_results=aggregate_walk_forward_results(fold_results),
    )


def test_one_symbol_failure_does_not_stop_remaining_symbols(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """1銘柄が失敗しても後続銘柄を評価し、理由を記録する。"""
    calls = []

    def fake_single_symbol(symbol: str, **_: object) -> WalkForwardResult:
        calls.append(symbol)
        if symbol == "9432.T":
            raise RuntimeError("mock download failure")
        return make_walk_forward_result()

    monkeypatch.setattr(
        multi_symbol,
        "run_single_symbol_experiment",
        fake_single_symbol,
    )

    result = multi_symbol.run_multi_symbol_experiment(
        ["7203.T", "9432.T", "8306.T"]
    )

    assert calls == ["7203.T", "9432.T", "8306.T"]
    assert result.success_count == 2
    assert result.failure_count == 1
    assert tuple(result.symbol_results) == ("7203.T", "8306.T")
    assert result.failures.loc[0, "Ticker"] == "9432.T"
    assert "mock download failure" in result.failures.loc[0, "Error_Message"]


def test_all_symbol_failures_raise_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """全銘柄が失敗した場合だけ、全理由を含むValueErrorにする。"""

    def always_fail(symbol: str, **_: object) -> WalkForwardResult:
        raise RuntimeError(f"{symbol} unavailable")

    monkeypatch.setattr(
        multi_symbol,
        "run_single_symbol_experiment",
        always_fail,
    )

    with pytest.raises(ValueError, match="全銘柄.*7203.T.*6758.T"):
        multi_symbol.run_multi_symbol_experiment(["7203.T", "6758.T"])


def test_symbol_and_overall_aggregation_are_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """銘柄別行数と全銘柄Fold平均を正しく集約する。"""
    results = {
        "AAA": make_walk_forward_result(metric_base=0.1),
        "BBB": make_walk_forward_result(metric_base=0.3),
    }
    monkeypatch.setattr(
        multi_symbol,
        "run_single_symbol_experiment",
        lambda symbol, **_: results[symbol],
    )

    result = multi_symbol.run_multi_symbol_experiment(["AAA", "BBB"])
    all_features = result.overall_summary.loc[
        result.overall_summary["Feature_Set"] == "All 20 Features"
    ].iloc[0]

    assert len(result.fold_results) == 16
    assert len(result.symbol_summary) == 8
    assert all_features["Successful_Symbol_Count"] == 2
    assert all_features["Total_Fold_Count"] == 4
    assert all_features["Accuracy_Mean"] == pytest.approx(0.215)
    assert all_features["Total_Return_Mean"] == pytest.approx(0.215)


def test_csv_files_are_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fold結果とSummaryを指定ディレクトリへCSV保存する。"""
    monkeypatch.setattr(
        multi_symbol,
        "run_single_symbol_experiment",
        lambda symbol, **_: make_walk_forward_result(),
    )
    result = multi_symbol.run_multi_symbol_experiment(["7203.T"])

    fold_path, summary_path = multi_symbol.save_multi_symbol_results(
        result,
        tmp_path / "results",
    )

    assert fold_path.name == "multi_symbol_fold_results.csv"
    assert summary_path.name == "multi_symbol_summary.csv"
    assert fold_path.exists()
    assert summary_path.exists()
    saved_fold = pd.read_csv(fold_path)
    saved_summary = pd.read_csv(summary_path)
    assert "Ticker" in saved_fold.columns
    assert set(saved_summary["Scope"]) >= {"Symbol", "Overall"}


def test_input_symbols_are_not_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """入力リストの順序・内容を変更せず、正規化コピーだけを使用する。"""
    symbols = ["7203.t", " 6758.T "]
    original = symbols.copy()
    monkeypatch.setattr(
        multi_symbol,
        "run_single_symbol_experiment",
        lambda symbol, **_: make_walk_forward_result(),
    )

    result = multi_symbol.run_multi_symbol_experiment(symbols)

    assert symbols == original
    assert tuple(result.symbol_results) == ("7203.T", "6758.T")


def test_single_symbol_runner_reuses_phase_11(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取得・特徴量作成後に指定条件のPhase 11関数を呼び出す。"""
    prices = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="B"),
    )
    training_data = pd.DataFrame({"mock": [1.0]})
    expected = make_walk_forward_result()
    captured = {}

    def fake_fetch(**kwargs: object) -> pd.DataFrame:
        captured["fetch"] = kwargs
        return prices

    monkeypatch.setattr(multi_symbol, "fetch_stock_data", fake_fetch)
    monkeypatch.setattr(
        multi_symbol,
        "create_training_data",
        lambda received: training_data if received is prices else None,
    )

    def fake_walk_forward(**kwargs: object) -> WalkForwardResult:
        captured["walk_forward"] = kwargs
        return expected

    monkeypatch.setattr(
        multi_symbol,
        "run_walk_forward_experiment",
        fake_walk_forward,
    )

    result = multi_symbol.run_single_symbol_experiment("7203.T")

    assert result is expected
    assert captured["fetch"] == {
        "ticker": "7203.T",
        "period": "2y",
        "moving_average_windows": (20, 50),
    }
    walk_forward_args = captured["walk_forward"]
    assert walk_forward_args["training_data"] is training_data
    assert walk_forward_args["initial_train_size"] == 100
    assert walk_forward_args["test_size"] == 20
    assert walk_forward_args["step_size"] == 20
    assert walk_forward_args["buy_threshold"] == 0.55


def test_failures_are_included_in_summary_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """一部失敗の理由をSummary CSVへ保存する。"""

    def fake_single_symbol(symbol: str, **_: object) -> WalkForwardResult:
        if symbol == "FAIL":
            raise ValueError("insufficient rows")
        return make_walk_forward_result()

    monkeypatch.setattr(
        multi_symbol,
        "run_single_symbol_experiment",
        fake_single_symbol,
    )
    result = multi_symbol.run_multi_symbol_experiment(["OK", "FAIL"])

    _, summary_path = multi_symbol.save_multi_symbol_results(result, tmp_path)
    summary = pd.read_csv(summary_path)
    failure = summary.loc[summary["Scope"] == "Failure"].iloc[0]

    assert failure["Ticker"] == "FAIL"
    assert failure["Error_Message"] == "insufficient rows"


def test_stability_summary_and_report_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """安定性指標の最小回数とレポート用件数を返す。"""
    monkeypatch.setattr(
        multi_symbol,
        "run_single_symbol_experiment",
        lambda symbol, **_: make_walk_forward_result(),
    )

    result = multi_symbol.run_multi_symbol_experiment(["7203.T"])
    report = multi_symbol.build_report_summary(result)

    assert len(result.stability_summary) == 4
    assert result.stability_summary["Most_Stable_By_Selected_Std"].all()
    assert report["Success_Count"] == 1
    assert report["Failure_Count"] == 0
    assert report["Successful_Symbols"] == ("7203.T",)
