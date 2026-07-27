"""複数銘柄へPhase 11ウォークフォワード検証を適用する。"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import pandas as pd

from src.backtest import DEFAULT_BUY_THRESHOLD
from src.constants import CLOSE_COLUMN
from src.data_loader import fetch_stock_data
from src.features import create_training_data
from src.model import DEFAULT_RANDOM_STATE
from src.walk_forward import (
    AGGREGATE_METRICS,
    FEATURE_SET_ORDER,
    WalkForwardResult,
    aggregate_walk_forward_results,
    run_walk_forward_experiment,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_SYMBOLS: Tuple[str, ...] = (
    "7203.T",
    "6758.T",
    "9432.T",
    "8306.T",
    "8035.T",
)
DEFAULT_PERIOD = "2y"
DEFAULT_INITIAL_TRAIN_SIZE = 100
DEFAULT_WALK_FORWARD_TEST_SIZE = 20
DEFAULT_STEP_SIZE = 20
DEFAULT_RESULTS_DIRECTORY = Path("data/results")

STABILITY_METRICS = (
    "Accuracy_Std",
    "F1_Std",
    "ROC_AUC_Std",
    "Total_Return_Std",
    "Sharpe_Ratio_Std",
)


@dataclass(frozen=True)
class MultiSymbolExperimentResult:
    """複数銘柄の成功結果、集約、失敗情報を保持する。"""

    symbol_results: Mapping[str, WalkForwardResult]
    fold_results: pd.DataFrame
    symbol_summary: pd.DataFrame
    overall_summary: pd.DataFrame
    failures: pd.DataFrame
    stability_summary: pd.DataFrame

    @property
    def success_count(self) -> int:
        """成功銘柄数を返す。"""
        return len(self.symbol_results)

    @property
    def failure_count(self) -> int:
        """失敗銘柄数を返す。"""
        return len(self.failures)


def _normalize_symbols(symbols: Sequence[str]) -> Tuple[str, ...]:
    """入力順を維持して銘柄コードを正規化し、重複を拒否する。"""
    normalized = tuple(
        symbol.strip().upper() if isinstance(symbol, str) else str(symbol)
        for symbol in symbols
    )
    if not normalized:
        raise ValueError("実験対象の銘柄コードを1つ以上指定してください。")
    duplicates = sorted(
        {symbol for symbol in normalized if normalized.count(symbol) > 1}
    )
    if duplicates:
        raise ValueError(f"重複した銘柄コードがあります: {', '.join(duplicates)}")
    return normalized


def run_single_symbol_experiment(
    symbol: str,
    period: str = DEFAULT_PERIOD,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    test_size: int = DEFAULT_WALK_FORWARD_TEST_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> WalkForwardResult:
    """1銘柄の取得・特徴量作成・Phase 11評価を実行する。

    Args:
        symbol: Yahoo Finance形式の銘柄コード。
        period: 株価取得期間。
        initial_train_size: Fold 1の分割直前学習件数。
        test_size: 各Foldのテスト件数。
        step_size: Foldごとに学習窓を拡張する件数。
        buy_threshold: Buy判定の予測確率閾値。
        random_state: LightGBMの乱数シード。

    Returns:
        Phase 11のFold別・集約結果。

    Raises:
        ValueError: 入力、特徴量、Fold、学習、評価が不正な場合。
        RuntimeError: 株価取得に失敗した場合。
    """
    prices = fetch_stock_data(
        ticker=symbol,
        period=period,
        moving_average_windows=(20, 50),
    )
    training_data = create_training_data(prices)
    close_prices = prices[CLOSE_COLUMN].dropna().copy()
    return run_walk_forward_experiment(
        training_data=training_data,
        close_prices=close_prices,
        initial_train_size=initial_train_size,
        test_size=test_size,
        step_size=step_size,
        buy_threshold=buy_threshold,
        random_state=random_state,
    )


def _combine_symbol_results(
    symbol_results: Mapping[str, WalkForwardResult],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """成功銘柄のFold別結果と銘柄別集約を連結する。"""
    fold_frames = []
    summary_frames = []
    for symbol, result in symbol_results.items():
        fold_frame = result.fold_results.copy()
        fold_frame.insert(0, "Ticker", symbol)
        fold_frames.append(fold_frame)

        summary_frame = result.aggregate_results.copy()
        summary_frame.insert(0, "Ticker", symbol)
        summary_frames.append(summary_frame)
    return (
        pd.concat(fold_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )


def aggregate_across_symbols(fold_results: pd.DataFrame) -> pd.DataFrame:
    """全成功銘柄の全Foldを特徴量セットごとに総合集約する。

    Args:
        fold_results: Ticker列を付けた全成功銘柄のFold別結果。

    Returns:
        成功銘柄数と全Fold数を含む特徴量セット別の総合集約。

    Raises:
        ValueError: 入力が空、またはTicker列がない場合。
    """
    if fold_results.empty:
        raise ValueError("総合集約するFold結果が空です。")
    if "Ticker" not in fold_results.columns:
        raise ValueError("総合集約にはTicker列が必要です。")

    aggregate_input = fold_results.copy()
    aggregate_input["Fold"] = (
        aggregate_input["Ticker"].astype(str)
        + "-"
        + aggregate_input["Fold"].astype(str)
    )
    overall = aggregate_walk_forward_results(aggregate_input)
    symbol_counts = (
        fold_results.groupby("Feature_Set", sort=False)["Ticker"].nunique()
    )
    overall.insert(
        1,
        "Successful_Symbol_Count",
        overall["Feature_Set"].map(symbol_counts).astype(int),
    )
    overall = overall.rename(columns={"Fold_Count": "Total_Fold_Count"})
    return overall


def calculate_stability_summary(symbol_summary: pd.DataFrame) -> pd.DataFrame:
    """銘柄ごとに標準偏差が最小となった指標数を集計する。"""
    required = {"Ticker", "Feature_Set", *STABILITY_METRICS}
    missing = sorted(required.difference(symbol_summary.columns))
    if missing:
        raise ValueError(f"安定性集計に必要な列がありません: {', '.join(missing)}")

    rows = []
    for symbol, group in symbol_summary.groupby("Ticker", sort=False):
        scores: Dict[str, int] = {
            feature_set: 0 for feature_set in FEATURE_SET_ORDER
        }
        for metric in STABILITY_METRICS:
            minimum = group[metric].min()
            winners = group.loc[group[metric] == minimum, "Feature_Set"]
            for feature_set in winners:
                scores[feature_set] += 1
        maximum_score = max(scores.values())
        for feature_set in FEATURE_SET_ORDER:
            rows.append(
                {
                    "Ticker": symbol,
                    "Feature_Set": feature_set,
                    "Stability_Win_Count": scores[feature_set],
                    "Most_Stable_By_Selected_Std": (
                        scores[feature_set] == maximum_score
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_multi_symbol_experiment(
    symbols: Sequence[str] = DEFAULT_SYMBOLS,
    period: str = DEFAULT_PERIOD,
    initial_train_size: int = DEFAULT_INITIAL_TRAIN_SIZE,
    test_size: int = DEFAULT_WALK_FORWARD_TEST_SIZE,
    step_size: int = DEFAULT_STEP_SIZE,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> MultiSymbolExperimentResult:
    """複数銘柄を個別評価し、失敗銘柄を記録して残りを継続する。

    Args:
        symbols: 評価対象銘柄。入力順を維持し、変更しない。
        period: 全銘柄共通の株価取得期間。
        initial_train_size: 全銘柄共通の初期学習件数。
        test_size: 全銘柄共通のテスト件数。
        step_size: 全銘柄共通のstep件数。
        buy_threshold: 全銘柄共通のBuy閾値。
        random_state: 全銘柄共通の乱数シード。

    Returns:
        銘柄別結果、総合集約、失敗理由を持つMultiSymbolExperimentResult。

    Raises:
        ValueError: 対象が空・重複、または全銘柄の評価に失敗した場合。
    """
    normalized_symbols = _normalize_symbols(symbols)
    successful: Dict[str, WalkForwardResult] = {}
    failure_rows = []
    for symbol in normalized_symbols:
        try:
            successful[symbol] = run_single_symbol_experiment(
                symbol=symbol,
                period=period,
                initial_train_size=initial_train_size,
                test_size=test_size,
                step_size=step_size,
                buy_threshold=buy_threshold,
                random_state=random_state,
            )
        except Exception as exc:
            LOGGER.warning("%sの複数銘柄実験に失敗しました: %s", symbol, exc)
            failure_rows.append(
                {
                    "Ticker": symbol,
                    "Error_Type": type(exc).__name__,
                    "Error_Message": str(exc),
                }
            )

    failures = pd.DataFrame(
        failure_rows,
        columns=["Ticker", "Error_Type", "Error_Message"],
    )
    if not successful:
        reasons = "; ".join(
            f"{row['Ticker']}: {row['Error_Message']}" for row in failure_rows
        )
        raise ValueError(f"全銘柄の実験に失敗しました。{reasons}")

    fold_results, symbol_summary = _combine_symbol_results(successful)
    overall_summary = aggregate_across_symbols(fold_results)
    stability_summary = calculate_stability_summary(symbol_summary)
    return MultiSymbolExperimentResult(
        symbol_results=dict(successful),
        fold_results=fold_results,
        symbol_summary=symbol_summary,
        overall_summary=overall_summary,
        failures=failures,
        stability_summary=stability_summary,
    )


def save_multi_symbol_results(
    result: MultiSymbolExperimentResult,
    output_directory: Path = DEFAULT_RESULTS_DIRECTORY,
) -> Tuple[Path, Path]:
    """Fold別結果と集約・失敗情報をCSVへ保存する。

    Args:
        result: 保存対象の複数銘柄実験結果。
        output_directory: CSV保存先。存在しない場合は作成する。

    Returns:
        Fold結果CSV、Summary CSVの順のパスタプル。

    Raises:
        OSError: ディレクトリ作成またはCSV書き込みに失敗した場合。
    """
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    fold_path = directory / "multi_symbol_fold_results.csv"
    summary_path = directory / "multi_symbol_summary.csv"

    result.fold_results.to_csv(fold_path, index=False)
    symbol_summary = result.symbol_summary.copy()
    symbol_summary.insert(0, "Scope", "Symbol")
    overall_summary = result.overall_summary.copy()
    overall_summary.insert(0, "Ticker", "ALL")
    overall_summary.insert(0, "Scope", "Overall")
    failure_summary = result.failures.copy()
    failure_summary.insert(0, "Scope", "Failure")
    summary = pd.concat(
        [symbol_summary, overall_summary, failure_summary],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(summary_path, index=False)
    return fold_path, summary_path


def build_report_summary(result: MultiSymbolExperimentResult) -> Dict[str, object]:
    """Markdown等の実験レポート作成に使う主要情報を返す。"""
    most_stable = result.stability_summary.loc[
        result.stability_summary["Most_Stable_By_Selected_Std"],
        ["Ticker", "Feature_Set", "Stability_Win_Count"],
    ].copy()
    return {
        "Success_Count": result.success_count,
        "Failure_Count": result.failure_count,
        "Successful_Symbols": tuple(result.symbol_results),
        "Failed_Symbols": tuple(result.failures.get("Ticker", pd.Series(dtype=str))),
        "Most_Stable_Feature_Sets": most_stable,
    }
