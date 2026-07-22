"""予測確率を翌営業日に執行するロング・キャッシュ戦略を評価する。"""

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
import pandas as pd

DEFAULT_BUY_THRESHOLD = 0.55
TRADING_DAYS_PER_YEAR = 252

MetricValue = Union[float, int]


@dataclass(frozen=True)
class BacktestResult:
    """バックテストの時系列結果と評価指標。"""

    signals: pd.Series
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    cumulative_strategy: pd.Series
    cumulative_benchmark: pd.Series
    metrics: Dict[str, MetricValue]


def generate_signals(
    probabilities: pd.Series,
    threshold: float = DEFAULT_BUY_THRESHOLD,
) -> pd.Series:
    """上昇確率から予測日時点のBuy・Cash判断を作成する。

    Args:
        probabilities: 日付をインデックスとする上昇クラス1の予測確率。
        threshold: Buyと判断する確率の下限。デフォルトは0.55。

    Returns:
        Buyを1、Cashを0とする整数Series。まだ執行日はずらさない。

    Raises:
        ValueError: 入力が空、比率が不正、日付重複、欠損、または無限値の場合。
    """
    if probabilities.empty:
        raise ValueError("予測確率が空です。")
    if not 0 <= threshold <= 1:
        raise ValueError("thresholdは0以上1以下で指定してください。")
    if probabilities.index.has_duplicates:
        raise ValueError("予測確率の日付インデックスに重複があります。")

    try:
        numeric_probabilities = pd.to_numeric(
            probabilities.sort_index().copy(), errors="raise"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("予測確率に数値へ変換できない値があります。") from exc
    if numeric_probabilities.isna().any():
        raise ValueError("予測確率に欠損値があります。")
    if not np.isfinite(numeric_probabilities.to_numpy(dtype=float)).all():
        raise ValueError("予測確率に無限値があります。")
    if not numeric_probabilities.between(0, 1).all():
        raise ValueError("予測確率は0以上1以下で指定してください。")

    signals = (numeric_probabilities >= threshold).astype("int8")
    signals.name = "Decision_Signal"
    return signals


def execute_signals_next_day(
    decision_signals: pd.Series,
    trading_index: pd.Index,
) -> pd.Series:
    """予測日の判断を、価格データ上の翌営業日へ割り当てる。

    Args:
        decision_signals: 予測日をインデックスとするBuy・Cash判断。
        trading_index: 日付昇順の取引日インデックス。

    Returns:
        最初の予測日から終値データ最終日までの完全な取引日Series。
        判断は翌営業日から有効となり、次の判断までは直前のポジションを維持する。

    Raises:
        ValueError: 取引日が重複・未整列、予測日が存在しない、または執行不能の場合。
    """
    if trading_index.has_duplicates:
        raise ValueError("価格データの日付インデックスに重複があります。")
    if not trading_index.is_monotonic_increasing:
        raise ValueError("価格データの日付インデックスを昇順にしてください。")

    positions = trading_index.get_indexer(decision_signals.index)
    if (positions < 0).any():
        raise ValueError("予測日に対応する価格データがありません。")
    executable = positions + 1 < len(trading_index)
    if not executable.any():
        raise ValueError("予測シグナルを執行できる翌営業日がありません。")

    # 最初の予測日から価格データ末尾まで、途中を省かない評価カレンダーを作る。
    evaluation_index = trading_index[positions.min():]
    scheduled = pd.Series(np.nan, index=evaluation_index, dtype=float)
    execution_dates = trading_index[positions[executable] + 1]
    scheduled.loc[execution_dates] = decision_signals.to_numpy(dtype="int8")[
        executable
    ]

    # bfillは使わない。最初の執行前はCash、執行後は次の判断まで維持する。
    executed = scheduled.ffill().fillna(0).astype("int8")
    executed.name = "Signal"
    return executed


def calculate_daily_returns(close_prices: pd.Series) -> pd.Series:
    """終値から日次リターンを計算する。

    Args:
        close_prices: 日付をインデックスとする終値。先頭から昇順に整列する。

    Returns:
        前営業日終値からの日次リターン。先頭行は0。

    Raises:
        ValueError: 終値が空、重複、欠損、無限値、非正数の場合。
    """
    if close_prices.empty:
        raise ValueError("終値データが空です。")
    if close_prices.index.has_duplicates:
        raise ValueError("終値の日付インデックスに重複があります。")
    try:
        numeric_close = pd.to_numeric(close_prices.sort_index().copy(), errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("終値に数値へ変換できない値があります。") from exc
    if numeric_close.isna().any():
        raise ValueError("終値に欠損値があります。")
    if not np.isfinite(numeric_close.to_numpy(dtype=float)).all():
        raise ValueError("終値に無限値があります。")
    if (numeric_close <= 0).any():
        raise ValueError("終値は0より大きい値で指定してください。")

    returns = numeric_close.pct_change(fill_method=None).fillna(0.0)
    returns.name = "Daily_Return"
    return returns.astype(float)


def calculate_strategy_returns(
    daily_returns: pd.Series,
    executed_signals: pd.Series,
) -> pd.Series:
    """翌営業日の執行シグナルを同日のリターンへ適用する。

    Args:
        daily_returns: 全取引日の日次リターン。
        executed_signals: 翌営業日へ割り当て済みのBuy・Cashシグナル。

    Returns:
        Buy日は市場リターン、Cash日は0となる戦略リターン。

    Raises:
        ValueError: シグナル日の日次リターンがない、またはシグナルが0・1でない場合。
    """
    if not executed_signals.isin([0, 1]).all():
        raise ValueError("執行シグナルは0または1で指定してください。")
    aligned_returns = daily_returns.reindex(executed_signals.index)
    if aligned_returns.isna().any():
        raise ValueError("執行日に対応する日次リターンがありません。")

    strategy_returns = aligned_returns * executed_signals.astype(float)
    strategy_returns.name = "Strategy_Return"
    return strategy_returns


def calculate_benchmark_returns(
    daily_returns: pd.Series,
    evaluation_index: pd.Index,
) -> pd.Series:
    """戦略と同じ評価期間のBuy & Holdリターンを取り出す。

    Args:
        daily_returns: 全取引日の日次リターン。
        evaluation_index: 戦略を評価する執行日インデックス。

    Returns:
        評価期間中、常に保有した場合の日次リターン。

    Raises:
        ValueError: 評価日に対応する日次リターンがない場合。
    """
    benchmark = daily_returns.reindex(evaluation_index)
    if benchmark.isna().any():
        raise ValueError("評価日に対応するベンチマークリターンがありません。")
    benchmark = benchmark.astype(float)
    # 評価開始日の終値から保有開始とし、それ以前からのリターンは含めない。
    benchmark.iloc[0] = 0.0
    benchmark.name = "Benchmark_Return"
    return benchmark


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """日次リターンを複利で累積する。

    Args:
        returns: 日次リターン。

    Returns:
        初期資産を1とした累積収益率。

    Raises:
        ValueError: リターンが空、欠損、無限値、または-100%未満の場合。
    """
    if returns.empty:
        raise ValueError("累積計算するリターンが空です。")
    if returns.isna().any() or not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("リターンに欠損値または無限値があります。")
    if (returns < -1).any():
        raise ValueError("日次リターンは-100%以上で指定してください。")

    cumulative = (1.0 + returns.astype(float)).cumprod() - 1.0
    cumulative.name = "Cumulative_Return"
    return cumulative


def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    """累積収益率から最大ドローダウンを計算する。

    Args:
        cumulative_returns: 初期資産1を基準とする累積収益率。

    Returns:
        過去の最高資産からの最大下落率。0以下の値。

    Raises:
        ValueError: 累積収益率が空の場合。
    """
    if cumulative_returns.empty:
        raise ValueError("最大ドローダウンを計算するデータが空です。")
    wealth = 1.0 + cumulative_returns.astype(float)
    # 評価開始時の資産1も過去最高値に含め、初日からの下落を見逃さない。
    running_peak = wealth.cummax().clip(lower=1.0)
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def calculate_sharpe_ratio(
    returns: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """無リスク金利0として年率換算Sharpe Ratioを計算する。

    Args:
        returns: 日次戦略リターン。
        trading_days: 1年間の取引日数。

    Returns:
        年率Sharpe Ratio。標準偏差が0の場合は0。

    Raises:
        ValueError: 取引日数が不正な場合。
    """
    if trading_days < 1:
        raise ValueError("trading_daysは1以上で指定してください。")
    daily_volatility = float(returns.std(ddof=1))
    if np.isnan(daily_volatility) or np.isclose(daily_volatility, 0.0):
        return 0.0
    return float(returns.mean() / daily_volatility * np.sqrt(trading_days))


def calculate_win_rate(
    strategy_returns: pd.Series,
    executed_signals: pd.Series,
) -> float:
    """保有日数に対する利益日数の割合を計算する。

    Args:
        strategy_returns: 日次戦略リターン。
        executed_signals: 同じ日付の執行シグナル。

    Returns:
        Buy保有日のうちリターンが正だった割合。保有日がなければ0。

    Raises:
        ValueError: 2つのSeriesのインデックスが一致しない場合。
    """
    if not strategy_returns.index.equals(executed_signals.index):
        raise ValueError("戦略リターンと執行シグナルの日付を一致させてください。")
    active_returns = strategy_returns[executed_signals == 1]
    if active_returns.empty:
        return 0.0
    return float((active_returns > 0).mean())


def calculate_average_gain(strategy_returns: pd.Series) -> float:
    """利益が出た保有日の平均日次利益率を返す。

    Args:
        strategy_returns: 日次戦略リターン。

    Returns:
        正の日次リターンの平均。利益日がなければ0。

    Raises:
        なし。
    """
    gains = strategy_returns[strategy_returns > 0]
    return 0.0 if gains.empty else float(gains.mean())


def calculate_average_loss(strategy_returns: pd.Series) -> float:
    """損失が出た保有日の平均日次損失率を返す。

    Args:
        strategy_returns: 日次戦略リターン。

    Returns:
        負の日次リターンの平均。損失日がなければ0。

    Raises:
        なし。
    """
    losses = strategy_returns[strategy_returns < 0]
    return 0.0 if losses.empty else float(losses.mean())


def count_trades(executed_signals: pd.Series) -> int:
    """CashからBuyへ切り替わったエントリー回数を返す。

    Args:
        executed_signals: Buyを1、Cashを0とする執行シグナル。

    Returns:
        評価期間中に新しくBuyへ切り替わった回数。

    Raises:
        なし。
    """
    previous = executed_signals.shift(1, fill_value=0)
    return int(((executed_signals == 1) & (previous == 0)).sum())


def calculate_backtest_metrics(
    strategy_returns: pd.Series,
    cumulative_strategy: pd.Series,
    executed_signals: pd.Series,
    trading_days: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, MetricValue]:
    """戦略リターンから指定されたバックテスト指標を計算する。

    Args:
        strategy_returns: 日次戦略リターン。
        cumulative_strategy: 戦略の累積収益率。
        executed_signals: 翌営業日に執行済みのシグナル。
        trading_days: 年率換算に用いる年間取引日数。

    Returns:
        リターン、リスク、勝敗、取引回数を持つ辞書。

    Raises:
        ValueError: 評価期間が空の場合。
    """
    if strategy_returns.empty:
        raise ValueError("バックテスト指標を計算するデータが空です。")

    total_return = float(cumulative_strategy.iloc[-1])
    wealth = 1.0 + total_return
    annual_return = -1.0 if wealth <= 0 else float(
        wealth ** (trading_days / len(strategy_returns)) - 1.0
    )
    annual_volatility = float(strategy_returns.std(ddof=1) * np.sqrt(trading_days))
    if np.isnan(annual_volatility):
        annual_volatility = 0.0

    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Annual Volatility": annual_volatility,
        "Sharpe Ratio": calculate_sharpe_ratio(strategy_returns, trading_days),
        "Max Drawdown": calculate_max_drawdown(cumulative_strategy),
        "Win Rate": calculate_win_rate(strategy_returns, executed_signals),
        "Average Gain": calculate_average_gain(strategy_returns),
        "Average Loss": calculate_average_loss(strategy_returns),
        "Total Trades": count_trades(executed_signals),
    }


def run_backtest(
    probabilities: pd.Series,
    close_prices: pd.Series,
    threshold: float = DEFAULT_BUY_THRESHOLD,
) -> BacktestResult:
    """予測確率と終値から翌営業日執行のバックテストを実行する。

    Args:
        probabilities: 予測日をインデックスとする上昇確率。
        close_prices: 予測日とその翌日を含む終値時系列。
        threshold: Buyとする確率の下限。

    Returns:
        時系列結果と指標を持つBacktestResult。

    Raises:
        ValueError: 入力値、日付対応、または計算対象期間が不正な場合。
    """
    daily_returns = calculate_daily_returns(close_prices)
    decisions = generate_signals(probabilities, threshold)
    executed_signals = execute_signals_next_day(decisions, daily_returns.index)
    strategy_returns = calculate_strategy_returns(daily_returns, executed_signals)
    benchmark_returns = calculate_benchmark_returns(
        daily_returns, executed_signals.index
    )
    cumulative_strategy = calculate_cumulative_returns(strategy_returns)
    cumulative_strategy.name = "Cumulative_Strategy"
    cumulative_benchmark = calculate_cumulative_returns(benchmark_returns)
    cumulative_benchmark.name = "Cumulative_Benchmark"
    metrics = calculate_backtest_metrics(
        strategy_returns, cumulative_strategy, executed_signals
    )

    return BacktestResult(
        signals=executed_signals,
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        cumulative_strategy=cumulative_strategy,
        cumulative_benchmark=cumulative_benchmark,
        metrics=metrics,
    )
