"""翌営業日執行バックテストの通信非依存テスト。"""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.backtest import (
    calculate_cumulative_returns,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    generate_signals,
    run_backtest,
)


@pytest.fixture
def close_prices() -> pd.Series:
    """リターンが既知の8営業日分の人工終値を返す。"""
    index = pd.date_range("2025-01-01", periods=8, freq="B", name="Date")
    return pd.Series(
        [100.0, 110.0, 99.0, 108.9, 108.9, 119.79, 107.811, 118.5921],
        index=index,
        name="Close",
    )


@pytest.fixture
def probabilities(close_prices: pd.Series) -> pd.Series:
    """終値の先頭7日を予測日とする固定確率を返す。"""
    return pd.Series(
        [0.60, 0.40, 0.55, 0.20, 0.90, 0.80, 0.10],
        index=close_prices.index[:-1],
        name="Probability",
    )


def test_signal_is_executed_on_next_business_day(
    probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """予測日の判断が同日でなく翌営業日のシグナルになることを確認する。"""
    result = run_backtest(probabilities, close_prices)

    assert result.signals.index[0] == probabilities.index[0]
    assert result.signals.index[-1] == close_prices.index[-1]
    assert result.signals.tolist() == [0, 1, 0, 1, 0, 1, 1, 0]


def test_future_probabilities_do_not_change_past_signals(
    probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """将来の予測確率を変更しても過去の執行シグナルが変わらない。"""
    changed = probabilities.copy()
    changed.iloc[4:] = 1.0 - changed.iloc[4:]

    original = run_backtest(probabilities, close_prices)
    modified = run_backtest(changed, close_prices)

    pdt.assert_series_equal(original.signals.iloc[:4], modified.signals.iloc[:4])
    pdt.assert_series_equal(
        original.strategy_returns.iloc[:4], modified.strategy_returns.iloc[:4]
    )


def test_cumulative_return_uses_compounding() -> None:
    """累積リターンが単純加算でなく複利計算されることを確認する。"""
    returns = pd.Series([0.10, -0.10])

    cumulative = calculate_cumulative_returns(returns)

    assert cumulative.iloc[-1] == pytest.approx(-0.01)


def test_max_drawdown() -> None:
    """過去最高資産からの最大下落率を計算できることを確認する。"""
    cumulative = pd.Series([0.10, 0.21, 0.089, 0.1979])

    assert calculate_max_drawdown(cumulative) == pytest.approx(-0.10)


def test_max_drawdown_includes_initial_capital() -> None:
    """評価開始直後の下落も初期資産1からのドローダウンとして扱う。"""
    cumulative = pd.Series([-0.10, -0.05])

    assert calculate_max_drawdown(cumulative) == pytest.approx(-0.10)


def test_sharpe_ratio_is_annualized() -> None:
    """日次平均と標準偏差から年率Sharpe Ratioを計算する。"""
    returns = pd.Series([0.01, -0.01, 0.02, 0.00])
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)

    assert calculate_sharpe_ratio(returns) == pytest.approx(expected)


def test_win_rate_and_trade_count(
    probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """保有日の勝率とCashからBuyへの切替回数を確認する。"""
    result = run_backtest(probabilities, close_prices)

    assert result.metrics["Win Rate"] == pytest.approx(0.75)
    assert result.metrics["Average Gain"] == pytest.approx(0.10)
    assert result.metrics["Average Loss"] == pytest.approx(-0.10)
    assert result.metrics["Total Trades"] == 3


def test_strategy_and_buy_hold_are_compared_on_same_dates(
    probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """戦略とBuy & Holdが同じ執行期間・実現リターンで比較される。"""
    result = run_backtest(probabilities, close_prices)

    assert result.strategy_returns.index.equals(result.benchmark_returns.index)
    assert result.cumulative_strategy.iloc[-1] == pytest.approx(0.1979)
    assert result.cumulative_benchmark.iloc[-1] == pytest.approx(0.185921)


def test_signal_threshold_includes_055(probabilities: pd.Series) -> None:
    """確率0.55をBuyに含み、それ未満をCashにする。"""
    signals = generate_signals(probabilities)

    assert signals.iloc[1] == 0
    assert signals.iloc[2] == 1


def test_backtest_metric_keys(
    probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """BacktestResultが指定された9指標をすべて持つ。"""
    result = run_backtest(probabilities, close_prices)

    assert set(result.metrics) == {
        "Total Return",
        "Annual Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Win Rate",
        "Average Gain",
        "Average Loss",
        "Total Trades",
    }


@pytest.fixture
def nonconsecutive_probabilities(close_prices: pd.Series) -> pd.Series:
    """予測日の間に複数営業日の空白がある固定確率を返す。"""
    return pd.Series(
        [0.60, 0.40, 0.70],
        index=close_prices.index[[0, 3, 6]],
        name="Probability",
    )


def test_nonconsecutive_predictions_preserve_full_trading_calendar(
    nonconsecutive_probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """予測がない途中営業日も戦略とベンチマークから欠落しない。"""
    result = run_backtest(nonconsecutive_probabilities, close_prices)

    assert result.strategy_returns.index.equals(close_prices.index)
    assert result.benchmark_returns.index.equals(close_prices.index)
    assert len(result.strategy_returns) == len(close_prices)


def test_buy_position_persists_until_next_nonconsecutive_signal(
    nonconsecutive_probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """Buy執行後、次のCashシグナル執行まではBuyを維持する。"""
    result = run_backtest(nonconsecutive_probabilities, close_prices)

    assert result.signals.loc[close_prices.index[1:4]].tolist() == [1, 1, 1]


def test_cash_position_persists_until_next_nonconsecutive_signal(
    nonconsecutive_probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """Cash執行後、次のBuyシグナル執行まではCashを維持する。"""
    result = run_backtest(nonconsecutive_probabilities, close_prices)

    assert result.signals.loc[close_prices.index[4:7]].tolist() == [0, 0, 0]


def test_position_is_cash_before_first_signal_execution(
    nonconsecutive_probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """最初の予測日にはまだ売買せずCashである。"""
    result = run_backtest(nonconsecutive_probabilities, close_prices)

    assert result.signals.iloc[0] == 0
    assert result.strategy_returns.iloc[0] == 0.0


def test_nonconsecutive_buy_hold_uses_complete_same_period(
    nonconsecutive_probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """Buy & Holdが省略のない戦略評価期間と完全に一致する。"""
    result = run_backtest(nonconsecutive_probabilities, close_prices)

    assert result.benchmark_returns.index.equals(result.strategy_returns.index)
    assert result.cumulative_benchmark.iloc[-1] == pytest.approx(0.185921)


def test_nonconsecutive_metrics_include_intervening_days(
    nonconsecutive_probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """非連続予測でも全営業日を使った主要指標と取引回数になる。"""
    result = run_backtest(nonconsecutive_probabilities, close_prices)
    expected_returns = pd.Series(
        [0.0, 0.10, -0.10, 0.10, 0.0, 0.0, 0.0, 0.10],
        index=close_prices.index,
    )

    assert result.metrics["Total Return"] == pytest.approx(0.1979)
    assert result.metrics["Sharpe Ratio"] == pytest.approx(
        calculate_sharpe_ratio(expected_returns)
    )
    assert result.metrics["Max Drawdown"] == pytest.approx(-0.10)
    assert result.metrics["Total Trades"] == 2


def test_future_nonconsecutive_prediction_does_not_change_past(
    nonconsecutive_probabilities: pd.Series,
    close_prices: pd.Series,
) -> None:
    """将来の予測変更が、それ以前の維持ポジションとリターンへ影響しない。"""
    changed = nonconsecutive_probabilities.copy()
    changed.iloc[-1] = 0.10

    original = run_backtest(nonconsecutive_probabilities, close_prices)
    modified = run_backtest(changed, close_prices)
    past_dates = close_prices.index[:7]

    pdt.assert_series_equal(
        original.signals.loc[past_dates], modified.signals.loc[past_dates]
    )
    pdt.assert_series_equal(
        original.strategy_returns.loc[past_dates],
        modified.strategy_returns.loc[past_dates],
    )
