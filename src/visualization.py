"""Plotlyを使った株価グラフを生成する。"""

from typing import Sequence

import pandas as pd
import plotly.graph_objects as go

from src.constants import CLOSE_COLUMN, moving_average_column


def create_price_chart(
    data: pd.DataFrame,
    company_name: str,
    moving_average_windows: Sequence[int] = (20, 50),
) -> go.Figure:
    """終値と移動平均線を表示する折れ線グラフを作成する。

    Args:
        data: 終値と移動平均列を含むDataFrame。
        company_name: グラフタイトルに表示する会社名。
        moving_average_windows: 表示する移動平均の日数。

    Returns:
        Streamlitで表示できるPlotly Figure。

    Raises:
        ValueError: データが空、または必要列が不足している場合。
    """
    if data.empty:
        raise ValueError("グラフに表示できる株価データがありません。")
    required = [CLOSE_COLUMN] + [moving_average_column(w) for w in moving_average_windows]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"グラフ作成に必要な列がありません: {', '.join(missing)}")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=data.index, y=data[CLOSE_COLUMN], mode="lines", name="終値")
    )
    for window in moving_average_windows:
        column = moving_average_column(window)
        figure.add_trace(
            go.Scatter(x=data.index, y=data[column], mode="lines", name=f"{window}日移動平均線")
        )

    figure.update_layout(
        title=f"{company_name} 株価チャート",
        xaxis_title="日付",
        yaxis_title="株価",
        legend_title="系列",
        hovermode="x unified",
        template="plotly_white",
    )
    return figure


def create_feature_importance_chart(importance: pd.DataFrame) -> go.Figure:
    """LightGBMの特徴量重要度を横棒グラフにする。

    Args:
        importance: Feature列とImportance列を持つDataFrame。

    Returns:
        重要度が高い順に上から表示されるPlotly Figure。

    Raises:
        ValueError: データが空、または必要列が不足している場合。
    """
    if importance.empty:
        raise ValueError("表示できる特徴量重要度がありません。")
    required = {"Feature", "Importance"}
    missing = sorted(required.difference(importance.columns))
    if missing:
        raise ValueError(f"特徴量重要度に必要な列がありません: {', '.join(missing)}")

    ordered = importance.sort_values("Importance", ascending=False, kind="stable")
    figure = go.Figure(
        go.Bar(
            x=ordered["Importance"],
            y=ordered["Feature"],
            orientation="h",
            name="重要度",
        )
    )
    figure.update_layout(
        title="Feature Importance",
        xaxis_title="重要度",
        yaxis_title="特徴量",
        template="plotly_white",
    )
    figure.update_yaxes(autorange="reversed")
    return figure


def create_cumulative_returns_chart(
    cumulative_strategy: pd.Series,
    cumulative_benchmark: pd.Series,
) -> go.Figure:
    """戦略とBuy & Holdの累積リターン比較グラフを作成する。

    Args:
        cumulative_strategy: 戦略の累積リターン。
        cumulative_benchmark: Buy & Holdの累積リターン。

    Returns:
        2系列を同じ日付軸で比較するPlotly Figure。

    Raises:
        ValueError: Seriesが空、または日付インデックスが一致しない場合。
    """
    if cumulative_strategy.empty or cumulative_benchmark.empty:
        raise ValueError("表示できる累積リターンがありません。")
    if not cumulative_strategy.index.equals(cumulative_benchmark.index):
        raise ValueError("戦略とBuy & Holdの日付が一致しません。")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=cumulative_strategy.index,
            y=cumulative_strategy,
            mode="lines",
            name="Strategy",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=cumulative_benchmark.index,
            y=cumulative_benchmark,
            mode="lines",
            name="Buy & Hold",
        )
    )
    figure.update_layout(
        title="累積リターン比較",
        xaxis_title="日付",
        yaxis_title="累積リターン",
        yaxis_tickformat=".1%",
        legend_title="運用方法",
        hovermode="x unified",
        template="plotly_white",
    )
    return figure
