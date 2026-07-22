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
        yaxis_title="株価（円）",
        legend_title="系列",
        hovermode="x unified",
        template="plotly_white",
    )
    return figure
