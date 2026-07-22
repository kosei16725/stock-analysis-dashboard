"""Stock Analysis DashboardのStreamlitエントリーポイント。"""

import logging

import pandas as pd
import streamlit as st

from config import APP_CONFIG
from src.constants import DISCLAIMER
from src.data_loader import StockDataError, fetch_stock_data
from src.backtest import BacktestResult, run_backtest
from src.features import FeatureEngineeringError, create_training_data
from src.model import ModelResult, run_model_pipeline
from src.visualization import (
    create_cumulative_returns_chart,
    create_feature_importance_chart,
    create_price_chart,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False)
def load_dashboard_data(ticker: str, period: str) -> pd.DataFrame:
    """指定銘柄のデータを取得し、入力ごとに1時間キャッシュする。

    Args:
        ticker: Yahoo Finance形式の銘柄コード。
        period: ``1y`` などの取得期間。

    Returns:
        移動平均を含む株価DataFrame。

    Raises:
        ValueError: 設定値が不正な場合。
        StockDataError: 株価の取得または検証に失敗した場合。
    """
    return fetch_stock_data(
        ticker=ticker,
        period=period,
        moving_average_windows=APP_CONFIG.moving_average_windows,
    )


def format_metric(value: object, percentage: bool = False) -> str:
    """画面表示用に評価指標を整形する。

    Args:
        value: 数値、None、または変換可能な値。
        percentage: Trueの場合は百分率で表示する。

    Returns:
        小数4桁または百分率の文字列。Noneは「算出不可」。

    Raises:
        ValueError: 数値へ変換できない場合。
    """
    if value is None or pd.isna(value):
        return "算出不可"
    numeric_value = float(value)
    return f"{numeric_value:.2%}" if percentage else f"{numeric_value:.4f}"


def render_data_summary(model_result: ModelResult) -> None:
    """学習・テスト件数と期間を画面へ表示する。

    Args:
        model_result: 時系列分割済みのモデル結果。

    Returns:
        なし。

    Raises:
        なし。
    """
    st.subheader("データ概要")
    columns = st.columns(4)
    columns[0].metric("学習件数", f"{len(model_result.X_train):,} 件")
    columns[1].metric("テスト件数", f"{len(model_result.X_test):,} 件")
    columns[2].metric(
        "学習期間",
        f"{model_result.X_train.index.min():%Y-%m-%d} ～ "
        f"{model_result.X_train.index.max():%Y-%m-%d}",
    )
    columns[3].metric(
        "テスト期間",
        f"{model_result.X_test.index.min():%Y-%m-%d} ～ "
        f"{model_result.X_test.index.max():%Y-%m-%d}",
    )


def render_model_metrics(model_result: ModelResult) -> None:
    """5種類のモデル評価指標を画面へ表示する。

    Args:
        model_result: 評価指標を含むモデル結果。

    Returns:
        なし。

    Raises:
        ValueError: 指標が数値へ変換できない場合。
    """
    st.subheader("モデル評価")
    names = ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC")
    columns = st.columns(len(names))
    for column, name in zip(columns, names):
        column.metric(name, format_metric(model_result.metrics[name]))


def render_feature_importance(model_result: ModelResult) -> None:
    """Gain・Splitの特徴量重要度をグラフと表で表示する。

    Args:
        model_result: 特徴量重要度を含むモデル結果。

    Returns:
        なし。

    Raises:
        ValueError: 特徴量重要度の形式が不正な場合。
    """
    st.subheader("特徴量重要度")
    importance = model_result.feature_importance
    top_features = importance.head(5)["Feature"].tolist()
    zero_gain_count = int((importance["Gain_Importance"] == 0).sum())
    zero_split_count = int((importance["Split_Importance"] == 0).sum())

    summary_columns = st.columns(3)
    summary_columns[0].metric(
        "Gain重要度0の特徴量数", f"{zero_gain_count} / {len(importance)}"
    )
    summary_columns[1].metric(
        "Split重要度0の特徴量数", f"{zero_split_count} / {len(importance)}"
    )
    summary_columns[2].write("Gain Importance 上位5特徴量")
    summary_columns[2].write("、".join(top_features))

    gain_tab, split_tab = st.tabs(["Gain Importance", "Split Importance"])
    with gain_tab:
        st.plotly_chart(
            create_feature_importance_chart(importance, "Gain"),
            use_container_width=True,
        )
    with split_tab:
        st.plotly_chart(
            create_feature_importance_chart(importance, "Split"),
            use_container_width=True,
        )
    st.dataframe(
        importance.loc[
            :,
            [
                "Feature",
                "Gain_Importance",
                "Gain_Percentage",
                "Split_Importance",
                "Split_Percentage",
            ],
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "特徴量重要度は因果関係や将来の有効性を示すものではありません。"
    )


def render_backtest_metrics(backtest_result: BacktestResult) -> None:
    """指定された6種類のバックテスト指標を表示する。

    Args:
        backtest_result: 指標を含むバックテスト結果。

    Returns:
        なし。

    Raises:
        ValueError: 指標が数値へ変換できない場合。
    """
    st.subheader("バックテスト")
    metrics = (
        ("Total Return", True),
        ("Annual Return", True),
        ("Sharpe Ratio", False),
        ("Max Drawdown", True),
        ("Win Rate", True),
    )
    first_row = st.columns(3)
    second_row = st.columns(3)
    for column, (name, percentage) in zip(first_row + second_row[:2], metrics):
        column.metric(name, format_metric(backtest_result.metrics[name], percentage))
    second_row[2].metric("Total Trades", str(backtest_result.metrics["Total Trades"]))


def main() -> None:
    """入力から取得・学習・バックテスト・可視化までを画面上で実行する。

    Returns:
        なし。

    Raises:
        なし。想定される例外は画面へ表示し、ログへ記録する。
    """
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("Stock Analysis Dashboard")
    st.write(
        "株価データの取得、特徴量作成、LightGBM評価、"
        "簡易バックテストを実行する学習用Webアプリです。"
    )
    st.warning(DISCLAIMER)

    with st.form("analysis_form"):
        input_columns = st.columns(3)
        ticker = input_columns[0].text_input(
            "銘柄コード",
            value=APP_CONFIG.ticker,
            help="Yahoo Finance形式で入力してください（例: 7203.T、AAPL）。",
        )
        period = input_columns[1].text_input(
            "取得期間",
            value=APP_CONFIG.period,
            help="例: 6mo、1y、2y、5y。短すぎる期間では学習できません。",
        )
        buy_threshold = input_columns[2].number_input(
            "Buy閾値",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.01,
            format="%.2f",
            help="上昇確率がこの値以上ならBuyと判断します。",
        )
        analysis_started = st.form_submit_button("分析開始", type="primary")

    if not analysis_started:
        st.info("銘柄コード、取得期間、Buy閾値を確認して「分析開始」を押してください。")
        return

    st.subheader("データ取得状況")
    try:
        with st.spinner("Yahoo Financeから株価データを取得しています..."):
            data = load_dashboard_data(ticker, period)
    except (StockDataError, ValueError) as exc:
        LOGGER.warning("株価データの取得に失敗しました: %s", exc)
        st.error(f"株価データを取得できませんでした。原因: {exc}")
        st.info("銘柄コード・取得期間・インターネット接続を確認してください。")
        return
    except Exception:
        LOGGER.exception("株価データ取得中に予期しないエラーが発生しました")
        st.error("株価データ取得中に予期しないエラーが発生しました。ログを確認してください。")
        return

    st.success("株価データを取得しました。")
    company_label = APP_CONFIG.company_name if ticker.strip().upper() == APP_CONFIG.ticker else ticker.upper()
    price_chart = create_price_chart(
        data=data,
        company_name=company_label,
        moving_average_windows=APP_CONFIG.moving_average_windows,
    )
    st.plotly_chart(price_chart, use_container_width=True)
    moving_average_label = "・".join(
        f"{window}日" for window in APP_CONFIG.moving_average_windows
    )
    st.caption(f"表示系列: 終値 / {moving_average_label}移動平均線")

    try:
        with st.spinner("特徴量と学習用データを作成しています..."):
            training_data = create_training_data(data)
    except (FeatureEngineeringError, ValueError) as exc:
        LOGGER.warning("特徴量作成に失敗しました: %s", exc)
        st.error(f"学習用データを作成できませんでした。原因: {exc}")
        st.info("より長い取得期間を指定してください。")
        return
    except Exception:
        LOGGER.exception("特徴量作成中に予期しないエラーが発生しました")
        st.error("特徴量作成中に予期しないエラーが発生しました。ログを確認してください。")
        return

    try:
        with st.spinner("LightGBMを学習し、テスト期間を評価しています..."):
            model_result = run_model_pipeline(training_data)
    except ValueError as exc:
        LOGGER.warning("モデル学習に失敗しました: %s", exc)
        st.error(f"モデルを学習できませんでした。原因: {exc}")
        st.info("十分な件数と、Targetの0・1を両方含む期間を指定してください。")
        return
    except Exception:
        LOGGER.exception("モデル学習中に予期しないエラーが発生しました")
        st.error("モデル学習中に予期しないエラーが発生しました。ログを確認してください。")
        return

    probabilities = pd.Series(
        model_result.probabilities,
        index=model_result.X_test.index,
        name="Probability",
    )
    try:
        with st.spinner("翌営業日執行のバックテストを実行しています..."):
            # 取得途中の未確定行は0リターンにせず、終値が確定した営業日だけを使う。
            valid_close = data["Close"].dropna()
            backtest_result = run_backtest(
                probabilities=probabilities,
                close_prices=valid_close,
                threshold=float(buy_threshold),
            )
    except ValueError as exc:
        LOGGER.warning("バックテストに失敗しました: %s", exc)
        st.error(f"バックテストを実行できませんでした。原因: {exc}")
        st.info("テスト期間の翌営業日を含む終値データが必要です。")
        return
    except Exception:
        LOGGER.exception("バックテスト中に予期しないエラーが発生しました")
        st.error("バックテスト中に予期しないエラーが発生しました。ログを確認してください。")
        return

    st.success("分析が完了しました。")
    render_data_summary(model_result)
    render_model_metrics(model_result)
    render_feature_importance(model_result)
    render_backtest_metrics(backtest_result)
    st.plotly_chart(
        create_cumulative_returns_chart(
            backtest_result.cumulative_strategy,
            backtest_result.cumulative_benchmark,
        ),
        use_container_width=True,
    )
    st.caption(
        "バックテストは取引手数料・税金・スリッページを考慮していません。"
        "過去の結果は将来の利益を保証しません。"
    )


if __name__ == "__main__":
    main()
