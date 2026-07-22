"""Stock Analysis DashboardのStreamlitエントリーポイント。"""

import logging

import pandas as pd
import streamlit as st

from config import APP_CONFIG
from src.constants import DISCLAIMER
from src.data_loader import StockDataError, fetch_stock_data
from src.visualization import create_price_chart

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


@st.cache_data(ttl=3600, show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    """設定済み銘柄のデータを取得し、1時間キャッシュする。

    Returns:
        移動平均を含む株価DataFrame。

    Raises:
        ValueError: 設定値が不正な場合。
        StockDataError: 株価の取得または検証に失敗した場合。
    """
    return fetch_stock_data(
        ticker=APP_CONFIG.ticker,
        period=APP_CONFIG.period,
        moving_average_windows=APP_CONFIG.moving_average_windows,
    )


def main() -> None:
    """Streamlit画面を構成し、株価チャートを表示する。

    Returns:
        なし。

    Raises:
        なし。想定される例外は画面へ表示し、ログへ記録する。
    """
    st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
    st.title("Stock Analysis Dashboard")
    st.write("株価データを取得し、終値と移動平均線を確認する学習用Webアプリです。")
    st.warning(DISCLAIMER)

    info_columns = st.columns(3)
    info_columns[0].metric("対象銘柄", APP_CONFIG.company_name)
    info_columns[1].metric("銘柄コード", APP_CONFIG.ticker)
    info_columns[2].metric("分析期間", APP_CONFIG.period_label)

    st.subheader("データ取得状況")
    try:
        with st.spinner("Yahoo Financeから株価データを取得しています..."):
            data = load_dashboard_data()
    except (StockDataError, ValueError) as exc:
        LOGGER.warning("株価データを表示できませんでした: %s", exc)
        st.error(f"株価データを表示できませんでした。原因: {exc}")
        st.info("インターネット接続を確認し、時間を置いて再読み込みしてください。")
        return
    except Exception as exc:  # UI全体の異常終了を防ぐ最後の防壁
        LOGGER.exception("予期しないエラーが発生しました")
        st.error("予期しないエラーが発生しました。詳細はアプリのログを確認してください。")
        return

    st.success("株価データを取得しました。")
    summary_columns = st.columns(3)
    summary_columns[0].metric("取得したデータ件数", f"{len(data):,} 件")
    summary_columns[1].metric("データの開始日", data.index.min().strftime("%Y-%m-%d"))
    summary_columns[2].metric("データの終了日", data.index.max().strftime("%Y-%m-%d"))

    chart = create_price_chart(
        data=data,
        company_name=APP_CONFIG.company_name,
        moving_average_windows=APP_CONFIG.moving_average_windows,
    )
    st.plotly_chart(chart, use_container_width=True)
    moving_average_label = "・".join(
        f"{window}日" for window in APP_CONFIG.moving_average_windows
    )
    st.caption(f"表示系列: 終値 / {moving_average_label}移動平均線")


if __name__ == "__main__":
    main()
