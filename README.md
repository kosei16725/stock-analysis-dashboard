# Stock Analysis Dashboard

## 概要

Yahoo Financeからトヨタ自動車（7203.T）の過去1年の日足株価を取得し、終値、20日移動平均線、50日移動平均線を確認できるStreamlitアプリです。現在は第1段階（環境構築と株価チャート）まで実装しています。

## 開発目的

就職活動用ポートフォリオとして、要件整理、Pythonによるデータ処理、外部データ取得、Webアプリ、テスト、Git/GitHub、文書化、公開までの開発過程を説明できる成果物を目指します。

## 主な機能

- yfinanceによる7203.Tの日足データ取得
- 終値、20日・50日移動平均線の計算とPlotly表示
- 取得件数、開始日、終了日の表示
- ローディング、空データ、通信エラーの画面表示
- 通信に依存しないpytest

株価予測、特徴量作成、時系列評価、バックテストは今後実装予定です。

## 使用技術

- Python 3.9以上
- Streamlit / pandas / NumPy / Plotly / yfinance
- pytest

scikit-learnとLightGBMは予測機能を実装する段階で追加予定です。

## システム構成

`app.py`（画面）から、`src/data_loader.py`（取得・整形）と`src/visualization.py`（グラフ）を呼び出す構成です。詳細は [docs/system_design.md](docs/system_design.md) を参照してください。

## ディレクトリ構成

```text
stock-analysis-dashboard/
├── app.py
├── config.py
├── src/                 # 取得・計算・可視化
├── tests/               # 自動テスト
├── data/                # raw / processed / models
├── docs/                # 設計書・開発記録
├── assets/screenshots/  # README掲載用画像
├── requirements.txt
├── README.md
└── LICENSE
```

## セットアップ方法

macOSのターミナルで次を実行します。仮想環境により、このプロジェクト用のライブラリを他のPython環境から分離できます。

```bash
cd stock-analysis-dashboard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 起動方法

```bash
streamlit run app.py
```

表示されたLocal URL（通常は `http://localhost:8501`）をブラウザで開きます。

## テスト方法

```bash
python -m pytest
```

主要テストはyfinanceをモックするため、インターネット接続に依存しません。

## 画面イメージ

スクリーンショットはアプリ画面の最終調整後に `assets/screenshots/` へ追加予定です。

## 工夫した点

- UI、データ取得、可視化の役割を分離しました。
- yfinanceのMultiIndex形式にも対応しました。
- 空データや通信失敗を明示的な例外にし、アプリ全体の異常終了を防ぎました。
- 依存ライブラリを第1段階で必要なものに絞りました。

## 苦労した点と解決方法

yfinanceはバージョンや取得銘柄数により列構造が変わる場合があります。列階層から対象銘柄を探して単一階層へ統一する処理を設けました。実際の検証内容は [docs/development_log.md](docs/development_log.md) に記録します。

## 機械学習における注意点

機械学習は今後実装予定です。目的変数以外に翌営業日の情報を使わず、過去を学習、未来をテストにする時系列分割を採用します。精度だけでなくPrecision、Recall、F1、混同行列、バックテストも確認する方針です。

## 今後の改善予定

- リターン、移動平均乖離率、ボラティリティ、RSI等の特徴量
- LightGBMによる翌営業日の上昇・下落予測
- 時系列分割とウォークフォワード検証
- 手数料を考慮したバックテスト
- Streamlit Community Cloud等への公開

## 免責事項

本アプリは学習・情報提供を目的としており、投資判断を推奨・保証するものではありません。表示内容の正確性や完全性を保証せず、投資に関する最終判断は利用者自身の責任で行ってください。

## ライセンス

MIT Licenseです。詳細は [LICENSE](LICENSE) を参照してください。
