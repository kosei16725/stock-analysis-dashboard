# Stock Analysis Dashboard

## 概要

Yahoo Financeからトヨタ自動車（7203.T）の過去1年の日足株価を取得し、終値、20日移動平均線、50日移動平均線を確認できるStreamlitアプリです。現在は第2段階として、未来情報を含まない特徴量と翌営業日の方向を表す目的変数の作成まで実装しています。モデル学習と画面への予測表示は今後実装予定です。

## 開発目的

就職活動用ポートフォリオとして、要件整理、Pythonによるデータ処理、外部データ取得、Webアプリ、テスト、Git/GitHub、文書化、公開までの開発過程を説明できる成果物を目指します。

## 主な機能

- yfinanceによる7203.Tの日足データ取得
- 終値、20日・50日移動平均線の計算とPlotly表示
- 取得件数、開始日、終了日の表示
- ローディング、空データ、通信エラーの画面表示
- 通信に依存しないpytest
- Daily Return、5日・20日リターン
- MA5、MA20、MA50、MA20乖離率
- 20日ボラティリティ、出来高変化率
- 翌営業日の終値が上昇したかを表すTarget（上昇=1、それ以外=0）
- 特徴量とTargetに必要な行だけを対象にした欠損処理

株価予測、学習・テスト分割、時系列評価、バックテストは今後実装予定です。

## 使用技術

- Python 3.9以上
- Streamlit / pandas / NumPy / Plotly / yfinance
- pytest

scikit-learnとLightGBMは予測機能を実装する段階で追加予定です。

## システム構成

`app.py`（画面）から、`src/data_loader.py`（取得・整形）と`src/visualization.py`（グラフ）を呼び出します。学習用データは `src/features.py` が株価DataFrameから独立して作成します。第2段階ではまだUIから特徴量処理を呼び出しません。詳細は [docs/system_design.md](docs/system_design.md) を参照してください。

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
- 特徴量計算では当日を終点とする処理だけを使い、翌日情報をTarget以外へ混入させない設計にしました。
- 欠損処理の対象列を特徴量とTargetに限定し、無関係な列の欠損による行削除を防ぎました。

## 苦労した点と解決方法

yfinanceはバージョンや取得銘柄数により列構造が変わる場合があります。列階層から対象銘柄を探して単一階層へ統一する処理を設けました。実際の検証内容は [docs/development_log.md](docs/development_log.md) に記録します。

## 機械学習における注意点

モデル学習は今後実装予定です。第2段階の特徴量は当日までの情報だけで計算し、翌営業日終値はTargetの作成だけに使用しています。次段階以降も過去を学習、未来をテストにする時系列分割を採用します。精度だけでなくPrecision、Recall、F1、混同行列、バックテストも確認する方針です。

## 今後の改善予定

- RSIなどの追加特徴量
- LightGBMによる翌営業日の上昇・下落予測
- 時系列分割とウォークフォワード検証
- 手数料を考慮したバックテスト
- Streamlit Community Cloud等への公開

## 免責事項

本アプリは学習・情報提供を目的としており、投資判断を推奨・保証するものではありません。表示内容の正確性や完全性を保証せず、投資に関する最終判断は利用者自身の責任で行ってください。

## ライセンス

MIT Licenseです。詳細は [LICENSE](LICENSE) を参照してください。
