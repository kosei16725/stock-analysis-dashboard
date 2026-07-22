# Stock Analysis Dashboard

## 概要

Yahoo Financeからトヨタ自動車（7203.T）の過去1年の日足株価を取得し、終値、20日移動平均線、50日移動平均線を確認できるStreamlitアプリです。現在は未来情報を含まない特徴量、時系列分割、LightGBMの学習・評価、予測確率を使った簡易バックテストまで実装しています。モデルとバックテスト結果の画面表示は今後実装予定です。

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
- 過去80%を学習、未来20%をテストにする時系列分割
- LightGBMによる分類、上昇確率、特徴量重要度
- Accuracy、Precision、Recall、F1、ROC-AUCによる評価
- 上昇確率0.55以上をBuy、それ未満をCashとするロング戦略
- 予測翌営業日の執行、Buy & Hold比較、リターン・リスク・売買指標

予測・バックテスト結果のStreamlit表示は今後実装予定です。

## 使用技術

- Python 3.9以上
- Streamlit / pandas / NumPy / Plotly / yfinance
- scikit-learn / LightGBM
- pytest

依存バージョンはPython 3.9とmacOS ARM環境で動作確認した値に固定しています。

## システム構成

`app.py`（画面）から、`src/data_loader.py`（取得・整形）と`src/visualization.py`（グラフ）を呼び出します。分析処理は `src/features.py` が学習用DataFrameを作り、`src/model.py` が時系列分割、学習、予測、評価、`src/backtest.py` が翌営業日執行の売買検証を担当します。現段階ではまだUIから学習・バックテスト処理を呼び出しません。詳細は [docs/system_design.md](docs/system_design.md) を参照してください。

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
- 日付で昇順に並べた後、過去側だけを学習に使い、未来側を評価に残しました。
- モデル、分割データ、予測、評価、重要度を小さなdataclassで一つの結果として扱えるようにしました。
- 予測日のシグナルを実際の取引日カレンダー上で翌営業日へ移し、同日のリターンへ適用しています。
- 戦略とBuy & Holdを同じ期間で比較し、複利リターンとリスク指標を確認できます。

## 苦労した点と解決方法

yfinanceはバージョンや取得銘柄数により列構造が変わる場合があります。列階層から対象銘柄を探して単一階層へ統一する処理を設けました。実際の検証内容は [docs/development_log.md](docs/development_log.md) に記録します。

## 機械学習における注意点

第2段階の特徴量は当日までの情報だけで計算し、翌営業日終値はTargetの作成だけに使用しています。株価では未来のデータを過去の学習へ混ぜられないため、ランダム分割を使わず、過去80%を学習、未来20%をテストにしています。

LightGBMは表形式データで非線形な関係を扱え、特徴量重要度を取得できるため採用しました。現在はAccuracy、Precision、Recall、F1、ROC-AUCを確認できます。ただし、限られた期間と単純な1回分割による結果であり、将来の予測性能や利益を保証するものではありません。テスト期間が単一クラスの場合、定義できないROC-AUCは `None` とします。

バックテストは上昇確率が0.55以上ならBuy、それ以外はCashとし、判断の翌営業日に執行します。予測日が非連続でも全取引日を残し、次のシグナルが執行されるまでは直前のBuyまたはCashを維持します。最初の執行前はCashで、未来方向への補完は行いません。ショート、取引手数料、税金、スリッページはまだ考慮していません。Total Return、Annual Return、Annual Volatility、Sharpe Ratio、Max Drawdown、Win Rate、Average Gain、Average Loss、Total Tradesを算出します。過去データ上の結果であり、将来の利益を保証するものではありません。

## 今後の改善予定

- RSIなどの追加特徴量
- ウォークフォワード検証とハイパーパラメータ検討
- 手数料、税金、スリッページを考慮したバックテスト
- ショートや売買閾値の検討
- 予測、評価指標、特徴量重要度のStreamlit表示
- Streamlit Community Cloud等への公開

## 免責事項

本アプリは学習・情報提供を目的としており、投資判断を推奨・保証するものではありません。表示内容の正確性や完全性を保証せず、投資に関する最終判断は利用者自身の責任で行ってください。

## ライセンス

MIT Licenseです。詳細は [LICENSE](LICENSE) を参照してください。
