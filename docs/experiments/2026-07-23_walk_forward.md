# Phase 11 Walk-Forward Validation Experiment

## Experiment Purpose

Phase 10で比較したAll 20、Gain上位15、Gain上位10、Baseline 9を、単一の80/20分割ではなく複数の時系列Foldで評価する。期間が変わった場合の分類・バックテスト指標と、Foldごとに選び直す特徴量の変化を確認する。

## Expanding Window

Fold 1は先頭100件を分割直前の学習期間、直後20件をテスト期間とする。以降は学習窓の開始を固定したまま末尾を20件ずつ拡張し、直後の20件を評価する。不完全な末尾Foldは使用しない。

各Foldの学習末尾Targetはテスト初日の終値を参照するため1件パージする。したがって、Fold 1は分割直後100件、実学習99件となる。Gain順位と4特徴量セットのモデルは、すべて同じパージ済み学習データを使用する。

## Conditions

- Ticker: 7203.T
- Period: 2y
- Price Rows: 488
- Model Samples: 438
- Initial Train Size: 100
- Test Size: 20
- Step Size: 20
- Complete Folds: 16
- Buy Threshold: 0.55
- Model / Hyperparameters: 既存LightGBMと同一
- Execution Date: 2026-07-23

## Data Leakage Controls

- Foldは日付昇順で作成し、各Foldの全train日をtest開始日より前にする。
- 学習末尾1件をパージし、テスト初日の価格から作られる境界Targetをランキングとモデル学習へ渡さない。
- Top 15 / Top 10は各Foldのパージ済みtrainだけでGain順位を再計算する。
- test特徴量、test Target、将来リターンを特徴量順位へ使用しない。
- 4セットはFold内で同じtrain、test、モデル設定、Buy Thresholdを使用する。
- バックテスト価格は最初のtest予測日から最後のtest予測日の実際の翌取引日までに限定する。

## Fold Overview

| Fold | Train Before / After Purge | Train Period | Test Period | Backtest End |
|---:|---:|---|---|---|
| 1 | 100 / 99 | 2024-10-03 ～ 2025-03-03 | 2025-03-05 ～ 2025-04-02 | 2025-04-03 |
| 2 | 120 / 119 | 2024-10-03 ～ 2025-04-01 | 2025-04-03 ～ 2025-05-01 | 2025-05-02 |
| 3 | 140 / 139 | 2024-10-03 ～ 2025-04-30 | 2025-05-02 ～ 2025-06-02 | 2025-06-03 |
| 4 | 160 / 159 | 2024-10-03 ～ 2025-05-30 | 2025-06-03 ～ 2025-06-30 | 2025-07-01 |
| 5 | 180 / 179 | 2024-10-03 ～ 2025-06-27 | 2025-07-01 ～ 2025-07-29 | 2025-07-30 |
| 6 | 200 / 199 | 2024-10-03 ～ 2025-07-28 | 2025-07-30 ～ 2025-08-27 | 2025-08-28 |
| 7 | 220 / 219 | 2024-10-03 ～ 2025-08-26 | 2025-08-28 ～ 2025-09-26 | 2025-09-29 |
| 8 | 240 / 239 | 2024-10-03 ～ 2025-09-25 | 2025-09-29 ～ 2025-10-27 | 2025-10-28 |
| 9 | 260 / 259 | 2024-10-03 ～ 2025-10-24 | 2025-10-28 ～ 2025-11-26 | 2025-11-27 |
| 10 | 280 / 279 | 2024-10-03 ～ 2025-11-25 | 2025-11-27 ～ 2025-12-24 | 2025-12-25 |
| 11 | 300 / 299 | 2024-10-03 ～ 2025-12-23 | 2025-12-25 ～ 2026-01-27 | 2026-01-28 |
| 12 | 320 / 319 | 2024-10-03 ～ 2026-01-26 | 2026-01-28 ～ 2026-02-26 | 2026-02-27 |
| 13 | 340 / 339 | 2024-10-03 ～ 2026-02-25 | 2026-02-27 ～ 2026-03-27 | 2026-03-30 |
| 14 | 360 / 359 | 2024-10-03 ～ 2026-03-26 | 2026-03-30 ～ 2026-04-24 | 2026-04-27 |
| 15 | 380 / 379 | 2024-10-03 ～ 2026-04-23 | 2026-04-27 ～ 2026-05-28 | 2026-05-29 |
| 16 | 400 / 399 | 2024-10-03 ～ 2026-05-27 | 2026-05-29 ～ 2026-06-25 | 2026-06-26 |

## Selected Features by Fold

All 20は`FEATURE_COLUMNS`全列、Baseline 9は`Daily_Return, Return_5D, Return_20D, MA_5, MA_20, MA_50, MA_Deviation_20, Volatility_20D, Volume_Change`で全Fold固定とした。

| Fold | Top 15 by Gain Importance | Top 10 by Gain Importance |
|---:|---|---|
| 1 | Volatility_20D, MACD, MA_Deviation_20, BB_Percent_B_20, RSI_14, Daily_Return, BB_Std_20, MA_50, MA_20, BB_Lower_20, BB_Width_20, Volume_Change, MACD_Signal, Return_5D, BB_Upper_20 | Volatility_20D, MACD, MA_Deviation_20, BB_Percent_B_20, RSI_14, Daily_Return, BB_Std_20, MA_50, MA_20, BB_Lower_20 |
| 2 | Volatility_20D, MA_Deviation_20, Daily_Return, BB_Percent_B_20, MACD_Signal, BB_Lower_20, Return_5D, Volume_Change, BB_Std_20, MA_20, RSI_14, BB_Width_20, MA_5, EMA_26, BB_Upper_20 | Volatility_20D, MA_Deviation_20, Daily_Return, BB_Percent_B_20, MACD_Signal, BB_Lower_20, Return_5D, Volume_Change, BB_Std_20, MA_20 |
| 3 | Daily_Return, BB_Upper_20, BB_Percent_B_20, Return_5D, BB_Lower_20, Volume_Change, BB_Std_20, MACD_Signal, MA_5, MA_Deviation_20, Volatility_20D, MACD_Histogram, EMA_12, MA_50, EMA_26 | Daily_Return, BB_Upper_20, BB_Percent_B_20, Return_5D, BB_Lower_20, Volume_Change, BB_Std_20, MACD_Signal, MA_5, MA_Deviation_20 |
| 4 | Daily_Return, MACD_Signal, Return_5D, Volatility_20D, BB_Upper_20, BB_Lower_20, Volume_Change, BB_Percent_B_20, BB_Width_20, RSI_14, BB_Std_20, EMA_26, MA_5, MA_Deviation_20, Return_20D | Daily_Return, MACD_Signal, Return_5D, Volatility_20D, BB_Upper_20, BB_Lower_20, Volume_Change, BB_Percent_B_20, BB_Width_20, RSI_14 |
| 5 | Daily_Return, BB_Upper_20, RSI_14, MA_50, Volume_Change, Return_5D, BB_Std_20, BB_Lower_20, MACD_Signal, MA_5, BB_Percent_B_20, MACD_Histogram, MA_20, MA_Deviation_20, Return_20D | Daily_Return, BB_Upper_20, RSI_14, MA_50, Volume_Change, Return_5D, BB_Std_20, BB_Lower_20, MACD_Signal, MA_5 |
| 6 | Daily_Return, MACD_Signal, BB_Lower_20, BB_Upper_20, Return_5D, BB_Percent_B_20, Volatility_20D, MA_5, BB_Width_20, MA_50, BB_Std_20, EMA_26, Volume_Change, Return_20D, MACD_Histogram | Daily_Return, MACD_Signal, BB_Lower_20, BB_Upper_20, Return_5D, BB_Percent_B_20, Volatility_20D, MA_5, BB_Width_20, MA_50 |
| 7 | Daily_Return, MACD_Signal, MA_5, BB_Percent_B_20, BB_Upper_20, Return_5D, BB_Width_20, Volatility_20D, BB_Lower_20, MACD_Histogram, Volume_Change, RSI_14, EMA_12, BB_Std_20, EMA_26 | Daily_Return, MACD_Signal, MA_5, BB_Percent_B_20, BB_Upper_20, Return_5D, BB_Width_20, Volatility_20D, BB_Lower_20, MACD_Histogram |
| 8 | Daily_Return, Volatility_20D, Return_5D, Volume_Change, MACD_Signal, MA_5, BB_Upper_20, BB_Percent_B_20, BB_Width_20, BB_Lower_20, BB_Std_20, MACD_Histogram, RSI_14, MA_Deviation_20, EMA_26 | Daily_Return, Volatility_20D, Return_5D, Volume_Change, MACD_Signal, MA_5, BB_Upper_20, BB_Percent_B_20, BB_Width_20, BB_Lower_20 |
| 9 | Volatility_20D, Daily_Return, Volume_Change, BB_Upper_20, MA_5, Return_5D, MACD_Signal, BB_Percent_B_20, BB_Std_20, BB_Width_20, MA_50, Return_20D, MACD_Histogram, BB_Lower_20, EMA_12 | Volatility_20D, Daily_Return, Volume_Change, BB_Upper_20, MA_5, Return_5D, MACD_Signal, BB_Percent_B_20, BB_Std_20, BB_Width_20 |
| 10 | Daily_Return, Volatility_20D, Return_5D, MA_5, Volume_Change, BB_Upper_20, BB_Percent_B_20, Return_20D, BB_Std_20, MA_50, MACD_Signal, EMA_12, MA_20, MA_Deviation_20, MACD_Histogram | Daily_Return, Volatility_20D, Return_5D, MA_5, Volume_Change, BB_Upper_20, BB_Percent_B_20, Return_20D, BB_Std_20, MA_50 |
| 11 | Volatility_20D, Return_5D, Daily_Return, Volume_Change, BB_Percent_B_20, BB_Std_20, Return_20D, BB_Upper_20, MACD_Signal, MA_5, MACD_Histogram, MACD, RSI_14, BB_Width_20, MA_20 | Volatility_20D, Return_5D, Daily_Return, Volume_Change, BB_Percent_B_20, BB_Std_20, Return_20D, BB_Upper_20, MACD_Signal, MA_5 |
| 12 | Daily_Return, Volatility_20D, Return_5D, Volume_Change, MACD_Signal, BB_Percent_B_20, BB_Upper_20, BB_Width_20, BB_Std_20, MACD_Histogram, Return_20D, MA_Deviation_20, MA_5, EMA_12, MACD | Daily_Return, Volatility_20D, Return_5D, Volume_Change, MACD_Signal, BB_Percent_B_20, BB_Upper_20, BB_Width_20, BB_Std_20, MACD_Histogram |
| 13 | Volatility_20D, Return_5D, Daily_Return, Volume_Change, BB_Percent_B_20, MACD_Signal, BB_Upper_20, BB_Std_20, Return_20D, MA_5, BB_Width_20, MA_Deviation_20, MA_20, MA_50, MACD_Histogram | Volatility_20D, Return_5D, Daily_Return, Volume_Change, BB_Percent_B_20, MACD_Signal, BB_Upper_20, BB_Std_20, Return_20D, MA_5 |
| 14 | Return_5D, Daily_Return, Volume_Change, Volatility_20D, MACD, Return_20D, BB_Width_20, BB_Upper_20, MACD_Signal, MA_5, BB_Percent_B_20, MACD_Histogram, RSI_14, MA_Deviation_20, BB_Std_20 | Return_5D, Daily_Return, Volume_Change, Volatility_20D, MACD, Return_20D, BB_Width_20, BB_Upper_20, MACD_Signal, MA_5 |
| 15 | Volatility_20D, Volume_Change, Return_5D, Daily_Return, Return_20D, BB_Upper_20, BB_Width_20, MACD, BB_Std_20, MACD_Signal, BB_Percent_B_20, MACD_Histogram, RSI_14, MA_5, MA_50 | Volatility_20D, Volume_Change, Return_5D, Daily_Return, Return_20D, BB_Upper_20, BB_Width_20, MACD, BB_Std_20, MACD_Signal |
| 16 | Return_5D, Daily_Return, Volume_Change, Volatility_20D, RSI_14, MACD_Signal, BB_Std_20, BB_Upper_20, BB_Width_20, MACD, MA_5, Return_20D, MACD_Histogram, BB_Percent_B_20, MA_Deviation_20 | Return_5D, Daily_Return, Volume_Change, Volatility_20D, RSI_14, MACD_Signal, BB_Std_20, BB_Upper_20, BB_Width_20, MACD |

## Fold Classification Metrics

各セルは`Accuracy / Precision / Recall / F1 / ROC-AUC`の順である。

| Fold | All 20 | Top 15 | Top 10 | Baseline 9 |
|---:|---|---|---|---|
| 1 | .5000 / .6000 / .2727 / .3750 / .5455 | .5000 / .6000 / .2727 / .3750 / .5404 | .5000 / .6000 / .2727 / .3750 / .5354 | .6000 / .6667 / .5455 / .6000 / .6313 |
| 2 | .3500 / .3846 / .5000 / .4348 / .4000 | .3500 / .3846 / .5000 / .4348 / .3400 | .3500 / .3846 / .5000 / .4348 / .3050 | .3500 / .3846 / .5000 / .4348 / .3000 |
| 3 | .6000 / .5455 / .6667 / .6000 / .6364 | .5500 / .5000 / .5556 / .5263 / .5657 | .5000 / .4615 / .6667 / .5455 / .6061 | .5500 / .5000 / .5556 / .5263 / .5152 |
| 4 | .4500 / .2857 / .8000 / .4211 / .5733 | .4500 / .3125 / 1.0000 / .4762 / .5067 | .4000 / .2667 / .8000 / .4000 / .5067 | .3500 / .2500 / .8000 / .3810 / .5267 |
| 5 | .5500 / .5833 / .6364 / .6087 / .4545 | .5000 / .5455 / .5455 / .5455 / .4343 | .7500 / .6875 / 1.0000 / .8148 / .7677 | .4000 / .4000 / .1818 / .2500 / .4949 |
| 6 | .6000 / .6923 / .6923 / .6923 / .5385 | .4000 / .5714 / .3077 / .4000 / .4725 | .5000 / .6667 / .4615 / .5455 / .5385 | .2000 / .0000 / .0000 / .0000 / .2418 |
| 7 | .4500 / .4615 / .6000 / .5217 / .5500 | .5500 / .5455 / .6000 / .5714 / .5200 | .5000 / .5000 / .5000 / .5000 / .5300 | .7000 / .6429 / .9000 / .7500 / .7100 |
| 8 | .3500 / .3846 / .5000 / .4348 / .2600 | .4000 / .4167 / .5000 / .4545 / .3100 | .5500 / .5556 / .5000 / .5263 / .4100 | .5500 / .5333 / .8000 / .6400 / .5200 |
| 9 | .5500 / .5625 / .8182 / .6667 / .5152 | .5500 / .5625 / .8182 / .6667 / .5051 | .6000 / .6000 / .8182 / .6923 / .5556 | .5500 / .5833 / .6364 / .6087 / .5758 |
| 10 | .5000 / .5455 / .5455 / .5455 / .4646 | .4000 / .4545 / .4545 / .4545 / .4343 | .3500 / .4000 / .3636 / .3810 / .4747 | .6000 / .6154 / .7273 / .6667 / .5354 |
| 11 | .5000 / .4286 / .3333 / .3750 / .5354 | .5000 / .4286 / .3333 / .3750 / .4040 | .5000 / .4000 / .2222 / .2857 / .5152 | .4000 / .3333 / .3333 / .3333 / .3737 |
| 12 | .7500 / .8462 / .7857 / .8148 / .8333 | .6500 / .8182 / .6429 / .7200 / .8452 | .7500 / .8462 / .7857 / .8148 / .8214 | .6000 / .8750 / .5000 / .6364 / .8571 |
| 13 | .6000 / .5556 / 1.0000 / .7143 / .5400 | .5500 / .5294 / .9000 / .6667 / .5000 | .6000 / .5556 / 1.0000 / .7143 / .4700 | .5500 / .5263 / 1.0000 / .6897 / .5800 |
| 14 | .3500 / .3125 / .7143 / .4348 / .4505 | .3500 / .2857 / .5714 / .3810 / .5495 | .3500 / .2857 / .5714 / .3810 / .4835 | .4000 / .3077 / .5714 / .4000 / .4945 |
| 15 | .5000 / .5000 / .7000 / .5833 / .6200 | .5500 / .5333 / .8000 / .6400 / .6300 | .5000 / .5000 / .6000 / .5455 / .5400 | .5500 / .5333 / .8000 / .6400 / .5900 |
| 16 | .5000 / .4118 / 1.0000 / .5833 / .4066 | .4000 / .3077 / .5714 / .4000 / .3956 | .3500 / .2000 / .2857 / .2353 / .2637 | .3000 / .2667 / .5714 / .3636 / .4725 |

## Fold Backtest Metrics

各セルは`Total Return / Annual Return / Sharpe Ratio / Max Drawdown / Win Rate / Total Trades`の順で、Return・Drawdown・Win Rateは百分率である。

| Fold | All 20 | Top 15 | Top 10 | Baseline 9 |
|---:|---|---|---|---|
| 1 | -5.80% / -51.16% / -3.1954 / -6.74% / 60.00% / 2 | -5.80% / -51.16% / -3.1954 / -6.74% / 60.00% / 2 | -6.74% / -56.71% / -3.8402 / -6.74% / 50.00% / 1 | -5.60% / -49.89% / -3.0737 / -7.63% / 66.67% / 5 |
| 2 | 5.57% / 91.67% / 1.4604 / -10.01% / 38.46% / 2 | 5.57% / 91.67% / 1.4604 / -10.01% / 38.46% / 2 | 6.07% / 102.71% / 1.5656 / -10.01% / 41.67% / 2 | 5.57% / 91.67% / 1.4604 / -10.01% / 38.46% / 2 |
| 3 | 5.47% / 89.45% / 3.1497 / -2.66% / 55.56% / 4 | 1.90% / 25.31% / 1.3834 / -3.19% / 50.00% / 3 | 3.69% / 54.44% / 2.0808 / -2.66% / 50.00% / 3 | -2.48% / -26.01% / -1.3500 / -3.93% / 44.44% / 5 |
| 4 | -4.09% / -39.38% / -2.8883 / -5.92% / 28.57% / 5 | -4.09% / -39.38% / -2.8883 / -5.92% / 28.57% / 5 | -4.14% / -39.79% / -2.9297 / -5.97% / 26.67% / 4 | -5.76% / -50.95% / -3.9585 / -7.56% / 25.00% / 3 |
| 5 | 9.58% / 199.79% / 2.3398 / -3.62% / 54.55% / 5 | -3.71% / -36.47% / -3.2389 / -5.22% / 50.00% / 5 | 16.34% / 514.76% / 3.8456 / -1.25% / 72.73% / 4 | -3.66% / -36.04% / -3.7165 / -4.98% / 25.00% / 4 |
| 6 | 1.88% / 25.11% / 1.2873 / -2.43% / 55.56% / 4 | 2.66% / 37.10% / 2.1287 / -1.51% / 57.14% / 2 | 2.71% / 37.84% / 2.1659 / -1.51% / 66.67% / 3 | -1.82% / -19.75% / -3.4641 / -1.82% / 0.00% / 1 |
| 7 | -1.42% / -15.76% / -0.8813 / -3.35% / 50.00% / 7 | -2.40% / -25.24% / -1.6069 / -3.35% / 44.44% / 6 | -0.99% / -11.24% / -1.1546 / -2.07% / 50.00% / 4 | 3.71% / 54.91% / 3.5922 / -2.18% / 63.64% / 5 |
| 8 | 4.45% / 68.69% / 2.6934 / -2.26% / 45.45% / 5 | 1.55% / 20.32% / 0.9901 / -2.26% / 44.44% / 5 | 2.28% / 31.13% / 1.4173 / -2.04% / 50.00% / 5 | 6.93% / 123.41% / 3.8545 / -2.26% / 50.00% / 5 |
| 9 | 2.65% / 36.79% / 1.5543 / -5.82% / 60.00% / 4 | 6.53% / 113.59% / 4.6284 / -2.25% / 64.29% / 4 | 4.18% / 63.51% / 2.4647 / -4.40% / 64.29% / 3 | -0.23% / -2.68% / -0.0118 / -3.78% / 60.00% / 3 |
| 10 | 4.74% / 74.32% / 2.3823 / -4.24% / 54.55% / 3 | 3.89% / 58.04% / 1.9821 / -4.24% / 45.45% / 1 | 2.94% / 41.54% / 1.5391 / -4.24% / 40.00% / 2 | 6.64% / 116.37% / 3.3885 / -2.50% / 66.67% / 3 |
| 11 | -2.80% / -28.89% / -2.7917 / -3.74% / 40.00% / 3 | -5.43% / -48.81% / -4.9413 / -5.43% / 0.00% / 2 | -4.58% / -43.00% / -3.9616 / -4.58% / 33.33% / 3 | -2.85% / -29.34% / -2.6140 / -4.24% / 28.57% / 3 |
| 12 | 12.60% / 315.59% / 9.3839 / -0.64% / 77.78% / 4 | 13.12% / 339.01% / 9.8275 / -0.64% / 80.00% / 5 | 14.57% / 411.52% / 10.8259 / -0.64% / 81.82% / 5 | 11.89% / 285.13% / 9.2589 / -0.24% / 87.50% / 5 |
| 13 | -9.12% / -68.27% / -3.3535 / -13.31% / 52.94% / 2 | -3.18% / -32.16% / -1.4195 / -7.64% / 56.25% / 1 | -9.12% / -68.27% / -3.3535 / -13.31% / 52.94% / 2 | -14.03% / -83.70% / -4.7323 / -14.03% / 50.00% / 2 |
| 14 | -5.09% / -46.60% / -1.9189 / -9.58% / 28.57% / 2 | -3.11% / -31.54% / -1.5896 / -5.65% / 27.27% / 3 | -6.25% / -53.92% / -3.0046 / -8.72% / 23.08% / 4 | -4.10% / -39.51% / -1.9670 / -8.42% / 33.33% / 3 |
| 15 | 1.40% / 18.18% / 0.8525 / -7.94% / 50.00% / 4 | 0.58% / 7.15% / 0.4384 / -6.56% / 50.00% / 5 | 2.01% / 26.98% / 1.4028 / -5.23% / 50.00% / 4 | 1.10% / 14.01% / 0.6896 / -8.64% / 50.00% / 3 |
| 16 | -6.26% / -53.95% / -4.2877 / -8.54% / 33.33% / 4 | -7.14% / -58.87% / -5.1928 / -9.40% / 18.18% / 5 | -4.56% / -42.86% / -3.8980 / -6.88% / 14.29% / 5 | -6.14% / -53.23% / -4.4077 / -8.42% / 30.77% / 4 |

## Aggregate Results

値は`Mean ± sample Std`である。Return、Drawdown、Win Rateは小数表記で、例えば0.0086は0.86%を表す。

| Feature Set | Folds | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| All 20 | 16 | .5063 ± .1063 | .5063 ± .1429 | .6603 ± .2037 | .5504 ± .1311 | .5202 ± .1249 |
| Top 15 | 16 | .4781 ± .0875 | .4873 ± .1340 | .5858 ± .2078 | .5055 ± .1170 | .4971 ± .1254 |
| Top 10 | 16 | .5031 ± .1284 | .4944 ± .1700 | .5842 ± .2445 | .5120 ± .1751 | .5202 ± .1395 |
| Baseline 9 | 16 | .4781 ± .1366 | .4637 ± .2075 | .5889 ± .2605 | .4950 ± .1989 | .5262 ± .1470 |

| Feature Set | Total Return | Annual Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|---|---:|---:|---:|---:|---:|---:|
| All 20 | .0086 ± .0613 | .3848 ± 1.0321 | .3617 ± 3.4800 | -.0568 ± .0345 | .4908 ± .1296 | 3.7500 ± 1.3904 |
| Top 15 | .0006 ± .0544 | .2304 ± .9920 | -.0771 ± 3.8222 | -.0500 ± .0272 | .4466 ± .1922 | 3.5000 ± 1.6733 |
| Top 10 | .0115 ± .0721 | .6054 ± 1.6619 | .3228 ± 3.9161 | -.0502 ± .0351 | .4797 ± .1816 | 3.3750 ± 1.2042 |
| Baseline 9 | -.0068 ± .0642 | .1840 ± .9522 | -.4407 ± 3.9136 | -.0567 ± .0372 | .4500 ± .2140 | 3.5000 ± 1.2649 |

| Feature Set | Positive Return Folds | Positive Return Rate |
|---|---:|---:|
| All 20 | 9 / 16 | 0.5625 |
| Top 15 | 8 / 16 | 0.5000 |
| Top 10 | 9 / 16 | 0.5625 |
| Baseline 9 | 6 / 16 | 0.3750 |

## Objective Interpretation

- 平均AccuracyはAll 20が0.5063、Top 10が0.5031、Top 15とBaseline 9が0.4781だった。
- 平均ROC-AUCはBaseline 9が0.5262、All 20とTop 10が約0.5202、Top 15が0.4971だった。
- 平均Total ReturnはTop 10が1.15%、All 20が0.86%、Top 15が0.06%、Baseline 9が-0.68%だった。
- プラスリターンFoldはAll 20とTop 10が9件、Top 15が8件、Baseline 9が6件だった。
- 標準偏差が最小の指標を数えると、Top 15はAccuracy、Precision、F1、Total Return、Max Drawdownで最小だった。この基準ではTop 15が比較的安定していた。
- 一方、Recall、ROC-AUC、Sharpe Ratio、Win RateはAll 20、Annual ReturnはBaseline 9、Total TradesはTop 10の標準偏差が最小であり、すべての指標で共通して最も安定したセットはなかった。
- FoldごとのTotal ReturnとSharpe Ratioは正負が混在し、平均値だけでは期間依存性を表現できない。
- 特徴量セットによって分類、収益、リスク、安定性の評価が異なり、特定セットが総合的または将来的に優れているとは断定できない。

## Limitations

- 単一銘柄7203.T、取得時点の2年間だけの結果である。
- 各テストFoldは20件と短く、年率リターンやSharpe Ratioの変動が大きい。
- Fold同士のテスト期間は非重複だが、expanding windowのため後のFoldは過去Foldのtestを学習へ取り込む。
- Gain順位、選択特徴量、評価結果はデータ期間、モデル設定、乱数シードに依存する。
- 取引手数料、税金、スリッページを考慮していない。
- ウォークフォワード結果は将来の分類性能や利益を保証しない。

## Next Improvements

- 複数銘柄と異なる開始期間での再検証
- initial train、test、stepの感度分析
- 取引コストを含むバックテスト
- 特徴量選択頻度とGain順位の安定性集計
- 閾値0.55を固定した比較と閾値検証の分離
- Fold間の予測を連結した全期間バックテスト
