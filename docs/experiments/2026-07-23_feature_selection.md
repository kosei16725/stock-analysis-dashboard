# Phase 10 Feature Selection Experiment

## Experiment Purpose

現在の20特徴量について、全特徴量、学習期間のGain Importance上位15・上位10、Phase 8以前のBaseline 9を同一条件で再学習・評価し、特徴量選択の影響を実測する。

## Conditions

- Ticker: 7203.T
- Period: 1y
- Buy Threshold: 0.55
- Train Samples before Boundary Purge: 154
- Excluded Boundary Samples: 1
- Train Samples used for Ranking and Models: 153
- Test Samples: 39
- Training Period: 2025-10-02 ～ 2026-05-22
- Test Period: 2026-05-26 ～ 2026-07-17
- Model、ハイパーパラメータ、時系列分割、バックテスト条件: 全セットで同一
- Gain順位: 境界1件を除外した学習期間153件だけで学習したランキング用モデルから取得
- 実行日: 2026-07-23

Targetは翌営業日の終値から作られるため、分割直後の学習末尾ラベルはテスト初日の価格を参照する。この境界サンプル1件を除外し、テスト特徴量、テストTarget、境界Target、将来リターンを特徴量順位やモデル学習に使用していない。順位決定後、各特徴量セットを同じパージ済み学習期間と同じテスト期間へ適用して個別に再学習した。

## Feature Sets

### All 20 Features

Daily_Return、Return_5D、Return_20D、MA_5、MA_20、MA_50、MA_Deviation_20、Volatility_20D、Volume_Change、RSI_14、EMA_12、EMA_26、MACD、MACD_Signal、MACD_Histogram、BB_Std_20、BB_Upper_20、BB_Lower_20、BB_Width_20、BB_Percent_B_20

### Top 15 by Gain Importance

Return_5D、Volume_Change、RSI_14、Daily_Return、BB_Width_20、MACD、MA_Deviation_20、MACD_Signal、BB_Upper_20、MA_20、EMA_26、MACD_Histogram、Return_20D、BB_Lower_20、MA_50

### Top 10 by Gain Importance

Return_5D、Volume_Change、RSI_14、Daily_Return、BB_Width_20、MACD、MA_Deviation_20、MACD_Signal、BB_Upper_20、MA_20

### Baseline 9 Features

Daily_Return、Return_5D、Return_20D、MA_5、MA_20、MA_50、MA_Deviation_20、Volatility_20D、Volume_Change

## Classification Metrics

| Feature Set | Count | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| All 20 Features | 20 | 0.4359 | 0.4444 | 0.8889 | 0.5926 | 0.3360 |
| Top 15 by Gain Importance | 15 | 0.4615 | 0.4571 | 0.8889 | 0.6038 | 0.3664 |
| Top 10 by Gain Importance | 10 | 0.4359 | 0.4286 | 0.6667 | 0.5217 | 0.3783 |
| Baseline 9 Features | 9 | 0.4615 | 0.4483 | 0.7222 | 0.5532 | 0.4048 |

## Backtest Metrics

| Feature Set | Total Return | Annual Return | Sharpe Ratio | Max Drawdown | Win Rate | Total Trades |
|---|---:|---:|---:|---:|---:|---:|
| All 20 Features | 4.35% | 30.78% | 1.2789 | -7.55% | 45.16% | 4 |
| Top 15 by Gain Importance | 3.06% | 20.89% | 0.9372 | -7.98% | 41.94% | 5 |
| Top 10 by Gain Importance | 3.00% | 20.49% | 0.9590 | -7.46% | 42.31% | 7 |
| Baseline 9 Features | -2.44% | -14.39% | -0.5517 | -9.68% | 44.44% | 6 |

## Interpretation

- Top 15はPrecisionとF1が最も高く、AccuracyはBaseline 9と同率で最も高かった。
- All 20とTop 15はRecallが同率で最も高かった。
- Baseline 9はROC-AUCが最も高かった一方、Total Return、Annual Return、Sharpe Ratioは4セット中で最も低かった。
- All 20はTotal Return、Annual Return、Sharpe Ratio、Win Rateが最も高く、Total Tradesが最も少なかった。
- Top 10はMax Drawdownの下落幅が最も小さかった一方、Total Tradesが最も多かった。
- 分類、収益、リスクの各指標で結果が混在しており、特定セットが総合的または将来的に優れているとは断定できない。
- Gain Importanceは学習済みモデル内での利用状況であり、因果関係や特徴量単独の有効性を示さない。

## Limitations

- テストデータが39件と短い。
- 単一銘柄7203.T、単一の1年期間だけの結果である。
- Gain順位と評価結果は、この学習期間・乱数シード・モデル設定に依存する。
- 境界サンプル1件の除外によってGain順位と評価結果が変化しており、時系列境界の定義に依存する。
- 取引手数料、税金、スリッページを考慮していない。
- この結果だけで特徴量の恒久的な削除や将来性能の改善を判断できない。

## Next Improvements

- 複数銘柄・複数期間での再現性確認
- ウォークフォワード方式で各学習窓内のGain順位を再計算
- 特徴量セットごとの重要度安定性の確認
- Buy Thresholdを固定した比較と閾値感度の分離
- ROC-AUC、収益、Sharpe Ratio、最大ドローダウンを含む多面的評価
- 相関の強い特徴量を考慮した削除候補の検討
