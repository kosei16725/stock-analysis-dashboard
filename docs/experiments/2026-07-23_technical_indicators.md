# Phase 8 Technical Indicators Experiment

## Experiment Purpose

既存9特徴量にRSI、MACD、Bollinger Bands関連の11特徴量を追加し、保存済みBaselineと同一条件で比較した実験である。

## Conditions

- Ticker: 7203.T
- Period: 1y
- Buy Threshold: 0.55
- Train Samples: 154
- Test Samples: 39
- Training Period: 2025-10-02 ～ 2026-05-25
- Test Period: 2026-05-26 ～ 2026-07-17
- Model、ハイパーパラメータ、時系列分割、バックテスト条件: Baselineと同一
- 実行日: 2026-07-23

## Added Features

- RSI_14
- EMA_12
- EMA_26
- MACD
- MACD_Signal
- MACD_Histogram
- BB_Std_20
- BB_Upper_20
- BB_Lower_20
- BB_Width_20
- BB_Percent_B_20

既存9特徴量と上記11特徴量を合わせ、モデルへ入力する全特徴量は20列となった。

RSIは、最初の14期間のGainとLossの算術平均を初期値とし、それ以降をWilderの再帰式で平滑化する厳密なWilder方式で計算した。

## Classification Metrics

| Metric | Baseline | Phase 8 | Difference |
|---|---:|---:|---:|
| Accuracy | 0.4359 | 0.4872 | +0.0513 |
| Precision | 0.4231 | 0.4706 | +0.0475 |
| Recall | 0.6111 | 0.8889 | +0.2778 |
| F1 | 0.5000 | 0.6154 | +0.1154 |
| ROC-AUC | 0.3995 | 0.3757 | -0.0238 |

## Backtest Metrics

割合で表す指標のDifferenceはパーセントポイント（pp）で記載する。

| Metric | Baseline | Phase 8 | Difference |
|---|---:|---:|---:|
| Total Return | 1.41% | 3.45% | +2.04 pp |
| Annual Return | 9.19% | 23.80% | +14.61 pp |
| Sharpe Ratio | 0.5314 | 1.0433 | +0.5119 |
| Max Drawdown | -5.44% | -7.55% | -2.11 pp |
| Win Rate | 44.00% | 43.33% | -0.67 pp |
| Total Trades | 5 | 4 | -1 |

## Interpretation

- Accuracy、Precision、Recall、F1はBaselineより上昇した。
- ROC-AUCは低下した。
- Total Return、Annual Return、Sharpe Ratioは上昇した。
- 最大ドローダウンは拡大した。
- Win Rateと取引回数は低下した。
- 指標ごとに改善と悪化が混在しており、Phase 8が総合的または将来的に優れているとは断定できない。
- テスト期間が39件と短く、単一銘柄・単一期間の結果である。
- 特徴量追加による因果的な改善を証明する結果ではない。

## Next Steps

- 特徴量重要度の確認
- 不要な特徴量の除外
- 複数銘柄・複数期間での検証
- Buy Thresholdの検証
- Optunaによるハイパーパラメータ調整
- ROC-AUCと最大ドローダウンを含む多面的評価
