"""OptunaとNested Walk ForwardによるLightGBM最適化を提供する。"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import zlib

from lightgbm import LGBMClassifier
import numpy as np
import optuna
from optuna.importance import get_param_importances
import pandas as pd

from src.backtest import DEFAULT_BUY_THRESHOLD, run_backtest
from src.constants import FEATURE_COLUMNS
from src.feature_selection import bound_backtest_prices
from src.model import (
    DEFAULT_RANDOM_STATE,
    calculate_metrics,
    predict_labels,
    predict_up_probabilities,
    separate_features_target,
)
from src.walk_forward import (
    WalkForwardSplit,
    create_walk_forward_splits,
)

DEFAULT_N_TRIALS = 20
DEFAULT_INNER_INITIAL_TRAIN_SIZE = 60
DEFAULT_INNER_TEST_SIZE = 20
DEFAULT_INNER_STEP_SIZE = 20
DEFAULT_RESULTS_DIRECTORY = Path("data/results")
LOW_TRADE_THRESHOLD = 5
LOW_TRADE_PENALTY = 1.0
INVALID_SHARPE_PENALTY = 10.0
FOLD_FAILURE_PENALTY = 10.0

SEARCH_SPACE_NAMES = (
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
)

COMPARISON_METRICS = (
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "Total_Return",
    "Annual_Return",
    "Sharpe_Ratio",
    "Max_Drawdown",
    "Win_Rate",
    "Total_Trades",
)


@dataclass(frozen=True)
class OptimizationResult:
    """Nested最適化のTrial、Outer Fold評価、集約結果を保持する。"""

    trials: pd.DataFrame
    outer_fold_results: pd.DataFrame
    comparison_summary: pd.DataFrame
    best_parameters_by_fold: Mapping[int, Mapping[str, object]]
    best_scores_by_fold: Mapping[int, float]
    recommended_parameters: Mapping[str, object]
    recommended_score: float
    parameter_importance: Mapping[str, float]


def _validate_positive_integer(name: str, value: object, minimum: int = 1) -> int:
    """boolを除く正の整数引数を検証する。"""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name}は整数で指定してください。")
    validated = int(value)
    if validated < minimum:
        raise ValueError(f"{name}は{minimum}以上で指定してください。")
    return validated


def suggest_lightgbm_parameters(trial: optuna.Trial) -> Dict[str, object]:
    """指定された探索空間からLightGBMパラメータを生成する。"""
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.005, 0.2, log=True
        ),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
    }


def build_classifier(
    parameters: Optional[Mapping[str, object]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> LGBMClassifier:
    """固定条件と探索パラメータからLightGBM分類器を作る。

    既存モデルと同じ100 trees、seed、単一スレッドを維持する。
    """
    supplied = dict(parameters or {})
    unknown = sorted(set(supplied).difference(SEARCH_SPACE_NAMES))
    if unknown:
        raise ValueError(f"未定義のLightGBMパラメータがあります: {', '.join(unknown)}")
    return LGBMClassifier(
        boosting_type="gbdt",
        objective="binary",
        n_estimators=100,
        learning_rate=float(supplied.get("learning_rate", 0.05)),
        num_leaves=int(supplied.get("num_leaves", 31)),
        max_depth=int(supplied.get("max_depth", -1)),
        min_child_samples=int(supplied.get("min_child_samples", 20)),
        feature_fraction=float(supplied.get("feature_fraction", 1.0)),
        bagging_fraction=float(supplied.get("bagging_fraction", 1.0)),
        bagging_freq=int(supplied.get("bagging_freq", 0)),
        random_state=random_state,
        verbosity=-1,
        n_jobs=1,
    )


def evaluate_parameters_on_fold(
    split: WalkForwardSplit,
    close_prices: pd.Series,
    parameters: Optional[Mapping[str, object]] = None,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, object]:
    """1つのパージ済みFoldを指定パラメータで学習・評価する。"""
    if tuple(split.X_train.columns) != FEATURE_COLUMNS:
        raise ValueError("Nested評価の学習列をFEATURE_COLUMNSと一致させてください。")
    if tuple(split.X_test.columns) != FEATURE_COLUMNS:
        raise ValueError("Nested評価のテスト列をFEATURE_COLUMNSと一致させてください。")
    if split.X_train.index.max() >= split.X_test.index.min():
        raise ValueError("Nested評価では学習期間をテスト期間より前にしてください。")
    if split.y_train.nunique() < 2:
        raise ValueError("学習データにはTargetの0と1の両方が必要です。")

    model = build_classifier(parameters, random_state)
    model.fit(split.X_train, split.y_train)
    predictions = predict_labels(model, split.X_test)
    probabilities = predict_up_probabilities(model, split.X_test)
    classification = calculate_metrics(split.y_test, predictions, probabilities)
    bounded_prices = bound_backtest_prices(close_prices, split.X_test.index)
    backtest = run_backtest(
        pd.Series(probabilities, index=split.X_test.index, name="Probability"),
        bounded_prices,
        buy_threshold,
    )
    return {
        "Accuracy": classification["Accuracy"],
        "Precision": classification["Precision"],
        "Recall": classification["Recall"],
        "F1": classification["F1"],
        "ROC_AUC": classification["ROC-AUC"],
        "Total_Return": backtest.metrics["Total Return"],
        "Annual_Return": backtest.metrics["Annual Return"],
        "Sharpe_Ratio": backtest.metrics["Sharpe Ratio"],
        "Max_Drawdown": backtest.metrics["Max Drawdown"],
        "Win_Rate": backtest.metrics["Win Rate"],
        "Total_Trades": backtest.metrics["Total Trades"],
        "Backtest_End": bounded_prices.index.max(),
    }


def calculate_penalized_sharpe(
    sharpe_ratio: object,
    total_trades: object,
) -> Tuple[float, float]:
    """Sharpeと取引数から目的関数への寄与値とPenaltyを返す。"""
    try:
        sharpe = float(sharpe_ratio)
    except (TypeError, ValueError):
        return -INVALID_SHARPE_PENALTY, INVALID_SHARPE_PENALTY
    if not math.isfinite(sharpe):
        return -INVALID_SHARPE_PENALTY, INVALID_SHARPE_PENALTY

    penalty = 0.0
    try:
        trades = float(total_trades)
    except (TypeError, ValueError):
        trades = 0.0
    if trades < LOW_TRADE_THRESHOLD:
        penalty += LOW_TRADE_PENALTY
    return sharpe - penalty, penalty


def create_optuna_objective(
    inner_splits: Sequence[WalkForwardSplit],
    close_prices: pd.Series,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """Inner Foldだけを評価するOptuna目的関数を作る。

    Outer Testは引数として受け取らないため、Trial選択へ利用できない。
    """
    splits = tuple(inner_splits)
    if not splits:
        raise ValueError("Optuna目的関数には1つ以上のInner Foldが必要です。")

    def objective(trial: optuna.Trial) -> float:
        parameters = suggest_lightgbm_parameters(trial)
        contributions: List[float] = []
        total_penalty = 0.0
        failed_folds = 0
        for split in splits:
            try:
                metrics = evaluate_parameters_on_fold(
                    split,
                    close_prices,
                    parameters,
                    buy_threshold,
                    random_state,
                )
                contribution, penalty = calculate_penalized_sharpe(
                    metrics["Sharpe_Ratio"],
                    metrics["Total_Trades"],
                )
            except Exception:
                contribution = -FOLD_FAILURE_PENALTY
                penalty = FOLD_FAILURE_PENALTY
                failed_folds += 1
            contributions.append(contribution)
            total_penalty += penalty
        score = float(np.mean(contributions))
        trial.set_user_attr("Inner_Fold_Count", len(splits))
        trial.set_user_attr("Failed_Fold_Count", failed_folds)
        trial.set_user_attr("Total_Penalty", total_penalty)
        return score

    return objective


def create_study(random_state: int = DEFAULT_RANDOM_STATE) -> optuna.Study:
    """再現可能な最大化Studyを作る。"""
    sampler = optuna.samplers.TPESampler(seed=random_state)
    return optuna.create_study(direction="maximize", sampler=sampler)


def optimize_inner_folds(
    inner_splits: Sequence[WalkForwardSplit],
    close_prices: pd.Series,
    n_trials: int = DEFAULT_N_TRIALS,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> optuna.Study:
    """Inner Foldの平均Penalty付きSharpeを最大化する。"""
    trial_count = _validate_positive_integer("n_trials", n_trials)
    study = create_study(random_state)
    study.optimize(
        create_optuna_objective(
            inner_splits,
            close_prices,
            buy_threshold,
            random_state,
        ),
        n_trials=trial_count,
        show_progress_bar=False,
    )
    return study


def _study_trials_frame(study: optuna.Study, outer_fold: int) -> pd.DataFrame:
    """Studyの全Trialを保存用DataFrameへ変換する。"""
    rows = []
    for trial in study.trials:
        row: Dict[str, object] = {
            "Outer_Fold": outer_fold,
            "Trial": trial.number,
            "Value": trial.value,
            "State": trial.state.name,
            "Inner_Fold_Count": trial.user_attrs.get("Inner_Fold_Count"),
            "Failed_Fold_Count": trial.user_attrs.get("Failed_Fold_Count"),
            "Total_Penalty": trial.user_attrs.get("Total_Penalty"),
        }
        row.update(trial.params)
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate_comparison(outer_results: pd.DataFrame) -> pd.DataFrame:
    """DefaultとTunedのOuter Test指標を平均・標準偏差へ集約する。"""
    rows = []
    for variant in ("Default", "Tuned"):
        group = outer_results.loc[outer_results["Model"] == variant]
        row: Dict[str, object] = {
            "Model": variant,
            "Outer_Fold_Count": int(group["Outer_Fold"].nunique()),
        }
        for metric in COMPARISON_METRICS:
            numeric = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_Mean"] = float(numeric.mean())
            row[f"{metric}_Std"] = (
                0.0 if numeric.count() <= 1 else float(numeric.std(ddof=1))
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_parameter_importance(study: optuna.Study) -> Mapping[str, float]:
    """Trialが1件以下の場合も安全なParameter Importanceを返す。"""
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed) <= 1:
        return {name: 0.0 for name in SEARCH_SPACE_NAMES}
    return get_param_importances(study)


def run_nested_walk_forward_optimization(
    training_data: pd.DataFrame,
    close_prices: pd.Series,
    n_trials: int = DEFAULT_N_TRIALS,
    outer_initial_train_size: int = 100,
    outer_test_size: int = 20,
    outer_step_size: int = 20,
    inner_initial_train_size: int = DEFAULT_INNER_INITIAL_TRAIN_SIZE,
    inner_test_size: int = DEFAULT_INNER_TEST_SIZE,
    inner_step_size: int = DEFAULT_INNER_STEP_SIZE,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> OptimizationResult:
    """Nested Walk Forwardで最適化し、未使用Outer Testで比較する。

    各Outer Foldのパージ済み学習期間だけからInner Foldを作り、独立した
    Optuna Studyを実行する。選ばれたパラメータを初めてOuter Testへ適用し、
    既定モデルと比較する。
    """
    _validate_positive_integer("n_trials", n_trials)
    X, y = separate_features_target(training_data)
    outer_splits = create_walk_forward_splits(
        X,
        y,
        outer_initial_train_size,
        outer_test_size,
        outer_step_size,
    )

    trial_frames = []
    outer_rows = []
    best_parameters_by_fold: Dict[int, Mapping[str, object]] = {}
    best_scores_by_fold: Dict[int, float] = {}
    studies: Dict[int, optuna.Study] = {}
    for outer in outer_splits:
        inner_splits = create_walk_forward_splits(
            outer.X_train,
            outer.y_train,
            inner_initial_train_size,
            inner_test_size,
            inner_step_size,
        )
        study = optimize_inner_folds(
            inner_splits,
            close_prices,
            n_trials,
            buy_threshold,
            random_state + outer.fold,
        )
        studies[outer.fold] = study
        best_parameters_by_fold[outer.fold] = dict(study.best_params)
        best_scores_by_fold[outer.fold] = float(study.best_value)
        trial_frames.append(_study_trials_frame(study, outer.fold))

        for model_name, parameters in (
            ("Default", None),
            ("Tuned", study.best_params),
        ):
            metrics = evaluate_parameters_on_fold(
                outer,
                close_prices,
                parameters,
                buy_threshold,
                random_state,
            )
            outer_rows.append(
                {
                    "Outer_Fold": outer.fold,
                    "Model": model_name,
                    "Train_Start": outer.X_train.index.min(),
                    "Train_End": outer.X_train.index.max(),
                    "Test_Start": outer.X_test.index.min(),
                    "Test_End": outer.X_test.index.max(),
                    "Train_Size": len(outer.X_train),
                    "Test_Size": len(outer.X_test),
                    "Best_Inner_Score": float(study.best_value),
                    **metrics,
                }
            )

    # 将来利用候補は、最も新しく最大学習期間を持つOuter FoldのInner Study
    # から選ぶ。Outer Testの指標による選択は行わない。
    latest_fold = max(studies)
    recommended_study = studies[latest_fold]
    outer_results = pd.DataFrame(outer_rows)
    return OptimizationResult(
        trials=pd.concat(trial_frames, ignore_index=True),
        outer_fold_results=outer_results,
        comparison_summary=_aggregate_comparison(outer_results),
        best_parameters_by_fold=best_parameters_by_fold,
        best_scores_by_fold=best_scores_by_fold,
        recommended_parameters=dict(recommended_study.best_params),
        recommended_score=float(recommended_study.best_value),
        parameter_importance=_safe_parameter_importance(recommended_study),
    )


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """PNGチャンクを生成する。"""
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def _save_simple_chart_png(
    values: Sequence[float],
    output_path: Path,
    chart_name: str,
    bars: bool = False,
) -> None:
    """追加描画依存なしで簡易折れ線・棒グラフPNGを保存する。"""
    width, height = 900, 480
    margin = 45
    pixels = bytearray([255] * width * height * 3)

    def set_pixel(x: int, y: int, color: Tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset: offset + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = round(x0 + (x1 - x0) * step / steps)
            y = round(y0 + (y1 - y0) * step / steps)
            set_pixel(x, y, color)

    line(margin, margin, margin, height - margin, (80, 80, 80))
    line(margin, height - margin, width - margin, height - margin, (80, 80, 80))
    numeric = [float(value) for value in values if math.isfinite(float(value))]
    if numeric:
        minimum, maximum = min(numeric), max(numeric)
        span = maximum - minimum or 1.0
        usable_width = width - 2 * margin
        usable_height = height - 2 * margin
        points = []
        for index, value in enumerate(numeric):
            x = margin + round(
                usable_width * index / max(len(numeric) - 1, 1)
            )
            y = height - margin - round(usable_height * (value - minimum) / span)
            points.append((x, y))
        if bars:
            bar_width = max(2, usable_width // max(len(points) * 2, 1))
            for x, y in points:
                for px in range(x - bar_width, x + bar_width + 1):
                    line(px, height - margin - 1, px, y, (55, 126, 184))
        else:
            for start, end in zip(points, points[1:]):
                line(*start, *end, (55, 126, 184))
            for x, y in points:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        set_pixel(x + dx, y + dy, (214, 39, 40))

    scanlines = b"".join(
        b"\x00" + bytes(pixels[row * width * 3: (row + 1) * width * 3])
        for row in range(height)
    )
    metadata = f"Title\x00{chart_name}".encode("latin-1", errors="replace")
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"tEXt", metadata)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )
    output_path.write_bytes(png)


def save_optimization_results(
    result: OptimizationResult,
    output_directory: Path = DEFAULT_RESULTS_DIRECTORY,
) -> Mapping[str, Path]:
    """Trial、推奨パラメータ、3種類のPNGを保存する。"""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    trials_path = output / "optuna_trials.csv"
    parameters_path = output / "best_parameters.json"
    result.trials.to_csv(trials_path, index=False)
    payload = {
        "recommended_parameters": dict(result.recommended_parameters),
        "recommended_score": result.recommended_score,
        "selection_rule": "latest outer fold inner-study only",
        "best_parameters_by_outer_fold": {
            str(fold): dict(parameters)
            for fold, parameters in result.best_parameters_by_fold.items()
        },
        "best_scores_by_outer_fold": {
            str(fold): score
            for fold, score in result.best_scores_by_fold.items()
        },
    }
    parameters_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    values = pd.to_numeric(result.trials["Value"], errors="coerce").fillna(
        -FOLD_FAILURE_PENALTY
    )
    best_so_far = values.groupby(result.trials["Outer_Fold"]).cummax()
    trial_path = output / "optuna_trial_values.png"
    objective_path = output / "optuna_objective_history.png"
    importance_path = output / "optuna_parameter_importance.png"
    _save_simple_chart_png(values.tolist(), trial_path, "Optuna Trial Values")
    _save_simple_chart_png(
        best_so_far.tolist(),
        objective_path,
        "Optuna Best Objective History",
    )
    _save_simple_chart_png(
        list(result.parameter_importance.values()),
        importance_path,
        "Optuna Parameter Importance",
        bars=True,
    )
    return {
        "trials_csv": trials_path,
        "best_parameters_json": parameters_path,
        "trial_values_png": trial_path,
        "objective_history_png": objective_path,
        "parameter_importance_png": importance_path,
    }
