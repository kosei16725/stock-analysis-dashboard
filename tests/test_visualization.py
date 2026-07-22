"""Plotlyによる可視化処理の通信非依存テスト。"""

import pandas as pd

from src.visualization import create_feature_importance_chart


def test_legacy_importance_is_labeled_as_split() -> None:
    """旧Importance列を誤ってGainと表示せず、Splitとして表示する。"""
    legacy = pd.DataFrame(
        {
            "Feature": ["Feature_B", "Feature_A"],
            "Importance": [2, 5],
        }
    )

    figure = create_feature_importance_chart(legacy)

    assert figure.layout.title.text == "Feature Importance (Split)"
    assert figure.layout.xaxis.title.text == "Split Importance"
    assert figure.data[0].name == "Split Importance"
    assert list(figure.data[0].x) == [5, 2]
    assert list(figure.data[0].y) == ["Feature_A", "Feature_B"]


def test_new_importance_can_display_gain() -> None:
    """新形式でGain列とGainラベルが対応することを確認する。"""
    importance = pd.DataFrame(
        {
            "Feature": ["Feature_A", "Feature_B"],
            "Gain_Importance": [1.5, 3.0],
            "Split_Importance": [8, 4],
        }
    )

    figure = create_feature_importance_chart(importance, "Gain")

    assert figure.layout.title.text == "Feature Importance (Gain)"
    assert figure.data[0].name == "Gain Importance"
    assert list(figure.data[0].x) == [3.0, 1.5]
    assert list(figure.data[0].y) == ["Feature_B", "Feature_A"]


def test_new_importance_can_display_split() -> None:
    """新形式でSplit列とSplitラベルが対応することを確認する。"""
    importance = pd.DataFrame(
        {
            "Feature": ["Feature_A", "Feature_B"],
            "Gain_Importance": [1.5, 3.0],
            "Split_Importance": [8, 4],
        }
    )

    figure = create_feature_importance_chart(importance, "Split")

    assert figure.layout.title.text == "Feature Importance (Split)"
    assert figure.data[0].name == "Split Importance"
    assert list(figure.data[0].x) == [8, 4]
    assert list(figure.data[0].y) == ["Feature_A", "Feature_B"]
