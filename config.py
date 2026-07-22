"""アプリケーションの変更可能な設定値を管理する。"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AppConfig:
    """第1段階で使用する銘柄と表示設定。"""

    ticker: str = "7203.T"
    company_name: str = "トヨタ自動車"
    period: str = "1y"
    period_label: str = "過去1年"
    moving_average_windows: Tuple[int, int] = (20, 50)


APP_CONFIG = AppConfig()
