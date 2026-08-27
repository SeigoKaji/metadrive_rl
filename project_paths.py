"""プロジェクト内で共有する成果物・設定ディレクトリを定義する。"""

from pathlib import Path
from typing import Final


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
CONFIG_DIR: Final[Path] = PROJECT_ROOT / "configs"
MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"
MONITOR_LOG_DIR: Final[Path] = LOG_DIR / "monitor"
TENSORBOARD_LOG_DIR: Final[Path] = LOG_DIR / "tensorboard"
OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
