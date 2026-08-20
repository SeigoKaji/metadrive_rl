"""公式再現と汎化実験を型付きprofileとして選択する。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from .generalization_config import (
    GENERALIZATION_DEFAULT_MODEL_NAME,
    GENERALIZATION_EVALUATION_ENV_CONFIG,
    GENERALIZATION_EVALUATION_EPISODES,
    GENERALIZATION_TRAIN_ENV_CONFIG,
    GENERALIZATION_TRAINING_CONFIG,
)
from .phase0_config import OFFICIAL_ENV_CONFIG, OFFICIAL_TRAINING_CONFIG


ProfileName: TypeAlias = Literal["official", "generalization"]


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    """1回の学習と評価に必要な設定一式。"""

    train_env_config: Mapping[str, object]
    evaluation_env_config: Mapping[str, object]
    training_config: Mapping[str, object]
    default_model_name: str
    evaluation_episodes: int


OFFICIAL_PROFILE: Final[ExperimentProfile] = ExperimentProfile(
    train_env_config=OFFICIAL_ENV_CONFIG,
    evaluation_env_config=OFFICIAL_ENV_CONFIG,
    training_config=OFFICIAL_TRAINING_CONFIG,
    default_model_name="phase0_official",
    evaluation_episodes=1,
)

GENERALIZATION_PROFILE: Final[ExperimentProfile] = ExperimentProfile(
    train_env_config=GENERALIZATION_TRAIN_ENV_CONFIG,
    evaluation_env_config=GENERALIZATION_EVALUATION_ENV_CONFIG,
    training_config=GENERALIZATION_TRAINING_CONFIG,
    default_model_name=GENERALIZATION_DEFAULT_MODEL_NAME,
    evaluation_episodes=GENERALIZATION_EVALUATION_EPISODES,
)

PROFILE_NAMES: Final[tuple[ProfileName, ...]] = ("official", "generalization")

_PROFILES: Final[dict[str, ExperimentProfile]] = {
    "official": OFFICIAL_PROFILE,
    "generalization": GENERALIZATION_PROFILE,
}


def get_experiment_profile(name: str) -> ExperimentProfile:
    """名前に対応するprofileを返し、未知の名前は明示的に拒否する。"""

    try:
        return _PROFILES[name]
    except KeyError:
        choices = ", ".join(PROFILE_NAMES)
        raise ValueError(
            f"unknown experiment profile {name!r}; choose from: {choices}"
        ) from None
