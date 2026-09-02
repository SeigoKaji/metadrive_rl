"""Reference-observation strategy and dimension checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from input_attribution.baselines import BaselineError, BaselineProvider
from input_attribution.config import BaselineReference


def _provider() -> BaselineProvider:
    return BaselineProvider(
        observations=np.asarray(
            [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
            dtype=np.float32,
        ),
        episode_ids=np.asarray([1, 1, 2, 2]),
        steps=np.asarray([1, 2, 1, 2]),
    )


def test_specified_and_sampled_baselines_are_reproducible() -> None:
    provider = _provider()
    specified = provider.resolve(
        "specified_steps",
        count=2,
        specified_steps=(BaselineReference(1, 2), BaselineReference(2, 1)),
    )
    first_sample = provider.resolve("sampled_observations", count=2, seed=17)
    second_sample = provider.resolve("sampled_observations", count=2, seed=17)

    np.testing.assert_array_equal(specified.values[0], [[2.0, 3.0], [4.0, 5.0]])
    np.testing.assert_array_equal(first_sample.values, second_sample.values)
    np.testing.assert_array_equal(first_sample.baseline_ids, second_sample.baseline_ids)


def test_external_npz_supports_multiple_rows_and_rejects_wrong_dimension(
    tmp_path: Path,
) -> None:
    provider = _provider()
    valid = tmp_path / "valid.npz"
    invalid = tmp_path / "invalid.npz"
    np.savez_compressed(valid, observations=np.asarray([[8.0, 9.0], [10.0, 11.0]]))
    np.savez_compressed(invalid, observations=np.zeros((1, 3), dtype=np.float32))

    resolved = provider.resolve(
        "external_npz",
        count=2,
        seed=3,
        external_npz_path=valid,
        external_npz_key="observations",
    )

    assert resolved.values.shape == (4, 2, 2)
    assert sorted(resolved.metadata["source_rows"]) == [0, 1]
    assert resolved.metadata["source_shape"] == [2, 2]
    assert len(str(resolved.metadata["sha256"])) == 64
    with pytest.raises(BaselineError, match="観測次元"):
        provider.resolve(
            "external_npz",
            external_npz_path=invalid,
            external_npz_key="observations",
        )
