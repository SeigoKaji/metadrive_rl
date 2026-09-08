"""Pure sidecar/checkpoint compatibility tests.

The tests create tiny synthetic ZIP files and never import SB3 or MetaDrive.
Run with ``PYTHONDONTWRITEBYTECODE=1 python -B -m unittest -v
lookahead_learning.test_checkpoint``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from types import MappingProxyType

from .checkpoint import CheckpointContractError, validate_checkpoint_metadata


def _write_synthetic_zip(path: Path, *, marker: str) -> str:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        # Keep names and the semantic shape stable while changing one payload
        # byte between otherwise equivalent checkpoint files.
        archive.writestr("data", json.dumps({"observation_shape": [265]}))
        archive.writestr("policy", marker.encode("utf-8"))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(checkpoint_hash: str) -> dict[str, object]:
    return {
        "model_sha256": checkpoint_hash,
        "mode": "lookahead_obs_pp_reward",
        "pp_weight": 0.1,
        "steering_sign": -1.0,
        "observation_schema": {
            "shape": [265],
            "dtype": "float32",
            "low": [0.0, 0.0, 0.0],
            "high": [1.0, 1.0, 1.0],
        },
        "source_identities": {
            "host": {"module": "start_lane_env", "sha256": "host-source"},
            "extension": {"module": "lookahead_learning", "version": "0.1.0"},
        },
        "normalization_config": {
            "distance_m": 10.0,
            "clip": [-1.0, 1.0],
        },
        "wrapper_order": ["raw_host", "lookahead_learning", "Monitor", "PPO"],
    }


class CheckpointValidationTests(unittest.TestCase):
    def test_equal_portable_semantic_metadata_passes_and_ignores_extra_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            digest = _write_synthetic_zip(checkpoint, marker="same-shape-a")
            saved = _metadata(digest)
            saved["path"] = "/machine/a/outputs/model.zip"
            expected = _metadata(digest)
            validate_checkpoint_metadata(checkpoint, saved, expected)

    def test_same_shape_zip_replacement_is_rejected_by_actual_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            original_hash = _write_synthetic_zip(checkpoint, marker="same-shape-a")
            saved = _metadata(original_hash)
            _write_synthetic_zip(checkpoint, marker="same-shape-b")
            with self.assertRaisesRegex(
                CheckpointContractError, r"model_sha256 mismatch"
            ):
                validate_checkpoint_metadata(checkpoint, saved, _metadata(original_hash))

    def test_missing_hash_is_rejected_before_semantic_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            digest = _write_synthetic_zip(checkpoint, marker="checkpoint")
            saved = _metadata(digest)
            del saved["model_sha256"]
            with self.assertRaisesRegex(
                CheckpointContractError, r"missing metadata field model_sha256"
            ):
                validate_checkpoint_metadata(checkpoint, saved, _metadata(digest))

    def test_saved_must_be_concrete_dict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            digest = _write_synthetic_zip(checkpoint, marker="checkpoint")
            expected = _metadata(digest)
            with self.assertRaisesRegex(CheckpointContractError, r"saved.*dict"):
                validate_checkpoint_metadata(
                    checkpoint,
                    MappingProxyType(expected),  # type: ignore[arg-type]
                    expected,
                )

    def test_missing_and_mismatched_nested_fields_report_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            digest = _write_synthetic_zip(checkpoint, marker="checkpoint")
            saved = _metadata(digest)
            expected = _metadata(digest)

            missing = copy.deepcopy(saved)
            del missing["observation_schema"]["dtype"]  # type: ignore[index]
            with self.assertRaisesRegex(
                CheckpointContractError, r"missing metadata field observation_schema\.dtype"
            ):
                validate_checkpoint_metadata(checkpoint, missing, expected)

            mismatched_weight = copy.deepcopy(expected)
            mismatched_weight["pp_weight"] = 0.2
            with self.assertRaisesRegex(
                CheckpointContractError, r"metadata field pp_weight mismatch"
            ):
                validate_checkpoint_metadata(checkpoint, saved, mismatched_weight)

            mismatched_bounds = copy.deepcopy(expected)
            mismatched_bounds["observation_schema"]["high"] = [  # type: ignore[index]
                1.0,
                1.0,
                0.5,
            ]
            with self.assertRaisesRegex(
                CheckpointContractError,
                r"metadata field observation_schema\.high\[2\] mismatch",
            ):
                validate_checkpoint_metadata(checkpoint, saved, mismatched_bounds)

            mismatched_order = copy.deepcopy(expected)
            mismatched_order["wrapper_order"] = [  # type: ignore[index]
                "raw_host",
                "Monitor",
                "lookahead_learning",
                "PPO",
            ]
            with self.assertRaisesRegex(
                CheckpointContractError, r"metadata field wrapper_order\[1\] mismatch"
            ):
                validate_checkpoint_metadata(checkpoint, saved, mismatched_order)

    def test_nested_list_length_and_type_are_strict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            digest = _write_synthetic_zip(checkpoint, marker="checkpoint")
            saved = _metadata(digest)
            expected = _metadata(digest)
            expected["observation_schema"]["shape"] = [265, 1]  # type: ignore[index]
            with self.assertRaisesRegex(
                CheckpointContractError,
                r"metadata field observation_schema\.shape list length mismatch",
            ):
                validate_checkpoint_metadata(checkpoint, saved, expected)

            expected = _metadata(digest)
            expected["observation_schema"]["shape"] = (265,)  # type: ignore[index]
            with self.assertRaisesRegex(
                CheckpointContractError, r"metadata field observation_schema\.shape must be a tuple"
            ):
                validate_checkpoint_metadata(checkpoint, saved, expected)

    def test_nonfinite_values_are_rejected_in_saved_or_expected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.zip"
            digest = _write_synthetic_zip(checkpoint, marker="checkpoint")
            saved = _metadata(digest)
            saved["normalization_config"]["distance_m"] = float("nan")  # type: ignore[index]
            with self.assertRaisesRegex(
                CheckpointContractError,
                r"normalization_config\.distance_m.*finite",
            ):
                validate_checkpoint_metadata(checkpoint, saved, _metadata(digest))

            saved = _metadata(digest)
            expected = _metadata(digest)
            expected["normalization_config"]["distance_m"] = float("inf")  # type: ignore[index]
            with self.assertRaisesRegex(
                CheckpointContractError,
                r"normalization_config\.distance_m.*finite",
            ):
                validate_checkpoint_metadata(checkpoint, saved, expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
