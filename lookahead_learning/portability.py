"""Relocation and portability acceptance checks for ``lookahead_learning``.

This module deliberately imports only Python's standard library.  It can be
copied with the other ``lookahead_learning/*.py`` files before optional MetaDrive
dependencies are installed, and its default checks therefore do not create a
simulator.  The public entry point is :func:`run_portability`::

    report = run_portability(
        project_root="/path/to/host",
        output_dir="/tmp/preview portability run",
        base_config="configs/official.toml",
        probe=False,
    )

The harness copies only the Python files required for the add-on and one
requested TOML.  It never copies assets, models, logs, or the source checkout
as a whole.  A fresh subprocess, with ``PYTHONPATH`` removed, runs ``--help``
and static ``doctor`` from a relocated directory containing a space.  An
explicit ``probe=True`` additionally runs a raw host doctor probe and a
two-worker spawn smoke check.  That probe is labeled as the host's raw
observation diagnostic; it does not construct a preview wrapper or claim that
the host satisfies the 262-dimensional lookahead contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib
from typing import Any, Mapping, Sequence


PORTABILITY_SCHEMA_VERSION = "lookahead_learning.portability.v1"
PACKAGE_NAME = "lookahead_learning"
HOST_PYTHON_FILES = ("env_factory.py", "start_lane_env.py", "project_paths.py")
HOST_MODULE_NAMES = (
    "env_factory",
    "start_lane_env",
    "project_paths",
    "configs",
    "configs.experiment_config",
)


class PortabilityError(RuntimeError):
    """The requested relocation inputs are invalid or cannot be copied."""


@dataclass(frozen=True)
class RawWorkerSpec:
    """Picklable raw-probe inputs; no environment or lane crosses a process."""

    project_root: str
    config_path: str
    worker: int
    seed: int
    steps: int = 1


def _json_safe(value: Any) -> Any:
    """Convert common values to strict JSON without importing third parties."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return repr(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _resolve_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise PortabilityError(f"project root is not a directory: {root}")
    return root


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _fresh_output(value: str | os.PathLike[str]) -> Path:
    output = Path(value).expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise PortabilityError(f"portability output is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"refusing to reuse non-empty portability output: {output}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _resolve_config(root: Path, value: str | os.PathLike[str]) -> tuple[Path, Path]:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not _inside(candidate, root):
        raise PortabilityError(
            f"base config must be inside project root: {candidate} (root={root})"
        )
    if candidate.suffix.lower() != ".toml" or not candidate.is_file():
        raise FileNotFoundError(f"base TOML does not exist: {candidate}")
    return candidate, candidate.relative_to(root)


def _configured_probe_seed(config: Path) -> int:
    """Return an evaluation scenario seed declared by the requested TOML.

    The raw probe is a smoke diagnostic, so it must use a scenario that the
    selected bundle actually exposes.  In particular, the official bundle
    has only scenario seed 5; inventing a second seed (for a second worker)
    would turn a process-isolation check into an out-of-range failure.  The
    evaluation range is preferred because that is the range used by doctor;
    training is a fallback for small custom bundles without an evaluation
    table.  ``tomllib`` is part of the Python standard library on supported
    runtimes, so portability remains dependency-free.
    """

    try:
        document = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise PortabilityError(f"cannot read base TOML for probe seed: {config}: {error}") from error
    environment = document.get("environment")
    if not isinstance(environment, Mapping):
        raise PortabilityError("base TOML has no [environment] table for probe seed")
    for stage in ("evaluation", "train"):
        stage_config = environment.get(stage)
        if not isinstance(stage_config, Mapping):
            continue
        value = stage_config.get("start_seed")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return int(value)
    raise PortabilityError(
        "base TOML has no non-negative environment.evaluation.start_seed "
        "or environment.train.start_seed"
    )


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required portability source is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_sources(root: Path, relocated: Path, config: Path, config_relative: Path) -> dict[str, Any]:
    package_source = root / PACKAGE_NAME
    package_destination = relocated / PACKAGE_NAME
    package_files = sorted(package_source.glob("*.py"))
    if not package_files:
        raise FileNotFoundError(f"no Python files found in add-on package: {package_source}")
    copied_preview: list[str] = []
    for source in package_files:
        destination = package_destination / source.name
        _copy_file(source, destination)
        copied_preview.append(str(destination.relative_to(relocated)))

    copied_host: list[str] = []
    for name in HOST_PYTHON_FILES:
        source = root / name
        destination = relocated / name
        _copy_file(source, destination)
        copied_host.append(str(destination.relative_to(relocated)))

    config_source = root / "configs"
    config_py_files = sorted(config_source.glob("*.py"))
    if not config_py_files:
        raise FileNotFoundError(f"no configuration Python files found: {config_source}")
    for source in config_py_files:
        destination = relocated / "configs" / source.name
        _copy_file(source, destination)
        copied_host.append(str(destination.relative_to(relocated)))

    config_destination = relocated / config_relative
    _copy_file(config, config_destination)
    return {
        "lookahead_learning": copied_preview,
        "host_python": copied_host,
        "requested_config": str(config_destination.relative_to(relocated)),
        "all_copied_files": copied_preview + copied_host + [
            str(config_destination.relative_to(relocated))
        ],
    }


def _child_environment(output: Path) -> dict[str, str]:
    """Create a child environment with no source checkout on ``PYTHONPATH``."""

    environment = dict(os.environ)
    # Importing from a relocated cwd must be the only project path selection.
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    cache = output / ".portability_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for name, relative in (
        ("MPLCONFIGDIR", "matplotlib"),
        ("XDG_CACHE_HOME", "xdg"),
        ("TMPDIR", "tmp"),
        ("TORCH_HOME", "torch"),
    ):
        path = cache / relative
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    return environment


def _run_child(
    command: Sequence[str],
    *,
    cwd: Path,
    output: Path,
    timeout: float = 30.0,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=_child_environment(output),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "ok": False,
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
            "duration_s": time.monotonic() - started,
        }
    return {
        "ok": completed.returncode == 0,
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_s": time.monotonic() - started,
    }


def _origin_from_spec(spec: Any) -> str | None:
    origin = getattr(spec, "origin", None)
    if isinstance(origin, str) and origin not in {"", "built-in", "frozen"}:
        return str(Path(origin).resolve())
    locations = getattr(spec, "submodule_search_locations", None)
    if locations:
        try:
            return str(Path(next(iter(locations))).resolve())
        except (StopIteration, TypeError, ValueError):
            return None
    return None


def _normal_import_script(expected_root: Path, original_root: Path) -> str:
    # Keep this script standard-library-only.  ``find_spec`` records origins
    # even for optional host modules whose third-party import is unavailable.
    return (
        "import importlib, importlib.util, json, os, pathlib, sys\n"
        f"expected = pathlib.Path({str(expected_root)!r}).resolve()\n"
        f"original = pathlib.Path({str(original_root)!r}).resolve()\n"
        f"names = {list(HOST_MODULE_NAMES)!r}\n"
        "result = {'expected_root': str(expected), 'original_root': str(original), 'modules': {}, 'residual': []}\n"
        "import lookahead_learning\n"
        "result['modules']['lookahead_learning'] = str(pathlib.Path(lookahead_learning.__file__).resolve())\n"
        "for name in names:\n"
        "    imported = None\n"
        "    error = None\n"
        "    try:\n"
        "        imported = importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        error = f'{type(exc).__name__}: {exc}'\n"
        "    try:\n"
        "        spec = importlib.util.find_spec(name)\n"
        "    except (ImportError, ModuleNotFoundError, ValueError):\n"
        "        spec = None\n"
        "    origin = None if spec is None else (spec.origin or (next(iter(spec.submodule_search_locations), None) if spec.submodule_search_locations else None))\n"
        "    if imported is not None and getattr(imported, '__file__', None):\n"
        "        origin = imported.__file__\n"
        "    result['modules'][name] = {'origin': None if origin is None else str(pathlib.Path(origin).resolve()), 'import_error': error}\n"
        "for name, module in list(sys.modules.items()):\n"
        "    if name == 'lookahead_learning' or name in names or any(name.startswith(item + '.') for item in names if item in ('configs', 'lookahead_learning')):\n"
        "        origin = getattr(module, '__file__', None)\n"
        "        if origin and not str(pathlib.Path(origin).resolve()).startswith(str(expected) + os.sep):\n"
        "            result['residual'].append({'name': name, 'origin': str(pathlib.Path(origin).resolve())})\n"
        "result['lookahead_learning_under_expected'] = result['modules']['lookahead_learning'].startswith(str(expected) + os.sep)\n"
        "result['host_origins_under_expected'] = all(result['modules'][name].get('origin') and result['modules'][name]['origin'].startswith(str(expected) + os.sep) for name in names)\n"
        "source_candidates = {'lookahead_learning': original / 'lookahead_learning' / '__init__.py', 'configs': original / 'configs' / '__init__.py', 'configs.experiment_config': original / 'configs' / 'experiment_config.py', 'env_factory': original / 'env_factory.py', 'start_lane_env': original / 'start_lane_env.py', 'project_paths': original / 'project_paths.py'}\n"
        "result['no_original_residual'] = not result['residual'] and all(item.get('origin') not in {str(path.resolve()) for path in source_candidates.values()} for name, item in result['modules'].items() if name != 'lookahead_learning' and isinstance(item, dict) and item.get('origin'))\n"
        "print('PORTABILITY_RESULT=' + json.dumps(result, sort_keys=True))\n"
    )


def _parse_marker(stdout: str, marker: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            try:
                value = json.loads(line[len(marker) :])
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, dict) else None
    return None


def _static_checks(
    *,
    relocated: Path,
    original_root: Path,
    config_relative: Path,
    output: Path,
) -> dict[str, Any]:
    environment_check = _run_child(
        [
            sys.executable,
            "-B",
            "-c",
            _normal_import_script(relocated, original_root),
        ],
        cwd=relocated,
        output=output,
    )
    import_result = _parse_marker(environment_check.get("stdout", ""), "PORTABILITY_RESULT=")
    environment_check["import_result"] = import_result
    environment_check["ok"] = bool(
        environment_check.get("ok")
        and isinstance(import_result, Mapping)
        and import_result.get("lookahead_learning_under_expected")
        and import_result.get("host_origins_under_expected")
        and import_result.get("no_original_residual")
    )

    help_check = _run_child(
        [sys.executable, "-B", "-m", "lookahead_learning", "--help"],
        cwd=relocated,
        output=output,
    )
    help_check["commands_present"] = all(
        command in help_check.get("stdout", "")
        for command in ("doctor", "train", "evaluate", "compare", "test")
    )
    help_check["ok"] = bool(help_check.get("ok") and help_check["commands_present"])

    doctor_check = _run_child(
        [
            sys.executable,
            "-B",
            "-m",
            "lookahead_learning",
            "doctor",
            "--project-root",
            ".",
            "--config",
            str(config_relative),
        ],
        cwd=relocated,
        output=output,
    )
    doctor_check["ok"] = bool(doctor_check.get("ok"))
    return {
        "normal_import": environment_check,
        "help": help_check,
        "static_doctor": doctor_check,
    }


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(item) for item in shape]
    except (TypeError, ValueError):
        return None


def raw_worker_entry(spec: RawWorkerSpec, result_queue: Any) -> None:
    """Spawn target that creates, resets, steps, and closes one raw env.

    The argument contains only strings/integers.  A parent-created engine,
    environment, or lane object is impossible to pass through this entry
    point.  Errors are serialized to the queue so the parent can close/join
    every child deterministically.
    """

    environment: Any = None
    original_path = list(sys.path)
    try:
        root = Path(spec.project_root).resolve()
        sys.path[:] = [str(root)] + [item for item in original_path if item != str(root)]
        importlib.invalidate_caches()
        preview_package = importlib.import_module("lookahead_learning")
        preview_origin = getattr(preview_package, "__file__", None)
        config_module = importlib.import_module("configs.experiment_config")
        selector = getattr(config_module, "select_experiment", None)
        if not callable(selector):
            raise PortabilityError("copied configs.experiment_config has no select_experiment")
        selection = selector(config_path=str(Path(spec.config_path).resolve()))
        profile = getattr(selection, "profile", None)
        config = getattr(profile, "evaluation_env_config", None)
        if not isinstance(config, Mapping):
            raise PortabilityError("copied config has no evaluation_env_config mapping")
        factory = importlib.import_module("env_factory")
        make_env = getattr(factory, "make_env", None)
        if not callable(make_env):
            raise PortabilityError("copied env_factory has no make_env")
        environment = make_env(dict(config))
        observation, info = environment.reset(seed=int(spec.seed))
        action_space = getattr(environment, "action_space", None)
        action: Any
        action_count = getattr(action_space, "n", None)
        if isinstance(action_count, int) and action_count > 4:
            action = 4
        elif isinstance(action_count, int):
            action = 0
        elif callable(getattr(action_space, "sample", None)):
            action = action_space.sample()
        else:
            action = 0
        steps: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        for index in range(max(1, min(int(spec.steps), 10))):
            next_observation, reward, terminated, truncated, step_info = environment.step(action)
            steps.append(
                {
                    "decision": index + 1,
                    "action_env": _json_safe(action),
                    "reward": _json_safe(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "observation_shape": _shape(next_observation),
                    "info_keys": sorted(str(key) for key in step_info)
                    if isinstance(step_info, Mapping)
                    else None,
                }
            )
            if terminated or truncated:
                break
        initial_shape = _shape(observation)
        result_queue.put(
            {
                "ok": True,
                "worker": int(spec.worker),
                "pid": os.getpid(),
                "seed": int(spec.seed),
                "raw_observation_shape": initial_shape,
                "raw_observation_dtype": str(getattr(observation, "dtype", None)),
                "raw_262_supported": initial_shape == [262],
                "operational_ab_supported": False,
                "engine_created_in_child": True,
                "environment_passed_from_parent": False,
                "lane_passed_from_parent": False,
                "lookahead_learning_origin": None
                if preview_origin is None
                else str(Path(preview_origin).resolve()),
                "lookahead_learning_under_project_root": (
                    preview_origin is not None
                    and _inside(Path(preview_origin), root)
                ),
                "steps": steps,
            }
        )
    except Exception as error:
        result_queue.put(
            {
                "ok": False,
                "worker": int(spec.worker),
                "pid": os.getpid(),
                "seed": int(spec.seed),
                "error": f"{type(error).__name__}: {error}",
                "engine_created_in_child": False,
                "environment_passed_from_parent": False,
                "lane_passed_from_parent": False,
                "lookahead_learning_origin": None,
                "lookahead_learning_under_project_root": False,
            }
        )
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def run_spawn_raw_probe(
    project_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    seeds: Sequence[int] = (5, 5),
    steps: int = 1,
) -> dict[str, Any]:
    """Run two independent raw reset/step workers using multiprocessing spawn."""

    root = _resolve_root(project_root)
    config, _config_relative = _resolve_config(root, config_path)
    output = _fresh_output(output_dir)
    seed_values = tuple(int(seed) for seed in seeds)
    if len(seed_values) != 2:
        raise ValueError("raw spawn probe requires exactly two seeds")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes: list[multiprocessing.Process] = []
    specs = [
        RawWorkerSpec(
            project_root=str(root),
            config_path=str(config),
            worker=index,
            seed=int(seed),
            steps=int(steps),
        )
        for index, seed in enumerate(seed_values)
    ]
    started = time.monotonic()
    for spec in specs:
        process = context.Process(target=raw_worker_entry, args=(spec, queue))
        process.start()
        processes.append(process)
    results: list[dict[str, Any]] = []
    for _ in processes:
        try:
            value = queue.get(timeout=120.0)
        except Exception as error:
            value = {"ok": False, "error": f"queue result unavailable: {type(error).__name__}: {error}"}
        results.append(dict(value) if isinstance(value, Mapping) else {"ok": False, "error": repr(value)})
    for process in processes:
        process.join(timeout=30.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
    pids = [item.get("pid") for item in results if item.get("pid") is not None]
    successful = all(
        item.get("ok")
        and item.get("lookahead_learning_under_project_root") is True
        and item.get("steps")
        and all(
            isinstance(step.get("observation_shape"), (list, tuple))
            and bool(step.get("observation_shape"))
            for step in item.get("steps", [])
        )
        for item in results
    )
    distinct_pids = len(set(pids)) == len(pids) == 2
    all_engines_created_in_child = all(
        item.get("engine_created_in_child") is True for item in results
    )
    all_preview_origins_under_project_root = all(
        item.get("lookahead_learning_under_project_root") is True for item in results
    )
    no_parent_environment_or_lane = all(
        item.get("environment_passed_from_parent") is False
        and item.get("lane_passed_from_parent") is False
        for item in results
    )
    return {
        "ok": bool(
            successful
            and len(results) == 2
            and distinct_pids
            and all_engines_created_in_child
            and all_preview_origins_under_project_root
            and no_parent_environment_or_lane
        ),
        "worker_count": 2,
        "results": sorted(results, key=lambda item: int(item.get("worker", 999))),
        "distinct_child_pids": distinct_pids,
        "all_engines_created_in_child": all_engines_created_in_child,
        "all_lookahead_learning_origins_under_project_root": all_preview_origins_under_project_root,
        "no_parent_environment_or_lane": no_parent_environment_or_lane,
        "duration_s": time.monotonic() - started,
        "diagnostic_label": "legacy_raw259_diagnostic",
        "operational_ab_supported": False,
    }


def run_portability(
    project_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    base_config: str | os.PathLike[str],
    probe: bool = False,
) -> dict[str, Any]:
    """Copy the add-on into a relocated host and run portability checks.

    The destination must be new or empty.  The returned report is also saved
    as ``portability_report.json`` below that destination.  With
    ``probe=False`` the function performs no simulator initialization.
    """

    root = _resolve_root(project_root)
    output = _fresh_output(output_dir)
    config, config_relative = _resolve_config(root, base_config)
    relocated = output / "relocated host"
    relocated.mkdir(parents=True, exist_ok=False)
    copied = _copy_sources(root, relocated, config, config_relative)
    checks = _static_checks(
        relocated=relocated,
        original_root=root,
        config_relative=config_relative,
        output=output,
    )
    report: dict[str, Any] = {
        "schema_version": PORTABILITY_SCHEMA_VERSION,
        "status": "static",
        "ok": all(item.get("ok") is True for item in checks.values()),
        "probe_requested": bool(probe),
        "source_project_root": str(root),
        "relocated_root": str(relocated),
        "output_dir": str(output),
        "copied": copied,
        "checks": checks,
        "assets_copied": False,
        "models_copied": False,
        "different_pc_verification": False,
        "notes": [
            "This is a local relocation/subprocess acceptance check, not verification on a different PC.",
            "No assets or model/checkpoint files are copied.",
            "Static checks do not initialize MetaDrive.",
        ],
    }
    if probe:
        probe_output = output / "raw probe"
        probe_seed = _configured_probe_seed(config)
        report["probe_scenario_seed"] = probe_seed
        # Leave this path absent: runner._fresh_run_dir intentionally refuses
        # an existing destination so a probe can never append to an old run.
        raw_doctor = _run_child(
            [
                sys.executable,
                "-B",
                "-m",
                "lookahead_learning",
                "doctor",
                "--project-root",
                ".",
                "--config",
                str(config_relative),
                "--probe",
                "--seed",
                str(probe_seed),
                "--steps",
                "1",
                "--output",
                str(probe_output),
            ],
            cwd=relocated,
            # ``_run_child`` creates only cache paths below this existing
            # parent output.  The explicit doctor destination itself must stay
            # absent until runner._fresh_run_dir creates it.
            output=output,
            timeout=180.0,
        )
        report["raw_doctor_probe"] = raw_doctor
        doctor_path = probe_output / "doctor.json"
        doctor_report: dict[str, Any] | None = None
        if doctor_path.is_file():
            try:
                value = json.loads(doctor_path.read_text(encoding="utf-8"))
                doctor_report = value if isinstance(value, dict) else None
            except (OSError, json.JSONDecodeError):
                doctor_report = None
        report["raw_doctor_report"] = doctor_report
        probe_data = doctor_report.get("probe") if isinstance(doctor_report, Mapping) else None
        reset_shape = probe_data.get("reset_observation_shape") if isinstance(probe_data, Mapping) else None
        steps_data = probe_data.get("steps") if isinstance(probe_data, Mapping) else None
        raw_step_ok = bool(
            isinstance(steps_data, Sequence)
            and len(steps_data) > 0
            and all(
                isinstance(step, Mapping)
                and isinstance(step.get("observation_shape"), Sequence)
                and len(step.get("observation_shape", ())) > 0
                for step in steps_data
            )
        )
        raw_shape_expected = reset_shape == [259]
        report["raw_probe_contract"] = {
            "diagnostic_label": "legacy_raw259_diagnostic",
            "reset_observation_shape": reset_shape,
            "expected_current_shape": [259],
            "expected_shape_observed": raw_shape_expected,
            "raw_262_supported": reset_shape == [262],
            "operational_ab_supported": False,
            "raw_step_ok": raw_step_ok,
        }
        # Run the multiprocessing pool from a fresh process whose cwd is the
        # relocated host.  Calling ``run_spawn_raw_probe`` directly here would
        # let ``spawn`` pickle this original checkout's module path; this
        # process instead imports the copied ``lookahead_learning.portability``.
        spawn_relative_output = Path("raw probe") / "spawn workers"
        spawn_process = _run_child(
            [
                sys.executable,
                "-B",
                "-m",
                "lookahead_learning.portability",
                "--spawn-probe",
                "--project-root",
                ".",
                "--config",
                str(config_relative),
                "--output",
                str(spawn_relative_output),
                "--seeds",
                str(probe_seed),
                str(probe_seed),
                "--steps",
                "1",
            ],
            cwd=relocated,
            # Keep child caches in the parent acceptance output.  The spawn
            # run directory itself is created by run_spawn_raw_probe and is
            # therefore fresh, as required by the normal runner contract.
            output=output,
            timeout=240.0,
        )
        spawn_result = _parse_marker(
            spawn_process.get("stdout", ""), "SPAWN_PROBE_RESULT="
        )
        report["spawn_raw_probe_process"] = {
            **spawn_process,
            "result": spawn_result,
        }
        report["spawn_raw_probe"] = spawn_result or {
            "ok": False,
            "diagnostic_label": "legacy_raw259_diagnostic",
            "error": "relocated spawn probe emitted no valid result marker",
            "operational_ab_supported": False,
        }
        report["probe_ok"] = bool(
            raw_doctor.get("ok")
            and raw_shape_expected
            and raw_step_ok
            and spawn_process.get("ok") is True
            and isinstance(report.get("spawn_raw_probe"), Mapping)
            and report["spawn_raw_probe"].get("ok") is True
        )
        report["ok"] = bool(report["ok"] and report["probe_ok"])
        report["status"] = "probe"
    report_path = output / "portability_report.json"
    report["report_path"] = str(report_path)
    _write_json(report_path, report)
    return _json_safe(report)


def _spawn_probe_main(argv: Sequence[str] | None = None) -> int:
    """CLI target used by a fresh relocated process for spawn validation."""

    parser = argparse.ArgumentParser(description="lookahead_learning raw spawn portability probe")
    parser.add_argument(
        "--spawn-probe",
        action="store_true",
        help="run the two-worker raw reset/step check (the module entry point already selects it)",
    )
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs=2, default=(5, 5))
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)
    result = run_spawn_raw_probe(
        args.project_root,
        args.config,
        args.output,
        seeds=tuple(args.seeds),
        steps=args.steps,
    )
    print("SPAWN_PROBE_RESULT=" + json.dumps(_json_safe(result), sort_keys=True))
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess
    raise SystemExit(_spawn_probe_main())


# Explicit descriptive aliases for callers that name the spawn harness.
spawn_raw_workers = run_spawn_raw_probe
portability_check = run_portability


__all__ = [
    "HOST_MODULE_NAMES",
    "HOST_PYTHON_FILES",
    "PACKAGE_NAME",
    "PORTABILITY_SCHEMA_VERSION",
    "PortabilityError",
    "RawWorkerSpec",
    "portability_check",
    "raw_worker_entry",
    "run_portability",
    "run_spawn_raw_probe",
    "spawn_raw_workers",
]
