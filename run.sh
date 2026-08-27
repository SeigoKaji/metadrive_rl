#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG_PATH="${1:-configs/official.toml}"

cd "$PROJECT_ROOT"

.venv/bin/python train.py --config "$CONFIG_PATH"
.venv/bin/python evaluate.py --config "$CONFIG_PATH"
