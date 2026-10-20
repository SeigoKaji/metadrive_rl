#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash run_scene_catalogs.sh TARGET [TARGET ...]

  curve          curve.toml: 全5パターン × 12 seed
  intersection   intersection.toml: 全6パターン × 12 seed
  overtake       overtake.toml: respawn/accidentの2・3車線、4パターン × 8 seed
  overtake_all   overtake.toml: 全8パターン × 8 seed
  slow           overtake_slow_lead.toml: 全11パターン × 8 seed（88シーン）
  alternating    overtake_alternating.toml: 連続追い越し4パターン × seed0
  gallery        生成済みの5カタログから統合HTMLだけを更新（シーンの再実行なし）
  all            curve → intersection → overtake → slow → alternating（256シーン）
  -h, --help     このヘルプを表示

slow は overtake_slow_lead でも指定できます。設定値は各TOMLをそのまま使います。
出力: outputs/scene_catalog/index.html（統合一覧）、<TOML名>/index.html とPNG/GIF/TOML。
overtake と overtake_all は同じ overtake/index.html を32行/64行で上書きします。
実行場所は任意です。Pythonの既定はリポジトリの .venv/bin/python です。
変更例: PYTHON_BIN=/path/to/python bash run_scene_catalogs.sh slow
EOF
}

if (($# == 0)); then
    usage
    exit 0
fi

targets=()
for target in "$@"; do
    case "$target" in
        -h|--help) usage; exit 0 ;;
        all) targets+=(curve intersection overtake slow alternating) ;;
        curve|intersection|overtake|overtake_all|slow|overtake_slow_lead|alternating|gallery)
            targets+=("$target") ;;
        *) printf '不明な対象: %s\n' "$target" >&2; usage >&2; exit 2 ;;
    esac
done

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
cd "$PROJECT_ROOT"

for target in "${targets[@]}"; do
    extra_args=()
    case "$target" in
        gallery)
            printf 'catalog_gallery=outputs/scene_catalog/index.html\n'
            "$PYTHON_BIN" tools/scene_catalog_gallery.py
            continue ;;
        slow|overtake_slow_lead) catalog=overtake_slow_lead ;;
        alternating) catalog=overtake_alternating ;;
        overtake)
            catalog=overtake
            extra_args=(--variant lane2_respawn lane3_respawn lane2_accident lane3_accident) ;;
        overtake_all) catalog=overtake ;;
        *) catalog="$target" ;;
    esac
    printf 'catalog_config=configs/scene_catalog/%s.toml\n' "$catalog"
    "$PYTHON_BIN" tools/scene_catalog.py "configs/scene_catalog/$catalog.toml" "${extra_args[@]}"
done
