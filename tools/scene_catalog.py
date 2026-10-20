"""TOMLで指定したシーン候補を俯瞰描画し、パラメータと並べたHTMLの表にする。

    .venv/bin/python tools/scene_catalog.py configs/scene_catalog/curve.toml
    .venv/bin/python tools/scene_catalog.py configs/scene_catalog/overtake.toml --variant lane2_trigger

出力は ``outputs/scene_catalog/<TOML名>/`` 以下に、一覧表 ``index.html`` と、
シーンごとの ``<variant>/seed_<seed>.png``（``gif = true`` なら ``.gif`` も）および
学習用TOMLへそのまま写せる環境設定 ``.toml``。
全画像で縮尺 [px/m] を固定し、表示倍率も揃えるため、カーブ半径や道路長は画像間で
直接比較できる。表の見出しをクリックするとその列で並べ替える。
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    # tools/のファイルを直接実行しても、repo直下の共通モジュールを読めるようにする。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from env_factory import make_env
from evaluation_visualization import derive_timing
from project_paths import OUTPUT_DIR


MARGIN_M = 10.0
THUMBNAIL_SCALE = 0.75  # 表に載せる画像の表示倍率。全画像で共通にして縮尺を揃える。
OUTCOME_KEYS = (
    "arrive_dest", "crash_vehicle", "crash_object", "crash_building", "crash_human", "out_of_road"
)


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """``map_config`` のようなtableは1段だけ再帰的に上書きする。"""

    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _toml_value(value: Any) -> str:
    """環境設定の値をTOMLの1行表記へ変換する。"""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        items = ", ".join(f"{key} = {_toml_value(item)}" for key, item in value.items())
        return "{ " + items + " }"
    # str・数値・配列はJSONとTOMLで同じ表記になる。
    return json.dumps(value)


def _route_summary(env: Any) -> dict[str, Any]:
    """ナビゲーション経路の総延長・正味の旋回角・カーブ半径を求める。

    旋回角は画像上の進行方向に対し、正が左折・反時計回り、負が右折・時計回り。
    """

    navigation = env.agent.navigation
    graph = navigation.map.road_network.graph
    checkpoints = list(navigation.checkpoints)
    length = 0.0
    turn = 0.0
    curves = []
    for start, end in zip(checkpoints[:-1], checkpoints[1:]):
        lane = graph[start][end][0]
        length += lane.length
        delta = lane.heading_theta_at(lane.length) - lane.heading_theta_at(0.0)
        delta = math.degrees((delta + math.pi) % (2.0 * math.pi) - math.pi)
        turn += delta
        radius = getattr(lane, "radius", None)
        if radius is not None and abs(delta) >= 1.0:
            curves.append(f"R{radius:.0f}:{delta:+.0f}")
    return {"route_m": round(length), "turn_deg": round(turn), "curves": " ".join(curves)}


def run_scene(
    env_config: Mapping[str, Any], seed: int, settings: Mapping[str, Any]
) -> tuple[list[np.ndarray], dict[str, Any], int]:
    """1シーンを走らせ、描画フレーム列・経路と走行結果・action間隔[ms]を返す。

    TopDownRendererの画面サイズとカメラ位置は初回render時に固定されるため、
    地図の外接矩形に合わせてシーンごとに環境を作り直す。
    """

    config = _merge(env_config, {"start_seed": seed, "num_scenarios": 1})
    if settings["ego_policy"] == "idm":
        from metadrive.policy.idm_policy import IDMPolicy

        config["agent_policy"] = IDMPolicy
    render_every = int(settings["render_every"])
    max_steps = int(settings["max_steps"])
    if render_every <= 0 or max_steps <= 0:
        raise ValueError("render_every and max_steps must be positive")
    env = make_env(config)
    try:
        env.reset(seed=seed)
        action_ms = (
            derive_timing(env.config).gif_frame_duration_ms if settings.get("gif", False) else 0
        )
        x_min, x_max, y_min, y_max = env.current_map.road_network.get_bounding_box()
        scaling = float(settings["scaling"])
        size = (
            int((x_max - x_min + 2 * MARGIN_M) * scaling),
            int((y_max - y_min + 2 * MARGIN_M) * scaling),
        )
        # MetaDriveはfilmの高さ/max(地図の幅, 高さ)-0.1で縮尺を制限する。
        side = math.ceil((max(x_max - x_min, y_max - y_min) + 2 * MARGIN_M) * (scaling + 0.1))
        view = {
            "mode": "topdown",
            "window": False,
            "screen_record": False,
            "film_size": (side, side),
            "screen_size": size,
            "scaling": scaling,
            "camera_position": ((x_min + x_max) / 2, (y_min + y_max) / 2),
            # 描画フレームを履歴として残し、車両の通った跡を描かせる。
            # 既定は全フレームで、最終フレームに走行軌跡全体が残る。
            "num_stack": int(
                settings.get("trail_frames", max_steps // render_every + 2)
            ),
        }
        summary = _route_summary(env)
        leads = getattr(env.engine, "lead_vehicle_manager", None)
        summary["lead_vehicles"] = (
            f"{len(leads.records)}台: " + "; ".join(
                f'{r["distance_m"]:.1f}m / {r["speed_kmh"]:.1f}km/h / {r["lane_index"]}'
                for r in leads.records
            ) if leads is not None else "0台"
        )
        frames = [np.asarray(env.render(**view), dtype=np.uint8)]
        if hasattr(env.action_space, "n"):
            action: Any = 0
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        lane_index = env.agent.lane_index
        lane_changes = 0
        outcome = "timeout"
        for step in range(1, max_steps + 1):
            _, _, terminated, truncated, info = env.step(action)
            current = env.agent.lane_index
            # 同じ道路区間内でlane番号だけ変わった場合を車線変更として数える。
            if current[:2] == lane_index[:2] and current[2] != lane_index[2]:
                lane_changes += 1
            lane_index = current
            done = terminated or truncated
            if step % render_every == 0 or done or step == max_steps:
                frames.append(np.asarray(env.render(**view), dtype=np.uint8))
            if done:
                outcome = next(
                    (k for k in OUTCOME_KEYS if info.get(k) and env.config.get(f"{k}_done", True)),
                    "timeout",
                )
                break
        summary["overtaken_count"] = leads.overtaken_count if leads is not None else 0
    finally:
        env.close()
    result = {**summary, "steps": step, "lane_changes": lane_changes, "outcome": outcome}
    return frames, result, action_ms


RESULT_COLUMNS = {
    "route_m": "経路長 [m]",
    "turn_deg": "正味の旋回角 [deg]",
    "curves": "カーブ 半径[m]:角度[deg]",
    "lead_vehicles": "先行車",
    "overtaken_count": "追い越した台数",
    "steps": "step数",
    "lane_changes": "車線変更",
    "outcome": "終了理由",
}
HTML_STYLE = """
body { font-family: sans-serif; margin: 16px; }
table { border-collapse: collapse; margin-bottom: 16px; }
th, td { border: 1px solid #ccc; padding: 4px 8px; vertical-align: top; }
th { background: #eee; }
#scenes th { position: sticky; top: 0; cursor: pointer; }
.code { font-family: monospace; white-space: pre; }
"""
# 見出しクリックで、数値なら数値順、それ以外は文字列順に並べ替える。
SORT_SCRIPT = """
document.querySelectorAll("#scenes th").forEach((th, col) => th.onclick = () => {
  const body = th.closest("table").tBodies[0];
  const asc = th.dataset.asc !== "true";
  th.dataset.asc = asc;
  const key = (row) => row.cells[col].textContent.trim();
  [...body.rows].sort((a, b) => {
    const diff = (parseFloat(key(a)) - parseFloat(key(b))) || key(a).localeCompare(key(b));
    return asc ? diff : -diff;
  }).forEach((row) => body.appendChild(row));
});
"""


def _params_toml(
    name: str, env_config: Mapping[str, Any], seed: int, result: Mapping[str, Any]
) -> str:
    """1シーンを学習・評価で再現する環境設定を返す。"""

    summary = "  ".join(f"{key}={value}" for key, value in result.items())
    lines = [
        f"# {name} seed {seed}: 学習用TOMLの同名tableへ写すと、この画像のシーンになる。",
        "# カーブ半径・交差点の進行方向・交通配置はseedで決まり、直接は指定できない。",
        f"# {summary}",
        "",
        "[environment.common]",
        *(f"{key} = {_toml_value(value)}" for key, value in env_config.items()),
        "",
        "[environment.train]",
        f"start_seed = {seed}",
        "num_scenarios = 1",
        "",
        "[environment.evaluation]",
        f"start_seed = {seed}",
        "num_scenarios = 1",
    ]
    return "\n".join(lines) + "\n"


def _write_html(
    path: Path, title: str, base: Mapping[str, Any], rows: list[dict[str, Any]]
) -> None:
    """全variantの共通設定と、1シーン1行の一覧表をHTMLへ書き出す。"""

    esc = html.escape
    base_rows = "".join(
        f"<tr><th>{esc(key)}</th><td class=code>{esc(_toml_value(value))}</td></tr>"
        for key, value in base.items()
    )
    headers = ["画像（クリックで原寸）", "パターン", "パターン固有の設定", "seed"]
    headers += [*RESULT_COLUMNS.values(), "設定ファイル"]
    body = []
    for row in rows:
        cells = [
            f'<a href="{esc(row["link"])}"><img src="{esc(row["image"])}" '
            f'width="{row["width"]}" loading="lazy"></a>',
            esc(row["variant"]),
            f'<span class=code>{esc(row["overrides"])}</span>',
            str(row["seed"]),
            *(esc(str(row["result"][key])) for key in RESULT_COLUMNS),
            f'<a href="{esc(row["toml"])}">{esc(Path(row["toml"]).name)}</a>',
        ]
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    path.write_text(
        f"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8"><title>{esc(title)}</title>
<style>{HTML_STYLE}</style></head><body>
<h1>{esc(title)}</h1>
<p>全画像で縮尺と表示倍率は共通。シーンを使うときは、行の設定ファイルを学習用TOMLへ写す。
カーブ半径・交差点の進行方向・交通配置はseedで決まる。見出しをクリックすると並べ替える。</p>
<h2>全パターン共通の設定</h2>
<table>{base_rows}</table>
<h2>シーン一覧</h2>
<table id=scenes><thead><tr>{"".join(f"<th>{esc(h)}</th>" for h in headers)}</tr></thead>
<tbody>{"".join(body)}</tbody></table>
<script>{SORT_SCRIPT}</script>
</body></html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="configs/scene_catalog/*.toml")
    parser.add_argument("--variant", nargs="*", help="指定したvariantだけ描画する")
    args = parser.parse_args()

    with args.config.open("rb") as file:
        catalog = tomllib.load(file)
    settings = catalog["settings"]
    output_dir = OUTPUT_DIR / "scene_catalog" / args.config.stem
    rows = []
    for variant in catalog["variants"]:
        name = variant["name"]
        if args.variant and name not in args.variant:
            continue
        overrides = variant.get("env", {})
        env_config = _merge(catalog["base"], overrides)
        variant_settings = _merge(settings, variant.get("settings", {}))
        seeds = variant_settings["seeds"]
        (output_dir / name).mkdir(parents=True, exist_ok=True)
        for seed in range(seeds["start"], seeds["start"] + seeds["count"]):
            frames, result, action_ms = run_scene(env_config, seed, variant_settings)
            stem = f"{name}/seed_{seed:04d}"
            Image.fromarray(frames[-1]).save(output_dir / f"{stem}.png")
            (output_dir / f"{stem}.toml").write_text(
                _params_toml(name, env_config, seed, result), encoding="utf-8"
            )
            link = f"{stem}.png"
            if variant_settings.get("gif", False):
                images = [Image.fromarray(frame) for frame in frames[1:]]
                every = int(variant_settings["render_every"])
                # reset直後の画像を除き、末尾の短い区間も実際のstep数に合わせる。
                durations = [every * action_ms] * (len(images) - 1)
                durations.append(((result["steps"] - 1) % every + 1) * action_ms)
                images[0].save(
                    output_dir / f"{stem}.gif",
                    save_all=True,
                    append_images=images[1:],
                    duration=durations,
                    loop=0,
                )
                link = f"{stem}.gif"
            rows.append(
                {
                    "image": link,
                    "link": link,
                    "width": round(frames[-1].shape[1] * THUMBNAIL_SCALE),
                    "variant": name,
                    "overrides": "\n".join(
                        f"{key} = {_toml_value(value)}" for key, value in overrides.items()
                    ),
                    "seed": seed,
                    "result": result,
                    "toml": f"{stem}.toml",
                }
            )
            print(f"[{name}] seed {seed} {result}")
    html_path = output_dir / "index.html"
    _write_html(html_path, f"scene catalog: {args.config.stem}", catalog["base"], rows)
    print(f"html_saved={html_path}")
    from tools.scene_catalog_gallery import build_gallery

    print(f"gallery_saved={build_gallery(output_dir=OUTPUT_DIR / 'scene_catalog')}")


if __name__ == "__main__":
    main()
