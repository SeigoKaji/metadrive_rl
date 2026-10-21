"""既存の正式カタログを、再シミュレーションなしで1ページにまとめる。"""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import sys
import tomllib
from urllib.parse import quote, unquote, urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.experiment_config import SCENE_GROUP_SWITCH_KEYS


# 一時的なprobe/smokeの設定や出力は統合しない。
CATALOGS = {
    "overtake_alternating": "連続車線変更の追い越し",
    "overtake_slow_lead": "遅い先行車の追い越し",
    "overtake": "通常交通の追い越し",
    "intersection": "交差点",
    "curve": "カーブ",
}
FIELDS = {
    "パターン": "variant",
    "パターン固有の設定": "settings",
    "seed": "seed",
    "経路長 [m]": "route_m",
    "正味の旋回角 [deg]": "turn_deg",
    "カーブ 半径[m]:角度[deg]": "curves",
    "先行車": "lead_vehicles",
    "追い越した台数": "overtaken_count",
    "step数": "steps",
    "車線変更": "lane_changes",
    "終了理由": "outcome",
}


def _toml_value(value: object) -> str:
    """書き出す値はPythonで引用し、JSには完成したTOML片だけ渡す。"""

    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_value(k)} = {_toml_value(v)}" for k, v in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    raise ValueError(f"書き出せないTOML値です: {type(value).__name__}")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _selection_data(scene: dict, directory: Path) -> None:
    """壊れた設定でも閲覧は維持し、選択時に理由を表示する。"""

    identity = "missing"
    scene.update(selection_error=None, random_traffic=False)
    try:
        source = (directory / unquote(scene["toml"])).read_text(encoding="utf-8")
        identity = source
        config = tomllib.loads(source)
        identity = _canonical(config)
        common = config.get("environment", {}).get("common")
        if not isinstance(common, dict):
            raise ValueError("[environment.common] がありません")
        switches = {key: common[key] for key in sorted(SCENE_GROUP_SWITCH_KEYS) if key in common}
        if ("map" in switches) == ("map_config" in switches):
            raise ValueError("map / map_config はどちらか一方だけ必要です")
        seed = int(scene["seed"])
        if seed < 0:
            raise ValueError("seed は0以上の整数が必要です")
        random_traffic = common.get("random_traffic", False)
        if not isinstance(random_traffic, bool):
            raise ValueError("random_traffic はboolが必要です")
        if "/" in scene["variant"] or "\\" in scene["variant"]:
            raise ValueError("パターン名をグループのbasenameに使えません")
        scene.update(
            switch_settings=switches, random_traffic=random_traffic, seed_literal=str(seed),
            group_key=_canonical([scene["catalog"], scene["variant"], switches]),
            common_toml="\n".join(f"{_toml_value(k)} = {_toml_value(v)}" for k, v in switches.items()),
        )
    except (OSError, UnicodeError, ValueError, TypeError, AttributeError) as exc:
        scene["selection_error"] = f"設定TOMLを選択に使えません: {exc}"
    scene["selection_key"] = hashlib.sha256(
        _canonical([scene["catalog"], scene["variant"], scene["seed"], identity]).encode("utf-8")
    ).hexdigest()


def _selection_names(scenes: list[dict]) -> None:
    # 素のパターン名の末尾が _2 等でも、別設定への連番と衝突させない。
    reserved = {f"{s['catalog']}__{s['variant']}" for s in scenes}
    names, used = {}, set()
    for scene in scenes:
        if scene["selection_error"]:
            continue
        key, base = scene["group_key"], f"{scene['catalog']}__{scene['variant']}"
        if key not in names:
            name, suffix = base, 2
            while name in used or (name != base and name in reserved):
                name, suffix = f"{base}_{suffix}", suffix + 1
            names[key] = _toml_value(name)
            used.add(name)
        scene["group_name_literal"] = names[key]


SELECTION_INSTRUCTIONS = """# scene_selection.toml（シーンカタログの選択）
# 【GitHub Copilot への指示】このファイルとベースの学習用configから、新しいconfigを作る。
# 1. ベースを configs/<新しい名前>.toml にコピーし、name を新しい名前にする
#    （指定がなければ selection.suggested_name）。[training]・[evaluation]・[lookahead] と
#    独自tableを含む environment 以外の設定は、name を除いてすべてそのまま残す。
# 2. environment.common / environment.train / environment.evaluation のすべてから
#    次の切り替えキーを削除する（グループ内の設定だけを使い、map と map_config の競合を防ぐ）:
#    __SWITCH_KEYS__
#    start_seed・num_scenarios・scenario_seeds も上記3tableすべてから削除する。
#    random_traffic は common に false を設定し、stage側にあれば false に揃える。
#    行動形式・報酬・終了条件など残りの設定は維持し、空になったstage tableは削除してよい。
# 3. ベースの既存 environment.scene_groups は残さず、このファイルの全グループで置換する。
#    [[environment.scene_groups]] と [environment.scene_groups.common] だけをすべて末尾へ貼る。
#    [selection] はメタ情報なので完成した実験configへコピーしない。
#    評価未選択のグループは、明記された学習と同じseedで評価する。
# 4. 作ったconfigで次の両方を実行し、PASSを確認して結果を報告する:
#    .venv/bin/python inspect_env.py --config <新しいconfig>
#    .venv/bin/python inspect_env.py --config <新しいconfig> --stage evaluation
""".replace("__SWITCH_KEYS__", ", ".join(sorted(SCENE_GROUP_SWITCH_KEYS)))


EXPORT_SCRIPT = r"""
function buildSelectionExport(selectedScenes, metadata) {
  const selected = selectedScenes.filter((s) => s.train || s.evaluation);
  if (!selected.length) throw new Error("学習または評価のシーンを選択してください。");
  const groups = new Map();
  selected.forEach((s) => {
    const label = `${s.catalog} / ${s.variant} / seed ${s.seed}`;
    if (s.selection_error) throw new Error(`${label}: ${s.selection_error}`);
    if (s.random_traffic) throw new Error(`${label}: random_traffic = true は書き出せません。選択を解除してください。`);
    if (!groups.has(s.group_key)) groups.set(s.group_key, {scene: s, train: [], evaluation: []});
    const group = groups.get(s.group_key);
    ["train", "evaluation"].forEach((stage) => {
      if (s[stage] && !group[stage].includes(s.seed_literal)) group[stage].push(s.seed_literal);
    });
  });
  groups.forEach((group) => {
    if (!group.train.length) throw new Error(`${group.scene.catalog} / ${group.scene.variant}: 評価だけのグループがあります。学習にもチェックしてください。`);
  });
  if (!/^[A-Za-z0-9_-]+$/.test(metadata.suggested_name) || !/^[0-9TZ:.+-]+$/.test(metadata.created_at)) {
    throw new Error("書き出し名または生成日時が不正です。");
  }
  const parts = [metadata.instructions, `[selection]\nschema = "scene_selection/1"\nsuggested_name = "${metadata.suggested_name}"\ncreated_at = "${metadata.created_at}"\n`];
  groups.forEach((group) => {
    const evaluation = group.evaluation.length ? group.evaluation : group.train;
    parts.push("[[environment.scene_groups]]\nname = " + group.scene.group_name_literal
      + "\ntrain_seeds = [" + group.train.join(", ") + "]\n"
      + (group.evaluation.length ? "" : "# 評価未選択のため、学習と同じseedで評価する。\n")
      + "evaluation_seeds = [" + evaluation.join(", ") + "]\n[environment.scene_groups.common]\n"
      + group.scene.common_toml + "\n");
  });
  return parts.join("\n");
}
"""


class _SceneTableParser(HTMLParser):
    """既存HTMLのscenes表だけを読み、セルの文字列とリンクを保持する。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.headers = []
        self.rows = []
        self._depth = 0
        self._cells = []
        self._cell = None
        self._header = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            if self._depth or attrs.get("id") == "scenes":
                self._depth += 1
            return
        if not self._depth:
            return
        if tag == "tr":
            self._cells = []
            self._header = False
        elif tag in {"th", "td"}:
            self._cell = {"text": "", "links": [], "image": None}
            self._header |= tag == "th"
        elif self._cell is not None:
            if tag == "a" and "href" in attrs:
                self._cell["links"].append(attrs["href"])
            elif tag == "img":
                self._cell["image"] = attrs
            elif tag == "br":
                self._cell["text"] += "\n"

    def handle_data(self, data):
        if self._depth and self._cell is not None:
            self._cell["text"] += data

    def handle_endtag(self, tag):
        if not self._depth:
            return
        if tag == "table":
            self._depth -= 1
        elif tag in {"th", "td"} and self._cell is not None:
            self._cell["text"] = self._cell["text"].strip()
            self._cells.append(self._cell)
            self._cell = None
        elif tag == "tr" and self._cells:
            if self._header:
                self.headers = [cell["text"] for cell in self._cells]
            else:
                self.rows.append(self._cells)


def _asset_url(catalog: str, value: str) -> str:
    """ローカルの相対リンクだけを受け付け、統合ページ基準へ直す。"""

    url = urlsplit(value)
    path = PurePosixPath(unquote(url.path))
    if (
        url.scheme or url.netloc or url.query or url.fragment
        or not url.path or path.is_absolute() or ".." in path.parts or "\\" in value
    ):
        raise ValueError(f"カタログ内の相対ファイルではありません: {value!r}")
    return quote(f"{catalog}/{path.as_posix()}", safe="/")


def _read_scenes(path: Path, catalog: str) -> list[dict]:
    parser = _SceneTableParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    required = set(FIELDS) - {"先行車", "追い越した台数"}
    if not required.issubset(parser.headers) or "設定ファイル" not in parser.headers:
        raise ValueError(f"シーン表の列を読み取れません: {path}")
    image_column = next(
        (i for i, title in enumerate(parser.headers) if title.startswith("画像（")), None
    )
    if image_column is None:
        raise ValueError(f"画像列がありません: {path}")
    scenes = []
    for cells in parser.rows:
        if len(cells) != len(parser.headers):
            raise ValueError(f"シーン表の列数が一致しません: {path}")
        columns = dict(zip(parser.headers, cells))
        media = cells[image_column]
        image = media["image"]
        toml = columns["設定ファイル"]["links"]
        if not image or not image.get("src") or not media["links"] or not toml:
            raise ValueError(f"シーンの画像または設定リンクがありません: {path}")
        width = int(image.get("width", 768))
        if width <= 0:
            raise ValueError(f"画像の表示幅が不正です: {path}")
        link = media["links"][0]
        preview = link if urlsplit(link).path.lower().endswith(".gif") else image["src"]
        scene = {
            "catalog": catalog,
            "catalog_label": CATALOGS[catalog],
            "image": _asset_url(catalog, preview),
            "link": _asset_url(catalog, link),
            "toml": _asset_url(catalog, toml[0]),
            "width": width,
        }
        scene.update({
            key: columns[title]["text"] if title in columns else None
            for title, key in FIELDS.items()
        })
        _selection_data(scene, path.parent.parent)
        scenes.append(scene)
    return scenes


PAGE = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>シーンカタログ</title>
<style>
:root { color-scheme: light; font-family: system-ui, sans-serif; color: #172535; background: #f2f5f8; }
* { box-sizing: border-box; }
body { margin: 0; overflow: hidden; }
main { display: grid; grid-template-columns: 280px minmax(0, 1fr); gap: 12px; height: 100vh; height: 100dvh; padding: 12px; }
main.sidebar-collapsed { grid-template-columns: 48px minmax(0, 1fr); }
.sidebar { min-width: 0; min-height: 0; overflow-y: auto; padding: 4px 4px 4px 0; }
.sidebar-toggle-bar { position: sticky; top: 0; z-index: 3; padding-bottom: 8px; background: #f2f5f8; }
.sidebar-toggle { display: flex; align-items: center; justify-content: center; gap: 8px; width: 100%; min-height: 40px; }
.sidebar-toggle:focus-visible { outline-offset: -3px; }
.sidebar-toggle-icon { font-size: 1.25rem; line-height: 1; }
.sidebar-collapsed .sidebar-toggle { flex-direction: column; gap: 2px; padding: 6px 2px; }
.sidebar-collapsed .sidebar-toggle-label { font-size: .75rem; }
.sidebar-controls > summary { display: none; }
h1 { margin: 0 0 6px; font-size: 1.5rem; }
p { margin: 6px 0; }
a { color: #155b94; }
button, select, input { font: inherit; }
button, select, input[type=search] { border: 1px solid #9eafbf; border-radius: 6px; padding: 7px 9px; background: white; color: inherit; }
button { cursor: pointer; }
button:disabled { opacity: .45; cursor: default; }
:focus-visible { outline: 3px solid #2676bb; outline-offset: 3px; }
.intro, .hint { color: #526476; font-size: .9rem; }
.toolbar { margin: 12px 0; padding: 12px; background: white; border: 1px solid #d0dbe5; border-radius: 9px; }
.filters { display: grid; gap: 10px; }
label { display: flex; flex-direction: column; gap: 5px; font-size: .85rem; font-weight: 600; min-width: 0; }
select, input { width: 100%; min-width: 0; }
.options { display: grid; gap: 10px; margin-top: 12px; }
.selection-actions { display: grid; gap: 8px; margin: 12px 0; padding: 12px; border: 1px solid #b8c8d7; border-radius: 9px; background: white; }
.selection-counts, .scene-selection { display: flex; flex-wrap: wrap; gap: 8px 16px; }
.scene-selection { margin-top: 8px; }
.scene-selection label { display: inline-flex; flex-direction: row; align-items: center; gap: 5px; }
.scene-selection input { width: auto; min-width: auto; }
#selectionWarning { margin: 0; color: #8c2923; font-size: .85rem; overflow-wrap: anywhere; }
.status-row, .pagination { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-top: 12px; }
#status { margin: 0; font-weight: 600; }
.gallery-scroll { min-width: 0; min-height: 0; height: 100%; overflow-y: auto; container: gallery / inline-size; border-radius: 9px; border: 1px solid #ced9e3; background: #e8eef3; }
.column-headings, .scene-summary { display: grid; grid-template-columns: minmax(180px, 1.4fr) 56px minmax(260px, 1.5fr) 66px; gap: 12px; align-items: center; padding: 12px 16px; }
.column-headings { position: sticky; top: 0; z-index: 2; background: #e0e8f0; font-size: .85rem; font-weight: 700; border-bottom: 1px solid #b9c8d6; }
.multi-column-heading { display: none; }
#sceneList { display: grid; grid-template-columns: minmax(0, 1fr); gap: 12px; padding: 12px; align-items: start; }
.gallery-scroll[data-columns="2"] #sceneList { grid-template-columns: repeat(2, minmax(0, 1fr)); }
.gallery-scroll[data-columns="2"] .column-headings { display: block; }
.gallery-scroll[data-columns="2"] .column-headings > span { display: none; }
.gallery-scroll[data-columns="2"] .column-headings .multi-column-heading { display: block; }
.scene-card { position: relative; min-width: 0; container: scene / inline-size; background: white; border: 1px solid #ccd8e3; border-radius: 8px; overflow: hidden; }
.scene-summary { min-height: 78px; border-bottom: 1px solid #e0e7ee; }
.catalog-label { display: block; color: #536a80; font-size: .8rem; margin-bottom: 4px; }
.variant { font-size: .95rem; overflow-wrap: anywhere; }
.seed { font-variant-numeric: tabular-nums; font-size: .9rem; }
.result { display: flex; flex-wrap: wrap; align-items: center; gap: 7px 12px; font-size: .85rem; }
.outcome { display: inline-block; font-weight: 700; padding: 4px 8px; border-radius: 20px; background: #edf2f7; }
.outcome.arrive_dest { background: #dff3e8; color: #185539; }
.outcome.timeout { background: #fff0cf; color: #765212; }
.outcome.problem { background: #fbe1df; color: #8c2923; }
.scene-details > summary { position: absolute; right: 16px; top: 21px; cursor: pointer; padding: 6px 8px; border: 1px solid #b8c8d7; border-radius: 6px; font-size: .85rem; background: #f5f8fb; }
.details-content { padding: 14px 16px; background: #f8fafc; border-bottom: 1px solid #d9e3ec; display: grid; grid-template-columns: minmax(180px, 1fr) minmax(0, 2fr); gap: 20px; }
dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 7px 12px; align-content: start; font-size: .85rem; }
dt { color: #566b7d; } dd { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
pre { margin: 0 0 8px; white-space: pre-wrap; overflow-wrap: anywhere; font-size: .8rem; line-height: 1.6; }
.detail-links { display: flex; flex-wrap: wrap; gap: 14px; font-size: .85rem; }
.preview-bar { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; padding: 8px 16px; font-size: .8rem; color: #526476; }
.preview-scroll { width: 100%; min-width: 0; overflow-x: auto; min-height: 110px; background: #fff; border-top: 1px solid #edf1f5; }
.preview-scroll > a { display: block; width: max-content; min-width: 100%; }
.preview-scroll img { display: block; max-width: none; height: auto; margin: auto; }
#empty { padding: 32px; text-align: center; background: white; }
[hidden] { display: none !important; }
.pagination { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0 0; }
.pagination button { grid-row: 2; }
#pageInfo { grid-column: 1 / -1; grid-row: 1; text-align: center; font-variant-numeric: tabular-nums; }
@container gallery (max-width: 700px) {
  .column-headings { grid-template-columns: minmax(0, 1fr) 60px 66px; gap: 8px; }
  .column-headings .result-heading { display: none; }
}
@container scene (max-width: 700px) {
  .scene-summary { grid-template-columns: minmax(0, 1fr) 60px 66px; gap: 8px; }
  .scene-summary .result { grid-column: 1 / -1; padding-right: 0; }
  .scene-details > summary { top: 18px; }
  .details-content { grid-template-columns: 1fr; gap: 12px; }
}
@media (max-width: 850px) {
  main, main.sidebar-collapsed { grid-template-columns: minmax(0, 1fr); grid-template-rows: auto minmax(0, 1fr); gap: 8px; padding: 8px; }
  .sidebar { max-height: 45vh; max-height: 45dvh; padding: 0; }
  .sidebar-toggle-bar { display: none; }
  .sidebar-controls > summary { display: list-item; cursor: pointer; margin: 10px 0; padding: 8px; border: 1px solid #b8c8d7; border-radius: 6px; background: white; }
}
@media (max-width: 480px) {
  #sceneList { gap: 8px; padding: 8px; }
  .column-headings, .scene-summary { padding-left: 12px; padding-right: 12px; }
  .scene-details > summary { right: 12px; }
}
</style></head><body><main id="galleryLayout">
<aside class="sidebar" aria-label="カタログの操作">
<div class="sidebar-toggle-bar"><button id="sidebarToggle" class="sidebar-toggle" type="button" aria-expanded="true" aria-controls="sidebarContent" aria-label="設定バーを折りたたむ" title="設定バーを折りたたむ"><span class="sidebar-toggle-icon" aria-hidden="true">‹</span><span class="sidebar-toggle-label">設定バーを折りたたむ</span></button></div>
<div id="sidebarContent">
<h1>シーンカタログ</h1>
<p class="intro">カーブ・交差点・追い越しの例を、同じ画面で絞り込んで比較できます。</p>
<section class="selection-actions" aria-label="学習・評価のシーン選択">
<div class="selection-counts"><span>学習 <strong id="selectionTrainCount">0</strong>件</span><span>評価 <strong id="selectionEvaluationCount">0</strong>件</span></div>
<button id="exportSelectionButton" type="button">選択を書き出す</button>
<button id="clearSelectionButton" type="button">選択をすべて解除</button>
<p class="hint">件数は全ページの選択。評価未選択のグループは学習と同じseedで評価します。</p>
<p id="selectionWarning" role="alert" hidden></p>
</section>
<details id="sidebarControls" class="sidebar-controls" open><summary>絞り込み・表示設定</summary>
<section class="toolbar" aria-label="カタログの絞り込み">
<div class="filters">
<label for="catalogFilter">カタログ<select id="catalogFilter"></select></label>
<label for="patternFilter">パターン<select id="patternFilter"></select></label>
<label for="outcomeFilter">終了理由<select id="outcomeFilter"></select></label>
<label for="searchInput">検索<input id="searchInput" type="search" placeholder="例：連続車線変更、random、seed 0"></label>
</div>
<div class="options">
<label for="sortSelect">並べ替え<select id="sortSelect"><option value="default">カタログ・パターン順</option><option value="seed">seedが小さい順</option><option value="steps">step数が少ない順</option><option value="overtaken">追い越し台数が多い順</option></select></label>
<label for="pageSizeSelect">1ページの件数<select id="pageSizeSelect"><option value="12">12件</option><option value="24">24件</option><option value="0">全件</option></select></label>
<label for="scaleSelect">画像の表示倍率<select id="scaleSelect"><option value="25">25%</option><option value="50">50%</option><option value="75" selected>75%</option><option value="100">100%（原寸）</option></select></label>
<label for="columnsSelect">一覧の列数<select id="columnsSelect"><option value="1">1列</option><option value="2">2列</option></select></label>
<p id="columnsHint" class="hint" hidden>狭い画面では1列で表示します（列数の選択は保持）。</p>
<button id="resetButton" type="button">絞り込みを解除</button>
</div>
<div class="status-row"><p id="status" role="status" aria-live="polite"></p><p class="hint">GIFは表示位置で自動再生。横長画像は画像内を左右にスクロールできます。</p></div>
</section>
<nav class="pagination" aria-label="シーンのページ"><button id="previousPage" type="button">前のページ</button><span id="pageInfo"></span><button id="nextPage" type="button">次のページ</button></nav>
</details>
<noscript><p>絞り込みと画像表示にはブラウザのJavaScriptを有効にしてください。</p></noscript>
</div>
</aside>
<div id="galleryScroll" class="gallery-scroll" data-columns="1">
<div class="column-headings" aria-hidden="true"><span>シーン</span><span>seed</span><span class="result-heading">走行結果</span><span>設定</span><span class="multi-column-heading">シーン一覧（2列）</span></div>
<div id="sceneList"></div><p id="empty" hidden>条件に一致するシーンがありません。絞り込みを解除してお試しください。</p>
</div>
<script id="gallery-data" type="application/json">__GALLERY_DATA__</script>
<script>
"use strict";
__EXPORT_SCRIPT__
const data = JSON.parse(document.getElementById("gallery-data").textContent);
const el = (id) => document.getElementById(id);
const selectionStorageKey = "scene_catalog_selection/1", selections = new Map();
let storageWarning = "";
try {
  const saved = JSON.parse(localStorage.getItem(selectionStorageKey) || "{}");
  data.scenes.forEach((s) => {
    const state = saved[s.selection_key];
    if (state && (state.train === true || state.evaluation === true)) {
      selections.set(s.selection_key, {train: state.train === true, evaluation: state.evaluation === true});
    }
  });
} catch (_) { storageWarning = "このブラウザでは選択の保存を利用できません。ページを閉じると選択は消えます。"; }
function exportMetadata() {
  const now = new Date().toISOString();
  const stamp = now.slice(0, 10).replace(/-/g, "") + "_" + now.slice(11, 16).replace(":", "");
  return {created_at: now, suggested_name: "scene_mix_" + stamp, instructions: data.selection_instructions};
}
function selectedScenes() {
  return data.scenes.filter((s) => selections.has(s.selection_key)).map((s) => ({...s, ...selections.get(s.selection_key)}));
}
function updateSelection() {
  const selected = selectedScenes();
  el("selectionTrainCount").textContent = selected.filter((s) => s.train).length;
  el("selectionEvaluationCount").textContent = selected.filter((s) => s.evaluation).length;
  el("exportSelectionButton").disabled = selected.length === 0;
  let warning = storageWarning;
  if (selected.length) {
    try { buildSelectionExport(selected, exportMetadata()); }
    catch (error) { warning = [warning, error.message].filter(Boolean).join("\n"); }
  }
  el("selectionWarning").textContent = warning; el("selectionWarning").hidden = !warning;
}
function saveSelection() {
  try { localStorage.setItem(selectionStorageKey, JSON.stringify(Object.fromEntries(selections))); }
  catch (_) { storageWarning = "選択を保存できません。ページを閉じると選択は消えます。"; }
  updateSelection();
}
el("clearSelectionButton").addEventListener("click", () => {
  selections.clear();
  document.querySelectorAll("input[data-selection-key]").forEach((input) => { input.checked = false; });
  saveSelection();
});
el("exportSelectionButton").addEventListener("click", () => {
  try {
    const text = buildSelectionExport(selectedScenes(), exportMetadata());
    const url = URL.createObjectURL(new Blob([text], {type: "application/toml;charset=utf-8"}));
    const link = document.createElement("a"); link.href = url; link.download = "scene_selection.toml";
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { el("selectionWarning").textContent = error.message; el("selectionWarning").hidden = false; }
});
const sidebarControls = el("sidebarControls");
const sidebarToggle = el("sidebarToggle"), sidebarContent = el("sidebarContent"), layout = el("galleryLayout");
const compactLayout = window.matchMedia("(max-width: 850px)");
let sidebarCollapsed = false;
function updateSidebarCollapse() {
  const collapsed = !compactLayout.matches && sidebarCollapsed;
  layout.classList.toggle("sidebar-collapsed", collapsed);
  sidebarContent.hidden = collapsed;
  sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
  const label = collapsed ? "設定バーを展開" : "設定バーを折りたたむ";
  sidebarToggle.title = label; sidebarToggle.setAttribute("aria-label", label);
  sidebarToggle.querySelector(".sidebar-toggle-icon").textContent = collapsed ? "›" : "‹";
  sidebarToggle.querySelector(".sidebar-toggle-label").textContent = collapsed ? "展開" : label;
}
sidebarToggle.addEventListener("click", () => { sidebarCollapsed = !sidebarCollapsed; updateSidebarCollapse(); });
function updateSidebar() { sidebarControls.open = !compactLayout.matches; updateSidebarCollapse(); }
compactLayout.addEventListener("change", updateSidebar);
updateSidebar();
const catalog = el("catalogFilter"), pattern = el("patternFilter"), outcome = el("outcomeFilter");
const search = el("searchInput"), sort = el("sortSelect"), size = el("pageSizeSelect"), scale = el("scaleSelect");
const list = el("sceneList"), scroll = el("galleryScroll");
const columns = el("columnsSelect");
function updateColumns() {
  scroll.dataset.columns = compactLayout.matches ? "1" : columns.value;
  el("columnsHint").hidden = !compactLayout.matches;
}
columns.addEventListener("change", updateColumns);
compactLayout.addEventListener("change", updateColumns);
updateColumns();
const outcomeLabels = {arrive_dest: "到着", timeout: "時間上限", crash_vehicle: "車両と衝突", crash_object: "物体と衝突", crash_building: "建物と衝突", crash_human: "歩行者と衝突", out_of_road: "道路逸脱"};
let page = 1;
function node(tag, text, className) {
  const n = document.createElement(tag);
  if (text !== undefined && text !== null) n.textContent = text;
  if (className) n.className = className;
  return n;
}
function option(select, value, label) { const n = node("option", label); n.value = value; select.append(n); }
option(catalog, "", "すべてのカタログ");
data.catalogs.forEach((c) => option(catalog, c.id, `${c.label}（${c.count}件）`));
option(outcome, "", "すべての終了理由");
[...new Set(data.scenes.map((s) => s.outcome))].forEach((key) => option(outcome, key, outcomeLabels[key] || key));
function updatePatterns() {
  const previous = pattern.value;
  pattern.replaceChildren(); option(pattern, "", "すべてのパターン");
  const names = new Set(data.scenes.filter((s) => !catalog.value || s.catalog === catalog.value).map((s) => s.variant));
  [...names].sort((a, b) => a.localeCompare(b, "ja", {numeric: true})).forEach((name) => option(pattern, name, name));
  pattern.value = names.has(previous) ? previous : "";
}
function anchor(url, text) {
  const a = node("a", text); a.href = url; a.target = "_blank"; a.rel = "noopener"; return a;
}
const observer = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
  entries.forEach(({target, isIntersecting}) => {
    if (isIntersecting && target.dataset.src) {
      target.src = target.dataset.src; delete target.dataset.src; observer.unobserve(target);
    }
  });
}, {root: scroll, rootMargin: "200px 0px"}) : null;
function card(s, index) {
  const article = node("article", null, "scene-card");
  const header = node("header", null, "scene-summary");
  const identity = node("div"); identity.append(node("span", s.catalog_label, "catalog-label"), node("strong", s.variant, "variant"));
  const choices = node("div", null, "scene-selection");
  [["train", "学習"], ["evaluation", "評価"]].forEach(([stage, title]) => {
    const label = node("label"), input = node("input"); input.type = "checkbox";
    input.dataset.stage = stage; input.dataset.selectionKey = s.selection_key;
    input.checked = !!selections.get(s.selection_key)?.[stage];
    input.setAttribute("aria-label", `${s.catalog_label} ${s.variant} seed ${s.seed} を${title}に使う`);
    input.addEventListener("change", () => {
      const state = {...selections.get(s.selection_key), [stage]: input.checked};
      if (state.train || state.evaluation) selections.set(s.selection_key, state); else selections.delete(s.selection_key);
      saveSelection();
    });
    label.append(input, node("span", title)); choices.append(label);
  });
  identity.append(choices);
  const result = node("div", null, "result");
  const state = ["arrive_dest", "timeout"].includes(s.outcome) ? s.outcome : "problem";
  result.append(node("span", outcomeLabels[s.outcome] || s.outcome, `outcome ${state}`));
  result.append(node("span", `追い越し ${s.overtaken_count === null ? "—" : s.overtaken_count + "台"}`));
  result.append(node("span", `車線変更 ${s.lane_changes}回`), node("span", `${s.steps} step`));
  header.append(identity, node("span", `seed ${s.seed}`, "seed"), result);
  const details = node("details", null, "scene-details");
  const summary = node("summary", "詳細"); summary.setAttribute("aria-label", `${s.variant} seed ${s.seed} の設定と経路`);
  const content = node("div", null, "details-content"), metrics = node("dl");
  [["経路長", s.route_m + " m"], ["正味の旋回角", s.turn_deg + "°"], ["カーブ", s.curves || "—"], ["先行車", s.lead_vehicles === null ? "—" : s.lead_vehicles], ["終了理由", s.outcome]].forEach(([title, value]) => metrics.append(node("dt", title), node("dd", value)));
  const config = node("div"); config.append(node("pre", s.settings));
  const links = node("div", null, "detail-links");
  links.append(anchor(s.toml, "このシーンの設定TOML"), anchor(s.catalog + "/index.html", "元カタログ・共通設定"));
  config.append(links); content.append(metrics, config); details.append(summary, content);
  const bar = node("div", null, "preview-bar");
  const animated = s.image.toLowerCase().endsWith(".gif");
  bar.append(node("span", animated ? "GIF · 表示中に自動再生" : "PNG · 静止画"), node("span", "↔ 横長画像は左右にスクロール"), anchor(s.link, "原寸で開く ↗"));
  const viewport = node("div", null, "preview-scroll"); viewport.tabIndex = 0;
  viewport.setAttribute("aria-label", `${s.variant} seed ${s.seed} の画像（左右にスクロール）`);
  const link = anchor(s.link), image = node("img");
  image.alt = `${s.catalog_label} / ${s.variant} / seed ${s.seed}`;
  image.width = Math.round(s.width * Number(scale.value) / 75);
  image.loading = "lazy"; image.decoding = "async"; image.dataset.src = s.image;
  link.append(image); viewport.append(link); article.append(header, details, bar, viewport);
  return article;
}
function filteredScenes() {
  const input = search.value.toLocaleLowerCase().trim();
  const seedQuery = input.match(/\bseed\s*[:=]?\s*(\d+)\b/);
  const words = input.replace(/\bseed\s*[:=]?\s*\d+\b/g, " ").split(/\s+/).filter(Boolean);
  const rows = data.scenes.filter((s) => {
    if (catalog.value && s.catalog !== catalog.value || pattern.value && s.variant !== pattern.value || outcome.value && s.outcome !== outcome.value) return false;
    if (seedQuery && Number(s.seed) !== Number(seedQuery[1])) return false;
    const text = `${data.search_fields.map((key) => s[key]).join(" ")} seed ${s.seed} ${outcomeLabels[s.outcome] || ""}`.toLocaleLowerCase();
    return words.every((word) => text.includes(word));
  });
  if (sort.value !== "default") {
    const key = {seed: "seed", steps: "steps", overtaken: "overtaken_count"}[sort.value];
    const number = (s) => s[key] === null ? -Infinity : Number(s[key]);
    rows.sort((a, b) => sort.value === "overtaken" ? number(b) - number(a) : number(a) - number(b));
  }
  return rows;
}
function render(resetScroll = true) {
  if (observer) observer.disconnect();
  list.querySelectorAll("img").forEach((image) => image.removeAttribute("src"));
  list.replaceChildren();
  const rows = filteredScenes(), perPage = Number(size.value) || Math.max(rows.length, 1);
  const pages = Math.max(1, Math.ceil(rows.length / perPage)); page = Math.min(page, pages);
  const start = (page - 1) * perPage, visible = rows.slice(start, start + perPage);
  visible.forEach((s, index) => list.append(card(s, index)));
  list.querySelectorAll("img").forEach((image) => {
    if (observer) observer.observe(image); else { image.src = image.dataset.src; delete image.dataset.src; }
  });
  el("empty").hidden = rows.length !== 0;
  el("status").textContent = `${rows.length ? start + 1 : 0}–${start + visible.length}件を表示 / 該当 ${rows.length}件 / 全 ${data.scenes.length}件（${data.catalogs.length}カタログ）`;
  el("pageInfo").textContent = `${page} / ${pages} ページ`;
  el("previousPage").disabled = page === 1; el("nextPage").disabled = page === pages;
  if (resetScroll) scroll.scrollTop = 0;
}
catalog.addEventListener("change", () => { updatePatterns(); page = 1; render(); });
[pattern, outcome, sort, size].forEach((control) => control.addEventListener("change", () => { page = 1; render(); }));
search.addEventListener("input", () => { page = 1; render(); });
scale.addEventListener("change", () => render(false));
el("previousPage").addEventListener("click", () => { page--; render(); });
el("nextPage").addEventListener("click", () => { page++; render(); });
el("resetButton").addEventListener("click", () => {
  catalog.value = ""; outcome.value = ""; search.value = ""; sort.value = "default"; size.value = "12";
  updatePatterns(); pattern.value = ""; page = 1; render();
});
updatePatterns(); render(); updateSelection();
</script></main></body></html>
"""


def build_gallery(root: Path | None = None, *, output_dir: Path | None = None) -> Path:
    """repo rootの正式設定に対応する既存indexをcatalog出力rootへ統合する。

    rootの既定はこのモジュールを置くtools/の親（リポジトリ直下）。
    output_dirは各カタログディレクトリを含む親（既定root/outputs/scene_catalog）。
    未生成カタログは飛ばし、各indexの表示行だけを採用する。
    """

    root = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir) if output_dir is not None else root / "outputs" / "scene_catalog"
    catalogs, scenes = [], []
    for name, label in CATALOGS.items():
        index = output_dir / name / "index.html"
        if not (root / "configs" / "scene_catalog" / f"{name}.toml").is_file() or not index.is_file():
            continue
        rows = _read_scenes(index, name)
        catalogs.append({"id": name, "label": label, "count": len(rows)})
        scenes.extend(rows)
    _selection_names(scenes)
    data = json.dumps({"catalogs": catalogs, "scenes": scenes,
                       "search_fields": ["catalog", "catalog_label", "image", "link", "toml", "width", *FIELDS.values()],
                       "switch_keys": sorted(SCENE_GROUP_SWITCH_KEYS),
                       "selection_instructions": SELECTION_INSTRUCTIONS}, ensure_ascii=False)
    # application/json内でもHTMLのscript終了タグとして解釈させない。
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "index.html"
    path.write_text(PAGE.replace("__EXPORT_SCRIPT__", EXPORT_SCRIPT).replace("__GALLERY_DATA__", data), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="リポジトリ直下（既定: tools/の親）")
    parser.add_argument("--output-dir", type=Path, help="各カタログを含む出力root（既定: root/outputs/scene_catalog）")
    args = parser.parse_args()
    print(f"gallery_saved={build_gallery(args.root, output_dir=args.output_dir)}")


if __name__ == "__main__":
    main()
