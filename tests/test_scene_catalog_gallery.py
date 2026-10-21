"""Combine existing catalog pages without importing or running MetaDrive."""

import html
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from urllib.parse import quote, unquote

import pytest

from configs.experiment_config import SCENE_GROUP_SWITCH_KEYS
from tools import scene_catalog_gallery
from tools.scene_catalog_gallery import build_gallery


class _GalleryData(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reading = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "gallery-data":
            self.reading = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.reading = False

    def handle_data(self, data):
        if self.reading:
            self.parts.append(data)


def _read_gallery(path):
    parser = _GalleryData()
    parser.feed(path.read_text(encoding="utf-8"))
    return json.loads("".join(parser.parts))


def _source(root, output_dir, catalog, variant, *, modern=False, gif=False):
    config = root / "configs" / "scene_catalog" / f"{catalog}.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("[settings]\n[base]\ntraffic_density = 0\n", encoding="utf-8")
    directory = output_dir / catalog
    asset = f"{variant}/seed_0003.{'gif' if gif else 'png'}"
    params = f"{variant}/seed_0003.toml"
    for name in [asset, params]:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    (directory / params).write_text(
        '[environment.common]\nmap = "C"\nhorizon = 1000\nrandom_traffic = false\nlog_level = 50\n',
        encoding="utf-8")
    settings = '注記 = "</script><script>alert(1)</script> & 雪"'
    headers = ["画像（クリックで原寸）" if modern else "画像（クリックで原寸/GIF）",
               "パターン", "パターン固有の設定", "seed", "経路長 [m]",
               "正味の旋回角 [deg]", "カーブ 半径[m]:角度[deg]"]
    cells = [f'<a href="{html.escape(asset)}"><img src="{html.escape(asset)}" '
             'width="750" loading="lazy"></a>', html.escape(variant),
             f'<span class=code>{html.escape(settings)}</span>', "3", "272", "-90", "R38:+90"]
    if modern:
        headers += ["先行車", "追い越した台数"]
        cells += [html.escape("2台: 30m & 80m / 低速<12>"), "2"]
    headers += ["step数", "車線変更", "終了理由", "設定ファイル"]
    cells += ["334", "2", "arrive_dest", f'<a href="{html.escape(params)}">seed_0003.toml</a>']
    (directory / "index.html").write_text(
        '<!DOCTYPE html><html lang="ja"><meta charset="utf-8"><body>'
        '<h2>全パターン共通の設定</h2><table><tr><th>traffic_density</th><td>0</td></tr></table>'
        '<table id=scenes><thead><tr>' + "".join(f"<th>{h}</th>" for h in headers)
        + "</tr></thead><tbody><tr>" + "".join(f"<td>{c}</td>" for c in cells)
        + "</tr></tbody></table></body></html>", encoding="utf-8")
    return asset, params, settings


def test_combines_legacy_and_current_rows_with_safe_unicode_and_relative_links(tmp_path):
    output_dir = tmp_path / "outputs" / "scene_catalog"
    legacy = _source(tmp_path, output_dir, "curve", '曲線 & <雪>"')
    current = _source(tmp_path, output_dir, "overtake_slow_lead", "追越 & 雪", modern=True, gif=True)

    path = build_gallery(tmp_path)
    assert path == output_dir / "index.html"
    payload = _read_gallery(path)
    scenes = {row["catalog"]: row for row in payload["scenes"]}
    assert set(scenes) == {"curve", "overtake_slow_lead"}
    for catalog, (asset, params, settings) in [("curve", legacy), ("overtake_slow_lead", current)]:
        scene = scenes[catalog]
        assert scene["variant"] == asset.split("/")[0]
        assert settings in scene["settings"]
        assert scene["seed"] == "3"
        assert scene["route_m"] == "272"
        assert scene["turn_deg"] == "-90"
        assert scene["curves"] == "R38:+90"
        assert scene["steps"] == "334"
        assert scene["lane_changes"] == "2"
        assert scene["outcome"] == "arrive_dest"
        assert scene["width"] == 750
        assert scene["image"] == scene["link"] == quote(f"{catalog}/{asset}", safe="/")
        assert scene["toml"] == quote(f"{catalog}/{params}", safe="/")
        for key in ["image", "link", "toml"]:
            assert (path.parent / unquote(scene[key])).is_file()
    assert scenes["curve"]["lead_vehicles"] is None
    assert scenes["curve"]["overtaken_count"] is None
    assert scenes["overtake_slow_lead"]["lead_vehicles"] == "2台: 30m & 80m / 低速<12>"
    assert scenes["overtake_slow_lead"]["overtaken_count"] == "2"


def test_uses_only_selected_rows_from_official_configured_catalogs(tmp_path):
    output_dir = tmp_path / "export with spaces"
    _source(tmp_path, output_dir, "overtake", "selected", modern=True)
    # Old images left by an earlier full run must not undo a --variant selection.
    stale = output_dir / "overtake" / "unselected"
    stale.mkdir()
    (stale / "seed_0000.png").write_bytes(b"old")
    (stale / "seed_0000.toml").write_text("[environment.common]\n", encoding="utf-8")
    _source(tmp_path, output_dir, "smoke", "unofficial", modern=True)
    _source(tmp_path, output_dir, "curve", "missing_config")
    (tmp_path / "configs" / "scene_catalog" / "curve.toml").unlink()
    (tmp_path / "configs" / "scene_catalog" / "intersection.toml").write_text("[base]\n")

    path = build_gallery(tmp_path, output_dir=output_dir)
    payload = _read_gallery(path)
    assert path == output_dir / "index.html"
    assert [(row["catalog"], row["variant"]) for row in payload["scenes"]] == [("overtake", "selected")]
    assert [(catalog["id"], catalog["count"]) for catalog in payload["catalogs"]] == [("overtake", 1)]


def test_missing_catalogs_produce_an_empty_gallery(tmp_path):
    path = build_gallery(tmp_path)
    payload = _read_gallery(path)
    assert payload["catalogs"] == []
    assert payload["scenes"] == []


def test_default_root_is_the_parent_of_tools(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs" / "scene_catalog"
    _source(tmp_path, output_dir, "curve", "default_root")
    monkeypatch.setattr(scene_catalog_gallery, "__file__", str(tmp_path / "tools" / "scene_catalog_gallery.py"))

    path = build_gallery()
    assert path == output_dir / "index.html"
    assert [row["variant"] for row in _read_gallery(path)["scenes"]] == ["default_root"]


def test_gallery_import_does_not_load_the_simulator():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import tools.scene_catalog_gallery; "
         "assert 'tools.scene_catalog' not in sys.modules; "
         "assert not {'metadrive', 'numpy', 'torch', 'PIL'}.intersection(sys.modules)"],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _export_scene(tmp_path, catalog, variant, seed, common='map = "C"\nhorizon = 1000\n'):
    relative = f"{catalog}/{variant}/seed_{seed:04d}.toml"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[environment.common]\n" + common, encoding="utf-8")
    scene = {"catalog": catalog, "variant": variant, "seed": str(seed), "toml": quote(relative, safe="/")}
    scene_catalog_gallery._selection_data(scene, tmp_path)
    return scene


def _node_export(scenes):
    scene_catalog_gallery._selection_names(scenes)
    metadata = {"created_at": "2026-10-21T12:34:56.000Z", "suggested_name": "scene_mix_20261021_1234",
                "instructions": scene_catalog_gallery.SELECTION_INSTRUCTIONS}
    script = scene_catalog_gallery.EXPORT_SCRIPT + r"""
const fs = require("node:fs"), input = JSON.parse(fs.readFileSync(0, "utf8"));
const before = JSON.stringify(input);
let result;
try { result = {text: buildSelectionExport(input.scenes, input.metadata)}; }
catch (error) { result = {error: error.message}; }
result.mutated = before !== JSON.stringify(input);
process.stdout.write(JSON.stringify(result));
"""
    node = shutil.which("node")
    assert node, "書き出し関数の検証には既存Nodeが必要です"
    result = subprocess.run([node, "-e", script], input=json.dumps({"scenes": scenes, "metadata": metadata}),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert not payload.pop("mutated"), "書き出し関数は入力を変更しない"
    return payload


def test_embeds_shared_switch_keys_and_nested_toml_without_static_settings(tmp_path):
    output_dir = tmp_path / "outputs" / "scene_catalog"
    _, params, _ = _source(tmp_path, output_dir, "overtake_slow_lead", '雪 "引用"', modern=True)
    common = ('map_config = { type = "block_sequence", config = "SSS", lane_num = 2 }\n'
              'lead_vehicles = { count = [1, 2], lane = "ego", extra = { "引用 key" = [true, 1.5, "雪\\n行"] } }\n'
              'horizon = 1000\nrandom_traffic = false\nlog_level = 50\n')
    (output_dir / "overtake_slow_lead" / params).write_text("[environment.common]\n" + common, encoding="utf-8")
    path = build_gallery(tmp_path)
    payload = _read_gallery(path)
    scene = payload["scenes"][0]
    assert payload["switch_keys"] == sorted(SCENE_GROUP_SWITCH_KEYS)
    assert scene["selection_error"] is None
    expected = tomllib.loads(common)
    expected.pop("random_traffic"); expected.pop("log_level")
    assert scene["switch_settings"] == expected
    assert tomllib.loads(scene["common_toml"]) == expected
    assert scene["seed_literal"] == "3"
    exported = _node_export([{**scene, "train": True}])["text"]
    parsed = tomllib.loads(exported)
    assert parsed["environment"]["scene_groups"][0]["name"] == 'overtake_slow_lead__雪 "引用"'
    instructions = payload["selection_instructions"]
    assert all(key in instructions for key in SCENE_GROUP_SWITCH_KEYS)
    assert "environment.common / environment.train / environment.evaluation" in instructions
    assert "既存 environment.scene_groups は残さず" in instructions
    assert "[selection] はメタ情報" in instructions
    page = path.read_text(encoding="utf-8")
    assert "__EXPORT_SCRIPT__" not in page and "<script>alert(1)</script>" not in page


def test_node_export_preserves_group_seed_order_and_explicit_evaluation_fallback(tmp_path):
    scenes = [
        {**_export_scene(tmp_path, "curve", "C", 5), "train": True},
        {**_export_scene(tmp_path, "curve", "C", 1), "train": True, "evaluation": True},
        {**_export_scene(tmp_path, "overtake_slow_lead", "lane2_one", 5), "train": True},
        {**_export_scene(tmp_path, "curve", "S", 3), "train": True},
        {**_export_scene(tmp_path, "overtake_slow_lead", "lane2_one", 7), "evaluation": True},
    ]
    text = _node_export(scenes)["text"]
    groups = tomllib.loads(text)["environment"]["scene_groups"]
    assert [(g["name"], g["train_seeds"], g["evaluation_seeds"]) for g in groups] == [
        ("curve__C", [5, 1], [1]), ("overtake_slow_lead__lane2_one", [5], [7]), ("curve__S", [3], [3])]
    assert "評価未選択のため、学習と同じseedで評価する。" in text
    assert tomllib.loads(text)["selection"]["schema"] == "scene_selection/1"


def test_canonical_groups_are_type_strict_order_independent_and_names_never_collide(tmp_path):
    scenes = [
        _export_scene(tmp_path, "curve", "C", 0, 'map = "C"\nhorizon = 1000\n'),
        _export_scene(tmp_path, "curve", "C", 1, 'horizon = 1000\nmap = "C"\n'),
        _export_scene(tmp_path, "curve", "C", 2, 'map = "C"\nhorizon = 1000.0\n'),
        _export_scene(tmp_path, "curve", "C_2", 0),
    ]
    result = _node_export([{**scene, "train": True} for scene in scenes])
    groups = tomllib.loads(result["text"])["environment"]["scene_groups"]
    assert [g["name"] for g in groups] == ["curve__C", "curve__C_3", "curve__C_2"]
    assert [g["train_seeds"] for g in groups] == [[0, 1], [2], [0]]
    assert isinstance(groups[0]["common"]["horizon"], int)
    assert isinstance(groups[1]["common"]["horizon"], float)


@pytest.mark.parametrize("invalid", ["evaluation_only", "random_traffic", "malformed", "missing", "non_finite"])
def test_invalid_selected_scene_aborts_the_whole_export(tmp_path, invalid):
    good = {**_export_scene(tmp_path, "curve", "C", 3), "train": True}
    bad = _export_scene(tmp_path, "intersection", "cross", 3,
                        'map = "X"\nrandom_traffic = ' + ("true" if invalid == "random_traffic" else "false") + "\n")
    if invalid in {"malformed", "missing", "non_finite"}:
        path = tmp_path / unquote(bad["toml"])
        if invalid == "malformed": path.write_text("[broken", encoding="utf-8")
        elif invalid == "non_finite": path.write_text('[environment.common]\nmap = "X"\ntraffic_density = nan\n', encoding="utf-8")
        else: path.unlink()
        scene_catalog_gallery._selection_data(bad, tmp_path)
    bad["evaluation" if invalid == "evaluation_only" else "train"] = True
    result = _node_export([good, bad])
    assert "text" not in result
    expected = {"evaluation_only": "評価だけ", "random_traffic": "random_traffic = true",
                "malformed": "設定TOML", "missing": "設定TOML", "non_finite": "設定TOML"}[invalid]
    assert expected in result["error"]


def test_bad_toml_keeps_gallery_browsable_and_config_identity_expires_saved_choices(tmp_path):
    output_dir = tmp_path / "outputs" / "scene_catalog"
    _, params, _ = _source(tmp_path, output_dir, "curve", "C")
    path = output_dir / "curve" / params
    before = _read_gallery(build_gallery(tmp_path))["scenes"][0]
    path.write_text('# 同じ設定\n[environment.common]\nlog_level = 50\nrandom_traffic = false\nhorizon = 1000\nmap = "C"\n', encoding="utf-8")
    reordered = _read_gallery(build_gallery(tmp_path))["scenes"][0]
    assert reordered["selection_key"] == before["selection_key"]
    path.write_text(path.read_text(encoding="utf-8").replace("log_level = 50", "log_level = 40"), encoding="utf-8")
    changed = _read_gallery(build_gallery(tmp_path))["scenes"][0]
    assert changed["selection_key"] != before["selection_key"]
    assert changed["group_key"] == before["group_key"]
    path.write_text("broken", encoding="utf-8")
    broken = _read_gallery(build_gallery(tmp_path))["scenes"][0]
    assert broken["image"] == before["image"]
    assert "設定TOML" in broken["selection_error"]


def test_search_ignores_export_settings_instead_of_matching_every_random_spawn_lane(tmp_path):
    output_dir = tmp_path / "outputs" / "scene_catalog"
    for catalog, variant in [("overtake_slow_lead", "lane2_one"), ("overtake", "random_lane")]:
        _, params, _ = _source(tmp_path, output_dir, catalog, variant, modern=True)
        (output_dir / catalog / params).write_text(
            '[environment.common]\nmap = "S"\nrandom_spawn_lane_index = false\n', encoding="utf-8")
    payload = _read_gallery(build_gallery(tmp_path))
    function = "function filteredScenes()" + scene_catalog_gallery.PAGE.split("function filteredScenes()", 1)[1].split("function render(", 1)[0]
    script = ('const data = JSON.parse(require("node:fs").readFileSync(0, "utf8")); '
              'const catalog = {value:""}, pattern = {value:""}, outcome = {value:""}, '
              'search = {value:"random"}, sort = {value:"default"}, outcomeLabels = {};\n'
              + function + '\nprocess.stdout.write(JSON.stringify(filteredScenes().map(s => s.variant)));')
    result = subprocess.run([shutil.which("node"), "-e", script], input=json.dumps(payload), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ["random_lane"]


@pytest.mark.parametrize("module", [False, True])
def test_gallery_cli_supports_direct_and_module_execution(tmp_path, module):
    output_dir = tmp_path / "outputs" / "scene_catalog"
    _source(tmp_path, output_dir, "curve", "C")
    repository = Path(__file__).resolve().parents[1]
    entry = ["-m", "tools.scene_catalog_gallery"] if module else [str(repository / "tools" / "scene_catalog_gallery.py")]
    result = subprocess.run([sys.executable, *entry, "--root", str(tmp_path)],
                            cwd=repository if module else tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(_read_gallery(output_dir / "index.html")["scenes"]) == 1
