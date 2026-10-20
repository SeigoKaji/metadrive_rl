"""Combine existing catalog pages without importing or running MetaDrive."""

import html
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote, unquote

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
