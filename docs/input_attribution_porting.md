# 入力寄与解析を別 PC・262 次元環境へ移植する手順

この手順は MetaDrive 本体を patch せず、GitHub Copilot に大きな移植を任せずに済むよう、変更箇所を schema と path に限定します。解析の目的と保存済み結果を先に把握する場合は [入力寄与解析の概要](input_attribution_overview.md) を参照してください。

実験の方法と設定項目は [入力寄与解析ガイド](input_attribution.md)、出力の確認方法は [結果の読み方](input_attribution_results.md)、schema の成立条件は [観測スキーマ](../observation_schemas/README.md) を参照してください。この文書では移植時に固有の変更だけを扱います。

## 3実験の保存先

移植後も実験番号と directory 名を変えません。通常走行の収集は実験00ではなく共通準備です。

| 所属 | 保存先 | 内容 |
| --- | --- | --- |
| 準備：通常走行の収集 | `shared/` | 共通 metadata、schema、rollout。collect はここを作り、3実験は扱わない |
| 実験01 入力置換（オフライン摂動） | `experiment_01_perturbation/` | JSDによる入力置換の結果と候補選択 |
| 実験02 Integrated Gradients（出力変化の入力への割当） | `experiment_02_integrated_gradients/` | 指定 output の入力別割当。実験03の候補選択には使わない |
| 実験03 入力固定での走行比較（paired closed-loop） | `experiment_03_closed_loop/` | 通常/介入の総報酬・完走・道路外への逸脱 |

compact は root `report.md`、3つの実験 `report.md` + `details.zip`、`shared/details.zip` を作ります。ZIP member path は run root 相対で、4 ZIP は同じ新しい run root へ展開します。full は同じ directory 構成で個別 artifact を保存し、元の詳細 report は `shared/report.md` として保全します。

## コピーするもの

コピー先プロジェクトへ、次をそのままコピーします。

- `analyze_input_attribution.py`
- `input_attribution/`
- `attribution_configs/`
- `observation_schemas/`
- `docs/input_attribution_overview.md`
- `docs/input_attribution.md`
- `docs/input_attribution_results.md`
- `docs/input_attribution_porting.md`
- `tests/test_attribution_*.py` と `tests/test_analyze_input_attribution_cli.py`

コピー先にも既存の `env_factory.py`、`evaluation_visualization.py`、`configs/experiment_config.py`、`project_paths.py` が必要です。SB3 / NumPy / pandas / matplotlib / PyTorch は既存 `requirements.txt` の範囲を使い、新たな重量級依存は不要です。

## 262 次元版の確定手順

1. 262 次元モデルと、その model を評価できる experiment TOML の正確な path を決めます。
2. `observation_schemas/custom_262_template.toml` を `observation_schemas/custom_262.toml` としてコピーします。
3. custom environment の observation source を確認し、追加 3 入力の**正確な** index、名前、意味、group、description を確認します。意味が未確定ならこの段階で止めます。
4. `custom_unresolved` block の `start`、`feature_names`、`description`、`group`、`resolved = true` を編集します。必要な場合だけ閉ループ `schema_constant` 用の `constants` も明示します。
5. 3 入力が末尾追加なら custom block は `start = 259`、LiDAR range は `start = 19` のままです。LiDAR の直前に挿入されるなら custom block を `start = 19`、LiDAR range を `start = 22` にします。Python source は編集しません。
6. `observation_dim = 262` と、全 block/range が 0--261 を重複・欠落なく覆うことを確認します。
7. 実環境と model を照合します。

```bash
.venv/bin/python analyze_input_attribution.py validate-schema \
  --config configs/custom_262.toml \
  --model models/custom_262_model.zip \
  --schema observation_schemas/custom_262.toml
```

8. 検証が成功してから run を行います。

```bash
.venv/bin/python analyze_input_attribution.py run \
  --config configs/custom_262.toml \
  --model models/custom_262_model.zip \
  --schema observation_schemas/custom_262.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --output-prefix custom_262_attribution_numbered
```

## 出力の確認順

通常は root `report.md` の実験対応表と実験03の走行比較表を読みます。実験01/02の主要表は各実験 `report.md` にあります。compact の詳細ファイルは4つの `details.zip` にあり、復元時は各 ZIP を同じ新しい run root へ展開します。移植後の検証では `shared/details.zip` の復元先 root を `analyze --rollout` に指定し、`analysis_metadata.json` の dimension と hash、および `feature_schema_expanded.csv` の 262 行が確認済みの index・意味と一致することを確認してください。詳細を個別ファイルで保存したい場合は `run` に `--output-mode full` を付けます。CSV の絞り方や各指標の符号は [入力寄与解析結果の読み方](input_attribution_results.md) にまとめています。

`custom_262.toml` の unresolved placeholder を残したままの実行や、追加 3 入力に推測した仮名を付けた結果の解釈は行いません。移植先で変えるべきファイルは原則として experiment TOML の path、`custom_262.toml`、必要なら analysis TOML の run / baseline / closed-loop key だけです。
