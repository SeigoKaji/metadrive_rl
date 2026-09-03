# 入力寄与解析を別 PC・262 次元環境へ移植する手順

この手順は MetaDrive 本体を patch せず、GitHub Copilot に大きな移植を任せずに済むよう、変更箇所を schema と path に限定します。

実験の方法と設定項目は [入力寄与解析ガイド](input_attribution.md)、出力の確認方法は [結果の読み方](input_attribution_results.md)、schema の成立条件は [観測スキーマ](../observation_schemas/README.md) を参照してください。この文書では移植時に固有の変更だけを扱います。

## コピーするもの

コピー先プロジェクトへ、次をそのままコピーします。

- `analyze_input_attribution.py`
- `input_attribution/`
- `attribution_configs/`
- `observation_schemas/`
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
  --output-prefix custom_262_attribution
```

## 出力の確認順

確認順、CSV の絞り方、各指標の符号は [入力寄与解析結果の読み方](input_attribution_results.md) に集約しています。移植後は、特に `analysis_metadata.json` の dimension と hash、および `feature_schema_expanded.csv` の 262 行が確認済みの index・意味と一致することを先に確認してください。

`custom_262.toml` の unresolved placeholder を残したままの実行や、追加 3 入力に推測した仮名を付けた結果の解釈は行いません。移植先で変えるべきファイルは原則として experiment TOML の path、`custom_262.toml`、必要なら analysis TOML の run / baseline / closed-loop key だけです。
