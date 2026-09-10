# 使い方

コマンドはリポジトリルートから実行し、設定の相対pathはそのconfigを基準に解決します。既存のMetaDrive／SB3依存をreport専用環境へ一括upgradeしません。合成demoと公式runの出力先を分けてください。

## 設定を確認する

schemaだけを依存なしで確認できます。

```bash
python3 -c 'from input_attribution.schema import standard_259_schema, lidar_partition; s=standard_259_schema(); p=lidar_partition(); print(len(s), {k: len(v) for k, v in p.items()})'
```

期待値は `259` とLiDAR各群 `60` です。設定ファイルは次の5種類を用途で選びます。

| 用途 | 設定 |
|---|---|
| 公式N=10、fixed=-1、GIF有効 | `input_attribution/configs/official_fixed_stress.toml` |
| 公式N=2スモーク、GIF有効、horizon=500 | `input_attribution/configs/official_smoke.toml` |
| 合成N=2、259幅、baseline127／affected50step | `input_attribution/configs/synthetic_demo.toml` |
| 262幅の移植チェック | `input_attribution/configs/host_262_migration_template.toml` |
| T13人工262幅の合成契約 | `input_attribution/configs/synthetic_262_demo.toml` |

公式設定の `model_path` は `models/official_baseline.zip`、環境は既存公式 `map=C`、3×3離散action（9 action）、scenario seed 5に合わせています。実行前にmodelの実在、modelの観測shape、実環境の観測shape、schemaをcheckで照合します。モデルのない環境で公式runを開始せず、合成設定を使います。

## check / run

実装済みCLIの次の順序で実行します。`check` はmodel・環境・schema・前処理・固定値到達・描画可否・報酬接続を確認し、`run` はbaseline保存後に①-A／①-Bと図・HTMLを生成します。

```bash
python3 -m pytest input_attribution/tests/test_schema.py input_attribution/tests/test_core.py input_attribution/tests/test_adapter.py input_attribution/tests/test_reporting.py input_attribution/tests/test_acceptance.py -q
python3 -m input_attribution check --config input_attribution/configs/synthetic_demo.toml
python3 -m input_attribution run --config input_attribution/configs/synthetic_demo.toml
```

公式の2groupスモークは次です。

```bash
python3 -m input_attribution check --config input_attribution/configs/official_smoke.toml
python3 -m input_attribution run --config input_attribution/configs/official_smoke.toml
```

N=10の公式runはmodelとMetaDrive assetが揃ってから実行します。

```bash
python3 -m input_attribution run --config input_attribution/configs/official_fixed_stress.toml
```

各patternは元観測の独立copyへ適用します。再集計だけを行うときは、保存済みrunのパスを次の `report` へ渡します。

## report

reportは保存データだけを読み、環境、model、reward関数、推論を呼びません。runが作ったタイムスタンプ付きディレクトリを `--run-dir` へ渡してください。

```bash
python3 -m input_attribution report --run-dir outputs/input_attribution/demo/synthetic/run-<timestamp>-<pid>
```

出力は指定したrun directory直下の `report.html`、6列summary、reward／JS PNG、利用可能なGIFへの参照です。別場所へrun directoryをコピーする場合、`data/` と `data/frames/` を一緒にコピーしてください。未比較step、自然終了後step、未接続reward termsを0で埋めません。

## fixed valueと単独index

`fixed_value = -1.0` が既定で、`-100.0` も有限値として指定できます。schemaの正常範囲外だからといってclip、拒否、修復しません。NaN／Inf、重複index、範囲外index、shape不一致は停止します。10groupを使わず単独indexを試す場合は、schemaを確認したうえでconfigのpatternを1つにして、`id`、`name`、`indices`、`fixed_value` を明示してください。

262 hostは `input_schema_262_template.toml` が全262行unknownのため、そのまま確認済みとは扱えません。移植先の実観測生成、前処理、model入口、参照レーン、有効フラグを確認し、明示schemaとpatternを用意してから実行します。T13の `synthetic_262_demo.toml` は人工schemaであり、実262環境の証拠ではありません。
