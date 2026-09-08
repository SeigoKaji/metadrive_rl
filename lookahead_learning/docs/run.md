# 実行手順

この追加モジュールは、ホストリポジトリのルートで次の形で起動します。

```text
python -B -m lookahead_learning <command>
```

実装の入口は [`../runner.py`](../runner.py) です。観測・行動・ホストの読み込み契約は [`../adapter.py`](../adapter.py)、前方注視とPP参照の処理は [`../env.py`](../env.py) と [`../geometry.py`](../geometry.py)、移設チェックは [`../portability.py`](../portability.py) にあります。実験の意味と指標は [`methods.md`](methods.md)、文書一覧は [`README.md`](README.md) を参照してください。

## Pythonと実行場所

コマンドは、`env_factory.py` と `configs/` があるホストリポジトリのルートから実行します。現在のcheckoutでの具体的な実行場所とPythonは次のとおりです。venvの相対パスはcheckout rootを基準にしています。別PCや移植先では、`cd`のパスをそのホストrootへ置き換え、選択した環境のPythonを指定してください。

```bash
cd /home/kajiseigo/workspace/metadrive_rl/metadrive_rl-lookahead
LOOKAHEAD_PY="../metadrive_rl-main/.venv/bin/python"
"$LOOKAHEAD_PY" -B -m lookahead_learning --help
```

別のPCでは、そのPCで選んだ同じ依存環境のPythonを使います。たとえばPATH上の環境を有効にしている場合は、`python`（または環境が提供する`python3`）に置き換えます。

```bash
cd /path/to/host-root
python -B -m lookahead_learning --help
```

このモジュールは依存ライブラリをインストールせず、`PYTHONPATH`を永続変更しません。監査済み環境の目安は次のとおりです。

| 項目 | 監査値 |
| --- | --- |
| Python | 3.12.3 |
| MetaDrive | 0.4.3、source commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` |
| Stable-Baselines3 | 2.9.0 |
| Gymnasium | 1.3.0 |
| Panda3D | 1.10.16 |
| PyTorch | 2.13.0 |
| NumPy | 2.5.2 |
| MetaDrive assets | version 0.4.3 |

## まず実行できる確認

出力先は毎回新しい名前にしてください。runnerは既存の出力ディレクトリを再利用せず、誤って上書きしないように終了します。

### CLIのヘルプ

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning --help
"$LOOKAHEAD_PY" -B -m lookahead_learning doctor --help
"$LOOKAHEAD_PY" -B -m lookahead_learning train --help
"$LOOKAHEAD_PY" -B -m lookahead_learning evaluate --help
"$LOOKAHEAD_PY" -B -m lookahead_learning compare --help
"$LOOKAHEAD_PY" -B -m lookahead_learning test --help
```

### static doctor

`doctor`にruntimeオプションを付けない場合、依存関係・assets・sourceの静的情報だけを調べます。`--output`を付けてもMetaDrive engineは起動せず、結果を`doctor.json`へ保存します。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning doctor \
  --config configs/official_start_lane_return.toml \
  --output outputs/lookahead_learning_docs/doctor-static-s0
```

### 単体・模擬環境テスト

`test`だけを実行すると、単体・模擬環境テスト（実MetaDrive起動・学習なし）を走らせます。runner・geometry・telemetry・checkpointなどを確認する経路です。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning test \
  --output outputs/lookahead_learning_docs/test-unit-s0
```

### static portability

`test --portability`は、空白を含む別ディレクトリへ必要なソースをコピーし、fresh subprocessからhelpとstatic doctorを実行します。`probe`を指定しないこの経路は、MetaDrive engineを起動しません。assetsやモデルはコピーしません。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning test --portability \
  --config configs/official_start_lane_return.toml \
  --output outputs/lookahead_learning_docs/portability-static-s0
```

この結果には、親の`portability_result.json`と、移設先の静的チェック結果が含まれます。`--integration`と同時には使わず、static portabilityとruntime診断を別々に実行してください。

## 任意のengine診断

次のコマンドは、意図的にMetaDrive/Panda3D engineを起動します。いずれもホストの未包装環境（raw環境）を調べる診断であり、`lookahead_obs` / `lookahead_obs_pp_reward`による前方注視比較や262/265次元の検証結果とは扱いません。

### raw probeとsteering pulse

`--probe`はホストの未包装環境（raw環境）をresetして数step進めます。`--pulse`を加えると、離散steeringの符号を確認する短いpulseも実行します。公式TOMLのscenario seedは5です。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning doctor \
  --config configs/official_start_lane_return.toml \
  --probe --pulse --seed 5 --steps 3 \
  --output outputs/lookahead_learning_docs/doctor-raw-pulse-s5
```

### integration test

`test --integration`は、内部的にはraw `doctor --probe`を実行する経路です。wrapper付きPPOテストではありません。`--integration --portability`を一つの確認として組み合わせず、上のstatic portabilityとは別に実行してください。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning test --integration \
  --config configs/official_start_lane_return.toml \
  --seed 5 --steps 3 \
  --output outputs/lookahead_learning_docs/test-integration-raw-s5
```

### 旧checkpointの診断

既存のSB3 zipを確認したい場合は、実在するzipへのパスを明示します。次の`<legacy-model.zip>`はプレースホルダーであり、この文書はモデルの存在を仮定しません。

metadataだけを読む場合:

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning doctor \
  --config configs/official_start_lane_return.toml \
  --checkpoint "<legacy-model.zip>" \
  --output outputs/lookahead_learning_docs/doctor-legacy-metadata-s0
```

ホストの未包装環境（raw環境）上で決定論的に短く評価する場合:

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning doctor \
  --config configs/official_start_lane_return.toml \
  --checkpoint "<legacy-model.zip>" --legacy-evaluate \
  --seed 5 --max-steps 500 \
  --output outputs/lookahead_learning_docs/doctor-legacy-evaluate-s5
```

これは`legacy_raw259_diagnostic`として保存されます。raw observationのshapeとActionがzipとhostで一致する必要があり、学習時のTOMLがsidecarで検証されたことを意味しません。運用評価で使うcheckpointは、後述のrunnerが書いた`metadata.json`付きの対応する学習runを使います。

## 条件付きの学習・評価

このcheckoutで監査されたhostは、raw observation `(259,)`・`float32`です。監査で確認された不足値はindex `259, 260, 261`で、追加adapterの`_VERIFIED_HOST_PREFIX_EVIDENCE`も現在は空です。必要な契約は次のとおりです。

| mode | 必要なpolicy observation |
| --- | ---: |
| `baseline` | `(262,)`。hostが提供する262値をそのまま使う |
| `lookahead_obs` | `(265,)`。検証済みhostの262値に前方注視3値を追加 |
| `lookahead_obs_pp_reward` | `(265,)`。観測は`lookahead_obs`と同じで、PP操舵参照との差ペナルティだけを追加 |

したがって現状では、上の診断コマンドは実行できますが、`train`と`evaluate`は運用契約を満たさず拒否されます。別PCでたまたま`(262,)`のvectorが返るだけでも十分ではありません。index 259--261の意味・順序・encoding・正規化をソースで確認し、追加adapterのregistryへ登録したhostが必要です。vectorのpadding、切り捨て、hostの意味を推測する設定は行いません。

その条件を満たすhostを用意した後に、まず次の短いsmokeを実行できます。各modeは同じTOML、learning seed `0`、scenario seed `5`、`timesteps=64`、`num_envs=1`、`n_steps=64`に揃え、model名とoutputを明示します。`lookahead_obs_pp_reward`だけは`--pp-weight`を明示します。

### 3 modeの短いsmoke学習

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning train \
  --config configs/official_start_lane_return.toml \
  --mode baseline --seed 0 \
  --timesteps 64 --num-envs 1 --n-steps 64 \
  --model-name baseline_smoke_s0 \
  --output outputs/lookahead_learning_docs/train-baseline-s0

"$LOOKAHEAD_PY" -B -m lookahead_learning train \
  --config configs/official_start_lane_return.toml \
  --mode lookahead_obs --seed 0 \
  --timesteps 64 --num-envs 1 --n-steps 64 \
  --model-name lookahead_obs_smoke_s0 \
  --output outputs/lookahead_learning_docs/train-lookahead-obs-s0

"$LOOKAHEAD_PY" -B -m lookahead_learning train \
  --config configs/official_start_lane_return.toml \
  --mode lookahead_obs_pp_reward --pp-weight 0.1 --seed 0 \
  --timesteps 64 --num-envs 1 --n-steps 64 \
  --model-name lookahead_obs_pp_reward_smoke_s0 \
  --output outputs/lookahead_learning_docs/train-lookahead-obs-pp-reward-s0
```

成功した場合、各学習runは指定したディレクトリに次を作ります。

```text
<train-output>/<model-name>.zip
<train-output>/metadata.json
```

ここでの`0.1`はコマンド形式を確認するための暫定smoke値です。学習効果を確認した値でも、長時間実験で推奨する値でもありません。

### 同じcheckpointのscenario 5評価

評価は3つとも同じTOML、evaluation seed `0`、`--scenario-seeds 5`、決定論的Action、最大500 stepで実行します。PP modeでは学習時と同じ`--pp-weight 0.1`を指定してください。

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning evaluate \
  --config configs/official_start_lane_return.toml \
  --mode baseline --checkpoint \
  outputs/lookahead_learning_docs/train-baseline-s0/baseline_smoke_s0.zip \
  --scenario-seeds 5 --seed 0 --deterministic --max-steps 500 \
  --output outputs/lookahead_learning_docs/eval-baseline-s0

"$LOOKAHEAD_PY" -B -m lookahead_learning evaluate \
  --config configs/official_start_lane_return.toml \
  --mode lookahead_obs --checkpoint \
  outputs/lookahead_learning_docs/train-lookahead-obs-s0/lookahead_obs_smoke_s0.zip \
  --scenario-seeds 5 --seed 0 --deterministic --max-steps 500 \
  --output outputs/lookahead_learning_docs/eval-lookahead-obs-s0

"$LOOKAHEAD_PY" -B -m lookahead_learning evaluate \
  --config configs/official_start_lane_return.toml \
  --mode lookahead_obs_pp_reward --pp-weight 0.1 --checkpoint \
  outputs/lookahead_learning_docs/train-lookahead-obs-pp-reward-s0/lookahead_obs_pp_reward_smoke_s0.zip \
  --scenario-seeds 5 --seed 0 --deterministic --max-steps 500 \
  --output outputs/lookahead_learning_docs/eval-lookahead-obs-pp-reward-s0
```

各評価runの`summary.json`を比較入力にします。evaluation側は、学習runと同じmodeとPP weightをmetadataで照合し、対応する学習runの`metadata.json`がないzipを運用checkpointとして読みません。

### 2つの比較

baselineと前方注視入力の差:

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning compare \
  outputs/lookahead_learning_docs/eval-baseline-s0/summary.json \
  outputs/lookahead_learning_docs/eval-lookahead-obs-s0/summary.json \
  --before-mode baseline --after-mode lookahead_obs \
  --output outputs/lookahead_learning_docs/compare-baseline-lookahead-obs-s0
```

前方注視入力とPP報酬の差:

```bash
"$LOOKAHEAD_PY" -B -m lookahead_learning compare \
  outputs/lookahead_learning_docs/eval-lookahead-obs-s0/summary.json \
  outputs/lookahead_learning_docs/eval-lookahead-obs-pp-reward-s0/summary.json \
  --before-mode lookahead_obs --after-mode lookahead_obs_pp_reward \
  --output outputs/lookahead_learning_docs/compare-lookahead-obs-lookahead-obs-pp-reward-s0
```

`compare`は既定でstrict metadataを使い、結果を指定先の`comparison.json`へ保存します。比較元のevaluation runを取り違えないよう、上記のmodeとファイル名を対応させてください。

## 長時間実験を計画する場合

[`configs/official_start_lane_return.toml`](../../configs/official_start_lane_return.toml) の長時間設定は、`total_timesteps = 300000`、`num_envs = 4`、`n_steps = 4096`、learning seed `0`です。train/evaluateの実行条件が未成立のため、この文書ではこの設定の学習・評価を実行済みとは扱いません。

複数条件を計画する場合は、scenario seed `5`を固定し、learning seed `0, 1, 2`を別runとして揃えます。learning seedをscenario seedの代わりに流用せず、各modeで予算・TOML・評価scenarioを一致させます。smokeで使った`0.1`を長時間runの推奨値へ読み替えず、採用するPP weightは実験条件として別途決めてmetadataへ残してください。
