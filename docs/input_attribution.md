# PPO 入力寄与・入力依存度解析

`analyze_input_attribution.py` は、固定済みの Stable-Baselines3 PPO `MlpPolicy` を対象に、MetaDrive の flat vector observation を解析します。学習、既存の `evaluate.py`、MetaDrive 本体は変更しません。初版は single-agent、flat `Box` 観測、single `Discrete` action、PPO `ActorCriticPolicy`/`MlpPolicy` を明示的に対象とし、未対応の policy / action space は失敗として停止します。

この実装の摂動という考え方は、入力の一部を変えて出力変化を観察する [Greydanus et al., *Visualizing and Understanding Atari Agents* (ICML 2018)](https://proceedings.mlr.press/v80/greydanus18a.html) を vector observation に適用したものです。Integrated Gradients (IG) は [Sundararajan, Taly, Yan, *Axiomatic Attribution for Deep Networks* (ICML 2017)](https://arxiv.org/abs/1703.01365) に従い、PyTorch autograd だけで計算します。結果は説明的・探索的な手掛かりであり、saliency を因果的な説明として扱わない [Atrey et al., *Exploratory Not Explanatory* (2020)](https://arxiv.org/abs/1912.05743) の注意を守ります。

## 実行前の確認

公式 259 次元 schema は `observation_schemas/metadrive_default_259.toml` を使います。適用できる MetaDrive 設定条件と 262 次元 template の編集方法は [observation_schemas/README.md](../observation_schemas/README.md) を先に確認してください。

実行時は env observation space、model observation space、`reset()` observation、schema の `observation_dim`、schema の index 全被覆を比較します。どれかが不一致なら解析を続行しません。`official_baseline.zip` が無い場合も、期待したモデル path を含む明確なエラーで停止します。

## 基本コマンド

プロジェクト root で実行します。

```bash
.venv/bin/python analyze_input_attribution.py run \
  --config configs/official.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --output-prefix official_left_curve
```

`run` は `collect → offline perturbation → IG → aggregation → plots → optional closed-loop` を一つの staging directory で実行し、成功時にだけ次へ公開します。

```text
outputs/<experiment-name>/attribution/<output-prefix>/
```

既に成功した同名 run は、次回 run が途中失敗しても置き換えません。`--output-prefix` は basename だけを受け付け、path traversal と symlink destination を拒否します。

個別のサブコマンドもあります。

```bash
# MetaDrive を走らせず、保存済み rollout を再解析する
.venv/bin/python analyze_input_attribution.py analyze \
  --config configs/official.toml --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --rollout outputs/official/attribution/official_left_curve \
  --output-prefix official_left_curve_reanalysis

# schema と実環境・model の shape を確認する
.venv/bin/python analyze_input_attribution.py validate-schema \
  --config configs/official.toml --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\analyze_input_attribution.py run `
  --config .\configs\official.toml `
  --model .\models\official_baseline.zip `
  --schema .\observation_schemas\metadrive_default_259.toml `
  --analysis-config .\attribution_configs\official_left_curve.toml `
  --output-prefix official_left_curve
```

Windows Command Prompt:

```bat
.venv\Scripts\python.exe analyze_input_attribution.py run ^
  --config configs\official.toml ^
  --model models\official_baseline.zip ^
  --schema observation_schemas\metadrive_default_259.toml ^
  --analysis-config attribution_configs\official_left_curve.toml ^
  --output-prefix official_left_curve
```

WSL / Linux:

```bash
.venv/bin/python analyze_input_attribution.py run \
  --config configs/official.toml --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --output-prefix official_left_curve
```

## 保存されるもの

通常 rollout は policy decision **前**の float32 observation copy、logits、log probabilities、probabilities、critic value、selected/environment action、episode / scenario / step / reward / done を `rollout_arrays.npz` に保存します。可読な scalar telemetry は `rollout_steps.jsonl`、runtime contract と seed は `rollout_metadata.json` です。

`run` はさらに `analysis_metadata.json`、`feature_schema_expanded.csv`、摂動 feature/group 表、IG attribution NPZ / feature / group / completeness 表、`closed_loop_runs.jsonl` / `closed_loop_summary.csv`、`report.md`、`plots/` を作ります。データ不足や closed-loop 無効時にも標準 plot 名は、理由を記した placeholder として残ります。

## 摂動と baseline

各 observation `x` の feature/group `G` だけを baseline `b` の値で置換して `x_tilde[G] = b[G]` とし、それ以外の index、元 `x`、baseline `b` は変更しません。主指標は actor probability distribution 間の Jensen–Shannon divergence です。補助的に action flip、元 selected action probability drop、centered logits/log-probability の距離、critic `value_delta` / absolute / squared delta を保存します。raw logits そのものは共通加算不変性があるため、raw-logit L2 を主指標にはしません。

baseline は `episode_start`、`specified_steps`、`sampled_observations`、`external_npz` を選べます。標準設定は理解しやすい `episode_start` です。ゼロ vector は MetaDrive における中立値とは限らないため標準 baseline にしません。複数 baseline の結果は平均し、選択内容を metadata に残します。

代表的な baseline 設定は次の形です。`specified_steps` の `episode_id` と `step` は保存済み rollout の値を指定します。`external_npz` の相対 path は analysis TOML の所在 directory から解決され、使用ファイルの SHA-256、配列 shape、選択 row も metadata に残ります。

```toml
[baseline]
strategy = "specified_steps"
count = 2
seed = 0
specified_steps = [
  { episode_id = 1, step = 20 },
  { episode_id = 1, step = 60 },
]

# または
[baseline]
strategy = "external_npz"
count = 2
seed = 7
path = "../references/verified_observations.npz"
key = "observations"
```

## Integrated Gradients

標準 target は現在の `x` で選ばれた `a* = argmax π(a|x)` の `log π(a*|z)` です。「`a*` を固定する」とは、baseline から `x` までの各補間点で argmax をやり直さず、**最初に決めた action index** の log probability を評価することです。環境にその action を強制する意味ではありません。

補助 target は selected-vs-runner-up raw-logit margin と critic value です。台形則、両端を含む `steps >= 2` で IG を近似し、signed IG、absolute IG、target action、`F(x)`、`F(x0)`、`sum(IG)`、completeness residual / relative error を保存します。completeness は signed IG でだけ確認します。

`mean_absolute_ig` は寄与の**大きさ**のランキングであり、正の支持ではありません。group では signed IG sum と absolute mass を別々に示します。IG は baseline から現在点までの出力差の帰属であり、現在点の微分感度そのものではありません。

`ig_group_summary.csv` には schema group と、`perturbation.lidar_sector_degrees` で生成した角度 sector が別の `group_kind` として入ります。sector の `group_signed_ig` は signed sum、`group_absolute_mass` は `sum(abs(IG))`、normalized mass は各 sample の LiDAR 全 absolute mass に対する比です。重複 group を含む一般 schema では group 間の値を単純加算できません。signed group の総和が出力差に対応するのは、それらが全 feature の排他的 partition である場合だけです。

## 時系列と閉ループ

集計は full episode、step 時系列、episode progress の early/middle/late 3 分割、および analysis config の明示 phase を出します。3 分割は「curve entry / apex / exit」を自動的に意味しません。

step phase は `[start_step, end_step)`、normalized progress phase は通常 `[start_progress, end_progress)` です。ただし `end_progress = 1.0` は episode 最終 decision を含みます。

```toml
[[phases]]
name = "verified_curve_interval"
start_step = 20
end_step = 60

[[phases]]
name = "second_half"
start_progress = 0.5
end_progress = 1.0
```

閉ループは top-K または明示した feature/group を、同じ scenario seed と RL seed で通常走行の paired baseline と比較します。env が返す observation の copy だけを policy 入力として置換し、env 内部状態、reward、done、model weights、env に渡す action を後から改変しません。`episode_start_constant`、`specified_reference_constant`、`dataset_median_constant`、schema に明示された場合の `schema_constant` を選べます。dataset median は実在する完全 observation とは限りません。

閉ループで performance が変わっても「入力が一般の環境タスクに不可欠」とは結論できません。これは固定済み方策がその input change にどの程度依存するかを調べる反実仮想的検証であり、Remove-and-Retrain は対象外です。

## 結果を読む際の注意

- 摂動依存度と IG 寄与度は別の問いに答えるため、単一の総合 score に合成しません。
- 摂動は置換 baseline に、IG は baseline と積分 path に依存します。
- individual feature の摂動値は相関・冗長性があるため加算できません。
- hybrid observation は学習分布外となる可能性があります。
- 1 scenario の結果を他道路、他 traffic、他 seed へ一般化しません。
- `report.md` は自動集計の読み方と警告を記録しますが、因果的な説明の証明ではありません。

初版では offline 再解析に observation 配列が必須なので `collection.save_observations = true` だけを受け付けます。`run.record_visualization` は MetaDrive の simulator recording を意味し、初版では `false` のみ対応します。解析用の静的 PNG plot はこの値に関係なく常に生成されます。未知 key、これらの未対応値、`steps < 2` は黙って無視せず設定読込時に拒否します。
