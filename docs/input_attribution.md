# PPO 入力寄与・入力依存度解析

## この実験で調べること

`analyze_input_attribution.py` は、学習済みの Stable-Baselines3 PPO を固定したまま、MetaDrive の各観測入力に方策がどの程度依存しているかを調べます。通常走行の収集を準備として、結果を次の三つの実験に分けます。

1. **実験01 入力置換（オフライン摂動）**：ある入力を基準値へ置き換えると、行動確率分布はどの程度変わるか。
2. **実験02 Integrated Gradients（出力変化の入力への割当）**：基準観測から現在観測までの出力変化を、各入力へどのように割り当てられるか。
3. **実験03 入力固定での走行比較（paired closed-loop）**：実験01のJSDで選んだ入力を走行中も置き換えると、報酬や完走成否がどう変わるか。

これは固定済み方策の**入力依存性を調べる探索的実験**です。入力が運転タスク一般に必要であることや、入力と成功・失敗の因果関係を証明する実験ではありません。学習、model weights、`evaluate.py`、MetaDrive 本体は変更しません。

全体像を先に確認する場合は [入力寄与解析の概要](input_attribution_overview.md) を参照してください。概要は目的・入力・方法と、実行後に見る表の形を平易な文章で説明し、この文書は方法・設定の詳細、[結果の読み方](input_attribution_results.md) は列・数式のリファレンスを扱います。観測 index の成立条件は [観測スキーマ](../observation_schemas/README.md)、262 次元環境への移植は [移植手順](input_attribution_porting.md) を参照してください。

## 実験全体の流れ

`run` サブコマンドは次の順で処理します。

```text
準備：通常走行の収集: x_t, π(·∣x_t), V(x_t), action, reward
               │
               ├─ 実験01 入力置換（オフライン摂動）
               │                  └─ 入力置換による方策分布・critic の変化
               │
               ├─ 実験02 Integrated Gradients
               │                  └─ 出力差を入力次元へ割当（実験03の候補選択には使わない）
               │
               └─ 実験03 入力固定での走行比較（paired closed-loop）
                                  └─ 実験01のJSD上位、または明示指定した入力を比較
```

各段階が答える問いは異なります。

| 所属 | 見るもの | 主な出力 | この段階だけで言えること |
| --- | --- | --- | --- |
| 準備：通常走行の収集 | 介入前の走行と方策出力 | `shared/` | 解析対象となった入力・行動・走行の事実 |
| 実験01 入力置換（オフライン摂動） | 入力置換前後の方策分布 | `experiment_01_perturbation/` | その置換に対する方策出力の依存度 |
| 実験02 Integrated Gradients（出力変化の入力への割当） | 基準値から現在値までの出力差 | `experiment_02_integrated_gradients/` | 指定した出力差を各入力へ割り当てた量 |
| 実験03 入力固定での走行比較（paired closed-loop） | 通常走行と入力固定走行の結果差 | `experiment_03_closed_loop/` | 固定方策に対する置換の走行上の影響 |

## 記号

ここで `D` は観測の次元数、`ℝᴰ` は D 個の実数を並べたベクトルの集合です。

| 記号 | 意味 |
| --- | --- |
| `xₜ ∈ ℝᴰ` | decision step `t` で、行動決定前に方策へ渡す観測 |
| `b ∈ ℝᴰ` | 比較の基準とする baseline 観測 |
| `G` | 同時に置き換える feature index の集合 |
| `π(a ∣ x)` | actor の離散行動確率 |
| `lₐ(x)` | action `a` の logit |
| `V(x)` | critic の value estimate |
| `a*` | 元の観測 `x` で確率最大の行動 |

本実装は single-agent、1 次元の flat `Box` 観測、単一 `Discrete` action、PPO `ActorCriticPolicy` / `MlpPolicy` を対象とします。未対応の policy・観測・action space は解析を続けずエラーにします。

## 準備：通常走行の収集

各 scenario を決定論的方策で通常走行し、`env.step(action)` より前の `xₜ` を float32 の copy として保存します。同じ行に logits、log probabilities、probabilities、critic value、選択 action、reward、終了状態を対応付けます。したがって保存データの 1 行は 1 simulation frame ではなく、**1 policy decision** です。

この段階は attribution ではありません。後続実験が同じ観測列を使うための測定データと、介入前走行の基準を作ります。

## 実験01 入力置換（オフライン摂動）

### 何を見るか

観測の feature、schema group、または LiDAR 角度 sector だけを baseline の値へ置き換え、方策出力がどの程度変わるかを測ります。対象外の index と環境状態は変えません。

置換後の観測は次のように定義します。

$$
\tilde{x}_{t,G,j}^{(b)} =
\begin{cases}
b_j & (j\in G),\\
x_{t,j} & (j\notin G)
\end{cases}
$$

上式の置換後ベクトルを、添字を省略して `x̃` と書きます。元の actor 分布を `p = π(· ∣ xₜ)`、置換後を `q = π(· ∣ x̃)`、平均分布を `m = (p + q) / 2` とすると、主指標は自然対数の Jensen–Shannon divergence です。

$$
D_{\mathrm{JS}}(p,q)
=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)
+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m)
$$

`D_JS = 0` なら分布は同一で、値が大きいほど置換前後の分布差が大きいことを表します。補助指標も別々に保存します。

| 指標 | 定義 | 符号・意味 |
| --- | --- | --- |
| `action_changed` | `1[argmax(p) ≠ argmax(q)]` | 1 なら決定論的 action が変化 |
| `selected_action_probability_drop` | `p(a*) − q(a*)` | 正なら元の選択 action の確率が低下 |
| `centered_logit_l2` | `‖(l′ − l̄′) − (l − l̄)‖₂` | 大きいほど相対的な logit 配置が変化 |
| `value_delta` | `V(x̃) − V(x)` | critic の予測差。実 reward の差ではない |

feature を個別に置換した値は、相関や冗長性があるため足し合わせられません。group の比較は group 全体を一度に置換する別実験であり、個別 feature の結果の合計でもありません。

### baseline の設定

baseline は摂動と IG の結論を変える実験条件です。ゼロ vector が MetaDrive の中立状態とは限らないため、標準設定では使いません。

| `baseline.strategy` | 選ばれる観測 | 主な用途 |
| --- | --- | --- |
| `episode_start` | 各 `xₜ` と同じ episode の最初の保存観測 | 現在の標準。開始状態との差を見る |
| `specified_steps` | 指定した `episode_id` / `step` | 意味を確認した状態を基準にする |
| `sampled_observations` | rollout から seed 付きで抽出 | 複数の実在観測に対する頑健性を見る |
| `external_npz` | 外部 NPZ の指定配列 | 別途検証した基準集合を使う |

`count > 1` では raw 結果に baseline 軸を残し、summary は各 step で baseline 間を平均してから統計量を計算します。`episode_start` の `seed` は選択に使われません。

## 実験02 Integrated Gradients（出力変化の入力への割当）

### 何を見るか

IG は「baseline `x₀` から現在観測 `x` へ変化したときの出力差を、各入力へどれだけ割り当てられるか」を調べます。ここで `F` は、観測を受け取り、調べたい出力を一つの数値として返す関数です。直線 path

$$
z(\alpha)=x_0+\alpha(x-x_0),\qquad 0\le\alpha\le1
$$

に対する feature `j` の attribution は

$$
\operatorname{IG}_j(x;x_0)
=(x_j-x_{0,j})
\int_0^1
\frac{\partial F(z(\alpha))}{\partial z_j}\,d\alpha
$$

です。実装は両端を含む `S = integrated_gradients.steps` 個の点を使い、台形則で近似します。

$$
\operatorname{IG}_j
\approx
(x_j-x_{0,j})\frac{1}{S-1}
\left[
\frac{g_j^{(0)}}{2}
+\sum_{k=1}^{S-2}g_j^{(k)}
+\frac{g_j^{(S-1)}}{2}
\right]
$$

上の式で使う関数 `F` は、解析設定ファイルの `integrated_gradients.targets`（IG で調べる出力を選ぶ設定項目）で指定します。これは TOML の `[integrated_gradients]` 節にある `targets` のことで、次の表の設定値に対応する関数を `F` として使います。

| `targets` に指定する値 | 上の式で使う関数 `F` | 調べる出力 |
| --- | --- | --- |
| `selected_log_probability` | `F(z) = log π(a* ∣ z)` | 元の選択 action の log probability |
| `selected_vs_runner_up_margin` | `F(z) = l[a*](z) − l[a₂](z)` | 元の上位 action と次点 action の margin |
| `critic_value` | `F(z) = V(z)` | critic value |

`l[a](z)` は観測 `z` に対する action `a` の logit であり、記号表の `lₐ(x)` の入力を `z` にしたものです。

`a*` と runner-up `a₂` は元の `x` で一度だけ決め、補間点ごとに選び直しません。環境へ action を強制する処理ではありません。

上の式で計算した各入力の寄与 `IGⱼ` は、正にも負にもなります。正なら基準観測からの出力差 `F(x) − F(x₀)` に対して出力を増やす方向、負なら減らす方向の寄与を表します。この符号を保った IG の値を **signed IG（符号付きの寄与度）** と呼びます。正なら良い運転、負なら悪い運転という意味ではありません。

一方、結果の列 `mean_absolute_ig` は、IG の絶対値を取ってから平均した値です。寄与の向きではなく、大きさを比較するときに使います。また、調べる出力 `F` の単位が異なるため、actor IG と critic IG の数値は直接比較しません。

数値積分の確認には completeness residual

$$
r=\sum_j\operatorname{IG}_j-\{F(x)-F(x_0)\}
$$

を使います。残差が大きい場合は、結果を解釈する前に `steps` を増やして再計算します。

## 実験03 入力固定での走行比較（paired closed-loop）

### 何を見るか

オフライン摂動で上位になった feature/group、または明示した target について、同じ scenario seed と RL seed で通常走行と介入走行を実行します。介入走行では env が返した観測の copy の `G` だけを一定値へ置き換えて方策へ渡します。env 内部状態、model weights、reward、done、および方策が返した action は改変しません。

各 outcome `M` の保存値は

$$
\Delta M_G
=M_{\mathrm{intervention},G}
-M_{\mathrm{baseline}}
$$

です。例えば `mean_delta_total_reward < 0` は介入走行の平均 reward が通常走行より低く、`mean_delta_success = -1` は success fraction が 1 減ったこと（この 1 scenario では通常成功・介入失敗）を表します。

| `closed_loop.replacement_strategy` | 介入中に固定する値 |
| --- | --- |
| `episode_start_constant` | 同じ scenario の reset 観測 |
| `specified_reference_constant` | 最初の解決済み baseline reference。reference ID は metadata で確認 |
| `dataset_median_constant` | 保存 rollout の feature ごとの median |
| `schema_constant` | schema に明示した定数 |

`run` で明示 target がない場合、**実験01**の `full_episode` / `mean_over_baselines` における `mean_js_divergence` 上位から `top_k_features` と `top_k_groups` を選びます。`top_k_groups` の候補は schema group と LiDAR sector の混合順位です。semantic group だけを固定して検証したい場合は `explicit_groups` を使います。実験02の IG の順位から選ぶ処理ではありません。

feature と group を明示した場合は、その側のランキングを追加しません。例えば、元の `[closed_loop]` block の他の key を維持したまま、次のように指定すると、`normalized_speed` と `navigation` だけを介入対象にできます。

```toml
[closed_loop]
enabled = true
top_k_features = 0
top_k_groups = 0
explicit_features = ["normalized_speed"]
explicit_groups = ["navigation"]
replacement_strategy = "episode_start_constant"
```

`explicit_features` と `explicit_groups` の両側を明示して `top_k_features = 0`、`top_k_groups = 0` とするため、摂動の順位から対象が追加されることはありません。model を変える場合は CLI の `--model`、比較対象を変える場合は analysis TOML の `[closed_loop]` を変更します。

paired closed-loop は、入力置換が固定方策の挙動へ波及したかを確かめます。それでも Remove-and-Retrain の効果や、元の入力がタスク一般に不可欠であることは示しません。

## 保存済み `official_baseline.zip` の結果例（`official_left_curve`）

設定値は `configs/official.toml`、`attribution_configs/official_left_curve.toml`、`observation_schemas/metadrive_default_259.toml` から、走行・解析の実測値は現在保存されている run から確認したものです。

| 項目 | 設定値・保存済み run の実測値 |
| --- | --- |
| 固定 model | `models/official_baseline.zip`、PPO `MlpPolicy`、deterministic 推論、CPU |
| 環境 | map `C`、horizon 500、traffic density 0、accident probability 0 |
| action | steering 3 × throttle/brake 3 の `Discrete(9)` |
| seed | RL seed 0、evaluation scenario seed 5 を 1 件 |
| 観測 | 259 次元: ego state 9、navigation 10、LiDAR 240 |
| 通常 rollout | 実測 1 episode、127 policy decisions、通常走行は success |
| baseline | `episode_start`、1 個。今回の 1 episode では全 step が rollout row 0 を参照 |
| 摂動 target | 259 features + 12 schema groups + 24 個の 15° LiDAR sectors = 295 |
| IG | `selected_log_probability` と `critic_value`、64 点台形則 |
| summary（出力仕様） | 全観測をまとめた full-episode のみ（保存済み run は 127 policy decisions） |
| closed-loop | 摂動上位 10 features + group/sector 混合順位の上位 5 件、1 scenario に対する 15 paired comparisons。今回選ばれた 5 件は全て schema group |
| 保存設定 | observation 保存は必須で有効、simulator recording は無効。静的な標準 plot は生成 |

重要なのは、`official_left_curve` は analysis config と出力 prefix の**名前**にすぎないことです。現状の結果は seed 5 の episode 全体を解析した 1 scenario の case study であり、左カーブ固有の結果や seed 間統計として報告してはいけません。

また現在の通常 rollout では LiDAR 240 次元が全 127 decision で 1.0 のままです。この run で LiDAR の摂動・IG が 0 になるのは、現在観測と episode-start baseline の LiDAR 値に差がないためです。方策が他の交通条件でも LiDAR を使わないという証拠ではありません。

## どこを変えるか

設定は役割ごとに分かれています。既存の公式結果を再現できるよう、元ファイルを直接上書きせず copy を作って変更してください。

| 変えたいもの | 変更場所 | 主な key / 引数 | 影響 |
| --- | --- | --- | --- |
| 道路、traffic、horizon、action | experiment TOML | `[environment.common]` | 収集・closed-loop の環境 |
| scenario の範囲 | experiment TOML | `[environment.evaluation].start_seed` / `num_scenarios` | rollout 数と paired comparison 数 |
| RL seed | experiment TOML または CLI | `[evaluation].seed` / `--seed` | action space 等の乱数。scenario 範囲は変えない |
| 解析する model | CLI | `--model` | 固定方策そのもの |
| observation の index と意味 | schema TOML | `observation_dim`、`[[blocks]]`、`[[ranges]]` | feature/group/angle の定義 |
| baseline | analysis TOML | `[baseline]` | 摂動と IG の基準 |
| 摂動対象・sector 幅 | analysis TOML | `[perturbation]` | feature/group/sector の計算対象 |
| IG target・近似精度 | analysis TOML | `[integrated_gradients].targets` / `steps` | 帰属する関数と積分点数 |
| closed-loop 対象・置換値 | analysis TOML | `[closed_loop]` | paired intervention |
| GPU/CPU | analysis TOML または CLI | `[run].device` / `--device` | PPO 推論と IG の device |
| 出力 directory 名 | CLI | `--output-prefix` | `outputs/<experiment>/attribution/<prefix>/` |

### 複数 scenario にする

```bash
cp configs/official.toml configs/official_attribution_multiseed.toml
cp attribution_configs/official_left_curve.toml \
  attribution_configs/official_attribution_multiseed.toml
```

experiment config の copy 側で少なくとも次を変更します。

```toml
name = "official_attribution_multiseed"

[environment.evaluation]
start_seed = 5
num_scenarios = 20
```

analysis config の copy 側でも、先頭の名前を `name = "official_attribution_multiseed"` に変えます。解析方法を変えない場合、それ以外の値は維持します。

model は再学習せず、次の完全なコマンドで元の model を明示します。

```bash
.venv/bin/python analyze_input_attribution.py run \
  --config configs/official_attribution_multiseed.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_attribution_multiseed.toml \
  --output-prefix official_attribution_multiseed_numbered \
  --output-mode compact
```

複数 scenario にすると結果の一般化可能性は改善しますが、同一設定内での seed 範囲に限った評価です。

### baseline を明示 step にする

copy した analysis TOML の既存 `[baseline]` block を、次の内容へ置き換えます。同名 block を末尾へ追加すると TOML が重複するため、追記はしません。

```toml
[baseline]
strategy = "specified_steps"
count = 2
seed = 0
specified_steps = [
  { episode_id = 1, step = 20 },
  { episode_id = 1, step = 60 },
]
```

指定行は入力に使う rollout 内に存在する必要があります。外部 baseline を使う場合は `strategy = "external_npz"` とし、`path` / `key` を指定します。相対 path は analysis TOML の directory から解決されます。

### 計算量・出力量を抑える

出力形式は `--output-mode compact`（既定）と、明示したときだけ使う `--output-mode full` から選べます。compact は root の対応表、実験01/02/03の各 `report.md` + `details.zip`、`shared/details.zip` を保存します。各 ZIP の member path は run root 相対です（例: `experiment_01_perturbation/details.zip` 内の `experiment_01_perturbation/perturbation_feature_summary.csv`、`shared/details.zip` 内の `shared/rollout_arrays.npz`）。full も同じ実験別 directory 構成を使いますが、ZIP ではなく各 directory に CSV、NPZ、JSONL、PNG、metadata を個別に保存し、元の詳細 report は `shared/report.md` として保全します。method を無効にした場合も、互換性のため空 CSV/NPZ や placeholder plot は所属する実験の詳細に含めます。

| 変更 | 減るもの | 失う情報・残るもの |
| --- | --- | --- |
| IG targets を 2→1 | IG 計算量と summary 行数が約半分 | 外した target の解釈を失う |
| IG `steps` を 64→32 | IG の勾配計算量 | raw 配列・summary の行数はほぼ変わらず、積分誤差が増える可能性 |
| `analyze_features = false` | feature ごとの摂動 | IG feature summary と LiDAR sectors は残る |
| `analyze_groups = false` | schema group の摂動 | LiDAR sectors は残る |
| `lidar_sector_degrees = 360` | 24 sectors→1 sector | 角度分解能を失う。sector の完全無効化設定はない |
| `closed_loop.enabled = false` | paired env 実行 | 空の closed-loop ファイルと placeholder plot は残る |
| closed-loop top-K を小さくする | paired episode 数 | offline 摂動・IG の量は変わらない |

旧 `official_left_curve` の flat な full 結果は読み取り互換のため保持します。新しい numbered output では、compact/full のどちらも実験別 directory と `shared/` に分けます。通常の提出物、数値根拠、再解析 archive の分け方は [結果の読み方](input_attribution_results.md#出力が多い理由と配布範囲) に示します。

## 実行手順

project root、つまり `metadrive_rl-main/` で、通常は次の `run` コマンドだけを実行します。

```bash
.venv/bin/python analyze_input_attribution.py run \
  --config configs/official.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --output-prefix official_left_curve_numbered
```

`run` は収集開始時に env・model・schema の observation/action dimension 契約を検証し、各 scenario の reset 観測も検証してから、準備の通常走行、実験01の摂動、実験02のIG、各実験の report、設定時の実験03まで続けて実行します。既定の compact を使うため、この実行例では output mode を省略しています。`validate-schema` は通常実行の前提ではなく、環境・model・schema を変更したときに結果 directory を作らず契約だけを診断する任意のコマンドです。

```bash
.venv/bin/python analyze_input_attribution.py validate-schema \
  --config configs/official.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml
```

| サブコマンド | 用途 | MetaDrive 実行 | 保存物 |
| --- | --- | --- | --- |
| `validate-schema` | env / reset / model / schema の dimension 契約を確認 | reset 1 回 | result directory は作らず、検証 JSON を標準出力 |
| `collect` | 準備：通常走行だけ収集 | あり | `shared/` の rollout と schema 関連ファイル、および準備用 root report。3実験は扱わない |
| `run` | 準備、実験01/02、各 report、設定時の実験03 | あり | numbered root、実験01/02/03、`shared/` の compact/full 構成。未実施実験も report に明記 |
| `analyze` | 保存 rollout を実験01/02として別設定で再解析 | なし。実験03は行わない | 指定した numbered output の実験別構成。旧 flat 入力も読み取り互換。3 report と未実施表記を保存 |
| `closed-loop` | 実験03の明示した feature/group の paired intervention だけ実行 | あり | root + `shared/` + 3 report。実験01/02は未実施として明記し、`experiment_03_closed_loop/` に結果を保存 |
| `compact` | 既存の full 結果を再実行なしで compact 化 | なし | 元 directory は変更せず、numbered output の各実験・`shared/`構成 |

`run`、`analyze`、`closed-loop` はいずれも `--output-mode {full,compact}` を受け付けます。既定は `compact` で、`--output-mode full` を明示したときだけ実験別 directory に個別 artifact 一式を保存します。compact は表示上のファイル数を減らすだけで、証拠を削除しません。4つの ZIP を復元するときは、各 ZIP を実験 directory へ個別に展開せず、4つすべてを同じ新しい run root へ展開します。復元後の `analyze --rollout` には、その run root を指定します。

保存済み rollout の再解析例です。`analyze` の入力 `--rollout` は full mode の元 directory、または `details.zip` を展開した詳細一式の directory を指定します。元 directory は上書きせず、compact 出力を作る場合は別 prefix を指定します。

```bash
.venv/bin/python analyze_input_attribution.py analyze \
  --config configs/official.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --rollout outputs/official/attribution/official_left_curve \
  --output-prefix official_left_curve_numbered_reanalysis
```

既存の full 結果を再解析せず compact 化する packaging の正確なコマンドは、CLI の numbered output 仕様に合わせて別途確認します。この文書では、元の flat directory を変更せず、実験別 directory と `shared/` を持つ出力へ分ける方針だけを示します。

`analyze` と、`--rollout` を指定する `closed-loop` は、保存 rollout の model/config/schema hash などが現在の指定と一致しなければ拒否します。新しく収集する `run` は、実環境・model・schema の observation/action dimension を検証します。`full` と `compact` の各出力は staging directory で作成し、全処理が成功した場合だけ numbered output を置き換えます。旧 flat directory は読み取り互換の入力として扱い、元の結果を直接変更しません。旧 flat の root `details.zip` は旧構造のまま保持し、新構造の4 ZIPとして自動展開しません。

## 保存先

保存先は

```text
metadrive_rl-main/outputs/<experiment-name>/attribution/<output-prefix>/
```

です。推奨サンプルコマンドでは `experiment-name = official`、`output-prefix = official_left_curve_numbered` なので、compact 出力は

```text
metadrive_rl-main/outputs/official/attribution/official_left_curve_numbered/
├── report.md
├── experiment_01_perturbation/
│   ├── report.md
│   └── details.zip
├── experiment_02_integrated_gradients/
│   ├── report.md
│   └── details.zip
├── experiment_03_closed_loop/
│   ├── report.md
│   └── details.zip
└── shared/
    └── details.zip                   # compact の共通 metadata・schema・通常 rollout
```

root の `report.md` は、実験01/02/03の対応表、各実験 `report.md` へのリンク、実験03の通常/介入比較表を含みます。各実験 directory の `report.md` は、その実験の要約です。compact では各実験と `shared/` に `details.zip` を置きます。full では同じ directory 構成に個別 artifact を保存し、元の詳細 report を `shared/report.md` として保全します。ZIP の member path は run root 相対で、4 ZIP は同じ新しい run root へ展開します。既存の `metadrive_rl-main/outputs/official/attribution/official_left_curve/` は flat な過去結果として保持し、新しい prefix で上書きしません。`outputs/official/attribution/` は複数 prefix を置く親 directory であり、それ自体が一つの結果ではありません。`outputs/` は通常 Git 管理外です。

### 実験別のファイル所属

| 所属 | 保存先 | 主な内容 |
| --- | --- | --- |
| 準備：通常走行の収集 | `shared/` | 共通 metadata、schema、rollout arrays/steps/metadata、設定・hash（compact は `details.zip`、full は個別 raw と保全用 `report.md`） |
| 実験01 入力置換（オフライン摂動） | `experiment_01_perturbation/` | 摂動 raw、feature/group summary、JSD の report |
| 実験02 Integrated Gradients | `experiment_02_integrated_gradients/` | IG raw、feature/group summary、completeness、IG の report |
| 実験03 入力固定での走行比較 | `experiment_03_closed_loop/` | paired runs/summary、走行比較 report |

通常の報告は root `report.md` と実験03の比較表だけを読みます。実験01/02の主要表はそれぞれの `report.md` にあり、raw 値や図を監査するときだけ archive を開き、共通の観測や条件は `shared/` を参照します。

### 数値・metadata ファイル

「報告」は通常の人向け説明、「数値根拠」は表の検算、「再解析」は既存 CLI での offline 再実行、「監査」は raw 値からの独自集計を意味します。以下の一覧は、full mode の実験別 directory、または compact mode の各 `details.zip` を展開した中身です。root `report.md` は実験対応表と実験03の比較表を先に読むための要約で、実験01/02の詳細 CSV・IG・JSD・PNG は各 archive 側のリファレンスです。

| ファイル | 所属 | 内容 | 通常の報告での扱い |
| --- | --- | --- | --- |
| `report.md` | 全体／実験01〜03 | root は3実験の対応表・各 report へのリンク・実験03の自動比較表。各実験 directory は所属実験の要約。full の `shared/report.md` は元の詳細 report を全体用に保全したもの | 普段は root と実験03の比較表だけを読み、詳細を調べるときだけ所属する `details.zip` または full directory を確認 |
| `analysis_metadata.json` | 共通 | model/config/schema の path・SHA-256、seed、dimension、baseline、IG/closed-loop 設定 | 再現条件として必須。外部共有前に絶対 path を確認 |
| `feature_schema_expanded.csv` | 共通 | 259 index と feature 名、block、group、LiDAR angle の対応 | 入力名の根拠として必須 |
| `rollout_arrays.npz` | 共通 | 観測、方策出力、action、reward、done 等の aligned arrays | 報告には不要。offline 再解析には必須 |
| `rollout_steps.jsonl` | 共通 | 1 policy decision / 行の scalar telemetry | 現行 `load_rollout` では再解析に必須 |
| `rollout_metadata.json` | 共通 | rollout shape、seed、runtime contract、episode-start 観測、実行 provenance | 報告では metadata の補助。再解析には必須で、外部共有前に絶対 path を確認 |
| `perturbation_feature_steps.npz` | 実験01 | feature だけでなく全 feature/group/sector の step × target × baseline raw 指標 | 通常は不要。独自再集計・監査用 |
| `perturbation_feature_summary.csv` | 実験01 | feature 摂動の full-episode 統計 | 摂動の数値根拠 |
| `perturbation_group_summary.csv` | 実験01 | schema group / LiDAR sector 摂動の full-episode 統計 | group 摂動の数値根拠 |
| `ig_attributions.npz` | 実験02 | step × target × baseline × feature の signed/absolute IG と診断値 | 通常は不要。独自再集計・監査用 |
| `ig_feature_summary.csv` | 実験02 | feature ごとの full-episode signed / absolute IG 統計 | IG の数値根拠 |
| `ig_group_summary.csv` | 実験02 | group/sector の full-episode signed sum / absolute mass | group IG の数値根拠 |
| `ig_completeness.csv` | 実験02 | full-episode IG の residual と relative error | IG を報告する前の数値確認に必須 |
| `closed_loop_runs.jsonl` | 実験03 | target × scenario ごとの通常/介入 outcome と差 | 通常は不要。paired result の監査用 |
| `closed_loop_summary.csv` | 実験03 | target ごとの通常/介入 outcome と平均差 | closed-loop を報告する場合の数値根拠 |

旧 flat な `official_left_curve` の full inventory は読み取り互換のため残ります。numbered output では、root の対応表と、実験01/02/03・`shared/` の各 report/archive（compact）または個別 artifact（full）を所属ごとに確認します。

### plot ファイル

標準 plot は有効データがない場合も、固定したファイル名と「表示できない理由」を持つ placeholder として生成されます。placeholder を実験結果の 0 と解釈してはいけません。

| ファイル | 所属 | 表示するもの | 使う場面 |
| --- | --- | --- | --- |
| `plots/perturbation_feature_top.png` | 実験01 | full-episode の feature 摂動上位 | actor の入力依存度ランキング |
| `plots/perturbation_group_top.png` | 実験01 | group/sector 摂動上位 | 入力群単位の依存度 |
| `plots/ig_feature_top_absolute.png` | 実験02 | 最初の IG target の absolute feature 上位 | 寄与の大きさ |
| `plots/ig_group_top_absolute.png` | 実験02 | 最初の IG target の group / LiDAR sector absolute mass | `group_kind` と `group_size` を確認して読む寄与量 |
| `plots/ig_group_signed.png` | 実験02 | 最初の IG target の signed group / LiDAR sector IG | `group_kind` を分けて読む、target を押し上げる/下げる方向 |
| `plots/attribution_over_time.png` | 実験02 | 最初の IG target の mean absolute IG の decision ごとの推移 | 寄与が大きい decision |
| `plots/perturbation_over_time.png` | 実験01 | feature 摂動 score の decision ごとの推移 | 依存度が大きい decision |
| `plots/lidar_ig_heatmap.png` | 実験02 | LiDAR angle × decision の absolute IG | LiDAR 寄与の大きさが現れる角度と時点 |
| `plots/lidar_perturbation_heatmap.png` | 実験01 | LiDAR angle × decision の JSD | LiDAR 置換への依存度が現れる角度と時点 |
| `plots/closed_loop_performance.png` | 実験03 | target ごとの paired reward 差 | 介入後の走行性能 |
| `plots/custom_feature_attribution_over_time.png` | 実験02 | 最初の IG target の custom feature の signed IG の decision ごとの推移 | custom feature がある場合のみ。標準259 schemaでは placeholder |

各 plot の軸、CSV filter、今回の数値例、結論文の作り方は [入力寄与解析結果の読み方](input_attribution_results.md) で説明します。

## 解釈の境界

- 摂動、IG、closed-loop は異なる問いに答えるため、一つの総合 score に合成しません。
- baseline または IG target が異なる値は、同じ尺度として直接比較しません。
- baseline との混成観測や IG の補間点は学習分布外になり得ます。
- group は大きさと membership が異なり、重複 group もあるため単純加算しません。
- 1 scenario の結果を他の道路、traffic、seed、policy へ一般化しません。
- closed-loop の変化も、入力がタスク一般に必要だという因果証明ではありません。

摂動の発想は [Greydanus et al., *Visualizing and Understanding Atari Agents* (ICML 2018)](https://proceedings.mlr.press/v80/greydanus18a.html)、IG は [Sundararajan, Taly, Yan, *Axiomatic Attribution for Deep Networks* (ICML 2017)](https://arxiv.org/abs/1703.01365) に対応します。saliency を因果説明として扱わない注意は [Atrey et al., *Exploratory Not Explanatory* (2020)](https://arxiv.org/abs/1912.05743) に従います。
