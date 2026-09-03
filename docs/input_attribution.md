# PPO 入力寄与・入力依存度解析

## この実験で調べること

`analyze_input_attribution.py` は、学習済みの Stable-Baselines3 PPO を固定したまま、MetaDrive の各観測入力に方策がどの程度依存しているかを調べます。中心となる問いは次の三つです。

1. ある入力を基準値へ置き換えると、行動確率分布はどの程度変わるか。
2. 基準観測から現在観測までの出力変化を、各入力へどのように帰属できるか。
3. 重要と判定された入力を走行中も置き換えると、報酬や完走成否がどう変わるか。

これは固定済み方策の**入力依存性を調べる探索的実験**です。入力が運転タスク一般に必要であることや、入力と成功・失敗の因果関係を証明する実験ではありません。学習、model weights、`evaluate.py`、MetaDrive 本体は変更しません。

実行後の数値をどう絞り、どう報告するかは [入力寄与解析結果の読み方](input_attribution_results.md) に分離しています。観測 index の成立条件は [観測スキーマ](../observation_schemas/README.md)、262 次元環境への移植は [移植手順](input_attribution_porting.md) を参照してください。

## 実験全体の流れ

`run` サブコマンドは次の順で処理します。

```text
通常走行を収集: x_t, π(·|x_t), V(x_t), action, reward
               │
               ├─ オフライン摂動 ── 入力置換による方策分布・critic の変化
               │
               ├─ Integrated Gradients ── 出力差を入力次元へ帰属
               │
               └─ 時間区分ごとの集計・上位入力の選択
                                      │
                                      └─ paired closed-loop
                                         同じ seed の通常走行と介入走行を比較
```

各段階が答える問いは異なります。

| 段階 | 見るもの | 主な出力 | この段階だけで言えること |
| --- | --- | --- | --- |
| 通常 rollout | 介入前の走行と方策出力 | `rollout_*` | 解析対象となった入力・行動・走行の事実 |
| オフライン摂動 | 入力置換前後の方策分布 | Jensen–Shannon divergence | その置換に対する方策出力の依存度 |
| Integrated Gradients (IG) | baseline から現在点までの出力差 | signed / absolute IG | 指定した出力差を各入力へ帰属した量 |
| 時系列集計 | episode 全体、episode 別、進捗区分、明示 phase | summary CSV | 依存度・寄与度がいつ大きかったか |
| paired closed-loop | 通常走行と入力置換走行の outcome 差 | reward、success、out-of-road 等の差 | 固定方策に対する置換の走行上の影響 |

## 記号

| 記号 | 意味 |
| --- | --- |
| \(x_t\in\mathbb{R}^D\) | decision step \(t\) で、行動決定前に方策へ渡す観測 |
| \(b\in\mathbb{R}^D\) | 比較の基準とする baseline 観測 |
| \(G\) | 同時に置き換える feature index の集合 |
| \(\pi(a\mid x)\) | actor の離散行動確率 |
| \(l_a(x)\) | action \(a\) の logit |
| \(V(x)\) | critic の value estimate |
| \(a^*\) | 元の観測 \(x\) で確率最大の行動 |

本実装は single-agent、1 次元の flat `Box` 観測、単一 `Discrete` action、PPO `ActorCriticPolicy` / `MlpPolicy` を対象とします。未対応の policy・観測・action space は解析を続けずエラーにします。

## 実験 0: 通常 rollout の収集

各 scenario を決定論的方策で通常走行し、`env.step(action)` より前の \(x_t\) を float32 の copy として保存します。同じ行に logits、log probabilities、probabilities、critic value、選択 action、reward、終了状態を対応付けます。したがって保存データの 1 行は 1 simulation frame ではなく、**1 policy decision** です。

この段階は attribution ではありません。後続実験が同じ観測列を使うための測定データと、介入前走行の基準を作ります。

## 実験 1: baseline 置換によるオフライン摂動

### 何を見るか

観測の feature、schema group、または LiDAR 角度 sector だけを baseline の値へ置き換え、方策出力がどの程度変わるかを測ります。対象外の index と環境状態は変えません。

置換後の観測は

$$
\tilde{x}_{t,G,j}^{(b)} =
\begin{cases}
b_j & (j\in G),\\
x_{t,j} & (j\notin G)
\end{cases}
$$

です。元の actor 分布を \(p=\pi(\cdot\mid x_t)\)、置換後を \(q=\pi(\cdot\mid\tilde{x}_{t,G}^{(b)})\)、\(m=(p+q)/2\) とすると、主指標は自然対数の Jensen–Shannon divergence です。

$$
D_{\mathrm{JS}}(p,q)
=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)
+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m)
$$

\(D_{\mathrm{JS}}=0\) なら分布は同一で、値が大きいほど置換前後の分布差が大きいことを表します。補助指標も別々に保存します。

| 指標 | 定義 | 符号・意味 |
| --- | --- | --- |
| `action_changed` | \(\mathbf{1}[\arg\max p\ne\arg\max q]\) | 1 なら決定論的 action が変化 |
| `selected_action_probability_drop` | \(p(a^*)-q(a^*)\) | 正なら元の選択 action の確率が低下 |
| `centered_logit_l2` | \(\lVert(l'-\bar l')-(l-\bar l)\rVert_2\) | 大きいほど相対的な logit 配置が変化 |
| `value_delta` | \(V(\tilde{x})-V(x)\) | critic の予測差。実 reward の差ではない |

feature を個別に置換した値は、相関や冗長性があるため足し合わせられません。group の比較は group 全体を一度に置換する別実験であり、個別 feature の結果の合計でもありません。

### baseline の設定

baseline は摂動と IG の結論を変える実験条件です。ゼロ vector が MetaDrive の中立状態とは限らないため、標準設定では使いません。

| `baseline.strategy` | 選ばれる観測 | 主な用途 |
| --- | --- | --- |
| `episode_start` | 各 \(x_t\) と同じ episode の最初の保存観測 | 現在の標準。開始状態との差を見る |
| `specified_steps` | 指定した `episode_id` / `step` | 意味を確認した状態を基準にする |
| `sampled_observations` | rollout から seed 付きで抽出 | 複数の実在観測に対する頑健性を見る |
| `external_npz` | 外部 NPZ の指定配列 | 別途検証した基準集合を使う |

`count > 1` では raw 結果に baseline 軸を残し、summary は各 step で baseline 間を平均してから統計量を計算します。`episode_start` の `seed` は選択に使われません。

## 実験 2: Integrated Gradients

### 何を見るか

IG は「baseline \(x_0\) から現在観測 \(x\) へ変化したとき、指定した scalar 出力 \(F\) の差をどの入力へ帰属するか」を調べます。直線 path

$$
z(\alpha)=x_0+\alpha(x-x_0),\qquad 0\le\alpha\le1
$$

に対する feature \(j\) の attribution は

$$
\operatorname{IG}_j(x;x_0)
=(x_j-x_{0,j})
\int_0^1
\frac{\partial F(z(\alpha))}{\partial z_j}\,d\alpha
$$

です。実装は両端を含む \(S=\) `integrated_gradients.steps` 個の点を使い、台形則で近似します。

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

設定できる target \(F\) は次の三つです。

| `integrated_gradients.targets` | target 関数 | 調べる出力 |
| --- | --- | --- |
| `selected_log_probability` | \(F(z)=\log\pi(a^*\mid z)\) | 元の選択 action の log probability |
| `selected_vs_runner_up_margin` | \(F(z)=l_{a^*}(z)-l_{a_2}(z)\) | 元の上位 action と次点 action の margin |
| `critic_value` | \(F(z)=V(z)\) | critic value |

\(a^*\) と runner-up \(a_2\) は元の \(x\) で一度だけ決め、補間点ごとに選び直しません。環境へ action を強制する処理ではありません。

signed IG の正負は \(F(x)-F(x_0)\) への方向を表し、`mean_absolute_ig` は方向を捨てた寄与の大きさです。target の単位が違うため、actor IG と critic IG の数値を直接比較しません。

数値積分の確認には completeness residual

$$
r=\sum_j\operatorname{IG}_j-\{F(x)-F(x_0)\}
$$

を使います。残差が大きい場合は、結果を解釈する前に `steps` を増やして再計算します。

## 実験 3: 時間区分ごとの集計

summary は常に `full_episode`、episode ごと、および `aggregation.progress_bins` 個の区分を作ります。episode progress は route completion ではなく、episode 内の decision step を

$$
\rho_t=\frac{t-t_{\min}}{t_{\max}-t_{\min}}
$$

で 0–1 に正規化した値です。したがって 3 分割は相対的な序盤・中盤・終盤であり、curve entry、apex、exit を自動検出した区分ではありません。

道路上の意味を持つ区間は `[[phases]]` で明示します。step phase は `[start_step, end_step)`、progress phase は通常 `[start_progress, end_progress)` です。`end_progress = 1.0` だけは最後の decision を含みます。

## 実験 4: paired closed-loop

### 何を見るか

オフライン摂動で上位になった feature/group、または明示した target について、同じ scenario seed と RL seed で通常走行と介入走行を実行します。介入走行では env が返した観測の copy の \(G\) だけを一定値へ置き換えて方策へ渡します。env 内部状態、model weights、reward、done、および方策が返した action は改変しません。

各 outcome \(M\) の保存値は

$$
\Delta M_G
=M_{\mathrm{intervention},G}
-M_{\mathrm{baseline}}
$$

です。例えば `mean_delta_total_reward < 0` は介入走行の平均 reward が通常走行より低く、`mean_delta_success = -1` は通常成功・介入失敗だったことを表します。

| `closed_loop.replacement_strategy` | 介入中に固定する値 |
| --- | --- |
| `episode_start_constant` | 同じ scenario の reset 観測 |
| `specified_reference_constant` | 最初の解決済み baseline reference。reference ID は metadata で確認 |
| `dataset_median_constant` | 保存 rollout の feature ごとの median |
| `schema_constant` | schema に明示した定数 |

`run` で明示 target がない場合、`full_episode` / `mean_over_baselines` の `mean_js_divergence` 上位から `top_k_features` と `top_k_groups` を選びます。`top_k_groups` の候補は schema group と LiDAR sector の混合順位です。semantic group だけを固定して検証したい場合は `explicit_groups` を使います。いずれも IG の順位から選ぶ処理ではありません。

paired closed-loop は、入力置換が固定方策の挙動へ波及したかを確かめます。それでも Remove-and-Retrain の効果や、元の入力がタスク一般に不可欠であることは示しません。

## 現在の公式サンプルで実際に行う実験

以下は `configs/official.toml`、`attribution_configs/official_left_curve.toml`、`observation_schemas/metadrive_default_259.toml` と、現在保存されている run を照合した値です。

| 項目 | 現在の設定・実測値 |
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
| 時間集計 | full episode 1 + episode 1 + progress bins 3 = 5 slices、明示 phase なし |
| closed-loop | 摂動上位 10 features + group/sector 混合順位の上位 5 件、1 scenario に対する 15 paired comparisons。今回選ばれた 5 件は全て schema group |
| 保存設定 | observation 保存は必須で有効、simulator recording は無効。静的な標準 plot は生成 |

重要なのは、`official_left_curve` は analysis config と出力 prefix の**名前**にすぎないことです。現在の `phases` は空で、左カーブ区間を選ぶ設定はありません。現状の結果は seed 5 の episode 全体を解析した 1 scenario の case study であり、左カーブ固有の結果や seed 間統計として報告してはいけません。

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
| 時間分割 | analysis TOML | `[aggregation].progress_bins` / `[[phases]]` | summary の行と区間 |
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
  --output-prefix official_attribution_multiseed
```

複数 scenario にすると結果の一般化可能性は改善しますが、同一設定内での seed 範囲に限った評価です。

### 意味を確認した区間を phase にする

```bash
cp attribution_configs/official_left_curve.toml \
  attribution_configs/verified_curve_interval.toml
```

`rollout_steps.jsonl` や走行可視化で区間を確認してから、copy の先頭にある analysis 名を `name = "verified_curve_interval"` へ変更し、次の phase をファイル末尾へ追加します。

```toml
[[phases]]
name = "verified_curve_interval"
start_step = 20
end_step = 60
```

20 と 60 は書式例であり、現在の走行における左カーブ境界を確認した値ではありません。確認していない区間へ `curve` や `apex` という名前を付けないでください。実行時は通常の `run` コマンドの `--analysis-config` と `--output-prefix` を、この新しい名前へ置き換えます。

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

現在の実装には「summary だけ保存する」artifact profile はありません。method を無効にしても互換性のため空 CSV/NPZ や placeholder plot は残ります。設定で減るものと減らないものを区別してください。

| 変更 | 減るもの | 失う情報・残るもの |
| --- | --- | --- |
| IG targets を 2→1 | IG 計算量と summary 行数が約半分 | 外した target の解釈を失う |
| IG `steps` を 64→32 | IG の勾配計算量 | raw 配列・summary の行数はほぼ変わらず、積分誤差が増える可能性 |
| `progress_bins = 1` | 現例では 5→3 slices | 時間変化の解像度を失う |
| `analyze_features = false` | feature ごとの摂動 | IG feature summary と LiDAR sectors は残る |
| `analyze_groups = false` | schema group の摂動 | LiDAR sectors は残る |
| `lidar_sector_degrees = 360` | 24 sectors→1 sector | 角度分解能を失う。sector の完全無効化設定はない |
| `closed_loop.enabled = false` | paired env 実行 | 空の closed-loop ファイルと placeholder plot は残る |
| closed-loop top-K を小さくする | paired episode 数 | offline 摂動・IG の量は変わらない |

現在の 26 ファイルは合計 2,592,845 bytes、約 2.47 MiB です。保存容量は大きくありませんが、人が読むには過剰です。全成果物を報告書へ添付する必要はありません。通常の提出物、数値根拠、再解析 archive の分け方は [結果の読み方](input_attribution_results.md#出力が多い理由と配布範囲) に示します。本当に生成ファイル自体を減らすには、raw archive、summary scope、plot の保存可否を選ぶコード変更と出力契約テストの更新が別途必要です。

## 実行手順

project root、つまり `metadrive_rl-main/` で実行します。最初に schema と実環境・model の shape を検証します。

```bash
.venv/bin/python analyze_input_attribution.py validate-schema \
  --config configs/official.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml
```

検証後に一式を実行します。

```bash
.venv/bin/python analyze_input_attribution.py run \
  --config configs/official.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --output-prefix official_left_curve
```

Windows PowerShell では行継続をバッククォートへ変えます。

```powershell
.\.venv\Scripts\python.exe .\analyze_input_attribution.py run `
  --config .\configs\official.toml `
  --model .\models\official_baseline.zip `
  --schema .\observation_schemas\metadrive_default_259.toml `
  --analysis-config .\attribution_configs\official_left_curve.toml `
  --output-prefix official_left_curve
```

| サブコマンド | 用途 | MetaDrive 実行 | 保存物 |
| --- | --- | --- | --- |
| `validate-schema` | env / reset / model / schema の dimension 契約を確認 | reset 1 回 | result directory は作らず、検証 JSON を標準出力 |
| `collect` | 通常 rollout だけ収集 | あり | rollout 3 ファイル + schema CSV の計 4 ファイル |
| `run` | 収集、offline 解析、plot/report、設定時 closed-loop | あり | 現行設定では、下表の 15 ファイル + PNG 11 枚の計 26 ファイル |
| `analyze` | 保存 rollout を別設定で再解析 | なし。closed-loop は行わない | rollout をコピーし、`run` と同名の計 26 ファイル。closed-loop 表は空、対応 plot は placeholder |
| `closed-loop` | 明示した feature/group の paired intervention だけ実行 | あり | metadata、schema、report、closed-loop 2 ファイル、PNG 11 枚の計 16 ファイル。rollout/摂動/IG は保存しない |

保存済み rollout の再解析例です。元 directory は上書きせず、別 prefix を指定します。

```bash
.venv/bin/python analyze_input_attribution.py analyze \
  --config configs/official.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --rollout outputs/official/attribution/official_left_curve \
  --output-prefix official_left_curve_reanalysis
```

`analyze` と、`--rollout` を指定する `closed-loop` は、保存 rollout の model/config/schema hash などが現在の指定と一致しなければ拒否します。新しく収集する `run` は、実環境・model・schema の observation/action dimension を検証します。各出力は staging directory で作成し、全処理が成功した場合だけ同名の結果 directory を置き換えます。

## 保存先

保存先は

```text
metadrive_rl-main/outputs/<experiment-name>/attribution/<output-prefix>/
```

です。サンプルコマンドでは `experiment-name = official`、`output-prefix = official_left_curve` なので、質問にある

```text
metadrive_rl-main/outputs/official/attribution/official_left_curve/
```

が 1 回の完全な結果 directory です。`outputs/official/attribution/` は複数 prefix を置く親 directory であり、それ自体が一つの結果ではありません。`outputs/` は通常 Git 管理外です。

### 数値・metadata ファイル

「報告」は通常の人向け説明、「数値根拠」は表の検算、「再解析」は既存 CLI での offline 再実行、「監査」は raw 値からの独自集計を意味します。

| ファイル | 内容 | 通常の報告での扱い |
| --- | --- | --- |
| `report.md` | 設定と full-episode 上位値をまとめた自動概要 | 最初に読む。ただし単独で結論にせず、外部共有前に絶対 path を確認 |
| `analysis_metadata.json` | model/config/schema の path・SHA-256、seed、dimension、baseline、IG/closed-loop 設定 | 再現条件として必須。外部共有前に絶対 path を確認 |
| `feature_schema_expanded.csv` | 259 index と feature 名、block、group、LiDAR angle の対応 | 入力名の根拠として必須 |
| `rollout_arrays.npz` | 観測、方策出力、action、reward、done 等の aligned arrays | 報告には不要。offline 再解析には必須 |
| `rollout_steps.jsonl` | 1 policy decision / 行の scalar telemetry | 区間確認用。現行 `load_rollout` では再解析にも必須 |
| `rollout_metadata.json` | rollout shape、seed、runtime contract、episode-start 観測、実行 provenance | 報告では metadata の補助。再解析には必須で、外部共有前に絶対 path を確認 |
| `perturbation_feature_steps.npz` | feature だけでなく全 feature/group/sector の step × target × baseline raw 指標 | 通常は不要。独自再集計・監査用 |
| `perturbation_feature_summary.csv` | feature 摂動の scope 別統計 | 摂動の数値根拠 |
| `perturbation_group_summary.csv` | schema group / LiDAR sector 摂動の scope 別統計 | group 摂動の数値根拠 |
| `ig_attributions.npz` | step × target × baseline × feature の signed/absolute IG と診断値 | 通常は不要。独自再集計・監査用 |
| `ig_feature_summary.csv` | feature ごとの signed / absolute IG 統計 | IG の数値根拠 |
| `ig_group_summary.csv` | group/sector の signed sum / absolute mass | group IG の数値根拠 |
| `ig_completeness.csv` | IG の residual と relative error | IG を報告する前の数値確認に必須 |
| `closed_loop_runs.jsonl` | target × scenario ごとの通常/介入 outcome と差 | 通常は不要。paired result の監査用 |
| `closed_loop_summary.csv` | target ごとの通常/介入 outcome と平均差 | closed-loop を報告する場合の数値根拠 |

ファイル数は上のサブコマンド表の通りです。無効な解析の空ファイルや placeholder も、固定した出力契約の一部として数えています。

### plot ファイル

標準 plot は有効データがない場合も、固定したファイル名と「表示できない理由」を持つ placeholder として生成されます。placeholder を実験結果の 0 と解釈してはいけません。

| ファイル | 表示するもの | 使う場面 |
| --- | --- | --- |
| `plots/perturbation_feature_top.png` | full-episode の feature 摂動上位 | actor の入力依存度ランキング |
| `plots/perturbation_group_top.png` | group/sector 摂動上位 | 入力群単位の依存度 |
| `plots/ig_feature_top_absolute.png` | 最初の IG target の absolute feature 上位 | 寄与の大きさ |
| `plots/ig_group_top_absolute.png` | 最初の IG target の group / LiDAR sector absolute mass | `group_kind` と `group_size` を確認して読む寄与量 |
| `plots/ig_group_signed.png` | 最初の IG target の signed group / LiDAR sector IG | `group_kind` を分けて読む、target を押し上げる/下げる方向 |
| `plots/attribution_over_time.png` | 最初の IG target の mean absolute IG 時系列 | 寄与が大きい時点 |
| `plots/perturbation_over_time.png` | feature 摂動 score の時系列 | 依存度が大きい時点 |
| `plots/lidar_ig_heatmap.png` | LiDAR angle × decision の absolute IG | LiDAR 寄与の大きさが現れる角度と時点 |
| `plots/lidar_perturbation_heatmap.png` | LiDAR angle × decision の JSD | LiDAR 置換への依存度が現れる角度と時点 |
| `plots/closed_loop_performance.png` | target ごとの paired reward 差 | 介入後の走行性能 |
| `plots/custom_feature_attribution_over_time.png` | schema で `kind = "custom"` の feature 時系列 | custom feature がある場合のみ |

各 plot の軸、CSV filter、今回の数値例、結論文の作り方は [入力寄与解析結果の読み方](input_attribution_results.md) で説明します。

## 解釈の境界

- 摂動、IG、closed-loop は異なる問いに答えるため、一つの総合 score に合成しません。
- baseline または IG target が異なる値は、同じ尺度として直接比較しません。
- baseline との混成観測や IG の補間点は学習分布外になり得ます。
- group は大きさと membership が異なり、重複 group もあるため単純加算しません。
- 1 scenario の結果を他の道路、traffic、seed、policy へ一般化しません。
- closed-loop の変化も、入力がタスク一般に必要だという因果証明ではありません。

摂動の発想は [Greydanus et al., *Visualizing and Understanding Atari Agents* (ICML 2018)](https://proceedings.mlr.press/v80/greydanus18a.html)、IG は [Sundararajan, Taly, Yan, *Axiomatic Attribution for Deep Networks* (ICML 2017)](https://arxiv.org/abs/1703.01365) に対応します。saliency を因果説明として扱わない注意は [Atrey et al., *Exploratory Not Explanatory* (2020)](https://arxiv.org/abs/1912.05743) に従います。
