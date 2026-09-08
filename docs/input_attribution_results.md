# PPO 入力寄与・入力依存度の結果を読む・報告する

この文書は、`analyze_input_attribution.py run` が作る結果を、人に説明・報告するための読み方です。全体像を先に確認する場合は [入力寄与解析の概要](input_attribution_overview.md) を参照してください。概要は目的・入力・方法と、実行後に見る表の形を平易な文章で説明し、方法・設定の詳細は [入力寄与解析ガイド](input_attribution.md)、この文書は列・数式のリファレンスを扱います。生成先は通常 `outputs/<experiment>/attribution/<output-prefix>/` です。root `report.md` は準備と実験01/02/03の対応表、各実験 `report.md` へのリンク、実験03の走行比較表を含みます。compact では各実験 directory に `report.md` + `details.zip`、`shared/` に `details.zip`、full では同じ directory に CSV/NPZ/JSONL/PNG/metadata を個別保存します。生成物は Git 管理外で別の環境には存在しないため、この文書では結果ファイルへの直接リンクは張りません。

普段は root `report.md` の対応表と、実験03 `report.md` の比較表だけを読めば十分です。表は最小限の実験条件と、実際に行った通常走行・各介入走行を次の列で示します。総報酬は走行中の評価点（reward）の合計、差分は「入力固定 − 通常」で、負なら入力固定後に低下したことを表します。複数 scenario の総報酬は走行ごとの合計の平均、完走と道路外への逸脱は率です。結果が取得できなかった値は 0 ではなく `未記録`、介入を実行していない項目は `未実施` として扱います。JSD・IG・PNG・raw CSV まで調べる必要がある場合だけ所属実験の `details.zip` を展開してください。

| 走らせ方（実験03） | 総報酬 | 完走 | 道路外への逸脱 |
| --- | ---: | ---: | ---: |
| 通常走行、各介入走行（1行ずつ） | scenario 平均 | scenario 間の完走率 | scenario 間の逸脱率 |

scenario が1つだけなら、完走と逸脱はその走行の成否（0 または 1）です。複数 scenario では同じ列が率になり、例えば完走率 `0.8` は5走行中4走行の完走を表します。

## 実験番号と保存先

| 所属 | 目的 | 保存先 | 主に読むもの |
| --- | --- | --- | --- |
| 準備：通常走行の収集 | 実験01/02/03が共有する観測・方策出力・報酬を記録 | `shared/` | 共通 metadata、schema、rollout（compact は共通 `details.zip`、full は個別 raw と保全用 `report.md`） |
| 実験01 入力置換（オフライン摂動） | 入力置換による方策確率の変化を測定し、実験03の候補をJSDで選ぶ | `experiment_01_perturbation/` | `report.md`、摂動 summary、JSD raw |
| 実験02 Integrated Gradients（出力変化の入力への割当） | 指定 output の変化を各入力へ割り当てる。実験03の候補選択には使わない | `experiment_02_integrated_gradients/` | `report.md`、IG summary、completeness、IG raw |
| 実験03 入力固定での走行比較（paired closed-loop） | 通常走行と入力固定走行の総報酬・完走・逸脱を比較 | `experiment_03_closed_loop/` | `report.md`、`closed_loop_summary`、paired run |

通常の報告では root `report.md` と実験03の4列比較表だけを読みます。実験01/02の値や監査用 raw を確認するときだけ、所属する archive または full directory と `shared/` を参照します。

compact の ZIP は各 member path が run root 相対です。4 ZIP はそれぞれの実験 directory へ個別に展開せず、同じ新しい run root へ復元します。例えば `experiment_01_perturbation/details.zip` には `experiment_01_perturbation/perturbation_feature_summary.csv`、`shared/details.zip` には `shared/rollout_arrays.npz` が入ります。実験01/02の主要表は各 `report.md` にあり、数値監査が必要な場合だけ ZIP を確認します。`shared/details.zip` を復元した run root は `analyze --rollout` に指定できます。旧 flat root の `details.zip` は旧構造のまま保持し、自動展開しません。

## 先に結論

この解析は、**固定済みの PPO 方策が、ある観測入力を設定した baseline（現行 `official_left_curve` では episode 開始時の値）へ置き換えられたとき、どの程度出力・走行結果を変えるか**を調べる実験です。学習済みモデルの重要度を一般に証明する実験でも、環境中の物理的な因果を証明する実験でもありません。

答える問いは三つに分かれます。

1. **実験01 入力置換（オフライン摂動）**: ある入力または入力群を基準値に置換すると、方策の行動確率分布はどれだけ変わるか。
2. **実験02 Integrated Gradients（出力変化の入力への割当）**: 基準観測から現在観測へ移る経路上で、方策の特定の出力をどの入力が増減させたか。
3. **実験03 入力固定での走行比較（paired closed-loop）**: 実験01のJSD上位、または明示した入力を、同じ開始条件の走行で固定すると、固定方策の報酬・成功・逸脱はどう変わるか。

したがって、報告の主語は「この checkpoint は、この基準値・scenario・入力置換の下で、入力に依存した」です。「この入力が運転の本質的原因である」「LiDAR は不要である」のような一般化はしません。

## 詳細指標を調べる場合の読順

|順番|最初に見るもの|ここで確定すること|通過しなければ|
|---|---|---|---|
|1|`report.md`|最小条件と、通常走行・全介入を並べた比較表|普段の報告はこの表を読む。詳細な帰属値が必要なら下記へ進む|
|2|`analysis_metadata.json`|checkpoint/config/schema の hash、seed、観測・action 次元、baseline、IG target、closed-loop の置換法|数値を比較・引用しない|
|3|`feature_schema_expanded.csv`|各 index の意味、block、所属 group、LiDAR 角度、`resolved`|feature 名だけで意味を推測しない|
|4|実験01の摂動 2 CSV|方策分布を変えた feature / group を、全観測の集計結果（full-episode summary）で、同一 baseline を使って順位付けする|JSD を実験02のIGや実験03の実報酬と混ぜない|
|5|実験02の IG 3 CSV|target ごとの寄与の向き・大きさと、数値積分の健全性を確認する|actor と critic の IG を混ぜない。実験03の候補選択には使わない|
|6|実験03の `closed_loop_summary.csv` と必要な PNG|候補入力の置換が paired 走行をどう変えたかを確認する|1 scenario の結果を一般化しない|

PNG は詳細調査や発表用の概観、CSV は引用する数値の正本、NPZ/JSONL は監査・再集計用です。普段の報告は自動生成 `report.md` の比較表で完結し、JSD・IG の順位や値を調べる場合だけ対応する CSV と metadata に戻って確認します。

## 詳細を調べる場合のリファレンス

以下の metadata、schema、CSV、IG、JSD、PNG の説明は、compact の比較表だけでは足りない検証・再解析のためのリファレンスです。

## 数値を読む前の metadata / schema gate

同じ名前の output directory でも、以下が異なれば同じ実験とは扱いません。`analysis_metadata.json` の値を報告書の「実験条件」表に転記します。

|確認項目|見る field|通過条件|
|---|---|---|
|入力・action 契約|`runtime_contract`|`env`、model、`reset`、schema の observation dimension が全て一致し、model と env の action dimension も一致する|
|解析対象|`model.sha256`、`experiment.sha256`、`schema.sha256`、`analysis_config.sha256`|比較する run 間で、意図した変更以外は同じ hash である|
|標本|`scenario_seed_range`、`rl_seed`、`rollout.scenario_seeds`、`rollout.row_count`|scenario 数・seed・policy decision 数を明記できる|
|基準観測|`baseline.strategy`、`baseline.count`、選択 row|同じ baseline 定義である。異なれば JSD/IG の大小を横比較しない|
|手法|`perturbation`、`integrated_gradients.targets`、`integrated_gradients.steps`|同じ target、積分点数であることを確認する|
|closed-loop|`closed_loop_provenance`、`closed_loop_targets`|同じ置換戦略・対象・paired seed である|

現行の `official_left_curve` では、runtime contract の observation dimension は全て **259**、action dimension は **9** です。scenario seed は **5 が 1 本**、RL seed は **0**、記録された policy decision は **127** です。`feature_schema_expanded.csv` は 259 行の index-to-meaning の正本であり、`groups` 列を見て重複所属も確認します。例えば `normalized_speed` は `ego_state` と `speed` の両方に入ります。

`analysis_metadata.json`、`rollout_metadata.json`、自動生成 `report.md` には、実行機の絶対 path が入ることがあります。外部共有用には path を相対表記または `<project-root>/...` に置換した**複製**を作り、内部監査用の原本は hash を含めて別保管してください。パスを伏せても checkpoint/config/schema の SHA-256、バージョン、seed は残します。

## この実験で実際にしていること

観測を $x \in \mathbb{R}^D$、固定した方策を $\pi(\cdot\mid x)$、critic を $V(x)$、基準観測を $b$ と書きます。現行 schema は $D=259$ で、ego state 9 次元、navigation 10 次元、LiDAR 240 次元です。

### 実験01：入力置換（オフライン摂動）

feature または group の index 集合を $G$ とすると、解析は環境を再実行せず、保存済みの各観測について次だけを変えます。

$$
\tilde{x}_i =
\begin{cases}
b_i & (i \in G)\\
x_i & (i \notin G)
\end{cases}
$$

すなわち、元の $x$ と基準 $b$ のうち $G$ だけを入れ替え、$p=\pi(\cdot\mid x)$ と $q=\pi(\cdot\mid\tilde{x})$ を同じ固定方策で比べます。元の環境状態、reward、保存済み action は変えません。

主指標は自然対数を使う Jensen–Shannon divergence（JSD）です。

$$
m=\frac{p+q}{2},\qquad
\operatorname{JSD}(p,q)=\frac{1}{2}\operatorname{KL}(p\parallel m)+
\frac{1}{2}\operatorname{KL}(q\parallel m)
$$

単位は **nats**、範囲は $[0,\ln 2]$ です。0 はこの置換で actor の確率分布が変わらなかったことを表します。大きいほどこの baseline replacement に対する actor 出力変化が大きい、という意味であり、重要度の普遍的な閾値はありません。モデル、action 数、baseline、scenario が違う JSD に共通の「大きい」基準はありません。

### 実験02：Integrated Gradients（出力変化の入力への割当）

IG は baseline から現在観測までの直線経路

$$
z(\alpha)=b+\alpha(x-b),\quad 0\leq\alpha\leq1
$$

に沿って、target $F$ の各入力への寄与を計算します。

$$
\operatorname{IG}_i(x;b)=
(x_i-b_i)\int_0^1
\frac{\partial F(z(\alpha))}{\partial z_i}\,d\alpha
$$

実装は両端を含む台形則です。現行 run は 64 点です。`selected_log_probability` の target は、元の $x$ で一度だけ選んだ $a^*=\arg\max_a\pi(a\mid x)$ を経路中で固定した

$$
F(z)=\log\pi(a^*\mid z)
$$

です。単位は log probability の **nats** です。`critic_value` は $F(z)=V(z)$ で、単位は critic が予測する return（学習 reward の尺度）です。二つの target の IG を同じ数値尺度や順位として混ぜてはいけません。

signed IG の正値は、baseline から現在観測へ進む経路上でその target を上げる向き、負値は下げる向きです。これは局所勾配そのものでも、環境上の因果効果でもありません。`mean_absolute_ig` は符号を捨てた順位用の大きさであり、「正に支持した」ことは示しません。

### 実験03：入力固定での走行比較（paired closed-loop）

closed-loop では target $G$ を各 decision で参照値に置換して方策へ渡し、その方策 action を環境へ渡します。通常走行と介入走行は同じ scenario seed と RL seed から開始します。現行 run の `episode_start_constant` は、各介入 episode の reset observation から参照値を作ります。

各 CSV の差分は

$$
\Delta M=M_{\text{intervention}}-M_{\text{baseline}}
$$

です。従って、`mean_delta_total_reward < 0` は介入後の平均総報酬が低い、`mean_delta_success = -1` は完走率が 1 減った（この 1 scenario では成功から失敗になった）、`mean_delta_out_of_road = +1` は道路外への逸脱率が 1 増えたことを示します。複数 scenario では、総報酬は scenario ごとの総報酬の平均、完走と逸脱は scenario 間の率として集計されるため、例えば `-0.2` は20ポイントの率の低下です。値が取れない場合は数値の 0 ではなく `未記録`、介入を走らせていない場合は `未実施` です。

これは「この**固定方策への入力置換**が、同一開始条件からの経路を変えた」ことの paired evidence です。最初の action 以降は環境軌跡も変わるため、個々の入力が事故や成功を一般に**因果した**とは結論しません。入力が学習分布外の hybrid observation になる可能性もあります。

## 保存済み `official_baseline.zip` の結果例（`official_left_curve`）

|項目|現行値|解釈への影響|
|---|---|---|
|環境|`configs/official.toml` の common 環境設定: map `C`、traffic density 0、discrete 3 x 3 action、horizon 500|交通・道路条件を変えた結果ではない|
|標本|scenario seed 5 を 1 本、127 decision|分散・信頼区間は推定できない|
|方策|固定済み `official_baseline.zip`、deterministic、CPU|再学習や stochastic action の比較ではない|
|baseline|`episode_start`、count 1（episode 1 の最初の保存 row）|現在値を「開始時の値」に戻す比較|
|摂動|259 feature、12 semantic group、15 度ごとの 24 LiDAR sector|feature と多次元 group は同じ単位の順位ではない|
|IG|`selected_log_probability` と `critic_value`、台形則 64 点|target ごとに別表として読む|
|closed-loop|摂動上位 10 feature と group/sector 混合順位の上位 5 件、計 15 intervention、`episode_start_constant`。実際に選ばれた 5 件は全て schema group|候補選択後の 1 scenario 検証である|

`official_left_curve` は analysis config と output の**名前**です。現行 run は scenario seed 5 の episode 全体を解析した 1 scenario の case study であり、名前だけを根拠に道路形状の結果として報告しません。

scenario 数、baseline、解析対象などを変える場所と実行例は、[入力寄与解析ガイドの「どこを変えるか」](input_attribution.md#どこを変えるか) に集約しています。条件を変えた run は旧 run と混ぜず、metadata の hash、変更 field、新しい実験名を報告してください。

## CSV の正しい抽出条件

報告用の全観測要約は、次の filter を**同時に**使います。`baseline_index` が空欄の `mean_over_baselines` 行から、`scope = full_episode` かつ `slice_name = all` の行を選びます。複数 baseline を使った run では、`mean_over_baselines` はまず各 decision で baseline 軸を平均してから、mean/median/p90 を集計した行です。標準 CLI は individual-baseline 行を summary CSV に出しません。baseline ごとのばらつきが必要なら raw NPZ の baseline 軸を独自集計するか、`include_baseline_rows` を設定へ露出するコード変更が必要です。

|目的|ファイル|必須 filter|順位・引用する列|
|---|---|---|---|
|feature の actor 依存度|`perturbation_feature_summary.csv`|`scope = full_episode`、`slice_name = all`、`baseline_scope = mean_over_baselines`、`target_kind = feature`|`mean_js_divergence` の降順。併記: `count`、`target_size`、必要なら `action_flip_rate`|
|group / sector の actor 依存度|`perturbation_group_summary.csv`|同上。semantic group だけなら `target_kind = group`、LiDAR sector だけなら `target_kind = lidar_sector`|`mean_js_divergence` の降順。必ず `target_size` を併記|
|actor の IG|`ig_feature_summary.csv`|`scope = full_episode`、`slice_name = all`、`baseline_scope = mean_over_baselines`、`target = selected_log_probability`|`mean_absolute_ig` の降順。符号の主張には `mean_signed_ig` と各 rate を使う|
|critic の IG|`ig_feature_summary.csv`|同上、ただし `target = critic_value`|actor と別表で `mean_absolute_ig` を読む|
|group / sector の IG|`ig_group_summary.csv`|同上に加え、対象 target を一つに固定。必要なら `group_kind = group` または `lidar_sector`|大きさは `group_absolute_mass`、向きは `group_signed_ig`。`group_size` と `group_indices` を必ず確認|
|IG 数値誤差|`ig_completeness.csv`|`scope = full_episode`、`slice_name = all`、`baseline_scope = mean_over_baselines`、target を一つに固定|`mean_relative_completeness_error`、`max_relative_completeness_error` と absolute residual|
|paired 走行|`closed_loop_summary.csv`|`replacement_strategy` を固定し、`target_name`/`target_kind` ごとに読む。`scenario_count` を必ず併記|`mean_delta_*` と baseline/intervention の両方の値|

今の run では full-episode 行の `count` はすべて **127** です。

CSV は feature/group ごと、IG target ごとに full-episode summary を一行ずつ持ちます。通常の出力（baseline ごとの個別行を含めない設定）では、header を除くデータ行数は次式になります。

$$
\begin{aligned}
&\text{摂動 feature}=F, &&\text{摂動 group}=G,\\
&\text{IG feature}=F T, &&\text{IG group}=G T,\\
&\text{IG completeness}=T.
\end{aligned}
$$

ここで $F$ は feature 数、$G$ は semantic group と LiDAR sector を合わせた数、$T$ は IG target 数です。現行設定では $F=259$、$G=36=12+24$、$T=2$ なので、新たに生成する CSV の header を除く期待データ行数は摂動 feature 259、摂動 group 36、IG feature 518、IG group 72、completeness 2 です（header を含めると順に 260、37、519、73、3）。

## 指標の符号・単位・比較してはいけないもの

### 摂動 CSV

|列|定義・単位|符号の読み方|比較禁止 / 注意|
|---|---|---|---|
|`mean_js_divergence`|$\operatorname{JSD}(p,q)$、nats、$[0,\ln2]$|非負。大きいほど actor 分布の変化が大きい|feature の JSD を足さない。baseline、action space、scenario が違う run を閾値比較しない|
|`action_flip_rate`|$\operatorname{mean}\,\mathbf{1}[a(x)\ne a(\tilde{x})]$、割合|大きいほど deterministic action が変わる decision が多い|確率分布の小さな変化を見落とす。JSD の代わりに順位付けしない|
|`mean_selected_action_probability_drop`|$p(a^*)-q(a^*)$、確率点|正なら元の選択 action の確率が下がった。負なら上がった|絶対値列と混同しない。action 自体が変わる場合にも元の $a^*$ を追う|
|`mean_centered_logit_l2`|各 action vector を平均中心化後の L2、logit 尺度|非負|action 確率は raw logit の共通加算で不変なので、raw-logit L2 を主指標にしない。確率点や報酬と比較しない|
|`mean_value_delta`|$V(\tilde{x})-V(x)$、critic return 尺度|負なら置換後に予測 value が低い|実際の総報酬ではない。closed-loop の reward delta と同一視しない|

### IG CSV

|列|定義・単位|符号の読み方|比較禁止 / 注意|
|---|---|---|---|
|`mean_signed_ig`|feature の signed IG。単位は target と同じ（actor は nats、critic は return 尺度）|正は $F(x)-F(b)$ を増やす向き、負は減らす向き|target が違う値を足したり順位比較したりしない|
|`mean_absolute_ig`|$\operatorname{mean}\lvert\operatorname{IG}_i\rvert$、target と同じ単位|符号なしの大きさ|「正の寄与」ではない。baseline/path が違う run と直接比較しない|
|`group_signed_ig`|group 内 signed IG の和を各 decision で取り、その平均|group 全体の符号付き方向|正負の打消しを含む。absolute mass と混同しない|
|`group_absolute_mass`|group 内 $\sum_i \lvert\operatorname{IG}_i\rvert$ の平均|非負の大きさ|group が大きいほど増えやすい。feature と同じ順位に混ぜない|
|`normalized_absolute_mass`|LiDAR sector のみ。各 decision の LiDAR 全 absolute mass に対する比|無次元の比|同一 target・full-episode summary 内の sector 比較に限る。LiDAR 全 mass が 0 の decision は実装上 0 と記録される|

semantic group は一般に排他的 partition ではありません。現行 schema では `speed` は `ego_state` に含まれ、`checkpoint_1` は `navigation` に含まれ、LiDAR には `lidar` と `lidar_all` の重なりがあります。`group_size` と `group_indices` を示さずに group score を「入力群の割合」とは呼ばないでください。重複 group の score、feature score、JSD、IG、closed-loop delta はいずれも加算できません。15 度 LiDAR sector 同士だけは LiDAR index を排他的に分けていますが、それでも semantic group との二重計上は避けます。

## IG completeness は何を保証し、何を保証しないか

IG の数値積分が target の差をどれだけ再現したかを、signed IG で確認します。

$$
\varepsilon=\sum_i\operatorname{IG}_i-\left(F(x)-F(b)\right),
\qquad
r=\frac{\lvert\varepsilon\rvert}{\max\left(\lvert F(x)-F(b)\rvert,10^{-12}\right)}
$$

`completeness_residual` は $\varepsilon$、`relative_completeness_error` は $r$ です。absolute IG の和でこの検査をしてはいけません。$F(x)-F(b)$ がほぼ 0 のときは相対誤差が大きく見えるため、relative と absolute residual を一緒に見ます。

現行 full-episode 集計（各 127 decision）では次の通りです。

|IG target|平均 relative error|最大 relative error|
|---|---:|---:|
|`selected_log_probability`|`1.69177e-4`|`0.00342662`|
|`critic_value`|`9.07602e-5`|`0.000417515`|

これはこの 64 点台形則の数値診断であり、IG の意味的妥当性・因果性・モデル性能の合格判定ではありません。用途横断の universal threshold はありません。steps や baseline を変えたときは、同じ target・full-episode summary でこの値の変化を確認します。

## 11 枚の標準 PNG を一枚ずつ読む

すべての PNG は欠損時にも placeholder として作られます。画像に「データなし」の理由が書かれている場合は、結果ではなく実行状態の表示です。数字の引用元は常に対応する CSV です。

|PNG|所属|表示しているもの|正しい読み方|
|---|---|---|---|
|`perturbation_feature_top.png`|実験01|full episode・baseline 平均の feature JSD 上位 20|actor 分布を最も変えた単一 feature の概観。`perturbation_feature_summary.csv` で値と `count` を確認する|
|`perturbation_group_top.png`|実験01|semantic group と LiDAR sector を含む JSD 上位 20|`target_kind` と `target_size` を必ず併記する。group と sector を一つの重要度順位にしない|
|`ig_feature_top_absolute.png`|実験02|最初に設定した IG target の feature `mean_absolute_ig` 上位 20|現行では actor の `selected_log_probability` のみ。critic の図ではない|
|`ig_group_top_absolute.png`|実験02|最初の IG target の semantic group / LiDAR sector `group_absolute_mass` 上位 20|`group_kind` と `group_size` を確認して大きさを読む。重複を確認し、符号は読まない|
|`ig_group_signed.png`|実験02|最初の IG target の semantic group / LiDAR sector `group_signed_ig` 上位 20（絶対値で選出）|`group_kind` を分け、0 より右/左で方向を見る。打消しがあり得るため absolute mass とセットで読む|
|`attribution_over_time.png`|実験02|最初の IG target について、baseline 平均後の signed IG の feature 次元平均絶対値を各 decision に表示|decision ごとの変化の概観。特定 feature の寄与を示すものではない|
|`perturbation_over_time.png`|実験01|各 decision の全単一 feature JSD の平均|どの decision で baseline replacement に敏感かの概観。group は含まない|
|`lidar_ig_heatmap.png`|実験02|最初の IG target の decision x LiDAR angle ごとの absolute IG|angle は schema の走査方向であり、道路座標の「前/左」を自動的には意味しない|
|`lidar_perturbation_heatmap.png`|実験01|decision x LiDAR feature ごとの JSD|sector 集計ではなく単一 LiDAR ray の置換結果。schema の index/angle を使って読む|
|`closed_loop_performance.png`|実験03|各 selected target の平均 $\Delta$ total reward|赤の負値は介入後に悪化。成功・逸脱・scenario 数は CSV で併読する|
|`custom_feature_attribution_over_time.png`|実験02|`kind = custom` feature の最初の IG target に対する decision ごとの signed 推移|現行 schema には custom feature がないため placeholder が正しい。LiDAR や state の図ではない|

## 現行結果の worked example

まず条件を一文で固定します。「固定 `official_baseline` を、map `C`・traffic 0・scenario seed 5 の 1 episode（127 decision）で deterministic に走らせ、各入力を episode 開始時の 1 観測に戻して解析した」です。

次に、同じ full-episode / baseline-mean filter の数値だけを使います。

|観測|値|この条件の下で言えること|
|---|---:|---|
|摂動 `normalized_speed`|`mean_js_divergence = 0.0262685` nats|speed 1 次元を開始時値にすると、平均でこの大きさだけ actor 分布が変わった|
|摂動 `navigation`|`mean_js_divergence = 0.0863061` nats、`target_size = 10`|navigation の 10 次元を同時に置換すると actor 分布の変化は大きい。speed の約 3.3 倍の JSD だが、group size が違うため「3.3 倍重要」とは言えない|
|actor IG `normalized_speed`|`mean_absolute_ig = 0.319833` nats|開始時観測から現在観測への経路で、選択 action の log probability に対する speed の平均的な寄与の大きさが大きい。正負の主張には signed 列が必要|
|closed-loop `navigation`|`mean_delta_total_reward = -106.549`、`mean_delta_success = -1`、`mean_delta_out_of_road = +1`|この 1 paired episode では、navigation 10 次元の入力置換後に報酬低下、success fraction の 1 減少（成功→失敗）、逸脱増加が観測された|
|closed-loop `normalized_speed`|`mean_delta_total_reward = -36.5696`|speed 置換でもこの paired episode の総報酬は低下した|

`speed` group は `normalized_speed` と同じ index 3 を指すため、両者を二つの独立した証拠として数えません。同様に `navigation` と `checkpoint_1` は重なります。

LiDAR については、保存済み rollout の **127 x 240 の全値が 1.0** です。episode-start baseline の LiDAR も 1.0 なので、各 LiDAR index では $x_i-b_i=0$ です。このため、LiDAR sector を置換しても policy 入力は変わらず JSD は 0、IG も 0 になります。これは「この rollout とこの baseline には LiDAR の変動がなかった」ことを示すだけです。traffic、scenario、seed、baseline を変えたときにも LiDAR が不要であることの証明ではありません。

この run からの適切な暫定結論は、「この一つの seed では、episode-start replacement に対して speed と navigation が固定 PPO 方策の actor 出力を変え、selected targets の closed-loop 置換では走行成績も悪化した。だが 1 scenario、入力置換による分布外化の可能性があるため、道路一般・交通一般・因果一般へは拡張しない」です。

## 出力が多い理由と配布範囲

大量の出力は通常の報告にすべて必要なわけではありません。compact 出力では root の `report.md` の実験対応表と実験03の比較表を共有し、数値確認が必要なときだけ所属実験の `details.zip` を展開します。介入行は実際に実施した target をすべて表示し、上位5件だけに切り詰めません。役割ごとに分けます。

|配布層|受け手|含めるもの|目的|
|---|---|---|---|
|報告本文|意思決定者・発表聴衆|root `report.md`（実験対応表と実験03の走らせ方・総報酬・完走・道路外への逸脱の自動比較表）、各実験の限界|普段の結果共有。詳細 CSV や PNG は必要な場合だけ添付|
|数値根拠|査読者・共同研究者|実験01/02/03の各 `details.zip` 内 summary と metadata、`shared/details.zip` 内の共通 metadata/schema|実験ごとの数値・filter・index の追跡|
|再解析・監査|再現担当|上記に加え `shared/details.zip` 内の rollout arrays/steps/metadata、各実験 archive 内の raw NPZ/JSONL、config/schema/checkpoint の hash|別集計、plot 再作成、run の追跡|

`shared/` の `rollout_arrays.npz` と rollout metadata/steps は、MetaDrive を再走行せず `analyze` で保存済み observation を再解析する基礎です。実験01/02の raw NPZ は各実験の per-step/per-target の監査や別集計に、実験03の JSONL は paired outcome の監査に必要です。一方で、通常の PDF やスライドへ巨大な NPZ を添付する必要はありません。compact は削除ではなく、root の要約と所属実験ごとの詳細 archive を分けます。

## 結論文のテンプレート

次の角括弧を metadata と CSV の実値で埋めます。

> 固定 checkpoint `[model hash/名前]` を `[環境、scenario seed 数、RL seed、N decision]` で解析し、baseline は `[strategy, count]` とした。full-episode / baseline-mean の offline 摂動では、`[feature/group]` の JSD は `[値]` nats であり、これは `[target_size]` 次元を開始時値へ置換した場合の actor 分布変化を表す。IG は `[target]` に対して `[feature]` の `mean_absolute_ig = [値]` `[nats または critic return 尺度]` を示したが、符号と target は別表で扱った。paired closed-loop では `[target]` の介入−通常差は total reward `[値]`、success `[値]`、out-of-road `[値]` だった。これはこの固定方策・baseline・`[scenario 数]` scenario に限る依存の証拠であり、入力の一般的必要性や環境因果の証明ではない。

この形式なら、実験条件、数値の由来、単位、限界が一段落で追跡できます。
