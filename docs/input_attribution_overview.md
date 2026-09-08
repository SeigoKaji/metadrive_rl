# 入力寄与解析：何を調べ、どう実験するか

## 何を知りたい実験か

学習済みの運転AIは、どの入力情報に依存して走っているか。

この実験では学習済みモデルを固定し、AI が行動を決める直前に受け取る数値の一部だけを差し替えます。
その結果、AI の判断が変わるか、そして通常走行と比べて走行結果が変わるかを見ます。
モデルを学習し直したり、道路そのものを書き換えたりする実験ではありません。

## MetaDrive から AI に渡る入力

ここでいう観測（observation）は、行動決定直前に AI に渡す数値の並びです。カメラ画像そのものではありません。
このプロジェクトの標準 schema `metadrive_default_259` では、次の 259 個で構成します。
ただし 259 次元が MetaDrive の全環境で常に成立するわけではなく、実行する環境・モデル・schema の組み合わせを確認します。

`normalized_speed` は車速を AI が扱いやすい尺度に変換した 1 個の値です。車の速度そのものと、AI が受け取るこの値は、実験では分けて考えます。

| 入力の部分 | 主な中身 | 個数 |
| --- | --- | ---: |
| 自車の状態（`ego_state`） | 左右の道路境界までの距離、車線に対する向き、車速、ハンドル、前回のハンドル・加減速操作、車の向きが変わる速さ、車線内の位置 | 9 |
| 進路案内（`navigation`） | 次の 2 つの通過点それぞれについて、前後・左右方向の位置や道の曲がり具合を表す 5 項目 | 10 |
| 周囲の距離情報（LiDAR） | 周囲との距離を測る LiDAR の各方向の読み値 | 240 |
| 合計 | `ego_state` + `navigation` + `lidar` | 259 |

入力番号と意味を確認するときは [観測スキーマ](../observation_schemas/README.md) を参照します。別の観測次元を持つ環境へ移すときは、259 用 schema をそのまま流用しません。

## 3つの実験と保存先

通常走行の収集は、3実験が共通して使う準備です。実験01が実験03の候補を選び、実験02は候補選択には使いません。

| 番号 | 実験名 | 何を判断するか | compact の保存先 |
| --- | --- | --- | --- |
| 準備 | 通常走行の収集 | 介入前の観測・操作・報酬を記録する | `shared/` |
| 実験01 | 入力置換（オフライン摂動） | 入力を基準値へ置いたとき、AI の操作確率が変わるか（JSD） | `experiment_01_perturbation/` |
| 実験02 | Integrated Gradients（出力変化の入力への割当） | 指定した AI 出力の変化を、どの入力へ割り当てるか | `experiment_02_integrated_gradients/` |
| 実験03 | 入力固定での走行比較（paired closed-loop） | 入力固定で総報酬・完走・道路外への逸脱が変わるか | `experiment_03_closed_loop/` |

各実験フォルダには `report.md` と `details.zip` があり、compact の共通準備は `shared/details.zip` にまとめます。
full では同じフォルダ分けで ZIP の代わりに個別ファイルを保存し、元の詳細 report は `shared/report.md` として保全します。

## 実験03では、どう入力を固定するか

まず通常走行を行い、各時点で AI に渡す観測と、AI が選んだ操作を記録します。走行比較では、同じ開始条件で別の介入走行を始め、
毎回、比較したい入力だけをその走行の開始時の値に戻した「AI 用の観測コピー」を作ります。環境内部を直接変更せず、
変更した観測から AI が選んだ操作で車を動かします。

- 速度を固定する場合：AI に見せる `normalized_speed` だけを開始時の読み値にします。車を止めたり、物理的な速度を固定したりはしません。車は選ばれた操作に従って走り続けます。
- 進路案内を固定する場合：AI に見せる通過点・進路情報だけを開始時の値にします。道路、通過点、地図を変更する処理ではありません。

通常走行と入力固定走行は同じ開始条件で対にして、総報酬、完走、道路外への逸脱を比べます。
この比較が答えるのは、「この固定モデルが、この条件で、その入力を開始時の値に固定されたとき走行結果を変えたか」です。

## 比較する入力の選び方

標準設定では、まず摂動で走行比較の候補を絞ります。ここでいう feature は入力 1 項目、
group は複数入力のまとまりです。

- **実験01 入力置換（オフライン摂動）**：環境を動かさず、基準値へ差し替える前後で操作の確率がどれだけ変わるか（JSD）を見る。既定ではこの JSD の上位 10 feature と上位 5 group を、実験03の対象にします。
- **実験02 Integrated Gradients（出力変化の入力への割当）**：開始時から現在までの観測変化が、指定した AI 出力の変化にどう割り振られるかを見る。IG は実験03の候補選択には使わず、出力変化を入力へ割り当てる別の補助分析です。

摂動と IG は AI の出力を調べる補助であり、走行の総報酬そのものではありません。詳しい指標や設定は
[入力寄与解析ガイド](input_attribution.md) にまとめています。

259 個すべてを走行中に固定する設定ではありません。比較対象は設定で指定・変更できます。対象を変更した場合は、同じ report の走らせ方の名前で区別して読みます。

## 実行後に読む結果

解析を実行すると root の `report.md` に、3実験の対応表と各実験 `report.md` へのリンクが入り、実験03の走行比較だけが次の 4 列で自動生成されます。
下表は出力の形を示す見本です。これは3実験すべての結果を一つにした表ではありません。結果の数値や成否は実行後に決まり、未測定の値を `0` とは扱いません。

| 走らせ方 | 総報酬 | 完走 | 道路外への逸脱 |
| --- | ---: | --- | --- |
| 通常走行 | 実行後に自動記入 | 実行後に自動記入 | 実行後に自動記入 |
| 入力固定（対象名） | 実行後に自動記入 | 実行後に自動記入 | 実行後に自動記入 |

複数の走行条件を指定した場合は、総報酬は平均、完走と道路外は割合としてまとめます。まずこの比較表と最小限の実験条件を読めばよく、
`details.zip` の CSV や plot は数値を追跡したい場合だけ確認します。

報酬が下がることは、その条件で入力への依存がある手がかりです。結果が変わらなくても、その入力が一般に不要だとは証明できません。
結果は実行したモデル、環境、開始条件、基準値、対象入力の選び方によって決まります。

## 通常の実行入口

次は `models/official_baseline.zip` を使う実行例です。別の学習済みモデルを調べる場合は `--model` を変更します。既定出力は compact 形式で、この一コマンドで root の対応表、3実験の `report.md` と4つの詳細 `details.zip` を保存します。

```bash
.venv/bin/python analyze_input_attribution.py run \
  --config configs/official.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --output-prefix official_left_curve_numbered
```

```text
outputs/official/attribution/official_left_curve_numbered/
├── report.md                         # 3実験の対応表と実験03の比較表
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
    └── details.zip                   # compact の metadata・schema・通常 rollout
```

パス中の `official` は `configs/official.toml` の評価条件名であり、解析方法の「実験01〜03」とは別です。同じ `official/<prefix>` の中に3実験をまとめます。

個別指標や設定を詳しく調べる場合は [結果の読み方](input_attribution_results.md)、schema の成立条件は
[観測スキーマ](../observation_schemas/README.md)、別 PC や別の観測次元への移植は [移植手順](input_attribution_porting.md) を参照してください。
