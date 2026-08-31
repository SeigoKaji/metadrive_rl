# 開始レーン維持実験の記録

## 目的

この実験群は、MetaDrive 上で車両が**reset 時の開始レーンを維持しながら前進し、目的地へ到達する**方策を学習できるかを確認するためのものです。単に開始レーンから出ないだけではなく、到着まで走行することを目的とします。

基準設定は [official_start_lane_return.toml](../configs/official_start_lane_return.toml) です。`StartLaneMetaDriveEnv` は reset 時のレーン ordinal を記録し、通常の MetaDrive の「現在いるレーン」基準の lateral reward ではなく、その開始レーンを基準に連続的な cost を与えます。

baseline の停止解を崩すため、まず一要因ずつ三つの ablation を実施しました。いずれも採用基準を満たさなかったため、追加案は一つだけという方針で、Duckietown 2021 の lane-distance 方式を第四実験として実施しました。

## 共通条件

| 項目 | 条件 |
| --- | --- |
| map / scenario | `map = "C"`、`start_seed = 5`、`num_scenarios = 1` |
| RL | Stable-Baselines3 PPO、`MlpPolicy`、RL seed `0`、4並列環境 |
| 学習量 | `total_timesteps = 300000` を要求。`n_steps = 4096` と4環境の rollout 境界により、実行された環境 step は各実験 `311296` |
| action / episode | 3×3 の離散 action、`horizon = 500`（1 decision は0.1秒、最大50秒） |
| 評価 | scenario seed `5`、評価 seed `0`、決定論的 action。各設定を1 episode 評価 |
| 開始レーン設定の基準値 | `start_lane_objective = "return"`、中心係数 `0.25`、wrong-lane 係数 `0.10`、許容幅比 `0.05`、開始レーン終端 penalty `200`、道路外 penalty `200`。各実験では、後述する単一差分だけを変更 |

実験番号は**実行順**です。番号は今後の設定ファイルを選びやすくするためのものであり、TOML 内の `name`、`default_model_name`、`evaluation.output_prefix` と、`default_model_name` から解決される学習モデル名は、既存のモデル・成果物との互換性のため番号なしのままです。

## 採用基準

以下をすべて満たすものを、開始レーン維持の成功候補としました。

- `arrive_dest = true` かつ route completion が `95%` 以上
- crash、道路外、wrong-lane arrival、開始レーン離脱がない
- 開始レーン滞在率が `1.0`
- 平均速度が `10 km/h` 以上

## 実験順と設定差

| 順番 | 設定ファイル | 内部profile / 成果物名 | baseline からの単一差分 | 仮説 |
| --- | --- | --- | --- | --- |
| baseline | [official_start_lane_return.toml](../configs/official_start_lane_return.toml) | `official_start_lane_return` | なし | 開始レーン基準の連続 cost で、走行とレーン維持が両立するか確認する。 |
| 01 | [01_official_start_lane_return_idle_penalty.toml](../configs/01_official_start_lane_return_idle_penalty.toml) | `official_start_lane_return_idle_penalty` | `low_speed_threshold_km_h = 10.0`、`low_speed_penalty_rate = 0.5` | 停止を継続するほど不利にし、前進探索を促す。 |
| 02 | [02_official_start_lane_return_progress_balance.toml](../configs/02_official_start_lane_return_progress_balance.toml) | `official_start_lane_return_progress_balance` | `start_lane_center_coef: 0.25 → 0.10` | 中心保持 cost が前進報酬を相殺しているなら、その均衡を緩める。 |
| 03 | [03_official_start_lane_return_timeout_penalty.toml](../configs/03_official_start_lane_return_timeout_penalty.toml) | `official_start_lane_return_timeout_penalty` | 純粋な horizon truncation 時だけ `timeout_penalty = 25.0` | 到着せず待つ方策を終端で不利にする。 |
| 04 | [04_official_start_lane_return_duckietown_progress.toml](../configs/04_official_start_lane_return_duckietown_progress.toml) | `official_start_lane_return_duckietown_progress` | `target_lane_progress_only = true` | 開始レーン内の正方向距離だけを報酬化し、レーン外の前進で報酬を得られないようにする。 |

## 決定論評価の結果

表中の reward は episode 合計、route は最終 route completion、速度は km/h です。すべて500 stepで `max_step = true` となり、`arrive_dest = false`、crash / 道路外 / wrong-lane arrival / 開始レーン離脱はすべて `false`、開始レーン滞在率はすべて `1.0` でした。

| 実験 | reward | route | 平均速度 | 最大速度 | 決定論 action | 補足 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| baseline | `0.0129559529` | `3.13238709%` | `0.013148` | `0.164170` | RIGHT + NEUTRAL | 到着せず時間切れ。 |
| 01 idle penalty | `-24.9937305` | `3.10985719%` | `0.016615` | `0.603871` | RIGHT + BRAKE | 低速 penalty 合計は `24.958462`。 |
| 02 progress balance | `0.0183293515` | `3.13549181%` | `0.013807` | `0.149331` | STRAIGHT + NEUTRAL | 到着せず時間切れ。 |
| 03 timeout penalty | `-25.0190750` | `3.10955875%` | `0.020256` | `0.834834` | STRAIGHT + BRAKE | timeout penalty 合計は `25.0`。 |
| 04 Duckietown progress | `0.0288386345` | `3.13549181%` | `0.013807` | `0.149331` | STRAIGHT + NEUTRAL | 計上された前進距離・前進報酬の合計はともに `0.0288386345`。 |

全設定が採用基準を満たしませんでした。開始レーン内に留まることには成功していますが、実際にはほぼ停止する方策です。

## 各実験の詳細

### baseline

baseline は開始レーンの中心からの正規化誤差と開始レーン外への逸脱に対する連続 cost を、MetaDrive の通常報酬から減算します。評価では route completion が約3.13%に留まり、RIGHT + NEUTRAL を500 step継続しました。レーン逸脱も終端失敗もないため、開始レーンを「維持」する停止解を選んでいます。

補足として、同じbaselineモデルを stochastic action selection で seed `0`〜`4` の5回評価しました。全 episode が時間切れ・未到着であり、平均 route completion は `6.16%`、平均速度は `0.648 km/h` でした。決定論評価だけに限定された失敗ではなく、sampling を有効にしても走行成功には至っていません。

### 01: 低速 penalty

低速 penalty は action duration を `dt`、速度を `v`、閾値を `v_th` として、次式で計算します。

```text
p_low = rate × dt × max(0, 1 - max(v, 0) / v_th)
```

この設定では `rate = 0.5 reward/s`、`dt = 0.1 s`、`v_th = 10 km/h` なので、停止時は1 decisionあたり `0.05`、500 stepでは概ね `25` の penalty です。実測の penalty 合計も `24.958462` でしたが、平均速度は `0.016615 km/h` に留まり、停止解は解消しませんでした。

### 02: 前進報酬とのバランス

中心保持の係数だけを `0.25` から `0.10` へ下げ、その他のPPO・環境・開始レーン設定をbaselineと一致させました。仮説は、中心に留まることの局所的な cost を緩めれば、前進報酬が相対的に有利になるというものです。結果はbaselineと同程度の約3.14%の進捗・`0.013807 km/h`で、前進にはつながりませんでした。

### 03: 時間切れ penalty

`timeout_penalty = 25.0` を、到着・失敗・strict departure・wrong-lane arrival を伴わない純粋な horizon truncation にだけ適用しました。MetaDrive の timeout は truncation のまま維持しています。評価の合計rewardはほぼ `-25` となりましたが、平均速度は `0.020256 km/h` に留まり、時間切れ penalty だけでは停止解を崩せませんでした。

### 04: Duckietown の lane-distance 方式

最初の三つが未達だったため、追加案は一つだけに限定しました。採用したのは、[Duckietown lane tracking の2021年論文](https://doi.org/10.21014/acta_imeko.v10i3.1020) と [著者実装の lane-distance reward](https://github.com/kaland313/Duckietown-RL/blob/22f17b3bdad8aca7cec345ad0661af9aed443d90/duckietown_utils/wrappers/reward_wrappers.py#L816-L895) を参考にした距離方式です。横偏差から復帰方向を作る orientation reward は実装せず、混在させませんでした。

一つ前と現在の両方で有効な開始レーン内にいる場合だけ、`navigation.travelled_length` の正方向差分を使います。

```text
d_t = max(0, travelled_length_t - travelled_length_(t-1))
r_t = driving_reward × d_t
```

それ以外（開始レーン外、再進入 transition、逆方向、target lane geometry が無効）は `d_t = 0` です。通常stepではMetaDriveの現在レーン基準の進捗・速度項と、開始レーン cost・低速 penalty を置き換えます。実装は `target_lane_forward_distance_m` と `target_lane_progress_reward` に各stepの値を残します。

この簡易実装は、MetaDrive の `navigation.travelled_length` が route reference lane ordinal `0` を基準に累積されることを利用しています。そのため、map C の開始レーン ordinal `0` だけを明示的にサポートします。開始レーン ordinal `1` / `2` や別の道路構成へ一般化するには、対象レーン自身のsegment横断累積座標を実装する必要があります。

結果は正方向距離 `0.0288386345 m`、同額の前進報酬に留まりました。方式自体は小さく実装できましたが、この学習条件では前進を十分に引き出せませんでした。

## 停止方策の診断

五つの決定論評価で共通して、開始レーン滞在率は `1.0`、開始レーン離脱は0回、終端失敗はありませんでした。一方で、route completion は約3.1%、平均速度は `0.013`〜`0.020 km/h` 程度で、到着前に時間切れになりました。

したがって、現状は「開始レーン維持」という制約だけを満たす局所解です。低速 penalty と timeout penalty は停止時の報酬を大きく下げたものの、探索中に開始レーン外・道路外へ出る損失を避ける方策を覆せませんでした。特に `out_of_road_penalty = 200` と `start_lane_terminal_penalty = 200` が大きいため、探索を避ける圧力になっている可能性があります。これはログからの診断であり、因果検証はまだ行っていません。

## 実装した telemetry

評価の `evaluation_steps.jsonl` に、既存の開始レーン状態に加えて次の値を記録します。

- `target_lane_cost`: 開始レーン中心・逸脱に関する通常step cost
- `low_speed_penalty`: 低速 penalty の当step値
- `timeout_penalty`: 純粋な時間切れで適用した当step値
- `target_lane_forward_distance_m`: Duckietown方式で計上した当stepの正方向距離
- `target_lane_progress_reward`: 上記距離から計算した当stepの報酬

これらにより、episode reward と各shaping項の対応、Duckietown方式で実際に前進距離が計上されたかを後から確認できます。

## ローカル成果物

各profileの内部名は番号なしで固定しているため、モデルと成果物ディレクトリも番号なしです。各評価フォルダには `evaluation.json` と `evaluation_steps.jsonl`、記録を有効にした実行では可視化成果物が置かれます。

| 実験 | モデル | 学習metadata | 評価フォルダ |
| --- | --- | --- | --- |
| baseline | `models/official_start_lane_return.zip` | `outputs/official_start_lane_return/training/official_start_lane_return/training_metadata.json` | `outputs/official_start_lane_return/evaluation/official_start_lane_return/` |
| 01 | `models/official_start_lane_return_idle_penalty.zip` | `outputs/official_start_lane_return_idle_penalty/training/official_start_lane_return_idle_penalty/training_metadata.json` | `outputs/official_start_lane_return_idle_penalty/evaluation/official_start_lane_return_idle_penalty/` |
| 02 | `models/official_start_lane_return_progress_balance.zip` | `outputs/official_start_lane_return_progress_balance/training/official_start_lane_return_progress_balance/training_metadata.json` | `outputs/official_start_lane_return_progress_balance/evaluation/official_start_lane_return_progress_balance/` |
| 03 | `models/official_start_lane_return_timeout_penalty.zip` | `outputs/official_start_lane_return_timeout_penalty/training/official_start_lane_return_timeout_penalty/training_metadata.json` | `outputs/official_start_lane_return_timeout_penalty/evaluation/official_start_lane_return_timeout_penalty/` |
| 04 | `models/official_start_lane_return_duckietown_progress.zip` | `outputs/official_start_lane_return_duckietown_progress/training/official_start_lane_return_duckietown_progress/training_metadata.json` | `outputs/official_start_lane_return_duckietown_progress/evaluation/official_start_lane_return_duckietown_progress/` |

既存成果物は番号付け前に生成したため、metadata の `config_source.path` には実行当時の番号なしのsource pathが残っています。TOML本文と内部profile / artifact名は変更しておらず、各metadataの `config_source.sha256` は現在の番号付き設定ファイルのSHA-256と一致します。

## 再現方法

以下はこのREADMEがあるプロジェクトrootから実行します。同じ内部profile名のモデル・出力先を使うため、既存成果物を保持する場合は事前に退避してください。

```bash
# baseline
.venv/bin/python train.py --config configs/official_start_lane_return.toml
.venv/bin/python evaluate.py --config configs/official_start_lane_return.toml

# 01: 低速 penalty
.venv/bin/python train.py --config configs/01_official_start_lane_return_idle_penalty.toml
.venv/bin/python evaluate.py --config configs/01_official_start_lane_return_idle_penalty.toml

# 02: 前進報酬とのバランス
.venv/bin/python train.py --config configs/02_official_start_lane_return_progress_balance.toml
.venv/bin/python evaluate.py --config configs/02_official_start_lane_return_progress_balance.toml

# 03: 時間切れ penalty
.venv/bin/python train.py --config configs/03_official_start_lane_return_timeout_penalty.toml
.venv/bin/python evaluate.py --config configs/03_official_start_lane_return_timeout_penalty.toml

# 04: Duckietown lane-distance
.venv/bin/python train.py --config configs/04_official_start_lane_return_duckietown_progress.toml
.venv/bin/python evaluate.py --config configs/04_official_start_lane_return_duckietown_progress.toml
```

## 検証と制約

実装時点でテスト全体は `171 passed` でした。番号付きファイル名と番号なしの内部profile / artifact名が別であること、各設定がbaselineから一要因だけ異なること、Duckietown profileが実環境でreset・1 stepできることを回帰テストで確認しています。

この結論は map C・scenario 5・RL seed 0 の小規模な比較に限られます。04はtarget lane ordinal `0` 専用であり、他レーンへの一般化は未実装です。次の候補として、`out_of_road_penalty` と `start_lane_terminal_penalty` を現在の `-200` 規模からMetaDrive公式の終端スケールである `-5` 付近へ下げる、単一要因の terminal-scale ablation が考えられます。これはまだ実装・学習していません。
