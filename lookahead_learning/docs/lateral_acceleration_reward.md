# 参照経路の必要横加速度に対する追加報酬

## 目的と評価対象

開始車線の維持、左右のふらつき抑制、設定速度の維持を目指す比較実験として、
前方の経路形状に対して速度が大きすぎる場合に負の報酬を加えます。
行動や目標速度を上書きする制御器ではなく、PPO に渡す報酬の追加項です。

評価対象は **開始車線に対応する固定参照経路を、その速度で走るための必要横加速度** です。
実測横加速度、実際の車両軌跡の曲率、PP が注視点へ結ぶ円弧の曲率とは異なります。
PP の kappa_pp は使いません。将来その場所へ到達した時の速度や横加速度も予測しません。

共通の経路・投影・注視点は [既存の幾何仕様](methods.md) と
[固定経路の定義](route_definition.md) を引き継ぎます。

## 区間と曲率

行動前の状態を $s_t$、開始車線の固定参照経路を $P(S)$、その弧長座標を $S$ [m] とします。
自車投影点から、既存の lookahead_m だけ先の注視点までを評価します。

$$
S_{\mathrm{goal},t}=S_{\mathrm{proj},t}+\mathrm{lookahead\_m},\qquad
K_t=\max_{S\in[S_{\mathrm{proj},t},S_{\mathrm{goal},t}]}
|\kappa_{\mathrm{route}}(S)|.
$$

注視点自体が直線上でも、その手前の短いカーブを検出するための区間です。
新たな先読み距離パラメータは追加しません。

| 車線形状 | 曲率の絶対値 [1/m] | 取得元 |
|---|---:|---|
| StraightLane | 0 | 検証済みの直線という形状契約 |
| CircularLane | $1/R$ | 対象車線中心線の実際の半径 R [m] |
| その他 | 対応外 | 検証済み経路の境界として扱い、直線へ置換しない |

[RouteCurvatureProfile](../geometry.py) は reset ごとに **既存の検証済み経路** の車線を
列挙し、車線ごとの曲率を保存します。MetaDrive では CircularLane.radius が中心線半径です。
lane.length や中心角から半径を推測しません。別 host がこの API を持たなければ、
MetaDrivePreviewProvider の radius_reader(lane) に、確認済みの半径 [m] を返す関数を渡します。
既存経路の再構築、等間隔の点サンプリング、別の投影計算は行いません。

### 境界の包含規則

P(S) の既存実装に合わせ、内部の接続点は **進行方向の次の車線** に属します。
各車線は [start, end)、最後の検証済み車線のみ右端を含みます。
調べる区間 [S_proj, S_goal] 自体は両端を含みます。

- 直線→円弧の境界に注視点がちょうど達したら、その円弧の曲率も含めます。
- 円弧→直線の境界に投影点がちょうど来たら、通過済み円弧を含めません。
- 区間の内部に円弧があれば、注視点が直線上でも円弧を含めます。
- 注視点が検証済み経路の終端と一致する場合は最後の車線を含めます。
  これは真の経路終端と、経路が途中で切れた検証済み prefix の終端の双方に適用されます。
- 終端を少しでも越えれば参照無効です。端へ丸めたり外挿したりしません。

既存の投影処理で接続点をどの候補へ帰属したかによらず、共通の累積 S に対してこの規則を使います。

## 時刻・単位・数式

遷移 $s_t\xrightarrow{a_t}s_{t+1}$ について、次の順序で計算します。

1. reset または前回 step 後に作った snapshot の K_t と参照有効性を保持します。
2. 既存 wrapper チェーンの env.step(a_t) を **ちょうど1回** 呼びます。
3. 行動後 snapshot の平面速度の大きさ $v_{t+1}$ [m/s] を使って採点します。
   行動後の新しい注視点や曲率で K_t を置き換えません。

$$
\begin{aligned}
A_{\mathrm{req},t}&=v_{t+1}^2K_t,\\
E_t&=\max(0,A_{\mathrm{req},t}/A_{\max}-1),\\
r_{\mathrm{lat},t}&=-w_{\mathrm{lat}}\Delta t\,E_t^2,\\
r_{\mathrm{total},t}&=r_{\mathrm{base},t}+r_{\mathrm{pp},t}+r_{\mathrm{lat},t}.
\end{aligned}
$$

| 記号・設定 | 意味・単位 |
|---|---|
| A_req | 参照区間に行動後の速度を当てはめた必要横加速度 [m/s²] |
| A_max / max_lateral_accel | 許容値 [m/s²]、有限かつ正 |
| E | 許容値で正規化した超過比、無次元 |
| w_lat / lateral_accel_weight | 非負の係数。報酬を単位付きで読むなら報酬単位/秒 |
| Δt | 1 decision のシミュレーション時間 [秒] |

simulation_dt_seconds() が physics_world_step_size × decision_repeat を検証します。
この host の通常値は 0.02 秒 × 5 = 0.1 秒です。実時間でも単独の physics step 幅でもありません。
同じ episode 内で decision 時間が変わる場合はエラーです。

### 速度の adapter 契約

確認した MetaDrive の base_class/base_object.py では、BaseObject.speed は
Bullet の平面速度 (vx, vy) の大きさを m/s で返します。host 内で [0, 100000] に clip されます。
速度 API の代替となる vehicle.velocity[:2] も m/s の平面ベクトルです。
[read_vehicle_state](../adapter.py) がこれを speed_m_s として返し、新項は同じ state_reader を再利用します。
進行方向の符号付き速度は既存の preview の後退判定専用であり、新項の v に流用しません。

別 host の state_reader は、平面速度の大きさを speed_m_s で明示的に返してください。
km/h なら [planar_speed_mps](../lateral_acceleration.py) の unit="km/h"（÷3.6）で変換します。
符号付きスカラーしかなければ、その絶対値が平面速度の大きさになる契約を確認した場合だけ
signed=True を指定します。横滑り中の前後方向速度だけではこの契約を満たしません。
値の大小から単位を推測しません。key 不在や reset 時点で取得不能なら接続エラーです。
確認済み reader が後続の状態で明示的に speed_m_s=None を返した場合だけ、一時的取得不能としてマスクします。

## 有効条件と異常値

新項が計算されるのは enabled=true、weight>0、**行動前の参照区間が有効**、
行動後の速度が取得可能、かつ terminated/truncated がどちらも false の場合です。

- 終端・打ち切り step は既存 PP と同じ方針で 0 とし、host の終了報酬を保ちます。
  **最終遷移の速度超過は採点対象外** になるという限界があります。
- 開始車線の参照欠損、曖昧な投影、経路切れ、範囲外の注視点など、
  既存 preview が参照無効と判定した場合は 0 とし、理由をログへ残します。
  現在の adapter は後退・前方にない注視点なども既存 preview の無効判定を引き継ぎます。
  PP の pp_valid は新項の有効条件に使いません。
- NaN/Inf、文字列・bool の数値、単位不明、曲率や速度 API の恒常的欠落はエラーです。
  欠損や契約違反を曲率・加速度 0 に置換しません。
- enabled=false と enabled=true/weight=0 は新しい半径・速度契約を要求せず、
  新項を 0 にします。旧コンストラクタと2キーの wrap_lookahead_env 呼出しも利用できます。
- 閾値以下・停止・直線では新項は 0 です。低速や停止に正のボーナスは与えません。

診断用の速度上限は $K_t>0$ で $v_{\mathrm{curve},t}=\sqrt{A_{\max}/K_t}$ [m/s] です。
K_t=0 のときはこの条件だけでは上限なしとし、
curve_speed_limit_mps=null、curve_speed_unlimited=true を記録します。
人工的な最小曲率、Infinity、NaN は使いません。参照無効や採点除外は null と false、
skip_reason で区別します。この値で Action や目標速度を制限する処理はありません。

## 数値例

R=50 m、A_max=0.8 m/s²、weight=0.1、Δt=0.1 秒の場合です。

| 行動後速度 | K [1/m] | A_req [m/s²] | E | 新項 |
|---:|---:|---:|---:|---:|
| 5 m/s | 0.02 | 0.5 | 0 | 0 |
| 10 m/s | 0.02 | 2.0 | 1.5 | −0.0225 |

対応する速度上限は sqrt(0.8×50) ≈ 6.3246 m/s ≈ 22.77 km/h です。
左右反転しても半径が同じなら同じ新項になります。

## TOML・PPとの組合せ・旧モデル

通常の train.py/evaluate.py が読む同じ TOML の [lookahead] を使います。専用 CLI はありません。
以下は **On の設定部分** です。host の既存の環境・PPO・scenario 設定と組み合わせます。

~~~toml
[lookahead]
lookahead_m = 6.0
pp_weight = 0.0
lateral_accel_reward_enabled = true
max_lateral_accel = 0.8
lateral_accel_weight = 0.1
~~~

**Off の設定部分**（既存2キーのみでも同じ動作）：

~~~toml
[lookahead]
lookahead_m = 6.0
pp_weight = 0.0
lateral_accel_reward_enabled = false
max_lateral_accel = 0.8
lateral_accel_weight = 0.1
~~~

| pp_weight | lateral_accel_reward_enabled | 返す報酬（新項の weight>0） |
|---:|---|---|
| 0 | false | r_base |
| 正 | false | r_base + r_pp |
| 0 | true | r_base + r_lateral_accel |
| 正 | true | r_base + r_pp + r_lateral_accel |

[lookahead] 自体がない場合は従来の baseline D 次元です。存在すれば、どの組合せでも D+3 です。
PP Off の場合は MetaDrivePPProvider や PP 用の車両・policy 契約を要求しません。
上限と重みは Off 時も型・有限性・範囲を検証します。enabled は bool だけを受け入れます。
未知キー、文字列の数値、数値としての bool、NaN/Inf を拒否します。
正規化の唯一の入口は [resolve_lookahead_config](../checkpoint.py) です。

新モデルの ZIP 属性は lookahead_config と lookahead_schema_version=2 です。
version 1 の2キー旧モデルは新項 Off として読むため、lookahead_m/pp_weight が同一なら
評価を続けられます。Off 時の上限・重み差は互換性を妨げません。
enabled=true/weight=0 も実効 Off として互換です。正の重みで On の場合は
上限・重み・On/Off を含む実効設定の一致が必要で、旧モデルを On で学習済み扱いにはしません。
未知 schema、baseline/active の不一致は拒否します。
検証はモデルを変更せず、旧 ZIP の上書き、sidecar、追加ハッシュ管理はしません。
TOML root の既存 schema_version とモデルの lookahead_schema_version は別の契約です。

## このリポジトリでの通常入口と確認

既存の configs/official_start_lane_return_lookahead.toml は Off のままです。
別ファイル configs/official_start_lane_return_lookahead_lateral_accel.toml を On の比較例として用意し、
実験名・保存モデル名・評価出力名も分けています。移植先には root の設定をコピーせず、
その host の設定へ上記5キーを接続してください。

作業環境にはこの worktree 内の .venv がなく、隣の既存 Python 3.12 環境を利用しました。
次の通常入口の --help と設定解決を確認しています。長時間学習やモデル評価完走を実施したという意味ではありません。

~~~bash
../metadrive_rl-main/.venv/bin/python -B train.py --help
../metadrive_rl-main/.venv/bin/python -B evaluate.py --help
~~~

同じ Python 環境から比較学習・評価を実行するコマンドです（学習後に同名 ZIP を評価）。

~~~bash
../metadrive_rl-main/.venv/bin/python train.py --config configs/official_start_lane_return_lookahead_lateral_accel.toml
../metadrive_rl-main/.venv/bin/python evaluate.py --config configs/official_start_lane_return_lookahead_lateral_accel.toml --no-record-gif
~~~

移植先ではその host の python を使います。検証コマンド・通常入口は [README](README.md#配布と確認) を参照してください。

MetaDrive 0.4.3 と取得済み assets で、通常の make_evaluation_env を使う1環境・8 decisionの
smoke testも実行しました。raw D=259、wrapper後D+3=262、PP providerなしで、
有限値、返却報酬と内訳・episode合計、JSON出力を確認しています。
この短い走行で採点した区間は直線で新項は0でした。円弧区間の非ゼロ超過は
同梱の純粋関数・fake hostテストで検証しており、実走行の性能改善を示す結果ではありません。

## ログと集計の読み方

info["lookahead_learning"] に r_base / r_pp / r_lateral_accel / r_total と、
対応する episode_r_* を出します。lateral_accel に、enabled、effective_enabled、
active、skip_reason、reference_valid、reference_invalid_reason、s_proj_m、s_goal_m、
kappa_abs_max_inv_m、speed_post_mps、required_lateral_accel_mps2、
max_lateral_accel_mps2、exceedance_ratio、curve_speed_limit_mps、curve_speed_unlimited を記録します。
完全な pre/post snapshot も残ります。

lateral_accel_episode の集計は次の定義です。reset・終端・打ち切り・Off・ゼロ重みは除外します。

| 出力 | 定義 |
|---|---|
| required_lateral_accel_max_mps2 | 有効に採点できた decision での A_req の最大値。未採点なら null |
| exceedance_time_ratio | A_req>A_max の decision 時間合計 / 有効採点時間 |
| reference_invalid_time_ratio | 行動前参照が無効の decision 時間 / 新項 On の非終端 decision 時間 |
| reference_seconds / reference_invalid_seconds | 上記の参照検査時間 / 無効時間 [秒] |
| evaluated_seconds / exceeded_seconds | 有効採点時間 / 超過時間 [秒] |

分母0は null とします。速度の一時欠損は参照無効には数えず、採点時間から除きます。
評価の evaluation_steps.jsonl に同じ lookahead_learning namespace、
evaluation.json の各 episode に報酬合計と lateral_accel_episode を保存します。
既存 Monitor は wrapper が返した合計報酬を記録します。root 側で新項を再加算しません。
これらは **decision 時点の参照必要量** であり、物理 step 内の実測ピークではありません。

## 根拠と今回の提案を区別する

- [Autoware の旧 Motion Velocity Smoother 文書](https://autowarefoundation.github.io/autoware.universe_planning/pr-5583/planning/motion_velocity_smoother/#apply-lateral-acceleration-limit)
  は参照軌跡の曲率と許容横加速度から曲線部の速度上限を求める処理を説明しています。
- [固定コミットの smoother_base.cpp](https://github.com/autowarefoundation/autoware_core/blob/42d6b699625024f8f3fdb9bb11890e7de950b3f8/planning/autoware_velocity_smoother/src/smoother/smoother_base.cpp#L264-L275)
  の computeVelocityLimitFromLateralAcc で、横加速度と曲率の比の平方根を使う関係を確認しました。
- **0.8 の出典** は、同じ固定コミットの
  [default_velocity_smoother.param.yaml](https://github.com/autowarefoundation/autoware_core/blob/42d6b699625024f8f3fdb9bb11890e7de950b3f8/planning/autoware_velocity_smoother/config/default_velocity_smoother.param.yaml#L13-L19)
  にある lateral_acceleration_limits の4要素がすべて 0.8 である設定です。
  旧文書と固定コミットは同一版ではなく、旧文書の既定値が0.8だと主張するものではありません。

今回の区間 [S_proj, S_goal]、正規化超過二乗ペナルティ、pre の区間と post の速度という設計は
**本プロジェクトの提案** です。Autoware の RL 報酬を再現したものではありません。
min_curve_velocity、人工的な曲率下限、減速・ジャーク処理や速度最適化全体は移植しません。
0.8 は実験用参考値、weight=0.1 は未調整の開始値であり、普遍的な安全限界・最適値・効果の保証ではありません。

## 限界と戻し方

この項だけでは一定速度維持、ふらつき解消、快適性、安全を保証しません。
既存の速度・進捗報酬との競合により、閾値超過が残る可能性があります。目標速度自体は変更しません。
区間最大曲率は保守的な先読み評価です。減速度・ジャーク・制動距離から速度計画を作っておらず、
注視距離が短ければ減速が間に合わない可能性があります。
単一点の D+3 入力を維持するため、区間内部の曲率を観測から一意に判別できない場合があります。
性能上の課題は比較実験へ記録し、この変更では観測を増やしません。
学習比較を行うまでは速度維持やふらつき抑制の改善を断定できません。

Off へ戻すには lateral_accel_reward_enabled=false とし、その実効設定で学習したモデルを使います。
旧 v1 モデルは同じ lookahead_m/pp_weight の Off 設定で評価できます。
On モデルを Off でそのまま評価すると設定不一致で停止します。受入検証を迂回しないでください。

フォルダ差し替え前には旧 lookahead_learning/ と host 独自変更を別の場所へバックアップしてください。
**差し替えで既に消えた独自変更は、このプロンプトだけでは復元できません。**
接続を戻す場合も、バックアップした host ファイル・TOML・対応するモデルを組み合わせます。
新規移植・差分更新の入口は共通の [copilot_porting_prompt.md](copilot_porting_prompt.md)、
バックアップと接続判定の詳細は [porting.md](porting.md) です。
