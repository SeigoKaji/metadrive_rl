# 通常入口への移植と既存接続の差分更新

最初に実施を依頼する文書は配布フォルダ直下の [START_HERE.md](../START_HERE.md) です。そこから詳細な [copilot_porting_prompt.md](copilot_porting_prompt.md) を読み、差分適用と検証まで進めます。
移植用ファイルは **lookahead_learning_update/** に一時配置します。
実際の稼働フォルダとの比較・バックアップ前に上書きしません。
仕様を確認する必要がある場合だけ [time_prediction.md](time_prediction.md) の該当節を読みます。

## 配布範囲と保護

[../PORTABLE_FILES.txt](../PORTABLE_FILES.txt) は移植元rootからの相対パスです。
そのファイル群を同じ構造で運ぶだけで、実行コード・必要テスト・6設定例・最小文書がそろいます。
rootのtrain.py/evaluate.py/env_factory.py/start_lane_env.pyはコピー対象ではありません。
モデル・動画・画像・Excel・assets・無関係なレポートも含みません。

移植元リポジトリ直下に配布済みの `lookahead_learning_update_acceleration.zip` があります。
再生成する場合は `python -B -m lookahead_learning.pack --output /tmp/lookahead_learning_update_acceleration.zip` を実行し、未作成の出力先を指定します。
ZIP内のファイルはすべてlookahead_learning_update/配下にあり、稼働中lookahead_learning/と別名です。
ZIPを別PCで空の場所へ展開し、そのlookahead_learning_update/を既存lookahead_learning/の隣へ配置します。
同名の更新フォルダが既にあれば上書きせず、別の配置先を選んでCopilotへパスを伝えます。

manifestの各パスは適用後の配置（lookahead_learning/...）を表します。
比較時は `lookahead_learning_update/相対パス` と `lookahead_learning/相対パス` を対応させます。
フォルダ全体のコピー置換・リネーム、更新フォルダのruntime importは行いません。
配置後はCopilotへ「lookahead_learning_update/START_HERE.md を読んで実施してください」と依頼するだけで、必要差分の移植と検証まで進められます。

Copilotは変更予定を特定してから、新しい空のバックアップ先へ原本・未コミットdiff・HEAD・元から無かったファイル一覧を保存します。
適用後のdiffとハッシュも保管します。独自adapterを丸ごと置換しません。

| 実接続の状態 | 作業 |
|---|---|
| 未導入 | 下記4接続を既存の通常入口に追加 |
| 機能1・2導入済み／旧版 | 設定・wrapper・metadata接続を再利用し、機能3の不足差分だけ追加 |
| 独自改変/部分適用 | hostの契約・独自処理を保持して必要箇所だけ調整 |
| 適用済み | 検証のみ |

## 4つの接続

### 1. config loader

```python
from lookahead_learning.checkpoint import resolve_lookahead_config
lookahead_config = resolve_lookahead_config(raw.get("lookahead"))
```

[lookahead]なしはNone、ありは解決済みmappingです。
このmapping全体をprofile→通常train/evaluate→factory→workerへ渡します。
閉じた許可キーや辞書再構築が旧キーだけになっていないか確認します。
T指定時はlookahead_mが未使用です。T省略・予測Offは従来動作です。

### 2. 共通env factory

```python
from lookahead_learning.adapter import wrap_lookahead_env
if lookahead_config is not None:
    raw_env = wrap_lookahead_env(raw_env, **lookahead_config)
```

既存の接続があれば再利用します。raw Env → LookaheadEnv → 既存Monitor → VecEnvの順序です。
評価へMonitorを新設する必要はありません。外側が返却rewardを集計するため、Monitorにも新項が1回だけ反映されます。
reset/stepをunwrappedから直接呼ばず、既存host wrapperを通します。
Dは実観測から取得し、既存prefix・Action・報酬・終了条件・開始車線特徴を保持します。

### 3. trainの保存 / evaluateの読込み

```python
from lookahead_learning.checkpoint import (
    set_lookahead_model_metadata, validate_lookahead_model_metadata,
)
set_lookahead_model_metadata(model, lookahead_config)      # PPO.save前
validate_lookahead_model_metadata(model, lookahead_config) # PPO.load直後
```

保存後再読込の確認にもvalidateを使います。既存接続があれば変更不要です。
ZIP内schema4、旧1/2の距離・予測Offと旧3のconstant_speed読み取り互換はhelperに任せます。旧ZIPは書き換えません。
移植先が独自のモデル属性保存除外を持つ場合、上記2属性が実際のZIPへ入ることを確認します。

### 4. 既存ログ

step traceで `info["lookahead_learning"]` をそのまま保存していれば変更不要です。
episodeで選択して保存する場合だけ既存リストに以下を追加します。

```python
"episode_r_prediction", "prediction_episode",
```

機能1・2で接続済みならこれらのキーも既にあります。この機能3追加で移植元のroot runtime変更はありません。
その他のroot差分は既存の通常入口・worker保持を検証するテストです。rootの実装を配布物へコピーする必要はありません。
成功率・速度・進捗・横ずれ・操舵変化量は既存評価出力/step traceを使い、新規レポート基盤は作りません。

## host adapterを維持する

`MetaDrivePreviewProvider`はreset時の固定参照経路と開始車線を継続利用します。
MetaDrive固有の意味・単位を移植先ソースで確認してください。
独自reader/providerの契約は [仕様1・5節](time_prediction.md) にあります。
契約不明ならその箇所を保留し、推測で速度・位置・曲率を作りません。
開始車線クラス名が異なるhostに、移植元start_lane_env.pyの必須importを追加しません。
既存の横加速度報酬を使用中なら、式、On/Off、max_lateral_accel、lateral_accel_weight、LateralReference/radius_reader接続を保持します。
今回追加するのは既存位置項のconstant_acceleration方式です。constant_speedとの両方加算はしません。例の `lateral_accel_reward_enabled=false` を使用中設定へ上書きしません。
host自身に既存の横加速度項がある場合はr_base内に残し、wrapper側で同じ項を再加算しません。
同じTで予測Off/Onを比較し、既存横加速度項が同一で、返却rewardの差がr_predictionだけであることを確認します。
今回の予測報酬は半径・PP・操舵モデルAPIを新たに要求しません。

## 機能3の読取専用参照取得口

独自preview_providerでは、実効On（prediction_reward_enabledかつweight>0）のconstant_acceleration時だけ、次を実装します。

```python
from lookahead_learning.prediction import MotionState, PredictionReference

def reference_at_distance(self, post: MotionState, *, distance_m: float) -> PredictionReference:
    # 保存済み固定経路と直前の通常previewのpost状態／投影で問い合わせる。
    # 有効時: PredictionReference(True, q_reward, S_proj, S_proj + distance_m)
    # 無効時: PredictionReference(False, ..., invalid_reason="具体的な幾何理由")
    ...
```

- 位置は車体中心ワールド2次元[m]、headingはrad、distance_mは既に停止を考慮したD(T)[m]です。同梱計算を再実装しません。
- 同じpostから同じreset固定経路へ投影します。入力previewのprojectionが使えれば再利用し、再計算でも同じ経路・post位置を使います。予測位置から再投影しません。
- 共通invalid（経路変更・開始車線喪失・投影失敗）と距離依存invalid（終端・未検証境界・前方・距離不足）を区別します。入力b=0だけを理由に報酬点も無効にしません。共通invalidは両者に適用します。
- valid=Trueならgoal_xy、s_proj_m、s_goal_mを有限値で返し、s_goal_mはS_proj＋渡されたdistance_m、invalid_reasonはNoneです。valid=Falseなら具体的な理由を必須とし、未計算値はNoneです。
- 読取専用です。providerのlookahead_m/T、入力q/b/snapshot、PPや横加速度の参照を変更せず、通常preview、host.reset/step、乱数・履歴更新を追加しません。通常snapshotは各状態1回です。
- 同梱MetaDrivePreviewProviderは通常previewで保持した経路・projection・共通invalidを利用し、追加のhost API読取りをしません。envはproviderの私有_routeへ依存しません。
- host API／単位の欠落・非有限値・戻り値の型違反はエラーです。幾何無効へ丸めたり等速版へfallbackしたりしません。機能3の実効On時に取得口がなければ構築時に明示的エラーとなります。

モデルの設定キーはprediction_motion_modelだけを追加します。既定constant_speed、許容値constant_speed / constant_acceleration。
Off時も型を検証し、metadata比較では実効On時だけ方式を照合します。旧schemaへ新キーを挿入せずhelperを使用します。

## 確認と今回差分だけの撤去

READMEの軽量テスト、CLI設定読込、可能なら既存assetsで数stepのsmokeを実行します。
import隔離やfake hostの成功は別PCの実接続確認とは分けて報告します。

撤去時は適用前原本・適用差分・現在の状態を三者比較します。
今回の機能3追加行だけを戻し、元からある機能1・2を保持します。元からあったlookahead・host接続・独自変更・後続変更を残します。
今回の新規ファイルも後続利用がないと確認したものだけ削除します。
競合時にバックアップを無条件で復元せず、その箇所を保留して報告します。
機能をOffにするだけなら設定を変更できますが、保存モデルとの実効設定の照合は引き続き必須です。
距離指定へ戻すにはTを省略し、対応する距離指定モデルを使用します。
