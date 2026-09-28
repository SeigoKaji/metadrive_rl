# 通常入口への移植と既存接続の差分更新

最初に渡す文書は [copilot_porting_prompt.md](copilot_porting_prompt.md) です。
移植用ファイルは **incoming/lookahead_learning/** に一時配置します。
実際の稼働フォルダとの比較・バックアップ前に上書きしません。
仕様を確認する必要がある場合だけ [time_prediction.md](time_prediction.md) の該当節を読みます。

## 配布範囲と保護

[../PORTABLE_FILES.txt](../PORTABLE_FILES.txt) は移植元rootからの相対パスです。
そのファイル群を同じ構造で運ぶだけで、実行コード・必要テスト・3設定例・最小文書がそろいます。
rootのtrain.py/evaluate.py/env_factory.py/start_lane_env.pyはコピー対象ではありません。
モデル・動画・画像・Excel・assets・無関係なレポートも含みません。

移植先rootでコードを直接上書きするコピーコマンドは使いません。
次は比較用配置の例です（SOURCE_BUNDLEはPORTABLE_FILESだけを含む配布root）。

```bash
mkdir -p incoming
cp -a /path/to/SOURCE_BUNDLE/lookahead_learning incoming/
git status --short
git diff -- lookahead_learning
```

incomingに既存配置がある場合も上書きせず、新しい一時ディレクトリを使います。
Copilotは変更予定を特定してから、新しい空のバックアップ先へ原本・未コミットdiff・HEAD・元から無かったファイル一覧を保存します。
適用後のdiffとハッシュも保管します。独自adapterを丸ごと置換しません。

| 実接続の状態 | 作業 |
|---|---|
| 未導入 | 下記4接続を既存の通常入口に追加 |
| 既存版 | 設定・wrapper・metadata接続を再利用し、新機能の不足差分だけ追加 |
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
ZIP内schema3、旧1/2の読み取り互換はhelperに任せます。旧ZIPは書き換えません。
移植先が独自のモデル属性保存除外を持つ場合、上記2属性が実際のZIPへ入ることを確認します。

### 4. 既存ログ

step traceで `info["lookahead_learning"]` をそのまま保存していれば変更不要です。
episodeで選択して保存する場合だけ既存リストに以下を追加します。

```python
"episode_r_prediction", "prediction_episode",
```

この移植元でもruntimeのroot変更は **evaluate.pyのepisode出力リストへの上記2キー追加だけ** です。
その他のroot差分は既存の通常入口・worker保持を検証するテストです。rootの実装を配布物へコピーする必要はありません。
成功率・速度・進捗・横ずれ・操舵変化量は既存評価出力/step traceを使い、新規レポート基盤は作りません。

## host adapterを維持する

`MetaDrivePreviewProvider`はreset時の固定参照経路と開始車線を継続利用します。
MetaDrive固有の意味・単位を移植先ソースで確認してください。
独自reader/providerの契約は [仕様1・5節](time_prediction.md) にあります。
契約不明ならその箇所を保留し、推測で速度・位置・曲率を作りません。
開始車線クラス名が異なるhostに、移植元start_lane_env.pyの必須importを追加しません。
既存の横加速度報酬を使用中なら、そのLateralReference/radius_reader接続も保持します。
今回の予測報酬は半径・PP・操舵モデルAPIを新たに要求しません。

## 確認と今回差分だけの撤去

READMEの軽量テスト、CLI設定読込、可能なら既存assetsで数stepのsmokeを実行します。
import隔離やfake hostの成功は別PCの実接続確認とは分けて報告します。

撤去時は適用前原本・適用差分・現在の状態を三者比較します。
今回追加した行だけを戻し、元からあったlookahead・host接続・独自変更・後続変更を残します。
今回の新規ファイルも後続利用がないと確認したものだけ削除します。
競合時にバックアップを無条件で復元せず、その箇所を保留して報告します。
機能をOffにするだけなら設定を変更できますが、保存モデルとの実効設定の照合は引き続き必須です。
距離指定へ戻すにはTを省略し、対応する距離指定モデルを使用します。
