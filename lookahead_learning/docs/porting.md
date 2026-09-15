# 新規移植と既存移植の差分更新

更新した lookahead_learning/ を移植先rootへ配置したあと、
[Copilot向けプロンプト](copilot_porting_prompt.md) の内容をそのまま渡してください。
新規・既存更新・部分移植・対応済みを同じプロンプトで扱います。
報酬の詳細は [参照経路の必要横加速度](lateral_acceleration_reward.md)、
既存観測とPPは [methods.md](methods.md) に同梱しています。元repositoryや外部サイトを開くことは移植作業の前提ではありません。

## 差し替える前のバックアップ

**旧 lookahead_learning/、host独自adapter、変更予定hostファイル、未コミット差分を、上書き前に別の場所へ保存してください。**
更新済みフォルダを置いた後に旧独自変更が消えていた場合、プロンプトからは復元できません。
モデルと既存設定は保管し、別の比較設定・出力名を使います。認証・共有設定・AGENTSは変更対象外です。

フォルダコピーの例です。BACKUP_DIRは新しい空の退避先を指定します。

~~~bash
SOURCE_ROOT=/path/to/source
HOST_ROOT=/path/to/host
BACKUP_DIR=/path/to/backup-before-lookahead-update
mkdir -p "$BACKUP_DIR"
if [ -d "$HOST_ROOT/lookahead_learning" ]; then
    cp -a "$HOST_ROOT/lookahead_learning" "$BACKUP_DIR/"
fi
cp -a "$SOURCE_ROOT/lookahead_learning" "$HOST_ROOT/"
~~~

host側で書き換えるファイルの一覧が分かったら、その原本と対象ファイルの git diff も保存します。
バックアップ先に既存バックアップを重ねて上書きしないでください。
host固有adapterは今後可能な範囲でパッケージの外へ配置し、コピー更新で消えない構成にします。
今回のためだけに既存hostを大きく再配置する必要はありません。

## コピーする範囲

次を含む **lookahead_learning/ フォルダ全体** を持ち運びます。

~~~text
lookahead_learning/
├── __init__.py
├── checkpoint.py
├── adapter.py
├── env.py
├── geometry.py
├── lateral_acceleration.py
├── test_checkpoint.py
├── test_geometry.py
├── test_env.py
├── test_lateral_acceleration.py
├── test_lateral_env.py
├── test_portability.py
└── docs/  （仕様、移植プロンプト、既存の図・用語集）
~~~

共通の実行コード6ファイル、設定解決、仕様、テストがこの範囲でそろいます。
移植元rootの configs/、train.py、evaluate.py、env_factory.py、start_lane_env.py を必須コピーにしません。
モデル、実行ログ、bytecode、Simulator assetsは配布物に含めません。
docs/assetsの既存の図や編集用資料は同梱資料です。

## フォルダの存在では判定しない

更新済みフォルダを置いた時点で新schemaは既に存在します。完了状態はhostの実接続から判定します。

| 状態 | host側の根拠 | 作業 |
|---|---|---|
| 新規 | 設定・wrapper・metadata接続が未導入 | 通常の入口へ1回だけ接続 |
| 既存版からの更新 | D+3とwrapperは接続済み、新3キーやmetadata検証・出力の一部が不足 | 不足箇所だけ補う |
| 部分移植／独自改変 | 接続が片側だけ、キー再構築、独自providerや報酬処理が存在 | 独自変更を保ち、必要な契約だけ補う |
| 既に今回仕様を満たす | 5キーが全経路へ届き、schema互換・ログ・単一wrapperが動く | 検証のみ、不要な変更を作らない |

限定して読むのは config loader の許可キーとresolver、train/evaluateの呼出し、
共通factory/worker、実際のwrapper、checkpoint helper、host固有adapterです。
全repositoryの再設計や全資料の調査は不要です。

## 共有する接続契約

### 設定とmetadata

~~~python
from lookahead_learning.checkpoint import (
    resolve_lookahead_config,
    set_lookahead_model_metadata,
    validate_lookahead_model_metadata,
)

lookahead_config = resolve_lookahead_config(raw.get("lookahead"))
# このmapping全体をprofile、train/evaluate、factory/workerで保持する。
set_lookahead_model_metadata(model, lookahead_config)  # PPO.save前
validate_lookahead_model_metadata(model, lookahead_config)  # PPO.load直後
~~~

loaderが許可キーを旧2つに限定していないか、factory引数を2つに再構築していないかを確認します。
boolを含む型でmapping全体を運び、既定値・数式は同梱resolver/helperを共通源とします。
新項の設定例とv1/v2互換規則は [新報酬の設定節](lateral_acceleration_reward.md) を参照してください。
旧モデルはOffとして読み、検証時には変更しません。active/baselineや有効設定の不一致を無視しません。

### 共通factory

~~~python
from lookahead_learning.adapter import wrap_lookahead_env

# 既存のhost生成処理で作ったraw_envを使う。
if lookahead_config is not None:
    raw_env = wrap_lookahead_env(raw_env, **lookahead_config)
~~~

既存移植でこの接続があれば再利用します。新たなwrapperを重ねてD+6にしたり、報酬を二重加算したりしません。
raw Env → LookaheadEnv → 既存Monitor → VecEnv の順序を保ち、評価にはMonitorを新設しません。
reset/stepは既存wrapperチェーンを通します。内部属性の読み取りはenv.unwrappedでも構いませんが、
unwrapped.step/resetで既存wrapperを迂回してはいけません。

既存hostの入力生成・開始車線特徴・reward_function・終了条件・Action・シナリオ・並列数を保ちます。
raw幅Dは実際のflat float32 Boxから読み、既存prefixへ3値を一度だけ追加します。

### host adapterと評価出力

StartLane系クラス名やファイル配置が異なっても、同じ意味の接続点を探します。
同梱MetaDrive adapterの任意のstart-lane resolverはfallback可能ですが、
その特定ファイル名を別hostへの新たな必須依存にしてはいけません。
別のhost固有providerから [LookaheadEnv](../env.py) を直接構成することもできます。

座標・速度単位と意味・decision dt・開始車線参照をソースで監査します。
新項Onでは既存previewと同じS_proj/S_goalの LateralReference を追加し、
state_reader は平面速度の大きさ speed_m_s を返します。
半径APIが異なる場合は MetaDrivePreviewProvider(radius_reader=...) で
対象中心線の半径[m]を明示します。PPの曲率やlane.lengthから推測しません。
Offとゼロ重みでは新APIを要求しません。PP OffではMetaDrivePPProviderを構築しません。

既存の評価traceへ info["lookahead_learning"] を渡し、episode出力には
episode_r_base、episode_r_pp、episode_r_lateral_accel、episode_r_total、
lateral_accel_episodeを渡します。host側で報酬式・集計式を再実装する必要はありません。

## 移植後の確認と戻し方

hostの既存Python環境で実行します。新たな依存更新やassets downloadは行いません。

~~~bash
python -B -m unittest discover -s lookahead_learning -t . -p 'test_*.py'
python train.py --help
python evaluate.py --help
~~~

test_checkpoint、test_geometry、test_lateral_accelerationは標準ライブラリのみです。
envのfakeテストはNumPy/Gymnasium、Monitor/VecEnv確認はSB3を使います。
test_portabilityはこのフォルダのみを一時コピーし、元root・MetaDriveのimportを禁止した状態で
設定・純粋関数・fake hostテストを実行します。コピー側の依存環境はhostの既存環境を使います。

hostでも、旧schema v1モデル＋Off、PP Off＋新項On、On/Offとゼロ重み、
単一wrapper・D/D+3・報酬単一加算・Monitor合計を確認します。
同梱fixtureは新規、旧版接続、更新済み接続を模した境界テストであり、
実際の移植先やGitHub Copilotを実行した証明ではありません。
Simulatorが利用可能なら1環境をresetし数stepで確認します。長時間学習は別の比較実験です。

今回仕様を満たすhostで同じプロンプトを再実行した場合は検証のみで済ませます。
接続を戻す場合は、変更前に退避したhostファイルと対応TOMLを戻し、対応する旧モデルを使います。
新項だけをOffにする手順とモデル設定の整合は [新報酬の戻し方](lateral_acceleration_reward.md) を参照してください。
