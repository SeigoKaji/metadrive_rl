# 前方注視を使う強化学習

lookahead_learning は、ホストの既存環境を包んで前方注視点の3値と任意の
Pure Pursuit（PP）不一致ペナルティを追加するruntimeです。運用時の入口は
ホスト直下の通常の train.py と evaluate.py です。

## 通常の実行

同じTOMLを学習と評価へ渡します。移植後の最小例は次のとおりです。

~~~bash
python train.py --config configs/your_experiment.toml
python evaluate.py --config configs/your_experiment.toml
~~~

このrepositoryで動作例として使う設定は
configs/official_start_lane_return_lookahead.toml です。これは参照用のTOMLであり、
移植先へconfigs/フォルダを丸ごとコピーする前提ではありません。

[lookahead] を省略するとbaselineです。tableがあり pp_weight=0.0ならraw観測へ
注視点3値を追加し、正のpp_weightなら同じ観測にPP不一致ペナルティも加えます。
lookahead_mは経路に沿った弧長[m]、pp_weightは追加ペナルティ係数です。raw観測の
幅Dは移植先hostで決まり、wrapper後はD+3になります。数を合わせるためのpaddingや
切り捨ては行いません。

checkpoint.py は Python 標準ライブラリだけで動作します。config loaderでは
次を一度呼びます。

~~~python
from lookahead_learning.checkpoint import resolve_lookahead_config

lookahead_config = resolve_lookahead_config(raw.get("lookahead"))
~~~

学習時は解決済み設定を model.lookahead_config と
model.lookahead_schema_version=1 としてPPO.save()前にZIPへ保存します。評価時は
PPO.load()直後に選択TOMLとZIPの設定を照合します。checkpointの属性はZIP内にあり、
追加ファイルやハッシュ照合をゲートには使いません。注視3値の順序・encoding・
報酬定義を変更する場合はschema versionを更新してください。

## 実装を追う4つの薄いhook

移植先で確認・編集する範囲は、既存rootの config loader、共通env factory、
train.py、evaluate.py の4つです。

1. **config loader**: configs.experiment_config.select_experiment() が
   [lookahead]を検証し、ExperimentProfile.lookahead_configにdictまたはNoneを
   保存します。
2. **train.py**: profileのlookahead_configを
   make_training_env(..., lookahead_config=...)へ渡し、PPO.save()前に
   lookahead_learning.checkpoint.set_lookahead_model_metadata(model, config)を呼びます。
3. **evaluate.py**: 同じlookahead_configを
   make_evaluation_env(..., lookahead_config=...)へ渡し、PPO.load()直後に
   lookahead_learning.checkpoint.validate_lookahead_model_metadata(model, config)を
   呼びます。reset/stepはwrapperを通し、MetaDrive固有の属性やrenderは
   env.unwrappedから読みます。
4. **共通env factory**: raw Envを作ったあと、設定がある場合だけ次の処理を行い、
   training側の既存factoryがその結果をMonitorで包みます。

   ~~~python
   from lookahead_learning.adapter import wrap_lookahead_env

   if lookahead_config is not None:
       raw_env = wrap_lookahead_env(raw_env, **lookahead_config)
   ~~~

   既存の学習用Monitorはこのwrapperの外側へ置き、評価側にMonitorを新設しません。
   evaluatorの単一環境はwrapperを直接使います。factoryは既存Envの生成責務を保ち、
   hostの StartLane系Subclass と LookaheadEnv の順序を変更しません。

adapter.py の MetaDrivePreviewProvider は有効なlookahead全てで使い、
MetaDrivePPProvider は pp_weight>0 の場合だけ使います。LookaheadEnvは基底環境の
reset/step返却値を使って追加値を計算し、既存のobservation prefix、報酬、終了条件、
Action適用経路やreward_functionを再実装・再呼出ししません。

host固有の入力・車両・Navigationの意味と単位はadapterの監査対象です。
注視座標の観測正規化はruntimeで10mを基準に行いますが、raw prefixの正規化や
車両単位を推測して変更しません。

## 文書

| 文書 | 内容 |
|---|---|
| [観測入力と報酬関数の仕様](methods.md) | 観測の構成、注視点の生成・正規化、PP 参照と追加報酬の定義 |
| [実行手順](run.md) | 通常train/evaluate、軽量テスト |
| [移植手順](porting.md) | 必要なruntime、4つの薄いhook、移植後の運用確認 |
| [GitHub Copilot向け移植プロンプト](copilot_porting_prompt.md) | 小さな移植依頼として貼れる指示文 |

## 配布と確認

本番に必要なのは adapter.py、checkpoint.py、env.py、geometry.py、
__init__.py です。test_checkpoint.py、test_env.py、test_geometry.py は
移植後の契約確認用で、本番起動には必要ありません。生成済みモデル、ログ、
assets、bytecodeは配布物へ含めません。

テストを同梱した場合の最小確認は次のとおりです。

~~~bash
python -B -m unittest -v lookahead_learning.test_checkpoint lookahead_learning.test_geometry lookahead_learning.test_env
~~~

test_checkpoint.pyだけは標準ライブラリのみで実行できます。test_env.pyと
test_geometry.py、およびrootの train.py/evaluate.py --help にはホストの依存関係が
必要です。この確認は短時間の契約・幾何テストで、実simやPPOの長時間学習は
依存関係を用意した移植先で通常の train.py / evaluate.py を使って別途実行します。
