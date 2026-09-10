# 移植手順

lookahead_learning は既存のMetaDriveホストへ追加する小さなruntimeです。
移植先の start_lane_env.py にあるStartLane系Subclassの入力生成、reward_function、
終了条件、Action適用を基底Envとしてそのまま使い、その外側にLookaheadEnvを
compositionで置きます。既存Subclassを書き換えたり、基底の観測・報酬を再実装したり、
observe/stepをwrapperから繰り返し呼び出したりしません。

## 配布するフォルダ

移植元から lookahead_learning/ フォルダを移植先rootへ置きます。本番の最小runtimeは
次の5ファイルです。

~~~text
lookahead_learning/
├── __init__.py
├── checkpoint.py
├── adapter.py
├── env.py
└── geometry.py
~~~

移植後の契約を確認する軽量テストを同梱する場合は、次の3ファイルを追加します。

~~~text
lookahead_learning/
├── test_checkpoint.py
├── test_env.py
└── test_geometry.py
~~~

docsは読み物として任意です。診断や比較のための補助機能は通常の学習・評価に
必要ありません。生成済みの
outputs、models、logs、assets、bytecodeも移植しません。

コピー元と移植先の例です。

~~~bash
SOURCE_ROOT=/path/to/metadrive_rl-lookahead
HOST_ROOT=/path/to/metadrive-host
cp -R "$SOURCE_ROOT/lookahead_learning" "$HOST_ROOT/"
~~~

このフォルダには本番runtime 5ファイル、任意の契約テスト3ファイル、任意のdocsが
含まれます。必要なファイルだけを選ぶ場合でも、同じlookahead_learning/の中から
runtimeとテストをコピーします。

外部の configs/、train.py、evaluate.py、env_factory.py、start_lane_env.py をこの
フォルダからコピーしません。移植先の既存ファイルと既存依存関係を使います。

## 移植先で確認する3接続点

移植時に確認・編集するのは、既存hostへ設定を伝える次の接続点です。

1. **config loader**

   既存の configs.experiment_config.select_experiment() が読むTOMLへ
   [lookahead] tableを追加し、resolverで次のmappingまたはNoneへ変換します。

   ~~~python
   from lookahead_learning.checkpoint import resolve_lookahead_config

   lookahead_config = resolve_lookahead_config(raw.get("lookahead"))
   ~~~

   tableなしは None、tableありは lookahead_m（有限で正）とpp_weight（有限で0以上）
   です。checkpoint.pyは標準ライブラリだけでこの解決を行います。移植先の設定形式に
   合わせてresolverを短く接続し、専用mode CLIや別の運用経路へ複製しません。

2. **通常train/evaluateと共通env factory**

   rootの train.py と evaluate.py は同じ選択結果の lookahead_config を使います。
   train.py は make_training_env(..., lookahead_config=lookahead_config)、
   evaluate.py は make_evaluation_env(..., lookahead_config=lookahead_config) を呼び、
   共通factoryはraw host Envを作ったあと、設定がある場合だけ次を呼びます。

   ~~~python
   from lookahead_learning.adapter import wrap_lookahead_env

   if lookahead_config is not None:
       raw_env = wrap_lookahead_env(raw_env, **lookahead_config)
   ~~~

   wrapperはraw Envの外側、trainingでは既存Monitorの内側です。既存の学習用Monitorは
   このwrapperの外側へ置き、評価側にMonitorを新設しません。既存のenv.reset/step返却値を
   使い、基底reward_functionを再呼出ししません。trainは
   lookahead_learning.checkpoint.set_lookahead_model_metadata(model, config)を保存前に
   呼び、evaluateはPPOロード後に
   lookahead_learning.checkpoint.validate_lookahead_model_metadata(model, config)を
   呼びます。設定は model.lookahead_config と model.lookahead_schema_version=1 として
   PPO ZIP内へ記録し、追加ファイルやハッシュ照合は使いません。

3. **host adapter**

   adapter.wrap_lookahead_env がraw Envのobservation_space、reset/step、
   Navigation、vehicle、Actionの実際の契約を確認します。raw観測はflat Boxの1次元
   float32、幅Dをopaque prefixとして保持し、注視値3つを末尾へ足してD+3にします。
   Dを特定の幅へ固定、padding、切り捨て、既存prefixの再正規化はしません。
   lookaheadを有効にするとMetaDrivePreviewProviderが経路と注視点を読み、
   pp_weightが正の場合だけMetaDrivePPProviderが車軸長、最大操舵角、操舵符号の
   ソースと単位を読みます。数値の大小から単位や符号を推測しないでください。

接続後のcall pathは次のようになります。

~~~text
TOML
  -> configs.experiment_config.select_experiment
  -> ExperimentProfile.lookahead_config
  -> train.py/evaluate.py
  -> env_factory.make_training_env/make_evaluation_env
  -> env_factory.make_env
  -> adapter.wrap_lookahead_env
  -> LookaheadEnv
  -> 既存StartLane系Subclassのreset/step（報酬計算は基底Env内）
~~~

環境の内部順序は「既存Subclass/raw Env → LookaheadEnv → Monitor → VecEnv」です。
既存Subclassの入力、報酬、終了条件はこの順序で保持されます。評価側は
LookaheadEnvを通してreset/stepし、MetaDrive固有のcustom propertyやrenderは
env.unwrappedから取得します。

## 設定ファイル

移植先では、そのhostが既に運用しているTOMLへ [lookahead] を追記します。

~~~toml
[lookahead]
lookahead_m = 6.0
pp_weight = 0.0
~~~

lookahead_mは経路に沿った弧長[m]、pp_weightは追加PP不一致ペナルティ係数です。
tableを省略すればbaseline、pp_weight=0.0なら注視点3値だけ、正値なら注視点と
追加PP報酬になります。学習と評価へ同じTOMLを渡します。

このrepositoryにある configs/official_start_lane_return_lookahead.toml は、この
repositoryだけの設定例です。移植先へ configs/ をコピーする前提ではなく、hostの
既存TOMLへ値を追記し、schema、環境設定、scenario範囲、PPO設定を維持します。
既存TOMLのroot schemaを変更する場合は、loaderの既存検証と衝突しないようにします。

## 依存と運用

移植先にはhostが通常使うPython、MetaDrive、Gymnasium、Stable-Baselines3、
NumPy、Panda3D assetsが既に必要です。このruntimeは依存インストールやassets
downloadを行いません。Pythonのimport pathを恒久変更する必要もありません。

まずhostが通常使う依存環境でroot入口と設定が解決できることを確認します。

~~~bash
cd /path/to/metadrive-host
python train.py --help
python evaluate.py --help
~~~

root入口の --help はMetaDriveなどhostの通常依存関係を必要とします。

次に同じ設定で学習・評価を実行します。

~~~bash
python train.py --config configs/your_experiment.toml
python evaluate.py --config configs/your_experiment.toml
~~~

評価時は、学習時に保存したZIPの lookahead_config と schema version が選択TOMLに
一致することを確認します。lookahead_mまたはpp_weightが異なるTOML、注視ありTOML
でbaseline ZIPを読む組み合わせは停止します。注視3値の順序・encoding・報酬定義を
変更する場合はschema versionを更新します。

テストを同梱した場合の最小確認は次のとおりです。

~~~bash
python -B -m unittest -v lookahead_learning.test_checkpoint lookahead_learning.test_geometry lookahead_learning.test_env
~~~

test_checkpoint.pyだけは標準ライブラリのみで実行できます。test_env.pyと
test_geometry.pyはruntime依存関係を必要とします。実simやPPOの長時間学習は必要な
環境で別途行い、軽量テストの成功だけから性能改善を判断しません。
