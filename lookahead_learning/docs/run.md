# 実行手順

lookaheadの本番経路は、ホストリポジトリ直下の通常の train.py と
evaluate.py です。学習と評価には同じTOMLを渡し、lookaheadの距離と報酬係数を
コマンドラインで再指定しません。

## 通常の学習と評価

移植先のrootで次を実行します。

~~~bash
python train.py --config configs/your_experiment.toml
python evaluate.py --config configs/your_experiment.toml
~~~

このrepositoryで動作例として使う設定は
configs/official_start_lane_return_lookahead.toml です。これは参照用のTOMLであり、
移植先へconfigs/フォルダを丸ごとコピーする前提ではありません。移植先の既存TOMLへ
[lookahead] tableを追記し、hostが持つ環境・scenario・PPO設定を維持します。

~~~toml
[lookahead]
lookahead_m = 6.0
pp_weight = 0.0
~~~

[lookahead] を省略するとraw host環境のままbaselineです。tableがあり
pp_weight=0.0ならraw観測へ注視点3値を追加します。正のpp_weightなら同じ3値に
PP操舵不一致ペナルティを加えます。lookahead_mは経路に沿った弧長[m]、
pp_weightは追加ペナルティ係数です。rawの観測幅はホストのDで決まり、wrapper後は
D+3になります。

## 実行時のcall path

同じ設定が次の順に伝わります。

~~~text
TOML
  -> configs.experiment_config.select_experiment
  -> ExperimentProfile.lookahead_config
  -> train.py/evaluate.py
  -> env_factory.make_training_env/make_evaluation_env
  -> env_factory.make_env
  -> adapter.wrap_lookahead_env
  -> LookaheadEnv
  -> 既存host Env（StartLane系Subclassを含む）
~~~

wrapperの位置は raw Env の外側、trainingでは既存Monitor の内側です。既存Subclassが生成する
observation、reward_function、終了条件、Action適用を変更せず、LookaheadEnvは
基底envのreset/step返却値を使って追加3値と任意の追加報酬を計算します。
基底reward_functionをwrapperから再呼出しません。

移植先の既存 start_lane_env.py にあるStartLane系Subclassの入力、報酬、終了条件を
基底Envとして保持します。評価側のreset/stepはLookaheadEnvを通し、MetaDrive固有の
custom propertyやrender対象は env.unwrapped から取得します。

train.pyは解決済みlookahead_configをfactoryへ渡し、PPOを保存する前に
lookahead_learning.checkpoint.set_lookahead_model_metadata(model, config)を呼びます。
ZIPには
model.lookahead_config と model.lookahead_schema_version=1 が保存され、保存後の
再読込でも検証されます。evaluate.pyはPPOロード直後に
lookahead_learning.checkpoint.validate_lookahead_model_metadata(model, config)を呼び、
選択TOMLとZIPの設定を照合します。checkpoint.pyは標準ライブラリだけで動作し、
設定はZIP内属性だけで検証します。追加ファイルやハッシュ照合は使いません。
学習・評価JSONにはlookahead dictまたはNoneを記録します。

## 設定の境界

次の2値だけがlookaheadのTOML設定です。

| キー | 検証 | 意味 |
|---|---|---|
| lookahead_m | 有限で正 | 経路に沿った注視距離[m] |
| pp_weight | 有限で0以上 | 追加PP不一致ペナルティ係数 |

raw prefixの順序・意味・encoding・正規化、車両単位、Actionの符号、物理step幅、
decision repeat、報酬関数、終了条件はhostとadapterの責務です。raw幅Dを特定の幅へ
固定、padding、切り捨ては行いません。PPの車軸長と最大操舵角はadapterがhostの
定義から読み、数値だけで単位や符号を推測しません。

注視3値の順序・encoding・報酬定義を変更するときは
lookahead_schema_versionを更新します。ソースコメントの変更は通常checkpointの利用
拒否条件ではありません。評価で拒否される
のはZIPのlookahead設定と選択TOMLが一致しない場合や、schemaが未対応の場合です。

## 通常入口と軽量確認

新しい学習・評価ではrootの同名スクリプトだけを使います。rootの --help は
MetaDriveなどホストの通常依存関係を必要とします。

~~~bash
python train.py --help
python evaluate.py --help
python train.py --config configs/your_experiment.toml
python evaluate.py --config configs/your_experiment.toml
~~~

テストを同梱した場合は、runtimeの契約と幾何を確認できます。

~~~bash
python -B -m unittest -v lookahead_learning.test_checkpoint lookahead_learning.test_geometry lookahead_learning.test_env
~~~

test_checkpoint.pyだけは標準ライブラリのみで実行できます。test_env.pyと
test_geometry.pyはNumPy、Gymnasiumなどruntime依存関係を必要とします。これは短時間の
契約確認で、実simやPPOの長時間学習を代替しません。テストやdocsは本番起動の必須入力
ではなく、移植時の補助です。

## 生成物と運用

生成物の場所や名前はTOMLのname、default_model_name、
evaluation.output_prefixに従います。学習runにはmodel ZIPと
training_metadata.json、評価runにはevaluation.jsonとstep traceが保存されます。
同じTOMLを使ったrunを単位にmodelとJSONを保管し、TOMLを変更した場合は別run名を
使います。

既存baseline ZIPを注視ありTOMLで読み込む場合や、lookahead_mまたはpp_weightが
異なるTOMLで読み込む場合は、評価側のmetadata検証で停止します。既存baselineを
baselineとして評価する場合は [lookahead] を省略したTOMLを渡します。
