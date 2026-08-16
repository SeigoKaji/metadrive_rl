# Phase 0 コードウォークスルー

## 1. この文書の読み方

このPhase 0は、固定された `map="C"` でMetaDrive標準Observationを受け取り、9種類の離散Actionから操作を選び、MetaDrive標準Rewardを最大化するPolicyをSB3 PPOで学習する構成である。自作コードは、設定、環境生成、学習開始、評価、検査、保存を接続する。道路生成、車両物理、Observation、Reward、終了判定そのものは変更しない。

この文書の自作コード行番号は、完成後に `nl -ba` で再確認した番号である。内部ライブラリの場所と開始行は、Python 3.11.16の現在の `.venv` から `inspect.getsourcefile()` と `inspect.getsourcelines()` を実行して確認した。

- MetaDrive: `MetaDrive-0.4.3` tag、commit `5bf8ea8909c4643a4099a250e6f5fb89c695d8b4` のcheckoutをeditable installしている。実際のimport先は `.external/metadrive-src/metadrive/`、distribution/package versionは `0.4.2.3` である。
- Stable-Baselines3: `.venv/lib/python3.11/site-packages/` に実際にインストールされたSB3 2.9.0を参照している。

以下のMetaDrive/SB3監査は、予定のpathや旧Python 3.12環境ではなく、最終Python 3.11 runtimeが実際にimportしたsourceに基づく。

## 2. 全体の処理経路

```text
phase0_config.py
  公式環境設定・公式PPO設定・保存先
          │
          v
env_factory.py
  MetaDriveEnv生成 + 記録専用Monitor
          │
          ├───────────────┐
          v               v
      train.py        evaluate.py
  SubprocVecEnv/PPO    PPO.load/predict
          │               │
          v               v
  model + metadata    evaluation JSON/GIF

inspect_env.py ── raw環境の仕様、Action変換、raw/adapted check_env、random走行を検査
tests/         ── 公式設定、Gymnasium戻り値、checker adapterを検査
```

`Monitor`はエピソード統計を記録するだけであり、Observation、Action、Reward、終了条件を書き換えない。学習用にはMetaDriveの1-process 1-instance制約を守るため、各環境を `SubprocVecEnv` の別processへ配置する。評価用には単一のraw環境を使う。seed-only inspection adapterは `inspect_env.py` と対応testだけが使用し、学習・評価の環境経路には入らない。

## 3. 自作コードの責任分担

### 3.1 設定・環境生成・検査

| ファイル・行 | 処理 | 入力 | 出力 | 担当すること | 担当しないこと |
| --- | --- | --- | --- | --- | --- |
| `phase0_config.py:7-23` | 公式環境設定 | 公式例で指定された値 | `OFFICIAL_ENV_CONFIG` | MetaDriveへ渡す11項目を一か所に固定 | MetaDrive defaultの再定義、環境生成 |
| `phase0_config.py:25-33` | 公式PPO設定 | 公式例で指定された値 | `OFFICIAL_TRAINING_CONFIG` | seed、環境数、rollout長、総step、Policy等を一元管理 | PPO instance生成、学習 |
| `phase0_config.py:35-55` | seedと出力先 | 上記公式設定、project位置 | `Path`と派生定数 | scenario seedとRL seedの区別、保存先の共通化 | directory作成、ファイル書込み |
| `env_factory.py:17-26` | raw環境生成 | `OFFICIAL_ENV_CONFIG` | `MetaDriveEnv` | 公式設定のcopyを渡してsimulatorを生成 | PPO学習、Reward計算、描画 |
| `env_factory.py:29-66` | 学習環境生成 | rank、RL seed、Monitor保存先 | `Monitor[MetaDriveEnv]` | workerごとのspace seedと衝突しないCSV名を設定 | scenarioをRL seedへ変更、タスク変換 |
| `env_factory.py:69-90` | 評価環境生成 | RL seed、GIF記録意図 | raw `MetaDriveEnv` | 単一評価環境のspaceをseed | GIF frame生成、モデル推論 |
| `inspect_env.py:37-65` | seed-only checker adapter | checkerのGym seed、raw環境 | 同じraw環境のreset結果 | checker seedをspacesへ適用し、underlying scenarioを公式5へ固定 | step、spaces、Observation、Reward、終了値の変換、学習・評価 |
| `inspect_env.py:68-76` | 既知衝突の識別 | raw checkerのexception | `bool` | `[5:6)` assertionだけを既知のAPI意味衝突として特定 | 他のAssertionErrorの握りつぶし |
| `inspect_env.py:111-127` | version表示 | distribution metadata | version文字列 | Pythonと要求packageのversion可視化 | versionの推測、package導入 |
| `inspect_env.py:130-136` | Observation検証 | 環境、Observation | NumPy array | space包含、NaN/Inf不在をassert | Observation加工 |
| `inspect_env.py:139-156` | Action変換監査 | reset済みraw環境 | 9組の数値 | 実際の `EnvInputPolicy` を取得し全IDを変換 | 独自Action変換 |
| `inspect_env.py:159-191` | random走行 | raw環境、最大step数 | 検査summary | 最大50 stepの5戻り値、数値Reward、bool終了値、再resetを確認 | Policy学習、性能評価 |
| `inspect_env.py:194-277` | 環境総合検査 | 公式raw環境 | console検査結果 | 実効config、spaces、info、raw/adapted `check_env`、random走行、close | warning抑制、未知のfatal errorの隠蔽 |
| `inspect_env.py:280-297` | 検査CLI | stdout/stderr | `outputs/inspect_env.log` | terminalとログへ同時出力しtracebackを保存 | 検査失敗を成功として扱うこと |

`env_factory.py:20-22` のMetaDrive importは意図的に遅延している。最終runtimeではMetaDriveをimportできるが、環境構築が将来失敗した場合にも `inspect_env.py` が先にログを開き、exception全文を `outputs/inspect_env.log` へ残せる。これはtaskの変更ではない。

### 3.2 学習

| ファイル・行 | 処理 | 入力 | 出力 | 担当すること | 担当しないこと |
| --- | --- | --- | --- | --- | --- |
| `train.py:146-199` | CLI解析 | command-line引数 | `argparse.Namespace` | 公式defaultと正の整数・model名を検証 | 学習実行 |
| `train.py:205-224` | 学習準備 | CLI設定 | directory、picklable factory群 | 出力先作成、RL乱数seed設定、rank別 `partial` 作成 | MetaDrive instanceの親process内生成 |
| `train.py:226-239` | VecEnv・PPO生成 | factory群、`MlpPolicy`、`n_steps`、device | `SubprocVecEnv`、`PPO` | process並列環境とActor/Criticを持つ学習器の初期化 | Reward、車両物理、終了判定 |
| `train.py:252` | `model.learn()` | PPO、VecEnv、目標timesteps | 更新済みPPO | rollout収集とPPO更新をSB3へ開始指示 | 自作のloss計算やbackpropagation |
| `train.py:254-267` | モデル保存・再読込 | 更新済みPPO | `.zip`、hash、再読込済みmodel | 保存存在・size・SHA-256・space互換性の検証 | 追加学習、評価episode実行 |
| `train.py:269-317` | metadata保存 | 実行設定、version、artifact情報 | training metadata JSON | 再現用の事実と保存先を記録 | 成功していない処理の捏造 |
| `train.py:318-321` | 環境解放 | `train_env` | なし | 成否にかかわらず全workerをclose | processを残したまま例外終了 |
| `train.py:324-346` | 学習CLI entry point | CLI | exit status、console log | stdout/stderr複製、traceback記録、main guard | task定義 |

学習の中心は `train.py:252` の1行だが、その1行の内部でSB3が多くの仕事をする。具体的にはPolicyでActionをsampleし、全環境へ渡し、戻り値をrollout bufferへ貯め、advantageとreturnを計算し、minibatchごとにActor/Criticの損失を計算してPyTorch optimizerで重みを更新する。自作コードはその内部処理を再実装していない。

公式本学習では1 rolloutが `4 environments × 4096 steps = 16,384 transitions` である。SB3はrollout途中ではPPO更新を始めず、完全なrollout単位で進むため、要求値300,000に最初に到達するのは19 rollout後の311,296 transitionsである。これはtask変更ではなくon-policy rollout境界による丸めである。

### 3.3 評価

| ファイル・行 | 処理 | 入力 | 出力 | 担当すること | 担当しないこと |
| --- | --- | --- | --- | --- | --- |
| `evaluate.py:243-258` | モデル検証・読込 | `.zip` path、device | 読込済みPPO、model hash | model存在・size・SHA-256確認と `PPO.load()` | モデル更新 |
| `evaluate.py:259-295` | 出力・環境準備 | CLI、読込済みPPO | 評価先、単一raw環境 | JSON/GIF先を決定しmodelとenvのspace互換性を確認 | 学習用並列化 |
| `evaluate.py:297-306` | episode初期化 | raw環境 | 初期Observation、reset info | `reset()`をseed引数なしで呼び固定scenario 5を選ぶ | RL seedをscenario indexに流用 |
| `evaluate.py:308-309` | `model.predict()` | 現在のObservation | 離散Action ID | `deterministic=True`で保存Policyから推論 | 探索、loss計算、重み更新 |
| `evaluate.py:310-315` | `env.step()` | Action ID | 次Observation、Reward、2種の終了値、info | MetaDrive simulationを1 step進め結果を集計 | PPO更新 |
| `evaluate.py:317-330` | top-down frame記録 | 評価中のenv | renderer内frame | MetaDrive 0.4.3の実APIでheadless frameを記録 | 学習taskやObservationの画像化 |
| `evaluate.py:332-369` | episode終了・分類 | `terminated`、`truncated`、info | episode result | 両終了値を区別し、終了理由と全flagを保存 | 欠落info keyの無条件参照 |
| `evaluate.py:371-416` | GIF生成 | 記録frame | `.gif`またはerror trace | rendererの実signatureでGIFを生成・検証 | GIF失敗をPolicy評価失敗に改変 |
| `evaluate.py:417-420` | 評価環境解放 | raw環境 | なし | 成否にかかわらずclose | singleton環境を残すこと |
| `evaluate.py:422-459` | 評価JSON保存 | episode群、model/config情報 | evaluation JSON | reward、長さ、flag、終了理由、aggregateを保存 | 実行していないepisodeの補完 |
| `evaluate.py:462-483` | 評価CLI entry point | CLI | exit status、console log | stdout/stderrとtracebackの保存、main guard | 学習開始 |

### 3.4 Contract test

| ファイル・行 | 処理 | 入力 | 出力 | 担当すること | 担当しないこと |
| --- | --- | --- | --- | --- | --- |
| `tests/test_phase0_contract.py:19-30` | 公式config固定 | `OFFICIAL_ENV_CONFIG` | test結果 | canonical JSONのSHA-256で全key/value変更を検出 | 同じ設定dictの二重管理 |
| `tests/test_phase0_contract.py:33-58` | raw環境contract | 公式raw環境、random Action 1個 | test結果 | `Discrete(9)`、reset 2要素、step 5要素、型、space包含、closeを検証 | 300,000 step学習、性能保証 |
| `tests/test_phase0_contract.py:61-75` | checker adapter contract | 同じ単一raw環境 | test結果 | 実SB3 checker完走、scenario 5と公式config不変、closeを検証 | adapterの学習・評価への導入 |

各環境testは `finally` で必ずcloseする。adapter testも新しいMetaDrive taskを作らず、同じraw instanceを包んでcheckerのreset seedだけを調停し、検査後に `env.current_seed == 5` と公式config全項目を再確認する。

## 4. 1 stepで起きること

### 4.1 学習時の順序

1. 現在のObservationをPPOへ渡す。
   MetaDrive標準Observationは車両周辺とnavigationに関する観測値であり、simulator内部の完全なStateそのものとは限らない。自作Observationへの変換はない。

2. Actorが9 Actionの確率分布を計算する。
   `MlpPolicy` のActor側はObservationから9個のActionに対応するlogitを作り、離散Action用のcategorical distributionにする。学習中はこの分布からActionをsampleするため探索が含まれる。

3. Action IDを選ぶ。
   SB3が選ぶ値は `0` から `8` の整数である。自作コードはこのIDの意味を置換しない。

4. `EnvInputPolicy` がAction IDを連続制御へ変換する。
   3 steering bins、3 throttle/brake binsではsteering側が最速に変化する。実sourceが返す数値は次のとおりである。

   | Action ID | steering | throttle_brake |
   | ---: | ---: | ---: |
   | 0 | -1.0 | -1.0 |
   | 1 | 0.0 | -1.0 |
   | 2 | +1.0 | -1.0 |
   | 3 | -1.0 | 0.0 |
   | 4 | 0.0 | 0.0 |
   | 5 | +1.0 | 0.0 |
   | 6 | -1.0 | +1.0 |
   | 7 | 0.0 | +1.0 |
   | 8 | +1.0 | +1.0 |

   ここでは符号に「左」「右」の名称を推測で付けず、MetaDrive sourceが返す数値だけを示す。

5. MetaDriveが車両物理simulationを進める。
   `BaseEnv.step()` は入力をagentごとの形式へ前処理し、engineへ渡してphysics substepを進める。このtransitionはMetaDriveの責任であり、PPOや自作コードは車両位置を直接更新しない。

6. MetaDriveがObservation、Reward、終了条件を計算する。
   `MetaDriveEnv.reward_function()` が標準Rewardを、`MetaDriveEnv.done_function()` が到着、道路外、衝突等のtask終了を判定する。`horizon=500` 到達は通常 `truncated=True` で表現され、defaultの `truncate_as_terminate=False` は上書きしていない。

7. SB3が結果をrollout bufferへ保存する。
   各stepのObservation、Action、Reward、episode start、value予測、Action log probability等が保存される。終了episodeのresetやtime-limit bootstrapもSB3のrollout収集処理が扱う。

8. rolloutが一定量貯まるとPPOが重みを更新する。
   1環境あたり `n_steps` 件を収集した後、advantageとreturnを使って複数epochのminibatch更新を行う。ActorとCriticのgradientはPyTorchが計算し、optimizerがparameterを更新する。

### 4.2 評価時の1 step

評価では `evaluate.py:309` が `model.predict(obs, deterministic=True)` を呼び、`evaluate.py:310` がActionを `env.step()` へ渡す。戻ったRewardは集計にだけ使い、rollout bufferへ保存せず、loss、backpropagation、optimizer stepも実行しない。`terminated or truncated` が真になった時点でloopを終了するため、task終了とtime-limitの両方を正しく扱う。

## 5. ActorとCritic

### 5.1 Actor

Actorは「現在のObservationで各Actionをどの程度選びたいか」を表す9個のlogitを出力する。離散Actionなので、これをcategorical probability distributionとして扱う。学習中は分布からsampleし、評価中の `deterministic=True` では最も選好度の高いActionを選ぶ。

Actorの出力はAction IDまでである。Action IDから `[steering, throttle_brake]` への変換、車両物理、Reward計算はMetaDriveが担当する。

### 5.2 Critic

Criticは現在のObservationから状態価値 `V(s)`、すなわち「この状態から将来どの程度の割引累積Rewardが期待できるか」を予測する。実際のreturnとの差を使って価値関数を学習し、Actorを更新するadvantage推定にも使われる。

Criticは車両を直接操作しない。Criticの数値が `env.step()` へActionとして渡ることもない。操作を決めるのはActor側であり、Criticは学習信号の分散を下げ、各Actionが期待より良かったかを評価する役割を持つ。

### 5.3 PPOが更新を制限する意味

同じrollout dataに対してPolicyを急激に変えると、そのdataを収集した旧Policyとのずれが大きくなり学習が不安定になる。PPOは旧Policyと新PolicyのAction確率比を使い、その比をclipした目的関数とclipしない目的関数の小さい側を採用する。これにより、改善方向へ進みつつ一度の更新が過大になりにくい。

SB3のPPO lossにはActorのclipped policy lossだけでなく、Criticのvalue lossと探索を支えるentropy項も含まれる。`model.learn()`を呼ぶと、SB3がrollout収集、advantage/return計算、minibatch作成、loss計算、gradient計算、gradient clipping、optimizer step、log出力まで担当する。自作 `train.py` はそれらを個別に実装しない。

## 6. 学習と評価の違い

| 項目 | 学習 | 評価 |
| --- | --- | --- |
| Entry point | `train.py:324-346` | `evaluate.py:462-483` |
| モデル | `PPO(...)` で生成 | `PPO.load(...)` で読込 |
| Action選択 | Policy分布からsampleし探索を含む | `deterministic=True` |
| Reward利用 | advantage、return、lossを通して重み更新に使用 | episode統計への集計のみ |
| Critic | value予測を学習しActor更新を補助 | 重みを更新しない |
| 逆伝播 | あり | なし |
| モデル重み | rollout後に更新される | 変更されない |
| 環境数 | 公式本学習は4 process × 1環境 | 1 process × 1環境 |
| `Monitor` | rank別CSV記録あり | なし |
| 描画 | 原則なし | 必要に応じtop-down GIF |
| 終了条件 | SB3が `terminated` と `truncated` をVecEnvのdoneへ統合してrollout管理 | 自作loopで両flagを別々に保存し、`or` でloop終了 |
| 出力 | PPO `.zip`、training metadata、Monitor/TensorBoard/console log | evaluation JSON、console log、任意GIF |

評価でRewardが得られても、その値は学習に戻らない。反対に、学習では描画を行わず、simulationと最適化に計算資源を使う。

## 7. ライブラリ内部ソース監査

### 7.1 実runtimeのMetaDrive 0.4.3 editable install

以下はPython 3.11.16の `.venv` から実際にimportされるsourceである。`metadrive.__file__` は `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.external/metadrive-src/metadrive/__init__.py` を指し、checkoutはtag `MetaDrive-0.4.3`、commit `5bf8ea8909c4643a4099a250e6f5fb89c695d8b4`、distribution versionは `0.4.2.3` である。開始行は実runtimeの `inspect.getsourcelines()` で測定し、source全文は転載せず責任だけを要約する。

| Class / function | ローカルsource・開始行 | 責任の要約 |
| --- | --- | --- |
| `MetaDriveEnv.reward_function` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.external/metadrive-src/metadrive/envs/metadrive_env.py:245` | lane方向の移動、速度、到着、道路外、衝突から標準Rewardとroute completion情報を計算する。終了時は該当terminal reward/penaltyでdense rewardを置換する。 |
| `MetaDriveEnv.done_function` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.external/metadrive-src/metadrive/envs/metadrive_env.py:132` | 到着、道路外、各種衝突、max stepのflagを作り、configに従ってtask上のdoneを決める。 |
| `BaseEnv.step` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.external/metadrive-src/metadrive/envs/base_env.py:435` | Action前処理、simulator step、Observation/Reward/終了/infoの収集を順に接続しGymnasiumの5戻り値を返す。 |
| `EnvInputPolicy.get_input_space` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.external/metadrive-src/metadrive/policy/env_input_policy.py:50` | global configからcontinuous Box、MultiDiscrete、またはDiscreteのAction spaceを構築する。今回は `3 × 3 = Discrete(9)`。開始行50は `@classmethod` decoratorを含む。 |
| `EnvInputPolicy.convert_to_continuous_action` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.external/metadrive-src/metadrive/policy/env_input_policy.py:40` | 離散Action IDをsteeringとthrottle/brakeの `[-1, 1]` binへ変換する。 |
| `EnvInputPolicy.act` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.external/metadrive-src/metadrive/policy/env_input_policy.py:26` | 外部Actionを取得し、必要ならspace check、離散変換、clipを行って車両制御値を返す。 |

`BaseEnv.step()` 自体が標準Rewardの式を所有するわけではない。stepは処理を編成し、具体的なRewardと終了判定はsubclassである `MetaDriveEnv` の2関数へ委譲する。逆に、`EnvInputPolicy` はPPO lossやActor/Criticを知らず、受け取ったActionを車両制御値へ変換するだけである。

### 7.2 実インストール済みStable-Baselines3 2.9.0

以下は `.venv` で実際にimportされるsourceである。

| Class / function | ローカルsource・開始行 | 責任の要約 |
| --- | --- | --- |
| `PPO` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.venv/lib/python3.11/site-packages/stable_baselines3/ppo/ppo.py:18` | PPO固有のhyperparameterとon-policy学習器を構成する。道路、物理、Rewardは定義しない。 |
| `OnPolicyAlgorithm.learn` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.venv/lib/python3.11/site-packages/stable_baselines3/common/on_policy_algorithm.py:300` | 目標timestepsまでrollout収集と `train()` を反復し、callbackとlogを管理するmain loop。 |
| `OnPolicyAlgorithm.collect_rollouts` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.venv/lib/python3.11/site-packages/stable_baselines3/common/on_policy_algorithm.py:162` | PolicyからAction/value/log-probabilityを得てVecEnvをstepし、transitionをrollout bufferへ保存する。 |
| `PPO.train` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.venv/lib/python3.11/site-packages/stable_baselines3/ppo/ppo.py:184` | rollout dataからclipped policy loss、value loss、entropy等を計算し、PyTorchでminibatch更新する。 |
| `SubprocVecEnv` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.venv/lib/python3.11/site-packages/stable_baselines3/common/vec_env/subproc_vec_env.py:79` | environment factoryごとに別processを起動し、pipe経由でreset/step/closeをまとめる。 |
| `Monitor` | `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.venv/lib/python3.11/site-packages/stable_baselines3/common/monitor.py:17` | wrapperとしてepisode reward、length、elapsed time等を記録する。taskのReward式は変更しない。 |

責任の流れは次のように整理できる。

```text
OnPolicyAlgorithm.learn
    ├─ collect_rollouts
    │    ├─ Policy Actor/CriticでAction・value等を計算
    │    ├─ SubprocVecEnv.step
    │    │    └─ 各workerのMonitor → MetaDriveEnv.step
    │    └─ rollout bufferへ保存
    └─ PPO.train
         └─ loss、backpropagation、optimizer step
```

## 8. `check_env` とMetaDrive scenario seedの互換性

これはPython 3.11 runtimeで実際に再現・確認したAPI互換性問題である。

SB3 2.9.0の `check_env()` は、Gymnasium APIがseed引数を受理するか確認するため、内部で必ず `env.reset(seed=0)` を呼ぶ。

- `check_env`: `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.venv/lib/python3.11/site-packages/stable_baselines3/common/env_checker.py:467`
- `env.reset(seed=0)` の呼出箇所: `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.venv/lib/python3.11/site-packages/stable_baselines3/common/env_checker.py:494`

一方、MetaDrive 0.4.3では `reset(seed=...)` のseedを一般的なRL乱数seedではなくscenario indexとして解釈する。

- `BaseEnv.reset`: `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.external/metadrive-src/metadrive/envs/base_env.py:512`
- scenario index範囲の検査: `/home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0/.external/metadrive-src/metadrive/envs/base_env.py:918` の `_reset_global_seed`

今回の公式設定は `start_seed=5`、`num_scenarios=1` なので許されるscenario indexは5だけ、半開区間では `[5, 6)` である。Python 3.11での実行結果も、raw公式環境への `check_env(env, warn=True)` が `AssertionError: scenario_index (seed) should be in [5:6)` になることを示した。これはraw環境のObservationやstepが不正だからではなく、同じ `seed` 引数に対するSB3とMetaDriveの意味が異なるためである。

### 8.1 raw checkerを先に実行する理由

`inspect_env.py:237-248` は最初に指定どおりraw環境へcheckerを実行し、fatalのtype、message、tracebackを隠さず `outputs/inspect_env.log` へ残す。`inspect_env.py:68-76` は、そのexceptionが既知の `[5:6)` seed衝突と完全に一致する場合だけ互換性分岐を許可する。他のAssertionErrorやchecker errorは既知問題に偽装せず、最終的にfatalとして再送出される。

したがって、raw checkerを「PASSした」とは報告しない。最終runtimeの事実は次のとおりである。

```text
raw check_env: fatal
  AssertionError: scenario_index (seed) should be in [5:6)

seed-only adapted check_env: PASS
  official scenario: 5
  task config: unchanged
```

### 8.2 seed-only inspection adapterの責任

既知衝突の場合だけ、`inspect_env.py:249-264` は同じ単一raw MetaDrive instanceを `inspect_env.py:37-65` の `_CheckEnvFixedScenarioAdapter` で一時的に包み、SB3 checkerをもう一度実行する。

adapterが行うことは2点だけである。

1. SB3 checkerが渡すGym seed 0をAction/Observation spaceのsampling seedへ適用する。
2. underlying raw MetaDriveのresetには公式scenario seed 5を渡す。

`gym.Wrapper`の標準委譲により、`step()`、Observation space、Action space、Observation値、Reward、`terminated`、`truncated`、`info`はraw環境のものをそのままcheckerへ渡す。adapterは非空のreset optionsを黙って無視せず `NotImplementedError` にし、reset後にはunderlying `current_seed == 5` をassertする。

このadapterは、raw checkerの結果を上書きして成功に見せるものではない。ログにはraw fatalとadapter PASSの両方を残す。adapterが別のfatalを返せば `checker_error` に保存され、`inspect_env.py:276-277` で再送出される。adapterがPASSした後も、`inspect_env.py:270-274` はraw環境で最大50 stepのrandom走行を行い、`finally`でcloseする。

`tests/test_phase0_contract.py:61-75` も実SB3 checkerをadapter経由で実行した後、同じraw instanceの `current_seed == 5` と全公式config値の不変を検証する。

### 8.3 学習・評価taskには入らない

seed-only adapterは検査専用であり、`env_factory.py`、`train.py`、`evaluate.py`からimportも使用もされない。

- `env_factory.py:55-61` と `:83-87` はAction/Observation spaceだけをRL seedでseedし、`env.reset(seed=RL_SEED)` を呼ばない。
- `train.py:215-224` は公式例と同じく `set_random_seed()` とworker別space seedを使う。`PPO(seed=...)` はSB3がVecEnvへseedを転送するため指定していない。
- `evaluate.py:299-300` はraw環境の `reset()` を引数なしで呼び、MetaDriveが唯一のscenario 5を選ぶようにする。

公式configの `start_seed` を0へ変更せず、Observation、Action、Reward、終了条件も変更していない。adapterはSB3 checkerとMetaDrive reset APIの意味だけを検査時に調停するため、学習・評価taskは公式ミニ例のままである。

## 9. 誰が何をしないか

最後に、混同しやすい境界をまとめる。

- `train.py` は「学習を開始する」が、道路、車両物理、Reward、終了条件を定義しない。
- `evaluate.py` は「保存済みmodelで推論し結果を集計する」が、重みを更新しない。
- SB3 ActorはAction IDを決めるが、IDをsteering/throttleへ変換しない。
- SB3 Criticは将来価値を予測するが、車両へActionを送らない。
- MetaDrive `EnvInputPolicy` はActionを連続制御値へ変換するが、Policy gradientを計算しない。
- MetaDriveはtransition、Observation、Reward、終了を計算するが、PPO lossやbackpropagationを実行しない。
- PyTorchはtensor計算とgradient更新を担うが、driving taskの意味を知らない。
- `Monitor` はepisode統計を記録するが、MetaDrive標準Rewardを再計算・加工しない。
- `_CheckEnvFixedScenarioAdapter` はcheckerのreset seedだけを調停するが、学習・評価、step、Reward、終了条件へ参加しない。

この境界を維持しているため、ファイル分割、検査、logging、保存、GIF対応を追加しても、Phase 0の学習task自体は公式ミニ例のままである。
