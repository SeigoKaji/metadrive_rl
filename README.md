# MetaDrive RL (Stable-Baselines3)

このディレクトリは、MetaDrive公式ドキュメントの「Training > stable-baselines3」にある最小構成を、学習タスクを変えずに通常のPythonスクリプトへ分割したものです。固定された `map="C"` の道路でMetaDrive標準Observationを受け取り、9種類の離散Actionから操作を選び、MetaDrive標準Rewardを最大化しながら目的地へ向かうPolicyをSB3 PPOで学習します。

既存の公式再現を既定の `official` profileとして残しつつ、複数の手続き生成道路で学習し、未見scenarioで評価する `generalization` profileも選択できます。追加設定と実行方法は17章にまとめています。

> **現在の環境:** Python 3.12.3の `.venv` を標準 `venv` で作成し、packageはpipで管理します。公式MetaDrive sourceは同階層の `metadrive/` に置き、`main` commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` を `-e ../metadrive` でeditable installしています。現在の検証結果は `RUN_REPORT.md` を参照してください。

## 1. 今回のタスク

### 学習すること

- 道路とシナリオは `map="C"`、`num_scenarios=1`、`start_seed=5` に固定します。確認したsourceではブロックID `C` は `Curve` です。公式ミニ例と同じ対象を使い、複数マップやseedに対する一般化は扱いません。
- 交通量を0、事故生成確率を0にし、周辺車との交渉ではなく、自己車両の経路追従と目的地到達という最小の接続を検証します。
- `discrete_steering_dim=3` と `discrete_throttle_dim=3` の直積を単一の `Discrete(9)` とし、操作候補を小さくします。
- PPOと `MlpPolicy` を使います。これは公式ミニ例と同じ組合せであり、離散Actionを直接扱えて、MetaDriveとSB3の接続確認に必要十分だからです。

### 学習しないこと

この章で定義するPhase 0公式再現では、他車回避、交通交渉、障害物回避、複数マップ・複数scenario seedへの一般化は対象外です。また、独自Observation、独自Reward、独自終了条件、独自Policyネットワーク、独自Feature Extractor、独自Controller、連続Action、画像Observation、マルチエージェント、SAC/TD3との比較、ハイパーパラメータ探索は導入しません。`generalization` profileは後続の任意設定であり、この固定契約を置換しません。

`map="C"`、離散Action、PPOを採用する第一の理由は、公式例の学習問題そのものを再現し、後続研究の変更を混ぜる前に「MetaDrive → SB3 → PPO更新 → 保存モデルの評価」という配線を検証するためです。

## 2. 固定する契約

環境へ明示的に渡す設定は次の11項目だけです。Rewardや終了条件に関する設定は追加せず、確認したMetaDrive sourceのデフォルトを使います。

```python
OFFICIAL_ENV_CONFIG = {
    "map": "C",
    "discrete_action": True,
    "discrete_throttle_dim": 3,
    "discrete_steering_dim": 3,
    "horizon": 500,
    "random_spawn_lane_index": False,
    "num_scenarios": 1,
    "start_seed": 5,
    "traffic_density": 0,
    "accident_prob": 0,
    "log_level": 50,
}
```

契約を短くまとめると次のとおりです。

| 項目 | Phase 0の値 |
| --- | --- |
| Observation | MetaDrive標準 |
| Action | `Discrete(9)` |
| Reward | MetaDrive標準 |
| 終了条件 | MetaDrive標準 |
| シナリオ | 1個、`start_seed=5` |
| 交通量 / 事故生成 | `0` / `0` |
| RL algorithm | SB3 PPO |
| Policy | `MlpPolicy` |

本学習の設定も公式例に合わせます。

```python
OFFICIAL_TRAINING_CONFIG = {
    "seed": 0,
    "num_envs": 4,
    "n_steps": 4096,
    "total_timesteps": 300_000,
    "log_interval": 4,
    "policy": "MlpPolicy",
}
```

PPOのその他のハイパーパラメータはSB3のデフォルトを使います。`device` だけは実行環境を曖昧にしないためCLIで指定し、既定を `cpu` とします。これはSB3がPPO + `MlpPolicy`にCPUを推奨していることに合わせた性能上の既定値であり、ホストGPUが存在しないという判定ではありません。CUDA対応PyTorch環境では`--device cuda`を明示してGPU学習も実行できます。

RL seedは公式例どおり `set_random_seed(0)` で設定し、`PPO(seed=...)` は指定しません。SB3 2.9.0でPPOへseedを渡すとVecEnvにもseedが転送されますが、MetaDrive 0.4.3の `reset(seed=...)` は通常の乱数seedではなくscenario indexとして扱われます。今回の有効なscenario indexは5だけなので、RL seed 0を環境resetへ流すと固定scenarioを壊して失敗します。Action/Observation spaceの乱数だけはworkerごとにseedし、環境resetは引数なしで行います。

### Action IDの意味

確認した `EnvInputPolicy.convert_to_continuous_action()` は、単一のAction IDを `id % 3` でsteering、`id // 3` でthrottle/brakeへ割り当てます。各軸は3段階なので、変換値は次のとおりです。符号へ「左」「右」の名称は付けず、ソースから直接確認できる数値だけを示します。

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

## 3. MDPとの対応

| MDP要素 | 今回の担当 |
| --- | --- |
| State / Observation | MetaDrive標準Observation |
| Action | 9種類の離散Action |
| Transition | MetaDriveの車両物理・シミュレーション |
| Reward | MetaDrive標準Reward |
| Discount factor | SB3 PPOの `gamma` |
| Policy | SB3の `MlpPolicy` |
| Episode termination | MetaDriveの終了判定 |
| Timeout | `horizon=500` による `truncated` |

環境の完全な内部状態（State）と、Policyへ渡される観測（Observation）は厳密には同じとは限りません。このPhase 0ではMetaDriveが標準で生成するObservationだけをPolicyへ渡し、独自の状態表現は追加しません。`gamma` も自作せず、実際に導入できたSB3 2.9.0 PPOのデフォルト `0.99` を使います。

## 4. データフロー

```text
MetaDrive標準Observation
        ↓
SB3 PPOのActor
        ↓
離散Action ID 0～8
        ↓
MetaDrive EnvInputPolicy
        ↓
[steering, throttle_brake]
        ↓
MetaDrive車両物理
        ↓
次のObservation・Reward・terminated・truncated・info
        ↓
SB3のrollout buffer
        ↓
PPOによるActor/Critic更新
```

評価時はActorからActionを得てMetaDriveを進めるところまでは同じですが、rollout bufferへの学習用蓄積、逆伝播、重み更新は行いません。

## 5. 責任分担

| 処理 | 担当 |
| --- | --- |
| 道路生成 | MetaDrive |
| 車両物理 | MetaDrive |
| Observation生成 | MetaDrive |
| Reward計算 | MetaDrive |
| 終了判定 | MetaDrive |
| 離散Actionから連続制御への変換 | MetaDrive `EnvInputPolicy` |
| rollout収集の制御 | SB3 |
| Actor/Critic | SB3 |
| 損失計算 | SB3 |
| 逆伝播・重み更新 | PyTorch/SB3 |
| 環境の並列化 | SB3 `SubprocVecEnv` |
| エピソードログ | SB3 `Monitor` |
| 学習開始指示 | 自作 `train.py` |
| 保存モデルによる推論 | 自作 `evaluate.py` ＋ SB3 |

自作wrapperの `Monitor` は記録だけを担当し、Observation、Action、Reward、終了条件を変更しません。4個の学習環境はMetaDriveの「1プロセスにつき1環境」という制約を守るため、`SubprocVecEnv` で別プロセスへ1個ずつ配置します。

## 6. MetaDrive標準Reward

以下は、兄弟directoryへcheckout・editable installした `main` commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` にある `metadrive/envs/metadrive_env.py` の実際の設定と `MetaDriveEnv.reward_function()`、およびPython 3.12.3上の実 `env.config` の確認結果に基づきます。

| 要素 | このsourceのデフォルト | 内容 |
| --- | ---: | --- |
| 前進Reward | `driving_reward=1.0` | 現在lane上の縦方向進捗 `long_now - long_last` に掛ける |
| 速度Reward | `speed_reward=0.1` | `speed_km_h / max_speed_km_h` に掛ける |
| 到着Reward | `success_reward=10.0` | 到着stepの最終Rewardへ置換 |
| 道路外Penalty | `out_of_road_penalty=5.0` | 道路外stepの最終Rewardを `-5.0` へ置換 |
| 車両衝突Penalty | `crash_vehicle_penalty=5.0` | 車両衝突stepの最終Rewardを `-5.0` へ置換 |
| 物体衝突Penalty | `crash_object_penalty=5.0` | 物体衝突stepの最終Rewardを `-5.0` へ置換 |
| 横方向係数 | `use_lateral_reward=False` | Phase 0では上書きしないため係数は常に `1.0` |

通常stepのdense rewardは概念的に次の和です。`positive_road` は道路方向に応じて `+1` または `-1` になります。

```text
driving_reward × 縦方向進捗 × lateral_factor × positive_road
+ speed_reward × 正規化速度 × positive_road
```

`use_lateral_reward=True` のときだけlane中心からの横ずれに応じて `lateral_factor` が0～1へ縮みます。確認したsourceの実際のデフォルトは `False` であり、Phase 0では指定にも追加しません。

重要なのは終了stepの扱いです。関数はいったんdense rewardを `step_info["step_reward"]` に保存した後、到着、道路外、車両衝突、物体衝突、歩道衝突の優先順で、返却するRewardをterminal reward/penaltyへ**置き換えます**。つまり、到着時にdense rewardへ `+10` を加える実装ではありません。複数フラグが同時に立つ可能性があっても、評価JSONには元フラグをすべて残します。

## 7. `terminated` と `truncated`

`terminated` は到着、道路外、衝突など、MetaDriveが定義するタスク上の終了です。確認したsourceの標準終了判定には、到着、道路外（lane外や設定対象の連続線・歩道衝突を含む）、車両・物体・建物・人との衝突などが含まれます。Phase 0ではその標準設定を変更しません。

`truncated` は時間制限による打切りです。今回は `horizon=500` なので、エピソード長が500stepに達すると `max_step=True` と `truncated=True` になります。確認した `BaseEnv` の実際のデフォルトは `truncate_as_terminate=False` です。そのため、時間切れだけなら `terminated=False` のままです。タスク終了と時間切れが同じstepで発生すれば、両方が `True` になることもあります。

評価ループを `terminated or truncated` で止めるのは、どちらもそのエピソードから先へ `step()` してはいけない境界だからです。ただし、結果では両者を混ぜず、終了フラグと `success`、`out_of_road`、`crash_vehicle`、`crash_object`、`other_termination`、`max_step_truncation`、`unknown` の理由を区別します。

## 8. rollout単位と実timesteps

SB3のon-policy学習は、`n_steps × num_envs` 件を1 rolloutとして収集してからPPO更新を行います。`total_timesteps` は厳密な停止位置ではなく下限なので、最後のrolloutを途中で切らず、指定値を超えることがあります。

- 本学習: `4096 × 4 = 16,384` step/rollout。`total_timesteps=300,000` では `ceil(300,000 / 16,384) = 19` rolloutとなるため、正常に完了すれば収集量は **311,296** stepです。
- Smoke Test: `256 × 1 = 256` step/rollout。`total_timesteps=2,000` では8 rolloutとなるため、正常に完了すれば収集量は **2,048** stepです。

TensorBoardの記録間隔は `log_interval` に依存するため、最終scalarのstepと実際の収集量が一致しない場合があります。実収集量はtraining metadataの `actual_total_timesteps` を確認します。

## 9. 公式例から変更したもの

| 項目 | 公式例 | 今回 |
| --- | --- | --- |
| 学習タスク | 公式設定 | 同じ |
| Observation | 標準 | 同じ |
| Reward | 標準 | 同じ |
| Action | `Discrete(9)` | 同じ |
| PPO | `MlpPolicy` | 同じ |
| 本学習 | 4環境、300,000 step | 同じ |
| コード構成 | Notebook的な一連の例 | 複数ファイルへ分割 |
| 環境検査 | なし | `check_env` 等を追加 |
| Smoke Test | `TEST_DOC` 時の簡略実行 | 明示的な短時間設定 |
| 終了判定 | サンプルでは簡略化 | `terminated` と `truncated` を区別 |
| バージョン記録 | 最小限 | tag・commit・package versionを保存 |
| device | 例の実行環境へ委任 | CLIで明示、既定 `cpu`、CUDA検証時は`cuda` |
| GIF出力先 | `demo.gif` | `outputs/` へ整理 |

変更はコード分割、検査、記録、CLI、出力整理に限られます。Observation、Action、Reward、終了条件、シナリオという学習タスクは変更しません。

## 10. ソースとバージョンの確認結果

一次資料として公式ドキュメントと公式Git repositoryを使い、実際に兄弟directoryへcheckoutしてruntimeがimportしたsourceを優先します。

- Git remote: `https://github.com/metadriverse/metadrive.git`
- branch: `main`
- clone時のHEAD: `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`
- commit date / subject: `2025-05-11` / `force set store_map=False for ScenarioOnlineEnv (#841)`
- checkout状態: `main...origin/main`、作業tree変更なし（assetsはupstreamのignore対象）
- `metadrive/version.py` / distribution / assets version: `0.4.3`
- install元: sibling checkout `../metadrive`
- import先: `../metadrive/metadrive/__init__.py`

これは最新release tagを固定したcheckoutではなく、取得時点の公式`main` HEADです。upstreamの`setup.py`はPython 3.12を拒否せず、Python 3.8以上では`panda3d>=1.10.14`等を選びます。また公式workflowはPython 3.9 / 3.12のtest matrixを持ちます。今回のsourceを変更せず、Python 3.12.3からeditable installできることも実測しました。

2026-08-18の単一 `.venv` で実測した主なversionは、Python 3.12.3、pip 26.2.1、MetaDrive 0.4.3、SB3 2.9.0、Gymnasium 1.3.0、PyTorch 2.13.0+cu130、NumPy 2.5.2、Panda3D 1.10.16です。標準pipが解決したPyTorchはCUDA 13.0 buildですが、現在の実行コンテキストではGPU accessがOSによりblockされ、`torch.cuda.is_available()`は`False`、`nvidia-smi`もNVML初期化に失敗しました。この値だけからホストGPUの有無は判定しません。

raw環境への `check_env` にあるseed契約衝突は最新`main`でも残っています。SB3 2.9.0のcheckerは `env.reset(seed=0)` を呼びますが、MetaDrive側はその値を乱数seedではなくscenario indexとして扱い、許容範囲は `[5, 6)` です。`inspect_env.py`はこのraw failureを隠さず表示した後、検査時だけscenario 5を維持するseed adapterでも検査し、全体として終了コード0になりました。adapterは学習・評価には使いません。

参照先:

- [MetaDrive Training](https://metadrive-simulator.readthedocs.io/en/latest/training.html)
- [MetaDrive Action and Policy](https://metadrive-simulator.readthedocs.io/en/latest/action.html)
- [MetaDrive Reward, Cost, Termination, and Step Information](https://metadrive-simulator.readthedocs.io/en/latest/reward_cost_done.html)
- [MetaDrive Installing MetaDrive](https://metadrive-simulator.readthedocs.io/en/latest/install.html)
- [MetaDrive公式repository](https://github.com/metadriverse/metadrive)
- [Python 3.12対応を追加したupstream commit](https://github.com/metadriverse/metadrive/commit/bb0a0c64f776769340b7daff81c154c04c8aa3a7)
- [今回checkoutしたupstream commit](https://github.com/metadriverse/metadrive/commit/85e5dadc6c7436d324348f6e3d8f8e680c06b4db)
- [SB3 Custom Environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
- [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## 11. ファイル構成

```text
metadrive-workspace/
├── metadrive-rl/                     # このRL project
│   ├── .venv/                        # Python 3.12の単一local venv（git管理外）
│   ├── requirements.txt              # MetaDrive sibling pathと直接依存
│   ├── requirements.lock.txt         # 検証済みvenvのversion snapshot
│   ├── configs/                       # 実験設定をまとめたpackage
│   │   ├── phase0_config.py           # 公式設定値と出力先の一元管理
│   │   ├── generalization_config.py   # 複数scenario学習と未見評価の設定
│   │   └── experiment_profiles.py     # official/generalizationの選択
│   ├── env_factory.py                 # MetaDrive生成と記録専用Monitor
│   ├── inspect_env.py                 # version・空間・Action変換・check_env・random走行検査
│   ├── train.py                       # SubprocVecEnvとPPO学習・モデル保存
│   ├── evaluate.py                    # 保存モデルの読込と決定論的推論を統括
│   ├── evaluation_results.py          # 評価JSON/JSONLとGIF/MP4/PNGの保持・保存
│   ├── evaluation_visualization.py    # step telemetry収集と右パネル合成
│   ├── generate_evaluation_telemetry_guide.py # 編集可能なExcelガイドの生成
│   ├── EVALUATION_TELEMETRY_GUIDE.xlsx # 右パネルの編集可能な日本語ガイド
│   ├── README.md                      # タスク、設計、実行手順（本書）
│   ├── CODE_WALKTHROUGH.md            # 自作・library内部処理の行番号付き解説
│   ├── RUN_REPORT.md                  # 現在の環境と検証結果
│   ├── tests/                         # 環境contractと汎化設定test
│   ├── models/                        # 学習済みmodel
│   ├── logs/                          # Monitor / TensorBoard / console log
│   └── outputs/                       # 検査log、metadata、評価JSON、GIF/MP4/PNG
└── metadrive/                         # 公式mainの独立checkout（RL project外）
```

`metadrive/` は `metadrive-rl/` の内部ではなく同階層です。`requirements.txt` の `-e ../metadrive` はこの配置を前提とするため、install commandは `metadrive-rl/` をcurrent directoryとして実行します。`.venv/`、runtime cache、モデル、Monitor/TensorBoardログ、GIF・動画・画像はgit管理外です。依存lockは現在の単一venv用 `requirements.lock.txt` だけを管理します。

## 12. 環境構築

MetaDrive sourceとRL projectを同階層へ置き、RL project内にはsourceをcloneしません。仮想環境自体はPython標準の`venv`で作り、その中のpackage管理だけをpipで行います。activationには依存しません。

### 12.1 MetaDrive最新版を兄弟directoryへ取得

初回は任意の場所にworkspaceを作り、RL projectとMetaDriveを兄弟directoryとしてcloneします。

```bash
mkdir -p metadrive-workspace
cd metadrive-workspace
git clone https://github.com/SeigoKaji/metadrive_rl.git metadrive-rl
git clone --branch main --single-branch \
  https://github.com/metadriverse/metadrive.git metadrive
git -C metadrive status --short --branch
git -C metadrive rev-parse HEAD
```

すでにclone済みなら、local変更がないことを確認してfast-forwardだけを許可して更新します。

```bash
cd metadrive-workspace
git -C metadrive status --short --branch
git -C metadrive pull --ff-only origin main
git -C metadrive rev-parse HEAD
```

2026-08-18に取得したHEADは `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` です。「latest」は更新時点で動くため、学習metadataと報告には毎回実commitを記録します。

### 12.2 Python 3.12 venvとpip install

次のコマンドは `metadrive-rl/` をcurrent directoryとして実行します。`requirements.txt` の `-e ../metadrive` は兄弟checkoutを指します。

```bash
cd metadrive-workspace/metadrive-rl
python3.12 --version
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

実測ではPython 3.12.3、pip 26.2.1でinstallが終了コード0となり、`pip check`は `No broken requirements found` でした。system Pythonへの直接install、`sudo`、sourceのPython version check改変は行いません。

### 12.3 requirementsの役割と更新

`requirements.txt` は直接依存だけを管理します。

```text
-e ../metadrive
stable-baselines3[extra]
Pillow
opencv-python
openpyxl
pytest
```

`requirements.lock.txt` は検証済みPython 3.12 venvからeditable packageを除いてfreezeしたversion群に、portableな `-e ../metadrive` とMetaDrive commitを付けたsnapshotです。通常の最新版再解決は `requirements.txt`、同じ依存versionを再現するときは `requirements.lock.txt` を使います。ただしどちらもlocal source pathを参照するため、MetaDrive commit自体はGit checkoutで管理します。

upstreamを更新したときは依存metadataの変更も反映します。

```bash
git -C ../metadrive pull --ff-only origin main
git -C ../metadrive rev-parse HEAD
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

## 13. 実行手順

12章で作った単一 `.venv` を使います。通常のローカル環境では、追加の環境変数を指定する必要はありません。すべて `metadrive-rl/` をcurrent directoryとして実行します。

### 13.1 assets取得

```bash
.venv/bin/python -m metadrive.pull_asset
```

### 13.2 環境検査

version、適用config、Observation/Action space、Gymnasium API、`check_env`、random走行を確認します。

```bash
.venv/bin/python inspect_env.py
```

raw環境へのSB3 `check_env`では既知のscenario seed衝突を報告します。その後、検査専用adapterとrandom走行が成功すればcommand全体は終了コード0です。

### 13.3 pytest

```bash
.venv/bin/python -m pytest -q
```

環境contractとprofile設定を検査します。長時間学習はtestへ含めません。

### 13.4 Smoke学習

PPO初期化、rollout、勾配更新、保存、再読込までを短時間で確認します。

```bash
.venv/bin/python train.py \
  --timesteps 2000 \
  --num-envs 1 \
  --n-steps 256 \
  --seed 0 \
  --device cpu \
  --model-name phase0_smoke
```

### 13.5 Smokeモデル評価

```bash
.venv/bin/python evaluate.py \
  --model models/phase0_smoke.zip \
  --episodes 1 \
  --output-prefix phase0_smoke
```

評価では全episodeのtop-down GIF、MP4、フレーム別PNGを既定で `outputs/official/evaluation/phase0_smoke/episodes/` 以下へepisode別に保存します。記録が不要な場合は `--no-record-gif` を指定します。同じprofileとoutput prefixで再実行すると、古い `episodes/` を評価開始時に削除して対応する成果物を置き換えます。`--no-record-gif` での再実行時も古い可視化は削除されるため、結果を残す場合は別のoutput prefixを指定してください。

### 13.6 公式設定の学習

```bash
.venv/bin/python train.py \
  --timesteps 300000 \
  --num-envs 4 \
  --n-steps 4096 \
  --seed 0 \
  --device cpu \
  --model-name phase0_official
```

### 13.7 モデル評価（既定でGIF/MP4/PNGを保存）

```bash
.venv/bin/python evaluate.py \
  --model models/phase0_official.zip \
  --episodes 1 \
  --output-prefix phase0_official
```

`evaluate.py` は既定で全評価episodeを、合成済みtelemetry panel付きのtop-down GIF、MP4、フレーム別PNGとしてepisode別に保存します。`--record-gif` は後方互換の記録スイッチ名で、`--record-gif` / `--no-record-gif` により全episodeの3種類すべてを切り替えます。GIF・MP4・PNGが不要な場合は、実行時に `--no-record-gif` を指定します。

評価のCLI、モデル読込、`model.predict(..., deterministic=True)` と `env.step()` の順序は `evaluate.py` が担当し、JSON/JSONLの直列化とepisode別artifactの保持・保存は `evaluation_results.py` が担当します。この分割は評価経路と出力形式を変えず、ファイルI/Oの責務だけを推論loopから分離しています。

```bash
.venv/bin/python evaluate.py \
  --model models/phase0_official.zip \
  --episodes 1 \
  --no-record-gif \
  --output-prefix phase0_official
```

### 13.8 CUDAを使う場合

```bash
.venv/bin/python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`torch.cuda.is_available()` が `True` の環境だけで `--device cuda` を指定します。PPO + `MlpPolicy` はCPUの方が効率的なことが多いため、既定値は `cpu` です。

### 13.9 TensorBoard

```bash
.venv/bin/python -m tensorboard.main --logdir logs/tensorboard
```

## 14. 評価とGIF/MP4 API

評価は単一環境で、保存済みPPOを `deterministic=True` で呼び出します。各エピソードは `terminated or truncated` で終了し、total reward、length、速度要約、Action切替回数・頻度、元の終了flag、`route_completion`、model SHA-256、実行時間を `outputs/<profile>/evaluation/<output-prefix>/evaluation.json` へ保存します。さらに全episodeの全stepを同じrunディレクトリの `evaluation_steps.jsonl` へ1行ずつ保存し、評価JSONの `step_telemetry` からpath・schema・row数を参照できます。各行はAction適用後の状態で、速度、Action、simulation時刻、現在区間のruntime道路値、Reward、終了flagを含みます。

全評価episodeは既定でGIF、MP4、フレーム別PNGへ記録され、`episodes/episode_<番号>_scenario_<seed>/` ごとに分離されます。`--no-record-gif` 指定時だけ全episodeの記録を行いません。3種類の出力は同じpost-stepの合成済みpanel frameを使います。GIFは元の600×600地図を隠さず、右側へ320pxのtelemetry panelを追加します。panelには、実行時configから導出した物理更新Hz・制御Hz、適用済みAction、速度、累積切替回数と `switches/s`、直近2秒のAction履歴、Reward・route進捗・終了状態を表示します。`switches/s` の分母は再生時間やwall-clockではなく、episode開始から現在frameまでのsimulation経過秒です。道路値は固定設定ではなく、そのframeで車両がいる区間から取得し、`CURRENT SEGMENT (runtime)` と明記します。取得できない道路値は評価を止めず `N/A` とします。`info` のoptional項目は `info.get(...)` で取得し、存在しないキーを成功・失敗どちらにも推測しません。

このcommitで確認できるtop-down記録APIは次の形です。

```python
env.render(
    mode="topdown",
    window=False,
    screen_record=True,
)
env.top_down_renderer.generate_gif(
    gif_name=(
        "outputs/official/evaluation/phase0_official/episodes/"
        "episode_0001_scenario_000005/evaluation.gif"
    ),
    duration=100,  # runtime action_dt=0.10 s, in milliseconds
)
# MP4 is written from the same panel frames with OpenCV VideoWriter
# (fps=control_hz, codec=mp4v).
```

`window=False` はoff-screen描画、`screen_record=True` は各render frameの保存を意味します。`TopDownRenderer.generate_gif` のsignatureは `(gif_name="demo.gif", duration=30)` で、`duration` の単位はmsです（実行時action dtに合わせて評価側から指定します）。Pillow/GIFのduration単位は10 msなので、評価側はruntimeのaction dtが正確に表現できる場合だけ記録します。現在のruntime既定値（物理step 0.02秒、`decision_repeat=5`）ではpost-step frameの1枚を100 msで表示し、MP4は `control_hz=10` fpsで書き出します。したがってN枚の再生時間はどちらも `N × 0.10` 秒で、frame kのpanel時刻 `t=k×0.10s` と一致します。初期reset時刻`t=0`のframeは追加せず、各frameはAction適用区間の終了状態です。MP4はOpenCV `VideoWriter` の `mp4v` codecを使い、temporary fileから最終ファイルへ確定します。

MetaDrive 0.4.3の公開 `screen_frames` propertyはdeep copyを返すため、右panel付きframeへの置換は `evaluation_visualization.py` の単一補助関数だけでrenderer内部listへ触れます。upstreamの `metadrive/` source自体は変更しません。

### 14.1 評価テレメトリExcelガイド

`EVALUATION_TELEMETRY_GUIDE.xlsx` は、現行パネル画像、`項目説明`、`STATUS・色`、メディア時間の説明を上から順にまとめた1シート（`テレメトリ説明`）の編集可能な資料です。現行コードで右パネルを再合成した画像を埋め込み、説明・例示値・凡例は通常セルとして直接編集できます。再生成するときは次を実行します。

```bash
.venv/bin/python generate_evaluation_telemetry_guide.py
```

入力元や出力先を変える場合は `--frame`、`--steps`、`--evaluation`、`--output` を指定できます。Excelファイルにはマクロ、外部ブック参照、外部リンクを含めません。

## 15. 主な成果物

`outputs/` は設定を選ぶprofileごとに分離します。`official` は `configs/phase0_config.py`、`generalization` は `configs/generalization_config.py` に対応し、その下を学習runと評価runに分けます。既存のroot直下の成果物は自動移動せず、新しい実行から次の構成を使います。

| 種類 | 保存先 |
| --- | --- |
| 学習model | `models/<model-name>.zip` |
| rank別Monitor | `logs/monitor/env_0.monitor.csv` ～ `env_3.monitor.csv` |
| TensorBoard | `logs/tensorboard/` |
| 学習・評価log | `logs/` |
| training metadata | `outputs/<profile>/training/<model-name>/training_metadata.json` |
| 評価JSON | `outputs/<profile>/evaluation/<output-prefix>/evaluation.json` |
| 全step telemetry（JSONL） | `outputs/<profile>/evaluation/<output-prefix>/evaluation_steps.jsonl` |
| episode別top-down GIF（既定で全episodeを生成） | `outputs/<profile>/evaluation/<output-prefix>/episodes/episode_<番号>_scenario_<seed>/evaluation.gif` |
| episode別top-down MP4（既定で全episodeを生成） | `outputs/<profile>/evaluation/<output-prefix>/episodes/episode_<番号>_scenario_<seed>/evaluation.mp4` |
| episode別の合成済みtop-down PNG連番 | `outputs/<profile>/evaluation/<output-prefix>/episodes/episode_<番号>_scenario_<seed>/frames/frame_*.png` |
| 評価テレメトリ表示ガイド（編集可能Excel） | `EVALUATION_TELEMETRY_GUIDE.xlsx` |
| 環境検査log | `outputs/inspect_env.log` |
| 直接依存 / dependency lock | `requirements.txt` / `requirements.lock.txt` |

## 16. 既知の制約

公式11設定のraw `MetaDriveEnv`へSB3 2.9.0の`check_env`を直接適用すると、最新`main`でもfatalになります。SB3が送る`reset(seed=0)`をMetaDriveがscenario indexとして解釈し、固定scenario範囲`[5, 6)`から外れるためです。検査専用seed adapter、50step random走行、対象pytest 30件はpassしていますが、adapter成功をraw直接検査成功とは扱いません。公式config、upstream source、SB3 checkerは改変していません。

upstreamの `metadrive.examples.profile_metadrive` は10,000 stepの最終統計まで出力した後、終了コード139になりました。確認したscriptは生成したenvを明示closeしません。同じruntimeでenvを `finally` からcloseする1,000-step走行、本projectのtest、`inspect_env.py` は終了コード0なので、install不能とは判定しませんが、upstream profileのプロセス全体もpassとは判定しません。

現在のfreezeにもMetaDrive由来の`pygame==2.6.1`とSB3 extra由来の`pygame-ce==2.5.8`が共存し、module namespaceを共有します。今回実際にimportされた実装はpygame 2.6.1で、`pip check`もpassしましたが、将来の再解決ではinstall順に依存する潜在的なnamespace競合としてimport実体を確認してください。またPyTorchはCUDA buildでも、現在の実行コンテキストではGPU accessがblockされているためCUDA学習は未検証です。

## 17. 複数scenarioへ一般化するprofile

`configs/generalization_config.py` は、固定されたCurve 1本ではなく、scenario seedごとに生成される3-block道路を学習対象にします。MetaDrive同梱のgeneralization例に合わせて学習用と評価用のseed集合を分離し、評価時はランダム抽選せず先頭から一度ずつ走査します。

| 項目 | 学習 | 未見評価 |
| --- | --- | --- |
| scenario seed | `[1000, 2000)` の1,000個 | `[0, 200)` の200個 |
| map | seed依存のprocedural 3-block | 学習と同じ生成規則 |
| lane幅・lane数・ego車種 | ランダム化 | ランダム化 |
| spawn lane | ランダム化 | ランダム化 |
| traffic density | `0.1` | `0.1` |
| traffic再抽選 | なし。同じscenarioを再現可能 | なし |
| accident probability | `0.0` | `0.0` |
| Observation / Action / Reward | MetaDrive標準 / `Discrete(9)` / MetaDrive標準 | 学習と同じ |
| horizon | 1,000 | 1,000 |

`store_map=False` は1,000 mapを各workerへ蓄積し続けないためのメモリ優先設定です。その代わり、再訪したmapも生成し直すため学習速度とはトレードオフになります。初期の学習予算は1,000,000 timestepsです。MetaDrive同梱のgeneralization benchmarkは10,000,000 timesteps相当なので、1Mで十分な性能を保証するものではありません。まずSmoke Testと未見評価を行い、必要なら `--timesteps 10000000` へ伸ばします。

`random_agent_model=True` により、確認した実環境の標準state ObservationはPhase 0の259次元から261次元になります。学習用と未見評価用は同じ261次元ですが、既存の `phase0_official.zip` とはspace互換性がありません。`generalization` profileでは `models/generalization.zip` を新規学習し、profileを混ぜないでください。評価時にもSB3のspace互換性検査を行います。

短い配線確認は次で実行します。

```bash
.venv/bin/python train.py \
  --profile generalization \
  --timesteps 2000 \
  --num-envs 1 \
  --n-steps 256 \
  --model-name generalization_smoke
```

本学習はprofileの既定値をそのまま使えます。

```bash
.venv/bin/python train.py --profile generalization
```

これにより、`models/generalization.zip`、`outputs/generalization/training/generalization/training_metadata.json`、Monitor、TensorBoard、console logを保存します。MetaDrive 0.4.3では `reset(seed=...)` がRL乱数seedではなくscenario番号なので、PPOのseed引数へscenario seedを流用していません。学習時のscenario順序はMetaDriveが設定範囲内から選び、RL seedとは別に管理されるため、RL seedだけでは同じscenario順序を再現できません。この制約はtraining metadataにも明記します。一方、評価時は下記のとおりscenario番号を明示して比較可能にします。

未見200 scenarioの決定論的評価は次で実行します。

```bash
.venv/bin/python evaluate.py --profile generalization
```

既定ではscenario 0から199を各1回評価し、各episodeの `scenario_seed`、reward、終了理由と、全体のsuccess rate / out-of-road rateをJSONへ保存するとともに、200 episodeすべてをepisode別のGIF/MP4/PNGへ記録します。短く確認するときは `--episodes 5` のように指定すると、scenario 0から4だけを重複なしで評価します。全件可視化はディスク使用量と実行時間が大きくなるため、数値評価だけが必要な場合は `--no-record-gif` を併用してください。既存のコマンドは `--profile official` が既定なので、Phase 0公式設定のtask自体は変わりません。
